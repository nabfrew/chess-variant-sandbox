"""Alpha-beta search with iterative deepening under a node budget.

Iterative deepening is not an optimisation here, it is what makes a *node* budget
usable at all. A fixed-depth search cannot honour "you may look at 20,000 positions"
-- it either finishes early or blows through. Deepening one ply at a time means the
search always has a completed, usable answer in hand when the budget runs out, so two
agents given different budgets are genuinely the same player thinking for different
lengths of time. The strategic-depth metric compares exactly those pairs.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from chess_variant_sandbox.agents.base import (
    NativeSpec,
    SearchStats,
    refuse_unknown_options,
    register,
    register_native,
)
from chess_variant_sandbox.agents.evaluation import (
    GOAL_PIECES,
    IMMUNE_PIECES,
    WIN_SCORE,
    Evaluator,
    _mean_non_royal,
    inexhaustible_types,
)
from chess_variant_sandbox.agents.matrix import (
    RoundSearch,
    matrix_round,
    pure,
    refuse_alternating_objective,
)
from chess_variant_sandbox.agents.nets import require_net
from chess_variant_sandbox.agents.objective import objective_for
from chess_variant_sandbox.agents.objective_options import objective_values as _values
from chess_variant_sandbox.agents.objective_options import objective_weights as _objective
from chess_variant_sandbox.engine import Engine, GameState, Outcome
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import (
    DUCK_STEP,
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_DUCK,
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_PUSH,
    MOVE_ROTATE,
    MOVE_SWEEP,
    decode_move,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import NO_SQUARE, SpyPhase

INFINITY = float("inf")
MAX_DEPTH = 64

# How deep a quiescence search may go past the horizon before it settles for the standing
# evaluation. Bounded because a game where every move is a capture -- antichess, under
# ``must_capture`` -- would otherwise let it run to the node budget on every leaf, which
# is a full-width search wearing a different name.
QUIESCENCE_DEPTH = 4

# How many plies from the root the killer table has room for. ``MAX_DEPTH`` is 64 and a
# quiescence search runs past it, so this is the sum with a little room.
MAX_PLY = 96

# What a stored value means. A search inside a window only ever proves a *bound* for
# every move except the best, so a table that records the number without recording which
# kind of number it is hands back a bound as though it were a value.
EXACT, LOWER, UPPER = 0, 1, 2
# The core's table key is a u64 and wraps; Python's integers do not, so the mixing is
# masked back down. Without it the two searches would key the same position differently
# and collide in different places.
_U64 = (1 << 64) - 1

# Q502i: the generic search repairs (`docs/reference/general-player.md` §2 F), one bit
# each, the mirror of `agents.rs`. Each is spelled by its own option, ``keep=1``,
# ``threats=1``, ``loud_drops=1``; an unspelled one takes its default below, and a default
# is sent explicitly so the core holds none of its own.
#
# * ``keep``: the transposition table survives into this player's next search; entries
#   older than the previous search are dropped.
# * ``threats``: quiescence also chases a quiet move after which the engine's terminal
#   check says the mover has won, or, on the first quiescence ply only, one that leaves
#   the other side's royal attacked.
# * ``loud_drops``: on the first quiescence ply a drop is chased as a capture is.
# * ``duck_quiesce`` (WK1, `docs/reference/wakwak.md` §3 A): quiescence does not stand pat
#   where the side to move owes its duck half; it plays the ducks that would block an
#   opponent's attack, and one that does nothing, into the opponent's half at the same depth.
# * ``duck_history`` (WK1, §3 B): a duck move's history is filed by mover and target, apart
#   from the piece moves'. Both are inert without a duck.
# * ``duck_refute`` (WK2, §3 C, §7): at a duck half of the main search, once a duck is
#   refuted, the remaining ducks onto the refuting reply's target or attack path go first.
# * ``duck_front`` (WK2): at a duck half of the main search, the ducks onto
#   ``_duck_blocks``' squares go first. Both order only, and are inert without a duck.
# * ``duck_mobility`` (WK3, §3 E, §8): a leaf on an owed duck half hands the objective the
#   move count of its turn's start (the piece half one ply up), not the duck's squares, so
#   mobility means one thing at either depth parity. Inert without a duck.
# * ``tt_move`` (WK4, §3 D, §9): every node of the main search files its best move beside its
#   value (for an upper bound, the move the entry already held), and a node that finds an
#   entry it cannot answer from tries that move first. Ordering only; every game.
#   Off: it lost its duel (0.491 +/- 0.004 over twelve cells, `wakwak.md` §9.3).
KEEP_TABLE, THREATS, LOUD_DROPS, DUCK_QUIESCE, DUCK_HISTORY = 1, 2, 4, 8, 16
DUCK_REFUTE, DUCK_FRONT, DUCK_MOBILITY, TT_MOVE = 32, 64, 128, 256
# * ``mopup`` (N4, `docs/reference/no-forced-mate.md` §10): where one side is bare and the other
#   is not, a stand pat adds, for the armed side, ``MOPUP_PIECES`` mean pieces times
#   how confined the bare side is (its box, ``MOPUP_AREA``, and its legal moves, ``MOPUP_SPAN``,
#   half each). On by default since its duel (§10.4). Built-in objective only;
#   moves only positions where a side is bare.
MOPUP = 512
# N4 (`docs/reference/no-forced-mate.md` §10): the mop-up term's weight in mean non-royal pieces,
# and the legal-move count and box size at and above which a bare side reads as unconfined.
# Here, not in `agents/evaluation.py`, because only a minimax player reads them and that file is
# the analysis probe's ruler (in the `analysis` and `metrics` fingerprints). ``rust/src/eval.rs``
# holds the core's copies, exported as ``_core.MOPUP_PIECES``, ``_core.MOPUP_SPAN`` and
# ``_core.MOPUP_AREA`` and held to these by ``tests/test_mopup.py``.
MOPUP_PIECES = 1.0
MOPUP_SPAN = 8.0
MOPUP_AREA = 64.0


def mopup_weight(spec: Any, evaluator: object) -> float:
    """N4: the mop-up term's weight in the units of ``evaluator``, or zero where it is undefined:
    an objective other than the built-in one (a net or a heuristic vector), a game without a
    royal, a negative material sign, or a duck, round, spy or long-turn game.
    ``rust/src/agents.rs::mopup_weight_for`` is the same rule (the search bit is the caller's)."""
    if not isinstance(evaluator, Evaluator):
        return 0.0
    if (
        not spec.royal_types
        or spec.duck is not None
        or spec.simultaneous is not None
        or spec.spy is not None
        or spec.turn is not None
        or evaluator.material_sign <= 0.0
    ):
        return 0.0
    return MOPUP_PIECES * _mean_non_royal(
        evaluator.values, spec.royal_types, inexhaustible_types(spec)
    )


def is_bare(spec: Any, state: GameState, player: int) -> bool:
    """N4: ``player`` has royals on the board and nothing else, none in hand
    (``Game::is_bare``)."""
    royal = set(spec.royal_types)
    mine = [spec.type_of(v) for v in state.board if v and spec.owner_of(v) == player]
    hand = state.hands[player] if state.hands else ()
    return bool(mine) and all(t in royal for t in mine) and all(n == 0 for n in hand)


def is_armed(spec: Any, state: GameState, player: int) -> bool:
    """N4: ``player`` holds anything but royals, on the board or in hand (``Game::is_armed``)."""
    royal = set(spec.royal_types)
    hand = state.hands[player] if state.hands else ()
    return any(
        v and spec.owner_of(v) == player and spec.type_of(v) not in royal for v in state.board
    ) or any(n > 0 for n in hand)


# A table entry that keeps no best move (`agents.rs`'s ``NO_MOVE``).
NO_MOVE = -1
REPAIR_OPTIONS = {
    "keep": KEEP_TABLE,
    "threats": THREATS,
    "loud_drops": LOUD_DROPS,
    "duck_quiesce": DUCK_QUIESCE,
    "duck_history": DUCK_HISTORY,
    "duck_refute": DUCK_REFUTE,
    "duck_front": DUCK_FRONT,
    "duck_mobility": DUCK_MOBILITY,
    "tt_move": TT_MOVE,
    "mopup": MOPUP,
}
# Which repairs a player that spells none of them searches with: the ones that won their
# pre-registered duel (general-player.md §4.3 E9, graded in §6.7). ``threats`` won and is
# on; ``keep`` and ``loud_drops`` lost and are off, spelled only by their options. Moving
# this is a new player at every budget and owes the library a replay; held to
# ``_core.DEFAULT_REPAIRS`` by a test, so it moves the build fingerprint too. Today's
# player before Q502i is ``threats=0``. WK2's ``duck_refute`` and ``duck_front`` won together
# (`docs/reference/wakwak.md` §7.3) and are on; inert without a duck, so they move only duck
# games. The duck player before WK2 is ``duck_refute=0,duck_front=0``. WK3's ``duck_mobility``
# won its duel (§8.3) and is on, inert without a duck; the duck player before WK3 is
# ``duck_mobility=0``, and before WK2 ``duck_refute=0,duck_front=0,duck_mobility=0``. N4's
# ``mopup`` won its duel (`no-forced-mate.md` §10.4) and is on, inert where nobody is bare; the
# player before N4 is ``mopup=0``.
DEFAULT_REPAIRS = THREATS | DUCK_REFUTE | DUCK_FRONT | DUCK_MOBILITY | MOPUP

# Every option a minimax player reads; any other is refused (`refuse_unknown_options`).
OPTIONS = (
    "nodes", "mobility", "max_depth", "goal", "immune", "h", "values", "net",
    *REPAIR_OPTIONS,
)


def repairs_of(options: dict[str, Any]) -> int:
    """The repair bits these options spell, each unspelled one at its default."""
    repairs = 0
    for option, bit in REPAIR_OPTIONS.items():
        spelled = options.get(option)
        on = bool(DEFAULT_REPAIRS & bit) if spelled is None else _flag(option, spelled)
        repairs |= bit if on else 0
    return repairs


def _flag(option: str, value: Any) -> bool:
    if value in (0, 1, False, True):
        return bool(value)
    raise ValueError(f"{option}={value!r}: a search repair is spelled 0 or 1")


class BudgetExhausted(Exception):
    """Raised mid-search to abandon an incomplete iteration."""


class MinimaxAgent:
    """Negamax with alpha-beta pruning, a transposition table, and capture ordering."""

    # Schema 27 (Q935b): its round arm plays a spy game's blind rounds only; a game played in
    # rounds from the start refuses it (`tournament/play.py::_play_rounds_python`).
    plays_rounds = False

    def __init__(
        self,
        engine: Engine,
        nodes: int = 20_000,
        mobility: float = 0.05,
        max_depth: int = MAX_DEPTH,
        goal: float = GOAL_PIECES,
        weights: tuple[float, ...] = (),
        values: tuple[float, ...] = (),
        immune: float = IMMUNE_PIECES,
        net: bytes = b"",
        repairs: int = DEFAULT_REPAIRS,
    ) -> None:
        self.engine = engine
        self.evaluator = objective_for(engine, mobility, goal, weights, values, immune, net)
        self._squares = engine.spec.n_squares
        self._killers: list[list[int]] = [[-1, -1] for _ in range(MAX_PLY)]
        self._history: dict[int, int] = {}
        # WK1 (``duck_history``): ``side * squares + target`` for a duck move.
        self._duck_history: dict[int, int] = {}
        # WK2 (``duck_refute``): per ply, the move that cut off at the node last searched
        # there (-1: none), cleared on entering a node, as `agents.rs`'s ``refuters``.
        self._refuters = [-1] * MAX_PLY
        # WK3 (``duck_mobility``): per ply, the legal count of the piece half last entered
        # there, as `agents.rs`'s ``turn_moves``; written only under the bit.
        self._turn_moves = [0] * MAX_PLY
        # A diagnostic, never read by play: when a list, the index of the cutoff move at
        # every duck half of the main search that cut off (`wakwak.md` §7).
        self.duck_cutoffs: list[int] | None = None
        self.budget = max(1, int(nodes))
        self.max_depth = max_depth
        self.name = f"minimax:nodes={self.budget}"
        self.stats = SearchStats()
        self._nodes = 0
        self._table: dict[int, tuple[int, float, int, int, int]] = {}
        self.repairs = repairs
        # N4 (``mopup``): the term's weight, zero where the bit is off or it is undefined.
        self._mopup = mopup_weight(engine.spec, self.evaluator) if repairs & MOPUP else 0.0
        self._mopup_cache: dict[int, int] = {}
        # Which search this is, for ``keep``'s aging; the core's is a wrapping u16 and
        # only equality with this search and the previous one is ever asked.
        self._generation = 0
        # Schema 27 (Q935b): a spy game's tree crosses into rounds (`docs/plan-spy.md` §3.3);
        # its blind rounds are played by the matrix search, built when first asked.
        self._spy = engine.spy
        self._rounds: RoundSearch | None = None
        self.strategy: list[float] = []

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        if len(moves) == 1:
            return moves[0]

        self._begin_search()

        # Shuffle once so that equal-scoring moves are not resolved by generation order.
        # Otherwise every game between two minimax agents of the same budget is
        # identical, and the repetitiveness metric ends up measuring the search.
        ordered = list(moves)
        rng.shuffle(ordered)
        best_move = ordered[0]

        reached = 0
        for depth in range(1, self.max_depth + 1):
            try:
                candidate = self._search_root(state, ordered, depth)
            except BudgetExhausted:
                break
            best_move = candidate
            reached = depth
            # Put the previous best first: alpha-beta prunes far harder with a good
            # first guess, so the next iteration is cheaper than this one was.
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if self._nodes >= self.budget:
                break

        self._end_search(reached)
        return best_move

    def _begin_search(self) -> None:
        self._nodes = 0
        self._generation = (self._generation + 1) & 0xFFFF
        if self.repairs & KEEP_TABLE:
            now, previous = self._generation, (self._generation - 1) & 0xFFFF
            self._table = {
                key: entry for key, entry in self._table.items() if entry[3] in (now, previous)
            }
        else:
            self._table.clear()
        for slot in self._killers:
            slot[0] = slot[1] = -1
        self._history.clear()
        self._duck_history.clear()
        self.stats.moves += 1

    def _end_search(self, reached: int) -> None:
        self.stats.nodes += self._nodes
        self.stats.depth_sum += reached
        self.stats.searches += 1

    # ------------------------------------------------------------ a spy game's rounds
    #
    # Schema 27 (Q935b, `docs/plan-spy.md` §3): the oracle for `rust/src/spy_players.rs`.
    # An announced round is two plies of the tree, the announcer's writing nothing to the
    # board and the answerer's resolving the round; neither mixes and neither draws.

    def select_announcement(
        self, state: GameState, seat: int, lists: tuple[list[int], list[int]], rng: random.Random
    ) -> int:
        """The announcer's move: the maximin by iterative deepening, pure (§3.2)."""
        mine = lists[seat]
        if len(mine) == 1:
            self.strategy = [1.0]
            return mine[0]
        self._begin_search()
        ordered, answers = list(mine), lists[1 - seat]
        best_move, reached = ordered[0], 0
        for depth in range(1, self.max_depth + 1):
            try:
                best_move = self._announce_root(state, seat, ordered, answers, depth)
            except BudgetExhausted:
                break
            reached = depth
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if self._nodes >= self.budget:
                break
        self._end_search(reached)
        self.strategy = pure(mine, best_move)
        return best_move

    def select_answer(
        self,
        state: GameState,
        seat: int,
        lists: tuple[list[int], list[int]],
        announced: int,
        rng: random.Random,
    ) -> int:
        """The answerer's move: the announced row searched alone, pure (§3.2)."""
        mine = lists[seat]
        if len(mine) == 1:
            self.strategy = [1.0]
            return mine[0]
        self._begin_search()
        ordered = list(mine)
        best_move, reached = ordered[0], 0
        for depth in range(1, self.max_depth + 1):
            try:
                best_move = self._answer_root(state, 1 - seat, announced, ordered, depth)
            except BudgetExhausted:
                break
            reached = depth
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if self._nodes >= self.budget:
                break
        self._end_search(reached)
        self.strategy = pure(mine, best_move)
        return best_move

    def select_round_move(
        self,
        state: GameState,
        seat: int,
        mine: list[int],
        theirs: list[int],
        rng: random.Random,
    ) -> int:
        """A spy game's blind round, played as `matrix:nodes=N` plays it (§3.3)."""
        if not self._spy:
            raise ValueError(
                "a simultaneous game (rules.simultaneous) is played by random, greedy and "
                "matrix:nodes=N; minimax and mcts search one mover at a time"
            )
        if self._rounds is None:
            assert isinstance(self.evaluator, Evaluator), "a round plays the built-in objective"
            self._rounds = RoundSearch(self.engine, self.evaluator)
        move, self.strategy, _ = matrix_round(
            self._rounds, self.stats, state, seat, mine, theirs, rng, self.budget, self.max_depth
        )
        return move

    def _announce_root(
        self, state: GameState, announcer: int, moves: list[int], answers: list[int], depth: int
    ) -> int:
        alpha = -INFINITY
        best_move = moves[0]
        for move in moves:
            value = -self._answer_row(state, announcer, move, answers, depth, 1, -INFINITY, -alpha)
            if value > alpha:
                alpha, best_move = value, move
        return best_move

    def _answer_root(
        self, state: GameState, announcer: int, announced: int, moves: list[int], depth: int
    ) -> int:
        alpha = -INFINITY
        best_move = moves[0]
        for move in moves:
            child = self.engine.apply_round(state, *_pair(announcer, announced, move))
            score = self._negamax(child, depth - 1, 1, alpha, INFINITY)
            if score > alpha:
                alpha, best_move = score, move
        return best_move

    def _round_leaf(self, state: GameState) -> float:
        outcome = self.engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        return self._round_leaf_lists(state)

    def _round_leaf_lists(self, state: GameState) -> float:
        engine = self.engine
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        return self._evaluate_round(state, lists)

    def _evaluate_round(self, state: GameState, lists: tuple[list[int], list[int]]) -> float:
        assert isinstance(self.evaluator, Evaluator), "a round plays the built-in objective"
        counts = (len(lists[0]), len(lists[1]))
        return self.evaluator.evaluate_round(self.engine, state, state.side_to_move, counts, lists)

    def _round_node(
        self, state: GameState, phase: SpyPhase, depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        """A round inside the tree: a blind one a leaf, an announced one two plies."""
        engine = self.engine
        outcome = engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        if phase.name != "announced":
            return self._round_leaf_lists(state)
        announcer = phase.announcer

        key = self._table_key(state)
        cached = self._table.get(key)
        if cached is not None and cached[0] >= depth:
            _, value, bound, _, _ = cached
            if (
                bound == EXACT
                or (bound == LOWER and value >= beta)
                or (bound == UPPER and value <= alpha)
            ):
                return value

        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        if depth <= 0:
            return self._evaluate_round(state, lists)
        announced = list(lists[announcer])
        answers = lists[1 - announcer]
        self._order(state, announced, ply)

        opened_at = alpha
        announcer_beta = -alpha
        announcer_alpha = -beta
        announcer_best = -INFINITY
        for move in announced:
            quiet = not self._is_loud(state, move)
            score = -self._answer_row(
                state, announcer, move, answers, depth, ply + 1, -announcer_beta, -announcer_alpha
            )
            if score > announcer_best:
                announcer_best = score
            if announcer_best > announcer_alpha:
                announcer_alpha = announcer_best
            if announcer_alpha >= announcer_beta:
                self._remember(move, quiet, depth, ply, state.side_to_move)
                break

        best = -announcer_best
        if best <= opened_at:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        self._table[key] = (depth, best, bound, self._generation, NO_MOVE)
        return best

    def _answer_row(
        self,
        state: GameState,
        announcer: int,
        announced: int,
        answers: list[int],
        depth: int,
        ply: int,
        alpha: float,
        beta: float,
    ) -> float:
        """The answerer's ply: the best of its moves against ``announced``, ordered at
        ``ply``; the side does not change across a round, so a child's value passes."""
        moves = list(answers)
        self._order(state, moves, ply)
        best = -INFINITY
        for move in moves:
            quiet = not self._is_loud(state, move)
            child = self.engine.apply_round(state, *_pair(announcer, announced, move))
            score = self._negamax(child, depth - 1, ply + 1, alpha, beta)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._remember(move, quiet, depth, ply, state.side_to_move)
                break
        return best

    def _search_root(self, state: GameState, moves: list[int], depth: int) -> int:
        engine = self.engine
        alpha = -INFINITY
        best_move = moves[0]
        if self.repairs & DUCK_MOBILITY:
            self._note_turn(moves, 0)
        for move in moves:
            # See ``_negamax``: the negation and the window flip are only right when the
            # side actually changed, which a turn longer than one move makes conditional.
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._negamax(child, depth - 1, 1, -INFINITY, -alpha)
            else:
                score = self._negamax(child, depth - 1, 1, alpha, INFINITY)
            if score > alpha:
                alpha, best_move = score, move
        return best_move

    def _negamax(
        self, state: GameState, depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        self._nodes += 1
        if self._nodes >= self.budget:
            raise BudgetExhausted
        self._refuters[min(ply, MAX_PLY - 1)] = -1

        engine = self.engine
        if self._spy:
            phase = engine.phase(state)
            if phase.is_round:
                return self._round_node(state, phase, depth, ply, alpha, beta)
        # Asked before the table, not after, and no longer only on the rulesets whose
        # endings the evaluator reads. A cached value is a value for a position still
        # being played; a position that has already ended has an outcome instead. Two
        # positions sharing a key can differ in ply, in the no-progress clock and in how
        # often the position has repeated -- none of which the key carries -- so this was
        # never a territorial problem. Ungating it moves fifteen committed rulesets,
        # which is why it waited for this phase and its campaign.
        outcome = engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        key = self._table_key(state)
        cached = self._table.get(key)
        hint = cached[4] if cached is not None else NO_MOVE
        if cached is not None and cached[0] >= depth:
            # A stored number is only usable as an answer when the search that produced
            # it proved the right kind of thing. This used to return every entry as
            # though it were exact, which hands back a beta cutoff's lower bound as a
            # value -- and a lower bound is exactly what a cutoff refuses to establish.
            _, value, bound, _, _ = cached
            if (
                bound == EXACT
                or (bound == LOWER and value >= beta)
                or (bound == UPPER and value <= alpha)
            ):
                return value

        moves = engine.legal_moves(state)
        outcome = engine.terminal(state, moves)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        if self.repairs & DUCK_MOBILITY:
            self._note_turn(moves, ply)

        if depth <= 0:
            # Not the standing evaluation. Iterative deepening stops on whichever depth
            # the budget allowed, which is as often odd as even, so a leaf is as likely
            # as not to be in the middle of an exchange -- and material scored there is a
            # number about a position nobody would ever stop in.
            return self._quiesce(state, QUIESCENCE_DEPTH, ply, alpha, beta)

        # WK2: a duck half under either ordering repair (``duck_front``, ``duck_refute``).
        duck_half = bool(self.repairs & (DUCK_FRONT | DUCK_REFUTE)) and (
            move_type(moves[0]) == MOVE_DUCK
        )
        first = moves[0]
        self._order(state, moves, ply)
        if duck_half and self.repairs & DUCK_FRONT:
            self._bring_forward(moves, 0, self._duck_blocks(state, first))
        # WK4 (``tt_move``): the move this position's entry kept goes first, the rest in order.
        tt_move = bool(self.repairs & TT_MOVE)
        if tt_move and hint != NO_MOVE and hint in moves:
            moves.remove(hint)
            moves.insert(0, hint)

        opened_at = alpha
        best = -INFINITY
        best_move = NO_MOVE
        for index in range(len(moves)):
            move = moves[index]
            # Negamax negates the child and flips the window because the side is assumed
            # to alternate. Once a turn can be longer than one move it does not always,
            # and when the same player moves again the child's value is already from
            # this player's point of view -- so value and window both pass through.
            #
            # Asked as "did the side change" rather than "is there a turn rule", because
            # that is the only form that cannot drift out of step with the engine: a ramp
            # means the answer differs between one ply and the next.
            quiet = not self._is_loud(state, move)
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._negamax(child, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = self._negamax(child, depth - 1, ply + 1, alpha, beta)
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._remember(move, quiet, depth, ply, state.side_to_move)
                self._refuters[min(ply, MAX_PLY - 1)] = move
                if self.duck_cutoffs is not None and move_type(move) == MOVE_DUCK:
                    self.duck_cutoffs.append(index)
                break
            # WK2 (``duck_refute``): the reply that refuted this duck, if the child found
            # one, sends the ducks that would stand in its way to the front of the rest.
            if duck_half and self.repairs & DUCK_REFUTE and index + 1 < len(moves):
                reply = self._refuters[min(ply + 1, MAX_PLY - 1)]
                if reply != -1 and move_type(reply) != MOVE_DUCK:
                    self._bring_forward(moves, index + 1, self._reply_blocks(state, reply))

        if best <= opened_at:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        if not tt_move:
            kept = NO_MOVE
        elif bound == UPPER:
            kept = hint
        else:
            kept = best_move
        self._table[key] = (depth, best, bound, self._generation, kept)
        return best

    def _quiesce(
        self, state: GameState, depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        """Search on past the horizon until nothing is hanging.

        The standing evaluation is a lower bound on what the side to move can get, since
        it can decline to do anything loud -- that is the "stand pat", and it is what
        makes this a search over a *subset* of moves rather than a broken one.
        """
        self._nodes += 1
        if self._nodes >= self.budget:
            raise BudgetExhausted
        self._refuters[min(ply, MAX_PLY - 1)] = -1

        engine = self.engine
        # Schema 27: quiescence stops at a round (a capture in a round can be dodged).
        if self._spy and engine.is_round(state):
            return self._round_leaf(state)
        outcome = engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        moves = engine.legal_moves(state)
        outcome = engine.terminal(state, moves)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        # WK1: an owed duck half is no place to stop (``duck_quiesce``).
        if self.repairs & DUCK_QUIESCE and depth > 0 and moves and move_type(moves[0]) == MOVE_DUCK:
            return self._quiesce_duck(state, moves, depth, ply, alpha, beta)
        # WK3 (``duck_mobility``): a duck half reads the turn's start's count, the ply above.
        count = len(moves)
        if self.repairs & DUCK_MOBILITY and moves:
            if move_type(moves[0]) == MOVE_DUCK:
                count = self._turn_moves[min(max(ply - 1, 0), MAX_PLY - 1)]
            else:
                self._turn_moves[min(ply, MAX_PLY - 1)] = count
        stand_pat = self.evaluator.evaluate(engine, state, state.side_to_move, count)
        if self._mopup != 0.0:
            stand_pat += self._mopup_term(state, count)
        if depth <= 0 or stand_pat >= beta:
            return stand_pat

        # Keep the loud moves, in generation order, so the two implementations settle
        # ties the same way.
        horizon = depth == QUIESCENCE_DEPTH
        loud = [
            move
            for move in moves
            if self._is_loud(state, move) or self._chases(state, move, horizon)
        ]
        self._order(state, loud, ply)

        best = stand_pat
        alpha = max(alpha, stand_pat)
        for move in loud:
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._quiesce(child, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = self._quiesce(child, depth - 1, ply + 1, alpha, beta)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._refuters[min(ply, MAX_PLY - 1)] = move
                break
        return best

    def _mopup_term(self, state: GameState, count: int) -> float:
        """N4 (``mopup``): the mop-up term at a stand pat, for the side to move: plus the
        weight's confined share where the opponent is bare and the mover armed, minus it the
        other way round, zero otherwise. The bare side's legal count is ``count`` on its own
        move, else the engine's answer for the same position with the side replaced and no en
        passant window. `agents.rs::mopup_term`."""
        spec = self.engine.spec
        side = state.side_to_move
        if is_bare(spec, state, 1 - side) and is_armed(spec, state, side):
            sign, bare = 1.0, 1 - side
        elif is_bare(spec, state, side) and is_armed(spec, state, 1 - side):
            sign, bare = -1.0, side
        else:
            return 0.0
        if bare == side:
            moves = count
        else:
            flipped = replace(state, side_to_move=bare, ep_target=NO_SQUARE, ep_victim=NO_SQUARE)
            moves = len(self.engine.legal_moves(flipped))
        area = MOPUP_AREA
        # The box is a function of the position alone, remembered by key (`agents.rs`'s cache).
        known = self._mopup_cache.get(state.key)
        if known is None:
            known = self._mopup_cache[state.key] = self._bare_reach(state, bare, int(area))
        reach = float(known)
        span = MOPUP_SPAN
        confined = 0.5 * ((1.0 - min(reach, area) / area) + (1.0 - min(float(moves), span) / span))
        return sign * (self._mopup * confined)

    def _bare_reach(self, state: GameState, bare: int, cap: int) -> int:
        """N4: how many positions ``bare`` could reach moving again and again while the other
        player stands still, at most ``cap``, each told by its key after the move: the engine's
        answer for every step, on the position with the side replaced and no en passant
        window. A set, so the walk's order does not matter. `Game::bare_reach`."""
        engine = self.engine
        seen: set[int] = set()

        def walk(position: GameState) -> None:
            # The key follows the side, so a position reached in an odd and an even number
            # of steps is told as one.
            key = position.key
            if position.side_to_move != bare:
                key ^= engine.spec.zobrist_side
            flipped = replace(
                position, side_to_move=bare, ep_target=NO_SQUARE, ep_victim=NO_SQUARE, key=key
            )
            for move in engine.legal_moves(flipped):
                if len(seen) >= cap:
                    return
                child = engine.apply(flipped, move)
                if child.key not in seen:
                    seen.add(child.key)
                    walk(child)

        walk(state)
        return len(seen)

    def _note_turn(self, moves: list[int], ply: int) -> None:
        """WK3 (``duck_mobility``): file a piece half's legal count under its ply, for the
        duck half below it; a duck half's own list is never filed. `agents.rs::note_turn`."""
        if moves and move_type(moves[0]) != MOVE_DUCK:
            self._turn_moves[min(ply, MAX_PLY - 1)] = len(moves)

    @staticmethod
    def _bring_forward(moves: list[int], start: int, squares: set[int]) -> None:
        """WK2: the moves of ``moves[start:]`` whose target is on ``squares`` first, order
        kept within each part; the mirror of `agents.rs::bring_forward`."""
        tail = moves[start:]
        moves[start:] = [m for m in tail if move_to(m) in squares] + [
            m for m in tail if move_to(m) not in squares
        ]

    def _reply_blocks(self, state: GameState, reply: int) -> set[int]:
        """WK2: where a duck would stand in the way of ``reply``, the opponent's refutation.

        The mirror of `agents.rs::reply_blocks`: its target and, for a plain move or a
        promotion, the path of every attack source of the opponent's onto that target from its
        origin by the piece standing there.
        """
        origin, target, kind, _ = decode_move(reply)
        squares = {target}
        if kind in (MOVE_NORMAL, MOVE_PROMOTION):
            spec = self.engine.spec
            piece = state.board[origin]
            for source in spec.attack_sources[1 - state.side_to_move][target]:
                start, piece_value, prefix, window = source[:4]
                if start == origin and piece_value == piece:
                    squares.update(prefix)
                    squares.update(window)
        return squares

    def _quiesce_duck(
        self, state: GameState, moves: list[int], depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        """WK1: quiescence at an owed duck half, the mirror of `agents.rs::quiesce_duck`.

        No stand pat. The ducks onto `_duck_blocks`' squares and the first other duck, in
        generation order, then ordered; each into the opponent's quiescence at the same depth.
        """
        blocks = self._duck_blocks(state, moves[0])
        kept: list[int] = []
        neutral = False
        for move in moves:
            blocking = move_to(move) in blocks
            if blocking or not neutral:
                neutral = neutral or not blocking
                kept.append(move)
        self._order(state, kept, ply)

        engine = self.engine
        best = -INFINITY
        for move in kept:
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._quiesce(child, depth, ply + 1, -beta, -alpha)
            else:
                score = self._quiesce(child, depth, ply + 1, alpha, beta)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _duck_blocks(self, state: GameState, first: int) -> set[int]:
        """WK1: where a duck would block an opponent's attack on one of the mover's pieces.

        The mirror of `agents.rs::duck_blocks`: every attack source of the opponent's onto a
        square the mover holds, its piece on its origin, with no screen and no sweep, whose
        path is empty but for the duck's own square (lifted when the duck steps off it).
        WK1b: where the duck's type sees, also every square from which its sight would bind
        an end of one of those attacks (the mover's piece attacked, or its attacker), read off
        the reverse sight table: the sources whose byte is the duck's, path clear likewise.
        """
        spec = self.engine.spec
        board = state.board
        origin, _, _, how = decode_move(first)
        lifted = origin if how == DUCK_STEP else -1
        mover = state.side_to_move
        other = 1 - mover
        sources = spec.attack_sources[other]
        sight = spec.sight
        duck = spec.duck
        duck_value = WILD * spec.side_stride + duck.piece_type + 1 if duck is not None else -1
        sees = sight is not None and duck is not None
        ends: list[tuple[int, int]] = []
        blocks: set[int] = set()
        for square, value in enumerate(board):
            if not value or spec.owner_of(value) != mover:
                continue
            attacked = False
            for source in sources[square]:
                start, piece_value, prefix, window, screens, _, _, sweep, _ = source
                if screens or sweep or board[start] != piece_value:
                    continue
                path = (*prefix, *window)
                if all(board[s] == 0 or s == lifted for s in path):
                    blocks.update(path)
                    if sees:
                        attacked = True
                        ends.append((start, other))
            if attacked:
                ends.append((square, mover))
        if sees:
            assert sight is not None
            for square, owner in ends:
                for start, piece_value, _, path in sight.sources[owner][
                    spec.type_of(board[square])
                ][square]:
                    if (
                        piece_value == duck_value
                        and (board[start] == 0 or start == lifted)
                        and all(board[s] == 0 or s == lifted for s in path)
                    ):
                        blocks.add(start)
        return blocks

    def _is_loud(self, state: GameState, move: int) -> bool:
        """Whether a move changes material.

        Read off the encoding and the board rather than off any piece's name: a move onto
        an occupied square, an en-passant capture, which lands on an empty one, and a
        promotion, which adds material without taking any.

        **Two kinds end on an occupied square and take nothing.** A rotation ends where
        it began, so its target holds the piece that is turning; a file-based castling
        can send the king to its own rook's square. A quiescence search that chased
        either would never come to rest.
        """
        _, target, kind, promoted = decode_move(move)
        if kind == MOVE_PUSH:
            # Q1074b3: a push lands on a piece and is loud only where it throws an enemy
            # off the board (`CompiledSpec.push_takes`) or the pusher promotes (P1), as
            # `agents.rs::is_loud` reads it.
            return promoted != 0 or self.engine.spec.push_takes(state.board, move)
        # Schema 22: a fold moved onto a piece takes nothing either.
        if kind not in (MOVE_ROTATE, MOVE_CASTLE, MOVE_FOLD, MOVE_DUCK) and state.board[target]:
            return True
        # Schema 19: a transit that promotes at its exit, as `agents.rs::is_loud` reads it.
        return kind in (MOVE_EN_PASSANT, MOVE_PROMOTION, MOVE_SWEEP) or (
            kind == MOVE_PORTAL and promoted != 0
        )

    def _chases(self, state: GameState, move: int, horizon: bool) -> bool:
        """Q502i: whether quiescence chases this quiet move under the player's repairs.

        The mirror of `agents.rs::chases`. ``horizon`` is the first quiescence ply.
        """
        if self.repairs & LOUD_DROPS and horizon and move_type(move) == MOVE_DROP:
            return True
        if not self.repairs & THREATS:
            return False
        engine = self.engine
        mover = state.side_to_move
        child = engine.apply(state, move)
        outcome = engine.terminal_prefix(child)
        if outcome is not None and not outcome.is_draw and outcome.winner == mover:
            return True
        return horizon and engine.in_check(child, 1 - mover)

    def _order(self, state: GameState, moves: list[int], ply: int) -> None:
        """Captures by what they take, then killers, then history.

        Captures-only ordering is close to no ordering at all in a game where captures
        are rare, and that is most of the corpus -- it is why a branching factor of 41
        bought only depth 2. Sorted on a tuple rather than a blended number: a single
        score would have to pick scale factors, and two orderings that disagree about a
        tie are two different games. Python's sort is stable and Rust's `sort_by` is too,
        so equal keys keep generation order on both sides.
        """
        board = state.board
        killers = self._killers[min(ply, MAX_PLY - 1)]
        squares = self._squares
        ducks = bool(self.repairs & DUCK_HISTORY)
        duck_slot = state.side_to_move * squares

        def rank(move: int) -> tuple[float, int, int]:
            origin, target, kind, _ = decode_move(move)
            killer = 2 if move == killers[0] else (1 if move == killers[1] else 0)
            if ducks and kind == MOVE_DUCK:
                heard = self._duck_history.get(duck_slot + target, 0)
            else:
                heard = self._history.get(origin * squares + target, 0)
            return (-self._capture_gain(board, move), -killer, -heard)

        moves.sort(key=rank)

    def _remember(self, move: int, quiet: bool, depth: int, ply: int, side: int) -> None:
        """Record that this move caused a cutoff, so the next search tries it sooner.

        ``side`` is the mover's: WK1's duck history files a duck move under it.
        """
        slot = self._killers[min(ply, MAX_PLY - 1)]
        if quiet and slot[0] != move:
            slot[1] = slot[0]
            slot[0] = move
        origin, target, kind, _ = decode_move(move)
        weight = max(depth, 0) ** 2
        if self.repairs & DUCK_HISTORY and kind == MOVE_DUCK:
            index = side * self._squares + target
            self._duck_history[index] = self._duck_history.get(index, 0) + weight
            return
        index = origin * self._squares + target
        self._history[index] = self._history.get(index, 0) + weight

    @staticmethod
    def _terminal_value(outcome: Outcome, state: GameState) -> float:
        """A finished position's value, from the side to move's point of view.

        Discounted by ply so a forced win found sooner is preferred and a loss is
        delayed. Without it the search is indifferent between mate now and mate in six,
        and games drift instead of finishing.
        """
        if outcome.is_draw:
            return 0.0
        signed = WIN_SCORE - state.ply
        return signed if outcome.winner == state.side_to_move else -signed

    def _table_key(self, state: GameState) -> int:
        """The key this search's transposition table is filed under.

        Not ``state.key``. That key deliberately omits a *permanent* colouring, and the
        argument for the omission is sound for the rule it was made for: a permanent
        claim is irreversible, so it resets the no-progress clock, so it clears the
        repetition window, so two positions the repetition rule ever compares always
        share a colouring. A search tree is not a repetition window -- two branches
        routinely reach the same placement having claimed different squares, and once
        the evaluator reads territory those are two positions at different distances
        from a win. The mixing matches ``agents.rs::table_key`` exactly, so the two
        searches collide in the same places as well as agreeing in the rest.
        """
        if self.engine.spec.territory is None:
            return state.key
        mine = state.colour.count(1) * 0x9E3779B97F4A7C15
        theirs = state.colour.count(2) * 0xC2B2AE3D27D4EB4F
        return state.key ^ (mine & _U64) ^ (theirs & _U64)

    def _capture_gain(self, board: bytes, move: int) -> float:
        """Order captures first, most valuable victim first. Cheap and generic.

        Same trap as `_is_loud`, and the mirror of `agents.rs::capture_gain`. A
        rotation's target is its own square and a file-based castling's can be its own
        rook's, so reading the victim off the target orders both ahead of every quiet
        move by the value of the mover's own piece. That is not a tie-break difference --
        on `weathervane` most of the move list is rotations, so the two searches would
        cut off in different places and play different games under one name.
        """
        if move_type(move) in (MOVE_ROTATE, MOVE_CASTLE, MOVE_FOLD, MOVE_DUCK):
            return 0.0
        spec = self.engine.spec
        values = self.evaluator.values
        victim = board[move_to(move)]
        if move_type(move) == MOVE_PUSH:
            # Q1074b3: a push gains only the enemy it throws off the board (Q1074b4: under a
            # row the line's last piece, not the target's); one that moves its piece, or
            # throws the pusher's own off, gains nothing.
            taken = spec.push_taken(board, move)
            return values[spec.type_of(taken)] if taken else 0.0
        gain = values[spec.type_of(victim)] if victim else 0.0
        # Schema 13: a sweep's path victims are summed onto the target's, in ray order, as
        # `agents.rs::capture_gain` sums them (`CompiledSpec.sweep_victims`).
        for _, swept in spec.sweep_victims(board, move):
            gain += values[spec.type_of(swept)]
        return gain


def _pair(announcer: int, announced: int, answer: int) -> tuple[int, int]:
    """A round's two moves in seat order, ``announced`` the announcer's."""
    return (announced, answer) if announcer == 0 else (answer, announced)


@register("minimax")
def _build_minimax(engine: Engine, options: dict[str, Any]) -> MinimaxAgent:
    refuse_unknown_options(options, OPTIONS, "minimax")
    if engine.spy:
        # Schema 27 (Q935b): a spy game is played on the built-in objective only, as a
        # round game is (`docs/plan-spy.md` §3.3).
        refuse_alternating_objective(options, "minimax", spy=True)
    return MinimaxAgent(
        engine,
        nodes=int(options.get("nodes", 20_000)),
        mobility=float(options.get("mobility", 0.05)),
        max_depth=int(options.get("max_depth", MAX_DEPTH)),
        goal=float(options.get("goal", GOAL_PIECES)),
        weights=_objective(options, engine.spec.hash),
        values=_values(options, engine.spec.hash),
        immune=float(options.get("immune", IMMUNE_PIECES)),
        net=require_net(options, engine.spec.hash),
        repairs=repairs_of(options),
    )


@register_native("minimax")
def _native_minimax(options: dict[str, Any], ruleset_hash: str = "") -> NativeSpec:
    refuse_unknown_options(options, OPTIONS, "minimax")
    return (
        "minimax",
        int(options.get("nodes", 20_000)),
        float(options.get("mobility", 0.05)),
        int(options.get("max_depth", MAX_DEPTH)),
        0,
        0.0,
        float(options.get("goal", GOAL_PIECES)),
        _objective(options, ruleset_hash),
        _values(options, ruleset_hash),
        float(options.get("immune", IMMUNE_PIECES)),
        require_net(options, ruleset_hash),
        repairs_of(options),
    )
