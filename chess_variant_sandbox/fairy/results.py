"""Does the translated variant end the game where ours does, and for the same side?

`verify` holds the two engines' moves together and never asks who won. This holds their
results together (Q404). At the same positions it asks both engines whether the game is
over and for whom, and compares the two answers. Ours answers through `Engine.terminal`.
Theirs answers through `pyffish` (`PyffishBackend.ending`), in the order their search
asks: an ending the variant declares, a draw a player may claim, then no legal move.
Python asks both and answers neither. The only thing decided here is which positions to
ask about.

**Which positions.** The endings that matter are the ones a short walk never reaches, so
the positions come from four places:

* **Random games played to the end in our core.** Half choose uniformly, which reaches
  the clocks and the ply limit. Half take any move that ends the game, which reaches
  mates, extinctions, flags and counted checks. Each game is compared at its last
  position, at the one before it, at every other kind of ending one move from that one,
  and at a few positions on the way.
* **The no-progress clock set one ply short** in a walked position, then one move of
  each kind (each piece's quiet move and capture, each drop, castling, promotion) played
  by their engine from there. That compares the limit and which moves reset the clock.
* **The ply counter set to the limit.** Their game has no length, so this is expected to
  differ, and it is counted as admitted rather than reported.
* **Two reversible moves played back and forth** until ours calls a repetition, with
  their engine replaying the same moves. Under ``repetition_draw: "never"`` the pair is
  played past three folds instead, and theirs must not call it either (Q703).

**Two ways of asking theirs.** From our FEN of a position with no moves, which is the
same position re-expressed. And from the FEN at the start of our repetition window with
the moves since, so that their engine reaches the position by its own moves and counts
the repetitions, the checks and the clock itself. The second is where a count that
their engine keeps differently shows up. A FEN carries no history, so the first way is
not used where ours calls a repetition.

**Two meanings are settled before comparing.** A draw theirs lets a player claim agrees
with a draw ours declares, because ours never waits for a claim. A difference the
translation already admits in `Variant.notes`, with its key in `Variant.approximates`,
is counted as admitted and does not fail the check. Each key admits one shape of
difference, the one its note describes, and nothing wider (`admits`, Q411). So a key
approximated for one reason does not hide a mistake in the same key for another.

The four sources are independent, so each runs to its first difference and the check
reports every source that found one. In the clock probe each kind of move is its own
question, and every kind that differs is reported. Like `verify`, this is agreement over the
positions it reached, not a proof. A `ResultCheck` records what it compared, what
differed as noted, and which of the result keys in play it never reached.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import NO_REPETITION_DRAW, CompiledSpec
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_EN_PASSANT, MOVE_NORMAL, decode_move
from chess_variant_sandbox.engine.state import GameState, Outcome
from chess_variant_sandbox.fairy import position as fairy_position
from chess_variant_sandbox.fairy.variant import Untranslatable, Variant, translate

# Random games played to the end, half uniform and half taking an ending when one is
# offered. Each costs about ten questions to the other engine, at about 20 ms apiece.
DEFAULT_GAMES = 6
# Positions compared on the way through each game, besides its last two.
SAMPLES = 3
# How many other kinds of ending one move from a game's last position are compared.
OTHER_ENDINGS = 3
# How many replies the search for a pair of reversible moves tries, in all.
CYCLE_TRIES = 400
# How many kinds of move the no-progress probe plays from its position. Shogi has the
# most in the corpus, about 36, and most rulesets have 10 to 20.
MOVE_KINDS = 32
# How many times the never-repetition probe has a position stand: one past the three
# folds their engine draws at by default, so a rule left on there is a difference.
NEVER_FOLDS = 4

# Which `rules` key decides each ending of ours, by the reason the core gives.
RESULT_KEYS: dict[str, str] = {
    "checkmate": "checkmate",
    "stalemate": "stalemate",
    "stalemate_loss": "stalemate",
    "stalemate_win": "stalemate",
    "ply_limit": "max_plies",
    "no_progress": "no_progress_draw",
    "repetition": "repetition_draw",
    "extinction": "extinction",
    "reached": "reach",
    "check_limit": "check_limit",
    "royal_captured": "win",
    "annihilated": "win",
    "royal_destroyed": "explode_on_capture",
}

_PLAYERS = ("first", "second")


class Referee(Protocol):
    """The two questions the result check asks the other engine."""

    def load(self, variant: Variant, config_path: str) -> None:
        ...

    def ending(self, fen: str, moves: Sequence[str]) -> tuple[str, int] | None:
        ...

    def fen_after(self, fen: str, moves: Sequence[str]) -> str:
        ...


@dataclass(frozen=True)
class Ending:
    """One engine's answer to *is the game over here, and for whom*."""

    over: bool
    # 0 or 1 for the player who won, ``None`` for a draw or a game still going on.
    winner: int | None = None
    # Ours: the core's reason. Theirs: which of their questions said so.
    how: str = ""

    @classmethod
    def ours(cls, outcome: Outcome | None) -> Ending:
        if outcome is None:
            return cls(over=False)
        return cls(over=True, winner=outcome.winner, how=outcome.reason)

    @classmethod
    def theirs(cls, answer: tuple[str, int] | None, side_to_move: int) -> Ending:
        """Their answer, with the value read from the side to move's point of view."""
        if answer is None:
            return cls(over=False)
        how, value = answer
        if value == 0:
            return cls(over=True, winner=None, how=how)
        winner = side_to_move if value > 0 else 1 - side_to_move
        return cls(over=True, winner=winner, how=how)

    def same(self, other: Ending) -> bool:
        """Whether the two say the same thing: over or not, and if over, for whom."""
        return self.over == other.over and (not self.over or self.winner == other.winner)

    def __str__(self) -> str:
        if not self.over:
            return "the game goes on"
        if self.winner is None:
            return f"drawn ({self.how})"
        return f"won by the {_PLAYERS[self.winner]} player ({self.how})"


@dataclass
class Difference:
    """One position the two engines called differently, with what each said.

    ``fen`` and ``moves`` are exactly what their engine was asked: the FEN it was handed
    and the moves it played from there itself. ``their_fen`` is where those moves took
    it, which is our position unless their engine applies a move differently.
    """

    # The `rules` key the difference is about, or "" where neither answer names one.
    key: str
    fen: str
    moves: list[str]
    ours: str
    theirs: str
    their_fen: str = ""

    def text(self) -> str:
        where = f" after {' '.join(self.moves)}" if self.moves else ""
        about = f"{self.key}: " if self.key else ""
        return f"{about}ours says {self.ours}{where}, theirs says {self.theirs}"


@dataclass
class ResultCheck:
    """What the two engines turned out to agree about when asked who won."""

    ruleset_hash: str
    name: str
    ok: bool
    # Why there was nothing to compare: a refusal, or a variant that would not load.
    reasons: list[str] = field(default_factory=list)
    # How many times the other engine was asked, and how many games were played out.
    questions: int = 0
    games: int = 0
    # Our ending reason -> how many positions both engines ended that way.
    agreed: dict[str, int] = field(default_factory=dict)
    # `rules` key -> how many positions differed in the way `Variant.notes` admits.
    admitted: dict[str, int] = field(default_factory=dict)
    # Result keys this ruleset has in play that no compared ending exercised, with why.
    unreached: dict[str, str] = field(default_factory=dict)
    # The first difference each source of positions found, in the order they ran.
    differences: list[Difference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.ruleset_hash,
            "name": self.name,
            "ok": self.ok,
            "reasons": self.reasons,
            "questions": self.questions,
            "games": self.games,
            "agreed": self.agreed,
            "admitted": self.admitted,
            "unreached": self.unreached,
            "differences": [asdict(difference) for difference in self.differences],
            "notes": self.notes,
        }

    def summary(self) -> str:
        """One line: the differences if there are any, otherwise what was compared."""
        if self.reasons:
            return "; ".join(self.reasons)
        if self.differences:
            return "; ".join(difference.text() for difference in self.differences)
        return f"agree at {self.questions} positions; {self.coverage()}"

    def coverage(self) -> str:
        """Which endings were compared, which differed as noted, and which were not met."""
        parts = []
        if self.agreed:
            parts.append(
                "ended by "
                + ", ".join(f"{reason} {count}" for reason, count in sorted(self.agreed.items()))
            )
        if self.admitted:
            parts.append(
                "differ as noted: "
                + ", ".join(f"{key} {count}" for key, count in sorted(self.admitted.items()))
            )
        if self.unreached:
            parts.append("not reached: " + ", ".join(sorted(self.unreached)))
        return "; ".join(parts) or "no ending compared"

    def reached(self) -> set[str]:
        """The result keys some compared ending exercised, agreed, admitted or not."""
        keys = {RESULT_KEYS[reason] for reason in self.agreed if reason in RESULT_KEYS}
        keys |= set(self.admitted)
        return keys | {difference.key for difference in self.differences if difference.key}


def in_play(compiled: CompiledSpec) -> list[str]:
    """The result keys that can decide a game of this ruleset.

    The clocks and stalemate can decide any game, and a repetition can decide any game
    but one under ``"never"`` (whose probe still asks theirs, Q703). ``checkmate`` needs a royal in check,
    so a rule forbidding check rules it out. ``win`` is listed only where its verdict
    is an ending of its own: under ``"checkmate"`` that ending is the ``checkmate`` key's,
    and ``"none"`` has none.
    """
    keys = ["stalemate", "max_plies", "no_progress_draw"]
    if compiled.repetition_draw != NO_REPETITION_DRAW:
        keys.append("repetition_draw")
    if compiled.royal_types and compiled.win == "checkmate" and not compiled.forbid_check:
        keys.append("checkmate")
    if compiled.win in ("royal_capture", "annihilation"):
        keys.append("win")
    if compiled.extinction is not None:
        keys.append("extinction")
    if compiled.reach is not None:
        keys.append("reach")
    if compiled.check_limit:
        keys.append("check_limit")
    if compiled.explosion is not None and compiled.royal_types:
        keys.append("explode_on_capture")
    return sorted(keys)


def verdicts(
    engine: Engine,
    variant: Variant,
    referee: Referee,
    state: GameState,
    start: GameState | None = None,
    moves: Sequence[str] = (),
) -> tuple[Ending, Ending]:
    """Both engines' answers at ``state``, ours first.

    Without ``start``, theirs is handed our FEN of ``state``. With it, theirs is handed
    our FEN of ``start`` and plays ``moves`` itself, which should reach ``state``. The
    referee must already have loaded ``variant``. Raises whatever the referee raises.
    """
    text, played = _question(engine, variant, state, start, moves)
    answer = referee.ending(text, played)
    side = fairy_position.their_side(engine.spec, state)
    return Ending.ours(engine.terminal(state)), Ending.theirs(answer, side)


# Our endings by a clock, and the `rules` key of each.
_CLOCKS = {"no_progress": "no_progress_draw", "ply_limit": "max_plies"}


def admits(
    engine: Engine,
    variant: Variant,
    state: GameState,
    ours: Ending,
    theirs: Ending,
    referee: Referee | None = None,
) -> str:
    """The key under which ``variant`` admits the two answers at ``state`` differing.

    Empty where it does not: where they agree, where ours says the game goes on, or where
    the difference is not the shape of any note the translation carries. Each key in
    `Variant.approximates` admits one shape:

    * ``max_plies`` and ``no_progress_draw``: ours ends the game by that clock, and
      theirs says what ours says with both clocks wound back. Their game has no length,
      and their no-progress clock can run longer than ours: an odd count is rounded up,
      and a drop that is not one-way resets theirs and not ours.
    * ``checkmate``: ours calls it a checkmate, theirs calls it a draw, and theirs with
      the clocks wound back calls it that mate. Ours looks for mate before the clocks
      (Q412), and theirs lets a player claim the draw first where the royal is
      pseudo-royal, since that is never in check. Their wound-back answer is theirs to
      give, so this shape needs ``referee``, and without one nothing is admitted.
      Until Q412 the shape was the other way round (ours a draw by the clock, theirs
      the mate), where the royal is in their king slot.
    * ``reach``: ours ends the game by an arrival that does not win. There arriving
      always wins.

    The engines answer the wound-back questions too, ours for a clock and theirs for a
    mate, so Python still answers neither.
    """
    if not ours.over or ours.same(theirs):
        return ""
    key = RESULT_KEYS.get(ours.how, "")
    if ours.how in _CLOCKS:
        unclocked = Ending.ours(engine.terminal(state.with_(halfmove=0, ply=0)))
        if not theirs.same(unclocked):
            return ""
        return key if key in variant.approximates else ""
    if key == "checkmate" and key in variant.approximates:
        drawn = _a_mate_their_clock_drew(engine, variant, referee, state, ours, theirs)
        return key if drawn else ""
    if key == "reach" and key in variant.approximates:
        return key
    return ""


def _a_mate_their_clock_drew(
    engine: Engine,
    variant: Variant,
    referee: Referee | None,
    state: GameState,
    ours: Ending,
    theirs: Ending,
) -> bool:
    """Whether theirs drew what ours calls a mate, and wound back calls it the same mate.

    Asked of their engine from our FEN of ``state`` with both clocks at zero. A FEN
    carries no history, and ours is a mate, which cannot be a position seen before.
    """
    if referee is None or not theirs.over or theirs.winner is not None:
        return False
    unclocked = fairy_position.variant_fen(
        variant, engine.spec, state.with_(halfmove=0, ply=0)
    )
    try:
        answer = referee.ending(unclocked, [])
    except Exception:  # noqa: BLE001 - an answer it cannot give admits nothing
        return False
    return Ending.theirs(answer, fairy_position.their_side(engine.spec, state)).same(ours)


def _question(
    engine: Engine,
    variant: Variant,
    state: GameState,
    start: GameState | None,
    moves: Sequence[str],
) -> tuple[str, list[str]]:
    """The FEN and the moves their engine is asked about ``state`` with."""
    if start is None:
        return fairy_position.variant_fen(variant, engine.spec, state), []
    return fairy_position.variant_fen(variant, engine.spec, start), list(moves)


def check_results(
    spec: dict[str, Any],
    referee: Referee,
    config_path: str = "",
    *,
    games: int = DEFAULT_GAMES,
    seed: int = 0,
) -> ResultCheck:
    """Ask both engines who won at the same positions, and report where they differ.

    ``config_path`` is passed to ``referee.load`` as `verify` passes it; `pyffish` reads
    the definition's text and ignores it.
    """
    try:
        written = translate(spec)
    except Untranslatable as error:
        return ResultCheck(
            ruleset_hash=_hash(spec),
            name=str(spec.get("name", "")),
            ok=False,
            reasons=list(error.reasons),
        )

    engine: Any = Engine(spec)
    if engine.spec.duck is not None:
        # Generation 42 (N4c, `docs/archive/plan-npc.md` §4.6): a duck game is asked about at turn
        # starts, a move being both halves there (`_Turns`).
        engine = _Turns(engine)
    check = ResultCheck(
        ruleset_hash=engine.spec.hash,
        name=engine.spec.name,
        ok=True,
        notes=list(written.notes),
    )
    try:
        referee.load(written, config_path)
    except Exception as error:  # noqa: BLE001 - any failure to load is a refusal
        check.ok = False
        check.reasons.append(f"the engine would not load the variant: {error}")
        return check

    asker = _Asker(engine, written, referee, check)
    played: list[_Line] = []
    for game in range(games):
        rng = random.Random((seed << 8) ^ game)
        seen = frozenset(_ended_by(engine, line) for line in played)
        played.append(_play_out(engine, written, rng, seek=game % 2 == 1, seen=seen))
        check.games += 1
        if not asker.game(played[-1]):
            break

    uniform = played[0] if played else _Line.opening(engine)
    base = _probe_position(engine, uniform)
    asker.no_progress(uniform, base)
    asker.ply_limit(uniform, base)
    asker.repetition(uniform, played)

    reached = check.reached()
    for key in in_play(engine.spec):
        if key not in reached:
            check.unreached.setdefault(key, "no compared position ended by it")
    check.ok = not check.differences
    return check


# ------------------------------------------------------------------------------ lines


@dataclass
class _Line:
    """A game in our core: every position, and each move spelled for the other engine."""

    states: list[GameState]
    moves: list[int]
    spelled: list[str]

    @classmethod
    def opening(cls, engine: Engine) -> _Line:
        return cls([engine.initial_state()], [], [])

    def prefix(self, index: int) -> _Line:
        """The line up to and including position ``index``."""
        return _Line(self.states[: index + 1], self.moves[:index], self.spelled[:index])

    def extended(self, engine: Engine, variant: Variant, move: int) -> _Line:
        state = self.states[-1]
        return _Line(
            [*self.states, engine.apply(state, move)],
            [*self.moves, move],
            [*self.spelled, _spell(engine, variant, state, move)],
        )

    def window(self, index: int) -> tuple[GameState, list[str]]:
        """Where our repetition window at ``index`` starts, and the moves since.

        Our window restarts at every irreversible move, so no position before it can
        repeat one after it here. Their engine is given the same stretch, so both count
        over the same history.
        """
        start = index
        while start > 0 and len(self.states[start].repetition) > 1:
            start -= 1
        return self.states[start], self.spelled[start:index]


def _spell(engine: Engine, variant: Variant, state: GameState, move: int) -> str:
    if isinstance(engine, _Turns):
        piece, duck, ended = _Turns.halves(move)
        if ended:
            # Our game ended on the piece move, with no duck half. Their word for it needs
            # a duck square, and the vacated origin is one every such move leaves empty.
            origin, target, _, _ = decode_move(piece)
            width = engine.spec.width
            head = fairy_position.uci_of(
                engine.spec, state, piece, variant.chars, variant.promotion_suffix
            )
            return f"{head},{_name(target, width)}{_name(origin, width)}"
        return fairy_position.turn_word(
            engine.spec, state, piece, duck, variant.chars, variant.promotion_suffix
        )
    return fairy_position.uci_of(
        engine.spec, state, move, variant.chars, variant.promotion_suffix
    )


def _name(square: int, width: int) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


class _Turns:
    """A duck game's engine with a turn for a move (generation 42, N4c, §4.6).

    Their move is both halves of our turn, so the result check walks our turns: each legal
    move here is one of our piece moves with one of the duck moves our engine then offers,
    packed into one integer (the piece move in the low 32 bits, so `decode_move` reads it),
    or the piece move alone where it ended our game. Every move and every ending is still
    our engine's; this only pairs them. Everything else is the engine's own.
    """

    _ENDED = 1 << 63

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    @classmethod
    def halves(cls, move: int) -> tuple[int, int, bool]:
        """The piece move, the duck move (0 where none) and whether the first ended it."""
        return move & 0xFFFFFFFF, (move >> 32) & 0x7FFFFFFF, bool(move & cls._ENDED)

    def legal_moves(self, state: GameState) -> list[int]:
        engine = self._engine
        turns: list[int] = []
        for piece in engine.legal_moves(state):
            half = engine.apply(state, piece)
            if engine.terminal(half) is not None:
                turns.append(piece | self._ENDED)
                continue
            turns.extend(piece | duck << 32 for duck in engine.legal_moves(half))
        return turns

    def apply(self, state: GameState, move: int) -> GameState:
        piece, duck, ended = self.halves(move)
        half = self._engine.apply(state, piece)
        return half if ended else self._engine.apply(half, duck)

    def terminal(self, state: GameState, legal: Any = None) -> Outcome | None:
        return self._engine.terminal(state)


def _play_out(
    engine: Engine,
    variant: Variant,
    rng: random.Random,
    *,
    seek: bool,
    seen: frozenset[str] = frozenset(),
) -> _Line:
    """One game in our core, to our ending.

    ``seek`` plays a move after which our engine calls the game over, preferring an
    ending not in ``seen`` (the ones earlier games reached), and otherwise a random
    move. Our engine decides every ending, so what this seeks is ours; theirs is only
    asked.
    """
    state = engine.initial_state()
    line = _Line([state], [], [])
    while True:
        legal = engine.legal_moves(state)
        if engine.terminal(state, legal) is not None:
            return line
        move = rng.choice(legal)
        if seek:
            order = list(legal)
            rng.shuffle(order)
            ending = None
            for candidate in order:
                outcome = engine.terminal(engine.apply(state, candidate))
                if outcome is None:
                    continue
                if ending is None:
                    ending = candidate
                if outcome.reason not in seen:
                    ending = candidate
                    break
            if ending is not None:
                move = ending
        next_state = engine.apply(state, move)
        line.states.append(next_state)
        line.moves.append(move)
        line.spelled.append(_spell(engine, variant, state, move))
        state = next_state


def _kind_of(compiled: CompiledSpec, state: GameState, move: int) -> tuple[int, int, bool]:
    """A move's kind, the piece it moves or drops, and whether it takes something."""
    origin, target, kind, promoted = decode_move(move)
    if kind == MOVE_DROP:
        return kind, promoted % compiled.n_types, False
    occupant = compiled.piece_at(state.board, origin)
    piece = occupant[1] if occupant is not None else -1
    takes = kind == MOVE_EN_PASSANT or state.board[target] != 0
    return kind, piece, takes


def _one_of_each_kind(engine: Engine, state: GameState) -> list[int]:
    """One legal move for each kind of move, piece and capture.

    Drops, promotions, castling and en passant first, then captures, then quiet moves,
    so that where there are more kinds than `MOVE_KINDS` the rarer ones are kept.
    """
    seen: dict[tuple[int, int, bool], int] = {}
    for move in engine.legal_moves(state):
        seen.setdefault(_kind_of(engine.spec, state, move), move)
    order = sorted(seen, key=lambda kind: (kind[0] == MOVE_NORMAL, not kind[2], kind))
    return [seen[kind] for kind in order][:MOVE_KINDS]


def _probe_position(engine: Engine, line: _Line) -> int:
    """The index of the walked position with the most kinds of move.

    The clock probes start there. More kinds of move is more of the clock's reset rule
    compared, and a capture among them counts double, since captures are what that rule
    is mostly about.
    """
    best, best_index = -1, 0
    for index, state in enumerate(line.states[:-1]):
        kinds = {_kind_of(engine.spec, state, move) for move in engine.legal_moves(state)}
        score = len(kinds) + (len(kinds) if any(takes for _, _, takes in kinds) else 0)
        if score > best:
            best, best_index = score, index
    return best_index


def _reversal(engine: Engine, state: GameState, move: int) -> int | None:
    """The legal move that takes ``move``'s piece straight back, if our engine has one."""
    origin, target, kind, _ = decode_move(move)
    if kind != MOVE_NORMAL:
        return None
    for candidate in engine.legal_moves(state):
        back_origin, back_target, back_kind, _ = decode_move(candidate)
        if back_kind == MOVE_NORMAL and back_origin == target and back_target == origin:
            return candidate
    return None


def _cycle(engine: Engine, state: GameState) -> list[int] | None:
    """Four moves, one there and back for each side, that our engine allows in turn.

    A first move whose piece cannot come straight back after the first reply tried is
    given up at once, which is what keeps a board full of pawns cheap to search.
    """
    budget = CYCLE_TRIES
    for first in engine.legal_moves(state):
        if decode_move(first)[2] != MOVE_NORMAL:
            continue
        one = engine.apply(state, first)
        if engine.terminal(one) is not None:
            continue
        for second in engine.legal_moves(one):
            budget -= 1
            if budget < 0:
                return None
            if decode_move(second)[2] != MOVE_NORMAL:
                continue
            two = engine.apply(one, second)
            if engine.terminal(two) is not None:
                continue
            first_back = _reversal(engine, two, first)
            if first_back is None:
                break
            three = engine.apply(two, first_back)
            if engine.terminal(three) is not None:
                continue
            second_back = _reversal(engine, three, second)
            if second_back is not None:
                return [first, second, first_back, second_back]
    return None


# ------------------------------------------------------------------------------ asking


class _Asker:
    """Asks both engines about positions and keeps the tally in a `ResultCheck`."""

    def __init__(
        self, engine: Engine, variant: Variant, referee: Referee, check: ResultCheck
    ) -> None:
        self.engine = engine
        self.variant = variant
        self.referee = referee
        self.check = check

    def ask(
        self,
        state: GameState,
        start: GameState | None = None,
        moves: Sequence[str] = (),
        *,
        about: str = "",
    ) -> bool:
        """Compare the two verdicts at ``state``. False on a difference nobody admitted.

        With ``start``, theirs is asked from ``start`` with ``moves`` played by their own
        engine, which reaches ``state``. Without it, from our FEN of ``state``.
        ``about`` names the key a difference is about when ours says the game goes on,
        which is the probe's own key; otherwise our ending names it.
        """
        ours = Ending.ours(self.engine.terminal(state))
        if start is None and ours.how == "repetition":
            return True  # a FEN carries no history, so theirs cannot see it
        text, played = _question(self.engine, self.variant, state, start, moves)
        key = RESULT_KEYS.get(ours.how, "") if ours.over else about
        self.check.questions += 1
        try:
            answer = self.referee.ending(text, played)
        except Exception as error:  # noqa: BLE001 - a crash in there is a difference
            return self._differ(key, text, played, ours, f"nothing: it failed ({error})")
        theirs = Ending.theirs(answer, fairy_position.their_side(self.engine.spec, state))
        if ours.same(theirs):
            if ours.over:
                self.check.agreed[ours.how] = self.check.agreed.get(ours.how, 0) + 1
            return True
        admitted = admits(self.engine, self.variant, state, ours, theirs, self.referee)
        if admitted:
            self.check.admitted[admitted] = self.check.admitted.get(admitted, 0) + 1
            return True
        return self._differ(key, text, played, ours, str(theirs))

    def _differ(
        self, key: str, text: str, played: list[str], ours: Ending, theirs: str
    ) -> bool:
        their_fen = ""
        if played:
            try:
                their_fen = self.referee.fen_after(text, played)
            except Exception:  # noqa: BLE001 - only a clue, and the difference stands
                their_fen = ""
        self.check.differences.append(
            Difference(key, text, played, str(ours), theirs, their_fen)
        )
        return False

    # -------------------------------------------------------------- what is asked

    def game(self, line: _Line) -> bool:
        """A game played to our ending.

        Its last position both ways, the one before it, every other kind of ending one
        move from that one, and a few positions on the way.
        """
        last = len(line.states) - 1
        if last == 0:
            return self.ask(line.states[0])
        on_the_way = sorted({step * last // (SAMPLES + 1) for step in range(1, SAMPLES + 1)})
        for index in on_the_way:
            if index < last - 1 and not self.ask(line.states[index]):
                return False
        start, moves = line.window(last)
        if not (
            self.ask(line.states[last - 1])
            and self.ask(line.states[last])
            and self.ask(line.states[last], start, moves)
        ):
            return False
        return self._other_endings(line, last - 1)

    def _other_endings(self, line: _Line, index: int) -> bool:
        """Each other kind of ending one move from ``index``, asked both ways."""
        state = line.states[index]
        ended = self.engine.terminal(line.states[index + 1])
        seen = {ended.reason} if ended is not None else set()
        start, moves = line.window(index)
        asked = 0
        for move in self.engine.legal_moves(state):
            if asked >= OTHER_ENDINGS:
                break
            child = self.engine.apply(state, move)
            outcome = self.engine.terminal(child)
            if outcome is None or outcome.reason in seen:
                continue
            seen.add(outcome.reason)
            asked += 1
            spelled = _spell(self.engine, self.variant, state, move)
            if not (self.ask(child) and self.ask(child, start, [*moves, spelled])):
                return False
        return True

    def no_progress(self, line: _Line, index: int) -> bool:
        """The no-progress clock one ply short, then one move of each kind.

        The FEN carries our clock, so asking from it compares the limit. Their engine
        playing each move itself compares which moves reset the clock. Every kind of
        move is its own question, so each one that differs is reported, not only the
        first: which kinds differ is the finding.
        """
        state = line.states[index]
        limit = self.engine.spec.no_progress_draw
        short = state.with_(halfmove=limit - 1)
        about = "no_progress_draw"
        if not (
            self.ask(short, about=about)
            and self.ask(state.with_(halfmove=limit), about=about)
        ):
            return False
        agreed = True
        for move in _one_of_each_kind(self.engine, short):
            child = self.engine.apply(short, move)
            spelled = _spell(self.engine, self.variant, short, move)
            agreed = self.ask(child, short, [spelled], about=about) and agreed
        return agreed

    def ply_limit(self, line: _Line, index: int) -> bool:
        """The ply counter one short of our limit, and at it.

        Nothing crosses for `max_plies`, and a note says so, so at the limit the two are
        expected to differ and the difference is counted as admitted.
        """
        state = line.states[index]
        limit = self.engine.spec.max_plies
        return self.ask(state.with_(ply=limit - 1), about="max_plies") and self.ask(
            state.with_(ply=limit), about="max_plies"
        )

    def repetition(self, uniform: _Line, played: list[_Line]) -> bool:
        """A pair of moves played back and forth until our engine calls a repetition.

        Tried from the opening, then from positions on the way through the first game.
        Their engine is asked at the repetition and one ply before it, from the start of
        our window with the same moves.
        """
        limit = self.engine.spec.repetition_draw
        if limit == NO_REPETITION_DRAW:
            return self._never_repeated(uniform)
        last = len(uniform.states) - 1
        for index in sorted({0, *(step * last // 6 for step in range(1, 6))}):
            line = self._repeated(uniform.prefix(index), limit)
            if line is None:
                continue
            end = len(line.states) - 1
            before_start, before_moves = line.window(end - 1)
            start, moves = line.window(end)
            about = "repetition_draw"
            return self.ask(
                line.states[end - 1], before_start, before_moves, about=about
            ) and self.ask(line.states[end], start, moves, about=about)
        if not any(
            _ended_by(self.engine, game) == "repetition" for game in played
        ):
            self.check.unreached["repetition_draw"] = (
                "no pair of moves could be played back and forth into a repetition"
            )
        return True

    def _never_repeated(self, uniform: _Line) -> bool:
        """Under ``"never"`` (Q703): a position repeated past three folds, still going on.

        The same back-and-forth pair, played until the position has stood
        `NEVER_FOLDS` times, which is past the fold count their engine would otherwise
        draw at. Their engine is asked from the opening with every move since, so it has
        the whole history to count over, at the last repetition and one ply before it. A
        draw of theirs there is a difference under ``repetition_draw``.
        """
        last = len(uniform.states) - 1
        for index in sorted({0, *(step * last // 6 for step in range(1, 6))}):
            line = uniform.prefix(index)
            if self.engine.terminal(line.states[-1]) is not None:
                continue
            cycle = _cycle(self.engine, line.states[-1])
            if cycle is None:
                continue
            for step in range(4 * (NEVER_FOLDS - 1)):
                move = cycle[step % 4]
                if move not in self.engine.legal_moves(line.states[-1]):
                    break
                line = line.extended(self.engine, self.variant, move)
                if self.engine.terminal(line.states[-1]) is not None:
                    break
            else:
                end = len(line.states) - 1
                about = "repetition_draw"
                start = line.states[0]
                return self.ask(
                    line.states[end - 1], start, line.spelled[: end - 1], about=about
                ) and self.ask(line.states[end], start, line.spelled, about=about)
        self.check.unreached["repetition_draw"] = (
            "no pair of moves could be played back and forth past three folds"
        )
        return True

    def _repeated(self, line: _Line, limit: int) -> _Line | None:
        """``line`` extended by a back-and-forth pair until ours calls a repetition."""
        if self.engine.terminal(line.states[-1]) is not None:
            return None
        cycle = _cycle(self.engine, line.states[-1])
        if cycle is None:
            return None
        for step in range(4 * (limit + 2)):
            move = cycle[step % 4]
            if move not in self.engine.legal_moves(line.states[-1]):
                return None
            line = line.extended(self.engine, self.variant, move)
            outcome = self.engine.terminal(line.states[-1])
            if outcome is not None:
                return line if outcome.reason == "repetition" else None
        return None


def _ended_by(engine: Engine, line: _Line) -> str:
    outcome = engine.terminal(line.states[-1])
    return outcome.reason if outcome is not None else ""


def _hash(spec: dict[str, Any]) -> str:
    from chess_variant_sandbox.spec.canonical import spec_hash

    try:
        return spec_hash(spec)
    except Exception:  # noqa: BLE001
        return ""
