"""Second-pass measurements that need an engine, computed once and persisted.

The index reads persisted records and never touches a live engine. That invariant is
what lets a corpus be re-scored under new weights without re-simulating, and it is not
negotiable. But three of the things most worth knowing about a game cannot be read off
a result list:

* how often the player actually faced a decision, rather than a move list with one
  sane entry in it;
* whether the pieces the rules define are pieces anyone uses;
* whether a single bad move ends the game.

All three need positions, and positions need an engine. So they are computed here, in
a pass that runs after the tournament and writes its findings next to the records.
Metrics then read *those*, and the invariant holds: the engine is upstream of the
persisted file, never inside the scorer.

The pass costs almost nothing to re-run, because a record stores moves and the engine
re-derives every position from them. An existing corpus can be analysed without playing
a single new game -- which is the whole reason these measurements were worth adding at
this point rather than after the next expensive re-run.

**Sampling is deliberate.** Scoring every root move at every ply of every game costs
more than the tournament did. Positions are sampled on a fixed stride, so the
measurement is reproducible and cheap, and the quantity it estimates is a mean over
plies that does not need every ply to be exact.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox._core import REGRET_ITERATIONS, regret_solve
from chess_variant_sandbox.agents.evaluation import WIN_SCORE, Evaluator
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_FOLD,
    MOVE_ROTATE,
    move_fold,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.tournament.play import eval_values, finish_game
from chess_variant_sandbox.tournament.record import GameRecord

# The components this pass -- and only this pass -- can supply. Named here rather than
# in the scorer so that a weights version asking for one of them can say what to run,
# instead of reporting a missing measurement as an anonymous gap.
ANALYSIS_COMPONENTS = (
    "decision_criticality",
    "piece_utilisation",
    "blunder_recoverability",
    "position_revisit_rate",
    "capture_availability",
    "instance_participation",
)

# Every Nth ply is scored in full. Eight is short enough that a 60-ply game still
# contributes several samples and long enough that the pass stays far cheaper than the
# games it is reading.
PLY_STRIDE = 8

# Node budget for the analysis search. Deliberately fixed and modest: this is not a
# player, it is a probe asking "how far apart were the options here?", and it must mean
# the same thing in every ruleset and at every point in the corpus. It is unrelated to
# any agent's budget and must not be tuned to match one.
PROBE_NODES = 1200
PROBE_MAX_DEPTH = 8

# How many games per corpus get the blunder treatment. Each one is a full extra
# playout, so this is the most expensive knob in the module.
BLUNDER_SAMPLE = 48

# Where the blunder goes: a third of the way in. Early enough that there is a game left
# to play, late enough that the position has some structure to damage.
BLUNDER_FRACTION = 0.34


@dataclass
class GameAnalysis:
    """Engine-derived facts about one game, persisted for the scorer to read."""

    seed: int
    agents: tuple[str, str]
    # Per sampled ply: how many legal moves were within the cheapest piece's value of
    # the best one, and how many there were in total. Kept as counts rather than as a
    # ratio so that the corpus mean is over plies rather than over games.
    near_best: list[int] = field(default_factory=list)
    legal_counts: list[int] = field(default_factory=list)
    # How many times each piece type moved and captured, indexed by type, so the spec's
    # own piece list is the only vocabulary involved.
    moves_by_type: list[int] = field(default_factory=list)
    captures_by_type: list[int] = field(default_factory=list)
    # Per sampled ply, alongside `legal_counts`: how many of those legal moves took
    # something. The armies-never-engage measurement.
    capture_counts: list[int] = field(default_factory=list)
    # Positions whose key had already occurred in this game, and how many plies were
    # walked. "Back and forth and back and forth", counted.
    revisited_plies: int = 0
    walked_plies: int = 0
    # Pieces standing on the opening board, and how many of them ever moved. The axis
    # `moves_by_type` averages away: five pieces sitting in the back are invisible to
    # an entropy over piece *types* as long as their type moved somewhere else.
    starting_instances: int = 0
    moved_instances: int = 0
    # The core's position key after every ply, in order, so a measurement over the
    # positions a game passed through reads the record instead of replaying it. Empty in
    # an analysis written before the field existed.
    position_keys: list[int] = field(default_factory=list)
    # Schema 26 (Q918d): what the rounds' resolutions did, counted over the game
    # (`ROUND_COUNTS`), read off the core's events as the pass replays each round. ``None``
    # for a game that alternates, and then absent from the file, so an alternating game's
    # `analysis.json` is the bytes it always was.
    rounds: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
        if self.rounds is None:
            del data["rounds"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameAnalysis:
        payload = dict(data)
        payload["agents"] = tuple(payload["agents"])
        return cls(**payload)


@dataclass
class BlunderTrial:
    """One game replayed with a single random move forced into it."""

    seed: int
    agents: tuple[str, str]
    ply: int
    original_winner: int | None
    blundering_side: int
    new_winner: int | None

    @property
    def recovered(self) -> bool:
        """Whether the side made to blunder still avoided losing."""
        return self.new_winner != (1 - self.blundering_side)

    @property
    def changed(self) -> bool:
        return self.new_winner != self.original_winner

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlunderTrial:
        payload = dict(data)
        payload["agents"] = tuple(payload["agents"])
        return cls(**payload)


@dataclass
class CorpusAnalysis:
    """Everything the second pass found, for one ruleset's corpus."""

    games: list[GameAnalysis] = field(default_factory=list)
    blunders: list[BlunderTrial] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": [g.as_dict() for g in self.games],
            "blunders": [b.as_dict() for b in self.blunders],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CorpusAnalysis:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            games=[GameAnalysis.from_dict(g) for g in data.get("games", [])],
            blunders=[BlunderTrial.from_dict(b) for b in data.get("blunders", [])],
        )


def analyse_corpus(
    spec: dict[str, Any],
    records: list[GameRecord],
    blunder_sample: int = BLUNDER_SAMPLE,
    seed: int = 0,
    progress: bool = False,
    workers: int = 0,
) -> CorpusAnalysis:
    """Run the second pass over a finished corpus.

    Parallel by game, for the same reason the tournament is: the probe search dominates
    the cost and every game is independent. Workers compile the ruleset once and keep
    it, because compilation builds a table per (player, piece type, square) triple and
    paying for that per game would cost more than the analysis.
    """
    engine = Engine(spec)
    epsilon = _epsilon_for(engine)

    graded = [r for r in records if _searching(r)] or records
    resolved = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    if resolved == 1 or len(graded) < 8:
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        analyses = [_analyse_game(engine, evaluator, record, epsilon) for record in graded]
        blunders = _blunder_trials(engine, graded, blunder_sample, seed, progress)
        return CorpusAnalysis(games=analyses, blunders=blunders)

    chosen = _blunder_choices(graded, blunder_sample, seed)
    try:
        with ProcessPoolExecutor(
            max_workers=resolved, initializer=_init_worker, initargs=(spec,)
        ) as pool:
            analyses = list(pool.map(_analyse_one, graded, chunksize=2))
            trials = list(pool.map(_blunder_one, [(r, seed) for r in chosen], chunksize=1))
    except BrokenExecutor:
        # A worker died and the child's cause does not survive to be reported. The one
        # case seen so far was a caller without an ``if __name__ == "__main__"`` guard,
        # which on Windows makes every spawned child re-run the caller's own work -- a
        # mistake in the caller, not here, but one that surfaces as a dead pool. A
        # calibration corpus takes hours; finishing serially and saying so is a far
        # better outcome than losing it.
        print(
            "  analysis worker pool died; falling back to a single process",
            flush=True,
        )
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        analyses = [_analyse_game(engine, evaluator, record, epsilon) for record in graded]
        trials = [_one_blunder(engine, record, blunder_rng(seed, record))
                  for record in chosen]

    return CorpusAnalysis(
        games=analyses, blunders=[t for t in trials if t is not None]
    )


def _epsilon_for(engine: Engine) -> float:
    """The value of the cheapest piece: the threshold for "these were alternatives".

    Spec-derived, so it means the same thing in a game with no pawns as in one with
    five kinds of them, and nothing about it assumes a chess piece exists. Asked of the
    *opening position* rather than of the value table, because on a ruleset whose pieces
    are worth what they may still take the two are different numbers -- see
    ``Evaluator.cheapest_piece``.
    """
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    return evaluator.cheapest_piece(engine.initial_state().board)


# Each worker compiles the ruleset once. Mirrors the tournament runner deliberately:
# two modules paying the same cost should pay it the same way.
_WORKER_ENGINE: Engine | None = None
_WORKER_EVALUATOR: Evaluator | None = None
_WORKER_EPSILON = 1.0


def _init_worker(spec: dict[str, Any]) -> None:
    global _WORKER_ENGINE, _WORKER_EVALUATOR, _WORKER_EPSILON
    _WORKER_ENGINE = Engine(spec)
    _WORKER_EVALUATOR = Evaluator.for_engine(_WORKER_ENGINE, mobility_weight=0.0)
    _WORKER_EPSILON = _epsilon_for(_WORKER_ENGINE)


def _analyse_one(record: GameRecord) -> GameAnalysis:
    assert _WORKER_ENGINE is not None and _WORKER_EVALUATOR is not None
    return _analyse_game(_WORKER_ENGINE, _WORKER_EVALUATOR, record, _WORKER_EPSILON)


def _blunder_one(task: tuple[GameRecord, int]) -> BlunderTrial | None:
    assert _WORKER_ENGINE is not None
    record, seed = task
    return _one_blunder(_WORKER_ENGINE, record, blunder_rng(seed, record))


def _blunder_choices(
    records: list[GameRecord], sample: int, seed: int
) -> list[GameRecord]:
    candidates = [r for r in records if r.plies >= 12]
    if not candidates:
        return []
    rng = random.Random(seed ^ 0x5EED)
    return candidates if len(candidates) <= sample else rng.sample(candidates, sample)


def _searching(record: GameRecord) -> bool:
    """Games where both seats were held by a player that actually chose."""
    return all(agent.split(":")[0] != "random" for agent in record.agents)


def _analyse_game(
    engine: Engine, evaluator: Evaluator, record: GameRecord, epsilon: float
) -> GameAnalysis:
    if engine.simultaneous:
        return _analyse_rounds(engine, evaluator, record, epsilon)
    n_types = engine.spec.n_types
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    near_best: list[int] = []
    legal_counts: list[int] = []
    capture_counts: list[int] = []

    state = engine.initial_state()
    # Which of the pieces standing at the start ever moved. Tracked by origin square,
    # which is the cheap version and is wrong in two bounded ways: a castling rook is
    # not credited, because the move is encoded at the king's square, and a piece
    # captured on its home square followed by an enemy piece moving off that square
    # credits the wrong army. Both are worth a handful of instances per game against a
    # denominator in the tens, and neither is worth a per-ply board diff to fix.
    starting_squares = _starting_squares(engine, state.board)
    moved_squares: set[int] = set()
    seen_keys = {state.key}
    position_keys: list[int] = []
    revisited = 0
    walked = 0

    for index, move in enumerate(record.moves):
        board = state.board
        kind = _mover_kind(engine, board, move)
        if kind is None:
            # The record and the spec have drifted apart. Stop counting rather than
            # inventing numbers; replay raises on the same condition with a message.
            break
        if _moves_an_instance(move) and move_from(move) in starting_squares:
            moved_squares.add(move_from(move))
        moves_by_type[kind] += 1
        if _takes(board, move):
            captures_by_type[kind] += 1

        if index % PLY_STRIDE == 0:
            legal = engine.legal_moves(state)
            if len(legal) > 1:
                scores = _score_roots(engine, evaluator, state, legal)
                best = max(scores)
                near_best.append(sum(1 for s in scores if best - s <= epsilon))
                legal_counts.append(len(legal))
                # Same sample, so the two ratios are over the same plies. "Occupied
                # destination" is the same reading of a capture that
                # `captures_by_type` uses two lines up; en passant is missed by both,
                # consistently.
                capture_counts.append(sum(1 for m in legal if _takes(board, m)))

        state = engine.apply(state, move)
        walked += 1
        position_keys.append(state.key)
        if state.key in seen_keys:
            revisited += 1
        else:
            seen_keys.add(state.key)

    return GameAnalysis(
        seed=record.seed,
        agents=record.agents,
        near_best=near_best,
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=walked,
        starting_instances=len(starting_squares),
        moved_instances=len(moved_squares),
        position_keys=position_keys,
    )


def _mover_kind(engine: Engine, board: bytes, move: int) -> int | None:
    """The internal type a move counts under in `moves_by_type`, or `None` from an empty
    origin, which only a record that has drifted from its spec plays.

    A drop counts under the type it drops. A fold's own move (schema 22, revised at 24)
    counts under its end's internal type: a fold stands on no byte of the board, so its
    origin holds nothing, or a piece standing in it that does not move (Q873, Q844's
    finding: the pass used to stop here). Every other move counts under the type on its
    origin, a momentum end's own move under its end's byte. `site/legality.py::walk` reads
    a move through this too.
    """
    n_types = engine.spec.n_types
    kind = move_type(move)
    if kind == MOVE_DROP:
        return move_promotion(move) % n_types
    if kind == MOVE_FOLD:
        portals = engine.spec.portals
        assert portals is not None
        return portals.end_types[move_fold(move)]
    occupant = board[move_from(move)]
    return None if occupant == 0 else (occupant - 1) % n_types


def _moves_an_instance(move: int) -> bool:
    """Whether a move moves the piece on its origin square: a drop moves none, and a fold's
    own move leaves the piece standing in it where it is."""
    return move_type(move) not in (MOVE_DROP, MOVE_FOLD)


def _starting_squares(engine: Engine, board: bytes) -> set[int]:
    """The squares of the opening board's pieces: the instances `instance_participation`
    counts.

    A portal is not one of them (Q875, `docs/reference/portal-showcase-ideas.md` beside
    finding 1). It is part of the board that pieces move through, held by no player and
    taken by no capture, so a fixed portal is not an idle piece and a moved one is not a
    piece that played. A fold stands on no byte and was never counted; a momentum end
    stands on its byte and is left out here, so the two kinds read alike.
    """
    portals = engine.spec.portals
    n_types = engine.spec.n_types
    return {
        square
        for square in range(engine.spec.n_squares)
        if board[square] != 0
        and (portals is None or not portals.is_end((board[square] - 1) % n_types))
    }


def _takes(board: bytes, move: int) -> bool:
    """Whether a move is read as a capture: it ends on an occupied square, and is not one of
    the three kinds that end on one and take nothing, as the core's `apply` reads them.

    A rotation's target is its own square, which holds the piece that turns (Q875: it was
    read as a capture, so a game whose pieces turn offered one in nearly every position).
    A file-based castling can land the king on its own rook, or on its own square, and
    moves the rook rather than removing it. A fold's own move may end on a piece and leave
    it standing (`docs/plan-portals-2.md` §3R.1 rule 6). These are the three
    `agents.rs::capture_gain` and `agents/minimax.py` leave out.
    """
    return board[move_to(move)] != 0 and move_type(move) not in (
        MOVE_ROTATE,
        MOVE_CASTLE,
        MOVE_FOLD,
    )


class _ProbeExhausted(Exception):
    """Raised to abandon a probe iteration that ran out of budget."""


class _NodeCounter:
    __slots__ = ("used", "budget")

    def __init__(self, budget: int) -> None:
        self.used = 0
        self.budget = budget


def _score_roots(
    engine: Engine, evaluator: Evaluator, state: GameState, legal: list[int]
) -> list[float]:
    """The probe, in the core where there is one.

    ``rust/src/probe.rs`` is a line-for-line port of ``_root_scores`` below, ragged edge
    included, and the two are held to producing identical vectors -- this pass is the
    dominant cost of scoring a ruleset and it is a search over one position at a time,
    which is exactly the shape the boundary could not serve until ``root_scores``
    existed. The Python version stays as the oracle and as the path for a ruleset the
    core cannot play.

    The ruler is fixed: the derived table from ``eval_values``, never a fitted
    objective, even where the rated players use one. ``docs/reference/decision-probe-ruler.md``.
    It reads the goal terms and, since Q471, the immunity term at the weights
    ``evaluator`` carries, which are the rated players' defaults.
    """
    if engine.core is not None:
        return engine.root_scores(
            state,
            eval_values(engine.spec),
            PROBE_NODES,
            PROBE_MAX_DEPTH,
            evaluator.goal_pieces,
            evaluator.immune_pieces,
        )
    return _root_scores(engine, evaluator, state, legal)


def _root_scores(
    engine: Engine, evaluator: Evaluator, state: GameState, legal: list[int]
) -> list[float]:
    """Evaluate every legal move at the probe budget, with a full window.

    The full window is the point. A playing search evaluates root moves inside an
    alpha-beta window, so everything except the best move comes back as a bound rather
    than a value -- fine for choosing a move, useless for asking how close the
    alternatives were. Reusing those numbers would systematically overstate how many
    real options a position offered, which is exactly the quantity being measured.

    Budget is nodes rather than depth, matching the rule that governs agents: a fixed
    depth is wildly different amounts of work in a game with eight legal moves and one
    with sixty, and this number has to be comparable across rulesets.
    """
    budget = max(len(legal) * 2, PROBE_NODES)
    counter = _NodeCounter(budget)
    scores = [0.0] * len(legal)
    for depth in range(1, PROBE_MAX_DEPTH + 1):
        try:
            for index, move in enumerate(legal):
                # See ``_probe``: the negation is only right when the side changed.
                child = engine.apply(state, move)
                value = _probe(engine, evaluator, child, depth - 1, counter)
                scores[index] = (
                    -value if child.side_to_move != state.side_to_move else value
                )
        except _ProbeExhausted:
            break
        if counter.used >= budget:
            break
    return scores


def _probe(
    engine: Engine, evaluator: Evaluator, state: GameState, depth: int, counter: _NodeCounter
) -> float:
    counter.used += 1
    if counter.used >= counter.budget:
        raise _ProbeExhausted

    moves = engine.legal_moves(state)
    outcome = engine.terminal(state, moves)
    if outcome is not None:
        if outcome.is_draw:
            return 0.0
        signed = WIN_SCORE - state.ply
        return signed if outcome.winner == state.side_to_move else -signed
    if depth <= 0:
        return evaluator.evaluate(engine, state, state.side_to_move, len(moves))

    best = float("-inf")
    for move in moves:
        # Negamax negates the child because the side is assumed to alternate. A turn
        # longer than one move makes that conditional: when the same player moves again
        # the child's value is already from this player's point of view. Asked of the
        # states rather than of the turn rule, since a ramp changes the answer ply by
        # ply. ``max`` keeps its first argument on a tie, and ``probe.rs`` agrees.
        child = engine.apply(state, move)
        value = _probe(engine, evaluator, child, depth - 1, counter)
        best = max(best, -value if child.side_to_move != state.side_to_move else value)
    return best


def _blunder_trials(
    engine: Engine,
    records: list[GameRecord],
    sample: int,
    seed: int,
    progress: bool,
) -> list[BlunderTrial]:
    """Force one random move into a game and finish it with the same players.

    The question is not whether the blunder gets punished -- a competent opponent
    punishes everything -- but whether the game is still a game afterwards. A ruleset
    where one arbitrary move always loses is brittle: it rewards not erring rather than
    playing well, and the winner is whoever slips second. One where the result never
    moves is not responding to move quality at all, which is the dull failure wearing a
    different hat. Both ends are bad, so this is scored as a band.
    """
    chosen = _blunder_choices(records, sample, seed)
    if not chosen:
        return []

    trials: list[BlunderTrial] = []
    for number, record in enumerate(chosen):
        if progress and number % 8 == 0:
            print(f"  blunder trial {number}/{len(chosen)}", flush=True)
        trial = _one_blunder(engine, record, blunder_rng(seed, record))
        if trial is not None:
            trials.append(trial)
    return trials


def blunder_rng(seed: int, record: GameRecord) -> random.Random:
    """The stream that picks the move forced into one game.

    Derived per record rather than drawn from one stream walked in order, so that which
    move gets forced into a game does not depend on how many games were analysed before
    it -- or on how the work was divided. This loop and the worker pool used to derive
    it differently, which made ``blunder_recoverability`` depend on ``--workers``: the
    same corpus analysed serially and in parallel picked different moves and could score
    differently. That is the same class of bug as a tournament whose result depends on
    its thread count, and it is ruled out the same way, by fixing every trial's seed
    before any work starts.
    """
    return random.Random((seed ^ 0x5EED) + record.seed)


def _one_blunder(
    engine: Engine, record: GameRecord, rng: random.Random
) -> BlunderTrial | None:
    if engine.simultaneous:
        return _one_round_blunder(engine, record, rng)
    target_ply = int(record.plies * BLUNDER_FRACTION)
    state = engine.initial_state()
    for move in record.moves[:target_ply]:
        if move not in engine.legal_moves(state):
            return None
        state = engine.apply(state, move)

    legal = engine.legal_moves(state)
    if len(legal) < 2:
        return None

    blundering = state.side_to_move
    played = record.moves[target_ply] if target_ply < len(record.moves) else None
    alternatives = [m for m in legal if m != played] or legal
    state = engine.apply(state, rng.choice(alternatives))

    # Same players and the same per-seat streams as the original game, so the only
    # difference between this playout and the recorded one is the move forced in. The
    # players are the *rated* ones: ``finish_game`` runs the core's agents wherever the
    # core implements them, so a trial is not quietly measuring a second implementation.
    resumed = finish_game(engine, record.agents, state, record.seed)

    return BlunderTrial(
        seed=record.seed,
        agents=record.agents,
        ply=target_ply,
        original_winner=record.winner,
        blundering_side=blundering,
        new_winner=resumed.winner,
    )


# ------------------------------------------------------- schema 26: games in rounds
#
# `docs/plan-simultaneous.md` §6.3 (Q918d). A round game's record keeps both moves of each
# round in seat order and every per-ply series twice, so a round is two plies; the pass
# walks it a round at a time through `apply_round`. Two of its measurements need a reading
# of their own, because with no mover there is no single list of options to score:
#
# * **the probe** values *each* player's moves against the opponent's equilibrium mix of
#   the round's matrix game, one round deep, at the probe's fixed ruler (the built-in
#   evaluator, mobility off, the goal and immunity terms at their weights; a finished round
#   at ±(`WIN_SCORE` − ply)). The matrix is solved by the round players' own solver, the
#   core's regret matching+ (`_core.regret_solve`). A move is near the best when it is
#   within the cheapest piece of the best against that mix, as an alternating probe's is;
#   each seat with a choice is one sample, so a sampled round gives two.
# * **a blunder** is a move whose value against the mix falls short of the best by more
#   than that threshold. The trial forces one, picked at random among them, into the
#   seat whose ply the trial falls on, beside the other seat's recorded move (it chose
#   without seeing either), and finishes the game with the same players. A position with
#   no such move offers no blunder and gives no trial.
#
# Everything else the pass reads -- types moved, instances, captures offered, positions
# revisited -- reads both moves of a round on the round's board, where both were chosen.

# What the pass counts of the rounds' resolutions, from the core's events (`Engine.
# apply_round_with_phase`: 0 a collision with its winner or -1, 1 a void move, 2 a tie the
# initiative decided, 3 a pair barred), and a swap read off the two move words.
ROUND_COUNTS = (
    "rounds",
    "collisions",
    "collisions_both_lost",
    "void_moves",
    "initiative_ties",
    "bars",
    "swaps",
)

_COLLISION, _VOID, _INITIATIVE, _BARRED = 0, 1, 2, 3

# The share of a mix read as the solver's residue rather than as a move the equilibrium
# plays: what the first two of the solver's iterations alone can leave on a move, since
# its average weights iteration t by t. Regret matching+'s average keeps that much of its
# opening iterations on moves the equilibrium abandons, and against a finished round worth
# ±`WIN_SCORE` a residue of 1e-5 is ten points, enough to push a sound move out of reach
# of the cheapest piece. Decided before any round game was measured (Q918d's plan entry).
DUST = 3.0 / (REGRET_ITERATIONS * (REGRET_ITERATIONS + 1) / 2)


@dataclass(frozen=True)
class RoundReading:
    """One round's matrix game read by the probe: each seat's legal moves, the mixes the
    solver found, and each move's value against the opponent's mix, from its own seat."""

    moves: tuple[list[int], list[int]]
    strategies: tuple[list[float], list[float]]
    values: tuple[list[float], list[float]]

    def near_best(self, seat: int, epsilon: float) -> int:
        values = self.values[seat]
        best = max(values)
        return sum(1 for value in values if best - value <= epsilon)

    def blunders(self, seat: int, epsilon: float) -> list[int]:
        """The seat's moves that fall short of its best against the mix by more than
        ``epsilon``, in generation order."""
        values = self.values[seat]
        best = max(values)
        return [
            move for move, value in zip(self.moves[seat], values, strict=True)
            if best - value > epsilon
        ]


def read_matrix(
    moves: tuple[list[int], list[int]], cells: list[float]
) -> RoundReading:
    """A round's matrix game, ``cells`` row-major from player 0's side (rows its moves,
    columns player 1's), solved and read: each seat's moves valued against the other's
    mix with its residue below `DUST` cleared. Sums run in plain loops, in order."""
    rows, cols = len(moves[0]), len(moves[1])
    p, q = regret_solve(cells, rows, cols)
    first, second = _purified(p), _purified(q)
    row_values = []
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += cells[i * cols + j] * second[j]
        row_values.append(total)
    col_values = []
    for j in range(cols):
        total = 0.0
        for i in range(rows):
            total += cells[i * cols + j] * first[i]
        col_values.append(-total)
    return RoundReading(moves=moves, strategies=(first, second), values=(row_values, col_values))


def _purified(strategy: list[float]) -> list[float]:
    kept = [weight if weight >= DUST else 0.0 for weight in strategy]
    total = 0.0
    for weight in kept:
        total += weight
    if total <= 0.0:
        return list(strategy)
    return [weight / total for weight in kept]


def read_round(
    engine: Engine,
    evaluator: Evaluator,
    state: GameState,
    moves: tuple[list[int], list[int]] | None = None,
) -> RoundReading | None:
    """The probe on one round's board, or ``None`` when either player has no move."""
    if moves is None:
        moves = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    if not moves[0] or not moves[1]:
        return None
    cells = []
    for first in moves[0]:
        for second in moves[1]:
            child = engine.apply_round(state, first, second)
            outcome = engine.terminal(child)
            if outcome is None:
                # Mobility is off at the ruler, so the move counts weigh nothing.
                cells.append(evaluator.evaluate_round(engine, child, 0, (0, 0)))
            elif outcome.is_draw:
                cells.append(0.0)
            else:
                signed = WIN_SCORE - child.ply
                cells.append(signed if outcome.winner == 0 else -signed)
    return read_matrix(moves, cells)


def replay_rounds(engine: Engine, record: GameRecord) -> list[GameState]:
    """The position before the first round and after every round of a round game's record.

    `play.replay`'s counterpart for a game in rounds: a record holds both moves of each
    round in seat order, and an illegal one means the record and its spec have drifted.
    """
    if len(record.moves) % 2:
        raise ValueError(
            f"game {record.seed} holds {len(record.moves)} moves; a game in rounds holds two "
            "a round"
        )
    state = engine.initial_state()
    states = [state]
    for index in range(0, len(record.moves), 2):
        first, second = record.moves[index], record.moves[index + 1]
        if (
            first not in engine.legal_moves_for(state, 0)
            or second not in engine.legal_moves_for(state, 1)
        ):
            raise ValueError(
                f"round {index // 2} of game {record.seed} is not legal under ruleset "
                f"{record.ruleset_hash}; the record and the spec disagree"
            )
        state = engine.apply_round(state, first, second)
        states.append(state)
    return states


def _is_swap(first: int, second: int) -> bool:
    """Two moves onto each other's origin, read off the words (§2.4: two chases)."""
    if MOVE_DROP in (move_type(first), move_type(second)):
        return False
    return move_to(first) == move_from(second) and move_to(second) == move_from(first)


def _analyse_rounds(
    engine: Engine, evaluator: Evaluator, record: GameRecord, epsilon: float
) -> GameAnalysis:
    """`_analyse_game` for a game in rounds (the reading above `ROUND_COUNTS`)."""
    n_types = engine.spec.n_types
    moves_by_type = [0] * n_types
    captures_by_type = [0] * n_types
    near_best: list[int] = []
    legal_counts: list[int] = []
    capture_counts: list[int] = []
    counts = dict.fromkeys(ROUND_COUNTS, 0)

    state = engine.initial_state()
    starting_squares = _starting_squares(engine, state.board)
    moved_squares: set[int] = set()
    seen_keys = {state.key}
    position_keys: list[int] = []
    revisited = 0
    walked = 0

    for index in range(0, len(record.moves) - 1, 2):
        pair = (record.moves[index], record.moves[index + 1])
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        if pair[0] not in lists[0] or pair[1] not in lists[1]:
            # Drifted from its spec; stop counting, as the alternating walk does.
            break
        board = state.board
        for move in pair:
            kind = _mover_kind(engine, board, move)
            if kind is None:  # pragma: no cover - a legal move stands on its origin
                continue
            if _moves_an_instance(move) and move_from(move) in starting_squares:
                moved_squares.add(move_from(move))
            moves_by_type[kind] += 1
            if _takes(board, move):
                captures_by_type[kind] += 1

        # The ply stride, read at the round's first ply: every fourth round.
        if index % PLY_STRIDE == 0:
            reading = read_round(engine, evaluator, state, lists)
            if reading is not None:
                for seat in (0, 1):
                    if len(lists[seat]) > 1:
                        near_best.append(reading.near_best(seat, epsilon))
                        legal_counts.append(len(lists[seat]))
                        capture_counts.append(
                            sum(1 for move in lists[seat] if _takes(board, move))
                        )

        state, _, events = engine.apply_round_with_phase(state, pair[0], pair[1])
        counts["rounds"] += 1
        counts["swaps"] += int(_is_swap(pair[0], pair[1]))
        for event, _square, player in events:
            if event == _COLLISION:
                counts["collisions"] += 1
                counts["collisions_both_lost"] += int(player < 0)
            elif event == _VOID:
                counts["void_moves"] += 1
            elif event == _INITIATIVE:
                counts["initiative_ties"] += 1
            elif event == _BARRED:
                counts["bars"] += 1

        # Two plies a round, as every per-ply series in the record: the key after the
        # round twice, and a revisit counted on both, so the rate is per round.
        walked += 2
        position_keys.extend((state.key, state.key))
        if state.key in seen_keys:
            revisited += 2
        else:
            seen_keys.add(state.key)

    return GameAnalysis(
        seed=record.seed,
        agents=record.agents,
        near_best=near_best,
        legal_counts=legal_counts,
        moves_by_type=moves_by_type,
        captures_by_type=captures_by_type,
        capture_counts=capture_counts,
        revisited_plies=revisited,
        walked_plies=walked,
        starting_instances=len(starting_squares),
        moved_instances=len(moved_squares),
        position_keys=position_keys,
        rounds=counts,
    )


def round_blunder(
    engine: Engine,
    state: GameState,
    seat: int,
    played: int,
    rng: random.Random,
) -> int | None:
    """The move a trial forces into ``seat`` on this round's board: one of its blunders
    other than the move it played, drawn by ``rng``, or ``None`` when it has none."""
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    reading = read_round(engine, evaluator, state)
    if reading is None or len(reading.moves[seat]) < 2:
        return None
    epsilon = _epsilon_for(engine)
    blunders = [move for move in reading.blunders(seat, epsilon) if move != played]
    return rng.choice(blunders) if blunders else None


def _one_round_blunder(
    engine: Engine, record: GameRecord, rng: random.Random
) -> BlunderTrial | None:
    """`_one_blunder` for a game in rounds: the trial's ply names a round and a seat."""
    target_ply = int(record.plies * BLUNDER_FRACTION)
    target_round, seat = divmod(target_ply, 2)
    start = 2 * target_round
    if start + 1 >= len(record.moves):
        return None
    state = engine.initial_state()
    for index in range(0, start, 2):
        first, second = record.moves[index], record.moves[index + 1]
        if (
            first not in engine.legal_moves_for(state, 0)
            or second not in engine.legal_moves_for(state, 1)
        ):
            return None
        state = engine.apply_round(state, first, second)

    played, other = record.moves[start + seat], record.moves[start + 1 - seat]
    if other not in engine.legal_moves_for(state, 1 - seat):
        return None
    forced = round_blunder(engine, state, seat, played, rng)
    if forced is None:
        return None
    pair = (forced, other) if seat == 0 else (other, forced)
    state = engine.apply_round(state, pair[0], pair[1])
    resumed = finish_game(engine, record.agents, state, record.seed)

    return BlunderTrial(
        seed=record.seed,
        agents=record.agents,
        ply=target_ply,
        original_winner=record.winner,
        blundering_side=seat,
        new_winner=resumed.winner,
    )
