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

from chess_variant_sandbox.agents.evaluation import WIN_SCORE, Evaluator
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import (
    MOVE_DROP,
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

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
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
    starting_squares = {
        square for square in range(engine.spec.n_squares) if state.board[square] != 0
    }
    moved_squares: set[int] = set()
    seen_keys = {state.key}
    position_keys: list[int] = []
    revisited = 0
    walked = 0

    for index, move in enumerate(record.moves):
        board = state.board
        if move_type(move) == MOVE_DROP:
            kind = move_promotion(move) % n_types
        else:
            origin = move_from(move)
            occupant = board[origin]
            if occupant == 0:
                # The record and the spec have drifted apart. Stop counting rather than
                # inventing numbers; replay raises on the same condition with a message.
                break
            kind = (occupant - 1) % n_types
            if origin in starting_squares:
                moved_squares.add(origin)
        moves_by_type[kind] += 1
        if board[move_to(move)] != 0:
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
                capture_counts.append(sum(1 for m in legal if board[move_to(m)] != 0))

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
