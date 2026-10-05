"""Winning chances: a per-game fit from a position's value to the side to move's result.

``docs/reference/win-probability.md`` is the record and its pre-registration (Q701). The
fit is ``p(x) = 1 / (1 + exp(-k x))`` over the decision probe's value of a position, in
units of the game's cheapest piece, labelled by how the game ended for the side to move
(a draw counts half). It is fitted on one half of a run's games and judged on the other.

Three layers, each a pure function of the one before:

- **readings** (``read_game``, ``read_run``): the core's root scores for every position of
  a game's stored moves. The only layer that asks the engine anything, and it only asks:
  every value is the core's ``root_scores`` (rule 3), Python picks the largest and finds
  the played move in the list the engine itself generated.
- **the fit** (``fit``): the slope, from readings and a seed. Deterministic.
- **the verdict** (``calibration``): the held-out table and the bound stated in the record
  before any data.

Nothing here is under ``metrics/`` or in any ``measurement`` source set, and nothing here
writes a record, a score, a rating or a verdict on a game: it reads. A slope or a chance
never leaves this module without the ``Basis`` that says what it was fitted on.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from chess_variant_sandbox.tournament.record import GameRecord

METHOD = "winprob-1"

# A value this far out is a search that saw the game end: ``WIN_SCORE`` minus the ply.
# Half of it is beyond any material sum and short of any real win.
DECIDED_FRACTION = 0.5

# The slope's search interval, per cheapest piece. 50 is a step function at a fraction
# of a piece; nothing that reads a position's value is sharper than that.
K_MAX = 50.0

# The bound, as pre-registered in `win-probability.md` §1. Never re-aimed.
BINS = 10
MIN_HELD_POSITIONS = 200
MIN_HELD_GAMES = 10
MAX_MEAN_ERROR = 0.05
BIN_FLOOR = 50
BIN_SLACK = 0.10

CALIBRATED = "calibrated"
ROUGH = "rough"
TOO_FEW = "too few"


class _Engine(Protocol):
    """What a reading needs of an engine: ``Engine`` has it, and so does a test's fake."""

    def initial_state(self) -> Any: ...

    def legal_moves(self, state: Any) -> list[int]: ...

    def apply(self, state: Any, move: int) -> Any: ...


# --------------------------------------------------------------------------- the ruler


@dataclass(frozen=True)
class Ruler:
    """How a position was valued: the probe's budget and weights, and the game's unit."""

    nodes: int
    max_depth: int
    goal: float
    immune: float
    unit: float
    decided_at: float
    build: str

    def scale(self, value: float) -> float:
        return value / self.unit


def ruler_for(engine: Any) -> Ruler:
    """The decision probe's ruler on this engine (`decision-probe-ruler.md`).

    Reused rather than chosen: it is the one fixed-budget, full-window valuation the
    project already has, and a second ruler would be a second thing to keep honest.
    """
    from chess_variant_sandbox import _core
    from chess_variant_sandbox.agents.evaluation import WIN_SCORE, Evaluator
    from chess_variant_sandbox.tournament.analysis import PROBE_MAX_DEPTH, PROBE_NODES

    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    unit = evaluator.cheapest_piece(engine.initial_state().board)
    return Ruler(
        nodes=PROBE_NODES,
        max_depth=PROBE_MAX_DEPTH,
        goal=evaluator.goal_pieces,
        immune=evaluator.immune_pieces,
        unit=unit if unit > 0 else 1.0,
        decided_at=WIN_SCORE * DECIDED_FRACTION,
        build=str(getattr(_core, "BUILD_FINGERPRINT", "")),
    )


def _root_scores(engine: Any, state: Any, ruler: Ruler) -> list[float]:
    from chess_variant_sandbox.tournament.play import eval_values

    return list(
        engine.root_scores(
            state, eval_values(engine.spec), ruler.nodes, ruler.max_depth, ruler.goal, ruler.immune
        )
    )


# ------------------------------------------------------------------------- readings


@dataclass(frozen=True)
class Position:
    """One position of a stored game, as the core valued it from the mover's side."""

    ply: int
    mover: int
    choices: int
    best: float
    played: float
    best_move: int

    @property
    def forced(self) -> bool:
        return self.choices < 2


@dataclass(frozen=True)
class GameReading:
    """A game's positions and how it ended, which is all a fit needs of it."""

    index: int
    seed: int
    agents: tuple[str, str]
    winner: int | None
    positions: tuple[Position, ...]

    def result_for(self, player: int) -> float:
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == player else 0.0


def read_game(
    engine: Any,
    record: GameRecord,
    ruler: Ruler,
    index: int = 0,
    plies: Iterable[int] | None = None,
    scores: Any = None,
) -> GameReading:
    """Replay ``record`` and ask the core to value the positions before ``plies``.

    Every ply when ``plies`` is ``None``. ``scores(engine, state, ruler)`` is the core's
    ``root_scores`` unless a test hands in its own. The score list must be the legal list
    in the order the engine generated it; a list of another length is an error, never
    lined up by guess.
    """
    ask = scores or _root_scores
    wanted = None if plies is None else set(plies)
    state = engine.initial_state()
    positions: list[Position] = []
    for ply, move in enumerate(record.moves):
        legal = engine.legal_moves(state)
        if move not in legal:
            raise ValueError(f"game {index}: ply {ply} plays a move the engine calls illegal")
        if wanted is None or ply in wanted:
            values = ask(engine, state, ruler)
            if len(values) != len(legal):
                raise ValueError(
                    f"game {index}: ply {ply} has {len(legal)} moves and {len(values)} scores"
                )
            top = max(range(len(values)), key=values.__getitem__)
            positions.append(
                Position(
                    ply=ply,
                    mover=int(state.side_to_move),
                    choices=len(legal),
                    best=float(values[top]),
                    played=float(values[legal.index(move)]),
                    best_move=int(legal[top]),
                )
            )
        state = engine.apply(state, move)
    return GameReading(
        index=index,
        seed=record.seed,
        agents=(str(record.agents[0]), str(record.agents[1])),
        winner=record.winner,
        positions=tuple(positions),
    )


def sample_plies(record: GameRecord, index: int, seed: int, cap: int | None) -> list[int] | None:
    """At most ``cap`` plies of one game, drawn by the seed and the game; ``None`` is all."""
    if cap is None or cap >= len(record.moves):
        return None
    rng = random.Random(_digest(f"plies:{seed}:{index}:{record.seed}:{record.agents}"))
    return sorted(rng.sample(range(len(record.moves)), cap))


def read_run(
    engine: Any,
    records: Sequence[GameRecord],
    ruler: Ruler,
    seed: int = 0,
    positions_per_game: int | None = None,
    scores: Any = None,
) -> list[GameReading]:
    """Every game's readings, each a seeded sample of at most ``positions_per_game``."""
    return [
        read_game(
            engine,
            record,
            ruler,
            index=index,
            plies=sample_plies(record, index, seed, positions_per_game),
            scores=scores,
        )
        for index, record in enumerate(records)
    ]


# --------------------------------------------------------------------------- the fit


def chances(k: float, x: float) -> float:
    """``p(x)`` for slope ``k``: monotone in ``x``, and ``chances(k, -x) == 1 - chances(k, x)``."""
    z = k * x
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def held_out(seed: int, reading: GameReading) -> bool:
    """Which half a game is in: by game, never by position, and only the seed moves it."""
    key = f"split:{seed}:{reading.index}:{reading.seed}:{reading.agents[0]}:{reading.agents[1]}"
    return _digest(key) % 2 == 1


@dataclass(frozen=True)
class Basis:
    """What a fit was made of. Every number this module gives out carries one."""

    method: str
    ruleset_hash: str
    seed: int
    records: str
    fit_games: int
    fit_positions: int
    held_games: int
    held_positions: int
    decided: int
    positions_per_game: int | None
    ruler: dict[str, Any]


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    count: int
    predicted: float
    observed: float

    @property
    def error(self) -> float:
        return abs(self.predicted - self.observed)

    @property
    def se(self) -> float:
        o = self.observed
        return math.sqrt(max(o * (1.0 - o), 0.0) / self.count) if self.count else 0.0


@dataclass(frozen=True)
class Calibration:
    """The held-out table and the pre-registered verdict on it."""

    bins: tuple[Bin, ...]
    positions: int
    games: int
    mean_error: float
    brier: float
    brier_constant: float
    verdict: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class WinFit:
    """A fitted slope, the ruler it reads, and the basis and verdict that go with it."""

    k: float
    ruler: Ruler
    basis: Basis
    calibration: Calibration

    def chances(self, value: float) -> float:
        """The side to move's chances at a raw value (decided values are 1 or 0)."""
        if value >= self.ruler.decided_at:
            return 1.0
        if value <= -self.ruler.decided_at:
            return 0.0
        return chances(self.k, self.ruler.scale(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "basis": asdict(self.basis),
            "calibration": {
                **{
                    key: value
                    for key, value in asdict(self.calibration).items()
                    if key not in ("bins", "failures")
                },
                "failures": list(self.calibration.failures),
                "bins": [asdict(b) for b in self.calibration.bins],
            },
        }


def _labelled(
    readings: Iterable[GameReading], ruler: Ruler
) -> tuple[list[tuple[float, float]], int]:
    """(scaled value, result) for every undecided position, and how many were decided."""
    pairs: list[tuple[float, float]] = []
    decided = 0
    for reading in readings:
        for position in reading.positions:
            if abs(position.best) >= ruler.decided_at:
                decided += 1
                continue
            pairs.append((ruler.scale(position.best), reading.result_for(position.mover)))
    return pairs, decided


def fit_slope(pairs: Sequence[tuple[float, float]]) -> float:
    """The maximum-likelihood ``k`` in ``[0, K_MAX]`` on soft labels.

    The log-likelihood is concave in ``k``, so its derivative ``sum x (y - p(kx))`` falls
    as ``k`` rises and a bisection on its sign finds the maximum. No data, or data that
    say a larger value means *lower* chances, gives 0: every position at even chances.
    """

    def slope(k: float) -> float:
        return sum(x * (y - chances(k, x)) for x, y in pairs)

    if not pairs or slope(0.0) <= 0.0:
        return 0.0
    if slope(K_MAX) >= 0.0:
        return K_MAX
    low, high = 0.0, K_MAX
    for _ in range(100):
        middle = (low + high) / 2.0
        if slope(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def calibration(k: float, pairs: Sequence[tuple[float, float]], games: int) -> Calibration:
    """The held-out table, and `win-probability.md` §1's four clauses read on it."""
    sums = [[0, 0.0, 0.0] for _ in range(BINS)]
    brier = brier_constant = 0.0
    for x, y in pairs:
        p = chances(k, x)
        slot = sums[min(int(p * BINS), BINS - 1)]
        slot[0] += 1
        slot[1] += p
        slot[2] += y
        brier += (p - y) ** 2
        brier_constant += (0.5 - y) ** 2
    bins = tuple(
        Bin(
            low=i / BINS,
            high=(i + 1) / BINS,
            count=int(n),
            predicted=(p / n) if n else 0.0,
            observed=(o / n) if n else 0.0,
        )
        for i, (n, p, o) in enumerate(sums)
    )
    total = len(pairs)
    mean_error = sum(b.count * b.error for b in bins) / total if total else 0.0
    brier = brier / total if total else 0.0
    brier_constant = brier_constant / total if total else 0.0

    failures: list[str] = []
    if total < MIN_HELD_POSITIONS or games < MIN_HELD_GAMES:
        verdict = TOO_FEW
        failures.append(
            f"{total} held-out positions from {games} games; the bound needs "
            f"{MIN_HELD_POSITIONS} from {MIN_HELD_GAMES}"
        )
    else:
        if mean_error > MAX_MEAN_ERROR:
            failures.append(f"mean calibration error {mean_error:.3f} > {MAX_MEAN_ERROR}")
        for b in bins:
            if b.count >= BIN_FLOOR and b.error > BIN_SLACK + 2.0 * b.se:
                failures.append(
                    f"bin {b.low:.1f}-{b.high:.1f} misses by {b.error:.3f} "
                    f"> {BIN_SLACK} + 2 x {b.se:.3f}"
                )
        if not brier < brier_constant:
            failures.append(f"Brier {brier:.4f} is not below the constant's {brier_constant:.4f}")
        verdict = ROUGH if failures else CALIBRATED
    return Calibration(
        bins=bins,
        positions=total,
        games=games,
        mean_error=mean_error,
        brier=brier,
        brier_constant=brier_constant,
        verdict=verdict,
        failures=tuple(failures),
    )


def records_digest(records: Iterable[GameRecord]) -> str:
    """The part of each record a fit reads: its moves, result, seed and players."""
    canonical = [
        [record.seed, list(record.agents), record.moves, record.winner] for record in records
    ]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()[:16]


def fit(
    readings: Sequence[GameReading],
    ruler: Ruler,
    seed: int,
    ruleset_hash: str,
    records: str,
    positions_per_game: int | None = None,
) -> WinFit:
    """The slope on one half of the games, judged on the other. Readings and seed decide it."""
    fit_half = [r for r in readings if not held_out(seed, r)]
    held_half = [r for r in readings if held_out(seed, r)]
    fit_pairs, fit_decided = _labelled(fit_half, ruler)
    held_pairs, held_decided = _labelled(held_half, ruler)
    k = fit_slope(fit_pairs)
    basis = Basis(
        method=METHOD,
        ruleset_hash=ruleset_hash,
        seed=seed,
        records=records,
        fit_games=len(fit_half),
        fit_positions=len(fit_pairs),
        held_games=len(held_half),
        held_positions=len(held_pairs),
        decided=fit_decided + held_decided,
        positions_per_game=positions_per_game,
        ruler=asdict(ruler),
    )
    return WinFit(
        k=k, ruler=ruler, basis=basis, calibration=calibration(k, held_pairs, len(held_half))
    )


def fit_run(
    engine: Any,
    records: Sequence[GameRecord],
    seed: int = 0,
    positions_per_game: int | None = None,
    ruler: Ruler | None = None,
    scores: Any = None,
) -> WinFit:
    """Read a run's games with the core and fit them: the whole pipeline in one call."""
    ruler = ruler or ruler_for(engine)
    readings = read_run(engine, records, ruler, seed, positions_per_game, scores)
    ruleset_hash = records[0].ruleset_hash if records else ""
    return fit(readings, ruler, seed, ruleset_hash, records_digest(records), positions_per_game)


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
