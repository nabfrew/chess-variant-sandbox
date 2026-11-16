"""`decision_cost`: how many moves give up a tenth of a game (`v14`, R-S6b).

`docs/reference/v14-registration.md` §2.1 change 5, from R-S15b's registration
(`docs/reference/score-dials.md` §11.1). `decision_criticality` calls a move costly when the
probe values it a cheapest piece or more below the best: a threshold in built-in units, whose
worth in results differs between games by more than a factor of ten (§10, §11). This reading
keeps the same probe, the same sampled plies and the same ruler, and puts the values on a map
to expected score fitted to the game's own results:

    s(v) = sigmoid(v / k)

with one ``k`` (built-in units per logit) per game, fitted by least squares between
``s(v*)``, ``v*`` each sampled position's value (its best move's) for the side to move, and
that side's result (1, ½, 0), ``k`` searched on a log scale over ``K_RANGE`` by golden
section. A move is costly when ``s(best) - s(move) > DELTA``; the reading is one minus the
mean near-best share, as `decision_criticality` reads its own.

**Out of sample.** A library run is read with the ``k`` fitted on the game's other runs
(`cross_run_fits`, called by whoever holds the runs). A run read alone (a visitor's, or one
whose caller gave no fit) fits ``k`` on its games of even index and reads the odd ones, and
the reverse (`one_run_fit`). The fit sees positions' values and results only, never a move's
gap to the best.

**Withheld.** Fewer than ``MIN_POSITIONS`` sampled positions in what a ``k`` is fitted on, or
a fit at either edge of ``K_RANGE``: the dial is absent from the run and the reason is in its
diagnostics. `v14` marks the dial optional, so its family is scored on the rest.

Nothing here touches an engine: it reads the values the analysis pass persisted
(`GameAnalysis.sample_values`, written under `analyse_corpus(readings=True)`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DELTA = 0.10
K_RANGE = (0.01, 1000.0)
TOLERANCE = 1e-4
MIN_POSITIONS = 100
# A fit this close to an end of the range (relative) is a fit at the edge.
EDGE = 1e-3

GameKey = tuple[int, tuple[str, str]]
Point = tuple[float, float]


def expected(value: float, k: float) -> float:
    """``sigmoid(value / k)``, exact at the win scores."""
    x = value / k
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def fit_scale(points: Sequence[Point], tolerance: float = TOLERANCE) -> float:
    """The ``k`` minimising the squared error of ``sigmoid(v / k)`` against ``y``, by golden
    section on ``log k`` over `K_RANGE`, to ``tolerance`` (R-S15b's fit, to the digit)."""

    def loss(log_k: float) -> float:
        k = math.exp(log_k)
        total = 0.0
        for value, result in points:
            total += (expected(value, k) - result) ** 2
        return total

    lo, hi = math.log(K_RANGE[0]), math.log(K_RANGE[1])
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fc, fd = loss(c), loss(d)
    while hi - lo > tolerance:
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi - golden * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + golden * (hi - lo)
            fd = loss(d)
    return math.exp((lo + hi) / 2.0)


def at_edge(k: float) -> bool:
    return k <= K_RANGE[0] * (1.0 + EDGE) or k >= K_RANGE[1] * (1.0 - EDGE)


@dataclass(frozen=True)
class CostFit:
    """The scale each game of a run is read with, or why there is none.

    ``scales`` maps each game (seed, agents) to its ``k``; empty when withheld, and then
    ``why`` says why. ``positions`` is how many sampled positions the fit (or the smaller of
    the two halves' fits) rested on.
    """

    scales: dict[GameKey, float] = field(default_factory=dict)
    why: str = ""
    positions: int = 0
    rule: str = ""

    @property
    def withheld(self) -> bool:
        return not self.scales

    def detail(self) -> dict[str, Any]:
        ks = sorted(set(self.scales.values()))
        return {
            "rule": self.rule,
            "k": [round(k, 4) for k in ks],
            "positions": self.positions,
            **({"withheld": self.why} if self.withheld else {}),
        }


def _results(records: Sequence[Any]) -> dict[GameKey, int | None]:
    return {(r.seed, tuple(r.agents)): r.winner for r in records}


def _result(mover: int, winner: int | None) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == mover else 0.0


def written(analysis: Any) -> bool:
    """Whether every game of the run carries the values (`analyse_corpus(readings=True)`)."""
    games = list(analysis.games)
    return bool(games) and all(g.sample_values is not None for g in games)


def game_points(analysis: Any, records: Sequence[Any]) -> list[tuple[GameKey, list[Point]]]:
    """Each analysed game's (value of the position, result for its mover) points."""
    results = _results(records)
    out: list[tuple[GameKey, list[Point]]] = []
    for game in analysis.games:
        key = (game.seed, tuple(game.agents))
        winner = results.get(key)
        points = [
            (max(values), _result(mover, winner))
            for values, mover in zip(game.sample_values or [], game.sample_movers or [],
                                     strict=True)
            if values
        ]
        out.append((key, points))
    return out


def _fit(points: list[Point]) -> tuple[float | None, str]:
    if len(points) < MIN_POSITIONS:
        return None, f"fewer than {MIN_POSITIONS} sampled positions ({len(points)}) to fit on"
    k = fit_scale(points)
    if at_edge(k):
        return None, f"the fit is at an edge of the range ({k:.4g})"
    return k, ""


def one_run_fit(analysis: Any, records: Sequence[Any]) -> CostFit:
    """A run read alone: the even games' fit reads the odd games, and the reverse."""
    games = game_points(analysis, records)
    halves = [[p for i, (_k, pts) in enumerate(games) if i % 2 == parity for p in pts]
              for parity in (0, 1)]
    fits = [_fit(half) for half in halves]
    positions = min(len(half) for half in halves)
    for k, why in fits:
        if k is None:
            return CostFit(why=why, positions=positions, rule="one run: even and odd games")
    k_even, k_odd = fits[0][0], fits[1][0]
    assert k_even is not None and k_odd is not None
    scales = {key: (k_odd if i % 2 == 0 else k_even) for i, (key, _p) in enumerate(games)}
    return CostFit(scales=scales, positions=positions, rule="one run: even and odd games")


def others_fit(analysis: Any, others: Sequence[tuple[Any, Sequence[Any]]]) -> CostFit:
    """One run's fit from the game's other runs (each run's analysis and records), as
    `cross_run_fits` gives it for that run: a run without values adds no points."""
    return cross_run_fits([(analysis, []), *others])[0]


def cross_run_fits(runs: Sequence[tuple[Any, Sequence[Any]]]) -> list[CostFit]:
    """Each run's fit from the game's other runs (``runs``: each run's analysis and records).

    A run with no values written gets a withheld fit; it adds no points to the others'.
    """
    points = [
        [p for _key, pts in game_points(a, r) for p in pts] if written(a) else []
        for a, r in runs
    ]
    fits = []
    for index, (analysis, _records) in enumerate(runs):
        others = [p for j, pts in enumerate(points) if j != index for p in pts]
        rule = f"the game's other {len(runs) - 1} runs"
        k, why = _fit(others)
        if k is None or not written(analysis):
            fits.append(CostFit(why=why or "this run's values are not written",
                                positions=len(others), rule=rule))
            continue
        scales = {(g.seed, tuple(g.agents)): k for g in analysis.games}
        fits.append(CostFit(scales=scales, positions=len(others), rule=rule))
    return fits


def reading(analysis: Any, records: Sequence[Any]) -> tuple[float | None, dict[str, Any]]:
    """The run's `decision_cost` and its detail, or ``None`` where it is not written or
    withheld. A run whose caller set no ``cost_fit`` is fitted by `one_run_fit`, once, and
    the fit kept on the analysis so a resample reads every game with the same ``k``."""
    if not written(analysis):
        return None, {"withheld": "the analysis pass did not write the move values"}
    fit = analysis.cost_fit
    if fit is None:
        fit = one_run_fit(analysis, records)
        analysis.cost_fit = fit
    detail = fit.detail()
    if fit.withheld:
        return None, detail
    fractions: list[float] = []
    for game in analysis.games:
        k = fit.scales.get((game.seed, tuple(game.agents)))
        if k is None:
            return None, {**detail, "withheld": "a game has no scale"}
        for values in game.sample_values or []:
            if len(values) < 2:
                continue
            scores = [expected(v, k) for v in values]
            best = max(scores)
            near = sum(1 for s in scores if best - s <= DELTA)
            fractions.append(near / len(values))
    if not fractions:
        return None, {**detail, "withheld": "no sampled position with a choice"}
    detail["samples"] = len(fractions)
    return 1.0 - sum(fractions) / len(fractions), detail
