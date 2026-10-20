"""The heuristic pass: what a rule of thumb can do in this game, and what depth adds.

A third pass over a ruleset, beside the tournament and the analysis pass, and it answers
a question neither of those can: **if a shallow heuristic is unbeatable, the game is
probably not great.**

The obvious experiment -- a heuristic player against `minimax:nodes=6400` -- is
confounded, and `docs/plan.md` stage 17 documents why at length: the built-in evaluator
reads material and mobility and nothing else, so on a game won by arrival, by territory
or by shedding material a decent heuristic wins because the *search* is blind, not
because the game is shallow. The comparison here therefore holds the objective fixed and
varies only depth:

    greedy:h=H   (one ply)   against   minimax:nodes=N,h=H   (deep)  -- the same H.

If thinking harder about the same objective buys nothing, the game is shallow. That is
the measurement the idea actually wants, and it is the only form that survives contact
with an evaluator that cannot see the win condition.

**These games are a separate corpus and never enter the ratings.** Heuristic players are
per-ruleset by construction -- the normalisation divides by board size, so the same
digest is not the same player on two boards -- while every skill component reads a
*fixed* ladder whose comparability across rulesets is the basis for the index being a
scale at all. Keeping the corpora apart is the guarantee; `metrics/components.py` also
filters the names, but that is defence in depth rather than the argument, because
`rating_spread` reads the whole table and no name filter can reach it.

**The depth control is not optional.** On the 9x9 RPS boards `minimax:nodes=6400` reaches
depth 2, so a "deep" arm there is nearly a one-ply player and headroom would read zero
for a reason that has nothing to do with the game. `N` is raised per ruleset until the
search actually gets somewhere, and a ruleset where it cannot is reported **unmeasured**
rather than shallow -- the same distinction `index.withheld_components` already draws
between a bad game and the wrong instrument.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.features import VOCABULARY_VERSION
from chess_variant_sandbox.agents.heuristics import Heuristic, HeuristicSet
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

# The three diagnostics and their control. Named here so the glossary's two-way tests
# can union them in: a synthetic corpus never reaches this pass, so a component it
# produces would otherwise read as "explained but never produced".
HEURISTIC_COMPONENTS = (
    "heuristic_headroom",
    "heuristic_agreement",
    "heuristic_depth_reached",
)

# The shallow arm is a fixed **depth**, not a fixed node budget, and this is the one place
# in the project where that is the right choice.
#
# Budget is nodes and never depth for *rated* agents, because a node count means the same
# amount of thinking on a board of any branching factor and a depth does not. The shallow
# arm is not a rated player: it is a yardstick, and a yardstick has to be the same length
# everywhere. Measured at a fixed 200 nodes it was 1.0 plies on chess and 3.5 on `dull`,
# which compares different things and calls the difference a property of the game.
#
# Two plies rather than one. A one-ply arm cannot see the opponent's reply at all, which
# makes it a worse model of somebody following a rule of thumb than of anything else, and
# it saturated the measurement: at one ply, twenty-four games per matchup still left four
# to five of every six heuristics at exactly 1.000. See `docs/reference/heuristic-players.md`.
SHALLOW_DEPTH = 2
# Generous, and capped by the depth: iterative deepening stops at `SHALLOW_DEPTH`, so this
# only has to be enough to complete it. Verified at exactly 2.00 plies on every committed
# ruleset, the widest board included.
SHALLOW_NODES = 20_000

# Budgets tried in order until the deep arm clears the shallow one by `DEPTH_MARGIN`.
# Stops at the first that does, so a small board pays nothing for a big board's problem.
NODE_LADDER = (6_400, 25_000, 100_000)

# How much deeper the deep arm has to be than the shallow one before the comparison
# means anything.
#
# **Relative, not absolute, and getting that wrong cost a whole corpus run.** The control
# started as "at least three plies" while the shallow arm was one, which was a two-ply
# margin. Making the shallow arm two plies -- stage 21, and the right fix for saturation --
# silently halved the margin to one, and `crazyhouse` promptly read headroom 0.000 at a
# deep-arm depth of 3.0. That is not a shallow game; it is a game where the deep arm was
# barely deeper than the thing it was being compared against. A margin expressed against
# the shallow arm cannot drift out of step with it.
DEPTH_MARGIN = 2.0
MIN_DEPTH = SHALLOW_DEPTH + DEPTH_MARGIN

# How far into the game the depth control looks. Long enough to leave the opening, short
# enough that the probe costs a fraction of one game.
DEPTH_PLIES = 12

# Games per ordered pairing. Both colours are always played, so this is half the games
# in a matchup.
MATCH_GAMES = 3



@dataclass
class MatchRow:
    """One heuristic, and how it did in each of the three comparisons."""

    name: str
    digest: str
    plan: str
    # The *deep* arm's score against the shallow arm, both using this heuristic.
    deep_score: float
    # This heuristic at one ply against a random mover.
    legibility: float
    # This heuristic against the built-in evaluator, at one ply and at the deep budget.
    agreement_shallow: float
    agreement_deep: float

    @property
    def headroom(self) -> float:
        """What depth over the same objective bought, on ``[0, 1]``.

        ``0.5`` is "depth added nothing" and maps to zero; a deep arm that wins every
        game maps to one. Losing to the shallow arm also maps to zero rather than going
        negative -- a search that is *worse* than one ply of the same objective is a
        statement about the search, and the floor keeps it from reading as a property of
        the game.
        """
        return max(0.0, min(1.0, 2.0 * (self.deep_score - 0.5)))

    @property
    def agreement(self) -> float:
        """Conservative on purpose: the heuristic must beat the built-in at both depths."""
        return min(self.agreement_shallow, self.agreement_deep)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "digest": self.digest,
            "plan": self.plan,
            "deep_score": self.deep_score,
            "legibility": self.legibility,
            "agreement_shallow": self.agreement_shallow,
            "agreement_deep": self.agreement_deep,
        }
        data["headroom"] = round(self.headroom, 4)
        data["agreement"] = round(self.agreement, 4)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchRow:
        return cls(
            name=str(data["name"]),
            digest=str(data["digest"]),
            plan=str(data.get("plan", "")),
            deep_score=float(data["deep_score"]),
            legibility=float(data["legibility"]),
            agreement_shallow=float(data["agreement_shallow"]),
            agreement_deep=float(data["agreement_deep"]),
        )


@dataclass
class HeuristicReport:
    """Everything the heuristic pass found for one ruleset."""

    ruleset_hash: str
    set_hash: str
    nodes: int
    depth_reached: float
    rows: list[MatchRow] = field(default_factory=list)
    vocabulary_version: int = VOCABULARY_VERSION
    games_per_pairing: int = MATCH_GAMES

    @property
    def k(self) -> int:
        return len(self.rows)

    @property
    def measurable(self) -> bool:
        """Whether the deep arm ever got deep enough for headroom to mean anything."""
        return self.depth_reached >= MIN_DEPTH

    def as_dict(self) -> dict[str, Any]:
        return {
            "ruleset_hash": self.ruleset_hash,
            "set_hash": self.set_hash,
            "vocabulary_version": self.vocabulary_version,
            "nodes": self.nodes,
            "depth_reached": self.depth_reached,
            "games_per_pairing": self.games_per_pairing,
            "k": self.k,
            "measurable": self.measurable,
            "rows": [row.as_dict() for row in self.rows],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeuristicReport:
        return cls(
            ruleset_hash=str(data["ruleset_hash"]),
            set_hash=str(data["set_hash"]),
            nodes=int(data["nodes"]),
            depth_reached=float(data["depth_reached"]),
            rows=[MatchRow.from_dict(row) for row in data.get("rows", [])],
            vocabulary_version=int(data.get("vocabulary_version", VOCABULARY_VERSION)),
            games_per_pairing=int(data.get("games_per_pairing", MATCH_GAMES)),
        )

    @classmethod
    def load(cls, path: str | Path) -> HeuristicReport:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def heuristic_components(report: HeuristicReport | None) -> dict[str, float]:
    """The raw measurements this pass contributes. Absent is reported as absent.

    Three reporting rules are part of the definitions rather than of the presentation:

    * **The median is the headline, not the max.** Best-of-k is biased upward in k, so a
      set of six and a set of twelve would not be the same quantity. The max travels in
      the diagnostics with k printed beside it.
    * **A ruleset the deep arm could not search is unmeasured**, not shallow.
    * **Index 0 is the built-in objective**, never proposed by anything, so
      ``heuristic_headroom`` always has one anchor that owes nothing to a sample.
    * **``H*`` is the plan that plays best against the built-in objective**, not the one
      that beats a random mover hardest. The second is what it used to be, and it
      saturated: nearly every vector beats a random mover, including vectors drawn at
      random, so the choice was close to arbitrary.
    """
    if report is None or not report.rows:
        return {}
    if not report.measurable:
        # The instrument, not the game. Reported so a reader can tell the two apart.
        return {"heuristic_depth_reached": round(report.depth_reached, 3)}

    headrooms = [row.headroom for row in report.rows]
    best = max(report.rows, key=lambda row: row.agreement_shallow)
    return {
        "heuristic_headroom": round(statistics.median(headrooms), 4),
        "heuristic_agreement": round(best.agreement, 4),
        "heuristic_depth_reached": round(report.depth_reached, 3),
    }


def heuristic_diagnostics(report: HeuristicReport | None) -> dict[str, Any]:
    """What the components deliberately leave out, for a reader rather than a curve."""
    if report is None or not report.rows:
        return {}
    headrooms = [row.headroom for row in report.rows]
    return {
        "k": report.k,
        "set_hash": report.set_hash,
        "nodes": report.nodes,
        "measurable": report.measurable,
        "headroom_best": round(max(headrooms), 4),
        "headroom_worst": round(min(headrooms), 4),
        # Index 0 is the built-in objective by convention, so this is the one point in
        # the set that no sample produced.
        "headroom_builtin": round(headrooms[0], 4),
        "best_plan": max(report.rows, key=lambda row: row.agreement_shallow).plan,
        # Kept as a diagnostic and no longer a component: "does this plan beat a random
        # mover" saturates at 1.000 on nearly every real game, and the version that does
        # not saturate is `agreement` measured a second time. See the report.
        "beats_random": round(max(row.legibility for row in report.rows), 4),
        # Risk 3's control: if headroom is high only where the heuristic cannot beat a
        # random mover, headroom is measuring search pathology rather than the game.
        "headroom_legibility_correlation": _correlation(
            headrooms, [row.legibility for row in report.rows]
        ),
    }


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3:
        return None
    try:
        return round(statistics.correlation(first, second), 3)
    except statistics.StatisticsError:
        # Constant on one side, which is a real answer rather than an error: nothing
        # varied, so nothing can covary.
        return None


def measure_heuristics(
    spec: dict[str, Any],
    heuristic_set: HeuristicSet,
    seed: int = 0,
    games: int = MATCH_GAMES,
    nodes: int | None = None,
    workers: int = 0,
) -> HeuristicReport:
    """Play the three comparisons for every heuristic in the set, against the built-in
    evaluator."""
    engine = Engine(spec)
    heuristic_set.register()

    budget = nodes if nodes is not None else _choose_budget(engine, heuristic_set, seed)
    depth = _depth_reached(engine, heuristic_set.heuristics[0], budget, seed)

    base_shallow = _shallow(None)
    base_deep = f"minimax:nodes={budget}"

    rows = []
    for heuristic in heuristic_set.heuristics:
        digest = heuristic.digest
        shallow = _shallow(digest)
        deep = f"minimax:h={digest},nodes={budget}"
        rows.append(
            MatchRow(
                name=heuristic.name,
                digest=digest,
                plan=heuristic.plan,
                deep_score=head_to_head(spec, deep, shallow, games, seed, workers),
                legibility=head_to_head(spec, shallow, "random", games, seed + 1_000, workers),
                agreement_shallow=head_to_head(
                    spec, shallow, base_shallow, games, seed + 2_000, workers
                ),
                agreement_deep=head_to_head(
                    spec, deep, base_deep, games, seed + 3_000, workers
                ),
            )
        )

    return HeuristicReport(
        ruleset_hash=engine.spec.hash,
        set_hash=heuristic_set.set_hash,
        nodes=budget,
        depth_reached=depth,
        rows=rows,
        games_per_pairing=games,
    )


def _shallow(digest: str | None) -> str:
    """The shallow arm: a fixed two plies, with or without a heuristic objective.

    ``None`` gives the same arm on the built-in evaluator, which is what
    ``heuristic_agreement`` compares against -- so the two sides of that comparison
    differ only in what they are maximising, never in how far they look.
    """
    prefix = f"h={digest}," if digest is not None else ""
    return f"minimax:{prefix}max_depth={SHALLOW_DEPTH},nodes={SHALLOW_NODES}"


def head_to_head(
    spec: dict[str, Any], first: str, second: str, games: int, seed: int, workers: int
) -> float:
    """``first``'s score, both colours, so a seat advantage cannot be mistaken for skill.

    Public because every acceptance test asks the same question of the same shape --
    does this objective beat the one it replaces, at equal budget -- and two definitions
    of "score, both colours" would be two ways to disagree about what a match result is.

    Run through the ordinary tournament runner rather than a loop over ``play_game``,
    which is not a detail: the runner schedules every game's seed before any thread
    starts, so the answer is identical at any worker count, and the games then run in
    parallel in the core. Serially this pass cost about seventeen minutes per ruleset,
    which is more than the tournament it sits beside.
    """
    result = run_tournament(
        spec,
        TournamentConfig(
            agents=(first, second),
            games_per_pairing=games,
            seed=seed,
            workers=workers,
            include_self_play=False,
            collect_series=False,
        ),
    )
    total = 0.0
    played = 0
    for record in result.records:
        if record.agents[0] == first:
            total += record.score_for(0)
        else:
            total += record.score_for(1)
        played += 1
    return round(total / played, 4) if played else 0.5


def _choose_budget(engine: Engine, heuristic_set: HeuristicSet, seed: int) -> int:
    """The smallest budget on the ladder at which the deep arm is actually deep."""
    anchor = heuristic_set.heuristics[0]
    budget = NODE_LADDER[0]
    for candidate in NODE_LADDER:
        budget = candidate
        if _depth_reached(engine, anchor, candidate, seed) >= MIN_DEPTH:
            break
    return budget


def _depth_reached(engine: Engine, heuristic: Heuristic, nodes: int, seed: int) -> float:
    """The deepest iteration the deep arm completes, averaged over the opening."""
    return plies_completed(
        engine, f"minimax:h={heuristic.digest},nodes={nodes},mobility=0", seed
    )


def plies_completed(engine: Engine, agent: str, seed: int) -> float:
    """How deep ``agent`` gets per search, averaged over the opening of one game.

    Read out of the core rather than out of the Python oracle. The two agree move for
    move, so either would give the same number -- but the Python search is sixty to a
    hundred times slower, and this probe runs once per budget on the ladder. Doing it in
    Python cost minutes per ruleset and made the whole pass unusable, which is the
    ordinary reason this project puts a whole unit of work behind the boundary.

    Public and taking an agent *string* because the searched valuation wants the same
    number for a plain ladder rung: two probes that answered "how deep does this rung
    see" slightly differently would be two answers to one question, and this pass's
    depth control and `docs/reference/search-depth.md` would stop being comparable.
    """
    from chess_variant_sandbox.agents import native_spec
    from chess_variant_sandbox.tournament.play import eval_values

    native = native_spec(agent, engine.spec.hash)
    if native is None:  # pragma: no cover - minimax is always native
        return 0.0

    values = eval_values(engine.spec)
    state = engine.initial_state()
    depths: list[int] = []
    for _ in range(DEPTH_PLIES):
        moves = engine.legal_moves(state)
        if not moves or engine.terminal(state, moves) is not None:
            break
        move, depth = engine.select_move_depth(state, native, values, seed)
        if move is None:
            break
        depths.append(depth)
        state = engine.apply(state, move)
    return round(sum(depths) / len(depths), 3) if depths else 0.0
