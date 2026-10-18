"""Raw measurements of a ruleset, computed from a tournament corpus.

Nothing here is normalised or weighted -- these are facts about what happened, and
turning facts into a judgement is the next module's job. Keeping the split sharp is
what lets the index be re-scored under new weights without re-simulating anything.

Two conventions run through the file:

**Draws and decisiveness are measured between strong players only.** Two random movers
draw constantly in any game; counting those would say more about the agents than the
rules. Where a measurement only means something at competence, it filters first.

**Self-play carries the balance measurements.** An agent against itself is the only
matchup where strength cancels exactly, so first-player advantage read from anywhere
else is contaminated by who happened to hold which seat.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import compress, repeat
from typing import Any, Generic, TypeVar

from chess_variant_sandbox.engine.compile import (
    MODE_CAPTURE,
    MODE_MOVE,
    CompiledSpec,
    compile_spec,
)
from chess_variant_sandbox.metrics.seat import (
    SEAT_DECIDED_CHANCE,
    TOP_RUNG_FIRST_SEAT_SCORE,
    TOP_RUNG_SEAT_DETAIL,
    TOP_THREE_FIRST_SEAT_SCORE,
    TOP_THREE_SEAT_DECIDED_CHANCE,
    TOP_THREE_SEAT_DETAIL,
    sign_test,
    transcript_digest,
)
from chess_variant_sandbox.spec.canonical import canonical_json
from chess_variant_sandbox.spec.complexity import SpecComplexity, measure_complexity
from chess_variant_sandbox.spec.reachability import pieces_a_player_can_hold
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
from chess_variant_sandbox.tournament.heuristics import (
    HeuristicReport,
    heuristic_components,
    heuristic_diagnostics,
)
from chess_variant_sandbox.tournament.ratings import RatingTable
from chess_variant_sandbox.tournament.record import GameRecord

OPENING_PREFIX_PLIES = 6
STALL_REASONS = frozenset({"ply_limit", "no_progress"})
# Named rather than inlined because the pooled rate and the worst-half rate have to be
# the same measurement on two denominators. Two literal 12s would be two measurements
# that happen to agree.
QUICK_KILL_PLIES = 12
_NODE_PATTERN = re.compile(r"nodes=(\d+)")
_HEURISTIC_PATTERN = re.compile(r"(^|[:,])h=")

# The four rates the outcome gates read, and which direction is *worse* for each. A
# ruleset that is two different games -- an opening trap against players too weak to see
# it, a dead shuffle against players who do -- passes every one of these pooled, because
# the trap games supply the decisiveness that rescues the stall gate and the grind games
# supply the length that rescues the quick-kill gate. Each failure is the other one's
# alibi. So the gates read the worse half of the budget ladder instead of the average of
# the two, under the `_worst_half` keys below.
#
# The direction is stated here rather than read from a weights file because the half is
# chosen before any weights are loaded, and because it is a property of the measurement
# and not of a threshold: quick kills and stalls are bad when high, decisiveness and
# length when low. A key whose direction disagreed with its gate would report the
# *safer* half, and the split gate would be weaker than the pooled one it replaced --
# which is the opposite of the point. `tests/test_metrics.py` holds the two together.
WORST_HALF_SUFFIX = "_worst_half"
WORST_HALF_RATES: dict[str, str] = {
    "quick_kill_rate": "high",
    "stall_rate": "high",
    "decisive_rate": "low",
    "mean_plies": "low",
}

# How each component a weights version scores reads a game played in rounds (schema 26,
# Q918d; `docs/archive/plan-simultaneous.md` §6.3). A round game's record keeps both moves of a
# round in seat order and every per-ply series twice, so a round is two plies and most
# components read it as they read an alternating game; the two the analysis pass supplies
# from a probe read each player's moves against the opponent's equilibrium mix. Every
# scored component has its entry (`tests/test_rounds_measured.py` holds the current
# version to it): one that could not read rounds would say so here, with its reason, and
# be withheld rather than scored.
ROUND_READINGS: dict[str, str] = {
    "first_player_advantage": (
        "seat 0's self-play score; nobody moves first, so a lean is the setup's "
        "asymmetry or the initiative's edge"
    ),
    "stall_rate": "how games end, as in any game",
    "mean_branching": (
        "per player: a round records each player's own legal moves, one per ply, never "
        "the matrix's size"
    ),
    "ply_spread": "in plies, two a round, so a length compares with an alternating game's",
    "position_divergence": "the position after each round, written once per ply",
    "decision_criticality": (
        "each player's moves valued against the opponent's equilibrium mix one round "
        "deep, near the best within the cheapest piece; each seat with a choice is a sample"
    ),
    "blunder_recoverability": (
        "the move forced in falls short of the best against the opponent's equilibrium "
        "mix by more than the cheapest piece; the other seat keeps its recorded move"
    ),
    "piece_utilisation": "both moves of every round, read on the board both chose on",
    "instance_participation": "both moves of every round",
    "skill_gradient": "the round players' ladder, matrix:nodes=1600 to 102400",
    "lead_conversion": "seat 0's material after each round, written once per ply",
}


def plays_in_rounds(spec: dict[str, Any]) -> bool:
    """Whether a spec's game is played in rounds (schema 26, ``rules.simultaneous``).

    True of a spy game too (schema 27), which declares the rule and plays rounds once a spy
    is lost; `plays_with_spies` says which."""
    rules = spec.get("rules")
    return isinstance(rules, dict) and rules.get("simultaneous") is not None


# How each scored component reads a spy game (schema 27, Q935d; `docs/plan-spy.md` §3.8),
# which alternates while both players have a spy and plays rounds once one is lost. Its
# record keeps a turn's move and a round's two in seat order, and every per-ply series once
# a turn and twice a round, so a round is two plies as in a round game and every component
# reads the mixed record as it reads either kind. The probe reads each step by what its
# players knew. Every scored component has its entry (`tests/test_spy_measured.py`), and a
# spy game's score carries these in place of `ROUND_READINGS`.
SPY_READINGS: dict[str, str] = {
    "first_player_advantage": (
        "seat 0's self-play score; seat 0 moves first while both spies stand, and who keeps "
        "a spy is decided by play"
    ),
    "stall_rate": "how games end, as in any game",
    "mean_branching": (
        "the mover's legal moves on a turn, each player's own on a round, one per ply"
    ),
    "ply_spread": "in plies: one a turn, two a round",
    "position_divergence": "the position after each step, written once per ply",
    "decision_criticality": (
        "a turn's moves as an alternating game's, the probe stopping at a round; an "
        "announced round's moves one round deep, the announcer's by their worst answer and "
        "the answerer's against the move announced; a blind round's against the "
        "opponent's equilibrium mix"
    ),
    "blunder_recoverability": (
        "a turn takes a random other move; a round's forced move falls short of the best "
        "by the reading decision_criticality uses, and an answerer answers a forced "
        "announcement as its player would"
    ),
    "piece_utilisation": "every move of every step, read on the board it was chosen on",
    "instance_participation": "every move of every step",
    "skill_gradient": "minimax:nodes=100 to 6400 on the built-in objective, every phase",
    "lead_conversion": "seat 0's material before each step, written once per ply",
}


def plays_with_spies(spec: dict[str, Any]) -> bool:
    """Whether a spec is a spy game (schema 27): some piece type is flagged ``"spy": true``."""
    pieces = spec.get("pieces")
    return isinstance(pieces, list) and any(
        isinstance(piece, dict) and piece.get("spy") is True for piece in pieces
    )


# A piece type moved fewer times than this share of all moves is, for practical
# purposes, not in the game. Used only as a diagnostic next to the entropy measure.
IDLE_PIECE_SHARE = 0.02

# Where ``lead_conversion`` reads the material edge. The same fraction the blunder
# replay uses, and for the same reason: far enough in that the position has structure,
# far enough from the end that there is still a game to convert.
LEAD_FRACTION = 0.34
# How big an edge has to be before it counts as a lead at all. One point of material,
# which in a spec-derived value scale is roughly the cheapest piece.
LEAD_EPSILON = 1.0
# Below this many qualifying games the answer is noise. A corpus of draws has nothing
# to say about whether a lead converts, and 0.0 would say the opposite of nothing.
MIN_LEAD_SAMPLE = 8


@dataclass(frozen=True)
class SpecFacts:
    """The measurements that read the rules and never the games.

    Split out because they cannot move between resamples and were being recomputed
    with every one of them. ``bootstrap_index`` calls ``compute_components`` 200 times
    over corpora drawn from the same tournament; on chess these three cost 116.7 ms of
    each of those calls -- roughly 23 seconds per ruleset scored, spent re-deriving an
    unchanging fact about the spec, against a whole bootstrap of 41.8 s.

    Both numbers are the state *before* this class existed and neither is current: the
    same stage measured 2.7 s a run over 53 canonical runs on 2026-09-06 (Q48). They are
    kept because they are what made the memoisation worth having, not as a costing.

    Memoised on the spec's canonical JSON by ``spec_facts``, so the numbers are
    identical by construction rather than by agreement.
    """

    complexity: SpecComplexity
    threat_asymmetry: float
    traversal_time: float


# Small on purpose. The access pattern is one ruleset measured hundreds of times in a
# row -- a bootstrap, a component sweep, one generation of the design loop -- rather
# than a working set, and a design run compiles thousands of distinct specs in one
# process, so an unbounded cache would hold every proposal's canonical JSON for the
# lifetime of the run.
_FACTS_CACHE_SIZE = 8
_FACTS: OrderedDict[str, SpecFacts] = OrderedDict()


def spec_facts(spec: dict[str, Any]) -> SpecFacts:
    """Everything about a ruleset that a corpus of games cannot change.

    Keyed on the *whole* canonical JSON rather than on the mechanics hash: two specs
    differing only in their description are the same game and hash the same, and
    ``description_words`` is a complexity count that reads the description.
    """
    key = canonical_json(spec)
    cached = _FACTS.get(key)
    if cached is not None:
        _FACTS.move_to_end(key)
        return cached

    # Compiled once and shared. These used to compile the spec four times between them
    # -- twice inside ``_displacements``, once for ``capture_allowed``, once more for
    # the board -- at 29-33 ms each.
    compiled = compile_spec(spec)
    facts = SpecFacts(
        complexity=measure_complexity(spec),
        threat_asymmetry=_threat_asymmetry(compiled),
        traversal_time=_traversal_time(compiled),
    )
    _FACTS[key] = facts
    if len(_FACTS) > _FACTS_CACHE_SIZE:
        _FACTS.popitem(last=False)
    return facts


@dataclass(frozen=True)
class GameReading:
    """What one game says about itself alone, whatever corpus it is read in."""

    lead_changes: float
    volatility: float
    irreversibility: float


class GameReadings:
    """Each game's own reading, taken once and kept by record identity.

    The bootstrap measures 200 corpora drawn from one list of records, and a resample
    holds the *same* record objects, so a game's own numbers are the same on every draw.
    Keyed on ``id()``, with the record held beside its reading so that the id cannot be
    reused while the memo lives. Only the per-game numbers are kept: every mean is still
    taken over the corpus in hand, in its order, so every sum adds the same terms in the
    same order as reading each game afresh.
    """

    def __init__(self) -> None:
        self._held: dict[int, tuple[GameRecord, GameReading]] = {}

    def of(self, record: GameRecord) -> GameReading:
        held = self._held.get(id(record))
        if held is None:
            reading = GameReading(
                lead_changes=_lead_changes(record),
                volatility=_volatility(record),
                irreversibility=_irreversibility(record),
            )
            held = self._held[id(record)] = (record, reading)
        return held[1]


@dataclass
class Components:
    """Every raw measurement for one ruleset, plus context for reading them."""

    raw: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.raw[key]

    def as_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "diagnostics": self.diagnostics}


def compute_components(
    spec: dict[str, Any],
    records: list[GameRecord],
    ratings: RatingTable,
    analysis: CorpusAnalysis | None = None,
    heuristics: HeuristicReport | None = None,
    readings: GameReadings | None = None,
) -> Components:
    """Measure a ruleset from its spec and the corpus its tournament produced.

    ``analysis`` carries the engine-derived second pass. It is optional because a
    corpus measured before that pass existed is still a valid corpus: the components
    that need it are simply absent, and any weights version that asks for them will
    say so rather than quietly scoring a missing number as zero.

    ``heuristics`` carries the third pass, and is optional for a stronger reason: it
    needs a heuristic store, which is generated once per ruleset and committed. A
    corpus without one produces none of its measurements, and none of them is scored by
    any weights version -- they sit beside ``ladder_monotonicity`` as statements about
    what a rule of thumb can do here, and about whether the instrument could read the
    game at all.

    ``readings`` holds what each game says about itself alone. A caller measuring many
    corpora drawn from the same records passes one, so each game is read once;
    ``bootstrap_index`` does.
    """
    if not records:
        raise ValueError("cannot measure a ruleset with no games")

    strong = [r for r in records if not _is_weak(r.agents[0]) and not _is_weak(r.agents[1])]
    graded = strong or records
    self_play = [r for r in records if r.agents[0] == r.agents[1]]

    raw: dict[str, float] = {}

    # --- how much rule is there -------------------------------------------------
    facts = spec_facts(spec)
    complexity = facts.complexity
    raw["rule_mass"] = complexity.rule_mass
    raw["piece_types"] = float(complexity.piece_types)
    raw["special_rules"] = float(complexity.special_rules)

    # --- does the game finish, and how ------------------------------------------
    raw.update(_outcome_rates(graded))
    raw["ply_spread"] = _spread([float(r.plies) for r in graded])

    # The same four rates on the worse half of the budget ladder. Scored by nothing --
    # ``stall_rate`` keeps its curve on the pooled value, because moving a scored
    # component's denominator would move its curve and this version anchors nothing --
    # and read only by the gates, which is where a pooled average over a ladder that is
    # playing two different games does its damage.
    worst_half, split_detail = _worst_half_rates(graded)
    raw.update(worst_half)

    # --- is the top of the ladder decided by the seat -----------------------------
    # Read by the seat gates and by nothing scored. The balance figure below reads 32
    # self-play games across the whole ladder and cannot see a seat effect that appears
    # only above a budget; these read the games among the strongest rungs.
    seat_raw, seat_details = _seat_readings(graded)
    raw.update(seat_raw)

    # --- how wide are the choices -----------------------------------------------
    branching = [b for r in graded for b in r.branching]
    raw["mean_branching"] = _mean([float(b) for b in branching]) if branching else 0.0

    # --- balance, from self-play only -------------------------------------------
    balance_source = self_play or graded
    first_seat_score = _mean([r.score_for(0) for r in balance_source])
    raw["first_player_score"] = first_seat_score
    raw["first_player_advantage"] = abs(first_seat_score - 0.5) * 2.0

    # --- repetitiveness ----------------------------------------------------------
    # Measured on competent games only. A random mover produces endless variety in
    # any ruleset whatsoever, so including those games would score the agent's dice
    # rather than the game's capacity to be played differently.
    raw["opening_diversity"] = _opening_diversity(graded)
    raw["distinct_game_rate"] = _distinct_game_rate(graded)
    # How far each game is from the nearest other one, read off the move lists. The
    # position forms need the analysis pass and are measured with it, below.
    raw["prefix_divergence"] = _prefix_divergence(graded)

    # --- how much the balance moves ---------------------------------------------
    per_game = [(readings or GameReadings()).of(r) for r in graded]
    raw["lead_changes"] = _mean([reading.lead_changes for reading in per_game])
    raw["material_volatility"] = _mean([reading.volatility for reading in per_game])

    # --- is anything happening that cannot be undone -----------------------------
    # The share of plies that reset the no-progress clock. A designer watching his own
    # game called the failure "the gameplay gets very repetitive" and diagnosed it as a
    # missing irreversible process -- a game that can shuffle forever has no clock on
    # it but its own. The series this reads has been persisted in every record since
    # the beginning and read by nothing.
    irreversibility = [
        reading.irreversibility
        for record, reading in zip(graded, per_game, strict=True)
        if record.irreversible
    ]
    if irreversibility:
        raw["progress_irreversibility"] = _mean(irreversibility)

    # Whether an advantage means anything. 0.5 says a lead is worth nothing, which is
    # a different failure from a game that draws a lot -- ``draw_slope`` asks whether
    # decisiveness survives better players, this asks whether an edge converts inside
    # one game. Withheld rather than scored where the ladder says the roster cannot
    # read the ruleset: material is the wrong objective in antichess and will be the
    # wrong objective in any cyclic-capture game, and a number that describes the
    # evaluator must not be reported as a property of the design.
    conversion, conversion_detail = _lead_conversion(graded)
    if conversion is not None:
        raw["lead_conversion"] = conversion

    # --- what the pieces can do to each other, before anything plays --------------
    raw["threat_asymmetry"] = facts.threat_asymmetry
    raw["traversal_time"] = facts.traversal_time

    # --- does thinking pay -------------------------------------------------------
    gradient, gradient_detail = _skill_gradient(ratings)
    raw["skill_gradient"] = gradient
    raw["rating_spread"] = ratings.spread()
    raw["search_advantage"] = _search_advantage(ratings)

    # --- does thinking *keep* paying ---------------------------------------------
    # The distinction the single gradient cannot draw. Both a deep game and a nearly
    # solved one reward search at modest budgets; only the deep one still rewards it at
    # the top of the ladder.
    persistence, persistence_detail = _gradient_persistence(ratings)
    raw["gradient_persistence"] = persistence

    # Whether the ladder is a ladder at all. Not a component -- a check on the
    # instrument. Every skill measurement here assumes that a bigger budget makes a
    # better player, and in a ruleset whose objective the evaluator has backwards that
    # assumption fails outright: in antichess the 6400-node search rates level with a
    # random mover and the 400-node search finishes last, because all three are
    # searching hard towards the wrong thing. The resulting near-zero gradient is a fact
    # about the agents, not about the game, and reporting it as a property of the game
    # would be the measurement lying about its own scope.
    raw["ladder_monotonicity"] = _ladder_monotonicity(ratings)

    # --- how much rule buys how much game ----------------------------------------
    # Elegance as yield rather than as thinness. Scoring low rule mass directly rewards
    # having less game in the box: across the reference corpus mass correlates +0.88
    # with branching, +0.85 with decisiveness and +0.82 with rating spread, so a
    # lower-is-better curve on it subtracts from the very signal the depth family adds,
    # and both deliberately broken controls take the top of the family. A ratio asks the
    # question the metric was always meant to ask -- what did these rules buy? -- and
    # puts the dull control below every real game.
    raw["rule_yield"] = (
        raw["rating_spread"] / complexity.rule_mass if complexity.rule_mass > 0 else 0.0
    )

    # --- how the game responds to skill, not to a snapshot of it ------------------
    decisive_by_tier, tier_detail = _decisive_by_tier(records)
    raw["draw_slope"] = _draw_slope(decisive_by_tier)

    diagnostics: dict[str, Any] = {
        "games": len(records),
        "graded_games": len(graded),
        "self_play_games": len(self_play),
        "skill_gradient_fits": gradient_detail,
        "gradient_persistence_fits": persistence_detail,
        "decisive_by_tier": tier_detail,
        "ladder_split": split_detail,
        **seat_details,
        "outcome_reasons": dict(Counter(r.reason for r in graded)),
        "complexity": complexity.as_dict(),
        "lead_conversion_sample": conversion_detail,
        # Read over every game, not only the graded ones: how deep a rung saw is a
        # property of the rung and the ruleset, and a pairing against the random mover
        # is as good a sample of it as any other.
        "depth_by_agent": _depth_by_agent(records),
    }

    if plays_with_spies(spec):
        # A spy game (schema 27) alternates and then plays rounds, and says so in its own
        # words rather than a round game's.
        diagnostics["spy_readings"] = dict(SPY_READINGS)
    elif plays_in_rounds(spec):
        # What each scored component means on a game played in rounds, carried in the
        # score itself so no reader takes a round game's number for an alternating one's.
        diagnostics["round_readings"] = dict(ROUND_READINGS)

    if analysis is not None:
        raw.update(_analysis_components(analysis, diagnostics, spec))
        raw.update(_keyed_components(graded, analysis, diagnostics))

    # The heuristic pass. Its games are a separate corpus and never reach `records`, so
    # nothing above can see them -- which is the guarantee that a per-ruleset player
    # cannot contaminate a ladder every ruleset is compared on.
    if heuristics is not None:
        raw.update(heuristic_components(heuristics))
        diagnostics["heuristics"] = heuristic_diagnostics(heuristics)

    return Components(raw=raw, diagnostics=diagnostics)


# ------------------------------------------------------------------ measurements


# Cached, as `_budget` is: both read an agent name, a score asks them about half a
# million times, and a roster holds a handful of names.
@lru_cache(maxsize=1024)
def _is_weak(agent: str) -> bool:
    return agent.split(":")[0] == "random"


def _is_heuristic(agent: str) -> bool:
    """Whether an agent string names a heuristic player.

    Defence in depth, not the guarantee. The guarantee is that the heuristic pass keeps
    its games in a separate corpus with a separate rating table, and it has to be --
    ``rating_spread`` reads the whole table and no name filter can reach it. This exists
    so that a record that somehow arrives here cannot quietly enter a budget ladder, and
    it matches on the option rather than the kind because ``_budget_tiers`` groups by
    ``name.split(":")[0]``: renaming the kind would still contribute its own slope.
    """
    return bool(_HEURISTIC_PATTERN.search(agent))


def _rate(records: list[GameRecord], predicate: Callable[[GameRecord], bool]) -> float:
    return sum(1 for r in records if predicate(r)) / len(records) if records else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _spread(values: list[float]) -> float:
    """Coefficient of variation: length variety relative to typical length.

    Absolute standard deviation would make every long game look varied. What matters
    is whether games differ from each other, not how big the numbers are.
    """
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


# ------------------------------------------------- the outcome gates' denominator


def _outcome_rates(records: list[GameRecord]) -> dict[str, float]:
    """The four rates the outcome gates read, over whatever set of games is handed in.

    One function so that the pooled reading and the worst-half reading cannot drift
    apart: they are the same four measurements on two denominators, and the whole
    content of the split is which denominator a gate is allowed to use.
    """
    return {
        "stall_rate": _rate(records, lambda r: r.reason in STALL_REASONS),
        "decisive_rate": _rate(records, lambda r: r.winner is not None),
        "mean_plies": _mean([float(r.plies) for r in records]),
        "quick_kill_rate": _rate(
            records, lambda r: r.plies < QUICK_KILL_PLIES and r.winner is not None
        ),
    }


@lru_cache(maxsize=1024)
def _budget(agent: str) -> int:
    """The node budget an agent string names, or zero where it names none.

    Zero is a rung, not an absence: ``greedy`` is rated and graded and searches nothing,
    so it is the bottom of the ladder the gates read rather than missing from it.
    """
    match = _NODE_PATTERN.search(agent)
    return int(match.group(1)) if match else 0


def _split_rung(records: list[GameRecord]) -> int | None:
    """The median rung of the graded ladder, in nodes; ``None`` when there is no ladder.

    Pre-registered in ``docs/reference/roster-confound.md`` section 3 as *the median rung of
    the graded ladder*, deliberately as English rather than as a node count, so that it
    could not be chosen after the corpus had been swept. On the canonical roster the
    graded ladder is ``greedy`` plus the six searching budgets -- seven rungs, whose
    fourth is 400.

    **The tie-break, fixed here while it decides nothing.** At an odd rung count the two
    readings of "median" agree, which is the case on every roster this project has
    played; at an even count they differ, and section 6.2 records that choosing between
    them *after* a roster changed would be choosing a threshold while looking at the
    control it excludes. So it is chosen now: the **lower** median. The reason is
    section 3's own sentence -- a game is in the strong half when its weaker seat stands
    *at or above* the midpoint rung, which puts the midpoint rung itself in the strong
    half and makes that half the larger one at an odd count. The lower median is the
    reading that keeps it the larger one at an even count too. Nothing on today's roster
    turns on it, and section 7.3 has already checked both readings against every
    candidate roster that has been costed.

    The rungs are read off the seats of the graded games the way ``_budget_tiers`` reads
    them. The heuristic pass keeps its games in a separate corpus with its own rating
    table, so no per-ruleset player can arrive here and put a rung on the ladder that
    only one ruleset has.
    """
    rungs = sorted({_budget(name) for r in records for name in r.agents})
    if len(rungs) < 2:
        return None
    return rungs[(len(rungs) - 1) // 2]


def _worst_half_rates(
    records: list[GameRecord],
) -> tuple[dict[str, float], dict[str, Any]]:
    """The four gated rates on the worse half of the budget ladder.

    A game is in the strong half when its *weaker* seat stands at or above the midpoint
    rung -- "can both players see it?" -- and in the weak half otherwise. Each half is
    measured on its own games, and each rate reports whichever half reads worse in that
    rate's own direction. The halves are chosen per rate rather than once for all four,
    because the failure this exists to catch is precisely a ruleset where the two halves
    fail *different* gates.

    Degenerate by construction rather than by accident: with fewer than two rungs there
    is no ladder to halve, and with every game on one side of the split the other half
    is empty. Both cases fall back to the pooled set, so a corpus the split cannot read
    gates exactly as it did before rather than gating on an empty denominator.
    """
    split = _split_rung(records)
    halves: dict[str, list[GameRecord]] = {}
    if split is not None:
        weak = [r for r in records if min(_budget(a) for a in r.agents) < split]
        strong = [r for r in records if min(_budget(a) for a in r.agents) >= split]
        halves = {name: half for name, half in (("weak", weak), ("strong", strong)) if half}
    if len(halves) < 2:
        halves = {"pooled": records}

    measured = {name: _outcome_rates(half) for name, half in halves.items()}

    raw: dict[str, float] = {}
    worst_of: dict[str, str] = {}
    for rate_name, direction in WORST_HALF_RATES.items():
        pick = max if direction == "high" else min
        worst = pick(measured, key=lambda name: measured[name][rate_name])
        raw[rate_name + WORST_HALF_SUFFIX] = measured[worst][rate_name]
        worst_of[rate_name] = worst

    detail: dict[str, Any] = {
        "split_rung": split,
        "rungs": sorted({_budget(name) for r in records for name in r.agents}),
        "games": {name: len(half) for name, half in halves.items()},
        "rates": {
            name: {rate: round(value, 4) for rate, value in rates.items()}
            for name, rates in measured.items()
        },
        "worst_half": worst_of,
    }
    return raw, detail


def _seat_readings(records: list[GameRecord]) -> tuple[dict[str, float], dict[str, Any]]:
    """The first seat's record at the top of the ladder, on two denominators.

    The ladder is the one ``_split_rung`` reads, and a reading's games are every one whose
    two seats are both in its band: both self-play pairings and both orders of each cross
    pairing. The top pair is 16 games on the canonical roster and the top three are 36.
    Narrow bands, because a seat effect that appears only above a budget is diluted by
    every rung added below -- ``tunnel-vertical`` gives the first seat 8 of 8 at its top
    two rungs' self-play and 8 of 24 beneath them.

    The wide reading needs three *searching* rungs. The design loop's roster has two, so
    its top three would take in ``greedy``, and a band whose bottom rung searches nothing
    is a different measurement from the one the threshold was set for. That roster reads
    the pair alone, which is the user's decision of 2026-09-18.
    """
    rungs = sorted({_budget(name) for r in records for name in r.agents})
    if len(rungs) < 2:
        return {}, {TOP_RUNG_SEAT_DETAIL: {"rungs": rungs, "games": 0}}

    raw, pair_detail = _seat_split(records, rungs[-2:], SEAT_DECIDED_CHANCE,
                                   TOP_RUNG_FIRST_SEAT_SCORE)
    details: dict[str, Any] = {TOP_RUNG_SEAT_DETAIL: pair_detail}

    if len(rungs) >= 3 and rungs[-3] > 0:
        wide_raw, wide_detail = _seat_split(records, rungs[-3:],
                                            TOP_THREE_SEAT_DECIDED_CHANCE,
                                            TOP_THREE_FIRST_SEAT_SCORE)
        raw.update(wide_raw)
        details[TOP_THREE_SEAT_DETAIL] = wide_detail
    return raw, details


def _seat_split(
    records: list[GameRecord], band: list[int], chance_key: str, score_key: str
) -> tuple[dict[str, float], dict[str, Any]]:
    """One band's sign test and its split, the test over each different decided game once.

    Each decided game's move list is kept once however often it was played. Two players
    whose draws are inert at one budget replay one game, so a band can hold many decided
    games and a handful of games actually played (`seat-effect-size.md` §15), and a line
    played sixteen times is one observation of what that pairing does, not sixteen. The
    sign test reads the distinct split (Q953a, `docs/reference/repeated-games.md` §6,
    reversing Q534's *the refusal does not read the count*); the split as played stays
    beside it. A transcript's winner is read off its first copy: one move list has one
    ending. Draws are in neither, for the reason they are not in the test.
    """
    top = set(band)
    games = [r for r in records if all(_budget(name) in top for name in r.agents)]
    wins = sum(1 for r in games if r.winner == 0)
    losses = sum(1 for r in games if r.winner == 1)
    once: dict[str, int] = {}
    for r in games:
        if r.winner in (0, 1):
            once.setdefault(transcript_digest(r.moves), int(r.winner))
    transcripts = sorted(once)
    first_seat = sorted(digest for digest, winner in once.items() if winner == 0)
    distinct_wins = len(first_seat)
    distinct_losses = len(transcripts) - distinct_wins
    chance = sign_test(distinct_wins, distinct_losses)
    raw = {
        chance_key: chance,
        score_key: _mean([r.score_for(0) for r in games]) if games else 0.5,
    }
    detail = {
        "rungs": sorted(top),
        "games": len(games),
        "wins": wins,
        "draws": len(games) - wins - losses,
        "losses": losses,
        "chance": chance,
        "distinct": len(transcripts),
        "distinct_wins": distinct_wins,
        "distinct_losses": distinct_losses,
        "transcripts": transcripts,
        # The decided transcripts the first seat won, so a pooled reading can split the
        # union of several runs' transcripts without their records.
        "first_seat_transcripts": first_seat,
    }
    return raw, detail


def _identity(record: GameRecord) -> tuple[str, int, tuple[str, str]]:
    """Which played game a record is: ``(ruleset_hash, seed, agents)``."""
    return (record.ruleset_hash, record.seed, record.agents)


def _one_per_game(records: list[GameRecord]) -> list[GameRecord]:
    """Each played game once, in first-seen order.

    A game's identity is ``(ruleset_hash, seed, agents)``, read from the record. A
    bootstrap resample draws with replacement and can hold one game several times.
    A component that counts distinct games reads each identity once, so a copy is
    never a repeat. On a stored corpus every identity is already unique and this
    returns the records unchanged.
    """
    seen: set[tuple[str, int, tuple[str, str]]] = set()
    unique = []
    for record in records:
        identity = _identity(record)
        if identity not in seen:
            seen.add(identity)
            unique.append(record)
    return unique


def _opening_diversity(records: list[GameRecord]) -> float:
    """Normalised entropy over opening move sequences, one entry per game.

    Reported against the maximum entropy the sample size allows, so it answers "were
    these games different from each other?" rather than "how many games were there?" --
    otherwise a bigger tournament would score as a more interesting game.
    """
    games = _one_per_game(records)
    prefixes = Counter(tuple(r.moves[:OPENING_PREFIX_PLIES]) for r in games if r.moves)
    total = sum(prefixes.values())
    if total <= 1 or len(prefixes) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in prefixes.values())
    return min(1.0, entropy / math.log(total))


def _distinct_game_rate(records: list[GameRecord]) -> float:
    """Share of games whose move sequence no other game played, one entry per game."""
    played = [tuple(r.moves) for r in _one_per_game(records) if r.moves]
    if not played:
        return 0.0
    return len(set(played)) / len(played)


# ------------------------------------------------ variety: the nearest other game
#
# `docs/archive/plan-variety.md` §3. Each measurement here is a mean over graded games of a
# distance to the nearest *other* game, where "other" means a different original game.
# The neighbours are `_one_per_game`'s entries, so a bootstrap resample that draws one
# game twice holds one identity, and a copy is never a neighbour at distance zero. The
# mean runs over the draws, as every averaging component's does: a drawn copy weighs
# twice, and a corpus doubled whole reads the same. Two games with different seeds and
# identical moves are two identities at distance zero, which is a replay read as one.
#
# The nearest neighbour rather than the average one, because the failure a player
# notices is "the same game keeps being replayed", which is about the nearest copy.

# The two that read the analysis pass's position keys, so a corpus whose pass predates
# the keys has neither and `index.score_components` names the pass to run. Declared here
# and not in `tournament.analysis.ANALYSIS_COMPONENTS`: that module is in the `analysis`
# fingerprint, and an edit to it would turn every stored run's re-score into a
# re-analysis.
KEYED_COMPONENTS = ("position_divergence", "event_divergence")

# A game as a nearest-neighbour measurement reads it: its moves, its positions, or its
# events, in order. A tuple, so that two games that read alike are one entry in a table.
_Content = tuple[int, ...]
# What the distance is taken between, made from a content once per table.
_Item = TypeVar("_Item")


def _prefix_distance(first: _Content, second: _Content) -> float:
    """One minus the share of the shorter game that the two play move for move.

    A game that is a prefix of the other is at distance zero, the empty game included.
    """
    shorter = min(len(first), len(second))
    if shorter == 0:
        return 0.0
    shared = 0
    for a, b in zip(first, second, strict=False):
        if a != b:
            break
        shared += 1
    return 1.0 - shared / shorter


def _set_distance(first: frozenset[int], second: frozenset[int]) -> float:
    """Jaccard distance: one minus the shared positions over the positions either saw."""
    shared = len(first & second)
    union = len(first) + len(second) - shared
    return 1.0 - shared / union if union else 0.0


@dataclass(frozen=True)
class _Measure(Generic[_Item]):
    """One nearest-neighbour measurement: what it makes of a game, and how far apart two are."""

    name: str
    prepare: Callable[[_Content], _Item]
    distance: Callable[[_Item, _Item], float]


def _as_moves(content: _Content) -> _Content:
    return content


_PREFIX = _Measure("prefix_divergence", _as_moves, _prefix_distance)
_POSITIONS = _Measure("position_divergence", frozenset, _set_distance)
_EVENTS = _Measure("event_divergence", frozenset, _set_distance)


class _NeighbourTable(Generic[_Item]):
    """Every distance between two of one corpus's games, and each game's others by nearness.

    A game is entered by its content, so two games that read alike share an entry, at
    the distance the measure gives a game and itself. An entry is not an identity:
    which games a call holds, and how many of them share an entry, is the caller's.
    """

    def __init__(
        self, contents: list[_Content], items: list[_Item], dist: list[list[float]]
    ) -> None:
        self.contents = contents
        self.items = items
        self.slot = {content: index for index, content in enumerate(contents)}
        self.dist = dist
        self.order = [
            sorted((other for other in range(len(dist)) if other != index), key=row.__getitem__)
            for index, row in enumerate(dist)
        ]

    @classmethod
    def build(
        cls,
        measure: _Measure[_Item],
        contents: list[_Content],
        base: _NeighbourTable[_Item] | None,
    ) -> _NeighbourTable[_Item]:
        """A table over ``base``'s entries and ``contents``, keeping every distance ``base`` has."""
        fresh = [c for c in contents if base is None or c not in base.slot]
        entered = (base.contents if base is not None else []) + fresh
        items = (base.items if base is not None else []) + [measure.prepare(c) for c in fresh]
        reused = len(entered) - len(fresh)
        dist = [[0.0] * len(entered) for _ in entered]
        for first in range(len(entered)):
            for second in range(first, len(entered)):
                if base is not None and second < reused:
                    gap = base.dist[first][second]
                else:
                    gap = measure.distance(items[first], items[second])
                dist[first][second] = gap
                dist[second][first] = gap
        return cls(entered, items, dist)

    def nearest(self, slot: int, present: set[int], twinned: bool) -> float:
        """The distance from ``slot`` to the nearest other entry in ``present``.

        ``twinned`` says a second identity holds this same entry, and is then at the
        distance the measure gives the entry and itself.
        """
        row = self.dist[slot]
        best = row[slot] if twinned else math.inf
        for other in self.order[slot]:
            if other in present:
                return min(best, row[other])
        return best


# Memoised because the bootstrap reads a corpus 200 times over. Built from nothing, the
# three variety measurements cost about 0.1 s a call on chess's 196 graded games, which is
# 20 s over a bootstrap against a whole scoring of about 3 s. A resample holds about two
# thirds of its corpus's games, so it reads the table its corpus built, and a first call
# that is itself a resample grows the table it mostly shares. A table is only ever read,
# so every answer is the one a table built for that call alone would give.
_TABLES_PER_MEASURE = 4
_TABLES: dict[str, list[_NeighbourTable[Any]]] = {}
_TABLES_LOCK = threading.Lock()


def _entered(
    measure: _Measure[_Item], contents: list[_Content]
) -> tuple[_NeighbourTable[_Item], list[int]]:
    """A table holding every one of ``contents``, and the slot each is entered at."""
    with _TABLES_LOCK:
        tables = _TABLES.setdefault(measure.name, [])
        for index, table in enumerate(tables):
            slots = list(map(table.slot.get, contents, repeat(-1)))
            if -1 not in slots:
                tables.insert(0, tables.pop(index))
                return table, slots
        # Grown only from a table this call mostly shares, and only to twice this call's
        # size, so two corpora that happen to share a game are never joined.
        base = next((table for table in tables if _extends(table, contents)), None)

    grown = _NeighbourTable.build(measure, list(dict.fromkeys(contents)), base)
    with _TABLES_LOCK:
        tables = _TABLES.setdefault(measure.name, [])
        if base is not None and base in tables:
            tables.remove(base)
        tables.insert(0, grown)
        del tables[_TABLES_PER_MEASURE:]
    return grown, [grown.slot[content] for content in contents]


def _extends(table: _NeighbourTable[Any], contents: list[_Content]) -> bool:
    shared = len(set(contents) & table.slot.keys())
    wanted = len(set(contents))
    return 2 * shared >= wanted and len(table.contents) + wanted - shared <= 2 * wanted


def _mean_nearest(
    graded: list[GameRecord],
    measure: _Measure[Any],
    content_of: Callable[[GameRecord], _Content | None],
) -> float:
    """The mean over ``graded`` of each game's distance to its nearest other game.

    ``content_of`` gives what the measure reads of one game, or ``None`` for a game that
    is nobody's neighbour and scores zero. Fewer than two games with a content have no
    neighbour to read, and the whole measurement is then zero: no variety was seen.
    """
    entries = _one_per_game(graded)
    contents = [content_of(entry) for entry in entries]
    eligible = [content for content in contents if content is not None]
    if len(eligible) < 2:
        return 0.0

    table, found = _entered(measure, eligible)
    held = Counter(found)
    present = set(held)
    reading = iter(found)
    nearest = {}
    for entry, content in zip(entries, contents, strict=True):
        if content is None:
            nearest[_identity(entry)] = 0.0
        else:
            slot = next(reading)
            nearest[_identity(entry)] = table.nearest(slot, present, held[slot] > 1)
    # `fsum`, so that a corpus doubled whole reads exactly what it read once.
    return math.fsum(nearest[_identity(record)] for record in graded) / len(graded)


def _prefix_divergence(graded: list[GameRecord]) -> float:
    """§3.1: how much of a game its nearest other game plays move for move.

    Reads the move lists alone. An unweighted diagnostic beside the position form, for
    §3.6's comparison: it reads a transposition as a different game and a first-ply
    tie-break as total divergence, and is blind to anything after the parting.
    """
    return _mean_nearest(graded, _PREFIX, _moves_of)


def _moves_of(record: GameRecord) -> _Content:
    return tuple(record.moves)


def _keys_by_game(
    analysis: CorpusAnalysis,
) -> dict[tuple[int, tuple[str, str]], list[int]] | None:
    """Each analysed game's position keys, by ``(seed, agents)``.

    ``None`` when the pass predates the keys: an ``analysis.json`` written before the
    field has an empty list on every game that walked a ply.
    """
    keys: dict[tuple[int, tuple[str, str]], list[int]] = {}
    for game in analysis.games:
        if game.walked_plies and not game.position_keys:
            return None
        keys[(game.seed, game.agents)] = game.position_keys
    return keys


def _keyed_components(
    graded: list[GameRecord], analysis: CorpusAnalysis, diagnostics: dict[str, Any]
) -> dict[str, float]:
    """§3.2 and §3.3, from the position keys the analysis pass persists.

    Read over the graded games the pass describes, looked up by ``(seed, agents)`` so a
    resampled corpus reads the analyses of the games it drew.
    """
    keys = _keys_by_game(analysis)
    if keys is None:
        return {}
    analysed = [r for r in graded if (r.seed, r.agents) in keys]
    if not analysed:
        return {}

    def positions(record: GameRecord) -> _Content:
        return tuple(keys[(record.seed, record.agents)])

    raw = {"position_divergence": _mean_nearest(analysed, _POSITIONS, positions)}

    # The irreversible series is empty on a corpus collected without its series, and a
    # game read as eventless for want of a series would score a zero it never earned.
    if all(record.irreversible or not record.moves for record in analysed):
        reached = {
            (record.seed, record.agents): _events(record, keys[(record.seed, record.agents)])
            for record in _one_per_game(analysed)
        }

        def events(record: GameRecord) -> _Content | None:
            return reached[(record.seed, record.agents)]

        raw["event_divergence"] = _mean_nearest(analysed, _EVENTS, events)
        diagnostics["eventless_games"] = sum(1 for found in reached.values() if found is None)
    return raw


def _events(record: GameRecord, keys: list[int]) -> _Content | None:
    """The positions reached by a ply that cannot be undone, or ``None`` if there were none.

    ``keys[p]`` is the position after ply ``p`` and ``record.irreversible[p]`` says
    whether that ply reset the no-progress clock, so the two series index alike.
    """
    events = tuple(compress(keys, record.irreversible))
    return events or None


def _lead_changes(record: GameRecord) -> float:
    """How often the material advantage crosses sides, per hundred plies.

    A game where the winner is settled by move six and never troubled is a different
    thing from one that turns over repeatedly, even if both end 1-0.
    """
    series = [m for m in record.material if m != 0.0]
    if len(series) < 2:
        return 0.0
    changes = sum(
        1 for a, b in zip(series, series[1:], strict=False) if (a > 0) != (b > 0)
    )
    return 100.0 * changes / len(record.material) if record.material else 0.0


def _irreversibility(record: GameRecord) -> float:
    """Share of this game's plies that could never be undone by playing on.

    ``record.irreversible`` is the series the record has always called ``captures``: a 1
    wherever the no-progress clock reset. ``Game::apply`` in ``rust/src/game.rs`` is the
    definition and it is wider than the name suggests -- any change in material, any move
    by a one-way piece, the promotion move itself, or a permanent colour laid on a
    square, which a territory game does every time it claims one. Divided by its own
    length rather than by ``plies``, because the series is empty when a corpus was
    collected with ``collect_series=False`` and a short series would otherwise read as
    a game with nothing happening in it.
    """
    series = record.irreversible
    return sum(series) / len(series) if series else 0.0


def _lead_conversion(
    records: list[GameRecord],
) -> tuple[float | None, dict[str, Any]]:
    """Among decisive games with a real material edge partway through, how often the
    side ahead went on to win.

    0.5 means a lead is worth nothing and the result is decided by whatever happens
    afterwards; 1.0 means the game is over as soon as anyone is ahead, which is the
    other failure. The measurement is the video's Go lesson stated as a number -- "in
    Go, a small advantage will straight up win you the game" -- and it is a different
    question from ``draw_slope``, which asks whether *better players* produce decisive
    games rather than whether an edge converts inside one.

    ``None`` when too few games qualify. A corpus of draws has nothing to say about
    conversion, and reporting 0.0 there would say the opposite of nothing.
    """
    edge_at = LEAD_FRACTION
    qualifying: list[bool] = []
    for record in records:
        if record.winner is None or not record.material:
            continue
        index = int(len(record.material) * edge_at)
        if index >= len(record.material):
            continue
        edge = record.material[index]
        if abs(edge) < LEAD_EPSILON:
            continue
        # `material` is stated from seat 0's point of view, in the same convention
        # `_lead_changes` reads it: positive means seat 0 is ahead.
        qualifying.append((edge > 0) == (record.winner == 0))

    detail = {"qualifying_games": len(qualifying), "decisive_games": sum(
        1 for r in records if r.winner is not None
    )}
    if len(qualifying) < MIN_LEAD_SAMPLE:
        return None, detail
    return sum(qualifying) / len(qualifying), detail


def _displacements(compiled: CompiledSpec, mask: int) -> list[set[tuple[int, int]]]:
    """Per piece type, the relative squares it can reach with a move of this kind.

    Read off the compiled tables rather than off the spec's move descriptors, so that
    everything the compiler bakes in -- symmetry expansion, regions, min ranges, lame
    leaps -- is already accounted for and this does not become a second interpreter of
    the DSL. Blockers are ignored on purpose: the question is what a piece is *able*
    to do, which is a fact about the rules and not about one position.

    ``mask`` selects capture reach or reach of any kind. The two are not the same
    question and one caller wants each: what a piece can do *to another piece* is not
    what it can do to the board.

    Every move table the compiler emits is read, ``sweeps`` included. A sweep ray is
    shaped exactly as a slide ray, so its squares are displacements on the same terms,
    and leaving it out made a sweeping piece measure as one that cannot move.

    Takes the compiled tables rather than the spec because compiling is 29-33 ms and
    both callers of this already hold them: reading them is the point of the function,
    and re-deriving them per call is what made ``spec_facts`` worth having.
    """
    width = compiled.width
    displacements: list[set[tuple[int, int]]] = [set() for _ in range(compiled.n_types)]
    for table in (compiled.leaps, compiled.slides, compiled.hops, compiled.sweeps):
        for piece_type in range(compiled.n_types):
            for square in range(compiled.n_squares):
                for entry in table[0][piece_type][square]:
                    targets, modes = entry[0], entry[1]
                    for target, mode in zip(targets, modes, strict=False):
                        if not mode & mask:
                            continue
                        displacements[piece_type].add(
                            (target % width - square % width, target // width - square // width)
                        )
    return displacements


def _threat_asymmetry(compiled: CompiledSpec) -> float:
    """Share of ordered piece-type pairs where one can strike from out of reach.

    The formal version of "try playing the game yourself with fifteen rooks". Fifteen
    rooks measure zero here: every rook attacks every square another rook attacks it
    from, so no pair is one-way. A game where the pieces differ in *what they threaten*
    rather than in how far they travel measures high, and that is a distinction every
    other component in this file is blind to.

    ``a`` threatens ``b`` one-way if there is some relative placement at which ``a`` may
    capture ``b`` and ``b``, standing there, may not capture back. Two things decide
    that and the measurement ANDs them: **geometry**, which is where a piece can strike
    from, and the **capture matrix**, which is what it is allowed to strike.

    The matrix half arrived with schema 3 and this read only geometry until stage 12's
    "re-read it then" was collected. Reach alone scores `rps-v1` at 0.000 -- exactly
    the `dull` control, fifteen rooks -- because rock, paper and scissors move
    identically and the whole asymmetry of that game is in what they may take. That is
    the one design this component exists to tell apart from a uniform army, so a
    version of it that cannot is not worth reading.

    Inert where no spec declares a matrix: ``capture_allowed`` is then true everywhere
    and the test reduces to the geometry it was. No committed ruleset outside the RPS
    corpus moves, and it is scored under no weights version in any case -- which is what
    made this safe to correct rather than schedule.
    """
    displacements = _displacements(compiled, MODE_CAPTURE)
    if len(displacements) < 2:
        return 0.0

    allowed = compiled.capture_allowed

    def one_way(attacker: int, victim: int) -> bool:
        if not allowed[attacker][victim]:
            # Not a threat at all, from anywhere. Note this is not symmetric, which is
            # the point: it is what makes a cycle read as three one-way threats.
            return False
        # The victim is the other player, so its own capture geometry is mirrored in y
        # exactly as the compiler mirrors it. A displacement `d` from attacker to
        # victim is `-d` from victim to attacker; the victim's mirrored set is what has
        # to be checked, or a forward-only piece reads as able to answer backwards.
        answers = (
            {(-x, y) for x, y in displacements[victim]}
            if allowed[victim][attacker]
            else set()
        )
        return any(
            (-dx, -dy) not in answers for dx, dy in displacements[attacker]
        )

    pairs = [
        (a, b)
        for a in range(len(displacements))
        for b in range(len(displacements))
        if a != b
    ]
    if not pairs:
        return 0.0
    asymmetric = sum(
        1 for a, b in pairs if one_way(a, b) and not one_way(b, a)
    )
    return asymmetric / len(pairs)


def _traversal_time(compiled: CompiledSpec) -> float:
    """Board diameter divided by how far a typical piece travels in one move.

    The version-6 rejection stated as a number: "a rock can traverse the entire board
    in just four moves." A game where crossing the board takes two moves is a game
    where macro strategy has no time to exist.

    A **diagnostic and a candidate grid axis, never a scored component.** Scoring it
    asserts that slower is better, which is a taste claim the corpus cannot support --
    gardner is fast and good, courier is slow and good.
    """
    diameter = float(max(compiled.width, compiled.height) - 1)
    reaches: list[float] = []
    # Any move, not just a capture. Crossing a board is a movement question, and a
    # ruleset with captures switched off has no capture geometry at all -- reading this
    # off the capture tables scored `mutant-no-captures` as instantaneous rather than
    # as slow, which is the opposite of the truth.
    for piece_type, displacement in enumerate(
        _displacements(compiled, MODE_MOVE | MODE_CAPTURE)
    ):
        if not displacement:
            continue
        # Chebyshev, because a board is a grid and a diagonal step crosses one rank.
        typical = _mean([float(max(abs(dx), abs(dy))) for dx, dy in displacement])
        # Weighted by how many of that piece stand on the board, so a game defined by
        # its pawns is not described by its one queen.
        count = sum(
            1
            for square in range(compiled.n_squares)
            if compiled.initial_board[square] == compiled.encode_piece(0, piece_type)
        )
        reaches.extend([typical] * max(count, 1))
    mean_reach = _mean(reaches)
    return diameter / mean_reach if mean_reach > 0 else 0.0


def _volatility(record: GameRecord) -> float:
    """Mean absolute change in material per ply."""
    if len(record.material) < 2:
        return 0.0
    return _mean(
        [abs(b - a) for a, b in zip(record.material, record.material[1:], strict=False)]
    )


def _skill_gradient(ratings: RatingTable) -> tuple[float, dict[str, float]]:
    """Elo gained per doubling of the thinking budget.

    This is the closest thing the project has to a direct measurement of strategic
    depth: in a game with real decisions, letting a player look at twice as many
    positions should reliably help; in a shallow one, it barely moves the needle.

    Fitted per agent kind and averaged, because minimax and MCTS fail in different
    ways -- a slope from one alone could be a property of the search rather than of
    the game.
    """
    by_kind: dict[str, list[tuple[float, float]]] = {}
    for standing in ratings.standings:
        if _is_heuristic(standing.name):
            continue
        match = _NODE_PATTERN.search(standing.name)
        if not match:
            continue
        nodes = int(match.group(1))
        if nodes <= 0:
            continue
        kind = standing.name.split(":")[0]
        by_kind.setdefault(kind, []).append((math.log2(nodes), standing.rating))

    slopes: dict[str, float] = {}
    for kind, points in by_kind.items():
        if len(points) < 2:
            continue
        slope = _least_squares_slope(points)
        if slope is not None:
            slopes[kind] = round(slope, 2)

    if not slopes:
        return 0.0, {}
    return sum(slopes.values()) / len(slopes), slopes


def _least_squares_slope(points: list[tuple[float, float]]) -> float | None:
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def _search_advantage(ratings: RatingTable) -> float:
    """Score the strongest agent takes off a random mover, on a 0-1 scale.

    The floor test. If a searching player cannot beat random, either the game has no
    handle to grip or the engine is wrong -- and in a design loop the second is at
    least as likely as the first.
    """
    randoms = [s.name for s in ratings.standings if _is_weak(s.name)]
    if not randoms or len(ratings.standings) < 2:
        return 0.0

    best = next((s.name for s in ratings.standings if not _is_weak(s.name)), None)
    if best is None:
        return 0.0

    scores: list[float] = []
    for random_name in randoms:
        forward = ratings.head_to_head.get((best, random_name))
        reverse = ratings.head_to_head.get((random_name, best))
        if forward:
            scores.append(forward[0])
        if reverse:
            scores.append(1.0 - reverse[0])
    return _mean(scores)


# ------------------------------------------------------- skill-response measures


def _budget_tiers(names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Searching agents grouped by kind, sorted by node budget."""
    by_kind: dict[str, list[tuple[int, str]]] = {}
    for name in names:
        if _is_heuristic(name):
            continue
        match = _NODE_PATTERN.search(name)
        if not match:
            continue
        nodes = int(match.group(1))
        if nodes > 0:
            by_kind.setdefault(name.split(":")[0], []).append((nodes, name))
    for entries in by_kind.values():
        entries.sort()
    return by_kind


def _gradient_persistence(ratings: RatingTable) -> tuple[float, dict[str, float]]:
    """Elo per doubling at the top of the ladder, over the same at the bottom.

    The most informative thing the index was not measuring. ``skill_gradient`` answers
    "does searching help?", and every real game answers yes -- which is why the corpus
    spread on it is mostly noise. The question that separates a deep game from a nearly
    solved one is whether searching *keeps* helping: a game with a reachable ceiling
    shows returns that decay as the budget climbs, while one with room left shows a
    slope that barely bends.

    A ratio, so branching factor and board size largely cancel. Reported as 1.0 when the
    lower slope is degenerate, which is the neutral reading rather than a free zero.

    Needs at least three budgets of one agent kind. With exactly three it is a
    two-point-over-two-point comparison and noisy; with five it is a fit over each half
    of the range. The shape of the question is the same either way.
    """
    ladders = _budget_tiers([s.name for s in ratings.standings])
    rating_of = {s.name: s.rating for s in ratings.standings}

    ratios: dict[str, float] = {}
    for kind, entries in ladders.items():
        if len(entries) < 3:
            continue
        points = [(math.log2(nodes), rating_of[name]) for nodes, name in entries]
        # Halves, overlapping by one point only when the count is odd and they must.
        # With four budgets the two fits share nothing, which is the whole reason four
        # is the minimum worth running: at three, "the bottom" and "the top" are the
        # same middle rating read twice and the ratio is mostly its noise.
        split = len(points) // 2
        lower = _least_squares_slope(points[: split + len(points) % 2])
        upper = _least_squares_slope(points[split:])
        if lower is None or upper is None or lower <= 0.0:
            continue
        # Clipped above: a game whose top of the ladder pulls away faster than its
        # bottom is real and good, but the ratio is unbounded and one lucky pairing
        # should not be allowed to dominate the corpus.
        ratios[kind] = round(min(max(upper / lower, 0.0), 2.5), 3)

    if not ratios:
        return 1.0, {}
    return sum(ratios.values()) / len(ratios), ratios


def _ladder_monotonicity(ratings: RatingTable) -> float:
    """The share of adjacent budget steps where more thinking rated higher.

    1.0 is a clean ladder. Around 0.5 is a coin flip, which means the roster is not
    ordering itself by strength and nothing downstream of that ordering -- the skill
    gradient, its persistence, the rating spread -- is about the game.

    Kept out of the weighted index on purpose. A game the instrument cannot read is not
    a bad game, and scoring it as one would be the same error as letting the cheap gate
    reject shatranj for being slow.
    """
    ladders = _budget_tiers([s.name for s in ratings.standings])
    rating_of = {s.name: s.rating for s in ratings.standings}

    steps = 0
    rising = 0
    for entries in ladders.values():
        if len(entries) < 2:
            continue
        for (_, lower), (_, higher) in zip(entries, entries[1:], strict=False):
            steps += 1
            if rating_of[higher] > rating_of[lower]:
                rising += 1
    return rising / steps if steps else 1.0


def _decisive_by_tier(records: list[GameRecord]) -> tuple[dict[str, float], dict[str, Any]]:
    """Decisive rate among the weakest searchers and among the strongest.

    Both tiers are same-strength pairings on purpose. A mixed pairing is decisive
    because one player is better, which says nothing about whether the *game* can be
    won by someone who knows what they are doing.
    """
    names = sorted({name for record in records for name in record.agents})
    ladders = _budget_tiers(names)
    if not ladders:
        return {}, {}

    kind = max(ladders, key=lambda k: len(ladders[k]))
    entries = ladders[kind]
    if len(entries) < 2:
        return {}, {}

    # Halves rather than the two extreme budgets. Restricting to the single weakest and
    # single strongest agent leaves only the self-play games of each -- six apiece on the
    # reference schedule, which cannot resolve a rate at all. Pooling the bottom and top
    # halves multiplies the sample by the square of the tier size for free.
    split = max(1, len(entries) // 2)
    tiers = {
        "low": {name for _, name in entries[:split]},
        "high": {name for _, name in entries[-split:]},
    }
    result: dict[str, float] = {}
    detail: dict[str, Any] = {"kind": kind}

    for label, tier in tiers.items():
        games = [r for r in records if r.agents[0] in tier and r.agents[1] in tier]
        detail[f"{label}_games"] = len(games)
        detail[f"{label}_agents"] = sorted(tier)
        if games:
            result[label] = sum(1 for r in games if r.winner is not None) / len(games)
            detail[f"{label}_decisive"] = round(result[label], 4)
    return result, detail


def _depth_by_agent(records: list[GameRecord]) -> dict[str, Any]:
    """How deep each rung actually saw, in completed plies per search.

    A diagnostic and deliberately not a component. Every skill measurement here reads a
    ladder of node budgets and treats a rung as the same player on every ruleset; a
    budget is not a fixed amount of search, because what it buys is a function of the
    branching factor, which is a property of the game being measured. At 6,400 nodes the
    ``dull`` control completes 8.5 plies and xiangqi completes 2.0, and the depth family
    compares them as though the rungs were the same players.

    Reported rather than gated, and reported on purpose. Both plausible actions -- a
    depth threshold, or a per-ruleset budget ladder -- trade a known confound for an
    unknown one, and a threshold chosen while looking at this table is a threshold
    chosen to catch one ruleset. See ``docs/bugs.md``.

    An agent that does not deepen contributes no searches and is absent from the table
    rather than present at zero: random and greedy have no depth, which is a different
    statement from "searched and completed no ply".
    """
    totals: dict[str, list[int]] = {}
    for record in records:
        for seat, name in enumerate(record.agents):
            searches = record.searches[seat]
            if not searches:
                continue
            entry = totals.setdefault(name, [0, 0])
            entry[0] += record.depth_sum[seat]
            entry[1] += searches

    return {
        name: {"mean_depth": round(depth / searches, 3), "searches": searches}
        for name, (depth, searches) in sorted(totals.items())
        if searches
    }


def _draw_slope(decisive: dict[str, float]) -> float:
    """How decisiveness holds up as the players get better.

    Two failures hide behind a single decisive rate measured at one strength. A game
    where the defence always holds once anyone can see two moves ahead collapses towards
    zero. A game whose result never stops being decisive is not converting skill into
    control -- it is chaotic, and the winner is whoever the position happened to favour.
    Reported as high-over-low, so 1.0 means "unchanged": roughly where a chaotic game
    sits, and a little above where a good one does.
    """
    low = decisive.get("low")
    high = decisive.get("high")
    if low is None or high is None or low <= 0.0:
        return 1.0
    return min(high / low, 2.0)


# --------------------------------------------------- engine-derived second pass


def held_type_indices(spec: dict[str, Any], n_types: int) -> list[int]:
    """Indices into ``moves_by_type`` of the types a player can hold.

    ``moves_by_type`` follows the compiled types: the spec's piece order, then one internal
    type per portal end (schema 19), which the compiler adds after the declared ones. A
    portal is held by no player, so neither its declared piece nor an end's type is
    counted (Q875, `docs/reference/portal-showcase-ideas.md` beside finding 1): until
    then the vector's length kept every index, and a portal game was charged a zero for
    its portal piece and for every fixed end, and credited a moved end as army. Any other
    length that does not match the spec keeps every index rather than guess which entry
    is which.
    """
    pieces = spec.get("pieces") or []
    rules = spec.get("rules")
    has_portals = isinstance(rules, dict) and bool(rules.get("portals"))
    if len(pieces) != n_types and not (has_portals and n_types > len(pieces)):
        return list(range(n_types))
    held = pieces_a_player_can_hold(spec)
    return [index for index, piece in enumerate(pieces) if piece.get("id") in held]


def _analysis_components(
    analysis: CorpusAnalysis, diagnostics: dict[str, Any], spec: dict[str, Any]
) -> dict[str, float]:
    """Components that needed positions, read back off the persisted analysis."""
    raw: dict[str, float] = {}

    # --- was there a decision to make --------------------------------------------
    # The share of the legal move list that would actually cost something. A move count
    # cannot tell a wide game from a game with a wide move list and one sane entry in
    # it; this can, and it is the thing ``mean_branching`` was standing in for.
    fractions = [
        near / legal
        for game in analysis.games
        for near, legal in zip(game.near_best, game.legal_counts, strict=False)
        if legal > 0
    ]
    if fractions:
        raw["decision_criticality"] = 1.0 - _mean(fractions)
        diagnostics["criticality_samples"] = len(fractions)

    # --- is the material doing anything ------------------------------------------
    totals: list[int] = []
    for game in analysis.games:
        if not totals:
            totals = [0] * len(game.moves_by_type)
        for index, count in enumerate(game.moves_by_type):
            totals[index] += count
    if sum(totals) > 0:
        # The entropy runs over the types a player can hold. A type only the wild phase
        # owns has no player to move it, so it is left out. A held type that never moves
        # stays in as a zero, which is how an idle army is charged.
        held = held_type_indices(spec, len(totals))
        counts = [totals[index] for index in held]
        # One piece type is a degenerate case, not a bad one: the single type is
        # trivially the one being used, so it scores full marks here. That is not a free
        # pass -- a game with one kind of piece has nothing to be good at, and it loses
        # on depth, where the geometric mean makes the loss decisive. Scoring it zero
        # instead would be measuring thinness twice, which is the exact mistake v2 made
        # with rule_mass.
        raw["piece_utilisation"] = _normalised_entropy(counts) if len(counts) > 1 else 1.0
        diagnostics["held_piece_types"] = len(counts)
        # Idle is judged on the army, the same held types the entropy reads: a portal is
        # the board's, so neither it nor an end counts as an idle piece, and the share is
        # of the army's moves (Q877; the board's own moves are in `moves_by_type`).
        played = sum(counts)
        diagnostics["idle_piece_types"] = (
            sum(1 for count in counts if count / played < IDLE_PIECE_SHARE) if played else 0
        )
        diagnostics["moves_by_type"] = totals

    # --- how much of it is the same position again --------------------------------
    # The direct measure of "back and forth and back and forth", and the one the engine
    # is best placed to give: the key is in hand at every replay step, so this costs a
    # set lookup per ply. A repetition *rule* only catches the third occurrence; this
    # catches a game that spends its life in a dozen positions without ever tripping it.
    revisited = sum(game.revisited_plies for game in analysis.games)
    walked = sum(game.walked_plies for game in analysis.games)
    if walked > 0:
        raw["position_revisit_rate"] = revisited / walked

    # --- do the armies ever meet ---------------------------------------------------
    # The failure of the video's versions 2 and 4: two players filling in a board and
    # never interacting. A game where almost no legal move takes anything is two
    # solitaires sharing a table.
    offered = [
        captures / legal
        for game in analysis.games
        for captures, legal in zip(game.capture_counts, game.legal_counts, strict=False)
        if legal > 0
    ]
    if offered:
        raw["capture_availability"] = _mean(offered)

    # --- are these pieces or is this scenery ---------------------------------------
    # `piece_utilisation` above is an entropy over piece *types* and cannot see five
    # rooks sitting in the back while a sixth does all the work. This counts instances.
    # It is not here because a mutant defeated the entropy -- fixing a component
    # against the corpus that beat it is fitting the index to the corpus -- but because
    # a designer playtesting a game this corpus has never seen named the failure, and
    # named it per instance.
    starting = sum(game.starting_instances for game in analysis.games)
    moved = sum(game.moved_instances for game in analysis.games)
    if starting > 0:
        raw["instance_participation"] = moved / starting

    # --- what the rounds' resolutions did (schema 26, Q918d) ------------------------
    # Counted by the pass off the core's events; scored by nothing. What a game in rounds
    # is made of, beside the components: how often the two moves met, how often a capture
    # found its target gone, how often the two crossed. `ROUND_COUNTS` names them.
    rounds = [game.rounds for game in analysis.games if game.rounds]
    if rounds:
        tallies = {key: sum(found.get(key, 0) for found in rounds) for key in rounds[0]}
        played = tallies.get("rounds", 0)
        shares = {
            f"{key}_per_round": round(tallies[key] / played, 4)
            for key in ("collisions", "void_moves", "swaps")
            if played and key in tallies
        }
        diagnostics["rounds"] = {**tallies, **shares}

    # --- what the spy's phases did (schema 27, Q935d) -----------------------------
    # Counted by the pass (`SPY_COUNTS`), scored by nothing: how often a spy fell, when,
    # how, and how long each phase lasted. The study's inputs (`docs/plan-spy.md` §5.3).
    spies = [game.spy for game in analysis.games if game.spy is not None]
    if spies:
        diagnostics["spy"] = spy_summary(spies)

    # --- does one mistake end it --------------------------------------------------
    if analysis.blunders:
        recovered = sum(1 for trial in analysis.blunders if trial.recovered)
        raw["blunder_recoverability"] = recovered / len(analysis.blunders)
        diagnostics["blunder_trials"] = len(analysis.blunders)
        diagnostics["blunder_changed_result"] = round(
            sum(1 for trial in analysis.blunders if trial.changed) / len(analysis.blunders),
            4,
        )

    return raw


def spy_summary(spies: list[dict[str, Any]]) -> dict[str, Any]:
    """A spy corpus's phases in a few numbers, from each game's `SPY_COUNTS`: the share of
    games in which a spy fell and in which both did (``both_at_once`` counts the games where
    one step took both, E4), the median ply of the first loss, the share that played the
    reply turn, the mean length of each phase, and how the spies were lost."""
    games = len(spies)
    lost = [
        [ply for ply in (spy["lost_ply_0"], spy["lost_ply_1"]) if ply is not None]
        for spy in spies
    ]
    firsts = sorted(min(plies) for plies in lost if plies)
    middle = len(firsts) // 2
    median = None
    if firsts:
        median = float(firsts[middle]) if len(firsts) % 2 else (
            (firsts[middle - 1] + firsts[middle]) / 2
        )
    how = Counter(
        spy[f"lost_how_{player}"]
        for spy in spies
        for player in (0, 1)
        if spy[f"lost_how_{player}"] is not None
    )
    return {
        "games": games,
        "spy_lost_share": round(len(firsts) / games, 4),
        "both_lost_share": round(sum(1 for plies in lost if len(plies) == 2) / games, 4),
        "both_at_once": sum(1 for plies in lost if len(plies) == 2 and plies[0] == plies[1]),
        "first_loss_median_ply": median,
        "reply_turn_share": round(sum(spy["reply_turns"] for spy in spies) / games, 4),
        "mean_alternating_plies": round(sum(s["alternating_plies"] for s in spies) / games, 2),
        "mean_announced_rounds": round(sum(s["announced_rounds"] for s in spies) / games, 2),
        "mean_blind_rounds": round(sum(s["blind_rounds"] for s in spies) / games, 2),
        "lost_how": dict(sorted(how.items())),
    }


def _normalised_entropy(counts: list[int]) -> float:
    """Shannon entropy over a count vector, against the maximum its size allows.

    Normalising by ``log(n)`` is what makes the number comparable between a game with
    three piece types and one with nine: it answers "is the material being used?"
    rather than "how much material is there?", which the elegance family already covers.
    """
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return min(1.0, entropy / math.log(len(counts)))
