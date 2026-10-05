"""Heuristic sets that need no model: the anchor, a hand-written spread, and the null.

Three jobs, and the third is the one that decides whether the LLM stage is worth having.

**The anchor.** Index 0 of every set is the built-in evaluator's own vector, never
proposed by anything. It is one point per ruleset that owes nothing to a sample, which
is what keeps a corpus interpretable even if every proposed heuristic turns out to be
junk -- and it is the only heuristic whose behaviour is already understood, because it
is the objective the whole index was calibrated on.

**The hand-written spread.** A small set of plans a person would actually state, each
reading rules the schema declares. Gated on the ruleset declaring the rule, so a set for
chess does not contain "claim ground" and a set for a territory game does.

**The null.** `random_set` draws vectors uniformly. It costs no model call and it
answers the question the whole LLM stage rests on: if proposed heuristics do not beat
random vectors, the proposing half buys nothing and only the vocabulary does. Reported
beside every real set rather than mentioned once.
"""

from __future__ import annotations

import random
from typing import Any

from chess_variant_sandbox.agents.features import (
    VOCABULARY,
    builtin_vector,
    feature_context,
)
from chess_variant_sandbox.agents.heuristics import Heuristic, HeuristicSet
from chess_variant_sandbox.engine.compile import CompiledSpec

# A plan is only offered where the rule it reads is declared. Each entry is
# (name, sentence, {term: weight}, does this ruleset have anything to say about it).
_PLANS: tuple[tuple[str, str, dict[str, float], str], ...] = (
    (
        "trade-up",
        "Win material and keep the pieces that can take the most.",
        {"material": 1.0, "mobility": 0.05},
        "always",
    ),
    (
        "shed",
        "Give material away as fast as possible.",
        {"material": -1.0, "mobility": 0.05},
        "always",
    ),
    (
        "swarm",
        "Push everything forward and keep the pieces active.",
        {"advance": 1.0, "mobility": 0.3, "material": 0.2},
        "always",
    ),
    (
        "hold-the-middle",
        "Occupy the centre and keep enough material to stay there.",
        {"centrality": 1.0, "material": 0.4},
        "always",
    ),
    (
        "hunt-the-king",
        "Keep the opponent's royal piece under threat and yours out of trouble.",
        {"royal_exposure": 1.0, "material": 0.5},
        "royal",
    ),
    (
        "claim-ground",
        "Take and hold as much of the board as possible.",
        {"territory_margin": 1.0, "advance": 0.2},
        "territory",
    ),
    (
        "count-at-the-clock",
        "Build a lead on the count that decides the game when time runs out.",
        {"score_urgency": 1.0, "score_margin": 0.3},
        "scoring",
    ),
    (
        "race",
        "Get a piece to a winning square as directly as possible.",
        {"reach_progress": 1.0, "material": 0.2},
        "reach",
    ),
    (
        "check-and-check-again",
        "Give check as often as possible.",
        {"check_margin": 1.0, "royal_exposure": 0.5},
        "check_limit",
    ),
    (
        "starve-them",
        "Take the pieces they cannot afford to lose.",
        {"extinction_margin": 1.0, "material": 0.3},
        "extinction",
    ),
    (
        "hoard",
        "Capture into hand and keep the reserve large.",
        {"hand_material": 1.0, "material": 0.5},
        "drops",
    ),
    (
        "promote",
        "Get promotable pieces onto the squares where they promote.",
        {"promotion_occupancy": 1.0, "advance": 0.4},
        "promotion",
    ),
    (
        "eat-what-you-can",
        "Value a piece by what it can still take on the board right now.",
        {"prey_weight": 1.0, "material": 1.0},
        "capture_matrix",
    ),
)


def _declares(spec: CompiledSpec, requirement: str) -> bool:
    """Whether a ruleset has anything for this plan to be about."""
    if requirement == "always":
        return True
    if requirement == "royal":
        return bool(spec.royal_types)
    if requirement == "territory":
        return spec.territory is not None
    if requirement == "scoring":
        return spec.scoring is not None
    if requirement == "reach":
        return spec.reach is not None
    if requirement == "check_limit":
        return spec.check_limit > 0
    if requirement == "extinction":
        return spec.extinction is not None
    if requirement == "drops":
        return spec.drops is not None
    if requirement == "promotion":
        return bool(spec.promotions)
    if requirement == "capture_matrix":
        return spec.has_capture_matrix
    raise ValueError(f"unknown plan requirement {requirement!r}")


def _vector(terms: dict[str, float]) -> tuple[float, ...]:
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return tuple(weights)


def anchor(spec: CompiledSpec) -> Heuristic:
    """Material and mobility, as a heuristic. Index 0 of every set, always.

    **This was the whole built-in objective when it was written, and is its capture half
    now.** Stage 17A gave the built-in evaluator goal terms for territory, for a score at
    the clock, for a race, for a check count and for extinction, and this vector does not
    carry them -- the vocabulary has all five, but reproducing their weighting exactly
    would mean matching a normalisation that was derived for a different purpose, and a
    vector that claims to be the built-in objective and is not would be worse than one
    that says what it is.

    Nothing downstream depends on the two being identical. `heuristic_agreement` plays
    against the *real* `greedy` and `minimax`, goal terms and all, so it is unaffected;
    what this anchors is a headroom reading that owes nothing to any sample, and
    "material and mobility" is a perfectly good fixed point for that.

    The name and the plan are outside the digest, so saying this more accurately changed
    no heuristic's identity and no committed store.
    """
    return Heuristic(
        name="material-and-mobility",
        plan="Win material, and keep your options open.",
        weights=builtin_vector(feature_context(spec)),
    )


def handwritten_set(spec: CompiledSpec, k: int = 6) -> HeuristicSet:
    """The anchor plus the plans this ruleset's rules can be about.

    Deterministic and free. Not a substitute for a proposal -- a fixed list cannot say
    anything a person did not already think of, which is exactly what a model is for --
    but it is the control every proposal has to beat, and it is what makes the pass
    runnable with no credentials at all.

    **A plan about a rule this game declares is chosen before a plan every game gets.**
    The four ``always`` plans used to take four of the five slots at ``k = 6`` because
    they come first in ``_PLANS``, so a game got exactly one plan about its own rules,
    whichever of them the list happened to name first. A capture-matrix game with a
    royal piece or a territory rule never reached ``eat-what-you-can``, the one plan that
    reads the matrix, and a check-count, hill, hand or promotion rule met the same fate
    behind the rules listed above it (Q162). Now the declared-rule plans are chosen first,
    in list order, and the ``always`` plans fill what is left, the last of them dropping
    first. ``k`` stays what it was, because the null control and the proposer are sized
    by the same ``k`` and a set's median moves with its size.

    The chosen plans are then written in ``_PLANS`` order, so a set whose members did not
    change is byte-identical to the one written before this rule.
    """
    offered = [
        (position, requirement, Heuristic(name=name, plan=plan, weights=_vector(terms)))
        for position, (name, plan, terms, requirement) in enumerate(_PLANS)
        if _declares(spec, requirement)
    ]
    # Stable, so each half keeps the list's own order.
    by_priority = sorted(offered, key=lambda entry: entry[1] == "always")

    heuristics = [anchor(spec)]
    chosen: list[tuple[int, Heuristic]] = []
    for position, _, candidate in by_priority:
        if len(heuristics) + len(chosen) >= k:
            break
        # Two plans that reduce to the same vector are one player; the digest says so.
        if any(existing.digest == candidate.digest for existing in heuristics) or any(
            existing.digest == candidate.digest for _, existing in chosen
        ):
            continue
        chosen.append((position, candidate))
    heuristics.extend(candidate for _, candidate in sorted(chosen, key=lambda c: c[0]))
    return HeuristicSet(
        ruleset_hash=spec.hash, heuristics=tuple(heuristics), model="hand-written"
    )


def random_set(spec: CompiledSpec, k: int = 6, seed: int = 0) -> HeuristicSet:
    """The null control: the anchor, then uniform random vectors.

    If the proposed heuristics do not beat these, the proposing half of the stage is
    dead weight and only the vocabulary is doing work. Costs no model call, so there is
    no reason ever to report a real set without it.
    """
    rng = random.Random(seed)
    heuristics = [anchor(spec)]
    while len(heuristics) < k:
        weights = tuple(round(rng.uniform(-1.0, 1.0), 4) for _ in VOCABULARY)
        if not any(weights):
            continue
        candidate = Heuristic(
            name=f"random-{len(heuristics)}",
            plan="No plan; a vector drawn at random, as a control.",
            weights=weights,
        )
        if any(existing.digest == candidate.digest for existing in heuristics):
            continue
        heuristics.append(candidate)
    return HeuristicSet(
        ruleset_hash=spec.hash, heuristics=tuple(heuristics), model="random-null"
    )


def describe_plans() -> list[dict[str, Any]]:
    """The hand-written plans, for a report or an interface."""
    return [
        {"name": name, "plan": plan, "terms": terms, "requires": requirement}
        for name, plan, terms, requirement in _PLANS
    ]
