"""Asking a model how it would play one game, as weights over a fixed vocabulary.

A different question from the one `propose.py` asks, and the difference is what makes
this stage safe. The proposer is asked *how do I make this game good*, which is one step
from the objective function and is guarded accordingly. This stage is asked *how would
you play this game*, which is a question about the rules in front of it and carries no
opinion about whether they are worth having.

What the model is shown: the rules in prose, and the vocabulary with its glosses. What
it is never shown: the index, its components, its weights, its curves, any score, and
any measurement of this or any other ruleset. The prompt is a pure function of the spec,
asserted by a test -- which is what makes it constant across candidates and therefore
free of any gradient the way the design brief is.

The proposal is untrusted input and is validated before anything plays it, exactly as a
rule spec is: a vector of the wrong length, a weight that is not finite, or an all-zero
vector is rejected by name, with one corrective retry carrying the specific errors.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.agents.features import MAX_ABS_WEIGHT, VOCABULARY, VOCABULARY_VERSION
from chess_variant_sandbox.agents.heuristics import Heuristic, HeuristicError, HeuristicSet
from chess_variant_sandbox.agents.seeds import anchor
from chess_variant_sandbox.engine.compile import CompiledSpec
from chess_variant_sandbox.llm.client import PROPOSAL_MODEL, Claude, cached_system
from chess_variant_sandbox.spec.describe import describe_spec

# How many heuristics to ask for, beyond the built-in anchor that every set carries at
# index 0. Fixed, and recorded in the store: any statistic taken over a set -- a median,
# and a maximum especially -- is a different quantity at a different k, so two campaigns
# at different k must not share a table.
DEFAULT_K = 5

HEURISTIC_BRIEF = """\
You are given the complete rules of a two-player board game, and a fixed list of things
that can be measured about a position. Your job is to write down several different ways
somebody might try to play this game.

Each way of playing is a short sentence a person could follow -- a plan, a rule of thumb
-- together with a number for each measurement saying how much that plan cares about it.
A positive number means the plan wants that quantity larger, a negative number means
smaller, and zero means the plan ignores it. Only the ratios between the numbers matter,
so keep them in a small range; anything beyond {max_weight} in either direction is
rejected. At least one number must be non-zero.

Write plans that genuinely disagree with each other. A set in which every plan is a
slight variation on winning material tells nobody anything. Where the rules offer a way
to win that is not capture -- getting somewhere, holding ground, running a counter down,
running the opponent out of a piece -- at least one plan should be built around it, and
one plan should deliberately pursue something the others neglect.

Several of the measurements only mean something when the rules declare the matching
feature; on a game without that feature they are always zero, and weighting them does
nothing. Read the rules and use the ones that apply.

Say what each plan is trying to do in the sentence. The sentence is read by people, not
by the program, and a plan nobody can state is not a plan."""


def heuristic_schema(k: int = DEFAULT_K) -> dict[str, Any]:
    """The model's contract, generated from the vocabulary rather than written out.

    Same construction as ``spec/jsonschema.rule_spec_schema`` and for the same reason:
    ``rules.turn`` shipped built, tested, documented and unreachable by the only thing
    that writes new rulesets, and nothing failed. A term the schema cannot express
    cannot be proposed at all, so the schema is derived from ``VOCABULARY`` and a test
    asserts the two agree.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["heuristics"],
        "properties": {
            "heuristics": {
                "type": "array",
                "minItems": k,
                "maxItems": k,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "plan", "weights"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "maxLength": 40,
                            "description": "a short label for this way of playing",
                        },
                        "plan": {
                            "type": "string",
                            "maxLength": 200,
                            "description": "one sentence a person could follow",
                        },
                        "weights": {
                            "type": "array",
                            "minItems": len(VOCABULARY),
                            "maxItems": len(VOCABULARY),
                            "items": {
                                "type": "number",
                                "minimum": -MAX_ABS_WEIGHT,
                                "maximum": MAX_ABS_WEIGHT,
                            },
                            "description": (
                                "one number per measurement, in the order listed: "
                                + ", ".join(VOCABULARY)
                            ),
                        },
                    },
                },
            }
        },
    }


def heuristic_brief() -> str:
    """The stable half of the prompt. A pure function of the vocabulary."""
    from chess_variant_sandbox.agents.heuristics import vocabulary_prose

    return (
        HEURISTIC_BRIEF.format(max_weight=MAX_ABS_WEIGHT)
        + "\n\nThe measurements, in the order the numbers must be given:\n\n"
        + vocabulary_prose()
    )


def propose_heuristics(
    spec: dict[str, Any],
    compiled: CompiledSpec,
    k: int = DEFAULT_K,
    client: Claude | None = None,
) -> HeuristicSet:
    """Ask for ``k`` ways of playing this game, and return them with the anchor first.

    The built-in evaluator's own vector is index 0 and is never proposed. It is the one
    point in every set that owes nothing to a sample, which is what keeps the corpus
    interpretable if a whole set comes back useless.
    """
    client = client or Claude(model=PROPOSAL_MODEL, effort="medium")
    schema = heuristic_schema(k)
    # The rules go after the cache breakpoint; the brief and the vocabulary before it.
    system = cached_system(client.model, heuristic_brief())
    user = f"The rules:\n\n{describe_spec(spec)}\n\nWrite {k} ways of playing this game."

    errors: list[str] = []
    for attempt in (1, 2):
        payload = client.structured(system=system, user=user, schema=schema)
        heuristics, errors = _validate(payload, k)
        if not errors:
            return HeuristicSet(
                ruleset_hash=compiled.hash,
                heuristics=(anchor(compiled), *heuristics),
                model=client.model,
            )
        if attempt == 1:
            user = (
                f"{user}\n\nYour previous attempt was rejected:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n\nReturn a corrected set. Fix every listed problem."
            )
    raise HeuristicError(
        "the model did not produce a usable set of heuristics after one correction: "
        + "; ".join(errors)
    )


def _validate(payload: dict[str, Any], k: int) -> tuple[list[Heuristic], list[str]]:
    """Model output is untrusted. Validate before anything plays it."""
    errors: list[str] = []
    heuristics: list[Heuristic] = []
    raw = payload.get("heuristics")
    if not isinstance(raw, list):
        return [], ["the response has no list of heuristics"]
    if len(raw) != k:
        errors.append(f"expected {k} heuristics, got {len(raw)}")

    seen: set[str] = set()
    for index, item in enumerate(raw):
        try:
            heuristic = Heuristic(
                name=str(item.get("name", f"plan-{index}")),
                plan=str(item.get("plan", "")),
                weights=tuple(float(w) for w in item["weights"]),
            )
        except (HeuristicError, KeyError, TypeError, ValueError) as error:
            errors.append(f"heuristic {index}: {error}")
            continue
        # Two plans that reduce to one vector are one player, and `k` would silently
        # shrink. Rejected by name so the retry can fix it rather than quietly
        # delivering a smaller set.
        if heuristic.digest in seen:
            errors.append(
                f"heuristic {index} ({heuristic.name!r}) is the same player as an "
                "earlier one; only the ratios between weights matter, so scaling a "
                "vector does not make a new plan"
            )
            continue
        seen.add(heuristic.digest)
        heuristics.append(heuristic)

    if not errors and len(heuristics) < k:  # pragma: no cover - covered by the checks above
        errors.append(f"only {len(heuristics)} usable heuristics of {k}")
    return heuristics, errors


def vocabulary_version() -> int:
    return VOCABULARY_VERSION
