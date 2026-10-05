"""What the API actually accepts, asked of the API.

**This is the test that did not exist**, and its absence is the whole history of this
corner of the project: three of the four response schemas were refused for their entire
lives, every judging call answered 400, and the offline suite passed throughout, because
a stub client is handed a schema and never sends a request. `pyproject.toml` has
declared an ``llm`` marker for "requires Anthropic API credentials" since before any of
that, and nothing had ever been marked with it.

It is opt-in twice over -- ``CHESS_VARIANT_SANDBOX_LIVE_API=1`` *and* resolvable credentials -- because
a test that spends money on a routine ``uv run pytest`` is a worse defect than the one it
catches. Run it deliberately:

    CHESS_VARIANT_SANDBOX_LIVE_API=1 uv run pytest tests/test_llm_live.py -m llm

The bill is a couple of cents. Each acceptance probe is one call with a small output cap:
what is being checked is whether the *request* is accepted, which the API decides before
it generates anything, so the reply is allowed to be cut off and that counts as a pass.
The comparison at the end is the exception and is a real judgement, on the cheap model,
with no transcripts.
"""

from __future__ import annotations

import copy
import os

import pytest

from chess_variant_sandbox.llm.client import (
    JUDGE_MODEL,
    PROPOSAL_MODEL,
    Claude,
    LLMUnavailable,
)

pytestmark = pytest.mark.llm

# Small, valid, and not a chess variant -- the same shape the offline end-to-end file
# uses, so a live failure can be read against an offline pass.
SPEC = {
    "schema_version": 1,
    "name": "lancers",
    "description": "A small skirmish game.",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "name": "captain", "royal": True, "value": 0, "moves": [
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
        ]},
        {"id": "L", "name": "lancer", "value": 5, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
        ]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "L", "squares": ["a1", "e1"]},
    ]},
    "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
              "no_progress_draw": 40},
}


@pytest.fixture(autouse=True)
def _live_only() -> None:
    if os.environ.get("CHESS_VARIANT_SANDBOX_LIVE_API") != "1":
        pytest.skip("set CHESS_VARIANT_SANDBOX_LIVE_API=1 to spend money on the real API")
    if not Claude.is_configured():
        pytest.skip("no Anthropic credentials resolved")


def _requests_this_project_makes() -> list[tuple[str, str, dict | None]]:
    """Every (stage, model, response schema) triple that reaches the API, as it reaches it.

    Paired with the model each is actually sent to rather than crossed with both: the
    question is whether the requests this project makes are accepted, and a pairing
    nothing sends is a pairing nothing needs to pay for.

    `propose_ruleset`'s schema is ``None`` because that is what it sends. The rule spec
    schema is 67 optional properties against a grammar-compiler limit of 24, so it has no
    response-format form at all and the contract is stated in the prompt -- which is what
    this row then checks: that an unconstrained call is accepted.
    """
    from chess_variant_sandbox.llm.client import constrainable
    from chess_variant_sandbox.llm.compare import COMPARISON_SCHEMA
    from chess_variant_sandbox.llm.heuristics import DEFAULT_K, heuristic_schema
    from chess_variant_sandbox.llm.judge import VERDICT_SCHEMA
    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

    spec_schema = rule_spec_schema()
    return [
        ("propose_ruleset", PROPOSAL_MODEL, spec_schema if constrainable(spec_schema) else None),
        ("propose_heuristics", PROPOSAL_MODEL, heuristic_schema(DEFAULT_K)),
        ("judge_ruleset", JUDGE_MODEL, VERDICT_SCHEMA),
        ("compare_rulesets", JUDGE_MODEL, COMPARISON_SCHEMA),
    ]


@pytest.mark.parametrize(
    ("stage", "model", "schema"),
    _requests_this_project_makes(),
    ids=[stage for stage, _, _ in _requests_this_project_makes()],
)
def test_the_api_accepts_the_request_this_stage_sends(
    stage: str, model: str, schema: dict | None
) -> None:
    """The request is built by the shipping code, and only the prompt is a stub.

    A 400 arrives before any generation, so a reply cut off by the output cap is a
    *pass*: the schema and the parameters were accepted, which is the only thing being
    asked. Anything else is the failure this file exists to surface, and its message is
    the API's own -- which is how the grammar-compiler limits were found, since no
    published list mentions them.
    """
    client = Claude(model=model, effort="medium")
    try:
        client.structured(
            system="Answer with the smallest valid JSON object the schema allows.",
            user="Produce one minimal example. Return only the JSON object.",
            schema=schema,
            max_tokens=1024,
        )
    except LLMUnavailable as error:
        # An unconstrained reply may also simply not be JSON at a 1,024-token cap, which
        # says nothing about whether the request was accepted.
        acceptable = ("cut off at max_tokens", "not JSON")
        assert any(reason in str(error) for reason in acceptable), f"{stage} on {model}: {error}"


def _probe(client: Claude, system: list[dict], user: str) -> None:
    """One call whose reply does not matter, made for its usage meters alone.

    Same tolerance as the acceptance probes above: a cut-off or non-JSON reply is a
    successful call, because what is being read is `client.usage` and the API has
    already billed and reported the input side by the time generation begins.
    """
    try:
        client.structured(system=system, user=user, schema=None, max_tokens=1024)
    except LLMUnavailable as error:
        assert any(r in str(error) for r in ("cut off at max_tokens", "not JSON")), error


def test_the_proposal_prefix_is_written_once_and_then_read() -> None:
    """`cached_system` on the call it exists for, measured rather than argued.

    The prefix is the real one -- brief, preamble and the whole DSL schema -- sent twice,
    byte-identical. What is being checked is the claim the 1-hour TTL rests on: that this
    prefix is above `claude-opus-5`'s minimum, that the marker opens an entry, and that
    the second call reads it back instead of opening another.

    Two calls cannot prove an entry survives an hour -- nothing here can, short of
    waiting one -- so this checks the half that is checkable, and the run report checks
    the other half on real generations.
    """
    import json

    from chess_variant_sandbox.llm.client import cached_system
    from chess_variant_sandbox.llm.propose import DESIGN_BRIEF, SCHEMA_PREAMBLE
    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

    stable = "\n\n".join(
        [
            DESIGN_BRIEF,
            SCHEMA_PREAMBLE,
            "The schema:",
            json.dumps(rule_spec_schema(), indent=2),
        ]
    )
    system = cached_system(PROPOSAL_MODEL, stable)
    assert "cache_control" in system[0]

    client = Claude(model=PROPOSAL_MODEL, effort="low")
    _probe(client, system, "Return {}. Nothing else.")
    written = client.usage.cache_write_tokens
    assert written > 0, "the marker opened no entry; the prefix is below the minimum"

    _probe(client, system, "Return {}. Nothing else.")
    assert client.usage.cache_read_tokens > 0, "the second call did not read the entry"
    # And it read the entry rather than opening a second one.
    assert client.usage.cache_write_tokens == written


def test_a_prefix_below_the_minimum_is_not_cached_and_does_not_say_so() -> None:
    """The measurement `CACHE_MINIMUM` is built on, and the reason it is a table.

    `llm/compare.py` and `llm/judge.py` carried a `cache_control` marker on prefixes of
    roughly 270 and 800 tokens for their whole lives. This sends the judge's own brief
    with the marker forced back on -- which is what the shipping code now refuses to do
    -- and asserts the silence: no error, no warning, and nothing written. That silence
    is the entire defect, and it is not something an offline test can observe.
    """
    from chess_variant_sandbox.llm.judge import JUDGE_BRIEF

    forced = [
        {"type": "text", "text": JUDGE_BRIEF, "cache_control": {"type": "ephemeral"}}
    ]
    client = Claude(model=JUDGE_MODEL)
    _probe(client, forced, "Return {}. Nothing else.")

    assert client.usage.calls == 1, "the request was refused, which is not the claim"
    assert client.usage.cache_write_tokens == 0, (
        f"{JUDGE_MODEL} opened an entry for a prefix below its stated minimum; "
        "CACHE_MINIMUM is wrong"
    )


def test_the_pairwise_judge_makes_a_comparison() -> None:
    """The bug in `docs/bugs.md`, closed or not closed, in one call each way.

    `loop/preference.py` fits a Bradley-Terry ordering from these and had never been
    given one to fit. Two rulesets and no transcripts: the judge is told to judge the
    rules, and this asks only whether it can answer at all.
    """
    from chess_variant_sandbox.llm.compare import compare_rulesets

    other = copy.deepcopy(SPEC)
    other["name"] = "lancers II"
    other["board"] = {"width": 6, "height": 6}

    comparison = compare_rulesets(SPEC, other, [], [], client=Claude(model=JUDGE_MODEL))

    from chess_variant_sandbox.spec.canonical import spec_hash

    assert comparison.first == spec_hash(SPEC)
    assert comparison.second == spec_hash(other)
    assert comparison.winner in (comparison.first, comparison.second, None)
    # Two answers, one per order, each with its reasoning. A flip is a legitimate
    # outcome -- it is recorded as a tie -- but silence is not.
    assert len(comparison.reasons) == 2
    assert all(reason.strip() for reason in comparison.reasons)
