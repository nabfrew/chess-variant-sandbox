"""Anthropic client wrapper for the two things this project asks a model to do.

Kept deliberately thin. The interesting decisions are in the prompts and in what the
model is and is not shown; this file only makes the call and gets typed JSON back.

Two model choices, for two different jobs:

* **Proposals and critiques** run on Claude Opus 5 with adaptive thinking. There are
  few of these per generation and each one sets the direction of an entire tournament,
  so this is the wrong place to economise.
* **Judging** runs on Claude Haiku 4.5. There are many judgements per generation --
  one per sampled game, plus one per ruleset -- and judge volume is where the cost of
  a long design run actually goes.

The ``anthropic`` package is an optional dependency, imported lazily. The engine,
tournament and metrics all work without it; only the proposal and judging stages need
credentials, and a sweep over hand-written rulesets should not require an API key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROPOSAL_MODEL = "claude-opus-5"
JUDGE_MODEL = "claude-haiku-4-5"

# Every call here is streamed, so the ceiling is not bounded by an HTTP timeout, and a
# ceiling is not a reservation -- an unused token costs nothing. What it does bound is
# the failure below: adaptive thinking spends tokens before the JSON starts, so a cap
# tight enough to be reached truncates the reply mid-object, and a truncated reply is a
# whole generation thrown away rather than a slow one.
MAX_OUTPUT_TOKENS = 64_000


# File names checked, in order, relative to the project root.  Both are git-ignored.
_KEY_FILE_NAMES = ("api-key", "api_key")


def _find_key_file() -> str | None:
    """Return the API key from a local key file, or *None*."""
    root = Path(__file__).resolve().parents[2]  # chess_variant_sandbox/llm/client.py -> project root
    for name in _KEY_FILE_NAMES:
        path = root / name
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    return None


def _ensure_key_env() -> None:
    """If no env-var key is set but a key file exists, inject it into the environment.

    Done once, before the SDK client is constructed, so that ``anthropic.Anthropic()``
    picks it up through its normal path.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    key = _find_key_file()
    if key is not None:
        os.environ["ANTHROPIC_API_KEY"] = key


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached. Never silently degraded.

    A design loop that quietly skipped the proposal step would keep running and keep
    producing generations, and the only symptom would be that nothing new ever
    appeared. Failing loudly is the whole point.
    """


# --------------------------------------------------------------- the wire request
#
# What a model accepts on the request *itself*, which is a second wire the schema layer
# below knows nothing about and which failed in exactly the same way: `structured` sent
# `thinking={"type": "adaptive"}` and an `effort` to every model unconditionally, and
# `JUDGE_MODEL` accepts neither. So `llm/judge.py` and `llm/compare.py` answered 400 on
# every call, the pairwise judge had never produced a comparison, and no offline test
# noticed -- a stub client is handed a schema and never a request.
#
#   claude-haiku-4-5 + thinking={"type": "adaptive"}
#       -> 400 "adaptive thinking is not supported on this model"
#
# **That line is measured**; it is the entry this table was written to close, and it was
# found by pointing the cheapest model at a real call while checking something else.
# The `effort` column is not: `output_config.effort` is documented as Opus 4.5, Opus 4.6,
# Sonnet 4.6 and later, so Haiku 4.5 is expected to refuse it too, and this repository
# has no way to confirm that without a key. It is withheld on the same row either way,
# because a judge does not need it and an unnecessary parameter is a way to be wrong.
#
# The judge is left with no thinking at all rather than given Haiku's older
# `budget_tokens` form. A budget must be under `max_tokens`, `compare_rulesets` calls at
# 2,000, and the minimum budget is 1,024 -- so what fits is a token count too small to
# change an answer, bought at the price of the one parameter this file most wants to
# stop guessing about. A judge that needs to think belongs on a model that thinks
# adaptively, which is a stage of its own and a cost decision rather than a bug fix.
#
# Unknown models are assumed current, because that is what a model added after this line
# was written will be, and because being wrong that way is a 400 that names the parameter
# rather than a run that quietly stops thinking. `tests/test_loop.py` asserts that every
# model this project actually names has a row here, so "we pointed a stage at a new
# model" is a failing test rather than a bill.


@dataclass(frozen=True)
class ModelWire:
    """Which of the two thinking controls one model accepts on the request."""

    adaptive_thinking: bool
    effort: bool


CURRENT_WIRE = ModelWire(adaptive_thinking=True, effort=True)

MODEL_WIRE: dict[str, ModelWire] = {
    PROPOSAL_MODEL: CURRENT_WIRE,
    JUDGE_MODEL: ModelWire(adaptive_thinking=False, effort=False),
}


def request_options(
    model: str, effort: str, schema: dict[str, Any] | None
) -> dict[str, Any]:
    """The parameters ``model`` accepts, and none that it does not.

    Split out of ``structured`` so the request can be inspected without making one.
    That is the whole difficulty with this class of defect: the offending request was
    built inside the one method no test can call, so nothing offline ever saw it.

    ``schema`` of ``None`` asks for no response format at all -- the reply must still be
    JSON, and the caller is stating the contract some other way. See `constrainable`.
    """
    wire = MODEL_WIRE.get(model, CURRENT_WIRE)

    output_config: dict[str, Any] = {}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": wire_schema(schema)}
    if wire.effort:
        output_config["effort"] = effort

    options: dict[str, Any] = {"output_config": output_config}
    if wire.adaptive_thinking:
        options["thinking"] = {"type": "adaptive"}
    return options


# ------------------------------------------------- what the grammar compiler will take
#
# A response format is compiled into a decoding grammar, and the compiler has size limits
# that have nothing to do with which keywords are legal. **Both of these are measured**,
# with the API's own words:
#
#   67 properties outside `required`
#       -> 400 "Schemas contains too many optional parameters (67), which would make
#               grammar compilation inefficient ... (limit: 24)"
#   101 properties written as `anyOf` or a type array
#       -> 400 "Schemas contains too many parameters with union types (101 ...). This
#               causes exponential compila[tion]"
#
# The second is what the obvious escape from the first runs into: making every property
# required and allowing each to be null is how a schema says "optional" without an
# optional property, and it trades 67 optionals for 101 unions.
#
# **So `rule_spec_schema` cannot be a response schema, and this is not a bug to fix.** It
# is 67 optional properties because that is what a rule DSL *is*: a game states the rules
# it has and says nothing about the rest, and `rules` alone accounts for 48 of them.
# Cutting it to 24 would mean deciding which two thirds of the vocabulary a proposal is
# allowed to reach, which is the defect `spec/jsonschema.py` exists to prevent.
#
# `llm/propose.py` therefore asks for no response format and states the contract in the
# prompt, where the whole schema already was. That costs the guarantee that a reply
# parses against the schema and costs nothing else, because what protects the pipeline is
# the validator that reads the reply -- the argument `wire_schema` was already making,
# now forced rather than chosen. The three schemas small enough to compile still use the
# format, and `tests/test_loop.py` asserts which are which, so a schema that grows past
# the limit is a failing test rather than a run that stops on its first generation.
MAX_OPTIONAL_PARAMETERS = 24


def constrainable(schema: dict[str, Any]) -> bool:
    """Whether ``schema`` is small enough for the API to compile into a grammar."""
    return _optional_parameters(wire_schema(schema)) <= MAX_OPTIONAL_PARAMETERS


def _optional_parameters(node: Any) -> int:
    """Properties not in ``required``, over the whole schema, as the API counts them."""
    if isinstance(node, list):
        return sum(_optional_parameters(item) for item in node)
    if not isinstance(node, dict):
        return 0

    found = 0
    if node.get("type") == "object":
        required = set(node.get("required", []))
        found += sum(1 for name in node.get("properties", {}) if name not in required)
    for key, value in node.items():
        if key in SCHEMA_MAPS and isinstance(value, dict):
            found += sum(_optional_parameters(sub) for sub in value.values())
        elif key in SCHEMA_VALUES:
            found += _optional_parameters(value)
    return found


# ---------------------------------------------------------------- the wire schema
#
# Constraints `output_config.format.schema` refuses. **Measured against the live API,
# not recalled**, because being wrong here is invisible from inside this repository: the
# schema still generates, every offline test still passes, and the only symptom is that
# no proposal can ever be made.
#
# That is not hypothetical. Three of this project's four response schemas were refused --
# `rule_spec_schema`, `heuristic_schema` and `VERDICT_SCHEMA` -- for their whole lives,
# and nothing failed, because no LLM stage had ever been run against the real API. It is
# the same defect `spec/jsonschema.py` was written to close, one level further out: there
# a rule the schema could not express could not be proposed, here a schema the API will
# not accept means nothing can be proposed at all.
#
#   {"type": "number"|"integer"} with minimum / maximum / exclusiveMinimum /
#       exclusiveMaximum / multipleOf   -> 400 "properties ... are not supported"
#   {"type": "array"} with maxItems     -> 400 "property 'maxItems' is not supported"
#   {"type": "array"} with minItems > 1 -> 400 "'minItems' values other than 0 or 1"
#
# String constraints -- `minLength`, `maxLength`, `pattern` -- are accepted and are
# deliberately left alone. The published subset lists them as unsupported and the SDK
# validates them client-side instead; either way a schema carrying them makes calls that
# work, which is the only claim being made here.
#
# Three *shapes* are refused as well, and a keyword list cannot see any of them. They are
# from the published subset rather than measured, for the reason given above the request
# table -- but each is a shape `rule_spec_schema` really does produce, so leaving them to
# be discovered by the first proposal ever sent would be leaving the rule proposer
# exactly as dead as the judge was.
#
#   {"type": ["string", "integer"]}         a type union; `anyOf` is the supported form
#   {"type": "object", "additionalProperties": {...}}
#                                           a map whose keys are not known in advance
#   {"type": "object"} with no additionalProperties
#                                           it must be present, and must be `false`
REFUSED_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "maxItems",
)

# Where a nested schema can be found, so a *property named* `minimum` is never mistaken
# for the keyword `minimum`. `rule_spec_schema` really does have properties named `0`
# and `1`, so walking a schema as an undifferentiated dict is not safe.
#
# Public because the test that holds this file to its job walks the schemas itself,
# independently of `_wire`, and the two must not be able to disagree about *where* a
# schema nests -- that disagreement would show up as the check passing.
SCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")
SCHEMA_VALUES = ("items", "additionalProperties", "not", "contains", "anyOf", "allOf", "oneOf")


def wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with the refused constraints restated in prose instead of dropped.

    **This weakens no guarantee, and the reason is worth stating.** A response schema is
    the model's *contract*; what protects the pipeline is the validator that reads the
    reply, and every caller has one -- `spec.schema.validate_spec` for a rule spec,
    `llm.heuristics._validate` for a weight vector. Loosening the wire schema makes a
    malformed response more likely, never more dangerous, and both callers already give
    one corrective retry carrying the specific errors.

    Restated rather than deleted, which is what the SDK itself does for the string
    constraints it handles: a model not told how many entries a vector has will not
    reliably produce that many, and the retry would then be spent on something the
    request could have said in the first place.

    **One thing here is deleted rather than restated, and it is the exception that
    proves the rule.** A property whose keys are not known in advance -- `rules.drops`'s
    per-piece region -- has no wire form at all, and the request enforces the schema it
    was sent, so a model told in prose to send a field the schema forbids cannot send
    it. Such a property is removed from its parent, and `tests/test_loop.py` names the
    ones that are removed today: what is lost is a rule the proposer can no longer
    reach, which is a thing this project insists on counting rather than discovering.
    """
    return _wire(schema)


def _is_open_map(node: Any) -> bool:
    """An object whose keys are not fixed by the schema -- the one inexpressible shape.

    `{"additionalProperties": {...schema...}}` is how a map from a piece id to something
    is written, and the wire takes `additionalProperties` only as `false`. There is no
    weaker form to fall back to: a closed object with no properties is one that must be
    empty, which is a worse lie than saying the field is not available.
    """
    return isinstance(node, dict) and isinstance(node.get("additionalProperties"), dict)


def _close_the_object(out: dict[str, Any]) -> None:
    """``additionalProperties: false`` on every object, which the wire requires.

    Added here rather than demanded of every generator, for the same reason the refused
    keywords are removed here: a generator states what its validator implies, and one
    place knows what the request takes. An open map that reached this point was not
    removable -- it is a whole schema rather than a property of one -- so it is closed
    too, and a test asserts that no schema this project sends contains one.
    """
    if out.get("type") != "object":
        return
    if out.get("additionalProperties") is not False:
        out["additionalProperties"] = False


def _wire(node: Any) -> Any:
    if isinstance(node, list):
        return [_wire(item) for item in node]
    if not isinstance(node, dict):
        return node

    unreachable = {
        name
        for name, sub in (node.get("properties") or {}).items()
        if _is_open_map(sub)
    }

    out: dict[str, Any] = {}
    dropped: dict[str, Any] = {}
    for key, value in node.items():
        if key in REFUSED_KEYWORDS:
            dropped[key] = value
        elif key == "properties" and isinstance(value, dict):
            out[key] = {
                name: _wire(sub)
                for name, sub in value.items()
                if name not in unreachable
            }
        elif key == "required" and isinstance(value, list):
            out[key] = [name for name in value if name not in unreachable]
        elif key in SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: _wire(sub) for name, sub in value.items()}
        elif key in SCHEMA_VALUES:
            out[key] = _wire(value)
        else:
            out[key] = value

    _close_the_object(out)

    kind = out.get("type")
    if isinstance(kind, list) and "anyOf" not in out:
        # A type union says "one of these", which is what `anyOf` says in the dialect
        # the API accepts. Sibling keywords still apply to whichever arm matches, so the
        # rest of the node is left exactly where it was.
        out["anyOf"] = [{"type": name} for name in kind]
        del out["type"]

    minimum_items = node.get("minItems")
    if isinstance(minimum_items, int) and minimum_items > 1:
        dropped["minItems"] = minimum_items
        out["minItems"] = 1

    note = _restate(dropped)
    if note:
        stated = out.get("description")
        out["description"] = f"{stated}; {note}" if stated else note
    return out


def _restate(dropped: dict[str, Any]) -> str:
    """The dropped constraints as a sentence the model can act on."""
    parts: list[str] = []
    low, high = dropped.get("minimum"), dropped.get("maximum")
    if low is not None and high is not None:
        parts.append(f"between {low} and {high}")
    elif low is not None:
        parts.append(f"at least {low}")
    elif high is not None:
        parts.append(f"at most {high}")
    if "exclusiveMinimum" in dropped:
        parts.append(f"greater than {dropped['exclusiveMinimum']}")
    if "exclusiveMaximum" in dropped:
        parts.append(f"less than {dropped['exclusiveMaximum']}")
    if "multipleOf" in dropped:
        parts.append(f"a multiple of {dropped['multipleOf']}")
    fewest, most = dropped.get("minItems"), dropped.get("maxItems")
    if fewest is not None and fewest == most:
        parts.append(f"exactly {fewest} items")
    elif fewest is not None and most is not None:
        parts.append(f"between {fewest} and {most} items")
    elif fewest is not None:
        parts.append(f"at least {fewest} items")
    elif most is not None:
        parts.append(f"at most {most} items")
    return ", ".join(parts)


@dataclass
class Usage:
    """Token accounting, accumulated across a run so cost stays visible."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    # Writes as well as reads, because a read alone cannot tell the two failures apart.
    # A run that reports nothing from cache is either one whose prefix is too short to
    # cache -- in which case nothing was spent and nothing is wrong -- or one that opened
    # an entry at 1.25x or 2x on every call and let it expire unread, which is strictly
    # worse than not caching at all. Those look identical from the read meter and are
    # obvious from this one, and telling them apart is the whole of what says whether
    # `CACHE_TTL` is right for a real run.
    cache_write_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    @property
    def cache_reads_per_write(self) -> float:
        """Reads for each token written, which is what decides whether caching paid.

        Zero is the signature of the failure above: an entry opened on every call and
        never read. Above roughly 2.5 the hour-long entry has repaid its doubled write.
        """
        if not self.cache_write_tokens:
            return 0.0
        return self.cache_read_tokens / self.cache_write_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass
class Claude:
    """A configured client for structured generation."""

    model: str = PROPOSAL_MODEL
    effort: str = "high"
    usage: Usage = field(default_factory=Usage)
    _client: Any = None

    def __post_init__(self) -> None:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - environment dependent
            raise LLMUnavailable(
                "the anthropic package is not installed; install the 'llm' extra "
                "(uv sync --extra llm) to use rule proposal or judging"
            ) from error

        _ensure_key_env()

        try:
            self._client = anthropic.Anthropic()
        except Exception as error:  # pragma: no cover - environment dependent
            raise LLMUnavailable(f"could not construct an Anthropic client: {error}") from error

    @staticmethod
    def is_configured() -> bool:
        """Whether a call is likely to succeed, without making one.

        An unset ``ANTHROPIC_API_KEY`` does not mean there are no credentials: the SDK
        also reads a profile written by ``ant auth login``, so that directory has to be
        checked for existence -- not merely expanded, which always yields a path.

        As a last resort, look for an ``api-key`` file in the project root -- a
        convenience for local development that keeps the key out of environment
        variables while still being git-ignored.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        if os.path.isdir(os.path.expanduser("~/.config/anthropic")):
            return True
        return _find_key_file() is not None

    def structured(
        self,
        system: str | list[dict[str, Any]],
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> dict[str, Any]:
        """One call that must return JSON matching ``schema``.

        ``schema`` may be ``None``, which asks for JSON without a response format --
        the only thing to do with a schema the grammar compiler refuses, and the reason
        the reply is then read with `json_object` rather than `json.loads`. The caller
        states the contract in its prompt and validates what comes back, which it does
        either way.

        Streamed regardless of size: a proposal with adaptive thinking on a large
        schema can run well past a non-streaming HTTP timeout, and a timeout halfway
        through a generation is an expensive way to learn that.

        ``schema`` is stated as the caller means it and sent through `wire_schema`,
        which is the one place that knows what the API will accept. Callers therefore
        write the schema their validator implies and never a dialect of it. The thinking
        parameters go through `request_options` for the same reason, one layer out: what
        a *model* accepts is not a property of the caller either.
        """
        import anthropic

        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **request_options(self.model, self.effort, schema),
            ) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as error:
            raise LLMUnavailable(f"the model call failed: {error}") from error
        except (TypeError, ValueError, OSError) as error:
            # The SDK raises a bare TypeError when it cannot resolve credentials, and
            # that happens at request time rather than at construction. Left unwrapped
            # it escapes the loop's handler and kills a run mid-generation instead of
            # reporting cleanly and stopping.
            raise LLMUnavailable(f"the model call could not be made: {error}") from error

        self.usage.add(message.usage)

        if message.stop_reason == "refusal":
            raise LLMUnavailable(
                "the model declined this request "
                f"({getattr(message.stop_details, 'category', 'unknown')})"
            )
        if message.stop_reason == "max_tokens":
            # A reply cut off mid-object is not JSON, and letting `json.loads` say so
            # raises out of every caller's handler and kills the run with a stack trace
            # about a delimiter. Say what happened instead: the cap is the fix, and it
            # is a number in this file.
            raise LLMUnavailable(
                f"the model's reply was cut off at max_tokens={max_tokens:,}; "
                "raise the cap or lower the effort"
            )

        text = next((block.text for block in message.content if block.type == "text"), None)
        if text is None:
            raise LLMUnavailable("the model returned no text block")
        try:
            return json_object(text)
        except ValueError as error:
            raise LLMUnavailable(
                f"the model returned text that is not JSON: {error}"
            ) from error


def json_object(text: str) -> dict[str, Any]:
    """The JSON object in ``text``, whether or not a grammar guaranteed it was alone.

    With a response format the whole reply is the object and this is `json.loads`. With
    a schema too big to compile there is no such guarantee, and a model asked for JSON
    in prose sometimes fences it or says a sentence first. Recovering from that here is
    cheaper than a corrective retry and much cheaper than a whole proposal thrown away.

    Deliberately not a repair: the braces have to balance and the content has to parse.
    Parsing rather than string-matching is the same rule as before -- current models
    vary their JSON escaping, so nothing here may look at the bytes between the braces.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_outermost_object(text))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _outermost_object(text: str) -> str:
    """From the first ``{`` to the brace that closes it, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("the reply's JSON object is not closed")


# ------------------------------------------------- what a cache entry costs to open
#
# A third wire fact, of the same kind as the two above and failing in the same silence:
# the API will not open a cache entry for a prefix below a per-model minimum, and it does
# not say so. There is no error and no warning -- `cache_creation_input_tokens` is simply
# zero, so a `cache_control` marker on a short prefix is a line of code that has never
# done anything.
#
# Two of this file's three callers were in exactly that position for their whole lives.
# `claude-haiku-4-5` wants 4,096 tokens; `llm/compare.py` sends a brief of **253
# tokens** and `llm/judge.py` a brief plus the rules at **858**. Neither has ever opened
# an entry, while `cached_system`'s own docstring said caching was "worth real
# money here". It is -- for `llm/propose.py`, whose prefix is **11,081 tokens** on a model
# that wants 512, which is the whole of the difference.
#
# Every figure in this paragraph is counted rather than guessed: the minimums against the
# published table, the token counts through `count_tokens`, and the silence itself by
# forcing the marker back on and watching nothing happen -- `tests/test_llm_live.py`.
#
# The minimum is **not** monotonic across model generations, which is why this has to be
# a table rather than a rule of thumb: 512 on the newest models against 4,096 on Haiku
# 4.5, so the cheap model is the demanding one, and a prompt that caches on the proposer
# silently will not on the judge.
#
# Unknown models take the smallest minimum, and the asymmetry here is the opposite of
# `MODEL_WIRE`'s. There, guessing permissively buys a 400 that names the parameter.
# Here, guessing permissively buys a marker that does nothing and costs nothing, while
# guessing *restrictively* withholds a marker that would have paid -- and that failure is
# invisible, which is the one this table exists to stop making.
SMALLEST_CACHE_MINIMUM = 512

CACHE_MINIMUM: dict[str, int] = {
    PROPOSAL_MODEL: SMALLEST_CACHE_MINIMUM,
    JUDGE_MODEL: 4_096,
}

# Characters per token, for deciding whether a prefix clears its model's minimum without
# spending a `count_tokens` call to find out. Deliberately low -- real text runs nearer 4
# for prose and 3.3 for the dense JSON that dominates the proposer's prefix -- so this
# over-counts, and over-counting errs toward emitting a marker, which is the direction
# that costs nothing when it is wrong. It is a threshold test rather than an estimate,
# and the two prefixes it exists to separate differ by more than an order of magnitude.
CHARS_PER_TOKEN = 3


def caches(model: str, stable: str) -> bool:
    """Whether ``model`` would open a cache entry for a prefix this size.

    Split out for the same reason as `request_options`: it is a claim about the wire, so
    it should be answerable without making a call. `tests/test_loop.py` asks it of every
    prompt this project actually sends.
    """
    minimum = CACHE_MINIMUM.get(model, SMALLEST_CACHE_MINIMUM)
    return len(stable) / CHARS_PER_TOKEN >= minimum


# ------------------------------------------------------ how long an entry has to live
#
# A cache entry's default life is five minutes, measured from the *start* of the request
# that opens it, and the proposal call is nowhere near that often. Between two proposals
# the loop runs a screen, a tournament, an analysis pass, a scoring pass and up to four
# judging calls; at the default `mechanical_rate` of 0.5 half the generations make no
# proposal at all, so the gap is frequently two whole generations. The entry is therefore
# cold every time, and every proposal has been paying 1.25x for 11,081 tokens and
# reading none of them back. The only reads this project has ever got are the corrective
# retries, which fire seconds later.
#
# An hour costs 2x to write against 1.25x, and reads at the same 0.1x. Over the four
# proposal calls a default run makes, in units of one uncached prefix:
#
#     five minutes, always cold   4 x 1.25            = 5.0
#     one hour                    2.0 + 3 x 0.1       = 2.3
#
# Break-even is three calls, which `generations` clears by default and a two-generation
# run would not -- so this is a saving on the runs long enough to be worth making.
#
# **The far end is not covered and is not guessed at.** If a run's generations are slow
# enough to push proposals more than an hour apart, neither duration helps and the honest
# answer is the cold miss; re-warming on a timer would be spending money to look busy.
# Which side a real run falls on is a measurement, and `Usage` reports the meters that
# settle it -- cache writes that are never read are the signature of the failure, and
# they are printed at the end of every run.
CACHE_TTL = "1h"


def cached_system(
    model: str, stable: str, volatile: str = ""
) -> list[dict[str, Any]]:
    """Build a system prompt, with a cache breakpoint after the stable part where one pays.

    Caching is worth real money on the proposal call: the stable half -- the DSL schema,
    the design brief, the worked example -- is around ten thousand tokens and is
    byte-identical on every call of a run. Anything that varies per call must come after
    the breakpoint, or the prefix changes and nothing is cached at all.

    It is worth nothing whatsoever on the two judging calls, and *that* is why the model
    is a parameter. A prefix under `CACHE_MINIMUM[model]` is not cached and does not say
    so; the caller states the prompt it means, and one place knows what the wire keeps.

    The blocks are otherwise identical either way -- the marker is added, never the text
    rearranged -- so whether a prompt caches cannot change what the model is asked.

    The entry is opened for `CACHE_TTL` rather than the default five minutes, because a
    generation takes longer than five minutes and an entry nothing reads is a surcharge.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": stable}]
    if caches(model, stable):
        blocks[0]["cache_control"] = {"type": "ephemeral", "ttl": CACHE_TTL}
    if volatile:
        blocks.append({"type": "text", "text": volatile})
    return blocks
