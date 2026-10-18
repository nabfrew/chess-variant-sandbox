"""The model-facing paths nothing offline reached (archive/plan-tests F9, Q751b).

`Claude` and `judge_ruleset` are the two places a run spends money, so every test here
stands a fake where the SDK's transport goes and never opens a socket: the API is never
called. What is held is what the code does *around* the call: what it refuses to start
without, how a reply that is not a verdict is reported, what a key file does to the
environment, and what a judgement hands the reader.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")

from chess_variant_sandbox.llm import client as client_module  # noqa: E402
from chess_variant_sandbox.llm.client import (  # noqa: E402
    JUDGE_MODEL,
    PROPOSAL_MODEL,
    Claude,
    LLMUnavailable,
)
from chess_variant_sandbox.llm.judge import (  # noqa: E402
    DEFAULT_SAMPLE,
    VERDICT_SCHEMA,
    Judgement,
    judge_ruleset,
)

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

VERDICT = {
    "verdict": "A quick game with one idea.",
    "strengths": ["clear goal", "short"],
    "weaknesses": ["the first move decides it"],
    "suspected_degeneracy": "the opening is solved",
    "would_play_again": True,
}


class Transport:
    """Stands in for `anthropic.Anthropic`: it records the request and returns a message."""

    def __init__(self, message: Any = None, error: Exception | None = None) -> None:
        self.message = message
        self.error = error
        self.requests: list[dict[str, Any]] = []

    @property
    def messages(self) -> Transport:
        return self

    def stream(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error

        @contextlib.contextmanager
        def opened() -> Any:
            yield SimpleNamespace(get_final_message=lambda: self.message)

        return opened()


def reply(payload: Any = None, *, text: str | None = None, stop: str = "end_turn",
          blocks: list[Any] | None = None, **usage: int) -> SimpleNamespace:
    body = text if text is not None else json.dumps(payload)
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=body)]
    return SimpleNamespace(
        usage=SimpleNamespace(**({"input_tokens": 11, "output_tokens": 7} | usage)),
        stop_reason=stop,
        stop_details=SimpleNamespace(category="policy"),
        content=content,
    )


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # set then delete, so that a variable the code under test adds is removed at teardown
    # (monkeypatch only undoes what it saw: deleting an absent name records nothing)
    for name in KEY_VARIABLES:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def client_over(monkeypatch: pytest.MonkeyPatch, transport: Transport, **kwargs: Any) -> Claude:
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: transport)
    return Claude(**kwargs)


class TestKeyFile:
    """`api-key` at the project root is a convenience that must never override a real key."""

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        # `_find_key_file` reads `<root>/api-key`, with root = parents[2] of this module.
        fake_module = tmp_path / "chess_variant_sandbox" / "llm" / "client.py"
        monkeypatch.setattr(client_module, "__file__", str(fake_module))
        return tmp_path

    def test_no_file_is_none(self, root: Path) -> None:
        assert client_module._find_key_file() is None

    def test_the_first_name_with_a_key_wins_and_whitespace_is_stripped(self, root: Path) -> None:
        (root / "api_key").write_text("second\n", encoding="utf-8")
        assert client_module._find_key_file() == "second"
        (root / "api-key").write_text("  first-key \n", encoding="utf-8")
        assert client_module._find_key_file() == "first-key"

    def test_an_empty_file_is_skipped_for_the_next_name(self, root: Path) -> None:
        (root / "api-key").write_text("\n", encoding="utf-8")
        assert client_module._find_key_file() is None
        (root / "api_key").write_text("fallback", encoding="utf-8")
        assert client_module._find_key_file() == "fallback"

    def test_the_file_is_put_in_the_environment_when_there_is_no_key(
        self, root: Path, no_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (root / "api-key").write_text("from-file", encoding="utf-8")
        client_module._ensure_key_env()
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "from-file"

    @pytest.mark.parametrize("name", KEY_VARIABLES)
    def test_a_key_already_in_the_environment_is_never_overridden(
        self, root: Path, no_keys: None, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        (root / "api-key").write_text("from-file", encoding="utf-8")
        monkeypatch.setenv(name, "from-env")
        client_module._ensure_key_env()
        import os

        assert os.environ[name] == "from-env"
        assert os.environ.get("ANTHROPIC_API_KEY") != "from-file"

    def test_no_file_and_no_key_leaves_the_environment_alone(
        self, root: Path, no_keys: None
    ) -> None:
        import os

        client_module._ensure_key_env()
        assert not any(name in os.environ for name in KEY_VARIABLES)


class TestIsConfigured:
    @pytest.fixture
    def nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_keys: None) -> None:
        """No key, no profile directory, no key file."""
        monkeypatch.setattr(client_module, "_find_key_file", lambda: None)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def test_nothing_is_not_configured(self, nothing: None) -> None:
        assert Claude.is_configured() is False

    @pytest.mark.parametrize("name", KEY_VARIABLES)
    def test_a_key_in_the_environment_is_configured(
        self, nothing: None, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        monkeypatch.setenv(name, "k")
        assert Claude.is_configured() is True

    def test_a_profile_written_by_the_sdks_login_is_configured_without_a_key(
        self, nothing: None, tmp_path: Path
    ) -> None:
        (tmp_path / ".config" / "anthropic").mkdir(parents=True)
        assert Claude.is_configured() is True

    def test_a_key_file_is_configured(
        self, nothing: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(client_module, "_find_key_file", lambda: "k")
        assert Claude.is_configured() is True

    def test_without_the_sdk_nothing_is_configured_even_with_a_key(
        self, nothing: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
        assert Claude.is_configured() is False


class TestConstruction:
    def test_the_sdk_client_is_built_after_the_key_file_is_in_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        monkeypatch.setattr(client_module, "_ensure_key_env", lambda: order.append("env"))
        monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: order.append("sdk"))
        Claude()
        assert order == ["env", "sdk"]

    def test_defaults_are_the_proposal_model_at_high_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        claude = client_over(monkeypatch, Transport())
        assert (claude.model, claude.effort) == (PROPOSAL_MODEL, "high")
        assert claude.usage.calls == 0

    def test_a_client_that_cannot_be_built_is_the_models_being_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no credentials")

        monkeypatch.setattr(anthropic, "Anthropic", refuse)
        with pytest.raises(LLMUnavailable, match="could not construct an Anthropic client"):
            Claude()

    def test_a_missing_sdk_says_which_extra_to_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
        with pytest.raises(LLMUnavailable, match="llm' extra"):
            Claude()


class TestStructured:
    def test_the_request_is_the_one_the_model_accepts_and_the_reply_is_the_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = Transport(reply({"a": 1}))
        claude = client_over(monkeypatch, transport, model=JUDGE_MODEL, effort="medium")
        data = claude.structured("system text", "user text", VERDICT_SCHEMA, max_tokens=123)
        assert data == {"a": 1}
        (request,) = transport.requests
        assert request["model"] == JUDGE_MODEL
        assert request["max_tokens"] == 123
        assert request["system"] == "system text"
        assert request["messages"] == [{"role": "user", "content": "user text"}]
        # the judge's model takes neither thinking nor effort
        assert "thinking" not in request and "effort" not in request["output_config"]
        assert request["output_config"]["format"]["type"] == "json_schema"

    def test_the_proposal_model_is_asked_to_think_at_the_chosen_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = Transport(reply({}))
        client_over(monkeypatch, transport, effort="low").structured("s", "u", None)
        (request,) = transport.requests
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": "low"}  # no schema, no format

    def test_every_call_is_added_to_the_usage_meter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = Transport(reply({}, cache_read_input_tokens=5, cache_creation_input_tokens=2))
        claude = client_over(monkeypatch, transport)
        claude.structured("s", "u", None)
        claude.structured("s", "u", None)
        assert claude.usage.as_dict() == {
            "calls": 2, "input_tokens": 22, "output_tokens": 14,
            "cache_read_tokens": 10, "cache_write_tokens": 4,
        }

    def test_a_reply_with_prose_around_the_object_is_still_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = 'Here you go:\n```json\n{"k": "a } brace in a string"}\n```'
        claude = client_over(monkeypatch, Transport(reply(text=text)))
        assert claude.structured("s", "u", None) == {"k": "a } brace in a string"}

    @pytest.mark.parametrize(
        ("message", "match"),
        [
            (reply({}, stop="refusal"), "declined this request .policy."),
            (reply({}, stop="max_tokens"), "cut off at max_tokens=64,000"),
            (reply(blocks=[SimpleNamespace(type="thinking", text="hm")]), "no text block"),
            (reply(text="no object here"), "not JSON"),
            (reply(text='{"never": "closed"'), "not JSON"),
            (reply(text="[1, 2]"), "not JSON"),
        ],
    )
    def test_a_reply_that_is_not_a_verdict_is_reported_as_the_model_being_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, message: Any, match: str
    ) -> None:
        claude = client_over(monkeypatch, Transport(message))
        with pytest.raises(LLMUnavailable, match=match):
            claude.structured("s", "u", None)
        assert claude.usage.calls == 1, "a refused reply was still paid for"

    def test_an_api_error_is_the_models_being_unavailable_and_costs_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        error = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x/"))
        claude = client_over(monkeypatch, Transport(error=error))
        with pytest.raises(LLMUnavailable, match="the model call failed"):
            claude.structured("s", "u", None)
        assert claude.usage.calls == 0

    @pytest.mark.parametrize("error", [TypeError("no credentials"), OSError("socket")])
    def test_a_failure_to_make_the_call_does_not_escape_as_its_own_type(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        claude = client_over(monkeypatch, Transport(error=error))
        with pytest.raises(LLMUnavailable, match="could not be made"):
            claude.structured("s", "u", None)


class FakeJudge:
    """What `judge_ruleset` needs of a client: a model name and `structured`."""

    model = JUDGE_MODEL

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.calls: list[dict[str, Any]] = []

    def structured(self, system: Any, user: str, schema: Any, max_tokens: int = 0) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "schema": schema, "max": max_tokens})
        return self.data


@pytest.fixture(scope="module")
def dull_records() -> tuple[dict[str, Any], list[Any]]:
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament.play import play_game

    spec = json.loads((RULESETS / "dull.json").read_text(encoding="utf-8"))
    engine = Engine(spec)
    return spec, [play_game(engine, ("greedy", "random"), seed) for seed in range(6)]


class TestJudgeRuleset:
    def test_one_call_carries_the_rules_and_the_games_and_returns_the_prose(
        self, dull_records: tuple[dict[str, Any], list[Any]]
    ) -> None:
        spec, records = dull_records
        judge = FakeJudge(VERDICT)
        judgement = judge_ruleset(spec, records, client=judge)
        (call,) = judge.calls
        assert call["schema"] is VERDICT_SCHEMA
        assert call["user"].count("Game: ") == DEFAULT_SAMPLE
        assert call["user"].endswith("Judge the game.")
        assert "The rules:" in json.dumps(call["system"])
        assert judgement == Judgement(
            verdict="A quick game with one idea.",
            strengths=["clear goal", "short"],
            weaknesses=["the first move decides it"],
            suspected_degeneracy="the opening is solved",
            would_play_again=True,
        )

    def test_the_sample_size_is_the_number_of_games_shown(
        self, dull_records: tuple[dict[str, Any], list[Any]]
    ) -> None:
        spec, records = dull_records
        judge = FakeJudge(VERDICT)
        judge_ruleset(spec, records, client=judge, sample=2)
        assert judge.calls[0]["user"].count("Game: ") == 2

    def test_a_reply_with_only_a_verdict_is_a_judgement_with_the_rest_empty(
        self, dull_records: tuple[dict[str, Any], list[Any]]
    ) -> None:
        spec, records = dull_records
        judgement = judge_ruleset(spec, records, client=FakeJudge({"verdict": "Fine."}))
        assert judgement == Judgement(verdict="Fine.")
        assert judgement.would_play_again is False

    def test_no_client_means_the_judge_model_at_medium_effort(
        self, dull_records: tuple[dict[str, Any], list[Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec, records = dull_records
        transport = Transport(reply(VERDICT))
        monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: transport)
        judgement = judge_ruleset(spec, records)
        assert judgement.verdict == VERDICT["verdict"]
        assert transport.requests[0]["model"] == JUDGE_MODEL
        assert transport.requests[0]["max_tokens"] == 4_000


class TestJudgementNotes:
    def test_notes_is_the_verdict_alone_when_nothing_is_suspected(self) -> None:
        assert Judgement(verdict="Works.").notes() == "Works."

    def test_a_suspected_degeneracy_is_appended_in_the_same_line(self) -> None:
        notes = Judgement(verdict="Works.", suspected_degeneracy="a lock").notes()
        assert notes == "Works. Suspected degeneracy: a lock"

    def test_as_dict_carries_prose_and_the_one_flag_and_no_number(self) -> None:
        data = Judgement(verdict="v", strengths=["s"]).as_dict()
        assert set(data) == set(VERDICT_SCHEMA["properties"])
        assert not any(isinstance(value, float) for value in data.values())
