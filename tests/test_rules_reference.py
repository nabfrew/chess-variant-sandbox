"""`RULES.md`, the rule language's public reference, is generated and cannot go stale.

`spec/reference.py` renders it from the schema and the validator. These tests hold the
committed file to what the module renders today, hold the module's own tables to the
schema (a note or a default for a key that no longer exists is a stale sentence), and play
the worked examples out of the committed file itself, so an example that stops validating
fails here rather than in a reader's hands.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.reference import (
    COMMAND,
    DEFAULTS,
    NOTES,
    described,
    render_reference,
    schema_paths,
)
from chess_variant_sandbox.spec.schema import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    validate_spec,
)

RULES = Path(__file__).resolve().parents[1] / "RULES.md"
_JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def _node(path: str) -> dict:
    node = rule_spec_schema(strict=True)
    for part in re.findall(r"[^.\[\]{}]+(?:\[\]|\{\})?", path):
        key = part.removesuffix("[]").removesuffix("{}")
        if "properties" not in node:
            node = next(branch for branch in node["anyOf"] if "properties" in branch)
        node = node["properties"][key]
        if part.endswith("[]"):
            node = node["items"]
        elif part.endswith("{}"):
            node = node["additionalProperties"]
    return node


class TestTheCommittedFile:
    def test_it_is_what_the_schema_renders_today(self) -> None:
        assert RULES.exists(), f"RULES.md is missing; generate it with `{COMMAND}`"
        assert RULES.read_text(encoding="utf-8") == render_reference(), (
            f"RULES.md differs from what the schema renders; regenerate it with `{COMMAND}`"
        )

    def test_it_names_the_schema_version_it_describes(self) -> None:
        text = RULES.read_text(encoding="utf-8")
        assert f"at schema version {SCHEMA_VERSION}." in text

    def test_its_title_names_the_current_schema_version(self) -> None:
        """Q905, the user: the reference must say what version it is for. The first line,
        so a copy anywhere (the public repository's, a reader's AI's) says it at once, and
        from `SCHEMA_VERSION`, so a bump moves it."""
        first = render_reference().splitlines()[0]
        assert first == f"# The rule language, schema version {SCHEMA_VERSION}"
        assert RULES.read_text(encoding="utf-8").splitlines()[0] == first

    def test_it_says_once_what_a_game_declares_and_that_later_versions_accept_it(
        self,
    ) -> None:
        text = render_reference()
        said = f'should declare\n`"schema_version": {SCHEMA_VERSION}`.'
        assert text.count(said) == 1
        assert "running a later version still\naccepts it" in text
        # True only while every version stays accepted, which is what lets it say so.
        assert tuple(range(1, SCHEMA_VERSION + 1)) == SUPPORTED_SCHEMA_VERSIONS

    def test_its_examples_validate_and_build_a_game(self) -> None:
        blocks = _JSON_BLOCK.findall(RULES.read_text(encoding="utf-8"))
        assert len(blocks) == 2
        portals = 0
        for block in blocks:
            spec = json.loads(block)
            assert validate_spec(spec) == [], spec["name"]
            assert spec["schema_version"] == SCHEMA_VERSION
            engine = Engine(spec)
            assert engine.legal_moves(engine.initial_state())
            portals += "portals" in spec["rules"]
        assert portals == 1

    def test_it_names_no_private_path(self) -> None:
        text = RULES.read_text(encoding="utf-8")
        for private in ("docs/", ".claude", "deploy/", "runs/", "coordinator"):
            assert private not in text, private


class TestTheTables:
    def test_every_key_has_words(self) -> None:
        silent = [path for path in schema_paths() if not described(path, _node(path))]
        assert silent == [], f"give these a description or a NOTES entry: {silent}"

    @pytest.mark.parametrize("table", [NOTES, DEFAULTS], ids=["NOTES", "DEFAULTS"])
    def test_every_entry_names_a_key_that_exists(self, table: dict[str, str]) -> None:
        stale = sorted(set(table) - set(schema_paths()))
        assert stale == []

    def test_a_default_stated_as_a_value_is_one_the_key_allows(self) -> None:
        for path, words in DEFAULTS.items():
            literal = re.fullmatch(r"`(.*)`", words)
            node = _node(path)
            if literal is None or "enum" not in node:
                continue
            assert json.loads(literal.group(1)) in node["enum"], path
