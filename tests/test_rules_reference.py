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
    _BASIC_PIECE_KEYS,
    COMMAND,
    DEFAULTS,
    FAMILIES,
    NOTES,
    described,
    render_reference,
    schema_paths,
)
from chess_variant_sandbox.spec.schema import (
    MOVE_KINDS,
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

    def test_it_says_once_what_a_game_declares_and_that_an_older_file_is_converted(
        self,
    ) -> None:
        text = render_reference()
        said = f'should declare\n`"schema_version": {SCHEMA_VERSION}`.'
        assert text.count(said) == 1
        assert "python -m chess_variant_sandbox.spec.spelling FILE" in text
        # Every earlier stamp is accepted, which is what lets it say the number limits nothing.
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


class TestTheMapOfTheLanguage:
    """Q994: *What the language can make* places every family of rule, so a reader who was
    given only this file sees the whole range before the key-by-key reference
    (`docs/reference/rules-md-blind-trial.md` §2.4). Held to the schema both ways: a key
    added to the language fails here until it is placed, and a key removed fails as a
    stale entry."""

    def _placed(self) -> list[str]:
        return [key for _, _, keys in FAMILIES for key in keys]

    def test_it_places_every_rule_piece_feature_board_shape_and_movement_kind(self) -> None:
        schema = rule_spec_schema(strict=True)
        rules = {f"rules.{key}" for key in schema["properties"]["rules"]["properties"]}
        pieces = {
            f"pieces[].{key}"
            for key in schema["properties"]["pieces"]["items"]["properties"]
            if key not in _BASIC_PIECE_KEYS
        }
        board = {
            f"board.{key}"
            for key in schema["properties"]["board"]["properties"]
            if key not in ("width", "height")
        }
        assert set(self._placed()) == rules | pieces | board | set(MOVE_KINDS)

    def test_each_key_is_placed_once(self) -> None:
        placed = self._placed()
        assert len(placed) == len(set(placed))

    def test_every_key_it_names_exists(self) -> None:
        paths = set(schema_paths())
        named = [key for key in self._placed() if key not in MOVE_KINDS]
        assert [key for key in named if key not in paths] == []

    def test_the_committed_file_carries_the_map_and_the_play_section_once(self) -> None:
        text = RULES.read_text(encoding="utf-8")
        assert text.count("## What the language can make") == 1
        assert text.count("## A valid game is not yet a good one") == 1
        for title, _, _ in FAMILIES:
            assert text.count(f"- **{title}.**") == 1, title
