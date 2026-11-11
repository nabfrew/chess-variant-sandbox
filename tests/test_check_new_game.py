"""`tests/check_new_game.py` and the `new_game` mark it runs (Q1156).

The command runs the marked tests, narrowed by `--games` to the games a designer touched. What
can go wrong silently is the narrowing: a parameter the plugin fails to read as a game would
keep every game (slow, but honest), and one it reads as the wrong game would drop the check
that should fire. These pin the reading, and that the command finds the guards the record
names.
"""

from __future__ import annotations

from pathlib import Path

import check_new_game
import new_game

ROOT = Path(__file__).resolve().parents[1]


class TestAParameterIsReadAsAGame:
    def test_a_path_in_the_corpus(self) -> None:
        path = ROOT / "rulesets" / "chess.json"
        assert new_game.game_file(path) == path.resolve()

    def test_a_relative_string_as_frozen_spells_it(self) -> None:
        assert new_game.game_file("archive/sardines.json") == (ROOT / "archive/sardines.json")

    def test_anything_else_is_no_game(self) -> None:
        for value in ("chess", 3, ROOT / "tests" / "corpus_pins.json", None):
            assert new_game.game_file(value) is None, value


class TestGamesNamesTheChosenFiles:
    def test_all_narrows_nothing(self) -> None:
        assert new_game.chosen("all") is None

    def test_an_empty_list_keeps_no_game(self) -> None:
        assert new_game.chosen("") == set()

    def test_a_list_is_resolved_against_the_repository(self) -> None:
        assert new_game.chosen("rulesets/chess.json, archive/sardines.json") == {
            (ROOT / "rulesets/chess.json").resolve(),
            (ROOT / "archive/sardines.json").resolve(),
        }


def test_the_command_reaches_the_guards_the_record_keeps() -> None:
    """corpus-pins.md §3's guards, by file: a file losing its mark drops a guard silently."""
    marked = set(check_new_game.marked_files())
    for name in ("test_schema_nine", "test_grid", "test_operators", "test_rules_used",
                 "test_schema_six", "test_schema_ten", "test_hex_readers", "test_hex_resolver",
                 "test_edits", "test_corpus"):
        assert f"tests/{name}.py" in marked, name


def test_a_game_with_a_row_owes_none() -> None:
    assert check_new_game.frozen_rows_owed(["rulesets/chess.json"]) == []
