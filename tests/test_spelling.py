"""The one current spelling, and the converter that writes it (Q1008b).

`docs/reference/schema-reboot-assessment.md` §5 step 3: `rulesets/`, `archive/` and the site's
stored specs were rewritten once by `spec/spelling.py`, which stays as the import path for an
old file. What is held here:

* every library file is in the current spelling (a file added later in an old one fails here,
  and the message says how to convert it);
* the converter moves no game: over the library and every other spelling `tests/respell.py`
  writes of it, the converted spec has the original's normal form, so its identity, and
  compiles to the original's tables, position keys included; the grandfathered retired game
  (`tunnel-horizontal`, a promotion rank past the board) keeps its rank, name and tables;
* it is one spelling: two spellings of a rule convert to one text, and converting twice is
  converting once;
* a file's own layout survives: only what changed is written again.

`tests/test_schema_nine.py::FROZEN` holds each re-spelt file's name and tables directly; the
store's half is `tests/test_site_one_spelling.py`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from respell import RESPELLINGS, every_respelling
from test_new_spec_refusals import RETIRED, RETIRED_HASH, RETIRED_TABLES
from test_schema_nine import FROZEN

from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.measurement import tables_fingerprint
from chess_variant_sandbox.spec import relate, spelling
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, stored_spec_errors, validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling, is_current, respell_text

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = sorted(FROZEN)


def _spec(path: str) -> dict[str, Any]:
    return load_spec(ROOT / path)


def _tables(spec: dict[str, Any]) -> dict[str, Any]:
    return canonical_tables(compile_spec(spec))


# ------------------------------------------------------------------------- the library


class TestTheLibraryIsInTheCurrentSpelling:
    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_each_file_is_already_current(self, path: str) -> None:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert respell_text(text) == text, (
            f"{path} is not in the current spelling: python -m "
            f"chess_variant_sandbox.spec.spelling {path}"
        )

    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_each_file_is_admitted_at_the_current_version(self, path: str) -> None:
        spec = _spec(path)
        assert spec["schema_version"] == SCHEMA_VERSION
        assert validate_spec(spec) == []


# --------------------------------------------------------------------- no game moves


class TestNoGameMoves:
    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_every_respelling_converts_to_the_same_game(self, path: str) -> None:
        spec = _spec(path)
        for name, respell in {**RESPELLINGS, "all": every_respelling}.items():
            other = respell(spec)
            converted = current_spelling(other)
            assert normal_form(converted) == normal_form(other), name
            assert spec_hash(converted) == spec_hash(spec) == FROZEN[path][0], name

    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_every_respelling_converted_compiles_to_the_files_tables(self, path: str) -> None:
        spec = _spec(path)
        converted = current_spelling(every_respelling(spec))
        assert validate_spec(converted) == []
        assert _tables(converted) == _tables(spec)

    def test_the_grandfathered_rank_past_the_board_is_kept(self) -> None:
        """`tunnel-horizontal`'s `rank: 9` on an 8-high board: `ranks: [9]` would be refused
        even for a stored spec, so it stays a lone `rank`, loaded only as stored: refused as a
        draft, forgiven at the load for this game alone (`schema.RANK_PAST_THE_BOARD_KEPT`)."""
        retired = load_spec(RETIRED)
        converted = current_spelling(retired)
        assert converted["rules"]["promotion"]["rank"] == 9
        assert "ranks" not in converted["rules"]["promotion"]
        assert converted["schema_version"] == SCHEMA_VERSION
        assert stored_spec_errors(converted) == []
        assert validate_spec(converted) == ["rules.promotion.rank 9 is not a rank on this board"]
        assert spec_hash(converted) == RETIRED_HASH
        assert tables_fingerprint(converted) == RETIRED_TABLES

    def test_it_never_changes_its_argument(self) -> None:
        spec = every_respelling(_spec("rulesets/shogi.json"))
        before = copy.deepcopy(spec)
        current_spelling(spec)
        assert spec == before

    def test_it_survives_malformed_json(self) -> None:
        for spec in (
            {}, {"rules": None}, {"pieces": "x", "rules": {"promotion": 3, "wrap": "x"}},
            {"board": [], "rules": {"promotion": [{}, 4], "turn": {"moves": "a"}}},
            {"board": {"height": 8}, "rules": {"promotion": [{}], "strike": []}},
        ):
            converted = current_spelling(spec)
            assert normal_form(converted) == normal_form(spec), spec


# ------------------------------------------------------------------------ one spelling


def _rules(**rules: Any) -> dict[str, Any]:
    return {"board": {"width": 8, "height": 8}, "rules": rules}


class TestOneSpelling:
    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_converting_twice_is_converting_once(self, path: str) -> None:
        once = current_spelling(every_respelling(_spec(path)))
        assert current_spelling(once) == once
        assert is_current(once)

    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_the_version_and_the_stated_off_rules_are_one_spelling(self, path: str) -> None:
        spec = _spec(path)
        for name in ("newest_version", "empty_rules"):
            assert current_spelling(RESPELLINGS[name](spec)) == current_spelling(spec), name

    def test_a_promotion_is_an_object_for_one_entry_and_a_list_for_more(self) -> None:
        entry = {"piece": "P", "rank": "last", "to": ["Q"]}
        one = current_spelling(_rules(promotion=[entry]))["rules"]["promotion"]
        assert one == entry == current_spelling(_rules(promotion=entry))["rules"]["promotion"]
        two = [entry, {"piece": "Q", "rank": "last", "to": ["K"]}]
        assert current_spelling(_rules(promotion=two))["rules"]["promotion"] == two

    def test_a_numbered_zone_is_ranks_and_the_last_rank_is_left_as_written(self) -> None:
        def zone(**entry: Any) -> Any:
            promotion = {"piece": "P", "to": ["Q"], **entry}
            return current_spelling(_rules(promotion=promotion))["rules"]["promotion"]

        assert zone(rank=5) == {"piece": "P", "to": ["Q"], "ranks": [5]}
        assert zone(ranks=[7, 5, 7]) == {"piece": "P", "to": ["Q"], "ranks": [5, 7]}
        assert zone(rank="last") == {"piece": "P", "to": ["Q"], "rank": "last"}
        assert zone() == {"piece": "P", "to": ["Q"]}
        assert zone(squares=["a8"]) == {"piece": "P", "to": ["Q"], "squares": ["a8"]}
        assert list(zone(rank=5, mandatory=False)) == ["piece", "to", "ranks", "mandatory"]

    def test_one_list_for_both_players_is_to(self) -> None:
        promotion = {"piece": "P", "to_by_player": {"0": ["Q"], "1": ["Q"]}}
        assert current_spelling(_rules(promotion=promotion))["rules"]["promotion"] == {
            "piece": "P", "to": ["Q"],
        }
        apart = {"piece": "P", "to_by_player": {"0": ["Q"], "1": ["R"]}}
        assert current_spelling(_rules(promotion=apart))["rules"]["promotion"] == apart

    def test_a_rule_stated_off_goes_and_a_setting_at_its_default_stays(self) -> None:
        rules = current_spelling(_rules(
            en_passant=False, drops={}, check_limit=0, portals=[], turn={"moves": 1},
            wrap={"files": "none", "ranks": "none"}, strike={"effect": "capture"},
            win="checkmate", stalemate="draw", max_plies=400, simultaneous={},
        ))["rules"]
        assert rules == {
            "win": "checkmate", "stalemate": "draw", "max_plies": 400, "simultaneous": {},
        }

    def test_a_wrap_is_written_key_by_key(self) -> None:
        def wrap(value: dict[str, Any]) -> Any:
            return current_spelling(_rules(wrap=value))["rules"].get("wrap")

        assert wrap({"files_by_player": {"0": "both", "1": "both"}, "ranks": "none"}) == {
            "files": "both"
        }
        assert wrap({"files": "left", "files_by_player": {"1": "none"}}) == {
            "files_by_player": {"0": "left", "1": "none"}
        }

    def test_the_relations_read_one_promotion_spelling(self) -> None:
        """`relate` names the rules a game has, so a spelling must not read as a rule."""
        chess = _spec("rulesets/chess.json")
        listed = copy.deepcopy(chess)
        listed["rules"]["promotion"] = [listed["rules"]["promotion"]]
        assert relate.identity_key(listed) == relate.identity_key(chess)


# ----------------------------------------------------------------------------- the file


class TestTheFileKeepsItsLayout:
    def test_a_hand_laid_out_file_changes_only_where_its_spelling_did(self) -> None:
        text = (ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")
        line = '"promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]}'
        assert line in text
        older = text.replace(f'"schema_version": {SCHEMA_VERSION},', '"schema_version": 1,')
        older = older.replace(line, line.replace('{"piece"', '[{"piece"') + "]")
        assert older != text
        assert respell_text(older) == text

    def test_a_plain_file_stays_plain(self) -> None:
        spec = _spec("rulesets/torus.json")
        spec["schema_version"] = 5
        spec["rules"]["promotion"] = {"piece": "P", "rank": 9, "to": ["Q", "R", "B", "N"]}
        spec["rules"]["en_passant"] = False
        out = respell_text(json.dumps(spec, indent=2) + "\n")
        assert out == json.dumps(current_spelling(spec), indent=2) + "\n"
        assert json.loads(out)["rules"]["promotion"]["ranks"] == [9]

    def test_a_missing_version_is_written_first(self) -> None:
        text = '{\n  "name": "x",\n  "board": {"width": 8, "height": 8}\n}\n'
        assert respell_text(text) == (
            f'{{\n  "schema_version": {SCHEMA_VERSION},\n  "name": "x",\n'
            '  "board": {"width": 8, "height": 8}\n}\n'
        )

    @pytest.mark.parametrize(
        ("rules", "kept"),
        [
            ('"en_passant": false, "win": "checkmate", "turn": {"moves": 1}', '"win": "checkmate"'),
            ('"win": "checkmate", "en_passant": false', '"win": "checkmate"'),
            ('"en_passant": false, "drops": {}, "win": "checkmate"', '"win": "checkmate"'),
            ('"en_passant": false', ""),
        ],
        ids=["first-and-last", "last", "a-leading-run", "all"],
    )
    def test_a_rule_stated_off_is_cut_with_its_comma(self, rules: str, kept: str) -> None:
        def file(body: str) -> str:
            return f'{{\n  "schema_version": {SCHEMA_VERSION},\n  "rules": {{{body}}}\n}}\n'

        out = respell_text(file(rules))
        assert json.loads(out) == json.loads(file(kept))
        if kept:
            assert out == file(kept)

    def test_the_cli_checks_then_converts(self, tmp_path: Path) -> None:
        path = tmp_path / "old.json"
        spec = _spec("rulesets/chess.json")
        spec["schema_version"] = 1
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        assert spelling.main(["--check", str(path)]) == 1
        assert spelling.main([str(path)]) == 0
        assert spelling.main(["--check", str(path)]) == 0
        assert json.loads(path.read_text(encoding="utf-8")) == current_spelling(spec)
