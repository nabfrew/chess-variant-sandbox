"""A game is named by its normal form, and every game that had a name kept it (Q1008a).

`docs/reference/schema-reboot-assessment.md` §5, steps 1 and 2. What is held here:

* every committed spec's identity is the name it was frozen under (`test_schema_nine.FROZEN`
  pins the hash and the tables; this file pins that the legacy table is what keeps them);
* the normal form, and every other spelling `tests/respell.py` writes, has the original's
  identity and compiles to the original's tables, position keys included;
* the legacy table's generator refuses two spellings of one game, and its merge refuses an
  entry that would rename a game.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from respell import RESPELLINGS, every_respelling
from retired_specs import REBUILT, RETIRED_HASHES
from test_schema_nine import FROZEN

from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import normal
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash, spelling_hash
from chess_variant_sandbox.spec.legacy_names import LEGACY_NAMES
from chess_variant_sandbox.spec.normal import (
    NameCollision,
    legacy_table,
    merge_legacy,
    normal_form,
    normal_hash,
)
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, stored_spec_errors
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = sorted(FROZEN)


def _spec(path: str) -> dict[str, Any]:
    return load_spec(ROOT / path)


def _tables(spec: dict[str, Any]) -> dict[str, Any]:
    return canonical_tables(compile_spec(spec))


def _admitted(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` stamped with a version the validator takes, its own if it will do.

    A portal game's normal form writes its group moves as spelling 30's stand-in piece
    (`schema.portal_stand_in_shape`), which the validator refuses since spelling 31, so it is
    read back through the converter, which turns it into the group's moves again.
    """
    if stored_spec_errors(spec):
        spec = dict(spec, schema_version=SCHEMA_VERSION)
    if stored_spec_errors(spec):
        spec = current_spelling(spec)
    assert stored_spec_errors(spec) == []
    return spec


class TestEveryNameIsKept:
    def test_every_frozen_name_is_in_the_legacy_table(self) -> None:
        """Every game frozen before Q1008a keeps its name through the table. A game added since
        (Q1007e's two, 2026-10-24) has no legacy name: it is named by its normal form, which
        must be no key of the table (that would rename it)."""
        kept = set(LEGACY_NAMES.values())
        missing = {
            path: digest for path, (digest, _) in FROZEN.items()
            if digest not in kept
            and not (normal_hash(_spec(path)) == digest and digest not in LEGACY_NAMES)
        }
        assert not missing

    @pytest.mark.parametrize("name", sorted(REBUILT))
    def test_a_retired_game_keeps_its_name(self, name: str) -> None:
        assert spec_hash(REBUILT[name]()) == RETIRED_HASHES[name]

    def test_a_game_with_no_legacy_name_is_named_by_its_normal_form(self) -> None:
        spec = _spec("rulesets/chess.json")
        spec["rules"]["max_plies"] = 399
        assert normal_hash(spec) not in LEGACY_NAMES
        assert spec_hash(spec) == normal_hash(spec) != spelling_hash(spec)

    def test_the_table_is_its_own_rendering(self) -> None:
        """The module is generated: it parses back to exactly what it renders."""
        text = normal.LEGACY_MODULE.read_text(encoding="utf-8")
        assert text == normal.render_legacy_module(LEGACY_NAMES)


class TestTheNormalForm:
    def test_it_takes_out_the_version_and_the_flavour(self) -> None:
        spec = _spec("rulesets/chess.json")
        form = normal_form(spec)
        assert set(form) == {"board", "pieces", "setup", "rules"}

    def test_it_never_changes_its_argument(self) -> None:
        spec = every_respelling(_spec("rulesets/shogi.json"))
        before = copy.deepcopy(spec)
        normal_form(spec)
        assert spec == before

    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_it_is_a_fixed_point(self, path: str) -> None:
        form = normal_form(_spec(path))
        assert normal_form(form) == form

    def test_it_survives_malformed_json(self) -> None:
        for spec in (
            {}, {"rules": None}, {"pieces": "x", "rules": {"promotion": 3, "wrap": "x"}},
            {"board": [], "setup": {"placements": [1]}, "rules": {"turn": {"moves": "a"}}},
        ):
            normal_form(spec)
            spec_hash(spec)

    def test_a_live_rule_is_kept(self) -> None:
        """What looks like a default inside a rule that is on stays: `{}` is no rule."""
        rules = {
            "turn": {"moves": 2}, "wrap": {"files": "both"},
            "strike": {"effect": "demote", "chain": [{"from": "Q", "to": "R"}]},
            "simultaneous": {}, "explode_on_capture": {"radius": 1},
        }
        assert normal_form({"rules": rules})["rules"] == rules

    def test_two_spellings_of_a_wrap_are_one(self) -> None:
        both = {"files_by_player": {"0": "both", "1": "both"}, "ranks": "none"}
        assert normal_form({"rules": {"wrap": both}})["rules"] == {"wrap": {"files": "both"}}
        split = {"files": "left", "files_by_player": {"1": "none"}}
        assert normal_form({"rules": {"wrap": split}})["rules"] == {
            "wrap": {"files_by_player": {"0": "left", "1": "none"}}
        }

    def test_a_promotion_zone_is_ranks(self) -> None:
        spec = {"board": {"height": 8}, "rules": {"promotion": {"piece": "P", "to": ["Q"]}}}
        assert normal_form(spec)["rules"]["promotion"] == [
            {"piece": "P", "to": ["Q"], "ranks": [7]}
        ]

    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_it_compiles_to_the_files_tables(self, path: str) -> None:
        """The normal form, put back into a file, is the same game key for key."""
        spec = _spec(path)
        form = _admitted(
            dict(normal_form(spec), name="normal", schema_version=spec["schema_version"])
        )
        assert spec_hash(form) == spec_hash(spec)
        assert _tables(form) == _tables(spec)


class TestAnotherSpellingIsTheSameGame:
    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_every_respelling_keeps_the_identity(self, path: str) -> None:
        spec = _spec(path)
        for name, respell in RESPELLINGS.items():
            assert spec_hash(respell(spec)) == spec_hash(spec), name
        assert spec_hash(every_respelling(spec)) == spec_hash(spec)

    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_every_respelling_at_once_compiles_to_the_same_tables(self, path: str) -> None:
        spec = _spec(path)
        assert _tables(_admitted(every_respelling(spec))) == _tables(spec)

    @pytest.mark.slow
    @pytest.mark.parametrize("path", LIBRARY, ids=lambda p: Path(p).stem)
    def test_each_respelling_compiles_to_the_same_tables(self, path: str) -> None:
        spec = _spec(path)
        tables = _tables(spec)
        for name, respell in RESPELLINGS.items():
            assert _tables(_admitted(respell(spec))) == tables, name


class TestTheGenerator:
    def test_two_spellings_of_one_game_fail_generation(self) -> None:
        spec = _spec("rulesets/chess.json")
        with pytest.raises(NameCollision, match="first.*second"):
            legacy_table([("first", spec), ("second", every_respelling(spec))])

    def test_one_spelling_twice_is_one_entry(self) -> None:
        spec = _spec("rulesets/chess.json")
        table = legacy_table([("a", spec), ("b", copy.deepcopy(spec))])
        assert table == {normal_hash(spec): spelling_hash(spec)}

    def test_the_table_names_every_re_spelt_file(self) -> None:
        """The library is named by the table, never by regenerating it from the files.

        Until Q1008b re-spelt `rulesets/` and `archive/`, the library regenerated the table's
        rows; since then each file's spelling hash is no name at all, and what holds is that
        its normal form is a key of the table whose value is the name it was frozen under.
        A game added since has no row and is named by its normal form, as any new game is.
        """
        def name(path: str) -> str:
            normal = normal_hash(_spec(path))
            return LEGACY_NAMES.get(normal, normal)

        named = {path: name(path) for path in LIBRARY}
        assert named == {path: digest for path, (digest, _) in FROZEN.items()}

    def test_a_merge_never_renames_a_game(self) -> None:
        table = {"aaaaaaaaaaaa": "111111111111"}
        assert merge_legacy(table, {"bbbbbbbbbbbb": "222222222222"}) == {
            "aaaaaaaaaaaa": "111111111111", "bbbbbbbbbbbb": "222222222222",
        }
        assert merge_legacy(table, {"aaaaaaaaaaaa": "111111111111"}) == table
        with pytest.raises(NameCollision):
            merge_legacy(table, {"aaaaaaaaaaaa": "333333333333"})

    def test_add_merges_a_dump_and_stops_on_a_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = tmp_path / "legacy_names.py"
        monkeypatch.setattr(normal, "LEGACY_MODULE", module)
        spec = _spec("rulesets/chess.json")
        spec["rules"]["max_plies"] = 399  # a game the table does not know
        dump = tmp_path / "specs.jsonl"
        dump.write_text(json.dumps(spec) + "\n", encoding="utf-8")
        assert normal.main(["add", str(dump)]) == 0
        assert f'"{normal_hash(spec)}": "{spelling_hash(spec)}"' in module.read_text("utf-8")

        module.unlink()
        clash = every_respelling(spec)
        dump.write_text(json.dumps(spec) + "\n" + json.dumps(clash) + "\n", encoding="utf-8")
        assert normal.main(["add", str(dump)]) == 1
        assert not module.exists()
