"""Engine mutation report, F-engine-4 and F-engine-5: wrap copies and frozen records.

The `compile.py` mutants of `docs/reference/mutation/engine.md` filed under
`tests/test_designer_hexagon.py` (the copies the designer reads across a seam) and
`tests/test_compile_memo.py` (the compiled records being frozen), written beside those
files rather than in them. Each test names the input its mutants could not survive.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from test_engine_defaults_and_variants import _hexagon

from chess_variant_sandbox.engine import compile as compiled_module
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


class TestEverySidesCopyCarriesItsOwnPermissions:
    def test_a_one_way_torus_lists_every_copy_with_each_players_own_crossing(self) -> None:
        """Each player leaves by one file edge and one rank edge, and the two players by
        opposite ones, so every side and corner has its own pair: a mutant that reads the
        other edge's flag or the other player's (`compile.py:2631`-`2647`) moves one."""
        spec = load_spec(RULESETS / "onewaytorus.json")
        assert spec["rules"]["wrap"] == {
            "files_by_player": {"0": "left", "1": "right"},
            "ranks_by_player": {"0": "top", "1": "bottom"},
        }
        compiled = compile_spec(spec)
        width, height = compiled.width, compiled.height
        listed = {one.side: (one.vector, one.permission) for one in compiled.grid_copies}
        assert listed == {
            "left": ((-width, 0), (True, False)),
            "right": ((width, 0), (False, True)),
            "bottom": ((0, -height), (False, True)),
            "top": ((0, height), (True, False)),
            "left+bottom": ((-width, -height), (False, False)),
            "left+top": ((-width, height), (True, False)),
            "right+bottom": ((width, -height), (False, True)),
            "right+top": ((width, height), (False, False)),
        }
        assert [one.side for one in compiled.grid_copies] == list(listed)


class TestAHexagonWithoutSeamsHasCopiesNoOneCrosses:
    def test_every_copy_of_an_unwrapped_hexagon_is_closed_to_both_players(self) -> None:
        """The designer offers a crossing where ``permission[0]`` holds (`compile.py:2612`)."""
        compiled = compile_spec(_hexagon())
        assert len(compiled.copies) == 6
        assert {one.side for one in compiled.copies} == {"n", "s", "ne", "sw", "nw", "se"}
        assert all(one.permission == (False, False) for one in compiled.copies)
        assert compiled.grid_copies == ()


# Every frozen record `compile.py` declares, by the line of its decorator in the report.
_FROZEN = [
    "WrapRule", "HexCopy", "StrikeRule", "ExplosionRule", "DropRule", "ReachRule",
    "ExtinctionRule", "TurnRule", "TerritoryRule", "ScoringRule", "NpcRule", "PortalRule",
    "FoldRule", "PromotionRule", "EntryPromotionRule", "CastlingRule", "SimRule",
    "CompiledSpec",
]


class TestTheCompiledRecordsAreFrozen:
    @pytest.mark.parametrize("name", _FROZEN)
    def test_no_field_can_be_set(self, name: str) -> None:
        """A compiled spec is memoised and shared (`conftest`'s compile memo), so a record
        one caller could change would change every other caller's game."""
        record = getattr(compiled_module, name)
        assert dataclasses.is_dataclass(record)
        field = dataclasses.fields(record)[0].name
        instance = record.__new__(record)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field, None)
