"""`probe-no-sliders` clips a sweep the way it clips a slide (Q244).

The probe removes reach along a ray. A sweep walks a ray, so a sweeper that kept its
range would keep the very power the probe exists to take away. Pinned on the sweep
showcase, `archive/chess-sweeping-bishops.json`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import uses_sweep
from chess_variant_sandbox.spec.transform import all_slides_to_leaps

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sweeping() -> dict[str, Any]:
    return json.loads((ROOT / "archive" / "chess-sweeping-bishops.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


def _piece(spec: dict[str, Any], name: str) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["name"] == name)


class TestASweepIsASlider:
    def test_no_sweep_survives_the_probe(self, sweeping: dict[str, Any]) -> None:
        assert uses_sweep(sweeping)
        mutant = all_slides_to_leaps(copy.deepcopy(sweeping))
        assert not uses_sweep(mutant)

    def test_the_sweeper_becomes_a_one_step_leap_on_the_same_lines(
        self, sweeping: dict[str, Any]
    ) -> None:
        before = _piece(sweeping, "bishop")["moves"]
        after = _piece(all_slides_to_leaps(copy.deepcopy(sweeping)), "bishop")["moves"]
        assert len(after) == len(before)
        for was, now in zip(before, after, strict=True):
            assert now["kind"] == "leap"
            assert "range" not in now and "min_range" not in now
            unchanged = {k: v for k, v in was.items() if k not in ("kind", "range", "min_range")}
            assert {k: v for k, v in now.items() if k != "kind"} == unchanged

    def test_the_sweeper_is_clipped_exactly_as_the_slider_beside_it(
        self, sweeping: dict[str, Any], chess: dict[str, Any]
    ) -> None:
        """Both bishops leave the probe the same piece, sweeping or sliding."""
        from_sweep = _piece(all_slides_to_leaps(copy.deepcopy(sweeping)), "bishop")
        from_slide = _piece(all_slides_to_leaps(copy.deepcopy(chess)), "bishop")
        assert from_sweep["moves"] == from_slide["moves"]

    def test_the_mutant_validates_and_compiles(self, sweeping: dict[str, Any]) -> None:
        mutant = all_slides_to_leaps(copy.deepcopy(sweeping))
        assert validate_spec(mutant) == []
        assert canonical_tables(compile_spec(mutant))

    def test_the_probe_on_chess_is_unmoved(self, chess: dict[str, Any]) -> None:
        """Chess holds no sweep, so the committed probe keeps its hash."""
        mutant = all_slides_to_leaps(copy.deepcopy(chess))
        clipped_slides_only = copy.deepcopy(chess)
        for piece in clipped_slides_only["pieces"]:
            for move in piece["moves"]:
                if move["kind"] == "slide":
                    move["kind"] = "leap"
                    move.pop("range", None)
                    move.pop("min_range", None)
        clipped_slides_only["rules"].pop("en_passant", None)
        assert spec_hash(mutant) == spec_hash(clipped_slides_only)
