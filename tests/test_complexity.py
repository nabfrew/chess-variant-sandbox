"""The elegance metric charges a rule only where the compiler keeps one.

`complexity._RULE_ACTIVE` decides from a rule's *value* whether there is a rule at all.
What a rule means lives in `engine/compile.py`, so each clause is held against the
compiled spec rather than against its own comment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from retired_specs import hex_glinski_promoting

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.schema import PLAYED_AS_CAPTURE, strike_changes_anything

ROOT = Path(__file__).resolve().parents[1]

COSTS = {"strike": 2.0, "contacts": 2.5}

KING_STEP = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
KING_STEP_DIRECTIONS = 8


def _archived(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


class Case(NamedTuple):
    base: str
    key: str
    rule: dict[str, Any]
    kept: bool

    def bare(self) -> dict[str, Any]:
        """The archived base with its strike or contact table taken out."""
        spec = _archived(self.base)
        spec["rules"].pop("strike", None)
        spec["rules"].pop("contacts", None)
        return spec

    def spec(self) -> dict[str, Any]:
        spec = self.bare()
        spec["rules"][self.key] = self.rule
        return spec


FULL_CHAIN = [
    {"from": "Q", "to": "R"},
    {"from": "R", "to": "B"},
    {"from": "B", "to": "N"},
    {"from": "N", "to": "P"},
]

PROMOTING_LINKS = [
    {"mover": "P", "onto": "N", "mover_to": "N"},
    {"mover": "N", "onto": "B", "mover_to": "B"},
]

# `hierarchy-fall` is a live demotion chain, `succession` a contact table that promotes
# the mover, and `hierarchy-rise-throw` a promotion chain thrown onto a king step. Each
# live spelling sits beside a dead one that the compiler drops (and, since Q538, that the
# validator refuses, but for the bare `{"effect": "capture"}`).
CASES = {
    "strike as committed": Case(
        "hierarchy-fall", "strike", _archived("hierarchy-fall")["rules"]["strike"], True
    ),
    "strike, plain capture spelled out": Case(
        "hierarchy-fall", "strike", {"effect": "capture"}, False
    ),
    "strike, one link, its victim immune": Case(
        "hierarchy-fall",
        "strike",
        {
            "effect": "demote",
            "position": "stay",
            "chain": [{"from": "N", "to": "P"}],
            "immune": ["K", "N"],
        },
        False,
    ),
    "strike, every victim immune": Case(
        "hierarchy-fall",
        "strike",
        {
            "effect": "demote",
            "position": "swap",
            "chain": FULL_CHAIN,
            "immune": ["K", "Q", "R", "B", "N"],
        },
        False,
    ),
    "strike, one victim of four still reachable": Case(
        "hierarchy-fall",
        "strike",
        {
            "effect": "demote",
            "position": "stay",
            "chain": FULL_CHAIN,
            "immune": ["K", "Q", "R", "B"],
        },
        True,
    ),
    "contacts as committed": Case(
        "succession", "contacts", _archived("succession")["rules"]["contacts"], True
    ),
    "contacts, every target immune": Case(
        "succession",
        "contacts",
        {"position": "advance", "links": PROMOTING_LINKS, "immune": ["K", "N", "B"]},
        False,
    ),
    "contacts, one target of two still reachable": Case(
        "succession",
        "contacts",
        {"position": "advance", "links": PROMOTING_LINKS, "immune": ["K", "N"]},
        True,
    ),
    "contacts, its one survivor immune": Case(
        "succession",
        "contacts",
        {
            "position": "stay",
            "links": [{"mover": "P", "onto": "N", "onto_to": "P"}],
            "immune": ["K", "N"],
        },
        False,
    ),
    "thrown strike as committed": Case(
        "hierarchy-rise-throw",
        "strike",
        _archived("hierarchy-rise-throw")["rules"]["strike"],
        True,
    ),
    # The two dead thrown spellings sit under `advance`, with no `landing`. Under `choose`
    # the validator refuses a rule that keeps nothing alive (Q271), so a landing never
    # stands beside a dropped rule, and `advance` is what that refusal says to write.
    "thrown strike, every victim immune": Case(
        "hierarchy-rise-throw",
        "strike",
        {
            **{
                key: value
                for key, value in _archived("hierarchy-rise-throw")["rules"]["strike"].items()
                if key != "landing"
            },
            "position": "advance",
            "immune": ["K", "P", "N", "B", "R"],
        },
        False,
    ),
    "thrown contacts": Case(
        "hierarchy-rise-throw",
        "contacts",
        {
            "position": "choose",
            "landing": KING_STEP,
            "links": [{"mover": "N", "onto": "B", "onto_to": "N"}],
            "immune": ["K"],
        },
        True,
    ),
    "thrown contacts, its target immune": Case(
        "hierarchy-rise-throw",
        "contacts",
        {
            "position": "advance",
            "links": [{"mover": "N", "onto": "B", "onto_to": "N"}],
            "immune": ["K", "B"],
        },
        False,
    ),
}


def _counts(spec: dict[str, Any]) -> dict[str, float]:
    """Every count but `spec_bytes`, which reads the spelling rather than the rule."""
    counts = measure_complexity(spec).as_dict()
    del counts["spec_bytes"]
    return counts


@pytest.mark.parametrize("label", sorted(CASES))
def test_a_strike_or_contact_table_is_charged_exactly_when_the_compiler_keeps_it(
    label: str,
) -> None:
    """A kept rule costs its entry and its landing. A dropped one measures as no rule.

    Since Q538 the validator refuses every dropped spelling but the bare
    ``{"effect": "capture"}``, so a dropped one is asked of the compiler's leaf; the
    charge is still held for it, because a spec is measured before it is validated.
    """
    case = CASES[label]
    spec = case.spec()
    ids = [piece["id"] for piece in spec["pieces"]]
    assert strike_changes_anything(case.rule, case.key, ids) is case.kept
    if case.kept or case.rule == {"effect": "capture"}:
        assert validate_spec(spec) == []
        assert (compile_spec(spec).strike is not None) is case.kept
    else:
        assert any(PLAYED_AS_CAPTURE in error for error in validate_spec(spec))

    with_rule, without = _counts(spec), _counts(case.bare())
    moved = {name: with_rule[name] - without[name] for name in with_rule
             if with_rule[name] != without[name]}

    expected: dict[str, float] = {}
    if case.kept:
        expected["special_rules"] = COSTS[case.key]
        if "landing" in case.rule:
            assert case.rule["landing"] == KING_STEP
            expected["move_descriptors"] = 1
            expected["distinct_directions"] = KING_STEP_DIRECTIONS
    assert moved == expected


# ---------------------------------------------------------------- schema 16's seams


def _old_wrap_active(value: Any) -> bool:
    """`_RULE_ACTIVE["wrap"]` before schema 16, which read a grid's two axes only."""
    return isinstance(value, dict) and any(
        edge not in (None, "none")
        for axis in ("files", "ranks")
        for edge in [value.get(axis)] + list((value.get(f"{axis}_by_player") or {}).values())
    )


# Every committed and archived grid spec, but schema 16's showcases (which the tests above
# read), split by whether it states a seam: only a spec that states one can differ
# from the old reading, and one case per spec of the others said nothing.
GRIDS = [
    path
    for path in sorted((ROOT / "rulesets").glob("*.json"))
    + sorted((ROOT / "archive").glob("*.json"))
    if "shape" not in json.loads(path.read_text(encoding="utf-8"))["board"]
]
WRAPPED_GRIDS = [
    path for path in GRIDS
    if (json.loads(path.read_text(encoding="utf-8")).get("rules") or {}).get("wrap")
]
UNWRAPPED_GRIDS = [path for path in GRIDS if path not in WRAPPED_GRIDS]


class TestAHexagonsSeamIsCharged:
    """Q281a4: a hexagon's pairs of sides cost `wrap`, and nothing else moved."""

    def test_the_keys_are_every_shapes_wrap_keys(self) -> None:
        from chess_variant_sandbox.spec.complexity import _WRAP_KEYS
        from chess_variant_sandbox.spec.schema import SHAPES, wrap_axes

        assert set(_WRAP_KEYS) == {key for shape in SHAPES for key, _ in wrap_axes(shape)}

    @pytest.mark.parametrize(
        "wrap", [{"nw_se": "both"}, {"n_s": "n"}, {"ne_sw_by_player": {"0": "none", "1": "sw"}}]
    )
    def test_a_hexagon_with_a_seam_charges_wrap(self, wrap: dict[str, Any]) -> None:
        from chess_variant_sandbox.spec.complexity import _RULE_COSTS

        twin = json.loads((ROOT / "archive" / "hex-glinski-hexagon.json").read_text(encoding="utf-8"))
        plain = measure_complexity(twin).special_rules
        twin["rules"]["wrap"] = wrap
        assert validate_spec(twin) == []
        assert measure_complexity(twin).special_rules == plain + _RULE_COSTS["wrap"]

    def test_the_native_twin_loses_the_walled_parents_obstacles_charge(self) -> None:
        from chess_variant_sandbox.spec.complexity import _RULE_COSTS

        def load(name: str) -> dict[str, Any]:
            return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))

        walled = measure_complexity(hex_glinski_promoting()).special_rules
        native = measure_complexity(load("hex-glinski-hexagon")).special_rules
        assert walled - native == _RULE_COSTS["obstacles"]

    @pytest.mark.parametrize(
        "path", WRAPPED_GRIDS, ids=lambda path: f"{path.parent.name}/{path.stem}"
    )
    def test_every_committed_grid_reads_as_before(self, path: Path) -> None:
        from chess_variant_sandbox.spec.complexity import _RULE_ACTIVE

        spec = json.loads(path.read_text(encoding="utf-8"))
        wrap = spec["rules"]["wrap"]
        active = _old_wrap_active(wrap)
        assert _RULE_ACTIVE["wrap"](wrap) is active
        # And what that reading decides is the charge: 1.5, once, or nothing.
        bare = json.loads(path.read_text(encoding="utf-8"))
        del bare["rules"]["wrap"]
        charged = measure_complexity(spec).special_rules - measure_complexity(bare).special_rules
        assert charged == (1.5 if active else 0.0)

    def test_a_grid_that_states_no_seam_is_not_charged_for_saying_none(self) -> None:
        assert UNWRAPPED_GRIDS, "the corpus has grids without a seam; so does this test"
        for path in UNWRAPPED_GRIDS:
            spec = json.loads(path.read_text(encoding="utf-8"))
            plain = measure_complexity(spec).special_rules
            spec.setdefault("rules", {})["wrap"] = {"files": "none", "ranks": "none"}
            assert measure_complexity(spec).special_rules == plain, path.stem
