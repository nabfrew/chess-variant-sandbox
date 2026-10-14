"""The reader of the rule building blocks a game uses (`docs/archive/plan-meta-game.md`, E2).

`rules_used.used()` must agree with `complexity._count_special_rules` on every
committed spec, asked one key at a time, and `available()` is the schema's own count:
five movement kinds plus every rule key `complexity._RULE_COSTS` charges for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import complexity, load_spec, rules_used, schema

ROOT = Path(__file__).resolve().parents[1]

ALL_SPECS = sorted(
    (path for folder in ("rulesets", "archive") for path in (ROOT / folder).glob("*.json")),
    key=lambda path: path.stem,
)


def _minimal_spec(rules: dict[str, Any], pieces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A spec skeleton with nothing in it but what `used()` reads: `pieces` and `rules`."""
    return {"pieces": pieces or [], "rules": rules}


# The test's own table of what a stated rule costs, written out here and read from nowhere
# in the package. It is the weights of `complexity._RULE_COSTS`, so a change to a weight
# fails this file until the table is changed with it, on purpose.
COSTS: dict[str, float] = {
    "win": 1.0, "stalemate": 1.0, "checkmate": 1.0,
    "promotion": 1.0, "en_passant": 1.0, "castling": 1.0,
    "drops": 2.0, "reach": 1.0, "extinction": 1.0, "bare_royal": 1.0, "check_limit": 1.0,
    "must_capture": 1.0, "forbid_check": 1.0, "royal_facing": 1.0,
    "explode_on_capture": 1.5, "forbid_immobility": 1.0, "scoring": 1.5, "territory": 2.5,
    "turn": 1.0, "wrap": 1.5, "obstacles": 1.0, "strike": 2.0, "contacts": 2.5,
    "npcs": 2.5, "promote_on_entry": 1.0, "portals": 2.5,
}
DEFAULTS = {"win": "checkmate", "stalemate": "draw", "checkmate": "loss", "royal_facing": "allowed"}


def _stated_and_live(key: str, value: Any) -> bool:
    """Whether a rule is one a player is told: the plain reading, not `complexity`'s code.

    A flag or block counts when it is present, not false or empty, and not the default.
    Four keys have a spelling that is present and still no rule, and each is read here
    the short way, which is enough for what the committed specs state.
    """
    if key in DEFAULTS:
        return bool(value != DEFAULTS[key])
    if key == "turn":
        return isinstance(value, dict) and (
            value.get("moves", 1) > 1 or any(length > 1 for length in value.get("ramp", []))
        )
    if key == "wrap":
        edges = [edge for name, edge in value.items() if not name.endswith("_by_player")]
        for table in (edge for name, edge in value.items() if name.endswith("_by_player")):
            edges.extend(table.values())
        return any(edge != "none" for edge in edges)
    if key == "strike":
        return (
            value.get("effect", "capture") != "capture"
            or value.get("position") in ("stay", "home")
            or bool(value.get("switches_sides"))
        )
    return bool(value)


def _independent_keys(spec: dict[str, Any]) -> set[str]:
    return {
        key for key, value in spec.get("rules", {}).items()
        if key in COSTS and _stated_and_live(key, value)
    }


@pytest.mark.parametrize("path", ALL_SPECS, ids=lambda path: path.stem)
def test_used_rule_keys_cost_the_same_as_count_special_rules(path: Path) -> None:
    spec = load_spec(path)
    # `used()` decides each key by asking `complexity._count_special_rules`, so checking its
    # keys against that function would check the function against itself. Both are
    # checked against `COSTS` and `_independent_keys` above, which share nothing with them.
    expected_keys = _independent_keys(spec)
    assert rules_used.used(spec) - set(schema.MOVE_KINDS) == expected_keys
    rules = spec.get("rules", {})
    # `portals` costs a flat rate per key plus what one key cannot say (a further pair, a
    # mover, a one-way or closed passage), charged by `_portal_extras` (Q771); the first
    # specs to state any of those are Q772a's showcases. Its own tests are in test_complexity.
    extras = complexity._portal_extras(rules.get("portals"))
    assert complexity._count_special_rules(rules) == (
        sum(COSTS[key] for key in expected_keys) + extras
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("win", "checkmate"),
        ("stalemate", "draw"),
        ("checkmate", "loss"),
        ("royal_facing", "allowed"),
    ],
)
def test_a_key_spelled_at_its_default_is_not_used(key: str, value: Any) -> None:
    spec = _minimal_spec({key: value})
    assert key not in rules_used.used(spec)


@pytest.mark.parametrize(
    "turn_rule",
    [{"moves": 1}, {"moves": 1, "ramp": [1, 1]}],
)
def test_a_turn_of_one_move_is_not_used(turn_rule: dict[str, Any]) -> None:
    spec = _minimal_spec({"turn": turn_rule})
    assert "turn" not in rules_used.used(spec)


def test_a_departure_from_default_is_used() -> None:
    spec = _minimal_spec({"win": "royal_capture", "turn": {"moves": 2}})
    assert {"win", "turn"} <= rules_used.used(spec)


def test_a_movement_kind_is_used_when_a_piece_declares_it() -> None:
    spec = _minimal_spec(
        {},
        pieces=[{"moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]}],
    )
    assert rules_used.used(spec) == {"slide"}


def test_available_holds_thirty_one_names_today() -> None:
    # Thirty-one since schema 19 added `portals` (Q771).
    names = rules_used.available()
    assert len(names) == 31
    assert len(set(names)) == 31
    assert names[: len(schema.MOVE_KINDS)] == schema.MOVE_KINDS


def test_available_is_exhaustive_over_charged_rule_costs() -> None:
    rule_keys = set(rules_used.available()) - set(schema.MOVE_KINDS)
    assert rule_keys == {key for key, cost in complexity._RULE_COSTS.items() if cost > 0}


def test_available_gains_a_key_when_a_test_charges_one(monkeypatch: pytest.MonkeyPatch) -> None:
    before = rules_used.available()
    patched = dict(complexity._RULE_COSTS)
    patched["a_key_this_test_made_up"] = 1.0
    monkeypatch.setattr(complexity, "_RULE_COSTS", patched)
    after = rules_used.available()
    assert len(after) == len(before) + 1
    assert "a_key_this_test_made_up" in after
