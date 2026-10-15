"""A value of the wrong JSON type is refused in words, never raised on (Q924, Q926).

`validate_spec` collects errors rather than stopping, so a coherence check runs over a
spec a shape check has already refused. A list or object where a piece id belongs is
unhashable, and a set or dict built from it raised `TypeError`. Each case below drew one
of those crashes (or an `AttributeError` on a non-object) before Q926 and must now draw a
refusal. The last test sweeps every field of two small specs with a wrong-typed value.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from test_schema_refusals import BASE, PORTALS, _patched

from chess_variant_sandbox.spec import validate_spec

UNPLAYED = {"piece": "R", "player": 1, "squares": ["a1"]}

# Q924, Q926a, Q926b: the three defects, with the exact refusal each must draw.
DEFECTS = [
    pytest.param(
        {"setup.mirror": False, "setup.placements.2": UNPLAYED},
        "setup.placements[2].squares 'a1' is already occupied by 'R'",
        id="Q924-two-players-on-one-square-without-a-mirror",
    ),
    pytest.param(
        {"pieces.0.id": ["K"]},
        "pieces[0].id must be a string of 1-2 characters",
        id="Q926a-royal-id-a-list",
    ),
    pytest.param(
        {"rules.contacts": {"immune": ["K"],
                            "links": [{"mover": ["R"], "onto": "R", "onto_to": "K"}]}},
        "rules.contacts.links[0].mover ['R'] is not a declared piece",
        id="Q926b-contact-mover-a-list",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), DEFECTS)
def test_the_three_defects_are_refused_in_words(changes: dict[str, Any], refusal: str) -> None:
    assert refusal in validate_spec(_patched(BASE, changes))


def test_one_army_on_one_square_twice_is_still_refused_with_a_mirror() -> None:
    twice = {"setup.placements.2": {"piece": "R", "squares": ["a1"]}}
    errors = validate_spec(_patched(BASE, twice))
    assert "setup.placements[2].squares 'a1' is already occupied by 'R'" in errors


def test_two_armies_on_different_squares_without_a_mirror_validate() -> None:
    apart = {
        "setup.mirror": False,
        "setup.placements.2": {"piece": "R", "player": 1, "squares": ["a5"]},
        "setup.placements.3": {"piece": "K", "player": 1, "squares": ["c5"]},
    }
    assert validate_spec(_patched(BASE, apart)) == []


# Every field the sweep below found a crash in, as a (path, wrong value) pair on BASE or
# PORTALS. Each must draw at least one refusal.
CRASHED = [
    pytest.param(BASE, {"pieces.0.id": []}, id="piece-id-empty-list"),
    pytest.param(BASE, {"pieces.1.id": {"a": 1}}, id="piece-id-object"),
    pytest.param(BASE, {"pieces.1.captures": [[]]}, id="captures-entry-list"),
    pytest.param(BASE, {"pieces": -1}, id="pieces-a-number"),
    pytest.param(BASE, {"setup": ["x"]}, id="setup-an-array"),
    pytest.param(BASE, {"setup.placements.0.piece": []}, id="placement-piece-list"),
    pytest.param(BASE, {"setup.placements.0.player": []}, id="placement-player-list"),
    pytest.param(BASE, {"setup.placements.0.squares": -1}, id="placement-squares-a-number"),
    pytest.param(BASE, {"setup.placements.0.facing": []}, id="placement-facing-list"),
    pytest.param(BASE, {"rules.castling": {"king": [], "rook": "R"}}, id="castling-king-list"),
    pytest.param(BASE, {"rules.castling": {"king": "K", "rook": []}}, id="castling-rook-list"),
    pytest.param(BASE, {"rules.reach": {"piece": [], "squares": ["a5"]}}, id="reach-piece-list"),
    pytest.param(BASE, {"rules.promotion": {"piece": [], "to": ["R"]}}, id="promotion-piece-list"),
    pytest.param(
        BASE, {"rules.promotion": [{"piece": [], "to": ["R"]}, {"piece": "R", "to": ["K"]}]},
        id="promotion-entry-piece-list",
    ),
    pytest.param(BASE, {"rules.extinction": {"pieces": [[], "R"], "per_type": True}},
                 id="extinction-pieces-entry-list"),
    pytest.param(BASE, {"rules.drops": {"stock": ["x"], "placement": True}},
                 id="drops-stock-an-array"),
    pytest.param(BASE, {"rules.npcs": [{"piece": [], "route": ["a3", "a4"]}]},
                 id="npc-piece-list"),
    pytest.param(
        BASE, {"rules.strike": {"effect": "capture", "immune": [[]]}}, id="strike-immune-list",
    ),
    pytest.param(
        BASE,
        {"rules.strike": {"effect": "capture", "chain": [{"from": [], "to": "R"}]}},
        id="strike-chain-from-list",
    ),
    pytest.param(
        BASE,
        {"rules.strike": {"effect": "capture", "chain": [{"from": "R", "to": []}]}},
        id="strike-chain-to-list",
    ),
    pytest.param(
        BASE, {"rules.strike": {"effect": "capture", "switches_sides": [[]]}},
        id="strike-switches-sides-list",
    ),
    pytest.param(BASE, {"rules.contacts": {"immune": [[]], "links": []}},
                 id="contacts-immune-list"),
    pytest.param(BASE, {"rules.contacts": {"links": -1}}, id="contacts-links-a-number"),
    pytest.param(
        BASE, {"rules.contacts": {"immune": ["K"], "links": [{"mover": "R", "onto": []}]}},
        id="contacts-onto-list",
    ),
    pytest.param(PORTALS, {"rules.portals.0.piece": []}, id="portal-piece-list"),
]


@pytest.mark.parametrize(("base", "changes"), CRASHED)
def test_a_wrong_typed_field_is_refused_not_raised_on(
    base: dict[str, Any], changes: dict[str, Any]
) -> None:
    errors = validate_spec(_patched(base, changes))
    assert errors and all(isinstance(error, str) for error in errors)


WRONG_VALUES: list[Any] = [[], ["x"], {}, {"a": 1}, "x", "", -1, 2.5, True, None, 10**12]


def _paths(node: Any, prefix: tuple[Any, ...] = ()) -> Any:
    yield prefix
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _paths(value, prefix + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _paths(value, prefix + (index,))


def _with(spec: dict[str, Any], path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    changed = copy.deepcopy(spec)
    node: Any = changed
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = copy.deepcopy(value)
    return changed


def _crashes(spec: dict[str, Any]) -> list[str]:
    found = []
    for path in _paths(spec):
        if not path:
            continue
        for value in WRONG_VALUES:
            try:
                errors = validate_spec(_with(spec, path, value))
            except Exception as exc:  # noqa: BLE001 -- the point of the sweep
                found.append(f"{path} = {value!r}: {type(exc).__name__}: {exc}")
                continue
            assert all(isinstance(error, str) for error in errors)
    return found


def _rich(base: dict[str, Any]) -> dict[str, Any]:
    """``base`` with every optional rule the sweep should reach written in."""
    spec = copy.deepcopy(base)
    spec["rules"].update({
        "castling": {"king": "K", "rook": "R"},
        "reach": {"piece": "K", "squares": ["a5"]},
        "promotion": [{"piece": "R", "to": ["K"]}],
        "extinction": {"pieces": ["K", "R"], "per_type": True},
        "strike": {"effect": "capture", "immune": ["K"],
                   "chain": [{"from": "R", "to": "R"}], "switches_sides": ["R"]},
        "contacts": {"immune": ["K"], "links": [{"mover": "R", "onto": "R", "onto_to": "K"}]},
        "drops": {"stock": {"R": 1}, "placement": True},
        "npcs": [{"piece": "R", "route": ["a3", "a4"], "immune": ["K"]}],
    })
    spec["setup"]["placements"][0]["facing"] = "forward"
    return spec


@pytest.mark.parametrize(
    "base", [BASE, PORTALS, _rich(BASE)], ids=["base", "portals", "every-rule"]
)
def test_no_field_of_a_spec_raises_on_a_wrong_typed_value(base: dict[str, Any]) -> None:
    assert _crashes(base) == []
