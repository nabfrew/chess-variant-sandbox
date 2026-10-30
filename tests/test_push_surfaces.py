"""The push rule's round-1 surfaces (Q1074b5, `docs/plan-push.md` §4): what `complexity`
charges for it, that `distance` sees each of its knobs, and that `relate` reads a pushing
game apart from its twin and two spellings of one push game as one.

Every spec here is chess with a push stated on some of its pieces; chess is used for its
army only, and nothing asserted depends on how chess plays.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.relate import identity_key, normal_form, relate, rules_involved

ROOT = Path(__file__).resolve().parents[1]


def _chess() -> dict[str, Any]:
    return load_spec(ROOT / "rulesets" / "chess.json")


def _sumo(push: Any = True, pieces: str = "NBRQ", **rule: Any) -> dict[str, Any]:
    """Chess whose ``pieces`` push with every move (``push`` as written on each entry)."""
    spec = _chess()
    for piece in spec["pieces"]:
        if piece["id"] in pieces:
            for move in piece["moves"]:
                move["push"] = copy.deepcopy(push)
    if rule:
        spec["rules"]["push"] = rule
    return spec


def _valid(spec: dict[str, Any]) -> dict[str, Any]:
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _weighed(chain: str, strength: dict[str, int] | None = None) -> dict[str, Any]:
    """Rooks and queens that push, everything weighed: a weight a piece, by its value."""
    spec = _sumo(pieces="RQ", chain=chain)
    for piece in spec["pieces"]:
        piece["weight"] = {"K": 1, "P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}[piece["id"]]
        if strength and piece["id"] in strength:
            piece["strength"] = strength[piece["id"]]
    return spec


# --------------------------------------------------------------------------- complexity


def _special(spec: dict[str, Any]) -> float:
    return measure_complexity(_valid(spec)).special_rules


def test_a_push_is_charged_once_for_the_game_however_many_moves_push() -> None:
    chess = _special(_chess())
    assert _special(_sumo(pieces="R")) == pytest.approx(chess + 1.0)
    assert _special(_sumo()) == pytest.approx(chess + 1.0)


def test_the_chain_and_a_strength_apart_are_charged_on_top() -> None:
    chess = _special(_chess())
    assert _special(_sumo(chain="row")) == pytest.approx(chess + 1.5)
    assert _special(_weighed("weight")) == pytest.approx(chess + 2.0)
    assert _special(_weighed("weight", {"R": 7})) == pytest.approx(chess + 2.25)
    # A strength written equal to the weight is its off value, and costs nothing.
    assert _special(_weighed("weight", {"R": 5})) == pytest.approx(chess + 2.0)
    # `whose`, `captures`, `along` and the edge are clauses of the push's sentence.
    for push in ({"whose": "any"}, {"captures": True}, {"along": "rank"}):
        assert _special(_sumo(push, pieces="N")) == pytest.approx(chess + 1.0), push
    assert _special(_sumo(edge="blocks")) == pytest.approx(chess + 1.0)


def test_a_pusher_s_list_is_a_line_of_the_rules_table_in_either_spelling() -> None:
    listed = _sumo(pieces="R")
    for piece in listed["pieces"]:
        if piece["id"] == "R":
            piece["pushes"] = ["P", "N"]
    shorthand = _sumo(pieces="R", chain="row")
    for piece in shorthand["pieces"]:
        piece["weight"] = {"R": 3, "P": 1, "N": 3}.get(piece["id"], 9)
    chess = measure_complexity(_chess()).capture_restrictions
    assert measure_complexity(_valid(listed)).capture_restrictions == chess + 1
    assert measure_complexity(_valid(shorthand)).capture_restrictions == chess + 1


# ----------------------------------------------------------------------------- distance


def test_a_push_game_is_not_its_twin_without_the_push() -> None:
    assert distance(_chess(), _valid(_sumo())) > 0
    assert features(_chess())["push_share"] == 0.0


@pytest.mark.parametrize(
    ("first", "second", "feature"),
    [
        ({}, {"whose": "any"}, "push_whose"),
        ({}, {"captures": True}, "push_captures"),
        ({}, {"along": "rank"}, "push_along"),
    ],
)
def test_each_move_knob_is_a_feature_of_its_own(first, second, feature) -> None:
    one, other = _valid(_sumo(first or True, pieces="N")), _valid(_sumo(second, pieces="N"))
    assert features(one)[feature] != features(other)[feature]
    assert distance(one, other) > 0


def test_the_rule_s_knobs_the_lists_and_the_numbers_are_features() -> None:
    plain = _valid(_sumo())
    assert features(_valid(_sumo(chain="row")))["push_chain"] == 0.5
    assert features(_valid(_sumo(edge="blocks")))["push_edge_blocks"] == 1.0
    listed = _sumo(pieces="R")
    for piece in listed["pieces"]:
        if piece["id"] == "R":
            piece["pushes"] = ["P"]
    assert features(_valid(listed))["push_lists"] == 1.0
    assert features(_valid(listed))["push_reach"] < features(plain)["push_reach"]
    weighed = features(_valid(_weighed("weight")))
    strong = features(_valid(_weighed("weight", {"R": 8, "Q": 12})))
    assert weighed["push_weight"] > 0 and weighed["push_strength"] == 7.0
    assert strong["push_strength"] == 10.0


def test_the_weight_shorthand_and_its_list_measure_alike() -> None:
    shorthand = _sumo(pieces="R", chain="row")
    for piece in shorthand["pieces"]:
        piece["weight"] = {"R": 3, "P": 1, "N": 3}.get(piece["id"], 9)
    listed = _sumo(pieces="R", chain="row")
    for piece in listed["pieces"]:
        if piece["id"] == "R":
            piece["pushes"] = ["N", "P", "R"]
    assert distance(_valid(shorthand), _valid(listed)) == 0.0


# ------------------------------------------------------------------------------- relate


def test_relate_reads_a_pushing_game_apart_from_its_twin() -> None:
    chess, sumo = _chess(), _valid(_sumo())
    assert identity_key(chess) != identity_key(sumo)
    units = {difference.unit for difference in relate(chess, sumo).differences}
    assert "piece.moves" in units
    assert "piece.push" in rules_involved(normal_form(sumo))
    assert "piece.push" not in rules_involved(normal_form(chess))


@pytest.mark.parametrize("push", [{"whose": "own"}, {"captures": True}, {"along": "file"}])
def test_relate_reads_each_move_knob(push: dict[str, Any]) -> None:
    assert identity_key(_valid(_sumo(pieces="N"))) != identity_key(_valid(_sumo(push, pieces="N")))


def test_relate_reads_lists_weights_and_strengths_as_rules_of_the_piece() -> None:
    listed = _sumo(pieces="R")
    for piece in listed["pieces"]:
        if piece["id"] == "R":
            piece["pushes"] = ["P"]
    relation = relate(_valid(_sumo(pieces="R")), _valid(listed))
    assert [d.unit for d in relation.differences] == ["piece.pushes"]
    weighed, strong = _valid(_weighed("weight")), _valid(_weighed("weight", {"R": 7}))
    assert identity_key(weighed) != identity_key(strong)
    assert [d.unit for d in relate(weighed, strong).differences] == ["piece.other"]


def test_relate_reads_one_push_game_written_two_ways_as_one() -> None:
    plain = _valid(_sumo())
    # The rule at its defaults, and a move's push written out at its defaults.
    assert identity_key(plain) == identity_key(_valid(_sumo(chain="one", edge="removes")))
    assert identity_key(plain) == identity_key(
        _valid(_sumo({"whose": "enemy", "captures": False, "along": "move"}))
    )
    # The weight shorthand and the list it stands for.
    shorthand = _sumo(pieces="R", chain="row")
    for piece in shorthand["pieces"]:
        piece["weight"] = {"R": 3, "P": 1, "N": 3}.get(piece["id"], 9)
    listed = _sumo(pieces="R", chain="row")
    for piece in listed["pieces"]:
        if piece["id"] == "R":
            piece["pushes"] = ["N", "P", "R"]
    assert identity_key(_valid(shorthand)) == identity_key(_valid(listed))
