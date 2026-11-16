"""R-L6c: `pieces[].region` binds every way a piece comes to stand, and `npcs[].immune` binds a
player's piece of a wild type (`docs/reference/engine-and-spec-record.md` *R-L6c*).

One test per site where a piece comes to stand other than by its own move's landing (which the
movement tables confine): a promotion's target, a strike's survivor, a link's or a switch's
changed capturer, a drop, a gate, a displacement; and each beside its oracle walk
(`tests/region_oracle.py`: the spec's moves are its region-free twin's less those that leave a
player's piece newly outside its region, at every node). A rebirth home and a blast's
placement remove a piece that may not stand where they would put it, as an occupied square
does; they have tests of their own. The fuzz finds F1 to F4 are `tests/test_fuzz_lifted.py`'s.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_DISPLACE,
    MOVE_DROP,
    MOVE_PROMOTION,
    move_from,
    move_gate,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.fuzz_lifted import generate
from chess_variant_sandbox.spec.schema import validate_spec
from tests.region_oracle import walk

ROOT = Path(__file__).resolve().parents[1]
LOW = [f"{f}{r}" for f in "abcdef" for r in (1, 2, 3)]  # ranks 1-3 of the six by six


def _six(**rules: Any) -> dict[str, Any]:
    spec = copy.deepcopy(generate.SIX)
    spec["rules"].update(rules)
    return spec


def _piece(spec: dict[str, Any], piece_id: str) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["id"] == piece_id)


def _valid(spec: dict[str, Any]) -> dict[str, Any]:
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _sq(engine: Engine, name: str) -> int:
    return engine.spec.square_index("abcdefghijklmnop".index(name[0]), int(name[1:]) - 1)


# ------------------------------------------------------------------------------- promotion

def test_a_promotion_target_is_offered_only_inside_its_region() -> None:
    spec = _six()
    _piece(spec, "B")["region"] = LOW + ["a6", "c6"]
    engine = Engine(_valid(spec))
    bishop = engine.spec.piece_ids.index("B")
    for pawn, expected in (("c5", {"c6"}), ("d5", set())):
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), pawn: ("P", 0)})
        landed = {move_to(m) for m in engine.legal_moves(state)
                  if move_type(m) == MOVE_PROMOTION and move_promotion(m) == bishop}
        assert landed == {_sq(engine, name) for name in expected}
    leaves, barred = walk(spec, engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "d5": ("P", 0),
                                                    "b5": ("P", 0), "e2": ("P", 1)}), 3)
    assert leaves > 0 and barred > 0


def test_a_mandatory_promotion_with_every_target_barred_is_no_move() -> None:
    spec = _six(promotion={"piece": "P", "to": ["B"]})
    _piece(spec, "B")["region"] = LOW
    engine = Engine(_valid(spec))
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "d5": ("P", 0)})
    assert not [m for m in engine.legal_moves(state) if move_from(m) == _sq(engine, "d5")]


# ---------------------------------------------------------------------------------- strike

def test_a_struck_piece_is_never_left_outside_its_new_type_s_region() -> None:
    """Seed 10910's shape: a rook struck in place becomes a bishop kept to ranks 1-3 (the second
    player's ranks 4-6), so a capture that would leave the second player's bishop on rank 2 is
    not offered; one that leaves it on rank 5 is."""
    chain = [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]
    spec = _six(strike={"effect": "demote", "position": "stay", "chain": chain, "immune": ["K"]})
    _piece(spec, "B")["region"] = LOW
    engine = Engine(_valid(spec))
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "a2": ("R", 0), "c2": ("R", 1),
                               "a5": ("R", 0), "c5": ("R", 1)})
    takes = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
    assert (_sq(engine, "a2"), _sq(engine, "c2")) not in takes
    assert (_sq(engine, "a5"), _sq(engine, "c5")) in takes
    leaves, barred = walk(spec, state, 3)
    assert leaves > 0 and barred > 0


def test_a_link_never_turns_its_mover_into_a_type_outside_its_region() -> None:
    spec = _six(contacts={"links": [{"mover": "P", "onto": "P", "mover_to": "B"}], "immune": ["K"]})
    _piece(spec, "B")["region"] = LOW
    engine = Engine(_valid(spec))
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c4": ("P", 0), "d5": ("P", 1),
                               "c2": ("P", 0), "b3": ("P", 1)})
    takes = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
    assert (_sq(engine, "c4"), _sq(engine, "d5")) not in takes
    assert (_sq(engine, "c2"), _sq(engine, "b3")) in takes
    leaves, barred = walk(spec, state, 3)
    assert leaves > 0 and barred > 0


def test_a_switcher_never_changes_sides_onto_a_square_outside_its_new_owner_s_region() -> None:
    """The region is reflected for the second player, so a rook kept to ranks 1-3 that switches
    sides on rank 2 would stand outside its new owner's ranks 4-6: that capture is no move."""
    spec = _six(strike={"switches_sides": ["N", "B", "R"], "immune": ["K"]})
    _piece(spec, "R")["region"] = LOW
    engine = Engine(_valid(spec))
    state = engine.make_state({"d1": ("K", 0), "f6": ("K", 1), "a2": ("R", 0), "c2": ("N", 1)})
    assert _sq(engine, "c2") not in {move_to(m) for m in engine.legal_moves(state) if move_from(m) == _sq(engine, "a2")}
    leaves, barred = walk(spec, state, 3)
    assert leaves > 0 and barred > 0


def test_a_reborn_piece_with_no_home_it_may_stand_on_is_removed() -> None:
    """Circe: a home square the reborn type may not stand on removes it, as an occupied home
    does. A rook struck to a bishop is reborn on a bishop's home; with the bishop kept off it,
    the capture removes it. The validator refuses a placement outside its region, so a home is
    always inside one (this is the core's own reading, played past the validator)."""
    chain = [{"from": "R", "to": "B"}]
    spec = _six(strike={"effect": "demote", "position": "home", "chain": chain, "immune": ["K"]})
    _piece(spec, "B")["region"] = [s for s in LOW if s != "e1"] + ["e6"]
    assert any("is placed outside its region" in error for error in validate_spec(spec))
    _piece(spec, "B")["region"] = LOW
    compiled = compile_spec(_valid(spec))
    bishop_type = compiled.piece_ids.index("B")
    homes = (compiled.square_index(4, 0), compiled.square_index(4, 5))  # e1, and its mirror e6
    stand = tuple(
        tuple(
            tuple(sq != homes[player] and ok for sq, ok in enumerate(row)) if piece_type == bishop_type else row
            for piece_type, row in enumerate(per_player)
        )
        for player, per_player in enumerate(compiled.region_stand)
    )
    engine = Engine(dataclasses.replace(compiled, region_stand=stand))
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c4": ("R", 0), "c5": ("R", 1)})
    take = next(m for m in engine.legal_moves(state)
                if (move_from(m), move_to(m)) == (_sq(engine, "c4"), _sq(engine, "c5")))
    after = engine.apply(state, take)
    assert not any(v and engine.spec.type_of(v) == engine.spec.piece_ids.index("B") for v in after.board)


def test_a_blast_s_struck_piece_is_placed_only_where_it_may_stand() -> None:
    """A blast that strikes (`explode_on_capture.strikes`) places each struck victim where it
    stood, onto an empty square only; a square its new type may not stand on is the same: it
    stays cleared."""
    chain = [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]
    spec = _six(strike={"effect": "demote", "position": "stay", "chain": chain, "immune": ["K"]},
                explode_on_capture={"radius": 1, "strikes": "blast"})
    _piece(spec, "B")["region"] = LOW
    if validate_spec(spec):
        spec = _six(strike={"effect": "demote", "position": "stay", "chain": chain, "immune": ["K"]},
                    explode_on_capture={"radius": 1, "strikes": "blast", "immune": ["K"]})
        _piece(spec, "B")["region"] = LOW
    engine = Engine(compile_spec(spec))
    bishop = engine.spec.piece_ids.index("B")
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "a4": ("R", 0), "c4": ("N", 1),
                               "d5": ("R", 1), "d3": ("R", 1)})
    take = next(m for m in engine.legal_moves(state)
                if (move_from(m), move_to(m)) == (_sq(engine, "a4"), _sq(engine, "c4")))
    after = engine.apply(state, take)
    # Each rook becomes the second player's bishop, kept to ranks 4-6: d5's stands, d3's not.
    d5, d3 = after.board[_sq(engine, "d5")], after.board[_sq(engine, "d3")]
    assert d5 and engine.spec.type_of(d5) == bishop and engine.spec.owner_of(d5) == 1
    assert d3 == 0


def test_a_player_s_piece_placed_outside_its_region_is_refused() -> None:
    spec = _six()
    _piece(spec, "B")["region"] = [s for s in LOW if s != "e1"]
    assert any("is placed outside its region, on e1" in error for error in validate_spec(spec))
    _piece(spec, "B")["region"] = LOW
    assert validate_spec(spec) == []


# ------------------------------------------------------------------------------ drops, gates

def test_a_piece_is_dropped_only_inside_its_region() -> None:
    spec = _six(drops={"from_captures": False, "stock": {"B": 1}})
    _piece(spec, "B")["region"] = LOW
    engine = Engine(_valid(spec))
    drops = [m for m in engine.legal_moves(engine.initial_state()) if move_type(m) == MOVE_DROP]
    assert drops and all(move_to(m) < 18 for m in drops)
    width = 6
    compiled = compile_spec(spec)
    assert compiled.drops is not None
    allowed = compiled.drops.regions[0][engine.spec.piece_ids.index("B")]
    assert allowed is not None and all(square // width < 3 for square in allowed)
    leaves, barred = walk(spec, None, 2)
    assert leaves > 0


def test_a_paired_face_or_an_entry_promotion_is_dropped_only_where_it_may_stand() -> None:
    """What a drop enters as is the core's to read (`Game::drop_stands`): a pawn dropped on
    its zone enters as a bishop, kept to ranks 1-3, so no pawn is dropped on rank 6."""
    spec = _six(drops={"from_captures": False, "stock": {"P": 1}}, promote_on_entry=True)
    _piece(spec, "B")["region"] = LOW + ["a6"]
    spec["rules"]["promotion"] = {"piece": "P", "to": ["B"], "squares": ["a6", "b6", "c6", "d6", "e6", "f6"]}
    if validate_spec(spec):
        spec["rules"]["promotion"] = {"piece": "P", "to": ["B"]}
    engine = Engine(_valid(spec))
    pawn = engine.spec.piece_ids.index("P")
    state = engine.make_state({"a1": ("K", 0), "f3": ("K", 1)}, hands=([0, 0, 0, 0, 1], [0] * 5))
    drops = {move_to(m) for m in engine.legal_moves(state) if move_type(m) == MOVE_DROP and move_promotion(m) == pawn}
    sixth = {_sq(engine, f"{f}6") for f in "abcdef"}
    assert drops & sixth == {_sq(engine, "a6")} - {_sq(engine, "f6")}


def test_a_gated_piece_enters_only_where_it_may_stand() -> None:
    """Gating (S1): a piece enters on the square its gating piece left, only if it may stand
    there: the bishop is kept off d1 to f1, so only the rook leaving a1 lets it in."""
    spec = _six(drops={"from_captures": False, "stock": {"B": 1}, "gating": {"ranks": [0]}})
    _piece(spec, "B")["region"] = ["a1", "b1", "c1"] + [s for s in LOW if s[1] != "1"]
    spec["setup"]["placements"] = [
        {"piece": "R", "squares": ["a1"]}, {"piece": "K", "squares": ["d1"]},
        {"piece": "N", "squares": ["f1"]}, {"piece": "P", "squares": ["b2", "c2", "e2"]}]
    engine = Engine(_valid(spec))
    gates = {move_from(m) for m in engine.legal_moves(engine.initial_state()) if move_gate(m)}
    assert gates == {_sq(engine, "a1")}
    leaves, barred = walk(spec, None, 2)
    assert leaves > 0 and barred > 0


# ---------------------------------------------------------------------------- displacement

def test_a_displacement_never_shifts_a_piece_out_of_its_region() -> None:
    """S2: in leapfrog with a guard kept to ranks 3-7 on d3, `Fd2-d4` would leave the guard on
    d2, so it is no move; the same hop with the guard on d4 (landing d5) is one."""
    spec = json.loads((ROOT / "rulesets" / "leapfrog.json").read_text(encoding="utf-8"))
    spec["pieces"].append({"id": "G", "name": "guard", "value": 1, "captures": [],
                           "region": [f"{f}{r}" for f in "abcdefg" for r in range(3, 8)],
                           "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]})
    engine = Engine(_valid(spec))
    state = engine.make_state({"d2": ("F", 0), "d3": ("G", 1), "a9": ("F", 1)})
    assert not [m for m in engine.legal_moves(state) if move_type(m) == MOVE_DISPLACE]
    clear = engine.make_state({"d3": ("F", 0), "d4": ("G", 1), "a9": ("F", 1)})
    assert [m for m in engine.legal_moves(clear) if move_type(m) == MOVE_DISPLACE]
    leaves, barred = walk(spec, engine.make_state({"d2": ("F", 0), "d3": ("G", 1), "e3": ("G", 0),
                                                    "e2": ("F", 0), "a9": ("F", 1)}), 3)
    assert leaves > 0 and barred > 0


# ------------------------------------------------------------------------------- immunity

def test_an_immunity_binds_a_player_s_piece_of_the_wild_type_and_the_wild_piece_still_eats() -> None:
    """F4: the player's knight may not take the wild knight its list does not name; the wild
    knight still eats the player's knight on its route (the victim's column, `capture_owned`)."""
    spec = _six(npcs=[{"piece": "N", "route": ["b3", "c3", "c4", "b4"], "immune": ["R"]}])
    engine = Engine(_valid(spec))
    knight = engine.spec.piece_ids.index("N")
    state = engine.make_state({"f1": ("K", 0), "f6": ("K", 1), "a1": ("N", 0), "b3": ("N", 2),
                               "c3": ("N", 0)})
    assert engine.spec.owner_of(state.board[13]) == WILD
    moves = engine.legal_moves(state)
    assert not [m for m in moves if move_to(m) == 13 and engine.spec.type_of(state.board[move_from(m)]) == knight]
    quiet = next(m for m in moves if move_from(m) == _sq(engine, "f1"))
    after, _ = engine.apply_with_phase(state, quiet)
    assert engine.spec.owner_of(after.board[_sq(engine, "c3")]) == WILD
    rook_spec = copy.deepcopy(spec)
    rook = Engine(rook_spec)
    rook_state = rook.make_state({"f1": ("K", 0), "f6": ("K", 1), "b1": ("R", 0), "b3": ("N", 2)})
    assert [m for m in rook.legal_moves(rook_state) if move_to(m) == 13]
