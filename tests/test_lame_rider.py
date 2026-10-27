"""Q1007a: the lame rider, `lame` with a `range` (`docs/archive/plan-schema-31.md` §2.2, §7).

Jump after jump along the vector, each jump blocked by a piece on its leg (the square
`compile._lame_path` names) and each landing ending the ride when occupied, captured if it
may be. `range` counts jumps (0 unlimited, 1 the single lame leap every stored file means);
`min_range` is the nearest jump it may stop on, the nearer landings still blocking. Beside it
the nightrider of option 1, a `slide` over a non-unit vector, which is blocked only where it
would land and which existed already.

The hand positions are on small specs built here; `_rider_targets` below is a second
statement of the rule, a declared test oracle (CLAUDE.md rule 3), that the core is held to
over seeded random games.
"""

from __future__ import annotations

import copy
import json
import random
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import _lame_ride, compile_spec
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import move_sentence

ROOT = Path(__file__).resolve().parent.parent

KING = {"id": "K", "name": "king", "value": 0, "royal": True,
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]}
ROOK_RIDER = {"kind": "lame", "vectors": [[2, 0]], "symmetry": "rotate4", "range": 0}
NIGHT_RIDER = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "range": 0}


def probe(pieces: list[dict[str, Any]], rules: dict[str, Any] | None = None,
          width: int = 8, height: int = 8, validate: bool = True) -> dict[str, Any]:
    spec = {
        "name": "lame rider probe",
        "board": {"width": width, "height": height},
        "pieces": [copy.deepcopy(KING), *copy.deepcopy(pieces)],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": [f"{chr(ord('a') + width - 1)}{height}"]},
        ]},
        "rules": {"win": "checkmate", **(rules or {})},
    }
    if validate:
        assert validate_spec(spec) == []
    return spec


def rider(pid: str, move: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"id": pid, "name": f"rider {pid}", "value": 4, "moves": [copy.deepcopy(move)],
            **extra}


@cache
def _engine(text: str) -> Engine:
    return Engine(json.loads(text))


def engine_of(spec: dict[str, Any]) -> Engine:
    return _engine(json.dumps(spec, sort_keys=True))


_MOVE = re.compile(r"^~?[A-Za-z]([a-z]\d+)[-x]([a-z]\d+)")


def targets(engine: Engine, state: GameState, origin: str) -> set[str]:
    """Where the piece on ``origin`` may go, by the legal list's own words."""
    found = set()
    for text in engine.move_texts(state):
        matched = _MOVE.match(text)
        assert matched, text
        if matched.group(1) == origin:
            found.add(matched.group(2))
    return found


# ------------------------------------------------------------------- the spelling


def test_range_and_min_range_are_accepted_on_lame_and_still_refused_on_leap() -> None:
    assert validate_spec(probe([rider("R", {**ROOK_RIDER, "min_range": 2})])) == []
    spec = probe([rider("R", {**ROOK_RIDER, "kind": "leap"})], validate=False)
    assert any("range means nothing on a leap" in e for e in validate_spec(spec))


def test_a_one_square_lame_rider_is_refused_as_the_one_square_lame_leap_is() -> None:
    spec = probe([rider("R", {**ROOK_RIDER, "vectors": [[1, 0]]})], validate=False)
    assert any("one-square 'lame' leap" in e for e in validate_spec(spec))


def test_range_one_is_the_lame_leap_every_stored_file_means() -> None:
    """The default is 1, so writing it is the same game, by name and by table."""
    horse = rider("H", {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"})
    spelt = rider("H", {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8",
                        "range": 1})
    assert spec_hash(probe([horse])) == spec_hash(probe([spelt]))
    assert canonical_tables(compile_spec(probe([horse]))) == canonical_tables(
        compile_spec(probe([spelt])))


def test_a_single_jump_builds_the_two_square_ray_it_always_had() -> None:
    """`_lame_ride` at one jump: `(leg, landing)` with modes `(0, mode)`, as before Q1007a."""

    def landing(f: int, r: int) -> int | None:
        return r * 8 + f if 0 <= f < 8 and 0 <= r < 8 else None

    d4 = 3 * 8 + 3
    assert _lame_ride(landing, 3, 3, d4, 1, 2, 8, 3, 1, 1, None, frozenset()) == (
        (4 * 8 + 3, 5 * 8 + 4), (0, 3))
    assert _lame_ride(landing, 3, 3, d4, 1, 2, 8, 3, 1, 8, None, frozenset()) == (
        (4 * 8 + 3, 5 * 8 + 4, 6 * 8 + 4, 7 * 8 + 5), (0, 3, 0, 3))


# ------------------------------------------------------------------ hand positions


def test_a_leg_blocks_a_landing_blocks_and_is_taken() -> None:
    engine = engine_of(probe([rider("R", ROOK_RIDER)]))
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d4": ("R", 0),
                               "d7": ("R", 0), "e4": ("R", 1), "b6": ("R", 1)})
    # d4: up stops at d7's leg (own piece on d7 is the second jump's leg); right is blocked
    # at its leg e4; down rides d2; left rides b4 and would go on but the board ends.
    assert targets(engine, state, "d4") == {"d6", "d2", "b4"}
    # d7: down d5 then d4's own rook blocks the next leg... the leg d4 is the second jump's
    # leg, so d5 alone; left rides b7; right rides f7 and h7.
    assert targets(engine, state, "d7") == {"d5", "b7", "f7", "h7"}
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d2": ("R", 0),
                               "d6": ("R", 1), "f2": ("R", 0)})
    # d2 up: d4 empty, then d6 holds an enemy: taken, and the ride ends there.
    # d2 right: f2 holds its own side's rook: the ride ends before it.
    assert targets(engine, state, "d2") == {"d4", "d6", "b2"}
    assert "Rd2xd6" in engine.move_texts(state)


def test_a_nightrider_counted_by_hand() -> None:
    """A lame nightrider on d4 of an empty board: twelve landings, each found by hand."""
    engine = engine_of(probe([rider("N", NIGHT_RIDER)]))
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d4": ("N", 0)})
    # (1, 2) e6 f8; (-1, 2) c6 b8; (1, -2) e2; (-1, -2) c2; (2, 1) f5 h6; (2, -1) f3 h2;
    # (-2, 1) b5; (-2, -1) b3: each jump's leg one step along its longer axis.
    assert targets(engine, state, "d4") == {
        "e6", "f8", "c6", "b8", "e2", "c2", "f5", "h6", "f3", "h2", "b5", "b3"}
    assert engine.perft(state, 1) == 12 + 3
    # A piece on e4, the leg of both rightward (2, ±1) jumps, takes four landings away.
    blocked = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d4": ("N", 0),
                                 "e4": ("N", 1)})
    assert targets(engine, blocked, "d4") == {"e6", "f8", "c6", "b8", "e2", "c2", "b5", "b3"}
    assert engine.perft(blocked, 1) == 8 + 3


def test_min_range_skips_the_nearer_landings_and_they_still_block() -> None:
    engine = engine_of(probe([rider("R", {**ROOK_RIDER, "min_range": 2})]))
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d2": ("R", 0)})
    # Up: d4 passed, d6 and d8; right: f2 passed, h2; left and down find no second jump.
    assert targets(engine, state, "d2") == {"d6", "d8", "h2"}
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d2": ("R", 0),
                               "d4": ("R", 1)})
    # The first landing holds an enemy: too near to take, and it ends the ride.
    assert targets(engine, state, "d2") == {"h2"}


def test_a_wall_on_a_leg_ends_the_ride_and_a_wall_on_a_landing_too() -> None:
    engine = engine_of(probe([rider("R", ROOK_RIDER)],
                             rules={"obstacles": {"squares": ["d5", "f2"]}}))
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "d2": ("R", 0)})
    # Up: d4, then the leg d5 is a wall. Right: the landing f2 is a wall.
    assert targets(engine, state, "d2") == {"d4", "b2"}


def test_a_wrapped_ride_stops_on_a_square_it_has_been_on() -> None:
    spec = probe([rider("R", ROOK_RIDER)], rules={"wrap": {"files": "both"}})
    engine = engine_of(spec)
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "b3": ("R", 0)})
    # Round the cylinder each way: d3, f3, h3, then b3 itself ends it; and no move twice.
    assert targets(engine, state, "b3") == {"d3", "f3", "h3", "b1", "b5", "b7"}
    texts = engine.move_texts(state)
    assert len(texts) == len(set(texts))
    # A 7-wide cylinder: each way, the fourth leg falls on the origin and ends the ride.
    spec = probe([rider("R", ROOK_RIDER)], rules={"wrap": {"files": "both"}}, width=7)
    engine = engine_of(spec)
    state = engine.make_state({"a1": ("K", 0), "g8": ("K", 1), "a3": ("R", 0)})
    # Right c3, e3, g3; left f3, d3, b3: an odd width lets it reach the whole rank.
    assert {t for t in targets(engine, state, "a3") if t.endswith("3")} == {
        "b3", "c3", "d3", "e3", "f3", "g3"}
    texts = engine.move_texts(state)
    assert len(texts) == len(set(texts))


def test_a_region_landing_is_passed_over_not_stopped_on() -> None:
    confined = rider("R", ROOK_RIDER, region=["a1", "a5", "e5"])
    spec = probe([confined], validate=False)
    spec["setup"]["placements"][0]["squares"] = ["h1"]
    engine = engine_of(spec)
    state = engine.make_state({"h1": ("K", 0), "h8": ("K", 1), "a1": ("R", 0)})
    # a3 is outside the region: passed over; a5 is the first landing it may stop on.
    assert targets(engine, state, "a1") == {"a5"}
    state = engine.make_state({"h1": ("K", 0), "h8": ("K", 1), "a1": ("R", 0),
                               "a3": ("R", 1)})
    assert targets(engine, state, "a1") == set()


def test_check_through_a_lame_rider_and_its_block_on_a_leg() -> None:
    engine = engine_of(probe([rider("R", ROOK_RIDER)]))
    state = engine.make_state({"e1": ("K", 0), "h8": ("K", 1), "e7": ("R", 1)})
    assert engine.in_check(state, 0)
    for blocker in ("e6", "e4", "e2", "e5", "e3"):  # a leg or a landing on the way
        state = engine.make_state({"e1": ("K", 0), "h8": ("K", 1), "e7": ("R", 1),
                                   blocker: ("R", 0)})
        assert not engine.in_check(state, 0), blocker
    # The rider on the leg e2 is pinned: it may move only along the line it blocks.
    state = engine.make_state({"e1": ("K", 0), "h8": ("K", 1), "e7": ("R", 1),
                               "e2": ("R", 0)})
    assert targets(engine, state, "e2") == {"e4", "e6"}


def test_a_lame_rider_pawns_window_is_its_first_jumps() -> None:
    """En passant follows `_ep_passes` over the ray: a ride to the first landing opens a
    window on its leg, and a longer ride opens none (pinned here, `docs/archive/plan-schema-31.md` §2.4)."""
    pawn = {"id": "P", "name": "pawn", "value": 1, "moves": [
        {"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
        {"kind": "lame", "vectors": [[0, 2]], "range": 2, "mode": "move",
         "condition": {"first_move": True}}]}
    queen = {"id": "Q", "name": "queen", "value": 9, "moves": [
        {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]}
    spec = probe([pawn, queen], rules={
        "en_passant": True, "promotion": {"piece": "P", "rank": "last", "to": ["Q"]}},
        validate=False)
    spec["setup"]["placements"].append({"piece": "P", "player": 0, "squares": ["e2"]})
    assert validate_spec(spec) == []
    compiled = compile_spec(spec)
    e2, e3, e4 = 1 * 8 + 4, 2 * 8 + 4, 3 * 8 + 4
    assert compiled.ep_passes[0][e2] == ((e4, e3),)
    engine = engine_of(spec)
    state = engine.make_state({"a1": ("K", 0), "h8": ("K", 1), "e2": ("P", 0)})
    assert targets(engine, state, "e2") == {"e3", "e4", "e6"}


# ------------------------------------------------------------- the R1-shaped spec


def r1_shaped() -> dict[str, Any]:
    """*Opposite colours* with its elephants made lame `[2, 0]` riders and its guard given
    one beside its diagonal step (`docs/archive/plan-schema-31.md` §6.1's file, the tunable parts as the
    parent's). Not the showcase, which is Q1007e's: a test spec of its shape."""
    # The first opposite colours, retired by Q1007e when R1 took its name (`docs/archive/plan-schema-31.md` §6.1).
    spec: dict[str, Any] = json.loads((ROOT / "tests" / "_retired_opposite_colours.json")
                                      .read_text("utf-8"))
    spec.pop("provenance", None)
    spec["name"] = "opposite colours with riders, test shape"
    for piece in spec["pieces"]:
        if piece["id"] == "E":
            piece.update(id="R", name="rook", value=5, moves=[copy.deepcopy(ROOK_RIDER)])
        elif piece["id"] == "G":
            piece["moves"] = [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"},
                              copy.deepcopy(ROOK_RIDER)]
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "E":
            placement["piece"] = "R"
    spec["rules"]["promotion"]["to"] = ["B", "R", "C", "G"]
    return spec


def test_the_r1_shaped_spec_opens_with_nineteen_moves() -> None:
    spec = r1_shaped()
    assert validate_spec(spec) == []
    engine = engine_of(spec)
    state = engine.initial_state()
    # 14 pawn moves, 4 camel jumps, the king's e3; the riders and bishops are boxed in.
    assert engine.perft(state, 1) == 19
    assert [engine.perft(state, depth) for depth in (2, 3)] == [361, 7616]  # frozen pins


# ------------------------------------------------------------------- the oracle


def _rider_targets(spec: dict[str, Any], board: bytes, side: int,
                   n_types: int) -> set[tuple[int, int]]:
    """Every (origin, target) the side's riders may make, on an unwrapped board with no
    walls, regions or facings: a second statement of §2.2 (and of option 1's slide)."""
    width, height = spec["board"]["width"], spec["board"]["height"]
    found: set[tuple[int, int]] = set()

    def owner(square: int) -> int | None:
        value = board[square]
        return None if value == 0 else (value - 1) // n_types

    for origin, value in enumerate(board):
        if value == 0 or (value - 1) // n_types != side:
            continue
        piece = spec["pieces"][(value - 1) % n_types]
        for move in piece["moves"]:
            if move["kind"] not in ("lame", "slide") or piece["id"] == "P":
                continue
            jumps = move.get("range", 1 if move["kind"] == "lame" else 0) or 64
            least = move.get("min_range", 1)
            for dx, dy in _directions(move):
                if side == 1:
                    dy = -dy
                if move["kind"] == "lame":
                    if abs(dx) == abs(dy):
                        leg = (int(dx / 2), int(dy / 2))
                    elif abs(dx) > abs(dy):
                        leg = ((dx > 0) - (dx < 0), 0)
                    else:
                        leg = (0, (dy > 0) - (dy < 0))
                else:
                    leg = None
                f, r = origin % width, origin // width
                for jump in range(1, jumps + 1):
                    if leg is not None:
                        lf, lr = f + leg[0], r + leg[1]
                        if not (0 <= lf < width and 0 <= lr < height) or board[lr * width + lf]:
                            break
                    f, r = f + dx, r + dy
                    if not (0 <= f < width and 0 <= r < height):
                        break
                    square = r * width + f
                    held = owner(square)
                    if jump >= least and held != side:
                        found.add((origin, square))
                    if held is not None:
                        break
    return found


def _directions(move: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for x, y in move["vectors"]:
        symmetry = move.get("symmetry", "none")
        base = {(x, y)}
        if symmetry in ("rotate4", "dihedral8"):
            base = {(x, y), (-y, x), (-x, -y), (y, -x)}
        if symmetry == "dihedral8":
            base |= {(b, a) for a, b in base}
        out |= base
    return out


def walk_spec() -> dict[str, Any]:
    """Lame riders of three shapes, a slide nightrider beside them, and pawns to block."""
    pieces = [
        {"id": "P", "name": "pawn", "value": 1, "moves": [
            {"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]},
        rider("R", ROOK_RIDER),
        rider("N", NIGHT_RIDER),
        rider("A", {"kind": "lame", "vectors": [[2, 2]], "symmetry": "rotate4", "range": 3,
                    "min_range": 2}),
        rider("S", {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8"}),
    ]
    spec = probe(pieces, validate=False)
    spec["setup"] = {"mirror": True, "placements": [
        {"piece": "R", "squares": ["a1", "h1"]}, {"piece": "N", "squares": ["b1", "g1"]},
        {"piece": "A", "squares": ["c1", "f1"]}, {"piece": "S", "squares": ["d1"]},
        {"piece": "K", "squares": ["e1"]},
        {"piece": "P", "squares": ["a2", "c2", "d2", "f2", "h2"]}]}
    spec["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["R", "N", "A", "S"]}
    spec["rules"]["max_plies"] = 200
    assert validate_spec(spec) == []
    return spec


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_core_and_the_oracle_agree_over_a_seeded_walk(seed: int) -> None:
    spec = walk_spec()
    engine = engine_of(spec)
    n_types = len(spec["pieces"])
    chooser = random.Random(seed)
    state = engine.initial_state()
    rider_types = {i for i, p in enumerate(spec["pieces"]) if p["id"] in "RNAS"}
    checked = 0
    for _ in range(120):
        found = set()
        for move in engine.pseudo_legal_moves(state):
            origin, target, _, _ = decode_move(move)
            if (state.board[origin] - 1) % n_types in rider_types:
                found.add((origin, target))
        assert found == _rider_targets(spec, state.board, state.side_to_move, n_types)
        checked += 1
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, chooser.choice(legal))
    assert checked > 20


# ------------------------------------------------------------------- the words


def test_both_readings_get_a_riders_sentence() -> None:
    """§2.5: the slide over a jump says it is stopped only where it lands; the lame rider
    that a piece on any jump's leg stops it too. Neither says *anything in the way*."""
    slide = move_sentence({"kind": "slide", "vectors": [[2, 0]], "symmetry": "rotate4"})
    lame = move_sentence(ROOK_RIDER)
    assert slide == ("any number of jumps of two squares forward, back, left and right, "
                     "stopped only by a piece on a square it would land on")
    assert lame == ("any number of jumps of two squares forward, back, left and right, "
                    "stopped by a piece on a square it would land on or on the square any "
                    "jump passes")
    night = move_sentence({"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8"})
    assert "anything in the way" not in night and night.startswith("any number of jumps of")
    assert move_sentence({**NIGHT_RIDER, "range": 3, "min_range": 2}).startswith(
        "two to three jumps of two and one, in all eight directions, stopped by a piece")
    # The single lame leap keeps its words.
    assert move_sentence({"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"}) == (
        "leaps two and one, in all eight directions, stopped by a piece standing on the "
        "square it passes")


# ---------------------------------------------------------------- Fairy-Stockfish


def chess_with(knight_move: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((ROOT / "rulesets" / "chess.json").read_text("utf-8"))
    for piece in spec["pieces"]:
        if piece["id"] == "N":
            piece["moves"] = [copy.deepcopy(knight_move)]
    return spec


def test_fairy_refuses_a_lame_rider_rather_than_writing_a_lame_leaper() -> None:
    """Before Q1007a `_descriptor_term` passed distance 1 for every `lame`, so a lame rider
    would have crossed as the lame leaper `nN`, a different game, silently."""
    with pytest.raises(Untranslatable) as refused:
        translate(chess_with(NIGHT_RIDER))
    assert any("lame rider" in reason for reason in refused.value.reasons)
    # The single lame leap still crosses, as the built-in horse (`nN`).
    assert "horse = n" in translate(chess_with(
        {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"})).config


def test_fairy_still_writes_option_one() -> None:
    assert "NN" in translate(chess_with(
        {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8"})).config
    assert "DD" in translate(chess_with(
        {"kind": "slide", "vectors": [[2, 0]], "symmetry": "rotate4"})).config


# --------------------------------------------------------------------- distance


def test_a_lame_rider_is_not_at_distance_zero_from_its_lame_leap() -> None:
    from chess_variant_sandbox.spec.distance import distance, features

    leap = probe([rider("H", {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"})])
    ride = probe([rider("H", NIGHT_RIDER)])
    assert features(leap)["lame_rider_share"] == 0.0
    assert features(ride)["lame_rider_share"] == 0.5
    assert distance(leap, ride) > 0
