"""T1: riders whose path turns, `pieces[].moves[].path` (`docs/plan-pieces-moves.md` §5).

``"circle"`` (the rose on the knight's eight, the ring on the king's eight) and ``"zigzag"``
(the bolt, Betza's crooked bishop) on a ``slide``; ``"straight"`` is the off value. The
compiler builds each path as a slide entry of its own (`compile._turning_rides`), so the core
and the bridge are unchanged. Held here: the spelling and its seven refusals in plain words,
§5.1's hand counts and perfts by the real compiler, blocking, walls and a seam, check and pins
against `path_oracle`'s attack walk, the oracle over seeded random games, the rose's mate in
one, the words, the Fairy-Stockfish refusal and the distance.
"""

from __future__ import annotations

import copy
import json
import random
import re
from functools import cache
from pathlib import Path
from typing import Any

import path_oracle
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import move_sentence
from chess_variant_sandbox.spec.plain import plain_problem
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

ROSE = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "circle"}
RING = {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "path": "circle"}
BOLT = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "path": "zigzag"}
RIDERS = {"rose": ROSE, "ring": RING, "bolt": BOLT}
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


def probe(move: dict[str, Any], at: str, extra: list[dict[str, Any]] = (),  # type: ignore[assignment]
          rules: dict[str, Any] | None = None, width: int = 8, height: int = 8,
          validate: bool = True) -> dict[str, Any]:
    """§5.1's probe: the rider X on ``at``, kings on a1 and the far corner, blockers P."""
    spec = {
        "name": "turning path probe",
        "board": {"width": width, "height": height},
        "pieces": [
            {"id": "X", "name": "rider", "value": 4, "moves": [copy.deepcopy(move)]},
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
            {"id": "P", "name": "blocker", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "X", "player": 0, "squares": [at]},
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": [f"{chr(ord('a') + width - 1)}{height}"]},
            *extra,
        ]},
        "rules": {"win": "checkmate", "max_plies": 200, **(rules or {})},
    }
    if validate:
        assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def blocker(square: str, player: int) -> dict[str, Any]:
    return {"piece": "P", "player": player, "squares": [square]}


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
        if matched and matched.group(1) == origin:
            found.add(matched.group(2))
    return found


def _name(square: int, width: int = 8) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


def _problems(spec: dict[str, Any]) -> list[str]:
    return validate_spec(spec)


# ------------------------------------------------------------------- the spelling


def test_both_values_are_accepted_on_a_slide() -> None:
    for move in RIDERS.values():
        probe(move, "d4")


def test_straight_is_the_slide_it_always_was() -> None:
    """The off value: the same name, the same tables, and the converter drops it."""
    plain = probe({"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}, "d4")
    written = probe({"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4",
                     "path": "straight"}, "d4")
    assert spec_hash(written) == spec_hash(plain)
    assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(plain))
    assert current_spelling(written) == current_spelling(plain)


def test_straight_on_a_portals_move_is_dropped_too() -> None:
    base = load_spec(ROOT / "archive" / "wormhole.json")
    written = copy.deepcopy(base)
    for group in written["rules"]["portals"]:
        for move in group.get("moves", []):
            move["path"] = "straight"
    assert written != base and validate_spec(written) == []
    assert spec_hash(written) == spec_hash(base)


def test_a_turning_path_is_a_new_game() -> None:
    plain = probe({"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}, "d4")
    assert spec_hash(probe(BOLT, "d4")) != spec_hash(plain)


def test_an_unknown_value_is_refused() -> None:
    spec = probe({**BOLT, "path": "spiral"}, "d4", validate=False)
    assert any("path must be one of" in line for line in _problems(spec))


# ------------------------------------------------------------------- the refusals
#
# §5.3, each on its scratch shape, each said in plain words.


def _refused_with(spec: dict[str, Any], phrase: str) -> str:
    lines = [line for line in _problems(spec) if phrase in line]
    assert lines, (phrase, _problems(spec))
    said = plain_problem(lines[0], spec)
    assert said[0].isupper() and said.endswith("."), said
    assert "_" not in said.replace("'", ""), said
    return said


@pytest.mark.parametrize(("kind", "reason"), [
    ("leap", "a leap is one jump"),
    ("lame", "a lame move's leg is fixed"),
    ("hop", "a hop's screen is the first piece on a line"),
    ("sweep", "a sweep removes what lies on a line"),
])
def test_refusal_1_a_path_on_another_kind(kind: str, reason: str) -> None:
    move = {"kind": kind, "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "circle"}
    _refused_with(probe(move, "d4", validate=False), reason)


@pytest.mark.parametrize(("move", "gap"), [
    ({"kind": "slide", "vectors": [[1, 2]], "path": "circle"}, 360),
    ({"kind": "slide", "vectors": [[1, 1]], "symmetry": "lateral", "path": "circle"}, 270),
    ({"kind": "slide", "vectors": [[1, 0], [0, 1], [-1, 1]], "path": "circle"}, 225),
    ({"kind": "slide", "vectors": [[1, 0]], "symmetry": "lateral", "path": "circle"}, 180),
])
def test_refusal_2_a_circle_with_a_gap(move: dict[str, Any], gap: int) -> None:
    _refused_with(probe(move, "d4", validate=False), f"leave a gap of {gap} degrees")


@pytest.mark.parametrize("move", [
    {"kind": "slide", "vectors": [[0, 1]], "path": "zigzag"},
    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "lateral", "path": "zigzag"},
])
def test_refusal_2_a_zigzag_with_nothing_to_turn_to(move: dict[str, Any]) -> None:
    _refused_with(probe(move, "d4", validate=False), "and these have none")


def test_refusal_2_two_directions_one_way() -> None:
    move = {"kind": "slide", "vectors": [[1, 0], [2, 0]], "symmetry": "rotate4",
            "path": "circle"}
    _refused_with(probe(move, "d4", validate=False), "point the same way")


def test_a_set_that_goes_round_with_no_symmetry_is_a_circle() -> None:
    """Three directions, no gap of half a turn: a circle needs no symmetry."""
    probe({"kind": "slide", "vectors": [[1, 0], [-1, 1], [-1, -1]], "path": "circle"}, "d4")
    # And a zigzag needs only one pair less than half a turn apart.
    probe({"kind": "slide", "vectors": [[1, 1]], "symmetry": "lateral", "path": "zigzag"}, "d4")


def test_refusal_3_one_step_never_turns() -> None:
    _refused_with(probe({**RING, "range": 1}, "d4", validate=False), "which never turns")


def test_refusal_4_a_hex_lattice() -> None:
    spec = load_spec(ROOT / "archive" / "hex-glinski-hexagon.json")
    slider = next(piece for piece in spec["pieces"]
                  if any(move["kind"] == "slide" for move in piece["moves"]))
    for move in slider["moves"]:
        if move["kind"] == "slide":
            move["path"] = "circle"
    _refused_with(spec, "is for the square lattice")


def test_refusal_5_beside_portals_and_on_a_portals_moves() -> None:
    spec = load_spec(ROOT / "archive" / "wormhole.json")
    queen = next(piece for piece in spec["pieces"] if piece["id"] == "Q")
    queen["moves"][0]["path"] = "circle"
    _refused_with(spec, "beside rules.portals")
    moved = load_spec(ROOT / "archive" / "wormhole.json")
    moved["rules"]["portals"][0]["moves"] = [
        {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "path": "circle"}]
    _refused_with(moved, "a portal moves straight")


def test_refusal_6_the_piece_en_passant_is_about() -> None:
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    pawn = next(piece for piece in spec["pieces"] if piece["id"] == "P")
    pawn["moves"].append({"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4",
                          "path": "zigzag", "mode": "capture"})
    _refused_with(spec, "the piece rules.en_passant is about")
    spec["rules"].pop("en_passant")
    assert validate_spec(spec) == []


def test_refusal_7_a_wild_piece() -> None:
    spec = load_spec(ROOT / "archive" / "wolves.json")
    wolf = next(piece for piece in spec["pieces"] if piece["id"] == "W")
    wolf["moves"] = [copy.deepcopy(RING)]
    _refused_with(spec, "a wild piece walks its route")


def test_a_sight_line_has_no_path() -> None:
    """A line of sight is outside the move key set, so a turning one is an unknown key."""
    spec = load_spec(TESTS / "_sight_sheltering_king.json")
    entry = next(entry for piece in spec["pieces"] for entry in piece.get("sight", []))
    entry["path"] = "circle"
    assert any("unknown keys: path" in line for line in _problems(spec))


# ------------------------------------------------------------- §5.1's hand counts


@pytest.mark.parametrize(("name", "at", "crowded", "count"), [
    ("rose", "d4", False, 13), ("rose", "d4", True, 13),
    ("ring", "d4", False, 32), ("ring", "d4", True, 31),
    ("bolt", "d4", False, 18), ("bolt", "d4", True, 16),
    ("rose", "e4", False, 14),
])
def test_the_hand_counts(name: str, at: str, crowded: bool, count: int) -> None:
    """§5.1, kings on a1 and h8; the crowd is enemy blockers on e6 and c5, an own one on f2."""
    extra = [blocker("e6", 1), blocker("f2", 0), blocker("c5", 1)] if crowded else []
    spec = probe(RIDERS[name], at, extra)
    engine = engine_of(spec)
    found = targets(engine, engine.initial_state(), at)
    assert len(found) == count
    # The oracle's walk says the same squares.
    state = engine.initial_state()
    origin = (int(at[1:]) - 1) * 8 + ord(at[0]) - ord("a")
    said = {_name(t) for o, t in path_oracle.path_targets(spec, state.board, 0) if o == origin}
    assert found == said


@pytest.mark.parametrize(("name", "perfts"), [
    ("rose", [16, 39, 639]), ("ring", [35, 87, 2405]), ("bolt", [21, 55, 1132]),
])
def test_the_perfts(name: str, perfts: list[int]) -> None:
    """§5.1's stand-in perfts, reproduced by the real compiler (frozen pins)."""
    engine = engine_of(probe(RIDERS[name], "d4"))
    state = engine.initial_state()
    assert [engine.perft(state, depth) for depth in (1, 2, 3)] == perfts


def test_the_test_specs_open_as_the_drafts_did() -> None:
    """§5.7's draft log: chess with roses opens with 30 moves, the ring queen and the bolts
    with chess's 20 (both boxed in)."""
    for name, opening in (("rose", 30), ("ring", 20), ("bolt", 20)):
        engine = Engine(load_spec(TESTS / f"_path_{name}.json"))
        assert len(engine.legal_moves(engine.initial_state())) == opening, name


# ---------------------------------------------------------------- blocking, walls, a seam


def test_a_bolt_blocked_on_its_first_square_loses_those_paths_and_keeps_the_others() -> None:
    """V1's hand position: an own pawn on e5 ends both bolt paths that start north-east (e5 f4
    g5 h4 and e5 d6 e7 d8). The six others are untouched, and still reach f4, h4, d6 and d8
    by their own way (e3 f4 g3 h4, c5 d6 c7 d8): a square reached two ways is one move."""
    open_ = engine_of(probe(BOLT, "d4"))
    shut = engine_of(probe(BOLT, "d4", [blocker("e5", 0)]))
    before = targets(open_, open_.initial_state(), "d4")
    after = targets(shut, shut.initial_state(), "d4")
    lost = {"e5", "g5", "e7"}
    assert lost <= before and not (lost & after)
    assert after == before - lost
    assert {"f4", "h4", "d6", "d8"} <= after


def test_a_capture_ends_the_path_and_an_own_piece_blocks() -> None:
    engine = engine_of(probe(BOLT, "d4", [blocker("e5", 1)]))
    found = targets(engine, engine.initial_state(), "d4")
    assert "e5" in found and not {"g5", "e7"} & found


def test_the_rose_passes_between_pieces() -> None:
    """A square a knight step jumps over is never looked at: pieces on d5 and e5 change
    nothing for a rose on d4."""
    open_ = engine_of(probe(ROSE, "d4"))
    crowded = engine_of(probe(ROSE, "d4", [blocker("d5", 0), blocker("e5", 0)]))
    assert targets(crowded, crowded.initial_state(), "d4") == targets(
        open_, open_.initial_state(), "d4")


def test_a_wall_ends_a_path() -> None:
    """A walled bolt: a wall on e5 ends the two north-east paths, and is never taken."""
    engine = engine_of(probe(BOLT, "d4", rules={"obstacles": {"squares": ["e5"]}}))
    found = targets(engine, engine.initial_state(), "d4")
    assert not {"e5", "g5", "e7"} & found and {"f4", "d6", "c5", "e3", "c3"} <= found


def test_a_wall_ends_a_path_against_a_walk_with_the_wall_as_a_piece() -> None:
    """The same squares as an own piece on the wall's square, bar the square itself."""
    walled = engine_of(probe(BOLT, "d4", rules={"obstacles": {"squares": ["e5"]}}))
    pieced = engine_of(probe(BOLT, "d4", [blocker("e5", 0)]))
    assert targets(walled, walled.initial_state(), "d4") == targets(
        pieced, pieced.initial_state(), "d4")


def test_a_rose_crosses_a_seam_and_stops_where_it_has_been() -> None:
    """On a cylinder (the files joined) a rose on a4 steps over the a-h seam; and on the
    cylinder a circle stops before a square it has already been on, so the table is finite and
    every landing is visited once per path."""
    spec = probe(ROSE, "a4", rules={"wrap": {"files": "both"}})
    engine = engine_of(spec)
    found = targets(engine, engine.initial_state(), "a4")
    assert "h6" in found or "h2" in found, found  # a (-1, +-2) first step wraps to the h-file
    compiled = compile_spec(spec)
    for entry in compiled.slides[0][0][24]:
        assert len(set(entry[0])) == len(entry[0]), entry
        assert 24 not in entry[0], entry


# ------------------------------------------------------------- check, pins and the oracle


def _legal_by_oracle(spec: dict[str, Any], engine: Engine, state: GameState) -> set[int]:
    """The core's pseudo-legal moves that leave the mover's royal unattacked, attack judged
    by the oracle's own walk."""
    me = state.side_to_move
    keep = set()
    for move in engine.pseudo_legal_moves(state):
        after = engine.apply(state, move)
        royal = path_oracle.royal_square(spec, after.board, me)
        if royal is None or not path_oracle.attacked(spec, after.board, royal, 1 - me):
            keep.add(move)
    return keep


_SETUPS = {
    "open": [{"piece": "X", "player": 1, "squares": ["e5"]}],
    "pin": [{"piece": "X", "player": 1, "squares": ["c6"]}, blocker("b3", 0), blocker("c2", 0),
            blocker("f6", 1), blocker("g6", 1)],
    "crowd": [{"piece": "X", "player": 1, "squares": ["f6"]}, blocker("c3", 0),
              blocker("e3", 0), blocker("d6", 1), blocker("g5", 1)],
}


def _check_tree(spec: dict[str, Any], depth: int) -> tuple[int, int]:
    engine = engine_of(spec)
    nodes = filtered = 0

    def walk(state: GameState, left: int) -> None:
        nonlocal nodes, filtered
        legal = set(engine.legal_moves(state))
        assert legal == _legal_by_oracle(spec, engine, state)
        nodes += 1
        filtered += len(legal) < len(set(engine.pseudo_legal_moves(state)))
        if left:
            for move in sorted(legal):
                walk(engine.apply(state, move), left - 1)

    walk(engine.initial_state(), depth)
    return nodes, filtered


@pytest.mark.parametrize("setup", sorted(_SETUPS))
@pytest.mark.parametrize("name", sorted(RIDERS))
def test_check_and_pins_agree_with_the_oracle_one_ply_deep(name: str, setup: str) -> None:
    nodes, _ = _check_tree(probe(RIDERS[name], "d4", _SETUPS[setup]), 1)
    assert nodes > 1


@pytest.mark.slow
@pytest.mark.parametrize("setup", sorted(_SETUPS))
@pytest.mark.parametrize("name", sorted(RIDERS))
def test_check_and_pins_agree_with_the_oracle_two_plies_deep(name: str, setup: str) -> None:
    """§5.1's check-and-pin walk: every node of a depth-2 tree."""
    nodes, filtered = _check_tree(probe(RIDERS[name], "d4", _SETUPS[setup]), 2)
    assert nodes > 30 and filtered > 0


def _walk(name: str, seed: int, plies: int = 120) -> int:
    """A seeded random game of the path test spec: after every move, the core's pseudo-legal
    moves of the turning piece equal the oracle's, and its legal list equals the pseudo-legal
    one filtered by the oracle's attack walk."""
    spec = load_spec(TESTS / f"_path_{name}.json")
    engine = Engine(spec)
    n_types = len(spec["pieces"])
    turning = {i for i, piece in enumerate(spec["pieces"])
               if any(move.get("path") for move in piece["moves"])}
    chooser = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(plies):
        found = {
            (origin, target)
            for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
            if (state.board[origin] - 1) % n_types in turning
        }
        assert found == path_oracle.path_targets(spec, state.board, state.side_to_move)
        legal = engine.legal_moves(state)
        assert set(legal) == _legal_by_oracle(spec, engine, state)
        checked += 1
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, chooser.choice(legal))
    return checked


@pytest.mark.parametrize("name", sorted(RIDERS))
def test_the_core_and_the_oracle_agree_over_a_seeded_game(name: str) -> None:
    """Three in the fast suite, one per piece (T1's acceptance)."""
    assert _walk(name, 0) >= 2


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(1, 21))
@pytest.mark.parametrize("name", sorted(RIDERS))
def test_the_core_and_the_oracle_agree_over_twenty_seeded_games(name: str, seed: int) -> None:
    assert _walk(name, seed) >= 2


def test_the_roses_first_move_mates() -> None:
    """§5.7's draft read: in chess with roses, b1-e4 (by way of c3) checks e8 along two
    octagons at once, by d6 and by f6. One block answers one, and the king cannot step, so it
    is mate on ply one."""
    engine = Engine(load_spec(TESTS / "_path_rose.json"))
    state = engine.initial_state()
    move = next(m for m in engine.legal_moves(state)
                if _MOVE.match(engine.describe_move(state, m)).groups() == ("b1", "e4"))
    after = engine.apply(state, move)
    assert engine.in_check(after)
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0, outcome


# ------------------------------------------------------------------- the words


def test_each_value_has_its_sentence() -> None:
    rose = move_sentence(ROSE)
    assert rose.startswith("any number of steps of two and one, in all eight directions, "
                           "each step turning to the next direction round"), rose
    assert "jumping over the squares between" in rose
    ring = move_sentence(RING)
    assert ring.startswith("any number of one-square steps in all eight directions"), ring
    bolt = move_sentence({**BOLT, "range": 3})
    assert bolt.startswith("up to three one-square steps along all four diagonals, turning "
                           "to a neighbouring direction and back again"), bolt
    for said in (rose, ring, bolt):
        assert "stopped by a piece on a square it lands on" in said


# ---------------------------------------------------------------- Fairy-Stockfish


def test_fairy_refuses_a_turning_path_by_name() -> None:
    """§5.4: pyffish reads XBetza's `q` and `z` as the plain piece, so the translation would
    be the knight or the bishop. Refused before any Betza is written."""
    for name in RIDERS:
        with pytest.raises(Untranslatable) as refused:
            translate(load_spec(TESTS / f"_path_{name}.json"))
        assert any("circular and crooked modifiers" in reason
                   for reason in refused.value.reasons), refused.value.reasons


def test_fairy_reads_straight_as_nothing() -> None:
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    for piece in spec["pieces"]:
        for move in piece["moves"]:
            move["path"] = "straight"
    assert translate(spec).config == translate(load_spec(ROOT / "rulesets" / "chess.json")).config


# --------------------------------------------------------------------- distance, cost


def test_a_turning_queen_is_not_at_distance_zero_from_the_queen() -> None:
    from chess_variant_sandbox.spec.distance import distance, features

    chess = load_spec(ROOT / "rulesets" / "chess.json")
    ring = load_spec(TESTS / "_path_ring.json")
    assert features(chess)["turning_share"] == 0.0
    assert features(ring)["turning_share"] > 0.0
    assert distance(chess, ring) > 0.0


def test_a_turning_path_costs_a_rule_once_per_kind() -> None:
    chess = measure_complexity(load_spec(ROOT / "rulesets" / "chess.json"))
    ring = measure_complexity(load_spec(TESTS / "_path_ring.json"))
    assert ring.special_rules == chess.special_rules + 1.0
