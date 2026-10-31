"""The push rule's keys and their compiled tables (Q1074b1, `docs/plan-push.md` §1 to §3).

Every key validates and refuses as the plan says, with a case either side of each refusal
(round 2's refusals included until they are lifted); the weight shorthand normalises to the
lists it stands for and hashes alike; the compiled push tables agree with
`tests/push_oracle.py`'s reading for a slider, a knight (along the move and along the rank), a
hex step and a wrapped edge; a push landing is an attack source; every stored spec compiles no
push; the bridge admits the single push (Q1074b2) and Fairy-Stockfish refuses the keys by
name. Q1074b2 folded the keys into `jsonschema.py` and the proposer's preamble.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import push_oracle
import pytest

from chess_variant_sandbox.engine.bridge import canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import (
    ATTACK_BY_CAPTURE,
    ATTACK_BY_PUSH,
    MODE_PUSH,
    PUSH_NO_DIRECTION,
    CompiledSpec,
    compile_spec,
)
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.refusals import PUSH_REFUSALS, refused_ids
from chess_variant_sandbox.spec.schema import SpecError, pushing_ids
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return loaded


def _leap(vectors: list[list[int]], symmetry: str = "dihedral8", **more: Any) -> dict[str, Any]:
    return {"kind": "leap", "vectors": vectors, "symmetry": symmetry, **more}


def _slide(vectors: list[list[int]], symmetry: str = "rotate4", **more: Any) -> dict[str, Any]:
    return {"kind": "slide", "vectors": vectors, "symmetry": symmetry, **more}


def _game(**rules: Any) -> dict[str, Any]:
    """A small push game: a king that does not push, a rook and a knight that do."""
    return {
        "name": "push test",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [_leap([[1, 0], [1, 1]])]},
            {"id": "R", "value": 5, "moves": [_slide([[1, 0]], push=True)]},
            {"id": "N", "value": 3, "moves": [_leap([[1, 2]], push=True)]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["b1"]},
            {"piece": "N", "squares": ["c1"]},
        ]},
        "rules": dict(rules),
    }


def _piece(spec: dict[str, Any], pid: str) -> dict[str, Any]:
    found: dict[str, Any] = next(p for p in spec["pieces"] if p["id"] == pid)
    return found


def _errors_with(spec: dict[str, Any], text: str) -> list[str]:
    return [error for error in validate_spec(spec) if text in error]


# ------------------------------------------------------------------------- the keys


def test_the_small_push_game_is_valid() -> None:
    assert validate_spec(_game()) == []


@pytest.mark.parametrize("push", [
    True, {}, {"whose": "own"}, {"whose": "any"}, {"captures": True},
    {"along": "move"}, {"whose": "any", "captures": True, "along": "move"},
])
def test_a_push_entry_accepts_its_values(push: Any) -> None:
    spec = _game()
    _piece(spec, "R")["moves"][0]["push"] = push
    assert validate_spec(spec) == []


@pytest.mark.parametrize(("push", "text"), [
    ("yes", "must be true, false or an object"),
    ({"whose": "nobody"}, "push.whose must be one of"),
    ({"captures": 1}, "push.captures must be a boolean"),
    ({"along": "diagonal"}, "push.along must be one of"),
    ({"pull": True}, "push has unknown keys: pull"),
])
def test_a_push_entry_refuses_a_malformed_value(push: Any, text: str) -> None:
    spec = _game()
    _piece(spec, "R")["moves"][0]["push"] = push
    assert _errors_with(spec, text)


def test_push_false_is_the_off_value() -> None:
    """`false` is the move without a push: accepted, and the game without it."""
    spec = _game()
    _piece(spec, "N")["moves"][0]["push"] = False
    without = _game()
    del _piece(without, "N")["moves"][0]["push"]
    assert validate_spec(spec) == []
    assert spec_hash(spec) == spec_hash(without)
    assert current_spelling(spec) == current_spelling(without)


def test_push_true_and_its_defaults_written_out_are_one_game() -> None:
    written = _game()
    _piece(written, "R")["moves"][0]["push"] = {
        "whose": "enemy", "captures": False, "along": "move",
    }
    empty = _game()
    _piece(empty, "R")["moves"][0]["push"] = {}
    assert spec_hash(written) == spec_hash(empty) == spec_hash(_game())
    assert normal_form(written) == normal_form(_game())


def test_captures_true_is_refused_where_there_is_no_capture() -> None:
    """§1.3: on a ``mode: "move"`` entry, and (ruled here) on a push of one's own pieces."""
    spec = _game()
    move = _piece(spec, "R")["moves"][0]
    move["push"] = {"captures": True}
    assert validate_spec(spec) == []
    move["mode"] = "move"
    assert _errors_with(spec, "never captures (mode 'move')")
    move["mode"] = "both"
    move["push"] = {"captures": True, "whose": "own"}
    assert _errors_with(spec, "one's own pieces only")


def test_a_sweep_that_pushes_is_refused_d6() -> None:
    spec = _game()
    _piece(spec, "R")["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    assert validate_spec(spec) == []
    _piece(spec, "R")["moves"][0]["push"] = True
    assert _errors_with(spec, "on a sweep (D6)")


def test_a_turning_path_that_pushes_is_refused_d7() -> None:
    spec = _game()
    _piece(spec, "R")["moves"] = [
        {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 3,
         "path": "circle"},
    ]
    assert validate_spec(spec) == []
    _piece(spec, "R")["moves"][0]["push"] = True
    assert _errors_with(spec, "on a turning path (D7)")


def test_along_a_rank_needs_a_rank_component_d3() -> None:
    spec = _game()
    _piece(spec, "N")["moves"][0]["push"] = {"along": "rank"}
    _piece(spec, "N")["moves"][0]["push"] = {"along": "file"}
    assert validate_spec(spec) == []
    _piece(spec, "R")["moves"][0]["push"] = {"along": "rank"}
    assert _errors_with(spec, "has no rank component")


def test_along_a_rank_is_refused_on_a_hex_lattice_h1() -> None:
    spec = _hex_game()
    assert validate_spec(spec) == []
    _piece(spec, "S")["moves"][0]["push"] = {"along": "rank"}
    assert _errors_with(spec, "on a hex lattice (H1)")


def test_rules_push_values() -> None:
    for push in ({"chain": "row"}, {"chain": "one", "edge": "blocks"}, {"walls": "block"},
                 {"no_push_back": False}):
        assert validate_spec(_game(push=push)) == [], push
    assert _errors_with(_game(push={"chain": "pile"}), "rules.push.chain must be one of")
    assert _errors_with(_game(push={"edge": "wraps"}), "rules.push.edge must be one of")
    assert _errors_with(_game(push={"slope": 1}), "rules.push has unknown keys")
    assert _errors_with(_game(push=[]), "rules.push must be an object")


def test_round_two_values_are_lifted() -> None:
    """W1's crush was lifted by Q1074b7 (`tests/test_push_crush.py`), refused only on a
    board with no wall; K5's no push back by Q1074b6 (`tests/test_push_back.py`)."""
    assert _errors_with(_game(push={"walls": "crush"}), "walls 'crush' (W1) on a board with no wall")
    assert validate_spec(_game(push={"walls": "block"})) == []
    assert validate_spec(_game(push={"no_push_back": True})) == []
    assert validate_spec(_game(push={"no_push_back": False})) == []


def test_rules_push_at_its_defaults_is_its_absence() -> None:
    off = _game(push={"chain": "one", "edge": "removes", "walls": "block", "no_push_back": False})
    assert validate_spec(off) == []
    assert spec_hash(off) == spec_hash(_game())
    chess = _load("rulesets/chess.json")
    written = copy.deepcopy(chess)
    written["rules"]["push"] = {"chain": "one"}
    assert validate_spec(written) == []
    assert spec_hash(written) == spec_hash(chess)
    assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(chess))
    assert current_spelling(written) == current_spelling(chess)


def test_push_keys_where_nothing_pushes_are_refused() -> None:
    """Each can never fire (§1.2); the same keys beside a push are accepted."""
    chess = _load("rulesets/chess.json")
    for edit, text in (
        (lambda s: s["rules"].update(push={"chain": "row"}), "no move pushes"),
        (lambda s: s["pieces"][1].update(pushes=["P"]), "no move in the game pushes"),
        (lambda s: s["pieces"][1].update(weight=2), "no move in the game pushes"),
    ):
        spec = copy.deepcopy(chess)
        edit(spec)
        assert [e for e in validate_spec(spec) if text in e], text
    assert validate_spec(_game(push={"chain": "row"})) == []


def test_a_pushes_list() -> None:
    spec = _game()
    _piece(spec, "R")["pushes"] = ["N", "K"]
    assert validate_spec(spec) == []
    _piece(spec, "R")["pushes"] = []
    assert _errors_with(spec, "its pushes list is empty")
    _piece(spec, "R")["pushes"] = ["Q"]
    assert _errors_with(spec, "pushes entry 'Q' is not a declared piece")
    spec = _game()
    _piece(spec, "K")["pushes"] = ["N"]
    assert _errors_with(spec, "none of its moves pushes")


@pytest.mark.parametrize("weight", [0, 10, -1, 1.5, True, "2"])
def test_a_weight_is_one_to_nine(weight: Any) -> None:
    spec = _game()
    _piece(spec, "R")["weight"] = weight
    assert _errors_with(spec, "weight must be a whole number from 1 to 9")


def test_the_weight_shorthand_under_one_and_row() -> None:
    spec = _game()
    _piece(spec, "R")["weight"] = 3
    _piece(spec, "N")["weight"] = 2
    assert validate_spec(spec) == []
    both = copy.deepcopy(spec)
    _piece(both, "R")["pushes"] = ["N"]
    assert _errors_with(both, "both a weight and a pushes list")
    unread = _game()
    _piece(unread, "K")["weight"] = 1
    _piece(unread, "R")["pushes"] = ["N"]
    assert _errors_with(unread, "only a pusher with a weight and no pushes list reads them")


def test_weights_under_chain_weight() -> None:
    spec = _game(push={"chain": "weight"})
    assert _errors_with(spec, "pushes under chain 'weight' and states neither a strength nor")
    for pid, weight in (("K", 1), ("R", 3), ("N", 2)):
        _piece(spec, pid)["weight"] = weight
    assert validate_spec(spec) == []
    _piece(spec, "R")["pushes"] = ["N"]  # both tests apply; a list beside a weight is fine
    _piece(spec, "N")["pushes"] = ["N"]
    assert _errors_with(spec, "piece 'K' states a weight and under chain 'weight' it neither")
    del _piece(spec, "K")["weight"]
    assert validate_spec(spec) == []
    del _piece(spec, "N")["weight"]
    assert _errors_with(spec, "piece 'N' may be pushed under chain 'weight' and states no weight")


def test_a_quarter_turning_piece_may_not_push_along_a_rank() -> None:
    spec = _game()
    _piece(spec, "N")["moves"] = [_slide([[1, 0]], "none", push={"along": "rank"})]
    assert validate_spec(spec) == []
    _piece(spec, "N")["facing"] = {"axes": 4}
    assert _errors_with(spec, "turns a quarter")
    # A knight's every turn keeps a rank component.
    _piece(spec, "N")["moves"] = [_leap([[1, 2]], "none", push={"along": "rank"})]
    assert not _errors_with(spec, "turns a quarter")


# --------------------------------------------------------------- the refused pairs


def _pushing(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with every capturing entry of every non-royal piece pushing."""
    out = copy.deepcopy(spec)
    for piece in out["pieces"]:
        if piece.get("royal"):
            continue
        for move in piece.get("moves", []):
            if move.get("mode", "both") != "move" and move.get("kind") != "sweep" and (
                move.get("path", "straight") == "straight"
            ):
                move["push"] = True
    return out


PAIRS = [
    ("push.W4", "archive/wormhole.json", None),
    ("push.X2", "archive/circe.json", None),
    ("push.X2", "archive/einstein.json", None),
    ("push.X3", "rulesets/atomic.json", None),
    ("push.S1", "tests/_simultaneous_gardner.json", None),
    ("push.E2", "rulesets/chess.json", None),
    ("push.N4", "archive/chess-with-preaching-bishops.json", None),
]

# Q1074b8 (`tests/test_push_round_two.py`): round 1's pairs that round 2 lifted, whole or in
# part; each case is the refused pair's own, now admitted.
LIFTED = [
    ("push.N4", "archive/madrasi.json"),
    ("push.N5", "tests/_rest_checkmate.json"),
    ("push.G1", "rulesets/kingofthehill.json"),
    ("push.G2", "rulesets/rps-v2.json"),
    ("push.M1", "tests/_purse_market.json"),
]


def test_every_refused_pair_has_a_case() -> None:
    assert {row.id for row in PUSH_REFUSALS} == {rid for rid, _, _ in PAIRS}


@pytest.mark.parametrize(("rid", "path"), LIFTED, ids=[f"{r}-{p}" for r, p in LIFTED])
def test_a_lifted_pair_is_admitted(rid: str, path: str) -> None:
    pushing = _pushing(_load(path))
    pushing["rules"].pop("en_passant", None)  # E2's, not this pair's
    assert validate_spec(pushing) == [], validate_spec(pushing)
    assert rust_supports(compile_spec(pushing)) is None


@pytest.mark.parametrize(("rid", "path", "rule"), PAIRS, ids=[f"{r}-{p}" for r, p, _ in PAIRS])
def test_a_refused_pair_either_side(rid: str, path: str, rule: Any) -> None:
    base = _load(path)
    assert validate_spec(base) == [], validate_spec(base)
    pushing = _pushing(base)
    if rule is not None:
        pushing["rules"]["push"] = rule
    assert rid in refused_ids(validate_spec(pushing)), validate_spec(pushing)


def test_rows_on_a_wrapped_board_are_allowed() -> None:
    """H3 was round 1's; Q1074b7 read rows on a seam (H4), so neither is refused."""
    torus = _pushing(_load("rulesets/torus.json"))
    del torus["rules"]["en_passant"]  # E2's, not a seam's
    assert validate_spec(torus) == []
    torus["rules"]["push"] = {"chain": "row"}
    assert validate_spec(torus) == []


def test_en_passant_beside_a_pawn_that_only_pushes_ahead_is_allowed() -> None:
    """E2 reads the en passant piece's capturing entries; its straight push is not one."""
    chess = _load("rulesets/chess.json")
    pawn = next(p for p in chess["pieces"] if p["id"] == "P")
    for move in pawn["moves"]:
        if move.get("mode") == "move":
            move["push"] = True
    assert "push.E2" not in refused_ids(validate_spec(chess))


def test_promotion_on_push_is_no_key_p2() -> None:
    chess = _load("rulesets/chess.json")
    chess["rules"]["promotion"]["on_push"] = True
    assert _errors_with(chess, "on_push")


def test_wild_pieces_and_the_duck_never_push_and_are_never_pushed() -> None:
    plain = _load("tests/_duck_chess.json")
    plain["rules"].pop("en_passant", None)  # E2 is its own case
    duck = _pushing(plain)
    assert validate_spec(duck) == [], validate_spec(duck)
    duck_id = duck["rules"]["duck"]["piece"]
    named = copy.deepcopy(duck)
    rook = next(p for p in named["pieces"] if p["id"] == "R")
    rook["pushes"] = ["P", duck_id]
    assert [e for e in validate_spec(named) if "which is nobody's" in e]
    pushing_duck = copy.deepcopy(duck)
    next(p for p in pushing_duck["pieces"] if p["id"] == duck_id)["moves"] = [
        _leap([[1, 0]], "rotate4", push=True)
    ]
    assert [e for e in validate_spec(pushing_duck) if "pushes and is nobody's" in e]
    compiled = compile_spec(duck)
    assert compiled.push is not None
    column = compiled.piece_ids.index(duck_id)
    assert not any(row[column] for row in compiled.push.allowed)


# ------------------------------------------------------------------ the normal form


def test_the_weight_shorthand_normalises_to_the_lists_and_hashes_alike() -> None:
    """The rook's weight 3 reads every type of weight 3 or less; the knight pushes by its own
    list, so it carries no weight of its own and is pushed by no weight."""
    weighed = _game(push={"chain": "row"})
    for pid, weight in (("K", 1), ("R", 3)):
        _piece(weighed, pid)["weight"] = weight
    _piece(weighed, "N")["pushes"] = ["K", "N"]
    listed = _game(push={"chain": "row"})
    _piece(listed, "R")["pushes"] = ["R", "K"]
    _piece(listed, "N")["pushes"] = ["N", "K"]
    assert validate_spec(weighed) == [] and validate_spec(listed) == []
    assert normal_form(weighed) == normal_form(listed)
    assert _piece(normal_form(weighed), "R")["pushes"] == ["K", "R"]
    assert spec_hash(weighed) == spec_hash(listed)
    assert compile_spec(weighed).push == compile_spec(listed).push


def test_weights_under_chain_weight_stay_in_the_name() -> None:
    light = _game(push={"chain": "weight"})
    for pid, weight in (("K", 1), ("R", 3), ("N", 2)):
        _piece(light, pid)["weight"] = weight
    heavy = copy.deepcopy(light)
    _piece(heavy, "N")["weight"] = 3
    assert spec_hash(light) != spec_hash(heavy)
    assert "weight" in _piece(normal_form(light), "R")


# --------------------------------------------------------------- the compiled tables


def _compiled_landings(
    compiled: CompiledSpec, pid: str, player: int, origin: int
) -> set[tuple[int, int, tuple[int, int]]]:
    """The landings that push, read off the movement tables and the push directions."""
    assert compiled.push is not None
    kind = compiled.piece_ids.index(pid)
    frame = player * compiled.n_facings
    found: set[tuple[int, int, tuple[int, int]]] = set()
    push = compiled.push
    for table, dirs in (
        (compiled.leaps, push.leap_dirs), (compiled.slides, push.slide_dirs),
        (compiled.hops, push.hop_dirs),
    ):
        for entry, entry_dirs in zip(table[frame][kind][origin], dirs[frame][kind][origin],
                                     strict=True):
            for landing, mode, direction in zip(entry[0], entry[1], entry_dirs, strict=True):
                if not mode & MODE_PUSH:
                    assert direction == PUSH_NO_DIRECTION
                    continue
                owner, dx, dy = push.directions[direction]
                assert owner == player
                found.add((landing, mode, (dx, dy)))
    return found


def _agrees_with_the_oracle(spec: dict[str, Any], pid: str) -> None:
    compiled = compile_spec(spec)
    assert compiled.push is not None
    for player in range(2):
        for origin in range(compiled.n_squares):
            if origin in compiled.obstacles:
                continue
            expected = push_oracle.landings(spec, pid, player, origin)
            assert _compiled_landings(compiled, pid, player, origin) == expected, (
                pid, player, origin,
            )
    for number, (player, dx, dy) in enumerate(compiled.push.directions):
        for square in range(compiled.n_squares):
            assert compiled.push.next[number][square] == push_oracle.next_square(
                spec, square, (dx, dy)
            ), (player, dx, dy, square)
    for pusher in compiled.piece_ids:
        for victim in compiled.piece_ids:
            assert compiled.push.allowed[compiled.piece_ids.index(pusher)][
                compiled.piece_ids.index(victim)
            ] == push_oracle.may_push(spec, pusher, victim), (pusher, victim)


def test_a_slider_agrees_with_the_oracle() -> None:
    spec = _game(obstacles={"squares": ["d4"]})
    _piece(spec, "R")["moves"][0]["push"] = {"whose": "any"}
    _agrees_with_the_oracle(spec, "R")


def test_a_knight_along_the_move_agrees_with_the_oracle() -> None:
    _agrees_with_the_oracle(_game(), "N")


def test_a_knight_along_the_rank_agrees_with_the_oracle() -> None:
    spec = _game()
    _piece(spec, "N")["moves"][0]["push"] = {"along": "rank", "captures": True}
    _agrees_with_the_oracle(spec, "N")
    compiled = compile_spec(spec)
    assert compiled.push is not None
    assert {(dx, dy) for _, dx, dy in compiled.push.directions} >= {(1, 0), (-1, 0)}


def test_an_alfil_pushes_one_diagonal_step_d2() -> None:
    spec = _game()
    _piece(spec, "N")["moves"] = [_leap([[2, 2]], "rotate4", push=True)]
    _agrees_with_the_oracle(spec, "N")
    landings = push_oracle.landings(spec, "N", 0, 14)
    assert {step for _, _, step in landings} == {(1, 1), (-1, 1), (1, -1), (-1, -1)}


def _hex_game() -> dict[str, Any]:
    """A hexagon of radius 2: a king and a pushing hex step."""
    step = {"kind": "leap", "vectors": [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]]}
    return {
        "name": "hex push test",
        "board": {"width": 5, "height": 5, "lattice": "hex", "shape": "hexagon"},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [step]},
            {"id": "S", "value": 2, "moves": [{**step, "push": True}]},
        ],
        "setup": {
            "orientation": "rotate",
            "placements": [{"piece": "K", "squares": ["c1"]}, {"piece": "S", "squares": ["c2"]}],
        },
        "rules": {},
    }


def test_a_hex_step_agrees_with_the_oracle_and_its_holes_are_off_the_board() -> None:
    spec = _hex_game()
    assert validate_spec(spec) == []
    _agrees_with_the_oracle(spec, "S")
    compiled = compile_spec(spec)
    assert compiled.push is not None and compiled.absent
    # W2: a step onto an absent corner is off the board, never a stop.
    assert -2 not in {square for row in compiled.push.next for square in row}


def test_a_wrapped_edge_agrees_with_the_oracle() -> None:
    spec = _game(wrap={"files": "both"})
    assert validate_spec(spec) == []
    _agrees_with_the_oracle(spec, "R")
    _agrees_with_the_oracle(spec, "N")
    compiled = compile_spec(spec)
    assert compiled.push is not None
    east = compiled.push.directions.index((0, 1, 0))
    assert compiled.push.next[east][5] == 0  # f1 pushed east goes round to a1 (H3)
    # A cylinder's rook reaches b1 from a1 both ways round, and pushes it either way.
    assert {(1, 0), (-1, 0)} <= {
        step for at, _, step in _compiled_landings(compiled, "R", 0, 0) if at == 1
    }


def test_captures_false_keeps_the_capture_bit_off_where_it_pushes_enemies() -> None:
    compiled = compile_spec(_game())
    landings = _compiled_landings(compiled, "R", 0, 7)
    assert landings and all(mode & 2 == 0 and mode & 4 for _, mode, _ in landings)
    spec = _game()
    _piece(spec, "R")["moves"][0]["push"] = {"captures": True}
    landings = _compiled_landings(compile_spec(spec), "R", 0, 7)
    assert all(mode & 2 and mode & 4 for _, mode, _ in landings)


def test_one_table_entry_pushes_a_landing_one_way() -> None:
    """Descriptors along one ray merge into one entry, which holds one step per landing: two
    of them that disagree there are refused (two entries may disagree, a seam's two ways)."""
    spec = _game()
    _piece(spec, "R")["moves"] = [
        _slide([[1, 1]], "none", push=True), _slide([[1, 1]], "none", push={"whose": "own"}),
    ]
    compile_spec(spec)  # one ray, one step
    _piece(spec, "R")["moves"][1]["push"] = {"along": "file"}
    assert validate_spec(spec) == []
    with pytest.raises(SpecError, match="two ways"):
        compile_spec(spec)


def test_the_weights_and_where_a_piece_may_stand() -> None:
    spec = _game(push={"chain": "weight"})
    for pid, weight in (("K", 1), ("R", 3), ("N", 2)):
        _piece(spec, pid)["weight"] = weight
    _piece(spec, "K")["region"] = ["a1", "b1", "a2", "b2"]
    compiled = compile_spec(spec)
    assert compiled.push is not None
    assert compiled.push.weight == (1, 3, 2)
    for player in range(2):
        for kind, pid in enumerate(compiled.piece_ids):
            row = compiled.push.stand[player][kind]
            if pid != "K":
                assert row == ()
                continue
            assert row == tuple(
                push_oracle.stands(spec, pid, player, square)
                for square in range(compiled.n_squares)
            )
    assert compile_spec(_game()).push.weight == ()  # type: ignore[union-attr]
    assert compile_spec(_game()).push.stand == ()  # type: ignore[union-attr]


def test_a_push_landing_is_an_attack_source_r1() -> None:
    compiled = compile_spec(_game())
    assert compiled.push is not None
    rook = compiled.piece_ids.index("R") + 1  # player 0, facing 0
    origin = 7
    for target, _, _ in push_oracle.landings(_game(), "R", 0, origin):
        sources = compiled.attack_sources[0][target]
        kinds = compiled.push.attack_kinds[0][target]
        assert len(kinds) == len(sources)
        found = [k for s, k in zip(sources, kinds, strict=True) if s[:2] == (origin, rook)]
        assert found == [ATTACK_BY_PUSH], (target, found)
    king = compiled.piece_ids.index("K") + 1
    assert all(
        kind == ATTACK_BY_CAPTURE
        for square in range(compiled.n_squares)
        for source, kind in zip(compiled.attack_sources[0][square],
                                compiled.push.attack_kinds[0][square], strict=True)
        if source[1] == king
    )


def test_a_push_of_ones_own_pieces_attacks_nothing() -> None:
    spec = _game()
    _piece(spec, "R")["moves"][0]["push"] = {"whose": "own"}
    _piece(spec, "R")["moves"][0]["mode"] = "move"
    compiled = compile_spec(spec)
    rook = compiled.piece_ids.index("R") + 1
    assert not any(
        source[1] == rook for square in compiled.attack_sources[0] for source in square
    )


# ------------------------------------------------------------- inert, held, refused


def _stored_specs() -> list[Path]:
    return sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))


def test_no_stored_spec_compiles_a_push() -> None:
    """The inertness proof's table half: a stored spec that states no push compiles no
    push rule or push bit, and the tables it crosses are the ones it always had (every
    stored spec's tables fingerprint was checked unmoved by hand, `docs/plan-push.md`).
    The push showcases (Q1074bg1's `rulesets/brink.json` the first) state one, and are
    skipped by what they state, never by name."""
    for path in _stored_specs()[:40]:
        spec = json.loads(path.read_text(encoding="utf-8"))
        if pushing_ids(spec):
            continue
        compiled = compile_spec(spec)
        assert compiled.push is None, path.name
        for table in (compiled.leaps, compiled.slides, compiled.hops):
            assert not any(
                mode & MODE_PUSH
                for frame in table for kind in frame for square in kind for entry in square
                for mode in entry[1]
            ), path.name


def test_the_bridge_admits_the_single_push_since_q1074b2_and_rows_since_q1074b4() -> None:
    assert rust_supports(compile_spec(_load("rulesets/chess.json"))) is None
    assert rust_supports(compile_spec(_game())) is None
    assert rust_supports(compile_spec(_game(push={"chain": "row"}))) is None


def test_fairy_stockfish_refuses_a_push_by_name() -> None:
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    with pytest.raises(Untranslatable) as refused:
        translate(_game())
    assert "pieces[].moves[].push" in str(refused.value)
    assert "never been through" not in str(refused.value)


def test_the_keys_are_offered_to_a_model_since_q1074b2() -> None:
    """Q1074b2 folded the keys into the proposable sets: the schema offers them (the chain
    as `"one"`, and since Q1074b4 `"row"` and `"weight"`; since Q1074b6 `no_push_back`, and
    since Q1074b7 `walls`) and the preamble explains them."""
    from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE

    schema = rule_spec_schema()
    push = schema["properties"]["rules"]["properties"]["push"]
    assert set(push["properties"]) == {"chain", "edge", "walls", "no_push_back"}
    assert push["properties"]["walls"]["enum"] == ["block", "crush"]
    assert push["properties"]["chain"]["enum"] == ["one", "row", "weight"]
    piece = schema["properties"]["pieces"]["items"]["properties"]
    assert {"pushes", "weight"} <= set(piece)
    assert "push" in piece["moves"]["items"]["properties"]
    assert "pieces[].moves[].push" in SCHEMA_PREAMBLE and "pieces[].pushes" in SCHEMA_PREAMBLE
    assert "rules.push.no_push_back" in SCHEMA_PREAMBLE
