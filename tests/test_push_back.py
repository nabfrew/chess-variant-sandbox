"""No push back (Q1074b6, K5, `docs/plan-push.md` §3.12): `rules.push.no_push_back`.

After a push that moved the piece landed on from `x` to `y` and did not throw it off, no push
on the very next move may carry that piece from `y` back onto `x`: not the piece itself
shoving back, and (Q1078, the user, 2026-10-31) not a different piece shoving the row back
from its far end, through a gap or through pieces in between, whoever's they are. The bar is state
(`Game::push_bar`, `GameState.push_bar`): set and cleared in `make`, restored in `unmake`, in
the key only under the rule. The core's push words, with the bar, agree with
`tests/push_oracle.py` over seeded games, both seats, the bar and the key checked every ply;
`make`/`unmake` agree with `apply` by perft from a barred position; the reversal is refused
for one move and allowed after; repetition counts the bar (two positions alike but for it
have different keys).
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import push_oracle
import pytest

from chess_variant_sandbox.engine.bridge import build_tables, rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PUSH, decode_move, move_push, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.normal import normal_form

ROOT = Path(__file__).resolve().parents[1]
NO_BAR = (-1, -1)


def _sumo(no_push_back: bool = True) -> dict[str, Any]:
    """Chess with every non-royal capture a push (enemy only), the knight pushing along the
    rank, no en passant, and (by default) no push back: §6.1's reading of the published
    rules."""
    spec: dict[str, Any] = json.loads((ROOT / "rulesets/chess.json").read_text(encoding="utf-8"))
    spec["name"] = "push back test"
    for piece in spec["pieces"]:
        if piece.get("royal"):
            continue
        for move in piece["moves"]:
            if move.get("mode") != "move":
                move["push"] = {"along": "rank"} if piece["id"] == "N" else True
    del spec["rules"]["en_passant"]
    if no_push_back:
        spec["rules"]["push"] = {"no_push_back": True}
    return spec


def _rows() -> dict[str, Any]:
    """A 7x7 board, no check, `chain: "row"`, no push back: rooks that push any piece's row on
    their lines, a pawn that pushes straight ahead and never captures."""
    return {
        "name": "push back rows",
        "board": {"width": 7, "height": 7},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            {"id": "R", "value": 5, "pushes": ["R", "P"], "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "P", "value": 1, "pushes": ["P"], "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move",
                 "push": True},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "g1"]},
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2", "g2"]},
        ]},
        "rules": {
            "win": "royal_capture",
            "push": {"chain": "row", "no_push_back": True},
            "max_plies": 200,
        },
    }


def _duel() -> dict[str, Any]:
    """Two kings far apart and a rook each that pushes on its lines, no check: the smallest
    game in which a push can be pushed straight back."""
    return {
        "name": "push back duel",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "push": True},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["f1"]},
            {"piece": "R", "squares": ["a1"]},
        ]},
        "rules": {"win": "royal_capture", "push": {"no_push_back": True}, "max_plies": 200},
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    assert rust_supports(compile_spec(spec)) is None
    return Engine(spec)


def _decoder(engine: Engine) -> Any:
    spec = engine.spec
    return lambda value: (spec.owner_of(value), spec.piece_ids[spec.type_of(value)])


def _as_oracle(engine: Engine, move: int) -> push_oracle.PushMove:
    push = engine.spec.push
    assert push is not None
    origin, target, _kind, promoted = decode_move(move)
    _player, dx, dy = push.directions[move_push(move)]
    return origin, target, (dx, dy), promoted


def _core_pushes(engine: Engine, state: GameState) -> set[push_oracle.PushMove]:
    return {
        _as_oracle(engine, move) for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_PUSH
    }


def _barred(engine: Engine, state: GameState, bar: tuple[int, int]) -> GameState:
    """``state`` with the push-back bar ``bar``, keyed afresh."""
    barred = state.with_(push_bar=bar)
    key = engine._compute_hash(barred)
    return barred.with_(key=key, repetition=(key,))


def _pushes(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    o, t = engine._square_index(origin), engine._square_index(target)
    return [
        move for move in engine.legal_moves(state)
        if move_type(move) == MOVE_PUSH and decode_move(move)[:2] == (o, t)
    ]


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    o, t = engine._square_index(origin), engine._square_index(target)
    found = [move for move in engine.legal_moves(state) if decode_move(move)[:2] == (o, t)]
    assert len(found) == 1, (origin, target, found)
    return found[0]


@pytest.mark.parametrize("make", [_sumo, _rows], ids=["sumo", "rows"])
def test_the_bar_equals_the_oracle_s_over_seeded_games(make: Any) -> None:
    """Both seats, every ply: the core's push words are the oracle's under the bar the oracle
    computed, the bar the core holds is the oracle's, and the key is the one computed from
    scratch. The bar must bind (take a push away) in some of these plies."""
    spec = make()
    engine = _engine(spec)
    decode = _decoder(engine)
    seen = {0: 0, 1: 0}
    bound = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        bar = NO_BAR
        for ply in range(120):
            player = state.side_to_move
            assert state.push_bar == bar, (seed, ply)
            expected = push_oracle.push_moves(spec, decode, state.board, player, bar)
            assert _core_pushes(engine, state) == expected, (seed, ply)
            unbarred = push_oracle.push_moves(spec, decode, state.board, player)
            bound += len(unbarred - expected)
            seen[player] += len(expected)
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            pushes = [move for move in legal if move_type(move) == MOVE_PUSH]
            move = rng.choice(pushes if pushes and rng.random() < 0.8 else legal)
            push = _as_oracle(engine, move) if move_type(move) == MOVE_PUSH else None
            bar = push_oracle.bar_after(spec, decode, state.board, push, player)
            state = engine.apply(state, move)
            assert engine.hash(state) == engine._compute_hash(state), (seed, ply)
    assert seen[0] > 50 and seen[1] > 50
    assert bound > 0


def test_the_reversal_is_refused_for_one_move_and_allowed_after() -> None:
    engine = _engine(_duel())
    state = engine.make_state(
        {"f1": ("K", 0), "a1": ("R", 0), "f6": ("K", 1), "a3": ("R", 1)},
        side_to_move=0, moved=["a1", "a3"],
    )
    state = engine.apply(state, _pushes(engine, state, "a1", "a3")[0])
    a3, a4 = engine._square_index("a3"), engine._square_index("a4")
    assert state.push_bar == (a4, a3)
    # Black's rook, pushed from a3 to a4, may not push White's rook from a3 straight back.
    assert _pushes(engine, state, "a4", "a3") == []
    # One move each, and the bar is gone: the same push is offered.
    state = engine.apply(state, _move(engine, state, "f6", "e6"))
    assert state.push_bar == NO_BAR
    state = engine.apply(state, _move(engine, state, "f1", "e1"))
    assert len(_pushes(engine, state, "a4", "a3")) == 1


def test_the_bar_is_cleared_by_any_other_move_and_names_only_its_own_pair() -> None:
    engine = _engine(_duel())
    state = engine.make_state(
        {"f1": ("K", 0), "a1": ("R", 0), "f6": ("K", 1), "a3": ("R", 1), "b4": ("R", 1)},
        side_to_move=0, moved=["a1", "a3", "b4"],
    )
    state = engine.apply(state, _pushes(engine, state, "a1", "a3")[0])
    # The barred rook's other moves, and the other rook's, stand.
    assert _move(engine, state, "a4", "a5")
    assert _move(engine, state, "b4", "b3")
    state = engine.apply(state, _move(engine, state, "b4", "b3"))
    assert state.push_bar == NO_BAR


def test_a_push_that_throws_the_piece_off_leaves_no_bar() -> None:
    engine = _engine(_duel())
    state = engine.make_state(
        {"f1": ("K", 0), "a4": ("R", 0), "f6": ("K", 1), "a6": ("R", 1)},
        side_to_move=0, moved=["a4", "a6"],
    )
    state = engine.apply(state, _pushes(engine, state, "a4", "a6")[0])
    assert sum(map(bool, state.board)) == 3
    assert state.push_bar == NO_BAR


def test_undo_restores_the_bar() -> None:
    """The core's perft makes and takes back every move in place from a barred position;
    the same tree counted through `apply` gives the same number."""
    engine = _engine(_rows())
    state = engine.make_state({
        "d1": ("K", 0), "a1": ("R", 0), "a3": ("R", 1), "c3": ("P", 0), "c4": ("P", 1),
        "d7": ("K", 1), "g7": ("R", 1), "g5": ("R", 0),
    }, side_to_move=0, moved=["a1", "a3", "c3", "c4", "g7", "g5"])
    state = engine.apply(state, _pushes(engine, state, "a1", "a3")[0])
    assert state.push_bar != NO_BAR

    def by_apply(position: GameState, depth: int) -> int:
        if depth == 0:
            return 1
        return sum(
            by_apply(engine.apply(position, move), depth - 1)
            for move in engine.legal_moves(position)
        )

    assert engine.perft(state, 3) == by_apply(state, 3)


def test_repetition_counts_the_bar() -> None:
    """Two positions alike but for the bar have different keys, so neither the repetition
    rule nor a transposition table takes one for the other; with the same bar they agree."""
    engine = _engine(_duel())
    state = engine.make_state(
        {"f1": ("K", 0), "a1": ("R", 0), "f6": ("K", 1), "a3": ("R", 1)},
        side_to_move=0, moved=["a1", "a3"],
    )
    pushed = engine.apply(state, _pushes(engine, state, "a1", "a3")[0])
    assert pushed.push_bar != NO_BAR
    unbarred = _barred(engine, pushed, NO_BAR)
    assert unbarred.board == pushed.board
    assert engine.hash(unbarred) != engine.hash(pushed)
    assert _barred(engine, unbarred, pushed.push_bar).key == pushed.key
    # A non-removing push keeps the repetition window open (it is reversible), and the
    # barred position's key is what the window holds.
    assert pushed.repetition[-1] == pushed.key and len(pushed.repetition) == 2


def test_the_bar_is_in_the_key_only_under_the_rule() -> None:
    without = compile_spec(_sumo(no_push_back=False))
    with_bar = compile_spec(_sumo())
    assert without.push is not None and without.push.zobrist_bar == ()
    assert with_bar.push is not None and len(with_bar.push.zobrist_bar) == 2 * with_bar.n_squares
    assert build_tables(without)["zobrist_push_bar"] == []
    engine = _engine(_sumo(no_push_back=False))
    state = engine.initial_state()
    for _ in range(12):
        legal = engine.legal_moves(state)
        pushes = [move for move in legal if move_type(move) == MOVE_PUSH]
        state = engine.apply(state, (pushes or legal)[0])
        assert state.push_bar == NO_BAR


def test_a_bar_from_python_is_read_only_under_the_rule() -> None:
    engine = _engine(_sumo(no_push_back=False))
    state = engine.initial_state()
    stray = state.with_(push_bar=(20, 12))
    assert engine._compute_hash(stray) == engine._compute_hash(state)


def test_the_validator_takes_the_rule_and_refuses_a_bar_nothing_can_meet() -> None:
    assert validate_spec(_sumo()) == []
    own = _duel()
    own["pieces"][1]["moves"][0]["push"] = {"whose": "own"}
    errors = validate_spec(own)
    assert any("no_push_back (K5)" in error and "can never bind" in error for error in errors)
    own["rules"]["push"] = {"no_push_back": False}
    assert validate_spec(own) == []
    # It names a new game: the hash moves, and the off value is dropped.
    assert spec_hash(_sumo()) != spec_hash(_sumo(no_push_back=False))
    off = _sumo(no_push_back=False)
    off["rules"]["push"] = {"no_push_back": False}
    assert spec_hash(off) == spec_hash(_sumo(no_push_back=False))
    assert normal_form(_sumo())["rules"]["push"] == {"no_push_back": True}


def test_the_schema_the_rules_and_the_prose_offer_it() -> None:
    rules = rule_spec_schema()["properties"]["rules"]["properties"]
    assert "no_push_back" in rules["push"]["properties"]
    assert "No push back" in describe_spec(_sumo())
    assert "No push back" not in describe_spec(_sumo(no_push_back=False))
    assert "no_push_back" in (ROOT / "RULES.md").read_text(encoding="utf-8")


def test_the_spec_is_left_untouched() -> None:
    spec = _sumo()
    kept = copy.deepcopy(spec)
    _engine(spec)
    assert spec == kept


# Q1078 (the user, 2026-10-31: *"shove no-shove-back rule doesn't cover shoving back-and-forth in
# rows where there's a piece in between"*). Each shape below was a 2-ply back-and-forth before:
# the row shoved, then shoved straight back by a piece other than the one landed on.


def _row_state(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    """`_rows`' game with the kings out of the way and `pieces` on the fourth rank, White to
    move."""
    placed = {"g1": ("K", 0), "g7": ("K", 1), **pieces}
    return engine.make_state(placed, side_to_move=0, moved=list(pieces))


def _shove(engine: Engine, state: GameState, origin: str, target: str) -> GameState:
    pushes = _pushes(engine, state, origin, target)
    assert len(pushes) == 1, (origin, target, pushes)
    return engine.apply(state, pushes[0])


def test_a_different_shover_at_the_far_end_may_not_shove_the_row_back() -> None:
    engine = _engine(_rows())
    state = _row_state(engine, {"a4": ("R", 0), "b4": ("R", 1), "c4": ("R", 1)})
    state = _shove(engine, state, "a4", "b4")
    b4, c4 = engine._square_index("b4"), engine._square_index("c4")
    assert state.push_bar == (c4, b4)
    # Black's d4 rook, not the one landed on, would shove c4 back onto b4 and White onto a4.
    assert _pushes(engine, state, "d4", "c4") == []
    # The piece landed on may not shove back either (Q1074b6's case).
    assert _pushes(engine, state, "c4", "b4") == []
    # The bar lasts one move: a move each, and the same shove is offered.
    state = engine.apply(state, _move(engine, state, "g7", "f7"))
    state = engine.apply(state, _move(engine, state, "g1", "f1"))
    assert len(_pushes(engine, state, "d4", "c4")) == 1


def test_a_shover_sliding_in_across_a_gap_may_not_shove_the_row_back() -> None:
    engine = _engine(_rows())
    state = _row_state(engine, {"a4": ("R", 0), "b4": ("R", 1), "g4": ("R", 1)})
    state = _shove(engine, state, "a4", "b4")
    # Black's g4 rook slides d4..f4 and lands on its own c4 rook: the row would go back.
    assert _pushes(engine, state, "g4", "c4") == []


def test_a_gap_between_the_shover_and_the_shoved_bars_the_shove_back() -> None:
    engine = _engine(_rows())
    state = _row_state(engine, {"a4": ("R", 0), "c4": ("R", 1), "e4": ("R", 1)})
    state = _shove(engine, state, "a4", "c4")
    # White slid a4 to c4; Black's d4 rook (pushed there) and its e4 rook behind it.
    assert _pushes(engine, state, "d4", "c4") == []
    assert _pushes(engine, state, "e4", "d4") == []


def test_an_own_piece_in_between_is_barred_from_coming_back() -> None:
    """White shoves its own b4 rook and Black's c4 one: the piece landed on is White's, which
    the old bar never named, and Black's d4 rook would shove both back."""
    engine = _engine(_rows())
    state = _row_state(engine, {"a4": ("R", 0), "b4": ("R", 0), "c4": ("R", 1)})
    state = _shove(engine, state, "a4", "b4")
    c4, b4 = engine._square_index("c4"), engine._square_index("b4")
    assert state.push_bar == (c4, b4)
    assert _pushes(engine, state, "d4", "c4") == []


def test_a_longer_row_through_mixed_pieces_is_barred_from_coming_back() -> None:
    engine = _engine(_rows())
    state = _row_state(
        engine, {"a4": ("R", 0), "b4": ("R", 1), "c4": ("R", 0), "d4": ("R", 1)}
    )
    state = _shove(engine, state, "a4", "b4")
    # Black's e4 rook would shove d4, c4 and b4 back, the piece just shoved among them.
    assert _pushes(engine, state, "e4", "d4") == []
    assert _pushes(engine, state, "c4", "b4") == []


def test_the_shoved_piece_may_still_be_shoved_aside() -> None:
    """Only carrying it straight back is barred: a shove of the piece just shoved across its
    file stands."""
    engine = _engine(_rows())
    state = _row_state(
        engine, {"a4": ("R", 0), "b4": ("R", 1), "c4": ("R", 1), "c7": ("R", 1)}
    )
    state = _shove(engine, state, "a4", "b4")
    # c7 slides down onto c4 (the piece just shoved) and shoves it to c3.
    assert len(_pushes(engine, state, "c7", "c4")) == 1


def test_dohyo_s_bouncer_may_not_shove_a_row_back_through_its_own_apprentice() -> None:
    """The library game: White's bouncer shoves an apprentice from c4 to d4; Black's bouncer
    on e4, which pushes either side's pieces, would shove the apprentice and White's bouncer
    straight back (weight 1 + 2, strength 3)."""
    spec = json.loads((ROOT / "rulesets/dohyo.json").read_text(encoding="utf-8"))
    engine = _engine(spec)
    state = engine.make_state(
        {"d1": ("C", 0), "d7": ("C", 1), "b4": ("O", 0), "c4": ("D", 1), "e4": ("O", 1)},
        side_to_move=0, moved=["b4", "c4", "e4"],
    )
    state = _shove(engine, state, "b4", "c4")
    assert _pushes(engine, state, "e4", "d4") == []
    state = engine.apply(state, _move(engine, state, "d7", "d6"))
    state = engine.apply(state, _move(engine, state, "d1", "d2"))
    assert len(_pushes(engine, state, "e4", "d4")) == 1
