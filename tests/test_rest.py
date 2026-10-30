"""Schema 29 (Q983a): rest as a piece-type property, in alternating play and in the core.

`docs/plan-rest.md` §3 is the rule and §4 its build; this file is the Q983a box's acceptance,
one test per decision of §3: the key and its refusals, the worked example of §3.2, the type that
rests after a promotion, a resting piece captured, displaced or transformed by the other player,
the lift on the final list (with check, with `must_capture`, with `forbid_immobility`), a resting
piece's check answered, long turns refused, castling, a drop. The core is held to the declared
oracle (`tests/rest_oracle.py`) move list by move list and rest list by rest list over random
games of the two alternating rest test specs (`tests/_rest_checkmate.json`, Gardner under
checkmate with the queen resting 2 and the rooks 1; `tests/_rest_drops.json`, Gardner by royal
capture with drops, knight and king resting 1, queen 2), with the incremental key held to the
slow definition at every move, and their perft is frozen at depths 1 to 3 after the oracle's
own perft agreed with it. The rounds are `test_simultaneous_oracle.py`'s and
`test_rest_equivalence.py`'s; the spy game's turns are `test_showcases_q935f.py`'s.
"""

from __future__ import annotations

import copy
import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from rest_oracle import after_turn, as_rest, end_of_move, legal, lists_of

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import piece_rests
from chess_variant_sandbox.spec.spelling import current_spelling
from tests.feature_inventory import _schema_twenty_nine_uses

HERE = Path(__file__).parent
ROOT = HERE.parent
GARDNER = json.loads((ROOT / "rulesets" / "gardner.json").read_text(encoding="utf-8"))
CHESS = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


def load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    return loaded


def resting(base: dict[str, Any], rests: dict[str, int], **rules: Any) -> dict[str, Any]:
    """``base`` at schema 29 with these rests, and these rules set (None removes one)."""
    spec = copy.deepcopy(base)
    spec["schema_version"] = 29
    for piece in spec["pieces"]:
        if piece["id"] in rests:
            piece["rest"] = rests[piece["id"]]
    for key, value in rules.items():
        if value is None:
            spec["rules"].pop(key, None)
        else:
            spec["rules"][key] = value
    return spec


@cache
def engine_of(text: str) -> Engine:
    return Engine(json.loads(text))


def engine(spec: dict[str, Any]) -> Engine:
    return engine_of(json.dumps(spec, sort_keys=True))


def sq(eng: Engine, name: str) -> int:
    return eng._square_index(name)


def place(
    eng: Engine,
    pieces: dict[str, tuple[str, int]],
    side: int = 0,
    rest: tuple[dict[str, int], dict[str, int]] = ({}, {}),
    moved: tuple[str, ...] = (),
) -> GameState:
    """A position with these pieces and these resting pieces, keyed by the slow hash."""
    state = eng.make_state(pieces, side_to_move=side, moved=moved or tuple(pieces))
    # A game with drops has hands of its types' length; the hand-built position starts empty.
    state = state.with_(hands=eng.initial_state().hands)
    state = state.with_(rest=tuple(
        tuple(sorted((sq(eng, name), left) for name, left in each.items())) for each in rest
    ))
    key = eng._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def move(eng: Engine, state: GameState, origin: str, target: str, kind: int | None = None) -> int:
    for word in eng.legal_moves(state):
        o, t, k, _ = decode_move(word)
        if (o, t) == (sq(eng, origin), sq(eng, target)) and (kind is None or k == kind):
            return word
    raise AssertionError(f"no legal move {origin}->{target}")


def froms(eng: Engine, state: GameState) -> set[int]:
    return {decode_move(word)[0] for word in eng.legal_moves(state)}


def play(eng: Engine, state: GameState, word: int) -> GameState:
    """One move through the core, held to the oracle's list and rest and the slow key."""
    assert eng.legal_moves(state) == legal(eng, state)
    after = eng.apply(state, word)
    expected = after_turn(
        eng, lists_of(state), after.board, state.side_to_move, end_of_move(eng, state, after, word)
    )
    assert after.rest == as_rest(expected)
    assert after.key == eng._compute_hash(after)
    return after


# ------------------------------------------------------------------ the key and its refusals


def test_the_collector_names_the_key_unless_it_is_off() -> None:
    assert _schema_twenty_nine_uses(resting(GARDNER, {"Q": 2})) == {"pieces.rest"}
    assert _schema_twenty_nine_uses(resting(GARDNER, {"Q": 0})) == set()
    assert _schema_twenty_nine_uses(GARDNER) == set()


def test_a_rest_of_0_is_the_game_without_it() -> None:
    """Q1008c: 0 is the off value, so it is accepted, named and tidied as its absence."""
    off = resting(GARDNER, {"Q": 0})
    assert validate_spec(off) == []
    assert spec_hash(off) == spec_hash(GARDNER)
    assert "rest" not in next(p for p in current_spelling(off)["pieces"] if p["id"] == "Q")


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (5, "from 1 to 4"),
        (True, "from 1 to 4"),
        (-1, "from 1 to 4"),
        ("2", "from 1 to 4"),
    ],
)
def test_a_rest_out_of_range_is_refused(value: Any, words: str) -> None:
    assert any(words in error for error in validate_spec(resting(GARDNER, {"Q": value})))


def test_a_rest_on_a_wild_piece_or_beside_a_long_turn_is_refused() -> None:
    wild = load("_simultaneous_wild")
    del wild["rules"]["simultaneous"]
    for piece in wild["pieces"]:
        piece.pop("beats", None)
    wild["rules"]["win"] = "checkmate"
    assert validate_spec(wild) == [], validate_spec(wild)
    assert any("wild piece" in error for error in validate_spec(resting(wild, {"N": 1})))
    long = resting(GARDNER, {"Q": 1}, turn={"moves": 2})
    assert "L2" in refused_ids(validate_spec(long))
    ramp = resting(GARDNER, {"Q": 1}, turn={"moves": 1, "ramp": [1, 2]})
    assert "L2" in refused_ids(validate_spec(ramp))


def test_the_old_switch_is_read_only_through_the_converter() -> None:
    """Schema 29 retired `rules.simultaneous.rest`; since Q1008c the validator knows no such
    key, and the converter writes it as `"rest": 1` on every piece a player moves."""
    from chess_variant_sandbox.spec.schema import SpecError

    old = copy.deepcopy(load("_simultaneous_gardner"))
    old["rules"]["simultaneous"]["rest"] = True
    assert any("rest" in error for error in validate_spec(old))
    with pytest.raises(SpecError):
        Engine(old)
    converted = current_spelling(old)
    assert "rest" not in converted["rules"]["simultaneous"]
    assert all(piece.get("rest") == 1 for piece in converted["pieces"] if piece.get("moves"))
    assert validate_spec(converted) == []


def test_without_a_rest_nothing_crosses_and_chess_is_chess() -> None:
    assert spec_hash(CHESS) == "831c77e75650"
    tables = canonical_tables(compile_spec(CHESS))
    assert "piece_rest" not in tables and "zobrist_rest" not in tables
    compiled = compile_spec(resting(GARDNER, {"Q": 2, "R": 1}))
    assert compiled.rest_max == 2
    assert len(compiled.zobrist_rest) == 2 * 2 * compiled.n_squares
    assert piece_rests(resting(GARDNER, {"Q": 2}))["Q"] == 2


# ----------------------------------------------- §3.2: what n means, and how the count runs


def test_the_worked_example_of_section_three_two() -> None:
    """White's queen rests 2: moved on turn 10, it is not offered on turns 11 and 12, and is
    on turn 13; Black's turns between never count."""
    eng = engine(resting(GARDNER, {"Q": 2}, win="royal_capture"))
    state = place(eng, {"a1": ("K", 0), "d1": ("Q", 0), "b1": ("N", 0),
                        "e5": ("K", 1), "a5": ("R", 1)})
    queen_moves = {sq(eng, "d1")}
    state = play(eng, state, move(eng, state, "d1", "d3"))  # turn 10
    assert state.rest == (((sq(eng, "d3"), 2),), ())
    state = play(eng, state, move(eng, state, "a5", "a4"))
    assert sq(eng, "d3") not in froms(eng, state)  # turn 11
    state = play(eng, state, move(eng, state, "b1", "c3"))
    assert state.rest == (((sq(eng, "d3"), 1),), ())
    state = play(eng, state, move(eng, state, "a4", "a5"))
    assert sq(eng, "d3") not in froms(eng, state)  # turn 12
    state = play(eng, state, move(eng, state, "c3", "b1"))
    assert state.rest == ((), ())
    state = play(eng, state, move(eng, state, "a5", "a4"))
    assert sq(eng, "d3") in froms(eng, state)  # turn 13
    assert queen_moves


def test_rest_one_is_the_next_own_turn_and_a_lifted_turn_counts() -> None:
    eng = engine(resting(GARDNER, {"R": 1, "K": 1}, win="royal_capture"))
    state = place(eng, {"a1": ("K", 0), "c1": ("R", 0), "e5": ("K", 1)})
    state = play(eng, state, move(eng, state, "c1", "c3"))
    state = play(eng, state, move(eng, state, "e5", "e4"))
    assert sq(eng, "c3") not in froms(eng, state)
    state = play(eng, state, move(eng, state, "a1", "a2"))
    assert state.rest == (((sq(eng, "a2"), 1),), ((sq(eng, "e4"), 1),))
    # Black's lone king is all Black has: the rest is lifted, and the lifted move rests afresh.
    lone = eng.legal_moves(state)
    assert lone and all(decode_move(m)[0] == sq(eng, "e4") for m in lone)
    state = play(eng, state, move(eng, state, "e4", "e5"))
    assert state.rest[1] == ((sq(eng, "e5"), 1),)


# --------------------------------------------- §3.3, §3.4: which piece rests, and where


def test_a_pawn_promoted_to_a_queen_rests_as_the_queen() -> None:
    eng = engine(resting(GARDNER, {"Q": 2}))
    state = place(eng, {"a1": ("K", 0), "d4": ("P", 0), "a5": ("K", 1)})
    promote = [
        m for m in eng.legal_moves(state)
        if decode_move(m)[:2] == (sq(eng, "d4"), sq(eng, "d5"))
        and eng.spec.piece_ids[decode_move(m)[3]] == "Q"
    ]
    assert promote
    after = play(eng, state, promote[0])
    assert after.rest == (((sq(eng, "d5"), 2),), ())


def test_a_resting_piece_captured_rests_no_more_and_its_captor_rests() -> None:
    eng = engine(resting(GARDNER, {"R": 1, "N": 1}, win="royal_capture"))
    state = place(eng, {"a1": ("K", 0), "c1": ("R", 0), "e5": ("K", 1), "b5": ("N", 1)})
    state = play(eng, state, move(eng, state, "c1", "c3"))
    state = play(eng, state, move(eng, state, "b5", "c3"))
    assert state.rest == ((), ((sq(eng, "c3"), 1),))


def test_a_resting_piece_taken_into_a_hand_arrives_free_and_its_drop_rests_afresh() -> None:
    eng = engine(load("_rest_drops"))
    knight = eng.spec.piece_ids.index("N")
    state = place(eng, {"a1": ("K", 0), "c3": ("N", 0), "e5": ("K", 1), "d4": ("P", 1)},
                  side=1, rest=({"c3": 1}, {}))
    state = play(eng, state, move(eng, state, "d4", "c3"))
    # The resting knight is taken: its rest ends, and it is in Black's hand, free.
    assert state.rest == ((), ())
    assert state.hands[1][knight] == 1
    state = play(eng, state, move(eng, state, "a1", "a2"))
    drop = next(
        m for m in eng.legal_moves(state)
        if decode_move(m)[2] == 4 and decode_move(m)[3] == knight
    )
    after = play(eng, state, drop)
    assert after.rest[1] == ((decode_move(drop)[1], 1),)


def test_a_drop_rests_as_the_dropped_type() -> None:
    eng = engine(load("_rest_drops"))
    knight = eng.spec.piece_ids.index("N")
    state = place(eng, {"a1": ("K", 0), "e5": ("K", 1)})
    hands = list(state.hands[0]) or [0] * eng.spec.n_types
    hands[knight] = 1
    state = state.with_(hands=(tuple(hands), state.hands[1] or tuple([0] * eng.spec.n_types)))
    state = state.with_(key=eng._compute_hash(state))
    drop = next(m for m in eng.legal_moves(state) if decode_move(m)[2] == 4)
    after = play(eng, state, drop)
    assert after.rest == (((decode_move(drop)[1], 1),), ())
    # A drop is never filtered by the rest: the hand has no resting piece.
    resting_board = place(eng, {"a1": ("K", 0), "c1": ("N", 0), "e5": ("K", 1)},
                          rest=({"c1": 1, "a1": 1}, {}))
    resting_board = resting_board.with_(hands=(tuple(hands), state.hands[1]))
    resting_board = resting_board.with_(key=eng._compute_hash(resting_board))
    moves = eng.legal_moves(resting_board)
    assert moves and all(decode_move(m)[2] == 4 for m in moves)


def _strike(position: str, effect: str = "promote") -> dict[str, Any]:
    return resting(
        GARDNER, {"R": 1, "N": 1, "P": 1}, win="royal_capture",
        strike={"effect": effect, "position": position, "immune": ["K"],
                "chain": [{"from": "N", "to": "B"}]},
        promotion=None,
    )


def test_a_resting_piece_displaced_by_a_swap_rests_no_more() -> None:
    eng = engine(_strike("swap"))
    state = place(eng, {"a1": ("K", 0), "c1": ("R", 0), "e5": ("K", 1), "b5": ("N", 1)},
                  side=1)
    state = play(eng, state, move(eng, state, "b5", "c3"))
    assert state.rest == ((), ((sq(eng, "c3"), 1),))
    # White's rook strikes the resting knight: it becomes a bishop on the rook's origin.
    state = play(eng, state, move(eng, state, "c1", "c3"))
    assert eng.spec.piece_ids[eng.spec.type_of(state.board[sq(eng, "c1")])] == "B"
    assert state.rest == (((sq(eng, "c3"), 1),), ())


def test_a_resting_piece_transformed_where_it_stands_rests_no_more_and_the_striker_rests_home() -> None:
    eng = engine(_strike("stay"))
    state = place(eng, {"a1": ("K", 0), "c1": ("R", 0), "e5": ("K", 1), "b5": ("N", 1)},
                  side=1)
    state = play(eng, state, move(eng, state, "b5", "c3"))
    state = play(eng, state, move(eng, state, "c1", "c3"))
    assert eng.spec.piece_ids[eng.spec.type_of(state.board[sq(eng, "c3")])] == "B"
    # The rook struck from where it stood, and rests there; the knight is a bishop now.
    assert state.rest == (((sq(eng, "c1"), 1),), ())


# --------------------------------------------- §3.6, §3.7: check, and the lift


def test_a_resting_piece_gives_check_and_its_check_must_be_answered() -> None:
    eng = engine(load("_rest_checkmate"))
    state = place(eng, {"a1": ("K", 0), "c2": ("R", 0), "e5": ("K", 1), "a4": ("Q", 1)},
                  rest=({}, {"a4": 2}))
    assert eng.in_check(state, 0)
    answers = eng.legal_moves(state)
    assert answers
    for word in answers:
        after = eng.apply(state, word)
        assert not eng.in_check(after, 0), eng.describe_move(state, word)


def test_a_check_only_a_resting_piece_can_answer_lifts_the_rest() -> None:
    eng = engine(load("_rest_checkmate"))
    # White's king on a1 is boxed by its own pieces and checked along the first rank; only
    # the resting rook can block or take.
    state = place(
        eng,
        {"a1": ("K", 0), "a2": ("P", 0), "b2": ("P", 0), "c2": ("R", 0),
         "e1": ("Q", 1), "e5": ("K", 1)},
        rest=({"c2": 1}, {}),
    )
    assert eng.in_check(state, 0)
    moves = eng.legal_moves(state)
    assert moves and all(decode_move(m)[0] == sq(eng, "c2") for m in moves)
    assert moves == legal(eng, state)
    assert eng.terminal(state, moves) is None


def test_must_capture_forces_a_capture_only_among_the_pieces_that_may_move() -> None:
    eng = engine(resting(GARDNER, {"R": 1}, win="royal_capture", must_capture=True))
    state = place(eng, {"a1": ("K", 0), "c1": ("R", 0), "e5": ("K", 1), "c5": ("N", 1)},
                  rest=({"c1": 1}, {}))
    moves = eng.legal_moves(state)
    # The rook's capture is the resting piece's, so nothing forces it: the king walks.
    assert moves and all(decode_move(m)[0] == sq(eng, "a1") for m in moves)
    assert moves == legal(eng, state)
    free = state.with_(rest=((), ()))
    assert {decode_move(m)[0] for m in eng.legal_moves(free)} == {sq(eng, "c1")}


IMMOBILE = {
    "schema_version": 29,
    "name": "rest and immobility",
    "board": {"width": 4, "height": 4},
    "pieces": [
        {"id": "K", "royal": True, "value": 100, "rest": 1,
         "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
        {"id": "P", "value": 1,
         "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["a1"]}, {"piece": "P", "squares": ["d2"]},
    ]},
    "rules": {"win": "royal_capture", "forbid_immobility": True, "max_plies": 60},
}


def test_the_lift_is_judged_on_the_final_list() -> None:
    """The rest leaves the pawn's one move, which `forbid_immobility` then takes (it would
    strand the pawn on the last rank): the list the rest shortened is empty, so it is made
    again without the rest, and the resting king may move (§3.7)."""
    assert validate_spec(IMMOBILE) == [], validate_spec(IMMOBILE)
    eng = engine(IMMOBILE)
    state = place(eng, {"a1": ("K", 0), "d3": ("P", 0), "d1": ("K", 1)}, rest=({"a1": 1}, {}))
    moves = eng.legal_moves(state)
    assert moves and all(decode_move(m)[0] == sq(eng, "a1") for m in moves)
    assert moves == legal(eng, state)


def test_a_resting_king_may_not_castle_and_a_resting_rook_is_carried() -> None:
    eng = engine(resting(CHESS, {"K": 1, "R": 1}))
    pieces = {"e1": ("K", 0), "h1": ("R", 0), "a2": ("P", 0), "e8": ("K", 1), "a7": ("P", 1)}
    king_rests = place(eng, pieces, rest=({"e1": 1}, {}), moved=("a2", "e8", "a7"))
    assert not any(decode_move(m)[2] == MOVE_CASTLE for m in eng.legal_moves(king_rests))
    rook_rests = place(eng, pieces, rest=({"h1": 1}, {}), moved=("a2", "e8", "a7"))
    castle = [m for m in eng.legal_moves(rook_rests) if decode_move(m)[2] == MOVE_CASTLE]
    assert castle
    after = play(eng, rook_rests, castle[0])
    # The king made the move and rests; the rook it carried rests no more.
    assert after.rest == (((decode_move(castle[0])[1], 1),), ())


# --------------------------------------------- the core and the oracle, held together

SPECS = ("_rest_checkmate", "_rest_drops")


def walk(eng: Engine, seed: int, plies: int) -> int:
    rng = random.Random(seed)
    state = eng.initial_state()
    lifted = 0
    for _ in range(plies):
        moves = eng.legal_moves(state)
        if eng.terminal(state, moves) is not None:
            break
        mover = state.side_to_move
        resting_squares = {square for square, _ in state.rest[mover]}
        lifted += any(decode_move(m)[0] in resting_squares and decode_move(m)[2] != 4 for m in moves)
        state = play(eng, state, rng.choice(moves))
    return lifted


@pytest.mark.parametrize("name", SPECS)
def test_the_core_and_the_oracle_agree_on_random_games(name: str) -> None:
    eng = engine(load(name))
    for seed in range(4):
        walk(eng, seed, 80)


@pytest.mark.slow
@pytest.mark.parametrize("name", SPECS)
def test_the_core_and_the_oracle_agree_on_many_random_games(name: str) -> None:
    eng = engine(load(name))
    for seed in range(100, 140):
        walk(eng, seed, 200)


def oracle_perft(eng: Engine, state: GameState, depth: int) -> int:
    moves = legal(eng, state)
    if depth == 1:
        return len(moves)
    total = 0
    for word in moves:
        after = eng.apply(state.with_(rest=state.rest), word)
        lists = after_turn(
            eng, lists_of(state), after.board, state.side_to_move,
            end_of_move(eng, state, after, word),
        )
        total += oracle_perft(eng, after.with_(rest=as_rest(lists)), depth - 1)
    return total


# Frozen after `oracle_perft` agreed at every depth (Q983a). Without their rests the two are
# 7, 53, 506 and 7, 53, 510: the rests bite at the third ply, where a knight that moved on the
# first would move again.
PERFT = {
    "_rest_checkmate": (7, 53, 466),
    "_rest_drops": (7, 53, 470),
}


@pytest.mark.parametrize("name", SPECS)
def test_the_perft_agrees_with_the_oracle_and_is_frozen(name: str) -> None:
    eng = engine(load(name))
    state = eng.initial_state()
    counted = tuple(eng.perft(state, depth) for depth in (1, 2, 3))
    assert counted == tuple(oracle_perft(eng, state, depth) for depth in (1, 2, 3))
    assert counted == PERFT[name]


# --------------------------------------------- Fairy-Stockfish and the rest-stripped twin


def test_fairy_stockfish_refuses_a_rest_by_name_and_its_twin_translates() -> None:
    """§5.5: Fairy-Stockfish has no per-piece wait, so a rest game is refused with its rule,
    read off the compiled table; the rest only filters the generator, so the generator is
    checked through the twin with every rest removed, which translates."""
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    spec = load("_rest_checkmate")
    with pytest.raises(Untranslatable) as refused:
        translate(spec)
    assert any(reason.startswith("pieces[].rest") for reason in refused.value.reasons)
    twin = copy.deepcopy(spec)
    for piece in twin["pieces"]:
        piece.pop("rest", None)
    translate(twin)
