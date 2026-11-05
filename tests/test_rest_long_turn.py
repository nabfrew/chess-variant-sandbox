"""SR7d (`docs/plan-schema-next.md` §2.2 row L2, box SR7d): `pieces[].rest` beside a long turn.

The ruling (2026-10-27, as recommended): *a piece that moved this turn may not move again this
turn, then rests its `rest` turns*. The rest list is written per move and counted per turn: an
entry's count is its player's turns still to rest, the turn in progress counted, so a piece that
moves before its turn's last move enters at ``rest + 1`` and the turn's last move counts the
list down. In a turn of one move nothing differs.

Readings settled here on the plan's principles (rest is a fact about the piece, and the lift
keeps a player from ever being left without a move by the rest alone):

- *The lift reads each move's list*, as it reads each turn's in a turn of one move: a later move
  of a turn whose every offer is a resting piece's (the pieces that moved earlier in the turn
  among them) is lifted for that move, and the piece it moves rests afresh.
- *A piece that moves on the turn's last move* rests its `rest` turns, as in a turn of one.
- *A commanded piece* (SR5e) enters its owner's list at `rest`, counted on its owner's turns,
  and neither player may move it while it rests, the commander's later moves of the same turn
  included.
- *A drop rests the dropped piece* as any move does, so it is not moved again this turn.

The oracle is `tests/rest_oracle.py`, widened (`after_turn`'s ``ends_turn``), a declared test
oracle (rule 3). `unmake` is held through `perft` from mid-turn against an `apply`-built count
and through searching games replayed. Hand-built specs and committed rest specs, not chess.
"""

from __future__ import annotations

import copy
import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.nets import Layout
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_FOLD, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import REFUSALS
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests import rest_oracle
from tests.feature_inventory import _schema_ninety_nine_uses
from tests.test_command_beside_rest import _spec as _preachers

HERE = Path(__file__).parent
ROOT = HERE.parent
TURNS: dict[str, dict[str, Any]] = {
    "two-after-one": {"moves": 2, "ramp": [1]},
    "three": {"moves": 3},
    "ramp-of-two": {"moves": 1, "ramp": [2, 2]},
}


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    return loaded


def _long(spec: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    out["rules"]["turn"] = copy.deepcopy(turn)
    out["name"] = f"{out['name']} in a long turn"
    return out


def _weary(rests: dict[str, int], turn: dict[str, Any]) -> dict[str, Any]:
    """`_rest_checkmate` (Gardner under checkmate) with these rests on top of its own."""
    spec = _long(_load("_rest_checkmate"), turn)
    for piece in spec["pieces"]:
        if piece["id"] in rests:
            piece["rest"] = rests[piece["id"]]
    return spec


@cache
def _engine_of(text: str) -> Engine:
    return Engine(json.loads(text))


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return _engine_of(json.dumps(spec, sort_keys=True))


def _step(engine: Engine, state: GameState, move: int) -> GameState:
    """One move through the core, held to the oracle's list, its rest lists and the slow key."""
    assert sorted(engine.legal_moves(state)) == sorted(rest_oracle.legal(engine, state))
    after, lists = rest_oracle.apply(engine, state, move)
    assert after.rest == rest_oracle.as_rest(lists)
    assert after.key == engine._compute_hash(after)
    return after


def _of(engine: Engine, state: GameState, pid: str) -> list[int]:
    """The legal moves of the side to move's pieces of type ``pid``."""
    spec = engine.spec
    return [
        m for m in engine.legal_moves(state)
        if decode_move(m)[2] != MOVE_DROP
        and spec.piece_ids[spec.type_of(state.board[decode_move(m)[0]])] == pid
    ]


def _away(engine: Engine, state: GameState, square: int) -> int:
    """A legal move that neither starts nor ends on ``square``."""
    return next(m for m in engine.legal_moves(state) if square not in decode_move(m)[:2])


def _moved_from(engine: Engine, state: GameState, square: int) -> bool:
    return any(
        decode_move(m)[0] == square and decode_move(m)[2] not in (MOVE_DROP, MOVE_FOLD)
        for m in engine.legal_moves(state)
    )


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("turn", list(TURNS.values()), ids=list(TURNS))
    def test_rest_specs_validate_in_a_long_turn(self, turn: dict[str, Any]) -> None:
        for name in ("_rest_checkmate", "_rest_drops"):
            assert validate_spec(_long(_load(name), turn)) == []

    def test_no_refusal_names_the_pair(self) -> None:
        assert "L2" not in {refusal.id for refusal in REFUSALS}

    def test_the_inventory_names_the_pair(self) -> None:
        assert _schema_ninety_nine_uses(_weary({}, TURNS["three"])) == {"pieces.rest.turn"}
        assert _schema_ninety_nine_uses(_load("_rest_checkmate")) == set()
        plain = _long(_load("_rest_checkmate"), TURNS["three"])
        for piece in plain["pieces"]:
            piece.pop("rest", None)
        assert _schema_ninety_nine_uses(plain) == set()

    def test_describe_says_it(self) -> None:
        text = describe_spec(_weary({}, TURNS["two-after-one"]))
        assert "may not move again in that turn" in text
        assert "may not move again in that turn" not in describe_spec(_load("_rest_checkmate"))

    def test_the_keys_and_the_net_plane_take_one_layer_more(self) -> None:
        """An entry made mid-turn holds ``rest + 1``: the key and the net's plane are one
        layer wider in a long turn, and the core's plane is the oracle's."""
        one = compile_spec(_load("_rest_checkmate"))
        long = compile_spec(_weary({}, TURNS["two-after-one"]))
        assert (one.rest_layers, long.rest_layers) == (2, 3)
        assert len(long.zobrist_rest) == 3 * 2 * long.n_squares
        engine = _engine(_weary({}, TURNS["two-after-one"]))
        assert engine.core.net_inputs() == Layout.of(engine.spec).n_inputs
        assert Layout.of(engine.spec).rest_turns == 3


# ------------------------------------------------------------------ the ruled reading, by hand


class TestTheReading:
    def test_a_piece_that_moved_is_not_offered_again_this_turn_then_rests(self) -> None:
        """Two moves a turn after a first of one; the knight and the rook rest 1. White's
        first turn is one move: the knight goes and rests White's next turn whole (count 1).
        On that turn the rook moves first: it is in the list at 2 (this turn and the next),
        not offered on the turn's second move; the turn's end spends the knight's rest and
        leaves the rook's at 1."""
        engine = _engine(_weary({"N": 1, "R": 1}, TURNS["two-after-one"]))
        state = engine.initial_state()
        knight = _of(engine, state, "N")[0]
        landing = decode_move(knight)[1]
        state = _step(engine, state, knight)  # a turn of one: the count is `rest`
        assert state.side_to_move == 1 and state.rest[0] == ((landing, 1),)
        for _ in range(2):
            state = _step(engine, state, _away(engine, state, landing))
        assert state.side_to_move == 0 and state.remaining == 2
        assert state.rest[0] == ((landing, 1),) and not _moved_from(engine, state, landing)
        rooks = _of(engine, state, "R")
        assert rooks  # the knight's square, if nothing else
        rook = decode_move(rooks[0])[1]
        state = _step(engine, state, rooks[0])
        assert state.side_to_move == 0
        assert state.rest[0] == tuple(sorted(((landing, 1), (rook, 2))))
        assert not _moved_from(engine, state, landing) and not _moved_from(engine, state, rook)
        state = _step(engine, state, engine.legal_moves(state)[0])
        assert state.side_to_move == 1 and state.rest[0] == ((rook, 1),)

    def test_a_mid_turn_mover_counts_this_turn_and_its_rest(self) -> None:
        """Three moves a turn; the rook rests 1. A rook that moves first is in the list at 2
        for the rest of the turn (not offered), 1 after the turn ends, and free once its
        player's next turn ends."""
        engine = _engine(_weary({"R": 1, "N": 0, "Q": 0}, TURNS["three"]))
        rng = random.Random(3)
        state = engine.initial_state()
        for _ in range(200):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                state = engine.initial_state()
                continue
            first_of_turn = state.remaining == 3
            rook_moves = _of(engine, state, "R")
            if first_of_turn and rook_moves:
                move = rook_moves[0]
                side = state.side_to_move
                landing = decode_move(move)[1]
                state = _step(engine, state, move)
                if state.board[landing] and engine.spec.owner_of(state.board[landing]) == side:
                    assert (landing, 2) in state.rest[side]
                    assert not _moved_from(engine, state, landing) or rest_oracle.lifted(engine, state)
                    while state.side_to_move == side:
                        state = _step(engine, state, engine.legal_moves(state)[0])
                    if state.board[landing] and engine.spec.owner_of(state.board[landing]) == side:
                        assert (landing, 1) in state.rest[side]
                    return
            state = _step(engine, state, rng.choice(moves))
        pytest.fail("no rook moved first in a turn")

    def test_the_lift_fires_mid_turn(self) -> None:
        """A king and a knight against a king, two moves a turn, both resting 1, the king
        already resting: the knight's moves are the whole list. Once it has moved, every
        offer is a resting piece's (the king's, and the knight's of this turn), so the turn's
        second move is lifted."""
        spec = _weary({"N": 1, "K": 1}, {"moves": 2})
        engine = _engine(spec)
        state = engine.make_state(
            {"a1": ("K", 0), "c2": ("N", 0), "e5": ("K", 1)}, side_to_move=0,
            moved=("a1", "c2", "e5"),
        )
        # The king rests, so after the knight moves every offer is a resting piece's.
        state = state.with_(rest=(((engine._square_index("a1"), 2),), ()), remaining=2)
        state = state.with_(key=engine._compute_hash(state))
        knight_moves = rest_oracle.legal(engine, state)
        assert knight_moves and all(decode_move(m)[0] == engine._square_index("c2") for m in knight_moves)
        state = _step(engine, state, knight_moves[0])
        assert state.side_to_move == 0 and state.remaining == 1
        assert rest_oracle.lifted(engine, state)
        state = _step(engine, state, rest_oracle.legal(engine, state)[0])
        assert state.side_to_move == 1


# --------------------------------------------------------------- the core and the oracle


def _walk(engine: Engine, seed: int, plies: int) -> tuple[int, int]:
    """A seeded walk held to the oracle; returns how often a later move of a turn found a
    piece of that turn resting, and how often the lift fired."""
    rng = random.Random(seed)
    state = engine.initial_state()
    barred = lifted = 0
    moved: set[int] = set()
    for _ in range(plies):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        mover = state.side_to_move
        squares = {square for square, _ in state.rest[mover]}
        if moved & squares:
            barred += 1
        lifted += rest_oracle.lifted(engine, state)
        move = rng.choice(moves)
        after = _step(engine, state, move)
        moved = set() if after.side_to_move != mover else moved | {decode_move(move)[1]}
        state = after
    return barred, lifted


@pytest.mark.parametrize("turn", list(TURNS.values()), ids=list(TURNS))
@pytest.mark.parametrize("name", ["_rest_checkmate", "_rest_drops"])
def test_seeded_walks_agree_with_the_oracle(name: str, turn: dict[str, Any]) -> None:
    engine = _engine(_long(_load(name), turn))
    barred = sum(_walk(engine, seed, 60)[0] for seed in range(3))
    assert barred > 0


def test_the_longest_rests_in_the_longest_turn_fit_the_lists() -> None:
    """Every piece but the king resting 4, four moves a turn: the most entries a list can
    hold, against the oracle."""
    engine = _engine(_weary({"P": 4, "N": 4, "B": 4, "R": 4, "Q": 4}, {"moves": 4, "ramp": [1]}))
    for seed in range(2):
        _walk(engine, seed, 60)


@pytest.mark.parametrize("captures_own", [False, True], ids=["plain", "captures_own"])
def test_a_command_beside_rest_in_a_long_turn(captures_own: bool) -> None:
    spec = _long(_preachers(captures_own=captures_own), {"moves": 2, "ramp": [1]})
    engine = _engine(spec)
    for seed in range(3):
        _walk(engine, seed, 60)


@pytest.mark.slow
@pytest.mark.parametrize("turn", list(TURNS.values()), ids=list(TURNS))
@pytest.mark.parametrize("name", ["_rest_checkmate", "_rest_drops"])
def test_long_seeded_walks_agree_with_the_oracle(name: str, turn: dict[str, Any]) -> None:
    engine = _engine(_long(_load(name), turn))
    for seed in range(100, 120):
        _walk(engine, seed, 200)


def test_a_turn_of_one_is_unchanged() -> None:
    """`rules.turn` of one move compiles to no turn rule: the keys and lists are the game's
    without it."""
    spec = _long(_load("_rest_checkmate"), {"moves": 1})
    assert compile_spec(spec).zobrist_rest == compile_spec(_load("_rest_checkmate")).zobrist_rest


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _apply_count(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state)
    )


@pytest.mark.parametrize("name", ["_rest_checkmate", "_rest_drops"])
def test_perft_from_mid_turn_matches_an_apply_built_count(name: str) -> None:
    engine = _engine(_long(_load(name), {"moves": 2}))
    state = engine.initial_state()
    # A knight first, so a piece of this turn rests mid-turn.
    state = engine.apply(state, _of(engine, state, "N")[0])
    assert state.remaining == 1 and state.rest[0]
    assert engine.perft(state, 3) == _apply_count(engine, state, 3)


def test_a_searching_game_replays() -> None:
    """The searching players make and unmake thousands of times a move; a rest list their
    unmake left wrong would offer a move `apply` does not."""
    engine = _engine(_long(_load("_rest_checkmate"), {"moves": 2, "ramp": [1]}))
    record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), 0)
    state = engine.initial_state()
    for move in record.moves:
        assert move in engine.legal_moves(state)
        state = engine.apply(state, move)
    outcome = engine.terminal(state)
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
