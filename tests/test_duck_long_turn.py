"""SR8c (`docs/plan-schema-next.md` §2.2 row L7, box SR8c): the duck beside `rules.turn`.

The ruling (2026-10-27, as recommended): *one duck move at the end of the turn*, for any
`rules.turn`, `moves: 1` included. The compiler makes a duck game's turn its piece moves plus
the duck's (`compile._compile_duck_turn`), so `remaining` 1 still owes the duck half
(`Game::duck_owed`) and a turn of one piece move compiles to the two halves it always did.

Readings settled here (the row left them unstated), on the duck's own principle that its half
is the turn's end:

- *The wild phase runs once a turn, after the duck half*, as in a duck game of one piece move
  (never after a piece move, mid-turn ones included); a pouncer's schedule alternates by turns.
- *A rest is counted down on the turn's last piece move*, the duck's half writing none
  (`Game::rest_turn_ends`), and a piece that moved earlier in the turn is not offered again
  this turn (SR7d's reading, unchanged).
- *A straight-back bar* set by a piece move with more piece moves to follow binds the mover's
  next move; set by the turn's last piece move it waits through the duck half for the other
  player (`Game::bar_reader`).
- *The en passant window* is opened by the mover's last double step of the turn and open for
  the opponent's first piece move only; the duck half leaves it (SR7a's reading, unchanged).
- *`max_plies`* may stop a game between two piece moves, as a long turn without the duck may,
  never between the turn's last piece move and its duck move (the validator).
- *The searches' `duck_mobility`* reads the count of the node one ply up, the turn's last
  piece move (its start only where the turn is one piece move).

The oracles: an `apply`-built count against `perft` (the core's make and unmake) from the
opening and from mid-turn; `tests/rest_oracle.py` (widened: `ends_turn` reads the duck's half)
for rest; the incremental key against `compute_hash` at every ply. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD, TurnRule, compile_spec, piece_turn
from chess_variant_sandbox.engine.moves import MOVE_DUCK, decode_move, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests import rest_oracle
from tests.test_duck_beside_portals import GAMES
from tests.test_duck_beside_portals import _game as _pond

HERE = Path(__file__).parent
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]
ROOK = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]
KNIGHT = [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]
PAWN = [
    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
     "condition": {"from_rank": 1}},
    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
]
TURNS: dict[str, dict[str, Any]] = {
    "two": {"moves": 2},
    "two-after-one": {"moves": 2, "ramp": [1]},
    "three": {"moves": 3},
    "ramp-of-two": {"moves": 1, "ramp": [2, 2]},
}


def _pen(turn: dict[str, Any] | None = None, **rules: Any) -> dict[str, Any]:
    """5x5, won by taking the king: a king, a rook, a knight and two pawns a side (en passant
    on), the duck starting on c3."""
    spec: dict[str, Any] = {
        "name": "the duck's pen (SR8c test spec)",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": KING},
            {"id": "R", "value": 5, "moves": ROOK},
            {"id": "N", "value": 3, "moves": KNIGHT},
            {"id": "P", "value": 1, "moves": PAWN},
            {"id": "D", "value": 0, "moves": []},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "player": 0, "squares": ["c1"]},
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "N", "player": 0, "squares": ["e1"]},
            {"piece": "P", "player": 0, "squares": ["b2", "d2"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
                  "en_passant": True, "promotion": {"piece": "P", "rank": "last", "to": ["R"]},
                  "duck": {"piece": "D", "start": "c3"}, **rules},
    }
    if turn is not None:
        spec["rules"]["turn"] = copy.deepcopy(turn)
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _plies_per_turn(turn: dict[str, Any], index: int) -> int:
    ramp = turn.get("ramp") or []
    return (ramp[index] if index < len(ramp) else turn.get("moves", 1)) + 1


# ------------------------------------------------------------------------- the lift


class TestTheLift:
    @pytest.mark.parametrize("turn", [{"moves": 1}, *TURNS.values()], ids=["one", *TURNS])
    def test_a_duck_validates_beside_any_turn(self, turn: dict[str, Any]) -> None:
        assert validate_spec(_pen(turn)) == []

    def test_a_turn_of_one_move_is_the_duck_game_it_was(self) -> None:
        plain, one = _pen(), _pen({"moves": 1})
        assert spec_hash(plain) == spec_hash(one)
        assert compile_spec(one).turn == compile_spec(plain).turn == TurnRule(moves=2, lengths=())
        assert compile_spec(one).zobrist_turn == compile_spec(plain).zobrist_turn

    def test_the_compiled_turn_ends_in_the_ducks_half(self) -> None:
        compiled = compile_spec(_pen({"moves": 2, "ramp": [1, 3]}))
        assert compiled.turn == TurnRule(moves=3, lengths=(2, 4))
        assert piece_turn(compiled.turn, True) == TurnRule(moves=2, lengths=(1, 3))
        assert piece_turn(TurnRule(moves=2, lengths=()), True) is None

    def test_a_turn_of_four_piece_moves_draws_one_count_more(self) -> None:
        """Four piece moves and the duck's is a count of five, one past the drawn counts: drawn
        last, so every other key is the game's without it."""
        four, three = compile_spec(_pen({"moves": 4})), compile_spec(_pen({"moves": 3}))
        assert len(four.zobrist_turn) == len(three.zobrist_turn) + 1
        assert len(set(four.zobrist_turn)) == len(four.zobrist_turn)

    def test_fairy_stockfish_refuses_the_pair(self) -> None:
        from chess_variant_sandbox.fairy import why_not

        reasons = why_not(_pen({"moves": 2}))
        assert any("rules.duck beside rules.turn" in reason for reason in reasons)


class TestTheClock:
    def test_a_clock_between_a_move_and_its_duck_move_is_refused(self) -> None:
        # Three plies a turn: 401 would stop the game on the duck half (402 ends a turn).
        assert validate_spec(_pen({"moves": 2}, max_plies=400)) == []
        assert any(
            "between a move and its duck move" in error
            for error in validate_spec(_pen({"moves": 2}, max_plies=401))
        )
        # The opening's turn of one: plies 2, then 3 a turn (5, 8, ...); 7 owes a duck move.
        ramped = {"moves": 2, "ramp": [1]}
        assert validate_spec(_pen(ramped, max_plies=6)) == []
        assert validate_spec(_pen(ramped, max_plies=7)) != []

    def test_a_turn_of_one_still_wants_an_even_clock(self) -> None:
        assert any("is odd" in error for error in validate_spec(_pen({"moves": 1}, max_plies=121)))


# ------------------------------------------------------------------------- the turn


def _walk(engine: Engine, turn: dict[str, Any], seed: int, plies: int) -> GameState:
    """A seeded walk: the duck owed exactly at each turn's last ply, the side flipping only
    after it, and the incremental key the full scan's at every ply."""
    rng = random.Random(seed)
    state = engine.initial_state()
    turn_index, into = 0, 0
    for _ in range(plies):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        length = _plies_per_turn(turn, turn_index)
        owed = into == length - 1
        assert engine.duck_owed(state) is owed
        assert all((decode_move(m)[2] == MOVE_DUCK) is owed for m in moves)
        assert state.remaining == length - into
        mover = state.side_to_move
        state = engine.apply(state, rng.choice(moves))
        assert state.key == engine._compute_hash(state)
        into += 1
        if into == length:
            turn_index, into = turn_index + 1, 0
            assert state.side_to_move == 1 - mover
        else:
            assert state.side_to_move == mover
    return state


@pytest.mark.parametrize("turn", list(TURNS.values()), ids=list(TURNS))
def test_one_duck_move_ends_every_turn(turn: dict[str, Any]) -> None:
    engine = _engine(_pen(turn))
    for seed in range(4):
        _walk(engine, turn, seed, 80)


def test_a_turn_of_four_piece_moves_keys_its_fifth_count() -> None:
    turn = {"moves": 4}
    engine = _engine(_pen(turn))
    for seed in range(2):
        _walk(engine, turn, seed, 40)


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _apply_count(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state)
    )


@pytest.mark.parametrize("turn", [{"moves": 2}, {"moves": 2, "ramp": [1]}], ids=["two", "ramped"])
def test_perft_matches_an_apply_built_count(turn: dict[str, Any]) -> None:
    """`perft` makes and unmakes; `apply` builds each position afresh. From the opening and
    from mid-turn, through the duck half and the turn's change."""
    engine = _engine(_pen(turn))
    state = engine.initial_state()
    assert engine.perft(state, 3) == _apply_count(engine, state, 3)
    state = engine.apply(state, engine.legal_moves(state)[0])
    assert engine.perft(state, 3) == _apply_count(engine, state, 3)


def test_a_searching_game_replays() -> None:
    """The searching players make and unmake thousands of times a move, the duck repairs
    reading a duck half one ply below a mid-turn piece move."""
    engine = _engine(_pen({"moves": 2, "ramp": [1]}))
    record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), 0)
    state = engine.initial_state()
    for move in record.moves:
        assert move in engine.legal_moves(state)
        state = engine.apply(state, move)
    outcome = engine.terminal(state)
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (record.winner, record.reason)


# ------------------------------------------------------------------------- what reads a turn


def test_en_passant_answers_the_last_double_step_of_the_turn() -> None:
    """The first player double-steps on its first piece move, then moves the knight: the window
    stays open through the second piece move and the duck half, for the opponent's first."""
    engine = _engine(_pen({"moves": 2}))
    state = engine.make_state(
        {"c1": ("K", 0), "e1": ("N", 0), "b2": ("P", 0), "c5": ("K", 1), "c4": ("P", 1),
         "e5": ("D", 2)},
        moved=["c4"],
    )
    state = state.with_(remaining=3, key=0)
    state = state.with_(key=engine._compute_hash(state), repetition=())
    b2, b4 = engine._square_index("b2"), engine._square_index("b4")
    state = engine.apply(state, next(
        m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == (b2, b4)
    ))
    assert state.ep_target == engine._square_index("b3")
    knight = next(m for m in engine.legal_moves(state)
                  if move_from(m) == engine._square_index("e1"))
    state = engine.apply(state, knight)
    assert engine.duck_owed(state) and state.ep_target == engine._square_index("b3")
    state = engine.apply(state, engine.legal_moves(state)[0])
    assert state.side_to_move == 1 and state.ep_target == engine._square_index("b3")
    takes = [m for m in engine.legal_moves(state)
             if move_from(m) == engine._square_index("c4") and move_to(m) == engine._square_index("b3")]
    assert takes
    # Its first piece move made, the window is gone for its second.
    state = engine.apply(state, next(m for m in engine.legal_moves(state)
                                     if move_from(m) == engine._square_index("c5")))
    assert state.ep_target < 0


def test_a_bar_binds_the_movers_next_piece_move_mid_turn() -> None:
    """Two piece moves a turn: the first steps the momentum end b3 to b2, and the mover's own
    second move may not take it straight back; the turn's last piece move sets a bar the other
    player meets after the duck half (`tests/test_duck_beside_portals.py`, a turn of one)."""
    spec = _pond(GAMES["momentum"])
    spec["rules"]["turn"] = {"moves": 2}
    engine = _engine(spec)
    state = engine.initial_state()
    b2, b3 = engine._square_index("b2"), engine._square_index("b3")
    state = engine.apply(state, next(
        m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == (b3, b2)
    ))
    assert state.side_to_move == 0 and not engine.duck_owed(state)
    assert state.key == engine._compute_hash(state)
    again = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
    assert (b2, b3) not in again and (b2, engine._square_index("c2")) in again
    # The end steps on (b2 to c2), the turn's last piece move: the bar waits through the duck.
    c2 = engine._square_index("c2")
    state = engine.apply(state, next(
        m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == (b2, c2)
    ))
    assert engine.duck_owed(state)
    state = engine.apply(state, engine.legal_moves(state)[0])
    assert state.side_to_move == 1 and state.key == engine._compute_hash(state)
    back = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
    assert (c2, b2) not in back and (c2, engine._square_index("d2")) in back
    assert state.board[c2] and engine.spec.owner_of(state.board[c2]) == WILD


def _rest_step(engine: Engine, state: GameState, move: int) -> GameState:
    assert sorted(engine.legal_moves(state)) == sorted(rest_oracle.legal(engine, state))
    after, lists = rest_oracle.apply(engine, state, move)
    assert after.rest == rest_oracle.as_rest(lists)
    assert after.key == engine._compute_hash(after)
    return after


@pytest.mark.parametrize("turn", [{"moves": 2}, {"moves": 3, "ramp": [1]}], ids=["two", "three"])
def test_rest_agrees_with_the_oracle(turn: dict[str, Any]) -> None:
    """A rook and a knight resting a turn: a piece that moved this turn is not offered again
    this turn, and the count runs down on the turn's last piece move, never on the duck's."""
    spec = _pen(turn)
    for piece in spec["pieces"]:
        if piece["id"] in ("R", "N"):
            piece["rest"] = 1
    engine = _engine(spec)
    barred = 0
    for seed in range(3):
        rng = random.Random(seed)
        state = engine.initial_state()
        moved: set[int] = set()
        for _ in range(60):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            mover = state.side_to_move
            if moved & {square for square, _ in state.rest[mover]}:
                barred += 1
            move = rng.choice(moves)
            after = _rest_step(engine, state, move)
            if after.side_to_move != mover:
                moved = set()
            elif decode_move(move)[2] != MOVE_DUCK:
                moved |= {decode_move(move)[1]}
            state = after
    assert barred > 0


@pytest.mark.parametrize("turn", [{"moves": 2}, {"moves": 3, "ramp": [1]}], ids=["two", "three"])
def test_the_wild_phase_runs_once_a_turn_after_the_duck_half(turn: dict[str, Any]) -> None:
    """The shepherd's duck (`tests/_duck_shepherd.json`) in a long turn: no wild piece moves on
    a piece move, mid-turn ones included; the wolves step on every duck half."""
    spec = json.loads((HERE / "_duck_shepherd.json").read_text(encoding="utf-8"))
    spec["rules"]["turn"] = copy.deepcopy(turn)
    engine = _engine(spec)
    rng = random.Random(0)
    state = engine.initial_state()
    stepped = 0
    for _ in range(40):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        owed = engine.duck_owed(state)
        state, phase = engine.apply_with_phase(state, rng.choice(moves))
        assert state.key == engine._compute_hash(state)
        if owed:
            stepped += len(phase)
        else:
            assert not list(phase)
    assert stepped > 0
