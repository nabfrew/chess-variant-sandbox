"""SR8d (R6, ruled 2026-10-27): castling in rounds.

`docs/plan-schema-next.md` §2.3 row R6 and `docs/reference/rounds.md` §4 step 4. A castling
is the king's move for the round's meetings: only the king's arrival collides, chases or is
taken. After the meetings the partner moves to its castled square if that square is empty,
and otherwise stays where it is; until then it is a standing piece, which the other player
may take on its square. Readings SR8d settled (the row left them unstated): the partner moves
whatever became of the king (the castling was made; only the king's arrival met anything);
a partner a file-based king lands on is lifted with the departures and goes to its castled
square if empty, else it is removed, nobody's capture (its own square holds the king: the one
arrival of the other player that fills the castled square cannot also meet the king); the
castling spends both pieces' rights, as `make`'s does.

Every round here is checked against the declared oracle (`tests/simultaneous_oracle.py`), and
`tests/test_simultaneous_oracle.py` walks the two `_simultaneous_castling*` specs.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

HERE = Path(__file__).parent
EVENT_COLLISION = 0


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


STEPS = load("castling")
FILES = load("castling_files")


def place(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    state = engine.make_state(pieces)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def move(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    o_want, t_want = engine._square_index(origin), engine._square_index(target)
    for m in engine.legal_moves_for(state, player):
        o, t, kind, _ = decode_move(m)
        if o == o_want and t == t_want and kind != MOVE_CASTLE:
            return m
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def castle(engine: Engine, state: GameState, player: int, target: str) -> int:
    want = engine._square_index(target)
    for m in engine.legal_moves_for(state, player):
        _, t, kind, _ = decode_move(m)
        if kind == MOVE_CASTLE and t == want:
            return m
    raise AssertionError(f"player {player} cannot castle to {target}")


def round_of(engine: Engine, state: GameState, first: int, second: int):
    after, _, events = engine.apply_round_with_phase(state, first, second)
    assert after == resolve_round(engine, state, first, second)
    assert after.key == engine._compute_hash(after)
    return after, events


def at(engine: Engine, state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[engine._square_index(name)]
    if value == 0:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def unmoved(engine: Engine, state: GameState, name: str) -> bool:
    return bool(state.unmoved >> engine._square_index(name) & 1)


# Player 0 castles by steps: king d1, rooks a1 and g1; kingside the king goes to f1 and
# the rook to e1. Player 1's king stands far off, with a rook on the second rank to reach.
CASTLER = {"a1": ("R", 0), "d1": ("K", 0), "g1": ("R", 0), "a6": ("K", 1)}


class TestTheLift:
    def test_castling_is_no_longer_refused_beside_rounds(self) -> None:
        assert validate_spec(STEPS) == []
        assert validate_spec(FILES) == []
        assert "R6" not in refused_ids(validate_spec(STEPS))

    def test_a_spy_game_may_castle(self) -> None:
        spec = json.loads((HERE / "_spy_small.json").read_text(encoding="utf-8"))
        spec["rules"]["castling"] = {"king": "K", "rook": "R", "king_steps": 2}
        assert "R6" not in refused_ids(validate_spec(spec))

    def test_each_player_is_offered_castling_on_the_round_board(self) -> None:
        engine = Engine(STEPS)
        state = place(engine, {**CASTLER, "a6": ("R", 1), "d6": ("K", 1), "g6": ("R", 1)})
        for player, rank in ((0, "1"), (1, "6")):
            targets = {decode_move(m)[1] for m in engine.legal_moves_for(state, player)
                       if decode_move(m)[2] == MOVE_CASTLE}
            assert targets == {engine._square_index(f"b{rank}"), engine._square_index(f"f{rank}")}


class TestTheKingAndItsPartner:
    def test_an_unmet_castling_is_the_castling(self) -> None:
        engine = Engine(STEPS)
        state = place(engine, CASTLER)
        after, events = round_of(engine, state, castle(engine, state, 0, "f1"),
                                 move(engine, state, 1, "a6", "b6"))
        assert at(engine, after, "f1") == ("K", 0) and at(engine, after, "e1") == ("R", 0)
        assert at(engine, after, "d1") is None and at(engine, after, "g1") is None
        assert not unmoved(engine, after, "g1") and not unmoved(engine, after, "e1")
        assert unmoved(engine, after, "a1")
        assert not events

    def test_the_king_s_arrival_collides_and_the_rook_still_moves(self) -> None:
        """A rook of player 1 lands on f1 as the king does: a *yes/yes* tie under the
        default, both go; the castled square e1 is empty, so the partner goes there."""
        engine = Engine(STEPS)
        state = place(engine, {**CASTLER, "f4": ("R", 1)})
        after, events = round_of(engine, state, castle(engine, state, 0, "f1"),
                                 move(engine, state, 1, "f4", "f1"))
        assert at(engine, after, "f1") is None
        assert at(engine, after, "e1") == ("R", 0) and at(engine, after, "g1") is None
        assert (EVENT_COLLISION, engine._square_index("f1"), -1) in events
        assert engine.terminal(after) is not None

    def test_a_collision_won_by_the_initiative_keeps_the_king(self) -> None:
        spec = copy.deepcopy(STEPS)
        spec["rules"]["simultaneous"] = {"tie": "initiative"}
        engine = Engine(spec)
        state = place(engine, {**CASTLER, "f4": ("R", 1)})
        after, _ = round_of(engine, state, castle(engine, state, 0, "f1"),
                            move(engine, state, 1, "f4", "f1"))
        # The initiative starts with player 1, whose rook wins the tie.
        assert at(engine, after, "f1") == ("R", 1)
        assert at(engine, after, "e1") == ("R", 0)

    def test_the_partner_stays_where_the_castled_square_is_filled(self) -> None:
        """Player 1's rook arrives on e1, the square the king crossed: the king lands on f1,
        the partner finds e1 taken and stays on g1."""
        engine = Engine(STEPS)
        state = place(engine, {**CASTLER, "e4": ("R", 1)})
        after, events = round_of(engine, state, castle(engine, state, 0, "f1"),
                                 move(engine, state, 1, "e4", "e1"))
        assert at(engine, after, "f1") == ("K", 0)
        assert at(engine, after, "e1") == ("R", 1)
        assert at(engine, after, "g1") == ("R", 0)
        assert not unmoved(engine, after, "g1")
        assert not events

    def test_the_partner_is_a_standing_piece_through_the_meetings(self) -> None:
        """Player 1 takes the rook on g1 as it stands: the king castles, the rook is gone."""
        engine = Engine(STEPS)
        state = place(engine, {**CASTLER, "g4": ("R", 1)})
        after, _ = round_of(engine, state, castle(engine, state, 0, "f1"),
                            move(engine, state, 1, "g4", "g1"))
        assert at(engine, after, "f1") == ("K", 0)
        assert at(engine, after, "g1") == ("R", 1)
        assert at(engine, after, "e1") is None

    def test_a_chase_onto_the_king_s_square_lands_behind_it(self) -> None:
        """A capture of the king on d1 finds it gone: the rook lands there, as any chase."""
        engine = Engine(STEPS)
        state = place(engine, {**CASTLER, "d4": ("R", 1)})
        after, _ = round_of(engine, state, castle(engine, state, 0, "f1"),
                            move(engine, state, 1, "d4", "d1"))
        assert at(engine, after, "d1") == ("R", 1)
        assert at(engine, after, "f1") == ("K", 0) and at(engine, after, "e1") == ("R", 0)


# Castling by file: player 0's king on f1 between rooks on b1 and g1. Kingside the king goes
# to g1, its own rook's square, and the rook to f1, the king's.
BY_FILE = {"b1": ("R", 0), "f1": ("K", 0), "g1": ("R", 0), "a6": ("K", 1)}


class TestAKingThatLandsOnItsPartner:
    def test_the_two_trade_places(self) -> None:
        engine = Engine(FILES)
        state = place(engine, BY_FILE)
        after, _ = round_of(engine, state, castle(engine, state, 0, "g1"),
                            move(engine, state, 1, "a6", "b6"))
        assert at(engine, after, "g1") == ("K", 0) and at(engine, after, "f1") == ("R", 0)

    def test_an_arrival_on_the_partner_s_square_meets_the_king(self) -> None:
        """Player 1's rook takes on g1 as the king lands there: the king's arrival collides
        (both go), and the lifted rook goes to f1, the king's emptied square."""
        engine = Engine(FILES)
        state = place(engine, {**BY_FILE, "g4": ("R", 1)})
        after, events = round_of(engine, state, castle(engine, state, 0, "g1"),
                                 move(engine, state, 1, "g4", "g1"))
        assert (EVENT_COLLISION, engine._square_index("g1"), -1) in events
        assert at(engine, after, "g1") is None
        assert at(engine, after, "f1") == ("R", 0)

    def test_a_lifted_partner_with_nowhere_to_go_is_removed(self) -> None:
        """Player 1's rook chases the king onto f1, the castled square: the lifted rook finds
        it filled and its own square under the king, so it is removed."""
        engine = Engine(FILES)
        state = place(engine, {**BY_FILE, "f4": ("R", 1)})
        after, _ = round_of(engine, state, castle(engine, state, 0, "g1"),
                            move(engine, state, 1, "f4", "f1"))
        assert at(engine, after, "g1") == ("K", 0)
        assert at(engine, after, "f1") == ("R", 1)
        assert sum(1 for v in after.board if v and engine.spec.owner_of(v) == 0) == 2
        assert after.halfmove == 0


def test_random_rounds_castle_and_agree_with_the_oracle() -> None:
    """Both test specs, played at random from the start: castlings happen, met and unmet,
    and every round agrees with the oracle (the walk itself is `test_simultaneous_oracle`'s)."""
    for spec in (STEPS, FILES):
        engine = Engine(spec)
        castled = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(30):
                if engine.terminal(state) is not None:
                    break
                lists = [engine.legal_moves_for(state, p) for p in (0, 1)]
                castles = [[m for m in ms if decode_move(m)[2] == MOVE_CASTLE] for ms in lists]
                picks = [rng.choice(c) if c and rng.random() < 0.7 else rng.choice(ms)
                         for c, ms in zip(castles, lists, strict=True)]
                castled += sum(decode_move(m)[2] == MOVE_CASTLE for m in picks)
                after = engine.apply_round(state, *picks)
                assert after == resolve_round(engine, state, *picks)
                state = after
        assert castled > 0
