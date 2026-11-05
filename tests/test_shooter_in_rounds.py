"""SR7e (R11, ruled 2026-10-27 as recommended): a shooter in rounds.

`docs/plan-schema-next.md` §2.3 row R11 and `docs/reference/rounds.md` §4 step 0. Under
`rules.strike` placed `"stay"` (rifle, or a strike that transforms its victim in place) a
capture in a round is a **shot**: it does not move the shooter, so the shooter never
collides, and the shots resolve before the round's moves. Each shot strikes the piece
standing on its target on the round's board; a piece a shot strikes does not make its own
move (`EVENT_LOST`); two shots at each other both fire. In step 5 a shooter nothing struck
or took becomes what its move made it (a promotion, a side switch), as `make`'s rifle does.
In an ordered round (schema 28) the moves keep their order, and a slower shot whose victim
the faster move took away is void, never a move.

Every round here is checked against the declared oracle (`tests/simultaneous_oracle.py`),
and `tests/test_simultaneous_oracle.py` walks the four `_simultaneous_*` specs SR7e added.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PROMOTION, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

HERE = Path(__file__).parent
EVENT_COLLISION, EVENT_VOID, EVENT_SWAP, EVENT_ORDER, EVENT_LOST = 0, 1, 4, 5, 6


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


RIFLE = load("rifle")
DEMOTING = load("demoting_shot")
SPEED_RIFLE = load("speed_rifle")


def engine_of(spec: dict[str, Any]) -> Engine:
    return Engine(spec)


def place(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    state = engine.make_state(pieces)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def find(engine: Engine, state: GameState, player: int, origin: str, target: str, promotion: int | None = None) -> int:
    o_want, t_want = engine._square_index(origin), engine._square_index(target)
    for move in engine.legal_moves_for(state, player):
        o, t, kind, p = decode_move(move)
        if o == o_want and t == t_want and (promotion is None or (kind == MOVE_PROMOTION and p == promotion)):
            return move
    raise AssertionError(f"player {player} has no move {origin}->{target}")


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


KINGS = {"e1": ("K", 0), "e5": ("K", 1)}


class TestTheLift:
    def test_a_strike_placed_stay_is_no_longer_refused_beside_rounds(self) -> None:
        assert validate_spec(RIFLE) == []
        assert "R11" not in refused_ids(validate_spec(RIFLE))
        assert validate_spec(DEMOTING) == []
        assert validate_spec(SPEED_RIFLE) == []

    def test_a_contact_that_stays_keeps_its_own_reading(self) -> None:
        """The shot is `rules.strike`'s: the core's flag is derived from the strike's
        collision switch, which a contact table never sets."""
        engine = engine_of(RIFLE)
        assert engine.spec.simultaneous is not None
        assert engine.spec.simultaneous.collision_strikes
        assert engine.spec.strike is not None and engine.spec.strike.position == "stay"


class TestAShot:
    def test_a_shot_strikes_a_piece_that_moves_away(self) -> None:
        """The victim is struck where it stood before it could move; the shooter stays."""
        engine = engine_of(RIFLE)
        state = place(engine, {**KINGS, "a1": ("R", 0), "a4": ("N", 1)})
        shot = find(engine, state, 0, "a1", "a4")
        flight = find(engine, state, 1, "a4", "c3")
        after, events = round_of(engine, state, shot, flight)
        assert at(engine, after, "a1") == ("R", 0)
        assert at(engine, after, "a4") is None
        assert at(engine, after, "c3") is None
        assert (EVENT_LOST, engine._square_index("a4"), 1) in events

    def test_a_shooter_shot_at_from_elsewhere_still_fires(self) -> None:
        """The knight's capture of the rook on b2 is a shot too: both fire on the round's
        board, so the knight is struck and the rook on b2 is struck as well."""
        engine = engine_of(RIFLE)
        state = place(engine, {**KINGS, "a1": ("R", 0), "a4": ("N", 1), "b2": ("R", 0)})
        shot = find(engine, state, 0, "a1", "a4")
        back = find(engine, state, 1, "a4", "b2")
        after, events = round_of(engine, state, shot, back)
        assert at(engine, after, "a1") == ("R", 0)
        assert at(engine, after, "a4") is None
        assert at(engine, after, "b2") is None
        assert not any(event[0] == EVENT_LOST for event in events)

    def test_two_shots_at_each_other_both_fire(self) -> None:
        engine = engine_of(RIFLE)
        state = place(engine, {**KINGS, "a1": ("R", 0), "a4": ("R", 1)})
        first = find(engine, state, 0, "a1", "a4")
        second = find(engine, state, 1, "a4", "a1")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "a1") is None
        assert at(engine, after, "a4") is None
        assert not any(event[0] in (EVENT_COLLISION, EVENT_SWAP, EVENT_LOST) for event in events)

    def test_a_shooter_never_collides_even_under_swap_collision(self) -> None:
        """R11's case: only the queen shoots (`by`), and the rook it shoots takes the queen
        by an ordinary capture. Read before SR7e, the rook's flight let the queen chase onto
        a4 and the two swapped, which `swap: "collision"` makes a collision. Now the shot
        strikes the rook before it moves, and the queen never left a1."""
        spec = copy.deepcopy(RIFLE)
        spec["rules"]["simultaneous"] = {"swap": "collision"}
        spec["rules"]["strike"] = {"position": "stay", "by": ["Q"], "immune": ["K"]}
        assert validate_spec(spec) == []
        engine = engine_of(spec)
        state = place(engine, {**KINGS, "a1": ("Q", 0), "a4": ("R", 1)})
        shot = find(engine, state, 0, "a1", "a4")
        take = find(engine, state, 1, "a4", "a1")
        after, events = round_of(engine, state, shot, take)
        assert at(engine, after, "a1") == ("Q", 0)
        assert at(engine, after, "a4") is None
        assert (EVENT_LOST, engine._square_index("a4"), 1) in events
        assert not any(event[0] in (EVENT_COLLISION, EVENT_SWAP) for event in events)

    def test_a_quiet_move_from_a_struck_square_is_lost(self) -> None:
        engine = engine_of(RIFLE)
        state = place(engine, {**KINGS, "a1": ("R", 0), "c3": ("B", 1)})
        quiet = find(engine, state, 0, "a1", "a5")
        shot = find(engine, state, 1, "c3", "a1")
        after, events = round_of(engine, state, quiet, shot)
        assert at(engine, after, "a1") is None
        assert at(engine, after, "a5") is None
        assert at(engine, after, "c3") == ("B", 1)
        assert (EVENT_LOST, engine._square_index("a1"), 0) in events

    def test_a_promoting_shot_promotes_the_shooter_where_it_stands(self) -> None:
        engine = engine_of(RIFLE)
        state = place(engine, {"e1": ("K", 0), "a5": ("K", 1), "b4": ("P", 0), "c5": ("N", 1)})
        queen = engine.spec.piece_ids.index("Q")
        shot = find(engine, state, 0, "b4", "c5", promotion=queen)
        wait = find(engine, state, 1, "a5", "a4")
        after, _ = round_of(engine, state, shot, wait)
        assert at(engine, after, "b4") == ("Q", 0)
        assert at(engine, after, "c5") is None


class TestAShotThatTransforms:
    def test_the_victim_is_demoted_in_place_and_its_move_lost(self) -> None:
        engine = engine_of(DEMOTING)
        state = place(engine, {**KINGS, "a1": ("R", 0), "a4": ("Q", 1)})
        shot = find(engine, state, 0, "a1", "a4")
        flight = find(engine, state, 1, "a4", "d4")
        after, events = round_of(engine, state, shot, flight)
        assert at(engine, after, "a1") == ("R", 0)
        assert at(engine, after, "a4") == ("R", 1)
        assert at(engine, after, "d4") is None
        assert (EVENT_LOST, engine._square_index("a4"), 1) in events

    def test_a_switching_shooter_changes_sides_where_it_stands(self) -> None:
        engine = engine_of(DEMOTING)
        state = place(engine, {**KINGS, "b1": ("N", 0), "c3": ("R", 1)})
        shot = find(engine, state, 0, "b1", "c3")
        wait = find(engine, state, 1, "e5", "d5")
        after, _ = round_of(engine, state, shot, wait)
        assert at(engine, after, "b1") == ("N", 1)
        assert at(engine, after, "c3") == ("B", 1)

    def test_a_struck_shooter_does_not_switch(self) -> None:
        engine = engine_of(DEMOTING)
        state = place(engine, {**KINGS, "b1": ("N", 0), "c3": ("N", 1)})
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "c3", "b1")
        after, _ = round_of(engine, state, first, second)
        # Each knight is demoted to a pawn by the other's shot and switches nobody.
        assert at(engine, after, "b1") == ("P", 0)
        assert at(engine, after, "c3") == ("P", 1)


class TestAShotInAnOrderedRound:
    def test_a_faster_shot_loses_the_slower_move_of_its_victim(self) -> None:
        engine = engine_of(SPEED_RIFLE)
        state = place(engine, {**KINGS, "b1": ("N", 0), "c3": ("Q", 1)})
        shot = find(engine, state, 0, "b1", "c3")
        flight = find(engine, state, 1, "c3", "c4")
        after, events = round_of(engine, state, shot, flight)
        assert at(engine, after, "b1") == ("N", 0)
        assert at(engine, after, "c3") is None
        assert at(engine, after, "c4") is None
        assert (EVENT_LOST, engine._square_index("c3"), 1) in events

    def test_a_slower_shot_whose_victim_fled_is_void_and_never_a_move(self) -> None:
        engine = engine_of(SPEED_RIFLE)
        state = place(engine, {**KINGS, "a1": ("R", 0), "a3": ("N", 1)})
        shot = find(engine, state, 0, "a1", "a3")
        flight = find(engine, state, 1, "a3", "b5")
        after, events = round_of(engine, state, shot, flight)
        assert at(engine, after, "a1") == ("R", 0)
        assert at(engine, after, "a3") is None
        assert at(engine, after, "b5") == ("N", 1)
        assert (EVENT_ORDER, engine._square_index("b5"), 1) in events
        assert (EVENT_VOID, engine._square_index("a1"), 0) in events
