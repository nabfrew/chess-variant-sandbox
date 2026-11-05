"""SR7f (R18, ruled 2026-10-27 as recommended): pieces nobody owns in rounds.

`docs/plan-schema-next.md` §2.3 row R18 and `docs/reference/rounds.md` §4 step 4. A piece
nobody owns that stands (a placement for nobody, no `control`) sits on the round's board as
any standing piece: either player may take it, and two players' moves onto its square are a
collision as any other, by `beats` and then the tie. The winner takes the standing piece
and the loser (two captures: the hand, the bounties); with no winner the standing piece goes
with the two, captured by nobody, as a wild piece on a collision square does.

Two moves that strike one nobody's piece without moving (two shots under `rules.strike`
placed `"stay"`, or two contacts that stay) meet the same way: by `beats` between the two
strikers and then the tie, the winner's strike made as if alone and the loser's missing
(its piece stands and changes nothing; it is not a void move, so no void run). With no
winner both miss and the piece stands. A shot and an arrival at one piece: the shot first
(step 0), and the arrival is made onto what it left, as chosen on the round's board.

A held piece (majority control) beside rounds was R18's narrower row until SR7n (R20,
`tests/test_held_in_rounds.py`).

Every round here is checked against the declared oracle (`tests/simultaneous_oracle.py`),
and `tests/test_simultaneous_oracle.py` walks the two `_simultaneous_nobody*` specs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

HERE = Path(__file__).parent
EVENT_COLLISION, EVENT_VOID, EVENT_INITIATIVE = 0, 1, 2
NOBODY = 2


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


STONES = load("nobody")
SHOT = load("nobody_rifle")
KINGS = {"c1": ("K", 0), "c5": ("K", 1)}


def place(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    state = engine.make_state(pieces).with_(hands=engine.initial_state().hands)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def find(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    o_want, t_want = engine._square_index(origin), engine._square_index(target)
    for move in engine.legal_moves_for(state, player):
        o, t, _, _ = decode_move(move)
        if o == o_want and t == t_want:
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


def held(engine: Engine, state: GameState, player: int, pid: str) -> int:
    drops = engine.spec.drops
    assert drops is not None
    return state.hands[player][drops.hand_slot(engine.spec.piece_ids.index(pid))]


class TestTheLift:
    def test_a_placement_for_nobody_is_no_longer_refused_beside_rounds(self) -> None:
        assert validate_spec(STONES) == []
        assert validate_spec(SHOT) == []

    def test_a_held_piece_beside_rounds_is_lifted_since_sr7n(self) -> None:
        """R18's narrower row (R20) went with SR7n: `tests/test_held_in_rounds.py`."""
        spec = copy.deepcopy(STONES)
        spec["pieces"][3]["control"] = "majority"
        assert "R18" not in refused_ids(validate_spec(spec))


class TestArrivalsOntoAStone:
    def test_one_arrival_takes_the_stone_into_its_hand(self) -> None:
        engine = Engine(STONES)
        state = place(engine, {**KINGS, "b2": ("P", 0), "c3": ("S", NOBODY), "e5": ("N", 1)})
        take = find(engine, state, 0, "b2", "c3")
        wait = find(engine, state, 1, "e5", "d3")
        after, _ = round_of(engine, state, take, wait)
        assert at(engine, after, "c3") == ("P", 0)
        assert held(engine, after, 0, "S") == 1

    def test_a_tie_by_default_leaves_the_square_empty_and_nobody_captured(self) -> None:
        """Two pawns (each beats the other): both go, and the stone goes with them, taken by
        nobody, so no hand gains a piece."""
        engine = Engine(STONES)
        state = place(engine, {**KINGS, "b2": ("P", 0), "d4": ("P", 1), "c3": ("S", NOBODY)})
        first = find(engine, state, 0, "b2", "c3")
        second = find(engine, state, 1, "d4", "c3")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "c3") is None
        assert at(engine, after, "b2") is None and at(engine, after, "d4") is None
        assert held(engine, after, 0, "S") == 0 and held(engine, after, 1, "S") == 0
        assert held(engine, after, 0, "P") == 1 and held(engine, after, 1, "P") == 1
        assert (EVENT_COLLISION, engine._square_index("c3"), -1) in events

    def test_the_winner_by_beats_takes_the_stone_and_the_loser(self) -> None:
        """The knight beats the pawn, which does not beat the knight."""
        engine = Engine(STONES)
        state = place(engine, {**KINGS, "b1": ("N", 0), "d4": ("P", 1), "c3": ("S", NOBODY)})
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "d4", "c3")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "c3") == ("N", 0)
        assert held(engine, after, 0, "S") == 1 and held(engine, after, 0, "P") == 1
        assert held(engine, after, 1, "S") == 0
        assert (EVENT_COLLISION, engine._square_index("c3"), 0) in events

    def test_the_initiative_decides_a_tie_onto_a_stone(self) -> None:
        spec = copy.deepcopy(STONES)
        spec["rules"]["simultaneous"] = {"tie": "initiative"}
        engine = Engine(spec)
        state = place(engine, {**KINGS, "b2": ("P", 0), "d4": ("P", 1), "c3": ("S", NOBODY)})
        holder = state.initiative
        first = find(engine, state, 0, "b2", "c3")
        second = find(engine, state, 1, "d4", "c3")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "c3") == ("P", holder)
        assert held(engine, after, holder, "S") == 1 and held(engine, after, 1 - holder, "S") == 0
        assert after.initiative == 1 - holder
        assert (EVENT_INITIATIVE, engine._square_index("c3"), holder) in events


class TestShotsAtAStone:
    def test_one_shot_strikes_the_stone(self) -> None:
        engine = Engine(SHOT)
        state = place(engine, {**KINGS, "a3": ("R", 0), "c3": ("S", NOBODY), "e5": ("N", 1)})
        shot = find(engine, state, 0, "a3", "c3")
        wait = find(engine, state, 1, "e5", "d3")
        after, _ = round_of(engine, state, shot, wait)
        assert at(engine, after, "a3") == ("R", 0)
        assert at(engine, after, "c3") is None
        assert held(engine, after, 0, "S") == 1

    def test_two_shots_by_beats_the_winner_strikes_and_the_loser_misses(self) -> None:
        """The knight beats the pawn: only the knight's shot strikes, and the stone goes to
        one hand, never two."""
        engine = Engine(SHOT)
        state = place(engine, {**KINGS, "b1": ("N", 0), "d4": ("P", 1), "c3": ("S", NOBODY)})
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "d4", "c3")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "b1") == ("N", 0) and at(engine, after, "d4") == ("P", 1)
        assert at(engine, after, "c3") is None
        assert held(engine, after, 0, "S") == 1 and held(engine, after, 1, "S") == 0
        assert (EVENT_VOID, engine._square_index("d4"), 1) in events
        assert after.initiative == state.initiative

    def test_two_tied_shots_go_to_the_initiative(self) -> None:
        engine = Engine(SHOT)
        state = place(engine, {**KINGS, "a3": ("R", 0), "e3": ("R", 1), "c3": ("S", NOBODY)})
        holder = state.initiative
        first = find(engine, state, 0, "a3", "c3")
        second = find(engine, state, 1, "e3", "c3")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "c3") is None
        assert held(engine, after, holder, "S") == 1 and held(engine, after, 1 - holder, "S") == 0
        assert after.initiative == 1 - holder
        assert (EVENT_INITIATIVE, engine._square_index("c3"), holder) in events

    def test_two_tied_shots_with_no_winner_both_miss(self) -> None:
        spec = copy.deepcopy(SHOT)
        spec["rules"]["simultaneous"] = {}
        engine = Engine(spec)
        state = place(engine, {**KINGS, "a3": ("R", 0), "e3": ("R", 1), "c3": ("S", NOBODY)})
        first = find(engine, state, 0, "a3", "c3")
        second = find(engine, state, 1, "e3", "c3")
        after, _ = round_of(engine, state, first, second)
        assert at(engine, after, "c3") == ("S", NOBODY)
        assert held(engine, after, 0, "S") == 0 and held(engine, after, 1, "S") == 0
        # Not a void round: the void run is a swap's.
        assert after.void_run == state.void_run

    def test_a_shot_first_then_the_arrival_onto_what_it_left(self) -> None:
        """Only rooks shoot (`by`): the rook's shot clears the stone before anything moves,
        and the pawn's capture is made onto the emptied square as chosen."""
        spec = copy.deepcopy(SHOT)
        spec["rules"]["strike"] = {"position": "stay", "by": ["R"], "immune": ["K"]}
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = place(engine, {**KINGS, "a3": ("R", 0), "d4": ("P", 1), "c3": ("S", NOBODY)})
        shot = find(engine, state, 0, "a3", "c3")
        take = find(engine, state, 1, "d4", "c3")
        after, _ = round_of(engine, state, shot, take)
        assert at(engine, after, "a3") == ("R", 0)
        assert at(engine, after, "c3") == ("P", 1)
        assert held(engine, after, 0, "S") == 1 and held(engine, after, 1, "S") == 0


class TestContactsThatStayOnAStone:
    def spec(self) -> dict[str, Any]:
        spec = copy.deepcopy(STONES)
        del spec["rules"]["drops"]
        spec["rules"]["contacts"] = {"position": "stay", "immune": ["K"], "links": [
            {"mover": "P", "onto": "S", "onto_to": "R"},
            {"mover": "N", "onto": "S", "onto_to": "N"},
        ]}
        assert validate_spec(spec) == []
        return spec

    def test_the_winner_by_beats_links_alone(self) -> None:
        """Before SR7f the two links fired in seat order, the second on what the first made;
        now the knight beats the pawn, so the stone becomes the knight's making and the
        pawn's link misses, whichever seat each player holds."""
        engine = Engine(self.spec())
        state = place(engine, {**KINGS, "d4": ("P", 1), "b1": ("N", 0), "c3": ("S", NOBODY)})
        first = find(engine, state, 0, "b1", "c3")
        second = find(engine, state, 1, "d4", "c3")
        after, events = round_of(engine, state, first, second)
        assert at(engine, after, "c3") == ("N", NOBODY)
        assert at(engine, after, "b1") == ("N", 0) and at(engine, after, "d4") == ("P", 1)
        assert (EVENT_VOID, engine._square_index("d4"), 1) in events

    def test_two_tied_links_both_miss(self) -> None:
        engine = Engine(self.spec())
        state = place(engine, {**KINGS, "b2": ("P", 0), "d4": ("P", 1), "c3": ("S", NOBODY)})
        first = find(engine, state, 0, "b2", "c3")
        second = find(engine, state, 1, "d4", "c3")
        after, _ = round_of(engine, state, first, second)
        assert at(engine, after, "c3") == ("S", NOBODY)
