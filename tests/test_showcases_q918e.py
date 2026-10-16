"""The showcase games for simultaneous moves (Q918e), held to their demonstration clauses.

`docs/plan-simultaneous.md`, *Q918e, pre-registered*, wrote these clauses before this file:
G1 `archive/gardner-in-rounds` (`rules.simultaneous`), G2 `...-with-meek-pawns`
(`pieces[].beats`), G3 `...-with-the-initiative` (`tie: "initiative"`), G4
`archive/watering-hole` (`tie: "impossible"`). Each drives the core through `Engine` on a
hand-built position, or on random play that chooses with a seeded `random.Random` from
`legal_moves_for`; the counting is `simultaneous_showcase_tally.RoundTally`, which reads the
core's events and decides nothing.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from simultaneous_showcase_tally import BOTH_LOST, COLLISION, RoundTally, is_swap

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec

ROOT = Path(__file__).parent.parent
SEEDS = range(20)
MAX_PLIES = 200


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


ROUNDS = load("gardner-in-rounds")
MEEK = load("gardner-in-rounds-with-meek-pawns")
INITIATIVE = load("gardner-in-rounds-with-the-initiative")
HOLE = load("watering-hole")


def square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def find(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    """The legal move of `player` from `origin` to `target` (the first, for a promotion)."""
    for move in engine.legal_moves_for(state, player):
        o, t, _kind, _promotion = decode_move(move)
        if o == square(engine, origin) and t == square(engine, target):
            return move
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def at(engine: Engine, state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[square(engine, name)]
    if value == 0:
        return None
    return engine.spec.piece_ids[engine.spec.type_of(value)], engine.spec.owner_of(value)


def place(engine: Engine, pieces: dict[str, tuple[str, int]], **changes: Any) -> GameState:
    state = engine.make_state(pieces)
    if changes:
        state = state.with_(**changes)
        key = engine._compute_hash(state)
        state = state.with_(key=key, repetition=(key,))
    return state


def meet(engine: Engine, state: GameState, first: tuple[str, str], second: tuple[str, str]):
    """One round of `first` (player 0's origin, target) and `second` (player 1's)."""
    after, _, events = engine.apply_round_with_phase(
        state, find(engine, state, 0, *first), find(engine, state, 1, *second)
    )
    return after, tuple(events)


def random_games(engine: Engine) -> tuple[RoundTally, list[list[int]], list[str]]:
    """Seeds 0 to 19 of random play, at most 200 plies: the tally, the moves, the endings."""
    tally = RoundTally()
    games: list[list[int]] = []
    endings: list[str] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        state = engine.initial_state()
        moves: list[int] = []
        tally.start_game()
        before = len(tally.collisions)
        outcome = engine.terminal(state)
        while outcome is None and len(moves) < MAX_PLIES:
            first = rng.choice(engine.legal_moves_for(state, 0))
            second = rng.choice(engine.legal_moves_for(state, 1))
            state, _ = tally.round(engine, state, first, second)
            moves += [first, second]
            outcome = engine.terminal(state)
        reason = outcome.reason if outcome else "cut"
        tally.end_game(len(moves), reason, outcome.winner if outcome else None,
                       len(tally.collisions) > before)
        games.append(moves)
        endings.append(reason)
    return tally, games, endings


@pytest.fixture(scope="module")
def rounds() -> Engine:
    return Engine(ROUNDS)


@pytest.fixture(scope="module")
def meek() -> Engine:
    return Engine(MEEK)


@pytest.fixture(scope="module")
def initiative() -> Engine:
    return Engine(INITIATIVE)


@pytest.fixture(scope="module")
def hole() -> Engine:
    return Engine(HOLE)


KINGS = {"e1": ("K", 0), "e5": ("K", 1)}


# ------------------------------------------------------------------ G1 gardner in rounds


class TestGardnerInRounds:
    def test_it_is_the_measured_test_spec(self) -> None:
        test_spec = json.loads((ROOT / "tests" / "_simultaneous_gardner.json").read_text())
        assert spec_hash(ROUNDS) == spec_hash(test_spec) == "e043fce6968e"

    def test_the_opening(self, rounds: Engine) -> None:
        state = rounds.initial_state()
        assert len(rounds.legal_moves_for(state, 0)) == len(rounds.legal_moves_for(state, 1)) == 7
        assert rounds.perft(state, 1) == 49

    def test_a_queen_and_a_pawn_meeting_are_both_removed(self, rounds: Engine) -> None:
        state = place(rounds, {**KINGS, "b1": ("Q", 0), "b4": ("P", 1)})
        after, events = meet(rounds, state, ("b1", "b3"), ("b4", "b3"))
        assert at(rounds, after, "b3") is None
        assert events == ((COLLISION, square(rounds, "b3"), BOTH_LOST),)

    def test_two_rooks_taking_each_other_trade_places(self, rounds: Engine) -> None:
        state = place(rounds, {**KINGS, "a1": ("R", 0), "a4": ("R", 1)})
        after, events = meet(rounds, state, ("a1", "a4"), ("a4", "a1"))
        assert at(rounds, after, "a4") == ("R", 0) and at(rounds, after, "a1") == ("R", 1)
        assert events == ()

    def test_two_pawn_captures_at_each_other_are_void(self, rounds: Engine) -> None:
        state = place(rounds, {**KINGS, "b2": ("P", 0), "c3": ("P", 1)})
        after, events = meet(rounds, state, ("b2", "c3"), ("c3", "b2"))
        assert at(rounds, after, "b2") == ("P", 0) and at(rounds, after, "c3") == ("P", 1)
        assert [event for event, _, _ in events] == [1, 1]

    def test_random_play_collides_and_every_collision_is_both_lost(self, rounds: Engine) -> None:
        tally, _, _ = random_games(rounds)
        assert tally.games_with_collision >= 8, tally.report("random")
        assert all(c.winner == BOTH_LOST for c in tally.collisions)


# -------------------------------------------------------------------- G2 meek pawns


class TestMeekPawns:
    def test_a_pawn_loses_a_collision_to_a_piece(self, meek: Engine) -> None:
        state = place(meek, {**KINGS, "b2": ("P", 0), "d4": ("N", 1)})
        after, events = meet(meek, state, ("b2", "b3"), ("d4", "b3"))
        assert at(meek, after, "b3") == ("N", 1)
        assert events == ((COLLISION, square(meek, "b3"), 1),)

    def test_the_same_with_the_colours_turned(self, meek: Engine) -> None:
        state = place(meek, {**KINGS, "d2": ("N", 0), "b4": ("P", 1)})
        after, events = meet(meek, state, ("d2", "b3"), ("b4", "b3"))
        assert at(meek, after, "b3") == ("N", 0)
        assert events == ((COLLISION, square(meek, "b3"), 0),)

    def test_two_pawns_meeting_are_both_removed(self, meek: Engine) -> None:
        state = place(meek, {**KINGS, "b2": ("P", 0), "b4": ("P", 1)})
        after, events = meet(meek, state, ("b2", "b3"), ("b4", "b3"))
        assert at(meek, after, "b3") is None
        assert events == ((COLLISION, square(meek, "b3"), BOTH_LOST),)

    def test_a_pawn_still_takes_a_piece_that_stands(self, meek: Engine) -> None:
        state = place(meek, {**KINGS, "b2": ("P", 0), "c3": ("N", 1), "a5": ("R", 1)})
        after, events = meet(meek, state, ("b2", "c3"), ("a5", "a4"))
        assert at(meek, after, "c3") == ("P", 0)
        assert events == ()

    def test_random_play_every_pawn_meeting_a_piece_loses(self, meek: Engine) -> None:
        tally, _, _ = random_games(meek)
        mixed = [c for c in tally.collisions if (c.first == "P") != (c.second == "P")]
        assert mixed, tally.report("random")
        assert all(c.winner == (1 if c.first == "P" else 0) for c in mixed)
        assert all(c.winner == BOTH_LOST for c in tally.collisions
                   if c.first == c.second == "P")


# ------------------------------------------------------------------ G3 the initiative


class TestTheInitiative:
    def test_black_holds_it_at_the_start(self, initiative: Engine) -> None:
        assert initiative.initial_state().initiative == 1

    @pytest.mark.parametrize("holder", [1, 0])
    def test_a_king_meeting_is_the_holders_win(self, initiative: Engine, holder: int) -> None:
        state = place(initiative, {"c2": ("K", 0), "c4": ("K", 1), "a1": ("R", 0)},
                      initiative=holder)
        after, events = meet(initiative, state, ("c2", "c3"), ("c4", "c3"))
        assert at(initiative, after, "c3") == ("K", holder)
        assert after.initiative == 1 - holder
        outcome = initiative.terminal(after)
        assert outcome is not None and outcome.winner == holder

    def test_two_pawn_meetings_go_one_each_way(self, initiative: Engine) -> None:
        state = place(initiative, {**KINGS, "a2": ("P", 0), "a4": ("P", 1),
                                   "d2": ("P", 0), "d4": ("P", 1)})
        after, events = meet(initiative, state, ("a2", "a3"), ("a4", "a3"))
        assert at(initiative, after, "a3") == ("P", 1)
        assert (COLLISION, square(initiative, "a3"), 1) in events
        assert after.initiative == 0
        after, events = meet(initiative, after, ("d2", "d3"), ("d4", "d3"))
        assert at(initiative, after, "d3") == ("P", 0)
        assert after.initiative == 1

    def test_random_play_every_tie_decided_and_alternating(self, initiative: Engine) -> None:
        tally, _, _ = random_games(initiative)
        assert tally.collisions and all(c.winner != BOTH_LOST for c in tally.collisions)
        for winners in tally.tie_winners:
            assert winners == [1 - index % 2 for index in range(len(winners))]


# --------------------------------------------------------------- G4 the watering hole


def without_tie(spec: dict[str, Any]) -> dict[str, Any]:
    plain = copy.deepcopy(spec)
    plain["rules"]["simultaneous"] = {}
    return plain


class TestTheWateringHole:
    def test_ties_are_proved_impossible(self) -> None:
        assert validate_spec(HOLE) == []

    @pytest.mark.parametrize("player,kind", [(0, "D"), (1, "E")])
    def test_a_kind_both_armies_hold_is_refused(self, player: int, kind: str) -> None:
        shared = copy.deepcopy(HOLE)
        shared["setup"]["placements"].append({"piece": kind, "player": player, "squares": ["d1"]})
        assert any('"impossible", but' in error for error in validate_spec(shared))

    def test_the_promise_changes_nothing_in_play(self, hole: Engine) -> None:
        _, games, endings = random_games(hole)
        plain = Engine(without_tie(HOLE))
        assert spec_hash(without_tie(HOLE)) != spec_hash(HOLE)
        _, plain_games, plain_endings = random_games(plain)
        assert plain_games == games and plain_endings == endings
        for moves in games:
            one, two = hole.initial_state(), plain.initial_state()
            for index in range(0, len(moves), 2):
                one = hole.apply_round(one, moves[index], moves[index + 1])
                two = plain.apply_round(two, moves[index], moves[index + 1])
                assert one.board == two.board

    @pytest.mark.parametrize("white,black,eater", [
        ("E", "D", ("E", 0)), ("C", "D", ("D", 1)), ("C", "R", ("C", 0)), ("E", "R", ("R", 1)),
    ])
    def test_a_meeting_leaves_the_eater(
        self, hole: Engine, white: str, black: str, eater: tuple[str, int]
    ) -> None:
        state = place(hole, {"a2": (white, 0), "c4": (black, 1), "g1": ("C", 0), "g7": ("R", 1)})
        after, events = meet(hole, state, ("a2", "b3"), ("c4", "b3"))
        assert at(hole, after, "b3") == eater
        assert events == ((COLLISION, square(hole, "b3"), eater[1]),)

    def test_arriving_beside_a_rat_does_not_win(self, hole: Engine) -> None:
        state = place(hole, {"c3": ("E", 0), "g1": ("C", 0), "c5": ("R", 1), "a7": ("D", 1)})
        after, _ = meet(hole, state, ("c3", "d4"), ("a7", "a6"))
        assert at(hole, after, "d4") == ("E", 0)
        assert hole.terminal(after) is None

    def test_arriving_with_no_rat_beside_the_hole_wins(self, hole: Engine) -> None:
        state = place(hole, {"c3": ("E", 0), "g1": ("C", 0), "a7": ("R", 1), "g7": ("D", 1)})
        after, _ = meet(hole, state, ("c3", "d4"), ("g7", "g6"))
        outcome = hole.terminal(after)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "reached"

    def test_an_elephant_on_the_hole_wins_when_its_rat_is_eaten(self, hole: Engine) -> None:
        state = place(hole, {"d4": ("E", 0), "b6": ("C", 0), "c5": ("R", 1), "a7": ("D", 1)})
        assert hole.terminal(state) is None
        after, _ = meet(hole, state, ("b6", "c5"), ("a7", "a6"))
        assert at(hole, after, "c5") == ("C", 0)
        outcome = hole.terminal(after)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "reached"

    def test_random_play_decides_every_meeting_and_ends_at_the_hole(self, hole: Engine) -> None:
        tally, games, endings = random_games(hole)
        assert all(c.winner != BOTH_LOST for c in tally.collisions)
        assert not any(is_swap(moves[i], moves[i + 1])
                       for moves in games for i in range(0, len(moves), 2))
        assert endings.count("reached") >= 10, endings
