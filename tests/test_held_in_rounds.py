"""SR7n (R20, staged 2026-11-05 by SR7g, built as recommended): a held piece in rounds.

`docs/plan-schema-next.md` §2.3 row R20 and `docs/reference/rounds.md` §4. A piece nobody owns of
a majority type (`control: "majority"`) is moved by whichever player has strictly more pieces
next to it, and in a round that is read on the **round's board**, before anything moves: the
holder's generator offers its moves (one holder at most, so its opponent never moves it in the
same round). A held piece's move is that player's one move of the round, an arrival as any
other: onto one square with the other player's move it collides by `beats` between the two
types, then the tie; a capture of it is the other player's capture of a nobody's piece (the
hand, the bounty). It stays nobody's whatever it meets: a strike or a link changes its type in
nobody's block, and it never changes sides. Its claim (O10) and its goal (O11) are its holder's,
read on the round's resolved board: an arrival on the holder's goal is read off the round's
moves (`goal_left_by`, as `make`'s `held_arrival` reads one move), since the piece standing
there is nobody's and decides nothing on the board. The holder after the round is read afresh.

`tie: "impossible"` is refused beside a held type by its own proof: a type either player may
hold is among both players' movers (`schema.simultaneous_movers`), so the pair (X, X) always
ties; the message names the held type.

Every round here is checked against the declared oracle (`tests/simultaneous_oracle.py`), and
`tests/test_simultaneous_oracle.py` walks the three `_simultaneous_held*` specs.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move, move_foreign
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import simultaneous_movers, validate_spec

HERE = Path(__file__).parent
EVENT_COLLISION, EVENT_VOID, EVENT_INITIATIVE = 0, 1, 2
NOBODY = 2


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


HELD = load("held")
GOAL = load("held_goal")
RIFLE = load("held_rifle")
# Kings out of the way, in the far corners.
KINGS = {"a1": ("K", 0), "f6": ("K", 1)}


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


def offers(engine: Engine, state: GameState, player: int, origin: str) -> bool:
    square = engine._square_index(origin)
    return any(decode_move(m)[0] == square for m in engine.legal_moves_for(state, player))


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


def in_hand(engine: Engine, state: GameState, player: int, pid: str) -> int:
    drops = engine.spec.drops
    assert drops is not None
    return state.hands[player][drops.hand_slot(engine.spec.piece_ids.index(pid))]


def colour(engine: Engine, state: GameState, name: str) -> int:
    return state.colour[engine._square_index(name)]


def mirror(name: str) -> str:
    """The square's image under the specs' mirrored setup (a six by six board)."""
    return name[0] + str(7 - int(name[1:]))


def seat(position: dict[str, tuple[str, int]], player: int) -> dict[str, tuple[str, int]]:
    """A position written for player 0, given to ``player`` (mirrored, the owners swapped)."""
    if player == 0:
        return position
    swapped = {0: 1, 1: 0, NOBODY: NOBODY}
    return {mirror(sq): (pid, swapped[owner]) for sq, (pid, owner) in position.items()}


def square(name: str, player: int) -> str:
    return name if player == 0 else mirror(name)


def pair(first: int, second: int, player: int) -> tuple[int, int]:
    """Two moves, ``first`` by ``player`` and ``second`` by the other, in seat order."""
    return (first, second) if player == 0 else (second, first)


class TestTheLift:
    def test_a_held_type_beside_rounds_is_valid(self) -> None:
        for spec in (HELD, GOAL, RIFLE):
            assert validate_spec(spec) == []

    def test_r18s_narrower_row_is_gone(self) -> None:
        spec = copy.deepcopy(HELD)
        assert "R18" not in refused_ids(validate_spec(spec))

    def test_a_held_type_is_among_both_players_movers(self) -> None:
        first, second = simultaneous_movers(HELD)
        assert "C" in first and "C" in second

    def test_tie_impossible_is_refused_by_the_proof_naming_the_held_type(self) -> None:
        spec = copy.deepcopy(HELD)
        spec["rules"]["simultaneous"] = {"tie": "impossible"}
        errors = validate_spec(spec)
        assert any('tie "impossible"' in error and "'C' is held" in error for error in errors), errors

    def test_tie_initiative_is_not_inert_beside_a_held_type(self) -> None:
        assert validate_spec(GOAL) == []


def circle_with_a_nobodys(held: bool, hand: bool) -> dict[str, Any]:
    """SR7n2: rock, paper and scissors for each player, unmirrored, every owned pair decisive
    (no tie between owned types), every piece able to take a nobody's ``N`` standing in the
    middle; ``N`` held (`control: "majority"`) or standing, a hand filling from captures or
    none, `tie: "impossible"`."""
    steps = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    beats = {"R": ["s"], "S": ["p"], "P": ["r"], "r": ["S", "R"], "s": ["P", "S"], "p": ["R", "P"]}
    pieces: list[dict[str, Any]] = [
        {"id": pid, "moves": steps, "captures": [*victims, "N"], "royal": pid in ("R", "r")}
        for pid, victims in beats.items()
    ]
    pieces.append({"id": "N", "moves": steps, "captures": []})
    if held:
        pieces[-1]["control"] = "majority"
    rules: dict[str, Any] = {"win": "royal_capture", "simultaneous": {"tie": "impossible"}}
    if hand:
        rules["drops"] = {"from_captures": True}
    return {
        "schema_version": 26,
        "name": "circle with a nobody's piece",
        "board": {"width": 4, "height": 4},
        "pieces": pieces,
        "setup": {"mirror": False, "placements": [
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "P", "player": 0, "squares": ["c1"]},
            {"piece": "r", "player": 1, "squares": ["d4"]},
            {"piece": "s", "player": 1, "squares": ["c4"]},
            {"piece": "p", "player": 1, "squares": ["b4"]},
            {"piece": "N", "player": "nobody", "squares": ["b2"]},
        ]},
        "rules": rules,
    }


def tie_errors(spec: dict[str, Any]) -> list[str]:
    return [error for error in validate_spec(spec) if 'tie "impossible"' in error]


class TestTheTieProofBesideAHand:
    """SR7n2: does a nobody's piece taken into a hand (`drops.from_captures`) weaken the
    `tie: "impossible"` proof? `simultaneous_holdings` skips nobody's placements, so a taken
    ``N`` dropped as its captor's own is in neither player's holdings. The proof holds anyway:
    a hand that fills from captures gives each player the other's types, so both players'
    holdings (and movers) are one set, and any type in it pairs with itself, (X, X), which
    always ties. Beside a filling hand the proof refuses whenever anyone can move anything,
    without needing ``N``; with no filling hand, a standing ``N`` is never anyone's."""

    def test_the_circle_alone_proves_no_tie(self) -> None:
        spec = circle_with_a_nobodys(held=False, hand=False)
        spec["pieces"].pop()
        spec["setup"]["placements"].pop()
        for piece in spec["pieces"]:
            piece["captures"].remove("N")
        assert validate_spec(spec) == []

    def test_a_standing_nobodys_without_a_hand_is_never_anyones(self) -> None:
        spec = circle_with_a_nobodys(held=False, hand=False)
        assert simultaneous_movers(spec) == ({"R", "S", "P"}, {"r", "s", "p"})
        assert validate_spec(spec) == []

    @pytest.mark.parametrize("held", [False, True])
    def test_a_filling_hand_makes_both_players_movers_one_set(self, held: bool) -> None:
        first, second = simultaneous_movers(circle_with_a_nobodys(held=held, hand=True))
        assert first == second
        assert {"R", "S", "P", "r", "s", "p"} <= first

    def test_a_standing_nobodys_taken_into_a_hand_is_refused_by_the_owned_pair(self) -> None:
        errors = tie_errors(circle_with_a_nobodys(held=False, hand=True))
        assert errors and "can collide and tie" in errors[0], errors

    @pytest.mark.parametrize("hand", [False, True])
    def test_a_held_nobodys_is_refused_naming_it(self, hand: bool) -> None:
        errors = tie_errors(circle_with_a_nobodys(held=True, hand=hand))
        assert errors and "'N' is held" in errors[0], errors

    def test_the_hand_refuses_even_with_no_nobodys_piece(self) -> None:
        """The proof never needed ``N``: the filling hand alone refuses."""
        spec = circle_with_a_nobodys(held=False, hand=True)
        spec["pieces"].pop()
        spec["setup"]["placements"].pop()
        for piece in spec["pieces"]:
            piece["captures"].remove("N")
        assert tie_errors(spec)


@pytest.mark.parametrize("player", [0, 1])
class TestTheHolder:
    def test_only_the_holder_moves_it(self, player: int) -> None:
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "e5": ("G", 1)}
        state = place(engine, seat(position, player))
        c3 = square("c3", player)
        assert offers(engine, state, player, c3)
        assert not offers(engine, state, 1 - player, c3)
        move = find(engine, state, player, c3, square("c4", player))
        assert move_foreign(move)

    def test_the_holder_is_read_on_the_rounds_board(self, player: int) -> None:
        """The other player's guard steps next to the held piece in the same round: the holder
        chose on the round's board, so the held move is made, and the holder after the round
        is read afresh: now the other player's."""
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c3": ("C", NOBODY), "e5": ("G", 1)}
        state = place(engine, seat(position, player))
        held = find(engine, state, player, square("c3", player), square("c4", player))
        step = find(engine, state, 1 - player, square("e5", player), square("d5", player))
        after, _ = round_of(engine, state, *pair(held, step, player))
        assert at(engine, after, square("c4", player)) == ("C", NOBODY)
        assert engine.controller_of(after, engine._square_index(square("c4", player))) == 1 - player


@pytest.mark.parametrize("player", [0, 1])
class TestAHeldMoveIsAnArrival:
    def test_a_collision_with_the_other_players_move_ties_by_default(self, player: int) -> None:
        """Each beats the other (captures absent): both go, two captures, each into its
        captor's hand -- the guard to the holder's, the condottiere to the other's."""
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "e5": ("G", 1)}
        state = place(engine, seat(position, player))
        d4 = square("d4", player)
        held = find(engine, state, player, square("c3", player), d4)
        guard = find(engine, state, 1 - player, square("e5", player), d4)
        after, events = round_of(engine, state, *pair(held, guard, player))
        assert at(engine, after, d4) is None
        assert in_hand(engine, after, player, "G") == 1
        assert in_hand(engine, after, 1 - player, "C") == 1
        assert (EVENT_COLLISION, engine._square_index(d4), -1) in events

    def test_the_initiative_decides_the_tie_for_its_holder(self, player: int) -> None:
        engine = Engine(GOAL)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "e5": ("G", 1)}
        state = place(engine, seat(position, player))
        token = state.initiative
        d4 = square("d4", player)
        held = find(engine, state, player, square("c3", player), d4)
        guard = find(engine, state, 1 - player, square("e5", player), d4)
        after, events = round_of(engine, state, *pair(held, guard, player))
        expected = ("C", NOBODY) if token == player else ("G", 1 - player)
        assert at(engine, after, d4) == expected
        assert after.initiative == 1 - token
        assert (EVENT_INITIATIVE, engine._square_index(d4), token) in events

    def test_a_standing_held_piece_is_taken_as_a_nobodys(self, player: int) -> None:
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "d4": ("G", 1)}
        state = place(engine, seat(position, player))
        wait = find(engine, state, player, square("b2", player), square("a2", player))
        take = find(engine, state, 1 - player, square("d4", player), square("c3", player))
        after, _ = round_of(engine, state, *pair(wait, take, player))
        assert at(engine, after, square("c3", player)) == ("G", 1 - player)
        assert in_hand(engine, after, 1 - player, "C") == 1

    def test_a_held_capture_takes_into_the_holders_hand(self, player: int) -> None:
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "d4": ("G", 1)}
        state = place(engine, seat(position, player))
        take = find(engine, state, player, square("c3", player), square("d4", player))
        wait = find(engine, state, 1 - player, square("f6", player), square("f5", player))
        after, _ = round_of(engine, state, *pair(take, wait, player))
        assert at(engine, after, square("d4", player)) == ("C", NOBODY)
        assert in_hand(engine, after, player, "G") == 1

    def test_a_chase_onto_the_held_pieces_square_lands(self, player: int) -> None:
        """The held piece leaves; the other player's guard, which may land on an empty
        square, arrives where it stood and takes nothing."""
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "d4": ("G", 1)}
        state = place(engine, seat(position, player))
        away = find(engine, state, player, square("c3", player), square("b3", player))
        chase = find(engine, state, 1 - player, square("d4", player), square("c3", player))
        after, _ = round_of(engine, state, *pair(away, chase, player))
        assert at(engine, after, square("b3", player)) == ("C", NOBODY)
        assert at(engine, after, square("c3", player)) == ("G", 1 - player)
        assert in_hand(engine, after, 1 - player, "C") == 0

    def test_a_held_arrival_claims_for_its_holder(self, player: int) -> None:
        engine = Engine(HELD)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "e5": ("G", 1)}
        state = place(engine, seat(position, player))
        held = find(engine, state, player, square("c3", player), square("c4", player))
        step = find(engine, state, 1 - player, square("e5", player), square("e4", player))
        after, _ = round_of(engine, state, *pair(held, step, player))
        assert colour(engine, after, square("c4", player)) == player + 1
        assert colour(engine, after, square("e4", player)) == 2 - player


@pytest.mark.parametrize("player", [0, 1])
class TestAHeldGoal:
    def test_a_safe_arrival_on_the_holders_goal_wins_for_the_holder(self, player: int) -> None:
        engine = Engine(GOAL)
        position = {**KINGS, "b3": ("G", 0), "c3": ("G", 0), "c4": ("C", NOBODY), "a5": ("G", 1)}
        state = place(engine, seat(position, player))
        assert engine.terminal(state) is None
        held = find(engine, state, player, square("c4", player), square("d5", player))
        step = find(engine, state, 1 - player, square("a5", player), square("a4", player))
        after, _ = round_of(engine, state, *pair(held, step, player))
        assert after.goal_left_by == player
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == player

    def test_an_unsafe_arrival_decides_nothing(self, player: int) -> None:
        """The other player's guard steps next to the goal square in the same round: on the
        resolved board the arrival is attacked, so it is no win, then or later."""
        engine = Engine(GOAL)
        position = {**KINGS, "b3": ("G", 0), "c3": ("G", 0), "c4": ("C", NOBODY), "e6": ("G", 1)}
        state = place(engine, seat(position, player))
        held = find(engine, state, player, square("c4", player), square("d5", player))
        step = find(engine, state, 1 - player, square("e6", player), square("e5", player))
        after, _ = round_of(engine, state, *pair(held, step, player))
        assert at(engine, after, square("d5", player)) == ("C", NOBODY)
        assert after.goal_left_by == -1
        assert engine.terminal(after) is None

    def test_the_other_players_goal_decides_nothing(self, player: int) -> None:
        """A goal is its owner's: the held piece stepping onto the other player's goal
        square wins for nobody."""
        engine = Engine(GOAL)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "d3": ("C", NOBODY), "a5": ("G", 1)}
        state = place(engine, seat(position, player))
        held = find(engine, state, player, square("d3", player), square("e2", player))
        step = find(engine, state, 1 - player, square("a5", player), square("a4", player))
        after, _ = round_of(engine, state, *pair(held, step, player))
        assert after.goal_left_by == -1
        assert engine.terminal(after) is None


@pytest.mark.parametrize("player", [0, 1])
class TestAHeldShot:
    def test_a_held_shot_strikes_for_its_holder_and_stays_nobodys(self, player: int) -> None:
        engine = Engine(RIFLE)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "d4": ("G", 1)}
        state = place(engine, seat(position, player))
        shot = find(engine, state, player, square("c3", player), square("d4", player))
        wait = find(engine, state, 1 - player, square("f6", player), square("f5", player))
        after, _ = round_of(engine, state, *pair(shot, wait, player))
        assert at(engine, after, square("c3", player)) == ("C", NOBODY)
        assert at(engine, after, square("d4", player)) is None
        assert in_hand(engine, after, player, "G") == 1

    def test_a_held_shot_strikes_the_piece_before_it_moves(self, player: int) -> None:
        engine = Engine(RIFLE)
        position = {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "d4": ("G", 1)}
        state = place(engine, seat(position, player))
        shot = find(engine, state, player, square("c3", player), square("d4", player))
        run = find(engine, state, 1 - player, square("d4", player), square("e4", player))
        after, _ = round_of(engine, state, *pair(shot, run, player))
        assert at(engine, after, square("e4", player)) is None
        assert at(engine, after, square("d4", player)) is None


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("position", ["stay", "advance"])
class TestAHeldLink:
    """A link that changes the held mover (`contacts`, `mover_to`): it becomes the new type in
    nobody's block, where it stands or where it lands, never its holder's piece."""

    def test_the_held_mover_becomes_the_linked_type_and_stays_nobodys(
        self, player: int, position: str
    ) -> None:
        spec = load("held_link")
        spec["rules"]["contacts"]["position"] = position
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = place(engine, seat(
            {**KINGS, "b2": ("G", 0), "c2": ("G", 0), "c3": ("C", NOBODY), "d4": ("G", 1)}, player,
        ))
        link = find(engine, state, player, square("c3", player), square("d4", player))
        wait = find(engine, state, 1 - player, square("f6", player), square("f5", player))
        after, _ = round_of(engine, state, *pair(link, wait, player))
        stands = square("c3", player) if position == "stay" else square("d4", player)
        left = square("d4", player) if position == "stay" else square("c3", player)
        assert at(engine, after, stands) == ("D", NOBODY)
        assert at(engine, after, left) is None


class TestTheWalksMoveHeldPieces:
    @pytest.mark.parametrize("name", ["held", "held_goal", "held_rifle", "held_link"])
    def test_a_random_walk_plays_held_moves(self, name: str) -> None:
        """The oracle walks would hold nothing if no held piece ever moved in them."""
        engine = Engine(load(name))
        rng = random.Random(7)
        held_moves = 0
        for _ in range(3):
            state = engine.initial_state()
            for _ in range(30):
                if engine.terminal(state) is not None:
                    break
                lists = [engine.legal_moves_for(state, player) for player in (0, 1)]
                first, second = rng.choice(lists[0]), rng.choice(lists[1])
                held_moves += move_foreign(first) + move_foreign(second)
                state = engine.apply_round(state, first, second)
        assert held_moves > 0


class TestTheAnalysisPassCountsHeldPieces:
    def test_each_round_is_two_side_turns_of_the_neutral_series(self) -> None:
        """Pieces nobody owns in rounds write `NEUTRAL_SERIES` as the turn path does: one entry
        a side-turn, each player's read on the round's board, where the holder is read."""
        from chess_variant_sandbox.agents.evaluation import Evaluator
        from chess_variant_sandbox.tournament import analysis as analysis_pass
        from chess_variant_sandbox.tournament.analysis import NEUTRAL_SERIES
        from chess_variant_sandbox.tournament.play import play_game

        engine = Engine(HELD)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        made = 0
        for seed in (1, 2):
            record = play_game(engine, ("greedy", "random"), seed)
            found = analysis_pass._analyse_game(
                engine, evaluator, record, analysis_pass._epsilon_for(engine)
            )
            neutral = found.neutral
            assert neutral is not None and list(neutral) == list(NEUTRAL_SERIES)
            assert all(len(series) == record.plies for series in neutral.values())
            state = engine.initial_state()
            for index in range(0, len(record.moves), 2):
                pair = record.moves[index], record.moves[index + 1]
                for seat in (0, 1):
                    legal = engine.legal_moves_for(state, seat)
                    held = sum(
                        1 for square, value in enumerate(state.board)
                        if value and engine.spec.owner_of(value) == NOBODY
                        and engine.controller_of(state, square) == seat
                    )
                    assert neutral["side"][index + seat] == seat
                    assert neutral["held"][index + seat] == held
                    assert neutral["neutral_offered"][index + seat] == sum(map(move_foreign, legal))
                    assert neutral["neutral_made"][index + seat] == int(move_foreign(pair[seat]))
                    made += int(move_foreign(pair[seat]))
                state = engine.apply_round(state, *pair)
        assert made > 0
