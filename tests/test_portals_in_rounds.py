"""SR8e (R9, ruled 2026-10-27): portals a player moves, in rounds.

`docs/plan-schema-next.md` §2.3 row R9 and `docs/reference/rounds.md` §4 step 1. The round's
portal moves are made before its piece moves, and a piece move then reads the ends where they
now stand. Readings SR8e settled (the row left them unstated): two moves of one end to two
squares, or of two ends to one square, are both void (the ends stay; one end to one square by
both is made once); a portal end meets nothing, so the round's one piece move is resolved
alone on the board the portal moves left, as an ordered round's slower move is; a transit or a
go-through keeps the end it went in by and the momentum it set out with and comes out where the
far end now stands, and the word so read must be offered on that board, else the move is void;
every other piece move must be offered as chosen (a move onto a square a momentum end now
holds is void). The straight-back bar stays refused beside rounds.

The test game is `tests/_simultaneous_portals_moved.json`: Q918a's simultaneous Gardner board,
a momentum pair (orange moved by player 0, blue by player 1) and a pair of folds both players
move. Every round here is checked against the declared oracle (`tests/simultaneous_oracle.py`),
and `tests/test_simultaneous_oracle.py` walks the spec at random.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, rust_supports
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_FOLD,
    MOVE_PORTAL,
    decode_move,
    move_fold,
    move_portal,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

HERE = Path(__file__).parent
EVENT_VOID = 1

MOVED = json.loads((HERE / "_simultaneous_portals_moved.json").read_text(encoding="utf-8"))
ENGINE = Engine(MOVED)
KINGS = {"e1": ("K", 0), "a5": ("K", 1)}
ENDS = {"b3": ("1o", WILD), "d3": ("1b", WILD)}


def sq(name: str) -> int:
    return ENGINE._square_index(name)


def place(pieces: dict[str, tuple[str, int]], folds: tuple[str, str] = ("a4", "c1")) -> GameState:
    state = ENGINE.make_state({**KINGS, **pieces}, folds=folds)
    key = ENGINE._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def find(state: GameState, player: int, origin: str, target: str, kind: int | None = None) -> int:
    """``player``'s move from ``origin`` to ``target`` (of ``kind``, when given), the one word."""
    words = [
        m for m in ENGINE.legal_moves_for(state, player)
        if decode_move(m)[0] == sq(origin) and decode_move(m)[1] == sq(target)
        and (kind is None or decode_move(m)[2] == kind)
    ]
    assert len(words) == 1, (origin, target, kind, [hex(m) for m in words])
    return words[0]


def through_fold(state: GameState, player: int, origin: str, target: str) -> int:
    words = [
        m for m in ENGINE.legal_moves_for(state, player)
        if decode_move(m)[0] == sq(origin) and decode_move(m)[1] == sq(target) and move_fold(m)
        and decode_move(m)[2] != MOVE_FOLD
    ]
    assert len(words) == 1, words
    return words[0]


def round_of(state: GameState, first: int, second: int):
    after, _, events = ENGINE.apply_round_with_phase(state, first, second)
    assert after == resolve_round(ENGINE, state, first, second)
    assert after.key == ENGINE._compute_hash(after)
    return after, events


def at(state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[sq(name)]
    if value == 0:
        return None
    spec = ENGINE.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def end_id(state: GameState, name: str) -> str | None:
    """The momentum end standing on ``name``, by its internal id, or None."""
    value = state.board[sq(name)]
    spec = ENGINE.spec
    assert spec.portals is not None
    if value == 0 or spec.owner_of(value) != WILD:
        return None
    return spec.piece_ids[spec.type_of(value)]


class TestTheLift:
    def test_a_portal_a_player_moves_validates_in_rounds(self) -> None:
        assert validate_spec(MOVED) == []
        assert rust_supports(compile_spec(MOVED)) is None

    def test_the_straight_back_bar_stays_refused(self) -> None:
        spec = copy.deepcopy(MOVED)
        spec["rules"]["portals"][0]["no_straight_back"] = True
        assert "R9" in refused_ids(validate_spec(spec))
        without = copy.deepcopy(spec)
        del without["rules"]["simultaneous"]
        assert "R9" not in refused_ids(validate_spec(without))

    def test_the_core_holds_the_bar_shut_past_the_validator(self) -> None:
        compiled = compile_spec(MOVED)
        assert compiled.portals is not None
        barred = dataclasses.replace(
            compiled, portals=dataclasses.replace(
                compiled.portals, no_back=(True,) * compiled.n_types
            ),
        )
        held = rust_supports(barred)
        assert held is not None and "no_straight_back" in held

    def test_fairy_stockfish_refuses_the_pair_by_name(self) -> None:
        """Fairy-Stockfish has neither half (no round, no portal), and says so for each."""
        from chess_variant_sandbox.fairy import why_not

        reasons = why_not(MOVED)
        assert any(reason.startswith("rules.simultaneous") for reason in reasons)
        assert any(reason.startswith("rules.portals") for reason in reasons)

    def test_describe_says_the_portal_moves_come_first(self) -> None:
        from chess_variant_sandbox.spec.describe import describe_rules

        assert any(
            line.startswith("In a round, a portal's move is made before the pieces move")
            for line in describe_rules(MOVED)
        )

    def test_each_player_is_offered_the_ends_they_move(self) -> None:
        state = place(ENDS)
        moved = [
            {decode_move(m)[0] for m in ENGINE.legal_moves_for(state, p)
             if decode_move(m)[2] == MOVE_FOLD or end_id(state, _name(decode_move(m)[0]))}
            for p in (0, 1)
        ]
        folds = {sq("a4"), sq("c1")}
        assert moved[0] == {sq("b3")} | folds
        assert moved[1] == {sq("d3")} | folds


def _name(square: int) -> str:
    width = ENGINE.spec.width
    return "abcdefgh"[square % width] + str(square // width + 1)


class TestATransitReadsTheEndsWhereTheyNowStand:
    def test_an_unmoved_transit_is_the_transit(self) -> None:
        state = place({**ENDS, "b1": ("R", 0)})
        transit = find(state, 0, "b1", "d4", MOVE_PORTAL)
        after, events = round_of(state, transit, find(state, 1, "a5", "b5"))
        assert at(after, "d4") == ("R", 0) and at(after, "b1") is None
        assert not events

    def test_the_far_end_moved_first_so_the_piece_comes_out_beside_it(self) -> None:
        """Player 1 moves the blue end d3 -> e3 as player 0's rook goes in by orange on b3,
        heading north: it comes out one step north of where blue now stands, on e4."""
        state = place({**ENDS, "b1": ("R", 0)})
        transit = find(state, 0, "b1", "d4", MOVE_PORTAL)
        assert ENGINE.spec.piece_ids[move_portal(transit)] == "1o"
        after, events = round_of(state, transit, find(state, 1, "d3", "e3"))
        assert at(after, "e4") == ("R", 0)
        assert at(after, "d4") is None and at(after, "b1") is None
        assert end_id(after, "e3") == "1b" and end_id(after, "d3") is None
        assert not events

    def test_an_exit_the_moved_end_puts_off_the_board_is_void(self) -> None:
        state = place({"b3": ("1o", WILD), "d4": ("1b", WILD), "b1": ("R", 0)})
        transit = find(state, 0, "b1", "d5", MOVE_PORTAL)
        after, events = round_of(state, transit, find(state, 1, "d4", "d5"))
        assert at(after, "b1") == ("R", 0)
        assert end_id(after, "d5") == "1b"
        assert (EVENT_VOID, sq("b1"), 0) in events

    def test_an_entry_end_moved_out_of_reach_makes_the_transit_void(self) -> None:
        """Player 1's rook heads south into orange on b3; player 0 moves orange to c2, off
        the file: the rook no longer reaches it, so its move is void and it stays."""
        state = place({**ENDS, "b5": ("R", 1)}, folds=("a4", "e2"))
        transit = find(state, 1, "b5", "d2", MOVE_PORTAL)
        after, events = round_of(state, find(state, 0, "b3", "c2"), transit)
        assert at(after, "b5") == ("R", 1) and at(after, "d2") is None
        assert end_id(after, "c2") == "1o"
        assert (EVENT_VOID, sq("b5"), 1) in events

    def test_an_entry_end_moved_along_the_line_is_still_entered(self) -> None:
        """Orange moves b3 -> b2, further down the rook's file: the rook still reaches it with
        the same momentum and comes out of blue, which did not move, on d2."""
        state = place({**ENDS, "b5": ("R", 1)}, folds=("a4", "e2"))
        transit = find(state, 1, "b5", "d2", MOVE_PORTAL)
        after, events = round_of(state, find(state, 0, "b3", "b2"), transit)
        assert at(after, "d2") == ("R", 1) and at(after, "b5") is None
        assert end_id(after, "b2") == "1o"
        assert not events

    def test_a_move_onto_the_square_a_momentum_end_now_holds_is_void(self) -> None:
        state = place({**ENDS, "a2": ("R", 0)})
        after, events = round_of(state, find(state, 0, "a2", "c2"), find(state, 1, "d3", "c2"))
        assert at(after, "a2") == ("R", 0) and end_id(after, "c2") == "1b"
        assert (EVENT_VOID, sq("a2"), 0) in events


class TestAGoThroughReadsTheFoldWhereItNowStands:
    def test_the_far_fold_moved_first_so_the_piece_lands_in_it(self) -> None:
        """Player 0's rook goes through the orange fold on a3 to the blue fold on e4; player
        1 moves blue to e5 first, so the rook lands on e5, standing in it."""
        state = place({**ENDS, "a1": ("R", 0)}, folds=("a3", "e4"))
        through = through_fold(state, 0, "a1", "e4")
        after, _ = round_of(state, through, find(state, 1, "e4", "e5", MOVE_FOLD))
        assert at(after, "e5") == ("R", 0) and at(after, "e4") is None
        assert after.folds == (sq("a3"), sq("e5"))

    def test_a_fold_moved_under_the_landing_square_holds_the_piece(self) -> None:
        state = place({**ENDS, "a1": ("R", 0)}, folds=("a4", "c3"))
        after, _ = round_of(state, find(state, 0, "a1", "a2"), find(state, 1, "c3", "b2", MOVE_FOLD))
        assert at(after, "a2") == ("R", 0)
        assert after.folds == (sq("a4"), sq("b2"))


class TestTwoPortalMoves:
    def test_one_end_moved_to_two_squares_is_both_void(self) -> None:
        state = place(ENDS, folds=("a4", "c1"))
        after, events = round_of(
            state, find(state, 0, "c1", "b2", MOVE_FOLD), find(state, 1, "c1", "d2", MOVE_FOLD)
        )
        assert after.folds == (sq("a4"), sq("c1"))
        assert (EVENT_VOID, sq("c1"), 0) in events and (EVENT_VOID, sq("c1"), 1) in events

    def test_one_end_moved_to_one_square_by_both_is_made_once(self) -> None:
        state = place(ENDS, folds=("a4", "c1"))
        after, events = round_of(
            state, find(state, 0, "c1", "c2", MOVE_FOLD), find(state, 1, "c1", "c2", MOVE_FOLD)
        )
        assert after.folds == (sq("a4"), sq("c2"))
        assert not events

    def test_two_ends_moved_to_one_square_are_both_void(self) -> None:
        state = place(ENDS)
        after, events = round_of(state, find(state, 0, "b3", "c3"), find(state, 1, "d3", "c3"))
        assert end_id(after, "b3") == "1o" and end_id(after, "d3") == "1b"
        assert end_id(after, "c3") is None
        assert len([e for e in events if e[0] == EVENT_VOID]) == 2

    def test_two_ends_moved_apart_are_both_made(self) -> None:
        state = place(ENDS)
        after, _ = round_of(state, find(state, 0, "b3", "b4"), find(state, 1, "d3", "d2"))
        assert end_id(after, "b4") == "1o" and end_id(after, "d2") == "1b"


def _children(state: GameState) -> int:
    total = 0
    lists = [ENGINE.legal_moves_for(state, p) for p in (0, 1)]
    for a in lists[0]:
        for b in lists[1]:
            after = ENGINE.apply_round(state, a, b)
            total += len(ENGINE.legal_moves_for(after, 0)) * len(ENGINE.legal_moves_for(after, 1))
    return total


def test_the_round_perft_unmakes_what_a_portal_round_made() -> None:
    """The core's joint perft makes and unmakes each round (a fold's square in the snapshot):
    its depth-two count equals the sum the applied rounds give."""
    state = place({**ENDS, "b1": ("R", 0), "b5": ("R", 1)}, folds=("a3", "e4"))
    assert ENGINE.perft(state, 2) == _children(state)


def test_random_rounds_move_portals_and_agree_with_the_oracle() -> None:
    """Played at random from the start, portal moves biased in: portal rounds happen, with a
    transit or a go-through beside them, and every round agrees with the oracle."""
    spec = ENGINE.spec
    assert spec.portals is not None
    portal_rounds = 0
    beside = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = ENGINE.initial_state()
        for _ in range(25):
            if ENGINE.terminal(state) is not None:
                break
            lists = [ENGINE.legal_moves_for(state, p) for p in (0, 1)]

            def moves_end(m: int, board: bytes = state.board) -> bool:
                origin, _, kind, _ = decode_move(m)
                value = board[origin]
                return kind == MOVE_FOLD or bool(value and spec.owner_of(value) == WILD)

            picks = []
            for ms in lists:
                ends = [m for m in ms if moves_end(m)]
                passes = [m for m in ms if decode_move(m)[2] == MOVE_PORTAL or (
                    move_fold(m) and decode_move(m)[2] != MOVE_FOLD)]
                roll = rng.random()
                pool = ends if ends and roll < 0.4 else passes if passes and roll < 0.8 else ms
                picks.append(rng.choice(pool))
            flags = [moves_end(m) for m in picks]
            portal_rounds += any(flags)
            beside += any(flags) and not all(flags)
            after = ENGINE.apply_round(state, *picks)
            assert after == resolve_round(ENGINE, state, *picks)
            assert after.key == ENGINE._compute_hash(after)
            state = after
    assert portal_rounds > 10 and beside > 5


def test_the_tables_name_both_players_as_movers() -> None:
    tables: dict[str, Any] = build_tables(compile_spec(MOVED))
    assert any(tables["portal_movers"])
