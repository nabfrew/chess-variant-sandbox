"""Q772a: the demonstration tests of schema 19's seven portal showcases.

`docs/reference/portal-showcases.md` pre-registered each prediction below, under *The
demonstration tests*, before this file was written; the class names follow it. Among them is
`docs/archive/plan-portals.md`'s N3: no move in a Divide game crosses the wall rank except a transit.

A test here drives the core through `Engine` and asserts what the core did. The only choice
of a move in Python is the seeded random walk, which takes `random.Random(seed)`'s choice from
`legal_moves`, as Q787's tests do. Squares are named, never computed from an index.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from retired_specs import REBUILT

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import MOVE_PORTAL, move_from, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(6)  # seeds 0 to 5, as pre-registered
PLIES = 300


def _spec(name: str) -> dict[str, Any]:
    # Divide's two other doors and `wormhole-adrift` left the library on 2026-10-12 (Q868);
    # their rules are still exercised here, rebuilt one key from the game each came from.
    if name in REBUILT:
        return REBUILT[name]()
    return load_spec(ROOT / "archive" / f"{name}.json")


def _engine(name: str) -> Engine:
    return Engine(_spec(name))


def _walk(engine: Engine, seed: int) -> Iterator[tuple[GameState, list[int], int]]:
    """A seeded uniformly random game: each (position, its legal moves, move played)."""
    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(PLIES):
        if engine.terminal(state) is not None:
            return
        moves = engine.legal_moves(state)
        if not moves:
            return
        move = rng.choice(moves)
        yield state, moves, move
        state = engine.apply(state, move)


def _squares(engine: Engine, files: str, ranks: range) -> set[int]:
    return {engine._square_index(f"{f}{r}") for f in files for r in ranks}


def _occupant(engine: Engine, state: GameState, square: int) -> str | None:
    value = state.board[square]
    return None if value == 0 else engine.spec.piece_ids[engine.spec.type_of(value)]


def _is_portal(engine: Engine, state: GameState, square: int) -> bool:
    value = state.board[square]
    return bool(value) and engine.spec.owner_of(value) == WILD


def _entered(engine: Engine, state: GameState, move: int) -> str:
    """The end a transit goes in at, by its name (`1o`, `2b`, ...)."""
    entry = engine.portal_entry(state, move)
    assert entry is not None
    name = _occupant(engine, state, entry)
    assert name is not None
    return name


def _transits(engine: Engine, state: GameState, moves: list[int]) -> list[str]:
    return [engine.describe_move(state, m) for m in moves if move_type(m) == MOVE_PORTAL]


def _ends(engine: Engine, state: GameState) -> dict[str, int]:
    return {
        name: square
        for square in range(len(state.board))
        if _is_portal(engine, state, square)
        and (name := _occupant(engine, state, square)) is not None
    }


def _bare(name: str, placement: dict[str, tuple[str, int]], side: int = 0) -> tuple[Engine, Any]:
    """``name``'s game on a hand-built position: its portals where the spec puts them, plus
    ``placement`` (square -> (piece, owner))."""
    spec = _spec(name)
    engine = Engine(spec)
    ends = {
        pair[end]: (f"{index + 1}{end[0]}", WILD)
        for index, pair in enumerate(spec["rules"]["portals"])
        for end in ("orange", "blue")
    }
    return engine, engine.make_state({**ends, **placement}, side_to_move=side)


# ------------------------------------------------------------------------------ Divide


class TestDivideWall:
    """N3: no move crosses rank 6 except a transit, and nothing stands on it. The wall holds
    for any door rule, so it is read on option A (one way), Q772e's option B (two way) and
    Q772f's option C (both ends fixed)."""

    @pytest.mark.parametrize("name", ["divide", "divide-two-way", "divide-fixed"])
    def test_only_a_transit_crosses_the_wall(self, name: str) -> None:
        engine = _engine(name)
        files = "abcdefghij"
        below, wall, above = (
            _squares(engine, files, range(1, 6)),
            _squares(engine, files, range(6, 7)),
            _squares(engine, files, range(7, 12)),
        )
        played = 0
        for seed in SEEDS:
            transited = False
            for state, moves, move in _walk(engine, seed):
                assert not any(state.board[sq] for sq in wall)
                for m in moves:
                    origin, target = move_from(m), move_to(m)
                    assert target not in wall, engine.describe_move(state, m)
                    crosses = (origin in below and target in above) or (
                        origin in above and target in below
                    )
                    if crosses:
                        assert move_type(m) == MOVE_PORTAL, engine.describe_move(state, m)
                if move_type(move) == MOVE_PORTAL:
                    transited = True
                    end = _entered(engine, state, move)
                    if name == "divide":  # one way: each player goes in at its own end only
                        assert end == ("1o" if state.side_to_move == 0 else "1b"), (
                            engine.describe_move(state, move)
                        )
            played += transited
        assert played >= 5

    def test_the_door_is_one_way(self) -> None:
        """A white rook beside Black's end (e9) has no transit and no way home."""
        engine, state = _bare(
            "divide", {"e8": ("R", 0), "f1": ("K", 0), "a11": ("K", 1)}
        )
        rook = engine._square_index("e8")
        below = _squares(engine, "abcdefghij", range(1, 6))
        moves = [m for m in engine.legal_moves(state) if move_from(m) == rook]
        assert moves
        assert not any(move_type(m) == MOVE_PORTAL for m in moves)
        assert not any(move_to(m) in below for m in moves)
        assert engine._square_index("e9") not in {move_to(m) for m in moves}

    @pytest.mark.parametrize("name", ["divide-two-way", "divide-fixed"])
    def test_the_two_way_door_brings_a_raider_home(self, name: str) -> None:
        """Options B (Q772e) and C (Q772f): the same rook goes in at Black's end and comes out
        beside White's (e3), one step on in the direction it travelled: home, on e4."""
        engine, state = _bare(name, {"e8": ("R", 0), "f1": ("K", 0), "a11": ("K", 1)})
        rook = engine._square_index("e8")
        transits = [
            m for m in engine.legal_moves(state)
            if move_from(m) == rook and move_type(m) == MOVE_PORTAL
        ]
        assert [move_to(m) for m in transits] == [engine._square_index("e4")]
        assert _entered(engine, state, transits[0]) == "1b"

    def test_the_fixed_ends_never_move(self) -> None:
        """Option C (Q772f): nobody may move an end, so in every position of a played game
        e3 and e9 hold the ends and no legal move starts on either."""
        engine = _engine("divide-fixed")
        ends = {engine._square_index("e3"), engine._square_index("e9")}
        plies = 0
        for seed in SEEDS:
            for state, moves, _move in _walk(engine, seed):
                assert all(state.board[sq] for sq in ends)
                assert not any(move_from(m) in ends for m in moves)
                plies += 1
        assert plies >= 100


# ------------------------------------------------------------------------------- Ferry


class TestFerry:
    def test_the_opening_transits(self) -> None:
        engine = _engine("ferry")
        texts = engine.move_texts(engine.initial_state())
        assert "Pc2-b3>xd5" in texts
        assert "Pd2-e3>xc5" in texts

    def test_both_players_move_portals_and_every_end_is_entered(self) -> None:
        engine = _engine("ferry")
        movers: set[int] = set()
        entered: set[str] = set()
        for seed in SEEDS:
            for state, _moves, move in _walk(engine, seed):
                if _is_portal(engine, state, move_from(move)):
                    movers.add(state.side_to_move)
                if move_type(move) == MOVE_PORTAL:
                    entered.add(_entered(engine, state, move))
        assert movers == {0, 1}
        assert entered == {"1o", "1b", "2o", "2b"}


# ---------------------------------------------------------------------------- Carousel


class TestCarousel:
    def test_no_move_enters_the_other_players_pair_and_no_end_moves(self) -> None:
        engine = _engine("carousel")
        start = _ends(engine, engine.initial_state())
        for seed in SEEDS:
            for state, moves, _move in _walk(engine, seed):
                assert _ends(engine, state) == start
                own = "1" if state.side_to_move == 0 else "2"
                for m in moves:
                    assert move_to(m) not in start.values(), engine.describe_move(state, m)
                    if move_type(m) == MOVE_PORTAL:
                        assert _entered(engine, state, m) == f"{own}o", (
                            engine.describe_move(state, m)
                        )

    def test_the_lift(self) -> None:
        engine = _engine("carousel")
        assert "Pb2-b3>c6" in engine.move_texts(engine.initial_state())

    def test_a_rook_reaches_its_lift_both_ways_round(self) -> None:
        engine, state = _bare(
            "carousel", {"e3": ("R", 0), "f1": ("K", 0), "a8": ("K", 1)}
        )
        assert sorted(_transits(engine, state, engine.legal_moves(state))) == [
            "Re3-b3>b5", "Re3-b3>d5",
        ]


# --------------------------------------------------------------------- the small specs


class TestTheSmallSpecs:
    def test_wormhole_opens_with_a_transit_capture(self) -> None:
        engine = _engine("wormhole")
        assert "Pa2-b3>xe4" in engine.move_texts(engine.initial_state())

    def test_wormhole_adrift_both_players_move_a_portal(self) -> None:
        engine = _engine("wormhole-adrift")
        start = engine.initial_state()
        assert any(_is_portal(engine, start, move_from(m)) for m in engine.legal_moves(start))
        movers = {
            state.side_to_move
            for seed in SEEDS
            for state, _moves, move in _walk(engine, seed)
            if _is_portal(engine, state, move_from(move))
        }
        assert movers == {0, 1}

    def test_shut_wormhole_is_shut_to_the_first_player_only(self) -> None:
        engine = _engine("shut-wormhole")
        ends = {engine._square_index("b3"), engine._square_index("d3")}
        second = 0
        for seed in SEEDS:
            for state, moves, move in _walk(engine, seed):
                if state.side_to_move == 0:
                    for m in moves:
                        assert move_type(m) != MOVE_PORTAL, engine.describe_move(state, m)
                        assert move_to(m) not in ends, engine.describe_move(state, m)
                elif move_type(move) == MOVE_PORTAL:
                    second += 1
        assert second >= 1

    def test_a_pair_closed_to_both_is_two_walls(self) -> None:
        """§2.1's reading of a closed end, as an identity: Wormhole's pair closed to both
        players has the perft of Gardner with walls on b3 and d3."""
        shut = copy.deepcopy(_spec("wormhole"))
        shut["rules"]["portals"][0]["passage"] = {"0": "closed", "1": "closed"}
        walled = load_spec(ROOT / "rulesets" / "gardner.json")
        walled["schema_version"] = 6
        walled["rules"]["obstacles"] = {"squares": ["b3", "d3"]}
        assert not validate_spec(shut) and not validate_spec(walled)
        a, b = Engine(shut), Engine(walled)
        assert [a.perft(a.initial_state(), d) for d in range(1, 5)] == [
            b.perft(b.initial_state(), d) for d in range(1, 5)
        ]

    def test_cylinder_wormhole_offers_two_exits(self) -> None:
        engine, state = _bare(
            "cylinder-wormhole", {"a4": ("R", 0), "e1": ("K", 0), "e8": ("K", 1)}
        )
        assert sorted(_transits(engine, state, engine.legal_moves(state))) == [
            "Ra4-d4>c5", "Ra4-d4>e5",
        ]

    def test_cylinder_wormhole_double_step_through_the_end(self) -> None:
        engine = _engine("cylinder-wormhole")
        assert "Pd2-d4>d6" in engine.move_texts(engine.initial_state())


@pytest.mark.parametrize(
    "name",
    ["divide", "ferry", "carousel", "wormhole", "wormhole-adrift", "shut-wormhole",
     "cylinder-wormhole", "divide-two-way", "divide-fixed"],
)
def test_each_showcase_validates(name: str) -> None:
    assert validate_spec(_spec(name)) == []
