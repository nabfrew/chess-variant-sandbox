"""Q798: the demonstration tests of `archive/circe-mortal-diagonals`, row 11's showcase.

`rules.strike.immune` naming a non-royal piece (`docs/reference/schema-showcase-coverage.md`,
*Q798*). Under a chain, immunity only restates a missing link; under `effect: "capture"` with
`position: "home"` it is the one way to say a kind is not reborn. The game is circe with the
queen and bishops immune: a taken queen or bishop is gone, every other taken piece goes home.
Each test below is the clause the document pre-registered (v3) before this file took its
present form; the numbering is its.

A test here drives the core through `Engine` and asserts what the core did. The only choice
of a move in Python is the seeded random walk of clause 4, Q787's probe, not an agent.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
GAME = "archive/circe-mortal-diagonals.json"
GAMES = 6  # seeds 0 to 5, as pre-registered
PLIES = 250
MORTAL = ("Q", "B")
REBORN = ("P", "N", "R")


def _engine(path: str) -> Engine:
    return Engine(load_spec(ROOT / path))


def _walk(engine: Engine, seed: int) -> Iterator[tuple[GameState, int]]:
    """A seeded uniformly random game: each (position, move played) in order."""
    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(PLIES):
        if engine.terminal(state) is not None:
            return
        moves = engine.legal_moves(state)
        if not moves:
            return
        move = rng.choice(moves)
        yield state, move
        state = engine.apply(state, move)


def _pieces(engine: Engine, state: GameState, player: int, piece: str | None = None) -> int:
    spec = engine.spec
    return sum(
        1
        for v in state.board
        if v
        and spec.owner_of(v) == player
        and (piece is None or spec.piece_ids[spec.type_of(v)] == piece)
    )


def _where(engine: Engine, state: GameState, piece: str, player: int) -> list[str]:
    spec = engine.spec
    width = spec.width
    return sorted(
        f"{chr(ord('a') + i % width)}{i // width + 1}"
        for i, v in enumerate(state.board)
        if v and spec.owner_of(v) == player and spec.piece_ids[spec.type_of(v)] == piece
    )


@pytest.fixture(scope="module")
def game() -> Engine:
    return _engine(GAME)


@pytest.fixture(scope="module")
def circe() -> Engine:
    return _engine("archive/circe.json")


def test_1_it_is_circe_plus_two_immune_entries() -> None:
    mine = load_spec(ROOT / GAME)
    theirs = load_spec(ROOT / "archive/circe.json")
    assert mine["rules"]["strike"].pop("immune") == ["K", "Q", "B"]
    assert theirs["rules"]["strike"].pop("immune") == ["K"]
    for spec in (mine, theirs):
        for key in ("name", "description", "provenance"):
            spec.pop(key)
    assert mine == theirs


def _after(
    engine: Engine, placed: dict[str, tuple[str, int]], prefix: str
) -> tuple[str, GameState]:
    """The one legal move whose text starts with ``prefix``, its full text, and the result."""
    state = engine.make_state(placed)
    (text,) = [t for t in engine.move_texts(state) if t.startswith(prefix)]
    return text, engine.apply(state, engine.find_move(state, text))


# Kings in the corners, out of every line below; every home square named below is empty.
KINGS = {"h1": ("K", 0), "h8": ("K", 1)}


@pytest.mark.parametrize(("piece", "home"), [("Q", "d8"), ("B", "f8")])
def test_2_a_taken_queen_or_bishop_stays_dead_and_only_here(
    game: Engine, circe: Engine, piece: str, home: str
) -> None:
    placed = {**KINGS, "b5": ("N", 0), "d6": (piece, 1)}
    text, here = _after(game, placed, "Nb5xd6")
    assert (text, _where(game, here, piece, 1)) == ("Nb5xd6", [])
    text, there = _after(circe, placed, "Nb5xd6")  # the control: circe sends it home
    assert (text, _where(circe, there, piece, 1)) == (f"Nb5xd6 (reborn {home})", [home])


@pytest.mark.parametrize(
    ("attacker", "origin", "piece", "home"), [("P", "e4", "N", "g8"), ("B", "b3", "R", "a8")]
)
def test_2_a_taken_knight_or_rook_still_goes_home(
    game: Engine, circe: Engine, attacker: str, origin: str, piece: str, home: str
) -> None:
    placed = {**KINGS, origin: (attacker, 0), "d5": (piece, 1)}
    for engine in (game, circe):
        text, after = _after(engine, placed, f"{attacker}{origin}xd5")
        assert (text, _where(engine, after, piece, 1)) == (
            f"{attacker}{origin}xd5 (reborn {home})",
            [home],
        ), engine.spec.name


def test_3_the_opening_perfts_are_circe_s_to_depth_4(game: Engine, circe: Engine) -> None:
    start, theirs = game.initial_state(), circe.initial_state()
    assert [game.perft(start, d) for d in range(1, 5)] == [
        circe.perft(theirs, d) for d in range(1, 5)
    ]


def test_3_depth_5_parts_from_circe(game: Engine, circe: Engine) -> None:
    assert game.perft(game.initial_state(), 5) != circe.perft(circe.initial_state(), 5)


def test_4_random_play_kills_queens_and_bishops_and_sends_the_rest_home(game: Engine) -> None:
    spec = game.spec
    with_a_death = 0
    with_a_rebirth = 0
    for seed in range(GAMES):
        death = rebirth = False
        for state, move in _walk(game, seed):
            _, target, _, _ = decode_move(move)
            occupant = state.board[target]
            victim_side = 1 - state.side_to_move
            if not occupant or spec.owner_of(occupant) != victim_side:
                continue
            victim = spec.piece_ids[spec.type_of(occupant)]
            after = game.apply(state, move)
            lost = _pieces(game, state, victim_side) - _pieces(game, after, victim_side)
            described = game.describe_move(state, move)
            if victim in MORTAL:
                kind_lost = _pieces(game, state, victim_side, victim) - _pieces(
                    game, after, victim_side, victim
                )
                assert (kind_lost, lost) == (1, 1), (seed, described)
                death = True
            elif victim in REBORN:
                rebirth = rebirth or lost == 0
        with_a_death += death
        with_a_rebirth += rebirth
    assert with_a_death >= 5, with_a_death
    assert with_a_rebirth >= 5, with_a_rebirth
