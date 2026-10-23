"""The mechanism test of the first *opposite colours* (Q999b, `docs/archive/plan-passive.md` §6.5).

Retired by Q1007e (2026-10-24, `docs/plan-schema-31.md` §6.1) when R1, the same game with
two-square riders as rook and queen, replaced it under its name; the file is kept whole as
`tests/_retired_opposite_colours.json`, and its mechanism still holds. The game in
`archive/opposite-colours.json` now is R1's (`tests/test_showcases_q1007e.py`).

The user's colour-bound showcase for schema 30's lines of sight: every piece's moves keep its
square colour, both armies have pieces on both colours, so captures happen only between pieces
on one colour and the colours touch only through lines of sight. Nothing in the file reads a
colour: the moves' vectors have an even ``dx + dy`` and the sights' an odd one.

§6.5 registered this test, before any measured game, in V1's form (§6.3): on 20 seeded random
games (`random.Random(seed)` over the legal moves, at most 200 plies; three in the fast suite,
the twenty under `slow`) the core's list equals the declared oracle's (`tests/sight_oracle.py`)
after every move; no move ever changes its piece's square colour; no capture is ever of a piece
on the other colour; no frozen piece moves, no disarmed piece captures, no shielded piece is
captured; every square a sight sees is of the other colour from its source; and an effect binds
(the effects remove a legal move) in at least 15 of the 20 games. On hand-built positions: a
pawn freezing the enemy king turns a check into mate, a guard's freeze answers a check, an
elephant shields a pawn beside it but not its own king, a camel's disarm takes a bishop's check
away. Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and
the oracle is the declared second reading.
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

import pytest
from sight_oracle import SightOracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import SIGHT_DISARM, SIGHT_FREEZE, SIGHT_SHIELD
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

FILE = Path(__file__).resolve().parent / "_retired_opposite_colours.json"
HASH = "690631b0e4eb"
GAMES, MAX_PLIES, FAST_GAMES = 20, 200, 3
BINDING_GAMES = 15  # of the 20: §6.5's bar for an effect binding in random play


@cache
def oracle() -> SightOracle:
    return SightOracle(load_spec(FILE))


def engine() -> Engine:
    return oracle().engine


def colour(square: int) -> int:
    width = engine().spec.width
    return (square % width + square // width) % 2


# ------------------------------------------------------------------ the file, as registered


def test_the_file_is_the_registered_game() -> None:
    spec = load_spec(FILE)
    assert spec_hash(spec) == HASH
    assert "preview" not in spec["provenance"], "joined the library from preview (Q1005)"
    assert "parents" not in spec["provenance"]
    assert spec["setup"].get("orientation", "mirror") == "mirror"
    assert spec["rules"]["win"] == "checkmate"


def test_every_move_keeps_its_colour_and_every_sight_looks_across() -> None:
    """Even ``dx + dy`` for every move vector, odd for every sight vector at range 1: the
    colours are a consequence of the vectors, never a rule."""
    spec = load_spec(FILE)
    sighted = set()
    for piece in spec["pieces"]:
        for move in piece["moves"]:
            assert all((dx + dy) % 2 == 0 for dx, dy in move["vectors"]), piece["id"]
        for entry in piece.get("sight", []):
            sighted.add(piece["id"])
            assert entry["range"] == 1
            assert all((dx + dy) % 2 == 1 for dx, dy in entry["vectors"]), piece["id"]
    assert sighted == {"P", "C", "E", "G"}
    effects = {
        piece["id"]: [(e["effect"], e["whose"]) for e in piece.get("sight", [])]
        for piece in spec["pieces"]
    }
    assert effects == {
        "P": [("freeze", "enemy")], "C": [("disarm", "enemy")], "E": [("shield", "own")],
        "B": [], "G": [("freeze", "enemy")], "K": [],
    }


def test_both_armies_have_pieces_on_both_colours_and_the_kings_are_apart() -> None:
    eng = engine()
    spec = eng.spec
    state = eng.initial_state()
    seen: dict[int, set[int]] = {0: set(), 1: set()}
    kings = {}
    for square, value in enumerate(state.board):
        if value:
            owner = spec.owner_of(value)
            seen[owner].add(colour(square))
            if spec.piece_ids[spec.type_of(value)] == "K":
                kings[owner] = colour(square)
    assert seen == {0: {0, 1}, 1: {0, 1}}
    assert kings[0] != kings[1]


def test_every_square_a_sight_sees_is_the_other_colour() -> None:
    """Read off the compiler's forward lines, for every type, frame and square."""
    spec = engine().spec
    sight = spec.sight
    assert sight is not None
    looked = 0
    for frame in sight.lines:
        for lines_of_type in frame:
            for origin, lines in enumerate(lines_of_type):
                for _, ray, seen in lines:
                    for index, square in enumerate(ray):
                        if seen[index]:
                            assert colour(square) != colour(origin)
                            looked += 1
    assert looked > 0


# ------------------------------------------------------------------ hand-built positions


def place(rows: dict[str, tuple[str, int]], side: int) -> GameState:
    return engine().make_state(rows, side_to_move=side, moved=list(rows))


def test_a_pawn_freezing_the_king_turns_a_check_into_mate() -> None:
    eng = engine()
    # The light bishop checks Black's king on e8 down a4-e8; White's dark pawn on e7 sees e8.
    with_pawn = place({"a1": ("K", 0), "a4": ("B", 0), "e7": ("P", 0), "e8": ("K", 1)}, 1)
    assert eng.in_check(with_pawn)
    assert eng.effects(with_pawn).bits[eng._square_index("e8")] & SIGHT_FREEZE
    outcome = eng.terminal(with_pawn)
    assert outcome is not None and outcome.winner == 0 and outcome.reason == "checkmate"
    without = place({"a1": ("K", 0), "a4": ("B", 0), "e8": ("K", 1)}, 1)
    assert eng.terminal(without) is None and eng.legal_moves(without)
    # The same board in the sight-stripped twin is not over either: the pawn cannot touch e8.
    assert oracle().twin.terminal(with_pawn) is None


def test_a_guard_freezing_the_checker_answers_the_check() -> None:
    eng = engine()
    # Black's dark bishop checks White's king e1 down a5-e1; White's light guard on c4.
    state = place({"e1": ("K", 0), "c4": ("G", 0), "a5": ("B", 1), "h8": ("K", 1)}, 0)
    assert eng.in_check(state)
    texts = eng.move_texts(state)
    assert "Gc4-b5" in texts and "Gc4-a4" in texts  # beside a5: the bishop is frozen
    assert "Gc4-d5" not in texts  # not beside it: the check stands
    after = eng.apply(state, eng.find_move(state, "Gc4-b5"))
    assert not eng.in_check(after.with_(side_to_move=0))


def test_an_elephant_shields_the_pawn_beside_it_and_not_its_king() -> None:
    eng = engine()
    # White's dark elephant on d4 sees c4, e4, d3, d5; a light pawn on e4 and the king on c4.
    state = place({"d4": ("E", 0), "e4": ("P", 0), "c4": ("K", 0), "h7": ("B", 1), "h1": ("K", 1)}, 1)
    bits = eng.effects(state).bits
    assert bits[eng._square_index("e4")] & SIGHT_SHIELD
    assert not bits[eng._square_index("c4")] & SIGHT_SHIELD
    assert "Bh7xe4" not in eng.move_texts(state)
    bare = place({"e4": ("P", 0), "c4": ("K", 0), "h7": ("B", 1), "h1": ("K", 1)}, 1)
    assert "Bh7xe4" in eng.move_texts(bare)


def test_a_camel_disarms_a_bishop_which_then_gives_no_check() -> None:
    eng = engine()
    # Black's dark bishop on a5 lines up on e1; White's light camel on c4 is a knight's jump away.
    state = place({"e1": ("K", 0), "c4": ("C", 0), "a5": ("B", 1), "h8": ("K", 1)}, 0)
    assert eng.effects(state).bits[eng._square_index("a5")] & SIGHT_DISARM
    assert not eng.in_check(state)
    assert eng.in_check(place({"e1": ("K", 0), "a5": ("B", 1), "h8": ("K", 1)}, 0))
    # Disarmed, it takes nothing: a White dark pawn on c3 stands on its line, untaken.
    black = place({"e1": ("K", 0), "c4": ("C", 0), "c3": ("P", 0), "a5": ("B", 1), "h8": ("K", 1)}, 1)
    assert "Ba5xc3" not in eng.move_texts(black)


# ------------------------------------------------------------------ random games


def walk(seed: int) -> int:
    """One seeded random game, every invariant held at every position; the side-turns at which
    the effects removed a legal move."""
    judge = oracle()
    eng = judge.engine
    spec = eng.spec
    rng = random.Random(seed)
    state = eng.initial_state()
    binds = 0
    for _ in range(MAX_PLIES):
        if eng.terminal(state) is not None:
            break
        moves = eng.legal_moves(state)
        assert moves == judge.legal_moves(state), (seed, state.ply)
        effects = eng.effects(state)
        binds += bool(effects.frozen or effects.disarmed or effects.shielded)
        for word in moves:
            origin, target, _, _ = decode_move(word)
            assert colour(origin) == colour(target), (seed, state.ply, word)
            assert not effects.bits[origin] & SIGHT_FREEZE
            victim = state.board[target]
            if victim:
                assert spec.owner_of(victim) != state.side_to_move
                assert colour(target) == colour(origin)
                assert not effects.bits[origin] & SIGHT_DISARM
                assert not effects.bits[target] & SIGHT_SHIELD
        if not moves:
            break
        state = eng.apply(state, rng.choice(moves))
    return binds


def test_three_random_games_hold_every_invariant() -> None:
    assert sum(walk(seed) > 0 for seed in range(FAST_GAMES)) >= 2


@pytest.mark.slow
def test_twenty_random_games_hold_every_invariant_and_the_colours_meet() -> None:
    assert sum(walk(seed) > 0 for seed in range(GAMES)) >= BINDING_GAMES
