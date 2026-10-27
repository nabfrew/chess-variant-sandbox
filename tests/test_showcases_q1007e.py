"""The mechanism tests of Q1007e's rider showcases (`docs/archive/plan-schema-31.md` §6.1, §6.2).

**R1**, `archive/opposite-colours.json` since 2026-10-24: the colour-bound game whose rook rides
any number of two-square jumps along files and ranks (a slide over `[2, 0]`), stopped only by a
piece on a square it would land on, and whose queen adds that ride to a bishop's diagonals, on a
board two ranks longer with the pawns a rank forward (the draft R1b). It replaced Q999b's game
under its name (`tests/_retired_opposite_colours.json`, still held by
`tests/test_showcase_opposite_colours.py`). **R3**, `archive/chess-with-nightriders.json`: chess
with the knight's move written as a slide over `[1, 2]`, the nightrider.

Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and the
sight oracle (`tests/sight_oracle.py`) is the declared second reading of R1's random games.
"""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

from sight_oracle import SightOracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import SIGHT_DISARM, SIGHT_FREEZE, SIGHT_SHIELD
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"
R1_FILE, R1_HASH = ARCHIVE / "opposite-colours.json", "ce406100becc"
R3_FILE, R3_HASH = ARCHIVE / "chess-with-nightriders.json", "cc5e28e9ed0f"
PARENT_HASH = "690631b0e4eb"
RIDE = {"kind": "slide", "vectors": [[2, 0]], "symmetry": "rotate4"}
FAST_GAMES, MAX_PLIES = 3, 200


@cache
def oracle() -> SightOracle:
    return SightOracle(load_spec(R1_FILE))


def r1() -> Engine:
    return oracle().engine


@cache
def r3() -> Engine:
    return Engine(load_spec(R3_FILE))


def colour(square: int) -> int:
    width = r1().spec.width
    return (square % width + square // width) % 2


def place(rows: dict[str, tuple[str, int]], side: int) -> GameState:
    return r1().make_state(rows, side_to_move=side, moved=list(rows))


# ------------------------------------------------------------------------- R1, the file


def test_r1_is_the_registered_game() -> None:
    spec = load_spec(R1_FILE)
    assert spec_hash(spec) == R1_HASH
    assert spec["name"] == "opposite colours"
    assert spec["provenance"]["parents"] == [PARENT_HASH]
    assert (spec["board"]["width"], spec["board"]["height"]) == (8, 10)
    moves = {piece["id"]: piece["moves"] for piece in spec["pieces"]}
    # The user's move rules (2026-10-23): fixed, never tuned.
    assert moves["R"] == [RIDE]
    assert moves["Q"] == [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}, RIDE]
    pawns = next(p for p in spec["setup"]["placements"] if p["piece"] == "P")["squares"]
    assert {square[1:] for square in pawns} == {"3"}


def test_r1_every_move_keeps_its_colour_and_every_sight_looks_across() -> None:
    spec = load_spec(R1_FILE)
    for piece in spec["pieces"]:
        for move in piece["moves"]:
            assert all((dx + dy) % 2 == 0 for dx, dy in move["vectors"]), piece["id"]
        for entry in piece.get("sight", []):
            assert entry["range"] == 1
            assert all((dx + dy) % 2 == 1 for dx, dy in entry["vectors"]), piece["id"]
    effects = {
        piece["id"]: [(e["effect"], e["whose"]) for e in piece.get("sight", [])]
        for piece in spec["pieces"]
    }
    assert effects == {
        "P": [("freeze", "enemy")], "C": [("disarm", "enemy")], "R": [("shield", "own")],
        "B": [], "Q": [("freeze", "enemy")], "K": [],
    }


def test_r1_opens_with_the_riders_boxed_in_and_its_perft_is_frozen() -> None:
    eng = r1()
    state = eng.initial_state()
    texts = eng.move_texts(state)
    # 14 pawn moves, 6 camel jumps, 4 bishop steps, the queen's two diagonal steps, the king's two.
    assert len(texts) == 28
    assert not [t for t in texts if t.startswith("R")]
    assert sorted(t for t in texts if t.startswith("Q")) == ["Qd1-c2", "Qd1-e2"]
    assert [eng.perft(state, depth) for depth in (1, 2, 3)] == [28, 780, 22383]


def test_r1_no_enemy_piece_starts_on_a_rook_s_squares() -> None:
    """A `[2, 0]` rider keeps its file's and its rank's parity, so it lives on a quarter of the
    board; with the pawns a rank forward no enemy starts on that quarter (the parent's draft v4
    took on move one; §6.1 *The drafts*)."""
    eng = r1()
    spec = eng.spec
    width = spec.width
    board = eng.initial_state().board
    rooks = [s for s, v in enumerate(board) if v and spec.piece_ids[spec.type_of(v)] == "R"]
    assert len(rooks) == 4
    for rook in rooks:
        owner = spec.owner_of(board[rook])
        for square, value in enumerate(board):
            if value and spec.owner_of(value) != owner:
                assert (square % width - rook % width) % 2 or (square // width - rook // width) % 2


def test_r1_a_rook_rides_over_pieces_and_stops_where_it_lands() -> None:
    eng = r1()
    # White's rook a1 rides a3, a5, a7 over pawns on a2 and a4 (squares it never lands on), and
    # takes Black's camel on a7; its own bishop on c1 ends the ride to the right at once.
    state = place({"a1": ("R", 0), "a2": ("P", 0), "a4": ("P", 1), "a7": ("C", 1),
                   "c1": ("B", 0), "e1": ("K", 0), "h10": ("K", 1)}, 0)
    rook = sorted(t for t in eng.move_texts(state) if t.startswith("R"))
    assert rook == ["Ra1-a3", "Ra1-a5", "Ra1xa7"]


def test_r1_a_queen_rides_and_slides_and_freezes_beside_it() -> None:
    eng = r1()
    state = place({"d1": ("Q", 0), "e1": ("K", 0), "d5": ("P", 1), "e9": ("K", 1)}, 0)
    texts = set(eng.move_texts(state))
    assert {"Qd1-d3", "Qd1xd5", "Qd1-b1", "Qd1-f1", "Qd1-h1", "Qd1-c2", "Qd1-a4"} <= texts
    assert "Qd1-d7" not in texts and "Qd1-d2" not in texts
    after = eng.apply(state, eng.find_move(state, "Qd1-d3"))
    # On d3 it freezes what stands beside it: nothing yet; a Black pawn on d4 would be frozen.
    frozen = place({"d3": ("Q", 0), "e1": ("K", 0), "d4": ("P", 1), "e9": ("K", 1)}, 1)
    assert eng.effects(frozen).bits[eng._square_index("d4")] & SIGHT_FREEZE
    assert after.side_to_move == 1


# ------------------------------------------------------------------------- R1, random games


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
        binds += bool(effects.binds)
        for word in moves:
            origin, target, _, _ = decode_move(word)
            assert colour(origin) == colour(target), (seed, state.ply, word)
            assert not effects.bits[origin] & SIGHT_FREEZE
            victim = state.board[target]
            if victim:
                assert spec.owner_of(victim) != state.side_to_move
                assert not effects.bits[origin] & SIGHT_DISARM
                assert not effects.bits[target] & SIGHT_SHIELD
        if not moves:
            break
        state = eng.apply(state, rng.choice(moves))
    return binds


def test_r1_three_random_games_agree_with_the_oracle_and_the_colours_meet() -> None:
    assert sum(walk(seed) > 0 for seed in range(FAST_GAMES)) >= 2


# ------------------------------------------------------------------------- R3


def test_r3_is_chess_with_one_move_rewritten() -> None:
    spec = load_spec(R3_FILE)
    assert spec_hash(spec) == R3_HASH
    chess = load_spec(ARCHIVE.parent / "rulesets" / "chess.json")
    assert spec["provenance"]["parents"] == [spec_hash(chess)[:12]]
    for ours, theirs in zip(spec["pieces"], chess["pieces"], strict=True):
        if ours["id"] == "N":
            assert ours["moves"] == [{"kind": "slide", "vectors": [[1, 2]],
                                      "symmetry": "dihedral8", "mode": "both"}]
        else:
            assert ours == theirs
    assert spec["setup"] == chess["setup"] and spec["rules"] == chess["rules"]


def test_r3_the_nightrider_rides_from_the_opening() -> None:
    eng = r3()
    state = eng.initial_state()
    rider = sorted(t for t in eng.move_texts(state) if t.startswith("N"))
    # b1 jumps c3, d5 and takes e7, three jumps out; d2 (its own pawn) blocks the other line.
    assert rider == ["Nb1-a3", "Nb1-c3", "Nb1-d5", "Nb1xe7",
                     "Ng1-e5", "Ng1-f3", "Ng1-h3", "Ng1xd7"]
    assert [eng.perft(state, depth) for depth in (1, 2, 3)] == [24, 576, 15586]


def test_r3_squares_jumped_over_never_block() -> None:
    eng = r3()
    rows = {"a1": ("N", 0), "b2": ("P", 0), "a2": ("P", 1), "b3": ("P", 1), "c3": ("P", 0),
            "e1": ("K", 0), "e8": ("K", 1)}
    state = eng.make_state(rows, side_to_move=0, moved=list(rows))
    rider = sorted(t for t in eng.move_texts(state) if t.startswith("Na1"))
    # [1, 2]: b3 is Black's (taken, the ride ends); [2, 1]: c2, e3, g4 over b2 and c3 and d2.
    assert rider == ["Na1-c2", "Na1-e3", "Na1-g4", "Na1xb3"]
