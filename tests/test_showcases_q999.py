"""V1, the mechanism of schema 30's three one-change showcases, without players (Q999d).

`docs/archive/plan-passive.md` §6.1 registered three games, each chess (`rulesets/chess.json`,
`4c1922ffae26`) with one key added, and §6.3 fixed this test before any of them was built:

- **S1** `archive/madrasi.json`: every type but the king freezes an enemy piece of its own type
  in its capture geometry (Madrasi without its en passant observation rule).
- **S2** `archive/chess-with-a-sheltering-king.json`: the king shields its own pieces on its
  eight lines to two squares, an aura (``through``), past any piece between.
- **S3** `archive/chess-with-daunting-knights.json`: a knight disarms every enemy piece a
  knight's jump away.

On 20 seeded random games of each (`random.Random(seed)` over the legal moves, at most 200
plies; three of each in the fast suite, at most 100 plies, the twenty under `slow`): the core's
list equals the declared oracle's (`tests/sight_oracle.py`) after every move; no frozen piece
moves; no disarmed piece captures; no shielded piece is captured. On hand-built positions: in
S1 a frozen queen on the royal's line is no check and the royal may step into that line, a check
answered by freezing the checker is offered, and two queens on one open line freeze each other;
in S2 a piece within two squares of its king, behind another piece, cannot be taken, and nothing
is shielded at three; in S3 a king a knight's jump from an enemy knight has no capture. Nothing
here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, the oracle is the
declared second reading, and the chess comparisons ask chess's own engine.

The files are the games Q999a wrote as test specs (`tests/_sight_madrasi.json`,
`_sight_sheltering_king`, `_sight_daunting_knights`), whose walks and S1 perft
`tests/test_sight_walks.py` holds; this file holds the archived games to that same mechanics.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path

import pytest
from sight_oracle import SightOracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import SIGHT_DISARM, SIGHT_FREEZE, SIGHT_SHIELD
from chess_variant_sandbox.engine.moves import MOVE_EN_PASSANT, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parent.parent
CHESS = ROOT / "rulesets" / "chess.json"
CHESS_HASH = "4c1922ffae26"

# label: (archived file, its hash, Q999a's test spec, the one sight §6.1 registered)
SHOWCASES = {
    "S1": ("madrasi", "59a118c8ee38", "_sight_madrasi", {
        "P": [{"vectors": [[1, 1]], "symmetry": "lateral", "range": 1, "effect": "freeze",
               "whose": "enemy", "affects": ["P"]}],
        "N": [{"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1, "effect": "freeze",
               "whose": "enemy", "affects": ["N"]}],
        "B": [{"vectors": [[1, 1]], "symmetry": "rotate4", "effect": "freeze", "whose": "enemy",
               "affects": ["B"]}],
        "R": [{"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "freeze", "whose": "enemy",
               "affects": ["R"]}],
        "Q": [{"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "effect": "freeze",
               "whose": "enemy", "affects": ["Q"]}],
    }),
    "S2": ("chess-with-a-sheltering-king", "1aa8c4705681", "_sight_sheltering_king", {
        "K": [{"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 2, "through": True,
               "effect": "shield", "whose": "own"}],
    }),
    "S3": ("chess-with-daunting-knights", "c2b124298181", "_sight_daunting_knights", {
        "N": [{"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1, "effect": "disarm",
               "whose": "enemy"}],
    }),
}
GAMES, MAX_PLIES = 20, 200
FAST_GAMES, FAST_PLIES = 3, 100


def path_of(label: str) -> Path:
    return ROOT / "archive" / f"{SHOWCASES[label][0]}.json"


@cache
def oracle(label: str) -> SightOracle:
    return SightOracle(load_spec(path_of(label)))


@cache
def chess() -> Engine:
    return Engine(load_spec(CHESS))


def place(engine: Engine, rows: dict[str, tuple[str, int]], side: int) -> GameState:
    return engine.make_state(rows, side_to_move=side, moved=list(rows))


def bits(engine: Engine, state: GameState, square: str) -> int:
    return engine.effects(state).bits[engine._square_index(square)]


# ------------------------------------------------------------------ the files, as registered


@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_the_file_is_chess_with_the_one_registered_sight(label: str) -> None:
    _, digest, test_spec, sights = SHOWCASES[label]
    raw = json.loads(path_of(label).read_text(encoding="utf-8"))
    parent = json.loads(CHESS.read_text(encoding="utf-8"))
    assert {p["id"]: p["sight"] for p in raw["pieces"] if "sight" in p} == sights
    stripped = [{k: v for k, v in p.items() if k != "sight"} for p in raw["pieces"]]
    assert stripped == parent["pieces"]
    for key in ("board", "setup", "rules"):
        assert raw[key] == parent[key], key
    spec = load_spec(path_of(label))
    assert spec_hash(spec) == digest
    assert spec_hash(load_spec(ROOT / "tests" / f"{test_spec}.json")) == digest
    assert spec_hash(load_spec(CHESS)) == CHESS_HASH
    assert "preview" not in spec["provenance"], "joined the library from preview (Q1005)"
    assert spec["provenance"]["parents"] == [CHESS_HASH]


# ------------------------------------------------------------------ S1, Madrasi


def test_s1_a_frozen_queen_on_the_kings_line_is_no_check_and_the_king_may_step_into_it() -> None:
    eng = oracle("S1").engine
    # Black's queen on e8 and White's on a4 face each other down a4-e8: both frozen. White's
    # king on d1 is beside the e-file the black queen stands on.
    rows = {"d1": ("K", 0), "a4": ("Q", 0), "e8": ("Q", 1), "h5": ("K", 1)}
    state = place(eng, rows, 0)
    assert bits(eng, state, "e8") & SIGHT_FREEZE and bits(eng, state, "a4") & SIGHT_FREEZE
    texts = eng.move_texts(state)
    assert "Kd1-e1" in texts and "Kd1-e2" in texts
    assert not any(text.startswith("Qa4") for text in texts)
    on_the_file = place(eng, {**{k: v for k, v in rows.items() if k != "d1"}, "e1": ("K", 0)}, 0)
    assert not eng.in_check(on_the_file)
    # In chess the same queen gives check, and the king may not step onto its file.
    assert chess().in_check(place(chess(), {**rows, "e1": ("K", 0), "d1": ("R", 0)}, 0))
    assert "Kd1-e1" not in chess().move_texts(place(chess(), rows, 0))


def test_s1_a_check_answered_by_freezing_the_checker_is_offered() -> None:
    eng = oracle("S1").engine
    # Black's rook on e8 checks White's king on e1; White's rook a1-a8 faces it along rank 8.
    state = place(eng, {"e1": ("K", 0), "a1": ("R", 0), "e8": ("R", 1), "h5": ("K", 1)}, 0)
    assert eng.in_check(state)
    texts = eng.move_texts(state)
    assert "Ra1-a8" in texts
    assert "Ra1-a2" not in texts  # not facing the checker: the check stands
    after = eng.apply(state, eng.find_move(state, "Ra1-a8"))
    assert not eng.in_check(after, 0)
    assert bits(eng, after, "e8") & SIGHT_FREEZE and bits(eng, after, "a8") & SIGHT_FREEZE


def test_s1_two_queens_on_one_open_line_freeze_each_other() -> None:
    eng = oracle("S1").engine
    rows = {"a1": ("K", 0), "d1": ("Q", 0), "d8": ("Q", 1), "h8": ("K", 1)}
    for side in (0, 1):
        state = place(eng, rows, side)
        assert bits(eng, state, "d1") & SIGHT_FREEZE and bits(eng, state, "d8") & SIGHT_FREEZE
        queen = "Qd1" if side == 0 else "Qd8"
        assert not any(text.startswith(queen) for text in eng.move_texts(state))
    # A piece between them frees both.
    blocked = place(eng, {**rows, "d4": ("P", 0)}, 0)
    assert not bits(eng, blocked, "d1") and not bits(eng, blocked, "d8")
    assert "Qd1-d3" in eng.move_texts(blocked)


# ------------------------------------------------------------------ S2, the sheltering king


def test_s2_a_piece_two_squares_from_its_king_behind_another_cannot_be_taken() -> None:
    eng = oracle("S2").engine
    # White's king e1, pawns e2 and e3 (e3 behind e2), e4 three squares out, f3 a knight's jump.
    rows = {
        "e1": ("K", 0), "e2": ("P", 0), "e3": ("P", 0), "e4": ("P", 0), "f3": ("P", 0),
        "h6": ("B", 1), "h7": ("B", 1), "h5": ("B", 1), "a8": ("K", 1),
    }
    state = place(eng, rows, 1)
    assert bits(eng, state, "e2") & SIGHT_SHIELD and bits(eng, state, "e3") & SIGHT_SHIELD
    assert not bits(eng, state, "e4") and not bits(eng, state, "f3")
    assert not bits(eng, state, "e1")  # the king never shelters itself
    texts = eng.move_texts(state)
    assert "Bh6xe3" not in texts
    assert "Bh7xe4" in texts and "Bh5xf3" in texts
    assert "Bh6xe3" in chess().move_texts(place(chess(), rows, 1))


# ------------------------------------------------------------------ S3, the daunting knights


def test_s3_a_king_a_knights_jump_from_an_enemy_knight_has_no_capture() -> None:
    eng = oracle("S3").engine
    # Black's knight on c5 checks White's king on e4 and disarms it; Black's bishop on f5 is
    # beside the king, attacked by nothing else.
    rows = {"e4": ("K", 0), "c5": ("N", 1), "f5": ("B", 1), "h8": ("K", 1)}
    state = place(eng, rows, 0)
    assert bits(eng, state, "e4") & SIGHT_DISARM
    texts = eng.move_texts(state)
    assert texts and not any("x" in text for text in texts)
    assert "Ke4xf5" in chess().move_texts(place(chess(), rows, 0))


def test_s3_two_knights_a_jump_apart_disarm_each_other() -> None:
    eng = oracle("S3").engine
    rows = {"a1": ("K", 0), "d4": ("N", 0), "e6": ("N", 1), "h1": ("K", 1)}
    state = place(eng, rows, 0)
    assert bits(eng, state, "d4") & SIGHT_DISARM and bits(eng, state, "e6") & SIGHT_DISARM
    assert "Nd4xe6" not in eng.move_texts(state)
    assert "Nd4-c6" in eng.move_texts(state)
    assert "Nd4xe6" in chess().move_texts(place(chess(), rows, 0))


# ------------------------------------------------------------------ random games


def walk(label: str, games: int, plies: int) -> int:
    """Seeded random games of one showcase, every invariant held at every position; the
    positions compared, so a walk that compared nothing fails."""
    judge = oracle(label)
    eng = judge.engine
    spec = eng.spec
    compared = 0
    for seed in range(games):
        rng = random.Random(seed)
        state = eng.initial_state()
        for _ in range(plies):
            if eng.terminal(state) is not None:
                break
            moves = eng.legal_moves(state)
            assert moves == judge.legal_moves(state), (label, seed, state.ply)
            compared += 1
            effects = eng.effects(state).bits
            mover = state.side_to_move
            for word in moves:
                origin, target, kind, _ = decode_move(word)
                assert not effects[origin] & SIGHT_FREEZE, (label, seed, state.ply, word)
                victim = state.board[target]
                if (victim and spec.owner_of(victim) != mover) or kind == MOVE_EN_PASSANT:
                    assert not effects[origin] & SIGHT_DISARM, (label, seed, state.ply, word)
                if victim and spec.owner_of(victim) != mover:
                    assert not effects[target] & SIGHT_SHIELD, (label, seed, state.ply, word)
            if not moves:
                break
            after = eng.apply(state, rng.choice(moves))
            for square, value in enumerate(state.board):
                taken = value and spec.owner_of(value) != mover and (
                    not after.board[square] or spec.owner_of(after.board[square]) == mover
                )
                assert not (taken and effects[square] & SIGHT_SHIELD), (label, seed, state.ply)
            state = after
    return compared


@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_a_few_random_games_hold_every_invariant(label: str) -> None:
    assert walk(label, FAST_GAMES, FAST_PLIES) > 30


@pytest.mark.slow
@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_twenty_random_games_hold_every_invariant(label: str) -> None:
    assert walk(label, GAMES, MAX_PLIES) > 200
