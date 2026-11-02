"""Castling with a piece that is not royal (SR1d, `docs/plan-schema-next.md` §1 rank 8, row Z2).

Under a rule with no royalty (`win: "annihilation"`) a commander always castled, the core
reading no attack for it (`Tables::needs_check_filter` is false). SR1d lifts Z2's refusal beside
royals: where check exists (checkmate beside a royal) the core tests the castling piece's start,
crossing and landing as a king's (`threatened_as` with its type), and `generate_legal`'s pin
shortcut asks every castle whose castling piece is not royal whether it leaves the royal
attacked, since its rook moves from a square that is not the move's origin and may be the
royal's sole blocker (royal b1, rook b2, an enemy rook on b8). Anywhere else (royal capture,
annihilation) the castle reads no attack.

Hand-built 8x8 specs, not chess. The walks hold the legal list to a declared oracle (the
pseudo-legal list less every move after which the mover's royal is attacked) and check the
incremental key against `_compute_hash` every ply.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, move_type
from chess_variant_sandbox.spec.describe import describe_rules
from chess_variant_sandbox.spec.schema import validate_spec

KING_STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
ROOK_LINE = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
BISHOP_LINE = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both"}


def _spec(white: dict[str, list[str]], black: dict[str, list[str]], win: str) -> dict[str, Any]:
    """K (royal unless `win` has no royalty), C a commander that castles, R a rook, B a bishop."""
    return {
        "schema_version": 31, "name": "commander castling", "description": "a probe",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": win != "annihilation",
             "moves": [KING_STEP]},
            {"id": "C", "name": "commander", "value": 3, "moves": [KING_STEP]},
            {"id": "R", "name": "rook", "value": 5, "moves": [ROOK_LINE]},
            {"id": "B", "name": "bishop", "value": 3, "moves": [BISHOP_LINE]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": piece, "squares": squares, "player": player}
            for player, army in ((0, white), (1, black)) for piece, squares in army.items()
        ]},
        "rules": {"win": win, "stalemate": "draw",
                  "castling": {"king": "C", "rook": "R", "king_steps": 2},
                  "max_plies": 200},
    }


def _castles(spec: dict[str, Any]) -> list[str]:
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(compile_spec(spec))
    state = engine.initial_state()
    return sorted(text for text in engine.move_texts(state) if "castle" in text)


def test_a_commander_castles_under_annihilation() -> None:
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "annihilation")
    assert _castles(spec) == ["Ce1-g1 (castle)"]


def test_without_check_the_castle_reads_no_attack() -> None:
    """A black rook on f8 attacks f1, the square the commander crosses: no royalty, no check,
    so the castle stands."""
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"], "R": ["f8"]},
                 "annihilation")
    assert _castles(spec) == ["Ce1-g1 (castle)"]


def test_beside_royals_under_royal_capture_the_castle_reads_no_attack() -> None:
    """Royals, but no check: the crossing under attack does not bar the castle."""
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"], "R": ["f8"]},
                 "royal_capture")
    assert _castles(spec) == ["Ce1-g1 (castle)"]


def test_beside_royals_under_checkmate_it_validates_and_castles() -> None:
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "checkmate")
    assert _castles(spec) == ["Ce1-g1 (castle)"]


@pytest.mark.parametrize("attacker", ["e8", "f8", "g8"])
def test_where_check_exists_the_castle_reads_attacks_on_the_commander(attacker: str) -> None:
    """The castling piece's start, crossing and landing are tested as a king's: a black rook
    on any of the three files bars the castle; off them, the castle stands."""
    attacked = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]},
                     {"K": ["a8"], "R": [attacker]}, "checkmate")
    assert _castles(attacked) == []
    quiet = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]},
                  {"K": ["a8"], "R": ["c8"]}, "checkmate")
    assert _castles(quiet) == ["Ce1-g1 (castle)"]


def test_a_castle_never_bares_the_royal() -> None:
    """Royal K on b1, rook b2 pinned by the black rook on b8; the commander on e2 castles
    long, carrying the rook to d2 and leaving the king in check, so it is not offered. With
    the black rook off the b-file it is."""
    pinned = _spec({"K": ["b1"], "R": ["b2"], "C": ["e2"]},
                   {"K": ["h8"], "R": ["b8"]}, "checkmate")
    assert _castles(pinned) == []
    free = _spec({"K": ["b1"], "R": ["b2"], "C": ["e2"]},
                 {"K": ["h8"], "R": ["a8"]}, "checkmate")
    assert _castles(free) == ["Ce2-c2 (castle)"]


def test_a_castle_never_bares_the_royal_along_a_diagonal() -> None:
    """The rook on c2 is the only piece between the royal on b1 and a black bishop on d3."""
    pinned = _spec({"K": ["b1"], "R": ["c2"], "C": ["f2"]},
                   {"K": ["h8"], "B": ["d3"]}, "checkmate")
    assert _castles(pinned) == []


def test_a_pinned_commander_does_not_castle() -> None:
    """The commander on b2 is the royal's sole blocker against b8: it may not castle off it."""
    spec = _spec({"K": ["b1"], "R": ["h2"], "C": ["b2"]},
                 {"K": ["h8"], "R": ["b8"]}, "checkmate")
    assert _castles(spec) == []


def test_describe_says_how_a_common_piece_castles() -> None:
    checkmate = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "checkmate")
    assert any("C is not royal, but castles as a royal piece would" in line
               for line in describe_rules(checkmate))
    capture = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "royal_capture")
    assert any("C is not royal, and no move here is barred" in line
               for line in describe_rules(capture))
    royal = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "checkmate")
    royal["rules"]["castling"]["king"] = "K"
    assert not any("is not royal" in line for line in describe_rules(royal))
    # Without a royal castling never read an attack, and the prose is unchanged.
    none = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "annihilation")
    assert not any("is not royal" in line for line in describe_rules(none))


_WALK_WHITE = {"K": ["d1"], "C": ["e2"], "R": ["a2", "h2"], "B": ["c1", "f1"]}
_WALK_BLACK = {"K": ["d8"], "C": ["e7"], "R": ["a7", "h7"], "B": ["c8", "f8"]}


@pytest.mark.parametrize("win", ["checkmate", "royal_capture"])
def test_walks_match_the_oracle_and_keep_the_key(win: str) -> None:
    """Seeded walks with the castling rank off the edge: every ply, the legal list is the
    pseudo-legal list less each move that leaves the mover's royal attacked (under
    checkmate; under royal capture the two lists are one), and the key is `_compute_hash`'s."""
    spec = _spec(_WALK_WHITE, _WALK_BLACK, win)
    assert validate_spec(spec) == []
    engine = Engine(compile_spec(spec))
    castles = 0
    for seed in range(12):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(60):
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            mover = state.side_to_move
            pseudo = engine.pseudo_legal_moves(state)
            if win == "checkmate":
                expected = [m for m in pseudo
                            if not engine.in_check(engine.apply(state, m), mover)]
            else:
                expected = list(pseudo)
            assert sorted(legal) == sorted(expected), (seed, engine.move_texts(state))
            if not legal:
                break
            castling = [m for m in legal if move_type(m) == MOVE_CASTLE]
            move = rng.choice(castling) if castling and rng.random() < 0.5 else rng.choice(legal)
            castles += move_type(move) == MOVE_CASTLE
            state = engine.apply(state, move)
            assert engine._compute_hash(state) == state.key
    assert castles > 0
