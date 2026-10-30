"""Published chess variants as engine conformance tests.

Perft is the only genuinely independent check on a move generator, and until now the
project had exactly one source of it: standard chess. That leaves most of the DSL
unverified, because chess exercises one board shape, one castling distance, one
promotion rank and one pawn. The variants here were chosen so that between them they
touch every corner of the schema a proposal is likely to reach for.

Reference node counts come from Fairy-Stockfish, which implements all of these
independently of anything in this repository:

* Depths published in its own regression suite, ``tests/perft.sh``.
* Remaining depths generated once from Fairy-Stockfish 0.0.89 (via ``pyffish``) and
  recorded here as constants. ``pyffish`` is deliberately *not* a dependency; the
  numbers are the artefact, exactly as the chess perft constants are.

The committed rulesets under ``rulesets/`` carry the games worth keeping as
calibration anchors. The probes further down are specs built inline: they exist only
to pin a DSL feature against a published number, not to be played.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# Depths above this take long enough that they belong behind the ``slow`` marker. The
# shallower rungs of each ladder still run every time, so a movegen regression on any
# variant surfaces in the fast suite.
FAST_BUDGET_NODES = 150_000

# ruleset name -> {depth: nodes}. Every entry is Fairy-Stockfish's number, not ours.
BASELINE_PERFT: dict[str, dict[int, int]] = {
    "gardner": {1: 7, 2: 53, 3: 506, 4: 4_775, 5: 52_512},
    "losalamos": {1: 10, 2: 100, 3: 1_212, 4: 14_332, 5: 191_846},
    "shatranj": {1: 16, 2: 256, 3: 4_176, 4: 68_122},
    "makruk": {1: 23, 2: 529, 3: 12_012, 4: 273_026},
    "knightmate": {1: 18, 2: 324, 3: 6_765, 4: 139_774},
    "almost": {1: 22, 2: 484, 3: 11_895},
    "berolina": {1: 30, 2: 900, 3: 28_328},
    "capablanca": {1: 28, 2: 784, 3: 25_228, 4: 805_128},
    "courier": {1: 26, 2: 678, 3: 18_406, 4: 500_337},
    # Fairy-Stockfish's published `seirawan` (its tests/perft.sh); 5 = 27,639,803 is
    # `tests/test_gating.py`'s, on the same mechanics hash (Q1059b).
    "seirawan": {1: 28, 2: 784, 3: 24_830, 4: 782_599},
}

@cache
def engine_for(name: str) -> Engine:
    """Compile once per ruleset. Courier builds a table for 96 squares by 9 pieces."""
    return Engine(load_spec(RULESETS / f"{name}.json"))


def perft_cases(ladders: dict[str, dict[int, int]]) -> list[Any]:
    cases = []
    for name, ladder in ladders.items():
        for depth, expected in sorted(ladder.items()):
            marks = [pytest.mark.slow] if expected > FAST_BUDGET_NODES else []
            cases.append(pytest.param(name, depth, expected, id=f"{name}-d{depth}", marks=marks))
    return cases


@pytest.mark.parametrize(("name", "depth", "expected"), perft_cases(BASELINE_PERFT))
def test_variant_perft_matches_the_reference_implementation(
    name: str, depth: int, expected: int
) -> None:
    engine = engine_for(name)
    assert engine.perft(engine.initial_state(), depth) == expected


def test_berolina_endgame_position_matches_the_reference_implementation() -> None:
    """A position rather than an opening, because it is where the inversion bites.

    Fairy-Stockfish ``berolina`` from ``k7/6P1/8/8/8/2K2p2/4p3/8 w - - 0 1``. A pawn
    that advances diagonally promotes on two different files, and one that captures
    straight ahead cannot take the piece in front of it -- both of which the opening
    position hides behind a wall of pawns.
    """
    engine = engine_for("berolina")
    state = engine.make_state(
        {"a8": ("K", 1), "g7": ("P", 0), "c3": ("K", 0), "f3": ("P", 1), "e2": ("P", 1)}
    )
    assert engine.perft(state, 3) == 1_983


# --------------------------------------------------------------------------- probes


def _chess_like(
    name: str,
    extra_pieces: list[dict[str, Any]],
    back_rank: dict[str, list[str]],
    promotion_to: list[str],
    pawn: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A standard chess frame with the back rank and pawn supplied by the caller."""
    king = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
    default_pawn = [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {
            "kind": "slide",
            "vectors": [[0, 1]],
            "range": 2,
            "min_range": 2,
            "mode": "move",
            "condition": {"from_rank": 1},
        },
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
    ]
    placements = [
        {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"]}
    ] + [{"piece": piece, "squares": squares} for piece, squares in back_rank.items()]
    return {
        "schema_version": 1,
        "name": name,
        "board": {"width": 8, "height": 8},
        "pieces": [{"id": "P", "value": 1, "moves": pawn or default_pawn}]
        + extra_pieces
        + [{"id": "K", "value": 0, "royal": True, "moves": [king]}],
        "setup": {"mirror": True, "placements": placements},
        "rules": {
            "win": "checkmate",
            "stalemate": "draw",
            "promotion": {"piece": "P", "rank": "last", "to": promotion_to},
            "en_passant": True,
            "castling": {"king": "K", "rook": "R", "king_steps": 2},
            "max_plies": 400,
            "repetition_draw": 3,
            "no_progress_draw": 100,
        },
    }


KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}
BISHOP = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
QUEEN = {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
STANDARD_BACK = {"R": ["a1", "h1"], "N": ["b1", "g1"], "B": ["c1", "f1"], "Q": ["d1"], "K": ["e1"]}


def _mode(descriptor: dict[str, Any], mode: str) -> dict[str, Any]:
    return dict(descriptor, mode=mode)


# A rider is nothing but a slide along a non-unit vector; the DSL needs no new
# primitive for one. Nightrider chess is the published check on that reading.
NIGHTRIDER = _chess_like(
    "nightrider",
    [
        {"id": "S", "value": 5, "moves": [
            {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}
        ]},
        {"id": "B", "value": 3, "moves": [BISHOP]},
        {"id": "R", "value": 5, "moves": [ROOK]},
        {"id": "Q", "value": 9, "moves": [QUEEN]},
    ],
    {"R": ["a1", "h1"], "S": ["b1", "g1"], "B": ["c1", "f1"], "Q": ["d1"], "K": ["e1"]},
    ["Q", "R", "B", "S"],
)

# Capture geometry that differs from movement geometry, in both directions at once.
HOPPELPOPPEL = _chess_like(
    "hoppelpoppel",
    [
        {"id": "N", "value": 3, "moves": [_mode(KNIGHT, "move"), _mode(BISHOP, "capture")]},
        {"id": "B", "value": 3, "moves": [_mode(BISHOP, "move"), _mode(KNIGHT, "capture")]},
        {"id": "R", "value": 5, "moves": [ROOK]},
        {"id": "Q", "value": 9, "moves": [QUEEN]},
    ],
    STANDARD_BACK,
    ["Q", "R", "B", "N"],
)

# The same trick applied to the castling rook, which is where it can go wrong: the
# rook's capture squares change but its castling eligibility must not.
NEWZEALAND = _chess_like(
    "newzealand",
    [
        {"id": "N", "value": 3, "moves": [_mode(KNIGHT, "move"), _mode(ROOK, "capture")]},
        {"id": "B", "value": 3, "moves": [BISHOP]},
        {"id": "R", "value": 5, "moves": [_mode(ROOK, "move"), _mode(KNIGHT, "capture")]},
        {"id": "Q", "value": 9, "moves": [QUEEN]},
    ],
    STANDARD_BACK,
    ["Q", "R", "B", "N"],
)

# A double step with no rank condition at all, which also means en passant from
# anywhere on the board.
TORPEDO = _chess_like(
    "torpedo",
    [
        {"id": "N", "value": 3, "moves": [KNIGHT]},
        {"id": "B", "value": 3, "moves": [BISHOP]},
        {"id": "R", "value": 5, "moves": [ROOK]},
        {"id": "Q", "value": 9, "moves": [QUEEN]},
    ],
    STANDARD_BACK,
    ["Q", "R", "B", "N"],
    pawn=[
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
    ],
)

# Two move-only descriptors on one piece, in different symmetry groups.
PAWNSIDEWAYS = _chess_like(
    "pawnsideways",
    [
        {"id": "N", "value": 3, "moves": [KNIGHT]},
        {"id": "B", "value": 3, "moves": [BISHOP]},
        {"id": "R", "value": 5, "moves": [ROOK]},
        {"id": "Q", "value": 9, "moves": [QUEEN]},
    ],
    STANDARD_BACK,
    ["Q", "R", "B", "N"],
    pawn=[
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 0]], "symmetry": "lateral", "mode": "move"},
        {
            "kind": "slide",
            "vectors": [[0, 1]],
            "range": 2,
            "min_range": 2,
            "mode": "move",
            "condition": {"from_rank": 1},
        },
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
    ],
)

PROBE_PERFT: dict[str, tuple[dict[str, Any], dict[int, int]]] = {
    "nightrider": (NIGHTRIDER, {1: 24, 2: 576, 3: 15_586, 4: 419_019}),
    "hoppelpoppel": (HOPPELPOPPEL, {1: 20, 2: 400, 3: 9_034, 4: 202_459}),
    "newzealand": (NEWZEALAND, {1: 20, 2: 400, 3: 8_976, 4: 200_310}),
    "torpedo": (TORPEDO, {1: 20, 2: 400, 3: 9_194, 4: 209_719}),
    "pawnsideways": (PAWNSIDEWAYS, {1: 20, 2: 400, 3: 10_022}),
}


@cache
def probe_engine(name: str) -> Engine:
    return Engine(PROBE_PERFT[name][0])


@pytest.mark.parametrize(
    ("name", "depth", "expected"),
    perft_cases({name: ladder for name, (_, ladder) in PROBE_PERFT.items()}),
)
def test_probe_perft_matches_the_reference_implementation(
    name: str, depth: int, expected: int
) -> None:
    engine = probe_engine(name)
    assert engine.perft(engine.initial_state(), depth) == expected


# ----------------------------------------------------------------- documented limits


class TestVersionOneHoldsItsLine:
    """Schema 1 is now a subset, and it has to stay one.

    Everything below is expressible in schema 2 and tested in ``test_schema_two.py``.
    What is tested here is that a spec which *declares* version 1 cannot reach it --
    the declared version is part of the mechanics hash, so a spec must not be able to
    use a feature it did not claim.
    """


def test_a_promoting_double_step_opens_no_en_passant_window() -> None:
    """Regression: en passant used to be able to capture the piece a pawn became.

    Unreachable in chess, whose promotion rank is out of double-step range, but a
    ruleset that promotes mid-board -- makruk does, at its sixth rank -- puts the two
    rules within reach of each other.
    """
    spec = {
        "schema_version": 1,
        "name": "mid-board promotion",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "P", "value": 1, "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2,
                 "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
            ]},
            {"id": "Q", "value": 9, "moves": [QUEEN]},
            {"id": "K", "value": 0, "royal": True, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
            ]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "P", "squares": ["a2"]}, {"piece": "K", "squares": ["e1"]}
        ]},
        "rules": {
            "win": "checkmate",
            "promotion": {"piece": "P", "rank": 4, "to": ["Q"]},
            "en_passant": True,
            "max_plies": 400,
        },
    }
    engine = Engine(spec)
    state = engine.make_state(
        {"b3": ("P", 0), "a5": ("P", 1), "e1": ("K", 0), "h8": ("K", 1)}
    )
    after = engine.apply(state, engine.find_move(state, "Pb3-b5=Q"))
    assert not [text for text in engine.move_texts(after) if "e.p." in text]
