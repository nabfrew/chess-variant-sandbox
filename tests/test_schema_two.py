"""Schema-2 conformance: the games version 1 could not express.

Same discipline as ``test_variants.py`` and the same source of truth. Every number here
is Fairy-Stockfish's, either published in its ``tests/perft.sh`` or generated once from
Fairy-Stockfish 0.0.89 and recorded as a constant. A perft that moves is a movegen
regression, not a new behaviour.

What each ruleset is here to pin down:

* ``xiangqi``    board regions, blockable leaps, cannons, the flying general
* ``shogi``      drops, six promotion rules, optional promotion bounded by immobility,
                 and since Q454 promotion on a move out of the zone (schema 15)
* ``crazyhouse`` drops from captures, with the hand carried on the state
* ``dobutsu``    all of the above at a size where every position can be checked by eye
* ``antichess``  compulsory capture, and a game with nothing royal on the board
* ``atomic``     captures with side effects
* ``horde``      asymmetric armies, one of them with no royal piece at all
* ``racingkings``  a win by arrival, and a ban on giving check
* ``kingofthehill``, ``threecheck``  arrival and a counter, on an otherwise normal game
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import BARE_ROYAL_HOLDER, royalty_can_multiply, validate_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
FAST_BUDGET_NODES = 150_000

# ruleset name -> {depth: nodes}, from the opening position.
OPENING_PERFT: dict[str, dict[int, int]] = {
    # Every rung is Fairy-Stockfish's, with one exception noted below.
    "dobutsu": {1: 4, 2: 17, 3: 123, 4: 976, 5: 8_122, 6: 71_677},
    "horde": {1: 8, 2: 128, 3: 1_274, 4: 23_310},
    "antichess": {1: 20, 2: 400, 3: 8_067, 4: 153_299},
    "kingofthehill": {1: 20, 2: 400, 3: 8_902},
    "threecheck": {1: 20, 2: 400, 3: 8_902},
    "crazyhouse": {1: 20, 2: 400, 3: 8_902, 4: 197_281},
    "atomic": {1: 20, 2: 400, 3: 8_902, 4: 197_326},
    "racingkings": {1: 21, 2: 421, 3: 11_264, 4: 296_242},
    # Depth 5 since Q454: the first depth at which a piece can leave its zone, published
    # shogi's number now that the committed file promotes on the way out (schema 15).
    "shogi": {1: 30, 2: 900, 3: 25_470, 4: 719_731, 5: 19_861_490},
    "xiangqi": {1: 44, 2: 1_920, 3: 79_666, 4: 3_290_240},
}

# King of the hill and three-check only ever end early in positions the opening cannot
# reach, so their published numbers are for the positions further down this file rather
# than for the opening. Dobutsu's published number is the depth-6 one; its shallower
# rungs are this engine's own, kept as regression pins so a change shows up in the fast
# lane instead of waiting for the deep run.

_ENGINES: dict[str, Engine] = {}


def engine_for(name: str) -> Engine:
    if name not in _ENGINES:
        _ENGINES[name] = Engine(load_spec(RULESETS / f"{name}.json"))
    return _ENGINES[name]


def perft_cases(ladders: dict[str, dict[int, int]]) -> list[Any]:
    cases = []
    for name, ladder in ladders.items():
        for depth, expected in sorted(ladder.items()):
            marks = [pytest.mark.slow] if expected > FAST_BUDGET_NODES else []
            cases.append(pytest.param(name, depth, expected, id=f"{name}-d{depth}", marks=marks))
    return cases


@pytest.mark.parametrize(("name", "depth", "expected"), perft_cases(OPENING_PERFT))
def test_opening_perft_matches_the_reference_implementation(
    name: str, depth: int, expected: int
) -> None:
    """Counted with the game's own endings honoured.

    ``stop_at_terminal`` is what makes these comparable at all: a ruleset that ends on
    arrival or on a count of checks keeps generating moves in positions that are
    already decided, and every other implementation has stopped.
    """
    engine = engine_for(name)
    assert engine.perft(engine.initial_state(), depth, stop_at_terminal=True) == expected


# ------------------------------------------------------------------- positions ----


def test_crazyhouse_drops_from_a_full_hand() -> None:
    """``2k5/8/8/8/8/8/8/4K3[QRBNPqrbnp] w - - 0 1``, depth 2.

    Two kings and ten pieces in hand: almost every move in the tree is a drop, which
    is the only way to be sure the drop generator is not quietly agreeing with the
    board generator by accident.
    """
    engine = engine_for("crazyhouse")
    index = {piece: i for i, piece in enumerate(engine.spec.piece_ids)}
    held = [0] * engine.spec.n_types
    for piece in ("Q", "R", "B", "N", "P"):
        held[index[piece]] = 1
    state = engine.make_state({"c8": ("K", 1), "e1": ("K", 0)}).with_(
        hands=(tuple(held), tuple(held))
    )
    assert engine.perft(state, 2, stop_at_terminal=True) == 75_353


def test_atomic_explosions_in_a_sharp_position() -> None:
    """``r4b1r/2kb1N2/p2Bpnp1/8/2Pp3p/1P1PPP2/P5PP/R3K2R b KQ - 0 1``, depth 2.

    Only 148 nodes, and that is the point: almost every reply is illegal because it
    would blow up the black king, so the count is dominated by the rule under test.
    """
    engine = engine_for("atomic")
    state = engine.make_state(
        {
            "a8": ("R", 1), "f8": ("B", 1), "h8": ("R", 1),
            "c7": ("K", 1), "d7": ("B", 1), "f7": ("N", 0),
            "a6": ("P", 1), "d6": ("B", 0), "e6": ("P", 1), "f6": ("N", 1), "g6": ("P", 1),
            "c4": ("P", 0), "d4": ("P", 1), "h4": ("P", 1),
            "b3": ("P", 0), "d3": ("P", 0), "e3": ("P", 0), "f3": ("P", 0),
            "a2": ("P", 0), "g2": ("P", 0), "h2": ("P", 0),
            "a1": ("R", 0), "e1": ("K", 0), "h1": ("R", 0),
        },
        side_to_move=1,
        moved=["c7"],
    )
    assert engine.perft(state, 2, stop_at_terminal=True) == 148


def test_king_of_the_hill_stops_when_a_king_reaches_the_centre() -> None:
    """``rnb2b1r/ppp2ppp/3k4/8/1PKp1pn1/3Pq3/PBP1P2P/RN1Q1B1R w - - 4 12``, depth 3."""
    engine = engine_for("kingofthehill")
    state = engine.make_state(
        {
            "a8": ("R", 1), "b8": ("N", 1), "c8": ("B", 1), "f8": ("B", 1), "h8": ("R", 1),
            "a7": ("P", 1), "b7": ("P", 1), "c7": ("P", 1), "f7": ("P", 1), "g7": ("P", 1),
            "h7": ("P", 1), "d6": ("K", 1), "b4": ("P", 0), "c4": ("K", 0), "d4": ("P", 1),
            "f4": ("P", 1), "g4": ("N", 1), "d3": ("P", 0), "e3": ("Q", 1),
            "a2": ("P", 0), "b2": ("B", 0), "c2": ("P", 0), "e2": ("P", 0), "h2": ("P", 0),
            "a1": ("R", 0), "b1": ("N", 0), "d1": ("Q", 0), "f1": ("B", 0), "h1": ("R", 0),
        },
        moved=["c4", "d6"],
        halfmove=4,
        ply=22,
    )
    assert engine.perft(state, 3, stop_at_terminal=True) == 19_003


def test_three_check_stops_on_the_third_check() -> None:
    """``7r/1p4p1/pk3p2/RN6/8/P5Pp/3p1P1P/4R1K1 w - - 1+3 1 39``, depth 3.

    White has two checks banked and needs one more, so the tree is pruned by the
    counter rather than by anything on the board.
    """
    engine = engine_for("threecheck")
    state = engine.make_state(
        {
            "h8": ("R", 1), "b7": ("P", 1), "g7": ("P", 1), "a6": ("P", 1), "b6": ("K", 1),
            "f6": ("P", 1), "a5": ("R", 0), "b5": ("N", 0), "a3": ("P", 0), "g3": ("P", 0),
            "h3": ("P", 1), "d2": ("P", 1), "f2": ("P", 0), "h2": ("P", 0),
            "e1": ("R", 0), "g1": ("K", 0),
        },
        moved=["g1", "b6"],
        halfmove=1,
        ply=76,
    ).with_(checks=(2, 0))
    assert engine.perft(state, 3, stop_at_terminal=True) == 12_407


def _decided_roots() -> list[Any]:
    """Positions the game has already ended in, by an ending one of the side to move's
    moves would undo.

    That second half is what makes them a test. An ending that outlives every move (a
    royal on its goal square, a banked third check) ends each child too, so a count
    that skips the root's own ending still finds nothing under it. Here some children
    are live: the arrival's piece can be taken off its goal square, and a capture
    resets the no-progress clock that has just run out.
    """
    arrival = engine_for("rps-v3")
    taken_off_the_goal = arrival.make_state(
        {"e9": ("R", 0), "a1": ("S", 0), "e8": ("P", 1), "i5": ("R", 1)},
        side_to_move=1,
    )
    clock = engine_for("kingofthehill")
    run_out = clock.make_state(
        {"a1": ("K", 0), "g7": ("R", 0), "h8": ("K", 1), "b7": ("P", 1)},
        side_to_move=1,
        moved=["a1", "g7", "h8"],
        halfmove=100,
        ply=120,
    )
    return [
        pytest.param(arrival, taken_off_the_goal, id="arrival"),
        pytest.param(clock, run_out, id="no-progress-clock"),
    ]


@pytest.mark.parametrize(("engine", "state"), _decided_roots())
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_threads_never_change_a_perft_at_a_decided_root(
    engine: Engine, state: Any, depth: int
) -> None:
    """A position that is already over has no leaves under ``stop_at_terminal``, on
    one thread or four (CLAUDE.md: *threads never change an answer*).

    The serial count asks the root's ending first; the split across threads used to go
    straight to the root's moves and count every child, so ``threads=4`` answered a
    different number from ``threads=1`` (Q945, the engine mutation report's side
    finding 1). The plain perft is asked too, so the root is known to have moves to
    count and the zero is the ending's, not an empty list's.
    """
    assert engine.terminal(state) is not None
    assert engine.legal_moves(state)
    assert engine.perft(state, depth, threads=1) == engine.perft(state, depth, threads=4) > 0
    assert engine.perft(state, depth, threads=1, stop_at_terminal=True) == 0
    assert engine.perft(state, depth, threads=4, stop_at_terminal=True) == 0


# ---------------------------------------------------------------------- probes ----
#
# Rulesets that pin one schema-2 feature each and are not worth committing as games.

KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}
BISHOP = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
QUEEN = {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
PAWN = [
    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
     "condition": {"from_rank": 1}},
    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
]
FILES8 = "abcdefgh"


def chess_frame(name: str, pieces: list[dict], placements: list[dict], rules: dict,
                **setup: Any) -> dict[str, Any]:
    return {
        "schema_version": 2, "name": name, "board": {"width": 8, "height": 8},
        "pieces": pieces,
        "setup": {"mirror": setup.pop("mirror", True), "placements": placements, **setup},
        "rules": rules,
    }


def standard_pieces(**flags: Any) -> list[dict]:
    return [
        {"id": "P", "value": 1, "moves": PAWN},
        {"id": "N", "value": 3, "moves": [KNIGHT]},
        {"id": "B", "value": 3, "moves": [BISHOP]},
        {"id": "R", "value": 5, "moves": [ROOK]},
        {"id": "Q", "value": 9, "royal": flags.get("royal_queen", False), "moves": [QUEEN]},
        {"id": "K", "value": 0, "royal": True, "moves": [KING]},
    ]


STANDARD_PLACEMENTS = [
    {"piece": "P", "squares": [f"{f}2" for f in FILES8]},
    {"piece": "R", "squares": ["a1", "h1"]},
    {"piece": "N", "squares": ["b1", "g1"]},
    {"piece": "B", "squares": ["c1", "f1"]},
    {"piece": "Q", "squares": ["d1"]},
    {"piece": "K", "squares": ["e1"]},
]
STANDARD_RULES = {
    "win": "checkmate", "stalemate": "draw",
    "promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
    "en_passant": True, "castling": {"king": "K", "rook": "R", "king_steps": 2},
    "max_plies": 400, "repetition_draw": 3, "no_progress_draw": 100,
}

# Both king and queen royal: the check filter has to keep two lines shut at once.
COREGAL = chess_frame("coregal", standard_pieces(royal_queen=True), STANDARD_PLACEMENTS,
                      dict(STANDARD_RULES))

# Every piece is royal-adjacent here only in the sense that losing the last one loses:
# kinglet ends when a player runs out of pawns, and nothing on the board is royal.
KINGLET = chess_frame(
    "kinglet",
    [p if p["id"] != "K" else {"id": "K", "value": 3, "moves": [KING]}
     for p in standard_pieces()],
    STANDARD_PLACEMENTS,
    {"win": "none", "stalemate": "draw",
     "extinction": {"pieces": ["P"], "min_count": 1, "result": "loss"},
     "promotion": {"piece": "P", "rank": "last", "to": ["K"]},
     "en_passant": True, "castling": {"king": "K", "rook": "R", "king_steps": 2},
     "max_plies": 400, "repetition_draw": 3, "no_progress_draw": 100},
)

# Losers chess: checkmate and stalemate both win, captures are compulsory, and the
# king stays royal so the ordinary check filter still runs.
LOSERS = chess_frame(
    "losers", standard_pieces(), STANDARD_PLACEMENTS,
    {**STANDARD_RULES, "checkmate": "win", "stalemate": "win", "must_capture": True,
     "extinction": {"pieces": "any", "min_count": 1, "result": "win"}},
)

# A grasshopper must land immediately past its screen, which is the whole difference
# between it and a cannon.
GRASSHOPPER = chess_frame(
    "grasshopper",
    standard_pieces() + [{"id": "G", "value": 3, "moves": [
        {"kind": "hop", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
         "screen_range": 1, "mode": "both"}]}],
    [
        {"piece": "G", "squares": [f"{f}2" for f in FILES8]},
        {"piece": "P", "squares": [f"{f}3" for f in FILES8]},
        *STANDARD_PLACEMENTS[1:],
    ],
    {"win": "checkmate", "stalemate": "draw",
     "promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N", "G"]},
     "castling": {"king": "K", "rook": "R", "king_steps": 2},
     "max_plies": 400, "repetition_draw": 3, "no_progress_draw": 100},
)
GRASSHOPPER["pieces"][0]["moves"] = [
    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
]

# Chigorin: the two players get different armies and different promotion choices.
CHIGORIN = chess_frame(
    "chigorin",
    standard_pieces() + [{"id": "C", "value": 8, "moves": [ROOK, KNIGHT]}],
    [
        {"piece": "P", "player": 0, "squares": [f"{f}2" for f in FILES8]},
        {"piece": "R", "player": 0, "squares": ["a1", "h1"]},
        {"piece": "N", "player": 0, "squares": ["b1", "c1", "f1", "g1"]},
        {"piece": "C", "player": 0, "squares": ["d1"]},
        {"piece": "K", "player": 0, "squares": ["e1"]},
        {"piece": "P", "player": 1, "squares": [f"{f}7" for f in FILES8]},
        {"piece": "R", "player": 1, "squares": ["a8", "h8"]},
        {"piece": "B", "player": 1, "squares": ["b8", "c8", "f8", "g8"]},
        {"piece": "Q", "player": 1, "squares": ["d8"]},
        {"piece": "K", "player": 1, "squares": ["e8"]},
    ],
    {**STANDARD_RULES,
     "promotion": {"piece": "P", "rank": "last",
                   "to_by_player": {"0": ["C", "R", "N"], "1": ["Q", "R", "B"]}}},
    mirror=False,
)

# Legan: the pawns march along a diagonal, so the second player's army is a rotation
# rather than a reflection.
LEGAN = chess_frame(
    "legan", standard_pieces(),
    [
        {"piece": "P", "squares": ["d1", "e2", "f2", "f3", "g3", "e4", "g4", "h5"]},
        {"piece": "R", "squares": ["e1", "h4"]},
        {"piece": "B", "squares": ["f1", "h2"]},
        {"piece": "N", "squares": ["g1", "h3"]},
        {"piece": "K", "squares": ["h1"]},
        {"piece": "Q", "squares": ["g2"]},
    ],
    {"win": "checkmate", "stalemate": "draw",
     "promotion": {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
     "max_plies": 400, "repetition_draw": 3, "no_progress_draw": 100},
    orientation="rotate",
)
LEGAN["pieces"][0]["moves"] = [
    {"kind": "leap", "vectors": [[-1, 1]], "mode": "move"},
    {"kind": "leap", "vectors": [[0, 1], [-1, 0]], "mode": "capture"},
]

# Breakthrough: no royalty at all, and the game is won by arriving on the far rank.
BREAKTHROUGH = chess_frame(
    "breakthrough",
    [{"id": "P", "value": 1, "moves": [
        {"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "both"}]}],
    [{"piece": "P", "squares": [f"{f}{r}" for r in (1, 2) for f in FILES8]}],
    {"win": "none", "stalemate": "loss",
     "reach": {"squares": [f"{f}8" for f in FILES8], "mirror": True, "result": "win"},
     "max_plies": 400, "repetition_draw": 3, "no_progress_draw": 200},
)

TEN = "abcdefghij"
# Janus: the two castlings travel different distances, so only a destination file can
# describe them.
JANUS = {
    "schema_version": 2, "name": "janus", "board": {"width": 10, "height": 8},
    "pieces": standard_pieces() + [{"id": "J", "value": 7, "moves": [BISHOP, KNIGHT]}],
    "setup": {"mirror": True, "placements": [
        {"piece": "P", "squares": [f"{f}2" for f in TEN]},
        {"piece": "R", "squares": ["a1", "j1"]},
        {"piece": "J", "squares": ["b1", "i1"]},
        {"piece": "N", "squares": ["c1", "h1"]},
        {"piece": "B", "squares": ["d1", "g1"]},
        {"piece": "K", "squares": ["e1"]},
        {"piece": "Q", "squares": ["f1"]},
    ]},
    "rules": {**STANDARD_RULES,
              "promotion": {"piece": "P", "rank": "last", "to": ["Q", "J", "R", "B", "N"]},
              "castling": {"king": "K", "rook": "R",
                           "king_file": {"kingside": "i", "queenside": "b"}}},
}

# Grand chess: a promotion zone three ranks deep that a pawn may decline to enter.
GRAND = {
    "schema_version": 2, "name": "grand", "board": {"width": 10, "height": 10},
    "pieces": standard_pieces() + [
        {"id": "A", "value": 7, "moves": [BISHOP, KNIGHT]},
        {"id": "C", "value": 8, "moves": [ROOK, KNIGHT]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "R", "squares": ["a1", "j1"]},
        {"piece": "N", "squares": ["b2", "i2"]},
        {"piece": "B", "squares": ["c2", "h2"]},
        {"piece": "Q", "squares": ["d2"]},
        {"piece": "K", "squares": ["e2"]},
        {"piece": "C", "squares": ["f2"]},
        {"piece": "A", "squares": ["g2"]},
        {"piece": "P", "squares": [f"{f}3" for f in TEN]},
    ]},
    "rules": {"win": "checkmate", "stalemate": "draw",
              "promotion": {"piece": "P", "ranks": [7, 8, 9], "mandatory": False,
                            "to": ["Q", "C", "A", "R", "B", "N"]},
              "en_passant": True, "max_plies": 500, "repetition_draw": 3,
              "no_progress_draw": 100},
}
GRAND["pieces"][0]["moves"] = [
    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
     "condition": {"from_rank": 2}},
    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
]

PROBES: dict[str, tuple[dict[str, Any], dict[int, int]]] = {
    "coregal": (COREGAL, {1: 20, 2: 400, 3: 8_882, 4: 195_896}),
    "kinglet": (KINGLET, {1: 20, 2: 400, 3: 8_902, 4: 197_742}),
    "losers": (LOSERS, {1: 20, 2: 400, 3: 8_067, 4: 152_955}),
    "grasshopper": (GRASSHOPPER, {1: 28, 2: 782, 3: 22_314, 4: 635_298}),
    "chigorin": (CHIGORIN, {1: 26, 2: 416, 3: 11_408}),
    "legan": (LEGAN, {1: 8, 2: 64, 3: 724, 4: 8_138}),
    "breakthrough": (BREAKTHROUGH, {1: 22, 2: 484, 3: 11_132, 4: 256_036}),
    "janus": (JANUS, {1: 28, 2: 782, 3: 24_747, 4: 772_074}),
    "grand": (GRAND, {1: 65, 2: 4_225, 3: 259_514}),
}


@pytest.mark.parametrize(
    ("name", "depth", "expected"),
    perft_cases({name: ladder for name, (_, ladder) in PROBES.items()}),
)
def test_probe_perft_matches_the_reference_implementation(
    name: str, depth: int, expected: int
) -> None:
    spec, _ = PROBES[name]
    if name not in _ENGINES:
        _ENGINES[name] = Engine(spec)
    engine = _ENGINES[name]
    assert engine.perft(engine.initial_state(), depth, stop_at_terminal=True) == expected


# ------------------------------------------------------------------- behaviour ----


class TestMultipleRoyalty:
    def test_a_second_royal_needs_the_version_that_supports_it(self) -> None:
        under_one = dict(COREGAL, schema_version=1)
        assert any("schema_version" in error for error in validate_spec(under_one))
        assert validate_spec(COREGAL) == []

    def test_both_royals_are_guarded(self) -> None:
        """The bug schema 1 could not have: a royal the check filter never looked at."""
        engine = Engine(COREGAL)
        state = engine.make_state(
            {"e1": ("K", 0), "d1": ("Q", 0), "d8": ("R", 1), "a8": ("K", 1)}
        )
        assert engine.in_check(state), "an attacked royal queen is check"
        assert "Ke1-f1" not in engine.move_texts(state), "and may not simply be left there"


class TestHopReach:
    """A cannon and a grasshopper differ only in how far past the screen they land."""

    def hop_spec(self, screen_range: int) -> dict[str, Any]:
        move: dict[str, Any] = {
            "kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"
        }
        if screen_range:
            move["screen_range"] = screen_range
        return {
            "schema_version": 2, "name": "hop", "board": {"width": 8, "height": 8},
            "pieces": [
                {"id": "K", "value": 0, "royal": True, "moves": [KING]},
                {"id": "C", "value": 5, "moves": [move]},
            ],
            "setup": {"mirror": True, "placements": [
                {"piece": "K", "squares": ["e1"]}, {"piece": "C", "squares": ["a1"]}]},
            "rules": {"win": "checkmate", "max_plies": 400},
        }

    def landings(self, screen_range: int) -> set[str]:
        engine = Engine(self.hop_spec(screen_range))
        state = engine.make_state(
            {"a1": ("C", 0), "a3": ("C", 1), "e1": ("K", 0), "h8": ("K", 1)}
        )
        return {text[-2:] for text in engine.move_texts(state) if text.startswith("Ca1")}

    def test_a_cannon_reaches_the_whole_line_beyond_its_screen(self) -> None:
        assert {"a4", "a8"} <= self.landings(0)

    def test_a_grasshopper_lands_only_immediately_beyond(self) -> None:
        landings = self.landings(1)
        assert "a4" in landings
        assert "a8" not in landings


def test_a_promoting_double_step_still_opens_no_en_passant_window() -> None:
    """The schema-1 regression, restated against the generalised promotion tables."""
    spec = {
        "schema_version": 2, "name": "mid-board promotion",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "P", "value": 1, "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2,
                 "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral",
                 "mode": "capture"}]},
            {"id": "Q", "value": 9, "moves": [QUEEN]},
            {"id": "K", "value": 0, "royal": True, "moves": [KING]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "P", "squares": ["a2"]}, {"piece": "K", "squares": ["e1"]}]},
        "rules": {"win": "checkmate",
                  "promotion": {"piece": "P", "rank": 4, "to": ["Q"]},
                  "en_passant": True, "max_plies": 400},
    }
    engine = Engine(spec)
    state = engine.make_state(
        {"b3": ("P", 0), "a5": ("P", 1), "e1": ("K", 0), "h8": ("K", 1)}
    )
    after = engine.apply(state, engine.find_move(state, "Pb3-b5=Q"))
    assert not [text for text in engine.move_texts(after) if "e.p." in text]


# --------------------------------------------------------------- integration ----
#
# Two failures that perft could never have caught, because neither is about counting
# moves. Both were found by playing the games rather than by measuring them.


@pytest.mark.parametrize("name", ["xiangqi", "shogi", "crazyhouse", "dobutsu", "atomic"])
def test_the_evaluator_reads_every_move_table(name: str) -> None:
    """Piece values are derived from the move tables, hop entries included.

    The hop table grew a fourth field when grasshoppers arrived, and the evaluator
    unpacked exactly three. Every agent shares this evaluator, so the failure was not
    a wrong number but a crash on any ruleset with a cannon in it.
    """
    from chess_variant_sandbox.agents.evaluation import Evaluator

    engine = engine_for(name)
    values = Evaluator.for_engine(engine).values
    assert len(values) == engine.spec.n_types
    assert any(value > 0 for value in values)


def test_a_forbidden_drop_mate_does_not_ask_itself_the_same_question_forever() -> None:
    """The drop-mate rule needs the opponent's escapes found without the rule applied.

    Otherwise "is the opponent mated?" asks the same of each of their pawn drops, and
    of each of the replies to those, with nothing to stop it. Perft never noticed
    because the positions it reaches are too shallow; a random playout hangs at once.
    """
    engine = engine_for("shogi")
    index = {piece: i for i, piece in enumerate(engine.spec.piece_ids)}
    held = [0] * engine.spec.n_types
    held[index["P"]] = 2
    state = engine.make_state(
        {"e1": ("K", 0), "e9": ("K", 1), "d5": ("R", 0)}, side_to_move=0
    ).with_(hands=(tuple(held), tuple(held)))
    assert engine.legal_moves(state), "the position is playable"
    assert engine.terminal(state) is None


_EVERY_RULESET = sorted(RULESETS.glob("*.json")) + sorted((RULESETS.parent / "archive").glob("*.json"))


def test_the_walk_below_reads_the_whole_corpus() -> None:
    assert len(_EVERY_RULESET) > 40, "the corpus moved; this test is reading the wrong tree"


# One test per committed spec (Q749): the walk was one test of 4 s in the gate.
@pytest.mark.parametrize("path", _EVERY_RULESET, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_every_committed_ruleset_plays_a_game_through(path: Path) -> None:
    """The whole stack, not the generator alone: compile, the core, movegen, apply,
    terminal detection and the incremental key.

    Cheap on purpose -- twenty random plies each. It is a smoke alarm, not a tournament,
    and it exists because a ruleset can pass every perft and still be unplayable once a
    game is under way. It is the fast suite's one walk over every committed spec, so it
    also stands in for two checks that used to walk the corpus on their own (Q377): that
    the core can play each ruleset (`Engine` raises when it cannot), and that the
    validator's reading of whether royalty can multiply is the compiler's
    (`test_multi_royal.py`'s `TestTheValidatorReadsTheSameRoutes` holds every route).

    The archive is walked too, and the reason is not completeness. Every ruleset with a
    strike rule lives there, and a strike is the one thing in the schema that changes no
    move count anywhere -- so the corpus this walk covered was exactly the corpus where
    the walk could not have caught anything a perft would not.
    """
    import random

    spec = load_spec(path)
    engine = Engine(spec)
    # One rule, two vocabularies: a drift either way changes which specs schema 1 admits.
    assert royalty_can_multiply(spec) is engine.spec.multi_royal, (
        f"{path.name}: the validator and the compiler disagree on whether royalty multiplies"
    )
    rng = random.Random(7)
    state = engine.initial_state()
    if engine.simultaneous:
        # Schema 26 (Q918e, its first library specs): ten rounds, both players' moves at once.
        # A spy game (schema 27, Q935f's showcases) alternates until a spy falls, so each
        # step asks the core which it is; in any other round game every step is a round.
        for _ in range(10):
            if engine.terminal(state) is not None:
                break
            if not engine.is_round(state):
                state = engine.apply(state, rng.choice(engine.legal_moves(state)))
                assert state.key == engine._compute_hash(state), f"{path.stem} hash drifted"
                continue
            first = rng.choice(engine.legal_moves_for(state, 0))
            second = rng.choice(engine.legal_moves_for(state, 1))
            state = engine.apply_round(state, first, second)
            assert state.key == engine._compute_hash(state), f"{path.stem} hash drifted"
        return
    for _ in range(20):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        state = engine.apply(state, rng.choice(moves))
        assert state.key == engine._compute_hash(state), f"{path.stem} hash drifted"


# ------------------------------------------------------- a stated ending's result ----
#
# ``ENDING_RESULTS`` promises three results, and ``validate_spec`` accepts all three for
# ``reach``, ``extinction`` and ``bare_royal``. The engine owed all three and delivered
# two: the result crossed the boundary as a boolean, so "draw" -- having nowhere to go --
# came out as whichever of win and loss the flag was not. Each rule failed differently
# (``bare_royal`` switched itself off, the other two inverted), and no committed ruleset
# says "draw", which is exactly why nothing noticed.
#
# Parametrised over all three results rather than testing the draw alone, because the
# interesting failure is a direction flip and a draw-only test cannot see one.


def _stated_ending(**rules: Any) -> dict[str, Any]:
    """A 5x5 board with a royal ``K`` and a plain ``S``, plus one stated ending."""
    return {
        "schema_version": 2,
        "name": "stated-ending",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {
                "id": "K",
                "royal": True,
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
            },
            {"id": "S", "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "S", "squares": ["a1"]},
            ],
        },
        "rules": {"win": "checkmate", **rules},
    }


# player 0 holds a soldier and player 1 does not, so player 1 is both the side that has
# run out and the side reduced to royalty alone. Both rules name that side as the actor.
_ACTOR_IS_PLAYER_ONE = {"c1": ("K", 0), "a1": ("S", 0), "c5": ("K", 1)}


@pytest.mark.parametrize(
    ("result", "winner"), [("win", 0), ("draw", None), ("loss", 1)]
)
def test_a_reach_result_is_honoured_in_all_three_directions(
    result: str, winner: int | None
) -> None:
    """Stated for the player who arrives, so a win is a win for them."""
    spec = _stated_ending(
        reach={"piece": "S", "squares": ["c5"], "mirror": False, "result": result}
    )
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.make_state({"c1": ("K", 0), "c5": ("S", 0), "a5": ("K", 1)})
    outcome = engine.terminal(state)
    assert outcome is not None
    assert outcome.reason == "reached"
    assert outcome.winner == winner


@pytest.mark.parametrize(
    ("result", "winner"), [("win", 1), ("draw", None), ("loss", 0)]
)
def test_an_extinction_result_is_honoured_in_all_three_directions(
    result: str, winner: int | None
) -> None:
    """Stated for the player who runs out: the horde loses, antichess wins."""
    spec = _stated_ending(
        extinction={"pieces": ["S"], "min_count": 1, "result": result}
    )
    assert validate_spec(spec) == []
    engine = Engine(spec)
    outcome = engine.terminal(engine.make_state(_ACTOR_IS_PLAYER_ONE))
    assert outcome is not None
    assert outcome.reason == "extinction"
    assert outcome.winner == winner


@pytest.mark.parametrize(
    ("result", "winner"), [("win", 1), ("draw", None), ("loss", 0)]
)
def test_a_bare_royal_result_is_honoured_in_all_three_directions(
    result: str, winner: int | None
) -> None:
    """Stated for the player stripped to royalty alone: shatranj's loss.

    ``"draw"`` is the case that disabled the rule outright, because the off switch and
    the drawn result used to be the same integer.
    """
    spec = _stated_ending(bare_royal=result)
    assert validate_spec(spec) == []
    engine = Engine(spec)
    outcome = engine.terminal(engine.make_state(_ACTOR_IS_PLAYER_ONE))
    assert outcome is not None
    assert outcome.reason == "bare_royal"
    assert outcome.winner == winner


@pytest.mark.parametrize("bared", [0, 1])
def test_every_text_explaining_bare_royal_names_the_side_the_core_decides_for(
    bared: int,
) -> None:
    """The core gives `bare_royal`'s result to the bared side; every text must say so.

    The polarity comes from the core. Under `"win"` the winner is whichever side the
    core found holding only its royal, from either seat, so the result belongs to the
    side reduced, and `BARE_ROYAL_HOLDER` names that side. The describer, the schema
    description and the proposer's preamble each state the key through that phrase,
    so none of them can name the reducer instead.
    """
    from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE

    other = 1 - bared
    spec = _stated_ending(bare_royal="win")
    engine = Engine(spec)
    position = {"c1": ("K", other), "a1": ("S", other), "c5": ("K", bared)}
    outcome = engine.terminal(engine.make_state(position))
    assert outcome is not None
    assert outcome.reason == "bare_royal"
    assert outcome.winner == bared
    assert "reduced to royalty alone" in BARE_ROYAL_HOLDER

    schema_text = rule_spec_schema()["properties"]["rules"]["properties"]["bare_royal"]
    texts = {
        "describe_spec": describe_spec(spec),
        "rule_spec_schema": schema_text["description"],
        "SCHEMA_PREAMBLE": SCHEMA_PREAMBLE,
    }
    for source, text in texts.items():
        assert f"result for {BARE_ROYAL_HOLDER}" in text, source


def test_no_stated_ending_at_all_leaves_the_game_running() -> None:
    """The companion to the draw cases: "off" must still be distinguishable.

    Same position, same pieces, no ending declared. If ``bare_royal: "draw"`` and
    ``bare_royal`` absent produced the same table entry, the three tests above would
    pass for the wrong reason.
    """
    engine = Engine(_stated_ending())
    assert engine.terminal(engine.make_state(_ACTOR_IS_PLAYER_ONE)) is None


# ------------------------------------------------- schema 3, inline probes -------
#
# Two features arrived with schema 3 and only one of them has a committed ruleset
# exercising it. `docs/reference/rps-versions.md` records that version 6 of the corpus
# jumps over its *own* pieces, and that its army size is not stated anywhere in the
# source -- so the ruleset is not committed, because inventing a number and filing it as
# a labelled negative would be inventing data. The primitive is still built, and this is
# where it is held down.


def _friendly_screen_spec(owner: str) -> dict[str, Any]:
    """A 5x5 board with a vaulter whose screen ownership is a parameter."""
    return {
        "schema_version": 3,
        "name": f"vaulter over {owner}",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {
                "id": "K",
                "royal": True,
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
            },
            {
                "id": "V",
                "value": 5,
                "moves": [
                    {
                        "kind": "hop",
                        "vectors": [[1, 0]],
                        "symmetry": "rotate4",
                        "mode": "both",
                        "screen": owner,
                    }
                ],
            },
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "V", "squares": ["a1"]},
            ],
        },
        "rules": {"win": "royal_capture"},
    }


@pytest.mark.parametrize(
    ("owner", "over_friend", "over_enemy"),
    [("any", True, True), ("own", True, False), ("enemy", False, True)],
)
def test_a_hop_can_be_told_whose_piece_it_may_jump(
    owner: str, over_friend: bool, over_enemy: bool
) -> None:
    """`hop` never cared whose piece the screen was, and a design wanted it to.

    The same position twice, differing only in who owns the piece being jumped. A hop
    still stops at the *first* occupied square whichever way this is set: a screen it may
    not vault blocks the line rather than being skipped, exactly as a piece a slide may
    not capture blocks the slide.
    """
    engine = Engine(_friendly_screen_spec(owner))
    assert validate_spec(_friendly_screen_spec(owner)) == []

    friend = engine.make_state(
        {"a1": ("V", 0), "a2": ("V", 0), "a3": ("K", 1), "e5": ("K", 0)}, side_to_move=0
    )
    enemy = engine.make_state(
        {"a1": ("V", 0), "a2": ("V", 1), "a3": ("K", 1), "e5": ("K", 0)}, side_to_move=0
    )
    assert ("Va1xa3" in engine.move_texts(friend)) is over_friend
    assert ("Va1xa3" in engine.move_texts(enemy)) is over_enemy


@pytest.mark.parametrize("owner", ["own", "enemy"])
def test_a_screen_it_may_not_jump_is_not_giving_check(owner: str) -> None:
    """The other half, and the half a move list cannot show.

    `attackers_of` is inverted out of the same tables, so a hop's screen ownership has
    to be honoured on the *attack* path as well -- otherwise a king would be in check
    from a piece with no legal move to reach it, which is a different game and not a
    failing assertion. Bastion earns its keep here for the same reason: a hop attack is
    the one thing that turns on the awkward branch of the legality filter.
    """
    engine = Engine(_friendly_screen_spec(owner))
    # Player 1's vaulter on a5, player 0's king on a3, one piece between them on a4.
    # The screen belongs to player 1, so an "own"-screened hop attacks and an
    # "enemy"-screened one does not.
    state = engine.make_state(
        {"a5": ("V", 1), "a4": ("V", 1), "a3": ("K", 0), "e1": ("K", 1)},
        side_to_move=0,
    )
    in_check = engine.in_check(state, 0)
    assert in_check is (owner == "own"), f"{owner}: in_check={in_check}"


def test_a_schema_two_spec_may_not_reach_for_schema_three() -> None:
    """The version stays an honest description of what the rules need.

    ``schema_version`` is part of the mechanics hash, so two specs whose rules differ
    only in a feature one of them was not entitled to use must not be able to collide.
    """
    spec = _friendly_screen_spec("own")
    spec["schema_version"] = 2
    errors = validate_spec(spec)
    assert any("schema 3" in error for error in errors), errors

    matrix = _friendly_screen_spec("any")
    matrix["schema_version"] = 2
    matrix["pieces"][1]["captures"] = ["K"]
    errors = validate_spec(matrix)
    assert any("captures" in error and "schema 3" in error for error in errors), errors


# ------------------------------------------- schema 3, the score-comparison ending ---
#
# `max_plies` and `no_progress_draw` were always a draw, which means a game with a real
# advantage in it could still end level. That is the Go lesson stated as a missing
# sentence: "a small advantage will straight up win you the game" is a claim about
# scoring, and without one every long game is drawn however far ahead someone is.


def _scored_spec(count: str, region: list[str] | None = None) -> dict[str, Any]:
    """A 5x5 board whose clock decides the game instead of drawing it."""
    scoring: dict[str, Any] = {"count": count}
    if region is not None:
        scoring["region"] = region
    return {
        "schema_version": 3,
        "name": f"decided on {count}",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {
                "id": "K",
                "royal": True,
                "value": 0,
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
            },
            {
                "id": "S",
                "value": 1,
                "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}],
            },
            {
                "id": "G",
                "value": 9,
                "moves": [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"}],
            },
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "S", "squares": ["a1"]},
                {"piece": "G", "squares": ["e1"]},
            ],
        },
        "rules": {"win": "checkmate", "max_plies": 2, "scoring": scoring},
    }


class TestTheClockCanDecideInsteadOfDrawing:
    def test_material_counts_the_declared_values(self) -> None:
        """Declared values, not the evaluator's derived ones.

        A rule has to be objective: the evaluator is an agent's *opinion* about what a
        piece is worth, and two agents holding different opinions must not be able to
        disagree about who won.
        """
        engine = Engine(_scored_spec("material"))
        # Three cheap pieces against one expensive one: 3 against 9.
        state = engine.make_state(
            {"c1": ("K", 0), "a1": ("S", 0), "a2": ("S", 0), "a3": ("S", 0),
             "e5": ("K", 1), "e4": ("G", 1)},
            side_to_move=0, ply=2,
        )
        outcome = engine.terminal(state)
        assert outcome is not None
        assert outcome.reason == "scored"
        assert outcome.winner == 1, "nine beats three"

    def test_pieces_counts_heads_and_reaches_the_opposite_verdict(self) -> None:
        """The same position, the other count, the other winner -- which is the check
        that the count is read at all rather than material being hardcoded."""
        engine = Engine(_scored_spec("pieces"))
        state = engine.make_state(
            {"c1": ("K", 0), "a1": ("S", 0), "a2": ("S", 0), "a3": ("S", 0),
             "e5": ("K", 1), "e4": ("G", 1)},
            side_to_move=0, ply=2,
        )
        outcome = engine.terminal(state)
        assert outcome is not None
        assert outcome.winner == 0, "four heads beat two"

    def test_a_region_is_reflected_for_the_second_player(self) -> None:
        """Hold-the-centre is symmetric and hold-their-half is not, and the compiler
        cannot tell which a spec meant without the orientation. The region is written
        for the first player and reflected, like every other region in the DSL."""
        engine = Engine(_scored_spec("region", ["a1", "b1"]))
        # Player 0 holds one of its own two squares; player 1 holds both of the
        # reflected pair.
        state = engine.make_state(
            {"c3": ("K", 0), "a1": ("S", 0), "c5": ("K", 1), "a5": ("S", 1), "b5": ("S", 1)},
            side_to_move=0, ply=2,
        )
        outcome = engine.terminal(state)
        assert outcome is not None
        assert outcome.winner == 1, "two held squares beat one"

    def test_a_level_count_is_still_a_draw(self) -> None:
        engine = Engine(_scored_spec("material"))
        state = engine.make_state(
            {"c1": ("K", 0), "a1": ("S", 0), "c5": ("K", 1), "a5": ("S", 1)},
            side_to_move=0, ply=2,
        )
        outcome = engine.terminal(state)
        assert outcome is not None
        assert outcome.winner is None
        assert outcome.reason == "ply_limit", "a drawn score keeps the clock's own reason"

    def test_the_score_is_not_compared_before_the_clock_runs_out(self) -> None:
        """The reason this is not part of `schema_two_ending`.

        That runs before the draw clocks and therefore at every ply. A score compared
        there would end the game at ply zero and award it to whoever the opening
        position happens to favour, which is not a rule anybody wants.
        """
        spec = _scored_spec("material")
        spec["rules"]["max_plies"] = 200
        engine = Engine(spec)
        lopsided = engine.make_state(
            {"c1": ("K", 0), "a1": ("S", 0), "e5": ("K", 1), "e4": ("G", 1)},
            side_to_move=0,
        )
        assert engine.terminal(lopsided) is None, "wildly unequal, and not over"

    def test_the_no_progress_clock_decides_it_too(self) -> None:
        """Both clocks, because a game that shuffles forever and a game that runs long
        are the same failure wearing two names."""
        spec = _scored_spec("material")
        spec["rules"]["max_plies"] = 500
        spec["rules"]["no_progress_draw"] = 4
        engine = Engine(spec)
        state = engine.make_state(
            {"c1": ("K", 0), "e5": ("K", 1), "e4": ("G", 1)},
            side_to_move=0, halfmove=4,
        )
        outcome = engine.terminal(state)
        assert outcome is not None
        assert outcome.reason == "scored" and outcome.winner == 1

    def test_repetition_is_left_alone(self) -> None:
        """A game that has repeated three times is drawn by agreement rather than by
        exhaustion. Handing it to whoever is ahead would make repetition a *tool* for
        converting a lead rather than a way to stop, which is a different rule and one
        nobody asked for."""
        spec = _scored_spec("material")
        spec["rules"]["max_plies"] = 500
        spec["rules"]["repetition_draw"] = 2
        engine = Engine(spec)
        state = engine.make_state(
            {"c1": ("K", 0), "e5": ("K", 1), "e4": ("G", 1)}, side_to_move=0
        )
        repeated = state.with_(repetition=(state.key, state.key))
        outcome = engine.terminal(repeated)
        assert outcome is not None
        assert outcome.winner is None
        assert outcome.reason == "repetition"

    def test_a_scoring_rule_costs_something_in_rule_mass(self) -> None:
        """An uncounted feature inflates `rule_yield`, which has rule mass in its
        denominator -- so a free rule is not merely unpriced, it is rewarded."""
        from chess_variant_sandbox.spec.complexity import measure_complexity

        with_scoring = measure_complexity(_scored_spec("material"))
        without = dict(_scored_spec("material"))
        without["rules"] = {k: v for k, v in without["rules"].items() if k != "scoring"}
        assert with_scoring.rule_mass > measure_complexity(without).rule_mass

    def test_the_validator_refuses_a_region_without_a_reason_for_one(self) -> None:
        assert any(
            "only means something when count is" in error
            for error in validate_spec(_scored_spec("material", ["a1"]))
        )
        assert any(
            "needs rules.scoring.region" in error
            for error in validate_spec(_scored_spec("region"))
        )

    def test_a_schema_two_spec_may_not_reach_for_a_scoring_rule(self) -> None:
        spec = _scored_spec("material")
        spec["schema_version"] = 2
        assert any("schema 3" in error for error in validate_spec(spec))


# ------------------------------------------- schema 4, more than one move per turn ----
#
# The last thing the RPS sequence tried and the one its designer discarded: "having two
# moves per turn was kind of pointless... some rules sound great on paper, but don't work
# very well in practice." He kept half of it -- a 1-2-2-2 opening ramp, so that whoever
# starts is not a whole move ahead forever -- and `first_player_advantage` is a component
# this project already measures, so that half is testable here in a way it was not for
# him. The measurement is at the bottom of this block.
#
# No committed ruleset uses it. Six of the seven RPS versions are expressible and none of
# them needs it; version 2 *trialled* it and the transcript does not say the army it was
# trialled with, so committing one would be inventing data.


def _turned_spec(**turn: Any) -> dict[str, Any]:
    """Gardner with a longer turn, so the counter is the only thing that changed."""
    spec = json.loads(
        (RULESETS / "gardner.json").read_text(encoding="utf-8")
    )
    spec["schema_version"] = 4
    spec["rules"]["turn"] = turn
    return spec


class TestATurnCanBeLongerThanOneMove:
    def test_the_side_flips_when_the_turn_runs_out_and_not_before(self) -> None:
        import random

        engine = Engine(_turned_spec(moves=2))
        state = engine.initial_state()
        rng = random.Random(2)
        seen = []
        for _ in range(8):
            seen.append((state.side_to_move, state.remaining))
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        assert seen == [
            (0, 2), (0, 1), (1, 2), (1, 1), (0, 2), (0, 1), (1, 2), (1, 1)
        ]

    def test_the_opening_ramp_shortens_the_first_turn(self) -> None:
        """The 1-2-2-2 pattern, which is the half the designer kept."""
        import random

        engine = Engine(_turned_spec(moves=2, ramp=[1]))
        state = engine.initial_state()
        rng = random.Random(2)
        seen = []
        for _ in range(7):
            seen.append((state.side_to_move, state.remaining))
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        assert seen == [(0, 1), (1, 2), (1, 1), (0, 2), (0, 1), (1, 2), (1, 1)]

    @pytest.mark.parametrize(
        ("turn", "ladder"),
        [
            ({"moves": 2}, {1: 7, 2: 67, 3: 511}),
            ({"moves": 2, "ramp": [1]}, {1: 7, 2: 53, 3: 512}),
        ],
    )
    def test_perft_counts_a_turn_rather_than_a_move(
        self, turn: dict[str, Any], ladder: dict[int, int]
    ) -> None:
        """The strong test of make and unmake with a counter in them.

        Depth 2 here is two plies by the *same* player, which is the case a search walks
        and takes back thousands of times a second. The ramp changes the count at depth
        2 and not at depth 1, which is exactly what a one-move first turn should do.
        """
        engine = Engine(_turned_spec(**turn))
        state = engine.initial_state()
        for depth, expected in ladder.items():
            assert engine.perft(state, depth, stop_at_terminal=True) == expected

    def test_moves_remaining_is_part_of_the_position(self) -> None:
        """Two positions with the same pieces and different moves left are different
        positions -- one side is about to move again -- so the key has to say so, or a
        repetition test calls them the same."""
        engine = Engine(_turned_spec(moves=2))
        start = engine.initial_state()
        mid_turn = start.with_(remaining=1)
        assert engine._compute_hash(start) != engine._compute_hash(mid_turn)

    def test_a_turn_of_one_move_is_the_game_that_was_already_there(self) -> None:
        """Declared explicitly, it compiles away: `moves: 1` is what every ruleset has
        always done, and a spec that says so must not become a different game."""
        from chess_variant_sandbox.engine.compile import compile_spec

        assert compile_spec(_turned_spec(moves=1)).turn is None
        plain = json.loads((RULESETS / "gardner.json").read_text(encoding="utf-8"))
        engine = Engine(_turned_spec(moves=1))
        assert engine.perft(engine.initial_state(), 3) == Engine(plain).perft(
            Engine(plain).initial_state(), 3
        )

    @pytest.mark.slow
    def test_a_larger_search_still_plays_better_when_a_side_moves_twice(self) -> None:
        """The half that is not bookkeeping.

        `agents.rs` and `probe.rs` negate the child's value and flip the alpha-beta
        window at every ply, because negamax assumes the side alternates. It does not,
        once a turn is longer than one move -- and a search that negated anyway would
        maximise the opponent's position on every second ply while every perft still
        passed. What that failure looks like from outside is a ladder that stops
        ordering, so that is what this checks.
        """
        from chess_variant_sandbox.metrics.components import _ladder_monotonicity
        from chess_variant_sandbox.tournament import TournamentConfig, rate, run_tournament

        # The search this was written against, before Q502i's threat-aware quiescence:
        # two games a pairing on a game the first mover wins nearly always is a coin
        # flip between the two minimax rungs, and it came down the other way under the
        # new default (100 over 400 on both seats' worth of first-mover wins, at 2 and at
        # 8 games a pairing). The threats search's negation under a long turn is held
        # move for move against its oracle in `test_search_repairs.py`.
        roster = (
            "random",
            "greedy",
            "minimax:nodes=100,threats=0",
            "minimax:nodes=400,threats=0",
        )
        result = run_tournament(
            _turned_spec(moves=2),
            TournamentConfig(agents=roster, games_per_pairing=2, seed=0, workers=0),
        )
        assert _ladder_monotonicity(rate(result.records)) == 1.0

    @pytest.mark.slow
    def test_the_opening_ramp_removes_the_advantage_it_was_invented_to_remove(
        self,
    ) -> None:
        """The designer's claim, measured rather than asserted -- and it did not survive
        a stronger search.

        He invented the 1-2-2-2 ramp because with two moves a turn whoever starts is a
        whole move ahead forever. Stage 16 measured that and agreed by a wide margin.
        Phase C gave the search bounds, quiescence and ordering beyond captures, and the
        same three arms now read:

            one move per turn          0.569  ->  0.819
            two moves per turn         0.931  ->  0.667
            two moves with the ramp    0.597  ->  0.222

        **The ordering inverted.** Against a deeper search, an unramped double-move turn
        favours the first seat *less* than an ordinary one does, and the ramp does not
        level it -- it over-corrects, handing the second seat a large edge. Stage 16's
        landslide was substantially a property of a search that could not defend against
        the double move; what a stronger one shows is a different game.

        Recorded rather than resolved, which is how stage 16 handled the same
        disagreement from the other side: it noted that the agents and the human were
        reading different things. They still are, and the agents have changed their mind.

        What is asserted below is the part that is durable -- the rule moves first-seat
        advantage a long way, and the ramp moves it downward from the unramped game --
        rather than the magnitudes, which are what a search change is free to move.

        The thresholds below are loose because the numbers are self-play over about a
        hundred games and the claim being tested is a landslide, not a lean.
        """
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament import TournamentConfig, rate, run_tournament

        # The roster the reported numbers came from. `first_player_score` is read off
        # self-play only, so three budgets at twelve games is thirty-six games and two
        # budgets at eight would be sixteen -- which is not enough to tell a landslide
        # from a lean, and said so when it was tried.
        roster = ("minimax:nodes=200", "minimax:nodes=800", "minimax:nodes=3200")

        def first_seat(spec: dict[str, Any]) -> float:
            result = run_tournament(
                spec,
                TournamentConfig(agents=roster, games_per_pairing=12, seed=0, workers=0),
            )
            measured = compute_components(spec, result.records, rate(result.records))
            return measured.raw["first_player_score"]

        plain = first_seat(
            json.loads((RULESETS / "gardner.json").read_text(encoding="utf-8"))
        )
        unramped = first_seat(_turned_spec(moves=2))
        ramped = first_seat(_turned_spec(moves=2, ramp=[1]))

        # The turn rule is not a cosmetic change to who starts: whatever the search, it
        # moves first-seat advantage a long way from where the plain game leaves it.
        assert abs(unramped - plain) > 0.1 or abs(ramped - plain) > 0.1, (
            f"a longer turn changed nothing: plain {plain}, two moves {unramped}, "
            f"ramped {ramped}"
        )
        # And the ramp does what it was invented to do, in direction if not in size: it
        # takes the advantage *away* from whoever moves first.
        assert ramped < unramped, (
            f"the ramp should cost the first seat, not help it: {ramped} against "
            f"{unramped} unramped"
        )

    def test_a_turn_that_says_nothing_new_costs_no_rule_mass(self) -> None:
        """`rule_mass` is a denominator, so a rule that is not there must not be counted.

        The turn rule is the first one whose *value* decides whether there is a rule at
        all: `{"moves": 1}` is what every game already does, and so is a ramp of ones.
        `_RULE_DEFAULTS` compares for equality and cannot say that, which is why
        `_RULE_ACTIVE` exists.
        """
        from chess_variant_sandbox.spec.complexity import measure_complexity

        plain = json.loads((RULESETS / "gardner.json").read_text(encoding="utf-8"))
        baseline = measure_complexity(plain).special_rules
        assert measure_complexity(_turned_spec(moves=1)).special_rules == baseline
        assert (
            measure_complexity(_turned_spec(moves=1, ramp=[1, 1])).special_rules
            == baseline
        )
        assert measure_complexity(_turned_spec(moves=2)).special_rules == baseline + 1.0
        # The ramp parameterises the rule rather than adding one, exactly as
        # `screen_range` parameterises a hop.
        assert (
            measure_complexity(_turned_spec(moves=2, ramp=[1])).special_rules
            == baseline + 1.0
        )


# --------------------------------------------------- castling onto an occupied square

_SWAP_KING = {
    "kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both",
}
_SWAP_ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}


def _castle_swap_spec(free_rook: str) -> dict[str, Any]:
    """`must_capture` beside a castling whose king lands on its own rook.

    Chess960's shape at a size that fits on one screen. The white king stands on b1 and
    its rook on c1, and the rule sends the king to c and the rook to b -- so the two
    swap, the king's destination is occupied, and the only thing that varies between the
    two positions is where white's *other* rook stands.
    """
    return {
        "schema_version": 2,
        "name": "castle-swap",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "value": 0, "royal": True, "moves": [_SWAP_KING]},
            {"id": "R", "value": 5, "moves": [_SWAP_ROOK]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["b1"]},
            {"piece": "R", "player": 0, "squares": ["c1", free_rook]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "R", "player": 1, "squares": ["d5"]},
        ]},
        "rules": {
            "win": "checkmate",
            "stalemate": "draw",
            "must_capture": True,
            "castling": {"king": "K", "rook": "R",
                         "king_file": {"kingside": "c", "queenside": "a"}},
            "max_plies": 80,
        },
    }


class TestCastlingIsNotACapture:
    """A file-based castling can end on an occupied square, and it still takes nothing.

    `must_capture` used to read a move as a capture whenever its target square was
    occupied, which is true of this castling and of nothing else the rule meant to
    catch -- so castling would have been compulsory wherever it was available, a rule
    nobody wrote. `make` read the same square the same way and moved the two pieces in
    an order that wrote one over the other.
    """

    def test_the_rule_neither_forces_nor_forbids_it(self) -> None:
        spec = _castle_swap_spec("a4")
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = engine.initial_state()
        texts = engine.move_texts(state)
        # White has no capture here, so `must_capture` binds on nothing and the castling
        # stands beside every quiet move rather than replacing them all.
        assert "Kb1-c1 (castle)" in texts
        assert {"Kb1-a1", "Rc1-c2", "Ra4-b4"} <= set(texts)

    def test_a_capture_on_offer_removes_it(self) -> None:
        spec = _castle_swap_spec("d4")
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = engine.initial_state()
        # The castling is generated; what removes it is the rule, and only because a
        # real capture is on offer.
        pseudo = {engine.describe_move(state, m) for m in engine.pseudo_legal_moves(state)}
        assert "Kb1-c1 (castle)" in pseudo
        assert engine.move_texts(state) == ["Rd4xd5"]

    def test_the_two_pieces_swap_rather_than_one_eating_the_other(self) -> None:
        spec = _castle_swap_spec("a4")
        engine = Engine(spec)
        state = engine.initial_state()
        after = engine.apply(state, engine.find_move(state, "Kb1-c1 (castle)"))
        king = engine.spec.piece_ids.index("K") + 1
        rook = engine.spec.piece_ids.index("R") + 1
        assert after.board[2] == king, "the king stands on c1"
        assert after.board[1] == rook, "its rook stands on b1"
        # Nothing left the board, so the no-progress clock has not been reset either.
        assert sum(1 for v in after.board if v) == sum(1 for v in state.board if v)
        assert after.halfmove == state.halfmove + 1


# ------------------------------------------------ Q922: what schema 1 may not use, per piece


class TestASchemaOnePieceMayNotUseSchemaTwo:
    """`_validate_version_one_stays_within_version_one`, over inline pieces.

    It reads only the pieces' own keys, so each case is a list of pieces and the one
    sentence it must produce (or nothing). The junk entries are what `validate_spec`
    feeds it while it is still collecting errors about them.
    """

    TAIL = ", added in schema 2; declare \"schema_version\": 2 to use it"
    LEAP = {"kind": "leap", "vectors": [[1, 0]]}

    CASES = {
        "a plain piece": ([{"id": "P", "moves": [LEAP]}], []),
        "a piece confined to a region": (
            [{"id": "P", "region": "x", "moves": [LEAP]}], ["a piece uses region"]),
        "a move with a screen range": (
            [{"id": "P", "moves": [{**LEAP, "screen_range": 2}]}], ["a piece uses screen_range"]),
        "a lame move": (
            [{"id": "P", "moves": [{"kind": "lame", "vectors": [[1, 2]]}]}],
            ["a piece uses the 'lame' move kind"]),
        "a condition with a region": (
            [{"id": "P", "moves": [{**LEAP, "condition": {"region": "x"}}]}],
            ["a piece uses region"]),
        "a condition that is not a dict": (
            [{"id": "P", "moves": [{**LEAP, "condition": "region"}]}], []),
        "the second move of the second piece": (
            [{"id": "A", "moves": [LEAP]},
             {"id": "B", "moves": [LEAP, {**LEAP, "screen_range": 1}]}],
            ["a piece uses screen_range"]),
        "several uses are one sentence, sorted": (
            [{"id": "P", "region": "x", "moves": [{"kind": "lame", "vectors": [[1, 2]]},
                                                  {**LEAP, "screen_range": 1}]}],
            ["a piece uses region, screen_range, the 'lame' move kind"]),
        "junk pieces and junk moves are skipped": (
            ["junk", None, {"id": "P", "moves": ["junk", None, {**LEAP, "screen_range": 1}]}],
            ["a piece uses screen_range"]),
        "a region on a piece whose moves are not a list is not read": (
            [{"id": "P", "region": "x", "moves": "x"}], []),
        "a region on a piece with no moves is not read": ([{"id": "P", "region": "x"}], []),
    }

    @pytest.mark.parametrize("label", sorted(CASES))
    def test_what_is_reported(self, label: str) -> None:
        from chess_variant_sandbox.spec.schema import (
            _validate_version_one_stays_within_version_one,
        )

        pieces, wanted = self.CASES[label]
        errors: list[str] = []
        _validate_version_one_stays_within_version_one({"pieces": pieces}, errors)
        assert errors == [w + self.TAIL for w in wanted]
