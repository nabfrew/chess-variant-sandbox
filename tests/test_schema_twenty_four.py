"""Schema 24 (Q867b): fold portals revised, `docs/archive/plan-portals-2.md` §3R.

A portal group's ``"kind": "fold"`` makes it a fold: not a piece, a square a piece of either
side may stand in and be taken in. A move that ends on an empty fold that lets the piece in is
two moves: stay there, or go through and end on the square of the end it leads to, judged as a
landing there (capturing an enemy piece standing on it). Nothing else about a move changes: a
fold is floor to a move that passes over it. Schema 22 (Q843c) introduced folds with other
rules; they were deleted, not kept behind the old number, so a fold at 22 or 23 is refused.

The core's moves and checks are held against `tests/fold_oracle.py`, a declared test oracle
that reads the spec's descriptors and walks them square by square, over random walks of five
test specs (a fold pair on Gardner, a fold ring on chess, a fold beside a momentum pair, a fold
pair on chess a pawn's diagonal goes through, a funnel on chess) and on every hand-built
position below, one test per rule of §3R.1.

`TestTheBumpMovedNothing` is the inertness proof: no committed or archived hash moved, no game
without a fold sends a fold key, and the momentum games' perfts are still the schema 20 core's
(`test_schema_twenty_one.FROZEN_PERFT`, run there).
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from fold_oracle import FoldOracle, Folds, Move
from test_schema_nine import FROZEN
from test_schema_nineteen import STEP, _core_portal_moves, _core_transits, _position, _unmoved
from test_schema_nineteen import _with_portals as _with_groups

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_NORMAL,
    MOVE_PROMOTION,
    move_fold,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_rules
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION
from tests.feature_inventory import _schema_twenty_four_uses, _schema_twenty_two_uses

ROOT = Path(__file__).resolve().parents[1]
FAST_BUDGET_NODES = 150_000


def _schema(spec: dict[str, Any]) -> dict[str, Any]:
    spec["schema_version"] = 24
    spec["name"] = spec["name"].replace("+portals", "+folds")
    return spec


def fold_pair() -> dict[str, Any]:
    """Gardner with a fold pair on the empty third rank, b3 and d3; either player may move
    the end on b3 a step, onto a piece or not."""
    return _schema(_with_groups(
        "gardner",
        [{"orange": "b3", "blue": "d3", "kind": "fold",
          "moved_by": {"orange": [0, 1]}}],
        [STEP],
    ))


def fold_ring() -> dict[str, Any]:
    """Chess with a ring of three folds, e2 to c5 to f4 to e2. The first starts under
    White's e-pawn (rule 10), so a landing on f4 goes through only once it moves; either
    player may move the end on c5."""
    return _schema(_with_groups(
        "chess",
        [{"kind": "fold", "ends": [
            {"at": "e2", "to": 1},
            {"at": "c5", "to": 2, "moved_by": [0, 1]},
            {"at": "f4", "to": 0},
        ]}],
        [STEP],
    ))


def fold_beside_momentum() -> dict[str, Any]:
    """Los Alamos with a momentum pair on b3 and e4 and a fold pair on c4 and d3, every end
    movable by either player."""
    both = {"orange": [0, 1], "blue": [0, 1]}
    return _schema(_with_groups(
        "losalamos",
        [
            {"orange": "b3", "blue": "e4", "moved_by": both},
            {"orange": "c4", "blue": "d3", "kind": "fold", "moved_by": both},
        ],
        [STEP],
    ))


def fold_pawns() -> dict[str, Any]:
    """Chess with a fixed fold pair on d3 and d6: a pawn's push onto d3 may come out on d6,
    and a pawn's diagonal onto the empty d3 comes out taking whatever stands on d6."""
    return _schema(_with_groups(
        "chess", [{"orange": "d3", "blue": "d6", "kind": "fold"}],
    ))


def fold_funnel() -> dict[str, Any]:
    """Chess with a fixed funnel of folds: c3 and f6 both lead to e5, which leads nowhere.
    Each side's queen's knight lands on its own way in from the start."""
    return _schema(_with_groups(
        "chess",
        [{"kind": "fold", "ends": [
            {"at": "c3", "to": 2}, {"at": "f6", "to": 2}, {"at": "e5"},
        ]}],
    ))


GAMES = {
    "fold_pair": fold_pair, "fold_ring": fold_ring, "beside": fold_beside_momentum,
    "pawns": fold_pawns, "funnel": fold_funnel,
}

# Perft of the five test specs at depths 1 to 4, frozen (Q867b) once the oracle agreed with
# the core over their random walks and on every hand-built position below. A count that moves
# is a movegen change until proven otherwise. Schema 22's first rules counted the first three
# (15, 218, 3344, 51242), (29, 812, 24816, 726162) and (31, 956, 30246, 954800).
FROZEN_PERFT: dict[str, tuple[int, ...]] = {
    "fold_pair": (17, 275, 4670, 77258),
    "fold_ring": (28, 809, 24331, 737769),
    "beside": (33, 1065, 35093, 1144012),
    "pawns": (21, 441, 10421, 244038),
    "funnel": (22, 484, 11657, 274827),
}


def _perft_cases() -> list[Any]:
    cases = []
    for name, counts in FROZEN_PERFT.items():
        for depth, expected in enumerate(counts, start=1):
            marks = [pytest.mark.slow] if expected > FAST_BUDGET_NODES else []
            cases.append(pytest.param(name, depth, expected, marks=marks, id=f"{name}-{depth}"))
    return cases


# -- reading the core ---------------------------------------------------------------------


def _names(engine: Engine) -> list[str]:
    """Each portal end's id (``1o``, ``2a``, ...), in end order."""
    portals = engine.spec.portals
    assert portals is not None
    return [engine.spec.piece_ids[piece_type] for piece_type in portals.end_types]


def _folds(engine: Engine, state: GameState) -> Folds:
    names = _names(engine)
    return {
        names[end]: square
        for end in range(len(names))
        if (square := engine.fold_square(state, end)) is not None
    }


def _core_ordinary(engine: Engine, state: GameState) -> set[Move]:
    spec = engine.spec
    names = _names(engine)
    found: set[Move] = set()
    for move in engine.pseudo_legal_moves(state):
        kind = move_type(move)
        if kind not in (MOVE_NORMAL, MOVE_PROMOTION):
            continue
        value = state.board[move_from(move)]
        if not value or spec.owner_of(value) == WILD:
            continue
        promoted = spec.piece_ids[move_promotion(move)] if kind == MOVE_PROMOTION else None
        tag = move_fold(move)
        found.add((move_from(move), move_to(move), promoted, names[tag - 1] if tag else None))
    return found


def _core_fold_moves(engine: Engine, state: GameState) -> set[tuple[str, int, int]]:
    names = _names(engine)
    return {
        (names[move_fold(move)], move_from(move), move_to(move))
        for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_FOLD
    }


def _last_fold(engine: Engine, state: GameState, move: int) -> tuple[str, int, int] | None:
    if move_type(move) != MOVE_FOLD:
        return None
    return _names(engine)[move_fold(move)], move_from(move), move_to(move)


def _agrees(engine: Engine, oracle: FoldOracle, state: GameState,
            last: tuple[str, int, int] | None = None) -> int:
    """The core and the oracle agree on every move of ``state`` and on its legality; the
    number of moves through a fold seen."""
    position = _position(engine, state)
    folds = _folds(engine, state)
    player = state.side_to_move
    unmoved = _unmoved(state)
    ordinary = oracle.ordinary_moves(position, folds, player, unmoved)
    assert _core_ordinary(engine, state) == ordinary
    assert _core_fold_moves(engine, state) == oracle.fold_moves(position, folds, player, last)
    assert _core_transits(engine, state) == oracle.momentum_transits(
        position, folds, player, unmoved,
    )
    assert {
        (origin, target) for origin, target in _core_portal_moves(engine, state)
    } == oracle.momentum_moves(position, folds, player, unmoved=unmoved)
    legal = set(engine.legal_moves(state))
    for move in engine.pseudo_legal_moves(state):
        after = engine.apply(state, move)
        placed = _position(engine, after)
        king = next(
            (sq for sq, (pid, owner) in placed.items() if pid == "K" and owner == player), None,
        )
        if king is None:
            continue
        exposed = oracle.attacked(placed, _folds(engine, after), 1 - player, king, _unmoved(after))
        assert (move in legal) == (not exposed), engine.describe_move(state, move)
    return sum(1 for item in ordinary if item[3] is not None)


# -- the inertness proof ----------------------------------------------------------------


class TestTheBumpMovedNothing:
    def test_every_hash_is_the_one_frozen(self) -> None:
        for path, (digest, _) in FROZEN.items():
            assert spec_hash(load_spec(ROOT / path)) == digest, path

    def test_only_the_showcase_uses_a_fold(self) -> None:
        """Nothing written before the bump uses a fold: the one user is Q844b's showcase
        (`docs/reference/portal-showcase-ideas.md`), in `archive/`."""
        assert [
            path for path in FROZEN if _schema_twenty_four_uses(load_spec(ROOT / path))
        ] == ["archive/crease.json"]

    @pytest.mark.parametrize("path", sorted(
        path for path in FROZEN if "portals" in (load_spec(ROOT / path).get("rules") or {})
        and not _schema_twenty_four_uses(load_spec(ROOT / path))
    ))
    def test_a_game_without_folds_sends_no_fold_key(self, path: str) -> None:
        tables = canonical_tables(compile_spec(load_spec(ROOT / path)))
        assert not [key for key in tables if "fold" in key or key == "portal_kind"]

    def test_chess_sends_no_fold_key(self) -> None:
        tables = canonical_tables(compile_spec(load_spec(ROOT / "rulesets" / "chess.json")))
        assert not [key for key in tables if "fold" in key or key == "portal_kind"]

    def test_a_fold_game_sends_only_where_the_folds_start_and_their_key(self) -> None:
        tables = canonical_tables(compile_spec(fold_pair()))
        assert sorted(key for key in tables if "fold" in key or key == "portal_kind") == [
            "fold_starts", "portal_kind", "zobrist_fold",
        ]


class TestTheVersionIsBumped:
    def test_the_schema_version_is_twenty_four_or_later(self) -> None:
        assert SCHEMA_VERSION >= 24

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_the_collector_names_kind_and_schema_twenty_two_keeps_nothing(self, name: str) -> None:
        spec = GAMES[name]()
        assert _schema_twenty_four_uses(spec) == {"rules.portals.kind"}
        assert _schema_twenty_two_uses(spec) == set()


# -- the oracle over random walks -------------------------------------------------------


def _walk(name: str, walks: int, plies: int, seed: int) -> None:
    """``walks`` random games of up to ``plies`` plies of the named test spec, the core and
    the oracle agreeing at every ply, the key incremental at every ply, and some move going
    through a fold."""
    spec = GAMES[name]()
    engine = Engine(spec)
    oracle = FoldOracle(spec)
    rng = random.Random(seed)
    routed = 0
    for _ in range(walks):
        state = engine.initial_state()
        last = None
        for _ in range(plies):
            if engine.terminal(state):
                break
            routed += _agrees(engine, oracle, state, last)
            move = rng.choice(engine.legal_moves(state))
            last = _last_fold(engine, state, move)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
    assert routed > 0


class TestTheOracleAgreesOnRandomWalks:
    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_random_walks(self, name: str) -> None:
        _walk(name, 2, 25, 24)

    @pytest.mark.slow
    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_long_random_walks(self, name: str) -> None:
        _walk(name, 8, 80, 2424)


class TestThePerftIsFrozen:
    @pytest.mark.parametrize(("name", "depth", "expected"), _perft_cases())
    def test_perft(self, name: str, depth: int, expected: int) -> None:
        engine = Engine(GAMES[name]())
        assert engine.perft(engine.initial_state(), depth) == expected


# -- hand-built positions, one test per rule of §3R.1 -------------------------------------


def _sparse(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with its setup cut to a king and a pawn a side, clear of every fold; a
    hand-built position ignores the setup, but the validator reads it."""
    spec = copy.deepcopy(spec)
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["e1"]}, {"piece": "P", "squares": ["a2"]},
    ]
    return spec


def _gardner_folds(**pair: Any) -> dict[str, Any]:
    """Gardner, a king and a pawn a side, and a fold pair on b3 and d3 (plus ``pair``)."""
    return _sparse(_schema(_with_groups(
        "gardner", [{"orange": "b3", "blue": "d3", "kind": "fold", **pair}],
        [STEP],
    )))


def _chess_folds(*groups: dict[str, Any], **rules: Any) -> dict[str, Any]:
    """Chess, a king, a rook and a pawn a side (so castling can still happen), and the fold
    groups given."""
    spec = _sparse(_schema(_with_groups(
        "chess", [{"kind": "fold", **group} for group in groups], [STEP], **rules,
    )))
    spec["setup"]["placements"].append({"piece": "R", "squares": ["h1"]})
    return spec


def _at(spec: dict[str, Any], placement: dict[str, tuple[str, int]], side: int = 0,
        folds: tuple[str, ...] = (), moved: tuple[str, ...] = (),
        ) -> tuple[Engine, GameState, list[str]]:
    """The position, its move texts, and the oracle's agreement with the core on it."""
    engine = Engine(spec)
    state = engine.make_state(placement, side_to_move=side, folds=folds, moved=moved)
    _agrees(engine, FoldOracle(spec), state)
    return engine, state, engine.move_texts(state)


def _pseudo_texts(engine: Engine, state: GameState) -> set[str]:
    return {engine.describe_move(state, move) for move in engine.pseudo_legal_moves(state)}


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


class TestTheRevisedFoldRules:
    """`docs/archive/plan-portals-2.md` §3R.1, one test per rule (and per case §3R.4 names), each
    position also held against the oracle."""

    def test_1_a_fold_is_not_a_piece(self) -> None:
        """A piece standing in a fold is captured there as anywhere, and any move may land
        on a fold."""
        spec = _gardner_folds()
        engine, state, texts = _at(spec, {
            "e1": ("K", 0), "e5": ("K", 1), "a3": ("R", 0), "b3": ("N", 1), "c1": ("N", 0),
        })
        assert "Ra3xb3" in texts
        after = engine.apply(state, engine.find_move(state, "Ra3xb3"))
        assert _position(engine, after)[_sq(engine, "b3")] == ("R", 0)
        assert after.folds == state.folds == (_sq(engine, "b3"), _sq(engine, "d3"))
        assert "Nc1-d3" in texts

    def test_2_a_landing_on_a_fold_may_stay_or_go_through(self) -> None:
        spec = _gardner_folds()
        engine, state, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0)})
        assert {"Rb1-b3", "Rb1-b3~d3"} <= set(texts)
        # The slide is not turned aside: it runs on past the fold as over any empty square.
        assert {"Rb1-b4", "Rb1-b5"} <= set(texts)
        through = engine.find_move(state, "Rb1-b3~d3")
        assert move_type(through) == MOVE_NORMAL and move_to(through) == _sq(engine, "d3")
        after = engine.apply(state, through)
        assert _position(engine, after)[_sq(engine, "d3")] == ("R", 0)
        assert _sq(engine, "b3") not in _position(engine, after)
        # A leap goes through as well.
        _, _, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "a1": ("N", 0)})
        assert {"Na1-b3", "Na1-b3~d3"} <= set(texts)

    def test_2_a_pawns_push_and_its_diagonal(self) -> None:
        spec = _chess_folds({"orange": "d3", "blue": "d6"})
        # The push goes through only to an empty far end; the double step passes over d3.
        _, _, texts = _at(spec, {"e1": ("K", 0), "e8": ("K", 1), "d2": ("P", 0), "c2": ("P", 0)})
        assert {"Pd2-d3", "Pd2-d3~d6", "Pd2-d4"} <= set(texts)
        # The diagonal, which may not stop on the empty d3, goes through only to take.
        assert not [text for text in texts if text.startswith("Pc2")
                    and "~" in text] and "Pc2-c3" in texts
        _, _, texts = _at(spec, {"e1": ("K", 0), "e8": ("K", 1), "d2": ("P", 0), "c2": ("P", 0),
                                 "d6": ("N", 1)})
        assert "Pc2-d3~xd6" in texts and "Pd2-d3" in texts
        assert not [text for text in texts if text.startswith("Pd2-d3~")]

    def test_2_a_capture_on_the_far_end_and_the_matrix_refusing_one(self) -> None:
        spec = _gardner_folds()
        engine, state, texts = _at(spec, {
            "e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "d3": ("B", 1),
        })
        assert "Rb1-b3~xd3" in texts
        after = engine.apply(state, engine.find_move(state, "Rb1-b3~xd3"))
        assert _position(engine, after)[_sq(engine, "d3")] == ("R", 0)
        picky = _gardner_folds()
        rook = next(piece for piece in picky["pieces"] if piece["id"] == "R")
        rook["captures"] = ["P"]
        _, _, texts = _at(picky, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "d3": ("B", 1)})
        assert "Rb1-b3" in texts and not [text for text in texts if "~" in text]

    def test_2_never_onto_its_own_piece(self) -> None:
        spec = _gardner_folds()
        _, _, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "d3": ("P", 0)})
        assert "Rb1-b3" in texts and not [text for text in texts if "~" in text]

    def test_2_the_pass_is_refused(self) -> None:
        """A rook standing on d3, the far end of b3, lands on b3 and stays: going through
        would bring it back to its own square."""
        spec = _gardner_folds()
        _, _, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "d3": ("R", 0)})
        assert "Rd3-b3" in texts
        assert not [text for text in texts if text.startswith("Rd3") and "~" in text]

    def test_2_promotion_is_read_where_the_move_ends(self) -> None:
        spec = _chess_folds({"orange": "c6", "blue": "f8"})
        engine, state, texts = _at(spec, {"e1": ("K", 0), "a8": ("K", 1), "c5": ("P", 0)})
        assert "Pc5-c6" in texts and "Pc5-c6~f8=Q" in texts
        move = engine.find_move(state, "Pc5-c6~f8=Q")
        assert move_type(move) == MOVE_PROMOTION and move_to(move) == _sq(engine, "f8")

    def test_3_one_fold_per_move(self) -> None:
        """In a ring a landing on one end comes out on the next and stops there, though
        that end leads on in turn; in a pair it does not come back."""
        spec = _chess_folds({"ends": [{"at": "b3", "to": 1}, {"at": "d5", "to": 2},
                                      {"at": "f3", "to": 0}]})
        _, _, texts = _at(spec, {"e1": ("K", 0), "h8": ("K", 1), "b1": ("R", 0)})
        assert "Rb1-b3~d5" in texts
        assert not [text for text in texts if text.startswith("Rb1") and "~f3" in text]

    def test_4_a_piece_in_the_fold_is_in_the_way(self) -> None:
        spec = _gardner_folds()
        _, _, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "b3": ("P", 1)})
        assert "Rb1xb3" in texts and not [text for text in texts if "~" in text]
        # And a piece standing in a fold reaches nothing through it.
        _, _, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b3": ("N", 0)})
        assert not [text for text in texts if "~" in text]
        assert "Nb3-c5" in texts and "Nb3-e4" not in texts

    def test_4_a_fold_that_does_not_admit_the_piece_is_floor(self) -> None:
        filtered = _gardner_folds(carries={"orange": ["N"]})
        _, _, texts = _at(filtered, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0),
                                     "a1": ("N", 0)})
        assert "Rb1-b3" in texts and not [text for text in texts if text.startswith("Rb1")
                                          and "~" in text]
        assert "Na1-b3~d3" in texts

    def test_4_a_funnel_and_its_way_out(self) -> None:
        spec = _chess_folds({"ends": [{"at": "c3", "to": 2}, {"at": "f6", "to": 2}, {"at": "e5"}]})
        _, _, texts = _at(spec, {"e1": ("K", 0), "e8": ("K", 1), "b1": ("N", 0), "b2": ("B", 0)})
        assert {"Nb1-c3", "Nb1-c3~e5"} <= set(texts)
        # The bishop reaches e5 straight along its diagonal, so the move through c3 is the
        # direct one; e5 leads nowhere, so a landing on it only stays.
        assert "Bb2-e5" in texts and not [text for text in texts if text.startswith("Bb2")
                                          and "~" in text]

    def test_5_castling_and_en_passant_never_go_through(self) -> None:
        spec = _chess_folds({"orange": "e4", "blue": "h5"}, en_passant=True)
        engine, state, texts = _at(spec, {"e1": ("K", 0), "a8": ("K", 1), "e2": ("P", 0)})
        assert {"Pe2-e3", "Pe2-e4", "Pe2-e4~h5"} <= set(texts)
        stay = engine.apply(state, engine.find_move(state, "Pe2-e4"))
        assert stay.ep_target == _sq(engine, "e3")
        through = engine.apply(state, engine.find_move(state, "Pe2-e4~h5"))
        assert through.ep_target == NO_SQUARE
        # A capture in passing onto a fold is that capture, and nothing more.
        spec = _chess_folds({"orange": "e6", "blue": "a3"}, en_passant=True)
        engine, state, _ = _at(spec, {
            "e1": ("K", 0), "h8": ("K", 1), "d5": ("P", 0), "e5": ("P", 1),
        }, moved=("e5",))
        state = state.with_(ep_target=_sq(engine, "e6"), ep_victim=_sq(engine, "e5"))
        state = state.with_(key=engine._compute_hash(state))
        _agrees(engine, FoldOracle(spec), state)
        passing = [move for move in engine.legal_moves(state) if move_type(move) == MOVE_EN_PASSANT]
        assert len(passing) == 1 and move_fold(passing[0]) == 0
        assert move_to(passing[0]) == _sq(engine, "e6")

    def test_6_a_fold_moves_onto_pieces_and_never_onto_a_portal(self) -> None:
        spec = _gardner_folds(moved_by={"orange": [0, 1]})
        engine, state, texts = _at(spec, {
            "e1": ("K", 0), "e5": ("K", 1), "b2": ("P", 0), "c4": ("P", 1),
        })
        assert {"1o:b3-b2", "1o:b3-c4", "1o:b3-a3"} <= set(texts)
        after = engine.apply(state, engine.find_move(state, "1o:b3-b2"))
        assert after.folds == (_sq(engine, "b2"), _sq(engine, "d3"))
        assert after.board == state.board
        beside = _gardner_folds(moved_by={"orange": [0, 1]})
        beside["rules"]["portals"][0]["blue"] = "c3"
        _, _, texts = _at(beside, {"e1": ("K", 0), "e5": ("K", 1)})
        assert "1o:b3-a3" in texts and "1o:b3-c3" not in texts

    def test_6_a_far_end_moved_under_a_piece_threatens_it(self) -> None:
        """A fold's far end moved under an enemy piece: every capture step onto the near end
        now takes it."""
        spec = _gardner_folds(moved_by={"blue": [0, 1]})
        engine, state, texts = _at(spec, {
            "e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "d4": ("N", 1),
        }, side=0)
        assert "Rb1-b3~d3" in texts and "Rb1-b3~xd4" not in texts
        _, _, texts = _at(spec, {
            "e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "d4": ("N", 1),
        }, folds=("b3", "d4"))
        assert "Rb1-b3~xd4" in texts

    def test_6_no_straight_back_holds_for_a_fold(self) -> None:
        spec = _gardner_folds(moved_by={"orange": [0, 1]}, no_straight_back=True)
        engine, state, _ = _at(spec, {"e1": ("K", 0), "e5": ("K", 1)})
        after = engine.apply(state, engine.find_move(state, "1o:b3-b4"))
        texts = engine.move_texts(after)
        assert "1o:b4-b3" not in texts and "1o:b4-c4" in texts

    def test_7_a_slide_crosses_a_fold_to_a_momentum_end_beyond(self) -> None:
        spec = _sparse(_schema(_with_groups("losalamos", [
            {"orange": "c5", "blue": "e3"},
            {"orange": "c3", "blue": "a4", "kind": "fold"},
        ], [STEP])))
        spec["setup"]["placements"][0]["squares"] = ["d1"]
        placement = {"c5": ("1o", WILD), "e3": ("1b", WILD), "d1": ("K", 0), "f6": ("K", 1),
                     "c1": ("R", 0)}
        _, _, texts = _at(spec, placement)
        assert {"Rc1-c3", "Rc1-c3~a4", "Rc1-c4", "Rc1-c5>e4"} <= set(texts)

    def test_7_a_momentum_exit_onto_a_fold_stays(self) -> None:
        out = _sparse(_schema(_with_groups("losalamos", [
            {"orange": "b3", "blue": "e4"},
            {"orange": "e5", "blue": "a6", "kind": "fold"},
        ], [STEP])))
        out["setup"]["placements"][0]["squares"] = ["d1"]
        engine, state, texts = _at(out, {"b3": ("1o", WILD), "e4": ("1b", WILD),
                                         "d1": ("K", 0), "f1": ("K", 1), "b1": ("R", 0)}, side=0)
        assert "Rb1-b3>e5" in texts
        assert not [text for text in texts if ">" in text and "~" in text]
        after = engine.apply(state, engine.find_move(state, "Rb1-b3>e5"))
        assert _position(engine, after)[_sq(engine, "e5")] == ("R", 0)

    def test_8_check_through_a_fold(self) -> None:
        spec = _gardner_folds()
        oracle = FoldOracle(spec)
        # Up the b-file onto the empty b3, out on d3: the king there is in check.
        engine, state, _ = _at(spec, {"e1": ("K", 0), "d3": ("K", 1), "b1": ("R", 0)}, side=1)
        assert engine.in_check(state, 1)
        assert oracle.attacked(_position(engine, state), _folds(engine, state), 0,
                               _sq(engine, "d3"), _unmoved(state))
        # A piece standing in b3 is in the way, and gives no check through it.
        engine, state, _ = _at(spec, {"e1": ("K", 0), "d3": ("K", 1), "b1": ("R", 0),
                                      "b3": ("P", 1)}, side=1)
        assert not engine.in_check(state, 1)
        # A royal standing in a fold is attacked on its own square, the far end's way in
        # included: the rook down the d-file lands on d3 and comes out on b3.
        engine, state, _ = _at(spec, {"b3": ("K", 0), "e5": ("K", 1), "d5": ("R", 1)})
        assert engine.in_check(state, 0)
        engine, state, _ = _at(spec, {"b3": ("K", 0), "e5": ("K", 1), "e4": ("R", 1)})
        assert not engine.in_check(state, 0)

    def test_8_a_royal_may_not_go_through_into_check(self) -> None:
        spec = _gardner_folds()
        engine, state, texts = _at(spec, {"a2": ("K", 0), "e5": ("K", 1), "d5": ("R", 1)})
        pseudo = _pseudo_texts(engine, state)
        assert {"Ka2-b3", "Ka2-b3~d3"} <= pseudo
        # d3 is on the rook's file; b3 is reached by the rook through d3.
        assert "Ka2-b3~d3" not in texts and "Ka2-b3" not in texts
        assert "Ka2-b2" in texts

    def test_9_a_fold_on_the_castling_span_is_floor(self) -> None:
        placement = {"e1": ("K", 0), "h1": ("R", 0), "e8": ("K", 1)}
        _, _, texts = _at(_chess_folds({"orange": "f1", "blue": "h5"}), placement)
        assert "Ke1-g1 (castle)" in texts
        # The king may not cross a square attacked through a fold: a rook down the a-file
        # lands on a5 and comes out on f1.
        attacked = dict(placement, a8=("R", 1))
        _, _, texts = _at(_chess_folds({"orange": "a5", "blue": "f1"}), attacked)
        assert "Ke1-g1 (castle)" not in texts
        _, _, texts = _at(_chess_folds({"orange": "a5", "blue": "f1"}),
                          dict(placement, b8=("R", 1)))
        assert "Ke1-g1 (castle)" in texts

    def test_9_a_drop_may_land_on_a_fold_and_stays(self) -> None:
        spec = _schema(_with_groups(
            "crazyhouse", [{"orange": "c3", "blue": "f6", "kind": "fold"}], [STEP],
        ))
        engine = Engine(spec)
        state = engine.make_state({"e1": ("K", 0), "e8": ("K", 1)})
        held = [0] * engine.spec.n_types
        held[engine.spec.piece_ids.index("N")] = 1
        state = state.with_(hands=(tuple(held), tuple([0] * engine.spec.n_types)))
        texts = engine.move_texts(state)
        assert "N@c3" in texts and not [text for text in texts if "~" in text]

    def test_9_only_fold_moves_is_not_stalemate(self) -> None:
        spec = _gardner_folds(moved_by={"orange": [0]})
        engine, state, _ = _at(spec, {"a1": ("K", 0), "c2": ("Q", 1), "e5": ("K", 1)})
        legal = engine.legal_moves(state)
        assert legal and all(move_type(move) == MOVE_FOLD for move in legal)
        assert engine.terminal(state) is None

    def test_10_a_fold_may_start_under_a_piece_and_a_momentum_end_may_not(self) -> None:
        assert validate_spec(fold_ring()) == []
        spec = fold_ring()
        del spec["rules"]["portals"][0]["kind"]
        assert any("a square setup.placements puts a piece on" in problem
                   for problem in validate_spec(spec))
        compiled = compile_spec(fold_ring())
        # A fold stands on no square of the board byte: e2 holds the pawn.
        assert compiled.piece_ids[compiled.type_of(compiled.initial_board[12])] == "P"

    def test_10_the_refusals_beside_portals_hold_for_folds(self) -> None:
        spec = fold_pair()
        spec["rules"]["explode_on_capture"] = {"radius": 1}
        assert any("refused beside rules.explode_on_capture" in p for p in validate_spec(spec))

    def test_a_duplicate_collapses_to_the_direct_move(self) -> None:
        """Along the third rank the rook reaches d3 straight and through b3: one move, the
        direct one."""
        spec = _gardner_folds()
        engine, state, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "a3": ("R", 0)})
        assert {"Ra3-b3", "Ra3-c3", "Ra3-d3", "Ra3-e3"} <= set(texts)
        assert not [text for text in texts if "~" in text]
        to_d3 = [move for move in engine.pseudo_legal_moves(state)
                 if move_to(move) == _sq(engine, "d3") and move_from(move) == _sq(engine, "a3")]
        assert len(to_d3) == 1 and move_fold(to_d3[0]) == 0


class TestWhatTheValidatorRefuses:
    def test_kind_momentum_spelled_out_is_the_group_without_it(self) -> None:
        """Q1008c: the off value is accepted and names the momentum group."""
        spec = fold_pair()
        del spec["rules"]["portals"][0]["kind"]
        written = copy.deepcopy(spec)
        written["rules"]["portals"][0]["kind"] = "momentum"
        assert validate_spec(written) == validate_spec(spec)
        assert spec_hash(written) == spec_hash(spec)

    def test_an_unknown_kind(self) -> None:
        spec = fold_pair()
        spec["rules"]["portals"][0]["kind"] = "warp"
        assert any("kind must be one of" in p for p in validate_spec(spec))

    def test_a_first_move_condition_on_a_folds_piece(self) -> None:
        spec = fold_pair()
        group = spec["rules"]["portals"][0]
        # A copy: `_with_portals` hands every group the one shared `STEP`.
        group["moves"] = [dict(group["moves"][0], condition={"first_move": True})]
        assert any("may not have a first_move condition" in p for p in validate_spec(spec))

    def test_a_momentum_end_on_a_folds_square(self) -> None:
        spec = fold_beside_momentum()
        spec["rules"]["portals"][0]["orange"] = "c4"
        assert any("two portals on one square" in p for p in validate_spec(spec))


class TestTheMoveWordAndTheState:
    def test_a_folds_own_move_is_its_own_kind_and_reversible(self) -> None:
        spec = _gardner_folds(moved_by={"orange": [0, 1]})
        engine, state, _ = _at(spec, {"e1": ("K", 0), "e5": ("K", 1)},
                               moved=("e1", "e5"))
        move = engine.find_move(state, "1o:b3-b4")
        assert move_type(move) == MOVE_FOLD and move_fold(move) == 0
        after = engine.apply(state, move)
        assert len(after.repetition) == len(state.repetition) + 1
        assert after.halfmove == state.halfmove + 1
        assert after.key == engine._compute_hash(after) != state.key

    def test_a_move_through_a_fold_ends_on_the_far_end_and_names_its_way_in(self) -> None:
        spec = _gardner_folds()
        engine, state, _ = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0)})
        move = engine.find_move(state, "Rb1-b3~d3")
        assert move_type(move) == MOVE_NORMAL and move_fold(move) == 1
        assert move_to(move) == _sq(engine, "d3")
        assert engine.fold_route(state, move) == (_sq(engine, "b3"), _sq(engine, "d3"))
        stay = engine.find_move(state, "Rb1-b3")
        assert move_fold(stay) == 0 and engine.fold_route(state, stay) is None

    def test_a_capture_through_a_fold_says_so(self) -> None:
        spec = _gardner_folds()
        _, _, texts = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0), "d3": ("N", 1)})
        assert "Rb1-b3~xd3" in texts

    def test_where_a_fold_stands_is_in_the_key(self) -> None:
        spec = _gardner_folds(moved_by={"orange": [0, 1]})
        engine = Engine(spec)
        placement = {"e1": ("K", 0), "e5": ("K", 1)}
        here = engine.make_state(placement)
        there = engine.make_state(placement, folds=("b4", "d3"))
        assert here.board == there.board and here.key != there.key


class TestTheSurfaces:
    def test_the_prose_says_what_a_fold_does(self) -> None:
        said = " ".join(describe_rules(fold_pair()))
        assert "A fold is a portal of another kind" in said
        assert "may stay there, or go through" in said
        assert "Fold pair 1 joins b3 and d3" in said
        momentum = " ".join(describe_rules(_with_groups(
            "gardner", [{"orange": "b3", "blue": "d3"}],
        )))
        assert "fold" not in momentum.lower()

    def test_the_page_is_sent_where_each_fold_stands(self) -> None:
        from chess_variant_sandbox.viz.wrapped import fold_payload, portal_payload

        engine = Engine(fold_beside_momentum())
        state = engine.initial_state()
        folds = fold_payload(engine, state)
        assert [(fold["square"], fold["pair"]) for fold in folds] == [
            (_sq(engine, "c4"), 2), (_sq(engine, "d3"), 2),
        ]
        payload = portal_payload(engine.spec)
        assert payload is not None
        assert [end["kind"] for end in payload["ends"]] == [
            "momentum", "momentum", "fold", "fold",
        ]
        assert fold_payload(Engine(load_spec(ROOT / "archive" / "ferry.json")),
                            engine.initial_state()) == []

    def test_the_page_is_sent_the_way_in_and_where_the_move_ends(self) -> None:
        from chess_variant_sandbox.viz.wrapped import fold_transit

        spec = _gardner_folds()
        engine, state, _ = _at(spec, {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0)})
        move = engine.find_move(state, "Rb1-b3~d3")
        sent = fold_transit(engine, state, move)
        assert sent is not None
        assert (sent["foldIn"], sent["foldOut"], sent["foldPair"]) == (
            _sq(engine, "b3"), move_to(move), 1,
        )
        assert fold_transit(engine, state, engine.find_move(state, "Rb1-b3")) is None

    def test_fairy_stockfish_refuses_folds(self) -> None:
        with pytest.raises(Untranslatable, match="rules.portals"):
            translate(compile_spec(fold_pair()))


class TestTheBoundary:
    """A fold game played through the Python loop and agents and through the core, move for
    move (`tests/test_backends.py`'s comparison): the Python search mirrors `agents.rs`'s
    capture tests, which count a capture through a fold (an occupied target under
    `MOVE_NORMAL`) and leave a fold moved onto a piece out (`MOVE_FOLD` takes nothing)."""

    @pytest.mark.parametrize("name", [
        "fold_pair",
        pytest.param("fold_ring", marks=pytest.mark.slow),
        "beside",
        pytest.param("pawns", marks=pytest.mark.slow),
        pytest.param("funnel", marks=pytest.mark.slow),
    ])
    def test_a_fold_game_is_played_the_same_on_both_sides(self, name: str) -> None:
        from test_backends import _assert_same_played

        _assert_same_played(Engine(GAMES[name]()), ("minimax:nodes=150", "greedy"))
