"""Schema 20 (Q807): a portal may not go straight back (`docs/plan-portals.md` *Ferry's shuffle*).

`rules.portals[].no_straight_back`: once an end of the pair is moved, the very next move of
the game may not move that end back onto the square it left. It is a portal's ko, the answer
to Ferry's repetition by shuffling (0.404 of its games, 404 of the 414 by an immediate undo).

The core keeps the last such move as `Game::portal_bar` (`GameState.portal_bar`), in the
Zobrist key. Its movegen is held against `tests/portal_oracle.py`, which reads the bar off the
move just played and the spec, over random walks of the new Ferry; the rest pins the rule's
edges (whoever moves next, only the one move, the bar in the key and across the boundary) and
that the bump moved nothing (`TestTheBumpMovedNothing` in schema 19's file covers every frozen
hash and tables fingerprint).
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from portal_oracle import PortalOracle
from test_schema_nineteen import (
    _core_portal_moves,
    _position,
    _sparse,
    _unmoved,
    ferry,
    wormhole,
)

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, _schema_twenty_uses

ROOT = Path(__file__).resolve().parents[1]


def barred(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` stamped 20 with ``no_straight_back`` on every pair."""
    spec = copy.deepcopy(spec)
    spec["schema_version"] = 20
    for pair in spec["rules"]["portals"]:
        pair["no_straight_back"] = True
    return spec


def ferry_barred() -> dict[str, Any]:
    return barred(ferry())


def _portal_move(engine: Engine, state: Any, origin: str, target: str) -> int:
    return next(
        move for move in engine.legal_moves(state)
        if move_from(move) == engine._square_index(origin)
        and move_to(move) == engine._square_index(target)
    )


def _has(engine: Engine, state: Any, origin: str, target: str) -> bool:
    return any(
        move_from(move) == engine._square_index(origin)
        and move_to(move) == engine._square_index(target)
        for move in engine.legal_moves(state)
    )


class TestTheVersionIsBumped:
    def test_the_schema_version_is_twenty(self) -> None:
        assert SCHEMA_VERSION == 20

    def test_the_collector_names_the_key_whatever_its_value(self) -> None:
        assert _schema_twenty_uses(ferry()) == set()
        assert _schema_twenty_uses(ferry_barred()) == {"rules.portals.no_straight_back"}
        spec = ferry()
        spec["rules"]["portals"][0]["no_straight_back"] = False
        assert _schema_twenty_uses(spec) == {"rules.portals.no_straight_back"}

    def test_an_older_stamp_is_refused_by_name(self) -> None:
        spec = ferry_barred()
        assert validate_spec(spec) == []
        spec["schema_version"] = 19
        assert any(
            "rules.portals.no_straight_back" in error and "schema 20" in error
            for error in validate_spec(spec)
        )

    def test_the_archived_game_is_valid_and_new(self) -> None:
        spec = load_spec(ROOT / "archive" / "ferry-no-straight-back.json")
        assert validate_spec(spec) == []
        assert spec_hash(spec) == "dd17d71412a7"
        assert spec["provenance"]["parents"] == ["1501615e08e2"]
        assert spec_hash(load_spec(ROOT / "archive" / "ferry.json")) == "1501615e08e2"


class TestTheValidator:
    def test_a_word_is_refused(self) -> None:
        spec = ferry_barred()
        spec["rules"]["portals"][0]["no_straight_back"] = "yes"
        assert "rules.portals[0].no_straight_back must be true or false" in validate_spec(spec)

    def test_a_bar_nobody_could_meet_is_refused(self) -> None:
        """A pair nobody moves never has a move to bar: one game under two hashes."""
        spec = barred(wormhole())
        assert any(
            "rules.portals[0].no_straight_back is true but nobody may move" in error
            for error in validate_spec(spec)
        )

    def test_false_is_the_default_spelled_out(self) -> None:
        spec = ferry()
        spec["schema_version"] = 20
        spec["rules"]["portals"][1]["no_straight_back"] = False
        assert validate_spec(spec) == []
        assert canonical_tables(compile_spec(spec)).keys() == canonical_tables(
            compile_spec(ferry())
        ).keys()


class TestTheTablesAreInert:
    def test_a_game_without_the_bar_sends_both_keys_empty(self) -> None:
        for spec in (ferry(), load_spec(ROOT / "rulesets" / "chess.json")):
            tables = build_tables(compile_spec(spec))
            assert tables["portal_no_back"] == []
            assert tables["zobrist_portal_bar"] == []
            canonical = canonical_tables(compile_spec(spec))
            assert "portal_no_back" not in canonical
            assert "zobrist_portal_bar" not in canonical

    def test_the_bar_draws_after_every_other_key(self) -> None:
        """The flag adds keys and moves none: the same spec hash would draw the same
        pieces, side, en passant, colour and turn keys with or without it."""
        plain = compile_spec(ferry())
        flagged = compile_spec(ferry_barred())
        tables = build_tables(flagged)
        assert len(tables["zobrist_portal_bar"]) == 2 * 36
        assert tables["portal_no_back"][plain.portals.end_types[0]] == 1
        assert sum(tables["portal_no_back"]) == 4
        assert len(flagged.zobrist_pieces) == len(plain.zobrist_pieces)


class TestTheRule:
    def test_the_end_may_not_go_straight_back(self) -> None:
        engine = Engine(ferry_barred())
        state = engine.initial_state()
        assert _has(engine, state, "b3", "c3")
        state = engine.apply(state, _portal_move(engine, state, "b3", "c3"))
        assert state.portal_bar == (engine._square_index("c3"), engine._square_index("b3"))
        # Black may not put it back, but may move it elsewhere, or move its partner.
        assert not _has(engine, state, "c3", "b3")
        assert _has(engine, state, "c3", "d3")
        assert _has(engine, state, "e4", "f4")

    def test_only_on_the_very_next_move(self) -> None:
        engine = Engine(ferry_barred())
        state = engine.initial_state()
        state = engine.apply(state, _portal_move(engine, state, "b3", "c3"))
        state = engine.apply(state, _portal_move(engine, state, "a5", "a4"))
        assert state.portal_bar == (-1, -1)
        assert _has(engine, state, "c3", "b3")

    def test_the_mover_may_not_take_it_back_either(self) -> None:
        """In a game of two moves a turn the next move is the same player's, and it is
        barred the same way: the rule is about the move, not the opponent."""
        spec = ferry_barred()
        spec["rules"]["turn"] = {"moves": 2}
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = engine.initial_state()
        state = engine.apply(state, _portal_move(engine, state, "b3", "c3"))
        assert state.side_to_move == 0
        assert not _has(engine, state, "c3", "b3")

    def test_without_the_flag_the_undo_is_legal(self) -> None:
        engine = Engine(ferry())
        state = engine.initial_state()
        state = engine.apply(state, _portal_move(engine, state, "b3", "c3"))
        assert state.portal_bar == (-1, -1)
        assert _has(engine, state, "c3", "b3")

    def test_a_bar_is_in_the_key(self) -> None:
        """The same board with and without a bar is two positions."""
        engine = Engine(ferry_barred())
        state = engine.initial_state()
        state = engine.apply(state, _portal_move(engine, state, "b3", "c3"))
        assert state.key == engine._compute_hash(state)
        unbarred = state.with_(portal_bar=(-1, -1))
        assert engine._compute_hash(unbarred) != state.key
        assert _has(engine, unbarred, "c3", "b3")

    def test_the_bar_crosses_the_boundary_and_back(self) -> None:
        engine = Engine(ferry_barred())
        state = engine.initial_state()
        state = engine.apply(state, _portal_move(engine, state, "e3", "d3"))
        again = engine.apply(state, engine.legal_moves(state)[0])
        assert engine.apply(state, engine.legal_moves(state)[0]) == again
        assert state.portal_bar == (engine._square_index("d3"), engine._square_index("e3"))

    def test_a_shuffle_cannot_repeat_by_the_undo(self) -> None:
        """Ferry's shuffle, b3-c3 and back, four times: under the bar the second leg is
        not a move at all."""
        plain = Engine(ferry())
        state = plain.initial_state()
        for _ in range(4):
            state = plain.apply(state, _portal_move(plain, state, "b3", "c3"))
            state = plain.apply(state, _portal_move(plain, state, "c3", "b3"))
        assert plain.terminal(state) is not None
        engine = Engine(ferry_barred())
        state = engine.apply(engine.initial_state(), _portal_move(engine, engine.initial_state(), "b3", "c3"))
        with pytest.raises(StopIteration):
            _portal_move(engine, state, "c3", "b3")

    def test_a_hand_built_position_carries_no_bar(self) -> None:
        engine = Engine(_sparse(ferry_barred()))
        state = engine.make_state(
            {"b3": ("1o", WILD), "e4": ("1b", WILD), "e3": ("2o", WILD), "b4": ("2b", WILD),
             "e1": ("K", 0), "a6": ("K", 1)},
        )
        assert state.portal_bar == (-1, -1)
        assert _has(engine, state, "b3", "c3")


class TestTheOracleAgreesOnRandomWalks:
    """The core's portal moves equal the oracle's, which reads the bar off the move just
    played, ply by ply, and the key stays the slow hash's."""

    def test_random_walks_of_the_new_ferry(self) -> None:
        spec = ferry_barred()
        engine = Engine(spec)
        oracle = PortalOracle(spec)
        rng = random.Random(20)
        bars = 0
        for _ in range(6):
            state = engine.initial_state()
            last: tuple[int, int] | None = None
            for _ in range(80):
                if engine.terminal(state):
                    break
                position = _position(engine, state)
                player = state.side_to_move
                expected = oracle.portal_moves(position, player, last)
                assert _core_portal_moves(engine, state) == expected
                assert oracle.transits(position, player, _unmoved(state)) is not None
                bars += state.portal_bar != (-1, -1)
                # Prefer portal moves, so the walk meets the bar often.
                legal = engine.legal_moves(state)
                portal = [m for m in legal if (move_from(m), move_to(m)) in expected]
                move = rng.choice(portal if portal and rng.random() < 0.6 else legal)
                last = (move_from(move), move_to(move))
                state = engine.apply(state, move)
                assert state.key == engine._compute_hash(state)
        assert bars > 50
