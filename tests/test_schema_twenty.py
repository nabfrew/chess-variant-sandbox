"""Schema 20 (Q807): a portal may not go straight back (`docs/archive/plan-portals.md` *Ferry's shuffle*).

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
from chess_variant_sandbox.spec import SpecError, load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION
from tests.feature_inventory import _schema_twenty_uses

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


def _sluice_draft() -> dict[str, Any]:
    """Sluice as Q844 drafted it (`40cdef319293`): the archived game plus a bar per group."""
    spec = load_spec(ROOT / "archive" / "sluice.json")
    for group in spec["rules"]["portals"]:
        group["no_straight_back"] = True
    return spec


def _one_way_ferry() -> dict[str, Any]:
    """The barred Ferry, its portal stepping only towards the h-file (the same seen from
    either side of a mirrored setup)."""
    spec = ferry_barred()
    for group in spec["rules"]["portals"]:
        group["moves"] = [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "none"}]
    return spec


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
    def test_the_schema_version_is_twenty_or_later(self) -> None:
        assert SCHEMA_VERSION >= 20

    def test_the_collector_names_the_key_whatever_its_value(self) -> None:
        assert _schema_twenty_uses(ferry()) == set()
        assert _schema_twenty_uses(ferry_barred()) == {"rules.portals.no_straight_back"}
        spec = ferry()
        spec["rules"]["portals"][0]["no_straight_back"] = False
        assert _schema_twenty_uses(spec) == {"rules.portals.no_straight_back"}

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

    def test_a_bar_only_one_player_meets_is_refused(self) -> None:
        """Q879: Sluice's draft (Q844). Each gate is moved by the player it threatens alone,
        so the move after a gate moves is never its mover's: a bar no move can meet."""
        errors = validate_spec(_sluice_draft())
        for index, player in ((0, 1), (1, 0)):
            assert (
                f"rules.portals[{index}].no_straight_back is true but it could never bar a "
                "move: it binds only the very next move, which is always the other "
                "player's, and each end that moves is moved by one player only "
                f"(ends[2] only by player {player}); drop it, or let both players move an end"
            ) in errors
        assert len(errors) == 2

    def test_a_stored_spec_with_such_a_bar_is_refused_too(self) -> None:
        """Q1008d: no spec the site stores, git held or `runs/` holds carries one, so the
        load refuses it as admission does."""
        with pytest.raises(SpecError, match="could never bar a move"):
            compile_spec(_sluice_draft())

    def test_a_bar_both_players_meet_is_accepted(self) -> None:
        spec = _sluice_draft()
        spec["rules"]["portals"][0]["ends"][2]["moved_by"] = [0, 1]
        errors = validate_spec(spec)
        assert not any("rules.portals[0]" in error for error in errors)
        assert any("rules.portals[1].no_straight_back" in error for error in errors)

    def test_a_bar_a_long_turn_meets_is_accepted(self) -> None:
        """A turn of two moves lets the gate's one mover move it again at once."""
        spec = _sluice_draft()
        spec["rules"].pop("en_passant")  # refused beside a long turn, for its own reason
        spec["rules"]["turn"] = {"moves": 2}
        assert validate_spec(spec) == []

    def test_a_bar_no_move_of_the_piece_reverses_is_refused(self) -> None:
        """A portal that only ever steps one way never stands next to the square it left."""
        spec = _one_way_ferry()
        for index in (0, 1):
            assert (
                f"rules.portals[{index}].no_straight_back is true but it could never bar a "
                "move: the group's moves have none in the opposite direction to another, so "
                "an end can never go straight back onto the square it left; drop it, or give "
                "the group a move back"
            ) in validate_spec(spec)

    def test_round_a_wrap_the_reach_is_not_asked(self) -> None:
        """Round a seam two steps one way can come back, so the validator does not refuse."""
        spec = _one_way_ferry()
        spec["rules"]["wrap"] = {"files": "both"}
        assert not any("could never bar" in error for error in validate_spec(spec))

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


def _play(engine: Engine, line: list[tuple[str, str]]) -> list[Any]:
    """Every state along ``line`` (each move named by its origin and target), first move's
    result first."""
    state = engine.initial_state()
    states = []
    for origin, target in line:
        state = engine.apply(state, _portal_move(engine, state, origin, target))
        assert state.key == engine._compute_hash(state)
        states.append(state)
    return states


class TestABarAMoveCanMeetIsStillKeyed:
    """The other half: where the next mover may move the end, the bar is a different
    position, with a different key and a different move list."""

    def test_the_same_board_after_a_portal_and_after_a_knight(self) -> None:
        engine = Engine(ferry_barred())
        line = [("b3", "c3"), ("b6", "a4"), ("b1", "a3"), ("a4", "b6"), ("a3", "b1")]
        states = _play(engine, line)
        barred, free = states[0], states[4]
        assert barred.board == free.board and barred.side_to_move == free.side_to_move
        assert barred.portal_bar == (engine._square_index("c3"), engine._square_index("b3"))
        assert free.portal_bar == (-1, -1)
        assert barred.key != free.key
        assert not _has(engine, barred, "c3", "b3")
        assert _has(engine, free, "c3", "b3")

    def test_a_long_turn_keeps_only_the_bar_its_mover_meets(self) -> None:
        """Two moves a turn: the gate's first step bars its mover's second move; the second
        step is followed by Black's move, who may not move the gate, so it bars nothing."""
        spec = _sluice_draft()
        spec["rules"].pop("en_passant")  # refused beside a long turn, for its own reason
        spec["rules"]["turn"] = {"moves": 2}
        engine = Engine(spec)
        first, second = _play(engine, [("e4", "f4"), ("f4", "g4")])
        assert first.portal_bar == (engine._square_index("f4"), engine._square_index("e4"))
        assert not _has(engine, first, "f4", "e4")
        assert second.side_to_move == 1
        assert second.portal_bar == (-1, -1)
