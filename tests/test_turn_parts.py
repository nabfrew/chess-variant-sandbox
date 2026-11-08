"""TS3 (`docs/plan-tessabi.md` §3): turn parts, ``rules.turn.parts`` and ``rules.turn.order``.

A turn written as 1 to 4 parts, each naming which pieces its move may be of (``"own"``,
``"shared"`` or a list of types) and whether the mover may keep it instead (a keep,
`MOVE_KEEP`, moving nothing). Under ``order: "fixed"`` the parts are filled in the order
written; under ``"any"`` each move fills the one unfilled part admitting its piece. A move is
offered only if the turn can still be finished after it (`rust/src/parts.rs::parts_filter`).

The declared oracle here is the engine's own pseudo-legal list read against the parts: every
offered move is one a part admits and after which the turn can be finished, and every
admitted pseudo-legal move after which it can be finished is offered.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_KEEP,
    encode_move,
    move_from,
    move_promotion,
    move_type,
)
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling
from tests.feature_inventory import _schema_one_hundred_thirteen_uses

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _drover() -> dict[str, Any]:
    return load_spec(RULESETS / "drover.json")


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _neutron() -> dict[str, Any]:
    """Drover's board with Neutron's turn: the boulder first, then a runner, both owed."""
    spec = _drover()
    spec["rules"]["turn"] = {"parts": [{"pieces": "shared"}, {"pieces": "own"}]}
    return spec


def _corridor() -> dict[str, Any]:
    """A 3 x 3 board, one forward-stepping runner a side on the a-file and a boulder on b2
    either player moves, owed before the runner (Neutron's order)."""
    return {
        "schema_version": 31,
        "name": "corridor",
        "board": {"width": 3, "height": 3},
        "pieces": [
            {"id": "R", "name": "runner", "value": 1, "captures": [],
             "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]},
            {"id": "B", "name": "boulder", "value": 1, "control": "shared", "captures": [],
             "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
        ],
        "setup": {
            "mirror": False,
            "placements": [
                {"piece": "R", "squares": ["a1"]},
                {"piece": "R", "player": 1, "squares": ["c3"]},
                {"piece": "B", "player": "nobody", "squares": ["b2"]},
            ],
        },
        "rules": {
            "win": "none",
            "stalemate": "loss",
            "turn": {"parts": [{"pieces": "shared"}, {"pieces": "own"}]},
        },
    }


def _admitted(engine: Engine, state: Any, move: int) -> bool:
    """Whether some unfilled part admits ``move``'s piece (the oracle's reading of §3)."""
    parts = engine.spec.parts
    assert parts is not None
    if move_type(move) == MOVE_KEEP:
        return True
    value = state.board[move_from(move)]
    if value == 0:
        return False
    owner, piece_type = engine.spec.owner_of(value), engine.spec.type_of(value)
    side = state.side_to_move
    filled = state.parts
    count = len(parts.own)
    unfilled = [p for p in range(count) if not filled >> p & 1]
    if not parts.any_order:
        unfilled = unfilled[:1]
    return any(parts.admits(p, owner, side, piece_type) for p in unfilled)


class TestTheSpelling:
    def test_the_showcase_is_valid_and_names_its_three_features(self):
        spec = _drover()
        assert validate_spec(spec) == []
        assert _schema_one_hundred_thirteen_uses(spec) == {
            "rules.turn.parts", "rules.turn.parts.optional", "rules.turn.order.any",
        }

    @pytest.mark.parametrize("moves", [1, 2, 3])
    def test_parts_of_every_own_piece_are_turn_moves(self, moves):
        """§3: ``turn.moves: n`` is n parts each admitting every piece the mover may move, and
        the normal form writes them so: no stored name moves."""
        chess = load_spec(RULESETS / "chess.json")
        as_moves = copy.deepcopy(chess)
        as_parts = copy.deepcopy(chess)
        if moves > 1:
            as_moves["rules"]["turn"] = {"moves": moves}
        as_parts["rules"]["turn"] = {"parts": [{"pieces": "own"}] * moves, "order": "fixed"}
        assert validate_spec(as_parts) == []
        assert spec_hash(as_parts) == spec_hash(as_moves)
        assert normal_form(as_parts) == normal_form(as_moves)
        assert current_spelling(as_parts)["rules"].get("turn") == as_moves["rules"].get("turn")
        assert canonical_tables(compile_spec(as_parts)) == canonical_tables(
            compile_spec(as_moves)
        )
        assert _schema_one_hundred_thirteen_uses(as_parts) == set()

    def test_the_off_values_are_the_turn_without_them(self):
        spec = _drover()
        written = copy.deepcopy(spec)
        written["rules"]["turn"]["parts"][0]["optional"] = False
        assert spec_hash(written) == spec_hash(spec)
        neutron = _neutron()
        fixed = copy.deepcopy(neutron)
        fixed["rules"]["turn"]["order"] = "fixed"
        assert spec_hash(fixed) == spec_hash(neutron)
        assert canonical_tables(compile_spec(fixed)) == canonical_tables(compile_spec(neutron))

    def test_a_type_list_is_a_set(self):
        spec = _drover()
        spec["pieces"].append(
            {"id": "S", "name": "scout", "value": 1, "captures": [],
             "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}
        )
        spec["setup"]["placements"].append({"piece": "S", "squares": ["d3"]})
        one = copy.deepcopy(spec)
        two = copy.deepcopy(spec)
        one["rules"]["turn"] = {"parts": [{"pieces": ["R", "S"]}]}
        two["rules"]["turn"] = {"parts": [{"pieces": ["S", "R"]}]}
        assert validate_spec(one) == []
        assert spec_hash(one) == spec_hash(two)

    @pytest.mark.parametrize(
        ("turn", "words"),
        [
            ({"moves": 2, "parts": [{"pieces": "own"}]}, "beside rules.turn.moves"),
            ({"parts": [{"pieces": "own"}], "ramp": [1]}, "ramp beside rules.turn.parts"),
            ({"parts": []}, "a list of 1 to 4 parts"),
            ({"parts": [{"pieces": "own"}] * 5}, "a list of 1 to 4 parts"),
            ({"parts": [{"pieces": "mine"}]}, '"own", "shared" or a non-empty list'),
            ({"parts": [{"pieces": ["Z"]}]}, "no declared piece"),
            ({"parts": [{"pieces": "own", "optional": True}]}, "every part is optional"),
            (
                {"parts": [{"pieces": "own"}, {"pieces": ["R"], "optional": True}], "order": "any"},
                'both admitting R',
            ),
            ({"moves": 2, "order": "any"}, "only parts have an order"),
            ({"parts": [{"pieces": "own"}], "order": "sometimes"}, "rules.turn.order must be"),
        ],
    )
    def test_the_validator_refuses_by_name(self, turn, words):
        spec = _drover()
        spec["rules"]["turn"] = turn
        errors = validate_spec(spec)
        assert any(words in error for error in errors), errors

    def test_a_shared_part_needs_a_shared_piece(self):
        spec = _drover()
        spec["pieces"][1]["control"] = "majority"
        errors = validate_spec(spec)
        assert any('is "shared" but no piece is shared' in error for error in errors), errors

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("must_capture", True),
            ("drops", {"from_captures": True}),
            ("simultaneous", {}),
        ],
    )
    def test_refused_beside_a_rule_no_part_reads(self, key, value):
        spec = _drover()
        spec["rules"][key] = value
        errors = validate_spec(spec)
        assert any(f"rules.turn.parts beside rules.{key}" in error for error in errors), errors

    def test_the_bridge_holds_what_the_validator_refuses(self):
        compiled = compile_spec(_drover())
        assert rust_supports(compiled) is None

    def test_described_and_refused_by_fairy_stockfish(self):
        text = describe_spec(_drover())
        assert "A turn is 2 parts, in any order" in text
        assert "may leave instead" in text
        with pytest.raises(Untranslatable):
            translate(_drover())


class TestTheCore:
    def test_the_opening_lists_runners_the_boulder_and_the_keep(self):
        engine = _engine(_drover())
        state = engine.initial_state()
        moves = engine.legal_moves(state)
        keeps = [m for m in moves if move_type(m) == MOVE_KEEP]
        assert keeps == [encode_move(0, 0, MOVE_KEEP, 1)]
        boulder = [m for m in moves if engine.spec.owner_of(state.board[move_from(m)]) == WILD]
        assert len(boulder) == 8
        assert len(moves) == 28

    def test_a_keep_fills_the_shared_part_and_leaves_the_runner_owed(self):
        engine = _engine(_drover())
        state = engine.initial_state()
        keep = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_KEEP)
        after = engine.apply(state, keep)
        assert after.side_to_move == 0 and after.parts == 0b10
        assert after.board == state.board
        moves = engine.legal_moves(after)
        assert moves and all(
            move_type(m) != MOVE_KEEP and engine.spec.owner_of(after.board[move_from(m)]) == 0
            for m in moves
        )
        runner = moves[0]
        done = engine.apply(after, runner)
        assert done.side_to_move == 1 and done.parts == 0

    def test_the_boulder_first_leaves_only_runners(self):
        engine = _engine(_drover())
        state = engine.initial_state()
        boulder = next(
            m for m in engine.legal_moves(state)
            if engine.spec.owner_of(state.board[move_from(m)]) == WILD
        )
        after = engine.apply(state, boulder)
        assert after.side_to_move == 0 and after.parts == 0b10
        assert all(
            engine.spec.owner_of(after.board[move_from(m)]) == 0
            for m in engine.legal_moves(after)
        )

    def test_a_runner_first_offers_the_boulder_and_the_keep(self):
        engine = _engine(_drover())
        state = engine.initial_state()
        runner = next(
            m for m in engine.legal_moves(state)
            if engine.spec.owner_of(state.board[move_from(m)]) == 0
        )
        after = engine.apply(state, runner)
        assert after.parts == 0b01
        moves = engine.legal_moves(after)
        assert any(move_type(m) == MOVE_KEEP and move_promotion(m) == 1 for m in moves)
        assert all(
            move_type(m) == MOVE_KEEP or engine.spec.owner_of(after.board[move_from(m)]) == WILD
            for m in moves
        )
        keep = next(m for m in moves if move_type(m) == MOVE_KEEP)
        assert engine.apply(after, keep).side_to_move == 1

    def test_neutron_order_is_the_boulder_then_a_runner(self):
        engine = _engine(_neutron())
        state = engine.initial_state()
        moves = engine.legal_moves(state)
        assert moves and all(
            engine.spec.owner_of(state.board[move_from(m)]) == WILD for m in moves
        )
        after = engine.apply(state, moves[0])
        assert all(
            engine.spec.owner_of(after.board[move_from(m)]) == 0
            for m in engine.legal_moves(after)
        )

    def test_a_boulder_move_that_leaves_no_runner_move_is_not_offered(self):
        """The completion filter (§3, *a gatekeeper-first action must leave a legal runner
        move*): on a 3 x 3 board each side has one runner that only steps forward, and the
        boulder, owed first, may step onto the one square that runner needs."""
        engine = _engine(_corridor())
        state = engine.initial_state()
        targets = {engine.describe_move(state, m)[-2:] for m in engine.legal_moves(state)}
        assert "a2" not in targets  # it would leave a1's runner nowhere to go
        assert targets == {"a3", "b1", "b3", "c1", "c2"}  # c3 holds a runner it may not take
        blocked = next(
            m for m in engine.pseudo_legal_moves(state)
            if engine.describe_move(state, m).endswith("a2")
            and engine.spec.owner_of(state.board[move_from(m)]) == WILD
        )
        assert not engine.legal_moves(engine.apply(state, blocked))

    @pytest.mark.parametrize("make", [_drover, _neutron])
    def test_every_offer_completes_and_every_completion_is_offered(self, make):
        engine = _engine(make())
        rng = random.Random(113)
        for game in range(3):
            state = engine.initial_state()
            for _ in range(160):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                offered = set(moves)
                candidates = [
                    m for m in engine.pseudo_legal_moves(state) if _admitted(engine, state, m)
                ]
                candidates += [m for m in moves if move_type(m) == MOVE_KEEP]
                for move in candidates:
                    after = engine.apply(state, move)
                    finished = (
                        after.side_to_move != state.side_to_move
                        or engine.terminal_prefix(after) is not None
                        or bool(engine.legal_moves(after))
                    )
                    assert (move in offered) == finished, (game, engine.describe_move(state, move))
                assert all(_admitted(engine, state, m) for m in moves)
                move = rng.choice(moves)
                state = engine.apply(state, move)
                assert engine._compute_hash(state) == state.key

    def test_the_mask_is_in_the_key(self):
        engine = _engine(_drover())
        state = engine.initial_state()
        keep = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_KEEP)
        after = engine.apply(state, keep)
        assert after.board == state.board and after.side_to_move == state.side_to_move
        assert after.key != state.key
        assert engine._compute_hash(after) == after.key
