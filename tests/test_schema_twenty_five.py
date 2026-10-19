"""Schema 25 (Q869): a facing no move turns, ``pieces[].facing.turns: false``.

`docs/archive/plan-library-audit.md`, *The rear pawn: a fixed facing* (the user, 2026-10-12): *"an
orientation can be set, but players can't change it."* A facing piece still reads its
vectors through its orientation and a placement still sets it, but the core generates no
rotation for it (``Tables::turns_of``), the move graph gives it no rotation edges, and
``facing_alignment`` leaves it out, since its frame is the setup's and not a choice.

The showcase is the rear guard: `archive/torus-less-extended.json`, whose second pawn type
``V`` (a pawn's moves written backwards, drawn right way up for one player only) became an
ordinary pawn placed facing backwards. The test that matters most is that the respelling is
the same game: the old two-pawn spelling is rebuilt here from the new file, hashes to the
frozen ``3194d3a2251b``, and agrees with the new one on perft and on random walks.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from test_backends import FACING_WEIGHTS, _assert_same_played, _facing_agent

from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator, feature_context
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_PROMOTION, MOVE_ROTATE, decode_move
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, _schema_twenty_five_uses
from chess_variant_sandbox.spec.transform import settle_schema_version

ROOT = Path(__file__).resolve().parents[1]
REAR_GUARD = ROOT / "archive" / "torus-less-extended.json"
TORUS_HAND = ROOT / "archive" / "crazyhouse-torus.json"
OLD_REAR_GUARD = "3194d3a2251b"


def _rear_guard() -> dict[str, Any]:
    return load_spec(REAR_GUARD)


def _two_pawn_spelling(spec: dict[str, Any]) -> dict[str, Any]:
    """The rear guard as it was spelled before schema 25: a second type ``V``.

    Built from the respelled file rather than kept beside it, so the two cannot drift: the
    pawn's own double step comes from rank 5 (one-based), ``V`` is the pawn's descriptors
    with every vector's rank reversed and its double step from rank 3, each promotes on its
    own rank, and the placements facing backwards become ``V``.
    """
    old = copy.deepcopy(spec)
    old["schema_version"] = 5
    pawn = next(piece for piece in old["pieces"] if piece["id"] == "P")
    pawn.pop("facing")
    rear = {"id": "V", "name": "rear pawn", "value": 1, "moves": []}
    for move in pawn["moves"]:
        if move.get("condition") == {"first_move": True}:
            move["condition"] = {"from_rank": 4}
        back = copy.deepcopy(move)
        back["vectors"] = [[dx, -dy] for dx, dy in move["vectors"]]
        if "condition" in back:
            back["condition"] = {"from_rank": 2}
        rear["moves"].append(back)
    old["pieces"].insert(1, rear)
    for placement in old["setup"]["placements"]:
        if placement.pop("facing", None) == "backward":
            placement["piece"] = "V"
    placements = old["setup"]["placements"]
    placements.insert(1, placements.pop(next(
        index for index, placement in enumerate(placements) if placement["piece"] == "V"
    )))
    promotion = old["rules"]["promotion"]
    front, back_rank = promotion["ranks"]
    old["rules"]["promotion"] = [
        {"piece": "P", "rank": front, "to": promotion["to"]},
        {"piece": "V", "rank": back_rank, "to": promotion["to"]},
    ]
    return old


def _gardner_pawns(**facing: Any) -> dict[str, Any]:
    """Gardner whose pawns have a facing, half of them placed facing backwards."""
    spec = load_spec(ROOT / "rulesets" / "gardner.json")
    spec["schema_version"] = SCHEMA_VERSION
    for piece in spec["pieces"]:
        if piece["id"] == "P":
            piece["facing"] = {"axes": 2, **facing}
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "P":
            squares = placement["squares"]
            placement["squares"] = squares[:3]
            spec["setup"]["placements"].append(
                {"piece": "P", "squares": squares[3:], "facing": "backward"}
            )
            break
    return spec


def _rotations(engine: Engine, state: Any) -> list[int]:
    return [m for m in engine.legal_moves(state) if decode_move(m)[2] == MOVE_ROTATE]


class TestTheValidator:
    def test_turns_is_a_boolean(self) -> None:
        spec = _gardner_pawns(turns="no")
        assert any("facing.turns must be a boolean" in e for e in validate_spec(spec))

    def test_an_older_version_may_not_say_it(self) -> None:
        spec = _gardner_pawns(turns=False)
        spec["schema_version"] = 24
        errors = validate_spec(spec)
        assert any("pieces[].facing.turns, added in schema 25" in e for e in errors), errors

    def test_the_collector_names_it_whatever_its_value(self) -> None:
        for value in (True, False):
            assert _schema_twenty_five_uses(_gardner_pawns(turns=value)) == {
                "pieces[].facing.turns"
            }
        assert _schema_twenty_five_uses(_gardner_pawns()) == set()

    def test_a_fixed_facing_every_piece_shares_is_refused(self) -> None:
        """No way into play but the setup leaves the initial facing, so a fixed facing
        every placement agrees with is no facing: the same game under a second hash."""
        spec = _gardner_pawns(turns=False)
        for placement in spec["setup"]["placements"]:
            placement.pop("facing", None)
        errors = validate_spec(spec)
        assert any("none ever faces another way" in e for e in errors), errors
        # The same placements with the piece free to turn are a game.
        spec["pieces"][0]["facing"].pop("turns")
        assert validate_spec(spec) == []

    def test_a_fixed_facing_held_two_ways_is_admitted(self) -> None:
        assert validate_spec(_gardner_pawns(turns=False)) == []
        assert validate_spec(_rear_guard()) == []

    def test_the_settled_version_is_twenty_five(self) -> None:
        spec = _gardner_pawns(turns=False)
        spec["schema_version"] = 1
        settle_schema_version(spec)
        assert spec["schema_version"] == 25


class TestTheCompiler:
    def test_a_turning_facing_compiles_as_before(self) -> None:
        compiled = compile_spec(_gardner_pawns())
        assert compiled.facing_turns == ()
        assert compiled.turns(0)

    def test_a_fixed_facing_is_a_table_and_a_one_way_piece(self) -> None:
        turning = compile_spec(_gardner_pawns(turns=True))
        fixed = compile_spec(_gardner_pawns(turns=False))
        pawn = fixed.piece_ids.index("P")
        assert fixed.facing_turns[pawn] is False and not fixed.turns(pawn)
        # Turning back is a two-cycle; a fixed pawn going either way never comes back.
        assert pawn not in turning.one_way_types
        assert pawn in fixed.one_way_types
        # Nothing else about the tables changes: the frames are the same readings.
        assert fixed.leaps == turning.leaps and fixed.slides == turning.slides

    def test_a_fixed_facing_that_changes_no_move_is_refused(self) -> None:
        spec = _gardner_pawns(turns=False)
        rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
        rook["facing"] = {"axes": 2, "turns": False}
        spec["setup"]["placements"].append({"piece": "R", "squares": ["c3"], "facing": "backward"})
        with pytest.raises(Exception, match="its facing changes nothing"):
            compile_spec(spec)


    @pytest.mark.parametrize(
        "path", ["archive/torus-less-extended.json", "archive/chess-oriented-pawns.json"]
    )
    def test_no_attack_source_carries_a_facing_its_type_never_holds(self, path: str) -> None:
        """Q869's pruning: a rook on a board where only the pawn faces has one frame per
        player, and its sources for the others were a quarter of the rear guard's perft."""
        compiled = compile_spec(load_spec(ROOT / path))
        for block in compiled.attack_sources[:2]:
            for sources in block:
                for source in sources:
                    value = source[1]
                    facing = compiled.facing_of(value)
                    assert facing % compiled.facing_step(compiled.type_of(value)) == 0


class TestTheCore:
    def test_no_move_turns_a_fixed_facing(self) -> None:
        turning = Engine(_gardner_pawns(turns=True))
        fixed = Engine(_gardner_pawns(turns=False))
        assert len(_rotations(turning, turning.initial_state())) == 5
        assert _rotations(fixed, fixed.initial_state()) == []
        # The moves that go somewhere are the same moves.
        others = {m for m in turning.legal_moves(turning.initial_state())
                  if decode_move(m)[2] != MOVE_ROTATE}
        assert others == set(fixed.legal_moves(fixed.initial_state()))

    def test_no_rotation_on_a_random_walk(self) -> None:
        engine = Engine(_rear_guard())
        rng = random.Random(25)
        state = engine.initial_state()
        for _ in range(120):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            assert not [m for m in moves if decode_move(m)[2] == MOVE_ROTATE]
            state = engine.apply(state, rng.choice(moves))

    def test_a_backward_pawn_promotes_on_its_own_rank(self) -> None:
        """A rear-guard pawn on a3 facing backwards double-steps to a1 and promotes on its
        next step, across the seam onto a14: the rank list holds both pawns' last ranks,
        and a backward pawn reaches the far one first."""
        spec = _rear_guard()
        spec["setup"]["placements"] = [
            {"piece": "P", "squares": ["a3"], "facing": "backward"},
            {"piece": "K", "squares": ["e4"]},
        ]
        del spec["rules"]["castling"]
        engine = Engine(spec)
        compiled = engine.spec
        a1, a2, a3, a14 = (compiled.square_index(0, rank) for rank in (0, 1, 2, 13))
        state = engine.initial_state()
        steps = {
            decode_move(m)[1:3] for m in engine.legal_moves(state) if decode_move(m)[0] == a3
        }
        assert steps == {(a2, 0), (a1, 0)}
        state = engine.apply(state, next(
            m for m in engine.legal_moves(state) if decode_move(m)[:2] == (a3, a1)
        ))
        state = engine.apply(state, engine.legal_moves(state)[0])
        from_a1 = [decode_move(m) for m in engine.legal_moves(state) if decode_move(m)[0] == a1]
        assert {(target, kind) for _, target, kind, _ in from_a1} == {(a14, MOVE_PROMOTION)}
        made = {compiled.piece_ids[promo] for _, _, _, promo in from_a1}
        assert made == {"Q", "R", "B", "N"}


class TestTheRearGuardIsTheGameItWas:
    def test_the_old_spelling_is_rebuilt_exactly(self) -> None:
        assert spec_hash(_two_pawn_spelling(_rear_guard())) == OLD_REAR_GUARD

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_perft_agrees(self, depth: int) -> None:
        old = Engine(_two_pawn_spelling(_rear_guard()))
        new = Engine(_rear_guard())
        assert new.perft(new.initial_state(), depth) == old.perft(old.initial_state(), depth)

    @pytest.mark.slow
    def test_perft_agrees_at_four(self) -> None:
        old = Engine(_two_pawn_spelling(_rear_guard()))
        new = Engine(_rear_guard())
        assert new.perft(new.initial_state(), 4) == old.perft(old.initial_state(), 4) == 2_112_956

    def test_random_walks_agree_move_for_move(self) -> None:
        """Every legal move and every ending, ply by ply, compared by squares and by the
        piece a promotion makes (the type indices differ: ``V`` is gone)."""
        old = Engine(_two_pawn_spelling(_rear_guard()))
        new = Engine(_rear_guard())

        def keyed(engine: Engine, state: Any) -> dict[tuple, int]:
            out = {}
            for move in engine.legal_moves(state):
                origin, target, kind, promo = decode_move(move)
                made = engine.spec.piece_ids[promo] if kind == MOVE_PROMOTION else promo
                out[(origin, target, kind, made)] = move
            return out

        plies = 0
        for seed in range(6):
            rng = random.Random(seed)
            a, b = old.initial_state(), new.initial_state()
            for _ in range(160):
                ka, kb = keyed(old, a), keyed(new, b)
                assert set(ka) == set(kb), (seed, plies)
                ta = old.terminal(a, list(ka.values()))
                tb = new.terminal(b, list(kb.values()))
                assert (ta and (ta.winner, ta.reason)) == (tb and (tb.winner, tb.reason))
                if ta is not None:
                    break
                choice = rng.choice(sorted(ka))
                a, b = old.apply(a, ka[choice]), new.apply(b, kb[choice])
                plies += 1
        assert plies > 500

    def test_the_rear_pawn_is_drawn_pointing_the_way_it_goes(self) -> None:
        """The user's complaint: only the first player's rear pawn drew right way up. A
        backward pawn now points down the board for the first player and up for the
        second, the way each one walks."""
        compiled = compile_spec(_rear_guard())
        assert compiled.facing_headings() == ((0, 180), (180, 0))


class TestTheHand:
    def test_a_captured_pawn_comes_back_facing_forward(self) -> None:
        compiled = compile_spec(load_spec(TORUS_HAND))
        pawn = compiled.piece_ids.index("P")
        assert compiled.facing_initial[pawn] == 0
        assert not compiled.turns(pawn)


class TestTheSurfaces:
    def test_describe_says_no_move_turns_it(self) -> None:
        text = describe_spec(_rear_guard())
        assert "no move turns it" in text
        assert "facing backward" in text
        assert "spend a whole move turning" not in text

    def test_fairy_refuses_it_by_name(self) -> None:
        with pytest.raises(Untranslatable, match="a facing no move turns"):
            translate(_gardner_pawns(turns=False))

    def test_distance_tells_fixed_from_turning(self) -> None:
        assert distance(_gardner_pawns(turns=False), _gardner_pawns(turns=True)) > 0

    def test_the_facing_term_leaves_a_fixed_facing_out(self) -> None:
        engine = Engine(_rear_guard())
        context = feature_context(engine.spec)
        reader = FeatureEvaluator(context, (1.0,) * len(VOCABULARY))
        index = VOCABULARY.index("facing_alignment")
        assert reader.features(engine, engine.initial_state(), 0)[index] == 0.0
        # The same board with pawns that may turn reads the backward ones.
        free = _rear_guard()
        free["pieces"][0]["facing"].pop("turns")
        engine = Engine(free)
        reader = FeatureEvaluator(feature_context(engine.spec), (1.0,) * len(VOCABULARY))
        assert reader.features(engine, engine.initial_state(), 0)[index] == 0.0
        assert context.facing_axes[engine.spec.piece_ids.index("P")] == 1


def _mixed() -> Engine:
    """Gardner with a fixed pawn and a turning one, placed every which way, so the facing
    term has a type to skip beside a type to read on both sides of the boundary."""
    spec = _gardner_pawns(turns=False)
    pawn = next(piece for piece in spec["pieces"] if piece["id"] == "P")
    spec["pieces"].append({
        "id": "T", "name": "turning pawn", "value": 1, "moves": copy.deepcopy(pawn["moves"]),
        "facing": {"axes": 2},
    })
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "P" and placement.get("facing") == "backward":
            placement["piece"] = "T"
            placement["squares"] = placement["squares"][:1]
    pawn_rows = [p for p in spec["setup"]["placements"] if p["piece"] == "P"]
    pawn_rows[0]["squares"] = pawn_rows[0]["squares"][:2]
    spec["setup"]["placements"].append({"piece": "P", "squares": ["e2"], "facing": "backward"})
    spec["setup"]["placements"].append({"piece": "T", "squares": ["c2"]})
    return Engine(spec)


def test_the_facing_term_skips_a_fixed_facing_the_same_way_on_both_sides() -> None:
    engine = _mixed()
    reader = FeatureEvaluator(feature_context(engine.spec), (1.0,) * len(VOCABULARY))
    index = VOCABULARY.index("facing_alignment")
    # Two turning pawns a side, one forward and one backward, so the term reads zero;
    # counting the fixed pawns (two forward, one backward) would not.
    assert reader.features(engine, engine.initial_state(), 0)[index] == 0.0
    _assert_same_played(engine, (_facing_agent(FACING_WEIGHTS, "facing-aware"), "greedy"))
