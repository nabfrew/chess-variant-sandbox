"""TS2 (`docs/plan-tessabi.md` §2): shared control, ``pieces[].control: "shared"``.

A type placed for nobody that either player may move as their move, whoever stands next to
it, in nobody's frame by its type's moves, onto an empty square or taking a non-royal piece of
the mover's opponent its captures allow; it stays nobody's. The core compiles it as a majority
kind with no neighbours and a per-type flag (`neutral_shared`), so `controller_of` answers -1
and `foreign.rs::push_held` asks the flag first.

`tests/neutral_oracle.py` is the declared oracle (its `shared` types); the spec is
`tests/test_neutral_pieces.py`'s scrum with its condottiere shared.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import CONTROL_MAJORITY, compile_spec
from chess_variant_sandbox.engine.moves import move_foreign
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from tests.neutral_oracle import NeutralOracle
from tests.test_neutral_pieces import WIDTH, _engine, _foreign, _position, _spec, _sq, _word


def _shared(**rules: Any) -> dict[str, Any]:
    spec = _spec(**rules)
    spec["pieces"][-1]["control"] = "shared"
    return spec


class TestEitherPlayerMovesIt:
    @pytest.mark.parametrize("side", [0, 1])
    def test_with_nobody_next_to_it_either_side_moves_it(self, side):
        engine = _engine(_shared())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "c3": ("C", 2)}, side=side)
        assert engine.controller_of(state, _sq("c3")) == -1
        assert _foreign(engine, state) == sorted(
            _word("c3", t) for t in ("b2", "c2", "d2", "b3", "d3", "b4", "c4", "d4")
        )

    def test_the_count_beside_it_changes_nothing(self):
        engine = _engine(_shared())
        crowd = {"a1": ("K", 0), "a6": ("K", 1), "c3": ("C", 2),
                 "b4": ("G", 1), "c4": ("G", 1), "d4": ("R", 1)}
        state = _position(engine, crowd, side=0)
        foreign = _foreign(engine, state)
        # Player 1 crowds it three to none, and player 0, to move, still moves it, taking the
        # other player's non-royal pieces it may take.
        assert _word("c3", "c4") in foreign and _word("c3", "d4") in foreign
        assert engine.controller_of(state, _sq("c3")) == -1

    def test_it_never_takes_the_movers_own_piece_or_a_royal(self):
        engine = _engine(_shared())
        state = _position(engine, {"b2": ("K", 0), "d4": ("K", 1), "c3": ("C", 2),
                                   "c2": ("G", 0), "c4": ("G", 1)}, side=0)
        foreign = _foreign(engine, state)
        assert _word("c3", "c4") in foreign
        assert _word("c3", "c2") not in foreign
        assert _word("c3", "d4") not in foreign

    def test_moving_it_leaves_it_nobodys(self):
        engine = _engine(_shared())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "c3": ("C", 2)}, side=0)
        after = engine.apply(state, _word("c3", "d3"))
        nobody = engine.spec.owner_of(after.board[_sq("d3")])
        assert nobody == 2
        assert _word("d3", "c3") in _foreign(engine, after)


def _walk(spec: dict[str, Any], seed: int, plies: int) -> int:
    engine = _engine(spec)
    oracle = NeutralOracle(spec, engine)
    rng = random.Random(seed)
    state = engine.initial_state()
    moved = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        core = sorted(engine.legal_moves(state))
        assert core == oracle.legal_moves(state), (seed, state.ply)
        for square in range(WIDTH * WIDTH):
            assert engine.controller_of(state, square) == oracle.holder(state, square)
        assert engine._compute_hash(state) == state.key
        foreign = [m for m in core if move_foreign(m)]
        move = rng.choice(foreign) if foreign and rng.random() < 0.5 else rng.choice(core)
        moved += bool(move_foreign(move))
        state = engine.apply(state, move)
    return moved


def _walk_spec() -> dict[str, Any]:
    spec = _shared()
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "f1"]},
        {"piece": "G", "squares": ["a2", "b2", "e2", "f2"]},
        {"piece": "C", "player": "nobody", "squares": ["b3", "e3"]},
    ]
    return spec


class TestTheOracle:
    def test_three_walks_agree(self):
        assert sum(_walk(_walk_spec(), seed, 60) for seed in range(3)) > 0

    def test_the_showcase_walks_agree(self):
        spec = load_spec("rulesets/boulder.json")
        engine = Engine(spec)
        oracle = NeutralOracle(spec, engine)
        state = engine.initial_state()
        rng = random.Random(0)
        for _ in range(40):
            if engine.terminal(state) is not None:
                break
            core = sorted(engine.legal_moves(state))
            assert core == oracle.legal_moves(state)
            assert engine._compute_hash(state) == state.key
            state = engine.apply(state, rng.choice(core))

    @pytest.mark.slow
    def test_twenty_walks_agree(self):
        assert sum(_walk(_walk_spec(), seed, 120) for seed in range(20)) > 20


class TestTheSpec:
    def test_the_flag_crosses_only_for_a_shared_type(self):
        shared = compile_spec(_shared())
        assert shared.neutrals is not None
        assert shared.neutrals.kind[3] == CONTROL_MAJORITY and shared.neutrals.shared == (3,)
        assert not shared.neutrals.has_count
        assert build_tables(shared)["neutral_shared"] == [0, 0, 0, 1]
        held = compile_spec(_spec())
        assert held.neutrals is not None and held.neutrals.shared == ()
        assert build_tables(held)["neutral_shared"] == []
        assert "neutral_shared" not in canonical_tables(held)

    def test_the_value_names_a_new_game(self):
        assert spec_hash(_shared()) != spec_hash(_spec())

    def test_fairy_stockfish_refuses_it_by_name(self):
        with pytest.raises(Untranslatable, match="either side may move"):
            translate(_shared())

    def test_the_words_say_it(self):
        words = describe_spec(_shared())
        assert "is shared: either player may move it as their move" in words
        assert "either player may move it (see the rules)" in words


def _refused(spec: dict[str, Any], words: str) -> None:
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


class TestRefusals:
    def test_beside_takes_royal(self):
        spec = _shared()
        spec["pieces"][-1]["takes_royal"] = True
        _refused(spec, "a piece both players move cannot be held against one")

    def test_in_rounds(self):
        spec = _shared(win="royal_capture", simultaneous={})
        _refused(spec, 'control "shared"')

    def test_with_a_price(self):
        spec = _shared(purse={"start": 2, "market": ["G"]})
        spec["pieces"][2]["price"] = 2
        spec["pieces"][-1]["price"] = 3
        _refused(spec, 'control "shared" has a price')

    def test_on_a_royal(self):
        spec = _shared()
        spec["pieces"][0]["control"] = "shared"
        _refused(spec, "on a royal")

    def test_a_goal_that_counts_and_admits_it(self):
        spec = _shared(win="none", stalemate="loss")
        spec["rules"]["reach"] = {"squares": ["a6"], "count": 2}
        _refused(spec, 'control "majority" or "shared"')

    def test_a_goal_that_counts_only_other_pieces_takes_it(self):
        spec = _shared(win="none", stalemate="loss")
        spec["setup"]["placements"][2]["squares"] = ["b1", "c1"]
        spec["rules"]["reach"] = {"piece": "G", "squares": ["a6", "b6"], "count": 2}
        assert validate_spec(spec) == []
