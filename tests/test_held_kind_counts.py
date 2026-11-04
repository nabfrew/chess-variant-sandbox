"""SR4d (`docs/plan-schema-next.md` §2.5 row H11, box SR4d): a held piece counts.

The ruling (2026-10-27): beside `rules.drops`, `extinction.per_type: "own"` counts a side's
hand as well as its board, so a side whose last piece of a kind is in hand has not run out
of it (it is not lost), and a kind a side holds in its opening stock is one it starts with.
The core adds the hand in `Game::short_of` under `own` (`ExtinctionRule::own`); the compiler
reads the stock into `side_types` (`compile._side_types`' `stocked`). `true` and the summed
count still read the board alone.

The oracle is this file's `_short`: a side has run out when some kind it started with
(board or stock) has fewer than `min_count` on its board and in its hand together (a
declared test oracle, rule 3). `make` is checked through `apply`; `unmake` through
`root_scores`, which makes and unmakes every root move on one game. Hand-built specs, not
chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_DROP, move_type
from chess_variant_sandbox.engine.state import GameState, Outcome
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.test_schema_twenty_seven import gap

WIDTH = 6
WAZIR = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
FERZ = {"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both"}


def _mirrored(**drops: Any) -> dict[str, Any]:
    """Both armies two wazirs `A` and two ferzes `B`, under `"own"` beside a capture hand: a
    taken enemy piece is a kind of one's own in hand, so the hand decides the count."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the pocket", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "A", "value": 3, "moves": [WAZIR]},
            {"id": "B", "value": 3, "moves": [FERZ]},
        ],
        "setup": {"placements": [
            {"piece": "A", "squares": ["b1", "e1"]}, {"piece": "B", "squares": ["c1", "d1"]},
        ]},
        "rules": {
            "win": "none", "stalemate": "draw", "max_plies": 120, "repetition_draw": 3,
            "extinction": {"pieces": ["A", "B"], "per_type": "own"},
            "drops": {"from_captures": True, **drops},
        },
    })


def _held(engine: Engine, state: GameState, player: int, **held: int) -> GameState:
    hands = [list(hand) for hand in state.hands]
    for pid, n in held.items():
        hands[player][engine.spec.piece_ids.index(pid)] = n
    return dataclasses.replace(state, hands=(tuple(hands[0]), tuple(hands[1])))


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int) -> GameState:
    state = engine.make_state(placement, side_to_move=side, ply=10)
    # `make_state` leaves the hands empty; a hand game's position holds one slot per type.
    return dataclasses.replace(state, hands=engine.initial_state().hands)


def _move(engine: Engine, state: GameState, text: str) -> int:
    return engine.find_move(state, text)


def _short(engine: Engine, state: GameState) -> tuple[bool, bool]:
    """The oracle: each side's started kinds against its board and hand together."""
    spec = engine.spec
    rule = spec.extinction
    assert rule is not None
    have = [[0] * spec.n_types, [0] * spec.n_types]
    for value in state.board[: spec.n_squares]:
        if value and spec.owner_of(value) < 2:
            have[spec.owner_of(value)][spec.type_of(value)] += 1
    for side in (0, 1):
        for piece_type, n in enumerate(state.hands[side]):
            have[side][piece_type] += n
    short = [any(have[s][t] < rule.min_count for t in rule.side_types[s]) for s in (0, 1)]
    return short[0], short[1]


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    def test_own_beside_a_hand_is_valid(self) -> None:
        assert validate_spec(gap(drops={"from_captures": True})) == []
        assert validate_spec(_mirrored()) == []

    def test_h11_is_gone(self) -> None:
        assert "H11" not in refused_ids(validate_spec(gap(drops={"from_captures": True})))

    def test_the_redirects_do_not_apply_beside_a_hand(self) -> None:
        """`true` and the summed count read the board alone, so beside a hand neither is
        `"own"`'s game under another name: both start with every kind, and no side with two
        kinds, are games here."""
        assert validate_spec(_mirrored()) == []
        one_each = gap({"pieces": ["A", "C"], "per_type": "own"}, drops={"from_captures": True})
        assert validate_spec(one_each) == []

    def test_a_kind_only_in_a_stock_is_started(self) -> None:
        """Player 0's wazirs start in hand: not *neither starts with*, not short at ply zero."""
        spec = gap(drops={"from_captures": True, "stock_by_player": {"0": {"A": 2}, "1": {}}})
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
        assert validate_spec(spec) == []

    def test_a_stock_short_of_min_count_is_refused(self) -> None:
        spec = gap(
            {"pieces": ["A", "B", "C", "E"], "per_type": "own", "min_count": 2},
            drops={"from_captures": True, "stock_by_player": {"0": {"A": 1}, "1": {}}},
        )
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
        errors = validate_spec(spec)
        assert any("player 0 starts with 1 of 'A'" in e for e in errors), errors

    def test_an_unlimited_slot_is_never_short(self) -> None:
        spec = gap(
            {"pieces": ["A", "B", "C", "E"], "per_type": "own", "min_count": 2},
            drops={"from_captures": True, "stock": {"A": "unlimited"}},
        )
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
        assert not [e for e in validate_spec(spec) if "'A'" in e]

    def test_describe_says_a_held_piece_counts(self) -> None:
        assert "Pieces in hand count" in describe_spec(_mirrored())
        assert "Pieces in hand count" not in describe_spec(gap())

    def test_fairy_stockfish_refuses_own_by_its_own_reason(self) -> None:
        with pytest.raises(Untranslatable, match='per_type "own"') as caught:
            translate(_mirrored())
        assert "counts over there and not here" not in str(caught.value)


# --------------------------------------------------------------------------- the compiler


class TestTheCompiledList:
    def test_a_stocked_kind_is_in_the_side_s_list(self) -> None:
        spec = gap(drops={"from_captures": True, "stock_by_player": {"0": {"A": 1}, "1": {}}})
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
        compiled = compile_spec(spec)
        index = {pid: i for i, pid in enumerate(compiled.piece_ids)}
        assert compiled.extinction is not None
        assert compiled.extinction.side_types == ((index["A"], index["B"]), (index["C"], index["E"]))

    def test_without_a_stock_the_lists_are_the_board_s(self) -> None:
        with_hand = build_tables(compile_spec(gap(drops={"from_captures": True})))
        assert with_hand["extinction_side_types"] == build_tables(compile_spec(gap()))[
            "extinction_side_types"
        ]


# --------------------------------------------------------------------------- the core


class TestTheAcceptance:
    def test_the_last_of_a_kind_in_hand_is_not_extinct(self) -> None:
        """Player 0's every wazir is in hand: not run out. Empty the hand and it has."""
        engine = Engine(gap(drops={"from_captures": True}))
        placement = {"c1": ("B", 0), "d1": ("B", 0), "b6": ("C", 1), "e6": ("C", 1),
                     "c6": ("E", 1), "d6": ("E", 1)}
        for side in (0, 1):
            bare = _position(engine, placement, side)
            assert engine.terminal(bare) == Outcome(winner=1, reason="extinction")
            assert engine.terminal(_held(engine, bare, 0, A=1)) is None
            # Held by the other side, it is not player 0's.
            assert engine.terminal(_held(engine, bare, 1, A=1)) == Outcome(1, "extinction")

    def test_a_stocked_army_is_not_over_at_ply_zero(self) -> None:
        spec = gap(drops={"from_captures": True, "stock_by_player": {"0": {"A": 2}, "1": {}}})
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
        engine = Engine(spec)
        state = engine.initial_state()
        assert engine.terminal(state) is None
        drops = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_DROP]
        assert drops

    def test_an_unlimited_slot_never_runs_out(self) -> None:
        spec = gap(drops={"from_captures": True, "stock": {"A": "unlimited"}})
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
        engine = Engine(spec)
        state = engine.initial_state()
        assert engine.terminal(state) is None
        drop = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_DROP)
        after = engine.apply(state, drop)
        assert engine.terminal(after) is None

    def test_the_summed_and_true_counts_read_the_board_alone(self) -> None:
        """Only `"own"` reads the hand: `true` beside a hand is unchanged."""
        spec = _mirrored()
        spec["rules"]["extinction"]["per_type"] = True
        engine = Engine(spec)
        placement = {"b1": ("A", 0), "c1": ("B", 0), "b6": ("A", 1), "c6": ("B", 1)}
        state = _held(engine, _position(engine, placement, 1), 0, A=1)
        assert engine.terminal(state) is None
        no_wazir = {"c1": ("B", 0), "b6": ("A", 1), "c6": ("B", 1)}
        state = _held(engine, _position(engine, no_wazir, 1), 0, A=1)
        assert engine.terminal(state) == Outcome(winner=1, reason="extinction")


class TestPlay:
    def test_a_capture_a_drop_and_a_capture_again(self) -> None:
        """Player 0 holds a wazir it took and stands on its last other: taking that one
        leaves it the held one, not run out; dropped back, it is on the board and the hand is
        empty; taken again, player 0 has run out."""
        engine = Engine(_mirrored())
        placement = {"b2": ("A", 0), "c1": ("B", 0), "b3": ("A", 1), "e6": ("A", 1),
                     "f6": ("B", 1)}
        state = _held(engine, _position(engine, placement, 1), 0, A=1)
        A = engine.spec.piece_ids.index("A")
        assert engine.terminal(state) is None
        state = engine.apply(state, _move(engine, state, "Ab3xb2"))
        assert state.hands[0][A] == 1 and state.hands[1][A] == 1
        assert engine.terminal(state) is None
        drop = _move(engine, state, "A@a2")
        state = engine.apply(state, drop)
        assert state.hands[0][A] == 0 and engine.terminal(state) is None
        state = engine.apply(state, _move(engine, state, "Ab2xa2"))
        assert engine.terminal(state) == Outcome(winner=1, reason="extinction")

    @pytest.mark.parametrize("seed", range(6))
    def test_random_walks_end_where_the_oracle_says(self, seed: int) -> None:
        engine = Engine(_mirrored())
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(120):
            short = _short(engine, state)
            outcome = engine.terminal(state)
            if any(short):
                mover = 1 - state.side_to_move
                loser = mover if short[mover] else 1 - mover
                assert outcome == Outcome(winner=1 - loser, reason="extinction")
                break
            assert outcome is None or outcome.reason != "extinction"
            if outcome is not None:
                break
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))

    def test_the_walks_reached_a_kind_held_off_the_board(self) -> None:
        """The walks bite: some ply had a side with none of a kind on the board and one in
        hand, which the board-only count would have ended."""
        engine = Engine(_mirrored())
        found = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            while engine.terminal(state) is None:
                rule = engine.spec.extinction
                for side in (0, 1):
                    for t in rule.side_types[side]:
                        on_board = sum(1 for v in state.board if v
                                       and engine.spec.owner_of(v) == side
                                       and engine.spec.type_of(v) == t)
                        found += on_board == 0 and state.hands[side][t] > 0
                state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        assert found


class TestUndo:
    def test_each_root_move_is_scored_on_its_own_hand(self) -> None:
        """`root_scores` makes and unmakes every root move on one game, and two plies deep
        every drop of the held wazir is made and unmade under each reply. Player 0's last
        board wazir stands where player 1 takes it; a hand left wrong by an unmake would
        score a capture as a win, or a quiet move as a loss. Each score's sign must agree
        with what `apply` and `terminal` say two plies on."""
        engine = Engine(_mirrored())
        placement = {"b2": ("A", 0), "c1": ("B", 0), "b3": ("A", 1), "e6": ("A", 1),
                     "f6": ("B", 1)}
        state = _held(engine, _position(engine, placement, 0), 0, A=1)
        values = [3.0, 3.0]
        moves = engine.legal_moves(state)
        assert any(move_type(m) == MOVE_DROP for m in moves)
        scores = engine.root_scores(state, values, 20000, 1)
        assert len(scores) == len(moves)
        for move, score in zip(moves, scores, strict=True):
            after = engine.apply(state, move)
            outcome = engine.terminal(after)
            assert outcome is None, engine.describe_move(state, move)
            assert abs(score) < 100, (engine.describe_move(state, move), score)
        deep = engine.root_scores(state, values, 200000, 2)
        for move, score in zip(moves, deep, strict=True):
            after = engine.apply(state, move)
            replies = engine.legal_moves(after)
            lost = any(
                (o := engine.terminal(engine.apply(after, reply))) is not None
                and o.winner == 1
                for reply in replies
            )
            assert (score < -100) == lost, (engine.describe_move(state, move), score)
