"""H4 (`docs/plan-hunger.md` §5, §R4): a coin's worth without a market.

The evaluator's coin term reads `coin_value`, the best rate over every channel coins buy. H4
adds **moves under upkeep**: `mean value / mean move cost` over the opening position's
non-royal pieces with a move cost, a type's move cost being the mean of its row of the compiled
move-cost table over the squares it may stand on (`PurseRule.upkeep_mean`). And **the income
term**, a minimax player's under the `income` search bit: each coin a side's pieces earn as they
stand counts `INCOME_TURNS` coins at that worth, mine minus theirs.

Specs, not chess: `tests/_purse_income.json` and `tests/_purse_upkeep.json` (chess with Hunger
Chess's charges) and edits of them; `tests/_purse_market.json` for a purse without upkeep. The
oracle for the means and the rate is this file's arithmetic over the spec's own keys (a declared
test oracle), never the compiled tables it checks.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import build_agent
from chess_variant_sandbox.agents.evaluation import (
    INCOME_TURNS,
    Evaluator,
    coin_value,
    income_earned,
    moves_rate,
)
from chess_variant_sandbox.agents.minimax import DEFAULT_REPAIRS, INCOME, REPAIR_OPTIONS
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec

HERE = Path(__file__).parent


def _load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _engine(spec: dict[str, Any]) -> Engine:
    return Engine(spec)


def test_the_horizon_is_held_to_the_cores() -> None:
    assert _core.INCOME_TURNS == INCOME_TURNS == 4.0


def test_the_income_bit_is_a_repair_the_core_knows() -> None:
    assert REPAIR_OPTIONS["income"] == INCOME == 1024
    assert DEFAULT_REPAIRS == _core.DEFAULT_REPAIRS


class TestTheMovesChannel:
    def test_a_types_mean_move_cost_is_its_row_over_the_board(self) -> None:
        """Eight rows from 0 to 7 under the `rows` measure: the mean row is 3.5, times the
        type's upkeep; a royal with none costs 0. Both players read alike (reflected)."""
        spec = _load("_purse_income.json")
        compiled = compile_spec(spec)
        assert compiled.purse is not None
        upkeep = {piece["id"]: piece.get("upkeep", 0) for piece in spec["pieces"]}
        ids = [piece["id"] for piece in spec["pieces"]]
        expected = tuple(upkeep[pid] * 3.5 for pid in ids)
        assert compiled.purse.upkeep_mean == expected + expected

    def test_a_wall_is_no_square_to_stand_on(self) -> None:
        spec = _load("_purse_income.json")
        spec["rules"]["obstacles"] = {"squares": ["d4", "e5"]}
        means = compile_spec(spec).purse.upkeep_mean  # type: ignore[union-attr]
        # Rows 3 and 4 each lose one square: 64 - 2 squares, sum 8 * 28 - 3 - 4.
        assert means[0] == pytest.approx((8 * 28 - 3 - 4) / 62)

    def test_the_rate_is_mean_value_over_mean_cost(self) -> None:
        spec = _load("_purse_income.json")
        engine = _engine(spec)
        values = Evaluator.for_spec(engine.spec).values
        upkeep = {piece["id"]: piece.get("upkeep", 0) for piece in spec["pieces"]}
        index = {piece["id"]: i for i, piece in enumerate(spec["pieces"])}
        value = cost = 0.0
        for placement in spec["setup"]["placements"]:
            pid = placement["piece"]
            if not upkeep[pid]:
                continue
            for _square in placement["squares"]:
                value += values[index[pid]]
                cost += upkeep[pid] * 3.5
        mirrored = 2  # the second player's army is the first's, reflected
        assert moves_rate(engine.spec, values) == pytest.approx(mirrored * value / (mirrored * cost))
        assert coin_value(engine.spec, values) == moves_rate(engine.spec, values) > 0

    def test_no_upkeep_no_channel(self) -> None:
        """A purse game without upkeep keeps the market's rate exactly: the channel adds 0."""
        engine = _engine(_load("_purse_market.json"))
        values = Evaluator.for_spec(engine.spec).values
        assert engine.spec.purse is not None and engine.spec.purse.upkeep_mean == ()
        assert moves_rate(engine.spec, values) == 0.0
        assert coin_value(engine.spec, values) == max(values[3] / 2, values[2] / 4)
        assert build_tables(engine.spec)["purse_upkeep_mean"] == []

    def test_the_mean_crosses_beside_the_table(self) -> None:
        compiled = compile_spec(_load("_purse_upkeep.json"))
        tables = build_tables(compiled)
        assert tables["purse_upkeep_mean"] == list(compiled.purse.upkeep_mean)  # type: ignore[union-attr]
        assert len(tables["purse_upkeep_mean"]) == 2 * compiled.n_types

    def test_a_cheaper_queen_is_worth_more_coins(self) -> None:
        """The channel is the tables' mean, so an arm that raises an upkeep lowers the rate."""
        spec = _load("_purse_income.json")
        dearer = copy.deepcopy(spec)
        for piece in dearer["pieces"]:
            if piece["id"] == "Q":
                piece["upkeep"] = 3
        base, raised = _engine(spec), _engine(dearer)
        assert moves_rate(raised.spec, Evaluator.for_spec(raised.spec).values) < moves_rate(
            base.spec, Evaluator.for_spec(base.spec).values
        )


class TestTheIncomeTerm:
    def test_off_unless_asked(self) -> None:
        engine = _engine(_load("_purse_income.json"))
        assert Evaluator.for_spec(engine.spec).income is False

    def test_a_piece_off_home_costs_its_income(self) -> None:
        """A knight leaves the first rank: the mover earns one coin less a turn, which the term
        prices at `INCOME_TURNS` coins at the coin's worth; the purse it paid is the coin term's."""
        engine = _engine(_load("_purse_income.json"))
        on = dataclasses.replace(Evaluator.for_spec(engine.spec), income=True)
        off = dataclasses.replace(on, income=False)
        state = engine.initial_state()
        move = next(m for m in engine.legal_moves(state) if engine.describe_move(state, m) == "Nb1-c3")
        after = engine.apply(state, move)
        assert income_earned(engine.spec, after, 0) == income_earned(engine.spec, state, 0) - 1
        rate = coin_value(engine.spec, on.values)
        for position, edge in ((state, 0), (after, -1)):
            gap = on.evaluate(engine, position, 0, 20) - off.evaluate(engine, position, 0, 20)
            assert gap == pytest.approx(rate * INCOME_TURNS * edge)

    def test_the_term_moves_the_players_choice(self) -> None:
        """The bit is not inert where income is paid: the player with the term and the player
        without it choose apart somewhere along a short walk. From full purses, since from
        the opening's ten coins both muster until the cap makes a pass waste the income (the
        coin's worth at work, `docs/plan-hunger.md` §5.1)."""
        spec = _load("_purse_income.json")
        spec["rules"]["purse"]["start"] = 99
        engine = _engine(spec)
        state = engine.initial_state()
        differ = False
        for _ply in range(6):
            moves = engine.legal_moves(state)
            picks = [
                build_agent(f"minimax:nodes=400,income={bit}", engine).select_move(
                    state, list(moves), random.Random(0)
                )
                for bit in (1, 0)
            ]
            differ |= picks[0] != picks[1]
            state = engine.apply(state, picks[1])
        assert differ


def test_mobility_reads_the_filtered_list() -> None:
    """§2.3, confirmed for H4: a piece the purse cannot move counts as immobile, because the
    evaluator's mobility is the length of the core's filtered list."""
    engine = _engine(_load("_purse_upkeep.json"))
    placement = {"e1": ("K", 0), "e8": ("K", 1), "d4": ("Q", 0)}
    broke = engine.make_state(placement, moved=list(placement), purse=(0, 0))
    rich = engine.make_state(placement, moved=list(placement), purse=(99, 99))
    stranded = engine.legal_moves(broke)
    assert len(stranded) < len(engine.legal_moves(rich))
    # Only what costs nothing is left: the king (no upkeep) and the queen onto her first row.
    assert all(engine.move_cost(broke, m) == 0 for m in stranded)
    assert any(engine.move_cost(rich, m) > 0 for m in engine.legal_moves(rich))
    evaluator = Evaluator.for_spec(engine.spec)
    assert evaluator.evaluate(engine, broke, 0) == evaluator.evaluate(engine, broke, 0, len(stranded))
