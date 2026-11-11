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
    bought_rate,
    coin_margins,
    coin_reserves,
    coin_scarcity,
    coin_value,
    income_earned,
    moves_rate,
)
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    INCOME,
    MARGIN,
    REPAIR_OPTIONS,
    RESERVE,
    SCARCITY,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game

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


def test_the_margin_bit_is_a_repair_the_core_knows() -> None:
    assert REPAIR_OPTIONS["margin"] == MARGIN == 2048


class TestTheMargin:
    """T1 (`docs/plan-player-weaknesses.md`, Q1169): a purse counts in full only up to
    `max(dearest price, 99 - INCOME_TURNS x the opening's income)`. The oracle here reads the
    spec's own keys: eight pieces on each home rank at one coin each, the dearest move a
    queen's (upkeep 2) or a pawn's (upkeep 1) seven rows out."""

    def test_hungers_margin_is_the_cap_less_four_turns_of_income(self) -> None:
        spec = _load("_purse_income.json")
        upkeep = max(piece.get("upkeep", 0) for piece in spec["pieces"])
        dearest = upkeep * 7
        assert dearest == 14
        assert coin_margins(compile_spec(spec)) == (99.0 - INCOME_TURNS * 8, 99.0 - INCOME_TURNS * 8)

    def test_the_dearest_price_floors_it(self) -> None:
        """Five coins a piece at home: the cap less four turns is negative, so the margin is
        the dearest move (a queen at three a row, seven rows out)."""
        spec = _load("_purse_income.json")
        spec["rules"]["income"] = {"per_piece": 5}
        for piece in spec["pieces"]:
            if piece["id"] == "Q":
                piece["upkeep"] = 3
        assert coin_margins(compile_spec(spec)) == (21.0, 21.0)

    def test_the_opening_income_not_the_positions(self) -> None:
        """The margin is read off the opening position: a piece leaving home lowers the side's
        income but not its margin, so the term prices no square."""
        spec = _load("_purse_income.json")
        margins = coin_margins(compile_spec(spec))
        assert margins is not None
        engine = _engine(spec)
        state = engine.initial_state()
        move = next(m for m in engine.legal_moves(state) if engine.describe_move(state, m) == "Nb1-c3")
        after = engine.apply(state, move)
        assert income_earned(engine.spec, after, 0) == 7
        assert coin_margins(engine.spec) == margins

    def test_no_income_no_margin(self) -> None:
        for name in ("_purse_upkeep.json", "_purse_market.json"):
            assert coin_margins(compile_spec(_load(name))) is None

    def test_a_coin_above_the_margin_counts_nothing(self) -> None:
        engine = _engine(_load("_purse_income.json"))
        on = dataclasses.replace(Evaluator.for_spec(engine.spec), margin=True)
        off = dataclasses.replace(on, margin=False)
        rate = coin_value(engine.spec, on.values)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d1": ("Q", 0), "d8": ("Q", 1)}
        for mine, theirs in ((10, 10), (67, 20), (80, 99), (99, 30)):
            state = engine.make_state(placement, moved=list(placement), purse=(mine, theirs))
            gap = on.evaluate(engine, state, 0, 20) - off.evaluate(engine, state, 0, 20)
            capped = min(mine, 67) - min(theirs, 67)
            assert gap == pytest.approx(rate * (capped - (mine - theirs)))

    def test_inert_without_income(self) -> None:
        engine = _engine(_load("_purse_upkeep.json"))
        on = dataclasses.replace(Evaluator.for_spec(engine.spec), margin=True)
        off = dataclasses.replace(on, margin=False)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d4": ("Q", 0)}
        state = engine.make_state(placement, moved=list(placement), purse=(99, 3))
        assert on.evaluate(engine, state, 0, 20) == off.evaluate(engine, state, 0, 20)

    def test_the_term_moves_the_players_choice(self) -> None:
        """From full purses the player with the margin spends what the other hoards: the two
        choose apart somewhere along a short walk."""
        spec = _load("_purse_income.json")
        spec["rules"]["purse"]["start"] = 99
        engine = _engine(spec)
        state = engine.initial_state()
        differ = False
        for _ply in range(6):
            moves = engine.legal_moves(state)
            picks = [
                build_agent(f"minimax:nodes=400,margin={bit},reserve=0", engine).select_move(
                    state, list(moves), random.Random(0)
                )
                for bit in (1, 0)
            ]
            differ |= picks[0] != picks[1]
            state = engine.apply(state, picks[1])
        assert differ


def test_the_scarcity_bit_is_a_repair_the_core_knows() -> None:
    assert REPAIR_OPTIONS["scarcity"] == SCARCITY == 4096


def _rooks_forage() -> dict[str, Any]:
    """Hunger's charges with income only from the two corner squares: two coins a turn."""
    spec = _load("_purse_income.json")
    spec["rules"]["income"] = {"region": ["a1", "h1"]}
    return spec


class TestTheScarcity:
    """T1b (`docs/plan-player-weaknesses.md`, Q1178): under income the moves channel counts only
    for the share of a turn's move the income does not pay, `max(0, 1 - income / mean move
    cost)`. The oracle reads the spec's own keys: eight pawns at one coin a row and seven pieces
    at two, each over a mean row of 3.5, so a mean move cost of `(8 x 3.5 + 7 x 7) / 15`."""

    MEAN_COST = (8 * 3.5 + 7 * 7.0) / 15

    def test_eight_coins_a_turn_pay_every_move(self) -> None:
        assert coin_scarcity(compile_spec(_load("_purse_income.json"))) == 0.0

    def test_two_coins_a_turn_pay_part_of_one(self) -> None:
        scarcity = coin_scarcity(compile_spec(_rooks_forage()))
        assert scarcity == pytest.approx(1 - 2 / self.MEAN_COST)

    def test_none_without_income_or_upkeep(self) -> None:
        for name in ("_purse_upkeep.json", "_purse_market.json"):
            assert coin_scarcity(compile_spec(_load(name))) is None

    def test_the_moves_channel_is_scaled(self) -> None:
        engine = _engine(_rooks_forage())
        off = Evaluator.for_spec(engine.spec)
        on = dataclasses.replace(off, scarcity=True)
        assert bought_rate(engine.spec, off.values) == 0.0
        rate = moves_rate(engine.spec, off.values)
        scaled = rate * (1 - 2 / self.MEAN_COST)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d1": ("Q", 0), "d8": ("Q", 1)}
        for mine, theirs in ((10, 10), (40, 20), (3, 60)):
            state = engine.make_state(placement, moved=list(placement), purse=(mine, theirs))
            gap = on.evaluate(engine, state, 0, 20) - off.evaluate(engine, state, 0, 20)
            assert gap == pytest.approx((scaled - rate) * (mine - theirs))

    def test_a_coin_worth_nothing_has_no_margin_and_no_income_term(self) -> None:
        engine = _engine(_load("_purse_income.json"))
        on = dataclasses.replace(
            Evaluator.for_spec(engine.spec), scarcity=True, margin=True, income=True
        )
        bare = dataclasses.replace(on, coins=False)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d1": ("Q", 0), "d8": ("Q", 1)}
        state = engine.make_state(placement, moved=list(placement), purse=(90, 5))
        assert on.evaluate(engine, state, 0, 20) == bare.evaluate(engine, state, 0, 20)

    def test_inert_without_income(self) -> None:
        engine = _engine(_load("_purse_upkeep.json"))
        off = Evaluator.for_spec(engine.spec)
        on = dataclasses.replace(off, scarcity=True)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d4": ("Q", 0)}
        state = engine.make_state(placement, moved=list(placement), purse=(99, 3))
        assert on.evaluate(engine, state, 0, 20) == off.evaluate(engine, state, 0, 20)


def test_the_reserve_bit_is_a_repair_the_core_knows() -> None:
    assert REPAIR_OPTIONS["reserve"] == RESERVE == 16384
    assert DEFAULT_REPAIRS & RESERVE and _core.DEFAULT_REPAIRS & RESERVE


class TestTheReserve:
    """T1c (`docs/plan-player-weaknesses.md`, Q1181): a purse counts in full only up to
    `INCOME_TURNS x max(0, dearest price - the opening's income)`, in place of the margin. The
    oracle reads the spec's own keys: eight pieces on each home rank at one coin each, the
    dearest move a queen's (upkeep 2) seven rows out, 14 coins."""

    def test_hungers_reserve_is_four_turns_of_what_income_does_not_pay(self) -> None:
        assert coin_reserves(compile_spec(_load("_purse_income.json"))) == (
            INCOME_TURNS * (14 - 8), INCOME_TURNS * (14 - 8)
        )

    def test_a_dearer_queen_raises_it(self) -> None:
        spec = _load("_purse_income.json")
        for piece in spec["pieces"]:
            if piece["id"] == "Q":
                piece["upkeep"] = 3
        assert coin_reserves(compile_spec(spec)) == (INCOME_TURNS * 13, INCOME_TURNS * 13)

    def test_income_that_pays_the_dearest_move_leaves_no_reserve(self) -> None:
        spec = _load("_purse_income.json")
        spec["rules"]["income"] = {"per_piece": 2}
        assert coin_reserves(compile_spec(spec)) == (0.0, 0.0)

    def test_the_cap_holds_it(self) -> None:
        """Two corners earn two coins a turn and the queen costs five a row: 4 x (35 - 2) is
        above the cap, so the reserve is the cap."""
        spec = _rooks_forage()
        for piece in spec["pieces"]:
            if piece["id"] == "Q":
                piece["upkeep"] = 5
        assert coin_reserves(compile_spec(spec)) == (99.0, 99.0)

    def test_no_income_no_reserve(self) -> None:
        for name in ("_purse_upkeep.json", "_purse_market.json"):
            assert coin_reserves(compile_spec(_load(name))) is None

    def test_a_coin_above_the_reserve_counts_nothing_margin_or_not(self) -> None:
        engine = _engine(_load("_purse_income.json"))
        off = Evaluator.for_spec(engine.spec)
        rate = coin_value(engine.spec, off.values)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d1": ("Q", 0), "d8": ("Q", 1)}
        for margin in (False, True):
            on = dataclasses.replace(off, reserve=True, margin=margin)
            for mine, theirs in ((10, 10), (24, 3), (67, 20), (80, 99), (5, 30)):
                state = engine.make_state(placement, moved=list(placement), purse=(mine, theirs))
                gap = on.evaluate(engine, state, 0, 20) - off.evaluate(engine, state, 0, 20)
                capped = min(mine, 24) - min(theirs, 24)
                assert gap == pytest.approx(rate * (capped - (mine - theirs)))

    def test_inert_without_income(self) -> None:
        engine = _engine(_load("_purse_upkeep.json"))
        off = Evaluator.for_spec(engine.spec)
        on = dataclasses.replace(off, reserve=True)
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d4": ("Q", 0)}
        state = engine.make_state(placement, moved=list(placement), purse=(99, 3))
        assert on.evaluate(engine, state, 0, 20) == off.evaluate(engine, state, 0, 20)

    def test_the_term_moves_the_players_choice(self) -> None:
        """From purses between the reserve (24) and the margin (67) the reserve player spends
        what the margin player keeps: the two choose apart somewhere along a short walk."""
        spec = _load("_purse_income.json")
        spec["rules"]["purse"]["start"] = 55
        engine = _engine(spec)
        state = engine.initial_state()
        differ = False
        for _ply in range(6):
            moves = engine.legal_moves(state)
            picks = [
                build_agent(f"minimax:nodes=400,reserve={bit}", engine).select_move(
                    state, list(moves), random.Random(0)
                )
                for bit in (1, 0)
            ]
            differ |= picks[0] != picks[1]
            state = engine.apply(state, picks[1])
        assert differ


ROOT = HERE.parent


def _without_income() -> list[Path]:
    """Every committed ruleset that does not pay income: the games the reserve must leave
    alone."""
    paths = sorted((ROOT / "rulesets").glob("*.json"))
    return [path for path in paths if not load_spec(path).get("rules", {}).get("income")]


def test_every_game_without_income_evaluates_bit_for_bit_as_before() -> None:
    """T1c's inertness, the oracle's half: in every committed game without income
    (every game without a purse among them), the evaluator under `reserve` reads each of a short
    random walk's first positions to the same float as without it."""
    paths = _without_income()
    assert len(paths) > 30
    for path in paths:
        engine = Engine(load_spec(path))
        if engine.simultaneous or engine.spec.spy is not None:
            continue  # rounds and spy phases have no alternating walk; neither pays income
        off = dataclasses.replace(Evaluator.for_spec(engine.spec), margin=True)
        on = dataclasses.replace(off, reserve=True)
        state = engine.initial_state()
        rng = random.Random(path.name)
        for _ply in range(3):
            moves = engine.legal_moves(state)
            assert on.evaluate(engine, state, 0, len(moves)) == off.evaluate(
                engine, state, 0, len(moves)
            ), path.name
            if not moves or engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(list(moves)))


@pytest.mark.parametrize(
    "spec_path",
    [
        ROOT / "rulesets" / "chess.json",
        HERE / "_purse_upkeep.json",
        HERE / "_purse_market.json",
    ],
    ids=lambda path: path.stem,
)
def test_the_core_plays_a_game_without_income_as_before(spec_path: Path) -> None:
    """T1c's inertness, the core's half: `reserve=1` and `reserve=0` play the same game, move for
    move and node for node, where nothing pays income (a purse without it included)."""
    spec = load_spec(spec_path)
    spec.setdefault("rules", {})["max_plies"] = 40
    engine = Engine(spec)
    records = [
        play_game(engine, (f"minimax:nodes=80,reserve={bit}", "minimax:nodes=80"), seed=3)
        for bit in (1, 0)
    ]
    played = [
        (record.moves, record.nodes, record.depth_sum) for record in records
    ]
    assert played[0] == played[1]
