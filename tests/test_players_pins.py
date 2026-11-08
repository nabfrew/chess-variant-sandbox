"""Pins on the smaller players' edges (Q1152, the `agents/` mutation group).

Each test holds one value a player, a builder or a helper produces on an input no other fast
test gives: a default an agent string leaves out, a native tuple whole, a move counter after
one move, a fallback draw past the strategy's total, a dedup that is not the last candidate.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox.agents import build_agent, native_spec, seeds
from chess_variant_sandbox.agents.base import SearchStats
from chess_variant_sandbox.agents.fairy_stockfish import FairyStockfishAgent
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.heuristics import Heuristic
from chess_variant_sandbox.agents.matrix import draw, greedy_announcement, greedy_answer
from chess_variant_sandbox.agents.objective import Objective
from chess_variant_sandbox.agents.valuations import Valuation
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.fairy import FairyEngineError
from chess_variant_sandbox.spec import load_spec

HERE = Path(__file__).parent
DULL = HERE.parent / "rulesets" / "dull.json"
ROUNDS = HERE / "_simultaneous_gardner.json"


def _dull() -> Engine:
    return Engine(load_spec(DULL))


# ------------------------------------------------------------------ simple.py


def test_a_random_player_counts_one_move_and_every_candidate() -> None:
    engine = _dull()
    agent: Any = build_agent("random", engine)
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    agent.select_move(state, moves, random.Random(1))
    assert agent.stats.moves == 1
    assert agent.stats.nodes == len(moves)


def test_a_greedy_player_counts_one_move_per_choice() -> None:
    engine = _dull()
    agent: Any = build_agent("greedy", engine)
    state = engine.initial_state()
    agent.select_move(state, engine.legal_moves(state), random.Random(1))
    assert agent.stats.moves == 1


def test_the_native_random_and_mcts_tuples_are_these() -> None:
    assert native_spec("random") == ("random", 0, 0.0, 0, 0, 0.0, 0.0, (), (), 0.0, b"", 0)
    assert native_spec("mcts") == (
        "mcts", 2000, 0.0, 0, 60, 1.4, 0.0, (), (), 0.0, b"", 0
    )


# ------------------------------------------------------------------ mcts.py


def test_an_mcts_player_without_options_rolls_out_sixty_plies() -> None:
    agent: Any = build_agent("mcts", _dull())
    assert agent.rollout_depth == 60
    assert agent.budget == 2000


# ------------------------------------------------------------------ matrix.py


def test_draw_past_the_total_falls_to_the_last_move() -> None:
    """A strategy summing under the draw leaves the loop: the last move, not the first or
    the second of three."""
    assert draw([10, 20, 30], [0.0, 0.0, 0.0], random.Random(0)) == 30


def test_greedy_spy_halves_with_one_move_count_it_and_play_it() -> None:
    stats = SearchStats()
    assert greedy_announcement(None, stats, None, 0, ([7], [1, 2])) == 7  # type: ignore[arg-type]
    assert stats.moves == 1
    assert greedy_answer(None, stats, None, 1, ([1, 2], [9]), 1) == 9  # type: ignore[arg-type]
    assert stats.moves == 2


def test_a_matrix_player_without_a_budget_has_sixteen_hundred_nodes() -> None:
    agent: Any = build_agent("matrix", Engine(load_spec(ROUNDS)))
    assert agent.nodes == 1600
    assert agent.name == "matrix:nodes=1600"


# ------------------------------------------------------------------ heuristics.py


def test_a_heuristic_is_frozen() -> None:
    weights = (1.0,) + (0.0,) * (len(VOCABULARY) - 1)
    heuristic = Heuristic(name="a", plan="b", weights=weights)
    with pytest.raises(dataclasses.FrozenInstanceError):
        heuristic.name = "c"  # type: ignore[misc]


# ------------------------------------------------------------------ objective.py


def test_the_objective_declares_values_read_only() -> None:
    assert isinstance(inspect.getattr_static(Objective, "values"), property)


# ------------------------------------------------------------------ valuations.py


def test_a_valuation_without_a_step_count_kept_none() -> None:
    valuation = Valuation(
        ruleset_hash="h",
        procedure="p",
        piece_ids=("a",),
        vector=(2.0,),
        derived=(1.0,),
        weights=(),
        accepted=True,
        accept_score=0.0,
        accept_error=0.0,
        accept_games=0,
    )
    assert valuation.accepted_steps == 0
    assert valuation.as_dict()["accepted_steps"] == 0


# ------------------------------------------------------------------ seeds.py


def test_a_hand_written_plan_that_repeats_an_earlier_one_is_skipped_not_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = (
        ("first", "x", {"advance": 1.0}, "always"),
        ("again", "x", {"advance": 1.0}, "always"),
        ("third", "x", {"centrality": 1.0}, "always"),
    )
    monkeypatch.setattr(seeds, "_PLANS", plans)
    chosen = seeds.handwritten_set(_dull().spec, k=4)
    assert [h.name for h in chosen.heuristics] == ["material-and-mobility", "first", "third"]


class _Scripted:
    """An RNG whose ``uniform`` reads a script: the vectors ``random_set`` will draw."""

    def __init__(self, seed: int) -> None:
        n = len(VOCABULARY)
        a = [0.5] + [0.0] * (n - 1)
        b = [0.0, 0.25] + [0.0] * (n - 2)
        self._values = iter([0.0] * n + a + a + b)

    def uniform(self, low: float, high: float) -> float:
        return next(self._values)


def test_a_random_set_skips_a_zero_and_a_repeated_draw_and_keeps_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seeds, "random", SimpleNamespace(Random=_Scripted))
    drawn = seeds.random_set(_dull().spec, k=3)
    assert [h.name for h in drawn.heuristics] == [
        "material-and-mobility", "random-1", "random-2"
    ]
    assert drawn.heuristics[1].weights[0] == 0.5
    assert drawn.heuristics[2].weights[1] == 0.25


# ------------------------------------------------------------------ fairy_stockfish.py


def test_an_untranslatable_ruleset_is_refused_before_any_process_starts() -> None:
    compiled = dataclasses.replace(_dull().spec, source="")
    with pytest.raises(FairyEngineError, match="Fairy-Stockfish cannot play .*carries no source"):
        FairyStockfishAgent(SimpleNamespace(spec=compiled))  # type: ignore[arg-type]


def test_a_fairy_player_with_one_move_plays_it_and_counts_no_search() -> None:
    agent = object.__new__(FairyStockfishAgent)
    agent.nodes = 5
    agent.stats = SearchStats()
    assert agent.select_move(None, [42], random.Random(0)) == 42  # type: ignore[arg-type]
    assert (agent.stats.moves, agent.stats.nodes) == (1, 0)


# ------------------------------------------------------------------ batch 2


def test_a_heuristic_set_with_no_vocabulary_version_reads_as_minus_one() -> None:
    from chess_variant_sandbox.agents.heuristics import HeuristicError, HeuristicSet

    with pytest.raises(HeuristicError, match=r"written against vocabulary -1 and"):
        HeuristicSet.from_dict({})


def test_uct_flips_only_the_children_the_other_side_moves_at() -> None:
    """One child the chooser moves at again, one the opponent moves at: the opponent's
    child is read as ``1 - winrate`` (0.7), beating the chooser's own 0.6."""
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    agent = MCTSAgent(None, nodes=1, exploration=0.0)  # type: ignore[arg-type]
    own = SimpleNamespace(visits=1, wins=0.6, state=SimpleNamespace(side_to_move=0))
    other = SimpleNamespace(visits=1, wins=0.3, state=SimpleNamespace(side_to_move=1))
    node = SimpleNamespace(visits=2, state=SimpleNamespace(side_to_move=0), children=[own, other])
    assert agent._best_child(node) is other  # type: ignore[arg-type]


def test_a_check_count_of_one_is_a_check_count() -> None:
    assert seeds._declares(SimpleNamespace(check_limit=1), "check_limit")  # type: ignore[arg-type]
    assert not seeds._declares(SimpleNamespace(check_limit=0), "check_limit")  # type: ignore[arg-type]


def test_the_null_control_is_seed_zero_and_k_long() -> None:
    drawn = seeds.random_set(_dull().spec, k=2)
    assert [h.name for h in drawn.heuristics] == ["material-and-mobility", "random-1"]
    assert drawn.heuristics[1].weights == (
        0.6888, 0.5159, -0.1589, -0.4822, 0.0225, -0.1901, 0.5676, -0.3934, -0.0468,
        0.1668, 0.8162, 0.0094, -0.4363, 0.5116, 0.2367,
    )


def _valuation() -> Valuation:
    return Valuation(
        ruleset_hash="h", procedure="p", piece_ids=("a",), vector=(2.0,), derived=(1.0,),
        weights=(), accepted=True, accept_score=0.0, accept_error=0.0, accept_games=0,
    )


def test_a_valuation_is_frozen_and_its_ladder_defaults_are_even() -> None:
    valuation = _valuation()
    assert (valuation.ladder_score, valuation.ladder_error) == (0.5, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        valuation.accepted = False  # type: ignore[misc]


def test_a_valuation_with_no_procedure_version_reads_as_minus_one() -> None:
    from chess_variant_sandbox.agents.valuations import ValuationError

    with pytest.raises(ValuationError, match=r"procedure version -1 and"):
        Valuation.from_dict({})


def test_a_valuation_is_saved_as_two_space_json(tmp_path: Path) -> None:
    text = _valuation().save(tmp_path / "v.json").read_text(encoding="utf-8")
    assert text.startswith('{\n  "ruleset_hash": "h",\n  "procedure": "p",\n')
    assert text.endswith("\n}\n")


# ------------------------------------------------------------------ batch 3


def test_a_set_names_its_pieces_only_when_a_heuristic_prices_them() -> None:
    from chess_variant_sandbox.agents.heuristics import HeuristicSet

    weights = (1.0,) + (0.0,) * (len(VOCABULARY) - 1)
    plain = HeuristicSet(
        ruleset_hash="h", heuristics=(Heuristic(name="a", plan="b", weights=weights),),
        piece_ids=("x",),
    )
    assert "piece_ids" not in plain.as_dict()


def test_the_native_matrix_and_a_bare_mcts_agent_carry_their_defaults() -> None:
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    spec = native_spec("matrix")
    assert spec is not None and spec[1] == 1600
    assert MCTSAgent(None).rollout_depth == 60  # type: ignore[arg-type]


def test_the_null_control_has_six_members_by_default() -> None:
    assert len(seeds.random_set(_dull().spec).heuristics) == 6


def test_a_stored_valuation_without_a_step_count_kept_none() -> None:
    data = _valuation().as_dict()
    data["weights"] = [0.0] * len(VOCABULARY)
    del data["accepted_steps"]
    assert Valuation.from_dict(data).accepted_steps == 0


# ------------------------------------------------------------------ batch 4


def test_the_fairy_default_budget_is_two_hundred_thousand_nodes() -> None:
    from chess_variant_sandbox.agents.fairy_stockfish import DEFAULT_NODES

    assert DEFAULT_NODES == 200_000


def test_a_set_written_without_its_vocabulary_list_loads() -> None:
    from chess_variant_sandbox.agents.heuristics import HeuristicSet

    weights = (1.0,) + (0.0,) * (len(VOCABULARY) - 1)
    written = HeuristicSet(
        ruleset_hash="h", heuristics=(Heuristic(name="a", plan="b", weights=weights),)
    ).as_dict()
    del written["vocabulary"]
    assert HeuristicSet.from_dict(written).heuristics[0].weights == weights


def test_draw_refuses_a_strategy_of_another_length() -> None:
    with pytest.raises(ValueError):
        draw([1, 2], [0.0], random.Random(0))


def test_a_matrix_agent_built_directly_has_sixteen_hundred_nodes() -> None:
    from chess_variant_sandbox.agents.matrix import MatrixAgent

    assert MatrixAgent(Engine(load_spec(ROUNDS))).nodes == 1600


def test_the_native_greedy_and_matrix_tuples_are_these() -> None:
    assert native_spec("greedy") == (
        "greedy", 0, 0.05, 0, 0, 0.0, 16.0, (), (), 1.0, b"", 0
    )
    assert native_spec("matrix") == (
        "matrix", 1600, 0.05, 64, 0, 0.0, 16.0, (), (), 1.0, b"", 0
    )


def test_an_mcts_budget_is_at_least_one_simulation() -> None:
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    assert MCTSAgent(None, nodes=1).budget == 1  # type: ignore[arg-type]
    assert MCTSAgent(None, nodes=0).budget == 1  # type: ignore[arg-type]


def test_a_valuation_counts_no_comparisons_by_default_and_refuses_a_non_finite_price() -> None:
    from chess_variant_sandbox.agents.valuations import ValuationError

    assert _valuation().comparisons == 0
    with pytest.raises(ValuationError, match="piece 0 is not worth a finite amount"):
        dataclasses.replace(_valuation(), vector=(float("nan"),))


def test_load_all_counts_what_it_registered(tmp_path: Path) -> None:
    from chess_variant_sandbox.agents import valuations

    stored = dataclasses.replace(
        _valuation(), ruleset_hash="q1152load", weights=(0.0,) * len(VOCABULARY)
    )
    stored.save(tmp_path / "q1152load-p.json")
    try:
        assert valuations.load_all(tmp_path) == 1
    finally:
        valuations._VALUATIONS.pop(("p", "q1152load"), None)


# ------------------------------------------------------------------ batch 5


class _Uci:
    def load(self, variant: Any, path: Any) -> None:
        pass

    def best_move(self, text: str, nodes: int) -> str:
        return "a1a2"

    def close(self) -> None:
        pass


def test_a_fairy_player_is_given_at_least_one_node(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_variant_sandbox.agents import fairy_stockfish

    monkeypatch.setattr(fairy_stockfish, "translate", lambda spec: SimpleNamespace())
    monkeypatch.setattr(fairy_stockfish, "config_path", lambda variant: None)
    monkeypatch.setattr(fairy_stockfish, "UciEngine", _Uci)
    agent = FairyStockfishAgent(SimpleNamespace(spec=None), nodes=1)  # type: ignore[arg-type]
    assert (agent.nodes, agent.name) == (1, "fairystockfish:nodes=1")
    assert FairyStockfishAgent(SimpleNamespace(spec=None), nodes=0).nodes == 1  # type: ignore[arg-type]


def test_a_fairy_position_without_check_counting_states_no_check_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_variant_sandbox.agents import fairy_stockfish

    seen: dict[str, Any] = {}

    def fen(*args: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "fen"

    monkeypatch.setattr(
        fairy_stockfish, "fairy_position",
        SimpleNamespace(fen=fen, move_of=lambda spec, state, moves, *rest: moves[1]),
    )
    agent = object.__new__(FairyStockfishAgent)
    agent.engine = SimpleNamespace(spec=SimpleNamespace(check_limit=3))
    agent.variant = SimpleNamespace(
        chars="", king=None, drops=False, check_counting=False, castling=None,
        promotion_suffix="",
    )
    agent.nodes = 5
    agent.stats = SearchStats()
    agent._uci = _Uci()
    assert agent.select_move(None, [7, 8], random.Random(0)) == 8  # type: ignore[arg-type]
    assert seen["check_limit"] == 0
    assert (agent.stats.moves, agent.stats.nodes) == (1, 5)


def test_a_heuristic_may_price_exactly_the_most_piece_types() -> None:
    from chess_variant_sandbox.agents.heuristics import HeuristicError
    from chess_variant_sandbox.spec.schema import MAX_PIECE_TYPES

    weights = (1.0,) + (0.0,) * (len(VOCABULARY) - 1)
    full = Heuristic(name="a", plan="b", weights=weights, values=(1.0,) * MAX_PIECE_TYPES)
    assert len(full.values) == 24
    with pytest.raises(HeuristicError, match="at most 24 entries"):
        Heuristic(name="a", plan="b", weights=weights, values=(1.0,) * 25)


def test_a_negative_zero_weight_is_the_same_heuristic_as_zero() -> None:
    import math

    rest = (0.0,) * (len(VOCABULARY) - 2)
    signed = Heuristic(name="a", plan="b", weights=(1.0, -0.0) + rest)
    plain = Heuristic(name="a", plan="b", weights=(1.0, 0.0) + rest)
    assert math.copysign(1.0, signed.normalised[1]) == 1.0
    assert signed.digest == plain.digest


def test_a_greedy_player_built_directly_weighs_mobility_as_the_builder_does() -> None:
    from chess_variant_sandbox.agents.simple import GreedyAgent

    engine = _dull()
    built: Any = build_agent("greedy", engine)
    assert GreedyAgent(engine).evaluator == built.evaluator


def test_a_greedy_player_never_plays_a_move_that_loses_at_once() -> None:
    from chess_variant_sandbox.agents.simple import GreedyAgent

    lost = SimpleNamespace(is_draw=False, winner=1)
    agent = object.__new__(GreedyAgent)
    agent.engine = SimpleNamespace(
        spy=False, apply=lambda state, move: move,
        terminal=lambda successor: lost if successor == 1 else None,
    )
    agent.evaluator = SimpleNamespace(evaluate=lambda engine, successor, player: 0.0)
    agent.stats = SearchStats()
    state = SimpleNamespace(side_to_move=0)
    assert agent.select_move(state, [1, 2], random.Random(0)) == 2  # type: ignore[arg-type]


def test_a_royal_slot_reads_as_an_unmoved_ratio() -> None:
    priced = dataclasses.replace(
        _valuation(), piece_ids=("a", "b"), vector=(0.0, 3.0), derived=(0.0, 2.0)
    )
    assert priced.ratios == (1.0, 1.5)


# ------------------------------------------------------------------ batch 6 (Q1153)


def test_a_greedy_player_prefers_half_a_unit_to_a_draw() -> None:
    """A draw is worth exactly zero: an evaluation of one half beats it."""
    from chess_variant_sandbox.agents.simple import GreedyAgent

    drawn = SimpleNamespace(is_draw=True, winner=None)
    agent = object.__new__(GreedyAgent)
    agent.engine = SimpleNamespace(
        spy=False, apply=lambda state, move: move,
        terminal=lambda successor: drawn if successor == 1 else None,
    )
    agent.evaluator = SimpleNamespace(evaluate=lambda engine, successor, player: 0.5)
    agent.stats = SearchStats()
    state = SimpleNamespace(side_to_move=0)
    assert agent.select_move(state, [1, 2], random.Random(0)) == 2  # type: ignore[arg-type]


def test_a_valuation_is_not_exhausted_by_default() -> None:
    assert _valuation().exhausted is False


def test_a_bare_mcts_agent_explores_at_one_point_four() -> None:
    from chess_variant_sandbox.agents.mcts import EXPLORATION, MCTSAgent

    assert EXPLORATION == 1.4
    assert MCTSAgent(None).exploration == 1.4  # type: ignore[arg-type]


def test_an_mcts_player_counts_one_move_per_search() -> None:
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    engine = _dull()
    agent = MCTSAgent(engine, nodes=2, rollout_depth=2)
    state = engine.initial_state()
    agent.select_move(state, engine.legal_moves(state), random.Random(0))
    assert (agent.stats.moves, agent.stats.nodes) == (1, 2)


def test_a_matrix_agent_of_one_node_keeps_one_node() -> None:
    from chess_variant_sandbox.agents.matrix import MatrixAgent

    agent = MatrixAgent(Engine(load_spec(ROUNDS)), nodes=1)
    assert agent.nodes == 1
    assert MatrixAgent(Engine(load_spec(ROUNDS)), nodes=0).nodes == 1


def test_a_matrix_search_stops_at_its_depth_however_much_budget_is_left() -> None:
    """Two moves each at the start, max depth one: the four cells of the first pass and no
    more, with 400 nodes unspent; depth two costs 333."""
    from chess_variant_sandbox.agents.matrix import matrix_search

    engine = Engine(load_spec(ROUNDS))
    search: Any = build_agent("matrix", engine).search
    state = engine.initial_state()
    mine, theirs = engine.legal_moves_for(state, 0)[:2], engine.legal_moves_for(state, 1)[:2]
    _v, _p, _q, done = matrix_search(search, state, 0, mine, theirs, 400, 1)
    assert (done, search.nodes) == (1, 4)
    _v, _p, _q, done = matrix_search(search, state, 0, mine, theirs, 400, 2)
    assert (done, search.nodes) == (2, 333)


# ------------------------------------------------------------------ batch 7 (Q1153)


def test_draw_never_plays_a_zero_weight_move_and_a_boundary_belongs_to_the_next() -> None:
    """A draw of exactly 0.0 skips a first move of weight zero; a draw landing on a running
    total (0.5 of 0.5 + 0.5) is past the first move, not on it."""
    assert draw([10, 20], [0.0, 1.0], SimpleNamespace(random=lambda: 0.0)) == 20  # type: ignore[arg-type]
    assert draw([10, 20], [0.5, 0.5], SimpleNamespace(random=lambda: 0.5)) == 20  # type: ignore[arg-type]
    assert draw([10, 20], [0.5, 0.5], SimpleNamespace(random=lambda: 0.4999)) == 10  # type: ignore[arg-type]


def test_a_bare_mcts_agent_has_two_thousand_simulations() -> None:
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    agent = MCTSAgent(None)  # type: ignore[arg-type]
    assert agent.budget == 2000
    assert agent.name == "mcts:nodes=2000"


def test_uct_keeps_the_choosers_own_child_above_a_weaker_opponent_child() -> None:
    """The chooser's own child at 0.8 beats the opponent's child read as 1 - 0.3 = 0.7."""
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    agent = MCTSAgent(None, nodes=1, exploration=0.0)  # type: ignore[arg-type]
    own = SimpleNamespace(visits=1, wins=0.8, state=SimpleNamespace(side_to_move=0))
    other = SimpleNamespace(visits=1, wins=0.3, state=SimpleNamespace(side_to_move=1))
    node = SimpleNamespace(visits=2, state=SimpleNamespace(side_to_move=0), children=[own, other])
    assert agent._best_child(node) is own  # type: ignore[arg-type]


def test_a_greedy_win_outranks_an_evaluation_just_under_a_billion() -> None:
    """A win is worth exactly 1e9: an evaluation of 999,999,999.5 loses to it."""
    from chess_variant_sandbox.agents.simple import GreedyAgent

    won = SimpleNamespace(is_draw=False, winner=0)
    agent = object.__new__(GreedyAgent)
    agent.engine = SimpleNamespace(
        spy=False, apply=lambda state, move: move,
        terminal=lambda successor: won if successor == 2 else None,
    )
    agent.evaluator = SimpleNamespace(evaluate=lambda engine, successor, player: 999_999_999.5)
    agent.stats = SearchStats()
    state = SimpleNamespace(side_to_move=0)
    for seed in range(4):
        assert agent.select_move(state, [1, 2], random.Random(seed)) == 2  # type: ignore[arg-type]


def test_the_greedy_round_value_spends_one_node_per_cell() -> None:
    """Budget ``None``: the one-round matrix of two moves each, four cells, four nodes."""
    from chess_variant_sandbox.agents.matrix import round_value_for_first

    engine = Engine(load_spec(ROUNDS))
    search: Any = build_agent("matrix", engine).search
    state = engine.initial_state()
    lists = (engine.legal_moves_for(state, 0)[:2], engine.legal_moves_for(state, 1)[:2])
    _value, done = round_value_for_first(search, state, lists, None)
    assert (done, search.nodes) == (1, 4)


# ------------------------------------------------------------------ batch 8 (Q1153)


def test_a_random_player_counts_one_move_per_round() -> None:
    engine = Engine(load_spec(ROUNDS))
    agent: Any = build_agent("random", engine)
    state = engine.initial_state()
    mine, theirs = engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
    agent.select_round_move(state, 0, mine, theirs, random.Random(0))
    assert (agent.stats.moves, agent.stats.nodes) == (1, len(mine))


def test_greedy_and_matrix_rounds_with_one_move_count_it_and_play_it() -> None:
    from chess_variant_sandbox.agents.matrix import greedy_round, matrix_round

    stats = SearchStats()
    assert greedy_round(None, stats, None, 0, [7], [1, 2], random.Random(0)) == (7, [1.0])  # type: ignore[arg-type]
    assert stats.moves == 1
    played = matrix_round(None, stats, None, 1, [9], [1, 2], random.Random(0), 10, 2)  # type: ignore[arg-type]
    assert played == (9, [1.0], 0)
    assert stats.moves == 2


def test_a_matrix_agent_built_directly_starts_at_depth_zero_and_weighs_as_the_builder() -> None:
    from chess_variant_sandbox.agents.matrix import MatrixAgent

    engine = Engine(load_spec(ROUNDS))
    agent = MatrixAgent(engine)
    built: Any = build_agent("matrix", engine)
    assert agent.depth == 0
    assert agent.strategy == []
    assert agent.search.evaluator == built.search.evaluator


def test_a_greedy_player_prefers_an_evaluation_just_over_minus_a_billion_to_a_loss() -> None:
    """A loss is worth exactly -1e9: an evaluation of -999,999,999.5 beats it."""
    from chess_variant_sandbox.agents.simple import GreedyAgent

    lost = SimpleNamespace(is_draw=False, winner=1)
    agent = object.__new__(GreedyAgent)
    agent.engine = SimpleNamespace(
        spy=False, apply=lambda state, move: move,
        terminal=lambda successor: lost if successor == 1 else None,
    )
    agent.evaluator = SimpleNamespace(evaluate=lambda engine, successor, player: -999_999_999.5)
    agent.stats = SearchStats()
    state = SimpleNamespace(side_to_move=0)
    for seed in range(4):
        assert agent.select_move(state, [1, 2], random.Random(seed)) == 2  # type: ignore[arg-type]


def test_a_valuation_may_price_exactly_the_most_piece_types_and_no_more() -> None:
    from chess_variant_sandbox.agents.valuations import ValuationError
    from chess_variant_sandbox.spec.schema import MAX_PIECE_TYPES

    full = (1.0,) * MAX_PIECE_TYPES
    priced = dataclasses.replace(_valuation(), piece_ids=(), vector=full, derived=full)
    assert len(priced.vector) == MAX_PIECE_TYPES
    over = full + (1.0,)
    with pytest.raises(ValuationError, match="at most"):
        dataclasses.replace(_valuation(), piece_ids=(), vector=over, derived=over)


def test_a_stored_valuation_without_a_score_or_comparisons_reads_zero() -> None:
    data = _valuation().as_dict()
    data["weights"] = [0.0] * len(VOCABULARY)
    data["accept_score"] = 0.75
    data["comparisons"] = 3
    del data["accept_score"], data["comparisons"]
    loaded = Valuation.from_dict(data)
    assert (loaded.accept_score, loaded.comparisons) == (0.0, 0)


# ------------------------------------------------------------------ batch 9


def test_uct_explores_by_the_log_of_one_more_than_the_parents_visits() -> None:
    """Parent visits 5, children 4 and 1, exploration 1: ``ln 6`` lifts the fresh child
    (0.25 + 1.3386 = 1.5886) over the proven one (0.9 + 0.6693 = 1.5693); ``ln 5`` would
    not (1.5186 against 1.5343)."""
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    agent = MCTSAgent(None, nodes=1, exploration=1.0)  # type: ignore[arg-type]
    proven = SimpleNamespace(visits=4, wins=3.6, state=SimpleNamespace(side_to_move=0))
    fresh = SimpleNamespace(visits=1, wins=0.25, state=SimpleNamespace(side_to_move=0))
    node = SimpleNamespace(
        visits=5, state=SimpleNamespace(side_to_move=0), children=[proven, fresh]
    )
    assert agent._best_child(node) is fresh  # type: ignore[arg-type]


def test_load_all_of_a_missing_directory_counts_none(tmp_path: Path) -> None:
    from chess_variant_sandbox.agents import valuations

    assert valuations.load_all(tmp_path / "absent") == 0


def test_a_stored_valuation_without_an_accept_game_count_reads_zero() -> None:
    data = dataclasses.replace(_valuation(), accept_games=40).as_dict()
    data["weights"] = [0.0] * len(VOCABULARY)
    assert Valuation.from_dict(data).accept_games == 40
    del data["accept_games"]
    assert Valuation.from_dict(data).accept_games == 0


# ------------------------------------------------------------------ batch 10 (Q1153)


def test_a_fresh_round_search_is_unbounded() -> None:
    """Budget 0 and never searched: a cell still evaluates rather than running out."""
    engine = Engine(load_spec(ROUNDS))
    search: Any = build_agent("matrix", engine).search
    state = engine.initial_state()
    mine, theirs = engine.legal_moves_for(state, 0)[:1], engine.legal_moves_for(state, 1)[:2]
    assert (search.bounded, search.budget) == (False, 0)
    values = search.cells(state, 0, mine, theirs, 1)
    assert (len(values), search.nodes) == (2, 2)


class _ClosingUci:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_a_fairy_player_closes_its_process_once_and_one_never_started_quietly() -> None:
    never_started = FairyStockfishAgent.__new__(FairyStockfishAgent)
    never_started.close()
    agent = FairyStockfishAgent.__new__(FairyStockfishAgent)
    uci = _ClosingUci()
    agent._uci = uci  # type: ignore[assignment]
    agent.close()
    assert uci.closed == 1


def test_a_valuation_stores_its_ratios_to_four_places() -> None:
    valuation = dataclasses.replace(_valuation(), vector=(2.0,), derived=(3.0,))
    assert valuation.as_dict()["ratios"] == [0.6667]


def test_a_stored_valuation_without_its_errors_reads_zero() -> None:
    data = dataclasses.replace(_valuation(), accept_error=0.25, ladder_error=0.125).as_dict()
    data["weights"] = [0.0] * len(VOCABULARY)
    loaded = Valuation.from_dict(data)
    assert (loaded.accept_error, loaded.ladder_error) == (0.25, 0.125)
    del data["accept_error"], data["ladder_error"]
    loaded = Valuation.from_dict(data)
    assert (loaded.accept_error, loaded.ladder_error) == (0.0, 0.0)


def test_a_valuation_saves_into_a_directory_that_does_not_exist_yet(tmp_path: Path) -> None:
    path = _valuation().save(tmp_path / "a" / "b" / "v.json")
    assert path == tmp_path / "a" / "b" / "v.json"
    assert json.loads(path.read_text(encoding="utf-8"))["values"] == [2.0]


# ------------------------------------------------------------------ batch 11


def test_uct_explores_by_the_log_of_one_more_and_not_two_more_than_the_parents_visits() -> None:
    """Parent visits 5, children 4 and 1, exploration 1: ``ln 6`` keeps the proven child
    (0.9 + 0.6693 = 1.5693) over the fresh one (0.22 + 1.3386 = 1.5586); ``ln 7`` would
    not (1.5975 against 1.6150)."""
    from chess_variant_sandbox.agents.mcts import MCTSAgent

    agent = MCTSAgent(None, nodes=1, exploration=1.0)  # type: ignore[arg-type]
    proven = SimpleNamespace(visits=4, wins=3.6, state=SimpleNamespace(side_to_move=0))
    fresh = SimpleNamespace(visits=1, wins=0.22, state=SimpleNamespace(side_to_move=0))
    node = SimpleNamespace(
        visits=5, state=SimpleNamespace(side_to_move=0), children=[proven, fresh]
    )
    assert agent._best_child(node) is proven  # type: ignore[arg-type]


def test_a_greedy_player_takes_a_loss_over_an_evaluation_just_under_minus_a_billion() -> None:
    """A loss is worth exactly -1e9: an evaluation of -1,000,000,000.5 is worse than it."""
    from chess_variant_sandbox.agents.simple import GreedyAgent

    lost = SimpleNamespace(is_draw=False, winner=1)
    agent = object.__new__(GreedyAgent)
    agent.engine = SimpleNamespace(
        spy=False, apply=lambda state, move: move,
        terminal=lambda successor: lost if successor == 1 else None,
    )
    agent.evaluator = SimpleNamespace(evaluate=lambda engine, successor, player: -1_000_000_000.5)
    agent.stats = SearchStats()
    state = SimpleNamespace(side_to_move=0)
    for seed in range(4):
        assert agent.select_move(state, [1, 2], random.Random(seed)) == 1  # type: ignore[arg-type]


def test_a_stored_valuation_without_a_ladder_score_reads_one_half() -> None:
    data = dataclasses.replace(_valuation(), ladder_score=0.75).as_dict()
    data["weights"] = [0.0] * len(VOCABULARY)
    assert Valuation.from_dict(data).ladder_score == 0.75
    del data["ladder_score"]
    assert Valuation.from_dict(data).ladder_score == 0.5


def test_registering_a_valuation_keeps_the_first_unless_told_to_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_variant_sandbox.agents import valuations

    monkeypatch.setattr(valuations, "_VALUATIONS", {})
    first = _valuation()
    second = dataclasses.replace(first, accept_games=7)
    valuations.register_valuation(first)
    valuations.register_valuation(second)
    assert valuations.registered_valuations()[("p", "h")] is first
    valuations.register_valuation(second, replace=True)
    assert valuations.registered_valuations()[("p", "h")] is second


def test_a_search_procedure_is_frozen_and_hashable() -> None:
    from chess_variant_sandbox.agents.valuations import SearchProcedure

    procedure = SearchProcedure()
    assert hash(procedure) == hash(SearchProcedure())
    with pytest.raises(dataclasses.FrozenInstanceError):
        procedure.nodes = 1  # type: ignore[misc]


# ------------------------------------------------------------------ batch 13 (Q1153)


def test_a_stored_valuation_without_an_exhausted_flag_reads_not_exhausted() -> None:
    data = dataclasses.replace(_valuation(), exhausted=True).as_dict()
    data["weights"] = [0.0] * len(VOCABULARY)
    assert Valuation.from_dict(data).exhausted is True
    del data["exhausted"]
    assert Valuation.from_dict(data).exhausted is False


def test_loading_every_valuation_skips_an_unreadable_file_and_keeps_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unreadable file sorts first, so a loop that stopped at it would load nothing."""
    from chess_variant_sandbox.agents import valuations

    monkeypatch.setattr(valuations, "_VALUATIONS", {})
    good = dataclasses.replace(_valuation(), weights=(0.0,) * len(VOCABULARY))
    good.save(tmp_path / "b.json")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert valuations.load_all(tmp_path) == 1
    assert valuations.registered_valuations()[("p", "h")] == good


# ------------------------------------------------------------------ objective.py


@pytest.mark.parametrize(
    ("weights", "values"),
    [((1.0,), ()), ((), (1.0,))],
    ids=["weights-alone", "values-alone"],
)
def test_a_net_beside_either_half_of_a_heuristic_is_refused(
    weights: tuple[float, ...], values: tuple[float, ...]
) -> None:
    from chess_variant_sandbox.agents.objective import objective_for

    with pytest.raises(ValueError, match="a net was given beside a weight vector or piece values"):
        objective_for(_dull(), 0.05, weights=weights, values=values, net=b"not a net")
