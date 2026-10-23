"""F10b group 4 (Q752b): the players' corners no test reached, held to the declared oracles.

`docs/plan-tests.md` F10b. The core's players against their Python oracles (`agents/*.py`,
CLAUDE.md rule 3) where no committed game took them: MCTS's tree on a short endgame, in
one-move turns and in two-move ones (a child of the same side); the move ordering's
capture gain of a sweep; the minimax teacher's value on a spy game; the round teacher's value
where a royal is caught by a faster piece or rests (`eval.rs`'s early answers: a second royal,
a royal in hand, a royal whose loss ends nothing, two attackers); and the thread pools of the
round teacher and of random playouts. Every comparison is the whole record or the value to
the bit.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from test_backends import _assert_same_played  # noqa: E402
from test_matrix_players import core_round, oracle_round  # noqa: E402
from test_matrix_players import walk as round_walk  # noqa: E402
from test_nets import _line as net_line  # noqa: E402
from test_player_teacher import _oracle_value as minimax_value  # noqa: E402
from test_player_teacher import _values as minimax_values  # noqa: E402
from test_round_nets import _oracle_score as score_value  # noqa: E402
from test_round_nets import _oracle_value as round_value  # noqa: E402

from chess_variant_sandbox.agents.nets import Layout, NetEvaluator, random_net  # noqa: E402
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.engine.engine import _pack  # noqa: E402
from chess_variant_sandbox.engine.moves import decode_move  # noqa: E402
from chess_variant_sandbox.spec import validate_spec  # noqa: E402
from chess_variant_sandbox.tournament.play import eval_values, native_spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _endgame(**rules: Any) -> Engine:
    """Gardner with a king, a queen and a rook against a lone king and a short ply limit:
    games that end, so a tree meets decided positions and a lone king has forced moves."""
    spec = _load(ROOT / "rulesets" / "gardner.json")
    spec["name"] = "Gardner endgame (F10b test spec)"
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "squares": ["e1"], "player": 0},
        {"piece": "Q", "squares": ["d1"], "player": 0},
        {"piece": "R", "squares": ["a1"], "player": 0},
        {"piece": "K", "squares": ["c5"], "player": 1},
    ]}
    del spec["rules"]["promotion"]
    spec["rules"].update({"max_plies": 40, **rules})
    assert validate_spec(spec) == []
    return Engine(spec)


MCTS = "mcts:nodes=24,rollout_depth=8"


@pytest.mark.parametrize("agents", [(MCTS, "random"), ("random", MCTS)],
                         ids=["mcts-attacks", "mcts-defends"])
def test_mcts_on_an_endgame_is_the_oracles_move_for_move(agents: tuple[str, str]) -> None:
    """The tree reaches decided positions (expanded and rolled out from), draws by
    stalemate, and the lone king's single legal move."""
    _assert_same_played(_endgame(), agents, seed=3)


def test_mcts_in_two_move_turns_is_the_oracles_move_for_move() -> None:
    """Two moves a turn: a child is often the same side's, whose win rate is read the right
    way round."""
    _assert_same_played(_endgame(turn={"moves": 2}), (MCTS, "random"), seed=4)


def test_a_sweeps_capture_gain_orders_the_search_as_the_oracle_does() -> None:
    """A rook that may also sweep two squares diagonally, taking what it passes: the move
    ordering sums the swept pieces onto the target's, and the core's game is the Python
    loop's, nodes and all."""
    spec = _load(ROOT / "rulesets" / "gardner.json")
    spec["name"] = "Gardner with sweeping rooks (F10b test spec)"
    rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
    rook["moves"].append({"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4",
                          "range": 2})
    assert validate_spec(spec) == []
    engine = Engine(spec)
    for agents in (("minimax:nodes=150", "greedy"), ("greedy", "minimax:nodes=150")):
        _assert_same_played(engine, agents, seed=2)


def test_the_minimax_teacher_on_a_spy_game_is_the_oracles_to_the_bit() -> None:
    """`search_values` on a spy game's turns (`tests/_spy_small.json`): the value, move and
    depth are the oracle's deepening's."""
    engine = Engine(_load(HERE / "_spy_small.json"))
    states = [engine.initial_state()]
    agent = "minimax:nodes=200"
    native = minimax_values(engine, states, agent, [7])
    assert native == [minimax_value(engine, agent, states[0], 7)]
    assert native[0] is not None


# -- the round teacher where a royal is caught or rests ------------------------------------


def _round_game(base: str, *, speeds: dict[str, int] | None = None, kings: bool = False,
                rests: dict[str, int] | None = None, **rules: Any) -> Engine:
    spec = copy.deepcopy(_load(HERE / f"{base}.json"))
    spec["name"] = "Gardner in rounds (F10b test spec)"
    for piece in spec["pieces"]:
        if speeds and piece["id"] in speeds:
            piece["speed"] = speeds[piece["id"]]
        if rests and piece["id"] in rests:
            piece["rest"] = rests[piece["id"]]
    if kings:
        for placement in spec["setup"]["placements"]:
            if placement["piece"] == "K":
                placement["squares"] = [*placement["squares"], "c1"]
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "B"
        ]
    spec["rules"].update(rules)
    assert validate_spec(spec) == []
    return Engine(spec)


# White's king on c1 (speed 1), Black's queens (speed 2) bearing on it from c5 and a3, and a
# Black pawn (speed 2) on b2 that takes it promoting (four words from one square).
CAUGHT = {"c1": ("K", 0), "e1": ("R", 0), "c5": ("Q", 1), "a3": ("Q", 1), "b2": ("P", 1),
          "e5": ("K", 1)}
SPEEDS = {"Q": 2, "P": 2}


def _positions(engine: Engine, placement: dict[str, tuple[str, int]],
               rest: tuple[tuple[str, int], ...] = ()) -> list[Any]:
    """The position, with each square named in ``rest`` resting that many rounds for White,
    and the same with one Black queen gone (one attacker) and with both gone."""
    out = []
    for gone in ((), ("a3",), ("a3", "c5")):
        state = engine.make_state({k: v for k, v in placement.items() if k not in gone})
        rested = tuple((engine._square_index(name), left) for name, left in rest)
        state = state.with_(rest=(rested, ()))
        out.append(state.with_(key=engine._compute_hash(state)))
    return out


def _teacher_agrees(engine: Engine, states: list[Any]) -> None:
    packed = [_pack(state) for state in states]
    values = eval_values(engine.spec)
    native = native_spec("greedy", engine.spec.hash)
    answers = engine.core.round_values(packed, native, values)
    assert engine.core.round_values(packed, native, values, threads=2) == answers
    for state, answer in zip(states, answers, strict=True):
        assert answer == round_value(engine, state, None)


@pytest.mark.parametrize("variant", ["plain", "no royal ending", "two kings", "royal in hand"])
def test_the_round_teacher_prices_a_caught_or_resting_royal_as_the_oracle(variant: str) -> None:
    """A royal caught by faster pieces, and resting, in Gardner in rounds with speeds and
    rests (`tests/_simultaneous_speed_resting.json`, queens and pawns made faster than the
    king): the round teacher's value is the oracle's to the bit, with no attacker, one and
    two, under each early answer the term has -- a royal whose loss ends nothing (`win:
    "none"` beside a bare royal), a second royal, a royal in hand (drops). (A side with no
    royal on the board and one in hand is never asked: its royal is gone, which decides the
    game first.)"""
    rules: dict[str, Any] = {}
    kings = False
    placement = dict(CAUGHT)
    if variant == "no royal ending":
        rules = {"win": "none", "bare_royal": "loss"}
    if variant == "two kings":
        kings = True
        placement["a1"] = ("K", 0)
    if variant == "royal in hand":
        rules = {"drops": {"from_captures": True}}
    # The king rests two rounds, so it still rests in the positions a round leads to.
    engine = _round_game("_simultaneous_speed_resting", speeds=SPEEDS, kings=kings,
                         rests={"K": 2}, **rules)
    states = _positions(engine, placement, rest=(("c1", 2),))
    if variant == "royal in hand":
        king = engine.spec.piece_ids.index("K")
        hand = tuple(int(t == king) for t in range(engine.spec.n_types))
        states = [s.with_(hands=(hand, (0,) * len(hand))) for s in states]
        states = [s.with_(key=engine._compute_hash(s)) for s in states]
    _teacher_agrees(engine, states)


def test_the_round_teacher_has_no_value_where_the_round_is_over() -> None:
    """A position where the game is decided before anyone chooses has no value, on one
    thread or two."""
    engine = _round_game("_simultaneous_speed_resting", speeds=SPEEDS)
    decided = engine.make_state({"c3": ("Q", 0), "e5": ("K", 1)})
    assert engine.terminal(decided) is not None
    native = native_spec("greedy", engine.spec.hash)
    for threads in (1, 2):
        assert engine.core.round_values(
            [_pack(decided)], native, eval_values(engine.spec), threads=threads,
        ) == [None]


def test_random_playouts_are_the_same_on_two_threads() -> None:
    """The cheap gate's random playouts: one stream per game fixed before any thread starts,
    so two threads give one thread's list."""
    engine = Engine(_load(ROOT / "rulesets" / "gardner.json"))
    one = engine.core.random_playouts(6, seed=3)
    assert engine.core.random_playouts(6, seed=3, threads=2) == one
    assert len(one) == 6


# -- the evaluator's corners, through the minimax teacher -----------------------------------


def _with(name: str, **rules: Any) -> Engine:
    spec = _load(ROOT / "rulesets" / f"{name}.json")
    spec["name"] = f"{name} (F10b test spec)"
    spec["rules"].update(rules)
    assert validate_spec(spec) == []
    return Engine(spec)


EVALUATOR_ARMS = {
    # A race to c5 by knights beside drops: no seal is defined (a capture can put a new
    # captor in the hand), and only knights are eligible to race.
    "a knights' race beside drops": ("gardner", {
        "reach": {"piece": "N", "squares": ["c5"], "mirror": True, "result": "win"},
        "drops": {"from_captures": True},
    }),
    # One queen a side and a count that ends at one: no room, so the term is zero.
    "a count with no room": ("gardner", {
        "extinction": {"pieces": ["Q"], "min_count": 1, "result": "loss"},
    }),
    # Losing every piece wins (material counts against), with drops: the hand counts
    # against too.
    "losing wins, with drops": ("antichess", {"drops": {"from_captures": True}}),
}


@pytest.mark.parametrize("arm", sorted(EVALUATOR_ARMS))
def test_the_evaluators_corners_value_as_the_oracle(arm: str) -> None:
    """`search_values` against the oracle's deepening (`test_player_teacher._oracle_value`)
    on three positions of a seeded line, value, move and depth to the bit."""
    name, rules = EVALUATOR_ARMS[arm]
    engine = _with(name, **rules)
    rng = random.Random(1)
    state, states = engine.initial_state(), []
    while len(states) < 3:
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            state = engine.initial_state()
            continue
        states.append(state)
        for _ in range(3):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is None:
                state = engine.apply(state, rng.choice(moves))
    agent = "minimax:nodes=120"
    seeds = [11, 12, 13]
    native = minimax_values(engine, states, agent, seeds)
    assert native == [minimax_value(engine, agent, s, seed)
                      for s, seed in zip(states, seeds, strict=True)]
    assert any(answer is not None for answer in native)


# -- the round players' budget, the score teacher's drawn cell, a net with no clock ----------


def test_a_matrix_player_whose_first_pass_spends_its_budget_is_the_oracles() -> None:
    """`matrix:nodes=20` on Gardner in rounds: the first pass (unbounded, one node a cell)
    alone spends more than the budget, so no deeper pass starts, and the strategy, move,
    nodes and depth are the oracle's."""
    engine = Engine(_load(HERE / "_simultaneous_gardner.json"))
    for index, state in enumerate(round_walk(engine, 3, seed=2)):
        for seat in (0, 1):
            core = core_round(engine, "matrix:nodes=20", state, seat, seed=index)
            assert core == oracle_round(engine, "matrix:nodes=20", state, seat, seed=index)
            assert core[3] == 1


def test_the_score_teacher_scores_a_round_that_takes_both_royals_as_a_draw() -> None:
    """Counted by hand: each side's rook bears on the other's king, so the round where both
    take is a draw, a cell worth one half on the score payoff; the label is the oracle's."""
    engine = Engine(_load(HERE / "_simultaneous_gardner.json"))
    state = engine.make_state({"e1": ("K", 0), "a1": ("R", 0), "a5": ("K", 1), "e5": ("R", 1)})
    both = engine.apply_round(state, engine.find_move(state, "Ra1xa5"),
                              _black(engine, state, "e5", "e1"))
    verdict = engine.terminal(both)
    assert verdict is not None and verdict.winner is None
    for agent in ("greedy", "matrix:nodes=1"):
        native = native_spec(agent, engine.spec.hash)
        (answer,) = engine.core.round_values(
            [_pack(state)], native, eval_values(engine.spec), score_scale=2.0,
        )
        assert answer == score_value(engine, state, 2.0)


def _black(engine: Engine, state: Any, origin: str, target: str) -> int:
    squares = (engine._square_index(origin), engine._square_index(target))
    (move,) = [m for m in engine.legal_moves_for(state, 1) if decode_move(m)[:2] == squares]
    return move


def test_a_net_on_a_game_without_a_no_progress_clock_is_the_oracles() -> None:
    """Gardner with no `no_progress_draw`: the net's clock inputs for it are off, and along a
    random game the core's inputs and raw value are the oracle layout's to the bit."""
    spec = _load(ROOT / "rulesets" / "gardner.json")
    spec["name"] = "Gardner without a no-progress draw (F10b test spec)"
    del spec["rules"]["no_progress_draw"]
    engine = Engine(spec)
    start, moves, states = net_line(engine, seed=3)
    blob = random_net(Layout.of(engine.spec).n_inputs, seed=1)
    oracle = NetEvaluator(engine, blob, ())
    answers = engine.core.net_line(_pack(start), blob, moves)
    for (raw, inputs, fresh), state in zip(answers, states, strict=True):
        assert inputs == oracle.layout.features(state)
        assert raw == fresh == oracle.raw(state)


def test_a_spy_games_perft_counts_the_position_at_depth_zero_and_stops_when_decided() -> None:
    """`tests/_spy_small.json`: perft at depth zero is one, stopped or not; a decided
    position (Black's king gone) counts nothing once stopped, though it has moves."""
    engine = Engine(_load(HERE / "_spy_small.json"))
    start = engine.initial_state()
    assert engine.perft(start, 0) == 1 == engine.perft(start, 0, stop_at_terminal=True)
    rng = random.Random(4)
    state = start
    while engine.terminal(state) is None:
        if engine.is_round(state):
            first = rng.choice(engine.legal_moves_for(state, 0))
            state = engine.apply_round(state, first, rng.choice(engine.legal_moves_for(state, 1)))
        else:
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
    assert engine.perft(state, 1, stop_at_terminal=True) == 0
