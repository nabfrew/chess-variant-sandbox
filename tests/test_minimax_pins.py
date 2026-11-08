"""Pins on the minimax oracle's helpers and table plumbing (Q1152, the agents/ mutation group).

Each test holds one helper's whole output, or one search's exact value and node count, on a
small input built so that every branch of the helper is landed on: the mop-up weight's six
refusals one at a time, a sight path's every kind of square, a table entry of each bound on
each side of its window, a duck's sight candidates along a seeded game, a spy round whose
answers hand the move over. Specs are inline or read from `rulesets/` and `tests/` as
fixtures; the subjects of the assertions are the helpers, never a piece.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox.agents import build_agent, native_spec
from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, Evaluator
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    DUCK_FRONT,
    DUCK_HISTORY,
    DUCK_MOBILITY,
    DUCK_QUIESCE,
    DUCK_REFUTE,
    EXACT,
    KEEP_TABLE,
    LOWER,
    MAX_PLY,
    MOPUP,
    NO_MOVE,
    TT_MOVE,
    UPPER,
    MinimaxAgent,
    _pair,
    _path_open,
    mopup_weight,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import SIGHT_PATH_AURA, WILD
from chess_variant_sandbox.engine.moves import DUCK_PLACE, MOVE_DUCK, decode_move, encode_move
from chess_variant_sandbox.tournament.play import eval_values, play_game, seat_rng

ROOT = Path(__file__).resolve().parents[1]
INF = float("inf")


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _ending(piece: str, own: str, king: str, theirs: str) -> Engine:
    """A royal and one ``piece`` for player 0 against a lone royal, on `rulesets/chess.json`'s
    rules with its castling, en passant and promotion taken out (a fixture)."""
    spec = _load("rulesets/chess.json")
    spec["name"] = f"ending {piece} {own} {king} {theirs}"
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "squares": [king], "player": 0},
        {"piece": piece, "squares": [own], "player": 0},
        {"piece": "K", "squares": [theirs], "player": 1},
    ]}
    for key in ("castling", "en_passant", "promotion"):
        spec["rules"].pop(key, None)
    spec["pieces"] = [p for p in spec["pieces"] if p["id"] in ("K", piece)]
    return Engine(spec)


def _gardner_duck() -> Engine:
    spec = _load("rulesets/gardner.json")
    spec["name"] = "gardner with a duck (pins)"
    spec["pieces"].append({"id": "D", "name": "duck", "value": 0, "moves": []})
    spec["rules"]["win"] = "royal_capture"
    spec["rules"]["duck"] = {"piece": "D"}
    spec["rules"]["max_plies"] = 60
    spec["rules"].pop("checkmate", None)
    return Engine(spec)


def _after(engine: Engine, plies: int, seed: int = 3) -> GameState:
    """The position ``plies`` into a seeded random game."""
    record = play_game(engine, ("random", "random"), seed)
    state = engine.initial_state()
    for move in record.moves[:plies]:
        state = engine.apply(state, move)
    return state


# ------------------------------------------------------------------ mopup_weight


class _Spec:
    """A compiled spec with some attributes replaced."""

    def __init__(self, spec: Any, **replaced: Any) -> None:
        self._spec = spec
        self.__dict__.update(replaced)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._spec, name)


def _signed(evaluator: Evaluator, sign: float) -> Evaluator:
    """``evaluator`` with its material sign replaced."""
    cls = type("Signed", (Evaluator,), {"material_sign": property(lambda self: sign)})
    return cls(evaluator.spec, evaluator.values)


def test_the_mopup_weight_refuses_each_rule_alone() -> None:
    engine = Engine(_load("rulesets/chess.json"))
    spec, evaluator = engine.spec, Evaluator.for_engine(engine)
    weight = 6.171428571428572
    assert mopup_weight(spec, evaluator) == weight
    for name in ("duck", "simultaneous", "spy", "turn"):
        assert mopup_weight(_Spec(spec, **{name: object()}), evaluator) == 0.0, name
    assert mopup_weight(_Spec(spec, royal_types=()), evaluator) == 0.0
    assert mopup_weight(spec, _signed(evaluator, 1.0)) == weight
    assert mopup_weight(spec, _signed(evaluator, 0.0)) == 0.0
    assert mopup_weight(spec, _signed(evaluator, -1.0)) == 0.0
    assert mopup_weight(spec, _signed(evaluator, 0.5)) == weight


# ------------------------------------------------------------------ _path_open


def test_a_sight_paths_every_kind_of_square() -> None:
    """Square 2 empty; 1 a plain piece (owner 1, a portal end's type); 3 a wild portal end;
    5 a wild piece of another type. Without the aura bit only empty or lifted is open; with
    it, anything but a wild portal end, and anything at all without portals."""
    portals = SimpleNamespace(is_end=lambda t: t == 1)

    def spec(with_portals: bool) -> Any:
        return SimpleNamespace(
            portals=portals if with_portals else None,
            owner_of=lambda v: v // 10,
            type_of=lambda v: v % 10,
        )

    assert WILD == 2
    board = bytes([0, 11, 0, 21, 0, 22, 0, 0])
    aura = SIGHT_PATH_AURA
    cases = [
        (2, -1, True), (1, -1, True), (1, 1, True), (3, -1, True),
        (aura | 1, -1, True), (aura | 3, -1, True), (aura | 5, -1, True), (aura | 3, -1, False),
        (aura | 2, -1, True),
    ]
    got = [_path_open(spec(p), board, entry, lifted) for entry, lifted, p in cases]
    assert got == [True, False, True, False, True, False, True, True, True]


# ------------------------------------------------------------------ the table probe


def _probe(agent: MinimaxAgent, state: GameState, entry: tuple[float, int], window: tuple[float, float]) -> float:
    value, bound = entry
    agent._table[agent._table_key(state)] = (9, value, bound, agent._generation, NO_MOVE)
    return agent._negamax(state, 1, 1, *window)


def test_a_table_entry_answers_only_on_its_side_of_the_window() -> None:
    """An exact entry answers any window; a lower bound only at or above beta, an upper bound
    only at or below alpha; otherwise the node is searched (depth 1 from gardner's opening
    is -0.35 at alpha -inf, and -0.35 failing low at alpha 0)."""
    engine = Engine(_load("rulesets/gardner.json"))
    state = engine.initial_state()

    def probe(entry: tuple[float, int], window: tuple[float, float]) -> float:
        return _probe(MinimaxAgent(engine, nodes=10**7), state, entry, window)

    searched = -0.35000000000000003
    assert probe((0.125, EXACT), (-INF, INF)) == 0.125
    assert probe((0.25, LOWER), (-INF, 0.25)) == 0.25
    assert probe((0.25, LOWER), (-INF, 0.5)) == searched
    assert probe((0.25, UPPER), (-INF, 0.0)) == searched
    assert probe((-0.5, UPPER), (-0.5, INF)) == -0.5
    assert probe((-0.5, LOWER), (0.0, INF)) == searched
    assert probe((-0.5, UPPER), (-0.75, INF)) == searched


def _spy_round() -> tuple[Engine, GameState, int]:
    """`tests/_spy_promoting.json`, an announced round: player 1 announces, and its capture
    promoting into a spy takes player 0's spy, so seven of player 0's ten answers hand the
    move over (SR7h)."""
    engine = Engine(_load("tests/_spy_promoting.json"))
    state = engine.make_state(
        {"a3": ("K", 0), "d1": ("S", 0), "b1": ("N", 0),
         "e5": ("K", 1), "e2": ("P", 1), "c4": ("R", 1)},
        side_to_move=0,
    )
    phase = engine.phase(state)
    assert (phase.name, phase.announcer) == ("announced", 1)
    announced = [
        m for m in engine.legal_moves_for(state, 1) if engine.describe_move(state, m) == "Pe2xd1=S"
    ]
    assert len(announced) == 1
    return engine, state, announced[0]


def test_a_rounds_table_entry_answers_only_on_its_side_of_the_window() -> None:
    engine, state, _ = _spy_round()
    phase = engine.phase(state)

    def probe(entry: tuple[float, int], window: tuple[float, float]) -> float:
        agent = MinimaxAgent(engine, nodes=10**7)
        value, bound = entry
        agent._table[agent._table_key(state)] = (9, value, bound, agent._generation, NO_MOVE)
        return agent._round_node(state, phase, 1, 1, *window)

    open_window, failing_low = -7.1725, -1.3325000000000005
    assert probe((0.125, EXACT), (-INF, INF)) == 0.125
    assert probe((0.25, LOWER), (-INF, 0.25)) == 0.25
    assert probe((0.25, LOWER), (-INF, 0.5)) == open_window
    assert probe((0.25, UPPER), (-INF, 0.0)) == open_window
    assert probe((-0.5, UPPER), (-0.5, INF)) == -0.5
    assert probe((-0.5, LOWER), (0.0, INF)) == failing_low
    assert probe((-0.5, UPPER), (-0.75, INF)) == failing_low


# ------------------------------------------------------------------ a spy round's answer


@pytest.mark.parametrize(
    "agent",
    [
        "minimax:nodes=5", "minimax:nodes=40", "minimax:nodes=100", "minimax:nodes=400",
        "minimax:nodes=100000,max_depth=2",
    ],
)
def test_the_answer_that_hands_the_move_over_is_the_cores(agent: str) -> None:
    engine, state, announced = _spy_round()
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, _, nodes, passes = engine.select_round(
        state, 0, native, eval_values(engine.spec), 7, announced
    )
    player = build_agent(agent, engine)
    assert isinstance(player, MinimaxAgent)
    assert player.select_answer(state, 0, lists, announced, seat_rng(7, 0)) == move
    assert (player.stats.nodes, player.stats.depth_sum) == (nodes, passes)


def test_an_answer_roots_value_and_cost() -> None:
    engine, state, announced = _spy_round()
    moves = engine.legal_moves_for(state, 0)
    got = []
    for depth in (1, 2, 3):
        agent = MinimaxAgent(engine, nodes=10**7)
        agent._begin_search()
        best = agent._answer_root(state, 1, announced, list(moves), depth)
        got.append((engine.describe_move(state, best), agent._nodes))
    assert got == [("Sd1-e1", 10), ("Sd1-e1", 183), ("Sd1-e1", 3294)]


def test_an_announcement_that_finishes_no_pass_reached_none() -> None:
    engine, state, _ = _spy_round()
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    for agent in ("minimax:nodes=5", "minimax:nodes=100"):
        native = native_spec(agent, engine.spec.hash)
        assert native is not None
        move, _, nodes, passes = engine.select_round(
            state, 1, native, eval_values(engine.spec), 7, None
        )
        player = build_agent(agent, engine)
        assert isinstance(player, MinimaxAgent)
        assert player.select_announcement(state, 1, lists, seat_rng(7, 1)) == move
        assert (player.stats.nodes, player.stats.depth_sum) == (nodes, passes)
        if agent == "minimax:nodes=5":
            assert passes == 0  # the budget ends inside the first pass


# ------------------------------------------------------------------ select_move's tally


def test_a_search_counts_one_move() -> None:
    engine = Engine(_load("rulesets/gardner.json"))
    agent = MinimaxAgent(engine, nodes=50)
    state = engine.initial_state()
    agent.select_move(state, engine.legal_moves(state), seat_rng(1, 0))
    assert (agent.stats.moves, agent.stats.searches, agent.stats.nodes) == (1, 1, 50)


def test_the_native_form() -> None:
    from chess_variant_sandbox.agents.minimax import _native_minimax

    assert _native_minimax({"nodes": 400}, "") == (
        "minimax", 400, 0.05, 64, 0, 0.0, GOAL_PIECES, (), (), IMMUNE_PIECES, b"", DEFAULT_REPAIRS,
    )


# ------------------------------------------------------------------ ordering and keys


def test_a_duck_moves_history_is_filed_where_the_bit_says() -> None:
    """Without ``duck_history`` a duck move is ordered by the shared from-to history, with it
    by the side-and-target one; each table alone moves the last duck to the front."""
    engine = _gardner_duck()
    state = _after(engine, 1)
    moves = engine.legal_moves(state)
    assert all(decode_move(m)[2] == MOVE_DUCK for m in moves) and len(moves) > 2
    last = moves[-1]
    origin, target, _, _ = decode_move(last)
    squares = engine.spec.n_squares
    plain = MinimaxAgent(engine, nodes=10)
    plain._history[origin * squares + target] = 5
    ordered = list(moves)
    plain._order(state, ordered, 1)
    assert ordered[0] == last
    filed = MinimaxAgent(engine, nodes=10, repairs=DEFAULT_REPAIRS | DUCK_HISTORY)
    filed._duck_history[state.side_to_move * squares + target] = 5
    ordered = list(moves)
    filed._order(state, ordered, 1)
    assert ordered[0] == last
    filed._duck_history.clear()
    filed._history[origin * squares + target] = 5
    ordered = list(moves)
    filed._order(state, ordered, 1)
    assert ordered == moves
    # Player 1's duck half reads its own row, side times squares on.
    theirs = _after(engine, 3)
    assert theirs.side_to_move == 1
    duck_moves = engine.legal_moves(theirs)
    assert decode_move(duck_moves[0])[2] == MOVE_DUCK
    other = MinimaxAgent(engine, nodes=10, repairs=DEFAULT_REPAIRS | DUCK_HISTORY)
    other._duck_history[squares + decode_move(duck_moves[-1])[1]] = 5
    ordered = list(duck_moves)
    other._order(theirs, ordered, 1)
    assert ordered[0] == duck_moves[-1] != duck_moves[0]


def test_the_table_key_mixes_the_colouring() -> None:
    """On a territory game one square coloured by player 0 moves the key by the core's
    multiplier (`agents.rs::table_key`)."""
    engine = Engine(_load("rulesets/rps-v2.json"))
    state = _after(engine, 1, seed=1)
    assert (state.colour.count(1), state.colour.count(2)) == (1, 0)
    agent = MinimaxAgent(engine, nodes=10)
    assert agent._table_key(state) ^ state.key == 0x9E3779B97F4A7C15
    # One more ply: a square coloured by player 1 too, by the core's second multiplier.
    both = _after(engine, 2, seed=1)
    assert (both.colour.count(1), both.colour.count(2)) == (1, 1)
    assert agent._table_key(both) ^ both.key == 0x9E3779B97F4A7C15 ^ 0xC2B2AE3D27D4EB4F
    # And one more: two of player 0's to one of player 1's, each count by its own multiplier.
    more = _after(engine, 3, seed=1)
    assert (more.colour.count(1), more.colour.count(2)) == (2, 1)
    mask = (1 << 64) - 1
    assert agent._table_key(more) ^ more.key == (
        (2 * 0x9E3779B97F4A7C15) & mask
    ) ^ 0xC2B2AE3D27D4EB4F


# ------------------------------------------------------------------ the mop-up term


def test_the_mopup_term_is_the_weight_times_the_confinement_signed_by_the_mover() -> None:
    """A lone royal on a1 boxed into fourteen squares and three moves: confined
    ``0.5 * ((1 - 14/64) + (1 - 3/8))``, plus for the armed side to move, minus for the bare."""
    engine = _ending("R", "h3", "h4", "a1")
    agent = MinimaxAgent(engine, nodes=100, repairs=repairs_of({"mopup": 1}))
    state = engine.initial_state()
    assert agent._mopup == 1.0
    armed = agent._mopup_term(state, len(engine.legal_moves(state)))
    bare_to_move = replace(state, side_to_move=1, key=state.key ^ engine.spec.zobrist_side)
    bare = agent._mopup_term(bare_to_move, len(engine.legal_moves(bare_to_move)))
    assert (armed, bare) == (0.703125, -0.703125)


def test_the_walk_steps_past_a_royal_capture() -> None:
    """The lone royal on b3 beside the standing royal on b2: its first step takes it, which is
    no step, and the walk goes on to the others."""
    engine = _ending("Q", "h8", "b2", "b3")
    agent = MinimaxAgent(engine, nodes=100, repairs=repairs_of({"mopup": 1}))
    assert agent._bare_reach(engine.initial_state(), 1, 64) == 36


# ------------------------------------------------------------------ plies past the tables


def test_a_cutoff_past_the_last_ply_is_filed_in_the_last_slot() -> None:
    """A node at ply 96 (``MAX_PLY``), reachable under ``max_depth`` above it, fails high as
    one at ply 1 does."""
    assert MAX_PLY == 96
    engine = Engine(_load("rulesets/gardner.json"))
    state = engine.initial_state()
    values = []
    for ply in (1, MAX_PLY, MAX_PLY + 3):
        agent = MinimaxAgent(engine, nodes=10**7)
        values.append(agent._negamax(state, 1, ply, -INF, -1e9))
        if ply > 1:
            # The cutoff's refuter and its (quiet) killer land in the last slot, ply 95.
            assert agent._refuters[MAX_PLY - 1] != -1 and agent._refuters[MAX_PLY - 2] == -1
            assert agent._killers[MAX_PLY - 1][0] != -1
            assert agent._killers[MAX_PLY - 2] == [-1, -1]
    assert values[0] == values[1] == values[2]


def test_a_duck_half_past_the_last_ply_reads_the_last_slot() -> None:
    engine = _gardner_duck()
    state = _after(engine, 1)
    assert decode_move(engine.legal_moves(state)[0])[2] == MOVE_DUCK
    values = []
    for ply in (1, MAX_PLY + 2):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=DEFAULT_REPAIRS & ~DUCK_QUIESCE)
        assert agent.repairs & DUCK_MOBILITY
        # The turn's start's count, filed where each ply reads it: slot 0 for ply 1, the
        # last slot (95) for any ply past it; slot 94 stays empty.
        agent._turn_moves[0] = agent._turn_moves[MAX_PLY - 1] = 9
        values.append(agent._quiesce(state, 4, ply, -INF, INF))
    assert values == [0.45000000000000534, 0.45000000000000534]


# ------------------------------------------------------------------ the root's plies


def test_the_roots_replies_are_ply_one() -> None:
    """The root is ply 0 and every reply ply 1, as in the core (`agents.rs::search_root`):
    each node files its turn's legal count under its ply (``duck_mobility``), so the root
    fills slot 0, its replies slot 1; on a duck game the replies are duck halves, filed
    nowhere, and slot 1 stays empty."""
    plain = Engine(_load("rulesets/gardner.json"))
    duck = _gardner_duck()
    got = []
    for engine, state in ((plain, plain.initial_state()), (duck, duck.initial_state())):
        agent = MinimaxAgent(engine, nodes=10**7)
        agent._begin_search()
        agent._search_root(state, engine.legal_moves(state), 3)
        got.append(agent._turn_moves[:6])
    assert got == [[7, 7, 9, 13, 7, 6], [7, 0, 6, 0, 0, 0]]


# ------------------------------------------------------------------ the seeing duck


def test_a_seeing_ducks_candidates_along_a_game() -> None:
    """`_duck_blocks` at every duck half of a seeded random game of `tests/_duck_shepherd.json`
    (a duck that sees), stepping and placing."""
    engine = Engine(_load("tests/_duck_shepherd.json"))
    agent = MinimaxAgent(engine, nodes=10, repairs=DUCK_QUIESCE)
    record = play_game(engine, ("random", "random"), 3)
    state = engine.initial_state()
    got = []
    for move in record.moves[:46]:
        moves = engine.legal_moves(state)
        if decode_move(moves[0])[2] == MOVE_DUCK:
            got.append(sorted(agent._duck_blocks(state, moves[0])))
        state = engine.apply(state, move)
    assert got == [
        [], [], [22, 30, 31, 37, 44, 51], [], [22, 29, 30, 31, 38, 44, 45, 46], [], [], [],
        [6, 22, 23, 31, 39, 47, 55], [], [6, 22, 23, 31, 39, 46, 55], [],
        [13, 19, 20, 21, 28, 35, 36, 37, 45, 51], [],
        [13, 14, 21, 23, 30, 31, 37, 38, 39, 45, 47, 55], [], [6, 14, 23, 31, 38, 39, 46, 55], [],
        [0, 6, 9, 14, 16, 17, 23, 24, 31, 32, 38, 39, 40, 46, 48, 55], [],
        [0, 6, 9, 14, 16, 17, 23, 24, 31, 32, 38, 39, 40, 46, 48, 55],
        [9, 10, 11, 16, 17, 18, 19, 20, 24, 26, 28, 29, 32, 36, 38, 39, 41, 42, 43, 46, 55],
        [0, 6, 9, 14, 16, 17, 18, 19, 20, 23, 24, 26, 28, 31, 32, 36, 38, 39, 40, 41, 42, 43,
         46, 48, 55],
    ]


# ------------------------------------------------------------------ the sentinels


def test_a_kept_move_whose_word_is_one_is_tried_first() -> None:
    """A table's kept move is a move word; ``NO_MOVE`` must be no word a move can have. The
    rook's step b1-a1 is word 1: kept, it goes first, ahead of a killer, and the search costs
    less than with nothing kept, where the killer leads."""
    engine = _ending("R", "b1", "e1", "e8")
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    assert encode_move(1, 0) == 1 and 1 in moves
    costs = []
    for kept in (1, NO_MOVE):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=(DEFAULT_REPAIRS & ~MOPUP) | TT_MOVE)
        agent._table[agent._table_key(state)] = (0, 0.0, EXACT, agent._generation, kept)
        agent._killers[1][0] = moves[3]
        assert agent._negamax(state, 3, 1, -INF, INF) == 0.9
        costs.append(agent._nodes)
    assert costs == [602, 614]


def test_a_duck_refuted_by_nothing_moves_nothing_forward() -> None:
    """At a duck half whose children all answer from the table (so none cuts off and none
    names a refutation), the ducks are searched in their order and the one onto the empty
    corner, the only one that fails high, is reached where that order puts it. A reply slot
    left at anything but -1 would read as a refutation onto square 0 and bring it forward.
    The history puts the corner last."""
    engine = _gardner_duck()
    record = play_game(engine, ("random", "random"), 3)
    state = engine.initial_state()
    for move in record.moves:
        moves = engine.legal_moves(state)
        if state.board[0] == 0 and decode_move(moves[0])[2] == MOVE_DUCK:
            break
        state = engine.apply(state, move)
    corner = [m for m in moves if decode_move(m)[1] == 0]
    assert len(corner) == 1
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_REFUTE)
    agent.duck_cutoffs = []
    squares = engine.spec.n_squares
    for move in moves:
        if move not in corner:
            origin, target, _, _ = decode_move(move)
            agent._history[origin * squares + target] = 1
        child = engine.apply(state, move)
        assert child.side_to_move != state.side_to_move
        value = -5.0 if move in corner else 5.0
        agent._table[agent._table_key(child)] = (9, value, EXACT, agent._generation, NO_MOVE)
    assert agent._negamax(state, 2, 0, -INF, 1.0) == 5.0
    assert (agent.duck_cutoffs, agent._nodes) == ([len(moves) - 1], len(moves) + 1)


def test_duck_refute_searches_cost_along_a_game() -> None:
    engine = _gardner_duck()
    costs = []
    for plies in (1, 3, 5):
        state = _after(engine, plies)
        agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_REFUTE)
        agent._begin_search()
        agent._negamax(state, 4, 0, -INF, INF)
        costs.append(agent._nodes)
    assert costs == [129, 144, 202]


# ------------------------------------------------------------------ batch 2


def test_a_royal_alone_with_a_piece_in_hand_is_not_bare() -> None:
    from chess_variant_sandbox.agents.minimax import is_armed, is_bare

    engine = Engine(_load("rulesets/crazyhouse.json"))
    spec = engine.spec
    held = [0] * len(spec.piece_ids)
    held[spec.piece_ids.index("N")] = 1
    state = engine.make_state(
        {"e1": ("K", 0), "e8": ("K", 1)}, hands=(held, [0] * len(spec.piece_ids))
    )
    assert state.hands
    assert not is_bare(spec, state, 0)
    assert is_bare(spec, state, 1)
    # One piece in hand arms its holder.
    assert is_armed(spec, state, 0) and not is_armed(spec, state, 1)


def test_a_cutoff_is_filed_by_its_kind_under_duck_history() -> None:
    """Under ``duck_history`` a duck move's cutoff adds depth squared under side and target,
    a piece move's under origin and target, each in its own table."""
    engine = _gardner_duck()
    squares = engine.spec.n_squares
    agent = MinimaxAgent(engine, nodes=10, repairs=DEFAULT_REPAIRS | DUCK_HISTORY)
    duck = encode_move(7, 12, MOVE_DUCK)
    piece = encode_move(1, 11)
    for _ in range(2):
        agent._remember(duck, True, 3, 1, 1)
    agent._remember(piece, True, 2, 1, 0)
    assert agent._duck_history == {1 * squares + 12: 18}
    assert agent._history == {1 * squares + 11: 4}


def test_a_piece_moves_history_is_the_shared_one_under_duck_history() -> None:
    engine = _gardner_duck()
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    assert all(decode_move(m)[2] != MOVE_DUCK for m in moves) and len(moves) > 2
    last = moves[-1]
    origin, target, _, _ = decode_move(last)
    squares = engine.spec.n_squares
    agent = MinimaxAgent(engine, nodes=10, repairs=DEFAULT_REPAIRS | DUCK_HISTORY)
    agent._history[origin * squares + target] = 5
    ordered = list(moves)
    agent._order(state, ordered, 1)
    assert ordered[0] == last


def test_a_refutation_by_a_slide_names_its_path_exactly() -> None:
    """`_reply_blocks` of a slide is its target and the squares it crosses, nothing of any
    other piece's sources from the same origin."""
    engine = _gardner_duck()
    agent = MinimaxAgent(engine, nodes=10)
    state = engine.make_state({"a1": ("R", 0), "e1": ("K", 0), "e5": ("K", 1)}, side_to_move=1)
    reply = encode_move(engine._square_index("a1"), engine._square_index("a5"))
    blocks = agent._reply_blocks(state, reply)
    assert blocks == {engine._square_index(s) for s in ("a2", "a3", "a4", "a5")}


def test_an_announced_rounds_value_and_cost() -> None:
    engine, state, _ = _spy_round()
    phase = engine.phase(state)
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._begin_search()
    value = agent._round_node(state, phase, 2, 1, -INF, INF)
    # An announced round is two plies: the answers' children are ply 3, where their turns'
    # counts are filed, and ply 4 below them.
    assert (value, agent._nodes, agent._turn_moves[:5]) == (-7.1725, 561, [0, 0, 0, 14, 12])


def test_an_answer_rows_value_and_cost_where_answers_hand_the_move_over() -> None:
    """The answerer's ply against the capture that promotes into a spy, from ply 1: the
    answers that hand the move over are searched one ply further down (ply 2)."""
    engine, state, announced = _spy_round()
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._begin_search()
    answers = engine.legal_moves_for(state, 0)
    value = agent._answer_row(state, 1, announced, answers, 2, 1, -INF, INF)
    assert (value, agent._nodes) == (-1.0324999999999998, 183)


def test_a_refutation_by_a_leap_names_no_slides_path() -> None:
    """A leap two squares along a file crosses nothing, though a slide from the same origin
    onto the same target would cross the square between: `_reply_blocks` is the target alone."""
    leap = [{"kind": "leap", "vectors": [[0, 2]], "symmetry": "rotate4"}]
    king = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    slide = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    spec = {
        "name": "leap beside a slide (pins)",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": king},
            {"id": "R", "name": "rook", "value": 5, "moves": slide},
            {"id": "L", "name": "leaper", "value": 2, "moves": leap},
            {"id": "D", "name": "duck", "value": 0, "moves": []},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["f1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 200,
                  "duck": {"piece": "D"}},
    }
    engine = Engine(spec)
    agent = MinimaxAgent(engine, nodes=10)
    state = engine.make_state(
        {"a1": ("L", 0), "f1": ("K", 0), "f6": ("K", 1)}, side_to_move=1
    )
    a1, a3 = engine._square_index("a1"), engine._square_index("a3")
    assert agent._reply_blocks(state, encode_move(a1, a3)) == {a3}


# ------------------------------------------------------------------ batch 3


def test_an_announcement_of_two_whole_passes_is_the_cores() -> None:
    """Under ``max_depth=2`` and a budget it never reaches, the announcer finishes two passes,
    the second led by the first's best, in the core's node count."""
    engine, state, _ = _spy_round()
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    agent = "minimax:nodes=100000,max_depth=2"
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, _, nodes, passes = engine.select_round(
        state, 1, native, eval_values(engine.spec), 7, None
    )
    player = build_agent(agent, engine)
    assert isinstance(player, MinimaxAgent)
    assert player.select_announcement(state, 1, lists, seat_rng(7, 1)) == move
    assert (player.stats.nodes, player.stats.depth_sum) == (nodes, passes) == (484, 2)
    # The announcement is ply 0, its answers ply 1, their children ply 2 (as in the core):
    # the turns' counts are filed from slot 2.
    assert player._turn_moves[:6] == [0, 0, 10, 4, 12, 0]


def test_a_duck_halfs_plies_follow_its_piece_half() -> None:
    """A piece half and the duck half after it are two plies, though one side moves both:
    from gardner-duck's opening a depth-3 search is -1.46 in 173 nodes, and the turns'
    counts sit at plies 0 and 2."""
    engine = _gardner_duck()
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._begin_search()
    value = agent._negamax(engine.initial_state(), 3, 0, -INF, INF)
    assert (value, agent._nodes, agent._turn_moves[:4]) == (-1.4600000000000026, 173, [7, 0, 6, 0])


def test_a_duck_half_at_the_last_quiescence_ply_is_searched_through() -> None:
    """Under ``duck_quiesce`` an owed duck half at quiescence depth 1 still plays its ducks."""
    engine = _gardner_duck()
    state = _after(engine, 1)
    got = []
    for depth in (1, 2):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
        got.append((agent._quiesce(state, depth, 1, -INF, INF), agent._nodes))
    assert got == [(-3.8799999999999972, 4), (-2.1199999999999966, 8)]
    # At depth 0 it stands pat: one node.
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
    assert (agent._quiesce(state, 0, 1, -INF, INF), agent._nodes) == (0.25000000000000533, 1)
    # Seven plies in, the ducks' window narrows as they improve: 22 nodes, not 26.
    later = _after(engine, 7)
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
    assert (agent._quiesce(later, 2, 1, -INF, INF), agent._nodes) == (-3.2800000000000007, 22)


def test_a_lone_legal_move_under_duck_quiesce_stands_pat() -> None:
    """A side with one legal move and nothing loud stands pat, duck repair or not."""
    engine = _ending("R", "h2", "c3", "a1")
    state = engine.initial_state()
    lone = replace(state, side_to_move=1, key=state.key ^ engine.spec.zobrist_side)
    assert len(engine.legal_moves(lone)) == 1
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
    assert (agent._quiesce(lone, 2, 1, -INF, INF), agent._nodes) == (-0.95, 1)


# ------------------------------------------------------------------ batch 4


def test_a_budget_is_its_nodes_and_defaults_to_twenty_thousand() -> None:
    engine = Engine(_load("rulesets/gardner.json"))
    assert [MinimaxAgent(engine, nodes=n).budget for n in (0, 1, 2)] == [1, 1, 2]
    default = build_agent("minimax", engine)
    assert isinstance(default, MinimaxAgent)
    assert (default.budget, default.name) == (20_000, "minimax:nodes=20000")


def test_a_quiescence_past_the_last_ply_clears_the_last_slot() -> None:
    engine = Engine(_load("rulesets/gardner.json"))
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._refuters[MAX_PLY - 1] = agent._refuters[MAX_PLY - 2] = 7
    agent._quiesce(engine.initial_state(), 0, MAX_PLY, -INF, INF)
    assert agent._refuters[MAX_PLY - 2:] == [7, -1]


KING_STEPS = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
ROOK_SLIDE = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]


def _small_duck(duck_moves: Any = ()) -> Engine:
    """A 6x6 game won by taking the royal, with a duck (by its moves when ``duck_moves``)."""
    duck: dict[str, Any] = {"piece": "D"}
    if duck_moves:
        duck["goes"] = "by_moves"
    return Engine({
        "name": "small duck (pins)",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING_STEPS},
            {"id": "R", "name": "rook", "value": 5, "moves": ROOK_SLIDE},
            {"id": "P", "name": "stone", "value": 1, "moves": KING_STEPS},
            {"id": "D", "name": "duck", "value": 0, "moves": list(duck_moves)},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["f1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 200, "duck": duck},
    })


def test_a_boxed_ducks_lone_pass_is_searched_through() -> None:
    """Under ``duck_quiesce`` a duck half whose only move is the pass plays it."""
    engine = _small_duck(KING_STEPS)
    state = engine.make_state(
        {"a1": ("D", 2), "a2": ("K", 0), "b1": ("R", 0), "b2": ("P", 1), "f6": ("K", 1)},
        duck_owed=True,
    )
    assert len(engine.legal_moves(state)) == 1
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
    assert (agent._quiesce(state, 2, 1, -INF, INF), agent._nodes) == (-999998.0, 6)


def _table_children(agent: MinimaxAgent, state: GameState, high: set[int]) -> None:
    """File every duck child of ``state`` as an exact entry: the ``high`` ducks fail high."""
    engine = agent.engine
    for move in engine.legal_moves(state):
        child = engine.apply(state, move)
        value = -5.0 if decode_move(move)[1] in high else 5.0
        agent._table[agent._table_key(child)] = (9, value, EXACT, agent._generation, NO_MOVE)


def test_duck_front_brings_the_blocks_to_the_very_front() -> None:
    """The opponent's rook on b1 attacks the mover's on b4 through b2 and b3; under
    ``duck_front`` the ducks onto those (and onto c1-e1, its path to the royal) go first,
    ahead even of the history's favourite."""
    engine = _small_duck()
    state = engine.make_state(
        {"b4": ("R", 0), "f1": ("K", 0), "b1": ("R", 1), "f6": ("K", 1)}, duck_owed=True
    )
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_FRONT)
    first = engine.legal_moves(state)[0]
    blocks = agent._duck_blocks(state, first)
    assert blocks == {engine._square_index(s) for s in ("b2", "b3", "c1", "d1", "e1")}
    favourite = next(m for m in engine.legal_moves(state) if decode_move(m)[1] not in blocks)
    origin, target, _, _ = decode_move(favourite)
    agent._history[origin * engine.spec.n_squares + target] = 9
    agent.duck_cutoffs = []
    _table_children(agent, state, blocks)
    assert agent._negamax(state, 2, 0, -INF, 1.0) == 5.0
    assert agent.duck_cutoffs == [0]


def test_a_duck_refuted_by_move_word_one_brings_its_target_forward() -> None:
    """The reply b1-a1 is move word 1. The history's favourite duck is searched first and
    refuted by it (a killer, inside a narrow window); every other duck answers from the table,
    and only the one onto a1, put last by the history, fails high. The refutation brings it
    to second place, where it cuts off, five nodes in."""
    engine = _small_duck()
    state = engine.make_state(
        {"c3": ("R", 0), "f1": ("K", 0), "b1": ("R", 1), "f6": ("K", 1)}, duck_owed=True
    )
    moves = engine.legal_moves(state)
    squares = engine.spec.n_squares
    favourite = next(m for m in moves if decode_move(m)[1] != 0)
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_REFUTE)
    for move in moves:
        origin, target, _, _ = decode_move(move)
        if target != 0:
            agent._history[origin * squares + target] = 9 if move == favourite else 1
        if move != favourite:
            child = engine.apply(state, move)
            value = -5.0 if target == 0 else 5.0
            agent._table[agent._table_key(child)] = (9, value, EXACT, agent._generation, NO_MOVE)
    agent._killers[1][0] = encode_move(1, 0)
    agent.duck_cutoffs = []
    assert agent._negamax(state, 2, 0, 0.9, 1.0) == 5.0
    assert (agent.duck_cutoffs, agent._nodes) == ([1], 5)


# ------------------------------------------------------------------ batch 5


def test_no_killer_before_any_cutoff() -> None:
    """A fresh agent and one whose search has begun hold no killer: the rook's b1-a1, move
    word 1, is ordered as any quiet move, behind the one the history favours, and the
    search costs 602 nodes either way."""
    engine = _ending("R", "b1", "e1", "e8")
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    assert 1 in moves
    squares = engine.spec.n_squares
    e1, d1 = engine._square_index("e1"), engine._square_index("d1")
    for begun in (False, True):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=DEFAULT_REPAIRS & ~MOPUP)
        if begun:
            agent._begin_search()
        agent._history[e1 * squares + d1] = 50
        for ply in (0, 1, MAX_PLY - 1):
            ordered = list(moves)
            agent._order(state, ordered, ply)
            assert ordered[0] == encode_move(e1, d1) and ordered[1] == 1
        agent._history.clear()
        assert (agent._negamax(state, 3, 1, -INF, INF), agent._nodes) == (0.9, 602)


def test_a_quiescence_cutoff_past_the_last_ply_is_filed_in_the_last_slot() -> None:
    engine = Engine(_load("rulesets/gardner.json"))
    state = _after(engine, 1)
    agent = MinimaxAgent(engine, nodes=10**7)
    stand_pat = 0.35000000000000003
    value = agent._quiesce(state, 2, MAX_PLY, stand_pat - 0.01, stand_pat + 0.01)
    assert (value, agent._refuters[MAX_PLY - 2:]) == (2.35, [-1, 2576])


# ------------------------------------------------------------------ batch 6


def test_a_cutoffs_history_weight_is_its_depth_squared() -> None:
    """A quiet cutoff at depth 3 adds 9 to its from-to history, one at depth 1 adds 1, and
    one at depth 0 or below adds nothing."""
    engine = _ending("R", "b1", "e1", "e8")
    agent = MinimaxAgent(engine, nodes=10)
    squares = engine.spec.n_squares
    move = encode_move(1, 0)
    for depth in (3, 1, 0, -2):
        agent._remember(move, True, depth, 1, 0)
    assert agent._history == {1 * squares + 0: 10}
    assert agent._killers[1] == [move, -1]


def _wide_duck() -> Engine:
    """A 16x16 game won by taking the royal, with a duck: the board's last square is 255."""
    return Engine({
        "name": "wide duck (pins)",
        "board": {"width": 16, "height": 16},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING_STEPS},
            {"id": "D", "name": "duck", "value": 0, "moves": []},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["a16"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 200,
                  "duck": {"piece": "D"}},
    })


def test_a_duck_refuted_by_nothing_on_a_wide_board_moves_nothing_forward() -> None:
    """On a 16x16 board square 255 is real. At a duck half whose children all answer from the
    table (none names a refutation), the duck onto 255, put last by the history and the only
    one that fails high, is reached last: a cleared reply slot reads as no refutation, not as
    one onto square 255."""
    engine = _wide_duck()
    state = engine.make_state({"a1": ("K", 0), "a16": ("K", 1)}, duck_owed=True)
    moves = engine.legal_moves(state)
    assert all(decode_move(m)[2] == MOVE_DUCK for m in moves) and len(moves) == 254
    corner = [m for m in moves if decode_move(m)[1] == 255]
    assert len(corner) == 1 and moves[-1] == corner[0]
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_REFUTE)
    agent.duck_cutoffs = []
    squares = engine.spec.n_squares
    for move in moves:
        if move not in corner:
            origin, target, _, _ = decode_move(move)
            agent._history[origin * squares + target] = 1
        child = engine.apply(state, move)
        value = -5.0 if move in corner else 5.0
        agent._table[agent._table_key(child)] = (9, value, EXACT, agent._generation, NO_MOVE)
    assert agent._negamax(state, 2, 0, -INF, 1.0) == 5.0
    assert (agent.duck_cutoffs, agent._nodes) == ([253], 255)


# ------------------------------------------------------------------ batch 7


def test_a_piece_halfs_count_past_the_last_ply_is_filed_in_the_last_slot() -> None:
    """``_note_turn`` files a piece half's count at its ply, clamped to the last slot (95);
    a duck half's own list is never filed."""
    engine = _gardner_duck()
    pieces = engine.legal_moves(engine.initial_state())
    ducks = engine.legal_moves(_after(engine, 1))
    assert len(pieces) == 7 and decode_move(ducks[0])[2] == MOVE_DUCK
    for ply in (MAX_PLY - 1, MAX_PLY + 3):
        agent = MinimaxAgent(engine, nodes=10)
        agent._note_turn(pieces, ply)
        assert agent._turn_moves[MAX_PLY - 3:] == [0, 0, 7]
        agent._note_turn(ducks, ply)
        agent._note_turn([], ply)
        assert agent._turn_moves[MAX_PLY - 3:] == [0, 0, 7]


def test_an_unheard_duck_has_no_history_under_duck_history() -> None:
    """A duck whose target holds no history ranks at zero: one filed at weight one, the least
    a cutoff files, comes first."""
    engine = _gardner_duck()
    state = _after(engine, 1)
    moves = engine.legal_moves(state)
    assert all(decode_move(m)[2] == MOVE_DUCK for m in moves) and len(moves) > 2
    last = moves[-1]
    agent = MinimaxAgent(engine, nodes=10, repairs=DEFAULT_REPAIRS | DUCK_HISTORY)
    agent._duck_history[state.side_to_move * engine.spec.n_squares + decode_move(last)[1]] = 1
    ordered = list(moves)
    agent._order(state, ordered, 1)
    assert ordered == [last, *moves[:-1]]


def test_a_duck_half_that_reaches_beta_exactly_stops() -> None:
    """Seed 1's game, seven plies in, owes a duck half whose depth-1 quiescence is -3.98 in 10
    nodes; with beta at that value the ducks stop on reaching it: 4 nodes."""
    engine = _gardner_duck()
    state = _after(engine, 7, seed=1)
    assert decode_move(engine.legal_moves(state)[0])[2] == MOVE_DUCK
    full = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
    value = full._quiesce(state, 1, 1, -INF, INF)
    assert (value, full._nodes) == (-3.980000000000001, 10)
    closed = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE)
    assert (closed._quiesce(state, 1, 1, value - 1, value), closed._nodes) == (value, 4)


# ------------------------------------------------------------------ batch 8


def test_a_repair_is_spelled_zero_or_one_and_nothing_else() -> None:
    assert repairs_of({"keep": 0}) & KEEP_TABLE == 0
    assert repairs_of({"keep": 1}) & KEEP_TABLE == KEEP_TABLE
    for value in (-1, 2):
        with pytest.raises(ValueError, match=rf"keep={value}: a search repair is spelled 0 or 1"):
            repairs_of({"keep": value})


def test_a_placed_duck_lifts_no_square() -> None:
    """A duck placed (not stepped) lifts nothing: a piece of the mover's on square 1 still
    screens its neighbour on square 0 from the slider beyond it, so only the open attack on
    the royal (through d1) is blocked."""
    engine = _gardner_duck()
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("N", 0), "c1": ("R", 1), "e1": ("K", 0), "e5": ("K", 1)},
        side_to_move=0,
    )
    agent = MinimaxAgent(engine, nodes=10)
    d3 = engine._square_index("d3")
    assert engine._square_index("b1") == 1
    placed = encode_move(d3, d3, MOVE_DUCK, DUCK_PLACE)
    assert agent._duck_blocks(state, placed) == {engine._square_index("d1")}


def test_a_round_that_reaches_beta_exactly_is_filed_as_a_lower_bound() -> None:
    engine, state, _ = _spy_round()
    phase = engine.phase(state)
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._begin_search()
    assert agent._round_node(state, phase, 1, 1, -INF, -7.1725) == -7.1725
    assert agent._table[agent._table_key(state)][:3] == (1, -7.1725, LOWER)


def test_duck_front_alone_brings_no_refutations_forward() -> None:
    """Seed 2's game nine plies in, searched to depth 3: ``duck_front`` alone costs 726 nodes,
    and only with ``duck_refute`` beside it do the refutations save 86 of them."""
    engine = _gardner_duck()
    state = _after(engine, 9, seed=2)
    got = []
    for repairs in (DUCK_FRONT, DUCK_FRONT | DUCK_REFUTE):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=repairs)
        agent._begin_search()
        got.append((agent._negamax(state, 3, 0, -INF, INF), agent._nodes))
    assert got == [(11.04, 726), (11.04, 640)]


def test_a_kept_move_that_is_not_legal_here_is_passed_over() -> None:
    """A key can be shared by positions with different moves (the key omits the clocks), so a
    kept move is tried first only when it is legal here, and only under ``tt_move``: a word no
    move here has is passed over, and without ``tt_move`` a planted word 1 is ignored; both cost
    what nothing kept costs, with the killer leading."""
    engine = _ending("R", "b1", "e1", "e8")
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    stranger = encode_move(63, 62)
    assert stranger not in moves and 1 in moves
    got = []
    for repairs, kept in (
        ((DEFAULT_REPAIRS & ~MOPUP) | TT_MOVE, stranger),
        (DEFAULT_REPAIRS & ~MOPUP, 1),
    ):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=repairs)
        agent._table[agent._table_key(state)] = (0, 0.0, EXACT, agent._generation, kept)
        agent._killers[1][0] = moves[3]
        got.append((agent._negamax(state, 3, 1, -INF, INF), agent._nodes))
    assert got == [(0.9, 614), (0.9, 614)]


# ------------------------------------------------------------------ batch 10


def test_a_bare_constructors_budget_is_twenty_thousand() -> None:
    """The constructor's own default, not `build_agent`'s spelling of it."""
    agent = MinimaxAgent(Engine(_load("rulesets/gardner.json")))
    assert (agent.budget, agent.name) == (20_000, "minimax:nodes=20000")


def test_a_node_past_the_last_ply_clears_the_last_refuter_slot() -> None:
    """A node at ply 96 answered from the table still clears its refuter slot, the last (95),
    and leaves slot 94 alone."""
    engine = Engine(_load("rulesets/gardner.json"))
    state = engine.initial_state()
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._table[agent._table_key(state)] = (9, 1.5, EXACT, agent._generation, NO_MOVE)
    agent._refuters[MAX_PLY - 1] = agent._refuters[MAX_PLY - 2] = 7
    assert agent._negamax(state, 2, MAX_PLY, -INF, INF) == 1.5
    assert agent._refuters[MAX_PLY - 2:] == [7, -1]


def test_a_duck_half_at_the_last_plies_reads_its_childs_refuter() -> None:
    """Under ``duck_refute`` a duck half at ply 95 or 96 reads its children's refutations from
    the last slot, and searches as one at ply 1 does: -3.88 in 28 nodes at depth 2."""
    engine = _gardner_duck()
    state = _after(engine, 1)
    assert decode_move(engine.legal_moves(state)[0])[2] == MOVE_DUCK
    got = []
    for ply in (1, MAX_PLY - 1, MAX_PLY):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_REFUTE)
        got.append((agent._negamax(state, 2, ply, -INF, INF), agent._nodes))
    assert got == [(-3.8799999999999972, 28)] * 3


def test_a_duck_half_at_the_root_reads_slot_zero() -> None:
    """A duck half at ply 0 has no ply above it: its turn's count is read from slot 0 (here 9),
    never the last slot (left 0), so it stands pat at 0.45, not 0."""
    engine = _gardner_duck()
    state = _after(engine, 1)
    assert decode_move(engine.legal_moves(state)[0])[2] == MOVE_DUCK
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DEFAULT_REPAIRS & ~DUCK_QUIESCE)
    agent._turn_moves[0] = 9
    assert agent._quiesce(state, 0, 0, -INF, INF) == 0.45000000000000534


def test_a_quiescent_piece_halfs_count_past_the_last_ply_is_filed_in_the_last_slot() -> None:
    engine = Engine(_load("rulesets/gardner.json"))
    for ply in (MAX_PLY - 1, MAX_PLY + 2):
        agent = MinimaxAgent(engine, nodes=10**7)
        assert agent.repairs & DUCK_MOBILITY
        agent._quiesce(engine.initial_state(), 0, ply, -INF, INF)
        assert agent._turn_moves[MAX_PLY - 3:] == [0, 0, 7]


# ------------------------------------------------------------------ batch 11 (Q1153)


def test_duck_refute_alone_brings_no_blocks_to_the_front() -> None:
    """The position `test_duck_front_brings_the_blocks_to_the_very_front` searches, under
    ``duck_refute`` alone: the duck half is still a duck half for it, but only ``duck_front``
    sends the blocks first, so the history's favourite leads and the first block to fail
    high is reached where the order puts it."""
    engine = _small_duck()
    state = engine.make_state(
        {"b4": ("R", 0), "f1": ("K", 0), "b1": ("R", 1), "f6": ("K", 1)}, duck_owed=True
    )
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_REFUTE)
    blocks = agent._duck_blocks(state, engine.legal_moves(state)[0])
    favourite = next(m for m in engine.legal_moves(state) if decode_move(m)[1] not in blocks)
    origin, target, _, _ = decode_move(favourite)
    agent._history[origin * engine.spec.n_squares + target] = 9
    agent.duck_cutoffs = []
    _table_children(agent, state, blocks)
    assert agent._negamax(state, 2, 0, -INF, 1.0) == 5.0
    assert (agent.duck_cutoffs, agent._nodes) == ([1], 3)


def _cannon_duck() -> Engine:
    """A 6x6 royal-capture game with a duck and a cannon that slides to move and hops to take."""
    return Engine({
        "name": "cannon duck (pins)",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING_STEPS},
            {"id": "R", "name": "rook", "value": 5, "moves": ROOK_SLIDE},
            {"id": "C", "name": "cannon", "value": 4, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"},
                {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
            ]},
            {"id": "D", "name": "duck", "value": 0, "moves": []},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["f1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 200,
                  "duck": {"piece": "D"}},
    })


def test_an_unscreened_hopper_threatens_nothing_a_duck_could_block() -> None:
    """The opponent's hopper on b1 has open lines to the mover's rook on b4 and royal on f1 but
    nothing to hop over, so it attacks neither: no duck square blocks anything. A screen-needing
    source must be passed over before its path is read."""
    engine = _cannon_duck()
    state = engine.make_state(
        {"b4": ("R", 0), "f1": ("K", 0), "b1": ("C", 1), "f6": ("K", 1)}, duck_owed=True
    )
    agent = MinimaxAgent(engine, nodes=10)
    assert agent._duck_blocks(state, engine.legal_moves(state)[0]) == set()


def test_a_seeing_duck_placed_lifts_no_corner() -> None:
    """In `tests/_duck_shepherd.json` (a duck that sees its eight neighbours) the opponent's
    leaper on d3 attacks the mover's on b2 and royal on e1. A placed duck lifts no square, so
    the mover's own piece on a1 (square 0) keeps the duck from standing there, though a1 is
    beside b2."""
    engine = Engine(_load("tests/_duck_shepherd.json"))
    state = engine.make_state(
        {"a1": ("R", 0), "b2": ("N", 0), "e1": ("K", 0), "d3": ("N", 1), "e8": ("K", 1)},
        side_to_move=0, duck_owed=True,
    )
    first = engine.legal_moves(state)[0]
    assert decode_move(first)[2:] == (MOVE_DUCK, DUCK_PLACE)
    agent = MinimaxAgent(engine, nodes=10)
    assert sorted(agent._duck_blocks(state, first)) == [
        1, 2, 3, 5, 8, 10, 11, 12, 13, 16, 17, 18, 20, 26, 27, 28,
    ]


# ------------------------------------------------------------------ batch 12


def test_a_duck_cutoff_without_duck_history_is_filed_in_the_shared_history() -> None:
    """With ``duck_history`` off a duck move's cutoff is filed like a piece move's, under
    origin and target in the shared table, and the duck table stays empty."""
    engine = _gardner_duck()
    squares = engine.spec.n_squares
    assert not DEFAULT_REPAIRS & DUCK_HISTORY
    agent = MinimaxAgent(engine, nodes=10)
    agent._remember(encode_move(7, 12, MOVE_DUCK), True, 3, 1, 1)
    assert agent._duck_history == {}
    assert agent._history == {7 * squares + 12: 9}


def test_a_duck_quiescences_replies_file_their_counts_one_ply_down() -> None:
    """Under ``duck_quiesce`` and ``duck_mobility``, the opponent's piece halves below an owed
    duck half at ply 1 are ply 2, so their turns' counts are filed in slot 2 and the duck
    halves under them read it."""
    engine = _gardner_duck()
    state = _after(engine, 1)
    agent = MinimaxAgent(engine, nodes=10**7, repairs=DUCK_QUIESCE | DUCK_MOBILITY)
    value = agent._quiesce(state, 2, 1, -INF, INF)
    got = (value, agent._nodes, agent._turn_moves[:5])
    assert got == (-2.1699999999999964, 8, [0, 0, 7, 0, 5])


def test_a_spy_games_minimax_refuses_a_net_by_name() -> None:
    """A spy game's minimax plays the built-in objective only: a ``net=`` is refused as one
    (no net plays a spy game), before any store is looked for."""
    engine, _, _ = _spy_round()
    with pytest.raises(ValueError, match=r"net names an objective .*\(no net plays a spy game\)"):
        build_agent("minimax:net=000000000000,nodes=100", engine)


def test_turn_counts_are_written_only_under_duck_mobility() -> None:
    """``_turn_moves`` is written only under ``duck_mobility``: without the bit a root search
    files no count, with it the root's own count (gardner-duck's opening, 7) sits in slot 0."""
    engine = _gardner_duck()
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    off = MinimaxAgent(engine, nodes=10**7, repairs=DEFAULT_REPAIRS & ~DUCK_MOBILITY)
    on = MinimaxAgent(engine, nodes=10**7, repairs=DEFAULT_REPAIRS | DUCK_MOBILITY)
    for agent in (off, on):
        agent._begin_search()
        agent._search_root(state, list(moves), 1)
    assert (off._turn_moves[0], on._turn_moves[0]) == (0, 7)


# ------------------------------------------------------------------ Q1153 (minimax-13)


def test_an_answer_that_hands_the_move_over_is_searched_one_ply_down() -> None:
    """Only the answers that hand the move over, from ply 1 at depth 1: each child is a node at
    ply 2, which clears that ply's refuter slot (and leaves ply 1's and ply 3's alone)."""
    engine, state, announced = _spy_round()
    agent = MinimaxAgent(engine, nodes=10**7)
    agent._begin_search()
    answers = [
        m for m in engine.legal_moves_for(state, 0)
        if engine.apply_round(state, *_pair(1, announced, m)).side_to_move != 0
    ]
    agent._refuters[1:5] = [7777, 7777, 7777, 7777]
    value = agent._answer_row(state, 1, announced, answers, 1, 1, -INF, INF)
    assert (len(answers), value) == (7, -7.2175)
    assert agent._refuters[1:4] == [7777, -1, 7777]


def test_a_killer_in_the_last_slot_orders_a_node_past_the_last_ply() -> None:
    """A killer filed in the last slot (95) goes first at ply 95 and at any ply past it; one
    in slot 94 is not read there."""
    engine = Engine(_load("rulesets/gardner.json"))
    state = engine.initial_state()
    moves = engine.legal_moves(state)
    last, other = moves[-1], moves[-2]
    for ply in (MAX_PLY - 1, MAX_PLY + 4):
        agent = MinimaxAgent(engine, nodes=10)
        agent._killers[MAX_PLY - 1] = [last, -1]
        agent._killers[MAX_PLY - 2] = [other, -1]
        ordered = list(moves)
        agent._order(state, ordered, ply)
        assert ordered[0] == last and ordered[1:] == [m for m in moves if m != last]


def test_a_duck_half_two_plies_from_the_last_reads_its_childs_refuter() -> None:
    """Under ``duck_refute`` a duck half at ply 94 reads its children's refutations from slot 95,
    where they file them, and searches as one at ply 1 does: -0.86 in 355 nodes at depth 3,
    against 367 with the repair off."""
    engine = _gardner_duck()
    state = _after(engine, 7)
    assert decode_move(engine.legal_moves(state)[0])[2] == MOVE_DUCK
    got = []
    for ply, repairs in ((1, DUCK_REFUTE), (MAX_PLY - 2, DUCK_REFUTE), (MAX_PLY - 2, 0)):
        agent = MinimaxAgent(engine, nodes=10**7, repairs=repairs)
        got.append((agent._negamax(state, 3, ply, -INF, INF), agent._nodes))
    assert got == [(-0.8600000000000008, 355)] * 2 + [(-0.8600000000000008, 367)]
