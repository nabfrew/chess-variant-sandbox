"""Pins on `agents/evaluation.py`'s helpers, each on a small inline spec (Q1152's survivors).

The evaluator's Python side is the declared oracle for `rust/src/eval.rs`; these hold the
values its gates and tables produce on inputs the boundary comparisons never build: a
capture relation beside an explosion, a strike, wild pieces or a hand; a race whose
arrival draws or loses; a promotion that changes what a piece may take; a sweeper that
turns on a board that is not square; a royal that starts in the stock.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from test_speed import gardner as speed_gardner

from chess_variant_sandbox.agents import evaluation
from chess_variant_sandbox.agents.evaluation import (
    Evaluator,
    _clamp,
    _measured_values,
    _royal_loss_ends_it,
    capture_threats,
    caught_royal_falls,
    has_goal_terms,
    immunity_threats,
    rested_royal_falls,
    seal_goals,
)
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.spec import load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]

STEP = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}]


def rps_spec(extra: tuple[dict[str, Any], ...] = (), **rules: Any) -> dict[str, Any]:
    """Three king-stepping types that take in a cycle, on 5x5; a race to the far corner.
    ``rules`` overrides (a value of ``None`` drops the key)."""
    spec: dict[str, Any] = {
        "schema_version": 31,
        "name": "evaluation pins",
        "description": "A test game: three types that take one another in a cycle.",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "R", "name": "rock", "value": 1, "captures": ["S"], "moves": STEP},
            {"id": "P", "name": "paper", "value": 1, "captures": ["R"], "moves": STEP},
            {"id": "S", "name": "scissors", "value": 1, "captures": ["P"], "moves": STEP},
            *copy.deepcopy(list(extra)),
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "P", "player": 0, "squares": ["b1"]},
            {"piece": "S", "player": 0, "squares": ["c1"]},
            {"piece": "R", "player": 1, "squares": ["e5"]},
            {"piece": "P", "player": 1, "squares": ["d5"]},
            {"piece": "S", "player": 1, "squares": ["c5"]},
        ]},
        "rules": {
            "win": "annihilation", "stalemate": "loss", "max_plies": 100,
            "repetition_draw": "never", "no_progress_draw": 50,
            "reach": {"squares": ["e5"], "mirror": "rotate", "result": "win"},
        },
    }
    spec["rules"].update(rules)
    spec["rules"] = {key: value for key, value in spec["rules"].items() if value is not None}
    return spec


def rps(extra: tuple[dict[str, Any], ...] = (), **rules: Any) -> Engine:
    spec = rps_spec(extra, **rules)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


WOLF = {"id": "W", "name": "wolf", "value": 3, "moves": STEP}


def with_wild() -> Engine:
    spec = rps_spec((WOLF,))
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["a3", "b3", "c3"]}]
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def variants() -> dict[str, Engine]:
    return {
        "explosion": rps(explode_on_capture={"radius": 1}),
        "strike": rps(strike={"effect": "capture", "position": "stay"}),
        "wild": with_wild(),
        "drops": rps(drops={"from_captures": True}),
    }


# ----------------------------------------------------------------- material and its sign


def test_running_out_of_anything_as_a_win_inverts_material_by_exactly_one() -> None:
    engine = rps(extinction={"pieces": "any", "result": "win"})
    evaluator = Evaluator.for_engine(engine)
    assert evaluator.material_sign == -1.0
    assert Evaluator.for_engine(rps()).material_sign == 1.0
    # An empty first square, then the pieces: a1 empty, b1/c1 player 0's, e5 player 1's.
    state = engine.make_state({"b1": ("P", 0), "c1": ("S", 0), "e5": ("R", 1)})
    assert evaluator.values == (1.0, 1.0, 1.0)
    assert evaluator._plain_material(state.board, 0) == 1.0
    assert evaluator._plain_material(state.board, 1) == -1.0
    assert evaluator.board_material(state.board, 0) == -1.0


def test_inverted_material_skips_a_wild_piece_and_counts_what_follows_it() -> None:
    """A wild piece on a2 (board order 5), player 0's three on rank 1 cleared, player 1's
    three on rank 5 after it: the wild piece is nobody's, and the sum goes on past it."""
    spec = rps_spec((WOLF,), extinction={"pieces": "any", "result": "win"})
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["a2", "b2", "c2"]}]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine)
    board = bytearray(engine.initial_state().board)
    assert board[5] and engine.spec.owner_of(board[5]) == evaluation.WILD
    board[0] = board[1] = board[2] = 0
    assert evaluator._plain_material(bytes(board), 0) == -3.0
    assert evaluator.board_material(bytes(board), 0) == 3.0


# ----------------------------------------------------------------------- the clamp


def test_clamp_holds_each_end_and_passes_the_middle() -> None:
    assert _clamp(1.5) == 1.0
    assert _clamp(-1.5) == -1.0
    assert _clamp(0.25) == 0.25
    assert _clamp(1.0) == 1.0


# ---------------------------------------------------------------- the goal gates


def test_a_colouring_that_wins_nothing_is_no_goal_term() -> None:
    assert has_goal_terms(rps(reach=None, territory={"mode": "permanent"}).spec) is False
    assert has_goal_terms(rps(reach=None, territory={"mode": "permanent", "win_at": 13}).spec)


def test_an_unpriced_race_counted_to_two_arrivals_is_a_goal_term() -> None:
    safe = {"squares": ["e5"], "mirror": "rotate", "safe": True}
    assert has_goal_terms(rps(reach={**safe, "count": 2}).spec) is True
    assert has_goal_terms(rps(reach=safe).spec) is False


def test_declared_values_below_one_are_scaled_by_the_smallest_of_them() -> None:
    """Every declared value halved to 0.5: the declared scale is 0.5 too, so the blended
    values are the same as at 1."""
    spec = rps_spec()
    for piece in spec["pieces"]:
        piece["value"] = 0.5
    assert validate_spec(spec) == [], validate_spec(spec)
    assert Evaluator.for_engine(Engine(spec)).values == (1.0, 1.0, 1.0)


def test_only_the_first_players_rock_immune_is_still_counted() -> None:
    """Player 1 holds no paper: player 0's rock (type 0, mask 1) is the only immune piece."""
    engine = rps(reach=None)
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"a1": ("R", 0), "b1": ("P", 0), "c1": ("S", 0), "e5": ("R", 1), "c5": ("S", 1)})
    assert evaluator.immunity(state, 0) == 1.0
    assert evaluator.immunity(state, 1) == -1.0


def test_a_shortfall_counts_each_side_against_its_own_opening_army() -> None:
    """Player 0 starts with three pieces, player 1 with two, a minimum of one: player 0
    down to two has used half its room, player 1 still on two none of it."""
    spec = rps_spec(reach=None, extinction={"pieces": "any", "min_count": 1, "result": "loss"})
    spec["setup"]["placements"] = spec["setup"]["placements"][:5]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine)
    assert evaluator._extinction_counts == (3, 2)
    state = engine.make_state({"a1": ("R", 0), "b1": ("P", 0), "e5": ("R", 1), "d5": ("P", 1)})
    assert evaluator._shortfall(state.board, 0) == 0.5
    assert evaluator._shortfall(state.board, 1) == 0.0


def test_a_spec_with_no_scoring_rule_scores_nothing() -> None:
    engine = rps()
    evaluator = Evaluator.for_engine(engine)
    assert evaluator.score_for(engine.initial_state(), 0) == 0.0
    assert evaluator.score_for(engine.initial_state(), 1) == 0.0


# ------------------------------------------------------------------------ the race


def _race_position(engine: Engine) -> GameState:
    # Player 0's paper on c3 is two steps from e5; player 1 holds no scissors, so the
    # paper is immune. Player 1's nearest piece to a1 is the rock on a5, four steps.
    return engine.make_state(
        {"a1": ("R", 0), "c3": ("P", 0), "c1": ("S", 0), "a5": ("R", 1), "b5": ("P", 1)}
    )


def test_the_race_and_the_immune_racer_are_signed_by_the_arrivals_result() -> None:
    seen = {}
    for result in ("win", "draw", "loss"):
        engine = rps(reach={"squares": ["e5"], "mirror": "rotate", "result": result})
        evaluator = Evaluator.for_engine(engine)
        state = _race_position(engine)
        seen[result] = (evaluator.immunity(state, 0), evaluator.race_progress(state, 0))
    assert seen == {"win": (1.5, 0.5), "draw": (1.0, 0.0), "loss": (0.5, -0.5)}


# ----------------------------------------------------------------------- immunity


def test_immunity_is_zero_where_the_relation_cannot_make_a_piece_immune() -> None:
    engine = Engine(load_spec(ROOT / "rulesets" / "chess.json"))
    evaluator = Evaluator.for_engine(engine)
    assert evaluator.immunity(engine.initial_state(), 0) == 0.0
    assert evaluator.immunity(engine.initial_state(), 1) == 0.0


def test_a_held_hunter_makes_its_prey_not_immune() -> None:
    engine = rps(drops={"from_captures": True})
    evaluator = Evaluator.for_engine(engine)
    # Player 1 has no scissors on the board: player 0's paper is immune...
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "c1": ("S", 0), "e5": ("R", 1), "d5": ("P", 1)}
    )
    assert evaluator.immunity(state, 0) == 1.0
    # ...until one is in player 1's hand.
    held = state.with_(hands=((0, 0, 0), (0, 0, 1)))
    assert evaluator.immunity(held, 0) == 0.0
    assert evaluator.immunity(held, 1) == 0.0
    # The mirror: player 0 has no scissors on the board, player 1's paper is immune until
    # player 0 holds one.
    mirror = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "e5": ("R", 1), "d5": ("P", 1), "c5": ("S", 1)}
    )
    assert evaluator.immunity(mirror, 1) == 1.0
    assert evaluator.immunity(mirror.with_(hands=((0, 0, 1), (0, 0, 0))), 1) == 0.0


def test_only_the_second_player_holding_an_immune_piece_is_still_counted() -> None:
    """Player 0 has no paper: player 1's rock (type 0, mask 1) is its only immune piece,
    and nothing of player 0's is immune."""
    engine = rps()
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"a1": ("R", 0), "c1": ("S", 0), "e5": ("R", 1), "d5": ("P", 1), "c5": ("S", 1)})
    assert evaluator.immunity(state, 1) == 1.0
    assert evaluator.immunity(state, 0) == -1.0


def test_a_colouring_that_wins_at_one_square_is_all_the_way_there() -> None:
    engine = rps(reach=None, territory={"mode": "permanent", "win_at": 1})
    evaluator = Evaluator.for_engine(engine)
    state = engine.initial_state()
    painted = state.with_(colour=b"\x01" + b"\x00" * 24)
    assert evaluator.goal_progress(painted, 0) == 1.0
    assert evaluator.goal_progress(painted, 1) == -1.0


def test_a_counted_arrival_is_half_of_a_count_of_two() -> None:
    """A safe race counted to two arrivals of rocks or papers: player 0's paper standing on
    e5 is half of a win; player 1's paper on player 0's goal is nothing of player 0's."""
    engine = rps(reach={"squares": ["e5"], "mirror": "rotate", "safe": True, "count": 2,
                        "piece": ["R", "P"]})
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"a1": ("R", 0), "e5": ("P", 0), "c5": ("S", 1), "d5": ("P", 1)})
    assert evaluation.reach_tally(engine.spec, state, 0) == 1.0
    assert evaluator.goal_progress(state, 0) == 0.5
    theirs = engine.make_state({"a1": ("R", 0), "c1": ("S", 0), "e5": ("P", 1), "d5": ("R", 1)})
    assert evaluation.reach_tally(engine.spec, theirs, 0) == 0.0
    # Counted to a draw, the same arrival is worth nothing either way.
    drawn = rps(reach={"squares": ["e5"], "mirror": "rotate", "safe": True, "count": 2,
                       "piece": ["R", "P"], "result": "draw"})
    assert Evaluator.for_engine(drawn).goal_progress(state, 0) == 0.0


def test_a_shortfall_past_the_threshold_is_one() -> None:
    """Three counted pieces and a minimum of one: two may go; all three gone is a whole
    shortfall, not one and a half."""
    engine = rps(reach=None, extinction={"pieces": ["R", "P", "S"], "min_count": 1, "result": "loss"})
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"e5": ("R", 1), "d5": ("P", 1), "c5": ("S", 1)})
    assert evaluator._shortfall(state.board, 0) == 1.0
    assert evaluator._shortfall(state.board, 1) == 0.0


def test_a_pieces_count_scores_one_a_piece() -> None:
    engine = rps(scoring={"count": "pieces"})
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"a1": ("R", 0), "b1": ("P", 0), "c1": ("S", 0), "e5": ("R", 1)})
    assert evaluator.score_for(state, 0) == 3.0
    assert evaluator.score_for(state, 1) == 1.0


def test_capture_reach_is_read_from_the_first_frame() -> None:
    """A turning piece whose only capture is three squares forward, on a board three high:
    facing forward it can take nothing anywhere; turned, it could."""
    spec = {
        "schema_version": 31, "name": "lancer", "description": "A test game.",
        "board": {"width": 6, "height": 3},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
            {"id": "L", "name": "lancer", "value": 3,
             "moves": [{"kind": "leap", "vectors": [[0, 3]], "mode": "capture"},
                       {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}],
             "facing": {"axes": 4, "initial": "forward"}},
            {"id": "G", "name": "gunner", "value": 3,
             "moves": [{"kind": "slide", "vectors": [[0, 1]], "min_range": 3, "mode": "capture"},
                       {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}],
             "facing": {"axes": 4, "initial": "forward"}},
        ],
        "setup": {"placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "L", "squares": ["c1"]},
                                 {"piece": "G", "squares": ["d1"]}]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 100},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    compiled = Engine(spec).spec
    assert evaluation.has_capture_reach(compiled, 1) is False
    assert evaluation.has_capture_reach(compiled, 2) is False
    assert evaluation.has_capture_reach(compiled, 0) is True
    # The measured values agree: neither turning piece takes anything facing forward.
    assert _measured_values(compiled) == (94 / 18, 0.0, 0.0)


def test_mean_magnitude_counts_each_non_royal_type_once() -> None:
    assert evaluation.mean_abs_non_royal((1.0, -3.0, 0.0), (2,)) == 2.0
    assert evaluation.mean_abs_non_royal((5.0,), (0,)) == 1.0


def test_the_default_evaluator_prices_a_move_at_five_hundredths() -> None:
    """The opening of the cycle game: material level, so the score is the mobility term
    alone, 0.05 a move."""
    engine = rps()
    evaluator = Evaluator.for_engine(engine)
    state = engine.initial_state()
    moves = len(engine.legal_moves(state))
    assert moves == 9
    assert evaluator.mobility_weight == 0.05
    assert evaluator._score(state, 0, moves) - evaluator._score(state, 0, 0) == 0.05 * 9


def test_the_cheapest_piece_is_priced_by_what_it_may_still_take() -> None:
    """At the opening each type may take one of the enemy's three prey: a third of a piece."""
    engine = rps()
    assert Evaluator.for_engine(engine).cheapest_piece(engine.initial_state().board) == 1 / 3


def test_a_table_with_nothing_priced_above_zero_falls_back_to_one_piece() -> None:
    engine = Engine(load_spec(ROOT / "rulesets" / "chess.json"))
    evaluator = Evaluator.for_engine(engine)
    unpriced = replace(evaluator, values=(0.0,) * engine.spec.n_types)
    assert unpriced.cheapest_piece(engine.initial_state().board) == 1.0


def test_an_empty_slot_in_the_stock_is_no_royal() -> None:
    """A game whose stock holds only stones: with player 1's royal gone from the opening
    board, the stock's zero for the royal type is not a royal."""
    spec = Engine(load_spec(ROOT / "archive" / "three-stones.json")).spec
    assert spec.drops is not None
    stock = spec.drops.stock_of(1)
    assert all(royal < len(stock) and stock[royal] == 0 for royal in spec.royal_types)
    stride = spec.n_facings * spec.n_types
    board = bytes(
        0 if value and (value - 1) // stride == 1 and spec.type_of(value) in spec.royal_types
        else value
        for value in spec.initial_board
    )
    headless = replace(spec, initial_board=board)
    assert _royal_loss_ends_it(headless, 0) is True
    assert _royal_loss_ends_it(headless, 1) is False


def test_a_side_without_a_type_does_not_hunt_with_it() -> None:
    """Player 0 holds no rock: player 1's scissors are immune, and nothing else is."""
    engine = rps()
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"b1": ("P", 0), "c1": ("S", 0), "e5": ("R", 1), "d5": ("P", 1), "c5": ("S", 1)})
    assert evaluator.immunity(state, 1) == 1.0


def test_hop_capture_reach_is_read_from_the_first_frame() -> None:
    """A turning cannon that hops two squares at a time forward, on a board two high: facing
    forward it has no step at all; turned, it would."""
    spec = {
        "schema_version": 31, "name": "short cannon", "description": "A test game.",
        "board": {"width": 6, "height": 2},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
            {"id": "C", "name": "cannon", "value": 3,
             "moves": [{"kind": "hop", "vectors": [[0, 2]], "mode": "capture"},
                       {"kind": "leap", "vectors": [[1, 0]], "mode": "move"}],
             "facing": {"axes": 4, "initial": "forward"}},
        ],
        "setup": {"placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "C", "squares": ["c1"]}]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 100},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    compiled = Engine(spec).spec
    assert evaluation.has_capture_reach(compiled, 1) is False
    # The measured values agree: a king on 6x2 reaches 4*3 + 8*5 = 52 squares in all, and
    # the cannon facing forward takes nothing anywhere (turned, it would).
    assert _measured_values(compiled) == (52 / 12, 0.0)


def test_sixty_four_goal_squares_a_side_can_still_be_sealed() -> None:
    """A 9x8 board whose first 64 squares are all goals (mirrored to the last 64): at the
    limit the seal is defined; one more square and it is not."""
    files = "abcdefghi"
    names = [f"{files[i % 9]}{i // 9 + 1}" for i in range(65)]
    spec = rps_spec(reach={"squares": names[:64], "mirror": "rotate", "result": "win"})
    spec["board"] = {"width": 9, "height": 8}
    spec["setup"]["placements"] = [
        {"piece": "R", "player": 0, "squares": ["a8"]},
        {"piece": "R", "player": 1, "squares": ["i1"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    goals = seal_goals(Engine(spec).spec)
    assert goals is not None
    assert (len(goals[0]), len(goals[1])) == (64, 64)
    spec["rules"]["reach"]["squares"] = names
    assert validate_spec(spec) == [], validate_spec(spec)
    assert seal_goals(Engine(spec).spec) is None


def test_immunity_is_undefined_beside_every_rule_that_removes_by_another_route() -> None:
    table = immunity_threats(rps().spec)
    assert table == (((4, 1, 2), (4, 1, 2)), 7)
    found = {name: immunity_threats(engine.spec) for name, engine in variants().items()}
    assert found == {"explosion": None, "strike": None, "wild": None, "drops": table}


def test_a_seal_is_undefined_beside_every_rule_that_can_bring_a_new_captor() -> None:
    assert seal_goals(rps().spec) == ((24,), (0,))
    found = {name: seal_goals(engine.spec) for name, engine in variants().items()}
    assert found == {"explosion": None, "strike": None, "wild": None, "drops": None}


def test_a_threat_follows_promotion_and_only_the_promoting_type() -> None:
    """A minnow moves only, and promotes to scissors on the last rank: its threat is the
    scissors', and no other type's threat grows by it."""
    minnow = {"id": "M", "name": "minnow", "value": 1, "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]}
    spec = rps_spec((minnow,), promotion={"piece": "M", "rank": "last", "to": ["S"]})
    spec["setup"]["placements"].append({"piece": "M", "player": 0, "squares": ["d1"]})
    spec["setup"]["placements"].append({"piece": "M", "player": 1, "squares": ["b5"]})
    assert validate_spec(spec) == [], validate_spec(spec)
    threats = capture_threats(Engine(spec).spec)
    rock, paper, scissors = 1 << 0, 1 << 1, 1 << 2
    row = (scissors, rock, paper, paper)
    assert threats == (row, row)


# --------------------------------------------------------------- measured values


def test_a_turning_sweepers_reach_is_read_from_its_first_frame() -> None:
    """On a 6x3 board a forward sweep reaches 18 squares in all, a sideways one 45: the
    value is the forward frame's, not a turned one's."""
    spec = {
        "schema_version": 31, "name": "sweeper", "description": "A test game.",
        "board": {"width": 6, "height": 3},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
            {"id": "W", "name": "sweeper", "value": 3,
             "moves": [{"kind": "sweep", "vectors": [[0, 1]], "mode": "both"}],
             "facing": {"axes": 4, "initial": "forward"}},
        ],
        "setup": {"placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "W", "squares": ["c1"]}]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 100},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    assert _measured_values(Engine(spec).spec) == (94 / 18, 1.0)


# ------------------------------------------------------------ a royal in the stock


def test_a_royal_loss_ends_it_reads_the_stock_and_the_board() -> None:
    placement = Engine(load_spec(ROOT / "archive" / "placement-chess.json")).spec
    assert _royal_loss_ends_it(placement, 0) is True
    assert _royal_loss_ends_it(placement, 1) is True
    chess = Engine(load_spec(ROOT / "rulesets" / "chess.json")).spec
    assert _royal_loss_ends_it(chess, 0) is True
    # The same tables with player 1's royal gone from the opening board.
    stride = chess.n_facings * chess.n_types
    board = bytes(
        0 if value and (value - 1) // stride == 1 and chess.type_of(value) in chess.royal_types
        else value
        for value in chess.initial_board
    )
    headless = replace(chess, initial_board=board)
    assert _royal_loss_ends_it(headless, 0) is True
    assert _royal_loss_ends_it(headless, 1) is False


# -------------------------------------------------------------------- rounds


def _placed(
    engine: Engine,
    pieces_at: dict[str, tuple[str, int]],
    rest: tuple[tuple[tuple[str, int], ...], ...] = ((), ()),
) -> GameState:
    rested = tuple(
        tuple(sorted((engine._square_index(name), left) for name, left in side)) for side in rest
    )
    state = engine.make_state(pieces_at).with_(rest=rested)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def _lists(engine: Engine, state: GameState) -> tuple[list[int], list[int]]:
    return (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))


def test_a_resting_royal_falls_among_other_pieces_and_not_beside_a_second_royal() -> None:
    """Rests up to 2: player 1's king resting on b4 is reached by player 0's standing bishop,
    with player 1's pawn and resting rook and player 0's king on the board."""
    engine = Engine(load_spec(ROOT / "tests" / "_retired_breather_by_weight.json"))
    at = {"e1": ("K", 0), "d2": ("B", 0), "a1": ("R", 0),
          "b4": ("K", 1), "e4": ("P", 1), "a5": ("R", 1)}
    state = _placed(engine, at, rest=((), (("a5", 2), ("b4", 1))))
    assert rested_royal_falls(engine.spec, state, 1, _lists(engine, state)) is True
    assert rested_royal_falls(engine.spec, state, 0, _lists(engine, state)) is False


# Player 1's king (speed 0) on a5, the a-file open to player 0's rook (speed 1) on a1.
OPEN = {"e1": ("K", 0), "a1": ("R", 0), "a5": ("K", 1), "e4": ("P", 1)}


def _ranked(speeds: dict[str, int]) -> Engine:
    spec = speed_gardner(speeds)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _falls(engine: Engine, state: GameState, side: int) -> bool:
    sim = engine.spec.simultaneous
    assert sim is not None and sim.speed
    return caught_royal_falls(engine.spec, state, side, _lists(engine, state), sim.speed)


def test_a_caught_royal_is_answered_only_by_a_strictly_faster_capture_of_its_attacker() -> None:
    engine = _ranked({"R": 1, "N": 2})
    assert _falls(engine, _placed(engine, OPEN), 1) is True
    # A knight of player 1's faster than the rook, moving elsewhere, answers nothing.
    assert _falls(engine, _placed(engine, {**OPEN, "e5": ("N", 1)}), 1) is True
    # The same knight on b3 takes the rook first.
    assert _falls(engine, _placed(engine, {**OPEN, "b3": ("N", 1)}), 1) is False
    # A knight only as fast as the rook does not.
    level = _ranked({"R": 1, "N": 1})
    assert _falls(level, _placed(level, {**OPEN, "b3": ("N", 1)}), 1) is True


def test_a_royal_whose_loss_ends_nothing_is_never_caught() -> None:
    engine = _ranked({"R": 1, "N": 2})
    state = _placed(engine, OPEN)
    sim = engine.spec.simultaneous
    assert sim is not None
    endless = replace(engine.spec, win="annihilation")
    assert caught_royal_falls(endless, state, 1, _lists(engine, state), sim.speed) is False
    # An empty hand under drops holds no second royal: the rook still catches the king.
    spec = speed_gardner({"R": 1, "N": 2}, drops={"from_captures": True})
    assert validate_spec(spec) == [], validate_spec(spec)
    dropping = Engine(spec)
    open_state = _placed(dropping, OPEN)
    assert any(len(hand) for hand in open_state.hands)
    assert _falls(dropping, open_state, 1) is True
    # A slower attacker of the royal (player 0's king beside it) is passed over, and the
    # rook along rank 5 is still found.
    slow = _placed(engine, {"b4": ("K", 0), "e5": ("R", 0), "a5": ("K", 1), "e3": ("P", 1)})
    assert _falls(engine, slow, 1) is True
    # And a side with no royal on the board has none to lose.
    kingless = _placed(engine, {"e1": ("K", 0), "a1": ("R", 0), "e4": ("P", 1)})
    assert _falls(engine, kingless, 1) is False


def test_a_round_asks_each_player_for_its_own_moves() -> None:
    """With no lists passed the engine is asked, each side for its own: player 1's faster
    knight answers the rook, so nobody's royal falls and no price is paid."""
    engine = _ranked({"R": 1, "N": 2})
    state = _placed(engine, {**OPEN, "b3": ("N", 1)})
    both = _lists(engine, state)
    counts = (len(both[0]), len(both[1]))
    evaluator = evaluation.Evaluator.for_engine(engine)
    given = evaluator.evaluate_round(engine, state, 0, counts, both)
    assert evaluator.evaluate_round(engine, state, 0, counts) == given
    assert evaluator.evaluate_round(engine, state, 1, counts) == evaluator.evaluate_round(
        engine, state, 1, counts, both
    )


# ------------------------------------------------------------ Q1153's survivors


def test_a_counted_arrival_that_loses_is_minus_half_of_a_count_of_two() -> None:
    """The same arrival counted to a loss: player 0's paper on e5 is half of the way to
    losing, -0.5 and not -1."""
    engine = rps(reach={"squares": ["e5"], "mirror": "rotate", "safe": True, "count": 2,
                        "piece": ["R", "P"], "result": "loss"})
    state = engine.make_state({"a1": ("R", 0), "e5": ("P", 0), "c5": ("S", 1), "d5": ("P", 1)})
    assert Evaluator.for_engine(engine).goal_progress(state, 0) == -0.5
    assert Evaluator.for_engine(engine).goal_progress(state, 1) == 0.5


def test_sweep_capture_reach_is_read_from_the_first_frame() -> None:
    """A turning sweeper whose only capture is three squares forward, on a board three high:
    facing forward it can take nothing anywhere; turned sideways on six files, it could."""
    spec = {
        "schema_version": 31, "name": "long sweeper", "description": "A test game.",
        "board": {"width": 6, "height": 3},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
            {"id": "W", "name": "sweeper", "value": 3,
             "moves": [{"kind": "sweep", "vectors": [[0, 3]], "mode": "capture"},
                       {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}],
             "facing": {"axes": 4, "initial": "forward"}},
        ],
        "setup": {"placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "W", "squares": ["c1"]}]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 100},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    compiled = Engine(spec).spec
    assert evaluation.has_capture_reach(compiled, 1) is False
    assert evaluation.has_capture_reach(compiled, 0) is True


def test_a_royal_first_in_the_table_is_not_counted_as_material() -> None:
    """A royal as type 0 beside the cycle: the counted mask is the three cycle types only."""
    spec = rps_spec(reach=None, win="royal_capture")
    royal = {"id": "K", "name": "king", "royal": True, "value": 0, "captures": [],
             "moves": STEP}
    spec["pieces"].insert(0, royal)
    spec["setup"]["placements"] += [
        {"piece": "K", "player": 0, "squares": ["e1"]},
        {"piece": "K", "player": 1, "squares": ["a5"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    table = immunity_threats(Engine(spec).spec)
    assert table is not None
    assert table[1] == 0b1110


def test_declared_values_all_zero_fall_back_to_a_declared_scale_of_one() -> None:
    """Every declared value zero: nothing to scale by, so the declared half adds nothing
    and each value is the measured half alone."""
    spec = rps_spec()
    for piece in spec["pieces"]:
        piece["value"] = 0
    assert validate_spec(spec) == [], validate_spec(spec)
    values = Evaluator.for_engine(Engine(spec)).values
    expected = 1.0 - evaluation.DECLARED_WEIGHT
    assert values == (expected, expected, expected)


# ------------------------------------------------------------ Q1153's ninth queue


def test_piece_values_are_frozen_and_hashable() -> None:
    """The value pair is a frozen record: a field cannot be reassigned, and it hashes."""
    values = evaluation.PieceValues.derived_only((1.0, 2.0))
    try:
        values.derived = (3.0,)  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("PieceValues accepted an assignment")
    assert hash(values) == hash(evaluation.PieceValues.derived_only((1.0, 2.0)))


def test_the_prey_pool_is_derived_and_left_out_of_the_repr() -> None:
    evaluator = Evaluator.for_spec(rps().spec)
    assert evaluator.prey == (True, True, True)
    assert "prey=" not in repr(evaluator)


def test_an_evaluator_built_from_the_tables_prices_a_move_at_five_hundredths() -> None:
    assert Evaluator.for_spec(rps().spec).mobility_weight == 0.05


def test_a_colouring_won_at_one_square_is_a_goal_term() -> None:
    assert has_goal_terms(rps(reach=None, territory={"mode": "permanent", "win_at": 1}).spec)


def test_each_player_hunts_with_what_its_own_promotion_makes() -> None:
    """A minnow promotes to scissors for player 0 and to paper for player 1: each player's
    minnow threatens what its own promotion takes, so the two rows differ."""
    minnow = {"id": "M", "name": "minnow", "value": 1, "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]}
    spec = rps_spec((minnow,), promotion={"piece": "M", "rank": "last",
                                          "to_by_player": {"0": ["S"], "1": ["P"]}})
    spec["setup"]["placements"].append({"piece": "M", "player": 0, "squares": ["d1"]})
    spec["setup"]["placements"].append({"piece": "M", "player": 1, "squares": ["b5"]})
    assert validate_spec(spec) == [], validate_spec(spec)
    threats = capture_threats(Engine(spec).spec)
    rock, paper, scissors = 1 << 0, 1 << 1, 1 << 2
    assert threats == ((scissors, rock, paper, paper), (scissors, rock, paper, rock))


def test_a_square_holding_the_other_sides_royal_is_not_this_sides_resting_royal() -> None:
    """Player 1's king rests on b4 and player 0 has no king on the board. Asked about player
    0 at b4, with lists in which a move lands on b4, the answer is no: the royal there is
    not player 0's."""
    engine = Engine(load_spec(ROOT / "tests" / "_retired_breather_by_weight.json"))
    at = {"d2": ("B", 0), "a1": ("R", 0), "b4": ("K", 1), "e4": ("P", 1), "a5": ("R", 1)}
    state = _placed(engine, at, rest=((), (("b4", 1),)))
    mine, theirs = _lists(engine, state)
    b4 = engine._square_index("b4")
    swapped = (theirs, mine)
    assert evaluation._royal_resting_falls(engine.spec, state, 1, b4, (mine, theirs)) is True
    assert evaluation._royal_resting_falls(engine.spec, state, 0, b4, swapped) is False


# ------------------------------------------------------------ Q1153's tenth queue


def test_the_immunity_term_is_one_piece_by_default() -> None:
    assert evaluation.IMMUNE_PIECES == 1.0
    assert Evaluator.for_engine(rps()).immune_pieces == 1.0


def test_the_cheapest_piece_skips_a_type_priced_at_zero() -> None:
    """The rock priced at nothing: the cheapest piece is the paper's or scissors' third, not
    the rock's zero."""
    engine = rps()
    unpriced = replace(Evaluator.for_engine(engine), values=(0.0, 1.0, 1.0))
    assert unpriced.cheapest_piece(engine.initial_state().board) == 1 / 3


def test_a_shortfall_with_no_room_above_the_minimum_is_none() -> None:
    """Player 1 starts on two pieces against a minimum of two: it has no room to lose, so it
    has used none of it; player 0, three against two, has used all of its one."""
    spec = rps_spec(reach=None, extinction={"pieces": "any", "min_count": 2, "result": "loss"})
    spec["setup"]["placements"] = spec["setup"]["placements"][:5]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state({"a1": ("R", 0), "e5": ("R", 1), "d5": ("P", 1)})
    assert evaluator._extinction_counts == (3, 2)
    assert evaluator._shortfall(state.board, 0) == 1.0
    assert evaluator._shortfall(state.board, 1) == 0.0


# ------------------------------------------------------------ Q1153's eleventh queue


def test_no_immune_piece_is_zero_before_any_division_by_the_mean() -> None:
    """At the cycle game's opening each side holds the hunter of every type, so nothing is
    immune and the term is zero before it divides by the mean piece -- which, on a table
    priced at nothing, is zero itself."""
    engine = rps()
    unpriced = replace(Evaluator.for_engine(engine), values=(0.0,) * engine.spec.n_types)
    assert immunity_threats(engine.spec) is not None
    assert unpriced.immunity(engine.initial_state(), 0) == 0.0
    assert unpriced.immunity(engine.initial_state(), 1) == 0.0


def test_an_immune_racer_past_the_diameter_races_for_nothing_rather_than_less() -> None:
    """9x3, goals a3 and i3 for player 0 (rotated to i1 and a1), a diameter of 4. Each side
    holds one paper and nobody a scissors, so both are immune. Player 1's paper seals a3, so
    player 0's paper on b1 is seven steps from its one open goal: past the diameter, a race
    of zero, not a negative one. Player 1's is two from a1: (4 - 2) / 4 = 1/2. Equal immune
    material, so the term is -1/2."""
    spec = rps_spec(reach={"squares": ["a3", "i3"], "mirror": "rotate", "result": "win"})
    spec["board"] = {"width": 9, "height": 3}
    spec["setup"]["placements"] = [
        {"piece": "P", "player": 0, "squares": ["b1"]},
        {"piece": "P", "player": 1, "squares": ["a3"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine)
    assert engine.spec.goal_diameter == 4
    state = engine.initial_state()
    assert evaluator.sealed_goals(state)[0] != 0
    assert evaluator.immunity(state, 0) == -0.5
    assert evaluator.immunity(state, 1) == 0.5


# ------------------------------------------------------------ Q1153's twelfth queue


def _royal_first_spec(**rules: Any) -> dict[str, Any]:
    """The cycle game with a royal as type 0, one a side, won by taking it."""
    spec = rps_spec(reach=None, win="royal_capture", **rules)
    royal = {"id": "K", "name": "king", "royal": True, "value": 0, "captures": [],
             "moves": STEP}
    spec["pieces"].insert(0, royal)
    spec["setup"]["placements"] += [
        {"piece": "K", "player": 0, "squares": ["e1"]},
        {"piece": "K", "player": 1, "squares": ["a5"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def test_an_inexhaustible_type_first_in_the_table_leaves_the_types_after_it_a_value() -> None:
    """Type 0 in an unlimited stock is priced at nothing; the two after it still priced."""
    spec = rps_spec(drops={"from_captures": False, "stock": {"R": "unlimited"}})
    assert validate_spec(spec) == [], validate_spec(spec)
    values = Evaluator.for_engine(Engine(spec)).values
    assert len(values) == 3
    assert values[0] == evaluation.INEXHAUSTIBLE_VALUE
    assert values[1] == values[2] > 0.0


def test_a_royal_first_in_the_table_with_an_empty_stock_is_read_off_the_board() -> None:
    """Drops but no stock: the stock is empty, so royal type 0 is past its end and the
    answer comes off the opening board."""
    spec = Engine(_royal_first_spec(drops={"from_captures": True})).spec
    assert spec.drops is not None
    assert spec.drops.stock_of(0) == ()
    assert _royal_loss_ends_it(spec, 0) is True
    assert _royal_loss_ends_it(spec, 1) is True


def test_a_hand_is_summed_per_side_from_zero_to_the_last_bit() -> None:
    """One piece in player 0's hand at a value of 0.1: exactly 0.1 and -0.1, as
    ``eval.rs::hand_value`` adds it (a side starting at anything but zero rounds it off)."""
    engine = rps(drops={"from_captures": True})
    evaluator = replace(Evaluator.for_engine(engine), values=(0.1, 0.2, 0.3))
    hands = ((1, 0, 0), (0, 0, 0))
    assert evaluator.hand_value(hands, 0) == 0.1
    assert evaluator.hand_value(hands, 1) == -0.1


def test_immunity_switched_off_is_never_computed() -> None:
    """Immunity weighted at zero on a table priced at nothing: the term would divide an
    immune piece by a mean of zero, so it must not be evaluated at all; the score is zero."""
    engine = rps(reach=None)
    evaluator = replace(
        Evaluator.for_engine(engine), values=(0.0, 0.0, 0.0), immune_pieces=0.0
    )
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "c1": ("S", 0), "e5": ("R", 1), "c5": ("S", 1)}
    )
    assert immunity_threats(engine.spec) is not None
    assert evaluator._score(state, 0, 0) == 0.0


def test_a_region_count_scores_one_for_each_own_piece_standing_in_it() -> None:
    """Region {a1, c1}, reflected for the second player (c1 lands on c5 either way): two of
    player 0's three pieces stand in it, one of player 1's two, so 2.0 and 1.0 exactly."""
    engine = rps(scoring={"count": "region", "region": ["a1", "c1"]})
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "c1": ("S", 0), "c5": ("S", 1), "b5": ("P", 1)}
    )
    assert evaluator.score_for(state, 0) == 2.0
    assert evaluator.score_for(state, 1) == 1.0


# ------------------------------------------------------------ Q1153's fourteenth queue


def test_an_override_worth_nothing_off_the_royals_is_refused() -> None:
    """Every non-royal entry zero is no opinion at all, and is refused; one non-zero entry is
    an opinion and is kept."""
    from chess_variant_sandbox.agents.evaluation import PieceValues

    derived = (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="every non-royal piece is worth nothing"):
        PieceValues.of(derived, (0.0, 0.0, 0.0), ())
    with pytest.raises(ValueError, match="every non-royal piece is worth nothing"):
        PieceValues.of(derived, (0.0, 0.0, 0.0), (0,))
    kept = PieceValues.of(derived, (0.0, 4.0, 0.0), ())
    assert kept.derived == derived


def test_an_evaluator_is_frozen() -> None:
    evaluator = Evaluator.for_engine(rps())
    with pytest.raises(FrozenInstanceError):
        evaluator.mobility_weight = 1.0  # type: ignore[misc]
    assert hash(evaluator) == hash(evaluator)


def test_running_out_as_a_loss_counts_half_the_way_as_minus_a_half() -> None:
    """The whole army counted, three pieces a side and one left at the threshold: player 0
    one piece down is half of the way to losing, -0.5 and not -1."""
    engine = rps(reach=None, extinction={"pieces": "any", "result": "loss"})
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "e5": ("R", 1), "d5": ("P", 1), "c5": ("S", 1)}
    )
    assert evaluator._shortfall(state.board, 0) == 0.5
    assert evaluator.goal_progress(state, 0) == -0.5
    assert evaluator.goal_progress(state, 1) == 0.5


def test_a_count_past_the_clock_mid_turn_is_worth_the_whole_lead_and_no_more() -> None:
    """Three moves a turn and a clock of 100: a turn starts at 99, so ply 101 is a mid-turn
    position past the clock. Four pieces to two is a lead of a third, and the ramp stops at
    one, so the term is that third and not a hundredth more."""
    engine = rps(reach=None, scoring={"count": "pieces"}, turn={"moves": 3}, max_plies=100)
    evaluator = Evaluator.for_engine(engine)
    placed = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "c1": ("S", 0), "a2": ("R", 0), "e5": ("R", 1),
         "d5": ("P", 1)}
    )
    assert evaluator.score_for(placed, 0) == 4.0
    assert evaluator.score_for(placed, 1) == 2.0
    at_clock = placed.with_(ply=100)
    past = placed.with_(ply=101)
    assert evaluator.goal_progress(at_clock, 0) == 1 / 3
    assert evaluator.goal_progress(past, 0) == 1 / 3
    assert evaluator.goal_progress(past, 1) == -1 / 3


def test_a_threshold_at_the_opening_count_has_no_shortfall() -> None:
    """One counted piece a side and a threshold of one: there is no room to run short, so the
    shortfall is zero whatever the board holds -- a second one included."""
    engine = rps(reach=None, extinction={"pieces": ["S"], "result": "loss"})
    evaluator = Evaluator.for_engine(engine)
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("S", 0), "c1": ("S", 0), "e5": ("R", 1), "c5": ("S", 1)}
    )
    assert evaluator._shortfall(state.board, 0) == 0.0
    assert evaluator._shortfall(state.board, 1) == 0.0
    assert evaluator.goal_progress(state, 0) == 0.0


def test_a_royal_first_in_the_table_with_no_hands_at_all_is_still_caught() -> None:
    """Without drops a state's hands are empty tuples, not a row of zeros: a royal of type 0
    must not index into them (Q1153). The rook on a1 catches player 1's king on a5."""
    spec = speed_gardner({"R": 1, "N": 2})
    spec["pieces"].sort(key=lambda piece: piece["id"] != "K")
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    assert sorted(engine.spec.royal_types) == [0]
    state = _placed(engine, OPEN)
    assert state.hands == ((), ())
    assert _falls(engine, state, 1) is True
    assert _falls(engine, state, 0) is False
