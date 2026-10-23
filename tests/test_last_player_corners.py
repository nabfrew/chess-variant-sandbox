"""F10c (Q752c): the players' and evaluators' last corners, held to the declared oracles.

`docs/plan-tests.md` F10c lists the behaviour lines F10b (`tests/test_player_corners.py`)
left in `eval.rs`, `features.rs`, `spy.rs` and `spy_players.rs`. Each is reached here on an
inline spec built for it, and compared with its Python oracle (CLAUDE.md rule 3) to the bit:

* **the built-in evaluator** (`agents/evaluation.py`): a race over more than 64 goal squares
  a side, where the seal is not defined; a seal beside a race of one kind, where a sealed
  goal is skipped and only the racing kind is read, by the race and by the immunity term;
  a game whose every type is royal, where the mean non-royal piece is no piece;
* **the learned evaluator's terms** (`agents/features.py`): a material score whose opening
  material is below one, an enemy army nothing may take, an empty board, a facing game
  with nothing on the board that turns;
* **the spy** (`tests/spy_oracle.py`, `agents/simple.py`, `agents/minimax.py`): the
  initiative dealt to the second player, a round player with one move as the announcer and
  as the answerer, and greedy's turn into a round that is over before anyone chooses.

The lines F10c lists that no test can reach are recorded in `docs/plan-tests.md` F10c with
the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from test_backends import _oriented  # noqa: E402
from test_feature_terms import _assert_valued_alike, _heuristic  # noqa: E402
from test_player_teacher import _oracle_value as minimax_value  # noqa: E402
from test_player_teacher import _values as minimax_values  # noqa: E402
from test_spy_players import _choose  # noqa: E402

from chess_variant_sandbox.agents import build_agent, native_spec  # noqa: E402
from chess_variant_sandbox.agents.evaluation import seal_goals  # noqa: E402
from chess_variant_sandbox.agents.features import VOCABULARY  # noqa: E402
from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.engine.engine import _pack  # noqa: E402
from chess_variant_sandbox.engine.moves import decode_move  # noqa: E402
from chess_variant_sandbox.engine.state import GameState  # noqa: E402
from chess_variant_sandbox.spec import load_spec, validate_spec  # noqa: E402
from chess_variant_sandbox.tournament.play import eval_values, seat_rng  # noqa: E402
from tests import spy_oracle as oracle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

STEP = [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both"}]


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == []
    return Engine(spec)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _place(engine: Engine, white: dict[str, str], black: dict[str, str],
           side: int = 0) -> GameState:
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    return engine.make_state(placement, side_to_move=side)


def _rekeyed(engine: Engine, state: GameState) -> GameState:
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


# -- the built-in evaluator ------------------------------------------------------------------


def _rps(width: int, height: int, goals: list[str], piece: str | None = None,
         extra: tuple[dict[str, Any], ...] = ()) -> dict[str, Any]:
    """Rock, paper and scissors stepping like kings, each taking only the one it beats
    (`rulesets/rps-v3.json`'s capture matrix), racing to ``goals`` (mirrored) by ``piece``
    or by any piece."""
    reach: dict[str, Any] = {"squares": goals, "mirror": True, "result": "win"}
    if piece is not None:
        reach["piece"] = piece
    return {
        "schema_version": 31, "name": "rps race (F10c test spec)",
        "board": {"width": width, "height": height},
        "pieces": [
            {"id": "R", "value": 1, "captures": ["S"], "moves": STEP},
            {"id": "P", "value": 1, "captures": ["R"], "moves": STEP},
            {"id": "S", "value": 1, "captures": ["P"], "moves": STEP},
            *extra,
        ],
        "setup": {"mirror": True, "placements": [{"piece": "R", "squares": ["a1"]},
                                                 {"piece": "P", "squares": ["b1"]},
                                                 {"piece": "S", "squares": ["c1"]}]},
        "rules": {"win": "annihilation", "stalemate": "draw", "max_plies": 100,
                  "reach": reach},
    }


# A kind that steps like a king, takes nothing and that no kind may take.
WALL = {"id": "W", "value": 1, "captures": [], "moves": STEP}


def _teacher_agrees(engine: Engine, states: list[GameState], agent: str = "minimax:nodes=120",
                    first_seed: int = 11) -> None:
    """`Core.search_values` against the oracle's deepening, value, move and depth to the
    bit, and every position valued."""
    seeds = [first_seed + index for index in range(len(states))]
    native = minimax_values(engine, states, agent, seeds)
    assert native == [minimax_value(engine, agent, state, seed)
                      for state, seed in zip(states, seeds, strict=True)]
    assert all(answer is not None for answer in native)


def test_a_race_over_more_than_64_goals_a_side_has_no_seal() -> None:
    """`eval.rs` 496. An 11 x 14 board whose last six ranks are White's goals (66 squares,
    Black's the first six, two ranks between): past the 64 a position's seal mask holds,
    so no goal is ever sealed. Black's walls, a kind nothing takes and that takes nothing,
    stand on a9, b9 and c9, the goals nearest White's rock on a8: sealed, they would put
    every square the rock can reach a step or two further from its nearest open goal. The
    oracle has no seal here (`seal_goals` is `None`), and the core values the race, the
    immunity term's included, as it does."""
    goals = [file + str(rank) for rank in range(9, 15) for file in "abcdefghijk"]
    engine = _engine(_rps(11, 14, goals, extra=(WALL,)))
    assert seal_goals(engine.spec) is None
    walls = {"a9": "W", "b9": "W", "c9": "W", "k14": "S"}
    states = [
        _place(engine, {"a8": "R"}, walls),
        _place(engine, {"b7": "R", "k1": "S"}, walls),
        _place(engine, {"f3": "R"}, {"f10": "P", "a14": "S"}),
    ]
    assert all(engine.terminal(state) is None for state in states)
    _teacher_agrees(engine, states)


def test_a_sealed_goal_beside_a_race_of_one_kind_is_the_oracles() -> None:
    """`eval.rs` 629 and 726. Rocks race to the last rank (9 x 9). A piece of Black's that
    White cannot take stands on one of White's nine goals, sealing it: a rock (White holds
    a rock and scissors, neither of which takes one) or a wall (a kind nothing takes and
    that takes nothing). The race and the immunity term read the open goals only, for the
    rock only: White's scissors is passed over, though beside walls it is immune too and
    nearer the goals than the rock."""
    engine = _engine(_rps(9, 9, [file + "9" for file in "abcdefghi"], piece="R",
                          extra=(WALL,)))
    assert seal_goals(engine.spec) is not None
    states = [
        _place(engine, {"d4": "R", "f3": "S"}, {"c9": "R", "h7": "S"}, side=side)
        for side in (0, 1)
    ] + [
        _place(engine, {"d4": "R", "f7": "S"}, {"c9": "W", "h8": "W"}, side=side)
        for side in (0, 1)
    ] + [_place(engine, {"d6": "R", "b2": "S"}, {"c9": "R", "g5": "S", "b4": "R"})]
    assert all(engine.terminal(state) is None for state in states)
    _teacher_agrees(engine, states)


def _all_royal() -> dict[str, Any]:
    """Two royal kinds a side and nothing else, a race to the far rank by any piece: the mean
    non-royal piece, the unit the goal and round terms are stated in, is no piece at all."""
    return {
        "schema_version": 31, "name": "all royal (F10c test spec)",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": STEP},
            {"id": "G", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]},
        ],
        "setup": {"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]},
                                                 {"piece": "G", "squares": ["b1"]}]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 60,
                  "reach": {"squares": ["a5", "b5", "c5", "d5", "e5"], "mirror": True,
                            "result": "win"}},
    }


def _all_royal_positions(engine: Engine) -> list[GameState]:
    return [
        engine.initial_state(),
        _place(engine, {"c3": "K", "a1": "G"}, {"c4": "K", "e5": "G"}, side=1),
        _place(engine, {"b4": "K", "d2": "G"}, {"e2": "K", "a3": "G"}),
    ]


def test_a_game_whose_every_type_is_royal_is_valued_as_the_oracle() -> None:
    """`eval.rs` 1269. With no non-royal type the mean is one, so the race still has a
    weight: the built-in evaluator's value of each position is the oracle's."""
    engine = _engine(_all_royal())
    _teacher_agrees(engine, _all_royal_positions(engine))


# -- the learned evaluator's terms -------------------------------------------------------------


EVERY = "every-term (F10c)"


def _every_term() -> str:
    return f"minimax:h={_heuristic((1.0,) * len(VOCABULARY), EVERY)},nodes=1000000,max_depth=1"


def test_a_material_score_under_one_is_scaled_by_one() -> None:
    """`features.rs` 137. Gardner scored by material at the clock, every piece declared at
    0.04: the opening's material is 0.8, under one, so the score terms are the gap itself.
    Every term at once on positions a queen apart."""
    spec = load_spec(ROOT / "rulesets" / "gardner.json")
    spec["name"] = "Gardner scored by light material (F10c test spec)"
    for piece in spec["pieces"]:
        piece["value"] = 0.0 if piece.get("royal") else 0.04
    spec["rules"]["scoring"] = {"count": "material"}
    engine = _engine(spec)
    states = [
        _place(engine, {"a1": "K", "b2": "Q", "c2": "P"}, {"e5": "K"}, side=side)
        for side in (0, 1)
    ] + [_place(engine, {"a1": "K"}, {"e5": "K", "c4": "R"})]
    _assert_valued_alike(engine, _every_term(), states)


def test_an_army_nothing_may_take_reads_no_prey_share() -> None:
    """`features.rs` 337. A fourth kind, a wall that steps like a king and that no kind may
    take, beside rock, paper and scissors: an army of walls is no prey, so the side facing
    it keeps its floor. Every term at once."""
    engine = _engine(_rps(9, 9, [file + "9" for file in "abcdefghi"], extra=(WALL,)))
    states = [
        _place(engine, {"e3": "R", "a1": "S"}, {"e7": "W", "b8": "W"}, side=side)
        for side in (0, 1)
    ]
    assert all(engine.terminal(state) is None for state in states)
    _assert_valued_alike(engine, _every_term(), states)


def _kamikaze() -> dict[str, Any]:
    """A 3 x 4 race to the far rank by one kind that steps forward, a capture removing both
    pieces, and a stocked hand dropped only on the middle ranks (a drop never wins at
    once): the last two pieces on the board can take each other off it, and the game goes
    on by drops."""
    return {
        "schema_version": 31, "name": "kamikaze drops (F10c test spec)",
        "board": {"width": 3, "height": 4},
        "pieces": [{"id": "A", "value": 1,
                    "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "both"}]}],
        "setup": {"mirror": True, "placements": [{"piece": "A", "squares": ["a1"]}]},
        "rules": {"win": "none", "stalemate": "draw", "max_plies": 40,
                  "reach": {"piece": "A", "squares": ["a4", "b4", "c4"], "mirror": True,
                            "result": "win"},
                  "explode_on_capture": {"radius": 0},
                  "drops": {"from_captures": False, "stock": {"A": 1},
                            "region": {"A": ["a2", "b2", "c2", "a3", "b3", "c3"]}}},
    }


def test_an_empty_board_has_no_centre() -> None:
    """`features.rs` 446. White's only move takes Black's last piece and the blast takes
    White's: the board is empty and Black, with a piece in hand, plays on. That position is
    the one leaf, so the root's value is every term read on an empty board."""
    engine = _engine(_kamikaze())
    state = _place(engine, {"a1": "A"}, {"a2": "A"})
    state = _rekeyed(engine, state.with_(hands=((0,), (1,))))
    (move,) = engine.legal_moves(state)
    after = engine.apply(state, move)
    assert not any(after.board) and engine.terminal(after) is None
    _assert_valued_alike(engine, _every_term(), [state])


def test_a_facing_game_with_nothing_that_turns_on_the_board() -> None:
    """`features.rs` 666. `test_backends._oriented` (Gardner whose pawns turn) with no pawn
    left: the facing term has no piece to read, and every term at once is the oracle's."""
    engine = _oriented()
    states = [
        _place(engine, {"a1": "K", "b1": "R"}, {"e5": "K", "d5": "N"}, side=side)
        for side in (0, 1)
    ]
    _assert_valued_alike(engine, _every_term(), states)


# -- the spy -----------------------------------------------------------------------------------


def _spy(**simultaneous: Any) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((HERE / "_spy_small.json").read_text(encoding="utf-8"))
    spec["name"] = "spy small (F10c test spec)"
    spec["rules"]["simultaneous"] = dict(simultaneous)
    return spec


def _stepping_spy() -> Engine:
    """`tests/_spy_small.json` with a king that only steps forward onto an empty square,
    and stalemate lost: positions where a side has one move, or none, are a few pieces."""
    spec = _spy()
    for piece in spec["pieces"]:
        if piece["id"] == "K":
            piece["moves"] = [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]
    spec["rules"]["stalemate"] = "loss"
    return _engine(spec)


def _black_move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    squares = (_sq(engine, origin), _sq(engine, target))
    if engine.phase(state).is_round:
        moves = engine.legal_moves_for(state, 1)
    else:
        moves = engine.legal_moves(state)
    (move,) = [m for m in moves if decode_move(m)[:2] == squares]
    return move


def test_the_initiative_dealt_to_the_second_player_is_keyed() -> None:
    """`spy.rs` 110 (E10, the other way round from `test_spy.py`'s): under `tie:
    "initiative"`, with the token White's, White takes Black's spy and Black's reply turn
    leaves the first announced round, Black announcing: the token is dealt to Black, and
    the key carries it. Each turn is the oracle's, key and all."""
    engine = _engine(_spy(tie="initiative"))
    state = _place(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})
    state = _rekeyed(engine, state.with_(initiative=0))
    take = next(m for m in engine.legal_moves(state)
                if decode_move(m)[:2] == (_sq(engine, "a4"), _sq(engine, "c4")))
    reply = engine.apply(state, take)
    assert reply == oracle.turn(engine, state, take)
    assert engine.phase(reply).name == "reply" and reply.initiative == 0
    retake = _black_move(engine, reply, "d5", "c4")
    after = engine.apply(reply, retake)
    assert after == oracle.turn(engine, reply, retake)
    phase = engine.phase(after)
    assert (phase.name, phase.announcer) == ("announced", 1)
    assert after.initiative == 1
    assert after.key == engine._compute_hash(after)


SPY_PLAYERS = ("greedy", "minimax:nodes=100")


@pytest.mark.parametrize("agent", ["minimax:nodes=100"])
def test_an_announcer_with_one_move_plays_it_without_a_search(agent: str) -> None:
    """`spy_players.rs` 239. White has lost its spy and its king steps only forward: in the
    announced round White has one move, which the announcer plays at once, no node spent,
    as the oracle does."""
    engine = _stepping_spy()
    state = _place(engine, {"c1": "K"}, {"e5": "K", "c5": "S"}, side=1)
    phase = engine.phase(state)
    assert (phase.name, phase.announcer) == ("announced", 0)
    assert len(engine.legal_moves_for(state, 0)) == 1
    move, nodes = _choose(engine, state, 0, agent)
    assert nodes == 0


@pytest.mark.parametrize("agent", SPY_PLAYERS)
def test_an_answerer_with_one_move_plays_it_without_a_search(agent: str) -> None:
    """`spy_players.rs` 309 (minimax) and 417 (greedy). Black holds its spy, boxed in by its
    own king and a white knight, and its king has one step: whatever White announces, the
    answer is that step, no node spent, as the oracle's."""
    engine = _stepping_spy()
    state = _place(engine, {"c1": "K", "e4": "N"}, {"d5": "K", "e5": "S"}, side=1)
    phase = engine.phase(state)
    assert (phase.name, phase.announcer) == ("announced", 0)
    (only,) = engine.legal_moves_for(state, 1)
    for announced in engine.legal_moves_for(state, 0)[:3]:
        move, nodes = _choose(engine, state, 1, agent, announced)
        assert move == only and nodes == 0


def test_greedy_turns_into_a_round_that_is_over_before_anyone_chooses() -> None:
    """`spy_players.rs` 450. Black has just lost its spy and plays its reply turn. White's
    spy is boxed in and its king steps only forward: Black's knight to c4 leaves White no
    move in the announced round that follows, which stalemate decides for Black before
    anyone chooses. Greedy takes that turn, valuing the round by its verdict, as the
    oracle does."""
    engine = _stepping_spy()
    state = _place(engine, {"a1": "S", "c3": "K"},
                   {"a2": "P", "b1": "P", "b2": "N", "e5": "K"}, side=1)
    assert engine.phase(state).name == "reply" and engine.terminal(state) is None
    block = _black_move(engine, state, "b2", "c4")
    decided = engine.apply(state, block)
    assert engine.phase(decided).is_round and engine.legal_moves_for(decided, 0) == []
    verdict = engine.terminal(decided)
    assert verdict is not None and verdict.winner == 1
    moves = engine.legal_moves(state)
    native = native_spec("greedy", engine.spec.hash)
    assert native is not None
    chosen = engine.core.select_move(
        _pack(state), native, eval_values(engine.spec), 7,
    )
    expected = build_agent("greedy", engine).select_move(state, moves, seat_rng(7, 0))
    assert chosen == expected == block
