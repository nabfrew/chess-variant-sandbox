"""The immunity term (Q466, F2): a piece the opponent can never capture.

`docs/reference/immunity-term.md` has the design and the duels. These hold four things:
what the term reads is the capture relation and nothing else (a constructed position, and
zero wherever the relation cannot make a piece immune); the core and the Python oracle
play the same game with it on; it is live, so the arms test something; and since Q471 it
is the default, for the players and for the `decision_criticality` probe alike.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents.evaluation import IMMUNE_PIECES, Evaluator, immunity_threats
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import (
    PROBE_MAX_DEPTH,
    PROBE_NODES,
    _root_scores,
    _score_roots,
)
from chess_variant_sandbox.tournament.play import eval_values, play_game, play_game_python

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _engine(name: str) -> Engine:
    return Engine(load_spec(RULESETS / f"{name}.json"))


def _type(engine: Engine, piece_id: str) -> int:
    return engine.spec.piece_ids.index(piece_id)


def _stripped(engine: Engine, side: int, piece_id: str) -> GameState:
    """The opening with every ``piece_id`` of ``side`` taken off.

    In *Intransitive*, side 1 with no scissors makes every paper of side 0 immune: the
    decisive exchange of Q462's games, constructed.
    """
    spec = engine.spec
    state = engine.initial_state()
    stride = spec.n_facings * spec.n_types
    victim = _type(engine, piece_id)
    board = bytearray(state.board)
    for square, value in enumerate(board):
        if value and (value - 1) // stride == side and (value - 1) % spec.n_types == victim:
            board[square] = 0
    return replace(state, board=bytes(board))


def test_the_default_is_the_cores() -> None:
    """One constant, stated on both sides of the boundary and held together here."""
    assert _core.IMMUNE_PIECES == IMMUNE_PIECES == 1.0


@pytest.mark.parametrize("name", ["chess", "shogi", "xiangqi", "gardner"])
def test_a_game_with_no_capture_matrix_has_no_term(name: str) -> None:
    """Where every hunter may take every type, the enemy's royal always can: undefined."""
    assert immunity_threats(_engine(name).spec) is None


def test_the_relation_reads_rock_paper_scissors() -> None:
    engine = _engine("intransitive")
    table = immunity_threats(engine.spec)
    assert table is not None
    threat, counted = table
    rock, paper, scissors = (_type(engine, piece) for piece in "RPS")
    for player in (0, 1):
        assert threat[player][rock] == 1 << scissors
        assert threat[player][paper] == 1 << rock
        assert threat[player][scissors] == 1 << paper
    assert counted == (1 << rock) | (1 << paper) | (1 << scissors)


def test_the_opening_has_nothing_immune() -> None:
    engine = _engine("intransitive")
    evaluator = Evaluator.for_engine(engine, immune_pieces=1.0)
    state = engine.initial_state()
    assert evaluator.immunity(state, 0) == 0.0
    assert evaluator.immunity(state, 1) == 0.0


def test_taking_every_scissors_makes_every_paper_immune() -> None:
    """The decisive exchange of Q462's games, constructed: side 1 keeps no scissors."""
    engine = _engine("intransitive")
    spec = engine.spec
    evaluator = Evaluator.for_engine(engine, immune_pieces=1.0)
    stride = spec.n_facings * spec.n_types
    paper = _type(engine, "P")
    stripped = _stripped(engine, 1, "S")
    papers = [
        square
        for square, value in enumerate(stripped.board)
        if value and (value - 1) // stride == 0 and (value - 1) % spec.n_types == paper
    ]
    assert papers
    nearest = min(spec.goal_distance[0][paper][square] for square in papers)
    diameter = float(spec.goal_diameter)
    proximity = (diameter - nearest) / diameter
    # Every type is priced alike here, so a paper is one mean piece.
    assert len(set(evaluator.values)) == 1
    assert evaluator.immunity(stripped, 0) == pytest.approx(len(papers) + proximity)
    assert evaluator.immunity(stripped, 1) == pytest.approx(-(len(papers) + proximity))
    # And the evaluator adds it at the mean non-royal piece.
    plain = Evaluator.for_engine(engine, immune_pieces=0.0)
    difference = evaluator.evaluate(engine, stripped, 0) - plain.evaluate(engine, stripped, 0)
    assert difference == pytest.approx(len(papers) + proximity)


@pytest.mark.parametrize("name", ["rps-hierarchy-chess", "chess"])
def test_the_term_changes_no_game_where_nothing_can_be_immune(name: str) -> None:
    """A royal that captures everything, or no matrix at all: the same game to the bit."""
    engine = _engine(name)
    plain = play_game(engine, ("minimax:nodes=150", "greedy"), seed=5)
    immune = play_game(engine, ("minimax:nodes=150,immune=4", "greedy:immune=4"), seed=5)
    assert plain.moves == immune.moves


# Seeds on which the immune player's game differs from the plain one's, so the identity
# is between two searches that both read the term, not two that ignore it. The opponent
# spells `immune=0`, the player every opponent was before the term became the default.
LIVE_ARMS = [
    (("greedy:immune=4", "greedy:immune=0"), 2),
    (("minimax:nodes=150,immune=4", "minimax:nodes=150,immune=0"), 4),
]


@pytest.mark.parametrize("agents,seed", LIVE_ARMS, ids=["greedy", "minimax"])
def test_the_term_is_searched_the_same_way_on_both_sides(agents: tuple[str, str], seed: int) -> None:
    engine = _engine("intransitive")
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.material == in_python.material
    assert native.nodes == in_python.nodes


@pytest.mark.parametrize("agents,seed", LIVE_ARMS, ids=["greedy", "minimax"])
def test_the_term_changes_the_game_on_intransitive(agents: tuple[str, str], seed: int) -> None:
    engine = _engine("intransitive")
    plain = (agents[0].replace("immune=4", "immune=0"), agents[1])
    assert play_game(engine, plain, seed=seed).moves != play_game(engine, agents, seed=seed).moves


# Q471: the default. Seeds on which the default's game differs from `immune=0`'s, so the
# identity with `immune=1` is between two players that read the term.
DEFAULT_ARMS = [("greedy", 3), ("minimax:nodes=150", 4)]
DEFAULT_OPPONENT = "minimax:nodes=150,immune=0"


def _spelled(player: str, option: str) -> str:
    return f"{player}{',' if ':' in player else ':'}{option}"


@pytest.mark.parametrize("player,seed", DEFAULT_ARMS, ids=["greedy", "minimax"])
def test_a_player_that_names_no_weight_reads_the_term(player: str, seed: int) -> None:
    """The default is `immune=1`, and `immune=0` is a different player (Q471)."""
    engine = _engine("intransitive")

    def moves(seat: str) -> tuple[int, ...]:
        return tuple(play_game(engine, (seat, DEFAULT_OPPONENT), seed=seed).moves)

    assert moves(player) == moves(_spelled(player, "immune=1"))
    assert moves(player) != moves(_spelled(player, "immune=0"))


@pytest.mark.parametrize("player,seed", DEFAULT_ARMS, ids=["greedy", "minimax"])
def test_the_default_is_searched_the_same_way_on_both_sides(player: str, seed: int) -> None:
    """The default crosses the boundary: the oracle and the core play one game."""
    engine = _engine("intransitive")
    agents = (player, DEFAULT_OPPONENT)
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


def test_the_probe_reads_the_term() -> None:
    """`decision_criticality`'s probe prices immunity at the rated players' weight.

    The core's probe against the Python oracle it ports, on the constructed exchange
    where the term reads non-zero, and a guard that the term moves the scores there.
    """
    engine = _engine("intransitive")
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    assert evaluator.immune_pieces == IMMUNE_PIECES
    stripped = _stripped(engine, 1, "S")
    assert evaluator.immunity(stripped, 0) != 0.0
    legal = engine.legal_moves(stripped)
    assert len(legal) > 1
    scores = _score_roots(engine, evaluator, stripped, legal)
    assert scores == _root_scores(engine, evaluator, stripped, legal)
    without = engine.root_scores(
        stripped,
        eval_values(engine.spec),
        PROBE_NODES,
        PROBE_MAX_DEPTH,
        evaluator.goal_pieces,
        0.0,
    )
    assert scores != without
