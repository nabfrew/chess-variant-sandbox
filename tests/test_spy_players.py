"""Schema 27's computer players across a spy game's phases (Q935b, `docs/plan-spy.md` §3).

`greedy` and `minimax:nodes=N` play every phase in the core (`rust/src/spy_players.rs`) and
in their declared Python oracles (`agents/simple.py`, `agents/minimax.py`), which agree move
for move, node for node and series for series over whole games of the spy test specs; the
probes (`Core.select_round` with an announced move) agree on single positions; and the
tactic tests of §3 (G3) hold on hand-built positions at every budget: the answerer dodges,
takes a standing royal and meets a fleeing one; the announcer never leaves its royal
standing attacked, never flees to a square the answerer reaches, and takes the exact
maximin of a 3x3 round.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents import build_agent, native_spec
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import eval_values, play_game, play_game_python, seat_rng

HERE = Path(__file__).parent

# Every budget the tactic tests hold at: greedy, the ladder's bottom, the canonical top and
# the study's strong band's first rung (`plan-spy.md` §3.4).
PLAYERS = ("greedy", "minimax:nodes=100", "minimax:nodes=1600", "minimax:nodes=6400")


def _load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


@cache
def _engine(name: str) -> Engine:
    return Engine(_load(name))


def _square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _position(
    engine: Engine, white: dict[str, str], black: dict[str, str], side: int = 0
) -> GameState:
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    state = engine.make_state(placement, side_to_move=side)
    assert engine.terminal(state) is None
    return state


def _lists(engine: Engine, state: GameState) -> tuple[list[int], list[int]]:
    return engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)


def _move(engine: Engine, moves: list[int], origin: str, target: str) -> int:
    a, b = _square(engine, origin), _square(engine, target)
    found = [m for m in moves if decode_move(m)[:2] == (a, b)]
    assert len(found) == 1, (origin, target)
    return found[0]


def _choose(
    engine: Engine, state: GameState, seat: int, agent: str, announced: int | None = None
) -> tuple[int, int]:
    """The core's choice for ``seat`` here and its nodes, held to the Python oracle's."""
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, strategy, nodes, _ = engine.select_round(
        state, seat, native, eval_values(engine.spec), 7, announced
    )
    assert move is not None
    player = build_agent(agent, engine)
    lists = _lists(engine, state)
    rng = seat_rng(7, seat)
    if engine.phase(state).name == "announced" and announced is None:
        expected = player.select_announcement(state, seat, lists, rng)
    elif announced is not None:
        expected = player.select_answer(state, seat, lists, announced, rng)
    else:
        expected = player.select_round_move(state, seat, lists[seat], lists[1 - seat], rng)
    assert move == expected, (agent, engine.describe_move(state, move), engine.describe_move(state, expected))
    assert nodes == player.stats.nodes, (agent, nodes, player.stats.nodes)
    assert strategy == player.strategy
    return move, nodes



# ------------------------------------------------------------- the core and its oracles

_PAIRINGS = (
    ("greedy", "greedy"),
    ("minimax:nodes=150", "greedy"),
    ("random", "minimax:nodes=200"),
)


_SPECS = ("_spy_small", "_spy_blast", "_spy_stock", "_spy_resting")


# Seed 0's game on two of these pairs is long, and the Python oracle's series (the evaluator's
# capture threats, 20-30 ms a ply) is where the time goes: 118 plies of minimax:nodes=150
# against greedy on `_spy_small` cost 3.5 s, 88 plies of greedy on `_spy_stock` 2.5 s (Q997b).
# The fast suite plays those two pairs at a seed whose game is shorter (46 and 38 plies, 1.2 s and 1.1 s);
# the slow tier plays seed 0's, below.
_LONG_AT_SEED_0 = {
    ("minimax:nodes=150", "greedy", "_spy_small"): 28,
    ("greedy", "greedy", "_spy_stock"): 10,
}


def _same_game(agents: tuple[str, str], name: str, seed: int) -> None:
    engine = _engine(name)
    native = play_game(engine, agents, seed)
    python = play_game_python(engine, agents, seed)
    assert native.moves == python.moves, (name, agents)
    assert (native.winner, native.reason) == (python.winner, python.reason)
    assert native.nodes == python.nodes
    assert native.depth_sum == python.depth_sum and native.searches == python.searches
    assert native.branching == python.branching and native.captures == python.captures
    assert native.material == python.material


@pytest.mark.parametrize("name", _SPECS)
@pytest.mark.parametrize("agents", _PAIRINGS, ids="-".join)
def test_the_core_plays_every_phase_as_the_oracles_do(agents: tuple[str, str], name: str) -> None:
    """Whole games, the core's loop against `play_game_python` over the oracles."""
    _same_game(agents, name, _LONG_AT_SEED_0.get((*agents, name), 0))


@pytest.mark.slow  # seed 0's 118- and 88-ply games: 3.5 s and 2.5 s in the Python oracle (Q997b)
@pytest.mark.parametrize("case", list(_LONG_AT_SEED_0), ids="-".join)
def test_the_long_games_at_seed_zero_are_the_oracles_too(case: tuple[str, str, str]) -> None:
    _same_game((case[0], case[1]), case[2], 0)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3))
def test_deeper_searches_agree_with_their_oracle(seed: int) -> None:
    """The same comparison at budgets that search announced rounds inside the tree several
    rounds deep, the table and the killers working across them."""
    for name in _SPECS:
        engine = _engine(name)
        agents = ("minimax:nodes=1200", "minimax:nodes=800")
        native = play_game(engine, agents, seed)
        python = play_game_python(engine, agents, seed)
        assert native.moves == python.moves, (name, seed)
        assert native.nodes == python.nodes and native.depth_sum == python.depth_sum


@pytest.mark.parametrize("agents", _PAIRINGS, ids="-".join)
def test_the_compared_games_cross_every_phase(agents: tuple[str, str]) -> None:
    """The comparison above is worth having only because the searching players chose in
    every phase: the same games (the core's, which the oracles matched) replayed."""
    from chess_variant_sandbox.tournament.play import replay

    phases: set[str] = set()
    for name in _SPECS:
        engine = _engine(name)
        states = replay(engine, play_game(engine, agents, 0))
        phases |= {engine.phase(state).name for state in states[:-1]}
    assert {"alternating", "announced", "blind"} <= phases, phases


# ------------------------------------------------------------------ the tactic tests (G3)
#
# On `_spy_small`'s 5x5 board (`plan-spy.md` §1.5): White holds the spy and is the
# answerer, Black has none and announces. Each holds for every player in `PLAYERS`, in the
# core; `_ORACLE_PLAYERS` are also held to the Python oracles, choice and nodes.

_ORACLE_PLAYERS = ("greedy", "minimax:nodes=100", "minimax:nodes=400")


def _ask(engine: Engine, state: GameState, seat: int, agent: str, announced: int | None = None) -> int:
    if agent in _ORACLE_PLAYERS:
        return _choose(engine, state, seat, agent, announced)[0]
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    move, _, _, _ = engine.select_round(state, seat, native, eval_values(engine.spec), 7, announced)
    assert move is not None
    return move


def _winner_after(engine: Engine, state: GameState, first: int, second: int) -> int | None:
    outcome = engine.terminal(engine.apply_round(state, first, second))
    return None if outcome is None else outcome.winner


@pytest.mark.parametrize("agent", PLAYERS)
def test_the_answerer_dodges_an_announced_capture(agent: str) -> None:
    engine = _engine("_spy_small")
    state = _position(engine, {"a1": "K", "c3": "N", "e1": "S"}, {"e5": "K", "c5": "R"})
    assert (engine.phase(state).name, engine.phase(state).announcer) == ("announced", 1)
    white, black = _lists(engine, state)
    # Nothing better is on offer: no white move takes anything.
    assert not any(state.board[decode_move(m)[1]] for m in white)
    announced = _move(engine, black, "c5", "c3")
    answer = _ask(engine, state, 0, agent, announced)
    assert decode_move(answer)[0] == _square(engine, "c3"), engine.describe_move(state, answer)


@pytest.mark.parametrize("agent", PLAYERS)
def test_the_answerer_takes_a_standing_royal(agent: str) -> None:
    engine = _engine("_spy_small")
    state = _position(engine, {"a1": "K", "a4": "R", "e1": "S"}, {"d4": "K", "b5": "N"})
    white, black = _lists(engine, state)
    announced = _move(engine, black, "b5", "c3")
    take = _move(engine, white, "a4", "d4")
    assert _winner_after(engine, state, take, announced) == 0
    assert _ask(engine, state, 0, agent, announced) == take


@pytest.mark.parametrize("agent", PLAYERS)
def test_the_answerer_meets_a_fleeing_royal_at_its_destination(agent: str) -> None:
    engine = _engine("_spy_small")
    state = _position(engine, {"a1": "K", "a4": "R", "e1": "S"}, {"d5": "K"})
    white, black = _lists(engine, state)
    announced = _move(engine, black, "d5", "d4")
    meet = _move(engine, white, "a4", "d4")
    assert _winner_after(engine, state, meet, announced) == 0
    assert _ask(engine, state, 0, agent, announced) == meet


@pytest.mark.parametrize("agent", PLAYERS)
def test_the_announcer_flees_and_never_to_a_square_the_answerer_reaches(agent: str) -> None:
    engine = _engine("_spy_small")
    state = _position(
        engine, {"a1": "K", "a4": "R", "b2": "N", "e1": "S"}, {"d4": "K", "e5": "P"}
    )
    white, black = _lists(engine, state)
    king = _square(engine, "d4")
    reached = {decode_move(m)[1] for m in white}
    assert king in reached  # the royal stands attacked
    flights = [m for m in black if decode_move(m)[0] == king]
    assert any(decode_move(m)[1] in reached for m in flights)  # some flights are caught
    assert any(decode_move(m)[1] not in reached for m in flights)  # and some are not
    assert any(decode_move(m)[0] != king for m in black)  # and it could stand instead
    chosen = _ask(engine, state, 1, agent)
    assert decode_move(chosen)[0] == king, engine.describe_move(state, chosen)
    assert decode_move(chosen)[1] not in reached, engine.describe_move(state, chosen)


# A 3x3 round (`plan-spy.md` §4, Q935b): walls leave Black (the announcer) three moves,
# Kb5, Kd5 and Nb3, and White three, Rd5, Rxc5 and Se3. Kd5 is met by the rook, Nb3 leaves
# the king standing to be taken, and Kb5 loses to nothing: the exact maximin, which does
# not depend on how deep anyone searches.
_WALLED = ["b4", "c4", "d4", "e4", "c2", "d1", "d2"]


def _walled_spec() -> dict[str, Any]:
    spec = _load("_spy_small")
    spec["pieces"] = [piece for piece in spec["pieces"] if piece["id"] != "P"]
    spec["rules"].pop("promotion")
    spec["rules"]["obstacles"] = {"squares": _WALLED}
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": "K", "player": 0, "squares": ["e1"]},
            {"piece": "S", "player": 0, "squares": ["e2"]},
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["c5"]},
            {"piece": "S", "player": 1, "squares": ["a5"]},
            {"piece": "N", "player": 1, "squares": ["b5"]},
        ],
    }
    spec["name"] = "spy walled"
    return spec


@pytest.mark.parametrize("agent", PLAYERS)
def test_the_announcer_takes_the_exact_maximin_of_a_3x3_round(agent: str) -> None:
    engine = Engine(_walled_spec())
    state = _position(engine, {"e1": "K", "e2": "S", "e5": "R"}, {"c5": "K", "a1": "N"})
    white, black = _lists(engine, state)
    assert (len(black), len(white)) == (3, 3)
    worst = {
        a: min(
            {None: 0, 0: -1, 1: 1}[_winner_after(engine, state, b, a)] for b in white
        )
        for a in black
    }
    maximin = [a for a, value in worst.items() if value == max(worst.values())]
    assert len(maximin) == 1 and worst[maximin[0]] == 0
    assert maximin[0] == _move(engine, black, "c5", "b5")
    assert _ask(engine, state, 1, agent) == maximin[0]


# ------------------------------------------------------------------ across the seams


@pytest.mark.parametrize("agent", PLAYERS)
def test_a_spy_capture_good_by_material_is_taken(agent: str) -> None:
    """§3.5: the alternating evaluator and the round one differ in the mobility term only,
    and a search that crosses from one to the other compares them. Here the one capture,
    the black spy, leads into rounds and every other move stays alternating. The spy is
    priced at its value, outside the prey share (a spy captures nothing, and the share
    would leave it a tenth of its value: taken or not, nearly nothing differed)."""
    engine = _engine("_spy_small")
    state = _position(engine, {"a1": "K", "c1": "R", "e1": "S"}, {"a5": "K", "c4": "S"})
    assert engine.phase(state).name == "alternating"
    take = _move(engine, engine.legal_moves(state), "c1", "c4")
    assert engine.phase(engine.apply(state, take)).name == "reply"
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    chosen = engine.select_move(state, native, eval_values(engine.spec), 7)
    assert chosen == take, engine.describe_move(state, chosen)
    if agent in _ORACLE_PLAYERS:
        player = build_agent(agent, engine)
        assert player.select_move(state, engine.legal_moves(state), seat_rng(7, 0)) == take


def test_minimax_plays_a_blind_root_as_the_matrix_search_does() -> None:
    """§3.3: once both spies are gone, `minimax:nodes=N` plays the round as
    `matrix:nodes=N` does, and its oracle with it."""
    engine = _engine("_spy_small")
    state = _position(engine, {"a1": "K", "a4": "R", "b1": "N"}, {"e5": "K", "d5": "R"})
    assert engine.phase(state).name == "blind"
    for seat in (0, 1):
        move, nodes = _choose(engine, state, seat, "minimax:nodes=400")
        assert nodes > 0
        native = native_spec("minimax:nodes=400", engine.spec.hash)
        _, strategy, _, depth = engine.select_round(state, seat, native, eval_values(engine.spec), 7)
        assert len(strategy) == len(_lists(engine, state)[seat]) and depth >= 1


def test_a_spy_tournament_is_the_same_at_one_and_two_workers() -> None:
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

    spec = load_spec(HERE / "_spy_small.json")
    runs = [
        run_tournament(
            spec,
            TournamentConfig(
                agents=("greedy", "minimax:nodes=200"), games_per_pairing=2, seed=935, workers=workers
            ),
        ).records
        for workers in (1, 2)
    ]
    assert [r.moves for r in runs[0]] == [r.moves for r in runs[1]]
    assert [r.nodes for r in runs[0]] == [r.nodes for r in runs[1]]


@pytest.mark.parametrize("agent", ("matrix:nodes=100", "mcts:nodes=50", "minimax:nodes=100,h=0123456789ab"))
def test_one_name_per_budget_on_a_spy_game(agent: str) -> None:
    """§3.3: `matrix:` and `mcts:` are refused on a spy game, whose `minimax:` plays its
    blind rounds, and so is every stored objective."""
    engine = _engine("_spy_small")
    with pytest.raises(ValueError):
        play_game(engine, ("random", agent), 0)


def test_the_screen_plays_a_spy_game_by_its_players() -> None:
    """Since Q935d the analysis pass and `measure` read a spy game's records, so the screen
    plays one by `spy_roster`, the same names, and refuses only a player that cannot."""
    from chess_variant_sandbox.loop.screen import SCREEN_AGENTS, roster_for, screen, spy_roster

    spec = _load("_spy_small")
    assert roster_for(spec, SCREEN_AGENTS) == SCREEN_AGENTS
    with pytest.raises(ValueError, match="cannot play a spy game"):
        roster_for(spec, ("matrix:nodes=1600", "matrix:nodes=6400"))
    result = screen(spec, workers=1)
    assert result.stats["games"] == 8.0
    refused = screen(spec, agents=("mcts:nodes=50", "minimax:nodes=100"), workers=1)
    assert not refused.passed and "cannot play a spy game" in refused.reasons[0]
    assert spy_roster(SCREEN_AGENTS + ("minimax:nodes=80,threats=0",)) == SCREEN_AGENTS + (
        "minimax:nodes=80,threats=0",
    )
    assert spy_roster(("random", "greedy", "minimax:nodes=6400")) == (
        "random", "greedy", "minimax:nodes=6400"
    )
    for refused in ("matrix:nodes=1600", "mcts:nodes=100", "minimax:h=abc123,nodes=100"):
        with pytest.raises(ValueError, match="cannot play a spy game"):
            spy_roster((refused,))


@pytest.mark.parametrize("agent", PLAYERS[1:])
def test_a_guarded_spy_is_not_taken_for_a_rook(agent: str) -> None:
    """The reply turn (§1.3) lets the spy's owner retake its taker, so guarding a spy means
    something; a search sees the retake across the seam. (Greedy, one ply deep, does not.)"""
    engine = _engine("_spy_small")
    state = _position(engine, {"e1": "K", "c1": "R", "a1": "S"}, {"a5": "K", "c4": "S", "b5": "P"})
    take = _move(engine, engine.legal_moves(state), "c1", "c4")
    retake = _move(engine, engine.legal_moves(engine.apply(state, take)), "b5", "c4")
    assert retake
    native = native_spec(agent, engine.spec.hash)
    assert native is not None
    assert engine.select_move(state, native, eval_values(engine.spec), 7) != take


# The searching players on the two games every bump holds still, taken on main's core
# 904e4c25ccc0edf4 before Q935b and on this one, and equal (`plan-spy.md` §2.3): the spy's
# instances of the searches and its evaluator line leave every game without a spy alone.
# Q986b (2026-10-20) re-pinned the round game on purpose: a finished round is worth
# `ROUND_WIN_PIECES` mean pieces in the round search, a new round player. It was a559af13... .
_INERT = {
    "rulesets/chess.json": "454a5507328148886b9f5c39b091907dc686caca91ee57beda7cf16a7b64a782",
    "tests/_retired_gardner_in_rounds.json":
        "ea68bf59be2363fef170199cada924fac4b354a18be67e43e2e3726c2d22c835",
}


@pytest.mark.parametrize("path", sorted(_INERT))
def test_the_searching_players_play_a_game_without_a_spy_as_before(path: str) -> None:
    from tests.inert_players import digest

    assert digest(path) == _INERT[path]
