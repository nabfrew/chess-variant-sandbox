"""R-P2b (`docs/reference/selective-search-duel.md` §3, §8): the three selective-search arms.

``null=`` (the null move, verified), ``lmr=`` (late moves one ply shallower) and ``pvs=`` (zero
windows after the first move) are repair bits of a minimax player, off by default. Held here:
their bits and options; the other players refusing them; ``null_for`` held to the pass's own
refusals; the core and the oracle alike, move for move and node for node, with each arm on; each
arm deterministic, inside its node budget and changing the search when on; and under every
compiled flag that switches an arm off, the arm spelled on playing as the arm spelled off.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents.base import build_agent, native_spec
from chess_variant_sandbox.agents.minimax import (
    DEFAULT_REPAIRS,
    LMR,
    NULL_MOVE,
    PVS,
    MinimaxAgent,
    lmr_for,
    null_for,
    pvs_for,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.refusals import PASS_REFUSALS
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("null=1", "lmr=1", "pvs=1", "pvs=1,tt_move=1", "null=1,lmr=1,pvs=1")


def _spec(path: str, plies: int | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if plies is not None:
        spec["rules"]["max_plies"] = plies
    return spec


def _engine(path: str, plies: int | None = None) -> Engine:
    return Engine(_spec(path, plies))


# --------------------------------------------------------------------- the names


def test_the_options_are_their_bits_and_none_is_on_by_default() -> None:
    assert (NULL_MOVE, LMR, PVS) == (1048576, 2097152, 4194304)
    for option, bit in (("null", NULL_MOVE), ("lmr", LMR), ("pvs", PVS)):
        assert repairs_of({option: 1}) == DEFAULT_REPAIRS | bit
        assert repairs_of({option: 0}) == repairs_of({}) == DEFAULT_REPAIRS
        assert not DEFAULT_REPAIRS & bit
        assert not _core.DEFAULT_REPAIRS & bit


@pytest.mark.parametrize("kind", ["greedy", "mcts"])
@pytest.mark.parametrize("option", ["null", "lmr", "pvs"])
def test_a_player_that_does_not_search_so_refuses_the_option(kind: str, option: str) -> None:
    engine = _engine("rulesets/gardner.json")
    with pytest.raises(ValueError, match=option):
        build_agent(f"{kind}:{option}=1", engine)
    with pytest.raises(ValueError, match=option):
        native_spec(f"{kind}:{option}=1")


@pytest.mark.parametrize("option", ["null", "lmr", "pvs"])
def test_the_round_player_refuses_the_option(option: str) -> None:
    engine = _engine("archive/watering-hole.json")
    with pytest.raises(ValueError, match=option):
        build_agent(f"matrix:nodes=100,{option}=1", engine)


def test_the_core_takes_the_bits() -> None:
    """A bit the core did not define is refused by `AgentSpec::from_parts`; these are not."""
    engine = _engine("rulesets/gardner.json", plies=4)
    agents = ("minimax:nodes=200,null=1,lmr=1,pvs=1", "minimax:nodes=200")
    assert play_game(engine, agents, seed=1).plies == 4


# --------------------------------------------------- where the null move is off


# Each of `PASS_REFUSALS`' rows, as the compiled flag that switches the null move off with it.
_PASS_FLAGS: dict[str, dict[str, Any]] = {
    "pass.P1": {"simultaneous": object()},
    "pass.P2": {"duck": object()},
    "pass.P3": {"turn": object()},
    "pass.P4": {"must_capture": True},
    "pass.P5": {"drops": SimpleNamespace(placement=True)},
    "pass.P6": {"npcs": object()},
    "pass.P7": {"portals": object()},
    "pass.P8": {"push": object()},
}
# §3.1's own two rows, and the parts half of P3.
_OWN_FLAGS: dict[str, dict[str, Any]] = {
    "spy": {"spy": object()},
    "parts": {"parts": object()},
    "extinction wins": {"extinction": SimpleNamespace(any_piece=True, result="win")},
    "stalemate wins": {"stalemate": "win"},
    "rest": {"rest": (0, 2)},
    "income without the pass": {"income": object()},
}


def _plain(**flags: Any) -> SimpleNamespace:
    spec = SimpleNamespace(
        simultaneous=None, spy=None, must_capture=False, extinction=None, stalemate="draw",
        duck=None, turn=None, parts=None, npcs=None, portals=None, push=None, drops=None,
        rest=(), income=None, passes=False,
    )
    for name, value in flags.items():
        setattr(spec, name, value)
    return spec


def test_null_for_is_held_to_the_pass_refusals() -> None:
    """Where the rules cannot say what a pass is, the search may not play one: every row of
    `PASS_REFUSALS` switches the null move off. A row lifted or added there fails here first."""
    assert {row.id for row in PASS_REFUSALS} == set(_PASS_FLAGS)
    assert null_for(_plain()) and lmr_for(_plain()) and pvs_for(_plain())
    for flags in (*_PASS_FLAGS.values(), *_OWN_FLAGS.values()):
        assert not null_for(_plain(**flags)), flags
    # Income beside the pass is the pass's own family, and on.
    assert null_for(_plain(income=object(), passes=True))
    assert null_for(_plain(extinction=SimpleNamespace(any_piece=False, result="win")))
    assert null_for(_plain(extinction=SimpleNamespace(any_piece=True, result="loss")))


def test_lmr_and_pvs_are_off_only_where_registered() -> None:
    for name in ("simultaneous", "spy"):
        assert not lmr_for(_plain(**{name: object()}))
        assert not pvs_for(_plain(**{name: object()}))
    assert not lmr_for(_plain(must_capture=True)) and pvs_for(_plain(must_capture=True))
    assert not lmr_for(_plain(npcs=object())) and pvs_for(_plain(npcs=object()))
    for flags in ({"duck": object()}, {"turn": object()}, {"passes": True}, {"push": object()}):
        assert lmr_for(_plain(**flags)) and pvs_for(_plain(**flags))


def _income_without_the_pass() -> dict[str, Any]:
    spec = _spec("tests/_purse_income.json", plies=30)
    spec["name"] = "the purse income without the pass"
    spec["rules"].pop("pass")
    return spec


# (arm, the game that carries the flag): §3's tables, one committed game a flag where the
# library has one, and income without the pass built (the library has none).
_OFF: list[tuple[str, str]] = [
    ("null=1", "rulesets/antichess.json"),  # forced capture; extinction and stalemate win
    ("null=1", "rulesets/duck-chess.json"),  # the duck
    ("null=1", "rulesets/drover.json"),  # a long turn, parts
    ("null=1", "archive/kraken-ford.json"),  # wild pieces
    ("null=1", "archive/carousel.json"),  # portals
    ("null=1", "rulesets/dohyo.json"),  # a push
    ("null=1", "archive/placement-chess.json"),  # a placement phase
    ("null=1", "archive/chess-with-a-weary-queen.json"),  # rest
    ("null=1", "income without the pass"),
    ("null=1", "tests/_spy_promoting.json"),  # a spy game
    ("lmr=1", "rulesets/antichess.json"),  # forced capture
    ("lmr=1", "archive/kraken-ford.json"),  # wild pieces
    ("lmr=1", "tests/_spy_promoting.json"),
    ("pvs=1", "tests/_spy_promoting.json"),
]


@pytest.mark.parametrize("arm, game", _OFF)
def test_under_an_off_flag_the_arm_plays_as_off(arm: str, game: str) -> None:
    spec = _income_without_the_pass() if game == "income without the pass" else _spec(game, 30)
    engine = Engine(spec)
    test = {"null=1": null_for, "lmr=1": lmr_for, "pvs=1": pvs_for}[arm]
    assert not test(engine.spec)
    off = play_game(engine, ("minimax:nodes=3000", "minimax:nodes=3000"), seed=4)
    on = play_game(engine, (f"minimax:nodes=3000,{arm}", f"minimax:nodes=3000,{arm}"), seed=4)
    assert on.moves == off.moves
    assert on.nodes == off.nodes
    assert on.depth_sum == off.depth_sum


# ------------------------------------------------------- core and oracle alike


def _assert_alike(engine: Engine, agents: tuple[str, str], seed: int = 3) -> None:
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum
    assert native.searches == in_python.searches


@pytest.mark.parametrize("arm", ARMS)
def test_native_and_oracle_alike(arm: str) -> None:
    engine = _engine("rulesets/gardner.json", plies=12)
    _assert_alike(engine, (f"minimax:nodes=3000,{arm}", f"minimax:nodes=3000,{arm}"))


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize(
    "game, plies",
    [
        ("rulesets/gardner.json", 40),
        ("rulesets/chess.json", 16),
        ("rulesets/hunger.json", 16),  # the pass is legal: the null move is a real move
        # No check (royal capture), with drops. Not dohyo: there the default player itself already
        # parts from its oracle at ply 8 under `threats=` (seed 3), with every arm off, on main too.
        ("rulesets/dobutsu.json", 16),
    ],
)
def test_native_and_oracle_alike_deeper(arm: str, game: str, plies: int) -> None:
    engine = _engine(game, plies)
    _assert_alike(engine, (f"minimax:nodes=3000,{arm}", f"minimax:nodes=2000,{arm}"))


# ------------------------------------------------ deterministic, budgeted, acting


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_deterministic_and_inside_its_budget(arm: str) -> None:
    engine = _engine("rulesets/chess.json", plies=24)
    agents = (f"minimax:nodes=6400,{arm}", f"minimax:nodes=1600,{arm}")
    first = play_game(engine, agents, seed=9)
    again = play_game(engine, agents, seed=9)
    assert (first.moves, first.nodes, first.depth_sum) == (again.moves, again.nodes, again.depth_sum)
    for seat, budget in enumerate((6400, 1600)):
        assert 0 < first.nodes[seat] <= budget * first.searches[seat]


@pytest.mark.parametrize("bit", [NULL_MOVE, LMR, PVS, NULL_MOVE | LMR | PVS])
def test_every_search_stays_inside_its_budget(bit: int) -> None:
    """Search by search, in the oracle (which the core matches node for node): every
    position an arm visits is a counted node, so no search spends past its budget."""
    engine = _engine("rulesets/gardner.json")
    state = engine.initial_state()
    import random

    for budget in (700, 2500):
        agent = MinimaxAgent(engine, nodes=budget, repairs=DEFAULT_REPAIRS | bit)
        rng = random.Random(2)
        position = state
        for _ in range(4):
            moves = engine.legal_moves(position)
            spent = agent.stats.nodes
            move = agent.select_move(position, moves, rng)
            assert agent.stats.nodes - spent <= budget
            position = engine.apply(position, move)


@pytest.mark.parametrize("arm", ["null=1", "lmr=1", "pvs=1", "pvs=1,tt_move=1"])
def test_each_arm_changes_the_search_when_on(arm: str) -> None:
    """Searched to a fixed depth on a budget that never runs out, so the nodes a search spends
    are the search's own size: each arm visits a different tree from the player without it."""
    engine = _engine("rulesets/chess.json", plies=6)
    control = "tt_move=1" if "tt_move" in arm else "pvs=0"
    budget = "nodes=100000000,max_depth=4"
    off = play_game(engine, (f"minimax:{budget},{control}",) * 2, seed=9)
    on = play_game(engine, (f"minimax:{budget},{arm}",) * 2, seed=9)
    assert on.nodes != off.nodes


def test_spelled_off_is_the_default_player() -> None:
    engine = _engine("rulesets/chess.json", plies=24)
    plain = play_game(engine, ("minimax:nodes=1600",) * 2, seed=9)
    spelled = play_game(engine, ("minimax:nodes=1600,null=0,lmr=0,pvs=0",) * 2, seed=9)
    assert (plain.moves, plain.nodes, plain.depth_sum) == (
        spelled.moves, spelled.nodes, spelled.depth_sum
    )
