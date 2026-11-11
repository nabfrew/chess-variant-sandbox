"""T3 (Q1171, `docs/plan-player-weaknesses.md`): repetition contempt, a diagnostic search bit.

A draw by repetition reached in the search scores `CONTEMPT_PIECES` mean non-royal pieces against the side that
completed it. Never on by default (the user, 2026-11-08): the default player, and every measured one, is the player
of before, held here move for move and node for node. Under the bit the core and the oracle agree move for move, and
on a game that repeats the bit changes the game. Dobutsu is used because its built-in players repeat at small
budgets (0.22 of its library's minimax games); nothing here is about its rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import minimax
from chess_variant_sandbox.agents.minimax import (
    CONTEMPT,
    DEFAULT_REPAIRS,
    REPAIR_OPTIONS,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game, play_game_python

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
# The seed at which one contempt seat changes the game at 120 nodes (found once, by looking).
CHANGED_SEED = 9


def _engine() -> Engine:
    return Engine(load_spec(RULESETS / "dobutsu.json"))


def test_the_constant_is_held_to_the_core() -> None:
    assert _core.CONTEMPT_PIECES == minimax.CONTEMPT_PIECES == 0.5


def test_contempt_is_a_repair_off_by_default() -> None:
    assert REPAIR_OPTIONS["contempt"] == CONTEMPT == 8192
    assert not DEFAULT_REPAIRS & CONTEMPT
    assert not _core.DEFAULT_REPAIRS & CONTEMPT
    assert repairs_of({}) == repairs_of({"contempt": 0}) == DEFAULT_REPAIRS
    assert repairs_of({"contempt": 1}) == DEFAULT_REPAIRS | CONTEMPT


@pytest.mark.parametrize("seed", [1, 2, CHANGED_SEED])
def test_the_default_player_is_unchanged(seed: int) -> None:
    """The default player and the one spelling ``contempt=0`` play one game, node for node, on
    seeds whose games end by repetition and one where the bit would change it."""
    engine = _engine()
    default = play_game(engine, ("minimax:nodes=120", "minimax:nodes=120"), seed=seed)
    spelled = play_game(engine, ("minimax:nodes=120,contempt=0", "minimax:nodes=120,contempt=0"), seed=seed)
    assert default.moves == spelled.moves
    assert default.nodes == spelled.nodes
    assert (default.winner, default.reason) == (spelled.winner, spelled.reason)


def test_contempt_changes_a_game_that_repeats() -> None:
    engine = _engine()
    plain = play_game(engine, ("minimax:nodes=120", "minimax:nodes=120"), seed=CHANGED_SEED)
    contempt = play_game(engine, ("minimax:nodes=120,contempt=1", "minimax:nodes=120"), seed=CHANGED_SEED)
    assert plain.reason == "repetition"
    assert contempt.moves != plain.moves


@pytest.mark.parametrize(
    "agents",
    [
        ("minimax:nodes=120,contempt=1", "minimax:nodes=120"),
        ("minimax:nodes=120", "minimax:nodes=120,contempt=1"),
        ("minimax:nodes=120,contempt=1", "minimax:nodes=120,contempt=1"),
    ],
    ids=lambda pair: "-vs-".join(pair),
)
@pytest.mark.parametrize("seed", [1, CHANGED_SEED])
def test_contempt_is_searched_the_same_way_on_both_sides(agents: tuple[str, str], seed: int) -> None:
    """`agents.rs::searched_terminal` against `MinimaxAgent._terminal_value`."""
    engine = _engine()
    native = play_game(engine, agents, seed=seed)
    in_python = play_game_python(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum
