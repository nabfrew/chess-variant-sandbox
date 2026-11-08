"""N4 (`docs/reference/no-forced-mate.md` §10): the mop-up term.

Where one side is bare (royals alone, nothing in hand) and the other is not, a minimax player's
stand pat adds, for the armed side, how confined the bare side is: half its box (positions it
reaches moving alone, `Game::bare_reach`), half its legal moves. Built-in objective only; it
reads the core's legal lists, never a square. Held here to the oracle move for move and node
for node, on endings built from chess's pieces (chess only for its familiar endings) and on a
small board; inert where nobody is bare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import minimax
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.minimax import (
    MOPUP,
    MinimaxAgent,
    is_armed,
    is_bare,
    mopup_weight,
    repairs_of,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _chess_ending(piece: str, squares: tuple[str, str, str], max_plies: int = 0) -> Engine:
    """Chess's rules from a constructed start: a king and ``piece`` against a lone king."""
    spec = _load("rulesets/chess.json")
    spec["name"] = f"chess ending K{piece} v K"
    own, king, theirs = squares
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": "K", "squares": [king], "player": 0},
            {"piece": piece, "squares": [own], "player": 0},
            {"piece": "K", "squares": [theirs], "player": 1},
        ],
    }
    for key in ("castling", "en_passant", "promotion"):
        spec["rules"].pop(key, None)
    spec["pieces"] = [p for p in spec["pieces"] if p["id"] in ("K", piece)]
    if max_plies:
        spec["rules"]["max_plies"] = max_plies
    return Engine(spec)


def test_the_constants_are_the_cores() -> None:
    assert minimax.MOPUP_PIECES == _core.MOPUP_PIECES
    assert minimax.MOPUP_SPAN == _core.MOPUP_SPAN
    assert minimax.MOPUP_AREA == _core.MOPUP_AREA


def test_the_option_is_bit_512() -> None:
    assert MOPUP == 512
    assert repairs_of({"mopup": 1}) == repairs_of({}) | MOPUP
    assert repairs_of({"mopup": 0}) == repairs_of({}) & ~MOPUP


def test_bare_and_armed() -> None:
    engine = _chess_ending("R", ("a1", "e1", "e6"))
    state = engine.initial_state()
    spec = engine.spec
    assert is_bare(spec, state, 1) and not is_bare(spec, state, 0)
    assert is_armed(spec, state, 0) and not is_armed(spec, state, 1)
    opening = Engine(_load("rulesets/chess.json")).initial_state()
    assert not is_bare(spec, opening, 0) and not is_bare(spec, opening, 1)


def test_the_weight_is_undefined_off_the_builtin_objective_and_without_royals() -> None:
    engine = Engine(_load("rulesets/chess.json"))
    assert mopup_weight(engine.spec, Evaluator.for_engine(engine)) > 0.0
    assert mopup_weight(engine.spec, object()) == 0.0
    antichess = Engine(_load("rulesets/antichess.json"))
    assert mopup_weight(antichess.spec, Evaluator.for_engine(antichess)) == 0.0


def test_the_box_of_a_cornered_king() -> None:
    """A lone king on a1 under a rook on h3 that its own king on h4 defends: it may walk the
    first two ranks but the rook's file, fourteen squares, and no further; the walk stops at the cap. The rook
    undefended (its king far away on a8) is taken, and the box opens to the whole board."""
    agent_for = lambda engine: MinimaxAgent(engine, nodes=100, repairs=repairs_of({"mopup": 1}))  # noqa: E731
    held = _chess_ending("R", ("h3", "h4", "a1"))
    assert agent_for(held)._bare_reach(held.initial_state(), 1, 64) == 14
    assert agent_for(held)._bare_reach(held.initial_state(), 1, 3) == 3
    loose = _chess_ending("R", ("h3", "a8", "a1"))
    assert agent_for(loose)._bare_reach(loose.initial_state(), 1, 64) == 64


@pytest.mark.parametrize(
    ("piece", "squares"), [("R", ("a1", "e1", "e6")), ("Q", ("h7", "b2", "d5"))]
)
def test_native_and_oracle_alike(piece: str, squares: tuple[str, str, str]) -> None:
    engine = _chess_ending(piece, squares, max_plies=12)
    agents = ("minimax:nodes=120,mopup=1", "minimax:nodes=120,mopup=1")
    native = play_game(engine, agents, seed=3)
    in_python = play_game_python(engine, agents, seed=3)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


@pytest.mark.slow
@pytest.mark.parametrize(
    ("piece", "squares", "nodes"),
    [("R", ("a1", "e1", "e6"), 200), ("Q", ("h7", "b2", "d5"), 300)],
)
def test_native_and_oracle_alike_whole(piece: str, squares: tuple[str, str, str], nodes: int) -> None:
    engine = _chess_ending(piece, squares)
    agents = (f"minimax:nodes={nodes},mopup=1", f"minimax:nodes={nodes},mopup=1")
    native = play_game(engine, agents, seed=3)
    in_python = play_game_python(engine, agents, seed=3)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes


def test_the_term_moves_a_bare_ending() -> None:
    engine = _chess_ending("Q", ("h7", "b2", "d5"))
    armed = play_game(engine, ("minimax:nodes=1600,mopup=1", "minimax:nodes=1600"), seed=0)
    control = play_game(engine, ("minimax:nodes=1600,mopup=0", "minimax:nodes=1600"), seed=0)
    assert armed.moves != control.moves
    assert armed.winner == 0


def test_inert_where_nobody_is_bare() -> None:
    """From chess's opening nobody is bare within a short game, so the bit changes nothing."""
    spec = _load("rulesets/chess.json")
    spec["rules"]["max_plies"] = 30
    engine = Engine(spec)
    on = play_game(engine, ("minimax:nodes=400,mopup=1",) * 2, seed=5)
    off = play_game(engine, ("minimax:nodes=400,mopup=0",) * 2, seed=5)
    assert on.moves == off.moves
    assert on.nodes == off.nodes


class _Watched:
    """The engine as the oracle's walk sees it, noting every position the walk walks from
    (each is asked for its legal moves) and every step it plays."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.walked: list[Any] = []
        self.played: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def legal_moves(self, state: Any) -> Any:
        self.walked.append(state)
        return self.engine.legal_moves(state)

    def apply(self, state: Any, move: int) -> Any:
        child = self.engine.apply(state, move)
        self.played.append(child)
        return child


def _royals(engine: Engine, state: Any, player: int) -> int:
    spec = engine.spec
    royal = set(spec.royal_types)
    return sum(
        1 for v in state.board if v and spec.owner_of(v) == player and spec.type_of(v) in royal
    )


# Kyoto, a drops game where the bare side's walk meets steps that take the standing king
# (Q1136, `docs/archive/bugs-resolved.md`): the first minimax's searches walk there.
_KYOTO_GAME = (("minimax:nodes=400", "greedy"), 2)


def test_the_walk_never_takes_the_standing_royal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q1138: a step that takes a royal of the side standing still is no step of the walk,
    so every position the walk walks from has one royal a side, as every position the rules
    reach does (`compile._royalty_can_multiply`), and the core's one-royal hint holds there.
    The game below offers such steps (some are played and refused), and the core walks as
    the oracle does: the same moves and nodes."""
    engine = Engine(_load("archive/kyoto.json"))
    watched: list[_Watched] = []
    walk = MinimaxAgent._bare_reach

    def watching(self: MinimaxAgent, state: Any, bare: int, cap: int) -> int:
        plain = self.engine
        self.engine = _Watched(plain)
        try:
            return walk(self, state, bare, cap)
        finally:
            watched.append(self.engine)
            self.engine = plain

    monkeypatch.setattr(MinimaxAgent, "_bare_reach", watching)
    agents, seed = _KYOTO_GAME
    in_python = play_game_python(engine, agents, seed=seed)
    native = play_game(engine, agents, seed=seed)
    assert native.moves == in_python.moves
    assert native.nodes == in_python.nodes
    walked = [s for w in watched for s in w.walked]
    played = [s for w in watched for s in w.played]
    assert walked
    assert any(_royals(engine, s, 0) == 0 or _royals(engine, s, 1) == 0 for s in played)
    for state in walked:
        assert _royals(engine, state, 0) == 1 and _royals(engine, state, 1) == 1


@pytest.mark.skipif(not getattr(_core, "DEBUG_ASSERTIONS", False), reason="a checking build")
@pytest.mark.parametrize("seed", range(6))
def test_kyoto_plays_clean_on_a_checking_build(seed: int) -> None:
    """On a core built with debug assertions (`.claude/rules/engine-and-spec.md`, *Royal hint
    and check memo*), the games whose searches walk past the standing king run without the
    royal hint's assert firing (Q1136 found it firing in the first)."""
    engine = Engine(_load("archive/kyoto.json"))
    for agents in (("greedy", "minimax:nodes=400"), ("minimax:nodes=400", "greedy")):
        assert play_game(engine, agents, seed=seed).moves
