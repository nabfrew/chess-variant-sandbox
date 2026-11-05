"""Q1107: a game the core reads as over at ply 0 (`docs/reference/fold-terminal-at-ply-zero.md`).

Q1101's scratch spec (an unmirrored setup with a fold portal and a blast) read as
`royal_captured` at ply 0. The fold is not the cause: the opening array puts the second
player's royal under the first player's attack with the first player to move, which is the
position `terminal_prefix`'s illegal-position guard (`rust/src/game.rs`) exists to end. The
core is right; what is missing is a gate that says so before anything is played. The spec
below is the smallest such game, with no fold and no blast.
"""

from __future__ import annotations

import copy
from typing import Any

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.loop.gates import smoke_test
from chess_variant_sandbox.spec.schema import validate_spec

# Player 0's slider on a1 sees player 1's royal on a3 up the a-file, and player 0 moves first.
_OVER_AT_THE_START: dict[str, Any] = {
    "name": "Q1107: the side to move starts attacking the other royal",
    "board": {"width": 3, "height": 3},
    "pieces": [
        {
            "id": "K",
            "royal": True,
            "value": 0,
            "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        },
        {
            "id": "R",
            "value": 5,
            "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
        },
    ],
    "setup": {
        "mirror": False,
        "placements": [
            {"piece": "K", "player": 0, "squares": ["c1"]},
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["a3"]},
        ],
    },
    "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 40},
}


def test_the_illegal_position_guard_ends_it_without_any_fold() -> None:
    """The diagnosis, asked of the engine: no portal, no blast, and still over at ply 0,
    because player 1 is in check on player 0's turn and player 0 is not."""
    spec = copy.deepcopy(_OVER_AT_THE_START)
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.initial_state()
    assert engine.in_check(state, 1) and not engine.in_check(state, 0)
    outcome = engine.terminal(state)
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (0, "royal_captured")


def test_moving_the_attacker_off_the_line_plays() -> None:
    """The same game with the slider off the a-file is not over: the verdict is the attack."""
    spec = copy.deepcopy(_OVER_AT_THE_START)
    spec["setup"]["placements"][1]["squares"] = ["b1"]
    engine = Engine(spec)
    assert engine.terminal(engine.initial_state()) is None


def test_the_smoke_gate_names_a_game_over_before_the_first_move() -> None:
    """Q1109: the first rung asks the engine for the opening's verdict and refuses by name,
    before any playout, rather than letting 100% of games end at ply 0 read as *"decided
    before it begins"*."""
    result = smoke_test(copy.deepcopy(_OVER_AT_THE_START), games=4, workers=1)
    assert not result.passed
    assert any("over before the first move" in reason for reason in result.reasons)
    assert any("royal piece open to capture" in reason for reason in result.reasons)
    assert any("first side has already won" in reason for reason in result.reasons)
    assert not any("decided before it begins" in reason for reason in result.reasons)


def test_the_same_game_off_the_line_is_not_refused_as_over() -> None:
    """The new refusal is the verdict's, not the spec's shape: off the a-file, no such reason."""
    spec = copy.deepcopy(_OVER_AT_THE_START)
    spec["setup"]["placements"][1]["squares"] = ["b1"]
    result = smoke_test(spec, games=4, workers=1)
    assert not any("over before the first move" in reason for reason in result.reasons)
