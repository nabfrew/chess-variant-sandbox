"""SR7b (`docs/plan-schema-next.md` §2.2 row L3, ruled 2026-10-27): wild pieces in a long turn.

The ruled reading: a phase follows every *move*, as it does in a turn of one (*"a phase follows
every move"*), so no legality filter needs to know whether the move it is testing ends the turn.
L3 refused the pair; the probe found the core already playing it so (`make` runs the phase
after every piece move, and the filter plays the move and the phase and asks the post-phase
question of the mover's royal), and the one reader that read the turn as alternating was the
board's drawing: `Core.coming_phases` gave a pouncer's second phase to the other player, where
in a long turn the side to move makes the next move too (`pounce.rs::pounce_coming_phases`).

The oracle is the one-move twin (a declared test oracle, rule 3): the same spec without
`rules.turn`, asked about the same position with the same side to move. Because the filter
reads no turn boundary, the long turn's legal list, its check, and each move's phase are the
twin's, at every position of a seeded walk, mid-turn included. The drawn phases are the twin's
phase played on the board the one before left, each with the mover the turn's schedule names.
The fixtures are the wild-piece files the earlier generations built, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.state import GameState, PhaseStep
from chess_variant_sandbox.spec.schema import validate_spec

HERE = Path(__file__).parent
ROOT = HERE.parent

# Every kind of wild piece the core has a phase for: a route (and its shield, sight), a
# respawn, a line that turns back, a pouncer on prey and on attackers, and a splitting one.
FIXTURES = {
    "wolves": ROOT / "archive" / "wolves-patrol.json",
    "respawn": HERE / "_respawn_wolf_hunt.json",
    "turning": HERE / "_turning_wolves.json",
    "sight": HERE / "_sight_wild_wolves.json",
    "hounds": HERE / "_pounce_hounds.json",
    "guards": HERE / "_pounce_guards.json",
    "hydra": HERE / "_pounce_hydra.json",
}
POUNCING = ("hounds", "guards", "hydra")
TURNS = ({"moves": 2, "ramp": [1]}, {"moves": 3})


def _one(name: str) -> dict[str, Any]:
    return json.loads(FIXTURES[name].read_text(encoding="utf-8"))


def _long(name: str, turn: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(_one(name))
    spec["rules"]["turn"] = copy.deepcopy(turn)
    return spec


def _engines(name: str, turn: dict[str, Any]) -> tuple[Engine, Engine]:
    spec = _long(name, turn)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec)), Engine(compile_spec(_one(name)))


def _twin_state(twin: Engine, state: GameState, side: int | None = None,
                board: bytes | None = None) -> GameState:
    """`state` as the one-move twin holds it: a turn of one move, the same position."""
    out = dataclasses.replace(
        state, remaining=1, turns_taken=0, repetition=(),
        side_to_move=state.side_to_move if side is None else side,
        board=state.board if board is None else board,
    )
    return dataclasses.replace(out, key=twin._compute_hash(out))


def _length(turn: dict[str, Any], index: int) -> int:
    ramp = turn.get("ramp", [])
    return ramp[index] if index < len(ramp) else turn["moves"]


def _steps(steps: tuple[PhaseStep, ...]) -> list[dict[str, Any]]:
    return [step.as_dict() for step in steps]


class TestTheValidatorLiftsTheRefusal:
    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("turn", TURNS, ids=["two-after-one", "three"])
    def test_every_wild_fixture_validates_in_a_long_turn(self, name: str, turn: dict) -> None:
        assert validate_spec(_long(name, turn)) == []

    def test_no_refusal_names_the_pair(self) -> None:
        from chess_variant_sandbox.spec.refusals import REFUSALS

        assert "L3" not in {refusal.id for refusal in REFUSALS}


class TestAPhaseAfterEveryMove:
    def test_both_moves_of_a_turn_are_followed_by_a_phase(self) -> None:
        """The wolves step on every move of the patrol, so each move of a two-move turn,
        the turn's first and its last, carries a phase, and the side changes only at the
        turn's end."""
        engine, _ = _engines("wolves", {"moves": 2})
        state = engine.initial_state()
        rng = random.Random(3)
        for ply in range(12):
            first = state.remaining == 2
            move = rng.choice(engine.legal_moves(state))
            after, steps = engine.apply_with_phase(state, move)
            assert steps, f"ply {ply}: no phase after the turn's {'first' if first else 'last'} move"
            assert (after.side_to_move == state.side_to_move) == first, ply
            state = after


def _walk(name: str, turn: dict[str, Any], seeds: range, plies: int) -> tuple[int, int]:
    """Seeded games held to the twin at every position; returns (mid-turn plies, phases)."""
    engine, twin = _engines(name, turn)
    inside = phased = 0
    for seed in seeds:
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(plies):
            where = (name, seed, ply)
            assert state.key == engine._compute_hash(state), where
            if engine.terminal(state) is not None:
                break
            other = _twin_state(twin, state)
            moves = sorted(engine.legal_moves(state))
            assert moves == sorted(twin.legal_moves(other)), where
            assert engine.in_check(state) == twin.in_check(other), where
            inside += state.remaining != _length(turn, state.turns_taken)
            move = rng.choice(moves)
            after, steps = engine.apply_with_phase(state, move)
            twin_after, twin_steps = twin.apply_with_phase(other, move)
            assert after.board == twin_after.board, where
            assert _steps(steps) == _steps(twin_steps), where
            phased += bool(steps)
            state = after
    return inside, phased


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_seeded_walks_are_the_one_move_twins(name: str) -> None:
    """The legal list, the check and each move's phase are the twin's, mid-turn too."""
    inside, phased = _walk(name, {"moves": 2, "ramp": [1]}, range(2), 40)
    assert inside > 0 and phased > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(FIXTURES))
@pytest.mark.parametrize("turn", TURNS, ids=["two-after-one", "three"])
def test_long_seeded_walks_are_the_one_move_twins(name: str, turn: dict) -> None:
    inside, phased = _walk(name, turn, range(6), 80)
    assert inside > 0 and phased > 0


def _played(board: bytes, steps: tuple[PhaseStep, ...]) -> bytes:
    """The board a drawn phase leaves (pounce games: no returns)."""
    out = bytearray(board)
    for step in steps:
        assert step.kind != "return"
        value = out[step.origin]
        if step.kind != "copy":
            out[step.origin] = 0
        out[step.target] = value
    return bytes(out)


def _schedule(turn: dict[str, Any], state: GameState, count: int) -> list[int]:
    """Who makes each of the next `count` moves, read off the spec's `rules.turn`."""
    movers, side, left, turns = [], state.side_to_move, state.remaining, state.turns_taken
    for _ in range(count):
        movers.append(side)
        left -= 1
        if left == 0:
            turns += 1
            left = _length(turn, turns)
            side = 1 - side
    return movers


@pytest.mark.parametrize("name", POUNCING)
@pytest.mark.parametrize("turn", TURNS, ids=["two-after-one", "three"])
def test_the_drawn_phases_follow_the_turns_schedule(name: str, turn: dict) -> None:
    """`coming_phases`: each phase is the twin's on the board the one before left, the
    pouncers hunting the player who makes the move it follows. Mid-turn, the second is the
    mover's own again; the walk must meet a drawn second phase that differs from the
    alternating reading's."""
    engine, twin = _engines(name, turn)
    differs = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(40):
            if engine.terminal(state) is not None:
                break
            drawn = coming_phases(engine, state, 3)
            board = state.board
            for k, mover in enumerate(_schedule(turn, state, 3)):
                expected = coming_phases(twin, _twin_state(twin, state, mover, board), 1)[0]
                assert _steps(drawn[k]) == _steps(expected), (name, seed, ply, k)
                if k == 1 and mover == state.side_to_move:
                    alternating = coming_phases(
                        twin, _twin_state(twin, state, 1 - mover, board), 1)[0]
                    differs += _steps(alternating) != _steps(expected)
                board = _played(board, drawn[k])
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
    assert differs > 0, "no mid-turn position drew a second phase the alternation would not"


def test_a_turn_of_one_still_alternates() -> None:
    """Without a turn rule the drawn phases are unchanged: the mover alternates."""
    twin = Engine(compile_spec(_one("hounds")))
    state = twin.initial_state()
    rng = random.Random(1)
    for _ in range(20):
        drawn = coming_phases(twin, state, 2)
        first = coming_phases(twin, state, 1)[0]
        then = coming_phases(
            twin, _twin_state(twin, state, 1 - state.side_to_move, _played(state.board, first)), 1)[0]
        assert (_steps(drawn[0]), _steps(drawn[1])) == (_steps(first), _steps(then))
        state = twin.apply(state, rng.choice(twin.legal_moves(state)))


def test_perft_from_mid_turn_matches_an_apply_built_count() -> None:
    """The core's perft from inside a long turn, against `apply` (perft has no terminal
    test, so neither has the count)."""
    engine, _ = _engines("hounds", {"moves": 2})
    state = engine.apply(engine.initial_state(), engine.legal_moves(engine.initial_state())[0])
    assert state.remaining == 1

    def count(position: GameState, depth: int) -> int:
        if depth == 0:
            return 1
        return sum(count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position))

    assert engine.perft(state, 3) == count(state, 3)
