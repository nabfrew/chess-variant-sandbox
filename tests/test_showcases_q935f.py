"""The spy's showcases (Q935f), held to G1 of `docs/plan-spy.md` §5.6: the mechanism, without players.

§1.6's three claims, which say why the study's verdict comes out as it does, are shown here as
rules rather than as habits of the computer players, on hand-built positions and on seeded
random play (20 games a showcase, `random.Random(seed)` choosing from the core's legal moves):

1. **The answerer cannot lose a piece that can move.** In every announced round of SP1
   (`archive/gardner-with-spies`) whose announced move ends on a square an answerer piece
   stands on, every answer that moves that piece to an empty square leaves it standing where
   it went.
2. **The announcer cannot recapture** in SP1: the taker of a guarded piece steps away from the
   announced recapture.
3. **`rest` makes the recapture land** in SP2 (`archive/gardner-with-spies-resting`): the taker
   rests, has no move, and every answer leaves it to be taken.

Also held: the study's files are what §5.1 says (SP2 is SP1 with `rest`; the twin T0 and the V5
arms are built here from SP1, and their hashes are the ones the study's runs carry, since
those drafts live in the gitignored `runs/`); SP1's alternating phase plays move for move as
T0 does; and a spy that meets another piece in a round is the one removed, as the
descriptions say.

Every answer is the core's, through `Engine`; the clauses compare squares and owners on the
boards it returns.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec

ROOT = Path(__file__).parent.parent
SEEDS = range(20)
MAX_STEPS = 120


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


SP1 = load("gardner-with-spies")
SP2 = load("gardner-with-spies-resting")
PARENT = load("gardner-in-rounds")


def twin(spec: dict[str, Any]) -> dict[str, Any]:
    """§5.1's T0: the file without the spy flag and without `rules.simultaneous`, the piece
    named for what it then is (a piece's name is mechanics: it is in the hash)."""
    out = copy.deepcopy(spec)
    out.pop("provenance", None)
    for piece in out["pieces"]:
        if piece.pop("spy", None):
            piece["name"] = "wazir"
    del out["rules"]["simultaneous"]
    return out


def valued(spec: dict[str, Any], value: int) -> dict[str, Any]:
    """§5.1's V5 arm: the spy's declared value changed."""
    out = copy.deepcopy(spec)
    out.pop("provenance", None)
    for piece in out["pieces"]:
        if piece.get("spy"):
            piece["value"] = value
    return out


# The mechanics hashes the study's runs carry (`plan-spy.md` §5.7).
STUDY_HASHES = {
    "SP1": "f4c62a33798d",
    "SP2": "1d798bedbf66",
    "T0": "436ef2d9739d",
    "V5 SP1": "545a64900418",
    "V5 SP2": "7dce453b8635",
}

_ENGINES: dict[str, Engine] = {}


def engine_of(name: str) -> Engine:
    if name not in _ENGINES:
        _ENGINES[name] = Engine({"SP1": SP1, "SP2": SP2, "T0": twin(SP1)}[name])
    return _ENGINES[name]


def square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def find(engine: Engine, moves: list[int], origin: str, target: str) -> int:
    a, b = square(engine, origin), square(engine, target)
    found = [m for m in moves if decode_move(m)[:2] == (a, b)]
    assert len(found) == 1, (origin, target)
    return found[0]


def owner(engine: Engine, state: GameState, at: int) -> int | None:
    value = state.board[at]
    return engine.spec.owner_of(value) if value else None


# ------------------------------------------------------------------ the files


def test_the_study_files_are_section_five_one() -> None:
    for spec in (SP1, SP2):
        assert not validate_spec(spec)
        assert spec["schema_version"] == 27
        spies = [p for p in spec["pieces"] if p.get("spy")]
        assert len(spies) == 1
        spy = spies[0]
        # A wazir that captures nothing, worth 2, on c2 (c4 by the mirror).
        assert spy["value"] == 2 and spy["captures"] == []
        assert spy["moves"] == [
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}
        ]
        assert {"piece": spy["id"], "squares": ["c2"]} in spec["setup"]["placements"]
        assert spec["setup"]["mirror"] is True
    # SP1 is the parent with the c-pawn a spy; SP2 is SP1 with the rest, and nothing else.
    assert SP1["rules"] == {**PARENT["rules"], "simultaneous": {}}
    assert SP2["rules"] == {**SP1["rules"], "simultaneous": {"rest": True}}
    assert SP1["pieces"] == SP2["pieces"] and SP1["setup"] == SP2["setup"]
    assert SP1["pieces"][:-1] == PARENT["pieces"]
    built = {
        "SP1": SP1, "SP2": SP2, "T0": twin(SP1),
        "V5 SP1": valued(SP1, 5), "V5 SP2": valued(SP2, 5),
    }
    for name, spec in built.items():
        assert not validate_spec(spec), name
        assert spec_hash(spec) == STUDY_HASHES[name], name


# --------------------------------------------------------- random play through the core


def walk(engine: Engine, seed: int) -> list[tuple[GameState, str, tuple[int, ...], GameState]]:
    """One seeded random game: each step's position, phase, moves and position after."""
    rng = random.Random(seed)
    state = engine.initial_state()
    steps = []
    for _ in range(MAX_STEPS):
        if engine.terminal(state) is not None:
            break
        phase = engine.phase(state)
        if phase.is_round:
            moves = (
                rng.choice(engine.legal_moves_for(state, 0)),
                rng.choice(engine.legal_moves_for(state, 1)),
            )
            after = engine.apply_round(state, moves[0], moves[1])
        else:
            moves = (rng.choice(engine.legal_moves(state)),)
            after = engine.apply(state, moves[0])
        steps.append((state, phase.name, moves, after))
        state = after
    return steps


def test_sp1_alternates_move_for_move_as_its_twin() -> None:
    """While both spies stand, SP1 is T0: the same legal moves and the same boards."""
    spy, plain = engine_of("SP1"), engine_of("T0")
    compared = 0
    for seed in SEEDS:
        for state, phase, moves, after in walk(spy, seed):
            if phase != "alternating":
                break
            # The two specs compile to the same board encoding (the spy flag adds no type),
            # so a position of one is a position of the other.
            assert sorted(spy.legal_moves(state)) == sorted(plain.legal_moves(state))
            assert plain.apply(state, moves[0]).board == after.board
            compared += 1
    assert compared > 100


def test_claim_one_an_answerer_piece_that_moves_away_survives_in_random_play() -> None:
    """§1.6's first claim over every announced strike at a standing answerer piece in 20
    random SP1 games, against every answer that moves the target to an empty square."""
    engine = engine_of("SP1")
    strikes = answers = 0
    for seed in SEEDS:
        for state, phase, _moves, _after in walk(engine, seed):
            if phase != "announced":
                continue
            announcer = engine.phase(state).announcer
            answerer = 1 - announcer
            mine = engine.legal_moves_for(state, announcer)
            theirs = engine.legal_moves_for(state, answerer)
            for strike in mine:
                target = move_to(strike)
                if owner(engine, state, target) != answerer:
                    continue
                strikes += 1
                for answer in theirs:
                    if move_from(answer) != target or state.board[move_to(answer)]:
                        continue
                    pair = (strike, answer) if announcer == 0 else (answer, strike)
                    after = engine.apply_round(state, *pair)
                    assert owner(engine, after, move_to(answer)) == answerer, seed
                    answers += 1
    assert strikes > 50 and answers > 50


# ---------------------------------------------------- the guarded pawn, hand-built (§1.6)

# Black holds the spy and is to move, so the round is announced and White announces. White's
# a-pawn is guarded by the rook; Black's knight takes it while White's king steps aside.
WHITE = {"e1": "K", "a1": "R", "a3": "P"}
BLACK = {"e5": "K", "b5": "N", "c5": "S"}


def guarded(engine: Engine) -> GameState:
    placement = {sq: (piece, 0) for sq, piece in WHITE.items()}
    placement.update({sq: (piece, 1) for sq, piece in BLACK.items()})
    state = engine.make_state(placement, side_to_move=1)
    phase = engine.phase(state)
    assert phase.name == "announced" and phase.announcer == 0
    taken = engine.apply_round(
        state,
        find(engine, engine.legal_moves_for(state, 0), "e1", "d1"),
        find(engine, engine.legal_moves_for(state, 1), "b5", "a3"),
    )
    a3 = square(engine, "a3")
    assert owner(engine, taken, a3) == 1 and engine.spec.piece_ids[
        engine.spec.type_of(taken.board[a3])
    ] == "N"
    assert engine.phase(taken).name == "announced"
    return taken


def test_claim_two_without_rest_the_taker_steps_away_from_the_recapture() -> None:
    engine = engine_of("SP1")
    state = guarded(engine)
    recapture = find(engine, engine.legal_moves_for(state, 0), "a1", "a3")
    knight = [m for m in engine.legal_moves_for(state, 1) if move_from(m) == square(engine, "a3")]
    assert knight, "the knight may move again: nothing rests in SP1"
    for answer in knight:
        after = engine.apply_round(state, recapture, answer)
        assert owner(engine, after, move_to(answer)) == 1
        assert owner(engine, after, square(engine, "a3")) == 0  # the rook landed on nothing


def test_claim_three_with_rest_the_recapture_lands() -> None:
    engine = Engine(SP2)
    state = guarded(engine)
    a3 = square(engine, "a3")
    recapture = find(engine, engine.legal_moves_for(state, 0), "a1", "a3")
    answers = engine.legal_moves_for(state, 1)
    assert answers and all(move_from(m) != a3 for m in answers), "the knight rests"
    for answer in answers:
        after = engine.apply_round(state, recapture, answer)
        assert owner(engine, after, a3) == 0, engine.describe_move(state, answer)
        assert not any(
            value and engine.spec.owner_of(value) == 1
            and engine.spec.piece_ids[engine.spec.type_of(value)] == "N"
            for value in after.board
        )


@pytest.mark.parametrize("name", ["SP1", "SP2"])
def test_a_spy_that_meets_a_piece_is_the_one_removed(name: str) -> None:
    engine = engine_of(name)
    state = engine.make_state(
        {"a1": ("K", 0), "c2": ("P", 0), "e5": ("K", 1), "c4": ("S", 1), "a5": ("N", 1)},
        side_to_move=1,
    )
    assert engine.phase(state).name == "announced"
    pawn = find(engine, engine.legal_moves_for(state, 0), "c2", "c3")
    spy = find(engine, engine.legal_moves_for(state, 1), "c4", "c3")
    after = engine.apply_round(state, pawn, spy)
    c3 = square(engine, "c3")
    assert owner(engine, after, c3) == 0
    assert engine.phase(after).name == "blind"
