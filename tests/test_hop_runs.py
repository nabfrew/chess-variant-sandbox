"""TS4 (`docs/plan-tessabi.md` §4): a hop over a run (``moves[].run``) and displacement
(``moves[].displace``).

The core's moves and positions are held to `tests/hop_run_oracle.py`, which reads the spec by
file and rank: every offered move, every board a displacement leaves, and the undo (perft,
which makes and unmakes, against a count built from fresh applies; the incremental key against
`_compute_hash` at every ply). The keys are inert at their off values, refused where the first
build has no reading, and a displacement makes no type one-way.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_DISPLACE,
    MOVE_NORMAL,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from tests import hop_run_oracle as oracle


def _frog(**hop: Any) -> dict[str, Any]:
    return {
        "id": "F", "name": "frog", "value": 1, "captures": [],
        "moves": [
            {"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "lateral", "mode": "move"},
            {"kind": "hop", "vectors": [[0, 1], [1, 1], [1, 0]], "symmetry": "lateral",
             "mode": "move", **hop},
        ],
    }


def _spec(*pieces: dict[str, Any], walls: list[str] | None = None) -> dict[str, Any]:
    rules: dict[str, Any] = {
        "win": "none", "stalemate": "loss", "max_plies": 200,
        "reach": {"piece": "F", "squares": ["a7", "b7", "c7", "d7", "e7", "f7"]},
    }
    if walls:
        rules["obstacles"] = {"squares": walls}
    return {
        "name": "hop run probe",
        "board": {"width": 6, "height": 7},
        "pieces": [*pieces, {"id": "S", "name": "stone", "value": 1, "moves": []}],
        "setup": {"mirror": True, "placements": [
            {"piece": "F", "squares": ["a1", "b1", "c1", "d1", "e1", "f1", "b2", "c2", "e2"]},
            {"piece": "S", "player": "nobody", "squares": ["c4", "d4"]},
        ]},
        "rules": rules,
    }


# Each arm one reading: a run of any length that displaces, a capped run that only vaults, a
# plain hop that displaces (a leapfrog, landing anywhere up to two past its screen), and a run
# over the mover's own pieces only, beside walls.
SPECS = {
    "run displaces": _spec(_frog(run=True, displace="back")),
    "capped run": _spec(_frog(run=2)),
    "leapfrog": _spec(_frog(displace="back", screen_range=2)),
    "own run, walls": _spec(_frog(run=True, displace="back", screen="own"), walls=["a4", "f4"]),
}


def _decode(engine: Engine) -> Any:
    spec = engine.spec
    return lambda value: (spec.owner_of(value), spec.piece_ids[spec.type_of(value)])


def _word(move: int) -> oracle.Move:
    kind = {MOVE_NORMAL: "normal", MOVE_DISPLACE: "displace"}[move_type(move)]
    assert move_promotion(move) == 0
    return (move_from(move), move_to(move), kind)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_the_core_offers_and_plays_what_the_oracle_reads(name: str) -> None:
    """Every position of a few playouts: the core's moves are the oracle's, each displacement
    leaves the oracle's board, and the incremental key is the computed one."""
    spec = SPECS[name]
    assert validate_spec(spec) == []
    engine = Engine(spec)
    decode = _decode(engine)
    rng = random.Random(name)
    seen_displaced = 0
    for _ in range(4):
        state = engine.initial_state()
        for _ply in range(60):
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            words = [_word(m) for m in legal]
            assert len(set(words)) == len(words)
            assert set(words) == oracle.moves(spec, decode, state.board, state.side_to_move)
            for move in legal:
                after = engine.apply(state, move)
                assert after.board == oracle.after(spec, state.board, _word(move)), move
                if move_type(move) == MOVE_DISPLACE:
                    seen_displaced += 1
            state = engine.apply(state, rng.choice(legal))
            assert engine.hash(state) == engine._compute_hash(state)
    if "displace" in str(spec):
        assert seen_displaced > 0


def _walk(engine: Engine, state: Any, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(_walk(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state))


@pytest.mark.parametrize("name", sorted(SPECS))
def test_perft_unmakes_what_make_played(name: str) -> None:
    """Perft makes and unmakes in place; the walk applies afresh. Equal counts from a busy
    middle position say the unmake put every displaced piece back."""
    engine = Engine(SPECS[name])
    state = engine.initial_state()
    rng = random.Random(1)
    for _ in range(10):
        state = engine.apply(state, rng.choice(engine.legal_moves(state)))
    assert engine.perft(state, 3) == _walk(engine, state, 3)


def test_the_off_values_are_the_game_without_the_keys() -> None:
    """``run: false`` and ``displace: "none"`` name the plain hop's game and compile to its
    tables, which cross no run codes."""
    plain = _spec(_frog(screen_range=1))
    written = copy.deepcopy(plain)
    written["pieces"][0]["moves"][1].update(run=False, displace="none")
    assert validate_spec(written) == []
    assert spec_hash(written) == spec_hash(plain)
    from chess_variant_sandbox.engine.bridge import build_tables

    assert build_tables(compile_spec(written))["hop_runs"] == []
    assert build_tables(compile_spec(SPECS["run displaces"]))["hop_runs"] != []


def test_a_displacement_makes_no_type_one_way() -> None:
    """A displaced piece moves back toward the hopper's origin, so in a game with one, no type
    is one-way; the same game without it keeps its forward-only frog one-way."""
    def forward(**hop: Any) -> dict[str, Any]:
        frog = _frog(**hop)
        frog["moves"][1]["vectors"] = [[0, 1], [1, 1]]
        return _spec(frog)

    assert compile_spec(forward(run=True)).one_way_types
    assert not compile_spec(forward(run=True, displace="back")).one_way_types


@pytest.mark.parametrize(
    ("edit", "words"),
    [
        (lambda s: s["pieces"][0]["moves"][1].update(mode="both"), "needs mode 'move'"),
        (lambda s: s["pieces"][0]["moves"][0].update(run=True), "only applies to a hop"),
        (lambda s: s["pieces"][0]["moves"][1].update(screen_range=1), "screen_range beside run"),
        (lambda s: s["pieces"][0]["moves"][1].update(run=1), "run must be"),
        (lambda s: s["pieces"][0].update(royal=True), "beside a royal"),
        (lambda s: s["rules"].update(en_passant=True), "rules.en_passant"),
        (lambda s: s["pieces"][1].update(moves=[{"kind": "sweep", "vectors": [[1, 0]]}]),
         "a sweep"),
        (lambda s: s["rules"].update(wrap={"files": "both"}), "rules.wrap"),
        (lambda s: s["pieces"][0].update(rest=1), "pieces[].rest"),
    ],
)
def test_the_first_build_refuses_by_name(edit: Any, words: str) -> None:
    spec = copy.deepcopy(SPECS["run displaces"])
    edit(spec)
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


def test_a_displaced_piece_is_named_in_the_move_text() -> None:
    engine = Engine(SPECS["run displaces"])
    state = engine.initial_state()
    texts = engine.move_texts(state)
    assert "Fb1-b3 (displaces b2)" in texts
    move = engine.find_move(state, "Fb1-b3 (displaces b2)")
    assert engine.spec.displace_line(state.board, move) == [(7, 1)]  # b2 onto b1
