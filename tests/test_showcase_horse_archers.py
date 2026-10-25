"""A2: *the horse archers*, the shooting capture's showcase (`docs/plan-pieces-moves.md` §6.7).

Chess whose knights shoot: `rules.strike` at rifle's reading with `by: ["N"]`, registered over
*the archers* (the bishops) by the fallback rule written before the draft read. This is V1 on
the file itself: the core's legal list equals the twin-type contact spelling's after every move
of seeded random games, and after every shot the knight stands on its origin and the target
square is empty; H1 to H4 of §6.1 on the file's own pieces (the knight the shooter); a pawn that
promotes to a knight shooting on its next capture; Fairy-Stockfish refusing the file.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec, validate_spec
from tests.test_shooting_capture import _at, _between, _one, _position, _square

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "archive" / "horse-archers.json"


def _file() -> dict[str, Any]:
    return copy.deepcopy(load_spec(FILE))


def _plain() -> dict[str, Any]:
    spec = _file()
    spec["rules"].pop("strike")
    return spec


def test_the_file_is_chess_with_the_knights_shooting() -> None:
    spec = _file()
    chess = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
    assert spec["rules"]["strike"] == {
        "effect": "capture", "position": "stay", "immune": ["K"], "by": ["N"],
    }
    assert _plain()["rules"] == chess["rules"]
    assert spec["pieces"] == chess["pieces"] and spec["setup"] == chess["setup"]
    assert spec["appearance"] == {"icons": {"N": "arrow"}}


class TestTheHandPositions:
    """§6.1's H1 to H4 with the knight the shooter, on the file's rules."""

    def test_h1_the_knight_shoots_and_the_bishop_takes(self) -> None:
        engine = Engine(_position(_file(), {"c2": "N", "c1": "B", "h1": "K"},
                                  {"e3": "N", "f4": "P", "h8": "K"}))
        start = engine.initial_state()
        shot = _one(engine, start, "c2", "e3")
        assert _at(engine, shot, "c2") == (0, "N") and _at(engine, shot, "e3") is None
        assert "(stays)" in engine.describe_move(start, _between(engine, start, "c2", "e3")[0])
        taken = _one(engine, start, "c1", "e3")
        assert _at(engine, taken, "e3") == (0, "B") and _at(engine, taken, "c1") is None

    def test_h2_a_pinned_knight_still_shoots(self) -> None:
        placed = ({"e2": "N", "e1": "K"}, {"e6": "R", "c3": "N", "g3": "N", "h8": "K"})
        engine = Engine(_position(_file(), *placed))
        start = engine.initial_state()
        for target in ("c3", "g3"):
            after = _one(engine, start, "e2", target)
            assert _at(engine, after, "e2") == (0, "N") and _at(engine, after, target) is None
        assert _between(engine, start, "e2", "d4") == []  # a walk off the pin line is not
        chess = Engine(_position(_plain(), *placed))
        assert all(move_from(m) != _square("e2") for m in chess.legal_moves(chess.initial_state()))

    def test_h3_a_shot_answers_a_check_only_by_removing_the_checker(self) -> None:
        engine = Engine(_position(_file(), {"d6": "N", "e1": "K"}, {"e8": "R", "h8": "K"}))
        start = engine.initial_state()
        shot = _one(engine, start, "d6", "e8")
        assert _at(engine, shot, "d6") == (0, "N") and _at(engine, shot, "e8") is None
        block = _one(engine, start, "d6", "e4")
        assert _at(engine, block, "e4") == (0, "N")
        assert _between(engine, start, "d6", "b5") == []

    def test_h4_a_shot_may_not_open_a_line_onto_its_own_royal(self) -> None:
        placed = ({"c4": "N", "e1": "K"}, {"e5": "N", "e8": "R", "a8": "K"})
        archers = Engine(_position(_file(), *placed))
        assert _between(archers, archers.initial_state(), "c4", "e5") == []
        chess = Engine(_position(_plain(), *placed))
        assert len(_between(chess, chess.initial_state(), "c4", "e5")) == 1

    def test_a_pawn_that_promotes_to_a_knight_shoots_from_its_next_capture(self) -> None:
        engine = Engine(_position(_file(), {"b7": "P", "e1": "K"},
                                  {"a8": "R", "c7": "P", "h8": "K"}))
        start = engine.initial_state()
        ids = engine.spec.piece_ids
        promote = [
            m for m in _between(engine, start, "b7", "a8")
            if ids[engine.spec.type_of(engine.apply(start, m).board[_square("a8")])] == "N"
        ]
        assert len(promote) == 1  # the promoting capture is a pawn's, so the pawn lands
        state = engine.apply(start, promote[0])
        assert _at(engine, state, "a8") == (0, "N") and _at(engine, state, "b7") is None
        state = _one(engine, state, "h8", "g8")
        shot = _one(engine, state, "a8", "c7")
        assert _at(engine, shot, "a8") == (0, "N") and _at(engine, shot, "c7") is None


def _twin() -> dict[str, Any]:
    """The file in the language it had before `by`: two knight types, `N` and `N2` (appended
    last, so every other index and move word keeps its meaning), under contacts at `stay`,
    each turning into the other when it captures."""
    spec = _plain()
    spec.pop("appearance", None)
    knight = next(p for p in spec["pieces"] if p["id"] == "N")
    spec["pieces"].append({**copy.deepcopy(knight), "id": "N2", "name": "knight two"})
    spec["rules"]["contacts"] = {
        "position": "stay",
        "immune": ["K"],
        "links": [
            {"mover": mover, "onto": onto, "mover_to": other}
            for mover, other in (("N", "N2"), ("N2", "N"))
            for onto in ("P", "N", "N2", "B", "R", "Q")
        ],
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _agree(seed: int, plies: int = 160) -> tuple[int, int]:
    """One seeded random game of the file beside the twin: equal legal lists at every node,
    equal boards with `N2` read as `N`, and every shot leaving its knight home and its target
    square empty."""
    engine, twin = Engine(_file()), Engine(_twin())
    knight, second = twin.spec.piece_ids.index("N"), twin.spec.piece_ids.index("N2")
    n_file, n_twin = len(engine.spec.piece_ids), len(twin.spec.piece_ids)

    def plain(board: Any, n_types: int, rename: bool) -> tuple[Any, ...]:
        out: list[Any] = []
        for value in board:
            if value == 0:
                out.append(0)
                continue
            owner, kind = (value - 1) // n_types, (value - 1) % n_types
            out.append((owner, knight if rename and kind == second else kind))
        return tuple(out)

    rng = random.Random(seed)
    a, b = engine.initial_state(), twin.initial_state()
    nodes = shots = 0
    for _ in range(plies):
        moves = engine.legal_moves(a)
        assert sorted(moves) == sorted(twin.legal_moves(b)), (seed, nodes)
        assert plain(a.board, n_file, False) == plain(b.board, n_twin, True)
        nodes += 1
        if not moves or engine.terminal(a, moves) is not None or twin.terminal(b) is not None:
            break
        taking = [m for m in moves if a.board[move_to(m)]]
        move = rng.choice(taking) if taking and rng.random() < 0.5 else rng.choice(moves)
        origin, target = move_from(move), move_to(move)
        value = a.board[origin]
        is_shot = engine.spec.piece_ids[engine.spec.type_of(value)] == "N" and a.board[target]
        a, b = engine.apply(a, move), twin.apply(b, move)
        if is_shot:
            shots += 1
            assert a.board[origin] == value and a.board[target] == 0, (seed, nodes)
    return nodes, shots


class TestTheTwinTypeSpellingAgrees:
    """§6.7 V1: the file's legal list equals the contact spelling's, every node."""

    @pytest.mark.parametrize("seed", range(3))
    def test_three_games_agree(self, seed: int) -> None:
        nodes, _shots = _agree(seed)
        assert nodes > 10  # seed 0's random game is mated at its 14th node

    @pytest.mark.slow
    def test_twenty_games_agree_and_the_knights_shoot(self) -> None:
        total_nodes = total_shots = 0
        for seed in range(20):
            nodes, shots = _agree(seed)
            total_nodes += nodes
            total_shots += shots
        assert total_nodes > 1000 and total_shots > 20


def test_fairy_refuses_the_file_by_name() -> None:
    with pytest.raises(Untranslatable, match="strike"):
        translate(_file())
