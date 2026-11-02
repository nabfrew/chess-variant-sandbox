"""SR3e (`docs/plan-schema-next.md` §2.9, row K3): the duck untouched by a blast, a strike or
contacts.

The ruled reading: the duck is untouched by all three, as it is out of the capture matrix. A
blast beside it leaves it standing (the compiler puts its type in the blast's immune set, so the
core's blast and the legality filter's spare it alike), and no strike or contact reaches it (both
happen only on a capture, and nothing captures the duck). None of the three rules' lists may
name its type.

The oracle is this file's `_blast_after`: the board a blast leaves, read off the compiled blast
table with the duck spared (a declared test oracle, rule 3). Seeded walks hold every piece half
of a blast game to it, check the duck stands where its own half put it after every piece half
of every game, that no piece half offers a move onto the duck, and the incremental key against
the full scan every ply; each walk counts the blasts that reached the duck's square, so a walk
where none did fails. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_DUCK, move_from, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
BISHOP = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}

RULES: dict[str, dict[str, Any]] = {
    "blast": {"explode_on_capture": {"radius": 1}},
    "shaped blast": {"explode_on_capture": {"blast": {"vectors": [[1, 0], [2, 0]],
                                                      "symmetry": "rotate4"},
                                            "immune": ["N"]}},
    "rifle": {"strike": {"effect": "capture", "position": "stay", "immune": ["K"]}},
    "demotion": {"strike": {"effect": "demote", "chain": [{"from": "R", "to": "B"},
                                                          {"from": "B", "to": "N"}],
                            "position": "stay", "immune": ["K"]}},
    "contacts": {"contacts": {"links": [{"mover": "N", "onto": "R", "onto_to": "N"},
                                        {"mover": "B", "onto": "N", "mover_to": "R"}],
                              "position": "swap", "immune": ["K"]}},
}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _game(rules: dict[str, Any], goes: str = "anywhere", start: str | None = None) -> dict[str, Any]:
    """6x6, won by taking the king: a king, two rooks, a bishop and a knight a side, the duck."""
    duck: dict[str, Any] = {"piece": "D"}
    if goes != "anywhere":
        duck["goes"] = goes
    if start is not None:
        duck["start"] = start
    return copy.deepcopy({
        "name": "the duck in the blast (SR3e test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "B", "value": 3, "moves": [BISHOP]},
            {"id": "N", "value": 3, "moves": [KNIGHT]},
            {"id": "D", "value": 0, "moves": [KING] if goes == "by_moves" else []},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "R", "player": 0, "squares": ["b1", "f1"]},
            {"piece": "B", "player": 0, "squares": ["c1"]},
            {"piece": "N", "player": 0, "squares": ["e1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "R", "player": 1, "squares": ["e6", "a6"]},
            {"piece": "B", "player": 1, "squares": ["d6"]},
            {"piece": "N", "player": 1, "squares": ["b6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
                  "duck": duck, **copy.deepcopy(rules)},
    })


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _duck_value(engine: Engine) -> int:
    duck = engine.spec.duck
    assert duck is not None
    return WILD * engine.spec.side_stride + duck.piece_type + 1


def _blast_after(engine: Engine, before: GameState, move: int) -> bytes | None:
    """The oracle: the board a move leaves under the blast, the duck spared, or ``None`` where
    the move sets no blast off (an ordinary move, which this oracle does not restate)."""
    spec = engine.spec
    explosion = spec.explosion
    assert explosion is not None
    origin, target = move_from(move), move_to(move)
    victim = before.board[target]
    if move_type(move) == MOVE_DUCK or not victim or spec.owner_of(victim) == WILD:
        return None
    mover = spec.type_of(before.board[origin])
    if not ((explosion.by_types is None or mover in explosion.by_types)
            and (explosion.of_types is None or spec.type_of(victim) in explosion.of_types)):
        return None
    board = bytearray(before.board)
    board[origin] = 0
    board[target] = 0
    duck = _duck_value(engine)
    for square in explosion.blast[target]:
        value = board[square]
        if value and value != duck and spec.type_of(value) not in explosion.immune_types:
            board[square] = 0
    return bytes(board)


def _walk(spec: dict[str, Any], seeds: range, plies: int) -> dict[str, int]:
    """Seeded walks leaning toward captures; returns what they met."""
    engine = _engine(spec)
    duck = _duck_value(engine)
    met = {"captures": 0, "blasts by the duck": 0}
    blast = engine.spec.explosion.blast if engine.spec.explosion is not None else None
    for seed in seeds:
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(plies):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            where = engine.duck_square(state)
            piece_half = not engine.duck_owed(state)
            if piece_half and where is not None:
                assert all(move_to(m) != where for m in moves), (seed, state.ply)
            takes = [m for m in moves if piece_half and state.board[move_to(m)]
                     and engine.spec.owner_of(state.board[move_to(m)]) != WILD]
            # A duck half leans toward a square beside pieces, where a blast may reach it.
            near = [m for m in moves if not piece_half and blast is not None
                    and any(state.board[s] for s in blast[move_to(m)])]
            leaning = takes or near
            move = rng.choice(leaning) if leaning and rng.random() < 0.7 else rng.choice(moves)
            after = engine.apply(state, move)
            assert after.key == engine._compute_hash(after), (seed, state.ply)
            if piece_half:
                if where is not None:
                    assert after.board[where] == duck, (seed, state.ply)
                    assert after.board.count(bytes([duck])) == 1
                if move in takes:
                    met["captures"] += 1
                if engine.spec.explosion is not None:
                    expected = _blast_after(engine, state, move)
                    if expected is not None:
                        assert after.board == expected, (seed, state.ply)
                        if where is not None and where in engine.spec.explosion.blast[move_to(move)]:
                            met["blasts by the duck"] += 1
            state = after
    return met


class TestValid:
    @pytest.mark.parametrize("name", list(RULES))
    def test_each_validates_and_the_core_takes_it(self, name: str) -> None:
        spec = _game(RULES[name])
        assert validate_spec(spec) == []
        assert rust_supports(compile_spec(spec)) is None

    @pytest.mark.parametrize("rule, key, value", [
        ("explode_on_capture", "immune", ["D"]),
        ("explode_on_capture", "by", ["R", "D"]),
        ("explode_on_capture", "of", ["D"]),
        ("strike", "immune", ["K", "D"]),
    ])
    def test_a_list_naming_the_duck_is_refused(self, rule: str, key: str, value: list[str]) -> None:
        rules = copy.deepcopy(RULES["blast" if rule == "explode_on_capture" else "rifle"])
        rules[rule][key] = value
        errors = validate_spec(_game(rules))
        assert any(f"is named in rules.{rule}.{key}" in error for error in errors), errors

    def test_a_link_naming_the_duck_is_refused(self) -> None:
        rules = copy.deepcopy(RULES["contacts"])
        rules["contacts"]["links"].append({"mover": "R", "onto": "B", "onto_to": "D"})
        errors = validate_spec(_game(rules))
        assert any("is named in rules.contacts.links" in error for error in errors), errors

    def test_the_compiler_spares_the_duck(self) -> None:
        compiled = compile_spec(_game(RULES["shaped blast"]))
        assert compiled.explosion is not None and compiled.duck is not None
        assert compiled.explosion.immune_types == {compiled.duck.piece_type,
                                                   compiled.piece_ids.index("N")}

    def test_the_core_refuses_a_blast_that_reaches_the_duck(self) -> None:
        compiled = compile_spec(_game(RULES["blast"]))
        assert compiled.explosion is not None
        bare = dataclasses.replace(compiled, explosion=dataclasses.replace(
            compiled.explosion, immune_types=frozenset()))
        with pytest.raises(ValueError, match="a blast that reaches the duck"):
            Engine(bare)

    def test_describe_says_so(self) -> None:
        assert "A blast beside the D leaves it standing." in " ".join(
            describe_spec(_game(RULES["blast"])).split())
        assert "No strike reaches the D" in " ".join(describe_spec(_game(RULES["rifle"])).split())
        assert "No contact reaches the D" in " ".join(describe_spec(_game(RULES["contacts"])).split())


class TestTheBlast:
    @pytest.mark.parametrize("side", [0, 1])
    def test_a_blast_beside_the_duck_leaves_it_standing(self, side: int) -> None:
        """A rook on b1 takes up the file on b4; the duck on c4 and a knight on a4 stand beside
        it: the knight goes, the duck stands."""
        engine = _engine(_game(RULES["blast"]))
        state = engine.make_state({"a1": ("K", side), "f6": ("K", 1 - side), "b1": ("R", side),
                                   "b4": ("B", 1 - side), "a4": ("N", 1 - side), "c4": ("D", 2)},
                                  side_to_move=side, moved=["a1", "f6", "b1", "b4", "a4"])
        move = next(m for m in engine.legal_moves(state)
                    if move_from(m) == _sq("b1") and move_to(m) == _sq("b4"))
        after = engine.apply(state, move)
        assert after.board[_sq("c4")] == _duck_value(engine)
        assert not after.board[_sq("a4")] and not after.board[_sq("b4")]
        assert after.board == _blast_after(engine, state, move)
        assert engine.duck_owed(after) and engine.duck_square(after) == _sq("c4")

    @pytest.mark.parametrize("name", ["blast", "shaped blast"])
    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_walks(self, name: str, goes: str) -> None:
        met = _walk(_game(RULES[name], goes), range(40), 120)
        assert met["blasts by the duck"] > 0, met

    def test_perft_matches_an_apply_count(self) -> None:
        engine = _engine(_game(RULES["blast"], start="c3"))
        state = engine.initial_state()

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)


class TestTheStrikeAndContacts:
    @pytest.mark.parametrize("name", ["rifle", "demotion", "contacts"])
    @pytest.mark.parametrize("side", [0, 1])
    def test_nothing_reaches_the_duck(self, name: str, side: int) -> None:
        """A rook on b1 and a knight on a2 both bear on the duck on b3 (the knight's leap, the
        rook's file): neither is offered it."""
        engine = _engine(_game(RULES[name]))
        state = engine.make_state({"a1": ("K", side), "f6": ("K", 1 - side), "b1": ("R", side),
                                   "a2": ("N", side), "b3": ("D", 2)},
                                  side_to_move=side, moved=["a1", "f6", "b1", "a2"])
        assert all(move_to(m) != _sq("b3") for m in engine.legal_moves(state))

    @pytest.mark.parametrize("name", ["rifle", "demotion", "contacts"])
    @pytest.mark.parametrize("goes", ["anywhere", "by_moves"])
    def test_walks(self, name: str, goes: str) -> None:
        met = _walk(_game(RULES[name], goes), range(40), 120)
        assert met["captures"] > 0, met
