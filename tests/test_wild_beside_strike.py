"""SR4e (`docs/plan-schema-next.md` §2.4 rows T6 and T7): wild pieces beside a strike and
beside contacts.

The ruling (2026-10-27): every wild type is in `strike.immune` and `contacts.immune` without
being listed, and a wild piece's eat and a player's capture of a wild piece are ordinary
captures under both rules.

The probe (T7's row asks that the first test probe the wild phase; SR2's method, fed past the
validator on SR4b's build, both rules, five readings): the phase already ate ordinarily under
every one -- it writes its own steps and never reaches `strike_writes` -- so the eaten piece was
gone with nothing reborn, placed or changed. A player's capture of a wild piece was struck: under
rifle the rook stayed home and the wolf vanished, under Andernach the rook took the wolf and
changed sides, a demoting chain from the wolf's type left a knight *owned by the third block*
walking the route, and a contact link onto the wolf changed the mover. So the lift is the
compiler's alone: `_compile_strike` and `_compile_contacts` take the wild types into `immune`
(`wild`), no core change. The validator reads the same set (`schema.wild_type_ids`, threaded
into `strike_survives_anything` and its siblings) and refuses any strike or contact list that
names a wild type, as it does the duck's (SR3e).

Hand-built 6x6 specs, not chess: two kings, a rook and a stepper each, and a wolf on a loop
through the middle. The walks hold every ply to a twin without the rule wherever the move is
not a capture of a player's piece (the only meeting either rule still changes) and check the
incremental key against `_compute_hash`.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_sixty_seven_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
LOOP = ["b3", "c3", "d3", "e3", "e4", "d4", "c4", "b4"]
WILD = 2

RULES: dict[str, tuple[str, dict[str, Any]]] = {
    "rifle": ("strike", {"effect": "capture", "position": "stay", "immune": ["K"]}),
    "circe": ("strike", {"effect": "capture", "position": "home", "immune": ["K"]}),
    "andernach": ("strike", {"effect": "capture", "switches_sides": ["R", "N"], "immune": ["K"]}),
    "demote": ("strike", {"effect": "demote", "position": "stay", "immune": ["K"],
                          "chain": [{"from": "R", "to": "N"}]}),
    "contacts": ("contacts", {"position": "stay", "immune": ["K"], "links": [
        {"mover": "N", "onto": "R", "onto_to": "N"},
        {"mover": "R", "onto": "N", "mover_to": "N"},
    ]}),
}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _base() -> dict[str, Any]:
    return copy.deepcopy({
        "schema_version": 31, "name": "the wolf and the rifle",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
            {"id": "W", "name": "wolf", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "N", "squares": ["c2", "d2"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 80,
                  "npcs": [{"piece": "W", "route": list(LOOP)}]},
    })


def _spec(reading: str) -> dict[str, Any]:
    spec = _base()
    key, value = RULES[reading]
    spec["rules"][key] = copy.deepcopy(value)
    return spec


def _twin(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game with neither rule: what an ordinary capture does."""
    twin = copy.deepcopy(spec)
    twin["rules"].pop("strike", None)
    twin["rules"].pop("contacts", None)
    return twin


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _byte(engine: Engine, pid: str, player: int) -> int:
    return engine.spec.encode_piece(player, engine.spec.piece_ids.index(pid))


READINGS = sorted(RULES)


# --------------------------------------------------------------------------- the probe


class TestThePhase:
    """The first test, as T7's row asks: the wild phase's eat, then a player's capture of a
    wild piece, each under every reading, against the twin without the rule."""

    @pytest.mark.parametrize("reading", READINGS)
    def test_a_wild_eat_is_an_ordinary_capture(self, reading: str) -> None:
        engine, twin = Engine(_spec(reading)), Engine(_twin(_spec(reading)))
        # The wolf on b3 steps to c3 after White's quiet move and eats the guard there. Under
        # circe the guard's home (c5) is empty, so a rebirth would show; under rifle, a demotion
        # or a contact it would stay or change; under Andernach nothing changes sides.
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1),
            "c3": ("N", 1), "b3": ("W", WILD),
        }, moved=["a1", "a6", "f1", "f6", "c3"])
        move = _move(engine, state, "a1", "a2")
        after, steps = engine.apply_with_phase(state, move)
        assert [(s.origin, s.target, s.eaten) for s in steps] == [
            (_sq("b3"), _sq("c3"), _byte(engine, "N", 1))
        ]
        assert after.board[_sq("c3")] == _byte(engine, "W", WILD)
        assert after.board == twin.apply(state, move).board
        assert after.key == engine._compute_hash(after)

    @pytest.mark.parametrize("reading", READINGS)
    def test_a_players_capture_of_a_wild_piece_is_ordinary(self, reading: str) -> None:
        engine, twin = Engine(_spec(reading)), Engine(_twin(_spec(reading)))
        # A rook (a striker under every reading, Andernach's switcher) and a guard (the contact
        # table's other mover) each take a wolf standing on its route.
        for mover, origin, target in (("R", "f3", "e3"), ("N", "c2", "b3")):
            state = engine.make_state({
                "a1": ("K", 0), "a6": ("K", 1), origin: (mover, 0), target: ("W", WILD),
            }, moved=["a1", "a6", origin])
            move = _move(engine, state, origin, target)
            after = engine.apply(state, move)
            assert after.board[_sq(target)] == _byte(engine, mover, 0), (reading, mover)
            assert after.board[_sq(origin)] == 0
            assert after.board == twin.apply(state, move).board
            assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- the compiler


class TestTheTables:
    @pytest.mark.parametrize("reading", READINGS)
    def test_the_wild_type_is_immune(self, reading: str) -> None:
        compiled = compile_spec(_spec(reading))
        strike = compiled.strike
        assert strike is not None
        n = strike.n_types
        wolf = compiled.piece_ids.index("W")
        assert wolf in strike.immune_types
        for mover in range(n):
            cell = mover * n + wolf
            assert strike.onto_to[cell] == -1 and strike.mover_to[cell] == -1
            assert not (strike.mover_flips and strike.mover_flips[cell])

    def test_without_wild_pieces_the_tables_are_the_rules_own(self) -> None:
        """The wild set is empty with no `npcs`, so every stored strike compiles as it did."""
        spec = _spec("rifle")
        spec["rules"].pop("npcs")
        compiled = compile_spec(spec)
        assert compiled.strike is not None
        assert compiled.piece_ids.index("W") not in compiled.strike.immune_types


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("reading", READINGS)
    def test_the_pair_is_valid(self, reading: str) -> None:
        errors = validate_spec(_spec(reading))
        assert errors == []
        assert not {"T6", "T7"} & refused_ids(errors)

    @pytest.mark.parametrize(
        ("reading", "path", "value"),
        [
            ("rifle", ("immune",), ["K", "W"]),
            ("demote", ("chain",), [{"from": "R", "to": "N"}, {"from": "W", "to": "N"}]),
            ("demote", ("chain",), [{"from": "R", "to": "W"}]),
            ("andernach", ("switches_sides",), ["R", "N", "W"]),
            ("rifle", ("by",), ["W"]),
            ("contacts", ("immune",), ["K", "W"]),
            ("contacts", ("links",), [{"mover": "N", "onto": "W", "onto_to": "N"}]),
            ("contacts", ("links",), [{"mover": "N", "onto": "R", "onto_to": "W"}]),
        ],
    )
    def test_a_list_naming_the_wild_type_is_refused(
        self, reading: str, path: tuple[str, ...], value: Any
    ) -> None:
        spec = _spec(reading)
        key = RULES[reading][0]
        spec["rules"][key][path[0]] = value
        errors = validate_spec(spec)
        assert any("makes a wild type" in error for error in errors), errors

    def test_a_strike_that_reaches_only_the_wild_type_states_nothing(self) -> None:
        """The validator asks the compiler's question with the wild set: a shot from which every
        player's type is immune reaches only the wolf, which it never reaches."""
        spec = _spec("rifle")
        spec["rules"]["strike"]["immune"] = ["K", "N", "R"]
        assert any("rules.strike" in error for error in validate_spec(spec))
        spec["rules"].pop("npcs")
        assert validate_spec(spec) == []

    def test_the_blast_is_lifted_too(self) -> None:
        """SR4f lifted T8 (`tests/test_wild_beside_blast.py` plays it)."""
        spec = _base()
        spec["rules"]["explode_on_capture"] = {"radius": 1, "immune": ["K"]}
        assert "T8" not in refused_ids(validate_spec(spec))

    def test_the_inventory_names_both(self) -> None:
        assert _schema_sixty_seven_uses(_spec("rifle")) == {"rules.strike.npcs"}
        assert _schema_sixty_seven_uses(_spec("contacts")) == {"rules.contacts.npcs"}
        assert _schema_sixty_seven_uses(_base()) == set()

    @pytest.mark.parametrize("reading", ["rifle", "contacts"])
    def test_describe_says_the_wolf_is_spared(self, reading: str) -> None:
        text = describe_spec(_spec(reading))
        assert "reaches a wolf, wild or a player's" in text
        assert "reaches a wolf" not in describe_spec(_base())

    @pytest.mark.parametrize("reading", ["rifle", "contacts"])
    def test_fairy_stockfish_refuses_the_wild_piece(self, reading: str) -> None:
        """A wild piece has no counterpart there; the pair is refused with its reason."""
        with pytest.raises(Untranslatable, match="npcs"):
            translate(_spec(reading))


# --------------------------------------------------------------------------- the walks


def _takes_a_players_piece(engine: Engine, state: GameState, move: int) -> bool:
    target = state.board[(move >> 8) & 0xFF]
    return bool(target) and engine.spec.owner_of(target) != WILD


class TestTheWalks:
    @pytest.mark.parametrize("reading", READINGS)
    def test_walks_hold_to_the_twin(self, reading: str) -> None:
        """Every ply whose move takes no player's piece plays as the twin plays it, the phase
        included; the key is the full scan's every ply. Over the walks wolves eat and are
        taken."""
        engine, twin = Engine(_spec(reading)), Engine(_twin(_spec(reading)))
        eats = taken = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(80):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                loud = [m for m in moves if state.board[(m >> 8) & 0xFF]]
                move = rng.choice(loud if loud and rng.random() < 0.6 else moves)
                after, steps = engine.apply_with_phase(state, move)
                if not _takes_a_players_piece(engine, state, move):
                    assert after.board == twin.apply(state, move).board, (reading, seed)
                    taken += bool(state.board[(move >> 8) & 0xFF])
                eats += sum(1 for step in steps if step.eaten)
                assert after.key == engine._compute_hash(after)
                state = after
        assert eats > 0 and taken > 0, (eats, taken)

    @pytest.mark.parametrize("reading", READINGS)
    def test_perft_counts_what_apply_counts(self, reading: str) -> None:
        engine = Engine(_spec(reading))
        state = engine.make_state({
            "a1": ("K", 0), "a6": ("K", 1), "c2": ("R", 0), "d5": ("R", 1),
            "d2": ("N", 0), "c5": ("N", 1), "c3": ("W", WILD),
        }, moved=["a1", "a6", "c2", "d5", "d2", "c5"])

        def count(state: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(state, m), depth - 1) for m in engine.legal_moves(state))

        assert engine.perft(state, 3) == count(state, 3)
