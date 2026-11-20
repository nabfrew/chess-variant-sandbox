"""R-L4b5 (`docs/reference/capture-effects-design.md` §6 stage 5, §3.3): a round on the list.

`rounds.md` §4 steps 5 and 6 are the capture list's walk over the round's takings
(`rounds.rs::walk_round`): a collision's loser is a taking of cause ``"collision"``, a blast's
victims the next round's takings. The declared oracles agree with the core:
`tests/simultaneous_oracle.py` resolves the round and hands its takings to
`tests/capture_oracle.py::walk_round`, which plays the list. Here the two agree, round by round,
on the two narrowed rows as the old keys spell them (T1.rounds: a strike beside a blast; T4.rounds:
contacts beside a blast), on the blast's strike switch, and on every list of up to two entries
(three, slow) handed to the compile past the validator; and each row's reading is named by a test.
"""

from __future__ import annotations

import copy
import itertools
import random
from typing import Any
from unittest import mock

import pytest
from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.compile import (
    CAPTURE_KINDS,
    CompiledSpec,
    compile_spec,
    with_capture_list,
)
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fuzz_lifted import generate, keys
from chess_variant_sandbox.spec.refusals import refusal
from chess_variant_sandbox.spec.schema import validate_spec

CHAIN = [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]
#: A blast set off by taking a rook, a knight or a pawn, never a bishop: so a capture of a
#: bishop reaches the strike or the link, and one of the others the blast.
BLAST = {"radius": 1, "immune": ["K"], "of": ["R", "N", "P"]}


def _rounds(**rules: Any) -> dict[str, Any]:
    """The fuzz's six-square army in rounds (royal capture), with ``rules``."""
    spec = copy.deepcopy(generate.SIX)
    spec["rules"]["simultaneous"] = {}
    spec["rules"].update(rules)
    return spec


def _strike(position: str) -> dict[str, Any]:
    return {"effect": "demote", "position": position, "immune": ["K"], "chain": CHAIN}


def _links(position: str) -> dict[str, Any]:
    """A contact table: a knight taking a bishop becomes a rook, a rook taking a knight a
    bishop; placed ``"swap"`` (or ``"stay"``) a bishop taken by a bishop, and a knight by a rook,
    also survive as pawns (``"advance"`` has nowhere to put a survivor)."""
    keeps = position != "advance"
    links: list[dict[str, Any]] = [{"mover": "N", "onto": "B", "mover_to": "R"}]
    links.append({"mover": "R", "onto": "N", "mover_to": "B", **({"onto_to": "P"} if keeps else {})})
    if keeps:
        links.append({"mover": "B", "onto": "B", "onto_to": "P"})
    return {"position": position, "immune": ["K"], "links": links}


def _is_capture(state: GameState, move: int) -> bool:
    return state.board[move_to(move)] != 0 and move_from(move) != move_to(move)


def _walk(engine: Engine, seed: int, rounds: int) -> int:
    """A seeded playout of ``rounds`` rounds, captures drawn more often than not, each round
    resolved by the core and by the oracle and the whole position compared; the rounds played."""
    rng = random.Random(seed)
    state = engine.initial_state()
    for played in range(rounds):
        if engine.terminal(state) is not None:
            return played
        pair = []
        for player in (0, 1):
            moves = engine.legal_moves_for(state, player)
            if not moves:
                return played
            captures = [m for m in moves if _is_capture(state, m)]
            pair.append(rng.choice(captures if captures and rng.random() < 0.7 else moves))
        expected = resolve_round(engine, state, pair[0], pair[1])
        actual = engine.apply_round(state, pair[0], pair[1])
        assert actual == expected, (
            f"round {played + 1} of seed {seed}: {engine.describe_move(state, pair[0])} | "
            f"{engine.describe_move(state, pair[1])}; list {[e.do for e in engine.spec.capture or ()]}"
        )
        assert actual.key == engine._compute_hash(actual)
        state = actual
    return rounds


# --------------------------------------------------------------- the rows, as the old keys

OLD_KEYS: dict[str, dict[str, Any]] = {
    "T1.rounds home": {"strike": _strike("home"), "explode_on_capture": BLAST},
    "T1.rounds swap": {"strike": _strike("swap"), "explode_on_capture": BLAST},
    "T1.rounds switched": {"strike": {**_strike("swap"), "switches_sides": ["N", "B"]},
                           "explode_on_capture": BLAST},
    "T1.rounds spared": {"strike": _strike("home"), "explode_on_capture": {**BLAST, "strikes": "target"}},
    "T1.rounds read": {"strike": _strike("home"), "explode_on_capture": {**BLAST, "strikes": "both"}},
    "T4.rounds advance": {"contacts": _links("advance"), "explode_on_capture": BLAST},
    "T4.rounds swap": {"contacts": _links("swap"), "explode_on_capture": BLAST},
    "T4.rounds read": {"contacts": _links("swap"), "explode_on_capture": {**BLAST, "strikes": "blast"}},
    "T1.rounds with a hand": {"strike": _strike("home"), "explode_on_capture": BLAST,
                              "drops": {"from_captures": True}},
}


@pytest.mark.parametrize("name", list(OLD_KEYS))
def test_the_old_keys_play_in_a_round_as_the_oracles_say(name: str) -> None:
    spec = _rounds(**OLD_KEYS[name])
    assert validate_spec(spec) == [], validate_spec(spec)
    row = name.split()[0]
    assert row in keys.pairs_present(spec)
    engine = Engine(spec)
    assert sum(_walk(engine, seed, 16) for seed in range(4)) > 0


def _setup(*placements: tuple[str, str]) -> dict[str, Any]:
    """Each side's pieces, mirrored (a pawn on f2 keeps the promotion rule live)."""
    pieces = [{"piece": p, "squares": [sq]} for p, sq in (*placements, ("P", "f2"))]
    return {"placements": pieces}


def _sq(engine: Engine, name: str) -> int:
    return engine.spec.square_index(ord(name[0]) - ord("a"), int(name[1:]) - 1)


def _round_with(spec: dict[str, Any], move: tuple[str, str], other: tuple[str, str]) -> tuple[Engine, GameState]:
    """``spec`` played one round from its opening: player 0's ``move`` and player 1's
    ``other``, each named by its two squares; held to the oracle; the engine and the position."""
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    state = engine.initial_state()
    words = []
    for player, (a, b) in enumerate((move, other)):
        found = [m for m in engine.legal_moves_for(state, player)
                 if move_from(m) == _sq(engine, a) and move_to(m) == _sq(engine, b)]
        assert found, (player, a, b)
        words.append(found[0])
    after = engine.apply_round(state, words[0], words[1])
    assert after == resolve_round(engine, state, words[0], words[1])
    return engine, after


def _at(engine: Engine, state: GameState, square: str) -> tuple[int, str] | None:
    value = state.board[_sq(engine, square)]
    if not value:
        return None
    return engine.spec.owner_of(value), engine.spec.piece_ids[engine.spec.type_of(value)]


class TestTheNamedRows:
    """Each narrowed row's reading, by one round from a small opening (mirrored, so black's
    king stands on a6 and steps to b6): the default order is ``blast`` before ``strike`` or
    ``contacts``, so a capture that sets off the blast is the blast's (its taker gone, its
    victim destroyed, no link read), and one that does not is the strike's or the link's."""

    def test_t1_rounds_a_capture_that_explodes_is_the_blast_s(self) -> None:
        spec = _rounds(strike=_strike("home"), explode_on_capture=BLAST)
        spec["setup"] = _setup(("K", "a1"), ("R", "c1"), ("N", "d3"))
        engine, after = _round_with(spec, ("c1", "c6"), ("a6", "b6"))
        assert _at(engine, after, "c6") is None  # the rook went with its blast
        assert _at(engine, after, "c1") is None  # and the rook taken was not reborn anywhere
        assert not any(_at(engine, after, f"{f}{r}") == (1, "B") for f in "abcdef" for r in range(1, 7))

    def test_t1_rounds_a_capture_the_blast_spares_is_the_strike_s(self) -> None:
        spec = _rounds(strike=_strike("swap"), explode_on_capture=BLAST)
        spec["setup"] = _setup(("K", "a1"), ("N", "b2"), ("B", "c3"))
        # Taking a bishop sets off no blast: the strike demotes it to a knight placed on the
        # captor's origin (swap), and the knight stands where it took.
        engine, after = _round_with(spec, ("b2", "c4"), ("a6", "b6"))
        assert _at(engine, after, "c4") == (0, "N")
        assert _at(engine, after, "b2") == (1, "N")

    def test_t4_rounds_a_link_is_read_where_the_blast_is_not_set_off(self) -> None:
        spec = _rounds(contacts=_links("advance"), explode_on_capture=BLAST)
        spec["setup"] = _setup(("K", "a1"), ("N", "b2"), ("B", "c3"))
        # The knight takes the bishop: no blast, and the link makes the knight a rook there.
        engine, after = _round_with(spec, ("b2", "c4"), ("a6", "b6"))
        assert _at(engine, after, "c4") == (0, "R")

    def test_t4_rounds_a_capture_that_explodes_meets_no_link(self) -> None:
        spec = _rounds(contacts=_links("advance"), explode_on_capture=BLAST)
        spec["setup"] = _setup(("K", "a1"), ("R", "c1"), ("N", "c4"))
        # The rook takes black's knight on c3 (a link to a bishop): the blast comes first, so
        # the rook goes and no bishop is made.
        engine, after = _round_with(spec, ("c1", "c3"), ("a6", "b6"))
        assert _at(engine, after, "c3") is None
        assert not any(_at(engine, after, f"{f}{r}") == (0, "B") for f in "abcdef" for r in range(1, 7))

    def test_what_stays_refused_is_a_strike_or_link_that_stays(self) -> None:
        assert refusal("T1", "rules.strike") in validate_spec(
            _rounds(strike=_strike("stay"), explode_on_capture=BLAST))
        assert refusal("T4", "rules.contacts") in validate_spec(
            _rounds(contacts=_links("stay"), explode_on_capture=BLAST))
        assert validate_spec(_rounds(strike=_strike("stay"))) == []  # a shot alone plays as before


# ------------------------------------------------------------ lists in their own order

def _every_rule(position: str) -> dict[str, Any]:
    """The army in rounds with a blast, a strike placed ``position``, a capture hand and a
    bounty on every piece but the king, counted in points."""
    spec = _rounds(strike=_strike(position), explode_on_capture=BLAST,
                   drops={"from_captures": True}, scoring={"count": "points"})
    for piece in spec["pieces"]:
        if not piece.get("royal"):
            piece["bounty"] = max(1, int(piece.get("value", 1)))
    return spec


def _compiled(position: str) -> CompiledSpec:
    with mock.patch.object(compile_module, "validate_or_raise", lambda spec: spec):
        return compile_spec(_every_rule(position))


def _lists(longest: int) -> list[list[dict[str, Any]]]:
    kinds: list[dict[str, Any]] = [{"do": kind} for kind in CAPTURE_KINDS]
    kinds += [
        {"do": "blast", "spares_taken": True}, {"do": "blast", "on": ["landing", "collision", "blast"]},
        {"do": "strike", "on": ["landing", "blast"]}, {"do": "strike", "on": ["landing"]},
        {"do": "hand", "on": ["landing"]}, {"do": "contacts", "on": ["landing", "collision"]},
    ]
    return [list(combo) for n in range(1, longest + 1) for combo in itertools.product(kinds, repeat=n)]


class TestEveryListInARound:
    @pytest.mark.parametrize("position", ["home", "swap"])
    def test_every_list_of_one_or_two_entries(self, position: str) -> None:
        compiled = _compiled(position)
        played = 0
        for index, entries in enumerate(_lists(2)):
            played += _walk(Engine(with_capture_list(compiled, entries)), seed=index, rounds=6)
        assert played > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("position", ["home", "swap", "advance"])
    def test_every_list_of_up_to_three_entries(self, position: str) -> None:
        compiled = _compiled(position)
        for index, entries in enumerate(_lists(3)):
            _walk(Engine(with_capture_list(compiled, entries)), seed=index, rounds=10)

    def test_a_list_in_a_round_passes_the_validator(self) -> None:
        spec = _rounds(capture=[{"do": "strike", **_strike("home")}, {"do": "blast", **BLAST}])
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        assert engine.spec.capture is not None and [e.do for e in engine.spec.capture] == ["strike", "blast"]
        assert sum(_walk(engine, seed, 12) for seed in range(4)) > 0

    def test_the_core_refuses_what_a_round_cannot_walk(self) -> None:
        with mock.patch.object(compile_module, "validate_or_raise", lambda spec: spec):
            stay = compile_spec(_rounds(strike=_strike("stay"), explode_on_capture=BLAST))
        with pytest.raises(ValueError, match="not built"):
            Engine(stay)
