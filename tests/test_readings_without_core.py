"""SR2b (`docs/plan-schema-next.md` §3.2, rows T9, T10, V5 and V6): the readings that should need
no core, each fed to the core past the validator and held to its ruled reading move by move.

* **T10** (`strike.by` beside `strike.switches_sides`): ``by`` narrows which captures strike and
  ``switches_sides`` which capturers change sides, independently. The core disagrees: it reads
  the strike's position for every capturer that flips, so under rifle a switcher outside ``by``
  stays where it stood (and `compile._compile_strike` flips only a listed striker). Refused
  still, handed to SR4j, a strict xfail here.
* **V5, V6** (a command beside `strike` or `contacts`): a command moves onto empty squares
  unless ``captures_own``, so without it the strike and the contacts never meet a commanded
  move, and the pair plays as each rule alone: lifted. A commanded own-side capture under
  ``captures_own`` should be an ordinary removal; the core instead panics beside a strike (the
  own-side table is empty) and reads an own contact link beside contacts, so that half stays
  refused, handed to SR4h with strict xfails here.
* **T9** (an own contact link beside `must_capture`): moving onto your own piece is no capture.
  The core's `restrict` counts any occupied target as one, so the merge is compelled; refused
  still, handed to SR4i, a strict xfail here.

The oracles are this file's: the command pair's legal list is the strike's (or contact's) own
moves and the command's foreign moves, each applied as its own game applies it; the switch's
apply is read off the two lists (`T10_EXPECTED`). Declared test oracles (rule 3). Built on chess only because its
setup is a convenient army; nothing here is about chess.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

import chess_variant_sandbox.engine.compile as compile_module
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

ROOT = Path(__file__).resolve().parents[1]

KNIGHT_COMMAND = {"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1,
                  "effect": "command", "whose": "enemy"}
KING_COMMAND = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
                "effect": "command", "whose": "enemy", "captures_own": True}

RIFLE = {"effect": "capture", "position": "stay", "immune": ["K"]}
CIRCE = {"effect": "capture", "position": "home", "immune": ["K"]}
ANDERNACH = {"effect": "capture", "immune": ["K"], "switches_sides": ["Q", "R", "B", "N"]}
DEMOTE = {"effect": "demote", "position": "stay", "immune": ["K"],
          "chain": [{"from": "Q", "to": "R"}, {"from": "R", "to": "B"}, {"from": "B", "to": "N"}]}
ARCHERS = {**RIFLE, "by": ["B"]}
ENEMY_LINKS = {"position": "stay", "immune": ["K"], "links": [
    {"mover": "N", "onto": "B", "onto_to": "N"}, {"mover": "B", "onto": "N", "mover_to": "N"}]}
OWN_LINKS = {"position": "advance", "immune": ["K"], "links": [
    {"mover": "P", "onto": "R", "own": True, "mover_to": "Q"},
    {"mover": "B", "onto": "P", "own": True, "mover_to": "N"}]}


def _chess(**rules: Any) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((ROOT / "rulesets" / "chess.json").read_text("utf-8"))
    spec.pop("provenance", None)
    spec["rules"].update(copy.deepcopy(rules))
    return spec


def _commanding(spec: dict[str, Any], pid: str, sight: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    for piece in spec["pieces"]:
        if piece["id"] == pid:
            piece.setdefault("sight", []).append(copy.deepcopy(sight))
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _past_the_validator(monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]) -> Engine:
    monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
    return Engine(compile_spec(spec))


def _between(engine: Engine, state: GameState, origin: str, target: str) -> int:
    wanted = (engine._square_index(origin), engine._square_index(target))
    found = [m for m in engine.legal_moves(state) if (m & 0xFF, (m >> 8) & 0xFF) == wanted]
    assert len(found) == 1, (origin, target, engine.move_texts(state))
    return found[0]


def _as(engine: Engine, state: GameState) -> GameState:
    """``state`` as ``engine`` keys it: the boards of the games compared here encode alike."""
    return dataclasses.replace(state, key=engine._compute_hash(state))


# --------------------------------------------------------------------------- T10


T10_STRIKE = {"effect": "capture", "position": "stay", "immune": ["K"], "by": ["B", "R"],
              "switches_sides": ["N", "B"]}
# (origin, target): (where the mover ends, its owner after), each capturer by its two lists.
T10_EXPECTED = {
    ("d4", "e6"): ("e6", 1),   # switches, does not strike: advances, changes sides
    ("c1", "g5"): ("c1", 1),   # both: stays, changes sides
    ("a1", "a7"): ("a1", 0),   # strikes, does not switch: stays
    ("h1", "h7"): ("h7", 0),   # neither: an ordinary capture
}
T10_POSITION = {
    "e1": ("K", 0), "d4": ("N", 0), "c1": ("B", 0), "a1": ("R", 0), "h1": ("Q", 0),
    "e6": ("P", 1), "g5": ("P", 1), "a7": ("P", 1), "h7": ("P", 1), "e8": ("K", 1),
}


def test_by_beside_switches_sides_stays_refused() -> None:
    errors = validate_spec(_chess(strike=T10_STRIKE))
    assert "T10" in refused_ids(errors), errors


@pytest.mark.xfail(strict=True, reason="SR4j: the core reads a strike's position for every "
                   "flipping capturer (a switcher outside `by` stays under rifle), and "
                   "`compile._compile_strike` flips only a listed striker")
def test_each_capturer_reads_its_two_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _past_the_validator(monkeypatch, _chess(strike=T10_STRIKE))
    spec = engine.spec
    state = engine.make_state(T10_POSITION, side_to_move=0, moved=["e1", "e8"])
    for (origin, target), (ends, owner) in T10_EXPECTED.items():
        move = _between(engine, state, origin, target)
        after = engine.apply(state, move)
        mover = state.board[engine._square_index(origin)]
        stood = after.board[engine._square_index(ends)]
        assert spec.type_of(stood) == spec.type_of(mover), (origin, target)
        assert spec.owner_of(stood) == owner, (origin, target)


# --------------------------------------------------------------------------- V5, V6


COMMANDS_BESIDE = {
    "rifle": {"strike": RIFLE},
    "circe": {"strike": CIRCE},
    "andernach": {"strike": ANDERNACH},
    "demote": {"strike": DEMOTE},
    "archers": {"strike": ARCHERS},
    "enemy links": {"contacts": ENEMY_LINKS},
    "own links": {"contacts": OWN_LINKS},
}


class TestACommandBesideAStrikeOrContacts:
    @pytest.mark.parametrize("rules", COMMANDS_BESIDE.values(), ids=COMMANDS_BESIDE.keys())
    def test_without_captures_own_the_pair_validates(self, rules: dict[str, Any]) -> None:
        assert validate_spec(_chess(**rules)) == []
        assert validate_spec(_commanding(_chess(**rules), "N", KNIGHT_COMMAND)) == []

    @pytest.mark.parametrize("rules", COMMANDS_BESIDE.values(), ids=COMMANDS_BESIDE.keys())
    def test_with_captures_own_it_stays_refused(self, rules: dict[str, Any]) -> None:
        rid = {"strike": "V5", "contacts": "V6"}[next(iter(rules))]
        errors = validate_spec(_commanding(_chess(**rules), "K", KING_COMMAND))
        assert rid in refused_ids(errors), errors

    @pytest.mark.parametrize("rules", COMMANDS_BESIDE.values(), ids=COMMANDS_BESIDE.keys())
    def test_the_pair_plays_as_each_rule_alone(self, rules: dict[str, Any]) -> None:
        """Over seeded walks of the pair, both seats: its legal list is the rule's own moves
        and the command's foreign moves, and each move leaves the board its own game leaves."""
        both = _engine(_commanding(_chess(**rules), "N", KNIGHT_COMMAND))
        alone = _engine(_chess(**rules))
        command = _engine(_commanding(_chess(), "N", KNIGHT_COMMAND))
        commanded = {0: 0, 1: 0}
        for seed in range(16):
            rng = random.Random(seed)
            state = both.initial_state()
            for _ in range(90):
                moves = both.legal_moves(state)
                if both.terminal(state, moves) is not None:
                    break
                own = [m for m in alone.legal_moves(_as(alone, state)) if not m & MOVE_FOREIGN]
                foreign = [m for m in command.legal_moves(_as(command, state)) if m & MOVE_FOREIGN]
                assert sorted(moves) == sorted(own + foreign), (seed, state.ply)
                ordered = [m for m in moves if m & MOVE_FOREIGN]
                takes = [m for m in moves if not m & MOVE_FOREIGN and state.board[(m >> 8) & 0xFF]]
                draw = rng.random()
                move = (rng.choice(ordered) if ordered and draw < 0.35
                        else rng.choice(takes) if takes and draw < 0.7 else rng.choice(moves))
                after = both.apply(state, move)
                twin = (command if move & MOVE_FOREIGN else alone)
                assert after.board == twin.apply(_as(twin, state), move).board, (seed, state.ply)
                assert after.key == both._compute_hash(after)
                if move & MOVE_FOREIGN:
                    commanded[state.side_to_move] += 1
                state = after
        assert commanded[0] and commanded[1]


    def test_a_commanded_piece_makes_no_own_contact(self) -> None:
        """Without ``captures_own`` a commanded piece lands only on empty squares, so an own
        link of its type is never its commander's: the black knight on c6 may be sent to
        the empty squares, never onto its own pawn on e7, which its owner may merge with."""
        links = {"position": "advance", "immune": ["K"],
                 "links": [{"mover": "N", "onto": "P", "own": True, "mover_to": "B"}]}
        engine = _engine(_commanding(_chess(contacts=links), "N", KNIGHT_COMMAND))
        placement = {"e1": ("K", 0), "d4": ("N", 0), "c6": ("N", 1), "e7": ("P", 1),
                     "h8": ("K", 1)}
        state = engine.make_state(placement, side_to_move=0, moved=["e1", "h8"])
        commanded = [t for t in engine.move_texts(state) if t.startswith("~")]
        assert commanded and not any("x" in t for t in commanded), commanded
        state = engine.make_state(placement, side_to_move=1, moved=["e1", "h8"])
        assert "Nc6xe7/N->B" in engine.move_texts(state)


# The position of `tests/test_command.py`'s K2: the queen beside the king, which may command it
# onto its own bishop (h5) or knight (b5).
K2_MATE = {"e1": ("K", 0), "e2": ("Q", 1), "h5": ("B", 1), "a8": ("K", 1), "b5": ("N", 1)}
OWN_CONTACT = {"position": "advance", "immune": ["K"], "links": [
    {"mover": "Q", "onto": "N", "own": True, "mover_to": "R"}]}


@pytest.mark.xfail(strict=True, reason="SR4h: a commanded own-side capture beside a strike or "
                   "contacts; the core panics on the strike's empty own-side table and reads an "
                   "own contact link")
@pytest.mark.parametrize("rules", [{"strike": RIFLE}, {"strike": ANDERNACH}, {"strike": CIRCE},
                                   {"contacts": OWN_CONTACT}],
                         ids=["rifle", "andernach", "circe", "own link"])
def test_a_commanded_own_side_capture_is_an_ordinary_removal(
    monkeypatch: pytest.MonkeyPatch, rules: dict[str, Any]
) -> None:
    engine = _past_the_validator(monkeypatch, _commanding(_chess(**rules), "K", KING_COMMAND))
    state = engine.make_state(K2_MATE, side_to_move=0)
    try:
        for target in ("h5", "b5"):
            move = _between(engine, state, "e2", target)
            assert move & MOVE_FOREIGN
            after = engine.apply(state, move)
            assert after.board[engine._square_index(target)] == state.board[
                engine._square_index("e2")]
            assert after.board[engine._square_index("e2")] == 0
    except BaseException as exc:  # a core panic is a BaseException
        raise AssertionError(f"the core: {exc}") from None


# --------------------------------------------------------------------------- T9


MERGE = {"position": "advance", "immune": ["K"],
         "links": [{"mover": "P", "onto": "R", "own": True, "mover_to": "Q"}]}


def test_an_own_link_beside_must_capture_stays_refused() -> None:
    errors = validate_spec(_chess(contacts=MERGE, must_capture=True))
    assert "T9" in refused_ids(errors), errors


@pytest.mark.xfail(strict=True, reason="SR4i: the core's `restrict` counts a move onto an own "
                   "piece as a capture, so must_capture compels the merge")
def test_an_own_contact_is_no_capture_for_must_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _past_the_validator(monkeypatch, _chess(contacts=MERGE, must_capture=True))
    merge_only = {"e1": ("K", 0), "b2": ("P", 0), "a3": ("R", 0), "e8": ("K", 1), "h7": ("P", 1)}
    state = engine.make_state(merge_only, side_to_move=0, moved=list(merge_only))
    texts = engine.move_texts(state)
    assert "Pb2xa3/P->Q" in texts and "Ke1-e2" in texts      # nothing to take: every move
    with_a_take = {**merge_only, "c3": ("N", 1)}
    state = engine.make_state(with_a_take, side_to_move=0, moved=list(with_a_take))
    assert sorted(engine.move_texts(state)) == ["Pb2xc3", "Ra3xc3"]
