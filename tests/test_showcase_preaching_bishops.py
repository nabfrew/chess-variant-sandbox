"""V1, the mechanism of Q1024's command showcase (`docs/archive/plan-schema-31.md` §6.5).

**P1**, *chess with preaching bishops* (`91c2c24305e5`): chess, and every enemy pawn on one of a
bishop's eight lines, at any distance and through any piece (an aura, `through`), may also be
moved by the bishop's owner, onto an empty square or onto a non-royal piece of its own side
(`captures_own`). Held here:

- the file is chess with exactly its registered key, chess (`4c1922ffae26`) its parent;
- hand-built positions: a commanded pawn's step and its turncoat capture, seen through a piece
  between; never a capture of the commander's piece or of the pawn's own king; no other type
  commanded; no promotion by command, and a commanded double step opening no en passant; the
  opening's four commanded pushes, the c- and f-pawns seen down their files;
- over seeded random games (three here, twenty under `slow`), the core's legal list equals the
  sight oracle's after every move (`test_command_walks.walk`), with every commanded move sound.

Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and the
oracle (`tests/sight_oracle.py`) is the declared second reading.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from sight_oracle import SightOracle
from test_command_walks import walk

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "archive" / "chess-with-preaching-bishops.json"
CHESS = ROOT / "rulesets" / "chess.json"
HASH = "bafc4a7011a3"  # Q1069a; was 91c2c24305e5
PREACHING = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "effect": "command",
             "whose": "enemy", "affects": ["P"], "through": True, "captures_own": True}


def game() -> dict[str, Any]:
    return json.loads(FILE.read_text(encoding="utf-8"))


def position(placement: dict[str, tuple[str, int]], side: int = 0) -> tuple[Engine, GameState]:
    built = Engine(game())
    return built, built.make_state(placement, side_to_move=side)


def commanded(built: Engine, state: GameState) -> list[str]:
    return sorted(text for text in built.move_texts(state) if text.startswith("~"))


def test_it_is_the_registered_game_and_chess_is_its_parent() -> None:
    spec = load_spec(FILE)
    assert spec_hash(spec)[:12] == HASH
    chess = json.loads(CHESS.read_text(encoding="utf-8"))
    # Q1069a: chess as it was, before both it and the showcase drew on dead material.
    unruled = copy.deepcopy(chess)
    unruled["rules"].pop("insufficient_material")
    assert spec_hash(unruled)[:12] == "4c1922ffae26"
    assert game()["provenance"]["parents"] == ["4c1922ffae26"]
    built = copy.deepcopy(chess)
    for piece in built["pieces"]:
        if piece["id"] == "B":
            piece["sight"] = [copy.deepcopy(PREACHING)]
    assert spec_hash(built)[:12] == HASH


BASE = {"a1": ("K", 0), "d1": ("B", 0), "d5": ("P", 1), "e4": ("N", 1), "h8": ("K", 1)}


def test_a_pawn_in_a_bishops_line_steps_and_takes_its_own_side() -> None:
    built, state = position(BASE)
    assert commanded(built, state) == ["~Pd5-d4", "~Pd5xe4"]


def test_the_bishop_is_heard_through_a_piece_between() -> None:
    built, state = position({**BASE, "d3": ("N", 0)})
    assert "~Pd5xe4" in commanded(built, state)
    built, state = position({**BASE, "d3": ("R", 1)})
    assert "~Pd5xe4" in commanded(built, state)


def test_a_commanded_pawn_never_takes_the_commanders_piece_or_its_own_king() -> None:
    built, state = position({**BASE, "c4": ("N", 0)})
    assert "~Pd5xc4" not in commanded(built, state)
    placement = {k: v for k, v in BASE.items() if k not in ("e4", "h8")}
    built, state = position({**placement, "e4": ("K", 1)})
    assert not any("xe4" in text for text in commanded(built, state))


def test_only_pawns_are_commanded() -> None:
    built, state = position({"a1": ("K", 0), "d1": ("B", 0), "d5": ("N", 1), "g4": ("R", 1),
                             "g7": ("Q", 1), "h8": ("K", 1)})
    assert commanded(built, state) == []


def test_no_promotion_and_no_window_by_command() -> None:
    built, state = position({"a8": ("K", 0), "b1": ("B", 0), "b2": ("P", 1), "d1": ("B", 0),
                             "d7": ("P", 1), "h8": ("K", 1)})
    texts = commanded(built, state)
    assert not any(text.startswith("~Pb2") for text in texts)  # every arrival must promote
    assert "~Pd7-d5" in texts
    after = built.apply(state, built.find_move(state, "~Pd7-d5"))
    assert after.ep_target == NO_SQUARE and after.ep_victim == NO_SQUARE


def test_the_opening_hears_the_c_and_f_pawns() -> None:
    built = Engine(game())
    state = built.initial_state()
    assert commanded(built, state) == ["~Pc7-c5", "~Pc7-c6", "~Pf7-f5", "~Pf7-f6"]
    assert len(built.legal_moves(state)) == 24


def _judge() -> SightOracle:
    return SightOracle(game())


def test_the_core_agrees_with_the_oracle_over_three_games() -> None:
    compared, played = walk("chess-with-preaching-bishops", games=3, plies=100, judge=_judge())
    assert compared > 30
    assert played > 0


@pytest.mark.slow
def test_the_core_agrees_with_the_oracle_over_twenty_games() -> None:
    compared, played = walk("chess-with-preaching-bishops", games=20, judge=_judge())
    assert compared > 200
    assert played > 20
