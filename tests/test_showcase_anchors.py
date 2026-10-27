"""V1 of I3's showcase *anchors* (`docs/archive/plan-pieces-moves.md` §4.8): the mechanism, without
players.

Chess with two anchors a side (`moves: []`) on b3 and g3 (b6 and g6), which only a pawn may
take (`captures` on every other type). The lists are the core's; nothing here decides a move's
legality (CLAUDE.md, rule 3): each test asks the core and reads its answer against what the
file says.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash

ROOT = Path(__file__).resolve().parent.parent
HASH, CHESS_HASH = "d6cb641394dd", "4c1922ffae26"


@cache
def _file() -> dict[str, Any]:
    return json.loads((ROOT / "archive" / "anchors.json").read_text(encoding="utf-8"))


@cache
def _engine() -> Engine:
    return Engine(_file())


def test_the_file_is_chess_with_anchors_only_pawns_take() -> None:
    spec = _file()
    assert spec_hash(spec) == HASH
    assert spec["provenance"]["parents"] == [CHESS_HASH]
    anchor = next(p for p in spec["pieces"] if p["id"] == "A")
    assert anchor["moves"] == [] and not anchor.get("royal")
    for piece in spec["pieces"]:
        if piece["id"] == "P":
            assert "captures" not in piece
        elif piece["id"] != "A":
            assert "A" not in piece["captures"]
    assert spec["appearance"]["icons"]["A"] == "anchor"
    chess = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
    assert spec["rules"] == chess["rules"]
    assert spec_hash(chess) == CHESS_HASH


def _anchor_type(engine: Engine) -> int:
    return engine.spec.piece_ids.index("A")


def _anchors(engine: Engine, board: Any) -> set[int]:
    a = _anchor_type(engine)
    return {s for s, v in enumerate(board) if v and engine.spec.type_of(v) == a}


def _random_games(games: int) -> None:
    engine = _engine()
    pawn = engine.spec.piece_ids.index("P")
    for seed in range(games):
        rng = random.Random(seed)
        state = engine.initial_state()
        count = len(_anchors(engine, state.board))
        assert count == 4
        for _ply in range(300):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            anchors = _anchors(engine, state.board)
            assert len(anchors) <= count
            count = len(anchors)
            for m in moves:
                origin, target = decode_move(m)[0], decode_move(m)[1]
                assert origin not in anchors
                if target in anchors:
                    assert engine.spec.type_of(state.board[origin]) == pawn
            state = engine.apply(state, rng.choice(moves))


def test_three_random_games_move_no_anchor_and_only_pawns_take_one() -> None:
    _random_games(3)


@pytest.mark.slow
def test_twenty_random_games_move_no_anchor_and_only_pawns_take_one() -> None:
    _random_games(20)


def _texts(placement: dict[str, tuple[str, int]], side: int = 0) -> list[str]:
    engine = _engine()
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    return engine.move_texts(state)


KINGS = {"a1": ("K", 0), "h8": ("K", 1)}


def test_a_line_stops_short_of_an_enemy_anchor_and_cannot_take_it() -> None:
    for piece in ("B", "R", "Q"):
        start, anchor, beyond = {"B": ("b2", "d4", "e5"), "R": ("d1", "d4", "d5"),
                                 "Q": ("d1", "d4", "d5")}[piece]
        texts = _texts({**KINGS, start: (piece, 0), anchor: ("A", 1)})
        mine = [t for t in texts if t.startswith(piece + start)]
        assert mine, piece
        assert not [t for t in mine if t.endswith(anchor) or t.endswith(beyond)], (piece, mine)


def test_a_knight_cannot_take_an_anchor() -> None:
    texts = _texts({**KINGS, "d4": ("N", 0), "e6": ("A", 1)})
    assert "Nd4-e6" not in texts and "Nd4xe6" not in texts
    assert "Nd4-c6" in texts


def test_a_pawn_diagonally_in_front_of_an_enemy_anchor_takes_it() -> None:
    engine = _engine()
    placement = {**KINGS, "c5": ("P", 0), "b6": ("A", 1)}
    state = engine.make_state(placement, moved=list(placement))
    after = engine.apply(state, engine.find_move(state, "Pc5xb6"))
    assert _anchors(engine, after.board) == set()
    # and Black's pawn takes White's anchor the same way, down the board
    placement = {**KINGS, "c4": ("P", 1), "b3": ("A", 0)}
    state = engine.make_state(placement, side_to_move=1, moved=list(placement))
    assert "Pc4xb3" in engine.move_texts(state)


def test_an_own_anchor_directly_ahead_stops_a_pawn() -> None:
    texts = _texts({**KINGS, "b2": ("P", 0), "b3": ("A", 0)})
    assert not [t for t in texts if t.startswith("Pb2-")]


def test_a_king_may_not_take_an_anchor_and_an_anchor_gives_no_check() -> None:
    engine = _engine()
    placement = {"d4": ("K", 0), "e5": ("A", 1), "h8": ("K", 1)}
    state = engine.make_state(placement, moved=list(placement))
    assert "Kd4xe5" not in engine.move_texts(state)
    assert not engine.in_check(state)
    # every square next to the anchor is open to the king: the anchor attacks nothing
    assert {"Kd4-d5", "Kd4-e4", "Kd4-e3"} <= set(engine.move_texts(state))


def test_fairy_refuses_the_file_by_name() -> None:
    with pytest.raises(Untranslatable, match="captures"):
        translate(_file())
