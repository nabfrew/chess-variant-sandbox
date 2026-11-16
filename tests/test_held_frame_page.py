"""SR8i on the play page: a held piece's moves are drawn for its holder.

The moves come from the core through `Engine` (`foreign.rs::push_held` in the holder's frame),
so the page draws whatever the engine lists. Held here by behaviour, on a 6x6 spec with a
pawn-like type nobody owns (`tests/test_held_in_holders_frame.py` holds the core):

- **The server** offers the second player's held pawn its step toward the first player, as a
  foreign move from its square, and the first player's the other way.
- **The board**, a held piece picked, lights the engine's squares for the holder and a tap
  there sends the engine's move.
- **A held piece that faces** is drawn by its arrow (nobody's facing read in the first
  player's row of headings, its frame), and its holder's turn is offered with arrows that
  read nobody's facings, each sending the engine's word.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from test_held_in_holders_frame import _arrow, _pawn
from test_portal_board import _game
from test_sight_page import _drawn
from test_turncoat_page import _with_hand

from chess_variant_sandbox.engine.moves import MOVE_FOREIGN
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.play import Session


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * 6 + ord(name[0]) - ord("a")


def _session(spec: dict[str, Any], placement: dict[str, tuple[str, int]], side: int) -> Session:
    engine = engine_for(spec)
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    return Session(id="s1", game=_game(spec), seats=("human", "human"), seed=0, state=state)


# The pawn nobody owns on c3, held by whoever has a guard beside it.
def _held_by(player: int) -> dict[str, tuple[str, int]]:
    guard = "b2" if player == 0 else "b4"
    return {"a1": ("K", 0), "f6": ("K", 1), guard: ("G", player), "c3": ("H", 2)}


def _held_moves(snap: dict[str, Any], square: int) -> list[dict[str, Any]]:
    return [m for m in snap["moves"] if m.get("from") == square and m["move"] & MOVE_FOREIGN]


class TestTheServer:
    def test_a_held_pawn_is_offered_its_holders_forward(self) -> None:
        for holder, ahead in ((0, "c4"), (1, "c2")):
            snap = _session(_pawn(), _held_by(holder), holder).snapshot()
            held = _held_moves(snap, _sq("c3"))
            assert [m["to"] for m in held] == [_sq(ahead)], (holder, held)
            assert snap["held"] == [[_sq("c3"), holder]]


class TestThePage:
    def test_picked_it_lights_the_holders_squares_and_a_tap_sends_the_engines_move(self) -> None:
        snap = _session(_pawn(), _held_by(1), 1).snapshot()
        move = _held_moves(snap, _sq("c3"))[0]
        steps = (f"let SENT = null; send = m => {{ SENT = m; }}; clickSquare({_sq('c3')});"
                 " const PICKED = SELECTED;")
        drawn, picked = _drawn(snap, steps, "[DRAWN, PICKED]")
        assert picked == _sq("c3")
        dots = {int(s) for s in re.findall(r'<circle[^>]*data-square="(\d+)"', drawn)}
        assert _sq("c2") in dots and _sq("c4") not in dots, dots
        steps += f" clickSquare({_sq('c2')});"
        assert _drawn(snap, steps, "SENT") == move["move"]

    def test_a_held_piece_that_faces_is_drawn_by_its_arrow_and_turned_by_nobodys_facings(
        self,
    ) -> None:
        spec = _arrow()
        snap = _session(spec, _held_by(1), 1).snapshot()
        engine = engine_for(spec)
        # Point it right, in nobody's frame (facing index 1), as the placement does.
        board = list(snap["board"])
        board[_sq("c3")] = (2 * engine.spec.n_facings + 1) * engine.spec.n_types + 3 + 1
        snap = copy.deepcopy(snap)
        snap["board"] = board
        # An icon turns by its owner's row of headings; nobody's is the first player's (this
        # scratch game's pieces are lettered, so the row is asked of the page directly).
        rows = snap["ruleset"]["facingAngles"]
        assert _drawn(snap, "", "[facingRow(2), facingRow(1)]") == [rows[0], rows[1]]
        held = f"pieceAt({_sq('c3')})"
        assert _drawn(snap, "", f"facingRow({held}.player)[{held}.facing]") == 90
        turns = {m["rotation"]: m["move"] for m in _held_moves(
            _session(spec, _held_by(1), 1).snapshot(), _sq("c3")) if "rotation" in m}
        assert set(turns) == {1, 2, 3}
        steps = (f"let SENT = null; send = m => {{ SENT = m; }}; clickSquare({_sq('c3')});"
                 " const SHOWN_HAND = HAND_HTML;")
        hand = _with_hand(_session(spec, _held_by(1), 1).snapshot(), steps, "SHOWN_HAND")
        # Nobody's index 2 points down the board for either holder: the down arrow.
        assert re.search(rf'data-move="{turns[2]}" title="face down"', hand), hand
