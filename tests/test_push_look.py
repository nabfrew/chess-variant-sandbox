"""A push's look on the play board (Q1074b5, `docs/plan-push.md` K6): the history entry
names each piece the push moved and where it went, from the engine (`viz.wrapped.push_transit`,
`CompiledSpec.push_line`), and the board slides the line one square as the pusher lands and
lets the piece thrown off fall away over the edge (`motion.js` `pushFall`) rather than knocking
it off as a capture. The page works none of it out from the two positions: a row of like pieces
shifted one square reads there as one piece jumping the row.

The game is chess whose king pushes its own side's pieces in rows, used for its army only (the
same draft `look.mjs`'s `play-push` row plays): from the opening the king steps onto its queen
and shoves the whole back rank to its left one file over, the rook off the board.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from page_text import PAGE

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.play import Session

ROOT = Path(__file__).resolve().parent.parent


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _spec(chain: str = "row") -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(
        (ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
    spec["name"] = f"chess with a shoving king ({chain})"
    for piece in spec["pieces"]:
        if piece["id"] == "K":
            for move in piece["moves"]:
                move["push"] = {"whose": "own"}
        if piece["id"] == "R":
            for move in piece["moves"]:
                move["push"] = True
    spec["rules"]["push"] = {"chain": chain}
    return spec


def _session(spec: dict[str, Any], placement: dict[str, tuple[str, int]] | None = None) -> Session:
    engine = engine_for(spec)
    state = engine.make_state(placement) if placement else engine.initial_state()
    game = Game(hash=spec_hash(spec), name=str(spec["name"]), description="", path=None,
                source="scratch", spec=copy.deepcopy(spec))  # type: ignore[arg-type]
    return Session(id="p1", game=game, seats=("human", "human"), seed=0, state=state)


def _play(session: Session, text: str) -> dict[str, Any]:
    (move,) = [m["move"] for m in session.snapshot()["moves"] if m["text"] == text]
    session.play(move)
    return session.history[-1]


class TestThePushedLine:
    def test_a_row_names_every_piece_it_moved_and_the_one_it_threw_off(self) -> None:
        session = _session(_spec())
        last = _play(session, "Ke1>>d1")
        assert last["pushed"] == [
            [_square("d1"), _square("c1")], [_square("c1"), _square("b1")],
            [_square("b1"), _square("a1")], [_square("a1"), None],
        ]
        # The step is the board's own: one file towards a, whoever moved.
        assert last["pushStep"] == [-1, 0]
        board = session.state.board
        assert board[_square("e1")] == 0 and board[_square("a1")] != 0

    def test_a_single_push_names_its_one_piece(self) -> None:
        session = _session(_spec("one"), {
            "a1": ("K", 0), "d4": ("R", 0), "d6": ("P", 1), "h8": ("K", 1),
        })
        last = _play(session, "Rd4>d6")
        assert last["pushed"] == [[_square("d6"), _square("d7")]]
        assert last["pushStep"] == [0, 1]

    def test_a_move_that_pushes_nothing_says_nothing(self) -> None:
        session = _session(_spec())
        last = _play(session, "Pe2-e4")
        assert "pushed" not in last and "pushStep" not in last


class TestTheLook:
    def test_the_board_reads_the_engine_s_line_and_never_diffs_it(self) -> None:
        assert "last.pushed" in PAGE and "last.pushStep" in PAGE
        # The line's pieces are paired from the engine's list before the diff's own pairing.
        assert PAGE.index("const pushed = last && Array.isArray(last.pushed)") < PAGE.index(
            "// Then anything else that moved without changing what it is")

    def test_a_piece_thrown_off_falls_rather_than_being_knocked_off(self) -> None:
        assert "function pushFall(" in PAGE
        assert PAGE.count("pushFall(svg, piece, x, y, cell, step, pushAt)") == 1
        assert "'data-journey', 'push-fall'" in PAGE
        assert "pushFall:" in PAGE  # its pace, beside the others in `PACE`
