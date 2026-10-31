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
        # Q1074bo: the board picks the fall or the crush by the engine's `pushCrush`.
        assert PAGE.count("const away = last.pushCrush ? pushCrush : pushFall;") == 1
        assert PAGE.count("away(svg, piece, x, y, cell, step, pushAt)") == 1
        assert "'data-journey', 'push-fall'" in PAGE
        assert "pushFall:" in PAGE  # its pace, beside the others in `PACE`

    def test_a_piece_crushed_on_a_wall_is_crushed_in_place(self) -> None:
        """Q1074bo: one crushed on a wall is pressed flat where it stood, not dropped over an
        edge that is not there."""
        assert "function pushCrush(" in PAGE
        assert "'data-journey', 'push-crush'" in PAGE
        assert "pushCrush:" in PAGE


def _crushing() -> dict[str, Any]:
    """Chess whose rooks push, a wall on d5 that crushes (Q1074b7)."""
    spec = _spec("one")
    spec["name"] = "chess with a crushing wall"
    spec["rules"]["push"] = {"walls": "crush"}
    spec["rules"]["obstacles"] = {"squares": ["d5"]}
    return spec


_CRUSH_PLACEMENT = {"a1": ("K", 0), "d3": ("R", 0), "d4": ("P", 1), "h8": ("K", 1)}


class TestTheCrush:
    """Q1074bo: the engine's account of a push says when its last piece was crushed on a
    wall, so the play board, the replay and the landing's stories crush it in place."""

    def test_a_piece_crushed_on_a_wall_is_named_as_crushed(self) -> None:
        session = _session(_crushing(), _CRUSH_PLACEMENT)
        last = _play(session, "Rd3>>d4")
        assert last["pushed"] == [[_square("d4"), None]]
        assert last["pushCrush"] is True
        assert last["pushStep"] == [0, 1]

    def test_a_piece_thrown_off_is_not_named_as_crushed(self) -> None:
        session = _session(_spec())
        assert "pushCrush" not in _play(session, "Ke1>>d1")

    def test_a_story_frame_carries_the_push_and_keys_a_removal(self) -> None:
        from chess_variant_sandbox.web.stories import Played, _did, _key

        engine = engine_for(_crushing())
        state = engine.make_state(_CRUSH_PLACEMENT)
        (move,) = [m for m in engine.legal_moves(state)
                   if engine.describe_move(state, m) == "Rd3>>d4"]
        item = _did(engine, state, engine.apply(state, move), move)
        assert item["push"]["pushed"] == [[_square("d4"), None]]
        assert item["push"]["pushCrush"] is True
        assert _key(Played(hash="x", ending=(None, "draw")), 3, 9, item)
        # A push that only shoves is a plain turn of the story.
        shove = engine.make_state({"h1": ("K", 0), "a3": ("R", 0), "a4": ("P", 1),
                                   "h8": ("K", 1)})
        (move,) = [m for m in engine.legal_moves(shove)
                   if engine.describe_move(shove, m) == "Ra3>a4"]
        assert not _key(Played(hash="x", ending=(None, "draw")), 3, 9,
                        _did(engine, shove, engine.apply(shove, move), move))

    def test_the_story_slides_the_line_and_draws_no_capture(self) -> None:
        assert "item.push" in PAGE
        assert "(push.pushCrush ? pushCrush : pushFall)(" in PAGE
        assert "if (before[item.to] && !push) {" in PAGE
