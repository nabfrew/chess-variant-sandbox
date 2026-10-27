"""A sweep's look on the play board (schema 13): the history entry names the squares the
sweep cleared, from the engine, and the board cuts them down along the sweeper's line
(`motion.js` `sweepCut`) rather than knocking them off as a capture. The landing's stories
read the same list and play the same cut. A game in rounds refuses a sweep, and the offline
replay animates nothing, so these are every surface a sweep is animated on."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from page_text import PAGE

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.page import tokens
from chess_variant_sandbox.web.play import Session
from chess_variant_sandbox.web.stories import _did

ROOT = Path(__file__).resolve().parent.parent


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _session() -> Session:
    spec: dict[str, Any] = json.loads(
        (ROOT / "archive" / "short-sweeping-bishop-chess.json").read_text(encoding="utf-8"))
    engine = engine_for(spec)
    # A bishop on c1 whose sweep up the long diagonal passes a piece of each side.
    state = engine.make_state({
        "c1": ("B", 0), "d2": ("P", 0), "e3": ("N", 1), "e1": ("K", 0), "h8": ("K", 1),
    })
    game = Game(hash=spec_hash(spec), name=str(spec["name"]), description="", path=None,
                source="scratch", spec=spec)  # type: ignore[arg-type]
    return Session(id="s1", game=game, seats=("human", "medium"), seed=0, state=state)


class TestTheSweptSquares:
    def test_a_sweep_names_every_square_it_cleared_in_order(self) -> None:
        session = _session()
        (move,) = [m["move"] for m in session.snapshot()["moves"]
                   if m["text"].startswith("Bc1xf4")]
        session.play(move)
        last = session.history[-1]
        assert last["swept"] == [_square("d2"), _square("e3")]
        assert not session.state.board[_square("d2")] and not session.state.board[_square("e3")]

    def test_a_move_that_sweeps_nothing_says_nothing(self) -> None:
        session = _session()
        (move,) = [m["move"] for m in session.snapshot()["moves"] if m["text"] == "Bc1-b2"]
        session.play(move)
        assert "swept" not in session.history[-1]


def _rook_session() -> Session:
    """The same game with the sweep on the rook's lines instead of the bishop's diagonals: a
    sweep is any line, and nothing downstream may assume a diagonal."""
    spec: dict[str, Any] = json.loads(
        (ROOT / "archive" / "short-sweeping-bishop-chess.json").read_text(encoding="utf-8"))
    spec["name"] = "sweeping rook"
    for piece in spec["pieces"]:
        if piece["id"] == "R":
            piece["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4",
                               "mode": "both", "range": 3}]
        if piece["id"] == "B":
            piece["moves"] = [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]
    engine = engine_for(spec)
    # A rook on d4 with two pieces on each of its four lines, both sides' among them.
    state = engine.make_state({
        "d4": ("R", 0), "a8": ("K", 0), "h1": ("K", 1),
        "d5": ("P", 0), "d6": ("N", 1), "e4": ("N", 1), "f4": ("P", 0),
        "d3": ("N", 1), "d2": ("P", 0), "c4": ("P", 0), "b4": ("N", 1),
    })
    game = Game(hash=spec_hash(spec), name=str(spec["name"]), description="", path=None,
                source="scratch", spec=spec)  # type: ignore[arg-type]
    return Session(id="r1", game=game, seats=("human", "medium"), seed=0, state=state)


class TestARookSweep:
    @pytest.mark.parametrize(("text", "cleared"), [
        ("Rd4xd7", ["d5", "d6"]), ("Rd4xg4", ["e4", "f4"]),
        ("Rd4xd1", ["d3", "d2"]), ("Rd4xa4", ["c4", "b4"]),
    ])
    def test_each_of_its_four_lines_names_what_it_cleared(
            self, text: str, cleared: list[str]) -> None:
        session = _rook_session()
        (move,) = [m["move"] for m in session.snapshot()["moves"] if m["text"].startswith(text)]
        before = session.state
        session.play(move)
        assert session.history[-1]["swept"] == [_square(name) for name in cleared]
        after = session.engine.apply(before, move)
        assert _did(session.engine, before, after, move)["sweep"] == [
            _square(name) for name in cleared]


class TestTheCut:
    def test_the_board_cuts_swept_pieces_instead_of_striking_them(self) -> None:
        assert "function sweepCut(" in PAGE
        assert PAGE.count("sweepCut(svg, node,") == 1
        assert "cut.has(sq)" in PAGE

    def test_the_landing_cuts_them_the_same_way(self) -> None:
        # The landing's stories play a sweep with the play board's cut, never the capture.
        assert PAGE.count("sweepCut(svg, moving,") == 1
        assert "capture(svg, before[square]" not in PAGE

    def test_a_story_names_the_squares_the_play_board_does(self) -> None:
        session = _session()
        engine = session.engine
        (move,) = [m["move"] for m in session.snapshot()["moves"]
                   if m["text"].startswith("Bc1xf4")]
        before = session.state
        after = engine.apply(before, move)
        assert _did(engine, before, after, move)["sweep"] == [_square("d2"), _square("e3")]

    def test_the_blade_draws_in_its_own_tokens(self) -> None:
        assert {"fx-blade", "fx-edge"} <= set(tokens.LIGHT)
        assert "var(--fx-blade)" in PAGE and "var(--fx-edge)" in PAGE
