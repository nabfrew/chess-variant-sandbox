"""The play page for a game with speeds (schema 28, Q943b, `docs/plan-speed.md` §4.5).

A person must see who moved first and why the second move did what it did. The server
half: each piece's speed crosses to its card and the ranks to the legend, and a round's
sentences lead with the order and say what came of the slower move, off the core's events.
The page half, in node against the page's own functions: `roundStory`'s ordered motions for
§2.5's cases A (an unplayed flight its piece is caught in), B (a slower move lost with its
piece), C in both forms (a capture of the faster piece where it landed, a push made void), D
(a slider blocked) and an unplayed move whose piece still stands; `animateRound` moving the
two in sequence, the faster first, and fading a lost move's arrow as the capture lands
(the user, 2026-10-16); the order marks, the move list's superscripts, a card's speed and
its hover line, and the legend's line of ranks.
"""

from __future__ import annotations

import json
from typing import Any

from test_play_rounds import _in_page, _move, _placed, _session, _square
from test_portal_travel import _STAGE
from test_speed import gardner

from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.web.library import Game, pieces
from chess_variant_sandbox.web.play import HUMAN


def _spec(speeds: dict[str, int]) -> dict[str, Any]:
    spec = gardner(speeds)
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _case(speeds: dict[str, int], placement: dict[str, tuple[str, int]],
          first: tuple[str, str], second: tuple[str, str]) -> tuple[list[int], dict[str, Any]]:
    """One round between two people at one screen from a hand-built position: the board
    before it and the snapshot after."""
    session = _session((HUMAN, HUMAN), spec=_spec(speeds))
    _placed(session, placement)
    before = list(session.state.board)
    session.play(_move(session, 0, *first))
    return before, session.play(_move(session, 1, *second))


def _story(before: list[int], snapshot: dict[str, Any]) -> dict[str, Any]:
    return _in_page(
        f"const S = {json.dumps(snapshot)}; const B = {json.dumps(before)};",
        "roundStory(B, S.board, S.lastMove.reveal)",
    )


def _kinds(story: dict[str, Any]) -> list[tuple[int, str, Any]]:
    return [(m["phase"], m["kind"], m.get("taken")) for m in story["moves"]]


# §2.5 A: player 0's rook (speed 1) strikes a5 first; player 1's king (0) is caught there,
# its flight to b4 never played.
CASE_A = ({"R": 1}, {"e1": ("K", 0), "a1": ("R", 0), "a5": ("K", 1)}, ("a1", "a5"), ("a5", "b4"))
# §2.5 B, the king faster: it takes the queen on b5 where it stands, and her move is lost.
CASE_B = ({"K": 1}, {"e1": ("K", 0), "b5": ("Q", 0), "b4": ("K", 1)}, ("b5", "b4"), ("b4", "b5"))
# §2.5 C: the knight (speed 1) lands on c3 first; the queen then takes it there, or the
# pawn's push onto it is void.
C_PIECES = {"e1": ("K", 0), "b1": ("N", 0), "a5": ("Q", 1), "c4": ("P", 1), "e5": ("K", 1)}
CASE_C = ({"N": 1}, C_PIECES, ("b1", "c3"), ("a5", "c3"))
CASE_C_PUSH = ({"N": 1}, C_PIECES, ("b1", "c3"), ("c4", "c3"))
# §2.5 D: the knight lands on c3, on the rook's road to e3.
CASE_D = ({"N": 1}, {"e1": ("K", 0), "b1": ("N", 0), "a3": ("R", 1), "e5": ("K", 1)},
          ("b1", "c3"), ("a3", "e3"))
# The rook takes the king; the pawn's push elsewhere was never played, and the pawn stands.
UNPLAYED = ({"R": 1}, {"e1": ("K", 0), "a1": ("R", 0), "a5": ("K", 1), "c4": ("P", 1)},
            ("a1", "a5"), ("c4", "c3"))


class TestTheServer:
    def test_a_speed_crosses_to_each_card_and_the_ranks_to_the_legend(self) -> None:
        spec = _spec({"N": 2, "R": 1})
        game = Game(hash="speed", name=spec["name"], description="", path=None,  # type: ignore[arg-type]
                    source="test", spec=spec)
        speeds = {piece["id"]: piece["speed"] for piece in pieces(game)}
        assert speeds == {"P": 0, "N": 2, "B": 0, "R": 1, "Q": 0, "K": 0}
        snapshot = _session((HUMAN, HUMAN), spec=spec).snapshot()
        assert snapshot["ruleset"]["speed"] == [0, 2, 0, 1, 0, 0]

    def test_a_game_without_a_speed_carries_none(self) -> None:
        spec = gardner(None)
        game = Game(hash="plain", name=spec["name"], description="", path=None,  # type: ignore[arg-type]
                    source="test", spec=spec)
        assert all("speed" not in piece for piece in pieces(game))
        assert _session((HUMAN, HUMAN), spec=spec).snapshot()["ruleset"]["speed"] == []

    def test_the_words_lead_with_the_order(self) -> None:
        _, snap = _case(*CASE_A)
        words = snap["lastMove"]["reveal"]["words"]
        assert words[0] == "White's rook was faster: Ra1xa5."
        assert "the game" in words[1] and "black's king's move was not played" in words[1]
        _, snap = _case(*CASE_B)
        reveal = snap["lastMove"]["reveal"]
        assert reveal["words"] == [
            "Black's king was faster: Kb4xb5.",
            "Black's king took white's queen on b5 before it could move: its move was lost.",
        ]
        _, snap = _case(*CASE_C)
        assert snap["lastMove"]["reveal"]["words"] == [
            "White's knight was faster: Nb1-c3.", "Black's queen then took it on c3."]
        _, snap = _case(*CASE_D)
        words = snap["lastMove"]["reveal"]["words"]
        assert words[0] == "White's knight was faster: Nb1-c3."
        assert "could no longer go to e3" in words[1] and "void" in words[1]


class TestThePage:
    def test_case_a_the_caught_flight_is_unplayed_and_taken(self) -> None:
        before, snap = _case(*CASE_A)
        story = _story(before, snap)
        assert story["ordered"] == 0
        assert _kinds(story) == [(1, "arrive", None), (2, "unplayed", True)]
        assert story["taken"] == [_square("a5")]

    def test_case_b_the_slower_move_is_lost_with_its_piece(self) -> None:
        before, snap = _case(*CASE_B)
        story = _story(before, snap)
        assert story["ordered"] == 1
        assert _kinds(story) == [(2, "lost", True), (1, "arrive", None)]
        assert story["taken"] == [_square("b5")]

    def test_case_c_the_faster_piece_falls_where_it_landed(self) -> None:
        before, snap = _case(*CASE_C)
        story = _story(before, snap)
        assert _kinds(story) == [(1, "fell", None), (2, "arrive", None)]
        assert story["taken"] == []

    def test_case_c_a_push_onto_it_is_void(self) -> None:
        before, snap = _case(*CASE_C_PUSH)
        assert _kinds(_story(before, snap)) == [(1, "arrive", None), (2, "void", False)]

    def test_case_d_a_blocked_slider_is_void(self) -> None:
        before, snap = _case(*CASE_D)
        story = _story(before, snap)
        assert _kinds(story) == [(1, "arrive", None), (2, "void", False)]
        assert story["taken"] == []

    def test_an_unplayed_move_whose_piece_stands_is_not_taken(self) -> None:
        before, snap = _case(*UNPLAYED)
        story = _story(before, snap)
        assert _kinds(story) == [(1, "arrive", None), (2, "unplayed", False)]
        assert story["taken"] == [_square("a5")]

    def test_an_equal_speed_round_keeps_its_one_beat(self) -> None:
        before, snap = _case({"R": 1}, {"e1": ("K", 0), "b1": ("N", 0), "d5": ("N", 1),
                                        "e5": ("K", 1)}, ("b1", "c3"), ("d5", "c3"))
        story = _story(before, snap)
        assert "ordered" not in story and story["meeting"] == {"square": _square("c3"),
                                                               "winner": -1}

    def test_the_order_marks_and_the_move_list(self) -> None:
        _, snap = _case(*CASE_C)
        marks = _in_page(f"const S = {json.dumps(snap)};", "roundOrder(S.lastMove)")
        assert marks == [{"square": _square("c3"), "n": 1, "player": 0},
                         {"square": _square("c3"), "n": 2, "player": 1}]
        row = _in_page(f"const S = {json.dumps(snap)}; SESSION = S;",
                       "roundRowHtml(1, S.history[0], S.history[1])")
        assert 'Nb1-c3<sup data-order="1"' in row and 'Qa5-c3<sup data-order="2"' in row
        _, snap = _case(*CASE_D)
        marks = _in_page(f"const S = {json.dumps(snap)};", "roundOrder(S.lastMove)")
        assert marks == [{"square": _square("c3"), "n": 1, "player": 0},
                         {"square": _square("a3"), "n": 2, "player": 1}]
        _, snap = _case(*CASE_A)
        assert _in_page(f"const S = {json.dumps(snap)};", "roundOrder(S.lastMove)") == [
            {"square": _square("a5"), "n": 1, "player": 0}]
        row = _in_page(f"const S = {json.dumps(snap)}; SESSION = S;",
                       "roundRowHtml(1, S.history[0], S.history[1])")
        assert 'data-result="unplayed">not played' in row
        badge = _in_page("", "orderBadge(0, 0, 10, 2, 1)")
        assert 'data-mark="order"' in badge and ">2</text>" in badge
        legend = _in_page("", "legendEntries(new Set(['order']), null, '', null)")
        assert any("faster moves first" in entry for entry in legend)

    def test_a_cards_speed_and_its_hover_line(self) -> None:
        assert _in_page("", "speedLine(2, [0, 1, 2])") == "speed 2: moves before speeds 0 and 1"
        assert _in_page("", "speedLine(0, [0, 1, 2])") == "speed 0: moves after speeds 1 and 2"
        assert _in_page("", "speedLine(1, [0, 1, 2])") == (
            "speed 1: moves before speed 0, after speed 2")
        pill = _in_page("", "piecesHtml({id: 'N', name: 'knight', speed: 2, moves: [], notes: []},"
                            " {ranks: [0, 1, 2]})")
        assert 'class="speed"' in pill and 'title="speed 2: moves before speeds 0 and 1"' in pill
        plain = _in_page("", "piecesHtml({id: 'N', name: 'knight', moves: [], notes: []}, {})")
        assert 'class="speed"' not in plain
        line = _in_page("", "speedLegendHtml([0, 2, 0, 1, 0, 1], "
                            "['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'])")
        assert "knight 2; rook, king 1; pawn, bishop, queen 0" in line
        assert _in_page("", "speedLegendHtml([], ['pawn'])") == ""


def _log(before: list[int], snap: dict[str, Any]) -> list[dict[str, Any]]:
    setup = f"SESSION = {json.dumps(snap)}; const BEFORE = {json.dumps(before)};" + _STAGE
    return _in_page(setup, "(jplay(), JLOG.map(e => ({ who: e.who, delay: e.opts.delay || 0,"
                           " duration: e.opts.duration || 0, frames: e.frames })))")


class TestTheReveal:
    def test_the_slower_move_starts_after_the_faster_lands(self) -> None:
        """Case C: the knight's ghost leaps to c3 alone; the queen sets out only once it has
        landed, and the knight's ghost is struck when she arrives."""
        before, snap = _case(*CASE_C)
        log = _log(before, snap)
        travels = [e for e in log if "transform" in e["frames"][0]
                   and e["frames"][-1].get("transform") == "translate(0px, 0px)"]
        knight = next(e for e in travels if e["who"] == "g")
        queen = next(e for e in travels if e["who"] == f"piece {_square('c3')}")
        assert queen["delay"] >= knight["delay"] + knight["duration"], log
        struck = [e for e in log if e["who"] == "g" and e["duration"] == 390]
        assert struck and struck[0]["delay"] > queen["delay"], log

    def test_a_lost_moves_arrow_fades_as_the_capture_lands(self) -> None:
        """Case B: the queen's sealed move is an arrow from b5 to b4, at full strength until the
        king's capture lands on her and faded by the time she is knocked off."""
        before, snap = _case(*CASE_B)
        log = _log(before, snap)
        arrow = next(e for e in log if len(e["frames"]) == 3
                     and all(set(frame) <= {"opacity", "offset"} for frame in e["frames"])
                     and e["frames"][0]["opacity"] == 0.95)
        struck = next(e for e in log if e["who"] == "g" and e["duration"] == 390)
        fades_at = arrow["frames"][1]["offset"] * arrow["duration"]
        assert abs(fades_at - struck["delay"]) < 1, (arrow, struck)
        assert abs(arrow["duration"] - (struck["delay"] + 390)) < 1

    def test_an_unplayed_move_is_a_struck_through_ghost_after_the_faster(self) -> None:
        before, snap = _case(*UNPLAYED)
        log = _log(before, snap)
        rook = next(e for e in log if e["who"] == f"piece {_square('a5')}")
        ghost = next(e for e in log if len(e["frames"]) == 4 and e["frames"][0] == {"opacity": 0})
        assert ghost["delay"] > rook["delay"] + rook["duration"], log
