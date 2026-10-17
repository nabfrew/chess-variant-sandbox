"""The play page for a spy game (schema 27, Q935c, `docs/plan-spy.md` §3.6).

The server half first, on `tests/_spy_small.json` and the plan's own positions (§1.5):

- **Every phase plays through the page's session**: turns while both spies stand, the reply
  turn, announced rounds (the announcer first, then the answerer knowing the move) and
  blind rounds, each step's kind the core's (`Engine.phase`), between two people at one
  screen, against the computer from either chair, and end to end through the web API.
- **The computer's answer is never asked before the announcement is in**, whatever the page
  asks first; as announcer it is asked when the round opens, with no announcement, and its
  move is shown at once.
- Undo takes back a turn one move and a round whole, to the last position a person chose
  in; a stored spy game replays from its flat move list; the roster is the spy ladder; and
  between two people at two screens (Q935e) the answerer waits for the announcement.

Then the page half, in node against the page's own functions: the phase line (`spyLine`) and
the announced arrow (`announcedArrow`, `announcedHtml`) as pure functions of the snapshot,
the legend's row for the arrow, the slot's *Announce* and *Answer*, and the move list's
turns and rounds in one list.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest
from test_web import _in_page

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.web import opponents
from chess_variant_sandbox.web.library import Game
from chess_variant_sandbox.web.play import HUMAN, Session, Sessions
from chess_variant_sandbox.web.server import (
    REQUEST_HEADER,
    REQUEST_VALUE,
    App,
    Server,
    handler_for,
)

SPEC_PATH = Path(__file__).parent / "_spy_small.json"


def _spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _session(seats: tuple[str, str], seed: int = 0) -> Session:
    spec = _spec()
    game = Game(hash="spy", name=spec["name"], description="", path=Path(), source="test",
                spec=spec)
    return Sessions().start(game, seats, seed=seed, scratch=True)


def _square(name: str) -> int:
    return (int(name[1]) - 1) * 5 + "abcde".index(name[0])


def _placed(session: Session, white: dict[str, str], black: dict[str, str], side: int = 0) -> None:
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    session.state = session.engine.make_state(placement, side_to_move=side)


def _move(session: Session, frm: str, to: str, seat: int | None = None) -> int:
    """The legal move from `frm` to `to`: the side to move's on a turn, `seat`'s in a round."""
    engine, state = session.engine, session.state
    moves = engine.legal_moves(state) if seat is None else engine.legal_moves_for(state, seat)
    found = [m for m in moves if move_from(m) == _square(frm) and move_to(m) == _square(to)]
    assert len(found) == 1, (frm, to, seat)
    return found[0]


# E1 (`plan-spy.md` §1.5): White to move takes the black spy; Black's reply turn retakes.
E1_WHITE = {"a1": "K", "a4": "R", "c2": "S"}
E1_BLACK = {"d5": "K", "c4": "S"}
# One spy each gone: White holds the only spy, Black announces (the position after E1's reply,
# with a black knight to move and White's king and spy to answer with).
HELD_WHITE = {"a1": "K", "c2": "S", "e1": "R"}
HELD_BLACK = {"d5": "K", "b4": "N"}


def _e1_into_rounds(session: Session) -> dict[str, Any]:
    """E1 by hand: the spy taken, the reply turn, and the first announced round opening."""
    _placed(session, E1_WHITE, E1_BLACK)
    opening = session.snapshot()
    assert opening["spy"] == {"phase": "alternating", "announcer": None}
    assert "round" not in opening and opening["sideToMove"] == 0
    reply = session.play(_move(session, "a4", "c4"))
    assert reply["spy"] == {"phase": "reply", "announcer": None}
    assert "round" not in reply and reply["sideToMove"] == 1
    return reply


# ------------------------------------------------------------------- the phases


class TestThePhasesAtOneScreen:
    def test_a_spy_game_starts_by_turns_and_is_not_refused(self) -> None:
        snapshot = _session((HUMAN, "greedy")).snapshot()
        assert snapshot["spy"] == {"phase": "alternating", "announcer": None}
        assert "round" not in snapshot and snapshot["sideToMove"] == 0
        engine = Engine(_spec())
        assert {m["move"] for m in snapshot["moves"]} == set(
            engine.legal_moves(engine.initial_state()))

    def test_e1_turns_then_the_reply_then_an_announced_round_then_its_reveal(self) -> None:
        session = _session((HUMAN, HUMAN))
        _e1_into_rounds(session)
        opened = session.play(_move(session, "d5", "c4"))
        assert opened["spy"] == {"phase": "announced", "announcer": 1}
        round_ = opened["round"]
        assert round_["announcer"] == 1 and round_["announced"] is None
        assert round_["choosing"] == 1 and round_["number"] == 1
        # The announcer is offered their own moves on the round's board, and nothing else.
        assert {m["move"] for m in opened["moves"]} == set(
            session.engine.legal_moves_for(session.state, 1))
        announcement = _move(session, "c4", "c3", seat=1)
        shown = session.play(announcement)
        round_ = shown["round"]
        assert round_["announced"]["move"] == announcement and round_["announced"]["text"]
        assert round_["sealed"] == [False, True] and round_["choosing"] == 0
        assert shown["sideToMove"] == 0 and not shown["awaitingAgent"]
        assert {m["move"] for m in shown["moves"]} == set(
            session.engine.legal_moves_for(session.state, 0))
        answer = _move(session, "c2", "d2", seat=0)
        revealed = session.play(answer)
        history = revealed["history"]
        assert [entry.get("round") for entry in history] == [None, None, 0, 0]
        assert [entry["player"] for entry in history] == [0, 1, 0, 1]
        assert history[-1]["reveal"]["announcer"] == 1
        assert revealed["round"]["number"] == 2 and revealed["round"]["announced"] is None
        assert session.played[-2:] == [answer, announcement]

    def test_the_announcement_is_shown_and_the_answerers_moves_are_its_own(self) -> None:
        """The announcement changes nothing the answerer may choose (§1.4, 3)."""
        session = _session((HUMAN, HUMAN))
        _placed(session, HELD_WHITE, HELD_BLACK, side=0)
        assert session.snapshot()["spy"] == {"phase": "announced", "announcer": 1}
        before = set(session.engine.legal_moves_for(session.state, 0))
        shown = session.play(_move(session, "b4", "c2", seat=1))
        assert {m["move"] for m in shown["moves"]} == before
        assert shown["round"]["announced"]["to"] == _square("c2")

    def test_blind_rounds_once_both_spies_are_lost(self) -> None:
        session = _session((HUMAN, "greedy"), seed=4)
        _placed(session, {"a1": "K", "e1": "R"}, {"e5": "K", "a5": "R"})
        snapshot = session.snapshot()
        assert snapshot["spy"] == {"phase": "blind", "announcer": None}
        assert "announcer" not in snapshot["round"] and snapshot["awaitingAgent"]
        sealed = session.play_agent()
        assert sealed["round"]["sealed"] == [False, True]
        assert sealed["history"] == []  # the computer's move is not shown before the reveal
        assert session.pending[1] is not None


# ------------------------------------------------------------------- the computer's order


class _Watched:
    """`Engine.select_round`, recording each seat it was asked for and what was announced."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[int, int | None]] = []
        asked = Engine.select_round

        def watched(engine: Engine, state: Any, seat: int, *args: Any, **kwargs: Any) -> Any:
            self.calls.append((seat, kwargs.get("announced")))
            return asked(engine, state, seat, *args, **kwargs)

        monkeypatch.setattr(Engine, "select_round", watched)


class TestTheComputersOrder:
    def test_its_answer_is_never_asked_before_the_announcement_is_in(self, monkeypatch) -> None:
        watched = _Watched(monkeypatch)
        session = _session(("greedy", HUMAN))
        _placed(session, HELD_WHITE, HELD_BLACK, side=0)
        opened = session.snapshot()
        assert opened["round"]["choosing"] == 1 and not opened["awaitingAgent"]
        # A page that asks the computer anyway is answered with nothing chosen.
        session.play_agent()
        assert watched.calls == [] and session.pending == [None, None]
        announcement = _move(session, "b4", "c2", seat=1)
        shown = session.play(announcement)
        assert watched.calls == [], "the seal asks the answerer nothing"
        assert shown["awaitingAgent"] and shown["moves"] == []
        assert shown["round"]["announced"]["move"] == announcement
        revealed = session.play_agent()
        assert watched.calls == [(0, announcement)]
        assert len(revealed["history"]) == 2 and revealed["history"][1]["reveal"]

    def test_as_announcer_it_is_asked_as_the_round_opens_and_shown_at_once(self, monkeypatch) -> None:
        watched = _Watched(monkeypatch)
        session = _session((HUMAN, "greedy"))
        _placed(session, HELD_WHITE, HELD_BLACK, side=0)
        opened = session.snapshot()
        assert opened["awaitingAgent"] and opened["moves"] == []
        assert opened["round"]["choosing"] is None and opened["sideToMove"] == 1
        shown = session.play_agent()
        assert watched.calls == [(1, None)]
        announced = shown["round"]["announced"]
        assert announced["move"] == session.pending[1] and announced["text"]
        assert not shown["awaitingAgent"] and shown["round"]["choosing"] == 0
        answer = shown["moves"][0]["move"]
        revealed = session.play(answer)
        assert watched.calls == [(1, None)]
        assert session.played[-2:] == [answer, announced["move"]]
        assert revealed["history"][-1]["reveal"]["announcer"] == 1

    def test_a_persons_answer_sent_unasked_waits_for_the_computers_announcement(
            self, monkeypatch) -> None:
        """A page that never asked still cannot answer ahead of the announcement: the
        seal asks the announcer first, then takes the answer, and the round is revealed."""
        watched = _Watched(monkeypatch)
        session = _session((HUMAN, "greedy"))
        _placed(session, HELD_WHITE, HELD_BLACK, side=0)
        answer = session.engine.legal_moves_for(session.state, 0)[0]
        revealed = session.play(answer)
        assert watched.calls == [(1, None)]
        assert len(revealed["history"]) == 2 and session.played[0] == answer

    def test_two_computers_show_the_announcement_before_the_answer(self) -> None:
        session = _session(("greedy", "minimax:nodes=400"))
        _placed(session, HELD_WHITE, HELD_BLACK, side=0)
        shown = session.play_agent()
        assert shown["round"]["announced"] is not None and shown["awaitingAgent"]
        assert shown["history"] == []
        revealed = session.play_agent()
        assert len(revealed["history"]) == 2


# ------------------------------------------------------------------- undo, replay, roster


class TestUndoReplayAndRoster:
    def test_undo_takes_a_round_whole_and_turns_back_to_the_persons(self) -> None:
        session = _session((HUMAN, "greedy"))
        _placed(session, E1_WHITE, E1_BLACK)
        session.play(_move(session, "a4", "c4"))
        session.play_agent()  # Black's reply turn
        assert session.snapshot()["spy"]["phase"] == "announced"
        session.play_agent()  # Black announces
        session.play(session.snapshot()["moves"][0]["move"])  # White answers
        assert len(session.history) == 4
        back = session.undo()
        assert len(session.history) == 2 and back["spy"]["phase"] == "announced"
        assert back["round"]["announced"] is None and back["awaitingAgent"]
        back = session.undo()
        assert len(session.history) == 0 and back["spy"]["phase"] == "alternating"
        assert back["sideToMove"] == 0 and not back["awaitingAgent"]

    def test_a_stored_spy_game_replays_from_its_flat_move_list(self) -> None:
        played = _session(("greedy", "random"), seed=2)
        snapshot = played.snapshot()
        while snapshot["outcome"] is None and len(played.history) < 80:
            snapshot = played.play_agent()
        assert any("round" in entry for entry in played.history), "the game reached rounds"
        again = _session(("greedy", "random"), seed=2)
        for move, depth in zip(played.played, played.depths, strict=True):
            again.replay(move, depth)
        first, second = played.snapshot(), again.snapshot()
        for key in ("board", "history", "spy", "outcome", "round"):
            assert first.get(key) == second.get(key), key

    def test_the_roster_is_the_spy_ladder(self) -> None:
        roster = opponents.for_game("spy", spec=_spec())
        agents = [entry.agent for entry in roster]
        assert "minimax:nodes=1600" in agents and "greedy" in agents
        assert not any(agent.startswith(("mcts:", "matrix:")) for agent in agents)
        assert opponents.seat_for("minimax:nodes=1600", roster) == "minimax:nodes=1600"

    def test_at_two_screens_the_answerer_waits_and_the_announcement_is_not_yours(self) -> None:
        """Between two people at two screens (`viewer`, Q935e): the answerer is offered
        nothing until the announcement is in, and the announcer's own announcement comes
        back as `announced`, which both readers get, never as a sealed move of their own."""
        session = _session((HUMAN, HUMAN))
        _placed(session, HELD_WHITE, HELD_BLACK, side=0)
        session.viewer = 0
        waiting = session.snapshot()
        assert waiting["moves"] == [] and waiting["round"]["choosing"] is None
        assert waiting["sideToMove"] == 1
        announcement = _move(session, "b4", "c2", seat=1)
        session.pending[1] = announcement
        answering = session.snapshot()
        assert answering["round"]["choosing"] == 0 and answering["moves"]
        assert answering["round"]["announced"]["move"] == announcement
        session.viewer = 1
        announced = session.snapshot()
        assert "yours" not in announced["round"] and announced["moves"] == []
        assert announced["round"]["announced"]["move"] == announcement
        assert announced["sideToMove"] == 0


# ------------------------------------------------------------------- end to end


class _Served:
    def __init__(self, tmp_path: Path) -> None:
        rules = tmp_path / "rulesets"
        rules.mkdir()
        self.app = App(rules, tmp_path / "runs")
        self.server = Server(("127.0.0.1", 0), handler_for(self.app))
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json",
                                   REQUEST_HEADER: REQUEST_VALUE}, method="POST")
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode())

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_a_spy_game_plays_through_its_phases_on_the_designers_board(tmp_path: Path) -> None:
    """The designer's *play it here* on a spy draft, through the web API: turns, the reply,
    announced rounds (the person announcing, the computer answering), to the game's end."""
    served = _Served(tmp_path)
    try:
        snapshot = served.post("/api/design/play", {"spec": _spec(), "seats": ["greedy", "human"],
                                                    "seed": 3})
        assert snapshot["seats"] == ["greedy", "human"]
        sid = snapshot["id"]
        phases: list[str] = []
        for _ in range(300):
            if snapshot["outcome"] is not None:
                break
            phases.append(snapshot["spy"]["phase"])
            if snapshot["awaitingAgent"]:
                snapshot = served.post(f"/api/session/{sid}/agent", {})
            else:
                snapshot = served.post(f"/api/session/{sid}/move",
                                       {"move": snapshot["moves"][0]["move"]})
        assert snapshot["outcome"] is not None
        assert {"alternating", "reply", "announced"} <= set(phases), phases
    finally:
        served.close()


# ------------------------------------------------------------------- the page, in node


def _hot_seat_snapshots() -> list[dict[str, Any]]:
    """E1 at one screen: the opening, the reply turn, the announced round before and after
    the announcement, and after its reveal."""
    session = _session((HUMAN, HUMAN))
    _placed(session, E1_WHITE, E1_BLACK)
    shots = [session.snapshot()]
    shots.append(session.play(_move(session, "a4", "c4")))
    shots.append(session.play(_move(session, "d5", "c4")))
    shots.append(session.play(_move(session, "c4", "c3", seat=1)))
    shots.append(session.play(_move(session, "c2", "d2", seat=0)))
    return shots


class TestThePage:
    def test_the_phase_line_is_a_function_of_the_snapshot(self) -> None:
        shots = _hot_seat_snapshots()
        lines = _in_page(
            f"const S = {json.dumps(shots)};",
            "S.map(s => spyLine(s.spy, s.round, s.sideToMove, s.ruleset.sides))",
        )
        assert lines[0] == "Both spies stand: the players take turns."
        assert lines[1] == ("black's spy is lost. black replies with one ordinary move; then "
                            "rounds begin, and black announces first.")
        assert lines[2] == ("Only white has a spy: black announces a move first, and white "
                            "answers knowing it.")
        text = shots[3]["round"]["announced"]["text"]
        assert lines[3] == f"black announced {text}; white answers."
        blind = _in_page("", "spyLine({phase: 'blind', announcer: null}, null, 0, ['A', 'B'])")
        assert blind == "Both spies are lost: both players choose in secret."

    def test_the_announced_arrow_is_a_function_of_the_round(self) -> None:
        shots = _hot_seat_snapshots()
        arrows = _in_page(f"const S = {json.dumps(shots)};",
                          "S.map(s => announcedArrow(s.round))")
        assert arrows[:3] == [None, None, None] and arrows[4] is None
        assert arrows[3] == {"from": _square("c4"), "to": _square("c3"), "dest": _square("c3"),
                             "player": 1, "text": shots[3]["round"]["announced"]["text"]}

    def test_the_arrow_is_its_own_mark_with_a_legend_row(self) -> None:
        shots = _hot_seat_snapshots()
        setup = (f"SESSION = {json.dumps(shots[3])}; FRAME = boardView(5, 5, wrapAxes(null),"
                 " {files: 0, ranks: 0}, false);")
        drawn = _in_page(setup, "announcedHtml(announcedArrow(SESSION.round), 20, 5, 5, null)")
        assert drawn.count('data-mark="announced"') == 2
        assert 'stroke="var(--announced)"' in drawn and "<line" in drawn
        assert "--held)" not in drawn and "stroke-dasharray" not in drawn
        drop = _in_page(setup, "announcedHtml({from: null, to: 7, dest: 7, player: 1}, 20, 5, 5,"
                               " null)")
        assert drop.count('data-mark="announced"') == 1 and "<line" not in drop
        legend = _in_page("", "legendEntries(new Set(['announced']), null, '', null)")
        assert len(legend) == 1 and "solid arrow" in legend[0] and "announced" in legend[0]
        key = _in_page("", "markKey('announced')")
        assert 'data-mark="announced"' in key

    def test_the_slot_announces_then_answers(self) -> None:
        shots = _hot_seat_snapshots()
        rows = _in_page(
            f"const S = {json.dumps(shots)};",
            "[2, 3].map(i => { SESSION = S[i]; const held = S[i].moves[0];"
            " return [announcedSendHtml(S[i].round, null)[0],"
            " announcedSendHtml(S[i].round, held)[1]]; })",
        )
        (says_first, button_first), (says_then, button_then) = rows
        assert "Announce your move" in says_first and ">Announce</button>" in button_first
        text = shots[3]["round"]["announced"]["text"]
        assert f"black announced {text}." in says_then and ">Answer</button>" in button_then

    def test_the_move_list_holds_the_turns_and_then_the_rounds(self) -> None:
        shots = _hot_seat_snapshots()
        rows = _in_page(f"SESSION = {json.dumps(shots[4])}; TIMELINE_FOR = null;",
                        "moveRows().map(row => row.html)")
        assert len(rows) == 3
        assert rows[0].startswith("1. ") and rows[1].startswith("&nbsp;&nbsp;&hellip; ")
        assert rows[2].startswith("round 1. ")
        assert rows[2].count("data-announced") == 1
