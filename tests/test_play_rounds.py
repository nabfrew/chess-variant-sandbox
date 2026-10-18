"""The play page for a game in rounds (schema 26, Q918c, `docs/archive/plan-simultaneous.md` §6.6).

What is held here, the server half first:

- **A round plays end to end through the web API**, over a real socket: a draft in the
  designer's play route and a library game's session, against the core's round player, to
  the game's end, each round two history entries in seat order.
- **The computer commits blind.** Its move is sealed on the round's board before the
  person's is accepted (`select_round` is asked while no person's move is stored, whatever
  the page asked first), it is the same move whatever the person then chooses, and no
  answer carries it before the reveal: two sessions whose computers sealed different moves
  answer byte for byte alike but for their ids, and neither answer names the move.
- Between two people at one screen, the second chooser is offered nothing of the first's.
- **The reveal** of each of §2.2's cases (a collision with no survivor and with a winner, a
  swap, a chase, a capture of a standing piece, a both-void swap, a one-void swap) carries
  the core's events, each move's result and the sentences; the void run and the bar (§2.7)
  are in the snapshot as the core's state and the barred capture is not offered.
- Undo takes a round back whole; a stored round game replays from its flat move list; the
  roster of a round game is its own ladder, and a page asking for *medium* by the usual
  agent string gets the round game's *medium*.

Then the page half, in node against the page's own functions: `roundStory` for each case
(the motions the reveal animates, a pure function of the two boards and the reveal, as
`portalJourney` is held), `roundMarks` for the run and the bar through a run, a reset and
a touch, `roundRowHtml`, and `animateRound` arriving the two pieces of a collision
together.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest
from test_designer import _kit_setup, _run_page
from test_portal_travel import _STAGE
from test_web import _in_page

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.web import cards, designing, opponents
from chess_variant_sandbox.web.library import Game
from chess_variant_sandbox.web.play import HUMAN, Session, Sessions
from chess_variant_sandbox.web.server import (
    REQUEST_HEADER,
    REQUEST_VALUE,
    App,
    Server,
    handler_for,
)

SPEC_PATH = Path(__file__).parent / "_simultaneous_gardner.json"
SERVE_FAST = {"poll_interval": 0.02}


def _spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _game(spec: dict[str, Any] | None = None) -> Game:
    spec = spec or _spec()
    return Game(hash="rounds", name=spec["name"], description="", path=Path(), source="test",
                spec=spec)


def _session(seats: tuple[str, str], seed: int = 0, spec: dict[str, Any] | None = None) -> Session:
    return Sessions().start(_game(spec), seats, seed=seed, scratch=True)


def _square(name: str) -> int:
    return (int(name[1]) - 1) * 5 + "abcde".index(name[0])


def _move(session: Session, seat: int, frm: str, to: str) -> int:
    """The legal move of `seat` from `frm` to `to` on the session's board."""
    engine = session.engine
    found = [
        move for move in engine.legal_moves_for(session.state, seat)
        if move_from(move) == _square(frm) and move_to(move) == _square(to)
    ]
    assert found, f"no move {frm}-{to} for seat {seat}"
    return found[0]


def _placed(session: Session, placement: dict[str, tuple[str, int]]) -> None:
    session.state = session.engine.make_state(placement)


def _round(session: Session, first: tuple[str, str], second: tuple[str, str]) -> dict[str, Any]:
    """One round between two people at one screen: seat 0's move, then seat 1's."""
    session.play(_move(session, 0, *first))
    return session.play(_move(session, 1, *second))


def _without_id(snapshot: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in snapshot.items() if k != "id"}, sort_keys=True)


# ------------------------------------------------------------------- end to end


class _Served:
    """The application on a real socket, over a ruleset directory holding the test spec."""

    def __init__(self, tmp_path: Path) -> None:
        rules = tmp_path / "rulesets"
        rules.mkdir()
        (rules / "simultaneous.json").write_text(SPEC_PATH.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
        self.app = App(rules, tmp_path / "runs")
        self.server = Server(("127.0.0.1", 0), handler_for(self.app))
        threading.Thread(target=self.server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.raw: list[str] = []

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json",
                                   REQUEST_HEADER: REQUEST_VALUE}, method="POST")
        with urlopen(request, timeout=60) as response:
            text = response.read().decode()
        self.raw.append(text)
        return json.loads(text)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def served(tmp_path: Path):
    served = _Served(tmp_path)
    try:
        yield served
    finally:
        served.close()


def _play_out(served: _Served, snapshot: dict[str, Any], rounds: int = 60) -> dict[str, Any]:
    """Play a person's first offered move each round, letting the computer seal first."""
    sid = snapshot["id"]
    for _ in range(rounds * 2):
        if snapshot["outcome"] is not None:
            break
        if snapshot["awaitingAgent"]:
            snapshot = served.post(f"/api/session/{sid}/agent", {})
            continue
        snapshot = served.post(f"/api/session/{sid}/move", {"move": snapshot["moves"][0]["move"]})
    return snapshot


class TestARoundEndToEnd:
    def test_a_draft_in_rounds_plays_in_the_designers_preview(self, served: _Served) -> None:
        """The designer's preview plays rounds through the same session, against the
        round ladder's *medium* when the page asks for the usual one."""
        snapshot = served.post("/api/design/play", {"spec": _spec()})
        assert snapshot["seats"] == ["human", "matrix:nodes=25600"]
        assert snapshot["round"]["number"] == 1 and snapshot["awaitingAgent"]
        assert snapshot["moves"] == []  # nothing is offered before the computer has sealed
        snapshot = _play_out(served, snapshot, rounds=8)
        history = snapshot["history"]
        assert len(history) % 2 == 0 and len(history) >= 2
        assert [entry["player"] for entry in history] == [0, 1] * (len(history) // 2)
        assert [entry["round"] for entry in history] == [i // 2 for i in range(len(history))]
        assert all("reveal" in entry for entry in history[1::2])

    def test_a_library_game_in_rounds_plays_to_its_end(self, served: _Served) -> None:
        digest = next(iter(served.app.known()))
        snapshot = served.post("/api/session", {"hash": digest, "seats": ["human", "greedy"]})
        snapshot = _play_out(served, snapshot)
        assert snapshot["outcome"] is not None, "Gardner in rounds ends within sixty rounds"
        assert snapshot["moves"] == [] and not snapshot["awaitingAgent"]

    def test_no_answer_carries_the_computers_sealed_move(self, served: _Served) -> None:
        """Two games whose computers sealed different moves answer alike but for their ids
        until the reveal, and no answer before it names the sealed move."""
        digest = next(iter(served.app.known()))
        sealed: dict[int, str] = {}
        answers: dict[int, str] = {}
        for seed in range(12):
            start = served.post("/api/session", {"hash": digest, "seats": ["human", "minimax:nodes=1600"],
                                                 "seed": seed})
            answer = served.post(f"/api/session/{start['id']}/agent", {})
            session = served.app.sessions.get(start["id"])
            move = session.pending[1]
            assert move is not None and answer["round"]["sealed"] == [False, True]
            text = session.engine.describe_move(session.state.with_(side_to_move=1), move)
            assert text not in served.raw[-1] and text not in served.raw[-2]
            assert answer["history"] == [] and answer["lastMove"] is None
            sealed[seed], answers[seed] = text, _without_id(answer)
        assert len(set(sealed.values())) > 1, "the seeds sealed more than one move"
        assert len(set(answers.values())) == 1


# ------------------------------------------------------------------- blindness


class TestTheComputerCommitsBlind:
    def test_it_seals_before_a_persons_move_is_accepted(self, monkeypatch) -> None:
        """Even where the page never asked for it: `seal` asks the computer first, and the
        computer is asked while no move of the person's is stored anywhere."""
        session = _session((HUMAN, "matrix:nodes=400"), seed=5)
        seen: list[list[int | None]] = []
        asked = Engine.select_round

        def watched(engine: Engine, *args: Any, **kwargs: Any) -> Any:
            seen.append(list(session.pending))
            return asked(engine, *args, **kwargs)

        monkeypatch.setattr(Engine, "select_round", watched)
        move = session.engine.legal_moves_for(session.state, 0)[0]
        session.play(move)
        assert seen == [[None, None]]
        assert len(session.history) == 2  # sealed, then revealed in the same request

    def test_its_move_is_the_same_whatever_the_person_chooses(self) -> None:
        probe = _session((HUMAN, "matrix:nodes=400"), seed=7)
        probe.play_agent()
        committed = probe.pending[1]
        for move in probe.engine.legal_moves_for(probe.state, 0):
            session = _session((HUMAN, "matrix:nodes=400"), seed=7)
            session.play(move)
            assert session.played == [move, committed]

    def test_between_two_people_the_second_is_shown_nothing_of_the_first(self) -> None:
        snapshots = set()
        for move in _session((HUMAN, HUMAN)).engine.legal_moves_for(
                _session((HUMAN, HUMAN)).state, 0):
            session = _session((HUMAN, HUMAN))
            after = session.play(move)
            assert after["round"]["sealed"] == [True, False]
            assert after["round"]["choosing"] == 1 and after["sideToMove"] == 1
            assert {entry["move"] for entry in after["moves"]} == set(
                session.engine.legal_moves_for(session.state, 1))
            snapshots.add(_without_id(after))
        assert len(snapshots) == 1


# ------------------------------------------------------------------- the reveal

KINGS = {"a1": ("K", 0), "e5": ("K", 1)}


class TestTheReveal:
    def test_a_collision_with_no_survivor(self) -> None:
        session = _session((HUMAN, HUMAN))
        _placed(session, {**KINGS, "b1": ("N", 0), "d5": ("N", 1)})
        snapshot = _round(session, ("b1", "c3"), ("d5", "c3"))
        reveal = snapshot["lastMove"]["reveal"]
        assert reveal["events"] == [{"kind": "collision", "square": _square("c3"), "player": -1}]
        assert reveal["results"] == ["both lost", "both lost"]
        assert "met on c3: both were lost" in " ".join(reveal["words"])
        assert snapshot["board"][_square("c3")] == 0

    def test_a_collision_with_a_winner(self) -> None:
        spec = _spec()
        pawn = next(piece for piece in spec["pieces"] if piece["id"] == "P")
        pawn["beats"] = []
        session = _session((HUMAN, HUMAN), spec=spec)
        _placed(session, {**KINGS, "b1": ("N", 0), "c4": ("P", 1)})
        snapshot = _round(session, ("b1", "c3"), ("c4", "c3"))
        reveal = snapshot["lastMove"]["reveal"]
        assert reveal["events"][0] == {"kind": "collision", "square": _square("c3"), "player": 0}
        assert reveal["results"] == ["won", "lost"]
        assert [h["result"] for h in snapshot["history"]] == ["won", "lost"]

    def test_a_swap(self) -> None:
        session = _session((HUMAN, HUMAN))
        _placed(session, {**KINGS, "b1": ("R", 0), "b3": ("R", 1)})
        snapshot = _round(session, ("b1", "b3"), ("b3", "b1"))
        reveal = snapshot["lastMove"]["reveal"]
        assert reveal["events"] == [] and reveal["results"] == [None, None]
        assert "swapped squares" in reveal["words"][0]

    def test_a_chase_and_a_capture_of_a_standing_piece(self) -> None:
        session = _session((HUMAN, HUMAN))
        _placed(session, {**KINGS, "b1": ("R", 0), "b4": ("N", 1)})
        chase = _round(session, ("b1", "b4"), ("b4", "c2"))["lastMove"]["reveal"]
        assert chase["events"] == [] and chase["words"] == []
        session = _session((HUMAN, HUMAN))
        _placed(session, {**KINGS, "b1": ("R", 0), "b4": ("N", 1)})
        standing = _round(session, ("b1", "b4"), ("e5", "d5"))
        assert standing["lastMove"]["reveal"]["events"] == []
        assert standing["board"][_square("b4")] == standing["board"][_square("b4")] != 0

    def test_both_void_then_barred(self) -> None:
        """Two void rounds in a row: a run of one, then the pair barred and its capture no
        longer offered (§2.7), all of it the core's state as the snapshot carries it."""
        session = _session((HUMAN, HUMAN))
        _placed(session, {"a1": ("K", 0), "a5": ("K", 1), "c2": ("P", 0), "d3": ("P", 1)})
        pair = [_square("c2"), _square("d3")]
        first = _round(session, ("c2", "d3"), ("d3", "c2"))
        reveal = first["lastMove"]["reveal"]
        assert [e["kind"] for e in reveal["events"]] == ["void", "void"]
        assert reveal["results"] == ["void", "void"]
        assert first["round"]["voidRun"] == pair and first["round"]["bars"] == []
        second = _round(session, ("c2", "d3"), ("d3", "c2"))
        assert second["round"]["bars"] == [pair] and second["round"]["voidRun"] is None
        assert "barred" in [e["kind"] for e in second["lastMove"]["reveal"]["events"]]
        offered = {(m["from"], m["to"]) for m in second["moves"]}
        assert (pair[0], pair[1]) not in offered

    def test_a_different_round_resets_the_run_and_a_touch_lifts_the_bar(self) -> None:
        session = _session((HUMAN, HUMAN))
        _placed(session, {"a1": ("K", 0), "a5": ("K", 1), "c2": ("P", 0), "d3": ("P", 1)})
        pair = [_square("c2"), _square("d3")]
        assert _round(session, ("c2", "d3"), ("d3", "c2"))["round"]["voidRun"] == pair
        assert _round(session, ("a1", "a2"), ("a5", "a4"))["round"]["voidRun"] is None
        _round(session, ("c2", "d3"), ("d3", "c2"))
        assert _round(session, ("c2", "d3"), ("d3", "c2"))["round"]["bars"] == [pair]
        assert _round(session, ("a2", "a1"), ("a4", "a5"))["round"]["bars"] == [pair]
        assert _round(session, ("c2", "c3"), ("a5", "a4"))["round"]["bars"] == []

    def test_a_one_void_swap_takes_the_void_piece(self) -> None:
        session = _session((HUMAN, HUMAN))
        _placed(session, {**KINGS, "c2": ("P", 0), "d3": ("B", 1)})
        snapshot = _round(session, ("c2", "d3"), ("d3", "c2"))
        reveal = snapshot["lastMove"]["reveal"]
        assert reveal["events"] == [{"kind": "void", "square": _square("c2"), "player": 0}]
        assert reveal["results"] == ["void", None]
        assert snapshot["board"][_square("c2")] != 0 and snapshot["board"][_square("d3")] == 0


# ------------------------------------------------------------------- the rest


class TestRoundsAndTheSession:
    def test_undo_takes_a_round_back_and_the_computer_seals_again(self) -> None:
        session = _session((HUMAN, "matrix:nodes=400"), seed=2)
        start = session.state
        session.play_agent()
        session.play(session.engine.legal_moves_for(session.state, 0)[0])
        session.play_agent()
        assert session.pending[1] is not None
        snapshot = session.undo()
        assert session.state == start and session.history == [] and session.played == []
        assert session.pending == [None, None] and snapshot["awaitingAgent"]

    def test_a_stored_round_game_replays_from_its_flat_move_list(self) -> None:
        played = _session((HUMAN, "greedy"), seed=4)
        for _ in range(5):
            played.play_agent()
            played.play(played.engine.legal_moves_for(played.state, 0)[-1])
        rebuilt = _session((HUMAN, "greedy"), seed=4)
        for move, depth in zip(played.played, played.depths, strict=True):
            rebuilt.replay(move, depth)
        assert rebuilt.state == played.state
        assert rebuilt.history == played.history

    def test_a_round_game_offers_its_own_ladder(self) -> None:
        roster = opponents.for_game("rounds", spec=_spec())
        agents = [entry.agent for entry in roster]
        assert agents == [agent for agent, _name, _tail in opponents.ROUND_RUNGS]
        assert not any(agent.startswith(("minimax", "mcts")) for agent in agents)
        assert opponents.seat_for("minimax:nodes=1600", roster) == "matrix:nodes=25600"
        assert opponents.seat_for("minimax:nodes=200000", roster) == "matrix:nodes=102400"
        assert opponents.seat_for("greedy", roster) == "greedy"
        assert opponents.seat_for("minimax:nodes=1600", opponents.ladder()) == "minimax:nodes=1600"
        medium = next(entry for entry in roster if entry.name == "medium")
        assert medium.about == "weighs 25,600 rounds before each move"

    def test_the_round_rungs_are_the_alternating_ones_at_sixteen_times(self) -> None:
        """Q934c: *easy*, *medium* and *hard* in rounds play `ROUND_BUDGET_FACTOR` times the
        alternating rungs' budgets, the measuring roster's rule (Q944a), so no rung is
        greedy under another name (`round-roster-budgets.md` §9)."""
        from chess_variant_sandbox.loop.screen import ROUND_BUDGET_FACTOR

        alternating = {name: agent for agent, name, _ in opponents.RUNGS}
        for agent, name, _ in opponents.ROUND_RUNGS:
            if name in ("easy", "medium", "hard"):
                budget = int(alternating[name].split("=")[1]) * ROUND_BUDGET_FACTOR
                assert agent == f"matrix:nodes={budget}"
        assert [agent for agent, _, _ in opponents.ROUND_RUNGS[3:]] == [
            "matrix:nodes=6400", "matrix:nodes=25600", "matrix:nodes=102400"
        ]

    def test_a_round_ladder_plays_its_rungs_against_each_other(self) -> None:
        """Every computer rung of a round game is its ladder's spine, so the matrix players
        meet one another rather than being ordered by their games against greedy alone."""
        entries = opponents.agents_of(opponents.ladder(rounds=True))
        plan = opponents._plan(entries, opponents.LADDER_GAMES, opponents.LADDER_SEED)
        assert plan.config.spine == opponents.ROUND_SPINE == tuple(entries)
        assert plan.config.games_for("matrix:nodes=6400", "matrix:nodes=102400") > 0
        # A game that alternates keeps its own five as the spine.
        alternating = opponents._plan(opponents.agents_of(opponents.ladder()), 2, 0)
        assert alternating.config.spine == opponents.SPINE

    def test_a_round_game_never_says_check_or_offers_a_draw(self) -> None:
        snapshot = _session((HUMAN, "greedy")).snapshot()
        assert snapshot["inCheck"] is False and snapshot["drawOffer"] is None


# ------------------------------------------------------------------- the page


def _reveal_case(placement: dict[str, tuple[str, int]], first: tuple[str, str],
                 second: tuple[str, str], spec: dict[str, Any] | None = None):
    session = _session((HUMAN, HUMAN), spec=spec)
    _placed(session, placement)
    before = list(session.state.board)
    snapshot = _round(session, first, second)
    return before, snapshot


def _story(before: list[int], snapshot: dict[str, Any]) -> dict[str, Any]:
    return _in_page(
        f"const S = {json.dumps(snapshot)}; const B = {json.dumps(before)};",
        "roundStory(B, S.board, S.lastMove.reveal)",
    )


class TestThePage:
    def test_each_case_of_a_round_is_a_story_of_its_own(self) -> None:
        c3 = _square("c3")
        before, snap = _reveal_case({**KINGS, "b1": ("N", 0), "d5": ("N", 1)},
                                    ("b1", "c3"), ("d5", "c3"))
        story = _story(before, snap)
        assert [m["kind"] for m in story["moves"]] == ["lose", "lose"]
        assert story["meeting"] == {"square": c3, "winner": -1} and story["taken"] == []

        spec = _spec()
        next(p for p in spec["pieces"] if p["id"] == "P")["beats"] = []
        before, snap = _reveal_case({**KINGS, "b1": ("N", 0), "c4": ("P", 1)},
                                    ("b1", "c3"), ("c4", "c3"), spec)
        assert [m["kind"] for m in _story(before, snap)["moves"]] == ["win", "lose"]

        before, snap = _reveal_case({**KINGS, "b1": ("R", 0), "b3": ("R", 1)},
                                    ("b1", "b3"), ("b3", "b1"))
        story = _story(before, snap)
        assert [m["kind"] for m in story["moves"]] == ["arrive", "arrive"]
        assert story["taken"] == [] and story["meeting"] is None

        before, snap = _reveal_case({**KINGS, "c2": ("P", 0), "d3": ("B", 1)},
                                    ("c2", "d3"), ("d3", "c2"))
        story = _story(before, snap)
        assert [(m["kind"], m.get("taken")) for m in story["moves"]] == [
            ("void", True), ("arrive", None)]
        assert story["taken"] == []

        before, snap = _reveal_case({"a1": ("K", 0), "a5": ("K", 1), "c2": ("P", 0),
                                     "d3": ("P", 1)}, ("c2", "d3"), ("d3", "c2"))
        story = _story(before, snap)
        assert [(m["kind"], m["taken"]) for m in story["moves"]] == [("void", False),
                                                                     ("void", False)]

        before, snap = _reveal_case({**KINGS, "b1": ("R", 0), "b4": ("N", 1)},
                                    ("b1", "b4"), ("e5", "d5"))
        story = _story(before, snap)
        assert [m["kind"] for m in story["moves"]] == ["arrive", "arrive"]
        assert story["taken"] == [_square("b4")]

    def test_the_standoff_marks_follow_the_cores_run_and_bar(self) -> None:
        """A void round shows a run of one; a second in a row shows the pair barred; a
        different round in between clears the run; a touch of either square clears the bar."""
        session = _session((HUMAN, HUMAN))
        _placed(session, {"a1": ("K", 0), "a5": ("K", 1), "c2": ("P", 0), "d3": ("P", 1)})
        pair = [_square("c2"), _square("d3")]
        steps = [
            (("c2", "d3"), ("d3", "c2"), [{"kind": "voidRun", "squares": pair}]),
            (("a1", "a2"), ("a5", "a4"), []),
            (("c2", "d3"), ("d3", "c2"), [{"kind": "voidRun", "squares": pair}]),
            (("c2", "d3"), ("d3", "c2"), [{"kind": "barred", "squares": pair}]),
            (("a2", "a1"), ("a4", "a5"), [{"kind": "barred", "squares": pair}]),
            (("c2", "c3"), ("a5", "a4"), []),
        ]
        rounds = [_round(session, first, second)["round"] for first, second, _ in steps]
        marks = _in_page(f"const R = {json.dumps(rounds)};", "R.map(roundMarks)")
        assert marks == [want for _, _, want in steps]

    def test_a_rounds_row_names_both_moves_and_what_they_came_to(self) -> None:
        before, snap = _reveal_case({**KINGS, "b1": ("N", 0), "d5": ("N", 1)},
                                    ("b1", "c3"), ("d5", "c3"))
        row = _in_page(f"const S = {json.dumps(snap)}; SESSION = S;",
                       "roundRowHtml(1, S.history[0], S.history[1])")
        assert row.startswith("1. Nb1-c3") and "Nd5-c3" in row
        assert row.count('data-result="both lost"') == 2 and "met on c3" in row

    def test_a_collisions_two_pieces_arrive_together_and_both_break(self) -> None:
        """`animateRound` times the shorter trip to wait, so the two pieces meet; both
        losers are struck on arrival, at the same moment."""
        before, snap = _reveal_case({"a1": ("K", 0), "e5": ("K", 1), "c1": ("R", 0),
                                     "d5": ("N", 1)}, ("c1", "c3"), ("d5", "c3"))
        setup = f"SESSION = {json.dumps(snap)}; const BEFORE = {json.dumps(before)};" + _STAGE
        log = _in_page(setup, "(jplay(), JLOG.map(e => ({ who: e.who, delay: e.opts.delay || 0,"
                              " duration: e.opts.duration || 0, n: e.frames.length })))")
        travels = [entry for entry in log if entry["n"] == 2 or entry["n"] == 3]
        fades = [entry for entry in log if entry["who"].startswith("g") and entry["n"] == 3
                 and entry["duration"] == 390]
        arrivals = {round(entry["delay"] + entry["duration"]) for entry in travels
                    if entry["duration"] not in (390, 330)}
        assert len(arrivals) == 1, log
        assert len(fades) == 2 and len({entry["delay"] for entry in fades}) == 1


class TestTheDesigner:
    def test_the_designer_writes_a_game_in_rounds_and_a_beats_list_that_validate(self) -> None:
        """*+ rule* adds `simultaneous` through the page's own default, turning a mate
        game's win over to the royal's capture (§2.5), and the piece card's *beats* row
        names pieces; the draft both make validates and plays in rounds."""
        assert ("beats", "pieces") in [(f["key"], f["control"]) for f in cards.card_model()["piece"]]
        spec = _spec()
        del spec["rules"]["simultaneous"]
        spec["rules"]["win"] = "checkmate"
        spec["schema_version"] = 25
        drafted = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(spec)};"
            " const field = cardModel().rules.find(f => f.key === 'simultaneous');"
            " DRAFT.rules.simultaneous = fieldDefault(field, ['simultaneous'], 'rules');"
            " RULE_COMPANIONS.simultaneous(DRAFT.rules);"
            " DRAFT.pieces.find(p => p.id === 'P').beats = [];",
            "DRAFT",
        )
        assert drafted["rules"]["simultaneous"] == {} and drafted["rules"]["win"] == "royal_capture"
        settled = designing.settled(drafted)
        assert validate_spec(settled) == [] and settled["schema_version"] == 26
        session = _session((HUMAN, HUMAN), spec=settled)
        _placed(session, {**KINGS, "b1": ("N", 0), "c4": ("P", 1)})
        snapshot = _round(session, ("b1", "c3"), ("c4", "c3"))
        assert snapshot["lastMove"]["reveal"]["results"] == ["won", "lost"]


# ------------------------------------------- schema 27's chase switches, on the page


def _chase_spec() -> dict[str, Any]:
    """Q942a's test spec with all three switches: `rest`, `swap: "collision"` and `both:
    "score"` under material scoring."""
    return json.loads((SPEC_PATH.parent / "_simultaneous_scored.json").read_text(encoding="utf-8"))


class TestTheChaseSwitches:
    """Q942b (`docs/archive/plan-round-chase.md` §4.2): the page marks a rested piece, draws a swap
    that met between its two squares, and says a both-fire's result in a sentence."""

    def test_each_rested_piece_is_in_the_round_and_badged(self) -> None:
        session = _session((HUMAN, HUMAN), spec=_chase_spec())
        _placed(session, {**KINGS, "c2": ("P", 0), "c4": ("P", 1)})
        opening = session.snapshot()
        assert opening["round"]["rest"] == [None, None]
        snapshot = _round(session, ("a1", "a2"), ("e5", "e4"))
        rest = [_square("a2"), _square("e4")]
        assert snapshot["round"]["rest"] == rest
        # The core leaves the rested king's moves out; the page only marks why.
        assert all(m["from"] != _square("a2") for m in snapshot["moves"])
        rests = _in_page(f"const R = {json.dumps(snapshot['round'])};", "roundRests(R)")
        assert rests == [{"player": 0, "square": rest[0]}, {"player": 1, "square": rest[1]}]
        badge = _in_page("", "restBadge(0, 0, 10, 1)")
        assert 'data-mark="rested"' in badge and 'fill="var(--rest)"' in badge
        legend = _in_page("", "legendEntries(new Set(['rested', 'clash']), null, '', null)")
        assert any("pause sign" in entry and "rests" in entry for entry in legend)
        assert any("burst" in entry and "swapped squares" in entry for entry in legend)

    def test_a_swap_that_met_is_a_clash_between_its_two_squares(self) -> None:
        before, snap = _reveal_case({**KINGS, "b1": ("R", 0), "b3": ("R", 1)},
                                    ("b1", "b3"), ("b3", "b1"), _chase_spec())
        reveal = snap["lastMove"]["reveal"]
        assert reveal["events"] == [{"kind": "swap", "square": _square("b3"), "player": -1}]
        assert reveal["results"] == ["both lost", "both lost"]
        clash = _in_page(f"const S = {json.dumps(snap)};", "roundClash(S.lastMove)")
        assert clash == {"squares": [_square("b1"), _square("b3")], "winner": -1}
        story = _story(before, snap)
        assert [(m["kind"], m.get("clash")) for m in story["moves"]] == [
            ("lose", True), ("lose", True)]
        assert story["clash"] == clash and story["taken"] == []
        drawn = _in_page("", "clashMark(5, 5, 5, 25, 10)")
        assert 'data-mark="clash"' in drawn and "<polygon" in drawn

    def test_a_swap_that_met_breaks_both_pieces_half_way_at_one_moment(self) -> None:
        before, snap = _reveal_case({**KINGS, "b1": ("R", 0), "b3": ("R", 1)},
                                    ("b1", "b3"), ("b3", "b1"), _chase_spec())
        setup = f"SESSION = {json.dumps(snap)}; const BEFORE = {json.dumps(before)};" + _STAGE
        log = _in_page(setup, "(jplay(), JLOG.map(e => ({ who: e.who, delay: e.opts.delay || 0,"
                              " duration: e.opts.duration || 0, n: e.frames.length })))")
        fades = [entry for entry in log if entry["who"].startswith("g") and entry["n"] == 3
                 and entry["duration"] == 390]
        assert len(fades) == 2 and len({entry["delay"] for entry in fades}) == 1, log

    def test_without_the_switch_a_swap_is_no_clash(self) -> None:
        _, snap = _reveal_case({**KINGS, "b1": ("R", 0), "b3": ("R", 1)},
                               ("b1", "b3"), ("b3", "b1"))
        assert _in_page(f"const S = {json.dumps(snap)};", "roundClash(S.lastMove)") is None

    def test_a_both_fire_settled_on_material_is_said_in_a_sentence(self) -> None:
        """The two kings strike each other: a swap that meets, both royals taken in one
        round, settled by `rules.scoring` for the side a rook ahead."""
        session = _session((HUMAN, HUMAN), spec=_chase_spec())
        _placed(session, {"a1": ("K", 0), "b2": ("K", 1), "e1": ("R", 0)})
        snapshot = _round(session, ("a1", "b2"), ("b2", "a1"))
        outcome = snapshot["outcome"]
        assert outcome["winner"] == 0 and outcome["reason"] == "royal_captured"
        assert outcome["how"] == "on material, as both royal pieces were taken in the same round"
        said = _in_page(f"SESSION = {json.dumps(snapshot)};", "verdictText(SESSION.outcome)")
        assert said.endswith("wins on material, as both royal pieces were taken in the same round.")

    def test_a_level_both_fire_is_a_draw_in_a_sentence(self) -> None:
        session = _session((HUMAN, HUMAN), spec=_chase_spec())
        _placed(session, {"a1": ("K", 0), "b2": ("K", 1)})
        outcome = _round(session, ("a1", "b2"), ("b2", "a1"))["outcome"]
        assert outcome["winner"] is None
        assert outcome["how"] == (
            "as both royal pieces were taken in the same round and the material was level")

    def test_one_royal_taken_keeps_the_reasons_words(self) -> None:
        session = _session((HUMAN, HUMAN), spec=_chase_spec())
        _placed(session, {"a1": ("K", 0), "e5": ("K", 1), "e1": ("R", 0), "c4": ("P", 1)})
        outcome = _round(session, ("e1", "e5"), ("c4", "c3"))["outcome"]
        assert outcome == {"winner": 0, "reason": "royal_captured"}
