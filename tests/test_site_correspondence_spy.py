"""A spy game between two people, by correspondence (schema 27, Q935e, `docs/plan-spy.md`
§3.7).

A spy game is played by turns while both spies stand; once a spy falls its player has one
ordinary reply, and then the game is played in rounds: **announced** ones while one player
alone has a spy (the other announces a move first, in the open, and the spy's holder answers
it), **blind** ones once both are lost. What is proved here, on a spy spec built for it (Q935a's
`tests/_spy_small.json` with a setup two moves from either kind of round), rather than on
chess:

- **the phases**: a spy game starts between people (the refusal is lifted); a turn and the
  reply turn are ordinary moves; the reply opens an announced round its own player announces;
  a reply that takes the last spy opens a blind one;
- **the announced round**: the announcement waits in the row, is shown to **both** readers
  and to nobody else, starts the answerer's time and is final; the answerer cannot answer
  before it; the answer reveals the round once, in seat order, under a race too; the game
  survives a restart between the two;
- **blind rounds** are Q918f's rounds, and nothing sealed in one leaks;
- **the mails** say whose turn it is and what is asked: the reply, the announcement, the
  answer to a named announcement, the secret rounds beginning; a move that leaves its mover
  owing again mails nobody;
- **the clock**: a turn's is the side to move's; in an announced round exactly one player
  owes, so the sweep resigns that one and never draws it;
- **the page**: the strip and slot of an announced round between two people, and a turn's
  phase line, run in node over hand-made snapshots.

Nothing here works out a legal move, a phase or a winner: every move is one the engine
offered the reader (`snapshot_for`'s `moves`), every phase the core's, every outcome the
engine's or the store's.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as every
site test does.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from site_words import hits
from test_site_matches import Scratch, _scratch

from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import mail, move_email
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library

SITE = "https://site.example"
NAME = "spies by post"


def _spy() -> dict[str, Any]:
    """Q935a's 5x5 spy spec with a setup two moves from either kind of round: White's rook
    takes the black spy (a turn), and Black's king replies by retaking the rook (then Black,
    without a spy, announces) or by taking the white spy (then both are lost: blind)."""
    spec = json.loads((Path(__file__).parent / "_spy_small.json").read_text("utf-8"))
    spec["name"] = NAME
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "player": 0, "squares": ["a1"]},
        {"piece": "R", "player": 0, "squares": ["a4"]},
        {"piece": "S", "player": 0, "squares": ["c3"]},
        {"piece": "P", "player": 0, "squares": ["e2"]},
        {"piece": "K", "player": 1, "squares": ["d4"]},
        {"piece": "S", "player": 1, "squares": ["c4"]},
        {"piece": "P", "player": 1, "squares": ["a5"]},
    ]}
    return spec


SPY = _spy()
SPY_HASH = spec_hash(SPY)


class Sent:
    """A sender that keeps the message instead of posting it."""

    def __init__(self) -> None:
        self.messages: list[mail.Message] = []

    def __call__(self, message: mail.Message) -> None:
        self.messages.append(message)


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(SPY_HASH, SPY)
        yield scratch


@pytest.fixture
def sent() -> Sent:
    return Sent()


@pytest.fixture
def app(store: Scratch, sent: Sent, monkeypatch: pytest.MonkeyPatch) -> Iterator[SiteApp]:
    monkeypatch.setenv(mail.URL_VAR, SITE)
    made = SiteApp(library.RULESET_DIR, library.RUNS_DIR, connect=store.open, send=sent)
    yield made
    made.viewer = None


def _as(app: SiteApp, viewer: int | None) -> SiteApp:
    app.viewer = viewer
    return app


def _pair(store: Scratch) -> tuple[m.Row, int, int]:
    ada, bo = store.user("ada"), store.user("bo")
    found = c.start(store.connection, SPY_HASH, [ada, bo])
    store.connection.commit()
    return found, ada, bo


def _read(store: Scratch, match_id: str, user: int | None, connection: Any = None) -> dict[str, Any]:
    """The game as this person may see it: the one way a route draws it."""
    connection = store.connection if connection is None else connection
    return c.snapshot_for(connection, m.row(connection, match_id), user)


def _offered(store: Scratch, match_id: str, user: int) -> dict[str, int]:
    """The moves the engine offers this person now, by their words."""
    return {str(move["text"]): int(move["move"]) for move in _read(store, match_id, user)["moves"]}


def _act(store: Scratch, match_id: str, user: int, text: str) -> dict[str, Any]:
    """This person's move by its words: a turn, an announcement, an answer or a seal."""
    return c.move(store.connection, match_id, user, _offered(store, match_id, user)[text])


def _played(store: Scratch, match_id: str) -> list[int]:
    return list(m.session(store.connection, match_id).played)


def _into_announced(store: Scratch) -> tuple[m.Row, int, int]:
    """White takes the black spy, Black's reply retakes: an announced round, Black's to
    announce (Black has no spy), White's to answer."""
    found, ada, bo = _pair(store)
    _act(store, found.id, ada, "Ra4xc4")
    _act(store, found.id, bo, "Kd4xc4")
    store.connection.commit()
    return found, ada, bo


def _into_blind(store: Scratch) -> tuple[m.Row, int, int]:
    """White takes the black spy, Black's reply takes the white one: blind rounds."""
    found, ada, bo = _pair(store)
    _act(store, found.id, ada, "Ra4xc4")
    _act(store, found.id, bo, "Kd4xc3")
    store.connection.commit()
    return found, ada, bo


def _owing(store: Scratch, match_id: str) -> tuple[int, ...]:
    return c.owing(store.connection, m.row(store.connection, match_id))


def _moved(store: Scratch, match_id: str) -> Any:
    with store.connection.cursor() as cursor:
        cursor.execute("select moved from matches where id = %s", (match_id,))
        return cursor.fetchone()[0]


# ------------------------------------------------------------------------- the phases


class TestThePhases:
    def test_a_spy_game_starts_between_people_by_turns(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        seen = _read(store, found.id, ada)
        assert seen["spy"] == {"phase": "alternating", "announcer": None}
        assert "round" not in seen and seen["sideToMove"] == 0 and seen["moves"]
        assert seen["correspondence"]["rules"] == list(c.SPY_RULES)
        assert _owing(store, found.id) == (0,)
        for line in c.SPY_RULES:
            assert hits(line) == [], (line, hits(line))

    def test_a_turn_then_the_reply_then_the_round_its_player_announces(
        self, store: Scratch
    ) -> None:
        found, ada, bo = _pair(store)
        _act(store, found.id, ada, "Ra4xc4")
        store.connection.commit()
        assert len(_played(store, found.id)) == 1
        reply = _read(store, found.id, bo)
        assert reply["spy"] == {"phase": "reply", "announcer": None} and "round" not in reply
        assert _owing(store, found.id) == (1,)
        with pytest.raises(m.MatchError, match="not your move"):
            c.move(store.connection, found.id, ada, 0)
        store.connection.rollback()
        _act(store, found.id, bo, "Kd4xc4")
        assert len(_played(store, found.id)) == 2
        opened = _read(store, found.id, bo)
        assert opened["spy"] == {"phase": "announced", "announcer": 1}
        assert opened["round"]["announcer"] == 1 and opened["round"]["announced"] is None
        assert opened["round"]["choosing"] == 1 and opened["moves"]
        assert _owing(store, found.id) == (1,), "the reply's player announces next"

    def test_a_reply_that_takes_the_last_spy_opens_blind_rounds(self, store: Scratch) -> None:
        found, ada, bo = _into_blind(store)
        seen = _read(store, found.id, ada)
        assert seen["spy"] == {"phase": "blind", "announcer": None}
        assert "announcer" not in seen["round"] and seen["round"]["choosing"] == 0
        assert _owing(store, found.id) == (0, 1)


# ---------------------------------------------------------------- the announced round


class TestTheAnnouncedRound:
    def test_the_announcement_is_shown_to_both_and_starts_the_answerers_time(
        self, store: Scratch
    ) -> None:
        found, ada, bo = _into_announced(store)
        clock = _moved(store, found.id)
        answer_moves = _offered(store, found.id, ada)
        assert answer_moves == {}, "nothing is offered to the answerer before the announcement"
        _act(store, found.id, bo, "Kc4-c5")
        assert len(_played(store, found.id)) == 2, "an announcement is not a move made"
        assert _moved(store, found.id) > clock, "the answerer's time starts at the announcement"
        assert _owing(store, found.id) == (0,)
        for reader, theirs in ((ada, True), (bo, False)):
            seen = _read(store, found.id, reader)
            round_ = seen["round"]
            assert round_["announced"]["text"] == "Kc4-c5", reader
            assert round_["sealed"] == [False, True]
            assert "yours" not in round_, "an announcement is not a sealed move of one's own"
            assert bool(seen["moves"]) is theirs and (round_["choosing"] == 0) is theirs
            assert seen["sideToMove"] == 0
        # The answerer's moves are their own on the round's board: the announcement changes
        # nothing they may choose (§1.4, 3).
        assert set(_offered(store, found.id, ada)) == {
            "Ka1-a2", "Ka1-b1", "Ka1-b2", "Pe2-e3", "Sc3-b3", "Sc3-c2", "Sc3-d3"}

    def test_an_announcement_declines_the_answerers_draw_offer(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        c.offer_draw(store.connection, found.id, ada)
        _act(store, found.id, bo, "Kc4-c5")
        assert c.offer_of(store.connection, found.id) is None, "announcing is moving"

    def test_the_answerer_cannot_answer_first(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        seen = _read(store, found.id, ada)
        assert seen["moves"] == [] and seen["round"]["choosing"] is None
        assert seen["sideToMove"] == 1, "the strip names the announcer"
        legal = m.session(store.connection, found.id)
        any_move = legal.engine.legal_moves_for(legal.state, 0)[0]
        with pytest.raises(m.MatchError, match="announces a move first"):
            c.move(store.connection, found.id, ada, any_move)
        store.connection.rollback()
        assert c.sealed_seat(store.connection, found.id) is None

    def test_the_announcer_cannot_change_it(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        other = _offered(store, found.id, bo)["Pa5-a4"]
        _act(store, found.id, bo, "Kc4-c5")
        store.connection.commit()
        with pytest.raises(m.MatchError, match="announcement is made"):
            c.move(store.connection, found.id, bo, other)
        store.connection.rollback()
        assert _read(store, found.id, ada)["round"]["announced"]["text"] == "Kc4-c5"

    def test_the_answer_reveals_the_round_in_seat_order(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        announcement = _offered(store, found.id, bo)["Kc4-c5"]
        c.move(store.connection, found.id, bo, announcement)
        answer = _offered(store, found.id, ada)["Ka1-a2"]
        revealed = c.move(store.connection, found.id, ada, answer)
        assert _played(store, found.id)[2:] == [answer, announcement], "White's first"
        assert c.sealed_seat(store.connection, found.id) is None
        assert revealed["history"][-1]["reveal"]["announcer"] == 1
        assert len(c.times_of(store.connection, found.id)) == 4
        assert revealed["round"]["number"] == 2 and revealed["round"]["announced"] is None
        assert _owing(store, found.id) == (1,), "Black announces the next round too"

    def test_the_game_survives_a_restart_between_announcement_and_answer(
        self, store: Scratch
    ) -> None:
        found, ada, bo = _into_announced(store)
        _act(store, found.id, bo, "Kc4-c5")
        store.connection.commit()
        fresh = store.open()  # a process that never held the position
        try:
            seen = _read(store, found.id, ada, fresh)
            assert seen["round"]["announced"]["text"] == "Kc4-c5"
            answer = {str(n["text"]): int(n["move"]) for n in seen["moves"]}["Sc3-c2"]
            c.move(fresh, found.id, ada, answer)
            fresh.commit()
        finally:
            fresh.close()
        store.connection.commit()
        assert len(_played(store, found.id)) == 4

    def test_nobody_outside_the_game_reads_the_announcement(
        self, store: Scratch, app: SiteApp
    ) -> None:
        found, _ada, bo = _into_announced(store)
        _act(store, found.id, bo, "Kc4-c5")
        store.connection.commit()
        cy = store.user("cy")
        with pytest.raises(KeyError, match="no game at this address"):
            _as(app, cy).get(f"/api/session/{found.id}")


# ------------------------------------------------------------------------- the race


def _race(store: Scratch, match_id: str, first: tuple[int, int],
          second: tuple[int, int]) -> list[BaseException]:
    """Two connections, as two requests are: the first holds its transaction open while the
    second arrives, which is the race at its narrowest."""
    one, two = store.open(), store.open()
    errors: list[BaseException] = []
    try:
        c.move(one, match_id, *first)

        def late() -> None:
            try:
                c.move(two, match_id, *second)
                two.commit()
            except BaseException as error:  # noqa: BLE001 - reported to the test
                two.rollback()
                errors.append(error)

        thread = threading.Thread(target=late)
        thread.start()
        thread.join(timeout=1.0)
        assert thread.is_alive(), "the second waits on the first's lock"
        one.commit()
        thread.join(timeout=30.0)
        assert not thread.is_alive()
    finally:
        one.close()
        two.close()
    return errors


class TestRacing:
    def test_an_answer_arriving_with_the_announcement_reveals_once(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        announcement = _offered(store, found.id, bo)["Kc4-c5"]
        session = m.session(store.connection, found.id)
        answer = next(n for n in session.engine.legal_moves_for(session.state, 0)
                      if session.engine.describe_move(session.state, n) == "Ka1-a2")
        store.connection.commit()
        errors = _race(store, found.id, (bo, announcement), (ada, answer))
        assert errors == []
        assert _played(store, found.id)[2:] == [answer, announcement]
        assert c.sealed_seat(store.connection, found.id) is None

    def test_the_answer_racing_itself_reveals_once(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        _act(store, found.id, bo, "Kc4-c5")
        offered = _offered(store, found.id, ada)
        store.connection.commit()
        errors = _race(store, found.id, (ada, offered["Ka1-a2"]), (ada, offered["Pe2-e3"]))
        assert len(errors) == 1 and "announces a move first" in str(errors[0])
        assert len(_played(store, found.id)) == 4, "one round revealed, once"

    def test_the_announcement_racing_itself_is_made_once(self, store: Scratch) -> None:
        found, _ada, bo = _into_announced(store)
        offered = _offered(store, found.id, bo)
        store.connection.commit()
        errors = _race(store, found.id, (bo, offered["Kc4-c5"]), (bo, offered["Pa5-a4"]))
        assert len(errors) == 1 and "announcement is made" in str(errors[0])
        assert _read(store, found.id, bo)["round"]["announced"]["text"] == "Kc4-c5"


# ------------------------------------------------------------------ blind rounds leak nothing


def _alike(answer: dict[str, Any]) -> str:
    kept = json.loads(json.dumps(answer))
    kept.pop("id", None)
    for stamp in ("started", "times"):  # when each game started and its moves landed
        (kept.get("correspondence") or {}).pop(stamp, None)
    return json.dumps(kept, sort_keys=True)


class TestBlindRounds:
    def test_a_blind_round_seals_and_reveals_as_any_round(self, store: Scratch) -> None:
        found, ada, bo = _into_blind(store)
        clock = _moved(store, found.id)
        sealed = _act(store, found.id, ada, "Ka1-a2")
        assert sealed["round"]["yours"]["text"] == "Ka1-a2"
        assert _moved(store, found.id) == clock, "a seal does not move a blind round's clock"
        theirs = _read(store, found.id, bo)
        assert "yours" not in theirs["round"] and theirs["round"]["sealed"] == [True, False]
        assert "Ka1-a2" not in json.dumps(theirs)
        _act(store, found.id, bo, "Kc3-d3")
        assert len(_played(store, found.id)) == 4 and _owing(store, found.id) == (0, 1)

    def test_the_other_seats_reads_are_alike_whatever_was_sealed(
        self, store: Scratch, app: SiteApp
    ) -> None:
        one, ada, bo = _into_blind(store)
        two = c.start(store.connection, SPY_HASH, [ada, bo])
        _act(store, two.id, ada, "Ra4xc4")
        _act(store, two.id, bo, "Kd4xc3")
        _act(store, one.id, ada, "Ka1-a2")
        _act(store, two.id, ada, "Pe2-e3")
        store.connection.commit()
        answers = [_as(app, bo).get(f"/api/session/{found.id}") for found in (one, two)]
        for said in map(json.dumps, answers):
            assert "Ka1-a2" not in said and "Pe2-e3" not in said and '"yours"' not in said
        assert _alike(answers[0]) == _alike(answers[1])


# ---------------------------------------------------------------------------- the mails


def _ticked(store: Scratch) -> tuple[m.Row, int, int]:
    found, ada, bo = _pair(store)
    for user in (ada, bo):
        move_email.set_on(store.connection, user, True)
    store.connection.commit()
    return found, ada, bo


def _post(app: SiteApp, store: Scratch, match_id: str, user: int, text: str) -> dict[str, Any]:
    chosen = _offered(store, match_id, user)[text]
    store.connection.commit()
    return _as(app, user).post(f"/api/session/{match_id}/move", {"move": chosen})


def _rows(store: Scratch) -> list[tuple[Any, ...]]:
    with store.connection.cursor() as cursor:
        cursor.execute("select recipient, ply from move_emails order by id")
        found = cursor.fetchall()
    store.connection.commit()
    return found


class TestTheMails:
    def test_each_says_whose_turn_and_what_is_asked(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, ada, bo = _ticked(store)
        _post(app, store, found.id, ada, "Ra4xc4")
        assert [message.to for message in sent.messages] == ["bo@example.invalid"]
        assert sent.messages[0].body.startswith(
            f"ada played Ra4xc4 in your game of {NAME}. Your spy is lost, so it's your move: "
            "one ordinary reply, and then the game is played in rounds:")
        _post(app, store, found.id, bo, "Kd4xc4")
        assert len(sent.messages) == 1, "bo owes the announcement: nobody is mailed"
        _post(app, store, found.id, bo, "Kc4-c5")
        assert [message.to for message in sent.messages][1:] == ["ada@example.invalid"]
        answer = sent.messages[1]
        assert answer.subject == f"Your answer: {NAME} against bo"
        assert answer.body.startswith(
            f"bo announced Kc4-c5 in your game of {NAME}. Your answer is due: choose your "
            "move with theirs in view, and both are made at once:")
        _post(app, store, found.id, ada, "Ka1-a2")
        assert [message.to for message in sent.messages][2:] == ["bo@example.invalid"]
        reveal = sent.messages[2]
        assert reveal.subject == f"Your announcement: {NAME} against ada"
        assert reveal.body.startswith(
            f"Round 1 of your game of {NAME} against ada is revealed: you announced Kc4-c5 "
            "and ada answered Ka1-a2.")
        assert "It's your turn to announce: your move is shown to ada, who answers it" in (
            reveal.body)
        assert _rows(store) == [(bo, 1), (ada, 2), (bo, 4)], "still one email a turn"
        for message in sent.messages:
            assert hits(message.body) == [], hits(message.body)

    def test_the_secret_rounds_beginning_is_said_and_a_seal_names_nothing(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, ada, bo = _ticked(store)
        _post(app, store, found.id, ada, "Ra4xc4")
        _post(app, store, found.id, bo, "Kd4xc3")
        assert sent.messages[1].to == "ada@example.invalid"
        assert sent.messages[1].body.startswith(
            f"bo played Kd4xc3 in your game of {NAME}. Both spies are lost, so from now on "
            "each of you seals a move in secret. It's your move: seal yours")
        _post(app, store, found.id, ada, "Ka1-a2")
        sealed = sent.messages[2]
        assert sealed.to == "bo@example.invalid"
        assert sealed.body.startswith(f"ada's move is in, sealed, in your game of {NAME}.")
        assert "Ka1-a2" not in sealed.body and "Ka1-a2" not in sealed.subject
        assert _rows(store) == [(bo, 1), (ada, 2), (bo, 2)]

    def test_the_message_builder_refuses_an_unknown_ask(self) -> None:
        with pytest.raises(ValueError, match="no such thing to ask"):
            mail.your_move(to="a@example.invalid", game="G", opponent="bo", played="a1-a2",
                           link=f"{SITE}/#/play/x", off=f"{SITE}/mail/off/t", asked="dance")


# ---------------------------------------------------------------------------- the clock


class TestTheClock:
    def test_a_turn_is_lost_by_the_side_to_move(self, store: Scratch) -> None:
        found, ada, _bo = _pair(store)
        store.backdate(found.id, timedelta(days=6))
        assert [row.id for row in m.sweep(store.connection)] == [found.id]
        row = m.row(store.connection, found.id)
        assert row.left and row.score_for(row.seat_of(r.person(ada))) == 0.0

    def test_an_announcer_who_does_not_announce_loses_and_it_is_never_a_draw(
        self, store: Scratch
    ) -> None:
        found, _ada, bo = _into_announced(store)
        store.backdate(found.id, timedelta(days=6))
        assert [row.id for row in m.sweep(store.connection)] == [found.id]
        row = m.row(store.connection, found.id)
        assert row.left and row.reason == "resigned"
        assert row.score_for(row.seat_of(r.person(bo))) == 0.0, "bo owed the announcement"

    def test_the_announcement_restarts_the_clock_and_then_the_answerer_owes(
        self, store: Scratch
    ) -> None:
        found, ada, bo = _into_announced(store)
        store.backdate(found.id, timedelta(days=4))
        _act(store, found.id, bo, "Kc4-c5")
        store.connection.commit()
        store.backdate(found.id, timedelta(days=2))
        assert m.sweep(store.connection) == ()
        store.backdate(found.id, timedelta(days=4))
        assert [row.id for row in m.sweep(store.connection)] == [found.id]
        row = m.row(store.connection, found.id)
        assert row.left and row.score_for(row.seat_of(r.person(ada))) == 0.0, "ada owed"
        assert c.sealed_seat(store.connection, found.id) is None

    def test_the_time_left_names_whoever_owes(self, store: Scratch) -> None:
        found, ada, bo = _into_announced(store)
        assert _read(store, found.id, ada)["correspondence"]["toMove"].startswith("bo has")
        assert _read(store, found.id, bo)["correspondence"]["toMove"].startswith("You have")
        _act(store, found.id, bo, "Kc4-c5")
        assert _read(store, found.id, ada)["correspondence"]["toMove"].startswith("You have")
        assert _read(store, found.id, bo)["correspondence"]["toMove"].startswith("ada has")


# ---------------------------------------------------------------------------- the routes


class TestTheRoutes:
    def test_a_whole_announced_round_through_the_routes(
        self, store: Scratch, app: SiteApp
    ) -> None:
        found, ada, bo = _pair(store)
        _post(app, store, found.id, ada, "Ra4xc4")
        _post(app, store, found.id, bo, "Kd4xc4")
        mine = {e["id"]: e["yourMove"] for e in _as(app, bo).get("/api/profile")["between"]["running"]}
        theirs = {e["id"]: e["yourMove"]
                  for e in _as(app, ada).get("/api/profile")["between"]["running"]}
        assert mine[found.id] and not theirs[found.id]
        shown = _post(app, store, found.id, bo, "Kc4-c5")
        assert shown["round"]["announced"]["text"] == "Kc4-c5" and shown["moves"] == []
        read = _as(app, ada).get(f"/api/session/{found.id}")
        assert read["round"]["announced"]["text"] == "Kc4-c5" and read["moves"]
        theirs = {e["id"]: e["yourMove"]
                  for e in _as(app, ada).get("/api/profile")["between"]["running"]}
        assert theirs[found.id]
        revealed = _post(app, store, found.id, ada, "Sc3-d3")
        assert revealed["history"][-1]["reveal"]["announcer"] == 1
        assert revealed["round"]["number"] == 2


# ------------------------------------------------------------------- the page, in node


_ANNOUNCED = """
SESSION = { id: 'g1', ruleset: { sides: ['white', 'black'], wrap: null },
  seats: ['human', 'human'], sideToMove: 1, inCheck: false, remaining: 1, outcome: null,
  board: new Array(25).fill(0), spy: { phase: 'announced', announcer: 1 },
  history: [{ player: 0, text: 'Ra4xc4' }, { player: 1, text: 'Kd4xc4' }],
  moves: [],
  round: { number: 1, sealed: [false, false], choosing: null, tie: 'both', initiative: -1,
    voidRun: null, bars: [], announcer: 1, announced: null },
  correspondence: { you: 0, players: ['ada', 'bo'], drawOffer: null, mayCancel: false,
    times: [], toMove: 'bo has 5 days to move.', rules: [] } };
"""

# bo has announced Kc4-c5: ada answers.
_SHOWN = """
SESSION.sideToMove = 0; SESSION.moves = [{ move: 3, text: 'Ka1-a2', from: 0, to: 5 }];
SESSION.round.sealed = [false, true]; SESSION.round.choosing = 0;
SESSION.round.announced = { move: 9, text: 'Kc4-c5', from: 17, to: 22 };
SESSION.correspondence.toMove = 'You have 5 days to move.';
"""

_BO = "SESSION.correspondence.you = 1;"

_SLOT = """
const SLOT = { hidden: false, innerHTML: '', held: false,
  classList: { toggle: (name, on) => { if (name === 'held') SLOT.held = !!on; } } };
document.getElementById = id => (id === 'corrSend' ? SLOT : null);
const KEPT = {};
window.localStorage = { getItem: k => (k in KEPT ? KEPT[k] : null),
  setItem: (k, v) => { KEPT[k] = String(v); }, removeItem: k => { delete KEPT[k]; } };
drawBoard = () => {};
"""


def _said(text: Any) -> str:
    return " ".join(str(text).split())


class TestThePage:
    def test_the_strip_says_each_readers_part(self) -> None:
        from test_web import _in_page

        # One node run: the reader's four parts, by changing the snapshot between reads.
        waiting, answering, made, announcing = (_said(text) for text in _in_page(
            _ANNOUNCED + "const SAID = [corrStatus()];" + _SHOWN + "SAID.push(corrStatus());"
            + _BO + "SAID.push(corrStatus());" + "SESSION.round.announced = null;"
            "SESSION.round.choosing = 1; SAID.push(corrStatus());", "SAID"))
        assert "<b>bo announces first.</b> You answer once their move is in." in waiting
        assert "bo has 5 days to move." in waiting
        assert "<b>bo announced Kc4-c5.</b> Your answer is due." in answering
        assert "<b>Your move to announce.</b> ada holds the only spy" in announcing
        assert "<b>You announced Kc4-c5.</b> ada answers it." in made

    def test_the_slot_announces_then_answers(self) -> None:
        from test_web import _in_page

        held = _in_page(_ANNOUNCED + _BO + _SLOT
                        + "SESSION.round.choosing = 1; SESSION.moves = [{ move: 9, text: "
                        "'Kc4-c5', from: 17, to: 22 }]; CORR_HELD = 9; drawCorrSend();", "SLOT")
        assert held["held"] and ">Announce</button>" in held["innerHTML"]
        answer = _in_page(_ANNOUNCED + _SHOWN + _SLOT + "CORR_HELD = 3; drawCorrSend();", "SLOT")
        assert ">Answer</button>" in answer["innerHTML"]
        waiting = _said(_in_page(_ANNOUNCED + _SLOT + "drawCorrSend();", "SLOT")["innerHTML"])
        assert "<b>bo announces first.</b>" in waiting and 'id="sealHeld"' not in waiting
        made = _said(_in_page(_ANNOUNCED + _SHOWN + _BO + _SLOT + "drawCorrSend();",
                              "SLOT")["innerHTML"])
        assert "<b>Your announcement is made:</b> Kc4-c5. ada answers it." in made

    def test_a_turn_is_sent_as_any_move_and_says_its_phase(self) -> None:
        from test_web import _in_page

        turn = """
        SESSION.round = undefined; SESSION.spy = { phase: 'reply', announcer: null };
        SESSION.sideToMove = 1; SESSION.correspondence.you = 1;
        SESSION.moves = [{ move: 9, text: 'Kd4xc4', from: 18, to: 17 }];
        """
        strip = _said(_in_page(_ANNOUNCED + turn, "corrStatus()"))
        assert strip.startswith("<b>Your move.</b>")
        assert "bo&#39;s spy is lost. bo replies with one ordinary move" in strip
        slot = _in_page(_ANNOUNCED + turn + _SLOT + "CORR_HELD = 9; drawCorrSend();", "SLOT")
        assert 'id="sendHeld"' in slot["innerHTML"], "the send button, not the phase alone"
