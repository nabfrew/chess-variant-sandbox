"""A game in rounds between two people, by correspondence (Q918f, `docs/archive/plan-simultaneous.md`
§6.6 and §7's Q918f row).

Each player **seals** a move for the round's position, in secret; the second seal
**reveals** both at once, through the core. What is proved here, on a round spec (Gardner's
minichess played in rounds, `tests/_simultaneous_gardner.json`) rather than on chess:

- **seal and reveal**: a first seal is stored and changes nothing else (not the moves, not
  the round's clock, not a draw offer); the second appends both moves in seat order, keeps
  each one's time and lapses a standing offer; a sealed move is final; the engine refuses an
  illegal one; a round game ends by its rules as any game does;
- **atomicity**: two seals racing on two connections reveal the round once, and the same
  seat racing itself seals once;
- **nothing leaks**: the other seat's reads of the game, its answers to its own acts and
  every list a person reads are byte for byte alike whatever was sealed against them, and
  never carry the sealed move; a finish drops a round never revealed; no module but
  `site/correspondence.py` names the column;
- **the mails**: a seal mails the other player that a move is in and names none; a reveal
  mails the first sealer the round, both moves and the core's sentences; still one email a
  turn;
- **the clock**: it is the round's, unmoved by a seal and restarted by a reveal; after
  `quota.CORRESPONDENCE_IDLE` the one who did not seal loses, and neither sealing is a draw;
- **the page**: the strip, the slot, the ghost of one's own sealed move, the poll's change
  test and the verdict's words, run in node over a hand-made snapshot.

Nothing here works out a legal move or a winner: every move is one the engine offered the
reader (`snapshot_for`'s `moves`), and every outcome is the engine's or the store's.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as
every site test does.
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

from chess_variant_sandbox.site import admin, mail, move_email, quota
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library

PACKAGE = Path(__file__).resolve().parents[1] / "chess_variant_sandbox"
SITE = "https://site.example"


def _rounds(max_plies: int) -> dict[str, Any]:
    spec = json.loads((Path(__file__).parent / "_simultaneous_gardner.json").read_text("utf-8"))
    spec["rules"]["max_plies"] = max_plies
    return spec


ROUNDS = _rounds(40)
ROUNDS_HASH = spec_hash(ROUNDS)
# One round and it is over: `max_plies = 2` counts a round as two plies (§2.6).
ONE = _rounds(2)
ONE_HASH = spec_hash(ONE)


class Sent:
    """A sender that keeps the message instead of posting it."""

    def __init__(self) -> None:
        self.messages: list[mail.Message] = []

    def __call__(self, message: mail.Message) -> None:
        self.messages.append(message)


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(ROUNDS_HASH, ROUNDS)
        scratch.game(ONE_HASH, ONE)
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


def _pair(store: Scratch, game: str = ROUNDS_HASH) -> tuple[m.Row, int, int]:
    ada, bo = store.user("ada"), store.user("bo")
    found = c.start(store.connection, game, [ada, bo])
    store.connection.commit()
    return found, ada, bo


def _read(store: Scratch, match_id: str, user: int | None) -> dict[str, Any]:
    """The game as this person may see it: the one way a route draws it."""
    return c.snapshot_for(store.connection, m.row(store.connection, match_id), user)


def _offered(store: Scratch, match_id: str, user: int, index: int = 0) -> dict[str, Any]:
    """One of the moves the engine offers this person in the round now open."""
    return _read(store, match_id, user)["moves"][index]


def _seal(store: Scratch, match_id: str, user: int, index: int = 0) -> dict[str, Any]:
    chosen = int(_offered(store, match_id, user, index)["move"])
    return c.move(store.connection, match_id, user, chosen)


def _moved(store: Scratch, match_id: str) -> Any:
    with store.connection.cursor() as cursor:
        cursor.execute("select moved from matches where id = %s", (match_id,))
        return cursor.fetchone()[0]


def _played(store: Scratch, match_id: str) -> list[int]:
    return list(m.session(store.connection, match_id).played)


# ------------------------------------------------------------------- seal and reveal


class TestSealAndReveal:
    def test_a_first_seal_waits_and_changes_nothing_else(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        row = m.row(store.connection, found.id)
        assert c.owing(store.connection, row) == (0, 1), "both owe a move as a round opens"
        clock = _moved(store, found.id)
        chosen = _offered(store, found.id, ada)
        answer = _seal(store, found.id, ada)
        row = m.row(store.connection, found.id)
        assert row.plies == 0 and _played(store, found.id) == []
        assert _moved(store, found.id) == clock, "a seal does not move the round's clock"
        assert c.times_of(store.connection, found.id) == []
        assert c.owing(store.connection, row) == (1,)
        # The sealer is given their own move back, and offered nothing more.
        assert answer["round"]["sealed"] == [True, False]
        assert answer["round"]["yours"]["move"] == chosen["move"]
        assert answer["round"]["yours"]["text"] == chosen["text"]
        assert answer["moves"] == [] and answer["round"]["choosing"] is None
        # The other is told a move is in, and offered their own.
        theirs = _read(store, found.id, bo)
        assert theirs["round"]["sealed"] == [True, False]
        assert "yours" not in theirs["round"] and theirs["round"]["choosing"] == 1
        assert theirs["moves"] and theirs["sideToMove"] == 1

    def test_the_second_seal_reveals_both_in_seat_order(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        second = int(_offered(store, found.id, bo)["move"])
        first = int(_offered(store, found.id, ada)["move"])
        c.move(store.connection, found.id, bo, second)  # the second seat seals first
        answer = c.move(store.connection, found.id, ada, first)
        assert _played(store, found.id) == [first, second], "stored in seat order"
        assert c.sealed_seat(store.connection, found.id) is None
        history = answer["history"]
        assert [entry["player"] for entry in history] == [0, 1]
        assert history[1]["reveal"]["moves"], "the core's account of the round"
        assert len(c.times_of(store.connection, found.id)) == 2
        assert answer["round"]["number"] == 2 and answer["round"]["sealed"] == [False, False]
        assert c.owing(store.connection, m.row(store.connection, found.id)) == (0, 1)

    def test_a_sealed_move_is_final_and_an_illegal_one_is_refused(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        other = int(_offered(store, found.id, ada, 1)["move"])
        _seal(store, found.id, ada)
        with pytest.raises(m.MatchError, match="sealed already"):
            c.move(store.connection, found.id, ada, other)
        legal = {int(move["move"]) for move in _read(store, found.id, bo)["moves"]}
        illegal = next(n for n in range(10_000) if n not in legal)
        with pytest.raises(ValueError, match="not legal"):
            c.move(store.connection, found.id, bo, illegal)
        store.connection.rollback()
        cy = store.user("cy")
        with pytest.raises(m.MatchError, match="only the two players"):
            c.move(store.connection, found.id, cy, 0)

    def test_a_game_in_rounds_ends_by_its_rules(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, ONE_HASH)
        _seal(store, found.id, ada)
        ended = _seal(store, found.id, bo)
        assert ended["outcome"] is not None
        row = m.row(store.connection, found.id)
        assert not row.running and row.result is not None and not row.left
        assert c.sealed_seat(store.connection, found.id) is None
        with pytest.raises(m.MatchError, match="over"):
            c.move(store.connection, found.id, ada, 0)

    def test_a_draw_offer_lapses_at_the_reveal_and_not_at_a_seal(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        c.offer_draw(store.connection, found.id, ada)
        _seal(store, found.id, bo)
        assert c.offer_of(store.connection, found.id) is not None, "a seal declines nothing"
        _seal(store, found.id, ada)
        assert c.offer_of(store.connection, found.id) is None

    def test_either_may_cancel_until_the_first_round_is_revealed(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        _seal(store, found.id, ada)
        c.cancel(store.connection, found.id, bo)
        assert m.row(store.connection, found.id).reason == m.CANCELLED
        assert c.sealed_seat(store.connection, found.id) is None
        later = c.start(store.connection, ROUNDS_HASH, [ada, bo])
        _seal(store, later.id, ada)
        _seal(store, later.id, bo)
        with pytest.raises(m.MatchError, match="can no longer be cancelled"):
            c.cancel(store.connection, later.id, ada)

    def test_the_rules_under_the_moves_are_the_rounds(self, store: Scratch) -> None:
        found, ada, _bo = _pair(store)
        said = _read(store, found.id, ada)["correspondence"]["rules"]
        assert said == list(c.ROUND_RULES)
        assert quota.CORRESPONDENCE_IDLE_SAID in c.ROUND_RULES[-1]
        for line in c.ROUND_RULES:
            assert hits(line) == [], (line, hits(line))


# ---------------------------------------------------------------------- atomicity


class TestRacingSeals:
    """Two connections, as two requests are: the first holds its transaction open while the
    second arrives, which is the race at its narrowest."""

    def _race(self, store: Scratch, match_id: str, first: tuple[int, int],
              second: tuple[int, int]) -> list[BaseException]:
        one, two = store.open(), store.open()
        errors: list[BaseException] = []
        try:
            c.move(one, match_id, *first)  # sealed, not committed: the row is locked

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
            assert thread.is_alive(), "the second seal waits on the first's lock"
            one.commit()
            thread.join(timeout=30.0)
            assert not thread.is_alive()
        finally:
            one.close()
            two.close()
        return errors

    def test_two_racing_seals_reveal_the_round_once(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        mine = int(_offered(store, found.id, ada)["move"])
        theirs = int(_offered(store, found.id, bo)["move"])
        store.connection.commit()
        errors = self._race(store, found.id, (ada, mine), (bo, theirs))
        assert errors == []
        assert _played(store, found.id) == [mine, theirs]
        assert c.sealed_seat(store.connection, found.id) is None
        assert len(c.times_of(store.connection, found.id)) == 2

    def test_one_seat_racing_itself_seals_once(self, store: Scratch) -> None:
        found, ada, _bo = _pair(store)
        mine = int(_offered(store, found.id, ada, 0)["move"])
        other = int(_offered(store, found.id, ada, 1)["move"])
        store.connection.commit()
        errors = self._race(store, found.id, (ada, mine), (ada, other))
        assert len(errors) == 1 and "sealed already" in str(errors[0])
        assert _played(store, found.id) == []
        assert _read(store, found.id, ada)["round"]["yours"]["move"] == mine


# --------------------------------------------------------------------- nothing leaks


def _alike(answer: dict[str, Any]) -> str:
    """An answer with what differs between two games anyway taken out: their ids and the
    stamp of when each started."""
    kept = json.loads(json.dumps(answer))
    kept.pop("id", None)
    between = kept.get("correspondence") or {}
    between.pop("started", None)
    return json.dumps(kept, sort_keys=True)


class TestNothingLeaks:
    def _two(self, store: Scratch) -> tuple[m.Row, m.Row, int, int, list[str]]:
        """Two games of the same rules where ada sealed two different moves."""
        ada, bo = store.user("ada"), store.user("bo")
        games = [c.start(store.connection, ROUNDS_HASH, [ada, bo]) for _ in range(2)]
        texts = []
        for index, found in enumerate(games):
            chosen = _offered(store, found.id, ada, index)
            texts.append(str(chosen["text"]))
            c.move(store.connection, found.id, ada, int(chosen["move"]))
        store.connection.commit()
        assert texts[0] != texts[1]
        return games[0], games[1], ada, bo, texts

    def test_the_other_seats_reads_and_answers_are_alike_whatever_was_sealed(
        self, store: Scratch, app: SiteApp
    ) -> None:
        one, two, _ada, bo, texts = self._two(store)
        for path, body in (("", None), ("/draw", {}), ("", None)):
            answers = []
            for found in (one, two):
                route = f"/api/session/{found.id}{path}"
                answer = _as(app, bo).get(route) if body is None else _as(app, bo).post(route, body)
                answers.append(answer)
                said = json.dumps(answer)
                assert '"yours"' not in said and not any(text in said for text in texts)
            assert _alike(answers[0]) == _alike(answers[1]), path
            assert answers[0]["round"]["sealed"] == [True, False]

    def test_no_list_a_person_reads_carries_it(self, store: Scratch, app: SiteApp) -> None:
        one, two, ada, bo, texts = self._two(store)
        for route in ("/api/profile", "/api/challenges", "/api/me"):
            said = json.dumps(_as(app, bo).get(route))
            assert not any(text in said for text in texts), route
        theirs = _as(app, bo).get("/api/profile")["between"]["running"]
        mine = _as(app, ada).get("/api/profile")["between"]["running"]
        assert all(entry["yourMove"] for entry in theirs), "bo owes a seal in both"
        assert not any(entry["yourMove"] for entry in mine), "ada's are in"
        waiting = _as(app, bo).get("/api/challenges")["waiting"]
        assert sorted(entry["id"] for entry in waiting) == sorted([one.id, two.id])
        assert _as(app, ada).get("/api/challenges")["waiting"] == []

    def test_a_round_never_revealed_stays_unread_after_any_finish(self, store: Scratch) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        cy = store.user("cy")
        endings = {
            "resign": lambda found: c.resign(store.connection, found.id, bo),
            "agreed": lambda found: (c.offer_draw(store.connection, found.id, bo),
                                     c.accept_draw(store.connection, found.id, ada)),
            "cancel": lambda found: c.cancel(store.connection, found.id, bo),
            "leave": lambda found: admin.forget(store.connection, "bo"),
        }
        for name, end in endings.items():
            found = c.start(store.connection, ROUNDS_HASH, [ada, bo])
            _seal(store, found.id, ada)
            end(found)
            row = m.row(store.connection, found.id)
            assert not row.running, name
            assert c.sealed_seat(store.connection, found.id) is None, name
            for reader in (ada, cy):
                seen = _read(store, found.id, reader)
                assert "yours" not in seen["round"] and seen["round"]["sealed"] == [False, False]

    def test_only_the_correspondence_module_names_the_column(self) -> None:
        named = sorted(
            str(path.relative_to(PACKAGE))
            for path in PACKAGE.rglob("*")
            if path.suffix in (".py", ".js") and "sealed_move" in path.read_text("utf-8")
        )
        assert named == [str(Path("site") / "correspondence.py")]


# ---------------------------------------------------------------------------- the mails


def _ticked(store: Scratch) -> tuple[m.Row, int, int]:
    found, ada, bo = _pair(store)
    for user in (ada, bo):
        move_email.set_on(store.connection, user, True)
    store.connection.commit()
    return found, ada, bo


def _rows(store: Scratch) -> list[tuple[Any, ...]]:
    with store.connection.cursor() as cursor:
        cursor.execute("select recipient, ply from move_emails order by id")
        found = cursor.fetchall()
    store.connection.commit()
    return found


def _post_seal(app: SiteApp, store: Scratch, match_id: str, user: int) -> dict[str, Any]:
    chosen = int(_offered(store, match_id, user)["move"])
    store.connection.commit()
    return _as(app, user).post(f"/api/session/{match_id}/move", {"move": chosen})


class TestTheMails:
    def test_a_seal_says_a_move_is_in_and_the_reveal_says_the_round(
        self, store: Scratch, app: SiteApp, sent: Sent
    ) -> None:
        found, ada, bo = _ticked(store)
        mine = str(_offered(store, found.id, ada)["text"])
        _post_seal(app, store, found.id, ada)
        assert [message.to for message in sent.messages] == ["bo@example.invalid"]
        first = sent.messages[0]
        assert first.body.startswith("ada's move is in, sealed, in your game of")
        assert mine not in first.body and mine not in first.subject
        theirs = str(_offered(store, found.id, bo)["text"])
        answer = _post_seal(app, store, found.id, bo)
        assert [message.to for message in sent.messages][1:] == ["ada@example.invalid"]
        reveal = sent.messages[1].body
        assert reveal.startswith("Round 1 of your game of")
        assert f"you played {mine} and bo played {theirs}." in reveal
        for line in answer["history"][1]["reveal"]["words"]:
            assert line in reveal
        for text in (first.body, reveal):
            assert hits(text) == [], hits(text)

    def test_still_one_email_a_turn(self, store: Scratch, app: SiteApp, sent: Sent) -> None:
        found, ada, bo = _ticked(store)
        _post_seal(app, store, found.id, ada)  # bo: ada's is in (ply 0)
        _post_seal(app, store, found.id, bo)   # ada: round 1 revealed (ply 2)
        _post_seal(app, store, found.id, bo)   # ada was told at ply 2 already: nothing
        _post_seal(app, store, found.id, ada)  # bo: round 2 revealed (ply 4)
        assert _rows(store) == [(bo, 0), (ada, 2), (bo, 4)]
        assert [message.to for message in sent.messages] == [
            "bo@example.invalid", "ada@example.invalid", "bo@example.invalid"]
        assert sent.messages[2].body.startswith("Round 2 of your game of")

    def test_the_message_builder_says_a_round_plainly(self) -> None:
        said = mail.your_move(
            to="ada@example.invalid", game="Rounds", opponent="bo", played=None,
            link=f"{SITE}/#/play/x", off=f"{SITE}/mail/off/t",
            revealed=mail.RoundSaid(number=3, yours="a2-a3", theirs="e4-e3",
                                    words=("The two pawns met on c3.",)),
        )
        assert said.subject == "Your move: Rounds against bo"
        assert said.body.startswith(
            "Round 3 of your game of Rounds against bo is revealed: you played a2-a3 and bo"
            " played e4-e3. The two pawns met on c3. It's your move for the next round:"
        )


# ---------------------------------------------------------------------------- the clock


class TestTheClock:
    def test_neither_sealing_is_a_draw_after_five_days(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        store.backdate(found.id, timedelta(days=4))
        assert m.sweep(store.connection) == ()
        store.backdate(found.id, timedelta(days=2))
        assert [row.id for row in m.sweep(store.connection)] == [found.id]
        row = m.row(store.connection, found.id)
        assert row.result == 0.5 and row.reason == c.UNSEALED and row.left
        assert _read(store, found.id, ada)["outcome"] == {"winner": None, "reason": "unsealed"}

    def test_the_one_who_did_not_seal_loses(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        _seal(store, found.id, ada)
        store.connection.commit()
        store.backdate(found.id, timedelta(days=6))
        swept = m.sweep(store.connection)
        assert [row.id for row in swept] == [found.id]
        row = m.row(store.connection, found.id)
        assert row.left and row.reason == "resigned"
        assert row.score_for(row.seat_of(r.person(bo))) == 0.0, "bo owed, and lost"
        assert c.sealed_seat(store.connection, found.id) is None

    def test_a_reveal_restarts_the_clock(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        _seal(store, found.id, ada)
        store.connection.commit()
        store.backdate(found.id, timedelta(days=4))
        _seal(store, found.id, bo)  # the round revealed: a new one opens now
        store.connection.commit()
        store.backdate(found.id, timedelta(days=2))
        assert m.sweep(store.connection) == ()

    def test_the_time_left_names_whoever_owes(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        assert _read(store, found.id, ada)["correspondence"]["toMove"].startswith("You have")
        assert _read(store, found.id, bo)["correspondence"]["toMove"].startswith("You have")
        _seal(store, found.id, ada)
        assert _read(store, found.id, ada)["correspondence"]["toMove"].startswith("bo has")
        assert _read(store, found.id, bo)["correspondence"]["toMove"].startswith("You have")


# ------------------------------------------------------------------- the page, in node


_ROUND = """
SESSION = { id: 'g1', ruleset: { sides: ['white', 'black'], wrap: null },
  seats: ['human', 'human'], sideToMove: 0, inCheck: false, remaining: 1, outcome: null,
  board: new Array(25).fill(0),
  history: [{ player: 0, text: 'b2-b3' }, { player: 1, text: 'd4-d3' }],
  moves: [{ move: 7, text: 'c2-c3', from: 7, to: 12 }],
  round: { number: 2, sealed: [false, true], choosing: 0, tie: 'both', initiative: -1,
    voidRun: null, bars: [] },
  correspondence: { you: 0, players: ['ada', 'bo'], drawOffer: null, mayCancel: false,
    times: [new Date(Date.now() - 3 * 3600e3).toISOString(),
            new Date(Date.now() - 3 * 3600e3).toISOString()],
    toMove: 'You have 4 days to move.', rules: [] } };
"""

# Your move sealed, waiting for bo's.
_SEALED = """
SESSION.sideToMove = 1; SESSION.moves = [];
SESSION.round.sealed = [true, false]; SESSION.round.choosing = null;
SESSION.round.yours = { move: 7, text: 'c2-c3', from: 7, to: 12 };
SESSION.correspondence.toMove = 'bo has 4 days to move.';
"""

_SLOT = """
const SLOT = { hidden: false, innerHTML: '', held: false,
  classList: { toggle: (name, on) => { if (name === 'held') SLOT.held = !!on; } } };
document.getElementById = id => (id === 'corrSend' ? SLOT : null);
const KEPT = {};
window.localStorage = { getItem: k => (k in KEPT ? KEPT[k] : null),
  setItem: (k, v) => { KEPT[k] = String(v); }, removeItem: k => { delete KEPT[k]; } };
drawBoard = () => {};
"""


class TestThePage:
    def test_the_strip_says_the_round_and_never_the_other_move(self) -> None:
        from test_web import _in_page

        said = _in_page(_ROUND, "corrStatus()")
        assert said.startswith('<span class="muted">Round 2.</span> <b>Your move.</b>')
        assert "bo has sealed a move, unseen." in said
        assert "Round 1 was revealed 3 hours ago." in said
        assert "You have 4 days to move." in said
        sealed = " ".join(_in_page(_ROUND + _SEALED, "corrStatus()").split())
        assert "<b>Your move is sealed:</b> c2-c3. bo has not sealed theirs yet." in sealed
        assert "bo has 4 days to move." in sealed

    def test_the_slot_seals_a_held_move_and_then_waits(self) -> None:
        from test_web import _in_page

        held = _in_page(_ROUND + _SLOT + "CORR_HELD = 7; drawCorrSend();", "SLOT")
        assert held["held"] and 'id="sealHeld"' in held["innerHTML"]
        assert 'id="sendHeld"' not in held["innerHTML"]
        waiting = _in_page(_ROUND + _SEALED + _SLOT + "drawCorrSend();", "SLOT")
        said = " ".join(waiting["innerHTML"].split())
        assert "<b>Your move is sealed.</b> It is revealed with bo's, once theirs is in." in said
        assert 'id="sealHeld"' not in said and not waiting["held"]
        kept = _in_page(_ROUND + _SLOT + "roundHold(7);", "keptHeld('g1')")
        assert kept == {"ply": 2, "move": 7, "text": "c2-c3"}
        reader = _in_page(_ROUND + _SLOT + "SESSION.correspondence.you = null; drawCorrSend();",
                          "SLOT")
        assert reader["hidden"] and reader["innerHTML"] == ""

    def test_your_sealed_move_is_drawn_held_and_nobody_elses(self) -> None:
        from test_web import _in_page

        drawn = _in_page(_ROUND + _SEALED, "corrHeldDrawn()")
        assert drawn["from"] == 7 and drawn["to"] == 12 and drawn["text"] == "c2-c3"
        assert _in_page(_ROUND, "corrHeldDrawn()") is None

    def test_the_poll_sees_a_seal_and_the_verdict_has_words(self) -> None:
        from test_web import _in_page

        changed = _in_page(
            _ROUND, "corrChanged(SESSION, Object.assign({}, SESSION, { round: Object.assign("
            "{}, SESSION.round, { sealed: [true, true] }) }))")
        assert changed is True
        assert _in_page(_ROUND, "corrChanged(SESSION, SESSION)") is False
        assert _in_page(_ROUND, "verdictText({ winner: null, reason: 'unsealed' })") == (
            "Drawn as neither player sealed a move in time."
        )
