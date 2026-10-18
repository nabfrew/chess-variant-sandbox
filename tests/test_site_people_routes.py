"""Routes B and C of `docs/reference/human-data.md`, and the check they rest on (Q768, H4, H7-H9).

- **H4**: a computer seat's agent string does not pin its player -- the core's build, the
  game's tables and the values handed to the core are not in it -- so the row keeps them
  (`matches.pinned`), one entry while the player is the same and a second when a reply is
  chosen by another (`site/pinning.py`).
- **H7**: each computer player's place on people's scale per game (`site/people_scale.py`),
  on `/admin` always and in the game page's *more* only once five people are established.
- **H8**: the bar of §3.3, read over rated games against today's top group, with the
  counts behind it and H4's split, on `/admin` (`site/alarm.py`).
- **H9**: a game over the bar enters the training queue by itself at the finish that puts
  it over, once per top group, after every premise entry (`site/training.py`).

The game is `test_site_finishes.py`'s forced one: one grunt a side on one file, every move
forced, the person in seat 0 wins on ply 3, against `minimax:nodes=400`, the top of the
table it is seeded with. The pure readings are checked on rows built by hand.

The database tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from test_site_finishes import AGENT, FORCED, FORCED_HASH, LADDER, Scratch, _play_out, _scratch

from chess_variant_sandbox import measurement
from chess_variant_sandbox.agents.base import parse_agent
from chess_variant_sandbox.site import admin, alarm, people_scale, pinning, training
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import library, opponents

WHEN = datetime(2026, 10, 9, tzinfo=timezone.utc)


@pytest.fixture
def store() -> Iterator[Scratch]:
    """The forced game, its table seeded from the ladder (`minimax:nodes=400` on top)."""
    with _scratch() as scratch:
        with scratch.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (FORCED_HASH, json.dumps(FORCED), FORCED["name"], "corpus"),
            )
        r.seed(scratch.connection, FORCED_HASH, LADDER)
        scratch.connection.commit()
        yield scratch


def _rated_win(store: Scratch, user: int) -> str:
    """One rated game the person in seat 0 wins against the top."""
    match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], user)
    _play_out(store.connection, match.id)
    return match.id


def _row(
    ident: str,
    user: int | None,
    agent: str,
    result: float,
    *,
    top: bool = True,
    game: str = "g",
    mode: str = m.LEADERBOARD,
    minutes: int = 0,
) -> m.Row:
    return m.Row(
        id=ident, game=game, mode=mode,
        seat0=r.person(user) if user is not None else None,
        seat1=r.computer(agent), seed=0, plies=3, result=result, reason="", left=False,
        top=top, top_players=(agent,), before=(None, None), source=None,
        started=WHEN, moved=WHEN, finished=WHEN + timedelta(minutes=minutes),
    )


def _table(*computers: tuple[str, float]) -> dict[r.PlayerKey, r.Player]:
    return {
        r.computer(agent): r.Player(r.computer(agent), rating, 100.0, 0.0)
        for agent, rating in computers
    }


# --------------------------------------------------------- H4: who the computer seat is


class TestTheAgentStringDoesNotPinThePlayer:
    def test_no_rung_names_a_build_tables_or_values(self) -> None:
        """The finding: what the string carries is a kind, a budget and a stored objective."""
        for agent, _label in opponents.LADDER:
            if agent == opponents.HUMAN:
                continue
            config = parse_agent(agent)
            assert not {"build", "core", "tables", "values"} & set(config.options), agent
            assert measurement.built_fingerprint() not in agent

    def test_one_string_on_two_games_is_two_players(self) -> None:
        other = json.loads(json.dumps(FORCED))
        other["rules"]["max_plies"] = 6
        assert pinning.identity(FORCED, [AGENT])["tables"] != pinning.identity(
            other, [AGENT]
        )["tables"]

    def test_compared(self) -> None:
        now = {"build": "b", "tables": "t", "players": "p"}
        assert pinning.compared(None, now) == pinning.UNRECORDED
        assert pinning.compared([], now) == pinning.UNRECORDED
        assert pinning.compared([now], now) == pinning.SAME
        assert pinning.compared([now, {**now, "build": "c"}], now) == pinning.EARLIER


class TestTheRowPinsIt:
    def test_a_start_records_who_the_computer_seat_is(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        [entry] = store.one("select pinned from matches where id = %s", match.id)[0]
        assert entry == pinning.identity(FORCED, [AGENT])
        assert entry["build"] == measurement.built_fingerprint()
        _play_out(store.connection, match.id)
        assert len(store.one("select pinned from matches where id = %s", match.id)[0]) == 1

    def test_a_reply_under_another_build_is_appended(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "human"])
        before = pinning.identity(FORCED, [AGENT])
        later = {**before, "build": "a-later-core"}
        monkeypatch.setattr(pinning, "identity", lambda spec, agents: later)
        m.play_agent(store.connection, match.id)
        pinned = store.one("select pinned from matches where id = %s", match.id)[0]
        assert pinned == [before, later]
        assert pinning.compared(pinned, later) == pinning.EARLIER

    def test_a_watch_pins_both_seats_together(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "greedy"])
        [entry] = store.one("select pinned from matches where id = %s", match.id)[0]
        assert entry == pinning.identity(FORCED, [AGENT, "greedy"])


# ------------------------------------------------------------------------- H8: the bar


class TestTheBar:
    def test_the_bar_is_the_plans(self) -> None:
        assert (alarm.BAR_PEOPLE, alarm.BAR_SCORE, alarm.BAR_GAMES) == (3, 0.30, 10)

    def test_over_needs_all_three(self) -> None:
        table = _table(("top", 300.0), ("low", 0.0))
        # Three people win once each, seven losses: 10 games, 0.30.
        rows = [_row(f"w{u}", u, "top", 1.0, minutes=u) for u in (1, 2, 3)]
        rows += [_row(f"l{i}", 4, "top", 0.0, minutes=10 + i) for i in range(7)]
        found = alarm.reading_from("g", rows, table)
        assert (found.games, found.people, found.beaten_by) == (10, 4, 3)
        assert found.score == pytest.approx(0.30)
        assert found.over

        assert not alarm.reading_from("g", rows[:-1], table).over, "nine games"
        two = [_row("w9", 1, "top", 1.0)] + rows[1:2] + rows[3:] + [_row("x", 4, "top", 0.0)]
        assert alarm.reading_from("g", two, table).beaten_by == 2
        assert not alarm.reading_from("g", two, table).over, "two people"
        low = rows[:3] + [_row(f"m{i}", 4, "top", 0.0) for i in range(8)]
        assert not alarm.reading_from("g", low, table).over, "3 points in 11 is 0.27"
        drawn = rows[:3] + [_row(f"d{i}", 4, "top", 0.5) for i in range(7)]
        assert alarm.reading_from("g", drawn, table).score == pytest.approx(0.65)

    def test_only_games_against_todays_top_count(self) -> None:
        """H4's reason: a game against last month's strongest is not one against this one's."""
        rows = [_row(f"w{u}", u, "old", 1.0) for u in range(1, 11)]
        rows.append(_row("n", 20, "low", 1.0, top=False))
        then = alarm.reading_from("g", rows, _table(("old", 300.0), ("low", 0.0)))
        assert then.over
        now = alarm.reading_from(
            "g", rows, _table(("new", 400.0), ("old", 300.0), ("low", 0.0))
        )
        assert (now.games, now.over) == (0, False)

    def test_not_counted(self) -> None:
        table = _table(("top", 300.0))
        rows = [
            _row("v", None, "top", 1.0),  # a visitor, who cannot play a rated game
            _row("t", 1, "top", 1.0, top=False),  # not the top when it started
            _row("o", 1, "top", 1.0, game="other"),
        ]
        assert alarm.reading_from("g", rows, table).games == 0

    def test_the_split_by_player(self) -> None:
        table = _table(("top", 300.0))
        rows = [_row(f"w{u}", u, "top", 1.0) for u in (1, 2, 3)]
        now = {"build": "b", "tables": "t", "players": "p"}
        pinned = {"w1": [now], "w2": [{**now, "build": "old"}]}
        found = alarm.reading_from("g", rows, table, pinned, {"top": now})
        assert (found.same, found.earlier, found.unrecorded) == (1, 1, 1)
        assert "1 were played by today's player" in alarm.said(found)


# -------------------------------------------------------------------- H9: the queue


class TestTheQueue:
    def test_the_finish_that_puts_a_game_over_queues_it_once(self, store: Scratch) -> None:
        people = [store.user(name) for name in ("ada", "bo", "cy")]
        for index in range(9):
            _rated_win(store, people[index % 3])
        assert training.entries(store.connection) == [], "nine games are under the bar"

        _rated_win(store, people[0])
        [entry] = training.entries(store.connection)
        assert (entry.game, entry.cause, entry.state) == (FORCED_HASH, training.PEOPLE, "queued")
        assert entry.evidence is not None
        assert entry.evidence["top"] == [AGENT]
        assert (entry.evidence["games"], entry.evidence["beatenBy"]) == (10, 3)

        _rated_win(store, people[1])
        assert len(training.entries(store.connection)) == 1, "still one entry"

        # Trained and done: the same top beaten again asks for nothing more.
        with store.connection.cursor() as cursor:
            cursor.execute("update jobs set state = 'done' where id = %s", (entry.id,))
        _rated_win(store, people[2])
        assert training.entries(store.connection) == []

    def test_a_practice_game_queues_nothing(self, store: Scratch) -> None:
        for _ in range(12):
            match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
            _play_out(store.connection, match.id)
        assert training.entries(store.connection) == []

    def test_premise_entries_are_served_first(self, store: Scratch) -> None:
        with store.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values"
                " ('other', '{}'::jsonb, 'other', 'corpus')"
            )
        people = training.queue(
            store.connection, FORCED_HASH, training.PEOPLE, {"top": [AGENT]}
        )
        premise = training.queue(store.connection, "other", training.PREMISE)
        assert [one.id for one in training.entries(store.connection)] == [premise, people]
        claimed = training.claim(store.connection, "the 16-core machine", "build")
        assert claimed is not None and claimed.id == premise
        assert training.queue(store.connection, "other", training.PEOPLE, {"top": ["x"]}) is None
        assert training.claim(store.connection, "the 16-core machine", "build").id == people
        assert training.claim(store.connection, "the 16-core machine", "build") is None

    def test_a_people_entry_names_its_top(self, store: Scratch) -> None:
        with pytest.raises(training.TrainingError):
            training.queue(store.connection, FORCED_HASH, training.PEOPLE)
        with pytest.raises(training.TrainingError):
            training.queue(store.connection, FORCED_HASH, "a guess")


# ----------------------------------------------------------- H7: people's scale


def _establish(store: Scratch, count: int) -> list[int]:
    """`count` people with an established rating on the forced game, rated 1000 to 2000."""
    users = [store.user(f"p{index}") for index in range(count)]
    r.store(store.connection, FORCED_HASH, [
        r.Player(r.person(user), 1000.0 + 1000.0 * index / max(1, count - 1), 80.0, 0.06, 15)
        for index, user in enumerate(users)
    ])
    return users


class TestPeoplesScale:
    def test_a_place(self) -> None:
        table = _table(("top", 1800.0), ("low", 1200.0))
        for user, rating in ((1, 1500.0), (2, 1700.0), (3, 1900.0)):
            table[r.person(user)] = r.Player(r.person(user), rating, 90.0, 0.06)
        table[r.person(4)] = r.Player(r.person(4), 1000.0, 300.0, 0.06)
        rows = [_row("a", 1, "top", 1.0), _row("b", 2, "top", 0.0)]
        seeds = {r.computer("top"): r.Player(r.computer("top"), 1750.0, 100.0, 0.0)}
        top, low = people_scale.places_from("g", rows, table, seeds)
        assert (top.agent, top.games, top.below, top.established, top.seed) == (
            "top", 2, 2, 3, 1750.0
        )
        assert (low.games, low.below, low.seed) == (0, 0, None)

    def test_the_page_says_nothing_below_five_established(self, store: Scratch) -> None:
        _establish(store, 4)
        _rated_win(store, store.user("ada"))
        shown = people_scale.shown(people_scale.places(store.connection, FORCED_HASH))
        assert shown["said"] == [] and shown["established"] == 4

    def test_from_five_the_page_places_each_played_computer(self, store: Scratch) -> None:
        _establish(store, 5)
        _rated_win(store, store.user("ada"))
        store.connection.commit()
        app = SiteApp(library.RULESET_DIR, library.RUNS_DIR)
        app.connect = store.open
        standing = app._standing(FORCED_HASH)["standing"]
        said = standing["peopleScale"]["said"]
        assert said[0] == people_scale.SCALE_SAID
        assert len(said) == 2, "the one computer player people have played"
        assert "people with an established rating" in said[1]
        assert AGENT not in json.dumps(said), "named, never by its agent string"

    def test_the_admin_page_has_both_readings(self, store: Scratch) -> None:
        people = [store.user(name) for name in ("ada", "bo", "cy")]
        for index in range(10):
            _rated_win(store, people[index % 3])
        store.connection.commit()
        app = SiteApp(library.RULESET_DIR, library.RUNS_DIR)
        app.connect = store.open
        app._request.role = admin.ADMIN
        payload = app.get("/api/admin")
        [scale] = payload["peopleScale"]
        assert scale["hash"] == FORCED_HASH and not scale["shown"]
        assert {place["agent"] for place in scale["places"]} == {AGENT, "greedy", "random"}
        [game] = payload["bar"]["games"]
        assert game["over"] and game["queued"]["cause"] == training.PEOPLE
        assert "10 were played by today's player" in game["said"]
        assert [(one["hash"], one["name"]) for one in payload["bar"]["queue"]] == [
            (FORCED_HASH, FORCED["name"])
        ]


def test_the_admin_markup_and_script_draw_them() -> None:
    from chess_variant_sandbox.site import pages

    page = pages.admin("ada")
    for anchor in ('id="bar"', 'id="training-queue"', 'id="people-scale"'):
        assert anchor in page
    script = (measurement.ROOT / "chess_variant_sandbox/site/static/admin.js").read_text(
        encoding="utf-8"
    )
    for drawn in ("function bar(", "function peopleScale(", "bar(payload.bar)",
                  "peopleScale(payload.peopleScale"):
        assert drawn in script
    assert "innerHTML" not in script


def test_the_game_page_draws_the_scale_in_the_opponent_lines_more() -> None:
    import page_text

    assert "standing.peopleScale" in page_text.PAGE

