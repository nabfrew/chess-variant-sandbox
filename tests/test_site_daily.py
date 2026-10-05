"""Q721: `/admin`'s *By day* table, read from the tables the site already keeps.

Each test writes rows with known moments into the session's migrated store
(`tests/site_store.py`) and reads them back a day at a time, so what is proved is the
queries against the real schema, not a fake's idea of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import page_text
import pytest
import site_store
import site_words

from chess_variant_sandbox.site import admin, daily, db, load, matches, pages, usage

UTC = timezone.utc
# Noon UTC, so "today" and "yesterday" are plain to read in every row below.
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
TODAY = NOW.date()
YESTERDAY = TODAY - timedelta(days=1)


@pytest.fixture
def store() -> Iterator[Any]:
    url, schema = site_store.emptied("no database to read the days from")
    connection = db.connect(url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
        connection.commit()
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _person(cursor: Any, handle: str, email: str, created: datetime) -> int:
    cursor.execute(
        "insert into users (handle, email, created) values (%s, %s, %s) returning id",
        (handle, email, created),
    )
    return int(cursor.fetchone()[0])


def _match(
    cursor: Any, name: str, started: datetime, seat0: int | None = None,
    seat1: int | None = None, source: str | None = None,
) -> None:
    cursor.execute(
        "insert into matches (id, game, mode, seat0_user, seat0_agent, seat1_user,"
        " seat1_agent, seed, started, source) values (%s, 'g', %s, %s, %s, %s, %s, 1, %s, %s)",
        (name, "sandbox" if seat0 is None and seat1 is None else "leaderboard",
         seat0, None if seat0 is not None or seat1 is None else "random",
         seat1, None if seat1 is not None or seat0 is None else "random",
         started, source),
    )


def _by_day(store: Any) -> dict[Any, daily.Day]:
    found = daily.read(store, now=NOW.timestamp())
    return {row.day: row for row in found.rows}


def test_every_day_of_the_fortnight_is_a_row_newest_first(store: Any) -> None:
    found = daily.read(store, now=NOW.timestamp())
    assert [row.day for row in found.rows] == [
        TODAY - timedelta(days=n) for n in range(daily.DAYS)
    ]
    assert all(row == daily.Day(row.day) for row in found.rows), "nothing kept, all zero"
    assert found.said() == ["No server errors in these days."]


def test_requests_visitors_errors_and_refused_mail_come_from_the_hourly_counts(
    store: Any,
) -> None:
    morning = NOW.replace(hour=9)
    with store.cursor() as cursor:
        for hour, route, status, kind, who, count in (
            (morning, "GET /api/library", 200, "", usage.VISITOR, 30),
            (morning, "GET /api/library", 200, "", usage.MEMBER, 5),
            (morning, "POST /api/admin", 400, daily.MAIL_REFUSED, usage.MEMBER, 1),
            (morning - timedelta(days=1), "POST /api/x", 500, "RuntimeError",
             usage.VISITOR, 2),
            # Before the fortnight: never read.
            (morning - timedelta(days=20), "GET /api/library", 200, "", usage.VISITOR, 99),
        ):
            cursor.execute(
                "insert into usage_hours values (%s, %s, '', %s, %s, %s, %s, 0, 0, 0)",
                (hour, route, status, kind, who, count),
            )
    store.commit()
    days = _by_day(store)
    assert (days[TODAY].requests, days[TODAY].by_visitors, days[TODAY].errors,
            days[TODAY].mail_refused) == (36, 30, 0, 1)
    assert (days[YESTERDAY].requests, days[YESTERDAY].errors) == (2, 2)
    said = daily.read(store, now=NOW.timestamp()).said()
    assert "POST /api/x RuntimeError x2" in said[0]


def test_games_players_joins_invitations_and_move_emails(store: Any) -> None:
    with store.cursor() as cursor:
        cursor.execute(
            "insert into games (hash, spec, name, origin) values ('g', '{}', 'g', 'test')"
        )
        ann = _person(cursor, "ann", "ann@example.org", NOW - timedelta(days=1))
        bob = _person(cursor, "bob", "bob@example.org", NOW - timedelta(hours=2))
        # The maintenance account and the loader's owner were not joins.
        assert admin.RESERVED_EMAIL.endswith(daily.NO_PERSON)
        assert load.MAINTAINER_EMAIL.endswith(daily.NO_PERSON)
        _person(cursor, "secret-admin", admin.RESERVED_EMAIL, NOW - timedelta(hours=1))
        _person(cursor, "maintainer", load.MAINTAINER_EMAIL, NOW - timedelta(hours=1))
        _match(cursor, "p1", NOW - timedelta(hours=3))
        _match(cursor, "p2", NOW - timedelta(hours=4))
        _match(cursor, "c1", NOW - timedelta(hours=1), seat0=ann)
        _match(cursor, "c2", NOW - timedelta(hours=1), seat1=ann)
        _match(cursor, "h1", NOW - timedelta(hours=1), seat0=ann, seat1=bob)
        _match(cursor, "y1", NOW - timedelta(days=1), seat0=ann)
        # A game played in the local application and loaded: not the site's.
        _match(cursor, "l1", NOW - timedelta(hours=1), seat0=bob, source=matches.LOCAL_APP)
        cursor.execute(
            "insert into invites (code, issuer, uses_left, email, created)"
            " values ('a', null, 0, 'ann@example.org', %s),"
            " ('b', %s, 1, 'cy@example.org', %s)",
            (NOW - timedelta(days=1, hours=1), ann, NOW - timedelta(hours=5)),
        )
        cursor.execute(
            "insert into move_emails (match, recipient, ply, sent, failed) values"
            " ('h1', %s, 1, %s, null), ('h1', %s, 2, %s, 'refused (422)')",
            (bob, NOW - timedelta(minutes=50), ann, NOW - timedelta(minutes=40)),
        )
    store.commit()
    days = _by_day(store)
    today, yesterday = days[TODAY], days[YESTERDAY]
    assert (today.practice, today.with_members, today.players) == (2, 3, 2)
    assert (yesterday.practice, yesterday.with_members, yesterday.players) == (0, 1, 1)
    assert (today.joined, yesterday.joined) == (1, 1)
    assert (today.invited, yesterday.invited) == (1, 1)
    assert (today.move_emails, today.move_emails_failed) == (2, 1)


def test_the_payload_names_nobody(store: Any) -> None:
    with store.cursor() as cursor:
        _person(cursor, "qzxv", "qzxv@example.org", NOW - timedelta(hours=1))
    store.commit()
    payload = daily.read(store, now=NOW.timestamp()).as_dict()
    text = repr(payload)
    assert "qzxv" not in text and "example.org" not in text
    assert set(payload) == {"rows", "said", "cannot"}
    assert all(isinstance(value, int) for row in payload["rows"]
               for key, value in row.items() if key != "day")


def test_the_admin_page_draws_the_table_from_text_nodes() -> None:
    page = page_text.inlined(pages.admin("boss"))
    assert '<tbody id="days"></tbody>' in page and 'id="days-said"' in page
    assert "usage(payload.usage, payload.days)" in page
    assert "innerHTML" not in page


def test_every_new_sentence_passes_the_words_guard() -> None:
    said = [*daily.CANNOT, *daily.Days(()).said(),
            *daily.Days((), (("POST /api/x", "RuntimeError", 2),)).said()]
    page = pages.admin("boss")
    start = page.index("<h3>By day, the last fortnight</h3>")
    said.append(page[start: page.index("<h3>The last day, by route</h3>")])
    trouble = [(text, site_words.hits(text)) for text in said if site_words.hits(text)]
    assert not trouble, trouble
