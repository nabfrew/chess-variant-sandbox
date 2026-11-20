"""The growth reading counts an invitation request as pending, granted or declined (GR1).

`docs/reference/growth.py` once counted requests as granted or not, so a declined request
read as one still awaiting the owner's answer (the 2026-11-20 audit). Here a request in each
of the three states goes through both halves of the report: the collector's store query, run
against the test database exactly as the box runs it, and the Markdown drawn from its rows.

The query half needs `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skips without it;
`tests/test_site.py` has the recipe for a scratch cluster. Every address here is invented.
"""

from __future__ import annotations

import json
import re
from typing import Any

import private_paths
import site_store

from chess_variant_sandbox.site.db import connect

GROWTH = "docs/reference/growth.py"
growth = private_paths.script(GROWTH, "growth_report")
pytestmark = private_paths.needs(GROWTH)


def _store_query() -> str:
    """The collector's one `select`, as the box runs it inside its read-only transaction."""
    found = re.search(r'SQL = """(.*?)"""', growth.REMOTE, re.S)
    assert found, "the collector's store query was not found"
    sql = found.group(1).strip()
    assert sql.startswith("begin transaction read only;") and sql.endswith("rollback;")
    return sql.removeprefix("begin transaction read only;").removesuffix("rollback;").strip()


def _data(store: dict[str, Any]) -> dict[str, Any]:
    return {"span": [0, 0], "weeks": {}, "crawler_names": {}, "game_names": {}, "store": store}


def _store_table(text: str) -> tuple[list[str], dict[str, list[str]]]:
    lines = text.split("**The store**", 1)[1].splitlines()
    rows = [line for line in lines if line.startswith("| ")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    body = {}
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        body[cells[0]] = cells
    return header, body


def _column(header: list[str], body: dict[str, list[str]], week: str, name: str) -> str:
    return body[week][header.index(name)]


STORE = {
    "first_hour": "2026-11-16T00:00:00+00:00",
    "last_hour": "2026-11-20T00:00:00+00:00",
    "usage": [],
    "signups": [],
    "invites": [],
}


class TestTheReport:
    """The Markdown, from rows shaped as the collector prints them."""

    def test_each_state_has_its_own_column_and_count(self) -> None:
        asks = [{"week": "2026-11-16", "n": 3, "pending": 1, "granted": 1, "declined": 1}]
        text = growth.report(_data({**STORE, "asks": asks}), None)
        header, body = _store_table(text)
        assert "of them granted" not in header
        for name, value in (
            ("invitation requests", "3"),
            ("pending", "1"),
            ("granted", "1"),
            ("declined", "1"),
        ):
            assert _column(header, body, "2026-11-16", name) == value
        assert "pending (nobody has answered) 1, granted 1, declined 1" in text
        assert "30 days past its answer" in text

    def test_a_declined_request_is_not_counted_as_pending(self) -> None:
        asks = [{"week": "2026-10-19", "n": 1, "pending": 0, "granted": 0, "declined": 1}]
        text = growth.report(_data({**STORE, "asks": asks}), None)
        header, body = _store_table(text)
        assert _column(header, body, "2026-10-19", "pending") == "0"
        assert _column(header, body, "2026-10-19", "declined") == "1"
        assert "pending (nobody has answered) 0," in text

    def test_a_saved_reading_from_before_cannot_tell_pending_from_declined(self) -> None:
        """An old `--json` has only `granted`: its ungranted request prints `?`, never 0 or 1."""
        asks = [{"week": "2026-10-19", "n": 1, "granted": 0}]
        text = growth.report(_data({**STORE, "asks": asks}), None)
        header, body = _store_table(text)
        assert _column(header, body, "2026-10-19", "pending") == "?"
        assert _column(header, body, "2026-10-19", "declined") == "?"
        assert "pending (nobody has answered) ?, granted 0, declined ?" in text


class TestTheStoreQuery:
    """The collector's query, against the test database, with a request in each state."""

    def test_it_counts_pending_granted_and_declined_apart(self) -> None:
        url, name = site_store.emptied("no database to read invitation requests from")
        connection = connect(url)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"set search_path to {name}")
                cursor.execute(
                    "insert into invite_asks (email, created, answered, granted) values"
                    " ('waiting@example.invalid', '2026-11-17 10:00+00', null, null),"
                    " ('welcome@example.invalid', '2026-11-17 11:00+00', now(), true),"
                    " ('refused@example.invalid', '2026-11-18 12:00+00', now(), false),"
                    " ('earlier@example.invalid', '2026-10-20 09:00+00', now(), false)"
                )
                cursor.execute(_store_query())
                (raw,) = cursor.fetchone()
            connection.rollback()
        finally:
            connection.close()
        store = raw if isinstance(raw, dict) else json.loads(raw)
        asks = {r["week"]: r for r in store["asks"]}
        assert asks["2026-11-16"] == {
            "week": "2026-11-16",
            "n": 3,
            "pending": 1,
            "granted": 1,
            "declined": 1,
        }
        assert asks["2026-10-19"] == {
            "week": "2026-10-19",
            "n": 1,
            "pending": 0,
            "granted": 0,
            "declined": 1,
        }
        text = growth.report(_data(store), None)
        assert "pending (nobody has answered) 1, granted 1, declined 2" in text
