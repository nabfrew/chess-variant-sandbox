"""One migrated store for the whole session, emptied before each test that asks for it.

The site's database tests used to build a schema from nothing for every test: create it,
run every migration, drop it at the end. Q377 measured that at about 0.5 s of the 0.68 s
each of those 913 tests took. The migrations seed no rows and create no functions,
triggers or views, so a store with every row but the ledger's deleted and every identity
restarted is the store a fresh migration makes. A rollback per test would not be: the
site opens a connection per unit of work, and several tests commit through a second
connection.

**`DELETE`, not `TRUNCATE`.** `TRUNCATE ... RESTART IDENTITY CASCADE` is the obvious
statement and costs 224 ms here, because it gives every table and index a new file and
Windows is slow at files; a `DELETE` per table, children before the tables they
reference, plus a restart of each identity's sequence, costs 2 ms (measured on the scratch
cluster, 15 tables, a row or two in each). The order is read off the catalog when the
schema is made, so a migration that adds a table adds it to the order.

`emptied(why)` is what a test's helper calls instead of creating a schema. It makes the
session's schema the first time -- a name of its own, so a server with other things on it
is left alone -- and empties it every time after. `drop()` takes it away at the end of the
session, from `conftest.py`. Under pytest-xdist every worker is a process of its own, so
each has its own schema.

A test about migrating -- an empty schema, a ledger it writes, a table it creates -- still
takes a schema of its own: `test_site.py`'s `_scratch_schema` and `test_site_links.py`'s
`_scratch_store`.
"""

from __future__ import annotations

import os
import secrets
from typing import Any

import pytest

URL_ENV = "CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL"

# Long enough for any test's own statements; short enough that a connection a test left
# holding a lock fails the next test by name instead of hanging the run.
LOCK_TIMEOUT = "20s"

_url: str | None = None
_schema: str | None = None
# The statements that empty the schema, worked out once when it is made.
_emptying: tuple[str, ...] = ()
# The connection that makes and empties it, kept for the session.
_connection: Any = None


def database(why: str) -> str:
    """The test database's address, or a skip that says what the test needed it for."""
    url = os.environ.get(URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{URL_ENV} is unset: {why}")
    pytest.importorskip("psycopg", reason="the site extra is not installed here")
    return url


def emptied(why: str) -> tuple[str, str]:
    """`(url, schema)`: the session's migrated schema, with no rows in it but the ledger.

    The emptying goes through one connection kept for the session: a connection costs
    about 70 ms here (a backend process per connection on Windows), which was the other
    half of what a test paid for its store. It holds no lock between tests; every use
    ends in a commit.
    """
    global _url, _schema, _emptying, _connection
    url = database(why)
    from chess_variant_sandbox.site import db, migrate

    if _connection is None or _connection.closed or _url != url:
        _connection = db.connect(url)
        with _connection.cursor() as cursor:
            cursor.execute(f"set lock_timeout = '{LOCK_TIMEOUT}'")
            if _schema is not None and _url == url:
                cursor.execute(f"set search_path to {_schema}")
        _connection.commit()
    try:
        with _connection.cursor() as cursor:
            if _schema is None or _url != url:
                name = f"session_test_{secrets.token_hex(6)}"
                cursor.execute(f"create schema {name}")
                cursor.execute(f"set search_path to {name}")
                _connection.commit()
                migrate.run(_connection)
                _url, _schema, _emptying = url, name, _statements(_connection, name)
            else:
                for statement in _emptying:
                    cursor.execute(statement)
        _connection.commit()
    except Exception:
        _connection.close()
        raise
    return url, _schema


def _statements(connection: Any, schema: str) -> tuple[str, ...]:
    """A `DELETE` per table, each before the tables it references, then each identity's
    restart. A cycle between tables would have no such order, and gets the `TRUNCATE`."""
    from chess_variant_sandbox.site import migrate

    with connection.cursor() as cursor:
        cursor.execute(
            "select tablename from pg_tables where schemaname = %s and tablename <> %s",
            (schema, migrate.LEDGER),
        )
        tables = {row[0] for row in cursor.fetchall()}
        cursor.execute(
            "select c.conrelid::regclass::text, c.confrelid::regclass::text"
            " from pg_constraint c join pg_namespace n on n.oid = c.connamespace"
            " where c.contype = 'f' and n.nspname = %s",
            (schema,),
        )
        # child -> the tables it references; a table referencing itself is emptied by
        # one statement and is no ordering constraint.
        references = {table: set() for table in tables}
        for child, parent in cursor.fetchall():
            if child != parent:
                references[child].add(parent)
        cursor.execute(
            "select c.relname from pg_class c join pg_namespace n on n.oid = c.relnamespace"
            " where c.relkind = 'S' and n.nspname = %s order by c.relname",
            (schema,),
        )
        sequences = [row[0] for row in cursor.fetchall()]

    order: list[str] = []
    waiting = dict(references)
    while waiting:
        # Deletable now: no table still waiting references it.
        free = sorted(
            table for table in waiting
            if not any(table in parents for parents in waiting.values())
        )
        if not free:
            listed = ", ".join(sorted(tables))
            return (f"truncate table {listed} restart identity cascade",)
        for table in free:
            order.append(table)
            del waiting[table]
    return (
        *(f"delete from {table}" for table in order),
        *(f"alter sequence {sequence} restart" for sequence in sequences),
    )


def drop() -> None:
    """Take the session's schema away, if a test ever made one."""
    global _url, _schema, _emptying, _connection
    if _connection is not None and not _connection.closed:
        _connection.close()
    _connection = None
    if _schema is None or _url is None:
        return
    from chess_variant_sandbox.site import db

    connection = db.connect(_url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"drop schema if exists {_schema} cascade")
        connection.commit()
    finally:
        connection.close()
    _url, _schema, _emptying = None, None, ()


STAND_IN_READING = {"state": "spot-checked", "stand-in": "a test's"}
"""What `on_trial` hands `drafts.publish` in place of an attached reading.

A real reading takes a browser's measurement, tier one's walk and tier two's replay
(`test_site_publish.py` makes one end to end). The tests about what an *On trial* game is --
listed, linked, everybody's end of a gap -- need only the publish, and get it through the
same `publish` the route calls, with this in the reading's place.
"""


def on_trial(store: Any, digest: str) -> None:
    """Put a saved draft on trial through `drafts.publish`, as its owner, and commit."""
    from chess_variant_sandbox.site import drafts, terms

    with store.cursor() as cursor:
        cursor.execute("select owner from games where hash = %s", (digest,))
        owner = int(cursor.fetchone()[0])
    drafts.publish(store, digest, owner, STAND_IN_READING, terms.VERSION)
    store.commit()
