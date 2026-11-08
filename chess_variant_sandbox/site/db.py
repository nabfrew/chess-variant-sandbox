"""The one place the site learns where its database is.

`CHESS_VARIANT_SANDBOX_DATABASE_URL` is read here and nowhere else in the package, and a test holds
that down. A second reader is a second answer, and a migration applied to one database
beside a query run against another is a failure worth closing off now, while there is
still nothing to migrate.

The box's value is a Unix-socket URL. Postgres there listens on no TCP port at all --
`deploy/postgres.sh` leaves it with a socket and peer authentication -- so the socket is
what a URL can reach rather than a preference about how to reach it.

No schema and no queries live here. This module opens a connection and says where to;
what goes in the database is task 1.2's.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
from typing import Any

ENV_VAR = "CHESS_VARIANT_SANDBOX_DATABASE_URL"

# What the box's unit sets. Named here so a refusal can show it, and never used as a
# fallback: a site that quietly connects to whatever is on the local socket is a site
# that can be pointed at the wrong database by a missing line in a unit file.
SOCKET_URL = "postgresql:///chess_variant_sandbox?host=/var/run/postgresql"


class ConfigError(ValueError):
    """The site cannot start as configured.

    A ``ValueError`` on purpose, like `web.server.PortInUse`: `cli.main` prints those as
    one line and exits 2, and a missing environment variable is something the person
    deploying did, not a stack trace.
    """


def database_url() -> str:
    """Where the database is, or a refusal naming the variable and the box's value."""
    url = os.environ.get(ENV_VAR, "").strip()
    if not url:
        raise ConfigError(
            f"{ENV_VAR} is not set, so the site does not know which database to use "
            f"(the box's value is {SOCKET_URL})"
        )
    return url


def connect(url: str | None = None) -> Any:
    """Open one connection, importing the driver only now.

    The import is here rather than at the top of the module because `psycopg` is under
    the `site` extra: importing `chess_variant_sandbox.site` must work in a checkout that never
    installed a driver, or the CLI cannot print its own help there.

    The session's time zone is UTC, whatever the server's `timezone` setting or the
    client's `PGTZ` says (Q744). Every stamp the site reads back then carries `+00:00`,
    and a page that prints a stamp's first ten characters prints UTC's day, the calendar
    the site files everything else on (`daily`, `whats_new`, `stages.day`). Before, the
    profile said a draft kept at midnight UTC was kept the day before wherever Postgres
    ran west of UTC. A `SET` rather than a startup option because libpq sends `PGTZ` as a
    startup parameter, which the server applies after `options` and so would win.
    """
    try:
        import psycopg
    except ModuleNotFoundError as error:
        raise ConfigError(
            "the site needs a database driver, which a plain `uv sync` does not install: "
            "run `uv sync --extra site`"
        ) from error
    connection = psycopg.connect(url or database_url())
    # Committed, so the setting is the session's and not one transaction's: a rollback
    # later would otherwise undo it.
    connection.execute("set time zone 'UTC'")
    connection.commit()
    return connection


def redacted(url: str) -> str:
    """The URL with any password blanked, for printing.

    The site prints where it is connecting on start and that line goes to the journal,
    which people read over shoulders and paste into issues. A socket URL has nothing to
    hide and comes back unchanged.
    """
    shown = re.sub(r"(?i)(password=)[^&\s]*", r"\1***", url)
    return re.sub(r"://([^/@]*):([^/@]*)@", r"://\1:***@", shown)


# How many idle connections `Pool` keeps; more than the box's two cores ever use at once.
POOL_KEPT = 4
# A kept connection idle longer than this is asked `select 1` before it is lent again, so a
# Postgres restarted under a running site costs nobody a failed request.
POOL_PING_SECONDS = 10.0


class Pool:
    """Connections kept between units of work (C15c, `docs/plan-performance.md`).

    `site/server.py` opened a connection for every unit of work, about 30 ms each, five of
    them on a game's page. A unit of work still begins and ends a transaction of its own;
    only the opened socket outlives it. A connection is kept only when it is a driver's
    connection (one with ``info.transaction_status``), idle in no transaction after its
    commit, and from the connect callable it is lent for now: one handed back after a
    rollback or an error is closed, as before, since a rollback can undo a session setting
    the connect callable made (a test's ``search_path``). Every lock the site takes is a
    transaction's (`pg_advisory_xact_lock`), so nothing a unit of work holds survives it.
    """

    def __init__(self, kept: int = POOL_KEPT) -> None:
        self._kept = kept
        self._idle: list[tuple[Any, float]] = []
        self._source: Any = None
        self._lock = threading.Lock()

    def take(self, connect: Any) -> Any:
        """A kept connection from ``connect``, alive, or a new one from it."""
        stale: list[Any] = []
        found = None
        with self._lock:
            if connect != self._source:
                stale = [held for held, _ in self._idle]
                self._idle = []
                self._source = connect
            elif self._idle:
                found = self._idle.pop()
        for one in stale:
            close_quietly(one)
        if found is not None:
            held, since = found
            if _usable(held, time.monotonic() - since):
                return held
            close_quietly(held)
        return connect()

    def give(self, connection: Any, connect: Any) -> None:
        """A connection whose unit of work committed: kept if it may be, else closed."""
        if _poolable(connection):
            with self._lock:
                if (connect == self._source and len(self._idle) < self._kept
                        and all(held is not connection for held, _ in self._idle)):
                    self._idle.append((connection, time.monotonic()))
                    return
        close_quietly(connection)

    def clear(self) -> None:
        """Close every kept connection."""
        with self._lock:
            idle, self._idle = self._idle, []
        for held, _ in idle:
            close_quietly(held)


def _poolable(connection: Any) -> bool:
    info = getattr(connection, "info", None)
    status = getattr(info, "transaction_status", None)
    # 0 is libpq's PQTRANS_IDLE: connected, and in no transaction.
    return (status is not None and int(status) == 0
            and not getattr(connection, "closed", True)
            and not getattr(connection, "broken", False))


def _usable(connection: Any, idle: float) -> bool:
    if not _poolable(connection):
        return False
    if idle < POOL_PING_SECONDS:
        return True
    try:
        connection.execute("select 1")
        connection.rollback()
    except Exception:  # noqa: BLE001 - a dead connection is replaced, whatever killed it
        return False
    return True


def close_quietly(connection: Any) -> None:
    with contextlib.suppress(Exception):
        connection.close()
