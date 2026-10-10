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

import os
import re
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
