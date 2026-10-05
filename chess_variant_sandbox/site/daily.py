"""The site by UTC day: who came, what they played, who joined, what mail went (Q721).

`/admin`'s *By day* table. Every number is read from a table the site already keeps for
another reason -- `usage_hours`, `matches`, `users`, `invites`, `move_emails` -- so the
reading adds no column and no row, and names nobody: what reaches the page is counts.
The privacy notice already lists every source (the hourly counts, the matches, the
address an invitation went to, when each move email was sent).

A visitor is never a person here, as in `usage`: the hourly counts hold no address and no
cookie, so the honest visitor numbers are their requests and the practice games they
started. How many different visitors came is the request log's question, read on the box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any

from chess_variant_sandbox.site import matches, usage

DAYS = 14
"""How many UTC days the table shows, today's included."""

UTC = timezone.utc

NO_PERSON = ".invalid"
"""The top-level domain of the addresses the site gives its own rows."""

MAIL_REFUSED = "MailError"
"""The kind a request is counted under when the mail service would not take a message."""

CANNOT = (
    "Today is the day so far. Every other row is a whole UTC day.",
    "Requests are the site's own API calls. A page load is not one, so the landing page "
    "seen and left counts nowhere here; the request log on the box has it.",
    "A visitor is counted as requests and practice games, never as a person: nothing "
    "the site keeps can tell two visitors apart, by design.",
    "Players are the different signed-in people seated in a game started that day.",
    "Mail refused counts invitations the mail service would not take; the invitation "
    "was not made. Move emails failed are written on the email's own row.",
    "The local application's games, loaded from this machine's runs, are left out.",
)


@dataclass(frozen=True)
class Day:
    """One UTC day of the site."""

    day: date
    requests: int = 0
    by_visitors: int = 0
    errors: int = 0
    mail_refused: int = 0
    practice: int = 0
    with_members: int = 0
    players: int = 0
    joined: int = 0
    invited: int = 0
    move_emails: int = 0
    move_emails_failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "requests": self.requests,
            "byVisitors": self.by_visitors,
            "errors": self.errors,
            "mailRefused": self.mail_refused,
            "practice": self.practice,
            "withMembers": self.with_members,
            "players": self.players,
            "joined": self.joined,
            "invited": self.invited,
            "moveEmails": self.move_emails,
            "moveEmailsFailed": self.move_emails_failed,
        }


@dataclass(frozen=True)
class Days:
    """The days, newest first, and the server errors behind the *errors* column."""

    rows: tuple[Day, ...]
    errors: tuple[tuple[str, str, int], ...] = field(default=())

    def said(self) -> list[str]:
        if not self.errors:
            return ["No server errors in these days."]
        return ["Server errors in these days, by route and kind: " + "; ".join(
            f"{route} {kind or 'unknown'} x{count}" for route, kind, count in self.errors
        ) + ". The journal has each one: journalctl -u chess-variant-sandbox-serve."]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "said": self.said(),
            "cannot": list(CANNOT),
        }


def window(now: float | None = None, days: int = DAYS) -> tuple[datetime, datetime]:
    """From midnight UTC `days - 1` days ago to now."""
    at = datetime.fromtimestamp(time.time() if now is None else now, UTC)
    midnight = at.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=days - 1), at


def _day(moment: Any) -> date:
    """A row's day: a `date` the database already cut, or a moment cut here in UTC."""
    if isinstance(moment, datetime):
        return (moment.astimezone(UTC) if moment.tzinfo else moment).date()
    return moment


def read(connection: Any, now: float | None = None, days: int = DAYS) -> Days:
    """The last `days` UTC days, newest first, every day present even when empty."""
    since, until = window(now, days)
    found: dict[date, Day] = {
        (since + timedelta(days=n)).date(): Day((since + timedelta(days=n)).date())
        for n in range(days)
    }

    def add(day: Any, **counts: int) -> None:
        key = _day(day)
        if key not in found:
            return
        row = found[key]
        found[key] = replace(row, **{
            name: getattr(row, name) + int(value or 0) for name, value in counts.items()
        })

    errors: dict[tuple[str, str], int] = {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select hour, route, who, status, kind, sum(count) from usage_hours"
            " where hour >= %s and hour < %s group by hour, route, who, status, kind",
            (since, until),
        )
        for hour, route, who, status, kind, count in cursor.fetchall():
            add(
                hour,
                requests=count,
                by_visitors=count if who == usage.VISITOR else 0,
                errors=count if int(status) >= 500 else 0,
                mail_refused=count if kind == MAIL_REFUSED else 0,
            )
            if int(status) >= 500:
                errors[(route, kind)] = errors.get((route, kind), 0) + int(count)

        # A match a visitor starts has nobody in either seat (`0003_visitor_sandbox`).
        cursor.execute(
            "select (started at time zone 'UTC')::date,"
            " count(*) filter (where seat0_user is null and seat1_user is null),"
            " count(*) filter (where seat0_user is not null or seat1_user is not null)"
            " from matches where started >= %s and started < %s"
            " and source is distinct from %s group by 1",
            (since, until, matches.LOCAL_APP),
        )
        for day, practice, with_members in cursor.fetchall():
            add(day, practice=practice, with_members=with_members)

        cursor.execute(
            "select seated.day, count(distinct seated.who) from matches,"
            " lateral (select (started at time zone 'UTC')::date as day,"
            " unnest(array[seat0_user, seat1_user]) as who) as seated"
            " where started >= %s and started < %s and source is distinct from %s"
            " and seated.who is not null group by seated.day",
            (since, until, matches.LOCAL_APP),
        )
        for day, players in cursor.fetchall():
            add(day, players=players)

        # The maintenance account and the loader's owner are rows nobody joined as, and
        # both have an address under `.invalid`, which no person's can be (RFC 2606).
        cursor.execute(
            "select (created at time zone 'UTC')::date, count(*) from users"
            " where created >= %s and created < %s"
            " and coalesce(email, '') not like %s group by 1",
            (since, until, "%" + NO_PERSON),
        )
        for day, joined in cursor.fetchall():
            add(day, joined=joined)

        cursor.execute(
            "select (created at time zone 'UTC')::date, count(*) from invites"
            " where created >= %s and created < %s group by 1",
            (since, until),
        )
        for day, invited in cursor.fetchall():
            add(day, invited=invited)

        cursor.execute(
            "select (sent at time zone 'UTC')::date, count(*), count(failed)"
            " from move_emails where sent >= %s and sent < %s group by 1",
            (since, until),
        )
        for day, sent, failed in cursor.fetchall():
            add(day, move_emails=sent, move_emails_failed=failed)

    return Days(
        rows=tuple(sorted(found.values(), key=lambda row: row.day, reverse=True)),
        errors=tuple(sorted(
            ((route, kind, count) for (route, kind), count in errors.items()),
            key=lambda one: (-one[2], one[0], one[1]),
        )),
    )

