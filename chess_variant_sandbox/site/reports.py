"""Report a game: a signed-in person says a game should not be on the site, in one row.

Q503d, `docs/plan-game-pipeline.md` §3.2 and §5. The game page carries a quiet link under
the rules, *Report this game (spam, offensive, a copy)*; pressing it and choosing one of
the three files a row here, and the maintainer reads every row on `/admin` with the game's
kill switch beside it.

**A report does nothing by itself.** It hides nothing, mails nobody and moves no count.
Killing a game stays an admin's act (`admin.kill_game`), so a report is something for a
person to look at, never a vote that takes a game down: a switch any account could pull
is a switch a grudge could pull.

Two limits hold with nobody watching. One report a person a game is the table's primary
key, and `quota.REPORTS_A_DAY` caps how many games one person may report in a rolling day.

**The note is text**, as a suggestion is: stored as typed and shown through
`textContent`, so there is no escaping step to forget.

Nothing here commits. The caller owns the transaction, the rule `votes` and
`suggestions` follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import identity, quota

SPAM = "spam"
OFFENSIVE = "offensive"
COPY = "copy"
REASONS: tuple[str, ...] = (SPAM, OFFENSIVE, COPY)
"""The three reasons, as the column's check constraint spells them. They are the plan's
own list (§3.2), the plainest there is; *something else* is what the note is for."""

LABELS: dict[str, str] = {
    SPAM: "spam",
    OFFENSIVE: "offensive",
    COPY: "a copy of another game",
}
"""What each reason is called wherever one is printed: the page's choices, the reporter's
own line and the maintainer's table read the same words."""


class ReportError(ValueError):
    """A report cannot be filed as asked. The server answers 400 with the sentence."""


@dataclass(frozen=True)
class Report:
    """One person's report of one game.

    `handle`, `name`, `killed` and `gone` are filled by `every`, which joins the users and
    games tables for the maintainer's page; `dismissed` by `dismissed` (Q518).
    """

    reporter: int
    game: str
    reason: str
    note: str
    created: datetime | None = None
    handle: str = ""
    name: str = ""
    killed: bool = False
    gone: bool = False
    dismissed: datetime | None = None

    @property
    def label(self) -> str:
        return LABELS.get(self.reason, self.reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.game,
            "reporter": self.reporter,
            "name": self.name,
            "reason": self.reason,
            "label": self.label,
            "note": self.note,
            "handle": self.handle,
            "created": None if self.created is None else self.created.isoformat(),
            # The kill switch's state, which is what the button beside the row acts on,
            # read exactly as `admin.Listed` reads it for the games table.
            "killed": self.killed,
            "gone": self.gone,
            "dismissed": None if self.dismissed is None else self.dismissed.isoformat(),
        }


def reasons() -> list[dict[str, str]]:
    """The choices, in order, for the game page to offer."""
    return [{"reason": reason, "label": LABELS[reason]} for reason in REASONS]


def file(connection: Any, reporter: int, game: str, reason: str, note: str = "") -> Report:
    """File one report, or refuse it in a sentence the reporter can read.

    Already reported comes first, so a second press is told that rather than being counted
    against the day; then the day's allowance; then the reason and the note. The game is
    the caller's to have checked: the site refuses a killed or deleted one with the 404 an
    unknown hash gets before this runs.
    """
    if of(connection, reporter, game) is not None:
        raise ReportError(
            "You have already reported this game. The maintainer reads every report."
        )
    quota.check_new_report(connection, reporter)
    if reason not in REASONS:
        named = ", ".join(f"{one} ({LABELS[one]})" for one in REASONS)
        raise ReportError(f"A report's reason is one of {named}, not {reason!r}.")
    said = str(note).strip()
    if len(said) > quota.REPORT_NOTE_LENGTH:
        raise ReportError(
            f"A report's note is up to {quota.REPORT_NOTE_LENGTH} characters and this one "
            f"is {len(said)}."
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into reports (reporter, game, reason, note) values (%s, %s, %s, %s)"
            " on conflict (reporter, game) do nothing"
            " returning reporter, game, reason, note, created",
            (reporter, game, reason, said or None),
        )
        found = cursor.fetchone()
    if found is None:
        # Two presses that raced past the check above: the key kept one of them.
        raise ReportError(
            "You have already reported this game. The maintainer reads every report."
        )
    return _to_report(found)


def of(connection: Any, reporter: int, game: str) -> Report | None:
    """This person's report of this game, or None."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select reporter, game, reason, note, created from reports"
            " where reporter = %s and game = %s",
            (reporter, game),
        )
        found = cursor.fetchone()
    return None if found is None else _to_report(found)


def dismiss(connection: Any, reporter: int, game: str) -> str:
    """Take one report off the maintainer's list and keep its row.

    Q503o. The row is the reason: `of` still finds it, so the same person still cannot
    report the same game, and the day's count still has it. Dismissing twice is harmless.
    Returns the handle the report came from, for the answer's sentence; an unknown
    (reporter, game) is a `KeyError` and so a 404. Nothing commits.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update reports set dismissed = coalesce(dismissed, now())"
            " where reporter = %s and game = %s"
            " returning coalesce((select handle from users where id = reporter), %s)",
            (reporter, game, identity.FORMER),
        )
        found = cursor.fetchone()
    if found is None:
        raise KeyError(f"no such report: {reporter} on {game}")
    return str(found[0])


def every(connection: Any) -> list[Report]:
    """Every open report, newest first, for the maintainer's page.

    A dismissed report (Q503o) is not listed here; its row stays, and `dismissed` lists it.

    A left join on users for the handle, so somebody who has left reads as
    `identity.FORMER`, the rule every other reader follows. The game's name and its two
    switches come with it, so the page can link the game and draw its kill button.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select r.reporter, r.game, r.reason, r.note, r.created,"
            "       coalesce(u.handle, %s), g.name,"
            "       g.revoked is not null, g.deleted is not null"
            "  from reports r"
            "  join games g on g.hash = r.game"
            "  left join users u on u.id = r.reporter"
            " where r.dismissed is null"
            " order by r.created desc, r.game, r.reporter",
            (identity.FORMER,),
        )
        return [_to_report(row) for row in cursor.fetchall()]


DISMISSED_SHOWN = 100
"""How many dismissed reports `/admin` lists, most recently dismissed first. The rows are
never deleted, so the view is capped rather than the table."""


def dismissed(connection: Any, limit: int = DISMISSED_SHOWN) -> tuple[list[Report], int]:
    """The dismissed reports, most recently dismissed first, and how many there are (Q518).

    The same joins as `every`, so a dismissed report of a game later taken off still says
    so, and a reporter who left still reads as `identity.FORMER`. The count is of every
    dismissed row, so the page can say when it shows only the newest `limit`.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select r.reporter, r.game, r.reason, r.note, r.created,"
            "       coalesce(u.handle, %s), g.name,"
            "       g.revoked is not null, g.deleted is not null, r.dismissed"
            "  from reports r"
            "  join games g on g.hash = r.game"
            "  left join users u on u.id = r.reporter"
            " where r.dismissed is not null"
            " order by r.dismissed desc, r.game, r.reporter"
            " limit %s",
            (identity.FORMER, limit),
        )
        rows = [_to_report(row) for row in cursor.fetchall()]
        cursor.execute("select count(*) from reports where dismissed is not null")
        found = cursor.fetchone()
    return rows, int(found[0]) if found is not None else 0


def _to_report(found: Sequence[Any]) -> Report:
    return Report(
        reporter=int(found[0]),
        game=str(found[1]),
        reason=str(found[2]),
        note=str(found[3] or ""),
        created=found[4],
        handle=str(found[5]) if len(found) > 5 else "",
        name=str(found[6]) if len(found) > 6 else "",
        killed=bool(found[7]) if len(found) > 7 else False,
        gone=bool(found[8]) if len(found) > 8 else False,
        dismissed=found[9] if len(found) > 9 else None,
    )
