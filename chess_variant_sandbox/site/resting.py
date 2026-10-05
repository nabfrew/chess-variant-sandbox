"""Resting: a trial game nobody plays leaves the listing, and may come back once.

Q503i of `docs/plan-game-pipeline.md` (§2.1 *On trial* "or it goes to rest", §2.2's last
row). A game on trial **rests** when it has been on trial `quota.RESTING_AFTER` (90 days,
from `games.published`) and nobody but its designer has finished a game on it in the last
`quota.RESTING_QUIET` (60). `rest` stamps `games.rested`, and `stages.stage` then reads
**Resting**: off the *On trial* tab and out of the listing (`drafts.listed`), with its page
and its link working as before and its place under `quota.PUBLISHED_AT_ONCE` freed.

**Finished** is the shortlist's reading (`shortlist.finishers`): a game ended on the board,
in either mode, a kept finish or a leaderboard row's own verdict. A resignation is not one,
and neither is a watch, which has nobody in it. Here it must also have ended inside the
quiet window, so the finish's own time is read (`finishes.finished`, `matches.finished`).

**What never rests.** A shortlisted game: nominating it was its designer's consent to the
round, the players' signal already bought it a place, and resting it would take a game off
the shortlist by a clock rather than by the round (§2.2 gives the round that job). A draft,
which is not listed. A Library or Baseline game, which the maintainer placed. A killed,
retired or deleted game, which is off the listing already.

**The way back** (§2.2: *"the designer may publish it again once"*). Its designer presses
*Publish* in the designer again. It goes through every gate a first publish does -- the
check, the same-game refusal, the caps and a current reading (`drafts.publish`) -- and then
clears `rested`, stamps `woken` and restarts `published`, so it has a fresh 90 days and a
fresh count toward the shortlist's 30. A game that rests again after that stays resting:
`quota.WAKES_A_GAME` is one.

**Who runs the sweep.** `chess-variant-sandbox site load`, which every deploy runs, calls
`rest` after the library is written (`load.load`). It is one `update` over the trial games,
and a second run the same day stamps nothing. Nothing here commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import quota
from chess_variant_sandbox.site.matches import AGREED
from chess_variant_sandbox.web.human import RESIGNED

DESIGNED = "designed"
PUBLIC = "public"

_QUIET_FINISH = (
    "select 1 from matches m where m.game = g.hash"
    " and (m.seat0_user is distinct from g.owner and m.seat0_user is not null"
    "      or m.seat1_user is distinct from g.owner and m.seat1_user is not null)"
    " and (exists (select 1 from finishes f where f.match = m.id and f.reason <> %(resigned)s"
    f"              and f.reason <> '{AGREED}'"
    "              and f.finished > now() - %(quiet)s)"
    "      or (m.result is not null and coalesce(m.reason, '') <> %(resigned)s"
    f"          and coalesce(m.reason, '') <> '{AGREED}'"
    "          and m.finished > now() - %(quiet)s))"
)
"""A match on the game with somebody other than its designer in it, finished on the board
inside the quiet window. A watch has nobody in either seat and is never one."""


@dataclass(frozen=True)
class Rested:
    """A game the sweep took off the listing."""

    hash: str
    name: str

    @property
    def said(self) -> str:
        return (
            f"rested {self.name!r} ({self.hash}): {quota.RESTING_AFTER.days} days on trial "
            f"and nobody else finished a game on it in {quota.RESTING_QUIET.days}, so it is "
            f"off the listing; its page and link stay"
        )


def rest(connection: Any) -> list[Rested]:
    """Stamp `rested` on every trial game nobody else has played lately (see the module).

    Returns the games it stamped, by name, so the load's output says which ones went. A
    game already resting keeps its first stamp.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update games g set rested = now()"
            " where g.origin = %(designed)s and g.visibility = %(public)s"
            " and g.nominated is null and g.rested is null and g.revoked is null"
            " and g.retired is null and g.deleted is null and g.published is not null"
            " and g.published <= now() - %(after)s"
            f" and not exists ({_QUIET_FINISH})"
            " returning g.hash, g.name",
            {
                "designed": DESIGNED,
                "public": PUBLIC,
                "after": quota.RESTING_AFTER,
                "quiet": quota.RESTING_QUIET,
                "resigned": RESIGNED,
            },
        )
        found = cursor.fetchall()
    return sorted((Rested(str(row[0]), str(row[1])) for row in found), key=lambda r: r.hash)


def wakes(woken: Any) -> bool:
    """Whether a resting game with this `woken` stamp may be published again."""
    return woken is None


def said(rested: datetime | None, woken: Any, mine: bool) -> str:
    """The Resting sentence with its date and whether it can come back, for its pages."""
    since = "" if rested is None else f" since {_day(rested)}"
    head = (
        f"Resting{since}: nobody but its designer finished a game on it in "
        f"{quota.RESTING_QUIET.days} days, so it is off the listing. Its page and link still "
        "work."
    )
    if not wakes(woken):
        return head + " It has come back from resting once already, so it stays here."
    if mine:
        return head + " You can publish it again, once, from the designer."
    return head + " Its designer can publish it again, once."


def _day(when: datetime) -> str:
    """A date as the page says it: *3 Oct 2026* (`shortlist.day`'s form)."""
    return f"{when.day} {when:%b %Y}"
