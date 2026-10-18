"""The shortlist's bar, the votes that count toward it, and *Nominate*.

Q503f of `docs/plan-game-pipeline.md` (§2.2's *On trial → Shortlisted* row; `docs/archive/plan-game-pipeline.md` §3.2's *Where
it stands*, §3.3's vote sentence). A game on trial becomes **Shortlisted** when its
designer presses *Nominate* while it is **eligible**, and pressing it is the designer's
consent to the round that may follow (§2.2 *Why the designer nominates*). The level itself
is still `stages.stage`'s, read off `games.nominated`, which `nominate` stamps and nothing
else on the site writes.

## The bar (the user's answer Q2, B)

Every number is `quota`'s, so raising the bar is one edit there. A game is eligible when
all five hold:

1. **`quota.SHORTLIST_DAYS` on trial**, counted from `games.published`.
2. **`quota.SHORTLIST_FINISHERS` people other than the designer finished a game on it**
   (`finishers`), sandbox games included (plan-meta-game's D, decision B).
3. **`quota.SHORTLIST_VOTES` qualifying votes** (`qualifying`): the voter finished a game
   on it, is not the designer, and has an account `quota.VOTER_ACCOUNT_AGE` old that has
   not been shut.
4. **Those votes average `quota.SHORTLIST_STARS` or more.**
5. **Its reading is current**: the spot-checked reading from its designer's computer that
   let it be published, on today's basis. The server hands it in (`drafts.attached`, the
   one function Q503e fills in), as it does to `drafts.publish`; until then no game
   carries one, and no game can be on trial either.

A game the round voted down or vetoed (Q503h, `site/rounds.py`) has a sixth line, **the
wait**: it may be nominated again only `quota.DECLINED_WAIT` (60 days) after that
decision. A game never declined has five.

**What finishing a game means here.** A game that ended on the board, in either mode:
`finishes` (Q502a) keeps every finish with a person in it, and a match row with a verdict
covers the leaderboard games from before that table. A resignation is not a finish for the
bar: resigning at the first move is not having played it, and `badges` reads *finished* the
same way for N1 *Crowd pleaser*. A watch (a computer player in both seats) has nobody in it.

Every count is shown on the game's page, plainly, with no prompt (meta-game test 6).

## The two caps

One shortlisted game a designer at a time (`quota.SHORTLISTED_A_DESIGNER`), and at most
`quota.SHORTLIST_SIZE` on the shortlist. Both are counted under one advisory lock held to
the caller's commit, so two presses one short of a cap cannot both pass. A full shortlist
refuses the press; a place opens when the round takes a game off it.

Nothing here commits. The caller owns the transaction, the rule `votes` and `drafts`
follow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from chess_variant_sandbox.site import quota, stages
from chess_variant_sandbox.site.matches import AGREED
from chess_variant_sandbox.web.human import RESIGNED

DESIGNED = "designed"

_LOCK = "shortlist"
"""The advisory lock's name. One lock for both caps: the shortlist's size is everybody's."""


class NominateError(ValueError):
    """A press of *Nominate* refused, in the sentence its designer reads.

    A `ValueError`, so the server answers 400 with the sentence, as `PublishError` does.
    """


@dataclass(frozen=True)
class Check:
    """One condition of the bar: whether it holds, and the plain sentence that says so."""

    key: str
    met: bool
    says: str

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "met": self.met, "says": self.says}


@dataclass(frozen=True)
class Bar:
    """Where one trial game stands against the shortlist's bar, counted from rows."""

    game: str
    owner: int | None
    since: datetime | None
    days: int
    finishers: int
    votes: int
    mean: float | None
    reading: bool
    again: datetime | None = None
    """When a game the round declined may be nominated again (Q503h), or None."""
    waiting: bool = False
    """Whether that day is still to come."""

    @property
    def opens(self) -> datetime | None:
        """The day the time on trial is up, or None for a row never published."""
        if self.since is None:
            return None
        return self.since + timedelta(days=quota.SHORTLIST_DAYS)

    @property
    def checks(self) -> list[Check]:
        """The lines as anybody but its designer reads them."""
        return self.lines(mine=False)

    def lines(self, *, mine: bool) -> list[Check]:
        """The conditions, each met or not with its sentence, worded for who reads them:
        its designer (`mine`) is not told that only its designer can refresh the reading
        (Q537)."""
        return [
            Check("days", self.days >= quota.SHORTLIST_DAYS, self._days_said()),
            Check(
                "finishers",
                self.finishers >= quota.SHORTLIST_FINISHERS,
                f"{self.finishers} of {quota.SHORTLIST_FINISHERS} other players have "
                "finished a game on it",
            ),
            Check(
                "votes",
                self.votes >= quota.SHORTLIST_VOTES,
                f"{self.votes} of {quota.SHORTLIST_VOTES} votes count so far",
            ),
            Check(
                "stars",
                self.mean is not None and self.mean >= quota.SHORTLIST_STARS,
                (
                    f"they average {self.mean:.1f} stars, of {quota.SHORTLIST_STARS:.1f} "
                    "needed"
                    if self.mean is not None
                    else f"no vote counts yet, and they must average "
                    f"{quota.SHORTLIST_STARS:.1f} stars"
                ),
            ),
            Check(
                "reading",
                self.reading,
                self._reading_said(mine),
            ),
            *self._wait(),
        ]

    def _reading_said(self, mine: bool) -> str:
        whose = "your computer" if mine else "its designer's computer"
        if self.reading:
            return f"its score from {whose} is up to date"
        if mine:
            return "its score from your computer is out of date, and needs measuring again"
        return (
            "its score from its designer's computer is out of date; only its designer "
            "can measure it again"
        )

    def _wait(self) -> list[Check]:
        """The sixth line, only on a game a round voted down or vetoed (§2.2 step 6)."""
        if self.again is None:
            return []
        days = quota.DECLINED_WAIT.days
        if self.waiting:
            says = (
                f"the round declined it, so it waits {days} days: it can be nominated "
                f"again after {day(self.again)}"
            )
        else:
            says = f"the {days} days' wait after the round declined it are over"
        return [Check("wait", not self.waiting, says)]

    @property
    def eligible(self) -> bool:
        return all(check.met for check in self.checks)

    def _days_said(self) -> str:
        if self.since is None:
            return f"it has no publishing date, and needs {quota.SHORTLIST_DAYS} days on trial"
        since = day(self.since)
        if self.days >= quota.SHORTLIST_DAYS:
            return f"on trial since {since}, over {quota.SHORTLIST_DAYS} days"
        opens = self.opens
        return f"on trial since {since}; its {quota.SHORTLIST_DAYS} days are up on " + (
            day(opens) if opens is not None else "a day to come"
        )


def day(when: datetime) -> str:
    """A date as the page says it: *3 Oct 2026*."""
    return f"{when.day} {when:%b %Y}"


# ------------------------------------------------------------------------ the counts


_FINISHED = (
    "select m.seat0_user, m.seat1_user from matches m where m.game = %s and ("
    " exists (select 1 from finishes f where f.match = m.id"
    f"   and f.reason <> %s and f.reason <> '{AGREED}')"
    f" or (m.result is not null and coalesce(m.reason, '') <> %s"
    f"   and coalesce(m.reason, '') <> '{AGREED}'))"
)
"""Every match on a game with a finish on the board in it: a kept finish (either mode,
take-backs and restarts included) or a verdict on the row itself. A draw two people
agreed (Q570) is not one: it can be agreed before a move, so counting it would let two
friends make finishers without playing."""


def finishers(connection: Any, game: str) -> set[int]:
    """Everybody who has finished a game on this one, on the board (see the module)."""
    with connection.cursor() as cursor:
        cursor.execute(_FINISHED, (game, RESIGNED, RESIGNED))
        rows = cursor.fetchall()
    return {int(person) for row in rows for person in row if person is not None}


def finished(connection: Any, game: str, person: int) -> bool:
    """Whether this person has finished a game on this one."""
    return person in finishers(connection, game)


def qualifying(connection: Any, game: str, owner: int | None) -> list[int]:
    """The stars that count toward this game's shortlist, one per voter.

    A vote counts when its voter finished a game on it, is not the designer, and holds an
    account at least `quota.VOTER_ACCOUNT_AGE` old that nobody has shut. The age is read
    against the database's clock, now: a vote cast on the first day starts counting a
    week later without being cast again.
    """
    played = finishers(connection, game)
    with connection.cursor() as cursor:
        cursor.execute(
            "select v.voter, v.stars from votes v join users u on u.id = v.voter"
            " where v.game = %s and u.revoked is null and u.created <= now() - %s"
            " order by v.voter",
            (game, quota.VOTER_ACCOUNT_AGE),
        )
        rows = cursor.fetchall()
    return [
        int(stars)
        for voter, stars in rows
        if int(voter) in played and (owner is None or int(voter) != owner)
    ]


def bar(connection: Any, game: str, reading: Mapping[str, Any] | None) -> Bar | None:
    """Where this game stands against the bar, or None for a game not designed here.

    `reading` is what `drafts.attached` found (the server passes it, so a test can hand
    one in). A reading that says it is not viable does not count.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select owner, origin, published,"
            " floor(extract(epoch from now() - published) / 86400)::int"
            " from games where hash = %s",
            (game,),
        )
        found = cursor.fetchone()
    if found is None or str(found[1]) != DESIGNED:
        return None
    owner = None if found[0] is None else int(found[0])
    # Imported here: `rounds` reads this module's counts at its own import.
    from chess_variant_sandbox.site import rounds

    out = rounds.declined(connection, game)
    with connection.cursor() as cursor:
        cursor.execute("select now()")
        now = cursor.fetchone()[0]
    everybody = finishers(connection, game)
    stars = qualifying(connection, game, owner)
    return Bar(
        game=game,
        owner=owner,
        since=found[2],
        days=0 if found[3] is None else int(found[3]),
        finishers=len(everybody - {owner} if owner is not None else everybody),
        votes=len(stars),
        mean=sum(stars) / len(stars) if stars else None,
        reading=reading is not None and reading.get("viable", True) is not False,
        again=None if out is None else out.again,
        waiting=out is not None and out.again > now,
    )


# ------------------------------------------------------------------------ the press


def shortlisted(connection: Any, owner: int | None = None) -> int:
    """How many games are shortlisted now, or how many of this designer's are."""
    clause = "" if owner is None else " and owner = %s"
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from games where origin = %s and visibility = 'public'"
            " and nominated is not null and revoked is null and deleted is null"
            f" and retired is null{clause}",
            (DESIGNED,) if owner is None else (DESIGNED, owner),
        )
        return int(cursor.fetchone()[0])


def _caps(connection: Any, owner: int) -> str:
    """Why a nomination cannot take a place now, or the empty string."""
    mine = shortlisted(connection, owner)
    if mine >= quota.SHORTLISTED_A_DESIGNER:
        return (
            f"You already have {mine} game{'' if mine == 1 else 's'} shortlisted, which is "
            f"the limit of {quota.SHORTLISTED_A_DESIGNER}. A place opens when the round "
            "decides on it."
        )
    everybody = shortlisted(connection)
    if everybody >= quota.SHORTLIST_SIZE:
        return (
            f"The shortlist is full: it holds {quota.SHORTLIST_SIZE} games. A place opens "
            "when the round takes one off it."
        )
    return ""


@dataclass(frozen=True)
class Nominated:
    """What a press of *Nominate* did."""

    hash: str
    already: bool

    @property
    def said(self) -> str:
        if self.already:
            return "Already shortlisted: members vote on it at the next round."
        return (
            "Nominated. It is shortlisted: members vote on it at the next monthly round, "
            "and the person who runs the site confirms or vetoes."
        )


def nominate(
    connection: Any, game: str, user: int, reading: Mapping[str, Any] | None
) -> Nominated:
    """Shortlist this person's trial game, or refuse in a sentence. Nothing commits.

    The refusals come in the order its designer can act on them: whose game it is, its
    level, the bar (every unmet condition named), then the two caps, which are counted
    under the lock. A killed or deleted game never reaches here: the server answers 404
    first.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select owner, origin, visibility, nominated, retired, rested from games"
            " where hash = %s for update",
            (game,),
        )
        found = cursor.fetchone()
    if found is None:
        raise KeyError(f"no game {game}")
    owner, origin, visibility, nominated, retired, rested = found
    if str(origin) != DESIGNED:
        raise NominateError("This game came with the site. There is nothing to nominate.")
    if owner is None or int(owner) != user:
        raise NominateError("Only the person who made this game can nominate it.")
    level = stages.stage(
        {"origin": origin, "visibility": visibility, "nominated": nominated,
         "retired": retired, "rested": rested}
    )
    if level == stages.SHORTLISTED:
        return Nominated(game, already=True)
    if level != stages.ON_TRIAL:
        raise NominateError(
            "Only a game on trial can be nominated. Publish it first, and nominate it once "
            "enough other people have played it and liked it."
        )
    standing = bar(connection, game, reading)
    if standing is None:  # pragma: no cover - the origin was read above
        raise KeyError(f"no game {game}")
    unmet = [check.says for check in standing.lines(mine=True) if not check.met]
    if unmet:
        raise NominateError("It can't be nominated yet: " + "; ".join(unmet) + ".")
    if getattr(connection, "autocommit", False):
        raise RuntimeError("the shortlist's caps hold their lock until the caller commits")
    with connection.cursor() as cursor:
        cursor.execute("select pg_advisory_xact_lock(hashtext(%s))", (_LOCK,))
    full = _caps(connection, user)
    if full:
        raise NominateError(full)
    with connection.cursor() as cursor:
        cursor.execute("update games set nominated = now() where hash = %s", (game,))
    return Nominated(game, already=False)


# ------------------------------------------------------------------ what the page is sent


def panel(
    connection: Any, game: str, viewer: int | None, reading: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The game page's shortlist lines under *Where it stands*, or None.

    Only a game on trial has them. `checks` are the five conditions, each met or not with
    its sentence; `says` sums them up; `nominable` is whether its designer's button works,
    and `waiting` why it does not. Only the designer is sent the button (`mine`).
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select origin, visibility, nominated, retired, revoked, deleted, rested"
            " from games where hash = %s",
            (game,),
        )
        found = cursor.fetchone()
    if found is None or found[4] is not None or found[5] is not None:
        return None
    level = stages.stage(
        {"origin": found[0], "visibility": found[1], "nominated": found[2],
         "retired": found[3], "rested": found[6]}
    )
    if level != stages.ON_TRIAL:
        return None
    standing = bar(connection, game, reading)
    if standing is None:
        return None
    mine = viewer is not None and standing.owner == viewer
    eligible = standing.eligible
    waiting = ""
    if mine and eligible and viewer is not None:
        waiting = _caps(connection, viewer)
    elif mine:
        waiting = "It can be nominated once every line above holds."
    if eligible:
        says = "It meets the shortlist's bar, so its designer can nominate it."
    elif standing.opens is not None and standing.days < quota.SHORTLIST_DAYS:
        says = f"It can be shortlisted after {day(standing.opens)}, when all of these hold:"
    else:
        says = "It can be shortlisted when all of these hold:"
    return {
        "checks": [check.as_dict() for check in standing.lines(mine=mine)],
        "eligible": eligible,
        "says": says,
        "mine": mine,
        "nominable": mine and eligible and not waiting,
        "waiting": waiting,
    }


COUNTS = (
    "Your stars count toward this game's shortlist, because you've finished a game on it."
)
NOT_YET = (
    "Your stars show here. They count toward the shortlist once you've finished a game "
    "on it."
)
TOO_NEW = (
    "Your stars show here. They count toward the shortlist once your account is "
    f"{quota.VOTER_ACCOUNT_AGE.days} days old."
)
OWN = "Your stars show here. A designer's own never count toward its shortlist."
"""§3.3's sentence beside the star control, by who is voting. The vote is never refused:
the sentence only says whether it counts (N13's spirit, behaviour first, no prompt)."""


def vote_counts(connection: Any, game: str, viewer: int | None) -> str:
    """§3.3's sentence for this person on this game, or empty where no shortlist is near.

    Said only on a game on trial, and only to somebody signed in.
    """
    if viewer is None:
        return ""
    with connection.cursor() as cursor:
        cursor.execute(
            "select g.origin, g.visibility, g.nominated, g.retired, g.owner,"
            " (select u.created <= now() - %s from users u where u.id = %s), g.rested"
            " from games g where g.hash = %s and g.revoked is null and g.deleted is null",
            (quota.VOTER_ACCOUNT_AGE, viewer, game),
        )
        found = cursor.fetchone()
    if found is None:
        return ""
    origin, visibility, nominated, retired, owner, seasoned, rested = found
    level = stages.stage(
        {"origin": origin, "visibility": visibility, "nominated": nominated,
         "retired": retired, "rested": rested}
    )
    if level != stages.ON_TRIAL:
        return ""
    if owner is not None and int(owner) == viewer:
        return OWN
    if not finished(connection, game, viewer):
        return NOT_YET
    if not seasoned:
        return TOO_NEW
    return COUNTS
