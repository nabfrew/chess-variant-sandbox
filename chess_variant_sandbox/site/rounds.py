"""The round: the members' vote, then the maintainer's confirm or veto.

Q503h of `docs/plan-game-pipeline.md` (§2.2 *The round, in order*, §3.2's *Where it
stands* lines, §3.3 the round page, §3.4 `/admin`'s *Round and reports*), at the user's
answers Q1 B (members vote for a week, the maintainer confirms or vetoes) and Q4 A (at
most 4 a month), with §6's first *needs decision* on its default (A, 2026-09-28).

## The round, in order

1. **It opens on the 1st of the month** (UTC), at the first sweep that month (`sweep`,
   which `site sweep`, `site load` and the round's own pages run). Its entrants are fixed
   then: first every game the last round passed that the maintainer neither confirmed nor
   vetoed, carried with its vote (step 5); then every shortlisted game whose four runs are
   current (`measurements.state = 'current'` with a headline). A game still waiting for its
   runs waits for the next round. A month whose first sweep comes after the vote would
   have closed opens no round.
2. **For `quota.ROUND_VOTING` (7 days)** a member votes yes or no on each new entrant, and
   may change it until the close. Who may vote is the shortlist's qualifying voter
   (`shortlist.qualifying`'s rule): finished a game on it, not its designer, an account
   `quota.VOTER_ACCOUNT_AGE` old that nobody shut (`eligibility`). **No count is shown
   before the close**: `round_entrants.yes` and `no` are null until then, and nothing sends
   a running tally.
3. **At the close** the sweep counts each new entrant's votes (a shut account's no longer
   counts) and writes them. **A nominee passes with more yes than no and at least
   `quota.ROUND_YES` (4) yes** (`passes`). One that does not is `voted-down`: it goes back
   to On trial (`games.nominated` cleared) and waits `quota.DECLINED_WAIT` (60 days) before
   its designer may nominate it again (`shortlist.nominate` asks `waiting_since`).
4. **The maintainer then confirms or vetoes** each entrant that passed, from `/admin`, with
   a reason either way (`decide`). There is no way to confirm a game the vote refused:
   *confirms or vetoes* is a brake, not a second way in. A veto sends the game back to On
   trial with the same 60-day wait.
5. **At most `quota.ROUND_PLACES` (4) are confirmed a round**, highest share of yes first,
   then the oldest nomination (`ranked`), carried games ahead of new ones. A confirm is
   refused to a game with four unvetoed passed games ranked above it (`running`); it stays
   shortlisted and is carried, vote kept, to the next round. So is any passed game the
   maintainer has not decided on when the next round opens.

Decisions are taken on the latest round only, after its close.

## Confirmed is not yet Library

A confirm records the decision and shows the maintainer the exact `chess-variant-sandbox
archive` command (`command`), with the reason in `--note` (written first in the spec's history) and the
designer in `--credit` (`provenance.author`; `corpus._provenance`). The site never writes the
repository (`plan-website.md` §3 rule 3). A confirmed game stays **Shortlisted**, its page
saying so, until that commit is deployed; then `site load` finds its spec in `archive/`,
sees the confirmation (`confirmed`) and takes the `designed` row over as a Library game,
keeping its designer as owner and writing a `level_moves` row from `designed` with the
confirm's reason. The loader stays the only writer of an origin.

Nothing here commits: the caller owns the transaction, as for `shortlist` and `votes`.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

from chess_variant_sandbox.site import quota, shortlist, stages

DESIGNED = "designed"
PUBLIC = "public"

CONFIRMED = "confirmed"
VETOED = "vetoed"
VOTED_DOWN = "voted-down"
DECLINES = (VETOED, VOTED_DOWN)
"""The two ways out of a round that send a game back to On trial with a wait."""

VOTING = "voting"
TALLYING = "tallying"
DECIDING = "deciding"
DECIDED = "decided"
"""A round's state as the pages say it: members voting; the vote closed and the sweep
not yet run; the maintainer yet to decide on a game that passed; nothing left to decide."""

REASON_LENGTH = 500
"""The longest reason a decision may carry. It is shown on the game's page and in *What's
new*, so it is a sentence or two."""

RULE = (
    "Members who have played it decide whether it joins the library. The person who runs "
    "the site can still veto it, with a reason."
)
"""§3.3's sentence beside the yes and no."""

ENTERS = (
    "A game enters a round when it is on the shortlist and the site has scored it by the "
    "1st. Its designer puts it on the shortlist once enough other people have played it "
    "and voted."
)
"""What puts a game into a round, said where a round has no entrant (L4 of
`docs/plan-launch.md` §3.4): the page that is not announced still says how to fill it."""

PREMISE = "not checked yet: no strength check is on file for this game"
COST = (
    "about 16 to 49 core-hours if its player needs training, under half a core-hour if not "
    "(plan-game-pipeline §2.5)"
)
"""What `/admin` says of an entrant's premise verdict where no check is on file, and of its
Library cost: the plan's own range, never a number the site made up. Where a check is on
file, `premise_said` reads it instead."""


def premise_said(digest: str, spec: Any, runs_dir: Path | None) -> str:
    """The entrant's strength-premise verdict in `/admin`'s words (Q502d), or `PREMISE`.

    Read from `runs/<hash>/premise.json` beside the score, judged against what a check
    taken now would rest on: a stale verdict says it is stale and why, never pass or fail.
    """
    if runs_dir is None:
        return PREMISE
    from chess_variant_sandbox import premise

    found = premise.read(digest, runs_dir, spec if isinstance(spec, dict) else None)
    if found is None:
        return PREMISE
    word = {premise.PASSES: "passes", premise.FAILS: "fails"}.get(found.state, "stale")
    return f"{word}: {found.words()}"

_LOCK = "round"


class RoundError(ValueError):
    """A refused vote or decision, in the sentence its reader reads (a 400)."""


def _now(when: datetime | None) -> datetime:
    return datetime.now(timezone.utc) if when is None else when


def month_of(when: datetime) -> date:
    """The 1st of the month this moment is in, in UTC: the round it would open."""
    return when.astimezone(timezone.utc).date().replace(day=1)


def next_month(day: date) -> date:
    return date(day.year + 1, 1, 1) if day.month == 12 else date(day.year, day.month + 1, 1)


def start(day: date) -> datetime:
    return datetime.combine(day, time(0), tzinfo=timezone.utc)


def said_day(when: datetime | date) -> str:
    """A date as the page says it: *3 Oct 2026* (`shortlist.day`'s form)."""
    return f"{when.day} {when:%b %Y}"


def passes(yes: int, no: int) -> bool:
    """§2.2 step 3: more yes than no, and at least `quota.ROUND_YES` yes."""
    return yes > no and yes >= quota.ROUND_YES


# ------------------------------------------------------------------------ the rounds


@dataclass(frozen=True)
class Round:
    day: date
    opens: datetime
    closes: datetime
    closed: datetime | None

    @property
    def slug(self) -> str:
        return self.day.isoformat()

    @property
    def words(self) -> str:
        return f"{self.day:%B %Y}"

    @property
    def named(self) -> str:
        return f"the round of {said_day(self.day)}"

    def voting(self, when: datetime) -> bool:
        return self.closed is None and self.opens <= when < self.closes


_ROUND = "day, opens, closes, closed"


def _round(row: Any) -> Round:
    return Round(row[0], row[1], row[2], row[3])


def get(connection: Any, day: date) -> Round | None:
    with connection.cursor() as cursor:
        cursor.execute(f"select {_ROUND} from rounds where day = %s", (day,))
        found = cursor.fetchone()
    return None if found is None else _round(found)


def latest(connection: Any) -> Round | None:
    with connection.cursor() as cursor:
        cursor.execute(f"select {_ROUND} from rounds order by day desc limit 1")
        found = cursor.fetchone()
    return None if found is None else _round(found)


def every(connection: Any) -> list[Round]:
    """Every round, newest first."""
    with connection.cursor() as cursor:
        cursor.execute(f"select {_ROUND} from rounds order by day desc")
        return [_round(row) for row in cursor.fetchall()]


def parse_day(text: str) -> date:
    """A round's date from a path, or the `KeyError` an unknown one gets (a 404)."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise KeyError(f"no round {text}") from None


# ------------------------------------------------------------------------ the entrants


@dataclass(frozen=True)
class Entrant:
    round: date
    game: str
    name: str
    owner: int | None
    designer: str
    nominated: datetime
    carried: date | None
    yes: int | None
    no: int | None
    decision: str | None
    reason: str
    decided: datetime | None
    score: float | None

    @property
    def tallied(self) -> bool:
        return self.yes is not None and self.no is not None

    @property
    def passed(self) -> bool:
        return self.yes is not None and self.no is not None and passes(self.yes, self.no)

    @property
    def share(self) -> float:
        if self.yes is None or self.no is None or self.yes + self.no == 0:
            return 0.0
        return self.yes / (self.yes + self.no)

    @property
    def tally(self) -> str:
        return f"{self.yes} yes, {self.no} no"


_ENTRANT = (
    "select e.round, e.game, g.name, g.owner, coalesce(u.handle, ''), e.nominated,"
    " e.carried, e.yes, e.no, d.decision, coalesce(d.reason, ''), d.decided,"
    " case when m.state = 'current' then m.headline end"
    " from round_entrants e join games g on g.hash = e.game"
    " left join users u on u.id = g.owner"
    " left join round_decisions d on d.round = e.round and d.game = e.game"
    " left join measurements m on m.hash = e.game"
)


def _entrant(row: Any) -> Entrant:
    return Entrant(
        round=row[0],
        game=str(row[1]),
        name=str(row[2]),
        owner=None if row[3] is None else int(row[3]),
        designer=str(row[4]) or "a former player",
        nominated=row[5],
        carried=row[6],
        yes=None if row[7] is None else int(row[7]),
        no=None if row[8] is None else int(row[8]),
        decision=None if row[9] is None else str(row[9]),
        reason=str(row[10]),
        decided=row[11],
        score=None if row[12] is None else float(row[12]),
    )


def entrants(connection: Any, day: date) -> list[Entrant]:
    """This round's entrants, in `ranked` order."""
    with connection.cursor() as cursor:
        cursor.execute(_ENTRANT + " where e.round = %s", (day,))
        return ranked(_entrant(row) for row in cursor.fetchall())


def ranked(found: Iterable[Entrant]) -> list[Entrant]:
    """The order the places go in (§2.2 step 5): carried games first, then the highest
    share of yes, then the oldest nomination. Before the close nothing is tallied, so the
    order is the nominations'."""
    return sorted(
        found, key=lambda e: (e.carried is None, -e.share, e.nominated, e.game)
    )


def running(found: list[Entrant]) -> set[str]:
    """The passed entrants a confirm may still reach: those with fewer than
    `quota.ROUND_PLACES` unvetoed passed entrants ranked above them."""
    ahead = 0
    reach: set[str] = set()
    for entrant in ranked(found):
        if not entrant.passed or entrant.decision == VETOED:
            continue
        if ahead < quota.ROUND_PLACES:
            reach.add(entrant.game)
        ahead += 1
    return reach


# ------------------------------------------------------------------------ open and close


@dataclass(frozen=True)
class Swept:
    """What one sweep did, as the command line says it."""

    opened: Round | None
    entered: list[str]
    closed: list[Round]
    voted_down: list[str]

    @property
    def lines(self) -> list[str]:
        said = []
        for found in self.closed:
            said.append(f"closed {found.named}")
        for name in self.voted_down:
            said.append(f"  voted down {name!r}: back on trial")
        if self.opened is not None:
            said.append(
                f"opened {self.opened.named} with {len(self.entered)} entrant(s)"
                + (": " + ", ".join(repr(n) for n in self.entered) if self.entered else "")
            )
        return said


def _lock(connection: Any, wait: bool) -> bool:
    """The round's advisory lock, to the caller's commit; whether it was had.

    Keyed by the schema too, so two stores on one server (the test suite's) never wait on
    each other's round."""
    verb = "pg_advisory_xact_lock" if wait else "pg_try_advisory_xact_lock"
    with connection.cursor() as cursor:
        cursor.execute(f"select {verb}(hashtext(current_schema() || %s))", (_LOCK,))
        found = cursor.fetchone()
    return not (not wait and found is not None and found[0] is False)


def due(connection: Any, when: datetime | None = None) -> bool:
    """Whether a sweep now would do anything: a round to close, or this month's to open."""
    moment = _now(when)
    day = month_of(moment)
    with connection.cursor() as cursor:
        cursor.execute(
            "select exists (select 1 from rounds where closed is null and closes <= %s),"
            " exists (select 1 from rounds where day = %s)",
            (moment, day),
        )
        closing, opened = cursor.fetchone()
    return bool(closing) or (not opened and moment < start(day) + quota.ROUND_VOTING)


def sweep(connection: Any, when: datetime | None = None, wait: bool = True) -> Swept:
    """Close every round whose week is over, then open this month's if it is due.

    Idempotent: a second sweep the same moment does nothing. One advisory lock, so two
    requests arriving together on the 1st open one round. A page that sweeps on its way
    (`wait=False`) asks `due` first and skips the sweep when somebody else holds the lock.
    """
    moment = _now(when)
    nothing = Swept(None, [], [], [])
    if not wait and not due(connection, moment):
        return nothing
    if getattr(connection, "autocommit", False):
        raise RuntimeError("the round's sweep holds its lock until the caller commits")
    if not _lock(connection, wait):
        return nothing
    closed, down = _close(connection, moment)
    opened, entered = _open(connection, moment)
    return Swept(opened, entered, closed, down)


def _close(connection: Any, moment: datetime) -> tuple[list[Round], list[str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_ROUND} from rounds where closed is null and closes <= %s order by day",
            (moment,),
        )
        due = [_round(row) for row in cursor.fetchall()]
    down: list[str] = []
    for found in due:
        for entrant in entrants(connection, found.day):
            if entrant.carried is not None:
                continue
            yes, no = _tally(connection, found.day, entrant.game)
            with connection.cursor() as cursor:
                cursor.execute(
                    "update round_entrants set yes = %s, no = %s where round = %s and game = %s",
                    (yes, no, found.day, entrant.game),
                )
            if passes(yes, no):
                continue
            _decline(
                connection, found.day, entrant.game, VOTED_DOWN, f"{yes} yes and {no} no",
                None, moment,
            )
            down.append(entrant.name)
        with connection.cursor() as cursor:
            cursor.execute("update rounds set closed = %s where day = %s", (moment, found.day))
    return due, down


def _tally(connection: Any, day: date, game: str) -> tuple[int, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) filter (where v.yes), count(*) filter (where not v.yes)"
            " from round_votes v join users u on u.id = v.person"
            " where v.round = %s and v.game = %s and u.revoked is null",
            (day, game),
        )
        yes, no = cursor.fetchone()
    return int(yes), int(no)


def _decline(
    connection: Any, day: date, game: str, decision: str, reason: str,
    decider: int | None, moment: datetime,
) -> None:
    """Record a voted-down or vetoed game and send it back to On trial."""
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into round_decisions (round, game, decision, reason, decided, decider)"
            " values (%s, %s, %s, %s, %s, %s)",
            (day, game, decision, reason, moment, decider),
        )
        cursor.execute("update games set nominated = null where hash = %s", (game,))


_SHORTLISTED = (
    "g.origin = 'designed' and g.visibility = 'public' and g.nominated is not null"
    " and g.revoked is null and g.deleted is null and g.retired is null"
)


def _open(connection: Any, moment: datetime) -> tuple[Round | None, list[str]]:
    day = month_of(moment)
    closes = start(day) + quota.ROUND_VOTING
    if moment >= closes or get(connection, day) is not None:
        return None, []
    before = latest(connection)
    if before is not None and before.day > day:  # pragma: no cover - a clock gone back
        return None, []
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into rounds (day, opens, closes) values (%s, %s, %s)",
            (day, moment, closes),
        )
    entered: list[str] = []
    carried: set[str] = set()
    if before is not None:
        for entrant in entrants(connection, before.day):
            if not entrant.passed or entrant.decision is not None:
                continue
            with connection.cursor() as cursor:
                cursor.execute(
                    f"select 1 from games g where g.hash = %s and {_SHORTLISTED}",
                    (entrant.game,),
                )
                still = cursor.fetchone() is not None
            if not still:
                continue
            with connection.cursor() as cursor:
                cursor.execute(
                    "insert into round_entrants (round, game, nominated, carried, yes, no)"
                    " values (%s, %s, %s, %s, %s, %s)",
                    (day, entrant.game, entrant.nominated, entrant.carried or before.day,
                     entrant.yes, entrant.no),
                )
            carried.add(entrant.game)
            entered.append(entrant.name)
    with connection.cursor() as cursor:
        cursor.execute(
            "select g.hash, g.name, g.nominated from games g"
            f" join measurements m on m.hash = g.hash where {_SHORTLISTED}"
            " and g.rested is null and m.state = 'current' and m.headline is not null"
            " and not exists (select 1 from round_decisions d where d.game = g.hash"
            "                 and d.decision = %s)"
            " and g.hash <> all(%s) order by g.nominated, g.hash",
            (CONFIRMED, sorted(carried)),
        )
        fresh = cursor.fetchall()
    for digest, name, nominated in fresh:
        with connection.cursor() as cursor:
            cursor.execute(
                "insert into round_entrants (round, game, nominated) values (%s, %s, %s)",
                (day, digest, nominated),
            )
        entered.append(str(name))
    return get(connection, day), entered


# ------------------------------------------------------------------------ the vote


def eligibility(
    connection: Any, game: str, owner: int | None, person: int | None, when: datetime
) -> str:
    """Why this person may not vote on this entrant, or the empty string."""
    if person is None:
        return "Sign in to vote. Members who have finished a game on it vote on it."
    if owner is not None and owner == person:
        return "A designer does not vote on their own game."
    with connection.cursor() as cursor:
        cursor.execute("select revoked, created from users where id = %s", (person,))
        found = cursor.fetchone()
    if found is None or found[0] is not None:
        return "This account cannot vote."
    if not shortlist.finished(connection, game, person):
        return (
            "Members who have finished a game on it vote on it. Play it to the end, and "
            "your vote opens."
        )
    if found[1] > when - quota.VOTER_ACCOUNT_AGE:
        return (
            f"Your account can vote once it is {quota.VOTER_ACCOUNT_AGE.days} days old."
        )
    return ""


def _entry(connection: Any, day: date, game: str) -> Entrant | None:
    with connection.cursor() as cursor:
        cursor.execute(_ENTRANT + " where e.round = %s and e.game = %s", (day, game))
        found = cursor.fetchone()
    return None if found is None else _entrant(found)


def vote(
    connection: Any, day: date, game: str, person: int, yes: bool,
    when: datetime | None = None,
) -> str:
    """Cast or change this member's yes or no, or refuse in a sentence."""
    moment = _now(when)
    found = get(connection, day)
    if found is None:
        raise KeyError(f"no round {day}")
    if not found.voting(moment):
        raise RoundError(
            f"The vote at {found.named} is closed."
            if moment >= found.closes or found.closed is not None
            else f"The vote at {found.named} has not opened."
        )
    entrant = _entry(connection, day, game)
    if entrant is None:
        raise RoundError(f"That game is not an entrant of {found.named}.")
    if entrant.carried is not None:
        raise RoundError(
            f"It keeps the vote it had at the round of {said_day(entrant.carried)}; there "
            "is no new vote on it."
        )
    why = eligibility(connection, game, entrant.owner, person, moment)
    if why:
        raise RoundError(why)
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into round_votes (round, game, person, yes, voted)"
            " values (%s, %s, %s, %s, %s) on conflict (round, game, person)"
            " do update set yes = excluded.yes, voted = excluded.voted",
            (day, game, person, yes, moment),
        )
    return (
        f"Your {'yes' if yes else 'no'} is in. You can change it until the vote closes on "
        f"{said_day(found.closes)}."
    )


def ballots(connection: Any, day: date, person: int | None) -> dict[str, bool]:
    """This member's own votes at a round, by game. Nobody else's are ever sent."""
    if person is None:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, yes from round_votes where round = %s and person = %s",
            (day, person),
        )
        return {str(row[0]): bool(row[1]) for row in cursor.fetchall()}


# ------------------------------------------------------------------------ the decision


@dataclass(frozen=True)
class Decided:
    game: str
    name: str
    decision: str
    command: str

    @property
    def said(self) -> str:
        if self.decision == CONFIRMED:
            return (
                f"confirmed {self.name}; it joins the library when its archive is deployed: "
                f"{self.command}"
            )
        return f"vetoed {self.name}; it is back on trial, and waits {quota.DECLINED_WAIT.days} days"


def command(entrant: Entrant, reason: str, day: date) -> str:
    """The maintainer's `archive` command for a confirmed game: the reason and the round
    in `--note`, written first in `provenance.history`, and the designer in `--credit`
    (`corpus._provenance`)."""
    note = f"{reason.strip()} Confirmed at the round of {said_day(day)}."
    return (
        f"chess-variant-sandbox archive {entrant.game} --credit "
        f"{shlex.quote(entrant.designer)} --note {shlex.quote(note)}"
    )


def decide(
    connection: Any, game: str, confirm: bool, reason: str, decider: int | None,
    when: datetime | None = None,
) -> Decided:
    """The maintainer's confirm or veto of an entrant that passed (see the module)."""
    moment = _now(when)
    if getattr(connection, "autocommit", False):
        raise RuntimeError("the round's places hold their lock until the caller commits")
    _lock(connection, wait=True)
    found = latest(connection)
    if found is None:
        raise RoundError("No round has opened yet.")
    if found.closed is None:
        raise RoundError(
            f"Members are voting until {said_day(found.closes)}; decide after the close."
        )
    field = entrants(connection, found.day)
    entrant = next((one for one in field if one.game == game), None)
    if entrant is None:
        raise RoundError(f"That game is not an entrant of {found.named}.")
    if not entrant.passed:
        raise RoundError(
            "The members did not pass it, and a game the vote refused cannot be added."
        )
    if entrant.decision is not None:
        raise RoundError(f"{entrant.name} is already decided: {entrant.decision}.")
    reason = " ".join(reason.split())
    if not reason:
        raise RoundError("A decision needs its reason: it is shown on the game's page.")
    if len(reason) > REASON_LENGTH:
        raise RoundError(f"A reason is at most {REASON_LENGTH} characters.")
    if not confirm:
        _decline(connection, found.day, game, VETOED, reason, decider, moment)
        return Decided(game, entrant.name, VETOED, "")
    if game not in running(field):
        raise RoundError(
            f"The {quota.ROUND_PLACES} places this round go to the passed games ranked above "
            "it. It stays shortlisted and goes first at the next round, its vote kept."
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into round_decisions (round, game, decision, reason, decided, decider)"
            " values (%s, %s, %s, %s, %s, %s)",
            (found.day, game, CONFIRMED, reason, moment, decider),
        )
    return Decided(game, entrant.name, CONFIRMED, command(entrant, reason, found.day))


def confirmed(connection: Any) -> dict[str, str]:
    """Every game a round confirmed, by hash, with the maintainer's reason.

    What `site load` reads to take a `designed` row over once its spec is in `archive/`.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, reason from round_decisions where decision = %s"
            " order by decided",
            (CONFIRMED,),
        )
        return {str(row[0]): str(row[1]) for row in cursor.fetchall()}


@dataclass(frozen=True)
class Decline:
    day: date
    decision: str
    reason: str
    decided: datetime

    @property
    def again(self) -> datetime:
        """When its designer may nominate it again."""
        return self.decided + quota.DECLINED_WAIT

    @property
    def why(self) -> str:
        if self.decision == VOTED_DOWN:
            return f"voted down, {self.reason}"
        return f"vetoed: {self.reason}"


def declined(connection: Any, game: str) -> Decline | None:
    """This game's latest voted-down or vetoed decision, or None."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select round, decision, reason, decided from round_decisions"
            " where game = %s and decision = any(%s) order by decided desc limit 1",
            (game, list(DECLINES)),
        )
        found = cursor.fetchone()
    return None if found is None else Decline(found[0], str(found[1]), str(found[2]), found[3])


# ------------------------------------------------------------------------ what pages read


def state(connection: Any, found: Round, when: datetime) -> str:
    if found.voting(when):
        return VOTING
    if found.closed is None:
        return TALLYING
    newest = latest(connection)
    if newest is not None and newest.day == found.day and any(
        one.passed and one.decision is None for one in entrants(connection, found.day)
    ):
        return DECIDING
    return DECIDED


def _states(found: Round, now_state: str) -> str:
    if now_state == VOTING:
        return (
            f"Members are voting until {said_day(found.closes)}. The counts are shown after "
            "the close, so early votes do not steer later ones."
        )
    if now_state == TALLYING:
        return "The vote has closed; the counts are shown once they are tallied."
    if now_state == DECIDING:
        return (
            "The vote has closed. The person who runs the site confirms or vetoes each game "
            f"that passed, with a reason, at most {quota.ROUND_PLACES} this round."
        )
    return "This round is decided."


def _decision_says(entrant: Entrant) -> str:
    if entrant.decision == CONFIRMED:
        return f"Confirmed: {entrant.reason}"
    if entrant.decision == VETOED:
        return f"Vetoed: {entrant.reason}"
    if entrant.decision == VOTED_DOWN:
        return "Voted down."
    if entrant.passed:
        return "Passed; the person who runs the site has yet to decide."
    return ""


def _admin_decision_says(entrant: Entrant) -> str:
    """`/admin`'s decision cell in words, never the stored `voted-down` (Q537): the
    decision and its reason, which for a vote is the count. Empty while none is taken."""
    words = {CONFIRMED: "Confirmed", VETOED: "Vetoed", VOTED_DOWN: "Voted down"}
    if entrant.decision is None:
        return ""
    said = words.get(entrant.decision, entrant.decision)
    return f"{said}: {entrant.reason}" if entrant.reason else f"{said}."


def page(
    connection: Any, day: date | None, viewer: int | None, when: datetime | None = None
) -> dict[str, Any]:
    """The round page's answer (§3.3): the round `day`, or the latest with `None`."""
    moment = _now(when)
    found = latest(connection) if day is None else get(connection, day)
    listed = [{"round": one.slug, "words": one.words} for one in every(connection)]
    if found is None:
        if day is not None:
            raise KeyError(f"no round {day}")
        first = next_month(month_of(moment))
        return {
            "round": None,
            "rounds": listed,
            "says": (
                "No round has opened yet. Rounds open on the 1st of each month; the next is "
                f"on {said_day(first)}. {ENTERS}"
            ),
            "rule": RULE,
            "entrants": [],
        }
    now_state = state(connection, found, moment)
    mine = ballots(connection, found.day, viewer)
    shown = []
    for entrant in entrants(connection, found.day):
        voting = now_state == VOTING and entrant.carried is None
        why = (
            eligibility(connection, entrant.game, entrant.owner, viewer, moment)
            if voting else ""
        )
        stars, star_votes = _stars(connection, entrant.game)
        players = shortlist.finishers(connection, entrant.game) - (
            {entrant.owner} if entrant.owner is not None else set()
        )
        counted = found.closed is not None or entrant.carried is not None
        shown.append({
            "hash": entrant.game,
            "name": entrant.name,
            "designer": entrant.designer,
            "score": entrant.score,
            "stars": stars,
            "starVotes": star_votes,
            "players": len(players),
            "carried": None if entrant.carried is None else said_day(entrant.carried),
            "voting": voting,
            "mayVote": voting and not why,
            "why": why,
            "yourVote": mine.get(entrant.game),
            "yes": entrant.yes if counted else None,
            "no": entrant.no if counted else None,
            "passed": entrant.passed if counted else None,
            "decision": entrant.decision,
            "decisionSays": _decision_says(entrant) if counted else "",
        })
    if not shown:
        says = (
            f"No game entered {found.named}: no shortlisted game had been scored by the "
            f"site by the 1st. The next round opens on "
            f"{said_day(next_month(month_of(moment)))}. {ENTERS}"
        )
    else:
        says = _states(found, now_state)
    return {
        "round": found.slug,
        "words": found.words,
        "named": found.named,
        "state": now_state,
        "closes": said_day(found.closes),
        "says": says,
        "rule": RULE,
        "entrants": shown,
        "rounds": listed,
    }


def announced(connection: Any) -> bool:
    """Whether the footer links the round: only while the latest round has an entrant (L4
    of `docs/plan-launch.md` §3.4). A round nobody entered is not announced; its address
    still answers, saying when the next opens and what enters a game (`page`)."""
    found = latest(connection)
    return found is not None and bool(entrants(connection, found.day))


def _stars(connection: Any, game: str) -> tuple[float | None, int]:
    with connection.cursor() as cursor:
        cursor.execute("select avg(stars), count(*) from votes where game = %s", (game,))
        mean, count = cursor.fetchone()
    return (None if mean is None else round(float(mean), 1)), int(count)


def on_the_game(
    connection: Any, game: str, viewer: int | None, when: datetime | None = None
) -> dict[str, Any] | None:
    """The game page's round line while members vote on it (§3.2 *In a round*), or None."""
    moment = _now(when)
    found = latest(connection)
    if found is None or not found.voting(moment):
        return None
    entrant = _entry(connection, found.day, game)
    if entrant is None or entrant.carried is not None:
        return None
    why = eligibility(connection, game, entrant.owner, viewer, moment)
    return {
        "round": found.slug,
        "named": found.named,
        # Empty: the level's own sentence above it (`said`) already names the round and
        # its close, and the page drew the two one under the other (Q517).
        "says": "",
        "rule": RULE,
        "mayVote": not why,
        "why": why,
        "yourVote": ballots(connection, found.day, viewer).get(game),
    }


def said(connection: Any, row: dict[str, Any], when: datetime | None = None) -> str | None:
    """The game page's sentence for a Shortlisted or On trial game the round has touched,
    or None for the level's plain one (`stages.said` asks)."""
    moment = _now(when)
    level = str(row.get("level") or stages.stage(row))
    digest = str(row.get("hash") or "")
    if not digest:
        return None
    if level == stages.ON_TRIAL:
        out = declined(connection, digest)
        if out is None:
            return None
        head = f"Not taken at the round of {said_day(out.day)}: {out.why}."
        if out.again > moment:
            return f"{head} It can be nominated again after {said_day(out.again)}."
        return f"{head} It can be nominated again."
    if level != stages.SHORTLISTED:
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            "select round from round_entrants where game = %s order by round desc limit 1",
            (digest,),
        )
        last = cursor.fetchone()
    entrant = None if last is None else _entry(connection, last[0], digest)
    if entrant is not None:
        found = get(connection, entrant.round)
        assert found is not None
        if entrant.decision == CONFIRMED:
            return (
                f"Confirmed at {found.named}: {entrant.reason} It joins the library at the "
                "site's next update."
            )
        if found.voting(moment) and entrant.carried is None:
            return f"In {found.named}: members are voting until {said_day(found.closes)}."
        if entrant.passed and entrant.decision is None:
            newest = latest(connection)
            if newest is not None and newest.day == found.day:
                head = f"Members passed it at {found.named} ({entrant.tally})."
                if entrant.game in running(entrants(connection, found.day)):
                    return head + (
                        " The person who runs the site confirms or vetoes it, with a reason."
                    )
                return (
                    head + f" This round's {quota.ROUND_PLACES} places went to games ranked "
                    "above it, so it goes first at the next round, its vote kept."
                )
    upcoming = next_month(month_of(moment))
    with connection.cursor() as cursor:
        cursor.execute(
            "select 1 from measurements where hash = %s and state = 'current'"
            " and headline is not null",
            (digest,),
        )
        measured = cursor.fetchone() is not None
    if measured:
        return f"Shortlisted: members vote on it at the round of {said_day(upcoming)}."
    return (
        "Shortlisted: members vote on it at the first round, on the 1st of a month, after "
        "the site has scored it (four runs)."
    )


# ------------------------------------------------------------------------ /admin


def admin_view(
    connection: Any, when: datetime | None = None, runs_dir: Path | None = None
) -> dict[str, Any]:
    """`/admin`'s *Round* (§3.4): the latest round, each entrant with what the maintainer
    needs to decide, and the buttons it may have. ``runs_dir`` is where each entrant's
    premise check is read from (`premise_said`); without one every entrant reads
    `PREMISE`."""
    moment = _now(when)
    found = latest(connection)
    if found is None:
        return {
            "round": None,
            "says": "No round has opened yet. The first opens on "
            f"{said_day(next_month(month_of(moment)))}.",
            "entrants": [],
        }
    now_state = state(connection, found, moment)
    field = entrants(connection, found.day)
    reach = running(field)
    specs: dict[str, Any] = {}
    if runs_dir is not None and field:
        with connection.cursor() as cursor:
            cursor.execute(
                "select hash, spec from games where hash = any(%s)",
                ([entrant.game for entrant in field],),
            )
            specs = {str(row[0]): row[1] for row in cursor.fetchall()}
    rows = []
    for entrant in field:
        counted = found.closed is not None or entrant.carried is not None
        open_to = now_state == DECIDING and entrant.passed and entrant.decision is None
        why = ""
        if now_state == VOTING:
            why = "members are voting"
        elif open_to and entrant.game not in reach:
            why = f"past this round's {quota.ROUND_PLACES} places; carried to the next round"
        rows.append({
            "hash": entrant.game,
            "name": entrant.name,
            "designer": entrant.designer,
            "score": entrant.score,
            "premise": premise_said(entrant.game, specs.get(entrant.game), runs_dir),
            "cost": COST,
            "carried": None if entrant.carried is None else said_day(entrant.carried),
            "yes": entrant.yes if counted else None,
            "no": entrant.no if counted else None,
            "passed": entrant.passed if counted else None,
            "decision": entrant.decision,
            "reason": entrant.reason,
            "decisionSays": _admin_decision_says(entrant),
            "mayConfirm": open_to and entrant.game in reach,
            "mayVeto": open_to,
            "why": why,
            "command": command(entrant, entrant.reason, found.day)
            if entrant.decision == CONFIRMED else "",
        })
    return {
        "round": found.slug,
        "words": found.words,
        "state": now_state,
        "says": _states(found, now_state),
        "entrants": rows,
    }


# ------------------------------------------------------------------------ What's new


@dataclass(frozen=True)
class Notice:
    """One line of *What's new* about a round, in `stages.Move`'s shape."""

    moved: datetime
    heading: str
    paragraphs: list[str]


def news(connection: Any) -> list[Notice]:
    """Each round's opening (with an entrant) and each decision, for *What's new*."""
    found: list[Notice] = []
    for one in every(connection)[: stages.NEWS_KEPT]:
        field = entrants(connection, one.day)
        if not field:
            continue
        names = ", ".join(entrant.name for entrant in field)
        found.append(Notice(
            one.opens,
            f"The round of {one.words} opens",
            [
                f"{len(field)} shortlisted game{'' if len(field) == 1 else 's'} "
                f"enter{'s' if len(field) == 1 else ''} it: {names}.",
                f"Members who have finished a game on one vote yes or no until "
                f"{said_day(one.closes)}. {RULE}",
            ],
        ))
        for entrant in field:
            if entrant.decision is None or entrant.decided is None:
                continue
            found.append(_decision_notice(one, entrant))
    return found


def _decision_notice(one: Round, entrant: Entrant) -> Notice:
    assert entrant.decided is not None
    again = said_day(entrant.decided + quota.DECLINED_WAIT)
    if entrant.decision == CONFIRMED:
        return Notice(entrant.decided, f"{entrant.name} is confirmed for the library", [
            f"Members passed it at {one.named} ({entrant.tally}), and the person who runs "
            "the site confirmed it.",
            f"The reason given: {entrant.reason}",
            f"It joins the library, credited to {entrant.designer}, at the site's next "
            "update.",
        ])
    if entrant.decision == VETOED:
        return Notice(entrant.decided, f"{entrant.name} is not taken at {one.named}", [
            f"Members passed it ({entrant.tally}), and the person who runs the site "
            "vetoed it.",
            f"The reason given: {entrant.reason}",
            f"It is back on trial, and can be nominated again after {again}.",
        ])
    return Notice(entrant.decided, f"{entrant.name} is not taken at {one.named}", [
        f"Members voted it down: {entrant.reason}. A game passes with more yes than no "
        f"and at least {quota.ROUND_YES} yes.",
        f"It is back on trial, and can be nominated again after {again}.",
    ])

