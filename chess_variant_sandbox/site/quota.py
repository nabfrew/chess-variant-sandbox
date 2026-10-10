"""Every number that rations the box, in one place, and the two checks that read them.

Task 1.8, closing §7's *"a per-user quota table with the numbers in one place -- the
leaderboard idle limit (§4), the daily canonical and picker quotas (§2), and the
concurrency cap on `minimax:nodes=200000` seats (§2) -- which is enforced in
`chess_variant_sandbox/site`, not `deploy/`"*.

Two of these are enforced here and now, at the one place a match is created, and both
refuse in a sentence a player can read. The strong-seat cap is also met by a game taking
its seat back after giving it up -- a move after sitting idle, a take-back that reopens a
finished game, a restart -- because that seats the strongest opponent as surely as a
start does (Q410). The other three are phase 2's queue, which does not exist yet; they
are written down here rather than in the plan's prose so that the worker reads a
constant instead of re-deriving a number from §2's costing.

## Where the numbers come from

§2 measured the box: 2 vCPU, `CPUQuota=150%` on the app. A move by
`minimax:nodes=200000` costs 0.52-0.61 s there, so thirty such games at a move every ten
seconds is 1.65 of the two cores -- the plan's own reason for rationing the strongest
agent rather than the site. A canonical measurement is 277 CPU-seconds and a picker
rating 1838, which is why one is a daily allowance and the other is not.

## What is not rationed

A visitor with no account is not counted, because there is no id to count by. The ceiling
on a visitor is the firewall's 240 connections a minute per source and Caddy's
`CPUQuota=75%` (§7), and a sandbox game is a row and a replay rather than a search. A
per-source quota on playing in here would be a second, worse copy of the front door's.

Signing in is the exception, and it is counted by source address (Q369). The front door
has a *connection* limit and no request limit — Caddy 2.11 ships no rate limiter — so
240 connections a minute per source is 240 password guesses a minute per source, against
a maintenance account whose handle a stranger can guess. `SignIns` below is the limit
that route actually has.

The designer's *show me* and *show the stuck game* are the exception. Each press plays
dozens of games on the server, so a visitor is refused both and asked to sign in. Counting a visitor would need a cookie or an
address the privacy notice does not allow.
"""

from __future__ import annotations

import ipaddress
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import timedelta
from typing import Any

# ------------------------------------------------------------------- §4, the sweep

IDLE_LIMIT = timedelta(days=30)
"""How long a leaderboard match may sit without a move before the sweep resigns it.

The user, 2026-09-11, made this deadline generous rather than tight. The sweep exists to
close the loophole of never finishing a losing game to keep a rating up, not to punish a
crash or a bad connection, so idle is measured in weeks. A move a fortnight ago keeps a
game alive. `matches.sweep` takes it as an argument, so moving it is a call site.
"""

IDLE_SAID = "thirty days without a move"
"""`IDLE_LIMIT` as a visitor reads it, in every sentence about the sweep
(`docs/plan-understanding.md` §3.2: *resigned after thirty days without a move*).
`tests/test_badges.py` holds the words to the limit, so moving one without the other fails."""

# ------------------------------------------------------- correspondence, between people

CORRESPONDENCE_IDLE = timedelta(days=5)
"""How long the player to move in a game between two people may sit before losing it.

The user, 2026-10-02, accepting `docs/plan-return.md` §1 with one change: *"let's cut the
no moves to resign to 5 days"*. A game between people is the one place one person's
absence holds another hostage, so the sweep resigns it for the player to move, and the row
records it as a game they left (§1.6). Shorter than `IDLE_LIMIT` on purpose: that one is
about a person and a computer, who waits for nobody. Nothing counts it down on any page
(test 2); the rule is said, never the time left.
"""

CORRESPONDENCE_IDLE_SAID = "five days without a move"
"""`CORRESPONDENCE_IDLE` as a player reads it, in the rule under a correspondence game's
moves. `tests/test_site_correspondence.py` holds the words to the limit."""

CORRESPONDENCE_AT_ONCE = 30
"""How many games between people one person may have running at once (§1.3).

It bounds the rows of *Games waiting on your move*, not anybody's play: ten people with a
few games each are nowhere near it. A two-person move is one legality check and a row
write, so the box is not what it protects.
"""

CHALLENGES_A_DAY = 20
"""How many challenges one person may send in a rolling day (§1.3), links included.
Enforced by `site/challenges.py` (Q571) under the challenger's lock."""

CHALLENGES_WAITING = 30
"""How many of one person's challenges may wait for an answer at once (Q571).

A challenge never expires (test 2), so without this a person could leave any number
standing; thirty is the games-at-once cap's number, so a person can always have as many
challenges out as games they could play. Withdrawing one on their page frees a place.
"""

MOVE_EMAILS_A_DAY = 80
"""How many move emails the whole site may send in a rolling day (Q572, §1.5).

The service's free tier, which also carries the invitations, is about a hundred a day
(est.). Past the cap the page is the only signal until the day rolls on, and nothing is
queued to arrive later in a burst: a move that would have sent one sends none. Ten people
with five games each and a move a day is about fifty (est.). Shown on `/admin` beside how
many went in the last day.
"""

# --------------------------------------------------------------- enforced at a start

MATCHES_A_DAY = 60
"""How many matches one person may start in a rolling day, both modes together.

§6's answer to spam and vandalism is *a per-user daily quota on every write*, and a match
is the only write phase 1 has that costs CPU. Sixty is far above a day of playing -- a
leaderboard game runs for minutes and there are only so many in an evening -- and far
below what a script would do with an account. Rolling rather than calendar, so the limit
does not reset at a time somebody can wait for.
"""

STRONG_NODES = 200_000
"""The node budget at which an opponent counts as the strongest one offered.

§2's number. An agent at or above it is what the concurrency cap is about; everything
below is cheap enough to ignore -- thirty games against `nodes=6400` cost 0.06 of a core.
"""

VISITOR_NODES = 6_400
"""The largest node budget a visitor with no account may play against: *hard*'s.

`docs/plan-launch.md` §3.6, decided A on 2026-10-01 (L15): with the shared password off
the front door, visitors play opponents up to *hard*, and *hardest* is for signed-in
people. Read off the budget rather than the ladder's names, so every player stronger than
*hard* -- *hardest*, the outside engine, anything a store adds above it -- needs an
account by the same rule, and the strong seats `STRONG_SEATS` rations are all among them.
"""

STRONG_SEATS = 8
"""How many games against an agent at `STRONG_NODES` or above may be live at once.

Site-wide, not per person: the cap exists because the box has two cores, and eight
players holding one seat each costs the same as one player holding eight. At §2's pace of
a move every ten seconds this is about 0.44 of a core, and even if all eight moved in the
same second it is under five CPU-seconds. §2's *"the number of `nodes=200000` seats at
once is capped"*, with a number put to it.
"""

SEAT_ACTIVE = timedelta(minutes=10)
"""How recently a match must have moved to still be holding a strong seat.

Without this the cap would be a cap on matches ever started and left open, since an
abandoned sandbox row never finishes and the sweep only closes leaderboard ones. What
costs the box is somebody moving, so what is counted is somebody moving. A game that sits
this long gives its seat up, and taking it back is checked like a start (Q410,
`check_resumed_match`).
"""

SUGGESTIONS_A_DAY = 10
"""How many suggestions one person may post in a rolling day.

Q279. §6's answer to spam is *a per-user daily quota on every write*, and the suggestion
box is a write. Ten is more ideas than anybody has in a day and few enough that a script
with an account fills no table. Rolling, like `MATCHES_A_DAY`, so there is no reset time
to wait for. A suggestion costs the box a row and no CPU, so the number is about the
maintainer's reading rather than about the machine.
"""

SUGGESTION_LENGTH = 2000
"""How long one suggestion may be, in characters.

Long enough for a rule described in a paragraph or two and short enough that the
maintainer's page stays a list. The column carries the same cap as a check constraint
(`0011_suggestions.sql`), so the limit is true of the table and not only of the one
writer that goes through `site/suggestions.py`.
"""

REPORTS_A_DAY = 10
"""How many games one person may report in a rolling day. Q503d, `docs/plan-game-pipeline.md` §5.

§6's answer to spam is *a per-user daily quota on every write*, and a report is one. The
other limit, one report a person a game, is the table's primary key (`0017_reports.sql`).
Ten is more than anybody reports honestly in a day and few enough that one account cannot
bury the maintainer's page. Rolling, like `SUGGESTIONS_A_DAY`.
"""

REPORT_NOTE_LENGTH = 500
"""How long the optional note on a report may be, in characters. Room for "a copy of X"
and a sentence of why; the column carries the same cap (`0017_reports.sql`)."""

# ------------------------------------------- Q831, a visitor asks for an invitation

ASKS_A_DAY = 20
"""How many requests for an invitation the whole site keeps in a rolling day.

Q831 (`site/asking.py`). The one write on the site a stranger makes, so the cap is the
site's rather than a person's: nobody is signed in to count against. Twenty is
`docs/plan-launch.md` §2 S5's sign-up cap, a day's worth for a small group and few enough
that a script fills `/admin` with a page, not a flood. Past it the visitor is told the
requests are full today; the sentence names no address, so it says nothing about anybody.
Counted over the rows written, so a repeat that wrote nothing does not count.
"""

ASKS_A_SOURCE = 3
"""How many requests one network address may send in a rolling day, written or not.

The other half of the daily cap: without it one source could spend the whole site's
twenty and leave nobody else a way in. Three covers a typo and a second thought. In memory,
like `SignIns`, so the privacy notice's list does not grow by a log of addresses, and a
restart forgets it.
"""

ASK_EMAIL_LENGTH = 254
"""The longest address a request may carry: the longest a mail server delivers to. The
column carries the same cap (`0034_invite_asks.sql`)."""

ASK_NOTE_LENGTH = 500
"""How long the optional note on a request may be, in characters: room to say how they
found the site and a line about themselves. The column carries the same cap."""

ASK_KEPT = timedelta(days=30)
"""How long an answered request is kept after its answer, then deleted.

While it is kept, the same address asking again writes nothing (one row an address, the
table's unique index), which is the cap per address. After it, the address may ask again,
and nothing about a stranger is kept longer than that. A pending request is kept until the
maintainer answers it."""

ASKS_TRACKED = 20_000
"""How many network addresses `Asks` holds counts for, the ceiling `SIGNIN_TRACKED` is."""


class Asks:
    """Requests for an invitation in the last rolling day, by network address (Q831).

    Every request counts, the ones that wrote a row and the ones that did not, so asking
    about many addresses from one place is three tries a day whatever the answers were.
    The count is taken before the address or the note is read, so a refused request costs
    no query. In memory with one lock, like `SignIns`.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, deque[float]] = {}

    def take(self, source: str) -> None:
        """Count one request from this address, or refuse it with the sentence the page shows."""
        window = timedelta(days=1).total_seconds()
        with self._lock:
            now = self._clock()
            times = self._seen.setdefault(source, deque())
            while times and times[0] <= now - window:
                times.popleft()
            if len(times) >= ASKS_A_SOURCE:
                raise Spent(ASKED_ENOUGH)
            times.append(now)
            for key in [key for key, kept in self._seen.items() if not kept]:
                del self._seen[key]
            while len(self._seen) > ASKS_TRACKED:
                oldest = min(self._seen, key=lambda key: self._seen[key][-1])
                del self._seen[oldest]


ASKED_ENOUGH = (
    "That is as many requests as one connection can send in a day. If you have sent one, "
    "it has arrived; there is nothing more to do."
)
"""The refusal past `ASKS_A_SOURCE`. It names no address, so it says nothing about any."""

ASKS_FULL = (
    "The site has had as many requests for an invitation as it takes today. Please try "
    "again tomorrow."
)
"""The refusal past `ASKS_A_DAY`, the same for everybody."""

# ------------------------------------------------- phase 2's queue, stated not enforced

CANONICAL_A_DAY = 1
"""Canonical measurements one person may queue in a rolling day.

§2 costed exactly this: thirty people with one a day each is 277 CPU-seconds apiece, so
2.3 CPU-hours out of the box's 48 -- *"comfortable, queued"*.
"""

PICKER_A_WEEK = 1
"""Picker ratings one person may queue in a rolling week.

§2 names this as *"the rung that needs the tight quota"*, at 1838 CPU-seconds each, and
says thirty a day would be a third of the machine; it puts no number to the allowance.
One a week per person is thirty runs a week, which is 15 CPU-hours of the 336 the box has
in a week. Chosen conservative on purpose, and raised by editing this line once phase 2's
queue has a week of real depth behind it.
"""

JOBS_IN_FLIGHT = 1
"""Queued jobs one person may have running at once. §6's *"one job per user in flight"*."""

# ----------------------------------------- Q503e, the checks of a browser's measurement
#
# `docs/reference/tier-costs-on-the-box.md`: Q309's caps, read again on the box on
# 2026-10-04 and unchanged. Priced in CPU-seconds of the runner thread, because one check of
# the library's largest board costs what ten of chess do; `site/spotcheck.py` reads them.

SAMPLE_JOB_CPU = 300.0
"""The most processor time one spot check may spend, in CPU-seconds of its thread.

A whole tier one and two on the library's largest board is at most about 300 (14.9 +
200.9 at the costliest sample + 83.8 for the acceptance sample); on chess at most 86. A
check past it stops where it is and the result stays `legal`, waiting for the server's
own measurement: running out of budget is never a mismatch.
"""

SAMPLE_JOB_WALL = 600.0
"""The most wall-clock seconds one spot check may take: ten minutes, at `nice 19` on the
site's one runner thread, so the box's other core stays the site's."""

CHECK_CPU_A_DAY = 1500.0
"""The checking one person's submissions may cost the box in a rolling day, in CPU-seconds.

Tier one's walk and tier two's replay alike, summed off `jobs.cpu_seconds`. The plan's
*ten a person a day* priced at the large board (ten times about 150), and about thirty
chess-sized checks. Asked when a submission is filed; one already filed runs to its own cap.
"""

CHECK_WINDOW = timedelta(days=1)
"""The window `CHECK_CPU_A_DAY` counts over."""

STALE_GRACE_DAYS = 30
"""How long a trial game's reading may be out of date and still count toward the shortlist.

`docs/plan-game-pipeline.md` §2.4: only the designer's browser repairs a trial game's
reading, so a core rebuild or a new weights version gives every designer this long to
measure again before the game stops counting. Timed from `provisional.stale_since`.
"""

# ------------------------------------------------------ enforced at a designer press

SHOW_ME_AN_HOUR = 50
"""How many times one person may press *show me* or *show the stuck game* in a rolling hour.

The user, 2026-09-13: *"give it 50 presses per user per hour"*. A *show me* press plays 32
core games, 0.3-0.6 CPU-seconds, so fifty is at most half a minute of one core an hour per
person. A *show the stuck game* press plays 16 random games and draws on the same fifty.
Rolling, like `MATCHES_A_DAY`, so there is no reset time to wait for.
"""

SHOW_ME_WINDOW = timedelta(hours=1)
"""The window `SHOW_ME_AN_HOUR` counts over."""

# --------------------------------------------- Q503a, enforced at a save and a publish

DESIGN_BODY_BYTES = 64 * 1024
"""The largest body a `POST /api/design/...` may carry, in bytes.

`docs/plan-game-pipeline.md` §2.2, closing the spec-size cap `routes.py` has asked for
since task 1.7. Chess as the designer sends it is about 6 KB, so 64 KB is ten of the
largest game anybody has built and still refuses a body sent to fill the box's memory.
The site's handler answers 413 before the body is read or anybody is identified.
"""

DRAFTS_A_PERSON = 200
"""How many designed games one person may keep on the site, deleted ones not counted.

§2.2. A draft is a `games` row and a directory under `runs/designed/`, so this bounds
what one account can make the box hold. Saving rules the store already has adds no row
and is never refused by it.
"""

SAVES_A_DAY = 30
"""How many saves one person may make in a rolling day.

§2.2. A save runs the playability check, up to `screening.CHECK_SECONDS` of one core, so
thirty is at most seven and a half core-minutes a person a day. Counted in memory like
`Presses`, and for its reason: a log of saves is not one of the things the privacy notice
says the database keeps. A restart hands a person a fresh day, which is a deploy's worth
of saves and nothing a script can ask for.
"""

SAVES_WINDOW = timedelta(days=1)
"""The window `SAVES_A_DAY` counts over."""

PUBLISHED_AT_ONCE = 5
"""How many games one person may have on trial at once. §2.2, the user's answer Q2's plan.

Counted off the store: designed rows this person owns that are public and not killed,
retired, deleted or resting. A game that leaves the listing frees its place.
"""

PUBLISHES_A_DAY = 1
"""How many games one person may publish in a rolling day, counted off `games.published`.

§2.2. Publishing will have the box check a reading (Q503e), 50-85 CPU-seconds, so one a
day a person is at most 43 CPU-minutes a day for the whole invited group.
"""

# ------------------------------------------ Q503f, the shortlist's bar and its two caps
#
# `docs/plan-game-pipeline.md` §2.2, at the user's answer Q2 of 2026-09-28 (B, the
# stricter of the three offered). `site/shortlist.py` reads them; raising the bar later is
# an edit here and nowhere else.

SHORTLIST_DAYS = 30
"""How long a game must have been on trial, from `games.published`, before it can be
nominated."""

SHORTLIST_FINISHERS = 8
"""How many people other than the designer must have finished a game on it."""

SHORTLIST_VOTES = 8
"""How many qualifying star votes it needs (`shortlist.qualifying`)."""

SHORTLIST_STARS = 4.0
"""The mean those qualifying votes must reach, or better."""

VOTER_ACCOUNT_AGE = timedelta(days=7)
"""How old a voter's account must be for their stars to count toward a shortlist, so a
fresh invitation cannot be spent on a friend's game the same day."""

SHORTLIST_SIZE = 8
"""The most games shortlisted at once. A nomination past it is refused until the round
takes one off; the order is the order designers press, which is first eligible first
among designers who press as soon as they may."""

SHORTLISTED_A_DESIGNER = 1
"""How many of one designer's games may be shortlisted at once."""

# ------------------------------------------------- Q503i, the trial's quiet way out

RESTING_AFTER = timedelta(days=90)
"""How long a game must have been on trial, from `games.published`, before it can rest.

`docs/plan-game-pipeline.md` §2.2's last row: *"90 days on trial with no finished game by
another person in the last 60"*. `resting.rest` reads it, and nothing else does.
"""

RESTING_QUIET = timedelta(days=60)
"""How long nobody but its designer may have gone without finishing a game on it."""

WAKES_A_GAME = 1
"""How many times a resting game's designer may publish it again (§2.2: *"once"*)."""

# ------------------------------------------------- Q503h, the round

ROUND_VOTING = timedelta(days=7)
"""How long members vote at a round, from the 1st of the month (`site/rounds.py`).

`docs/plan-game-pipeline.md` §2.2 *The round, in order*, step 2 (answer Q1 B: *"members
vote for a week"*). The round closes this long after the 1st, whenever it opened."""

ROUND_YES = 4
"""The fewest yes votes a nominee passes with; it also needs more yes than no.

§6's first *needs decision*, settled on its default (option A, 2026-09-28): *"a pass
needs more yes than no and at least 4 yes"*. Who may vote is the shortlist's own
qualifying voter: finished a game on it, not its designer, an account
`VOTER_ACCOUNT_AGE` old that nobody shut."""

ROUND_PLACES = 4
"""The most games the maintainer confirms at one round (answer Q4 A: *"up to 4 a
month"*). A passed and unvetoed game past them is carried to the next round, vote kept."""

DECLINED_WAIT = timedelta(days=60)
"""How long a game voted down or vetoed waits before its designer may nominate it again
(§2.2 step 6)."""

# ------------------------------------------------- Q369, enforced at a failed sign-in

SIGNIN_WINDOW = timedelta(minutes=15)
"""The rolling window every sign-in limit below counts over.

Short enough that somebody who mistyped their password five times is let back in while
they are still at the keyboard, and long enough that waiting it out costs a guessing run
almost all of its rate.
"""

SIGNIN_TRIES_A_SOURCE = 10
"""Failed attempts one address may make in `SIGNIN_WINDOW`, over every credential route.

Ten covers mistyping a password, forgetting which handle is this site's, and trying an
old one. A guessing run gets 40 an hour instead of the firewall's 14400, which is the
whole of the change: at 40 an hour a ten-character password is not reached.
"""

SIGNIN_TRIES_AN_ACCOUNT = 30
"""Failed attempts against one handle in `SIGNIN_WINDOW`, whatever address they come from.

Only bites above three addresses, so an ordinary person never meets it and a distributed
run does. Deliberately three times the per-address number rather than equal to it: an
account limit is also a stranger's way to shut the owner out for a quarter of an hour, so
it is set where it costs an attacker their spread and costs the owner as little as a
limit of this shape can.
"""

SIGNIN_TRACKED = 20_000
"""How many addresses and handles `SignIns` will hold counts for at once.

A limiter keyed by what a stranger sends is a table a stranger can grow, so it has a
ceiling. Above it the entry whose last failure is oldest is dropped, which hands that
address a fresh window and never refuses anybody.
"""


class SignIns:
    """Failed sign-ins in the last rolling window, by source address and by handle.

    Q369. The site's only limit on guessing a password: Caddy 2.11 has no request
    limiter, so before this a source could spend the firewall's 240 connections a minute
    on `/api/signin`.

    Three properties the route needs and this class is the whole of.

    **A success is never slowed and never queued.** The check is a dictionary lookup
    under a lock, ahead of the 50 ms scrypt, and it clears both counts on the way out.
    Nothing sleeps: the server answers a request per thread, so a delay would spend a
    thread rather than an attacker's time.

    **The answer says nothing about whether an account exists.** Only failures are
    counted and a handle nobody has fails every time, so a made-up handle and a real one
    trip at the same attempt with the same sentence.

    **One address cannot spend another's allowance.** The address counts are separate
    keys; only the far looser per-handle count is shared, and that is the trade §7's
    costing writes down.

    In memory, like `Presses`, and for the same reason: the privacy notice lists what the
    database keeps and a log of failed attempts is not one of them. A restart forgets the
    counts, which is a window an attacker cannot ask for.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: dict[tuple[str, str], deque[float]] = {}

    def check(self, source: str, account: str | None = None) -> None:
        """Whether this attempt may be made. Silence, or the refusal the person reads.

        Called before the password or the code is looked at, so a refused attempt costs
        no hash. `account` is the handle being claimed, or `None` where the credential
        names none -- an invitation code is a secret on its own and is held by address.
        """
        window = SIGNIN_WINDOW.total_seconds()
        with self._lock:
            now = self._clock()
            for key, limit in self._keys(source, account):
                failed = self._recent(key, now, window)
                if len(failed) >= limit:
                    raise Spent(_wait_sentence(failed[0] + window - now))

    def failed(self, source: str, account: str | None = None) -> None:
        """Count one refused attempt against this address and this handle."""
        window = SIGNIN_WINDOW.total_seconds()
        with self._lock:
            now = self._clock()
            for key, _ in self._keys(source, account):
                self._recent(key, now, window).append(now)
            self._forget(now, window)

    def passed(self, source: str, account: str | None = None) -> None:
        """Drop what this address and this handle had failed. A success is a clean slate.

        Somebody who mistyped four times and then got it right starts their next bad
        evening from zero, and an attacker gains nothing: the credential they cleared it
        with was the right one.
        """
        with self._lock:
            for key, _ in self._keys(source, account):
                self._failures.pop(key, None)

    def _keys(self, source: str, account: str | None) -> list[tuple[tuple[str, str], int]]:
        """The counters this attempt touches, each with the limit it is held to."""
        keys = [(("source", source), SIGNIN_TRIES_A_SOURCE)]
        named = (account or "").strip().lower()
        if named:
            keys.append((("account", named), SIGNIN_TRIES_AN_ACCOUNT))
        return keys

    def _recent(self, key: tuple[str, str], now: float, window: float) -> deque[float]:
        """This counter with everything older than the window dropped off the front."""
        failed = self._failures.setdefault(key, deque())
        while failed and failed[0] <= now - window:
            failed.popleft()
        return failed

    def _forget(self, now: float, window: float) -> None:
        """Keep the table the size of the window, and under `SIGNIN_TRACKED` either way."""
        for key in [key for key, times in self._failures.items() if not times]:
            del self._failures[key]
        while len(self._failures) > SIGNIN_TRACKED:
            oldest = min(self._failures, key=lambda key: self._failures[key][-1])
            del self._failures[oldest]


def _wait_sentence(seconds: float) -> str:
    """The 429's whole body: what happened, and when to try again.

    One sentence for every credential route and both counters, because which limit was
    reached is information for whoever is guessing.
    """
    minutes = max(1, math.ceil(seconds / 60))
    return (
        f"Too many failed sign-in attempts. Wait {minutes} "
        f"minute{'' if minutes == 1 else 's'} and try again, or ask whoever invited you "
        f"for a new invitation if you have lost the password."
    )


def source_address(peer: str, forwarded: str | None) -> str:
    """Which address an attempt is counted against, given the socket's peer and the header.

    The app listens on loopback behind Caddy, so the peer is the proxy and the client is
    in `X-Forwarded-For`. Caddy **appends** the connecting address to whatever the client
    sent, so the trustworthy entry is the **last** one and everything before it is the
    client's own writing. Taking the first would let a stranger spend a different
    allowance on every attempt by sending a different address.

    A peer that is not loopback is not the proxy, so its header is read from nobody and
    the peer itself is the answer.
    """
    if not forwarded or not _is_loopback(peer):
        return peer
    entries = [part.strip() for part in forwarded.split(",") if part.strip()]
    if not entries:
        return peer
    return entries[-1][:64]


def _is_loopback(peer: str) -> bool:
    """Whether this peer is the front server on the same box, and so worth listening to."""
    try:
        return ipaddress.ip_address(peer.strip("[]")).is_loopback
    except ValueError:
        return False


class QuotaError(ValueError):
    """A limit was reached, said the way the person who reached it needs to hear it.

    A `ValueError` like `matches.MatchError`, so the server answers 400 with the sentence
    and the page prints it where the click was. Every message here says what the limit is
    and what to do next, because a refusal a player cannot act on reads as a fault.
    """


class Spent(QuotaError):
    """A person has used up an hourly allowance. The site answers 429 with the sentence."""


class Presses:
    """Each person's designer presses in the last rolling hour, in memory.

    *Show me* and *show the stuck game* both count here, against one allowance.

    Nothing is written to the store. The privacy notice lists the four things the database
    keeps, and a log of presses is not one of them. A restart forgets every count, which
    at worst hands a person a fresh hour. The server answers each request on its own
    thread, so one lock guards the counts.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._pressed: dict[int, deque[float]] = {}

    def press(self, user: int | None) -> None:
        """Count one press, or refuse it with the sentence the page shows."""
        if user is None:
            raise QuotaError(
                "Sign in to use show me. It plays dozens of games on the server for "
                "each press, so the site counts presses by account."
            )
        window = SHOW_ME_WINDOW.total_seconds()
        with self._lock:
            now = self._clock()
            pressed = self._pressed.setdefault(user, deque())
            while pressed and pressed[0] <= now - window:
                pressed.popleft()
            if len(pressed) >= SHOW_ME_AN_HOUR:
                minutes = max(1, math.ceil((pressed[0] + window - now) / 60))
                raise Spent(
                    f"Between show me and show the stuck game you have pressed "
                    f"{SHOW_ME_AN_HOUR} times in the last hour, "
                    f"which is the limit. Your next press is available in {minutes} "
                    f"minute{'' if minutes == 1 else 's'}."
                )
            pressed.append(now)
            # Forget people whose hour has emptied, so the table stays the size of the
            # last hour's players.
            for other in [key for key, times in self._pressed.items() if not times]:
                del self._pressed[other]

    def counts(self) -> list[int]:
        """How many presses each person made in the last rolling hour, naming nobody.

        `site/usage.py` reads *show me*'s distinct people, buckets and rolling peak here.
        """
        window = SHOW_ME_WINDOW.total_seconds()
        with self._lock:
            now = self._clock()
            for pressed in self._pressed.values():
                while pressed and pressed[0] <= now - window:
                    pressed.popleft()
            for other in [key for key, times in self._pressed.items() if not times]:
                del self._pressed[other]
            return [len(times) for times in self._pressed.values()]


class Saves:
    """Each person's saves in the last rolling day, in memory. `SAVES_A_DAY` is the cap.

    The same shape as `Presses`, over a day. A visitor never reaches it: saving needs a
    signed-in person (`routes.py`), so the id is always there.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._saved: dict[int, deque[float]] = {}

    def save(self, user: int) -> None:
        """Count one save, or refuse it with the sentence the designer shows."""
        window = SAVES_WINDOW.total_seconds()
        with self._lock:
            now = self._clock()
            saved = self._saved.setdefault(user, deque())
            while saved and saved[0] <= now - window:
                saved.popleft()
            if len(saved) >= SAVES_A_DAY:
                raise Spent(
                    f"You have saved {SAVES_A_DAY} times in the last day, which is the "
                    f"limit, because every save runs the playability check on the "
                    f"server. The count rolls: your next save is available in "
                    f"{_in_words(saved[0] + window - now)}. Playing a draft without "
                    f"saving it is not counted."
                )
            saved.append(now)
            for other in [key for key, times in self._saved.items() if not times]:
                del self._saved[other]


def nodes_in(agent: str) -> int:
    """The node budget in an agent string, or 0 for one that carries none.

    Budget is part of an agent's identity (`CLAUDE.md`), so it is in the string --
    `minimax:nodes=200000`. Nothing here parses the rest of the name: what the cap is
    about is the search, and the search is the number.
    """
    found = re.search(r"nodes=(\d+)", agent)
    return 0 if found is None else int(found.group(1))


def is_strong(agent: str | None) -> bool:
    """Whether seating this agent takes one of the strong seats."""
    return agent is not None and nodes_in(agent) >= STRONG_NODES


def members_only(agent: str | None) -> bool:
    """Whether only a signed-in person may play this computer player (`VISITOR_NODES`)."""
    return agent is not None and nodes_in(agent) > VISITOR_NODES


WHOLE_SITE = 0
"""The strong-seat cap's lock key, beside each person's own id for the daily allowance.

No person has id 0 -- `users.id` is an identity and starts at 1 -- so the site-wide key
and every person's key are different locks.
"""


def _hold(connection: Any, key: int) -> None:
    """Take one limit's lock, held until the caller's transaction ends (Q408).

    `check_new_match` reads a count and `matches.start` writes the row after it. Without
    this, two requests one short of a limit each read the count before the other's row
    existed, each found the one place free, and both were seated. With it, the second
    waits here until the first has committed or rolled back, and then reads a count with
    the first's row in it.

    A transaction-scoped advisory lock, because it fits the store as it is. A counter row
    to `SELECT ... FOR UPDATE` would be a new table; a conditional insert still reads the
    count from a snapshot that cannot see the other request's uncommitted row, so both
    would pass it; `SERIALIZABLE` would need a retry at every caller. The lock needs no
    migration and is released by the commit or rollback the caller already owns, so a
    refusal or an exception leaves nothing held.

    Keyed on this store's `matches` table and then on the limit: `(matches, person)` for
    the daily allowance and `(matches, WHOLE_SITE)` for the strong-seat cap. The table's
    oid is resolved through the search path, so two stores in one database -- the test
    sessions do this, a schema each -- never queue behind each other.

    Three things the lock relies on, each true of the site as it is:

    - **A statement of its own, ahead of the count.** Under `READ COMMITTED` a statement
      reads a snapshot taken when it began, so a count in the same statement as the lock
      would still be read from before the wait.
    - **A transaction.** An autocommit connection would release the lock at the end of
      this statement, before the row is written, and the race would be back with nothing
      to show for it; so that is refused rather than run.
    - **Nothing written before it.** `start` reads the game and then checks, so a start
      waiting here holds no row lock that the one it waits for could need. The person's
      lock is always taken before the site's, so two starts cannot wait on each other. A
      game taking its seat back (`check_resumed_match`) takes the site's lock alone, and
      also before it writes, so it waits on nobody's row either.

    What it costs a start: one round trip per limit the start meets -- one for a person's
    game against a cheaper opponent, none for a visitor's, two at most. A start waits only
    on the same person's other start, or on another start that seats the strongest
    opponent, and then for the rest of that request, which is milliseconds.
    """
    if getattr(connection, "autocommit", False):
        raise RuntimeError(
            "the match quota holds its lock until the caller commits, so it needs a "
            "connection inside a transaction, not an autocommit one"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "select pg_advisory_xact_lock('matches'::regclass::oid::int, %s::int)", (key,)
        )


def matches_today(connection: Any, user: int) -> int:
    """How many matches this person has started in the last rolling day."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from matches"
            " where (seat0_user = %s or seat1_user = %s) and started > now() - %s",
            (user, user, timedelta(days=1)),
        )
        return int(cursor.fetchone()[0])


_LIVE = "finished is null and coalesce(moved, started) > now() - %s"
"""A row still holding its seats: unfinished, and moved inside the window it is given.

The one spelling of *live*, read by the count and by a game asking whether it still holds
its own seat, so the two can never disagree about which rows are in the count.
"""

_MARGIN = timedelta(minutes=1)
"""How long before `SEAT_ACTIVE` runs out a game's move stops being waved past the lock.

A move waved through is counted as live by any start that counts before it commits, and
it has moved once it commits, so its seat is never out of the count -- provided the
request finishes inside this margin. A strongest-opponent reply costs about half a second
on the box (§2), so a minute is two orders of magnitude of room. It is not a limit anybody
meets, so `summary` does not print it.
"""


def strong_seats(connection: Any) -> int:
    """How many strong seats are live right now, across the whole site.

    Live is *unfinished and moved inside `SEAT_ACTIVE`*. The agent strings come back and
    are read here rather than matched in SQL, because `nodes_in` is the one reading of a
    budget and a regexp in a query would be a second one.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select seat0_agent, seat1_agent from matches where {_LIVE}", (SEAT_ACTIVE,)
        )
        rows = cursor.fetchall()
    return sum(1 for row in rows for agent in row if is_strong(agent))


def busy(connection: Any) -> bool:
    """Whether a new game against the strongest opponent would be refused right now.

    What the opponent list says *before* anybody picks it (`docs/plan-launch.md` §5.2,
    L11), so the refusal in `check_new_match` is never the first a person hears of it. A
    reading without the lock, so it can be a moment out of date either way; the start
    still counts under the lock, and is the answer that binds.
    """
    return strong_seats(connection) >= STRONG_SEATS


BUSY_SAID = "busy, back in a few minutes"
"""What the opponent list puts after a strong player's name while `busy` holds."""


def _holds_seats(connection: Any, match_id: str, within: timedelta) -> bool:
    """Whether this match is live, read as `strong_seats` reads it, over `within`."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select exists (select 1 from matches where id = %s and {_LIVE})",
            (match_id, within),
        )
        return bool(cursor.fetchone()[0])


def check_new_match(connection: Any, user: int | None, *agents: str) -> None:
    """Whether this person may start this match now. Silence, or the reason they may not.

    Called by `matches.start` before the row is written, so a refused match leaves
    nothing behind. A visitor -- `user` is None -- skips the daily count and still meets
    the strong-seat cap, which is site-wide and about the box rather than about a person.

    `agents` is every computer player the match would seat, all in this one call: one
    for a game somebody plays, two for a watch. The strong seats among them are counted
    together against what is free. Asked one seat at a time, a watch of two strong players
    put the same remaining allowance to both of its seats, so one seat short of the cap
    each found a seat free and both were seated (Q301).

    Each limit is read under its own lock, which is held until the caller commits the row
    or rolls it back, so two starts arriving together are counted one after the other
    rather than both against the same place (Q408, `_hold`). The caller has to write the
    row in the same transaction as this call; `matches.start` does.
    """
    if user is not None:
        _hold(connection, user)
        started = matches_today(connection, user)
        if started >= MATCHES_A_DAY:
            raise QuotaError(
                f"you have started {started} games in the last day, which is the limit of "
                f"{MATCHES_A_DAY}; the count rolls, so the next one opens up as the oldest "
                f"falls out of the day"
            )
    strong = [agent for agent in agents if is_strong(agent)]
    if not strong:
        return
    _hold(connection, WHOLE_SITE)
    free = STRONG_SEATS - strong_seats(connection)
    if len(strong) <= free:
        return
    if len(strong) == 1:
        # The opponent list already said *busy* (`busy`, L11); this is the same fact at
        # the press, short enough to read in two lines on a phone, the way out first.
        raise QuotaError(
            f"{_named(strong[0])} is busy: it plays {STRONG_SEATS} games at once and all "
            f"{STRONG_SEATS} are running; pick a weaker opponent, or come back in a few "
            f"minutes"
        )
    raise QuotaError(
        f"both players in this game are the strongest the site offers, and only "
        f"{STRONG_SEATS} places for them run at once; {_free_seats(free)}, so put a "
        f"weaker player on at least one side, or come back in a few minutes"
    )


def _named(agent: str) -> str:
    """What a refusal calls the strong player: its name off the ladder (*hardest*), never
    its agent string (`docs/plan-understanding.md` §3.3, S25)."""
    from chess_variant_sandbox.web.opponents import name_of

    return name_of(agent)


def _free_seats(free: int) -> str:
    """How many strong seats are free, in the words a refused watch reads."""
    if free <= 0:
        return "none is free"
    return "only one is free" if free == 1 else f"only {free} are free"


def check_resumed_match(connection: Any, match_id: str, *agents: str) -> None:
    """Whether this match may go on holding its strong seats. Silence, or why not (Q410).

    A game holds its seats while it is live -- unfinished, and moved inside `SEAT_ACTIVE`
    -- and one that sat past that, or ended, handed them back to the count. So the next
    move in it, a take-back that reopens it and a restart each seat the strongest
    opponent again, as surely as a start does, and each is checked against the same cap
    under the same lock. `matches` calls this ahead of any of those writes with the row's
    own agents, so a refused one leaves the game exactly as it was.

    **An active game is never re-checked.** Its seats are already in every count, so its
    moves go past without the lock; a lock on every move would queue each strong game's
    search behind every other's. Going past needs `_MARGIN` to spare, which keeps the
    game's seat in any count taken before this request commits. A game in its last
    minute takes the lock and is read again under it: still live, it goes on uncounted,
    and nothing can count while it holds the lock. The second reading is also what lets a
    double click through -- two requests on one idle game, the second of which waited
    here while the first took the seat back and so finds it held.

    **The daily allowance is not charged.** Carrying on is the one match that was counted
    when it started, so nothing here reads `matches_today`; a person who has used the
    day's allowance can still finish a game they began inside it.
    """
    strong = [agent for agent in agents if is_strong(agent)]
    if not strong or _holds_seats(connection, match_id, SEAT_ACTIVE - _MARGIN):
        return
    _hold(connection, WHOLE_SITE)
    if _holds_seats(connection, match_id, SEAT_ACTIVE):
        return
    free = STRONG_SEATS - strong_seats(connection)
    if len(strong) <= free:
        return
    minutes = int(SEAT_ACTIVE.total_seconds() // 60)
    if len(strong) == 1:
        raise QuotaError(
            f"{_named(strong[0])} is busy: it plays {STRONG_SEATS} games at once and all "
            f"{STRONG_SEATS} are running, and this one gave its place up when it ended or "
            f"went {minutes} minutes without a move; it is kept as it stands, so come "
            f"back in a few minutes to carry on, or start a new game against a weaker "
            f"opponent"
        )
    raise QuotaError(
        f"both players in this game are the strongest the site offers, and only "
        f"{STRONG_SEATS} places for them run at once; this game gave its places up when it "
        f"went {minutes} minutes without a move or ended, and {_free_seats(free)} now; it "
        f"is kept as it stands, so come back in a few minutes to watch the rest, or start "
        f"a new one with a weaker player on at least one side"
    )


def suggestions_today(connection: Any, user: int) -> int:
    """How many suggestions this person has posted in the last rolling day."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from suggestions"
            " where author = %s and created > now() - %s",
            (user, timedelta(days=1)),
        )
        return int(cursor.fetchone()[0])


def check_new_suggestion(connection: Any, user: int) -> None:
    """Whether this person may post another suggestion now. Silence, or the reason not.

    Called by `suggestions.post` before the row is written and before the text itself is
    judged, so somebody who has spent the day's allowance is told that rather than being
    told their wording is fine.
    """
    posted = suggestions_today(connection, user)
    if posted >= SUGGESTIONS_A_DAY:
        raise QuotaError(
            f"you have posted {posted} suggestions in the last day, which is the limit of "
            f"{SUGGESTIONS_A_DAY}; the count rolls, so the next one opens up as the oldest "
            f"falls out of the day"
        )


def reports_today(connection: Any, user: int) -> int:
    """How many games this person has reported in the last rolling day."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from reports"
            " where reporter = %s and created > now() - %s",
            (user, timedelta(days=1)),
        )
        return int(cursor.fetchone()[0])


def check_new_report(connection: Any, user: int) -> None:
    """Whether this person may report another game now. Silence, or the reason not."""
    filed = reports_today(connection, user)
    if filed >= REPORTS_A_DAY:
        raise QuotaError(
            f"You have reported {filed} games in the last day, which is the limit of "
            f"{REPORTS_A_DAY}. The count rolls, so the next one opens up as the oldest "
            f"falls out of the day."
        )


def _in_words(seconds: float) -> str:
    """A wait as a person reads it: minutes under two hours, hours after that."""
    minutes = max(1, math.ceil(seconds / 60))
    if minutes < 120:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    return f"{math.ceil(minutes / 60)} hours"


def check_new_draft(connection: Any, user: int, digest: str = "") -> None:
    """Whether this person may keep another draft. Silence, or the reason not.

    `digest` is the draft's hash when the site could work it out before the save; rules
    the store already holds add no row, so they pass whatever the count.
    """
    if digest:
        with connection.cursor() as cursor:
            cursor.execute("select 1 from games where hash = %s", (digest,))
            if cursor.fetchone() is not None:
                return
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from games"
            " where owner = %s and origin = 'designed' and deleted is null",
            (user,),
        )
        kept = int(cursor.fetchone()[0])
    if kept >= DRAFTS_A_PERSON:
        raise QuotaError(
            f"You have {kept} games saved on the site, which is the limit of "
            f"{DRAFTS_A_PERSON}. Saving rules you have already saved still works, and "
            f"playing a draft without saving it needs no room at all."
        )


def on_trial(connection: Any, user: int) -> int:
    """How many of this person's games are on trial now: public, and still standing.

    A resting game (Q503i) has left the listing, so it frees its place.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from games where owner = %s and origin = 'designed'"
            " and visibility = 'public' and revoked is null and retired is null"
            " and deleted is null and rested is null",
            (user,),
        )
        return int(cursor.fetchone()[0])


def published_today(connection: Any, user: int) -> int:
    """How many games this person published in the last rolling day."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from games where owner = %s and published > now() - %s",
            (user, timedelta(days=1)),
        )
        return int(cursor.fetchone()[0])


def check_publish(connection: Any, user: int) -> None:
    """Whether this person may publish another game now. Silence, or the reason not.

    Both counts are read under the person's own lock on `games`, held until the caller
    commits, for `_hold`'s reason: two presses one short of a cap must not both pass.
    """
    if getattr(connection, "autocommit", False):
        raise RuntimeError("the publish quota holds its lock until the caller commits")
    with connection.cursor() as cursor:
        cursor.execute(
            "select pg_advisory_xact_lock('games'::regclass::oid::int, %s::int)", (user,)
        )
    held = on_trial(connection, user)
    if held >= PUBLISHED_AT_ONCE:
        raise QuotaError(
            f"You have {held} games on trial, which is the limit of {PUBLISHED_AT_ONCE}. "
            f"A place opens when one of them leaves the listing."
        )
    today = published_today(connection, user)
    if today >= PUBLISHES_A_DAY:
        raise QuotaError(
            f"You have published {today} game{'' if today == 1 else 's'} in the last "
            f"day, which is the limit of {PUBLISHES_A_DAY}. The count rolls, so the next "
            f"one opens a day after the last."
        )


def checked_today(connection: Any, user: int) -> float:
    """The CPU-seconds checking this person's submissions spent in the last rolling day."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select coalesce(sum(cpu_seconds), 0) from jobs where requester = %s"
            " and created > now() - %s and cpu_seconds is not null",
            (user, CHECK_WINDOW),
        )
        return float(cursor.fetchone()[0])


def check_checking(connection: Any, user: int) -> None:
    """Whether the site will check another score of this person's today, or the reason not.

    Asked when a submission is filed (`legality.file`); a check already filed runs to its
    own cap (`SAMPLE_JOB_CPU`), so the day can end a little past the line, never far.
    """
    spent = checked_today(connection, user)
    if spent >= CHECK_CPU_A_DAY:
        raise QuotaError(
            "Checking your scores has used the site's day for one person "
            f"({spent:,.0f} of {CHECK_CPU_A_DAY:,.0f} CPU-seconds). The count rolls, so "
            "send this one again tomorrow."
        )


def summary() -> tuple[tuple[str, str], ...]:
    """Every number here as a label and a value, for the admin page to print.

    The page reads this rather than naming the numbers itself, so there is still only one
    place a limit is written down.
    """
    return (
        ("Matches one person may start a day", str(MATCHES_A_DAY)),
        (f"Live seats against nodes={STRONG_NODES} or stronger", str(STRONG_SEATS)),
        ("A seat counts as live for", f"{int(SEAT_ACTIVE.total_seconds() // 60)} minutes"),
        ("A leaderboard match is resigned after", f"{IDLE_LIMIT.days} idle days"),
        (
            "A game between two people is lost by the player to move after",
            f"{CORRESPONDENCE_IDLE.days} idle days",
        ),
        ("Games between people one person may have running", str(CORRESPONDENCE_AT_ONCE)),
        ("Challenges one person may send a day", str(CHALLENGES_A_DAY)),
        ("Move emails the whole site may send a day", str(MOVE_EMAILS_A_DAY)),
        ("Challenges of one person's that may wait for an answer at once",
         str(CHALLENGES_WAITING)),
        ("Show me and stuck-game presses one person may make an hour", str(SHOW_ME_AN_HOUR)),
        ("Suggestions one person may post a day", str(SUGGESTIONS_A_DAY)),
        ("Failed sign-ins one address may make", str(SIGNIN_TRIES_A_SOURCE)),
        ("Failed sign-ins against one handle", str(SIGNIN_TRIES_AN_ACCOUNT)),
        (
            "The window both sign-in limits count over",
            f"{int(SIGNIN_WINDOW.total_seconds() // 60)} minutes",
        ),
        ("Characters one suggestion may be", str(SUGGESTION_LENGTH)),
        ("Games one person may report a day", str(REPORTS_A_DAY)),
        ("Characters one report's note may be", str(REPORT_NOTE_LENGTH)),
        ("Requests for an invitation the site keeps a day", str(ASKS_A_DAY)),
        ("Requests for an invitation one connection may send a day", str(ASKS_A_SOURCE)),
        ("Characters one request's note may be", str(ASK_NOTE_LENGTH)),
        ("An answered request for an invitation is deleted after", f"{ASK_KEPT.days} days"),
        ("Bytes one designer request may carry", str(DESIGN_BODY_BYTES)),
        ("Games one person may keep saved", str(DRAFTS_A_PERSON)),
        ("Saves one person may make a day", str(SAVES_A_DAY)),
        ("Games one person may have on trial at once", str(PUBLISHED_AT_ONCE)),
        ("Games one person may publish a day", str(PUBLISHES_A_DAY)),
        (
            "The shortlist's bar",
            f"{SHORTLIST_DAYS} days on trial, {SHORTLIST_FINISHERS} other finishers, "
            f"{SHORTLIST_VOTES} qualifying votes averaging {SHORTLIST_STARS:.1f} or more",
        ),
        ("A qualifying voter's account is at least", f"{VOTER_ACCOUNT_AGE.days} days old"),
        ("Games shortlisted at once", str(SHORTLIST_SIZE)),
        ("Games one designer may have shortlisted at once", str(SHORTLISTED_A_DESIGNER)),
        (
            "A trial game rests after",
            f"{RESTING_AFTER.days} days on trial with no game finished on it by anybody "
            f"else in the last {RESTING_QUIET.days}",
        ),
        ("Times its designer may publish a resting game again", str(WAKES_A_GAME)),
        (
            "A round",
            f"members vote for {ROUND_VOTING.days} days from the 1st; a game passes with "
            f"more yes than no and at least {ROUND_YES} yes",
        ),
        ("Games the maintainer may confirm at one round", str(ROUND_PLACES)),
        (
            "Before a game voted down or vetoed may be nominated again",
            f"{DECLINED_WAIT.days} days",
        ),
        ("Canonical measurements a person may queue a day", f"{CANONICAL_A_DAY} (phase 2)"),
        ("Picker ratings a person may queue a week", f"{PICKER_A_WEEK} (phase 2)"),
        ("Queued jobs one person may run at once", f"{JOBS_IN_FLIGHT} (phase 2)"),
        (
            "Checking one person's scores may cost the site a day",
            f"{CHECK_CPU_A_DAY:,.0f} CPU-seconds",
        ),
        (
            "One spot check of a score may take",
            f"{SAMPLE_JOB_CPU:,.0f} CPU-seconds or {SAMPLE_JOB_WALL / 60:.0f} minutes",
        ),
        (
            "A trial game's score may be out of date and still count toward the shortlist for",
            f"{STALE_GRACE_DAYS} days",
        ),
    )
