"""*Unbeaten*: the tag a game carries until a person beats its top computer player.

Task 1.5 of `docs/plan-website.md`, against §5's mechanic. The user, on why it exists:
*"I'm thinking in terms of game design -- a little something as a motivator to engage with
the games, keep the completionists playing and trying games."* So it is small on purpose.
One tag on a game, one record on a profile, three counts across the library, and no
points, badges or streaks beyond those.

## What clears it

A clear is a person winning a leaderboard match against the top, and everything narrow
about that sentence is load-bearing. Sandbox counts for nothing. A draw is half a point
and the user's word is *beaten*. A win over a computer player that was not the top is a
*won*, not a clear. The rows the loader brought in from the local application carry
`source = 'local-app'`, were played with undo on hand, and clear nothing either.

One win from either seat clears, and the seat is recorded; the user settled that on
2026-09-10, together with naming the first clear and counting the rest. The lever §5 names
and does not pull -- requiring a win from both seats -- is a rule change over the same
rows, not a migration.

## Why the row and not the table

The tag reads the match rows, never the picker's top rung today. A computer player rated
later can take the top, and the target then changes for every game not yet cleared -- but
a clear is decided at the time and stays. So the row carries what it was played against:
`top` says this match's own opponent was in the top group, and `top_players` says who the
whole group was. Both are written once, at the start.

`top_players` is the correction task 1.5 made to §5. *Clearing a tied game means beating
each of the tied*, and one boolean cannot say whether the opponent was alone at the top or
one of two. With the group on the row, a tie stays uncleared until one person has beaten
every member of it, and a clear against a lone top survives the top changing hands
afterwards. Migration `0002_top_group.sql` is that column; a row written before it has an
empty group, which reads as *this opponent alone*, which is what the bit meant then.

Nothing here commits and nothing here writes. Every reading is a fold over rows
`matches.leaderboard_matches` returned, in the finish order it returns them in, which is
the order a first clear is decided in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import identity, rating
from chess_variant_sandbox.site import matches as match_rows

UNTRIED = "untried"
TRIED = "tried"
WON = "won"
BEAT_THE_TOP = "top"
STATES = (UNTRIED, TRIED, WON, BEAT_THE_TOP)
"""§5's four states for one person on one game, weakest first. *Untried* is no finished
leaderboard match, *tried* is at least one of any result, *won* is a win over any computer
player -- `random` included, because the rung says which one -- and *beat the top* is a
clear."""


# ----------------------------------------------------------------------- what a row says


@dataclass(frozen=True)
class Beaten:
    """One computer player a person has beaten, as their record names it."""

    agent: str
    rating: float | None
    seat: int
    when: datetime | None
    match: str
    top: bool


@dataclass(frozen=True)
class Clear:
    """A person's *beat the top* on a game, seen from the game's side."""

    user: int
    handle: str
    when: datetime | None
    seat: int
    agent: str
    match: str


@dataclass(frozen=True)
class Tag:
    """What a game's card and page show.

    `unbeaten` is None on an unrated game -- one whose table has no computer players,
    because its picker ladder has never been rated -- since that game is not unbeaten, it
    is unrateable. §5: it offers sandbox only, carries no tag, and says so.

    `reached` is every clear in the order they were made, which the game page lists under
    the first (box B1 of `docs/archive/plan-meta-game.md`). The first clear and the count
    are readings of it, so the card and the page cannot disagree about who was first.
    """

    game: str
    rated: bool
    top: tuple[str, ...]
    top_rating: float | None
    unbeaten: bool | None
    reached: tuple[Clear, ...]

    @property
    def first(self) -> Clear | None:
        return self.reached[0] if self.reached else None

    @property
    def clears(self) -> int:
        return len(self.reached)

    @property
    def cleared(self) -> bool:
        return bool(self.reached)


@dataclass(frozen=True)
class Record:
    """One person's record on one game."""

    game: str
    state: str
    best: Beaten | None
    played: int
    won: int
    drawn: int
    lost: int

    @property
    def beat_the_top(self) -> bool:
        return self.state == BEAT_THE_TOP


@dataclass(frozen=True)
class Counts:
    """The three counts §5 puts on a profile, each as *n of N* over the rated games."""

    rated: int
    tried: int
    won: int
    top: int


def _rated_row(row: match_rows.Row) -> bool:
    """Whether this row is one the tag and the counts read at all.

    Finished, in leaderboard mode, with a person on one side and a computer player on the
    other, and not loaded from the local application. `leaderboard_matches` has already
    dropped sandbox rows and, by default, unfinished ones.
    """
    return (
        row.finished is not None
        and row.result is not None
        and row.source is None
        and row.is_computer(0) != row.is_computer(1)
    )


def _human_seat(row: match_rows.Row) -> int:
    return 1 if row.is_computer(0) else 0


def _group(row: match_rows.Row, agent: str) -> frozenset[str]:
    """The tied top this match was started against.

    Empty on a row written before `0002_top_group.sql`, which is read as *this opponent
    alone* -- the only thing the bit could have meant then.
    """
    return frozenset(row.top_players) if row.top_players else frozenset({agent})


# ------------------------------------------------------------------- a game's own tag


def tag_from(
    game: str,
    rows: Sequence[match_rows.Row],
    table: Mapping[rating.PlayerKey, rating.Player] | None,
    handles: Mapping[int, str] | None = None,
) -> Tag:
    """One game's tag, folded over its rows in finish order.

    Pure, so the mechanic can be checked without a database, and so the listing can fold
    sixty games out of one query. `table` is the game's live rating table, which is read
    only to *name* the target -- who is top now -- never to decide a clear.
    """
    named = handles or {}
    top = rating.top(table or {})
    rated = bool(top)
    reached: list[Clear] = []
    seen: set[int] = set()
    beaten: dict[int, set[str]] = {}
    for row in rows:
        if row.game != game or not _rated_row(row):
            continue
        seat = _human_seat(row)
        person = row.player(seat).user
        if person is None or person in seen or not row.top:
            continue
        if row.score_for(seat) != 1.0:
            continue
        agent = str(row.player(1 - seat).agent)
        won = beaten.setdefault(person, set())
        won.add(agent)
        if _group(row, agent) <= won:
            seen.add(person)
            reached.append(
                Clear(
                    user=person,
                    handle=named.get(person, ""),
                    when=row.finished,
                    seat=seat,
                    agent=agent,
                    match=row.id,
                )
            )
    return Tag(
        game=game,
        rated=rated,
        top=tuple(str(player.key.agent) for player in top),
        top_rating=top[0].rating if top else None,
        unbeaten=None if not rated else not reached,
        reached=tuple(reached),
    )


def tags_from(
    games: Iterable[str],
    rows: Sequence[match_rows.Row],
    tables: Mapping[str, Mapping[rating.PlayerKey, rating.Player]],
    handles: Mapping[int, str] | None = None,
) -> dict[str, Tag]:
    """Every game's tag from one pass of rows and one pass of tables."""
    by_game: dict[str, list[match_rows.Row]] = {}
    for row in rows:
        by_game.setdefault(row.game, []).append(row)
    return {
        game: tag_from(game, by_game.get(game, ()), tables.get(game), handles)
        for game in games
    }


def tag(connection: Any, game: str) -> Tag:
    """One game's tag, read from the store."""
    return tag_from(
        game,
        match_rows.leaderboard_matches(connection, game),
        rating.load(connection, game),
        handles(connection),
    )


def tags(connection: Any, games: Iterable[str] = ()) -> dict[str, Tag]:
    """Every game's tag, in three queries rather than three a game."""
    wanted = tuple(games) or tuple(rating.load_many(connection))
    return tags_from(
        wanted,
        match_rows.leaderboard_matches(connection),
        rating.load_many(connection, wanted),
        handles(connection),
    )


# --------------------------------------------------------------- one person's record


def record_from(game: str, user: int, rows: Sequence[match_rows.Row]) -> Record:
    """What this person has done on this game.

    The best is the highest-rated computer player they have beaten, at the rating it held
    going into that match -- §5's *its rating at the time*, which the match row carries in
    its before-columns and no later result can move.
    """
    played = won = drawn = lost = 0
    best: Beaten | None = None
    cleared = False
    top_wins: set[str] = set()
    for row in rows:
        if row.game != game or not _rated_row(row):
            continue
        seat = row.seat_of(rating.person(user))
        if seat < 0:
            continue
        played += 1
        score = row.score_for(seat)
        if score == 0.5:
            drawn += 1
            continue
        if score != 1.0:
            lost += 1
            continue
        won += 1
        agent = str(row.player(1 - seat).agent)
        found = Beaten(
            agent=agent,
            rating=row.rating_before(1 - seat),
            seat=seat,
            when=row.finished,
            match=row.id,
            top=row.top,
        )
        if best is None or _strength(found) > _strength(best):
            best = found
        if row.top:
            top_wins.add(agent)
            cleared = cleared or _group(row, agent) <= top_wins
    return Record(
        game=game,
        state=_state(played, won, cleared),
        best=best,
        played=played,
        won=won,
        drawn=drawn,
        lost=lost,
    )


# The ladder row's three marks (R7, `docs/plan-return.md` §3 C3), best first. A loss is
# *not yet*: the row says how far up you have got, not what went wrong on the way.
LADDER_WON = "won"
LADDER_DREW = "drew"
LADDER_NOT_YET = "not yet"
LADDER_MARKS = (LADDER_WON, LADDER_DREW, LADDER_NOT_YET)


def ladder_from(
    game: str,
    user: int,
    rows: Sequence[match_rows.Row],
    table: Mapping[rating.PlayerKey, rating.Player],
) -> tuple[tuple[str, str], ...]:
    """This person's best result against each computer player on the game's table.

    The table's computer players weakest first, each with `LADDER_MARKS`' best mark over
    the same rows `record_from` reads (finished rated games, a person against a computer
    player); one never played is *not yet*. Every player on the table is a step, as for
    M4 *the whole ladder* (`milestones._whole_ladder`), so a row of *won* is that badge.
    Pure, so the marks can be checked without a database; the results are the matches'.
    """
    best: dict[str, float] = {}
    for row in rows:
        if row.game != game or not _rated_row(row):
            continue
        seat = row.seat_of(rating.person(user))
        if seat < 0:
            continue
        score = row.score_for(seat)
        if score is None:
            continue
        agent = str(row.player(1 - seat).agent)
        best[agent] = max(best.get(agent, 0.0), score)
    steps = reversed(rating.computers(table))
    return tuple(
        (
            str(player.key.agent),
            LADDER_WON if best.get(str(player.key.agent)) == 1.0
            else LADDER_DREW if best.get(str(player.key.agent)) == 0.5
            else LADDER_NOT_YET,
        )
        for player in steps
    )


def _strength(beaten: Beaten) -> tuple[float, str]:
    """How a record ranks two computer players it has beaten.

    A rating the row never recorded sorts below every rating it did, rather than above
    them: an unrated row is not evidence of a strong opponent.
    """
    return (-1e9 if beaten.rating is None else beaten.rating, beaten.agent)


def _state(played: int, won: int, cleared: bool) -> str:
    if cleared:
        return BEAT_THE_TOP
    if won:
        return WON
    return TRIED if played else UNTRIED


def record(connection: Any, game: str, user: int) -> Record:
    """One person's record on one game, read from the store."""
    return record_from(
        game, user, match_rows.leaderboard_matches(connection, game, user=user)
    )


def records(connection: Any, user: int) -> dict[str, Record]:
    """This person's record on every rated game, tried or not.

    Keyed by every game whose table has computer players, so a profile can say *n of N*
    without a second listing: the games with no record are the untried ones.
    """
    tables = rating.load_many(connection)
    rated = [game for game, table in tables.items() if rating.top(table)]
    rows = match_rows.leaderboard_matches(connection, user=user)
    return {game: record_from(game, user, rows) for game in rated}


def counts(connection: Any, user: int) -> Counts:
    """§5's three counts across the library, each over the rated games."""
    return counts_from(records(connection, user).values())


def counts_from(found: Iterable[Record]) -> Counts:
    rows = list(found)
    return Counts(
        rated=len(rows),
        tried=sum(1 for row in rows if row.state != UNTRIED),
        won=sum(1 for row in rows if row.state in (WON, BEAT_THE_TOP)),
        top=sum(1 for row in rows if row.state == BEAT_THE_TOP),
    )


# --------------------------------------------------------------------------- handles


def handles(connection: Any) -> dict[int, str]:
    """Who the people are, so a first clear can be named rather than numbered.

    A row with no handle is somebody who left, and it reads as `identity.FORMER` -- the
    privacy notice's promise that the games stay and the name goes, kept here because
    this is what prints a name onto somebody else's game.
    """
    with connection.cursor() as cursor:
        cursor.execute("select id, coalesce(handle, %s) from users", (identity.FORMER,))
        return {int(found[0]): str(found[1]) for found in cursor.fetchall()}
