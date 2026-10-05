"""A game in progress, as a row rather than as memory, in one of two modes.

Task 1.4 of `docs/plan-website.md`, against §4. The local application holds a game in a
dictionary that dies with the process, which is the right lifetime for a tool on one
person's machine. A public site wants the other lifetime: a tab closed on the train is a
game to come back to, and a deploy that restarts the unit must not throw away everybody's
board.

So a match is a row. What is stored is the move list, and nothing else about the
position -- replaying those moves through the engine rebuilds the whole session, history
included, because every question on the way is put to the engine again. Python holds a
position and asks; it never answers. An undo is a shorter list, a restart is an empty
one, and a server restart costs a replay.

Beside the moves the row keeps `depths`, how far the machine looked before each one
(`0014_depths.sql`, Q424). That is not about the position, and it is the one answer a
replay cannot get again: the core gave it once, with the reply, and a replayed move is
not searched. So it is stored as the core said it and handed back to the replayed move,
and a row that never kept any -- one from before the column -- loads with none.

Because a take-back and a restart rewrite `moves`, a finish is also copied, with its
moves and its verdict, into `finishes` (`0018_finishes.sql`, Q502a), which nothing
rewrites: a person's win survives *play again*. `site/finishes.py` reads and replays
those rows, and `takebacks` on this row counts the undos a finish copies beside them.

## The two modes

**Sandbox** has undo and restart and counts for nothing. Its row exists only so a tab can
be reopened: no rating moves, no *top* bit is written, and every reading in this module
filters it out. §4's words are *"no statistics, no leaderboard"*.

**Leaderboard** refuses undo and refuses restart. Leaving is a resignation, which is what
makes the abandonment count a real number rather than a hole -- `resign` records it as a
finish, so the rating update runs on it exactly as it runs on a checkmate. `sweep` is the
same act on a timer, for the tab that was closed rather than resigned. Two calls into
`site/rating.py` bracket a leaderboard match: `is_top` at the start, writing whether the
computer seat was the top of the table *then*, and `finish` at the end, which moves both
seats. Neither commits, and nor does anything here; the caller owns the transaction, so a
finish and the ratings it moved land together or not at all.

A leaderboard match is offered only on a game whose picker ladder has been loaded, since
its result is rated on the table that ladder seeds (§5). An unrated game is refused in a
sentence a player can read, and offers sandbox.

## A visitor with no account

Sandbox mode needs no identity, because it counts for nothing. So a visitor who has not
signed in gets a sandbox row with nobody named in their seat -- both seat columns null,
which migration `0003` is what allows -- and plays, undoes and restarts like anybody
else. `Row.seat0` and `Row.seat1` are `None` for that seat, and `Row.is_computer` is how
a reading asks about a seat without minding which of the two it is.

Leaderboard mode still names a person. Its result moves a rating table whose rows are
people and computer players, so `start` refuses it without a user in a sentence a player
can read, and the constraint refuses it one layer down.

## A game nobody is playing

Two computer players at the same board is a *watch*, and the local application has always
offered it -- the game page seats each side from the same picker, and the designer's
*watch it play itself* seats one player twice. Task 1.4 wrote this site's `start` for the
one shape phase 1 had and refused everything else, so the site lost it (Q262).

A watch is a sandbox row with an agent in both seats, which makes it nobody's game by
construction. No seat holds a user, so no rating moves, no badge is granted, no milestone
counts it and no leaderboard reading can see it -- each of those reads leaderboard rows
with people in them. What it is not is a rematch: nobody played the game before it, so
`start` refuses `rematch_of` on a watch and refuses a watch as the match one follows.

What it costs is two searches a ply instead of one, and the rationing is the same: both
seats are put to `quota.check_new_match` in one call, so the strong-seat cap counts them
together and a watch of two strong players one seat short of the cap is refused rather
than seated one over it (Q301). The daily count is read for whoever asked, but the row
names nobody, so the watch itself is not counted in anybody's day. The page seats `minimax:nodes=6400` for a watch, as the designer's does, which is
6-28 ms a move on the heaviest committed rulesets against 0.24-0.67 s for the rung above.

The cap is met again by any game, watched or played, that takes its strong seats back.
A game that goes `quota.SEAT_ACTIVE` without a move, or ends, drops out of the count, so
its next move, a take-back that reopens it and a restart each check the cap under the
same lock a start does (Q410, `_retake_seats`). A game still being played is never asked,
and the daily allowance is never asked again: going on is still the one match.

## Two things this task settled

**`matches.id` stays a secret.** `0001_store.sql` left the choice here. It is still
`secrets.token_urlsafe(9)`, the string the local app's page already carries, and
unguessability is still the access control -- 1.3's cookie can narrow it further, and
until there is one, a printable name would be a way to read somebody else's board.

**A match needs its game in `games` first.** The foreign key says so, and the loader
(task 1.6) is what fills that table from `rulesets/` and `archive/`. Starting a match on
a game the store has never seen is refused by name rather than by a constraint violation.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chess_variant_sandbox.site import finishes, quota, rating, usage
from chess_variant_sandbox.web import opponents
from chess_variant_sandbox.web.human import HUMAN, RESIGNED, HumanGame, summarise
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.play import Session
from chess_variant_sandbox.web.play import Sessions as PlaySessions

SANDBOX = "sandbox"
LEADERBOARD = "leaderboard"
MODES = (SANDBOX, LEADERBOARD)

NEEDS_AN_ACCOUNT = (
    "sign in to play a rated game; practice needs no account and has undo and restart"
)
"""Why leaderboard mode is refused a visitor, in the words the page shows them."""

SIGN_IN_TO_PLAY = "sign in to play"
"""What the opponent list puts after a player only a signed-in person may play (L15)."""


def members_only_said(agent: str) -> str:
    """Why a visitor may not start a game against this player, in the page's words.

    `docs/plan-launch.md` §3.6 (L15): visitors play opponents up to *hard*; *hardest*,
    and anything stronger, is for signed-in people. The player by its name off the ladder,
    never its agent string.
    """
    return (
        f"{opponents.name_of(agent)} is for people with an account: sign in to play it. "
        f"Without one you can play every computer player up to hard"
    )


WATCH_IS_SANDBOX = (
    "a game between two computer players counts for nothing, because nobody played it; "
    "watch it as practice, or play one side yourself to play a rated game"
)
"""Why leaderboard mode is refused a watch, in the words the page shows them."""

NEVER_RATED = (
    "the learned player is never rated, so a game against it is practice; play it as "
    "practice, or pick another computer player for a rated game"
)
"""Why leaderboard mode is refused against a learned net (Q683), in the page's words."""

UNPLAYABLE = (
    "the learned player cannot be loaded on this game right now, so it will not play; "
    "pick another computer player"
)
"""Why a learned seat whose net cannot be loaded refuses to start or to move (Q709), in
the page's words. Never the built-in evaluation under its name instead."""

IDLE_LIMIT = quota.IDLE_LIMIT
"""The sweep's deadline, which lives in `site/quota.py` with every other rationed number.

Task 1.8 gathered them there, as §7 asked. Kept as a name here because `sweep` defaults
to it and this is the module the sweep is in.
"""

# What the local application's rows carry once the loader has put them in `matches`. They
# were played with undo on hand and predate the modes, so they are readable by the anchor
# and counted by nothing -- `rating.finished` drops them from the fold, and so does every
# reading here that a leaderboard depends on.
LOCAL_APP = "local-app"

# The two endings a game between two people has that the engine does not decide (Q570,
# `docs/plan-return.md` §1.3). `agreed` is a draw both people chose; `cancelled` is a
# match either person called off before both had moved, which finishes the row with no
# result, so no reading that wants a result counts it. `site/correspondence.py` writes
# both; they are named here because the readings below filter on them.
AGREED = "agreed"
CANCELLED = "cancelled"

BETWEEN_PEOPLE = "seat0_user is not null and seat1_user is not null"
"""A row with a person in both seats, as SQL: a game between two people (Q570)."""


class MatchError(ValueError):
    """A match cannot be started, moved or ended as asked.

    A `ValueError` like `rating.RatingError`, so the server answers 400 with the sentence
    and the CLI prints one line. Every message here is written for a player to read.
    """


# --------------------------------------------------------------------------- the row


@dataclass(frozen=True)
class Row:
    """One match, as the readings want it.

    `seat0` and `seat1` are the two players, each a `rating.PlayerKey`, so a seat reads
    the same way here as it does on the table. A seat is `None` on a sandbox row a
    visitor with no account is playing, which is the one seat the rating table has no row
    for. `result` is seat 0's score and is None while the match runs.

    `top` and `top_players` are the pair the *Unbeaten* tag reads (§5). `top` is whether
    this match's own computer seat was the top of the game's table when the match
    started; `top_players` is the whole tied group it was read against. Both are written
    once, at the start, so a target that moves later cannot rewrite what this match was
    played against. `before` is each seat's rating going in, written at the finish by
    `rating.finish`, and it is what "the highest computer beaten, at its rating then"
    reads.
    """

    id: str
    game: str
    mode: str
    seat0: rating.PlayerKey | None
    seat1: rating.PlayerKey | None
    seed: int
    plies: int
    result: float | None
    reason: str
    left: bool
    top: bool
    top_players: tuple[str, ...]
    before: tuple[float | None, float | None]
    source: str | None
    started: datetime
    moved: datetime | None
    finished: datetime | None
    # Each seat's rating deviation going in, beside `before` and written by the same
    # call. It says whether a rating was established *then*, which is what
    # `badges.due_from` reads and what a table that has moved since cannot answer.
    # Last and defaulted so a row built by hand for a reading that does not want it
    # stays a row.
    deviations: tuple[float | None, float | None] = (None, None)

    # The match this one was a rematch of, or None on every match started any other way
    # (box C1 of `docs/archive/plan-meta-game.md`). Only the board's rematch control
    # writes it, so None means *not a rematch* rather than *unknown*. Defaulted for the
    # same reason `deviations` is.
    rematch_of: str | None = None

    @property
    def running(self) -> bool:
        return self.finished is None

    def seat_of(self, key: rating.PlayerKey) -> int:
        """Which side of the board a player sat on, or -1 for one who was not in it."""
        if self.seat0 == key:
            return 0
        if self.seat1 == key:
            return 1
        return -1

    def is_computer(self, seat: int) -> bool:
        """Whether that seat is a computer player.

        The reading that does not mind which kind of person the other seat holds, so a
        visitor's seat answers *no* here rather than raising.
        """
        key = self.seat0 if seat == 0 else self.seat1
        return key is not None and key.is_computer

    def score_for(self, seat: int) -> float | None:
        """That seat's score, from seat 0's, which is what the column holds."""
        if self.result is None:
            return None
        return self.result if seat == 0 else 1.0 - self.result

    def player(self, seat: int) -> rating.PlayerKey:
        """That seat's player, for a reading the rating table is behind.

        A visitor's seat has no key and is refused here rather than answered with one,
        because every caller is asking a question only a rated seat has an answer to.
        Ask `is_computer` instead where the seat may be a visitor's.
        """
        key = self.seat0 if seat == 0 else self.seat1
        if key is None:
            raise MatchError(f"seat {seat} of match {self.id} is a visitor with no account")
        return key

    def rating_before(self, seat: int) -> float | None:
        """That seat's rating going into the match, or None on a row nothing rated."""
        return self.before[0] if seat == 0 else self.before[1]

    def deviation_before(self, seat: int) -> float | None:
        """That seat's rating deviation going in, or None on a row nothing rated."""
        return self.deviations[0] if seat == 0 else self.deviations[1]


_ROW_COLUMNS = (
    "id, game, mode, seat0_user, seat0_agent, seat1_user, seat1_agent, seed,"
    " jsonb_array_length(moves), result, reason, \"left\", top, top_players,"
    " seat0_rating_before, seat1_rating_before, source, started, moved, finished,"
    " seat0_deviation_before, seat1_deviation_before, rematch_of"
)


def _key(user: Any, agent: Any) -> rating.PlayerKey | None:
    """One seat's two columns as a player, or None for a visitor with no account."""
    if user is None and agent is None:
        return None
    return rating.PlayerKey(user=user, agent=agent)


def _to_row(found: Sequence[Any]) -> Row:
    return Row(
        id=str(found[0]),
        game=str(found[1]),
        mode=str(found[2]),
        seat0=_key(found[3], found[4]),
        seat1=_key(found[5], found[6]),
        seed=int(found[7]),
        plies=int(found[8]),
        result=None if found[9] is None else float(found[9]),
        reason=str(found[10] or ""),
        left=bool(found[11]),
        top=bool(found[12]),
        top_players=tuple(found[13] or ()),
        before=(
            None if found[14] is None else float(found[14]),
            None if found[15] is None else float(found[15]),
        ),
        source=found[16],
        started=found[17],
        moved=found[18],
        finished=found[19],
        deviations=(
            None if found[20] is None else float(found[20]),
            None if found[21] is None else float(found[21]),
        ),
        rematch_of=None if found[22] is None else str(found[22]),
    )


def row(connection: Any, match_id: str) -> Row:
    """One match. A `KeyError` for one that does not exist, which the server answers 404."""
    with connection.cursor() as cursor:
        cursor.execute(f"select {_ROW_COLUMNS} from matches where id = %s", (match_id,))
        found = cursor.fetchone()
    if found is None:
        # A visitor reads this (P20 of `docs/plan-understanding.md` §3.3): an address,
        # not an id.
        raise KeyError("There is no game at this address.")
    return _to_row(found)


# ------------------------------------------------------------------- starting a match


def start(
    connection: Any,
    game: str,
    mode: str,
    seats: Sequence[str],
    user: int | None = None,
    seed: int = 0,
    rematch_of: str | None = None,
) -> Row:
    """Seat a person against a computer player, or two computer players, and write the row.

    `seats` is the pair the page asked for, in board order, with `"human"` for the
    person's side -- the local application's own spelling, so the site's session route
    takes the body it already takes.

    Two people are `site/correspondence.py`'s, which writes their row itself (Q570).

    `user` is None for a visitor who has not signed in, and their seat is written with
    nobody in it. Sandbox takes that, because a sandbox game counts for nothing;
    leaderboard refuses it, because its result moves a table whose rows are people.

    Two computer players and nobody is a *watch*, and it is sandbox only: nobody played
    it, so there is nothing for a rating to move. Both seats are rationed, together,
    because the box pays for two searches a ply rather than one.

    In leaderboard mode three things happen that sandbox does not do. The game is checked
    for a rating table, because a game with no picker ladder has no computer players to
    rate against and §5 offers it sandbox only. The top is read live and written once --
    the bit for this match's own opponent and the whole tied group beside it -- so a
    target that moves later cannot rewrite what this match was played against. And any
    leaderboard match this person already has running on this game is resigned, which is
    §4's rule against keeping a losing board open while starting a fresh one.

    Two limits are read before the row is written, both `site/quota.py`'s and both
    refusing in a sentence: how many matches this person has started today, and how many
    seats against the strongest opponent are live across the site (§2, task 1.8). Both
    are read once for the whole match, with every computer seat it would add. A refused
    match leaves nothing behind, because nothing has been written yet.

    Each limit is read under a lock that holds until the caller commits (Q408), so two
    starts arriving together one place short seat one and refuse the other, rather than
    both reading the count before either row exists. That is why the row is written in
    the caller's transaction and why nothing is written before the check: a start that
    waits there must hold no row lock the start it waits on could need.

    `rematch_of` names the match the board's rematch control was pressed on, and this is
    the one call that writes it (C1). The new match is a rematch in that column and a
    fresh match in every other respect -- its own id, seats, seed, moves, quota and
    rating update -- and the row it names is left alone. What is checked is that the
    named match is the same player's on the same game, so the column says nothing the two
    rows do not already hold; anything else is refused rather than written as a guess.
    """
    if mode not in MODES:
        raise MatchError(f"a match is {SANDBOX} or {LEADERBOARD}, not {mode!r}")
    if len(seats) != 2:
        raise MatchError("a match has two seats")
    people = list(seats).count(HUMAN)
    if people > 1:
        raise MatchError("a match on the site seats one person for now")
    watched = people == 0

    if user is None:
        # L15: a visitor plays opponents up to *hard*. Refused before the game is read or
        # anything is counted, so the refusal is the same on every game.
        held = [str(seat) for seat in seats if seat != HUMAN and quota.members_only(str(seat))]
        if held:
            raise MatchError(members_only_said(held[0]))

    name, _spec = _game_row(connection, game, starting=True)
    # Every computer seat in one call, so a watch's two strong seats are counted against
    # the allowance together rather than each against the same one (Q301). It locks what
    # it counts until this transaction ends (Q408), so it stays ahead of every write.
    quota.check_new_match(connection, user, *(str(seat) for seat in seats if seat != HUMAN))
    if watched and rematch_of is not None:
        raise MatchError("a game nobody played is not a rematch of anything")
    followed = _followed(connection, game, user, rematch_of)

    is_top = False
    group: tuple[str, ...] = ()
    if mode == LEADERBOARD:
        if watched:
            raise MatchError(WATCH_IS_SANDBOX)
        if user is None:
            raise MatchError(NEEDS_AN_ACCOUNT)
        agent = seats[1 - list(seats).index(HUMAN)]
        if opponents.never_rated(agent):
            # Q683: the learned player is offered as practice only, whatever a table holds.
            raise MatchError(NEVER_RATED)
        table = rating.load(connection, game)
        if not rating.computers(table):
            raise MatchError(
                f"the computer players on {name} have no ratings yet, so it can only be "
                f"played as practice; rated play opens once they have ratings"
            )
        if rating.computer(agent) not in table:
            # S23: the player by its name, or as *that computer player* where the ladder
            # has none for it -- never its agent string.
            named = opponents.name_of(agent)
            who = "that computer player" if named == "the computer" else named
            raise MatchError(
                f"{who} has no rating on {name} yet, so a result against it cannot be "
                f"rated; play it as practice, or pick one of the computer players listed "
                f"for this game"
            )
        group = tuple(player.key.agent or "" for player in rating.top(table))
        is_top = agent in group
        for running in _running(connection, game, user):
            resign(connection, running.id, by=user)

    # Only a learned seat's column is named, so every other match writes exactly what it
    # wrote before `0025_played.sql` and the column stays null.
    played = {
        f"seat{index}_played": net
        for index, seat in enumerate(seats)
        if (net := _played(seat, game)) is not None
    }
    match_id = secrets.token_urlsafe(9)
    columns, players = _seat_columns(seats, user)
    named = "".join(f", {column}" for column in played)
    with connection.cursor() as cursor:
        cursor.execute(
            f"insert into matches"
            f" (id, game, mode, seed, top, top_players, rematch_of,"
            f" {columns[0]}, {columns[1]}{named})"
            f" values (%s, %s, %s, %s, %s, %s, %s, %s, %s{', %s' * len(played)})",
            (match_id, game, mode, seed, is_top, list(group), followed, *players,
             *played.values()),
        )
    return row(connection, match_id)


def _played(seat: str, game: str) -> str | None:
    """What `seat{n}_played` holds for this seat: its net, or None for any other player.

    A learned seat whose net cannot be loaded is refused here, before anything is
    written, in the page's words (Q709).
    """
    from chess_variant_sandbox.agents.nets import NetError, played_by

    if seat == HUMAN:
        return None
    try:
        return played_by(seat, game) or None
    except NetError as error:
        raise MatchError(UNPLAYABLE) from error


def _check_played(connection: Any, found: Row) -> None:
    """Refuse a reply whose learned seat would not play the net its row records (Q709).

    A net that has gone (a deploy without its `.net`, a store refused since) or changed
    under the same name stops the game in a sentence rather than move: what the row says
    played is then always what did. A row from before `0025_played.sql` records nothing
    and is held to the net being loadable at all.
    """
    recorded = played(connection, found.id)
    for seat, key in enumerate((found.seat0, found.seat1)):
        if key is None or key.agent is None:
            continue
        now = _played(key.agent, found.game)
        if recorded[seat] is not None and now != recorded[seat]:
            raise MatchError(UNPLAYABLE)


def played(connection: Any, match_id: str) -> tuple[str | None, str | None]:
    """Which net each learned seat of this match really played, `net:<12 hex>`.

    `seat{n}_played` (`0025_played.sql`, Q709): None for every other seat and every row
    written before it. The agent string names a training procedure, the same on every
    board; this names the net. Read on its own rather than in `_ROW_COLUMNS`, so a row
    still reads on a store that has not taken the migration yet.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select seat0_played, seat1_played from matches where id = %s", (match_id,)
        )
        found = cursor.fetchone()
    if found is None:
        return None, None
    return found[0], found[1]


def _seat_columns(
    seats: Sequence[str], user: int | None
) -> tuple[tuple[str, str], tuple[Any, Any]]:
    """Which column each seat is written to, and what goes in it.

    A person's seat holds their id, or nothing at all for a visitor; every other seat
    names a computer player. Both seats are read the same way, so a watch is two agent
    columns rather than a third shape spelled out here.
    """
    names: list[str] = []
    values: list[Any] = []
    for index, seat in enumerate(seats):
        holds_person = seat == HUMAN
        names.append(f"seat{index}_user" if holds_person else f"seat{index}_agent")
        values.append(user if holds_person else seat)
    return (names[0], names[1]), (values[0], values[1])


def _followed(
    connection: Any, game: str, user: int | None, rematch_of: str | None
) -> str | None:
    """The match a rematch followed, checked, or None for a match that is not one.

    Four things make the column trustworthy, and each is refused in a sentence rather
    than dropped, because a link silently turned into a null would report fewer rematches
    than happened. The named match has to exist, it has to be on this same game, somebody
    has to have been playing it, and it has to be this player's. A visitor has no id to
    match on, so what is asked of their rematch is that the named match had nobody signed
    in either -- the one shape a visitor ever plays -- which keeps a signed-out board from
    naming somebody else's.
    """
    if rematch_of is None:
        return None
    try:
        previous = row(connection, rematch_of)
    except KeyError:
        raise MatchError(f"no match {rematch_of} to play again") from None
    if previous.game != game:
        raise MatchError("a rematch is another game of the same rules")
    if previous.is_computer(0) and previous.is_computer(1):
        raise MatchError("nobody played that game, so there is no rematch of it")
    if user is None:
        seated = [
            key
            for key in (previous.seat0, previous.seat1)
            if key is not None and key.user is not None
        ]
        if seated:
            raise MatchError("sign in to play that game again; it is not this board's")
    elif previous.seat_of(rating.person(user)) < 0:
        raise MatchError("only a player of that game can play it again")
    return previous.id


def _game_row(
    connection: Any, game: str, starting: bool = False
) -> tuple[str, dict[str, Any]]:
    """The game's name and spec, or the refusal that names the loader.

    A game the maintainer has killed (1.8's per-game switch, `games.revoked`) is refused
    here too. The site does not list it and does not open its page, and this is the third
    door: a hash somebody kept cannot be played by pasting it back in.

    `starting` also refuses a game the loader retired -- one whose spec has left
    `rulesets/` and `archive/`. That refusal belongs to a new match alone: `session`
    rebuilds a finished match from this same row, and a game somebody played last month
    has to stay readable after the spec it was played under was superseded.

    A deleted game is refused whatever is being asked, because the refusal is not a
    policy here but the absence of a spec: the rules were deleted, so there is nothing to
    replay the moves through. The row and the result stand; the board does not come back.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select name, spec, revoked, retired, deleted from games where hash = %s",
            (game,),
        )
        found = cursor.fetchone()
    if found is None:
        raise MatchError(
            "this game isn't on the site, so there is nothing to play"
        )
    if found[4] is not None:
        raise MatchError(
            "these rules have been deleted, so this game cannot be opened or played; "
            "the matches played on it, and their results, are unchanged"
        )
    if found[2] is not None:
        raise MatchError(
            f"{found[0]} has been taken off the site and cannot be played; the games "
            f"already played on it are unchanged"
        )
    if starting and found[3] is not None:
        raise MatchError(
            f"{found[0]} has been retired from the library, so no new game can be started "
            f"on it; the games already played on it are unchanged and still readable"
        )
    return str(found[0]), dict(found[1])


def _running(connection: Any, game: str, user: int) -> tuple[Row, ...]:
    """This person's unfinished leaderboard matches against the computer on this game.

    Against the computer only: §4's rule is about a person and a computer, and a game
    against another person is never resigned by starting something else (Q570,
    `docs/plan-return.md` §1.3's *Many games at once*). Two friends' games of the same
    rules run side by side.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_ROW_COLUMNS} from matches"
            f" where game = %s and mode = '{LEADERBOARD}' and finished is null"
            f" and (seat0_user = %s or seat1_user = %s)"
            f" and not ({BETWEEN_PEOPLE}) order by started",
            (game, user, user),
        )
        return tuple(_to_row(found) for found in cursor.fetchall())


# -------------------------------------------------------------------------- playing


def session(connection: Any, match_id: str) -> Session:
    """The position this match is in, rebuilt by replaying its moves through the engine.

    The one call that turns a row back into a game. Every move goes through
    `Session.replay`, which asks the engine whether it is legal before applying it, so a
    row that could not have happened fails here rather than drawing a board that never
    existed. Each move takes back the depth stored beside it, so a reloaded board says
    how far the machine looked wherever the board it replaced did.

    The row's mode travels with the position, because every snapshot the page is handed
    comes from here -- a move, a reply, a take-back, a restart and a resignation all
    rebuild the session first. So the board cannot be drawn without being told whether
    the game it is drawing counts.

    So does a finish the position does not show. A resignation ends the match in a
    position that is still playable, and the engine has no answer about it -- what ended
    it is the row, so the row is what says so.
    """
    found = row(connection, match_id)
    name, spec = _game_row(connection, found.game)
    seats = (_seat_name(found.seat0), _seat_name(found.seat1))
    nets = played(connection, match_id)
    built = Session(
        id=found.id,
        mode=found.mode,
        verdict=_verdict(found),
        game=Game(
            hash=found.game,
            name=name,
            description=str(spec.get("description") or ""),
            path=Path(),
            source="site",
            spec=spec,
        ),
        seats=seats,
        seed=found.seed,
        state=engine_for(spec).initial_state(),
        played_by=(nets[0] or "", nets[1] or ""),
    )
    for move, depth in _moves(connection, match_id):
        built.replay(move, depth)
    return built


def _verdict(found: Row) -> dict[str, Any] | None:
    """How this match ended, as the row has it, or None while it runs.

    `result` is seat 0's score, so a whole point to seat 0 is seat 0 winning and a half
    is a draw. Nothing here reads the position: a game the engine can see the end of
    reports it itself, and this is for the endings it cannot -- a resignation, and the
    sweep's resignation on behalf of somebody who closed the tab.
    """
    if found.finished is not None and found.reason == CANCELLED:
        # Called off before both people moved (Q570): over, with nobody winning.
        return {"winner": None, "reason": CANCELLED}
    if found.finished is None or found.result is None:
        return None
    winner = None if found.result == 0.5 else (0 if found.result > 0.5 else 1)
    return {"winner": winner, "reason": found.reason}


def _seat_name(key: rating.PlayerKey | None) -> str:
    """What the page calls this seat. A visitor's seat is a person like any other."""
    return key.agent if key is not None and key.agent is not None else HUMAN


def _moves(connection: Any, match_id: str) -> list[tuple[int, int | None]]:
    """The stored moves in order, each with the depth stored beside it.

    A row that never kept depths -- `depths` null, which is every row written before
    `0014_depths.sql` and a match nobody has moved in -- gives `None` for every move. The
    constraint keeps the two arrays one length, so an entry is always the move's own.
    """
    with connection.cursor() as cursor:
        cursor.execute("select moves, depths from matches where id = %s", (match_id,))
        found = cursor.fetchone()
    if found is None:
        return []
    moves = list(found[0] or [])
    depths = list(found[1]) if found[1] is not None else [None] * len(moves)
    return [
        (int(move), None if depth is None else int(depth))
        for move, depth in zip(moves, depths, strict=True)
    ]


def play(connection: Any, match_id: str, move: int) -> dict[str, Any]:
    """The person's move, applied by the engine and appended to the row.

    Not for a game between two people, whose move has to say whose it is and is sent
    through `correspondence.move`, which checks that before it calls `_advance`.
    """
    _not_between_people(row(connection, match_id))
    return _advance(connection, match_id, lambda game: game.play(move))


def _not_between_people(found: Row) -> None:
    """Refuse an act that names no person on a game two people are playing."""
    seats = (found.seat0, found.seat1)
    if all(seat is not None and seat.user is not None for seat in seats):
        raise MatchError(
            "this is a game between two people, so a move or a resignation in it is "
            "sent by one of its players"
        )


def play_agent(connection: Any, match_id: str) -> dict[str, Any]:
    """The computer player's reply, chosen by the core's own agent and appended.

    The agent is the rated one, because `Session.play_agent` runs `choose_move`; an
    opponent built any other way would be a different player from the one the table has
    a row for. How far it looked is stored beside the move, where the core reported one.
    """
    def checked(found: Row) -> None:
        # The budget for the usage row, and the net the row says plays (Q709).
        _note_budget(found)
        _check_played(connection, found)

    return _advance(connection, match_id, lambda game: game.play_agent(), checked)


def _agents(found: Row) -> list[str]:
    """The computer players seated in this match, in board order."""
    return [
        seat.agent for seat in (found.seat0, found.seat1) if seat is not None and seat.agent
    ]


def _note_budget(found: Row) -> None:
    """The opponent's node budget, for the usage row this reply lands on.

    The largest of the seated budgets, which is the opponent's in a game somebody is
    playing and the heavier of the two in a watch.
    """
    budgets = [quota.nodes_in(agent) for agent in _agents(found)]
    if budgets:
        usage.note(f"nodes={max(budgets)}")


def _retake_seats(connection: Any, found: Row) -> None:
    """Refuse to go on if this game gave its strong seats up and none is free now (Q410).

    Called ahead of every write that leaves a row live -- a move, a reply, a take-back and
    a restart -- and never ahead of a resignation, which gives a seat up rather than
    taking one. A game still being played goes straight past; `quota.check_resumed_match`
    says why. Nothing is written first, because the check may wait on the site's lock.
    """
    quota.check_resumed_match(connection, found.id, *_agents(found))


def _advance(
    connection: Any, match_id: str, act: Any, noted: Any = None
) -> dict[str, Any]:
    """Apply one move to a running match, store the new move list, and finish if it ended."""
    found = row(connection, match_id)
    if noted is not None:
        noted(found)
    if not found.running:
        raise MatchError("this game is over")
    _retake_seats(connection, found)
    game = session(connection, match_id)
    snapshot = act(game)
    _store_moves(connection, match_id, game.played, game.depths, moved=True)
    outcome = snapshot.get("outcome")
    if outcome is not None:
        _finish(
            connection,
            match_id,
            result=_score(outcome.get("winner")),
            reason=str(outcome.get("reason") or ""),
            left=False,
        )
    return snapshot


def _score(winner: int | None) -> float:
    """Seat 0's score, which is what `matches.result` holds."""
    if winner is None:
        return 0.5
    return 1.0 if winner == 0 else 0.0


def _store_moves(
    connection: Any,
    match_id: str,
    moves: Sequence[int],
    depths: Sequence[int | None],
    moved: bool,
) -> None:
    """The move list and the depth beside each move, written in one statement.

    Together, because `matches_depths_beside_moves` holds the two arrays to one length and
    an entry of `depths` means nothing apart from the move at the same place.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set moves = %s::jsonb, depths = %s::jsonb"
            + (", moved = now()" if moved else ", moved = null")
            + " where id = %s",
            (
                json.dumps([int(move) for move in moves]),
                json.dumps([None if depth is None else int(depth) for depth in depths]),
                match_id,
            ),
        )


def undo(connection: Any, match_id: str) -> dict[str, Any]:
    """Take back to the last position the person was asked to move in. Sandbox only.

    One ply is not enough with a computer player opposite -- taking back your own move
    and leaving the reply on the board is not a position anybody was ever in -- so the
    session decides how far back that is and the row is truncated to match. Taking back
    the move that ended the game reopens it, which costs nothing: no sandbox row was ever
    rated, so there is no verdict to unpick.

    A take-back is somebody at the board, so it counts as a move for the strong seat: a
    game that sat or ended meets the cap first, and the row's `moved` is written even when
    nothing is left on the board. Writing it empty there handed an active game's seat
    back the moment it reached the opening position, and its next move then met the cap
    as though it had been left (Q410).
    """
    found = _sandbox_only(connection, match_id, "undone")
    game = session(connection, match_id)
    if not game.played:
        return game.snapshot()
    _retake_seats(connection, found)
    snapshot = game.undo()
    _store_moves(connection, match_id, game.played, game.depths, moved=True)
    _reopen(connection, match_id)
    # Counted, so a finish reached with the undo button says so (`0018_finishes.sql`).
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set takebacks = takebacks + 1 where id = %s", (match_id,)
        )
    return snapshot


def restart(connection: Any, match_id: str) -> dict[str, Any]:
    """Put the board back to the opening position, keeping the row. Sandbox only.

    It is the same match rather than another one, so the daily allowance is not asked
    again. The strong-seat cap is, when the game had sat or ended, because a restart
    makes it live (Q410).
    """
    found = _sandbox_only(connection, match_id, "restarted")
    _retake_seats(connection, found)
    _store_moves(connection, match_id, (), (), moved=False)
    _reopen(connection, match_id)
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set started = now(), takebacks = 0 where id = %s", (match_id,)
        )
    return session(connection, match_id).snapshot()


def _sandbox_only(connection: Any, match_id: str, action: str) -> Row:
    found = row(connection, match_id)
    if found.seat0 is not None and found.seat1 is not None and not (
        found.seat0.is_computer or found.seat1.is_computer
    ):
        # A practice game between two people is kept and counts for nothing, and still
        # has no take-back: undoing a move is undoing the other person's turn (§1.3).
        raise MatchError(
            f"a game between two people cannot be {action}; sent moves are final"
        )
    if found.mode != SANDBOX:
        raise MatchError(
            f"a rated game cannot be {action}, because its result counts; "
            f"practice is the one that takes moves back"
        )
    return found


def _reopen(connection: Any, match_id: str) -> None:
    """Clear a finish. Only ever called on a sandbox row, which nothing rated."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set result = null, reason = null, finished = null,"
            " \"left\" = false where id = %s",
            (match_id,),
        )


# ------------------------------------------------------------------------ finishing


def resign(
    connection: Any, match_id: str, by: int | None = None, left: bool = True
) -> Row:
    """End a running match as a loss for the person, and rate it as one.

    §4: a leaderboard match the player leaves is resigned for them, and the row records
    `reason = resigned` and `left = true`. A resignation is a finish, so the rating
    update runs on it exactly as it runs on a checkmate -- which is the whole point of
    recording it rather than deleting the row.

    `by` names the seat that resigned when it is not obvious; without it the person's
    seat is the one that loses, which is the only shape a game against the computer has.

    `left` is what the row records. Against the computer a resignation is the same act
    as leaving and is always recorded so. Between two people (Q570) a person resigning
    finished the game, and only the sweep's loss after five days is a game they left,
    which is what the abandonment line reads (`docs/plan-return.md` §1.6).
    """
    found = row(connection, match_id)
    if not found.running:
        raise MatchError("this game is already over")
    seat = _loser(found, by)
    # `result` is seat 0's score, and the loser's seat index equals it. Seat 0
    # resigning scores 0, seat 1 resigning leaves seat 0 with the point.
    _finish(connection, match_id, result=float(seat), reason=RESIGNED, left=left)
    return row(connection, match_id)


def _loser(found: Row, by: int | None) -> int:
    if by is not None:
        seat = found.seat_of(rating.person(by))
        if seat < 0:
            raise MatchError(f"user {by} is not in this game")
        return seat
    people = [seat for seat in (0, 1) if not found.is_computer(seat)]
    if not people:
        raise MatchError("nobody is playing this game, so there is nobody to resign it")
    if len(people) != 1:
        raise MatchError(
            "this game has two people in it, so which of them resigned has to be said"
        )
    return people[0]


def _finish(
    connection: Any, match_id: str, result: float, reason: str, left: bool
) -> rating.Update | None:
    """Record the verdict and move the table, in that order and in one transaction.

    The rating call is unconditional because `rating.finish` is what knows which rows the
    table counts: it returns None, and writes nothing, for a sandbox match and for a row
    loaded from the local application.

    The badges of §2.9 are granted last, in this same transaction, so a finish and the
    badges it completed land together or neither does. They run after the rating call
    because *Giant-killer* reads the ratings going in, which that call is what writes.
    `site/badges.py` is imported here rather than at the top because it reads this
    module's rows.
    """
    from chess_variant_sandbox.site import badges

    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set result = %s, reason = %s, \"left\" = %s, finished = now()"
            " where id = %s",
            (result, reason, left, match_id),
        )
    # Kept apart from the row, which a sandbox take-back or restart rewrites (Q502a).
    finishes.record(connection, match_id)
    update = rating.finish(connection, match_id)
    badges.grant_for_match(connection, match_id)
    return update


def sweep(
    connection: Any,
    idle: timedelta = IDLE_LIMIT,
    between: timedelta = quota.CORRESPONDENCE_IDLE,
) -> tuple[Row, ...]:
    """Resign every leaderboard match that has sat untouched past the limit.

    What the deploy's timer runs, and `chess-variant-sandbox site sweep` on the command line. It
    resigns and never deletes: the user's retention answer of 2026-09-11 is *the account
    goes, the games stay*, and the rating table is a fold over these rows, so a deleted
    row would change the numbers of everyone who was in it.

    Idle is measured from the last move, or from the start for a match nobody has moved
    in, and the clock is the database's so that two workers cannot disagree about what
    *now* is.

    A match with two people in it has its own limit, `between` (five days, the user's
    of 2026-10-02), in either mode, and the one it is resigned for is **the player to
    move**, as the engine says from the replayed position (Q570, `docs/plan-return.md`
    §1.6). It is recorded as a game they left. A game where it is your opponent's move
    is never lost by you for their silence.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_ROW_COLUMNS} from matches"
            f" where mode = '{LEADERBOARD}' and finished is null and source is null"
            f" and coalesce(moved, started) < now() - %s order by started, id",
            (idle,),
        )
        stale = [_to_row(found) for found in cursor.fetchall()]
    resigned = []
    for found in stale:
        if found.is_computer(0) == found.is_computer(1):
            continue
        resigned.append(resign(connection, found.id))
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_ROW_COLUMNS} from matches"
            f" where {BETWEEN_PEOPLE} and finished is null and source is null"
            f" and coalesce(moved, started) < now() - %s order by started, id",
            (between,),
        )
        waiting = [_to_row(found) for found in cursor.fetchall()]
    for found in waiting:
        # Whose move it is is the engine's answer about the position the row replays to.
        mover = session(connection, found.id).state.side_to_move
        absent = found.player(mover).user
        resigned.append(resign(connection, found.id, by=absent, left=True))
    return tuple(resigned)


# -------------------------------------------- what tasks 1.5 and 1.6 read off the rows


def leaderboard_matches(
    connection: Any,
    game: str = "",
    user: int | None = None,
    running: bool = False,
) -> tuple[Row, ...]:
    """A game's rated matches in finish order, with the *top* bit and both seats.

    The one query the stars, the sort keys and the *Unbeaten* tag (task 1.5) and the
    loader's own checks (task 1.6) are built on, so that neither writes SQL against this
    table. Sandbox rows are not in it at all, which is §4's *"read by no statistic, no
    rating, no anchor row and no sort key"*. Nor is a game between two people that was
    cancelled before both had moved (Q570): its row stays, and no reading counts it.

    Finish order, because that is the order the rating fold replays in and the order a
    first clear is decided in; `id` breaks a tie on the timestamp so two readings agree.
    Everything else is a filter over the same rows: `game` narrows to one game, `user` to
    one person's, and `running` adds the matches still being played, which sort after the
    finished ones. `Row.source` is on every row, so a caller that must exclude the
    pre-site local-application games can, and the anchor -- which wants them -- need not.
    """
    where = [f"mode = '{LEADERBOARD}'", f"coalesce(reason, '') <> '{CANCELLED}'"]
    values: list[Any] = []
    if not running:
        where.append("finished is not null")
    if game:
        where.append("game = %s")
        values.append(game)
    if user is not None:
        where.append("(seat0_user = %s or seat1_user = %s)")
        values.extend((user, user))
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_ROW_COLUMNS} from matches where {' and '.join(where)}"
            f" order by finished nulls last, id",
            tuple(values),
        )
        return tuple(_to_row(found) for found in cursor.fetchall())


def human_games(connection: Any, game: str = "") -> list[HumanGame]:
    """The leaderboard rows as the human anchor reads them.

    `web/human.py`'s own rows, out of `matches` instead of out of a JSONL file, and
    leaderboard only -- a sandbox game was played with undo on hand and says nothing
    about how hard the game is. The pre-site rows the loader brings in are here, because
    they are exactly what the anchor kept them for (§5); they are excluded from the
    rating fold by `rating.finished` and from the tag by their `source`.
    """
    names = _names(connection)
    played = _finished_moves(connection, game)
    found = []
    for match in leaderboard_matches(connection, game, running=True):
        if match.is_computer(0) == match.is_computer(1):
            continue
        seat = 1 if match.is_computer(0) else 0
        result = match.score_for(seat)
        found.append(
            HumanGame(
                session=match.id,
                ruleset_hash=match.game,
                ruleset_name=names.get(match.game, ""),
                opponent=_seat_name(match.seat1 if seat == 0 else match.seat0),
                human_seat=seat,
                seed=match.seed,
                started=match.started.timestamp(),
                plies=match.plies,
                result=result,
                winner=None if result is None or result == 0.5 else (seat if result else 1 - seat),
                reason=match.reason,
                finished=None if match.finished is None else match.finished.timestamp(),
                moves=played.get(match.id, ()),
            )
        )
    return found


def _finished_moves(connection: Any, game: str = "") -> dict[str, tuple[int, ...]]:
    """Each finished leaderboard match's moves, as `HumanGame.moves` carries them.

    Finished only, because that is when the local application writes them too: a game
    still running is a verdict and a move list nobody has yet.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select id, moves from matches where mode = '{LEADERBOARD}'"
            " and finished is not null" + (" and game = %s" if game else ""),
            (game,) if game else (),
        )
        return {
            str(found[0]): tuple(int(move) for move in found[1] or ())
            for found in cursor.fetchall()
        }


def _names(connection: Any) -> dict[str, str]:
    with connection.cursor() as cursor:
        cursor.execute("select hash, name from games")
        return {str(found[0]): str(found[1]) for found in cursor.fetchall()}


class Record:
    """`HumanRecord`'s reading half, over `matches` rather than `runs/human/games.jsonl`.

    The site keeps what the local application kept, in the table §3 gave it, so the
    anchor's own summary is the same function over rows from a different place. The
    writing half is gone on purpose: on the site a game is recorded by being played, and
    a second store to keep in step with the first is the bug this replaces.
    """

    def __init__(self, connect: Any) -> None:
        self._connect = connect

    def games(self, ruleset_hash: str = "") -> list[HumanGame]:
        with self._connect() as connection:
            return human_games(connection, ruleset_hash)

    def summary(self, ruleset_hash: str, ratings: dict[str, float] | None) -> dict[str, Any]:
        return summarise(self.games(ruleset_hash), ruleset_hash, ratings)

    def note_start(self, *_args: Any, **_kwargs: Any) -> None:
        raise MatchError("the site records a game as a match row, not in the anchor's file")

    def note_result(self, *_args: Any, **_kwargs: Any) -> None:
        raise MatchError("the site records a result on the match row, not in the anchor's file")


class Sessions:
    """`web.play.Sessions` over the store, so the site's read route needs no second reading.

    `App.get` answers `/api/session/<id>` with `self.sessions.get(id).snapshot()`, and
    that is the whole of what this has to offer: a position, rebuilt from the row.
    Starting a library game goes through `start` above instead, because a match needs a
    mode and may name a person, neither of which the local application's signature has
    anywhere to put.

    A draft is the exception, and stays in memory. `POST /api/design/play` plays rules
    nobody has kept, so there is no `games` row for a match row to point at and §2's own
    note on that route says it writes nothing. Those sessions live in `drafts` -- the
    local application's own store, with the local application's lifetime -- and `get`
    looks there first, so one route reads both kinds.
    """

    def __init__(self, connect: Any, drafts: PlaySessions | None = None) -> None:
        self._connect = connect
        self.drafts = drafts if drafts is not None else PlaySessions()

    def get(self, session_id: str) -> Session:
        found = self.drafts.find(session_id)
        if found is not None:
            return found
        with self._connect() as connection:
            return session(connection, session_id)

    def start(self, *args: Any, **kwargs: Any) -> Session:
        if not kwargs.get("scratch"):
            raise MatchError("a match on the site is started with a mode; see site/matches.py")
        return self.drafts.start(*args, **kwargs)
