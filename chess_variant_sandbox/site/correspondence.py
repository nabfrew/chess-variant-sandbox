"""A game between two people, played by correspondence (Q570, `docs/plan-return.md` §1).

Two members play one game on a game's rules, each moving when they like. The row is an
ordinary `matches` row with a person in both seats, so everything `site/matches.py` does
for a row -- the replay through the engine, the finish, the rating update, the kept
finish, the badges -- is the same call here. What this module adds is the part where two
people are involved: whose move it is, a draw both of them agree to, calling a game off
before it has started, and the loss for the player to move after five days.

## Rule 3

The engine answers every question about the game. Whose move it is is the replayed
position's `side_to_move`; whether a move is legal is `Session.play`, which asks the core;
how the game ends on the board is the core's `terminal`. The two endings a person
decides -- a resignation and an agreed draw -- are not rules of the game, and a
cancellation is the absence of one. Nothing here works out a position.

## Rated, or practice

A game whose rules have computer players with ratings (a loaded picker ladder) is a
leaderboard row: both people's rows on that game's table move at the finish, exactly as
`rating.finish` already moves two seats. A game without one is a sandbox row, which this
module calls *practice*: kept, readable by both, rated nowhere, and with no take-back or
restart (`matches._sandbox_only` refuses both on a row with two people). Neither reaches a
game's score: nothing under `metrics/` reads the store at all
(`tests/test_site_correspondence.py`).

## The time left to move, as a fact

The five days are `quota.CORRESPONDENCE_IDLE`, applied by `matches.sweep` from the last
move (or the start). The user, 2026-10-02 (`docs/plan-return.md` §9): *"there should be an
indication how long there is left to move ... It would be incredibly annoying for it to be
a surprise lose on time."* So `view` and the profile say it, worked out here and only here
from the same instant the sweep reads (`coalesce(moved, started)`) and the database's
clock: *You have 4 days 6 hours to move.* It is the dated exception to test 2 of
`docs/plan-meta-game.md` and stays a plain statement: whole hours, rounded down so it never
promises time that is not there, and no finer than an hour under one day, so a page that
asks every 15 seconds never ticks. The page is handed words, never a deadline to count
down to.

## Who may read one

A running game between two people is readable by its two players only; spectating is in
`docs/limbo/site.md`. A finished one is readable by any member (§1.9). Anybody else is
told there is no game at the address, which is the answer a wrong id gets, so a running
game's existence is not confirmed to somebody outside it.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from chess_variant_sandbox.site import identity, matches, quota, rating
from chess_variant_sandbox.site.matches import AGREED, BETWEEN_PEOPLE, CANCELLED, MatchError, Row
from chess_variant_sandbox.web import draw_offer

PRACTICE = matches.SANDBOX
"""The mode a game between two people takes on rules with no rated computer players."""

PAUSED = "correspondence-paused"
"""The `switches` row `/admin` writes to pause new games between people (§1.7)."""

PAUSED_SAID = "New games between people are paused for now. Your running games carry on."
"""What a member reads while the pause stands, in the plan's own words (§1.7)."""

NOT_HERE = "There is no game at this address."
"""What somebody outside a running game between two people is told: a wrong id's answer."""

RULES = (
    "There is no clock: each of you moves when you like.",
    "A move you pick is held until you send it, and a sent move is final.",
    "Either of you may offer a draw; it stands until the other accepts it, declines it or "
    "moves. Each of you may offer one in any five of your own moves.",
    "Either of you may cancel the game until both of you have moved; it then counts for "
    "nothing.",
    f"The player to move loses after {quota.CORRESPONDENCE_IDLE_SAID}.",
)
"""The rules of a correspondence game as the page states them under the moves (test 7)."""


# ------------------------------------------------------------------------- reading


def between_people(found: Row) -> bool:
    """Whether both seats hold a person (a member, or a former one)."""
    return (
        found.seat0 is not None
        and found.seat1 is not None
        and found.seat0.user is not None
        and found.seat1.user is not None
    )


def seat_of(found: Row, user: int | None) -> int:
    """This person's seat in a game between two people, or the refusal that says why not."""
    if not between_people(found):
        raise MatchError("this is not a game between two people")
    seat = -1 if user is None else found.seat_of(rating.person(user))
    if seat < 0:
        raise MatchError("only the two players of this game can act in it")
    return seat


def may_read(found: Row, viewer: int | None) -> bool:
    """Whether this viewer may see this match. Every other kind of row is unchanged."""
    if not between_people(found):
        return True
    if found.running:
        return viewer is not None and found.seat_of(rating.person(viewer)) >= 0
    return viewer is not None


@dataclass(frozen=True)
class Offer:
    """A standing draw offer: the seat that made it, and the ply it was made at."""

    seat: int
    ply: int


def offer_of(connection: Any, match_id: str) -> Offer | None:
    with connection.cursor() as cursor:
        cursor.execute(
            "select draw_offer, draw_offer_ply from matches where id = %s", (match_id,)
        )
        found = cursor.fetchone()
    if found is None or found[0] is None:
        return None
    return Offer(seat=int(found[0]), ply=int(found[1]))


def times_of(connection: Any, match_id: str) -> list[str | None]:
    """When each move landed, as ISO text, `None` for a move with no time kept."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select jsonb_array_length(moves), times from matches where id = %s",
            (match_id,),
        )
        found = cursor.fetchone()
    if found is None:
        return []
    kept = [None if when is None else str(when) for when in (found[1] or [])]
    plies = int(found[0])
    return (kept + [None] * plies)[:plies]


# ------------------------------------------------------------- the time left to move


def now(connection: Any) -> datetime:
    """The database's clock: the one `matches.sweep` measures idleness against."""
    with connection.cursor() as cursor:
        cursor.execute("select now()")
        found = cursor.fetchone()
    return found[0]


def time_left(found: Row, at: datetime) -> timedelta:
    """How long the player to move has before the sweep resigns the game for them.

    Measured from the instant the sweep measures from (the last move, or the start), so
    the page and the rule cannot disagree. Zero or less once the five days are up and the
    sweep has not run yet.
    """
    return (found.moved or found.started) + quota.CORRESPONDENCE_IDLE - at


def _count(number: int, unit: str) -> str:
    return f"{number} {unit}{'' if number == 1 else 's'}"


def duration_said(left: timedelta) -> str:
    """A time left in words, rounded down to the hour: *4 days 6 hours*, *2 days*,
    *5 hours*, *less than an hour*. Never finer than an hour, so it does not tick."""
    hours = int(left.total_seconds() // 3600)
    if hours < 1:
        return "less than an hour"
    days, hours = divmod(hours, 24)
    if not days:
        return _count(hours, "hour")
    return _count(days, "day") + (f" {_count(hours, 'hour')}" if hours else "")


def to_move_said(left: timedelta, who: str = "You") -> str:
    """The sentence: *You have 4 days 6 hours to move.*, *bo has 2 days to move.*

    `who` is *You*, *They* or a handle. Once the time is up (and before the sweep has
    run) it says so as plainly: the game is about to end, as the rule says.
    """
    have = "have" if who in ("You", "They") else "has"
    if left <= timedelta(0):
        return f"{who} {have} no time left to move: the game is about to end as a loss on time."
    return f"{who} {have} {duration_said(left)} to move."


def started_order(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Your started games in the order your page lists them (Q627).

    First the games where it is your move, by your time left, least
    first (the most urgent; the user, 2026-10-02), then the games waiting on the other person, by their time left,
    least first. Each entry carries `yourMove` and `left` (seconds); the id breaks a tie so
    two readings agree.
    """
    return sorted(
        entries,
        key=lambda entry: (
            not entry["yourMove"],
            entry["left"],
            entry["id"],
        ),
    )


def view(
    connection: Any,
    found: Row,
    viewer: int | None,
    mover: int | None = None,
    at: datetime | None = None,
) -> dict[str, Any]:
    """What the match page needs about a game between two people, beside the snapshot.

    `you` is the viewer's seat, or None for a member reading a finished game. Times are
    when things happened; `toMove` is the one exception, the time the player to move has
    left (`mover`, the engine's side to move), as a sentence and only while the game
    runs. `at` is the clock, the database's unless a test sets it.
    """
    you = -1 if viewer is None else found.seat_of(rating.person(viewer))
    offer = offer_of(connection, found.id)
    seated = [found.player(seat).user or 0 for seat in (0, 1)]
    players = [identity.handle_of(connection, user) for user in seated]
    return {
        "you": you if you >= 0 else None,
        "players": players,
        # Each seat's generated mark (R2), drawn beside its handle; None for a former player.
        "marks": [identity.mark_of(user, handle) for user, handle in zip(seated, players, strict=True)],
        "rated": found.mode == matches.LEADERBOARD,
        "drawOffer": None if offer is None or not found.running else offer.seat,
        # Whether the viewer may make an offer now, and why not (Q790): accepting the
        # other's standing offer is always open, a second offer of one's own is not.
        **_may_offer(connection, found, you, offer),
        "mayCancel": found.running and not _both_moved(connection, found.id),
        "times": times_of(connection, found.id),
        "started": found.started.isoformat(),
        "left": found.left,
        "toMove": _to_move(connection, found, you, players, mover, at),
        "rules": list(RULES),
    }


def _may_offer(
    connection: Any, found: Row, you: int, offer: Offer | None
) -> dict[str, Any]:
    """`mayOffer` and `offerWhy` for the viewer's seat (Q790, the one-in-ten rule)."""
    if you < 0 or not found.running:
        return {"mayOffer": False, "offerWhy": None}
    if offer is not None:
        mine = offer.seat == you
        return {
            "mayOffer": not mine,
            "offerWhy": "Your offer stands until the other player answers it." if mine else None,
        }
    refusal = draw_offer.gap_refusal(matches.draw_offers(connection, found.id), you, found.plies)
    return {"mayOffer": refusal is None, "offerWhy": refusal}


def _to_move(
    connection: Any,
    found: Row,
    you: int,
    players: list[str],
    mover: int | None,
    at: datetime | None,
) -> str | None:
    """The time left sentence for the match page, said to either seat: *You have …* to
    the player to move, *ada has …* to the one waiting and to a reader."""
    if not found.running or mover is None or mover not in (0, 1):
        return None
    left = time_left(found, now(connection) if at is None else at)
    return to_move_said(left, "You" if mover == you else players[mover])


def _both_moved(connection: Any, match_id: str) -> bool:
    """Whether each seat has made a move, as the replayed history records who moved."""
    history = matches.session(connection, match_id).history
    return {int(entry["player"]) for entry in history} >= {0, 1}


def running_for(connection: Any, user: int) -> tuple[Row, ...]:
    """This person's games between two people still being played, oldest first."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {matches._ROW_COLUMNS} from matches"
            f" where {BETWEEN_PEOPLE} and finished is null"
            f" and (seat0_user = %s or seat1_user = %s) order by started, id",
            (user, user),
        )
        return tuple(matches._to_row(found) for found in cursor.fetchall())


def finished_line(connection: Any, user: int) -> str | None:
    """The abandonment line (§1.6), as what a person finished rather than what they dropped.

    Counted over this person's finished games between two people, cancelled ones aside.
    A game *dropped* is one the sweep resigned for them (`left`, and they lost it). None
    until there is a finished game, so the line never reads *0 of 0* (test 5).
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select count(*), count(*) filter (where \"left\" and ("
            f"   (seat0_user = %(user)s and result = 0)"
            f"   or (seat1_user = %(user)s and result = 1)))"
            f" from matches where {BETWEEN_PEOPLE} and finished is not null"
            f" and result is not null and (seat0_user = %(user)s or seat1_user = %(user)s)",
            {"user": user},
        )
        total, dropped = cursor.fetchone()
    total, dropped = int(total), int(dropped)
    if total == 0:
        return None
    if dropped == 0:
        if total == 1:
            return "Finished the one correspondence game played."
        return f"Finished all {total} correspondence games played."
    return f"Finished {total - dropped} of {total} correspondence games played."


# ------------------------------------------------------------------------ the switch


def paused(connection: Any) -> bool:
    """Whether new games between people are paused (§1.7)."""
    with connection.cursor() as cursor:
        cursor.execute("select 1 from switches where name = %s", (PAUSED,))
        return cursor.fetchone() is not None


def set_paused(connection: Any, on: bool, by: int | None = None) -> None:
    """Pause or resume new games between people. Games in flight are never touched."""
    with connection.cursor() as cursor:
        if on:
            cursor.execute(
                "insert into switches (name, set_by) values (%s, %s)"
                " on conflict (name) do nothing",
                (PAUSED, by),
            )
        else:
            cursor.execute("delete from switches where name = %s", (PAUSED,))


def running_count(connection: Any) -> int:
    """How many games between people are being played across the site, for `/admin`."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select count(*) from matches where {BETWEEN_PEOPLE} and finished is null"
        )
        found = cursor.fetchone()
    return 0 if found is None else int(found[0])


# ------------------------------------------------------------------------- starting


def start(
    connection: Any,
    game: str,
    players: Sequence[int],
    by: int | None = None,
    seed: int = 0,
) -> Row:
    """Seat two people at one game, in board order, and write the row.

    The one door: a challenge accepted (`site/challenges.py`, Q571) and `/admin`'s
    pairing both call this.
    `by` is whoever asked for the game (the challenger, or the maintainer pairing two
    people from `/admin`), whose daily allowance of matches it counts against; it
    defaults to the first seat.

    Refused, in a sentence, while the pause stands, for a person playing themself, for an
    account that cannot sign in, and for a person already at `quota.CORRESPONDENCE_AT_ONCE`
    games in flight. The mode is the game's, never a choice: rated on rules with rated
    computer players, practice elsewhere (§1.3).
    """
    if len(players) != 2:
        raise MatchError("a game between people has two of them")
    first, second = (int(players[0]), int(players[1]))
    if first == second:
        raise MatchError("a game between people needs two different people")
    if paused(connection):
        raise MatchError(PAUSED_SAID)
    for person in (first, second):
        _member(connection, person)
    matches._game_row(connection, game, starting=True)
    quota.check_new_match(connection, first if by is None else int(by))
    for person in (first, second):
        if len(running_for(connection, person)) >= quota.CORRESPONDENCE_AT_ONCE:
            whose = identity.handle_of(connection, person)
            raise MatchError(
                f"{whose} already has {quota.CORRESPONDENCE_AT_ONCE} games between people "
                f"running, which is the most at once; one of them has to finish first"
            )
    mode = mode_for(connection, game)
    match_id = secrets.token_urlsafe(9)
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into matches (id, game, mode, seed, seat0_user, seat1_user, times)"
            " values (%s, %s, %s, %s, %s, %s, '[]'::jsonb)",
            (match_id, game, mode, seed, first, second),
        )
    return matches.row(connection, match_id)


def mode_for(connection: Any, game: str) -> str:
    """Rated where the game's rules have rated computer players, practice elsewhere (§1.3).

    The game's, never a choice: the challenge sheet says which before anybody accepts.
    """
    if rating.computers(rating.load(connection, game)):
        return matches.LEADERBOARD
    return PRACTICE


def _member(connection: Any, user: int) -> None:
    """Refuse an id that is not a member who can sign in now."""
    with connection.cursor() as cursor:
        cursor.execute("select revoked, forgotten from users where id = %s", (user,))
        found = cursor.fetchone()
    if found is None:
        raise MatchError(f"there is no member {user}")
    if found[0] is not None or found[1] is not None:
        raise MatchError("that member can no longer play, so no game can be started with them")


# -------------------------------------------------------------------------- playing


def _running(connection: Any, match_id: str, user: int | None) -> tuple[Row, int]:
    found = matches.row(connection, match_id)
    seat = seat_of(found, user)
    if not found.running:
        raise MatchError("this game is over")
    return found, seat


def move(connection: Any, match_id: str, user: int | None, chosen: int) -> dict[str, Any]:
    """A sent move: checked as this person's turn by the engine, then played by it.

    Whose turn it is is the replayed position's side to move, and the move's legality is
    the engine's (`Session.play`). The time it landed is kept beside it, and a draw offer
    made by the other person lapses, because moving is how an offer is declined (§1.3).
    """
    found, seat = _running(connection, match_id, user)
    position = matches.session(connection, match_id)
    if position.state.side_to_move != seat:
        raise MatchError("it is not your move; the other player's move comes first")
    snapshot = matches._advance(connection, match_id, lambda game: game.play(chosen))
    offer = offer_of(connection, match_id)
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set times = coalesce(times, '[]'::jsonb)"
            " || jsonb_build_array(to_jsonb(now())) where id = %s",
            (match_id,),
        )
        if offer is not None and offer.seat != seat:
            cursor.execute(
                "update matches set draw_offer = null, draw_offer_ply = null where id = %s",
                (match_id,),
            )
    return snapshot


def offer_draw(connection: Any, match_id: str, user: int | None) -> Offer | None:
    """Offer a draw. Offering into the other person's standing offer accepts it.

    Returns the offer that now stands, or None when the offer was an acceptance and the
    game is over.
    """
    found, seat = _running(connection, match_id, user)
    standing = offer_of(connection, match_id)
    if standing is not None and standing.seat != seat:
        accept_draw(connection, match_id, user)
        return None
    if standing is not None:
        return standing
    # One offer in any ten plies a side (Q790, §e.1 of
    # `docs/reference/intransitive-quiet-endgame.md`), counted from the last one made.
    refusal = draw_offer.gap_refusal(matches.draw_offers(connection, match_id), seat, found.plies)
    if refusal is not None:
        raise MatchError(refusal)
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set draw_offer = %s, draw_offer_ply = %s where id = %s",
            (seat, found.plies, match_id),
        )
    matches.note_draw_offer(connection, match_id, {"seat": seat, "ply": found.plies})
    return Offer(seat=seat, ply=found.plies)


def accept_draw(connection: Any, match_id: str, user: int | None) -> Row:
    """Accept the other person's standing offer: a draw, with the reason *agreed*."""
    _found, seat = _running(connection, match_id, user)
    standing = offer_of(connection, match_id)
    if standing is None or standing.seat == seat:
        raise MatchError("there is no draw offer from the other player to accept")
    _clear_offer(connection, match_id)
    matches._finish(connection, match_id, result=0.5, reason=AGREED, left=False)
    return matches.row(connection, match_id)


def decline_draw(connection: Any, match_id: str, user: int | None) -> Row:
    """Decline the other person's standing offer (Q790). The game goes on as it was.

    Moving declines it too, as it always has; this is the same answer without a move, so
    the person to move is not made to choose a move to say no. The offerer may offer
    again once ten plies have passed since the offer (`draw_offer.GAP`).
    """
    _found, seat = _running(connection, match_id, user)
    standing = offer_of(connection, match_id)
    if standing is None or standing.seat == seat:
        raise MatchError("there is no draw offer from the other player to decline")
    _clear_offer(connection, match_id)
    return matches.row(connection, match_id)


def resign(connection: Any, match_id: str, user: int | None) -> Row:
    """Give the game up. A finish they chose, so it is not a game they left."""
    _running(connection, match_id, user)
    _clear_offer(connection, match_id)
    return matches.resign(connection, match_id, by=user, left=False)


def cancel(connection: Any, match_id: str, user: int | None) -> Row:
    """Call the game off before both people have moved (§1.3).

    The row stays, finished with no result and `reason = cancelled`; nothing is rated,
    kept as a finish or granted, because `matches._finish` is never called.
    """
    _running(connection, match_id, user)
    if _both_moved(connection, match_id):
        raise MatchError(
            "both of you have moved, so the game can no longer be cancelled; it can be "
            "resigned or drawn by agreement"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set finished = now(), reason = %s, result = null,"
            " draw_offer = null, draw_offer_ply = null where id = %s",
            (CANCELLED, match_id),
        )
    return matches.row(connection, match_id)


def _clear_offer(connection: Any, match_id: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set draw_offer = null, draw_offer_ply = null where id = %s",
            (match_id,),
        )


def leave(connection: Any, user: int) -> tuple[Row, ...]:
    """Resign every running game between people this person is in, as games they left.

    What leaving the site and being revoked from `/admin` both do (§1.6): the other
    player is not left waiting five days for somebody who cannot come back.
    """
    return tuple(
        matches.resign(connection, found.id, by=user, left=True)
        for found in running_for(connection, user)
    )


def member_named(connection: Any, handle: str) -> int:
    """The id of the member with this handle, for `/admin`'s pairing of two people."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select id from users where lower(handle) = lower(%s)", (handle.strip(),)
        )
        found = cursor.fetchone()
    if found is None:
        raise MatchError(f"there is no member called {handle.strip()!r}")
    return int(found[0])
