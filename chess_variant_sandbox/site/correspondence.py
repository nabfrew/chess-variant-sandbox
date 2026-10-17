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

## A game in rounds (Q918f, `docs/plan-simultaneous.md` §6.6)

On rules that play in rounds (schema 26) neither player waits for the other's move: each
**seals** one for the round's position, in secret, and both are made at once by the core
when the second is in. The first one in waits in `matches.sealed_move`
(`0036_sealed_moves.sql`), which this module alone reads, never in `moves`, so no replay,
record, finish or route that rebuilds a position can show it. The position a reader is
handed is built for them (`snapshot_for`): the board offers their own moves while they
owe one, says of the other seat only *sealed* or not, and gives the sealed move back to
the seat that sealed it and to nobody else.

**The reveal is one transaction.** Every seal locks the row first (`select ... for
update`), then reads what is sealed. Of two seals racing, the second waits on the first's
lock and then finds its move, so exactly one of them reveals: both moves appended to
`moves` in seat order through `matches._advance` (the engine replays them and resolves the
round), `sealed_move` cleared, a standing draw offer lapsed, each move's time kept.

**A sealed move is final**, as a sent move is in a game that alternates; the hold before
the seal is where it can be changed, on the page. **The clock** is the round's: it starts
when the round opens (the last reveal, or the start) and a seal does not move it, so the
sweep (`sweep_round`) resigns, after `quota.CORRESPONDENCE_IDLE`, the one player who has
not sealed, and draws the game if neither has.
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
from chess_variant_sandbox.web.library import engine_for

PRACTICE = matches.SANDBOX
"""The mode a game between two people takes on rules with no rated computer players."""

PAUSED = "correspondence-paused"
"""The `switches` row `/admin` writes to pause new games between people (§1.7)."""

PAUSED_SAID = "New games between people are paused for now. Your running games carry on."
# Schema 27: a spy game is played against the computer or at one screen (Q935c); between two
# people its announced rounds need the announcement shown to both and the answer revealed
# under the row's lock, which is Q935e's (`docs/plan-spy.md` §3.7). Refused by name until then.
SPY_SAID = (
    "a game with a spy cannot be played between people yet: its announced rounds, where one "
    "player's move is shown to the other before they answer, are still to come here"
)


def spy_refusal(spec: dict[str, Any]) -> str | None:
    """`SPY_SAID` for a spy game's rules, None for any other (Q935c)."""
    from chess_variant_sandbox.spec.schema import is_spy_game

    return SPY_SAID if is_spy_game(spec) else None

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

ROUND_RULES = (
    "There is no clock: each of you seals your move for a round when you like.",
    "Each round you both choose a move for the same position. A move you pick is held "
    "until you seal it, and a sealed move is final; neither of you sees the other's until "
    "both are in, and then both are made at once.",
    "Either of you may offer a draw; it stands until the other accepts it or declines it, "
    "or the round is revealed. Each of you may offer one in any five of your own moves.",
    "Either of you may cancel the game until the first round is revealed; it then counts "
    "for nothing.",
    f"After {quota.CORRESPONDENCE_IDLE_SAID} in a round, a player who has not sealed one "
    "loses; if neither of you has, the game is drawn.",
)
"""`RULES` for a game played in rounds (Q918f): the seal, the reveal and the round's clock."""

UNSEALED = "unsealed"
"""The store's reason for a game in rounds the sweep drew: neither player sealed a move in
time (Q918f). A thing two people did not do, like `resigned`, not a way a position ended."""


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


# ------------------------------------------------------------ a game in rounds (Q918f)


@dataclass(frozen=True)
class Sealed:
    """The one move sealed in a round not yet revealed: whose, and which."""

    seat: int
    move: int


def _sealed(connection: Any, match_id: str, lock: bool = False) -> Sealed | None:
    """The move sealed in this row's open round, or None. `lock` takes the row's lock
    first (`for update`), which every seal and the sweep's round arm do before they read
    it, so whatever they then decide is decided once."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select sealed_move from matches where id = %s" + (" for update" if lock else ""),
            (match_id,),
        )
        found = cursor.fetchone()
    if found is None or found[0] is None:
        return None
    return Sealed(seat=int(found[0]["seat"]), move=int(found[0]["move"]))


def sealed_seat(connection: Any, match_id: str) -> int | None:
    """Which seat has sealed in the open round, or None: the seat alone, never the move."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select (sealed_move ->> 'seat')::int from matches where id = %s", (match_id,)
        )
        found = cursor.fetchone()
    return None if found is None or found[0] is None else int(found[0])


def _drop_sealed(connection: Any, match_id: str) -> None:
    """Forget a sealed move at a finish: a round never revealed stays unread."""
    with connection.cursor() as cursor:
        cursor.execute("update matches set sealed_move = null where id = %s", (match_id,))


def in_rounds(position: Any) -> bool:
    """Whether this replayed game plays in rounds, as the compiled rules say (schema 26)."""
    return bool(position.engine.simultaneous)


def owing(connection: Any, found: Row) -> tuple[int, ...]:
    """The seats that owe a move now: the engine's side to move in a game that alternates;
    in a game in rounds, each seat that has not sealed this round. Empty once it is over."""
    if not found.running:
        return ()
    position = matches.session(connection, found.id)
    if in_rounds(position):
        sealed = sealed_seat(connection, found.id)
        return tuple(seat for seat in (0, 1) if seat != sealed)
    return (int(position.state.side_to_move),)


def snapshot_for(connection: Any, found: Row, viewer: int | None) -> dict[str, Any]:
    """The match as this reader may see it, with a game between two people's own block.

    The one place a snapshot of a game between two people is made for a route. In a game
    in rounds it is the reader's: the board offers their moves while they owe one, a
    move the other seat sealed is in it only as `round.sealed`'s boolean, and the reader's
    own sealed move comes back to them alone (`round.yours`).
    """
    position = matches.session(connection, found.id)
    between = between_people(found)
    if between and in_rounds(position):
        seat = -1 if viewer is None else found.seat_of(rating.person(viewer))
        position.viewer = seat
        sealed = _sealed(connection, found.id) if found.running else None
        if sealed is not None and sealed.seat == seat:
            position.pending[seat] = sealed.move
        elif sealed is not None:
            position.sealed_unseen = (sealed.seat == 0, sealed.seat == 1)
    snapshot = position.snapshot()
    if between:
        snapshot["correspondence"] = view(
            connection,
            matches.row(connection, found.id),
            viewer,
            mover=snapshot.get("sideToMove"),
            rounds="round" in snapshot,
        )
    return snapshot


def view(
    connection: Any,
    found: Row,
    viewer: int | None,
    mover: int | None = None,
    at: datetime | None = None,
    rounds: bool = False,
) -> dict[str, Any]:
    """What the match page needs about a game between two people, beside the snapshot.

    `you` is the viewer's seat, or None for a member reading a finished game. Times are
    when things happened; `toMove` is the one exception, the time the player to move has
    left (`mover`, the engine's side to move), as a sentence and only while the game
    runs. `at` is the clock, the database's unless a test sets it. `rounds` is a game in
    rounds (Q918f): its rules are `ROUND_RULES`, and `mover` is the reader while their
    move is not in, else the seat the round waits on.
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
        "rules": list(ROUND_RULES if rounds else RULES),
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
    _name, spec = matches._game_row(connection, game, starting=True)
    refused = spy_refusal(spec)
    if refused is not None:
        raise MatchError(refused)
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

    The game's, never a choice: the challenge sheet says which before anybody accepts. A
    preview is practice whatever its table holds (Q931, `matches.is_preview`).
    """
    if matches.is_preview(connection, game):
        return PRACTICE
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

    In a game in rounds the move is a seal (`seal`).
    """
    found, seat = _running(connection, match_id, user)
    if _plays_in_rounds(connection, found):
        return seal(connection, match_id, user, chosen)
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


def _plays_in_rounds(connection: Any, found: Row) -> bool:
    """Whether this row's game plays in rounds, from its compiled rules, without a replay."""
    _name, spec = matches._game_row(connection, found.game)
    return bool(engine_for(spec).simultaneous)


def seal(connection: Any, match_id: str, user: int | None, chosen: int) -> dict[str, Any]:
    """This person's move for the round now open, sealed; the round revealed if the other
    person's is in already (Q918f, `docs/plan-simultaneous.md` §6.6).

    The row is locked before anything is read, so of two seals racing the second waits for
    the first to commit and then finds its move: one reveal, never two, and never a round
    revealed with one move. Whether the move is legal is the engine's, asked through the
    replayed position drawn for this seat (`Session.seal`, which refuses a move the round's
    position does not offer it; its answer is thrown away). A first seal is
    stored in `sealed_move` with its time and nothing else changes: not `moves`, not the
    round's clock (`moved`), not a draw offer. The second reveals through `_advance`, which
    replays both moves in seat order, so the engine resolves the round and a finish is
    recorded as any finish is; then the seal is cleared, each move's time kept in seat
    order, and a standing draw offer lapses.

    Answers the position after it as the sealer may see it (`snapshot_for`).
    """
    standing = _sealed(connection, match_id, lock=True)
    found, seat = _running(connection, match_id, user)
    if standing is not None and standing.seat == seat:
        raise MatchError(
            "your move for this round is sealed already; it is revealed with the other "
            "player's, once theirs is in"
        )
    position = matches.session(connection, match_id)
    position.viewer = seat
    position.seal(int(chosen))  # the engine's check: a ValueError for an illegal move
    if standing is None:
        with connection.cursor() as cursor:
            cursor.execute(
                "update matches set sealed_move = jsonb_build_object("
                "'seat', %s::int, 'move', %s::int, 'at', to_jsonb(now())) where id = %s",
                (seat, int(chosen), match_id),
            )
        return snapshot_for(connection, matches.row(connection, match_id), user)
    pair = (standing.move, int(chosen)) if standing.seat == 0 else (int(chosen), standing.move)

    def reveal(game: Any) -> dict[str, Any]:
        for move in pair:
            game.replay(move)
        snapshot: dict[str, Any] = game.snapshot()
        return snapshot

    matches._advance(connection, match_id, reveal)
    first_at = "sealed_move -> 'at'" if standing.seat == 0 else "to_jsonb(now())"
    second_at = "to_jsonb(now())" if standing.seat == 0 else "sealed_move -> 'at'"
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set times = coalesce(times, '[]'::jsonb)"
            f" || jsonb_build_array({first_at}, {second_at}),"
            " sealed_move = null, draw_offer = null, draw_offer_ply = null where id = %s",
            (match_id,),
        )
    return snapshot_for(connection, matches.row(connection, match_id), user)


def sweep_round(connection: Any, found: Row, between: timedelta) -> Row | None:
    """The sweep's arm for a game in rounds between two people (Q918f): after `between`
    from the round's opening, the one player who has not sealed loses it, as a game they
    left; if neither has, it is drawn, `reason = unsealed`, left by both.

    The row is locked and the deadline read again under the lock, so a seal that revealed
    the round while the sweep was on its way (and so started a new one) is not undone.
    None when the row is no longer due.
    """
    standing = _sealed(connection, found.id, lock=True)
    with connection.cursor() as cursor:
        cursor.execute(
            "select finished is null and coalesce(moved, started) < now() - %s"
            " from matches where id = %s",
            (between, found.id),
        )
        due = cursor.fetchone()
    if due is None or not due[0]:
        return None
    _drop_sealed(connection, found.id)
    _clear_offer(connection, found.id)
    if standing is None:
        matches._finish(connection, found.id, result=0.5, reason=UNSEALED, left=True)
        return matches.row(connection, found.id)
    absent = found.player(1 - standing.seat).user
    return matches.resign(connection, found.id, by=absent, left=True)


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
    _drop_sealed(connection, match_id)
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
    _drop_sealed(connection, match_id)
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
            " draw_offer = null, draw_offer_ply = null, sealed_move = null where id = %s",
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
    resigned = []
    for found in running_for(connection, user):
        _drop_sealed(connection, found.id)
        resigned.append(matches.resign(connection, found.id, by=user, left=True))
    return tuple(resigned)


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
