"""Finding an opponent for a game between two people (Q571, `docs/plan-return.md` §1.4).

Four small ways, sized for ten people who know each other, and nothing for a crowd that
is not there (no lobby, no pairing queue, no matchmaking, no *online now*):

1. **Open to a game of this.** A member ticks a box on a game's page and their handle is
   listed there for other members, with their rating on that game or *hasn't played this
   one yet* (test 5: never a low number for somebody who has done nothing). Per person,
   per game, **off by default**, and it never lapses (test 1). Ticking it sends nothing to
   anybody.
2. **A challenge by handle**: any member, named in a box that suggests members' handles.
3. **A challenge link** (`/c/<token>`): one game, one seat, works once, for one member.
   The site sends nothing; the challenger sends the link however they like. Opened by
   somebody without an account it says whose challenge it is and that accounts are by
   invitation: it is never an invitation itself.
4. **Rematch** in a finished game's banner, which is a challenge to the same person with
   the seats swapped.
5. **The maker takes challengers** (R10, Q606): a designer who switches it on, on their
   own page, is named first in *Play a person* on each of their published games, and is
   challenged there in the same way as anybody else (`maker`). A challenge is never more
   than a game, a seat and a person or a link: there is no message to write.

A challenge waits until it is accepted, declined or withdrawn, and **never expires**
(test 2). Accepting starts the match through `correspondence.start`, Q570's one door, so
every refusal that door has (the pause, the cap on games at once, an account that can no
longer sign in) holds here too and leaves the challenge waiting. Declining tells the
challenger nothing beyond the challenge showing as *declined* on their page.

## Rule 3

Nothing here is about the game's rules. Who moves first is the challenger's choice or the
match's seed, whether the game is rated is the game's ladder (`correspondence.mode_for`),
and everything after the start is the engine's, as for any match.

## Abuse limits

`quota.CHALLENGES_A_DAY` challenges sent in a rolling day and `quota.CHALLENGES_WAITING`
waiting at once, a person; one waiting challenge from one person to another on one game.
Each is counted under the challenger's lock (`quota._hold`), as a match start is.

## Leaving

A member who leaves or is revoked has every waiting challenge they sent or were sent
withdrawn, and their *open to* ticks removed (`leave`, called where `correspondence.leave`
is).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from chess_variant_sandbox.site import correspondence, identity, matches, quota, rating, stages
from chess_variant_sandbox.site.matches import MatchError

SEATS: dict[str, int | None] = {"first": 0, "second": 1, "either": None}
"""The challenger's seat as the page names it: moving first, second, or either, drawn."""

ACCEPTED = "accepted"
DECLINED = "declined"
WITHDRAWN = "withdrawn"

NO_LINK = "There is no challenge at this address."
"""What a token nobody made is answered with, as a wrong address is."""

DECLINED_SHOWN = timedelta(days=7)
"""How long a declined challenge stays on the challenger's page, marked *declined*.

The plan's only word on a decline is that it leaves the *sent* list as *declined* (§1.4);
a week is long enough to be seen by somebody who looks in now and then, and it is a
tidy-up of a list, not a lapse of anything still waiting.
"""

LINK_PATH = "/c/"
"""Where a challenge link lives: `LINK_PATH + token`."""


@dataclass(frozen=True)
class Challenge:
    """One row of `challenges`."""

    id: int
    challenger: int
    challenged: int | None
    token: str | None
    game: str
    seat: int | None
    rematch_of: str | None
    made: datetime
    answer: str | None
    answered: datetime | None
    match: str | None

    @property
    def waiting(self) -> bool:
        return self.answer is None

    @property
    def link(self) -> bool:
        return self.token is not None


_COLUMNS = (
    "id, challenger, challenged, token, game, seat, rematch_of, made, answer, answered, match"
)


def _to_challenge(found: Any) -> Challenge:
    return Challenge(
        id=int(found[0]),
        challenger=int(found[1]),
        challenged=None if found[2] is None else int(found[2]),
        token=found[3],
        game=str(found[4]),
        seat=None if found[5] is None else int(found[5]),
        rematch_of=found[6],
        made=found[7],
        answer=found[8],
        answered=found[9],
        match=found[10],
    )


def _select(connection: Any, where: str, params: Any) -> list[Challenge]:
    with connection.cursor() as cursor:
        cursor.execute(f"select {_COLUMNS} from challenges where {where}", params)
        return [_to_challenge(found) for found in cursor.fetchall()]


def find(connection: Any, challenge_id: int) -> Challenge:
    found = _select(connection, "id = %s", (int(challenge_id),))
    if not found:
        raise KeyError(f"there is no challenge {challenge_id}")
    return found[0]


def by_token(connection: Any, token: str) -> Challenge:
    found = _select(connection, "token = %s", (str(token),))
    if not found:
        raise KeyError(NO_LINK)
    return found[0]


# ------------------------------------------------------------------- open to a game


def set_open(connection: Any, user: int, game: str, on: bool) -> bool:
    """Tick or untick *open to a game of this* for this person on this game.

    Unticking always works. Ticking is refused while new games between people are paused
    (the lists are hidden then) and on a game no new match can start on.
    """
    with connection.cursor() as cursor:
        if not on:
            cursor.execute(
                "delete from open_to where person = %s and game = %s", (user, game)
            )
            return False
    if correspondence.paused(connection):
        raise MatchError(correspondence.PAUSED_SAID)
    correspondence._member(connection, user)
    matches._game_row(connection, game, starting=True)
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into open_to (person, game) values (%s, %s) on conflict do nothing",
            (user, game),
        )
    return True


def is_open(connection: Any, user: int, game: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "select 1 from open_to where person = %s and game = %s", (user, game)
        )
        return cursor.fetchone() is not None


def open_games(connection: Any, user: int) -> list[dict[str, str]]:
    """The games this person ticked, oldest tick first, each with its name."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select o.game, g.name from open_to o join games g on g.hash = o.game"
            " where o.person = %s order by o.since, o.game",
            (user,),
        )
        return [{"game": str(game), "name": str(name)} for game, name in cursor.fetchall()]


def _said(table: rating.Table, user: int) -> str:
    """A listed person's line: their rating on this game, or that they have not played it.

    A rating is shown only for somebody who has finished a rated game on it: a newcomer's
    starting number is nobody's measure, and reads as a deficit (test 5).
    """
    found = table.get(rating.person(user))
    if found is None or found.matches == 0:
        return "hasn't played this one yet"
    return f"rated {round(found.rating)} here"


# --------------------------------------------------------- the maker takes challengers
#
# R10 (`docs/plan-return.md` §3 C4, Q606). One switch on a person's own page,
# `users.takes_challengers`, off by default: switched on, they are named first in *Play a
# person* on each of their own published games, and any other member may challenge them
# there through the ordinary challenge, which carries no message of any kind. A rated win
# over them there earns E11 *Beat the maker* (`badges.BEAT_THE_MAKER`); a practice game
# earns nothing. The switch stays as it is set, and sends nothing.

MAKER_SAID = "List me on the games I made as taking challengers."
MAKER_UNDER = (
    "Off unless you switch it on. On each of your published games, members then see you "
    "first under Play a person and can challenge you there; you accept or decline as with "
    "any challenge. Nothing is sent to anyone, and you can switch it off at any time."
)
NO_GAME_YET = (
    "You have no published game of your own yet. The switch opens once one of your games "
    "is on trial."
)
"""Why the switch is refused to a person with nothing to take challengers on."""


def _games_made(connection: Any, user: int | None = None, game: str | None = None) -> list[
    tuple[str, str, int]
]:
    """`(hash, name, owner)` of the published designed games, by owner or by hash.

    A designed game is told by `origin`, never by its owner (`badges.designer_of`);
    published is any level but Draft (`stages.stage`), and a killed, deleted or retired
    game has no *Play a person* to be listed in.
    """
    where, params = ("owner = %s", (user,)) if game is None else ("hash = %s", (game,))
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, name, owner, origin, visibility, nominated, retired, rested"
            " from games where origin = %s and owner is not null and revoked is null"
            f" and deleted is null and retired is null and {where} order by created, hash",
            (stages.DESIGNED, *params),
        )
        rows = cursor.fetchall()
    found: list[tuple[str, str, int]] = []
    for digest, name, owner, origin, visibility, nominated, retired, rested in rows:
        level = stages.stage({
            "origin": origin, "visibility": visibility, "nominated": nominated,
            "retired": retired, "rested": rested,
        })
        if level != stages.DRAFT:
            found.append((str(digest), str(name), int(owner)))
    return found


def takes_challengers(connection: Any, user: int) -> bool:
    """Whether this person switched on *taking challengers*. Off by default."""
    with connection.cursor() as cursor:
        cursor.execute("select takes_challengers from users where id = %s", (user,))
        found = cursor.fetchone()
    return bool(found and found[0])


def set_takes_challengers(connection: Any, user: int, on: bool) -> bool:
    """Switch it on or off; the switch as it now stands.

    Switching off always works. Switching on is refused while new games between people
    are paused, and to a person with no published game of their own.
    """
    if on:
        if correspondence.paused(connection):
            raise MatchError(correspondence.PAUSED_SAID)
        correspondence._member(connection, user)
        if not _games_made(connection, user=user):
            raise MatchError(NO_GAME_YET)
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set takes_challengers = %s where id = %s", (bool(on), user)
        )
    return takes_challengers(connection, user)


def maker_box(connection: Any, user: int) -> dict[str, Any]:
    """Your own page's switch, with its sentences and the games it lists you on.

    `games` is empty, and the switch is not drawn, for somebody with no published game.
    """
    return {
        "on": takes_challengers(connection, user),
        "said": MAKER_SAID,
        "under": MAKER_UNDER,
        "games": [
            {"game": digest, "name": name}
            for digest, name, _owner in _games_made(connection, user=user)
        ],
    }


def maker(connection: Any, game: str, viewer: int) -> dict[str, Any] | None:
    """*‹handle› made this game and takes challengers*, for *Play a person*, or None.

    Only on a published designed game whose owner switched it on and can still sign in.
    `you` is set when the viewer is the maker: they read that they are listed, and are
    never offered a challenge to themselves.
    """
    found = _games_made(connection, game=game)
    if not found:
        return None
    owner = found[0][2]
    with connection.cursor() as cursor:
        cursor.execute(
            "select handle from users where id = %s and takes_challengers"
            " and handle is not null and revoked is null and forgotten is null",
            (owner,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    handle = str(row[0])
    rated = correspondence.mode_for(connection, game) == matches.LEADERBOARD
    return {
        "handle": handle,
        "mark": identity.mark_of(owner, handle),
        "you": owner == viewer,
        "said": (
            "You made this game and take challengers: members see you here first."
            if owner == viewer else "made this game and takes challengers."
        ),
        # The badge's name and when it is earned, on the page that offers the game (test 7).
        "earns": (
            f"A win over {handle} in a rated game here earns the badge Beat the maker."
            if rated and owner != viewer else None
        ),
    }


def people(connection: Any, game: str, viewer: int) -> dict[str, Any]:
    """*Play a person* on a game's page, for a signed-in member (§1.4, s1 of the mock).

    The members open to a game of this one (never the viewer), whether the viewer is,
    every other member's handle for the box's suggestions, and whether a game here would
    be rated. While new games between people are paused, the lists are empty and the
    sentence says why (§1.7).
    """
    paused = correspondence.paused(connection)
    answer: dict[str, Any] = {
        "paused": paused,
        "pausedSays": correspondence.PAUSED_SAID if paused else None,
        "rated": correspondence.mode_for(connection, game) == matches.LEADERBOARD,
        "mine": is_open(connection, viewer, game),
        "open": [],
        "members": [],
        "maker": None,
        "rules": list(correspondence.RULES),
    }
    if paused:
        return answer
    table = rating.load(connection, game)
    with connection.cursor() as cursor:
        cursor.execute(
            "select u.id, u.handle from open_to o join users u on u.id = o.person"
            " where o.game = %s and u.id <> %s and u.handle is not null"
            " and u.revoked is null and u.forgotten is null order by lower(u.handle)",
            (game, viewer),
        )
        listed = cursor.fetchall()
        cursor.execute(
            "select handle from users where id <> %s and handle is not null"
            " and revoked is null and forgotten is null order by lower(handle)",
            (viewer,),
        )
        members = [str(found[0]) for found in cursor.fetchall()]
    made = maker(connection, game, viewer)
    answer["maker"] = made
    # The maker, listed first under their own line, is not listed a second time.
    shown = set() if made is None else {made["handle"]}
    answer["open"] = [
        {"handle": str(handle), "said": _said(table, int(user)),
         "mark": identity.mark_of(int(user), str(handle))}
        for user, handle in listed
        if str(handle) not in shown
    ]
    answer["members"] = members
    return answer


# ------------------------------------------------------------------------ challenging


def _limits(connection: Any, by: int) -> None:
    """The two abuse limits on sending, counted under the challenger's lock."""
    quota._hold(connection, by)
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) filter (where made > now() - %s),"
            " count(*) filter (where answer is null)"
            " from challenges where challenger = %s",
            (timedelta(days=1), by),
        )
        today, waiting = (int(n) for n in cursor.fetchone())
    if today >= quota.CHALLENGES_A_DAY:
        raise quota.QuotaError(
            f"you have sent {today} challenges in the last day, which is the limit of "
            f"{quota.CHALLENGES_A_DAY}; the count rolls, so the next one opens up as the "
            f"oldest falls out of the day"
        )
    if waiting >= quota.CHALLENGES_WAITING:
        raise quota.QuotaError(
            f"you have {waiting} challenges waiting for an answer, which is the most at "
            f"once; withdraw one on your page, or wait for an answer, to send another"
        )


def challenge(
    connection: Any,
    by: int,
    game: str,
    seat: int | None,
    handle: str | None = None,
    *,
    link: bool = False,
    rematch_of: str | None = None,
) -> Challenge:
    """Send a challenge to a member named by handle, or make a link (one or the other).

    `seat` is the challenger's: 0 to move first, 1 second, None for either, drawn when
    the game starts. Refused, in a sentence, while the pause stands, on a game no new
    match can start on, to oneself, to an account that cannot sign in, twice to one person
    on one game while the first waits, and past the two limits.
    """
    if seat not in (0, 1, None):
        raise MatchError("the seat is first, second or either")
    named = (handle or "").strip()
    if bool(named) == bool(link):
        raise MatchError("a challenge goes to one member by handle, or is a link; not both")
    if correspondence.paused(connection):
        raise MatchError(correspondence.PAUSED_SAID)
    correspondence._member(connection, by)
    matches._game_row(connection, game, starting=True)
    challenged: int | None = None
    if named:
        challenged = correspondence.member_named(connection, named)
        if challenged == by:
            raise MatchError("a challenge goes to somebody else; you cannot play yourself")
        correspondence._member(connection, challenged)
    _limits(connection, by)
    if challenged is not None and _select(
        connection,
        "challenger = %s and challenged = %s and game = %s and answer is null",
        (by, challenged, game),
    ):
        raise MatchError(
            f"you have already challenged {identity.handle_of(connection, challenged)} to "
            f"this game, and it is waiting for an answer on your page"
        )
    token = secrets.token_urlsafe(9) if link else None
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into challenges (challenger, challenged, token, game, seat, rematch_of)"
            f" values (%s, %s, %s, %s, %s, %s) returning {_COLUMNS}",
            (by, challenged, token, game, seat, rematch_of),
        )
        return _to_challenge(cursor.fetchone())


def rematch(connection: Any, match_id: str, by: int) -> Challenge:
    """*Rematch* on a finished game between two people: a challenge with the seats swapped.

    Only one of its two players may ask, only once it is over, and never on a cancelled
    game, which nobody played.
    """
    found = matches.row(connection, match_id)
    seat = correspondence.seat_of(found, by)
    if found.running:
        raise MatchError("a rematch is offered once this game is over")
    if found.reason == matches.CANCELLED:
        raise MatchError("this game was cancelled before it started, so there is nothing to rematch")
    other = found.player(1 - seat).user
    if other is None:
        raise MatchError("the other player has left, so there is nobody to rematch")
    return challenge(
        connection,
        by,
        found.game,
        1 - seat,
        identity.handle_of(connection, other),
        rematch_of=match_id,
    )


# -------------------------------------------------------------------------- answering


def _start(connection: Any, found: Challenge, acceptor: int) -> matches.Row:
    """Start the match a challenge became, through Q570's door, and record the answer.

    *Either* is drawn from the match's seed, so who moves first is reproducible from the
    row. A refusal from `correspondence.start` leaves the challenge waiting.
    """
    seed = secrets.randbelow(2**31)
    first = found.seat if found.seat is not None else seed % 2
    players = [found.challenger, acceptor] if first == 0 else [acceptor, found.challenger]
    made = correspondence.start(connection, found.game, players, by=found.challenger, seed=seed)
    with connection.cursor() as cursor:
        cursor.execute(
            "update challenges set answer = %s, answered = now(), match = %s,"
            " challenged = %s where id = %s",
            (ACCEPTED, made.id, acceptor, found.id),
        )
    return made


def _waiting(found: Challenge) -> None:
    if found.answer == ACCEPTED:
        raise MatchError("this challenge has been taken already")
    if found.answer is not None:
        raise MatchError(f"this challenge was {found.answer}, so it can no longer be answered")


def accept(connection: Any, challenge_id: int, user: int) -> matches.Row:
    """The challenged person accepts a named challenge: the match starts."""
    found = find(connection, challenge_id)
    if found.challenged != user or found.link:
        raise KeyError(f"there is no challenge {challenge_id} for you")
    _waiting(found)
    return _start(connection, found, user)


def accept_link(connection: Any, token: str, user: int) -> matches.Row:
    """A member accepts a challenge link: the match starts, and the link is used up."""
    found = by_token(connection, token)
    _waiting(found)
    if found.challenger == user:
        raise MatchError("this is your own challenge link; send it to the member you want to play")
    return _start(connection, found, user)


def decline(connection: Any, challenge_id: int, user: int) -> Challenge:
    """The challenged person says no. The challenger sees it as *declined*, and no more."""
    found = find(connection, challenge_id)
    if found.challenged != user or found.link:
        raise KeyError(f"there is no challenge {challenge_id} for you")
    _waiting(found)
    return _answer(connection, found.id, DECLINED)


def withdraw(connection: Any, challenge_id: int, user: int) -> Challenge:
    """The challenger takes a waiting challenge or link back."""
    found = find(connection, challenge_id)
    if found.challenger != user:
        raise KeyError(f"there is no challenge {challenge_id} of yours")
    _waiting(found)
    return _answer(connection, found.id, WITHDRAWN)


def _answer(connection: Any, challenge_id: int, answer: str) -> Challenge:
    with connection.cursor() as cursor:
        cursor.execute(
            f"update challenges set answer = %s, answered = now() where id = %s"
            f" returning {_COLUMNS}",
            (answer, challenge_id),
        )
        return _to_challenge(cursor.fetchone())


def leave(connection: Any, user: int) -> int:
    """Withdraw every waiting challenge this person sent or was sent, and untick them.

    What leaving the site and being revoked do (§1.4), beside `correspondence.leave`.
    Returns how many challenges were withdrawn.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update challenges set answer = %s, answered = now()"
            " where answer is null and (challenger = %s or challenged = %s)",
            (WITHDRAWN, user, user),
        )
        withdrawn = int(cursor.rowcount)
        cursor.execute("delete from open_to where person = %s", (user,))
        cursor.execute(
            "update users set takes_challengers = false where id = %s", (user,)
        )
    return withdrawn


# ---------------------------------------------------------------------------- reading


def seat_words(seat: int | None) -> str:
    """A seat, as the person who will sit in it reads it, a clause after a comma."""
    if seat is None:
        return "who moves first is drawn when it starts"
    return "you'd move first" if seat == 0 else "you'd move second"


def _names(connection: Any, games: set[str]) -> dict[str, str]:
    if not games:
        return {}
    with connection.cursor() as cursor:
        cursor.execute("select hash, name from games where hash = any(%s)", (list(games),))
        return {str(digest): str(name) for digest, name in cursor.fetchall()}


def _yours(seat: int | None) -> int | None:
    """The acceptor's seat, given the challenger's."""
    return None if seat is None else 1 - seat


def for_you(connection: Any, user: int) -> list[dict[str, Any]]:
    """*A challenge for you*: the named challenges waiting on this person, oldest first."""
    found = _select(
        connection,
        "challenged = %s and token is null and answer is null order by made, id",
        (user,),
    )
    names = _names(connection, {each.game for each in found})
    challengers = {each.challenger: identity.handle_of(connection, each.challenger) for each in found}
    return [
        {
            "id": each.id,
            "from": challengers[each.challenger],
            # The challenger's generated mark (R2), beside their handle.
            "fromMark": identity.mark_of(each.challenger, challengers[each.challenger]),
            "game": each.game,
            "name": names.get(each.game, ""),
            "seat": seat_words(_yours(each.seat)),
            "rated": correspondence.mode_for(connection, each.game) == matches.LEADERBOARD,
            "rematch": each.rematch_of is not None,
            "made": each.made.isoformat(),
        }
        for each in found
    ]


def sent_by(connection: Any, user: int) -> list[dict[str, Any]]:
    """*Challenges you sent*: every waiting one and link, then those declined this week.

    A link carries its address, so the page can offer it to copy again.
    """
    found = _select(
        connection,
        "challenger = %s and (answer is null or (answer = %s and answered > now() - %s))"
        " order by (answer is null) desc, made, id",
        (user, DECLINED, DECLINED_SHOWN),
    )
    names = _names(connection, {each.game for each in found})
    return [
        {
            "id": each.id,
            "to": None if each.challenged is None else identity.handle_of(
                connection, each.challenged
            ),
            "link": None if each.token is None else LINK_PATH + each.token,
            "game": each.game,
            "name": names.get(each.game, ""),
            "seat": seat_words(each.seat),
            "answer": each.answer,
            "made": each.made.isoformat(),
        }
        for each in found
    ]


def link_view(connection: Any, token: str, viewer: int | None) -> dict[str, Any]:
    """What a challenge link's page says, to whoever opened it (s8 of the mock).

    Anybody holding the link may read whose challenge it is and to which game: that is
    what the challenger sent it to say. `state` is `waiting`, `yours` (the challenger
    opened their own), `taken` (with the two players), `withdrawn`, or `paused`;
    `visitor` says the reader has to sign in to accept it.
    """
    found = by_token(connection, token)
    name = _names(connection, {found.game}).get(found.game, "")
    challenger = identity.handle_of(connection, found.challenger)
    state = "waiting"
    players: list[str] = []
    if found.answer == ACCEPTED:
        state = "taken"
        players = [challenger, identity.handle_of(connection, found.challenged or 0)]
    elif found.answer is not None:
        state = "withdrawn"
    elif viewer is not None and viewer == found.challenger:
        state = "yours"
    elif correspondence.paused(connection):
        state = "paused"
    return {
        "from": challenger,
        "game": found.game,
        "name": name,
        "seat": seat_words(_yours(found.seat)),
        "rated": correspondence.mode_for(connection, found.game) == matches.LEADERBOARD,
        "state": state,
        "players": players,
        "match": found.match if state == "taken" and viewer in (
            found.challenger, found.challenged
        ) else None,
        "visitor": viewer is None,
        "pausedSays": correspondence.PAUSED_SAID if state == "paused" else None,
        "rules": list(correspondence.RULES),
    }


