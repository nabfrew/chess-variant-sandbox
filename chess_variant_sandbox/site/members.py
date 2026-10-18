"""A member's page, `/players/<handle>` (R1, `docs/archive/plan-return.md` §6.2).

The user's decision 2 of 2026-10-02, option A: members see each other's page -- recent
games, designed games, what they have done, who they play most with -- and five-star games
only by the person's own choice. **Signed-in members only, never visitors**: the site's
handler sends a visitor to the door before this module is asked anything.

What the page holds, and nothing else (§6.2's table):

- **the header**: the mark (R2, `web/page/marks.py`), the handle, *member since ‹month›*,
  the abandonment line (`correspondence.finished_line`, once there is a finished game
  between people), and *Challenge ‹handle›* for another member: the games they made first,
  then the rated games (Q760);
- **Games ‹handle› played lately**: the last `RECENT` finished rated games and games
  between people, newest first (Q760: R1's *Recently* mixed these with firsts and level
  changes, which are said once each under the two sections below);
- **Games ‹handle› made**: published games only, newest first, each with the day it was
  published, the level word and how many other people have tried it (a count, never who);
- **What ‹handle› has done**: the milestones and badges, as on your own page;
- **Favourite games**: the games they gave five stars, **only if they ticked**
  `users.show_favourites` on their own page;
- **Plays most with**: the people they have finished most games with, and the computer
  players they have finished most rated games against, `PLAYS_WITH` of each, in order and
  with no count beside a name.

**A draft never shows.** Every line that names a game is dropped unless the game is
published (any level but Draft) and still has a page (`admin.unopenable_games`), so a
practice game on a draft's link, a designer's first-draft line or a badge on a draft is
absent rather than a way to find the draft. The *forked* line keeps its game and loses
its *by ‹handle›*, because the fork is somebody else's and may be a draft.

No rank, no position, no *last seen* (§6.2's tests): nothing here orders people against
each other, and the store records no visit. A section with nothing in it is absent.

Nothing here commits; it only reads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import (
    admin,
    awards,
    badges,
    browse,
    correspondence,
    designs,
    identity,
    matches,
    milestones,
    rating,
    stages,
)
from chess_variant_sandbox.web import opponents

RECENT = 8
"""How many finished games *Games ‹handle› played lately* lists (§6.2)."""

MADE_SHOWN = 3
"""How many of a member's published games the people page names beside them (Q760)."""

PLAYS_WITH = 3
"""How many people, and how many computer players, *Plays most with* names."""

PATH = "/players/"
"""Where a member's page lives: this, then the handle."""


def address(handle: str) -> str:
    """A member's page's address. Handles are `identity.HANDLE`'s narrow set, so the
    handle needs no quoting."""
    return PATH + handle


def find(connection: Any, handle: str) -> tuple[int, str, datetime] | None:
    """`(id, handle, joined)` of the member with this handle, or None.

    None for a handle nobody holds, for somebody who left (their handle is gone), and for
    somebody shut out from `/admin`: none of them has a page, and each is the same 404 as
    a handle never taken, so the page cannot be used to learn who was revoked. Matched
    without regard to case, as signing in is; the page names the handle as it is held.
    """
    name = handle.strip()
    if not identity.HANDLE.match(name):
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, handle, created from users where lower(handle) = lower(%s)"
            " and revoked is null and forgotten is null",
            (name,),
        )
        found = cursor.fetchone()
    if found is None:
        return None
    return int(found[0]), str(found[1]), found[2]


def shows_favourites(connection: Any, user: int) -> bool:
    """Whether this person ticked *show my five-star games on my page*. Off by default."""
    with connection.cursor() as cursor:
        cursor.execute("select show_favourites from users where id = %s", (user,))
        found = cursor.fetchone()
    return bool(found and found[0])


def set_favourites(connection: Any, user: int, on: bool) -> bool:
    """Tick or untick the box; the box as it now stands."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set show_favourites = %s where id = %s", (bool(on), user)
        )
    return shows_favourites(connection, user)


FAVOURITES_SAID = "Show the games I gave five stars on my page, to other members."
FAVOURITES_UNDER = (
    "Off unless you tick it. Your stars are private otherwise: a game's page shows only "
    "how many people voted, the average and how many chose each tag, never who gave what."
)


def favourites_box(connection: Any, user: int) -> dict[str, Any]:
    """Your own page's box, with its sentences."""
    return {
        "on": shows_favourites(connection, user),
        "said": FAVOURITES_SAID,
        "under": FAVOURITES_UNDER,
    }


def month(when: datetime) -> str:
    """*October 2026*, for *member since*."""
    return f"{when:%B %Y}"


def _published(connection: Any, hashes: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The rows of the games among these that a member's page may name, by hash.

    Published (any level but Draft) and with a page: not killed, not deleted.
    """
    rows = stages.read(connection, hashes)
    if not rows:
        return {}
    shut = admin.unopenable_games(connection)
    return {
        digest: row for digest, row in rows.items()
        if row["level"] != stages.DRAFT and digest not in shut
    }


def _finished(connection: Any, user: int) -> list[matches.Row]:
    """This person's finished games that count here, newest first: rated games, and games
    between two people whether rated or practice. Cancelled games (no result) and the
    local application's rows are left out; a practice game against the computer is no
    event."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {matches._ROW_COLUMNS} from matches"
            " where (seat0_user = %(user)s or seat1_user = %(user)s)"
            " and finished is not null and result is not null"
            " and (source is null or source <> %(local)s)"
            f" and (mode = %(rated)s or ({matches.BETWEEN_PEOPLE}))"
            " order by finished desc, id",
            {"user": user, "local": matches.LOCAL_APP, "rated": matches.LEADERBOARD},
        )
        return [matches._to_row(found) for found in cursor.fetchall()]


def with_pages(connection: Any) -> dict[int, str]:
    """Everybody who has a page: a handle, not shut out, not left."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, handle from users"
            " where handle is not null and revoked is null and forgotten is null"
        )
        return {int(found[0]): str(found[1]) for found in cursor.fetchall()}


DIRECTORY_SAID = (
    "Every member is listed here: a handle is chosen to be seen by other members, and "
    "the box that challenges somebody already suggests every member's handle. Each name "
    "is the way to that member's page. Visitors cannot see this list or the pages."
)
"""Who the people page lists, and why, said on the page itself (Q760)."""


def directory(connection: Any, viewer: int | None = None) -> list[dict[str, Any]]:
    """The people page's list (Q760): everybody who has a page, by handle, A to Z.

    Each with exactly what their page's header already shows every member -- the mark, the
    handle, *member since* -- and the names of the newest `MADE_SHOWN` of their published
    games with how many more there are, so somebody can be found by a game of theirs.
    Nothing is counted against anybody else and nothing here is a position: the order is
    the alphabet's. No draft, no vote, no game played and no setting is read, so nothing
    a member has not chosen to show can reach the payload. `you` marks the reader's own
    row, so they can find themselves.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, handle, created from users"
            " where handle is not null and revoked is null and forgotten is null"
            " order by lower(handle), id"
        )
        # Only a handle `find` answers for, so every row's link opens a page.
        people = [
            (int(row[0]), str(row[1]), row[2]) for row in cursor.fetchall()
            if identity.HANDLE.match(str(row[1]))
        ]
        cursor.execute(
            "select hash, owner from games where origin = %s and owner = any(%s)",
            (designs.DESIGNED, [person[0] for person in people]),
        )
        owned = [(str(row[0]), int(row[1])) for row in cursor.fetchall()]
    public = _published(connection, (digest for digest, _ in owned))
    names = browse.names(connection, sorted(public))
    made: dict[int, list[tuple[float, str]]] = {}
    for digest, owner in owned:
        if digest not in public:
            continue
        published = public[digest].get("published")
        stamp = published.timestamp() if isinstance(published, datetime) else 0.0
        made.setdefault(owner, []).append((-stamp, digest))
    listed = []
    for user, handle, joined in people:
        theirs = [digest for _, digest in sorted(made.get(user, []))]
        listed.append({
            "handle": handle,
            "mark": identity.mark_of(user, handle),
            "page": address(handle),
            "since": month(joined),
            "you": user == viewer,
            "made": [
                {"game": digest, "name": names.get(digest, "")}
                for digest in theirs[:MADE_SHOWN]
            ],
            "moreMade": max(0, len(theirs) - MADE_SHOWN),
        })
    return listed


def _seat(found: matches.Row, user: int) -> int:
    return 0 if found.seat0 is not None and found.seat0.user == user else 1


def _against(found: matches.Row, user: int, people: Mapping[int, str]) -> dict[str, Any]:
    """Who sat across: a member (with their mark and a page), somebody without a page
    (*a former player*), or a computer player by the ladder's name for it."""
    other = found.seat1 if _seat(found, user) == 0 else found.seat0
    if other is None:
        return {"person": identity.FORMER}
    if other.user is not None:
        handle = people.get(int(other.user))
        if handle is None:
            return {"person": identity.FORMER}
        return {
            "person": handle, "id": int(other.user),
            "mark": identity.mark_of(int(other.user), handle),
        }
    return {"computer": opponents.name_of(str(other.agent))}


def _result(found: matches.Row, user: int) -> str:
    score = found.score_for(_seat(found, user))
    if score is None or score == 0.5:
        return "drew"
    return "won" if score > 0.5 else "lost"


def read(connection: Any, handle: str, viewer: int) -> dict[str, Any] | None:
    """Everything `/players/<handle>` draws, or None for a handle with no page (a 404).

    `viewer` is the signed-in member asking; `own` says whether it is their own page,
    which drops *Challenge* and adds the way to their settings. What the page shows of
    the person is the same for every member, the person included.
    """
    found = find(connection, handle)
    if found is None:
        return None
    user, held_handle, joined = found
    people = with_pages(connection)

    tables = rating.load_many(connection)
    rated = milestones._rated(tables)
    everyone = matches.leaderboard_matches(connection)
    made = designs.of(connection, user)
    own_games = [d.hash for d in made if d.owner == user]
    tries = designs.tries(connection, own_games)
    reading = milestones.read_from(
        user, everyone, tables, badges.cells(connection, rated), made, tries
    )
    held = awards.held(connection, holder=user)
    games = _finished(connection, user)

    named_games = {
        *(row.game for row in games),
        *(line.game for line in reading.milestones),
        *(award.game for award in held if award.game),
        *own_games,
    }
    with connection.cursor() as cursor:
        cursor.execute(
            "select game from votes where voter = %s and stars = 5", (user,)
        )
        starred = [str(row[0]) for row in cursor.fetchall()]
    favourites_on = shows_favourites(connection, user)
    if favourites_on:
        named_games.update(starred)
    public = _published(connection, named_games)
    names = browse.names(connection, sorted(public))

    def named(digest: str) -> dict[str, Any]:
        return {"game": digest, "name": names.get(digest, "")}

    # Q760: the games they played lately, newest first, the last RECENT. (R1's *Recently*
    # mixed these with firsts and level changes; the firsts are under *What ‹handle› has
    # done* and a game's level is beside it under *Games ‹handle› made*, so each fact is
    # said once.)
    played: list[dict[str, Any]] = []
    for row in games:
        if row.game not in public or row.finished is None:
            continue
        played.append({
            "when": row.finished, **named(row.game),
            "result": _result(row, user), "against": _against(row, user, people),
            "between": correspondence.between_people(row),
            "rated": row.mode == matches.LEADERBOARD,
        })
        if len(played) == RECENT:
            break
    lines = [line for line in reading.milestones if line.game in public]
    mine_shown = {
        digest: row for digest, row in public.items() if digest in set(own_games)
    }

    def newest(digest: str) -> tuple[float, str, str]:
        published = mine_shown[digest].get("published")
        stamp = published.timestamp() if isinstance(published, datetime) else 0.0
        return (-stamp, names.get(digest, "").lower(), digest)

    # Every published game they made, the newest first, each with the day it went out.
    designed = [
        {
            **named(digest),
            "level": mine_shown[digest]["level"],
            "published": mine_shown[digest].get("published"),
            "tried": len({t.user for t in tries if t.game == digest and t.user != user}),
        }
        for digest in sorted(mine_shown, key=newest)
    ]

    badges_shown = [
        {**badges.line(award), "name_of": names.get(award.game or "", "")}
        for award in held
        if not award.game or award.game in public
    ]

    with_people: Counter[str] = Counter()
    with_computers: Counter[str] = Counter()
    for row in games:
        against = _against(row, user, people)
        if correspondence.between_people(row):
            if "mark" in against:
                with_people[against["person"]] += 1
        elif "computer" in against and row.mode == matches.LEADERBOARD:
            with_computers[against["computer"]] += 1
    ids = {handle: user_id for user_id, handle in people.items()}

    def most(counted: Counter[str]) -> list[str]:
        return [name for name, _ in sorted(counted.items(), key=lambda kv: (-kv[1], kv[0]))]

    favourites = (
        sorted(
            (named(digest) for digest in starred if digest in public),
            key=lambda game: (str(game["name"]).lower(), game["game"]),
        )
        if favourites_on else []
    )

    own = user == viewer
    challenge = None
    if not own:
        unlisted = admin.unlisted_games(connection)
        # Q760: the games they made come first, since a challenge from somebody's page is
        # most often to one of theirs; then every rated game. A game of theirs nobody can
        # start (taken off the site) is in neither.
        theirs = [game["game"] for game in designed if game["game"] not in unlisted]
        offered = [game for game in rated if game not in unlisted and game not in theirs]
        offered_names = browse.names(connection, offered)
        challenge = {
            "paused": correspondence.paused(connection),
            "pausedSays": correspondence.PAUSED_SAID,
            "theirs": [named(game) for game in theirs],
            "games": sorted(
                ({"game": game, "name": offered_names.get(game, "")} for game in offered),
                key=lambda game: (str(game["name"]).lower(), game["game"]),
            ),
        }

    return {
        "user": user,
        "handle": held_handle,
        "mark": identity.mark_of(user, held_handle),
        "since": month(joined),
        "finished": correspondence.finished_line(connection, user),
        "own": own,
        "challenge": challenge,
        "played": played,
        "designed": designed,
        "milestones": [
            {
                "kind": line.kind,
                "rule": milestones.RULES[line.kind],
                "when": line.when,
                **named(line.game),
                "seat": line.seat if line.seat >= 0 else None,
            }
            for line in lines
        ],
        "badges": badges_shown,
        "favourites": favourites,
        "playsWith": {
            "people": [
                {"handle": name, "id": ids[name], "mark": identity.mark_of(ids[name], name)}
                for name in most(with_people)[:PLAYS_WITH]
            ],
            "computers": most(with_computers)[:PLAYS_WITH],
        },
    }
