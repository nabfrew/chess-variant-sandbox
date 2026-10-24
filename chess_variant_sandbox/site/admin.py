"""What a maintainer can see and do about people and games, and leaving.

Task 1.8 of `docs/plan-website.md`: the users, their invites and their sessions, *revoke
all sessions*, a per-user kill switch and a per-game kill switch. Plus the one thing
`/privacy` promises that had no code behind it -- `chess-variant-sandbox site forget`.

Nothing here is a new mechanism. `users.revoked` and `games.revoked` are columns
`0001_store.sql` wrote for this row, and `identity.sign_in` and `identity.resolve`
already refuse a revoked user, so the per-user switch needed only a way to set the
column. What 1.8 adds is the reading, the page, and the per-game half, which nothing
consulted before.

## Who is an admin

`users.role`, which is `member` for everybody an invitation makes and `admin` for the
maintainer the loader creates. A person becomes an admin one way only: a maintainer runs
`chess-variant-sandbox site admin <handle>` against the database, on the box. There is no button on
this page that makes another admin, because the page is reached by being one -- a
promotion loop with no floor is how one mistaken account becomes every account.

## The maintainer's own account, from the environment

There is a second way to *be* one, and it makes no new mechanism either. `ADMIN_PASSWORD`
in the box's environment gives the site one account of its own, named by
`ADMIN_ACCOUNT_NAME` and called `secret-admin` when that key is unset. It is an ordinary
`users` row with `role = 'admin'` and a scrypt hash in `password`, so it signs in through
`/api/signin` like anybody else, is compared in constant time like anybody else, and is
throttled by whatever is in front of that route. Nothing anywhere reads a password out of
the environment at a check.

The row is reconciled from the environment at every sign-in attempt, whichever handle
it names (`ensure_account`, called by `server.SiteApp._signin`), and it is found by its
reserved address rather than by its handle, so renaming the account renames the row
instead of leaving a second admin behind. The reconcile hashes nothing: the environment's
password is hashed once per process (`account_hash`), so an attempt at this handle costs
what an attempt at anybody's does and its answer takes as long (Q379).

**With no `ADMIN_PASSWORD` the account cannot sign in**: the reconcile takes the hash
away rather than leaving a stale one, and with no row it writes none. A handle somebody
else already holds is refused outright rather than taken over.

**A refusal turns the account off and never stops the site** (Q365b). A password under
the site's own minimum, or a name that is not a usable handle, means no maintenance
account; the site serves and says which key it refused and why, once, at the start.

## The two kill switches, and what each one means

**A killed user cannot sign in and their open sessions stop working.** Both already fall
out of `revoked is not null` in the two identity queries; `kill_user` deletes their
session rows as well, so the answer does not wait on a cookie expiring and the admin page
does not show sessions for somebody who has been shut out.

## A retired game is a different state, and this page shows it

`games.retired` is the loader's, not an admin's: a spec that has left `rulesets/` and
`archive/` is stamped by `chess-variant-sandbox site load` and reads *retired* here. It is off the
listing and refused a new match, and unlike a killed game its row and its past matches
are read as before. The buttons on this page act on `revoked` alone, so a retired game can
still be killed and a killed one can still be put back. Bringing a retired game back is
putting its spec in the tree, which is the only place the library is decided from.

**A killed game is not listed, cannot be opened and cannot be started.** `revoked`
overrides `visibility` rather than borrowing a value from it, because visibility is the
owner's answer and this one is not. The site drops the game from `/api/library`, answers
`/api/game/<hash>` with the 404 an unknown hash gets, and `matches.start` refuses it by
name. Matches already played on it stay and still count: killing a game says *no more of
this*, never *none of that happened*.

## Deleting a game outright

`delete_game` is the third state and the only one with no way back (migration `0010`).
The creator of a game and an admin may take a retired game's rules away for good: the
spec is overwritten with null, the name becomes *deleted game* and `deleted` is stamped.
The row itself stays, because every reference to a game in this store cascades -- the
matches people played, their ratings and their votes hang off `games.hash`, and deleting
the row would take all three with it. So a delete removes the rules and changes nobody's
statistics.

**Only a retired game.** A game whose spec is still in `rulesets/` or `archive/` has its
rules in the repository, so a delete would be undone by the next deploy's load and would
not have deleted anything meanwhile. The way to delete such a game is to take its spec
out of the tree, deploy, and delete the retired row. The loader knows the other half of
this: a deleted row is never written again, even if its spec comes back.

## Leaving

`forget` is the privacy notice's paragraph and exactly that paragraph. The email address
and the password hash are deleted, every session ends, the handle goes, and the matches
stay -- each game's ladder is a fold over its matches from the first, so removing rows
would move other people's ratings along with the departing player's. A row with no handle
reads as *a former player* everywhere a handle is printed, which is one `coalesce` in
`unbeaten.handles` and one in `identity.handle_of` rather than a rewrite of the match
rows. `forgotten` says when, and is a different column from `revoked` because the two are
different acts: a killed account can be let back in and a forgotten one cannot.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import identity

ADMIN = "admin"
MEMBER = "member"
ROLES = (MEMBER, ADMIN)

FORMER = identity.FORMER
"""What a departed player is called. Defined beside the queries that read a handle."""

DELETED_NAME = "deleted game"
"""What a game whose rules were deleted is called, everywhere it is still referenced.
The user's word, 2026-09-18. Written into the row at the delete rather than applied by
each reader, so a match, a badge and a profile row all read it from the query they
already make."""


class AdminError(ValueError):
    """An administrative act cannot be carried out as asked.

    A `ValueError`, so the server answers 400 with the sentence and the CLI prints one
    line. Every message names the thing that was not found or the state that refused.
    """


# ------------------------------------------------------------------------- who is one


def is_admin(connection: Any, user: int | None) -> bool:
    """Whether this person is an admin, asked of the database rather than of a cookie."""
    if user is None:
        return False
    with connection.cursor() as cursor:
        cursor.execute("select role from users where id = %s and revoked is null", (user,))
        found = cursor.fetchone()
    return found is not None and str(found[0]) == ADMIN


def set_role(connection: Any, handle: str, role: str = ADMIN) -> str:
    """Make somebody an admin, or put them back to a member. The maintainer's command.

    By handle, because that is what a maintainer knows, and it is matched the way signing
    in matches it.
    """
    if role not in ROLES:
        raise AdminError(f"a role is one of {', '.join(ROLES)}, not {role!r}")
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set role = %s where lower(handle) = lower(%s) returning handle",
            (role, handle.strip()),
        )
        found = cursor.fetchone()
    if found is None:
        raise AdminError(f"no user with the handle {handle!r}")
    return str(found[0])


# ------------------------------------------------- the maintainer's account, from the
# environment

NAME_VAR = "ADMIN_ACCOUNT_NAME"
PASSWORD_VAR = "ADMIN_PASSWORD"

DEFAULT_NAME = "secret-admin"
"""What the account is called when `ADMIN_ACCOUNT_NAME` is unset. The user's word,
2026-09-19."""

RESERVED_EMAIL = "admin@account.invalid"
"""The address the environment's account is found by.

`users.email` is `not null unique` and this account has nobody to write to, so it carries
a reserved address under RFC 2606's `.invalid`, which no mail can reach and no invitation
can claim. Reading the row by this rather than by its handle is what makes renaming the
account a rename: the handle moves and the row stays one row."""


def account_name() -> str:
    """The handle the environment's account signs in under, as configured."""
    return os.environ.get(NAME_VAR, "").strip() or DEFAULT_NAME


def account_setting() -> tuple[str | None, str | None]:
    """What the environment asks for, and why it cannot be had.

    `(handle, None)` when there is an account, `(None, None)` when the environment asks
    for none, and `(None, sentence)` when it asks for one the site will not make. It
    raises nothing and opens no connection.

    **A refusal here turns the account off; it never stops the site.** On 2026-09-19 this
    check raised before `serve` bound, systemd gave up after five starts and the site was
    down for four minutes over a password that was four characters long. The site is not a
    maintenance account with a site attached, so the failure it belongs to is *no
    maintenance account*, said once in the journal.

    The sentence names the key and the reason. Neither the password nor its length is in
    it, because the journal it goes to is read by whoever can read the box.
    """
    name = account_name()
    if not identity.HANDLE.match(name):
        return None, (
            f"{NAME_VAR} is {name!r}, which is not a usable handle: "
            "3 to 24 characters, letters, digits, hyphen and underscore, "
            "starting with a letter or a digit"
        )
    password = os.environ.get(PASSWORD_VAR, "")
    if not password:
        return None, None
    if len(password) < identity.MIN_PASSWORD:
        return None, (
            f"{PASSWORD_VAR} is shorter than {identity.MIN_PASSWORD} characters, "
            "which is the site's own minimum"
        )
    return name, None


def account_check() -> str | None:
    """The handle the environment's account signs in under, or `None` for no account.

    Read by `server.serve` before it binds and by `ensure_account` at every sign-in, so a
    password the environment shortened takes the stored hash away at the next attempt
    rather than going on opening the door.
    """
    return account_setting()[0]


def account_refusal() -> str | None:
    """Why there is no account although the environment asked for one, or `None`.

    Printed once by `serve`. `ensure_account` does not print it: a sentence emitted per
    sign-in attempt is a journal anybody can fill from outside.
    """
    return account_setting()[1]


def account_hash() -> str | None:
    """The stored string for the environment's password, or `None` for no account.

    Hashed once per process and per password, so the reconcile at a sign-in costs no
    scrypt of its own (Q379). Before, every attempt at the maintenance handle hashed the
    environment's password and then the one submitted, twice what anybody else's costs,
    and the time the answer took named the handle. `serve` calls this before it binds, so
    no attempt pays for it; a test that changes the environment pays at its next sign-in,
    whichever handle that names.

    The password is kept here as its SHA-256 and nothing else, beside the string made
    from it: enough to see that the environment changed, and nothing a stray repr prints.
    """
    if account_check() is None:
        return None
    password = os.environ[PASSWORD_VAR]
    key = hashlib.sha256(password.encode("utf-8")).hexdigest()
    global _hashed
    with _hashing:
        if _hashed is None or _hashed[0] != key:
            _hashed = (key, identity.hash_password(password))
        return _hashed[1]


_hashed: tuple[str, str] | None = None
"""The environment's password as last hashed here: its SHA-256, and the stored string."""

_hashing = threading.Lock()
"""So two first sign-ins at once hash the environment's password once between them."""


def ensure_account(connection: Any) -> str | None:
    """Make the environment's account real, or make sure nobody can sign in as it.

    Returns the handle it can be signed in under, or `None` when there is no account. The
    caller commits before the sign-in it runs ahead of, so a wrong password still leaves
    the row as the environment says it should be.

    **Run at every sign-in, whichever handle it names** (Q379), and it hashes nothing:
    the stored string is `account_hash`'s, made once per process. Before, it ran only
    for the maintenance handle and hashed each time, so that handle's answer took twice
    as long as anybody's. It writes only when the row is not already what the
    environment says, so an attempt is two small reads and, after a restart or a changed
    key, one write -- the same whichever handle is being tried.

    A handle a real person already holds turns the account off and leaves them alone, the
    way a short password does (Q365b). It used to raise, and the 400 that reached the
    person signing in named `ADMIN_ACCOUNT_NAME` and so told a stranger both that there
    is a maintenance account and that they hold its handle (Q369). The reason goes to the
    journal once per process instead, and a row written under an earlier name loses its
    password, as it does when the password goes.
    """
    name = account_check()
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, handle, password, role, revoked from users where email = %s",
            (RESERVED_EMAIL,),
        )
        found = cursor.fetchone()
        if name is not None and not _handle_is_free(cursor, name):
            name = None
        if name is None:
            if found is not None and found[2] is not None:
                cursor.execute(
                    "update users set password = null where id = %s", (found[0],)
                )
            return None
        hashed = account_hash()
        if found is None:
            # `on conflict`: two first sign-ins at once both find no row, and the second
            # insert is the first one's row rather than a unique violation and a 500.
            cursor.execute(
                "insert into users (handle, email, password, role)"
                " values (%s, %s, %s, %s) on conflict do nothing",
                (name, RESERVED_EMAIL, hashed, ADMIN),
            )
        elif tuple(found[1:]) != (name, hashed, ADMIN, None):
            cursor.execute(
                "update users set handle = %s, password = %s, role = %s, revoked = null"
                " where id = %s",
                (name, hashed, ADMIN, found[0]),
            )
    return name


def _handle_is_free(cursor: Any, name: str) -> bool:
    """Whether this handle is the environment's to take, and say so once if it is not.

    Without this, pointing `ADMIN_ACCOUNT_NAME` at a name somebody already signs in under
    would hand their account the admin role and a password from the environment.

    False leaves both accounts as they are: the person signs in as themselves, and the
    environment gets no maintenance account until the key names a free handle. Nothing is
    said to whoever is signing in, so the line goes to the journal, once, because this
    runs at an attempt anybody can make.
    """
    cursor.execute(
        "select 1 from users where lower(handle) = lower(%s) and email <> %s",
        (name, RESERVED_EMAIL),
    )
    if cursor.fetchone() is None:
        return True
    global _said_the_clash
    if not _said_the_clash:
        _said_the_clash = True
        print(
            f"maintenance: {NAME_VAR} names a handle somebody already has, "
            "so there is no maintenance account; choose a name nobody has"
        )
    return False


_said_the_clash = False
"""Whether the clash above has been printed in this process. One line, not one a minute."""


# ---------------------------------------------------------------------- what is there


@dataclass(frozen=True)
class Person:
    """One row of the users table, with what hangs off it counted."""

    id: int
    handle: str
    email: str
    role: str
    created: datetime | None
    revoked: datetime | None
    forgotten: datetime | None
    sessions: int
    invites: int
    matches: int
    # Q719: how many invitations this person may still send themselves.
    to_give: int = 0

    @property
    def state(self) -> str:
        """Three states, and a page that shows one word rather than three columns."""
        if self.forgotten is not None:
            return "left"
        if self.revoked is not None:
            return "killed"
        return "active"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "handle": self.handle,
            "email": self.email,
            "role": self.role,
            "state": self.state,
            "created": _when(self.created),
            "sessions": self.sessions,
            "invites": self.invites,
            "matches": self.matches,
            "toGive": self.to_give,
        }


@dataclass(frozen=True)
class Invitation:
    """One invitation, as the page lists it. The code itself is never sent out."""

    code: str
    email: str
    issuer: str
    uses_left: int
    expiry: datetime | None
    created: datetime | None
    accepted: bool
    expired: bool

    @property
    def state(self) -> str:
        """Accepted once its address has an account or its code is spent, else open or expired.

        Accepted is read per address, so an earlier invitation to somebody who joined by a
        later one reads as accepted too. The page answers *did this person get in*.
        """
        if self.accepted:
            return "accepted"
        return "expired" if self.expired else "open"

    def as_dict(self) -> dict[str, Any]:
        return {
            "email": self.email,
            "issuer": self.issuer,
            "usesLeft": self.uses_left,
            "expiry": _when(self.expiry),
            "created": _when(self.created),
            "accepted": self.accepted,
            "state": self.state,
        }


@dataclass(frozen=True)
class Listed:
    """One game the store holds, with the kill switch's state on it."""

    hash: str
    name: str
    origin: str
    revoked: datetime | None
    matches: int
    owner: str = ""
    visibility: str = "public"
    retired: datetime | None = None
    deleted: datetime | None = None

    @property
    def state(self) -> str:
        """One word for five states, the strongest first.

        A deleted game reads *deleted* whatever else was stamped on it: its rules are
        gone and no button on this page has anything left to act on. Below it, a killed
        game that has also left the tree still reads *killed*, because the admin's switch
        is what the buttons act on and a person reading this page needs to know which act
        to undo.

        A standing row nobody has stamped is *listed* when it is public and a *draft*
        otherwise: since Q503a a save lists nothing, so most designed rows are drafts that
        open by their link alone, and calling them listed would say the opposite.
        """
        if self.deleted is not None:
            return "deleted"
        if self.revoked is not None:
            return "killed"
        if self.retired is not None:
            return "retired"
        return "listed" if self.visibility == "public" else "draft"

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "name": self.name,
            "origin": self.origin,
            "owner": self.owner,
            "visibility": self.visibility,
            "state": self.state,
            # The buttons act on the kill switch alone, so the page reads this rather
            # than the state word: a retired game can still be killed, and a killed one
            # is put back whether or not its spec is in the tree.
            "killed": self.revoked is not None,
            # A deleted game has no rules left, so neither button means anything on it.
            "gone": self.deleted is not None,
            "matches": self.matches,
        }


@dataclass(frozen=True)
class GameState:
    """One game's three switches and its owner, for the page and for the delete.

    Read by hash, once, because every question about what may be done to a game is
    answered from these five columns together: the page asks whether to draw a *retired*
    label and a delete button, and the delete itself asks who is allowed and whether the
    rules are the store's only copy.
    """

    hash: str
    name: str
    owner: int | None
    revoked: datetime | None
    retired: datetime | None
    deleted: datetime | None

    def may_delete(self, viewer: int | None, admin: bool = False) -> bool:
        """Whether this person may take these rules away for good.

        The creator or an admin, and a retired game only. A game still in the tree is not
        the store's to delete: the repository holds its rules and the next deploy's load
        would write the row again.
        """
        if self.deleted is not None or self.retired is None:
            return False
        return admin or (viewer is not None and viewer == self.owner)


def state_of(connection: Any, game: str) -> GameState | None:
    """One game's switches, or `None` when the store has no such row."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, name, owner, revoked, retired, deleted from games"
            " where hash = %s",
            (game,),
        )
        found = cursor.fetchone()
    if found is None:
        return None
    return GameState(
        hash=str(found[0]),
        name=str(found[1]),
        owner=None if found[2] is None else int(found[2]),
        revoked=found[3],
        retired=found[4],
        deleted=found[5],
    )


def people(connection: Any) -> list[Person]:
    """Everybody, newest last, with sessions, invitations issued and matches counted.

    Three counted columns in one query rather than three queries and a join in Python.
    `matches` counts both seats, because a person is in a match whichever side they sat.

    A handle that has been emptied reads as `FORMER` here for the same reason it does
    everywhere else: this page prints names, and a blank cell says *something is missing*
    where the truth is *somebody left*.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select u.id, coalesce(u.handle, %s), coalesce(u.email, ''), u.role,"
            "       u.created, u.revoked, u.forgotten,"
            "       (select count(*) from sessions s where s.user_id = u.id),"
            "       (select count(*) from invites i where i.issuer = u.id),"
            "       (select count(*) from matches m"
            "         where m.seat0_user = u.id or m.seat1_user = u.id),"
            "       u.invites_to_give"
            "  from users u order by u.id",
            (FORMER,),
        )
        return [
            Person(
                id=int(row[0]),
                handle=str(row[1]),
                email=str(row[2]),
                role=str(row[3]),
                created=row[4],
                revoked=row[5],
                forgotten=row[6],
                sessions=int(row[7]),
                invites=int(row[8]),
                matches=int(row[9]),
                to_give=int(row[10]),
            )
            for row in cursor.fetchall()
        ]


def invitations(connection: Any) -> list[Invitation]:
    """Every invitation, newest first, without the codes.

    A code in the listing would make this page a way to take over an invitation that is
    still open, and the page has no use for one -- the link went to the address beside it.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select i.code, coalesce(i.email, ''), coalesce(u.handle, ''), i.uses_left,"
            "       i.expiry, i.created,"
            "       i.uses_left < 1 or exists"
            "         (select 1 from users a where lower(a.email) = lower(i.email)),"
            "       coalesce(i.expiry <= now(), false)"
            "  from invites i left join users u on u.id = i.issuer"
            " order by i.created desc, i.code"
        )
        return [
            Invitation(
                code=str(row[0]),
                email=str(row[1]),
                issuer=str(row[2]),
                uses_left=int(row[3]),
                expiry=row[4],
                created=row[5],
                accepted=bool(row[6]),
                expired=bool(row[7]),
            )
            for row in cursor.fetchall()
        ]


def games(connection: Any) -> list[Listed]:
    """Every game the store holds, killed and retired ones included, by name.

    Killed games are in this listing and in no other: the page whose job is the kill
    switch is the one place a killed game still has to be visible, or there would be no
    way to un-kill it. Unlisted drafts are here too, with who saved each one, and so are
    the games the loader retired, which is where a maintainer sees what a deploy took off
    the listing.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select g.hash, g.name, g.origin, g.revoked,"
            "       (select count(*) from matches m where m.game = g.hash),"
            "       case when g.owner is null then '' else coalesce(u.handle, %s) end,"
            "       g.visibility, g.retired, g.deleted"
            "  from games g left join users u on u.id = g.owner"
            " order by g.revoked nulls last, g.retired nulls last, lower(g.name)",
            (FORMER,),
        )
        return [
            Listed(
                hash=str(row[0]),
                name=str(row[1]),
                origin=str(row[2]),
                revoked=row[3],
                matches=int(row[4]),
                owner=str(row[5]),
                visibility=str(row[6]),
                retired=row[7],
                deleted=row[8],
            )
            for row in cursor.fetchall()
        ]


def revoked_games(connection: Any) -> set[str]:
    """The hashes the site must not list, open or seat anybody at.

    One query, asked once per listing, because the alternative is a `revoked` check
    threaded through every reading that touches a game.
    """
    with connection.cursor() as cursor:
        cursor.execute("select hash from games where revoked is not null")
        return {str(row[0]) for row in cursor.fetchall()}


def retired_games(connection: Any) -> set[str]:
    """The hashes whose spec has left the tree, as the last load found it.

    Weaker than `revoked_games` and asked separately for that reason. A retired game is
    off the listing and refused a new match, and its row, its matches and its ratings are
    read exactly as before.
    """
    with connection.cursor() as cursor:
        cursor.execute("select hash from games where retired is not null")
        return {str(row[0]) for row in cursor.fetchall()}


def deleted_games(connection: Any) -> set[str]:
    """The hashes whose rules are gone. Stronger than either of the two above.

    A deleted row has no spec at all, so there is nothing to draw a page from and nothing
    to seat anybody at. `_playable` asks this alongside `revoked_games` and answers both
    with the 404 an unknown hash gets.
    """
    with connection.cursor() as cursor:
        cursor.execute("select hash from games where deleted is not null")
        return {str(row[0]) for row in cursor.fetchall()}


def unopenable_games(connection: Any) -> set[str]:
    """Every hash whose own page is the 404 an unknown hash gets: killed or deleted.

    The narrower door beside `unlisted_games`, asked once per game page. What parts the
    two is the retired game: off the listing, and its page still drawn.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash from games where revoked is not null or deleted is not null"
        )
        return {str(row[0]) for row in cursor.fetchall()}


def unlisted_games(connection: Any) -> set[str]:
    """Every hash the library must leave out -- killed, retired or deleted -- in one query.

    The listing is the one reading all three states answer the same way, so it asks once.
    The other doors are narrower and keep their own questions: `_playable` refuses a
    killed or a deleted game and lets a retired one through, because a retired game's row
    still draws a page, and `matches.start` refuses all three because none takes a new
    match.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash from games"
            " where revoked is not null or retired is not null or deleted is not null"
        )
        return {str(row[0]) for row in cursor.fetchall()}


# --------------------------------------------------------------------------- the acts


def kill_user(connection: Any, user: int) -> str:
    """Shut somebody out: stamp `revoked` and end every session they have.

    The stamp is what refuses the next sign-in; deleting the sessions is what stops the
    one they are in the middle of. Both, because either alone leaves a door open for the
    length of a cookie. Their games and their ratings are untouched -- a leaderboard with
    a hole in it is a leaderboard that lies about the people still on it.

    Their running games against other people are resigned for them, as games they left
    (Q570, `docs/archive/plan-return.md` §1.6), so nobody waits five days on an account that
    cannot sign in. Imported here, because `site/correspondence.py` reads the store's
    match rows and this module is imported by everything.
    """
    from chess_variant_sandbox.site import challenges, correspondence

    with connection.cursor() as cursor:
        cursor.execute(
            "update users set revoked = coalesce(revoked, now()) where id = %s"
            " returning coalesce(handle, %s)",
            (user, FORMER),
        )
        found = cursor.fetchone()
        if found is None:
            raise AdminError(f"no user with the id {user}")
        cursor.execute("delete from sessions where user_id = %s", (user,))
    correspondence.leave(connection, user)
    challenges.leave(connection, user)
    return str(found[0])


def revive_user(connection: Any, user: int) -> str:
    """Let somebody back in. They sign in again with the password they already had.

    A person who has left cannot be revived: their password hash is gone, so there is
    nothing to sign in with, and an account nobody can reach is worse than an absent one.
    """
    with connection.cursor() as cursor:
        cursor.execute("select forgotten from users where id = %s", (user,))
        found = cursor.fetchone()
        if found is None:
            raise AdminError(f"no user with the id {user}")
        if found[0] is not None:
            raise AdminError(
                "that account has been forgotten, so there is no password to come back "
                "with; invite the person again instead"
            )
        cursor.execute(
            "update users set revoked = null where id = %s returning handle", (user,)
        )
        return str(cursor.fetchone()[0])


def kill_game(connection: Any, game: str) -> str:
    """Take a game off the site. Matches already played on it stay and still count."""
    return _set_game_revoked(connection, game, kill=True)


def revive_game(connection: Any, game: str) -> str:
    """Put a game back on the site."""
    return _set_game_revoked(connection, game, kill=False)


def delete_game(connection: Any, game: str, viewer: int | None, admin: bool = False) -> str:
    """Take a retired game's rules away for good. Nothing commits; the caller decides.

    Returns the name the game had, because that name is about to stop existing and the
    answer to the button press is the one place it is still said.

    Three writes and no delete. The spec becomes null, the name becomes `DELETED_NAME`
    and `deleted` is stamped; `0010_deleted.sql`'s check constraint holds the first two
    together. Every row that references this game keeps referencing it, so the matches
    people played, their ratings, their votes and their badges are the same rows and the
    same numbers afterwards -- they just name a deleted game.

    The measurement goes with the rules. It is a reading of a spec nobody can read any
    more, and `measurements` is a view the loader rebuilds from `runs/` in any case, so
    it loses nothing a load cannot put back for the games that remain. (The store's
    `links` rows went too, until near took the map's meaning and the table retired: M5 of
    `docs/archive/plan-map.md`.)
    """
    state = state_of(connection, game)
    if state is None:
        raise AdminError(f"no game with the hash {game!r} is in the store")
    if state.deleted is not None:
        raise AdminError(f"{state.name} has already been deleted")
    if state.retired is None:
        raise AdminError(
            "only a retired game can be deleted; this one's rules are still in the "
            "repository, so a deploy would write them back"
        )
    if not state.may_delete(viewer, admin):
        raise AdminError(
            "only the person who made a game, or an admin, can delete it"
        )
    from chess_variant_sandbox.site import provisional

    # A number measured on the designer's machine goes the way the measurement does. Its
    # row stays, superseded, and the next `provisional.sweep` takes its records (Q312).
    provisional.withdraw(connection, game)
    with connection.cursor() as cursor:
        cursor.execute("delete from measurements where hash = %s", (game,))
        cursor.execute(
            "update games set spec = null, name = %s, deleted = now() where hash = %s",
            (DELETED_NAME, game),
        )
    return state.name


def _set_game_revoked(connection: Any, game: str, kill: bool) -> str:
    # The stamp is the server's clock, like every other timestamp in the store, so the
    # literal is in the statement rather than a value from a Python `now()`.
    stamp = "now()" if kill else "null"
    with connection.cursor() as cursor:
        cursor.execute(
            f"update games set revoked = {stamp} where hash = %s returning name", (game,)
        )
        found = cursor.fetchone()
    if found is None:
        raise AdminError(f"no game with the hash {game!r} is in the store")
    return str(found[0])


def forget(connection: Any, handle: str) -> str:
    """What `/privacy` says leaving does, and nothing besides.

    The email address and the password hash are deleted, the handle goes so that every
    match reads *a former player*, every session ends, and the matches themselves stay.
    Their rating rows stay too: they are the other half of somebody else's result, and a
    ladder is replayed from the first match.

    Returns the handle as it was, so the command can print who it was that left.

    Their running games against other people are resigned for them as games they left,
    as revoking does (Q570). Their solved marks on proven positions are deleted (R9b).
    """
    from chess_variant_sandbox.site import challenges, correspondence, proven

    name = handle.strip()
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, forgotten from users where lower(handle) = lower(%s)", (name,)
        )
        found = cursor.fetchone()
        if found is None:
            raise AdminError(f"no user with the handle {name!r}")
        user = int(found[0])
        cursor.execute("delete from sessions where user_id = %s", (user,))
        cursor.execute(
            "update users set email = null, password = null, handle = null,"
            " move_email = false, mail_off = null, show_favourites = false,"
            " takes_challengers = false, invites_to_give = 0,"
            " forgotten = coalesce(forgotten, now()) where id = %s",
            (user,),
        )
    correspondence.leave(connection, user)
    challenges.leave(connection, user)
    # R9b: a person's solved marks on proven positions are theirs alone, and go with them.
    proven.forget(connection, user)
    return name


def forget_id(connection: Any, user: int) -> str:
    """The same act, reached by id, which is what the admin page holds."""
    handle = identity.handle_of(connection, user)
    if handle == FORMER:
        raise AdminError(f"user {user} has already left")
    if not handle:
        raise AdminError(f"no user with the id {user}")
    return forget(connection, handle)


def _when(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()
