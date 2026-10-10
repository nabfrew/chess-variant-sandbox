"""Who is playing: an invitation, an account, and a cookie that says so.

Task 1.3 of `docs/plan-website.md`. The whole path is four rows and one function each.
A maintainer issues an invitation to an address; the person who follows the link chooses a
handle and a password; both land in `users` and a `sessions` row backs the cookie; coming
back is the same cookie or the same password. One command deletes every session row, which
is §8's *revoke everybody at once on unexpected traffic*.

Three things this module settles that the plan left open.

**A password is hashed with `hashlib.scrypt`** at n=16384, r=8, p=1 -- 16 MB and about
50 ms a check, which is a cost worth paying at one sign-in a day and one nobody can skip.
The stored string carries its own parameters, so the cost can be raised later without a
migration and an old row still verifies.

**A session token is stored as its SHA-256.** The cookie holds the random string; the
table holds a digest of it, so a database dump is not a drawer of live cookies. The
lookup is by digest and costs the same.

**Invitations expire in two weeks and a session in thirty days.** A link that works
forever is a link that works after the person forgot it was sent; thirty days is long
enough that a player is not signing in every visit and short enough that an abandoned
laptop stops being a way in. `seen` is updated no more than once an hour, so reading a
page is not a write.

Nothing here decides what a page shows. `Refused` is every no -- a wrong password, a spent
invitation, a handle somebody has -- and its message is written to be read by the person
who hit it.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Collection
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from chess_variant_sandbox.site import mail
from chess_variant_sandbox.web.page import marks

# ---------------------------------------------------------------------------- the rules

HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,23}$")
"""Letters, digits, hyphen and underscore, 3 to 24 characters, starting with a letter or
a digit. Narrow on purpose: a handle is printed on every page beside somebody else's
game, so the set it is drawn from is the first defence and escaping on the page is the
second (§6, task 1.10)."""

MIN_PASSWORD = 10
"""Length and nothing else. Composition rules buy little and cost every person a
rejected password they would have remembered."""

INVITE_LIFETIME = timedelta(days=14)
SESSION_LIFETIME = timedelta(days=30)

SEEN_RESOLUTION = timedelta(hours=1)
"""How stale `sessions.seen` is allowed to get before a request writes it. The column
answers *when was this session last used*, which does not need to be exact, and a write
on every request would be a write on every page."""

# scrypt, at the parameters the stored string records. 128 * r * n is the memory, so this
# is 16 MB; `hashlib.scrypt` needs `maxmem` above that or OpenSSL refuses.
SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
SALT_BYTES = 16
DIGEST_BYTES = 64
"""What `hashlib.scrypt` returns when no length is asked for. Read by `_nobody`."""

TOKEN_BYTES = 32
CODE_BYTES = 16

FORMER = "a former player"
"""What somebody who has left is called wherever a handle is printed.

The privacy notice's own words, and the whole of what `chess-variant-sandbox site forget` leaves
behind: the row keeps its id so the matches still point somewhere, and every reading of a
handle finds this instead. One string, read by `handle_of` here and by
`unbeaten.handles`, so the notice and the pages cannot drift apart.
"""


class Refused(ValueError):
    """A no the person on the other end is meant to read.

    A `ValueError`, so the local application's handler answers it as a 400 with the
    sentence in it; the site's handler answers a sign-in refusal as a 401. Every case is
    something about the request rather than a fault: a wrong password, a spent
    invitation, a handle that is taken.
    """


@dataclass(frozen=True)
class Invite:
    """One issued invitation, as the command that made it prints it."""

    code: str
    email: str
    uses: int


@dataclass(frozen=True)
class Viewer:
    """Who a cookie says is asking."""

    id: int
    handle: str
    role: str


# ------------------------------------------------------------------------- passwords


Stored = tuple[bytes, int, int, int, str]
"""A stored string read back: salt, n, r, p and the digest in hexadecimal."""


def hash_password(password: str) -> str:
    """A password as the string `users.password` holds.

    `scrypt$n$r$p$salt$digest`, all hexadecimal, so the parameters travel with the digest
    and a row hashed under today's cost still verifies after the cost is raised.
    """
    if len(password) < MIN_PASSWORD:
        raise Refused(f"a password needs at least {MIN_PASSWORD} characters")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify(stored: str | None, password: str) -> bool:
    """Whether this password made that string. False for an account with no password.

    A row with a null password is the loader's maintainer and an invitation not yet
    accepted; neither can sign in, and both answer here rather than anywhere else.

    **Every call costs one scrypt, whatever it is handed** (Q379). No row, a null
    password, a string `hash_password` did not write: each is checked against `_nobody()`
    instead and answers False, so how long the answer takes says nothing about whether
    there was a row behind it. `sign_in` passes `None` for a handle nobody holds.
    """
    ours = _parsed(stored)
    salt, n, r, p, digest = ours or _nobody()
    try:
        found = _scrypt(password, salt, n, r, p)
    except ValueError:
        return False
    matched = hmac.compare_digest(found.hex(), digest)
    return matched and ours is not None


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    """The one place a password is hashed, so a test can count what a request costs."""
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=SCRYPT_MAXMEM
    )


def _parsed(stored: str | None) -> Stored | None:
    """A stored string's salt, cost and digest, or `None` for one `hash_password` did
    not write. Nothing is hashed here."""
    if not stored:
        return None
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return None
        return bytes.fromhex(salt), int(n), int(r), int(p), digest
    except ValueError:
        return None


def _nobody() -> Stored:
    """What `verify` checks a password against when there is no row to check it against.

    Today's cost and a salt of today's length, so the check takes what a real row's
    takes, and a digest `verify` never lets match. Read at each call rather than frozen
    at import, so raising `SCRYPT_N` moves it with the rows it stands in for.
    """
    return bytes(SALT_BYTES), SCRYPT_N, SCRYPT_R, SCRYPT_P, "00" * DIGEST_BYTES


# --------------------------------------------------------------------------- invitations


def invite(
    connection: Any,
    email: str,
    *,
    issuer: int | None = None,
    uses: int = 1,
    lifetime: timedelta = INVITE_LIFETIME,
) -> Invite:
    """Write an invitation for one address and return it. `send_invitation` mails it.

    An address that already has an account is refused, because `accept` would refuse its
    link and the person would be mailed a code that cannot work. An address with only an
    earlier invitation is invited again. It gets a fresh code and a fresh expiry, and the
    earlier link keeps working until it expires or an account is made.
    """
    address = _address(email)
    if uses < 1:
        raise Refused("an invitation has to be good for at least one use")
    code = secrets.token_urlsafe(CODE_BYTES)
    with connection.cursor() as cursor:
        _unclaimed(cursor, address)
        cursor.execute(
            "insert into invites (code, issuer, uses_left, expiry, email)"
            " values (%s, %s, %s, now() + %s, %s)",
            (code, issuer, uses, lifetime, address),
        )
    return Invite(code=code, email=address, uses=uses)


def send_invitation(
    connection: Any,
    email: str,
    base: str,
    send: mail.Sender | None,
    *,
    issuer: int | None = None,
    uses: int = 1,
    lifetime: timedelta = INVITE_LIFETIME,
) -> tuple[Invite, str]:
    """Write an invitation and mail its link. `chess-variant-sandbox site invite` and `/admin` both call this.

    The row is written before the message goes and the caller commits, so a send that
    fails rolls the row back and leaves no code nobody was told about. `send=None` writes
    the row and mails nothing, for `--no-send`. The link comes back either way.
    """
    made = invite(connection, email, issuer=issuer, uses=uses, lifetime=lifetime)
    where = link(base, made.code)
    if send is not None:
        send(
            mail.invitation(
                made.email, where, _inviter(connection, issuer), lifetime.days
            )
        )
    return made, where


def _inviter(connection: Any, user: int | None) -> str | None:
    """The handle an invitation names its sender by, or None: no issuer, or one whose
    account has no handle (left, forgotten, or never finished joining)."""
    if user is None:
        return None
    with connection.cursor() as cursor:
        cursor.execute("select handle from users where id = %s", (user,))
        row = cursor.fetchone()
    return None if row is None else row[0]


def link(base: str, code: str) -> str:
    """The address in the invitation. One place, so the page and the mail agree."""
    return f"{base.rstrip('/')}/join?code={code}"


def accept(
    connection: Any,
    code: str,
    handle: str,
    password: str,
    *,
    reserved: Collection[str] = (),
) -> tuple[Viewer, str]:
    """Spend an invitation on a new account, and open a session for it.

    The spend and the account are one transaction -- the caller's -- so a handle that is
    taken leaves the invitation unspent rather than burning it on a failed form.

    `reserved` is handles nobody may claim although no row holds them yet: the
    maintenance account's, before its first sign-in writes it. They are refused at the
    point and in the words a held handle is (Q369), so no form -- a short password, a
    spent code -- answers differently for one of them than for anybody's.
    """
    name = _handle(handle)
    hashed = hash_password(password)
    with connection.cursor() as cursor:
        cursor.execute(
            "update invites set uses_left = uses_left - 1"
            " where code = %s and uses_left > 0 and (expiry is null or expiry > now())"
            " returning email, issuer",
            (code,),
        )
        found = cursor.fetchone()
        if found is None:
            raise Refused(
                "that invitation has expired or has already been used; "
                "ask whoever invited you for another"
            )
        email, issuer = found[0], found[1]
        if not email:
            raise Refused("that invitation names no address, so it cannot be accepted")
        _free(cursor, name, str(email), reserved)
        cursor.execute(
            "insert into users (handle, email, password, invited_by)"
            " values (%s, %s, %s, %s) returning id, handle, role",
            (name, str(email), hashed, issuer),
        )
        row = cursor.fetchone()
    viewer = Viewer(id=int(row[0]), handle=str(row[1]), role=str(row[2]))
    return viewer, open_session(connection, viewer.id)


def _free(cursor: Any, handle: str, email: str, reserved: Collection[str] = ()) -> None:
    """Refuse a handle or an address somebody already has, by name.

    The two unique constraints would refuse these anyway, as a failed transaction with a
    driver's message in it. This is the same refusal in a sentence the person can act on.
    A reserved handle is one condition with a held one, so the two cannot drift apart.
    """
    cursor.execute("select 1 from users where lower(handle) = lower(%s)", (handle,))
    held = cursor.fetchone() is not None
    if held or handle.lower() in {name.strip().lower() for name in reserved}:
        raise Refused(f"the handle {handle} is taken; choose another")
    cursor.execute("select 1 from users where email = %s", (email,))
    if cursor.fetchone() is not None:
        raise Refused("that address already has an account; sign in instead")


def _unclaimed(cursor: Any, address: str) -> None:
    """Refuse to invite an address that already has an account, naming the account.

    A forgotten account has no address left, so the person who left can be invited again.
    A killed one keeps its address, and the way back for it is letting the person back in.
    """
    cursor.execute(
        "select handle, revoked from users where lower(email) = %s", (address,)
    )
    found = cursor.fetchone()
    if found is None:
        return
    handle, revoked = found
    if revoked is not None:
        raise Refused(
            f"{address} already has an account ({handle}), which is shut out; "
            "let them back in instead of inviting them again"
        )
    raise Refused(
        f"{address} already has an account ({handle}), so there is nobody to invite; "
        "they can sign in"
    )


# ----------------------------------------------------------------------------- sessions


def sign_in(connection: Any, handle: str, password: str) -> tuple[Viewer, str]:
    """A handle and a password in, a viewer and a session token out.

    One refusal for both halves. Which of the handle and the password was wrong is
    information for whoever is guessing, and no use to whoever mistyped.

    And one scrypt for every attempt (Q379). A handle nobody holds, a revoked one and a
    row with no password all reach `verify` with nothing to check against, and `verify`
    spends on nothing what it would have spent on a row. Before, only a held handle cost
    the hash, so the time an answer took said whether the handle existed.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, handle, role, password from users"
            " where lower(handle) = lower(%s) and revoked is null",
            (handle.strip(),),
        )
        found = cursor.fetchone()
    stored = None if found is None else found[3]
    if not verify(stored, password):
        raise Refused("that handle and password do not match")
    viewer = Viewer(id=int(found[0]), handle=str(found[1]), role=str(found[2]))
    return viewer, open_session(connection, viewer.id)


def open_session(connection: Any, user: int, lifetime: timedelta = SESSION_LIFETIME) -> str:
    """A new session row, returning the token the cookie carries.

    The row holds the digest; the caller holds the token. Nothing can read a cookie back
    out of the table.
    """
    token = secrets.token_urlsafe(TOKEN_BYTES)
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into sessions (token, user_id, expires) values (%s, %s, now() + %s)",
            (digest_of(token), user, lifetime),
        )
    return token


def resolve(connection: Any, token: str | None) -> Viewer | None:
    """Who this cookie is, or `None` for a visitor.

    `None` covers every way of not being signed in -- no cookie, an unknown token, an
    expired session, a revoked account -- because the page has one thing to do about all
    four and that is to offer signing in.
    """
    if not token:
        return None
    hashed = digest_of(token)
    with connection.cursor() as cursor:
        cursor.execute(
            "select u.id, u.handle, u.role, s.seen from sessions s"
            " join users u on u.id = s.user_id"
            " where s.token = %s and (s.expires is null or s.expires > now())"
            "   and u.revoked is null",
            (hashed,),
        )
        found = cursor.fetchone()
        if found is None:
            return None
        cursor.execute(
            "update sessions set seen = now()"
            " where token = %s and seen < now() - %s",
            (hashed, SEEN_RESOLUTION),
        )
    return Viewer(id=int(found[0]), handle=str(found[1]), role=str(found[2]))


def sign_out(connection: Any, token: str | None) -> None:
    """Drop one session. A token nobody has is already signed out."""
    if not token:
        return
    with connection.cursor() as cursor:
        cursor.execute("delete from sessions where token = %s", (digest_of(token),))


def revoke_sessions(connection: Any) -> int:
    """Delete every session and say how many. §8's *revoke everybody at once*.

    Accounts, games and ratings are untouched: everybody is signed out and everybody can
    sign in again, which is the answer to unexpected traffic rather than to an
    unwelcome person -- that one is 1.8's per-user switch.
    """
    with connection.cursor() as cursor:
        cursor.execute("delete from sessions")
        return int(cursor.rowcount)


def mark_of(user: int | None, handle: str | None) -> dict[str, Any] | None:
    """A person's generated mark (R2, `web/page/marks.py`), for a payload beside their handle.

    None for nobody, and for somebody who left: they are *a former player* everywhere, so
    their games must not be told apart by a mark the forgetting left behind.
    """
    if not user or not handle or handle == FORMER:
        return None
    return marks.mark(user)


def handle_of(connection: Any, user: int) -> str:
    """What this person is called, for a page that has their id.

    A row whose handle has been emptied is somebody who left (`chess-variant-sandbox site forget`), and
    it reads as the privacy notice's own words rather than as a blank. An id nobody has
    is the empty string, which is the caller's *no such person*.
    """
    with connection.cursor() as cursor:
        cursor.execute("select handle from users where id = %s", (user,))
        found = cursor.fetchone()
    if found is None:
        return ""
    return FORMER if found[0] is None else str(found[0])


def members(connection: Any) -> int:
    """How many people are here: accounts with a handle that nobody shut out and nobody
    left. An invitation not yet accepted has no handle and is nobody yet (L3 of
    `docs/plan-launch.md` §3.3, *‹n› members*)."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from users"
            " where handle is not null and revoked is null and forgotten is null"
        )
        return int(cursor.fetchone()[0])


def digest_of(token: str) -> str:
    """What `sessions.token` holds for this cookie."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ reading


def _handle(handle: str) -> str:
    name = handle.strip()
    if not HANDLE.match(name):
        raise Refused(
            "a handle is 3 to 24 characters, letters, digits, hyphen and underscore, "
            "starting with a letter or a digit"
        )
    return name


def address(email: str) -> str:
    """An address as an invitation would be written to it, or `Refused`. The check a
    request for an invitation shares (`site/asking.py`), so the two cannot drift."""
    return _address(email)


def _address(email: str) -> str:
    """An address, lowered. One `@` with something either side is the whole check.

    Anything stricter refuses addresses that work, and the real check is that a person
    reads the invitation.
    """
    address = email.strip().lower()
    local, _, domain = address.partition("@")
    if not local or not domain or "." not in domain or any(c.isspace() for c in address):
        raise Refused(f"{email!r} does not look like an email address")
    return address
