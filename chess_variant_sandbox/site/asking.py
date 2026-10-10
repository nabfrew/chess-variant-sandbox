"""A visitor asks for an invitation, and the maintainer grants it or not (Q831).

`docs/plan-website.md`, *Public or invite-only, indefinitely*, decided 2026-10-10. The
user: *"Make a request invite button, with an optional message, suggesting they say how
they found the site."* Accounts stay by invitation, sent from `/admin` or from a member's
allowance (`site/inviting.py`); this is the third door to an invitation, and it opens
nothing by itself. A visitor leaves an address and, if they like, a note; `/admin` lists
the pending requests and the maintainer grants one, which sends **the ordinary
invitation** (`identity.send_invitation`, the maintainer as its issuer), or declines it.

**What the visitor is told** is one sentence, `CONFIRMED`, whatever happened to the
request: written, or quietly not written because the address already has an account or
already has a request kept. So the form cannot be used to learn whether an address is a
member's or has asked before, and nothing about the outcome reaches the visitor but the
invitation itself, if one is sent. Only a malformed form, and the two caps, are refused in
words, and none of those sentences names or depends on the address.

**The limits**, the plan's threat notes (`docs/plan-launch.md` §2 S5: scripted writes,
abusive text) sized to the one write a stranger makes on the site:

- *A cap per address*: one kept row an address, the table's unique index
  (`0034_invite_asks.sql`). A repeat writes nothing. An answered row is deleted
  `quota.ASK_KEPT` after its answer (`sweep`, run on every request and on every read of
  `/admin`), and the address may ask again after that.
- *A daily cap overall*: `quota.ASKS_A_DAY` rows written in a rolling day, site-wide.
- *A cap per connection*: `quota.ASKS_A_SOURCE` a day from one network address, in memory
  (`quota.Asks`), so one source cannot spend the site's day.
- *The text*: `checked` lowers and checks the address as an invitation's is checked
  (`identity`'s rule, one `@` with something either side, no spaces), caps it at
  `quota.ASK_EMAIL_LENGTH`, and refuses control characters, which PostgreSQL's `text` would
  refuse with a fault; the note is trimmed, its control characters other than a line break
  dropped, empty read as none, and capped at `quota.ASK_NOTE_LENGTH`. Both caps are check
  constraints too. Like a suggestion, the note is stored as typed and every page that shows
  it sets `textContent`, so a note full of tags is a note full of tags.
- *No mail*: nothing is sent when a request arrives, to the visitor or to the maintainer.
  The one message this can lead to is the invitation the maintainer chooses to send.

Nothing here commits; the caller owns the transaction, as in `suggestions` and `reports`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import identity, mail, quota

PATH = "/request-invitation"
"""The page with the form. A constant page: it reads no cookie and no store."""

BUTTON = "Request an invitation"
"""The words on the way to the form, wherever the site says accounts are by invitation."""

CONFIRMED = (
    "Thank you. Your request has gone to the person who runs the site. If they send you "
    "an invitation, it comes to that address by email, and the site sends nothing else."
)
"""The one answer a request gets, written or not (see the module's docstring)."""

NOT_AN_ADDRESS = (
    "That does not look like an email address. The invitation is sent to one, so it is "
    "the one thing a request needs."
)
"""The refusal of a missing or malformed address."""

NOTE_HINT = (
    "Optional. How did you find the site? A friend, a forum, a search: a line about it "
    "helps."
)
"""The note's hint on the form: the user's own suggestion, that they say how they found
the site."""

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
"""Control characters but the tab and the line break, which a note may hold. Applied
after line endings are made `\\n`, so a carriage return is never left to strip."""


class AskRefused(ValueError):
    """A request that cannot be taken as written. A `ValueError`, so the answer is a 400
    carrying the sentence the form prints."""


class AskGone(ValueError):
    """`/admin` answered a request that is no longer pending."""


@dataclass(frozen=True)
class Ask:
    """One pending request, as `/admin` lists it.

    `invited` is when the address was last sent an invitation, if ever: an address can ask
    while an earlier invitation sits in its junk folder, and granting it sends a fresh one.
    """

    id: int
    email: str
    note: str | None
    created: datetime
    invited: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "note": self.note,
            "created": self.created.isoformat(),
            "invited": None if self.invited is None else self.invited.isoformat(),
        }


def checked(email: Any, note: Any) -> tuple[str, str | None]:
    """The address and the note as they would be kept, or the sentence refusing them.

    Pure, and first, so a form with a typo costs the visitor none of their tries and the
    site no query.
    """
    said = email if isinstance(email, str) else ""
    if len(said.strip()) > quota.ASK_EMAIL_LENGTH or _CONTROL.search(said.strip()):
        raise AskRefused(NOT_AN_ADDRESS)
    try:
        address = identity.address(said)
    except identity.Refused as error:
        raise AskRefused(NOT_AN_ADDRESS) from error
    return address, _note(note)


def _note(note: Any) -> str | None:
    if note is None:
        return None
    if not isinstance(note, str):
        raise AskRefused("A note is text: a line or two about how you found the site.")
    kept = _CONTROL.sub("", note.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not kept:
        return None
    if len(kept) > quota.ASK_NOTE_LENGTH:
        raise AskRefused(
            f"A note is up to {quota.ASK_NOTE_LENGTH} characters and this one is "
            f"{len(kept)}. A line or two is plenty."
        )
    return kept


def ask(connection: Any, address: str, note: str | None) -> bool:
    """Keep one request, unless the address already has an account or a kept request.

    `address` and `note` are `checked`'s. Answers whether a row was written, which the
    caller does not pass on: the visitor reads `CONFIRMED` either way. The site's daily cap
    is checked before the address is looked at, so a full day refuses every address alike.
    """
    sweep(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from invite_asks where created > now() - interval '1 day'"
        )
        if int(cursor.fetchone()[0]) >= quota.ASKS_A_DAY:
            raise quota.Spent(quota.ASKS_FULL)
        cursor.execute("select 1 from users where lower(email) = %s", (address,))
        if cursor.fetchone() is not None:
            return False
        cursor.execute(
            "insert into invite_asks (email, note) values (%s, %s)"
            " on conflict (email) do nothing returning id",
            (address, note),
        )
        return cursor.fetchone() is not None


def sweep(connection: Any) -> None:
    """Delete the requests answered more than `quota.ASK_KEPT` ago."""
    with connection.cursor() as cursor:
        cursor.execute(
            "delete from invite_asks where answered is not null and answered < now() - %s",
            (quota.ASK_KEPT,),
        )


def pending(connection: Any) -> list[Ask]:
    """Every request nobody has answered yet, oldest first: the order they were made in."""
    sweep(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "select a.id, a.email, a.note, a.created,"
            " (select max(i.created) from invites i where lower(i.email) = a.email)"
            " from invite_asks a where a.answered is null order by a.created, a.id"
        )
        return [
            Ask(id=int(row[0]), email=str(row[1]), note=row[2], created=row[3], invited=row[4])
            for row in cursor.fetchall()
        ]


def answered(connection: Any) -> dict[str, int]:
    """How many requests were granted and declined in the kept window, for `/admin`."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) filter (where granted), count(*) filter (where not granted)"
            " from invite_asks where answered is not null"
        )
        granted, declined = cursor.fetchone() or (0, 0)
    return {"granted": int(granted or 0), "declined": int(declined or 0),
            "days": quota.ASK_KEPT.days}


def grant(
    connection: Any, ask_id: int, base: str, sender: mail.Sender | None, issuer: int | None
) -> str:
    """Send the ordinary invitation to a pending request's address, and mark it granted.

    The row is locked while the invitation is written and mailed, so two presses send one
    invitation. In the caller's transaction: an address that has gained an account since it
    asked is refused by `identity.invite`, and a send that fails raises, and either way the
    request is left pending. Answers the address.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select email from invite_asks where id = %s and answered is null for update",
            (ask_id,),
        )
        found = cursor.fetchone()
    if found is None:
        raise AskGone("that request has already been answered")
    made, _ = identity.send_invitation(connection, str(found[0]), base, sender, issuer=issuer)
    _answer(connection, ask_id, granted=True)
    return made.email


def decline(connection: Any, ask_id: int) -> str:
    """Mark a pending request declined. Nothing is sent. Answers the address."""
    found = _answer(connection, ask_id, granted=False)
    if found is None:
        raise AskGone("that request has already been answered")
    return found


def _answer(connection: Any, ask_id: int, *, granted: bool) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute(
            "update invite_asks set answered = now(), granted = %s"
            " where id = %s and answered is null returning email",
            (granted, ask_id),
        )
        found = cursor.fetchone()
    return None if found is None else str(found[0])
