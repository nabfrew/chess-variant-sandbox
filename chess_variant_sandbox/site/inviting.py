"""A member's own invitations: how many they may send, and sending one (Q719).

`docs/plan-launch.md` §5.3a. The person who runs the site gives a member a number of
invitations on `/admin` (`users.invites_to_give`, zero by default, `0028_invites_to_give.sql`);
the member sends them from their own page, one address at a time, and each one sent takes
one off. The invitation is the ordinary one: `identity.send_invitation`, the same row, the
same mail and the same link, with the member as its issuer, so the mail names them as the
person who invited. The member writes nothing but the address.

**The limits.** The allowance is the limit: zero unless somebody gave some, and taken off
in the same statement that checks it, so two presses at once cannot both spend the last
one. It is taken off in the same transaction as the invitation's row and its mail, so an
address that is refused, or a send that fails, gives it back. Nothing else is new: the
address is checked by `identity.invite` as an invitation from `/admin` is, and the front
door's connection limit holds as it does for every request.

**What a member is told.** An address that already has an account is refused without the
account's handle, which `/admin` shows and a member has no business reading.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.site import identity, mail

MOST = 50
"""The most invitations `/admin` may give one person at a time. A typing slip guard: ten
invited people each asked to bring a few is the scale this is for."""

NONE_LEFT = (
    "You have no invitations left to send. Ask the person who runs the site if you "
    "would like to invite somebody else."
)
"""The refusal when the count is zero, in the words the page shows."""

ALREADY_JOINED = (
    "That address already has an account, so there is nobody to invite: they can sign in."
)
"""The refusal for an address with an account, without whose account it is."""


class InviteRefused(ValueError):
    """An invitation a member asked for cannot be sent. A `ValueError`, so the answer is a
    400 carrying the sentence the page prints beside the form."""


def to_give(connection: Any, user: int) -> int:
    """How many invitations this person may still send."""
    with connection.cursor() as cursor:
        cursor.execute("select invites_to_give from users where id = %s", (user,))
        found = cursor.fetchone()
    return 0 if found is None else int(found[0])


def set_to_give(connection: Any, user: int, count: Any) -> tuple[str, int]:
    """Give this person `count` invitations to send, replacing what they had.

    `/admin`'s act. A number from zero to `MOST`; somebody who has left, or is shut out,
    is refused, since they cannot sign in to send one. Answers the handle and the count.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise InviteRefused("the number of invitations has to be a whole number")
    if not 0 <= count <= MOST:
        raise InviteRefused(f"the number of invitations has to be from 0 to {MOST}")
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set invites_to_give = %s"
            " where id = %s and forgotten is null and revoked is null returning handle",
            (count, user),
        )
        found = cursor.fetchone()
    if found is None:
        raise InviteRefused(f"user {user} cannot sign in, so cannot send invitations")
    return str(found[0]), count


def send(
    connection: Any, user: int, email: str, base: str, sender: mail.Sender | None
) -> tuple[identity.Invite, int]:
    """Spend one of this person's invitations on one address, and mail it.

    The count is taken off first, under the check that there is one, and the invitation
    written and mailed after it in the caller's transaction, so any refusal or failed send
    after the take raises and the caller's rollback puts the count back. Answers the
    invitation and how many are left.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set invites_to_give = invites_to_give - 1"
            " where id = %s and invites_to_give > 0"
            " and forgotten is null and revoked is null"
            " returning invites_to_give",
            (user,),
        )
        found = cursor.fetchone()
    if found is None:
        raise InviteRefused(NONE_LEFT)
    try:
        made, _ = identity.send_invitation(connection, email, base, sender, issuer=user)
    except identity.Refused as error:
        if "already has an account" in str(error):
            raise InviteRefused(ALREADY_JOINED) from error
        raise InviteRefused(_sentence(str(error))) from error
    return made, int(found[0])


def _sentence(said: str) -> str:
    """`identity`'s refusals are clauses; the page prints a sentence."""
    said = said.strip()
    return (said[:1].upper() + said[1:]).rstrip(".") + "." if said else said


def box(connection: Any, user: int) -> dict[str, Any] | None:
    """What a person's own page draws under *Invite somebody*, or None for nothing.

    Drawn for somebody who has invitations to send, or who has sent one of their own, so
    a member nobody gave any to sees no form they cannot use. `left` is the plain count.
    """
    left = to_give(connection, user)
    with connection.cursor() as cursor:
        cursor.execute("select count(*) from invites where issuer = %s", (user,))
        sent = int(cursor.fetchone()[0])
    if left == 0 and sent == 0:
        return None
    return {"left": left, "said": left_said(left), "sent": sent,
            "under": "They get an email from the site, naming you, with a link to choose "
                     "a handle and a password. The link works once and expires in "
                     f"{identity.INVITE_LIFETIME.days} days."}


def left_said(left: int) -> str:
    """The plain count, as the page says it."""
    if left <= 0:
        return NONE_LEFT
    return (
        "You can invite one more person."
        if left == 1
        else f"You can invite {left} more people."
    )
