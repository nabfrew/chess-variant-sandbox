"""The one email a member may switch on: when it becomes their move (Q572).

`docs/plan-return.md` §1.5 and the dated exception to test 3 of `docs/plan-meta-game.md`
(the user, 2026-10-01, *chosen B*): a member may opt in, **off by default**, to one email
when it is their move in a game between two people, and nothing else is ever sent. This
module is the box and the decision; `mail.your_move` is the message, and the server hands
it to the sender only after the move has committed.

## When one is sent

When a sent move lands that makes it the other person's move, and only then:

- the game is still running after the move (an ending is seen on the page, never mailed);
- the engine's side to move after it is the other seat, not the mover's again: whose move
  it is is the replayed position's answer (rule 3), and a game whose rules give one side
  two moves in a row sends nothing until the turn really passes, so it is **one email a
  turn** and never a second one before the person has moved;
- the person to move ticked the box, still has an address and can still sign in;
- they have not read the match in the last `OPEN_FOR` (the match page asks every 15
  seconds while it waits on the other person, so a recent read is a page that is open,
  and the move has been seen);
- the site has sent fewer than `quota.MOVE_EMAILS_A_DAY` in the last rolling day; past it
  the page is the only signal, and nothing waits to go out later.

Nothing else sends one: not a challenge, a draw offer, a game's end, the five days running
low or a game that waits. There is no reminder and no digest.

## A game in rounds (Q918f)

Both players owe a move from the moment a round opens, and a sent move is a seal, so the
email goes to the other player on each seal, never to the one who sealed, on the same
conditions as above: *their move is in*, naming no move, while the round is open; the
round's result (both moves and what came of them, the core's words) when the seal revealed
it, since the round that opens makes it their move again. The key is the game's length,
which in a game in rounds is the round's first ply, per recipient: a person mailed that a
round was revealed is not mailed again in that round that the other's move is in, so it is
still one email a turn.

## A failed send never blocks a move

`due` writes its row inside the move's transaction and returns the message; the server
hands it to the sender after the commit, catches any failure and writes it on the row
(`failed`). Nothing retries.

## Off

One click, no sign-in: `/mail/off/<token>`, in the body of every message and in its
`List-Unsubscribe` header. The token is the person's own (`users.mail_off`), made the first
time they tick the box and kept, so an old message's link still works. It can do nothing
but switch the box off, so a mail program that follows it ahead of the person does the
safe thing.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from chess_variant_sandbox.site import browse, identity, mail, matches, quota, rating
from chess_variant_sandbox.site.mail import RoundSaid

OPEN_FOR = timedelta(minutes=1)
"""How recently a player must have read a running match for it to count as open (§1.5)."""

BOX_SAID = "Email me when it's my move in a game against another person."
"""The box on a person's own page, s6 of the mock."""

BOX_UNDER = (
    "Off unless you tick it. One email each time it becomes your move, to the address "
    "your invitation came to, with a link that switches it off. The site sends nothing "
    "else, ever: no reminders and no news."
)
"""What the box does, said under it."""

OFF_SAID = (
    "Move emails are off. Nothing more will be sent to you. You can switch them back on "
    "from your own page."
)
"""What `/mail/off/<token>` says when the token is somebody's."""

OFF_UNKNOWN = "This link does not switch anything off: it is not one the site made."
"""What it says to a token nobody has."""

_LOCK = "move-emails"
"""The advisory lock the daily count is taken under, so two moves at once cannot pass the
cap together."""


# ------------------------------------------------------------------------- the box


def is_on(connection: Any, user: int) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("select move_email from users where id = %s", (user,))
        found = cursor.fetchone()
    return bool(found and found[0])


def set_on(connection: Any, user: int, on: bool) -> bool:
    """Tick or untick the box. Ticking makes the person's off token once and keeps it."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set move_email = %s, mail_off = coalesce(mail_off, %s)"
            " where id = %s and forgotten is null returning move_email",
            (bool(on), secrets.token_urlsafe(24), user),
        )
        found = cursor.fetchone()
    return bool(found and found[0])


def turn_off(connection: Any, token: str) -> bool:
    """The one-click off. Whether the token was somebody's; the box is off either way."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set move_email = false where mail_off = %s returning id",
            (token,),
        )
        return cursor.fetchone() is not None


def seen(connection: Any, found: matches.Row, user: int | None) -> None:
    """Note that one of a running match's two players has it open (a read of it)."""
    if user is None or not found.running:
        return
    seat = found.seat_of(rating.person(user))
    if seat not in (0, 1):
        return
    with connection.cursor() as cursor:
        cursor.execute(
            f"update matches set seen{seat} = now() where id = %s", (found.id,)
        )


def sent_today(connection: Any) -> int:
    """How many move emails went in the last rolling day, failed ones included."""
    with connection.cursor() as cursor:
        cursor.execute("select count(*) from move_emails where sent > now() - interval '1 day'")
        found = cursor.fetchone()
    return 0 if found is None else int(found[0])


# ---------------------------------------------------------------------- the decision


@dataclass(frozen=True)
class Due:
    """A move email written down and ready for the sender: its row, and the message."""

    id: int
    message: mail.Message


def due(
    connection: Any,
    match_id: str,
    mover: int,
    snapshot: dict[str, Any],
    before: int | None = None,
) -> Due | None:
    """The move email this move sends, written down, or None when it sends none.

    `mover` is the seat that just moved and `snapshot` the position after it, whose side
    to move is the engine's. Every condition is the module's list; the row is written
    under the cap's lock, and `(match, recipient, ply)` makes a second one for the same
    turn impossible.

    In a game in rounds (`snapshot` has a `round`, Q918f) the move was a seal, and the
    email goes to the other player, never the one who sealed: *their move is in* (no move
    named) when the round is still open, or the round's result when this seal revealed it
    (`before`, the game's length before the seal, is shorter than it is now), since that
    opens the next round and so makes it their move again.
    """
    found = matches.row(connection, match_id)
    rounds = "round" in snapshot
    after = 1 - mover if rounds else snapshot.get("sideToMove")
    if not found.running or after not in (0, 1) or after == mover:
        return None
    revealed = rounds and before is not None and found.plies > before
    recipient = found.player(int(after)).user
    if recipient is None:
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            f"select u.email, u.move_email, u.mail_off, u.revoked, u.forgotten,"
            f" m.seen{int(after)} > now() - %s"
            f" from users u, matches m where u.id = %s and m.id = %s",
            (OPEN_FOR, recipient, match_id),
        )
        person = cursor.fetchone()
    if person is None:
        return None
    email, on, token, revoked, forgotten, open_now = person
    if not on or not email or not token or revoked is not None or forgotten is not None:
        return None
    if open_now:
        return None
    try:
        site = mail.site_url()
    except mail.MailError:
        return None
    with connection.cursor() as cursor:
        cursor.execute("select pg_advisory_xact_lock(hashtext(%s))", (_LOCK,))
    if sent_today(connection) >= quota.MOVE_EMAILS_A_DAY:
        return None
    history = snapshot.get("history") or []
    played = str(history[-1].get("text", "")) if history else ""
    moved_by = found.player(mover).user
    message = mail.your_move(
        to=str(email),
        game=browse.names(connection, [found.game]).get(found.game) or "a game",
        opponent=identity.handle_of(connection, moved_by or 0) or identity.FORMER,
        played=None if rounds else (played or "a move"),
        link=f"{site}/#/play/{found.id}",
        off=f"{site}/mail/off/{token}",
        revealed=_round_said(history, int(after)) if revealed else None,
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into move_emails (match, recipient, ply) values (%s, %s, %s)"
            " on conflict (match, recipient, ply) do nothing returning id",
            (found.id, recipient, found.plies),
        )
        written = cursor.fetchone()
    if written is None:
        return None
    return Due(id=int(written[0]), message=message)


def _round_said(history: list[dict[str, Any]], reader: int) -> RoundSaid:
    """The round just revealed, for the email to `reader`: its number, each move from the
    reader's side (*you played*, *they played*) and the round's sentences, all the core's
    as the reveal recorded them (`web.play.round_reveal`), none worked out here."""
    pair = history[-2:]
    mine = next((entry for entry in pair if entry.get("player") == reader), {})
    theirs = next((entry for entry in pair if entry.get("player") != reader), {})
    reveal = pair[-1].get("reveal") or {} if pair else {}
    return RoundSaid(
        number=int(pair[-1].get("round", 0)) + 1 if pair else 0,
        yours=str(mine.get("text") or "a move"),
        theirs=str(theirs.get("text") or "a move"),
        words=tuple(str(line) for line in reveal.get("words") or ()),
    )


def failed(connection: Any, row: int, why: str) -> None:
    """Write the sender's refusal on the row. The move it was for stands."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update move_emails set failed = %s where id = %s", (why[:300] or "failed", row)
        )
