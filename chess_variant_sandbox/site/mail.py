"""One outgoing message, through one transactional sender.

Task 1.3. The site sends two kinds of mail and builds them here and nowhere else: an
invitation carrying a link, and (Q572) the one email a member may switch on, when it
becomes their move in a game between two people (`your_move`). Each goes through one
function behind a small interface: an address, a subject, a body and any headers go in,
and either the message was accepted or a `MailError` says why. Tests pass their own
function in and never reach the network, and `tests/test_site_move_email.py` holds the
package to these two builders (test 3 of `docs/plan-meta-game.md`).

The service is Resend (the user's choice, 2026-09-13), reached over its HTTPS API with
`urllib` so the site takes no dependency for one POST. Two variables configure it and
they are read here and nowhere else, the way `db.py` owns the database URL:

    MAIL_API_KEY    the service's key; secret, so the box keeps it in a file of its own
    MAIL_FROM       the address mail is sent as, which the service's DNS records cover;
                    `from_field` puts the site's name before a bare one (Q639)

`CHESS_VARIANT_SANDBOX_SITE_URL` sits beside them because a link in an invitation has to name the site
from the outside, and the app behind Caddy never sees that address on a request it
answers. All three are absent on a development machine, which is why `chess-variant-sandbox site invite
--no-send` prints the link instead.
"""

from __future__ import annotations

import html
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

API_URL = "https://api.resend.com/emails"

KEY_VAR = "MAIL_API_KEY"
FROM_VAR = "MAIL_FROM"
URL_VAR = "CHESS_VARIANT_SANDBOX_SITE_URL"

# Long enough for a slow API, short enough that a maintainer's command does not appear to
# hang. Nothing retries: a send that failed is one the person can run again.
TIMEOUT = 30.0

# The service sits behind Cloudflare, which refuses urllib's default `Python-urllib/3.x`
# with a 403 and error code 1010.
USER_AGENT = "chess-variant-sandbox-site/1"


class MailError(ValueError):
    """The message was not sent.

    A `ValueError` like `db.ConfigError`, so `cli.main` prints it as one line and exits 2.
    Every case is a missing variable or the service saying no, and neither is a fault in
    this process.
    """


@dataclass(frozen=True)
class Message:
    """What a sender takes: who it goes to, what it says."""

    to: str
    subject: str
    body: str
    headers: tuple[tuple[str, str], ...] = ()


# What a sender is. The live one posts to the service; a test's records the message.
Sender = Callable[[Message], None]


def site_url() -> str:
    """The address the site answers on from the outside, with no trailing slash."""
    found = os.environ.get(URL_VAR, "").strip().rstrip("/")
    if not found:
        raise MailError(
            f"{URL_VAR} is not set, so a link in an invitation would have no site to "
            f"point at (the box's unit reads it from /etc/chess-variant-sandbox/mail.env)"
        )
    return found


def sender_address() -> str:
    """Who the mail comes from."""
    found = os.environ.get(FROM_VAR, "").strip()
    if not found:
        raise MailError(f"{FROM_VAR} is not set, so there is no address to send as")
    return found


SITE_NAME = "Chess variant sandbox"
"""The name a mail program shows as the sender (Q639). A bare address on a new domain, with
no name before it, is one more thing that makes a message look like it came from nobody."""


def from_field() -> str:
    """The `From` the service is given: the site's name, then the address.

    `MAIL_FROM` is a bare address on the box (Q639 found it so). One that already carries
    a name (`Name <address>`) is passed through as written, so the box can change the name
    without a release.
    """
    found = sender_address()
    if "<" in found:
        return found
    return f"{SITE_NAME} <{found}>"


def html_of(body: str) -> str:
    """The HTML part, made from the plain text and saying nothing it does not (Q639).

    A paragraph is a run of lines between blank ones; a paragraph that is one indented
    link becomes that link, its visible text the address itself, so what a reader sees is
    where it goes. No styles, no images, no tracking, nothing remote: a mail program that
    prefers HTML shows the same words as one that reads the text.
    """
    parts = []
    for paragraph in body.strip().split("\n\n"):
        words = " ".join(paragraph.split())
        if paragraph.startswith("    ") and " " not in words and "://" in words:
            link = html.escape(words, quote=True)
            parts.append(f'<p><a href="{link}">{link}</a></p>')
        else:
            parts.append(f"<p>{html.escape(words, quote=False)}</p>")
    return '<!DOCTYPE html><html lang="en"><body>' + "\n".join(parts) + "</body></html>"


def configured() -> bool:
    """Whether this machine can send at all. Printed at start; never a promise it works."""
    return bool(os.environ.get(KEY_VAR, "").strip() and os.environ.get(FROM_VAR, "").strip())


PURPOSE = (
    "a place to find chess-like games worth playing. People invent games there, in a "
    "designer that needs no programming, and anyone can play them in the browser, "
    "against the computer or against each other. What no other site does is measure "
    "them: the computer plays every game against itself hundreds of times and scores how "
    "interesting the play is. That score is the best estimate available today, not a "
    "verdict, and the people who play there are the test the computer cannot replace."
)
"""What the site is, in the invitation: `docs/plan-understanding.md` §1.1's statement of
purpose (tone A), its sentences as written but for the clause on what the score weighs
and the one on remeasuring, which an email has no room for. The join page carries the
whole paragraph (`pages.PURPOSE`)."""


JUNK_TIP = (
    "If this was in your junk folder, please mark it not junk: the site is new, and mail "
    "programs don't know it yet."
)
"""The invitation's last line (Q639): the trial's first email went to junk on the user's
own phone, and a person who marks it *not junk* teaches their mail program the sender, so a
move email they switch on later arrives. `docs/plan-launch.md` §5.3 prints it for the user."""


def _lasts(days: int) -> str:
    """How long an invitation's link works, in words."""
    if days == 14:
        return "two weeks"
    if days == 7:
        return "a week"
    return "a day" if days == 1 else f"{days} days"


def invitation(to: str, link: str, inviter: str | None = None, days: int = 14) -> Message:
    """The message that brings a person in, one of the two the site sends.

    Plain text, no markup and no images (`send` adds an HTML part made from it, saying
    the same words). `docs/plan-launch.md` §5.3 (L8): who invited
    them, since a message from an unknown sender about an unknown site reads as spam; what
    the site is; that this is a trial and what they are asked to do; that it costs nothing
    and nothing counts down; and the one promise test 3 needs, that no other email comes
    unless they switch on the move email themselves (Q572). `inviter` is the issuer's
    handle, absent for an invitation nobody signed (`site invite` with no `--issuer`).
    """
    who = inviter or "Somebody"
    tell = inviter or "us"
    return Message(
        to=to,
        subject=(
            f"{inviter} invited you to try Chess variant sandbox"
            if inviter
            else "You're invited to try Chess variant sandbox"
        ),
        body=(
            f"{who} has invited you to Chess variant sandbox: {PURPOSE}\n\n"
            "This is a trial: a few people are invited before anybody else, and some "
            "things will be rough. Hearing where is the reason for it.\n\n"
            f"Open this link to choose a handle and a password:\n\n    {link}\n\n"
            "What we'd love: play a few games against the computer, try one you've never "
            f"seen, and tell {tell} what confused you. You can make a game of your own too. "
            "It costs nothing and takes as long as you like; nothing on the site counts "
            "down.\n\n"
            "The site sends you no other email unless you ask it to: you can switch on one "
            "email for when it's your move in a game against another person, and switch it "
            f"off in one click. The link works once and expires in {_lasts(days)}; if you "
            "weren't expecting it, ignore it and nothing happens.\n\n"
            f"{JUNK_TIP}\n"
        ),
    )


@dataclass(frozen=True)
class RoundSaid:
    """A round of a game in rounds, revealed, as the move email tells it (Q918f): its
    number, the reader's move and the other's, and the round's sentences (the core's
    events in words, `web.play.round_reveal`); none when both moves did what they said."""

    number: int
    yours: str
    theirs: str
    words: tuple[str, ...] = ()
    # A spy game's announced round (Q935e): who announced, from the reader's side --
    # `you` or `they` -- or empty for a round chosen in secret.
    announced: str = ""


ASKED = ("reply", "announce", "answer", "seal")
"""What a spy game's move email can ask of its reader (Q935e), beside a plain move."""

_ASKED_SAID = {
    "reply": "Your spy is lost, so it's your move: one ordinary reply, and then the game is "
    "played in rounds:",
    "announce": "It's your turn to announce: your move is shown to {opponent}, who answers "
    "it, and then both are made at once:",
    "answer": "Your answer is due: choose your move with theirs in view, and both are made "
    "at once:",
    "seal": "Both spies are lost, so from now on each of you seals a move in secret. It's "
    "your move: seal yours, and both are made at once:",
}

_SUBJECTS = {"announce": "Your announcement", "answer": "Your answer"}


def your_move(
    to: str,
    game: str,
    opponent: str,
    played: str | None,
    link: str,
    off: str,
    revealed: RoundSaid | None = None,
    announced: str | None = None,
    asked: str | None = None,
) -> Message:
    """The move email (Q572, `docs/plan-return.md` §1.5, s6 of the mock).

    Whose move, in which game, the link to it, and how to switch these off: one click, no
    sign-in, the same address in the `List-Unsubscribe` header with RFC 8058's one-click
    POST beside it, so a mail program's own *unsubscribe* works too. Nothing else: no
    time left, no other game, no count. Built only for a person who switched it on
    (`site/move_email.py`).

    In a game in rounds (Q918f) there is no move to name while the round is open:
    `played` is None and the email says the other's move is in, sealed. `revealed` is the
    round the reader's own seal was waiting on, now revealed, and the email says what both
    moves were and what came of them.

    A spy game (schema 27, Q935e, `docs/plan-spy.md` §3.7) says what is asked of the
    reader, as one of `ASKED`: `reply` (their spy is lost; one ordinary move, then
    rounds), `announce` (their move first, in the open), `answer` (to `announced`, the
    other's announcement, named since it is shown to both) or `seal` (both spies lost: the
    rounds are secret from now). None says it as any game does.
    """
    if asked is not None and asked not in ASKED:
        raise ValueError(f"no such thing to ask: {asked}")
    if revealed is not None:
        said = " ".join(revealed.words)
        if revealed.announced == "you":
            both = f"you announced {revealed.yours} and {opponent} answered {revealed.theirs}"
        elif revealed.announced == "they":
            both = f"{opponent} announced {revealed.theirs} and you answered {revealed.yours}"
        else:
            both = f"you played {revealed.yours} and {opponent} played {revealed.theirs}"
        happened = (
            f"Round {revealed.number} of your game of {game} against {opponent} is "
            f"revealed: {both}.{' ' + said if said else ''}"
        )
        then = "It's your move for the next round:"
    elif announced is not None:
        happened = f"{opponent} announced {announced} in your game of {game}."
        then = _ASKED_SAID["answer"]
    elif played is None:
        happened = f"{opponent}'s move is in, sealed, in your game of {game}."
        then = "It's your move: seal yours, and both are made at once:"
    else:
        happened = f"{opponent} played {played} in your game of {game}."
        then = "It's your move:"
    if asked is not None:
        then = _ASKED_SAID[asked].format(opponent=opponent)
    opening = f"{happened} {then}\n\n"
    return Message(
        to=to,
        subject=f"{_SUBJECTS.get(asked or '', 'Your move')}: {game} against {opponent}",
        body=(
            f"{opening}"
            f"    {link}\n\n"
            "You get this because you switched on move emails: one email each time it "
            "becomes your move in a game against another person, and nothing else. "
            "Switch them off (one click, no sign-in):\n\n"
            f"    {off}\n"
        ),
        headers=(
            ("List-Unsubscribe", f"<{off}>"),
            ("List-Unsubscribe-Post", "List-Unsubscribe=One-Click"),
        ),
    )


def send(message: Message) -> None:
    """Hand one message to the service, or say why it would not take it."""
    key = os.environ.get(KEY_VAR, "").strip()
    if not key:
        raise MailError(
            f"{KEY_VAR} is not set, so nothing can be sent from here; "
            f"`chess-variant-sandbox site invite --no-send` prints the link instead"
        )
    fields: dict[str, Any] = {
        "from": from_field(),
        "to": [message.to],
        "subject": message.subject,
        "text": message.body,
        "html": html_of(message.body),
    }
    if message.headers:
        fields["headers"] = dict(message.headers)
    payload = json.dumps(fields).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            _accepted(answer.status, answer.read())
    except urllib.error.HTTPError as error:
        _accepted(error.code, error.read())
    except OSError as error:
        raise MailError(f"the sending service could not be reached: {error}") from error


def _accepted(status: int, body: bytes) -> None:
    """What the service said, as a sentence when it is a refusal.

    The body is the service's own message and can name the address and the key's
    permissions, so it is trimmed rather than printed whole.
    """
    if 200 <= status < 300:
        return
    detail = _detail(body)
    raise MailError(f"the sending service refused the message ({status}): {detail}")


def _detail(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    try:
        found: Any = json.loads(text)
    except ValueError:
        return text[:200] or "no reason given"
    if isinstance(found, dict):
        message = found.get("message") or found.get("error") or ""
        if message:
            return str(message)[:200]
    return text[:200]
