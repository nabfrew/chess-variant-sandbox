"""The suggestion box: an idea for a rule, or an idea for the site, in one row.

Q279, from the user's inbox of 2026-09-17: *"Add a place to add ideas of game-rules or
other website ideas that could be added."*

A suggestion is a short text and one of two kinds. Anybody signed in can post one and
read their own; the maintainer reads all of them on `/admin` and marks one done or
deletes it. Nothing here is public, which is the reversible default -- making the box
public later is a reader and a page, while un-publishing what people wrote in private is
not possible.

**Nothing reads this table but people.** No entry becomes a queue item, a `docs/` line or
a job, and no mail is sent when one is posted. The box is a place to write something
down, and what happens to it afterwards is somebody's judgement.

**The text is text.** It is stored as it was typed and every page that shows it sets
`textContent`, so a suggestion full of tags is a suggestion full of tags. There is no
escaping step to forget because there is no markup step at all, which is the rule
`web/changelog.py` already follows.

Two limits hold without anybody watching. `quota.SUGGESTION_LENGTH` caps the text and
`quota.SUGGESTIONS_A_DAY` caps how many one person may post in a rolling day; the first
is also a check constraint on the column, so the cap is true of the table rather than of
this module.

Nothing here commits. The caller owns the transaction, the same rule `votes` and
`matches` follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import identity, quota

RULE = "rule"
SITE = "site"
KINDS: tuple[str, ...] = (RULE, SITE)
"""The two kinds, as the column's check constraint spells them. A third would be a
string the page cannot produce."""

LABELS: dict[str, str] = {
    RULE: "a rule or game idea",
    SITE: "something for the site",
}
"""What each kind is called wherever one is printed. Held here so the form's two
buttons, the poster's own list and the maintainer's table read the same words."""


class SuggestionError(ValueError):
    """A suggestion cannot be recorded as asked.

    A `ValueError` like `votes.VoteError`, so the server answers 400 with the sentence
    rather than a constraint violation from Postgres.
    """


@dataclass(frozen=True)
class Suggestion:
    """One idea somebody wrote down.

    `handle` is filled by the readings, which join the users table; a person who has left
    reads as `identity.FORMER` there, exactly as they do on a match and on a badge.
    """

    id: int
    author: int
    kind: str
    body: str
    created: datetime | None = None
    done: datetime | None = None
    handle: str = ""

    @property
    def label(self) -> str:
        """The kind in words, for a page that prints it beside the text."""
        return LABELS.get(self.kind, self.kind)

    @property
    def state(self) -> str:
        return "done" if self.done is not None else "open"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "body": self.body,
            "state": self.state,
            "handle": self.handle,
            "created": None if self.created is None else self.created.isoformat(),
            "done": None if self.done is None else self.done.isoformat(),
        }


def post(connection: Any, author: int, kind: str, body: str) -> Suggestion:
    """Write one suggestion down, or refuse it in a sentence the poster can read.

    The three refusals are the kind, an empty text and a text over the cap, and each one
    says what it wanted. The rolling-day allowance is checked first and separately
    (`quota.check_new_suggestion`), because a person who has hit it should be told that
    rather than being told their text is fine.
    """
    quota.check_new_suggestion(connection, author)
    said = str(body).strip()
    if kind not in KINDS:
        named = " or ".join(f"{one} ({LABELS[one]})" for one in KINDS)
        raise SuggestionError(f"a suggestion is {named}, not {kind!r}")
    if not said:
        raise SuggestionError("write the suggestion first; an empty one says nothing")
    if len(said) > quota.SUGGESTION_LENGTH:
        raise SuggestionError(
            f"a suggestion is up to {quota.SUGGESTION_LENGTH} characters and this one is "
            f"{len(said)}; the box is for the idea rather than the whole design"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into suggestions (author, kind, body) values (%s, %s, %s)"
            " returning id, author, kind, body, created, done",
            (author, kind, said),
        )
        return _to_suggestion(cursor.fetchone())


def mine(connection: Any, author: int) -> list[Suggestion]:
    """What this person has posted, newest first."""
    return _read(connection, "where s.author = %s", (author,), "s.created desc")


def every(connection: Any) -> list[Suggestion]:
    """Every suggestion, for the maintainer's page: open ones first, newest first.

    Open before done, because the page is a box somebody works through and a done row is
    a record rather than a thing to do. Within each half the newest is on top.
    """
    return _read(connection, "", (), "(s.done is not null), s.created desc")


def mark(connection: Any, suggestion: int, done: bool = True) -> Suggestion:
    """Mark one suggestion done, or put it back. Reversible on purpose.

    A mistaken press on the wrong row would otherwise be a lost idea, and nothing about
    the mark is destructive: the row and its text are untouched either way.
    """
    stamp = "now()" if done else "null"
    with connection.cursor() as cursor:
        cursor.execute(
            f"update suggestions set done = {stamp} where id = %s"
            f" returning id, author, kind, body, created, done",
            (suggestion,),
        )
        found = cursor.fetchone()
    if found is None:
        raise SuggestionError(f"no suggestion with the id {suggestion}")
    return _to_suggestion(found)


def drop(connection: Any, suggestion: int) -> Suggestion:
    """Delete one suggestion outright, text included. There is no way back.

    A real delete rather than a stamp, because nothing in the store references a
    suggestion: no match, rating, vote or badge hangs off it, so no count moves and no
    row is left pointing at something that is gone. That is the difference from a game,
    whose row has to stay (`0010_deleted.sql`).

    The row as it was comes back, handle and all, so the answer to the press can say what
    went. It is read before the delete rather than returned by it, because the handle is
    the users table's and a `delete ... returning` cannot join.
    """
    found = one(connection, suggestion)
    if found is None:
        raise SuggestionError(f"no suggestion with the id {suggestion}")
    with connection.cursor() as cursor:
        cursor.execute("delete from suggestions where id = %s", (suggestion,))
    return found


def one(connection: Any, suggestion: int) -> Suggestion | None:
    """One suggestion by id, or None. Used where an act has to name what it acted on."""
    found = _read(connection, "where s.id = %s", (suggestion,), "s.created desc")
    return found[0] if found else None


def _read(
    connection: Any, clause: str, values: Sequence[Any], order: str
) -> list[Suggestion]:
    """The shared query, with the author's handle joined on.

    A left join rather than an inner one: `author` is `not null`, so every row has a user
    to join to, and the join is left only so that a suggestion could never be hidden by a
    users row somebody removed by hand. A departed person's handle is null and reads as
    `identity.FORMER`, which is `unbeaten.handles`' rule and `admin.people`'s.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select s.id, s.author, s.kind, s.body, s.created, s.done,"
            f" coalesce(u.handle, %s)"
            f" from suggestions s left join users u on u.id = s.author"
            f" {clause} order by {order}",
            (identity.FORMER, *values),
        )
        return [_to_suggestion(row) for row in cursor.fetchall()]


def _to_suggestion(found: Sequence[Any]) -> Suggestion:
    return Suggestion(
        id=int(found[0]),
        author=int(found[1]),
        kind=str(found[2]),
        body=str(found[3]),
        created=found[4],
        done=found[5],
        handle=str(found[6]) if len(found) > 6 else "",
    )
