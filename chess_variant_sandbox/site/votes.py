"""Stars and fixed tags: what a player thinks of a game, in one row.

Task 1.5 of `docs/plan-website.md`, against §5's first crowdsourced quantity. The user,
2026-09-10: *"let players give a star rating / 5"*. So a vote is one to five stars, one
row per person per game, replaced when the person votes again, and the games page shows
the mean with its count beside it.

This is the cheap, biased half of §5's *games ranked by people* and it is kept apart from
the other two on purpose. Stars are an opinion, the behaviour column is what people
actually did, and the index is the measurement; merging any two of them would produce a
number that answers no question. The three orderings are offered switchably and never
summed.

**No text from the voter** (Q599, `docs/archive/plan-return.md` R4; the user, 2026-10-02: *"Votes
only, no comments"*). A vote is its stars and up to `MOST_TAGS` of the eight `TAGS`, chosen
by key. The key a voter sends is looked up and the store is handed this module's own
string, never theirs, and anything else in the request is refused by name: a note, an
unknown tag, a fourth tag. `0026_member_page.sql` holds the same rule in the table, so no
path can store a sentence a voter wrote. `votes.note` survives as a column that must be
empty; nothing reads it.

Nothing here commits. The caller owns the transaction, the same rule `matches` and
`rating` follow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

LOWEST = 1
HIGHEST = 5
"""The scale, out of five. Held here because the DDL's check constraint and the page's
control both have to mean the same range, and a constraint violation is not a sentence a
player can read."""


TAGS: dict[str, str] = {
    "again": "I'd play it again",
    "clever": "a clever rule",
    "first-easy": "first player has it easy",
    "draws": "lots of draws",
    "long": "goes on too long",
    "fast": "over too fast",
    "hard-to-learn": "hard to learn",
    "new": "feels new",
}
"""The eight tags a vote may carry, key to the words on the page, in the page's order:
`docs/archive/plan-return.md` §5.1's set as `docs/archive/plan-understanding.md` §6 words them.
The keys are the stored strings and `0026_member_page.sql` lists them too, so a new tag or a
renamed key is a migration; rewording one is not."""

MOST_TAGS = 3
"""How many tags one vote may carry (§5.1): a choice, not a checklist."""

ALLOWED_KEYS = frozenset({"hash", "stars", "tags"})
"""Everything `POST /api/vote` reads. Any other field is refused rather than ignored, so a
note sent by a script is told it cannot be kept instead of being dropped in silence."""


class VoteError(ValueError):
    """A vote cannot be recorded as asked.

    A `ValueError` like `matches.MatchError`, so the server answers 400 with the sentence
    rather than a constraint violation from Postgres.
    """


@dataclass(frozen=True)
class Vote:
    """One person's opinion of one game."""

    voter: int
    game: str
    stars: int
    tags: tuple[str, ...]
    updated: datetime | None = None


@dataclass(frozen=True)
class Stars:
    """A game's votes, as the card shows them.

    `mean` is None where nobody has voted, never zero: an unvoted game is not a game
    everybody hated, and a sort that treated it as one would put every new game last.
    """

    mean: float | None
    count: int

    @property
    def voted(self) -> bool:
        return self.count > 0


NONE = Stars(None, 0)


def cast(
    connection: Any,
    voter: int,
    game: str,
    stars: int,
    tags: Sequence[str] = (),
) -> Vote:
    """Record one person's vote on one game, replacing whatever they said before.

    The primary key is (voter, game), so changing a vote is an upsert and there is no
    second row to reconcile. A person can vote as often as they like and the table only
    ever holds their latest answer, tags included: a vote sent without tags has none.
    """
    if int(stars) < LOWEST or int(stars) > HIGHEST:
        raise VoteError(f"a vote is {LOWEST} to {HIGHEST} stars, not {stars}")
    chosen = fixed_tags(tags)
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into votes (voter, game, stars, tags, note)"
            " values (%s, %s, %s, %s, null)"
            " on conflict (voter, game) do update set"
            " stars = excluded.stars, tags = excluded.tags, note = null,"
            " updated = now()"
            " returning voter, game, stars, tags, updated",
            (voter, game, int(stars), list(chosen)),
        )
        return _to_vote(cursor.fetchone())


def fixed_tags(asked: Sequence[str]) -> tuple[str, ...]:
    """The tags a vote carries, as this module's own strings, in `TAGS`' order.

    Each asked key is matched against `TAGS` and the key stored is the one from the
    table, so what reaches the row was written here. An unknown key or more than
    `MOST_TAGS` is refused with a sentence rather than trimmed, because a tag silently
    dropped is a vote recorded other than as cast.
    """
    if isinstance(asked, str):
        raise VoteError("tags are a list of the fixed choices")
    wanted = {str(tag) for tag in asked}
    unknown = sorted(wanted - set(TAGS))
    if unknown:
        raise VoteError(
            "a vote's tags are fixed choices; there is no tag called "
            + ", ".join(repr(tag[:40]) for tag in unknown[:3])
        )
    chosen = tuple(key for key in TAGS if key in wanted)
    if len(chosen) > MOST_TAGS:
        raise VoteError(f"a vote carries at most {MOST_TAGS} tags, not {len(chosen)}")
    return chosen


def checked_body(body: Mapping[str, Any]) -> None:
    """Refuse a vote request carrying anything but the game, the stars and the tags.

    A note, a message, a comment: none is kept, so none is accepted (Q599).
    """
    extra = sorted(set(body) - ALLOWED_KEYS)
    if extra:
        raise VoteError(
            "a vote is stars and fixed tags only, with no text; this one also sent "
            + ", ".join(repr(str(key)[:40]) for key in extra[:3])
        )


def tag_counts(connection: Any, game: str) -> list[dict[str, Any]]:
    """How many votes on this game carry each fixed tag, in `TAGS`' order.

    Only tags somebody chose, so an untagged game reads as nothing rather than as eight
    noughts; and only the eight, so a string stored before `0026_member_page.sql` is never
    shown. A count of a quality and never who chose it.
    """
    return tag_counts_by_game(connection, (game,)).get(game, [])


def tag_counts_by_game(
    connection: Any, games: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    """`tag_counts` for several games in one query; a game nobody tagged is absent."""
    wanted = sorted(set(games))
    if not wanted:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, tag, count(*) from votes, unnest(tags) as tag"
            " where game = any(%s) and tag = any(%s) group by game, tag",
            (wanted, list(TAGS)),
        )
        found: dict[str, dict[str, int]] = {}
        for game, tag, count in cursor.fetchall():
            found.setdefault(str(game), {})[str(tag)] = int(count)
    return {
        game: [
            {"tag": key, "words": TAGS[key], "count": counted[key]}
            for key in TAGS if key in counted
        ]
        for game, counted in found.items()
    }


def withdraw(connection: Any, voter: int, game: str) -> bool:
    """Take a vote back. True when there was one to take back."""
    with connection.cursor() as cursor:
        cursor.execute(
            "delete from votes where voter = %s and game = %s", (voter, game)
        )
        return bool(cursor.rowcount)


def of(connection: Any, voter: int, game: str) -> Vote | None:
    """What this person said about this game, or None."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select voter, game, stars, tags, updated"
            " from votes where voter = %s and game = %s",
            (voter, game),
        )
        found = cursor.fetchone()
    return None if found is None else _to_vote(found)


def stars(connection: Any, game: str) -> Stars:
    """One game's mean and count."""
    return by_game(connection, (game,)).get(game, NONE)


def by_game(connection: Any, games: Iterable[str] = ()) -> dict[str, Stars]:
    """Every game's mean and count, or just the ones asked for.

    One query for the whole listing rather than one per card: the games page draws sixty
    cards and sixty round trips to show a star count is sixty round trips too many.
    """
    wanted = tuple(games)
    clause = " where game = any (%s)" if wanted else ""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select game, avg(stars)::double precision, count(*) from votes{clause}"
            f" group by game",
            (list(wanted),) if wanted else (),
        )
        rows = cursor.fetchall()
    return {str(game): Stars(float(mean), int(count)) for game, mean, count in rows}


def _to_vote(found: Sequence[Any]) -> Vote:
    return Vote(
        voter=int(found[0]),
        game=str(found[1]),
        stars=int(found[2]),
        tags=tuple(tag for tag in (found[3] or ()) if tag in TAGS),
        updated=found[4],
    )
