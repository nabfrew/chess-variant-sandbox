"""*Lately* on your own page (UP3, `docs/plan-user-page.md` §5, choice C1 of §4).

What happened on your games and in your games, dated, newest first, the last `LIMIT` at
most, each line a sentence in the site's words. One reader over five stores:

- **your games between people**: the last move in each one still running (whose move it
  is now is the engine's answer, `correspondence.owing`, read where the profile already
  asked it), a game that has not had a move yet as begun, and every finished game between
  people or rated game against a computer player, with its result;
- **challenges**: each one sent to you by name, and each of yours somebody answered;
- **milestones and badges**: the lines *What you have done* lists with their rules, said
  here once each without them (but D3, *played by somebody else*, which is the first of the
  first plays below);
- **your games, played by others**: each vote on a game you made, and each person's first
  game of it. Neither names anybody: a vote is private (only the count and the average are
  shown anywhere) and *played by somebody else* never says who (the user's decision B of
  2026-09-14), so both lines say *somebody*.

**Nothing unread, nothing counted, nothing on the header** (choice C1). The site keeps no
last visit for a person and nothing here reads one: the list is the same whenever it is
read, and no line says how many of anything there are.

**No line about another person's private game.** A line naming a game is kept only when
the game is published (any level but Draft) or is one you made, so a practice game on
somebody's draft link, a challenge to it or a badge on it never names their draft here.
Games between other people are never read.

Nothing here commits; it only reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import (
    awards,
    badges,
    browse,
    challenges,
    designs,
    identity,
    members,
    milestones,
    stages,
)

LIMIT = 20
"""The most lines *Lately* holds (§3.3's bound: the last 20)."""

MOVED = "moved"
BEGAN = "began"
FINISHED = "finished"
CHALLENGED = "challenged"
ANSWERED = "answered"
MILESTONE = "milestone"
BADGE = "badge"
VOTED = "voted"
FIRST_PLAY = "first-play"
KINDS = (MOVED, BEGAN, FINISHED, CHALLENGED, ANSWERED, MILESTONE, BADGE, VOTED, FIRST_PLAY)
"""Every kind of line, in the order two lines of one moment are read in."""

SOMEBODY = "Somebody"
"""Who did a thing this page may not name: a voter, a first player, a link's taker."""

Part = str | dict[str, str]
"""One piece of a sentence: words, a game (`game`, `name`) or a person (`person`)."""


def _person(handle: str) -> dict[str, str]:
    return {"person": handle}


def _line(when: datetime, kind: str, game: str | None, *parts: Part) -> dict[str, Any]:
    return {"when": when, "kind": kind, "game": game, "parts": list(parts)}


def _stamp(said: Any) -> datetime | None:
    if isinstance(said, datetime):
        return said
    try:
        return datetime.fromisoformat(str(said))
    except ValueError:
        return None


def _shown(connection: Any, user: int, games: Iterable[str]) -> set[str]:
    """Of these games, the ones a line may name: published, or this person's own."""
    rows = stages.read(connection, games)
    return {
        digest for digest, row in rows.items()
        if row["level"] != stages.DRAFT or row.get("owner") == user
    }


def _owned(connection: Any, user: int) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash from games where origin = %s and owner = %s order by hash",
            (designs.DESIGNED, user),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _running(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The last move in each game between people still running, or the game begun."""
    found = []
    for entry in entries:
        when = _stamp(entry.get("moved"))
        if when is None:
            continue
        game = str(entry["game"])
        other = _person(str(entry.get("opponent") or identity.FORMER))
        named = {"game": game, "name": str(entry.get("name") or "")}
        if not entry.get("plies"):
            found.append(_line(when, BEGAN, game, "Your game of ", named, " against ", other,
                               " began."))
            continue
        found.append(_line(
            when, MOVED, game, "A move in ", named, " against ", other,
            "; it is your move." if entry.get("yourMove") else "; it is their move.",
        ))
    return found


def _finished(connection: Any, user: int, people: Mapping[int, str]) -> list[dict[str, Any]]:
    """Finished games between people and rated games against a computer player, each
    with its result, the newest `LIMIT`."""
    found = []
    for row in members._finished(connection, user)[:LIMIT]:
        if row.finished is None:
            continue
        against = members._against(row, user, people)
        who: Part = (
            _person(str(against["person"])) if "person" in against
            else str(against.get("computer", "the computer"))
        )
        result = members._result(row, user)
        game = {"game": row.game}
        if result == "won":
            parts: tuple[Part, ...] = ("You beat ", who, " at ", game, ".")
        elif result == "lost":
            parts = (who, " beat you at ", game, ".")
        else:
            parts = ("You drew with ", who, " at ", game, ".")
        found.append(_line(row.finished, FINISHED, row.game, *parts))
    return found


def _challenges(connection: Any, user: int, people: Mapping[int, str]) -> list[dict[str, Any]]:
    """Each challenge sent to this person by name, and each of theirs answered."""
    found = []
    with connection.cursor() as cursor:
        cursor.execute(
            "select challenger, game, made from challenges"
            " where challenged = %s and token is null order by made desc, id desc limit %s",
            (user, LIMIT),
        )
        for challenger, game, made in cursor.fetchall():
            sender = people.get(int(challenger), identity.FORMER)
            found.append(_line(made, CHALLENGED, str(game), _person(sender),
                               " challenged you to ", {"game": str(game)}, "."))
        cursor.execute(
            "select c.challenged, c.game, c.answer, c.answered, m.seat0_user, m.seat1_user"
            " from challenges c left join matches m on m.id = c.match"
            " where c.challenger = %s and c.answer = any(%s)"
            " order by c.answered desc, c.id desc limit %s",
            (user, [challenges.ACCEPTED, challenges.DECLINED], LIMIT),
        )
        for named, game, answer, answered, first, second in cursor.fetchall():
            taker = named if named is not None else next(
                (seat for seat in (first, second) if seat is not None and seat != user), None
            )
            who: Part = SOMEBODY if taker is None else _person(
                people.get(int(taker), identity.FORMER)
            )
            found.append(_line(answered, ANSWERED, str(game), who, f" {answer} your challenge to ",
                               {"game": str(game)}, "."))
    return found


def _records(
    lines: Iterable[milestones.Milestone], held: Iterable[awards.Award], people: Mapping[int, str]
) -> list[dict[str, Any]]:
    """The milestones and badges, a line each, without the rules *What you have done*
    already says under them; *played by somebody else* (D3) is said by the first plays."""
    found = []
    for line in lines:
        # D3 is the first of the first plays below, which say it game by game.
        if line.kind == milestones.PLAYED_BY_OTHERS:
            continue
        heading, join = milestones.HEADINGS.get(line.kind, (line.kind, "on"))
        tail: tuple[Part, ...] = (
            (", by ", _person(people.get(line.by, identity.FORMER)), ".")
            if line.by is not None else (".",)
        )
        found.append(_line(line.when, MILESTONE, line.game, heading, f", {join} ",
                           {"game": line.game}, *tail))
    for award in held:
        if award.earned is None:
            continue
        name = badges.name_for(award)
        if award.game:
            found.append(_line(award.earned, BADGE, award.game, f"Badge: {name}, on ",
                               {"game": award.game}, "."))
        else:
            found.append(_line(award.earned, BADGE, None, f"Badge: {name}."))
    return found


def _played_by_others(connection: Any, user: int) -> list[dict[str, Any]]:
    """Each vote on a game this person made, and each other person's first game of one,
    naming nobody."""
    own = _owned(connection, user)
    if not own:
        return []
    found = []
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, updated from votes where game = any(%s) and voter <> %s"
            " order by updated desc limit %s",
            (own, user, LIMIT),
        )
        for game, updated in cursor.fetchall():
            found.append(_line(updated, VOTED, str(game), f"{SOMEBODY} voted on ",
                               {"game": str(game)}, "."))
    first: dict[tuple[str, int], datetime] = {}
    for played in designs.tries(connection, own):
        if played.user != user:
            first.setdefault((played.game, played.user), played.started)
    for (game, _), started in first.items():
        found.append(_line(started, FIRST_PLAY, game, f"{SOMEBODY} played ", {"game": game},
                           " for the first time."))
    return found


def read(
    connection: Any,
    user: int,
    *,
    running: Sequence[Mapping[str, Any]] = (),
    lines: Iterable[milestones.Milestone] = (),
    held: Iterable[awards.Award] = (),
) -> list[dict[str, Any]]:
    """*Lately*: this person's last `LIMIT` lines, newest first, each `{when, kind, parts}`.

    `running` is the profile's own entries for the games between people still running
    (`server._between_entry`: `game`, `name`, `opponent`, `yourMove`, `plies`, `moved`),
    so whose move it is is asked of the engine once. `lines` and `held` are the
    milestones and badges the profile already read. `parts` is the sentence in order:
    words, `{"game", "name"}` or `{"person"}`; a person who left is `identity.FORMER`.
    """
    people = members.with_pages(connection)
    found = [
        *_running(running),
        *_finished(connection, user, people),
        *_challenges(connection, user, people),
        *_records(lines, held, people),
        *_played_by_others(connection, user),
    ]
    named = {line["game"] for line in found if line["game"]}
    shown = _shown(connection, user, named)
    kept = [line for line in found if not line["game"] or line["game"] in shown]
    order = {kind: index for index, kind in enumerate(KINDS)}
    kept.sort(key=lambda line: (-line["when"].timestamp(), order[line["kind"]]))
    kept = kept[:LIMIT]
    names = browse.names(connection, sorted({line["game"] for line in kept if line["game"]}))
    return [
        {
            "when": line["when"].isoformat(),
            "kind": line["kind"],
            "parts": [
                {**part, "name": names.get(part["game"], part.get("name", ""))}
                if isinstance(part, dict) and "game" in part else part
                for part in line["parts"]
            ],
        }
        for line in kept
    ]
