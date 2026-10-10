"""The bar: games whose strongest computer player people beat often (route C, H8).

`docs/plan-human-data.md` §3.3 fixed the bar on 2026-10-08, before any person's game was
read: **on the game's rated games against the top group, at least 3 distinct people have
beaten it, and people score at least 0.30 against it over at least 10 games.** It is
stricter than the trust line's *questioned* because it spends a training run, and it never
loosens after a quiet month. This module reads it and nothing else; `/admin` shows every
game with a game against the top beside the bar (H8), and `site/training.py` queues a game
over it for a stronger computer player (H9, the user's answer B of 2026-10-08).

**Which games count.** The rows `unbeaten` reads (`unbeaten._rated_row`: finished, rated, a
person against a computer player, not loaded from the local application) whose opponent
was in the top group when the match started (`Row.top`) **and is in the top group now**.
The second condition is H4's: the top can change hands when a computer player is rated
later, and a game against last month's strongest is not a game against this month's, so it
leaves the count rather than being read as one. *Beaten* is a win against a member of that
group; a draw is half a point in the score and beats nothing.

**Which player, exactly.** The agent string does not pin the player (`site/pinning.py`),
so each reading also says how many of its games were played by today's player -- the same
build, tables and values -- by an earlier one, or before the row recorded it. That split is
reported beside the bar and does not move it: the bar's rows were fixed in the plan by the
top group, and a reading that dropped every game played before the last core change would
be a different bar, chosen after the fact.

Rule 5: this reads the site's rows and moves no score, fingerprint or trust level. Nothing
under `metrics/`, `agents/` or `tournament/` imports it (`tests/test_rule_five_reach.py`).
Nothing here writes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import pinning, rating, unbeaten

# Pre-registered 2026-10-08 (`docs/plan-human-data.md` §3.3); never re-aimed.
BAR_PEOPLE = 3
BAR_SCORE = 0.30
BAR_GAMES = 10

BAR_SAID = (
    f"Over the bar when at least {BAR_PEOPLE} different people have beaten the strongest "
    f"computer player, and people score at least {BAR_SCORE:.2f} against it over at least "
    f"{BAR_GAMES} rated games."
)


@dataclass(frozen=True)
class Reading:
    """One game against the bar."""

    game: str
    top: tuple[str, ...]
    games: int
    people: int
    beaten_by: int
    points: float
    latest: datetime | None
    # H4's split of `games`; None where the reading was not asked for it.
    same: int | None = None
    earlier: int | None = None
    unrecorded: int | None = None

    @property
    def score(self) -> float | None:
        """People's mean score against the top group, or None with no games."""
        return self.points / self.games if self.games else None

    @property
    def over(self) -> bool:
        score = self.score
        return (
            self.beaten_by >= BAR_PEOPLE
            and self.games >= BAR_GAMES
            and score is not None
            and score >= BAR_SCORE
        )

    def evidence(self) -> dict[str, Any]:
        """What a queue entry keeps of the reading it was raised on."""
        return {
            "top": list(self.top),
            "games": self.games,
            "people": self.people,
            "beatenBy": self.beaten_by,
            "score": self.score,
            "latest": None if self.latest is None else self.latest.isoformat(),
        }


def counted(row: match_rows.Row, group: frozenset[str]) -> bool:
    """Whether this row is one of the bar's games against the top group `group`."""
    if not unbeaten._rated_row(row) or not row.top:
        return False
    seat = unbeaten._human_seat(row)
    person = row.seat0 if seat == 0 else row.seat1
    opponent = row.seat1 if seat == 0 else row.seat0
    return (
        person is not None
        and person.user is not None
        and opponent is not None
        and opponent.agent in group
    )


def reading_from(
    game: str,
    rows: Sequence[match_rows.Row],
    table: Mapping[rating.PlayerKey, rating.Player],
    pinned: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    now: Mapping[str, Mapping[str, str]] | None = None,
) -> Reading:
    """One game's reading over its rows. Pure, so the bar is checked without a database.

    `pinned` is each row's `matches.pinned` by id and `now` today's identity for each top
    agent (`pinning.identity`); with both, the reading carries H4's split.
    """
    top = tuple(str(player.key.agent) for player in rating.top(table))
    group = frozenset(top)
    games = 0
    points = 0.0
    people: set[int] = set()
    beaten: set[int] = set()
    latest: datetime | None = None
    split = {pinning.SAME: 0, pinning.EARLIER: 0, pinning.UNRECORDED: 0}
    for row in rows:
        if row.game != game or not counted(row, group):
            continue
        seat = unbeaten._human_seat(row)
        user = row.player(seat).user
        if user is None:  # `counted` already refused it; for the type checker
            continue
        score = row.score_for(seat) or 0.0
        games += 1
        points += score
        people.add(user)
        if score == 1.0:
            beaten.add(user)
        if row.finished is not None and (latest is None or row.finished > latest):
            latest = row.finished
        if pinned is not None and now is not None:
            agent = str(row.player(1 - seat).agent)
            split[pinning.compared(pinned.get(row.id), now.get(agent, {}))] += 1
    asked = pinned is not None and now is not None
    return Reading(
        game=game,
        top=top,
        games=games,
        people=len(people),
        beaten_by=len(beaten),
        points=points,
        latest=latest,
        same=split[pinning.SAME] if asked else None,
        earlier=split[pinning.EARLIER] if asked else None,
        unrecorded=split[pinning.UNRECORDED] if asked else None,
    )


def read(connection: Any, game: str) -> Reading:
    """One game's reading from the store, without H4's split: what a finish asks."""
    return reading_from(
        game, match_rows.leaderboard_matches(connection, game), rating.load(connection, game)
    )


def _specs(connection: Any, games: Iterable[str]) -> dict[str, dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, spec from games where hash = any(%s) and spec is not null",
            (list(games),),
        )
        return {str(found[0]): dict(found[1]) for found in cursor.fetchall()}


def readings(connection: Any) -> list[Reading]:
    """Every game with at least one game against its top group, with H4's split.

    Over the bar first, then by games played; today's identity is worked out only for the
    games that have such a game, since it compiles the game once per process.
    """
    tables = rating.load_many(connection)
    rows = match_rows.leaderboard_matches(connection)
    by_game: dict[str, list[match_rows.Row]] = {}
    for found in rows:
        by_game.setdefault(found.game, []).append(found)
    plain = {
        game: reading_from(game, by_game.get(game, ()), table)
        for game, table in tables.items()
    }
    live = [game for game, found in plain.items() if found.games]
    specs = _specs(connection, live)
    wanted = [
        found.id
        for game in live
        for found in by_game[game]
        if counted(found, frozenset(plain[game].top))
    ]
    pinned = pinning.recorded(connection, wanted)
    out: list[Reading] = []
    for game in live:
        spec = specs.get(game)
        now = (
            {agent: pinning.identity(spec, (agent,)) for agent in plain[game].top}
            if spec is not None
            else {}
        )
        out.append(reading_from(game, by_game[game], tables[game], pinned, now))
    return sorted(out, key=lambda found: (not found.over, -found.games, found.game))


def said(found: Reading) -> str:
    """The reading in one sentence, for `/admin`."""
    score = found.score
    text = (
        f"{found.games} rated game(s) against the strongest by {found.people} "
        f"people; {found.beaten_by} have beaten it; people scored "
        f"{0.0 if score is None else score:.2f}."
    )
    if found.same is not None:
        text += (
            f" Of those games, {found.same} were played by today's player, "
            f"{found.earlier} by an earlier version of it and {found.unrecorded} before the "
            "site recorded which."
        )
    return text
