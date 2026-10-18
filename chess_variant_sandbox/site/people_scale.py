"""Each computer player's place on people's scale, per game (route B, H7).

`docs/reference/human-data.md` §3.2. A game's rating table holds people and computer players
together, and people's rated results move the computer players' rows (`site/rating.py`).
Reading the table therefore says where each computer player sits among the *people* who
play the game: *the strongest computer player plays here about as a member rated 1650
would*. Nothing read that link before.

What one place says, for one computer player on one game:

- its rating and deviation on the table now, which is the people's scale;
- where it was seeded from the computer-only picker ladder (`rating.seeds_for`), so how far
  people's results have moved it is visible beside the number;
- how many rated games people have played against it;
- how many of the game's **established** people (deviation under
  `rating.ESTABLISHED_DEVIATION`, the table's own word) are rated below it.

**Where it is shown.** On `/admin` for every game whose table holds a person. On the game's
page, inside the opponent line's *more*, only once **at least five people are established
on the game's table** -- the minimum the plan pre-registered on 2026-10-08, before any table
was read -- and only for the computer players people have played. Below that the page says
nothing, since a scale held by fewer than five people says more about them than about the
player.

Rule 5: shown beside the score, labelled as people's, and read by no score's component or
anchor. This is under `site/`, and nothing under `metrics/`, `agents/` or `tournament/`
reaches it (`tests/test_rule_five_reach.py`). Its one limit, the plan's: people play the
site's opponents, not the measuring players at their node budgets, so this places the
site's players, and the link to a score's players runs through each one's ladder rating.

Nothing here writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import rating, unbeaten

# Pre-registered 2026-10-08 (`docs/reference/human-data.md` §3.2); never re-aimed.
MINIMUM_ESTABLISHED = 5

SCALE_SAID = (
    "Where each computer player sits among the people rated on this game, from their "
    "rated games against it. It is people's scale and moves no score."
)


@dataclass(frozen=True)
class Place:
    """One computer player on one game's table, read against its people."""

    agent: str
    name: str
    rating: float
    deviation: float
    seed: float | None
    games: int
    below: int
    established: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "name": self.name,
            "rating": self.rating,
            "deviation": self.deviation,
            "seed": self.seed,
            "games": self.games,
            "below": self.below,
            "established": self.established,
        }


def established(table: Mapping[rating.PlayerKey, rating.Player]) -> tuple[rating.Player, ...]:
    """The people on the table whose rating is established."""
    return tuple(
        player for player in table.values() if not player.is_computer and player.established
    )


def places_from(
    game: str,
    rows: Sequence[match_rows.Row],
    table: Mapping[rating.PlayerKey, rating.Player],
    seeds: Mapping[rating.PlayerKey, rating.Player] | None = None,
) -> tuple[Place, ...]:
    """Every computer player on the table, strongest first. Pure."""
    from chess_variant_sandbox.web.opponents import table_name

    games: dict[str, int] = {}
    for row in rows:
        if row.game != game or not unbeaten._rated_row(row):
            continue
        seat = unbeaten._human_seat(row)
        person = row.seat0 if seat == 0 else row.seat1
        if person is None or person.user is None:
            continue
        agent = str(row.player(1 - seat).agent)
        games[agent] = games.get(agent, 0) + 1
    people = established(table)
    found = seeds or {}
    return tuple(
        Place(
            agent=str(player.key.agent),
            name=table_name(str(player.key.agent)),
            rating=player.rating,
            deviation=player.deviation,
            seed=found[player.key].rating if player.key in found else None,
            games=games.get(str(player.key.agent), 0),
            below=sum(1 for one in people if one.rating < player.rating),
            established=len(people),
        )
        for player in rating.computers(table)
    )


def places(connection: Any, game: str) -> tuple[Place, ...]:
    """One game's places, read from the store."""
    return places_from(
        game,
        match_rows.leaderboard_matches(connection, game),
        rating.load(connection, game),
        rating.seeds_for(connection, game),
    )


def place_said(place: Place) -> str:
    """One place in a sentence a visitor reads."""
    return (
        f"{place.name.capitalize()} plays about as a person rated {place.rating:.0f} here "
        f"would: above {place.below} of the {place.established} people with an established "
        f"rating, after {place.games} rated game{'s' if place.games != 1 else ''} against "
        "people."
    )


def shown(found: Sequence[Place]) -> dict[str, Any]:
    """What the game page's payload carries: the sentences, once the minimum holds.

    `said` is empty below `MINIMUM_ESTABLISHED`, and leaves out a computer player people
    have not played, whose rating is still the ladder's and says nothing of people.
    """
    count = found[0].established if found else 0
    played = [place for place in found if place.games]
    if count < MINIMUM_ESTABLISHED or not played:
        return {"established": count, "minimum": MINIMUM_ESTABLISHED, "said": []}
    return {
        "established": count,
        "minimum": MINIMUM_ESTABLISHED,
        "said": [SCALE_SAID, *(place_said(place) for place in played)],
    }


def admin_rows(connection: Any) -> list[dict[str, Any]]:
    """`/admin`'s *Computer players on people's scale*: every game whose table holds a person.

    Each game with its name, how many people are established there, whether the page shows
    the line yet, and every computer player's place.
    """
    tables = rating.load_many(connection)
    peopled = sorted(
        game for game, table in tables.items()
        if any(not player.is_computer for player in table.values())
        and rating.computers(table)
    )
    if not peopled:
        return []
    with connection.cursor() as cursor:
        cursor.execute("select hash, name from games where hash = any(%s)", (peopled,))
        names = {str(found[0]): str(found[1]) for found in cursor.fetchall()}
    rows = match_rows.leaderboard_matches(connection)
    out = []
    for game in peopled:
        found = places_from(game, rows, tables[game], rating.seeds_for(connection, game))
        count = len(established(tables[game]))
        out.append({
            "hash": game,
            "name": names.get(game, game[:12]),
            "established": count,
            "shown": count >= MINIMUM_ESTABLISHED,
            "places": [place.as_dict() for place in found],
        })
    return out
