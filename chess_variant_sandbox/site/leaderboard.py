"""A game's leaderboard: its rating table as the game page draws it (L2).

`docs/plan-leaderboard.md` §2.1, §2.5 and §3, with Q2 decided on 2026-10-10 (*A: everyone
who has finished a rated game on it, a settling number marked `?`*). The store already is the
design: one Glicko-2 table per game, people and computer players as rows of it
(`site/rating.py`). This reads that table and says who is on it, in what order and with what
marks, so the page draws rows and sorts, rounds and decides nothing.

Who is a row:

- **every computer player the table holds**, by the name the picker gives it (*random*,
  *easy*, *hardest*, a rule of thumb's name), its agent string staying the identity, so two
  budgets are two rows. A player the picker no longer offers keeps its row, marked, because
  people's games against it are in the table's history; its name is then
  `opponents.table_name`'s, never its agent string;
- **every person with at least one finished rated game on this game**, by handle. A person
  row with none (a seat the table holds and nobody has finished a game from) is not a row,
  so nobody is ever shown last at 1500 for having played nothing;
- **never a practice-only player** (the learned player, Q683): no game against it is rated,
  so it has no place on a table of ratings, whether or not the table holds a row for it.

Columns: the name, the rating rounded, a `?` where a person's deviation is still at or over
`rating.ESTABLISHED_DEVIATION` (the table's own *established* line, about ten to fifteen
games), and *rated games here*, the row's `matches` (for a computer player, its games against
people; the ladder's own games are said once, in the sentence above the table). The order is
strongest first, ties by name, with no places. The strongest computer players
(`rating.top`, every one within `TOP_BAND`) are marked as the one *Unbeaten* is about, and
the reader's own row is marked.

Before the computer players are rated (§2.5), there are no rows: `waiting` names the
players the picker offers, in the picker's order, which puts the standard players weakest
first and the rest after them, and the page says the game is practice until they are rated.
A member's game is rated on the site itself when it is published, the five standard players
first (L5, `site/spine.py`), and `box` says how far that has got.

Nothing here writes, and nothing under `metrics/` reads it (rule 5).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from chess_variant_sandbox.site import identity, rating
from chess_variant_sandbox.web import opponents


def _practice_only(roster: Iterable[opponents.Opponent]) -> frozenset[str]:
    """The roster's practice-only agents. `_counted` also asks `opponents.never_rated`,
    which reads the agent string, so a practice-only row is left out offered or not."""
    return frozenset(entry.agent for entry in roster if entry.practice_only)


def _counted(player: rating.Player, practice: frozenset[str]) -> bool:
    """Whether this row of the table is a row of the leaderboard."""
    if player.is_computer:
        agent = player.key.agent or ""
        return agent not in practice and not opponents.never_rated(agent)
    return player.matches >= 1


def rows(
    table: Mapping[rating.PlayerKey, rating.Player],
    *,
    named: Mapping[int, str],
    roster: Sequence[opponents.Opponent],
    viewer: int | None,
) -> list[dict[str, Any]]:
    """The leaderboard's rows, strongest first, each as the page draws it."""
    practice = _practice_only(roster)
    kept = {key: player for key, player in table.items() if _counted(player, practice)}
    offered = {entry.agent: entry.name for entry in roster if entry.agent != opponents.HUMAN}
    strongest = {player.key for player in rating.top(kept)}
    you = rating.person(viewer) if viewer is not None else None

    def name(player: rating.Player) -> str:
        if player.is_computer:
            agent = player.key.agent or ""
            return offered.get(agent) or opponents.table_name(agent)
        return named.get(int(player.key.user or 0), identity.FORMER)

    ordered = sorted(kept.values(), key=lambda player: (-player.rating, name(player)))
    return [
        {
            "name": name(player),
            "computer": player.is_computer,
            "rating": round(player.rating),
            "settling": not player.is_computer and not player.established,
            "games": player.matches,
            "top": player.key in strongest,
            "gone": player.is_computer and (player.key.agent or "") not in offered,
            "you": player.key == you,
        }
        for player in ordered
    ]


def shown(
    table: Mapping[rating.PlayerKey, rating.Player],
    *,
    named: Mapping[int, str],
    roster: Sequence[opponents.Opponent],
    viewer: int | None,
    library: bool,
    box: str | None = None,
) -> dict[str, Any]:
    """`standing.leaderboard`: the rows, and what the page says around them.

    `rated` is whether any computer player has a row, which is the game offering rated play
    at all. `people` counts the person rows, so the page knows whether to say that nobody
    has played a rated game here yet. `unrated` names the offered players with no row (a
    rule of thumb proposed since the ladder was rated): absent from the table, said in one
    sentence, never given an invented number. `waiting` is §2.5's list for a game whose
    computer players are not rated yet. `library` says whose machine rates them and when:
    a library game's ladder is rated on our own machine within about a day.

    `box` is a member's game's own rating on the site (L5, `spine.state`; `None` for a
    library game), so the page can say where its computer players are: *draft* (rated when
    it is published), *rating* (the five standard ones, now, in a few minutes), *rated*
    (they were, here, and the rest -- `unrated` -- join when our own machine has rated them),
    or *slow*, *large* and *waiting* (the site's caps refused them, or the rating failed, so
    they wait for our own machine). The page words them; this decides nothing else.
    """
    drawn = rows(table, named=named, roster=roster, viewer=viewer)
    rated = any(row["computer"] for row in drawn)
    seated = {player.key.agent for player in table.values() if player.is_computer}
    rateable = set(opponents.agents_of(list(roster)))
    playable = [entry for entry in roster if entry.agent in rateable]
    return {
        "rated": rated,
        "rows": drawn,
        "people": sum(1 for row in drawn if not row["computer"]),
        "unrated": (
            [entry.name for entry in playable if entry.agent not in seated] if rated else []
        ),
        "waiting": [] if rated else [entry.name for entry in playable],
        "library": library,
        "box": box,
    }
