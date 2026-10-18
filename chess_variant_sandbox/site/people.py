"""People's counts on a game: the E-H evidence `trust.read` takes as `people`.

Task T3 of `docs/archive/plan-confidence.md` (§1, kinds E-H; §2.3, the rule). One fold over the
rows `matches.leaderboard_matches` returns, per game, says four things:

- **games**: finished leaderboard games a person played against the top group -- the
  match's own opponent was in it when the match started (`Row.top`, written once at the
  start, so a later change at the top does not move an old game in or out);
- **people**: distinct people who played those games;
- **cleared**: distinct people who cleared the top, read off `unbeaten.tag_from`'s own
  fold rather than counted again, so the trust line and the *Unbeaten* tag cannot disagree
  about who has beaten a game;
- **latest**: the UTC date of the newest of those games, ISO, or None when there are none.

The rows read are the ones the tag reads (`unbeaten._rated_row`): finished, a person
against a computer player, not loaded from the local application. Sandbox and cancelled
games are not in `leaderboard_matches` at all.

The direction is one way: this module imports `trust`, and nothing `trust` or `metrics/`
imports reaches the site (`tests/test_site_correspondence.py::TestTheRules`). People's
counts are evidence beside a score, never in it (rule 5); whether they move the level is
`trust.RULE.people_move_level`, True since the user answered plan §5 Q3 A on 2026-10-03.
The site's library cards and game pages fold them in (`SiteApp._library`, `_trust_line`).

Nothing here commits and nothing here writes.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chess_variant_sandbox import trust
from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import unbeaten


def counts_from(game: str, rows: Sequence[match_rows.Row]) -> trust.People:
    """One game's people's counts, folded over its rows in finish order.

    Pure, so the counts can be checked without a database. Rows of other games are
    skipped, as `unbeaten.tag_from` skips them.
    """
    games = 0
    people: set[int] = set()
    latest: datetime.datetime | None = None
    for row in rows:
        if row.game != game or not unbeaten._rated_row(row) or not row.top:
            continue
        seat = unbeaten._human_seat(row)
        key = row.seat0 if seat == 0 else row.seat1
        if key is None or key.user is None:
            continue
        games += 1
        people.add(key.user)
        if row.finished is not None and (latest is None or row.finished > latest):
            latest = row.finished
    # The tag's own fold; its rating table only names the target, so none is needed.
    cleared = unbeaten.tag_from(game, rows, None).clears
    return trust.People(
        games=games,
        people=len(people),
        cleared=cleared,
        latest=_day(latest),
    )


def _day(when: datetime.datetime | None) -> str | None:
    if when is None:
        return None
    if when.tzinfo is not None:
        when = when.astimezone(datetime.timezone.utc)
    return when.date().isoformat()


def counts_many(
    games: Iterable[str], rows: Sequence[match_rows.Row]
) -> dict[str, trust.People]:
    """Every named game's counts from one pass of rows."""
    by_game: dict[str, list[match_rows.Row]] = {}
    for row in rows:
        by_game.setdefault(row.game, []).append(row)
    return {game: counts_from(game, by_game.get(game, ())) for game in games}


def counts(connection: Any, game: str) -> trust.People:
    """One game's people's counts, read from the store."""
    return counts_from(game, match_rows.leaderboard_matches(connection, game))


def counts_of(connection: Any, games: Iterable[str]) -> dict[str, trust.People]:
    """Several games' counts in one query rather than one a game."""
    return counts_many(games, match_rows.leaderboard_matches(connection))


def read(
    connection: Any,
    digest: str,
    runs_dir: Path,
    spec: dict[str, Any] | None,
    *,
    nets_dir: str | Path | None = None,
    rule: trust.Rule = trust.RULE,
    people: Mapping[str, trust.People] | None = None,
) -> trust.Trust:
    """`trust.read` with this game's people's counts handed in as `people`.

    `people` is an already-folded mapping (from `counts_of`) for a caller reading many
    games; without it the store is asked for this one.
    """
    found = people[digest] if people is not None and digest in people else None
    if found is None:
        found = counts(connection, digest)
    return trust.read(digest, runs_dir, spec, found, nets_dir=nets_dir, rule=rule)
