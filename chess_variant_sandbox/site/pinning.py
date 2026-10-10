"""Which computer player sat in a seat, exactly: H4 of `docs/plan-human-data.md` §1.2.

A computer seat is named by its agent string, and the plan asked whether the string pins
the player. **It does not.** The string names a kind, a budget and, where there is one, a
stored objective (`minimax:fit=<hash>,nodes=6400`, `minimax:h=<digest>,nodes=6400`), so a
fitted or written objective is pinned by its hash. What it does not name is everything else
a node-budgeted search is a function of: the core's build, the game's compiled tables and
the piece values handed to the core. A core change that alters the search plays a different
game under the same string, and so does a compiler or evaluator change. A learned seat's
net was already pinned by `seat*_played` (`0025_played.sql`, Q709).

So the row keeps the rest beside the string (`matches.pinned`, `0033_pinned_and_alarm.sql`):
a list of the identities its computer moves were chosen under, each the three digests
`measurement.py` already stamps a score's basis with. One entry is one player throughout;
a deploy mid-game appends a second. A row from before the column is *unrecorded*, which is
never read as *the same player*.

What this does **not** change: *Unbeaten*, the opponent line and the trust line still read
the agent string alone, and a clear is decided at the time and stays (`unbeaten.py`). That
those counts span a change of the player behind one string is recorded in `docs/bugs.md`;
the readings this plan adds (the bar, `site/alarm.py`) say how many of their games were
played by the player there is today.

Nothing here commits.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

SAME = "same"
EARLIER = "earlier"
UNRECORDED = "unrecorded"


@lru_cache(maxsize=256)
def _identity(spec_json: str, agents: tuple[str, ...]) -> tuple[str, str, str]:
    from chess_variant_sandbox import measurement

    spec = json.loads(spec_json)
    return (
        measurement.built_fingerprint(),
        measurement.tables_fingerprint(spec),
        measurement.players_fingerprint(spec, agents),
    )


def identity(spec: Mapping[str, Any], agents: Sequence[str]) -> dict[str, str]:
    """What the computer seats of a match on `spec` are, beyond their agent strings.

    `agents` is the row's computer seats in board order. Memoised per process: none of the
    three can change under a running process (the build is the loaded extension's, and the
    tables and values are functions of the spec and the loaded code).
    """
    build, tables, players = _identity(
        json.dumps(spec, sort_keys=True, separators=(",", ":")), tuple(agents)
    )
    return {"build": build, "tables": tables, "players": players}


def note(connection: Any, match_id: str, entry: Mapping[str, str]) -> None:
    """Append `entry` to the row's `pinned` unless it is already the last one.

    A store that has not taken `0033_pinned_and_alarm.sql` yet records nothing, as a row
    from before it, and the match plays on: `start` and a reply keep working on a store a
    deploy has not migrated, as they have across every earlier column (`tests/
    test_site_matches.py`'s match carried across `0014`). The statement runs in a savepoint
    so the refusal leaves the caller's transaction as it was.
    """
    from psycopg import errors

    with connection.cursor() as cursor:
        # The connection opens its transaction at the first statement, so this is always
        # a savepoint inside the caller's, never a commit of its own.
        cursor.execute("savepoint pinning")
        try:
            cursor.execute(
                "update matches set pinned = coalesce(pinned, '[]'::jsonb)"
                " || jsonb_build_array(%s::jsonb)"
                " where id = %s and (pinned is null or pinned -> -1 <> %s::jsonb)",
                (json.dumps(dict(entry)), match_id, json.dumps(dict(entry))),
            )
        except errors.UndefinedColumn:
            cursor.execute("rollback to savepoint pinning")
        cursor.execute("release savepoint pinning")


def recorded(connection: Any, match_ids: Sequence[str]) -> dict[str, list[dict[str, str]]]:
    """Each named row's `pinned`, absent for a row that has none."""
    if not match_ids:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, pinned from matches where id = any(%s) and pinned is not null",
            (list(match_ids),),
        )
        return {str(found[0]): list(found[1]) for found in cursor.fetchall()}


def compared(entries: Sequence[Mapping[str, Any]] | None, now: Mapping[str, str]) -> str:
    """`SAME` when every move of the row was chosen by the player `now` names.

    `EARLIER` when any entry differs (a game played wholly or partly by another build,
    table or value set), `UNRECORDED` for a row from before the column.
    """
    if not entries:
        return UNRECORDED
    return SAME if all(dict(entry) == dict(now) for entry in entries) else EARLIER
