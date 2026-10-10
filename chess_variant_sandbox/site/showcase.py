"""The landing page's *Made by members* (Q741): the newest games members published here.

The user, 2026-10-05: *"Switch up the main page to highlight user-created games"*, then
*"just ones that have been published"*. So the shelf holds a game only when a member
designed it on the site and pressed *Publish* (`drafts.publish`): its row's origin is
`designed` and its level (`stages.stage`) is On trial or Shortlisted. A draft is never
here, and neither is a game the site came with (the launch set, the library, the
Baseline), nor a members' game the round has taken into the library, which is the
library's now. What the listing already leaves out stays out: a game resting, taken
off the site or deleted has no card (`drafts.offered`), so it has no place here.

**Newest published first, at most `SHOWN`.** `games.published` is the press of
*Publish* (a resting game's return restarts it), so the shelf is what members have
just put up, the same order as the *On trial* tab's. Four fit across the landing page
at 1400 px; on a phone the row scrolls sideways inside its own box. `total` is how many
qualify, so the page can say there are more on the games page.

**The maker is named as the row reads now**: a renamed member reads as their new
handle and one who left as `identity.FORMER`, with no mark, as on the game page's
*where it came from* (`server._designer_marks`).

Display only: a shelf decides nothing about any game and reads no score.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from chess_variant_sandbox.site import identity, stages

SHOWN = 4
"""How many games the shelf shows: one row across the landing page at 1400 px."""

LEVELS = (stages.ON_TRIAL, stages.SHORTLISTED)
"""The levels a published members' game stands at. Library and Baseline are the site's
own games, Draft is unpublished, and Resting and Retired are not listed."""


def made_here(
    connection: Any,
    cards: Iterable[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The shelf: the newest published members' games among the listing's `cards`.

    `rows` is `stages.read` over the same hashes. A card with no row, a row that is not
    a designed game, has no owner or was never published, is not on the shelf.
    """
    found = []
    for card in cards:
        row = rows.get(str(card["hash"]))
        if (
            row is None
            or row.get("origin") != stages.DESIGNED
            or row.get("level") not in LEVELS
            or row.get("owner") is None
            or row.get("published") is None
        ):
            continue
        found.append((card, row))
    found.sort(key=lambda pair: (-pair[1]["published"].timestamp(), str(pair[0]["hash"])))
    shown = found[:SHOWN]
    handles = _handles(connection, {int(row["owner"]) for _card, row in shown})
    games = []
    for card, row in shown:
        owner = int(row["owner"])
        handle = handles.get(owner, identity.FORMER)
        games.append({
            "hash": str(card["hash"]),
            "name": str(card.get("name") or ""),
            "by": handle,
            "mark": identity.mark_of(owner, handle),
            "level": row["level"],
            "published": row["published"].isoformat(),
        })
    return {"games": games, "total": len(found)}


def _handles(connection: Any, owners: set[int]) -> dict[int, str]:
    """Each owner's handle now, `identity.FORMER` for somebody who left."""
    if not owners:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, coalesce(handle, %s) from users where id = any(%s)",
            (identity.FORMER, sorted(owners)),
        )
        return {int(user): str(handle) for user, handle in cursor.fetchall()}
