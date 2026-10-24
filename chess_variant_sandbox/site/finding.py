"""Who may be shown a game: the line between *openable* and *findable*.

G7 of `docs/archive/plan-game-graph.md` drew it. A game is **findable** by a person when
it is a live row -- not killed, not deleted -- and either public and still in the library,
or that person's own. An admin is shown every live row, which is no more than `/admin`
already lists. An unlisted draft opens by its link, as it always has, but nobody else may
build from it: the gap button asks this of both ends of a gap (`web/server.App._between`)
and of a draft's own page before it offers one (`App._nearest`), so a person can build
the game between their own draft and a library game and nobody can build one out of
somebody else's unlisted draft.

Until M5 of `docs/archive/plan-map.md` this module also kept the store's `links` table -- every
pair of the store's games with `spec/relate.py`'s count of rules and similarity index --
because a game's neighbour list was ranked by them and narrowed per person. Near now has
one meaning, the map's (`web/graphing.near`), over the map's own games, which are the same
for everybody, so no list ranks by the table and `0038_links_retired.sql` dropped it.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

PUBLIC = "public"
"""A game everybody may find. `site/drafts.py` writes the same word."""


def _findable(viewer: int | None, admin: bool) -> tuple[str, list[Any]]:
    """The predicate that decides whether a game may be named to this person, on `g`.

    Killed and deleted are out for everybody: one is the admin's *no more of this* and
    the other took the rules away. Past those, a person is shown the library -- public
    and still in the tree -- and their own games, whatever state those are in. An admin
    is shown every live row, which is what `/admin` already lists.

    A signed-out visitor has no id, and `owner = null` is never true, so the second
    clause costs them nothing and needs no branch of its own.
    """
    live = "g.deleted is null and g.revoked is null"
    if admin:
        return live, []
    return (
        f"{live} and ((g.visibility = %s and g.retired is null) or g.owner = %s)",
        [PUBLIC, viewer],
    )


def findable(
    connection: Any,
    viewer: int | None = None,
    admin: bool = False,
    among: Collection[str] | None = None,
) -> set[str]:
    """Which games this person may be shown at all, or which of `among` they may.

    The gap button asks it about the two ends of a gap (`web/server.App._between`), which
    is why `among` exists: two rows rather than the whole library.

    A hash the store holds no row for is not findable, whatever the tree says. The deploy
    writes every library game before the app restarts, so that answer belongs to a store
    nothing has loaded yet, and the narrow answer is the safe one to be wrong with.
    """
    where, params = _findable(viewer, admin)
    if among is not None:
        where = f"{where} and g.hash = any(%s)"
        params = [*params, sorted(set(among))]
    with connection.cursor() as cursor:
        cursor.execute(f"select g.hash from games g where {where}", params)
        return {str(row[0]) for row in cursor.fetchall()}
