"""Which game a `/game/<hash>` address opens on the site, and what it says (G3).

`web/page/game.py` draws; this decides what is drawn, from the store and the library the
site already holds. The doors are the game page's own (`SiteApp._playable`, `drafts.listed`,
`SiteApp.near`), asked here as a visitor would be answered, because the page is the same
bytes for everybody: it reads no cookie, so a crawler, a visitor and a member get one page,
and what a member may see beyond it is the router's to draw.

- A killed or a deleted game is the 404 an unknown hash is, as on `/api/game/<hash>`.
- A listed game (`drafts.listed`: public, not killed, retired, deleted or resting) is
  indexable. So is a library file the store has no row for yet, as the listing offers it.
- Any other game that opens by its link (an unlisted draft, a retired or a resting game, a
  draft with files and no row) is drawn with `noindex`.
- An address naming no game the store or the library holds is the 404, without re-reading
  the library: an invented hash costs one query, never a walk of the tree.

The near games are the page's own list in its order (`library.near_names`, the caller's
`near_of`): the map's games most like this one by the map's likeness, which is one answer
for everybody (M5 of `docs/plan-map.md`). They are cached per game for `NEAR_SECONDS`,
because a game the map does not draw is ranked by a relation over the whole library and a
crawler reads every page in a burst. A game killed in that window can stay named on a
neighbour's page for at most that long; its own address answers 404 at once.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.site import drafts, seo
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import front
from chess_variant_sandbox.web.page import game as drawing

NEAR_SECONDS = 300.0
"""How long one game's near list is reused before it is asked for again."""

LISTED_SECONDS = 300.0
"""How long the list of indexable games (the sitemap's, and `/game`'s without script) is
reused before the store is asked again (G4)."""

ROUTE = "GET /game/<hash>"
"""The page's row in `usage_hours`: every read is a visitor's, since the page reads no cookie."""


@dataclass(frozen=True)
class Row:
    """The store's switches on one game: what decides whether its page opens and is listed."""

    visibility: str
    revoked: bool
    retired: bool
    deleted: bool
    rested: bool

    @property
    def openable(self) -> bool:
        return not (self.revoked or self.deleted)

    @property
    def listed(self) -> bool:
        """`drafts.listed`'s predicate, on one row."""
        return (
            self.visibility == drafts.PUBLIC
            and not (self.revoked or self.retired or self.deleted or self.rested)
        )


def row_of(connection: Any, digest: str) -> Row | None:
    """One game's switches, or `None` when the store has no row for it."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select visibility, revoked, retired, deleted, rested from games where hash = %s",
            (digest,),
        )
        found = cursor.fetchone()
    if found is None:
        return None
    return Row(
        visibility=str(found[0]),
        revoked=found[1] is not None,
        retired=found[2] is not None,
        deleted=found[3] is not None,
        rested=found[4] is not None,
    )


def site_base() -> str:
    """The site's address for the canonical (`seo.site_base`), or empty where none is
    configured (a development machine), when the canonical is the path alone."""
    return seo.site_base()


def indexable(row: Row | None, game: library.Game) -> bool:
    """Whether a game's own address is for a search engine's index: a listed row, or, with
    no row yet, a library file the listing offers. The one rule the page's `noindex` and
    the sitemap both read."""
    if row is None:
        return game.source in drafts.CORPUS_SOURCES
    return row.listed


def _rows(connection: Any) -> dict[str, tuple[Row, Any]]:
    """Every game's switches and the day its row was made, in one query."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, visibility, revoked, retired, deleted, rested, created from games"
        )
        found = cursor.fetchall()
    return {
        str(one[0]): (
            Row(visibility=str(one[1]), revoked=one[2] is not None, retired=one[3] is not None,
                deleted=one[4] is not None, rested=one[5] is not None),
            one[6],
        )
        for one in found
    }


def _day(created: Any, game: library.Game) -> str:
    """The day the page last changed, as far as anything says: the later of the row's
    making and the score's measuring (the score line is on the page)."""
    days = []
    if created is not None and hasattr(created, "date"):
        days.append(created.date().isoformat())
    measured = str((game.score or {}).get("measured_at") or "")
    if re.match(r"^\d{4}-\d{2}-\d{2}", measured):
        days.append(measured[:10])
    return max(days) if days else ""


def listed_pages(
    connection: Any, known: Mapping[str, library.Game]
) -> tuple[front.Listed, ...]:
    """Every game whose own address is indexable, by name: the sitemap's games, and `/game`'s
    list without script. A killed or deleted game is never one, whatever its file says."""
    rows = _rows(connection)
    listed = []
    for digest, game in known.items():
        found = rows.get(digest)
        row = None if found is None else found[0]
        if row is not None and not row.openable:
            continue
        if indexable(row, game):
            listed.append(front.Listed(hash=digest, name=game.name,
                                       day=_day(None if found is None else found[1], game)))
    return tuple(sorted(listed, key=lambda one: (one.name.casefold(), one.hash)))


class ListedCache:
    """The indexable games, kept `LISTED_SECONDS`: one list per server, thread-safe. A game
    published or taken off joins or leaves the sitemap within that time; its own address
    answers at once."""

    def __init__(
        self, seconds: float = LISTED_SECONDS, clock: Callable[[], float] = time.monotonic
    ):
        self.seconds = seconds
        self.clock = clock
        self._held: tuple[float, tuple[front.Listed, ...]] | None = None
        self._lock = threading.Lock()

    def get(self, read: Callable[[], tuple[front.Listed, ...]]) -> tuple[front.Listed, ...]:
        now = self.clock()
        with self._lock:
            found = self._held
        if found is not None and now - found[0] < self.seconds:
            return found[1]
        listed = read()
        with self._lock:
            self._held = (now, listed)
        return listed

    def clear(self) -> None:
        with self._lock:
            self._held = None


class NearCache:
    """Each game's near list for a visitor, kept `NEAR_SECONDS`. One per server; thread-safe,
    since the server answers each request on a thread of its own."""

    def __init__(self, seconds: float = NEAR_SECONDS, clock: Callable[[], float] = time.monotonic):
        self.seconds = seconds
        self.clock = clock
        self._held: dict[str, tuple[float, tuple[drawing.Near, ...]]] = {}
        self._lock = threading.Lock()

    def get(
        self, digest: str, read: Callable[[], tuple[drawing.Near, ...]]
    ) -> tuple[drawing.Near, ...]:
        now = self.clock()
        with self._lock:
            found = self._held.get(digest)
        if found is not None and now - found[0] < self.seconds:
            return found[1]
        near = read()
        with self._lock:
            self._held[digest] = (now, near)
        return near

    def clear(self) -> None:
        with self._lock:
            self._held.clear()


def drawn(
    connection: Any,
    digest: str,
    known: Mapping[str, library.Game],
    game_of: Callable[[str], library.Game],
    near: NearCache,
    base: str,
    near_of: Callable[[library.Game], tuple[drawing.Near, ...]],
    picture_of: Callable[[str, library.Game], str | None] | None = None,
) -> drawing.Drawn | None:
    """What `/game/<digest>` draws, or `None` for the 404.

    `known` is the library as last read and `game_of` the site's one-game reader
    (`SiteApp.game`: the tree, then the store's copy of a retired game's rules).
    `picture_of` names the game's own link-preview picture where one is made
    (`game_pictures.Pictures.address`, G6), and the site's stands in where it says nothing.
    """
    row = row_of(connection, digest)
    if row is None:
        listed = known.get(digest)
        if listed is None:
            return None
        game = listed
    else:
        if not row.openable:
            return None
        try:
            game = game_of(digest)
        except KeyError:
            return None
    nearby = (
        () if game.source == library.PREVIEW
        else near.get(digest, lambda: near_of(game))
    )
    return drawing.Drawn(
        hash=digest,
        name=game.name,
        description=game.description,
        spec=game.spec,
        score=game.score,
        stale=game.stale,
        near=nearby,
        indexable=indexable(row, game),
        base=base,
        picture=(picture_of(digest, game) if picture_of is not None else None) or "",
    )
