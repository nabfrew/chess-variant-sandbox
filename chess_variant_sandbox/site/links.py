"""How near two of the store's games are, and who may be shown that.

G7 of `docs/archive/plan-game-graph.md`. `spec/relate.py` compares two specs rule by rule
and `web/graphing.py` keeps its readings in `runs/graph/pairs.json`. A file is keyed by
hash and by nothing else, so it can say *these two games are one rule apart* and cannot
say *and this person may not know that either of them exists*. Phase 2 puts drafts in the
library that not everybody may open, and a link spells out what two games differ in. So
the site keeps its pairs in the store, beside the `games` rows that answer who may see
what.

## What a link is, and what it is not

A row is one unordered pair, `a < b`, with the count of rules that differ and the
similarity index with its six counts (`0013_similarity.sql`). Within `NEAR` it also
carries the reading -- the kinds, the schema position and the sentence -- and past it
carries nothing but the numbers, because a list of nine differences is a ranking of how
unlike everything else a game is rather than a reading of it. Every pair gets a row,
including the far ones, because a game nothing is near is shown its nearest few and that
ranking needs them all.

Nothing here decides *what* differs. `relate` does, this stores what it said, and the row
carries `RELATE` so a reading that moves is recomputed rather than trusted.

## The order, which is the local application's

`web/library.py::neighbours` orders a game's links and this orders the same rows the same
way, so the site and the local application name the same neighbours in the same order
(`docs/archive/plan-game-graph.md` §9.5, T3 to T5). Within `NEAR`: the count of rules,
then the index, then the name, then the hash. With nothing within `NEAR`: the four nearest by
`spec/distance.py`, which is `library.nearest_by_feature` and not a second copy of it.

## Who may be shown a link

One predicate, `_findable`, and every reader uses it. A game is findable when it is a live
row -- not killed, not deleted -- and either public and still in the library, or this
person's own. An admin is shown every live row, which is no more than `/admin` already
lists. The predicate is on *the other end* of a link: whose page is being drawn is the
caller's question and it has already been answered (`site/server._playable`).

That makes an unlisted draft's existence its owner's business. It opens by its link, as it
always has, but it is named on nobody else's page and counted in nobody else's neighbour
list -- which is the difference between *openable* and *findable*, and the one this stage
is about. The gap button asks the same predicate of a gap's two ends (`findable`), so a
person can build the game between their own draft and a library game and nobody can
build one out of somebody else's unlisted draft.

## What it costs

A draft against the seventy games the library holds is about 50 ms in a cold process and
half that warm, because `relate` memoises each spec's normal form and each pair it has
compared. A whole library against itself is 2,415 pairs and about 1.4 s, paid once by the
deploy's load step; a second load compares nothing, because the rows already carry this
version. A far game's page reads the specs of the games it may be shown and pays
`distance.py` against each, about 25 ms over the corpus, which is what the local
application pays for the same page.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.spec import relate
from chess_variant_sandbox.web.library import MOST_LINKS, NEAR, PREVIEW, nearest_by_feature

PUBLIC = "public"
"""A game everybody may find. `site/drafts.py` writes the same word."""

RELATE = (
    f"{relate.RELATE_VERSION}"
    f" similarity {relate.SIMILARITY_RATIO}:1, setting {relate.SETTING_WEIGHT:g}"
)
"""The reading a stored row was taken under. A row carrying anything else is recomputed.

`RELATE_VERSION` moves when what differs is read differently. The stored index also
moves with the two constants it is weighed by, and those can change without a version
bump -- `web/graphing.py` keeps the counts and not the number for that reason -- so they
are in the stamp and a changed ratio recomputes the table rather than leaving it stale.
"""


@dataclass
class Written:
    """What one pass over the pairs did, and how much of it was arithmetic.

    `computed` is the acceptance G7's row states: a second load must compare no pair. It
    is separate from `added` because a pair can be written without being compared -- a row
    corrected from an older `relate` version is both, and a pair already current is
    neither.
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    computed: int = 0

    @property
    def written(self) -> int:
        return self.added + self.updated

    @property
    def rows(self) -> int:
        return self.added + self.updated + self.unchanged

    def __str__(self) -> str:
        return (
            f"{self.rows} pair(s): {self.added} added, {self.updated} updated,"
            f" {self.computed} compared"
        )


@dataclass(frozen=True)
class Neighbour:
    """One link, read from the other game's side.

    `units` is the count of rules and `similarity` the index over `counts`, both written
    for every pair; the rest is empty past `NEAR`, which is what the table holds.
    `similarity` is `None` only on a row no load has recompared since `0013`. The score
    gap is not here and never will be: it is a reading of two measurements that move
    under the games, and `web/library.py` is where it is worked out once.
    """

    hash: str
    name: str
    units: int
    similarity: float | None
    counts: dict[str, int]
    kind: str
    kinds: tuple[str, ...]
    unit: str
    sentence: str
    differences: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "name": self.name,
            "units": self.units,
            "similarity": self.similarity,
            "counts": dict(self.counts),
            "kind": self.kind,
            "kinds": list(self.kinds),
            "unit": self.unit,
            "sentence": self.sentence,
            "differences": [dict(one) for one in self.differences],
        }


# ------------------------------------------------------------------------ the writing


def specs_of(connection: Any, without: str = "") -> dict[str, dict[str, Any]]:
    """Every game the store still holds rules for, by hash.

    A deleted game has no spec at all, so it is absent by the column rather than by a
    clause. A killed or retired game is here: its row and its links stay, and which of
    them a person is shown is `_findable`'s question and not this one.

    A preview is not here (Q931, `load.PREVIEW`): a link carries the gap between two
    scores, and a preview is in no comparison, so it neither has links nor is one.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, spec from games where spec is not null and origin <> %s",
            (PREVIEW,),
        )
        found = {str(row[0]): row[1] for row in cursor.fetchall()}
    found.pop(without, None)
    return {digest: spec for digest, spec in found.items() if isinstance(spec, dict)}


def fill(connection: Any, specs: Mapping[str, dict[str, Any]]) -> Written:
    """Every pair among these games, comparing only the pairs no current row covers.

    The deploy's load step, and the acceptance it has to meet. The pairs already stored
    under this `relate` version are read in one query before anything is compared, so a
    second load compares nothing and writes nothing.
    """
    digests = sorted(specs)
    held = _held(connection)
    written = Written()
    rows = [
        _row(first, second, specs, held, written)
        for index, first in enumerate(digests)
        for second in digests[index + 1 :]
    ]
    _upsert(connection, rows)
    return written


def write(connection: Any, digest: str, spec: dict[str, Any]) -> Written:
    """One game's pairs against every other game the store holds.

    What a save calls, so a draft has its links from the moment it exists rather than
    from the next deploy. Repeating a save costs one query and no comparison, because
    the pairs are already current.
    """
    others = specs_of(connection, without=digest)
    specs = {digest: spec, **others}
    held = _held(connection, digest)
    written = Written()
    rows = [
        _row(*_ordered(digest, other), specs, held, written) for other in sorted(others)
    ]
    _upsert(connection, rows)
    return written


def forget(connection: Any, digest: str) -> int:
    """Remove every row that names this game. Returns how many went.

    `admin.delete_game` calls it. The rules were deleted and a link is a sentence about
    them, so the sentence goes with them; the `games` row itself stays, because every
    match, rating and vote in this store hangs off its hash.
    """
    with connection.cursor() as cursor:
        cursor.execute("delete from links where a = %s or b = %s", (digest, digest))
        return int(cursor.rowcount or 0)


def _ordered(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _held(connection: Any, digest: str = "") -> dict[tuple[str, str], str]:
    """Which pairs are stored already, and under which reading of them.

    One query rather than one per pair: a library of 500 games is 125,000 pairs, and a
    round trip apiece is the difference between a deploy step and a deploy. The version
    comes back with the pair so that a row written under an older `relate` is an update
    and a pair nothing has read yet is an insert, without a second query to tell them
    apart.

    A row without the similarity index is not held, whatever its stamp says. That is a
    row from before `0013_similarity.sql`, and the first load after the migration
    compares it again; it comes back stamped with nothing, so it is an update rather
    than an insert.
    """
    where = ""
    params: list[Any] = []
    if digest:
        where = " where a = %s or b = %s"
        params = [digest, digest]
    with connection.cursor() as cursor:
        cursor.execute(
            "select a, b,"
            " case when similarity is null or counts is null then '' else relate end"
            f" from links{where}",
            params,
        )
        return {(str(row[0]), str(row[1])): str(row[2]) for row in cursor.fetchall()}


def _row(
    first: str,
    second: str,
    specs: Mapping[str, dict[str, Any]],
    held: dict[tuple[str, str], str],
    written: Written,
) -> tuple[Any, ...] | None:
    """One pair's row, compared only when no row already carries this reading of it.

    `None` for a pair already current. The index is `relate.similarity`'s as it stands,
    twins included: this never compiles a spec, so two spellings of one game keep the
    index `relate` gives them, as they keep their own `units`. The map is what merges a
    twin and reads it at 0.
    """
    stored = held.get((first, second))
    if stored == RELATE:
        written.unchanged += 1
        return None
    one, other = relate.normal_form(specs[first]), relate.normal_form(specs[second])
    relation = relate.relate_forms(one, other)
    alike = relate.similarity_forms(one, other)
    written.computed += 1
    if stored is None:
        written.added += 1
    else:
        written.updated += 1
    return (
        first,
        second,
        relation.units,
        alike.index,
        json.dumps(alike.counts),
        *_reading(relation),
        RELATE,
    )


def _upsert(connection: Any, rows: Iterable[tuple[Any, ...] | None]) -> None:
    """Write the rows `_row` made, in one batch rather than one round trip apiece.

    `executemany` sends them pipelined, so the store is waited on once per batch and not
    once per pair. A round trip apiece was estimated at about 60 s of a 115 s version bump
    at 500 games; over the corpus's 2,485 pairs the writing went from 0.79 s to 0.18 s
    (Q354, a scratch store on this machine). An upsert rather than a `copy`, because a
    row written under an older reading is replaced in place.
    """
    batch = [row for row in rows if row is not None]
    if not batch:
        return
    with connection.cursor() as cursor:
        cursor.executemany(
            "insert into links (a, b, units, similarity, counts, kind, kinds, unit,"
            " sentence, differences, relate)"
            " values (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s)"
            " on conflict (a, b) do update set units = excluded.units,"
            " similarity = excluded.similarity, counts = excluded.counts,"
            " kind = excluded.kind, kinds = excluded.kinds, unit = excluded.unit,"
            " sentence = excluded.sentence, differences = excluded.differences,"
            " relate = excluded.relate, computed = now()",
            batch,
        )


def _reading(relation: relate.Relation) -> tuple[str, list[str], str, str, str | None]:
    """The words on a link, within `NEAR`, and nothing at all past it.

    Past three rules the count is the whole answer, so the columns are left empty rather
    than filled with a paragraph nobody draws. `NEAR` is `web/library.py`'s, so the page
    and the table agree on where a reading stops without a second number saying so.
    """
    if relation.units > NEAR or not relation.differences:
        return "", [], "", "", None
    first = relation.differences[0]
    return (
        first.kind,
        sorted({difference.kind for difference in relation.differences}),
        first.unit,
        relation.sentence,
        json.dumps([difference.as_dict() for difference in relation.differences]),
    )


# ------------------------------------------------------------------------ the reading


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

    The same predicate the neighbour query applies, asked on its own for a caller that
    is narrowing something other than one game's links. The gap button asks it about the
    two ends of a gap (`web/server.App._between`), which is why `among` exists: two rows
    rather than the whole library. The map's node set will ask it too, when the map
    learns to draw a person's own games.

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


def neighbours(
    connection: Any,
    digest: str,
    viewer: int | None = None,
    admin: bool = False,
    near: int = NEAR,
    most: int = MOST_LINKS,
) -> list[Neighbour]:
    """This game's links, nearest first, with everything this person may not see left out.

    Within `near` the answer is every such link, capped at `most`, by the count of rules,
    then the similarity index, then the name, then the hash -- `library.neighbours`'
    order, so the cap cuts where the local application's does. The name is compared
    lower-cased and byte by byte, as Python compares it, rather than by the database's
    collation, which skips spaces and hyphens.

    With nothing inside that range it is the four nearest by `spec/distance.py`, because
    *nothing here is like this game* is the most useful thing a page can say about a game
    the library has no relatives for, and `docs/archive/plan-game-graph.md` §9.5's T5
    found the features name a far game's relatives better than the count or the index do.
    The four are picked among the games this person may be shown, from the specs the store
    holds.
    """
    where, params = _findable(viewer, admin)
    other = "case when l.a = %s then l.b else l.a end"
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {other} as other, g.name, l.units, l.similarity, l.counts, l.kind,"
            "       l.kinds, l.unit, l.sentence, l.differences"
            f"  from links l join games g on g.hash = {other}"
            f" where (l.a = %s or l.b = %s) and {where}"
            ' order by l.units, l.similarity, lower(g.name) collate "C", other',
            [digest, digest, digest, digest, *params],
        )
        found = [
            Neighbour(
                hash=str(row[0]),
                name=str(row[1]),
                units=int(row[2]),
                similarity=None if row[3] is None else float(row[3]),
                counts={str(key): int(value) for key, value in (row[4] or {}).items()},
                kind=str(row[5]),
                kinds=tuple(str(kind) for kind in (row[6] or ())),
                unit=str(row[7]),
                sentence=str(row[8]),
                differences=tuple(dict(one) for one in (row[9] or ())),
            )
            for row in cursor.fetchall()
        ]
    inside = [link for link in found if link.units <= near]
    if inside:
        return inside[:most]
    return _far(connection, digest, found)


def _far(connection: Any, digest: str, found: list[Neighbour]) -> list[Neighbour]:
    """The four of `found` nearest this game by feature, nearest first.

    `library.nearest_by_feature` decides, over the specs the store holds for this game
    and for each of `found`, so the site and the local application cannot pick by two
    rules. Only a far game's page asks, and it reads one spec per game it may be shown.
    """
    if not found:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            "select hash, spec from games where hash = any(%s) and spec is not null",
            ([digest, *(link.hash for link in found)],),
        )
        specs = {str(row[0]): row[1] for row in cursor.fetchall()}
    mine = specs.get(digest)
    if not isinstance(mine, dict):
        # A game with links has its rules: `admin.delete_game` takes both together.
        return []
    by_hash = {link.hash: link for link in found}
    four = nearest_by_feature(
        mine,
        (
            (link.hash, link.name, specs[link.hash])
            for link in found
            if isinstance(specs.get(link.hash), dict)
        ),
    )
    return [by_hash[other] for _, other in four]
