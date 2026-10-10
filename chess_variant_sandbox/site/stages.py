"""The one word for where a game stands, decided in one place.

Q503b of `docs/plan-game-pipeline.md` (§2.1 the levels, §3 the pages that show them). A
game is at exactly one level, and every page that names it -- the card, the library's
tabs, the designer's *Where this game stands*, the game page's *Where it stands*, the
profile's *Your games* -- asks `stage` (on the site) or `local` (the application on a
checkout, which has no rows). No page script holds a level word: each draws the word
and the track it is sent.

**On the site the row decides** (`stage`), from five columns:

- `origin`: `archive` is **Library** and `corpus` is **Baseline** (`load.ARCHIVED`,
  `load.CORPUS`). Where a game's file sits is the maintainer's act, so it outranks
  everything the site itself records.
- `retired`: the loader's stamp when a Library or Baseline game's file has left the tree.
  **Retired** outranks the origin: a game that left the tree is at no level on it.
- `visibility`: a designed row that is not `public` is a **Draft** (`drafts.keep` writes
  every save `unlisted`; *Publish* makes it `public`).
- `nominated`: a public designed row with the stamp is **Shortlisted** (`shortlist.nominate` sets it),
  and without it **On trial**.

- `rested`: a designed row the sweep stamped is **Resting** (Q503i, `site/resting.py`): a
  trial game nobody plays leaves the listing, keeps its page, and may be published again
  once. A shortlisted game is never stamped.

**The round** (Q503h, `site/rounds.py`) adds no level and no column `stage` reads. A
round's vote down or veto clears `nominated`, so the game reads On trial again; a confirm
leaves it Shortlisted until `site load` finds its spec in `archive/` and takes the row over
with `origin = 'archive'` (a `level_moves` row from `designed`), and it then reads Library.
`said` asks `rounds.said` for a Shortlisted or On trial game's sentence: voting, passed,
confirmed, or declined and waiting.

A column a row does not carry reads as absent, so a caller holding fewer columns gets the
level those columns allow rather than an error.

**The Baseline's moves** (Q503n). Only the maintainer moves a game between `archive/`
(Library) and `rulesets/` (Baseline), by the procedure in
`docs/reference/baseline-procedure.md`; the site learns of it when `site load` corrects
the row's origin, and the loader writes a `level_moves` row with the maintainer's reason.
`said` gives the game page's sentence (*"One of the games every score is measured against,
since 29 Sep 2026: ..."*) and `whats_new` the *What's new* entries. No route writes an
origin.

The kill switch and deletion are not levels. They decide whether a game is shown at all,
and `drafts.listed` and `admin.unopenable_games` already answer that before a level is
asked for.

**Locally the file decides** (`local`): `corpus.find` and `library.discover` call a game
in `rulesets/` *committed* and one in `archive/` *archived*, which are Baseline and Library
here too. Everything else on a checkout is the maintainer's own work in `runs/` -- a design
run's candidate, a hand-built game -- and reads as Draft. A checkout has no trial, no
shortlist and no round, so it never says those words.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

DRAFT = "Draft"
ON_TRIAL = "On trial"
SHORTLISTED = "Shortlisted"
LIBRARY = "Library"
BASELINE = "Baseline"
RESTING = "Resting"
RETIRED = "Retired"

LEVELS = (DRAFT, ON_TRIAL, SHORTLISTED, LIBRARY, BASELINE, RESTING, RETIRED)
"""Every word `stage` and `local` return, in the order the profile groups by."""

TRACK = (DRAFT, ON_TRIAL, SHORTLISTED, LIBRARY)
"""The four steps a designer's game can take (§3.1). The Baseline is the maintainer's
act, not a step, so a Baseline game shows its own word where Library would be lit."""

ARCHIVE = "archive"
CORPUS = "corpus"
DESIGNED = "designed"
"""The loader's two origins (`load.ARCHIVED`, `load.CORPUS`), spelled here rather than
imported so that the local application can read this module without the store's."""

PUBLIC = "public"

COMMITTED = "committed"
ARCHIVED = "archived"
LOCAL_RETIRED = "retired"
"""`corpus.find`'s and `library.discover`'s words for a file's place on a checkout."""

TABS = (
    {
        "key": "library",
        "label": LIBRARY,
        "levels": [LIBRARY, BASELINE, SHORTLISTED],
        "sort": "index",
    },
    {"key": "trial", "label": ON_TRIAL, "levels": [ON_TRIAL], "sort": "onTrialSince"},
)
"""The site's library tabs (§3.2), which levels each lists, and the card key it opens
sorted by, best or newest first. The page opens on the first. A level in no tab --
Draft, Resting, Retired -- is never listed."""

SAYS = {
    DRAFT: "A draft: only people with its link can open it. Its designer can publish it "
    "to put it on trial.",
    ON_TRIAL: "On trial: listed for anybody to play and vote on. Its designer can "
    "nominate it for the shortlist once enough other people have played it and liked it.",
    SHORTLISTED: "Shortlisted: members vote on it at the next monthly round, and the "
    "person who runs the site confirms or vetoes.",
    LIBRARY: "In the library: scored by the site and listed for everyone.",
    BASELINE: "One of the games every score is measured against.",
    RESTING: "Resting: nobody has finished a game on it lately, so it is off the listing. "
    "Its page and link still work.",
    RETIRED: "Retired: its rules have left the library, so it is not offered for new "
    "games. Its page and its games stay.",
}
"""One sentence per level, for the game page's panel and the profile's rows."""

NEXT = {
    # Its link works already, and was the one thing a designer could not find (the
    # first-visit study's B3): the page's *copy link* sends it. Publishing is the next
    # step, in the designer's own words (Q563 owns the button's sentences).
    DRAFT: "Its link works now: share it with the copy link button on its page. Publish "
    "it from the designer to put it on trial.",
    ON_TRIAL: "It can be nominated once enough other people have played it and voted.",
    SHORTLISTED: "Waiting for the next round.",
    RESTING: "Publish it again from the designer to put it back on trial. That works once.",
}
"""What the designer does next, for *Your games*. A level with nothing for the designer
to do has no entry."""

WAITING = (
    "Waiting for the site's check: publishing will be complete once the site has checked "
    "the score measured on your computer, and it goes on trial by itself."
)
"""A draft whose designer pressed *Publish* while the site was still checking its score
(Q742, `drafts.settle`), as its designer reads it on the game's page and in *Your games*.
Anybody else reads the draft's plain sentence: the press is the designer's business."""


def stage(row: Mapping[str, Any]) -> str:
    """The level of the game this `games` row holds. See the module for the order.

    `row` is any mapping with the row's columns by name; one it lacks reads as absent.
    """
    if row.get("retired") is not None:
        return RETIRED
    origin = str(row.get("origin") or "")
    if origin == ARCHIVE:
        return LIBRARY
    if origin == CORPUS:
        return BASELINE
    if row.get("rested") is not None:
        return RESTING
    if str(row.get("visibility") or "") != PUBLIC:
        return DRAFT
    if row.get("nominated") is not None:
        return SHORTLISTED
    return ON_TRIAL


def local(source: str) -> str:
    """The level of a game on a checkout, from where its file is (see the module)."""
    if source == COMMITTED:
        return BASELINE
    if source == ARCHIVED:
        return LIBRARY
    if source == LOCAL_RETIRED:
        return RETIRED
    return DRAFT


def track(level: str) -> list[dict[str, Any]]:
    """The designer's four steps with this level lit, as the pages draw them.

    A Baseline game shows *Baseline* in Library's place, lit. Resting and Retired light
    no step: the panel's heading names them.
    """
    steps = [BASELINE if step == LIBRARY and level == BASELINE else step for step in TRACK]
    return [{"word": step, "lit": step == level} for step in steps]


def tab(level: str) -> str | None:
    """The key of the library tab that lists a game at this level, or `None`."""
    for entry in TABS:
        if level in entry["levels"]:
            return str(entry["key"])
    return None


GLOSSES = {BASELINE: "baseline"}
"""The (i) a level word carries, by the page's key for it (`core.js` `GLOSS`): a word the
site's vocabulary keeps and defines (`docs/plan-understanding.md` §3.2). The page holds no
level word of its own, so it asks this rather than comparing the word (G41)."""


def shown(level: str, says: str | None = None) -> dict[str, Any]:
    """What a page is sent about a level: the word, its sentence, the track and its (i).

    `says` replaces the level's plain sentence with one that knows the game (`said`).
    """
    sentence = SAYS.get(level, "") if says is None else says
    return {"level": level, "levelSays": sentence, "track": track(level),
            "levelGloss": GLOSSES.get(level, "")}


COLUMNS = (
    "hash", "origin", "visibility", "nominated", "retired", "published", "rested", "woken",
    "owner", "publish_asked",
)
"""What `read` asks of a row. `stage` reads five of them; `published` dates a trial
game for the trial tab's newest-first order; `woken` and `owner` word a Resting game's
sentence (`said`); `publish_asked` and `owner` word a draft waiting for the site's check
(`WAITING`)."""


def read(connection: Any, hashes: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Each game's row, by hash, with its `level` beside the columns `stage` read.

    A hash the store has no row for is absent from the answer, and its caller keeps what
    it already had.
    """
    wanted = sorted({str(digest) for digest in hashes})
    if not wanted:
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {', '.join(COLUMNS)} from games where hash = any(%s)", (wanted,)
        )
        rows = [dict(zip(COLUMNS, found, strict=True)) for found in cursor.fetchall()]
    for row in rows:
        row["level"] = stage(row)
    return {str(row["hash"]): row for row in rows}


def yours(connection: Any, owner: int) -> list[dict[str, Any]]:
    """The profile's *Your games* (§3.4): this person's games, grouped by level.

    Groups come in `LEVELS` order and only where they hold a game; each game is named with
    its next step (`NEXT`), which its page's designer link takes. A game taken off the site
    or deleted is not here: neither opens, and the admin's act is not a level.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"select name, {', '.join(COLUMNS)} from games where owner = %s"
            " and revoked is null and deleted is null order by lower(name), hash",
            (owner,),
        )
        found = [
            (str(row[0]), dict(zip(COLUMNS, row[1:], strict=True)))
            for row in cursor.fetchall()
        ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, row in found:
        level = stage(row)
        # A resting game that has come back once has no next step (Q503i).
        wakes = level == RESTING and row.get("woken") is None
        # The next step is taken in the designer for a draft (*Publish*) and a resting
        # game that may come back, and on the game's own page, which the name links to,
        # for everything after.
        grouped.setdefault(level, []).append({
            "hash": str(row["hash"]),
            "name": name,
            "next": WAITING if row.get("publish_asked") is not None
            else "" if level == RESTING and not wakes else NEXT.get(level, ""),
            "inDesigner": level == DRAFT or wakes,
        })
    order = [*LEVELS, *sorted(set(grouped) - set(LEVELS))]
    return [
        {"level": level, "says": SAYS.get(level, ""), "games": grouped[level]}
        for level in order
        if level in grouped
    ]


# ------------------------------------------------------- Q503n, the Baseline's moves


@dataclass(frozen=True)
class Move:
    """One move of a game between the Library and the Baseline, as `site load` found it."""

    game: str
    name: str
    was: str
    became: str
    reason: str
    moved: datetime

    @property
    def joined(self) -> bool:
        """Whether it joined the Baseline, rather than left it for the Library."""
        return self.became == CORPUS

    @property
    def entered(self) -> bool:
        """Whether a player-made game came in from a round (Q503h): `was` is `designed`."""
        return self.was == DESIGNED

    @property
    def page(self) -> str:
        """The game page's sentence while the game stays where this move put it."""
        tail = f": {self.reason}" if self.reason else "."
        if self.entered and not self.joined:
            return f"In the library since {day(self.moved)}{tail}"
        if self.joined:
            return (
                f"One of the games every score is measured against, since "
                f"{day(self.moved)}{tail}"
            )
        return (
            f"{SAYS[LIBRARY]} It left the games every score is measured against on "
            f"{day(self.moved)}{tail}"
        )

    @property
    def heading(self) -> str:
        if self.entered:
            return f"{self.name} joins the " + ("Baseline" if self.joined else "Library")
        if self.joined:
            return f"{self.name} joins the Baseline"
        return f"{self.name} leaves the Baseline for the Library"

    @property
    def paragraphs(self) -> list[str]:
        if self.entered:
            lines = [
                "Its designer's game, passed by the members at a round and confirmed by the "
                "person who runs the site, is now in the library: scored by the site and "
                "listed for everyone."
            ]
            if self.reason:
                lines.append(f"The reason given: {self.reason}")
            return lines
        said = (
            "It is now one of the games every score is measured against."
            if self.joined
            else "It stays in the library with its score, and no longer sets the scale "
            "the other games are measured against."
        )
        lines = [said]
        if self.reason:
            lines.append(f"The reason given: {self.reason}")
        lines.append(
            "Scores stay as they are until every game is re-scored, which happens after "
            "several such changes."
        )
        return lines


def day(when: datetime) -> str:
    """A date as the page says it: *3 Oct 2026* (`shortlist.day`'s form), on UTC's
    calendar, as `whats_new` files the same move, whatever the database session's zone
    (Q730)."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc)
    return f"{when.day} {when:%b %Y}"


_MOVE_COLUMNS = "m.game, g.name, m.was, m.became, m.reason, m.moved"


def _move(row: Any) -> Move:
    return Move(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), row[5])


def last_move(connection: Any, digest: str) -> Move | None:
    """This game's latest move between the Library and the Baseline, or `None`."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_MOVE_COLUMNS} from level_moves m join games g on g.hash = m.game"
            " where m.game = %s order by m.moved desc, m.id desc limit 1",
            (digest,),
        )
        found = cursor.fetchone()
    return None if found is None else _move(found)


NEWS_KEPT = 50
"""How many of the latest moves *What's new* carries. A Baseline change is rare by design
(`docs/plan-game-pipeline.md` §2.1), so this is years of them."""


def moves(connection: Any, limit: int = NEWS_KEPT) -> list[Move]:
    """The latest moves, newest first, for *What's new*. A deleted game's are left out."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_MOVE_COLUMNS} from level_moves m join games g on g.hash = m.game"
            " where g.deleted is null order by m.moved desc, m.id desc limit %s",
            (limit,),
        )
        return [_move(row) for row in cursor.fetchall()]


def said(connection: Any, row: Mapping[str, Any], viewer: int | None) -> str:
    """The game page's sentence for this row's level, knowing the game.

    A Resting game says since when and whether it can come back (`resting.said`). A
    Baseline or Library game that `site load` saw move says since when and the
    maintainer's reason, while it is still where that move put it. Every other level, and
    a Baseline game that has always been one, says its plain sentence (`SAYS`).
    """
    level = str(row.get("level") or stage(row))
    if row.get("publish_asked") is not None and level in (DRAFT, RESTING):
        owner = row.get("owner")
        if viewer is not None and owner is not None and int(owner) == viewer:
            return WAITING
    if level == RESTING:
        # Imported here: the local application reads this module without the store's.
        from chess_variant_sandbox.site import resting

        owner = row.get("owner")
        mine = viewer is not None and owner is not None and int(owner) == viewer
        return resting.said(row.get("rested"), row.get("woken"), mine)
    if level in (BASELINE, LIBRARY) and row.get("hash"):
        move = last_move(connection, str(row["hash"]))
        if move is not None and move.became == str(row.get("origin") or ""):
            return move.page
    if level in (SHORTLISTED, ON_TRIAL) and row.get("hash"):
        # Q503h. Where the round has put it: voting, passed, confirmed, or declined and
        # waiting. Imported here for the same reason as `resting`.
        from chess_variant_sandbox.site import rounds

        found = rounds.said(connection, {**row, "level": level})
        if found:
            return found
    return SAYS.get(level, "")


class Notice(Protocol):
    """Anything *What's new* files by date: a Baseline move, or a round's line (Q503h)."""

    @property
    def moved(self) -> datetime: ...

    @property
    def heading(self) -> str: ...

    @property
    def paragraphs(self) -> list[str]: ...


def news(found: Iterable[Notice]) -> list[dict[str, Any]]:
    """The moves as *What's new* days, newest first, in `web/changelog.py`'s shape.

    Every piece is a plain `text` run, never markup: a game's name and the maintainer's
    reason reach the page as strings, as the changelog's own entries do.
    """
    days: dict[str, dict[str, Any]] = {}
    for move in found:
        when = move.moved.astimezone(timezone.utc).date()
        stamp = when.isoformat()
        entry = {
            "heading": move.heading,
            "blocks": [
                {"kind": "paragraph", "runs": [{"t": "text", "s": line}]}
                for line in move.paragraphs
            ],
        }
        days.setdefault(stamp, {
            "date": stamp,
            "words": f"{when.day} {when:%B} {when.year}",
            "entries": [],
        })["entries"].append(entry)
    return [days[stamp] for stamp in sorted(days, reverse=True)]


def whats_new(payload: Mapping[str, Any], found: Iterable[Notice]) -> dict[str, Any]:
    """The changelog's days with the moves filed among them by date, newest first.

    A move lands at the top of the file's day with the same date, or as a day of its own
    before the first dated day older than it. The file's own order is otherwise kept.
    """
    days = [dict(entry) for entry in payload.get("days") or []]
    for extra in news(found):
        same = next(
            (at for at, known in enumerate(days) if known.get("date") == extra["date"]),
            None,
        )
        if same is not None:
            days[same] = {
                **days[same], "entries": extra["entries"] + list(days[same]["entries"])
            }
            continue
        older = next(
            (
                at for at, known in enumerate(days)
                if known.get("date") and str(known["date"]) < extra["date"]
            ),
            len(days),
        )
        days.insert(older, extra)
    note = str(payload.get("note") or "")
    return {**payload, "days": days, "note": "" if days else note}
