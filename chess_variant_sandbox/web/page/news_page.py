"""*What's new* drawn on the server, `/whats-new` (G11 of `docs/plan-growth.md`).

The changelog reaches the single page as data and `web/static/page/whatsnew.js` builds an
element per piece, so a crawler, a feed reader and a person with script off saw an empty page
at `#/whats-new` (a fragment no server ever receives). This is the same view as markup: the
site answers `/whats-new` with the entries in the page, in the shell the site's other prose
pages wear (`site/pages.py`), and `site/feeds.py` writes the same entries as Atom.

**The payload is `web/changelog.py`'s** (`for_visitors`, with the Baseline's moves filed in
by `site/stages.py`), so this page, the script's view and the feed read one answer, and an
entry marked for maintainers is in none of them. Every piece of text is escaped here and
nothing from the file becomes markup: `runs_html` writes a `<code>` or an `<em>` for those
two kinds of piece and escapes the string inside, and an unknown kind is its own text,
as `whatsnew.js`'s `newsRun` has it.

**An entry's anchor** is `<day>-<its heading as a slug>`, with `-2`, `-3` for a heading a day
uses twice, so a link to an entry (and a feed's id for it) names the entry, not its place in
the file: a new entry above it moves nothing. An entry with no heading is `<day>-entry`.
Editing a heading in the file moves its anchor, which is the one thing that makes a reader
see it as new.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from chess_variant_sandbox.web import changelog
from chess_variant_sandbox.web.page import front

PATH = front.NEWS_PATH

TITLE = front.short_title("What's new")

DESCRIPTION = (
    "Every change to Chess variant sandbox in plain words, newest first: new games and "
    "rules to try free online in your browser. There is a feed to follow."
)

LEAD = (
    "What this project can do now that it could not before, newest first. It is written as "
    "the work lands, so it is a developer's diary of a research project rather than release "
    "notes."
)

EMPTY = "The changelog holds no entries yet."


@dataclass(frozen=True)
class Item:
    """One entry of the changelog as a page and a feed read it."""

    date: str
    words: str
    heading: str
    anchor: str
    blocks: tuple[Mapping[str, Any], ...]


def slug(heading: str) -> str:
    """A heading as the words of an anchor: lower case, letters and digits, one dash between."""
    flat = re.sub(r"[^a-z0-9]+", "-", heading.casefold()).strip("-")
    return flat[:60].strip("-")


def items(payload: Mapping[str, Any]) -> list[Item]:
    """The entries a visitor reads, newest day first, each with its anchor.

    `changelog.for_visitors` runs here too, so an entry marked for maintainers is never an
    item whatever payload it is handed; the server has run it already, and running it twice
    changes nothing. A day whose heading is not a date has no `date` and is kept for the
    page (it shows its own words) and left out of the feed (an entry with no day has no
    time to say).
    """
    found: list[Item] = []
    taken: set[str] = set()
    for day in changelog.for_visitors(dict(payload)).get("days") or []:
        stamp = str(day.get("date") or "")
        for entry in day.get("entries") or []:
            base = f"{stamp or 'undated'}-{slug(str(entry.get('heading') or '')) or 'entry'}"
            anchor, count = base, 1
            while anchor in taken:
                count += 1
                anchor = f"{base}-{count}"
            taken.add(anchor)
            found.append(Item(
                date=stamp, words=str(day.get("words") or stamp),
                heading=str(entry.get("heading") or ""), anchor=anchor,
                blocks=tuple(entry.get("blocks") or ()),
            ))
    return found


def runs_html(runs: Sequence[Mapping[str, Any]]) -> str:
    """One paragraph's pieces as markup: `code`, `em` and plain text, every string escaped."""
    parts = []
    for run in runs:
        said = escape(str(run.get("s") or ""))
        kind = run.get("t")
        parts.append(f"<code>{said}</code>" if kind == "code"
                     else f"<em>{said}</em>" if kind == "em" else said)
    return "".join(parts)


def blocks_html(blocks: Sequence[Mapping[str, Any]]) -> str:
    """An entry's blocks as markup: a paragraph each, and a rule for a rule."""
    return "".join(
        "<hr>" if block.get("kind") == "rule" else f"<p>{runs_html(block.get('runs') or [])}</p>"
        for block in blocks
    )


def body(payload: Mapping[str, Any]) -> str:
    """The page's main: the title, the lead, the two feeds, then each day's entries."""
    entries = items(payload)
    note = str(payload.get("note") or "")
    if not entries:
        said = f'<p class="muted">{escape(note or EMPTY)}</p>'
        return f"<h1>What's new</h1>\n{said}\n{feeds_html()}"
    days: list[str] = []
    at = 0
    while at < len(entries):
        end = at
        while end < len(entries) and (entries[end].date, entries[end].words) == (
                entries[at].date, entries[at].words):
            end += 1
        sheet = "".join(
            f'<article class="newsEntry" id="{escape(one.anchor, quote=True)}">'
            + (f"<h4>{escape(one.heading)}</h4>" if one.heading else "")
            + blocks_html(one.blocks) + "</article>"
            for one in entries[at:end]
        )
        days.append(
            f'<section class="newsDay"><h3>{escape(entries[at].words)}</h3>'
            f'<div class="panel newsSheet">{sheet}</div></section>'
        )
        at = end
    return (
        "<h1>What's new</h1>\n"
        f'<div class="muted small">{escape(LEAD)}</div>\n'
        f"{feeds_html()}\n" + "\n".join(days)
    )


def feeds_html() -> str:
    """The two feeds as links under the lead: one for new games, one for this page."""
    return (
        '<p class="small newsFeeds">Follow it: '
        f'<a href="{front.NEWS_FEED}" type="{front.FEED_TYPE}">this page as a feed</a> · '
        f'<a href="{front.GAMES_FEED}" type="{front.FEED_TYPE}">new games as a feed</a>.</p>'
    )
