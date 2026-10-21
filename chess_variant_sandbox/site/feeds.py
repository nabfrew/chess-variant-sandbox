"""The site's two Atom feeds (G11 of `docs/plan-growth.md`, route 8 of §2).

- **`/feeds/games.atom`**: the newest games a search engine may index, library and published
  alike, newest first, each at its own address. A game is *new* on the day it arrived on the
  site: the day a designer pressed *Publish* (`games.published`) for a designed game, else
  the day the loader made its row (`games.created`) for a library game. The list is
  `game_page.listed_pages`' own (the sitemap's), so a saved draft, a resting, retired, killed
  or deleted game is in no feed any more than in the sitemap, and one that leaves is gone
  from the next feed. `FEED_LIMIT` entries, the newest.
- **`/feeds/whats-new.atom`**: the changelog's entries as *What's new* shows them, from the
  same payload as the page (`web/page/news_page.py`): an entry marked *For maintainers.*
  (`changelog.MAINTAINERS`) is in no feed, because the page does not show it. An entry is
  as dated as its day heading says (midnight UTC), and its text is the page's, as markup.

Both feeds are Atom 1.0 (RFC 4287), written by hand: a feed is a few elements, and a
dependency for it would be the larger thing. **Every id is a whole address that does not
move**: a game's is its own address (the hash is its identity, and a rename moves the
name, never the address), and a changelog entry's is the page's address plus the entry's
anchor (`news_page.items`), so a reader that has shown an entry once shows it once. `updated`
is when the entry appeared and is never a re-measurement's day, so a score that moves
reorders nothing. The feed's own `updated` is its newest entry's: the same bytes until
something is new, which is what lets a reader and a cache believe a 304.

Nothing here reads a cookie: the same bytes for everybody.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from chess_variant_sandbox.web.page import front, news_page
from chess_variant_sandbox.web.page import game as drawing
from chess_variant_sandbox.web.page.shell import NAME

GAMES_PATH = front.GAMES_FEED
NEWS_PATH = front.NEWS_FEED

TYPE = f"{front.FEED_TYPE}; charset=utf-8"
"""What the feeds answer as; `application/atom+xml` is Atom's registered type (RFC 4287 §7)."""

CACHE = "public, max-age=900"
"""A reader polls; a quarter of an hour is as fresh as a feed needs, and the listing behind
the games' feed is reused for five minutes besides (`game_page.LISTED_SECONDS`)."""

FEED_LIMIT = 50
"""The newest entries a feed carries. A reader keeps what it has seen; a new reader wants
the recent past, and a feed that grew with the site would be the games list again."""

ROUTES: dict[str, str] = {
    front.NEWS_PATH: "GET /whats-new",
    front.GAMES_FEED: "GET /feeds/games.atom",
    front.NEWS_FEED: "GET /feeds/whats-new.atom",
}
"""Each address's row in `usage_hours` (the plan's measure for the feeds: fetches by hour)."""

GAMES_TITLE = front.FEEDS[0][1]
NEWS_TITLE = front.FEEDS[1][1]

GAMES_SUBTITLE = "Games that have joined the site, newest first, each playable in the browser."
NEWS_SUBTITLE = "Every change to the site in plain words, newest first."

NOWHERE = "http://localhost"
"""Where links point when a request names no usable host and the site has no address
configured: an Atom link must be whole, and nothing here can know better."""

EPOCH = "1970-01-01T00:00:00Z"
"""A feed with no entry has no newest one to be updated at."""

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# What XML 1.0 forbids in text: control characters but tab, newline and return.
_UNWRITABLE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def stamp(when: datetime) -> str:
    """A moment as RFC 3339 in UTC: `2026-10-21T09:30:00Z`."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(said: str) -> str:
    return escape(_UNWRITABLE.sub("", said))


def _element(tag: str, said: str, **attributes: str) -> str:
    given = "".join(f" {key}={quoteattr(value)}" for key, value in attributes.items())
    return f"<{tag}{given}>{_text(said)}</{tag}>"


def _link(rel: str, href: str, kind: str = "") -> str:
    typed = f" type={quoteattr(kind)}" if kind else ""
    return f"<link rel={quoteattr(rel)}{typed} href={quoteattr(href)}/>"


def _whole(base: str) -> str:
    return base or NOWHERE


def _feed(base: str, path: str, title: str, subtitle: str, alternate: str,
          entries: Sequence[tuple[str, str]]) -> str:
    """One feed document. `entries` are (updated, markup) newest first; the feed's own
    `updated` is the newest entry's."""
    updated = max((when for when, _ in entries), default=EPOCH)
    head = [
        _element("id", base + path),
        _element("title", title),
        _element("subtitle", subtitle),
        _element("updated", updated),
        _link("self", base + path, front.FEED_TYPE),
        _link("alternate", base + alternate, "text/html"),
        f"<author>{_element('name', NAME)}</author>",
        _element("generator", NAME),
    ]
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">\n'
        + "\n".join(head) + "\n" + "\n".join(markup for _, markup in entries)
        + ("\n" if entries else "") + "</feed>\n"
    )


def games(base: str, listed: Sequence[front.Listed]) -> str:
    """The new-games feed: the listed games that arrived, newest first, `FEED_LIMIT` of them.

    A game the listing has no arrival day for (a library file with no row in the store yet)
    is left out: with no day it has no news to give.
    """
    base = _whole(base)
    ordered = sorted((one for one in listed if one.arrived),
                     key=lambda one: (one.arrived, one.name.casefold(), one.hash),
                     reverse=True)[:FEED_LIMIT]
    rows = []
    for one in ordered:
        address = base + front.game_path(one.hash)
        said = drawing.paragraphs(one.about)
        name = " ".join(one.name.split()) or "An unnamed game"
        rows.append((one.arrived, "\n".join([
            "<entry>",
            _element("id", address),
            _element("title", name),
            _element("updated", one.arrived),
            _link("alternate", address, "text/html"),
            _element("category", "", term=one.kind or "library"),
            _element("summary", drawing.first_sentence(said[0], 400) if said else
                     f"{name}, a chess-like game: play it in the browser against the "
                     "computer, with no account.", type="text"),
            "</entry>",
        ])))
    return _feed(base, GAMES_PATH, GAMES_TITLE, GAMES_SUBTITLE, front.GAMES_PATH, rows)


def news(base: str, payload: Mapping[str, Any]) -> str:
    """The changelog feed: `news_page.items` of the payload, newest first.

    The page's own entries, so what the site does not show (an entry for maintainers) is not
    here; an entry under a day that is not a date has no time to be updated at and is left
    out. Each is one `html` content: the paragraphs the page draws, escaped by the same
    function.
    """
    base = _whole(base)
    rows = []
    for item in news_page.items(payload):
        if not _DAY.match(item.date):
            continue
        address = f"{base}{news_page.PATH}#{item.anchor}"
        when = f"{item.date}T00:00:00Z"
        title = " ".join(item.heading.split()) or f"Changes on {item.words}"
        rows.append((when, "\n".join([
            "<entry>",
            _element("id", address),
            _element("title", title),
            _element("updated", when),
            _link("alternate", address, "text/html"),
            _element("content", news_page.blocks_html(item.blocks), type="html"),
            "</entry>",
        ])))
    # The payload is newest day first and the file's order within a day; a stable sort on
    # the day keeps both and puts a day the file filed out of order where its date says.
    rows.sort(key=lambda row: row[0], reverse=True)
    return _feed(base, NEWS_PATH, NEWS_TITLE, NEWS_SUBTITLE, news_page.PATH, rows[:FEED_LIMIT])


__all__ = ["CACHE", "GAMES_PATH", "NEWS_PATH", "ROUTES", "TYPE", "games", "news", "stamp"]
