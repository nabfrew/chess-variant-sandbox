"""The doors for crawlers: `robots.txt`, `sitemap.xml`, `llms.txt`, and the site's own
address (G4, G8).

G4 of `docs/plan-growth.md`, after the user's answer to G-Q1 (2026-10-19, A): search engines
are let in. Three things are decided here, each in one place, so they cannot disagree:

- **What may be crawled.** `DISALLOWED` is every address a crawler is asked to leave alone:
  the API, the doors to an account, everything a member reads about other members, the
  suggestion box, the admin page, the token links (a challenge, a move email's *switch them
  off*) and the browser's measuring runtime. `robots` writes it out, and `pages._shell`
  reads the same table to put `noindex` on a page under one of them, so a page robots keeps
  out also says so itself.
- **What is listed.** `sitemap` writes the site's fixed public pages (`PAGES`) and every
  game a search engine may index: exactly the games whose own address carries no `noindex`
  (`game_page.listed_pages`, which decides by `game_page.indexable`, the rule the game page
  draws by), and those games' proven positions, the list and each position offered (G10,
  `SiteApp.proven_pages`).
- **What an assistant is pointed to** (G8, `llms.txt`, after llmstxt.org's proposal): what
  the site is, the rule language as Markdown and as a page, and the games list with every
  game `sitemap` lists. Whether any assistant reads it is unknown; the plan lists it as a
  guess (`docs/plan-growth.md` §2.4).
- **The site's own address**, for a canonical and a sitemap's `<loc>`: the
  `CHESS_VARIANT_SANDBOX_SITE_URL` the box's unit reads (`deploy/deploy.sh` writes it).
  Where it is unset (a development machine), a canonical is the path alone, and the two
  files, which must name whole addresses, take the address the request was made to.

Nothing here reads a cookie: both files are the same bytes for everybody.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from chess_variant_sandbox.site import asking, mail, terms
from chess_variant_sandbox.spec import reference
from chess_variant_sandbox.web.page import front
from chess_variant_sandbox.web.page.shell import NAME

ROBOTS_PATH = "/robots.txt"
SITEMAP_PATH = "/sitemap.xml"
LLMS_PATH = "/llms.txt"

ROBOTS_TYPE = "text/plain; charset=utf-8"
SITEMAP_TYPE = "application/xml; charset=utf-8"
LLMS_TYPE = "text/plain; charset=utf-8"

# Both files change only with a deploy or a game listed or taken off, so a crawler may keep
# them an hour; the sitemap's own list is reused for `game_page.LISTED_SECONDS` besides.
CACHE = "public, max-age=3600"

DISALLOWED: tuple[str, ...] = (
    "/api/",
    "/signin",
    "/join",
    "/admin",
    "/profile",
    # `/players` and every member's page under it, `/players/<handle>`.
    "/players",
    "/suggestions",
    # A move email's one-click *switch them off*, `/mail/off/<token>`.
    "/mail/",
    # A challenge link, `/c/<token>`: somebody's own invitation to a game.
    "/c/",
    # The browser's measuring runtime: Pyodide and the package's wheel, nothing to read.
    "/browser/",
)
"""Every address prefix a crawler is asked not to fetch. Robots matches by prefix, so
`/players` keeps out `/players/<handle>` too."""

PAGES: tuple[str, ...] = (
    "/",
    front.GAMES_PATH,
    reference.PAGE,
    front.NEWS_PATH,
    "/badges",
    terms.PATH,
    "/privacy",
    "/credits",
    asking.PATH,
)
"""The site's fixed public pages, in the sitemap ahead of the games; `/rules`, the rule
language as a page, since G8, and `/whats-new`, the changelog as a page, since G11."""

_HOST = re.compile(r"^[A-Za-z0-9.-]+(:\d{1,5})?$")


def site_base() -> str:
    """The site's address with no trailing slash, or empty where none is configured (a
    development machine): then a canonical is the path alone."""
    try:
        return mail.site_url()
    except mail.MailError:
        return ""


def base_for(host: str | None) -> str:
    """A whole address to write a file's links on: the site's own, else the one the
    request was made to (a development machine), when its `Host` is a plain host name."""
    base = site_base()
    if base:
        return base
    asked = (host or "").strip()
    return f"http://{asked}" if _HOST.match(asked) else ""


def disallowed(path: str) -> bool:
    """Whether `robots.txt` asks a crawler to leave this address alone."""
    return path.startswith(DISALLOWED)


def robots(base: str) -> str:
    """`robots.txt`: everything is open but `DISALLOWED`, and the sitemap is named."""
    lines = ["User-agent: *", *(f"Disallow: {path}" for path in DISALLOWED), "Allow: /"]
    if base:
        lines += ["", f"Sitemap: {base}{SITEMAP_PATH}"]
    return "\n".join(lines) + "\n"


def sitemap(base: str, games: Sequence[front.Listed],
            positions: Sequence[tuple[str, str]] = ()) -> str:
    """`sitemap.xml`: the fixed pages, then each listed game's own address with the day its
    page last changed, when that is known (the sitemaps.org protocol, 0.9), then the proven
    positions' addresses of those games (G10: `positions`, each a path and the day its
    record was proved)."""
    rows = [f"<url><loc>{escape(base + path)}</loc></url>" for path in PAGES]
    for game in games:
        day = f"<lastmod>{escape(game.day)}</lastmod>" if game.day else ""
        rows.append(f"<url><loc>{escape(base + front.game_path(game.hash))}</loc>{day}</url>")
    for path, made in positions:
        day = f"<lastmod>{escape(made)}</lastmod>" if made else ""
        rows.append(f"<url><loc>{escape(base + path)}</loc>{day}</url>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(rows)
        + "\n</urlset>\n"
    )


LLMS_SUMMARY = (
    "A site for designing chess-like strategy games and playing them in the browser. Every "
    "game is one JSON file in a declarative rule language; a validator checks a file and "
    "names every mistake, a game engine plays it, and computer players measure it."
)
"""`llms.txt`'s one-line summary, its blockquote."""

LLMS_HOW = (
    "To write a new game, read the rule language, write the game as a JSON file, and load it "
    "in the designer ({designer}) to check it, see it on the board and play it against the "
    "computer. A score on a game is a preliminary estimate from computer play, and each one "
    "says what it rests on."
)
"""`llms.txt`'s paragraph on what to do with the rule language, with the designer's address."""

DESIGNER = "/#/design"
"""The designer, a view of the single page (`shell.SECTIONS`' *Build*)."""


LLMS_STORIES = (
    "The front page tells one of these stories, chosen at random each time it is opened: an "
    "idea written as a rule, what went wrong with the game it made, the one change, and what "
    "that change did. In the page's order:"
)


def story_lines(base: str, stories: Sequence[Mapping[str, Any]]) -> list[str]:
    """`llms.txt`'s paragraph on the front page's stories (LP6, `docs/plan-landing.md` §4.1):
    each holding story's pair in the pool's order, a library game by its name at its own
    address, a constructed one said to be an example made for the page, with no address.
    Nothing in it is random: the same stories hold, the same lines."""
    if not stories:
        return []

    def named(side: Mapping[str, Any], example: bool) -> str:
        name = _bracketed(str(side.get("name") or "An unnamed game"))
        return name if example else f"[{name}]({base}{front.game_path(str(side['hash']))})"

    rows = ["## The front page's stories", "", LLMS_STORIES, ""]
    for story in stories:
        example = bool(story.get("example"))
        failed, fixed = story["failed"], story["fixed"]
        if example:
            rows.append(f"- {story['title']} ({story['family']}): {named(failed, True)} and "
                        f"{named(fixed, True)}, an example made for the page, which has no "
                        "address of its own.")
        else:
            rows.append(f"- {story['title']} ({story['family']}): {named(failed, False)} was "
                        f"ruled out; one change made {named(fixed, False)}, which stays in.")
    return [*rows, ""]


def llms(base: str, games: Sequence[front.Listed],
         stories: Sequence[Mapping[str, Any]] = ()) -> str:
    """`llms.txt`: what the site is, where the rule language is (as Markdown, the file to
    give an assistant, and as a page), the front page's stories (`story_lines`), and the
    games list, with every game `sitemap` lists by its name at its own address."""
    rows = [
        f"# {NAME}",
        "",
        f"> {LLMS_SUMMARY}",
        "",
        LLMS_HOW.format(designer=base + DESIGNER),
        "",
        "## The rule language",
        "",
        f"- [The rule language, as Markdown]({base}{reference.PUBLISHED}): the whole language "
        "a game is written in, every key with its type, default and meaning, how a wrong "
        "file is refused, and two worked examples. One file, to give an assistant to write "
        "a game from.",
        f"- [The rule language, as a page]({base}{reference.PAGE}): the same text as a web "
        "page, with a link to every section.",
        "",
        *story_lines(base, stories),
        "## The games",
        "",
        f"- [The games]({base}{front.GAMES_PATH}): every game, each playable in the browser "
        "against the computer.",
        f"- [New games, as an Atom feed]({base}{front.GAMES_FEED}): the newest games to "
        "join, library and published.",
        f"- [What's new]({base}{front.NEWS_PATH}): every change to the site in plain words, "
        f"also [as an Atom feed]({base}{front.NEWS_FEED}).",
    ]
    rows += [f"- [{_bracketed(game.name or 'An unnamed game')}]({base}"
             f"{front.game_path(game.hash)})" for game in games]
    return "\n".join(rows) + "\n"


def _bracketed(name: str) -> str:
    """A game's name as a Markdown link's words, on one line: a bracket in it cannot end
    the link, and a line break cannot end the row."""
    said = " ".join(name.split())
    return said.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
