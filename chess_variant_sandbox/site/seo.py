"""The doors for crawlers: `robots.txt`, `sitemap.xml`, and the site's own address (G4).

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
  draws by).
- **The site's own address**, for a canonical and a sitemap's `<loc>`: the
  `CHESS_VARIANT_SANDBOX_SITE_URL` the box's unit reads (`deploy/deploy.sh` writes it).
  Where it is unset (a development machine), a canonical is the path alone, and the two
  files, which must name whole addresses, take the address the request was made to.

Nothing here reads a cookie: both files are the same bytes for everybody.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from html import escape

from chess_variant_sandbox.site import asking, mail, terms
from chess_variant_sandbox.web.page import front

ROBOTS_PATH = "/robots.txt"
SITEMAP_PATH = "/sitemap.xml"

ROBOTS_TYPE = "text/plain; charset=utf-8"
SITEMAP_TYPE = "application/xml; charset=utf-8"

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
    "/badges",
    terms.PATH,
    "/privacy",
    "/credits",
    asking.PATH,
)
"""The site's fixed public pages, in the sitemap ahead of the games. `/rules`, the rule
language as a page, joins them with G8."""

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


def sitemap(base: str, games: Sequence[front.Listed]) -> str:
    """`sitemap.xml`: the fixed pages, then each listed game's own address with the day its
    page last changed, when that is known (the sitemaps.org protocol, 0.9)."""
    rows = [f"<url><loc>{escape(base + path)}</loc></url>" for path in PAGES]
    for game in games:
        day = f"<lastmod>{escape(game.day)}</lastmod>" if game.day else ""
        rows.append(f"<url><loc>{escape(base + front.game_path(game.hash))}</loc>{day}</url>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(rows)
        + "\n</urlset>\n"
    )
