"""The two front doors, `/` and `/game`, as the site draws them, and the head lines every
page drawn on the server carries (G4 of `docs/plan-growth.md`).

Both addresses answer the single page, which the router fills; to a reader without script
and to a crawler they were one untitled page, the same bytes at both, whose words were all
inside a `<template>`. The site now answers each with its own head (a title, a description
and a canonical address) and its own view:

- **`/`** carries the landing's headline and first paragraph in `#view`
  (`landing.drawn_intro`), the start of the hero's own markup, so the template the router
  clones over it lands the headline where it stood: a person sees the landing unchanged.
- **`/game`** carries every listed game as a link to its own address inside `<noscript>`,
  read by a reader without script and by a crawler, and never drawn where the router runs.

Nothing here reads a store, a cookie or a file: the site hands over its address and the
list (`site/game_page.py` decides who is listed), and this draws. `head_lines` is the one
spelling of a description, a canonical, a `noindex` and the link preview's tags (G5), for
these, a game's own address (`game.py`) and the site's short pages (`site/pages.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import page, picture
from chess_variant_sandbox.web.page.shell import NAME

GAMES_PATH = "/game"
"""The games, `web/server.py`'s `GAMES_PATH`; a game's own address is under it."""

HOME_TITLE = NAME
"""The front page's title: the site's name alone, which `core.js`'s `setTitle('')` writes."""

GAMES_TITLE = f"Games · {NAME}"
"""The games' title, as `core.js`'s `viewTitle` names the view (*Games*) and `setTitle` writes it."""

HOME_DESCRIPTION = f"{landing.HEADLINE} {landing.LEAD}"
"""The search result's line for the front page: its own headline and first paragraph."""

GAMES_DESCRIPTION = (
    "Every chess-like game here, each playable in the browser against the computer: search "
    "them, filter by the kind of game, and sort by any ingredient of the score."
)

GAMES_HEADING = "The games"


def game_path(digest: str) -> str:
    """A game's own address (`game.py`'s `path`)."""
    return f"{GAMES_PATH}/{digest}"


@dataclass(frozen=True)
class Listed:
    """One game a search engine may index: its hash, its name and the day its page last
    changed (`YYYY-MM-DD`, or empty when nothing says)."""

    hash: str
    name: str
    day: str = ""


PREVIEW_TYPE = "website"
"""Every page's `og:type`: the site is one website, and none of its pages is an article."""

PREVIEW_CARD = "summary_large_image"
"""Every page's `twitter:card`: the picture across the top, the words under it."""


def picture_address(base: str) -> str:
    """The site's link-preview picture (`picture.py`) on the site's address `base`."""
    return base + picture.site_path()


def _whole(address: str | None) -> bool:
    """Whether an address is a whole one, which a preview's address and picture must be: a
    development machine with no site address has only paths, and leaves them out."""
    return bool(address) and str(address).startswith(("https://", "http://"))


def preview_lines(title: str, description: str, canonical: str | None = None,
                  image: str = "", image_alt: str = picture.ALT) -> str:
    """The link preview's tags (G5 of `docs/plan-growth.md`, §2.2): what a chat program, a
    post or a search engine's card shows when somebody pastes the address. Open Graph's
    title, description, type, site name, address and picture, and Twitter's card with its
    own title, description and picture. The title leaves off the site's name, which
    `og:site_name` carries. The address and the picture are whole addresses or absent.
    The picture's size and type are the site picture's (`picture.WIDTH`, `HEIGHT`, `TYPE`),
    which a game's own picture (G6, `game_picture.py`) keeps; `image_alt` says what it shows."""
    said = title.removesuffix(f" · {NAME}") or NAME
    rows: list[tuple[str, str, str]] = [
        ("property", "og:site_name", NAME),
        ("property", "og:type", PREVIEW_TYPE),
        ("property", "og:title", said),
        ("property", "og:description", description),
    ]
    if canonical is not None and _whole(canonical):
        rows.append(("property", "og:url", canonical))
    if _whole(image):
        rows += [
            ("property", "og:image", image),
            ("property", "og:image:type", picture.TYPE),
            ("property", "og:image:width", str(picture.WIDTH)),
            ("property", "og:image:height", str(picture.HEIGHT)),
            ("property", "og:image:alt", image_alt),
        ]
    rows += [
        ("name", "twitter:card", PREVIEW_CARD),
        ("name", "twitter:title", said),
        ("name", "twitter:description", description),
    ]
    if _whole(image):
        rows += [("name", "twitter:image", image), ("name", "twitter:image:alt", image_alt)]
    return "".join(
        f'<meta {kind}="{key}" content="{escape(value, quote=True)}">\n'
        for kind, key, value in rows
    )


def head_lines(title: str, description: str, canonical: str | None = None,
               index: bool = True, image: str = "", image_alt: str = picture.ALT) -> str:
    """The head's lines beyond the title: a description, a canonical address when the page
    has one, `noindex` when it is not for a search engine's index, and the link preview's
    tags (`preview_lines`) for the page's `title` and its picture `image`, which shows
    `image_alt`."""
    lines = [f'<meta name="description" content="{escape(description, quote=True)}">']
    if canonical is not None:
        lines.append(f'<link rel="canonical" href="{escape(canonical, quote=True)}">')
    if not index:
        lines.append('<meta name="robots" content="noindex">')
    return "\n".join(lines) + "\n" + preview_lines(title, description, canonical, image,
                                                    image_alt)


def home(base: str = "") -> str:
    """`/`: the single page with the landing's headline and first paragraph drawn in."""
    return page(
        HOME_TITLE,
        head_lines(HOME_TITLE, HOME_DESCRIPTION, base + "/", image=picture_address(base)),
        landing.drawn_intro(),
    )


def games_view(listed: Sequence[Listed]) -> str:
    """What `/game` says without script: every listed game, by name, at its own address."""
    items = "".join(
        f'<li><a href="{escape(game_path(one.hash), quote=True)}">'
        f"{escape(one.name or 'An unnamed game')}</a></li>"
        for one in listed
    )
    return (
        '<noscript><section class="drawnGames">'
        f'<h2 style="margin:10px 0 0">{escape(GAMES_HEADING)}</h2>'
        f"<p>{escape(GAMES_DESCRIPTION)}</p>"
        + (f'<ul class="drawnList">{items}</ul>' if items else "")
        + "<p>Choosing, sorting and playing them needs JavaScript switched on.</p>"
        "</section></noscript>"
    )


def games(base: str, listed: Sequence[Listed]) -> str:
    """`/game`: the single page with the listed games drawn in for a reader without script."""
    return page(
        GAMES_TITLE,
        head_lines(GAMES_TITLE, GAMES_DESCRIPTION, base + GAMES_PATH,
                   image=picture_address(base)),
        games_view(listed),
    )
