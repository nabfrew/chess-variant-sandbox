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
spelling of a description, a canonical and a `noindex`, for these, a game's own address
(`game.py`) and the site's short pages (`site/pages.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import page
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


def head_lines(description: str, canonical: str | None = None, index: bool = True) -> str:
    """The head's lines beyond the title: a description, a canonical address when the page
    has one, and `noindex` when it is not for a search engine's index."""
    lines = [f'<meta name="description" content="{escape(description, quote=True)}">']
    if canonical is not None:
        lines.append(f'<link rel="canonical" href="{escape(canonical, quote=True)}">')
    if not index:
        lines.append('<meta name="robots" content="noindex">')
    return "\n".join(lines) + "\n"


def home(base: str = "") -> str:
    """`/`: the single page with the landing's headline and first paragraph drawn in."""
    return page(HOME_TITLE, head_lines(HOME_DESCRIPTION, base + "/"), landing.drawn_intro())


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
    return page(GAMES_TITLE, head_lines(GAMES_DESCRIPTION, base + GAMES_PATH),
                games_view(listed))
