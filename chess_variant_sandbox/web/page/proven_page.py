"""A game's proven positions, and each position, drawn on the server (G10 of
`docs/plan-growth.md`, §2.5).

Q993b gave the positions an address of their own, `/game/<hash>/proven` and
`/game/<hash>/proven/<position>`, which the server answered with the game's page: to a
crawler and a link unfurler every position was the game. Now each answers with its own head
and its own view, before any script runs:

- **a position** carries a title (*‹game›, position 3: find the winning move*), a
  description that says whose move it is and what to find, never the answer, a canonical
  address with the position's name in it (a digest of the position, `puzzles.name`, so the
  address outlives a re-proof), and a picture of the position itself (`game_picture.svg`,
  turned for the second player as the page turns the board);
- **the list** carries its own title and description and, in `#view`, every position
  offered as a link to its own address, grouped by the length of the win, so a crawler
  that reaches the list reaches each position.

The router replaces `#view` once it draws, so a person with script sees the positions' page
they always saw. Nothing here reads a store or a file and nothing here works out a move: the
site hands over what is drawn (`Drawn`, made by `site/proven.py` from the record and the
core's replay), and this draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from chess_variant_sandbox.web.page import front, page
from chess_variant_sandbox.web.page import picture as site_picture
from chess_variant_sandbox.web.page.shell import NAME


@dataclass(frozen=True)
class Row:
    """One position on the list, as a link: its name, its place and its words."""

    id: str
    number: int
    said: str


@dataclass(frozen=True)
class Group:
    """The positions that win in the same number of turns, under their heading."""

    heading: str
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Drawn:
    """Everything a proven positions' address draws, and nothing it would look up.

    `title` is the page's own title without the site's name, `path` its canonical path (the
    list's for a position no longer offered), `picture` the path of its link-preview picture
    (empty: the site's stands in) and `picture_alt` what that shows. `back` and `back_said`
    are the crumb's link and words, `said` the paragraphs under the heading, `groups` the
    list's links (a position draws none), `basis` how the positions were proved, and `game`
    the line that links the game's own page."""

    hash: str
    title: str
    description: str
    path: str
    indexable: bool
    heading: str
    back: str
    back_said: str
    said: tuple[str, ...] = ()
    groups: tuple[Group, ...] = ()
    basis: str = ""
    game: str = ""
    base: str = ""
    picture: str = ""
    picture_alt: str = site_picture.ALT


NEEDS_SCRIPT = "Finding the move and playing a position out need JavaScript switched on."


def full_title(drawn: Drawn) -> str:
    """The tab's and the search result's title; `proven.js` writes the same."""
    return f"{drawn.title} · {NAME}"


def canonical(drawn: Drawn) -> str:
    return drawn.base + drawn.path


def picture(drawn: Drawn) -> str:
    """The page's link-preview picture as a whole address: its own where one is made, the
    site's until then, so the tag never names a picture that is not there."""
    if drawn.picture:
        return drawn.base + drawn.picture
    return front.picture_address(drawn.base)


def picture_alt(drawn: Drawn) -> str:
    return drawn.picture_alt if drawn.picture else site_picture.ALT


def head(drawn: Drawn) -> str:
    """The lines the head carries beyond the title (`front.head_lines`)."""
    return front.head_lines(full_title(drawn), drawn.description, canonical(drawn),
                            drawn.indexable, image=picture(drawn),
                            image_alt=picture_alt(drawn))


def _groups(drawn: Drawn) -> str:
    out = []
    for group in drawn.groups:
        items = "".join(
            f'<li><a href="{escape(row_path(drawn.hash, row.id), quote=True)}">'
            f"{escape(row.said)}</a></li>"
            for row in group.rows
        )
        out.append(f"<h3>{escape(group.heading)}</h3>\n"
                   f'<ol class="drawnList" start="{group.rows[0].number}">{items}</ol>')
    return "\n".join(out)


def row_path(digest: str, position: str) -> str:
    """A position's own address (`game.proven_path`, spelt here so this module reads no
    other drawer)."""
    return f"{front.game_path(digest)}/proven/{position}"


def view(drawn: Drawn) -> str:
    """The page as a reader without script reads it; the router draws over it."""
    said = "".join(f"<p>{escape(line)}</p>\n" for line in drawn.said)
    groups = _groups(drawn) if drawn.groups else ""
    return (
        '<article class="drawnGame drawnProven" id="drawnProven">\n'
        f'<a class="crumb" href="{escape(drawn.back, quote=True)}">&larr; '
        f"{escape(drawn.back_said)}</a>\n"
        f'<h2 style="margin:10px 0 0">{escape(drawn.heading)}</h2>\n'
        + said
        + (groups + "\n" if groups else "")
        + (f'<p class="muted">{escape(drawn.basis)}</p>\n' if drawn.basis else "")
        + (f'<p><a href="{escape(front.game_path(drawn.hash), quote=True)}">'
           f"{escape(drawn.game)}</a></p>\n" if drawn.game else "")
        + f'<noscript><p class="muted">{escape(NEEDS_SCRIPT)}</p></noscript>\n'
        "</article>"
    )


def draw(drawn: Drawn) -> str:
    """The whole page for a proven positions' address."""
    return page(full_title(drawn), head(drawn), view(drawn))
