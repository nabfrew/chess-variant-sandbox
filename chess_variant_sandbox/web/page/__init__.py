"""The single page: library, game, board, run.

Vanilla everything: one HTML document, its stylesheet a file the code writes
(``STYLESHEET``), and its JavaScript in plain files under ``web/static/page/``, one per view, loaded by ``<script src>`` in this order:

=================  ==================================================================
``core.js``        shared state, the router, ``api``, and the on-demand explanations
``wrapped.js``     a board without edges, shared with the offline replay (``viz/``)
``stillboard.js``  a still board drawn from a spec, for any view that shows one
``browse.js``      the library, and finding one game in it
``graph.js``       the same library as a map, and what joins two games on it
``explorer.js``    the same map, drawn by the vendored graph library and steered
``upkeep.js``      keeping every score current, and the tags that say whether one is
``detail.js``      one game: what it is, how it scored, what can be done with it
``design.js``      building a game, and building an opponent for it
``teaching.js``    teaching the designer a move, and where the pieces start
``cards.js``       the schema's field tree, rendered as controls
``portals.js``     the designer's portals: the tray's row, the pair's card, the preview
``measure.js``     measuring a draft in the browser (``measure.py`` says what it is)
``board.js``       drawing a position, animating a move, playing one
``runs.js``        a design run, watched while it goes
``whatsnew.js``    the changelog, day by day
``round.js``       the monthly round: its entrants, the yes and no, what became of each
``landing.js``     the landing page: what a visitor meets first, and its opening board
``boot.js``        the lines that start the page, last because they *run*
=================  ==================================================================

The stylesheet is ``style.py``. ``wrapped.js`` lives in ``viz/``, beside the payloads it
draws, because the offline replay inlines it too.

**Classic scripts, not modules.** Classic scripts share one global scope, ``let`` and
``const`` included, so the files are the one program the single ``<script>`` block they
came out of was (Q513, ``docs/reference/page-javascript.md`` §1.6). Only function
hoisting stops at a file's edge, and the few top-level lines that run while the files load
call only what their own or an earlier file declares (``tests/test_page_files.py`` loads
them one by one to hold that).

**No script is inline, and no style.** The page carries no ``<script>`` body, no
``<style>`` and no ``style=`` attribute, so a policy that refuses both inline
(``script-src 'self'``, ``style-src 'self'``) costs it nothing; a test holds every served
page to that. Each file is answered under a name with its digest in it, for a year
(``assets.py``, G12): ``SCRIPTS`` below is the whole of what the route answers, and the
page names each through ``assets.name_of``, drawn per request so it always names the
current bytes. What used to be a ``style=`` is a ``u-`` class (``utilities.py``) or, for a
value the script works out, ``styleClass`` (``viz/wrapped.js``).

The palette and the board drawing are lifted
from ``viz/html.py`` deliberately -- a replay of a tournament game and a game a person
plays should not look like two different programs -- and the piece icons arrive as data
URIs by the same route, with the spec's letter as the fallback for every invented piece
the icon library has never heard of.

Four things this page is careful about, because all four were wrong:

**It never shows an identifier where a sentence exists.** A cell reached the library as
``1-1-0-0-1-0-0`` and a component as ``gradient_persistence``; both are real names for
real things and neither says anything to a reader. Everything the interface prints now
has prose behind it, served from ``loop/grid.py`` and ``metrics/glossary.py``, which is
also why the prose is *there* and not here -- a page that wrote its own explanations
would be a second opinion about what the numbers mean.

**It never computes a rule.** The animation diffs two positions the engine produced and
plays the difference; it does not derive one. Whether a move looks like a slide or a
hop is decided by geometry, which is presentation and not legality: getting it wrong
makes a piece take a silly path across the board and cannot make an illegal move look
legal.

**The rules are where the game is.** They used to be one panel back, which is one panel
too far when the question is "wait, what does this piece do".

**A draw nobody asked for waits for the one before it.** Drawing the board replaces its
contents outright, so it destroys whatever the last move was still animating. There is
exactly one draw a person does not ask for -- the agent's reply -- and an agent answers
in a few tens of milliseconds, so it used to wipe the move the *person* had just made
before that move had been shown. Captures made it obvious, because a capture's strike
waits for the capturing piece to land: every capture the agent made played out and none
of the person's did.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path

from chess_variant_sandbox.viz.wrapped import WRAPPED_FILE
from chess_variant_sandbox.web import landing
from chess_variant_sandbox.web.page import assets, utilities
from chess_variant_sandbox.web.page.shell import (
    NAME,
    TAGLINE,
    brand,
    footer,
    head_links,
    sections,
)
from chess_variant_sandbox.web.page.style import STYLE

# Where the page's scripts are served, and where they are read from.
SCRIPT_PREFIX = "/static/page/"
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "static" / "page"

# The views in the order they load: the order they were concatenated in, kept.
_ORDER = (
    "core", "wrapped", "wood", "stillboard", "motion", "browse", "graph", "explorer", "upkeep", "detail",
    "design", "teaching", "cards", "portals", "measure", "board", "simultaneous", "runs", "whatsnew",
    "round", "proven", "landing",
    "boot",
)

# Route -> file, in load order: the whole of what `SCRIPT_PREFIX` answers, each under its
# versioned name (`assets.name_of`) for a year and under this plain one `no-store`.
SCRIPTS: dict[str, Path] = {
    SCRIPT_PREFIX + name + ".js": WRAPPED_FILE if name == "wrapped" else SCRIPT_DIR / f"{name}.js"
    for name in _ORDER
}
assets.serve_files(SCRIPTS)

# The page's stylesheet (G12): `style.STYLE` and the rules that were `style=` attributes,
# one file the code writes, named by its digest. The site's pages and the rule page link it
# first, then their own.
STYLESHEET = assets.sheet("page", utilities.VENDOR + STYLE + utilities.CSS)

# The `<link>` the page's head carries. Its id is the one Cytoscape looks for before it
# writes a `<style>` of its own (`.__________cytoscape_container{position:relative}`),
# which the policy would refuse: the rule is in `utilities.VENDOR`, so the element with
# that id is the stylesheet that holds it.
CYTOSCAPE_SHEET_ID = "__________cytoscape_stylesheet"


def stylesheet_links(*routes: str, page_sheet_id: str = "") -> str:
    """The head's `<link rel="stylesheet">` lines: the page's own first, then `routes`."""
    named = f' id="{page_sheet_id}"' if page_sheet_id else ""
    return f'<link rel="stylesheet" href="{STYLESHEET}"{named}>\n' + "".join(
        f'<link rel="stylesheet" href="{route}">\n' for route in routes
    )


def script_tags(routes: Iterable[str]) -> str:
    """One `<script src>` a route, each under its current versioned name."""
    return "".join(f'<script src="{assets.name_of(route)}"></script>\n' for route in routes)


def _head(title: str, head: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n{head}" + head_links()
        + stylesheet_links(page_sheet_id=CYTOSCAPE_SHEET_ID)
    )


# The header (plan-visual.md §3.4, V2): the mark and the name, the tagline the views write
# their one-line summary into, the sections, and the corner. The footer: the links
# `paintIdentity` writes into `#foot`, and the theme switch, which no view rewrites. Then the
# landing page (Q718, `web/landing.py`), inert until `landing.js` clones it into the view,
# and beside it (LP3) the band of shelves `/game` opens on and *How the score works*.
_TOP = f"""</head>
<body>
<div class="wrap">
  <header class="top">
    <h1 id="home" class="brand">{brand()}</h1>
    <span class="muted tagline" id="tagline">{TAGLINE}</span>
    <span class="grow"></span>
    {sections()}
    <span class="muted" id="runs"></span>
    <span class="muted small" id="who"></span>
  </header>
  <div id="view">"""

def _bottom() -> str:
    """The page's foot: the footer, and the landing (LP4: drawn from the stories the server
    last rendered, `landing.current_html`), the band of shelves and the score's view."""
    return f"""</div>
  {footer()}
</div>
<template id="landingPage">{landing.current_html()}</template>
<template id="shelvesBand">{landing.SHELVES_HTML}</template>
<template id="scoreView">{landing.SCORE_HTML}</template>
"""


def page(title: str = NAME, head: str = "", view: str = "") -> str:
    """The single page, with its own `<title>`, extra `<head>` lines and a view drawn on the
    server (G3, `docs/plan-growth.md`): a game's own address carries its name, description
    and canonical in the head and its page in `#view`, which a reader without script reads
    and the router replaces once it draws. `head` and `view` are markup the caller has
    already escaped; with neither, this is `PAGE`. Drawn per call, because the scripts'
    names are their bytes' digests and a development server's files can change under it."""
    return (_head(title, head) + _TOP + view + _bottom() + script_tags(SCRIPTS)
            + "</body>\n</html>\n")


PAGE = page()
