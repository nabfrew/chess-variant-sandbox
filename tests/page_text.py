"""The pages' text as the tests read it: every script's file in place of its tag.

The page's JavaScript was Python strings until Q513 and is plain files now, loaded by
`<script src>` (`docs/reference/page-javascript.md` §4.1). The suite reads the page as a
string -- `"..." in PAGE`, `PAGE.count(...)`, a view's constant, the Node runs that cut
the `<script>` out -- and all of that reads from here:

- `PAGE` is the served page with its run of `<script src>` tags replaced by one
  `<script>` holding the files' text in load order. Classic scripts share one global
  scope, so that is the one program the browser runs, and it is what
  `web.page.PAGE` was before the move. The served page itself is `web.page.PAGE`.
- `CORE`, `GRAPH`, ... are the views' files, by the names their Python constants had.
  `DESIGN` is `design.js` and the three files that were concatenated into it
  (`teaching.js`, `cards.js`, `measure.js`), as it was.
- `inlined(html)` does to any page what `PAGE` is, and `script_of(html)` is the program
  a page runs, whether its scripts are files or (the offline replay) inline.

Each file is read through the table its server answers from (`web.page.SCRIPTS`,
`site.pages.SCRIPTS`), so a tag naming a route nobody serves is a `KeyError` here, not a
page that silently runs less. A page names each file by its versioned name (G12,
`web/page/assets.py`), which is read here as the plain route it stands for, after checking
that the version is the file's current one; and `inlined` puts each `<link rel=stylesheet>`
back as the `<style>` it was, so `in` over `PAGE` still finds a rule.

The shell's script (`web/static/shell.js`, in every page's `<head>`) is served too but is
not part of a page's program here: it reads and writes only `<html>`'s theme, the footer's
switch and the header's menu, which no view touches, and `tests/test_shell.py` runs it on
its own. `inlined` and `script_of` leave it out, so the page-level runs are what they were.
The landing's draw (`web/static/story.js`, LP6, after it in the single page's head) is left
out the same way, and `tests/test_story_draw.py` runs it on its own: without it the page
keeps `?story=` or the first story, as a browser that did not run it would.
"""

from __future__ import annotations

import re

from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.web import page
from chess_variant_sandbox.web.page import assets, shell

# Route -> file, for every script a server answers.
SERVED = {
    **page.SCRIPTS, **site_pages.SCRIPTS, **shell.HEAD_SCRIPTS
}

_SCRIPT = re.compile(r"<script([^>]*)>(.*?)</script>", re.S)
_SRC = re.compile(r'<script src="([^"]+)"></script>\n')
_RUN = re.compile(r'(?:<script src="[^"]+"></script>\n)+')


_SHEET = re.compile(r'<link rel="stylesheet" href="([^"]+)"[^>]*>\n')


def plain(route: str) -> str:
    """The plain route a page's versioned name stands for, which must be the current one:
    a name whose version is not the file's bytes is a `KeyError`, as an unserved one is."""
    if route != assets.plain(route) and assets.answer(route) is None:
        raise KeyError(f"{route}: not the current version of {assets.plain(route)}")
    return assets.plain(route)


# A JavaScript function from markup to the same markup with each `styleClass` class (G12,
# `viz/wrapped.js`) spelt out as the declarations its rule holds, kept in `STYLE_CLASSES`:
# what a test under Node, where there is no sheet to read, checks a drawing's styles by.
STYLED = ("(html => html.replace(/sc-[0-9a-z]+/g, name => "
          "[...STYLE_CLASSES].find(([, held]) => held === name)[0]))")


def text(route: str) -> str:
    """One served script's text, by its plain or versioned route."""
    return SERVED[plain(route)].read_text(encoding="utf-8")


def inlined(html: str) -> str:
    """`html` with each run of `<script src>` tags as one `<script>` of their text, and each
    stylesheet link as the `<style>` of its text."""
    def block(run: re.Match[str]) -> str:
        routes = [route for route in _SRC.findall(run[0])
                  if plain(route) not in shell.HEAD_SCRIPTS]
        if not routes:
            return run[0]
        return "<script>\n" + "".join(text(route) for route in routes) + "</script>\n"

    def sheet(link: re.Match[str]) -> str:
        if assets.answer(link[1]) is None:
            raise KeyError(f"{link[1]}: no such stylesheet")
        return "<style>" + assets.sheet_text(link[1]) + "</style>\n"

    return _SHEET.sub(sheet, _RUN.sub(block, html))


def script_of(html: str) -> str:
    """Every script a page runs, in order, as one program: a file's text for a `src`, an
    inline body as it stands, and nothing for a JSON data tag, which never runs."""
    parts = []
    for attributes, body in _SCRIPT.findall(html):
        if "application/json" in attributes:
            continue
        found = re.search(r'src="([^"]+)"', attributes)
        if found and plain(found[1]) in shell.HEAD_SCRIPTS:
            continue
        parts.append(text(found[1]) if found else body)
    return "\n".join(parts)


def data_of(html: str) -> str:
    """The text of a page's JSON data tag, or `""` where it has none."""
    found = re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S)
    return found[1] if found else ""


def view(name: str) -> str:
    """One of the page's files, by its name without `.js`."""
    return text(f"{page.SCRIPT_PREFIX}{name}.js")


PAGE = inlined(page.PAGE)

CORE = view("core")
WRAPPED = view("wrapped")
STILLBOARD = view("stillboard")
BROWSE = view("browse")
GRAPH = view("graph")
EXPLORER = view("explorer")
UPKEEP = view("upkeep")
DETAIL = view("detail")
TEACHING = view("teaching")
CARDS = view("cards")
MEASURE = view("measure")
DESIGN = view("design") + TEACHING + CARDS + MEASURE
BOARD = view("board")
RUNS = view("runs")
WHATSNEW = view("whatsnew")
