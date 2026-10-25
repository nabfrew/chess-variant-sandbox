"""The rule language as a web page, `/rules` (G8 of `docs/plan-growth.md`, §2.4).

`/rules.md` is the whole rule language in one Markdown file, written for a person or an AI
to write a game from (`spec/reference.py`). This draws the same text as a page: every
heading an element with an id a link can name, a map of the sections beside it, a title and
a description for a search result, and the way to the Markdown file for anybody who wants
to hand it to an AI.

**One text.** The page is drawn from `reference.render_reference()`, the very string the
two servers answer `/rules.md` with, by `render` below, which knows only the few Markdown
forms that file is written in: headings, paragraphs, lists (nested, and with a code block
inside an item), fenced code blocks, inline code, bold and italics, and the comment that
says where the file comes from. `tests/test_rules_page.py` holds the page's words to the
file's, word for word and in order, and fails on a form the file starts to use that this
does not know, so a change to the reference cannot leave the page behind.

**The ids are GitHub's.** A heading's id is the slug GitHub gives the same heading in the
public repository's `RULES.md` (lower case, punctuation dropped, spaces as hyphens, `-1`
for a repeat), so a link to a section reads the same on both.

**Two drawings of the frame, one of the page.** `body()` is the page itself. The site wears
it in its own pages' shell (`site/pages.py`'s `rules`), with the site's footer and canonical;
the local application, which has no such shell, answers `/rules` with `document()`, the
same body under the application's header. Both are constants, made once per process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from html import escape

from chess_variant_sandbox.spec import reference
from chess_variant_sandbox.web.page import assets, front, shell, stylesheet_links

PATH = reference.PAGE
"""The page's address, `reference.PAGE`; `/rules.md` stays the Markdown file."""

TITLE = f"The rule language: write a chess-like game of your own · {shell.NAME}"

DESCRIPTION = (
    "The rule language every game here is written in: the board, the pieces and their moves, "
    "the setup and the rules, for a person or an AI to write a game from."
)
"""What a search result says (under `game.DESCRIPTION_LIMIT`, never another page's)."""

START = (
    ("Trying a game", "how to write a file, check it and play it"),
    ("What the language can make", "the rule families, by what each does to play"),
    ("A valid game is not yet a good one", "what to look for before calling it a game"),
)
"""The three sections the top of the page points a newcomer to, by their headings."""


@dataclass(frozen=True)
class Heading:
    """One heading of the reference: its level (1 to 3), its words, its id, and the rule key
    it names, when it is a key's own section (`rules.win`, `pieces[].moves`)."""

    level: int
    text: str
    id: str
    key: str = ""


# ------------------------------------------------------------------ the Markdown, read

_FENCE = re.compile(r"^```([\w-]*)\s*$")
_HEADING = re.compile(r"^(#{1,6}) +(.+?)\s*$")
_ITEM = re.compile(r"^(-|\d+\.) +(.*)$")
_COMMENT = re.compile(r"^<!--.*-->\s*$")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])")
_KEYED = re.compile(r"^`([^`]+)`(?::|$)")
_HOLD = "\x00{}\x00"


def slug(words: str, taken: dict[str, int]) -> str:
    """GitHub's id for a heading with these words: lower case, everything but letters,
    digits, spaces, hyphens and underscores dropped, spaces as hyphens, and `-1`, `-2` ... on
    a repeat (counted in `taken`)."""
    base = re.sub(r"[^\w\- ]", "", words.strip().lower()).replace(" ", "-")
    seen = taken.get(base, 0)
    taken[base] = seen + 1
    return base if seen == 0 else f"{base}-{seen}"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def plain(markdown: str) -> str:
    """A line of the reference's inline Markdown as the words a reader reads."""
    return _CODE.sub(lambda found: found.group(1), markdown).replace("**", "").replace("*", "")


class _Reader:
    """One pass over the reference: the blocks as HTML, and the headings met on the way."""

    def __init__(self, keys: dict[str, str]) -> None:
        # A rule key's own section, by key: inline code naming exactly that key links there.
        self.keys = keys
        self.headings: list[Heading] = []
        self.taken: dict[str, int] = {}

    # -------------------------------------------------------------------- inline

    def inline(self, text: str, link: bool = True) -> str:
        """Code spans first, held aside so nothing inside one is read as emphasis; then the
        rest escaped and its bold and italics marked."""
        held: list[str] = []

        def hold(found: re.Match[str]) -> str:
            code = found.group(1)
            drawn = f"<code>{escape(code, quote=False)}</code>"
            target = self.keys.get(code) if link else None
            if target:
                drawn = f'<a href="#{target}">{drawn}</a>'
            held.append(drawn)
            return _HOLD.format(len(held) - 1)

        said = escape(_CODE.sub(hold, text), quote=False)
        said = _BOLD.sub(r"<strong>\1</strong>", said)
        said = _ITALIC.sub(r"<em>\1</em>", said)
        return re.sub("\x00(\\d+)\x00", lambda found: held[int(found.group(1))], said)

    # -------------------------------------------------------------------- blocks

    def blocks(self, lines: list[str]) -> list[tuple[str, str]]:
        """Each block of `lines` (already out of any list item's indent) as (kind, HTML)."""
        out: list[tuple[str, str]] = []
        at = 0
        while at < len(lines):
            line = lines[at]
            if not line.strip():
                at += 1
            elif _FENCE.match(line):
                at = self._code(lines, at, out)
            elif _HEADING.match(line):
                self._heading(line, out)
                at += 1
            elif _COMMENT.match(line):
                at += 1
            elif _ITEM.match(line):
                at = self._list(lines, at, out)
            else:
                said = []
                while at < len(lines) and lines[at].strip() and not self._starts(lines[at]):
                    said.append(lines[at].strip())
                    at += 1
                out.append(("p", self.inline("\n".join(said))))
        return out

    def _starts(self, line: str) -> bool:
        return bool(_FENCE.match(line) or _HEADING.match(line) or _ITEM.match(line)
                    or _COMMENT.match(line))

    def _code(self, lines: list[str], at: int, out: list[tuple[str, str]]) -> int:
        found = _FENCE.match(lines[at])
        language = found.group(1) if found else ""
        at += 1
        held = []
        while at < len(lines) and lines[at].strip() != "```":
            held.append(lines[at])
            at += 1
        kind = f' class="code {language}"' if language else ' class="code"'
        out.append(("pre", f"<pre{kind}><code>{escape(chr(10).join(held), quote=False)}"
                           "</code></pre>"))
        return at + 1

    def _heading(self, line: str, out: list[tuple[str, str]]) -> None:
        found = _HEADING.match(line)
        assert found is not None
        level, words = len(found.group(1)), found.group(2)
        keyed = _KEYED.match(words)
        heading = Heading(level, plain(words), slug(plain(words), self.taken),
                          keyed.group(1) if keyed else "")
        self.headings.append(heading)
        out.append((f"h{level}", (
            f'<h{level} id="{heading.id}">{self.inline(words, link=False)}'
            f'<a class="here" href="#{heading.id}" aria-label="link to this section">#</a>'
            f"</h{level}>")))

    def _list(self, lines: list[str], at: int, out: list[tuple[str, str]]) -> int:
        first = _ITEM.match(lines[at])
        assert first is not None
        ordered = first.group(1) != "-"
        items: list[str] = []
        while at < len(lines):
            found = _ITEM.match(lines[at])
            if found is None or (found.group(1) != "-") != ordered:
                break
            # The marker and its spaces: a line indented this far is still the item's.
            width = len(lines[at]) - len(found.group(2))
            body = [found.group(2)]
            at += 1
            loose = False
            while at < len(lines):
                line = lines[at]
                if line.strip():
                    if _indent(line) < width:
                        break
                    body.append(line[width:])
                    at += 1
                    continue
                ahead = at
                while ahead < len(lines) and not lines[ahead].strip():
                    ahead += 1
                if ahead == len(lines) or _indent(lines[ahead]) < width:
                    break
                body.extend([""] * (ahead - at))
                loose = True
                at = ahead
            items.append(self._item(body, loose))
            ahead = at
            while ahead < len(lines) and not lines[ahead].strip():
                ahead += 1
            following = _ITEM.match(lines[ahead]) if ahead < len(lines) else None
            if following is None or (following.group(1) != "-") != ordered:
                break
            at = ahead
        tag = "ol" if ordered else "ul"
        out.append((tag, f"<{tag}>\n" + "".join(items) + f"</{tag}>"))
        return at

    def _item(self, body: list[str], loose: bool) -> str:
        """One item: a tight one's words straight in the `<li>`, a loose one's (a blank line
        inside it, as before a code block) each in a paragraph."""
        drawn = [f"<p>{html}</p>" if kind == "p" and loose else html
                 for kind, html in self.blocks(body)]
        return "<li>" + "\n".join(drawn) + "</li>\n"


def _keys(markdown: str) -> dict[str, str]:
    """Each rule key with a section of its own, and that section's id: a first reading of
    the headings alone, so a key named before its section still links to it."""
    taken: dict[str, int] = {}
    keys: dict[str, str] = {}
    fenced = False
    for line in markdown.splitlines():
        if _FENCE.match(line.strip()):
            fenced = not fenced
            continue
        found = None if fenced else _HEADING.match(line)
        if found is None:
            continue
        words = found.group(2)
        ident = slug(plain(words), taken)
        keyed = _KEYED.match(words)
        if keyed:
            keys.setdefault(keyed.group(1), ident)
    return keys


def render(markdown: str) -> tuple[str, list[Heading]]:
    """The reference's Markdown as HTML, and its headings in order."""
    reader = _Reader(_keys(markdown))
    blocks = reader.blocks(markdown.splitlines())
    drawn = [f"<p>{html}</p>" if kind == "p" else html for kind, html in blocks]
    return "\n".join(drawn) + "\n", reader.headings


# ------------------------------------------------------------------ the page


def _map(headings: list[Heading]) -> str:
    """The sections, each a link: the second-level headings in order, and under *rules*
    its own rules by key, small, so the whole map fits beside the text on a desk."""
    rows: list[str] = []
    under: list[str] = []

    def close() -> None:
        if under:
            rows[-1] = rows[-1].removesuffix("</li>") + (
                '<ul class="rulesKeys">' + "".join(under) + "</ul></li>")
            under.clear()

    for heading in headings:
        if heading.level == 2:
            close()
            rows.append(f'<li><a href="#{heading.id}">{_said(heading)}</a></li>')
        elif heading.level == 3 and rows:
            words = escape(heading.key) if heading.key else escape(heading.text)
            inner = f"<code>{words}</code>" if heading.key else words
            under.append(f'<li><a href="#{heading.id}" title="{escape(heading.text)}">'
                         f"{inner}</a></li>")
    close()
    return (
        '<nav class="rulesMap" aria-label="the sections">'
        '<p class="rulesMapHead">The sections</p>'
        f'<ol class="rulesSections">{"".join(rows)}</ol></nav>'
    )


def _said(heading: Heading) -> str:
    """A section's name in the map: a key's section by its key, in code."""
    if heading.key:
        return f"<code>{escape(heading.key)}</code>"
    return escape(heading.text)


def _top(title: Heading, headings: list[Heading]) -> str:
    """The page's own head: the reference's title, the way to the Markdown file, and the
    three sections a newcomer starts from."""
    ids = {heading.text: heading.id for heading in headings}
    starts = "".join(
        f'<li><a href="#{ids[name]}">{escape(name)}</a>: {escape(why)}</li>'
        for name, why in START if name in ids
    )
    return (
        '<header class="rulesTop">'
        f'<h1 id="{title.id}">{escape(title.text)}'
        f'<a class="here" href="#{title.id}" aria-label="link to this section">#</a></h1>'
        '<p class="lead">The same text is one Markdown file at '
        f'<a href="{reference.PUBLISHED}"><code>{reference.PUBLISHED}</code></a>, to give to '
        "an AI and ask it for a game. New to it? Start here:</p>"
        + (f'<ul class="rulesStart">{starts}</ul>' if starts else "")
        + "</header>"
    )


@lru_cache(maxsize=1)
def drawn() -> tuple[str, tuple[Heading, ...]]:
    """The reference as the running code renders it, drawn once: the schema cannot change
    under a running server."""
    html, headings = render(reference.render_reference())
    return html, tuple(headings)


def body() -> str:
    """The page inside its frame: the head, the map and the reference."""
    html, headings = drawn()
    title = headings[0]
    # The reference's own first line is its title; the page draws it in its head instead.
    article = html.split("\n", 1)[1] if html.startswith("<h1") else html
    return (
        '<div class="rulesPage">'
        + _top(title, list(headings))
        + _map(list(headings[1:]))
        + f'<article class="rulesDoc">\n{article}</article></div>'
    )


STYLE = """
  /* The rule language as a page (G8): the reference at a reading measure, its map beside it
     on a desk and above it on a phone, every heading linkable by the mark that shows beside
     it. */
  main.rules { max-width: none; }
  main.rules .rulesPage { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0 32px; }
  main.rules .rulesTop { max-width: 46rem; }
  main.rules .rulesTop .lead { margin: 0 0 12px; }
  main.rules ul.rulesStart { margin: 0 0 18px; padding-left: 18px; }
  main.rules ul.rulesStart li { margin: 0 0 4px; }
  main.rules .rulesMap { font-size: 14px; line-height: 20px; margin: 0 0 24px; padding: 12px 16px;
              background: var(--paper) 0 0 / 180px, var(--panel);
              border: 1px solid var(--line); border-radius: var(--r-panel); }
  main.rules .rulesMap p.rulesMapHead { margin: 0 0 6px; font-weight: 600; color: var(--muted);
                             font-size: 13px; }
  main.rules .rulesMap ol { margin: 0; padding-left: 20px; }
  main.rules .rulesMap li { margin: 0 0 4px; }
  main.rules .rulesMap a { text-decoration: none; border-bottom: 1px solid var(--line); }
  main.rules .rulesMap a:hover { border-bottom-color: var(--muted); }
  main.rules ul.rulesKeys { list-style: none; padding: 0; margin: 4px 0 6px; display: flex;
                 flex-wrap: wrap; gap: 2px 10px; font-size: 13px; line-height: 18px; }
  main.rules ul.rulesKeys li { margin: 0; }
  main.rules .rulesDoc { max-width: 46rem; min-width: 0; }
  main.rules .rulesDoc code, main.rules .rulesTop code, main.rules .rulesMap code { overflow-wrap: anywhere; }
  main.rules .rulesPage h1, main.rules .rulesPage h2, main.rules .rulesPage h3 { scroll-margin-top: 12px; }
  main.rules .rulesPage h1 { font-size: 28px; line-height: 34px; margin: 0 0 8px; cursor: default; }
  main.rules .rulesDoc h2 { font-size: 20px; line-height: 26px; margin: 28px 0 8px; padding-top: 18px;
                 border-top: 1px solid var(--line); }
  main.rules .rulesDoc h3 { font-size: 16px; line-height: 22px; margin: 22px 0 6px; color: var(--ink); }
  main.rules .rulesDoc p { margin: 0 0 14px; }
  main.rules .rulesDoc ul, main.rules .rulesDoc ol { margin: 0 0 14px; padding-left: 20px; }
  main.rules .rulesDoc li { margin: 0 0 8px; }
  main.rules .rulesDoc li > ul { margin: 8px 0 0; }
  main.rules .rulesDoc li > p { margin: 0 0 8px; }
  main.rules a.here { margin-left: 8px; color: var(--muted); text-decoration: none; border: none;
           font-weight: 400; opacity: .45; }
  main.rules a.here:hover, main.rules a.here:focus-visible, main.rules :target > a.here { opacity: 1; color: var(--act); }
  main.rules .rulesDoc pre.code { margin: 0 0 14px; padding: 10px 12px; background: var(--sunk);
                       border: 1px solid var(--line); border-radius: var(--r-ctl);
                       font: 13px/1.5 var(--f-mono); }
  main.rules .rulesDoc pre.code.json { white-space: pre; overflow-x: auto; }
  main.rules .rulesDoc li > pre.code { margin: 0 0 8px; }
  @media (min-width: 1000px) {
    main.rules .rulesPage { grid-template-columns: 15rem minmax(0, 46rem); }
    main.rules .rulesTop { grid-column: 1 / -1; }
    main.rules .rulesMap { grid-column: 1; grid-row: 2; position: sticky; top: 12px; align-self: start;
                max-height: calc(100vh - 24px); overflow-y: auto; margin: 0; }
    main.rules .rulesDoc { grid-column: 2; grid-row: 2; }
    main.rules ul.rulesKeys { display: block; }
    main.rules ul.rulesKeys li { margin: 0 0 2px; }
    main.rules .rulesDoc > h2:first-child { margin-top: 0; }
  }
"""
"""The page's own rules, beside the application's stylesheet on both drawings."""

# The two files the drawings link after the application's (G12, `assets.py`): the site's
# shell links `SHEET` after its own rules, and the local application's drawing links
# `LOCAL_SHEET`, the same rules and the one its own header needs.
SHEET = assets.sheet("rules", STYLE)
LOCAL_SHEET = assets.sheet(
    "rules-local", STYLE + "  header h1 a { color: var(--ink); border-bottom: none; }\n"
)


def document() -> str:
    """The whole page as the local application answers `/rules`: the body under the
    application's header and footer, with the footer's links the application's page writes
    on the local application (`core.js`'s `paintIdentity`)."""
    links = " · ".join(
        f'<a href="{where}">{words}</a>'
        for where, words in ((front.GAMES_PATH, "games"), ("/", "about"),
                             ("/#/score", "the score"), ("/#/whats-new", "what's new"))
    )
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(TITLE)}</title>\n{front.head_lines(TITLE, DESCRIPTION)}"
        f"{shell.head_links()}{stylesheet_links(LOCAL_SHEET)}</head>\n"
        '<body>\n<div class="wrap">\n  <header class="top">\n'
        f'    <h1 class="brand"><a href="/">{shell.brand()}</a></h1>\n'
        f'    <span class="muted tagline">{shell.TAGLINE}</span>\n'
        '    <span class="grow"></span>\n'
        f"    {shell.sections('/')}\n"
        '    <span class="muted small" id="who"></span>\n  </header>\n'
        f'<main class="rules">\n{body()}\n</main>\n{shell.footer(links)}\n</div>\n'
        "</body>\n</html>\n"
    )


@lru_cache(maxsize=1)
def local_bytes() -> bytes:
    """`document()` as the local server sends it, encoded once."""
    return document().encode("utf-8")
