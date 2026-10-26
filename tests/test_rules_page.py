"""The rule language as a page, `/rules` (G8 of `docs/plan-growth.md`, §2.4).

What a reader, a crawler and an assistant get, read off the served bytes of the local
application (the site's own drawing is `test_site_doors.py`'s, over a store):

- **the same text** as `/rules.md`: every word of the Markdown file, in order, and nothing
  of its syntax left over, so the two cannot say different things;
- **every heading an anchor**: each heading of the committed `RULES.md` an element of its
  level with the id GitHub gives it, so a link to a section reads the same on both, and
  every link on the page to a section lands on one;
- **the head**: a title and a description of its own that fit a search result;
- **`/rules.md` unchanged**: the same bytes as the committed `RULES.md`, the same type and
  cache, as before the page existed;
- **the ways in**: the landing page and the designer link to it, and the designer still to
  the Markdown file for an AI.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from test_shell import _get

from chess_variant_sandbox.spec import reference
from chess_variant_sandbox.web import designing, landing
from chess_variant_sandbox.web import server as web_server
from chess_variant_sandbox.web.page import assets, rules_page, shell
from chess_variant_sandbox.web.page import game as drawing

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "RULES.md"
DESIGN_JS = ROOT / "chess_variant_sandbox" / "web" / "static" / "page" / "design.js"

_BLOCKS = {"h1", "h2", "h3", "h4", "p", "li", "ul", "ol", "pre", "article", "header", "nav",
           "div", "section", "br"}


class _Words(HTMLParser):
    """The words of a piece of HTML as a reader reads them, the heading marks (`a.here`)
    left out, and a break at every block so two blocks' words never run together."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.said: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and ("class", "here") in attrs:
            self.skipping += 1
        if tag in _BLOCKS:
            self.said.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.skipping:
            self.skipping -= 1
        if tag in _BLOCKS:
            self.said.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.said.append(data)


def words_of_html(html: str) -> list[str]:
    reader = _Words()
    reader.feed(html)
    return "".join(reader.said).split()


def words_of_markdown(text: str) -> list[str]:
    """The Markdown file's words, its syntax taken off by a reading of its own (not the
    page's): the comment and the fences dropped, a heading's hashes and a list item's marker
    taken off the line, and every backtick and asterisk removed."""
    kept = []
    for line in text.splitlines():
        bare = line.strip()
        if bare.startswith("<!--") or bare.startswith("```"):
            continue
        # A heading's hashes, or else a list item's marker.
        bare = bare.lstrip("#") if bare.startswith("#") else re.sub(r"^(-|\d+\.)\s+", "", bare)
        kept.append(bare.replace("`", "").replace("*", ""))
    return " ".join(kept).split()


def github_slug(words: str, taken: dict[str, int]) -> str:
    """GitHub's anchor for a heading, as its Markdown renderer makes one."""
    base = re.sub(r"[^\w\- ]", "", words.strip().lower()).replace(" ", "-")
    seen = taken.get(base, 0)
    taken[base] = seen + 1
    return base if not seen else f"{base}-{seen}"


def headings_of(text: str) -> list[tuple[int, str]]:
    """Every heading of a Markdown file outside its code blocks: (level, words)."""
    found = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
        elif not fenced and (heading := re.match(r"^(#{1,6}) +(.+?)\s*$", line)):
            found.append((len(heading.group(1)), heading.group(2).replace("`", "")))
    return found


def _served() -> tuple[tuple[int, dict[str, str], bytes], tuple[int, dict[str, str], bytes]]:
    page, markdown = _get(rules_page.PATH, reference.PUBLISHED)
    return page, markdown


class TestTheSameText:
    def test_the_page_says_every_word_of_the_markdown_file_in_order(self) -> None:
        """The page's title and reference against `/rules.md`'s own bytes: word for word."""
        (status, _, body), (_, _, markdown) = _served()
        assert status == 200
        html = body.decode("utf-8")
        title = re.search(r'<header class="rulesTop"><h1 [^>]*>.*?</h1>', html, re.S)
        article = re.search(r'<article class="rulesDoc">(.*?)</article>', html, re.S)
        assert title is not None and article is not None
        drawn = words_of_html(title.group(0)) + words_of_html(article.group(1))
        written = words_of_markdown(markdown.decode("utf-8"))
        assert len(drawn) == len(written)
        for index, (seen, wanted) in enumerate(zip(drawn, written, strict=True)):
            assert seen == wanted, (index, drawn[max(0, index - 8):index + 8])

    def test_no_markdown_is_left_in_the_words_outside_code(self) -> None:
        """An emphasis, a code span or a list the page did not read would show its marks."""
        html, _ = rules_page.render(reference.render_reference())
        prose = re.sub(r'<(pre|code)\b.*?</\1>|<a class="here"[^>]*>#</a>', " ", html,
                       flags=re.S)
        said = unescape(re.sub(r"<[^>]+>", " ", prose))
        assert "`" not in said and "*" not in said
        # A line starting as a heading or a list item would (a worked example's heading
        # starts with its number, which is its words).
        body = re.sub(r"<h(\d)\b.*?</h\1>", " ", prose, flags=re.S)
        assert not re.search(r"^\s*(#{1,6}|-|\d+\.) ", re.sub(r"<[^>]+>", " ", body), re.M)

    def test_code_is_drawn_as_written(self) -> None:
        """The worked examples' JSON and the refusals' lines, byte for byte in their blocks."""
        text = reference.render_reference()
        html, _ = rules_page.render(text)
        blocks = [unescape(found) for found in
                  re.findall(r'<pre class="code[^"]*"><code>(.*?)</code></pre>', html, re.S)]
        # A block inside a list item is indented with the item; its lines are not.
        written = [re.sub(f"(?m)^{indent}", "", found) for indent, found in
                   re.findall(r"(?m)^( *)```\w*\n(.*?)\n\1```", text, re.S)]
        assert blocks == written
        assert sum(1 for block in blocks if block.lstrip().startswith("{")) == 2

    def test_a_form_the_reader_does_not_know_fails_here(self) -> None:
        """A table, say, which the file does not use today: its pipes would reach the
        words, and the comparison above would no longer hold."""
        html, _ = rules_page.render("| a | b |\n|---|---|\n| 1 | 2 |\n")
        assert "|" in unescape(re.sub(r"<[^>]+>", " ", html))


class TestEveryHeadingIsAnAnchor:
    def test_each_heading_of_rules_md_is_an_element_with_githubs_id(self) -> None:
        (_, _, body), _ = _served()
        html = body.decode("utf-8")
        taken: dict[str, int] = {}
        wanted = headings_of(RULES.read_text(encoding="utf-8"))
        assert len(wanted) > 40
        for level, words in wanted:
            ident = github_slug(words, taken)
            found = re.search(rf'<h{level} id="{re.escape(ident)}">(.*?)</h{level}>', html, re.S)
            assert found is not None, (level, words, ident)
            assert " ".join(words_of_html(found.group(0))) == " ".join(words.split()), words
            assert f'href="#{ident}"' in html, ident
        ids = re.findall(r'\bid="([^"]+)"', html)
        assert len(ids) == len(set(ids))

    def test_the_ids_read_as_githubs(self) -> None:
        ids = {heading.text: heading.id for heading in rules_page.drawn()[1]}
        assert ids["A valid game is not yet a good one"] == "a-valid-game-is-not-yet-a-good-one"
        assert ids["rules.win: how the game is won"] == "ruleswin-how-the-game-is-won"
        assert ids["pieces[].moves: one entry per way of moving"] == (
            "piecesmoves-one-entry-per-way-of-moving")

    def test_every_link_to_a_section_lands_on_one(self) -> None:
        (_, _, body), _ = _served()
        html = body.decode("utf-8")
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        targets = re.findall(r'href="#([^"]*)"', html)
        assert targets and set(targets) <= ids
        # A rule key named in the text links to the key's own section.
        assert '<a href="#ruleswin-how-the-game-is-won"><code>rules.win</code></a>' in html

    def test_the_top_points_a_newcomer_to_the_three_sections_to_start_from(self) -> None:
        (_, _, body), _ = _served()
        top = re.search(r'<header class="rulesTop">(.*?)</header>', body.decode("utf-8"), re.S)
        assert top is not None
        for name, _why in rules_page.START:
            assert f">{name}</a>" in top.group(1), name
        assert f'href="{reference.PUBLISHED}"' in top.group(1)


class TestTheHead:
    def test_a_title_and_a_description_of_its_own(self) -> None:
        (status, headers, body), _ = _served()
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Content-Security-Policy"] == web_server.CONTENT_SECURITY_POLICY
        html = body.decode("utf-8")
        assert re.search(r"<title>(.*?)</title>", html).group(1) == rules_page.TITLE  # type: ignore[union-attr]
        found = re.search(r'<meta name="description" content="([^"]*)">', html)
        assert found is not None and unescape(found.group(1)) == rules_page.DESCRIPTION
        assert len(rules_page.DESCRIPTION) <= drawing.DESCRIPTION_LIMIT
        assert '<meta name="robots"' not in html
        # No script of its own: the shell's alone.
        assert re.findall(r"<script[^>]*>", html) == [
            f'<script src="{assets.name_of(shell.SHELL_SCRIPT_ROUTE)}">']


class TestTheMarkdownFileIsUnchanged:
    def test_rules_md_is_the_committed_file_byte_for_byte(self) -> None:
        _, (status, headers, body) = _served()
        assert status == 200
        assert body == RULES.read_bytes()
        assert headers["Content-Type"] == "text/markdown; charset=utf-8"
        assert headers["Cache-Control"] == "public, max-age=300"


class TestTheWaysIn:
    def test_the_landing_page_links_it(self) -> None:
        # LP4: the wall of words' lead links the rule language, and every word its section.
        drawn = landing.html_page()
        assert f'href="{rules_page.PATH}"' in drawn
        assert drawn.count(f'href="{rules_page.PATH}#') == sum(len(b.words) for b in landing.BLOCKS)

    def test_the_designer_links_the_page_and_the_file(self) -> None:
        kit = designing.rule_file_kit()
        assert (kit["rulesPage"], kit["rulesUrl"]) == (reference.PAGE, reference.PUBLISHED)
        assert reference.PAGE == rules_page.PATH == "/rules"
        script = DESIGN_JS.read_text(encoding="utf-8")
        assert "file.rulesPage" in script and "file.rulesUrl" in script
