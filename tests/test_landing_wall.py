"""The landing's wall of words (LP2, `docs/plan-landing.md` §3.2 and §5): the palette's
families, one row each, every word a rule.

`web/landing.py`'s `BLOCKS` is the one table of the families and their words. These hold it
to the language from three sides, so a new primitive, a retired key or a renamed card fails
here rather than leaving the page wrong: `RULES.md`'s *What the language can make*
(`reference.FAMILIES`: every family has a row, or a reason), the designer (every word is a
control the designer draws, under the words `card` gives), and `/rules` (every link lands on
the heading of the section that writes the rule down). And the wall's words are a visitor's
(`site_words`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_style import _delta_e, _rgb

from chess_variant_sandbox.spec import reference, relate
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import MOVE_KINDS
from chess_variant_sandbox.web import cards, landing, stories
from chess_variant_sandbox.web.page import rules_page, style, tokens

WORDS = [(block, word) for block in landing.BLOCKS for word in block.words]
WORD_IDS = [f"{block.hue}:{word.said}" for block, word in WORDS]

DESIGNER_SCRIPTS = Path(cards.__file__).resolve().parent / "static" / "page"
DESIGNER_TEXT = "".join(
    (DESIGNER_SCRIPTS / name).read_text(encoding="utf-8")
    for name in ("design.js", "cards.js", "portals.js", "teaching.js"))

MODEL = cards.card_model()


def _wall() -> str:
    start = landing.html_page().index('id="landing-palette"')
    return landing.html_page()[start:landing.html_page().index("</section>", start)]


def _sections() -> dict[str, str]:
    """Each keyed section of the reference, by its key, with its text up to the next heading."""
    found: dict[str, str] = {}
    key, lines = "", []
    for line in reference.render_reference().splitlines():
        heading = re.match(r"^#{1,6} +`([^`]+)`", line)
        if heading or line.startswith("#"):
            if key:
                found.setdefault(key, "\n".join(lines))
            key, lines = (heading.group(1) if heading else ""), []
            continue
        lines.append(line)
    if key:
        found.setdefault(key, "\n".join(lines))
    return found


def _map_kind(key: str) -> str:
    """The kind of rule the map (`spec/relate.py`) files a wall word's key under."""
    if key.startswith("rules."):
        return relate.RULE_KINDS.get(key[len("rules."):], "")
    if key == "pieces[].captures":
        return "capture"
    return "board" if key.startswith("board.") else "pieces"


SECTIONS = _sections()


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    if node.get("fields"):
        return node["fields"]
    if node.get("item"):
        return _children(node["item"])
    return []


def _find(nodes: list[dict[str, Any]], key: str) -> dict[str, Any]:
    found = [node for node in nodes if node["key"] == key]
    assert found, f"the designer has no control {key!r} among {[n['key'] for n in nodes]}"
    return found[0]


# ------------------------------------------------------------------ RULES.md's families


class TestTheFamilies:
    def test_every_family_of_the_language_has_a_row_or_a_reason(self) -> None:
        titles = [title for title, _, _ in reference.FAMILIES]
        shown = {title for block in landing.BLOCKS for title in block.shows}
        missing = [title for title in titles
                   if title not in shown and title not in landing.NOT_ON_THE_WALL]
        assert missing == []
        assert set(landing.NOT_ON_THE_WALL) <= set(titles)
        assert not shown & set(landing.NOT_ON_THE_WALL)
        assert all(reason.strip() for reason in landing.NOT_ON_THE_WALL.values())

    def test_the_families_are_the_ones_rules_md_lists(self) -> None:
        # `RULES.md` is `render_reference()` (held equal by `tests/test_rules_reference.py`).
        text = reference.render_reference()
        for title, _, _ in reference.FAMILIES:
            assert f"- **{title}.**" in text, title

    def test_a_rows_words_are_drawn_from_the_families_it_shows(self) -> None:
        keys = {title: set(found) for title, _, found in reference.FAMILIES}
        for block in landing.BLOCKS:
            assert block.shows and set(block.shows) <= set(keys), block.family
            allowed = set().union(*(keys[title] for title in block.shows))
            for word in block.words:
                assert word.key in allowed, (block.family, word.said, word.key)

    def test_rows_and_colours_are_one_each(self) -> None:
        families = [block.family for block in landing.BLOCKS]
        assert len(set(families)) == len(families)
        assert all(family == family.lower() for family in families)
        hues = [block.hue for block in landing.BLOCKS]
        colours = [block.colour for block in landing.BLOCKS]
        assert len(set(hues)) == len(hues) and len(set(colours)) == len(colours)
        for colour in colours:
            assert colour in tokens.LIGHT and colour in tokens.DARK, colour

    def test_each_colour_class_names_its_token_in_the_stylesheet(self) -> None:
        for block in landing.BLOCKS:
            assert f".{block.hue} {{ --family: var(--{block.colour}); }}" in style.STYLE

    def test_no_word_twice_and_every_row_has_words(self) -> None:
        said = [word.said for _, word in WORDS]
        assert len(set(said)) == len(said)
        assert all(block.words for block in landing.BLOCKS)

    def test_the_stories_read_their_families_from_the_wall(self) -> None:
        assert {block.family: block.hue for block in landing.BLOCKS} == stories.FAMILIES
        assert {story.family for story in stories.STORIES} <= set(stories.FAMILIES)

    def test_a_rows_colour_is_its_kind_of_rule_on_the_map(self) -> None:
        """Choice E1 (`plan-visual.md` §3.2): a map hue means a kind of rule wherever one is
        named. A row in a kind's map colour holds a rule the map files under that kind; a
        row in no kind's colour (*by looking*, which the map counts among the pieces' rules
        beside the move row, and *chess's own*) wears a colour no kind wears, so the map's
        colours never say a second thing here."""
        kind_tokens = {f"kind-{kind}" for kind in relate.KINDS}
        for block in landing.BLOCKS:
            kinds = {_map_kind(word.key) for word in block.words}
            if block.colour in kind_tokens:
                assert block.colour[len("kind-"):] in kinds, (block.family, kinds)
                continue
            for theme in (tokens.LIGHT, {**tokens.LIGHT, **tokens.DARK}):
                worn = {theme[name].lower() for name in kind_tokens}
                assert theme[block.colour].lower() not in worn, block.family

    def test_the_rows_colours_are_told_apart(self) -> None:
        """No two rows in near-identical hues: the sight row once borrowed
        `--kind-valuation`'s lavender beside `--kind-hand`'s (12.7 apart in the dark), and
        has `--kind-sight` of its own (Q1038). *Chess's own* row is the neutral `--muted`,
        no hue, and is left out here; every kind hue keeps 20 from it and from each other
        (`tests/test_kind_hues.py`, Q1040)."""
        for theme in (tokens.LIGHT, {**tokens.LIGHT, **tokens.DARK}):
            colours = [block.colour for block in landing.BLOCKS if block.colour != "muted"]
            for at, one in enumerate(colours):
                for two in colours[at + 1:]:
                    apart = _delta_e(_rgb(theme[one]), _rgb(theme[two]))
                    assert apart >= 20, (one, two, round(apart, 1))


# ------------------------------------------------------------------ the designer


@pytest.mark.parametrize(("block", "word"), WORDS, ids=WORD_IDS)
def test_every_word_is_a_rule_the_designer_offers(block: landing.Block,
                                                  word: landing.Word) -> None:
    piece_fields = {node["key"]: node for node in MODEL["piece"]}
    if word.key.startswith("rules."):
        rule = _find(MODEL["rules"], word.key.removeprefix("rules."))
        assert word.key.removeprefix("rules.") in MODEL["ruleOrder"]  # *add a rule* offers it
        assert word.card == rule["label"]
        node = rule
        for part in word.field.split(".") if word.field else []:
            node = _find(_children(node), part)
        if word.value is not None:
            assert word.value in [option["value"] for option in node["options"]]
        return
    if word.key.startswith("pieces[]."):
        leaf = word.key.removeprefix("pieces[].")
        if leaf in piece_fields:
            assert not word.field
            assert word.card == piece_fields[leaf]["label"]
            return
        # A control the designer draws by hand: a piece's lines of sight.
        assert leaf in cards.OWNED["piece"]
        assert leaf == "sight" and word.field == "effect"
        assert word.value in MODEL["sight"]["effects"]
    elif word.key.startswith("board."):
        leaf = word.key.removeprefix("board.")
        board = rule_spec_schema(strict=True)["properties"]["board"]["properties"]
        assert word.value in board[leaf]["enum"]
        assert f"lattice === '{word.value}'" in DESIGNER_TEXT
    else:
        # A kind of move, the move row's own menu.
        assert word.key in MOVE_KINDS and "kind" in cards.OWNED["move"]
        assert f"['{word.key}', " in DESIGNER_TEXT
    # The hand-drawn control's own words, as the designer's script writes them.
    assert f"'{word.card}'" in DESIGNER_TEXT, word.card


# ------------------------------------------------------------------ /rules


@pytest.mark.parametrize(("block", "word"), WORDS, ids=WORD_IDS)
def test_every_link_lands_on_its_heading_of_rules(block: landing.Block,
                                                  word: landing.Word) -> None:
    _, headings = rules_page.drawn()
    by_id = {heading.id: heading for heading in headings}
    link = landing.word_link(word, landing.rule_sections())
    path, _, anchor = link.partition("#")
    assert path == reference.PAGE
    assert anchor in by_id, link
    section = landing.section_key(word.key)
    assert by_id[anchor].key == section
    # The section it lands on writes the rule down: its key, its field, the option.
    text = SECTIONS[section]
    leaf = word.key.rsplit(".", 1)[-1].removeprefix("pieces[]")
    if section != word.key:
        assert f"`{leaf}`" in text or f'`"{leaf}"`' in text, (section, leaf)
    if word.field:
        assert f"**`{word.field.rsplit('.', 1)[-1]}`**" in text, (section, word.field)
    if isinstance(word.value, str):
        assert f'`"{word.value}"`' in text or f"'{word.value}'" in text, (section, word.value)


def test_every_word_on_the_page_links_where_the_table_says() -> None:
    links = re.findall(r'<a class="wallWord" href="([^"]+)" title="[^"]*" data-door="[^"]*">'
                       r'([^<]+)</a>', _wall())
    sections = landing.rule_sections()
    expected = [(landing.word_link(word, sections), word.said.replace("'", "&#x27;"))
                for _, word in WORDS]
    assert links == expected


def test_the_ids_are_the_rules_pages() -> None:
    _, headings = rules_page.drawn()
    assert landing.rule_sections() == {h.key: h.id for h in headings if h.key}


# ------------------------------------------------------------------ the page


class TestTheWall:
    def test_one_row_per_family_in_the_tables_order(self) -> None:
        rows = re.findall(r'<div class="wallRow ([a-z-]+)" data-wall="([^"]+)">', _wall())
        assert rows == [(block.hue, block.family.replace("'", "&#x27;"))
                        for block in landing.BLOCKS]

    def test_the_lead_links_the_rule_language_and_the_band_starts_a_game(self) -> None:
        wall = _wall()
        assert f'<a href="{reference.PAGE}">the rule language</a>' in wall
        assert '<a class="button primary" href="#/design">Start building a game</a>' in wall
        assert wall.count("button primary") == 1

    def test_its_words_are_a_visitors(self) -> None:
        assert site_words.hits(site_words.visible(_wall())) == []
        for _, word in WORDS:
            # A word is never the language's own spelling of a key.
            assert not re.search(r"[_.\[\]`]", word.said), word.said

    def test_no_inline_style(self) -> None:
        assert "style=" not in _wall()
