"""The designer and the map in the club room's look (archive/plan-visual.md §8, V8).

Spec-side tests of the stylesheet and the two views' scripts as text: the designer's and the
map's rules read only tokens, set for both themes; their radii are the two of §3.4; nothing
inside a panel is drawn as a box of its own; the scripts' inline styles name no colour and no
typeface; the map's controls are one sheet that folds on a phone. Nothing here opens a
browser: the widths are checked by construction, from the rules.
"""

from __future__ import annotations

import re

from page_text import DESIGN, EXPLORER, GRAPH

from chess_variant_sandbox.web.page import style, tokens

_RULES = style.RULES
SECTION = _RULES[_RULES.index("/* --- the designer --- */"):_RULES.index("/* --- what's new")]
# A phone's width less its two 16 px gutters (§3.4): the most any fixed width may ask for.
PHONE_COLUMN = 375 - 2 * 16


def _rule(selector: str) -> str:
    """The body of the first rule whose selector list is exactly `selector`."""
    found = re.search(r"(?:^|\n)\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", _RULES)
    assert found, f"no rule for {selector}"
    return found.group(1)


class TestTheTokens:
    def test_every_value_the_two_views_read_is_a_token_set_in_both_themes(self) -> None:
        read = set(re.findall(r"var\(--([\w-]+)", SECTION))
        # Set on the element by a script (`fitSticky`), or a family (`FAMILIES`).
        own = {"stick-top", "f-mono"}
        for name in read - own:
            assert name in tokens.LIGHT or name in tokens.SIZES, name
            if name in tokens.LIGHT and name not in tokens.SIZES:
                # Dark sets every colour light sets, so neither theme falls back to the other.
                assert name in tokens.DARK, name

    def test_the_rules_name_no_colour(self) -> None:
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(|oklch\(", SECTION)

    def test_every_radius_is_one_of_the_two(self) -> None:
        for radius in re.findall(r"border-radius:\s*([^;]+);", SECTION):
            assert radius.strip() in {"var(--r-ctl)", "var(--r-panel)", "50%", "999px", "0"}, radius

    def test_the_scripts_inline_styles_name_no_colour_and_no_typeface(self) -> None:
        """What were `style=` attributes are `u-` classes since G12 (`utilities.py`), each a
        rule the stylesheet holds: the scripts here name none that sets a face or an edge's
        colour, and no rule of the table names a colour but by its token."""
        from chess_variant_sandbox.web.page.utilities import UTILITIES

        for name, text in (("design", DESIGN), ("explorer", EXPLORER), ("graph", GRAPH)):
            assert 'style="' not in text, name
            for used in set(re.findall(r"(?<![\w-])u-[a-z0-9_-]*[a-z0-9_]", text)):
                assert not used.startswith(("u-ff-", "u-bc-")), (name, used)
        for declaration in UTILITIES.values():
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", declaration), declaration

    def test_the_game_as_text_is_in_the_numbers_face(self) -> None:
        assert 'id="dJson" class="jsonText"' in DESIGN
        assert "var(--f-mono)" in _rule(".jsonText")


class TestFlattened:
    """No panel inside a panel (§3.4): a group inside one is set off by a rule or by space."""

    def test_a_piece_is_a_sheet_and_the_one_in_hand_is_bordered_in_the_action(self) -> None:
        sheet = _rule(".editorPiece")
        for part in ("var(--paper)", "var(--panel)", "var(--r-panel)", "var(--shadow)"):
            assert part in sheet
        assert "var(--act)" in _rule(".editorPiece.on")
        assert "style.borderColor" not in DESIGN

    def test_inside_a_panel_a_group_is_a_rule_not_a_box(self) -> None:
        for selector in (".ruleCard", ".freshNote", ".ruled", ".hereStep"):
            body = _rule(selector)
            assert not re.search(r"(?<![-\w])border:\s*1px", body), selector
            assert "box-shadow" not in body and "background" not in body, selector
            assert re.search(r"border-(top|left):", body), selector
        assert DESIGN.count('class="ruled"') == 2
        assert "inPanel ? 'hereStep' : 'ruled'" in DESIGN
        assert "border-top:1px solid" not in DESIGN

    def test_the_maps_kinds_are_no_longer_a_panel_in_the_controls(self) -> None:
        """The fold of which rule a colour means left with the rule-step lines (M4 of
        `docs/archive/plan-map.md`); no panel of its own came back in its place."""
        assert 'id="mapLegend"' not in GRAPH
        assert "mapKinds" not in GRAPH


class TestTheMapsControls:
    def test_one_sheet_above_the_drawing(self) -> None:
        body = EXPLORER[EXPLORER.index("function explorerBodyHtml"):]
        body = body[:body.index("\n}\n")]
        opened = body.index('<details class="panel mapControls" id="mapControls" open>')
        closed = body.index("</details>")
        for part in ("explorerLinesHtml()", "mapLegendHtml()", "explorerSizeHtml()",
                     "explorerKeepHtml()"):
            assert opened < body.index(part) < closed, part
        assert closed < body.index('<div class="mapView">')

    def test_its_rows_are_a_class_not_inline_spacing(self) -> None:
        assert EXPLORER.count('class="row mapCtl') == 4
        assert 'style="gap:10px; margin:8px 0; flex-wrap:wrap"' not in EXPLORER
        assert "flex: 1 1 200px" in _rule(".mapSay") and "min-width: 0" in _rule(".mapSay")

    def test_it_folds_on_a_phone_so_the_drawing_is_on_the_first_screen(self) -> None:
        assert "controls.open = window.innerWidth > 720;" in EXPLORER
        assert "var(--hit)" in _rule(".mapControls > summary")

    def test_the_frame_and_its_bar_are_the_sheets(self) -> None:
        frame = _rule(".mapFrame")
        assert "var(--r-panel)" in frame and "var(--shadow)" in frame
        assert "var(--r-ctl)" in _rule(".mapLocal")


class TestTheWidths:
    def test_no_fixed_width_outside_a_desk_query_is_wider_than_a_phone(self) -> None:
        # Media queries' own conditions are widths of the screen, not of a box.
        rules = re.sub(r"@media[^{]*\{", "", SECTION)
        for width in re.findall(r"(?<![-\w(])(?:min-)?width:\s*(\d+)px", rules):
            assert int(width) <= PHONE_COLUMN, width
        for basis in re.findall(r"flex:\s*\d+ \d+ (\d+)px", rules):
            assert int(basis) <= PHONE_COLUMN, basis

    def test_the_designer_stacks_and_the_map_takes_one_column_below_900(self) -> None:
        assert "@media (max-width: 900px) { .designer { grid-template-columns: 1fr; } }" in SECTION
        narrow = SECTION[SECTION.index("@media (max-width: 900px) {\n    .mapView"):]
        assert ".mapView { grid-template-columns: 1fr; }" in narrow


class TestTheLabelledControlKeepsItsOwnShape:
    """`label.pick` is the labelled control of the map, the designer and the library; the
    landing page's picks row reused the class name for its panels with a bare `.pick` rule
    (Q788), and every such label became a 200 px grid box: the map's settings spread over
    the first screen at 1400 and the designer's *royal* box sat 70 px from its word (Q777).
    A panel of the row is styled only under `.landingPicks`."""

    @staticmethod
    def _selectors() -> list[str]:
        rules = re.sub(r"/\*.*?\*/", "", _RULES, flags=re.S)
        found: list[str] = []
        for head in re.findall(r"([^{}]+)\{", rules):
            if head.strip().startswith("@"):
                continue
            found += [one.strip() for one in head.split(",")]
        return found

    def test_the_views_use_the_class_for_their_labelled_controls(self) -> None:
        for name, text in (("design", DESIGN), ("explorer", EXPLORER)):
            assert '<label class="pick' in text, name

    def test_no_rule_styles_the_class_by_itself(self) -> None:
        bare = [one for one in self._selectors()
                if re.fullmatch(r"\.pick(?![\w-])\S*", one)]
        assert not bare, bare

    def test_the_picks_rows_panels_are_styled_inside_the_row(self) -> None:
        panels = [one for one in self._selectors()
                  if re.search(r"\.pick(?![\w-])", one) and not one.startswith(("label.pick", ".controls"))]
        assert panels
        for one in panels:
            # LP3: the same row inside `/game`'s band of shelves.
            assert one.startswith((".landingPicks > .pick", ".shelves .landingPicks > .pick")), one


class TestMarkedLinesHangUnderTheirWords:
    """*Publish*'s two ticks and the shortlist's bar: at 375 px *a score measured on your
    computer; the site is checking it now* wrapped back under its tick (Q777's look). Each
    is a list of `.marks`, its mark a column of its own."""

    def test_the_rule_puts_the_mark_in_a_column(self) -> None:
        assert "display: flex" in _rule(".marks > li")
        column = _rule(".marks > li > span:first-child")
        assert "flex: none" in column and "width: 1.4em" in column
        assert "list-style: none" in _rule(".marks")

    def test_both_lists_are_marks_and_no_mark_is_an_inline_block(self) -> None:
        from page_text import DETAIL

        assert '<ul class="small marks ' in DESIGN and '<ul class="small marks ' in DETAIL
        publish = DESIGN[DESIGN.index("function publishHtml"):]
        publish = publish[:publish.index("\n}\n")]
        shortlist = DETAIL[DETAIL.index("function shortlistHtml"):]
        shortlist = shortlist[:shortlist.index("\n}\n")]
        for body in (publish, shortlist):
            assert "u-d-inline-block u-w-1_4em" not in body


def _body(text: str, start: str) -> str:
    """One function of a script, from its `start` to the brace that closes it."""
    body = text[text.index(start):]
    return body[:body.index("\n}\n")]


class TestTheCommunitiesAreListedOnce:
    """At 1400 x 900 the sheet listed the communities twice, as the key and as the chips
    that fold them, and the drawing started at y 675 (Q777's look). The key is the one
    list (Q801): a community folds from its panel, opens from its folded dot's, and its
    chip in the key says it is folded."""

    def test_no_second_list_of_chips_folds_them(self) -> None:
        assert "exploreFolds" not in EXPLORER and "explorerGroupsHtml" not in EXPLORER
        assert 'data-fold="' not in EXPLORER
        assert "mapFolds" not in EXPLORER and ".mapFolds" not in _RULES

    def test_a_community_of_three_or_more_folds_from_its_panel(self) -> None:
        panel = _body(GRAPH, "function mapCommunityPanel")
        assert "community.size >= MAP_COLOURED" in panel
        assert 'data-fold-group="${esc(community.id)}">fold it into one dot</button>' in panel
        assert "[data-open-group], [data-fold-group]" in EXPLORER

    def test_folding_the_lit_community_holds_up_its_dot(self) -> None:
        toggle = _body(EXPLORER, "function explorerFoldToggle")
        assert "MAP_PICK.kind === 'community'" in toggle and "lit ||" in toggle
        pick = _body(EXPLORER, "function explorerPickCommunity")
        assert "EXPLORER_FOLDED.has(community.id)" in pick and "kind: 'group'" in pick

    def test_a_folded_communitys_chip_says_so(self) -> None:
        assert re.search(r"data-folded\s+hidden>&middot; folded</span>", GRAPH)
        paint = _body(EXPLORER, "function explorerPaintCommunityChips")
        assert "folded.hidden = !EXPLORER_FOLDED.has(id)" in paint
        assert EXPLORER.count("explorerPaintCommunityChips();") == 2

    def test_reset_ends_the_size_row(self) -> None:
        size = _body(EXPLORER, "function explorerSizeHtml")
        assert size.count("${explorerResetHtml()}") == 2
        assert 'class="muted small mapSay" id="exploreSizeSay"' in size


class TestTheBarForAGameThatPlays:
    """At 375 px the bar for a game that plays took 181 of 812 px: the verdict in its own
    box on two lines, then *save it*, *play it on its own page* and *watch it on its own
    page* on two rows (Q777's look). In the bar (Q801) *on its own page* is said once,
    before the pair, so the three buttons fit one row, and the verdict is a sentence in the
    good colour with no box of its own; the button's name stays whole."""

    def _narrow(self) -> str:
        at = SECTION.index("@media (max-width: 900px) {\n    .designer .footbar {")
        return SECTION[at:SECTION.index("\n  }\n", at)]

    def test_the_bar_says_on_its_own_page_once(self) -> None:
        verdict = DESIGN[DESIGN.index("function drawVerdict("):]
        verdict = verdict[:verdict.index("\nfunction ", 1)]
        assert '<span class="awayLead muted small">${esc(AWAY_TAIL.trim())}:</span>' in verdict
        for words in ("PLAY_AWAY", "WATCH_AWAY"):
            assert f'aria-label="${{esc({words})}}"' in verdict
            assert f">${{awayLabel({words})}}</button>" in verdict
        assert "const AWAY_TAIL = ' on its own page';" in DESIGN

    def test_only_the_narrow_bar_swaps_the_words(self) -> None:
        assert "display: none" in _rule(".designer .footbar .awayLead")
        narrow = self._narrow()
        assert ".designer .footbar .awayLead { display: inline;" in narrow
        assert ".designer .footbar .awayTail { display: none; }" in narrow

    def test_the_verdict_that_plays_is_no_box_in_the_bar(self) -> None:
        narrow = self._narrow()
        ok = narrow[narrow.index(".designer .footbar > .says > .ok {"):]
        ok = ok[:ok.index("}")]
        assert "border: none" in ok and "padding: 2px 0 0" in ok

