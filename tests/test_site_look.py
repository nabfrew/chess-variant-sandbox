"""V7 of `docs/archive/plan-visual.md` §8: the site's own pages in the club room's look.

The footer signed with the mark on both servers, the not-found page every address
without a page answers (and the one a member asking for `/admin` gets), the site's
tables, counts and prose as the panel's paper, and *What's new* as dated sheets. The
pages' photographs are the look check's; these hold the markup and the rules that make
them.
"""

from __future__ import annotations

from page_text import PAGE as PAGE_WITH_FILES

from chess_variant_sandbox.site import pages, routes
from chess_variant_sandbox.site.server import served_elsewhere
from chess_variant_sandbox.web.page import PAGE, shell
from chess_variant_sandbox.web.page.shell import NAME, mark_path


class TestTheFooter:
    def test_both_servers_sign_the_footer_with_the_mark_and_the_name(self) -> None:
        for page in (PAGE, pages.credits(), pages.privacy(), pages.signin()):
            footer = page[page.index("<footer"):page.index("</footer>")]
            assert 'class="sign"' in footer
            assert mark_path() in footer and NAME in footer
            assert 'id="foot"' in footer and "data-theme-pick" in footer

    def test_the_site_footer_keeps_its_links_and_leaves_out_the_page_being_read(self) -> None:
        credits = pages.credits()
        footer = credits[credits.index("<footer"):]
        assert 'href="/privacy"' in footer and 'href="/credits"' not in footer

    def test_the_signature_is_quiet(self) -> None:
        assert "footer .sign {" in pages.STYLE and "color: var(--muted)" in pages.STYLE


class TestTheNotFoundPage:
    def test_it_is_a_page_in_the_shell_with_two_ways_on(self) -> None:
        lost = pages.not_found()
        assert lost.startswith("<!doctype html>")
        assert "<h1>No page here</h1>" in lost
        assert 'href="/"' in lost and 'href="/#/whats-new"' in lost
        assert shell.mark(64) in lost
        assert lost.count('class="button primary"') == 1  # one primary per view (S6)

    def test_it_names_nothing_about_the_address(self) -> None:
        """A constant, so `/admin` asked by a member is byte for byte an invented address."""
        lost = pages.not_found()
        main = lost[lost.index("<main"):lost.index("</main>")]
        assert lost == pages.not_found()
        assert "admin" not in main.lower() and 'id="who"></span>' in lost

    def test_only_addresses_with_no_page_get_it(self) -> None:
        for path in ("/nowhere", "/admin/x", "/games", "/profile/someone", "/c"):
            assert not served_elsewhere(path), path
        for path in (
            "/", "/index.html", "/api/me", "/api/nowhere", "/favicon.svg", "/static/shell.js",
            "/static/page/core.js", "/static/fonts/x.woff2", "/static/vendor/x.js",
        ):
            assert served_elsewhere(path), path

    def test_no_page_of_the_site_is_shadowed(self) -> None:
        for path in routes.PAGES:
            assert not served_elsewhere(path)  # answered before the check, by `_page`


class TestTheSitesSheets:
    def test_a_table_is_a_sheet_of_paper_with_mono_numbers(self) -> None:
        own = pages.STYLE
        table = own[own.index("  main table {"):own.index("}", own.index("  main table {"))]
        assert "var(--paper)" in table and "var(--r-panel)" in table
        assert "main td.num { font: 14px/20px var(--f-mono)" in own

    def test_a_row_of_buttons_never_wraps_inside_a_scrolling_table(self) -> None:
        assert ".scroller td:has(> button) { white-space: nowrap; }" in pages.STYLE

    def test_the_counts_are_sheets_and_absent_when_empty(self) -> None:
        own = pages.STYLE
        assert ".counts:empty { display: none; }" in own
        counts = own[own.index("  .counts div {"):own.index("}", own.index("  .counts div {"))]
        assert "var(--paper)" in counts

    def test_prose_has_a_reading_measure_and_ruled_sections(self) -> None:
        assert "main.prose { max-width: 42rem; }" in pages.STYLE
        assert "main.prose h2 { padding-top: 18px; border-top: 1px solid var(--line); }" in pages.STYLE

    def test_your_settings_sit_on_a_sheet(self) -> None:
        assert "#move-email:not(:empty) {" in pages.STYLE

    def test_the_header_mark_is_never_muted_by_a_page_rule(self) -> None:
        """The site's old `.mark` (the players' *you*) is a span's rule now, so the header's
        rook is the ink on the site as in the application."""
        assert "\n  .mark {" not in pages.STYLE
        assert "span.mark { color: var(--muted); }" in pages.STYLE


class TestWhatsNew:
    def test_each_day_is_a_sheet_with_its_date_beside_it(self) -> None:
        assert "sheet.className = 'panel newsSheet';" in PAGE_WITH_FILES
        assert ".newsSheet {" in PAGE and "grid-template-columns: 150px minmax(0, 1fr);" in PAGE

    def test_the_title_is_the_page_title_size(self) -> None:
        assert ".news > h2 { font-size: 28px; line-height: 34px;" in PAGE
