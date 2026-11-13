"""What the score has been tested against, `/score-tests` (R-S9 of `docs/roadmap-2026-11.md`).

The rows are `chess_variant_sandbox/web/score_tests.json`, one entry per test; a later test's
task appends its own. These tests hold every entry to its fields, the page to drawing every
field of every entry (misses as hits), both servers to answering it for anybody, and the
links to it from the footers and from a score's basis. The site's half needs the scratch
database every site test needs, and skips without one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import escape

import page_text
import pytest
from test_page_files import RULESETS, _get
from test_site_admin import Site, _running_site

from chess_variant_sandbox.site import pages, routes, seo
from chess_variant_sandbox.web import landing, library
from chess_variant_sandbox.web.page import assets, game, rules_page, tokens
from chess_variant_sandbox.web.page import score_tests_page as page
from chess_variant_sandbox.web.server import App


def _entries() -> list[dict[str, object]]:
    entries = page.raw()["tests"]
    assert isinstance(entries, list)
    return entries


class TestTheRows:
    def test_every_row_has_every_field(self) -> None:
        entries = _entries()
        assert entries, "the page opens with every test read so far"
        for entry in entries:
            assert page.problems(entry) == [], entry.get("id") if isinstance(entry, dict) else entry
        assert len(page.tests()) == len(entries)

    def test_each_test_has_one_row(self) -> None:
        ids = [test.id for test in page.tests()]
        assert len(ids) == len(set(ids))

    def test_the_rows_that_exist_today(self) -> None:
        """The tests read as of 2026-11-13 (R-S9's brief): each its own row. A later test
        adds to this, never removes from it."""
        ids = {test.id for test in page.tests()}
        assert ids >= {"R-S5", "R-S1b", "R-S2c", "R-S4b", "R-S10", "R-S11", "R-S12", "R-S14",
                       "R-S11b", "R-S15", "R-S15b", "R-S14a", "R-S13", "R-S14b"}

    def test_a_row_missing_a_field_is_refused(self) -> None:
        entry = dict(_entries()[0])
        del entry["predicted"]
        assert any("no predicted" in said for said in page.problems(entry))
        entry = dict(_entries()[0], outcome="pass")
        assert any("outcome" in said for said in page.problems(entry))
        entry = dict(_entries()[0], read="13 November")
        assert any("YYYY-MM-DD" in said for said in page.problems(entry))
        entry = dict(_entries()[0], extra="x")
        assert any("unknown" in said for said in page.problems(entry))

    def test_the_wording_rule(self) -> None:
        """The owner, 2026-11-12: never state as fact that the score measures likeness to
        chess or marks games unlike chess down. The page may name it only as a question."""
        said = " ".join(" ".join(vars(test).values()) for test in page.tests()).lower()
        for claim in ("the score marks", "marks down games", "measures how chess-like"):
            assert claim not in said, claim
        # *Measures likeness* only where the sentence says it is not shown.
        for found in re.finditer(r"measures likeness", said):
            assert "not show" in said[max(0, found.start() - 40):found.start()], \
                said[max(0, found.start() - 60):found.end()]


class TestThePage:
    def test_every_field_of_every_row_is_drawn(self) -> None:
        body = page.body()
        for test in page.tests():
            for words in (test.question, test.asked, test.predicted, test.found, test.tally,
                          test.shows, test.record, test.id, page.day(test.read),
                          page.OUTCOMES[test.outcome]):
                assert escape(words) in body, (test.id, words[:40])
            assert f'id="{page.anchor(test)}"' in body
            assert f'href="#{page.anchor(test)}"' in body

    def test_a_miss_is_drawn_as_a_hit_is(self) -> None:
        """The outcome is a word in the same mark for every outcome: one class, no colour
        that tells a hit from a miss."""
        marks = re.findall(r'<span class="([^"]+)" data-outcome="([^"]+)">', page.body())
        assert {kind for kind, _ in marks} == {"testOutcome"}
        assert {"hit", "miss"} <= {outcome for _, outcome in marks}
        assert "data-outcome" not in page.STYLE

    def test_newest_first(self) -> None:
        shown = page.newest_first(page.tests())
        assert [test.read for test in shown] == sorted((t.read for t in shown), reverse=True)
        body = page.body()
        last = page.tests()[-1]
        first_dated = min(page.tests(), key=lambda test: test.read)
        assert body.index(page.anchor(last)) < body.index(page.anchor(first_dated))

    def test_it_says_who_ran_the_tests(self) -> None:
        assert "AI agents" in page.body() and "before its numbers were computed" in page.body()

    def test_its_rules_name_no_colour_and_read_only_set_tokens(self) -> None:
        rules = re.sub(r"/\*.*?\*/", "", page.STYLE, flags=re.S)
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", rules)
        read = set(re.findall(r"var\(--([a-z0-9-]+)", rules))
        assert read <= set(tokens.LIGHT) | set(tokens.SIZES)
        assert assets.sheet_text(page.SHEET) == page.STYLE

    def test_title_and_description_fit(self) -> None:
        assert len(page.DESCRIPTION) <= game.DESCRIPTION_LIMIT
        assert len(page.TITLE) <= 70


class TestTheLinks:
    def test_the_local_application_answers_it(self, tmp_path) -> None:
        app = App(RULESETS, library.RUNS_DIR, browser_dir=tmp_path / "absent")
        [(status, headers, body)] = _get(app, page.PATH)
        text = body.decode("utf-8")
        assert status == 200 and headers["Content-Type"].startswith("text/html")
        assert f"<title>{escape(page.TITLE)}</title>" in text
        assert escape(page.tests()[0].found) in text
        assert "<script>" not in text and "style=" not in text

    def test_the_site_lists_it_as_public(self) -> None:
        assert page.PATH in routes.STATIC
        assert page.PATH not in routes.PERSONAL
        assert page.PATH in seo.PAGES

    def test_every_footer_links_it_after_the_score_history(self) -> None:
        core = page_text.text("/static/page/core.js")
        history = core.index("link('/score-history', 'score history'),")
        tests = core.index("' · ', link('/score-tests', 'score tests')")
        assert 0 < tests - history < 80
        pair = '<a href="/score-history">score history</a> · <a href="/score-tests">score tests</a>'
        assert pair in rules_page.document()
        assert pair in pages.not_found() and pair in pages.privacy()
        # The site's own page drops the link to itself, as every page does.
        assert 'href="/score-tests">score tests' not in pages.score_tests()

    def test_a_scores_basis_links_it(self) -> None:
        detail = page_text.text("/static/page/detail.js")
        assert '<a href="/score-tests">what the score has been tested against</a>' in detail
        assert "&middot; ${SCORE_TESTS_LINK}</div>" in detail
        drawn = game.Drawn(hash="0" * 16, name="A game", description="", spec={}, score=None,
                           stale=False, near=(), indexable=True, base="")
        assert f'<a href="{page.PATH}">{game.SCORE_TESTS_WORDS}</a>' in game.view(drawn)
        assert f'<a href="{page.PATH}">' in landing.score_html()


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


class TestTheSite:
    def test_a_visitor_reads_every_row(self, site: Site) -> None:
        answer = site.browser().get(page.PATH)
        assert answer.status == 200, answer.text
        assert f"<title>{escape(page.TITLE)}</title>" in answer.text
        assert f'<link rel="canonical" href="{page.PATH}">' in answer.text
        for test in page.tests():
            assert escape(test.found) in answer.text and escape(test.shows) in answer.text
        assert assets.name_of("/static/style/score-tests.css") in answer.text

    def test_its_stylesheet_is_served(self, site: Site) -> None:
        answer = site.browser().get(page.SHEET)
        assert answer.status == 200, answer.text
        assert "testOutcome" in answer.text
