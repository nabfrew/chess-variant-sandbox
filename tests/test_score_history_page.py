"""The library's score history page, `/score-history` (SH3 of `docs/plan-score-history.md`).

The page itself: its route on the local application (the site's is
`test_site_score_history.py`'s), the script in the page's table, the links to it, its
glosses, and what `history.js` does with a payload, run in node against a made-up one: a
pick's hue follows the pick (D4), a provisional score is never a point, and the page works
out nothing but positions, ranks within one column and a median (§2.1, §2.5).
"""

from __future__ import annotations

import json
import re

import node_session
import page_text
from test_page_files import RULESETS, _get

from chess_variant_sandbox.site import routes, seo
from chess_variant_sandbox.web import library, page
from chess_variant_sandbox.web.page import front
from chess_variant_sandbox.web.server import App

ROUTE = "/static/page/history.js"


def _script() -> str:
    return page_text.text(ROUTE)


def test_the_script_is_in_the_pages_table() -> None:
    assert ROUTE in page.SCRIPTS
    order = list(page.SCRIPTS)
    # After core.js, whose router calls it, and before boot.js, which starts the page.
    assert order.index("/static/page/core.js") < order.index(ROUTE) < order.index(
        "/static/page/boot.js")


def test_the_local_application_answers_the_route(tmp_path) -> None:
    app = App(RULESETS, library.RUNS_DIR, browser_dir=tmp_path / "absent")
    [(status, headers, body)] = _get(app, front.HISTORY_PATH)
    text = body.decode("utf-8")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert f"<title>{front.HISTORY_TITLE}</title>" in text
    assert f'<link rel="canonical" href="{front.HISTORY_PATH}">' in text
    # The single page, so the router draws the view; a reader without script is told so.
    assert front.HISTORY_NEEDS_SCRIPT in text and "history." in text


def test_the_site_lists_it_as_public() -> None:
    """Public (D1): answered ahead of dispatch with no cookie read, and in the sitemap."""
    assert front.HISTORY_PATH in routes.STATIC
    assert front.HISTORY_PATH not in routes.PERSONAL
    assert front.HISTORY_PATH in seo.PAGES
    assert routes.find("GET", "/api/library/history") in routes.ALLOWED


def test_the_listing_and_admin_link_to_it() -> None:
    from chess_variant_sandbox.site import pages

    assert f'href="{front.HISTORY_PATH}"' in page_text.text("/static/page/browse.js")
    assert f'href="{front.HISTORY_PATH}"' in pages.admin("boss")


def test_the_router_names_the_view() -> None:
    core = page_text.text("/static/page/core.js")
    assert "showHistory()" in core and "'score-history': 'How the scores have moved'" in core
    assert f"const HISTORY_PATH = '{front.HISTORY_PATH}';" in _script()


def test_every_gloss_the_page_reads_is_written() -> None:
    core = page_text.text("/static/page/core.js")
    read = set(re.findall(r"info\('([A-Za-z]+)'\)", _script()))
    assert read == {"snapshot", "weightsVersion", "refusedScore", "historyProvisional"}
    for key in read:
        assert re.search(rf"\n  {key}: \['", core), key


def test_it_draws_in_tokens_only() -> None:
    script = _script()
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", script)
    assert "rgb(" not in script and "style=" not in script
    for token in set(re.findall(r"var\(--([a-z0-9-]+)\)", script)) - {"pick-${slot + 1}"}:
        assert token in {"trace", "panel", "ink", "muted", "line", "pick-1"}, token


# A made-up payload in `history.library_history`'s shape: three snapshots (a board, a
# ledger version and the current one), four games, one of them provisional now.
PAYLOAD = {
    "currentVersion": "v13",
    "snapshots": [
        {"kind": "board", "id": "v12a", "weights_version": "v12", "date": "2026-10-03",
         "basis": "the boards"},
        {"kind": "ledger", "id": "v12", "weights_version": "v12", "date": "2026-11-07",
         "basis": "the ledgers"},
        {"kind": "current", "id": "v13", "weights_version": "v13", "date": "2026-11-08",
         "basis": "now"},
    ],
    "games": [
        {"hash": "a" * 12, "name": "alpha", "source": "committed",
         "points": {"v12a": {"index": 0.9, "viable": True}, "v12": {"index": 0.8, "viable": True},
                    "v13": {"index": 0.95, "viable": True}},
         "current": {"status": "current", "revision": "v13", "measured": "2026-11-08",
                     "runs": 4, "index": 0.95, "viable": True},
         "days": [{"index": 0.8, "viable": True, "weights_version": "v13",
                   "measured": "2026-11-01", "replaced": "2026-11-08", "why": "x"}]},
        {"hash": "b" * 12, "name": "beta", "source": "committed",
         "points": {"v12a": {"index": 0.5, "viable": False}, "v12": {"index": 0.6, "viable": True},
                    "v13": {"index": 0.7, "viable": True}},
         "current": None, "days": []},
        {"hash": "c" * 12, "name": "gamma", "source": "archived",
         "points": {"v12a": {"index": 0.7, "viable": True}, "v12": {"index": 0.7, "viable": True},
                    "v13": {"index": 0.8, "viable": True}},
         "current": None, "days": []},
        {"hash": "d" * 12, "name": "delta", "source": "archived",
         "points": {"v12": {"index": 0.4, "viable": True}},
         "current": {"status": "provisional", "revision": "v12", "measured": "2026-11-02",
                     "runs": 4, "index": 0.41, "viable": True},
         "days": [{"index": 0.4, "viable": True, "weights_version": "v12",
                   "measured": "2026-10-20", "replaced": "2026-11-02", "why": "y"}]},
    ],
    "cohort": ["a" * 12, "b" * 12, "c" * 12],
    "joins": [],
    "maintainer": False,
}

_STUB = """
const location = { search: '', pathname: '/score-history', hash: '' };
let RENDERED = '';
const esc = s => String(s);
const info = key => '(' + key + ')';
"""


def _run(probe: str) -> dict:
    source = _STUB + _script() + f"\nconst DATA = {json.dumps(PAYLOAD)};\n" + probe
    done = node_session.run(source)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_columns_are_labelled_and_a_late_one_says_so() -> None:
    found = _run("console.log(JSON.stringify(historyPrepared(DATA).snapshots.map(s => s.label)));")
    assert found == ["v12", "v12 late", "v13"]


def test_the_opening_picks_are_the_cohorts_highest_middle_and_lowest_now() -> None:
    found = _run("""
      HISTORY_DATA = historyPrepared(DATA);
      console.log(JSON.stringify(historyOpeningPicks(HISTORY_DATA)));""")
    assert found == ["a" * 12, "c" * 12, "b" * 12]


def test_with_every_current_score_provisional_the_picks_come_from_the_newest_column() -> None:
    """A new core makes every current score provisional: the current column and the cohort
    are empty, and the opening picks are read from the newest column that has points."""
    found = _run("""
      const bare = { ...DATA, cohort: [], games: DATA.games.map(g => {
        const points = { ...g.points }; delete points.v13; return { ...g, points }; }) };
      HISTORY_DATA = historyPrepared(bare);
      console.log(JSON.stringify(historyOpeningPicks(HISTORY_DATA)));""")
    assert found == ["a" * 12, "c" * 12, "d" * 12]


def test_a_pick_keeps_its_hue_and_the_fourth_drops_the_oldest() -> None:
    found = _run("""
      HISTORY_DATA = historyPrepared(DATA);
      ['a', 'b', 'c'].forEach(h => historyTake(h.repeat(12)));
      const before = historyHue('b'.repeat(12));
      historyTake('d'.repeat(12));
      console.log(JSON.stringify({
        picked: HISTORY_PICKED, before, after: historyHue('b'.repeat(12)),
        dropped: historyHue('a'.repeat(12)), fourth: historyHue('d'.repeat(12)) }));""")
    assert found["picked"] == ["b" * 12, "c" * 12, "d" * 12]
    assert found["before"] == found["after"] == "var(--pick-2)"
    assert found["dropped"] == "var(--trace)" and found["fourth"] == "var(--pick-1)"


def test_a_provisional_score_is_never_a_point() -> None:
    found = _run("""
      const data = historyPrepared(DATA);
      const delta = data.byHash['d'.repeat(12)];
      const alpha = data.byHash['a'.repeat(12)];
      console.log(JSON.stringify({
        deltaNow: historySteps(delta, 'v13', 'v13').end,
        deltaV12: historySteps(delta, 'v12', 'v13').steps.length,
        alphaNow: historySteps(alpha, 'v13', 'v13').end,
        column: historyViable(data, 'v13').map(([g]) => g.name) }));""")
    assert found["deltaNow"] is None and found["deltaV12"] == 1
    assert found["alphaNow"]["index"] == 0.95
    assert "delta" not in found["column"]


def test_the_band_counts_only_what_was_not_refused() -> None:
    found = _run("""
      const data = historyPrepared(DATA);
      const values = historyViable(data, 'v12a').map(([, p]) => p.index);
      console.log(JSON.stringify({ values, median: historyMedian(values) }));""")
    assert sorted(found["values"]) == [0.7, 0.9] and found["median"] == 0.8


def test_rank_is_within_one_column() -> None:
    found = _run("""
      const data = historyPrepared(DATA);
      console.log(JSON.stringify(data.ids.map(id => historyRanked(data.cohort, data, id))));""")
    a, b, c = "a" * 12, "b" * 12, "c" * 12
    assert found == [[a, c, b], [a, c, b], [a, c, b]]
