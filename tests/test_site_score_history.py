"""`GET /api/library/history` on the site (SH2 of `docs/plan-score-history.md`).

Public (D1, answered A by the user on 2026-11-08): a visitor reaches it signed out and is
sent the values, dates, run counts and versions, never a ledger line's basis; an admin is
sent it whole, as the local application is; a game taken off the site is not in it. The
payload's own shape is `tests/test_history.py::TestTheLibraryHistory`'s.

Needs the scratch database every site test needs; skips without one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from test_corpus import SPEC, world  # noqa: F401  (`world` is a fixture)
from test_history import _boards, _line, _stored, game, stubbed  # noqa: F401  (fixtures)
from test_site_admin import Site, _running_site

from chess_variant_sandbox import history
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.site import admin
from chess_variant_sandbox.web import library

ROUTE = "/api/library/history"


@pytest.fixture
def scored_site(world, game, monkeypatch) -> Iterator[Site]:  # noqa: F811
    """The site over `world`'s tree: one archived game, measured under the current
    weights, with two ledger lines and a test's boards."""
    monkeypatch.delenv(admin.NAME_VAR, raising=False)
    monkeypatch.delenv(admin.PASSWORD_VAR, raising=False)
    monkeypatch.setattr(history, "boards", lambda: _boards(world["hash"]))
    _stored(world, index=0.8)
    _line(world, 1, 0.7, "v12", "2026-11-05", net="b")
    _line(world, 2, 0.75, "v12", "2026-11-06", engine="c")
    with _running_site(world["runs"].parent) as running:
        yield running


def _history(browser: Any) -> dict[str, Any]:
    answer = browser.get(ROUTE)
    assert answer.status == 200, answer.text
    return dict(answer.payload)


class TestTheScoreHistoryOnTheSite:
    def test_a_visitor_reads_it_without_the_maintainers_fields(
        self, scored_site: Site, world  # noqa: F811
    ) -> None:
        sent = _history(scored_site.browser())
        assert sent["maintainer"] is False and sent["currentVersion"] == DEFAULT_WEIGHTS
        assert all("source" not in snapshot for snapshot in sent["snapshots"])
        [row] = sent["games"]
        assert row["hash"] == world["hash"]
        assert row["points"][DEFAULT_WEIGHTS] == {"index": 0.8, "viable": True}
        assert [day["index"] for day in row["days"]] == [0.7, 0.75]
        for day in row["days"]:
            assert not set(day) & set(history.MAINTAINER_DAY_FIELDS)
        assert sent["cohort"] == [world["hash"]]

    def test_a_member_reads_what_a_visitor_reads(self, scored_site: Site) -> None:
        member = scored_site.signed_in("ada")
        assert _history(member) == _history(scored_site.browser())

    def test_an_admin_reads_it_whole(self, scored_site: Site) -> None:
        boss = scored_site.signed_in("boss", admin.ADMIN)
        sent = _history(boss)
        assert sent["maintainer"] is True
        assert all("source" in snapshot for snapshot in sent["snapshots"]
                   if snapshot["kind"] == "board")
        [row] = sent["games"]
        assert row["days"][0]["basis"] == {"weights_version": "v12"}
        assert row["days"][1]["moved"] == ["engine"]

    def test_a_game_taken_off_the_site_is_not_in_it(self, scored_site: Site, world) -> None:  # noqa: F811
        scored_site.game(world["hash"], {**SPEC, "name": "lancers"})
        with scored_site.store.cursor() as cursor:
            cursor.execute("update games set revoked = now() where hash = %s",
                           (world["hash"],))
        scored_site.store.commit()
        library._HISTORIES.clear()
        sent = _history(scored_site.browser())
        assert sent["games"] == [] and sent["cohort"] == []

    def test_a_provisional_score_reaches_a_visitor_marked(
        self, scored_site: Site, world  # noqa: F811
    ) -> None:
        import json

        from chess_variant_sandbox import corpus

        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["basis"] = {**stored["basis"], "engine": "another-core"}
        path.write_text(json.dumps(stored), encoding="utf-8")
        [row] = _history(scored_site.browser())["games"]
        assert row["current"]["status"] == "provisional"
        assert row["current"]["revision"] == DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS not in row["points"]


class TestThePageOnTheSite:
    """SH3: `/score-history` is the single page with its own head, for anybody, and
    `/admin` links to it beside *Earlier scores*."""

    def test_a_visitor_reaches_the_page(self, scored_site: Site) -> None:
        from chess_variant_sandbox.web.page import front

        answer = scored_site.browser().get(front.HISTORY_PATH)
        assert answer.status == 200, answer.text
        assert f"<title>{front.HISTORY_TITLE}</title>" in answer.text
        assert f'{front.HISTORY_PATH}">' in answer.text and "history." in answer.text

    def test_admin_links_to_it(self, scored_site: Site) -> None:
        boss = scored_site.signed_in("boss", admin.ADMIN)
        assert 'href="/score-history"' in boss.get("/admin").text


class TestTheGamePagesChart:
    """SH4: a game page's *earlier scores* fold draws, for a visitor, from `for_visitors`'
    lines alone (`earlier`), each carrying its version; the ledger stays the admin's."""

    def test_a_visitor_is_sent_for_visitors_lines_and_no_ledger(
        self, scored_site: Site, world  # noqa: F811
    ) -> None:
        answer = scored_site.browser().get(f"/api/game/{world['hash']}")
        assert answer.status == 200, answer.text
        sent = dict(answer.payload)
        assert sent["history"] == []
        # The replacement that moved the players, with its version; the one that moved only
        # the engine is no line for a visitor.
        assert sent["earlier"] == [
            {"index": 0.7, "runs": 4, "measured": "2026-11-05", "weights_version": "v12"}]

    def test_an_admin_is_sent_the_ledger_too(self, scored_site: Site, world) -> None:  # noqa: F811
        boss = scored_site.signed_in("boss", admin.ADMIN)
        sent = dict(boss.get(f"/api/game/{world['hash']}").payload)
        assert sorted(line["index"] for line in sent["history"]) == [0.7, 0.75]
        assert [line["index"] for line in sent["earlier"]] == [0.7]
