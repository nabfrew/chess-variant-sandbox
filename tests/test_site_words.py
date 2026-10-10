"""The words the running site serves to a visitor, held against `site_words.PHRASES` (W0).

`docs/plan-understanding.md` §7, task W0, Q607. `test_candidacy.py::TestTheSitesWordsAreAVisitors`
holds the page's own script, run with `ACCOUNTS` true, to the same list. This file reads
what a running site *sends*: its pages over HTTP, the scripts of its own pages, and the
JSON the library and a game's page are drawn from, for three library games and a draft
somebody saved.

It needs a database, because the site does (`/api/game` reads the games table), and so it
skips without `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL`. `tests/test_site.py` has the
recipe for a scratch cluster.

The list starts with the phrases already gone from what the site serves. The ones that are
not yet gone are `site_words.NOT_YET`, each with the task that removes it; a word task
moves its phrases out of that set, and the test below fails if it forgets to.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_candidacy import served_by_the_page
from test_site_admin import (
    SITE_URL,
    Browser,
    Site,
    _running_site,
    _saved,
    _screened,
    _template,
)

from chess_variant_sandbox.site import mail
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import library

# Three library games that differ in what they have to say: the calibration game, the
# anchor of the low end, and one whose pieces and rules are not chess's.
LIBRARY = ("chess", "dull", "intransitive")


# What `/api/game` sends that the site's page never draws. `keeping` is the local
# application's promote-and-archive panel: the page draws it only where
# `offered('POST', '/api/promote')`, and the site withholds that route (checked below).
UNDRAWN = (".keeping",)


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """A running site over three library games, each with the row a listed game has."""
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    for name in LIBRARY:
        shutil.copy(library.RULESET_DIR / f"{name}.json", tmp_path / "rulesets")
    with _running_site(tmp_path) as running:
        for name in LIBRARY:
            spec = json.loads((tmp_path / "rulesets" / f"{name}.json").read_text("utf-8"))
            running.game(spec_hash(load_spec(tmp_path / "rulesets" / f"{name}.json")), spec)
        yield running


def _get(browser: Browser, path: str) -> Any:
    answer = browser.get(path)
    assert answer.status == 200, (path, answer.status, answer.text[:200])
    return answer


def _served_text(site: Site, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """`(where, words)` for every piece of prose the site sends, as a visitor and as a member."""
    _screened(monkeypatch, passed=True)
    found: list[tuple[str, str]] = []

    def page(where: str, markup: str) -> None:
        found.append((where, site_words.visible(markup)))

    def data(where: str, payload: Any) -> None:
        found.extend((f"{where}{path}", text) for path, text in site_words.strings(payload)
                     if not path.startswith(UNDRAWN))

    visitor = site.browser()
    member = site.signed_in("ada")

    for path in ("/", "/signin", "/join?code=abc", "/credits", "/privacy", "/terms", "/badges",
                 "/request-invitation"):
        page(path, _get(visitor, path).text)
    for path in ("/profile", "/players", "/suggestions"):
        page(path, _get(member, path).text)
    for route in site_pages.SCRIPTS:
        if route.endswith("/admin.js"):
            continue  # the maintainer's own page, whose voice is right for its reader
        script = _get(visitor, route).text
        found.extend((route, text) for text in site_words.literals(script))

    data("/api/library", _get(visitor, "/api/library").payload)
    for name in LIBRARY:
        digest = spec_hash(load_spec(library.RULESET_DIR / f"{name}.json"))
        data(f"/api/game/{name}", _get(visitor, f"/api/game/{digest}").payload)
    draft = _saved(member, _template(name="a draft somebody kept")).payload["hash"]
    data("/api/game/<a draft>", _get(member, f"/api/game/{draft}").payload)
    return found


def _server_sentences() -> list[tuple[str, str]]:
    """W6 (rows S13-S25): the server's sentences a visitor reads that the wire read above
    does not reach without a profile, a round or a refused start: the milestones' rules
    (`/api/profile`), every badge's (`/badges`, read above too), the levels' and the
    round's own sentences, the refusals of `matches` and the strong-seat cap's, each
    produced by the code that says it."""
    from chess_variant_sandbox.site import (
        asking,
        badges,
        inviting,
        matches,
        milestones,
        quota,
        rounds,
        shortlist,
        stages,
        terms,
    )

    found = [(f"milestone {kind}", rule) for kind, rule in milestones.RULES.items()]
    found += [(f"badge {kind}", f"{badge.sentence} {badge.rule}")
              for kind, badge in badges.BADGES.items()]
    found += [(f"level {level}", said) for level, said in stages.SAYS.items()]
    found += [("the round's rule", rounds.RULE),
              ("a visitor's rated game", matches.NEEDS_AN_ACCOUNT),
              ("a watched rated game", matches.WATCH_IS_SANDBOX),
              ("a rated game against the learned player", matches.NEVER_RATED),
              ("a nomination", shortlist.Nominated(hash="", already=False).said),
              # L15 and Q719 (`docs/plan-launch.md` §3.6, §5.3a).
              ("a visitor against hardest", matches.members_only_said("minimax:nodes=200000")),
              ("no invitations left", inviting.NONE_LEFT),
              ("an address with an account", inviting.ALREADY_JOINED),
              # Q831: a visitor's request for an invitation, its answer and refusals.
              ("a request's answer", asking.CONFIRMED),
              ("a request's bad address", asking.NOT_AN_ADDRESS),
              ("a request's hint", asking.NOTE_HINT),
              ("the requests' full day", quota.ASKS_FULL),
              ("a connection's requests", quota.ASKED_ENOUGH),
              # Q722: the terms' sentence at *Publish* and its two refusals.
              ("the terms at publish", terms.DEDICATION),
              ("a publish without the box", terms.NOT_ACCEPTED),
              ("a publish under old terms", terms.STALE),
              *((f"the terms: {point.key}", f"{point.heading} {point.plain} {point.legal}")
                for point in terms.points()),
              *((f"{left} invitations left", inviting.left_said(left)) for left in (1, 2))]
    # What's new, for each of a level move's three shapes: joined, left, and entered.
    moved = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for was, became in (("archive", stages.CORPUS), (stages.CORPUS, "archive"),
                        (stages.DESIGNED, "archive")):
        move = stages.Move(game="", name="gardner", was=was, became=became, reason="Why.",
                           moved=moved)
        found.append((f"a move {was} to {became}",
                      " ".join([move.heading, move.page, *move.paragraphs])))
    return found


def test_the_servers_own_sentences_hold_no_maintainers_phrase() -> None:
    """No database: what the site's server modules say, read where they say it."""
    found = _server_sentences()
    assert len(found) > 30
    trouble = [f"{where}: {phrase!r} in {text[:90]!r}"
               for where, text in found for phrase in site_words.hits(text)]
    assert not trouble, "\n".join(trouble)


def _every_library_spec() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{path.parent.name}/{path.stem}", json.loads(path.read_text("utf-8")))
            for folder in (library.RULESET_DIR, library.RULESET_DIR.parent / "archive")
            for path in sorted(folder.glob("*.json"))]


class TestTheLibrarysOwnWords:
    """W7 (Q614, `docs/plan-understanding.md` §3.3 C1-C4, G39): every game the library and
    the archive hold, as the site sends its spec and its provenance, in a player's words.
    The wire test above reads three games; this reads all of them without a database."""

    def test_no_game_the_site_sends_holds_a_maintainers_phrase(self) -> None:
        from chess_variant_sandbox.site.server import visitors_provenance

        specs = _every_library_spec()
        assert len(specs) > 50
        trouble = []
        for name, spec in specs:
            sent = {**spec, "provenance": visitors_provenance(spec.get("provenance") or {})}
            trouble += [f"{name}{where}: {phrase!r} in {text[:90]!r}"
                        for where, text in site_words.strings(sent)
                        for phrase in site_words.hits(text)]
        assert not trouble, "\n".join(trouble)

    def test_no_description_speaks_of_the_project(self) -> None:
        """C1-C4: the words W7 took out of descriptions, which the guard's list is too
        general to name (*roster*, *ply*, *spec*, a schema's number, *calibration*)."""
        pattern = re.compile(r"\broster\b|\bply\b|\bspecs?\b|calibration|interestingness|"
                             r"\bmetric\b|control arm|labelled negative|showcase", re.I)
        found = [f"{name}: {match.group(0)!r}" for name, spec in _every_library_spec()
                 for match in [pattern.search(spec.get("description", ""))] if match]
        assert not found, found

    def test_the_site_keeps_a_provenance_s_citation_and_drops_its_records(self) -> None:
        from chess_variant_sandbox.site.server import visitors_provenance

        sent = visitors_provenance({
            "origin": "generated", "author": "stage 25", "history": "Made in runs/abc.",
            "notes": "hash 15e3eb35bfbc", "run": "live-1", "theme": "a board with no corners",
            "source": "docs/game-ideas.md, Q281"})
        assert sent == {"origin": "generated", "theme": "a board with no corners"}
        kept = {"origin": "human", "author": "Martin Gardner", "year": "1969",
                "source": "webgoatguy https://www.youtube.com/watch?v=x"}
        assert visitors_provenance(kept) == kept

    def test_the_site_sends_a_games_page_without_the_records(
        self, served: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """G39 on the wire, for a game whose file holds a history: neither the page's
        provenance nor the spec it carries sends it."""
        digest = spec_hash(load_spec(library.RULESET_DIR / "chess.json"))
        payload = _get(served.browser(), f"/api/game/{digest}").payload
        assert "history" not in payload["provenance"]
        assert "history" not in payload["spec"]["provenance"]
        assert payload["spec"]["description"].startswith("Standard international chess.")


class TestTheSitesWordsAreAVisitorsOnTheWire:
    def test_nothing_the_site_sends_holds_a_maintainers_phrase(
        self, served: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        found = _served_text(served, monkeypatch)
        assert len(found) > 100, "the guard read almost nothing"
        trouble = [f"{where}: {phrase!r} in {text[:90]!r}"
                   for where, text in found for phrase in site_words.hits(text)]
        assert not trouble, "\n".join(trouble[:40])

    def test_what_the_guard_leaves_unread_is_not_drawn_on_the_site(self) -> None:
        from chess_variant_sandbox.site import routes

        assert routes.find("POST", "/api/promote") is None  # `keeping` is drawn only if offered

    @pytest.mark.skipif(not site_words.NOT_YET, reason="nothing is listed as not yet")
    def test_a_phrase_still_listed_as_not_yet_is_still_served(
        self, served: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the list cannot go stale: the task that removes a phrase from the site moves
        it to `ENFORCED` in the same change, and this fails until it does. The page's own
        script is read by `test_candidacy.py`, so a phrase that only it still serves is
        looked for there as well.

        A mark rather than a skip in the body (Q745), so the empty list is decided before
        the site fixture is built and the skip is never read as a pass that asserted
        nothing."""
        text = " ".join(words for _, words in _served_text(served, monkeypatch))
        text += " " + " ".join(words for _, words in served_by_the_page())
        gone = [name for name in site_words.NOT_YET if not site_words.hits(text, (name,))]
        assert not gone, f"gone from everything the guard reads; move to ENFORCED: {gone}"
