"""The designer's milestones, D1 to D4 of `docs/archive/plan-meta-game.md` (Q252).

Three claims.

**The reading.** `milestones.designer_from` and `marks_from` over hand-built rows: one
line of each kind however many games (the seven tests' sixth), the designer's own plays
and forks never counting, D3 naming nobody, and nothing depending on when it is read (the
first: nothing lapses, no streak).

**The store.** A running site over a scratch store: saves through the designer's route,
matches through the board's, forks with a parent, and `/api/profile` and `/api/game/<hash>`
read back the four lines and the three marks. These need
`CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it.

**The pages.** The profile's dated list and the game page's *Where it stands* draw what
they are sent, in node, and draw nothing where there is nothing to say (absent is not
zero, test 5).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from test_site import _done
from test_site_admin import (
    AGENT,
    Browser,
    Site,
    _saved,
    _screened,
    _template,
    _user_id,
    drafting,  # noqa: F401  (the fixture, used by name)
)
from test_web import _in_page

from chess_variant_sandbox.site import designs, identity, matches, milestones, server
from chess_variant_sandbox.site.milestones import Design, Try

ADA, BOB, CY = 1, 2, 3


def _day(day: int) -> datetime:
    return datetime(2026, 9, 1, tzinfo=UTC) + timedelta(days=day)


def _design(
    digest: str, owner: int | None, day: int, parent: str | None = None, passed: bool = False
) -> Design:
    return Design(
        hash=digest, owner=owner, parent=parent, created=_day(day), screen_passed=passed
    )


def _try(game: str, user: int, day: int, match: str = "") -> Try:
    return Try(game=game, user=user, started=_day(day), match=match or f"{game}-{user}-{day}")


# ------------------------------------------------------------------------ the reading


class TestTheLines:
    DESIGNS = (
        _design("one", ADA, 1),
        _design("two", ADA, 2, passed=True),
        _design("three", ADA, 3, passed=True),
        _design("bobs", BOB, 5, parent="two"),
        _design("cys", CY, 6, parent="one"),
    )
    TRIES = (
        _try("three", BOB, 4),
        _try("one", CY, 7),
    )

    def _kinds(self, designs_: tuple[Design, ...], tries: tuple[Try, ...]) -> dict[str, Any]:
        return {
            line.kind: line for line in milestones.designer_from(ADA, designs_, tries)
        }

    def test_each_of_the_four_is_the_first_of_its_kind(self) -> None:
        found = self._kinds(self.DESIGNS, self.TRIES)
        assert set(found) == set(milestones.DESIGNER_KINDS)
        assert (found[milestones.FIRST_DRAFT].game, found[milestones.FIRST_DRAFT].when) == (
            "one", _day(1)
        )
        assert (found[milestones.PASSES_SCREEN].game,
                found[milestones.PASSES_SCREEN].when) == ("two", _day(2))
        played = found[milestones.PLAYED_BY_OTHERS]
        assert (played.game, played.when, played.by) == ("three", _day(4), None)
        forked = found[milestones.FORKED]
        assert (forked.game, forked.when, forked.by, forked.match) == (
            "two", _day(5), BOB, "bobs"
        )
        assert all(line.seat == -1 for line in found.values())

    def test_one_line_each_however_many_games(self) -> None:
        """Test 6: nothing rewards volume. Twenty drafts, all passing, all played and
        forked by other people, still read as four lines."""
        many = tuple(_design(f"d{n}", ADA, n, passed=True) for n in range(20))
        forks = tuple(_design(f"f{n}", BOB, 30 + n, parent=f"d{n}") for n in range(20))
        tries = tuple(_try(f"d{n}", BOB, 30 + n) for n in range(20))
        lines = milestones.designer_from(ADA, many + forks, tries)
        assert sorted(line.kind for line in lines) == sorted(milestones.DESIGNER_KINDS)

    def test_their_own_plays_and_forks_are_not_somebody_elses(self) -> None:
        mine = (_design("one", ADA, 1), _design("again", ADA, 2, parent="one"))
        found = self._kinds(mine, (_try("one", ADA, 3),))
        assert set(found) == {milestones.FIRST_DRAFT}

    def test_a_game_somebody_else_designed_is_not_theirs(self) -> None:
        """Bob's game forked and played is Bob's news, even when Ada's rows are read with it."""
        rows = (_design("bobs", BOB, 1, passed=True), _design("cys", CY, 2, parent="bobs"))
        assert milestones.designer_from(ADA, rows, (_try("bobs", CY, 3),)) == ()

    def test_somebody_who_designed_nothing_has_no_lines(self) -> None:
        assert milestones.designer_from(ADA, (), ()) == ()

    def test_a_draft_that_has_not_passed_gives_no_screen_line(self) -> None:
        found = self._kinds((_design("one", ADA, 1),), ())
        assert milestones.PASSES_SCREEN not in found

    def test_the_order_rows_arrive_in_does_not_matter(self) -> None:
        forward = milestones.designer_from(ADA, self.DESIGNS, self.TRIES)
        backward = milestones.designer_from(
            ADA, tuple(reversed(self.DESIGNS)), tuple(reversed(self.TRIES))
        )
        assert forward == backward

    def test_the_lines_join_the_players_list_newest_first(self) -> None:
        reading = milestones.read_from(ADA, (), {}, {}, self.DESIGNS, self.TRIES)
        assert [line.kind for line in reading.milestones] == [
            milestones.FORKED,
            milestones.PLAYED_BY_OTHERS,
            milestones.PASSES_SCREEN,
            milestones.FIRST_DRAFT,
        ]
        assert milestones.read_from(ADA, (), {}, {}).milestones == ()

    def test_every_kind_names_its_rule(self) -> None:
        """Test 7: no hidden rule."""
        for kind in milestones.DESIGNER_KINDS:
            assert milestones.RULES[kind].endswith(".")


class TestTheMarks:
    GAME = _design("mine", ADA, 1, passed=True)

    def test_a_count_of_other_people_and_the_forkers_in_order(self) -> None:
        tries = (
            _try("mine", BOB, 2), _try("mine", BOB, 3), _try("mine", CY, 4),
            _try("mine", ADA, 5), _try("elsewhere", 9, 6),
        )
        forks = (
            _design("cys", CY, 3, parent="mine"),
            _design("bobs", BOB, 7, parent="mine"),
            _design("bobs-again", BOB, 8, parent="mine"),
            _design("adas", ADA, 9, parent="mine"),
            _design("unrelated", 9, 9, parent="other"),
        )
        marks = milestones.marks_from(self.GAME, forks, tries)
        assert marks == milestones.Marks(passes_screen=True, tried=2, forked_by=(CY, BOB))

    def test_a_game_nobody_has_touched_has_nothing_to_say(self) -> None:
        marks = milestones.marks_from(replace(self.GAME, screen_passed=False), (), ())
        assert marks == milestones.Marks(passes_screen=False, tried=0, forked_by=())


# -------------------------------------------------------------------------- the store


@pytest.fixture
def studio(drafting: Site, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:  # noqa: F811
    _screened(monkeypatch, passed=True)
    yield drafting


def _dated(site: Site, digest: str, day: int) -> None:
    """Move a kept row's `created` to a day of the test's choosing."""
    with site.store.cursor() as cursor:
        cursor.execute("update games set created = %s where hash = %s", (_day(day), digest))
    site.store.commit()


def _played(browser: Browser, digest: str) -> str:
    started = browser.post(
        "/api/session",
        {"hash": digest, "seats": ["human", AGENT], "mode": matches.SANDBOX},
    )
    assert started.status == 200, started.text
    return str(started.payload["id"])


def _lines(browser: Browser) -> dict[str, dict[str, Any]]:
    payload = browser.get("/api/profile").payload
    return {
        line["kind"]: line
        for line in payload["milestones"]
        if line["kind"] in milestones.DESIGNER_KINDS
    }


class TestTheStore:
    def test_the_profile_carries_the_four_lines(
        self, studio: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada = studio.signed_in("ada")
        bob = studio.signed_in("bob")
        _screened(monkeypatch, passed=False)
        first = _saved(ada, _template(0, "ada's first")).payload["hash"]
        _screened(monkeypatch, passed=True)
        second = _saved(ada, _template(1, "ada's second")).payload["hash"]
        _dated(studio, first, 1)
        _dated(studio, second, 2)
        assert set(_lines(ada)) == {milestones.FIRST_DRAFT, milestones.PASSES_SCREEN}

        _played(bob, second)
        fork = _saved(bob, _template(2, "bob's take"), parent=second).payload["hash"]

        found = _lines(ada)
        assert set(found) == set(milestones.DESIGNER_KINDS)
        assert (found["first-draft"]["game"], found["first-draft"]["name"]) == (
            first, "ada's first"
        )
        assert found["first-draft"]["when"].startswith("2026-09-02")
        assert found["passes-screen"]["game"] == second
        assert found["played-by-others"]["game"] == second
        assert "by" not in found["played-by-others"]
        assert (found["forked"]["game"], found["forked"]["by"]) == (second, "bob")
        assert all(line["seat"] is None for line in found.values())
        assert all(line["rule"] == milestones.RULES[kind] for kind, line in found.items())
        # Bob kept a fork, so his first draft is his; somebody else's game played is not.
        assert set(_lines(bob)) == {milestones.FIRST_DRAFT, milestones.PASSES_SCREEN}
        assert _lines(bob)["first-draft"]["game"] == fork

    def test_the_designers_own_play_and_a_visitors_count_for_nothing(
        self, studio: Site
    ) -> None:
        ada = studio.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        _played(ada, digest)
        _played(studio.browser(), digest)

        assert milestones.PLAYED_BY_OTHERS not in _lines(ada)
        marks = ada.get(f"/api/game/{digest}").payload["designer"]
        assert marks["tried"] == 0

    def test_the_game_page_carries_the_three_marks(self, studio: Site) -> None:
        ada = studio.signed_in("ada")
        bob = studio.signed_in("bob")
        cy = studio.signed_in("cy")
        digest = _saved(ada, _template()).payload["hash"]
        for person in (bob, bob, cy, ada):
            _played(person, digest)
        _saved(cy, _template(1, "cy's"), parent=digest)
        _saved(bob, _template(2, "bob's"), parent=digest)

        # Any reader sees the same marks, a visitor included.
        for reader in (ada, studio.browser()):
            marks = reader.get(f"/api/game/{digest}").payload["designer"]
            assert marks["passesScreen"] is True
            assert marks["tried"] == 2
            assert marks["forkedBy"] == ["cy", "bob"]
            assert set(marks["rules"]) == {"passesScreen", "tried", "forkedBy"}
            # D3 is a count and never who (decision B): no handle travels with it.
            assert "tried" in marks and "triedBy" not in marks

    def test_a_saved_draft_is_credited_to_its_designers_handle(self, studio: Site) -> None:
        """Q578, the first-visit study's F9. The page credited a saved draft to *a design
        run here … found in round 0*; the store knows whose it is, and says so to every
        reader. The rules carry nothing new, so the hash the save answered still names
        the game."""
        ada = studio.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        for reader in (ada, studio.browser()):
            payload = reader.get(f"/api/game/{digest}").payload
            assert payload["designer"]["by"] == "ada"
            # What the page reads to know this is a person's game and not a run's.
            assert payload["source"] == "designed"
            assert payload["hash"] == digest

    def test_a_designer_who_left_is_credited_as_a_former_player(self, studio: Site) -> None:
        ada = studio.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        with studio.store.cursor() as cursor:
            cursor.execute(
                "update users set handle = null where id = %s", (_user_id(studio, "ada"),)
            )
        studio.store.commit()
        marks = studio.browser().get(f"/api/game/{digest}").payload["designer"]
        assert marks["by"] == identity.FORMER

    def test_a_game_nobody_designed_here_has_no_marks(self, studio: Site) -> None:
        from test_site_admin import DULL_HASH

        studio.game()
        assert studio.browser().get(f"/api/game/{DULL_HASH}").payload["designer"] is None

    def test_somebody_who_left_is_named_as_a_former_player(self, studio: Site) -> None:
        ada = studio.signed_in("ada")
        bob = studio.signed_in("bob")
        digest = _saved(ada, _template()).payload["hash"]
        _saved(bob, _template(1, "bob's"), parent=digest)
        with studio.store.cursor() as cursor:
            cursor.execute(
                "update users set handle = null where id = %s", (_user_id(studio, "bob"),)
            )
        studio.store.commit()

        assert _lines(ada)["forked"]["by"] == identity.FORMER
        marks = ada.get(f"/api/game/{digest}").payload["designer"]
        assert marks["forkedBy"] == [identity.FORMER]

    def test_the_readers_ask_only_designed_rows(self, studio: Site) -> None:
        """A game loaded from the tree is not a design: `around` has nothing for it, and a
        fork of it has no designer upstream to tell."""
        studio.game()
        from test_site_admin import DULL_HASH

        ada = studio.signed_in("ada")
        fork = _saved(ada, _template(), parent=DULL_HASH).payload["hash"]
        assert designs.around(studio.store, DULL_HASH) == (None, ())
        game, forks = designs.around(studio.store, fork)
        assert game is not None and game.parent == DULL_HASH and forks == ()
        assert [d.hash for d in designs.of(studio.store, _user_id(studio, "ada"))] == [fork]


# -------------------------------------------------------------------------- the pages


class TestTheProfileDraws:
    LINES = [
        {"kind": "forked", "rule": "The first fork.", "when": "2026-09-14T12:00:00+00:00",
         "game": "aaa", "name": "alpha", "seat": None, "by": "bob"},
        {"kind": "played-by-others", "rule": "The first play.",
         "when": "2026-09-13T12:00:00+00:00", "game": "aaa", "name": "alpha", "seat": None},
        {"kind": "first-draft", "rule": "The first draft.",
         "when": "2026-09-12T12:00:00+00:00", "game": "aaa", "name": "alpha", "seat": None},
    ]

    def test_a_designer_line_has_no_seat_and_the_fork_says_who(self) -> None:
        said = _done(self.LINES)
        assert "Built on by somebody else, on alpha, by bob, 2026-09-14." in said
        assert "Played by somebody else, on alpha, 2026-09-13." in said
        assert "First draft kept, on alpha, 2026-09-12." in said
        assert "playing" not in said
        assert "The first play." in said


class TestTheGamePageDraws:
    RULES = server.DESIGNER_RULES

    def _drawn(self, marks: dict[str, Any] | None) -> str:
        game = {"level": "Draft", "levelSays": "", "track": [], "designer": marks}
        return " ".join(_in_page("", f"whereHtml({json.dumps(game)})").split())

    def test_each_mark_with_the_sentence_that_says_what_it_reads(self) -> None:
        drawn = self._drawn(
            {"passesScreen": True, "tried": 3, "forkedBy": ["bob", "<cy>"], "rules": self.RULES}
        )
        assert "<b>Passes the playability check</b>" in drawn  # G32
        assert "<b>3</b> people have tried this" in drawn
        assert "Built on by <b>bob</b>, <b>&lt;cy&gt;</b>" in drawn  # G33
        for sentence in self.RULES.values():
            assert sentence in drawn

    def test_one_person_is_one_person(self) -> None:
        drawn = self._drawn({"passesScreen": False, "tried": 1, "forkedBy": [], "rules": {}})
        assert "<b>1</b> person has tried this" in drawn

    def test_a_designed_game_carries_a_copy_link(self) -> None:
        """Q578, the study's B3: a draft's link is the one way anybody else reaches it,
        and its page had no control to send it."""
        game = {"hash": "abc123", "source": "designed", "level": "Draft", "levelSays": "",
                "track": [], "designer": None}
        drawn = _in_page("", f"whereHtml({json.dumps(game)})")
        assert 'data-copy-link="abc123"' in drawn and ">copy link</button>" in drawn
        committed = {**game, "source": "committed", "level": "Baseline"}
        assert "data-copy-link" not in _in_page("", f"whereHtml({json.dumps(committed)})")

    def test_the_credit_is_the_designers_and_not_the_machinery(self) -> None:
        """F9: *designed by ‹handle›*, and no design run or round on a person's game."""
        lineage = {"origin": "evolved", "run": "designed", "generation": 0, "parents": []}
        # The panel wires its parents' buttons a tick later, against a real view; the
        # stub has none, so the tick is not run.
        quiet = "globalThis.setTimeout = () => 0;"
        game = {"source": "designed", "designer": {"by": "<ada>"}}
        drawn = _in_page(quiet, f"provenanceHtml({json.dumps(lineage)}, {json.dumps(game)})")
        assert "designed by &lt;ada&gt;" in drawn
        for machinery in ("design run", "found in round", "invented by a design run"):
            assert machinery not in drawn
        # The local application has no accounts: where it was made, still no run.
        local = _in_page(
            quiet, f"provenanceHtml({json.dumps(lineage)}, {json.dumps({'source': 'designed'})})"
        )
        assert "built in the designer here" in local and "found in round" not in local
        # A design run's own candidate keeps its run and its round.
        found = _in_page(
            quiet, f"provenanceHtml({json.dumps(lineage)}, {json.dumps({'source': 'evolved'})})"
        )
        assert "invented by a design run here" in found and "found in round" in found

    def test_the_site_draws_no_maintainers_record(self) -> None:
        """G39 (W7, Q614): a provenance's `history` and `notes` are drawn locally and
        never on the site, which does not send them either (`server.visitors_provenance`)."""
        lineage = {"origin": "human", "history": "Made from runs/abc 15e3eb35bfbc.",
                   "notes": "How it went in schema 9."}
        call = f"provenanceHtml({json.dumps(lineage)}, {json.dumps({'source': 'committed'})})"
        quiet = "globalThis.setTimeout = () => 0;"
        site = _in_page(quiet + "ACCOUNTS = true;", call)
        assert "runs/abc" not in site and "schema 9" not in site
        local = _in_page(quiet, call)
        assert "runs/abc" in local and "schema 9" in local

    def test_nothing_to_say_draws_nothing(self) -> None:
        """Test 5: a game nobody has tried reads as nothing, never as nought."""
        for marks in (None, {"passesScreen": False, "tried": 0, "forkedBy": [], "rules": {}}):
            drawn = self._drawn(marks)
            assert "designerMarks" not in drawn
            assert "tried this" not in drawn and "Forked by" not in drawn


class TestNoDarkPatterns:
    """The seven tests of `docs/reference/no-dark-patterns.md`, where D1 to D4 could fail them."""

    def test_no_clock_is_read_so_nothing_lapses(self) -> None:
        """Tests 1 and 2: the reading takes no *now*, so a person away a year reads what
        they read the day they left, and nothing counts down."""
        import inspect

        for reader in (milestones.designer_from, milestones.marks_from):
            source = inspect.getsource(reader)
            assert "now(" not in source and "today" not in source

    def test_no_page_says_how_many_lines_there_are_to_get(self) -> None:
        """Test 5 and 6: the page draws what was reached, never a tally of designs, a
        total or how close anything else is."""
        drawn = _done(TestTheProfileDraws.LINES)
        assert "of 4" not in drawn and "designs" not in drawn
