"""A member's page and the vote's fixed tags (Q596 and Q599, `docs/plan-return.md` R1 and R4).

**Q596.** `/players/<handle>` is read by signed-in members only: a visitor is sent to the
door whatever the handle, another member reads §6.2's sections, the owner reads the same
page with one line pointing at their own, and a handle with no page (never taken, left,
shut out) is the site's not-found page. A draft never shows, five-star games show only
when their voter ticked the box, and the page holds no rank, no position and no *last
seen*. Proved on specs rather than chess (rule 5 of CLAUDE.md): the `dull` variants of
`test_site_matches.py`.

**Q599.** The vote holds no text from its voter (the user, 2026-10-02: *votes only, no
comments*): the route refuses a note and any field besides the game, the stars and the
tags; a tag is one of eight fixed keys, at most three; and every string the store holds
for a vote is one of this module's own. A vote moves no score.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as
every site test does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from test_site import Site, _joined, _running_site
from test_site_matches import (
    LADDER,
    SHORT,
    SHORT_HASH,
    UNRATED,
    UNRATED_HASH,
    Scratch,
    _scratch,
)

from chess_variant_sandbox.site import admin, members, pages, routes, votes
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.web.page import marks

# ------------------------------------------------------------------------- the store


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store with one rated game (a loaded ladder), one practice-only game, nobody yet."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


def _play_out(store: Scratch, match_id: str) -> None:
    """Both players send the engine's first offered move; the short game then ends."""
    for _ in range(2):
        found = m.row(store.connection, match_id)
        snapshot = m.session(store.connection, match_id).snapshot()
        who = found.player(snapshot["sideToMove"]).user
        c.move(store.connection, match_id, who, int(snapshot["moves"][0]["move"]))
    store.connection.commit()


def _designed(store: Scratch, owner: int, name: str, public: bool) -> str:
    """A designed game of `owner`'s: published (on trial) or a draft. Its spec is the
    practice game's under a new name, which is all a page about it reads."""
    digest = f"d{abs(hash((owner, name))) % 10**11:011d}"
    with store.connection.cursor() as cursor:
        cursor.execute(
            "insert into games (hash, spec, name, origin, owner, visibility, published)"
            " values (%s, '{}'::jsonb, %s, 'designed', %s, %s,"
            " case when %s then now() end)",
            (digest, name, owner, "public" if public else "unlisted", public),
        )
    store.connection.commit()
    return digest


def _page(store: Scratch, handle: str, viewer: int) -> str:
    reading = members.read(store.connection, handle, viewer)
    assert reading is not None, handle
    return pages.member(reading, "reader", viewer=viewer)


class TestWhoHasAPage:
    def test_a_member_has_one_and_nobody_else_does(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        assert members.read(store.connection, "ada", bea) is not None
        # Matched without regard to case, as signing in is; named as the handle is held.
        assert members.read(store.connection, "ADA", bea)["handle"] == "ada"  # type: ignore[index]
        assert members.read(store.connection, "nobody-here", bea) is None
        # Not a handle at all: never reaches the store.
        assert members.read(store.connection, "a b<script>", bea) is None
        admin.kill_user(store.connection, ada)
        store.connection.commit()
        assert members.read(store.connection, "ada", bea) is None

    def test_somebody_who_left_has_none(self, store: Scratch) -> None:
        store.user("ada")
        bea = store.user("bea")
        admin.forget(store.connection, "ada")
        store.connection.commit()
        assert members.read(store.connection, "ada", bea) is None
        assert members.read(store.connection, "a former player", bea) is None


class TestWhatAnotherMemberReads:
    def test_the_header(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        page = _page(store, "ada", bea)
        assert f"<h1>{marks.svg(ada, 40)}ada</h1>" in page
        assert re.search(r"Member since [A-Z][a-z]+ \d{4}\.", page)
        assert 'id="challenge-open">Challenge ada</button>' in page
        assert "This is your page as other members see it" not in page
        # Empty sections are absent, not headings over nothing (test 5).
        for heading in ("Recently", "Games ada designed", "What ada has done",
                        "Favourite games", "Plays most with"):
            assert f"<h2>{heading}</h2>" not in page

    def test_a_finished_game_between_people_and_who_they_play(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _play_out(store, c.start(store.connection, SHORT_HASH, [ada, bea]).id)
        reading = members.read(store.connection, "ada", bea)
        assert reading is not None
        games = [event for event in reading["recent"] if event["kind"] == "game"]
        assert len(games) == 1
        event = games[0]
        assert (event["game"], event["between"], event["rated"]) == (SHORT_HASH, True, True)
        assert event["against"] == {"person": "bea", "id": bea, "mark": marks.mark(bea)}
        assert reading["playsWith"]["people"] == [
            {"handle": "bea", "id": bea, "mark": marks.mark(bea)}
        ]
        assert reading["finished"] == "Finished the one correspondence game played."
        page = pages.member(reading, "bea", viewer=bea)
        assert '<a href="/players/bea">' in page and "· correspondence, rated ·" in page
        assert "<h2>Plays most with</h2>" in page
        # A fact about a person, never a standing: no count beside a name, no place.
        section = page[page.index("<h2>Plays most with</h2>"):]
        assert not re.search(r"\(\d+", section)

    def test_a_draft_never_shows(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _designed(store, ada, "Secret draft", public=False)
        shown = _designed(store, ada, "Out on trial", public=True)
        page = _page(store, "ada", bea)
        assert "Secret draft" not in page
        assert "<h2>Games ada designed</h2>" in page and "Out on trial" in page
        assert f'href="/#/game/{shown}"' in page and "· On trial" in page
        assert "went on trial" in page

    def test_a_cancelled_game_is_no_event(
        self, store: Scratch
    ) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        started = c.start(store.connection, UNRATED_HASH, [ada, bea])
        c.cancel(store.connection, started.id, ada)
        store.connection.commit()
        reading = members.read(store.connection, "ada", bea)
        assert reading is not None and reading["recent"] == []

    def test_five_star_games_only_by_their_choice(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        votes.cast(store.connection, ada, SHORT_HASH, 5)
        votes.cast(store.connection, ada, UNRATED_HASH, 4)
        store.connection.commit()
        assert members.shows_favourites(store.connection, ada) is False
        assert "<h2>Favourite games</h2>" not in _page(store, "ada", bea)
        assert members.set_favourites(store.connection, ada, True) is True
        store.connection.commit()
        reading = members.read(store.connection, "ada", bea)
        assert reading is not None
        assert reading["favourites"] == [{"game": SHORT_HASH, "name": SHORT["name"]}]
        assert "<h2>Favourite games</h2>" in pages.member(reading, "bea", viewer=bea)

    def test_no_rank_no_place_no_last_seen(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _play_out(store, c.start(store.connection, SHORT_HASH, [ada, bea]).id)
        page = _page(store, "ada", bea)
        page = page[page.index("<main"):].lower()
        for word in ("rank", "last seen", "position", "points", "streak"):
            assert word not in page, word

    def test_the_rules_are_about_the_person_not_the_reader(self) -> None:
        from chess_variant_sandbox.site import milestones

        for rule in milestones.RULES.values():
            assert " you " not in f" {pages._theirs(rule)} ".replace("you designed", ""), rule


class TestTheOwner:
    def test_their_own_page_points_at_their_settings_and_offers_no_challenge(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        reading = members.read(store.connection, "ada", ada)
        assert reading is not None and reading["own"] is True
        assert reading["challenge"] is None
        page = pages.member(reading, "ada", viewer=ada)
        assert "This is your page as other members see it" in page
        assert '<a href="/profile">Your own page</a>' in page
        assert "Challenge ada" not in page
        # The mark is drawn in the header: the corner, and the heading.
        assert page.count(marks.svg(ada, 20)) == 1 and marks.svg(ada, 40) in page

    def test_the_settings_box_is_off_by_default(self, store: Scratch) -> None:
        ada = store.user("ada")
        box = members.favourites_box(store.connection, ada)
        assert box == {
            "on": False, "said": members.FAVOURITES_SAID, "under": members.FAVOURITES_UNDER
        }


# ----------------------------------------------------------------------- over HTTP


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


class TestTheThreeViewers:
    """The page as a visitor, another member and the owner get it, and a handle with
    no page, over HTTP with the cookie (the reading is the store's; this is the door)."""

    def test_a_visitor_is_sent_to_the_door_whatever_the_handle(self, site: Site) -> None:
        _joined(site, "ada")
        visitor = site.browser()
        for handle in ("ada", "nobody-here"):
            answer = visitor.get(f"/players/{handle}")
            assert answer.status == 303, handle
            assert answer.headers["Location"] == "/signin"
            assert "ada" not in answer.text

    def test_another_member_reads_the_page(self, site: Site) -> None:
        _joined(site, "ada")
        bea = _joined(site, "bea")
        answer = bea.get("/players/ada")
        assert answer.status == 200
        assert "<h1>" in answer.text and "ada</h1>" in answer.text
        # A rated game to name, or the way to one: either way, the challenge is offered.
        assert "hallenge ada" in answer.text
        assert "ada@example.com" not in answer.text
        assert "This is your page as other members see it" not in answer.text

    def test_the_owner_reads_their_own(self, site: Site) -> None:
        ada = _joined(site, "ada")
        answer = ada.get("/players/ada")
        assert answer.status == 200
        assert "This is your page as other members see it" in answer.text
        assert "Challenge ada" not in answer.text

    def test_a_handle_with_no_page_is_the_not_found_page(self, site: Site) -> None:
        bea = _joined(site, "bea")
        lost = pages.not_found()
        for path in ("/players/nobody-here", "/players/", "/players/a%20b", "/players/x/y"):
            answer = bea.get(path)
            assert answer.status == 404, path
            assert answer.text == lost, path

    def test_your_own_page_carries_the_way_to_it_and_the_box(self, site: Site) -> None:
        ada = _joined(site, "ada")
        profile = ada.get("/api/profile").payload
        assert profile["page"] == "/players/ada"
        assert profile["favourites"]["on"] is False
        ticked = ada.post("/api/favourites", {"on": True})
        assert ticked.status == 200 and ticked.payload["on"] is True
        assert ada.get("/api/profile").payload["favourites"]["on"] is True

    def test_a_visitor_cannot_tick_anybodys_box(self, site: Site) -> None:
        answer = site.browser().post("/api/favourites", {"on": True})
        assert answer.status in (400, 401)

    def test_the_route_and_the_prefix_are_registered(self) -> None:
        assert routes.MEMBERS == members.PATH == "/players/"
        assert routes.find("POST", "/api/favourites") is not None

    def test_the_privacy_notice_says_what_members_see(self) -> None:
        notice = pages.privacy()
        assert "<h2>What other members see</h2>" in notice
        assert "never a draft" in notice and "Visitors are sent to the sign-in page" in notice
        assert "only if you tick the box" in notice


# ------------------------------------------------------------------ Q599, the vote


def _vote_rows(store: Any) -> list[tuple[Any, ...]]:
    with store.cursor() as cursor:
        cursor.execute("select tags, note from votes")
        return list(cursor.fetchall())


class TestTheVoteHoldsNoText:
    def test_a_note_or_any_other_field_is_refused_and_nothing_is_written(
        self, site: Site
    ) -> None:
        ada = _joined(site, "ada")
        digest = _a_game(site)
        for extra in ({"note": "lovely"}, {"comment": "x"}, {"message": "to the designer"},
                      {"text": ""}):
            answer = ada.post("/api/vote", {"hash": digest, "stars": 4, **extra})
            assert answer.status == 400, extra
            assert "no text" in answer.payload["error"]
        assert _vote_rows(site.store) == []

    def test_a_tag_must_be_one_of_the_fixed_eight_and_three_at_most(
        self, site: Site
    ) -> None:
        ada = _joined(site, "ada")
        digest = _a_game(site)
        for tags in (["not a tag"], ["again", "Great game, well done!"], "again",
                     ["again", "clever", "new", "fast"]):
            answer = ada.post("/api/vote", {"hash": digest, "stars": 4, "tags": tags})
            assert answer.status == 400, tags
        assert _vote_rows(site.store) == []

    def test_every_stored_string_is_the_sites_own(self, site: Site) -> None:
        ada = _joined(site, "ada")
        bea = _joined(site, "bea")
        digest = _a_game(site)
        cast = ada.post("/api/vote", {"hash": digest, "stars": 5, "tags": ["new", "again"]})
        assert cast.status == 200, cast.text
        assert cast.payload["tags"] == ["again", "new"]
        assert bea.post("/api/vote", {"hash": digest, "stars": 3, "tags": ["new"]}).status == 200
        rows = _vote_rows(site.store)
        assert len(rows) == 2
        for tags, note in rows:
            assert note is None
            assert set(tags) <= set(votes.TAGS)
        counts = site.app.get(f"/api/game/{digest}")["standing"]["tagCounts"]
        assert counts == [
            {"tag": "again", "words": votes.TAGS["again"], "count": 1},
            {"tag": "new", "words": votes.TAGS["new"], "count": 2},
        ]

    def test_the_table_refuses_a_note_whoever_writes_it(self, site: Site) -> None:
        _joined(site, "ada")
        digest = _a_game(site)
        with site.store.cursor() as cursor:
            cursor.execute("select id from users where handle = 'ada'")
            ada = int(cursor.fetchone()[0])
        for columns, values in (("note", "'hello'"), ("tags", "array['free text']")):
            with pytest.raises(Exception, match="votes_"), site.store.cursor() as cursor:
                cursor.execute(
                    f"insert into votes (voter, game, stars, {columns})"
                    f" values (%s, %s, 3, {values})",
                    (ada, digest),
                )
            site.store.rollback()

    def test_the_tags_are_the_plans_words(self) -> None:
        """`docs/plan-understanding.md` §6's eight, in its words and order."""
        assert list(votes.TAGS.values()) == [
            "I'd play it again", "a clever rule", "first player has it easy",
            "lots of draws", "goes on too long", "over too fast", "hard to learn",
            "feels new",
        ]
        assert votes.MOST_TAGS == 3

    def test_a_vote_moves_no_score(self, site: Site) -> None:
        """Rule 5 of the vote (`site/votes.py`): stars and tags are an opinion, kept
        apart from the measurement. The game's index is the same before and after."""
        ada = _joined(site, "ada")
        digest = _a_game(site)
        before = site.app.get(f"/api/game/{digest}")
        ada.post("/api/vote", {"hash": digest, "stars": 1, "tags": ["long", "draws"]})
        after = site.app.get(f"/api/game/{digest}")
        # Everything the game page is sent but the people's panel, where the stars live.
        assert {k: v for k, v in before.items() if k != "standing"} == {
            k: v for k, v in after.items() if k != "standing"
        }
        assert after["standing"]["votes"] == 1


def _a_game(site: Site) -> str:
    """A game the store holds, for a vote to be about."""
    with site.store.cursor() as cursor:
        cursor.execute(
            "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)"
            " on conflict do nothing",
            (SHORT_HASH, json.dumps(SHORT), SHORT["name"], "corpus"),
        )
    site.store.commit()
    return SHORT_HASH

