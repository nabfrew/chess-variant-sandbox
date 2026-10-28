"""The showcase: a member's page as others see it (UP4, `docs/archive/plan-user-page.md` §5).

Choice D1: members only, as before, now with each published game's board, the first
sentence of its description, its level, its votes with their count, how many people have
tried it and *Challenge ‹handle› to it*; favourites as small boards; and, in direction B,
*as members see it* on `/profile` showing exactly that page's sections.

The two claims the box names: **a draft is never on it** (not in the reading, not on the
page, not in either page's board data), and **the sections are `members.read`'s** (the
profile's members' view and `/players/<handle>` are one drawing, `pages._showcase`, of one
reading). Proved on the `dull` specs of `test_site_matches.py`, not chess. The store tests
need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as every site test does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_site import Site, _joined, _running_site
from test_site_matches import LADDER, SHORT, SHORT_HASH, UNRATED, UNRATED_HASH, Scratch, _scratch
from test_site_members import _make_designed

from chess_variant_sandbox.site import members, pages, votes
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.web import library

IDEA = "Two pieces race to the far side. Nothing else happens."


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


def _known(*pairs: tuple[str, dict[str, Any]]) -> dict[str, library.Game]:
    """The listing as `members.read` is handed it: each game with its rules and a
    description of two sentences."""
    return {
        digest: library.Game(
            hash=digest, name=str(spec["name"]), description=IDEA, path=Path("x.json"),
            source="designed", spec=spec,
        )
        for digest, spec in pairs
    }


KNOWN = _known((SHORT_HASH, SHORT), (UNRATED_HASH, UNRATED))


def _headings(html: str) -> list[str]:
    """The `<h2>` headings inside the showcase's own block, in order."""
    return re.findall(r"<h2>(.*?)</h2>", _showcase_of(html))


def _showcase_of(html: str) -> str:
    """The showcase's own block, from its opening `<div>` to the one that closes it."""
    start = html.index('<div class="showcase">')
    depth = 0
    for found in re.finditer(r"<div\b|</div>", html[start:]):
        depth += 1 if found.group() == "<div" else -1
        if depth == 0:
            return html[start:start + found.end()]
    raise AssertionError("the showcase is never closed")


class TestTheReading:
    def test_each_game_they_made_has_its_board_its_idea_and_its_votes(
        self, store: Scratch
    ) -> None:
        ada, bea, cy = store.user("ada"), store.user("bea"), store.user("cy")
        _make_designed(store, UNRATED_HASH, ada)
        votes.cast(store.connection, bea, UNRATED_HASH, 5)
        votes.cast(store.connection, cy, UNRATED_HASH, 4)
        store.connection.commit()
        reading = members.read(store.connection, "ada", bea, KNOWN)
        assert reading is not None
        (game,) = reading["designed"]
        assert game["still"] == library.still(KNOWN[UNRATED_HASH]) and game["still"]
        assert game["description"] == "Two pieces race to the far side."
        assert game["stars"] == {"mean": 4.5, "count": 2}
        # Every drawing the boards name is sent once.
        stems = {icon["stem"] for pair in game["still"]["icons"] for icon in pair if icon}
        assert stems <= set(reading["pieceArt"])

    def test_a_game_kept_only_in_the_store_is_drawn_from_its_stored_rules(
        self, store: Scratch, tmp_path: Path
    ) -> None:
        """A member's game is not in the tree's listing: its board and idea come from the
        store's copy of its rules, as the workshop's do."""
        ada, bea = store.user("ada"), store.user("bea")
        _make_designed(store, UNRATED_HASH, ada)
        reading = members.read(store.connection, "ada", bea, {}, tmp_path)
        assert reading is not None
        (game,) = reading["designed"]
        assert game["still"] is not None and game["still"]["width"] == UNRATED["board"]["width"]

    def test_nobody_voted_is_no_stars_not_a_nought(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _make_designed(store, UNRATED_HASH, ada)
        reading = members.read(store.connection, "ada", bea, KNOWN)
        assert reading is not None and reading["designed"][0]["stars"] is None
        page = pages.member(reading, "bea", viewer=bea)
        assert "★" not in page and not re.search(r"\b0 votes?\b", page)

    def test_a_favourite_has_its_board(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        votes.cast(store.connection, ada, SHORT_HASH, 5)
        members.set_favourites(store.connection, ada, True)
        store.connection.commit()
        reading = members.read(store.connection, "ada", bea, KNOWN)
        assert reading is not None
        assert reading["favourites"][0]["still"] == library.still(KNOWN[SHORT_HASH])
        page = pages.member(reading, "bea", viewer=bea)
        favs = page[page.index('<div class="showCards favs">'):]
        assert f'data-still="{SHORT_HASH}"' in favs


class TestADraftIsNeverOnIt:
    def test_not_in_the_reading_the_page_or_the_board_data(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _make_designed(store, UNRATED_HASH, ada, public=False)
        _make_designed(store, SHORT_HASH, ada)
        # Even a draft starred five times by its own maker, with favourites shown.
        votes.cast(store.connection, ada, UNRATED_HASH, 5)
        members.set_favourites(store.connection, ada, True)
        store.connection.commit()
        for viewer in (bea, ada):
            reading = members.read(store.connection, "ada", viewer, KNOWN)
            assert reading is not None
            assert UNRATED_HASH not in json.dumps(reading, default=str)
            assert UNRATED_HASH not in pages.member(reading, "x", viewer=viewer)
            assert set(pages._showcase_data(reading)["stills"]) == {SHORT_HASH}
        own = members.read(store.connection, "ada", ada, KNOWN)
        profile = pages.profile("ada", user=ada, showcase=own)
        assert UNRATED_HASH not in profile


class TestTheSectionsAreMembersReads:
    def test_the_profiles_members_view_is_the_member_pages_sections(
        self, store: Scratch
    ) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _make_designed(store, SHORT_HASH, ada)
        votes.cast(store.connection, ada, UNRATED_HASH, 5)
        members.set_favourites(store.connection, ada, True)
        store.connection.commit()
        theirs = members.read(store.connection, "ada", bea, KNOWN)
        own = members.read(store.connection, "ada", ada, KNOWN)
        assert theirs is not None and own is not None
        page = pages.member(theirs, "bea", viewer=bea)
        profile = pages.profile("ada", user=ada, showcase=own)
        assert _headings(page) == _headings(profile) == [
            "Games ada made", "What ada has done", "Favourite games",
        ]
        # The same drawing, less the buttons nobody can press on their own games.
        without = re.sub(r'<div class="row"><button type="button" class="askTo".*?</div>',
                         "", _showcase_of(page))
        assert without == _showcase_of(profile)
        assert 'id="members-view" hidden' in profile
        held = profile.index('id="page-data">') + len('id="page-data">')
        data = json.loads(profile[held: profile.index("</script>", held)])
        assert data["SHOWCASE"] == pages._showcase_data(own)

    def test_challenge_them_to_it_for_another_member_only(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        _make_designed(store, SHORT_HASH, ada)
        theirs = members.read(store.connection, "ada", bea, KNOWN)
        own = members.read(store.connection, "ada", ada, KNOWN)
        assert theirs is not None and own is not None
        asked = f'data-challenge-game="{SHORT_HASH}">Challenge ada to it</button>'
        assert asked in pages.member(theirs, "bea", viewer=bea)
        assert "Challenge ada to it" not in pages.member(own, "ada", viewer=ada)
        profile = pages.profile("ada", user=ada, showcase=own)
        assert "to it</button>" not in profile
        assert "Another member also sees <em>Challenge ada</em>" in profile

    def test_no_members_view_for_somebody_without_a_page(self) -> None:
        profile = pages.profile("ada", user=1)
        assert 'id="members-view"' not in profile and '"SHOWCASE": null' in profile


class TestThePageDrawsTheBoards:
    STILL = (site_pages.SCRIPT_DIR / "still.js").read_text(encoding="utf-8")
    MEMBER = (site_pages.SCRIPT_DIR / "member.js").read_text(encoding="utf-8")

    def test_the_member_page_loads_the_still_board_then_still_js_then_its_own(self) -> None:
        page = pages.member(
            {"handle": "ada", "user": 1, "since": "October 2026", "own": False}, "bea"
        )
        at = [page.index(f"/static/page/{name}") for name in site_pages.BOARD_SCRIPTS]
        at += [page.index("/static/site/still."), page.index("/static/site/member.")]
        assert at == sorted(at)

    def test_a_board_is_the_still_boards_drawing_put_in_the_servers_box(self) -> None:
        assert "fillStills(" in self.MEMBER and "boardPreview" not in self.MEMBER
        assert "innerHTML" not in self.STILL + self.MEMBER
        assert self.STILL.count("createContextualFragment(") == 1
        for word in ("squareXY", "squareName", "% preview.width"):
            assert word not in self.STILL, word


# ----------------------------------------------------------------------- over HTTP


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


class TestOverHTTP:
    def test_profile_carries_the_switch_and_the_same_sections(self, site: Site) -> None:
        ada = _joined(site, "ada")
        bea = _joined(site, "bea")
        profile = ada.get("/profile").text
        page = bea.get("/players/ada").text
        assert 'id="members-view" hidden' in profile
        assert _headings(profile) == _headings(page)
