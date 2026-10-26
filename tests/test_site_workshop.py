"""What the user page needs, from the server (UP1, `docs/plan-user-page.md` §5).

`/api/profile` sends, per game a person made, its still, its track, its rungs with what is
known of each, its parent's name, level and score with that score's basis, and the change
from the parent in the designer's words; per waiting game, the position as it stands and
its last move. `site/workshop.py` reads them; the page works nothing out.

The claims are the box's four, plus the waiting games': a draft's rungs and level are the
ones `/api/design/standing/<hash>` gives it; the change is the designer's for the same
pair; a provisional parent score carries its mark; a member is sent only their own games;
and a waiting game's board and last move are the session's own.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from test_site_admin import DULL, DULL_HASH, Site, _saved, _screened, _template, drafting
from test_site_correspondence import _as, _move, _pair, app, store
from test_site_matches import AGENT, SHORT_HASH, Scratch

from chess_variant_sandbox.site import challenges, matches, stages, workshop
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import candidacy, library

__all__ = ["app", "drafting", "store"]


def _made_from_dull(name: str) -> dict[str, Any]:
    """`dull` with one rule changed, so the change from its parent is one sentence or so."""
    spec = copy.deepcopy(DULL)
    spec["name"] = name
    spec["rules"]["max_plies"] = 80
    return spec


@pytest.fixture
def workshop_site(drafting: Site, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """The drafting site with `dull`'s row, so a draft made from it keeps its parent."""
    _screened(monkeypatch, passed=True)
    drafting.game()
    yield drafting


def _games(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {game["hash"]: game for group in payload["yourGames"] for game in group["games"]}


class TestEachGameAMemberMade:
    def test_a_drafts_rungs_and_level_are_the_designers_standing(
        self, workshop_site: Site
    ) -> None:
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _made_from_dull("ada's dull"), parent=DULL_HASH).payload["hash"]
        payload = ada.get("/api/profile").payload
        group = next(g for g in payload["yourGames"] if g["games"][0]["hash"] == digest)
        game = group["games"][0]
        standing = ada.get(f"/api/design/standing/{digest}").payload

        assert group["level"] == standing["level"] == stages.DRAFT
        assert game["track"] == standing["track"]
        assert [(rung["key"], rung["name"]) for rung in game["rungs"]] == [
            (rung["key"], {**rung, **rung["site"]}["name"]) for rung in standing["rungs"]
        ]
        states = {rung["key"]: rung["state"] for rung in game["rungs"]}
        # The save's check passed; no score from the designer's computer is attached.
        assert states == {"validate": "done", "smoke": "done", "screen": "done",
                          "save": "done", "canonical": "ahead", "nominate": "ahead"}
        assert [rung["key"] for rung in game["rungs"] if rung["next"]] == ["canonical"]
        # What the designer's publish ticks say of the check, the profile says too.
        tick = standing["publish"]["ticks"][0]["says"]
        assert next(r for r in game["rungs"] if r["key"] == "screen")["says"] == tick

    def test_a_check_that_did_not_pass_fails_its_rung_and_guesses_no_other(
        self, workshop_site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _screened(monkeypatch, passed=False, reasons=("nobody wins",))
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _template(0, "ada's stuck one")).payload["hash"]
        rungs = _games(ada.get("/api/profile").payload)[digest]["rungs"]
        states = {rung["key"]: rung["state"] for rung in rungs}
        assert states["screen"] == "failed"
        assert states["smoke"] == "ahead"
        assert next(r for r in rungs if r["key"] == "smoke")["says"] == workshop.SCREEN_UNKNOWN
        assert [rung["key"] for rung in rungs if rung["next"]] == ["smoke"]

    def test_the_still_is_the_listings_own_with_its_drawings_sent_once(
        self, workshop_site: Site
    ) -> None:
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _made_from_dull("ada's dull")).payload["hash"]
        payload = ada.get("/api/profile").payload
        still = _games(payload)[digest]["still"]
        game = workshop_site.app.known()[digest]
        assert still == library.still(game)
        stems = {icon["stem"] for pair in still["icons"] for icon in pair if icon}
        assert stems and stems <= set(payload["pieceArt"])

    def test_the_change_is_the_designers_for_the_same_pair(self, workshop_site: Site) -> None:
        ada = workshop_site.signed_in("ada")
        spec = _made_from_dull("ada's dull")
        digest = _saved(ada, spec, parent=DULL_HASH).payload["hash"]
        game = _games(ada.get("/api/profile").payload)[digest]
        checked = ada.post("/api/design/check", {"spec": spec, "parent": DULL_HASH}).payload

        assert game["change"] == checked["changes"]
        assert game["change"], "a changed rule is a sentence"
        made_from = game["parent"]
        assert (made_from["hash"], made_from["name"], made_from["level"]) == (
            DULL_HASH, DULL["name"], stages.BASELINE
        )
        assert made_from["levelGloss"] == stages.GLOSSES[stages.BASELINE]
        # `dull` here has no measurement, so no number at all rather than a bare one.
        assert made_from["score"] is None

    def test_a_game_made_from_nothing_has_no_parent_and_no_change(
        self, workshop_site: Site
    ) -> None:
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _template(0, "ada's own")).payload["hash"]
        game = _games(ada.get("/api/profile").payload)[digest]
        assert game["parent"] is None and game["change"] is None

    def test_a_member_is_sent_only_their_own_games(self, workshop_site: Site) -> None:
        ada = workshop_site.signed_in("ada")
        bob = workshop_site.signed_in("bob")
        hers = _saved(ada, _made_from_dull("ada's dull"), parent=DULL_HASH).payload["hash"]
        his = _saved(bob, _template(1, "bob's draft")).payload["hash"]

        theirs = bob.get("/api/profile")
        assert set(_games(theirs.payload)) == {his}
        assert hers not in theirs.text and "ada's dull" not in theirs.text
        assert set(_games(ada.get("/api/profile").payload)) == {hers}

    def test_the_workshop_stands_each_game_in_its_column_with_its_next_step(
        self, workshop_site: Site
    ) -> None:
        """UP2: the page's columns are the server's, and so is each card's next step."""
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _made_from_dull("ada's dull"), parent=DULL_HASH).payload["hash"]
        payload = ada.get("/api/profile").payload
        game = _games(payload)[digest]
        assert game["column"] == stages.DRAFT
        assert game["step"] == workshop.step(
            workshop.kept_rows(workshop_site.store, _owner(workshop_site, digest))[digest],
            game["next"])
        columns = payload["workshop"]
        assert [column["word"] for column in columns] == list(workshop.COLUMNS)
        assert columns[0]["games"] == [digest]
        # Nobody else's games are on it.
        theirs = workshop_site.signed_in("bo").get("/api/profile").payload
        assert not _games(theirs) and theirs["workshop"] == []


def _owner(site: Site, digest: str) -> int:
    with site.store.cursor() as cursor:
        cursor.execute("select owner from games where hash = %s", (digest,))
        return int(cursor.fetchone()[0])


class TestARowThatNoLongerCompiles:
    def test_the_profile_survives_it_with_no_board_and_no_score(
        self, workshop_site: Site
    ) -> None:
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _template(0, "ada's draft")).payload["hash"]
        broken = "b" * 12
        with workshop_site.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, owner, origin, visibility, parent)"
                " select %s, '{}'::jsonb, 'broken', owner, 'designed', 'unlisted', %s"
                " from games where hash = %s",
                (broken, digest, digest),
            )
        workshop_site.store.commit()
        answer = ada.get("/api/profile")
        assert answer.status == 200, answer.text
        game = _games(answer.payload)[broken]
        assert game["still"] is None and game["change"] is None
        assert game["parent"]["hash"] == digest


class TestTheParentsScore:
    @staticmethod
    def _game(status: str) -> Any:
        return SimpleNamespace(status=status, score={
            "index": 0.61, "revision": "v11", "measured_at": "2026-09-30T10:00:00+00:00",
            "certified": None, "viable": True, "weights_version": "v11",
            "seeds": [0, 1, 2, 3], "totals_used": 4,
        })

    def test_a_provisional_score_carries_its_mark_and_its_basis(self) -> None:
        shown = workshop.score(self._game(library.PROVISIONAL))
        assert shown is not None
        assert shown["status"] == library.PROVISIONAL
        assert (shown["revision"], shown["measuredAt"]) == ("v11", "2026-09-30T10:00:00+00:00")
        assert shown["seeds"] == [0, 1, 2, 3]

    def test_a_current_score_says_so_and_none_is_no_number(self) -> None:
        shown = workshop.score(self._game(library.CURRENT))
        assert shown is not None and shown["status"] == library.CURRENT
        assert workshop.score(self._game(library.NONE)) is None
        assert workshop.score(None) is None


class TestTheRungsWords:
    def test_every_rung_is_the_designers_in_the_sites_words(self) -> None:
        found = workshop.Kept(hash="h", name="n", spec={}, parent="", screened=True,
                              level=stages.ON_TRIAL, waking=False, reading="none")
        rungs = workshop.rungs(found)
        assert [(r["key"], r["name"]) for r in rungs] == [
            (r["key"], {**r, **r["site"]}["name"]) for r in candidacy.rungs()
        ]
        # On trial was reached through *Publish*, which needs the checked score.
        assert {r["key"]: r["state"] for r in rungs}["canonical"] == "done"
        assert [r["key"] for r in rungs if r["next"]] == ["nominate"]


class TestEachWaitingGame:
    def test_a_game_between_people_carries_its_position_and_last_move(
        self, store: Scratch, app: SiteApp
    ) -> None:
        found, ada, bo = _pair(store)
        _move(store, found.id, ada, bo)
        store.connection.commit()
        snapshot = matches.session(store.connection, found.id).snapshot()
        app.connect = store.open
        try:
            page = _as(app, ada).get("/api/profile")
        finally:
            app.viewer = None
        entry = page["between"]["running"][0]
        assert entry["position"]["board"] == snapshot["board"]
        assert entry["position"]["toMove"] == snapshot["sideToMove"]
        assert entry["lastMove"] == snapshot["lastMove"] and entry["lastMove"] is not None

    def test_a_rated_game_left_running_carries_its_position(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada = store.user("ada")
        found = matches.start(store.connection, SHORT_HASH, matches.LEADERBOARD,
                              ["human", AGENT], ada)
        store.connection.commit()
        snapshot = matches.session(store.connection, found.id).snapshot()
        app.connect = store.open
        try:
            page = _as(app, ada).get("/api/profile")
        finally:
            app.viewer = None
        entry = next(e for e in page["carryOn"] if e["id"] == found.id)
        assert entry["position"]["board"] == snapshot["board"]
        assert entry["lastMove"] == snapshot["lastMove"]
        stems = {i["stem"] for pair in entry["position"]["icons"] for i in pair if i}
        assert stems <= set(page["pieceArt"])

    def test_a_challenge_carries_the_games_opening(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada = store.user("ada")
        challenges.challenge(store.connection, store.user("bo"), SHORT_HASH, None, "ada")
        store.connection.commit()
        app.connect = store.open
        try:
            page = _as(app, ada).get("/api/profile")
        finally:
            app.viewer = None
        entry = page["challenges"]["forYou"][0]
        assert entry["position"] == library.still(app.game(SHORT_HASH))
        assert entry["lastMove"] is None
