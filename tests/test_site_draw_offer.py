"""A draw offered in a game on the site, to the computer or to the other person (Q790).

`docs/reference/intransitive-quiet-endgame.md` §e.1 is the rule; `tests/test_draw_offer.py`
holds the judge on §b's own positions with no store. Here are the flows with a row behind
them: the computer accepts (and a rated draw moves the table as a draw), the computer
declines (and says so, and the game goes on), an offer the rules do not allow yet is
refused in a sentence and logs nothing, a repeated offer waits its ten plies, a restart
starts the count again; and between two people the offer is accepted or declined by the
other player, with the same one-in-ten rule on the offerer.

Nothing here judges a position: every verdict is the core's, through `draw_offer.judge`.
To reach an offer without playing forty plies the tests lower `draw_offer.EARLIEST` (and
`GAP`, for the repeat), never the band or the judge. Two specs carry the verdicts, both
`dull`: `SHORT` ends at ply 2 by its limit, so the judge's search sees a draw from the
start and accepts; `LOPSIDED` gives the first player five grunts against one, so it sees
the first player well ahead and declines. Both need the store and skip without one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from test_site_correspondence import _move as _corr_move
from test_site_correspondence import _mover, _pair
from test_site_matches import (
    AGENT,
    DULL,
    LADDER,
    SHORT,
    SHORT_HASH,
    UNRATED,
    UNRATED_HASH,
    Scratch,
    _scratch,
)

from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import draw_offer, library


def _lopsided() -> dict[str, Any]:
    spec = json.loads(json.dumps(DULL))
    spec["name"] = "dull, five against one"
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": "G", "player": 0, "squares": ["a1", "b1", "c1", "d1", "e1"]},
            {"piece": "G", "player": 1, "squares": ["a5"]},
        ],
    }
    return spec


LOPSIDED = _lopsided()
LOPSIDED_HASH = spec_hash(LOPSIDED)


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store with the two-ply game rated, the lopsided one, and the unrated one."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        scratch.game(LOPSIDED_HASH, LOPSIDED)
        yield scratch


@pytest.fixture
def early(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offers open from the first ply, so a test need not play forty."""
    monkeypatch.setattr(draw_offer, "EARLIEST", 0)


def _start(store: Scratch, game: str, mode: str = m.SANDBOX, user: int | None = None) -> m.Row:
    return m.start(store.connection, game, mode, ["human", AGENT], user)


def _snapshot(store: Scratch, match_id: str) -> dict[str, Any]:
    return m.session(store.connection, match_id).snapshot()


def _person_moves_and_computer_replies(store: Scratch, match_id: str) -> None:
    first = _snapshot(store, match_id)["moves"][0]["move"]
    m.play(store.connection, match_id, first)
    m.play_agent(store.connection, match_id)


# --------------------------------------------------------------- against the computer


@pytest.mark.usefixtures("early")
class TestAgainstTheComputer:
    def test_the_computer_accepts_a_level_game_and_it_ends_drawn(self, store: Scratch) -> None:
        match = _start(store, SHORT_HASH)
        ended = m.offer_draw(store.connection, match.id)
        assert ended["outcome"] == {"winner": None, "reason": "agreed"}
        assert ended["drawOffer"] is None, "a finished game offers no draw"
        row = m.row(store.connection, match.id)
        assert row.result == 0.5 and row.reason == m.AGREED and not row.left
        (entry,) = m.draw_offers(store.connection, match.id)
        assert entry["seat"] == 0 and entry["ply"] == 0 and entry["accepted"] is True
        assert entry["value"] is not None and abs(entry["value"]) < entry["band"]

    def test_a_rated_draw_by_agreement_moves_the_table_as_a_draw(self, store: Scratch) -> None:
        user = store.user("ada")
        match = _start(store, SHORT_HASH, m.LEADERBOARD, user)
        machine = r.load(store.connection, SHORT_HASH)[r.computer(AGENT)]
        m.offer_draw(store.connection, match.id, by=user)
        table = r.load(store.connection, SHORT_HASH)
        assert table[r.person(user)].matches == 1
        assert table[r.computer(AGENT)].matches == machine.matches + 1
        assert m.row(store.connection, match.id).score_for(0) == 0.5

    def test_the_computer_declines_a_game_one_side_is_winning(self, store: Scratch) -> None:
        match = _start(store, LOPSIDED_HASH)
        answered = m.offer_draw(store.connection, match.id)
        assert answered["outcome"] is None, "the game goes on"
        assert answered["drawOffer"]["said"] == draw_offer.DECLINED_SAID
        assert answered["drawOffer"]["may"] is False, "a second offer at once"
        assert m.row(store.connection, match.id).running
        (entry,) = m.draw_offers(store.connection, match.id)
        assert entry["accepted"] is False and entry["value"] > entry["band"]

    def test_an_offer_is_made_on_the_persons_own_turn(self, store: Scratch) -> None:
        match = _start(store, LOPSIDED_HASH)
        first = _snapshot(store, match.id)["moves"][0]["move"]
        m.play(store.connection, match.id, first)
        with pytest.raises(m.MatchError, match="your own turn"):
            m.offer_draw(store.connection, match.id)
        assert m.draw_offers(store.connection, match.id) == []

    def test_a_repeated_offer_waits_its_plies(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(draw_offer, "GAP", 2)
        match = _start(store, LOPSIDED_HASH)
        m.offer_draw(store.connection, match.id)
        with pytest.raises(m.MatchError, match="offer again after 1 more move"):
            m.offer_draw(store.connection, match.id)
        _person_moves_and_computer_replies(store, match.id)
        again = m.offer_draw(store.connection, match.id)
        assert again["outcome"] is None
        assert [entry["ply"] for entry in m.draw_offers(store.connection, match.id)] == [0, 2]

    def test_a_restart_starts_the_count_again(self, store: Scratch) -> None:
        match = _start(store, LOPSIDED_HASH)
        m.offer_draw(store.connection, match.id)
        m.restart(store.connection, match.id)
        assert m.draw_offers(store.connection, match.id) == []
        assert _snapshot(store, match.id)["drawOffer"]["may"] is True

    def test_nobody_offers_a_draw_in_a_watch_or_for_somebody_else(self, store: Scratch) -> None:
        watch = m.start(store.connection, SHORT_HASH, m.SANDBOX, [AGENT, AGENT])
        with pytest.raises(m.MatchError, match="nobody is playing"):
            m.offer_draw(store.connection, watch.id)
        ada, bo = store.user("ada"), store.user("bo")
        hers = _start(store, SHORT_HASH, m.LEADERBOARD, ada)
        with pytest.raises(m.MatchError, match="not in this game"):
            m.offer_draw(store.connection, hers.id, by=bo)

    def test_an_offer_to_a_person_goes_through_their_game(self, store: Scratch) -> None:
        found, _ada, _bo = _pair(store)
        with pytest.raises(m.MatchError, match="between two people"):
            m.offer_draw(store.connection, found.id)


class TestTheGuard:
    def test_before_ply_forty_the_computer_is_not_asked(self, store: Scratch) -> None:
        match = _start(store, LOPSIDED_HASH)
        assert _snapshot(store, match.id)["drawOffer"] == {
            "may": False, "why": draw_offer.TOO_EARLY_SAID, "said": None,
        }
        with pytest.raises(m.MatchError, match="has made 20 moves"):
            m.offer_draw(store.connection, match.id)
        assert m.draw_offers(store.connection, match.id) == []


# ------------------------------------------------------------------ between two people


class TestBetweenPeople:
    def test_the_other_player_declines_and_the_game_goes_on(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        c.offer_draw(store.connection, found.id, ada)
        with pytest.raises(m.MatchError, match="no draw offer from the other player"):
            c.decline_draw(store.connection, found.id, ada)
        c.decline_draw(store.connection, found.id, bo)
        assert c.offer_of(store.connection, found.id) is None
        assert m.row(store.connection, found.id).running

    def test_the_other_player_accepts_and_it_is_a_draw(self, store: Scratch) -> None:
        found, ada, bo = _pair(store)
        c.offer_draw(store.connection, found.id, bo)
        c.accept_draw(store.connection, found.id, ada)
        row = m.row(store.connection, found.id)
        assert row.result == 0.5 and row.reason == m.AGREED

    def test_a_declined_offer_waits_before_it_is_made_again(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(draw_offer, "GAP", 2)
        found, ada, bo = _pair(store, UNRATED_HASH)
        c.offer_draw(store.connection, found.id, ada)
        c.decline_draw(store.connection, found.id, bo)
        with pytest.raises(m.MatchError, match="offer again"):
            c.offer_draw(store.connection, found.id, ada)
        view = c.view(store.connection, m.row(store.connection, found.id), ada)
        assert view["mayOffer"] is False and "offer again" in view["offerWhy"]
        # The other player may still offer: the count is each side's own.
        assert c.view(store.connection, m.row(store.connection, found.id), bo)["mayOffer"]
        _corr_move(store, found.id, ada, bo)
        _corr_move(store, found.id, ada, bo)
        assert c.offer_draw(store.connection, found.id, ada) is not None
        assert len(m.draw_offers(store.connection, found.id)) == 2

    def test_the_viewer_whose_offer_stands_is_told_so(self, store: Scratch) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        c.offer_draw(store.connection, found.id, ada)
        row = m.row(store.connection, found.id)
        assert c.view(store.connection, row, ada)["mayOffer"] is False
        assert c.view(store.connection, row, bo)["mayOffer"] is True, "accepting is open"
        assert _mover(store, found.id, ada, bo) in (ada, bo)


# ---------------------------------------------------------------------- through the app


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


@pytest.mark.usefixtures("early")
class TestTheRoutes:
    def test_a_draw_offered_to_the_computer_through_the_session_route(
        self, store: Scratch, app: SiteApp
    ) -> None:
        user = store.user("ada")
        app.connect = store.open
        app.viewer = user
        try:
            snapshot = app.post("/api/session", {"hash": LOPSIDED_HASH, "seats": ["human", AGENT]})
            declined = app.post(f"/api/session/{snapshot['id']}/draw", {})
            level = app.post("/api/session", {"hash": SHORT_HASH, "seats": ["human", AGENT]})
            agreed = app.post(f"/api/session/{level['id']}/draw", {})
        finally:
            app.viewer = None
        assert declined["outcome"] is None
        assert declined["drawOffer"]["said"] == draw_offer.DECLINED_SAID
        assert agreed["outcome"] == {"winner": None, "reason": "agreed"}

    def test_decline_between_people_through_the_session_route(
        self, store: Scratch, app: SiteApp
    ) -> None:
        found, ada, bo = _pair(store, UNRATED_HASH)
        app.connect = store.open
        try:
            app.viewer = ada
            offered = app.post(f"/api/session/{found.id}/draw", {})
            app.viewer = bo
            answered = app.post(f"/api/session/{found.id}/decline", {})
        finally:
            app.viewer = None
        assert offered["correspondence"]["drawOffer"] == 0
        assert answered["correspondence"]["drawOffer"] is None
        assert answered["outcome"] is None
