"""The badges for finishing proven positions (Q993e, `docs/archive/plan-proven.md` §4).

*Every proven position* is a win from every position a game offers, each one played out
against hardest, once per game; *Game after game* is a series over how many games a
person holds it on, at `badges.PROVEN_STEPS` (5, 10 and 25), fixed in the plan before any
person's marks were read. Both are kept rows: a re-proof that adds positions never takes
one away, and the game's own line shows the new total beside it.

Proved here: the reader on its own (due on the last position, dated by it, a find never
counts, a game with nothing offered earns nothing, the series at each step and never past
25, a re-proof removes nothing); and through the site against a store: granted on the try
that wins the last position, in that finish's transaction, kept through a re-proof, drawn
on the profile and the member's page and listed on `/badges`, and nothing for a visitor's
finds or for a member's. On the test game of `test_site_proven.py` (rule 5); the store
tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it.
"""

from __future__ import annotations

import html
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_site_matches import Scratch, _scratch
from test_site_proven import (
    DIGEST,
    ELSEWHERE,
    ENGINE,
    LOSES,
    LOSES_AT,
    ONE,
    SPEC,
    TAKE,
    _at,
    _named,
    _position,
    _record,
    _write,
)

from chess_variant_sandbox.site import awards, badges, members, pages, proven
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import library

ADA = 7
DAY = datetime(2026, 10, 21, 12, tzinfo=UTC)

# The winning move of each position used here, named so the engine finds it.
LOSES_TAKE = ENGINE.find_move(_at(LOSES_AT), "Gd2xc3")
# Two more, for a re-proof that adds positions: the second player's grunt on c2, taken
# from d1 in the first and from b1 in the second, which ends the game.
THIRD_AT, FOURTH_AT = ELSEWHERE[1], ELSEWHERE[2]
THIRD_TAKE = ENGINE.find_move(_at(THIRD_AT), "Gd1xc2")
FOURTH_TAKE = ENGINE.find_move(_at(FOURTH_AT), "Gb1xc2")
THIRD = _position("0-3-4", {THIRD_TAKE: "won"}, THIRD_AT, ends_in=1)
FOURTH = _position("0-4-4", {FOURTH_TAKE: "won"}, FOURTH_AT, ends_in=1)
WINS = {_named(ONE): TAKE, _named(LOSES): LOSES_TAKE, _named(THIRD): THIRD_TAKE,
        _named(FOURTH): FOURTH_TAKE}


# ------------------------------------------------------------------- the reader alone


def _offer(game: str, *ids: str, made: str = "2026-10-18") -> proven.Record:
    """A record offering these positions, as the loader would hand it over."""
    return proven.Record(
        game=game, made=made, basis={}, counts={}, dropped=0,
        positions=tuple(
            proven.Position(id=pid, number=n, moves=(), side=0, ends_in=1, pieces=2,
                            single=True, others_lost=False, scores=())
            for n, pid in enumerate(ids, start=1)
        ),
    )


def _won(game: str, position: str, hours: int, match: str | None = None) -> badges.Conversion:
    return badges.Conversion(game=game, position=position, when=DAY + timedelta(hours=hours),
                             match=match or f"{game}-{position}")


def _finished(count: int) -> tuple[list[badges.Conversion], dict[str, proven.Record]]:
    """`count` games of one position each, finished an hour apart."""
    games = [f"g{index:02d}" for index in range(count)]
    return ([_won(game, "p", index) for index, game in enumerate(games)],
            {game: _offer(game, "p") for game in games})


class TestTheReader:
    def test_the_marks_it_reads_are_the_sites_converted(self) -> None:
        assert badges.PROVEN_WAYS == (proven.CONVERTED,)
        assert proven.FOUND not in badges.PROVEN_WAYS

    def test_it_is_due_on_the_last_position_and_dated_by_it(self) -> None:
        records = {"alpha": _offer("alpha", "a", "b", "c")}
        two = [_won("alpha", "a", 1), _won("alpha", "c", 3)]
        assert badges.proven_due_from(ADA, two, records) == ()
        three = [*two, _won("alpha", "b", 5, match="last")]
        (award,) = badges.proven_due_from(ADA, three, records)
        assert award.kind == badges.EVERY_PROVEN and award.game == "alpha"
        assert award.holder == ADA and award.tier is None
        assert award.earned == DAY + timedelta(hours=5)
        assert award.evidence["positions"] == 3 and award.evidence["proved"] == "2026-10-18"
        assert sorted(award.evidence["matches"]) == ["alpha-a", "alpha-c", "last"]
        assert award.rule == {"ways": [proven.CONVERTED]}

    def test_a_mark_on_a_position_no_longer_offered_neither_helps_nor_hinders(self) -> None:
        records = {"alpha": _offer("alpha", "a")}
        (award,) = badges.proven_due_from(
            ADA, [_won("alpha", "gone", 0), _won("alpha", "a", 2)], records)
        assert award.evidence["positions"] == 1 and award.evidence["matches"] == ["alpha-a"]

    def test_a_game_with_nothing_offered_or_no_record_earns_nothing(self) -> None:
        """Every position of none is not a finish: a record whose positions were all too
        easy, or a game with no record left, earns nothing however it was marked."""
        marked = [_won("alpha", "a", 0), _won("beta", "a", 0)]
        assert badges.proven_due_from(ADA, marked, {"alpha": _offer("alpha"), "beta": None}) == ()

    def test_what_is_held_is_not_offered_again(self) -> None:
        records = {"alpha": _offer("alpha", "a")}
        (award,) = badges.proven_due_from(ADA, [_won("alpha", "a", 0)], records)
        assert badges.proven_due_from(ADA, [_won("alpha", "a", 0)], records, [award]) == ()

    def test_a_reproof_that_adds_positions_removes_nothing(self) -> None:
        """The held row stands; the reader offers nothing new for that game, and the series
        still counts it, because the series reads the kept rows and not the live marks."""
        held = [badges.proven_due_from(ADA, *_finished(5))]
        kept = list(held[0])
        assert [a.kind for a in kept] == [badges.EVERY_PROVEN] * 5 + [badges.GAME_AFTER_GAME]
        conversions, records = _finished(5)
        grown = {game: _offer(game, "p", "new") for game in records}
        assert badges.proven_due_from(ADA, conversions, grown, kept) == ()
        # With only the rows, and no record left at all, the step is still held and due
        # for nobody again.
        assert badges.proven_due_from(ADA, [], {}, kept) == ()

    @pytest.mark.parametrize(("games", "steps"), [
        (1, []), (4, []), (5, [5]), (9, [5]), (10, [5, 10]), (24, [5, 10]),
        (25, [5, 10, 25]), (40, [5, 10, 25]),
    ])
    def test_game_after_game_at_its_steps(self, games: int, steps: list[int]) -> None:
        due = badges.proven_due_from(ADA, *_finished(games))
        series = [a for a in due if a.kind == badges.GAME_AFTER_GAME]
        assert [a.tier for a in series] == steps
        assert sum(1 for a in due if a.kind == badges.EVERY_PROVEN) == games
        for award in series:
            # Dated by the game that reached the step, which finished at hour step - 1.
            assert award.earned == DAY + timedelta(hours=int(award.tier or 0) - 1)
            assert award.evidence["games"] == [f"g{i:02d}" for i in range(int(award.tier or 0))]
            assert award.game is None and award.rule == {"steps": [5, 10, 25]}

    def test_the_series_counts_kept_rows_as_well_as_new_ones(self) -> None:
        """Four games held from before, the fifth finished now: the step is due, dated by
        the fifth."""
        conversions, records = _finished(5)
        earlier = badges.proven_due_from(ADA, conversions[:4], records)
        assert [a.kind for a in earlier] == [badges.EVERY_PROVEN] * 4
        due = badges.proven_due_from(ADA, conversions, records, earlier)
        assert [(a.kind, a.tier) for a in due] == [(badges.EVERY_PROVEN, None),
                                                  (badges.GAME_AFTER_GAME, 5)]
        assert due[1].earned == DAY + timedelta(hours=4)

    def test_another_persons_rows_count_for_nothing(self) -> None:
        conversions, records = _finished(5)
        theirs = badges.proven_due_from(ADA + 1, conversions, records)
        due = badges.proven_due_from(ADA, conversions[:4], records, theirs)
        assert not any(a.kind == badges.GAME_AFTER_GAME for a in due)

    def test_the_line_quotes_what_was_finished(self) -> None:
        (award,) = badges.proven_due_from(
            ADA, [_won("alpha", "a", 0), _won("alpha", "b", 0)], {"alpha": _offer("alpha", "a", "b")})
        line = badges.line(award)
        assert line["name"] == "Every proven position"
        assert line["quoted"] == "2 proven positions on the game"
        assert "against hardest" in line["sentence"] and "Finding the move" in line["rule"]
        one = awards.Award(kind=badges.EVERY_PROVEN, evidence={"positions": 1})
        assert badges.quoted(one) == "1 proven position on the game"
        series = awards.Award(kind=badges.GAME_AFTER_GAME, tier=5, evidence={"games": ["x"]})
        assert badges.sentence_for(series) == "Every proven position, on 5 different games."
        assert badges.quoted(series) == ""


class TestTheProfileDrawsIt:
    def test_a_held_badge_is_a_dated_line_with_its_figure_and_rule(self) -> None:
        """The profile's own script over `badges.line`'s answer, as `test_site` runs it."""
        from test_site import _done

        award = awards.Award(kind=badges.EVERY_PROVEN, holder=ADA, game="aaa",
                             earned=DAY, evidence={"positions": 3})
        said = _done([], [{**badges.line(award), "name_of": "alpha"}])
        assert "Every proven position, on alpha, 21 Oct 2026." in said
        assert "On that day, 3 proven positions on the game." in said
        assert badges.BADGES[badges.EVERY_PROVEN].rule in said


class TestTheList:
    def test_both_are_on_badges_with_their_rules_and_steps(self) -> None:
        page = pages.badges()
        for kind in (badges.EVERY_PROVEN, badges.GAME_AFTER_GAME):
            badge = badges.BADGES[kind]
            assert html.escape(badge.name) in page and html.escape(badge.rule) in page
        assert "In steps, at 5, 10 and 25." in page
        assert "except the two for proven positions" in page


# ------------------------------------------------------------------ through the site


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(DIGEST, SPEC)
        yield scratch


@pytest.fixture
def app(tmp_path: Path) -> SiteApp:
    """The site over a record of two positions with one winning move each, the game known
    as a library game."""
    _write(tmp_path, _record([ONE, LOSES]))
    site = SiteApp(library.RULESET_DIR, tmp_path)
    game = library.Game(hash=DIGEST, name="proven test", description="", path=Path(),
                        source="committed", spec=SPEC)
    site.known = lambda: {DIGEST: game}  # type: ignore[method-assign]
    return site


def _convert(app: SiteApp, position: str) -> dict[str, Any]:
    """A try from the position, played out with its winning move, through the site."""
    started = app.post("/api/session", {"hash": DIGEST, "puzzle": position})
    done = app.post(f"/api/session/{started['id']}/move", {"move": WINS[position]})
    assert done["outcome"]["winner"] == done["puzzle"]["side"]
    return done


def _held(store: Scratch, person: int) -> list[awards.Award]:
    connection = store.open()
    try:
        return awards.held(connection, holder=person)
    finally:
        connection.close()


class TestThroughTheSite:
    def test_granted_on_the_win_from_the_last_position(self, store: Scratch, app: SiteApp) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        store.connection.commit()
        app.connect = store.open
        try:
            app.viewer = ada
            _convert(app, _named(ONE))
            assert _held(store, ada) == []
            done = _convert(app, _named(LOSES))
            (award,) = _held(store, ada)
            assert (award.kind, award.game, award.holder) == (badges.EVERY_PROVEN, DIGEST, ada)
            assert award.evidence["positions"] == 2
            assert len(award.evidence["matches"]) == 2
            # Dated by the mark it rests on, written in the same finish.
            with store.connection.cursor() as cursor:
                cursor.execute("select max(marked) from proven_marks where person = %s", (ada,))
                assert award.earned == cursor.fetchone()[0]
            assert done["puzzle"]["converted"] is True
            # On the profile, with its sentence and its figure, and the game's own line.
            profile = app.get("/api/profile")
            (line,) = profile["badges"]
            assert line["kind"] == badges.EVERY_PROVEN and line["name_of"] == "proven test"
            assert line["quoted"] == "2 proven positions on the game"
            assert profile["proven"][0]["converted"] == 2 and profile["proven"][0]["of"] == 2
            # Nobody else holds anything for it.
            assert _held(store, bea) == []
        finally:
            app.viewer = None

    def test_a_reproof_that_adds_positions_never_removes_it(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada = store.user("ada")
        store.connection.commit()
        app.connect = store.open
        try:
            app.viewer = ada
            _convert(app, _named(ONE))
            _convert(app, _named(LOSES))
            (before,) = _held(store, ada)
            # The record is proved again and offers two more.
            where = _write(app.runs_dir, _record([ONE, LOSES, THIRD, FOURTH]))
            os.utime(where, ns=(10**18, 10**18))
            # A win again on a position already converted, and one on a new position.
            _convert(app, _named(ONE))
            _convert(app, _named(THIRD))
            (after,) = _held(store, ada)
            assert after == before
            assert after.evidence["positions"] == 2
            # The game's line shows the new total beside the kept badge.
            profile = app.get("/api/profile")
            assert profile["proven"][0]["converted"] == 3 and profile["proven"][0]["of"] == 4
            assert [b["quoted"] for b in profile["badges"]] == ["2 proven positions on the game"]
            # Finishing the new set adds nothing: the badge is once per game.
            _convert(app, _named(FOURTH))
            assert _held(store, ada) == [before]
        finally:
            app.viewer = None

    def test_finds_earn_nothing_a_visitors_or_a_members(self, store: Scratch, app: SiteApp) -> None:
        ada = store.user("ada")
        store.connection.commit()
        app.connect = store.open
        try:
            for viewer in (None, ada):
                app.viewer = viewer
                for position in (_named(ONE), _named(LOSES)):
                    found = app.post(f"/api/proven/{DIGEST}/{position}/find",
                                     {"move": WINS[position]})
                    assert found["right"] is True
            # Ada has found both, so her share reads 100%, and no badge rests on finds.
            page = app.get(f"/api/proven/{DIGEST}")
            assert page["share"]["said"] == "100%"
            with store.connection.cursor() as cursor:
                cursor.execute("select count(*) from awards")
                assert cursor.fetchone()[0] == 0
            # A try that ends otherwise grants nothing either.
            started = app.post("/api/session", {"hash": DIGEST, "puzzle": _named(ONE)})
            app.post(f"/api/session/{started['id']}/resign", {})
            assert _held(store, ada) == []
        finally:
            app.viewer = None

    def test_a_visitors_try_earns_nothing(self, store: Scratch) -> None:
        """A visitor cannot sit at hardest, and a win against a weaker player from the
        position, started directly, is nobody's to mark or reward."""
        found = m.start(store.connection, DIGEST, m.SANDBOX, ["human", "minimax:nodes=6400"],
                        None, puzzle=(_named(ONE), ONE["moves"]))
        m.play(store.connection, found.id, TAKE)
        assert proven.won_by(store.connection, found.id) is None

    def test_it_is_drawn_on_the_members_page(self, store: Scratch, app: SiteApp) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        store.connection.commit()
        app.connect = store.open
        try:
            app.viewer = ada
            _convert(app, _named(ONE))
            _convert(app, _named(LOSES))
        finally:
            app.viewer = None
        connection = store.open()
        try:
            reading = members.read(connection, "ada", bea)
        finally:
            connection.close()
        assert reading is not None
        (shown,) = reading["badges"]
        assert shown["kind"] == badges.EVERY_PROVEN and shown["game"] == DIGEST
        page = pages.member(reading, "bea", viewer=bea)
        assert "<b>Every proven position</b>" in page
