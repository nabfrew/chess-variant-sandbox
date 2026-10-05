"""People's counts (task T3 of `docs/plan-confidence.md`): the E-H evidence `trust.read` takes.

One fold over `matches.leaderboard_matches` per game: finished games against the top
group, distinct people, distinct people who cleared, and the latest date. The clears are
`unbeaten.tag_from`'s own fold, and a test below checks the two agree on every store this
file builds, rather than trusting that they were written alike.

The games are `test_site_unbeaten.py`'s: `dull` under other names and ply limits, one with
a clear top and one with a tied top. These need a real Postgres and skip with a reason
without one. Results are stated, never derived: nothing here works out an outcome.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_site_unbeaten import (
    ALPHA,
    ALPHA_HASH,
    BETA,
    BETA_HASH,
    BOTTOM,
    CLEAR_LADDER,
    GAMMA,
    GAMMA_HASH,
    MIDDLE,
    RIVAL,
    TIED_LADDER,
    TOP,
    Scratch,
    _scratch,
)

from chess_variant_sandbox import trust
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import people, unbeaten

GAMES = (ALPHA_HASH, BETA_HASH, GAMMA_HASH)


@pytest.fixture
def store() -> Iterator[Scratch]:
    """`test_site_unbeaten`'s three games: a clear top, a tied top, and no ladder."""
    with _scratch() as scratch:
        scratch.game(ALPHA_HASH, ALPHA, CLEAR_LADDER)
        scratch.game(BETA_HASH, BETA, TIED_LADDER)
        scratch.game(GAMMA_HASH, GAMMA)
        yield scratch


def _at(store: Scratch, match_id: str, when: str) -> None:
    """Move a finished match's timestamp outright, so a date can be stated."""
    with store.connection.cursor() as cursor:
        cursor.execute("update matches set finished = %s where id = %s", (when, match_id))
    store.connection.commit()


def _agrees(store: Scratch) -> None:
    """The counts' clears are the tag's clears, game by game, on whatever the store holds."""
    tags = unbeaten.tags(store.connection, GAMES)
    for game in GAMES:
        assert people.counts(store.connection, game).cleared == tags[game].clears, game


class TestTheCounts:
    def test_no_games_is_zero_and_no_date(self, store: Scratch) -> None:
        found = people.counts(store.connection, ALPHA_HASH)
        assert found == trust.People(games=0, people=0, cleared=0, latest=None)

    def test_one_person_is_one_however_often_they_play(self, store: Scratch) -> None:
        ada = store.user("ada")
        for result in (0.0, 0.5, 0.0):
            store.play(ALPHA_HASH, ada, TOP, result=result)
        found = people.counts(store.connection, ALPHA_HASH)
        assert (found.games, found.people, found.cleared) == (3, 1, 0)
        assert found.latest == "2026-09-11"

    def test_several_people_are_counted_once_each(self, store: Scratch) -> None:
        ada, bob, cy = store.user("ada"), store.user("bob"), store.user("cy")
        store.play(ALPHA_HASH, ada, TOP, result=0.0)
        store.play(ALPHA_HASH, bob, TOP, result=0.0, seat=1)
        store.play(ALPHA_HASH, bob, TOP, result=0.5)
        store.play(ALPHA_HASH, cy, TOP, result=1.0)
        found = people.counts(store.connection, ALPHA_HASH)
        assert (found.games, found.people, found.cleared) == (4, 3, 1)

    def test_only_games_against_the_top_count(self, store: Scratch) -> None:
        """A game against a computer player below the top is not a game against the top,
        and neither is sandbox, an unfinished match, or a row from the local app."""
        ada, bob = store.user("ada"), store.user("bob")
        store.play(ALPHA_HASH, ada, MIDDLE, result=1.0)
        store.play(ALPHA_HASH, ada, BOTTOM, result=0.0)
        store.play(ALPHA_HASH, ada, TOP, result=1.0, mode=m.SANDBOX)
        store.play(ALPHA_HASH, ada, TOP, result=None)
        store.play(ALPHA_HASH, bob, TOP, result=1.0, source="local-app")
        assert people.counts(store.connection, ALPHA_HASH) == trust.People(0, 0, 0, None)

    def test_a_game_with_no_ladder_has_nothing_to_count(self, store: Scratch) -> None:
        assert people.counts(store.connection, GAMMA_HASH) == trust.People(0, 0, 0, None)

    def test_one_games_rows_never_reach_another(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        assert people.counts(store.connection, BETA_HASH).games == 0
        rows = m.leaderboard_matches(store.connection)
        assert people.counts_from(BETA_HASH, rows).games == 0
        assert people.counts_from(ALPHA_HASH, rows).games == 1


class TestTheClearing:
    """A clear is exactly `unbeaten`'s, and the agreement is checked on each store."""

    def test_a_win_against_the_top_clears(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0, seat=1)
        assert people.counts(store.connection, ALPHA_HASH).cleared == 1
        _agrees(store)

    def test_a_draw_or_a_loss_does_not(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=0.5)
        store.play(ALPHA_HASH, ada, TOP, result=0.0)
        assert people.counts(store.connection, ALPHA_HASH).cleared == 0
        _agrees(store)

    def test_one_person_clearing_twice_is_one(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        found = people.counts(store.connection, ALPHA_HASH)
        assert (found.games, found.people, found.cleared) == (2, 1, 1)
        _agrees(store)

    def test_a_tied_top_needs_each_of_the_tied(self, store: Scratch) -> None:
        """Beating one of a tied pair is a game against the top and not yet a clear."""
        ada = store.user("ada")
        store.play(BETA_HASH, ada, TOP, result=1.0)
        found = people.counts(store.connection, BETA_HASH)
        assert (found.games, found.people, found.cleared) == (1, 1, 0)
        _agrees(store)
        store.play(BETA_HASH, ada, RIVAL, result=1.0)
        found = people.counts(store.connection, BETA_HASH)
        assert (found.games, found.people, found.cleared) == (2, 1, 1)
        _agrees(store)

    def test_a_clear_survives_the_top_changing_hands(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        store.lift(ALPHA_HASH, MIDDLE, 900.0)
        found = people.counts(store.connection, ALPHA_HASH)
        assert (found.games, found.cleared) == (1, 1)
        _agrees(store)

    def test_the_many_game_fold_is_the_one_game_fold(self, store: Scratch) -> None:
        ada, bob = store.user("ada"), store.user("bob")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        store.play(BETA_HASH, bob, TOP, result=0.0)
        store.play(BETA_HASH, bob, RIVAL, result=1.0)
        many = people.counts_of(store.connection, GAMES)
        assert many == {game: people.counts(store.connection, game) for game in GAMES}
        _agrees(store)


class TestTheDate:
    def test_the_latest_is_the_newest_not_the_last_read(self, store: Scratch) -> None:
        ada, bob = store.user("ada"), store.user("bob")
        first = store.play(ALPHA_HASH, ada, TOP, result=0.0)
        second = store.play(ALPHA_HASH, bob, TOP, result=0.0)
        _at(store, first.id, "2026-09-20 08:00:00+00")
        _at(store, second.id, "2026-09-14 08:00:00+00")
        assert people.counts(store.connection, ALPHA_HASH).latest == "2026-09-20"

    def test_two_games_finished_at_the_same_moment(self, store: Scratch) -> None:
        """A tie in the timestamp is one date, and both games and both people count."""
        ada, bob = store.user("ada"), store.user("bob")
        first = store.play(ALPHA_HASH, ada, TOP, result=1.0)
        second = store.play(ALPHA_HASH, bob, TOP, result=1.0)
        for row in (first, second):
            _at(store, row.id, "2026-09-30 23:30:00+00")
        found = people.counts(store.connection, ALPHA_HASH)
        assert found == trust.People(games=2, people=2, cleared=2, latest="2026-09-30")
        _agrees(store)

    def test_the_date_is_utc(self, store: Scratch) -> None:
        ada = store.user("ada")
        row = store.play(ALPHA_HASH, ada, TOP, result=0.0)
        _at(store, row.id, "2026-10-01 01:30:00+05")
        assert people.counts(store.connection, ALPHA_HASH).latest == "2026-09-30"

    def test_a_naive_moment_is_read_as_it_stands(self) -> None:
        assert people._day(datetime.datetime(2026, 9, 1, 12, 0)) == "2026-09-01"
        assert people._day(None) is None


class TestHandedToTrust:
    def test_trust_reads_the_counts_and_no_score_has_no_level(
        self, store: Scratch, tmp_path: Path
    ) -> None:
        """No score on disk: the level is None, and the people's item carries the counts
        and is counted (`RULE.people_move_level`, plan §5 Q3, answered A)."""
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        read = people.read(store.connection, ALPHA_HASH, tmp_path, None, nets_dir=tmp_path)
        item = next(found for found in read.items if found.kind == trust.PEOPLE)
        assert item.state == trust.CURRENT
        assert item.detail == {"games": 1, "people": 1, "cleared": 1}
        assert item.date == "2026-09-11"
        assert trust.RULE.people_move_level and item.counted
        assert read.level is None

    def test_a_folded_mapping_is_used_as_given(self, store: Scratch, tmp_path: Path) -> None:
        given = {ALPHA_HASH: trust.People(games=7, people=3, cleared=0, latest="2026-01-02")}
        read = people.read(
            store.connection, ALPHA_HASH, tmp_path, None, nets_dir=tmp_path, people=given
        )
        item = next(found for found in read.items if found.kind == trust.PEOPLE)
        assert item.detail == {"games": 7, "people": 3, "cleared": 0}



def test_it_is_in_no_fingerprint_set() -> None:
    """People's counts sit beside a score and never in it: no stamped fingerprint moves."""
    from chess_variant_sandbox import measurement

    ours = (measurement.ROOT / "chess_variant_sandbox/site/people.py").resolve()
    for kind in ("engine", "metrics", "analysis"):
        assert ours not in {path.resolve() for path in measurement.fingerprinted(kind)}, kind
