"""A badge is the record of an event, written once and never updated.

Box E1 of `docs/archive/plan-meta-game.md`, against its §2.9. Three
properties carry the whole file. The same badge granted twice leaves one row, whichever of
the three nullable key columns is empty. A badge that quotes an index can be granted only
while that game's measurement reads current, which is what `score_evidence` answers and
what it refuses. And a row keeps the thresholds it was earned under, so it still explains
itself after the constant moves.

No badge kind is defined here or in `site/awards.py`. The kinds, their thresholds and
their sentences are box E3's, in `site/badges.py`.

The store tests need a real Postgres and skip with a reason without one, each on
the session's migrated store (`tests/site_store.py`), emptied first -- a real server rather than
`py-pglite`, which does not work on Windows (task 1.2). The measurement rows are written by hand in the store's own shape; the loader is
what fills them in life, and `tests/test_site_load.py` holds that end down.

Nothing here works out a score. The rows say what they say and the reader either quotes
them or refuses to.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import site_store

from chess_variant_sandbox.site import awards as a
from chess_variant_sandbox.site import migrate
from chess_variant_sandbox.site.db import connect

GAME = "aaaaaaaaaaaa"
OTHER = "bbbbbbbbbbbb"

# What `measurements.basis` holds, cut down to the two fields a quotation reads off it.
# The whole stamp travels into `evidence`, so the test asserts on it whole.
BASIS = {"ruleset_hash": GAME, "weights_version": "v10", "seed": 0}

HEADLINE = 0.5231
LOW = 0.4712
HIGH = 0.5803


# ------------------------------------------------------------------------- the fixture


class Scratch:
    """A store with the schema in it, on a schema of its own."""

    def __init__(self, url: str, schema: str) -> None:
        self.url = url
        self.schema = schema
        self.connection = self.open()

    def open(self) -> Any:
        connection = connect(self.url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {self.schema}")
        return connection

    def user(self, handle: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into users (handle, email) values (%s, %s) returning id",
                (handle, f"{handle}@example.invalid"),
            )
            found = cursor.fetchone()
        self.connection.commit()
        return int(found[0])

    def game(self, digest: str) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps({"name": digest}), digest, "corpus"),
            )
        self.connection.commit()
        return digest

    def measurement(
        self,
        digest: str,
        state: str,
        headline: float | None = HEADLINE,
        ci_low: float | None = LOW,
        ci_high: float | None = HIGH,
    ) -> None:
        """One row in the shape `load.load_measurements` writes it."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into measurements"
                " (hash, directory, state, basis, headline, ci_low, ci_high)"
                " values (%s, %s, %s, %s::jsonb, %s, %s, %s)",
                (digest, f"{digest}-0", state, json.dumps(BASIS), headline,
                 ci_low, ci_high),
            )
        self.connection.commit()

    def rows(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("select count(*) from awards")
            return int(cursor.fetchone()[0])

    def rule_of(self, kind: str) -> dict[str, Any]:
        with self.connection.cursor() as cursor:
            cursor.execute("select rule from awards where kind = %s", (kind,))
            return dict(cursor.fetchone()[0])


@contextlib.contextmanager
def _scratch() -> Iterator[Scratch]:
    url, name = site_store.emptied("no database to grant a badge in")
    scratch = Scratch(url, name)
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    scratch.connection.commit()
    try:
        yield scratch
    finally:
        scratch.connection.close()


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store holding two games and one person."""
    with _scratch() as scratch:
        scratch.game(GAME)
        scratch.game(OTHER)
        yield scratch


@pytest.fixture
def player(store: Scratch) -> int:
    return store.user("ada")


# ------------------------------------------------------------------ no database needed


class TestTheValues:
    """What an `Award` and a quotation are, without a store in reach."""

    def test_an_awards_key_is_what_the_store_is_unique_on(self) -> None:
        award = a.Award(kind="beat_easy", tier=5, holder=3, game=GAME)
        assert award.key == ("beat_easy", 5, 3, GAME)

    def test_an_award_with_no_series_and_no_person_is_all_nulls_but_its_kind(self) -> None:
        assert a.Award(kind="crowd_pleaser", game=GAME).key == (
            "crowd_pleaser", None, None, GAME
        )

    def test_a_quotation_carries_the_range_and_the_weights_into_the_evidence(self) -> None:
        """§2.9: a badge that reads the index stores all five, and the basis whole."""
        quoted = a.ScoreEvidence(
            game=GAME, headline=HEADLINE, ci_low=LOW, ci_high=HIGH,
            weights="v10", spec_hash=GAME, basis=BASIS,
        )
        assert quoted.as_dict() == {
            "headline": HEADLINE,
            "ci_low": LOW,
            "ci_high": HIGH,
            "weights": "v10",
            "spec_hash": GAME,
            "basis": BASIS,
        }
        assert quoted.ranged

    def test_a_quotation_with_one_end_missing_is_not_ranged(self) -> None:
        """A measurement too small to resample has a headline and no ends, and a badge
        that tests the whole range cannot be granted on one."""
        quoted = a.ScoreEvidence(
            game=GAME, headline=HEADLINE, ci_low=None, ci_high=None,
            weights="v10", spec_hash=GAME, basis=BASIS,
        )
        assert not quoted.ranged


# ------------------------------------------------------------------------ the migration


class TestTheAwardsTable:
    def test_the_migration_applies_after_the_ones_already_there(
        self, store: Scratch
    ) -> None:
        """The fixture ran every file in order; this is what the last one built."""
        with store.connection.cursor() as cursor:
            cursor.execute(
                f"select name from {migrate.LEDGER} where name like %s", ("0008%",)
            )
            applied = cursor.fetchone()
        assert applied is not None and applied[0] == "0008_awards.sql"
        assert store.rows() == 0

    def test_an_award_names_a_game_that_is_there(self, store: Scratch) -> None:
        """The key is a foreign key, so a badge cannot sit on a game the store lost."""
        import psycopg

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            a.grant(store.connection, [a.Award(kind="held_top", game="nosuchgame01")])
        store.connection.rollback()


class TestGrantingOnce:
    def test_the_same_award_twice_leaves_one_row(
        self, store: Scratch, player: int
    ) -> None:
        award = a.Award(kind="beat_easy", tier=5, holder=player, game=GAME)
        first = a.grant(store.connection, [award])
        second = a.grant(store.connection, [award])
        store.connection.commit()
        assert len(first) == 1 and second == []
        assert store.rows() == 1

    @pytest.mark.parametrize(
        "award",
        [
            pytest.param(a.Award(kind="held_top", game=GAME), id="no tier, no holder"),
            pytest.param(a.Award(kind="long_way", tier=None), id="everything null"),
        ],
    )
    def test_a_null_tier_holder_or_game_still_grants_only_once(
        self, store: Scratch, award: a.Award
    ) -> None:
        """Postgres counts nulls as distinct by default, which would let the same badge
        be granted every time an event fired. The constraint is `nulls not distinct`."""
        a.grant(store.connection, [award, award])
        a.grant(store.connection, [award])
        store.connection.commit()
        assert store.rows() == 1

    def test_a_null_holder_and_a_real_one_are_two_badges(
        self, store: Scratch, player: int
    ) -> None:
        """A badge on a library game has no holder; the same kind on a designed game has
        its owner. The two are different rows, which is what the key has to allow."""
        a.grant(store.connection, [
            a.Award(kind="crowd_pleaser", game=GAME),
            a.Award(kind="crowd_pleaser", game=GAME, holder=player),
        ])
        store.connection.commit()
        assert store.rows() == 2

    def test_the_steps_of_a_series_are_separate_rows(
        self, store: Scratch, player: int
    ) -> None:
        a.grant(store.connection, [
            a.Award(kind="beat_easy", tier=1, holder=player, game=GAME),
            a.Award(kind="beat_easy", tier=5, holder=player, game=GAME),
        ])
        store.connection.commit()
        assert store.rows() == 2

    def test_nothing_commits(self, store: Scratch, player: int) -> None:
        """A badge is granted inside the transaction of the event that completed it, so
        the event and the badge land together or neither does."""
        a.grant(store.connection, [a.Award(kind="held_top", holder=player, game=GAME)])
        store.connection.rollback()
        assert store.rows() == 0

    def test_an_award_with_no_kind_is_refused(self, store: Scratch) -> None:
        with pytest.raises(a.AwardError):
            a.grant(store.connection, [a.Award(kind="  ")])


class TestWhatARowKeeps:
    def test_a_row_reads_its_own_rule_after_the_constant_moves(
        self, store: Scratch
    ) -> None:
        """§2.9: every row copies the thresholds it was earned under, so an old row stays
        readable after a threshold moves. A later grant under the new constant conflicts
        and writes nothing, which is what leaves the first reading intact."""
        a.grant(store.connection, [a.Award(
            kind="crowd_pleaser", game=GAME, rule={"people": 5, "stars": 4.0}
        )])
        a.grant(store.connection, [a.Award(
            kind="crowd_pleaser", game=GAME, rule={"people": 20, "stars": 4.5}
        )])
        store.connection.commit()
        assert store.rows() == 1
        assert store.rule_of("crowd_pleaser") == {"people": 5, "stars": 4.0}

    def test_an_older_event_dates_the_row(self, store: Scratch, player: int) -> None:
        """A badge completed before its box landed arrives with the person's next event,
        dated by the row that completed it rather than by the pass that noticed."""
        then = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        [written] = a.grant(store.connection, [a.Award(
            kind="long_way", holder=player, game=GAME, earned=then
        )])
        store.connection.commit()
        assert written.earned == then

    def test_an_award_with_no_date_takes_the_moment_it_was_written(
        self, store: Scratch, player: int
    ) -> None:
        [written] = a.grant(store.connection, [
            a.Award(kind="held_top", holder=player, game=GAME)
        ])
        store.connection.commit()
        assert written.earned is not None
        assert abs(written.earned - datetime.now(timezone.utc)) < timedelta(minutes=5)

    def test_deleting_a_game_removes_its_awards_and_leaves_the_rest(
        self, store: Scratch, player: int
    ) -> None:
        """§2.9: a row goes with its game, by the store's cascade, or by the admin's
        hand -- and by nothing else."""
        a.grant(store.connection, [
            a.Award(kind="held_top", holder=player, game=GAME),
            a.Award(kind="held_top", holder=player, game=OTHER),
        ])
        with store.connection.cursor() as cursor:
            cursor.execute("delete from games where hash = %s", (GAME,))
        store.connection.commit()
        assert [award.game for award in a.held(store.connection, holder=player)] == [OTHER]


class TestReadingThemBack:
    def test_a_persons_badges_come_back_oldest_first(
        self, store: Scratch, player: int
    ) -> None:
        older = datetime(2026, 1, 2, tzinfo=timezone.utc)
        newer = datetime(2026, 5, 6, tzinfo=timezone.utc)
        a.grant(store.connection, [
            a.Award(kind="grand_tour", holder=player, earned=newer),
            a.Award(kind="held_top", holder=player, game=GAME, earned=older),
        ])
        store.connection.commit()
        found = a.held(store.connection, holder=player)
        assert [award.kind for award in found] == ["held_top", "grand_tour"]
        assert found[0].game == GAME and found[1].game is None

    def test_a_games_badges_include_the_ones_with_no_holder(
        self, store: Scratch, player: int
    ) -> None:
        a.grant(store.connection, [
            a.Award(kind="crowd_pleaser", game=GAME, evidence={"voters": 9}),
            a.Award(kind="held_top", holder=player, game=GAME),
            a.Award(kind="held_top", holder=player, game=OTHER),
        ])
        store.connection.commit()
        found = a.held(store.connection, game=GAME)
        assert sorted(award.kind for award in found) == ["crowd_pleaser", "held_top"]
        assert [award.evidence for award in found if award.holder is None] == [
            {"voters": 9}
        ]

    def test_a_read_with_no_filter_is_refused(self, store: Scratch) -> None:
        """An unfiltered read is the whole table, which no page wants."""
        with pytest.raises(a.AwardError):
            a.held(store.connection)


# ------------------------------------------------- the rule for quoting a current score


class TestScoreEvidence:
    def test_a_current_measurement_gives_the_whole_quotation(
        self, store: Scratch
    ) -> None:
        store.measurement(GAME, "current")
        quoted = a.score_evidence(store.connection, GAME)
        assert quoted is not None
        assert (quoted.headline, quoted.ci_low, quoted.ci_high) == (HEADLINE, LOW, HIGH)
        assert (quoted.weights, quoted.spec_hash) == ("v10", GAME)
        assert quoted.basis == BASIS

    @pytest.mark.parametrize("state", ["re-score", "replay", "missing"])
    def test_a_measurement_that_is_not_current_answers_nothing(
        self, store: Scratch, state: str
    ) -> None:
        """Every state but `current` is *stale*, never a value. `Staleness.needs` is
        where those two words come from and this reader asks nothing else of them."""
        store.measurement(GAME, state, headline=None, ci_low=None, ci_high=None)
        assert a.score_evidence(store.connection, GAME) is None

    def test_a_stale_row_that_still_holds_a_number_is_not_quoted_either(
        self, store: Scratch
    ) -> None:
        """The loader nulls the headline when a measurement goes stale, and this is what
        makes the rule the reader's rather than the writer's."""
        store.measurement(GAME, "replay")
        assert a.score_evidence(store.connection, GAME) is None

    def test_a_current_row_with_no_headline_answers_nothing(
        self, store: Scratch
    ) -> None:
        store.measurement(GAME, "current", headline=None)
        assert a.score_evidence(store.connection, GAME) is None

    def test_a_game_that_has_never_been_measured_answers_nothing(
        self, store: Scratch
    ) -> None:
        assert a.score_evidence(store.connection, GAME) is None

    def test_a_game_the_store_does_not_hold_answers_nothing(
        self, store: Scratch
    ) -> None:
        assert a.score_evidence(store.connection, "nosuchgame01") is None

    def test_a_current_measurement_taken_before_the_range_was_stored_has_none(
        self, store: Scratch
    ) -> None:
        """A row loaded before `0008` ran, or a corpus too small to resample: the
        headline can be quoted and the range cannot."""
        store.measurement(GAME, "current", ci_low=None, ci_high=None)
        quoted = a.score_evidence(store.connection, GAME)
        assert quoted is not None and not quoted.ranged
        assert quoted.headline == HEADLINE
