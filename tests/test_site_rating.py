"""The site's one rating table per game: Glicko-2, the seeding, the top, and the fold.

Task 1.11 of `docs/plan-website.md`, against §5. `tests/test_site.py` was already long
before this arrived, so the table's tests live here; the two files share nothing but a
scratch schema, which is small enough to keep in both rather than couple them through a
conftest.

Two halves. The first needs no database at all, because §5's arithmetic is pure: the
update, the seeding, the top query and the fold are functions over a table and a list of
matches. The published Glicko-2 example is pinned first, since a rating system whose
arithmetic drifts is worse than none. The second half is the thin database layer, which
runs against a real Postgres and skips with a reason without one -- each test inside a
scratch schema it drops, the recipe task 1.2 settled.

Nothing here plays a game or computes a move. Ratings read match rows.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
import site_store

from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.db import connect

GAME = "abc123def456"

# One picker ladder in the shape `runs/opponents/<hash>.json` has, which is where every
# computer player enters from. Its ratings sum to zero, as a real one's do, because the
# fit centres on the roster's mean.
LADDER = {
    "ruleset_hash": GAME,
    "standings": [
        {"name": "minimax:nodes=6400", "rating": 300.0, "games": 24},
        {"name": "greedy", "rating": 0.0, "games": 24},
        {"name": "random", "rating": -300.0, "games": 24},
    ],
}


# The same game rated again: every rung moved, and not all by the same amount.
RERUN = {
    "standings": [
        {"name": "minimax:nodes=6400", "rating": 250.0},
        {"name": "greedy", "rating": 40.0},
        {"name": "random", "rating": -290.0},
    ]
}


def _table(*players: r.Player) -> r.Table:
    return {player.key: player for player in players}


# ------------------------------------------------------------------ the published update


class TestGlicko2:
    def test_the_published_worked_example(self) -> None:
        """Glickman's own numbers, to the digits his paper prints.

        A player at 1500/200 with the default volatility, against three opponents in one
        rating period, comes back at mu' = -0.2069, phi' = 0.8722, sigma' = 0.05999.
        `rate_period` is the general form rather than the one-match one the site actually
        calls exactly so that this example, which has three results in it, can be pinned.

        Two of the paper's printed values are its own rounding rather than the answer.
        Its Elo-scale 1464.06 is the *rounded* mu' converted back; carried through, mu' is
        -0.2069397 and the rating is 1464.05. Its sigma' of 0.05999 is 0.0599960 cut at
        five places. So the internal values carry four digits here and the two rounded
        ones are pinned a digit further out, where they are exact.
        """
        player = r.Player(r.person(1), 1500.0, 200.0, 0.06)
        period = [
            (r.Player(r.person(2), 1400.0, 30.0, 0.06), 1.0),
            (r.Player(r.person(3), 1550.0, 100.0, 0.06), 0.0),
            (r.Player(r.person(4), 1700.0, 300.0, 0.06), 0.0),
        ]
        after = r.rate_period(player, period)
        mu = (after.rating - r.SITE_ORIGIN) / r.GLICKO2_SCALE
        phi = after.deviation / r.GLICKO2_SCALE
        assert round(mu, 4) == -0.2069
        assert round(phi, 4) == 0.8722
        assert round(after.volatility, 6) == 0.059996
        assert round(after.rating, 2) == 1464.05
        assert round(after.deviation, 2) == 151.52
        assert after.matches == 3

    def test_the_site_calls_it_with_one_match(self) -> None:
        """A rating period of one match, which is what §5 asks for."""
        player = r.Player(r.person(1), 1500.0, 200.0, 0.06)
        opponent = r.Player(r.person(2), 1400.0, 30.0, 0.06)
        assert r.update(player, opponent, 1.0) == r.rate_period(player, [(opponent, 1.0)])

    def test_the_update_ignores_the_seat(self) -> None:
        """As Elo does. The seat stays on the match row for the readings that want it."""
        person = r.Player(r.person(1), 1500.0, 350.0, 0.06)
        machine = r.Player(r.computer("greedy"), 1800.0, 100.0, 0.0)
        first = r.apply_match(
            _table(person, machine), r.Finished("m", person.key, machine.key, 1.0)
        )[0]
        second = r.apply_match(
            _table(person, machine), r.Finished("m", machine.key, person.key, 0.0)
        )[0]
        assert first == second


# ------------------------------------------------------------------------ what a win does


class TestAWinAgainstTheTop:
    def test_it_moves_both_ratings_in_the_right_directions(self) -> None:
        """§5's reason for a live table: human results must move computer players."""
        person = r.Player(r.person(1), 1500.0, 350.0, 0.06)
        machine = r.Player(r.computer("minimax:nodes=6400"), 1800.0, 100.0, 0.0)
        table = _table(person, machine)
        assert r.top(table) == (machine,)

        after, moved = r.apply_match(table, r.Finished("m1", person.key, machine.key, 1.0))

        assert after[person.key].rating > person.rating
        assert after[machine.key].rating < machine.rating
        assert moved.seat0.before == person and moved.seat0.after == after[person.key]
        assert moved.seat1.before == machine and moved.seat1.after == after[machine.key]

    def test_a_newcomers_win_moves_the_computer_player_less(self) -> None:
        """The discount is the newcomer's own deviation, which the update applies itself.

        §5 leans on this: a deviation of 100 on the computer player means a single upset
        moves the top by about 20 Elo and not 50, and an uncertain winner moves it less
        still.
        """
        machine = r.Player(r.computer("greedy"), 1800.0, 100.0, 0.0)
        newcomer = r.Player(r.person(1), 1500.0, 350.0, 0.06)
        settled = r.Player(r.person(2), 1500.0, 80.0, 0.06)

        by_newcomer = r.update(machine, newcomer, 0.0)
        by_settled = r.update(machine, settled, 0.0)

        assert machine.rating - by_newcomer.rating < machine.rating - by_settled.rating
        assert 10.0 < machine.rating - by_newcomer.rating < 30.0

    @pytest.mark.parametrize(
        ("result", "person_moves", "machine_moves"),
        [(0.5, "up", "down"), (0.0, "down", None)],
        ids=["a-draw", "a-loss"],
    )
    def test_a_draw_or_a_loss_moves_the_person_the_way_it_should(
        self, result: float, person_moves: str, machine_moves: str | None
    ) -> None:
        """A draw is half a point each, and clears nothing; the user's word is *beaten*,
        which is 1.5's to enforce. A loss costs the person rating, which is what keeps the
        *Unbeaten* list from clearing in an afternoon (§5). Q754 merged the two tests."""
        person = r.Player(r.person(1), 1500.0, 120.0, 0.06)
        machine = r.Player(r.computer("greedy"), 1800.0, 100.0, 0.0)
        after, _ = r.apply_match(
            _table(person, machine), r.Finished("m", person.key, machine.key, result)
        )
        assert (after[person.key].rating > person.rating) == (person_moves == "up")
        assert (after[person.key].rating < person.rating) == (person_moves == "down")
        if machine_moves == "down":
            assert after[machine.key].rating < machine.rating


class TestDeviation:
    def test_a_computer_players_deviation_never_inflates(self) -> None:
        """§5: its strength does not change, so its uncertainty cannot grow with time.

        The pinned volatility is what makes that arithmetic rather than an approximation
        -- the inflation term of the update is exactly the volatility, so at zero there is
        none. The worst case for it is a match against a wildly uncertain opponent, where
        the match itself says almost nothing, so it is the one checked.
        """
        machine = r.Player(r.computer("greedy"), 1800.0, 100.0, r.COMPUTER_VOLATILITY)
        wild = r.Player(r.person(1), 1500.0, 350.0, 0.06)
        running = machine
        for _ in range(20):
            before = running
            running = r.update(running, wild, 0.5)
            assert running.deviation <= before.deviation
        assert r.rate_period(machine, []).deviation == machine.deviation

    def test_a_person_is_provisional_until_the_deviation_falls_under_110(self) -> None:
        assert not r.newcomer(r.person(1)).established
        assert r.Player(r.person(1), 1500.0, 109.9, 0.06).established
        assert not r.Player(r.person(1), 1500.0, 110.0, 0.06).established

    def test_a_computer_player_is_established_from_its_seed(self) -> None:
        seeded = r.seeds_from_ladder(r.rungs(LADDER))
        assert all(player.established for player in seeded.values())

    def test_a_provisional_person_still_moves_their_opponent(self) -> None:
        machine = r.Player(r.computer("greedy"), 1800.0, 100.0, 0.0)
        newcomer = r.newcomer(r.person(1))
        assert r.update(machine, newcomer, 0.0).rating < machine.rating


# ---------------------------------------------------------------------------- seeding


class TestSeeding:
    def test_a_ladder_enters_at_its_rating_plus_1500(self) -> None:
        """The site's scale is the ladder's plus 1500, so a newcomer sits mid-roster."""
        seeded = r.seeds_from_ladder(r.rungs(LADDER))
        assert seeded[r.computer("greedy")].rating == 1500.0
        assert seeded[r.computer("minimax:nodes=6400")].rating == 1800.0
        assert seeded[r.computer("random")].rating == 1200.0
        assert all(player.deviation == 100.0 for player in seeded.values())
        assert all(player.volatility == r.COMPUTER_VOLATILITY for player in seeded.values())
        assert all(player.matches == 0 for player in seeded.values())

    def test_a_second_seeding_adds_nothing(self) -> None:
        """The loader adds computer players it has not seen and never touches the rest."""
        table = r.seeds_from_ladder(r.rungs(LADDER))
        assert r.seeds_from_ladder(r.rungs(LADDER), table) == {}

    def test_a_re_rated_ladder_reseeds_every_row_nobody_has_played(self) -> None:
        """L3's reseed rule (`plan-leaderboard.md` §2.3): an unplayed row is only its seed.

        So a newer ladder replaces it. Before L3 a re-rated ladder reseeded nothing, and a
        table nobody had played kept its first ladder's numbers for good.
        """
        table = r.seeds_from_ladder(r.rungs(LADDER))
        found = r.seeding(r.rungs(RERUN), table)
        assert found.added == () and found.kept == 0
        # The spine's mean is 0, so the seats are the ratings plus 1500.
        assert {row.key.agent: row.rating for row in found.reseeded} == {
            "minimax:nodes=6400": 1750.0, "greedy": 1540.0, "random": 1210.0,
        }
        assert all(row.deviation == r.COMPUTER_DEVIATION and row.matches == 0
                   for row in found.reseeded)

    def test_a_reseed_leaves_the_rows_people_have_played(self) -> None:
        """A played row keeps the number people gave it, and fixes the scale the rest
        arrive on: the newer ladder's gaps, hung from the played rung."""
        table = r.seeds_from_ladder(r.rungs(LADDER))
        top = r.computer("minimax:nodes=6400")
        played = r.Player(top, 1650.0, 90.0, 0.0, matches=3)
        table[top] = played
        found = r.seeding(r.rungs(RERUN), table)
        assert top not in {row.key for row in found.written}
        assert found.kept == 1
        # The shift is 1650 - 250 = 1400, read off the one row people placed.
        assert {row.key.agent: row.rating for row in found.reseeded} == {
            "greedy": 1440.0, "random": 1110.0,
        }
        after = {**table, **r.seeds_from_ladder(r.rungs(RERUN), table)}
        assert after[top] == played

    def test_a_second_reading_of_a_newer_ladder_writes_nothing(self) -> None:
        """Idempotent: the shift reads only played rows, which a reseed never writes."""
        table = r.seeds_from_ladder(r.rungs(LADDER))
        top = r.computer("minimax:nodes=6400")
        table[top] = r.Player(top, 1650.0, 90.0, 0.0, matches=3)
        rerun = {"standings": [{"name": "mcts:nodes=1600", "rating": 610.0},
                               *RERUN["standings"]]}
        table.update(r.seeds_from_ladder(r.rungs(rerun), table))
        assert r.seeds_from_ladder(r.rungs(rerun), table) == {}
        assert r.seeding(r.rungs(rerun), table).kept == 4

    def test_a_row_the_newer_ladder_does_not_name_is_left_alone(self) -> None:
        """A player the picker no longer offers keeps its row: nothing is ever removed."""
        table = r.seeds_from_ladder(r.rungs(LADDER))
        rerun = {"standings": [{"name": "greedy", "rating": 10.0},
                               {"name": "random", "rating": -10.0}]}
        written = r.seeds_from_ladder(r.rungs(rerun), table)
        assert set(written) == {r.computer("greedy"), r.computer("random")}

    def test_a_later_arrival_is_translated_by_the_shared_rungs(self) -> None:
        """§5: the ladder's zero moves when the roster does, so the shift is measured.

        The second run's roster has a fourth player in it, which pulls every rung's number
        down by 75 without anybody having got weaker. The arrival must land where the
        shared rungs say, not 75 Elo low; nobody has played, so the spine's mean is what
        says (§2.4), and the three old rows are already at their seats.
        """
        table = r.seeds_from_ladder(r.rungs(LADDER))
        rerun = {
            "standings": [
                {"name": "fairy-stockfish", "rating": 525.0},
                {"name": "minimax:nodes=6400", "rating": 225.0},
                {"name": "greedy", "rating": -75.0},
                {"name": "random", "rating": -375.0},
            ]
        }
        added = r.seeds_from_ladder(r.rungs(rerun), table)
        assert set(added) == {r.computer("fairy-stockfish")}
        assert added[r.computer("fairy-stockfish")].rating == pytest.approx(2100.0)

    def test_a_human_correction_on_the_spine_pulls_a_later_arrival_down(self) -> None:
        """The shift reads the table as it stands, not as it was seeded.

        People beat the top rung down 150 Elo, which is real information about that rung's
        strength and so about anything measured against it. That row is the only one people
        have played, so it alone fixes the scale (L3): the arrival lands 300 above it, at
        1950 rather than 2100, and the unplayed spine rungs are reseeded beside it. (Before
        L3 the shift was a mean over every shared rung, played or not, and gave 2050.)
        """
        table = r.seeds_from_ladder(r.rungs(LADDER))
        beaten = table[r.computer("minimax:nodes=6400")]
        table[beaten.key] = r.Player(beaten.key, beaten.rating - 150.0, 90.0, 0.0, matches=4)
        rerun = {
            "standings": [
                {"name": "mcts:nodes=1600", "rating": 600.0},
                {"name": "minimax:nodes=6400", "rating": 300.0},
                {"name": "greedy", "rating": 0.0},
                {"name": "random", "rating": -300.0},
            ]
        }
        written = r.seeds_from_ladder(r.rungs(rerun), table)
        assert written[r.computer("mcts:nodes=1600")].rating == pytest.approx(1950.0)
        assert written[r.computer("greedy")].rating == pytest.approx(1350.0)
        assert beaten.key not in written

    def test_a_ladder_with_no_standings_seeds_nobody(self) -> None:
        assert r.rungs({}) == ()
        assert r.seeds_from_ladder(r.rungs({})) == {}


class TestTheAnchor:
    """§2.4 of `plan-leaderboard.md`: the first seeding puts 1500 at the spine's mean."""

    SPINE = {
        "random": -500.0,
        "greedy": -300.0,
        "minimax:nodes=400": -100.0,
        "minimax:nodes=1600": 100.0,
        "minimax:nodes=6400": 300.0,
    }

    def _ladder(self, extra: dict[str, float]) -> tuple[r.Rung, ...]:
        rows = {**self.SPINE, **extra}
        # Centred on the whole roster's mean, as the fit centres a real ladder.
        mean = sum(rows.values()) / len(rows)
        return tuple(r.Rung(name, rating - mean) for name, rating in rows.items())

    def test_the_spines_mean_is_1500_whatever_else_is_on_the_roster(self) -> None:
        """Fourteen rules of thumb at 6,400 nodes no longer move a game's 1500."""
        thumbs = {f"minimax:h={index:012x},nodes=6400": 400.0 + index for index in range(14)}
        for extra in ({}, {"minimax:nodes=200000": 900.0}, thumbs):
            seats = r.seeds_from_ladder(self._ladder(extra))
            spine = [seats[r.computer(name)].rating for name in self.SPINE]
            assert sum(spine) / len(spine) == pytest.approx(r.SITE_ORIGIN)
            # And the gaps are the ladder's own.
            assert seats[r.computer("minimax:nodes=6400")].rating == pytest.approx(1900.0)

    def test_a_ladder_with_no_spine_player_centres_on_its_whole_roster(self) -> None:
        seats = r.seeds_from_ladder((r.Rung("a", 100.0), r.Rung("b", 300.0)))
        assert seats[r.computer("a")].rating == pytest.approx(1400.0)
        assert seats[r.computer("b")].rating == pytest.approx(1600.0)

    def test_a_played_table_keeps_its_scale(self) -> None:
        """A table people have played hangs from its played rows and is never re-anchored."""
        table = r.seeds_from_ladder(self._ladder({}))
        easy = r.computer("minimax:nodes=400")
        table[easy] = r.Player(easy, 1777.0, 95.0, 0.0, matches=2)
        seats = r.seeds_from_ladder(self._ladder({"minimax:nodes=200000": 900.0}), table)
        # The newer ladder's gap from easy to hardest is 1,000; easy stays at 1777.
        assert seats[r.computer("minimax:nodes=200000")].rating == pytest.approx(2777.0)
        assert easy not in seats


# -------------------------------------------------------------------------- the top


class TestTheTop:
    def test_a_tied_top_comes_back_whole(self) -> None:
        """A tie the ladder could not break stays a tie until a match breaks it."""
        table = _table(
            r.Player(r.computer("a"), 1800.0, 100.0, 0.0),
            r.Player(r.computer("b"), 1796.0, 100.0, 0.0),
            r.Player(r.computer("c"), 1794.0, 100.0, 0.0),
        )
        assert {player.key.agent for player in r.top(table)} == {"a", "b"}
        assert r.is_top(table, r.computer("a")) and r.is_top(table, r.computer("b"))
        assert not r.is_top(table, r.computer("c"))

    def test_the_band_is_five_elo(self) -> None:
        table = _table(
            r.Player(r.computer("a"), 1800.0, 100.0, 0.0),
            r.Player(r.computer("b"), 1795.0, 100.0, 0.0),
        )
        assert len(r.top(table)) == 2
        table[r.computer("b")] = r.Player(r.computer("b"), 1794.99, 100.0, 0.0)
        assert len(r.top(table)) == 1

    def test_a_person_is_never_the_top(self) -> None:
        """The top is read among computer players, however strong a person gets."""
        table = _table(
            r.Player(r.person(1), 2400.0, 60.0, 0.06),
            r.Player(r.computer("greedy"), 1800.0, 100.0, 0.0),
        )
        assert [player.key.agent for player in r.top(table)] == ["greedy"]
        assert not r.is_top(table, r.person(1))

    def test_a_game_with_no_computer_players_has_no_top(self) -> None:
        """Which is a game with no ladder, and so no leaderboard mode at all."""
        assert r.top(_table(r.newcomer(r.person(1)))) == ()

    def test_the_top_is_read_live_and_can_change_hands(self) -> None:
        table = _table(
            r.Player(r.computer("a"), 1800.0, 100.0, 0.0),
            r.Player(r.computer("b"), 1780.0, 100.0, 0.0),
            r.Player(r.person(1), 1500.0, 120.0, 0.06),
        )
        assert r.top(table)[0].key == r.computer("a")
        for index in range(3):
            table, _ = r.apply_match(
                table, r.Finished(f"m{index}", r.person(1), r.computer("a"), 1.0)
            )
        assert r.top(table)[0].key == r.computer("b")

    def test_the_order_is_strongest_first(self) -> None:
        table = _table(
            r.Player(r.computer("weak"), 1200.0, 100.0, 0.0),
            r.Player(r.computer("strong"), 1800.0, 100.0, 0.0),
            r.Player(r.computer("middle"), 1500.0, 100.0, 0.0),
        )
        assert [player.key.agent for player in r.computers(table)] == [
            "strong",
            "middle",
            "weak",
        ]


# --------------------------------------------------------------------------- the fold


# A fixed list of matches, the same one every replay gets. Two people, three computer
# players, wins, losses and a draw, and one person-versus-person match.
MATCHES = (
    r.Finished("m1", r.person(1), r.computer("random"), 1.0),
    r.Finished("m2", r.computer("greedy"), r.person(1), 1.0),
    r.Finished("m3", r.person(2), r.computer("minimax:nodes=6400"), 0.5),
    r.Finished("m4", r.person(1), r.computer("minimax:nodes=6400"), 1.0),
    r.Finished("m5", r.person(1), r.person(2), 0.0),
    r.Finished("m6", r.computer("random"), r.person(2), 0.0),
)


class TestTheFold:
    def test_replaying_the_rows_reproduces_the_table_to_the_digit(self) -> None:
        seeds = r.seeds_from_ladder(r.rungs(LADDER))
        first = r.replay(seeds, MATCHES)
        second = r.replay(seeds, MATCHES)
        assert first == second
        assert set(first) == set(seeds) | {r.person(1), r.person(2)}

    def test_a_replay_leaves_the_seed_alone_and_reads_the_order(self) -> None:
        """Three facts about one seed table and one list of matches (Q754 merged the tests).

        The seed is not mutated by a replay, otherwise the second replay would start
        somewhere the first did not; a seeded computer player keeps its seed when no match
        names it; and the fold is sequential, so a different finish order is a different
        table, which is why `finished` orders by the timestamp with the id behind it.
        """
        seeds = r.seeds_from_ladder(r.rungs(LADDER))
        before = dict(seeds)
        forwards = r.replay(seeds, MATCHES)
        assert seeds == before

        played = [match for match in MATCHES if r.computer("greedy") not in (match.seat0, match.seat1)]
        table = r.replay(seeds, played)
        assert table[r.computer("greedy")] == seeds[r.computer("greedy")]

        backwards = r.replay(seeds, tuple(reversed(MATCHES)))
        assert forwards != backwards

    def test_a_person_enters_at_1500_when_their_first_match_lands(self) -> None:
        table = r.replay(r.seeds_from_ladder(r.rungs(LADDER)), ())
        assert r.person(1) not in table
        played = r.replay(
            r.seeds_from_ladder(r.rungs(LADDER)),
            (r.Finished("m", r.person(1), r.computer("random"), 1.0),),
        )
        assert played[r.person(1)].matches == 1
        assert played[r.person(1)].rating > r.NEWCOMER_RATING

    def test_a_computer_player_the_table_has_not_seeded_is_refused(self) -> None:
        """Inventing 1500 for it would put a strong player in the middle of its own table."""
        with pytest.raises(r.RatingError) as error:
            r.replay({}, (r.Finished("m", r.person(1), r.computer("greedy"), 1.0),))
        assert "picker ladder" in str(error.value)

    def test_a_player_is_a_person_or_a_computer_player_and_not_both(self) -> None:
        """The DDL's check constraint, in Python, where the fold can trip over it."""
        for user, agent in ((None, None), (1, "greedy")):
            with pytest.raises(r.RatingError):
                r.PlayerKey(user=user, agent=agent)


# ------------------------------------------------------------- the database, thinly

# The second half. Each test takes the session's migrated store (`tests/site_store.py`),
# emptied first, on a schema of its own name so these can run against a server that has
# other things on it -- a real server, since `py-pglite` does not work here (task 1.2).


@contextlib.contextmanager
def _store() -> Iterator[Any]:
    """The migrated store, emptied, with one game and two people already in it."""
    url, name = site_store.emptied("no database to fold match rows in")
    connection = connect(url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, '{}'::jsonb, 'a game', 'corpus')",
                (GAME,),
            )
            for email in ("one@example.test", "two@example.test"):
                cursor.execute("insert into users (email) values (%s)", (email,))
        connection.commit()
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _people(connection: Any) -> tuple[int, int]:
    with connection.cursor() as cursor:
        cursor.execute("select id from users order by id")
        return tuple(row[0] for row in cursor.fetchall())  # type: ignore[return-value]


def _match(
    connection: Any,
    match_id: str,
    seat0: r.PlayerKey,
    seat1: r.PlayerKey,
    result: float | None,
    *,
    mode: str = "leaderboard",
    minute: int = 0,
    source: str | None = None,
) -> None:
    """One `matches` row, finished at a stated minute so the fold's order is stated too."""
    finished = None if result is None else f"2026-09-11 12:{minute:02d}:00+00"
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into matches"
            " (id, game, mode, seat0_user, seat0_agent, seat1_user, seat1_agent,"
            "  seed, result, finished, source)"
            " values (%s, %s, %s, %s, %s, %s, %s, 0, %s, %s, %s)",
            (
                match_id,
                GAME,
                mode,
                seat0.user,
                seat0.agent,
                seat1.user,
                seat1.agent,
                result,
                finished,
                source,
            ),
        )


class TestTheStore:
    def test_seeding_writes_one_row_per_computer_player_and_a_second_run_adds_none(
        self,
    ) -> None:
        with _store() as connection:
            added = r.seed(connection, GAME, LADDER)
            connection.commit()
            assert {player.key.agent for player in added} == {
                "minimax:nodes=6400",
                "greedy",
                "random",
            }
            table = r.load(connection, GAME)
            assert table[r.computer("greedy")].rating == 1500.0
            assert table[r.computer("greedy")].deviation == 100.0
            assert table[r.computer("greedy")].established

            assert r.seed(connection, GAME, LADDER) == ()
            assert r.load(connection, GAME) == table

    def test_a_reseed_leaves_played_rows_and_replaces_the_rest(self) -> None:
        """§2.3's rule against the store: the row people played keeps their number, the
        rows nobody played take the newer ladder's seats, a newcomer is added, and a second
        reseed from the same file writes nothing."""
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            _match(connection, "m1", r.person(first), r.computer("minimax:nodes=6400"), 1.0)
            r.finish(connection, "m1")
            connection.commit()
            before = r.load(connection, GAME)
            top = before[r.computer("minimax:nodes=6400")]
            assert top.matches == 1

            newer = {"standings": [{"name": "mcts:nodes=1600", "rating": 600.0},
                                   *RERUN["standings"]]}
            seated = r.reseed(connection, GAME, newer)
            connection.commit()
            assert [row.key.agent for row in seated.added] == ["mcts:nodes=1600"]
            assert {row.key.agent for row in seated.reseeded} == {"greedy", "random"}
            assert seated.kept == 1

            after = r.load(connection, GAME)
            assert after[top.key] == top
            assert after[r.person(first)] == before[r.person(first)]
            shift = top.rating - 250.0
            assert after[r.computer("greedy")].rating == pytest.approx(40.0 + shift)
            assert after[r.computer("mcts:nodes=1600")].rating == pytest.approx(600.0 + shift)

            again = r.reseed(connection, GAME, newer)
            assert again.written == () and again.kept == 4
            assert r.load(connection, GAME) == after

    def test_a_finished_match_moves_both_seats_and_stamps_the_row(self) -> None:
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            top_before = r.top(r.load(connection, GAME))
            assert [player.key.agent for player in top_before] == ["minimax:nodes=6400"]

            _match(connection, "m1", r.person(first), r.computer("minimax:nodes=6400"), 1.0)
            moved = r.finish(connection, "m1")
            connection.commit()

            assert moved is not None
            assert moved.seat0.after.rating > r.NEWCOMER_RATING
            assert moved.seat1.after.rating < 1800.0

            table = r.load(connection, GAME)
            assert table[r.person(first)] == moved.seat0.after
            assert table[r.computer("minimax:nodes=6400")] == moved.seat1.after

            with connection.cursor() as cursor:
                cursor.execute(
                    "select seat0_rating_before, seat0_rating_after,"
                    " seat1_rating_before, seat1_rating_after from matches where id = 'm1'"
                )
                before0, after0, before1, after1 = cursor.fetchone()
            assert (before0, before1) == (r.NEWCOMER_RATING, 1800.0)
            assert after0 == pytest.approx(moved.seat0.after.rating)
            assert after1 == pytest.approx(moved.seat1.after.rating)

    @pytest.mark.parametrize(
        ("match_id", "outcome", "extra"),
        [("s1", 1.0, {"mode": "sandbox"}), ("r1", None, {})],
        ids=["a-sandbox-match", "a-match-still-running"],
    )
    def test_a_match_that_is_not_rated_moves_nothing(
        self, match_id: str, outcome: float | None, extra: dict[str, str]
    ) -> None:
        """A sandbox match, and a match still running (`outcome` None), are finished by
        no one's rating (Q754 merged the two tests of it)."""
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            before = r.load(connection, GAME)
            _match(
                connection, match_id, r.person(first), r.computer("greedy"), outcome, **extra
            )
            assert r.finish(connection, match_id) is None
            assert r.load(connection, GAME) == before

    def test_a_local_app_row_is_rated_by_nothing(self) -> None:
        """The 54 pre-site games enter no fold and clear nothing (§3)."""
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            before = r.load(connection, GAME)
            _match(
                connection,
                "l1",
                r.person(first),
                r.computer("greedy"),
                1.0,
                source="local-app",
            )
            assert r.finish(connection, "l1") is None
            assert r.finished(connection, GAME) == ()
            assert r.load(connection, GAME) == before

    def test_a_second_finish_on_the_same_match_is_refused(self) -> None:
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            _match(connection, "m1", r.person(first), r.computer("greedy"), 1.0)
            r.finish(connection, "m1")
            with pytest.raises(r.RatingError) as error:
                r.finish(connection, "m1")
            assert "already moved" in str(error.value)

    def test_an_unknown_match_is_refused(self) -> None:
        with _store() as connection, pytest.raises(r.RatingError):
            r.finish(connection, "nothing")

    def test_the_rows_come_back_in_finish_order(self) -> None:
        with _store() as connection:
            first, second = _people(connection)
            _match(connection, "late", r.person(first), r.person(second), 1.0, minute=30)
            _match(connection, "early", r.person(first), r.person(second), 0.0, minute=10)
            assert [match.id for match in r.finished(connection, GAME)] == ["early", "late"]


class TestTheRebuild:
    def test_replaying_the_rows_reproduces_the_table(self) -> None:
        """The repair §5 names: a bug in the update is fixed by replaying the matches."""
        with _store() as connection:
            first, second = _people(connection)
            r.seed(connection, GAME, LADDER)
            plan = (
                ("m1", r.person(first), r.computer("random"), 1.0),
                ("m2", r.computer("greedy"), r.person(first), 1.0),
                ("m3", r.person(second), r.computer("minimax:nodes=6400"), 0.5),
                ("m4", r.person(first), r.person(second), 0.0),
            )
            for minute, (match_id, seat0, seat1, result) in enumerate(plan):
                _match(connection, match_id, seat0, seat1, result, minute=minute)
                r.finish(connection, match_id)
            connection.commit()
            played = r.load(connection, GAME)

            # Scribble over every row, then fold the match rows back over the wreckage.
            with connection.cursor() as cursor:
                cursor.execute("update ratings set rating = 1, deviation = 9, matches = 0")
            rebuilt = r.rebuild(connection, GAME)

            assert rebuilt == played
            assert r.load(connection, GAME) == played

    def test_the_seed_survives_a_later_ladder_run(self) -> None:
        """Recovered from the before-columns, which no update ever writes twice.

        The ladder file is re-run and every rung's number moves. A rebuild that re-seeded
        from the file would give the played rungs different histories; this one does not,
        because a player's seed is what their first rated match saw.
        """
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            _match(connection, "m1", r.person(first), r.computer("greedy"), 1.0)
            r.finish(connection, "m1")
            connection.commit()
            played = r.load(connection, GAME)

            seeds = r.seeds_for(connection, GAME)
            assert seeds[r.computer("greedy")].rating == 1500.0
            assert seeds[r.computer("greedy")].matches == 0
            assert seeds[r.person(first)] == r.newcomer(r.person(first))
            assert r.rebuild(connection, GAME) == played

    def test_an_untouched_computer_player_keeps_its_seed_across_a_rebuild(self) -> None:
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            seeded = r.load(connection, GAME)[r.computer("minimax:nodes=6400")]
            _match(connection, "m1", r.person(first), r.computer("random"), 1.0)
            r.finish(connection, "m1")
            rebuilt = r.rebuild(connection, GAME)
            assert rebuilt[r.computer("minimax:nodes=6400")] == seeded

    def test_a_fold_over_the_rows_still_rebuilds_the_table_after_a_reseed(self) -> None:
        """L3: a reseed writes only rows with no matches, which `seeds_for` reads as their
        own seed, and a reseeded row played later has its seat in that match's
        before-columns. So the table stays a fold of its match rows."""
        with _store() as connection:
            first, second = _people(connection)
            r.seed(connection, GAME, LADDER)
            _match(connection, "m1", r.person(first), r.computer("minimax:nodes=6400"), 1.0,
                   minute=1)
            r.finish(connection, "m1")
            newer = {"standings": [{"name": "mcts:nodes=1600", "rating": 600.0},
                                   *RERUN["standings"]]}
            seated = r.reseed(connection, GAME, newer)
            assert {row.key.agent for row in seated.reseeded} == {"greedy", "random"}
            # Played after the reseed: one reseeded row, one added, one never touched.
            _match(connection, "m2", r.computer("greedy"), r.person(second), 0.0, minute=2)
            r.finish(connection, "m2")
            _match(connection, "m3", r.person(first), r.computer("mcts:nodes=1600"), 0.5,
                   minute=3)
            r.finish(connection, "m3")
            connection.commit()
            played = r.load(connection, GAME)
            assert played[r.computer("random")].matches == 0, "reseeded, never played"

            # Scribble over every row a match moved; an unplayed row *is* its seed.
            with connection.cursor() as cursor:
                cursor.execute("update ratings set rating = 1, deviation = 9, matches = 0"
                               " where matches > 0")
            assert r.rebuild(connection, GAME) == played
            assert r.load(connection, GAME) == played

    def test_a_sandbox_row_changes_no_rebuild(self) -> None:
        with _store() as connection:
            first, _ = _people(connection)
            r.seed(connection, GAME, LADDER)
            _match(connection, "m1", r.person(first), r.computer("greedy"), 1.0)
            r.finish(connection, "m1")
            rated = r.rebuild(connection, GAME)
            _match(
                connection,
                "s1",
                r.person(first),
                r.computer("greedy"),
                1.0,
                mode="sandbox",
                minute=5,
            )
            assert r.rebuild(connection, GAME) == rated
