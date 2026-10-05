"""`chess_variant_sandbox/site/milestones.py`, box A1 of `docs/archive/plan-meta-game.md`.

Every row and table is built here as a plain value, the way `unbeaten.record_from` is
proved. The games are named alpha, beta and gamma, and no piece or ruleset is involved.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import milestones, rating

ADA = 1
BOB = 2
WEAK = "random"
MIDDLE = "greedy"
TOP = "minimax:nodes=400"

_ids = itertools.count()


def _day(day: int) -> datetime:
    return datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=day)


def _person(user: int) -> rating.PlayerKey:
    return rating.person(user)


def _computer(agent: str) -> rating.PlayerKey:
    return rating.computer(agent)


def _row(
    game: str,
    seat0: rating.PlayerKey,
    seat1: rating.PlayerKey,
    result: float,
    day: int,
    *,
    mode: str = match_rows.LEADERBOARD,
    source: str | None = None,
    finished: bool = True,
) -> match_rows.Row:
    """One match. `result` is seat 0's score; a match against `TOP` carries the top bit."""
    against_top = TOP in (seat0.agent, seat1.agent)
    return match_rows.Row(
        id=f"m{next(_ids):04d}",
        game=game,
        mode=mode,
        seat0=seat0,
        seat1=seat1,
        seed=0,
        plies=20,
        result=result if finished else None,
        reason="",
        left=False,
        top=against_top,
        top_players=(TOP,) if against_top else (),
        before=(None, None),
        source=source,
        started=_day(day),
        moved=None,
        finished=_day(day) if finished else None,
    )


def _ladder() -> rating.Table:
    """Three computer players, `TOP` alone at the top."""
    return {
        _computer(agent): rating.Player(_computer(agent), value, 100.0, 0.0)
        for agent, value in ((WEAK, 800.0), (MIDDLE, 1100.0), (TOP, 1500.0))
    }


def _cell(**bins: int) -> tuple[int, ...]:
    """A grid cell with every axis in bin 0 except the ones named."""
    return tuple(bins.get(axis.name, 0) for axis in grid.AXES)


TABLES: dict[str, rating.Table] = {"alpha": _ladder(), "beta": _ladder()}
CELLS = {"alpha": _cell(), "beta": _cell(board=1)}


def _kinds(reading: milestones.Reading) -> list[tuple[str, str]]:
    return [(line.kind, line.game) for line in reading.milestones]


def _read(rows: list[match_rows.Row], user: int = ADA) -> milestones.Reading:
    return milestones.read_from(user, rows, TABLES, CELLS)


def _everything() -> list[match_rows.Row]:
    """Rows on which ADA holds all seven kinds of milestone."""
    return [
        _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
        _row("alpha", _computer(MIDDLE), _person(ADA), 0.0, 2),
        _row("alpha", _computer(TOP), _person(ADA), 0.0, 3),
        _row("beta", _person(ADA), _computer(WEAK), 0.0, 4),
        _row("beta", _person(ADA), _person(BOB), 1.0, 5),
        _row("beta", _person(BOB), _person(ADA), 0.0, 6),
    ]


class TestWhatIsRead:
    def test_the_fixture_lights_every_kind(self) -> None:
        """Keeps the filtering tests below from passing on a reading that was empty anyway."""
        found = {line.kind for line in _read(_everything()).milestones}
        # The designer's four read designs rather than matches: `test_site_designer.py`.
        assert found == set(milestones.KINDS) - set(milestones.DESIGNER_KINDS)

    @pytest.mark.parametrize(
        "change",
        [
            {"mode": match_rows.SANDBOX},
            {"source": match_rows.LOCAL_APP},
        ],
        ids=["sandbox", "local-app"],
    )
    def test_sandbox_and_local_app_rows_are_read_by_nothing(self, change: dict[str, str]) -> None:
        """Remarked rows give exactly the reading of no rows at all, on every field."""
        rows = _everything()
        remarked = [replace(row, **change) for row in rows]
        assert _read(remarked) == _read([])
        assert _read(remarked + rows) == _read(rows)

    def test_an_unfinished_row_is_read_by_nothing(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 1.0, 1, finished=False)]
        assert _read(rows) == _read([])

    def test_the_order_rows_arrive_in_does_not_matter(self) -> None:
        rows = _everything()
        assert _read(list(reversed(rows))) == _read(rows)

    def test_milestones_are_newest_first_and_dated_by_the_completing_match(self) -> None:
        found = _read(_everything()).milestones
        assert [(line.kind, line.when) for line in found] == [
            (milestones.PERSON_BOTH_SEATS, _day(6)),
            (milestones.FIRST_PERSON, _day(5)),
            (milestones.FIRST_HERE, _day(4)),
            (milestones.EVERY_GAME, _day(4)),
            (milestones.EVERY_KIND, _day(4)),
            (milestones.WHOLE_LADDER, _day(3)),
            (milestones.BOTH_SEATS, _day(2)),
            (milestones.FIRST_HERE, _day(1)),
        ]

    def test_every_kind_names_its_rule(self) -> None:
        assert set(milestones.RULES) == set(milestones.KINDS)
        assert all(sentence.endswith(".") for sentence in milestones.RULES.values())


class TestBothSeats:
    def test_a_win_from_one_seat_is_not_enough(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("alpha", _person(ADA), _computer(TOP), 1.0, 2),
        ]
        assert (milestones.BOTH_SEATS, "alpha") not in _kinds(_read(rows))

    def test_a_draw_from_the_other_seat_is_not_a_win(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("alpha", _computer(WEAK), _person(ADA), 0.5, 2),
        ]
        assert (milestones.BOTH_SEATS, "alpha") not in _kinds(_read(rows))

    def test_one_line_per_game_however_many_wins(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("alpha", _computer(WEAK), _person(ADA), 0.0, 2),
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 3),
            _row("alpha", _computer(WEAK), _person(ADA), 0.0, 4),
        ]
        lines = [line for line in _read(rows).milestones if line.kind == milestones.BOTH_SEATS]
        assert [(line.when, line.seat) for line in lines] == [(_day(2), 1)]

    def test_wins_over_a_person_do_not_count(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("alpha", _person(BOB), _person(ADA), 0.0, 2),
        ]
        assert (milestones.BOTH_SEATS, "alpha") not in _kinds(_read(rows))


class TestTheWholeLadder:
    def test_two_rungs_of_three_are_not_enough(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(TOP), 1.0, 1),
            _row("alpha", _person(ADA), _computer(MIDDLE), 1.0, 2),
        ]
        assert (milestones.WHOLE_LADDER, "alpha") not in _kinds(_read(rows))

    def test_the_weakest_rung_completes_it_like_any_other(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(TOP), 1.0, 1),
            _row("alpha", _person(ADA), _computer(MIDDLE), 0.0, 2),
            _row("alpha", _person(ADA), _computer(MIDDLE), 1.0, 3),
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 4),
        ]
        lines = [line for line in _read(rows).milestones if line.kind == milestones.WHOLE_LADDER]
        assert [line.when for line in lines] == [_day(4)]

    def test_a_computer_player_not_on_the_table_counts_for_nothing(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer("mcts:nodes=50"), 1.0, 1),
            _row("alpha", _person(ADA), _computer(TOP), 1.0, 2),
            _row("alpha", _person(ADA), _computer(MIDDLE), 1.0, 3),
        ]
        assert (milestones.WHOLE_LADDER, "alpha") not in _kinds(_read(rows))


class TestTheFirstHumanGameHere:
    def test_somebody_elses_earlier_game_takes_it(self) -> None:
        rows = [
            _row("alpha", _person(BOB), _computer(WEAK), 0.0, 1),
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 2),
        ]
        assert (milestones.FIRST_HERE, "alpha") not in _kinds(_read(rows, ADA))
        assert (milestones.FIRST_HERE, "alpha") in _kinds(_read(rows, BOB))

    def test_a_loss_still_takes_it(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 0.0, 1)]
        assert (milestones.FIRST_HERE, "alpha") in _kinds(_read(rows))

    def test_a_game_between_computer_players_is_not_a_human_game(self) -> None:
        rows = [
            _row("alpha", _computer(WEAK), _computer(TOP), 0.0, 1),
            _row("alpha", _person(ADA), _computer(WEAK), 0.0, 2),
        ]
        assert (milestones.FIRST_HERE, "alpha") in _kinds(_read(rows))

    def test_a_sandbox_game_does_not_take_it_from_the_first_leaderboard_game(self) -> None:
        rows = [
            _row("alpha", _person(BOB), _computer(WEAK), 0.0, 1, mode=match_rows.SANDBOX),
            _row("alpha", _person(ADA), _computer(WEAK), 0.0, 2),
        ]
        assert (milestones.FIRST_HERE, "alpha") in _kinds(_read(rows, ADA))

    def test_both_people_in_a_first_game_between_people_hold_it(self) -> None:
        rows = [_row("alpha", _person(ADA), _person(BOB), 0.5, 1)]
        assert (milestones.FIRST_HERE, "alpha") in _kinds(_read(rows, ADA))
        assert (milestones.FIRST_HERE, "alpha") in _kinds(_read(rows, BOB))


class TestTheTour:
    def test_every_rated_game_tried_is_dated_at_the_last_first_try(self) -> None:
        rows = [
            _row("beta", _person(ADA), _computer(WEAK), 0.0, 1),
            _row("beta", _person(ADA), _computer(WEAK), 0.0, 2),
            _row("alpha", _person(ADA), _computer(WEAK), 0.0, 3),
        ]
        lines = [line for line in _read(rows).milestones if line.kind == milestones.EVERY_GAME]
        assert [(line.when, line.game) for line in lines] == [(_day(3), "alpha")]

    def test_one_rated_game_untried_leaves_it_open(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 0.0, 1)]
        assert milestones.EVERY_GAME not in {kind for kind, _ in _kinds(_read(rows))}

    def test_a_game_against_a_person_is_not_a_try(self) -> None:
        """A try means what it means on the profile's table, a game against a computer."""
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 0.0, 1),
            _row("beta", _person(ADA), _person(BOB), 1.0, 2),
        ]
        reading = _read(rows)
        assert milestones.EVERY_GAME not in {kind for kind, _ in _kinds(reading)}
        board = reading.still_open[0]
        assert [kind.untried for kind in board.kinds] == [(), ("beta",)]

    def test_an_unrated_game_is_not_asked_for(self) -> None:
        tables = {**TABLES, "gamma": {}}
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 0.0, 1),
            _row("beta", _person(ADA), _computer(WEAK), 0.0, 2),
        ]
        kinds = {line.kind for line in milestones.milestones_from(ADA, rows, tables, CELLS)}
        assert milestones.EVERY_GAME in kinds

    def test_every_kind_needs_one_game_from_each_bin_the_rated_games_cover(self) -> None:
        tables = {**TABLES, "gamma": _ladder()}
        cells = {**CELLS, "gamma": _cell(board=1, hand=1)}
        base = [_row("alpha", _person(ADA), _computer(WEAK), 0.0, 1)]
        without_drops = base + [_row("beta", _person(ADA), _computer(WEAK), 0.0, 2)]
        with_drops = base + [_row("gamma", _person(ADA), _computer(WEAK), 0.0, 3)]

        def kinds(rows: list[match_rows.Row]) -> set[str]:
            return {line.kind for line in milestones.milestones_from(ADA, rows, tables, cells)}

        assert milestones.EVERY_KIND not in kinds(without_drops)
        assert milestones.EVERY_KIND in kinds(with_drops)
        assert milestones.EVERY_GAME not in kinds(with_drops)

    def test_a_rated_game_with_no_cell_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            milestones.read_from(ADA, [], TABLES, {"alpha": _cell()})


class TestAgainstAPerson:
    def test_the_first_game_against_a_person_lights_for_either_seat(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("beta", _person(BOB), _person(ADA), 1.0, 2),
            _row("alpha", _person(ADA), _person(BOB), 1.0, 3),
        ]
        for user in (ADA, BOB):
            lines = [
                line for line in _read(rows, user).milestones
                if line.kind == milestones.FIRST_PERSON
            ]
            assert [line.when for line in lines] == [_day(2)]

    def test_a_win_over_a_person_from_each_seat_may_span_games(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _person(BOB), 1.0, 1),
            _row("beta", _person(BOB), _person(ADA), 0.5, 2),
            _row("beta", _person(BOB), _person(ADA), 0.0, 3),
        ]
        lines = [
            line for line in _read(rows).milestones if line.kind == milestones.PERSON_BOTH_SEATS
        ]
        assert [(line.when, line.game, line.seat) for line in lines] == [(_day(3), "beta", 1)]
        assert milestones.PERSON_BOTH_SEATS not in {kind for kind, _ in _kinds(_read(rows, BOB))}


class TestHighestPerson:
    @staticmethod
    def _table(*people: tuple[int, float, float]) -> rating.Table:
        table = _ladder()
        for user, value, deviation in people:
            table[_person(user)] = rating.Player(_person(user), value, deviation, 0.06)
        return table

    def test_the_highest_established_person_holds_it_whatever_the_computers_are(self) -> None:
        tables = {"alpha": self._table((ADA, 1300.0, 60.0), (BOB, 1200.0, 60.0))}
        assert milestones.highest_count(ADA, tables) == 1
        assert milestones.highest_count(BOB, tables) == 0

    def test_nobody_holds_it_until_established(self) -> None:
        tables = {"alpha": self._table((ADA, 1900.0, 300.0), (BOB, 1200.0, 60.0))}
        assert milestones.highest_count(ADA, tables) == 0
        assert milestones.highest_count(BOB, tables) == 1
        assert milestones.highest_people(self._table((ADA, 1900.0, 300.0))) == ()

    def test_a_tie_is_held_by_everyone_in_it(self) -> None:
        table = self._table((ADA, 1300.0, 60.0), (BOB, 1300.0, 60.0))
        assert [row.key.user for row in milestones.highest_people(table)] == [ADA, BOB]

    def test_counted_across_games(self) -> None:
        tables = {game: self._table((ADA, 1300.0, 60.0)) for game in ("alpha", "beta")}
        assert milestones.highest_count(ADA, tables) == 2


class TestStillOpen:
    def test_grouped_by_every_axis_in_grid_order(self) -> None:
        still = _read([]).still_open
        assert [axis.axis for axis in still] == [axis.name for axis in grid.AXES]

    def test_a_bin_no_rated_game_sits_in_is_left_out(self) -> None:
        board, army = _read([]).still_open[:2]
        assert [kind.bin for kind in board.kinds] == ["small", "medium"]
        assert [kind.bin for kind in army.kinds] == ["spare"]

    def test_a_newcomer_has_everything_open_and_no_kind_tried(self) -> None:
        board = _read([]).still_open[0]
        assert [
            (kind.tried, kind.untried, kind.unwon, kind.unbeaten) for kind in board.kinds
        ] == [
            (False, ("alpha",), ("alpha",), ("alpha",)),
            (False, ("beta",), ("beta",), ("beta",)),
        ]

    def test_a_loss_moves_a_game_from_untried_to_unwon(self) -> None:
        rows = [_row("beta", _person(ADA), _computer(WEAK), 0.0, 1)]
        medium = _read(rows).still_open[0].kinds[1]
        assert (medium.tried, medium.untried, medium.unwon) == (True, (), ("beta",))

    def test_somebody_elses_clear_takes_a_game_off_the_unbeaten_list(self) -> None:
        rows = [_row("alpha", _person(BOB), _computer(TOP), 1.0, 1)]
        small = _read(rows, ADA).still_open[0].kinds[0]
        assert (small.untried, small.unbeaten) == (("alpha",), ())

    def test_a_win_below_the_top_leaves_it_unbeaten_but_won(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 1.0, 1)]
        small = _read(rows).still_open[0].kinds[0]
        assert (small.untried, small.unwon, small.unbeaten) == ((), (), ("alpha",))


# --------------------------------------------------------- A3, the overall page

HANDLES = {ADA: "ada", BOB: "bob"}
GAMES = ("alpha", "beta")


def _standings(rows: list[match_rows.Row]) -> list[tuple[str, int, int, int]]:
    """Each person as the page reads them, in the order the reader returns."""
    return [
        (row.handle, row.counts.tried, row.counts.won, row.counts.top)
        for row in milestones.standings_from(rows, TABLES, HANDLES)
    ]


class TestWhoIsListed:
    """S3 under the user's decision of 2026-09-14: only people with a finished game."""

    def test_an_empty_site_lists_nobody(self) -> None:
        assert milestones.standings_from([], TABLES, HANDLES) == ()

    def test_somebody_who_has_finished_nothing_is_on_no_list(self) -> None:
        """§3's fifth test. Not a row of noughts at the bottom -- absent."""
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 1.0, 1)]
        assert [row.handle for row in milestones.standings_from(rows, TABLES, HANDLES)] \
            == ["ada"]

    def test_a_running_game_has_not_been_finished(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 1.0, 1, finished=False)]
        assert milestones.standings_from(rows, TABLES, HANDLES) == ()

    @pytest.mark.parametrize(
        "change",
        [
            {"mode": match_rows.SANDBOX},
            {"source": "local-app"},
        ],
    )
    def test_a_row_no_milestone_reads_puts_nobody_on_the_page(
        self, change: dict[str, object]
    ) -> None:
        rows = [
            replace(_row("alpha", _person(ADA), _computer(WEAK), 1.0, 1), **change)  # type: ignore[arg-type]
        ]
        assert milestones.standings_from(rows, TABLES, HANDLES) == ()

    def test_somebody_who_left_reads_as_the_store_names_them(self) -> None:
        """Never by the handle they had. `unbeaten.handles` coalesces it, and the reader
        prints whatever that map gave."""
        rows = [_row("alpha", _person(BOB), _computer(WEAK), 1.0, 1)]
        found = milestones.standings_from(rows, TABLES, {BOB: "a former player"})
        assert [row.handle for row in found] == ["a former player"]

    def test_a_person_the_map_has_no_name_for_is_still_a_row(self) -> None:
        rows = [_row("alpha", _person(BOB), _computer(WEAK), 1.0, 1)]
        assert [row.user for row in milestones.standings_from(rows, TABLES, {})] == [BOB]


class TestTheThreeCounts:
    def test_they_are_the_counts_a_profile_shows(self) -> None:
        """One reader, so a person cannot read differently on the two pages."""
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("beta", _person(ADA), _computer(WEAK), 0.0, 2),
        ]
        assert _standings(rows) == [("ada", 2, 1, 0)]

    def test_beating_the_top_is_counted_where_winning_is(self) -> None:
        rows = [_row("alpha", _computer(TOP), _person(ADA), 0.0, 1)]
        assert _standings(rows) == [("ada", 1, 1, 1)]

    def test_playing_one_game_many_times_moves_no_column_past_one(self) -> None:
        """§3's sixth test. The columns count games, never matches."""
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, day)
            for day in range(1, 21)
        ]
        assert _standings(rows) == [("ada", 1, 1, 0)]

    def test_a_game_against_another_person_is_counted_by_no_column(self) -> None:
        """The three counts are §5's, read over a person against a computer player. A
        row with two people seated still puts both on the page."""
        rows = [_row("alpha", _person(ADA), _person(BOB), 1.0, 1)]
        assert sorted(_standings(rows)) == [("ada", 0, 0, 0), ("bob", 0, 0, 0)]


class TestTheOrder:
    def test_tops_then_wins_then_games_tried(self) -> None:
        rows = [
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 1),
            _row("beta", _person(ADA), _computer(WEAK), 0.0, 2),
            _row("alpha", _computer(TOP), _person(BOB), 0.0, 3),
        ]
        assert _standings(rows) == [("bob", 1, 1, 1), ("ada", 2, 1, 0)]

    def test_a_tie_is_broken_by_the_handle_and_never_left_to_the_store(self) -> None:
        rows = [
            _row("alpha", _person(BOB), _computer(WEAK), 1.0, 1),
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 2),
        ]
        assert [row[0] for row in _standings(rows)] == ["ada", "bob"]

    def test_two_people_who_have_both_left_still_come_out_in_one_order(self) -> None:
        rows = [
            _row("alpha", _person(BOB), _computer(WEAK), 1.0, 1),
            _row("alpha", _person(ADA), _computer(WEAK), 1.0, 2),
        ]
        gone = {ADA: "a former player", BOB: "a former player"}
        assert [row.user for row in milestones.standings_from(rows, TABLES, gone)] \
            == [ADA, BOB]


class TestWhatIsUnclaimed:
    def test_an_empty_site_has_every_game_open_on_both_counts(self) -> None:
        open_now = milestones.unclaimed_from([], TABLES, GAMES)
        assert open_now.unplayed == GAMES
        assert open_now.unbeaten == GAMES
        assert open_now.offered == 2

    def test_a_finished_game_takes_one_off_the_unplayed_list_alone(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(WEAK), 0.0, 1)]
        open_now = milestones.unclaimed_from(rows, TABLES, GAMES)
        assert open_now.unplayed == ("beta",)
        assert open_now.unbeaten == GAMES

    def test_beating_the_top_takes_it_off_both(self) -> None:
        rows = [_row("alpha", _person(ADA), _computer(TOP), 1.0, 1)]
        open_now = milestones.unclaimed_from(rows, TABLES, GAMES)
        assert open_now.unplayed == ("beta",) and open_now.unbeaten == ("beta",)

    def test_a_game_nobody_can_start_is_on_neither_list(self) -> None:
        """What can be started is the caller's answer, because it is the store's."""
        open_now = milestones.unclaimed_from([], TABLES, ("alpha",))
        assert open_now.unplayed == ("alpha",) and open_now.offered == 1

    def test_a_sandbox_game_claims_nothing(self) -> None:
        rows = [
            replace(
                _row("alpha", _person(ADA), _computer(TOP), 1.0, 1),
                mode=match_rows.SANDBOX,
            )
        ]
        open_now = milestones.unclaimed_from(rows, TABLES, GAMES)
        assert open_now.unplayed == GAMES and open_now.unbeaten == GAMES
