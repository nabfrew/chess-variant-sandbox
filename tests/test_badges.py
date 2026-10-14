"""The play badges, `docs/plan-meta-game.md` §2.9, as box E3 built them.

E3's box and its *done when* are in `docs/archive/plan-meta-game.md`; a bare § below is
the live plan's.

The reader is pure, so most of this builds rows as plain values the way
`tests/test_milestones.py` does. The games are called alpha, beta and the rest, and no
piece or ruleset is involved. What the store has to prove is the other half -- that a
finish writes the badges it completed inside its own transaction, that a second finish
writes no duplicate, and that a badge an older match completed arrives dated by that
older match -- and those tests need a real Postgres and skip with a reason without one,
each on the session's migrated store (`tests/site_store.py`), emptied first.

Nothing here works out a result, a legal move or a rating. Every figure a badge quotes is
read off a row somebody else wrote.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import secrets
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import site_store

from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.site import awards, badges, drafts, rating, votes
from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.spec import rules_used, schema, spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.human import RESIGNED
from chess_variant_sandbox.web.opponents import LADDER

ADA = 1
BOB = 2

# The four named rungs, by the agent string that is each one's identity. Written out here
# rather than read off `badges` so that the table and the reader are checked against the
# picker rather than against each other.
EASY = "minimax:nodes=400"
MEDIUM = "minimax:nodes=1600"
HARD = "minimax:nodes=6400"
HARDEST = "minimax:nodes=200000"
UNNAMED = "greedy"

# A library game's seeds, which a measurement's basis names and a quoted score counts.
# Written out rather than read off `corpus`, so the quotation is checked against what a
# basis holds rather than against the constant it was written from.
FOUR = (0, 1000, 2000, 3000)

_ids = itertools.count()


def _day(day: int) -> datetime:
    return datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=day)


def _row(
    game: str,
    agent: str,
    result: float,
    day: int,
    *,
    user: int = ADA,
    seat: int = 0,
    reason: str = "checkmate",
    mode: str = match_rows.LEADERBOARD,
    source: str | None = None,
    left: bool = False,
    top: bool = False,
    mine: float | None = None,
    theirs: float | None = None,
    deviation: float | None = None,
    rematch_of: str | None = None,
) -> match_rows.Row:
    """One counted match. `result` is this person's score, not seat 0's."""
    me = rating.person(user)
    them = rating.computer(agent)
    seat0, seat1 = (me, them) if seat == 0 else (them, me)
    return match_rows.Row(
        id=f"m{next(_ids):04d}",
        game=game,
        mode=mode,
        seat0=seat0,
        seat1=seat1,
        seed=0,
        plies=20,
        result=result if seat == 0 else 1.0 - result,
        reason=reason,
        left=left,
        top=top,
        top_players=(agent,) if top else (),
        before=(mine, theirs) if seat == 0 else (theirs, mine),
        source=source,
        started=_day(day),
        moved=None,
        finished=_day(day),
        deviations=(deviation, None) if seat == 0 else (None, deviation),
        rematch_of=rematch_of,
    )


def _ladder() -> rating.Table:
    """A table with the four named rungs on it, `HARDEST` alone at the top."""
    return {
        rating.computer(agent): rating.Player(rating.computer(agent), value, 100.0, 0.0)
        for agent, value in (
            (EASY, 800.0), (MEDIUM, 1100.0), (HARD, 1400.0), (HARDEST, 1700.0)
        )
    }


def _cell(**bins: int) -> tuple[int, ...]:
    """A grid cell with every axis in bin 0 except the ones named."""
    return tuple(bins.get(axis.name, 0) for axis in grid.AXES)


GAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
TABLES: dict[str, rating.Table] = {game: _ladder() for game in GAMES}
CELLS = {game: _cell() for game in GAMES}


def _due(
    rows: list[match_rows.Row],
    user: int = ADA,
    tables: dict[str, rating.Table] | None = None,
    cells: dict[str, tuple[int, ...]] | None = None,
    held: tuple[awards.Award, ...] = (),
) -> tuple[awards.Award, ...]:
    return badges.due_from(
        user,
        rows,
        TABLES if tables is None else tables,
        CELLS if cells is None else cells,
        held,
    )


def _kinds(found: tuple[awards.Award, ...]) -> set[str]:
    return {award.kind for award in found}


def _steps(found: tuple[awards.Award, ...], kind: str) -> list[int | None]:
    return [award.tier for award in found if award.kind == kind]


# ------------------------------------------------------------------------- the table


class TestTheTable:
    """One table holds each kind's thresholds, its sentence and its rule sentence."""

    def test_the_four_named_rungs_are_the_pickers_own(self) -> None:
        """A rung the picker renames would silently stop being earnable, so this is what
        notices. The agent strings are never written down in `site/badges.py`."""
        found = {name: agent for name, _gloss, agent in badges._rungs()}
        assert found == {
            "easy": EASY, "medium": MEDIUM, "hard": HARD, "hardest": HARDEST
        }
        assert [agent for _name, _gloss, agent in badges._rungs()] == [
            EASY, MEDIUM, HARD, HARDEST
        ], "the series are weakest first, which is what fills the weaker ones"
        source = Path(badges.__file__ or "").read_text(encoding="utf-8")
        for agent in (EASY, MEDIUM, HARD, HARDEST):
            assert agent not in source, f"{agent} is written down rather than read"

    def test_the_rungs_are_read_off_the_ladder_and_not_copied(self) -> None:
        """The picker's own labels, budget and all."""
        labels = dict(LADDER)
        for _name, gloss, agent in badges._rungs():
            assert gloss and gloss in labels[agent]

    def test_every_badge_names_what_it_reads(self) -> None:
        """§3 test 7. The rule sentence reads the same for a holder and for everybody
        else, which is what the plain list of badges shows."""
        for kind, badge in badges.BADGES.items():
            assert badge.kind == kind
            assert badge.name and badge.name[0].isupper()
            assert badge.rule.endswith(".")
            assert badge.sentence.endswith(".")

    def test_the_sweeps_words_are_its_limit(self) -> None:
        """W6 (S16): *Every which way* and `/badges` say the sweep as *resigned after
        thirty days without a move*, from `quota.IDLE_SAID`, so the words and the limit the
        sweep reads cannot drift apart."""
        from chess_variant_sandbox.site import quota

        assert timedelta(days=30) == quota.IDLE_LIMIT
        assert quota.IDLE_SAID == "thirty days without a move"
        assert quota.IDLE_SAID in badges.BADGES[badges.EVERY_WHICH_WAY].rule

    def test_a_series_sentence_fills_in_its_step(self) -> None:
        award = awards.Award(kind=badges.BEAT[0], tier=5, holder=ADA)
        assert badges.sentence_for(award) == "A win over easy on 5 different games."

    def test_the_first_step_is_one_game_rather_than_one_games(self) -> None:
        award = awards.Award(kind=badges.BEAT[0], tier=1, holder=ADA)
        assert badges.sentence_for(award) == "A win over easy on 1 different game."

    def test_a_badge_with_no_series_says_one_thing(self) -> None:
        award = awards.Award(kind=badges.GIANT_KILLER, holder=ADA, game="alpha")
        assert badges.sentence_for(award) == badges.BADGES[badges.GIANT_KILLER].sentence

    def test_a_kind_from_a_later_box_is_not_a_crash(self) -> None:
        """E7 to E11 add kinds here; a page drawing a row is not where that is found out."""
        unknown = "voted-with-experience"
        assert unknown not in badges.BADGES
        assert badges.sentence_for(awards.Award(kind=unknown, tier=50)) == unknown


# -------------------------------------------------------------- U2, the four series


class TestBeatTheLadder:
    def test_a_win_over_hard_fills_the_three_weaker_series_and_not_the_top_one(
        self,
    ) -> None:
        """E3's first *done when*, and the reading §2.9 approved: nobody replays easy on
        25 games after beating hardest there."""
        found = _due([_row("alpha", HARD, 1.0, 1)])
        assert _kinds(found) == {badges.BEAT[0], badges.BEAT[1], badges.BEAT[2]}
        assert all(award.tier == 1 for award in found)

    def test_a_win_over_hardest_fills_all_four(self) -> None:
        found = _due([_row("alpha", HARDEST, 1.0, 1)])
        assert _kinds(found) == set(badges.BEAT)

    def test_the_fifth_game_earns_the_second_step_once(self) -> None:
        """E3's *done when*: the fifth such game earns step 5 once."""
        rows = [_row(game, HARD, 1.0, day) for day, game in enumerate(GAMES[:5])]
        found = _due(rows)
        assert _steps(found, badges.BEAT[2]) == [1, 5]
        assert _steps(found, badges.BEAT[3]) == []

    def test_a_series_counts_games_and_never_matches(self) -> None:
        """§3 test 6 as the plan reshaped it. Five wins on one game are one game."""
        rows = [_row("alpha", HARD, 1.0, day) for day in range(5)]
        assert _steps(_due(rows), badges.BEAT[2]) == [1]

    def test_a_step_is_dated_by_the_win_that_reached_it(self) -> None:
        rows = [_row(game, EASY, 1.0, day) for day, game in enumerate(GAMES[:5])]
        found = [a for a in _due(rows) if a.kind == badges.BEAT[0] and a.tier == 5]
        assert [award.earned for award in found] == [_day(4)]

    def test_a_weaker_series_takes_the_stronger_wins_date_on_that_game(self) -> None:
        """Beating hard on day 1 is beating easy on day 1, whenever easy was played."""
        rows = [_row("alpha", HARD, 1.0, 1), _row("alpha", EASY, 1.0, 9)]
        found = [a for a in _due(rows) if a.kind == badges.BEAT[0]]
        assert [award.earned for award in found] == [_day(1)]

    def test_the_evidence_names_the_player_by_its_budget_and_the_games(self) -> None:
        """A computer player is its agent string, node budget included."""
        rows = [_row(game, HARD, 1.0, day) for day, game in enumerate(GAMES[:5])]
        found = [a for a in _due(rows) if a.kind == badges.BEAT[2] and a.tier == 5][0]
        assert found.evidence["player"] == HARD
        assert found.evidence["games"] == list(GAMES[:5])
        assert len(found.evidence["matches"]) == 5
        assert found.rule == {"steps": list(badges.GAME_STEPS)}

    def test_a_loss_and_a_draw_earn_nothing(self) -> None:
        assert _due([_row("alpha", HARD, 0.0, 1), _row("beta", HARD, 0.5, 2)]) == ()

    def test_a_win_over_a_player_the_picker_does_not_name_earns_nothing(self) -> None:
        """`greedy` is on the ladder and is not one of the four named rungs."""
        assert _kinds(_due([_row("alpha", UNNAMED, 1.0, 1)])) == set()

    def test_the_series_reads_a_row_from_either_seat(self) -> None:
        found = _due([_row("alpha", HARD, 1.0, 1, seat=1)])
        assert _kinds(found) == {badges.BEAT[0], badges.BEAT[1], badges.BEAT[2]}

    def test_another_persons_win_is_not_this_persons(self) -> None:
        assert _due([_row("alpha", HARD, 1.0, 1, user=BOB)]) == ()


# -------------------------------------------------------- what no badge is allowed to see


class TestWhatIsRead:
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"mode": match_rows.SANDBOX}, id="a sandbox win"),
            pytest.param({"source": match_rows.LOCAL_APP}, id="a local-application row"),
            pytest.param({"reason": RESIGNED, "left": True}, id="a resignation"),
            pytest.param({"left": True}, id="a game the sweep closed"),
        ],
    )
    def test_a_row_outside_the_mode_that_counts_earns_nothing(
        self, change: dict[str, Any]
    ) -> None:
        """E3's *done when*: a sandbox win, a local-app row and a resignation earn
        nothing. A win is the strongest case, since it is what would otherwise fire."""
        rows = [_row(game, HARDEST, 1.0, day, top=True, **change)
                for day, game in enumerate(GAMES)]
        assert _due(rows) == ()

    def test_the_same_rows_without_the_change_earn_plenty(self) -> None:
        """Keeps the four above from passing on rows that were inert anyway."""
        rows = [_row(game, HARDEST, 1.0, day, top=True) for day, game in enumerate(GAMES)]
        assert len(_due(rows)) > 4


# ------------------------------------------------------------------ N2, the giant-killer


class TestGiantKiller:
    def test_an_established_person_beating_somebody_far_above_earns_it(self) -> None:
        found = _due([_row("alpha", HARD, 1.0, 1, mine=1200.0, theirs=1500.0,
                           deviation=90.0)])
        assert badges.GIANT_KILLER in _kinds(found)

    def test_a_gap_one_point_short_earns_nothing(self) -> None:
        """E3's *done when*: a gap of 299 is refused."""
        found = _due([_row("alpha", HARD, 1.0, 1, mine=1200.0, theirs=1499.0,
                           deviation=90.0)])
        assert badges.GIANT_KILLER not in _kinds(found)

    def test_a_rating_not_yet_established_earns_nothing(self) -> None:
        """E3's *done when*. `rating.ESTABLISHED_DEVIATION` is the same line the ranked
        table draws, so nobody holds this before they are on it."""
        found = _due([_row("alpha", HARD, 1.0, 1, mine=1200.0, theirs=1500.0,
                           deviation=rating.ESTABLISHED_DEVIATION)])
        assert badges.GIANT_KILLER not in _kinds(found)

    def test_a_row_nothing_rated_earns_nothing(self) -> None:
        """No ratings going in means no answer, rather than a gap of zero."""
        assert badges.GIANT_KILLER not in _kinds(_due([_row("alpha", HARD, 1.0, 1)]))

    def test_it_sits_on_the_game_and_quotes_both_ratings(self) -> None:
        found = [a for a in _due([_row("alpha", HARD, 1.0, 1, mine=1200.0,
                                       theirs=1500.0, deviation=90.0)])
                 if a.kind == badges.GIANT_KILLER][0]
        assert found.game == "alpha"
        assert found.holder == ADA
        assert found.evidence["gap"] == 300.0
        assert found.evidence["opponent"] == HARD
        assert found.rule == {"gap": badges.GIANT_GAP,
                              "established": rating.ESTABLISHED_DEVIATION}

    def test_a_second_giant_on_the_same_game_is_the_same_badge(self) -> None:
        rows = [_row("alpha", HARD, 1.0, day, mine=1200.0, theirs=1500.0, deviation=90.0)
                for day in (1, 2)]
        found = [a for a in _due(rows) if a.kind == badges.GIANT_KILLER]
        assert [award.earned for award in found] == [_day(1)]


# --------------------------------------------------------------------- N3, held the top


class TestHeldTheTop:
    def test_a_draw_against_the_top_earns_it(self) -> None:
        found = _due([_row("alpha", HARDEST, 0.5, 1, top=True)])
        assert badges.HELD_THE_TOP in _kinds(found)

    def test_a_draw_against_somebody_who_was_not_the_top_earns_nothing(self) -> None:
        """The `top` bit is the match's own, written when it started, so a table that
        moved afterwards neither grants this nor takes it away."""
        assert badges.HELD_THE_TOP not in _kinds(_due([_row("alpha", HARDEST, 0.5, 1)]))

    def test_a_win_over_the_top_is_a_different_badge(self) -> None:
        assert badges.HELD_THE_TOP not in _kinds(_due([_row("alpha", HARDEST, 1.0, 1,
                                                            top=True)]))

    def test_it_sits_on_the_game(self) -> None:
        found = [a for a in _due([_row("alpha", HARDEST, 0.5, 1, top=True)])
                 if a.kind == badges.HELD_THE_TOP][0]
        assert (found.game, found.holder, found.tier) == ("alpha", ADA, None)


# -------------------------------------------------------------------- N4, the grand tour


class TestTheGrandTour:
    """One kind is one bin of one grid axis, which is M6's reading called rather than
    restated, so the badge and the milestone cannot disagree about what a kind is."""

    def _two_kinds(self) -> tuple[dict[str, rating.Table], dict[str, tuple[int, ...]]]:
        tables = {"alpha": _ladder(), "beta": _ladder()}
        return tables, {"alpha": _cell(), "beta": _cell(board=1)}

    def test_covering_every_kind_with_wins_over_the_top_earns_it(self) -> None:
        tables, cells = self._two_kinds()
        rows = [_row("alpha", HARDEST, 1.0, 1, top=True),
                _row("beta", HARDEST, 1.0, 2, top=True)]
        found = [a for a in _due(rows, tables=tables, cells=cells)
                 if a.kind == badges.GRAND_TOUR]
        assert len(found) == 1
        assert found[0].earned == _day(2), "dated by the win that completed it"
        assert found[0].game is None, "it is earned across the library"

    def test_a_kind_with_no_win_over_the_top_leaves_it_open(self) -> None:
        tables, cells = self._two_kinds()
        rows = [_row("alpha", HARDEST, 1.0, 1, top=True),
                _row("beta", HARDEST, 0.5, 2, top=True)]
        assert badges.GRAND_TOUR not in _kinds(_due(rows, tables=tables, cells=cells))

    def test_a_win_over_somebody_other_than_the_top_does_not_cover_a_kind(self) -> None:
        tables, cells = self._two_kinds()
        rows = [_row("alpha", HARDEST, 1.0, 1, top=True), _row("beta", HARD, 1.0, 2)]
        assert badges.GRAND_TOUR not in _kinds(_due(rows, tables=tables, cells=cells))

    def test_the_bins_it_was_earned_under_are_stored(self) -> None:
        """§2.9: the kinds are the ones the library held that day, so a game added later
        cannot make an old row read as incomplete."""
        tables, cells = self._two_kinds()
        rows = [_row("alpha", HARDEST, 1.0, 1, top=True),
                _row("beta", HARDEST, 1.0, 2, top=True)]
        found = [a for a in _due(rows, tables=tables, cells=cells)
                 if a.kind == badges.GRAND_TOUR][0]
        assert len(found.rule["bins"]) == len(grid.AXES) + 1, "one axis has two bins here"
        assert "board: small" in found.rule["bins"]
        assert sorted(found.evidence["games"]) == ["alpha", "beta"]

    def test_a_library_with_nothing_rated_in_it_earns_nothing(self) -> None:
        rows = [_row("alpha", HARDEST, 1.0, 1, top=True)]
        assert badges.GRAND_TOUR not in _kinds(_due(rows, tables={}, cells={}))


# ------------------------------------------------------------------ N7, every which way


class TestEveryWhichWay:
    def _endings(self, count: int) -> list[match_rows.Row]:
        reasons = ("checkmate", "reached", "extinction", "repetition", "stalemate")
        return [_row(GAMES[index], HARD, 1.0, index, reason=reasons[index])
                for index in range(count)]

    def test_four_different_endings_earn_it(self) -> None:
        found = [a for a in _due(self._endings(badges.ENDINGS))
                 if a.kind == badges.EVERY_WHICH_WAY]
        assert len(found) == 1
        assert found[0].earned == _day(badges.ENDINGS - 1)
        assert found[0].rule == {"endings": badges.ENDINGS}

    def test_three_endings_leave_it_open(self) -> None:
        assert badges.EVERY_WHICH_WAY not in _kinds(_due(self._endings(3)))

    def test_the_same_ending_four_times_leaves_it_open(self) -> None:
        rows = [_row(GAMES[index], HARD, 1.0, index, reason="checkmate")
                for index in range(4)]
        assert badges.EVERY_WHICH_WAY not in _kinds(_due(rows))

    def test_a_fifth_ending_does_not_earn_a_second_row(self) -> None:
        found = [a for a in _due(self._endings(5)) if a.kind == badges.EVERY_WHICH_WAY]
        assert [award.earned for award in found] == [_day(badges.ENDINGS - 1)]

    def test_losses_do_not_count_toward_it(self) -> None:
        rows = [_row(GAMES[index], HARD, 0.0, index, reason=reason)
                for index, reason in enumerate(("checkmate", "reached", "extinction",
                                                "repetition"))]
        assert badges.EVERY_WHICH_WAY not in _kinds(_due(rows))


# ----------------------------------------------------------------- N8, the long way round


class TestTheLongWayRound:
    @pytest.mark.parametrize("reason", badges.MOVE_LIMIT)
    def test_a_game_that_reached_the_limit_earns_it(self, reason: str) -> None:
        """Both endings the engine writes at the limit: the undecided one and the one a
        scoring rule settled when it got there."""
        found = [a for a in _due([_row("alpha", HARD, 0.5, 1, reason=reason)])
                 if a.kind == badges.LONG_WAY_ROUND]
        assert len(found) == 1
        assert found[0].game is None and found[0].evidence["game"] == "alpha"

    def test_a_game_that_ended_sooner_earns_nothing(self) -> None:
        assert badges.LONG_WAY_ROUND not in _kinds(_due([_row("alpha", HARD, 1.0, 1)]))

    def test_stalling_twice_earns_one_row_dated_by_the_first(self) -> None:
        """§3 test 6: once per person, so the second long game is worth nothing."""
        rows = [_row(game, HARD, 0.5, day, reason="ply_limit")
                for day, game in enumerate(GAMES[:2])]
        found = [a for a in _due(rows) if a.kind == badges.LONG_WAY_ROUND]
        assert [award.earned for award in found] == [_day(0)]


# --------------------------------------------------------------- N9, and another thing


class TestAndAnotherThing:
    """E9. The link is `matches.rematch_of`, which only the board's control writes, so
    two matches in a row are not a rematch and the badge never guesses at one."""

    def _pair(self, second: float = 1.0) -> list[match_rows.Row]:
        """A loss on alpha, and the game started again from it."""
        lost = _row("alpha", HARD, 0.0, 1)
        return [lost, _row("alpha", HARD, second, 2, rematch_of=lost.id)]

    def test_a_win_in_the_game_started_again_from_a_loss_earns_it(self) -> None:
        found = [a for a in _due(self._pair()) if a.kind == badges.AND_ANOTHER_THING]
        assert len(found) == 1
        assert found[0].game is None and found[0].tier is None

    def test_it_is_dated_by_the_win_and_names_both_matches(self) -> None:
        rows = self._pair()
        found = [a for a in _due(rows) if a.kind == badges.AND_ANOTHER_THING][0]
        assert found.earned == _day(2)
        assert found.evidence == {
            "game": "alpha", "lost": rows[0].id, "won": rows[1].id,
        }

    def test_a_win_after_a_loss_with_no_link_earns_nothing(self) -> None:
        rows = [_row("alpha", HARD, 0.0, 1), _row("alpha", HARD, 1.0, 2)]
        assert badges.AND_ANOTHER_THING not in _kinds(_due(rows))

    def test_a_rematch_of_a_win_earns_nothing(self) -> None:
        first = _row("alpha", HARD, 1.0, 1)
        rows = [first, _row("alpha", HARD, 1.0, 2, rematch_of=first.id)]
        assert badges.AND_ANOTHER_THING not in _kinds(_due(rows))

    def test_a_drawn_rematch_earns_nothing(self) -> None:
        assert badges.AND_ANOTHER_THING not in _kinds(_due(self._pair(second=0.5)))

    def test_a_rematch_named_on_another_game_earns_nothing(self) -> None:
        """`matches._followed` refuses the link, and the reader refuses the rows too:
        it is handed rows rather than a store, so it says what it reads."""
        lost = _row("alpha", HARD, 0.0, 1)
        rows = [lost, _row("beta", HARD, 1.0, 2, rematch_of=lost.id)]
        assert badges.AND_ANOTHER_THING not in _kinds(_due(rows))

    def test_the_loss_it_names_has_to_be_this_persons(self) -> None:
        """Ada's win names Bob's loss. A row from the other side of the site is not in
        Ada's own rows at all, so nothing links up."""
        lost = _row("alpha", HARD, 0.0, 1, user=BOB)
        rows = [lost, _row("alpha", HARD, 1.0, 2, rematch_of=lost.id)]
        assert badges.AND_ANOTHER_THING not in _kinds(_due(rows))

    def test_a_resignation_is_not_the_loss(self) -> None:
        """`_counted` drops a resignation and the sweep's forfeit for every badge here,
        so leaving a game and winning the next one earns nothing."""
        lost = _row("alpha", HARD, 0.0, 1, reason=RESIGNED)
        rows = [lost, _row("alpha", HARD, 1.0, 2, rematch_of=lost.id)]
        assert badges.AND_ANOTHER_THING not in _kinds(_due(rows))

    def test_a_sandbox_pair_earns_nothing(self) -> None:
        lost = _row("alpha", HARD, 0.0, 1, mode=match_rows.SANDBOX)
        rows = [lost, _row("alpha", HARD, 1.0, 2, rematch_of=lost.id,
                           mode=match_rows.SANDBOX)]
        assert badges.AND_ANOTHER_THING not in _kinds(_due(rows))

    def test_losing_and_winning_twice_earns_one_row_dated_by_the_first(self) -> None:
        """§3 test 6: once per person, so the second comeback is worth nothing."""
        first = _row("alpha", HARD, 0.0, 1)
        second = _row("beta", HARD, 0.0, 3)
        rows = [
            first, _row("alpha", HARD, 1.0, 2, rematch_of=first.id),
            second, _row("beta", HARD, 1.0, 4, rematch_of=second.id),
        ]
        found = [a for a in _due(rows) if a.kind == badges.AND_ANOTHER_THING]
        assert [award.earned for award in found] == [_day(2)]


# ----------------------------------------------------------------- what is already held


class TestWhatIsAlreadyHeld:
    def test_a_badge_already_held_is_not_offered_again(self) -> None:
        rows = [_row("alpha", HARD, 1.0, 1)]
        first = _due(rows)
        assert _due(rows, held=first) == ()

    def test_the_next_step_is_still_offered(self) -> None:
        rows = [_row(game, HARD, 1.0, day) for day, game in enumerate(GAMES[:5])]
        held = _due([_row("alpha", HARD, 1.0, 0)])
        assert _steps(_due(rows, held=held), badges.BEAT[2]) == [5]


# ------------------------------------------------------- the designer badges, E6's own

ADA_GAMES = ("one", "two", "three", "four")
BOB_GAMES = ("five", "six")
CAT = 3
CAT_GAMES = ("seven",)

BLOCKS = rules_used.available()
"""The building blocks themselves, so that a test hands the reader names it could
really see. Which names they are is `tests/test_rules_used.py`'s business."""


def _design(
    game: str,
    owner: int | None,
    *,
    parent: str | None = None,
    day: int = 0,
    origin: str = drafts.DESIGNED,
    screen_passed: bool = True,
    revoked: bool = False,
    blocks: tuple[str, ...] = (),
) -> badges.Design:
    """One `games` row. The defaults are a design that passed the screen."""
    return badges.Design(
        game=game,
        owner=owner,
        parent=parent,
        origin=origin,
        screen_passed=screen_passed,
        created=_day(day),
        revoked=revoked,
        blocks=frozenset(blocks),
    )


def _library(game: str, owner: int = 99, day: int = 0) -> badges.Design:
    """A game the loader wrote. It has an owner -- the maintainer's account -- which is
    why `origin` and never a null owner is what tells the two apart."""
    return _design(game, owner, day=day, origin="corpus", blocks=BLOCKS[:8])


def _designer_due(
    user: int,
    games: list[badges.Design],
    rows: list[match_rows.Row] | None = None,
    held: tuple[awards.Award, ...] = (),
) -> tuple[awards.Award, ...]:
    return badges.due_from(user, rows or [], TABLES, CELLS, held, games)


def _for_kind(found: tuple[awards.Award, ...], kind: str) -> list[awards.Award]:
    return [award for award in found if award.kind == kind]


class TestTheStepsAreAShareOfTheBuildingBlocks:
    """U5's thresholds are the user's answer of 2026-09-17, read as a share rather than a
    count, because `rules_used.available()` grows with each schema version."""

    def test_a_quarter_a_half_and_three_quarters_of_thirty_are_eight_fifteen_and_23(
        self,
    ) -> None:
        """§2.9's arithmetic, against the 30 building blocks the schema held that day."""
        assert [badges._needed(share, 30) for share in badges.TOOLBOX_STEPS] == [8, 15, 23]

    def test_a_step_never_lands_on_half_a_rule(self) -> None:
        assert badges._needed(25, 31) == 8 and badges._needed(50, 31) == 16


class TestTheWholeToolbox:
    """U5. A person's passing designs, and the blocks they draw on between them."""

    def _first_step(self) -> int:
        return badges._needed(badges.TOOLBOX_STEPS[0], len(BLOCKS))

    def test_the_first_step_lands_at_the_step_s_eighth_block(self) -> None:
        """E6's *done when*, over today's table: the eighth distinct building block."""
        step = self._first_step()
        half = step // 2
        games = [
            _design(ADA_GAMES[0], ADA, day=1, blocks=BLOCKS[:half]),
            _design(ADA_GAMES[1], ADA, day=2, blocks=BLOCKS[half:step]),
        ]
        found = _for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX)
        assert [(award.tier, award.earned) for award in found] == [(25, _day(2))]

    def test_one_block_short_earns_nothing(self) -> None:
        games = [_design(ADA_GAMES[0], ADA, blocks=BLOCKS[: self._first_step() - 1])]
        assert _for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX) == []

    def test_a_block_counts_once_however_many_games_use_it(self) -> None:
        """§3 test 6: five near-copies are worth one game."""
        step = self._first_step()
        games = [
            _design(name, ADA, day=day, blocks=BLOCKS[:step])
            for day, name in enumerate(ADA_GAMES)
        ]
        found = _for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX)
        assert [award.tier for award in found] == [25]

    def test_the_half_step_earns_both_it_and_the_quarter_at_once(self) -> None:
        needed = badges._needed(50, len(BLOCKS))
        games = [_design(ADA_GAMES[0], ADA, day=4, blocks=BLOCKS[:needed])]
        found = _for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX)
        assert [award.tier for award in found] == [25, 50]
        assert {award.earned for award in found} == {_day(4)}

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"screen_passed": False}, id="a draft that failed the screen"),
            pytest.param({"revoked": True}, id="a game taken off the site"),
            pytest.param({"origin": "corpus"}, id="a game that came with the site"),
            pytest.param({"owner": BOB}, id="somebody else's design"),
        ],
    )
    def test_a_game_outside_the_screen_counts_for_nothing(
        self, change: dict[str, Any]
    ) -> None:
        """E6's *done when* for the unlisted draft, and the three neighbouring readings.
        A library game is told by `origin`, never by its owner."""
        row = dict(change)
        games = [
            _design(
                ADA_GAMES[0],
                row.pop("owner", ADA),
                blocks=BLOCKS[: self._first_step()],
                **row,
            )
        ]
        assert _for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX) == []

    def test_the_same_game_inside_the_screen_earns_it(self) -> None:
        """Keeps the four above from passing on a row that was inert anyway."""
        games = [_design(ADA_GAMES[0], ADA, blocks=BLOCKS[: self._first_step()])]
        assert len(_for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX)) == 1

    def test_the_row_stores_n_of_n_the_schema_and_the_names(self) -> None:
        """§2.9: a rules badge stores *n of N*, the schema version and the names counted.
        N grows with the schema, so the row keeps the count it was earned under."""
        step = self._first_step()
        games = [_design(ADA_GAMES[0], ADA, blocks=BLOCKS[:step])]
        [found] = _for_kind(_designer_due(ADA, games), badges.WHOLE_TOOLBOX)
        assert found.holder == ADA and found.game is None
        assert found.evidence["used"] == step
        assert found.evidence["available"] == len(BLOCKS)
        assert found.evidence["schema"] == schema.SCHEMA_VERSION
        assert found.evidence["names"] == sorted(BLOCKS[:step])
        assert found.evidence["games"] == [ADA_GAMES[0]]
        assert found.rule == {
            "shares": list(badges.TOOLBOX_STEPS),
            "share": 25,
            "needed": step,
            "available": len(BLOCKS),
        }

    def test_the_sentence_fills_in_the_share(self) -> None:
        award = awards.Award(kind=badges.WHOLE_TOOLBOX, tier=25, holder=ADA)
        assert badges.sentence_for(award).endswith("use 25% of the building blocks.")


class TestBothHats:
    """N6. One person who has beaten some game's top and owns a passing design."""

    def _win(self, day: int) -> match_rows.Row:
        return _row("alpha", HARDEST, 1.0, day, top=True)

    def test_the_design_coming_second_dates_it(self) -> None:
        """E6's *done when*: it lands on whichever of the two events comes second."""
        found = _for_kind(
            _designer_due(ADA, [_design(ADA_GAMES[0], ADA, day=5)], [self._win(2)]),
            badges.BOTH_HATS,
        )
        assert [award.earned for award in found] == [_day(5)]

    def test_the_win_coming_second_dates_it(self) -> None:
        found = _for_kind(
            _designer_due(ADA, [_design(ADA_GAMES[0], ADA, day=2)], [self._win(5)]),
            badges.BOTH_HATS,
        )
        assert [award.earned for award in found] == [_day(5)]
        assert found[0].game is None and found[0].holder == ADA

    def test_it_names_the_match_and_the_design(self) -> None:
        rows = [self._win(2)]
        found = _for_kind(
            _designer_due(ADA, [_design(ADA_GAMES[0], ADA, day=5)], rows),
            badges.BOTH_HATS,
        )
        assert found[0].evidence["match"] == rows[0].id
        assert found[0].evidence["game"] == "alpha"
        assert found[0].evidence["design"] == ADA_GAMES[0]

    def test_a_design_with_no_win_over_a_top_earns_nothing(self) -> None:
        rows = [_row("alpha", HARDEST, 1.0, 2)]
        found = _designer_due(ADA, [_design(ADA_GAMES[0], ADA)], rows)
        assert _for_kind(found, badges.BOTH_HATS) == []

    def test_a_draw_with_the_top_is_not_beating_it(self) -> None:
        rows = [_row("alpha", HARDEST, 0.5, 2, top=True)]
        found = _designer_due(ADA, [_design(ADA_GAMES[0], ADA)], rows)
        assert _for_kind(found, badges.BOTH_HATS) == []

    def test_a_win_with_no_design_of_the_persons_own_earns_nothing(self) -> None:
        games = [
            _design(ADA_GAMES[0], ADA, screen_passed=False),
            _library("library-one"),
            _design(BOB_GAMES[0], BOB),
        ]
        assert _for_kind(_designer_due(ADA, games, [self._win(2)]), badges.BOTH_HATS) == []


class TestFamilyTree:
    """N12's three levels, read as narrowly as §2.9 reads them."""

    def _chain(self) -> list[badges.Design]:
        """A's game, B's built from it, and C's built from B's."""
        return [
            _design(ADA_GAMES[0], ADA, day=1),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2),
            _design(CAT_GAMES[0], CAT, parent=BOB_GAMES[0], day=3),
        ]

    def test_building_on_another_persons_design_earns_both_lower_levels(self) -> None:
        """E6's first *done when*."""
        games = self._chain()[:2]
        builder = _designer_due(BOB, games)
        assert [award.kind for award in builder] == [badges.ANCESTOR]
        assert builder[0].game is None and builder[0].earned == _day(2)
        [owner] = _for_kind(_designer_due(ADA, games), badges.DESCENDANT)
        assert (owner.holder, owner.game, owner.earned) == (ADA, ADA_GAMES[0], _day(2))

    def test_each_row_names_the_games_and_no_handle(self) -> None:
        games = self._chain()
        [ancestor] = _for_kind(_designer_due(CAT, games), badges.ANCESTOR)
        assert ancestor.evidence == {"game": CAT_GAMES[0], "parent": BOB_GAMES[0]}
        [descendant] = _for_kind(_designer_due(BOB, games), badges.DESCENDANT)
        assert descendant.evidence == {"game": BOB_GAMES[0], "child": CAT_GAMES[0]}
        [tree] = _for_kind(_designer_due(ADA, games), badges.FAMILY_TREE)
        assert tree.evidence == {
            "game": ADA_GAMES[0],
            "child": BOB_GAMES[0],
            "grandchild": CAT_GAMES[0],
        }

    def test_a_third_person_completes_the_top_level(self) -> None:
        """E6's *done when*: C building on B's game gives C *Ancestor*, B's game
        *Descendant* and A's game *Family tree*."""
        games = self._chain()
        assert {award.kind for award in _designer_due(CAT, games)} == {badges.ANCESTOR}
        assert {award.kind for award in _designer_due(BOB, games)} == {
            badges.ANCESTOR,
            badges.DESCENDANT,
        }
        found = _for_kind(_designer_due(ADA, games), badges.FAMILY_TREE)
        assert [(award.game, award.earned) for award in found] == [(ADA_GAMES[0], _day(3))]

    def test_the_first_owner_building_on_the_second_earns_no_top_level(self) -> None:
        """E6's *done when*: the top level takes three people, so A building on B's game
        gives A's game no *Family tree*."""
        games = [
            _design(ADA_GAMES[0], ADA, day=1),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2),
            _design(ADA_GAMES[1], ADA, parent=BOB_GAMES[0], day=3),
        ]
        found = {award.kind for award in _designer_due(ADA, games)}
        assert badges.FAMILY_TREE not in found
        assert found == {badges.ANCESTOR, badges.DESCENDANT}

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"screen_passed": False}, id="a draft that failed the screen"),
            pytest.param({"revoked": True}, id="a game taken off the site"),
        ],
    )
    def test_a_new_game_outside_the_screen_earns_nothing_either_way(
        self, change: dict[str, Any]
    ) -> None:
        """E6's *done when* for the failed save. A killed game counts at no level."""
        games = [
            _design(ADA_GAMES[0], ADA, day=1),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2, **change),
        ]
        assert _designer_due(BOB, games) == ()
        assert _for_kind(_designer_due(ADA, games), badges.DESCENDANT) == []

    def test_a_game_built_from_a_library_game_earns_nothing(self) -> None:
        """E6's *done when*, and §2.9's narrow reading: the library is not a person, so
        there is no *Descendant* to pair the lower level with."""
        games = [_library("library-one"), _design(BOB_GAMES[0], BOB, parent="library-one")]
        assert _designer_due(BOB, games) == ()
        assert _designer_due(99, games) == ()

    def test_building_on_your_own_game_earns_nothing(self) -> None:
        """E6's *done when*: every link crosses between two people."""
        games = [
            _design(ADA_GAMES[0], ADA, day=1),
            _design(ADA_GAMES[1], ADA, parent=ADA_GAMES[0], day=2),
        ]
        assert _designer_due(ADA, games) == ()

    def test_a_second_builder_writes_no_second_descendant(self) -> None:
        """E6's *done when*, and §3 test 6: once per game."""
        games = [
            _design(ADA_GAMES[0], ADA, day=1),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2),
            _design(CAT_GAMES[0], CAT, parent=ADA_GAMES[0], day=3),
        ]
        found = _for_kind(_designer_due(ADA, games), badges.DESCENDANT)
        assert [award.earned for award in found] == [_day(2)]

    def test_ancestor_is_once_per_person_however_many_games_they_build_on(self) -> None:
        games = [
            _design(ADA_GAMES[0], ADA, day=1),
            _design(ADA_GAMES[1], ADA, day=1),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2),
            _design(BOB_GAMES[1], BOB, parent=ADA_GAMES[1], day=3),
        ]
        found = _for_kind(_designer_due(BOB, games), badges.ANCESTOR)
        assert [award.earned for award in found] == [_day(2)]

    def test_the_game_built_from_need_not_be_listed(self) -> None:
        """§2.9: an unlisted game opens by its link, and building on one is still
        building on it."""
        games = [
            _design(ADA_GAMES[0], ADA, day=1, screen_passed=False),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2),
        ]
        assert {award.kind for award in _designer_due(BOB, games)} == {badges.ANCESTOR}
        assert {award.kind for award in _designer_due(ADA, games)} == {badges.DESCENDANT}

    def test_a_killed_game_is_built_on_by_nobody(self) -> None:
        games = [
            _design(ADA_GAMES[0], ADA, day=1, revoked=True),
            _design(BOB_GAMES[0], BOB, parent=ADA_GAMES[0], day=2),
        ]
        assert _designer_due(BOB, games) == ()
        assert _designer_due(ADA, games) == ()

    def test_a_game_with_no_parent_earns_nothing(self) -> None:
        assert _designer_due(ADA, [_design(ADA_GAMES[0], ADA)]) == ()

    def test_a_badge_already_held_is_not_offered_again(self) -> None:
        games = self._chain()
        first = _designer_due(ADA, games)
        assert _designer_due(ADA, games, held=first) == ()


# ----------------------------------------------------------- U4, well read, E5's own

SHELF: dict[str, tuple[str, ...]] = {
    "alpha": BLOCKS[0:2],
    "beta": BLOCKS[2:4],
    "gamma": BLOCKS[4:8],
}
"""Three rated games whose blocks do not overlap, eight between them.

E5's *done when* is written over exactly this shelf: two games of two blocks and one of
four, with eight in play.
"""

UNRATED = "zeta"
"""A game with no table, which is on the site but not among the blocks in play."""


def _shelf(
    blocks: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[badges.Design], dict[str, rating.Table]]:
    """A library of rated games with their blocks, and the tables that rate them."""
    shelf = SHELF if blocks is None else blocks
    games = [
        _design(game, 99, origin="corpus", blocks=used) for game, used in shelf.items()
    ]
    return games, {game: _ladder() for game in shelf}


def _read_due(
    rows: list[match_rows.Row],
    games: list[badges.Design] | None = None,
    tables: dict[str, rating.Table] | None = None,
    held: tuple[awards.Award, ...] = (),
) -> list[awards.Award]:
    """What U4 is due to ADA over this shelf."""
    shelf, rated = _shelf()
    found = badges.due_from(
        ADA,
        rows,
        rated if tables is None else tables,
        CELLS,
        held,
        shelf if games is None else games,
    )
    return _for_kind(found, badges.WELL_READ)


def _played(game: str, day: int, **change: Any) -> match_rows.Row:
    """A finished game on this shelf. A loss, so that no other badge fires."""
    return _row(game, EASY, 0.0, day, **change)


class TestWellRead:
    """U4. The building blocks the games a person has finished draw on between them,
    against the blocks the rated games use.
    """

    def test_half_the_blocks_in_play_earns_the_first_step(self) -> None:
        """E5's *done when*: two games of two blocks each, four of the eight in play."""
        rows = [_played("alpha", 1), _played("beta", 2)]
        assert [(a.tier, a.earned) for a in _read_due(rows)] == [(50, _day(2))]

    def test_the_third_game_earns_the_two_upper_steps_together(self) -> None:
        """E5's *done when*: the four blocks of the third game take it to eight of eight,
        which clears 75 and 100 at once, both dated by that game."""
        rows = [_played("alpha", 1), _played("beta", 2), _played("gamma", 3)]
        found = _read_due(rows)
        assert [award.tier for award in found] == [50, 75, 100]
        assert [award.earned for award in found[1:]] == [_day(3), _day(3)]

    def test_one_game_short_of_a_step_earns_nothing(self) -> None:
        assert _read_due([_played("alpha", 1)]) == []

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"mode": match_rows.SANDBOX}, id="a sandbox game"),
            pytest.param({"source": match_rows.LOCAL_APP}, id="a local-application row"),
            pytest.param({"reason": RESIGNED, "left": True}, id="a resignation"),
            pytest.param({"left": True}, id="a game the sweep closed"),
            pytest.param({"user": BOB}, id="somebody else's game"),
        ],
    )
    def test_a_game_outside_the_rows_that_count_adds_no_blocks(
        self, change: dict[str, Any]
    ) -> None:
        """E5's *done when* for the sandbox game, and the four neighbouring readings. The
        two games that do count still earn the first step, so this is the third game
        adding nothing rather than the reader falling over."""
        rows = [
            _played("alpha", 1),
            _played("beta", 2),
            _played("gamma", 3, **change),
        ]
        assert [award.tier for award in _read_due(rows)] == [50]

    def test_the_same_game_played_twice_is_one_game(self) -> None:
        rows = [_played("alpha", 1), _played("alpha", 2), _played("beta", 3)]
        found = _read_due(rows)
        assert [(a.tier, a.earned) for a in found] == [(50, _day(3))]
        assert found[0].evidence["games"] == ["alpha", "beta"]

    def test_a_loss_counts_as_a_game_read(self) -> None:
        """*Finished*, not *won*. A person who lost every game has still read the rules."""
        rows = [_played(game, day) for day, game in enumerate(SHELF)]
        assert [award.tier for award in _read_due(rows)] == [50, 75, 100]

    def test_a_block_no_rated_game_uses_is_outside_the_count(self) -> None:
        """The denominator is what the rated games use, so a game with no table adds
        neither to what is in play nor to what this person has read."""
        shelf, rated = _shelf()
        shelf.append(_design(UNRATED, 99, origin="corpus", blocks=BLOCKS[8:20]))
        rows = [_played("alpha", 1), _played(UNRATED, 2)]
        found = _read_due(rows, games=shelf, tables=rated)
        assert found == [], "twelve blocks nobody can be rated on are worth none"

    def test_a_library_with_no_rated_game_earns_nothing(self) -> None:
        shelf, _rated = _shelf()
        assert _read_due([_played("alpha", 1)], games=shelf, tables={}) == []

    def test_the_row_stores_n_of_n_the_schema_and_the_names(self) -> None:
        """§2.9: a rules badge stores *n of N*, the schema version and the names counted,
        and a badge earned by play names the matches that completed it."""
        rows = [_played("alpha", 1), _played("beta", 2)]
        [found] = _read_due(rows)
        assert found.holder == ADA and found.game is None
        assert found.evidence["used"] == 4
        assert found.evidence["available"] == 8
        assert found.evidence["schema"] == schema.SCHEMA_VERSION
        assert found.evidence["names"] == sorted(BLOCKS[0:4])
        assert found.evidence["games"] == ["alpha", "beta"]
        assert found.evidence["matches"] == [row.id for row in rows]
        assert found.rule == {
            "shares": list(badges.READING_STEPS),
            "share": 50,
            "needed": 4,
            "available": 8,
        }

    def test_a_row_earned_at_eight_of_eight_still_reads_eight_of_eight(self) -> None:
        """E5's *done when*. A game added to the library raises N, and a badge already
        held is not re-offered, so the kept row keeps the count it was earned under."""
        rows = [_played(game, day) for day, game in enumerate(SHELF)]
        held = tuple(_read_due(rows))
        assert [award.evidence["used"] for award in held] == [4, 8, 8]

        wider = dict(SHELF, delta=BLOCKS[8:12])
        shelf, rated = _shelf(wider)
        assert _read_due(rows, games=shelf, tables=rated, held=held) == []
        kept = held[-1].evidence
        assert (kept["used"], kept["available"], held[-1].tier) == (8, 8, 100)

    def test_a_wider_library_moves_the_step_for_a_badge_not_yet_earned(self) -> None:
        """The other side of that reading: N is read when the step is earned, so eight
        blocks of twelve is the first step and no longer the third."""
        shelf, rated = _shelf(dict(SHELF, delta=BLOCKS[8:12]))
        rows = [_played(game, day) for day, game in enumerate(SHELF)]
        found = _read_due(rows, games=shelf, tables=rated)
        assert [(award.tier, award.evidence["available"]) for award in found] == [
            (50, 12)
        ]

    def test_the_sentence_fills_in_the_share(self) -> None:
        award = awards.Award(kind=badges.WELL_READ, tier=75, holder=ADA)
        assert badges.sentence_for(award).endswith("use 75% of the building blocks.")


# ------------------------------------------------- N13, speaks from experience, E7's own


def _rating(game: str, day: int, voter: int = ADA) -> badges.Rated:
    """One row of `votes` as N13 reads it. There is no star to pass."""
    return badges.Rated(voter=voter, game=game, rated=_day(day))


def _rated_due(
    rows: list[match_rows.Row],
    ratings: list[badges.Rated],
    user: int = ADA,
    held: tuple[awards.Award, ...] = (),
) -> list[awards.Award]:
    """What N13 is due to this person over these rows and these ratings."""
    found = badges.due_from(user, rows, TABLES, CELLS, held, (), ratings)
    return _for_kind(found, badges.SPEAKS_FROM_EXPERIENCE)


def _three_on(game: str, first: int = 1, user: int = ADA) -> list[match_rows.Row]:
    """Three finished games on one game, on three consecutive days. Losses, so that
    nothing else fires and the badge under test is the only one in the answer."""
    return [_row(game, EASY, 0.0, first + each, user=user) for each in range(3)]


class TestSpeaksFromExperience:
    """N13. A rating counts after three finished games on that game, and the value of
    the rating is not something this reader can see.
    """

    def test_a_rating_after_three_finished_games_earns_the_first_step(self) -> None:
        """One input, five facts about the row it earns (Q754 merged five tests of it)."""
        rows = _three_on("alpha")
        found = _rated_due(rows, [_rating("alpha", 9)])
        assert [(award.tier, award.earned) for award in found] == [(1, _day(9))]
        # It sits on the person and not on a game, under the thresholds it was earned under.
        assert found[0].holder == ADA
        assert found[0].game is None
        assert found[0].rule == {"steps": [1, 5, 10, 25], "played": 3}
        # It holds the games and the three matches under each, and no star value.
        assert set(found[0].evidence) == {"games", "ratings"}
        assert "stars" not in json.dumps(found[0].evidence)
        assert found[0].evidence["games"] == ["alpha"]
        assert found[0].evidence["ratings"] == [
            {
                "game": "alpha",
                "rated": _day(9).isoformat(),
                "matches": [row.id for row in rows],
            }
        ]

    def test_a_rating_after_two_earns_nothing(self) -> None:
        rows = _three_on("alpha")[:2]
        assert _rated_due(rows, [_rating("alpha", 9)]) == []

    def test_a_rating_given_before_the_third_game_earns_nothing(self) -> None:
        """The third game finishing does not reach back and make an older rating count.
        `votes.updated` is when the person last said it, and they said it too early."""
        rows = [_row("alpha", EASY, 0.0, day) for day in (1, 2, 9)]
        assert _rated_due(rows, [_rating("alpha", 5)]) == []

    def test_rating_again_after_the_third_game_earns_the_step(self) -> None:
        """The same rows, with the rating given again afterwards. A re-vote moves
        `votes.updated`, which is the whole of what changed."""
        rows = [_row("alpha", EASY, 0.0, day) for day in (1, 2, 9)]
        found = _rated_due(rows, [_rating("alpha", 10)])
        assert [(award.tier, award.earned) for award in found] == [(1, _day(10))]

    def test_a_resigned_game_does_not_count_toward_the_three(self) -> None:
        rows = _three_on("alpha")
        rows[1] = _row("alpha", EASY, 0.0, 2, reason=RESIGNED, left=True)
        assert _rated_due(rows, [_rating("alpha", 9)]) == []

    def test_a_swept_game_does_not_count_toward_the_three(self) -> None:
        """The idle sweep resigns on the player's behalf, so its row is one somebody
        left. A badge nobody could earn by playing is not earnable by walking away."""
        rows = _three_on("alpha")
        rows[2] = _row("alpha", EASY, 1.0, 3, reason=RESIGNED, left=True)
        assert _rated_due(rows, [_rating("alpha", 9)]) == []

    def test_a_sandbox_game_does_not_count_toward_the_three(self) -> None:
        rows = _three_on("alpha")
        rows[0] = _row("alpha", EASY, 0.0, 1, mode=match_rows.SANDBOX)
        assert _rated_due(rows, [_rating("alpha", 9)]) == []

    def test_three_games_on_another_game_do_not_count(self) -> None:
        """The three have to be on the game that was rated."""
        assert _rated_due(_three_on("beta"), [_rating("alpha", 9)]) == []

    def test_one_star_and_five_stars_earn_the_same_row(self) -> None:
        """There is no way to say which was given, which is the point. §2.9 refuses the
        badge the star value so that it pulls no mean in either direction."""
        assert not hasattr(badges.Rated, "stars")
        assert "stars" not in badges.Rated.__dataclass_fields__

    def test_only_the_first_three_matches_travel_with_the_rating(self) -> None:
        rows = _three_on("alpha") + [_row("alpha", EASY, 0.0, 5)]
        found = _rated_due(rows, [_rating("alpha", 9)])
        assert found[0].evidence["ratings"][0]["matches"] == [row.id for row in rows[:3]]

    def test_rating_the_same_game_again_adds_no_step(self) -> None:
        """`votes` holds one row per person per game, so the series counts games. Two
        rows on one game are impossible in the store and earn one step here either way."""
        rows = _three_on("alpha")
        found = _rated_due(rows, [_rating("alpha", 9), _rating("alpha", 10)])
        assert [award.tier for award in found] == [1]
        assert found[0].earned == _day(9), "the first qualifying rating dates it"

    def test_five_rated_games_earn_the_second_step(self) -> None:
        rows: list[match_rows.Row] = []
        ratings: list[badges.Rated] = []
        for index, game in enumerate(GAMES[:5]):
            rows += _three_on(game, first=1 + index * 3)
            ratings.append(_rating(game, 20 + index))
        found = _rated_due(rows, ratings)
        assert [award.tier for award in found] == [1, 5]
        assert found[1].earned == _day(24), "the fifth rating completed the step"

    def test_the_steps_are_the_ones_the_ladder_badges_use(self) -> None:
        """§2.9 copies U2's ladder so that a reader learns one ladder, not two."""
        assert badges.RATING_STEPS == badges.GAME_STEPS == (1, 5, 10, 25)

    def test_another_persons_rating_earns_nothing(self) -> None:
        assert _rated_due(_three_on("alpha"), [_rating("alpha", 9, voter=BOB)]) == []

    def test_what_is_already_held_is_not_offered_again(self) -> None:
        held = awards.Award(kind=badges.SPEAKS_FROM_EXPERIENCE, tier=1, holder=ADA)
        assert _rated_due(_three_on("alpha"), [_rating("alpha", 9)], held=(held,)) == []

    def test_the_rule_sentence_says_play_comes_first_and_the_stars_are_not_read(
        self,
    ) -> None:
        """§3 test 7 over the one clause a reader has to know before they rate."""
        rule = badges.BADGES[badges.SPEAKS_FROM_EXPERIENCE].rule
        assert "3 rated games" in rule
        assert "not read" in rule


# --------------------------------------- N1 and U1, the badges on the game, E8's own

CROWD = (11, 12, 13, 14, 15)
"""Five people with nothing else to their names, which is `CROWD_PEOPLE` exactly."""

OWNER = 20
"""Whoever designed the game under test, where one did."""


def _finished(user: int, day: int, game: str = "alpha") -> match_rows.Row:
    """One person's finished leaderboard game on this game. A loss against `EASY`."""
    return _row(game, EASY, 0.0, day, user=user)


def _star(voter: int, stars: int, day: int) -> badges.Given:
    """One rating as N1 and U1 read it."""
    return badges.Given(voter=voter, stars=stars, rated=_day(day))


def _players(first: int = 1, who: tuple[int, ...] = CROWD) -> list[match_rows.Row]:
    return [_finished(user, first + index) for index, user in enumerate(who)]


def _liked(
    first: int = 1, stars: int = 5, who: tuple[int, ...] = CROWD
) -> list[badges.Given]:
    return [_star(user, stars, first + index) for index, user in enumerate(who)]


def _score(
    headline: float = 0.50,
    low: float | None = 0.44,
    high: float | None = 0.58,
    weights: str = "v9",
) -> awards.ScoreEvidence:
    """What `awards.score_evidence` answers for a game whose measurement reads current.

    A stale, missing or never-measured game answers `None` there, which is what the
    tests below hand the reader instead of a number.
    """
    return awards.ScoreEvidence(
        game="alpha",
        headline=headline,
        ci_low=low,
        ci_high=high,
        weights=weights,
        spec_hash="alpha",
        basis={"weights_version": weights, "ruleset_hash": "alpha", "seeds": list(FOUR)},
    )


def _game_due(
    rows: list[match_rows.Row] | None = None,
    given: list[badges.Given] | None = None,
    design: badges.Design | None = None,
    score: awards.ScoreEvidence | None = None,
    held: tuple[awards.Award, ...] = (),
) -> tuple[awards.Award, ...]:
    return badges.game_due_from(
        "alpha",
        design,
        _players() if rows is None else rows,
        _liked() if given is None else given,
        score,
        held,
    )


class TestCrowdPleaser:
    """N1. Five different people with a finished game on it, and five ratings at four
    stars or better. Distinct people and never matches.
    """

    def test_the_fifth_player_completes_it_when_the_ratings_are_there(self) -> None:
        found = _game_due(rows=_players(first=10), given=_liked(first=1))
        assert _kinds(found) == {badges.CROWD_PLEASER}
        assert found[0].earned == _day(14), "the fifth player came second"

    def test_the_fifth_voter_completes_it_when_the_players_are_there(self) -> None:
        found = _game_due(rows=_players(first=1), given=_liked(first=10))
        assert found[0].earned == _day(14), "the fifth rating came second"

    def test_four_players_earn_nothing(self) -> None:
        assert _game_due(rows=_players(who=CROWD[:4])) == ()

    def test_four_voters_earn_nothing(self) -> None:
        assert _game_due(given=_liked(who=CROWD[:4])) == ()

    def test_a_mean_under_four_stars_earns_nothing(self) -> None:
        given = _liked(stars=4)
        given[0] = _star(CROWD[0], 3, 1)
        assert _game_due(given=given) == ()

    def test_a_mean_of_exactly_four_stars_earns_it(self) -> None:
        assert _kinds(_game_due(given=_liked(stars=4))) == {badges.CROWD_PLEASER}

    def test_one_person_playing_it_many_times_counts_once(self) -> None:
        rows = [_finished(CROWD[0], day) for day in range(1, 20)]
        assert _game_due(rows=rows) == (), "matches are not people"

    def test_a_computer_seat_is_not_a_person(self) -> None:
        """Every row here is somebody against `EASY`, so the count would be ten if the
        other seat were counted."""
        found = _game_due(rows=_players(who=CROWD[:4]), given=_liked())
        assert found == ()

    def test_a_sandbox_game_does_not_count_toward_the_five(self) -> None:
        rows = _players()
        rows[4] = _row("alpha", EASY, 0.0, 5, user=CROWD[4], mode=match_rows.SANDBOX)
        assert _game_due(rows=rows) == ()

    def test_a_resigned_game_does_not_count_toward_the_five(self) -> None:
        rows = _players()
        rows[4] = _row("alpha", EASY, 0.0, 5, user=CROWD[4], reason=RESIGNED, left=True)
        assert _game_due(rows=rows) == ()

    def test_the_row_holds_the_mean_the_voters_and_the_players(self) -> None:
        found = _game_due(given=_liked(stars=4))
        assert found[0].evidence == {
            "game": "alpha", "people": 5, "voters": 5, "mean": 4.0
        }

    def test_a_later_rating_that_drops_the_mean_removes_nothing(self) -> None:
        """§2.9's *kept once earned*, over a badge nothing had yet written. The walk is
        over the ratings in date order, so the five that completed it still do."""
        given = _liked(stars=4) + [_star(16, 1, 9)]
        found = _game_due(given=given)
        assert found[0].evidence["mean"] == 4.0
        assert found[0].earned == _day(5), "the fifth rating, not the sixth"

    def test_the_badge_sits_on_a_library_game_with_no_holder(self) -> None:
        found = _game_due(design=_library("alpha"))
        assert found[0].game == "alpha"
        assert found[0].holder is None, "the maintainer's account is nobody's profile"

    def test_a_designed_games_badge_sits_on_its_owner(self) -> None:
        found = _game_due(design=_design("alpha", OWNER))
        assert found[0].holder == OWNER
        assert found[0].game == "alpha"

    def test_a_designed_games_owner_counts_for_neither_number(self) -> None:
        """E8's *done when*: the owner voting five stars and playing it counts for
        neither. Four other people and the owner is four people, not five."""
        who = (OWNER,) + CROWD[:4]
        found = _game_due(
            rows=_players(who=who), given=_liked(who=who), design=_design("alpha", OWNER)
        )
        assert found == ()

    def test_the_owner_of_a_library_game_is_left_out_of_nothing(self) -> None:
        """The library's owner is the loader's maintainer account rather than a player,
        so a library game leaves nobody out."""
        found = _game_due(design=_library("alpha", owner=CROWD[0]))
        assert found[0].evidence["people"] == 5

    def test_a_killed_game_earns_nothing(self) -> None:
        assert _game_due(design=_design("alpha", OWNER, revoked=True)) == ()

    def test_what_is_already_held_is_not_offered_again(self) -> None:
        held = awards.Award(kind=badges.CROWD_PLEASER, game="alpha")
        assert _game_due(held=(held,)) == ()

    def test_the_row_keeps_the_thresholds_it_was_earned_under(self) -> None:
        assert _game_due()[0].rule == {"people": 5, "stars": 4.0}


class TestProvedTheClankerWrong:
    """U1. What N1 reads, and a current measurement whose headline and whole range sit
    under the mark. The score comes from `awards.score_evidence` and from nowhere else.
    """

    def test_a_low_score_writes_both_badges(self) -> None:
        found = _game_due(score=_score())
        assert _kinds(found) == {badges.CROWD_PLEASER, badges.CLANKER}

    def test_a_stale_measurement_writes_no_clanker(self) -> None:
        """A stale score is not a number, so `score_evidence` answers nothing and there
        is none to get. E8's *done when*: the same events write N1 and no U1."""
        found = _game_due(score=None)
        assert _kinds(found) == {badges.CROWD_PLEASER}

    def test_a_range_reaching_the_mark_writes_no_clanker(self) -> None:
        found = _game_due(score=_score(headline=0.60, low=0.55, high=badges.CLANKER_SCORE))
        assert _kinds(found) == {badges.CROWD_PLEASER}

    def test_a_headline_at_the_mark_writes_no_clanker(self) -> None:
        found = _game_due(score=_score(headline=badges.CLANKER_SCORE, low=0.55, high=0.64))
        assert _kinds(found) == {badges.CROWD_PLEASER}

    def test_a_measurement_with_no_range_writes_no_clanker(self) -> None:
        """A score taken before the interval was written down has a headline and no
        ends, and a badge about a whole range cannot be granted on one."""
        found = _game_due(score=_score(headline=0.40, low=None, high=None))
        assert _kinds(found) == {badges.CROWD_PLEASER}

    def test_a_game_nobody_has_loved_earns_no_clanker_however_low_it_scores(self) -> None:
        assert _game_due(given=_liked(stars=3), score=_score()) == ()

    def test_the_row_holds_the_headline_the_range_and_the_weights(self) -> None:
        found = _for_kind(_game_due(score=_score()), badges.CLANKER)
        assert found[0].evidence["score"] == {
            "headline": 0.50,
            "ci_low": 0.44,
            "ci_high": 0.58,
            "weights": "v9",
            "spec_hash": "alpha",
            "basis": {"weights_version": "v9", "ruleset_hash": "alpha", "seeds": list(FOUR)},
        }

    def test_the_row_holds_what_crowd_pleaser_holds_as_well(self) -> None:
        found = _for_kind(_game_due(score=_score()), badges.CLANKER)
        assert found[0].evidence["people"] == 5
        assert found[0].evidence["voters"] == 5

    def test_both_badges_are_dated_by_the_same_event(self) -> None:
        found = _game_due(score=_score())
        assert len({award.earned for award in found}) == 1

    def test_the_row_keeps_the_mark_it_was_earned_under(self) -> None:
        found = _for_kind(_game_due(score=_score()), badges.CLANKER)
        assert found[0].rule == {"people": 5, "stars": 4.0, "score": 0.65}

    def test_holding_the_clanker_leaves_crowd_pleaser_on_offer(self) -> None:
        held = awards.Award(kind=badges.CLANKER, game="alpha")
        assert _kinds(_game_due(score=_score(), held=(held,))) == {badges.CROWD_PLEASER}


def _quoted(**score: Any) -> str:
    """What a held U1 line quotes, with this score in its evidence and nothing else."""
    evidence = {
        "score": {"headline": 0.41, "ci_low": 0.36, "ci_high": 0.48, "weights": "v10",
                  **score},
    }
    return badges.quoted(awards.Award(kind=badges.CLANKER, game="alpha", evidence=evidence))


class TestAQuotedScoreSaysHowManyRuns:
    """Q339. A score badge quotes the runs its score was the mean of, off the kept basis.

    A library game's score is the mean of four runs and a visitor's is one, and a number
    that does not say which is not a number `CLAUDE.md` lets a reader show. The words are
    the page's `runsPhrase`, so a badge and a card say one measurement alike.
    """

    def test_a_library_score_is_the_mean_of_four_runs(self) -> None:
        assert _quoted(basis={"seeds": list(FOUR)}) == (
            "a score of 0.41 (likely 0.36 to 0.48; the average of four runs,"
            " scoring rules version 10)"
        )

    def test_a_visitors_score_is_one_run(self) -> None:
        assert _quoted(basis={"seeds": [0]}) == (
            "a score of 0.41 (likely 0.36 to 0.48; one run, scoring rules version 10)"
        )

    def test_a_basis_from_before_several_runs_is_one_run(self) -> None:
        """`seed: 0` is what a basis wrote before a game could have more than one run,
        and `measurement.Basis.from_dict` reads it as exactly one."""
        assert "; one run," in _quoted(basis={"seed": 0})

    def test_a_count_past_four_is_spelled_as_the_page_spells_it(self) -> None:
        assert "the average of twelve runs" in _quoted(basis={"seeds": list(range(12))})
        assert "the average of 21 runs" in _quoted(basis={"seeds": list(range(21))})

    def test_a_row_whose_basis_names_no_seeds_quotes_no_score(self) -> None:
        """A reader that cannot say which claim a number is shows no number."""
        for basis in (None, {}, {"seeds": []}, {"seeds": "0"}, {"seeds": [0, "x"]},
                      {"weights_version": "v10", "ruleset_hash": "alpha"}):
            assert _quoted(basis=basis) == "", basis
        assert _quoted() == ""

    def test_the_rest_of_the_row_is_still_quoted_when_the_score_is_not(self) -> None:
        award = awards.Award(kind=badges.CLANKER, game="alpha", evidence={
            "score": {"headline": 0.41, "ci_low": 0.36, "ci_high": 0.48, "weights": "v10"},
            "mean": 4.4, "voters": 9, "people": 12,
        })
        assert badges.quoted(award) == (
            "4.4 stars from 9 people and 12 people who had played it"
        )

    def test_the_count_sits_inside_the_score_clause_of_the_whole_quotation(self) -> None:
        award = awards.Award(kind=badges.CLANKER, game="alpha", evidence={
            "score": {**_score(headline=0.41, low=0.36, high=0.48).as_dict()},
            "mean": 4.4, "voters": 9, "people": 12,
        })
        assert badges.quoted(award) == (
            "a score of 0.41 (likely 0.36 to 0.48; the average of four runs, scoring"
            " rules version 9), 4.4 stars from 9 people and 12 people who had played it"
        )

    def test_the_rule_says_the_runs_are_quoted(self) -> None:
        """§3's seventh test: the rule under the line names what the line quotes."""
        assert "with its run count and scoring rules version" in badges.BADGES[
            badges.CLANKER].rule


# ------------------------------------------------------------------------- the store


DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))


def _short() -> dict[str, Any]:
    """`dull` with its ply limit at two, which ends after one move each.

    The cheapest finish there is, and it ends *at the limit*, so the match a store test
    plays out is one `The long way round` reads.
    """
    spec = json.loads(json.dumps(DULL))
    spec["rules"]["max_plies"] = 2
    return spec


SHORT = _short()
SHORT_HASH = spec_hash(SHORT)

# A picker ladder in the shape `runs/opponents/<hash>.json` has, with `EASY` at the top.
PICKER = {
    "ruleset_hash": SHORT_HASH,
    "standings": [
        {"name": EASY, "rating": 300.0, "games": 24},
        {"name": "greedy", "rating": 0.0, "games": 24},
        {"name": "random", "rating": -300.0, "games": 24},
    ],
}


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

    def game(self, digest: str, spec: dict[str, Any], ladder: Any = None) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps(spec), spec["name"], "corpus"),
            )
        if ladder is not None:
            rating.seed(self.connection, digest, ladder)
        self.connection.commit()
        return digest

    def older_win(self, user: int, when: datetime) -> str:
        """A finished leaderboard win over `EASY`, written straight into the table.

        The match happened before this box landed, which is the case §2.9 describes: the
        badge it completed arrives with the person's next finish, dated by this row.
        """
        match_id = f"older{secrets.token_hex(2)}"
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into matches"
                " (id, game, mode, seed, seat0_user, seat1_agent, moves, result,"
                "  reason, top, started, finished)"
                " values (%s, %s, %s, 0, %s, %s, '[]'::jsonb, 1.0, %s, true, %s, %s)",
                (match_id, SHORT_HASH, match_rows.LEADERBOARD, user, EASY,
                 "checkmate", when, when),
            )
        self.connection.commit()
        return match_id

    def awards(self, user: int) -> list[awards.Award]:
        return awards.held(self.connection, holder=user)

    def on_game(self, game: str = SHORT_HASH) -> list[awards.Award]:
        return awards.held(self.connection, game=game)

    def crowd(self, size: int = badges.CROWD_PEOPLE) -> list[int]:
        """That many people with accounts, so that N1 and U1 have somebody to count."""
        return [self.user(f"p{index}") for index in range(size)]

    def vote(self, user: int, stars: int, game: str = SHORT_HASH) -> None:
        """A rating and the badges it completed, as `/api/vote` does them."""
        votes.cast(self.connection, voter=user, game=game, stars=stars)
        badges.grant_for_vote(self.connection, user, game)
        self.connection.commit()

    def measure(
        self,
        state: str,
        headline: float | None,
        ci_low: float | None = None,
        ci_high: float | None = None,
        weights: str = "v9",
        game: str = SHORT_HASH,
        seeds: tuple[int, ...] | None = FOUR,
    ) -> None:
        """One `measurements` row, in whatever state the test needs it.

        Written by hand rather than through the loader, because what these tests are
        about is which states `awards.score_evidence` will quote from. `seeds` is what
        the basis names, and `None` writes a basis that names none.
        """
        basis: dict[str, Any] = {"weights_version": weights, "ruleset_hash": game}
        if seeds is not None:
            basis["seeds"] = list(seeds)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into measurements (hash, directory, state, headline, basis,"
                " ci_low, ci_high) values (%s, %s, %s, %s, %s::jsonb, %s, %s)"
                " on conflict (hash) do update set state = excluded.state,"
                " headline = excluded.headline, basis = excluded.basis,"
                " ci_low = excluded.ci_low, ci_high = excluded.ci_high",
                (game, f"{game}-0", state, headline, json.dumps(basis), ci_low, ci_high),
            )
        self.connection.commit()

    def design(
        self, digest: str, owner: int, parent: str = "", passed: bool = True
    ) -> drafts.Kept:
        """A saved design, written the way the save route writes it, and committed.

        The spec is the same short game every time. What a spec draws on is the pure
        reader's business and `tests/test_rules_used.py`'s; what the store has to prove
        is that the row and its badges land together.
        """
        found = drafts.keep(
            self.connection, digest, SHORT, owner=owner, parent=parent, passed=passed
        )
        self.connection.commit()
        return found

    def save(self, digest: str, owner: int, parent: str = "", passed: bool = True) -> None:
        """A save and the badges it completed, as the route does them, then the commit."""
        self.design(digest, owner, parent=parent, passed=passed)
        badges.grant_for_save(self.connection, owner, digest)
        self.connection.commit()


@contextlib.contextmanager
def _scratch() -> Iterator[Scratch]:
    url, name = site_store.emptied("no database to finish a match in")
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
    """A store holding one rated game."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, PICKER)
        yield scratch


@pytest.fixture
def player(store: Scratch) -> int:
    return store.user("ada")


def _play_out(store: Scratch, user: int) -> str:
    """Start a leaderboard match and play its two plies, which finishes it."""
    match = match_rows.start(
        store.connection, SHORT_HASH, match_rows.LEADERBOARD, ["human", EASY], user
    )
    snapshot = match_rows.session(store.connection, match.id).snapshot()
    match_rows.play(store.connection, match.id, snapshot["moves"][0]["move"])
    match_rows.play_agent(store.connection, match.id)
    store.connection.commit()
    return match.id


class TestTheFinishWritesThem:
    """E3's *done when* over a real store. The match played is `dull` with its ply limit
    at two, so it reaches the limit against the top of its own table -- which is `The
    long way round` and `Held the top`, and nothing had to be hand-written to get them.
    """

    def test_finishing_a_match_writes_the_badges_it_completed(
        self, store: Scratch, player: int
    ) -> None:
        _play_out(store, player)
        found = {award.kind for award in store.awards(player)}
        assert badges.LONG_WAY_ROUND in found
        assert badges.HELD_THE_TOP in found

    def test_the_rows_land_in_the_finishs_own_transaction(
        self, store: Scratch, player: int
    ) -> None:
        """A finish and the badges it completed land together or neither does, so a
        connection that never saw the match reads them after the commit and not before."""
        match = match_rows.start(
            store.connection, SHORT_HASH, match_rows.LEADERBOARD, ["human", EASY], player
        )
        snapshot = match_rows.session(store.connection, match.id).snapshot()
        match_rows.play(store.connection, match.id, snapshot["moves"][0]["move"])
        match_rows.play_agent(store.connection, match.id)

        other = store.open()
        try:
            with other.cursor() as cursor:
                cursor.execute("select count(*) from awards")
                assert int(cursor.fetchone()[0]) == 0, "written before the commit"
            store.connection.commit()
            with other.cursor() as cursor:
                cursor.execute("select count(*) from awards")
                assert int(cursor.fetchone()[0]) > 0
        finally:
            other.close()

    def test_a_second_finish_writes_no_duplicate(
        self, store: Scratch, player: int
    ) -> None:
        _play_out(store, player)
        first = len(store.awards(player))
        _play_out(store, player)
        kinds = [award.kind for award in store.awards(player)]
        assert len(kinds) == first
        assert kinds.count(badges.LONG_WAY_ROUND) == 1

    def test_a_badge_an_older_match_completed_keeps_the_older_matchs_date(
        self, store: Scratch, player: int
    ) -> None:
        """§2.9: a badge completed before its box landed arrives with the person's next
        event, dated by the row that completed it."""
        when = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
        store.older_win(player, when)
        _play_out(store, player)
        earned = {award.kind: award.earned for award in store.awards(player)}
        assert earned[badges.BEAT[0]] == when
        assert earned[badges.LONG_WAY_ROUND] > when

    def test_a_sandbox_finish_writes_nothing(self, store: Scratch, player: int) -> None:
        """The guard the hook opens with, so a sandbox finish costs no queries either."""
        match = match_rows.start(
            store.connection, SHORT_HASH, match_rows.SANDBOX, ["human", EASY], player
        )
        snapshot = match_rows.session(store.connection, match.id).snapshot()
        match_rows.play(store.connection, match.id, snapshot["moves"][0]["move"])
        match_rows.play_agent(store.connection, match.id)
        store.connection.commit()
        assert store.awards(player) == []

    def test_well_read_counts_the_blocks_of_the_game_that_was_played(
        self, store: Scratch, player: int
    ) -> None:
        """U4 over a real store. The one rated game here is the whole of what is in play,
        so finishing it reads every block there is to read."""
        _play_out(store, player)
        found = [a for a in store.awards(player) if a.kind == badges.WELL_READ]
        assert [award.tier for award in found] == list(badges.READING_STEPS)
        blocks = rules_used.used(SHORT)
        assert found[0].evidence["names"] == sorted(blocks)
        assert found[0].evidence["available"] == len(blocks)
        assert found[0].evidence["games"] == [SHORT_HASH]

    def test_a_sandbox_finish_reads_no_blocks(
        self, store: Scratch, player: int
    ) -> None:
        """E5's *done when*: a sandbox game adds nothing, over the store as well as over
        hand-built rows."""
        match = match_rows.start(
            store.connection, SHORT_HASH, match_rows.SANDBOX, ["human", EASY], player
        )
        snapshot = match_rows.session(store.connection, match.id).snapshot()
        match_rows.play(store.connection, match.id, snapshot["moves"][0]["move"])
        match_rows.play_agent(store.connection, match.id)
        store.connection.commit()
        assert badges.due_for(store.connection, player) == ()

    def test_the_row_carries_the_deviation_each_seat_went_in_with(
        self, store: Scratch, player: int
    ) -> None:
        """What *Giant-killer* reads. Written by `rating.finish` beside the ratings, so
        the hook runs after it."""
        match_id = _play_out(store, player)
        found = match_rows.row(store.connection, match_id)
        seat = found.seat_of(rating.person(player))
        assert found.deviation_before(seat) is not None
        assert found.rating_before(seat) is not None


class TestTheCellsComeFromTheStore:
    def test_a_rated_games_cell_is_read_off_its_spec(self, store: Scratch) -> None:
        """No cell column, so nothing can fall out of step with the spec beside it."""
        found = badges.cells(store.connection, [SHORT_HASH])
        assert found == {SHORT_HASH: grid.descriptor(SHORT)}

    def test_no_games_is_no_query(self, store: Scratch) -> None:
        assert badges.cells(store.connection, []) == {}


class TestTheSaveWritesThem:
    """E6's *done when* over a real store. A save is `drafts.keep` and then
    `badges.grant_for_save`, which is what `server._save_design` does inside one
    transaction.
    """

    def _three(self, store: Scratch) -> tuple[int, int, int]:
        return store.user("ada"), store.user("bob"), store.user("cat")

    def test_building_on_another_persons_design_writes_both_lower_levels(
        self, store: Scratch
    ) -> None:
        ada, bob, _cat = self._three(store)
        store.save("design-a", ada)
        store.save("design-b", bob, parent="design-a")
        assert [award.kind for award in store.awards(bob)] == [badges.ANCESTOR]
        [owner] = store.awards(ada)
        assert (owner.kind, owner.game, owner.holder) == (
            badges.DESCENDANT, "design-a", ada
        )

    def test_a_third_person_writes_the_top_level_for_the_first_owner(
        self, store: Scratch
    ) -> None:
        ada, bob, cat = self._three(store)
        store.save("design-a", ada)
        store.save("design-b", bob, parent="design-a")
        store.save("design-c", cat, parent="design-b")
        assert {award.kind for award in store.awards(cat)} == {badges.ANCESTOR}
        assert {award.kind for award in store.awards(bob)} == {
            badges.ANCESTOR, badges.DESCENDANT
        }
        tree = [a for a in store.awards(ada) if a.kind == badges.FAMILY_TREE]
        assert [award.game for award in tree] == ["design-a"]

    def test_a_second_pass_writes_no_duplicate(self, store: Scratch) -> None:
        ada, bob, _cat = self._three(store)
        store.save("design-a", ada)
        store.save("design-b", bob, parent="design-a")
        first = len(store.awards(ada)) + len(store.awards(bob))
        badges.grant_for_save(store.connection, bob, "design-b")
        store.connection.commit()
        assert len(store.awards(ada)) + len(store.awards(bob)) == first

    def test_the_rows_land_in_the_saves_own_transaction(self, store: Scratch) -> None:
        """A save and the badges it completed land together or neither does."""
        ada, bob, _cat = self._three(store)
        store.save("design-a", ada)
        drafts.keep(
            store.connection, "design-b", SHORT, owner=bob, parent="design-a", passed=True
        )
        badges.grant_for_save(store.connection, bob, "design-b")
        other = store.open()
        try:
            with other.cursor() as cursor:
                cursor.execute("select count(*) from awards")
                assert int(cursor.fetchone()[0]) == 0, "written before the commit"
            store.connection.commit()
            with other.cursor() as cursor:
                cursor.execute("select count(*) from awards")
                assert int(cursor.fetchone()[0]) == 2
        finally:
            other.close()

    def test_a_draft_that_failed_the_screen_writes_nothing(self, store: Scratch) -> None:
        ada, bob, _cat = self._three(store)
        store.save("design-a", ada)
        store.save("design-b", bob, parent="design-a", passed=False)
        assert store.awards(bob) == [] and store.awards(ada) == []

    def test_a_game_built_from_a_library_game_writes_nothing(
        self, store: Scratch
    ) -> None:
        """The loader's games have the maintainer's account as owner, so this is the
        reading that `origin` and not the owner tells them apart."""
        _ada, bob, _cat = self._three(store)
        store.save("design-b", bob, parent=SHORT_HASH)
        assert store.awards(bob) == []

    def test_the_building_blocks_come_from_the_spec_the_store_holds(
        self, store: Scratch
    ) -> None:
        """U5 reads `games.spec` through `rules_used.used` and through nothing else. A
        library row carries its blocks too, which is what U4 counts and U5 ignores."""
        ada, _bob, _cat = self._three(store)
        store.design("design-a", ada)
        found = {design.game: design for design in badges.designs(store.connection)}
        assert found["design-a"].blocks == rules_used.used(SHORT)
        assert found["design-a"].passes and found["design-a"].owner == ada
        assert found[SHORT_HASH].blocks == rules_used.used(SHORT)
        assert not found[SHORT_HASH].passes

    def test_both_hats_lands_at_the_finish_when_the_design_came_first(
        self, store: Scratch, player: int
    ) -> None:
        """N6 can complete on either event, so the match hook asks for it too. The older
        win is over the top of this game's table, which is the half a save cannot give."""
        store.design("design-a", player)
        store.older_win(player, datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        _play_out(store, player)
        assert badges.BOTH_HATS in {award.kind for award in store.awards(player)}

    def test_both_hats_waits_for_the_second_of_its_two_events(
        self, store: Scratch, player: int
    ) -> None:
        store.older_win(player, datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        _play_out(store, player)
        assert badges.BOTH_HATS not in {award.kind for award in store.awards(player)}
        store.save("design-a", player)
        assert badges.BOTH_HATS in {award.kind for award in store.awards(player)}


def _visibility(store: Scratch, digest: str) -> str:
    with store.connection.cursor() as cursor:
        cursor.execute("select visibility from games where hash = %s", (digest,))
        return str(cursor.fetchone()[0])


class TestADraftForkCounts:
    """Q543. *Descendant*'s rule once said the fork *is listed*, and the grant read only
    `screen_passed`. The plan settles it for the grant: `plan-game-pipeline.md` §5 (Q503a)
    has *passes the screen* badges read `screen_passed`, and `plan-meta-game.md` keeps
    D2's meaning, *a draft that passed*. So the rule's words moved, and a fork counts
    listed or not. No game with the corpus game's spec is loaded here, because `twin`
    would refuse to put a copy of it on trial.
    """

    @pytest.fixture
    def empty(self) -> Iterator[Scratch]:
        with _scratch() as scratch:
            yield scratch

    def test_an_unlisted_draft_fork_earns_descendant(self, empty: Scratch) -> None:
        ada, bob = empty.user("ada"), empty.user("bob")
        empty.save("design-a", ada)
        empty.save("design-b", bob, parent="design-a")
        assert _visibility(empty, "design-b") == drafts.UNLISTED
        [owner] = _for_kind(tuple(empty.awards(ada)), badges.DESCENDANT)
        assert (owner.game, owner.evidence) == (
            "design-a", {"game": "design-a", "child": "design-b"}
        )

    def test_a_listed_fork_earns_descendant(self, empty: Scratch) -> None:
        """The fork is on trial before the pass runs, so the grant reads a listed row."""
        ada, bob = empty.user("ada"), empty.user("bob")
        empty.save("design-a", ada)
        empty.design("design-b", bob, parent="design-a")
        site_store.on_trial(empty.connection, "design-b")
        assert _visibility(empty, "design-b") == drafts.PUBLIC
        badges.grant_for_save(empty.connection, bob, "design-b")
        empty.connection.commit()
        [owner] = _for_kind(tuple(empty.awards(ada)), badges.DESCENDANT)
        assert (owner.game, owner.evidence) == (
            "design-a", {"game": "design-a", "child": "design-b"}
        )

    def test_publishing_a_draft_fork_writes_no_second_descendant(
        self, empty: Scratch
    ) -> None:
        ada, bob = empty.user("ada"), empty.user("bob")
        empty.save("design-a", ada)
        empty.save("design-b", bob, parent="design-a")
        site_store.on_trial(empty.connection, "design-b")
        badges.grant_for_save(empty.connection, bob, "design-b")
        empty.connection.commit()
        assert len(_for_kind(tuple(empty.awards(ada)), badges.DESCENDANT)) == 1

    @pytest.mark.parametrize(
        "kind",
        [
            badges.WHOLE_TOOLBOX,
            badges.BOTH_HATS,
            badges.ANCESTOR,
            badges.DESCENDANT,
            badges.FAMILY_TREE,
        ],
    )
    def test_no_passes_the_screen_rule_says_listed(self, kind: str) -> None:
        """Test 7, no hidden rule: the sentence must not ask for what the grant ignores."""
        assert "listed" not in badges.BADGES[kind].rule


class TestTheVoteWritesThem:
    """E7's *done when* over a real store. A vote is `votes.cast` and then
    `badges.grant_for_vote`, which is what `server._vote` does inside one transaction.
    """

    def _play(self, store: Scratch, user: int, times: int) -> None:
        for _each in range(times):
            _play_out(store, user)

    def _steps(self, store: Scratch, user: int) -> list[int | None]:
        return [
            award.tier
            for award in store.awards(user)
            if award.kind == badges.SPEAKS_FROM_EXPERIENCE
        ]

    def test_a_rating_after_three_finished_games_earns_the_first_step(
        self, store: Scratch, player: int
    ) -> None:
        self._play(store, player, 3)
        store.vote(player, 5)
        assert self._steps(store, player) == [1]

    def test_a_rating_after_two_earns_nothing(
        self, store: Scratch, player: int
    ) -> None:
        self._play(store, player, 2)
        store.vote(player, 5)
        assert self._steps(store, player) == []

    def test_a_rating_given_too_early_earns_the_step_when_it_is_given_again(
        self, store: Scratch, player: int
    ) -> None:
        """E7's *done when*, both halves. The third game finishing writes nothing, and
        the same person rating again afterwards writes the step."""
        self._play(store, player, 2)
        store.vote(player, 4)
        self._play(store, player, 1)
        assert self._steps(store, player) == [], "the finish does not reach back"
        store.vote(player, 4)
        assert self._steps(store, player) == [1]

    def test_one_star_earns_the_row_five_stars_would(
        self, store: Scratch, player: int
    ) -> None:
        self._play(store, player, 3)
        store.vote(player, votes.LOWEST)
        found = [
            award
            for award in store.awards(player)
            if award.kind == badges.SPEAKS_FROM_EXPERIENCE
        ]
        assert [award.tier for award in found] == [1]
        assert "stars" not in json.dumps(found[0].evidence)
        assert set(found[0].evidence["ratings"][0]) == {"game", "rated", "matches"}

    def test_rating_the_same_game_again_adds_no_step(
        self, store: Scratch, player: int
    ) -> None:
        self._play(store, player, 3)
        store.vote(player, 5)
        store.vote(player, 4)
        assert self._steps(store, player) == [1]

    def test_withdrawing_the_rating_removes_no_row(
        self, store: Scratch, player: int
    ) -> None:
        """§2.9's *kept once earned*, which is why nobody has a reason to leave a
        careless rating standing."""
        self._play(store, player, 3)
        store.vote(player, 5)
        votes.withdraw(store.connection, player, SHORT_HASH)
        store.connection.commit()
        badges.grant_for_vote(store.connection, player, SHORT_HASH)
        store.connection.commit()
        assert self._steps(store, player) == [1]

    def test_the_row_lands_in_the_votes_own_transaction(
        self, store: Scratch, player: int
    ) -> None:
        self._play(store, player, 3)
        votes.cast(store.connection, voter=player, game=SHORT_HASH, stars=5)
        badges.grant_for_vote(store.connection, player, SHORT_HASH)
        other = store.open()
        try:
            with other.cursor() as cursor:
                cursor.execute("select count(*) from votes")
                assert int(cursor.fetchone()[0]) == 0, "written before the commit"
            store.connection.commit()
            assert self._steps(store, player) == [1]
        finally:
            other.close()

    def test_a_sandbox_game_does_not_count_toward_the_three(
        self, store: Scratch, player: int
    ) -> None:
        self._play(store, player, 2)
        match = match_rows.start(
            store.connection, SHORT_HASH, match_rows.SANDBOX, ["human", EASY], player
        )
        snapshot = match_rows.session(store.connection, match.id).snapshot()
        match_rows.play(store.connection, match.id, snapshot["moves"][0]["move"])
        match_rows.play_agent(store.connection, match.id)
        store.connection.commit()
        store.vote(player, 5)
        assert self._steps(store, player) == []


ON_THE_GAME = (badges.CROWD_PLEASER, badges.CLANKER)
"""The two kinds E8 writes. A game's row also carries N2's and N3's, which sit on the
game but belong to whoever played the match."""


class TestTheBadgesOnTheGame:
    """E8's *done when* over a real store. The game is the library's, so nobody is left
    out of either count and the rows sit on the game with no holder.
    """

    def _crowd(self, store: Scratch, votes_first: bool) -> list[int]:
        """Five people who each rated the game five stars and each finished a game on it.

        `votes_first` decides which of the two is the second event, which is the thing
        E8's first clause is about.
        """
        people = store.crowd()
        if votes_first:
            for user in people:
                store.vote(user, 5)
            for user in people:
                _play_out(store, user)
        else:
            for user in people:
                _play_out(store, user)
            for user in people:
                store.vote(user, 5)
        return people

    def _kinds(self, store: Scratch) -> set[str]:
        """Which of the two badges on the game are there.

        `awards.held(game=...)` answers everything sitting on the game, N3 *Held the top*
        included, and that one is somebody's rather than the game's. These two are told
        apart by their kind.
        """
        return {award.kind for award in store.on_game()} & set(ON_THE_GAME)

    def _rows(self, store: Scratch, kind: str) -> list[awards.Award]:
        return [award for award in store.on_game() if award.kind == kind]

    def test_the_fifth_player_writes_crowd_pleaser(self, store: Scratch) -> None:
        self._crowd(store, votes_first=True)
        assert self._kinds(store) == {badges.CROWD_PLEASER}

    def test_the_fifth_voter_writes_crowd_pleaser(self, store: Scratch) -> None:
        self._crowd(store, votes_first=False)
        assert self._kinds(store) == {badges.CROWD_PLEASER}

    def test_the_row_holds_the_mean_the_voters_and_the_players(
        self, store: Scratch
    ) -> None:
        self._crowd(store, votes_first=False)
        found = self._rows(store, badges.CROWD_PLEASER)[0]
        assert found.holder is None
        assert found.game == SHORT_HASH
        assert found.evidence["people"] == badges.CROWD_PEOPLE
        assert found.evidence["voters"] == badges.CROWD_PEOPLE
        assert found.evidence["mean"] == 5.0

    def test_four_people_write_nothing(self, store: Scratch) -> None:
        for user in store.crowd(4):
            store.vote(user, 5)
            _play_out(store, user)
        assert self._kinds(store) == set()

    def test_a_later_vote_that_drops_the_mean_removes_nothing(
        self, store: Scratch
    ) -> None:
        self._crowd(store, votes_first=False)
        store.vote(store.user("sour"), votes.LOWEST)
        assert self._kinds(store) == {badges.CROWD_PLEASER}
        found = self._rows(store, badges.CROWD_PLEASER)[0]
        assert found.evidence["mean"] == 5.0

    def test_a_stale_measurement_writes_no_clanker_and_a_later_load_writes_it(
        self, store: Scratch
    ) -> None:
        """E8's *done when*, the whole clause. The events write N1 while the score reads
        stale, and the pass `chess-variant-sandbox site load` runs writes U1 once it reads current."""
        store.measure("replay", None)
        self._crowd(store, votes_first=False)
        assert self._kinds(store) == {badges.CROWD_PLEASER}

        store.measure("current", 0.41, 0.36, 0.48)
        badges.grant_for_games(store.connection, [SHORT_HASH])
        store.connection.commit()
        assert self._kinds(store) == {badges.CROWD_PLEASER, badges.CLANKER}

    def test_a_never_measured_game_writes_no_clanker(self, store: Scratch) -> None:
        self._crowd(store, votes_first=False)
        assert badges.CLANKER not in self._kinds(store)

    def test_a_range_reaching_the_mark_writes_no_clanker(self, store: Scratch) -> None:
        store.measure("current", 0.60, 0.55, badges.CLANKER_SCORE)
        self._crowd(store, votes_first=False)
        assert self._kinds(store) == {badges.CROWD_PLEASER}

    def test_a_low_score_writes_the_clanker_with_the_range_and_the_weights(
        self, store: Scratch
    ) -> None:
        store.measure("current", 0.41, 0.36, 0.48, weights="v9")
        self._crowd(store, votes_first=False)
        found = self._rows(store, badges.CLANKER)
        assert found[0].evidence["score"]["headline"] == 0.41
        assert found[0].evidence["score"]["ci_low"] == 0.36
        assert found[0].evidence["score"]["ci_high"] == 0.48
        assert found[0].evidence["score"]["weights"] == "v9"

    def test_the_held_line_quotes_the_runs_the_score_was_the_mean_of(
        self, store: Scratch
    ) -> None:
        """Q339. The row keeps the basis, and the line a page draws counts its seeds."""
        store.measure("current", 0.41, 0.36, 0.48, weights="v9", seeds=FOUR)
        self._crowd(store, votes_first=False)
        found = self._rows(store, badges.CLANKER)
        assert found[0].evidence["score"]["basis"]["seeds"] == list(FOUR)
        assert badges.line(found[0])["quoted"].startswith(
            "a score of 0.41 (likely 0.36 to 0.48; the average of four runs, scoring"
            " rules version 9), "
        )

    def test_a_one_run_score_is_quoted_as_one_run(self, store: Scratch) -> None:
        """A visitor's game is measured once, and its badge says so rather than reading
        like the library's mean of four."""
        store.measure("current", 0.41, 0.36, 0.48, weights="v9", seeds=(0,))
        self._crowd(store, votes_first=False)
        quoted = badges.line(self._rows(store, badges.CLANKER)[0])["quoted"]
        assert "; one run, scoring rules version 9)" in quoted
        assert "average of" not in quoted

    def test_a_kept_basis_naming_no_seeds_quotes_no_score(self, store: Scratch) -> None:
        """The badge is still earned -- the measurement read current on the day -- but a
        line cannot quote a number it cannot say the runs of."""
        store.measure("current", 0.41, 0.36, 0.48, weights="v9", seeds=None)
        self._crowd(store, votes_first=False)
        quoted = badges.line(self._rows(store, badges.CLANKER)[0])["quoted"]
        assert "index score" not in quoted and "0.41" not in quoted
        assert quoted == "5.0 stars from 5 people and 5 people who had played it"

    def test_a_second_event_writes_no_duplicate(self, store: Scratch) -> None:
        store.measure("current", 0.41, 0.36, 0.48)
        people = self._crowd(store, votes_first=False)
        _play_out(store, people[0])
        store.vote(people[0], 5)
        kinds = [award.kind for award in store.on_game() if award.kind in ON_THE_GAME]
        assert sorted(kinds) == sorted(ON_THE_GAME)


def test_the_repository_ships_the_ruleset_this_file_plays() -> None:
    """The store tests skip without a database; this one does not, so a `dull.json` that
    moved is a failure here rather than a silent skip everywhere."""
    assert Path(library.RULESET_DIR / "dull.json").exists()
