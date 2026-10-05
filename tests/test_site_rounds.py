"""The round: the members' vote, then the maintainer's confirm or veto (Q503h).

`docs/plan-game-pipeline.md` §2.2 *The round, in order*, §3.2's *Where it stands* lines,
§3.3's round page and §3.4's `/admin`, at the user's answers Q1 B and Q4 A and §6's
first *needs decision* on its default. `site/rounds.py` holds it; the claims here are:

**The sweep.** On the 1st a round opens with every shortlisted game whose four runs are
current, and a passed game the last round left undecided, carried first with its vote. A
month whose first sweep comes after the week opens none. At the close the votes are
counted, and a game short of more yes than no and 4 yes goes back on trial.

**The vote.** Only a member who finished a game on the nominee, is not its designer and
holds a week-old account votes, yes or no, changeable until the close; no count is sent
before it.

**The decision.** The maintainer confirms or vetoes a passed game with a reason, at most
four a round in rank order, never one the vote refused. A veto, like a vote down, starts
the 60-day wait before a new nomination.

**The handover.** A confirmed game stays Shortlisted until its spec is in `archive/`; then
`site load` takes the row over as Library, its designer kept, with the confirm's reason on
its page and in *What's new*.

**`site sweep`** runs the resting sweep and the round's.

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_admin import DULL, Site, _running_site, _saved, _screened
from test_site_shortlist import _match, _person

from chess_variant_sandbox import cli
from chess_variant_sandbox.site import db, quota, resting, rounds, shortlist, stages
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.spec import spec_hash

READING = site_store.STAND_IN_READING
UTC = timezone.utc
NOW = datetime.now(UTC)
FIRST = rounds.start(rounds.next_month(rounds.month_of(NOW))) + timedelta(hours=9)
"""The next month's 1st, 09:00: every clock-driven test runs its round there, so the
real date never decides whether a round opens."""
DURING = FIRST + timedelta(days=2)
AFTER = FIRST + timedelta(days=8)
NEXT = rounds.start(rounds.next_month(FIRST.date())) + timedelta(hours=9)


# ------------------------------------------------------------------------ rows by hand


class World:
    """A running site over an empty store, and the nominees and voters a round needs."""

    def __init__(self, site: Site, root: Path) -> None:
        self.site = site
        self.root = root
        self.made = 0

    @property
    def store(self) -> Any:
        return self.site.store

    def designer(self, handle: str) -> int:
        return _person(self.site, handle)

    def nominee(
        self, name: str, measured: bool = True, nominated_ago: int = 5,
        spec: dict[str, Any] | None = None,
    ) -> str:
        """A shortlisted player-made game, as its row would be, with its four runs current
        (`measured`)."""
        self.made += 1
        owner = self.designer(f"maker{self.made}")
        digest = f"{self.made:064x}" if spec is None else spec_hash(spec)
        with self.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, owner, visibility, origin, published,"
                " nominated) values (%s, %s::jsonb, %s, %s, 'public', 'designed',"
                " now() - interval '40 days', now() - %s)",
                (digest, json.dumps(spec or {}), name, owner, timedelta(days=nominated_ago)),
            )
            if measured:
                self._measured(cursor, digest)
        self.store.commit()
        return digest

    @staticmethod
    def _measured(cursor: Any, digest: str) -> None:
        cursor.execute(
            "insert into measurements (hash, directory, state, headline)"
            " values (%s, %s, 'current', 0.61)",
            (digest, digest),
        )

    def voters(self, game: str, n: int) -> list[int]:
        """`n` members with week-old accounts who each finished a game on `game`."""
        people = []
        for _ in range(n):
            self.made += 1
            person = _person(self.site, f"voter{self.made}", age=timedelta(days=90))
            _match(self.site, game, person)
            people.append(person)
        return people

    def cast(self, game: str, yes: int, no: int, when: datetime = DURING) -> None:
        for index, person in enumerate(self.voters(game, yes + no)):
            rounds.vote(self.store, FIRST.date(), game, person, index < yes, when)
        self.store.commit()

    def sweep(self, when: datetime) -> rounds.Swept:
        done = rounds.sweep(self.store, when)
        self.store.commit()
        return done

    def decide(self, game: str, confirm: bool, reason: str = "A fine game.") -> rounds.Decided:
        done = rounds.decide(self.store, game, confirm, reason, None, AFTER + timedelta(days=1))
        self.store.commit()
        return done

    def level(self, game: str) -> str:
        level = str(stages.read(self.store, [game])[game]["level"])
        self.store.commit()
        return level

    def column(self, game: str, column: str) -> Any:
        with self.store.cursor() as cursor:
            cursor.execute(f"select {column} from games where hash = %s", (game,))  # noqa: S608
            found = cursor.fetchone()[0]
        self.store.commit()
        return found


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    _screened(monkeypatch, passed=True)
    with _running_site(tmp_path) as site:
        yield World(site, tmp_path)


def _entrant(**given: Any) -> rounds.Entrant:
    values: dict[str, Any] = {
        "round": date(2026, 10, 1), "game": "g", "name": "g", "owner": 1, "designer": "ada",
        "nominated": datetime(2026, 9, 1, tzinfo=UTC), "carried": None, "yes": 5, "no": 1,
        "decision": None, "reason": "", "decided": None, "score": 0.6,
    }
    values.update(given)
    return rounds.Entrant(**values)


# ------------------------------------------------------------------------ the rules alone


class TestTheRules:
    def test_the_numbers_live_in_quota(self) -> None:
        assert (timedelta(days=7), 4, 4, timedelta(days=60)) == (
            quota.ROUND_VOTING, quota.ROUND_YES, quota.ROUND_PLACES, quota.DECLINED_WAIT
        )
        said = dict(quota.summary())
        assert "at least 4 yes" in said["A round"]
        assert said["Games the maintainer may confirm at one round"] == "4"

    @pytest.mark.parametrize(
        ("yes", "no", "passed"),
        [(4, 0, True), (4, 3, True), (3, 0, False), (4, 4, False), (9, 5, True), (5, 6, False)],
    )
    def test_a_pass_is_more_yes_than_no_and_four_yes(
        self, yes: int, no: int, passed: bool
    ) -> None:
        assert rounds.passes(yes, no) is passed

    def test_the_places_go_carried_first_then_share_then_the_oldest_nomination(self) -> None:
        old = datetime(2026, 8, 1, tzinfo=UTC)
        field = [
            _entrant(game="a", yes=5, no=1),
            _entrant(game="b", yes=9, no=0),
            _entrant(game="c", yes=5, no=1, nominated=old),
            _entrant(game="d", yes=4, no=3, carried=date(2026, 9, 1)),
            _entrant(game="e", yes=6, no=0),
            _entrant(game="f", yes=1, no=5),
        ]
        assert [one.game for one in rounds.ranked(field)] == ["d", "b", "e", "c", "a", "f"]
        # Four places: the fifth passed game waits, the one voted down never counts.
        assert rounds.running(field) == {"d", "b", "e", "c"}
        # A veto frees a place for the next in rank.
        vetoed = [
            _entrant(**{**one.__dict__, "decision": rounds.VETOED}) if one.game == "b" else one
            for one in field
        ]
        assert rounds.running(vetoed) == {"d", "e", "c", "a"}


# ------------------------------------------------------------------------ the sweep


class TestOpeningAndClosing:
    def test_on_the_first_the_measured_shortlist_enters(self, world: World) -> None:
        ready = world.nominee("ready")
        world.nominee("unmeasured", measured=False)
        done = world.sweep(FIRST)
        assert done.opened is not None and done.opened.day == FIRST.date()
        assert done.entered == ["ready"]
        assert done.opened.closes == rounds.start(FIRST.date()) + quota.ROUND_VOTING
        assert [one.game for one in rounds.entrants(world.store, FIRST.date())] == [ready]
        # A second sweep does nothing.
        again = world.sweep(FIRST + timedelta(hours=1))
        assert again.opened is None and again.closed == [] and again.lines == []

    def test_a_month_first_swept_after_its_week_opens_no_round(self, world: World) -> None:
        world.nominee("late")
        assert world.sweep(FIRST + timedelta(days=7)).opened is None
        assert world.site.count("rounds") == 0

    def test_a_game_that_is_not_shortlisted_never_enters(self, world: World) -> None:
        trial = world.nominee("trial only")
        with world.store.cursor() as cursor:
            cursor.execute("update games set nominated = null where hash = %s", (trial,))
        world.store.commit()
        assert world.sweep(FIRST).entered == []

    def test_the_close_counts_and_sends_a_game_voted_down_back_on_trial(
        self, world: World
    ) -> None:
        up = world.nominee("up")
        down = world.nominee("down")
        world.sweep(FIRST)
        world.cast(up, yes=4, no=1)
        world.cast(down, yes=3, no=0)
        assert world.sweep(DURING).closed == []
        done = world.sweep(AFTER)
        assert [one.day for one in done.closed] == [FIRST.date()]
        assert done.voted_down == ["down"]
        found = {one.game: one for one in rounds.entrants(world.store, FIRST.date())}
        assert (found[up].yes, found[up].no, found[up].passed) == (4, 1, True)
        assert (found[down].yes, found[down].no, found[down].decision) == (
            3, 0, rounds.VOTED_DOWN
        )
        assert world.level(up) == stages.SHORTLISTED
        assert world.level(down) == stages.ON_TRIAL
        assert world.column(down, "nominated") is None
        # `/admin` says the decision in words, never the stored `voted-down` (Q537).
        rows = {row["hash"]: row for row in rounds.admin_view(world.store, AFTER)["entrants"]}
        world.store.commit()
        assert rows[down]["decisionSays"] == "Voted down: 3 yes and 0 no"
        assert rows[up]["decisionSays"] == ""
        assert all(rounds.VOTED_DOWN not in row["decisionSays"] for row in rows.values())

    def test_a_shut_accounts_vote_is_not_counted(self, world: World) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=4, no=0)
        with world.store.cursor() as cursor:
            cursor.execute(
                "update users set revoked = now() where id = (select max(person) from"
                " round_votes)"
            )
        world.store.commit()
        world.sweep(AFTER)
        assert world.level(game) == stages.ON_TRIAL


# ------------------------------------------------------------------------ the vote


class TestWhoVotes:
    def test_a_finisher_votes_and_may_change_it_until_the_close(self, world: World) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        [bea] = world.voters(game, 1)
        said = rounds.vote(world.store, FIRST.date(), game, bea, True, DURING)
        assert said.startswith("Your yes is in.")
        rounds.vote(world.store, FIRST.date(), game, bea, False, DURING)
        world.store.commit()
        assert rounds.ballots(world.store, FIRST.date(), bea) == {game: False}
        assert world.site.count("round_votes") == 1
        with pytest.raises(rounds.RoundError, match="is closed"):
            rounds.vote(world.store, FIRST.date(), game, bea, True, AFTER)
        world.store.rollback()

    def test_the_designer_a_non_finisher_and_a_new_account_may_not(self, world: World) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        owner = int(world.column(game, "owner"))
        stranger = _person(world.site, "stranger")
        # A day old when it votes, whenever that is.
        newcomer = _person(world.site, "newcomer", age=NOW - DURING + timedelta(days=1))
        _match(world.site, game, newcomer)
        _match(world.site, game, owner)
        refusals = {
            owner: "own game",
            stranger: "finished a game on it",
            newcomer: f"{quota.VOTER_ACCOUNT_AGE.days} days old",
        }
        for person, why in refusals.items():
            with pytest.raises(rounds.RoundError, match=why):
                rounds.vote(world.store, FIRST.date(), game, person, True, DURING)
            world.store.rollback()

    def test_no_count_is_shown_before_the_close(self, world: World) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=5, no=0)
        during = rounds.page(world.store, FIRST.date(), None, DURING)
        assert during["state"] == rounds.VOTING
        [shown] = during["entrants"]
        assert (shown["yes"], shown["no"], shown["passed"]) == (None, None, None)
        world.sweep(AFTER)
        after = rounds.page(world.store, FIRST.date(), None, AFTER)
        assert after["state"] == rounds.DECIDING
        [shown] = after["entrants"]
        assert (shown["yes"], shown["no"], shown["passed"]) == (5, 0, True)
        assert shown["score"] == pytest.approx(0.61)


# ------------------------------------------------------------------------ the decision


class TestTheMaintainersDecision:
    def test_nothing_is_decided_before_the_close_or_without_a_reason(
        self, world: World
    ) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=4, no=0)
        with pytest.raises(rounds.RoundError, match="decide after the close"):
            rounds.decide(world.store, game, True, "Good.", None, DURING)
        world.store.rollback()
        world.sweep(AFTER)
        with pytest.raises(rounds.RoundError, match="needs its reason"):
            world.decide(game, True, reason="   ")
        world.store.rollback()

    def test_a_game_the_vote_refused_cannot_be_confirmed(self, world: World) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=2, no=0)
        world.sweep(AFTER)
        with pytest.raises(rounds.RoundError, match="the vote refused"):
            world.decide(game, True)
        world.store.rollback()

    def test_a_confirm_stays_shortlisted_and_says_so_with_the_command(
        self, world: World
    ) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=4, no=0)
        world.sweep(AFTER)
        done = world.decide(game, True, reason="Plays like nothing else here.")
        assert done.decision == rounds.CONFIRMED
        assert done.command.startswith(f"chess-variant-sandbox archive {game} --credit maker1 --note ")
        assert "Plays like nothing else here. Confirmed at the round of" in done.command
        assert world.level(game) == stages.SHORTLISTED
        row = stages.read(world.store, [game])[game]
        said = stages.said(world.store, row, None)
        world.store.commit()
        assert said.startswith(f"Confirmed at the round of {rounds.said_day(FIRST)}: Plays")
        assert "It joins the library at the site's next update." in said
        with pytest.raises(rounds.RoundError, match="already decided"):
            world.decide(game, False)
        world.store.rollback()

    def test_a_veto_sends_it_back_on_trial_with_the_wait(self, world: World) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=6, no=1)
        world.sweep(AFTER)
        world.decide(game, False, reason="Too close to chess.")
        assert world.level(game) == stages.ON_TRIAL
        row = stages.read(world.store, [game])[game]
        said = stages.said(world.store, row, None)
        world.store.commit()
        assert said.startswith(
            f"Not taken at the round of {rounds.said_day(FIRST)}: vetoed: Too close to chess."
        )
        assert "It can be nominated again after " in said

    def test_the_admin_row_reads_an_entrants_premise_check(
        self, world: World, tmp_path: Path
    ) -> None:
        """Q502d: where a check is on file its verdict replaces *not checked yet*."""
        from test_premise import SPEC, _synthetic

        from chess_variant_sandbox import premise

        checked = world.nominee("checked", spec=SPEC)
        plain = world.nominee("unchecked")
        world.sweep(FIRST)
        runs = tmp_path / "premise-runs"
        premise.write(_synthetic(premise.FAILS), runs)
        admin = rounds.admin_view(world.store, DURING, runs_dir=runs)
        world.store.commit()
        rows = {one["hash"]: one for one in admin["entrants"]}
        assert rows[checked]["premise"].startswith("fails: Scored by players")
        assert rows[plain]["premise"] == rounds.PREMISE
        # Without a runs directory nothing is read.
        admin = rounds.admin_view(world.store, DURING)
        world.store.commit()
        assert {one["premise"] for one in admin["entrants"]} == {rounds.PREMISE}

    def test_at_most_four_are_confirmed_and_the_fifth_is_carried(self, world: World) -> None:
        games = [world.nominee(f"pick {n}", nominated_ago=20 - n) for n in range(5)]
        world.sweep(FIRST)
        for game in games:
            world.cast(game, yes=5, no=1)
        world.sweep(AFTER)
        for game in games[:4]:
            world.decide(game, True)
        with pytest.raises(rounds.RoundError, match="places this round go to"):
            world.decide(games[4], True)
        world.store.rollback()
        admin = rounds.admin_view(world.store, AFTER)
        world.store.commit()
        last = next(one for one in admin["entrants"] if one["hash"] == games[4])
        assert last["mayConfirm"] is False and last["mayVeto"] is True
        assert "carried to the next round" in last["why"]

        # The next round opens with it first, its vote kept and no new vote on it.
        fresh = world.nominee("fresh")
        done = world.sweep(NEXT)
        assert done.entered == ["pick 4", "fresh"]
        field = rounds.entrants(world.store, NEXT.date())
        assert [one.game for one in field] == [games[4], fresh]
        assert (field[0].carried, field[0].yes, field[0].no) == (FIRST.date(), 5, 1)
        [someone] = world.voters(games[4], 1)
        with pytest.raises(rounds.RoundError, match="keeps the vote it had"):
            rounds.vote(world.store, NEXT.date(), games[4], someone, True,
                        NEXT + timedelta(days=1))
        world.store.rollback()
        # The confirmed four never enter again.
        assert not {one.game for one in field} & set(games[:4])

    def test_a_veto_frees_a_place_for_the_next_in_rank(self, world: World) -> None:
        games = [world.nominee(f"pick {n}", nominated_ago=20 - n) for n in range(5)]
        world.sweep(FIRST)
        for game in games:
            world.cast(game, yes=5, no=1)
        world.sweep(AFTER)
        world.decide(games[0], False)
        world.decide(games[4], True)
        assert world.level(games[4]) == stages.SHORTLISTED


# ------------------------------------------------------------------------ the wait


class TestTheSixtyDayWait:
    def test_a_declined_game_waits_sixty_days_before_a_new_nomination(
        self, world: World
    ) -> None:
        game = world.nominee("game")
        world.sweep(FIRST)
        world.cast(game, yes=1, no=4)
        world.sweep(AFTER)
        owner = int(world.column(game, "owner"))
        # The rest of the bar, met: a month on trial, 8 finishers voting 5 stars.
        with world.store.cursor() as cursor:
            cursor.execute(
                "insert into votes (voter, game, stars) select person, game, 5"
                " from round_votes where game = %s",
                (game,),
            )
        world.store.commit()
        for person in world.voters(game, 3):
            with world.store.cursor() as cursor:
                cursor.execute(
                    "insert into votes (voter, game, stars) values (%s, %s, 5)", (person, game)
                )
            world.store.commit()
        bar = shortlist.bar(world.store, game, READING)
        world.store.commit()
        assert bar is not None
        unmet = [check.key for check in bar.checks if not check.met]
        assert unmet == ["wait"]
        with pytest.raises(shortlist.NominateError, match="can be nominated again after"):
            shortlist.nominate(world.store, game, owner, READING)
        world.store.rollback()

        with world.store.cursor() as cursor:
            cursor.execute(
                "update round_decisions set decided = now() - interval '61 days'"
                " where game = %s",
                (game,),
            )
        world.store.commit()
        bar = shortlist.bar(world.store, game, READING)
        world.store.commit()
        assert bar is not None and bar.eligible
        shortlist.nominate(world.store, game, owner, READING)
        world.store.commit()
        assert world.level(game) == stages.SHORTLISTED

    def test_a_game_never_declined_has_five_lines(self, world: World) -> None:
        game = world.nominee("game")
        bar = shortlist.bar(world.store, game, READING)
        world.store.commit()
        assert bar is not None
        assert [check.key for check in bar.checks] == [
            "days", "finishers", "votes", "stars", "reading"
        ]


# ------------------------------------------------------------------------ the handover


def _spec(max_plies: int, name: str, history: str = "") -> dict[str, Any]:
    spec = json.loads(json.dumps(DULL))
    spec["rules"]["max_plies"] = max_plies
    spec["name"] = name
    spec["provenance"] = {"origin": "human"}
    if history:
        spec["provenance"]["history"] = history
    return spec


def _load(world: World) -> loader.Report:
    report = loader.load(world.store, world.root)
    world.store.commit()
    return report


class TestTheLibraryHandover:
    def test_a_confirmed_game_becomes_library_when_its_archive_is_loaded(
        self, world: World
    ) -> None:
        spec = _spec(9, "crowd pick")
        game = world.nominee("crowd pick", spec=spec)
        owner = world.column(game, "owner")
        world.sweep(FIRST)
        world.cast(game, yes=4, no=0)
        world.sweep(AFTER)
        world.decide(game, True, reason="The members loved its endgame.")

        # Loaded before the archive is deployed: nothing changes.
        _load(world)
        assert world.level(game) == stages.SHORTLISTED
        assert world.column(game, "origin") == "designed"

        # The maintainer's `archive --credit --note`: the same mechanics, the note first.
        archived = _spec(9, "crowd pick", "The members loved its endgame. Designed by maker1.")
        assert spec_hash(archived) == game
        (world.root / "archive" / "crowd-pick.json").write_text(
            json.dumps(archived), encoding="utf-8"
        )
        report = _load(world)
        assert report.moved.added == 1
        assert any("joined the Library from its round" in note for note in report.notes)
        assert world.level(game) == stages.LIBRARY
        assert world.column(game, "origin") == "archive"
        assert world.column(game, "owner") == owner, "the designer keeps the credit"
        assert shortlist.shortlisted(world.store) == 0
        world.store.commit()
        with world.store.cursor() as cursor:
            cursor.execute("select was, became, reason from level_moves where game = %s", (game,))
            assert cursor.fetchall() == [
                ("designed", "archive", "The members loved its endgame.")
            ]
        world.store.commit()

        # The test's own clock (Q730): late in UTC's day, so a session zone ahead of UTC
        # would name the next day, and after every changelog entry.
        moved = datetime(2099, 1, 1, 23, 30, tzinfo=UTC)
        with world.store.cursor() as cursor:
            cursor.execute("update level_moves set moved = %s where game = %s", (moved, game))
        world.store.commit()
        page = world.site.browser().get(f"/api/game/{game}").payload
        assert page["level"] == stages.LIBRARY
        assert page["levelSays"] == (
            "In the library since 1 Jan 2099: The members loved its endgame."
        )
        news = world.site.browser().get("/api/changelog").payload
        headings = [entry["heading"] for day in news["days"] for entry in day["entries"]]
        assert "crowd pick joins the Library" in headings
        assert "crowd pick is confirmed for the library" in headings

        # A second load writes nothing, and keeps the designer as owner.
        again = _load(world)
        assert again.games.written == 0 and again.moved.written == 0
        assert world.column(game, "owner") == owner

    def test_an_unconfirmed_games_spec_in_the_tree_leaves_its_row_alone(
        self, world: World
    ) -> None:
        spec = _spec(11, "not voted")
        game = world.nominee("not voted", spec=spec)
        (world.root / "archive" / "not-voted.json").write_text(json.dumps(spec), encoding="utf-8")
        _load(world)
        assert world.column(game, "origin") == "designed"
        assert world.level(game) == stages.SHORTLISTED

    def test_the_load_indexes_a_shortlisted_games_own_runs(self, world: World) -> None:
        spec = _spec(13, "measured by hand")
        game = world.nominee("measured by hand", measured=False, spec=spec)
        assert loader.shortlisted_specs(world.store, set()) == {game: (spec, loader.ARCHIVED)}
        world.store.commit()


# ------------------------------------------------------------------------ over HTTP


def _round_now(world: World) -> date:
    """A round open now, whatever today's date: this month's, its week moved round now."""
    day = rounds.month_of(NOW)
    world.sweep(rounds.start(day) + timedelta(minutes=1))
    with world.store.cursor() as cursor:
        cursor.execute(
            "update rounds set opens = now() - interval '1 day',"
            " closes = now() + interval '6 days' where day = %s",
            (day,),
        )
    world.store.commit()
    return day


class TestTheRoundIsShownOnlyWithAnEntrant:
    """L4 of `docs/plan-launch.md` §3.4. The footer links the round only while the latest
    round has an entrant; a round nobody entered is not announced, and its address still
    says when the next opens and what puts a game into it."""

    def test_with_no_round_nothing_is_announced(self, world: World) -> None:
        assert rounds.announced(world.store) is False
        assert world.site.browser().get("/api/me").payload["round"] is False
        page = rounds.page(world.store, None, None, FIRST - timedelta(days=3))
        assert page["round"] is None
        assert page["says"].startswith("No round has opened yet.")
        assert f"the next is on {rounds.said_day(FIRST)}." in page["says"]
        assert page["says"].endswith(rounds.ENTERS)

    def test_a_round_nobody_entered_is_not_announced_and_says_how_to_enter(
        self, world: World
    ) -> None:
        world.sweep(FIRST)
        assert rounds.latest(world.store) is not None
        assert rounds.announced(world.store) is False
        page = rounds.page(world.store, None, None, DURING)
        assert page["entrants"] == []
        assert page["says"].startswith("No game entered")
        assert f"The next round opens on {rounds.said_day(NEXT)}." in page["says"]
        assert page["says"].endswith(rounds.ENTERS)

    def test_an_entrant_announces_the_round(self, world: World) -> None:
        world.nominee("entered")
        _round_now(world)
        assert rounds.announced(world.store) is True
        assert world.site.browser().get("/api/me").payload["round"] is True


class TestThePages:
    def test_with_no_round_the_page_says_when_the_first_opens(self, world: World) -> None:
        page = world.site.browser().get("/api/round").payload
        # In the first week of a month the page's own sweep opens that month's round,
        # which then has no entrant; any other day there is none yet.
        if page["round"] is None:
            assert "No round has opened yet" in page["says"]
        else:
            assert page["entrants"] == [] and "No game entered" in page["says"]
        assert world.site.browser().get("/api/round/not-a-date").status == 404
        assert world.site.browser().get("/api/round/1999-01-01").status == 404

    def test_a_member_votes_through_the_page_and_the_game_page(self, world: World) -> None:
        game = world.nominee("voted on")
        day = _round_now(world)
        bea = world.site.signed_in("bea")
        with world.store.cursor() as cursor:
            cursor.execute(
                "update users set created = now() - interval '30 days' where handle = 'bea'"
            )
            cursor.execute("select id from users where handle = 'bea'")
            bea_row = int(cursor.fetchone()[0])
        world.store.commit()

        page = bea.get(f"/api/round/{day.isoformat()}").payload
        [entry] = page["entrants"]
        assert entry["hash"] == game and entry["mayVote"] is False
        assert "finished a game" in entry["why"]
        refused = bea.post(f"/api/round/{day.isoformat()}/vote", {"hash": game, "yes": True})
        assert refused.status == 400 and "finished a game" in refused.payload["error"]

        _match(world.site, game, bea_row)
        answer = bea.post(f"/api/round/{day.isoformat()}/vote", {"hash": game, "yes": True})
        assert answer.status == 200, answer.text
        assert answer.payload["said"].startswith("Your yes is in.")
        [entry] = answer.payload["entrants"]
        assert entry["yourVote"] is True and entry["yes"] is None

        latest = bea.get("/api/round").payload
        assert latest["round"] == day.isoformat() and latest["state"] == rounds.VOTING

        visitor = world.site.browser().post(
            f"/api/round/{day.isoformat()}/vote", {"hash": game, "yes": True}
        )
        assert visitor.status == 400 and "Sign in" in visitor.payload["error"]

    def test_the_game_page_names_the_round_and_offers_the_vote(self, world: World) -> None:
        ada = world.site.signed_in("ada")
        game = _saved(ada, _spec(15, "ada's nominee")).payload["hash"]
        site_store.on_trial(world.store, game)
        with world.store.cursor() as cursor:
            cursor.execute("update games set nominated = now() where hash = %s", (game,))
            World._measured(cursor, game)
        world.store.commit()
        day = _round_now(world)
        page = world.site.browser().get(f"/api/game/{game}").payload
        assert page["level"] == stages.SHORTLISTED
        assert page["levelSays"].startswith(
            f"In the round of {rounds.said_day(day)}: members are voting until "
        )
        assert page["round"]["round"] == day.isoformat()
        assert page["round"]["mayVote"] is False and "Sign in" in page["round"]["why"]
        # Q517: the panel draws the level's line and the round's one under the other, so
        # the close is said once, not twice.
        closes = rounds.said_day(rounds.get(world.store, day).closes)
        world.store.commit()
        assert (page["levelSays"] + " " + page["round"]["says"]).count(closes) == 1
        # Its designer is told why not, too.
        mine = ada.get(f"/api/game/{game}").payload["round"]
        assert mine["mayVote"] is False and "own game" in mine["why"]

    def test_the_maintainer_confirms_and_vetoes_from_admin(self, world: World) -> None:
        kept = world.nominee("kept")
        refused = world.nominee("refused")
        day = rounds.month_of(NOW)
        world.sweep(rounds.start(day) + timedelta(minutes=1))
        for game in (kept, refused):
            for person in world.voters(game, 4):
                rounds.vote(world.store, day, game, person, True,
                            rounds.start(day) + timedelta(hours=1))
        world.store.commit()
        # Its week is over by now, or moved so: the page's own sweep closes it.
        with world.store.cursor() as cursor:
            cursor.execute(
                "update rounds set closes = now() - interval '1 minute' where day = %s",
                (day,),
            )
        world.store.commit()
        boss = world.site.signed_in("boss", role="admin")
        view = boss.get("/api/admin").payload["round"]
        assert view["state"] == rounds.DECIDING
        rows = {row["hash"]: row for row in view["entrants"]}
        assert rows[kept]["yes"] == 4 and rows[kept]["mayConfirm"] is True
        assert rows[kept]["premise"] == rounds.PREMISE and rows[kept]["cost"] == rounds.COST

        member = world.site.signed_in("member")
        assert member.post(
            "/api/admin", {"act": "confirm-entrant", "hash": kept, "reason": "Yes."}
        ).status == 404
        done = boss.post(
            "/api/admin", {"act": "confirm-entrant", "hash": kept, "reason": "A keeper."}
        )
        assert done.status == 200, done.text
        assert done.payload["command"].startswith(f"chess-variant-sandbox archive {kept}")
        vetoed = boss.post(
            "/api/admin", {"act": "veto-entrant", "hash": refused, "reason": "Too slow."}
        )
        assert vetoed.status == 200, vetoed.text
        no_reason = boss.post("/api/admin", {"act": "veto-entrant", "hash": kept})
        assert no_reason.status == 400
        assert world.level(refused) == stages.ON_TRIAL
        rows = {row["hash"]: row for row in boss.get("/api/admin").payload["round"]["entrants"]}
        assert rows[kept]["decision"] == rounds.CONFIRMED and rows[kept]["command"]
        assert rows[refused]["decision"] == rounds.VETOED
        assert rows[kept]["decisionSays"] == "Confirmed: A keeper."
        assert rows[refused]["decisionSays"] == "Vetoed: Too slow."

        page = world.site.browser().get(f"/api/round/{day.isoformat()}").payload
        says = {one["hash"]: one["decisionSays"] for one in page["entrants"]}
        assert says == {kept: "Confirmed: A keeper.", refused: "Vetoed: Too slow."}

    def test_the_round_page_and_the_admin_panel_are_drawn_from_files(self) -> None:
        from page_text import PAGE

        from chess_variant_sandbox.site import pages
        from chess_variant_sandbox.web.page import SCRIPTS

        assert "/static/page/round.js" in SCRIPTS
        source = SCRIPTS["/static/page/round.js"].read_text(encoding="utf-8")
        assert "function showRound" in source and "/api/round" in source
        assert "parts[0] === 'round'" in PAGE
        admin_page = pages.admin("boss")
        assert 'id="round-entrants"' in admin_page
        admin_js = (Path(pages.__file__).parent / "static" / "admin.js").read_text(
            encoding="utf-8"
        )
        assert "confirm-entrant" in admin_js and "veto-entrant" in admin_js
        # Q537: the round's table is a block per entrant on a phone, its cells labelled by
        # their headings, and the decision cell prints the server's words.
        assert 'class="round" id="round-table"' in admin_page
        assert "table.round thead { display: none; }" in admin_page
        assert "td.dataset.label" in admin_js and "one.decisionSays" in admin_js
        assert "one.decision +" not in admin_js
        for text in (admin_page, source):
            assert " onclick=" not in text and "<script>" not in text


# ------------------------------------------------------------------------ site sweep


class TestSiteSweep:
    def test_the_command_rests_and_runs_the_round(
        self, world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        quiet = world.nominee("quiet trial")
        with world.store.cursor() as cursor:
            cursor.execute(
                "update games set nominated = null, published = now() - %s where hash = %s",
                (quota.RESTING_AFTER + timedelta(days=1), quiet),
            )
        world.store.commit()
        world.nominee("entrant")

        class Held:
            def __enter__(self) -> Any:
                return world.store

            def __exit__(self, *exc: Any) -> None:
                return None

        monkeypatch.setattr(db, "database_url", lambda: "postgresql:///scratch")
        monkeypatch.setattr(db, "connect", lambda url: Held())
        monkeypatch.setattr(rounds, "_now", lambda when: FIRST if when is None else when)
        assert cli.main(["site", "sweep"]) == 0
        out = capsys.readouterr().out
        assert "resting: 1 trial game(s) rested" in out
        assert f"round: opened the round of {rounds.said_day(FIRST)} with 1 entrant(s)" in out
        assert world.level(quiet) == stages.RESTING
        assert resting.rest(world.store) == []
        world.store.commit()
