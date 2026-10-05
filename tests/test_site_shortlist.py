"""The shortlist's bar, the votes that count toward it, and *Nominate* (Q503f).

`docs/plan-game-pipeline.md` §2.2's *On trial → Shortlisted* row, at the user's answer Q2:
30 days on trial, 8 other people finished a game on it, 8 qualifying votes averaging 4
stars or more, and a current reading. `site/shortlist.py` counts each from rows, and the
claims here are four.

**The counts.** A finish is a game ended on the board, sandbox or leaderboard; a
resignation and a watch are not, and the designer is never one of the others. A vote
qualifies from somebody who finished a game on it, is not the designer, and holds an
account a week old that nobody shut.

**The press.** `POST /api/game/<hash>/nominate` stamps `games.nominated` for the owner of
an eligible trial game, so `stages.stage` then reads Shortlisted; it refuses anybody else,
a game short of the bar (naming every unmet line), a second shortlisted game of one
designer, and a full shortlist.

**The page.** The game page carries the bar's five lines and the designer's button, and
the star panel carries §3.3's sentence; the page draws both as sent. The designer's own
track (`#/design`'s *Where this game stands*) carries the same bar and button (Q518).

**The checkout's designer** draws its track from `candidacy.standing` (Q503b's loose end).

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_admin import (
    DULL,
    DULL_HASH,
    Site,
    _running_site,
    _saved,
    _screened,
    _template,
)
from test_web import _in_page

from chess_variant_sandbox.site import provisional, quota, shortlist, stages
from chess_variant_sandbox.web import candidacy, library

READING = site_store.STAND_IN_READING


# ------------------------------------------------------------------------ rows by hand


def _person(site: Site, handle: str, age: timedelta = timedelta(days=30)) -> int:
    found = site.person(handle)
    with site.store.cursor() as cursor:
        cursor.execute("update users set created = now() - %s where id = %s", (age, found))
    site.store.commit()
    return found


def _match(
    site: Site,
    game: str,
    person: int,
    mode: str = "sandbox",
    reason: str = "checkmate",
    kept: bool = True,
    verdict: bool = False,
    watch: bool = False,
) -> str:
    """One match on `game` with `person` in seat 0: its finish kept (`kept`, as
    `finishes.record` writes it) and its verdict on the row (`verdict`)."""
    match = secrets.token_hex(6)
    with site.store.cursor() as cursor:
        cursor.execute(
            "insert into matches (id, game, mode, seat0_user, seat0_agent, seat1_agent,"
            " seed, result, reason, finished)"
            " values (%s, %s, %s, %s, %s, 'random', 0, %s, %s, %s)",
            (
                match,
                game,
                mode,
                None if watch else person,
                "random" if watch else None,
                1.0 if verdict else None,
                reason if verdict else None,
                datetime.now(timezone.utc) if verdict else None,
            ),
        )
        if kept:
            cursor.execute(
                "insert into finishes (match, ordinal, moves, result, reason, takebacks)"
                " values (%s, 1, '[]'::jsonb, 1, %s, 0)",
                (match, reason),
            )
    site.store.commit()
    return match


def _vote(site: Site, game: str, person: int, stars: int) -> None:
    with site.store.cursor() as cursor:
        cursor.execute(
            "insert into votes (voter, game, stars) values (%s, %s, %s)",
            (person, game, stars),
        )
    site.store.commit()


def _published(site: Site, game: str, ago: timedelta) -> None:
    with site.store.cursor() as cursor:
        cursor.execute("update games set published = now() - %s where hash = %s", (ago, game))
    site.store.commit()


class Trial:
    """A running site, ada's game on trial and a crowd who may play and vote on it."""

    def __init__(self, site: Site, ada: Any, ada_id: int, game: str) -> None:
        self.site = site
        self.ada = ada
        self.ada_id = ada_id
        self.game = game
        self.made = 0

    def crowd(self, n: int, stars: int = 5) -> list[int]:
        """`n` more people who each finish a game on it and vote `stars`."""
        people = []
        for _ in range(n):
            self.made += 1
            person = _person(self.site, f"player{self.made}")
            _match(self.site, self.game, person)
            _vote(self.site, self.game, person, stars)
            people.append(person)
        return people

    def eligible(self) -> None:
        _published(self.site, self.game, timedelta(days=quota.SHORTLIST_DAYS + 1))
        self.crowd(quota.SHORTLIST_FINISHERS)

    def bar(self, reading: Any = READING) -> shortlist.Bar:
        found = shortlist.bar(self.site.store, self.game, reading)
        self.site.store.commit()
        assert found is not None
        return found

    def level(self, game: str | None = None) -> str:
        digest = game or self.game
        return str(stages.read(self.site.store, [digest])[digest]["level"])


@pytest.fixture
def trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Trial]:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    _screened(monkeypatch, passed=True)
    with _running_site(tmp_path) as site:
        ada = site.signed_in("ada")
        with site.store.cursor() as cursor:
            cursor.execute("select id from users where handle = 'ada'")
            ada_id = int(cursor.fetchone()[0])
        game = _saved(ada, _template(0, "ada's trial")).payload["hash"]
        site_store.on_trial(site.store, game)
        yield Trial(site, ada, ada_id, game)


# ------------------------------------------------------------------------ the bar alone


def _bar(**given: Any) -> shortlist.Bar:
    values: dict[str, Any] = {
        "game": "g",
        "owner": 1,
        "since": datetime(2026, 10, 3, tzinfo=timezone.utc),
        "days": quota.SHORTLIST_DAYS,
        "finishers": quota.SHORTLIST_FINISHERS,
        "votes": quota.SHORTLIST_VOTES,
        "mean": quota.SHORTLIST_STARS,
        "reading": True,
    }
    values.update(given)
    return shortlist.Bar(**values)


class TestTheBar:
    def test_the_users_numbers_live_in_quota(self) -> None:
        assert (30, 8, 8, 4.0, timedelta(days=7), 8, 1) == (
            quota.SHORTLIST_DAYS,
            quota.SHORTLIST_FINISHERS,
            quota.SHORTLIST_VOTES,
            quota.SHORTLIST_STARS,
            quota.VOTER_ACCOUNT_AGE,
            quota.SHORTLIST_SIZE,
            quota.SHORTLISTED_A_DESIGNER,
        )
        said = dict(quota.summary())
        assert "The shortlist's bar" in said

    def test_exactly_at_the_bar_is_eligible(self) -> None:
        assert _bar().eligible
        assert [check.key for check in _bar().checks] == [
            "days", "finishers", "votes", "stars", "reading"
        ]

    @pytest.mark.parametrize(
        ("short", "key"),
        [
            ({"days": quota.SHORTLIST_DAYS - 1}, "days"),
            ({"finishers": quota.SHORTLIST_FINISHERS - 1}, "finishers"),
            ({"votes": quota.SHORTLIST_VOTES - 1}, "votes"),
            ({"mean": quota.SHORTLIST_STARS - 0.01}, "stars"),
            ({"mean": None, "votes": 0}, "stars"),
            ({"reading": False}, "reading"),
        ],
    )
    def test_each_line_short_of_it_is_named(self, short: dict[str, Any], key: str) -> None:
        found = _bar(**short)
        assert not found.eligible
        assert key in [check.key for check in found.checks if not check.met]

    def test_the_lines_say_the_counts_plainly(self) -> None:
        found = _bar(days=12, finishers=5, votes=4, mean=4.2)
        said = {check.key: check.says for check in found.checks}
        assert said["finishers"] == "5 of 8 other players have finished a game on it"
        assert said["votes"] == "4 of 8 votes count so far"
        assert said["stars"] == "they average 4.2 stars, of 4.0 needed"
        assert said["days"] == "on trial since 3 Oct 2026; its 30 days are up on 2 Nov 2026"

    def test_the_reading_line_speaks_to_whoever_reads_it(self) -> None:
        """Its designer is not told that only its designer can measure it again (Q537), and
        nobody is told about a *reading* (G44)."""
        stale = _bar(reading=False)
        theirs = {check.key: check.says for check in stale.lines(mine=False)}["reading"]
        mine = {check.key: check.says for check in stale.lines(mine=True)}["reading"]
        assert theirs == ("its score from its designer's computer is out of date; only its "
                          "designer can measure it again")
        assert "reading" not in theirs and "reading" not in mine
        assert "designer" not in mine and "your computer" in mine
        assert [c.met for c in stale.lines(mine=True)] == [c.met for c in stale.checks]
        current = {c.key: c.says for c in _bar().lines(mine=True)}["reading"]
        assert current == "its score from your computer is up to date"


# ------------------------------------------------------------------------ the counts


class TestWhoFinishedAGame:
    def test_a_sandbox_finish_counts_and_so_does_a_leaderboard_verdict(
        self, trial: Trial
    ) -> None:
        bea = _person(trial.site, "bea")
        cal = _person(trial.site, "cal")
        _match(trial.site, trial.game, bea)
        # A leaderboard game from before `finishes` kept every finish: its row's verdict.
        _match(trial.site, trial.game, cal, mode="leaderboard", kept=False, verdict=True)
        assert shortlist.finishers(trial.site.store, trial.game) == {bea, cal}
        assert trial.bar().finishers == 2

    def test_a_resignation_a_watch_and_an_unfinished_game_do_not(self, trial: Trial) -> None:
        bea = _person(trial.site, "bea")
        cal = _person(trial.site, "cal")
        dee = _person(trial.site, "dee")
        _match(trial.site, trial.game, bea, reason="resigned")
        _match(trial.site, trial.game, bea, mode="leaderboard", reason="resigned",
               kept=False, verdict=True)
        _match(trial.site, trial.game, cal, watch=True)
        _match(trial.site, trial.game, dee, kept=False)
        assert shortlist.finishers(trial.site.store, trial.game) == set()

    def test_the_designer_is_never_one_of_the_others(self, trial: Trial) -> None:
        _match(trial.site, trial.game, trial.ada_id)
        assert shortlist.finishers(trial.site.store, trial.game) == {trial.ada_id}
        assert trial.bar().finishers == 0

    def test_a_game_on_another_ruleset_is_not_this_ones(self, trial: Trial) -> None:
        other = _saved(trial.ada, _template(1, "another")).payload["hash"]
        bea = _person(trial.site, "bea")
        _match(trial.site, other, bea)
        assert shortlist.finishers(trial.site.store, trial.game) == set()


class TestWhichVotesCount:
    def test_a_finisher_with_a_week_old_account_counts(self, trial: Trial) -> None:
        trial.crowd(3, stars=4)
        found = trial.bar()
        assert (found.votes, found.mean) == (3, 4.0)

    def test_a_voter_who_never_finished_a_game_does_not(self, trial: Trial) -> None:
        bea = _person(trial.site, "bea")
        _vote(trial.site, trial.game, bea, 5)
        _match(trial.site, trial.game, bea, reason="resigned")
        assert trial.bar().votes == 0

    def test_the_designers_own_vote_does_not(self, trial: Trial) -> None:
        _match(trial.site, trial.game, trial.ada_id)
        _vote(trial.site, trial.game, trial.ada_id, 5)
        with trial.site.store.cursor() as cursor:
            cursor.execute("update users set created = now() - interval '60 days'")
        trial.site.store.commit()
        assert trial.bar().votes == 0

    def test_a_new_account_waits_a_week_and_a_shut_one_never_counts(
        self, trial: Trial
    ) -> None:
        new = _person(trial.site, "new", age=timedelta(days=6))
        shut = _person(trial.site, "shut")
        for person in (new, shut):
            _match(trial.site, trial.game, person)
            _vote(trial.site, trial.game, person, 5)
        with trial.site.store.cursor() as cursor:
            cursor.execute("update users set revoked = now() where id = %s", (shut,))
        trial.site.store.commit()
        assert trial.bar().votes == 0
        with trial.site.store.cursor() as cursor:
            cursor.execute(
                "update users set created = now() - interval '8 days' where id = %s", (new,)
            )
        trial.site.store.commit()
        assert trial.bar().votes == 1

    def test_the_mean_is_over_the_counted_votes_only(self, trial: Trial) -> None:
        trial.crowd(2, stars=5)
        bea = _person(trial.site, "bea")
        _vote(trial.site, trial.game, bea, 1)  # never played: shown, not counted
        found = trial.bar()
        assert (found.votes, found.mean) == (2, 5.0)

    def test_the_time_on_trial_is_counted_from_publishing(self, trial: Trial) -> None:
        assert trial.bar().days == 0
        _published(trial.site, trial.game, timedelta(days=30, hours=1))
        assert trial.bar().days == 30

    def test_a_game_with_no_current_reading_is_short_of_the_bar(self, trial: Trial) -> None:
        trial.eligible()
        assert trial.bar().eligible
        assert not trial.bar(reading=None).eligible
        assert not trial.bar(reading={"viable": False}).eligible


# ------------------------------------------------------------------------ the press


class TestNominating:
    def test_an_eligible_game_is_shortlisted_by_its_designer(self, trial: Trial) -> None:
        trial.eligible()
        done = shortlist.nominate(trial.site.store, trial.game, trial.ada_id, READING)
        trial.site.store.commit()
        assert not done.already and "shortlisted" in done.said
        assert trial.level() == stages.SHORTLISTED
        again = shortlist.nominate(trial.site.store, trial.game, trial.ada_id, READING)
        trial.site.store.commit()
        assert again.already

    def test_nobody_else_may_nominate_it(self, trial: Trial) -> None:
        trial.eligible()
        bea = _person(trial.site, "bea")
        with pytest.raises(shortlist.NominateError, match="Only the person who made"):
            shortlist.nominate(trial.site.store, trial.game, bea, READING)
        trial.site.store.rollback()
        assert trial.level() == stages.ON_TRIAL

    def test_a_game_short_of_the_bar_is_refused_naming_every_line(self, trial: Trial) -> None:
        trial.crowd(3)
        with pytest.raises(shortlist.NominateError) as refused:
            shortlist.nominate(trial.site.store, trial.game, trial.ada_id, None)
        trial.site.store.rollback()
        said = str(refused.value)
        for line in ("30 days", "3 of 8 other players", "3 of 8 votes", "score from your computer"):
            assert line in said
        assert trial.level() == stages.ON_TRIAL

    def test_a_draft_and_a_library_game_cannot_be_nominated(self, trial: Trial) -> None:
        draft = _saved(trial.ada, _template(1, "still a draft")).payload["hash"]
        with pytest.raises(shortlist.NominateError, match="Only a game on trial"):
            shortlist.nominate(trial.site.store, draft, trial.ada_id, READING)
        trial.site.store.rollback()
        with trial.site.store.cursor() as cursor:
            cursor.execute("update games set origin = 'archive' where hash = %s", (draft,))
        trial.site.store.commit()
        with pytest.raises(shortlist.NominateError, match="came with the site"):
            shortlist.nominate(trial.site.store, draft, trial.ada_id, READING)
        trial.site.store.rollback()

    def test_one_shortlisted_game_a_designer(self, trial: Trial) -> None:
        second = _saved(trial.ada, _template(1, "ada's second")).payload["hash"]
        with trial.site.store.cursor() as cursor:
            cursor.execute(
                "update games set visibility = 'public', published = now() - interval"
                " '40 days', nominated = now() where hash = %s",
                (second,),
            )
        trial.site.store.commit()
        trial.eligible()
        with pytest.raises(shortlist.NominateError, match="already have 1 game shortlisted"):
            shortlist.nominate(trial.site.store, trial.game, trial.ada_id, READING)
        trial.site.store.rollback()
        assert trial.level() == stages.ON_TRIAL

    def test_the_shortlist_holds_at_most_eight(self, trial: Trial) -> None:
        trial.eligible()
        with trial.site.store.cursor() as cursor:
            for index in range(quota.SHORTLIST_SIZE):
                owner = _person(trial.site, f"designer{index}")
                cursor.execute(
                    "insert into games (hash, spec, name, owner, visibility, origin,"
                    " published, nominated) values (%s, '{}'::jsonb, %s, %s, 'public',"
                    " 'designed', now() - interval '40 days', now())",
                    (f"{index:064x}", f"pick {index}", owner),
                )
        trial.site.store.commit()
        assert shortlist.shortlisted(trial.site.store) == quota.SHORTLIST_SIZE
        with pytest.raises(shortlist.NominateError, match="The shortlist is full"):
            shortlist.nominate(trial.site.store, trial.game, trial.ada_id, READING)
        trial.site.store.rollback()
        # A place opens when the round takes one off (a killed game holds none either).
        with trial.site.store.cursor() as cursor:
            cursor.execute("update games set revoked = now() where hash = %s", ("0" * 64,))
        trial.site.store.commit()
        shortlist.nominate(trial.site.store, trial.game, trial.ada_id, READING)
        trial.site.store.commit()
        assert trial.level() == stages.SHORTLISTED


# ------------------------------------------------------------------------ over HTTP


class TestTheRoute:
    def test_the_designer_nominates_through_the_route(
        self, trial: Trial, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial.eligible()
        monkeypatch.setattr(provisional, "counted", lambda connection, digest, *rest: READING)
        answer = trial.ada.post(f"/api/game/{trial.game}/nominate")
        assert answer.status == 200, answer.text
        assert answer.payload["level"] == stages.SHORTLISTED
        assert answer.payload["levelSays"] == stages.SAYS[stages.SHORTLISTED]
        assert answer.payload["shortlist"] is None
        assert trial.level() == stages.SHORTLISTED
        with trial.site.store.cursor() as cursor:
            cursor.execute("select nominated from games where hash = %s", (trial.game,))
            assert cursor.fetchone()[0] is not None
        trial.site.store.commit()

    def test_until_a_reading_attaches_the_route_refuses(self, trial: Trial) -> None:
        """No spot-checked reading, current or in its grace, so the bar's last line is unmet."""
        trial.eligible()
        answer = trial.ada.post(f"/api/game/{trial.game}/nominate")
        assert answer.status == 400
        assert "score from your computer" in answer.payload["error"]
        assert trial.level() == stages.ON_TRIAL

    def test_a_visitor_and_another_member_are_refused(
        self, trial: Trial, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial.eligible()
        monkeypatch.setattr(provisional, "counted", lambda connection, digest, *rest: READING)
        visitor = trial.site.browser().post(f"/api/game/{trial.game}/nominate")
        assert visitor.status == 400 and "Sign in" in visitor.payload["error"]
        bea = trial.site.signed_in("bea")
        other = bea.post(f"/api/game/{trial.game}/nominate")
        assert other.status == 400 and "Only the person" in other.payload["error"]
        assert trial.level() == stages.ON_TRIAL

    def test_a_killed_game_is_not_there(
        self, trial: Trial, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial.eligible()
        monkeypatch.setattr(provisional, "counted", lambda connection, digest, *rest: READING)
        with trial.site.store.cursor() as cursor:
            cursor.execute("update games set revoked = now() where hash = %s", (trial.game,))
        trial.site.store.commit()
        assert trial.ada.post(f"/api/game/{trial.game}/nominate").status == 404

    def test_the_game_page_carries_the_bar(
        self, trial: Trial, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial.crowd(5, stars=4)
        _published(trial.site, trial.game, timedelta(days=3))
        page = trial.ada.get(f"/api/game/{trial.game}").payload
        bar = page["shortlist"]
        assert bar["mine"] is True and bar["eligible"] is False and bar["nominable"] is False
        said = {check["key"]: check for check in bar["checks"]}
        assert said["finishers"]["says"] == "5 of 8 other players have finished a game on it"
        assert said["votes"]["says"] == "5 of 8 votes count so far"
        assert said["stars"]["met"] is True
        assert bar["says"].startswith("It can be shortlisted after ")
        # Somebody else reads the same counts, with no button.
        other = trial.site.browser().get(f"/api/game/{trial.game}").payload["shortlist"]
        assert other["mine"] is False
        theirs = {check["key"]: check for check in other["checks"]}
        assert {k: v for k, v in theirs.items() if k != "reading"} == {
            k: v for k, v in said.items() if k != "reading"
        }
        # The reading's line speaks to its reader (Q537): only the other player is told
        # that only its designer can measure it again.
        assert "only its designer" in theirs["reading"]["says"]
        assert "designer" not in said["reading"]["says"]

        trial.eligible()
        monkeypatch.setattr(provisional, "counted", lambda connection, digest, *rest: READING)
        ready = trial.ada.get(f"/api/game/{trial.game}").payload["shortlist"]
        assert ready["eligible"] is True and ready["nominable"] is True
        assert ready["waiting"] == ""

        trial.ada.post(f"/api/game/{trial.game}/nominate")
        after = trial.ada.get(f"/api/game/{trial.game}").payload
        assert after["level"] == stages.SHORTLISTED and after["shortlist"] is None

    def test_a_full_shortlist_disables_the_button_and_says_why(
        self, trial: Trial, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trial.eligible()
        monkeypatch.setattr(quota, "SHORTLIST_SIZE", 0)
        monkeypatch.setattr(provisional, "counted", lambda connection, digest, *rest: READING)
        bar = trial.ada.get(f"/api/game/{trial.game}").payload["shortlist"]
        assert bar["eligible"] is True and bar["nominable"] is False
        assert "The shortlist is full" in bar["waiting"]

    def test_the_designers_track_carries_the_bar_and_nominates(
        self, trial: Trial, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q518: `#/design`'s standing sends the game page's `shortlist.panel` too."""
        standing = f"/api/design/standing/{trial.game}"
        early = trial.ada.get(standing).payload
        assert early["level"] == stages.ON_TRIAL
        bar = early["shortlist"]
        assert bar["mine"] is True and bar["nominable"] is False
        assert bar["waiting"] == "It can be nominated once every line above holds."
        assert bar == trial.ada.get(f"/api/game/{trial.game}").payload["shortlist"]
        # Anybody else opening it in the designer gets the counts and no button.
        other = trial.site.signed_in("bea").get(standing).payload["shortlist"]
        assert other["mine"] is False and other["nominable"] is False

        trial.eligible()
        monkeypatch.setattr(provisional, "counted", lambda connection, digest, *rest: READING)
        ready = trial.ada.get(standing).payload["shortlist"]
        assert ready["eligible"] is True and ready["nominable"] is True
        assert ready["waiting"] == ""

        # The button's press is the game page's route; afterwards the track has moved.
        assert trial.ada.post(f"/api/game/{trial.game}/nominate").status == 200
        after = trial.ada.get(standing).payload
        assert after["level"] == stages.SHORTLISTED and after["shortlist"] is None
        assert [step["lit"] for step in after["track"]] == [False, False, True, False]

    def test_a_draft_in_the_designer_has_no_bar(self, trial: Trial) -> None:
        with trial.site.store.cursor() as cursor:
            cursor.execute(
                "update games set visibility = 'unlisted' where hash = %s", (trial.game,)
            )
        trial.site.store.commit()
        found = trial.ada.get(f"/api/design/standing/{trial.game}").payload
        assert found["level"] == stages.DRAFT
        assert found["shortlist"] is None

    def test_the_star_panel_says_whether_a_vote_counts(self, trial: Trial) -> None:
        def said(browser: Any) -> str:
            return str(browser.get(f"/api/game/{trial.game}").payload["standing"]["voteCounts"])

        assert said(trial.ada) == shortlist.OWN
        assert said(trial.site.browser()) == ""
        bea = trial.site.signed_in("bea")
        assert said(bea) == shortlist.NOT_YET
        with trial.site.store.cursor() as cursor:
            cursor.execute("select id from users where handle = 'bea'")
            bea_id = int(cursor.fetchone()[0])
        trial.site.store.commit()
        _match(trial.site, trial.game, bea_id)
        assert said(bea) == shortlist.TOO_NEW
        with trial.site.store.cursor() as cursor:
            cursor.execute(
                "update users set created = now() - interval '8 days' where id = %s",
                (bea_id,),
            )
        trial.site.store.commit()
        assert said(bea) == shortlist.COUNTS
        # Off a trial game there is no shortlist to count toward.
        with trial.site.store.cursor() as cursor:
            cursor.execute("update games set nominated = now() where hash = %s", (trial.game,))
        trial.site.store.commit()
        assert said(bea) == ""


# ------------------------------------------------------------------------ the page


class TestThePageDrawsWhatItIsSent:
    BAR = {
        "checks": [
            {"key": "days", "met": True, "says": "on trial since <3 Oct>"},
            {"key": "votes", "met": False, "says": "4 of 8 votes count so far"},
        ],
        "eligible": False,
        "says": "It can be shortlisted when all of these hold:",
        "mine": True,
        "nominable": False,
        "waiting": "It can be nominated once every line above holds.",
    }

    def _where(self, bar: dict[str, Any]) -> str:
        game = {"level": "Word", "track": [], "shortlist": bar}
        return " ".join(_in_page("", f"whereHtml({json.dumps(game)})").split())

    def test_the_lines_and_a_disabled_button_with_its_reason(self) -> None:
        drawn = self._where(self.BAR)
        assert "on trial since &lt;3 Oct&gt;" in drawn
        assert "4 of 8 votes count so far" in drawn
        assert 'data-met="yes"' in drawn and 'data-met="no"' in drawn
        assert 'id="nominateIt"' in drawn and "disabled" in drawn
        assert "once every line above holds" in drawn

    def test_somebody_else_sees_the_counts_and_no_button(self) -> None:
        drawn = self._where({**self.BAR, "mine": False, "waiting": ""})
        assert "4 of 8 votes count so far" in drawn
        assert "nominateIt" not in drawn

    def test_an_eligible_game_offers_its_designer_a_working_button(self) -> None:
        drawn = self._where({**self.BAR, "eligible": True, "nominable": True, "waiting": ""})
        button = drawn[drawn.index('id="nominateIt"'):]
        assert "disabled" not in button[: button.index(">")]

    def test_no_bar_draws_nothing(self) -> None:
        assert "shortlist" not in self._where(None)  # type: ignore[arg-type]

    def _track(self, bar: dict[str, Any] | None) -> str:
        standing = {"level": "Word", "track": [], "levelSays": "", "shortlist": bar}
        return " ".join(_in_page("", f"publishHtml({json.dumps(standing)})").split())

    def test_the_designers_track_draws_the_bar_and_the_button(self) -> None:
        """Q518: the designer's *Where this game stands* draws the game page's bar."""
        drawn = self._track({**self.BAR, "eligible": True, "nominable": True, "waiting": ""})
        assert "4 of 8 votes count so far" in drawn
        assert 'id="nominateIt"' in drawn
        button = drawn[drawn.index('id="nominateIt"'):]
        assert "disabled" not in button[: button.index(">")]
        # Inside the track's own panel, above where a press is answered.
        assert drawn.index('id="dPublish"') < drawn.index('id="nominateIt"')
        assert drawn.index('id="nominateIt"') < drawn.index('id="dPublishSays"')

    def test_the_designers_track_draws_no_bar_for_somebody_else_or_none(self) -> None:
        assert "shortlist" not in self._track({**self.BAR, "mine": False, "waiting": ""})
        assert "shortlist" not in self._track(None)

    def test_the_designers_button_presses_the_game_pages_route(self) -> None:
        page = _in_page("", "wireLadder.toString()")
        assert "document.querySelector('#dPublish #nominateIt')" in page
        assert "api('/api/game/' + hash + '/nominate', {})" in page

    def test_the_star_panel_carries_the_sentence(self) -> None:
        standing = {"signedIn": True, "votes": 0, "voteCounts": "Your stars <count>."}
        drawn = _in_page("", f"voteBody({json.dumps(standing)})")
        assert 'id="voteCounts"' in drawn and "Your stars &lt;count&gt;." in drawn
        assert "voteCounts" not in _in_page(
            "", f"voteBody({json.dumps({**standing, 'voteCounts': ''})})"
        )


# ------------------------------------------------------------------------ the checkout


class TestTheCheckoutsDesignerDrawsItsTrack:
    """Q503b's loose end: `candidacy.standing` now carries the level and the track."""

    def test_the_standing_carries_the_local_level(self, tmp_path: Path) -> None:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        (tmp_path / "rulesets" / "dull.json").write_text(json.dumps(DULL), encoding="utf-8")
        rulesets, runs, archive = (tmp_path / name for name in ("rulesets", "runs", "archive"))
        games = {
            game.hash: game for game in library.discover(rulesets, runs, archive_dir=archive)
        }
        found = candidacy.standing(games, DULL_HASH, runs, rulesets, archive)
        assert found["level"] == stages.BASELINE
        assert found["levelSays"] == stages.SAYS[stages.BASELINE]
        assert found["track"][-1] == {"word": stages.BASELINE, "lit": True}

        draft = candidacy.standing(games, "f" * 64, runs, rulesets, archive)
        assert draft["level"] == stages.DRAFT
        assert [step["lit"] for step in draft["track"]] == [True, False, False, False]

    def test_the_designer_draws_it(self) -> None:
        standing = {
            "level": stages.DRAFT, "levelSays": stages.SAYS[stages.DRAFT],
            "track": stages.track(stages.DRAFT),
        }
        drawn = " ".join(_in_page("", f"publishHtml({json.dumps(standing)})").split())
        assert f"Where this game stands: {stages.DRAFT}" in drawn
        assert 'class="track"' in drawn
