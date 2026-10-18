"""A member's game gets its leaderboard when it is published (L5, `site/spine.py`).

`docs/archive/plan-leaderboard.md` §2.6 step 1 and §4's L5 row, with Q1 decided on 2026-10-10 (*A,
as soon as it is published*). What is held, and where:

* the spine's ladder is the one `web.opponents` makes for any game -- shared, not forked --
  a pure function of the game's hash and its seeds at any thread count, and the child
  process hands back exactly it -- `TestTheLadder`, `TestTheChild`;
* the child caps itself (CPU-seconds, memory, `nice 19`) where the platform can, and the
  parent's wall clock refuses a game too slow for it -- `TestTheChild`, `TestTheCaps`;
* publishing files the job, the runner rates the five standard players and seeds the
  table, a rating row for each of them, centred on 1500, with the ladder kept under
  `runs/opponents/`; rated play opens against them, and the stronger ones stay practice --
  `TestPublishSeedsTheTable`;
* re-publishing a game does not rate it again, nor does a game a cap refused get retried;
  a game published before L5 is filed when the runner starts; checks of people's
  measurements go first -- `TestOnceAGame`;
* a game too slow for the cap is refused, gets no rating, waits for our own machine, and
  its page says so -- `TestTheCaps`;
* the page's sentences for each state -- `TestThePage`.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it (a skipped
test proves nothing: count them). The draft is the designer's first template, whose spine
costs about a second.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_admin import (  # noqa: F401 -- `drafting` is a fixture
    Site,
    _saved,
    _screened,
    _template,
    _user_id,
    drafting,
)
from test_site_leaderboard import _drawn

from chess_variant_sandbox.site import (
    drafts,
    leaderboard,
    matches,
    provisional,
    rating,
    spine,
    terms,
)
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import opponents
from chess_variant_sandbox.web.designing import settled

REPOSITORY = Path(__file__).resolve().parents[1]
DULL = load_spec(REPOSITORY / "rulesets" / "dull.json")
DULL_HASH = spec_hash(DULL)
STAND_IN = site_store.STAND_IN_READING


def _without_dates(table: dict[str, Any]) -> dict[str, Any]:
    """A ladder less the two stamps that name when and on what build it was rated."""
    return {key: value for key, value in table.items() if key not in ("rated", "built")}


# ------------------------------------------------------------------------ the ladder


class TestTheLadder:
    def test_it_is_the_five_standard_players_at_the_ladders_games_and_seed(self) -> None:
        table, records = opponents.rate_spine(DULL, DULL_HASH)
        assert [entry.agent for entry in opponents.spine_roster()] == list(opponents.SPINE)
        assert table["roster"] == table["spine"] == list(opponents.SPINE)
        assert sorted(row["name"] for row in table["standings"]) == sorted(opponents.SPINE)
        assert table["format"] == opponents.LADDER_FORMAT and table["top_ups"] == []
        assert (table["games_per_pairing"], table["seed"]) == (2, 0)
        # Every ordered pair of five, twice: 40 games, seeds from 0, none repeated.
        assert table["games"] == len(records) == 40
        assert sorted(record.seed for record in records) == list(range(40))
        labels = {row["name"]: row["label"] for row in table["standings"]}
        assert labels["minimax:nodes=400"].startswith("easy")

    def test_it_is_a_pure_function_of_the_hash_and_the_seeds_at_any_thread_count(
        self,
    ) -> None:
        one, games_one = opponents.rate_spine(DULL, DULL_HASH, workers=1)
        many, games_many = opponents.rate_spine(DULL, DULL_HASH, workers=4)
        assert _without_dates(one) == _without_dates(many)
        assert opponents.game_lines(games_one) == opponents.game_lines(games_many)

    def test_it_is_the_ladder_the_rate_button_makes_for_the_same_five(
        self, tmp_path: Path
    ) -> None:
        """Shared, not forked: `Ratings` over a roster of the spine writes the same table and
        the same games the box's child hands back."""
        run = opponents.Ratings(tmp_path).start(
            DULL, DULL_HASH, "dull", opponents.spine_roster()
        )
        while run.state == "running":
            import time

            time.sleep(0.05)
        assert run.state == "finished", run.error
        table, records = opponents.rate_spine(DULL, DULL_HASH)
        written = json.loads(
            opponents.stored_path(DULL_HASH, tmp_path).read_text(encoding="utf-8"))
        assert _without_dates(written) == _without_dates(table)
        kept = opponents.read_games(DULL_HASH, tmp_path)
        assert kept is not None
        assert opponents.game_lines(kept) == opponents.game_lines(records)

    def test_the_ladders_script_plays_the_same_games_and_seed(self) -> None:
        path = REPOSITORY / "deploy" / "ladders.py"
        found = importlib.util.spec_from_file_location("ladders_for_spine", path)
        assert found is not None and found.loader is not None
        module = importlib.util.module_from_spec(found)
        found.loader.exec_module(module)
        assert module.GAMES_PER_PAIRING == opponents.LADDER_GAMES
        assert module.SEED == opponents.LADDER_SEED


# ------------------------------------------------------------------------- the child


class TestTheChild:
    def test_the_child_hands_back_the_ladder_and_its_games(self) -> None:
        answer = spine.in_a_child(DULL, DULL_HASH, spine.caps())
        table, records = opponents.rate_spine(DULL, DULL_HASH)
        assert _without_dates(answer["table"]) == _without_dates(table)
        assert answer["games"] == opponents.game_lines(records)
        assert answer["cpu"] > 0

    def test_it_caps_itself_where_the_platform_can(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`RLIMIT_CPU` at the cap with five seconds' grace, `RLIMIT_DATA` at the memory
        cap, and `nice 19`: asked of a stand-in `resource`, so it holds on Windows too."""
        asked: list[tuple[Any, ...]] = []
        stand_in = types.SimpleNamespace(
            RLIMIT_CPU="cpu", RLIMIT_DATA="data",
            setrlimit=lambda which, limits: asked.append((which, limits)),
        )
        monkeypatch.setitem(sys.modules, "resource", stand_in)
        niced: list[tuple[int, int, int]] = []
        monkeypatch.setattr(spine.os, "setpriority",
                            lambda which, who, value: niced.append((which, who, value)),
                            raising=False)
        applied = spine.limit_self(300.0, 512 * 1024 * 1024)
        assert applied == ["cpu", "memory", "nice"]
        assert asked == [("cpu", (300, 305)), ("data", (512 * 1024 * 1024,) * 2)]
        assert [value for _, _, value in niced] == [19]

    def test_the_caps_are_the_plans(self) -> None:
        limits = spine.caps()
        assert limits.cpu == 300.0, "§2.6: 300 CPU-s"
        assert limits.wall >= 2 * limits.cpu
        assert 0 < limits.memory <= 1024 * 1024 * 1024, "under the unit's MemoryHigh"

    def test_how_a_childs_ending_is_read(self) -> None:
        assert spine._out_of_memory(1, "Traceback ... MemoryError")
        assert spine._out_of_memory(-6, "memory allocation of 8388608 bytes failed")
        assert not spine._out_of_memory(1, "ValueError: no")
        if hasattr(signal, "SIGXCPU"):
            assert spine._out_of_time(-int(signal.SIGXCPU))
        assert not spine._out_of_time(1) and not spine._out_of_time(None)

    def test_a_child_past_the_clock_is_over_time(self) -> None:
        with pytest.raises(spine.OverCap) as over:
            spine.in_a_child(DULL, DULL_HASH, spine.Caps(cpu=300.0, wall=0.01, memory=0))
        assert over.value.reason == spine.OVER_TIME
        assert "took longer than this site allows" in over.value.said


# ---------------------------------------------------------------- over the store


def _published(site: Site, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str, Any]:
    """ada's draft, saved, read and published through the route: the browser, the hash, the
    answer. The reading is a stand-in, as `test_site_terms` hands one in."""
    _screened(monkeypatch, passed=True)
    ada = site.signed_in("ada")
    digest = _saved(ada, _template()).payload["hash"]
    monkeypatch.setattr(drafts, "attached", lambda connection, digest, *rest: STAND_IN)
    answer = ada.post("/api/design/publish", {"hash": digest, "terms": terms.VERSION})
    assert answer.status == 200, answer.payload
    return ada, digest, answer


def _jobs(site: Site, digest: str) -> list[tuple[Any, ...]]:
    with site.store.cursor() as cursor:
        cursor.execute(
            "select state, requester, error, log, cpu_seconds from jobs"
            " where kind = %s and hash = %s order by id",
            (spine.KIND, digest),
        )
        found = cursor.fetchall()
    site.store.commit()
    return found


def _table(site: Site, digest: str) -> dict[rating.PlayerKey, rating.Player]:
    found = rating.load(site.store, digest)
    site.store.commit()
    return found


def _board(browser: Any, digest: str) -> dict[str, Any]:
    answer = browser.get(f"/api/game/{digest}")
    assert answer.status == 200, answer.text
    return dict(answer.payload["standing"]["leaderboard"])


class TestPublishSeedsTheTable:
    def test_publishing_files_the_rating_and_says_so(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _, digest, answer = _published(drafting, monkeypatch)
        assert answer.payload["listed"] is True
        assert answer.payload["said"].endswith(drafts.RATING_NOW)
        [(state, requester, *_)] = _jobs(drafting, digest)
        assert state == spine.QUEUED
        assert requester is None, "the site's own work, nobody's day of checking"

    def test_the_runner_rates_the_five_and_seeds_the_table(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        ada, digest, _ = _published(drafting, monkeypatch)
        board = _board(ada, digest)
        assert board["rated"] is False and board["box"] == spine.RATING

        assert drafting.app.checks.drain() == [], "the checks' answer holds no spine"

        [(state, _, error, log, cpu)] = _jobs(drafting, digest)
        assert state == spine.DONE and error is None
        assert log.startswith("rated 5 players in 40 games") and cpu > 0
        table = _table(drafting, digest)
        computers = rating.computers(table)
        # A rating row for each spine player, and nobody else's.
        assert sorted(player.key.agent for player in computers) == sorted(opponents.SPINE)
        # The first seeding centres on the spine (§2.4): their mean is 1500.
        mean = sum(player.rating for player in computers) / len(computers)
        assert mean == pytest.approx(1500.0, abs=1e-6)
        # The ladder is kept as any ladder is, for the rest of the picker's top-up (L6).
        kept = json.loads(opponents.stored_path(digest, drafting.app.runs_dir)
                          .read_text(encoding="utf-8"))
        assert kept["spine"] == list(opponents.SPINE) and kept["games"] == 40
        assert len(opponents.read_games(digest, drafting.app.runs_dir) or ()) == 40
        assert kept["roster"] == list(opponents.SPINE)

        board = _board(ada, digest)
        assert board["rated"] is True and board["box"] == spine.RATED
        assert len(board["rows"]) == 5 and board["people"] == 0
        assert "hardest" in board["unrated"], "the stronger ones join later (L6)"

    def test_rated_play_opens_against_the_five_and_not_the_rest(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _, digest, _ = _published(drafting, monkeypatch)
        drafting.app.checks.drain()
        bo = _user_id(drafting, "ada")
        started = matches.start(
            drafting.store, digest, matches.LEADERBOARD, ["human", "minimax:nodes=400"], bo)
        assert started is not None
        drafting.store.rollback()
        with pytest.raises(matches.MatchError, match="has no rating on"):
            matches.start(drafting.store, digest, matches.LEADERBOARD,
                          ["human", "minimax:nodes=200000"], bo)
        drafting.store.rollback()

    def test_the_ladder_the_box_seeds_from_is_the_one_this_machine_makes(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _, digest, _ = _published(drafting, monkeypatch)
        drafting.app.checks.drain()
        kept = json.loads(opponents.stored_path(digest, drafting.app.runs_dir)
                          .read_text(encoding="utf-8"))
        spec = settled(_template())
        assert spec_hash(spec) == digest
        here, _ = opponents.rate_spine(spec, digest, workers=0)
        assert _without_dates(kept) == _without_dates(here)


class TestOnceAGame:
    def test_publishing_again_does_not_rate_again(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        """A resting game that comes back keeps its table and files nothing."""
        ada, digest, _ = _published(drafting, monkeypatch)
        drafting.app.checks.drain()
        before = _table(drafting, digest)
        with drafting.store.cursor() as cursor:
            # Rested after its 90 days, so the day's publish cap has long since rolled.
            cursor.execute("update games set rested = now(),"
                           " published = now() - interval '91 days' where hash = %s",
                           (digest,))
        drafting.store.commit()

        again = ada.post("/api/design/publish", {"hash": digest, "terms": terms.VERSION})

        assert again.status == 200, again.payload
        assert drafts.RATING_NOW not in again.payload["said"]
        assert len(_jobs(drafting, digest)) == 1
        drafting.app.checks.drain()
        assert _table(drafting, digest) == before
        assert spine.file(drafting.store, digest) is None
        drafting.store.rollback()

    def test_a_game_published_before_l5_is_filed_when_the_runner_starts(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        draft = _saved(ada, _template(1, "a draft nobody published")).payload["hash"]
        with drafting.store.cursor() as cursor:
            cursor.execute("update games set visibility = 'public', published = now()"
                           " where hash = %s", (digest,))
        drafting.store.commit()
        assert spine.owed(drafting.store) == [digest], "a draft is owed nothing"
        assert spine.state(drafting.store, draft) == spine.DRAFT
        [job] = spine.file_owed(drafting.store)
        drafting.store.commit()
        assert spine.owed(drafting.store) == [] and spine.file_owed(drafting.store) == []
        drafting.store.commit()
        assert [row[0] for row in _jobs(drafting, digest)] == [spine.QUEUED]
        assert job > 0

    def test_a_library_game_is_never_filed(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        digest = drafting.game()
        assert spine.file(drafting.store, digest) is None
        assert spine.state(drafting.store, digest) is None
        drafting.store.rollback()

    def test_the_checks_of_peoples_measurements_go_first(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _, digest, _ = _published(drafting, monkeypatch)
        with drafting.store.cursor() as cursor:
            cursor.execute("insert into jobs (kind, state) values (%s, 'queued')",
                           (provisional.WALK,))
        drafting.store.commit()
        assert drafting.app.checks.rate() is None
        assert [row[0] for row in _jobs(drafting, digest)] == [spine.QUEUED]

    def test_a_stopped_process_puts_its_rating_back(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _, digest, _ = _published(drafting, monkeypatch)
        assert spine.claim(drafting.store) is not None
        drafting.store.commit()
        assert drafting.app.checks.requeue() == 1
        assert [row[0] for row in _jobs(drafting, digest)] == [spine.QUEUED]

    def test_turned_off_the_runner_rates_nothing(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        _, digest, _ = _published(drafting, monkeypatch)
        drafting.app.checks.rates = False
        drafting.app.checks.drain()
        assert [row[0] for row in _jobs(drafting, digest)] == [spine.QUEUED]


class TestTheCaps:
    def test_a_game_too_slow_is_refused_waits_and_its_page_says_so(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        ada, digest, _ = _published(drafting, monkeypatch)
        drafting.app.checks.spine_caps = spine.Caps(cpu=300.0, wall=0.01, memory=0)

        drafting.app.checks.drain()

        [(state, _, error, log, _)] = _jobs(drafting, digest)
        assert state == spine.REFUSED and log is None
        assert error.startswith(spine.OVER_TIME + ": rating the five standard players took")
        assert rating.computers(_table(drafting, digest)) == ()
        assert not opponents.stored_path(digest, drafting.app.runs_dir).exists()
        board = _board(ada, digest)
        assert board["rated"] is False and board["box"] == spine.SLOW
        html = _drawn(board)
        assert "took longer than this site allows" in html
        assert "waiting to be rated on our own machine" in html
        # Waiting for our own machine, not for the box again: nothing more is filed.
        drafting.app.checks.spine_caps = None
        assert spine.file(drafting.store, digest) is None
        drafting.store.rollback()
        drafting.app.checks.drain()
        assert len(_jobs(drafting, digest)) == 1

    def test_a_child_that_fails_otherwise_is_failed_not_refused(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        ada, digest, _ = _published(drafting, monkeypatch)

        def broken(*_: Any) -> dict[str, Any]:
            raise RuntimeError("the rating's child failed: boom")

        monkeypatch.setattr(spine, "in_a_child", broken)
        drafting.app.checks.drain()
        [(state, _, error, _, _)] = _jobs(drafting, digest)
        assert state == spine.FAILED and "boom" in error
        assert _board(ada, digest)["box"] == spine.WAITING


# ------------------------------------------------------------------------- the page


def _shown(box: str | None, rated: bool, library: bool = False) -> dict[str, Any]:
    roster = opponents.ladder()
    table = (
        {rating.computer(agent): rating.Player(rating.computer(agent), 1500.0, 60.0, 0.0, 0)
         for agent in opponents.SPINE}
        if rated else {}
    )
    return leaderboard.shown(table, named={}, roster=roster, viewer=None, library=library,
                             box=box)


class TestThePage:
    def test_a_draft_says_its_players_are_rated_when_it_is_published(self) -> None:
        html = _drawn(_shown(spine.DRAFT, rated=False))
        assert ("every game here is practice for now. They are rated on this site when the"
                " game is published.") in html

    def test_while_the_five_are_rated_it_says_minutes(self) -> None:
        html = _drawn(_shown(spine.RATING, rated=False))
        assert ("The five standard ones are being rated on this site now; this usually"
                " takes a few minutes.") in html
        assert "usually takes a day" not in html

    def test_a_refused_game_says_it_waits_for_our_own_machine(self) -> None:
        for box, why in ((spine.SLOW, "took longer than"),
                         (spine.LARGE, "needed more memory than"),
                         (spine.WAITING, "could not be rated on this site")):
            html = _drawn(_shown(box, rated=False))
            assert why in html and "waiting to be rated on our own machine" in html, box
            assert "minutes" not in html and "a day" not in html

    def test_rated_here_the_rest_join_when_our_machine_has_rated_them(self) -> None:
        shown = _shown(spine.RATED, rated=True)
        assert shown["unrated"][:1] == ["hardest"]
        html = _drawn(shown)
        assert ("The five standard computer players were rated on this site when the game"
                " was published. <b>hardest</b>, <b>tree search</b> join this table when our"
                " own machine has rated them; until then a game against them is practice."
                ) in html
        assert "have no rating on this game yet" not in html

    def test_a_library_game_keeps_its_own_words(self) -> None:
        html = _drawn(_shown(None, rated=False, library=True))
        assert "They are rated on our own machine; this usually takes a day." in html
        rated = _drawn(_shown(None, rated=True, library=True))
        assert "have no rating on this game yet, so they are not on it." in rated
        assert "rated on this site" not in rated
