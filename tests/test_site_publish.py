"""Tier two, the spot check, and the publish it opens (Q503e).

`docs/plan-client-compute.md`'s row *Tier two, the sample job* and `docs/plan-game-pipeline.md`
Q503e. The acceptance, and what holds each part:

* the sample is the same when drawn twice and differs when one byte of the submission
  differs -- `TestTheDraw`, which needs no database;
* an honest submission gives `spot-checked`, with the sample's size in its stamp, and the
  number then shows to everybody -- `TestTheSpotCheck`;
* one altered move in a sampled game gives `mismatch`, logged with the game and the ply and
  both build fingerprints, and so does an altered search counter sent through the route,
  which tier one cannot see -- `TestAMismatch`;
* a mismatch flags the account, which then cannot publish and whose next submission is
  walked but never spot-checked, until an admin clears the flag -- `TestTheFlag`;
* a check past its budget stops and leaves the result `legal`, and a person's day of
  checking is priced in CPU-seconds -- `TestTheCaps`;
* a draft measured in the browser, walked and spot-checked, **publishes through the route**,
  and nothing short of that does -- `TestASuccessfulPublish`;
* a spot-checked reading made stale by the site still counts toward the shortlist for 30
  days, then stops -- `TestTheShortlistsGrace`.

The draft is `dull` on `test_site_legality.py`'s small roster, measured once as a browser
measures (every unit through B0's request text). The roster has fewer than 32 tournament
games, so every one is in the sample and an altered game is always found; the draw's own
sizes are `TestTheDraw`'s. The store tests need a real Postgres and skip without one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_legality import Measured, lines, measure_as_a_browser, small_roster

from chess_variant_sandbox import measurement
from chess_variant_sandbox.measurement import LEGAL, Basis, sample_check
from chess_variant_sandbox.site import db, drafts, legality, provisional, quota, spotcheck, terms
from chess_variant_sandbox.site.provisional import SPOT_CHECKED
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.tournament import slices

REPOSITORY = Path(__file__).resolve().parents[1]
DATABASE = "no database to spot-check in"


# ------------------------------------------------------------------------ the draw


class TestTheDraw:
    KEY = b"k" * 32

    def test_the_same_text_draws_the_same_sample(self) -> None:
        text = '{"schema":1,"phases":{}}'
        one, two = spotcheck.seal(text, self.KEY), spotcheck.seal(text, self.KEY)
        assert one == two and len(one) == 64
        sizes = {"games": 256, "picks": list(range(0, 240, 5)), "fit": 48, "accept": 48}
        assert spotcheck.draw(one, **sizes) == spotcheck.draw(two, **sizes)

    def test_one_byte_more_or_changed_draws_another(self) -> None:
        text = '{"schema":1,"phases":{"tournament":"{\\"nodes\\":400}"}}'
        changed = text.replace("400", "401")
        sizes = {"games": 256, "picks": list(range(0, 240, 5)), "fit": 48, "accept": 48}
        drawn = spotcheck.draw(spotcheck.seal(text, self.KEY), **sizes)
        assert spotcheck.draw(spotcheck.seal(changed, self.KEY), **sizes) != drawn
        assert spotcheck.draw(spotcheck.seal(text + " ", self.KEY), **sizes) != drawn
        # Under the key the server draws and forgets, nobody can name the sample ahead.
        assert spotcheck.seal(text) != spotcheck.seal(text)

    def test_its_sizes_are_the_plans(self) -> None:
        sealed = spotcheck.seal("anything", self.KEY)
        picks = list(range(3, 243, 5))
        drawn = spotcheck.draw(sealed, games=256, picks=picks, fit=48, accept=48)
        assert len(drawn.games) == spotcheck.GAMES == 32
        assert len(set(drawn.games)) == 32 and all(0 <= g < 256 for g in drawn.games)
        assert len(drawn.trials) == spotcheck.TRIALS == 6 and set(drawn.trials) <= set(picks)
        assert (len(drawn.fit), len(drawn.accept)) == (6, 6)
        few = spotcheck.draw(sealed, games=12, picks=[1, 2], fit=5, accept=0)
        assert few.games == tuple(range(12)) and few.trials == (1, 2)
        assert (len(few.fit), few.accept) == (1, ())


# --------------------------------------------------------------------- the store


@pytest.fixture(scope="module")
def honest(tmp_path_factory: pytest.TempPathFactory) -> Measured:
    """Dull, measured once as a browser would, on the small roster."""
    spec = load_spec(REPOSITORY / "rulesets" / "dull.json")
    return measure_as_a_browser(spec, small_roster(), tmp_path_factory.mktemp("browser"))


@dataclass
class Draft:
    """A draft saved on the site by ada, the site over it, and what she measured."""

    app: Any
    connection: Any
    runs: Path
    game: str
    ada: int
    bo: int
    admin: int
    measured: Measured

    def basis(self, game: str, spec: Any) -> Basis | None:
        if not isinstance(spec, dict):
            return None
        assert spec_hash(spec) == game
        return self.measured.current

    def as_(self, viewer: int | None) -> Any:
        self.app.viewer = viewer
        self.app._request.role = "admin" if viewer == self.admin else "member"
        return self.app

    def send(self, upload: str | None = None) -> int:
        answer = self.as_(self.ada).post(
            "/api/provisional",
            {"game": self.game, "upload": self.measured.upload() if upload is None else upload},
        )
        assert answer["state"] == legality.QUEUED, answer
        return int(answer["job"])

    def status(self, job: int) -> dict[str, Any]:
        return dict(self.as_(self.ada).get(f"/api/provisional/{job}"))

    def query(self, sql: str, *args: Any) -> list[tuple[Any, ...]]:
        self.connection.rollback()
        with self.connection.cursor() as cursor:
            cursor.execute(sql, args)
            rows = cursor.fetchall()
        self.connection.commit()
        return [tuple(row) for row in rows]

    def results(self) -> list[tuple[int, str]]:
        return [(int(r[0]), str(r[1])) for r in self.query(
            "select id, state from provisional order by id")]

    def publish(self, viewer: int | None = None) -> Any:
        return self.as_(self.ada if viewer is None else viewer).post(
            "/api/design/publish", {"hash": self.game, "terms": terms.VERSION}
        )


@pytest.fixture
def draft(tmp_path: Path, honest: Measured) -> Iterator[Draft]:
    url, schema = site_store.emptied(DATABASE)
    from chess_variant_sandbox.site.server import SiteApp

    connection = db.connect(url)
    digest = spec_hash(honest.spec)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
            ids = []
            for handle, role in (("ada", "member"), ("bo", "member"), ("max", "admin")):
                cursor.execute(
                    "insert into users (handle, email, role) values (%s, %s, %s) returning id",
                    (handle, f"{handle}@example.org", role),
                )
                ids.append(int(cursor.fetchone()[0]))
            # As a save leaves it (Q503a): unlisted, designed, the check passed.
            cursor.execute(
                "insert into games (hash, spec, name, owner, visibility, origin, screen_passed)"
                " values (%s, %s::jsonb, %s, %s, 'unlisted', 'designed', true)",
                (digest, json.dumps(honest.spec), honest.spec["name"], ids[0]),
            )
        connection.commit()
        for name in ("rulesets", "archive", "runs"):
            (tmp_path / name).mkdir()

        def open_one() -> Any:
            opened = db.connect(url)
            with opened.cursor() as cursor:
                cursor.execute(f"set search_path to {schema}")
            return opened

        app = SiteApp(
            tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive",
            connect=open_one,
        )
        made = Draft(app, connection, tmp_path / "runs", digest, *ids, honest)
        app.checks.basis = made.basis
        yield made
        app.checks.stop()
    finally:
        connection.rollback()
        connection.close()


def sampled(draft: Draft) -> int:
    """How many tournament games the small roster plays: all of them are in the sample."""
    count = len(draft.measured.rows(slices.TOURNAMENT))
    assert count < spotcheck.GAMES
    return count


class TestTheSpotCheck:
    def test_an_honest_submission_is_spot_checked_and_shows_to_everybody(
        self, draft: Draft
    ) -> None:
        job = draft.send()
        walked, checked = draft.app.checks.drain()
        assert walked.state == legality.DONE and walked.sample == checked.job
        assert isinstance(checked, spotcheck.Outcome)
        assert checked.state == spotcheck.DONE, checked.said
        result = checked.result
        assert result is not None and result.state == SPOT_CHECKED
        assert result.sample == sampled(draft)
        assert result.basis == replace(
            draft.measured.stamp, checked=sample_check(sampled(draft))
        )
        assert checked.said.startswith("spot-checked: ") and "equal" in checked.said
        # The job keeps tier one's number; the sample job names the result.
        assert result.job == job

        answer = draft.status(job)
        assert answer["spotCheck"] == {"state": "done"}
        assert answer["result"]["state"] == SPOT_CHECKED
        assert answer["result"]["index"] == draft.measured.index
        # Everybody who can see the game sees it now, labelled; nobody saw it before.
        shown = provisional.shown(draft.connection, [draft.game], None, draft.basis)
        assert shown[draft.game]["state"] == SPOT_CHECKED
        assert shown[draft.game]["yours"] is False

        # Both tiers' processor time is on their jobs, which a person's day sums.
        spent = draft.query(
            "select kind, cpu_seconds, sample from jobs order by id"
        )
        assert [row[0] for row in spent] == [legality.KIND, spotcheck.KIND]
        assert all(row[1] is not None and row[1] > 0 for row in spent)
        assert spent[0][2] is None and len(spent[1][2]) == 64

    def test_while_it_waits_the_page_is_told_to_keep_asking(self, draft: Draft) -> None:
        job = draft.send()
        with draft.app._connection() as connection:
            first = legality.claim(connection)
        with draft.app._connection() as connection:
            legality.run(connection, draft.runs, int(first or 0), basis=draft.basis)
        answer = draft.status(job)
        assert answer["result"]["state"] == LEGAL
        assert answer["spotCheck"] == {"state": spotcheck.QUEUED}

    def test_a_result_superseded_before_its_check_is_left_alone(self, draft: Draft) -> None:
        draft.send()
        with draft.app._connection() as connection:
            legality.run(connection, draft.runs, int(legality.claim(connection) or 0),
                         basis=draft.basis)
        # A second send while the first check waits: it is walked first, and the first
        # check then finds nothing to replay.
        draft.send()
        done = draft.app.checks.drain()
        said = [getattr(one, "said", "") for one in done]
        assert any(text.startswith("nothing to check: the result is superseded")
                   for text in said), said
        assert [state for _, state in draft.results()] == ["superseded", SPOT_CHECKED]


class TestAMismatch:
    def _walked(self, draft: Draft) -> tuple[int, Path]:
        """Tier one passed and the check filed, not yet run: the records it will read."""
        job = draft.send()
        with draft.app._connection() as connection:
            legality.run(connection, draft.runs, int(legality.claim(connection) or 0),
                         basis=draft.basis)
        [(number, _)] = draft.results()
        return job, draft.runs / "provisional" / str(number)

    def test_one_altered_move_in_a_sampled_game_is_a_mismatch_named_by_game_and_ply(
        self, draft: Draft
    ) -> None:
        job, where = self._walked(draft)
        rows = [json.loads(line) for line in (where / "games.jsonl").read_text().splitlines()]
        game = next(i for i, row in enumerate(rows) if len(row["moves"]) > 6)
        ply = len(rows[game]["moves"]) // 2
        rows[game]["moves"][ply] = rows[game]["moves"][ply] + 1
        (where / "games.jsonl").write_text(lines(rows), encoding="utf-8")

        [checked] = draft.app.checks.drain()
        assert checked.state == spotcheck.REFUSED
        seed = rows[game]["seed"]
        assert f"the tournament's game {game} (seed {seed}) parts from the site's replay at ply {ply}" in checked.said
        build = measurement.built_fingerprint()
        assert f"the site's build {build}, the score's build {build}" in checked.said
        [(error,)] = draft.query("select error from jobs where kind = %s", spotcheck.KIND)
        assert error == f"{spotcheck.MISMATCH}: {checked.said}"

        # Never shown again, its records kept for the log, and the page told why.
        assert draft.results()[0][1] == provisional.MISMATCH
        assert where.is_dir()
        assert provisional.shown(draft.connection, [draft.game], draft.ada, draft.basis) == {}
        answer = draft.status(job)
        assert answer["superseded"] == provisional.MISMATCH
        assert answer["spotCheck"] == {"state": spotcheck.MISMATCH}

    def test_an_altered_search_counter_passes_tier_one_and_not_tier_two(
        self, draft: Draft
    ) -> None:
        """What tier one cannot prove (`legality`'s *What stays unproven*), sent through
        the route as a forger would send it."""
        rows = draft.measured.rows(slices.TOURNAMENT)
        game = next(i for i, row in enumerate(rows) if row.get("nodes"))
        rows[game]["nodes"] = [n + 1 for n in rows[game]["nodes"]] if isinstance(
            rows[game]["nodes"], list) else rows[game]["nodes"] + 1
        draft.send(draft.measured.upload(tournament=lines(rows)))
        walked, checked = draft.app.checks.drain()
        assert walked.state == legality.DONE
        assert checked.state == spotcheck.REFUSED
        assert f"the tournament's game {game} " in checked.said
        assert "differs from the site's replay in `nodes`" in checked.said


class TestTheFlag:
    def _mismatched(self, draft: Draft) -> None:
        rows = draft.measured.rows(slices.TOURNAMENT)
        game = next(i for i, row in enumerate(rows) if row.get("nodes"))
        rows[game]["nodes"] = [n + 1 for n in rows[game]["nodes"]] if isinstance(
            rows[game]["nodes"], list) else rows[game]["nodes"] + 1
        draft.send(draft.measured.upload(tournament=lines(rows)))
        draft.app.checks.drain()

    def test_a_mismatch_flags_the_account_which_cannot_publish(self, draft: Draft) -> None:
        self._mismatched(draft)
        assert provisional.flagged(draft.connection, draft.ada)
        assert not provisional.flagged(draft.connection, draft.bo)
        with pytest.raises(drafts.PublishError) as refused:
            draft.publish()
        assert str(refused.value) == provisional.FLAGGED
        panel = drafts.standing(draft.connection, draft.game, draft.ada, draft.basis)
        assert panel is not None
        assert (panel["open"], panel["waiting"]) == (False, provisional.FLAGGED)
        draft.connection.commit()

    def test_its_next_submission_is_walked_and_never_spot_checked(self, draft: Draft) -> None:
        self._mismatched(draft)
        job = draft.send()
        [walked] = draft.app.checks.drain()
        assert walked.state == legality.DONE and walked.sample is None
        [(log,)] = draft.query("select log from jobs where id = %s", job)
        assert log.endswith("not spot-checked: the account is flagged")
        answer = draft.status(job)
        assert answer["result"]["state"] == LEGAL and answer["spotCheck"] is None
        with pytest.raises(drafts.PublishError, match="did not match"):
            draft.publish()

    def test_an_admin_sees_it_and_clears_it(self, draft: Draft) -> None:
        self._mismatched(draft)
        view = draft.as_(draft.admin).get("/api/admin")
        [row] = view["flagged"]
        assert (row["user"], row["handle"]) == (draft.ada, "ada")
        assert row["found"].startswith(f"{spotcheck.MISMATCH}: the tournament's game ")
        said = draft.as_(draft.admin).post("/api/admin", {"act": "clear-flag", "user": draft.ada})
        assert "cleared" in said["said"]
        assert not provisional.flagged(draft.connection, draft.ada)
        assert draft.as_(draft.admin).get("/api/admin")["flagged"] == []
        # Cleared, the next honest reading is spot-checked and opens *Publish*.
        draft.send()
        draft.app.checks.drain()
        assert draft.publish()["listed"] is True

    def test_the_admin_page_lists_them_with_the_clearing_button(self) -> None:
        """Q730. No database: the page has the table, and its script draws the payload's
        `flagged` rows as text and posts `clear-flag` for the row's account."""
        from chess_variant_sandbox.site import pages

        page = pages.admin("me")
        for mark in ('id="flagged"', 'id="flagged-empty"', 'id="flagged-note"'):
            assert mark in page
        script = (REPOSITORY / "chess_variant_sandbox/site/static/admin.js").read_text(
            encoding="utf-8")
        assert "flagged(payload.flagged || []);" in script
        assert "act({act: 'clear-flag', user: one.user}, 'flagged-note')" in script
        assert "innerHTML" not in script


class TestTheCaps:
    def test_a_check_past_its_budget_stops_and_the_result_stays_legal(
        self, draft: Draft
    ) -> None:
        draft.app.checks.cpu_cap = 0.0
        job = draft.send()
        walked, checked = draft.app.checks.drain()
        assert checked.state == spotcheck.DONE and checked.said.startswith(spotcheck.STOPPED)
        assert draft.results()[0][1] == LEGAL
        assert not provisional.flagged(draft.connection, draft.ada)
        assert draft.status(job)["spotCheck"] == {"state": "stopped"}
        with pytest.raises(drafts.PublishError) as refused:
            draft.publish()
        assert str(refused.value) == drafts.NEEDS_A_READING

    def test_the_wall_clock_cap_stops_it_too(self, draft: Draft) -> None:
        draft.app.checks.wall_cap = 0.0
        draft.send()
        _, checked = draft.app.checks.drain()
        assert checked.said.startswith(spotcheck.STOPPED)

    def test_a_spent_day_refuses_the_next_submission(self, draft: Draft) -> None:
        with draft.connection.cursor() as cursor:
            cursor.execute(
                "insert into jobs (kind, hash, requester, state, cpu_seconds)"
                " values (%s, %s, %s, 'done', %s)",
                (spotcheck.KIND, draft.game, draft.ada, quota.CHECK_CPU_A_DAY),
            )
        draft.connection.commit()
        assert quota.checked_today(draft.connection, draft.ada) == quota.CHECK_CPU_A_DAY
        with pytest.raises(quota.QuotaError, match="send this one again tomorrow"):
            draft.send()
        # A day later the same seconds no longer count.
        with draft.connection.cursor() as cursor:
            cursor.execute("update jobs set created = now() - interval '25 hours'")
        draft.connection.commit()
        assert quota.checked_today(draft.connection, draft.ada) == 0
        draft.send()

    def test_the_caps_are_the_box_records(self) -> None:
        """`docs/reference/tier-costs-on-the-box.md`, read again on 2026-10-04."""
        assert (quota.SAMPLE_JOB_CPU, quota.SAMPLE_JOB_WALL) == (300.0, 600.0)
        assert quota.CHECK_CPU_A_DAY == 1500.0
        labels = [label for label, _ in quota.summary()]
        assert "One spot check of a score may take" in labels


class TestASuccessfulPublish:
    """The tests Q503a left for this task: a draft that publishes through the route."""

    def test_a_draft_with_a_spot_checked_reading_publishes(self, draft: Draft) -> None:
        # Before any reading: refused, and the panel says how to get one.
        panel = drafts.standing(draft.connection, draft.game, draft.ada, draft.basis)
        assert panel is not None and panel["open"] is False
        assert panel["waiting"] == drafts.NEEDS_A_READING
        draft.connection.commit()
        with pytest.raises(drafts.PublishError) as refused:
            draft.publish()
        assert str(refused.value) == drafts.NEEDS_A_READING

        # Measured in the browser, walked, spot-checked.
        draft.send()
        draft.app.checks.drain()
        reading = drafts.attached(draft.connection, draft.game, draft.basis)
        assert reading is not None and reading["state"] == SPOT_CHECKED
        panel = drafts.standing(draft.connection, draft.game, draft.ada, draft.basis)
        assert panel is not None
        assert [tick["met"] for tick in panel["ticks"]] == [True, True]
        assert (panel["open"], panel["waiting"]) == (True, "")
        draft.connection.commit()

        # Somebody else cannot press it, and the designer can.
        with pytest.raises(drafts.PublishError, match="Only the person who saved"):
            draft.publish(draft.bo)
        done = draft.publish()
        assert done == {
            "hash": draft.game,
            "listed": True,
            "said": "Published. It is on trial: others can find it, play it and vote on it.",
        }
        [(visibility, published, version)] = draft.query(
            "select visibility, published, terms_version from games where hash = %s",
            draft.game,
        )
        assert (visibility, version) == ("public", terms.VERSION) and published is not None
        assert draft.publish()["said"] == "Already on trial: listed with the other games."

    def test_a_reading_that_is_only_legal_does_not_open_it(self, draft: Draft) -> None:
        draft.send()
        with draft.app._connection() as connection:
            legality.run(connection, draft.runs, int(legality.claim(connection) or 0),
                         basis=draft.basis)
        assert drafts.attached(draft.connection, draft.game, draft.basis) is None
        draft.connection.commit()
        with pytest.raises(drafts.PublishError) as refused:
            draft.publish()
        assert str(refused.value) == drafts.NEEDS_A_READING

    def test_a_reading_on_another_basis_does_not_open_it(self, draft: Draft) -> None:
        draft.send()
        draft.app.checks.drain()
        moved = replace(draft.measured.current, weights_version="v999")
        assert drafts.attached(draft.connection, draft.game, lambda g, s: moved) is None
        draft.connection.commit()
        draft.app.checks.basis = lambda game, spec: moved
        with pytest.raises(drafts.PublishError) as refused:
            draft.publish()
        assert str(refused.value) == drafts.NEEDS_A_READING


class TestTheShortlistsGrace:
    def _checked(self, draft: Draft) -> provisional.Result:
        draft.send()
        draft.app.checks.drain()
        [(result, _spec)] = provisional.standing(draft.connection, [draft.game]).values()
        draft.connection.commit()
        return result

    def test_a_current_reading_counts(self, draft: Draft) -> None:
        self._checked(draft)
        found = provisional.counted(draft.connection, draft.game, draft.basis, 30)
        assert found is not None and "stale" not in found

    def test_a_stale_reading_counts_for_thirty_days_then_stops(self, draft: Draft) -> None:
        result = self._checked(draft)
        moved = replace(draft.measured.current, weights_version="v999")

        def later(game: str, spec: Any) -> Basis:
            return moved

        assert provisional.mark_stale(draft.connection, later) == 1
        draft.connection.commit()
        [(since,)] = draft.query("select stale_since from provisional where id = %s", result.id)
        assert since is not None
        found = provisional.counted(draft.connection, draft.game, later, quota.STALE_GRACE_DAYS)
        assert found is not None and found["stale"] is True
        late = since + timedelta(days=quota.STALE_GRACE_DAYS)
        assert provisional.counted(
            draft.connection, draft.game, later, quota.STALE_GRACE_DAYS, now=late
        ) is None
        just = late - timedelta(minutes=1)
        assert provisional.counted(
            draft.connection, draft.game, later, quota.STALE_GRACE_DAYS, now=just
        ) is not None
        # *Publish* never takes a stale one, grace or not.
        assert provisional.attached(draft.connection, draft.game, later) is None
        # A basis that comes back (a deploy rolled back) clears the clock.
        assert provisional.mark_stale(draft.connection, draft.basis) == 0
        draft.connection.commit()
        assert draft.query(
            "select stale_since from provisional where id = %s", result.id
        ) == [(None,)]

    def test_a_stale_reading_nobody_marked_starts_its_clock_when_asked(
        self, draft: Draft
    ) -> None:
        result = self._checked(draft)
        moved = replace(draft.measured.current, weights_version="v999")
        found = provisional.counted(draft.connection, draft.game, lambda g, s: moved, 30)
        assert found is not None and found["stale"] is True
        draft.connection.commit()
        [(since,)] = draft.query("select stale_since from provisional where id = %s", result.id)
        assert since is not None

    def test_the_runner_stamps_stale_readings_when_it_starts(self, draft: Draft) -> None:
        result = self._checked(draft)
        moved = replace(draft.measured.current, weights_version="v999")
        draft.app.checks.basis = lambda game, spec: moved
        draft.app.checks.every = 30.0
        draft.app.checks.start()
        draft.app.checks.stop()
        [(since,)] = draft.query("select stale_since from provisional where id = %s", result.id)
        assert since is not None
