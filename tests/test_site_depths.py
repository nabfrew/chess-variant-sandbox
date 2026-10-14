"""Q813: `/admin`'s *Status*, the box's disk and the site's queues (`site/depths.py`).

The queues are `site/depths.sql`, run against the session's migrated store
(`tests/site_store.py`) with rows written at known distances from the database's clock, so
what is proved is the statement against the real schema. Every row is written in the
test's own transaction, where `now()` stands still, so an age is exact to the second.
The disk is plain arithmetic and is tested without a store.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import page_text
import pytest
import site_store
import site_words

from chess_variant_sandbox.site import db, depths, pages, quota

ORDER = ["checks", "publishes", "ratings", "sweep", "asks", "reports", "training"]
DEPLOY = Path(__file__).resolve().parent.parent / "deploy"


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The view's disk, fixed: this machine's own must not decide a queue's test."""
    monkeypatch.setattr(depths, "disk", lambda path: depths.Disk(used=10 * 2**30, free=30 * 2**30))


@pytest.fixture
def store() -> Iterator[Any]:
    url, schema = site_store.emptied("no database to read the queues from")
    connection = db.connect(url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
        connection.commit()
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _ago(cursor: Any, seconds: float) -> Any:
    """The database's `now()` less `seconds`: a moment the statement will age exactly."""
    cursor.execute("select now() - make_interval(secs => %s)", (seconds,))
    return cursor.fetchone()[0]


def _person(cursor: Any, handle: str) -> int:
    cursor.execute(
        "insert into users (handle, email) values (%s, %s) returning id",
        (handle, f"{handle}@example.org"),
    )
    return int(cursor.fetchone()[0])


def _game(cursor: Any, name: str, publish_asked: float | None = None) -> str:
    asked = None if publish_asked is None else _ago(cursor, publish_asked)
    cursor.execute(
        "insert into games (hash, spec, name, origin, publish_asked)"
        " values (%s, '{}', %s, 'test', %s)",
        (name, name, asked),
    )
    return name


def _job(cursor: Any, kind: str, state: str, seconds: float) -> None:
    cursor.execute(
        "insert into jobs (kind, state, created) values (%s, %s, %s)",
        (kind, state, _ago(cursor, seconds)),
    )


def _match(
    cursor: Any, name: str, idle: timedelta, *, mode: str = "leaderboard",
    seat0: int | None = None, seat1: int | None = None, finished: bool = False,
    source: str | None = None,
) -> None:
    """A match last moved `idle` ago; a seat with no person is a computer player."""
    moved = _ago(cursor, idle.total_seconds())
    cursor.execute(
        "insert into matches (id, game, mode, seat0_user, seat0_agent, seat1_user,"
        " seat1_agent, seed, started, moved, finished, source)"
        " values (%s, 'g', %s, %s, %s, %s, %s, 1, %s, %s, %s, %s)",
        (name, mode, seat0, None if seat0 is not None else "random", seat1,
         None if seat1 is not None else "random", moved, moved,
         moved if finished else None, source),
    )


def _queues(store: Any) -> dict[str, depths.Queue]:
    found = depths.queues(store)
    assert [one.queue for one in found] == ORDER
    return {one.queue: one for one in found}


# ------------------------------------------------------------------------ the statement


def test_an_empty_store_has_every_queue_empty_and_nothing_past_its_limit(store: Any) -> None:
    found = _queues(store)
    for one in found.values():
        assert (one.depth, one.age, one.oldest_said, one.problem) == (0, None, None, False)
        assert one.said.startswith(one.label + ": none")
    assert found["checks"].said == "checks of submitted scores: none (a problem past 30 min)"
    assert found["training"].said == "games queued for training: none (no limit yet)"
    view = depths.view(store, Path("."))
    assert view["said"] == "Nothing past its limit."
    assert [one["queue"] for one in view["queues"]] == ORDER


def test_checks_count_walks_and_spot_checks_in_flight_and_age_the_oldest(store: Any) -> None:
    with store.cursor() as cursor:
        _job(cursor, "legality", "queued", 10 * 60)
        _job(cursor, "sample", "running", 20 * 60)
        _job(cursor, "legality", "done", 90 * 60)
        _job(cursor, "sample", "failed", 90 * 60)
        _job(cursor, "spine", "queued", 5 * 60)
    checks = _queues(store)["checks"]
    assert (checks.depth, checks.age, checks.oldest_said) == (2, 20 * 60, "20 min")
    assert not checks.problem
    with store.cursor() as cursor:
        _job(cursor, "legality", "queued", 31 * 60)
    checks = _queues(store)["checks"]
    assert (checks.depth, checks.problem) == (3, True)
    assert checks.said == (
        "checks of submitted scores: 3, the oldest 31 min -- PROBLEM, past 30 min"
    )


def test_a_publish_waits_from_its_press_and_reads_as_a_problem_past_an_hour(
    store: Any,
) -> None:
    with store.cursor() as cursor:
        _game(cursor, "settled")
        _game(cursor, "pressed", publish_asked=50 * 60)
    publishes = _queues(store)["publishes"]
    assert (publishes.depth, publishes.oldest_said, publishes.problem) == (1, "50 min", False)
    with store.cursor() as cursor:
        _game(cursor, "stuck", publish_asked=61 * 60)
    publishes = _queues(store)["publishes"]
    assert (publishes.depth, publishes.oldest_said, publishes.problem) == (2, "61 min", True)


def test_ratings_at_publish_are_spine_jobs_in_flight_and_a_problem_past_two_hours(
    store: Any,
) -> None:
    with store.cursor() as cursor:
        _job(cursor, "spine", "queued", 119 * 60)
        _job(cursor, "spine", "done", 9 * 3600)
    ratings = _queues(store)["ratings"]
    assert (ratings.depth, ratings.oldest_said, ratings.problem) == (1, "119 min", False)
    with store.cursor() as cursor:
        _job(cursor, "spine", "running", 3 * 3600)
    ratings = _queues(store)["ratings"]
    assert (ratings.depth, ratings.oldest_said, ratings.problem) == (2, "3 h", True)


def test_the_sweep_counts_what_matches_sweep_would_resign_aged_from_when_it_fell_due(
    store: Any,
) -> None:
    day = timedelta(days=1)
    with store.cursor() as cursor:
        _game(cursor, "g")
        ann, bea = _person(cursor, "ann"), _person(cursor, "bea")
        # Due, and the sweep resigns them: a person against a computer on the
        # leaderboard past thirty days, and two people past five, in either mode.
        _match(cursor, "lb-due", 30 * day + timedelta(minutes=10), seat0=ann)
        _match(cursor, "people-due", 5 * day + timedelta(minutes=20), mode="sandbox",
               seat0=ann, seat1=bea)
        # Not due, or never the sweep's.
        _match(cursor, "lb-fresh", 29 * day, seat0=ann)
        _match(cursor, "people-fresh", 4 * day, seat0=ann, seat1=bea)
        _match(cursor, "computers", 90 * day)
        _match(cursor, "practice", 90 * day, mode="sandbox", seat0=ann)
        _match(cursor, "over", 90 * day, seat0=ann, finished=True)
        _match(cursor, "imported", 90 * day, seat0=ann, source="file")
    sweep = _queues(store)["sweep"]
    assert (sweep.depth, sweep.age, sweep.problem) == (2, 20 * 60, False)
    with store.cursor() as cursor:
        _match(cursor, "lb-late", 30 * day + timedelta(minutes=31), seat1=bea)
    sweep = _queues(store)["sweep"]
    assert (sweep.depth, sweep.oldest_said, sweep.problem) == (3, "31 min", True)


def test_the_sweeps_two_limits_are_the_sites() -> None:
    """The statement writes the two limits as SQL intervals; they are `quota`'s."""
    text = depths.statement()
    for limit in (quota.IDLE_LIMIT, quota.CORRESPONDENCE_IDLE):
        assert limit == timedelta(days=limit.days), "a whole number of days"
        assert text.count(f"interval '{limit.days} days'") == 2


def test_requests_for_an_invitation_wait_until_answered_and_a_week_is_a_problem(
    store: Any,
) -> None:
    with store.cursor() as cursor:
        cursor.execute(
            "insert into invite_asks (email, created, answered, granted) values"
            " ('a@example.org', %s, null, null), ('b@example.org', %s, now(), true)",
            (_ago(cursor, 6 * 86400), _ago(cursor, 20 * 86400)),
        )
    asks = _queues(store)["asks"]
    assert (asks.depth, asks.oldest_said, asks.problem) == (1, "6 days", False)
    with store.cursor() as cursor:
        cursor.execute(
            "insert into invite_asks (email, created) values ('c@example.org', %s)",
            (_ago(cursor, 8 * 86400),),
        )
    asks = _queues(store)["asks"]
    assert (asks.depth, asks.oldest_said, asks.problem) == (2, "8 days", True)


def test_open_reports_are_a_problem_past_two_days_and_a_dismissed_one_is_not_counted(
    store: Any,
) -> None:
    with store.cursor() as cursor:
        _game(cursor, "g")
        ann, bea = _person(cursor, "ann"), _person(cursor, "bea")
        cursor.execute(
            "insert into reports (reporter, game, reason, created, dismissed)"
            " values (%s, 'g', 'spam', %s, null), (%s, 'g', 'copy', %s, now())",
            (ann, _ago(cursor, 30 * 3600), bea, _ago(cursor, 9 * 86400)),
        )
    reports = _queues(store)["reports"]
    assert (reports.depth, reports.oldest_said, reports.problem) == (1, "30 h", False)
    with store.cursor() as cursor:
        cursor.execute("update reports set created = %s", (_ago(cursor, 49 * 3600),))
    reports = _queues(store)["reports"]
    assert (reports.depth, reports.oldest_said, reports.problem) == (1, "2 days", True)


def test_training_is_shown_and_never_a_problem_while_nothing_takes_it(store: Any) -> None:
    with store.cursor() as cursor:
        _job(cursor, "train", "queued", 400 * 86400)
        _job(cursor, "train", "failed", 400 * 86400)
    training = _queues(store)["training"]
    assert (training.depth, training.oldest_said, training.problem) == (1, "400 days", False)
    assert training.said == "games queued for training: 1, the oldest 400 days (no limit yet)"


@pytest.mark.parametrize(("seconds", "said"), [
    (0, "0 s"), (119, "119 s"), (120, "2 min"), (7199, "119 min"), (7200, "2 h"),
    (172799, "47 h"), (172800, "2 days"),
])
def test_an_age_is_worded_in_one_unit_rounded_down(
    store: Any, seconds: int, said: str,
) -> None:
    with store.cursor() as cursor:
        _job(cursor, "train", "queued", seconds)
    assert _queues(store)["training"].oldest_said == said


def test_the_view_names_what_is_past_its_limit_and_no_person(store: Any) -> None:
    with store.cursor() as cursor:
        _job(cursor, "legality", "queued", 3600)
        cursor.execute(
            "insert into invite_asks (email, created) values ('qzxv@example.org', %s)",
            (_ago(cursor, 30 * 86400),),
        )
    view = depths.view(store, Path("."))
    assert view["said"] == (
        "A problem: checks of submitted scores, requests for an invitation."
    )
    assert "qzxv" not in repr(view)
    assert set(view) == {"disk", "queues", "said"}
    assert set(view["queues"][0]) == {
        "queue", "label", "depth", "oldest", "limit", "problem", "said", "why",
    }


def test_every_label_and_reason_survives_psqls_separator(store: Any) -> None:
    """`deploy/status.sh` reads `psql -A` output split at `|`, a row to a line."""
    for one in depths.queues(store):
        for text in (one.label, one.said, one.why, one.limit_said):
            assert "|" not in text and "\n" not in text, text
        assert one.why.endswith(".")


# ------------------------------------------------------------------------ no store needed


def test_the_statement_carries_nothing_either_reader_would_interpret() -> None:
    """psycopg runs it with no parameters and psql reads it raw: no `%`, no `:name`."""
    text = depths.statement()
    code = "\n".join(line.split("--")[0] for line in text.splitlines())
    assert "%" not in code
    assert not re.search(r"(?<!:):[A-Za-z_'\"{]", code)
    for word in ("insert ", "update ", "delete ", "drop ", "alter ", "create "):
        assert word not in code.lower(), word


def test_the_thresholds_follow_the_caps_they_are_derived_from() -> None:
    """Each reason names the cap it is a multiple of; a cap that moves must move it."""
    text = depths.statement()
    assert quota.SAMPLE_JOB_WALL * 3 == 1800 and "(1, 'checks'" in text
    assert "1800, '30 min'" in text
    assert quota.SPINE_WALL * 8 == 7200 and "7200, '2 h'" in text


@pytest.mark.parametrize(("used", "free", "percent", "said", "problem"), [
    (0, 0, 0, "0% used, 0.0 GB free", False),
    (4_061_380 * 1024, 33_337_752 * 1024, 11, "11% used, 31.8 GB free", False),
    (89 * 2**30, 11 * 2**30, 89, "89% used, 11.0 GB free", False),
    (89 * 2**30 + 1, 11 * 2**30 - 1, 90, "90% used, 11.0 GB free", True),
    (95 * 2**30, 5 * 2**30, 95, "95% used, 5.0 GB free", True),
])
def test_the_disk_is_worked_out_as_df_works_it_out(
    used: int, free: int, percent: int, said: str, problem: bool,
) -> None:
    found = depths.Disk(used=used, free=free)
    assert (found.used_percent, found.said(), found.problem) == (percent, said, problem)
    assert found.as_dict()["limit"] == depths.DISK_LIMIT


def test_the_disk_reads_a_real_directory_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    found = depths.disk(tmp_path)
    assert found is not None and 0 <= found.used_percent <= 100
    # A `runs/` not made yet is read on the filesystem it will be made on.
    assert depths.disk(tmp_path / "not" / "there") is not None

    def refused(path: Path) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(depths.shutil, "disk_usage", refused)
    assert depths.disk(tmp_path) is None


def test_the_disk_limit_is_the_scripts() -> None:
    script = (DEPLOY / "status.sh").read_text(encoding="utf-8")
    assert f'[ "$used" -lt {depths.DISK_LIMIT} ]' in script


def test_the_box_half_words_the_disk_as_the_page_does() -> None:
    """`df -P -k`'s used and available, in KiB, through the box half's arithmetic."""
    script = (DEPLOY / "remote" / "status.sh").read_text(encoding="utf-8")
    assert "df -P -k /" in script
    assert "u = int(p); if (u < p) u++" in script and "$4 / 1048576" in script


# ------------------------------------------------------------------------ the page


def test_the_admin_page_draws_the_status_from_text_nodes() -> None:
    page = page_text.inlined(pages.admin("boss"))
    assert '<tbody id="status-queues"></tbody>' in page
    assert 'id="status-said"' in page and 'id="status-disk"' in page
    assert "status(payload.status)" in page
    assert "innerHTML" not in page
    # Second on the page, under the one button a phone visit is for.
    assert page.index("<h2>Sessions</h2>") < page.index("<h2>Status</h2>") < page.index(
        "<h2>People</h2>")


def test_every_status_sentence_passes_the_words_guard(store: Any) -> None:
    page = pages.admin("boss")
    said = [page[page.index("<h2>Status</h2>"): page.index("<h2>People</h2>")]]
    view = depths.view(store, Path("."))
    said += [view["said"], view["disk"]["said"]] if view["disk"] else [view["said"]]
    for one in view["queues"]:
        said += [one["label"], one["said"], one["why"]]
    trouble = [(text, site_words.hits(text)) for text in said if site_words.hits(text)]
    assert not trouble, trouble
