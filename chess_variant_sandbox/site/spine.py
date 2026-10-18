"""A member's game gets its leaderboard when it is published (L5).

`docs/archive/plan-leaderboard.md` §2.6 step 1, with Q1 decided on 2026-10-10 (*A, as soon
as it is published*). When a member publishes a game, the site rates the five standard
computer players (`opponents.SPINE`: *random*, *greedy*, *easy*, *medium*, *hard*) against
one another on its own server and seeds the game's rating table from them, so rated games,
the leaderboard and *Unbeaten* are on from the start. The stronger players (*hardest*, *tree
search*, rules of thumb, Fairy-Stockfish) join later, as a top-up over these same games on
the 16-core machine (L6, Q821 in `docs/plan-game-pipeline.md` §5); until then a game against
one of them is practice, and the page says so (`site/leaderboard.py`'s `box`).

## The job

A `jobs` row of kind `spine`, filed by `drafts.publish` in the transaction that lists the
game (`file`), and by the runner when it starts for any game on trial that has none
(`file_owed`: the games published before this existed). **Once a game**: a game whose table
already holds a computer player, or that has had a spine job in any state, is never filed
again, so re-publishing a resting game does not rate it again, and a game refused by a cap
is not retried on the box. The job has no requester: it is the site's work, not part of
anybody's day of checking (`quota.CHECK_CPU_A_DAY` sums a requester's jobs).

It runs on `legality.Checker`'s thread, the site's one runner, after every queued legality
and spot-check job (each of those is somebody's *Publish* waiting), one game at a time.

## Why a child process, and its caps

For `screening`'s reason: the core's work has to be *stopped*, not waited on, and a Python
thread cannot cancel a call inside the core. So the games are played in a child -- this
file run as a script -- that the runner kills at `quota.SPINE_WALL`, and which caps itself
before it plays anything: `RLIMIT_CPU` at `quota.SPINE_CPU` (the kernel ends it with
`SIGXCPU`), `RLIMIT_DATA` at `quota.SPINE_MEMORY`, and `nice 19`, where the platform has
them (Linux, the box; elsewhere the wall clock is the cap). It plays at one thread
(`WORKERS`), so a publish costs the box one core. It writes nothing: it hands the table and
its games back on stdout, and the runner writes the ladder (`opponents.write_ladder`, under
`runs/opponents/` as any ladder) and seeds the table (`rating.reseed`, the loader's own
rule, so the first seeding centres on the spine as every game's does).

**A game past a cap is refused, not failed**: the job is `refused` with the cap it hit, the
game gets no ladder here and waits for the 16-core machine, and its page says so.

## Determinism

The ladder is `opponents.rate_spine`: `_plan` over the spine at the ladders' own games and
seed (`LADDER_GAMES`, `LADDER_SEED`), fixed before any thread starts, so it is a pure
function of the game's hash and those seeds -- the same table and games this machine's
`Ratings` makes for the same five players, at any thread count.

Nothing here commits, except `run`, which owns its transactions because it holds none while
the child plays.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KIND = "spine"
"""`jobs.kind` for the rating of a member's spine at publish."""

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
REFUSED = "refused"
FAILED = "failed"
IN_FLIGHT: tuple[str, ...] = (QUEUED, RUNNING)

WORKERS = 1
"""The core's threads for the child's games: one publish, one core (`screening`'s rule)."""

DESIGNED = "designed"
PUBLIC = "public"

# What a refused job's error begins with: which cap it hit.
OVER_TIME = "over-time"
OVER_MEMORY = "over-memory"

# What the game page is told about the site's own rating of a member's game
# (`state`, read by `leaderboard.shown` as `box`).
DRAFT = "draft"
"""Not published, so not rated: they are rated when it is."""
RATING = "rating"
"""Published, and the five are being rated now (queued or running)."""
RATED = "rated"
"""The five were rated here; the rest wait for our own machine."""
SLOW = "slow"
"""Refused by the time caps: waiting for our own machine."""
LARGE = "large"
"""Refused by the memory cap: waiting for our own machine."""
WAITING = "waiting"
"""The rating here failed for another reason: waiting for our own machine."""


@dataclass(frozen=True)
class Caps:
    """The child's limits: CPU-seconds, wall-clock seconds and bytes of memory."""

    cpu: float
    wall: float
    memory: int


def caps() -> Caps:
    """The quota's caps (`quota.SPINE_CPU`, `SPINE_WALL`, `SPINE_MEMORY`)."""
    from chess_variant_sandbox.site import quota

    return Caps(cpu=quota.SPINE_CPU, wall=quota.SPINE_WALL, memory=quota.SPINE_MEMORY)


# ------------------------------------------------------------------- the queue


def _rated(connection: Any, digest: str) -> bool:
    """Whether the game's table already holds a computer player."""
    from chess_variant_sandbox.site import rating

    return bool(rating.computers(rating.load(connection, digest)))


def file(connection: Any, digest: str) -> int | None:
    """Queue the rating of a published member's game, once ever. Returns the job.

    `None`, and nothing queued, when the game is not a member's public game, when its table
    already has computer players (it was rated, or a ladder reached it), or when it has had
    a spine job in any state: re-publishing never rates a game again, and a game a cap
    refused waits for our own machine rather than spending the box's cap a second time.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select origin, visibility, revoked, deleted from games where hash = %s",
            (digest,),
        )
        found = cursor.fetchone()
    if found is None:
        return None
    origin, visibility, revoked, deleted = found
    if str(origin) != DESIGNED or str(visibility) != PUBLIC:
        return None
    if revoked is not None or deleted is not None:
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            "select 1 from jobs where kind = %s and hash = %s limit 1", (KIND, digest)
        )
        if cursor.fetchone() is not None:
            return None
    if _rated(connection, digest):
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into jobs (kind, hash, state) values (%s, %s, %s) returning id",
            (KIND, digest, QUEUED),
        )
        return int(cursor.fetchone()[0])


def owed(connection: Any) -> list[str]:
    """Every member's public game with no computer player rated and no spine job yet.

    The games published before L5, and any a publish missed. Oldest publish first.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select g.hash from games g"
            " where g.origin = %s and g.visibility = %s and g.revoked is null"
            " and g.deleted is null and g.spec is not null"
            " and not exists (select 1 from jobs j where j.kind = %s and j.hash = g.hash)"
            " and not exists (select 1 from ratings r where r.game = g.hash"
            "                 and r.player_agent is not null)"
            " order by g.published nulls last, g.hash",
            (DESIGNED, PUBLIC, KIND),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def file_owed(connection: Any) -> list[int]:
    """`file` every game `owed` names. What the runner does when it starts (a deploy)."""
    filed = []
    for digest in owed(connection):
        job = file(connection, digest)
        if job is not None:
            filed.append(job)
    return filed


def waiting_ahead(connection: Any) -> bool:
    """Whether a legality walk or a spot check is queued: those go first, every time."""
    from chess_variant_sandbox.site import provisional

    with connection.cursor() as cursor:
        cursor.execute(
            "select 1 from jobs where kind = any(%s) and state = %s limit 1",
            ([provisional.WALK, provisional.SPOT], QUEUED),
        )
        return cursor.fetchone() is not None


def claim(connection: Any, worker: str = "server") -> int | None:
    """Take the oldest queued spine job, or `None`, stamped with the server's build."""
    from chess_variant_sandbox.measurement import built_fingerprint

    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, claimed = now(), worker = %s, fingerprint = %s"
            " where id = (select id from jobs where kind = %s and state = %s"
            " order by created, id limit 1 for update skip locked) returning id",
            (RUNNING, worker, built_fingerprint(), KIND, QUEUED),
        )
        found = cursor.fetchone()
    return None if found is None else int(found[0])


def requeue(connection: Any) -> int:
    """Put back the spine jobs a stopped process left `running`."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, claimed = null, worker = null"
            " where kind = %s and state = %s",
            (QUEUED, KIND, RUNNING),
        )
        return int(cursor.rowcount or 0)


def state(connection: Any, digest: str) -> str | None:
    """What the game page says about the site's own rating of this game, or `None`.

    `None` for a game that is not a member's (a library game is rated on our own machine,
    and the page says so in its own words). Otherwise `DRAFT` before it is published, then
    the newest spine job's: `RATING` while it waits or runs, `RATED` once it seeded the
    table, `SLOW` or `LARGE` when a cap refused it and `WAITING` when it failed otherwise --
    the last three all waiting for our own machine. A member's public game with no job is
    `RATING`: the runner files it on its next start.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select origin, visibility from games where hash = %s", (digest,)
        )
        found = cursor.fetchone()
    if found is None or str(found[0]) != DESIGNED:
        return None
    if str(found[1]) != PUBLIC:
        return DRAFT
    with connection.cursor() as cursor:
        cursor.execute(
            "select state, error from jobs where kind = %s and hash = %s"
            " order by id desc limit 1",
            (KIND, digest),
        )
        job = cursor.fetchone()
    if job is None:
        return RATED if _rated(connection, digest) else RATING
    job_state, error = str(job[0]), str(job[1] or "")
    if job_state in IN_FLIGHT:
        return RATING
    if job_state == DONE:
        return RATED
    if job_state == REFUSED:
        return LARGE if error.startswith(OVER_MEMORY) else SLOW
    return WAITING


# --------------------------------------------------------------------- the run


@dataclass(frozen=True)
class Outcome:
    """How one spine job ended."""

    job: int
    hash: str
    state: str
    said: str = ""
    cpu_seconds: float | None = None
    seeded: int = 0


class OverCap(Exception):
    """The child ran past one of its caps. `reason` is `OVER_TIME` or `OVER_MEMORY`."""

    def __init__(self, reason: str, said: str) -> None:
        super().__init__(said)
        self.reason = reason
        self.said = said


def run(
    connect: Callable[[], Any],
    runs_dir: Path,
    job: int,
    limits: Caps | None = None,
) -> Outcome:
    """Rate one claimed job's game in the capped child, then write its ladder and seed it.

    `connect` is a context-manager factory that commits on the way out (`SiteApp._connection`):
    the game is read in one transaction, the child plays with none open, and the ladder's
    rows and the job's end land in a second, together. A fault is the server's, recorded on
    the job as `failed`, and the game waits for our own machine.
    """
    limits = limits or caps()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "select j.kind, j.state, j.hash, g.spec, g.revoked, g.deleted from jobs j"
            " left join games g on g.hash = j.hash where j.id = %s",
            (job,),
        )
        found = cursor.fetchone()
    if found is None:
        raise ValueError(f"no job numbered {job}")
    kind, job_state, digest, spec, revoked, deleted = found
    if kind != KIND:
        raise ValueError(f"job {job} is a {kind} job, not a spine rating")
    if job_state not in IN_FLIGHT:
        raise ValueError(f"job {job} is already {job_state}")
    digest = str(digest)
    if spec is None or revoked is not None or deleted is not None:
        said = "the game was taken off the site before its players were rated"
        with connect() as connection:
            _finish(connection, job, FAILED, None, error=said)
        return Outcome(job, digest, FAILED, said)

    started = time.monotonic()
    try:
        answer = in_a_child(dict(spec), digest, limits)
    except OverCap as over:
        with connect() as connection:
            _finish(
                connection, job, REFUSED,
                limits.cpu if over.reason == OVER_TIME else None,
                error=f"{over.reason}: {over.said}",
            )
        return Outcome(job, digest, REFUSED, over.said)
    except Exception as error:  # noqa: BLE001 - a fault is recorded on the job, never lost
        said = f"{type(error).__name__}: {error}"
        with connect() as connection:
            _finish(connection, job, FAILED, None, error=said)
        return Outcome(job, digest, FAILED, said)

    from chess_variant_sandbox.site import rating
    from chess_variant_sandbox.web import opponents

    table, lines = answer["table"], answer["games"]
    cpu = float(answer.get("cpu") or 0.0)
    peak = answer.get("peak")
    memory = "" if not peak else f", peak {int(peak) // (1024 * 1024)} MB"
    written = ""
    try:
        path = opponents.write_ladder(digest, runs_dir, table, lines)
        written = f" -> {path.name}"
    except OSError as error:
        # The table still seeds: rated play is what the publish was promised. The kept
        # games are what a later top-up needs, so the log says they are missing.
        written = f"; its ladder file could not be written ({error})"
    with connect() as connection:
        seated = rating.reseed(connection, digest, table)
        said = (
            f"rated {len(table.get('standings') or ())} players in {len(lines)} games, "
            f"{cpu:.1f} CPU-s, {time.monotonic() - started:.1f} s{memory}{written}"
        )
        _finish(connection, job, DONE, cpu, log=said)
    return Outcome(job, digest, DONE, said, cpu, len(seated.written))


def _finish(
    connection: Any,
    job: int,
    job_state: str,
    cpu: float | None,
    error: str = "",
    log: str = "",
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, finished = clock_timestamp(), error = %s, log = %s,"
            " cpu_seconds = %s where id = %s",
            (job_state, error or None, log or None,
             None if cpu is None else round(max(cpu, 0.0), 3), job),
        )


def in_a_child(spec: dict[str, Any], digest: str, limits: Caps) -> dict[str, Any]:
    """Play the spine's ladder in a child under the caps; its table, games and CPU-seconds.

    Raises `OverCap` when a cap ended it and `RuntimeError` when it died otherwise.
    """
    asked = json.dumps({
        "spec": spec,
        "hash": digest,
        "cpu": limits.cpu,
        "memory": limits.memory,
        "workers": WORKERS,
    })
    # `-I` keeps this directory off the child's `sys.path`, as `screening` does, so running
    # the file by path cannot shadow an import with a module that sits beside it.
    child = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = child.communicate(asked, timeout=limits.wall)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise OverCap(OVER_TIME, over_time(limits)) from None
    said: dict[str, Any] = {}
    if out.strip():
        try:
            said = json.loads(out)
        except ValueError:
            said = {}
    if child.returncode == 0 and isinstance(said.get("table"), dict):
        return said
    if said.get("over") == OVER_MEMORY or _out_of_memory(child.returncode, err or ""):
        raise OverCap(OVER_MEMORY, over_memory(limits))
    if _out_of_time(child.returncode):
        raise OverCap(OVER_TIME, over_time(limits))
    lines = (err or "").strip().splitlines()
    last = lines[-1][:200] if lines else f"exit {child.returncode}"
    print(f"spine: the rating's child failed: {last}", file=sys.stderr)
    raise RuntimeError(f"the rating's child failed: {last}")


def over_time(limits: Caps) -> str:
    return (
        f"rating the five standard players took longer than this site allows "
        f"({limits.cpu:g} CPU-seconds, or {limits.wall:g} seconds on the clock)"
    )


def over_memory(limits: Caps) -> str:
    return (
        f"rating the five standard players needed more memory than this site allows "
        f"({limits.memory // (1024 * 1024)} MB)"
    )


def _out_of_time(code: int | None) -> bool:
    """The kernel's `RLIMIT_CPU` ending: `SIGXCPU` at the soft limit, `SIGKILL` at the hard."""
    sigxcpu = getattr(signal, "SIGXCPU", None)
    return code is not None and sigxcpu is not None and code == -int(sigxcpu)


def _out_of_memory(code: int | None, err: str) -> bool:
    """A child that ran out of memory: Python's `MemoryError`, the core's allocation abort,
    or a kill from outside (the unit's memory ceiling)."""
    if "MemoryError" in err or "memory allocation of" in err:
        return True
    kill = getattr(signal, "SIGKILL", None)
    return code is not None and kill is not None and code == -int(kill)


# ------------------------------------------------------------------ the child


def limit_self(cpu: float, memory: int) -> list[str]:
    """Cap this process before it plays: CPU-seconds, memory and priority. What it set.

    `RLIMIT_CPU`'s soft limit sends `SIGXCPU`, which ends a Python process, and its hard
    limit five seconds on is the kernel's `SIGKILL`. `RLIMIT_DATA` covers the heap and
    every private writable mapping, which is where the core's search tables live. `nice 19`
    so the site's request threads come first. A platform without `resource` (Windows) has
    none of these, and the parent's wall clock is the cap.
    """
    applied = []
    resource = _resource()
    if resource is not None:
        seconds = max(1, int(cpu))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))
        applied.append("cpu")
        if memory > 0:
            resource.setrlimit(resource.RLIMIT_DATA, (int(memory), int(memory)))
            applied.append("memory")
    setpriority = getattr(os, "setpriority", None)
    if setpriority is not None:
        try:
            setpriority(getattr(os, "PRIO_PROCESS", 0), 0, 19)
            applied.append("nice")
        except OSError:
            pass
    return applied


def _resource() -> Any:
    """The `resource` module where the platform has one (Unix), else `None`.

    Imported by name so the type checker, run on any platform, reads it as what it is: a
    module that may not exist here.
    """
    import importlib

    try:
        return importlib.import_module("resource")
    except ImportError:
        return None


def _peak_memory() -> int | None:
    """This process's peak resident memory in bytes, where the platform says (Linux)."""
    resource = _resource()
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _child() -> None:
    asked = json.loads(sys.stdin.read())
    limit_self(float(asked["cpu"]), int(asked["memory"]))
    try:
        from chess_variant_sandbox.web import opponents

        table, records = opponents.rate_spine(
            asked["spec"], str(asked["hash"]), workers=int(asked["workers"])
        )
        answer = {
            "table": table,
            "games": opponents.game_lines(records),
            "cpu": time.process_time(),
            "peak": _peak_memory(),
        }
    except MemoryError:
        json.dump({"over": OVER_MEMORY}, sys.stdout)
        sys.stdout.flush()
        sys.exit(3)
    json.dump(answer, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    _child()
