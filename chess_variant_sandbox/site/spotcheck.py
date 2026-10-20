"""Tier two: the spot check, which replays a sample of a browser's measurement exactly.

`docs/archive/plan-client-compute.md`'s row *Tier two, the sample job*, re-aimed at *Publish* by
`docs/archive/plan-game-pipeline.md` Q503e. Tier one (`legality`) has walked every submitted game
for legal moves and outcomes and scored the records itself. What it
cannot prove is that the rostered players chose those moves, the search counters, the
probe's `near_best` counts and each blunder trial's new winner. This job proves them for a
sample: it plays the sampled games again on the server's own core and requires every byte
of each record to come out the same.

## What is replayed

From the result's own records under `runs/provisional/<id>/`, drawn by `draw`:

- **32 of the tournament's games** (all of them under 32), each played again from its
  schedule index by the ordinary runner, and compared with its stored record field for
  field, `duration_ms` aside (a wall clock, the one field no two machines share);
- **the probe of every sampled game the analysis pass grades**, compared with its stored
  analysis;
- **6 of the blunder sampler's picks**, each trial run again and compared with the stored
  trial for that game, or with its absence when the trial found no move to force.

The sample is an HMAC of the submission's text under a key the server draws once and does
not keep (`seal`), stored on the job (`jobs.sample`). So it is the same when drawn twice
from the same text, it differs when one byte of the text differs, and the person who sent
the text cannot know it in advance: a forger who swaps a few games for flattering legal
ones is caught when the draw lands on one. **It is a detector, not a certificate** (T2,
`docs/reference/forgery-shift.md`): swapping 8 of 256 games is missed about one time in
three, which is why a spot-checked score is labelled as such and never ranks or awards.

## What follows

- **Every row equal**: the result moves to `spot-checked`, its stamp names the sample's
  size, everybody who can see the game sees the number, labelled, and it is the reading
  *Publish* needs (`provisional.attached`): a publish its designer pressed while this
  check ran goes through on the runner's next settle (`drafts.settle`, Q742).
- **A row that differs**: the result moves to `mismatch` and is never shown again, its
  records are kept, and the account is flagged (`provisional.flag`): it cannot publish, and
  what it sends later is walked by tier one but never spot-checked, so it waits for the
  server's own measurement until an admin clears the flag. The job's error names the phase,
  the game and its seed, the first ply where the moves part (or the first field that
  differs) and both build fingerprints: the site's now, and the one the result was stamped
  with.
- **Out of budget** (`quota.SAMPLE_JOB_CPU` CPU-seconds of this thread, or
  `quota.SAMPLE_JOB_WALL` seconds): the check stops between two games, the result stays
  `legal`, and it waits for the server's own measurement. Running out is never a mismatch.

## Where it runs

On `legality.Checker`'s thread, the site's one runner, after every queued legality job
(tier one runs ahead of every other kind), at `nice 19` where the platform has it, one
game at a time on one core. Each job records the processor time its thread spent in
`jobs.cpu_seconds`, and a person's day of checking is the sum (`quota.CHECK_CPU_A_DAY`).

Nothing here commits. The caller owns the transaction, as in `legality`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import secrets
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.measurement import LEGAL, built_fingerprint
from chess_variant_sandbox.site import provisional, quota
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig

KIND = provisional.SPOT
"""`jobs.kind` for tier two. It runs after every queued legality job."""

# The jobs table's words, as `legality` spells them.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
REFUSED = "refused"
FAILED = "failed"
IN_FLIGHT: tuple[str, ...] = (QUEUED, RUNNING)

MISMATCH = "mismatch"
"""The refusal's name in `jobs.error` when a replayed row differs."""

GAMES = 32
"""Tournament games replayed, at most (`plan-client-compute.md`'s tier table)."""

TRIALS = 6
"""Blunder trials replayed, at most."""

IGNORED = frozenset({"duration_ms"})
"""Fields no two machines share, left out of every comparison."""

Now = provisional.Current
"""The server's basis for a game's one-run measurement, from its hash and its rules."""


# ---------------------------------------------------------------------- the draw


def seal(text: str | bytes, key: bytes | None = None) -> str:
    """The sample's seed: an HMAC-SHA256 of the submission under a key nobody keeps.

    `key` is for a test; the server draws a fresh one per submission and forgets it, so
    the seal is unpredictable to the sender and fixed once stored.
    """
    body = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    chosen = secrets.token_bytes(32) if key is None else key
    return hmac.new(chosen, body, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Sample:
    """What one spot check replays, as indices into the result's own records."""

    games: tuple[int, ...]
    trials: tuple[int, ...]


def draw(sealed: str, *, games: int, picks: Sequence[int]) -> Sample:
    """The sample a seal names: a pure function of the seal and the records' sizes.

    `games` is how many tournament games there are, `picks` the blunder sampler's picks as
    tournament indices (in its draw order). Its own `random.Random`, seeded by the seal, never the module's.
    """
    rng = random.Random(int(sealed, 16))
    chosen = tuple(sorted(rng.sample(range(games), min(GAMES, games))))
    trials = tuple(sorted(rng.sample(list(picks), min(TRIALS, len(picks)))))
    return Sample(games=chosen, trials=trials)


# ------------------------------------------------------------------ filing a check


def file(
    connection: Any, *, game: str, submitter: int, result: int, text: str | bytes
) -> int | None:
    """Queue the spot check of a result tier one has just recorded. Returns the job.

    `None`, and nothing queued, for a flagged account: its results wait for the server's
    own measurement (`provisional.flag`).
    """
    if provisional.flagged(connection, submitter):
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into jobs (kind, hash, requester, state, result, sample)"
            " values (%s, %s, %s, %s, %s, %s) returning id",
            (KIND, game, submitter, QUEUED, result, seal(text)),
        )
        return int(cursor.fetchone()[0])


def claim(connection: Any, worker: str = "server") -> int | None:
    """Take the oldest queued spot check, or `None`, stamped with the server's build."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, claimed = now(), worker = %s, fingerprint = %s"
            " where id = (select id from jobs where kind = %s and state = %s"
            " order by created, id limit 1 for update skip locked) returning id",
            (RUNNING, worker, built_fingerprint(), KIND, QUEUED),
        )
        found = cursor.fetchone()
    return None if found is None else int(found[0])


def of_result(connection: Any, result: int) -> dict[str, Any] | None:
    """The newest spot check of a result, as the page is told it, or `None` before one.

    `state` is the job's; a refusal is always `mismatch` here, and `stopped` is a check
    that ran out of budget and left the result as it was.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select state, log from jobs where kind = %s and result = %s"
            " order by id desc limit 1",
            (KIND, int(result)),
        )
        found = cursor.fetchone()
    if found is None:
        return None
    state, log = str(found[0]), str(found[1] or "")
    if state == DONE and log.startswith(STOPPED):
        return {"state": "stopped"}
    if state == REFUSED:
        return {"state": MISMATCH}
    return {"state": state}


# --------------------------------------------------------------------- the check


@dataclass(frozen=True)
class Outcome:
    """How one spot check ended, and what it spent."""

    job: int
    state: str
    result: provisional.Result | None = None
    said: str = ""
    cpu_seconds: float = 0.0


STOPPED = provisional.SPOT_STOPPED
"""How a check that ran out of budget begins its log line (`of_result` reads it)."""


class _OverBudget(Exception):
    pass


class _Differs(Exception):
    def __init__(self, said: str) -> None:
        super().__init__(said)
        self.said = said


@dataclass
class _Budget:
    cpu: float
    wall: float
    started_cpu: float
    started_wall: float

    @classmethod
    def start(cls, cpu: float, wall: float) -> _Budget:
        return cls(cpu, wall, time.thread_time(), time.monotonic())

    @property
    def spent(self) -> float:
        return time.thread_time() - self.started_cpu

    def check(self) -> None:
        """Between two games: past either cap, the check stops here."""
        if self.spent >= self.cpu or time.monotonic() - self.started_wall >= self.wall:
            raise _OverBudget


def run(
    connection: Any,
    runs_dir: Path,
    job: int,
    worker: str,
    basis: Now,
    cpu_cap: float = quota.SAMPLE_JOB_CPU,
    wall_cap: float = quota.SAMPLE_JOB_WALL,
) -> Outcome:
    """Replay one result's sample and move the job, the result and perhaps the account.

    `basis` is the server's one-run basis for a game (`legality.one_run_basis`, which a
    test replaces with a smaller roster's); `cpu_cap` and `wall_cap` are the quota's, which
    a test lowers to see a check stop. A fault here is the server's, not the sender's: the
    job is `failed` and the result stays as it was.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select kind, hash, requester, state, result, sample from jobs where id = %s"
            " for update",
            (job,),
        )
        found = cursor.fetchone()
    if found is None:
        raise ValueError(f"no job numbered {job}")
    kind, game, requester, state, result, sealed = found
    if kind != KIND:
        raise ValueError(f"job {job} is a {kind} job, not a spot check")
    if state not in IN_FLIGHT:
        raise ValueError(f"job {job} is already {state}")
    if state == QUEUED:
        with connection.cursor() as cursor:
            cursor.execute(
                "update jobs set state = %s, claimed = now(), worker = %s,"
                " fingerprint = %s where id = %s",
                (RUNNING, worker, built_fingerprint(), job),
            )

    budget = _Budget.start(cpu_cap, wall_cap)
    with connection.cursor() as cursor:
        cursor.execute("savepoint spot_check")
    try:
        outcome = _check(
            connection, Path(runs_dir), job, str(game), int(requester), result, sealed,
            basis, budget,
        )
    except Exception as error:  # a fault is recorded on the job, never lost
        with connection.cursor() as cursor:
            cursor.execute("rollback to savepoint spot_check")
            cursor.execute("release savepoint spot_check")
        said = f"{type(error).__name__}: {error}"
        spent = budget.spent
        _finish(connection, job, FAILED, spent, error=said)
        return Outcome(job=job, state=FAILED, said=said, cpu_seconds=spent)
    with connection.cursor() as cursor:
        cursor.execute("release savepoint spot_check")
    spent = budget.spent
    if outcome.state == REFUSED:
        _finish(connection, job, REFUSED, spent, error=f"{MISMATCH}: {outcome.said}")
    else:
        _finish(connection, job, DONE, spent, log=outcome.said)
    return Outcome(
        job=job, state=outcome.state, result=outcome.result, said=outcome.said,
        cpu_seconds=spent,
    )


def _check(
    connection: Any,
    runs_dir: Path,
    job: int,
    game: str,
    requester: int,
    number: Any,
    sealed: Any,
    basis: Now,
    budget: _Budget,
) -> Outcome:
    if number is None or not isinstance(sealed, str):
        raise ValueError(f"job {job} names no result or no sample")
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {provisional._COLUMNS} from provisional where id = %s for update",
            (int(number),),
        )
        row = cursor.fetchone()
        cursor.execute("select spec from games where hash = %s", (game,))
        held = cursor.fetchone()
    if row is None:
        return Outcome(job=job, state=DONE, said="nothing to check: the result is gone")
    result = provisional._to_result(row)
    if result.state != LEGAL:
        return Outcome(
            job=job, state=DONE, said=f"nothing to check: the result is {result.state}"
        )
    spec = held[0] if held is not None else None
    current = basis(game, spec) if isinstance(spec, dict) else None
    if not isinstance(spec, dict) or current is None or result.basis.stale_against(current.in_browser(LEGAL)).needs:
        return Outcome(
            job=job, state=DONE, said="nothing to check: the result's stamp is out of date"
        )
    if result.directory is None:
        raise ValueError(f"result {result.id} has no records left to replay")
    where = runs_dir / result.directory
    stored = _Stored.read(where)
    roster = slices.canonical_config(
        agents=tuple(current.agents),
        games_per_pairing=current.games_per_pairing,
        seed=current.seeds[0],
        workers=1,
    )
    picks = [g for unit in slices.blunder_units(stored.records, seed=roster.seed)
             for g in unit.games]
    sample = draw(sealed, games=len(stored.records), picks=picks)
    stamped = result.basis.engine.rpartition("-")[2]
    try:
        replayed = _replay(spec, roster, stored, sample, budget)
    except _OverBudget:
        return Outcome(
            job=job,
            state=DONE,
            said=f"{STOPPED} after {budget.spent:.1f} CPU-seconds; the score stays as it "
            "was and waits for the site's own measurement",
        )
    except _Differs as differs:
        said = (
            f"{differs.said}; the site's build {built_fingerprint()}, the score's build "
            f"{stamped}"
        )
        print(f"spotcheck: result {result.id} for {game}: {said}", file=sys.stderr)
        moved = provisional.mismatched(connection, result.id)
        provisional.flag(connection, requester)
        return Outcome(job=job, state=REFUSED, result=moved, said=said)
    # `provisional.job` stays tier one's: it is how the sender's status finds the result,
    # and `jobs.result` is how this job names it.
    moved = provisional.spot_checked(connection, result.id, len(sample.games))
    return Outcome(job=job, state=DONE, result=moved, said=f"spot-checked: {replayed}")


@dataclass
class _Stored:
    """A result's records, as tier one stored them."""

    records: list[GameRecord]
    analysis: CorpusAnalysis

    @classmethod
    def read(cls, where: Path) -> _Stored:
        def text(name: str) -> str:
            path = where / name
            return path.read_text(encoding="utf-8") if path.is_file() else ""

        return cls(
            records=slices.parse_records(text(provisional.GAMES)),
            analysis=CorpusAnalysis.load(where / "analysis.json"),
        )


def _replay(
    spec: dict[str, Any],
    roster: TournamentConfig,
    stored: _Stored,
    sample: Sample,
    budget: _Budget,
) -> str:
    """Every sampled row played again and compared, cheapest signal first; what it did."""
    records = stored.records
    graded = [index for index, _ in slices._graded(records)]
    analysed_at = {index: n for n, index in enumerate(graded)}
    graded_set = set(graded)
    trials = {(trial.seed, tuple(trial.agents)): trial for trial in stored.analysis.blunders}
    for index in sample.games:
        budget.check()
        text = slices.run_unit(
            slices.Unit(slices.TOURNAMENT, 0, 1, (index,)), spec, config=roster
        )
        _same(_one(text), records[index].as_dict(), "the tournament's", index, records)
    probed = [index for index in sample.games if index in graded_set]
    for index in probed:
        budget.check()
        text = slices.run_unit(
            slices.Unit(slices.ANALYSIS, 0, 1, (index,)), spec,
            records=[records[index]], config=roster,
        )
        _same(
            _one(text), stored.analysis.games[analysed_at[index]].as_dict(),
            "the analysis of the tournament's", index, records,
        )
    for index in sample.trials:
        budget.check()
        text = slices.run_unit(
            slices.Unit(slices.BLUNDER, 0, 1, (index,)), spec,
            records=[records[index]], config=roster,
        )
        record = records[index]
        kept = trials.get((record.seed, tuple(record.agents)))
        lines = [line for line in text.splitlines() if line.strip()]
        mine = json.loads(lines[0]) if lines else None
        if (mine is None) != (kept is None):
            found = "no trial" if mine is None else "a trial"
            raise _Differs(
                f"the blunder trial on the tournament's game {index} (seed "
                f"{record.seed}): the site's replay found {found} and the score did not"
            )
        if mine is not None and kept is not None:
            _same(mine, kept.as_dict(), "the blunder trial on the tournament's", index,
                  records)
    return (
        f"{len(sample.games)} games, {len(probed)} analyses, {len(sample.trials)} trials "
        "equal"
    )


def _one(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"a replay of one game gave {len(lines)} lines")
    found: dict[str, Any] = json.loads(lines[0])
    return found


def _same(
    mine: Mapping[str, Any],
    theirs: Mapping[str, Any],
    what: str,
    index: int,
    records: Sequence[GameRecord],
) -> None:
    """Raise `_Differs` naming the game, its seed and where, unless the rows are equal."""
    left = {k: v for k, v in json.loads(json.dumps(dict(mine))).items() if k not in IGNORED}
    right = {k: v for k, v in json.loads(json.dumps(dict(theirs))).items() if k not in IGNORED}
    if left == right:
        return
    seed = records[index].seed if 0 <= index < len(records) else None
    where = f"{what} game {index} (seed {seed})"
    moves, sent = left.get("moves"), right.get("moves")
    if isinstance(moves, list) and isinstance(sent, list) and moves != sent:
        ply = next(
            (n for n, (a, b) in enumerate(zip(moves, sent, strict=False)) if a != b),
            min(len(moves), len(sent)),
        )
        raise _Differs(f"{where} parts from the site's replay at ply {ply}")
    field = next(
        name for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)
    )
    raise _Differs(f"{where} differs from the site's replay in `{field}`")


def _finish(
    connection: Any, job: int, state: str, cpu: float, error: str = "", log: str = ""
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, finished = clock_timestamp(), error = %s, log = %s,"
            " cpu_seconds = %s where id = %s",
            (state, error or None, log or None, round(max(cpu, 0.0), 3), job),
        )


def requeue(connection: Any) -> int:
    """Put back the spot checks a stopped process left `running`."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update jobs set state = %s, claimed = null, worker = null"
            " where kind = %s and state = %s",
            (QUEUED, KIND, RUNNING),
        )
        return int(cursor.rowcount or 0)
