"""Test, archive or promote, from the browser.

The three actions in `chess_variant_sandbox/corpus.py`, wired for the one place a person is actually
looking at a game they might want to keep. Two of them are a file write and return in
milliseconds; the first is a tournament and cannot happen inside a request, so it runs
on a thread with a state a page can poll -- the same shape as `evolving.Runs` and
`opponents.Ratings`, for the same reason.

Nothing here decides anything. The refusals -- untested, not viable, no reason given --
all live in `corpus.py`, so the browser and the CLI cannot come to different conclusions
about whether a game may enter the corpus.

A `Sweep` is the same measurement, over a list. `chess-variant-sandbox measure` brings the whole corpus
up to date from a terminal and was the only thing that could, which made the browser a
place where you could *see* that forty scores were stale and not a place where you could
do anything about it. It is one thread and one game at a time on purpose: the tournament
underneath already uses every worker it is given, and two sweeps racing for the same
`runs/<hash>-<seed>` directory would interleave two measurements into one.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox import corpus


@dataclass
class Measurement:
    """One canonical-roster measurement, in progress or finished."""

    ruleset_hash: str
    name: str
    state: str = "running"  # running | finished | failed
    error: str = ""
    result: dict[str, Any] | None = None
    started: float = field(default_factory=time.monotonic)
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=40))

    # True while another game holds the turn. The state stays `running`, so every page
    # that polls for `running` keeps polling while this one waits.
    waiting: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.ruleset_hash,
            "name": self.name,
            "state": self.state,
            "waiting": self.waiting,
            "error": self.error,
            # Elapsed rather than a count, and for the reason `RatingRun` gives: the
            # games run inside one call with nowhere to report back to.
            "seconds": round(time.monotonic() - self.started, 1),
            "result": self.result,
            "log": list(self.lines),
        }


@dataclass
class SweepItem:
    """One game's place in a sweep: what it needed when the sweep was planned, and what
    became of it.

    `index`, `viable` and `seeds` are the finished measurement's, and they travel
    together or not at all. A mean of four runs and a single run are not the same claim,
    so the item carries the seeds its number rests on rather than leaving the page to
    look them up somewhere else -- and `finish` keeps no number that arrives without
    them, or whose measurement does not read current.
    """

    ruleset_hash: str
    name: str
    needs: str  # what the survey said before anything ran -- rescore, reanalyse or replay
    state: str = "queued"  # queued | running | finished | failed | skipped
    error: str = ""
    index: float | None = None
    viable: bool | None = None
    # False only for *not measured* (plan-teacher §10.16), which the page says plainly.
    measured: bool | None = None
    seeds: tuple[int, ...] = ()
    seconds: float = 0.0

    def finish(self, result: dict[str, Any]) -> None:
        """Keep what a finished measurement says, if its basis still holds.

        `result` is `Measurement.as_dict`: the stored summary, its top-level `seeds` and
        its `staleness`. A measurement that does not read current, or that names no
        seeds, leaves the item with no number, which the page reads as *no score*.
        """
        self.state = "finished"
        staleness = result.get("staleness")
        seeds = result.get("seeds")
        if not isinstance(staleness, dict) or not staleness.get("current"):
            return
        if not isinstance(seeds, list) or not seeds:
            return
        self.index = result.get("index")
        self.viable = result.get("viable")
        self.measured = result.get("measured") is not False
        self.seeds = tuple(int(seed) for seed in seeds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.ruleset_hash,
            "name": self.name,
            "needs": self.needs,
            "state": self.state,
            "error": self.error,
            "index": self.index,
            "viable": self.viable,
            "measured": self.measured,
            # How many runs `index` is the mean of, and which. Empty exactly when there
            # is no number to go with it.
            "seeds": list(self.seeds),
            "seconds": round(self.seconds, 1),
        }


@dataclass
class Sweep:
    """A run of measurements over several games, in progress or finished.

    Stopping is cooperative and between games rather than inside one. A tournament killed
    half way through leaves a `runs/` directory that looks measured and is not, which is
    the exact failure `measure` exists to stop making -- so *stop* here means "do not
    start the next one", and the page says so where it offers it.
    """

    scope: str
    force: bool
    items: list[SweepItem]
    state: str = "running"  # running | finished | stopped
    started: float = field(default_factory=time.monotonic)

    @property
    def done(self) -> bool:
        return self.state != "running"

    def as_dict(self) -> dict[str, Any]:
        items = [item.as_dict() for item in self.items]
        return {
            "scope": self.scope,
            "force": self.force,
            "state": self.state,
            "seconds": round(time.monotonic() - self.started, 1),
            "total": len(items),
            "done": sum(1 for item in items if item["state"] not in ("queued", "running")),
            "failed": sum(1 for item in items if item["state"] == "failed"),
            "current": next((i for i in items if i["state"] == "running"), None),
            "items": items,
        }


class Measurements:
    """On-demand corpus-roster measurements, one per ruleset, on a thread."""

    def __init__(
        self,
        runs_dir: str | Path,
        ruleset_dir: str | Path = corpus.RULESET_DIR,
        archive_dir: str | Path = corpus.ARCHIVE_DIR,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        # Where `stored` looks a game up, to ask how many runs its rung owes.
        self.ruleset_dir = Path(ruleset_dir)
        self.archive_dir = Path(archive_dir)
        self._runs: dict[str, Measurement] = {}
        self._sweep: Sweep | None = None
        self._stopping = False
        self._lock = threading.Lock()

        # Held for the length of one game's measurement. A button on a game's page, a
        # button on a draft's page and a sweep all measure through `_measure`, so they
        # share one turn: one game at a time, whoever asked.
        self._turn = threading.Lock()

        # Wall-clock of the last measurement this process finished, for the designer's
        # statement of what the long rung costs on this machine.
        self.last_seconds: float | None = None

    def stored(self, ruleset_hash: str) -> dict[str, Any] | None:
        """A measurement from an earlier run, if there is one, with its staleness.

        Serialised rather than handed over whole, because everything downstream of here
        is JSON on its way to a browser. `as_dict` is what carries the staleness, so a
        stale measurement reaches the page *as* a stale measurement rather than as a
        number the page has no way to doubt.

        Read against the runs the game's rung owes, `corpus.expected_seeds`, as every other
        web reader is since Q488: a candidate's one run is current here, a library game's
        one run is not. A hash nobody can find gets the reader's default, which a
        measurement with no spec behind it reads stale under anyway.
        """
        try:
            candidate = corpus.find(
                ruleset_hash, runs_dir=self.runs_dir,
                ruleset_dir=self.ruleset_dir, archive_dir=self.archive_dir,
            )
        except corpus.CorpusError:
            stored = corpus.read_measurement(ruleset_hash, self.runs_dir)
        else:
            stored = corpus.read_measurement(
                ruleset_hash, self.runs_dir, spec=candidate.spec,
                seeds=corpus.expected_seeds(candidate.stage),
            )
        return None if stored is None else stored.as_dict()

    def get(self, ruleset_hash: str) -> Measurement | None:
        with self._lock:
            return self._runs.get(ruleset_hash)

    def start(
        self,
        candidate: corpus.Candidate,
        force: bool = False,
        seeds: tuple[int, ...] | None = None,
    ) -> Measurement:
        """Begin measuring this game, or return the run already doing it.

        ``force`` replays a tournament whose measurement is already current. Not the
        default and it must not become one: the reason to replay games that nothing has
        invalidated is that you doubt the machinery, which is a deliberate thing to do
        and costs the same minutes as the first measurement did.

        ``seeds`` is how many runs the score is the mean of, and ``None`` is whatever the
        game's rung expects. The designer's own measure rung leaves it alone and gets one
        run; the promotion panel asks for `corpus.LIBRARY_SEEDS`, which is four
        tournaments and is the rung before promotion.
        """
        with self._lock:
            existing = self._runs.get(candidate.hash)
            if existing is not None and existing.state == "running":
                return existing
            run = Measurement(ruleset_hash=candidate.hash, name=candidate.name)
            # Said before the thread has asked for the turn, because this is the answer
            # the button's press gets back: a run queued behind another reads queued from
            # its first moment, not *measuring* for the one poll before its thread looks.
            # `_measure` settles it either way.
            run.waiting = self._turn.locked()
            self._runs[candidate.hash] = run

        thread = threading.Thread(
            target=self._measure,
            args=(run, candidate, force, seeds),
            name=f"measure-{candidate.hash}",
            daemon=True,
        )
        thread.start()
        return run

    def _measure(
        self,
        run: Measurement,
        candidate: corpus.Candidate,
        force: bool = False,
        seeds: tuple[int, ...] | None = None,
    ) -> None:
        if not self._turn.acquire(blocking=False):
            run.waiting = True
            run.lines.append("waiting its turn: measurements run one game at a time")
            self._turn.acquire()
            run.started = time.monotonic()
        run.waiting = False
        try:
            # `refresh` rather than `measure`: a game whose games are still valid and
            # whose weights merely moved is re-scored in seconds instead of replaying a
            # tournament the button's owner is waiting on.
            run.result = corpus.refresh(
                candidate, runs_dir=self.runs_dir, report=run.lines.append, force=force,
                seeds=seeds,
            ).as_dict()
            run.state = "finished"
            self.last_seconds = time.monotonic() - run.started
        except Exception as error:  # noqa: BLE001 - the browser is the only reader
            # Same rule as a design run: the person who pressed the button is looking at
            # a web page, not a terminal, so the failure has to arrive as text.
            run.state = "failed"
            run.error = f"{type(error).__name__}: {error}"
            run.lines.append(run.error)
        finally:
            self._turn.release()

    # ------------------------------------------------------------ all of them at once

    def sweep(self) -> Sweep | None:
        """The sweep in progress, or the last one that ran. ``None`` before any has."""
        with self._lock:
            return self._sweep

    def start_sweep(
        self, candidates: list[tuple[corpus.Candidate, str]], scope: str, force: bool
    ) -> Sweep:
        """Measure these games, one at a time, on a thread.

        Returns the sweep already running if there is one rather than starting a second:
        two sweeps would be two threads writing the same `runs/<hash>-<seed>` directories.
        """
        with self._lock:
            if self._sweep is not None and not self._sweep.done:
                return self._sweep
            sweep = Sweep(
                scope=scope,
                force=force,
                items=[
                    SweepItem(ruleset_hash=candidate.hash, name=candidate.name, needs=needs)
                    for candidate, needs in candidates
                ],
            )
            self._sweep = sweep
            self._stopping = False

        thread = threading.Thread(
            target=self._run_sweep,
            args=(sweep, [candidate for candidate, _ in candidates], force),
            name="measure-sweep",
            daemon=True,
        )
        thread.start()
        return sweep

    def stop_sweep(self) -> Sweep | None:
        """Ask the sweep to stop after the game it is on. See `Sweep` for why not sooner."""
        with self._lock:
            self._stopping = True
            return self._sweep

    def _run_sweep(
        self, sweep: Sweep, candidates: list[corpus.Candidate], force: bool
    ) -> None:
        for item, candidate in zip(sweep.items, candidates, strict=True):
            with self._lock:
                stopping = self._stopping
            if stopping:
                item.state = "skipped"
                continue

            started = time.monotonic()
            item.state = "running"
            # Through the same per-game record the single button uses, so a game the
            # sweep is measuring shows as measuring on its own page, with its own log,
            # instead of looking untouched until the whole sweep ends.
            run = Measurement(ruleset_hash=candidate.hash, name=candidate.name)
            with self._lock:
                self._runs[candidate.hash] = run
            self._measure(run, candidate, force)

            item.seconds = time.monotonic() - started
            if run.state == "failed":
                # One game that cannot be measured does not stop the other forty. The
                # failure is kept on the item and counted, because a sweep that reported
                # "finished" over three silent failures would be worse than no sweep.
                item.state = "failed"
                item.error = run.error
                continue
            item.finish(run.result or {})

        with self._lock:
            stopped = self._stopping
        sweep.state = "stopped" if stopped else "finished"
