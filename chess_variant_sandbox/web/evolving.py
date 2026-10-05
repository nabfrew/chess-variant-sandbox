"""Starting a design run from the browser, and watching it go.

A run is minutes to hours of tournaments and paid model calls, so it cannot happen
inside a request. It happens on a thread, writes its report into a ring buffer, and the
page polls. Nothing here duplicates the loop: it is `loop.evolve` with its `report`
callback pointed somewhere a browser can read.

One deliberate restriction: a run is started from a *chosen* ruleset, which the loop
measures on its own roster before revising it. A starting point scored somewhere else is
not comparable to anything the run will produce, and the archive must never contain that
comparison.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chess_variant_sandbox.loop import LoopConfig, evolve

# Enough scrollback to see what happened in the last few generations. A run's real
# record is the archive on disk; this is the window onto it while it is running.
LOG_LINES = 400


@dataclass
class Run:
    """One design run, in progress or finished."""

    id: str
    directory: Path
    theme: str
    parent: str
    generations: int
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    state: str = "running"  # running | finished | failed
    error: str = ""

    def report(self, line: str) -> None:
        for part in str(line).splitlines() or [""]:
            self.lines.append(part)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "directory": str(self.directory),
            "theme": self.theme,
            "parent": self.parent,
            "generations": self.generations,
            "state": self.state,
            "error": self.error,
            "log": list(self.lines),
        }


class Runs:
    """Design runs started from this server."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def start(
        self,
        run_id: str,
        spec_path: Path,
        parent_name: str,
        theme: str,
        directory: Path,
        generations: int,
        agents: tuple[str, ...] | None = None,
        judge: bool = True,
        games_per_pairing: int = 2,
    ) -> Run:
        run = Run(
            id=run_id,
            directory=directory,
            theme=theme,
            parent=parent_name,
            generations=generations,
        )
        config = LoopConfig(
            generations=generations,
            games_per_pairing=games_per_pairing,
            theme=theme,
            seeds=(spec_path,),
            judge=judge,
            directory=directory,
        )
        if agents:
            config = replace(config, agents=agents)
        run.report(f"starting from {parent_name}")
        run.report(f"theme: {theme or '(none given)'}")

        def target() -> None:
            try:
                evolve(config, report=run.report)
                run.state = "finished"
            except Exception as error:  # noqa: BLE001 - the browser is the only reader
                # A design run dies for ordinary reasons -- no API key, a network
                # failure, a proposal that will not validate -- and the person who
                # started it is looking at a web page, not a traceback in a terminal.
                run.state = "failed"
                run.error = f"{type(error).__name__}: {error}"
                run.report(run.error)
                run.report(traceback.format_exc(limit=3))

        thread = threading.Thread(target=target, name=f"evolve-{run_id}", daemon=True)
        with self._lock:
            self._runs[run_id] = run
        thread.start()
        return run

    def get(self, run_id: str) -> Run:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError("no such run")
        return run

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {k: v for k, v in run.as_dict().items() if k != "log"}
                for run in self._runs.values()
            ]
