"""The refresher: the library kept on the current scorer, one game at a time (EV5).

`docs/plan-evolving-score.md` §6. A revision of the scorer (a weights file, a metrics edit,
a roster change, a new player generation) makes estimates *provisional* at once and costs
nothing; this is what makes them *current* again, with nobody starting a campaign.
`chess-variant-sandbox refresh --library --watch` loops:

1. **What is owed**, read as `measure --check` reads it (`corpus.survey`), at the
   refresher's own commit, over the main checkout's `runs/`. No lease.
2. **The order** (§6.2): cheapest repair first (`refit`, `rescore`, `reanalyse`,
   `replay`), then value within a repair kind (`value_tier`), then the oldest estimate,
   then the hash, so the order is reproducible. The order decides *when* a number
   arrives, never *what* it is: every repair is `corpus.refresh` on one game, a pure
   function of the game's records and the code (`CLAUDE.md`, *Determinism*).
3. **One game per lease**: the lease that game's repair needs (`LEASE_FOR`), held for
   that game alone, so a coordinator or a person can take the machine between games. It
   does not start while another worktree holds a `heavy` or `quiet` lease
   (`Machine.busy`): a replay or a training run has the machine, and the refresher waits.
4. **Its own worktree, pinned at a commit.** The refresher never measures from the main
   checkout (`CLAUDE.md`: never edit the package under a running campaign; a refresher
   is a permanent one). Each game is repaired by a fresh process in that worktree, so it
   imports the code of exactly one commit. Between games, and only between games, it
   checks `main`; if `main` moved it fast-forwards, rebuilds the core under a `build`
   lease if `rust/` moved, and goes on. No game spans two commits (`GitMachine.repair`
   checks the commit before and after, and refuses a game whose tree moved under it).
5. **Delivery through `history.deliver`** after each game, so the main checkout's
   `runs/` is current within one game and what a repair replaces is kept in the history
   ledger. It never deploys.
6. **A refusal stops that game and is never retried**: a stale build (which stops the
   whole loop until `main` moves, since every game would refuse), a fit that cannot be
   reproduced on this core, a gate the repair newly fails, or a repair that raised. It is
   logged, the game is skipped, and the next one is taken. A refused game comes back only
   when its stored measurement changes (somebody repaired it) or a person clears it
   (`refresh --forget <hash>`).

**What it writes**, beside the measurements in the main checkout's `runs/`:
`refresh-status.json` (what it is doing, what is owed, the last game, the last refusal,
every standing refusal), which `/admin` and the local application read through
`read_status`, and `refresh-log.jsonl`, one line per game or event. Nothing under
`metrics/` reads either, and this module is in no fingerprint set (`tests/test_refresher.py`):
running the refresher moves no score by itself.

Every side effect (git, leases, processes, the clock) goes through `Machine`, so the loop's
logic is tested with a fake; `GitMachine` is the real one.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

STATUS_FILE = "refresh-status.json"
LOG_FILE = "refresh-log.jsonl"
# The branch and directory the refresher's worktree lives on, under the main checkout.
BRANCH = "refresher"
WORKTREE = Path(".claude") / "worktrees" / "refresher"

# §6.2's first key: what each repair costs. A refit plays the fitting procedure and no
# game; a re-score reads stored games; a re-analysis runs the pass over them first; a
# replay plays every game again.
REPAIR_ORDER = ("refit", "rescore", "reanalyse", "replay")

# The lease each repair takes (§6.1): `heavy` for anything that plays or analyses,
# `quiet` for a re-score, whose bootstrap takes the whole machine for its seconds.
LEASE_FOR = {"refit": "heavy", "rescore": "quiet", "reanalyse": "heavy", "replay": "heavy"}

# Leases another worktree may hold that keep the refresher waiting: a replay, a training
# run, a campaign or a timed bench has the machine (the task's *never while a replay or
# another heavy job holds the machine*). A build or the fast suite does not: the lease's
# own slot arithmetic decides whether a game fits beside them.
YIELDS_TO = ("heavy", "quiet")

# §6.2's value, within one repair kind. The launch set's nine games, by name
# (`docs/plan-launch-set.md` §0 and §3): the games a visitor meets first. The site
# features nothing else by code today; when it does, its list joins this one.
LAUNCH_SET = (
    "dobutsu",
    "los alamos",
    "cylinder chess",
    "three-check",
    "crazyhouse",
    "quarry",
    "reckoning",
    "wolves",
    "hex glinski hexagon",
)
# Tier 0 the launch set; 1 the top quarter of the leaderboard by its last number, current
# or provisional; 2 within one interval of a gate; 3 the rest.
TOP_SHARE = 0.25

# Refusals by kind, so the loop can tell one that stops it from one that skips a game.
STALE_BUILD = "stale build"
FIT_NOT_REPRODUCED = "fit not reproduced"
GATE_NEWLY_FAILED = "gate newly failed"
COMMIT_MOVED = "commit moved"
CHANGED_COMMITTED_FILES = "changed committed files"
ERROR = "error"
HALTS = (STALE_BUILD,)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------- the order


@dataclass(frozen=True)
class Stored:
    """What a game's stored file says, current or not: the last number a reader shows.

    `ScoringStatus` withholds a stale number on purpose (it is no *current* number); the
    order needs the provisional one too, to rank by it, and never shows or compares it.
    """

    index: float | None = None
    viable: bool | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # The runs disagreed on the verdict: the nearest a stored file comes to saying the
    # game sits on a gate's edge.
    split: bool = False
    measured: str = ""
    # The bytes of `canonical.json`, hashed, or "absent": what a refusal is keyed on.
    digest: str = "absent"


def stored_of(digest: str, runs_dir: Path) -> Stored:
    """`runs/<hash>/canonical.json` read raw, whatever its basis."""
    path = Path(runs_dir) / digest / "canonical.json"
    try:
        body = path.read_bytes()
        data = json.loads(body)
    except (OSError, ValueError):
        return Stored()
    if not isinstance(data, dict):
        return Stored()
    raw = data.get("verdict")
    verdict: dict[str, Any] = raw if isinstance(raw, dict) else {}
    runs = [run for run in data.get("runs") or [] if isinstance(run, dict)]
    viable_runs = {run.get("viable") for run in runs if run.get("viable") is not None}
    split = (bool(verdict.get("accept")) and bool(verdict.get("refuse"))) or len(viable_runs) > 1
    stamp = os.path.getmtime(path)
    return Stored(
        index=_number(data.get("index")),
        viable=data.get("viable") if isinstance(data.get("viable"), bool) else None,
        ci_low=_number(data.get("ci_low")),
        ci_high=_number(data.get("ci_high")),
        split=split,
        measured=datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        digest=hashlib.sha256(body).hexdigest()[:16],
    )


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class Item:
    """One game the refresher owes, and where it stands in the queue."""

    hash: str
    name: str
    needs: str
    reason: str
    tier: int
    measured: str
    stored: str  # `Stored.digest` when the queue was read

    @property
    def key(self) -> tuple[int, int, str, str]:
        # A game never measured has no date and sorts as the oldest.
        return (REPAIR_ORDER.index(self.needs), self.tier, self.measured, self.hash)

    @property
    def lease(self) -> str:
        return LEASE_FOR[self.needs]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "lease": self.lease}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Item:
        return cls(
            hash=str(data["hash"]), name=str(data.get("name", "")), needs=str(data["needs"]),
            reason=str(data.get("reason", "")), tier=int(data.get("tier", 3)),
            measured=str(data.get("measured", "")), stored=str(data.get("stored", "absent")),
        )


def _normal(name: str) -> str:
    return " ".join(name.lower().split())


def value_tier(name: str, digest: str, top: set[str], stored: Stored) -> int:
    """§6.2's value, as a tier (lower first)."""
    if _normal(name) in LAUNCH_SET:
        return 0
    if digest in top:
        return 1
    if stored.split:
        return 2
    return 3


def top_quarter(stored: Mapping[str, Stored]) -> set[str]:
    """The games in the top quarter of the leaderboard by their last number.

    Only games that hold a number and were not gated out, current or provisional alike:
    it is an order, so a provisional number may rank here and nowhere a reader looks.
    """
    ranked = sorted(
        (reading.index, digest) for digest, reading in stored.items()
        if reading.index is not None and reading.viable is not False
    )
    ranked.reverse()
    take = math.ceil(len(ranked) * TOP_SHARE)
    return {digest for _index, digest in ranked[:take]}


def order(statuses: Iterable[Any], stored: Mapping[str, Stored]) -> list[Item]:
    """The queue: every game not current, in §6.2's order.

    ``statuses`` are `corpus.ScoringStatus` (or anything with ``hash``, ``name``,
    ``current``, ``needs`` and ``reason``). The top quarter is read over every game
    surveyed, current ones included, since they are on the same leaderboard.
    """
    statuses = list(statuses)
    top = top_quarter(stored)
    items = []
    for status in statuses:
        if status.current or status.needs not in REPAIR_ORDER:
            continue
        reading = stored.get(status.hash, Stored())
        items.append(Item(
            hash=status.hash, name=status.name, needs=status.needs, reason=status.reason,
            tier=value_tier(status.name, status.hash, top, reading),
            measured=reading.measured, stored=reading.digest,
        ))
    return sorted(items, key=lambda item: item.key)


def queue(runs_dir: Path, weights: str | None = None) -> list[Item]:
    """What this tree owes over ``runs_dir``, in order. Reads; spends nothing."""
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS

    statuses = corpus.survey(runs_dir=Path(runs_dir), weights=weights or DEFAULT_WEIGHTS)
    stored = {status.hash: stored_of(status.hash, runs_dir) for status in statuses}
    return order(statuses, stored)


# ------------------------------------------------------------------- one game, here


@dataclass(frozen=True)
class Outcome:
    """What repairing one game did. ``refusal`` is ``""`` for a repair that stands."""

    hash: str
    needs: str
    refusal: str = ""
    reason: str = ""
    index: float | None = None
    viable: bool | None = None
    commit: str = ""
    seconds: float = 0.0
    lines: tuple[str, ...] = ()

    @property
    def refused(self) -> bool:
        return bool(self.refusal)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["lines"] = list(self.lines)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Outcome:
        return cls(
            hash=str(data["hash"]), needs=str(data.get("needs", "")),
            refusal=str(data.get("refusal", "")), reason=str(data.get("reason", "")),
            index=_number(data.get("index")),
            viable=data.get("viable") if isinstance(data.get("viable"), bool) else None,
            commit=str(data.get("commit", "")), seconds=float(data.get("seconds") or 0.0),
            lines=tuple(str(line) for line in data.get("lines") or ()),
        )


def repair(
    digest: str,
    runs_dir: Path,
    weights: str | None = None,
    workers: int = 0,
    say: Callable[[str], None] | None = None,
) -> Outcome:
    """Repair one game in this tree, or say why not: what `refresh <hash>` runs.

    `corpus.refresh` does the work, exactly as `measure <hash>` would; this adds the three
    refusals §6.3 names, each checked where it can be known. A stale build is read before
    anything runs. A stored fit that this core does not vouch for is checked first, and one
    that does not reproduce (or does not exist) is refused before a game is played on it,
    since a refit is a new player and a person decides whether to adopt one. A gate the
    stored file passed and the new one fails is refused after the repair, and the
    supervisor then delivers nothing.
    """
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.measurement import stale_build
    from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
    from chess_variant_sandbox.tournament.fitting import FITTED, REFIT

    weights = weights or DEFAULT_WEIGHTS
    lines: list[str] = []
    needs = ""

    def note(line: str) -> None:
        lines.append(line)
        if say is not None:
            say(line)

    started = time.monotonic()

    def done(needs: str, refusal: str = "", reason: str = "", **rest: Any) -> Outcome:
        return Outcome(hash=digest, needs=needs, refusal=refusal, reason=reason,
                       seconds=round(time.monotonic() - started, 1), lines=tuple(lines),
                       **rest)

    behind = stale_build()
    if behind is not None:
        return done("", STALE_BUILD, behind)
    try:
        candidate = corpus.find(digest, runs_dir=runs_dir)
        before = stored_of(digest, runs_dir)
        stored = corpus.read_measurement(
            digest, runs_dir, weights, candidate.spec, corpus.expected_seeds(candidate.stage)
        )
        needs = "replay" if stored is None else stored.staleness.needs
        if stored is not None and stored.current:
            note(f"{candidate.name}: already current")
            return done("", index=_number(stored.stored.get("index")),
                        viable=stored.stored.get("viable"))
        if stored is not None and stored.staleness.refit:
            found = corpus.check_fit(candidate.spec, workers=workers)
            if found.outcome in (REFIT, FITTED):
                why = (
                    "no stored fit to reproduce" if found.outcome == FITTED
                    else "the stored fit does not reproduce on this core: "
                    + "; ".join(str(failure) for failure in found.failures)
                )
                return done(needs, FIT_NOT_REPRODUCED, why)
        measured = corpus.refresh(candidate, runs_dir, weights, workers=workers, report=note)
    except Exception as error:  # noqa: BLE001 - every failure is a logged refusal
        return done(needs, ERROR, f"{type(error).__name__}: {error}")
    index = _number(measured.stored.get("index"))
    viable = measured.stored.get("viable")
    if before.viable is True and viable is False:
        violations = "; ".join(str(v) for v in measured.stored.get("violations") or ())
        return done(needs, GATE_NEWLY_FAILED,
                    "the stored measurement passed its gates and the repair fails "
                    + (violations or "one"), index=index, viable=False)
    return done(needs, index=index, viable=viable if isinstance(viable, bool) else None)


# ------------------------------------------------------------------- what it writes


@dataclass
class Ledger:
    """The status file and the log, in the main checkout's `runs/`."""

    runs_dir: Path
    status: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.status = read_status(self.runs_dir) or {}
        self.status.setdefault("refusals", [])
        self.status.setdefault("owed_merge", [])

    @property
    def refusals(self) -> list[dict[str, Any]]:
        return list(self.status.get("refusals") or [])

    def refused(self, item: Item) -> bool:
        """Whether this game was refused with the stored file it has now."""
        return any(r.get("hash") == item.hash and r.get("stored") == item.stored
                   for r in self.refusals)

    def refuse(self, item: Item, outcome: Outcome) -> dict[str, Any]:
        record = {"hash": item.hash, "name": item.name, "needs": item.needs,
                  "stored": item.stored, "kind": outcome.refusal, "reason": outcome.reason,
                  "commit": outcome.commit, "at": _now()}
        kept = [r for r in self.refusals if r.get("hash") != item.hash]
        self.status["refusals"] = [*kept, record]
        self.status["last_refusal"] = record
        return record

    def lapse(self, owed: Iterable[Item]) -> list[dict[str, Any]]:
        """Drop the refusals that no longer stand, and return them.

        A refusal is of one stored file (`refused`): once the game's file has changed, or
        the game is owed nothing at all, it refuses nothing, and a standing count or an
        `/admin` row that still showed it would say a game is held back that is not
        (Q693r's first real run). ``owed`` is the whole survey, before `--only`.
        """
        still = {(item.hash, item.stored) for item in owed}
        kept = [r for r in self.refusals if (r.get("hash"), r.get("stored")) in still]
        gone = [r for r in self.refusals if (r.get("hash"), r.get("stored")) not in still]
        if gone:
            self.status["refusals"] = kept
        return gone

    def forget(self, digest: str) -> bool:
        kept = [r for r in self.refusals if r.get("hash") != digest]
        found = len(kept) != len(self.refusals)
        self.status["refusals"] = kept
        self.write()
        return found

    def update(self, **fields: Any) -> None:
        self.status.update(fields)
        self.write()

    def write(self) -> None:
        self.status["updated"] = _now()
        self.status["pid"] = os.getpid()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / STATUS_FILE
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(self.status, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(partial, path)

    def log(self, event: str, **fields: Any) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        line = {"at": _now(), "event": event, **fields}
        with (self.runs_dir / LOG_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


def read_status(runs_dir: Path) -> dict[str, Any] | None:
    """The refresher's status file, for `/admin` and the local application, or ``None``.

    Read as written, plus ``age_s``: how long since the refresher last wrote it, so a page
    can say a refresher that stopped writing is not running whatever its last state says.
    """
    path = Path(runs_dir) / STATUS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        updated = datetime.datetime.fromisoformat(str(data.get("updated")))
        data["age_s"] = round((datetime.datetime.now(datetime.timezone.utc) - updated)
                              .total_seconds())
    except ValueError:
        data["age_s"] = None
    return data


def read_log(runs_dir: Path) -> list[dict[str, Any]]:
    path = Path(runs_dir) / LOG_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def owed_counts(items: Iterable[Item]) -> dict[str, int]:
    counts = {needs: 0 for needs in REPAIR_ORDER}
    for item in items:
        counts[item.needs] += 1
    return counts


# ------------------------------------------------------------------------ the loop


class Machine(Protocol):
    """Everything the loop does outside itself. `GitMachine` is the real one."""

    def head(self) -> str: ...  # the worktree's commit
    def behind(self) -> bool: ...  # main has commits the worktree lacks
    def advance(self) -> list[str]: ...  # move the worktree to main; the paths that changed
    def rebuild(self) -> str | None: ...  # rebuild the core; a refusal, or None
    def busy(self) -> str | None: ...  # why another job has the machine, or None
    def acquire(self, profile: str, tag: str) -> bool: ...
    def release(self) -> None: ...
    def survey(self) -> list[Item]: ...  # the queue, at the worktree's commit
    def repair(self, item: Item) -> Outcome: ...  # one game, at the worktree's commit
    def deliver(self, item: Item) -> None: ...  # `history.deliver` into main's runs/
    def changed(self) -> list[str]: ...  # committed files the last repair changed
    def keep(self, item: Item, paths: list[str]) -> str: ...  # commit them; the sha
    def discard(self, paths: list[str]) -> None: ...  # put them back
    def unmerged(self, commits: list[str]) -> list[str]: ...  # those main does not hold
    def sleep(self, seconds: float) -> None: ...


# A repair may change committed files under this directory alone: a fit reproduced on
# this core adds its sidecar (`tournament/fitting.py`), which the main checkout needs too
# before it reads the delivered file as current. Anything else changed is refused.
KEPT_CHANGES = ("fitted/",)


def rebuild_needed(paths: Iterable[str]) -> bool:
    """Whether a move of `main` touched what the core is built from."""
    return any(path.startswith("rust/") or path in ("pyproject.toml", "uv.lock")
               for path in paths)


@dataclass
class Refresher:
    """The loop of §6.1, over a `Machine`, writing a `Ledger`."""

    machine: Machine
    ledger: Ledger
    watch: bool = False
    interval: float = 600.0
    say: Callable[[str], None] = lambda _line: None
    # Stops a watching loop after this many passes through it; tests only.
    passes: int | None = None
    # Only these games (`refresh --only`), for a first run on one small game.
    only: frozenset[str] | None = None

    def run(self) -> int:
        """Until nothing is owed, or forever when watching. 1 if it stopped on a halt.

        Without ``watch`` it is one pass over the queue: a busy machine or a lease not yet
        granted is waited for, and it returns when nothing is owed (0) or it halted (1).
        """
        self.ledger.update(state="starting", since=_now(), doing=None,
                           commit=self.machine.head())
        turns = 0
        try:
            while self.passes is None or turns < self.passes:
                turns += 1
                state = self.step()
                if not self.watch and state in ("idle", "halted"):
                    return 0 if state == "idle" else 1
                if state in ("idle", "halted", "waiting"):
                    self.machine.sleep(self.interval)
        except KeyboardInterrupt:
            self.say("interrupted")
            self.ledger.log("interrupted")
        finally:
            self.ledger.update(state="stopped", doing=None)
        return 0

    def step(self) -> str:
        """One turn of the loop: catch up with main, then at most one game."""
        halted = self.ledger.status.get("state") == "halted"
        if self.machine.behind():
            moved = self.machine.advance()
            self.ledger.log("advanced", commit=self.machine.head(), paths=len(moved))
            self.say(f"main moved: now at {self.machine.head()[:12]}")
            if rebuild_needed(moved) or halted:
                self.say("rebuilding the core")
                started = time.monotonic()
                refusal = self._rebuild()
                if refusal is not None:
                    return self._halt(refusal)
                # Logged, since nothing else says the core was rebuilt (Q693r).
                seconds = round(time.monotonic() - started, 1)
                self.ledger.log("rebuilt", commit=self.machine.head(), seconds=seconds)
                self.say(f"rebuilt the core in {seconds:.0f}s")
            halted = False
        if halted:
            return "halted"

        # A fit commit main now holds is owed no merge: without this `owed_merge` only
        # grows, and `/admin` asks for a merge that was made (Q693r's first real run).
        owed_merge = list(self.ledger.status.get("owed_merge") or [])
        if owed_merge:
            still = self.machine.unmerged(owed_merge)
            if still != owed_merge:
                self.ledger.log("merged", commits=[c for c in owed_merge if c not in still])
                self.ledger.update(owed_merge=still)

        why = self.machine.busy()
        if why is not None:
            self.ledger.update(state="waiting", waiting_for=why, doing=None)
            self.say(f"waiting: {why}")
            return "waiting"

        try:
            items = self.machine.survey()
        except Exception as error:  # noqa: BLE001 - logged; main moving is the way on
            return self._halt(f"the survey failed: {error}", kind=ERROR)
        for record in self.ledger.lapse(items):
            self.ledger.log("lapsed", hash=record.get("hash"), name=record.get("name"),
                            kind=record.get("kind"), stored=record.get("stored"))
        if self.only is not None:
            items = [item for item in items if item.hash in self.only]
        todo = [item for item in items if not self.ledger.refused(item)]
        self.ledger.update(
            commit=self.machine.head(), waiting_for=None,
            owed=[item.as_dict() for item in todo], owed_counts=owed_counts(todo),
            refused_owed=len(items) - len(todo),
        )
        if not todo:
            self.ledger.update(state="idle", doing=None)
            return "idle"

        item = todo[0]
        if not self.machine.acquire(item.lease, f"refresh {item.hash}"):
            self.ledger.update(state="waiting", waiting_for=f"a {item.lease} lease",
                               doing=None)
            return "waiting"
        self.ledger.update(state="working", doing={**item.as_dict(), "started": _now()})
        self.say(f"{item.name} [{item.hash}]: {item.needs}")
        try:
            outcome = self.machine.repair(item)
        finally:
            self.machine.release()
        return self._settle(item, outcome)

    def _settle(self, item: Item, outcome: Outcome) -> str:
        changed = self.machine.changed()
        foreign = [path for path in changed if not path.startswith(KEPT_CHANGES)]
        if not outcome.refused and foreign:
            outcome = replace(
                outcome, refusal=CHANGED_COMMITTED_FILES,
                reason="the repair changed committed files: " + ", ".join(foreign),
            )
        if outcome.refused:
            if changed:
                self.machine.discard(changed)
            if outcome.refusal in HALTS:
                return self._halt(outcome.reason, item, outcome, kind=outcome.refusal)
            record = self.ledger.refuse(item, outcome)
            self.ledger.log("refused", **record)
            self.ledger.update(state="working", doing=None)
            self.say(f"  refused ({outcome.refusal}): {outcome.reason}")
            return "refused"
        try:
            self.machine.deliver(item)
        except Exception as error:  # noqa: BLE001 - logged; nothing is retried
            return self._halt(f"the delivery failed: {error}", item, kind=ERROR)
        kept = ""
        if changed:
            kept = self.machine.keep(item, changed)
            self.ledger.status["owed_merge"] = [*self.ledger.status["owed_merge"], kept]
        last = {"hash": item.hash, "name": item.name, "needs": item.needs,
                "index": outcome.index, "viable": outcome.viable, "commit": outcome.commit,
                "seconds": outcome.seconds, "at": _now(), **({"kept": kept} if kept else {})}
        self.ledger.log("repaired", **last)
        self.ledger.update(state="working", doing=None, last_done=last)
        self.say(f"  delivered in {outcome.seconds:.0f}s")
        return "repaired"

    def _rebuild(self) -> str | None:
        if not self.machine.acquire("build", "refresh rebuild"):
            return "no build lease for the rebuild"
        try:
            return self.machine.rebuild()
        finally:
            self.machine.release()

    def _halt(self, reason: str, item: Item | None = None,
              outcome: Outcome | None = None, kind: str = STALE_BUILD) -> str:
        """Stop taking games until `main` moves: what every game would refuse, or a
        survey or a delivery that failed, is not one game's to skip."""
        record = {"kind": kind, "reason": reason, "at": _now(),
                  "commit": self.machine.head(),
                  **({"hash": item.hash, "name": item.name} if item else {})}
        self.ledger.log("halted", **record)
        self.ledger.update(state="halted", doing=None, last_refusal=record,
                           waiting_for="main to move (a rebuild is tried then)")
        self.say(f"halted: {reason}")
        return "halted"


# --------------------------------------------------------------- the real machine


def _python(worktree: Path) -> Path:
    """The worktree's own interpreter, so a game imports that tree's code and core."""
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    name = "python.exe" if sys.platform == "win32" else "python"
    return worktree / ".venv" / scripts / name


@dataclass
class GitMachine:
    """The refresher's worktree, the lease script and child processes.

    ``main_checkout`` holds `runs/` and the lease directory; ``worktree`` is the
    refresher's own checkout on branch `BRANCH`. A game is repaired in the worktree's own
    `runs/` (gitignored): its stored files are copied in from the main checkout first, the
    repair runs there, and `history.deliver` brings the result back.
    """

    main_checkout: Path
    worktree: Path
    weights: str = ""
    workers: int = 0
    wait_s: int = 3600
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    # The `runs/` read and delivered into: the main checkout's unless `refresh --runs`.
    runs_dir: Path | None = None

    @property
    def main_runs(self) -> Path:
        return Path(self.runs_dir) if self.runs_dir is not None else self.main_checkout / "runs"

    @property
    def staging(self) -> Path:
        return self.worktree / "runs"

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        return self._git_raw(*args, cwd=cwd).strip()

    def _git_raw(self, *args: str, cwd: Path | None = None) -> str:
        done = self.run(["git", *args], cwd=cwd or self.worktree, capture_output=True,
                        text=True, check=True)
        return done.stdout

    def _lease(self, *args: str) -> subprocess.CompletedProcess[str]:
        script = self.main_checkout / ".claude" / "scripts" / "compute_lease.py"
        return self.run([sys.executable, str(script), *args], cwd=self.worktree,
                        capture_output=True, text=True, check=False)

    def _child(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run([str(_python(self.worktree)), "-m", "chess_variant_sandbox.cli", *args],
                        cwd=self.worktree, capture_output=True, text=True, check=False)

    def exists(self) -> bool:
        return (self.worktree / ".git").exists()

    def create(self) -> None:
        """Make the worktree on `BRANCH`, locked so `prune_worktrees.py` never takes it.

        Git alone; its core is built after, under the worktree's own `build` lease
        (`rebuild`), since a lease belongs to the worktree it is taken from.
        """
        branches = self._git("branch", "--list", BRANCH, cwd=self.main_checkout)
        add = ["worktree", "add", "--lock"]
        add += ([str(self.worktree), BRANCH] if branches
                else ["-b", BRANCH, str(self.worktree), "main"])
        self._git(*add, cwd=self.main_checkout)

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def behind(self) -> bool:
        # Not `main != HEAD`: after a merge of main into fit commits of its own, HEAD is
        # ahead of main and equal to nothing, and the worktree is up to date.
        done = self.run(["git", "merge-base", "--is-ancestor", "main", "HEAD"],
                        cwd=self.worktree, capture_output=True, text=True, check=False)
        return done.returncode != 0

    def advance(self) -> list[str]:
        before = self.head()
        own = self._git("rev-list", "main..HEAD")
        if own:
            # Fit sidecars committed here and not yet merged: a merge, never a rewrite.
            self._git("merge", "--no-edit", "main")
        else:
            self._git("merge", "--ff-only", "main")
        changed = self._git("diff", "--name-only", before, "HEAD")
        return [line for line in changed.splitlines() if line]

    def rebuild(self) -> str | None:
        done = self.run(
            ["uv", "sync", "--extra", "dev", "--reinstall-package", "chess-variant-sandbox"],
            cwd=self.worktree, capture_output=True, text=True, check=False,
        )
        if done.returncode != 0:
            return f"the rebuild failed: {(done.stderr or done.stdout).strip()[-400:]}"
        return None

    def busy(self) -> str | None:
        done = self._lease("status", "--json")
        try:
            leases = json.loads(done.stdout).get("leases") or []
        except (ValueError, AttributeError):
            return None
        for lease in leases:
            if (lease.get("profile") in YIELDS_TO
                    and Path(str(lease.get("worktree"))).resolve() != self.worktree.resolve()):
                return (f"a {lease['profile']} lease is held by "
                        f"{Path(str(lease['worktree'])).name} ({lease.get('tag', '')})")
        return None

    def acquire(self, profile: str, tag: str) -> bool:
        return self._lease("acquire", "--profile", profile, "--tag", tag,
                           "--wait", str(self.wait_s)).returncode == 0

    def release(self) -> None:
        self._lease("release")

    def survey(self) -> list[Item]:
        args = ["refresh", "--plan", "--json", "--runs", str(self.main_runs)]
        done = self._child(*args, *(["--weights", self.weights] if self.weights else []))
        if done.returncode != 0:
            raise RuntimeError(f"the survey failed: {done.stderr.strip()[-400:]}")
        return [Item.from_dict(row) for row in json.loads(done.stdout)]

    def stage(self, digest: str) -> None:
        """Copy one game's stored measurement and runs from the main checkout's `runs/`."""
        stage_game(digest, self.main_runs, self.staging)

    def repair(self, item: Item) -> Outcome:
        commit = self.head()
        self.stage(item.hash)
        args = ["refresh", item.hash, "--json", "--runs", str(self.staging),
                "--workers", str(self.workers)]
        done = self._child(*args, *(["--weights", self.weights] if self.weights else []))
        after = self.head()
        try:
            outcome = Outcome.from_dict(json.loads(done.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError, KeyError):
            outcome = Outcome(hash=item.hash, needs=item.needs, refusal=ERROR,
                              reason=f"the repair process said nothing readable "
                                     f"(exit {done.returncode}): {done.stderr.strip()[-400:]}")
        outcome = replace(outcome, commit=commit)
        if after != commit:
            return Outcome(hash=item.hash, needs=item.needs, refusal=COMMIT_MOVED,
                           reason=f"the worktree moved from {commit[:12]} to {after[:12]} "
                                  "during the repair", commit=commit)
        return outcome

    def deliver(self, item: Item) -> None:
        done = self.run(
            [str(_python(self.worktree)), "-m", "chess_variant_sandbox.history", "deliver",
             str(self.staging), str(self.main_runs), item.hash],
            cwd=self.worktree, capture_output=True, text=True, check=False,
        )
        if done.returncode != 0:
            raise RuntimeError(f"the delivery failed: {done.stderr.strip()[-400:]}")

    def changed(self) -> list[str]:
        # Unstripped: a porcelain line starts with its two status columns, often a space.
        text = self._git_raw("status", "--porcelain", "--untracked-files=all")
        return sorted(line[3:].strip() for line in text.splitlines() if line.strip())

    def keep(self, item: Item, paths: list[str]) -> str:
        self._git("add", "--", *paths)
        self._git("commit", "-m", f"refresh: fit evidence for {item.name} [{item.hash}]",
                  "-m", "Written by `chess-variant-sandbox refresh --library` (EV5); merge "
                        "into main so its checkout reads the delivered measurement as current.")
        return self.head()

    def discard(self, paths: list[str]) -> None:
        self._git("checkout", "--", ".")
        self._git("clean", "-fd", "--", *paths)

    def unmerged(self, commits: list[str]) -> list[str]:
        """The commits the worktree's branch holds and main does not yet. One the branch no
        longer holds (it was reset away) is never merged by merging the branch, and is
        owed nothing either."""

        def holds(ref: str, commit: str) -> bool:
            return self.run(["git", "merge-base", "--is-ancestor", commit, ref],
                            cwd=self.worktree, capture_output=True, text=True,
                            check=False).returncode == 0

        return [c for c in commits if holds("HEAD", c) and not holds("main", c)]

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def stage_game(digest: str, source_runs: Path, staging: Path) -> list[str]:
    """Copy what a repair of one game reads from ``source_runs`` into ``staging``.

    The game's `runs/<hash>/` files (its `canonical.json` and index files; not its history
    or `previous/`, which stay the main checkout's) and the `runs/<hash>-<seed>/` of every
    seed the stored file names. Whatever ``staging`` held of the game before is removed
    first, so a repair never reads a leftover. Returns what was copied, relative.
    """
    source_runs, staging = Path(source_runs), Path(staging)
    for old in [staging / digest, *staging.glob(f"{digest}-*")]:
        if old.is_dir():
            shutil.rmtree(old)
    copied: list[str] = []
    home = source_runs / digest
    if not home.is_dir():
        return copied
    (staging / digest).mkdir(parents=True, exist_ok=True)
    for item in sorted(home.iterdir()):
        if item.is_file() and item.name != "history.jsonl":
            shutil.copy2(item, staging / digest / item.name)
            copied.append(f"{digest}/{item.name}")
    try:
        seeds = json.loads((home / "canonical.json").read_text(encoding="utf-8")).get("seeds")
    except (OSError, ValueError):
        seeds = None
    for seed in seeds or ():
        run = source_runs / f"{digest}-{int(seed)}"
        if run.is_dir():
            shutil.copytree(run, staging / run.name)
            copied.append(run.name)
    return copied


def main_checkout_of(path: Path) -> Path:
    """The main checkout of the repository ``path`` is in (the parent of the common .git)."""
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=path,
                            capture_output=True, text=True, check=True).stdout.strip()
    git = Path(common)
    if not git.is_absolute():
        git = Path(path) / git
    return git.resolve().parent


def plain_status(status: Mapping[str, Any] | None) -> list[str]:
    """The status file in a few lines, for `refresh --status`."""
    if not status:
        return ["the refresher has not run here"]
    lines = [f"state: {status.get('state')} (written {status.get('age_s')}s ago, "
             f"at {str(status.get('commit', ''))[:12]})"]
    doing = status.get("doing")
    if doing:
        lines.append(f"doing: {doing.get('name')} [{doing.get('hash')}] {doing.get('needs')}")
    if status.get("waiting_for"):
        lines.append(f"waiting for: {status['waiting_for']}")
    counts = status.get("owed_counts") or {}
    owed = [f"{counts[k]} {k}" for k in REPAIR_ORDER if counts.get(k)]
    refused_owed = int(status.get("refused_owed") or 0)
    lines.append("owed: " + (", ".join(owed) or "nothing")
                 + (f" ({refused_owed} more owed and refused)" if refused_owed else ""))
    last = status.get("last_done")
    if last:
        lines.append(f"last: {last.get('name')} [{last.get('hash')}] {last.get('needs')} "
                     f"index {last.get('index')}")
    refusal = status.get("last_refusal")
    if refusal:
        lines.append(f"last refusal: {refusal.get('name', '')} [{refusal.get('hash', '')}] "
                     f"{refusal.get('kind')}: {refusal.get('reason')}")
    standing: Sequence[Mapping[str, Any]] = status.get("refusals") or []
    if standing:
        lines.append(f"refused, not retried: {len(standing)} "
                     "(`refresh --forget <hash>` clears one)")
    if status.get("owed_merge"):
        lines.append(f"owed a merge into main: {', '.join(status['owed_merge'])} "
                     f"(branch {BRANCH})")
    return lines
