"""The refresher: the library kept on the current scorer, one game at a time (EV5).

`docs/archive/plan-evolving-score.md` §6. A revision of the scorer (a weights file, a metrics edit,
a roster change, a new player generation) makes estimates *provisional* at once and costs
nothing; this is what makes them *current* again, with nobody starting a campaign.
`chess-variant-sandbox refresh --library --watch` loops:

1. **What is owed**, read as `measure --check` reads it (`corpus.survey`), at the
   refresher's own commit, over the main checkout's `runs/`. No lease.
2. **The order** (§6.2): cheapest repair first (`rescore`, `reanalyse`, `replay`), then
   value within a repair kind (`value_tier`), then the oldest estimate, then the hash, so
   the order is reproducible. The order decides *when* a number
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
   whole loop until `main` moves, since every game would refuse), a repair that changed
   committed files, a gate the repair newly fails, or a repair that raised. It is
   logged, the game is skipped, and the next one is taken. A refused game comes back only
   when its stored measurement changes (somebody repaired it) or a person clears it
   (`refresh --forget <hash>`).
7. **The slow tier, weekly, when idle** (Q743, `docs/archive/plan-tests.md` §8 F1). The gate runs
   `pytest -m "not slow"` only, so a `slow` test went red the day the tree moved past it
   and stayed red. When a watching refresher finds nothing owed and the machine free, and
   the last slow-tier run finished `SLOW_TIER_EVERY` or more ago, it runs `pytest -m slow`
   in its own worktree under a `heavy` lease and writes the result to `slow-tier.json`
   beside its status file (`read_slow_tier`). A failure is recorded, never acted on: it
   repairs nothing, refuses nothing and moves no score. Committed files the run changed are
   put back and named. A one-pass `--library` or an `--only` run never runs it.
8. **The ladders, after scores** (L7, `docs/archive/plan-leaderboard.md` §2.2). Only when no score
   is owed (scores keep priority) does it read each library game's picker ladder, as
   `deploy/ladders_check.py` reads it (`ladder_items`), and take one ladder a lease, under
   `heavy`: a missing ladder first (`ladder`), then a short one a top-up can pay
   (`top-up`), then one with old players or short with no kept games (`re-rate`); within
   each, the launch set, then the oldest ladder, then the hash. The work is L3's own code,
   called and never repeated: `ladders.rate(..., top_up=...)` and `ladders.rerate`, in a
   child of the worktree writing straight into the main checkout's `runs/opponents/` as
   `ladders.py --runs` does (`rate_ladder`). A ladder is never delivered: `rerate` moves
   the old one aside and puts it back if the rating fails. A rating that fails, or that
   leaves the ladder still owing, is refused like a game, keyed on the ladder file.
9. **Proven positions, after ladders** (Q993d, `docs/plan-proven.md` §5). Only when no
   score and no ladder is owed does it read which Library and Baseline games are owed a
   proven-positions record (`proof_items`, `puzzles.survey`): none yet (`prove`), then one
   whose positions no longer replay or whose format or probe version is older than the
   code's (`re-prove`); within each, the launch set, then the oldest record, then the
   hash. One game a `heavy` lease: `puzzles.make` in a child of the worktree, writing the
   record straight into the main checkout's `runs/puzzles/` as a ladder is written
   (`prove_game`), its work resumable under `runs/puzzles/work/<hash>/`. A game whose runs
   hold no small ending gets a record with none, so it is asked once. A proof that fails,
   or leaves the game still owed, is refused keyed on the game's record file, and not
   retried until that file changes or `--forget`.

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

# §6.2's first key: what each repair costs. A re-score reads stored games; a re-analysis
# runs the pass over them first; a replay plays every game again.
REPAIR_ORDER = ("rescore", "reanalyse", "replay")

# The lease each repair takes (§6.1): `heavy` for anything that plays or analyses,
# `quiet` for a re-score, whose bootstrap takes the whole machine for its seconds.
LEASE_FOR = {"rescore": "quiet", "reanalyse": "heavy", "replay": "heavy"}

# Point 8 below: the ladder work, after every score (`docs/archive/plan-leaderboard.md` §2.2, L7).
# A game with no picker ladder is rated; a short one is topped up (its newcomers play the
# spine only); one with old players, or short and keeping no games to top up, is rated
# again whole (`deploy/ladders.py --rerate`). Each is a tournament, under `heavy`, as
# `ladders.py` takes.
LADDER_ORDER = ("ladder", "top-up", "re-rate")
LEASE_FOR.update({needs: "heavy" for needs in LADDER_ORDER})
# Point 9: a game's proven positions, after every ladder (Q993d). A game with no record is
# proved; one whose record no longer holds is proved again (`puzzles.NEEDS`). Each is up to
# a few core-hours of the core's probe, under `heavy`.
PROOF_ORDER = ("prove", "re-prove")
LEASE_FOR.update({needs: "heavy" for needs in PROOF_ORDER})
# The whole queue's first key: every score repair before any ladder, every ladder before
# any proof.
QUEUE_ORDER = REPAIR_ORDER + LADDER_ORDER + PROOF_ORDER
# A ladder item's `stored` is its ladder file hashed under this prefix, never to be
# mistaken for its game's `canonical.json` digest; a proof item's is its record file's,
# under `puzzles.STORED` (`proofs:`).
LADDER_STORED = "ladder:"
# The three parts of the queue, by what an item needs.
SCORES, LADDERS, PROOFS = "scores", "ladders", "proofs"


def part_of(needs: str) -> str:
    """Which part of the queue a need belongs to: a score, a ladder or a proof."""
    if needs in LADDER_ORDER:
        return LADDERS
    if needs in PROOF_ORDER:
        return PROOFS
    return SCORES

# Leases another worktree may hold that keep the refresher waiting: a replay, a training
# run, a campaign or a timed bench has the machine (the task's *never while a replay or
# another heavy job holds the machine*). A build or the fast suite does not: the lease's
# own slot arithmetic decides whether a game fits beside them.
YIELDS_TO = ("heavy", "quiet")

# §6.2's value, within one repair kind. The launch set's nine games, by name
# (`docs/reference/opponent-line.md` §3): the games a visitor meets first. The site
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
    "glinski hexagonal chess",
)
# Tier 0 the launch set; 1 the top quarter of the leaderboard by its last number, current
# or provisional; 2 within one interval of a gate; 3 the rest.
TOP_SHARE = 0.25

# Refusals by kind, so the loop can tell one that stops it from one that skips a game.
STALE_BUILD = "stale build"
GATE_NEWLY_FAILED = "gate newly failed"
COMMIT_MOVED = "commit moved"
CHANGED_COMMITTED_FILES = "changed committed files"
ERROR = "error"
HALTS = (STALE_BUILD,)

# The slow tier (point 7 above): its result file in the main checkout's `runs/`, how often
# it runs, the lease it takes, the workers it runs on (a `heavy` lease books ten slots),
# and how long a run may take before it is stopped and recorded as one that did not finish.
SLOW_TIER_FILE = "slow-tier.json"
SLOW_TIER_EVERY = datetime.timedelta(days=7)
SLOW_TIER_LEASE = "heavy"
SLOW_TIER_WORKERS = 8
SLOW_TIER_TIMEOUT_S = 3 * 3600
# How many failing test ids the status file and its summary carry; the rest are counted.
SLOW_TIER_NAMES = 50


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


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
        return (QUEUE_ORDER.index(self.needs), self.tier, self.measured, self.hash)

    @property
    def ladder(self) -> bool:
        """Whether this is ladder work (point 8), not a score repair."""
        return self.needs in LADDER_ORDER

    @property
    def proofs(self) -> bool:
        """Whether this is a game's proven positions (point 9)."""
        return self.needs in PROOF_ORDER

    @property
    def part(self) -> str:
        return part_of(self.needs)

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
    # What a refusal is keyed on when it is not the item's own `stored`: a ladder rated
    # and still owing is refused as the file it now is, so it is not taken again.
    stored: str = ""

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
            stored=str(data.get("stored") or ""),
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
    anything runs. A gate the
    stored file passed and the new one fails is refused after the repair, and the
    supervisor then delivers nothing.
    """
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.measurement import stale_build
    from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS

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

    @property
    def refusals(self) -> list[dict[str, Any]]:
        return list(self.status.get("refusals") or [])

    def refused(self, item: Item) -> bool:
        """Whether this game was refused with the stored file it has now."""
        return any(r.get("hash") == item.hash and r.get("stored") == item.stored
                   for r in self.refusals)

    def refuse(self, item: Item, outcome: Outcome) -> dict[str, Any]:
        record = {"hash": item.hash, "name": item.name, "needs": item.needs,
                  "stored": outcome.stored or item.stored, "kind": outcome.refusal,
                  "reason": outcome.reason, "commit": outcome.commit, "at": _now()}
        # A game's score refusal, its ladder's and its proofs' are three: one never
        # replaces another.
        kept = [r for r in self.refusals
                if r.get("hash") != item.hash or _part(r) != item.part]
        self.status["refusals"] = [*kept, record]
        self.status["last_refusal"] = record
        return record

    def lapse(self, owed: Iterable[Item], ladders: bool = False,
              part: str | None = None) -> list[dict[str, Any]]:
        """Drop the refusals that no longer stand, and return them.

        A refusal is of one stored file (`refused`): once the game's file has changed, or
        the game is owed nothing at all, it refuses nothing, and a standing count or an
        `/admin` row that still showed it would say a game is held back that is not
        (Q693r's first real run). ``owed`` is the whole survey, before `--only`: the score
        survey's, or with ``ladders`` the ladder survey's (``part`` names any of the three,
        `PROOFS` the proofs survey's), and only refusals of that part are read against it.
        """
        part = part or (LADDERS if ladders else SCORES)
        still = {(item.hash, item.stored) for item in owed}
        gone = [r for r in self.refusals if _part(r) == part
                and (r.get("hash"), r.get("stored")) not in still]
        kept = [r for r in self.refusals if r not in gone]
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


def _part(record: Mapping[str, Any]) -> str:
    return part_of(str(record.get("needs") or ""))


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


def ladder_counts(items: Iterable[Item]) -> dict[str, int]:
    counts = {needs: 0 for needs in LADDER_ORDER}
    for item in items:
        counts[item.needs] += 1
    return counts


def proof_counts(items: Iterable[Item]) -> dict[str, int]:
    counts = {needs: 0 for needs in PROOF_ORDER}
    for item in items:
        counts[item.needs] += 1
    return counts


# ------------------------------------------------------------------- the ladders (L7)

# This checkout: its `deploy/` holds L3's ladder code, run by the code of the same commit.
REPOSITORY = Path(__file__).resolve().parent.parent


def _deploy(name: str) -> Any:
    """`deploy/<name>.py` of this checkout, or ``None`` where there is no `deploy/` (the
    public export ships none, and owes no ladders).

    Imported by its plain name, as `deploy/ladders.py` imports `ladders_check`, so both
    read one module and a test's stand-in for `ladders_check.roster` reaches both.
    """
    import importlib

    directory = REPOSITORY / "deploy"
    if not (directory / f"{name}.py").is_file():
        return None
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module(name)


def ladder_stored(digest: str, runs_dir: Path) -> str:
    """The ladder file's bytes, hashed: what a ladder's refusal is keyed on."""
    try:
        body = (Path(runs_dir) / "opponents" / f"{digest}.json").read_bytes()
    except OSError:
        return LADDER_STORED + "absent"
    return LADDER_STORED + hashlib.sha256(body).hexdigest()[:16]


def ladder_work(found: Any) -> tuple[str, str] | None:
    """What one `ladders_check.Ladder` is owed next, as ``(needs, reason)``, or ``None``.

    L3's reading decides, never a second one: `owes` (missing, short, old players) and
    `top_up` (whether a short one can be paid by its newcomers' games alone). A short
    ladder a top-up cannot pay is rated again whole, with the old players.
    """
    if not found.owes:
        return None
    if not found.present:
        return "ladder", "missing: no picker ladder"
    names = ", ".join((found.names or {}).get(agent, agent) for agent in found.short)
    if found.top_up:
        return "top-up", f"short: {names}"
    why = []
    if found.old:
        why.append(f"old players: rated {found.rated or 'on an unknown date'}, before "
                   f"{found.old[-1][0]}")
    if found.short:
        why.append(f"short: {names}" + ("" if found.kept else " (no kept games)"))
    return "re-rate", "; ".join(why)


def ladder_items(runs_dir: Path, root: Path | None = None) -> list[Item]:
    """The ladder work this tree owes over ``runs_dir``, in order. Reads; spends nothing.

    The library is ``root``'s (default the working directory, as `corpus.survey` reads
    it); each game read by `ladders_check.ladder`, the check `deploy/ladders.py` acts on.
    """
    checks = _deploy("ladders_check")
    if checks is None:
        return []
    root = Path(root) if root is not None else Path.cwd()
    items = []
    for game in checks.library(root):
        found = checks.ladder(game, Path(runs_dir), root)
        work = ladder_work(found)
        if work is None:
            continue
        needs, reason = work
        items.append(Item(
            hash=game.digest, name=game.name, needs=needs, reason=reason,
            tier=value_tier(game.name, game.digest, set(), Stored()),
            measured=found.rated or "", stored=ladder_stored(game.digest, runs_dir),
        ))
    return sorted(items, key=lambda item: item.key)


def rate_ladder(
    digest: str,
    runs_dir: Path,
    root: Path | None = None,
    say: Callable[[str], None] | None = None,
) -> Outcome:
    """Pay what one game's ladder owes now, or say why not: what `refresh <hash> --ladder`
    runs.

    L3's calls do the work (`ladders.rate`, with ``top_up`` for a short ladder, and
    `ladders.rerate`), into ``runs_dir``'s `opponents/` as `ladders.py --runs` writes. A
    stale build is refused first, as for a score. A rating that fails is an error (and
    `rerate` has put the old ladder back); one that finishes and leaves the ladder still
    owing is refused as the file it left, so the loop does not rate it again and again.
    """
    from chess_variant_sandbox.measurement import stale_build

    lines: list[str] = []
    started = time.monotonic()
    runs = Path(runs_dir)

    def note(line: str) -> None:
        lines.append(str(line).strip())
        if say is not None:
            say(str(line))

    def done(needs: str, refusal: str = "", reason: str = "", **rest: Any) -> Outcome:
        return Outcome(hash=digest, needs=needs, refusal=refusal, reason=reason,
                       seconds=round(time.monotonic() - started, 1), lines=tuple(lines),
                       **rest)

    behind = stale_build()
    if behind is not None:
        return done("", STALE_BUILD, behind)
    needs = ""
    try:
        checks, ladders = _deploy("ladders_check"), _deploy("ladders")
        if checks is None or ladders is None:
            return done("", ERROR, "this checkout has no deploy/ladders.py")
        root = Path(root) if root is not None else Path.cwd()
        games = [game for game in checks.library(root) if game.digest == digest]
        if not games:
            return done("", ERROR, f"no library game has the hash {digest}")
        game = games[0]
        work = ladder_work(checks.ladder(game, runs, root))
        if work is None:
            note(f"{game.name}: its ladder owes nothing")
            return done("")
        needs = work[0]
        if needs == "re-rate":
            finished = ladders.rerate(game, root, runs, say=note)
        else:
            finished = ladders.rate(game, root, runs, say=note, top_up=needs == "top-up")
        if not finished:
            return done(needs, ERROR, f"the rating did not finish: {lines[-1] if lines else ''}")
        after = ladder_work(checks.ladder(game, runs, root))
    except Exception as error:  # noqa: BLE001 - every failure is a logged refusal
        return done(needs, ERROR, f"{type(error).__name__}: {error}")
    if after is not None:
        return done(needs, ERROR, f"rated, and its ladder still owes {after[0]}: {after[1]}",
                    stored=ladder_stored(digest, runs))
    return done(needs)


# ------------------------------------------------------- proven positions (Q993d)


def proof_items(runs_dir: Path, root: Path | None = None) -> list[Item]:
    """The proven-positions work this tree owes over ``runs_dir``, in order. Reads (each
    record's positions replayed through the engine); proves nothing.

    The library is ``root``'s (default the working directory), read by `puzzles.survey`:
    a game with no record (`prove`) before one whose record no longer holds (`re-prove`),
    then the launch set, the oldest record, the hash.
    """
    from chess_variant_sandbox import puzzles

    root = Path(root) if root is not None else Path.cwd()
    items = [
        Item(hash=found.hash, name=found.name, needs=found.needs, reason=found.reason,
             tier=value_tier(found.name, found.hash, set(), Stored()),
             measured=found.made, stored=found.stored)
        for found in puzzles.survey(runs_dir, root) if found.owed
    ]
    return sorted(items, key=lambda item: item.key)


def prove_game(
    digest: str,
    runs_dir: Path,
    root: Path | None = None,
    workers: int = 0,
    say: Callable[[str], None] | None = None,
) -> Outcome:
    """Prove one game's positions now, or say why not: what `refresh <hash> --proofs` runs.

    `puzzles.make` does the work, writing the record into ``runs_dir``'s `puzzles/` (its
    work under `puzzles/work/<hash>/`, picked up where a killed run left it). A stale build
    is refused first, as for a score. A proof that raises is an error; one that finishes
    and leaves the game still owed is refused as the record it left, so the loop does not
    prove it again and again.
    """
    from chess_variant_sandbox import puzzles
    from chess_variant_sandbox.measurement import stale_build

    lines: list[str] = []
    started = time.monotonic()
    runs = Path(runs_dir)

    def note(line: str) -> None:
        lines.append(str(line).strip())
        if say is not None:
            say(str(line))

    def done(needs: str, refusal: str = "", reason: str = "", **rest: Any) -> Outcome:
        return Outcome(hash=digest, needs=needs, refusal=refusal, reason=reason,
                       seconds=round(time.monotonic() - started, 1), lines=tuple(lines),
                       **rest)

    behind = stale_build()
    if behind is not None:
        return done("", STALE_BUILD, behind)
    needs = ""
    try:
        root = Path(root) if root is not None else Path.cwd()
        games = [game for game in puzzles.library(root) if game[0] == digest]
        if not games:
            return done("", ERROR, f"no library game has the hash {digest}")
        _digest, name, spec = games[0]
        before = puzzles.standing(digest, name, spec, runs)
        if not before.owed:
            note(f"{name}: its proven positions owe nothing ({before.reason})")
            return done("")
        needs = before.needs
        note(f"{name} [{digest}]: {needs} ({before.reason})")
        made = puzzles.make(spec, runs, workers=workers or puzzles.WORKERS, say=note)
        took = ", ".join(f"{stage} {seconds:.0f} s" for stage, seconds in made["seconds"].items())
        note(f"{name}: {made['proved']} proved, {made['offered']} offered; {took}")
        after = puzzles.standing(digest, name, spec, runs)
    except Exception as error:  # noqa: BLE001 - every failure is a logged refusal
        return done(needs, ERROR, f"{type(error).__name__}: {error}")
    if after.owed:
        return done(needs, ERROR, f"proved, and it is still owed {after.needs}: {after.reason}",
                    stored=after.stored)
    return done(needs)


# ------------------------------------------------------------------ the slow tier


@dataclass(frozen=True)
class SlowTier:
    """One run of `pytest -m slow` in the refresher's worktree, as `slow-tier.json` keeps it.

    ``failing`` holds the ids of the tests that failed or errored, in the order the report
    gave them. ``note`` says why a run did not finish, or what it changed that was put back.
    """

    commit: str = ""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failing: tuple[str, ...] = ()
    seconds: float = 0.0
    exit_code: int = 0
    note: str = ""

    @property
    def outcome(self) -> str:
        """`green`, `red` (a test failed or errored), or `unfinished` (no verdict at all)."""
        if self.failed or self.errors:
            return "red"
        if self.exit_code == 0 and self.passed > 0:
            return "green"
        return "unfinished"

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "failing": list(self.failing), "outcome": self.outcome}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SlowTier:
        return cls(
            commit=str(data.get("commit") or ""),
            passed=int(data.get("passed") or 0),
            failed=int(data.get("failed") or 0),
            errors=int(data.get("errors") or 0),
            skipped=int(data.get("skipped") or 0),
            failing=tuple(str(name) for name in data.get("failing") or ()),
            seconds=float(data.get("seconds") or 0.0),
            exit_code=int(data.get("exit_code") or 0),
            note=str(data.get("note") or ""),
        )


def slow_tier_summary(tier: SlowTier, finished: str) -> str:
    """One line a person or a file-reading tool can print as it is."""
    day = finished[:10]
    counts = f"{tier.passed} passed, {tier.failed} failed"
    counts += f", {tier.errors} errors" if tier.errors else ""
    counts += f", {tier.skipped} skipped" if tier.skipped else ""
    line = f"{tier.outcome} on {day} at {tier.commit[:12]}: {counts}"
    if tier.failing:
        shown = ", ".join(tier.failing[:3])
        more = len(tier.failing) - 3
        line += f" ({shown}{f' and {more} more' if more > 0 else ''})"
    if tier.note:
        line += f"; {tier.note}"
    return line


def parse_junit(path: Path) -> SlowTier:
    """Counts and failing ids from pytest's `--junitxml` report, and nothing else."""
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(path).getroot()
    passed = failed = errors = skipped = 0
    failing: list[str] = []
    for case in root.iter("testcase"):
        kinds = {child.tag for child in case}
        name = "::".join(part for part in (case.get("classname"), case.get("name")) if part)
        if "error" in kinds:
            errors += 1
            failing.append(name)
        elif "failure" in kinds:
            failed += 1
            failing.append(name)
        elif "skipped" in kinds:
            skipped += 1
        else:
            passed += 1
    return SlowTier(passed=passed, failed=failed, errors=errors, skipped=skipped,
                    failing=tuple(failing))


def read_slow_tier(runs_dir: Path) -> dict[str, Any] | None:
    """The last slow-tier run's file, or ``None`` if the refresher has run none here.

    Plain JSON with a ready ``summary`` line, so a tool that reads files and imports
    nothing (`.claude/scripts/slow_tier_status.py`) can show it.
    """
    try:
        data = json.loads((Path(runs_dir) / SLOW_TIER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_slow_tier(runs_dir: Path, tier: SlowTier, started: str,
                    finished: str) -> dict[str, Any]:
    record = {**tier.as_dict(), "failing": list(tier.failing[:SLOW_TIER_NAMES]),
              "failing_total": len(tier.failing), "started": started,
              "finished": finished, "summary": slow_tier_summary(tier, finished)}
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / SLOW_TIER_FILE
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)
    return record


def slow_tier_due(record: Mapping[str, Any] | None, now: datetime.datetime,
                  every: datetime.timedelta = SLOW_TIER_EVERY) -> bool:
    """Whether the last run finished `every` or more before ``now``, or there was none.

    A run that did not finish is a run: a tier that cannot run here is not retried every
    ten minutes, and its file says why.
    """
    if not record:
        return True
    try:
        finished = datetime.datetime.fromisoformat(str(record.get("finished")))
    except ValueError:
        return True
    return now - finished >= every


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
    def ladders(self) -> list[Item]: ...  # the ladder work, at the worktree's commit
    def proofs(self) -> list[Item]: ...  # the proven-positions work, at that commit
    def repair(self, item: Item) -> Outcome: ...  # one game, ladder or proof, at that commit
    def deliver(self, item: Item) -> None: ...  # `history.deliver` into main's runs/
    def changed(self) -> list[str]: ...  # committed files the last repair changed
    def discard(self, paths: list[str]) -> None: ...  # put them back
    def slow_tier(self) -> SlowTier: ...  # `pytest -m slow` at the worktree's commit
    def sleep(self, seconds: float) -> None: ...



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
    # The weekly slow tier (point 7): ``None`` runs it exactly when watching.
    slow_tier: bool | None = None
    clock: Callable[[], datetime.datetime] = _utcnow

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
        if todo:
            # Scores keep priority: the ladders and proofs are not even read while one is owed.
            self.ledger.update(ladders_owed=None, ladder_counts=None,
                               proofs_owed=None, proof_counts=None)
        else:
            try:
                todo = self._ladders()
            except Exception as error:  # noqa: BLE001 - logged; main moving is the way on
                return self._halt(f"the ladder survey failed: {error}", kind=ERROR)
            if todo:
                # Point 9: the proofs wait on every ladder, as the ladders wait on scores.
                self.ledger.update(proofs_owed=None, proof_counts=None)
            else:
                try:
                    todo = self._proofs()
                except Exception as error:  # noqa: BLE001 - logged; main moving is the way on
                    return self._halt(f"the proofs survey failed: {error}", kind=ERROR)
        if not todo:
            if self._slow_tier_due():
                return self._slow_tier()
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

    def _ladders(self) -> list[Item]:
        """Point 8: the ladder work owed, once no score is, written to the status file."""
        items = self.machine.ladders()
        for record in self.ledger.lapse(items, ladders=True):
            self.ledger.log("lapsed", hash=record.get("hash"), name=record.get("name"),
                            kind=record.get("kind"), stored=record.get("stored"))
        if self.only is not None:
            items = [item for item in items if item.hash in self.only]
        todo = [item for item in items if not self.ledger.refused(item)]
        self.ledger.update(
            ladders_owed=[item.as_dict() for item in todo], ladder_counts=ladder_counts(todo),
            ladders_refused=len(items) - len(todo),
        )
        return todo

    def _proofs(self) -> list[Item]:
        """Point 9: the proven-positions work owed, once no score or ladder is, written to
        the status file."""
        items = self.machine.proofs()
        for record in self.ledger.lapse(items, part=PROOFS):
            self.ledger.log("lapsed", hash=record.get("hash"), name=record.get("name"),
                            kind=record.get("kind"), stored=record.get("stored"))
        if self.only is not None:
            items = [item for item in items if item.hash in self.only]
        todo = [item for item in items if not self.ledger.refused(item)]
        self.ledger.update(
            proofs_owed=[item.as_dict() for item in todo], proof_counts=proof_counts(todo),
            proofs_refused=len(items) - len(todo),
        )
        return todo

    def _settle(self, item: Item, outcome: Outcome) -> str:
        # A repair changes no committed file: one that did is refused, and put back.
        changed = self.machine.changed()
        if not outcome.refused and changed:
            outcome = replace(
                outcome, refusal=CHANGED_COMMITTED_FILES,
                reason="the repair changed committed files: " + ", ".join(changed),
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
            # A ladder was written in place, into main's `runs/opponents/` (point 8), and
            # a record into main's `runs/puzzles/` (point 9): only a score is delivered.
            if item.part == SCORES:
                self.machine.deliver(item)
        except Exception as error:  # noqa: BLE001 - logged; nothing is retried
            return self._halt(f"the delivery failed: {error}", item, kind=ERROR)
        last = {"hash": item.hash, "name": item.name, "needs": outcome.needs or item.needs,
                "index": outcome.index, "viable": outcome.viable, "commit": outcome.commit,
                "seconds": outcome.seconds, "at": _now()}
        if item.proofs and outcome.lines:
            # What the proof found and what each stage took (`prove_game`'s last line).
            last["said"] = outcome.lines[-1]
        self.ledger.log("repaired", **last)
        self.ledger.update(state="working", doing=None, last_done=last)
        self.say(f"  {'delivered' if item.part == SCORES else 'written'} in "
                 f"{outcome.seconds:.0f}s")
        return "repaired"

    def _slow_tier_due(self) -> bool:
        wanted = self.watch if self.slow_tier is None else self.slow_tier
        return (wanted and self.only is None
                and slow_tier_due(read_slow_tier(self.ledger.runs_dir), self.clock()))

    def _slow_tier(self) -> str:
        """Run the slow tier once, under its own lease, and record it; never act on it."""
        if not self.machine.acquire(SLOW_TIER_LEASE, "refresh slow tier"):
            self.ledger.update(state="waiting", doing=None,
                               waiting_for=f"a {SLOW_TIER_LEASE} lease for the slow tier")
            return "waiting"
        started = _now()
        # `working`, the one state a page waits hours on before it calls the loop quiet
        # (`web/refreshing.QUIET_AFTER_S`); a run is bounded by `SLOW_TIER_TIMEOUT_S`.
        self.ledger.update(state="working", doing={
            "name": "the slow tier", "hash": "", "needs": "running the slow tests",
            "started": started})
        self.say("nothing owed: running the slow tier")
        try:
            tier = self.machine.slow_tier()
        except Exception as error:  # noqa: BLE001 - recorded; the loop goes on
            tier = SlowTier(commit=self.machine.head(), exit_code=-1,
                            note=f"the run raised: {error}")
        finally:
            self.machine.release()
        changed = self.machine.changed()
        if changed:
            self.machine.discard(changed)
            put_back = "it changed committed files, put back: " + ", ".join(changed)
            tier = replace(tier, note=f"{tier.note}; {put_back}" if tier.note else put_back)
        record = write_slow_tier(self.ledger.runs_dir, tier, started, _now())
        self.ledger.log("slow tier", outcome=tier.outcome, commit=tier.commit,
                        passed=tier.passed, failed=tier.failed, errors=tier.errors,
                        failing=record["failing"], seconds=tier.seconds)
        self.ledger.update(state="idle", doing=None, slow_tier={
            "outcome": tier.outcome, "finished": record["finished"],
            "summary": record["summary"]})
        self.say(f"  slow tier {record['summary']}")
        return "idle"

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
        done = self.run(["git", "merge-base", "--is-ancestor", "main", "HEAD"],
                        cwd=self.worktree, capture_output=True, text=True, check=False)
        return done.returncode != 0

    def advance(self) -> list[str]:
        before = self.head()
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
        args = ["refresh", "--plan", "--part", "scores", "--json", "--runs",
                str(self.main_runs)]
        done = self._child(*args, *(["--weights", self.weights] if self.weights else []))
        if done.returncode != 0:
            raise RuntimeError(f"the survey failed: {done.stderr.strip()[-400:]}")
        return [Item.from_dict(row) for row in json.loads(done.stdout)]

    def ladders(self) -> list[Item]:
        done = self._child("refresh", "--plan", "--part", "ladders", "--json",
                           "--runs", str(self.main_runs))
        if done.returncode != 0:
            raise RuntimeError(f"the ladder survey failed: {done.stderr.strip()[-400:]}")
        return [Item.from_dict(row) for row in json.loads(done.stdout)]

    def proofs(self) -> list[Item]:
        done = self._child("refresh", "--plan", "--part", "proofs", "--json",
                           "--runs", str(self.main_runs))
        if done.returncode != 0:
            raise RuntimeError(f"the proofs survey failed: {done.stderr.strip()[-400:]}")
        return [Item.from_dict(row) for row in json.loads(done.stdout)]

    def stage(self, digest: str) -> None:
        """Copy one game's stored measurement and runs from the main checkout's `runs/`."""
        stage_game(digest, self.main_runs, self.staging)

    def repair(self, item: Item) -> Outcome:
        commit = self.head()
        if item.ladder:
            # Rated in place into main's `runs/opponents/`, as `ladders.py --runs` writes:
            # a ladder has no history to deliver through, and `rerate` keeps the old one.
            done = self._child("refresh", item.hash, "--ladder", "--json",
                               "--runs", str(self.main_runs))
        elif item.proofs:
            # Proved in place into main's `runs/puzzles/`, from the four library runs there:
            # a record has no history to deliver through, and `make` keeps the old one.
            done = self._child("refresh", item.hash, "--proofs", "--json",
                               "--runs", str(self.main_runs), "--workers", str(self.workers))
        else:
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

    def discard(self, paths: list[str]) -> None:
        self._git("checkout", "--", ".")
        self._git("clean", "-fd", "--", *paths)

    def slow_tier(self) -> SlowTier:
        """`pytest -m slow` in the worktree, on its own interpreter, read from its report.

        The report goes to a scratch directory, so the run writes nothing in the worktree
        that the loop reads; whatever a test leaves in committed files the loop puts back.
        """
        import tempfile

        commit = self.head()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="slow-tier-") as scratch:
            report = Path(scratch) / "slow.xml"
            command = [str(_python(self.worktree)), "-m", "pytest", "-m", "slow", "-q",
                       "-p", "no:cacheprovider", "-n", str(SLOW_TIER_WORKERS),
                       f"--junitxml={report}"]
            try:
                done = self.run(command, cwd=self.worktree, capture_output=True, text=True,
                                check=False, timeout=SLOW_TIER_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return SlowTier(commit=commit, exit_code=-1,
                                seconds=round(time.monotonic() - started, 1),
                                note=f"stopped after {SLOW_TIER_TIMEOUT_S}s")
            seconds = round(time.monotonic() - started, 1)
            if not report.exists():
                said = (done.stdout or "") + (done.stderr or "")
                return SlowTier(commit=commit, exit_code=done.returncode, seconds=seconds,
                                note=f"pytest wrote no report: {said.strip()[-400:]}")
            tier = parse_junit(report)
        note = "" if self.head() == commit else "the worktree moved during the run"
        return replace(tier, commit=commit, seconds=seconds, exit_code=done.returncode,
                       note=note)

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
    following = (list(status.get("owed") or []) + list(status.get("ladders_owed") or [])
                 + list(status.get("proofs_owed") or []))
    if following:
        first = following[0]
        lines.append(f"next: {first.get('name')} [{first.get('hash')}] {first.get('needs')}"
                     f" ({first.get('reason')})")
    ladders = status.get("ladder_counts")
    if isinstance(ladders, Mapping):
        owed_ladders = [f"{ladders[k]} {k}" for k in LADDER_ORDER if ladders.get(k)]
        refused_ladders = int(status.get("ladders_refused") or 0)
        lines.append("ladders owed: " + (", ".join(owed_ladders) or "nothing")
                     + (f" ({refused_ladders} more owed and refused)"
                        if refused_ladders else ""))
    elif counts and any(counts.values()):
        lines.append("ladders: read once no score is owed (scores first)")
    proofs = status.get("proof_counts")
    if isinstance(proofs, Mapping):
        owed_proofs = [f"{proofs[k]} {k}" for k in PROOF_ORDER if proofs.get(k)]
        refused_proofs = int(status.get("proofs_refused") or 0)
        lines.append("proofs owed: " + (", ".join(owed_proofs) or "nothing")
                     + (f" ({refused_proofs} more owed and refused)"
                        if refused_proofs else ""))
    elif (counts and any(counts.values())) or (isinstance(ladders, Mapping)
                                               and any(ladders.values())):
        lines.append("proofs: read once no score or ladder is owed (scores, then ladders)")
    last = status.get("last_done")
    if last:
        lines.append(f"last: {last.get('name')} [{last.get('hash')}] {last.get('needs')}"
                     + ("" if part_of(str(last.get("needs") or "")) != SCORES
                        else f" index {last.get('index')}"))
    refusal = status.get("last_refusal")
    if refusal:
        lines.append(f"last refusal: {refusal.get('name', '')} [{refusal.get('hash', '')}] "
                     f"{refusal.get('kind')}: {refusal.get('reason')}")
    standing: Sequence[Mapping[str, Any]] = status.get("refusals") or []
    if standing:
        lines.append(f"refused, not retried: {len(standing)} "
                     "(`refresh --forget <hash>` clears one)")
    tier = status.get("slow_tier")
    lines.append(f"slow tier: {tier.get('summary')}" if isinstance(tier, Mapping)
                 else "slow tier: not run here yet (weekly, when watching and idle)")
    return lines
