"""Campaigns that survive a crash (Q732): each piece on disk as it finishes, resumed by content.

**What it is.** A *campaign directory*, given as ``chess-variant-sandbox --checkpoint DIR
<command> ...`` (or ``with checkpoint.active(DIR):`` in a script), under which every long
unit of work the process runs is a *stage*:

- the games of a schedule played by the core (a tournament, each seed of a measurement, a
  self-play, a duel, a replay set, with or without random openings): ``games``;
- a harvest's positions: ``harvest``;
- a training run: ``train``, saved every ``--checkpoint-minutes`` (default ten).

A stage is named by what it computes: its kind and identity (the ruleset hash, the agents
and the SHA-256 of any net they play, the schedule's seeds, the procedure, the data's
digest). Games and harvests are cut into at most ``PIECES`` *pieces* from the schedule
alone, never from the worker count, before any work starts, and each piece is written
atomically (temp file, fsync, rename), sealed with its SHA-256, when it finishes. A
restart of the same command on the same directory reads the manifest, checks the build,
loads every sealed piece and plays only the rest. A training run resumes from its newest
save whose seal holds; a torn one is reported and the one before it is used.

**Why the answer cannot move.** A game is a pure function of (ruleset hash, seed, agent
configs) and the pieces are fixed by the schedule, so a piece played after a crash is the
piece that was lost, and the records come back in schedule order; a harvest is
concatenated in game order whatever its chunks; a training save holds the weights, Adam's
moments, the step, the epoch's random state and the position in it, so the steps after a
resume are the steps that would have run. A resumed run's aggregate is byte-identical to an
uninterrupted one's (``tests/test_checkpoint.py`` kills runs and compares); only wall time
(``duration_ms``, the printed seconds) differs.

**What it refuses.** A directory begun on another build (``measurement.built_fingerprint``):
its pieces would mix two cores' games under one result. Finish it on its build or start a
new directory.

**Without a campaign** nothing here runs and every caller takes its path of before, so a
run without ``--checkpoint`` plays, and means, exactly what it did.

**Who computes what** (CLAUDE.md rule 2). The core still plays whole games; this module
only decides which pieces are owed and writes what comes back.

The layout, which ``.claude/scripts/campaign_status.py`` reads without importing this::

    DIR/manifest.json        {"kind": "campaign", "format", "build", "commands", "stages"}
    DIR/progress.log         one line per piece, save and command: the stall check's clock
    DIR/<kind>-<key>/stage.json   the identity and the pieces' units
    DIR/<kind>-<key>/piece-00000.ckpt ...   sealed pieces, or train-000001.ckpt ... saves
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import io
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

FORMAT = 1
KIND = "campaign"
MANIFEST = "manifest.json"
LOG = "progress.log"
STAGE = "stage.json"
MAGIC = b"CVSCKPT1"
# The most pieces a stage is cut into: a 400-game duel is a game a piece, a self-play of
# 80,000 games is 80 a piece. A crash loses at most the pieces in flight.
PIECES = 1000
SAVE_MINUTES = 10.0
# Training saves kept: the newest and the two before it, so a torn newest has a fallback.
KEEP_SAVES = 3


class CampaignError(ValueError):
    """A campaign directory that cannot be resumed by this process as it stands."""


# ------------------------------------------------------------------ atomic, sealed files


def atomic_write(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` so a reader sees the old file or the whole new one.

    The bytes go to a temp file beside it, are flushed to the disk, and the temp is renamed
    over the target, which is atomic on one volume (``os.replace``).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    return target


def atomic_json(path: str | Path, payload: Any) -> Path:
    return atomic_write(path, (json.dumps(payload, indent=1) + "\n").encode("utf-8"))


def seal(payload: bytes) -> bytes:
    """``payload`` behind a magic and its SHA-256, so a torn or altered file is known."""
    return MAGIC + hashlib.sha256(payload).digest() + payload


def unseal(blob: bytes) -> bytes | None:
    """The payload ``seal`` wrapped, or ``None`` if the seal does not hold."""
    head = len(MAGIC) + 32
    if len(blob) < head or not blob.startswith(MAGIC):
        return None
    payload = blob[head:]
    if hashlib.sha256(payload).digest() != blob[len(MAGIC) : head]:
        return None
    return payload


def read_sealed(path: str | Path) -> bytes | None:
    """A sealed file's payload; ``None`` when it is missing or its seal does not hold."""
    try:
        blob = Path(path).read_bytes()
    except OSError:
        return None
    return unseal(blob)


def digest(value: Any) -> str:
    """SHA-256 of ``value`` as canonical JSON: what names a stage."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ the campaign


class Campaign:
    """One campaign directory, opened by this process. Thread-safe for its writes."""

    def __init__(self, root: Path, build: str, save_minutes: float) -> None:
        self.root = root
        self.build = build
        self.save_minutes = save_minutes
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        root: str | Path,
        build: str | None = None,
        save_minutes: float = SAVE_MINUTES,
        command: Sequence[str] | None = None,
    ) -> Campaign:
        """Begin ``root``, or resume it if its manifest is this build's."""
        if build is None:
            from chess_variant_sandbox.measurement import built_fingerprint

            build = built_fingerprint()
        path = Path(root)
        campaign = cls(path, build, save_minutes)
        manifest = path / MANIFEST
        if manifest.exists():
            stored = json.loads(manifest.read_text(encoding="utf-8"))
            if stored.get("kind") != KIND or stored.get("format") != FORMAT:
                raise CampaignError(f"{manifest} is not a campaign manifest of format {FORMAT}")
            if stored.get("build") != build:
                raise CampaignError(
                    f"campaign {path} was begun on build {stored.get('build')} and this core is "
                    f"{build}: its pieces would mix two cores' games. Finish it on its own "
                    "build, or give a new --checkpoint directory."
                )
            campaign.log(f"resume on build {build}")
        else:
            atomic_json(
                manifest,
                {"kind": KIND, "format": FORMAT, "build": build, "commands": [], "stages": {}},
            )
            campaign.log(f"begin on build {build}")
        if command is not None:
            argv = list(command)

            def remember(manifest: dict[str, Any]) -> None:
                if argv not in manifest["commands"]:
                    manifest["commands"].append(argv)

            campaign._update(remember)
            campaign.log("start " + " ".join(argv))
        return campaign

    def manifest(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        return loaded

    def _update(self, change: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            manifest = self.manifest()
            change(manifest)
            atomic_json(self.root / MANIFEST, manifest)

    def log(self, text: str) -> None:
        """One line in ``progress.log``: the clock the stall check reads."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.root / LOG, "a", encoding="utf-8") as handle:
                handle.write(f"[{_now()}] {text}\n")

    def stage(
        self, kind: str, identity: dict[str, Any], units: Sequence[Any], size: int | None = None
    ) -> Stage:
        """The stage that computes ``identity`` over ``units``, cut into pieces.

        The cut is a function of the units alone (at most ``PIECES`` pieces), so a resumed
        run cuts the same pieces whatever its worker count.
        """
        units = list(units)
        size = size or max(1, math.ceil(len(units) / PIECES))
        pieces = [units[start : start + size] for start in range(0, len(units), size)]
        body = {"kind": kind, "identity": identity, "pieces": pieces}
        name = f"{kind}-{digest(body)[:16]}"
        directory = self.root / name
        described = directory / STAGE
        if described.exists():
            stored = json.loads(described.read_text(encoding="utf-8"))
            if stored != json.loads(json.dumps(body)):
                raise CampaignError(f"{described} does not describe the stage its name says")
        else:
            atomic_json(described, body)

        def add(manifest: dict[str, Any]) -> None:
            # Also when the stage's file outlived a crash before the manifest learnt of it.
            manifest["stages"].setdefault(name, {
                "kind": kind,
                "pieces": len(pieces),
                "units": len(units),
                "label": identity.get("label", ""),
            })

        if name not in self.manifest()["stages"]:
            self._update(add)
        return Stage(self, kind, name, directory, pieces)

    def saves(self, identity: dict[str, Any]) -> Saves:
        """A training run's saves: one stage, saved every ``save_minutes``."""
        stage = self.stage("train", identity, [])
        return Saves(stage, self.save_minutes * 60.0)


class Stage:
    """One stage's pieces: which are on disk, and writing one when it finishes."""

    def __init__(
        self, campaign: Campaign, kind: str, name: str, directory: Path, pieces: list[list[Any]]
    ) -> None:
        self.campaign = campaign
        self.kind = kind
        self.name = name
        self.directory = directory
        self.pieces = pieces
        self._done = 0

    def path(self, index: int) -> Path:
        return self.directory / f"piece-{index:05d}.ckpt"

    def load(self, index: int) -> bytes | None:
        return read_sealed(self.path(index))

    def save(self, index: int, payload: bytes) -> None:
        atomic_write(self.path(index), seal(payload))
        self._done += 1
        self.campaign.log(
            f"{self.name} piece {index + 1}/{len(self.pieces)} done "
            f"({self._done}/{len(self.pieces)} on disk)"
        )

    def run(
        self,
        submit: Callable[[list[Any]], Future[T]],
        encode: Callable[[T], bytes],
        decode: Callable[[bytes, list[Any]], T | None],
        on_piece: Callable[[int], None] | None = None,
    ) -> list[T]:
        """Every piece's result, in piece order: loaded where sealed, computed otherwise.

        ``submit`` starts one piece's work (on a thread or process pool the caller owns)
        and returns its future; ``decode`` returns ``None`` for a payload that is not this
        piece's, which is then computed again. A piece is written the moment it comes back,
        so a crash loses only the pieces in flight. ``on_piece`` is told how many units are
        back after each piece, never asked.
        """
        results: dict[int, T] = {}
        missing: list[int] = []
        for index, units in enumerate(self.pieces):
            payload = self.load(index)
            value = None if payload is None else decode(payload, units)
            if value is None:
                if self.path(index).exists():
                    self.campaign.log(f"{self.name} piece {index + 1} torn or foreign; redone")
                missing.append(index)
            else:
                results[index] = value
        self._done = len(results)
        if results:
            self.campaign.log(
                f"{self.name} resumed: {len(results)}/{len(self.pieces)} pieces on disk"
            )
        units_done = sum(len(self.pieces[index]) for index in results)
        if results and on_piece is not None:
            on_piece(units_done)
        futures: dict[Future[T], int] = {}
        try:
            for index in missing:
                futures[submit(self.pieces[index])] = index
            for future in as_completed(futures):
                index = futures[future]
                value = future.result()
                self.save(index, encode(value))
                results[index] = value
                units_done += len(self.pieces[index])
                if on_piece is not None:
                    on_piece(units_done)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        if missing:
            self.campaign.log(f"{self.name} complete: {len(self.pieces)} pieces")
        return [results[index] for index in range(len(self.pieces))]


class Saves:
    """A training run's sealed saves, newest last: ``train-000001.ckpt``, ..."""

    def __init__(self, stage: Stage, every_seconds: float) -> None:
        self.stage = stage
        self.every_seconds = every_seconds
        self.last = time.monotonic()

    def _paths(self) -> list[Path]:
        return sorted(self.stage.directory.glob("train-*.ckpt"))

    def due(self) -> bool:
        return time.monotonic() - self.last >= self.every_seconds

    def save(self, arrays: dict[str, Any], meta: dict[str, Any]) -> Path:
        """Write the arrays and ``meta`` (JSON) as the next save, and keep ``KEEP_SAVES``."""
        import numpy as np

        paths = self._paths()
        number = int(paths[-1].stem.split("-")[1]) + 1 if paths else 1
        buffer = io.BytesIO()
        blob = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez(buffer, __meta__=blob, **arrays)
        path = atomic_write(self.stage.directory / f"train-{number:06d}.ckpt", seal(buffer.getvalue()))
        self.last = time.monotonic()
        for old in self._paths()[:-KEEP_SAVES]:
            old.unlink(missing_ok=True)
        self.stage.campaign.log(f"{self.stage.name} saved {path.name} ({meta.get('where', '')})")
        return path

    def latest(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """The newest save whose seal holds, as (arrays, meta); a torn one is skipped."""
        import numpy as np

        for path in reversed(self._paths()):
            payload = read_sealed(path)
            if payload is None:
                self.stage.campaign.log(f"{self.stage.name} {path.name} is torn; the one before it")
                continue
            with np.load(io.BytesIO(payload)) as found:
                arrays = {key: found[key] for key in found.files if key != "__meta__"}
                meta = json.loads(found["__meta__"].tobytes().decode("utf-8"))
            self.stage.campaign.log(f"{self.stage.name} resumes from {path.name}")
            return arrays, meta
        return None


# ------------------------------------------------------------------ the active campaign

_ACTIVE: Campaign | None = None


def current() -> Campaign | None:
    """The campaign this process writes into, or ``None`` (every caller's path of before)."""
    return _ACTIVE


@contextlib.contextmanager
def active(
    root: str | Path,
    save_minutes: float = SAVE_MINUTES,
    command: Sequence[str] | None = None,
    build: str | None = None,
) -> Iterator[Campaign]:
    """Run the block with ``root`` as this process's campaign directory."""
    global _ACTIVE
    previous = _ACTIVE
    campaign = Campaign.open(root, build=build, save_minutes=save_minutes, command=command)
    _ACTIVE = campaign
    outcome = "raised"
    try:
        yield campaign
        outcome = "returned"
    finally:
        _ACTIVE = previous
        what = " ".join(command) if command is not None else "block"
        campaign.log(f"end {what}: {outcome}")


# ------------------------------------------------------------------ games


def agent_identity(agents: Sequence[str], ruleset_hash: str) -> list[dict[str, Any]]:
    """Each agent's name, and the SHA-256 of the net or fit it plays where it names one.

    A net's name is its procedure; the bytes are what plays, so a stage is keyed by them.
    A fitted seat (``fit=<procedure>``, the library roster's, Q900) is the same: its name
    is the procedure and the store's vector is what plays, so the stage is keyed by the
    native spec the core is handed, and a store that changed between a crash and its
    resume plays its pieces again rather than mixing two players under one name.
    """
    from chess_variant_sandbox.agents import native_spec, parse_agent

    out: list[dict[str, Any]] = []
    for agent in agents:
        options = dict(parse_agent(agent).options)
        entry: dict[str, Any] = {"agent": agent}
        if "net" in options:
            from chess_variant_sandbox.agents.nets import net_bytes

            entry["net_sha256"] = hashlib.sha256(net_bytes(options, ruleset_hash)).hexdigest()
        if "fit" in options:
            entry["native_sha256"] = digest(_plain(native_spec(agent, ruleset_hash)))
        out.append(entry)
    return out


def _plain(value: Any) -> Any:
    """A native spec as JSON: tuples as lists, bytes as their SHA-256."""
    if isinstance(value, bytes):
        return "sha256:" + hashlib.sha256(value).hexdigest()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def play_games(
    identity: dict[str, Any],
    agents: Sequence[str],
    tasks: Sequence[tuple[int, str, str]],
    play_piece: Callable[[list[tuple[int, str, str]]], list[Any]],
    workers: int,
    on_progress: Callable[[int], None] | None = None,
) -> list[Any] | None:
    """``tasks``' records through the active campaign, or ``None`` without one.

    ``play_piece`` plays a piece's tasks in order on one thread (the core releases the
    interpreter for each game); ``workers`` pieces run side by side. ``identity`` names
    what else decides the games (the ruleset hash, whether series are kept, the opening).
    """
    campaign = current()
    if campaign is None:
        return None
    from chess_variant_sandbox.tournament.record import GameRecord

    roster = list(dict.fromkeys(agents))
    where = {agent: position for position, agent in enumerate(roster)}
    full = dict(identity, agents=agent_identity(roster, str(identity.get("ruleset_hash", ""))))
    units = [[int(seed), where[first], where[second]] for seed, first, second in tasks]
    stage = campaign.stage("games", full, units)

    def encode(records: list[Any]) -> bytes:
        return json.dumps([record.as_dict() for record in records]).encode("utf-8")

    def decode(payload: bytes, piece: list[Any]) -> list[Any] | None:
        records = [GameRecord.from_dict(row) for row in json.loads(payload.decode("utf-8"))]
        if len(records) != len(piece):
            return None
        for record, (seed, first, second) in zip(records, piece, strict=True):
            if record.seed != seed or record.agents != (roster[first], roster[second]):
                return None
        return records

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pieces = stage.run(
            lambda piece: pool.submit(
                play_piece, [(seed, roster[first], roster[second]) for seed, first, second in piece]
            ),
            encode,
            decode,
            on_progress,
        )
    return [record for piece in pieces for record in piece]


def run_tournament(
    spec: dict[str, Any],
    config: Any,
    progress: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> Any:
    """``runner.run_tournament``, resumable inside a campaign directory.

    Without a campaign it *is* ``runner.run_tournament``. With one, the games both of whose
    seats the core plays go through ``play_games`` (sealed pieces, a piece a thread, one
    core call a piece), the rest through the runner's own Python loop, and every record is
    put back at its place in the schedule, as the runner's ``_play`` does. The runner
    itself is left alone on purpose: it is in the engine fingerprint set
    (``measurement.ENGINE_SOURCES``), and an edit there would stale every library score.
    """
    from chess_variant_sandbox.agents import native_spec
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.spec import spec_hash
    from chess_variant_sandbox.tournament import runner
    from chess_variant_sandbox.tournament.play import build_record, eval_values

    if current() is None:
        return runner.run_tournament(spec, config, progress=progress, on_progress=on_progress)
    tasks = runner.schedule(config)
    engine = Engine(spec)
    natives = [native_spec(agent, engine.spec.hash) for agent in config.agents]
    inside, outside = runner._split_by_native_form(
        tasks, config.agents, natives, core=engine.core is not None
    )
    roster = [
        (agent, native) for agent, native in zip(config.agents, natives, strict=True)
        if native is not None
    ]
    position_of = {agent: position for position, (agent, _) in enumerate(roster)}
    specs = [native for _, native in roster]
    values = eval_values(engine.spec) if inside else []

    def piece(chunk: list[tuple[int, str, str]]) -> list[Any]:
        results = engine.core.run_tournament(
            specs,
            values,
            [(seed, position_of[first], position_of[second]) for seed, first, second in chunk],
            config.collect_series,
            1,
        )
        return [
            build_record(engine, (first, second), seed, result)
            for (seed, first, second), result in zip(chunk, results, strict=True)
        ]

    placed: dict[int, Any] = {}
    if inside:
        played = play_games(
            {"ruleset_hash": engine.spec.hash, "collect_series": config.collect_series},
            [agent for agent, _ in roster],
            [tasks[index] for index in inside],
            piece,
            min(config.resolved_workers(), len(inside)),
            on_progress=on_progress,
        )
        assert played is not None
        placed.update(zip(inside, played, strict=True))
        if progress:
            print(f"  {len(inside)}/{len(tasks)} games", flush=True)
    if outside:
        loop = runner._run_python(
            engine, [tasks[index] for index in outside], config.collect_series, progress,
            done=len(inside), total=len(tasks), on_progress=on_progress,
        )
        placed.update(zip(outside, loop, strict=True))
    return runner.TournamentResult(
        ruleset_hash=spec_hash(spec),
        ruleset_name=spec["name"],
        config=config,
        records=[placed[index] for index in range(len(tasks))],
    )
