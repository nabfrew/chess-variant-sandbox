"""The test session's tables cache, warmed once per run when the checkout's is stale (Q882).

`conftest.tables_are_cached_somewhere_temporary` gives each test process its own copy of
the checkout's `runs/tables-fingerprints.json` and `values-fingerprints.json` (Q874). That
copy answers only under the compiler-source key it was stored under, and the checkout's
file moves to a new key only when something outside the suite (`measure --check`, a
listing, `serve`) lists the library again. So the first fast run after a change to
`engine/`, `spec/` or anything else `measurement.compile_sources` reads found every answer
dropped, and each of the four workers compiled the library inside whichever test listed it
first: 13.3 s of `test_added`'s card test on one core, where a current seed costs 4.1 s.

This moves that compile out of the tests and divides it between the workers. When the
seed is stale, every library file whose hash the seed held an answer for is re-answered
under the current key, in `SHARDS` slices: each worker claims a slice it finds unclaimed
(an exclusive create in a directory every worker of the run shares), answers it with the
ordinary `measurement.tables_fingerprint` and `values_fingerprint`, and writes the slice's
answers beside the claim. Then it waits for every slice and writes its own cache from them.
All of it runs in a session fixture's setup, which the duration guard charges to no test.

**No answer is read under a key it was not computed under.** A slice records the keys its
worker computed under, and is dropped unread when they are not this process's; the seed's
answers are kept only when the seed's key is current. Nothing is written to the checkout's
`runs/`: the cache files written here are the session's own, in the pytest temp dir.

The format written is `measurement`'s own (`{"sources": key, "tables": {...}}`, for both
files); `tests/test_tables_cache.py` holds the two together by reading one back through
`measurement`.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# More slices than workers, so a worker that finishes a cheap slice takes another and a
# wrapped board does not leave three workers waiting on one.
SHARDS = 16
# How long a worker waits for slices another claimed. Past it, a slice whose worker died
# is given up on: its games compile lazily, as they did before, and nothing is wrong.
WAIT_S = 300.0
POLL_S = 0.05

TABLES_NAME = "tables-fingerprints.json"
VALUES_NAME = "values-fingerprints.json"


def stored(path: Path) -> tuple[str | None, dict[str, str]]:
    """A cache file's key and answers, or ``(None, {})`` when it cannot be read."""
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, {}
    if not isinstance(held, dict) or not isinstance(held.get("tables"), dict):
        return None, {}
    sources = held.get("sources")
    return (sources if isinstance(sources, str) else None), {
        str(k): str(v) for k, v in held["tables"].items()
    }


def write(path: Path, key: str, answers: dict[str, str]) -> None:
    """One cache file in `measurement`'s format, written whole and renamed into place."""
    payload = json.dumps(
        {"sources": key, "tables": dict(sorted(answers.items()))},
        indent=0, separators=(",", ":"),
    )
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def library_specs(roots: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    """Every spec file directly under ``roots``, keyed by its mechanics hash."""
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.spec.canonical import spec_hash

    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(Path(root).glob("*.json")):
            try:
                spec = load_spec(path)
                found[spec_hash(spec)] = spec
            except Exception:  # noqa: BLE001 -- a listing reports a bad file its own way
                continue
    return found


@contextlib.contextmanager
def _cache_at(directory: Path) -> Iterator[None]:
    """`measurement` caches in ``directory`` for the length of the block."""
    from chess_variant_sandbox.measurement import TABLES_CACHE_ENV

    previous = os.environ.get(TABLES_CACHE_ENV)
    os.environ[TABLES_CACHE_ENV] = str(directory / TABLES_NAME)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TABLES_CACHE_ENV, None)
        else:
            os.environ[TABLES_CACHE_ENV] = previous


def _answer(shard: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    from chess_variant_sandbox import measurement

    tables: dict[str, str] = {}
    values: dict[str, str] = {}
    for digest, spec in shard:
        with contextlib.suppress(Exception):  # a spec that will not compile stays a miss
            tables[digest] = measurement.tables_fingerprint(spec)
            values[digest] = measurement.values_fingerprint(spec)
    return {
        "tables_key": measurement.compile_source_key(),
        "values_key": measurement.values_source_key(),
        "tables": tables,
        "values": values,
    }


def warm(
    cache_dir: Path,
    seed: Path,
    shared: Path,
    scratch: Path,
    roots: tuple[Path, ...],
    worker: int = 0,
    workers: int = 1,
) -> dict[str, int]:
    """Write ``cache_dir``'s two cache files from the seed, re-answering what it held stale.

    ``seed`` is the checkout's tables cache, its values cache beside it as `measurement`
    keeps them. ``shared`` is a directory every worker of the run sees and no other run does;
    ``scratch`` is this worker's own, where `measurement` writes while a slice is answered
    (so its in-process copy of the cache is of that file, and the one written here is
    read afresh when the tests start). Returns counts, for a test and for the record.
    """
    from chess_variant_sandbox import measurement

    counts = {"stale": 0, "warmed": 0, "claimed": 0}
    seeds = {
        TABLES_NAME: stored(seed),
        VALUES_NAME: stored(seed.with_name(VALUES_NAME)),
    }
    seeded = any(answers for _, answers in seeds.values())
    if not seeded and not any(seed.parent.glob("*/canonical.json")):
        return counts  # a worktree's `runs/`: nothing was asked, so nothing to answer
    keys = {
        TABLES_NAME: measurement.compile_source_key(),
        VALUES_NAME: measurement.values_source_key(),
    }
    # A `runs/` that holds measurements but no seed (Q893): a listing asks every game's
    # fingerprint all the same, and whichever test lists first would compile the library
    # (11.8 s of `test_launch`, 8.6 s of `test_site` on one run), so every game is answered.
    stale: set[str] = set() if seeded else set(library_specs(roots))
    kept: dict[str, dict[str, str]] = {}
    for name, (key, answers) in seeds.items():
        if key == keys[name]:
            kept[name] = dict(answers)
        else:
            kept[name] = {}
            stale |= set(answers)
    counts["stale"] = len(stale)
    if stale:
        wanted = sorted((d, s) for d, s in library_specs(roots).items() if d in stale)
        shards = [wanted[j::SHARDS] for j in range(SHARDS)]
        shared.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        for name in seeds:  # a slice's current half is a hit, not a compile
            if kept[name]:
                write(scratch / name, keys[name], kept[name])
        start = (worker * SHARDS) // max(workers, 1)
        for j in [(start + step) % SHARDS for step in range(SHARDS)]:
            if not shards[j]:
                continue
            try:
                (shared / f"shard-{j}.claim").open("x").close()
            except FileExistsError:
                continue
            counts["claimed"] += 1
            try:
                with _cache_at(scratch):
                    result = _answer(shards[j])
            except Exception:  # noqa: BLE001 -- the slice is still written, empty
                result = {}
            done = shared / f"shard-{j}.json"
            temporary = done.with_name(f"{done.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(result), encoding="utf-8")
            os.replace(temporary, done)
        deadline = time.monotonic() + WAIT_S
        for j, shard in enumerate(shards):
            done = shared / f"shard-{j}.json"
            while shard and not done.exists() and time.monotonic() < deadline:
                time.sleep(POLL_S)
            if not shard:
                continue
            try:
                result = json.loads(done.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for name, half in ((TABLES_NAME, "tables"), (VALUES_NAME, "values")):
                if result.get(f"{half}_key") == keys[name]:
                    kept[name].update(result.get(half) or {})
        counts["warmed"] = len(kept[TABLES_NAME].keys() & stale)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, answers in kept.items():
        if answers:
            write(cache_dir / name, keys[name], answers)
    return counts
