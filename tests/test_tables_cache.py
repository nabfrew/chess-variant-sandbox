"""`tests/tables_cache.py` re-answers a stale seed under the current key, and only that key.

Two small library files in a temporary root stand in for the library; a seed naming them
under a key no compiler has stands in for the checkout's cache after a compiler edit.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import tables_cache

from chess_variant_sandbox import measurement
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
FILES = ("dull.json", "losalamos.json")


@pytest.fixture
def library(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    root = tmp_path / "rulesets"
    root.mkdir()
    specs = {}
    for name in FILES:
        shutil.copyfile(RULESETS / name, root / name)
        spec = load_spec(root / name)
        specs[spec_hash(spec)] = spec
    return root, specs


def _seed(directory: Path, key_tables: str, key_values: str, digests: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tables_cache.write(directory / tables_cache.TABLES_NAME, key_tables, dict.fromkeys(digests, "0" * 16))
    tables_cache.write(directory / tables_cache.VALUES_NAME, key_values, dict.fromkeys(digests, "0" * 16))
    return directory / tables_cache.TABLES_NAME


def _warm(tmp_path: Path, seed: Path, root: Path, worker: int = 0, workers: int = 1) -> tuple[Path, dict[str, int]]:
    cache_dir = tmp_path / f"cache-{worker}"
    counts = tables_cache.warm(
        cache_dir, seed, shared=tmp_path / "shared", scratch=tmp_path / f"scratch-{worker}",
        roots=(root,), worker=worker, workers=workers,
    )
    return cache_dir, counts


def test_a_stale_seed_is_answered_again_under_the_current_key(tmp_path, library, monkeypatch):
    root, specs = library
    seed = _seed(tmp_path / "seed", "an-older-compiler", "an-older-compiler", sorted(specs))
    cache_dir, counts = _warm(tmp_path, seed, root)
    assert counts == {"stale": 2, "warmed": 2, "claimed": len(specs)}
    key, answers = tables_cache.stored(cache_dir / tables_cache.TABLES_NAME)
    assert key == measurement.compile_source_key()
    assert answers == {d: measurement.tables_fingerprint(s) for d, s in specs.items()}
    key, values = tables_cache.stored(cache_dir / tables_cache.VALUES_NAME)
    assert key == measurement.values_source_key()
    assert values == {d: measurement.values_fingerprint(s) for d, s in specs.items()}
    # The seed is left as it was: nothing is written back to the checkout's file.
    assert tables_cache.stored(seed)[0] == "an-older-compiler"

    # And `measurement` reads what was written, compiling nothing.
    monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(cache_dir / tables_cache.TABLES_NAME))
    monkeypatch.setattr(measurement, "_TABLES", {})
    monkeypatch.setattr(measurement, "_VALUES", {})

    def no_compile(*_: object) -> None:
        raise AssertionError("compiled where the warmed cache had the answer")

    monkeypatch.setattr(measurement, "_compiled", no_compile)
    for digest, spec in specs.items():
        assert measurement.tables_fingerprint(spec) == answers[digest]
        assert measurement.values_fingerprint(spec) == values[digest]


def test_a_second_worker_reads_the_slices_the_first_answered(tmp_path, library):
    root, specs = library
    seed = _seed(tmp_path / "seed", "an-older-compiler", "an-older-compiler", sorted(specs))
    first, _ = _warm(tmp_path, seed, root, worker=0, workers=2)
    second, counts = _warm(tmp_path, seed, root, worker=1, workers=2)
    assert counts["claimed"] == 0 and counts["warmed"] == 2
    for name in (tables_cache.TABLES_NAME, tables_cache.VALUES_NAME):
        assert tables_cache.stored(second / name) == tables_cache.stored(first / name)


def test_a_slice_answered_under_another_key_is_dropped_unread(tmp_path, library):
    root, specs = library
    seed = _seed(tmp_path / "seed", "an-older-compiler", "an-older-compiler", sorted(specs))
    shared = tmp_path / "shared"
    shared.mkdir()
    for j in range(tables_cache.SHARDS):  # another worker, on other sources, took them all
        (shared / f"shard-{j}.claim").touch()
        (shared / f"shard-{j}.json").write_text(json.dumps({
            "tables_key": "other", "values_key": "other",
            "tables": dict.fromkeys(specs, "f" * 16), "values": dict.fromkeys(specs, "f" * 16),
        }), encoding="utf-8")
    cache_dir, counts = _warm(tmp_path, seed, root)
    assert counts == {"stale": 2, "warmed": 0, "claimed": 0}
    assert not (cache_dir / tables_cache.TABLES_NAME).exists()


def test_a_current_seed_is_copied_and_nothing_compiled(tmp_path, library):
    root, specs = library
    digests = sorted(specs)
    seed = _seed(tmp_path / "seed", measurement.compile_source_key(),
                 measurement.values_source_key(), digests)
    cache_dir, counts = _warm(tmp_path, seed, root)
    assert counts == {"stale": 0, "warmed": 0, "claimed": 0}
    assert tables_cache.stored(cache_dir / tables_cache.TABLES_NAME) == tables_cache.stored(seed)
    assert not (tmp_path / "shared").exists()


def test_no_seed_costs_nothing(tmp_path, library):
    root, _ = library
    cache_dir, counts = _warm(tmp_path, tmp_path / "absent" / tables_cache.TABLES_NAME, root)
    assert counts == {"stale": 0, "warmed": 0, "claimed": 0}
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_measurements_and_no_seed_are_answered_for_every_game(tmp_path, library):
    """Q893: a `runs/` that holds measurements asks every game's fingerprint of a listing,
    seed or no seed, so the run answers them in setup rather than in the first test to list."""
    root, specs = library
    runs = tmp_path / "runs"
    (runs / "0123456789ab").mkdir(parents=True)
    (runs / "0123456789ab" / "canonical.json").write_text("{}", encoding="utf-8")
    cache_dir, counts = _warm(tmp_path, runs / tables_cache.TABLES_NAME, root)
    assert counts == {"stale": len(specs), "warmed": len(specs), "claimed": len(specs)}
    key, answers = tables_cache.stored(cache_dir / tables_cache.TABLES_NAME)
    assert key == measurement.compile_source_key()
    assert answers == {d: measurement.tables_fingerprint(s) for d, s in specs.items()}
