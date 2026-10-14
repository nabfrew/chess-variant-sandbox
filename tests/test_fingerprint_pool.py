"""The cold fingerprint fill (Q904): a process pool, and a seed from the main checkout.

Neither changes an answer. The pool calls the functions the serial path calls, so its
fingerprints are byte-identical at any worker count; the seed is read only under the key
the local file would be written under, so a match is the same bytes the local compile
would produce. `docs/reference/perf-survey-2026-09.md` 5.9.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chess_variant_sandbox import measurement
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash

REPO = Path(__file__).resolve().parents[1]
SPREAD = ("dull", "losalamos", "gardner")
TABLES = "tables-fingerprints.json"
VALUES = "values-fingerprints.json"


def _wipe() -> None:
    """A cold cache: no file on disk either."""
    for path in (measurement.tables_cache_path(), measurement.values_cache_path()):
        path.unlink(missing_ok=True)


def _forget() -> None:
    """A fresh process's view of the answers."""
    measurement._TABLES.clear()
    measurement._DISK.update(path=None, key=None, tables={})
    measurement._VALUES.clear()
    measurement._VALUES_DISK.update(path=None, key=None, tables={})
    measurement._LAST_COMPILED.clear()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(tmp_path / "cache" / TABLES))
    _forget()
    yield
    monkeypatch.undo()
    _forget()


def _library(names: tuple[str, ...]) -> list[dict]:
    return [load_spec(REPO / "rulesets" / f"{name}.json") for name in names]


def _serial(specs: list[dict]) -> dict[str, tuple[str, str]]:
    """The reference: the ordinary lazy path, one spec at a time, in this process."""
    answers = {
        spec_hash(spec): (measurement.tables_fingerprint(spec), measurement.values_fingerprint(spec))
        for spec in specs
    }
    _forget()
    return answers


def _pooled(specs: list[dict], workers: int | None) -> tuple[int, dict[str, tuple[str, str]]]:
    """What `warm_fingerprints` filled, read from memory and from the files it wrote."""
    _forget()
    _wipe()
    done = measurement.warm_fingerprints(specs, workers=workers)
    key = measurement.compile_source_key()
    in_memory = {d: (measurement._TABLES[d], measurement._VALUES[d]) for d in measurement._TABLES}
    on_disk = measurement._stored_under(measurement.tables_cache_path(), key)
    assert on_disk == {d: t for d, (t, _) in in_memory.items()}
    values = measurement._stored_under(measurement.values_cache_path(), measurement.values_source_key())
    assert values == {d: v for d, (_, v) in in_memory.items()}
    return done, in_memory


class TestThePoolChangesNoAnswer:
    @pytest.mark.parametrize("workers", ["one", "two", "default"])
    def test_a_spread_of_library_specs_is_byte_identical(self, workers):
        count = {"one": 1, "two": 2, "default": measurement.fingerprint_workers()}[workers]
        specs = _library(SPREAD)
        expected = _serial(specs)
        assert len(expected) == len(SPREAD)
        done, got = _pooled(specs, count)
        assert done == (0 if count == 1 else len(specs))
        if count == 1:  # the serial path stays: nothing was filled, the lazy path answers
            assert got == {}
            got = {
                spec_hash(s): (measurement.tables_fingerprint(s), measurement.values_fingerprint(s))
                for s in specs
            }
        assert got == expected

    def test_one_or_two_cold_specs_are_left_to_the_serial_path(self):
        specs = _library(SPREAD[: measurement.POOL_MIN_SPECS - 1])
        assert measurement.warm_fingerprints(specs, workers=4) == 0
        assert measurement._TABLES == {}

    def test_what_is_already_cached_is_not_asked_of_a_worker(self):
        specs = _library(SPREAD)
        _serial(specs)  # fills the disk cache
        assert measurement.warm_fingerprints(specs, workers=4) == 0

    @pytest.mark.slow
    def test_every_library_spec_is_byte_identical_at_one_two_and_the_default(self):
        from chess_variant_sandbox import corpus

        specs = [
            load_spec(path)
            for directory in (REPO / corpus.RULESET_DIR, REPO / corpus.ARCHIVE_DIR)
            for path in sorted(directory.glob("*.json"))
        ]
        expected = _serial(specs)
        for workers in (1, 2, measurement.fingerprint_workers()):
            _forget()
            done, got = _pooled(specs, workers)
            if workers == 1:
                assert done == 0
                continue
            assert done == len(expected)
            assert got == expected, f"workers={workers}"

    def test_the_worker_count_is_bounded_and_the_variable_moves_it(self, monkeypatch):
        monkeypatch.delenv(measurement.FINGERPRINT_WORKERS_ENV, raising=False)
        assert 1 <= measurement.fingerprint_workers() <= 4
        monkeypatch.setenv(measurement.FINGERPRINT_WORKERS_ENV, "7")
        assert measurement.fingerprint_workers() == 7
        monkeypatch.setenv(measurement.FINGERPRINT_WORKERS_ENV, "nonsense")
        assert 1 <= measurement.fingerprint_workers() <= 4


def _main_cache(main_runs: Path, key: str, answers: dict[str, str]) -> dict[str, bytes]:
    main_runs.mkdir(parents=True)
    for name in (TABLES, VALUES):
        (main_runs / name).write_text(
            json.dumps({"sources": key, "tables": answers}), encoding="utf-8"
        )
    return {name: (main_runs / name).read_bytes() for name in (TABLES, VALUES)}


@pytest.fixture
def worktree(tmp_path, monkeypatch):
    """This process as a worktree: a default cache location of its own, a stand-in main."""
    main_runs = tmp_path / "main" / "runs"
    local = tmp_path / "worktree" / "runs" / TABLES
    monkeypatch.delenv(measurement.TABLES_CACHE_ENV)
    monkeypatch.setattr(measurement, "TABLES_CACHE", local)
    monkeypatch.setattr(measurement, "main_runs_dir", lambda: main_runs)
    return main_runs, local


class TestAWorktreeSeedsFromTheMainCheckout:
    def test_a_matching_key_is_seeded_and_main_is_not_written(self, worktree, monkeypatch):
        main_runs, local = worktree
        spec = _library(("dull",))[0]
        digest = spec_hash(spec)
        before = _main_cache(main_runs, measurement.compile_source_key(), {digest: "a" * 16})
        # Values live under their own key; give main's the values key.
        (main_runs / VALUES).write_text(
            json.dumps({"sources": measurement.values_source_key(), "tables": {digest: "b" * 16}}),
            encoding="utf-8",
        )
        before[VALUES] = (main_runs / VALUES).read_bytes()

        def no_compile(*_: object) -> None:
            raise AssertionError("compiled where main's cache had the answer")

        monkeypatch.setattr(measurement, "_compiled", no_compile)
        assert measurement.tables_fingerprint(spec) == "a" * 16
        assert measurement.values_fingerprint(spec) == "b" * 16
        # The local files now hold the seed, under the same keys.
        assert measurement._stored_under(local, measurement.compile_source_key()) == {digest: "a" * 16}
        assert measurement._stored_under(
            local.with_name(VALUES), measurement.values_source_key()
        ) == {digest: "b" * 16}
        # And main's files are byte-for-byte what they were.
        assert {name: (main_runs / name).read_bytes() for name in (TABLES, VALUES)} == before

    def test_a_different_key_is_not_seeded(self, worktree):
        main_runs, local = worktree
        spec = _library(("dull",))[0]
        digest = spec_hash(spec)
        before = _main_cache(main_runs, "a-key-no-tree-has", {digest: "a" * 16})
        # The answer is the compile's own, never main's, and is stored under this tree's key.
        assert measurement.tables_fingerprint(spec) == measurement._compute_tables(spec, digest)
        assert measurement.tables_fingerprint(spec) != "a" * 16
        assert measurement.values_fingerprint(spec) != "a" * 16
        assert measurement._stored_under(local, measurement.compile_source_key()) == {
            digest: measurement.tables_fingerprint(spec)
        }
        assert {name: (main_runs / name).read_bytes() for name in (TABLES, VALUES)} == before

    def test_a_cache_moved_by_the_variable_is_never_seeded(self, worktree, monkeypatch, tmp_path):
        main_runs, _ = worktree
        spec = _library(("dull",))[0]
        _main_cache(main_runs, measurement.compile_source_key(), {spec_hash(spec): "a" * 16})
        monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(tmp_path / "moved" / TABLES))
        assert measurement.tables_fingerprint(spec) != "a" * 16

    def test_a_local_file_under_the_current_key_is_kept(self, worktree):
        main_runs, local = worktree
        spec = _library(("dull",))[0]
        digest = spec_hash(spec)
        _main_cache(main_runs, measurement.compile_source_key(), {digest: "a" * 16})
        local.parent.mkdir(parents=True)
        local.write_text(
            json.dumps({"sources": measurement.compile_source_key(), "tables": {digest: "c" * 16}}),
            encoding="utf-8",
        )
        assert measurement.tables_fingerprint(spec) == "c" * 16


class TestTheMainCheckoutIsFoundFromGit:
    def test_it_is_the_parent_of_gits_common_dir(self):
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        main = (REPO / common).resolve().parent
        found = measurement.main_runs_dir()
        assert found == (None if main == REPO.resolve() else main / "runs")

    def test_a_worktrees_pointer_files_are_followed(self, tmp_path, monkeypatch):
        main = tmp_path / "main"
        admin = main / ".git" / "worktrees" / "w"
        admin.mkdir(parents=True)
        (admin / "commondir").write_text("../..\n", encoding="utf-8")
        root = tmp_path / "w"
        root.mkdir()
        (root / ".git").write_text(f"gitdir: {admin.as_posix()}\n", encoding="utf-8")
        monkeypatch.setattr(measurement, "ROOT", root)
        assert measurement.main_runs_dir() == main.resolve() / "runs"

    def test_the_main_checkout_has_nothing_to_seed_from(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(measurement, "ROOT", tmp_path)
        assert measurement.main_runs_dir() is None
