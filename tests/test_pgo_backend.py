"""The PGO build backend's decisions (Q1143): when it declines and what it says, when it accepts.

`rust/pgo/pgo_backend.py` is a PEP 517 backend that adds `-Cprofile-use=<profile>` to the
build when the committed profile fits the compiler, and otherwise builds as maturin would.
Nothing here builds: `maturin` is a stub, `_toolchain` and `_source_fingerprint` are
monkeypatched, and the profile and stamp are files in a temp directory.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "rust" / "pgo" / "pgo_backend.py"
RUSTC = "rustc 1.99.0 (abc 2030-01-01)"
HOST = "x86_64-test-host"


@pytest.fixture
def backend(monkeypatch, tmp_path):
    stub = types.ModuleType("maturin")
    stub.build_wheel = lambda *a, **k: ("wheel", __import__("os").environ.get("RUSTFLAGS"))
    stub.build_editable = lambda *a, **k: ("editable", __import__("os").environ.get("RUSTFLAGS"))
    stub.build_sdist = lambda *a, **k: "sdist"
    monkeypatch.setitem(sys.modules, "maturin", stub)
    spec = importlib.util.spec_from_file_location("pgo_backend_under_test", BACKEND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The profile and its stamp live in a temp tree shaped like the repository's.
    root = tmp_path / "repo"
    pgo = root / "rust" / "pgo"
    pgo.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "PROFILE", pgo / "core.profdata")
    monkeypatch.setattr(module, "STAMP", pgo / "core.profdata.json")
    monkeypatch.setattr(module, "_toolchain", lambda: (RUSTC, HOST))
    monkeypatch.setattr(module, "_source_fingerprint", lambda: "src-1")
    for variable in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"):
        monkeypatch.delenv(variable, raising=False)
    return module


def _write_profile(module, **stamp) -> None:
    module.PROFILE.write_bytes(b"profile")
    fields = {"rustc": RUSTC, "host": HOST, "source": "src-1"} | stamp
    module.STAMP.write_text(json.dumps(fields), encoding="utf-8")


def test_accepts_a_matching_profile(backend, capsys) -> None:
    _write_profile(backend)
    assert backend.profile_flags() == f"-Cprofile-use={backend.PROFILE}"
    err = capsys.readouterr().err
    assert err == "pgo: building with rust/pgo/core.profdata\n"


def test_accepts_a_profile_trained_on_older_source_and_says_stale(backend, capsys) -> None:
    _write_profile(backend, source="src-0")
    assert backend.profile_flags() == f"-Cprofile-use={backend.PROFILE}"
    err = capsys.readouterr().err
    assert "pgo: stale: trained on source src-0, rust/src is now src-1" in err
    assert "building with it anyway" in err
    assert err.rstrip().endswith("building with rust/pgo/core.profdata")


@pytest.mark.parametrize("variable", ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"])
def test_declines_when_the_caller_set_its_own_flags(backend, monkeypatch, capsys, variable) -> None:
    _write_profile(backend)
    monkeypatch.setenv(variable, "-Ctarget-cpu=native")
    assert backend.profile_flags() is None
    assert f"pgo: {variable} is set by the caller; building with it, without the profile" in (
        capsys.readouterr().err)


def test_declines_without_a_profile(backend, capsys) -> None:
    backend.STAMP.write_text("{}", encoding="utf-8")
    assert backend.profile_flags() is None
    assert "no rust/pgo/core.profdata or its stamp; building without PGO" in capsys.readouterr().err


def test_declines_without_a_stamp(backend, capsys) -> None:
    backend.PROFILE.write_bytes(b"profile")
    assert backend.profile_flags() is None
    assert "no rust/pgo/core.profdata or its stamp; building without PGO" in capsys.readouterr().err


def test_declines_for_another_target(backend, monkeypatch, capsys) -> None:
    _write_profile(backend)
    monkeypatch.setenv("CARGO_BUILD_TARGET", "wasm32-unknown-unknown")
    assert backend.profile_flags() is None
    assert (f"building for wasm32-unknown-unknown, the profile is {HOST}'s; building without PGO"
            in capsys.readouterr().err)


def test_accepts_when_the_target_is_the_profiles_host(backend, monkeypatch) -> None:
    _write_profile(backend)
    monkeypatch.setenv("CARGO_BUILD_TARGET", HOST)
    assert backend.profile_flags() == f"-Cprofile-use={backend.PROFILE}"


@pytest.mark.parametrize("stamp", [{"rustc": "rustc 1.50.0 (old)"}, {"host": "other-host"}])
def test_declines_a_stale_toolchain(backend, capsys, stamp) -> None:
    _write_profile(backend, **stamp)
    assert backend.profile_flags() is None
    err = capsys.readouterr().err
    assert err.startswith("pgo: stale: made by ")
    assert f"this is {RUSTC} on {HOST}; building without PGO" in err


def test_a_failing_check_builds_without_pgo(backend, monkeypatch, capsys) -> None:
    _write_profile(backend)

    def boom():
        raise OSError("no rustc")

    monkeypatch.setattr(backend, "_toolchain", boom)
    assert backend.build_wheel("out") == ("wheel", None)
    assert "could not check the profile (OSError('no rustc')); building without PGO" in (
        capsys.readouterr().err)


def test_the_wrapped_hook_sets_the_flag_for_the_build_only(backend, monkeypatch) -> None:
    import os

    _write_profile(backend)
    assert backend.build_wheel("out") == ("wheel", f"-Cprofile-use={backend.PROFILE}")
    assert backend.build_editable("out") == ("editable", f"-Cprofile-use={backend.PROFILE}")
    assert "RUSTFLAGS" not in os.environ


def test_a_declined_hook_passes_through_unchanged(backend) -> None:
    assert backend.build_wheel("out") == ("wheel", None)
    assert backend.build_sdist("out") == "sdist"
