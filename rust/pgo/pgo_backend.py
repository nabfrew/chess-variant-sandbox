"""The package's build backend: maturin's, with the core's PGO profile when it fits (Q1142).

`pyproject.toml` names this module as `build-backend` (with `backend-path = ["rust/pgo"]`),
so every PEP 517 build of the package -- `uv sync`, `uv build`, `pip install .`, `pyodide
build` -- comes through here. It does one thing before handing the call to maturin unchanged:
when the committed profile `rust/pgo/core.profdata` was made by the `rustc` and host that
are about to build, it sets `RUSTFLAGS=-Cprofile-use=<absolute path>` for that build. So a
plain `uv sync --reinstall-package chess-variant-sandbox` builds the same core as
`rust/pgo/regenerate.py --build` (docs/reference/perf-c8-pgo.md).

Why here and not in cargo's or maturin's config: the flag needs the profile's path, and no
static config can give one that holds for every crate. cargo runs `rustc` for a registry
dependency from that dependency's own directory, for this crate from `rust/`, and for its
`--print cfg` probe from the repository root, so a relative path is missing for two of the
three and rustc stops (`file ... passed to -C profile-use does not exist`); an absolute one
is machine-specific. A build backend is code, so it can compute the path, and it can decline.

It declines, with a `pgo:` line on stderr, and builds exactly as maturin would without it:

- `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` is already set. The caller chose its flags:
  `regenerate.py` (training sets `-Cprofile-generate`, which `-Cprofile-use` may not join),
  `pyodide build` (the browser core), an aligned-arms bench.
- `CARGO_BUILD_TARGET` names another target than the profile's host.
- The profile or its stamp `core.profdata.json` is missing.
- The stamp's `rustc` or host is not this one's. A profile from another compiler matches no
  function, and an older LLVM may not read it; the box trains its own (deploy/remote/build.sh).
- Anything here fails at all. PGO only ever makes the core faster; it never costs a build.

A profile trained on older `rust/src` is still used, with a `pgo: stale` line: functions
whose control flow moved lose their profile, the rest keep it, and the answer never changes.
`uv sync` shows a backend's stderr only with `-v`; `python rust/pgo/regenerate.py --check`
asks the same question at any time.

`maturin develop` does not come through here (it is not a PEP 517 build), so it builds
without PGO; `.claude/skills/bench-core` names the command that builds the shipped core.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import maturin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILE = HERE / "core.profdata"
STAMP = HERE / "core.profdata.json"


def _say(message: str) -> None:
    print(f"pgo: {message}", file=sys.stderr, flush=True)


def _toolchain() -> tuple[str, str]:
    """The `rustc -vV` first line and host of the compiler cargo will use for `rust/`."""
    rustc = os.environ.get("RUSTC", "rustc")
    # From rust/, so a rustup override there (a toolchain file) is the one asked.
    out = subprocess.run([rustc, "-vV"], cwd=ROOT / "rust", capture_output=True, text=True,
                         check=True, timeout=60).stdout
    fields = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    return out.splitlines()[0], fields["host"]


def _source_fingerprint() -> str | None:
    sys.path.insert(0, str(ROOT))
    try:
        from chess_variant_sandbox import measurement

        return measurement.rust_source_fingerprint()
    finally:
        sys.path.remove(str(ROOT))


def profile_flags() -> str | None:
    """The `RUSTFLAGS` value that builds with the profile, or None (said why) to build without."""
    for variable in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        if os.environ.get(variable):
            _say(f"{variable} is set by the caller; building with it, without the profile")
            return None
    if not PROFILE.is_file() or not STAMP.is_file():
        _say(f"no {PROFILE.relative_to(ROOT).as_posix()} or its stamp; building without PGO")
        return None
    stamp = json.loads(STAMP.read_text(encoding="utf-8"))
    rustc, host = _toolchain()
    target = os.environ.get("CARGO_BUILD_TARGET")
    if target and target != stamp["host"]:
        _say(f"building for {target}, the profile is {stamp['host']}'s; building without PGO")
        return None
    if (stamp["rustc"], stamp["host"]) != (rustc, host):
        _say(f"stale: made by {stamp['rustc']} on {stamp['host']}, this is {rustc} on {host}; "
             "building without PGO (`python rust/pgo/regenerate.py` trains this machine's)")
        return None
    source = _source_fingerprint()
    if source is not None and stamp.get("source") != source:
        _say(f"stale: trained on source {stamp.get('source')}, rust/src is now {source}; "
             "building with it anyway (`python rust/pgo/regenerate.py` refreshes it)")
    _say(f"building with {PROFILE.relative_to(ROOT).as_posix()}")
    return f"-Cprofile-use={PROFILE}"


def _with_profile(hook):
    def build(*args, **kwargs):
        try:
            flags = profile_flags()
        except Exception as error:  # never let PGO cost a build its core
            _say(f"could not check the profile ({error!r}); building without PGO")
            flags = None
        if flags is None:
            return hook(*args, **kwargs)
        os.environ["RUSTFLAGS"] = flags
        try:
            return hook(*args, **kwargs)
        finally:
            del os.environ["RUSTFLAGS"]

    build.__name__ = hook.__name__
    build.__doc__ = hook.__doc__
    return build


build_wheel = _with_profile(maturin.build_wheel)
build_editable = _with_profile(maturin.build_editable)

# The rest of PEP 517's hooks, maturin's own, unchanged (an absent optional hook stays absent).
for _name in (
    "build_sdist",
    "get_requires_for_build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
):
    if hasattr(maturin, _name):
        globals()[_name] = getattr(maturin, _name)
del _name
