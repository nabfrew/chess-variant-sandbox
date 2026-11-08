"""The core's PGO profile, `rust/pgo/core.profdata`: regenerate it, check it, build with it.

    python rust/pgo/regenerate.py [-- <uv sync arguments>]          train, then build
    python rust/pgo/regenerate.py --build [-- <uv sync arguments>]  build with the profile on disk
    python rust/pgo/regenerate.py --check                           is the profile this machine's?

Training (the default; the one command after a change to the core's hot path):

1. Builds an instrumented core (`RUSTFLAGS=-Cprofile-generate`) into the venv by the same
   `uv sync` that builds the real one (see `sync`), and runs `train.py` with it.
2. If anything fails, rebuilds with the profile already on disk, so the venv never keeps
   the instrumented core.
3. Merges the raw profiles with the toolchain's own `llvm-profdata` (the `llvm-tools`
   component: `rustup component add llvm-tools`) into `core.profdata`, and writes
   `core.profdata.json`: the `rustc -vV` and host that made it, and the core's source
   fingerprint it was trained on.
4. Builds, as `--build` does.

`--build` runs `uv sync --reinstall-package chess-variant-sandbox` plus the arguments after
`--` (default `--extra dev`) with `RUSTFLAGS=-Cprofile-use=<this directory>/core.profdata`.
Since Q1142 a plain `uv sync --reinstall-package chess-variant-sandbox` builds the same core:
the package's build backend (`pgo_backend.py`, beside this file) sets the same flag when the
profile is this toolchain's, and declines otherwise (another compiler or host, the browser
core's target, a missing profile). It leaves a caller's own `RUSTFLAGS` alone, so this
script's instrumented build never meets `-Cprofile-use`. `maturin develop` builds without it.

`--check` says whether the committed profile was made by this toolchain on this source
(exit 0) or not (exit 1, with the reason). A profile from another `rustc` or host matches no
function (cargo's symbol hashes carry the compiler's version), so such a machine trains its
own; the box does this at every build. A profile trained on older source still builds:
functions whose control flow changed lose their profile and compile as without PGO.

Training needs `heavy` (two builds and about a minute's play); `--build` needs `build`.
Record: docs/reference/perf-c8-pgo.md.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILE = HERE / "core.profdata"
STAMP = HERE / "core.profdata.json"


def toolchain() -> dict:
    vv = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True).stdout
    fields = dict(line.split(": ", 1) for line in vv.splitlines() if ": " in line)
    return {"rustc": vv.splitlines()[0], "host": fields["host"], "llvm": fields.get("LLVM version", "")}


def source_fingerprint() -> str:
    sys.path.insert(0, str(ROOT))
    from chess_variant_sandbox import measurement

    return measurement.rust_source_fingerprint() or "unknown"


def check() -> int:
    if not STAMP.exists() or not PROFILE.exists():
        print("no committed profile")
        return 1
    stamp = json.loads(STAMP.read_text(encoding="utf-8"))
    here = toolchain()
    reasons = []
    if (stamp["rustc"], stamp["host"]) != (here["rustc"], here["host"]):
        reasons.append(f"made by {stamp['rustc']} on {stamp['host']}, this is {here['rustc']} on {here['host']}")
    if stamp.get("source") != source_fingerprint():
        reasons.append(f"trained on source {stamp.get('source')}, rust/src is now {source_fingerprint()}")
    for reason in reasons:
        print("stale:", reason)
    if not reasons:
        print("current: this toolchain and this source")
    return 1 if reasons else 0


def profdata_tool() -> str:
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True).stdout.strip()
    found = glob.glob(os.path.join(sysroot, "lib", "rustlib", "*", "bin", "llvm-profdata*"))
    if not found:
        sys.exit("regenerate.py: no llvm-profdata; run `rustup component add llvm-tools`")
    return found[0]


def sync(sync_args: list[str], rustflags: str) -> None:
    """`uv sync` with these RUSTFLAGS: the same build invocation whether instrumenting or using.

    The two must be built alike. A profile names functions by their symbols, and cargo's
    symbol hashes move with how it is invoked (a `--target` build's differ from a host
    build's), so a profile trained on a `cargo build --target` core matches nothing in the
    core `uv sync` builds (the record's *mismatched* arm).
    """
    env = dict(os.environ, RUSTFLAGS=rustflags)
    subprocess.run(["uv", "sync", *sync_args, "--reinstall-package", "chess-variant-sandbox"],
                   cwd=ROOT, env=env, check=True)


def build(sync_args: list[str]) -> None:
    if not PROFILE.exists():
        sys.exit("regenerate.py: no rust/pgo/core.profdata to build with")
    sync(sync_args, f"-Cprofile-use={PROFILE}")


def main(argv: list[str]) -> int:
    if "--check" in argv:
        return check()
    sync_args = argv[argv.index("--") + 1:] if "--" in argv else ["--extra", "dev"]
    if "--build" in argv[: argv.index("--") if "--" in argv else len(argv)]:
        build(sync_args)
        return 0
    tool = profdata_tool()
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        try:
            sync(sync_args, f"-Cprofile-generate={raw}")
            train_env = dict(os.environ, LLVM_PROFILE_FILE=str(raw / "core-%p-%m.profraw"))
            subprocess.run([sys.executable, str(HERE / "train.py")], cwd=ROOT, env=train_env, check=True)
            files = sorted(map(str, raw.glob("core-*.profraw")))  # not the build scripts' own
            if not files:
                raise RuntimeError("the training run wrote no profile")
            subprocess.run([tool, "merge", "--sparse", "-o", str(PROFILE), *files], check=True)
        except BaseException:
            # Never leave the instrumented core in the venv: build with the profile on disk.
            print("regenerate.py: training failed; rebuilding with the profile already on disk", file=sys.stderr)
            build(sync_args)
            raise
    stamp = {**toolchain(), "source": source_fingerprint(), "trained_by": "rust/pgo/train.py"}
    STAMP.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PROFILE.relative_to(ROOT)} ({PROFILE.stat().st_size:,} bytes) from {len(files)} raw profiles",
          flush=True)
    build(sync_args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
