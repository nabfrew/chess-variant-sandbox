"""The browser core's compiler is older than the box's, and the core must build with both.

Q808: schema 19 used `usize::is_multiple_of` (stable in 1.87). The box (Rust 1.98) and this
machine built it; the browser build (`deploy/pyodide/build.sh`, then a February 2025
nightly, rustc 1.86) failed with E0658, and the site deployed without browser measuring for
a day because a deploy treats that failure as "measuring off".

The guard is in two places. `rust/Cargo.toml`'s `rust-version` is the newest stable whose
whole std the browser's nightly carries, and it denies clippy's `incompatible_msrv`, so
`cargo clippy` refuses an API past it at the edit. This file holds `rust-version` to the pin
in `deploy/versions.env`, with no compiler and no Docker, so moving either one alone fails
the fast suite.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import tomllib

REPOSITORY = Path(__file__).resolve().parent.parent
CARGO_TOML = REPOSITORY / "rust" / "Cargo.toml"
VERSIONS = REPOSITORY / "deploy" / "versions.env"
BUILD_SH = REPOSITORY / "deploy" / "pyodide" / "build.sh"

# Rust 1.1 shipped on 2015-06-25 and every release since is six weeks after the last, so
# 1.k ships on this date plus 42k days (1.88 on 2025-06-26, 1.84 on 2025-01-09).
_RELEASE_ORIGIN = datetime.date(2015, 5, 14)
_CADENCE_DAYS = 42


def _release_date(minor: int) -> datetime.date:
    return _RELEASE_ORIGIN + datetime.timedelta(days=_CADENCE_DAYS * minor)


def carried_stable(toolchain: str) -> tuple[int, int]:
    """The newest stable `(1, minor)` whose every std API a toolchain has.

    A stable pin carries itself. A nightly dated D, with 1.k the last release before D, is
    1.(k+2)-nightly: 1.(k+1) branched to beta from it before 1.k shipped, so it carries all
    of 1.(k+1) and only part of 1.(k+2). Counting from the release before D, rather than on
    D, is the conservative side of the boundary.
    """
    stable = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", toolchain)
    if stable:
        return int(stable.group(1)), int(stable.group(2))
    nightly = re.fullmatch(r"nightly-(\d{4})-(\d{2})-(\d{2})", toolchain)
    assert nightly, f"{toolchain!r} is neither a stable version nor a dated nightly"
    built = datetime.date(*(int(part) for part in nightly.groups()))
    minor = 1
    while _release_date(minor + 1) < built:
        minor += 1
    return 1, minor + 1


def _pins() -> dict[str, str]:
    body = VERSIONS.read_text(encoding="utf-8")
    return dict(re.findall(r"^(\w+)=(\S+)$", body, re.MULTILINE))


def _cargo() -> dict:
    with CARGO_TOML.open("rb") as handle:
        return tomllib.load(handle)


def test_the_release_arithmetic_matches_what_rustc_said() -> None:
    # Read from the compilers themselves: the old pin's `rustc --version` was
    # "1.86.0-nightly (854f22563 2025-01-31)", the new one's "1.90.0-nightly".
    assert _release_date(84) == datetime.date(2025, 1, 9)
    assert _release_date(88) == datetime.date(2025, 6, 26)
    assert carried_stable("nightly-2025-02-01") == (1, 85)
    assert carried_stable("nightly-2025-06-27") == (1, 89)
    assert carried_stable("1.97.1") == (1, 97)


def test_rust_version_is_what_the_browser_compiler_carries() -> None:
    declared = _cargo()["package"].get("rust-version")
    assert declared, "rust/Cargo.toml must state rust-version, the browser compiler's floor"
    major, minor = (int(part) for part in declared.split(".")[:2])
    carried = carried_stable(_pins()["PYODIDE_RUST_TOOLCHAIN"])
    assert (major, minor) <= carried, (
        f"rust-version {declared} is newer than {carried[0]}.{carried[1]}, all that "
        f"PYODIDE_RUST_TOOLCHAIN carries: the browser build would fail"
    )


def test_clippy_refuses_an_api_past_rust_version() -> None:
    lints = _cargo().get("lints", {}).get("clippy", {})
    assert lints.get("incompatible_msrv") == "deny"


def test_the_image_recipe_covers_every_pin_the_image_is_built_from() -> None:
    # The tag names Pyodide's two versions only, so a moved toolchain or sysroot pin has
    # to move the recipe label, or build.sh reuses the old image and the old compiler.
    script = BUILD_SH.read_text(encoding="utf-8")
    recipe = script[script.index('RECIPE="$(') : script.index("current() {")]
    passed = set(re.findall(r'--build-arg "(\w+)=', script))
    digested = set(re.findall(r'"(\w+)=\$\1"', recipe))
    assert passed and passed == digested
