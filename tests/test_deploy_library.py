"""A deploy refuses a core the library was not measured on (R-T1b).

`deploy/library_current.py` reads every tracked library game's stamp the way
`measure --check` does, for the core the ref builds; `deploy/deploy.sh` asks it before the
box is taken and stops on anything but current, unless `--allow-provisional`.

The first half runs the check on a fixture library: a git repository of its own whose one
archived game is the five-square `lancers` (never chess), stamped in its `runs/` with
today's basis or with a field moved. The second half lifts deploy.sh's block out and runs
it against a stand-in `uv` and box, so that nothing here contacts the box.

`deploy/` is not exported, so the public copy collects this module and skips it.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import private_paths
import pytest
from test_corpus import SPEC, world  # noqa: F401  (`world` is a fixture)
from test_history import stubbed  # noqa: F401  (`stubbed` is a fixture)
from test_refresher import library_game

from chess_variant_sandbox import corpus

# `deploy/` stays out of the public export: the module skips there (tests/private_paths.py).
pytestmark = private_paths.needs("deploy/library_current.py", "deploy/deploy.sh")
DEPLOY = private_paths.path("deploy")
# Registered under its name, because a dataclass reads its module back from sys.modules.
current = private_paths.script("deploy/library_current.py", "deploy_library_current")
try:
    BASH = importlib.import_module("test_deploy").BASH
except ImportError:  # test_deploy stays out of the public export too
    BASH = None

needs_bash = pytest.mark.skipif(BASH is None, reason="no working bash here")

OTHER_CORE = "a0b1c2d3e4f5a6b7-0011223344556677"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(world, stubbed, monkeypatch: pytest.MonkeyPatch) -> Path:  # noqa: F811
    """``world``'s directory as a repository at one commit: `rust/src/` (so the ref has a
    core to fingerprint) and nothing else yet. This process stands in as that checkout on
    that core."""
    root = Path(world["archive"]).parent
    (root / "rust" / "src").mkdir(parents=True)
    (root / "rust" / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8")
    _git(root, "init", "-q")
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "t"),
                       ("core.autocrlf", "false")):
        _git(root, "config", key, value)
    monkeypatch.setattr(current, "package_root", lambda: root.resolve())
    monkeypatch.setattr(current, "loaded_core", lambda: current.ref_builds(root, "HEAD"))
    return root


def _commit(root: Path) -> None:
    _git(root, "add", "rust", "archive")
    _git(root, "commit", "-q", "-m", "library")


def _check(root: Path, allow: bool = False) -> tuple[int, str]:
    code, lines = current.check(root, "HEAD", root / "runs", allow)
    return code, "\n".join(lines)


def _stored(root: Path, **moved: Any) -> str:
    digest = library_game(root, SPEC, **moved)
    _commit(root)
    return digest


class TestTheCheck:
    def test_a_library_current_on_the_refs_core_passes(self, repo: Path) -> None:
        _stored(repo)
        code, said = _check(repo)
        assert code == 0, said
        assert "all 1 tracked games current" in said
        assert current.ref_builds(repo, "HEAD") in said

    def test_a_score_measured_on_another_core_refuses_and_names_the_sample(
            self, repo: Path) -> None:
        digest = _stored(repo, engine=OTHER_CORE)
        code, said = _check(repo)
        assert code == 1, said
        builds = current.ref_builds(repo, "HEAD")
        assert "1 of 1 tracked library games were not measured on the core" in said
        assert f"({builds})" in said
        assert f"[{digest}] measured on 0011223344556677; by sample" in said
        assert current.REPAIR in said
        assert said.splitlines()[-1].startswith("library: refused: 1 of 1")

    def test_a_score_played_by_another_roster_names_the_replay(self, repo: Path) -> None:
        _stored(repo, agents=["random", "greedy"])
        code, said = _check(repo)
        assert code == 1, said
        assert "; by replay; " in said
        assert "replayed in full" in said

    def test_allow_provisional_goes_on_and_says_so(self, repo: Path) -> None:
        _stored(repo, engine=OTHER_CORE)
        code, said = _check(repo, allow=True)
        assert code == 0, said
        last = said.splitlines()[-1]
        assert last.startswith("library: --allow-provisional: deploying anyway; 1 of 1")
        assert "provisional on the site" in last

    def test_an_untracked_game_does_not_count(self, repo: Path) -> None:
        _stored(repo)
        # Another session's file in archive/, never committed and never measured.
        stray = {**SPEC, "name": "lancers stray", "rules": {**SPEC["rules"], "max_plies": 90}}
        (repo / "archive" / "stray.json").write_text(json.dumps(stray), encoding="utf-8")
        assert len(corpus.corpus_hashes(repo / "rulesets", repo / "archive")) == 2
        code, said = _check(repo)
        assert code == 0, said
        assert "all 1 tracked games current" in said

    def test_a_checkout_that_is_not_the_ref_cannot_answer(self, repo: Path) -> None:
        _stored(repo)
        game = next((repo / "archive").glob("*.json"))
        game.write_text(game.read_text(encoding="utf-8") + "\n ", encoding="utf-8")
        code, said = _check(repo)
        assert code == 2, said
        assert "differ from" in said and game.name in said
        assert said.splitlines()[-1].startswith("library: refused: unchecked")

    def test_a_core_that_is_not_the_refs_cannot_answer(
            self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stored(repo)
        monkeypatch.setattr(current, "loaded_core", lambda: "ffffffffffffffff")
        code, said = _check(repo)
        assert code == 2, said
        assert "loaded core is ffffffffffffffff" in said
        assert current.ref_builds(repo, "HEAD") in said
        assert current.REBUILD in said
        code, said = _check(repo, allow=True)
        assert code == 0 and "--allow-provisional" in said.splitlines()[-1]


# ------------------------------------------------------------------ deploy.sh's block

RULE = "# ---------------------------------------------------------------- "
BLOCK = (RULE + "the library's scores\n", RULE + "end of the library's scores\n")
LAST = ("printf '\\n== deployed", 'if [ "$BROWSER" = off ]; then')


def _lift(local: str, start: str, end: str) -> str:
    first = local.index(start)
    return local[first : local.index(end, first)]


def test_the_check_runs_before_the_build_and_the_lock() -> None:
    local = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    block = local.index(BLOCK[0])
    assert local.index('COMMIT="$(git') < block < local.index(RULE + "the browser artefact\n")
    assert block < local.index('feed_box lock "')
    assert "--allow-provisional) ALLOW_PROVISIONAL=yes" in local


def _run(tmp_path: Path, *, verdict: str, rc: int, allow: bool = False) -> tuple[Any, list[str]]:
    """The block, then the lock, then the last line, against a `uv` that prints the
    check's ``verdict`` (or the allowed one, when it is passed the flag) and exits ``rc``."""
    assert BASH
    out, bin_dir = tmp_path / "out", tmp_path / "bin"
    out.mkdir()
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$OUT/uv"\n'
        'echo "  1 of 1 tracked library games were not measured on the core c builds (b):"\n'
        'case "$*" in *--allow-provisional*)\n'
        '    echo "  library: --allow-provisional: deploying anyway; 1 of 1 read provisional"\n'
        "    exit 0 ;; esac\n"
        f'echo "  {verdict}"\nexit {rc}\n',
        encoding="utf-8", newline="\n")
    (bin_dir / "uv").chmod(0o755)
    local = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    script = (
        "set -euo pipefail\n"
        'stub="$1"; if command -v cygpath >/dev/null 2>&1; then stub="$(cygpath -u "$stub")"; fi\n'
        'export PATH="$stub:$PATH"\n'
        'say() { printf "== %s\\n" "$*"; }\nnote() { printf "  %s\\n" "$*"; }\ntook() { :; }\n'
        'feed_box() { printf "%s\\n" "$1" >> "$OUT/modes"; }\n'
        f"ALLOW_PROVISIONAL={'yes' if allow else 'no'}\nstarted=$SECONDS\nBROWSER=current\n"
        f"{_lift(local, *BLOCK)}"
        'feed_box lock "someone" "${COMMIT:0:12}"\n'
        f"{_lift(local, *LAST)}"
    )
    result = subprocess.run(
        [BASH, "-c", script, "local", str(bin_dir)], capture_output=True, text=True,
        timeout=120,
        env={**os.environ, "HERE": DEPLOY.as_posix(), "REPO": tmp_path.as_posix(),
             "COMMIT": "c" * 40, "OUT": out.as_posix()},
    )
    modes = out / "modes"
    return result, modes.read_text(encoding="utf-8").split() if modes.exists() else []


@needs_bash
def test_a_current_library_takes_the_box(tmp_path: Path) -> None:
    result, modes = _run(tmp_path, verdict="library: all 1 tracked games current", rc=0)
    assert result.returncode == 0, result.stderr
    assert modes == ["lock"]
    assert "provisional" not in result.stdout
    uv = (tmp_path / "out" / "uv").read_text(encoding="utf-8")
    assert "library_current.py" in uv and "c" * 40 in uv


@needs_bash
@pytest.mark.parametrize("rc", [1, 2])
def test_a_library_not_current_refuses_before_the_lock(tmp_path: Path, rc: int) -> None:
    result, modes = _run(tmp_path, verdict="library: refused: 1 of 1", rc=rc)
    assert result.returncode == 1
    assert modes == []
    assert "refused before the box was taken" in result.stdout
    assert "measured on the core" in result.stdout
    assert "== deployed" not in result.stdout


@needs_bash
def test_allow_provisional_takes_the_box_and_says_so_twice(tmp_path: Path) -> None:
    result, modes = _run(tmp_path, verdict="library: refused: 1 of 1", rc=1, allow=True)
    assert result.returncode == 0, result.stderr
    assert modes == ["lock"]
    assert "!! --allow-provisional: deploying cccccccccccc" in result.stdout
    tail = result.stdout[result.stdout.index("== deployed"):]
    assert "library: --allow-provisional: deploying anyway; 1 of 1" in tail
