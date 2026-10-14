"""Every deploy refreshes the public repository (Q907).

`deploy/public.sh COMMIT REF` is the step `deploy/deploy.sh` runs after its proofs: the
commit's own exporter in a scratch worktree of it, the scans, and one fast-forward commit
pushed to the public repository when a file changed. Every way it can fail prints one line
and exits 0, so it never fails a deploy.

Each test here builds a small private repository holding the real `public.sh` and the real
`export_public.py`, and a bare repository in a temp dir standing in for the public one.
Nothing here reaches the network or the real public repository.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

test_deploy = pytest.importorskip("test_deploy")  # absent from the public export

BASH = test_deploy.BASH
DEPLOY = test_deploy.DEPLOY
GIT = test_deploy.GIT
PUBLIC = DEPLOY / "public.sh"
EXPORTER = DEPLOY.parent / ".claude" / "scripts" / "export_public.py"

pytestmark = pytest.mark.skipif(
    BASH is None or GIT is None or not PUBLIC.is_file() or not EXPORTER.is_file(),
    reason="needs bash, git, deploy/public.sh and the exporter",
)

# The private repository's author: the exporter forbids the name in any exported file.
PRIVATE = {"GIT_AUTHOR_NAME": "Pat Private", "GIT_AUTHOR_EMAIL": "pat@private.example",
           "GIT_COMMITTER_NAME": "Pat Private", "GIT_COMMITTER_EMAIL": "pat@private.example"}
# The public repository's author, whom every export commit must be made as.
FACE = {"GIT_AUTHOR_NAME": "Public Face", "GIT_AUTHOR_EMAIL": "face@users.noreply.example",
        "GIT_COMMITTER_NAME": "Public Face", "GIT_COMMITTER_EMAIL": "face@users.noreply.example"}


def _git(repo: Path, *args: str, who: dict[str, str] | None = None) -> str:
    assert GIT
    env = {**os.environ, **(who or PRIVATE)}
    out = subprocess.run([GIT, "-C", str(repo), "-c", "core.autocrlf=false",
                          "-c", "commit.gpgsign=false", *args],
                         check=True, capture_output=True, text=True, timeout=60, env=env)
    return out.stdout.strip()


def _write(root: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))


class World:
    """A private repository, a bare public one, and the step's state directory."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.private = tmp / "private"
        self.public = tmp / "public.git"
        self.state = tmp / "state"
        self.private.mkdir()
        _git(self.private, "init", "-q", "-b", "main")
        (self.private / "deploy").mkdir()
        shutil.copy(PUBLIC, self.private / "deploy" / "public.sh")
        (self.private / ".claude" / "scripts").mkdir(parents=True)
        shutil.copy(EXPORTER, self.private / ".claude" / "scripts" / "export_public.py")
        self.commit({"README.md": "# A game\n\nThe first rules.\n", "LICENSE": "a licence\n",
                     "chess_variant_sandbox/__init__.py": "VERSION = 1\n"})

        seed = tmp / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main")
        _write(seed, {"README.md": "# An older export\n"})
        _git(seed, "add", "-A")
        _git(seed, "commit", "-q", "-m", "Export of 2026-10-11", who=FACE)
        _git(tmp, "clone", "-q", "--bare", str(seed), str(self.public))
        self.first_tip = self.tip()

    def commit(self, files: dict[str, str], message: str = "a change") -> str:
        _write(self.private, files)
        _git(self.private, "add", "-A")
        _git(self.private, "commit", "-q", "-m", message)
        return _git(self.private, "rev-parse", "HEAD")

    def tip(self, bare: Path | None = None) -> str:
        return _git(bare or self.public, "rev-parse", "refs/heads/main")

    def show(self, path: str, bare: Path | None = None) -> str:
        return _git(bare or self.public, "show", f"main:{path}")

    def run(self, commit: str | None = None, ref: str = "main",
            **env: str) -> subprocess.CompletedProcess[str]:
        assert BASH
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("CHESS_VARIANT_SANDBOX_PUBLIC")}
        environment.update({
            "CHESS_VARIANT_SANDBOX_PUBLIC_REMOTE": self.public.as_posix(),
            "CHESS_VARIANT_SANDBOX_PUBLIC_DIR": self.state.as_posix(),
            "CHESS_VARIANT_SANDBOX_PYTHON": Path(sys.executable).as_posix(),
            "CHESS_VARIANT_SANDBOX_ENV_FILE": (self.tmp / "no-server-details.env").as_posix(),
            **env,
        })
        sha = commit or _git(self.private, "rev-parse", "HEAD")
        return subprocess.run([BASH, str(self.private / "deploy" / "public.sh"), sha, ref],
                              capture_output=True, text=True, timeout=180, env=environment)


def _summary(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.stderr
    assert len(lines) == 1, result.stdout  # one line, whatever happened
    return lines[0]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_a_changed_export_is_committed_once_and_pushed_as_a_fast_forward(world: World) -> None:
    sha = _git(world.private, "rev-parse", "HEAD")
    line = _summary(world.run())
    tip = world.tip()
    assert re.fullmatch(rf"public repository: pushed {tip[:12]} "
                        r"\(Export of \d{4}-\d\d-\d\d, 3 files changed\)", line), line
    assert _git(world.public, "rev-parse", "main~1") == world.first_tip  # a fast-forward
    assert _git(world.public, "log", "-1", "--format=%an <%ae>|%cn <%ce>", "main") == \
        "Public Face <face@users.noreply.example>|Public Face <face@users.noreply.example>"
    subject, body = _git(world.public, "log", "-1", "--format=%s%n%b", "main").split("\n", 1)
    assert subject.startswith("Export of 20")
    assert body.strip() == f"Source: main at {sha[:12]}."
    tracked = set(_git(world.public, "ls-tree", "-r", "--name-only", "main").splitlines())
    assert tracked == {"README.md", "LICENSE", "chess_variant_sandbox/__init__.py",
                       "EXPORT-MANIFEST.json"}
    assert world.show("README.md") == "# A game\n\nThe first rules."
    assert sha in world.show("EXPORT-MANIFEST.json")
    # The scratch is gone; the clone stays for the next deploy.
    assert (world.state / "clone" / ".git").is_dir()
    assert not (world.state / "source").exists() and not (world.state / "export").exists()
    assert "state" not in _git(world.private, "worktree", "list")


def test_an_equal_export_pushes_nothing_even_when_its_manifest_moved(world: World) -> None:
    _summary(world.run())
    pushed = world.tip()
    # A commit whose change stays private moves only the manifest's source commit.
    world.commit({"docs/notes.md": "private notes\n"})
    assert _summary(world.run()) == f"public repository: level with {pushed[:12]}, nothing to push"
    assert world.tip() == pushed
    assert _git(world.state / "clone", "status", "--porcelain") == ""


def test_a_scan_refusal_pushes_nothing_and_exits_zero(world: World) -> None:
    world.commit({"README.md": "# A game\n\nkey = " + "sk-ant-" + "a1b2" * 6 + "\n"})
    line = _summary(world.run())
    assert line.startswith("public repository: skipped: the export's scans refused ")
    assert "1 finding(s)" in line
    assert world.tip() == world.first_tip
    assert not (world.state / "export").exists()


def test_a_push_the_moved_public_repository_refuses_exits_zero_and_forces_nothing(
        world: World) -> None:
    # The clone a previous deploy left, and somebody else's push to the public repository
    # between this step's fetch and its push: the push goes to a copy that has moved on,
    # which the fetch never saw.
    _git(world.tmp, "clone", "-q", str(world.public), str(world.state / "clone"))
    moved = world.tmp / "moved.git"
    _git(world.tmp, "clone", "-q", "--bare", str(world.public), str(moved))
    theirs = _git(moved, "commit-tree", "main^{tree}", "-p", "main", "-m", "moved", who=FACE)
    _git(moved, "update-ref", "refs/heads/main", theirs)
    _git(world.state / "clone", "config", "remote.origin.pushurl", moved.as_posix())

    world.commit({"README.md": "# A game\n\nThe second rules.\n"})
    line = _summary(world.run())
    assert line.startswith("public repository: skipped: the push was refused: ")
    assert "moved since the fetch; nothing was forced" in line
    assert world.tip(moved) == theirs  # not overwritten
    assert world.tip() == world.first_tip
    assert _git(world.state / "clone", "status", "--porcelain") == ""


def test_an_unreachable_public_repository_is_a_line_not_a_failure(world: World) -> None:
    line = _summary(world.run(CHESS_VARIANT_SANDBOX_PUBLIC_REMOTE=(
        world.tmp / "nowhere.git").as_posix()))
    assert line.startswith("public repository: skipped: could not clone ")
    assert world.tip() == world.first_tip


def test_the_export_is_the_refs_and_never_the_working_trees(world: World) -> None:
    committed = world.commit({"README.md": "# A game\n\nThe committed rules.\n"})
    # Uncommitted, untracked, and a working-tree exporter that would fail if it were run.
    _write(world.private, {
        "README.md": "# A game\n\nAn edit nobody committed.\n",
        "chess_variant_sandbox/untracked.py": "LEAK = True\n",
        ".claude/scripts/export_public.py": "raise SystemExit(3)\n",
    })
    line = _summary(world.run(committed))
    assert line.startswith("public repository: pushed ")
    assert world.show("README.md") == "# A game\n\nThe committed rules."
    tracked = _git(world.public, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert "chess_variant_sandbox/untracked.py" not in tracked


def test_switched_off_it_touches_nothing(world: World) -> None:
    line = _summary(world.run(CHESS_VARIANT_SANDBOX_PUBLIC="off"))
    assert line == "public repository: skipped: switched off (CHESS_VARIANT_SANDBOX_PUBLIC=off)"
    assert world.tip() == world.first_tip
    assert not world.state.exists()


def test_deploy_runs_it_after_the_proofs_and_never_fails_on_it() -> None:
    body = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    call = body.index('bash "$HERE/public.sh" "$COMMIT" "$REF"')
    # After the restart's failure exit and the proofs' failure exit, before the last line.
    assert body.index("the app did not come back, so this deploy failed") < call
    assert body.index('if [ "$FAILED" = yes ]; then') < call
    assert call < body.index("== deployed %s in %ss")
    assert body[call:].split("\n", 2)[1].strip().startswith("|| note ")
    assert "--no-public)   PUBLIC=no ;;" in body
    assert 'if [ "$PUBLIC" = yes ]; then' in body[:call]
