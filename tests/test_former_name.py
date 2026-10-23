"""The project's former name stays gone, except where a record has to keep it.

The project was renamed *Chess variant sandbox* (the user, 2026-09-23: "fully purge the old
name"); `docs/archive/plan-rename.md` is the record. This runs the former name's pattern over
every tracked file and fails on any occurrence that `ALLOWED` does not name.

The pattern is the token in lower or upper case, not joined to a letter on either side, so
English *new game* (a space), the UCI command (a letter before it) and the repository's
folder name (capitals) never match. Nothing in this file spells the name but the pattern
itself, so an entry's fragment is the text beside the name, never the name.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parent.parent
FORMER = re.compile(r"(?<![A-Za-z])(newgame|NEWGAME)(?![a-z])")
# The pattern's two spellings, as bytes: a file holding neither cannot match it.
SPELLINGS = tuple(word.encode("ascii") for word in FORMER.pattern.split("(")[2].split(")")[0].split("|"))


class Allowed(NamedTuple):
    path: str
    # A line holding the occurrence must contain this text; None allows the whole file.
    fragment: str | None
    reason: str


ALLOWED: tuple[Allowed, ...] = (
    *(
        Allowed(
            f"chess_variant_sandbox/site/migrations/{name}",
            None,
            "an applied migration is checksummed; an edit makes every deploy's migrate refuse",
        )
        for name in ("0001_store.sql", "0004_invite_email.sql", "0005_leaving.sql")
    ),
    Allowed("docs/archive/plan-rename.md", None, "the record of the rename"),
    Allowed(
        "docs/archive/queue-2026-09.md",
        None,
        "a closed month's queue: the rename's tasks quoting the user, a branch and a docker "
        "image that still exist under the former name, and the session logs that name them",
    ),
    Allowed(
        "docs/archive/queue-2026-10.md",
        "adds `tests/test_former_name.py`, drops",
        "the queue entry for the sweep itself (Q383), naming what it drops from the lease hook",
    ),
    Allowed(
        "docs/archive/decisions.md",
        "consistently to 'Chess variant sandbox' not",
        "the rename's own decision, quoting the user",
    ),
    Allowed(
        "docs/changelog.md",
        "command no longer exists",
        "the rename's announcement for the command (2026-09-24); the user allowed it to name the old name",
    ),
    Allowed(
        "docs/changelog.md",
        "where they used to say",
        "the rename's announcement for the site (2026-09-23); the user allowed it to name the old name",
    ),
    Allowed(
        "docs/reference/test-audit/redundancy-candidates.json",
        None,
        "a dated audit's output: test ids as they stood when it ran, which its counts in "
        "docs/archive/plan-tests.md section 4 add up from",
    ),
    Allowed("tests/test_former_name.py", "FORMER = re.compile(", "it has to spell what it looks for"),
)


def occurrences() -> list[tuple[str, int, str]]:
    """(path, line number, line) for every occurrence of the former name in a tracked file."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout.decode("utf-8")
    found = []
    for path in sorted(p for p in listed.split("\0") if p):
        # Only a file holding one of the pattern's two spellings can match, so the bytes
        # are asked that first, and only those files are decoded and read line by line:
        # decoding and splitting all 96 MB of the tree cost 2.5 s (Q1009). A file that does
        # not decode is skipped either way.
        try:
            data = (REPO / path).read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            continue
        if not any(spelling in data for spelling in SPELLINGS):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            found.extend((path, number, line) for _ in FORMER.finditer(line))
    return found


def allowed_by(path: str, line: str) -> Allowed | None:
    for entry in ALLOWED:
        if entry.path == path and (entry.fragment is None or entry.fragment in line):
            return entry
    return None


needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO / ".git").exists(), reason="not a git checkout"
)


@needs_git
def test_the_former_name_appears_only_where_allowed() -> None:
    stray = [
        f"{path}:{number}: {line.strip()[:120]}"
        for path, number, line in occurrences()
        if allowed_by(path, line) is None
    ]
    assert stray == [], "the former name is back; rename it, or add an ALLOWED entry with its reason"


@needs_git
def test_every_allowance_is_still_needed() -> None:
    """An entry nothing matches is stale, and would let the name back in unnoticed."""
    used = Counter(allowed_by(path, line) for path, _, line in occurrences())
    assert [entry for entry in ALLOWED if not used[entry]] == []


def test_every_allowance_gives_its_reason_and_never_spells_the_name() -> None:
    for entry in ALLOWED:
        assert entry.reason.strip(), entry
        assert not FORMER.search(entry.reason + (entry.fragment or "")), entry
