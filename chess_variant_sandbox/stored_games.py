"""The stored computer games of a game's measurement runs, read (`docs/plan-website.md`,
*The study page* and *The guided first game*).

One reader of `runs/<hash>-<seed>/games.jsonl`, shared by the study record (`study.py`,
R-W4b1) and the lesson record (R-W2b1, `lessons.py`), so the two never disagree about
which games a measurement holds or in what order. It yields each record with its run seed
and its place in the file (0-based line), and digests each file as `puzzles.py` already
does (`_file_digest`), so a record built from these files can say which files it read.

It reads; it plays, replays and judges nothing. The engine answers every question about a
move, by the caller's own call.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.puzzles import _file_digest, runs_missing

GAMES_FILE = "games.jsonl"


@dataclass(frozen=True)
class Stored:
    """One stored game: its run's seed, its place in that run's file, and the record."""

    seed: int
    place: int
    record: dict[str, Any]

    @property
    def moves(self) -> list[int]:
        return list(self.record.get("moves") or [])

    @property
    def agents(self) -> tuple[str, str]:
        found = self.record.get("agents") or ["", ""]
        return str(found[0]), str(found[1])

    @property
    def winner(self) -> int | None:
        found = self.record.get("winner")
        return found if found in (0, 1) else None

    @property
    def order(self) -> tuple[int, int]:
        """The tie-break every rule shares: the lower run seed, then the earlier game."""
        return self.seed, self.place


def games_file(runs_dir: Path | str, digest: str, seed: int) -> Path:
    return Path(runs_dir) / f"{digest}-{seed}" / GAMES_FILE


def missing(runs_dir: Path | str, digest: str, seeds: Iterable[int]) -> list[int]:
    """The seeds whose run has no games on disk (`puzzles.runs_missing`)."""
    return runs_missing(runs_dir, digest, seeds)


def digests(runs_dir: Path | str, digest: str, seeds: Iterable[int]) -> dict[str, str]:
    """Each present run's games file, digested, keyed by its seed as text (JSON's keys)."""
    found: dict[str, str] = {}
    for seed in seeds:
        path = games_file(runs_dir, digest, seed)
        if path.is_file():
            found[str(seed)] = _file_digest(path)
    return found


def played_on(runs_dir: Path | str, digest: str, seeds: Iterable[int]) -> str | None:
    """The day the latest of the runs' games files was written, ISO, or None: when the
    measurement's games were played, as far as the files can say (a stamp names no day)."""
    stamps = [games_file(runs_dir, digest, seed).stat().st_mtime for seed in seeds
              if games_file(runs_dir, digest, seed).is_file()]
    if not stamps:
        return None
    return datetime.date.fromtimestamp(max(stamps)).isoformat()


def read(runs_dir: Path | str, digest: str, seeds: Iterable[int]) -> Iterator[Stored]:
    """Every stored game of the runs at ``seeds`` that are on disk, in seed order as given,
    then file order. A line that is not a record (a torn last line) is passed over and
    keeps its place, so a place always names the file's own line."""
    for seed in seeds:
        path = games_file(runs_dir, digest, seed)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for place, line in enumerate(handle):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("moves"), list):
                    yield Stored(seed, place, record)
