"""The box's disk and the site's queues, for `/admin`'s *Status* (Q813).

`docs/plan-website.md`, *Operations still open*, the *Monitoring* box: disk and queue depth
on a status page. The ping and the disk had landed as `deploy/status.sh` (Q721); this is
the rest, and the same numbers on both sides.

**The queues are one SQL statement**, `depths.sql`, beside this file. It names each queue,
counts what waits in it, ages the oldest, holds the threshold and the reason for it, and
judges. This module runs it unchanged; `deploy/status.sh` feeds the same file to the box's
`psql`. Nothing here restates a threshold, so the two cannot drift: the page and the
script read one answer. The file's header says which queues, and why not the others.

**The disk** is the filesystem `runs/` lives on, which on the box is `/`, the one
`deploy/remote/status.sh` asks `df` about. Its numbers are worked out as `df` works them
out -- *used* is what is allocated, *free* is what an ordinary user may still write, and
the percentage is used over the two together, rounded up -- so the page and the script
print the same line. Past `DISK_LIMIT` it reads as a problem (`deploy/README.md`, *The
disk fills*).

Nothing here writes, and nothing names a person.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SQL_FILE = Path(__file__).with_name("depths.sql")
"""The statement both readers run. `deploy/status.sh` sends this very file to the box."""

DISK_LIMIT = 90
"""Percent of the disk used at which the disk reads as a problem.

The box has 38 GB; past 90% is under 4 GB left, about what a night's dump, a core build
(`/opt/build`) and a few measurements need together, so the next fill is an outage. The
same number is `deploy/status.sh`'s, which a test holds equal."""

GIGABYTE = 2**30


@dataclass(frozen=True)
class Disk:
    """One filesystem, as `df -P` reads it."""

    used: int
    free: int

    @property
    def used_percent(self) -> int:
        whole = self.used + self.free
        return 0 if whole <= 0 else math.ceil(self.used * 100 / whole)

    @property
    def problem(self) -> bool:
        return self.used_percent >= DISK_LIMIT

    def said(self) -> str:
        return f"{self.used_percent}% used, {self.free / GIGABYTE:.1f} GB free"

    def as_dict(self) -> dict[str, Any]:
        return {
            "usedPercent": self.used_percent,
            "freeGb": round(self.free / GIGABYTE, 1),
            "said": self.said(),
            "limit": DISK_LIMIT,
            "problem": self.problem,
        }


def disk(path: Path) -> Disk | None:
    """The filesystem under `path`, or `None` where it cannot be read.

    A `runs/` not made yet is read where it will be made: its nearest existing parent.
    """
    try:
        here = path.resolve()
        while not here.exists() and here != here.parent:
            here = here.parent
        found = shutil.disk_usage(here)
    except OSError:
        return None
    return Disk(used=int(found.used), free=int(found.free))


@dataclass(frozen=True)
class Queue:
    """One row of `depths.sql`, in its column order."""

    queue: str
    problem: bool
    said: str
    label: str
    depth: int
    age: int | None
    oldest_said: str | None
    limit_seconds: int | None
    limit_said: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue": self.queue,
            "label": self.label,
            "depth": self.depth,
            "oldest": self.oldest_said,
            "limit": self.limit_said,
            "problem": self.problem,
            "said": self.said,
            "why": self.why,
        }


def statement() -> str:
    return SQL_FILE.read_text(encoding="utf-8")


def queues(connection: Any) -> list[Queue]:
    """Every queue, in the file's order, read off the store now."""
    with connection.cursor() as cursor:
        cursor.execute(statement())
        rows = cursor.fetchall()
    return [
        Queue(
            queue=str(row[0]),
            problem=bool(row[1]),
            said=str(row[2]),
            label=str(row[3]),
            depth=int(row[4]),
            age=None if row[5] is None else int(row[5]),
            oldest_said=None if row[6] is None else str(row[6]),
            limit_seconds=None if row[7] is None else int(row[7]),
            limit_said=str(row[8]),
            why=str(row[9]),
        )
        for row in rows
    ]


def view(connection: Any, path: Path) -> dict[str, Any]:
    """The `status` key on `GET /api/admin`: the disk, every queue, and one sentence."""
    found_disk = disk(path)
    found = queues(connection)
    problems = [one.label for one in found if one.problem]
    if found_disk is not None and found_disk.problem:
        problems.insert(0, "the disk")
    return {
        "disk": None if found_disk is None else found_disk.as_dict(),
        "queues": [one.as_dict() for one in found],
        "said": ("A problem: " + ", ".join(problems) + ".") if problems
        else "Nothing past its limit.",
    }
