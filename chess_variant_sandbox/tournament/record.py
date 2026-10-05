"""What a finished game leaves behind.

Records are the boundary between simulation and analysis: metrics are computed from
these files and never from a live engine, so the index can be recomputed under new
weights without re-simulating a tournament. That is only true if a record carries
enough to answer the questions the index asks -- which is why per-ply branching and
material are stored rather than recovered by replay.

The reproduction triple (``ruleset_hash``, ``seed``, ``agents``) is stored on every
record. A surprising result should be investigable, not re-rollable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GameRecord:
    """One completed game."""

    ruleset_hash: str
    ruleset_name: str
    seed: int
    agents: tuple[str, str]
    moves: list[int]
    winner: int | None
    reason: str
    plies: int

    # Per-ply series, indexed by ply. Together these answer most of what the index
    # asks -- how wide the choices were, how sharply the balance moved -- without
    # needing the engine that produced them.
    branching: list[int] = field(default_factory=list)
    material: list[float] = field(default_factory=list)
    # Misnamed, and kept misnamed. A 1 here means the ply reset the no-progress clock,
    # which happens on a capture, on a move by a piece that can never return to a square
    # it left, or on a promotion -- so this is a per-ply irreversibility flag rather than
    # a capture count. Read it through
    # ``irreversible`` below; the JSON key cannot be renamed because ``from_dict`` does
    # ``cls(**payload)`` and every persisted run under ``runs/`` spells it this way.
    captures: list[int] = field(default_factory=list)

    nodes: tuple[int, int] = (0, 0)
    # How deep each seat actually saw: plies completed summed over the game's searches,
    # and how many searches that was. A budget is nodes, and what nodes buy is a
    # function of the branching factor -- a property of the game being measured -- so a
    # ladder whose rungs are compared as though they were the same players is partly
    # reading the board's width. The quantity was always computed and always thrown
    # away; carrying it costs two integers per seat.
    #
    # Zero and zero for an agent that does not deepen. Defaulted because
    # ``from_dict`` does ``cls(**payload)`` and every run recorded before this field
    # existed is still a valid corpus.
    depth_sum: tuple[int, int] = (0, 0)
    searches: tuple[int, int] = (0, 0)
    duration_ms: float = 0.0

    def mean_depth(self, seat: int) -> float | None:
        """Plies per search for one seat, or ``None`` if that seat never searched.

        ``None`` rather than zero: an agent that does not deepen and an agent whose
        budget ran out before it completed one ply are different things, and only the
        second is a statement about the ruleset.
        """
        if not self.searches[seat]:
            return None
        return self.depth_sum[seat] / self.searches[seat]

    @property
    def irreversible(self) -> list[int]:
        """Per ply: whether that ply could never be undone by playing on.

        The name the series should have had. A game in which almost no ply is
        irreversible is one that can go back and forth indefinitely, which is the
        failure a designer names as "repetitive" long before any metric here notices.
        """
        return self.captures

    def score_for(self, seat: int) -> float:
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == seat else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
        for pair in ("nodes", "depth_sum", "searches"):
            data[pair] = list(getattr(self, pair))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameRecord:
        payload = dict(data)
        payload["agents"] = tuple(payload["agents"])
        for pair in ("nodes", "depth_sum", "searches"):
            payload[pair] = tuple(payload.get(pair) or (0, 0))
        return cls(**payload)


def write_records(records: list[GameRecord], path: str | Path) -> Path:
    """Write records as JSON Lines.

    One game per line, so a sweep can be streamed to disk as it runs and a crashed
    run still leaves every completed game readable. A single JSON array would need the
    whole corpus in memory and would be worthless if truncated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), separators=(",", ":")) + "\n")
    return path


def read_records(path: str | Path) -> Iterator[GameRecord]:
    """Stream records back. A corrupt trailing line from a killed run is skipped."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield GameRecord.from_dict(data)
