"""The design history: what was tried, what came from what, and how it did.

Rejected candidates are archived alongside accepted ones. That is not tidiness -- a
proposal that failed its gate is the most useful thing the next prompt can carry,
because it names a specific way this design space goes wrong. Throwing failures away
means rediscovering them.

Entries are append-only JSON Lines keyed by the spec's mechanic hash, so a run can be
stopped and resumed, and two runs can be concatenated. The hash ignores name and
description, which means a proposal that renames an already-tested game collides with
it and is skipped before a tournament is spent on it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.loop import grid as grid_module
from chess_variant_sandbox.metrics.bootstrap import clears

LINEAGE_FILE = "lineage.jsonl"
SPEC_DIR = "specs"

# Statuses that mean "this candidate is not a going concern". An evaluated entry counts
# too when it failed a gate: it was played and disqualified, which is a failure with far
# more evidence behind it than one caught by the schema.
_FAILED = ("invalid", "rejected")


@dataclass
class Entry:
    """One evaluated candidate."""

    hash: str
    name: str
    generation: int
    parent: str | None
    status: str  # proposed | invalid | duplicate | rejected | evaluated
    index: float | None = None
    # Sampling standard deviation of ``index``. Stored on the entry rather than derived
    # later because a leaderboard that ranks scores without their intervals implies a
    # confidence the evidence does not carry -- and in the reference corpus the interval
    # is wider than the gaps between most adjacent entries.
    index_sd: float = 0.0
    # The interval itself, not just its width. Separation is a comparison of ends, and
    # reconstructing an end from ``index +/- 1.96 sd`` would assume a symmetry the
    # bootstrap distribution does not have -- the geometric mean's Jensen gap puts its
    # centre about 0.03 below the point estimate on the reference corpus.
    index_ci_low: float | None = None
    index_ci_high: float | None = None
    # The seeds ``index`` was played at, one run each. A design run plays one tournament
    # a candidate at its one seed, so the loop writes ``[config.seed]``. Recorded rather
    # than assumed, because a number says how many runs it rests on or is not shown
    # (`CLAUDE.md`): an entry written before this field loads with none, and every
    # reader of it shows *no score* rather than a count it would have to guess.
    seeds: list[int] = field(default_factory=list)
    # Whether this candidate beat the parent it was revised from by more than the noise
    # in both. ``None`` where there was no parent to beat. A run whose entries are all
    # ``False`` found nothing, however much its leaderboard moved.
    separated: bool | None = None
    # Which cell of the behaviour grid this game belongs to -- board size, army size,
    # whether captures come back, how it is won, how many special rules it carries,
    # whether what a piece may take depends on what it is, and whether the board has
    # edges at all.
    # Derived from the spec alone, so it is known before the ruleset is played and can
    # therefore be *aimed at*. See ``loop/grid.py``.
    cell: str = ""
    objective: float | None = None
    # An absolute 0-1 judgement, which the loop no longer asks for: the model is asked
    # which of two games is better instead, and the resulting ordering breaks ties rather
    # than entering the score. Kept on the record because an archive written before that
    # change still carries it, and ``Entry(**json)`` would drop those entries whole if the
    # field disappeared.
    judge: float | None = None
    viable: bool = False
    weights_version: str = ""
    notes: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def runs(self) -> int:
        """How many runs ``index`` rests on, or 0 where the entry does not say."""
        return len(self.seeds) if isinstance(self.seeds, list) else 0

    def scored_as(self) -> str:
        """``index``, its spread and the runs it rests on, or *no score*.

        In the page's words (`web/static/page/core.js`'s `runsPhrase`): *as the one run*, *as
        the mean of four runs*. An entry that records no runs shows no number at all,
        since a number that cannot say which claim it is makes none.
        """
        if self.index is None or not self.runs:
            return "no score"
        spread = f" +/-{self.index_sd:.3f}" if self.index_sd else ""
        return f"index {self.index:.3f}{spread} as the {_runs_phrase(self.runs)}"

    def beats(self, other: Entry) -> bool:
        """Whether this entry scores above another by more than the noise in both.

        One rule, shared with ``IndexResult.separated_from`` and
        ``BootstrapResult.separates`` through ``metrics.bootstrap.clears``. Where either
        entry has no interval the comparison falls back to the point estimates, which is
        what an archive written before intervals existed can support.
        """
        mine, theirs = self.index or 0.0, other.index or 0.0
        if self.index_ci_low is None or other.index_ci_high is None:
            return mine > theirs
        return mine > theirs and clears(self.index_ci_low, other.index_ci_high)


_RUN_WORDS: tuple[str, ...] = (
    "no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
)
"""The start of the page's `RUN_WORDS`, so the command line spells a count as a card does."""


def provisional_sentence(provisional: int, scored: int) -> str:
    """The leaderboard's header: *4 of 73 scores are provisional while they are measured
    again* (`docs/plan-evolving-score.md` §5.2). One wording for the site's listing and
    the `leaderboard` command; a caller with none provisional says nothing at all."""
    if provisional == 1:
        return f"1 of {scored} scores is provisional while it is measured again."
    return f"{provisional} of {scored} scores are provisional while they are measured again."


def _scored_under(entry: Entry) -> str:
    """Which revision a provisional row came from, or that it does not say."""
    if entry.weights_version:
        return f"scored under {entry.weights_version}"
    return "no scorer revision recorded"


def _runs_phrase(count: int) -> str:
    """`one run` or `mean of four runs`: the page's `runsPhrase`."""
    if count == 1:
        return "one run"
    word = _RUN_WORDS[count] if count < len(_RUN_WORDS) else str(count)
    return f"mean of {word} runs"


def _failed(entry: Entry) -> bool:
    return entry.status in _FAILED or (entry.status == "evaluated" and not entry.viable)


class Archive:
    """Append-only record of a design run, resumable across processes."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / SPEC_DIR).mkdir(exist_ok=True)
        self.path = self.directory / LINEAGE_FILE
        self._entries: list[Entry] = list(self._load())

    # ------------------------------------------------------------------ reading

    def _load(self) -> Iterator[Entry]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Entry(**json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    def seen_hashes(self) -> set[str]:
        return {entry.hash for entry in self._entries}

    def next_generation(self) -> int:
        return max((entry.generation for entry in self._entries), default=-1) + 1

    def evaluated(self) -> list[Entry]:
        return [e for e in self._entries if e.status == "evaluated" and e.index is not None]

    def leaderboard(self, limit: int = 10) -> list[Entry]:
        """Best viable candidates first. Non-viable entries never rank.

        A non-viable ruleset is not a low-scoring one -- it failed a gate, meaning
        something about it does not work as a game. Letting it hold a leaderboard slot
        would send the next generation to build on a broken foundation.

        The order is by point estimate, which is what a reader expects of a
        leaderboard -- but it is not what anything should *select* on. Use
        ``top_group`` for that: on this corpus the interval is wider than the gaps
        between most adjacent rows, so the ordering here is largely a display of
        sampling noise.
        """
        ranked = [e for e in self.evaluated() if e.viable]
        ranked.sort(key=lambda e: -(e.index or 0.0))
        return ranked[:limit]

    def top_group(self, limit: int = 10) -> list[Entry]:
        """The leader, plus everything the leader does not beat.

        The honest shape of a leaderboard whose rows overlap. A strict order implies a
        confidence the evidence does not carry, so the loop selects uniformly from this
        group rather than sharpening an ordering the intervals say is not real.

        With no interval recorded -- an old run, or a corpus too small to resample --
        this degenerates to the strict ordering, because that is the only comparison
        available and pretending otherwise would tie everything together forever.
        """
        board = self.leaderboard(limit)
        if not board:
            return []
        leader = board[0]
        return [entry for entry in board if entry is leader or not leader.beats(entry)]

    def grid(self, prefer: Any | None = None) -> grid_module.Grid:
        """The archive as a behaviour grid: one elite per kind of game.

        The leaderboard is one view over this rather than the other way round. A ranked
        board answers "what is the best game found", which is a question whose answer
        sits inside its own error bars; the grid answers "what kinds of game have been
        found, and what is the best of each", which is a question this search can settle.
        """
        return grid_module.build(self.evaluated(), prefer=prefer)

    def failures(self, limit: int = 8) -> list[Entry]:
        return [e for e in self._entries if e.status in _FAILED][-limit:]

    def failed_children(self, parent_hash: str | None, limit: int = 6) -> list[Entry]:
        """What was already tried from this parent and did not work.

        A revision's failed siblings are the most specific thing the next prompt can
        carry: they name ways *this* design goes wrong rather than ways designs in
        general do. Includes candidates that were played and disqualified, not only
        those rejected before a tournament -- a gate failure after 144 games is the
        most informative failure of the three.
        """
        if parent_hash is None:
            return []
        siblings = [
            entry
            for entry in self._entries
            if entry.parent == parent_hash and _failed(entry)
        ]
        return siblings[-limit:]

    def load_spec(self, digest: str) -> dict[str, Any] | None:
        path = self.directory / SPEC_DIR / f"{digest}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as handle:
            spec: dict[str, Any] = json.load(handle)
        return spec

    # ------------------------------------------------------------------ writing

    def record(self, entry: Entry, spec: dict[str, Any] | None = None) -> Entry:
        if spec is not None:
            path = self.directory / SPEC_DIR / f"{entry.hash}.json"
            if not path.exists():
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(spec, handle, indent=2, sort_keys=True)
                    handle.write("\n")

        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.as_dict(), separators=(",", ":")) + "\n")
        self._entries.append(entry)
        return entry

    def summary(self, revision: str | None = None) -> str:
        """The run in words: counts, the board, the grid and whether anything improved.

        With a ``revision`` (the `leaderboard` command passes `DEFAULT_WEIGHTS`) the board
        is that revision's (EV3, `docs/plan-evolving-score.md` §3, §5.2): a row scored
        under another is still placed by its number, marked *provisional* with the
        revision it came from, and never compared -- *tied for first* reads only rows
        under ``revision`` -- and a header says how many rows are provisional. Without
        one, as the design loop calls it, every row is read as its own run pinned it
        (§5.3), as before.
        """
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        board = self.leaderboard(5)

        lines = [
            f"{len(self._entries)} candidates: "
            + ", ".join(f"{status} {count}" for status, count in sorted(counts.items()))
        ]
        tied: set[str] = set()
        if revision is not None:
            rows = self.leaderboard(len(self._entries))
            current = [entry for entry in rows if entry.weights_version == revision]
            if len(current) < len(rows):
                lines.append(f"  {provisional_sentence(len(rows) - len(current), len(rows))}")
            if current:
                leader = current[0]
                tied = {
                    entry.hash for entry in current
                    if entry is not leader and not leader.beats(entry)
                }
        elif board:
            tied = {entry.hash for entry in self.top_group(len(board))}
        if board:
            lines.append("\n  best so far:")
            for rank, entry in enumerate(board, 1):
                # The objective is the same kind of number as the index, so it is shown
                # only beside a count, inside the parenthesis the count governs.
                said = []
                if entry.runs and entry.objective is not None:
                    said.append(f"objective {entry.objective:.3f}")
                if entry.judge is not None:
                    said.append(f"judge {entry.judge:.2f}")
                lines.append(
                    f"  {rank}. {entry.name} [{entry.hash}] {entry.scored_as()}"
                    + (f" ({', '.join(said)})" if said else "")
                    + ("  = tied for first" if entry.hash in tied and rank > 1 else "")
                    + (
                        f"  (provisional: {_scored_under(entry)})"
                        if revision is not None and entry.weights_version != revision
                        else ""
                    )
                )
            if any(not entry.runs for entry in board):
                lines.append(
                    "  (no score: written before the archive recorded how many runs a "
                    "number rests on, so the number is not shown)"
                )
        else:
            lines.append("  no viable candidate yet")

        # The line that decides whether a run found anything. A leaderboard rises
        # whether or not the rise means anything, so it is reported next to the count
        # of generations that actually beat their parent -- and when that count is
        # zero, this says so in those words rather than leaving a rising board to
        # imply otherwise.
        grid = self.grid()
        if grid.filled:
            kinds = "kind" if grid.filled == 1 else "kinds"
            lines.append(
                f"\n  {grid.filled} {kinds} of game found, of {grid_module.CELLS} the "
                "grid can express"
            )

        judged = [e for e in self._entries if e.separated is not None]
        if judged:
            wins = sum(1 for e in judged if e.separated)
            lines.append(
                f"\n  {wins} of {len(judged)} revisions beat the parent by more than "
                "the noise in both"
                + ("" if wins else " -- this run has not yet found an improvement")
            )
        return "\n".join(lines)
