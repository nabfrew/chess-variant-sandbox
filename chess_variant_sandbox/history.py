"""A score's history: what it was, kept and dated, when a new measurement replaces it (Q649).

`docs/plan-confidence.md` §3.2 step 3 (T4). **The score on the page is always the newest
current measurement; the history says what it was.** Nothing here makes a number: a
replaced measurement is kept exactly as it was written, under the game's own directory,
and a line is appended to a ledger beside it.

**When.** Whenever `corpus` writes a measurement over one with a *different basis* (any
stale outgoing file: another roster, net, build, weights, metrics or seed set). A re-run
under the same basis replaces nothing a reader could tell apart (`CLAUDE.md`,
*Determinism*) and keeps nothing.

**Where.** `runs/<hash>/previous/<n>/`: the outgoing `canonical.json` and its
`index-<weights>.json`, and its runs as `previous/<n>/<hash>-<seed>/`, moved where the new
measurement plays them again and copied where it keeps their games (a re-score). Numbered
from 1, never deleted and never overwritten. `runs/<hash>/history.jsonl` gets one line per
replacement: the headline, its interval and run count, the basis, the dates, why it was
replaced and which basis fields moved (`Entry`).

**What reads it.** `web.library.Game.history` for the local application and `/admin`.
Nothing under `metrics/` imports this file and it is in no fingerprint set
(`tests/test_history.py`), so keeping a history moves no score and stales nothing. **Nothing
in it is ever a headline**: the headline is `canonical.json`'s or *out of date*, and a
history line is a past value labelled past. A site visitor sees one as a *was* line under
the trust line's *more* (the user's answer to the plan's §5 Q1, A, 2026-10-03), so
`SHOW_WAS_TO_VISITORS` is ``True``.

**How much a score has moved** (`volatility`, EV6, `docs/plan-evolving-score.md` §7.1):
the spread of a game's stored scores at one weights version, beside its interval, in
words with no cut-off. `/admin` and the local scores panel only; it changes no score and
no trust level, and a visitor is sent none of it.

**A campaign that measures in another tree** and delivers by copying would replace the file
with no history; `deliver` is the delivery that keeps it (`python -m
chess_variant_sandbox.history deliver <from-runs> <to-runs>`).
"""

from __future__ import annotations

import datetime
import json
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HISTORY_FILE = "history.jsonl"
PREVIOUS_DIR = "previous"

# §5 Q1, *may a game's page show its previous score?*, answered A by the user on 2026-10-03:
# yes, under *more*, dated and labelled with the players that made it. A visitor reads
# `for_visitors`' lines as *was* sentences in the trust line (`trust.shown`); the full
# history, with bases, stays the maintainer's (local application, `/admin` and an admin's
# view of a game page on the site). False sends a visitor none of it.
SHOW_WAS_TO_VISITORS = True

# The basis fields that say *who played*: the roster, its fitted objective, its nets and
# what the runner hands the core for them (`measurement.Basis`). A replacement that moved
# one of these was made by other players; `convergence` reads the same set.
PLAYER_FIELDS = ("agents", "fit", "net", "players", "agents_source")


def _measurement_dir(digest: str, runs_dir: Path) -> Path:
    # `corpus.measurement_dir`, without importing `corpus` (which imports this module).
    return Path(runs_dir) / digest


def history_path(digest: str, runs_dir: Path) -> Path:
    return _measurement_dir(digest, runs_dir) / HISTORY_FILE


def previous_root(digest: str, runs_dir: Path) -> Path:
    return _measurement_dir(digest, runs_dir) / PREVIOUS_DIR


def digest_of(basis: Mapping[str, Any] | None) -> str:
    """`trust.measurement_digest` of a stamp, ``""`` for a measurement that recorded none."""
    from chess_variant_sandbox.trust import measurement_digest

    return measurement_digest(basis) if isinstance(basis, Mapping) else ""


def moved_fields(old: Mapping[str, Any] | None, new: Mapping[str, Any] | None) -> list[str]:
    """Which basis fields differ between two stamps, sorted; ``["basis"]`` if one is absent."""
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        return ["basis"]
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def file_date(path: Path) -> str | None:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).date().isoformat()


# ------------------------------------------------------------------ keeping the outgoing


def _numbers(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    return sorted(int(child.name) for child in root.iterdir()
                  if child.is_dir() and child.name.isdigit())


def kept_already(digest: str, runs_dir: Path, canonical: Path) -> int | None:
    """The `previous/<n>` already holding this exact `canonical.json`, or ``None``.

    What makes keeping idempotent: a measurement that failed after its outgoing file was
    kept is retried with the same outgoing file still in place (it is copied, never moved,
    until the new one is written), and is not kept a second time.
    """
    try:
        body = canonical.read_bytes()
    except OSError:
        return None
    root = previous_root(digest, runs_dir)
    for number in _numbers(root):
        held = root / str(number) / "canonical.json"
        if held.exists() and held.read_bytes() == body:
            return number
    return None


def keep(
    digest: str,
    runs_dir: Path,
    canonical: Path,
    seeds: Iterable[int],
    moving: Iterable[int] = (),
) -> int:
    """Keep the outgoing measurement under `previous/<n>/` and return ``n``.

    ``canonical`` is the outgoing file (in `runs/<hash>/`, or the old single-run path).
    ``seeds`` are its runs; those in ``moving`` are about to be played again and are moved
    whole, the rest are copied, because a re-score reads their games where they are. The
    outgoing `canonical.json` and its index files are copied: the new measurement's write
    replaces them, so a measurement that fails half way still has its file in place.
    """
    runs_dir = Path(runs_dir)
    already = kept_already(digest, runs_dir, canonical)
    if already is not None:
        return already
    root = previous_root(digest, runs_dir)
    number = (_numbers(root) or [0])[-1] + 1
    target = root / str(number)
    target.mkdir(parents=True)
    shutil.copy2(canonical, target / "canonical.json")
    if canonical.parent == _measurement_dir(digest, runs_dir):
        for index in sorted(canonical.parent.glob("index-*.json")):
            shutil.copy2(index, target / index.name)
    moving = set(moving)
    for seed in sorted(set(seeds)):
        run = runs_dir / f"{digest}-{seed}"
        if not run.is_dir():
            continue
        if seed in moving:
            shutil.move(str(run), str(target / run.name))
        else:
            shutil.copytree(run, target / run.name)
    return number


# ------------------------------------------------------------------ the ledger


@dataclass(frozen=True)
class Entry:
    """One replaced measurement, as `history.jsonl` records it. A past value, never a
    headline: nothing that draws a score reads this type."""

    n: int
    index: float | None
    ci_low: float | None
    ci_high: float | None
    sd: float | None
    runs: int
    seeds: tuple[int, ...]
    viable: bool | None
    weights_version: str
    basis: Mapping[str, Any] | None
    digest: str  # `trust.measurement_digest` of `basis`, `""` without one
    to: str  # the same, of the measurement that replaced it
    measured: str | None  # the outgoing file's date, UTC
    replaced: str  # the replacement's date, UTC
    needs: str  # `Staleness.needs` of the outgoing against the new basis
    reasons: tuple[str, ...]
    moved: tuple[str, ...]  # basis fields that differ
    convergence: str = ""  # what T5 made of the pair: `held`, `moved`, or why no verdict

    @property
    def path(self) -> str:
        return f"{PREVIOUS_DIR}/{self.n}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "path": self.path,
            "index": self.index,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "sd": self.sd,
            "runs": self.runs,
            "seeds": list(self.seeds),
            "viable": self.viable,
            "weights_version": self.weights_version,
            "basis": dict(self.basis) if self.basis is not None else None,
            "digest": self.digest,
            "to": self.to,
            "measured": self.measured,
            "replaced": self.replaced,
            "needs": self.needs,
            "reasons": list(self.reasons),
            "moved": list(self.moved),
            "convergence": self.convergence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Entry:
        basis = data.get("basis")

        def number(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            n=int(data["n"]),
            index=number("index"),
            ci_low=number("ci_low"),
            ci_high=number("ci_high"),
            sd=number("sd"),
            runs=int(data.get("runs") or 0),
            seeds=tuple(int(seed) for seed in data.get("seeds") or ()),
            viable=None if data.get("viable") is None else bool(data["viable"]),
            weights_version=str(data.get("weights_version") or ""),
            basis=dict(basis) if isinstance(basis, Mapping) else None,
            digest=str(data.get("digest") or ""),
            to=str(data.get("to") or ""),
            measured=data.get("measured"),
            replaced=str(data.get("replaced") or ""),
            needs=str(data.get("needs") or ""),
            reasons=tuple(str(reason) for reason in data.get("reasons") or ()),
            moved=tuple(str(name) for name in data.get("moved") or ()),
            convergence=str(data.get("convergence") or ""),
        )


def entry_for(
    n: int,
    outgoing: Mapping[str, Any],
    incoming_basis: Mapping[str, Any] | None,
    measured: str | None,
    needs: str,
    reasons: Iterable[str],
    convergence: str = "",
) -> Entry:
    """The ledger line for an outgoing `canonical.json`'s contents, replaced by a
    measurement stamped ``incoming_basis``."""
    basis = outgoing.get("basis")
    basis = dict(basis) if isinstance(basis, Mapping) else None
    runs = [run for run in outgoing.get("runs") or [] if isinstance(run, Mapping)]

    def number(key: str) -> float | None:
        value = outgoing.get(key)
        return None if value is None else float(value)

    viable = outgoing.get("viable")
    return Entry(
        n=int(n),
        index=number("index"),
        ci_low=number("ci_low"),
        ci_high=number("ci_high"),
        sd=number("sd"),
        runs=len(runs),
        seeds=tuple(int(seed) for seed in outgoing.get("seeds") or ()),
        viable=None if viable is None else bool(viable),
        weights_version=str(outgoing.get("weights_version") or ""),
        basis=basis,
        digest=digest_of(basis),
        to=digest_of(incoming_basis),
        measured=measured,
        replaced=_today(),
        needs=needs,
        reasons=tuple(reasons),
        moved=tuple(moved_fields(basis, incoming_basis)),
        convergence=convergence,
    )


def append(digest: str, runs_dir: Path, entry: Entry) -> None:
    path = history_path(digest, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.as_dict(), sort_keys=True) + "\n")


def read(digest: str, runs_dir: Path) -> list[Entry]:
    """Every replaced measurement of this game, oldest first. A line that cannot be read is
    skipped, never guessed."""
    path = history_path(digest, runs_dir)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    found: list[Entry] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                found.append(Entry.from_dict(data))
        except (ValueError, KeyError, TypeError):
            continue
    return found


def every(runs_dir: Path) -> dict[str, list[Entry]]:
    """Every game under ``runs_dir`` with a history, by hash: what `/admin` lists."""
    found: dict[str, list[Entry]] = {}
    for path in sorted(Path(runs_dir).glob(f"*/{HISTORY_FILE}")):
        entries = read(path.parent.name, Path(runs_dir))
        if entries:
            found[path.parent.name] = entries
    return found


# ------------------------------------------------------------------ what a page is sent


def _players(basis: Mapping[str, Any] | None) -> str:
    if not basis:
        return "players not recorded"
    agents = [str(agent) for agent in basis.get("agents") or ()]
    words = f"{len(agents)} players"
    if basis.get("net"):
        words += f", nets {basis['net']}"
    elif basis.get("fit"):
        fit = str(basis["fit"])
        words += f", {fit}" if fit.startswith("fit:") else f", fit {fit}"
    return words


def maintainer(entries: Iterable[Entry]) -> list[dict[str, Any]]:
    """The history as the local application and `/admin` show it: every line, newest
    first, with its basis and the reason it was replaced."""
    return [
        {**entry.as_dict(), "players": _players(entry.basis)}
        for entry in sorted(entries, key=lambda entry: entry.n, reverse=True)
    ]


def admin_rows(runs_dir: Path, names: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """`/admin`'s table: each game with a history, its newest replacement first."""
    names = names or {}
    rows = [
        {"hash": digest, "name": names.get(digest, ""), "entries": maintainer(entries)}
        for digest, entries in every(runs_dir).items()
    ]
    return sorted(rows, key=lambda row: (row["entries"][0]["replaced"], row["hash"]),
                  reverse=True)


def for_visitors(entries: Iterable[Entry], current_index: float | None) -> list[dict[str, Any]]:
    """What a visitor is sent (§5 Q1, answered A): a value, its date and that the players
    were earlier ones, only where the players changed and the number differs.
    Empty while `SHOW_WAS_TO_VISITORS` is ``False``; `trust.shown` words each one."""
    if not SHOW_WAS_TO_VISITORS:
        return []
    return [
        {"index": entry.index, "runs": entry.runs, "measured": entry.measured}
        for entry in sorted(entries, key=lambda entry: entry.n, reverse=True)
        if entry.index is not None and entry.index != current_index
        and set(entry.moved) & set(PLAYER_FIELDS)
    ]


# ------------------------------------------------------------------ how much it has moved

# EV6 (`docs/plan-evolving-score.md` §7.1, pre-registered before this was written): how
# many of the newest same-weights revisions the reading compares.
VOLATILITY_K = 5


def _figure(value: float) -> str:
    return f"{value:.3f}"


def _what_moved(moved: Iterable[str]) -> str:
    moved = set(moved)
    if not moved:
        return "nothing recorded moved"
    if moved & set(PLAYER_FIELDS):
        return "the players changed"
    return "its " + ", ".join(sorted(moved)) + " changed"


def volatility(digest: str, runs_dir: Path) -> dict[str, Any]:
    """How much this game's stored score has moved between its revisions (EV6, §7.1).

    The revisions are `history.jsonl`'s lines, oldest first, then the present
    `canonical.json`. Only the trailing run at the newest revision's weights version is
    compared, at most `VOLATILITY_K` of it; revisions before a weights change, or with no
    score, are counted and left out. ``spread`` is the largest minus the smallest index
    compared, ``largest_step`` the largest change between two consecutive ones; both are
    ``None`` with fewer than two, never zero. Beside them the present interval's
    half-width. Threshold-free, and read by nothing that makes a score or a trust level:
    `/admin` and the local app's scores panel only.
    """
    runs_dir = Path(runs_dir)
    # A line records the change that replaced it, so its `moved` is the step to the next.
    revisions: list[dict[str, Any]] = [
        {"index": entry.index, "weights": entry.weights_version,
         "moved": tuple(entry.moved), "replaced": entry.replaced}
        for entry in sorted(read(digest, runs_dir), key=lambda entry: entry.n)
    ]
    half: float | None = None
    try:
        present = json.loads((_measurement_dir(digest, runs_dir) / "canonical.json")
                             .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        present = None
    if isinstance(present, Mapping):
        index = present.get("index")
        revisions.append({"index": None if index is None else float(index),
                          "weights": str(present.get("weights_version") or ""),
                          "moved": (), "replaced": None})
        low, high = present.get("ci_low"), present.get("ci_high")
        if low is not None and high is not None:
            half = (float(high) - float(low)) / 2
    scored = [revision for revision in revisions if revision["index"] is not None]
    unscored = len(revisions) - len(scored)
    weights = scored[-1]["weights"] if scored else ""
    run: list[dict[str, Any]] = []
    for revision in reversed(scored):
        if revision["weights"] != weights:
            break
        run.append(revision)
    run.reverse()
    other_weights = len(scored) - len(run)
    compared = run[-VOLATILITY_K:]
    spread = step = None
    step_moved: tuple[str, ...] = ()
    step_on = None
    for older, newer in zip(compared, compared[1:], strict=False):
        size = abs(newer["index"] - older["index"])
        if step is None or size > step:
            step, step_moved, step_on = size, older["moved"], older["replaced"]
    if len(compared) >= 2:
        values = [revision["index"] for revision in compared]
        spread = max(values) - min(values)

    at = f"at weights {weights}" if weights else "with no weights version recorded"
    if not scored:
        said = "No score stored, so nothing to compare."
    elif len(compared) < 2:
        said = f"One score {at}, so nothing to compare yet."
    else:
        assert spread is not None and step is not None
        said = (f"Moved {_figure(spread)} across its last {len(compared)} scores {at} "
                f"(largest single change {_figure(step)}, when {_what_moved(step_moved)}).")
    if half is not None and scored:
        said += f" Its interval is ±{_figure(half)}."
    left_out = []
    if other_weights:
        left_out.append(f"{other_weights} earlier at other weights")
    if unscored:
        left_out.append(f"{unscored} with no score")
    if left_out:
        said += " Not compared: " + "; ".join(left_out) + "."
    return {
        "hash": digest,
        "revisions": len(compared),
        "weights_version": weights,
        "spread": spread,
        "largest_step": step,
        "step_moved": list(step_moved),
        "step_replaced": step_on,
        "interval_half": half,
        "other_weights": other_weights,
        "unscored": unscored,
        "said": said,
    }


def volatility_rows(runs_dir: Path, names: Mapping[str, str]) -> list[dict[str, Any]]:
    """`volatility` for each named game, by name: `/admin`'s and the local scores panel's
    *How much scores have moved*. Every game is listed, with one revision or none too."""
    rows = [{**volatility(digest, runs_dir), "name": name or digest[:12]}
            for digest, name in names.items()]
    return sorted(rows, key=lambda row: (str(row["name"]).lower(), row["hash"]))


# ------------------------------------------------------------------ delivery


Judge = Callable[[str, Path, int, Mapping[str, Any], Any], str]


def deliver(
    source_runs: Path,
    runs_dir: Path,
    digests: Iterable[str] | None = None,
    judge: Judge | None = None,
    say: Callable[[str], None] = lambda _line: None,
) -> list[str]:
    """Copy measurements from another tree's `runs/` into this one, keeping what they replace.

    The delivery `corpus.measure` does in place, for a measurement taken elsewhere: for each
    game with a `canonical.json` under ``source_runs``, a file here with a *different basis*
    is kept under `previous/<n>/` (its runs that the new one carries moved, the rest
    copied) and a history line is appended; one with the same basis is overwritten, as a
    re-run under the same basis is. ``judge`` is T5's convergence read
    (`corpus.judge_convergence`), called after the copy. Returns the hashes delivered.
    """
    from chess_variant_sandbox.measurement import Basis

    source_runs, runs_dir = Path(source_runs), Path(runs_dir)
    if digests is None:
        digests = [path.parent.name for path in sorted(source_runs.glob("*/canonical.json"))]
    delivered: list[str] = []
    for digest in digests:
        incoming_path = source_runs / digest / "canonical.json"
        incoming = json.loads(incoming_path.read_text(encoding="utf-8"))
        incoming_basis = incoming.get("basis")
        seeds = [int(seed) for seed in incoming.get("seeds") or ()]
        here = _measurement_dir(digest, runs_dir) / "canonical.json"
        number = None
        outgoing: dict[str, Any] = {}
        measured = None
        if here.exists():
            outgoing = json.loads(here.read_text(encoding="utf-8"))
            if digest_of(outgoing.get("basis")) != digest_of(incoming_basis):
                measured = file_date(here)
                held = [int(seed) for seed in outgoing.get("seeds") or ()]
                number = keep(digest, runs_dir, here, held, moving=set(held) & set(seeds))
        target = _measurement_dir(digest, runs_dir)
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted((source_runs / digest).iterdir()):
            # The measurement's own files. Another tree's history, previous runs and
            # convergence record are that tree's, never this one's.
            if item.is_file() and item.name not in (HISTORY_FILE, "convergence.json"):
                shutil.copy2(item, target / item.name)
        for seed in seeds:
            run = source_runs / f"{digest}-{seed}"
            if run.is_dir():
                shutil.copytree(run, runs_dir / run.name, dirs_exist_ok=True)
        if number is not None:
            old = Basis.from_dict(outgoing.get("basis"))
            new = Basis.from_dict(incoming_basis)
            if old is not None and new is not None:
                staleness = old.stale_against(new)
                needs, reasons = staleness.needs, staleness.reasons
            else:
                needs, reasons = "replay", ("it does not record what it was measured under",)
            verdict = judge(digest, runs_dir, number, outgoing, incoming) if judge else ""
            append(digest, runs_dir, entry_for(
                number, outgoing, incoming_basis, measured, needs, reasons, verdict))
            say(f"{digest}: kept the outgoing measurement as previous/{number}"
                + (f"; convergence {verdict}" if verdict else ""))
        else:
            say(f"{digest}: delivered")
        delivered.append(digest)
    return delivered


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[0] != "deliver":
        print("usage: python -m chess_variant_sandbox.history deliver <from-runs> <to-runs> "
              "[hash ...]", file=sys.stderr)
        return 2
    from chess_variant_sandbox.corpus import judge_convergence

    deliver(Path(args[1]), Path(args[2]), args[3:] or None, judge=judge_convergence,
            say=print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
