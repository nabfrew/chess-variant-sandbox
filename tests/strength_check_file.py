# compute-lease: light
"""Writes, or checks, the tracked strength-check file from R-S4b's stored readings (R-S4c).

`chess_variant_sandbox/strength_check.json` is the one place a reader learns whether a library
score was found to depend on the players that measured it. It is a projection of R-S4b's
readings (`runs/R-S4/readings/<hash>.json` in the main checkout, `docs/reference/
strength-check-all-games.md` §14), field for field, plus one row for every library game the
check never reached, with its reason. Nothing here plays a game or reads a score.

    uv run python tests/strength_check_file.py write [--readings <dir>]
    uv run python tests/strength_check_file.py check [--readings <dir>]

`check` exits non-zero and names each row that differs; `tests/test_strength_reading.py` runs it
whenever the main checkout's readings are on disk.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main_runs() -> Path:
    """The main checkout's `runs/`, where R-S4b delivered its readings, from any worktree.

    Read from the worktree's `.git` file (`gitdir: <main>/.git/worktrees/<name>`), so no
    process is started; the main checkout's own `.git` is a directory and names itself."""
    marker = ROOT / ".git"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:"):].strip())
            if gitdir.parent.name == "worktrees":
                return gitdir.parent.parent.parent / "runs"
    return ROOT / "runs"


READINGS = main_runs() / "R-S4" / "readings"
TARGET = ROOT / "chess_variant_sandbox" / "strength_check.json"

# The one game the check had no duel for, and why (R-S4c's brief): joined the library after
# R-S4b's last slice. Named by file, so a re-spelling cannot lose it.
JOINED_LATER = "this game joined the library after the check was made on 13 November 2026"

# Readings withdrawn after they were made, by hash, with the reason a visitor reads. Their
# rows go to `not_checked`; the reading stays on disk under `runs/R-S4/readings`.
WITHDRAWN = {
    # Q1219 (2026-11-15): atomic's word rested on clause B alone, its games against
    # Fairy-Stockfish, whose blast is a different atomic from ours where the two kings come
    # within one blast (touching kings; `docs/reference/fairy-stockfish.md`, *Atomic's
    # touching kings*). Fairy-Stockfish is refused on atomic now, so B cannot be read again.
    "6bd4d9c5769e": (
        "its check rested on games against an outside engine that, it turned out, plays a "
        "different rule when the two kings stand close; the check will be made again"
    ),
}

KEPT = ("name", "word", "holding", "c", "c_error", "f1", "f1_error", "f4", "f4_error",
        "fairy_plays", "read_on")


def _row(reading: dict[str, Any], day: str) -> dict[str, Any]:
    basis = reading.get("basis") or {}
    row = {key: reading.get(key) for key in KEPT}
    row["beside"] = [key for key in ("deep", "drawn_out")
                     if (reading.get("beside") or {}).get(key)]
    row["player"] = basis.get("player") or (reading.get("players") or {}).get("T")
    row["engine"] = basis.get("engine", "")
    row["tables"] = basis.get("tables", "")
    row["build"] = basis.get("build", "")
    row["fairy_engine"] = basis.get("fairy_engine", "")
    row["rule"] = basis.get("rule", "")
    row["script"] = basis.get("script", "")
    row["read_at"] = day
    return row


def build(readings: Path, library: list[str], names: dict[str, str]) -> dict[str, Any]:
    games: dict[str, Any] = {}
    for path in sorted(readings.glob("*.json")):
        reading = json.loads(path.read_text(encoding="utf-8"))
        day = datetime.datetime.fromtimestamp(
            path.stat().st_mtime, datetime.timezone.utc).date().isoformat()
        if str(reading["hash"]) in WITHDRAWN:
            continue
        games[str(reading["hash"])] = _row(reading, day)
    missing = {digest: {"name": names.get(digest, digest),
                        "why": WITHDRAWN.get(digest, JOINED_LATER)}
               for digest in library if digest not in games}
    return {
        "about": ("R-S4b's strength check of every library game, one row a game, as read "
                  "(docs/reference/strength-check-all-games.md sections 2.3 and 14). "
                  "Written by tests/strength_check_file.py from runs/R-S4/readings; "
                  "never edited by hand."),
        "record": "docs/reference/strength-check-all-games.md#14",
        "weights": "v13",
        "games": dict(sorted(games.items())),
        "not_checked": dict(sorted(missing.items())),
    }


def _library() -> tuple[list[str], dict[str, str]]:
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.spec.canonical import load_spec, spec_hash

    hashes = corpus.corpus_hashes(ROOT / "rulesets", ROOT / "archive")
    names: dict[str, str] = {}
    for directory in (ROOT / "rulesets", ROOT / "archive"):
        for path in sorted(directory.glob("*.json")):
            try:
                spec = load_spec(path)
                names[spec_hash(spec)] = str(spec.get("name", path.stem))
            except Exception:  # noqa: BLE001 - a file that is not a ruleset names nothing
                continue
    return hashes, names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--readings", type=Path, default=READINGS)
    args = parser.parse_args(argv)
    hashes, names = _library()
    made = build(args.readings, hashes, names)
    text = json.dumps(made, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    if args.action == "write":
        TARGET.write_text(text, encoding="utf-8")
        print(f"wrote {TARGET.name}: {len(made['games'])} read, "
              f"{len(made['not_checked'])} not checked")
        return 0
    held = json.loads(TARGET.read_text(encoding="utf-8"))
    bad = [digest for digest in set(held["games"]) | set(made["games"])
           if held["games"].get(digest) != made["games"].get(digest)]
    for digest in sorted(bad):
        print(f"differs: {digest}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
