# compute-lease: light
"""Copy the library stories' recorded games into `recorded.json` beside this file (LP1).

    uv run python chess_variant_sandbox/web/stories/copy_records.py [RUNS]

A library story's two games are records of its games' measurement, line `line` of
`RUNS/<hash>-<seed>/games.jsonl` (default: this checkout's `runs/`, else the main
checkout's when this is a worktree). The site is never sent a `runs/<hash>-<seed>/` game
directory (`deploy/copy-runs.sh`), so the moves the page replays are copied here, each with
the run and line it came from; nothing is played or edited, and `web/stories.py` replays
them through `Engine`. `tests/test_stories.py` holds each copy to its record wherever the
run is on disk. Read only: nothing under `runs/` is written.
"""

import json
import sys
from pathlib import Path

from chess_variant_sandbox.measurement import main_runs_dir
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.stories import LIBRARY, RECORDED_FILE, STORIES


def runs_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    story = next(s for s in STORIES if s.kind == LIBRARY)
    first = story.wrong.game
    local = library.RUNS_DIR / f"{first.hash}-{first.seed}"
    return library.RUNS_DIR if local.exists() else (main_runs_dir() or library.RUNS_DIR)


def copy_record(runs: Path, digest: str, seed: int, line: int) -> dict[str, object]:
    """One recorded game as `recorded.json` keeps it."""
    rows = (runs / f"{digest}-{seed}" / "games.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[line])
    if row.get("ruleset_hash") != digest:
        raise SystemExit(f"runs/{digest}-{seed} line {line} is another game's record")
    return {
        "hash": digest, "run": f"runs/{digest}-{seed}", "seed": seed, "line": line,
        "of": len(rows), "agents": list(row["agents"]), "moves": list(row["moves"]),
        "winner": row["winner"], "reason": row["reason"], "plies": row["plies"],
    }


def main() -> None:
    runs = runs_root()
    out: dict[str, object] = {}
    for story in STORIES:
        if story.kind != LIBRARY:
            continue
        for played in (story.wrong.game, story.right.game):
            assert played.seed is not None and played.line is not None
            out[played.record_key] = copy_record(runs, played.hash, played.seed, played.line)
            print(played.record_key, "from", runs)
    RECORDED_FILE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
