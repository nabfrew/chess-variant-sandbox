"""Assemble a session's record into unlike/sessions/<name>.md and unlike/sessions/<name>/.

    python assemble.py <worktree> <session dir> <verdict line> <reader dir> [<reader dir> ...]

A first hand-in (session/first-handin) is kept beside the final one. Readers are given in order.
"""
import json
import shutil
import sys
from pathlib import Path

wt, session, verdict = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
readers = [Path(p) for p in sys.argv[4:]]
spec = json.loads((session / "game.json").read_text(encoding="utf-8"))
name = spec["name"].replace(" ", "-")
out = wt / "unlike" / "sessions"
side = out / name
side.mkdir(parents=True, exist_ok=True)


def copy_handin(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "game.json", dst / "game.json")
    if (dst / "transcripts").exists():
        shutil.rmtree(dst / "transcripts")
    shutil.copytree(src / "transcripts", dst / "transcripts")


parts = [f"# {spec['name']}: designer session record (`unlike-search.md` §2)\n", verdict + "\n"]
first = session / "first-handin"
if first.exists():
    copy_handin(first, side / "first-handin")
    parts += ["\n## First hand-in: design reason\n", (first / "reason.md").read_text(encoding="utf-8"),
              "\n## First hand-in: break attempts and design log\n", (first / "log.md").read_text(encoding="utf-8")]
copy_handin(session, side / ("revised-handin" if first.exists() else "handin"))
label = "Revised hand-in" if first.exists() else "Hand-in"
parts += [f"\n## {label}: design reason\n", (session / "reason.md").read_text(encoding="utf-8"),
          f"\n## {label}: break attempts and design log\n", (session / "log.md").read_text(encoding="utf-8")]
for i, r in enumerate(readers, 1):
    parts += [f"\n## Blind read {i} ({r.name})\n", (r / "reading.md").read_text(encoding="utf-8")]
(out / f"{name}.md").write_text("\n".join(parts), encoding="utf-8")
print(out / f"{name}.md", side)
