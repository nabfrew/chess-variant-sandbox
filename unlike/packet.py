"""Write a blind reader's packet for a designed game (`unlike-search.md` §2.2 item 3).

    uv run python unlike/packet.py <session dir> <packet dir>

The reader is given the rules in words (`describe_rules`), the summary, the design reason and
the six transcripts, and neither the designer's log nor any number. The spec is copied beside
them only so that `play` and `Engine` can load the game. Reads specs and text only.
"""

# compute-lease: light

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    from chess_variant_sandbox.spec.describe import describe_rules

    session, packet = Path(argv[0]), Path(argv[1])
    spec = json.loads((session / "game.json").read_text(encoding="utf-8"))
    (packet / "transcripts").mkdir(parents=True, exist_ok=True)
    rules = "\n".join(describe_rules(spec))
    (packet / "rules.txt").write_text(
        f"{spec.get('name', '')}\n\nSummary: {spec.get('summary', '')}\n\n{rules}\n", encoding="utf-8"
    )
    shutil.copy(session / "reason.md", packet / "reason.md")
    shutil.copy(session / "game.json", packet / "game.json")
    for t in sorted((session / "transcripts").iterdir()):
        shutil.copy(t, packet / "transcripts" / t.name)
    print(f"packet for {spec.get('name')} in {packet}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
