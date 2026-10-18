"""The searching players' tournaments on the two games every bump holds still (Q935b).

`tests/test_spy_players.py` pins the digests; this module computes one, so the same code
can be run against main's core to take the pin (`python -m tests.inert_players` from a
checkout, under a `heavy` lease). Records with their durations left out, hashed in order.
"""

# compute-lease: heavy

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each game's players: the alternating ladder's searching names on chess (quiescence, the
# table, the repairs), the round players on Gardner in rounds.
TOURNAMENTS = {
    "rulesets/chess.json": ("greedy", "minimax:nodes=400", "minimax:nodes=1600"),
    "tests/_retired_gardner_in_rounds.json": ("greedy", "matrix:nodes=1600"),
}


def digest(path: str, root: Path = ROOT) -> str:
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

    spec = load_spec(root / path)
    result = run_tournament(
        spec,
        TournamentConfig(agents=TOURNAMENTS[path], games_per_pairing=2, seed=935, workers=2),
    )
    rows = []
    for record in result.records:
        row = record.as_dict()
        row.pop("duration_ms", None)
        rows.append(json.dumps(row, sort_keys=True))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    for path in TOURNAMENTS:
        print(path, digest(path, root))
