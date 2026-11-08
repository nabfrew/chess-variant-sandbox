"""The PGO training run: the workload whose profile `core.profdata` records.

Run by `regenerate.py` with an instrumented core installed; it only drives the CLI, so it can
also be run by hand against any build. The games are a spread of the library's families
(drops, palace and river, wrap-free variants, a rock-paper-scissors capture rule, a race, a
hill, a torus) and its three agent kinds at small budgets, plus perft. It deliberately leaves
out chess, crazyhouse and torus chess at `minimax:nodes=25600`, seed 0: those are the timed
tournaments that judged the profile (docs/reference/perf-c8-pgo-and-lto.md), and a profile
trained on its own exam would grade itself.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERFT = [("rulesets/chess.json", 5), ("rulesets/shogi.json", 4), ("rulesets/xiangqi.json", 4)]
GAMES = [
    "rulesets/shogi.json",
    "rulesets/xiangqi.json",
    "rulesets/rps-hierarchy-chess.json",
    "rulesets/racingkings.json",
    "rulesets/kingofthehill.json",
    "archive/torus-less-extended.json",
    "rulesets/atomic.json",
    "rulesets/duck-chess.json",
]
AGENTS = ["random", "greedy", "minimax:nodes=1600"]


def cli(*args: str) -> None:
    subprocess.run(["uv", "run", "--no-sync", "chess-variant-sandbox", *args], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def main() -> int:
    for path, depth in PERFT:
        cli("bench", path, "--depth", str(depth), "--threads", "1")
    with tempfile.TemporaryDirectory() as tmp:
        for i, path in enumerate(GAMES):
            cli("tournament", path, "--agents", *AGENTS, "--games", "2", "--seed", "1000",
                "--workers", "4", "--out", str(Path(tmp) / str(i)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
