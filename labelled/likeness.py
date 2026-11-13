"""Place a set game in its likeness quarter (labelled-set.md §2.1), from the specs alone.

    uv run python labelled/likeness.py labelled/<name>.json [...]

Each game is appended *alone* to the library's games (`corpus.corpus_hashes()`: `rulesets/`
and `archive/`) and its `spec.likeness` (`k1`) similarity to chess is read off that relation,
IDF over the library plus the one game. The library must be as it stood at `74eaa6ca2`, the
commit R-S2b branched from; `--check` asserts that `rulesets/` and `archive/` have not changed
since. Reads specs only: no run, no engine, no score.
"""

# compute-lease: light

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "74eaa6ca2"
EDGES = (0.0525, 0.1575, 0.4075)


def quarter(value: float) -> str:
    return next((f"Q{i + 1}" for i, edge in enumerate(EDGES) if value < edge), "Q4")


def library_unchanged() -> bool:
    out = subprocess.run(
        ["git", "diff", "--quiet", BASE, "HEAD", "--", "rulesets", "archive"], cwd=ROOT,
        check=False,
    )
    return out.returncode == 0


def likeness_to_chess(spec: dict) -> float:
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.spec import load_spec, spec_hash
    from chess_variant_sandbox.spec.likeness import likeness

    hashes = corpus.corpus_hashes()
    paths = {}
    for directory in (corpus.RULESET_DIR, corpus.ARCHIVE_DIR):
        paths.update({d: p for d, p in corpus._committed_paths(Path(directory)).items() if d not in paths})
    specs = [load_spec(paths[h]) for h in hashes]
    chess = spec_hash(load_spec(ROOT / "rulesets" / "chess.json"))
    relation = likeness([*specs, spec])
    return relation.similarity(len(specs), hashes.index(chess))


def main(argv: list[str]) -> int:
    from chess_variant_sandbox.spec import load_spec

    if not library_unchanged():
        print(f"the library has changed since {BASE}: §2.1 reads it as it stood there")
        return 1
    for name in argv:
        value = likeness_to_chess(load_spec(Path(name)))
        print(f"{name}: likeness {value:.4f} {quarter(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
