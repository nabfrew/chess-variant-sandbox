"""The designer brief's limits (a) to (c), from the spec alone (`unlike-search.md` §2.1, §3).

    uv run python unlike/limits.py <spec.json> [...]

(a) likeness to chess below 0.0525, the game appended alone to the library's games
    (`corpus.corpus_hashes()`: `rulesets/` and `archive/`), `spec.likeness` `k1`;
(b) no more than one of its piece kinds is a piece of chess (a `piece:` word shared with
    `rulesets/chess.json`);
(c) no library game nearer than 0.50 by the same measure.

(d), `validate` and `smoke`, are the CLI's. Reads specs only: no run, no engine, no score.
Exit 0 when every game given passes (a) to (c).
"""

# compute-lease: light

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CUT = 0.0525
NEAREST = 0.50
CHESS_PIECES_MAX = 1


def library_specs() -> tuple[list[str], list[dict[str, Any]]]:
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.spec import load_spec

    hashes = corpus.corpus_hashes()
    paths: dict[str, Path] = {}
    for directory in (corpus.RULESET_DIR, corpus.ARCHIVE_DIR):
        for digest, path in corpus._committed_paths(Path(directory)).items():
            paths.setdefault(digest, path)
    return hashes, [load_spec(paths[h]) for h in hashes]


def read(spec: dict[str, Any], hashes: list[str], specs: list[dict[str, Any]]) -> dict[str, Any]:
    from chess_variant_sandbox.spec import load_spec, spec_hash
    from chess_variant_sandbox.spec.likeness import likeness, words_of

    chess_spec = load_spec(ROOT / "rulesets" / "chess.json")
    chess_at = hashes.index(spec_hash(chess_spec))
    relation = likeness([*specs, spec])
    me = len(specs)
    like = relation.similarity(me, chess_at)
    near = max(range(me), key=lambda i: relation.similarity(me, i))
    chess_words = {w for w in words_of(chess_spec) if w.startswith("piece:")}
    shared = len({w for w in words_of(spec) if w.startswith("piece:")} & chess_words)
    nearest = relation.similarity(me, near)
    return {
        "hash": spec_hash(spec),
        "likeness": like,
        "chess_pieces": shared,
        "nearest": (specs[near].get("name", hashes[near]), hashes[near], nearest),
        "a": like < CUT,
        "b": shared <= CHESS_PIECES_MAX,
        "c": nearest < NEAREST,
    }


def main(argv: list[str]) -> int:
    from chess_variant_sandbox.spec import load_spec

    hashes, specs = library_specs()
    ok = True
    for name in argv:
        r = read(load_spec(Path(name)), hashes, specs)
        n_name, n_hash, n_sim = r["nearest"]
        print(f"{name} [{r['hash']}]")
        print(f"  (a) likeness to chess {r['likeness']:.4f}  below {CUT}: {'yes' if r['a'] else 'NO'}")
        print(f"  (b) pieces of chess {r['chess_pieces']}  at most {CHESS_PIECES_MAX}: "
              f"{'yes' if r['b'] else 'NO'}")
        print(f"  (c) nearest library game {n_name} [{n_hash}] {n_sim:.4f}  below {NEAREST}: "
              f"{'yes' if r['c'] else 'NO'}")
        ok = ok and r["a"] and r["b"] and r["c"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
