"""Check a new or edited ruleset against what the corpus asks of it, in seconds (Q1156).

    uv run python tests/check_new_game.py                     # the games changed since main
    uv run python tests/check_new_game.py rulesets/x.json     # these games
    uv run python tests/check_new_game.py --all               # every game (about two minutes)

Three steps, all read-only:

1. `tests/corpus_pins.py --check`: the generated freeze rows (`tests/corpus_pins.json`). If
   any is owed it says to run `uv run python tests/corpus_pins.py`, which writes them.
2. The tests marked `new_game` (`tests/new_game.py`), on one core: the checks a game added,
   edited or retired trips -- `FROZEN`, the grid ceiling, showcase coverage, `COSTS`, the
   names, the edits table and the rest that `docs/reference/corpus-pins.md` §5 lists with
   what to do when each fires. The ones that take a game as a parameter run for the chosen
   games only; the ones that read the whole corpus at once always run.
3. For a chosen game with no `FROZEN` row, the row it would take, printed for the designer
   to paste (never written: the row is the game's identity, added on purpose).

The games, unless named: every `rulesets/*.json` and `archive/*.json` that differs from where
this branch left `main` (committed or not) or that git does not track yet. It plays no game
and runs no `slow` test, so it needs no lease. Exit 0 only when both steps are clean.
"""

# compute-lease: light

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("rulesets", "archive")
MARKED = "pytest.mark.new_game"


def _git(*args: str) -> list[str]:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return done.stdout.split() if done.returncode == 0 else []


def changed_games() -> list[str]:
    """The corpus files this branch touched since main, plus those git does not track."""
    base = (_git("merge-base", "HEAD", "main") or ["HEAD"])[0]
    touched = _git("diff", "--name-only", base, "--", *FOLDERS)
    touched += _git("ls-files", "--others", "--exclude-standard", "--", *FOLDERS)
    return sorted(
        {
            name
            for name in touched
            if name.endswith(".json")
            and name.split("/")[0] in FOLDERS
            and name.count("/") == 1
            and (ROOT / name).exists()
        }
    )


def marked_files() -> list[str]:
    """Every test file that carries the mark, found by reading them (no import)."""
    return sorted(
        f"tests/{path.name}"
        for path in (ROOT / "tests").glob("test_*.py")
        if MARKED in path.read_text(encoding="utf-8")
    )


def frozen_rows_owed(games: list[str]) -> list[str]:
    """`FROZEN` rows the chosen games have none of, spelled as the table spells them.

    Printed, never written: adding the row is the designer's act (corpus-pins.md §5), and a
    game that *has* a row whose numbers moved is an edit, which owes the old hash in a
    comment and a row in `spec/edits.py::EDITED`, not a fresh row.
    """
    source = (ROOT / "tests" / "test_schema_nine.py").read_text(encoding="utf-8")
    missing = [game for game in games if f'"{game}": (' not in source]
    if not missing:
        return []
    from chess_variant_sandbox.measurement import tables_fingerprint
    from chess_variant_sandbox.spec import load_spec, spec_hash

    rows = []
    for game in missing:
        spec = load_spec(ROOT / game)
        rows.append(f'    "{game}": ("{spec_hash(spec)}", "{tables_fingerprint(spec)}"),')
    return rows


def main(argv: list[str]) -> int:
    started = time.perf_counter()
    every = "--all" in argv
    named = [arg.replace("\\", "/") for arg in argv if not arg.startswith("-")]
    games = named or ([] if every else changed_games())

    print("1. the generated freeze rows (tests/corpus_pins.py --check)", flush=True)
    pins = subprocess.run(
        [sys.executable, "tests/corpus_pins.py", "--check"], cwd=ROOT, check=False
    ).returncode
    if pins:
        print("   owed: run `uv run python tests/corpus_pins.py`, which writes them", flush=True)

    which = "every game" if every else (", ".join(games) or "no game changed since main")
    print(f"2. the corpus checks marked new_game, per-game ones for: {which}", flush=True)
    tests = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-m", "new_game and not slow",
            "-p", "no:cacheprovider", "--games", "all" if every else ",".join(games),
            *marked_files(),
        ],
        cwd=ROOT,
        check=False,
    ).returncode

    rows = frozen_rows_owed(games)
    if rows:
        print(
            "3. no FROZEN row (tests/test_schema_nine.py) for a new game; add it with a comment "
            "naming the task, in path order:"
        )
        print("\n".join(rows))

    verdict = "clean" if not (pins or tests) else "something to do (above)"
    print(f"check-new-game: {verdict} in {time.perf_counter() - started:.1f} s")
    return 1 if pins or tests else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
