# compute-lease: heavy
"""R-L4b's record replay (`docs/reference/capture-effects-design.md` §4, *The proof that no game
changed its play*): the stored records of every game with a capture entry, and chess, applied
move by move through the core this process loads. It replays and never re-plays.

For each record in the main checkout's `runs/<hash>-<seed>/games.jsonl` of such a game: every
move must be in the core's legal list, the list's length must equal the record's `branching` at
that ply, and the position after the last move must end the game with the record's winner and
reason after the record's plies. The script also prints, per run directory, a digest of every
final position (`repr` of the `GameState`, which carries the board, the key, the hands, the
points and the purse), so two cores can be held to the same final positions as well as to the
records: run it once with `--core OLD.pyd` and once on the venv's core, and compare.

Which games: every file under `rulesets/` and `archive/` whose rules write a capture entry
(`strike`, `contacts`, `explode_on_capture`, a capture hand, a bounty, or wild pieces, whose
route bounty makes `Tables::pays` true), and chess. R-L4b5 (a round on the list) adds every game
in rounds (`rules.simultaneous`), with or without a capture entry, since `rounds.rs` steps 5 and
6 are the list's walk: a round's record is pairs in seat order, each move in its player's list
(`legal_moves_for`), the two lists' lengths the record's two `branching` entries.

    uv run python tests/capture_record_replay.py --runs ../../../runs [--core OLD.pyd] \\
        [--digests out.json] [--workers 4] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_core(path: str | None) -> None:
    """Load `path` as `chess_variant_sandbox._core` in place of the venv's own."""
    if not path:
        return
    from importlib.machinery import ExtensionFileLoader

    loader = ExtensionFileLoader("chess_variant_sandbox._core", path)
    found = importlib.util.spec_from_file_location("chess_variant_sandbox._core", path, loader=loader)
    assert found is not None
    import chess_variant_sandbox

    module = importlib.util.module_from_spec(found)
    sys.modules["chess_variant_sandbox._core"] = module
    loader.exec_module(module)
    chess_variant_sandbox._core = module  # type: ignore[attr-defined]


def main_runs() -> Path:
    """The main checkout's `runs/`, from this worktree or the main checkout itself."""
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return (ROOT / common).resolve().parent / "runs"


def has_capture_entry(spec: dict[str, Any]) -> bool:
    rules = spec.get("rules", {})
    if any(key in rules for key in ("strike", "contacts", "explode_on_capture")):
        return True
    drops = rules.get("drops")
    if drops is not None and drops.get("from_captures", True):
        return True
    if rules.get("npcs"):
        return True
    return any(piece.get("bounty") for piece in spec.get("pieces", []))


def chosen_files() -> list[Path]:
    out = []
    for path in sorted([*(ROOT / "rulesets").glob("*.json"), *(ROOT / "archive").glob("*.json")]):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if "simultaneous" in spec.get("rules", {}):
            out.append(path)  # R-L4b5: every round game (its takings are the list's walk)
            continue
        if path.name == "chess.json" and path.parent.name == "rulesets" or has_capture_entry(spec):
            out.append(path)
    return out


def replay_dir(job: tuple[str, str, str | None, int]) -> dict[str, Any]:
    """Replay one run directory's records; the problems found and the final positions' digest."""
    file, directory, core, limit = job
    load_core(core)
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.tournament.record import read_records

    engine = Engine(load_spec(Path(file)))
    digest = hashlib.sha256()
    problems: list[str] = []
    count = 0
    for record in read_records(Path(directory) / "games.jsonl"):
        if limit and count >= limit:
            break
        count += 1
        state = engine.initial_state()
        where = f"{Path(directory).name} seed {record.seed}"
        ply = 0
        moves = record.moves
        while ply < len(moves):
            if engine.is_round(state):
                # R-L4b5: a round, its two moves in seat order, each in its player's list.
                pair = moves[ply:ply + 2]
                lists = [engine.legal_moves_for(state, player) for player in (0, 1)]
                bad = [
                    f"{len(lists[i])} legal moves for {i}, the record {record.branching[ply + i]}"
                    for i in (0, 1)
                    if record.branching and ply + i < len(record.branching)
                    and len(lists[i]) != record.branching[ply + i]
                ]
                bad += [f"move {pair[i]} not legal for {i}" for i in range(len(pair)) if pair[i] not in lists[i]]
                if len(pair) != 2 or bad:
                    problems.append(f"{where} ply {ply}: " + ("; ".join(bad) or "half a round"))
                    break
                state = engine.apply_round(state, pair[0], pair[1])
                ply += 2
                continue
            move = moves[ply]
            legal = engine.legal_moves(state)
            if record.branching and ply < len(record.branching) and len(legal) != record.branching[ply]:
                problems.append(f"{where} ply {ply}: {len(legal)} legal moves, the record {record.branching[ply]}")
                break
            if move not in legal:
                problems.append(f"{where} ply {ply}: move {move} not legal")
                break
            state = engine.apply(state, move)
            ply += 1
        else:
            outcome = engine.terminal(state)
            ended = (None, None) if outcome is None else (outcome.winner, outcome.reason)
            if ended != (record.winner, record.reason) or len(record.moves) != record.plies:
                problems.append(f"{where}: ends {ended} after {len(record.moves)}, "
                                f"the record {(record.winner, record.reason)} after {record.plies}")
        digest.update(f"{record.seed}:{state!r}\n".encode())
        if len(problems) >= 5:
            break
    from chess_variant_sandbox import _core

    return {"dir": Path(directory).name, "records": count, "problems": problems, "digest": digest.hexdigest(),
            "core": str(getattr(_core, "BUILD_FINGERPRINT", "unknown"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="")
    parser.add_argument("--core", default=None)
    parser.add_argument("--digests", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="records per run directory (0: all)")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    runs = Path(args.runs) if args.runs else main_runs()
    load_core(args.core)
    from chess_variant_sandbox.spec import load_spec, spec_hash

    jobs = []
    for path in chosen_files():
        name = spec_hash(load_spec(path))
        for directory in sorted(runs.glob(f"{name}-*")):
            if (directory / "games.jsonl").exists():
                jobs.append((str(path), str(directory), args.core, args.limit))
    print(f"{len(chosen_files())} games, {len(jobs)} run directories under {runs}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(replay_dir, jobs):
            results.append(result)
            if result["problems"]:
                print(f"{result['dir']}: " + "; ".join(result["problems"]), flush=True)
    records = sum(result["records"] for result in results)
    bad = [result for result in results if result["problems"]]
    cores = sorted({result["core"] for result in results})
    print(f"{records} records in {len(results)} directories replayed on core {', '.join(cores)}; "
          f"{len(bad)} directories with a problem")
    if args.digests:
        Path(args.digests).write_text(
            json.dumps({result["dir"]: result["digest"] for result in results}, indent=1, sort_keys=True),
            encoding="utf-8",
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
