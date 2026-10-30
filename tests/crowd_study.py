# compute-lease: light
"""The crowd, read off its runs (TC8, `docs/plan-turncoat.md` §8.1).

The instrument V2 to V4 and the predictions are graded on, written with the registration and
before the first measured game. Not a test (no ``test_`` prefix); it lives beside
`neutral_oracle.py` because the showcase's areas are the tests and the plan. It reads persisted
records only and replays them through the core, as `analyse` does; it never re-plays a game, and
it decides no move, outcome, control or legality itself (CLAUDE.md, rule 3). Every figure is the
records' own, or a core's:

- **Endings**, the decided share, plies and the first seat's share of decided games are the
  records' own (`reason`, `winner`, `plies`).
- **A neutral move offered** (V2): at a side-turn, the core's legal list holds a foreign move
  (`MOVE_FOREIGN`) from a square a piece nobody owns stands on, as `analysis._count_neutral`
  counts it. The share is of the band's side-turns.
- **A neutral move made** (V2): the record's move is such a move. The share is of side-turns.
- **A neutral at ply 40** (V2): a piece nobody owns stands on the board after ply 40, or, in a
  game shorter than 40 plies, at its end. The share is of the band's games.
- **The predictions**: pieces nobody owns taken (a move that empties a square one stood on and
  is not that piece's own move), against neutral moves made; a neutral move after which the
  side to move is in check (`Engine.in_check`: a held piece gives no check, so any check it
  leaves is discovered), per game.

The bands (`docs/archive/plan-schema-31.md` §6): *graded* is every game with no `random` seat;
*strong* both seats among the roster's top two rungs, *top3* among its top three; the rungs are
the searching players ordered by `nodes`. Per run, the stored `index-v12.json`: total, verdict,
`ladder_monotonicity` and `skill_gradient` (the instrument rule).

    uv run python tests/crowd_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, move_from

NOBODY = 2  # the third owner
PLY_MARK = 40


def nodes_of(agent: str) -> int:
    found = re.search(r"nodes=(\d+)", agent)
    return int(found.group(1)) if found else -1


def rungs(records: list[Any]) -> list[str]:
    """The searching players, weakest first."""
    names = {name for record in records for name in record.agents}
    return sorted((n for n in names if nodes_of(n) >= 0), key=nodes_of)


def nobodys_squares(engine: Engine, board: bytes) -> set[int]:
    return {sq for sq, v in enumerate(board) if v and engine.spec.owner_of(v) == NOBODY}


def read_game(engine: Engine, record: Any) -> dict[str, int]:
    state = engine.initial_state()
    out: Counter[str] = Counter()
    standing_at_mark = None
    for ply, word in enumerate(record.moves, start=1):
        before = nobodys_squares(engine, state.board)
        out["turns"] += 1
        offered = any(
            legal & MOVE_FOREIGN and move_from(legal) in before
            for legal in engine.legal_moves(state)
        )
        made = bool(word & MOVE_FOREIGN) and move_from(word) in before
        out["offered"] += int(offered)
        out["made"] += int(made)
        after = engine.apply(state, word)
        now = nobodys_squares(engine, after.board)
        # A held neutral never takes a piece nobody owns, so the count falls only by a taking.
        out["taken"] += len(before) - len(now)
        if made and engine.in_check(after):
            out["discovered_check"] += 1
        if ply == PLY_MARK:
            standing_at_mark = bool(now)
        state = after
    if standing_at_mark is None:
        standing_at_mark = bool(nobodys_squares(engine, state.board))
    out["standing_at_mark"] = int(standing_at_mark)
    return dict(out)


def read_band(engine: Engine, games: list[Any]) -> dict[str, Any]:
    n = max(1, len(games))
    decided = [r for r in games if r.winner is not None]
    total: Counter[str] = Counter()
    checked_games = 0
    for record in games:
        one = read_game(engine, record)
        total.update(one)
        checked_games += int(one.get("discovered_check", 0) > 0)
    turns = max(1, total["turns"])
    endings = Counter(
        f"{r.reason}/{'draw' if r.winner is None else 'p' + str(r.winner)}" for r in games)
    return {
        "games": len(games),
        "mean_plies": sum(r.plies for r in games) / n,
        "median_plies": statistics.median(r.plies for r in games) if games else 0,
        "min_plies": min((r.plies for r in games), default=0),
        "max_plies": max((r.plies for r in games), default=0),
        "decided": len(decided) / n,
        "decided_games": len(decided),
        "first_seat_wins": sum(r.winner == 0 for r in decided),
        "first_seat_share_of_decided": sum(r.winner == 0 for r in decided) / max(1, len(decided)),
        "turns": total["turns"],
        "offered_share_of_turns": total["offered"] / turns,
        "made_share_of_turns": total["made"] / turns,
        "standing_at_40_share_of_games": total["standing_at_mark"] / n,
        "neutral_moves_made": total["made"],
        "neutrals_taken": total["taken"],
        "discovered_checks": total["discovered_check"],
        "games_with_discovered_check": checked_games,
        "endings": dict(endings.most_common()),
    }


def read_run(run: Path) -> dict[str, Any]:
    index = json.loads((run / "index-v12.json").read_text(encoding="utf-8"))
    raw = index.get("raw", {})
    return {
        "run": run.name,
        "viable": index.get("viable"),
        "total": index.get("total"),
        "ladder_monotonicity": raw.get("ladder_monotonicity"),
        "skill_gradient": raw.get("skill_gradient"),
        "violations": index.get("violations", []),
    }


def study(spec_path: Path, runs: list[Path], pooled: Path | None = None) -> dict[str, Any]:
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.tournament.record import read_records

    engine = Engine(load_spec(spec_path))
    every = [r for run in runs for r in read_records(Path(run) / "games.jsonl")]
    graded = [r for r in every if "random" not in r.agents]
    ladder = rungs(every)
    bands = {"graded": graded}
    for name, top in (("strong", 2), ("top3", 3)):
        members = set(ladder[-top:])
        bands[name] = [r for r in graded if set(r.agents) <= members]
    out: dict[str, Any] = {
        "spec": str(spec_path),
        "runs": [str(r) for r in runs],
        "rungs": {"strong": ladder[-2:], "top3": ladder[-3:]},
    }
    for name, games in bands.items():
        out[name] = read_band(engine, games)
    out["per_run"] = [read_run(r) for r in runs if (r / "index-v12.json").exists()]
    if pooled is not None and (pooled / "index-v12.json").exists():
        index = json.loads((pooled / "index-v12.json").read_text(encoding="utf-8"))
        out["pooled"] = {k: index.get(k) for k in ("total", "viable", "violations", "interval")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--pooled", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = study(args.spec, args.runs, args.pooled)
    print(f"strong: {out['rungs']['strong']}; top3: {out['rungs']['top3']}")
    for name in ("graded", "strong", "top3"):
        b = out[name]
        print(f"== {name}: {b['games']} games, plies mean {b['mean_plies']:.1f} median "
              f"{b['median_plies']} range {b['min_plies']} to {b['max_plies']}")
        print(f"  decided {b['decided']:.4f}; first seat {b['first_seat_wins']} of "
              f"{b['decided_games']} ({b['first_seat_share_of_decided']:.4f})")
        print(f"  of {b['turns']} side-turns: neutral move offered {b['offered_share_of_turns']:.4f},"
              f" made {b['made_share_of_turns']:.4f}; a neutral at ply 40 in "
              f"{b['standing_at_40_share_of_games']:.4f} of games")
        print(f"  neutral moves made {b['neutral_moves_made']}, neutrals taken "
              f"{b['neutrals_taken']}; discovered checks {b['discovered_checks']} in "
              f"{b['games_with_discovered_check']} games")
        print(f"  endings: {b['endings']}")
    for run in out["per_run"]:
        print(f"  run {run['run']}: viable {run['viable']}, total {run['total']}, "
              f"ladder {run['ladder_monotonicity']}, gradient {run['skill_gradient']}")
    if "pooled" in out:
        print(f"  pooled: {out['pooled']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
