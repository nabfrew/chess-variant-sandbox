# compute-lease: light
"""The condottieri, read off its runs (TC10, `docs/plan-turncoat.md` §8.3).

The instrument V2 to V4 and the predictions are graded on, written with the registration and
before the first measured game. Not a test (no ``test_`` prefix); it lives beside
`war_chest_study.py` because the showcase's areas are the tests and the plan. It reads persisted
records only and replays them through the core, as `analyse` does; it never re-plays a game, and
it decides no move, outcome or legality itself (CLAUDE.md, rule 3). Every figure is the records'
own, or a core's:

- **Endings**, the decided share, plies and the first seat's share of decided games are the
  records' own (`reason`, `winner`, `plies`).
- **A hire** (V2): the record's move is a `MOVE_HIRE` whose square held a piece nobody owns
  before it. **A bribe** (V2): a `MOVE_HIRE` whose square held the opponent's piece. V2's bars
  are the share of the band's games holding at least one hire, and at least one bribe.
- **The first hire by ply 20** (a prediction): the share of the band's games whose first hire is
  at ply 20 or earlier (a game with no hire is not), *most* read as more than half.
- **Bribes rarer than hires by at least two to one** (a prediction): the band's hires at least
  twice its bribes.
- **A mercenary changes hands** (a prediction): each piece nobody owns standing at the start is
  followed square to square through the replay (it moves when a move leaves its square and a
  piece of its type stands on the move's target after it; it is gone when its square holds
  anything else); each hire or bribe of it is one change of hands. A game passes when no
  mercenary changes hands more than twice; the share of the band's games that pass.
- **Side reads, unregistered**: mercenaries taken (followed pieces gone), the coins earned and
  spent (rises and falls of `GameState.purse` across the side's own move), and the first
  briber's share of decided games holding a bribe (the plan's *the richer side always wins*).

The bands (`docs/archive/plan-schema-31.md` §6): *graded* is every game with no `random` seat;
*strong* both seats among the roster's top two rungs, *top3* among its top three; the rungs are
the searching players ordered by `nodes`. Per run, the stored `index-v12.json`: total, verdict,
`ladder_monotonicity` and `skill_gradient` (the instrument rule).

    uv run python tests/condottieri_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
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
from chess_variant_sandbox.engine.moves import MOVE_HIRE, move_from, move_to, move_type

NOBODY = 2


def nodes_of(agent: str) -> int:
    found = re.search(r"nodes=(\d+)", agent)
    return int(found.group(1)) if found else -1


def rungs(records: list[Any]) -> list[str]:
    """The searching players, weakest first."""
    names = {name for record in records for name in record.agents}
    return sorted((n for n in names if nodes_of(n) >= 0), key=nodes_of)


def read_game(engine: Engine, record: Any) -> dict[str, Any]:
    spec = engine.spec
    state = engine.initial_state()
    # The mercenaries: the types standing for nobody at the start (here the only priced type).
    priced = {spec.type_of(v) for v in state.board if v and spec.owner_of(v) == NOBODY}
    followed = {sq: i for i, sq in enumerate(
        sq for sq, v in enumerate(state.board) if v and spec.type_of(v) in priced)}
    changes: Counter[int] = Counter()
    out: Counter[str] = Counter()
    first_hire: int | None = None
    first_briber: int | None = None
    for ply, word in enumerate(record.moves, start=1):
        side = (ply - 1) % 2
        out["turns"] += 1
        after = engine.apply(state, word)
        if move_type(word) == MOVE_HIRE:
            square = move_to(word)
            owner = spec.owner_of(state.board[square])
            if owner == NOBODY:
                out["hires"] += 1
                first_hire = ply if first_hire is None else first_hire
            else:
                out["bribes"] += 1
                first_briber = side if first_briber is None else first_briber
            if square in followed:
                changes[followed[square]] += 1
        else:
            moved: dict[int, int] = {}
            for square, ident in followed.items():
                value = after.board[square]
                if value and spec.type_of(value) in priced and value == state.board[square]:
                    moved[square] = ident
                    continue
                target = move_to(word)
                landed = after.board[target]
                if square == move_from(word) and landed and spec.type_of(landed) in priced:
                    moved[target] = ident
                else:
                    out["taken"] += 1
            followed = moved
        delta = after.purse[side] - state.purse[side]
        out["earned"] += max(0, delta)
        out["spent"] += max(0, -delta)
        state = after
    return {
        "counts": dict(out),
        "first_hire": first_hire,
        "first_briber": first_briber,
        "most_changes": max(changes.values(), default=0),
    }


def read_band(engine: Engine, games: list[Any]) -> dict[str, Any]:
    n = max(1, len(games))
    decided = [r for r in games if r.winner is not None]
    total: Counter[str] = Counter()
    with_hire = with_bribe = hire_by_20 = at_most_two = 0
    briber_games = briber_wins = 0
    for record in games:
        one = read_game(engine, record)
        total.update(one["counts"])
        with_hire += int(one["counts"].get("hires", 0) > 0)
        with_bribe += int(one["counts"].get("bribes", 0) > 0)
        hire_by_20 += int(one["first_hire"] is not None and one["first_hire"] <= 20)
        at_most_two += int(one["most_changes"] <= 2)
        if record.winner is not None and one["first_briber"] is not None:
            briber_games += 1
            briber_wins += int(record.winner == one["first_briber"])
    endings = Counter(
        f"{r.reason}/{'draw' if r.winner is None else 'p' + str(r.winner)}" for r in games)
    hires, bribes = total["hires"], total["bribes"]
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
        "hires": hires,
        "bribes": bribes,
        "hires_per_bribe": hires / bribes if bribes else None,
        "games_with_a_hire": with_hire,
        "games_with_a_hire_share": with_hire / n,
        "games_with_a_bribe": with_bribe,
        "games_with_a_bribe_share": with_bribe / n,
        "first_hire_by_ply_20_share": hire_by_20 / n,
        "at_most_two_changes_share": at_most_two / n,
        "mercenaries_taken": total["taken"],
        "coins_earned": total["earned"],
        "coins_spent": total["spent"],
        "decided_with_a_bribe": briber_games,
        "first_briber_wins": briber_wins,
        "first_briber_share_of_decided": briber_wins / max(1, briber_games),
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
        print(f"  hires {b['hires']}, bribes {b['bribes']} (hires per bribe {b['hires_per_bribe']}); "
              f"a hire in {b['games_with_a_hire']} ({b['games_with_a_hire_share']:.4f}), a bribe in "
              f"{b['games_with_a_bribe']} ({b['games_with_a_bribe_share']:.4f})")
        print(f"  first hire by ply 20: {b['first_hire_by_ply_20_share']:.4f}; at most two changes: "
              f"{b['at_most_two_changes_share']:.4f}; mercenaries taken {b['mercenaries_taken']}")
        print(f"  coins earned {b['coins_earned']}, spent {b['coins_spent']}; first briber won "
              f"{b['first_briber_wins']} of {b['decided_with_a_bribe']} decided")
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
