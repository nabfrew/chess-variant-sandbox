# compute-lease: light
"""The war chest, read off its runs (TC9, `docs/plan-turncoat.md` §8.2).

The instrument V2 to V4 and the predictions are graded on, written with the registration and
before the first measured game. Not a test (no ``test_`` prefix); it lives beside
`crowd_study.py` because the showcase's areas are the tests and the plan. It reads persisted
records only and replays them through the core, as `analyse` does; it never re-plays a game, and
it decides no move, outcome or legality itself (CLAUDE.md, rule 3). Every figure is the records'
own, or a core's:

- **Endings**, the decided share, plies and the first seat's share of decided games are the
  records' own (`reason`, `winner`, `plies`).
- **A buy** (V2): the record's move is a `MOVE_BUY`. *A buy made at ≥ 0.03 of graded
  side-turns, in ≥ 0.50 of graded games* is read as two clauses, both owed: the band's buys as
  a share of its side-turns, and the share of its games holding at least one buy. The type
  bought is the core's own: the type of the piece standing on the buy's square after it.
- **The first capture** (V2's snowball bar): the first ply after which the board holds fewer
  pieces than before it (a buy adds one, a promotion none, so only a taking lowers the count);
  its side is the player who made that ply's move (the record's plies alternate, the first
  seat moving first). The bar reads the band's decided games that hold a capture: the share
  won by the side that captured first.
- **Coins**: each side-turn's purse is the core's (`GameState.purse`); earned is a rise across
  the side's own move, spent a fall. The purse at the end and its peak are reported.
- **The predictions**: the pawns' share of all buys; the band's mean plies against chess's
  library mean (87.9 graded, `docs/plan-turncoat.md` §8.1's table).

The bands (`docs/archive/plan-schema-31.md` §6): *graded* is every game with no `random` seat;
*strong* both seats among the roster's top two rungs, *top3* among its top three; the rungs are
the searching players ordered by `nodes`. Per run, the stored `index-v12.json`: total, verdict,
`ladder_monotonicity` and `skill_gradient` (the instrument rule).

    uv run python tests/war_chest_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
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
from chess_variant_sandbox.engine.moves import MOVE_BUY, move_to, move_type

CHESS_GRADED_MEAN_PLIES = 87.9  # chess's library runs, as §8.1's table reads them


def nodes_of(agent: str) -> int:
    found = re.search(r"nodes=(\d+)", agent)
    return int(found.group(1)) if found else -1


def rungs(records: list[Any]) -> list[str]:
    """The searching players, weakest first."""
    names = {name for record in records for name in record.agents}
    return sorted((n for n in names if nodes_of(n) >= 0), key=nodes_of)


def pieces_on(board: bytes) -> int:
    return sum(1 for v in board if v)


def read_game(engine: Engine, record: Any) -> dict[str, Any]:
    spec = engine.spec
    state = engine.initial_state()
    out: Counter[str] = Counter()
    bought: Counter[str] = Counter()
    first_capture: int | None = None
    peak = max(state.purse)
    for ply, word in enumerate(record.moves, start=1):
        side = (ply - 1) % 2
        out["turns"] += 1
        after = engine.apply(state, word)
        if move_type(word) == MOVE_BUY:
            out["buys"] += 1
            value = after.board[move_to(word)]
            bought[spec.piece_ids[spec.type_of(value)]] += 1
        if first_capture is None and pieces_on(after.board) < pieces_on(state.board):
            first_capture = side
        delta = after.purse[side] - state.purse[side]
        out["earned"] += max(0, delta)
        out["spent"] += max(0, -delta)
        peak = max(peak, *after.purse)
        state = after
    return {
        "counts": dict(out),
        "bought": dict(bought),
        "first_capture": first_capture,
        "end_purse": sum(state.purse),
        "peak_purse": peak,
    }


def read_band(engine: Engine, games: list[Any]) -> dict[str, Any]:
    n = max(1, len(games))
    decided = [r for r in games if r.winner is not None]
    total: Counter[str] = Counter()
    bought: Counter[str] = Counter()
    with_buy = 0
    snowball_games = 0
    snowball_wins = 0
    end_purse = 0
    peak = 0
    for record in games:
        one = read_game(engine, record)
        total.update(one["counts"])
        bought.update(one["bought"])
        with_buy += int(one["counts"].get("buys", 0) > 0)
        end_purse += one["end_purse"]
        peak = max(peak, one["peak_purse"])
        if record.winner is not None and one["first_capture"] is not None:
            snowball_games += 1
            snowball_wins += int(record.winner == one["first_capture"])
    turns = max(1, total["turns"])
    buys = total["buys"]
    endings = Counter(
        f"{r.reason}/{'draw' if r.winner is None else 'p' + str(r.winner)}" for r in games)
    mean_plies = sum(r.plies for r in games) / n
    return {
        "games": len(games),
        "mean_plies": mean_plies,
        "plies_against_chess": mean_plies / CHESS_GRADED_MEAN_PLIES,
        "median_plies": statistics.median(r.plies for r in games) if games else 0,
        "min_plies": min((r.plies for r in games), default=0),
        "max_plies": max((r.plies for r in games), default=0),
        "decided": len(decided) / n,
        "decided_games": len(decided),
        "first_seat_wins": sum(r.winner == 0 for r in decided),
        "first_seat_share_of_decided": sum(r.winner == 0 for r in decided) / max(1, len(decided)),
        "turns": total["turns"],
        "buys": buys,
        "buy_share_of_turns": buys / turns,
        "games_with_a_buy": with_buy,
        "games_with_a_buy_share": with_buy / n,
        "bought": dict(bought.most_common()),
        "pawn_share_of_buys": bought.get("P", 0) / max(1, buys),
        "coins_earned": total["earned"],
        "coins_spent": total["spent"],
        "mean_end_purse_per_game": end_purse / n,
        "peak_purse": peak,
        "decided_with_a_capture": snowball_games,
        "first_capturer_wins": snowball_wins,
        "first_capturer_share_of_decided": snowball_wins / max(1, snowball_games),
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
        print(f"== {name}: {b['games']} games, plies mean {b['mean_plies']:.1f} "
              f"({b['plies_against_chess']:.2f} of chess) median {b['median_plies']} "
              f"range {b['min_plies']} to {b['max_plies']}")
        print(f"  decided {b['decided']:.4f}; first seat {b['first_seat_wins']} of "
              f"{b['decided_games']} ({b['first_seat_share_of_decided']:.4f})")
        print(f"  of {b['turns']} side-turns: buys {b['buys']} ({b['buy_share_of_turns']:.4f}); "
              f"a buy in {b['games_with_a_buy']} games ({b['games_with_a_buy_share']:.4f}); "
              f"bought {b['bought']}, pawns {b['pawn_share_of_buys']:.3f}")
        print(f"  first capturer won {b['first_capturer_wins']} of {b['decided_with_a_capture']} "
              f"decided ({b['first_capturer_share_of_decided']:.4f})")
        print(f"  coins earned {b['coins_earned']}, spent {b['coins_spent']}; end purse "
              f"{b['mean_end_purse_per_game']:.1f} a game (both sides), peak {b['peak_purse']}")
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
