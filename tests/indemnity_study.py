# compute-lease: light
"""The indemnity, read off its runs and graded against its registration (TC9c,
`docs/plan-turncoat.md` §8.6).

Written with the registration and before the first measured game. Not a test (no ``test_``
prefix). It reads persisted records only, replayed through the core by
`war_chest_study.study` (TC9's instrument, reused unchanged, so the three purse showcases and
chess are read by one definition), and adds what §8.6 registers on top of it:

- **The market**: the most-bought type's share of all the band's buys, and every type's share
  (as `levy_study`).
- **The snowball, against chess**: the first capturer's share of decided games holding a
  capture, graded (≤ chess's 0.7374 + 0.03) and in the top three rungs (≤ chess's 0.4924 +
  0.10); chess's numbers are its library runs read by the same script on this build
  (`runs/TC9c/chess-read.json`).
- **Coins to the loser**: under `bounty_to: "loser"` a capture fills the purse of the side
  that did not move. ``paid_to_loser`` sums every rise in the non-mover's purse across a
  ply (the core's `GameState.purse`); TC9's ``coins_earned`` (the mover's own rise) reads 0.
- **Who buys** (a prediction): at each buy, the buyer's declared material on the board
  (non-royal pieces, the spec's own ``value``) against the opponent's, before the buy:
  behind, level or ahead. Python reads a board here; it decides nothing (CLAUDE.md, rule 3).

    uv run python tests/indemnity_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from war_chest_study import rungs, study  # noqa: E402

from chess_variant_sandbox.engine import Engine  # noqa: E402
from chess_variant_sandbox.engine.moves import MOVE_BUY, move_type  # noqa: E402

# Chess on the same instrument, read on this build before any measured game (§8.6).
CHESS_GRADED_SNOWBALL = 511 / 693  # 0.7374
CHESS_TOP3_SNOWBALL = 65 / 132  # 0.4924
GRADED_MARGIN = 0.03
TOP3_MARGIN = 0.10


def _material(engine: Engine, board: bytes, values: dict[str, float], player: int) -> float:
    spec = engine.spec
    return sum(
        values.get(spec.piece_ids[spec.type_of(v)], 0.0) for v in board
        if v and spec.owner_of(v) == player
    )


def read_coins(engine: Engine, games: list[Any], values: dict[str, float]) -> dict[str, Any]:
    paid = 0
    games_paid = 0
    who: Counter[str] = Counter()
    for record in games:
        state = engine.initial_state()
        one = 0
        for ply, word in enumerate(record.moves, start=1):
            side = (ply - 1) % 2
            after = engine.apply(state, word)
            one += max(0, after.purse[1 - side] - state.purse[1 - side])
            if move_type(word) == MOVE_BUY:
                mine = _material(engine, state.board, values, side)
                theirs = _material(engine, state.board, values, 1 - side)
                who["behind" if mine < theirs else "level" if mine == theirs else "ahead"] += 1
            state = after
        paid += one
        games_paid += int(one > 0)
    buys = max(1, sum(who.values()))
    return {
        "paid_to_loser": paid,
        "games_with_a_payment_share": games_paid / max(1, len(games)),
        "buys_by_standing": dict(who),
        "buys_behind_share": who["behind"] / buys,
    }


def market(band: dict[str, Any]) -> dict[str, Any]:
    buys = max(1, band["buys"])
    shares = {k: v / buys for k, v in band["bought"].items()}
    return {"shares": shares, "top_share": max(shares.values(), default=0.0)}


def grade(out: dict[str, Any]) -> dict[str, Any]:
    graded, top3 = out["graded"], out["top3"]
    m = market(graded)
    return {
        "a_buys_of_side_turns": (
            graded["buy_share_of_turns"], graded["buy_share_of_turns"] >= 0.03),
        "b_games_with_a_buy": (
            graded["games_with_a_buy_share"], graded["games_with_a_buy_share"] >= 0.50),
        "c_snowball_graded": (
            graded["first_capturer_share_of_decided"],
            graded["first_capturer_share_of_decided"] <= CHESS_GRADED_SNOWBALL + GRADED_MARGIN),
        "d_snowball_top3": (
            top3["first_capturer_share_of_decided"],
            top3["first_capturer_share_of_decided"] <= CHESS_TOP3_SNOWBALL + TOP3_MARGIN),
        "e_market_top_share": (m["top_share"], m["top_share"] <= 0.70),
        "market_shares": m["shares"],
        "v3_plies": (graded["mean_plies"], 40 <= graded["mean_plies"] <= 400),
        "v4_first_seat": (
            graded["first_seat_share_of_decided"],
            0.45 <= graded["first_seat_share_of_decided"] <= 0.65),
    }


def full_study(spec_path: Path, runs: list[Path], pooled: Path | None = None) -> dict[str, Any]:
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.tournament.record import read_records

    out = study(spec_path, runs, pooled)
    spec = load_spec(spec_path)
    engine = Engine(spec)
    values = {
        p["id"]: float(p.get("value", 0)) for p in spec["pieces"] if not p.get("royal")
    }
    every = [r for run in runs for r in read_records(Path(run) / "games.jsonl")]
    graded = [r for r in every if "random" not in r.agents]
    ladder = rungs(every)
    bands = {"graded": graded}
    for name, top in (("strong", 2), ("top3", 3)):
        members = set(ladder[-top:])
        bands[name] = [r for r in graded if set(r.agents) <= members]
    for name, games in bands.items():
        out[name].update(read_coins(engine, games, values))
    out["grade"] = grade(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--pooled", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = full_study(args.spec, args.runs, args.pooled)
    for name in ("graded", "strong", "top3"):
        b = out[name]
        print(f"== {name}: {b['games']} games, plies {b['mean_plies']:.1f} "
              f"({b['plies_against_chess']:.2f} of chess, median {b['median_plies']}, "
              f"{b['min_plies']} to {b['max_plies']}), decided {b['decided']:.4f}, first seat "
              f"{b['first_seat_wins']}/{b['decided_games']}; buys {b['buys']} "
              f"({b['buy_share_of_turns']:.4f}), in {b['games_with_a_buy_share']:.4f} of games, "
              f"{b['bought']}; first capturer {b['first_capturer_wins']}/"
              f"{b['decided_with_a_capture']} ({b['first_capturer_share_of_decided']:.4f}); "
              f"paid to the loser {b['paid_to_loser']} (in {b['games_with_a_payment_share']:.3f} "
              f"of games), spent {b['coins_spent']}; buys by standing {b['buys_by_standing']}; "
              f"{b['endings']}")
    for run in out["per_run"]:
        print(f"  run {run['run']}: viable {run['viable']}, total {run['total']}, "
              f"ladder {run['ladder_monotonicity']}, gradient {run['skill_gradient']}")
    if "pooled" in out:
        print(f"  pooled: {out['pooled']}")
    for key, value in out["grade"].items():
        print(f"  {key}: {value}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
