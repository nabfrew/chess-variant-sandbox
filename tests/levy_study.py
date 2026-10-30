# compute-lease: light
"""The levy, read off its runs and graded against its registration (TC9b, `docs/plan-turncoat.md` §8.4).

Written with the registration and before the first measured game. Not a test (no ``test_``
prefix). It reads persisted records only, replayed through the core by
`war_chest_study.study` (TC9's instrument, reused unchanged so the two purse showcases and
chess are read by one definition), and adds the clauses §8.4 registers on top of it:

- **The market**: the most-bought type's share of all the band's buys (V2, ≤ 0.70), and every
  type's share.
- **The snowball, against chess**: the first capturer's share of decided games holding a
  capture, in the graded band (≤ chess's 0.7374 + 0.03) and in the top three rungs (≤ chess's
  0.4924 + 0.10). Chess's numbers are its library runs read by the same script on the build
  the registration was written on (`runs/TC9b/chess-read.json`).

It decides no move, outcome or legality itself (CLAUDE.md, rule 3).

    uv run python tests/levy_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from war_chest_study import study  # noqa: E402

CHESS_GRADED_SNOWBALL = 511 / 693  # 0.7374, chess's library runs, graded
CHESS_TOP3_SNOWBALL = 65 / 132  # 0.4924, the top three rungs
GRADED_MARGIN = 0.03
TOP3_MARGIN = 0.10


def market(band: dict[str, Any]) -> dict[str, Any]:
    buys = max(1, band["buys"])
    shares = {k: v / buys for k, v in band["bought"].items()}
    return {"shares": shares, "top_share": max(shares.values(), default=0.0)}


def grade(out: dict[str, Any]) -> dict[str, Any]:
    graded, top3 = out["graded"], out["top3"]
    m = market(graded)
    return {
        "buys_of_side_turns": (graded["buy_share_of_turns"], graded["buy_share_of_turns"] >= 0.03),
        "games_with_a_buy": (graded["games_with_a_buy_share"], graded["games_with_a_buy_share"] >= 0.50),
        "snowball_graded": (
            graded["first_capturer_share_of_decided"],
            graded["first_capturer_share_of_decided"] <= CHESS_GRADED_SNOWBALL + GRADED_MARGIN),
        "snowball_top3": (
            top3["first_capturer_share_of_decided"],
            top3["first_capturer_share_of_decided"] <= CHESS_TOP3_SNOWBALL + TOP3_MARGIN),
        "market_top_share": (m["top_share"], m["top_share"] <= 0.70),
        "market_shares": m["shares"],
        "plies": (graded["mean_plies"], 40 <= graded["mean_plies"] <= 400),
        "first_seat": (
            graded["first_seat_share_of_decided"],
            0.45 <= graded["first_seat_share_of_decided"] <= 0.65),
        "strong_market_shares": market(out["strong"])["shares"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--pooled", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = study(args.spec, args.runs, args.pooled)
    out["grade"] = grade(out)
    for name in ("graded", "strong", "top3"):
        b = out[name]
        print(f"== {name}: {b['games']} games, plies {b['mean_plies']:.1f} "
              f"({b['plies_against_chess']:.2f} of chess), decided {b['decided']:.4f}, first seat "
              f"{b['first_seat_wins']}/{b['decided_games']}; buys {b['buys']} "
              f"({b['buy_share_of_turns']:.4f}), in {b['games_with_a_buy_share']:.4f} of games, "
              f"{b['bought']}; first capturer {b['first_capturer_wins']}/"
              f"{b['decided_with_a_capture']} ({b['first_capturer_share_of_decided']:.4f}); "
              f"coins {b['coins_earned']} earned, {b['coins_spent']} spent; {b['endings']}")
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
