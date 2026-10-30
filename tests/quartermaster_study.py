# compute-lease: light
"""The quartermaster, read off its runs and graded against its registration (TC9d,
`docs/plan-turncoat.md` §8.7).

Written with the registration and before the first measured game. Not a test (no ``test_``
prefix). It reads persisted records only, replayed through the core by
`indemnity_study.full_study` (TC9c's instrument, itself TC9's `war_chest_study.study`, reused
unchanged, so the four purse showcases and chess are read by one definition), and grades
§8.7's clauses on its output:

- (a) buys at ≥ 0.03 of graded side-turns; (b) a buy in ≥ 0.50 of graded games;
- (c) the first capturer's share of decided graded games ≤ chess's 0.7374 + 0.03;
- (d) the same in the top three rungs ≤ chess's 0.4924 + 0.10 (chess read by the same script
  on this build, `runs/TC9d/chess-read.json`, the numbers TC9b and TC9c read);
- (e) no market type above 0.70 of graded buys (the top three rungs' split is reported, not
  barred);
- (f) buys made from behind (less declared material than the opponent) ≥ 0.50 of graded buys.

    uv run python tests/quartermaster_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from indemnity_study import full_study, market  # noqa: E402

# Chess on the same instrument, read on this build before any measured game (§8.7).
CHESS_GRADED_SNOWBALL = 511 / 693  # 0.7374
CHESS_TOP3_SNOWBALL = 65 / 132  # 0.4924
GRADED_MARGIN = 0.03
TOP3_MARGIN = 0.10
MARKET_TOP = 0.70


def grade(out: dict[str, Any]) -> dict[str, Any]:
    graded, top3 = out["graded"], out["top3"]
    m, m3 = market(graded), market(top3)
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
        "e_market_top_share": (m["top_share"], m["top_share"] <= MARKET_TOP),
        "reported_market_top_share_top3": (m3["top_share"], m3["top_share"] <= MARKET_TOP),
        "f_buys_from_behind": (graded["buys_behind_share"], graded["buys_behind_share"] >= 0.50),
        "market_shares": m["shares"],
        "market_shares_top3": m3["shares"],
        "v3_plies": (graded["mean_plies"], 40 <= graded["mean_plies"] <= 400),
        "v4_first_seat": (
            graded["first_seat_share_of_decided"],
            0.45 <= graded["first_seat_share_of_decided"] <= 0.65),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--pooled", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = full_study(args.spec, args.runs, args.pooled)
    out["grade"] = grade(out)
    for name in ("graded", "strong", "top3"):
        b = out[name]
        print(f"== {name}: {b['games']} games, plies {b['mean_plies']:.1f} "
              f"({b['plies_against_chess']:.2f} of chess, median {b['median_plies']}), decided "
              f"{b['decided']:.4f}, first seat {b['first_seat_wins']}/{b['decided_games']}; buys "
              f"{b['buys']} ({b['buy_share_of_turns']:.4f}), in {b['games_with_a_buy_share']:.4f} "
              f"of games, {b['bought']}; first capturer {b['first_capturer_wins']}/"
              f"{b['decided_with_a_capture']} ({b['first_capturer_share_of_decided']:.4f}); "
              f"paid to the loser {b['paid_to_loser']}, spent {b['coins_spent']}; buys by "
              f"standing {b['buys_by_standing']}; {b['endings']}")
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
