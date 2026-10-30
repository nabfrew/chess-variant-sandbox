# compute-lease: light
"""The free companies, read off its runs and graded against its registration (TC10b, `docs/plan-turncoat.md` §8.5).

Written with the registration and before the first measured game. Not a test (no ``test_``
prefix). It reads persisted records only, replayed through the core by TC10's instrument
`condottieri_study.study` (hires, bribes, the first hire, changes of hands, the first briber),
reused unchanged so the two hiring showcases are read by one definition, and by TC9's
`war_chest_study.study` for the first capturer (a side read here, read for chess by the same
script on the same build). It adds the clauses §8.5 registers on top of them:

- **V2 (a), (b)**: the shares of the graded band's games holding a hire, and a bribe.
- **V2 (c), the bribe against chess's first capture**: the first briber's share of decided graded
  games holding a bribe, at most chess's first capturer's share of decided graded games
  (0.7374, `CHESS_GRADED_SNOWBALL`) plus 0.03; read only where at least 30 decided graded games
  hold a bribe, else reported unread.
- **V4 against chess**: the first seat's share of decided graded games within 0.10 of chess's
  (0.5245, 364 of 694).

It decides no move, outcome or legality itself (CLAUDE.md, rule 3).

    uv run python tests/free_companies_study.py <spec> <run-dir>... [--pooled runs/<hash>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import condottieri_study  # noqa: E402
import war_chest_study  # noqa: E402

# Chess's library runs `4c1922ffae26-{0,1000,2000,3000}`, read by these scripts on build
# `cb3da3ed95ed9358` before the first measured game (`runs/TC10b/chess-read.json`,
# `runs/TC10b/chess-war-chest-read.json`).
CHESS_GRADED_SNOWBALL = 511 / 693  # 0.7374
CHESS_FIRST_SEAT = 364 / 694  # 0.5245
SNOWBALL_MARGIN = 0.03
SEAT_MARGIN = 0.10
BRIBE_SAMPLE = 30


def grade(out: dict[str, Any]) -> dict[str, Any]:
    g = out["graded"]
    briber = g["first_briber_share_of_decided"]
    read = g["decided_with_a_bribe"] >= BRIBE_SAMPLE
    seat = g["first_seat_share_of_decided"]
    return {
        "a_hire": (g["games_with_a_hire_share"], g["games_with_a_hire_share"] >= 0.60),
        "a_bribe": (g["games_with_a_bribe_share"], g["games_with_a_bribe_share"] >= 0.20),
        "first_briber": (
            briber, g["decided_with_a_bribe"],
            (briber <= CHESS_GRADED_SNOWBALL + SNOWBALL_MARGIN) if read else "unread"),
        "plies": (g["mean_plies"], 40 <= g["mean_plies"] <= 400),
        "first_seat": (seat, abs(seat - CHESS_FIRST_SEAT) <= SEAT_MARGIN),
        "hires_per_bribe": g["hires_per_bribe"],
        "first_hire_by_ply_20": g["first_hire_by_ply_20_share"],
        "at_most_two_changes": g["at_most_two_changes_share"],
        "spent_of_earned": g["coins_spent"] / max(1, g["coins_earned"]),
    }


def study(spec: Path, runs: list[Path], pooled: Path | None = None) -> dict[str, Any]:
    out = condottieri_study.study(spec, runs, pooled)
    capture = war_chest_study.study(spec, runs)
    for band in ("graded", "strong", "top3"):
        out[band]["first_capturer_wins"] = capture[band]["first_capturer_wins"]
        out[band]["decided_with_a_capture"] = capture[band]["decided_with_a_capture"]
        out[band]["first_capturer_share_of_decided"] = capture[band]["first_capturer_share_of_decided"]
    out["grade"] = grade(out)
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
        print(f"== {name}: {b['games']} games, plies {b['mean_plies']:.1f} (median {b['median_plies']}), "
              f"decided {b['decided']:.4f}, first seat {b['first_seat_wins']}/{b['decided_games']} "
              f"({b['first_seat_share_of_decided']:.4f})")
        print(f"  hires {b['hires']}, bribes {b['bribes']}; a hire in {b['games_with_a_hire_share']:.4f}, "
              f"a bribe in {b['games_with_a_bribe_share']:.4f}; first hire by 20 "
              f"{b['first_hire_by_ply_20_share']:.4f}; at most two changes "
              f"{b['at_most_two_changes_share']:.4f}; mercenaries taken {b['mercenaries_taken']}")
        print(f"  coins earned {b['coins_earned']}, spent {b['coins_spent']}; first briber "
              f"{b['first_briber_wins']}/{b['decided_with_a_bribe']}; first capturer "
              f"{b['first_capturer_wins']}/{b['decided_with_a_capture']} "
              f"({b['first_capturer_share_of_decided']:.4f}); {b['endings']}")
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
