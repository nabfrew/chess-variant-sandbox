# compute-lease: heavy
"""Play the landing's constructed examples (`docs/archive/plan-landing.md` §2.1, LPR2; moved here by LP1).

    uv run python chess_variant_sandbox/web/stories/make_examples.py

A story may be a *constructed example* where the library's pair is unclear: a rule option
shown on a game made for the page, said to be an example, never presented as a measured
library game. Its boards must still be legal under the rule shown, so its games are played
here by the engine's own computer players (`tournament.play.play_game`, the core's minimax,
a pure function of the spec, the seed and the players) and never written by hand.
`web/stories.py` replays the moves through `Engine` on the server, exactly as it replays a
library game's record, and asks the designer's two quick answers of each spec itself
(`designing.check`, `candidacy.screen_draft`), so nothing here is quoted on the page but
the games.

Writes `examples.json` beside this file: per example the spec's file and hash, the players,
the seed, the moves as the core's move words, the ending, and the games the shown one
was picked from (so the page can say how typical it is).
"""

import json
from pathlib import Path

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.tournament.play import play_game

HERE = Path(__file__).resolve().parent
# key, spec, the two players, seed. The seeds were chosen after looking at six games of each
# (seeds 0 to 5, 2026-10-24): for the repainted ground the one whose count stalls longest
# short of the goal (five of the six ran out of moves), for ground kept for good one that
# finishes by the count (all six did). Chosen to show the rule, and the page says so.
PLAYER = "minimax:nodes=1600"
EXAMPLES = [
    ("ground-repaint", "ground-repaint.json", (PLAYER, PLAYER), 1),
    ("ground-for-good", "ground-for-good.json", (PLAYER, PLAYER), 3),
]
PLAYED = 6


def main() -> None:
    out = {}
    for key, path, agents, seed in EXAMPLES:
        spec = load_spec(HERE / path)
        engine = Engine(spec)
        played = [play_game(engine, agents, s, collect_series=False) for s in range(PLAYED)]
        record = played[seed]
        out[key] = {
            "spec": path, "hash": spec_hash(spec), "agents": list(agents), "seed": seed,
            "moves": list(record.moves), "winner": record.winner, "reason": record.reason,
            "plies": record.plies,
            "played": [{"seed": s, "plies": r.plies, "winner": r.winner, "reason": r.reason}
                       for s, r in enumerate(played)],
        }
        print(key, record.plies, record.winner, record.reason)
    (HERE / "examples.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8",
                                        newline="\n")


if __name__ == "__main__":
    main()
