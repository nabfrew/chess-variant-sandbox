"""Fill the designer and reader prompts for sessions 3 to 8 (`unlike-search.md` §9.3).

    python fill_prompts.py <worktree> <scratchpad>

Writes `<scratchpad>/prompts/designer-<n>.md` and `reader-<n>.md`. The brief inside the
designer prompt is §2.1's, verbatim; what this adds is the session's seed, the paths, the
tournament-cap arithmetic (from session 4 on) and the coordinator's folders in the blindness
lists. Stdlib only.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE, SCRATCHPAD = sys.argv[1], sys.argv[2]
OUT = Path(SCRATCHPAD) / "prompts"
OUT.mkdir(parents=True, exist_ok=True)

tpl = (HERE / "designer-prompt.md").read_text(encoding="utf-8")
tpl = tpl.replace("{WORKTREE}", WORKTREE).replace("{SCRATCHPAD}", SCRATCHPAD)
tpl = tpl.replace(
    "(`...\\scratchpad\\session-*` other than yours, ",
    "(`...\\scratchpad\\session-*` other than yours, any `read-*`, `asm-*` or `prompts` folder there, ",
)
assert "any `read-*`" in tpl
CAP = "(games per ordered pairing times ordered pairings)."
assert CAP in tpl
tpl = tpl.replace(
    CAP,
    CAP + " Ordered pairings include each agent against itself: two agents are four pairings"
    " (`--games` at most 24), three are nine (at most 10), four are sixteen (at most 6). Two"
    " earlier sessions broke this cap by forgetting the self-pairings; do the arithmetic before"
    " every tournament.",
)
seeds = {
    3: ("ground that is claimed and re-claimed, counted at the clock", "`rules.territory` `repaint`, `rules.scoring`"),
    4: ("a race that banks its arrivals, with a hand", "`rules.reach` `removes` and `count`, `rules.drops`"),
    5: ("armies that beat each other in a circle, lost by running out of a kind", "`pieces[].captures`, `rules.extinction` `per_type`"),
    6: ("a hexagonal board with hoppers and sweepers", '`board.lattice: "hex"`, `"hop"`, `"sweep"`'),
    7: ("lines of sight that freeze or command", "`pieces[].sight`"),
    8: ("a turn of two parts", "`rules.turn.parts`"),
}
for n, (seed, keys) in seeds.items():
    text = tpl.replace("{N}", str(n)).replace("{SEED}", seed).replace("{KEYS}", keys)
    (OUT / f"designer-{n}.md").write_text(text, encoding="utf-8")

rt = (HERE / "reader-prompt.md").read_text(encoding="utf-8").replace("{WORKTREE}", WORKTREE)
rt = rt.replace(
    "never open any `session-*` folder",
    "never open any `session-*`, `read-*` (other than your packet), `asm-*` or `prompts` folder",
)
assert "`asm-*`" in rt
for n, (seed, _) in seeds.items():
    (OUT / f"reader-{n}.md").write_text(rt.replace("{SEED}", seed), encoding="utf-8")
print(OUT)
