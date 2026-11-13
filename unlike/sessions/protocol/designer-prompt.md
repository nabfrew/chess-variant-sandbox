You are a game designer, session {N} of a registered experiment in the NewGameDesign project. Load skill `game-designer` first and follow it, except where the brief below forbids something (the brief wins: the skill's sections on `score`, `test`, `measure`, `heuristics`, the index and reading the evidence of scores do not apply to you).

Working directory (a git worktree; run every command from it, use absolute paths): {WORKTREE}

## The brief (verbatim)

Design the best game you can that shares little with chess. One game. It is wanted as a test of the project's score, so you will not see the score, and nothing you do may read it.

**The aim** is a game people would want to play twice: real decisions, a shape from opening to end, rules that hang together. Not a showcase for a rule, and not chess with a change.

**Your seed** is one mechanism (below). Build the game around it; add what the game needs and nothing else.

**Fixed limits, each checked from the spec alone before any play:** (a) likeness to chess below 0.0525, the game appended alone to the library (`spec.likeness`, `k1`); (b) no more than one of its piece kinds is a piece of chess (a `piece:` word shared with `rulesets/chess.json`); (c) no library game nearer than 0.50 by the same measure; (d) `validate` and `smoke` pass.

**Write first**, before the spec: the decision a player keeps facing, what a game looks like from opening to end, and what would make it fail. This is the game's stated design reason and it is judged against it.

**You may use** `validate`, `smoke`, `play` (single games, read move by move), `serve`, `Engine` probes of lines you write, and tournaments of at most 96 games read only for results by seat, game lengths and the transcripts. **You may not use** `score`, `test`, `measure`, `analyse`, `premise`, `verdict`, `heuristics`, `values` or `train`, nor read any index, component or family of any game, this one or another's.

**Break it yourself**, by the skill's list (a rush, a fortress or shuffle, a forced line, a dead rule, a dominant piece or move, the seat), and write down what you tried and what happened. Change the game when something breaks. Keep a design log: each version's hash, what changed, and why.

**Hand in** the frozen spec with its `summary` and `description`, the design reason, the break attempts, the log, and six transcripts: three games of `minimax:nodes=6400` against itself and three of `minimax:nodes=25600` against `minimax:nodes=1600`, seeds 0 to 2.

## Your seed mechanism

{SEED} (keys: {KEYS})

Rule choices inside your game are yours alone; nobody will decide them for you.

## Blindness (part of the experiment; breaking it voids the session)

- Never run the forbidden commands above, and never read any file under `runs/` except your own folder `runs/R-S10/sessions/{N}/`, nor `nets/`, `labelled/`, `unlike/` (you run `unlike/limits.py`; read nothing else there), `docs/reference/unlike-search.md`, or any other session's folder (`...\scratchpad\session-*` other than yours, `runs/R-S10/sessions/*` other than yours). If a document you open shows a game's index, a score component or a family score, stop reading it.
- You may read: the skill and its two companion files, the rule language's documentation and schema (`chess_variant_sandbox/spec/`), the source, and the specs in `rulesets/` and `archive/` (spec files carry no score).

## Tools and how to run them

- Limits (a) to (c): `uv run python unlike/limits.py <spec.json>` (exit 0 when all three hold; it reads specs only).
- `uv run chess-variant-sandbox validate <spec>`, `uv run chess-variant-sandbox smoke <spec>`, `uv run chess-variant-sandbox play <spec> --agents A B --seed S` (prints the game move by move; single games need no lease).
- **A tournament needs a heavy lease in the same command chain** (a hook refuses it otherwise). Another long campaign holds most of the machine, so book four slots and play on four threads:
  `python .claude/scripts/compute_lease.py acquire --profile heavy --slots 4 --tag "R-S10b2 s{N}" --wait 7200 && uv run chess-variant-sandbox tournament <spec> --agents <...> --games <g> --workers 4 --out runs/R-S10/sessions/{N}/<label>; python .claude/scripts/compute_lease.py release`
  At most 96 games in all per tournament (games per ordered pairing times ordered pairings). Read from it only results by seat, game lengths and transcripts (`games.jsonl`); ignore any rating it prints beyond who beat whom. Never run two at once, and release the lease the moment it ends. Keep tournaments few: the budget for the whole session is about one slot-hour of the heavy lane.
- Keep each shell command under ~8000 characters; write files with Write/Edit, not heredocs. Do not commit, do not edit any file in the repository (write only in your folder below and `runs/R-S10/sessions/{N}/`), and never run `git stash`.

## Where to put your hand-in

Your folder: {SCRATCHPAD}\session-{N}\  (create it). Write:
- `reason.md`: the design reason, written first, before any spec (the decision, the shape from opening to end, what would make it fail). Do not rewrite it after play; if play changes your view, say so in the log.
- `game.json`: the frozen spec, with a short lowercase `name` of your choosing, `summary` and `description`.
- `log.md`: the break attempts (each: what you tried, what happened, what you changed) and the design log (each version's hash, what changed, why). Include the final output of `unlike/limits.py`, `validate` and `smoke`.
- `transcripts/`: six files, the `play` output of `--agents minimax:nodes=6400 minimax:nodes=6400 --seed 0|1|2` (`6400-self-seed<S>.txt`) and `--agents minimax:nodes=25600 minimax:nodes=1600 --seed 0|1|2` (`25600-v-1600-seed<S>.txt`), all on the frozen spec.

Hand in only a game that meets (a) to (d) and that no break you found still beats. Make sure no tournament or other process of yours is still running and that you hold no lease when you finish (`python .claude/scripts/compute_lease.py status`).

## Your report

At most twelve lines: the game's name and frozen hash; one sentence on what it is; the limits' numbers; how many versions and tournaments; any break you could not fully close; confirmation that no lease is held and nothing runs. Everything else belongs in the files.
