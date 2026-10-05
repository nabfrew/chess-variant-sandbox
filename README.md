# Chess variant sandbox

**Chess variant sandbox is a place to find chess-like games worth playing.** People invent games in a designer that
needs no programming, and anyone can play them in the browser, against the computer or against each other. What sets
it apart is that it measures them: the computer plays every game against itself hundreds of times, at several
strengths, and scores how interesting the play is — whether skill pays, whether the first player has it easy, whether
games go somewhere or stall. That score is an estimate, not a verdict: it says what it rests on and how far to trust
it, and it is measured again as the computer players get stronger.

So if you have invented a chess-like game, this is where to find out whether it is any good: build it, play it, and
read what the measurement says about it and how much weight that reading can bear.

Everything runs on your own machine: `uv run chess-variant-sandbox serve --open` (see *Quick start*).

## How games are made

Games are data, written in a small declarative rule language (*What the rule language can say*, below). They come
from three places:

- **People**, in the designer: stand a piece, click where it may go, add rule cards, play the draft, keep it.
- **Known variants**, written as specs. `rulesets/chess.json` is the measurement's calibration baseline, not the
  goal; `rulesets/dull.json` anchors the low end.
- **The design loop**, which is where the project began. A language model proposes rules, cheap gates throw out the
  obviously broken, the engine plays a tournament under them, a set of metrics scores how interesting the result
  was, a second model gives an independent qualitative read, and the loop feeds all of that back into the next
  proposal.

Whichever way a game is made, it is measured the same way.

The rules, the scoring and the orchestration are Python. The part that has to run millions of positions — move
generation, search, the agents, whole tournaments — is a Rust extension module built from `rust/`.

## How it fits together

| | what it does |
|---|---|
| **Rule language** | Games are declarative JSON, checked by a validator that rejects unknown keys rather than ignoring them. A spec declares which schema version it uses, and every version ever cut is still accepted. |
| **Engine** | Interprets a compiled spec. A Rust core behind a Python API, pinned by published perft counts — chess's own, and Fairy-Stockfish's for the variants in the corpus. |
| **Agents** | `random`, `greedy`, `minimax`, `mcts`, a learned evaluator, and Fairy-Stockfish as an optional outside opponent. All behind one protocol, all budgeted in nodes rather than depth, so the same option means the same work in any game. |
| **Tournaments** | Threaded round-robin. Colours are played both ways, self-pairings are included on purpose (the one matchup where strength cancels exactly, which is how first-player advantage is read), and ratings come from a Bradley-Terry maximum-likelihood fit. |
| **Interestingness index** | Measurements in six families — balance, pacing, variety, drama, decisions, depth — combined geometrically under versioned weights, behind hard gates that can call a game not viable outright. |
| **Design loop** | Propose, gate, simulate, analyse, score, judge, archive, revise. Resumable, with full lineage; everything tried is kept, failures included. |
| **Local app and site** | `chess-variant-sandbox serve`: browse the library, play any game against the same agents the index rates, build a new one, and start a design run. The website is the same pages with accounts and ratings. |

## Quick start

You need [uv](https://docs.astral.sh/uv/) and a Rust toolchain ([rustup](https://rustup.rs)).

```bash
uv sync --extra dev
```

That builds the Rust core in release mode; it takes about a minute the first time. Add `--extra llm` for the stages
that call a language model (the design loop and the judge); everything else runs without an API key.

Check a ruleset compiles, and see what the engine makes of it:

```bash
uv run chess-variant-sandbox validate rulesets/chess.json
```

Play a game and watch it:

```bash
uv run chess-variant-sandbox play rulesets/chess.json --agents minimax:nodes=2000 greedy
```

Run a tournament, replay it for the second-pass measurements, then score it:

```bash
uv run chess-variant-sandbox tournament rulesets/chess.json --agents random greedy minimax:nodes=400 minimax:nodes=1600 --games 4 --workers 8 --out runs/demo
uv run chess-variant-sandbox analyse runs/demo
uv run chess-variant-sandbox score runs/demo
```

`analyse` is not optional: several components of the index come from that engine-derived second pass, and `score`
refuses by name rather than quietly scoring without them.

Render a game as a standalone HTML replay, or open the whole library and play one yourself:

```bash
uv run chess-variant-sandbox viz runs/demo --game 0
uv run chess-variant-sandbox serve --open
```

Run the design loop (spends money: needs `ANTHROPIC_API_KEY`):

```bash
uv run chess-variant-sandbox evolve --generations 8 --workers 8
```

`uv run chess-variant-sandbox --help` lists the rest — calibration, the Fairy-Stockfish cross-check, the heuristic
and piece-value fits, and the staleness check.

**After changing `rust/`**, a plain `uv sync` does not rebuild the core. Rebuild with
`uv sync --extra dev --reinstall-package chess-variant-sandbox`.

## Tests

```bash
uv run pytest -m "not slow" -n 4      # the fast suite
uv run pytest                         # everything, including the slow tests
uv run ruff check . && uv run mypy chess_variant_sandbox
cargo test --manifest-path rust/Cargo.toml
```

A handful of tests skip without things this repository does not ship: a Fairy-Stockfish binary, the `anthropic`
package and an API key, a PostgreSQL database for the site's tests (`CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL`), or
locally generated measurements under `runs/`.

## What the rule language can say

Rules are JSON. A piece is a list of movement descriptors built from four primitives:

- **slide** — repeat a step until blocked (rook, bishop, queen)
- **leap** — jump straight to the target, ignoring anything between (knight, king)
- **lame** — a leap that a piece in the way can block (the xiangqi horse and elephant)
- **hop** — jump exactly one screening piece, then continue (the xiangqi cannon)

`symmetry` expands base vectors so you do not list them all: `[[1,2]]` with `dihedral8` is the knight's eight moves.
`mode` separates moving from capturing, which is what makes a pawn a pawn. Vectors are written from the first
player's view; the engine mirrors them.

```json
{
  "id": "N", "name": "knight", "value": 3,
  "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]
}
```

Above the pieces sit the rules: how the game ends and who wins, promotion and castling, captured pieces returning
from a hand, board regions and obstacles, capture side effects, scoring and territory, boards whose opposite edges
are the same edge, pieces that face a direction, and pieces belonging to neither player. The engine reproduces, move
for move against an independent implementation, xiangqi, shogi, crazyhouse, atomic, antichess, horde, racing kings,
king of the hill, three-check, shatranj, makruk, Capablanca, Los Alamos and the rest of what is in `rulesets/`.

When the language cannot express a rule, the fix is a new primitive and a schema version bump, never an escape
hatch. The bumps accumulate rather than replace: a spec's declared version is part of its content hash, so the
version a spec declares is an honest statement of what it needs. `SCHEMA_VERSION` in
[`chess_variant_sandbox/spec/schema.py`](chess_variant_sandbox/spec/schema.py) says how many there have been.

`uv run chess-variant-sandbox validate` reports every problem at once, and `uv run chess-variant-sandbox smoke` tells
you within a couple of seconds whether the game is playable at all.

## What the index says, and what it does not

Two rulesets anchor the scale. `chess.json` is the calibration baseline, deliberately not near 1.0 — a baseline that
maxes the scale leaves the search nothing to aim at. `dull.json` is a deliberately shallow control, so a metric that
cannot tell the two apart is visibly not measuring anything.

The honest headline: **the index separates a game from a non-game far better than it ranks games.** Most adjacent
pairs on the library's leaderboard have confidence intervals that overlap.

**No index value is quoted here, on purpose.** A score carries the basis it was taken under — spec hash, weights
version, agent roster, seed, and fingerprints of the engine and metrics sources — and a stored score whose basis no
longer matches is *stale* rather than a number. Weights are versioned (`chess_variant_sandbox/metrics/weights/`),
and scores under different versions never share a leaderboard. `uv run chess-variant-sandbox measure --check` says
what is out of date and why.

### The one that earns its keep

`rulesets/bastion.json` was written by hand as a plausible-looking siege game. It validates, it passes the smoke
gate, and it looks fine:

```bash
uv run chess-variant-sandbox play rulesets/bastion.json --agents minimax:nodes=1500 greedy --seed 3
```

Three plies. A cannon fires down the file using the defender's own pawn as its screen, against a king smothered by
its own army. For several weights versions the index read it as viable, because its outcome gates pooled games
across opponents of every strength: against weak ones it is a quick kill, against strong ones a dead shuffle, and
each failure hid the other. Reading those gates on the weaker half of the ladder fixed it; the reasoning is written
into [`chess_variant_sandbox/metrics/weights/v10.json`](chess_variant_sandbox/metrics/weights/v10.json) beside the
weights it justifies. It is kept as a control because of it.

## Where a game lives

- **`runs/`** — every candidate a tournament or a design loop produced. Generated, and ignored by git.
- **`archive/`** — games worth keeping: committed, playable, and read by nothing in `metrics/` or `tests/`. See
  [`archive/README.md`](archive/README.md).
- **`rulesets/`** — the corpus. Every file here is part of the human band the index is calibrated against and is held
  to the cheap gates by the test suite.

`chess-variant-sandbox archive`, `test` and `promote` are the steps between them. A spec's identity is its mechanics
hash, so an in-place edit is a different game, not a rename.

## Design notes

**Rules are data, never code.** Nothing generated is ever executed — no `exec`, no plugins. That is what makes a spec
the whole game: its hash names it, the gates read it and Fairy-Stockfish re-expresses it, which only works if rules
cannot hide in code.

**The proposing model never sees the index.** It gets prose observations from real games, never component names,
scores or weights: shown the objective function, a model optimises the proxy. The judge is kept ignorant of the
numbers for the same reason.

**Python may hold a position and ask the engine questions; it may not answer them.** Any Python that works out a legal
move, an outcome or a move choice is either a declared test oracle or a bug. The command line, the visualiser and the
web app all drive the core one move at a time through the same `Engine`, so the opponent you play in the browser is
the one whose rating the index reads.

**Threads never change an answer.** A game is a pure function of `(ruleset hash, seed, agent configs)`, each seat
draws from its own random stream, and every seed is fixed before any thread starts, so the results are identical at
any worker count.

Measure the engine on your machine:

```bash
uv run chess-variant-sandbox bench rulesets/chess.json --depth 5 --threads 1
```

## Licence

Copyright (C) 2026 the Chess variant sandbox authors.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License, version 3, as published by the Free Software Foundation. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See [`LICENSE`](LICENSE) for the full text.

**Except where noted:** the piece drawings, the typefaces and the vendored JavaScript keep their own licences.
[`NOTICE`](NOTICE) lists them.

Fairy-Stockfish is not included. The optional Fairy-Stockfish opponent runs a program you install yourself, as a
separate process, and the `pyffish` test dependency is installed from PyPI.

## Contributing

This repository is a published copy of a project developed elsewhere, refreshed from time to time as one commit.
**Outside contributions are not accepted for now**, so please do not open pull requests. Bug reports and questions
are welcome through the issue tracker or at [github.com/nabfrew](https://github.com/nabfrew).
