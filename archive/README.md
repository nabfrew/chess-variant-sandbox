# Games worth keeping, that are not part of the corpus

The middle rung of the pathway out of a design run. A candidate lives under `runs/`,
which git ignores; a game in here is committed, playable, and linked to — and is **not**
part of what the index is calibrated against.

That distinction is the reason this directory exists rather than a `keep` flag on
`rulesets/`. Every file in `rulesets/` is read by `metrics/calibration.py` as the human
band, held to the cheap gates by `tests/test_index_quality.py`, and pinned by frozen
perft ladders — so a game admitted there changes what the instruments say about every
other game. Nothing in `metrics/` or `tests/` reads this directory.

What that buys is a place to put a game whose tournament nobody wants to pay for yet,
and a game that is good and simply does not pin anything down. Archiving asks for no
measurement and no argument; it says only that the game was worth inventing.

```bash
uv run chess-variant-sandbox archive <hash> --name "Longshot" --note "why it is worth keeping"
uv run chess-variant-sandbox test <hash>        # then, if it is worth measuring properly
uv run chess-variant-sandbox promote <hash> --reason "what it pins down that nothing else does"
```

Or from the app — `uv run chess-variant-sandbox serve --open`, then the *What becomes of it* panel on
any game's page.

Files here carry the same `provenance` block a promoted ruleset gets, filled from the
design run's own lineage rather than from the model that wrote the spec: a model asked
to describe its own ancestry would invent one.

## What is in here

`cannonade.json`, `reckoning.json` and `marchfield.json` — the first three rulesets this
project's design loop proposed against a live model, from two runs seeded on `gardner`.
Two of the three beat that seed by more than the noise in both. They are archived rather
than promoted on purpose: promotion is a claim that a game pins down something no other
corpus entry does, and "the first thing the loop proposed" is a fact about this project's
history rather than about the game.

The runs that made them are gone — `runs/` is gitignored, and their transcripts are
reproducible from `(ruleset_hash, seed, agent_configs)` anyway. The rulesets are not:
the same prompt gives a different game. What they measured is written up in
`docs/reference/first-live-runs.md`.

`quarry.json` and four `hierarchy-*.json` -- schema 6's showcase games, written by hand
rather than found by the loop. They are here rather than in `rulesets/` because a game
built to exercise a rule is not evidence about what the index measures; it is evidence
about the rule.

`quarry` is chess plus four walls and nothing else, which is what isolates the obstacle
rule from the strike rule. The other four are a 2x2 over the strike rule's two axes --
which way the chain points, and where the two pieces end up:

| index `v9`, chess 0.8350 | `position: stay` | `position: swap` |
|---|---|---|
| `effect: demote` | `hierarchy-fall` 0.7287 | **`hierarchy-fall-swap` 0.8157** |
| `effect: promote` | `hierarchy-rise-stay` 0.6650 | `hierarchy-rise` 0.7273 |

The plan asked for the diagonal only. `hierarchy-fall-swap` and `hierarchy-rise-stay`
were added after the first pair measured and before anything was concluded from it,
because a pair that differs in two things at once cannot attribute either result -- and
the finding they produced, that the position rule is as large a lever as the chain's
direction, was reachable no other way. What they all measured is written up in
`docs/reference/schema-6.md`.

`hierarchy-extended.json` -- `hierarchy-rise` with the chain continued three rungs past
the queen, `Q->A->D->Z` (queen plus knight, then leaping diagonals, then leaping
orthogonals as well), and nothing else changed; row 3 of
`docs/game-ideas.md`. Measured 2026-09-06 under `v9`:
**0.628 +/- 0.041, not viable** -- `quick_kill_rate` 0.597 against the gate's 0.5,
`mean_plies` 19.7, every self-play game from 800 nodes up over by ply 7. Under `swap` a
struck queen now climbs to an amazon on the striker's square, so a queen beside the enemy
king cannot be taken from close range without planting an amazon in check of one's own
king, and a queen offered deep in the enemy camp comes back as an amazon on the captor's
square. `D` moved seven times in 256 games and `Z` never. The prediction, its grading and
the lines are in that file's note 8.

`weathervane.json` and `weathervane-torus.json` -- schema 8's showcase and its wrapped
twin, and the cleanest controls this project has. Each is `rulesets/sardines.json` with
directional pawns, the torus one with `rules.wrap` as well, and **nothing else changed**:
same 8x8 board, same 32-piece army, same rules, every square occupied from the start. Both
are archived rather than promoted for the reason `quarry` is -- a game built to exercise a
rule is evidence about the rule.

**Both were rebuilt on 2026-09-11 (Q180) as two cells of the sardines family**, four games
from one generator covering wrap x facing:
`docs/reference/sardines-family-generator.py`,
seeds 20261254 and 20262067 of a campaign pre-registered in
`sardines-family.md`. The equalities above hold by
construction now rather than by a test noticing afterwards, which is what
`tests/test_schema_eight.py::TestTheShowcase` reads off the specs. `weathervane-torus`
replaced `weathervane-chaos-torus.json` (`a6ebba755d77`, 0.8557), and `runs/a6ebba755d77-0`
stays as that game's measurement.

| index `v10`, canonical seed | sardines `9f538ab85a77` | weathervane `1442513d38d7` | weathervane torus `9777cd6c9ed1` |
|---|---|---|---|
| index | 0.8069 +/- 0.0155 | **0.7240 +/- 0.0196** | **0.8292 +/- 0.0158** |

`weathervane-half-turn.json` -- `8fc3939243d5`, Q28b, 2026-09-13. `weathervane` with a
two-axis pawn (`facing: {"axes": 2, "initial": "backward"}`), the only spec that compiles
one. Its 27 sideways pawns were each turned a quarter turn anticlockwise as drawn, and the
25 backward pawns name no facing and take it from `initial`. It closes rows 15 and 16 of
`schema-showcase-coverage.md`. Measured
**0.6933 +/- 0.0164** under `v10`, below its pre-registered 0.72--0.81: branching fell as
predicted (59.5 to 38.8), but games halved in length and `quick_kill_rate_worst_half` rose
to 0.42, so the position it inherits plays faster and more lopsided than the parent's.

The schema-8 campaign that first measured a facing pawn ran against the superseded specs
(`b88e98dab286` at 0.761 +/- 0.015 against sardines' 0.837 +/- 0.025 under `v9`, with
`mean_branching` 20.77 -> 57.15, `piece_utilisation` 0.845 -> 0.629 and `quick_kill_rate`
0.031 -> 0.133). Two of its four pre-registered predictions missed, and the write-up is
`docs/reference/schema-8.md`.

**`boar.json` and its three arms were removed on 2026-09-07 (Q67).** Schema 9's showcase was
chess plus two **boars** — pieces neither player owns, standing on c3 and c6 from ply zero,
walking `c3-c4-d4-d3` and its mirror after every player move and eating what they walk into —
with `boar-tame` (`captures: []`), `boar-lone` (`mirror: false`) and `boar-bounty`
(`bounty` 6). The user played one and found what the campaign had not thought to measure: the
boars are taken in the opening. White's first move takes one in 232 of 256 games, the median
game is clear of both by ply 4, and a boar stands for 7.4 plies of 86.7. The index the
campaign published, 0.7949 against chess's 0.8350, is a fair score for chess with a forced
pawn capture on move one and not a measurement of a wild piece, so the four games were
retired rather than kept as an example of the rule.

The numbers, the deltas needed to rebuild any of the four, and what the campaign does and
does not support are in `docs/reference/schema-9.md`; the
extraction is `docs/reference/boar-survival.py`.

## `wolves.json` — the wild piece that stays in the game

`ceb4916bdabc`, Q68, 2026-09-07. Chess plus **four wolves**: wild pieces on two five-square
patrols, `a4 b5 c4 b3 a3` and `h4 g5 f4 g3 h3`, both `mirror: true`, so each player has one
pacing their own half of each wing. **Five is odd**, which is what Q67 said a mirrored pair
needs. There is **no bounty and no scoring rule** — a wolf is a hazard, not a prize — and
`captures` lists on `P`, `B`, `R`, `Q` and `K` mean **only a knight may bring one down**.

Built against a criterion fixed before it existed
(`docs/reference/wild-piece-that-matters.md`),
because the boars above failed one nobody had written. On its canonical-roster measurement a
wolf is on the board for **0.983 of all plies**, **253 of 256 games end with one standing**,
**no wolf is taken before ply 3**, and the walk eats **5.8 player pieces a game** — against
the boar's 0.085, its ply-1 capture in 232 of 256 games, and 7.4 plies of 86.7.

| | chess | `wolves` |
|---|---|---|
| index `v9` | 0.8350 +/- 0.0181 | **0.7916 +/- 0.0150** — intervals overlap |
| `quick_kill_rate` | 0.0357 | **0.0000** |
| `mean_plies` | 85.9 | 102.2 |
| `variety` / `pacing` | 0.8221 / 0.8551 | **0.9047 / 0.8751** |
| `elegance` | 0.4195 | **0.2875** (rule mass 81 against 52) |
| `first_player_advantage` | 0.0625 | 0.1562 |
| the fit | accepted 0.5417 | refused 0.1667 |

Three of six predictions hit, one missed clean and two half. It is archived rather than
promoted: it is chess with four hazards eating a quarter of the material, and what it pins
down that nothing else does is a **live third actor**.

## `wolves-bounty.json` — the same wolves, with a price on them

`54b38fb8cba8`, Q76, 2026-09-13. `wolves` plus `"scoring": {"count": "points"}` and
`bounty: 1` on both routes, and nothing else. Taking a wolf pays one point, and a game that
ends on the no-progress clock or the ply limit goes to the player with more points. It is
the one-key experiment on what a bounty does, measured against `wolves` on the same build.

| | `wolves` | `wolves-bounty` |
|---|---|---|
| index `v10` | 0.7548 +/- 0.019 | **0.7483 +/- 0.016** — intervals overlap |
| wolves taken, of 1,024 | 216 | **717** |
| games ending with a wolf standing | 253 | **156** |
| first capture of a game, median ply | 22 | **8** |
| games decided `scored` | 0 | 11 |
| `first_player_advantage` | 0.1562 | 0.0625 |

The prize is what kills a wild piece, and the pull is small. The built-in searcher pays
about 2 to 4 pawns for the first wolf at the plies where it was taken. **The bounty's
number changes no move**: every evaluator divides the tally by a sum that scales with it,
so 1 plays exactly as 3 or 6 would. Five of six pre-registered predictions hit. The miss
is the first-player advantage, which fell instead of rising. The account is
`docs/reference/wild-piece-that-matters.md`
§*Q76*.

## `wolves-rotated.json` — the same wolves, one step later

`213f7b6604e4`, Q77, 2026-09-14. `wolves` with both routes rotated by one square, so every
wolf starts one step further along its patrol. Nothing else changes. It asks whether
`wolves`' first-player edge comes from the wolves' timing.

| | `wolves` | `wolves-rotated` |
|---|---|---|
| index `v10` | 0.7548 +/- 0.019 | 0.7613 +/- 0.021 |
| `first_player_advantage`, 32 self-play games | 0.1562 | 0.0938 |
| first seat's score over all 256 games, 95% | 0.500 [0.482, 0.518] | 0.527 [0.506, 0.551] |

The 0.1562 was noise. `first_player_advantage` rests on 32 games and its interval is about
0.37 wide. Over all 256 games the rotated arm shows a small first-mover edge the control
lacks, and the pre-registered rule reads that as a phase effect, by a margin of 0.006. The
account is
`docs/reference/wild-piece-that-matters.md`
§*Q77*.

## `wolves-rotated-2.json`, `-3.json`, `-4.json` — the other three phases

Q222, 2026-09-14. `wolves` with both routes rotated by two, three and four squares. Both
routes have five squares, so with `wolves` and `wolves-rotated` these five specs are every
joint alignment the patrols allow. Nothing else changes.

| rotation | spec | index `v10` | first seat's score over 256 games, cluster 95% |
|---|---|---|---|
| 0 | `wolves` `ceb4916bdabc` | 0.7548 +/- 0.019 | 0.500 [0.472, 0.532] |
| 1 | `wolves-rotated` `213f7b6604e4` | 0.7613 +/- 0.021 | 0.527 [0.500, 0.557] |
| 2 | `wolves-rotated-2` `8ec0ce4678e9` | 0.7626 +/- 0.022 | 0.518 [0.496, 0.542] |
| 3 | `wolves-rotated-3` `2e94e1688601` | 0.7812 +/- 0.019 | 0.502 [0.484, 0.522] |
| 4 | `wolves-rotated-4` `87a488975531` | 0.7346 +/- 0.016 | 0.481 [0.450, 0.510] |

The phase does not move the first seat's edge by any amount these runs can see. The three
new arms pooled score exactly 0.5000, and the pre-registered rule reads noise. Q77's small
edge was a draw at the edge of its interval. The account is
`docs/reference/wild-piece-that-matters.md`
§*Q222*.

## `hex-glinski-promoting.json` — schema 14, hexagonal chess

`279f92a97530`, Q229, 2026-09-15. Glinski's 91-cell hexagon and army on an 11x11 axial
rhombus, `board.lattice: "hex"` declared and thirty walls cutting the rhombus to the
hexagon. Six-neighbour rooks, six-diagonal bishops and twelve-leap knights, all spelled
`symmetry: none`. The second army is rotated, not mirrored. A pawn promotes on the far
edge, the eleven cells with no forward step, which `rules.promotion.squares` states as a
square list. It is schema 14's showcase for both of that bump's features and the only spec
in the tree that compiles a promotion zone that is not whole ranks.

It scores **0.8020 +/- 0.0229 under `v10`, viable**, against chess's 0.7943, with a
`skill_gradient` of 176.55 and a monotone ladder. The reasoning behind the spelling is
`docs/reference/hex-convention.md`, and the grading
is `docs/reference/corpus-replay-2026-09-15.md`
§*`hex-glinski-promoting` against its own pre-registration*.

## `hex-glinski-hexagon.json` and the two flanks — schema 16, the native hexagon

Q281a4, 2026-09-28, all three unmeasured (Q281d measures them). Each adds one thing to the
game above, and the pre-registration is `docs/game-ideas.md` §*Q281 specified*, *The
showcases*.

- `hex-glinski-hexagon.json` (`6eaaad41673b`) is `hex-glinski-promoting` with
  `board.shape: "hexagon"` in place of its thirty walls: the corners are not cells at all.
  It compiles to the parent's tables but for the Zobrist draws, so its score is predicted
  to equal the parent's to the last printed digit.
- `hex-glinski-flanks.json` (`5ba53615c752`) joins the hexagon's upper-left and lower-right
  sides, `rules.wrap.nw_se: "both"`: the two flanks no army stands on, so nothing crosses
  at move one and every crossing is a middle-game move. Predicted: at least half the graded
  games cross, and an index within 0.05 of the parent's.
- `hex-glinski-flanks-one-way.json` (`2edcd9a30022`) is the same seam one way per player,
  the first player leaving by the lower right and the second by the upper left, which is
  the rule turned half a turn. Predicted: at least 0.3 of games cross.

## Schema 10's seven — one feature each, and two of them broken on purpose

Q101, 2026-09-07. `SCHEMA_VERSION` went to 10 with five features and no game showing any
of them, which
`docs/reference/schema-showcase-coverage.md`
rows 23--30 counted as a debt. These seven pay it. **One feature per ruleset**, because
schema 9's first showcase moved four things at once and could not say which one moved the
index; each is one key from a control already measured under `v10`; and each carried a
demonstration clause and an index band fixed in a commit of its own before it was played
(`docs/reference/schema-ten-showcases.md`).
Every clause passed and four of the finer predictions missed, which that file records
rather than repairs.

| ruleset | feature | control | index `v10` | the clause |
|---|---|---|---|---|
| `succession` | `rules.contacts` | chess 0.7943 | 0.7516 +/- 0.0260 | 0.0750 of captures fire a link |
| `queen-bomb` | `explode_on_capture.by` | atomic 0.6631 | 0.7294 +/- 0.0248 | 0.1064 of captures explode |
| `queens-wake` | `explode_on_capture.of` | atomic 0.6631 | **0.7464 +/- 0.0207, not viable** | 0.0818 of captures explode |
| `perpetual` | `repetition_draw: "never"` | chess 0.7943 | 0.8113 +/- 0.0185 | 0 games end by repetition |
| `wolves-immune` | `npcs[].immune` | wolves 0.7548 | 0.7647 +/- 0.0192 | identical tables |
| `stones` | `drops.stock: "unlimited"` | chess 0.7943 | **0.3144 +/- 0.0563, not viable** | 0.5198 of moves are drops |
| `three-stones` | `drops.stock` as a count | stones 0.3144 | 0.8058 +/- 0.0151 | 0.8398 of seats spend all three |

**The last two rows are of games that no longer exist**, and the row above is the reason
they are still printed. On 2026-09-08 (Q131) the user found that both stone descriptions
say a stone *"is never captured and never leaves"* while neither spec carried a capture
matrix -- so anything could take one, and 0.9668 of every capture in `stones`' own run was a
stone being taken. The user's call was that **the description is the specification**, so
both specs gained the matrix and both became new games:

| ruleset | was | is | index `v10` | the clause |
|---|---|---|---|---|
| `stones` | `d9a411e36af4` | `93b02b591c87` | **0.0000, unreadable by the roster** | 0.9643 of moves are drops |
| `three-stones` | `707d1f6f1dbc` | `03aa7c3f2f18` | 0.7199 +/- 0.0160 | 512 of 512 seats spend all three |

Both pre-registered bands missed. `stones` is not a bad game any more, it is not a game: the
opening position has exactly 32 empty squares, sixteen drops a side fills them, and a fully
stoned board is a **stalemate by construction** -- 196 of 196 graded games end that way in
33.3 plies. The account, including the second free-material defect the repair exposed, is
`docs/reference/schema-ten-showcases.md` and `docs/bugs.md`.

**`stones` is kept on purpose** (the user, 2026-09-09) as an example of a game that cannot
really be played. Its spec says so in `provenance.exhibit`, outside the mechanics hash, and
the web pages print that sentence beside its number wherever the number is shown (Q141).
The number has moved since the table above: the 2026-09-11 replay reads it not viable
rather than unreadable, and the library's is the current one.

Two were **drawn rather than written**, by operators that had landed hours earlier:
`succession` by `become-what-you-take` at seed 1 on chess (a capture that promotes the
capturer: P×N→N through R×Q→Q), `stones` by `place-a-stone` on chess. The other five are
hand-written one-key deltas, because a drawn `by`/`of` list or a drawn stock count would
have been an arbitrary piece and an arbitrary number.

Three things they found that outlive the rows:

**`stones` is a non-game and the rule is not to blame.** 204 of its 256 games hit the
400-ply limit, the mean game is 344.9 plies and the median seat places 41 stones. A
moveless piece has reach 0 in `_measured_values` and `DECLARED_WEIGHT` is 0.25, so a stone
is worth a quarter of a pawn and *placing* one is free material — the ladder does nothing
else. `three-stones` is the identical game with `stock: 3` and scores 0.8058, viable. A
finite stock is not a smaller unlimited one; it is the difference between a game and a
non-game.

> **Half of that held up and half of it did not (Q131, 2026-09-08).** The valuation *was*
> the defect it names, and Q119 fixed it. But once the rules were repaired to match the
> description and measured again, `stones` got **worse** — 0.3144 to 0.0000 — because the
> unlimited stock is broken a second time, in the rule this time: permanent stones fill a
> board that cannot be unfilled, and a full board is a stalemate. "The rule is not to blame"
> was true of the number it was written about and false of the game.

**`wolves-immune` reproduced `wolves` game for game**, all 256 move lists identical, every
non-elegance family equal to four decimals. The record had said it would not, because the
Zobrist seed is the mechanics hash and a shorter spelling reseeds it; the tables do differ
in all five draws and not one game changed. So the whole of 0.7548 → 0.7647 is the elegance
term (`capture_restrictions` 5 → 0, `spec_bytes` 1898 → 1744, rule mass 81 → 71), which is
what was predicted to the fourth decimal before the run.

**`by` and `of` are symmetric in the schema and not in play.** `queens-wake` fails
`quick_kill_rate_worst_half` at 0.553 with a mean game of 39.7 plies: when the *taking of a
queen* is the one explosive act, the weaker rungs walk into an opening trap. Its sibling,
narrowed by the capturer instead, is viable and scores 0.7294.

## `merger.json` and `hierarchy-rise-throw.json` — schemas 11 and 12, one rule each

Q19b and Q20, 2026-09-14. Each is one key from its parent, and neither is measured yet.
Their numbers come with Q22's corpus `measure`.

| ruleset | hash | feature | parent | perft against the parent |
|---|---|---|---|---|
| `merger` | `85555467a4ca` | `contacts.links[].own` (schema 11) | chess | equal through depth 4 |
| `hierarchy-rise-throw` | `78130df01edd` | `position: "choose"` with a `landing` set (schema 12) | `hierarchy-rise` | equal through depth 2; 9092 against 8900 at depth 3 |

`merger` is chess in which a pawn may move diagonally onto its own rook. The rook leaves,
the pawn becomes a queen, and a king can never be merged.

`hierarchy-rise-throw` keeps `hierarchy-rise`'s promoting strike, but the attacker takes
the square and throws the promoted piece onto any empty square a king's step from where it
stood. A piece with every such square occupied cannot be taken. It is the third value on
the axis `hierarchy-rise` (`swap`) and `hierarchy-rise-stay` (`stay`) already measure. Its
pre-registration is in `docs/game-ideas.md` §*Q20 specified*, and
the engine records are
`docs/reference/engine-and-spec-record.md`
§*Schema 11* and §*Schema 12*.

## `chess-sweeping-bishops.json` — schema 13, the sweep move kind

Q21, `42bd7d06828d`, 2026-09-14. Chess with one change: the bishop's slide is written as a
`sweep`. Instead of stopping at the first piece on its diagonal, it jumps to any square on
that diagonal, and every piece it passes over is removed, its own included. It is schema
13's one feature, and not measured yet — its number comes with Q22's corpus `measure`.

Perft(1) is 30 against chess's 20. The ten extra moves are every sweep over one of the
bishop's own pawns — `Bc1` to a3, e3, f4, g5 and h6, `Bf1` to d3, c4, b5, a6 and h3
(`tests/test_schema_thirteen.py`). The rule is live from the first move, which is also the
risk the pre-registration flags: `Bf1xb5` is mate on white's first move, sweeping the pawn
on e2 clear, so `quick_kill_rate` is the gate to watch. Its pre-registration — index 0.68
to 0.78 under the canonical roster, at least 0.9 of graded games holding a `MOVE_SWEEP`, at
least 0.2 of captured pieces taken by sweep — is in
`docs/game-ideas.md` §*Q21 specified*, *The showcase*. The engine
record is
`docs/reference/engine-and-spec-record.md`
§*Schema 13*.
