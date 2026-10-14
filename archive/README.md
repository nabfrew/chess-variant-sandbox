# Games worth keeping, that are not part of the corpus

The middle rung of the pathway out of a design run. A candidate lives under `runs/`,
which git ignores; a game in here is committed, playable, and linked to — and is **not**
part of what the index is calibrated against.

That distinction is the reason this directory exists rather than a `keep` flag on
`rulesets/`. Every file in `rulesets/` is read by `metrics/calibration.py` as the human
band, held to the cheap gates by `tests/test_index_quality.py`, and pinned by frozen
perft ladders — so a game admitted there becomes part of what every other game is
checked against. Nothing in `metrics/` or `tests/` reads this directory.

Since `v12` (2026-10-03, `docs/plan-evolving-score.md` §4) the band **checks the scale and
places none of it**. Every curve end comes from the measurement's domain, its gate, a
one-sentence meaning, or a quantile of the published band games read once and pinned as a
number (`docs/reference/v12-calibration.md`), so admitting a game to `rulesets/` moves no
curve and no other game's score. What it changes is what `calibrate` reports, whether the
scale puts real games above known-bad ones, and, for a published game, the set a later
revision would re-read a pinned end from, under its own pre-registration. That is still a
claim about every other game, and still the reason a game worth keeping comes here first.
(Under `v11` and before, the band's extremes placed the curve ends, so admitting a game
moved every other game's score at the next re-anchor; `docs/reference/baseline-review.md`
§1.5 has what that cost.)

The scores quoted below are as measured on the date and under the weights version each
names (`v10` for most). A score is an evolving estimate: it is worked out again whenever the
computer players or the measurements change, and a game's current number, or its last one
marked *provisional*, is on its page. The quoted numbers are records of what was measured
then, not today's scores.

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
`docs/reference/schema-6.md`. **`hierarchy-fall-swap` was retired on 2026-10-12** by the
library audit (Q868, `docs/plan-library-audit.md`): under `v12` it plays and scores as
`hierarchy-fall` does, and the position rule's lever is shown by the three `rise` games.
The table above is the `v9` record and stays as it was measured.

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
stays as that game's measurement. **`weathervane-torus` was retired on 2026-10-12** (Q868)
with the family's other wrap cell, `rulesets/sardines-torus`: wrapping a full board changed
nothing a player could see, so the family is now `sardines` and `weathervane`.

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

## `wolves-rotated.json` — the same wolves, one step later (retired)

**Retired on 2026-10-12** with its three siblings by the library audit (Q868): an experiment's
arms that found no effect. The specs are in git history at `1ebf1313`; the record below stands.

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

## `wolves-rotated-2.json`, `-3.json`, `-4.json` — the other three phases (retired)

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

## `hex-glinski-promoting.json` — schema 14, hexagonal chess (retired)

**Retired on 2026-10-12 for its native twin `hex-glinski-hexagon`** (Q281e, decided inside
the library audit, Q868): the same game move for move, drawn with thirty walls. The tests
that read the walled spelling rebuild it from the twin (`tests/retired_specs.py`); the spec
is in git history at `1ebf1313`.

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

## `hex-glinski-hexagon.json` and the two flanks — schema 16, the native hexagon (flanks retired)

**The two flanks were retired on 2026-10-13 for `pinwheel`** (Q870, the library audit's second
pass): Glinski's game with a seam tagged on, which a game built around a hexagon's one-way seam
shows better (below). The tests that read them rebuild them from the hexagon
(`tests/retired_specs.py`); the specs are in git history at `542bd2de`.

Q281a4, 2026-09-28, all three unmeasured (Q281d measures them). Each adds one thing to the
game above, and the pre-registration is `docs/game-ideas.md` §*Q281 specified*, *The
showcases*.

- `hex-glinski-hexagon.json` (`6eaaad41673b`) is `hex-glinski-promoting` with
  `board.shape: "hexagon"` in place of its thirty walls: the corners are not cells at all.
  Since the walled parent's retirement (Q868) it is the library's Glinski game.
  It compiles to the parent's tables but for the Zobrist draws, so its score is predicted
  to equal the parent's to the last printed digit.
- `hex-glinski-flanks.json` (`5ba53615c752`) joins the hexagon's upper-left and lower-right
  sides, `rules.wrap.nw_se: "both"`: the two flanks no army stands on, so nothing crosses
  at move one and every crossing is a middle-game move. Predicted: at least half the graded
  games cross, and an index within 0.05 of the parent's.
- `hex-glinski-flanks-one-way.json` (`2edcd9a30022`) is the same seam one way per player,
  the first player leaving by the lower right and the second by the upper left, which is
  the rule turned half a turn. Predicted: at least 0.3 of games cross.

## `hex-glinski-turned.json` and `hex-glinski-turned-mirrored.json` — schema 18b, hex symmetries (retired)

**Retired on 2026-10-13 for `pinwheel`** (Q870): on Glinski's pieces `rotate6` and `dihedral12`
write the same moves, so both played the hexagon game move for move. `pinwheel`'s one-handed
knight is the first piece on which the two spellings differ. The tests rebuild both from the
walled game (`tests/retired_specs.py`); the specs are in git history at `542bd2de`.

Q782, 2026-10-10. `hex-glinski-promoting` with every officer's move written once and turned
by the hexagon's own symmetries, one feature each: `hex-glinski-turned` (`30c1003b9a61`) by
`rotate6` (the rook is `[[1, 0]]`, the bishop `[[1, 1]]`, the knight `[[1, 2], [2, 1]]`) and
`hex-glinski-turned-mirrored` (`461e29677d6d`) by `dihedral12` (the knight is `[[1, 2]]`). The
same game: tables equal to the parent's but for the Zobrist draws, and measured, every one
of each's 1,024 games the parent's move for move, 0.9248, viable, as pre-registered
(`docs/reference/schema-showcase-coverage.md` rows 51 and 52). The committed hex specs keep
their listed vectors, since rewriting them would make new hashes and orphan their runs.
Their walled parent is retired (Q868); the two stay because each is the only game using its
symmetry, and their descriptions say they are the hexagon game written the short way.

## `pinwheel.json` — the hexagon set's designed game (Q870)

`61c4e247dc3c`, 2026-10-13. Built to show what four retired variants of `hex-glinski-hexagon`
were kept for, in one game: a 61-cell hexagon, Glinski's army scaled to it (seven pawns), the
rook, bishop, queen and king written once and `dihedral12`, and a knight written as one jump
`[[1, 2]]` turned `rotate6`, so it reaches the six cells of one hand of the hexagonal knight's
twelve and never their mirror images. The two empty flanks are joined one way per player, as
`hex-glinski-flanks-one-way`'s were, so both armies go round the board the same way, and the
half turn keeps both the knights' hand and the seam's direction the same for both players.
Pre-registered in `docs/plan-library-audit.md` (*`pinwheel`*) before it was measured;
`tests/test_pinwheel.py` holds the demonstration clauses. Measured 2026-10-13 (v12, four runs,
the schema-24 core): **0.895 [0.878, 0.920], viable (4-0)**, against the parent's 0.925; all
five measured clauses hit: the seam crossed in 0.93 of games, knights made 10% of the moves,
games averaged 74 plies. The second player leads (first-player score 0.406).

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
| `wolves-immune` (retired, Q876) | `npcs[].immune` | wolves 0.7548 | 0.7647 +/- 0.0192 | identical tables |
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

**`wolves-immune` reproduced `wolves` game for game** (and for that reason the library audit
retired it on 2026-10-13 for `pitchfork`, a game in which the key changes the play; Q876), all 256 move lists identical, every
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

## `rifle.json`, `circe.json`, `andernach.json`, `kamikaze.json` — schema 17, what a capture leaves behind

Q540's four showcases (2026-10-09), each chess with one rule added and nothing else, and
each a variant older than this project: rifle chess (a capture removes the victim and the
capturer stays where it stood), circe (a taken piece is reborn on its home square, or is
gone if that square is occupied), Andernach (a capturer other than the king changes sides
on the square it took) and kamikaze (a capture destroys the capturer with its victim, and
nothing else). None changes an opening move, so each one's perft equals chess's until the
first capture. In twenty seeded random games each rule fired in all twenty
(`docs/reference/schema-showcase-coverage.md`, rows 39 to 42). Unmeasured until the
library's schema-17 replay. The rules are in `docs/reference/engine-and-spec-record.md`,
*Schema 17*.

## `einstein.json`, `omega.json` — schema 18a, a keeping link and a walled rank

Q781's two showcases (2026-10-09), each a variant older than this project written from
Q533's audit spec (`docs/reference/variant-expressibility-2026-09.md`, rows 18 and 13) with
the one rule the audit could not say added. `einstein` is Einstein chess: a capture promotes
the capturer one step up pawn, knight, bishop, rook, queen and a quiet move demotes it one
step down, and a queen that captures stays a queen (schema 18's keeping link). `omega` is
Omega chess, a 10x10 board with a wizard cell outside each corner, walled on a 12x12, with
champions and wizards, and castling as in chess on a rank whose walls stand beyond the rooks
(schema 18's walled rank). In twenty seeded random games the keeping link fired in 7 and
castling in 1, both short of their pre-registered shares
(`docs/reference/schema-showcase-coverage.md`, rows 43 and 44). Measured 2026-10-09 (v12,
four runs): `einstein` 0.849, `omega` 0.928, both viable. The rules are in
`docs/reference/engine-and-spec-record.md`, *Schema 18*.

## `extinction-chess.json`, `grand.json` — schema 18c, a count per type and the lost pieces

Q783's two showcases (2026-10-09), written from Q533's audit specs as 18a's were
(`docs/reference/variant-expressibility-2026-09.md`, rows 29 and 15). `extinction-chess` is
Extinction chess: nothing is royal, and a side loses when it has no king, no queen, no rooks,
no bishops, no knights or no pawns left, each kind counted on its own (schema 18's
`extinction.per_type`); a pawn may promote to a king. It is not called `extinction` because
that is a rule key, and a game's name may not be a word the code uses. `grand` is Grand
chess on 10x10 with the archbishop and chancellor, and a pawn promotes only into a piece its
side has lost (schema 18's `promotion.only_lost`). In twenty seeded random games every
`extinction-chess` game ended by one kind running out and every `grand` game offered a pawn a
zone square with a target withheld (`docs/reference/schema-showcase-coverage.md`, rows 45 and
46). Measured 2026-10-09 (v12, four runs, on the 18c core): `extinction-chess` 0.830,
`grand` 0.894, both viable. Both verify against Fairy-Stockfish (`extinction-chess` without
its castling, which the translator refuses for a king that is not royal). The rules are in
`docs/reference/engine-and-spec-record.md`, *Schema 18*.

## `placement-chess.json`, `sittuyin.json` — schema 18d, the placement phase

Q784's two showcases (2026-10-09) for `drops.placement`: while a player holds a piece in
hand, placing one is their only move, and the king starts in hand. `placement-chess` is
chess with the pawns placed and the other eight pieces in each hand, dropped on the back
rank; it leaves out Fairy-Stockfish's opposite-coloured bishops and the castling of a dropped
king and rook, which the DSL cannot say, and is otherwise exact (it verifies). It is not called
`placement`, a word the code uses. `sittuyin` is Burmese chess: makruk's pieces, the pawns
staggered across the third and fourth ranks, the rest dropped on the three back ranks (a
chariot on the first), and a pawn on its long diagonal may become a general only once its
side's general is lost. It counts **with a caveat**: the pawn promotes on arrival rather than
in place, and the counting rules are the plain no-progress clock; Fairy-Stockfish refuses its
diagonal zone, and its promotion-free twin verifies. In twenty seeded random games of each,
every position with a piece in hand offered only drops and withheld a pawn's step
(`docs/reference/schema-showcase-coverage.md`, rows 47 and 48). Measured 2026-10-09 (v12, four
runs, on the 18d core): `placement-chess` 0.899, `sittuyin` 0.851, both viable.

## `kyoto.json` — schema 18e, drop either face

Q785's showcase (2026-10-09) for `drops.either_face`: a piece in hand may be dropped as itself
or as its other face, the piece whose `promoted_from` it is. Kyoto shogi on five by five:
every piece but the king turns over after each move it makes (pawn and rook, silver and
bishop, knight and gold, lance and tokin, spelled as mandatory promotions over the whole board
both ways), a captured piece enters the hand as its plain face, and it may be dropped anywhere
empty as either. It is Q533's audit spec (`ef312a2e4736`) with each turned face's
`promoted_from` and the flag, and it is exact: it verifies against Fairy-Stockfish's
`dropPromoted` and `pieceDemotion`, and our moves equal the built-in `kyotoshogi`'s. In twenty
seeded random games every hand held only plain faces, every held piece was offered as both
faces on the same squares, and a turned face was dropped in all twenty (0.494 of the drops
played) (`docs/reference/schema-showcase-coverage.md`, row 49). Measured 2026-10-09 (v12, four
runs, on the 18e core): 0.887, viable.

## `jeson-mor.json` — schema 18f, win by leaving a square

Q786's showcase (2026-10-09) for `reach.on: "leave"`: the game is decided by the move that takes
the named piece off a goal square, and standing there decides nothing. Jeson Mor on nine by
nine: nine knights a side on the first and last ranks, and a knight that moves off e5 wins
(taking every enemy knight also wins, and a side with no move loses). It is Q533's audit spec
(`jeson-mor-approx`, `9447c056a0c0`, which won on arrival) with the flag. Fairy-Stockfish's
`jesonmor` is another rule (its `flagMove` calls the game one move after a knight reaches e5),
so `fairy` refuses it. In twenty seeded random games a knight stood on e5 without ending the
game in all twenty, and every game ended on a move off e5
(`docs/reference/schema-showcase-coverage.md`, row 50). Measured 2026-10-09 (v12, four runs, on
the 18f core): 0.698, **not viable**: the worst half's games are over in about six plies.

## `shogi-drop-promoted.json`, `shogi-backward-drops.json`, `pawn-knockdown.json` — the three gap showcases

Q787's three showcases (2026-10-09), each a one-rule delta from a corpus game, for three
features that had no game (`docs/reference/schema-showcase-coverage.md`, *The 2026-10-09
audit*, gaps G1 to G3). `shogi-drop-promoted` is shogi plus schema 9's `promote_on_entry`: a
piece dropped into the far three ranks arrives promoted. `shogi-backward-drops` is shogi with
a pawn that faces a way and turns (schema 8): every placed pawn faces the enemy and a
dropped pawn arrives facing its own side, schema 8's `facing.initial` on a piece that enters
play. `pawn-knockdown` is chess in which only the pawn, when it captures, knocks the piece
down a rank and throws it onto a free square next to the capture square, the pawn's player
choosing which (schema 12's `contacts.position: "choose"`). None changes shogi's or chess's
opening captures, so the opening perfts equal their parents' (`shogi-backward-drops` has 39
opening moves, shogi's 30 and nine pawn turns). Unmeasured until the library's next replay.
`tests/test_showcases_q787.py` shows each feature acting.

## `divide.json`, `ferry.json`, `carousel.json`, `wormhole.json`, `wormhole-adrift.json`, `shut-wormhole.json`, `cylinder-wormhole.json` — schema 19, portals

Q772a's showcases (2026-10-10) for `rules.portals` (`docs/plan-portals.md` §7 P4 and *Showcase
games*): a piece that ends its move on a portal comes out one step beyond the other end of the
pair, in the direction it was travelling. **Divide** is the walled game the user asked for:
Capablanca chess on ten by eleven with a wall across rank 6, crossed only by one pair whose
ends each player moves on its own side, one way per player (the plan's option A). **Ferry**
is Los Alamos with two numbered pairs that either player may move. **Carousel** is Los Alamos
on a six by eight cylinder with a private one-way lift for each player, closed to the other.
The four small ones each add one thing to a measured parent: **wormhole** (Gardner, one fixed
two-way pair on b3/d3), **wormhole-adrift** (the same pair moved by either player),
**shut-wormhole** (the same pair closed to the first player) and **cylinder-wormhole**
(cylinder chess with a pair on d4/d5, reached both ways round, the player choosing the exit).
Pre-registered and measured 2026-10-10 (v12, four runs, the schema-19 core):
`divide` 0.666, `ferry` 0.694, `carousel` 0.771, `shut-wormhole` 0.726 and
`cylinder-wormhole` 0.858, viable; `wormhole` 0.584 and `wormhole-adrift` 0.644, **not
viable**: a pawn through the door on Gardner's middle row lands a step from promotion, and
the first mover gets there first. Divide's strong players draw on the no-progress clock
(the plan's named miss), and Ferry's players shuffle portals into repetition (0.404 of
games). `docs/reference/portal-showcases.md` has the clauses and the misses;
`tests/test_showcases_q772a.py` shows each feature acting, and that no move crosses Divide's
wall but through the door. **`wormhole-adrift` was retired on 2026-10-12** (Q868): movable
portals are shown by `ferry` in a playable game, and the tests rebuild it from `wormhole`.

## `ferry-no-straight-back.json` — schema 20, no straight back

Q807's answer to Ferry's shuffle (2026-10-10, `docs/plan-portals.md` *Ferry's shuffle*):
Ferry with `rules.portals[].no_straight_back` on both pairs and nothing else changed, so a
portal that has just been moved may not be moved straight back where it came from on the very
next move (a portal's ko; in 404 of Ferry's 414 repetition games the opponent had done exactly
that). A new game beside Ferry, which keeps its own number. Pre-registered and measured
2026-10-10 (v12, four runs, the schema-20 core): **0.827, viable (4-0)**, above its
pre-registered band; repetition 0.072 of games (Ferry 0.404, Los Alamos 0.091), the bar
removing a legal move at 0.457 of plies, the strongest two deciding 26 of 32 cross games.
`docs/reference/portal-showcases.md` *Q807* has the clauses; `tests/test_schema_twenty.py`
holds the rule against `tests/portal_oracle.py`.

## `divide-two-way.json` — schema 19, Divide's two-way door (retired)

Q772e, 2026-10-12, `2fa712feb218`. Divide's option B (`docs/plan-portals.md` *Divide's return
rule*): `divide` with the pair's `passage` removed and nothing else changed, so either
player's pieces may go in at either end and a raider can come home by the door it came out
beside; each end is still moved only by the player in whose half it stands. A new game beside
`divide`, which keeps its own number. Pre-registered and measured 2026-10-12 (v12, four runs,
the schema-23 core): **0.671 [0.636, 0.686], viable (3-1)**, A's numbers on every family. The
door is used both ways (a return in 0.25 of games), and the strong players still draw: 4 of
their 32 cross games decided against a bar of 0.4, so neither Divide is kept as the answer and
option C is owed. `docs/reference/portal-showcases.md` *Q772e* has the clauses;
`tests/test_showcases_q772a.py` holds the wall and the return.

## `divide-fixed.json` — schema 19, Divide's fixed door (retired)

Q772f, 2026-10-12, `31ead85da009`. Divide's option C (`docs/plan-portals.md` *Divide's return
rule*): `divide-two-way` with the pair's `moved_by` removed and nothing else changed, so the
ends stand on e3 and e9 for the whole game, nobody may move them, and either player's pieces may
go in at either end. Pre-registered and measured 2026-10-12 (v12, four runs, the schema-23
core): **0.605 [0.565, 0.626], not viable (1-3)**, the weak half of the ladder parking in three
runs. The strongest two decide none of their 32 cross games, so by the pre-registered rule
**`divide` (option A) is kept as Divide** and this game and `divide-two-way` stayed beside it as
the record that no door rule fixed the draws, until the library audit retired both on
2026-10-12 (Q868; the specs are in git history at `1ebf1313` and the tests rebuild them from
`divide`); a sixteen-fold search budget on `divide` decided
4 of 24 cross games, so the draw is the walled board's. `docs/reference/portal-showcases.md`
*Q772f* has the clauses; `tests/test_showcases_q772a.py` holds the wall, the return and the
fixed ends.

## `wolves-patrol.json` — schema 23, two wolves on one walk

Q850, 2026-10-12, `b2011455c4af`. **The user's game** (2026-10-08): two wild wolves walk one
sixteen-square circuit through the middle of the board, `a4` to `h4`, up to `h5`, back to
`a5`, and round again, one starting on `a4` and one on `h5`, half a lap apart. The rule
language could not say it until schema 23's `rules.npcs[].also_at`
(`docs/reference/designer-case-study-wolves.md`); everything else is `wolves`' (a wolf eats
anything but a wolf, only a knight may take one, no bounty). Pre-registered and measured
2026-10-12 (v12, four runs, the schema-23 core): **0.911 [0.893, 0.930], viable (4-0)**,
beside `wolves`' 0.907. Five of six clauses hit; the headline missed: both wolves stand in
0.656 of positions (bar 0.80), because knights take 0.53 wolves a game. The walk eats 6.8
player pieces a game. `docs/reference/wolves-patrol.md` has the clauses;
`tests/test_schema_twenty_three.py` holds the rule (the queue rule, which never fires here,
included).

## `pitchfork.json` — wolves only a pawn may take (Q876)

`d2a722b461ff`, 2026-10-13. Built to show what `wolves-immune` was kept for, `npcs[].immune`,
doing something in play: chess plus **eight wolves**, two to a ring, nose to tail, on the two
centre rings of Q68's screen (`b4 c5 d4 c3 b3` and `g4 f5 e4 f3 g3`, both `mirror: true`;
`also_at` puts the second wolf on each ring), and `immune: ["P"]`, so **only a pawn may take
one**. The rings hold the squares the knights and bishops would come out to, so the pawns
clear the way, and a pawn that takes the leading wolf of a pair is eaten at once by the one
behind. No bounty, no score. Pre-registered in `docs/plan-library-audit.md` (*`pitchfork`*)
before it was measured; `tests/test_pitchfork.py` holds the demonstration clauses (the word's
three readings on one position, the five lists it stands for, the pair's price, random play).
Measured 2026-10-13 (v12, four runs): **0.907 [0.885, 0.928], viable (4-0)**, beside `wolves`'
0.907; all five clauses hit: **3.84 wolves taken a game, every one of 3,929 by a pawn**, a wolf
on the board at 0.978 of plies and 4.16 of 8 left at the end, half the takers eaten by the
pair's second wolf. With the key deleted the officers clear the centre instead (a 100-game
screen: 5.87 taken a game, 0.38 of them by pawns).

## `sluice.json`, `crease.json`, `pawnproof-wormhole.json` — schemas 21 and 24, portal maps, filters and folds (Q844b)

2026-10-13. The showcases of what `docs/plan-portals-2.md` added, on the rules as they now
stand; `docs/reference/portal-showcase-ideas.md` has the decision on each of Q844's seven ideas
(two games and one variant kept, four dropped, each with its reason), the designs and the
predictions, written before anything was measured, and `tests/test_portal_showcases_q844b.py`
the demonstration clauses.

- **`sluice`** (`f3c624d2f297`): chess with a funnel a side, the game of schema 21's maps
  (`ends`, `to`, `enter`). White's doors a3 and h3 lead, for White only, one step beyond White's
  gate on e5, a block in Black's half that Black moves; Black's doors a6 and h6 lead beyond
  Black's gate on e4, which White moves. The attacker picks the door, the defender where the
  raid lands.
- **`crease`** (`0448722861a3`): Capablanca chess with two fold pairs crossing the board, b4
  with i5 and i4 with b5, every fold moved by either player, never straight back: the game of
  schema 24's folds. A landing on an empty fold stays or goes through to the other end,
  capturing there.
- **`pawnproof-wormhole`** (`feaad3754a7e`): `wormhole` with `carries` on its pair naming every
  piece but the pawn, kept as a variant because the contrast is the point: Wormhole's first
  player wins the pawn's race through the door, and the filter takes the race away.

Measured 2026-10-13 (v12, four runs each, one campaign), all three **viable (4-0)**: Sluice
**0.848 [0.822, 0.869]**, all six clauses hit (a transit in 0.991 of games, every door in two
games of three, the gates moved at 0.155 of plies); Crease **0.901 [0.878, 0.918]**, five of
eight hit (folds moved at 0.328 of plies and onto a piece in 0.589 of games, but by both players
in only 0.786 of games, and a go-through
in 0.700 of games against 0.9 predicted, and a capture through in 0.368 against 0.5); pawnproof
wormhole **0.572 [0.546, 0.606]**, all five hit (no pawn transit, the first seat 23 of the top
band's 46 decided games against Wormhole's 33 of 33).

## Retired by the library audit (Q868, 2026-10-12)

The user asked for a library in which every game has a purpose (`docs/plan-library-audit.md`,
which has a verdict for each of the 101 games). Fourteen near copies left the tree, ten of
them from here: `wolves-rotated`, `-2`, `-3` and `-4`, `hex-glinski-promoting`,
`divide-two-way`, `divide-fixed`, `wormhole-adrift`, `hierarchy-fall-swap` and
`weathervane-torus` (and `rulesets/sardines-torus`, `tunnel-vertical`, `onewaytunnel` and
`torus-extended`). Their runs stay in `runs/`, their specs are in git history at `1ebf1313`,
and on the site each is *retired*: off the listing and refused for a new match, with its page,
its matches and its ratings kept. The kept near copies' descriptions now say what their one
difference does in play.

## The audit's second pass (Q870, 2026-10-13)

The user asked whether a near copy kept for its extra rule would show that rule better as a
game built around it, and whether the names said what the games are. Four hexagon variants
left the tree for one designed game, `pinwheel` (above): `hex-glinski-flanks`,
`hex-glinski-flanks-one-way`, `hex-glinski-turned` and `hex-glinski-turned-mirrored`. Eighteen
games were renamed (only `name` moved, so no hash, link, run or rating did), among them the
hierarchy games, which are now *demotion chess*, *gift chess*, *gift from afar*, *thrown
gifts* and *amazon ladder*; `docs/plan-library-audit.md` has the table and the reason for each.
The pass's last item, Q876, retired `wolves-immune` for a designed game, `pitchfork` (above); its
spec is in git history at `c91613a3`, and the tests rebuild it from `wolves`.
