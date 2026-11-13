# triad: designer session record (`unlike-search.md` §2)

**Admitted at the first read** (`8716890978a9`: i y, ii y, iii y, iv y, v y). Deviation: the session's heavy-lane use was about 1.5 to 2 slot-hours against the designer prompt's one; every tournament stayed within 96 games.


## Hand-in: design reason

# Design reason (written before any spec)

Seed: armies that beat each other in a circle, lost by running out of a kind.

## The game in one idea

Three kinds of piece on a hexagonal board, each able to take exactly one enemy kind, round a
circle, and each kind moving in its own way so that the circle is also a circle of *reach*:

- the **spear** slides along the six lines of the board and takes only horses;
- the **horse** jumps (the hex knight's twelve jumps, over anything) and takes only shields;
- the **shield** steps one cell and takes only spears.

You lose the moment you have none left of any one kind. So every kind is at once a hunter, a
quarry and a lifeline, and the army you hold is only as strong as its scarcest kind.

## The decision a player keeps facing

Which of my kinds is my weak link, and which of the opponent's is theirs, and can I hunt
theirs faster than I expose mine? Every hunter that goes forward to strike the opponent's
scarce kind walks toward its own predator: a spear that lines up on a horse from afar is safe
from that horse but dead if a shield can reach the cell next to it; a horse that jumps into a
wall of shields to take one is then on an open line for a spear; a shield that walks up to a
spear is horse food. Protection is never "the same piece behind it": a piece is only guarded by
the kind that eats its attacker, so every safe group is a mixed group of all three kinds, and
every capture is answered (or not) by a different kind than the one that made it. Shields also
block spear lines just by standing, so the slow kind is both the spear's killer and the horse's
cover. Shields only advance or move sideways, never back, so committing the shield line forward
is the one irreversible decision, and its timing is the strategic question of the middle game.

## A game from opening to end

Opening: both armies start behind a front row of shields with spears and horses behind them,
no capture possible on the first moves. Players open lines for their spears and bring horses
out, while deciding how far to push the shield front, which both protects and spends shields.

Middle game: contact. Horses raid the enemy shield front; spears punish the horses that raid;
shields advance to close in on spears that took a horse. Exchanges are three-cornered and
counted in kinds, not material: a trade that leaves me two of everything against their three,
three and one is a win for me, because their last kind is now the target.

Ending: one side is down to its last one or two of a kind and must hide that piece inside a
mixed guard while its other kinds still fight; the other side brings the one kind that eats it.
The game ends on a capture that extinguishes a kind, ideally after a real chase.

## What would make it fail

- **A fortress.** The last piece of a kind can be walled in by its own guards so that its one
  predator can never arrive safely, and the game draws by the clock. (Most likely risk: a last
  spear hiding among horses, since only a shield, the slowest piece, can take it.)
- **A dominant kind.** If spears snipe horses from across the board with no answer, the game is
  a spear shootout and the other kinds are scenery; if shields are too slow to ever catch a
  spear, spears are immortal and the circle is broken.
- **A rush.** An opening line where one side extinguishes a scarce kind in a few moves.
- **A shuffle.** If no attack is ever safe, both sides sit behind mixed walls and repeat.
- **A dead rule.** If the shields' no-retreat rule never matters, or extinction is always
  decided the same way as "take more pieces", the seed is decoration.
- **The seat.** A first player who lands the first raid before the second can close lines.


## Hand-in: break attempts and design log

# triad: design log and break attempts (session 5)

Frozen spec: `game.json`, hash **8716890978a9** (= draft v14; name, summary and description are
outside the hash). Drafts `v1.json` ... `v14*.json` are kept beside it. Single games are in `g/`
(named `<version>-<tag>-<seed>.txt`), tournaments in `runs/R-S10/sessions/5/`. Nothing here read
a score, index, component or family; tournament output was read only for results by seat,
lengths and the replayed transcripts (`seat.py`, `ends.py`: which kind the loser ran out of,
moves by kind, promotions). `tally.py` reads the same from `play` text.

Instruments used: `validate`, `smoke`, `unlike/limits.py`, `play` (about 150 single games),
five tournaments of at most 96 games, and `probe.py` (a scripted plan for one seat against the
core's player; every legal move, outcome and agent choice is the core's).

## Design log

Each version's hash, what changed and why. "self" = `minimax:nodes=6400` against itself.

| ver | hash | change | why / what happened |
|---|---|---|---|
| v1 | 5e9ea619ae5f | First spec, as `reason.md`: hex hexagon of 61 cells; spear slides the 6 lines and takes horses; horse leaps the Glinski knight's 12 (distance 3) and takes shields; shield steps forward/sideways (never back), takes spears adjacent; 3 S, 4 H, 5 W; per-kind extinction. | Limits fine (chess 0.0023). Spears were immortal (shields could never reach them) and sniped horses from across the board; 6400 lost to 1600 3 of 4. The circle was broken: **dominant spear**. |
| v2 | 848bc477cdc7 | Spear moves anywhere but captures only up to 3; shield captures up to 2 along lines; 4 of each kind; equal values. | Stronger won 4/4, spears now died, but games 24-37 plies: horses reach the shield line in two jumps and 4 shields go fast (**rush**). |
| v3 | 5d6cb47a1059 | 6 shields. | Longer, but 2 of 4 self games drew by no-progress with one or two lone pieces each fleeing: **shuffle** in the bare ending. |
| v4 | c1ac1e976682 | Shields that reach the far edge become a spear or a horse. | Every side always keeps one shield (or it has lost), so every ending has a runner. No draws in 8 games; promotions fired in every game. |
| v5 | e88e638b15ff | 11-cell-side board (91 cells), bigger armies. | To slow the horse raids. 4 of 4 self games drew (ply limit / no-progress): too sparse. Back to 61 cells. |
| v6 | bc756cd992cb | v4 with shield value 2 (a hint to the bots only). | Tournament t1 (greedy, 1600, 6400; 90 games): strength order clean, seat 39/48/3. Replays: the loser ran out of **horses in 33 of 37** decisive minimax games. The extinction rule was in practice "kill the horses". |
| v6r, v6u | 3f28dbee3b2c, 8d1bf6a6fe70 | Dead-rule probes: shields may retreat; spear captures at any range. | 4 games each, inconclusive (one 300-ply draw with retreat). |
| v7a | 1716540815af | 5 horses (g1). | t2 (1600, 6400; 96 games): seat 52/42/2, median 103 plies, 6400 beat 1600 39 of 48. Still H ran out 74, W 18, **S 2**: spears never ran out. |
| v7b, v7c | 2b3f58efd9f3, 5670c31155a2 | Spear's move limited to 3 (alone, and with 5 horses). | No clear change; not kept. |
| v8 | 332209c1e47c | 3 spears, 6 horses, spear range 3. | Shields wiped out in 41-62 plies by six horses. Overcorrected. |
| v9a | d8d67c84ed42 | v7a with shield capture range 3. | t3: H 57, W 28, S 3. Spears still safe; 8 draws. |
| v10 | b8454b3a8c74 | 3 S, 6 H, 7 W, shield range 3. | t4: W 58, H 33, S 2; **seat 56/37**; 6400 v 1600 only 0.67. Spears still never the lifeline that broke. Diagnosis: the spear's killer (the shield) had to cross the horse-filled middle to reach a spear standing at home. |
| v10r | c62816ba79aa | Dead-rule probe: shields may retreat. | Two 22- and 26-ply games in 10: a retreating shield opened a horse capture. |
| v11 | 712cf99d5c94 | Shield captures a spear anywhere down an open line. | Now spears ran out (5 of 9), H none; three games under 40 plies (3 spears too fragile). |
| v12 | e49885fc380e | v11 with 4 spears, 5 horses, 7 shields. | All three kinds run out, but the first player's horse took a shield on move 2 (Hf1-e4, e4xd7): every game opened with a melee; p1 won 6 of 8. |
| v13 | 2f41ac35acb9 | v12's rules on the 91-cell board. | p0 won 6, p1 0, 4 draws. Rejected. |
| **v14** | **8716890978a9** | v12 with the horse jumping to any cell exactly two away (12 cells, over anything) instead of the distance-3 knight. | First contact needs three horse moves, not two. 10 self games: ran out H 3, S 3, W 4, no draws, 55-94 plies. t5 (96 games): W 51, H 25, S 15, 5 draws, seat 50/41, median 66 plies, 6400 v 1600 0.76. **Frozen.** |
| v14np, v14r | 57b718098b3f, 583444bda3cd | Dead-rule probes on v14 (below). | Both rules live. |

Views that changed during play (`reason.md` is left as written): the reason expected the spear
to be the hardest kind to kill and named "a last spear hiding among horses" as the likeliest
fortress. Play confirmed the first half far more strongly than expected (v1-v10: spears almost
never the kind that ran out), and the cure was not to protect other kinds but to give the
shield, the spear's killer, a long reach: it crawls but fires down a whole open line. The
promotion rule was not in the reason; it entered at v4 as the cure for bare-ending shuffles,
and it serves the seed (a side short of a kind can buy it back). The shields' no-retreat rule
was in the reason and survived.

## Break attempts on the frozen game (v14)

1. **A rush.** (a) `probe.py rush`: the planned seat sends horses up the board every move and
   takes every shield offered. Lost 6 of 6 against `minimax:nodes=6400` in 21-28 plies, both
   seats: raiding horses are taken by spears and the raider runs out of horses. (b) The
   opening capture on move 2 that v12 had (Glinski knight to the shield line in two jumps) is
   gone: the distance-two horse needs three moves to touch the enemy shields, and the shortest
   tournament game was 36 plies (3 of 96 under 40). (c) In the hand-in transcripts the 39-ply
   win of 25600 over 1600 is the weaker player putting spears on open shield lines
   (`Se9-i5`, `Sa9-a5`), not a forced line. **Not broken.**
2. **A fortress or shuffle.** `probe.py shuffle`: the planned seat moves a spear back and forth
   in its camp and only plays the 1600 player's move when that move is a capture. Lost 6 of 6
   (53-114 plies) against 6400: sitting still lets horses eat the shield line and the shields'
   long fire is never used. Draws in t5: 5 of 96, all by repetition, none by no-progress or the
   ply limit; in 10 self games none. Without promotion (v14np) 2 of 10 drew and two games ran
   206 and 259 plies, so the bare-ending shuffle is real but promotion closes it. I could not
   construct a last-piece fortress: every kind has a long-range killer (spear 3 cells, shield
   whole line) or an unblockable one (horse), so no wall of one's own pieces covers a piece
   from all three. **Not broken** by what I tried; a hidden last horse against an opponent
   with few spears left is the case I would look at next.
3. **A forced line.** I found none: the bots' openings vary (horse sorties on either wing,
   shield advances), and the stronger player wins by a margin rather than by one line (25600 v
   1600 6 of 6 from both seats in single games; 6400 v 1600 won 36, lost 10, drew 2 in t5). Deterministic bots
   repeat a game for some seeds (hand-in seeds 1 and 2 of 25600 v 1600 are the same game),
   which says nothing about the game. **Not found.**
4. **A dead rule.** (a) Promotion off (v14np): 2 draws in 10 and two games over 200 plies,
   against none in v14; 45 promotions in t5's 96 games. Live. (b) Shields may retreat (v14r):
   1 draw in 10 and games much longer (55-238 plies, median about 148 against about 82):
   no-retreat is what makes pushing the shields a commitment and the game decisive. Live.
   (c) The spear's short strike (v1, v6u): an unlimited strike made spears dominant. Live.
   (d) The capture circle and per-kind extinction are the game. Live by construction; t5 shows
   all three kinds being the one that runs out (W 51, H 25, S 15).
5. **A dominant piece or move.** Moves by kind in t5: horse 3059, spear 2101, shield 1819; every
   kind captures and every kind is the one that runs out in some games. Shields are the kind
   lost most often (51 of 91), which matches their role as the front line. No single kind does
   all the work. Earlier dominant pieces (spear in v1-v10, horse in v12's openings) were fixed.
   **Not broken.**
6. **Seat.** First mover arrives first with a horse sortie; the recapture comes from the second
   player. t5: p0 50, p1 41, draws 5 (0.55 for the first player). Within mirror pairings 1600
   v 1600 went 17-6-1 to the first player, but 6400 v 6400 went 14-8-2 in t5 and 3-7 in ten
   single games, so 6400 self pooled is 17-15-2. **A mild first-player lean at the weaker
   budget is the break I could not fully close**; I found no mechanism that makes it grow
   with strength.

## Tournaments (each under a 4-slot heavy lease, released after)

| label | spec | agents | games | read |
|---|---|---|---|---|
| v6-t1 | v6 | greedy, 1600, 6400 | 90 (10 x 9 pairings) | seat 39/48/3; horses ran out 33 of 37 |
| v7a-t2 | v7a | 1600, 6400 | 96 (24 x 4) | seat 52/42/2; S ran out 2 |
| v9a-t3 | v9a | 1600, 6400 | 96 | seat 48/40/8; S ran out 3 |
| v10-t4 | v10 | 1600, 6400 | 96 | seat 56/37/3; S ran out 2 |
| v14-t5 | v14 (frozen) | 1600, 6400 | 96 | seat 50/41/5; W 51, H 25, S 15; median 66 plies |

Plus two `probe.py` sets (v6 and v14, 18 games each) under the same lease.

## Final checks (frozen `game.json`)

```
$ uv run python unlike/limits.py game.json
game.json [8716890978a9]
  (a) likeness to chess 0.0085  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game pinwheel [61c4e247dc3c] 0.1838  below 0.5: yes
exit 0

$ uv run chess-variant-sandbox validate game.json
triad [8716890978a9] is valid
  board       9x9
  pieces      S, H, W
  opening     28 legal moves
  rule mass   48.5

$ uv run chess-variant-sandbox smoke game.json
triad [8716890978a9]: passed
  draw_rate        0.2
  forced_rate      0.0
  games            150.0
  mean_branching   50.62
  mean_plies       231.09
  opening_moves    28.0
  stall_rate       0.2
```

(smoke's figures are random playouts.)

## Hand-in transcripts (`transcripts/`, frozen spec)

| file | result | ran out |
|---|---|---|
| 6400-self-seed0 | p1 by extinction, 86 plies | horses |
| 6400-self-seed1 | p1 by extinction, 80 plies | spears |
| 6400-self-seed2 | p1 by extinction, 92 plies | spears |
| 25600-v-1600-seed0 | 25600 by extinction, 97 plies | spears |
| 25600-v-1600-seed1 | 25600 by extinction, 39 plies | shields |
| 25600-v-1600-seed2 | identical to seed 1 (deterministic players) | shields |


## Blind read 1 (read-5)

# Blind reading: triad

Tools used: `play` (single games, no lease), and `probe.py` in this folder (light: replays the
recorded transcripts through `Engine.find_move`/`apply`, counts pieces by kind, and asks
`Engine.root_scores` for a full-window score of every root move; it plays no game). No score,
test, measure, analyse, tournament or other forbidden command was run.

## Attempts

1. **Seat.** Self-play over extra seeds 3-10:
   `minimax:nodes=1600` first seat won 7/8; `minimax:nodes=6400` first seat won 4/8 (with the
   three 6400 transcripts, where the second seat won all three, that is 4/11); `minimax:nodes=25600`
   first seat won 4/4 (seeds 0-3, 39-71 plies); `minimax:nodes=102400` split 1-1 (seeds 0-1).
   With the stronger player second: `1600 v 25600` the 25600 won 4/4; `25600 v 102400` the
   102400 (second seat) won 2, one draw. The lean flips with the budget; I found no structural
   first- or second-mover win. The 25600-self 4/4 sweep is the one worrying number.
2. **Rush (horse raid on the shield wall).** Horses take shields and shields cannot hit back, so
   the obvious plan is to jump horses into the enemy front at once. It is what the bots do: the
   first capture comes on move 3 in every game. Root scores show a natural developing move
   (1...Ha8-c6, played in four of the six transcripts) already lets the first player win a shield
   by force (+2.17 after move 2 in 25600-v-1600 seed 0 and 6400-self seed 2), while quiet shield
   moves keep 0.00. So there is an opening trap. Against the weak 1600 the raid wipes out all seven
   shields in 20 moves (25600-v-1600 seeds 1 and 2), but `greedy` raiding as first seat lost both
   games to 6400, and the stronger second seat beats it (25600 v 102400 above). Opening trap: yes.
   Forced rush: not found.
3. **Fortress / last piece hidden.** The designer's named risk is a last spear walled in. In the
   transcripts the last spear survives a long time but is reached: 25600-v-1600 seed 0, Black
   down to one spear from ply 27, hides it in the a9/b8 corner (moves 37-45, Sa9-b8 back and
   forth) while a White shield walks up the edge. It is taken at ply 97 by a shield that
   captures and promotes in one move (49. Wh4xc9=S). 6400-self seed 2: White's last spear lives
   from ply 47 to 92. Black is +8 for about 20 moves without converting, then 45...Sf5-g4 is a
   forced win (every White reply scores as lost), so it ends in zugzwang. 6400-self seed 1:
   Black on one spear from ply 21 survives 60 plies and **wins**. The one draw I saw (25600 v
   102400 seed 1, repetition at ply 67) was White, ahead at +1.97, repeating Hh1-h3/h1 against a
   chasing spear when other moves kept +1.97. That was a bot's choice, not a wall. No fortress found.
4. **Shuffle.** No game reached the 100-turn or 300-turn draw. Every one of the 6 transcripts and
   all 26 of my games but one ended by extinction. The back-and-forth stretches (seed 0, moves
   37-45; 6400 seed 2, moves 41-45) are a defender waiting while the attacker comes closer, and the
   attacker gets there both times. No shuffle.
5. **Dominant piece / move.** Shields strike spears down any open line in all six directions,
   backwards too (6400 seed 2 ends 46...Wd2xd9, a capture across the whole board). That makes
   every open line from an enemy shield a no-go cell for spears, and the weak bot keeps dropping
   spears onto them (5...Se9-i5 and 8...Sa9-a5 in 25600-v-1600 seeds 1/2). Horse-takes-shield is the
   most common capture, and in seeds 1/2 the winner's spears never captured. But in the 6400
   games spears take horses often (6400 seed 0 has 8 spear captures), and every kind makes the
   deciding capture somewhere: shield (25600 s0, 6400 s1, 6400 s2), horse (25600 s1, s2), spear
   (6400 s0). No single dominant kind.
6. **Dead rules.** Promotion fires and decides (25600 s0: 14. Wf4xa9=H, chosen at +9.96 over
   =S at +7.66, and the winning capture 49. Wh4xc9=S). The spear's range-3 limit on captures
   matters (horses raid from four cells away). Shield no-retreat: I could not show that it ever
   changed a choice. Shields mostly drift forward anyway, and their captures go backwards freely.
   "No legal move loses" never fired. Not a dead seed, but no-retreat is unshown.
7. **Variety, noted.** 25600-v-1600 seeds 1 and 2 are the same game for 15 moves and differ only
   in move 16 and 19 of Black. That is a bot's sameness, not a rule's.

## The five answers

**(i) Yes.** 6400-self seed 2, move 19 Black (after 19. He5xc7, White's horse has just taken a
shield). Black has three captures. Root scores at 1M nodes: 19...Sa9xc7 +5.01 (played),
19...Wd5xd3 +3.56, 19...Sf1xg1 +3.52; the next best is +1.80.
- **19...Sa9xc7** removes the raiding horse but puts the spear on the c-line, where 20. Wc3xc7
  shoots it. Black gives up a spear (and then trades back with Hc9xc7), which brings its spears
  closer to running out.
- **19...Wd5xd3** takes a White spear for nothing now but leaves the horse on c7 among Black's
  remaining shields. It gives up shield safety for a spear.
Both stay clearly winning, and they differ in which of Black's kinds runs short. A second example:
25600-v-1600 seed 0, move 6 White. 6. Hg7-i5 (+8.20, the horse escapes) against 6. Wf3xe4 (+7.63,
played: the shield takes a spear but lets the g7 horse go to 6...Se9xg7).

**(ii) Yes, with a caveat.**
- **Opening:** horse raids on the shield fronts and opening traps (first capture on move 3
  every game).
- **Middle:** three-cornered exchanges counted in kinds (6400 s2, moves 19-20: spear takes
  horse, shield takes spear, horse takes shield).
- **End:** hunting one side's last piece of a kind, where promotion and zugzwang matter
  (25600 s0 plies 27-97; 6400 s2 plies 47-92; 6400 s1 plies 21-80).

Caveat: two of the six transcripts (25600-v-1600 seeds 1/2) have no ending. They are 20-move
shield wipeouts.

**(iii) Yes.** All six end by extinction, and in each the kind count, not total material, sets
the hunt:
- **25600-v-1600 s0:** Black is on its last spear from ply 27. The game is a 70-ply hunt for it,
  ended by a shield capturing as it promotes. Black ends 0/1/1 against 3/3/2.
- **25600-v-1600 s1:** Black's seven shields are eaten by horses, and only spears could guard
  them. Black loses holding 1 spear and all 5 horses.
- **25600-v-1600 s2:** the same: shields extinct, and Black still has all five horses.
- **6400-self s0:** White loses its last horse to a spear (43...Sh1xi1) while still holding a
  spear and a shield. From ply 60 it is 1/1/1 against 2/2/1, a race over which single piece falls first.
- **6400-self s1:** Black is on one spear from ply 21, yet White's spear is taken first
  (40...Wf8xf5). The side that was scarce first wins, which is not "more pieces wins".
- **6400-self s2:** White's last spear is hunted from ply 47 and taken at ply 92 after zugzwang,
  by a shield striking across the whole board.

**(iv) Yes (none found).** I tried:
- the seat, over 26 extra games at four budgets: the lean flips with the budget;
- the horse rush: an opening trap that wins a shield after the natural 1...Ha8-c6, but no forced
  win, and greedy raiding loses;
- last-spear fortresses: all three long hides were broken, by promotion, zugzwang, or the hider
  winning;
- shuffles: no clock draw in 32 games;
- a dominant kind: deciding captures are spread over all three kinds.

Weaknesses that fall short of a break: the opening trap; the 25600-self first-seat 4/4; the
unrestricted backward shield strike making spears very fragile.

**(v) Yes, mostly.** What the reason says and the games show:
- the weak-link decision (6400 s1, where the scarce side wins);
- exchanges that go round the circle (6400 s2, moves 19-20);
- endings that hunt the last of a kind "after a real chase" (25600 s0, 6400 s2);
- promotion buying back a kind.

The opening differs from the reason: it predicts quiet lines and horse development with "no
capture possible on the first moves", but contact comes on move 2-3 and a natural horse move is
already a shield-losing trap. The "timing of the shield push" as the middle game's strategic
question is not visible in these games.
