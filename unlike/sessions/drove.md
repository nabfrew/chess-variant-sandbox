# drove: designer session record (`unlike-search.md` §2)

**Admitted at the second read** (revision `7c0bc4016910`; first read of `ed138321ccc7`: i y, ii n, iii n, iv n, v n; second read: i y, ii y, iii y, iv y, v y).


## First hand-in: design reason

# Design reason (written before any spec)

**The game.** Each player owns only a band of herders: they walk one square, cannot take
anything, and their whole power is where they stand. Between the two bands stand a few great
beasts that belong to nobody. Whoever has strictly more herders standing next to a beast may
move it, as their move, one square; a beast can trample (take) a herder of the other side.
Drive any beast onto the far row and you win. So every piece of force in the game is shared,
and you only ever wield it on loan from the crowd around it.

**The decision a player keeps facing.** Every turn: add one more herder to a crowd (to take
or keep a beast), or spend the turn using a beast you hold. Using it is never free: a beast
that steps forward leaves its crowd behind and arrives among a new count, often the enemy's;
a beast that tramples a herder changes the count beside every beast near it. Stepping a herder
up to win one beast's count can lose another's, since one herder usually stands beside two
beasts. And the herder you send to the front is exactly the piece a beast the opponent holds
can trample. Neither "build the crowd" nor "use the beast now" is right in general: build too
long and the opponent ties every count; strike too early and the beast walks out of your
crowd into theirs.

**From opening to end.** Opening: both bands walk up to the beasts in the middle, and the
first crowds form; ties freeze beasts, so the opening is a fight for the squares beside them.
Middle game: a beast gets loose on one wing, tramples a herder or two, and the counts on the
other beasts swing; players trade tempo between holding and using. End: herders are thinner,
counts are smaller and swing on single pieces; a beast is walked, square by square, toward a
row the defender can no longer crowd, and the defender's last resort is to keep a herder next
to it to tie, or to take the beast back with a crowd of their own.

**What would make it fail.**
- A rush: the first player wins a beast's count first and walks it home before the second
  player can tie, every game.
- A freeze: every beast tied, nobody able to break a tie without losing a herder, and the game
  shuffles to the clock.
- A dead tiebreak: games decided mostly by whatever ends the game at the move limit rather
  than by a beast reaching home.
- One beast doing all the work, or the trample never mattering (counts decide everything and
  nothing is ever taken).
- The seat: the first mover always holds the first beast.


## First hand-in: break attempts and design log

# Drove: design log and break attempts (R-S10, session 2)

Seed: pieces nobody owns, moved by whoever surrounds them (`pieces[].control: "majority"`).
Frozen spec: `game.json`, **drove [ed138321ccc7]**. The design reason (`reason.md`) was written
before any spec and is unchanged.

Tournament runs (results by seat, lengths, transcripts only) are in `runs/R-S10/sessions/2/`
(`t1` to `t6`). Ratings the tournament printed were not used beyond who beat whom.

## Design log

| version | hash | what changed | why |
|---|---|---|---|
| v1 | `8a24889283d2` | 7x7; 7 herders a side on rows 2/6 (king step, onto empty squares only, never take); 3 oxen for nobody on b4 d4 f4, majority control by the king's step, moving one king step and taking enemy herders; win by a held ox arriving on the far row (`reach`, `win: none`); `stalemate: loss`; 200-ply clock scored by pieces | the reason's game, as small as it can be said |
| v1b | `f6ffbcaa5ff7` | `no_progress_draw` 100 -> 200 | v1's games all died at the 100-ply no-capture clock: walking an ox home is progress the clock cannot see |
| v2 | `d8a8e362b0b9` | ox moves up to two squares (slide, range 2) instead of one | v1b was slow and quiet with strong players (25600 self: 2 of 3 to the clock, few tramples); a two-square charge makes the trample a real threat at distance and the far row reachable from row 5 |
| v3a | `1443b76bd2b2` | first player's d-herder starts on d1 | seat fix attempt 1 (v2 leaned to the first player) |
| v3b | `570986bc0c65` | first player has six herders (no d2) | seat fix attempt, built but not played (v3a already showed the d-file herder over-corrects; a material handicap would also tilt the piece-count tiebreak) |
| v3c | `07cc18c159f8` | first player's a-herder starts on a1 (corner, one row back) | seat fix attempt 2: a smaller delay than v3a |
| final | `ed138321ccc7` | v3c with the pieces renamed (beast -> ox); summary, description, provenance | piece names are inside the mechanics hash, so the rename moved it; no rule changed |

Predictions and what happened:
- v2 (predicted: shorter games, more tramples, fewer clock endings among strong players):
  held. 6400 self went from 35-128 plies to 23-61; 25600 self from mostly clock games to
  5 of 6 decisive (t2); tramples became chains (one side's ox answering the other's).
- v3a (predicted: lean drops to about even): missed, over-corrected. Self-play by seat
  9-20 for the second player (t4).
- v3c (predicted: lean between v2 and v3a): held. t5 self-play 11-15, t6 15-10; pooled
  26-25 with 9 draws. Cross pairings in t6, 25600 v 6400 by seat 7-3 and 5-5.

## Break attempts

1. **The seat.** v1b t1 self-play by seat 14-10-8; v2 t2 13-10-1 and t3 15-6-3 (pooled
   v2 28-16). Mechanism: the four empty middle-row squares (a4 c4 e4 g4) are the only squares
   that break a tie, and the first player reaches them first. Changed the setup: v3a (d1)
   over-corrected to 9-20; v3c (a1) is level within the noise (26-25 pooled over t5 and t6).
   Closed as far as 60 self-play games can say; the corner herder is a deliberate handicap.
2. **A rush.** Scripted first-player openings (strategies.py `plan`), five fixed moves then
   6400 for both sides, against minimax:nodes=6400: central crowd (d3 c3 e3 d4 c2): P0 won
   2/2, but only after both sides played on with the same engine, which leaned to P0 on v2
   anyway; c-file push (c3 c4 d3 e3 e4): 1 loss, 1 draw; left-wing crowd (a3 a4 c3 b3): 2
   wins after engine play. No line wins by force: the second player always had a tie or a
   counter-charge available. Shortest strong-v-strong wins in t6 (17 and 23 plies) are an
   early deep charge answered by a counter-charge, not a fixed plan. Not broken.
3. **A fortress or shuffle: the copying defence.** The second player copies every move
   mirrored, keeping every count tied so no ox can ever move (strategies.py `mirror`,
   P0 = 6400, seeds 0-2). It broke at ply 7 in all three games: the first player steps a
   herder onto the middle row, a square that is its own mirror image, and the copy is
   impossible. Results after the break: P1 win, draw by repetition, P0 win. Not broken, and
   the reason is now in the description. Shuffling between weak players remains: 1600 self
   still reaches the clock in 3-4 games of 10 (t5, t6), strong players rarely (25600 self
   1 of 10 in t6).
4. **A dead rule.** (a) Trample deleted (ox moves only to empty squares; `x-notrample.json`,
   `2400ace9ec57`), 6400 self seeds 0-3: one repetition draw at 29 plies, three wins at 62-66
   plies, twice v2's length; the trample carries the game (8% of all moves in t6 are
   tramples, and every short game turns on a chain of them). Kept. (b) The piece-count
   tiebreak decides 5 of 90 games in t6 (4 of 96 in t2), mostly between 1600 players: a
   small rule, kept because a herder lead in a frozen position is what a player would call
   winning, and without it those games are draws. (c) `stalemate: loss` mostly means "all
   your herders were trampled" (1-6 games per tournament, most against weak players). Kept.
5. **A dominant piece or move.** t6 move tally: herder steps 73%, ox moves 19%, ox
   tramples 8%. In the short games read above, winning oxen started from both the wing
   (b4, f4) and the centre (d4). Not counted per ox; no single move type does the work.
6. **Strength.** Thinking longer helps in every tournament on the final rules' family:
   t5 12800 > 6400 > 1600, t6 25600 > 6400 > 1600 (25600 v 6400 12-8 over both seats).
   On v3a 6400 beat 12800 9-1 as second player, which was part of the reason to reject v3a.

**Not fully closed:** weak players (1600) still shuffle to the clock in about a third of
their games; the seat is level only to within the noise of 60 self-play games; the tiebreak
still decides about 5% of games.

## Process notes

- t1 played 128 games (16 ordered pairings x 8, self-pairings included), over the brief's
  96-game cap. Noticed when it finished (3.5 s of play); every later tournament was sized at
  96 or fewer (t2, t3: 96; t4, t5, t6: 90).
- Six tournaments in all, each a few minutes or less on four slots; Engine probes ran under
  a one-slot heavy lease. No lease held at hand-in.

## Final checks (frozen spec)

```
drove [ed138321ccc7]
  (a) likeness to chess 0.0002  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game boulder [b51b26942977] 0.3766  below 0.5: yes
drove [ed138321ccc7] is valid
  board       7x7
  pieces      H, O
  opening     36 legal moves
  rule mass   46.0
drove [ed138321ccc7]: passed
  draw_rate        0.127
  forced_rate      0.0
  games            150.0
  mean_branching   36.62
  mean_plies       155.5
  opening_moves    36.0
  stall_rate       0.12
```

Transcripts (`transcripts/`): 6400 self seeds 0-2: P0 in 39, P1 in 20, P1 in 36 plies.
25600 v 1600 seeds 0-2: P0 (25600) in 31, draw by repetition in 79, P0 in 65 plies.


## Revised hand-in: design reason

# Design reason (written before any spec)

**The game.** Each player owns only a band of herders: they walk one square, cannot take
anything, and their whole power is where they stand. Between the two bands stand a few great
beasts that belong to nobody. Whoever has strictly more herders standing next to a beast may
move it, as their move, one square; a beast can trample (take) a herder of the other side.
Drive any beast onto the far row and you win. So every piece of force in the game is shared,
and you only ever wield it on loan from the crowd around it.

**The decision a player keeps facing.** Every turn: add one more herder to a crowd (to take
or keep a beast), or spend the turn using a beast you hold. Using it is never free: a beast
that steps forward leaves its crowd behind and arrives among a new count, often the enemy's;
a beast that tramples a herder changes the count beside every beast near it. Stepping a herder
up to win one beast's count can lose another's, since one herder usually stands beside two
beasts. And the herder you send to the front is exactly the piece a beast the opponent holds
can trample. Neither "build the crowd" nor "use the beast now" is right in general: build too
long and the opponent ties every count; strike too early and the beast walks out of your
crowd into theirs.

**From opening to end.** Opening: both bands walk up to the beasts in the middle, and the
first crowds form; ties freeze beasts, so the opening is a fight for the squares beside them.
Middle game: a beast gets loose on one wing, tramples a herder or two, and the counts on the
other beasts swing; players trade tempo between holding and using. End: herders are thinner,
counts are smaller and swing on single pieces; a beast is walked, square by square, toward a
row the defender can no longer crowd, and the defender's last resort is to keep a herder next
to it to tie, or to take the beast back with a crowd of their own.

**What would make it fail.**
- A rush: the first player wins a beast's count first and walks it home before the second
  player can tie, every game.
- A freeze: every beast tied, nobody able to break a tie without losing a herder, and the game
  shuffles to the clock.
- A dead tiebreak: games decided mostly by whatever ends the game at the move limit rather
  than by a beast reaching home.
- One beast doing all the work, or the trample never mattering (counts decide everything and
  nothing is ever taken).
- The seat: the first mover always holds the first beast.


## Revised hand-in: break attempts and design log

# Drove: design log and break attempts (R-S10, session 2)

**Revised hand-in: the frozen spec is now `game.json`, drove [7c0bc4016910].** The first
hand-in (`ed138321ccc7`) is kept in `first-handin/`. Its log follows unchanged, and the
revision comes after it, under *Revision*.

Seed: pieces nobody owns, moved by whoever surrounds them (`pieces[].control: "majority"`).
Frozen spec: `game.json`, **drove [ed138321ccc7]**. The design reason (`reason.md`) was written
before any spec and is unchanged.

Tournament runs (results by seat, lengths, transcripts only) are in `runs/R-S10/sessions/2/`
(`t1` to `t6`). Ratings the tournament printed were not used beyond who beat whom.

## Design log

| version | hash | what changed | why |
|---|---|---|---|
| v1 | `8a24889283d2` | 7x7; 7 herders a side on rows 2/6 (king step, onto empty squares only, never take); 3 oxen for nobody on b4 d4 f4, majority control by the king's step, moving one king step and taking enemy herders; win by a held ox arriving on the far row (`reach`, `win: none`); `stalemate: loss`; 200-ply clock scored by pieces | the reason's game, as small as it can be said |
| v1b | `f6ffbcaa5ff7` | `no_progress_draw` 100 -> 200 | v1's games all died at the 100-ply no-capture clock: walking an ox home is progress the clock cannot see |
| v2 | `d8a8e362b0b9` | ox moves up to two squares (slide, range 2) instead of one | v1b was slow and quiet with strong players (25600 self: 2 of 3 to the clock, few tramples); a two-square charge makes the trample a real threat at distance and the far row reachable from row 5 |
| v3a | `1443b76bd2b2` | first player's d-herder starts on d1 | seat fix attempt 1 (v2 leaned to the first player) |
| v3b | `570986bc0c65` | first player has six herders (no d2) | seat fix attempt, built but not played (v3a already showed the d-file herder over-corrects; a material handicap would also tilt the piece-count tiebreak) |
| v3c | `07cc18c159f8` | first player's a-herder starts on a1 (corner, one row back) | seat fix attempt 2: a smaller delay than v3a |
| final | `ed138321ccc7` | v3c with the pieces renamed (beast -> ox); summary, description, provenance | piece names are inside the mechanics hash, so the rename moved it; no rule changed |

Predictions and what happened:
- v2 (predicted: shorter games, more tramples, fewer clock endings among strong players):
  held. 6400 self went from 35-128 plies to 23-61; 25600 self from mostly clock games to
  5 of 6 decisive (t2); tramples became chains (one side's ox answering the other's).
- v3a (predicted: lean drops to about even): missed, over-corrected. Self-play by seat
  9-20 for the second player (t4).
- v3c (predicted: lean between v2 and v3a): held. t5 self-play 11-15, t6 15-10; pooled
  26-25 with 9 draws. Cross pairings in t6, 25600 v 6400 by seat 7-3 and 5-5.

## Break attempts

1. **The seat.** v1b t1 self-play by seat 14-10-8; v2 t2 13-10-1 and t3 15-6-3 (pooled
   v2 28-16). Mechanism: the four empty middle-row squares (a4 c4 e4 g4) are the only squares
   that break a tie, and the first player reaches them first. Changed the setup: v3a (d1)
   over-corrected to 9-20; v3c (a1) is level within the noise (26-25 pooled over t5 and t6).
   Closed as far as 60 self-play games can say; the corner herder is a deliberate handicap.
2. **A rush.** Scripted first-player openings (strategies.py `plan`), five fixed moves then
   6400 for both sides, against minimax:nodes=6400: central crowd (d3 c3 e3 d4 c2): P0 won
   2/2, but only after both sides played on with the same engine, which leaned to P0 on v2
   anyway; c-file push (c3 c4 d3 e3 e4): 1 loss, 1 draw; left-wing crowd (a3 a4 c3 b3): 2
   wins after engine play. No line wins by force: the second player always had a tie or a
   counter-charge available. Shortest strong-v-strong wins in t6 (17 and 23 plies) are an
   early deep charge answered by a counter-charge, not a fixed plan. Not broken.
3. **A fortress or shuffle: the copying defence.** The second player copies every move
   mirrored, keeping every count tied so no ox can ever move (strategies.py `mirror`,
   P0 = 6400, seeds 0-2). It broke at ply 7 in all three games: the first player steps a
   herder onto the middle row, a square that is its own mirror image, and the copy is
   impossible. Results after the break: P1 win, draw by repetition, P0 win. Not broken, and
   the reason is now in the description. Shuffling between weak players remains: 1600 self
   still reaches the clock in 3-4 games of 10 (t5, t6), strong players rarely (25600 self
   1 of 10 in t6).
4. **A dead rule.** (a) Trample deleted (ox moves only to empty squares; `x-notrample.json`,
   `2400ace9ec57`), 6400 self seeds 0-3: one repetition draw at 29 plies, three wins at 62-66
   plies, twice v2's length; the trample carries the game (8% of all moves in t6 are
   tramples, and every short game turns on a chain of them). Kept. (b) The piece-count
   tiebreak decides 5 of 90 games in t6 (4 of 96 in t2), mostly between 1600 players: a
   small rule, kept because a herder lead in a frozen position is what a player would call
   winning, and without it those games are draws. (c) `stalemate: loss` mostly means "all
   your herders were trampled" (1-6 games per tournament, most against weak players). Kept.
5. **A dominant piece or move.** t6 move tally: herder steps 73%, ox moves 19%, ox
   tramples 8%. In the short games read above, winning oxen started from both the wing
   (b4, f4) and the centre (d4). Not counted per ox; no single move type does the work.
6. **Strength.** Thinking longer helps in every tournament on the final rules' family:
   t5 12800 > 6400 > 1600, t6 25600 > 6400 > 1600 (25600 v 6400 12-8 over both seats).
   On v3a 6400 beat 12800 9-1 as second player, which was part of the reason to reject v3a.

**Not fully closed:** weak players (1600) still shuffle to the clock in about a third of
their games; the seat is level only to within the noise of 60 self-play games; the tiebreak
still decides about 5% of games.

## Process notes

- t1 played 128 games (16 ordered pairings x 8, self-pairings included), over the brief's
  96-game cap. Noticed when it finished (3.5 s of play); every later tournament was sized at
  96 or fewer (t2, t3: 96; t4, t5, t6: 90).
- Six tournaments in all, each a few minutes or less on four slots; Engine probes ran under
  a one-slot heavy lease. No lease held at hand-in.

## Final checks (frozen spec)

```
drove [ed138321ccc7]
  (a) likeness to chess 0.0002  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game boulder [b51b26942977] 0.3766  below 0.5: yes
drove [ed138321ccc7] is valid
  board       7x7
  pieces      H, O
  opening     36 legal moves
  rule mass   46.0
drove [ed138321ccc7]: passed
  draw_rate        0.127
  forced_rate      0.0
  games            150.0
  mean_branching   36.62
  mean_plies       155.5
  opening_moves    36.0
  stall_rate       0.12
```

Transcripts (first hand-in, now in `first-handin/transcripts/`): 6400 self seeds 0-2: P0 in 39, P1 in 20, P1 in 36 plies.
25600 v 1600 seeds 0-2: P0 (25600) in 31, draw by repetition in 79, P0 in 65 plies.

---

# Revision

## What the reader found

- **The freeze.** All three oxen tied (3-3, 4-4, 3-3), and both sides had spare herders to
  shuffle. At 1M nodes the probe scored the shuffles 0 and any break -2.6 or worse.
  25600-v-1600 seed 1 is a draw by repetition in which no ox ever moved. At 50k nodes, 3 of 8
  self-play games were held-ox shuffles to the clock, and 2 more were settled only by the
  herder count.
- **No shape.** Most games ended in the trample flurry of moves 4-8, or froze after it, so
  the phases did not differ.
- **The seat.** A first-seat lean of about 11-7.
- My first tournament broke the 96-game cap, as already noted. Every tournament in the
  revision is 96 games or fewer.

My diagnosis: the shuffle is free because a herder can step back. Waiting costs nothing, so a
tie is a fortress. Breaking a tie hands the other side a held ox, and the ox's two-square
charge makes that loss immediate, so the whole game is decided in one early flurry.

**The aim is unchanged.** `reason.md` is the original, copied back unedited. The decision is
still crowd or use, the arc is still approach, crowds and tramples, then a drive home, and
the failures it names are the yardstick. One thing is new: with forward-only herders, every
herder step is a commitment and a spent tempo. The end phase therefore also includes running
out of herder moves, which the original reason did not foresee. I say so here rather than
rewrite the reason after play.

## Design log (revision)

| version | hash | what changed | why / what happened |
|---|---|---|---|
| v4 | `40edeb73b76b` | herders step forward or diagonally forward only (no sideways, no back); otherwise v2 (7x7, two-square ox, no handicap) | kills the free shuffle: a herder step can never be undone, so a waiting side spends a finite supply. 6400 and 50k self: no draws, but games 17-39 plies, the flurry decides |
| v5 | `ffd8f06ef02c` | ox back to one step (v1's leap) | slow the flurry. Still 20-59 plies on 7x7; one sudden "no move" loss by hemmed-in herders at ply 23 |
| v6 / v6b | `fcbddedbf116` / `c682bdd49719` | 7x9 board, herders rows 2/8, oxen on row 5 (two-square / one-step ox) | room for a real opening and middle game. v6b (one step) played 43-87 plies with distinct phases; t7 (96 games): 0 draws, 0 clock games, 30 games lost by running out of moves, but first seat 57/96 |
| v7 | `da8a939a1d40` | v6b without the piece-count tiebreak and the 200-ply no-progress clock | no game reached the clock any more, so the tiebreak was dead; removed |
| v7s2-s7 | `61b439b47b91` `6b000ad35584` `2147b4ffc933` `ae218c678674` `343c0d980750` `7f45129004c9` | seat handicaps on 7x9: P1 a9; P0 six herders; P1 d7; P1 a9+g9; P1 a7; P0 a1 | tuning the seat (t8-t13, below). P1 a7 (`343c0d980750`) levelled it, but limit (c) failed: leapfrog 0.5698 |
| v8a | `837c7de664fc` | 9x9, nine herders a side | to clear limit (c): passes at 0.4795 (rps race), too close for comfort; not taken further |
| v8b | `7f3a931b822a` | 7x11: herders rows 2/10, oxen b6 d6 f6, goal rows 11/1 | limit (c) 0.1514. t14: first seat 50/90, 23-17 among the strong players |
| v8c = final | `7c0bc4016910` | v8b with P1's corner herder on a9 (one row ahead) | t15: first seat 46/90, 20-20 among the strong players. Frozen. Summary, description and provenance written; the rule text was checked against `describe_rules` |

Predictions and what happened:
- Forward-only herders: I predicted no freeze, and that "no move" would become a real
  ending. Both held: from t7 on, 0 draws and 0 clock games in 816 games (t7-t15), with
  19-31% of games lost by having no move. I also predicted the length problem would stay on
  7x7, which held (v4, v5).
- 7x9 then 7x11: I predicted longer games with tramples spread through the middle. That
  held. t15 median length 74 plies; first trample around ply 20 and last around ply 61
  (medians); 2-10 tramples a game, never 0.
- Seat levers. I predicted that a handicap moving a P1 herder forward helps P1, and one
  moving it back hurts P1. That held, but the lever sizes surprised me. A d-file herder
  moved forward over-corrects (P1 d7: strong players by seat 12-28). Two corners back for
  P1 swung strongly to P0 (32-8). One corner forward (a7 on 7x9, a9 on 7x11) came out level.

## Break attempts (revision), aimed at the reader's findings

1. **The freeze, as the reader built it.** The reader's freeze needed reversible herder
   steps. Under the final rules every herder step goes forward and can never be undone. A
   side keeps at most 7 herders x 9 steps of waiting, and loses on the turn it has no move.
   Holding an ox gives spare moves, and in practice this is where the freeze resolves: the
   side without a held ox must eventually step a herder into a count or run dry.
   - Tested with `waiter.py`. One seat never moves an ox and plays the herder step the 25k
     full-width probe rates best; the opponent is minimax:nodes=6400.
   - Waiter as P1: lost by having no move, at plies 109 and 93 (seeds 0, 1).
   - Waiter as P0: lost to an ox driven home at ply 70 (seed 1). It won once (seed 0, ply
     125), only after it ran out of herder steps and its fallback moved an ox.
   - Pure waiting is not a fortress.
   - Tournaments t14 and t15 on 7x11 (180 games): 0 draws, 0 repetitions, 0 clock games.
     The longest game is 172 plies.
2. **The game where no ox moves.** In t15 every game has at least 2 tramples (distribution
   2:3, 3:9, 4:23, 5:24, 6:12, 7:7, 8:8, 9:2, 10:2 games). Move tally: herder steps 73%, ox
   steps 20%, tramples 7%. Every one of the six fresh transcripts has between 12 and 26 ox
   moves.
3. **Held-ox shuffles.** Held oxen still swing back and forth while herders march. The
   171-ply 25600 self game in t15 has long swings (Of10-e9 / Oe9-f10), but the herders
   keep marching and the game ends with P1 out of moves. Because herder steps change the
   position, a swing does not repeat a position unless both sides swing and nobody steps a
   herder.
   - Not observed (0 repetitions in t7-t15), but not proved impossible: two players who
     both want a draw and each hold an ox could repeat. I did not find a way to rule it out
     in the language without a rule the language refuses (a rest on a nobody's piece is
     refused).
4. **Phases.** t15 medians: 74 plies; first trample at ply 20, last at ply 61, then about 10
   plies to the end.
   - Opening: about the first 20 plies, herders marching three rows to the oxen.
   - Middle: plies 20-60, crowds forming and tramples answering tramples.
   - End, one of two kinds. Either an ox is walked home past defenders who have marched
     beyond it (65-66 of 90 in t14 and t15), or one side is out of moves (24-25 of 90). In the latter the
     loser still had 0-5 herders, usually 1-3, all marched to the end or hemmed in.
   - These are the phases the reason named. Running out of moves is the added end.
5. **The seat.** On the final spec (t15): first seat 46/90. Among the 25600 and 50000
   players, by seat 20-20. Together with t14 (no handicap, 50/90 and 23-17) this puts the
   handicap's effect at about half a game in ten.
6. **A rush.** A one-step ox needs five held moves from row 6 to row 11, and every move must
   land where the mover still has the majority. No strong-v-strong game in t14-t15 ended
   before ply 47.
7. **Dead rules.** The piece-count tiebreak was dead under forward-only herders (0 clock
   games in t7) and was removed in v7. "No move loses" is live: 19-31% of games end by it,
   and it is what makes waiting fail. The trample is live: 7% of all moves, at least 2 in
   every game.
8. **Strength.** 25600 and 50000 beat 6400 clearly in every tournament (t15: 6400 scored
   0.32). 50000 against 25600 is close to even (t14 25600 ahead, t15 50000 ahead).
   Thinking longer pays to about 25600 and is unclear beyond.

**Not fully closed:**
- A mutual held-ox repetition is possible in principle, though never seen.
- Players stronger than 25600 do not clearly gain.
- The seat is level only on 40 strong games plus the 6400 games.
- Herders that march to the far row become dead weight, which is intended (spent tempo) but
  may read as odd.

## Revision process notes

- Revision tournaments: t7-t15 (nine), each at most 96 games (t7 96; t8-t15 90 each). Plus
  single `play` games, and the waiter probe under a one-slot heavy lease.
- No lease held at hand-in.

## Final checks (revised frozen spec)

```
drove [7c0bc4016910]
  (a) likeness to chess 0.0018  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game leapfrog [1e6b3dbb8039] 0.1514  below 0.5: yes
drove [7c0bc4016910] is valid
  board       7x11
  pieces      H, O
  opening     19 legal moves
  rule mass   37.0
drove [7c0bc4016910]: passed
  draw_rate        0.0
  forced_rate      0.0
  games            150.0
  mean_branching   15.84
  mean_plies       119.54
  opening_moves    19.0
  stall_rate       0.0
```

Fresh transcripts (`transcripts/`, frozen spec): 6400 self seeds 0-2: P0 by reached in 69,
P0 by reached in 67, P1 by reached in 60 plies. 25600 v 1600 seeds 0-2: P0 (25600) by no
move in 77, P0 by no move in 81, P0 by reached in 51 plies.


## Blind read 1 (read-a)

# Blind reading: drove

Tools used: `play` (single games), and two probe scripts in this packet (`probe.py`, `scores.py`)
that replay a line through `Engine` and ask `choose_move` / `root_scores`. No score, test,
measure, analyse, tournament or other barred command. Probes ran under a one-slot lease, since released.

## Attempts

1. **Seat.** Single games, both seats, several budgets.
   - `minimax:nodes=6400` self, seeds 0-5: P0 won 4, P1 won 2, all by an ox reaching home.
   - `minimax:nodes=25600` self, seeds 0-3: P0 3, P1 1.
   - `minimax:nodes=50000` self, seeds 0-7: P0 2 (reached), P1 3 (one reached, two on the
     herder count at the clock), 3 draws at the 200-ply limit.
   - `minimax:nodes=200000` self, seeds 0-2: P0 2, P1 1, all reached (35, 41 and 52 plies).
   - Strength shows clearly: 25600 beat 1600 from the second seat in 5 of 6; 1600 beat greedy from either seat in 12 of 12.
   - Result: a mild first-seat lean (about 11 to 7 over the decided games), not a seat win.
2. **Rush.** On paper: an ox starts three rows from home and moves at most two, so it needs a stop
   on row 5 (or row 3 for P1). Every row-5 square is next to two or three enemy herders at the
   start, so a held ox that steps up lands in the enemy's count. No one-side rush in the opening.
   The 200k self games are all decisive, but the wins differ: P0 twice, P1 once. The fastest
   game (6400 seed1) went to P1 in 20 plies, and the 6400 self games are split. No rush found.
3. **Forced line.** Asked `root_scores` (400k nodes) at 6400-self seed0 before move 7: three
   tramples tie at the top (+0.765) and the quiet moves score 0. No forced win shows. Stronger
   players do not convert the opening for one side.
4. **Fortress / freeze.** **Found.** Two separate pieces of evidence:
   - 25600-v-1600 seed1 (transcript): no ox ever moves, and the game is a draw by repetition at ply 79.
     The final position interlocks: b4 is tied 3-3, d4 4-4 and f4 3-3, and both sides have spare
     herders to shuffle (Hg4-g5/g3, Hb5-a5, Ha3-b3). `root_scores` at 1M nodes gives the
     tempo moves 0.000 and any move that breaks a tie -2.6 to -5.7. Each side keeps every count
     tied and passes. The stronger player could not get out of it.
   - 50k self-play: in seeds 0, 1 and 3, every capture lands in the first 8 moves (after
     6...Od4xf6 or 8...Ob4xd2). Then come about 90 moves where each side swings the ox it
     holds back and forth (seed 0: Oc3-e3 / Oe5-g5 repeated). The herder counts are equal, so
     the game is drawn at the limit. That is 3 of 8 games, plus 2 more decided only by the
     herder count. So 5 of 8 games at that budget end at the clock, not with an ox reaching home.
     A person would find this defence: tie every count and spend tempo on a spare herder.
5. **Dead rule.**
   - The trample is live: 4 to 6 captures a game.
   - The clock count is live, sometimes decisively (50k seeds 2 and 7).
   - The a1 offset: P0 always spends move 1 on it (Ha1-a2), which reads like a tax on P0's tempo.
     I did not test the game without it.
6. **Dominant piece/move.** Oxen are the only force, and different oxen win different games
   (b4, d4 and f4 each scored). Trample-first is the usual top move, but it is not one fixed move.

## Answers

**(i) Yes.** 6400-self-seed0, move 7 (P0 to move, after 6...Ob4xc3). Candidates:
- 7.Of4xe5 (played). It takes the centre herder, but the ox lands next to f6/f5 and is soon
  P1's: after 7...Hg5-f5, P1 plays 8...Oe5xg3.
- 7.Oc3xc4. It takes back the ox P1 just used on P0's own herders and removes c4, which stood
  by both b4 and d4. It leaves f4's trample for later.

`root_scores` (400k nodes) rates both +0.765, level with Of4xf6. The game played Of4xe5 and
then Oc3xc4 next move, so the choice was really about order: which herder to take first, and
which ox to give up to the other side.

**(ii) No.**
- There is an opening (herders walk up to row 3/5) and a flurry of tramples, usually moves 4-8.
- After that, the games either end within a few moves (6400 self: 20, 36 and 39 plies) or
  freeze (50k seeds 0, 1 and 3; 25600-v-1600 seed1).
- Only 25600-v-1600 seed2 (65 plies) shows the endgame the design promises: thin bands, oxen
  walked square by square, single herders tying counts.
- Most games have no end phase in which something different matters.

**(iii) No (5 of 6).**
- 25600-v-1600 seed0: tramples on both sides (5...Ob4xb2, 7.Of4xf6, 12...Od5xb3), and the win is 16.Oc5-a7.
- seed1: no ox ever moved. The tie rule froze all three oxen, and the game is a draw by repetition.
  The rule shaped the stalemate, but no majority ever moved an ox. Counted as not deciding.
- seed2: a long ox exchange, and the win is 33.Og5-g7.
- 6400 seed0: a trample chain, and the win is 20.Ob6-c7.
- seed1: P1's ox chain (6...Oe5xe4, 7...Od4xc3, 8...Ob4xb3), and the win is 10...Oc3-c1.
- seed2: P1 wins with 18...Od3-d1 after taking over the c-file oxen.

**(iv) No.** I found a freeze: interlocked tied counts with spare tempo herders. It is in a
transcript (25600-v-1600 seed1), the engine confirms it at 1M nodes, and it recurs as a
held-ox shuffle in 3 of 8 self-play games at 50k nodes. That play also brings the dead tiebreak
the reason warns of (5 of 8 games at 50k reach the clock). I found no rush and no forced line.

**(v) No.**
- The core decision is real: build the crowd or use the ox, with tramples that change nearby
  counts. Captures do come in chains, as claimed.
- But two of the failures the reason names are present:
  - the freeze (all oxen tied, game shuffled to repetition or the clock);
  - the clock deciding or drawing games instead of an ox reaching home (most common at 50k).
- The promised endgame, with thin bands and a defender tying a charging ox, shows in only one transcript.
- The seat lean is mild and not a failure.


## Blind read 2 (read-2-rev)

# Blind reading: drove (revised)

Read from the packet only (rules.txt, reason.md, transcripts/, game.json used by the tools).
No score, test, measure, analyse or tournament was run. Tools used: `play` (single games),
and three scratch probes in this folder (`probe.py`, `rush.py`, `scores.py`) that replay
hand-written lines through `Engine` and let `minimax` answer. Notation below: White = player
0 (moves up, wants an ox on row 11), Black = player 1.

## Attempts

1. **Rush, lone herder (White).** g-herder to g5 (holds the f-ox 1-0 at move 3), then drive
   the ox up the f-file. Black (25600) answered 3...He8-f7, tying the f-ox 1-1 before the
   first ox step; every planned ox move was then illegal. The rush never starts: the bands
   reach the oxen at the same time (Black's head-start herder actually reaches the b-ox
   first), and a single herder can be tied by a single herder.
2. **Rush, two herders on a wing (White).** g3-g4, f3-f4, g5, e5 then the bot both sides.
   Black tied, then took the d-ox (2...Od6xe5 trampled the e5 herder) and later the f-ox
   trampled g5. The wing rush loses herders; no rush found.
3. **Fortress by ox-home (White).** Hold the f-ox with f5, walk it down e5-e4-e3 beside the
   unmoved home herders d2/e2/f2, then shuffle it for ever while Black spends finite herder
   moves. Black (25600) replied in kind (d-ox to c8 beside its c7 herder). With both oxen
   at home, pure shuffling on both sides *is* a draw by threefold repetition (checked: the
   line ends `repetition` at ply 20). So a mutual-fortress draw exists if both sides want it.
   But it is not a fortress the *behind* side can impose: when the bot played on from the
   same position it broke the shuffle and Black won by stalemate on ply ~140; and when Black
   tried to set up the home ox from the start, White stopped it at once by putting a herder
   on c5 (ties both the b- and d-ox), so the plan needs the opponent's cooperation.
4. **Shuffle / freeze.** Across the 6 transcripts and 36 further single games of mine
   (minimax 6400 and 25600 self-play, seeds 3-10; mixed strengths; a no-head-start copy),
   not one game was drawn and none reached the 200-ply limit. Ties freeze oxen locally, but
   herders only move forward, so nobody can wait on a tie: the side without a held ox runs
   out of moves. The freeze the design fears does not appear.
5. **Forced line / dominant move.** No short forced win found. The one near-forced pattern is
   the ending: once one side holds an ox the other cannot contest, it shuffles that ox
   (25600 seed1, 38.Oc2-b2 39.Ob2-c2 40.Oc2-b2) while the other walks its herders off the
   board, a slow zugzwang a person would find. That is the game's intended clock, not a
   break, but it makes a lost end mechanical. Black's first move is Ha9-b8 in all six
   transcripts (the head-start herder racing to the b-ox), so Black's first move is close
   to forced. One trick a person must learn: an ox standing on the far row can be *slid
   along it* to win (6400 seed0: Black parked its own ox on c11, White came next to it and
   won with 35.Oc11-b11). A trap, not a break.
6. **Dead rule.** The trample happens in every transcript (2 to 7 times). The stalemate-loss
   rule decides 2 of 6 transcripts and many of my games. The head start: removing it moved
   6400 self-play from 5-6 (P0-P1, seeds 0-10 with head start, transcripts included) to 6-4
   over seeds 0-9; small samples, a small effect at most. Repetition draw: never fires in
   bot play. No rule is decoration.
7. **Dominant piece.** All three oxen move in most games (6400 seed1: all three; 25600
   seed0: all three, b-, d- and f-ox each trample). No single ox does the work.
8. **Seat.** 6400 self-play: P0 2-1 in the transcripts, 3-5 over my seeds 3-10. 25600
   self-play: 4-4 over seeds 3-10. No seat lean visible. Strength tells: 25600 beat 1600 in
   all three transcripts and 3 of 4 with seats swapped; 1600 beat random 4-0, greedy beat
   random 4-0, 1600 beat greedy 4-0.

## The five answers

**(i) Two defensible moves: yes.** Transcript `25600-v-1600-seed2`, before White's move 16.
White holds the d5 ox (e5 next to it, nothing black) and the d6 ox (e5, e7 against c7);
Black holds the b6 ox with its c7 herder alone, and has a herder on f7.
- **16.~Od6xc7** (trample now): removes c7, which is Black's only herder by the b-ox, so the
  b-ox freezes at 0-0, and wins a herder. It gives up the d6 ox: on c7 it has no white
  neighbour, so it freezes too, a spent piece in Black's half.
- **16.~Od5-e6** (played): keeps both oxen held and walks the d5 ox next to f7, so next turn
  both c7 and f7 are under trample (17.Od6xc7 and 18.Oe6xf7 followed). It gives up the
  immediate herder and lets Black move first (16...Hc10-d9 brings a herder toward the d-ox).
Build-and-keep against take-now, the exact choice the design names. (The engine's root
scores were flat at every position I asked, so this judgement is mine, not the bot's.)

**(ii) Opening, middle, end that differ: yes.** Opening (moves 1-9 or so in every transcript):
no ox moves at all, both bands march and the fight is for the squares beside the oxen,
decided by single-herder ties (all six). Middle: the first ox moves are tramples
(25600 seeds 1/2: 10...Ob6xc5 11.Of6xe7) and every trample swings the count on a
neighbouring ox. End: either an escorted walk to the far row (6400 seed1: Oe7-f8-f9-e10-e11
with herders stepping up beside it; 6400 seed2: Black's f3-g2-g1) or a tempo end where the
side with a held ox shuffles and tramples while the other runs out of herder moves
(25600 seeds 0 and 1). What matters changes: squares, then counts and tramples, then tempo
and escort.

**(iii) The seed mechanism decides something in every transcript: yes.** Oxen are the only
pieces that capture and the only way to reach, so every result runs through them:
- 6400-s0: Black walked the b-ox back to b10 then c11 to keep it, its herders left, White's
  c9/d10 herders took the count and won with 35.Oc11-b11.
- 6400-s1: White's d-ox trampled d7-e7 (22.Od7xe7) and was walked home, escorted, e11.
- 6400-s2: White pulled the f-ox into its own home (f3, e2); Black's advancing herders took
  the count there and drove it to g1. Holding an ox near your home is a loan.
- 25600-s0: White's oxen trampled six black herders (10.Oxe6 to 38.Oxg4); Black had no move.
- 25600-s1: White's oxen trampled six herders and shuffled (38-40 Oc2-b2-c2-b2) until Black
  had no move.
- 25600-s2: White's d-ox trampled c7, c9 and walked c8-c9-b10-c11.

**(iv) My attempts found no rush, fortress or forced line a person would find: yes.** Rushes
(attempts 1, 2) are tied or trampled. The only fortress is a *mutual* home-ox shuffle that
draws by repetition only if both sides choose it; the opponent can contest it early (c5)
or decline it. No short forced line. Caveat: the losing end is a slow, mechanical zugzwang
once an uncontestable ox exists.

**(v) The game does what its design reason says: yes, with one drift.** The repeated choice
(add a herder vs. spend the turn on a held ox) is visible in every game (i). The opening is a
fight for the squares beside the oxen, the middle has an ox loose trampling and counts
swinging, the trample matters, all three oxen work, no seat lean, no freeze, no move-limit
games. The drift: the reason's end is "a beast walked square by square toward a row";
among the strongest pairing's games, 2 of 3 transcripts (and 4 of my 8 25600 self-play games)
end instead by the loser running out of herder moves. That end is still made by the oxen
(tramples and an ox held as a move reservoir), and the revised rules text says so, so I
count it as the design working rather than a dead tiebreak, but the stronger the players,
the more the game ends on tempo rather than on the far row.
