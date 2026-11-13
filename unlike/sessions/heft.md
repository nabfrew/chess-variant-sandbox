# heft: designer session record (`unlike-search.md` §2)

**Not admitted; the session failed twice and adds no game** (§2.2). First hand-in `f596ad2440e2`, blind read 1: i yes, ii no, iii yes, iv no, v no. Revised hand-in `068d83b22de1`, blind read 2 (a new reader): i yes, ii yes, iii yes, iv yes, v no.


## First hand-in: design reason

# Design reason (written before any spec)

Seed: pushing by weight, with no capture.

## The game in one line

A shoving match on a small hexagonal ring. Nothing is ever taken: a piece that moves onto an
enemy shoves it, together with everything standing behind it in that line, one cell on, and a
piece shoved off the ring is out. Push enough of the enemy out, or hold the middle of the ring
when the shoving stalls, and you win.

## The decision a player keeps facing

**Stack or strike.** Every shove is a sum: it goes through only if the shover's strength is at
least the total weight of the line in front of it. So a piece standing alone is light and easy
to throw, and a piece with friends lined up behind it is heavy and safe. The pieces that hit
hardest are not the pieces that make a line heavy: a heavy, strong piece that is busy bracing a
friend is not shoving anyone, and a quick light piece that runs out to strike leaves the line it
was part of lighter and itself alone near the edge. Every turn asks whether to add weight to
your own lines (safe, slow, gives up ground) or send a piece out to shove (gains ground or a
knock-out, but thins what is behind it). Because lines run in six directions on a hex board, a
group braced against a shove from one side is usually open along another, so no shape is safe
from every direction at once.

## What a game looks like from opening to end

- **Opening:** both sides start on their own edge of the ring, the most dangerous place to be.
  The first moves are a race into the middle, which is safest (far from every edge) and also what
  wins a stalled game. Who gets there first, and with which pieces, sets up the fight.
- **Middle:** the two groups meet in the centre and lean on each other. Shoves move whole lines,
  so a single good shove can drive two or three enemy pieces toward the rim at once and leave
  them strung out. The play is about building lines heavy enough not to be shoved back, finding
  the direction a group is light in, and luring a light piece toward the edge.
- **End:** either one side runs out of pieces it can afford to lose (it has been pushed out too
  many times) or the shoving locks up and a clock gives the game to whoever holds more of the
  centre cells. The side losing on the centre count has to break the lock, which keeps an even
  ending from being a dead draw.

## What would make it fail

- **Fortress:** both sides huddle into blobs too heavy to shove from any direction and the game
  becomes a count of who got to the middle first, decided by the opening race (a seat or rush
  problem).
- **Dominant piece:** one kind (the strong one) does all the shoving and the others are only
  weight; or the light quick piece is so exposed it is never worth moving.
- **Shuffle and repetition:** two pieces shoving each other back and forth without progress.
- **First-move advantage:** the first player arrives in the centre a tempo ahead and the second
  can never get it back.
- **Edge lottery:** knock-outs so cheap near the rim that the game is decided in a few moves by
  whoever strikes first, with no build-up.


## First hand-in: break attempts and design log

# heft: design log and break attempts (session 1, seed: pushing by weight, with no capture)

Frozen spec: `game.json`, hash **f596ad2440e2** (= v10 below). `reason.md` was written before any
spec and is unchanged. Tools used: validate, smoke, play, Engine probes (scratchpad scripts
`view.py`, `tx.py`, `tally.py`, `seats.py`, `rush.py`, `rules.py`; each only replays engine moves
or picks among the engine's legal moves), two tournaments (72 games each, results by seat,
lengths and transcripts only). Nothing forbidden was run or read.

## Where play changed my view (reason.md left as written)

- reason.md expected the centre clock to matter in many games. In play it rarely fires (1 of 72
  games in the last tournament): knock-outs decide almost every game, and the clock is a backstop
  against a lock-up rather than a regular ending.
- reason.md did not foresee the tempo problem: with one move a turn, whoever steps into reach
  first is shoved first, so the second player was favoured. The two-move turn (v10) is the answer,
  and it also sharpens the stated decision: a turn is literally two moves to split between
  striking and stacking.

## Design log

| v | hash | change | why / what happened |
|---|------|--------|---------------------|
| 1 | 065a8fc7f6c5 | Hexagon of 37 cells; three kinds: dart (w1 s2, slides 2), bull (w3 s5, steps 1), stone (w4, steps 1, never shoves); 9 a side on own rim; lose at 3 out; centre-7 region decides a 40-move stall. | First game: darts shoved each other back and forth, draw by repetition in 6 plies. |
| 2 | b3113fdd4e9d | + no shoving straight back. | Ping-pong gone. But darts did all the work (s2 let a dart move dart pairs), bulls and stones barely moved; games ran to the clock after long dart shuffles. |
| 3 | e58a9312697e | Dart strength 1 (moves only a lone dart); bulls into the front row; lose at 4 out. | Bulls now dominant and too strong: s5 shoved two-piece lines off the rim from the start position. White won 3/3 at 6400; edge lottery. |
| 4 | 00ba78ab8f8e | Bull strength 4. | Black won 6/6 at 6400: the side that approaches first is shoved first. |
| 5 | 425fdb8e7372 | 61-cell hexagon, armies one row in from the rim, 11 a side. | Nobody engaged: every game went to the 40-move clock, decided by centre count. Board too big for one-cell shoves. |
| 6 | 9d32df3ecdba | v5 + a crushing pit in the centre, count pieces at the clock. | 80% stalls in smoke, 4/6 draws. Dropped the big board. |
| 7 | 977d3e8a130b | v5 with scoring on the enemy's home rows (invade the rim). | Shuffles (stones stepping back and forth), 72% draws in smoke. Dropped. |
| 8 | 8f6f055d529f | Back to v4 (37 cells); stones placed behind the bulls. | 6400 self-play: black 5/6. 25600 self-play: 2-2, so the 6400 lean looked partly a bot artifact. |
| 9 | 1dcff2553900 | Stall clock 40 -> 60 moves (a clock at move 40 from the start handed black the last move before the count). | Tournament t1 (72 games): strength ladder clean (25600 > 6400 > 1600), 66/72 by knock-out, but seat 0 29, seat 1 42; equal pairs 7-1 and 6-2 to the second player. Seat broken. |
| 10 | f596ad2440e2 | Two moves a turn, first player's first turn one; stall clock 80, cap 300. | Tournament t2 (72 games): seat 0 30, seat 1 34, 8 draws; ladder clean (25600 0.74, 6400 0.56, 1600 0.20); 63 knock-outs, 8 repetitions (5 of them 1600 v 1600), 1 clock. **Frozen.** |

## Break attempts (on v10 unless stated)

1. **A rush.** Scripted rusher (`rush.py`): always push off if possible, else push, else the
   furthest advance by bulls. As white and as black vs minimax:nodes=6400, seeds 0-1: lost all
   four (27-41 plies). As black vs minimax:nodes=1600: lost both (17, 21 plies). Dart-rusher as
   white vs 6400: lost both. No rush found. On v3 a bull charge did work (s5 cleared two-deep
   lines off the start rim), which is why bull strength came down to 4.
2. **A fortress or shuffle.** (a) Turtle at home (`rush.py TURTLE`: push off when offered, else
   the least advancing move): lost to 6400 in 38 and 59 plies and to 1600 in 21 as black. The
   home rim is the most dangerous place, so sitting still is not safe. (b) v1 had a two-dart
   shuffle that drew by repetition in 6 plies; no_push_back closed it. Removing no_push_back from
   v10 brought a repetition draw back in 1 of 4 6400 games, so the rule stays. (c) v5-v7 (big
   board) were shuffles and stalls and were abandoned. (d) Not closed: a heavy centre blob
   (stones and bulls braced in every direction) is not something my scripts can build; in t2
   only 1 game reached the clock and 8 drew by repetition, mostly between weak players, so the
   bots do not find a fortress, but I have not proven none exists.
3. **A forced line.** No game in t1/t2 under 28 plies except between 1600 players (four 12-ply
   games, 1600 self-play blunders). Openings vary by seed; no repeated winning line found.
4. **A dead rule.** no_push_back: alive (see 2b). Two-move turn: alive (moved seat from 29/42 to
   30/34). Centre clock: fires rarely (1/72) but is the only thing that decides a locked
   position; kept as the anti-stall rule, knowingly a backstop. stalemate as a loss: never seen;
   it is there so a smothered side loses rather than draws. Stones: move 12% of the time and never
   shove, but they are what make lines too heavy (the stack half of the decision).
5. **A dominant piece or move.** t2 tallies (all 72 games): moves bull 2003, dart 1780, stone
   505; on-board shoves bull 1026, dart 662; push-offs bull 262, dart 126. Bulls are the main
   hitters, darts a real second; stones are ballast by design. On v3 bulls were dominant to the
   point of a rush; v2's darts dominated; v10 is the balance between those.
6. **The seat.** One move a turn favoured the second player (v4: black 6/6 at 6400; v9: 42-29
   over 72). Reason: approaching first gets you shoved first. v10's two-move turn with a one-move
   first turn brought it to 30-34 with 8 draws; equal-strength self-play still leans to the second
   player (25600 3-5, 6400 1-6) while 25600 as first beat 6400 6-2. **Residual lean to the second
   player not fully closed**; the samples are small.

## Hand-in transcripts (frozen spec, `play`)

- 6400 v 6400: seed 0 first player by knock-out, 56 plies; seed 1 second, 55; seed 2 first, 73.
- 25600 v 1600: seed 0 **1600 won** (knock-out, 83 plies); seed 1 25600, 29; seed 2 25600, 93.

## Final checks (game.json)

```
heft [f596ad2440e2] is valid
  board       7x7
  pieces      D, B, S
  opening     14 legal moves
  rule mass   63.0
game.json [f596ad2440e2]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game watering hole [417c19c2f6bd] 0.2159  below 0.5: yes
limits exit 0
heft [f596ad2440e2]: passed
  draw_rate        0.2
  forced_rate      0.0
  games            150.0
  mean_branching   27.97
  mean_plies       156.65
  opening_moves    14.0
  stall_rate       0.2
```

Tournaments: 2 (t1-v9, t2-v10; 72 games each, 4 workers, heavy lease of 4 slots, released after
each), output in `runs/R-S10/sessions/1/`. No lease held at hand-in.


## Revised hand-in: design reason

# Design reason (written before any spec)

Seed: pushing by weight, with no capture.

## The game in one line

A shoving match on a small hexagonal ring. Nothing is ever taken: a piece that moves onto an
enemy shoves it, together with everything standing behind it in that line, one cell on, and a
piece shoved off the ring is out. Push enough of the enemy out, or hold the middle of the ring
when the shoving stalls, and you win.

## The decision a player keeps facing

**Stack or strike.** Every shove is a sum: it goes through only if the shover's strength is at
least the total weight of the line in front of it. So a piece standing alone is light and easy
to throw, and a piece with friends lined up behind it is heavy and safe. The pieces that hit
hardest are not the pieces that make a line heavy: a heavy, strong piece that is busy bracing a
friend is not shoving anyone, and a quick light piece that runs out to strike leaves the line it
was part of lighter and itself alone near the edge. Every turn asks whether to add weight to
your own lines (safe, slow, gives up ground) or send a piece out to shove (gains ground or a
knock-out, but thins what is behind it). Because lines run in six directions on a hex board, a
group braced against a shove from one side is usually open along another, so no shape is safe
from every direction at once.

## What a game looks like from opening to end

- **Opening:** both sides start on their own edge of the ring, the most dangerous place to be.
  The first moves are a race into the middle, which is safest (far from every edge) and also what
  wins a stalled game. Who gets there first, and with which pieces, sets up the fight.
- **Middle:** the two groups meet in the centre and lean on each other. Shoves move whole lines,
  so a single good shove can drive two or three enemy pieces toward the rim at once and leave
  them strung out. The play is about building lines heavy enough not to be shoved back, finding
  the direction a group is light in, and luring a light piece toward the edge.
- **End:** either one side runs out of pieces it can afford to lose (it has been pushed out too
  many times) or the shoving locks up and a clock gives the game to whoever holds more of the
  centre cells. The side losing on the centre count has to break the lock, which keeps an even
  ending from being a dead draw.

## What would make it fail

- **Fortress:** both sides huddle into blobs too heavy to shove from any direction and the game
  becomes a count of who got to the middle first, decided by the opening race (a seat or rush
  problem).
- **Dominant piece:** one kind (the strong one) does all the shoving and the others are only
  weight; or the light quick piece is so exposed it is never worth moving.
- **Shuffle and repetition:** two pieces shoving each other back and forth without progress.
- **First-move advantage:** the first player arrives in the centre a tempo ahead and the second
  can never get it back.
- **Edge lottery:** knock-outs so cheap near the rim that the game is decided in a few moves by
  whoever strikes first, with no build-up.


## Revised hand-in: break attempts and design log

# heft: design log and break attempts (session 1, seed: pushing by weight, with no capture)

**Revised hand-in: frozen hash 068d83b22de1 (v12).** The first hand-in (f596ad2440e2) was not
admitted. It is kept in `first-handin/`. Everything down to *Revision* below is the first
hand-in's log, unchanged. The revision follows it.

Frozen spec: `game.json`, hash **f596ad2440e2** (= v10 below). `reason.md` was written before any
spec and is unchanged. Tools used: validate, smoke, play, Engine probes (scratchpad scripts
`view.py`, `tx.py`, `tally.py`, `seats.py`, `rush.py`, `rules.py`; each only replays engine moves
or picks among the engine's legal moves), two tournaments (72 games each, results by seat,
lengths and transcripts only). Nothing forbidden was run or read.

## Where play changed my view (reason.md left as written)

- reason.md expected the centre clock to matter in many games. In play it rarely fires (1 of 72
  games in the last tournament): knock-outs decide almost every game, and the clock is a backstop
  against a lock-up rather than a regular ending.
- reason.md did not foresee the tempo problem: with one move a turn, whoever steps into reach
  first is shoved first, so the second player was favoured. The two-move turn (v10) is the answer,
  and it also sharpens the stated decision: a turn is literally two moves to split between
  striking and stacking.

## Design log

| v | hash | change | why / what happened |
|---|------|--------|---------------------|
| 1 | 065a8fc7f6c5 | Hexagon of 37 cells; three kinds: dart (w1 s2, slides 2), bull (w3 s5, steps 1), stone (w4, steps 1, never shoves); 9 a side on own rim; lose at 3 out; centre-7 region decides a 40-move stall. | First game: darts shoved each other back and forth, draw by repetition in 6 plies. |
| 2 | b3113fdd4e9d | + no shoving straight back. | Ping-pong gone. But darts did all the work (s2 let a dart move dart pairs), bulls and stones barely moved; games ran to the clock after long dart shuffles. |
| 3 | e58a9312697e | Dart strength 1 (moves only a lone dart); bulls into the front row; lose at 4 out. | Bulls now dominant and too strong: s5 shoved two-piece lines off the rim from the start position. White won 3/3 at 6400; edge lottery. |
| 4 | 00ba78ab8f8e | Bull strength 4. | Black won 6/6 at 6400: the side that approaches first is shoved first. |
| 5 | 425fdb8e7372 | 61-cell hexagon, armies one row in from the rim, 11 a side. | Nobody engaged: every game went to the 40-move clock, decided by centre count. Board too big for one-cell shoves. |
| 6 | 9d32df3ecdba | v5 + a crushing pit in the centre, count pieces at the clock. | 80% stalls in smoke, 4/6 draws. Dropped the big board. |
| 7 | 977d3e8a130b | v5 with scoring on the enemy's home rows (invade the rim). | Shuffles (stones stepping back and forth), 72% draws in smoke. Dropped. |
| 8 | 8f6f055d529f | Back to v4 (37 cells); stones placed behind the bulls. | 6400 self-play: black 5/6. 25600 self-play: 2-2, so the 6400 lean looked partly a bot artifact. |
| 9 | 1dcff2553900 | Stall clock 40 -> 60 moves (a clock at move 40 from the start handed black the last move before the count). | Tournament t1 (72 games): strength ladder clean (25600 > 6400 > 1600), 66/72 by knock-out, but seat 0 29, seat 1 42; equal pairs 7-1 and 6-2 to the second player. Seat broken. |
| 10 | f596ad2440e2 | Two moves a turn, first player's first turn one; stall clock 80, cap 300. | Tournament t2 (72 games): seat 0 30, seat 1 34, 8 draws; ladder clean (25600 0.74, 6400 0.56, 1600 0.20); 63 knock-outs, 8 repetitions (5 of them 1600 v 1600), 1 clock. **Frozen.** |

## Break attempts (on v10 unless stated)

1. **A rush.** Scripted rusher (`rush.py`): always push off if possible, else push, else the
   furthest advance by bulls. As white and as black vs minimax:nodes=6400, seeds 0-1: lost all
   four (27-41 plies). As black vs minimax:nodes=1600: lost both (17, 21 plies). Dart-rusher as
   white vs 6400: lost both. No rush found. On v3 a bull charge did work (s5 cleared two-deep
   lines off the start rim), which is why bull strength came down to 4.
2. **A fortress or shuffle.** (a) Turtle at home (`rush.py TURTLE`: push off when offered, else
   the least advancing move): lost to 6400 in 38 and 59 plies and to 1600 in 21 as black. The
   home rim is the most dangerous place, so sitting still is not safe. (b) v1 had a two-dart
   shuffle that drew by repetition in 6 plies; no_push_back closed it. Removing no_push_back from
   v10 brought a repetition draw back in 1 of 4 6400 games, so the rule stays. (c) v5-v7 (big
   board) were shuffles and stalls and were abandoned. (d) Not closed: a heavy centre blob
   (stones and bulls braced in every direction) is not something my scripts can build; in t2
   only 1 game reached the clock and 8 drew by repetition, mostly between weak players, so the
   bots do not find a fortress, but I have not proven none exists.
3. **A forced line.** No game in t1/t2 under 28 plies except between 1600 players (four 12-ply
   games, 1600 self-play blunders). Openings vary by seed; no repeated winning line found.
4. **A dead rule.** no_push_back: alive (see 2b). Two-move turn: alive (moved seat from 29/42 to
   30/34). Centre clock: fires rarely (1/72) but is the only thing that decides a locked
   position; kept as the anti-stall rule, knowingly a backstop. stalemate as a loss: never seen;
   it is there so a smothered side loses rather than draws. Stones: move 12% of the time and never
   shove, but they are what make lines too heavy (the stack half of the decision).
5. **A dominant piece or move.** t2 tallies (all 72 games): moves bull 2003, dart 1780, stone
   505; on-board shoves bull 1026, dart 662; push-offs bull 262, dart 126. Bulls are the main
   hitters, darts a real second; stones are ballast by design. On v3 bulls were dominant to the
   point of a rush; v2's darts dominated; v10 is the balance between those.
6. **The seat.** One move a turn favoured the second player (v4: black 6/6 at 6400; v9: 42-29
   over 72). Reason: approaching first gets you shoved first. v10's two-move turn with a one-move
   first turn brought it to 30-34 with 8 draws; equal-strength self-play still leans to the second
   player (25600 3-5, 6400 1-6) while 25600 as first beat 6400 6-2. **Residual lean to the second
   player not fully closed**; the samples are small.

## Hand-in transcripts (frozen spec, `play`)

- 6400 v 6400: seed 0 first player by knock-out, 56 plies; seed 1 second, 55; seed 2 first, 73.
- 25600 v 1600: seed 0 **1600 won** (knock-out, 83 plies); seed 1 25600, 29; seed 2 25600, 93.

## Final checks (game.json)

```
heft [f596ad2440e2] is valid
  board       7x7
  pieces      D, B, S
  opening     14 legal moves
  rule mass   63.0
game.json [f596ad2440e2]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game watering hole [417c19c2f6bd] 0.2159  below 0.5: yes
limits exit 0
heft [f596ad2440e2]: passed
  draw_rate        0.2
  forced_rate      0.0
  games            150.0
  mean_branching   27.97
  mean_plies       156.65
  opening_moves    14.0
  stall_rate       0.2
```

Tournaments: 2 (t1-v9, t2-v10; 72 games each, 4 workers, heavy lease of 4 slots, released after
each), output in `runs/R-S10/sessions/1/`. No lease held at hand-in.


---

# Revision (the one allowed)

## What the blind reader found (read-1/reading.md, the only file read there)

Admission failed on ii, iv and v.
- **Fortress:** a ring of stones and bulls on the seven middle cells, round a dart, cannot be
  shoved. Every line through it weighs at least 5 and the strongest shove was 4. The holder keeps
  the ring by stepping a stone out and back within its two-move turn, and wins the count.
- **Tug:** the two-move turn gets round the no-push-back bar, because the bar lasts one move. That
  gave a 24-cycle bull tug in 25600-v-1600 seed 0, and threefold draws at 12 plies in 4 of 8 games
  at 1600.
- **Shape:** knock-outs start at ply 4, so the opening, the middle game and the end are one
  activity. The clock ending almost never fires.
- **Seat:** the first player won 16-7-1 at equal budgets of 6400 and up.

## What I changed and why

- **The ram (fortress).** A new piece: weight 1, strength 8, steps one cell. It is the only
  piece that can move a line of two heavy pieces (4+3, 4+4, 3+3, or 4+1+3 through a ring's
  centre). It is also the lightest piece, so a lone ram is thrown by anything, a dart included.
  It replaces one dart (e1). With this army no blob is ram-proof. That would need every exposed
  line to weigh 9 or more, which takes three pieces in every line. A seven-cell flower has
  two-piece lines, and the heaviest pair is S+S = 8.
- **The turn in two parts (tug, and the stated decision).** In place of "two moves, the first
  turn one", each turn is now: first a dart, bull or ram moves, then you may step one stone
  (`rules.turn.parts`, the second part optional).
  - A shoved piece's owner moves next with a shoving piece, so the bar applies: neither the
    shoved piece nor any row through it can be shoved straight back.
  - The stone half cannot shove at all, so the reader's out-and-back dodge of the bar is gone.
  - The turn now carries reason.md's choice: one shoving move, and one place to add weight.
- **Knock-outs later (shape).** No rule was added for this. It follows from the one-shove turn
  and from darts no longer being the only quick strikers. The first push-off fell at a median of
  ply 17 in t2 (v10); it is ply 39 in t4 (v12) and ply 29 in t5. A ply is one part of a turn, so
  that is 7 to 10 turns a side. The clock now decides about a quarter of games (17 of 72 in t4).
- **Seat (handicap).** With the parts turn (v11) the second player led 42-27 (t3; knock-outs
  34-19). The second player now starts with one dart fewer (no dart on b7), so it loses after
  three pieces out where the first player loses after four. This is a declared handicap, like
  komi. Nothing structural I could express both fixed the tug and evened the tempo.
- reason.md is unchanged, because the aim is the same: stack or strike, a scramble off the rim
  into the middle, and knock-outs or a centre count at the end. The revision changes the
  mechanisms that serve it.

## Design log (continued)

| v | hash | change | why / what happened |
|---|------|--------|---------------------|
| 11 | 14befaaf9ec0 | Ram replaces the e1 dart. Turn = striker part then optional stone part. Rest as v10. | Fortress probe: the reader's ring is now shoved by a ram from 8 of 9 neighbouring cells. Tournament t3 (72): ladder clean (25600 0.72, 6400 0.49, 1600 0.29). Seat 27-42-3 to the second player. Median 117 plies; 16 decided by the count. |
| 12 | 068d83b22de1 | v11 with the second player one dart short (b7 empty, `mirror: false`, black written out). | t4 (72): seat 38-33-1; ladder clean (0.71 / 0.50 / 0.29); first push-off at a median of ply 39; 17 decided by the count. t5 (64, 6400 and 25600 only): see the seat below. **Frozen.** |

## Fresh break attempts on the frozen spec

1. **The fortress (reader's ring).** I built it with `fortress.py`: holder player 0 has stones on
   e4, c5, d3, bulls on d5, c4, e3, a dart on d4, and spares. I put the attacker's ram alone on
   each empty cell next to it, with the attacker's other pieces far away. It shoves into the ring
   from b4, b5, c3, c6, e5, f3, f4 and c2. From d6 there is no shove, because the far side is
   blocked.
   - From the reader-style position, with every piece crowded round, three 6400-v-6400 games went
     2 to 1 to the attacker. Before the ram the holder won 3/3.
   - A ram next to the ring stands alone and can be thrown by the ring's own bulls and darts, so
     breaking the ring needs an escort.
   - With this army no shape is safe from the ram (the reasoning is under the ram above). Fortress
     closed as far as I can test it.
2. **The tug.** I counted, per game, the most times any one shove text recurs (`tug.py`).
   - v10 (t2): up to 19, with eight games at 4 and fourteen at 5 or more.
   - v12 (t4): at most 4 (two games), and every other game 3 or less.
   - t5: at most 5, except one game at 8.
   - That game (25600 v 6400) is a **rotating tug**: four shoves land on d6 from four directions
     in turn (Bd7>d6, Bd5>d6, Rc7>d6, Be5>d6). Each comes from a different line, so the bar does
     not apply. The stall clock ended it as a draw (no_progress, equal count). This is 1 game
     in 64, bounded by the clock, and **not closed**. Repetition draws: 1 in t4, 0 in t5.
3. **The seat.**
   - Overall: t4 38-33-1 and t5 37-24-3.
   - At equal budgets, pooling t4 and t5: 25600 v 25600 9-14 (the second player ahead). 6400 v
     6400 18-5-1 (the first player ahead).
   - Cross-budget play favours the first player: in t5, 25600 as first won 14-1, and 6400 as
     first won 5-10.
   - Without the handicap (t3), equal budgets went 25600 3-5 and 6400 2-6.
   - So the handicap is about right for the strongest player I ran (25600, slightly to the second
     player) and too large for 6400 (0.75 to the first player). **Residual seat lean depends on
     strength and is not closed.** I kept the handicap because the strongest player is the better
     guide. On the six hand-in transcripts the first seat won all six; three of them are
     25600 v 1600 with the stronger player moving first.
4. **Rush.** `rush.py` (a scripted rusher picking among the engine's legal moves) against
   minimax:nodes=6400, seeds 0-1. A ram rush lost all four games, two as white and two as black,
   in 33 to 67 plies. A bull rush also lost all four. No rush found.
5. **Turtle at home.** `TURTLE` against 6400, seeds 0-1: lost in 83 and 75 plies. Sitting on
   your own rim is not safe.
6. **Dominant piece.** t4, all games:
   - Moves: stone 2985, bull 1791, dart 1694, ram 638, leaving the stone half 1084 times.
   - Push-offs: bull 177, dart 77, ram 33.
   - The ram is a specialist (259 on-board shoves, mostly cracking pairs) and does not dominate.
   - The stones move most because they have their own half of every turn. Some of that is
     shuffling during quiet stretches, which the clock bounds.
7. **Shape.** The first push-off (`firstko.py`) fell at a median of ply 39 in t4 (quartiles 29
   and 57) and ply 29 in t5. There are about 4 to 5 push-offs per game, against a median ply 17
   in v10. The count decided 17 of 72 games (t4) and 5 of 64 (t5). Some quiet stretches are
   waiting moves (darts stepping back and forth) until the clock or a ram break.

## Hand-in transcripts (frozen spec 068d83b22de1, `play`)

- 6400 v 6400: seeds 0-2 all won by the first player by knock-out, in 109, 81 and 113 plies.
- 25600 v 1600: seeds 0-2 all won by 25600 by knock-out, in 57, 101 and 85 plies.

## Final checks (game.json, revised)

```
heft [068d83b22de1] is valid
  board       7x7
  pieces      D, B, R, S
  opening     12 legal moves
  rule mass   71.5
game.json [068d83b22de1]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game watering hole [417c19c2f6bd] 0.2104  below 0.5: yes
limits exit 0
heft [068d83b22de1]: passed
  draw_rate        0.253
  forced_rate      0.0
  games            150.0
  mean_branching   13.73
  mean_plies       122.11
  opening_moves    12.0
  stall_rate       0.253
```

Revision tournaments: 3 (t3-v11 72 games, t4-v12 72, t5-final 64), each under a heavy lease
of 4 slots and released right after. Output is in `runs/R-S10/sessions/1/`. No lease is held
and nothing is running.


## Blind read 1 (read-1)

# Blind reading: heft (seed: pushing by weight, with no capture)

Tools used: `play` (single games), `validate`, and Engine probes written in this folder
(`probe.py` replays a transcript and prints the board with `root_scores`; `scan.py` finds plies
whose two best moves are close and of different kinds; `fortress.py` builds a hand position).
No score, test, measure, analyse or tournament was run. Plies below count from 1; the turn
schedule is P0 ply 1, P1 plies 2-3, P0 4-5, and so on.

## Attempts

1. **Seat.** I played equal-budget minimax games beyond the transcripts: 6400 self seeds 3-12,
   25600 self seeds 0-7, 102400 self seeds 0-2. Adding the three 6400 transcripts, the first
   player won 16, the second 7, with 1 draw (24 games, ~0.69 for the first seat). At 1600 self
   (seeds 0-7): 1 first-seat win, 2 second-seat wins, 5 draws by repetition. Random self
   (seeds 0-5) and greedy self come out about even. **There is a lean to the first seat at every
   budget that plays properly.** It is not overwhelming and not proven, but it is the lean the
   design reason names as a failure.
2. **Rush / forced line.** More search wins every game: 25600 beats 1600 in 14 of 14 games
   across both seats, and 102400 beats 25600 in 6 of 6. Games are short (34-57 plies at
   102400). From the start, `root_scores` at 20k nodes and depth 6 sees no knock-out, and I
   found no opening knock-out sequence by hand. Knock-outs do start very early, though:
   6400-self-seed0 has a dart thrown off at ply 4 (`De4>>e6`). I found no rush, only a steep
   tactical ladder where whoever sees deeper wins.
3. **Fortress (found).** Six heavy pieces in a ring on the seven middle cells, alternating stone
   and bull (e4 S, d5 B, c5 S, c4 B, d3 S, e3 B), around a dart on d4. Every line through
   the blob then weighs at least 5 (each ring pair 7, each line through the centre 8), and the
   strongest shove in the game is 4. `fortress.py` puts all nine enemy pieces around it: the
   enemy has 29 legal moves and **none lands on a blob cell**. The only pushes it has hit the
   holder's two spare darts. The holder can keep the shape without ever breaking it: a two-move
   turn lets a stone step out and back (`Sd3-e2`, `Se2-d3` gives an identical board). The holder
   can lose both spare darts and still have 7 pieces (the game is lost only below 6). It owns
   all seven scoring cells for the clock. Three 6400-self games played from that position all
   went to the fortress side by extinction (36, 132 and 60 plies), its bulls striking out from
   the blob. The defender's only resource I can see is a draw by threefold repetition, if the
   holder has nothing left to vary. I did not show that the fortress can be built from the
   opening against resistance. But "put the heavy pieces in a ring in the middle" is a plan a
   person would find. The bots never try it: their evaluator is close to flat here
   (`root_scores` mostly ties at 0 / ±0.25), so they cannot see it.
4. **Shuffle (found).** The two-move turn defeats the no-push-back bar. The bar lasts one move,
   so the shoved piece spends its owner's first move elsewhere and shoves straight back with
   the second. Two examples:
   - 25600-v-1600-seed0: `Bb5>b6` / `Bb7>b6` alternate from ply 17 to ply 63, about 24 cycles,
     half of every turn for both sides.
   - 1600 self seed 1 (my game): darts ping-pong on c4/e4 (`De3>e4`, `Dc5>c4`, ...) and the game
     is drawn by threefold repetition at ply 12. The same 12-ply draw happens in 4 of 8 1600-self
     seeds.

   The same tug of war on e3/e4/e5 runs in 102400-v-25600 seed 2 (`Be3>e4`, `Be5>e4` x5).
   Stronger players break out of it, so it is not a forced draw.
5. **Dead rule.** I deleted `no_push_back` (`nobar.json`) and replayed 6400 self seeds 0-5.
   5 of 6 results changed, and seed 0 became a repetition draw. The rule is live, but per
   attempt 4 it only delays the push-back by one move.
6. **Dominant piece.** Across the six transcripts, bulls threw off 24 pieces and darts 16;
   stones made 1-16 moves a game and never shove. That is roughly the split the design wants
   (bull the hitter, dart the light striker, stone the ballast). No single piece dominates.
7. **Centre-count ending.** It decides none of the six transcripts (all extinction). In my 30-odd
   extra games it decided one (6400 self seed 9).

## The five questions

**(i) Yes.** 6400-self-seed2, before ply 18 (`9... Dd2>>e1`), with P1 (the second player) to
make both moves of its turn. White's bull on b5 threatens to shove P1's lone bull on a6 off the
corner (strength 4 against weight 3). `root_scores` at 40k nodes and depth 8 rates two moves
equal at 0.00, with the third at -0.25:
- `Dd2>>e1` throws White's dart off at once. It gives up safety: the dart is left alone on
  White's rim at e1, and the bull's danger must now be met with the turn's last move. The game
  then played `Ba6>b5`, shoving White's bull back.
- `Ba6-b6` takes the bull off the rim first. It gives up the knock-out: White's dart on e1 gets
  a turn to step away.

**(ii) No.** All three are the same activity. Knock-outs begin in the opening (ply 4 in
6400-self-seed0; ply 13 in 25600-seed0), and the game ends at the fourth knock-out. The one
ending that would differ, the centre count after a stall, decides none of the six. The opening
is the same move in all six (`Dg2-e4`, answered by `Da6-c4` in five), but it is already a shoving
contest on e4/c4, not a distinct race phase. The games are short, 29-93 plies, about 7-23 turns
a side.

**(iii) Yes.** Every transcript is won by shoving pieces off the rim, and shoves are allowed or
refused by strength against weight:
- 25600 s0: the second player wins by throwing off five; the final `Bc3>>b3` is a bull throwing
  a lone piece. The long `Bb5>b6` / `Bb7>b6` tug is bull (4) against lone bull (3).
- 25600 s1: bulls throw off all six (`Bc4>>b5`, `Bc3>>c2`, `Bc6>>d6`, `Bc2>>d1`, `Bd6>>e6`,
  `Be6>>d7`).
- 25600 s2: the darts' shoving contest on e3/e4, where only a lone dart can be shoved, then the
  final `De2>>e1`.
- 6400 s0: `De4>>e6` at ply 4 (dart throws dart), ending with `Bd6>>e6`.
- 6400 s1: seven throws, bulls and darts, ending with `Dc3>>e1`.
- 6400 s2: `Dd2>>e1` (the position in (i)), ending with `Bc6>>b7`.

One caveat: the *stacking* half of the seed shows mostly as absence. The throws I can see are of
lone or light lines; braced lines are simply not attacked.

**(iv) No.** I found a fortress a person would find: the stone-bull ring on the seven middle
cells. It is verified immune to every legal shove, can be kept in place, wins the count, and the
bots won from it 3 of 3. I also found a shuffle that the two-move turn lets past the no-push-back
bar (24 cycles in 25600-seed0, and threefold draws at 12 plies at 1600). No rush and no forced
line from the opening were found. The first-seat lean was about 0.69.

**(v) No.** Some of the design reason holds: bulls hit, darts strike light pieces, stones are
ballast, and stack-or-strike choices like (i) exist. But several things it promises or warns
against happen:
- It warns against a fortress; a provably unshovable centre fortress exists.
- It warns against shuffling; the push-back tug is common and the bar does not stop it.
- It warns against a first-move advantage; the first seat leads 16-7.
- It warns against an edge lottery; knock-outs start at ply 4, and games are short and decided
  by search depth.
- It promises a clock ending that "keeps an even ending from being a dead draw"; that ending
  almost never fires.
- It promises a three-phase shape; I could not see one.


## Blind read 2 (read-1b)

# Blind reading: heft (seed: pushing by weight, with no capture)

I took the rules from rules.txt. All my probes are in this folder: replay.py (replays a transcript
through Engine, prints push-offs and boards), turtle.py (a scripted 1-ply "person's plan" for one
seat against a minimax opponent), alts.py / scan.py (root_scores at transcript positions),
weights.py (what each push moved), even.json (the spec with player 1's missing dart restored at
b7). Only single games, played by `play` or my own scripts. I read no score.

## Who lost what in the six transcripts (replay.py)

Every game is won by player 0, by extinction (the opponent below 6 pieces). Player 0 starts with
9 pieces and player 1 with 8, so player 1 dies after its 3rd loss and player 0 only after its 4th.
- 25600-v-1600 s0: P1 lost 3, P0 lost 0. s1: P1 lost 3, P0 lost 2. s2: P1 lost 3, P0 lost 1.
- 6400-self s0: both sides lost 3 pieces, and P1 lost because it hit 5 first (P0 stood on 6).
  s1: P1 lost 3, P0 lost 2. s2: both lost 3 again, and P1 hit 5 first.
Knock-outs by piece: bull 19, dart 7, ram 3. Every push in 25600-v-1600 s0 (13 of them) was a bull.

## Break attempts

1. **Seat.** Self-play from seeds 3 to 20 with `play`:
   - 1600 self (s3 to s6): P0 2, P1 2. Three of the four ended *on the centre count at ply 80*:
     nobody was pushed out in the first 40 turns.
   - 6400 self (s3 to s20, 18 games): P0 13, P1 4, 1 draw. With the three transcripts that makes
     P0 16 of 21.
   - 25600 self (s3 to s14, 12 games): P0 5, P1 7.
   - even.json (P1 given its dart back, so armies are equal): 6400 self, P1 won 8 of 10.
     25600 self, P0 won 6 of 8.
   So which seat leads flips with the budget, and the handicap does not fix a stable first-move
   edge. The design reason assumed one, and I could not find it. The handicap does have a mechanical cost the rules hide.
   The loss limit is absolute (fewer than 6), so the missing dart cuts P1's margin from 4
   losses to 3. That alone decided 6400-self s0 and s2: the losses were even at 3 to 3 and P1 lost.
2. **Fortress / turtle** (turtle.py, plan `turtle`). Each turn: take a knock-out if one is on offer.
   Otherwise go for the seven middle cells, keep off the rim and never leave a piece that can be
   shoved off next move.
   - Against minimax 1600 (4 games): it never lost a piece, and every game ended on the centre count
     at ply 80 (turtle won 3 of 4). At that level the shoving does stall.
   - Against 6400 and 25600 (12 games, both seats): turtle won 2 and lost 10, mostly by extinction
     within 60 to 155 plies. Two-move set-ups (line up a shove, then strike) crack a one-move-safe
     huddle, so no simple fortress holds against a player who looks ahead.
3. **Rush / raid** (plan `raid`: always take a knock-out, else close in on enemy rim pieces).
   Against 1600 it won 2 of 4 and none of them quickly (decided on the count at ply 80 to 194). No rush.
4. **Edge lottery / short games.** The shortest strong games (6400 self s9, s14: 37 plies) are
   both rim fights over the start squares. In s14, P1 lost a stone, its ram and a dart in plies 29
   to 37, all knocked off from where they began. Even these come after 14 quiet turns of
   development, so I found no forced knock-out from the opening. Starting on your own rim
   is dangerous, as the design intends.
5. **Forced line.** At every transcript position I scored (scan.py and alts.py), several moves came
   out level, and I found no short forced win. One forced reply did appear, a defence rather than
   an attack: 25600-v-1600 s2 move 6, Rc7>c6 scores -0.75 and every alternative -3 or worse.
6. **Dead rule / dominant piece.** Bulls shove a lone stone exactly (4 against weight 4) and do
   most of the knocking out. 25600-v-1600 s0 is bulls only: P1's ram moved once in 29 turns, and
   no push moved more than one piece. The ram is the rule that makes weight matter. 6400-self s0
   moves 48 and 50 are ram shoves of a dart, stone and bull line weighing exactly 8, which is the
   ram's strength. In the other games the ram moves 11 to 16 times and its shoves knock pieces out.
   Darts move a lot and knock out 7. So no rule is dead, but the bull carries the knock-outs.
7. **Shuffle.** 6400-self s1 moves 13 to 22: both sides rock Bb4-b5/Bb5-b4, Da4-a6/Da6-a4 and
   stone b6/b7 back and forth for ten turns before breaking out. Repetition is counted only at a
   turn's start, so it did not end the game. The shuffle failure mode does appear.
8. **Rules text.** rules.txt says "80 turns without a capture", but the 1600 games end on the
   count at exactly 80 *plies* (40 turns, 20 each). The summary's "80 moves" is closer.

## The five answers

**(i) Yes.** 25600-v-1600 seed 1, move 6, player 1 to move (after 10 plies). Player 0's dart has
run in to c6. From there it threatens three knock-outs: Dc6>>c7 (the ram), Dc6>>a6 and Dc6>>e6.
The engine scores two replies level (-0.71 and -0.75) and every other reply -5.6 or worse:
- **Rc7-d6** (played), with the stone half Sa7-b6. The ram steps aside and the stone plugs b6,
  which blocks the dart's slide to a6. The intruder stays alone and stuck deep in P1's camp, a
  target for later. The cost is a whole turn spent on defence, the stone pulled off the back
  rim, and the threats left in place for one more turn.
- **Rc7>c6** (played in a similar position in seed 2). The ram shoves the dart to c5, and that clears
  every threat at once (P0 has no shove left: checked). The cost is that it sends the dart back
  to P0's group in the middle, and leaves the lightest piece (weight 1) standing in front of it.

**(ii) Yes, with a caveat.** The opening is development with no contact: no push in the first
7 to 14 turns in any transcript, darts racing ahead and stones walking up behind. The middle is
contact in the centre: bulls and rams shoving lone stones and lines (6400-self s0 m33 to m52,
25600-v-1600 s2 m13 to m30). The end is rim knock-offs, where the margin of pieces you can
still lose is what matters (25600-v-1600 s1 m44 to m51 hunts P1's last pieces into the g1/f2
corner). The caveat is that the centre-count ending the reason describes almost never happens
at strength: the count decided only 1 of my 20 strong self-play games, and it decided most of the
1600 games at ply 80.

**(iii) Yes.** There is no capture, so every result is a push-off decision, and the weight sum
decides which pushes exist:
- 25600-v-1600 s0: three bulls shoving lone stones (weight 4, bull strength 4) decide it. No dart
  can move a stone. All 13 pushes are bull-on-stone or lighter.
- 25600-v-1600 s1: ram shoves of bulls (m20) and stones (m24). The finish is bull shoves of a
  ram+dart pair and a bull (m44 to m51).
- 25600-v-1600 s2: multi-piece shoves: dart+bull = 4 by a bull (m17, m19), dart+stone = 5 and
  stone+ram = 5 by the ram (m26, m39).
- 6400-self s0: the ram moves a dart+stone+bull line of weight exactly 8 (m48, m50), then knocks
  out a dart+stone line (m54).
- 6400-self s1: three bull-on-stone shoves (m29, m30, m32) produce two of the three knock-outs.
- 6400-self s2: bull and ram shoves of bulls and stones (m27 to m56). The finish, Rc3>>c2 and
  Bf4>>g4, is ram and bull knocking out heavy pieces.
Caveat: in s0 and s1 the weight that matters is a single stone against a bull. Building lines
for weight ("stacking") is visible only in the other four.

**(iv) Yes** (I found no rush, fortress or forced line a person would find). I tried:
- a turtle: held against 1600 and lost 10 of 12 against 6400 and 25600;
- a rim raid: never quick;
- the shortest games: still 14 quiet turns before contact;
- forced lines: none found at the positions I scored.
What I did find instead: a seat lean that depends on the budget, with a handicap that costs player 1 a
loss margin; a ten-turn shuffle; bulls doing most of the knock-outs; and a stall to the
count at ply 80 at low strength.

**(v) No.** The core loop matches the reason: a race to the middle, shoving there, with lines,
rams cracking heavy lines and knock-outs at the rim. But three of the reason's claims fail:
- **The seat fix.** The reason names "first-move advantage" as the failure, and the missing dart as
  its cure. With equal armies, the leading seat flips with the budget (P1 8 of 10 at 6400, P0 6 of 8
  at 25600). With the handicap, P0 won 16 of 21 at 6400. Because the loss limit is absolute, the
  handicap really takes one of P1's losses away. That decided two of the three self-play
  transcripts at 3 losses each. All six transcripts go to player 0.
- **Dominant piece.** The bull makes 19 of 29 knock-outs, and one transcript is played with bulls alone.
- **Shuffle.** The shuffle the reason lists as a failure appears (6400-self s1 m13 to m22).
- **The centre count.** The reason says the count ending keeps a locked game from going dead. It
  almost never fires at strength, and at low strength it ends the game after 40 turns, before the
  sides really meet.
