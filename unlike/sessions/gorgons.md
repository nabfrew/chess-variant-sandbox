# gorgons: designer session record (`unlike-search.md` §2)

**Admitted at the second read** (revision `43e5b764b9a8`; first read of `2054034c332b`: i y, ii y, iii n, iv y, v y; second read: i y, ii y, iii y, iv y, v y). Deviation: the session's heavy-lane use was about 80 slot-minutes before the revision and two more 96-game tournaments in it, over the designer prompt's one slot-hour; every tournament stayed within 96 games.


## First hand-in: design reason

# Design reason (written before the spec)

## The seed

Lines of sight that freeze or command. The game is built so that *lines* are the whole
board: what a piece is worth depends less on where it can go than on which lines it opens,
closes or stands in.

## The game in one paragraph

A hexagonal board. Each side has a royal piece to protect (the oracle), two gorgons, a lure
and a handful of plain fighters (spearmen). A gorgon stares down three lines, 120 degrees
apart, to the edge of the board; the first enemy piece on each line is frozen while it stands
there (it cannot move, and threatens nothing). A gaze stops at the first piece of either side,
so any piece in the way is a screen. The lure looks two cells along each of the six lines; an
enemy piece it sees may be moved by you, as its owner would move it, onto an empty cell. Only
spearmen and the oracle take. You win by taking the oracle, or when your opponent has no
legal move at all (a side frozen solid has lost).

## The decision a player keeps facing

Every move is a question about lines: *open one or close one?* A piece standing in an
enemy gaze is frozen and helpless, a target for a spearman; but it is also a screen, the only
thing keeping the gaze off whatever stands behind it. Moving a screen away "discovers" the gaze
onto the next piece, yours or theirs; stepping a piece into a line sacrifices its freedom to
blind the gorgon behind it. Two gorgons that see each other freeze each other, a standoff only
a third piece can break. The lure turns the question round: it reaches into the enemy's
screen and pulls a piece out of it, so a wall that looks solid is solid only out of the lure's
reach. The trade-off is never settled, because a screen that protects one line is usually
standing in another.

## The shape from opening to end

- **Opening:** the armies start screened, every gorgon's line blocked by its own pieces.
  Spearmen advance into the empty middle and both sides try to bring a gorgon to bear on a
  line the other cannot easily block, while keeping their own gorgons out of the opponent's
  gaze (gorgons are blind on three sides, so flanks matter).
- **Middle game:** frozen pieces pile up as living walls. Captures come from combinations:
  freeze, then strike; or lure a screen away, then strike or freeze what it hid. Mutual
  gorgon standoffs fix parts of the board, and the fight moves to the lines they do not cover.
- **End:** with fewer screens left, gazes reach far. The oracle, which moves one cell, must
  stay out of open gaze lines; one frozen oracle with a spearman two cells away is lost. The
  game ends by striking a frozen or cornered oracle, or by freezing everything the opponent has.

## What would make it fail

- **A rush:** a gorgon line onto the oracle that opens in the first few moves and cannot be
  blocked, or a spearman raid that wins before the lines matter.
- **A fortress or shuffle:** both sides freeze each other's front pieces and nobody can make
  progress, so games end at the repetition or the clock; or a side that is behind can wall
  itself in with frozen pieces.
- **A dominant piece:** spearmen doing all the work while the gorgons and the lure are
  decoration (a dead seed), or gorgons so strong that whoever lands the first gaze wins.
- **A dead rule:** the lure's command rarely used, or "no legal move loses" never firing
  and adding nothing.
- **The seat:** the first player lands the first gaze or the first strike and converts it.


## First hand-in: break attempts and design log

# gorgons: design log and break attempts (session 7)

Frozen spec: `game.json`, **gorgons [2054034c332b]** (= draft v13). Seed: lines of sight
that freeze or command. Design reason: `reason.md`, written before any spec and not edited
since.

Instruments used: `validate`, `smoke`, `unlike/limits.py`, `play`, Engine probes (scripts in
this folder: `probe.py` draws a position, `summ.py` reads results by seat, lengths and end
reasons from `games.jsonl`, `tally.py` and `dump.py` replay transcripts through `Engine` to
count who moves, who takes, how much stands frozen and how often a command is on offer,
`fortress.py` plays a built position out). Tournaments: 13, every one exactly 96 games (two
agents, four ordered pairings, 24 games each), all `minimax:nodes=6400` against
`minimax:nodes=25600` except the first (1600 against 6400). They ran 25 to 122 s each on four
slots, roughly 80 slot-minutes in all with the fortress probe: a little over the one
slot-hour budget. No score, test, measure, analyse, premise, verdict, heuristics, values or
train was run, and no index, component or family was read.

## The design log

Every version: hexagonal board of 61 cells (`hex`, `hexagon`, 9), rotated setup, nine pieces
a side (oracle g1, gorgons f1 h1, lure g2, spearmen d3 to h3), every piece steps one cell,
royal capture, a player with no legal move loses, 300-ply limit.

| version | hash | change | why | what happened |
|---|---|---|---|---|
| v1 | 2899ce2f91eb | first draft: spearmen slide up to 2 cells six ways; gorgon freezes enemy on three lines 120 degrees apart, unlimited; lure commands enemy on six lines, range 2 | the design reason | one 6400 game: spearmen traded off in the first ten moves and the gaze game started only after; limits (a)-(c) all yes |
| v2 | f26f5bfab39c | spearmen step one cell | predicted: capture needs freeze-then-strike, gorgons central, slower | as predicted: 1.7 to 1.9 pieces frozen per ply, freeze-outs (no legal move) ended 16 of 96 games, commands 3 to 6% of moves. At 25600 self 7-8-9 (seat0-seat1-draw), but 6 of the 9 draws were a **tug of war**: a spearman steps into the lure's reach, the lure pulls it back, repeat (break 2) |
| v3 | b5b09b142c0e | rest 1 on spearman, gorgon and lure | stop the tug: a moved piece must stand a turn | worse. The oracle became the only free piece and was shuffled (oracle moves 4224 against spearmen 3349); 25600 self 3-9-12, the ladder flattened (25600 vs 6400 10-9). Reverted |
| v4 | add3719b0559 | v2, spearmen step but capture up to 2 cells along a line | give attackers tempo against the lure | draws unchanged (25600 self 7-8-9), first capture moved to ply 7 (a trading opening again), commands fell. Reverted |
| v5 | 7773f5b08f8b | v2 plus no repetition draw, 60-ply stall clock scored by pieces on the seven centre cells | punish the side that only tugs | worse: half the games decided at the clock, seat 0 51 to 27, and 6400 beat 25600 13-7 as first player. Reverted |
| v6 | 853a190b1e80 | v2, lure commands only gorgons and lures | no spearman to tug | decisive (25600 self 9-12-3) but commands fell to 0.7% of moves: the seed's second half nearly dead. Kept as evidence |
| v7 | d0dad035ba83 | v2, lure range 1 | smaller tug zone | 25600 self 8-9-7, commands 1.1% |
| v8 | 96c644a1c254 | v2, lure sees only the three lines the gorgon does not, range 2 | tidy complement to the gorgon, smaller zone | 25600 self 10-10-4, commands 1.2%, on offer 5% of plies |
| v8n | 1c55d46698e9 | v8 with the lure's sight deleted | dead-rule test of v8's command | played the same as v8 (25600 self 10-11-3, same lengths, same captures): **v8's command was decoration** (break 5). v8 rejected |
| v9 | 90e8f28cf2d3 | v8, lure's three lines unlimited | a command that matters | early tugs again: three 15-ply repetitions, 6400 self 12-1-11. Rejected |
| v10 | f18539c3cdae | v8 plus `captures_own` (a pulled spearman may be made to take its own side) | give the command teeth | 8 turncoat captures in 96 games, commands still 1.2%: barely alive. Rejected |
| v11 | 287ec2885e14 | v2 plus rest 1 on spearmen only | break the spearman tug while gorgons and lure stay free | 25600 self 10-11-3, commands back to 5.8% and on offer 13.5% of plies. But the tug moved to gorgons (a pulled gorgon has no rest; a 6400 game ended `~Gd6-c7 Gc7-d6` repeated) |
| **v13** | **2054034c332b** | v11, lure commands spearmen only | a commanded piece is now always one that must then stand a turn, so no command can be undone at once | 25600 self 10-12-2; 25600 vs 6400 15-1-8 and 16-5-3 (stronger side wins at both seats); commands 4.4% of moves, on offer 7.9% of plies; seat totals 36-35. **Frozen** |

(v12 was a spelling I discarded before running.)

What play changed in my view, relative to `reason.md`: the lure as first written (pull any
piece within two cells) was a perpetual-defence machine, because a command onto an empty cell
can be undone by its owner on the next move. The fix that kept the command alive was not to
weaken the lure but to make a moved spearman stand still a turn: then a pull is a real
displacement (the pulled spearman stays where the lure put it, in a gaze or next to an enemy
spearman), and the lure was narrowed to spearmen so that nothing it can pull can be pulled
straight back. The rest rule is the one rule added beyond the reason, and it is there for that.

## Break attempts

1. **A rush.** The oracles start screened: a gaze can reach the first player's oracle (g1)
   only down the g file (blocked by the lure and a spearman) or along the bottom row, and the
   second player's symmetrically. Every short decisive game I read (v13: 45, 52, 65 plies) was
   an oracle that walked forward to take something and was frozen beside a spearman, not an
   opening plan; the first capture comes at ply 28 (median) on v13. No rush found. On v1 the
   opening *was* a forced-looking spearman trade (fixed by v2).
2. **A shuffle: the tug of war.** Found on v2 (6 of 9 draws at 25600 self were a pulled
   spearman pushed back and pulled again; `t2-v2` games 74, 78). Tried four fixes (v3 rest on
   all, v5 centre scoring, v6 to v10 narrower lures) before v11/v13. On v13 no tug is possible
   (a pulled spearman stands through its owner's turn and nobody may move it), and 25600 self
   has 2 draws in 24, both on the 100-move stall clock.
3. **A fortress.** `fortress.py` on v13: the weaker side (oracle, gorgon, lure, three
   spearmen) tucks its oracle in the e1 corner behind three spearmen; the other side has an
   extra gorgon and spearman. Both played by `minimax:nodes=25600`, four seeds each side to
   move first: the stronger side won 5, drew 2 (a stall-clock and a ply-limit game), lost 1.
   Frozen walls do not hold, because a frozen piece can still be taken. Not a fortress, though
   the side ahead does not always convert.
4. **A forced line.** None found in the transcripts; captures are spread (spearmen 533,
   oracles 173 of the takings in `t-v13`) and lengths run 45 to 300 plies.
5. **A dead rule.** The command, on v8: deleting it changed nothing (v8n), so v8 was rejected.
   On v13 the command is on offer in 8% of positions and played in 4.4% of moves, with the
   freeze at about 1.9 pieces frozen per ply. "No legal move loses" decides 4 to 7 games in 96
   (freeze-outs; without it those are draws). The rest rule is the tug fix (v2 against v13).
6. **A dominant piece or move.** On v13 own moves split spearmen 3710, gorgons 3402, oracles
   3101, lures 2099. The oracle is a real fighter (a sixth of the captures), which is also its
   danger: most short games are an oracle caught after it went forward.
7. **The seat.** v13 over the tournament: seat 0 36, seat 1 35, draws 25; at 25600 self 10 to
   12. v2's 6400 self leaned to seat 1 (14 to 26 over 48 games), which went away at 25600 and on v13.

## What I could not fully close

- **Weak players shuffle.** At `minimax:nodes=6400` self-play v13 draws 12 of 24 (8 by
  repetition): gorgons and oracles dancing in a closed position (for example `Gh4-i4 Gb6-a6`
  repeated, or two oracles stepping back and forth), not a tug. At 25600 the same opening is
  played out (2 draws in 24). I read it as the weaker bot finding no plan rather than a rule
  that forbids one, but a person who also finds no plan will draw. Two of the six hand-in
  transcripts (6400 self, seeds 0 and 2) end in draws.
- The stall clock (100 plies without a capture) still ends a few strong games.

## Final checks (frozen spec)

```
gorgons [2054034c332b] is valid
  board       9x9
  pieces      O, G, L, S
  opening     28 legal moves
  rule mass   51.5
game.json [2054034c332b]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game jeson mor [fb8a67625131] 0.1299  below 0.5: yes
limits exit 0
gorgons [2054034c332b]: passed
  draw_rate        0.48
  forced_rate      0.0
  games            150.0
  mean_branching   25.13
  mean_plies       202.0
  opening_moves    28.0
  stall_rate       0.48
```

(`smoke` plays random movers; its 0.48 draw rate is random players running into the stall
clock in a game where everything steps one cell.)

## Hand-in transcripts (frozen spec)

| file | result |
|---|---|
| `transcripts/6400-self-seed0.txt` | draw, no progress, 199 plies |
| `transcripts/6400-self-seed1.txt` | player 0 by taking the oracle, 211 plies |
| `transcripts/6400-self-seed2.txt` | draw, repetition, 81 plies |
| `transcripts/25600-v-1600-seed0.txt` | player 0 (25600) by taking the oracle, 221 plies |
| `transcripts/25600-v-1600-seed1.txt` | player 0 (25600) by freeze-out (no legal move), 127 plies |
| `transcripts/25600-v-1600-seed2.txt` | player 0 (25600) by taking the oracle, 151 plies |

Tournament records are in `runs/R-S10/sessions/7/` (`t1-v2` to `t7-v7`, `t-v8`, `t-v8n`,
`t-v9`, `t-v10`, `t-v11`, `t-v13`).


## Revised hand-in: design reason

# Design reason (written before the spec)

## The seed

Lines of sight that freeze or command. The game is built so that *lines* are the whole
board: what a piece is worth depends less on where it can go than on which lines it opens,
closes or stands in.

## The game in one paragraph

A hexagonal board. Each side has a royal piece to protect (the oracle), two gorgons, a lure
and a handful of plain fighters (spearmen). A gorgon stares down three lines, 120 degrees
apart, to the edge of the board; the first enemy piece on each line is frozen while it stands
there (it cannot move, and threatens nothing). A gaze stops at the first piece of either side,
so any piece in the way is a screen. The lure looks two cells along each of the six lines; an
enemy piece it sees may be moved by you, as its owner would move it, onto an empty cell. Only
spearmen and the oracle take. You win by taking the oracle, or when your opponent has no
legal move at all (a side frozen solid has lost).

## The decision a player keeps facing

Every move is a question about lines: *open one or close one?* A piece standing in an
enemy gaze is frozen and helpless, a target for a spearman; but it is also a screen, the only
thing keeping the gaze off whatever stands behind it. Moving a screen away "discovers" the gaze
onto the next piece, yours or theirs; stepping a piece into a line sacrifices its freedom to
blind the gorgon behind it. Two gorgons that see each other freeze each other, a standoff only
a third piece can break. The lure turns the question round: it reaches into the enemy's
screen and pulls a piece out of it, so a wall that looks solid is solid only out of the lure's
reach. The trade-off is never settled, because a screen that protects one line is usually
standing in another.

## The shape from opening to end

- **Opening:** the armies start screened, every gorgon's line blocked by its own pieces.
  Spearmen advance into the empty middle and both sides try to bring a gorgon to bear on a
  line the other cannot easily block, while keeping their own gorgons out of the opponent's
  gaze (gorgons are blind on three sides, so flanks matter).
- **Middle game:** frozen pieces pile up as living walls. Captures come from combinations:
  freeze, then strike; or lure a screen away, then strike or freeze what it hid. Mutual
  gorgon standoffs fix parts of the board, and the fight moves to the lines they do not cover.
- **End:** with fewer screens left, gazes reach far. The oracle, which moves one cell, must
  stay out of open gaze lines; one frozen oracle with a spearman two cells away is lost. The
  game ends by striking a frozen or cornered oracle, or by freezing everything the opponent has.

## What would make it fail

- **A rush:** a gorgon line onto the oracle that opens in the first few moves and cannot be
  blocked, or a spearman raid that wins before the lines matter.
- **A fortress or shuffle:** both sides freeze each other's front pieces and nobody can make
  progress, so games end at the repetition or the clock; or a side that is behind can wall
  itself in with frozen pieces.
- **A dominant piece:** spearmen doing all the work while the gorgons and the lure are
  decoration (a dead seed), or gorgons so strong that whoever lands the first gaze wins.
- **A dead rule:** the lure's command rarely used, or "no legal move loses" never firing
  and adding nothing.
- **The seat:** the first player lands the first gaze or the first strike and converts it.


## Revised hand-in: break attempts and design log

# gorgons: design log and break attempts (session 7)

**Revised hand-in: `game.json` is now gorgons [43e5b764b9a8] (draft r2). See *Revision* at
the end. The sections up to there are the first hand-in's log, kept as written; its spec was
gorgons [2054034c332b] (= draft v13), preserved in `first-handin\`.**

First hand-in's frozen spec: **gorgons [2054034c332b]** (= draft v13). Seed: lines of sight
that freeze or command. Design reason: `reason.md`, written before any spec and not edited
since.

Instruments used: `validate`, `smoke`, `unlike/limits.py`, `play`, Engine probes (scripts in
this folder: `probe.py` draws a position, `summ.py` reads results by seat, lengths and end
reasons from `games.jsonl`, `tally.py` and `dump.py` replay transcripts through `Engine` to
count who moves, who takes, how much stands frozen and how often a command is on offer,
`fortress.py` plays a built position out). Tournaments: 13, every one exactly 96 games (two
agents, four ordered pairings, 24 games each), all `minimax:nodes=6400` against
`minimax:nodes=25600` except the first (1600 against 6400). They ran 25 to 122 s each on four
slots, roughly 80 slot-minutes in all with the fortress probe: a little over the one
slot-hour budget. No score, test, measure, analyse, premise, verdict, heuristics, values or
train was run, and no index, component or family was read.

## The design log

Every version: hexagonal board of 61 cells (`hex`, `hexagon`, 9), rotated setup, nine pieces
a side (oracle g1, gorgons f1 h1, lure g2, spearmen d3 to h3), every piece steps one cell,
royal capture, a player with no legal move loses, 300-ply limit.

| version | hash | change | why | what happened |
|---|---|---|---|---|
| v1 | 2899ce2f91eb | first draft: spearmen slide up to 2 cells six ways; gorgon freezes enemy on three lines 120 degrees apart, unlimited; lure commands enemy on six lines, range 2 | the design reason | one 6400 game: spearmen traded off in the first ten moves and the gaze game started only after; limits (a)-(c) all yes |
| v2 | f26f5bfab39c | spearmen step one cell | predicted: capture needs freeze-then-strike, gorgons central, slower | as predicted: 1.7 to 1.9 pieces frozen per ply, freeze-outs (no legal move) ended 16 of 96 games, commands 3 to 6% of moves. At 25600 self 7-8-9 (seat0-seat1-draw), but 6 of the 9 draws were a **tug of war**: a spearman steps into the lure's reach, the lure pulls it back, repeat (break 2) |
| v3 | b5b09b142c0e | rest 1 on spearman, gorgon and lure | stop the tug: a moved piece must stand a turn | worse. The oracle became the only free piece and was shuffled (oracle moves 4224 against spearmen 3349); 25600 self 3-9-12, the ladder flattened (25600 vs 6400 10-9). Reverted |
| v4 | add3719b0559 | v2, spearmen step but capture up to 2 cells along a line | give attackers tempo against the lure | draws unchanged (25600 self 7-8-9), first capture moved to ply 7 (a trading opening again), commands fell. Reverted |
| v5 | 7773f5b08f8b | v2 plus no repetition draw, 60-ply stall clock scored by pieces on the seven centre cells | punish the side that only tugs | worse: half the games decided at the clock, seat 0 51 to 27, and 6400 beat 25600 13-7 as first player. Reverted |
| v6 | 853a190b1e80 | v2, lure commands only gorgons and lures | no spearman to tug | decisive (25600 self 9-12-3) but commands fell to 0.7% of moves: the seed's second half nearly dead. Kept as evidence |
| v7 | d0dad035ba83 | v2, lure range 1 | smaller tug zone | 25600 self 8-9-7, commands 1.1% |
| v8 | 96c644a1c254 | v2, lure sees only the three lines the gorgon does not, range 2 | tidy complement to the gorgon, smaller zone | 25600 self 10-10-4, commands 1.2%, on offer 5% of plies |
| v8n | 1c55d46698e9 | v8 with the lure's sight deleted | dead-rule test of v8's command | played the same as v8 (25600 self 10-11-3, same lengths, same captures): **v8's command was decoration** (break 5). v8 rejected |
| v9 | 90e8f28cf2d3 | v8, lure's three lines unlimited | a command that matters | early tugs again: three 15-ply repetitions, 6400 self 12-1-11. Rejected |
| v10 | f18539c3cdae | v8 plus `captures_own` (a pulled spearman may be made to take its own side) | give the command teeth | 8 turncoat captures in 96 games, commands still 1.2%: barely alive. Rejected |
| v11 | 287ec2885e14 | v2 plus rest 1 on spearmen only | break the spearman tug while gorgons and lure stay free | 25600 self 10-11-3, commands back to 5.8% and on offer 13.5% of plies. But the tug moved to gorgons (a pulled gorgon has no rest; a 6400 game ended `~Gd6-c7 Gc7-d6` repeated) |
| **v13** | **2054034c332b** | v11, lure commands spearmen only | a commanded piece is now always one that must then stand a turn, so no command can be undone at once | 25600 self 10-12-2; 25600 vs 6400 15-1-8 and 16-5-3 (stronger side wins at both seats); commands 4.4% of moves, on offer 7.9% of plies; seat totals 36-35. **Frozen** |

(v12 was a spelling I discarded before running.)

What play changed in my view, relative to `reason.md`: the lure as first written (pull any
piece within two cells) was a perpetual-defence machine, because a command onto an empty cell
can be undone by its owner on the next move. The fix that kept the command alive was not to
weaken the lure but to make a moved spearman stand still a turn: then a pull is a real
displacement (the pulled spearman stays where the lure put it, in a gaze or next to an enemy
spearman), and the lure was narrowed to spearmen so that nothing it can pull can be pulled
straight back. The rest rule is the one rule added beyond the reason, and it is there for that.

## Break attempts

1. **A rush.** The oracles start screened: a gaze can reach the first player's oracle (g1)
   only down the g file (blocked by the lure and a spearman) or along the bottom row, and the
   second player's symmetrically. Every short decisive game I read (v13: 45, 52, 65 plies) was
   an oracle that walked forward to take something and was frozen beside a spearman, not an
   opening plan; the first capture comes at ply 28 (median) on v13. No rush found. On v1 the
   opening *was* a forced-looking spearman trade (fixed by v2).
2. **A shuffle: the tug of war.** Found on v2 (6 of 9 draws at 25600 self were a pulled
   spearman pushed back and pulled again; `t2-v2` games 74, 78). Tried four fixes (v3 rest on
   all, v5 centre scoring, v6 to v10 narrower lures) before v11/v13. On v13 no tug is possible
   (a pulled spearman stands through its owner's turn and nobody may move it), and 25600 self
   has 2 draws in 24, both on the 100-move stall clock.
3. **A fortress.** `fortress.py` on v13: the weaker side (oracle, gorgon, lure, three
   spearmen) tucks its oracle in the e1 corner behind three spearmen; the other side has an
   extra gorgon and spearman. Both played by `minimax:nodes=25600`, four seeds each side to
   move first: the stronger side won 5, drew 2 (a stall-clock and a ply-limit game), lost 1.
   Frozen walls do not hold, because a frozen piece can still be taken. Not a fortress, though
   the side ahead does not always convert.
4. **A forced line.** None found in the transcripts; captures are spread (spearmen 533,
   oracles 173 of the takings in `t-v13`) and lengths run 45 to 300 plies.
5. **A dead rule.** The command, on v8: deleting it changed nothing (v8n), so v8 was rejected.
   On v13 the command is on offer in 8% of positions and played in 4.4% of moves, with the
   freeze at about 1.9 pieces frozen per ply. "No legal move loses" decides 4 to 7 games in 96
   (freeze-outs; without it those are draws). The rest rule is the tug fix (v2 against v13).
6. **A dominant piece or move.** On v13 own moves split spearmen 3710, gorgons 3402, oracles
   3101, lures 2099. The oracle is a real fighter (a sixth of the captures), which is also its
   danger: most short games are an oracle caught after it went forward.
7. **The seat.** v13 over the tournament: seat 0 36, seat 1 35, draws 25; at 25600 self 10 to
   12. v2's 6400 self leaned to seat 1 (14 to 26 over 48 games), which went away at 25600 and on v13.

## What I could not fully close

- **Weak players shuffle.** At `minimax:nodes=6400` self-play v13 draws 12 of 24 (8 by
  repetition): gorgons and oracles dancing in a closed position (for example `Gh4-i4 Gb6-a6`
  repeated, or two oracles stepping back and forth), not a tug. At 25600 the same opening is
  played out (2 draws in 24). I read it as the weaker bot finding no plan rather than a rule
  that forbids one, but a person who also finds no plan will draw. Two of the six hand-in
  transcripts (6400 self, seeds 0 and 2) end in draws.
- The stall clock (100 plies without a capture) still ends a few strong games.

## Final checks (frozen spec)

```
gorgons [2054034c332b] is valid
  board       9x9
  pieces      O, G, L, S
  opening     28 legal moves
  rule mass   51.5
game.json [2054034c332b]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game jeson mor [fb8a67625131] 0.1299  below 0.5: yes
limits exit 0
gorgons [2054034c332b]: passed
  draw_rate        0.48
  forced_rate      0.0
  games            150.0
  mean_branching   25.13
  mean_plies       202.0
  opening_moves    28.0
  stall_rate       0.48
```

(`smoke` plays random movers; its 0.48 draw rate is random players running into the stall
clock in a game where everything steps one cell.)

## Hand-in transcripts (frozen spec)

| file | result |
|---|---|
| `transcripts/6400-self-seed0.txt` | draw, no progress, 199 plies |
| `transcripts/6400-self-seed1.txt` | player 0 by taking the oracle, 211 plies |
| `transcripts/6400-self-seed2.txt` | draw, repetition, 81 plies |
| `transcripts/25600-v-1600-seed0.txt` | player 0 (25600) by taking the oracle, 221 plies |
| `transcripts/25600-v-1600-seed1.txt` | player 0 (25600) by freeze-out (no legal move), 127 plies |
| `transcripts/25600-v-1600-seed2.txt` | player 0 (25600) by taking the oracle, 151 plies |

Tournament records are in `runs/R-S10/sessions/7/` (`t1-v2` to `t7-v7`, `t-v8`, `t-v8n`,
`t-v9`, `t-v10`, `t-v11`, `t-v13`).

---

# Revision (after the blind reading of 2054034c332b)

The reader answered no to (iii): in `6400-self-seed2` the lines of sight decided nothing (two
gorgons lost by stepping next to unfrozen spearmen, no command, a repetition at ply 81). It
also noted that the bots fail to convert material and cannot break a passive side, and that
the second player led 20 to 9 in its self-play at 12800 nodes and up.

## Diagnosis

The three complaints share a cause. In v13 every piece steps any way and a moved spearman
must stand a turn, so **approaching is punished**: a spearman that steps up stands still and
is taken first. Whoever must approach first is worse off (the second-player lead), the
natural strategy is to wait (shuffles, unconverted edges), and while both sides wait the gazes
have nothing to bite on (the game where sight decided nothing). The rest rule was there only
to stop the lure's tug of war, which comes from a pull being undoable: the owner moves the
pulled spearman straight back.

The fix aimed at the cause: **spearmen only go forward** (one step to either of the two cells
ahead, taking there), and the rest rule goes. A pulled spearman is moved as its owner would
move it, so it can only be pulled forward, toward the puller, and its owner can never move it
back. That ends the tug structurally, with no rest. Spearmen that cannot retreat also make
every spearman move permanent, so a game cannot idle as long: the armies close, and a side
whose spearmen run out of road and whose other pieces are frozen has no move and loses.
Prediction, written before running it: fewer draws, sight deciding more games (freeze-outs),
seat near even, commands rarer (a forward pull also brings the spearman nearer your pieces,
where it threatens them).

## Versions in the revision

| version | hash | change | why | what happened |
|---|---|---|---|---|
| r1 | d673aad014d0 | v13 plus a second gorgon sight: freeze enemy spearmen on the three adjacent cells it does not stare at | stop gorgons dying to adjacent spearmen | 6 single 6400 self games: 3 draws, all shuffles of oracles and lures. The fragility was not the problem. Dropped, no tournament |
| **r2** | **43e5b764b9a8** | v13 with spearmen forward-only (`[0,1]`, `[-1,1]`, range 1, move and take), no rest | the diagnosis above | single 6400 games: 5 of 6 decisive, 4 by freeze-out. Tournament `rev-r2` (6400 vs 25600, 96 games): 25600 self **12-12-0**; 25600 beats 6400 17-4-3 and 16-5-3; 6400 self 8-11-5; seat totals 42-43, draws 11. Endings: freeze-out 41, oracle taken 44. 2.5 pieces frozen per ply. Commands 1.6% of moves, on offer in 12.5% of positions. **Frozen** |
| r3 | d260aba341b7 | r2 with the lure's reach unlimited | more commands | 6 single 6400 games: commands 10 in all (r2: 9), 5 of 6 decisive but 5 won by the second player. No gain; dropped, no tournament |
| r2n | cc318753666c | r2 with the lure's sight deleted | dead-rule test of the command | tournament `rev-r2n`: 25600 self 7-12-5 (4 repetitions) against r2's 12-12-0; 6400 self 14-8-2. Without the command the strong games draw by repetition and lean to the second seat; with it neither. Small samples, but the command is not decoration |

Two tournaments in the revision (`rev-r2`, `rev-r2n`), each 96 games with self-pairings counted,
plus two small probes on one slot (the fortress probe, 8 games; the passive-defence probe, 4 games).

`reason.md` is kept unchanged: it still describes the game (the decision, open or close a
line; the shape; the failure list). Two details in it are now narrower than the game: the lure
moves only enemy spearmen, and spearmen only step forward, so a pulled one moves toward you.
The gorgon standoff it describes still happens and now matters more (see below).

## Break attempts on r2

1. **A rush.** The reader's f-file rush (1.Sf3-e4 discovers the f1 gorgon onto f7) is still
   there and is still answered by a screen; with forward-only spearmen the screen cannot be
   pulled back either. Shortest decisive games in `rev-r2`: 47 plies at 25600 self, all
   middle-game freezes, none an opening plan. No rush found.
2. **A shuffle or tug.** The tug cannot happen (a pulled spearman cannot go back). 25600 self
   has no draws in 24. Shuffles remain among weak players (6400 self 5 draws in 24).
3. **A fortress.** `fortress.py` on r2 (the turtle in the e1 corner, a gorgon and a spearman
   down; both sides `minimax:nodes=25600`, 8 games): the stronger side won 6, the turtle 2, no
   draws (v13: 5 won, 2 drawn, 1 lost).
4. **Passive defence** (the reader's test). `passive.py`: the second player only shuffles a
   gorgon b9/a9/a8 while it can, against `minimax:nodes=25600`. Result: one win for the active
   side (207 plies), two games to the 300-ply limit. In the seed-0 line the bot does find the
   plan (gorgon up the f-file freezes f7, a spearman walks g4-g5-g6 and takes it, 109.Sg6xf7)
   but then wanders, and the stall clock and the ply limit come first. The passive side never
   had to leave its shuffle there. So the bots still cannot convert against pure passivity; the
   winning plan exists and the bot itself plays its first stage. **Not closed** as a bot
   matter; I found no fortress that holds against a plan.
5. **A dead rule.** The command: r2n above. The freeze: 2.5 pieces frozen per ply; freeze-outs
   end 41 of 96 games. "No legal move loses": those 41 games. Nothing dead.
6. **A dominant piece.** Own moves in `rev-r2`: gorgons 5173, oracles 3198, spearmen 2611,
   lures 2476. Takers: spearmen 558, oracles 391. The oracle is a big fighter now that spearmen
   cannot turn back to defend; it is also what gets frozen and taken.
7. **The seat.** `rev-r2`: seat 0 42, seat 1 43; 25600 self 12-12. The reader's 20-9 lean
   for v13 is not reproduced. (r3's 6-game sample leaned to seat 1, one reason it was dropped.)

## What I could not fully close

- **Locked gorgons in the endgame.** Two of the three drawn hand-in transcripts end with all
  four gorgons frozen in two mutual standoffs and only oracles and a few spearmen left
  (`25600-v-1600-seed0`, no progress, and `seed2`, repetition, where the 25600 side was two
  spearmen and a lure up). Breaking a standoff costs a piece stepped into the line; with few
  pieces left nobody does it, and spearmen that only go forward cannot hunt an oracle behind
  them. These draws are made by the lines of sight, but they are draws.
- **Conversion against passivity** (break 4), as above.
- The 25600-vs-1600 hand-in seeds came out 1 win, 2 draws, although in `rev-r2` 25600 beat
  6400 33 to 9 with 6 draws.

## Final checks (revised frozen spec)

```
gorgons [43e5b764b9a8] is valid
  board       9x9
  pieces      O, G, L, S
  opening     18 legal moves
  rule mass   45.5
game.json [43e5b764b9a8]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game jeson mor [fb8a67625131] 0.1224  below 0.5: yes
limits exit 0
gorgons [43e5b764b9a8]: passed
  draw_rate        0.093
  forced_rate      0.0
  games            150.0
  mean_branching   16.22
  mean_plies       156.93
  opening_moves    18.0
  stall_rate       0.087
```

## Hand-in transcripts (revised frozen spec)

| file | result | the lines of sight in it |
|---|---|---|
| `transcripts/6400-self-seed0.txt` | player 1 takes the oracle, 62 plies | the oracle (e5) was frozen when taken, 31...Oe6xe5 |
| `transcripts/6400-self-seed1.txt` | draw, 300-ply limit | one command (not checked further) |
| `transcripts/6400-self-seed2.txt` | player 0 by freeze-out (no legal move), 213 plies | decided by the freeze |
| `transcripts/25600-v-1600-seed0.txt` | draw, no progress, 243 plies | ends with all four gorgons in mutual freezes |
| `transcripts/25600-v-1600-seed1.txt` | player 0 (25600) by freeze-out, 53 plies | decided by the freeze; two commands |
| `transcripts/25600-v-1600-seed2.txt` | draw, repetition, 108 plies | two commands; ends with all four gorgons in mutual freezes |

Revision records: `runs/R-S10/sessions/7/rev-r2`, `rev-r2n`.


## Blind read 1 (read-7)

# Reading: gorgons (reader 7)

Tools I used: `play` (single games), and position or line probes through `Engine` (`find_move`,
`apply`, `effects`, `select_move`, `root_scores`), in scripts in this packet (`probe.py`,
`stats.py`, `at.py`, `finish.py`, `close.py`, `rush.py`, `both.py`). No score, test, measure,
analyse, tournament, heuristics or values. No lease taken.

Hex coordinates: a gorgon (White) sees up its file, down-right diagonally and left along its
rank; Black's are mirrored. Rank 9 holds a9-e9 only, rank 1 e1-i1.

## Attempts

1. **Rush on the opened f-file.** 1.Sf3-e4 at once opens White's f1 gorgon up the f-file and
   freezes Black's f7 spearman. I wrote the obvious follow-up (g-pawn walk g4-g5-g6, then
   Sg6xf7, which reaches f7 from off the file so it does not screen the gaze) and let
   minimax:nodes=25600 answer for Black. Black put its own spearman into the gaze as a screen
   (3...Se7-f6), then moved another spearman in front of it (4...Se6-f5). That freed f6, which
   took my g5 spearman (5...Sf6xg5). The rush loses a spearman. Against a Black that does
   nothing the same plan does win f7 (checked by replaying both sides by hand), so the threat
   is real, but the screen answers it.
2. **Oracle march.** The oracle never rests and takes, so it might be the best fighter. I walked
   it to f4 by move 4 and let the bot play on both sides from there. Nothing came of it: Black
   won a spearman with a lure pull (`~Sd5-d6`, then `Se6xd6`), and the oracle had no target.
   No rush.
3. **Fortress/shuffle as Black.** Black shuffles one gorgon a9-a8-a9 forever and touches nothing
   else. White at minimax:nodes=25600 and again at 102400 never broke in. It shuffled its own
   gorgons until a no-progress draw (25600) and a repetition (102400). That is a failure of the
   bots, not a fortress: by hand, with both sides scripted, the f-file freeze plus a spearman
   reaching g6 takes the frozen f7 against exactly this passive Black. So I found no fortress
   that holds against a person. But the bots cannot break a passive side, and that matters for
   the draws below.
4. **Converting a material edge.** 6400-self-seed2 ends in a repetition at ply 81 with White two
   gorgons up. 6400-self-seed0 ends in a no-progress draw with White a lure and two spearmen up,
   no gorgons left on either side. A 100000-node root search at ply 80 of seed2 scores every
   White move the same (+0.1), so the evaluator sees no plan. I could not show either position
   is a fortress. I think both are winnable by hand: guarded spearman pairs against the oracle,
   and a gorgon on e9 to freeze an oracle on b9 along rank 9. So I count them as bot failures,
   not proven fortresses, but they are unresolved.
5. **Dead rule.** The lure's command appears in 5 of 6 transcripts (9+3, 6, 7+7, 8, 2+1 commanded
   moves). It is the source of a recurring tactic: pull a spearman next to yours, where the rest
   rule keeps it from escaping, then take it. "No legal move loses" decides 25600-v-1600-seed1,
   and about 10 of the roughly 50 extra games I played (for example 25600 self-play seeds 6
   and 11, 12800 seeds 2, 4, 7, 8, 9). The rest rule shapes every approach. No dead rule found.
6. **Dominant piece.** Captures are by spearmen (most) and oracles (about 9 across the six
   games). But each decisive capture depends on a freeze (below) and most material swings on a
   lure pull. Gorgons take nothing and are often lost by stepping next to an unfrozen spearman
   (6400 seeds 0 and 2), so they are fragile, not dominant. No dominant piece.
7. **Seat.** Self-play, first seat W / second seat W / draws:
   - minimax:nodes=6400, seeds 3-12: 3 / 3 / 4.
   - minimax:nodes=12800, seeds 0-13: 6 / 8 / 0.
   - minimax:nodes=25600, seeds 0-13: 2 / 9 / 3.
   - minimax:nodes=51200, seeds 0-3: 1 / 3 / 0.

   At 12800 nodes and above the second player leads 20 to 9. A likely cause is that the rest
   rule punishes whoever approaches first: a spearman that steps up must stand a turn and is
   taken. This is the opposite of the first-mover worry in the design reason. It is a lean, not
   a forced line. The first player still wins games at every budget.
8. **Skill gradient (sanity).** minimax:nodes=25600 beat greedy 3/3 and random 3/3 (both seats),
   and minimax:nodes=6400 beat mcts 3/3. Search depth pays.

## The five questions

**(i) Yes.** 6400-self-seed1, move 13, Black to move (after 13.Gi3-i2). Black's lure has just
pulled White's spearman to d6 (12...~Sd5-d6). It is no longer resting and threatens c7 and the
lure on d7. White's c4 spearman attacks Black's gorgon on b5, which is freezing White's c5.
- **13...Sc7xd6 (played)** wins the spearman at once. It gives up the gorgon (14.Sc4xb5) and
  with it the freeze on c5.
- **13...Gb5-a6** keeps the gorgon. From a6 its gaze along rank 6 runs b6, c6 and freezes d6,
  so the threat is stopped and d6 stays a target. It gives up the immediate capture and unfreezes
  c5, and White can screen the rank-6 gaze with Sc5-c6, which brings d6 back to life.

A 100000-node root search scores the two at +3.14 and +3.09, nearly level. The choice is the
game's own: a gaze kept against material taken now.

**(ii) Yes.** The three phases differ in what matters:
- **Opening:** whose gaze is screened. Moving f3 discovers the f-file onto f7 at move 1, Black
  answers by screening with its own spearman (25600 seeds 0 and 1, 3...Se7-f6), and the gorgons
  walk to the blind flanks (every Black game starts Gb9-a9-a8-a7-a6-b5).
- **Middle:** lure pulls and freeze-then-strike swaps, with frozen pieces as walls (up to 7
  frozen at once, 6400-seed0 ply 38). There are mutual gorgon standoffs: Gf5 and g5 freeze
  each other in 6400-seed0, and the oracle takes the frozen gorgon, 21...Of6xg5.
- **End:** a hunt for a frozen oracle or a freeze-out.

The caveat is that the middle can stall. In 25600-seed0 there are about 40 moves (moves 43-82)
with no capture, all gorgon, oracle and lure shuffles, before contact resumes.

**(iii) No.** It decides something in five of the six, not in all six.
- **25600-v-1600-seed0:** won by 111.Sc6xd6 on an oracle that Gd3 had frozen up the d-file for
  the last 6 plies. Lure pulls (85.~Sf5-e5 86.Sd5xe5, 87...~Se5-e6 88...Sd7xe6) won the material.
- **25600-v-1600-seed1:** a freeze-out. 64.Gg4-f4 freezes Black's last piece, the oracle on f5,
  up the f-file, and Black has no legal move. 14.~Sf6-f5 15.Sg4xf5 is a lure pull that won a
  spearman.
- **25600-v-1600-seed2:** 76.Od2xc3 takes an oracle that Gf3 has frozen along rank 3 since
  ply 147. Both sides used the lure 7 times.
- **6400-self-seed0 (draw):** the mutual freeze of the f5 and g5 gorgons let Black's oracle take
  the frozen g5 gorgon (21...Of6xg5). White's lure pulls won three spearmen (31-40). Sight
  decided the material, not the result.
- **6400-self-seed1:** 106.Od7xe6 takes an oracle frozen up the e-file by Ge5 for the last 10
  plies. 12...~Sd5-d6 13...Sc7xd6 was a lure pull.
- **6400-self-seed2 (repetition, ply 81): nothing.** No command is played. Both captures
  (16.Sd4xc5, 19.Sc5xd5) are of Black gorgons that stepped next to an unfrozen spearman, and the
  gaze had no part in either. The only standing effect is a Black f6 spearman frozen all game,
  and the game ends in a shuffle.

**(iv) Yes,** with a caveat. I found no rush, fortress or forced line a person would find:
- The f-file rush is answered by a screen and loses a spearman.
- The oracle march goes nowhere.
- A passive gorgon shuffle holds against the bots, even at 102400 nodes, but loses to a simple
  hand plan.
- I found no short forced line.

The caveat: the bots cannot make progress against passivity, or convert two extra gorgons
(6400-seed2), or convert an extra lure and two spearmen without gorgons (6400-seed0). I believe
these are player failures and not fortresses, but I did not prove the conversions. There is
also a second-seat lean of 20-9 at 12800 nodes and up.

**(v) Yes,** with reservations. The stated decision, open or close a line, is visible and
recurs:
- A discovered gaze at move 1.
- Screens made by stepping your own piece into a gaze (3...Se7-f6; 8...Sf7-f6 in 6400-seed2).
- Mutual gorgon standoffs broken by a third piece (6400-seed0, Of6xg5).
- The lure pulling a spearman out of a wall and striking it, in 5 of 6 games.

The stated ends occur: a frozen oracle is struck in 4 of 6, and a side frozen solid loses in
1 of 6 (plus many of my extra games). Against its own failure list:
- **Rush:** none found.
- **Dominant piece:** none. The gorgons are fragile rather than decisive.
- **Dead rule:** none. The lure and the no-move loss both fire.
- **Fortress or shuffle:** partly present. 2 of 6 transcripts are draws by repetition or no
  progress, there are long contact-free stretches, and 3 of 14 of my 25600 self-play games
  drew by no progress. I showed the passive defence is beatable, so this reads as bot weakness,
  but a person playing it will meet long waiting phases.
- **Seat:** a lean exists, toward the second player, not the first.


## Blind read 2 (read-7b)

# Blind reading: gorgons (packet read-7b)

Tools used: single `play` games and Engine probes (`replay.py`, `caps.py`,
`at.py`, `probe.py`, `vs.py` in this folder; `mine/` holds one saved game and a copy of the
spec with the lure's command deleted). No score, test, tournament or lease. All freeze/command
claims below were read from `Engine.effects`, not worked out by hand.

## Attempts

1. **Spearman rush on the oracle.** Hand line Sd3-c4, Sc4-c5, Se3-d4, Sc5-c6, Sd4-d5 against
   minimax:nodes=6400. Black simply took both spearmen (4...Sc7xc6, 5...Sc6xd5) and was two
   spearmen up by move 6. Spearmen meet head-on and the defender takes first. Fails. (It does
   beat `random` in 6 moves; irrelevant.)
2. **Gorgon gaze rush on the oracle.** The black oracle (c9) can only be gazed down the c-file
   or along row 9; Sc7, Lc8 and Gd9 screen both at the start, and only the lure could pull one
   out, which needs the lure two cells from c7. No line opens in the first moves. Root scores
   from the start at 4M nodes / depth 40: every move 0.00; no forced line visible to the core.
3. **Fortress / shuffle.** Three of the six transcripts are draws that end in long shuffles
   with the gorgons locked in mutual-freeze pairs (25600 s0: ~70 moves of lure/oracle
   shuffling, Gf2/Gf5 and Gg4/Gg5 frozen; 25600 s2: repetition, Gd4/Gd5 and Gg4/Gg5 frozen;
   6400 s1: 300-ply limit). I tested each as a fortress by replaying from inside the shuffle
   with minimax:nodes=25600 on both sides:
   - 25600 s0 from ply 144: **Black wins** (royal captured, move 88). Forcing White to break a
     standoff with the lure (Lg3-f3 / Lg3-f4) or to retreat the oracle (Ob8-c7) also lost
     every time; a version with 6400 lost by White being stalemated, everything frozen. The
     "materially ahead" side's oracle was stranded at b8; the draw was the 1600 player missing it.
   - 25600 s2 from ply 80: **White wins** (Black's lone oracle frozen, no move).
   - 6400 s1 from ply 140: **White wins** (stalemate, lone frozen oracle).
   None of the shuffles was a fortress; each was a won position the weaker bot did not convert.
4. **Dead rule (lure command).** Commanded moves appear only 5 times in 6 transcripts (25600
   s1 ~Sc7-c8, ~Sd3-d4; 25600 s2 ~Sf4-g3, ~Sf7-f8; 6400 s1 ~Sb5-a6), none in three games. But
   deleting the command (mine/nolure.json, 10 games 6400 self): White 7, Black 2, draw 1,
   against Black 8, White 2, draw 3 in the 13 real-rule 6400 self games. The threat of the pull
   changes results even when it is not played. Rarely used, not dead.
5. **Dead rule ("no legal move loses").** Fires constantly: 2 of the 3 decisive transcripts
   (25600 s1, 6400 s2) and 11 of my 23 decisive extra self-play games end by stalemate_loss,
   always a side whose remaining pieces are frozen or are spearmen on the last row.
6. **Dominant piece.** Only spearmen and oracles take, by rule, but 29 of the 57 captures in
   the transcripts take a frozen piece (caps.py), i.e. the gorgon made the capture possible.
   Gorgons are not decoration; nor do they dominate: gorgon pairs freeze each other.
7. **Seat.** Self-play outside the packet: 6400 seeds 3-12: Black 7, White 1, draw 2;
   25600 seeds 0-7: Black 5, White 2, draw 1; 1600 seeds 0-9: Black 5, White 3, draw 2. With
   the three 6400 transcripts: 31 self-play games, **Black (second) 18, White 7, draws 6**.
   Not a first-gaze conversion as the reason feared, but a lean to the second player; it
   flips to White when the command is deleted, so it looks like a fragile balance rather than
   a mechanism I could name. Strength converts poorly too: with seats swapped, 1600 (White) beat
   25600 twice in six.

## The five answers

**(i) Yes.** Transcript `6400-self-seed0`, move 15 for Black, after 15.Sf3-e4 (White's e4
spearman now attacks Black's e5 spearman, and White's gorgon stands on f4 freezing Sf5).
- **15...Se5xe4** (played): removes the attacker, but the capturing spearman lands on e4 inside
  White Ge2's file gaze and is frozen at once (a screen, not a fighter); it gives up the gorgon
  on f4, which keeps Sf5 frozen.
- **15...Se5xf4**: takes the gorgon, the bigger piece, and thaws Sf5, but leaves White's e4
  spearman alive and lets White take the initiative on the queen side (Sc4-b5, Gd3-c4 in my
  25600 continuation).
Core root scores at 400k nodes: xe4 +2.0, xf4 +0.4, every other move -2.0 or worse; both
continuations played out by 25600 keep Black level or better.

**(ii) Yes.** Opening: gorgons sidestep (every game 1.Gh1-i1 Gb9-a9), spearmen advance, and the
first opened file freezes an enemy spearman across the board at once (25600 s1 2.Sf3-e4
freezes Sf7). Middle: spearman exchanges on e/f where frozen spearmen are taken (25600 s1/s2
5.Sg5xf6 and 8.Sg5xf6 both take frozen pieces), gorgon standoffs form (25600 s1 move 12,
Gg3/Gg5), lures pull. End: few spearmen, several reached the last row and are dead weight
(Sb9, Sf8, Sc8); oracles walk out, and the question becomes freezing the enemy oracle or
freezing everything: 6400 s0 ends by taking an oracle frozen beside a gorgon for ten moves,
25600 s1 and 6400 s2 by a side frozen solid.

**(iii) Yes.**
- 25600 s0: Black's Oe8 frozen from ~move 40; 36.Sg4xf5, 49.Od3xc4, 56...Of7xf6 take frozen
  pieces; the four gorgons locked in two mutual pairs from move ~58 set the ending.
- 25600 s1: White wins by stalemate: 25.Ge5-d6 freezes the black oracle on d7, Gg5 is held by
  Gg3, Sb6 frozen; Black has no move after 27.Lh2-i1. Two commands decide material
  (20...~Sc7-c8 then Oxc8).
- 25600 s2: 26.~Sf4-g3 pulls a black spearman out for 27.Oh2xg3 (frozen); the draw is the
  double gorgon standoff Gd4/Gd5, Gg4/Gg5.
- 6400 s0: 20...Gd7-e6 freezes White's oracle on e5; it never moves again; 31...Oe6xe5.
- 6400 s1: White oracle frozen on h3 (moves ~49-77) and Black oracle on d6 (~71-108);
  82.Oh4xi4 takes a frozen gorgon; 5 of its 10 captures take frozen pieces.
- 6400 s2: 9 of 13 captures take frozen pieces (incl. 91.Oc4xd4 gorgon, 95.Oc5xb6 lure); White
  wins when Black's last two pieces, Oi3 and Ge7, are both frozen.

**(iv) Yes.** I found no rush (spearman raid refuted, no early gaze line, flat root scores), no
fortress (all three shuffle draws were won positions when 25600 played both sides from inside
them), and no forced line. What I did find is a second-seat lean (18-7-6 over 31 bot games) and
poor conversion by the bots; neither is a plan a person would play.

**(v) Yes, with one named risk.** Lines decide the game: half the captures take frozen pieces,
gorgon pairs freeze each other and fix parts of the board, the oracle that steps into a gaze
is lost (6400 s0), and "frozen solid loses" ends games. The opening-middle-end shape is the
one the reason describes, and the endings stay sharp (breaking a standoff is a real, losing or
winning decision, attempt 3). Against it: the lure's pull is played rarely (5 times in 6 games),
though deleting it changes results; and the seat is not even under these players, the
reason's own failure condition, though on the opposite side from the one it predicted and not
traceable to a mechanism I could name.
