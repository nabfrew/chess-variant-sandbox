# tide: designer session record (`unlike-search.md` §2)

**Not admitted twice; the session adds no game** (first hand-in `17468bf127d7`: i y, ii y, iii y, iv y, v n; revision `ddddfc622703`: i y, ii y, iii n, iv y, v n). Deviation: the first tournament played 128 games, over the brief's 96-game cap (self-pairings forgotten); no score was read.


## First hand-in: design reason

# Design reason (written before any spec)

## The game in one breath

A fight for ground on a small hexagonal board. Every piece paints the cell it lands on in its
side's colour, and the other side can paint it back the same way. Nothing is decided until a
fixed number of turns has been played: then the cells of each colour are counted and the larger
count wins. Each side also holds a few stones in reserve, which may be set down on any empty cell
instead of moving; a stone paints its cell and never moves again, so it holds that cell until an
enemy piece takes it.

## The decision a player keeps facing

Paint now, or be able to paint at the end. A move paints at most one cell, and almost any cell can
be painted back, so the colour of the board in the middle of the game is mostly a promise. What
counts is the colour when the clock stops. A cell is really yours at the count only if you stand
on it, a stone of yours stands on it, or no enemy piece can reach it in the moves that are left.
So every turn asks the same question in three forms:

- take a blank or enemy cell now (+1 or +2 at once), or move a piece to where it reaches more
  cells, or blocks the enemy's reach, for the end;
- take an enemy piece (which paints its cell, and removes a painter and a threat for the rest of
  the game), or keep painting;
- spend a stone now, where it walls a lane and fixes a cell but can be picked off over many turns,
  or keep it for the last turns, where it is a sure +1 or +2 the opponent cannot answer in time.

A retreat onto your own colour paints nothing: defending costs a cell, so there is a price on
every cautious move.

## What a game looks like from opening to end

- **Opening**: both sides spread into the blank middle, painting freely, and try to place their
  long movers on open lanes and their short jumpers near the border between the colours.
- **Middle**: the border is where the fight is. Pieces that stand on the border can be taken
  (taking paints), so a side trades, threatens, and starts using stones to cut lanes. Material
  and reach decide who will have the better end.
- **End**: the last dozen turns are a sprint. Each side repaints as many cells as it can reach,
  takes the pieces that would repaint back, and drops its kept stones on cells the opponent can no
  longer answer. The count decides.

## What would make it fail

- **Only the end matters.** If painting in the middle is wholly reversible and pieces can reach
  anything, the first nine-tenths are a shuffle and the result is decided by who moves last.
  (Seat: the second player makes the final move, the first player the first.)
- **Hoarding is dominant.** If keeping every stone to the end is always right, the stones are a
  fixed bonus, not a decision; if dropping them all at once is always right, likewise.
- **A fortress.** If a side can huddle on its own colour where nothing can take it, and the count
  freezes, the leader just stops playing.
- **Material does not matter.** One move a turn means a bigger army does not paint faster; if
  captures do not change what can be reached and held, the fighting is decoration.
- **The ping-pong.** If the best move is always "repaint the cell they just painted", the game
  never develops a shape.


## First hand-in: break attempts and design log

# tide: design log and break attempts

Frozen: `game.json`, **tide [17468bf127d7]**. Seed: `rules.territory` `repaint` counted by
`rules.scoring` `territory` at a 120-ply clock.

## Where play changed the view in `reason.md`

`reason.md` (written first, not edited) planned a stone reserve as the one permanent claim and
pieces free to move anywhere. Play showed that a free-moving repaint game freezes (the
*ping-pong* failure it named), and stones did not fix it (v3, v4 below). The decision it
states survived: paint now, or be where you can paint at the end. What carries it now is a
different rule: the armies can only flow forward (the leaper alone can turn back), and
captured pieces come back from the hand onto your own two back rows. The "what would make it
fail" list was the checklist the versions were tested against.

## Design log (versions, hash, what changed, why)

| v | hash | change | why / what happened |
|---|------|--------|---------------------|
| 1 | b3ccc1c60cd3 | Hexagon of 61 cells; per side 2 skimmers (6 hex lines, range 3), 3 divers (step or 2-cell diagonal jump), 5 stones in hand (never move); win none, territory repaint, scored at 100 plies, no repetition draw, stalemate loss. | First draft of `reason.md`. One 1600 self-play game: material gone by ply 30 (8 of 10 movers taken), stones dropped next to enemies and taken at once, only 37 of 61 cells ever painted. |
| 2 | 6b1a525436e4 | Skimmer and diver step become move-only (only the diver's jump takes); 120 plies; 4 stones. | Fewer trades. Result: **ping-pong**. From ply 40 to 110 two pieces each side swapped among d4, d5, e4, e5, repainting each other's cells; the count froze at 5-9 for 70 plies. Diagnosis: a repaint swings the count by 2, a blank by 1, so both sides (bots and arithmetic alike) keep repainting the same few cells and the board is never claimed. The rps-repaint failure in the library, at the clock instead of at a threshold. |
| 3 | 3c03c72d26eb | Skimmer captures only stones; 12 stones. | Hoped stones (+2 permanent on an empty enemy cell) would outbid fluid repaints. Bots never dropped a stone (a dropped stone is material in reach of a diver) and the count froze at 7-9 for 60 plies (engine probe at ply 60: every quiet move scored alike; the chosen ones were repaints of the same cells). |
| 4 | 932bbb4b689c | Stones unlimited. | Stones now cost nothing, so they were dropped, but only into a ko-like loop: drop, take, drop again on the cell vacated (`O@c6 Db8xc6 O@b8 Dc6xb8 ...` from ply 57 to 120); count frozen at 7-6. Conclusion: in this seed, any piece that can return to a cell keeps the ping-pong alive. Redesign. |
| 5 | a303a11a1a1f | **Redesign, "the tide".** No stones. Waves (one step to either forward cell or sideways), surges (slide up to 3 along either forward line), one leaper (2-cell diagonal jump, forward only). Captures go to hand (`from_captures`), dropped only on your own two back rows. Board, clock, scoring as v2. | A piece that cannot come back cannot ping-pong; every cell passed is given up; the hand makes captures feed a second wave from home. 1600 self-play: painting grew steadily to 54/61, captures spread through the game. 6400 self-play: 52 and 59 cells painted, 32 and 17 captures. Tournament t1 (below): leaper nearly dead (5% of actions). |
| 6 = frozen | 17468bf127d7 | Leaper jumps the 2-cell diagonal in all six directions: the one piece that can turn back. | Give the leaper a job no other piece has: hunting raiders behind your lines. Prediction: leaper use rises well above 5%, surge share falls a little. t2: leaper 11% of actions, surge 30%, waves 58%. Kept. |

Words only after that (`summary`, `description`; outside the hash, re-checked: 17468bf127d7).

## Tournaments (results by seat, lengths, reasons only)

1. **t1, v5**: `greedy, minimax:nodes=400/1600/6400`, `--games 8`. **This was 128 games, over
   the 96-game cap**: I counted 12 ordered pairings and the tournament also plays each agent
   against itself (16). Read for seat and endings only. Seat 0 58, seat 1 66, draws 4; self-play
   13 vs 17. Endings: 69 scored, 55 stalemate (48 of them in greedy's games: greedy pushes every
   piece forward and runs out of moves), 4 equal counts. Among 1600/6400 games: no stalemate.
2. **t2, v6**: `minimax:nodes=400/1600/6400`, `--games 10` = 90 games (self-pairings counted).
   Seat 0 39, seat 1 49, draws 2; self-play 12 vs 17. Strength strictly ordered (6400 beat 400
   20/20 and 1600 15/20; 1600 beat 400 20/20). 23 stalemates, 20 of them a stronger player
   running a 400 player out of moves, one in 6400 self-play (a 29-16 rout ending at ply 111).
   Final margins: wide spread (-26 to +22), 50-60 of 61 cells painted at the clock; **in none
   of the 90 games did the last move change the result**.
3. **t3, frozen spec**: `minimax:nodes=1600/6400/25600`, `--games 6 --seed 1000` = 54 games.
   Seat 0 25, seat 1 27, draws 2; self-play 7 vs 9. 50 scored, 2 stalemate, 2 equal. Lengths:
   min 99, median 120. 25600 beat 6400 11/12 and 1600 8/12.

Pooled self-play over the three: seat 0 32, seat 1 43, draws 5 (0.57 to seat 1, about 1.3
standard errors from even); the frozen spec alone is 7-9.

## Break attempts

Each by the skill's list: what I tried, what happened, what I changed.

- **The ping-pong** (named in `reason.md`). v2 to v4: broke the game every time (counts frozen
  for 60-70 plies). Changed: the redesign to forward-only armies (v5). After it, no frozen
  stretch in any read transcript; the count rises through the whole game.
- **A rush / flood.** Scripted seat 0 (`fortress.py flood`, choosing among the engine's legal
  moves: capture if possible, else the move landing furthest forward) against
  `minimax:nodes=1600`, seeds 0 and 1: lost both by having no move left, at ply 52 and 48
  (colour 6-23, 5-23). Greedy, which plays much like this, lost 48 games of t1 the same way.
  Holds: pushing everything is punished by the no-move loss. Nothing changed.
- **A fortress or shuffle.** Scripted seat 0 that never lands beyond rank 2, 3 or 4 (captures
  first, else the lowest landing; waves can always step sideways, so it never runs out of
  moves) against 1600, two seeds each: lost every game, 6-35 at the clock. A second version,
  `half`, that paints every cell it can within its own half (ranks up to 4, 5 or 6) and takes
  whatever comes: lost all six, 19-39, 10-46, and four by running out of moves. Sitting back
  cedes the middle and most of your side; the sideways waiting move does not make a fortress.
  Nothing changed. (A strong human fortress was not tried; the bot is the opponent here.)
- **A forced line.** The opening has 13 moves, no capture is possible for the first several
  turns (the armies start five rows apart), and the first captures in t2/t3 games come around
  plies 15-30 in varied places. No short forced sequence found. Not fully searched.
- **A dead rule.** The leaper in v5 (5% of actions in two 6400 games): fixed in v6 (11% over
  ten 6400 games, 24 captures). The no-move loss: fires in strong play only as the end of a
  rout, but the flood probe shows it is what punishes over-pushing, and without it a stuck side
  would simply stop. The hand: 162 drops in ten 6400 games, used throughout. Territory repaint
  is the seed. `repetition_draw: "never"` and `no_progress_draw: 200`: inert by design (the
  clock ends every game); stated so the clock is the only ending.
- **A dominant piece or move.** v6, ten 6400 self-play games: waves 693 actions (112 captures),
  surges 359 (102 captures), leapers 139 (24 captures). The surge is the main striker but not
  the whole game. Not changed.
- **The seat.** Mechanism I expected: the second player makes the last, unanswered repaint.
  Checked directly: the last move decided no result in 90 games. A milder lean to seat 1 (0.57
  pooled) remains, likely the first player having to commit to contact first; the frozen spec's
  own tournament is 25-27. Not changed; I considered a 119-ply clock (first player moves first
  and last) and did not take it, since the last move is not what decides games.

## What I could not close

- **Seat 1 lean** of about 0.57 in pooled self-play (not significant; 7-9 on the frozen spec).
- **Bot blind spots worth knowing when reading transcripts**: the players often end the game
  with pieces still in hand (25600 seed 2 finishes holding six), though a drop on a back cell is
  a free +1 or +2 in the last turns; a person would empty the hand. The players also read the
  count only through a term that grows with the clock, so early painting is under-valued.
- 25600 against 1600 was 8/12 in t3 and 1/3 in the hand-in transcripts (6400 v 1600 was 24/32
  across t2/t3): the top of the ladder is noisy here.

## Final checks (frozen spec)

```
game.json [17468bf127d7]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game sittuyin [771c1a22e182] 0.1401  below 0.5: yes
exit=0

tide [17468bf127d7] is valid
  board       9x9
  pieces      W, U, L
  opening     13 legal moves
  rule mass   62.5

tide [17468bf127d7]: passed
  draw_rate        0.093
  forced_rate      0.0
  games            150.0
  mean_branching   27.06
  mean_plies       120.0
  opening_moves    13.0
  stall_rate       0.093
```

Transcripts (`transcripts/`, all on 17468bf127d7): 6400 self-play seeds 0, 1, 2 won by seat 1
(22-30), seat 0, seat 0; 25600 v 1600 seeds 0, 1, 2 won by 1600 (25-28), 1600 (25-31), 25600
(36-20).

Probe scripts kept beside this log: `replay.py`, `probe.py`, `show.py`, `seats.py`,
`pieces.py`, `margins.py`, `fortress.py` (all hold positions and ask the engine; the scripted
plans only choose among the engine's legal moves). Tournament runs:
`runs/R-S10/sessions/3/t1-v5`, `t2-v6`, `t3-final`.


## Revised hand-in: design reason

# Design reason (revision)

This replaces the first reason, which described a game with a reserve of stones that play
showed could not work (kept as `reason-v1-original.md`, and in `first-handin/`). It is
written for the game handed in now, tide [ddddfc622703], after the redesign and before the
revision's tests and transcripts; `log.md` says why.

## The game in one breath

Ground on a hexagonal board that is painted and painted back, counted when sixty turns each
have been played. Every piece paints the cell it stops on. The armies cannot go home: waves
and surges only move ahead (waves also sideways), and only the leaper can turn back. A captured
piece goes into your hand, and instead of moving you may set it down on any empty cell of your
own half, painting that cell.

## The decision a player keeps facing

**Spend a piece's forward journey now, or keep it in reserve for later.** Every forward move
paints a cell and uses up road that piece will never travel again. So each turn asks:

- push a piece into the opponent's half, where it paints cells they must come back for but can
  be taken by the pieces still waiting there, and where it can never return to defend; or hold
  it back, guarding your own ground and keeping a move in hand;
- take an enemy piece (which paints its cell and gives you a piece in hand), or keep painting;
- put a held piece down now, as a fresh piece in your half, or keep it until the last turns,
  when it re-claims a cell of your half that a raider has painted, and the opponent may not have
  time to answer.

## What a game looks like from opening to end

- **Opening.** No contact for the first several turns. Both sides paint their own half and
  choose lanes for their surges.
- **Middle.** The armies meet across the middle row. Contact is dangerous for whoever steps in
  first, so it is a fight over lanes, trades and raids. Captures fill the hands, and the side
  with more pieces in hand can rebuild in its half.
- **End.** Surges have spent themselves on the far rows, leapers hunt raiders, waves step
  sideways to repaint, and the hands are emptied onto cells of your own half that the
  opponent's raiders have painted.

## What would make it fail

- **Flooding wins**, or **sitting back wins**: if pushing everything forward, or holding
  everything at home, beats a balanced player, there is no decision.
- **The ping-pong**: if pieces can keep repainting the same few cells back and forth, the board
  is never claimed. The forward-only rule is there to prevent it.
- **The hand is decoration**: if held pieces go unspent at the clock, the reserve is not a
  decision.
- **Decided too early**: if the leader after the opening nearly always wins, the middle and end
  are only playing it out.
- **The seat**: the second player makes the last, unanswered repaint.
- **A dead piece**: the leaper, confined to a third of the cells by its diagonal jump, does no
  work.


## Revised hand-in: break attempts and design log

# tide: design log and break attempts

**Revised hand-in, frozen: `game.json`, tide [ddddfc622703]** (see *Revision* at the end).
The first hand-in, tide [17468bf127d7], is preserved in `first-handin/`; the sections below up
to *Revision* are its log, unchanged. Seed: `rules.territory` `repaint` counted by
`rules.scoring` `territory` at a 120-ply clock.

## Where play changed the view in `reason.md`

`reason.md` (written first, not edited) planned a stone reserve as the one permanent claim and
pieces free to move anywhere. Play showed that a free-moving repaint game freezes (the
*ping-pong* failure it named), and stones did not fix it (v3, v4 below). The decision it
states survived: paint now, or be where you can paint at the end. What carries it now is a
different rule: the armies can only flow forward (the leaper alone can turn back), and
captured pieces come back from the hand onto your own two back rows. The "what would make it
fail" list was the checklist the versions were tested against.

## Design log (versions, hash, what changed, why)

| v | hash | change | why / what happened |
|---|------|--------|---------------------|
| 1 | b3ccc1c60cd3 | Hexagon of 61 cells; per side 2 skimmers (6 hex lines, range 3), 3 divers (step or 2-cell diagonal jump), 5 stones in hand (never move); win none, territory repaint, scored at 100 plies, no repetition draw, stalemate loss. | First draft of `reason.md`. One 1600 self-play game: material gone by ply 30 (8 of 10 movers taken), stones dropped next to enemies and taken at once, only 37 of 61 cells ever painted. |
| 2 | 6b1a525436e4 | Skimmer and diver step become move-only (only the diver's jump takes); 120 plies; 4 stones. | Fewer trades. Result: **ping-pong**. From ply 40 to 110 two pieces each side swapped among d4, d5, e4, e5, repainting each other's cells; the count froze at 5-9 for 70 plies. Diagnosis: a repaint swings the count by 2, a blank by 1, so both sides (bots and arithmetic alike) keep repainting the same few cells and the board is never claimed. The rps-repaint failure in the library, at the clock instead of at a threshold. |
| 3 | 3c03c72d26eb | Skimmer captures only stones; 12 stones. | Hoped stones (+2 permanent on an empty enemy cell) would outbid fluid repaints. Bots never dropped a stone (a dropped stone is material in reach of a diver) and the count froze at 7-9 for 60 plies (engine probe at ply 60: every quiet move scored alike; the chosen ones were repaints of the same cells). |
| 4 | 932bbb4b689c | Stones unlimited. | Stones now cost nothing, so they were dropped, but only into a ko-like loop: drop, take, drop again on the cell vacated (`O@c6 Db8xc6 O@b8 Dc6xb8 ...` from ply 57 to 120); count frozen at 7-6. Conclusion: in this seed, any piece that can return to a cell keeps the ping-pong alive. Redesign. |
| 5 | a303a11a1a1f | **Redesign, "the tide".** No stones. Waves (one step to either forward cell or sideways), surges (slide up to 3 along either forward line), one leaper (2-cell diagonal jump, forward only). Captures go to hand (`from_captures`), dropped only on your own two back rows. Board, clock, scoring as v2. | A piece that cannot come back cannot ping-pong; every cell passed is given up; the hand makes captures feed a second wave from home. 1600 self-play: painting grew steadily to 54/61, captures spread through the game. 6400 self-play: 52 and 59 cells painted, 32 and 17 captures. Tournament t1 (below): leaper nearly dead (5% of actions). |
| 6 = frozen | 17468bf127d7 | Leaper jumps the 2-cell diagonal in all six directions: the one piece that can turn back. | Give the leaper a job no other piece has: hunting raiders behind your lines. Prediction: leaper use rises well above 5%, surge share falls a little. t2: leaper 11% of actions, surge 30%, waves 58%. Kept. |

Words only after that (`summary`, `description`; outside the hash, re-checked: 17468bf127d7).

## Tournaments (results by seat, lengths, reasons only)

1. **t1, v5**: `greedy, minimax:nodes=400/1600/6400`, `--games 8`. **This was 128 games, over
   the 96-game cap**: I counted 12 ordered pairings and the tournament also plays each agent
   against itself (16). Read for seat and endings only. Seat 0 58, seat 1 66, draws 4; self-play
   13 vs 17. Endings: 69 scored, 55 stalemate (48 of them in greedy's games: greedy pushes every
   piece forward and runs out of moves), 4 equal counts. Among 1600/6400 games: no stalemate.
2. **t2, v6**: `minimax:nodes=400/1600/6400`, `--games 10` = 90 games (self-pairings counted).
   Seat 0 39, seat 1 49, draws 2; self-play 12 vs 17. Strength strictly ordered (6400 beat 400
   20/20 and 1600 15/20; 1600 beat 400 20/20). 23 stalemates, 20 of them a stronger player
   running a 400 player out of moves, one in 6400 self-play (a 29-16 rout ending at ply 111).
   Final margins: wide spread (-26 to +22), 50-60 of 61 cells painted at the clock; **in none
   of the 90 games did the last move change the result**.
3. **t3, frozen spec**: `minimax:nodes=1600/6400/25600`, `--games 6 --seed 1000` = 54 games.
   Seat 0 25, seat 1 27, draws 2; self-play 7 vs 9. 50 scored, 2 stalemate, 2 equal. Lengths:
   min 99, median 120. 25600 beat 6400 11/12 and 1600 8/12.

Pooled self-play over the three: seat 0 32, seat 1 43, draws 5 (0.57 to seat 1, about 1.3
standard errors from even); the frozen spec alone is 7-9.

## Break attempts

Each by the skill's list: what I tried, what happened, what I changed.

- **The ping-pong** (named in `reason.md`). v2 to v4: broke the game every time (counts frozen
  for 60-70 plies). Changed: the redesign to forward-only armies (v5). After it, no frozen
  stretch in any read transcript; the count rises through the whole game.
- **A rush / flood.** Scripted seat 0 (`fortress.py flood`, choosing among the engine's legal
  moves: capture if possible, else the move landing furthest forward) against
  `minimax:nodes=1600`, seeds 0 and 1: lost both by having no move left, at ply 52 and 48
  (colour 6-23, 5-23). Greedy, which plays much like this, lost 48 games of t1 the same way.
  Holds: pushing everything is punished by the no-move loss. Nothing changed.
- **A fortress or shuffle.** Scripted seat 0 that never lands beyond rank 2, 3 or 4 (captures
  first, else the lowest landing; waves can always step sideways, so it never runs out of
  moves) against 1600, two seeds each: lost every game, 6-35 at the clock. A second version,
  `half`, that paints every cell it can within its own half (ranks up to 4, 5 or 6) and takes
  whatever comes: lost all six, 19-39, 10-46, and four by running out of moves. Sitting back
  cedes the middle and most of your side; the sideways waiting move does not make a fortress.
  Nothing changed. (A strong human fortress was not tried; the bot is the opponent here.)
- **A forced line.** The opening has 13 moves, no capture is possible for the first several
  turns (the armies start five rows apart), and the first captures in t2/t3 games come around
  plies 15-30 in varied places. No short forced sequence found. Not fully searched.
- **A dead rule.** The leaper in v5 (5% of actions in two 6400 games): fixed in v6 (11% over
  ten 6400 games, 24 captures). The no-move loss: fires in strong play only as the end of a
  rout, but the flood probe shows it is what punishes over-pushing, and without it a stuck side
  would simply stop. The hand: 162 drops in ten 6400 games, used throughout. Territory repaint
  is the seed. `repetition_draw: "never"` and `no_progress_draw: 200`: inert by design (the
  clock ends every game); stated so the clock is the only ending.
- **A dominant piece or move.** v6, ten 6400 self-play games: waves 693 actions (112 captures),
  surges 359 (102 captures), leapers 139 (24 captures). The surge is the main striker but not
  the whole game. Not changed.
- **The seat.** Mechanism I expected: the second player makes the last, unanswered repaint.
  Checked directly: the last move decided no result in 90 games. A milder lean to seat 1 (0.57
  pooled) remains, likely the first player having to commit to contact first; the frozen spec's
  own tournament is 25-27. Not changed; I considered a 119-ply clock (first player moves first
  and last) and did not take it, since the last move is not what decides games.

## What I could not close

- **Seat 1 lean** of about 0.57 in pooled self-play (not significant; 7-9 on the frozen spec).
- **Bot blind spots worth knowing when reading transcripts**: the players often end the game
  with pieces still in hand (25600 seed 2 finishes holding six), though a drop on a back cell is
  a free +1 or +2 in the last turns; a person would empty the hand. The players also read the
  count only through a term that grows with the clock, so early painting is under-valued.
- 25600 against 1600 was 8/12 in t3 and 1/3 in the hand-in transcripts (6400 v 1600 was 24/32
  across t2/t3): the top of the ladder is noisy here.

## Final checks (frozen spec)

```
game.json [17468bf127d7]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game sittuyin [771c1a22e182] 0.1401  below 0.5: yes
exit=0

tide [17468bf127d7] is valid
  board       9x9
  pieces      W, U, L
  opening     13 legal moves
  rule mass   62.5

tide [17468bf127d7]: passed
  draw_rate        0.093
  forced_rate      0.0
  games            150.0
  mean_branching   27.06
  mean_plies       120.0
  opening_moves    13.0
  stall_rate       0.093
```

Transcripts (`transcripts/`, all on 17468bf127d7): 6400 self-play seeds 0, 1, 2 won by seat 1
(22-30), seat 0, seat 0; 25600 v 1600 seeds 0, 1, 2 won by 1600 (25-28), 1600 (25-31), 25600
(36-20).

Probe scripts kept beside this log: `replay.py`, `probe.py`, `show.py`, `seats.py`,
`pieces.py`, `margins.py`, `fortress.py` (all hold positions and ask the engine; the scripted
plans only choose among the engine's legal moves). Tournament runs:
`runs/R-S10/sessions/3/t1-v5`, `t2-v6`, `t3-final`.

---

# Revision (after the first blind reading)

The reading of 17468bf127d7 answered (v) no, for three reasons: the reserve stones in
`reason.md` do not exist in the game; held pieces often went unspent at the clock, because they
could only be dropped on the two back rows, which were nearly always already your colour; and
the forward-only flow that shapes the game was not in the reason. It also noted that the
leader at ply 40 won all six transcripts, and a lean to the second seat in its own games.

## A new `reason.md`

**I wrote a new `reason.md`** for the game handed in now. Why: the first one described a game
(free-moving pieces and a stone reserve) that v2 to v4 showed cannot work, so it could not be the
reason for tide. The original is kept as `reason-v1-original.md` (and in `first-handin/`).
The new one was written after the v7 change and its first tournament (t4), but before the
revision's break attempts and the hand-in transcripts. It names forward-only flow as the
mechanism, spending a piece's road now or keeping it as the decision, and the hand as a reserve.
Its failure list includes the reader's two findings (an unspent hand, decided too early) and
the seat.

## Design log, continued

| v | hash | change | why / what happened |
|---|------|--------|---------------------|
| 7 = frozen | ddddfc622703 | Drops allowed on any empty cell of your own half (ranks 1 to 4, 26 cells) instead of only your two back rows (11 cells). Nothing else changed. | The hand was meant to be a reserve, but the back rows were almost always your own colour by the end, so a drop painted nothing. Prediction (written before t4): far fewer pieces held at the clock; the leader after the opening wins less often, since a side behind can drop held pieces to win back its half; the seat about the same. Result: pieces held at the end (both hands) fell from 6.3 (t3, v6) to 2.7 (t4, v7). The ply-40 leader won 15 of 32 decided games in t4, against 62 of 78 in t2 and 29 of 44 in t3. The ply-80 leader won 39 of 42 (70 of 75 in t2). Self-play seats 10-13 (one draw). Kept. |

Words for v7: `summary` line 5 and the `description` now say "any empty cell of your own half"
and describe the hand as a reserve. Checked against the spec (hash unmoved by words).

## Tournament (revision)

4. **t4, v7**: `minimax:nodes=1600, minimax:nodes=6400`, `--games 12 --seed 2000` = 48 games
   (four ordered pairings including self-play, within the cap). Seat 0 22, seat 1 24, draws 2;
   self-play 10 vs 13 (one draw). Endings: 43 scored, 3 stalemate, 2 equal counts. Lengths: min
   84, median 120. 6400 beat 1600 21 of 24 (11-1 as first player, 10-1-1 as second), a clearer
   step than v6's 15/20 and 9/12. Last move changed no result. Actions in the 12 6400 self-play
   games: waves 950 (98 captures, 95 drops), surges 297 (67 captures, 49 drops), leapers 193
   (24 captures).

## Break attempts (revision, on v7)

- **Flood** (scripted seat 0 against 1600, as before), seeds 0 and 1: lost both by having no
  move, at ply 56 (4-23). Holds.
- **Fortress** (never land beyond rank 3), seeds 0 and 1: lost 6-37 and 7-39. **Half** (paint
  everything reachable within rank 4, take what comes), seeds 0 and 1: lost 20-40, 20-36. Holds.
- **Drop-front** (new rule, new risk: a held piece dropped on rank 4 starts near the front, so a
  hand might become a rush). Seat 0 plays `minimax:nodes=1600` but always drops when it can, on
  the furthest-forward cell; against 1600, seeds 0 to 2: lost 25-26 and 19-31, won 31-26. Not a
  winning plan, not suicide either: forward drops are a real option, not a dominant one.
- **Hoard** (dead-rule test of the wider drop): seat 0 plays 1600 but never drops (it takes the
  probe's best non-drop move instead); seeds 0 to 2: lost 24-29, lost 9-44, won 33-16. Losing
  two of three to the same bot shows the drop matters; the third is the variance of the plan.
- **Unspent hand.** 2.7 pieces held at the clock on average in t4 (6.3 before). Where pieces are
  still held at the end, checked one (6400 self-play seed 2, ply 119): the first player had
  several drops worth +1 and one move to make, and the last move of each side was a drop
  (`60. U@i3 60... W@f8`). Pieces left in hand at the end are what one move a turn leaves over,
  not a dead rule.
- **Decided after the opening.** The ply-40 leader now wins about half the games (15/32 in t4;
  3 of 6 in the hand-in transcripts). The ply-80 leader wins most (39/42): the middle decides,
  and the last twenty turns change margins more than results.
- **Seat.** Self-play over t1 to t4: seat 0 42, seat 1 56, draws 6 (0.57 to seat 1); t4 alone
  10-13. Not changed. The last move flipped no result in the 192 games read for it (t2 to t4), so
  an odd clock would treat a mechanism that is not what decides games. Open (below).
- **The leaper's third of the board.** The diagonal jump reaches only one of three cell
  colours, so one leaper covers a third of the board; a dropped one may be placed on any colour.
  It made 193 actions and 24 captures in 12 6400 games. Narrow but working; not changed.

## What I could not close (revision)

- **The seat-1 lean**, about 0.57 in pooled self-play (42-56-6), about 1.4 standard errors from even. The
  reader found 16-9-1. It is not a forced result.
- **The top of the ladder is noisy.** 25600 beat 1600 in all three hand-in transcripts (12-, 13-
  and 25-cell margins, the last by leaving it no move), but I measured 25600 only in t3 (v6).
- **Bot habits.** The players read the count through a term that grows with the clock, so they
  under-value early painting. A person may open more greedily for ground than these
  transcripts do.

## Final checks (revised, frozen spec)

```
game.json [ddddfc622703]
  (a) likeness to chess 0.0000  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game crazyhouse [bb5d97520084] 0.2234  below 0.5: yes
exit=0

tide [ddddfc622703] is valid
  board       9x9
  pieces      W, U, L
  opening     13 legal moves
  rule mass   62.5

tide [ddddfc622703]: passed
  draw_rate        0.053
  forced_rate      0.0
  games            150.0
  mean_branching   28.93
  mean_plies       120.0
  opening_moves    13.0
  stall_rate       0.053
```

Transcripts (`transcripts/`, overwritten, all on ddddfc622703; colour first-second, ply 40 then end):
6400 self-play seed 0: 17-16, then 26-29, second player; seed 1: 18-15, then 17-42, second
player; seed 2: 14-12, then 25-22, first player. 25600 v 1600 seed 0: 15-14, then 31-19, 25600;
seed 1: 16-15, then 27-14, 25600; seed 2: 14-15, then 35-10 at ply 107, 25600 (the 1600 player
had no move left).

Run: `runs/R-S10/sessions/3/t4-v7`. Revision scripts: `words.py` (writes the frozen file's
words), and `fortress.py` gained the `dropfront` and `hoard` plans.


## Blind read 1 (read-3)

# Reading: tide (reader 3)

Tools used: `validate`, `play` (single games), and Engine probes written in this folder
(`replay.py`, `analyse_games.py`, `reclaim.py`, `show.py`, `rootprobe.py`, `scan.py`,
`strategy.py`). The probes replay transcripts, print boards and ask `root_scores` / the core's
own `minimax:nodes=6400` for moves. No score, test, measure, tournament or other forbidden
command was run, and no lease was taken.

## What the transcripts show (replayed)

Colour counts (first player-second player) every 20 plies:

| game | 20 | 40 | 60 | 80 | 100 | 120 | result |
|---|---|---|---|---|---|---|---|
| 25600-v-1600 s0 | 8-8 | 14-15 | 16-22 | 21-24 | 25-26 | 25-28 | 2nd |
| 25600-v-1600 s1 | 9-9 | 15-18 | 21-22 | 21-26 | 23-29 | 25-31 | 2nd |
| 25600-v-1600 s2 | 9-8 | 16-14 | 24-20 | 28-23 | 33-22 | 36-20 | 1st |
| 6400-self s0 | 10-9 | 14-18 | 19-23 | 20-29 | 21-28 | 22-30 | 2nd |
| 6400-self s1 | 8-9 | 16-15 | 21-19 | 26-24 | 29-27 | 30-27 | 1st |
| 6400-self s2 | 7-10 | 15-14 | 23-18 | 28-24 | 28-25 | 30-26 | 1st |

Per game: 39-52 cells flipped from one colour to the other, 11-22 captures, 9-18 drops. At ply
10 every game stands 5-5: no contact in the first five moves each. Of the cells coloured at the
end, 12-22 were last laid in plies 1-40, so mid-game painting survives to the count.

## Break attempts

1. **Rush (everything forward, captures first)** against `minimax:nodes=6400`, both seats, two
   seeds each. It lost all four, each time by having no legal move (plies 45-66): the
   forward-only army spends itself at the far edge and is picked off. Rushing is
   self-defeating.
2. **Fortress / hold back** (only moves that end on your own ranks 1-4, plus drops; otherwise
   the bot's choice): lost all four (17-36, 20-31, 36-17, 35-18 against it). Sitting at home
   hands the opponent the middle and your far side. No fortress.
3. **Shuffle** (a sideways wave step whenever one exists): lost all four, by 3-45 to 54-5, once
   by having no move. No do-nothing defence exists.
4. **Hoarding / dead-rule test on drops** (the bot, never allowed to drop): lost all four
   (17-35, 10-44, 29-15, 33-17). Drops are not decoration.
5. **Dead rule: stalemate loss.** It fires: minimax beats `random` and `greedy` by stalemate
   in 47-64 plies, all four tries, by capturing everything. Between equal bots it fired once in
   26 games (1600 self, seed 9: one wave left shuffling e7-f7 until taken). It is a real
   ending, mostly a punishment for losing material.
6. **Forced line.** I found none. The opening has only 13 legal moves and the bots open much
   alike (e2-e3 / h2-g3, then surges), but nothing in the first moves forces anything: no contact
   before ply 12 in any transcript.
7. **Dominant piece.** Waves do most of the painting (six of nine pieces). Surges are one-way
   and end their lives as dead pieces on the far edge. The leaper's hex-diagonal leap
   confines it to about a third of the cells (a flood fill from g1 reached 14 cells). No one
   piece or move dominates, but the leaper is narrower than its billing as the raider-hunter.
8. **Seat.** Over 26 equal-strength games (10 at 1600 self, 16 at 6400 self including the
   transcripts and my own) the second player won 16, the first 9, with 1 draw. The second player
   makes the last move, which is an unanswered repaint. That is a lean of about 60%, not a forced
   win: the first player won 3 of the 6 transcripts.
9. **Skill gradient (a side observation).** `minimax:nodes=25600` against `1600` over 15 games
   (the 3 transcripts plus 12 of mine, both seats): 8 wins, 5 losses, 2 draws. Sixteen times the
   search barely beats the weaker player. The bots may be blind here, or the game may be noisy at
   this depth. I can't tell which, so this counts against confidence, not as a break.
10. **Hand at the clock.** Pieces held at the end are dead weight. At ply 119 in 25600-v-1600 s1
   the second player held seven pieces and could drop them on many back-row cells, but every
   one of those cells was already its own colour, so a drop paints nothing. 6400-self s2 shows
   the same, with five held. A drop gains only where an enemy raider has repainted your back rows.

## The five questions

**(i) Yes.** 6400-self-seed1, move 35... (ply 69; second player to move, behind 22-25).
- **35...Ld7xf6** takes the first player's surge on f6, a deep raider aimed at c9/d8 and
  f7-f9, and repaints f6. It gives up the leaper: the wave on g5 recaptures on f6. That trades
  leaper for surge, each side gets a piece in hand, and f6 is repainted back.
- **35...Ld7-e5** (played) paints an untouched cell, e5, and keeps the leaper. It gives up
  stopping the surge, which went 36.Uf6-c9 and painted c9 before 38...Ld7xc9 finally took it.
- The probe's root scores (20000 nodes) were 1.40 and 1.36, the two best moves.
- A second instance: 6400-self-seed0, move 28... (ply 55). Ud8-d5 (played) advances a surge to
  paint the middle and spends its forward range. W@c8 instead drops a piece at home now and
  keeps the surge back. The probe scored the two equal.

**(ii) Yes.**
- **Opening** (to about ply 12-20): no contact; both sides paint their own halves, 5-5 at ply
  10 in all six. First captures come at plies 12-22.
- **Middle**: material and lanes. There are 11-22 captures a game and the hands fill. Material
  swings decide reach: in 25600-v-1600 s1 the second player fell to 2 pieces on the board at
  ply 100 and lived off drops. In 6400-self s0 the second player's lead grew from 14-18 to 20-29
  in this phase.
- **End**: surges are spent on the far edge, leapers hunt, waves step sideways repainting, and
  tempo matters. In 6400-self s1 the margin went -1 to +3 in the last ten plies, and in
  6400-self s2 it went 0 to +4.
- The caveat: the leader at ply 40 won all six. The end changes margins far more often than it
  changes results.

**(iii) Yes.** Every game ended by the count at ply 120. Each time, the winner's total
includes more re-claimed cells (cells that were the other colour at some point) than the
winning margin:
- 25600-v-1600 s0: 25-28; the second player holds 11 re-claimed cells against a margin of 3.
  The 1600 player out-repainted the 25600 player in the middle (16-22 at ply 60).
- 25600-v-1600 s1: 25-31; 13 re-claimed against a margin of 6. The weaker side won the paint
  war while losing material (2 pieces on the board at ply 100) by dropping.
- 25600-v-1600 s2: 36-20; the first player holds 21 re-claimed against a margin of 16, a
  steady repaint of the second player's half after it won material.
- 6400-self s0: 22-30; 12 re-claimed against a margin of 8. The second player's raid by its
  surge to i2/i1 and the repaint of the first player's back rows.
- 6400-self s1: 30-27; 11 re-claimed against a margin of 3. The lead turned from -1 to +3 in
  the last ten plies.
- 6400-self s2: 30-26; 12 re-claimed against a margin of 4. Level 28-28 at ply 110, decided by
  the last repaints.

**(iv) Yes (I found none).**
- A rush stalemates itself (0/4).
- A home fortress loses (0/4), a shuffle loses (0/4) and hoarding the hand loses (0/4).
- I found no forced opening line.
- Annihilation wins exist only against weak play.
- The second seat leans about 60% (16-9-1), which is not a forced result.
- The caveat: the weak skill gradient (25600 vs 1600 at 8-5-2) means a stronger player could
  still find something these bots and I did not.

**(v) No.**
- The design reason's core question is present: paint now, or stand where you can paint at the
  end. Several of its failure modes are avoided: the middle counts, material matters, there is
  no fortress, and ping-pong repaints were 7-15 of 120 plies.
- But the game is not the one the reason describes. The reason's third decision rests on
  reserve stones held from the start, droppable on any empty cell and never moving again. None
  of that exists:
  - The hand comes only from captures.
  - Drops are limited to your own two back rows.
  - A dropped piece moves like any other.
- The reason's end, "drops its kept stones on cells the opponent can no longer answer", does
  not happen. Held pieces end the game unspent (seven in 25600-v-1600 s1, five in 6400-self s2)
  because the back rows they may enter are already their colour.
- The shape that carries the game, forward-only armies that spend themselves, appears nowhere
  in the reason. It is a different design that happens to share the reason's first question.


## Blind read 2 (read-3b)

# Blind reading: tide (packet read-3b)

Tools used: `play` (single games), and Engine probes written in this folder (`probe.py`
replays a transcript and counts paint; `dump.py` prints the board and its colours at a ply;
`roots.py` scores every legal move at a ply with `root_scores`; `scan.py` looks for
close-scored moves of different kinds; `flips.py` counts repaints, captures and drops per
20-ply phase). No score, test, measure, analyse, tournament or other barred command was run.
No lease taken. The board has 61 cells; final painted totals run 45 to 59.

## Attempts

1. **Seat.** 24 equal-strength games with `play`, seeds 0-11, at `minimax:nodes=1600` and
   `minimax:nodes=6400` self-play. Second seat won 7/12 at each budget (14/24 overall), first
   seat 10/24, no draws. In the transcripts' three self-play games the second seat won 2 of 3.
   A mild second-seat lean, consistent with the reason's "last unanswered repaint" worry, but
   nowhere near decisive. Not broken.
2. **Rush / annihilation.** The capture rule (a taken piece joins the taker's hand) swings
   material both ways, so I looked for a wipe-out plan. Between equals it is rare: 2 of 12
   games at 6400 ended with a side unable to move (plies 74 and 80; in seed 3 the loser had
   spent every piece forward and been eaten while the winner held 5 waves and a leaper in
   hand). Against a weaker player it is the rule: greedy vs `minimax:nodes=6400` (both seats,
   seeds 0-2) and random vs `minimax:nodes=400` (seeds 0-2) all nine ended by the weaker side
   having no move, at plies 38-72, before the clock. So the ground count only arbitrates
   between near-equals; a real skill gap is settled by eating the opponent. Not a rush a person
   finds in the opening (no game I saw lost material by force in its first 10 plies), but it
   means the "no legal move" rule, not the ground, is how lopsided games end.
3. **Opening forced line.** Surges are corked behind their own waves at the start; the first
   contact is an enemy surge sliding three cells onto an advanced wave (e.g. `...Ub9-b6`,
   `...Ub6xe3` in two 25600 games). In both it was recaptured at once (`Wf2xe3`), a surge given
   for a wave. The 6400 self-play games had 0, 0 and 3 captures in their first 20 plies. I
   found no forced material win in the opening.
4. **Fortress / shuffle.** Waves step sideways, so a wave with an empty neighbour has an
   unlimited supply of moves: the forward-only rule does not stop a back-and-forth. It happens:
   6400-self seed1, moves 45-55, first player `Wf7-g7`/`Wg7-f7` eleven times in a row, the second
   player `Wh1-g1-f1-e1-f1`; 25600 seed2, moves 37-40, `Wc4-b4`/`Wb4-c4`. These shuffle over
   cells already one's own, so they are passes, not contested repaints, and they did not save
   the shuffler: in seed1 the shuffling side fell from 25-23 to 17-42. No fortress, but the
   ping-pong of pieces the reason says forward-only prevents is present as a tempo device.
5. **Dead rule / dominant piece.** Every piece type works in every transcript (wave 58-86
   moves, surge 16-26, leaper 9-20 per game; drops 6-23 per game). The leaper is not dead: it
   hunts raiders and is traded (`...Le8xf6 We6xf6`, `Lc6xa7`, `Ld4xc6`). The no-progress draw
   (200 turns) can never fire inside a 120-ply game; harmless decoration. The hand is used, but
   **pieces are left in hand at the clock in all five transcripts that reached it** (25600 s0:
   first player 1 wave; s1: first player 4 waves + 1 surge; 6400 s0: first player 1 surge; s1:
   second player wave, surge, leaper; s2: first player 4 waves + 1 leaper), mostly by the
   winner, with empty cells of their own half still available. Only one action per turn, so a
   surplus in hand cannot be spent; the hand's value saturates at about one piece.
6. **Decided too early.** Between equals, no: in the three 6400 self-play games the side
   ahead at ply 60 lost every time (25-22 → lost 26-29; 25-23 → lost 17-42; 14-19 → won
   25-22), and the eventual winner took a lasting lead only at ply 118, 68 and 115. In the
   25600-vs-1600 games the stronger side led from ply 33-51 on: that is the skill gap.

## The five answers

**(i) Yes.** 25600-v-1600 seed1, move 27, second player to move (position after
`27. Lg4-f6`; paint 19-14 to the first player; second player holds one wave). Two moves score
equal at a 400 000-node `root_scores` probe (-13.71 each), and they differ in kind:
- `27... Le8xf6` takes the raiding leaper that has jumped into the second player's half and
  repaints f6. It gives up the second player's own leaper, its only piece that can turn back,
  because `We6xf6` recaptures (played: `28. We6xf6`), and the first player gets a leaper into
  hand.
- `27... W@a8` drops the held wave onto a8, a cell of its own half the first player has
  painted, winning it back and keeping the leaper. It gives up the hand's only piece (the late
  re-claim the reason describes) and leaves the enemy leaper free inside its half to keep
  repainting.
This is the reason's "take a piece, or keep painting / spend the hand now or later" choice,
visible in a real position.

**(ii) Yes.** Phases differ measurably (`flips.py`). Opening: in the 6400 games repaints per
20 plies are 1, 0, 3 and captures 0, 0, 3; both sides paint their own half and route surges.
Middle (plies 20-60): captures peak (4-6 per 20 plies), the hands fill and the first drops
come. End (plies 60-120): repaints climb to 10-17 per 20 plies, surges are parked on the far
rows (e.g. a8, a9, b9 surges in 25600 s0/s1), leapers hunt, and drops land on own-half cells
(8 drops in the last 20 plies of 6400 s2). Two of three equal games changed leader in the last
six plies. What matters shifts from lanes, to material and hand, to which cells are the last
painted.

**(iii) No.** The ground count decides the result in five of the six, not all:
- 25600 s0: the count (31-19) names the winner; the stronger side's lead was steady from ply 37.
- 25600 s1: the count (27-14) names the winner; steady lead from ply 33.
- 25600 s2: **not the ground.** The game ended at ply 107 because the second player had no
  legal move (all its pieces captured, hand empty). The paint (35-10) was never counted.
- 6400 s0: re-claiming decides it: the first player led 26-22 at ply 80; the second player
  re-took cells late and won 29-26, leading for good only at ply 118.
- 6400 s1: a middle-game re-claim (25-23 at ply 60 to 18-32 at ply 80) decides it, 42-17.
- 6400 s2: the end decides it: second player led 19-14 at ply 60; the first player re-took
  ground and won 25-22, leading for good only at ply 115.
So the mechanism is real and decisive between equals, but one transcript was settled by
annihilation, and in the two 25600 games it scored was more a ledger of a material gap than a
fight over ground.

**(iv) Yes.** I found no rush, fortress or forced line a person would find (attempts 1-4):
seat lean is mild (14/24 second seat), no forced opening material win, the sideways shuffle
is a pass that did not save the side using it, and wipe-outs need a skill gap. Caveat kept
honest: wipe-out by capture is how every game between unequal players I ran ended.

**(v) No.** Much of the reason holds: the decision it names is visible (answer i), the
opening/middle/end shape is as described (ii), the leaper works, the seat lean is small and
the game is not decided early between equals. But on its own stated failure tests the game
does not fully pass: (a) "the hand is decoration: if held pieces go unspent at the clock" —
pieces sat unspent at the clock in all five games that reached it, often four or five of
them; (b) "the ping-pong ... the forward-only rule is there to prevent it" — sideways waves
shuffle back and forth for ten moves at a time in 6400 s1 and 25600 s2; (c) the description's
"push everything and you run out of moves" is true, but the larger fact is that any real
skill gap ends by eating the opponent before the clock (9 of 9 weak-vs-strong games, and 25600
s2), so the ground the game is built around arbitrates only between near-equals.
