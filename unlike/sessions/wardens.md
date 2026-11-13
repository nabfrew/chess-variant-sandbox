# wardens: designer session record (`unlike-search.md` §2)

**Admitted at the first read** (`0ee8d3ecb57b`: i y, ii y, iii y, iv y, v y).


## Hand-in: design reason

# Design reason (written before any spec)

Seed: a turn of two parts.

**The game.** Each side has two kinds of piece and every turn moves exactly one of each,
in either order: one *warden* and one *raider*. Raiders are the only pieces that take, and
they take only at a distance: a raider steps one square to move, but captures by jumping
exactly two squares (straight or diagonal). Wardens never take; a warden stops (freezes)
every enemy raider standing next to it, so a frozen raider can neither move nor capture
while the warden stays. Raiders can take wardens. There is no king: a player who cannot
make a whole turn (both parts) loses, so losing both wardens, losing every raider, or
having every raider frozen all lose.

**The decision a player keeps facing.** Which pair of moves. Both parts are compulsory, so a
warden that is perfectly placed must still move unless another warden can move instead, and
a raider standing in a good spot may have to give it up when no other raider move is safe.
Every turn asks: where does my warden go (to pin an enemy raider in place so it can be taken
next turn, or to shield my own raiders from being hit, since a frozen raider cannot capture)
and which raider do I commit (to step into a striking distance of two, where whoever moves
first takes, or to stand back). Order matters: warden first can freeze the raider that would
otherwise answer my capture.

**Shape, opening to end.** Opening: the armies start out of reach (more than two squares
apart); raiders creep forward and wardens are walked up in front as escorts, since stepping a
raider to distance two of an enemy raider unescorted just offers it. Middle game: contact.
Wardens lean on enemy raiders, raiders hunt wardens from two squares away, and exchanges
are counted by who has the spare tempo. End: material thins, the two-part rule bites as
zugzwang (with one warden left it must move every turn, and walking it next to enemy
raiders is the only way to stay useful but also puts it in the line of fire), and games end
by a warden hunt, by running out of raiders, or by freezing the last free raider.

**What would make it fail.** (1) Shuffling: both sides can oscillate a warden and a raider
behind a wall of wardens and never engage, so the stronger player cannot force anything and
games draw by repetition. (2) A dominant rush: if the first player can walk a warden next to
the enemy raider line before it can respond, freeze-then-take could snowball (seat lean).
(3) A dead rule: if wardens never matter (freeze rarely binds), the game is just distance-two
capture; or if raiders never threaten wardens, the warden loss condition never fires.
(4) The second part is decoration: if one kind of piece always has an obvious harmless move,
the forced double move adds nothing over an ordinary turn.


## Hand-in: break attempts and design log

# wardens: design log and break attempts (session 8)

Seed: a turn of two parts (`rules.turn.parts`). Frozen spec: `game.json`, hash `0ee8d3ecb57b`.
Working files (drafts, tournament records, probe scripts) are in
`runs/R-S10/sessions/8/` of the worktree: `v1.json` to `v6.json`, `t1-v3/` to `t4-v5-strong/`,
`summ.py` (results by seat and lengths only), `replay.py`, `board.py`, `stats.py` (replay a
record and count what the rules do), `rush.py` (a scripted line against the engine's player),
`mk.py` (write a variant). No score, index, component or rating was read; tournament output
was read only as results by seat, lengths, endings and transcripts.

`reason.md` was written before any spec and is unchanged. Where play changed my view, it says so below.

## Design log

| version | hash | what changed | why | what happened |
|---|---|---|---|---|
| v1 | dbe47bfc7030 | 7x7; 2 wardens (king step, no capture, freeze adjacent enemy raiders) + 5 raiders (king step to move, leap 2 in eight directions to move or capture); turn = one warden + one raider, any order; lose when you cannot finish a turn | the idea as written in reason.md | limits pass (nearest drover 0.17), smoke passes but draw 23%. A 6400 self game: wardens retreated to corners and raiders shuffled; draw by the no-progress clock at 150 plies. **Failure mode (1), shuffling, confirmed.** |
| v2 | 1383670cbf23 | + a warden reaching the far row wins | give the wardens a reason to advance | two 6400 games both won by player 0 in 25 and 33 plies by walking a warden down an edge file. **A rush and a seat race (failure mode 2).** Discarded. |
| v3 | b6207c364bec | raiders go forward only (one step straight or diagonally forward, move only); they capture only, by leaping two squares in any of eight directions; 7 raiders on rank 2, wardens b1 f1; a raider on the far row wins (warden goal dropped) | turn the compulsory raider part into a clock: raiders cannot shuffle, so the board must tighten and zugzwang appears, which is what makes the two-part turn bite | tournament 1 (90 games, minimax 400/1600/6400): seat 45-45, no draws, endings half by reaching, half by a player unable to finish a turn; stronger players won clearly (6400 v 400: 20-0). But short: median 38 plies (about 9 turns each), shortest 15. |
| v4a | fe2f298c102c | 9x9, 9 raiders, 2 wardens | longer game, more shape | four 1600 self games: player 1 won all four. v4b (3 wardens, 7828b019125f) likewise 4-0 for player 1. Suspected a seat effect from the gap between the armies. |
| v4c | 980843ea3dcf | 9x8, 9 raiders, 2 wardens | one fewer empty rank between the armies | tournament 2 (96 games, 1600/6400): seat p0 42 p1 54, 6400 self 8-16; median 63 plies. Replays: breakthroughs on a flank where neither of the two wardens stood. Lean toward the second player, and two wardens are too thin a defence on 9 files. |
| v5 | 5ecf8523a015 | 9x8, 9 raiders, 3 wardens (b1 e1 h1) | a warden per third of the board; more defence makes contact a fight rather than a race | tournament 3 (96 games, 1600/6400): seat p0 51 p1 45; 1600 self 12-12, 6400 self 14-10; 6400 beat 1600 31-17; no draws; median 66 plies (31 to 106). Tournament 4 (48 games, 6400/25600): seat 23-25; 25600 beat 6400 18-6; 25600 self 4-8, 6400 self 5-7; median 62 plies. |
| v6 | 0ee8d3ecb57b | `no_progress_draw: 1000` | the description says the game cannot be drawn; under the default clock (100 capture-free plies) a draw was possible in principle, because a raider's backward capture makes its move graph cyclic, so a raider step does not reset the clock. The game is bounded anyway (below), so the clock can only ever draw a game that was still in progress. | the longest capture-free stretch in the 144 v5 tournament games was 47 plies, so no recorded game would have played differently. **Frozen.** |

Why v6 cannot be drawn: every turn contains a raider move, and a raider move is a forward step
or a capture, so no position repeats. Each side has at most 9 x 5 = 45 forward steps before a
raider reaches the far row and wins, and at most 12 captures (each worth at most two more steps
if it goes backward), so a side has at most 81 turns: at most 324 plies in all, under the
400-ply limit, and under the 1000-ply clock.

## Break attempts (on v5, the mechanics of the frozen game)

1. **A rush.** Scripted lines for player 0 against `minimax:nodes=6400` (`rush.py`):
   - edge rush: raider a2-a3-a4-a5 with the b1 warden following diagonally behind. Player 1
     captured it at once (a7xa5) the first time it came within two squares; the escort cannot
     help, because a warden freezes only raiders *next to* it, and the capture comes from two
     squares away.
   - centre rush: raider e2 to e5 with the e1 warden directly behind. Same result (e7xe5).
   - In the tournaments the shortest v5 games (31 plies) were breakthroughs by the stronger or
     luckier side after a defensive mistake, not a repeatable plan: the same agents at the same
     strength split their self games. No rush found. (In v2 a rush did exist; see the log.)
2. **A fortress or a shuffle.** Impossible by construction from v3 on: raiders cannot move
   backward except by capturing and a raider must move every turn, so the side behind cannot
   sit still; holding back only spends its own raider moves and leads to a position where it
   cannot finish a turn. 0 draws in 144 v5 tournament games and 150 random smoke games. (v1 was
   exactly this shuffle; the forward-only raider is the change that killed it.)
3. **A forced line.** No opening capture is possible (the armies start four empty ranks apart
   and captures reach two). Opening turns vary in the records: 31 first-move options, and the
   six hand-in games open differently. I found no forced win; the closest thing is the parity
   of the gap, which made 9x9 (five empty ranks) favour the second player in my first games, and
   is why the board is 9x8.
4. **A dead rule.** (a) The freeze: in tournament 3, 559 of 3114 warden moves (18%) froze a new
   enemy raider, and 33 of the 41 games lost by "cannot finish a turn" ended with every one of
   the loser's raiders frozen (4 more with some frozen and the rest blocked; 3 with no raider
   left; 1 with no warden left). With the freeze deleted (x-nofreeze, six 1600 self games), 5
   of 6 games ended by reaching and player 0 won 5 of 6: the freeze is what makes defence
   possible. (b) The compulsory warden part: with it made optional (x-optional, six 1600 self
   games), the players declined to move a warden on 17 turns of about 200 (roughly 8%), mostly
   late. So the compulsion binds, as a zugzwang, but is not on every turn; I kept it compulsory
   because it is the seed and it is what makes losing every warden a loss. (c) The order:
   about a third of turns (1061 of 3169) started with the warden, two thirds with the raider,
   so "either order" is used both ways. (d) Warden capture: raiders took 131 wardens in 96
   games; "no warden left" ended only one game, so losing every warden works as a threat that
   shapes play more than as an ending.
5. **A dominant piece or move.** Both kinds of piece move every turn by rule, so the question
   is whether one kind's choice is trivial: warden moves often freeze (18%), and raiders split
   2309 steps against 860 captures (729 of raiders, 131 of wardens). Forward captures dominate
   captures (719 forward, 128 sideways, 13 backward); that is the natural direction of
   contact, and the sideways and backward ones exist mainly to take wardens and stragglers.
6. **The seat.** The worry in reason.md was that the first player strikes first. On 9x8 with
   three wardens: 51-45 in tournament 3 and 23-25 in tournament 4 (74-70 overall); self-play
   by seat 12-12 (1600), 19-17 (6400), 4-8 (25600, only 12 games). On 9x9 the second player
   won 8 of 8 early games and on 9x8 with two wardens 54-42, so the seat is sensitive to the
   gap and the defence, and v5/v6 is the balanced setting I found. **Not fully closed:** the
   25600 self sample is small and leans 4-8 to the second player; I could not afford a larger
   strong sample within the session's compute budget.

What changed my view against reason.md: the "shuffle" failure (1) was real in the first design
and needed a structural answer (forward-only raiders), which also changed the shape I had
described: the opening is a slow forced advance rather than a free manoeuvre, and the end is
more often a freeze-lock or a breakthrough than a warden hunt (the warden-loss ending is rare).

## Tournaments (four, all under a 4-slot heavy lease, released after each)

| # | spec | agents | games | game-minutes |
|---|---|---|---|---|
| t1 | v3 | minimax 400, 1600, 6400 (9 pairings x 10) | 90 | 1.1 |
| t2 | v4c | minimax 1600, 6400 (4 pairings x 24) | 96 | 5.5 |
| t3 | v5 | minimax 1600, 6400 (4 pairings x 24) | 96 | 7.0 |
| t4 | v5 | minimax 6400, 25600 (4 pairings x 12) | 48 | 12.6 |

About 26 game-minutes in all, well inside the session's slot-hour.

## Final checks on game.json (v6, 0ee8d3ecb57b)

`uv run python unlike/limits.py game.json`:
```
game.json [0ee8d3ecb57b]
  (a) likeness to chess 0.0001  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game runners [a628a77ab8e3] 0.3192  below 0.5: yes
exit 0
```
`uv run chess-variant-sandbox validate game.json`:
```
wardens [0ee8d3ecb57b] is valid
  board       9x8
  pieces      W, R
  opening     31 legal moves
  rule mass   49.5
```
`uv run chess-variant-sandbox smoke game.json`:
```
wardens [0ee8d3ecb57b]: passed
  draw_rate        0.0
  forced_rate      0.0
  games            150.0
  mean_branching   21.58
  mean_plies       81.96
  opening_moves    31.0
  stall_rate       0.0
```

## Transcripts (frozen spec, `transcripts/`)

- 6400 self, seed 0: player 0 wins in 58 plies, every player-1 raider frozen by two wardens.
- 6400 self, seed 1: player 0 reaches the far row in 49 plies.
- 6400 self, seed 2: player 0 reaches the far row in 69 plies.
- 25600 v 1600, seeds 0 to 2: 25600 wins all three by reaching (45, 69, 85 plies).


## Blind read 1 (read-8)

# Reading 8: wardens

Read from `rules.txt`, `reason.md` and the six transcripts only. Tools: `play` (single games), `validate`,
and Engine probes in this folder (`probe.py`, `step.py`, `plan.py`, `mirror.py`, `alt.py`, `tx.py`,
`tx2.py`, `pos.py`). Playouts below are `minimax:nodes=6400` for both sides unless named, over 4 seeds
("W/L" is the mover's result). Note on notation: in the transcripts `N.` and `N...` are the two
parts of one turn by one player (turn 1 White, turn 2 Black, ...).

## Attempts

1. **Seat.** The six transcripts are all first-seat wins, but they are confounded (the three
   25600-v-1600 games put the stronger player first). Extra single games: `minimax:nodes=6400`
   self-play seeds 3-6 went P1, P1, P0, P1; `minimax:nodes=1600` self-play seeds 0-9 went 4 P0, 6 P1;
   1600 (first) v 25600 (second) seeds 0-2: P1, P1, P0, so the weaker player won once from first
   seat. No seat lean I can see. Strength matters: 25600 beat 1600 in 5 of 6 games across both seats.
2. **Fortress / hold back.** Scripted White pushes each raider one rank only and shuffles a
   warden at home. It lost in 19 plies: a black raider on h5 (three ranks from White's line)
   takes any white raider stepping to rank 3 within two squares, and with no warden walked up
   to freeze it, it took h3 and then the h1 warden onto the goal row. Raiders must move every
   turn and only forward, so there is no standing still; repetition is impossible.
3. **Shuffle without the seed (dead-rule test of the two-part turn).** A copy with the turn
   rule removed (move any one piece, `v_onepart.json`): 1600 self-play seeds 0-5 drew 5 of 6 by the
   400-ply limit (players shuffle wardens and never commit raiders). The two-part turn is what
   makes the game end.
4. **Dead-rule test of the freeze.** A copy without the warden's freeze (`v_nofreeze.json`): 1600
   self-play still decisive, 41-79 plies, 5 of 6 by reaching. So the freeze is not what makes the
   game end, but in the real transcripts it binds hard: up to 12 of the mover's moves removed by
   freezes in a turn (25600 seed0 T20, Black left with one raider move), and it decides two games
   directly (attempt 9). Not dead.
5. **Mirror strategy (Black copies White across the middle).** v 6400, seeds 0-2: lost, lost, won
   (the win came mostly from fallback moves). Mirroring hands White first strike at every
   distance-two standoff. Not a holding strategy.
6. **Warden storm (rush).** White walks the centre warden to e6 as fast as it can (a warden on rank 6
   freezes three rank-7 raiders and no rank-7 raider can hit it). v 6400: Black won 2 of 2. With
   25600 finishing both sides: White 2 of 3. The warden got in and froze d7-f7, but Black used the
   tempo elsewhere; it did not snowball.
7. **Edge runner (rush).** a-pawn straight up with a warden escort. v 6400: lost 2 of 2; with 25600
   finishing: 1 of 2. Rank 5 is covered by two rank-7 raiders on the edge, and the escort warden must
   cross that rank itself.
8. **Forced line.** Shortest decisive game I saw was 37 plies (storm v 25600), shortest transcript
   45 plies; no early forced tactic turned up. Root scores at a few early-middle positions sit at 0
   for every move (no line the search sees as winning).
9. **Dominant move/piece.** Raiders move every turn by rule; wardens move every turn, and the warden
   part is sometimes indifferent (25600 seed1 T14: all warden parts score the same) and sometimes a
   back-and-forth (retractions 0-7 per game, of 22-42 turns). Order of the two parts mattered in
   3-6 turns per game, and every one of them is the trivial "raider vacates a square, warden steps
   into it" (tx2.py); I found no turn where order changed anything else.

## The five questions

**(i) Yes.** 6400-self-seed1, turn 15 (White, parts 15./15...; game played `We2-f1` + `Rc3xc5`).
Black has wardens on c5 and h4 (h4 freezes White's g3), and White can take either one:
- `Rc3xc5` takes the queenside warden; it gives up the c3 raider to `Rc7xc5` and leaves the h4
  warden freezing g3. Playouts: 4 of 4 wins.
- `Rh2xh4` takes the kingside warden and unfreezes g3; it gives up the h2 raider to `Rh6xh4` or
  `Rf6xh4` and leaves the c5 warden in White's half. Playouts: 3 of 4 wins.
And the warden part is not free: `Wg2-f1` + `Rh2xh4` (g2 leaves the h-side) lost 4 of 4.

**(ii) Yes.** Opening (turns 1-8 in every transcript): no captures, raiders creep to ranks 3-4 (5-6
for Black), wardens walked up behind; 21-26 raider moves available. Middle: contact and exchanges by
distance-two jumps, wardens freezing raiders and being taken (6400 seed1 T15-T19 two black wardens
lost; 25600 seed2 T10-T26). End: raider options collapse to 1-6 (25600 seed0 Black 1 at T20; 25600
seed2 White 2 at T35/T37; 6400 seed0 Black 2 at T24/T28), and what matters is zugzwang and a single
runner: 6400 seed2 T32 is a full zugzwang (every one of Black's five raider moves lost in all
playouts), 6400 seed0 ends with all of Black's raiders frozen.

**(iii) Yes,** with one of the six resting on my probe rather than the game alone.
- 6400 seed0: Black loses by being unable to make the raider part: its last raiders (e7, e5, i5)
  are all frozen by wardens d6 and h5 at T29, though its wardens could still move.
- 6400 seed1: the warden part chosen beside the capture at T15 decides the result in playouts
  (`We2-f1`+`Rh2xh4` 3/4, `Wg2-f1`+`Rh2xh4` 0/4); after losing two wardens Black's last warden must
  move every turn (5 warden options) while White's runner goes h6-i7-h8.
- 6400 seed2: T32 zugzwang: Black must move b7 or c7; moving c7 abandons the guard of a7, moving b7
  steps into capture, and all four tried alternatives lose; White goes a6-a7-a8.
- 25600 seed0: Black is down to one legal raider move at T20 (`Ri7-h6`) and two at T22; its g7, the
  only raider that could take e7, is frozen by the f6 warden, and Rc5xe7-e8 wins.
- 25600 seed1: T33 White plays `We4-f5` + `Re6-e7` in one turn: the warden freezes e5 and g5, the
  only two black raiders that could take e7, and the raider walks in; f8 next turn.
- 25600 seed2: White is squeezed to two raider moves at T35 and T37 (forced `Ri2-i3`, `Ri3-h4`),
  and the forced h4 raider becomes the `Rh4xf6` capture that opens g7-g8. Black's c3/c4/f4 raiders
  sit frozen. (This one shows the compulsion shaping the end more than deciding it.)

**(iv) Yes.** I found no rush, fortress or forced line: hold-back loses, mirror loses, storm and
edge-runner each win about half against a finishing bot, seats split 4-6 and 4-3, the shortest
game was 37 plies, and the game cannot shuffle (raiders must advance) and has no draws. Without the
two-part turn it would shuffle (5 of 6 drawn), so the seed is doing that work.

**(v) Yes, mostly.** The decision it names (which warden, which raider, every turn, both compulsory)
is the game: the warden part can swing a result (i), a warden can freeze the answerer the same turn
the raider walks in (25600 seed1 T33), and the end is zugzwang and frozen-out losses as predicted.
Its failure modes don't show: no shuffle, no seat lean, the freeze binds. Two claims don't hold:
the *order* of the parts is nearly meaningless (only vacate-then-fill; freezing the answerer works
in either order, since nothing moves between the parts), and the loss by having no warden left never
fired in the six transcripts (I did not inspect my extra games for it), though wardens are taken often. The warden part is also at
times a free shuffle (design failure mode 4, partly present).
