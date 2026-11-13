# smugglers: designer session record (`unlike-search.md` §2)

**Admitted at the first read** (`73da5f8f7bd1`: i y, ii y, iii y, iv y, v y).


## Hand-in: design reason

# Design reason (written before any spec)

Seed: a race that banks its arrivals, with a hand (`rules.reach` `removes` and `count`, `rules.drops`).

## The game in one line

Smugglers cross a border. Each side has couriers, which only walk (forward or sideways) and
never fight, and patrols, which step one square any way and take anything. A courier that
reaches the far row is delivered: it leaves the board and counts. The first side to deliver a
set number wins. A courier or patrol you take is turned: it goes into your hand, and you may
later put it back on your own home rows as one of yours instead of moving.

## The decision a player keeps facing

Every turn is a choice between three uses of one move:

1. **Run**: push a courier forward. It is the only way to score, but a courier cannot defend
   itself; one that walks into a patrol's reach unescorted is taken, and a taken courier is
   worth double to the opponent (one fewer for you, one more for them, delivered from their
   own home row).
2. **Hunt or escort**: move a patrol. A patrol that hunts takes couriers and patrols and fills
   the hand; a patrol that escorts stands next to a courier so that taking it costs the taker
   a patrol. A patrol cannot do both, and a patrol far forward hunting is not at home stopping
   the other side's couriers.
3. **Turn a prisoner**: drop a piece from the hand. A turned courier enters on your home row,
   the full length of the board from its goal; a turned patrol enters on your second row,
   ready to defend. Which to bring back, and when, is the hand's question: a courier is a
   point a long way off, a patrol is safety now.

Neither answer is obviously right: running without escort feeds the other side's hand,
escorting is slow, and hunting leaves your own goal open.

## The shape of a game

- **Opening**: both sides walk couriers up behind a screen of patrols; the patrols meet in the
  middle. The first contact is a patrol-for-patrol or patrol-for-courier trade, and the hands
  start to fill.
- **Middle**: convoys (a courier or two with a patrol beside) try to get through the other
  side's patrols, while lone patrols raid for stragglers. Material flows back and forth
  through the hands, and turned couriers re-enter at home, so a side that loses couriers
  sees them come back for the other side.
- **End**: deliveries thin the board, since every courier delivered is gone for good. The
  count shrinks to one or two needed, the race is close, and the last courier's run through
  the last patrols decides it.

## What would make it fail

- **A wall**: if patrols (plus pieces dropped back) can close a whole row in front of a goal,
  the defender sits and the game is drawn. Dropping a patrol onto the very row the enemy is
  aiming for would make that easy, so patrols must not be able to stand on their own home
  row (the row the enemy is racing to), and the board must be wider than one side's patrols.
- **A rush**: if a courier can be walked straight through before the patrols engage (the
  first player is a move ahead in any race), the game is a sprint and the patrols are scenery.
- **A shuffle**: sideways steps let a courier mark time; if waiting is always safest, nobody
  runs and the clock decides.
- **A dead hand**: if captures are rare or dropping is always wrong, the hand is decoration.
- **Patrols decide everything**: if hunting is so strong that couriers never get through
  except when the other side runs out of patrols, the race is a pretext for a fight.
- **Seat**: whoever runs first may simply arrive first.


## Hand-in: break attempts and design log

# smugglers: design log and break attempts (session 4)

Frozen: `game.json`, **smugglers [73da5f8f7bd1]**. Working files (drafts, probes, play
outputs, tournaments) are in `runs/R-S10/sessions/4/` of the worktree; the generator is
`gen.py` beside this file. Nothing forbidden was run or read: no `score`, `test`, `measure`,
`analyse`, `premise`, `verdict`, `heuristics`, `values` or `train`; tournaments were read only
for results by seat, lengths and transcripts (`seats.py`, `summ.py`, `moves.py`).

## Design log

| version | hash | change | why | what happened |
|---|---|---|---|---|
| v1 | 811b992a81a5 | 9x9; couriers (step forward or sideways, take nothing) a1 c1 e1 g1 i1; patrols (one step any way, take anything, never on own back row) b2 d2 f2 h2; deliver 4 of the far row (`reach` `removes`, `count` 4); captures to hand, couriers dropped on own row 1, patrols on row 2 | the design reason as first written | mm6400 self-play: patrols ate every courier, couriers in hand were never dropped, draw by repetition (131 plies). Running was never right: a courier one step a turn cannot outpace a patrol that steps any way. |
| v2 | 119c9f082ec6 | courier forward step up to 2 | give couriers pace so patrols must intercept ahead, not chase | Both seeds drawn at the 300-ply limit; plenty of drops and trades but 1-3 deliveries a side in 300 plies. |
| v3 | d199cc35d4d0 | v2 + count 3 | race too long for the material churn | Decisive 2 of 3 but 200+ plies; much of it sideways shuffles. |
| v4 | f79d23ed082d | v2 + escort shield (a courier next to its own patrol cannot be taken) | make convoys real | No clear gain (1 decisive of 3, a repetition); dropped as an extra rule that did not earn its place. |
| v5 | 21af289744d2 | 7x7, 4 couriers a1 c1 e1 g1, 3 patrols b2 d2 f2, count 3 | smaller board, shorter race | 3 of 4 decisive but one repetition at 38 plies: a courier and a patrol mirrored sideways steps. Two games ended by a side being stripped bare (no legal move). |
| v6 | 10a354bf43bd | v5 courier: forward step 1, sideways step, and a jump forward (straight or diagonal) over an adjacent piece | a courier blocked head-on by a patrol had no answer; the jump makes couriers slippery and fast in company | Repetitions (40, 95 plies): two couriers facing each other, the defender's on its own back row, shadowing sideways step for step. The sideways step is the shuffle. |
| **v7** | **73da5f8f7bd1** | courier: no sideways step; forward step straight or diagonal; the forward jump kept | every courier move irreversible, so no shuffle; a single blocker can be walked round | 6 of 6 decisive, 44-136 plies, close races (W2 B3, W3 B1...). Kept: this is the frozen game. |
| v8 | 92d3aba1e026 | v7 without the patrol region, both kinds dropped on the back row | test whether the region rule is needed and whether one drop row is simpler | One draw at the ply limit, and seed 0 shows the failure the reason predicted: four patrols dropped straight onto the back row (the other side's goal) walled it, the attacker's couriers were eaten and it lost with no move. Rejected; v7 kept. |

Probe-only variants of v7 (not candidates): `x-nodrops` afdb333e7b4f, `x-nohop` 541a98c7b9c3,
`x-noregion` 55260cf61ac1 (see dead-rule tests below).

The frozen file adds only `name`, `summary` and `description` to v7 (outside the hash; the
hash did not move).

My view after play, against the reason (which is unchanged): the decision I wrote holds, and
play added one thing the reason did not foresee: the jump makes couriers move in company
(leapfrogging their own patrols and each other), so escorting is not only protection but
speed. The reason's "sideways step" for couriers was the shuffle it warned of, and went.
Also not foreseen: a side stripped of every piece with an empty hand loses by having no move;
that is about one game in eight, and reads as the fight winning instead of the race.

## Break attempts

1. **A rush** (`probe_rush.py`): a scripted rusher that always plays the courier move that gains
   most ground (dropping couriers when none can move), against mm6400, 4 games each seat on
   v7. Result: the rusher lost all 8 (31-44 plies). The patrols opposite start right in front
   of the goal row, so an unescorted dash is eaten. Closed.
2. **A fortress / shuffle**:
   - v1 to v6 had shuffles (patrols mirroring on 9x9; courier sideways steps on 7x7); removing
     the courier's sideways step (v7) closed it: in the 96-game tournament on v7 there was one
     repetition and six ply-limit draws, all in pairings with the weak 1600-node player; the 48
     stronger games on the frozen spec had no draw at all.
   - A turtle (`probe_turtle.py`): one side never moves a courier past its second row and
     otherwise plays the engine's best-scored move; vs mm6400, 3 games each seat on v7: the
     turtle lost all 6 (48-79 plies), by deliveries.
   - A wall turtle (same probe, `WALL=1`): additionally always maximises the number of its own
     back-row squares occupied. On v7 it lost every game that finished (3 of 3 in seat 0; the
     seat-1 run stopped on a probe bug, an empty candidate list, not a game result). On the
     no-region variant it lost all 6 too.
   - But v8 (patrols dropped onto the back row) produced a real wall in self-play: four dropped
     patrols on the goal row, after which the attacker was ground down. The region rule plus
     patrol drops on the second row is the fix, kept in v7: the border must be held from in front.
   Closed for what I could find.
3. **A forced line**: shortest games on the frozen spec were 38-41 plies, all decided by
   several captures and deliveries, not a repeated sequence; the three 6400 self-play
   transcripts open differently after move 1 and run 44, 136 and 68 plies. Every game opens
   with a courier jumping its own patrol (a natural developing move, not a forced line).
   No short forced win found. Not proven absent.
4. **A dead rule** (6 mm6400 self-play games per arm):
   - no drops (`x-nodrops`): 3 repetition/no-progress draws and 3 wins by stripping a side bare
     in 38-48 plies; deliveries 0-2 a side. The hand is what keeps the race alive: kept.
   - no jump (`x-nohop`): 3 draws at the 240-ply limit, the rest 120-171 plies with 21-46
     captures. The jump is what lets couriers get through: kept.
   - no patrol region (`x-noregion`, drops unchanged): played much like v7 (6 decisive, 57-80
     plies) and its turtles lost; the bots never stood a patrol on its back row. The rule is
     kept on v8's evidence (dropping patrols onto the goal row walls it) and because without it
     a patrol walking back is two moves from the same wall; I could not show that wall holding
     against the bots, so this rule's case rests on v8 and on reasoning, not on a measured break.
   - "a player who cannot move loses" fires in 12 of 96 (v7) and 7 of 48 (frozen) games, always
     a side with nothing left on the board or in hand.
5. **A dominant piece or move**: move kinds over the six v7 seed games: courier steps 28%,
   courier jumps 23%, patrol moves 21%, patrol captures 18%, courier drops 6%, patrol drops 5%.
   Both piece kinds and the hand all do real work; no single move type decides games.
6. **Seat**: the first player moves first in a race, but the other side's patrols stand in front
   of its goal. Tournament on v7 (mm1600, mm6400; 96 games): seat 0 40, seat 1 49, draws 7; mm6400
   self 14-10. Frozen spec (mm6400, mm25600; 48 games): seat 0 26, seat 1 22, no draws; mm25600
   self 7-5, mm6400 self 8-4. No lean beyond noise at these sizes.

Strength reads through: mm6400 beat mm1600 37-9 (v7); mm25600 beat mm6400 23-1 (frozen), and
games get longer as the players get stronger (median 68 plies at 6400 self, 109 at 25600 self).

**Break not fully closed**: the wall the region rule exists to prevent was seen only in v8
(drops onto the back row); whether a patient human could still build a slow wall in front of
the border (second row) with a full hand is not tested beyond the turtle probes.

## Tournaments (2, both under a heavy x4 lease, released after each)

- `t1-v7`: v7, `minimax:nodes=1600 minimax:nodes=6400`, `--games 24` (4 ordered pairings x 24 = 96).
- `t2-final`: frozen spec, `minimax:nodes=6400 minimax:nodes=25600`, `--games 12` (4 x 12 = 48).

Probe runs (rush, turtle, wall turtle) were also run under the heavy x4 lease, one at a time.

## Final checks on game.json

```
$ uv run python unlike/limits.py game.json
game.json [73da5f8f7bd1]
  (a) likeness to chess 0.0011  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game crossings [c538506f18e4] 0.3036  below 0.5: yes

$ uv run chess-variant-sandbox validate game.json
smugglers [73da5f8f7bd1] is valid
  board       7x7
  pieces      C, T
  opening     25 legal moves
  rule mass   50.0

$ uv run chess-variant-sandbox smoke game.json
smugglers [73da5f8f7bd1]: passed
  draw_rate        0.02
  forced_rate      0.0
  games            150.0
  mean_branching   24.01
  mean_plies       115.59
  opening_moves    25.0
  stall_rate       0.02
```

## Transcripts (frozen spec)

- `6400-self-seed0.txt`: player 1 by delivery, 44 plies.
- `6400-self-seed1.txt`: player 1 by delivery, 136 plies.
- `6400-self-seed2.txt`: player 1 by delivery, 68 plies.
- `25600-v-1600-seed0.txt`: 25600 by delivery, 61 plies.
- `25600-v-1600-seed1.txt`: 25600 by delivery, 51 plies.
- `25600-v-1600-seed2.txt`: 25600 by leaving 1600 with no move, 75 plies.


## Blind read 1 (read-4)

# Blind reading: smugglers (reader 4)

Rules from `rules.txt`; spec loaded only through the tools. Tools used: `play` (single games),
`validate`-free Engine probes through `ask.py` in this packet (replays a written line and asks
the core for the legal moves and one agent's choice; plays no game). No lease taken, no score
or forbidden command used.

Note before the attempts: the rules differ from the reason in one detail. The reason has
couriers walk "forward or sideways"; the rules have them step forward straight or diagonally
and jump forward over an adjacent piece. Couriers therefore cannot mark time at all; only
patrols can wait.

## Attempts

1. **Rush, unescorted (a-file leapfrog).** White: 1.Cc1-a3 2.Ca1-a2 3.Ca2-a4 4.Ca3-a5 5.Ca4-a6,
   black answering with `minimax:nodes=25600` each move. Black replied Cg7-e5, Ce7-g5, Tb6-b5,
   Tb5xa5, Ta5xa6: both runners taken by one patrol, black two couriers up in hand. A courier
   run without escort feeds the opponent's hand, as the reason says. Failed.
2. **Rush by the bots' own opening.** Every transcript opens with couriers jumping over their
   own patrols to the third/fourth row (Cc1-a3, Cg1-e3, Ca1-a2-a4). The first delivery in the
   six transcripts comes no earlier than ply 25 (6400 seed 0, 16...Cd2-e1 after a patrol
   exchange sequence); in none is a courier walked through before the patrols engage. No
   sprint found.
3. **Wall / fortress.** Patrols cannot stand on their own back row (the enemy's goal row); a
   courier dropped onto your own back row does occupy a goal square, but enemy patrols may
   stand on that row and take it, and you have at most a handful of couriers for seven
   squares. A patrol standing directly in front of a courier with an empty square beyond is a
   screen the courier jumps over, so a static line in front of the goal is a liability, not a
   wall. In 40 games I saw or played (6 transcripts, 12 at 6400 self, 8 at 25600 self,
   4 at 1600 v 25600 reversed, 4 random, 6 greedy v 1600) there was not one draw: no
   repetition, no 100-turn, no 240-ply. Fortress not found.
4. **Shuffle.** Couriers cannot wait (no sideways or backward step). Patrols can: 6400 seed 1
   moves 38-46 is a patrol dance (Tf4-e4 / Te6-f6 repeated) that the players broke out of with
   drops (44...T@c6, 47.Ta4xb5). Waiting did not become the safest plan in any game.
5. **Forced line.** I found none. Different budgets disagree in the middle positions I probed
   (below), which is the opposite of a forced line.
6. **Dead rule.** Jumps are used in every game (most openings, and deliveries such as
   8...Cf4-f2 over a patrol). Drops appear in all six transcripts. Patrol back-row ban: it is
   what keeps the goal row open (attempt 3). Annihilation (no legal move) ends 4 of the 30 minimax-v-minimax games (and all 6 greedy games, where minimax wiped greedy out);
   it is live, mostly as a consequence of banked couriers leaving the board.
7. **Dominant piece/move.** Patrols do most of the capturing, as designed; couriers are the
   only scorers. In 6400 seed 0 white lost all three patrols by move 14 and black then walked
   three couriers in, so "patrols decide" happens in some games (also 6400 seed 2), but in
   others the race was 3-2 or 2-2 to the end.
8. **Seat.** Transcript self-play (6400, seeds 0-2): second seat 3/3. My extra self-play:
   6400 seeds 3-14 first seat 11 of 12; 25600 seeds 0-7 first seat 6 of 8. Those 20 extra
   games: first seat 17, second 3; with the transcripts, 17 of 23 (74%) to the first seat. Not decisive (the second seat wins a quarter of
   games, and strength dominates: 25600 beat 1600 from both seats, 7/7), but a real lean to
   the first mover, consistent with the reason's own worry that whoever runs first arrives
   first.

## The five questions

**(i) Yes.** Transcript `25600-v-1600-seed2.txt`, white's move 28 (after 27...T@d6). White has
banked 1, black has banked 1 and has a fresh courier on b7 heading for white's goal; white
holds a courier and a patrol in hand. Played: **28.C@a1**, a runner for the race, which gives up
a defender in front of the home row while black's b7 courier comes down. Alternative:
**T@d2** (the choice of `minimax:nodes=102400` when I asked at that position), a patrol on the
second row to meet that courier, which gives up a tempo in the race. 1600 and 6400 chose a
third move (Tg7-f6), 25600 under my probe seed C@d1. In the game white played both drops in
turn (28.C@a1, 29.T@b2) and the black courier still got through (33...Cf2-g1), so the
defensive drop was not simply right either.

**(ii) Yes.** Opening: couriers leapfrog over their own patrols to rows 3-4 and patrols step
to the centre (all six transcripts' first 4-8 moves). Middle: patrol-for-patrol and
patrol-for-courier exchanges fill both hands, and drops come back on the home rows
(25600 seed 0 moves 9-25: 9...Td5xd6, 12...Ta5xb5, 13.Tc4xb5, C@f7, C@d7; 6400 seed 1 moves
10-63). End: the board thins as couriers bank and leave, the count is one away, and a single
courier (often a dropped one) runs past the last patrols (25600 seed 0 moves 26-31,
C@a1-c3-c4-a6-a7; 25600 seed 1 moves 21-26). What matters moves from tempo, to material in
hand, to the race count. Caveat: in 6400 seed 0 the middle is a few moves long.

**(iii) Yes.** In each transcript the banked race with a hand decides the game:
- 25600 seed 0: white wins 3-1; the third delivery is a captured courier dropped 27.C@a1 and
  run to a7.
- 25600 seed 1: white wins 3-2 with black one short; the winner is a dropped courier, 21.C@a1
  to b7 at move 26.
- 25600 seed 2: 2-2; banked couriers leave the board, so black, having banked two and dropped
  its whole hand (C@g7, C@b7, T@d6), runs out of pieces and loses by having no move at 38.
  The banking-removes rule plus the emptied hand decide it.
- 6400 seed 0: black wins 3-1 with its original couriers after capturing white's patrols;
  white's defensive drop 20.T@e2 cannot stop the race. The count decides; the hand is used
  and fails.
- 6400 seed 1: black wins 3-2; the third is a dropped courier, 63...C@c7 to b1.
- 6400 seed 2: black wins 3-1; the third is a dropped courier, 28...C@e7 to b1.

**(iv) Yes.** I found no rush (an unescorted run lost both couriers in five moves; no delivery
before ply 25 in any transcript), no fortress or wall (the back-row ban and the jump make a
static line fail; zero draws in 40 games), and no forced line (budgets disagree in the middle).
Caveats: my probes were hand lines against the bots, not exhaustive, and there is a ~74%
first-seat lean in self-play that I could not turn into a forced win.

**(v) Yes, with reservations.** The three-way choice of the reason (run, hunt/escort, turn a
prisoner) is visible in every game, the hand is live in all six, the wall and shuffle failure
modes do not occur, and the shape opening/middle/end matches. Reservations: couriers do not
walk sideways as the reason says (they jump instead); in two of six transcripts one side
stripped the other's patrols and then walked in ("patrols decide everything" in part); four of
30 minimax games end by annihilation rather than a race; and the first seat leans (17 of 23 self-play games).
