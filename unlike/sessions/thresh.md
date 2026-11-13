# thresh: designer session record (`unlike-search.md` §2)

**Admitted at the first read** (`a52971eab2d1`: i y, ii y, iii y, iv y, v y).


## Hand-in: design reason

# Design reason (written before any spec)

Seed: a hexagonal board with hoppers and sweepers.

## The game in one idea

Two kinds of attacker want opposite boards. A **sweeper** runs along a line and clears
everything it passes, its own side's pieces as well, but it cannot cut through a **stone**:
stones stop it dead. A **hopper** moves by vaulting one adjacent piece and landing just
beyond it, taking what it lands on, so it is helpless on an open board and deadly in a
crowded one. Each side protects one piece, its keeper; take the other keeper and you win.

## The decision a player keeps facing

How close to pack. Stones packed round your keeper make it safe from the enemy sweepers,
which cannot pass a stone; but every stone next to your keeper is a step an enemy hopper can
vault onto it. Spread out, and the hoppers have nothing to climb, but the sweepers' lines
open. Every stone move asks: is this a wall or a ladder? And every sweeper move asks whether
the line is worth the friendly pieces it will mow down on the way, since a sweep clears its
own hoppers too. The same pieces are both armour and exposure, and which they are depends on
which enemy attacker is nearer.

## A game from opening to end

- **Opening:** both armies stand behind a row of stones. Nothing can take on the first move:
  the sweepers are walled in and the hoppers have only friendly screens leading nowhere. The
  players advance stones and hoppers to build the shape they want, a shield in front of the
  keeper and a ladder pointed at the enemy.
- **Middle game:** contact. Hoppers vault stones to pick off stones and each other; every
  stone taken opens a line, and the sweepers come out through the gaps, clearing whole lines
  of hoppers (often costing a friendly piece). Material thins.
- **Endgame:** with fewer stones, the keepers are increasingly exposed to sweepers, while
  hoppers lose their screens and dwindle in power. The last stones near each keeper are worth
  more than anything else; the game ends when a sweeper finds an open line or a hopper a
  stone beside the keeper.

## What would make it fail

- **A fortress:** a keeper ringed by stones with a gap (so no hopper can land next to it
  with a screen) that nothing can break, so the game draws.
- **A rush:** a first-move sweeper or hopper line to the keeper, or a forced early win for
  the first player.
- **A dead mechanism:** sweepers never get out (stones never fall), or hoppers never find
  screens, so one of the two seed pieces is decoration.
- **A dominant piece:** sweepers so strong on the open board that the game is just
  "open a line first", or hoppers so strong that stones are pure liability.
- **Shuffling:** both sides wait behind walls because any advance gives the other a ladder.


## Hand-in: break attempts and design log

# thresh: design log and break attempts (session 6)

Seed: hex board, hoppers and sweepers. Frozen game: `game.json`, hash **a52971eab2d1**.
`reason.md` was written before any spec and is unchanged. One change of view since: the
"keeper" there was a piece that walks; play (v1 to v4) showed a walking royal cannot be
caught, so the final game's keeper is an immobile **hearth**. The decision the reason names
(stone as wall against scythes or ladder for hares) is unchanged, and is sharper with a
hearth that cannot run from it.

All tournaments: 96 games (two agents, four ordered pairings x 24), 4 workers, heavy lease of
4 slots, released after each. Read only for results by seat, lengths, endings and
transcripts (replayed through `Engine` to see positions and final blows). Seven
tournaments in all, each under a minute except t7 (a few minutes): well inside the budget.

## Design log

| ver | hash | change | why | predicted | happened |
|---|---|---|---|---|---|
| v1 | 8ace98943f5f | Walking keeper (hex step, takes anything), 2 scythes (unlimited sweep, cannot take stones), 4 hares, 9 stones (step fwd/sideways, never take). Royal capture. | First draft of the reason. | Opening walled, middle hare fights, scythes out through gaps. | Hare was a classical grasshopper: it slides any distance to its screen (hop reads "range 0" as unlimited), so long-range, unlike the reason. One 1600 game: scythes traded early, then keeper-and-stones endgame drawn by no-progress at 277 plies. |
| v2 | 1ee33db6f282 | Hare hop `range 2` (screen must be adjacent); stones promote on the far edge to scythe or hare. | Match the reason (hare = crowd piece); give stones an endgame purpose. | Fewer dead endgames. | t1 (1600 v 6400): 31/96 ply-limit draws at 300, median 248 plies; 6400 clearly stronger; seat even. Lopsided material (up to +22) still drawn: the winning side could not catch a walking keeper on the open hex. |
| v3 | a85ad727ef3c | + `bare_royal: loss`. | Let material convert: strip the keeper bare. | Ply-limit draws < 15. | **Missed.** t2: 29 ply-limit draws, median 229; 13 bare-royal wins but losers keep a stone or two. Captures continue to ply ~290: the board clogs with stones only hares and keepers can remove. |
| v4 | 5134b420fe7c | Smaller army (3 hares, 7 stones), max plies 240. | Less clog, shorter games. | Median ~150, fewer draws. | **Missed.** t3: still 29 ply-limit draws, median 194, and the 6400/1600 gap shrank to 0.55. Diagnosis: the walking keeper itself, not the army size. |
| v5 | b08cb1cf2b27 | Keeper becomes an immobile hearth (`moves: []`, at g1 behind the stones); stalemate loses. Back to v3's army. | Remove the chase; make it the siege the reason describes. | Decisive, much shorter. | t4: 0 draws, median 43, but **strength did not matter** (6400 scored 0.49 v 1600) and scythes gave 93/96 final blows: they ran the open middle onto the hearth's back row (`Wd9xc9`). Stones never protected the hearth: they start in front of it and could not go back. |
| v6 | ab7b9c6ebb48 | Scythe range 3. | Stop cross-board raids. | Hares share the kills, strength matters. | t5: 6400 0.573; hares 8/95 kills only; **second player 60 wins to 35**. Hearth still exposed along the back row. |
| v7 | 215490eae962 | Hearth moved to g2 and fully ringed by six stones; stones step in all six directions; scythes on the corners e1/i1, hares d2 e2 i2 i3. | Put the reason's decision on the board from move one: the ring is the wall and the ladder. | Hares kill more; seat evens. | t6: 0 draws, median 44 plies, 6400 0.583, hares 20/96 kills, seat 42 v 54. Promotion and bare-royal never fired (0 promotions in 96 games). |
| v8 = final | a52971eab2d1 | Delete promotion and bare-royal. | Dead rules (deletion test: they never fired). Stalemate-loss kept: it is what makes a bare or frozen hearth lost rather than drawn. | Same play. | t7 (6400 v 25600): 25600 0.594; seat 42 v 43 over the decided games; hares 29/85 kills, scythes 56; 11 draws (10 repetition, 1 no-progress), of which 6 are one deterministic line repeated; median 52 plies, max 146. |

## Break attempts

1. **Rush.** (a) v5: a scythe raid across the open middle onto the hearth's back row won
   in 15 plies in three different pairings, the same line each time
   (`Wf1-b5 ... Wh5-d9 ... Wd9xc9`). Closed by v6 (range 3) and v7 (ringed hearth).
   (b) v8: the hearth starts with two hare entry cells empty on each side (for the second
   player e6 and c6, behind ring stones d7 and c7). The shortest v8 game (13 plies, t7 game
   70) is a hare walking to e6 and vaulting d7. Probed with `Engine` (`probe2.py`: a
   white hare dropped on each entry cell of the opening array): e6 is guarded (`He8xe6`,
   vaulting e7), **c6 is not** (no black capture there; the white hare would threaten
   `Hc6xc8`). The same holds for the first player's g4 by symmetry. It is not a forced
   win: a hare needs four or five moves to reach c6 from its camp, and on arrival the
   defender answers by stepping the c7 ring stone aside (no screen, no vault; the
   attacker's scythes are too far away to use the gap at that point). In t7 the
   short wins came from the defender moving a guard or a ring stone itself (`Sd7-c7`),
   not from this cell. I left the array as it is: guarding c6 would need an extra stone on
   h3, and the open entry is part of what the opening is about (where each side's hares go
   first). Recorded as a weakness a player must know, not a break.
2. **Fortress or shuffle.** (a) Walking keeper (v1 to v4): an unbreakable chase, not a
   fortress, ~30% ply-limit draws with large material leads; closed by the immobile hearth.
   (b) Ringed hearth as a fortress (v7/v8): reasoned through. A full ring stops scythes but
   each ring stone with a cell two steps beyond it is an entry (four of the six on the
   board for a hearth on g2). The defender must hold those cells and recapture hares
   landing there; when the defender runs out of hares the hearth falls, so material
   decides and there is no static fortress. The bots never sat in their rings (they
   dismantle them), so this is untested in play and is the break I could not close by
   experiment. (c) Shuffle: t7 has repetition draws at 25600, one of them reached by a
   walked-in stone parked beside the enemy hearth with both sides content to oscillate
   (game 84), and one line (game 57) repeated six times by determinism. About 10% of
   games; I left it, since a stronger player has moves to try in each.
3. **Forced line.** None found beyond 1(a), which is closed. Final-blow and length
   distributions (min 13, median 52) show no recurring short win at 25600.
4. **Dead rule.** Promotion (v2 to v7) and bare-royal (v3 to v7): never fired in 96 games of
   v7 (0 promotions). Deleted in v8. Stalemate-loss also never fired, but it decides the
   one ending a lost side would otherwise draw (a hearth with nothing able to move), so it
   stays. The hare's plain step is used constantly (repositioning to entry cells).
5. **Dominant piece or move.** v5: scythes 93/96 final blows (closed by v6 to v7). v8:
   scythes 56, hares 29 of 85 decided games; stones are the most-moved piece (walls, and
   invaders beside the enemy hearth), so all three move kinds carry the game.
6. **Seat.** v6 leaned to the second player (60 v 35); the mechanism I suspect is that the
   first mover must commit a stone first, and every stone move makes a screen. With the
   v7 ring it fell to 42 v 54 (t6) and 42 v 43 (t7, stronger agents). Self-play 25600
   alone was 7 v 14 over 21 decided games, so a small second-player lean may remain;
   not closed with certainty at this sample size.

## Hand-in transcripts (frozen spec)

- 6400 self, seed 0: first player wins, 75 plies; seed 1: second player, 40; seed 2: first, 75.
- 25600 v 1600, seed 0: 25600 wins, 87 plies; seed 1: **1600 wins**, 58; seed 2: 25600 wins, 107.

## Final checks (frozen spec, game.json)

```
thresh [a52971eab2d1] is valid
  board       9x9
  pieces      S, H, W, K
  opening     22 legal moves
  rule mass   31.5
game.json [a52971eab2d1]
  (a) likeness to chess 0.0067  below 0.0525: yes
  (b) pieces of chess 0  at most 1: yes
  (c) nearest library game intransitive [29d5213b02b2] 0.1009  below 0.5: yes
thresh [a52971eab2d1]: passed
  draw_rate        0.373
  forced_rate      0.0
  games            150.0
  mean_branching   40.03
  mean_plies       154.83
  opening_moves    22.0
  stall_rate       0.373
```

(The smoke draw rate is random play shuffling; it passes the gate.)

Working files beside this log: `v1.json` to `v8.json`, probe scripts (`probe1.py`,
`summ.py`, `replay.py`, `board.py`, `shortest.py`), tournament outputs in
`runs/R-S10/sessions/6/t1-v2` to `t7-v8`.


## Blind read 1 (read-6)

# Blind reading: thresh (packet read-6)

Sources: rules.txt, reason.md, the six transcripts, game.json loaded into the engine, my own probes
(`probe.py` board printer, `rush.py` scripted side against a bot, `scores.py` Engine.root_scores
over a line). No score/test/measure/analyse/tournament was used. Board facts used below: the
White hearth g2 starts ringed by six stones (f2, h2, f3, g3, g1, h1). A hare can take it only from
e2, i2, e4 or g4, and only when the cell between holds a piece. Black's mirror is c8, with attack
cells a8, e8, c6, e6. Hare vaults go both ways: a hare that guards an attack cell over a screen
can be taken from that cell over the same screen.

## Attempts

1. **Hare rush, west (White d2-d3-c4-c5-c6, then c6xc8 over c7), against minimax:25600.** Failed.
   Black answered by taking the screen away (Sc7-b7), so c6xc8 was illegal. That opens b7 as a
   ladder to a8 instead (in a 25600 self-play game, seed 6, White did take a8 that way).
   No forced win.
2. **Hare rush, centre (White e2-e4 by vault, e5, e6, then e6xc8 over d7), against 25600.** Failed.
   Black took the hare on e6 with We9xe6, sweeping its own e8 hare to do it.
3. **Black hare rush to g4 (e8-e6 by vault, f5, g4, then g4xg2 over g3), against 25600 and 1600.**
   Failed both times. White stepped Sg3-h3 and took the screen away.
4. **Passive fortress: keep the full six-stone ring and shuffle (Sc3-c4-c3, or Hi3-h4-i3), against
   25600.** Lost in 9 moves. A black hare walked to g4 and vaulted g3 onto the hearth. The ring
   is a ladder, so doing nothing loses. No fortress.
5. **"Gap" fortress: empty every cell next to the hearth and put stones two out on each open line
   (Sg3-g4, Sf3-e4, Sf2-e2, Sh2-i2, after vacating the hares), so no hare has a screen and no
   scythe a line; then bots continue.** Lost in 7 more Black moves against 25600, and in 18
   against 6400. A black hare took the e4 stone (He6xe4), a black stone walked in to f3 as a
   screen, and He4xg2 followed. An enemy stone is a screen the defender cannot cut away. No
   fortress.
6. **Seat.** Single games: 6400 self-play over seeds 0-12 gave first seat 8, second 5.
   25600 self-play over seeds 0-7 gave 5-2 with 1 draw. Together, 13-7-1 for the first seat: a
   mild lean, not a forced first-move win. The stronger player wins from either seat (25600 as
   Black beat 1600 in 4 of 4; as White it won 2 of 3 in the packet).
7. **Dead rule or dominant piece.** Moves by piece type over the six transcripts: stones 188, hares
   145, scythes 109. Captures: hares 19, scythes 27. Both attackers capture in every game. Both
   attackers delivered wins: scythes in 4 games, hares in 2.
8. **Forced line.** The strongest tactic I found is the hare-scythe battery on a file: in
   6400-self-seed1 after 19... Wg7-g5, Black's g4 hare and g5 scythe face g3/g2. A stone on g3
   becomes a ladder for the hare, and leaving g3 empty lets the scythe through
   (Sf3-g3 and Wd4xg4 both lose). It is a real mating net, but it has to be built in the middle
   game. It is not an opening line.
9. **Draws and shuffling.** 1 draw (repetition) in 21 bot games. There is some shuffling in the
   transcripts (25600-v-1600-seed2 moves 19-22, Wb6-e3 / Hc3-b4 back and forth; 6400-self-seed0
   moves 25-30), but every one of those games was still decided.

## Answers

**(i) Yes.** Transcript 25600-v-1600-seed2, move 6, White to move (ply 10). There are two moves:
- **We1xe3 (sweeps e2)**, played. White mows down its own e2 hare to free the walled scythe, which
  then threatens the black hare on b6 and takes it next move (7. We3xb6). It gives up a hare and
  the hare guard in front of the hearth, in return for an active scythe in Black's half and
  material won back at once.
- **Sh2-h3** (or another quiet stone move). White keeps every piece and keeps the scythe walled
  in. It gives up the tempo, and the b6 hare survives.
A depth-5 root probe puts the quiet moves at 0.0 and the sweep at -2.65 (under one hare). The
sweep wins its material back the next move, and White went on to win. Both moves are defensible.
This is the reason's "is the line worth the friendly pieces it mows down" choice, met over the
board.

**(ii) Yes, with a caveat.** Opening: no contact for 6-17 moves in every transcript. Both sides
push stones and hares into the middle (first capture at moves 18, 13, 6, 17, 8 and 11). Middle:
hare fights over the cells two out from a hearth, with scythes coming out by sweeping their own
hares (e2 in seed2, a8 and e8 in 25600 self-play, i2 in seed1). End: in the longer games
(25600-v-1600 seeds 0 and 2, 6400 seeds 0 and 2), hares are nearly gone. What matters then is
stones walked up beside the enemy hearth as screens (seed2: White's c7 stone, then Hc6xc8), or
open scythe lines on a thinned board (seed0: Wc6xc8). The caveat: 6400-self-seed1 (ply 40) and
25600-v-1600-seed1 (ply 58) end by a middle-game knockout with most material still on. Also,
stones mostly survive to the end. The reason's "fewer stones" endgame is really "stones that have
moved", not stones that have been taken.

**(iii) Yes.** In each game:
- 25600-v-1600-seed0: hare vaults took stones (Hb6xb8, Hd7xf7, Hf7xd7) and opened the c-file. The
  scythe won with Wc6xc8 along an open hex line.
- 25600-v-1600-seed1: White's Wi1xi3 swept its own i2 hare, and a black hare vaulted and took that
  scythe (Hg5xi3). The black scythe entered through the broken ring: We3-g3, Wg3xg2.
- 25600-v-1600-seed2: an early own-hare sweep (We1xe3) and scythe raids. It ended with a hare
  vault, Hc6xc8, over Black's own stone on c7, a ladder.
- 6400-self-seed0: a scythe exchange on c3/b6 thinned the hares. White's hare vaulted Black's d8
  stone onto the hearth (He8xc8).
- 6400-self-seed1: the hare-and-scythe battery on the g-file. White could neither wall (that
  makes a ladder) nor leave the line open: Wg5xg2.
- 6400-self-seed2: the scythe swept through its own e6 hare along the f5-c8 diagonal to win
  (Wf5xc8 sweeps e6).

**(iv) Yes.** I found no rush, fortress or forced line that wins. Three hare rushes were met by
taking the screen away or by a scythe sweep. Two fortress shapes (the full ring held passively,
and the empty-neighbour ring with stones two out) both lost quickly, because each shape is a
ladder against one attacker. The seat lean is 13-7 over 21 bot games, mild. The g-file battery is
a real tactic, but it comes in the middle game, not as a line a person can play from the start.
Limit: my probes used bots as the opponent, and the bots are visibly weak here (6400 walks into
nets, and 25600 lost one game to 1600). A deeper forced line may exist that I did not find.

**(v) Yes.** The wall-or-ladder decision is the game's real tension. The ring held passively loses
to a hare ladder, the ring opened loses to an enemy stone screen, and a stone placed to block a
scythe can hand a hare its vault (g-file battery). Self-sweeps that cost friendly hares recur in
5 of 6 games. Both seed attackers decide games, and no fortress, rush or dead piece turned up.
Where it departs from the reason:
- "Nothing can take on the first move" is not strictly true: Wi1xi4 sweeps two of your own hares.
  It is a capture only of your own pieces.
- The endgame is reached through stones moving, not stones being taken.
- Some games end before an endgame exists.
