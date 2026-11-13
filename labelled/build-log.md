# The labelled set's build log

Appended to, never rewritten. Each entry says what was built, skipped, left empty or dropped,
and why, so that nothing was selected by a reason nobody can read. The design is
`docs/reference/labelled-set.md`; section numbers below are its.

## R-S2b, 2026-11-11: the home, the guard, the good games

Branched from main at `74eaa6ca2`. Nothing in this entry was measured or played by any of the
project's players; no run directory was created for any game in `labelled/`.

### The 21 good library members (§6 step 2)

Labelled from `outside-test.md` §2.3's tiers, unchanged, each `measured_before_label: true`.
Their likeness was recomputed today with `spec.likeness` (`k1`, IDF over the 131 library games
at `74eaa6ca2`), and every one fell in the quarter §2.2's table gives it: **Q1 7, Q2 7, Q3 5,
Q4 2**, as asserted. Sittuyin and placement chess keep their *flagged* mark; the thirteen of
§1.1 (b) carry `in_scale: true`. What I had seen of their scores is in each label's `seen`: in
this task I opened no score; labelled-set.md itself quotes dobutsu's and Kyoto shogi's.

### The new published games (§6 step 3): six written, five skipped

Walked in §2.2's order until six passed both tests. Test one is the game (the rule language
says it with edge-rule departures only, no new primitive); test two is the label (lowest W at 2
or above, from a source read today). Sources were read on 2026-11-11.

| # | candidate | verdict | why |
|---|---|---|---|
| 1 | janggi | **skipped, test one** | Two departures that are not edge rules. *The opening*: the players choose before the first move which of four arrays of horses and elephants to set up; a spec is one array, and the language has no setup chosen by the players. *The ending*: bikjang, generals facing on an open file with the player to move not breaking it, ends the game (a draw in Wikipedia's account, a count of points under the federation's rules); `rules.royal_facing` says only *allowed* (nothing happens) or *forbidden* (xiangqi's rule, an illegal move), neither of which is janggi's. Also left unsaid but edge: two passes in a row draw. Everything else is expressible since schemas SR1a, SR3c and H3 (stated legs, a screen named by type, the pass). A reader who counts a fixed sanctioned array and a facing draw as edge rules could admit it as flagged; §2.2 as written skips it. Missing: a players' choice of setup, and a facing rule that ends the game. |
| 2 | minishogi | **written** (`labelled/minishogi.json`, Q1, flagged) | Test one: Q533's audit spec, its pawn's promotion made optional (`forbid_immobility` already forces it on the one-rank zone, so the game is the same) so that Fairy-Stockfish takes it. One edge departure, stated in its description: a fourth repetition is drawn, where the published rule makes it the first player's loss. Test two: on PyChess's list (H); computer tournaments yearly and a Japanese Minishogi Association (Wikipedia). W2. |
| 3 | Glinski's hexagonal chess, mirrored armies | **written** (`labelled/glinski-hexagonal-chess.json`, Q1, flagged) | Test one: the archive's hexagon with the second army placed as the first's mirror image (file kept, rank reflected: in axial cells `(q, r) -> (q, 15 - q - r)`, 0-based), kings on one file; en passant on; the double step from any pawn starting cell of its side (`condition.region`), which is Glinski's rule as Wikipedia states it, where the archive game has `first_move`. Pawns, bishops, rooks, queen, king and knight read alike under a 180-degree turn and a mirror, so the second player's moves are the rule's. One edge departure: stalemate is a draw (Glinski: three quarters of a point to the stalemating side). Opening: 51 legal moves for each side. Test two: over half a million players at its peak, 130,000 sets sold, a federation and a European championship (Wikipedia, Hexagonal chess). W2. Fairy-Stockfish has no hexagonal board, so `validate` alone. |
| 4 | tori shogi | **skipped, test one** | The swallow may not be dropped on a file already holding *two* of its side's unpromoted swallows; `drops.no_doubled` says one, and leaving it out says none. A limit on which drops are legal is a rule of how pieces enter play, not an edge rule. Also, the eagle's move is stated two ways in the source (whether its two-square limit covers all its ranging). Missing: a count on `no_doubled`. |
| 5 | Cambodian chess (ouk chatrang) | **skipped, test one** | The king's one-time knight leap is barred while in check; a move condition has `from_rank`, `first_move` and `region`, and no *not in check* (`variant-expressibility-2026-09.md` row 17). A departure in how a piece moves. Missing: `condition.not_in_check`. |
| 6 | breakthrough | **written** (`labelled/breakthrough.json`, Q1) | Test one: exact, two rows of runners each, a step straight or diagonally forward, capture diagonally forward only, the far row or taking every piece wins. Who moves first is the players' choice in the source, as colour is in chess; here the first player moves first. Test two: added to PlayStrategy, a public server of 40 games, in July 2024 with tournaments (its blog), and the 2001 8x8 Game Design Competition (Wikipedia). W2. |
| 7 | minixiangqi | **written** (`labelled/minixiangqi.json`, Q1, flagged) | Test one: Q533's audit spec, Fairy-Stockfish's array. One edge departure: perpetual check and chasing are an ordinary repetition draw, as in the library's xiangqi. Test two: on PyChess's list. W2. |
| 8 | Judkins shogi | **skipped, test two** | Expressible, but no source I could find names organised play: no PyChess or Lishogi place, only a Game Courier preset (which exists for nearly every variant and is not a standing place). W1 at best. |
| 9 | Janus chess | **written** (`labelled/janus.json`, Q3) | Test one: exact; castling by destination file (king to b or i, rook beside it inside), `tests/test_schema_two.py`'s perft-pinned spelling. Test two: regular tournaments in Europe and a place on BrainKing's server (Wikipedia). W2. |
| 10 | shako | **written** (`labelled/shako.json`, Q3) | Test one: Q533's audit spec, Fairy-Stockfish's array and castling. Test two: on PyChess's list. W2. |
| 11 | lines of action | **not reached** | Six were written at row 10. (It would fail test one: a move's length is the count of pieces on its line, and the win is connecting one's pieces.) |

**Q1 now holds 11 good games** (7 library, 4 new), above §2.2's floor of 9; Q2 7, Q3 7, Q4 2.

**Likeness of the new games**, each appended alone to the 131 library games at `74eaa6ca2`
(§2.1): minixiangqi 0.0099, breakthrough 0.0155, Glinski 0.0178, minishogi 0.0270 (Q1); shako
0.1685, Janus 0.2742 (Q3).

**Proved by `validate`** for all six. **`verify` against Fairy-Stockfish** (`chess-variant-sandbox
fairy`, depth 3, three 400-ply walks) for the five it can play: shako and Janus verified against
the installed binary, with results agreeing against `pyffish`. Minishogi verified there too.
Minixiangqi diverged on the 2021 binary only, whose custom royal steps as a king (it offered a
diagonal general's move, `d1c2`, the binary's documented lack of custom royals); verified against
`pyffish` 0.0.90 instead, to depth 3, and its results agree. Breakthrough diverged on the binary
only, which cannot read `flagPiece = *`; verified against `pyffish` the same way, results
agreeing. Glinski: no Fairy-Stockfish board.

**`tests/check_new_game.py`** took the six paths: the corpus pins are current and the 31
`new_game` checks it ran pass. It prints a `FROZEN` row for each; none was added, since
`test_schema_nine.py::FROZEN` covers `rulesets/` and `archive/` only. Each game's identity is
pinned by its label's `spec_hash` instead (`tests/test_labelled_set.py`); its compiled-tables
fingerprint is not pinned anywhere.

### Not done here

No bad games, controls, certificates or seal: those are R-S2b2's. `verified.json` is empty.

## R-S2b2, 2026-11-12: the bad games, the controls, the seal

Branched from main at `b6e02eec2`; the library (`rulesets/`, `archive/`) is unchanged since
`74eaa6ca2`, so every likeness here is read against the same 131 games as R-S2b's
(`labelled/likeness.py`, which refuses to run if the library has moved). Nothing in this entry
was measured or played by any of the project's players: no tournament, `measure`, `test`,
`smoke`, `play`, `premise`, duel or designer measurement was run on any game here or on any
parent, and no run directory was created. I saw no score of any game built here or of any
parent (labelled-set.md quotes 0.756 for a flat-army xiangqi mutant, which is not xiangqi).

**How the proofs were made.** Defects were found by thought and by exhaustive searches over
the engine's legal moves (`labelled/prove.py`: an AND-OR search for forced wins, a
reachable-position search with one side held to a recipe, both asking `Engine` only for
moves, positions and endings). Each proof is a certificate walked by `labelled/verify.py`
(seven) or an argument whose mechanical part is data in the label (`proof.checks`) re-run by
`labelled/check.py`. `verify.py` now also takes a winner who moves second (the tree opens
with the loser's every first move) and a loser's reply that itself ends the game as the
winner's win (a forced capture in antichess), written as a null leaf.

**The blind reads.** Each proof was read by a fresh agent given the spec, `describe_rules`'
text, the claim and the proof and nothing else, told to try to refute it with the engine and
by hand, and to run no player. Seven readers took three or four games each; the replacement
game below had a reader of its own. Most readers wrote their own searches (keyed on the
board, or counting plies) rather than trusting `prove.py`. 27 verdicts are `holds`; the
reader's notes are in each label's `review`.

### What was built: 23 bad games and 4 controls

| quarter | plain forced-win | plain no-win | plain cannot-lose | hidden forced-win | hidden no-win | hidden cannot-lose |
|---|---|---|---|---|---|---|
| Q1 | breakthrough-runner-on-the-seventh | racing-kings-with-a-low-ceiling | racing-kings-with-a-tilted-ceiling | breakthrough-on-three-files | held-runners | two-runners-one-king |
| Q2 | antichess-with-a-lone-queen | chess-without-captures | grand-chess-without-a-white-king | antichess-pawn-ending | pawn-wall-race | pawn-wall-race-late-king |
| Q3 | king-of-the-hill-head-start | capablanca-locked-wall | horde-without-extinction | horde-endgame | capablanca-held-runners | capablanca-two-runners |
| Q4 | atomic-forward-knight | chess-locked-wall | three-check-without-a-white-king | **empty** | chess-held-runners | chess-two-runners |

Controls (`null`): xiangqi-with-a-sentry (Q1), makruk-with-a-sentry (Q2),
king-of-the-hill-with-a-sentry (Q3), three-check-with-a-sentry (Q4): the parent with a piece
type added to the list that no army includes and nothing can create.

§2.3's minimums hold: 23 of 24 (at least 20), 6, 6, 6 and 5 by quarter (at least 4), and every
kind-and-stratum pair has at least 3 (at least 2). Parents per quarter: Q1 breakthrough 2,
racing kings 2, none 2; Q2 antichess 2, chess 1, grand 1, none 2; Q3 Capablanca 3, horde 2,
king of the hill 1; Q4 chess 3, atomic 1, three-check 1.

**The held-runner family.** Eight hidden games share one construction: a locked pawn wall no
king can cross, one runner a side already through it, and a king that holds the enemy runner
by stepping between two squares (and taking the runner if it steps up). The no-win games give
both sides that recipe; the cannot-lose games give it to one side and leave the other's
result unclaimed (in `two-runners-one-king` the exact value, from a retrograde solve while
building, is a draw; in `pawn-wall-race-late-king` too; the chess and Capablanca ones were not
solved). Where the parent wins only by checkmate, the reachable-win lines run through two
promotions to queens before mate.

### Cells left empty, games dropped

- **Q4 hidden forced-win: left empty after two attempts the builder could not prove.**
  (1) King and rook against king and pawn on the chess board, looking for a forced mate in 6
  to 8 with a certificate small enough to keep: the AND-OR search did not finish one position
  in 60 s, too slow to find one. (2) The locked chess wall with Black's king in a pocket of
  stacked pawns and two white knights to mate it: no forced mate within 8 White moves (the
  stacked pawns guard every square a knight would mate from). A Q4 parent other than chess
  (atomic, three-check) falls to Q3 once its army is thinned, and chess thinned needs its
  castling rule removed, which leaves no second change for a rule that would make a short
  forced win expressible.
- **three-check-rook-wall (Q4 plain no-win): dropped by the builder before its blind read.**
  Its argument said no capture was ever possible; `check.py`'s `no_capture` found one: a black
  rook can step onto a5, c5, e5 or g5, where White's wall pawns take it. Replaced by
  `chess-locked-wall` (kings and pawns only, every position visited).
- **antichess-endgame (Q2 hidden forced-win): refuted by its reader, dropped, rebuilt.** Its
  claim said White forces a win in 7 of its moves; the reader's own search found one by
  Black's forced reply to White's 6th. The flaw was in `prove.py`: with one move left the
  search looked only for moves that end the game at once and missed a win that comes on the
  loser's forced reply, so its "no sooner win" count could be one too high in a game of forced
  captures. Fixed (the early return removed), and the other two hidden forced wins were
  re-checked with the fixed search; their readers had counted plies independently and agree.
  The cell was rebuilt as a different game, `antichess-pawn-ending`, found with the fixed
  search and read blind by a new reader (holds: a forced win on Black's reply to White's 7th
  move, none sooner). The dropped spec and certificate were removed in the same branch, before
  the seal.

### Not done here

Nothing measured; that is R-S2c's. `labelled/SEAL.json` is written by the commit after this
entry's.

## Q1215, 2026-11-13: the seal's digests re-recorded on LF bytes; no member changed

**What was wrong.** The seal (57df879ff) and `verified.json` hashed each file's working-copy
bytes. The R-S2b2 files were written by Python on Windows, whose `write_text` turns `\n` into
`\r\n`, so the sealer hashed CRLF; git stores every one of them with LF (`.gitattributes`:
`* text=auto eol=lf`), and a fresh checkout on Windows or Linux holds LF. So 27 of the 54
label digests (all 23 bad games and 4 controls; the 27 good-library labels were LF and matched)
and all 7 certificate digests in `verified.json` matched no checkout but the sealer's.
`test_the_seal_matches_the_files_it_lists` and `test_proof_labels_hold_and_their_certificates_verify`
failed on main from then on.

**Checked before touching anything.** For all 61 digests (54 labels, 7 certificates), each was
compared with the committed blob at 621c15883 (raw, from `git archive` with
`core.autocrlf=false`): 27 equal the blob's SHA-256 as stored, 34 equal the SHA-256 of the blob
with every LF written as CRLF, and none matches neither. Every blob is LF only. So the content
sealed is the content committed, byte for byte up to line endings.

**The fix, at the root.** `labelled/verify.py`'s `digest` folds CRLF to LF before hashing,
which is the committed blob on any checkout; the guard test hashes with that same function. The
34 digests were re-recorded on that basis: `SEAL.json` `labels` (27) by a one-off script, and
`verified.json` (7) by re-running `uv run python labelled/verify.py`, which walked all 7
certificates again (7 of 7 hold). No label, game or certificate file was edited (their
single-commit history is what `test_each_file_is_written_once...` checks, and it passes).

**Checking the old digests.** `SEAL.json` now carries `rebased`: the date, the basis, the reason,
and under `previous` every replaced digest (`labels`, `verified`). Each previous digest is
`sha256(file.replace(b"\n", b"\r\n"))` of today's file, and
`test_the_seals_rebase_keeps_the_old_digests_checkable` asserts exactly that, so the old seal
stays evidence that this content was fixed on 2026-11-12. The old `SEAL.json` itself is at
57df879ff. R-S2c's reading (11bf34048) used these same files; nothing in it moves.

## R-S10b1, 2026-11-13: four published additions for the search for strong games unlike chess

Branched from main at `34057d4c5`. These are additions under `labelled-set.md` §3.3, entered
for `docs/reference/unlike-search.md` (§1.2, §5), and in no registered reading of this set.
Nothing here was measured or played by any of the project's players, and no run directory
exists for any of the four (checked in this worktree's and the main checkout's `runs/` before
the labels were committed). I opened no score of any game here or of any parent, nor the
four built candidates' scores R-S2c holds (`reading.json`, `runs/R-S2c`, §9 of the design).

**The walk** (`unlike-search.md` §1.2's order, until four were written; sources read today):

| # | candidate | verdict | why |
|---|---|---|---|
| 1 | janggi | **written** (`labelled/janggi.json`, `58fe9dad9711`, Q1, flagged) | Test one under unlike-search.md §1's widening, which admits what row 1 of R-S2b's walk skipped: one sanctioned array (both elephants inside), bikjang as `royal_facing: "allowed"`; also left unsaid, both edge: two passes in a row drawn, and the count of points (repetition and the no-capture clock draw). Everything else as Wikipedia and PyChess state it: the pass (not in check, so no stalemate), the palace's diagonals for general, guards, chariots, cannons and soldiers (moves under `condition.region`), the cannon that neither jumps nor takes a cannon (`screen` as a list, `captures`), the elephant blocked on two points (`legs`). Moves checked against Fairy-Stockfish's built-in janggi (pyffish 0.0.90): 40 random games, 8,094 positions, the same legal moves everywhere except where the generals face, where theirs restricts the reply to breaking it or passing and ends the game on a pass (93 positions, 28 games): the bikjang departure, as registered. Test two: Wikipedia (popular on the Korean Peninsula, professional games, a world tournament at Harbin in 2009), PyChess's place. W2 at least. |
| 2 | gorogoro plus | **written** (`labelled/gorogoro-plus.json`, `d4bf795ef1fe`, Q1, flagged) | Test one: PyChess's own definition (its `variants.ini` section: Fairy-Stockfish's `gorogoro` with a lance and a knight in each hand) loaded into pyffish and walked beside ours, 10 games, 1,560 positions, the same moves (20 positions where theirs lists a mating pawn drop as a losing move, the tolerance `verify` grants). One edge departure: a fourth repetition drawn. Test two: PyChess's place. W2. |
| 3 | clobber | **written** (`labelled/clobber.json`, `53355cbda889`, Q1, flagged *thin record*) | Test one: exact, on the published 5x6 board; the first player (the source leaves it open) is Fairy-Stockfish's, the stones on b1. `chess-variant-sandbox fairy` verifies moves and results against pyffish, whose opening is the built-in `clobber`'s. Test two: a Computer Olympiad event since 2005 and a competition at Tilburg in 2011 (Wikipedia), mostly between programs, hence the flag the record gives it. W2. |
| 4 | cannon shogi | **written** (`labelled/cannon-shogi.json`, `f9242927e9f4`, Q1, flagged) | Test one: PyChess's own definition (`variants.ini` section `cannonshogi`, read for the opening, which neither the guide page's text nor Wikipedia states) walked beside ours, 15 games, 4,372 positions, the same moves, with promoted cannons on the board in most of them. A flying cannon's short jump is a hop over an adjacent piece with `range` 2 (Betza's `pB2`); each promoted cannon is its own type, so a captured one goes to the hand as the cannon it was. One edge departure: a fourth repetition drawn. Test two: PyChess's place (Wikipedia lists it, Peter Michaelsen 1998). W2. |
| 5 | Manchu chess | **not reached** | Four were written at row 4. |

**Likeness to chess**, each appended alone to the 131 library games at `34057d4c5` (whose
mechanics are those of `74eaa6ca2`: the one file changed since, `archive/los-alamos-with-crushing-pillars.json`,
changed only its provenance, and the four R-S2b games read their recorded values again):
clobber 0.0003, janggi 0.0061, gorogoro plus 0.0093, cannon shogi 0.0126. All Q1, all below
unlike-search.md §3's cut of 0.0525. Pieces of chess (`piece:` words shared with
`rulesets/chess.json`): janggi 0, clobber 0, gorogoro plus 1 (the king), cannon shogi 3 (king,
rook, bishop).

**`tests/check_new_game.py`** took the four paths: the corpus pins are current and the 31
`new_game` checks pass. No `FROZEN` row was added (that table covers `rulesets/` and `archive/`);
each game's identity is pinned by its label's `spec_hash`.

**Found on the way, not fixed here** (outside this task's files): the Fairy-Stockfish translator
writes a hop's Betza term without its `range` (`fairy/variant.py`, the `hop` branch passes
`distance=0`), so cannon shogi's flying cannons translate as `pB` where the game is `pB2`. The
walk above used PyChess's definition, not the translation, so the spec is unaffected.
