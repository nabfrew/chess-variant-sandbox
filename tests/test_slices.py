"""A measurement cut into units, and gathered back into the same bytes.

The property under test is the one the whole design rests on: **how the work is divided
must not change an answer**. It is the same guarantee `--workers` already carries for a
tournament, said about a browser's web workers instead of a thread pool, and the check
is the same shape -- run the thing at one slice, at three and at four, and demand the
same files.

The other half is what the module is allowed to touch. `runner.py`, `play.py` and
`analysis.py` are fingerprinted, so an edit to one owes a replay of every measurement in
the corpus. `TestTheModuleIsDownstream` is what makes "a new file rather than an
argument on the old ones" checkable rather than a claim in a docstring.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import test_index_quality

from chess_variant_sandbox import measurement
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.analysis import analyse_corpus
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

REPOSITORY = Path(__file__).resolve().parents[1]
RULESETS = REPOSITORY / "rulesets"

# The two specs the acceptance measures, by file stem. Until Q558 they were picked by the
# fitted objective's verdict, one that accepted its fit and one that refused it, so both arms
# of the verdict every later unit was handed were exercised. The fitted objective is gone and
# the canonical roster is the plain ladder, so the pair is B0's original one: both cheap to
# measure, and dobutsu drops pieces, which dull never does.
MEASURED = ("dobutsu", "dull")

# The library, read from the main checkout even in a worktree, as the index audit reads it.
# Read only. Before Q398 this was the worktree's own `runs/`, which does not exist, so the
# stamp check skipped in every worker and a declared fingerprint move reached main red.
LIBRARY = test_index_quality.RUNS

# `docs/archive/plan-variety.md` §5.4's stale window, as the one stamp it leaves on disk. Every
# library `canonical.json` carried this after Q356's re-score, and still did when Q387
# (`v11` A4) became the first box to move `metrics` without a re-score. A1 and Q258 move
# `metrics` again, A5 moves `analysis` too, and C2 moves `metrics` once more, all with the
# library left stale by re-score (by re-analysis once A5 lands) until C3. So the window is
# named by the stamp that stays, not by any value the tree reads, and none of those boxes
# edits this. It clears a stored basis only as a whole: a `v10` stamp carrying both
# values. Anything else still has to match what the tree reads now.
#
# **`v11` C3 empties it** (`STALE_WINDOW = {}`) once the library is re-scored under `v11`,
# and the check then demands the tree's own reading again. The next version that opens a
# window puts its own stamp here.
#
# **Re-stamped 2026-09-25 by Q391 (`v11` C2).** B1's library replay (Q390) closed the first
# window early: it re-measured every library game under `v10` on the tree of its day, so the
# stamp on disk became that tree's reading (`metrics` "ca7c2e00d88a129e", `analysis`
# "39b9d85481a2febe") and the old stamp ("b76d7a1d71f3132e", "cff5777bf713d819") is on no
# `canonical.json` any more. C2 then moved `metrics` (`DEFAULT_WEIGHTS` to `v11`, a
# `calibration.DOMAIN` entry) and left the library stale by re-score until C3, which is a
# window again -- the one B1 left, named here. `analysis` did not move.
#
# **Emptied 2026-09-25 by Q391 (`v11` C3).** The library was re-scored under `v11` on
# `metrics` "d4eb4881f9e0aba1", and every `canonical.json` carries that stamp, so the window
# closes and the check demands the tree's own reading again. The stamp it held (`v10`,
# "ca7c2e00d88a129e", "39b9d85481a2febe") is kept in `runs/v11-rescore/canonical-v10/`.
#
# **Opened 2026-09-26 by Q471** (the immunity term as the default): the
# `decision_criticality` probe now reads the term, which moves `analysis` and `metrics`
# (`tests/test_measurement.py`'s pins say so). The same commit moves the build fingerprint,
# so the library is stale by build as well, and the schema-15 batch's one replay (Q460)
# re-plays, re-analyses and re-scores it. The window is the stamp C3 left on every
# `canonical.json`; **Q460 empties it** once the library carries the tree's own reading.
#
# **Emptied 2026-09-26 by Q460.** The batch replay re-played, re-analysed and re-scored all
# 70 library games on build `4980cb1900bd34ee`, so every `canonical.json` carries `metrics`
# "ca14fefabce8c4ac" and `analysis` "2c8e4e21096c5fd5", and the check demands the tree's own
# reading again. The replaced library, with C3's stamp, is kept in
# `runs/batch-0926-replay/replaced/` (`docs/reference/batch-0926-replay.md`).
#
# **Opened 2026-09-27 by Q497** (the scoring batch): it moves `metrics` and no number, and
# the library is left stale by re-score until Q281a6's re-score, which is owed anyway. The
# window is the stamp Q460 left on every `canonical.json`; **Q281a6 empties it** once the
# library carries the tree's own reading.
#
# **Re-opened 2026-10-02 by Q631** (the players digest, `reference/compute-economy.md` §2.3): Q281a6 and
# Q534's re-score left every `canonical.json` on the tree's reading, and Q631 moves both
# `metrics` and `analysis` (`agents/evaluation.py` joined the pass's sets; no code moved)
# along with `engine` and the new `players`, so the library owes one replay, which the
# coordinator batches. The window is the stamp the library carries now; **that replay
# empties it**.
#
# **Emptied 2026-10-02 by Q644**, that replay: all 73 library games re-played, re-analysed
# and re-scored on build `57b71b6e78003911` under `metrics` "963d8d775b1c542d", so every
# `canonical.json` carries the tree's own reading again. The replaced library, with the
# stamp this window held, is kept in `runs/replay-2026-10-02/replaced/`
# (`docs/reference/replay-2026-10-02.md`).
#
# **Opened 2026-10-04 by Q708** (F1, the race term reads who holds the goal): the probe's
# ruler `agents/evaluation.py` moves `metrics` and `analysis`, and the build fingerprint
# moves with it, so the library owes one replay, batched with Q696's. The window is the
# stamp the library carries now (Q689's re-score); **that replay empties it**.
#
# **Re-opened 2026-10-09 by Q540** (schema 17, what a capture leaves behind): the
# describer of a move (`engine/engine.py`) and the rule-mass charge (`spec/complexity.py`)
# learn the four new rules, which moves `metrics` and `analysis` with no number on a
# library game, and the core moves the build fingerprint. The library owes one replay; the
# window is the stamp it carries now (Q708's replay); **that replay empties it**.
#
# **Emptied 2026-10-09 by Q778**, that replay: all 73 library games re-played, re-analysed
# and re-scored on build `13fbbe06736fae90` under `metrics` "84159537759edd2a" and
# `analysis` "8a435f46ce02cd4d" (no game, file or score moved), and the four schema-17
# showcases measured for the first time, so every `canonical.json` carries the tree's own
# reading again. The replaced measurements, with the stamp this window held, are kept under
# each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-09.md`).
#
# **Opened 2026-10-09 by Q786** (schema 18f): the position crossing the boundary gains
# `goal_left_by` (`engine.py`, `state.py`), which moves `metrics` and `analysis` with no
# number on a library game; the core had moved the build fingerprint already (18c to 18f).
# The window is the stamp the library carries now (Q778's replay); **the batched replay,
# Q803, empties it**.
#
# **Emptied 2026-10-10 by Q803**, that replay: all 87 library games other than *jeson-mor*
# (measured on this build by Q786) re-played, re-analysed and re-scored on build
# `f3e1bb0c7586c2a4` under `metrics` "c5771af6075c5f99" and `analysis` "a8d24b3ad16d45ac"
# (no game, file or score moved), so every `canonical.json` carries the tree's own reading
# again. The replaced measurements, with the stamp this window held, are kept under each
# game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-09b.md`).
#
# **Opened 2026-10-10 by Q771** (schema 19, portals): `complexity.py`, `engine.py` and the
# compiler move `metrics` and `analysis` with no number on a library game, and the core moves
# the build fingerprint. The window is the stamp the library carries now (Q803's replay and
# Q782's re-score); **the library replay Q771 owes empties it**.
#
# **Emptied 2026-10-10 by Q806**, that replay: all 90 library games re-played, re-analysed
# and re-scored on build `1f812b55db2dba87` under `metrics` "5da33cbfe349e4b7" and
# `analysis` "a0ed8e3487536646" (no game, file or score moved), so every `canonical.json`
# carries the tree's own reading again. The replaced measurements, with the stamp this window
# held, are kept under each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-10.md`).
#
# **Opened 2026-10-10 by Q807** (schema 20, no straight back): `complexity.py`, `distance.py`,
# the boundary's `portal_bar` (`engine.py`, `state.py`) and the compiler move `metrics` and
# `analysis` with no number on a library game, and the core moves the build fingerprint. The
# window is the stamp the library carries now (Q806's replay); **the library replay Q807
# owes empties it**.
#
# **Emptied 2026-10-10 by Q810**, that replay: all 97 library games other than *ferry no
# straight back* (measured on this build by Q807) re-played, re-analysed and re-scored on build
# `49565ddb722a1195` under `metrics` "c33efb470398b44f" and `analysis` "bc59672c7e718716" (no
# game, file or score moved), so every `canonical.json` carries the tree's own reading again.
# The replaced measurements, with the stamp this window held, are kept under each game's
# `runs/<hash>/previous/` (`docs/reference/replay-2026-10-10b.md`).
#
# **Opened 2026-10-11 by Q843a** (schema 21, portal maps and filters): `complexity.py` charges
# a map of ends and a filter (no library spec writes either, so no `rule_mass` moves), which
# moves `metrics` with no number on a library game; `analysis` is unmoved; the core moves the
# build fingerprint. The window is the stamp the library carries now (Q810's replay); **the
# library replay Q843a owes, batched with Q843c's, empties it**.
#
# **Widened 2026-10-11 by Q843c** (schema 22, fold portals): the boundary's `folds` and the
# fold move text (`engine.py`, `state.py`, `moves.py`) move `analysis` as well as `metrics`,
# with no number on a library game, and the core moves the build fingerprint again. The
# window now holds the whole stamp the library carries (Q810's replay); **the one replay for
# schemas 21 and 22 (Q843r) empties it**.
#
# **Emptied 2026-10-11 by Q843r**, that replay: all 98 library games re-played, re-analysed and
# re-scored on build `234d2bf85141905e` under `metrics` "cfffc0d7c5f19c35" and `analysis`
# "84d3505a6a0baa2c" (no game, score or verdict moved), so every `canonical.json` carries the
# tree's own reading again. The replaced measurements, with the stamp this window held, are
# kept under each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-11.md`).
# (Q846 then moved both, and Q847 re-analysed and re-scored the library under them, no replay.)
#
# **Opened 2026-10-13 by Q867b** (schema 24, folds revised): the `complexity.py` comment and
# the move text through a fold (`engine.py`) move `metrics` and `analysis` with no number on a
# library game, and the core moves the build fingerprint. The window is the stamp the library
# carries now (Q847's); **the library replay Q867b owes (Q867r) empties it**.
#
# **Opened 2026-10-15 by Q918a** (schema 26, simultaneous moves): the boundary's round state
# (`state.py`, `engine.py`) and the rule's cost (`complexity.py`) move `metrics` and
# `analysis` with no number on a library game, and the core moves the build fingerprint. The
# window is the stamp the library carries now (Q926's); **the library replay Q918a owes
# empties it**.
#
# **Emptied 2026-10-15 by Q929**, that replay: the library re-played, re-analysed and
# re-scored on core `273221deaf99f3c9` under `metrics` "002cc354dbee7c2d" and `analysis`
# "7dfe04cc79c30b1f" (nothing moved), so every library `canonical.json` carries that stamp.
#
# **Opened 2026-10-15 by Q918d** (the analysis pass and score read games in rounds): the
# round replay, probe and blunders (`tournament/analysis.py`) and the round readings
# (`metrics/components.py`) move `analysis` to "2a75bbd0f846bec7" and `metrics` to
# "d85bf7e60288beab" (`FROZEN_ANALYSIS`/`FROZEN_METRICS` say so); the core is unmoved. The
# window is the stamp the library carries now (Q929's); **the library re-analysis and
# re-score Q918d owes (no replay) empties it**.
#
# **Emptied 2026-10-16 by Q938**, that re-analysis and re-score: the 87 library games
# re-analysed and re-scored from their stored records on core `273221deaf99f3c9` under
# `metrics` "d85bf7e60288beab" and `analysis` "2a75bbd0f846bec7" (no game played; no score,
# verdict or file moved, only those two basis keys), so every library `canonical.json` carries
# the tree's own reading again. The replaced measurements, with the stamp this window held, are
# kept under each game's `runs/<hash>/previous/` (`docs/reference/rescore-2026-10-16.md`).
#
# **Opened 2026-10-16 by Q942a** (schema 27's chase switches): the boundary's rested squares
# (`state.py`, `engine.py`) and the switches' cost (`complexity.py`) move `metrics` and
# `analysis` with no number on a library game (none uses a switch), and the core moves the
# build fingerprint. The window is the stamp the library carries now (Q938's); **the one
# library replay owed after Q935a closes schema 27 empties it**.
#
# **Emptied 2026-10-16 by Q950**, that replay: the 91 library games re-played, re-analysed and
# re-scored on core `ee5390a5b9c9b75c` under `metrics` "186853932cd8cf63" and `analysis`
# "413c68553fd35509" (all 93,184 records equal; no score or verdict moved; only the round
# games' `round_readings.skill_gradient` text, which Q950 corrected), so every library
# `canonical.json` carries the tree's own reading again. The replaced measurements, with the
# stamp this window held, are kept under each game's `runs/<hash>/previous/`
# (`docs/reference/replay-2026-10-16b.md`).
#
# **Opened 2026-10-17 by Q942b's merge with main**: `agents/evaluation.py` (the rested-royal
# term, `FROZEN_METRICS` and `FROZEN_ANALYSIS`) moves `metrics` and `analysis` with no number on
# a library game (none declares `rest`), and the core moves the build fingerprint. The window is
# the stamp Q950's replay left the library; **the next library replay empties it**. Q935b and
# Q935d (the spy's players, and its records and analysis) move both again with no number on a
# game without a spy; the window stays as it is.
#
# **Emptied 2026-10-17 by Q952**, that replay: the 91 library games re-played, re-analysed and
# re-scored on core `5903dcaee44ac687` under `metrics` "80f0e4c40298a8de" and `analysis`
# "b1de98c45e315f78" (all 93,184 records and all 1,183 files equal; no score or verdict moved),
# so every library `canonical.json` carries the tree's own reading again. The replaced
# measurements, with the stamp this window held, are kept under each game's
# `runs/<hash>/previous/` (`docs/reference/replay-2026-10-17.md`).
#
# **Opened 2026-10-17 by Q935h** (the evaluator's spy term): `agents/evaluation.py` moves
# `metrics` and `analysis` (`FROZEN_METRICS`, `FROZEN_ANALYSIS`) with no number on a library
# game (none has a spy), and the core moves the build fingerprint. The window is the stamp
# Q952's replay left the library; **the next library replay empties it**.
#
# **Emptied 2026-10-17 by Q954**, that replay: the 97 library games (89 and 8 previews)
# re-played, re-analysed and re-scored on core `b8f5d84c871f1211` under `metrics`
# "55b5cc39b3896211" and `analysis` "031be810615f81e7". On the 95 without a spy all 97,280
# records were equal and no score or verdict moved; the two spy previews' records moved under
# the term and were re-scored (still not viable). Every library `canonical.json` carries the
# tree's own reading again; the replaced measurements, with the stamp this window held, are kept
# under each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-17b.md`).
#
# **Opened 2026-10-19 by Q943a** (schema 28, speed in rounds): `spec/complexity.py` and
# `engine/engine.py` move `metrics` (`FROZEN_METRICS`) with no number on a library game (none
# declares a speed), and the core moves the build fingerprint. The window is the stamp the
# library carries now; **the library replay owed with Q961 empties it**.
#
# **Emptied 2026-10-19 by Q966**, that replay: the 91 library games (90 and one preview)
# re-played, re-analysed and re-scored on core `9c09c4929d2fdc4a` under `metrics`
# "a10833f919319aab" and `analysis` "031be810615f81e7" (all 93,184 records equal; no score or
# verdict moved; 264 run `index-v12.json` files moved at `description_words` alone, from the
# description rewrites). Every library `canonical.json` carries the tree's own reading again;
# the replaced measurements, with the stamp this window held, are kept under each game's
# `runs/<hash>/previous/` (`docs/reference/replay-2026-10-19.md`).
#
# **Opened 2026-10-19 by Q943b** (speed: the faster-attacker term and the tallies):
# `agents/evaluation.py` and `tournament/analysis.py` move `metrics` and `analysis`
# (`FROZEN_METRICS`, `FROZEN_ANALYSIS`) with no number on a library game (none declares a
# speed), and the core moves the build fingerprint. The window is the stamp Q966's replay left
# the library; **the next library replay empties it**.
#
# **Emptied 2026-10-19 by Q968**, that replay: the 91 library games (90 and one preview)
# re-played, re-analysed and re-scored on core `2acbc7a08a2785f2` under `metrics`
# "ed8e842a885544a3" and `analysis` "8a7d2819c34b14a5" (all 93,184 records equal; all 1,183
# files byte-equal; no score or verdict moved; only the stamp moved). Every library
# `canonical.json` carries the tree's own reading again; the replaced measurements, with the
# stamp this window held, are kept under each game's `runs/<hash>/previous/`
# (`docs/reference/replay-2026-10-19b.md`).
#
# **Opened 2026-10-19 by Q558** (the fitted objective removed), **2026-10-20 by Q983a** (schema
# 29, rest as a piece-type property) and **Q986b** (a finished round at `ROUND_WIN_PIECES` mean
# pieces), **merged into one window by Q558m** (2026-10-20). Each was built and held unmerged
# against the same stamp, the one Q968's replay left the library, so together they open one
# window, not three. Q558's metrics sources (`metrics/glossary.py`'s near-best sentence,
# `tournament/heuristics.py`), Q983a's `spec/complexity.py`, `engine/state.py`,
# `engine/engine.py` and `agents/evaluation.py`, and Q986b's `agents/evaluation.py` move
# `metrics` (`FROZEN_METRICS`); Q983a and Q986b move `analysis` (`FROZEN_ANALYSIS`); the core
# moves the build fingerprint; the canonical roster's searching rungs become `minimax:nodes=N`
# (a replay of every library game); and breather and the spy preview are new hashes owed their
# measurement. **Q558r's replay empties it**, paying all three at once.
#
# **Emptied 2026-10-21 by Q558r**, that replay: the 95 library games (93 measured and the two
# new hashes) re-played, re-analysed and re-scored on core `90fc7dd1f00152c4` under `metrics`
# "c57c854e32129d3a" and `analysis` "a0aa597a07b63b94", on `minimax:nodes=N` rungs. The 71 games
# whose players did not change (66 on a refused fit, 5 on a net) moved no record, file or score;
# the 18 that lost an accepted fit and the round games moved. Every library `canonical.json`
# carries the tree's own reading again; the replaced measurements, with the stamp this window
# held, are kept under each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-20.md`).
#
# **Opened 2026-10-22 by Q999a** (schema 30, lines of sight): `spec/complexity.py` and the engine's
# Python side move `metrics` and `analysis` (`FROZEN_METRICS`, `FROZEN_ANALYSIS`), and the core
# moves the build fingerprint; no library game declares a sight, so no game is owed a new
# measurement. The window is the stamp Q558r's replay left the library; **the next library replay
# empties it**.
#
# **Emptied 2026-10-22 by R30**, that replay: the 100 library games without a sight re-played,
# re-analysed and re-scored on core `27cf83295d37ea9d` under `metrics` "7557cf49f08b4705" and
# `analysis` "59c2e597b2f08d51". No record, analysis or score moved (102,400 records equal outside
# the clock; two games' respelled names and description word counts aside); every library
# `canonical.json` carries the tree's own reading again, and the replaced measurements, with the
# stamp this window held, are kept under each game's `runs/<hash>/previous/`
# (`docs/reference/replay-2026-10-22.md`).
#
# **Opened 2026-10-22 by Q983b** (rest in the players, the tallies and the page,
# `docs/plan-rest.md` §5.4, §5.6): `tournament/analysis.py` and the engine's Python side move
# `metrics` and `analysis` (`FROZEN_METRICS`, `FROZEN_ANALYSIS`), and the core moves the build
# fingerprint (`Core.rest_turn`, the rest plane). No game's records should move: no library game
# plays a net where a piece rests, and the tallies are written only where one does. The window is
# the stamp R30's replay left the library; **the library's re-certification, re-analysis and
# re-score owed after Q983b empty it**.
#
# **Emptied 2026-10-24 by R31**, that re-certification: the 98 library games re-certified from a
# sample of 128 of each run's 1,024 games on core `4dc6bfc1c77dcfd9` (the 96 outside rest under a
# certificate; *breather* and *chess with a weary queen* replayed whole, their stored analyses
# lacking the rest tallies), then re-analysed and re-scored under `metrics` "0d041f41fc5bfba1" and
# `analysis` "c7d6042d1d057fc2". No record, score, interval or verdict moved (100,352 records equal
# outside the clock); the replaced measurements, with the stamp this window held, are kept under
# each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-23.md`).
#
# **Opened 2026-10-24 by Q1020** (a circe rebirth in a round): `engine/engine.py`'s
# `describe_move` asks the core (`Core.board_alone`) where a round move's struck piece went, so
# the engine's Python side moves `metrics` and `analysis` (`FROZEN_METRICS`, `FROZEN_ANALYSIS`)
# and the core moves the build fingerprint. No library game is circe in rounds, so no game's
# records should move. The window is the stamp R31 left the library; **the next library replay
# empties it**.
#
# **Emptied 2026-10-25 by R32**, that replay: the 99 library games re-certified from a sample of
# 128 of each run's 1,024 games on core `347121fa1bb6c149` (98 under a certificate; *opposite
# colours*, whose records still named it *opposite colours with riders*, replayed whole), then
# re-analysed and re-scored under `metrics` "5a65f47b2629c080" and `analysis` "c38b69ba6532f04b".
# No record, score, interval or verdict moved (101,376 records equal outside the clock and that
# one game's name); the replaced measurements, with the stamp this window held, are kept under
# each game's `runs/<hash>/previous/` (`docs/reference/replay-2026-10-25.md`).
#
# **Opened 2026-10-25 by T1** (turning paths): `spec/complexity.py` charges a slide whose path
# turns one special rule per kind of turn, which moves `metrics` (`FROZEN_METRICS`); `analysis`
# and the build fingerprint are unmoved. No library game writes a path, so no number should
# move. The window is the stamp R32 left the library; **the next library re-score empties it**.
#
# **Emptied 2026-10-25 by R33**, that re-score: the 99 library games re-scored from their stored
# records and analyses under `metrics` "f745667ed7f652e7" (no game played, no analysis pass; the 98
# certificates carried). No library game writes a `path` or a `strike.by`; no file but each
# `canonical.json`'s `basis.metrics` moved, and no score, interval or verdict; the replaced
# measurements are kept under each game's `runs/<hash>/previous/`
# (`docs/reference/replay-2026-10-26.md`).
#
# **Opened 2026-10-25 by N1** (wild pieces that come back): `engine/state.py` and
# `engine/engine.py` carry the respawn clocks and the phase log's kind, which moves `analysis`
# and so `metrics` (`FROZEN_ANALYSIS`, `FROZEN_METRICS`), and the core's new state moves the
# build fingerprint. No library game writes `npcs[].respawn`, so no number should move. The
# window is the stamp R33 left the library; **the next library replay empties it**.
#
# **Emptied 2026-10-25 by R34**, that replay: the 103 library games re-certified from a sample of
# 128 of each run's 1,024 games on core `76bdeae6de748873` (412 runs, none refused), then
# re-analysed and re-scored under `metrics` "4f17f5b9d98ac076" and `analysis` "32b8a10b8d1f7b6e".
# No library game writes `npcs[].respawn`; no record, analysis, score, interval or verdict moved
# (105,472 records equal outside the clock; *the horse archers*' stored description word count
# aside); the replaced measurements, with the stamp this window held, are kept under each game's
# `runs/<hash>/previous/` (`docs/reference/replay-2026-10-27.md`).
#
# **Opened 2026-10-25 by N2** (wild pieces that see): `tournament/analysis.py`'s wild column,
# `engine/state.py`'s `Effects.wild` and `engine/engine.py` move `analysis` and so `metrics`
# (`FROZEN_ANALYSIS`, `FROZEN_METRICS`), and the core's wild sight sources and shield snapshot
# move the build fingerprint. No library game gives a wild piece a sight or a shield beside wild
# pieces, so no number should move. The window is the stamp R34 left the library; **the next
# library replay empties it** (the coordinator batches it after N5).
#
# **Held open 2026-10-25 by N5** (wild pieces that turn back): `engine/state.py`'s headings and
# `engine/engine.py` move `analysis` and `metrics` again, and the core's lines and heading move
# the build fingerprint; no library game turns back. The window keeps R34's stamp; R35, the
# replay for N2 and N5 together, empties it.
#
# **Emptied 2026-10-26 by R35**, that replay: the 103 library games re-certified from a sample of
# 128 of each run's 1,024 games on core `e5a2461fe09a7fa6` (412 runs, none refused), then
# re-analysed and re-scored under `metrics` "c5a6b2b99d42a76f" and `analysis` "63ae2694b7785b01".
# No library game gives a wild piece a sight or turns back; no record, analysis, score, interval
# or verdict moved (105,472 records equal outside the clock; 1,751 of 1,751 files byte-equal); the
# replaced measurements, with the stamp this window held, are kept under each game's
# `runs/<hash>/previous/` (`docs/reference/replay-2026-10-28.md`).
#
# **Opened 2026-10-26 by N3b** (wild pieces that pounce): `tournament/analysis.py`'s wild tally
# and `engine/state.py`'s `"pounce"` kind move `analysis` and so `metrics` (`FROZEN_ANALYSIS`,
# `FROZEN_METRICS`), and the core's pounce phase and pass test move the build fingerprint. No
# library game pounces, so no number should move; the six wild games' `analysis.json` gain the
# `wild` key. The window is the stamp R35 left the library; **the replay after N3d empties it**.
# N3c (pounce on attackers) moved both again (the tally's `pounces_on_attackers`, and the core);
# the window still holds R35's stamp, which is what the library was scored under.
# N3d (split) moved both again (the tally's `copies`, the `"copy"` kind, and the core); the
# window still holds R35's stamp.
#
# **Emptied 2026-10-26 by R36**, that replay: 97 library games re-certified from a sample of 128
# of each run's 1,024 games on core `2cf4bab8b7f84052` (388 runs, none refused); the six games
# with wild pieces replayed whole (their 24 runs refused at the sample's analysis comparison, the
# pass now writing the `wild` tally), their 6,144 records equal outside `duration_ms`; all 103
# re-analysed and re-scored under `metrics` "fdac473216822a6e" and `analysis` "5baf21c0ef18e834".
# No library game pounces or splits; no record, score, interval or verdict moved, and no analysis
# but the six games' `.games[i].wild (added)`; the replaced measurements, with the stamp this
# window held, are kept under each game's `runs/<hash>/previous/`
# (`docs/reference/replay-2026-10-29.md`).
#
# **Opened 2026-10-26 by N4b** (the duck): `spec/complexity.py`'s cost for `rules.duck`, the
# analysis pass's `duck` tally and its duck-aware readers, `engine/engine.py`'s duck helpers and
# words and `engine/compile.py`'s `DuckRule` move `analysis` and so `metrics` (`FROZEN_ANALYSIS`,
# `FROZEN_METRICS`), and the core's duck half moves the build fingerprint. No library game has a
# duck, so no number should move. The window is the stamp R36 left the library; **the next
# library replay empties it** (the coordinator decides it after N4c).
#
# **Emptied 2026-10-26 by R37**, that replay: all 103 library games re-certified from a sample of
# 128 of each run's 1,024 games on core `774f4d9c5cb34e9a` (412 runs, none refused, the six NPC
# games' first certificate since R36 replayed them), then re-analysed and re-scored under
# `metrics` "9f35f8d5328f71d5" and `analysis` "087563291466a060". N4b also moved `agents_source`
# (`agents/minimax.py`), which decides nothing here: `players` reads equal on both sides on all
# 103. No library game has a duck; no record, analysis, score, interval or verdict moved (105,472
# records equal; 1,751 of 1,751 files byte-equal); the replaced measurements, with the stamp this
# window held, are kept under each game's `runs/<hash>/previous/`
# (`docs/reference/replay-2026-10-30.md`).
STALE_WINDOW: dict[str, str] = {}


@pytest.fixture(scope="module")
def dull_spec() -> dict:
    return load_spec(RULESETS / "dull.json")


@pytest.fixture(scope="module")
def played(dull_spec):
    """A small real corpus and the config that produced it, played once and reused.

    Two searching seats, so every game is one the analysis pass grades and a cut of the
    analysis phase covers the whole schedule. `TestTheCut` has the mixed roster that
    checks the pass's own filter instead.
    """
    config = TournamentConfig(
        agents=("greedy", "minimax:nodes=100"), games_per_pairing=2, seed=7
    )
    return dull_spec, config, run_tournament(dull_spec, config).records


def _without_duration(records) -> list[dict]:
    rows = []
    for record in records:
        row = record.as_dict()
        row.pop("duration_ms")
        rows.append(row)
    return rows


class TestTheCut:
    """What a phase's units cover, before anything is played."""

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8])
    def test_the_units_cover_the_schedule_once_and_in_order(self, count) -> None:
        config = TournamentConfig(agents=("random", "greedy"), games_per_pairing=2)
        units = slices.playing_units(slices.TOURNAMENT, config, count)
        covered = [index for unit in units for index in unit.games]
        assert covered == list(range(8))
        assert [unit.part for unit in units] == list(range(len(units)))
        assert {unit.parts for unit in units} == {len(units)}

    def test_no_unit_is_empty_even_when_more_slices_are_asked_for_than_there_is_work(
        self,
    ) -> None:
        config = TournamentConfig(agents=("random",), games_per_pairing=2)
        units = slices.playing_units(slices.TOURNAMENT, config, 20)
        assert len(units) == 2
        assert all(unit.games for unit in units)

    def test_a_phase_with_nothing_to_do_has_no_units(self, played) -> None:
        _spec, _config, records = played
        assert slices.blunder_units(records, 3, sample=0) == []

    def test_the_analysis_phase_covers_only_the_games_the_pass_grades(
        self, dull_spec
    ) -> None:
        """A random seat chose nothing, so its game is not one the pass reads."""
        config = TournamentConfig(
            agents=("random", "greedy"), games_per_pairing=1, seed=3
        )
        records = run_tournament(dull_spec, config).records
        covered = [
            index for unit in slices.analysis_units(records, 2) for index in unit.games
        ]
        assert covered == [
            index
            for index, record in enumerate(records)
            if "random" not in record.agents
        ]
        assert 0 < len(covered) < len(records)

    def test_the_sizes_differ_by_at_most_one(self) -> None:
        config = TournamentConfig(agents=("random", "greedy"), games_per_pairing=5)
        units = slices.playing_units(slices.TOURNAMENT, config, 3)
        sizes = sorted(len(unit.games) for unit in units)
        assert sizes[-1] - sizes[0] <= 1


class TestAUnitAsText:
    """The wire format, which is the half B3's worker and B5's page deal in."""

    def test_a_unit_round_trips(self) -> None:
        unit = slices.Unit(phase=slices.ANALYSIS, part=1, parts=4, games=(3, 4, 5))
        assert slices.Unit.from_dict(unit.as_dict()) == unit

    def test_a_phase_nothing_defines_is_refused(self) -> None:
        with pytest.raises(slices.SliceError):
            slices.Unit.from_dict({"phase": "guess", "part": 0, "parts": 1, "games": []})

    def test_a_request_names_its_games_by_schedule_index_and_carries_no_seed(
        self, played
    ) -> None:
        """A seed in a unit's name would be the sender's word for which game this was."""
        spec, config, _records = played
        unit = slices.playing_units(slices.TOURNAMENT, config, 2)[1]
        payload = json.loads(slices.request(unit, spec, config=config))
        assert set(payload) == {"unit", "spec", "roster"}
        assert payload["unit"]["games"] == list(unit.games)
        assert "seed" not in payload["unit"]
        assert payload["roster"]["seed"] == config.seed

    def test_a_reading_unit_is_sent_only_the_games_it_reads(self, played) -> None:
        spec, config, records = played
        unit = slices.analysis_units(records, 4)[-1]
        payload = json.loads(slices.request(unit, spec, records=records, config=config))
        sent = slices.parse_records(payload["records"])
        assert len(sent) == len(unit.games)
        assert [r.seed for r in sent] == [records[i].seed for i in unit.games]

    def test_a_reading_unit_with_no_records_is_refused(self, played) -> None:
        spec, config, records = played
        unit = slices.analysis_units(records, 1)[0]
        with pytest.raises(slices.SliceError):
            slices.request(unit, spec, config=config)

    def test_a_damaged_line_is_raised_on_rather_than_skipped(self) -> None:
        """`read_records` drops a truncated last line; a unit is whole or it is re-run."""
        with pytest.raises(slices.SliceError):
            slices.parse_records('{"seed":0}\n{"seed":')

    def test_a_blank_line_is_not_damage(self) -> None:
        assert slices.parse_records("\n\n") == []


class TestSlicingDoesNotChangeTheGames:
    """One slice, three and four, against the one call they are a cut of."""

    @pytest.mark.parametrize("count", [1, 3, 4])
    def test_the_tournament_gathers_to_what_one_call_played(self, played, count) -> None:
        spec, config, records = played
        gathered = []
        for unit in slices.playing_units(slices.TOURNAMENT, config, count):
            text = slices.run_request(slices.request(unit, spec, config=config))
            gathered.extend(slices.parse_records(text))
        assert _without_duration(gathered) == _without_duration(records)

    @pytest.mark.parametrize("count", [1, 3, 4])
    def test_the_analysis_pass_gathers_to_what_one_call_produced(
        self, played, count
    ) -> None:
        spec, config, records = played
        reference = analyse_corpus(spec, records, seed=config.seed, workers=1)

        analyses = []
        for unit in slices.analysis_units(records, count):
            text = slices.run_request(
                slices.request(unit, spec, records=records, config=config)
            )
            analyses.extend(slices.parse_analyses(text))
        trials = []
        for unit in slices.blunder_units(records, count, seed=config.seed):
            text = slices.run_request(
                slices.request(unit, spec, records=records, config=config)
            )
            trials.extend(slices.parse_blunders(text))

        assert slices.gather_analysis(analyses, trials).as_dict() == reference.as_dict()

    def test_the_blunder_trials_keep_the_order_the_sampler_drew_them_in(
        self, played
    ) -> None:
        """Sorting them would reorder a list `analysis.json` holds and the scorer reads.

        A sample smaller than the corpus, so the sampler actually draws rather than
        handing back every eligible game in schedule order.
        """
        spec, config, records = played
        reference = analyse_corpus(
            spec, records, blunder_sample=3, seed=config.seed, workers=1
        )
        drawn = [
            index
            for unit in slices.blunder_units(records, 2, sample=3, seed=config.seed)
            for index in unit.games
        ]
        assert len(drawn) == 3
        assert drawn != sorted(drawn)
        assert [records[index].seed for index in drawn] == [
            trial.seed for trial in reference.blunders
        ]


@pytest.fixture(scope="module")
def round_dull(dull_spec) -> dict:
    """Dull switched to rounds by the designer's own control, as a person would (Q998)."""
    from chess_variant_sandbox.web import designing

    answer = designing.take_turns(json.loads(json.dumps(dull_spec)), "rounds", {})
    assert answer["draft"] is not None, answer["errors"]
    return answer["draft"]


class TestAGameInRounds:
    """The page's measurement of a game played in rounds (Q998).

    The user switched a copy of chess to *in rounds* in the designer and pressed *measure
    it here*: the first unit failed, because the gathering half dealt every game the
    alternating roster, whose `minimax:` the core refuses on a round game. A round game is
    measured on `corpus.ROUND_AGENTS`, and the page's measurement must deal the roster the
    basis it stamps names. Dull, switched to rounds by the designer's own control, is the
    game: small enough to play a unit of here.
    """

    def test_the_page_deals_a_round_game_the_round_roster(self, round_dull) -> None:
        from chess_variant_sandbox.corpus import ROUND_AGENTS

        first = json.loads(
            slices.gather_request(json.dumps({"spec": round_dull, "state": None}))
        )
        assert first["step"] == slices.TOURNAMENT and first["units"]
        rosters = {
            tuple(json.loads(unit["request"])["roster"]["agents"]) for unit in first["units"]
        }
        assert rosters == {ROUND_AGENTS}

    def test_a_dealt_unit_of_a_round_game_plays(self, round_dull) -> None:
        """Before Q998 every unit raised the core's refusal of `minimax:` in rounds, even
        one holding only a game between two random players, so one such game is enough."""
        first = json.loads(
            slices.gather_request(json.dumps({"spec": round_dull, "state": None}))
        )
        payload = json.loads(first["units"][0]["request"])
        payload["unit"]["games"] = payload["unit"]["games"][:1]
        records = slices.parse_records(slices.run_request(json.dumps(payload)))
        assert len(records) == 1
        assert records[0].moves

    def test_an_alternating_game_is_still_dealt_the_canonical_roster(self, dull_spec) -> None:
        from chess_variant_sandbox.corpus import CANONICAL_AGENTS

        assert slices.roster_for(dull_spec).agents == CANONICAL_AGENTS


class TestTheModuleIsDownstream:
    """What lets a new module be added without staling a single stored score.

    A fingerprinted source that imported this one would read it, and a score would then
    depend on code its basis does not cover -- the same blindness
    `tests/test_measurement.py` pins for the glossary. The import arrow only points one
    way, and this is the check.
    """

    PATH = "chess_variant_sandbox/tournament/slices.py"

    def test_the_module_is_in_no_source_set(self) -> None:
        path = measurement.ROOT / self.PATH
        assert path.exists()
        for kind in ("engine", "metrics", "analysis"):
            assert path not in measurement.fingerprinted(kind), kind
        for names in (
            measurement.ENGINE_SOURCES,
            measurement.METRICS_SOURCES,
            measurement.ANALYSIS_SOURCES,
        ):
            assert path not in set(measurement._sources(*names))

    def test_no_fingerprinted_source_imports_it(self) -> None:
        offenders = []
        sources = {
            path
            for kind in ("engine", "metrics", "analysis")
            for path in measurement.fingerprinted(kind)
        }
        for path in sorted(sources):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                if any("slices" in word for word in names):
                    offenders.append(path.name)
        assert not offenders, offenders

    def test_the_fingerprints_of_the_three_sets_are_what_the_corpus_was_stamped_with(
        self,
    ) -> None:
        """The acceptance, said as an assertion a reader can run after any edit here.

        Chess's stored basis is the corpus's oldest live stamp on this machine. If a
        change under this task had reached a fingerprinted file, its `metrics` or
        `analysis` field would no longer match what the tree reports -- unless the
        library is inside a declared stale window, when the stamp is `STALE_WINDOW`'s
        and `FROZEN_METRICS` in `tests/test_measurement.py` is the tripwire instead.
        """
        stored = LIBRARY / "4c1922ffae26" / "canonical.json"
        if not stored.exists():
            pytest.skip(f"no chess measurement in {LIBRARY}")
        basis = json.loads(stored.read_text(encoding="utf-8")).get("basis") or {}
        unmatched = _unmatched(basis, STALE_WINDOW)
        assert not unmatched, {
            kind: f"stamped {basis.get(kind)}, tree reads "
            f"{measurement.source_fingerprint(kind)}, window {STALE_WINDOW}"
            for kind in unmatched
        }


def _unmatched(basis: dict, window: dict[str, str]) -> list[str]:
    """The fingerprinted sets whose stamp the tree does not read, a stale window aside.

    A stamp matches a set where it holds the tree's code reading or its byte reading.
    Both are what the tree reads *now*: the byte reading is the LF bytes of the whole
    glob, glossary and comments included, which every basis stamped before the code
    reading recorded, so it moves with any byte and names no stored value. Otherwise only
    the whole of `window` clears a stamp.
    """
    if window and all(basis.get(key) == value for key, value in window.items()):
        return []
    return [
        kind
        for kind in ("metrics", "analysis")
        if basis.get(kind)
        not in (measurement.source_fingerprint(kind), measurement.bytes_fingerprint(kind))
    ]


class TestTheStaleWindow:
    """What `STALE_WINDOW` clears, and what it still refuses, on stamps built here."""

    WINDOW = {"weights_version": "v10", "metrics": "m" * 16, "analysis": "a" * 16}

    @staticmethod
    def _current(**overrides: str) -> dict[str, str]:
        stamp = {
            "weights_version": "v10",
            "metrics": measurement.source_fingerprint("metrics"),
            "analysis": measurement.source_fingerprint("analysis"),
        }
        return {**stamp, **overrides}

    def test_the_trees_own_reading_clears_with_or_without_a_window(self) -> None:
        for window in ({}, self.WINDOW):
            assert _unmatched(self._current(), window) == []
            assert _unmatched(self._current(weights_version="v11"), window) == []

    def test_the_byte_reading_clears_a_stamp_from_before_the_code_reading(self) -> None:
        stamp = self._current(
            metrics=measurement.bytes_fingerprint("metrics"),
            analysis=measurement.bytes_fingerprint("analysis"),
        )
        assert _unmatched(stamp, {}) == []

    def test_the_windows_own_stamp_clears(self) -> None:
        assert _unmatched(dict(self.WINDOW), self.WINDOW) == []

    def test_a_stamp_that_is_neither_is_named(self) -> None:
        assert _unmatched(self._current(metrics="0" * 16), self.WINDOW) == ["metrics"]
        assert _unmatched(self._current(analysis="0" * 16), self.WINDOW) == ["analysis"]

    def test_the_window_clears_only_as_a_whole(self) -> None:
        """Its `metrics` beside some other `analysis` is a stamp nobody declared."""
        stamp = {**self.WINDOW, "analysis": "0" * 16}
        assert _unmatched(stamp, self.WINDOW) == ["metrics", "analysis"]
        mixed = self._current(metrics=self.WINDOW["metrics"])
        assert _unmatched(mixed, self.WINDOW) == ["metrics"]

    def test_the_window_belongs_to_one_weights_version(self) -> None:
        """A `v11` stamp carrying the old values was never what the plan left on disk."""
        stamp = {**self.WINDOW, "weights_version": "v11"}
        assert _unmatched(stamp, self.WINDOW) == ["metrics", "analysis"]

    def test_an_empty_window_clears_nothing_the_tree_does_not_read(self) -> None:
        """What `STALE_WINDOW = {}` at C3 leaves: the old stamp is refused again."""
        assert _unmatched(dict(self.WINDOW), {}) == ["metrics", "analysis"]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """`corpus.measure` on each of the two rulesets, run once and read three times.

    The real canonical roster, at one thread, and at the one seed the comparison reads.
    A committed ruleset is measured at `EXPECTED_LIBRARY_SEEDS` by default, and until Q437
    this played all four and compared one. Both games are cheap to measure, which is the
    only reason B0's acceptance can be a test rather than a run recorded in a document.
    """
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.spec.canonical import spec_hash

    work = tmp_path_factory.mktemp("slices-measured")
    done = {}
    for name in MEASURED:
        spec = load_spec(RULESETS / f"{name}.json")
        digest = spec_hash(spec)
        corpus.measure(
            corpus.Candidate(
                hash=digest, name=str(spec["name"]), spec=spec,
                path=RULESETS / f"{name}.json", stage="committed",
            ),
            runs_dir=work / name,
            workers=1,
            seeds=(corpus.CANONICAL_SEED,),
        )
        done[name] = (spec, work / name / f"{digest}-{corpus.CANONICAL_SEED}")
    return work, done


@pytest.mark.slow
@pytest.mark.parametrize("name", MEASURED)
@pytest.mark.parametrize("count", [1, 3, 4])
def test_a_sliced_measurement_writes_what_corpus_measure_writes(
    name, count, measured
) -> None:
    """B0's acceptance: the canonical roster, on two rulesets, at one slice, three and four.

    Every phase runs, so what is compared is a whole measurement. `games.jsonl` is compared
    with `duration_ms` dropped, because a slice is timed separately from the whole; the
    other three files are compared as bytes.
    """
    work, done = measured
    spec, whole = done[name]

    sliced = work / f"sliced-{name}-{count}"
    slices.measure_in_slices(
        spec, sliced, slices=count, config=slices.canonical_config(workers=1)
    )

    assert _without_duration(
        list(slices.parse_records((sliced / "games.jsonl").read_text(encoding="utf-8")))
    ) == _without_duration(
        list(slices.parse_records((whole / "games.jsonl").read_text(encoding="utf-8")))
    )
    for filename in ("tournament.json", "analysis.json"):
        assert (sliced / filename).read_bytes() == (whole / filename).read_bytes(), filename
    index = next(whole.glob("index-*.json"))
    assert (sliced / index.name).read_bytes() == index.read_bytes()

    # A sliced measurement is provisional: the file every reader trusts is not its to
    # write, and the basis stamp belongs where the result is re-derived.
    assert not (sliced / "canonical.json").exists()
