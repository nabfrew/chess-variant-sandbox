"""What a stored score rests on, and the ways it can stop being true.

The failure these tests exist to prevent has a shape, and it is not "a number is wrong".
It is "a number is *stale* and nothing can tell". A score is a function of the spec, the
roster, the weights, the code that played the games and the code that scored them; four
of those were recorded nowhere, so every reader defaulted to trusting whatever it found.

The permissive direction is the dangerous one throughout. A fingerprint that fails to
notice a change reports "current" forever and is invisible; one that notices a change
that did not happen costs a re-measure and announces itself. So the tests below lean on
the first: that the globs match real files, that each field is actually compared, and
that a replay is never quietly downgraded to a re-score.
"""

from __future__ import annotations

import ast
import json
import os
import random
from pathlib import Path

import pytest

from chess_variant_sandbox import measurement
from chess_variant_sandbox.measurement import Basis, Staleness


def _basis(**overrides) -> Basis:
    base = {
        "ruleset_hash": "abc123",
        "weights_version": "v8",
        "agents": ("random", "greedy"),
        "games_per_pairing": 4,
        "seeds": (0,),
        "engine": "engine-fingerprint",
        "metrics": "metrics-fingerprint",
        "tables": "tables-fingerprint",
        "analysis": "analysis-fingerprint",
    }
    return Basis(**{**base, **overrides})


class TestTheFingerprintsPointAtSomething:
    """The silent failure mode, checked directly.

    A glob with a typo in it hashes the empty set. Every measurement then agrees with
    every other measurement forever, `--check` reports a clean corpus, and nothing ever
    says otherwise -- the fingerprint would be *load-bearing* and *inert* at the same
    time. Cheap to check, and impossible to notice any other way.
    """

    def test_every_source_pattern_matches_at_least_one_file(self) -> None:
        for pattern in (
            measurement.ENGINE_SOURCES
            + measurement.METRICS_SOURCES
            + measurement.ANALYSIS_SOURCES
        ):
            assert list(measurement.ROOT.glob(pattern)), pattern

    def test_the_analysis_pass_is_inside_the_metrics_set_and_outside_the_engine(
        self,
    ) -> None:
        """The one deliberate overlap, and the reason it must stay.

        Every basis stamped before the pass had its own fingerprint recorded `metrics`
        over a set that included `analysis.py`. That is what lets such a basis still be
        read: where its `metrics` matches, the pass has not moved. Take the file out of
        `METRICS_SOURCES` and the current `metrics` fingerprint moves for every stored
        measurement at once, reporting the whole corpus stale for an edit nobody made.
        And it must not be in the engine set, or an edit to the pass would charge a
        replay for games that are still valid.
        """
        analysis = {p for pattern in measurement.ANALYSIS_SOURCES
                    for p in measurement.ROOT.glob(pattern)}
        metrics = {p for pattern in measurement.METRICS_SOURCES
                   for p in measurement.ROOT.glob(pattern)}
        engine = {p for pattern in measurement.ENGINE_SOURCES
                  for p in measurement.ROOT.glob(pattern)}
        assert analysis
        assert analysis <= metrics
        assert not (analysis & engine)


# What `source_fingerprint("metrics")` read when the seat check was restated as a
# tolerated share over a game's runs pooled, 2026-09-19 (Q356), and the library was
# re-scored under it. It was "9089683b62ca9225" from 2026-09-18 until then. Every run on
# disk is scored under the current value, so this is the tripwire that says whether a
# later change reached into `metrics/` without meaning to.
#
# **A real change to the metrics is allowed to move it.** When it does, the repair is to
# re-score every run, put the new value here with the date, and say in the commit which
# change moved it. What is not allowed is for it to move quietly.
#
# **Moved 2026-09-24 by Q387 (`v11` A4)**: `calibration.calibrate` refuses null stability
# over nulls with no base (Q71). It was "b76d7a1d71f3132e" from Q356 until then. This one
# is *not* followed by a re-score. `docs/archive/plan-variety.md` §5.4's stale window keeps the
# library stale by re-score from the first `v11` box that moves `metrics` until C3
# re-scores it under `v11`, so until C3 the runs on disk carry the old value. The tripwire
# still catches any move nobody declared.
#
# **Moved again 2026-09-24 by Q384 (`v11` A1)**: the variety components in
# `metrics/components.py` and the reader in `metrics/index.py` that names the pass. It was
# "a3c35ae59e9c518c" after Q387 alone. The same stale window holds, and C3 pays for both.
#
# **Moved again 2026-09-24 by Q388 (`v11` A5, repair E)**: `analysis.BLUNDER_SAMPLE` raised
# from 24 to 48 (`docs/reference/ladder-statistic.md` §4 and §6). It was "b51975342a34cd7d"
# after Q384. `analysis.py` sits in both `ANALYSIS_SOURCES` and `METRICS_SOURCES`, so this
# is the first `v11` box to move the library's repair from re-score to re-analyse; the
# same stale window still holds until C3.
#
# **Moved again 2026-09-24 by Q383 (the package rename, `docs/archive/plan-rename.md` stage 4)**:
# the package's name is in the code of every file that imports another, so the rename moves
# the code reading without changing what anything computes. It was "900ddeecea463d7b" after
# Q388. The user chose to wait for a replay owed for another reason (plan §8 question 1),
# which re-stamps the library under this value.
#
# **Moved 2026-09-25 by Q391 (`v11` C2)**: `DEFAULT_WEIGHTS` is `v11` (`metrics/scoring.py`)
# and `calibration.DOMAIN` gains `position_divergence`. It was "ca7c2e00d88a129e" from the
# rename until then, and B1's replay (Q390) left every library `canonical.json` stamped with
# it under `v10`. Not followed by a re-score here: C3 re-scores the library under `v11`, and
# until then it reads stale by re-score, inside the window `tests/test_slices.py`'s
# `STALE_WINDOW` names.
#
# **Moved 2026-09-25 by Q391 (`v11` C3), and followed by the re-score**:
# `IndexResult.summary` (`metrics/index.py`) says the run count itself (*as the one run*),
# which Q416 had added from `cli.py` so as not to restale the library for a sentence, and
# prints *no score* where a placeholder 0.0 stands for no total. Neither changes a number.
# It was "6717e191f41faf89" after C2. C3 then re-scored all 284 library runs under `v11`
# on this value, so the library carries it and the stale window closes.
#
# **Moved 2026-09-26 by Q471 (the immunity term as the default, F2 decision C)**: the
# `decision_criticality` probe (`analysis._score_roots`, `Engine.root_scores`) now reads
# the immunity term, which moves the pass on the six specs where the term can read
# non-zero (`docs/reference/immunity-term.md` section 6). It was "d4eb4881f9e0aba1" after
# C3. Not followed by a re-score here: the same commit moves the build fingerprint, and
# the batch's one replay (Q460) re-plays, re-analyses and re-scores the library under it.
#
# **Moved 2026-09-27 by Q497 (the scoring batch, `docs/reference/perf-survey-2026-09.md`
# R2 and R3's invisible form)**: the bootstrap reads each game's own numbers once
# (`components.GameReadings`), `_budget` and `_is_weak` are cached, and the Bradley-Terry
# fit works on lists. No number moves: `tests/test_scoring_batch.py` holds every score and
# rating to the old path's to the last bit. It was "ca14fefabce8c4ac" after Q471/Q460. Not
# followed by a re-score here: the library is re-scored by Q281a6, which is owed anyway,
# and until then it reads stale by re-score inside `tests/test_slices.py`'s `STALE_WINDOW`.
#
# **Moved 2026-09-28 by Q281a4 (schema 16, the native hexagon)**: `complexity._RULE_ACTIVE`
# charges `wrap` for a hexagon's pairs of sides too (`complexity._WRAP_KEYS`). Every
# committed spec's `rule_mass` is unchanged (`tests/test_complexity.py`); only the source
# moved. It was "276e5e811877087b" after Q497. The re-score is Q281a6, local.
#
# **Moved 2026-09-29 by Q534 (`seat-effect-size.md` §15's distinct-transcript floor)**: a
# run's seat bands store each decided game's transcript digest and a `distinct` count, and
# the pooled lean's floor reads distinct games rather than decided ones. No refusal, total,
# component or interval moves; a printed lean can disappear. It was "c0874c9cb4baaefe"
# after Q281a4. The re-score is local (no `runs/` in the cloud), batched with Q281a6's.
#
# **Moved 2026-10-02 by Q631 (`reference/compute-economy.md` §2.3, the players digest)**: `agents/*.py`
# left the `engine` set, so the probe's ruler, `agents/evaluation.py`, joined the analysis
# pass's sets, this one included. No code moved. It was "4101e53269e06084" after Q534. The
# same change moves `engine` and adds `players`, so the library owes one replay, which
# re-analyses and re-scores it under this value; the coordinator batches it.
#
# **Moved 2026-10-02 by Q535 and Q625**: the Bradley-Terry fit stops when the
# renormalised log-strengths converge (`perf-survey-2026-09.md` §5.8: no stored rating
# rounds differently), and the seat lean compares the exact split with two in three, so a
# split of exactly two in three prints no lean (`atomic`, `intransitive`). It was
# "4094776cca045468" after Q631. Both ride Q631's owed replay.
#
# **Moved 2026-10-03 by Q689 (EV4, `v12`: the scale's ends by source)**: `DEFAULT_WEIGHTS`
# is `v12`, `parse_weights` reads and checks each end's `source`, and
# `calibration.check_ends` is `anchor`'s report. It was "963d8d775b1c542d" after Q535 and
# Q625. The same task re-scored the library under it (`docs/reference/v12-calibration.md`).
#
# **Moved 2026-10-04 by Q708 (F1, `intransitive-quiet-endgame.md` section d)**: the race
# term reads who holds the goal (`agents/evaluation.py`'s seal, the probe's ruler and the
# oracle of `eval.rs`), a real change to the pass on the two specs with a priced race.
# It was "e75a741b1e0d5900" after Q689. The same commit moves the build fingerprint, so
# the library replay batched with Q696 re-analyses and re-scores under this value.
#
# **Moved 2026-10-09 by Q540 (schema 17, what a capture leaves behind)**: `complexity.py`
# charges the four new rules (no library spec's `rule_mass` moves) and `engine.py` names
# what a rifle, circe or Andernach capture left behind, and stops printing a promotion a
# contact overrides (Q539). It was "469490dd3142a17d" after
# Q708. The core moves the build fingerprint in the same change, so the library owes one
# replay, which re-analyses and re-scores under this value.
#
# **Moved 2026-10-09 by Q786 (schema 18f, win by leaving a square)**: the position crossing
# the boundary gains `goal_left_by` (`engine.py`'s `_pack`/`_unpack`, `state.py`), which
# moves no number on a library game (the field is -1 wherever the rule is absent). It was
# "84159537759edd2a" after Q540. The core moved the build fingerprint already (18c to 18f),
# and the library's batched replay (Q803) re-analyses and re-scores under this value.
#
# **Moved 2026-10-10 by Q782 (schema 18b, hex symmetries)**: `complexity.expand_symmetry`
# turns `rotate6` and `dihedral12`, which no library spec but the two new showcases writes,
# so no stored number moves. It was "c5771af6075c5f99" after Q786. The core is unchanged;
# the same task re-scores the library under this value (no replay).
#
# **Moved 2026-10-10 by Q771 (schema 19, portals)**: `complexity.py` charges `rules.portals`
# (no library spec's `rule_mass` moves), `engine.py` writes a transit's text and a portal's
# own move, and the compiler adds the portal tables (every library game's `tables`
# fingerprint unmoved). It was "8e1bf99a3a129d3d" after Q782. The core moves the build
# fingerprint in the same change, so the library owes one replay, which re-analyses and
# re-scores under this value (`STALE_WINDOW` holds the library's stamp until it lands).
#
# **Moved 2026-10-10 by Q807 (schema 20, no straight back)**: `complexity.py` charges
# `rules.portals[].no_straight_back` (no library spec states it), `distance.py` reads it,
# `engine.py` and `state.py` carry the bar across the boundary, and the compiler adds its
# tables (sent empty where no pair states it, so every library game's `tables` fingerprint
# is unmoved). It was "5da33cbfe349e4b7" after Q771. The core moves the build fingerprint
# in the same change, so the library owes one replay, which re-analyses and re-scores
# under this value (`STALE_WINDOW` holds the library's stamp until it lands).
#
# **Moved 2026-10-11 by Q843a (schema 21, portal maps and filters)**: `complexity.py`
# charges a map of ends and a filter (no library spec writes either, so no `rule_mass`
# moves). It was "c33efb470398b44f" after Q807. The core moves the build fingerprint in the
# same change, so the library owes one replay, batched with Q843c's, which re-analyses and
# re-scores under this value (`STALE_WINDOW` holds the library's stamp until it lands).
#
# **Moved 2026-10-11 by Q843c (schema 22, fold portals)**: `complexity.py` charges a fold
# group, and the boundary's `folds` crosses in `engine.py` and `state.py` (with the fold move
# text and `MOVE_FOLD` in `moves.py`); no library spec writes a fold, so no `rule_mass` or
# move text of a library game moves. It was "e6aaa52a77b6097f" after Q843a. The core moves the
# build fingerprint in the same change, so the library owes one replay, batched with
# Q843a's (Q843r), which re-analyses and re-scores under this value (`STALE_WINDOW` holds
# the library's stamp until it lands).
#
# **Moved 2026-10-11 by Q846 (the evaluator oracle on momentum portals)**:
# `agents/evaluation.py::Evaluator.immunity` skips a portal end's third-owner byte, as
# `eval.rs` does; before, it raised IndexError on one. A board byte in the third owner's
# block exists only beside portals or wild pieces, and the term is undefined beside wild
# pieces, so no number of a game the old oracle could evaluate moves, and the core, which
# scores the library, is untouched (no build fingerprint move). It was "cfffc0d7c5f19c35"
# after Q843c. Q847 re-analysed and re-scored the library under it (no replay), 2026-10-11:
# every `canonical.json` equal outside `basis`, no score, interval or verdict moved.
#
# **Moved 2026-10-13 by Q867b (schema 24, folds revised)**: the comment above
# `complexity._PORTAL_FOLD` says the revised rule (its weight unchanged), and `engine.py`'s
# move text writes a move through a fold the new way; no library spec writes a fold, so no
# `rule_mass` or move text of a library game moves. It was "4f58577892896e3d" after Q846. The
# core moves the build fingerprint in the same change, so the library owes one replay (Q867r),
# which re-analyses and re-scores under this value (`STALE_WINDOW` holds the library's stamp
# until it lands).
#
# **Moved 2026-10-13 by Q873 (the analysis pass reads a fold's own move)**:
# `tournament/analysis.py` counts a `MOVE_FOLD` under its end's type, credits it no instance
# and no capture, and walks on, where it stopped at the empty origin. No library spec writes a
# fold, so no number of a library game moves. It was "66014a3376e6c295" after Q867b; the
# replay Q867b owes (Q867r) re-analyses and re-scores under this value.
#
# **Moved 2026-10-13 by Q875 (rotations take nothing; a portal is not part of the army)**:
# `tournament/analysis.py` reads a rotation (and a castling onto its own rook) as no capture,
# and leaves a momentum end out of the starting instances; `metrics/components.py` counts
# only a player's declared types in `piece_utilisation`, where a portal game's longer count
# vector had kept every index. Library numbers move at the replay: `capture_availability`
# (unweighted) on the four rotation games (chess with turning pawns, backward drop shogi,
# about face, weathervane), and `piece_utilisation` and `instance_participation` on the
# seven portal games (wormhole, cylinder wormhole, shut wormhole, carousel, divide, ferry,
# steady ferry). It was "5182a7e993216dee" after Q873; Q867r's replay re-analyses and
# re-scores under this value.
#
# **Moved 2026-10-13 by Q877 (a portal is no idle piece)**: `metrics/components.py` counts
# `idle_piece_types` over the held types, as `piece_utilisation` does, where it took every
# entry of a portal game's longer count vector (and any wild-only type) for a piece. A
# diagnostic only, shown to the proposer; no component, score or weight reads it. It was
# "f8a2637a683e942a" after Q875; Q867r's replay re-analyses and re-scores under this value.
#
# **Moved 2026-10-15 by Q926 (the validator refuses what it should)**: `spec/reachability.py`
# reads a piece id through `_in`/`_str_set`, so a list where an id belongs is skipped by its
# reachability walk instead of raising `TypeError`. No valid spec reads differently, so no
# score changes; every run reads stale until the refresher re-scores it under this value. It
# was "9547343221dc0dc2" after Q877.
#
# **Moved 2026-10-15 by Q918a (schema 26, simultaneous moves)**: `engine/state.py` carries the
# initiative, the void run and the bars, `engine/engine.py` packs them and plays a round, and
# `spec/complexity.py` costs the rule and a piece's `beats`. No alternating game reads
# differently; every run reads stale until the refresher re-scores it under this value (the
# core build moved too, which is the refresher's replay). It was "f8af43c0447b76b4" after Q926.
#
# **Moved 2026-10-15 by Q918b (the computer players for rounds)**: `agents/evaluation.py` gains
# `Evaluator.evaluate_round` (mobility mine minus theirs), with `evaluate` sharing its body in
# the same arithmetic, and `engine/engine.py` gains `select_round`. No alternating game reads
# differently; the library's replay owed for Q918a (batched as Q929) re-scores under this
# value. It was "db5723c0687589a6" after Q918a.
#
# **Moved 2026-10-15 by Q918d (records, metrics and the analysis pass for rounds)**:
# `tournament/analysis.py` reads a game played in rounds (see `FROZEN_ANALYSIS`), and
# `metrics/components.py` carries `ROUND_READINGS` and the rounds' tallies in a round game's
# diagnostics. No alternating game reads differently: its analysis is the same bytes and its
# components gain nothing. Every run reads stale by re-analyse until the refresher re-analyses
# and re-scores the library under this value. It was "002cc354dbee7c2d" after Q918b.
#
# **Moved 2026-10-16 by Q942a (schema 27's chase switches)**: `engine/state.py` and
# `engine/engine.py` carry the rested squares across the boundary (`GameState.rest`), and
# `spec/complexity.py` charges each switch as one exception. No library game uses a switch, so
# no number should move; the replay owed after Q935a closes schema 27 re-scores under this
# value. It was "d85bf7e60288beab" after Q918d.
#
# **Moved 2026-10-16 by Q935a (the spy, closing schema 27)**: `engine/engine.py` asks the core
# a spy game's phase (`Engine.phase`, `is_round`, `steps`), `engine/state.py` and
# `engine/__init__.py` carry `SpyPhase`, and `spec/complexity.py` charges a spy as a rule. No
# library game has a spy, so no number should move; the one replay owed after Q935a re-scores
# under this value. It was "c93d3b4c84cad5ab" after Q942a.
#
# **Moved 2026-10-16 by Q942b (the rested-royal term)**: `agents/evaluation.py` prices a rested
# royal a standing enemy piece reaches under `rules.simultaneous.rest` (`RESTED_ROYAL`,
# `rested_royal_falls`), read by `evaluate_round` only while a piece rests. No library game
# declares `rest`, so no number should move; the replay batched after Q950 re-scores under
# this value. It was "1f42f04756a50c60" after Q935a.
#
# **Moved 2026-10-16 by Q950 (before its replay)**: `metrics/components.py::ROUND_READINGS`
# describes `skill_gradient` as the round roster Q944a measured, `matrix:nodes=1600 to
# 102400` (`docs/reference/round-roster-budgets.md` §6.1), not 100 to 6400. The string is
# written into each round game's run `index-v12.json` diagnostics and nowhere else; no number
# moves. The replay Q950 launches re-scores under this value. It was "1f42f04756a50c60"
# after Q935a.
# The merged value covers both Q942b's evaluator change and Q950's `ROUND_READINGS` fix.
#
# **Moved 2026-10-17 by Q935b (the spy's players)**: `agents/evaluation.py` prices a spy type
# outside the prey share (`_live_values`, read only where `spec.spy` is set), and
# `engine/engine.py::select_round` passes an announced move. No library game has a spy, so no
# number should move; the one replay owed after the bump re-scores under this value. It was
# "37b18b7a885114d9" after Q942b and Q950.
#
# **Moved 2026-10-17 by Q935d (records, metrics and the analysis pass for a spy game)**:
# `tournament/analysis.py` walks a spy game a step at a time by its phase (see
# `FROZEN_ANALYSIS`), and `metrics/components.py` carries `SPY_READINGS` and the spy's
# summary in a spy game's diagnostics. No game without a spy reads differently: a seeded
# chess and Gardner-in-rounds tournament, their analyses, and `measure` on gardner and
# Gardner in rounds are the same bytes and numbers as before (`runs/Q935d/inert/`). The
# replay owed after the bump re-scores under this value. It was "e0aadf993349b4be" after
# Q935b.
#
# **Moved 2026-10-17 by Q935h (the evaluator's spy term)**: `agents/evaluation.py` prices
# holding the only spy at `SPY_PIECES` mean non-royal pieces (`Evaluator.spy_edge`, read only
# where `spec.spy` is set). No game without a spy reads differently (`test_spy_players.py::
# _INERT`: a seeded chess and Gardner-in-rounds tournament the same bytes as main's). It was
# "80f0e4c40298a8de" after Q935d and Q952's replay; the next library replay re-scores under it.
#
# **Moved 2026-10-17 by Q947e (dead code deleted)**: `engine/state.py` loses `PLAYER_ZERO` and
# `PLAYER_ONE`, and `engine/__init__.py` their re-exports; nothing read either, so no number
# moves. It was "d92ab76eeb9dd2a0" after Q935h; the next library replay re-scores under it.
#
# **Moved 2026-10-17 by Q953a (the seat counts each different game once; every score
# carries its repeats)**: `metrics/seat.py` and `components._seat_split` test each distinct
# decided transcript once, and `metrics/repeats.py` is new (the share of different games and
# the distinct re-score beside a score, `docs/reference/repeated-games.md` §6). No total,
# component or interval moves; the seat's verdicts move on §4.3's games. It was
# "55b5cc39b3896211" after Q947e; Q953a's library re-score is under it.
#
# **Moved 2026-10-19 by Q943a (schema 28, speed in rounds)**: `spec/complexity.py` charges
# speeds (`_speed_extras`) and `engine/engine.py` documents the ordered round's events. No
# library game declares a speed, so no number should move: every committed and archived spec
# keeps its complexity, tables and prose byte for byte (main against the branch). It was
# "983645dfe3d2a18f" after Q953a; the library replay owed with Q961 re-scores under it.
#
# **Moved 2026-10-19 by Q943b (speed: the faster-attacker term and the tallies)**:
# `agents/evaluation.py` prices a royal a strictly faster piece takes next round, and
# `tournament/analysis.py` counts `SPEED_COUNTS`, both only where a piece declares a speed.
# No library game declares one, so no number should move (the round players' tournaments
# and two analysis files are pinned equal in `tests/test_speed_players.py`). It was
# "a10833f919319aab" after Q943a and Q966's replay; the next library replay re-scores under it.
#
# **Moved 2026-10-19 by Q558 (the fitted objective removed)**: `tournament/heuristics.py`
# loses `measure_heuristics`'s `baseline_fit` and `HeuristicReport.baseline`, which only the
# removed `heuristics --against-fit` read, and `metrics/glossary.py`'s near-best sentence
# says "learned net" for "fitted objective". No score, component or interval reads either,
# so no number moves. It was "ed8e842a885544a3" after Q943b and Q968's replay; on
# `claude/Q558`, held unmerged, it read "5e841a83a9a1d10f".
#
# **Moved 2026-10-20 by Q983a (schema 29, rest as a piece-type property)**: `spec/complexity.py`
# charges `pieces[].rest` (`_rest_extras`, the old switch's 1.0 for a uniform rest),
# `engine/state.py` and `engine/engine.py` carry the resting list as a field of every position,
# and `agents/evaluation.py` reads the rested-royal term over that list. The library's one
# rest game, breather, was rewritten to a new hash (the same game move for move) and the spy
# preview to the nearest new game; no other game declares a rest, so no other number should
# move. It was "ed8e842a885544a3" after Q943b and Q968's replay; on `claude/Q983a` it read
# "dbb60c88575f6cc6".
#
# **Moved 2026-10-20 by Q986b (a finished round at `ROUND_WIN_PIECES` mean pieces)**:
# `agents/evaluation.py` gains `ROUND_WIN_PIECES` (100) and `round_win`, and prices a rested
# or caught royal at `RESTED_ROYAL_PIECES` (50) mean pieces, not half of `WIN_SCORE`; the term
# is read only where a piece rests or declares a speed, so only round games move, and those
# are replayed anyway (a new round player). On `claude/Q986b`, built on Q983a, it read
# "1bae673288ef0490".
#
# **Merged 2026-10-20 by Q558m**: the three branches were built against the same stamp and held
# unmerged; merged, the metrics sources are Q558's and Q983a's and Q986b's at once, which is
# neither branch's value. No number moves for Q558's part; Q983a's and Q986b's are as above.
# The library replay Q558r re-scores under it, paying all three moves once.
#
# **Moved 2026-10-22 by Q999a (schema 30, lines of sight)**: `spec/complexity.py` charges
# `pieces[].sight` (`_sight_extras`), and `engine/engine.py`, `engine/state.py` and
# `engine/__init__.py` carry the `Effects` record and `Engine.effects`/`sight_of`. No library game
# declares a sight, so no number should move. It was "c57c854e32129d3a" after Q558r's replay; the
# next library replay re-scores under it.
#
# **Moved 2026-10-22 by Q999c**: `tournament/analysis.py` records the sight tallies
# (`SIGHT_SERIES`, `docs/archive/plan-passive.md` §4.8), only for a game with a line of sight; no library
# game has one, so no number moves. It was "4888feeaf7240d65" after Q999a; the replay R30 owed
# for Q999a re-scores under it.
#
# **Moved 2026-10-22 by Q983b (rest in the players, the tallies and the page)**:
# `tournament/analysis.py` records the rest tallies (`REST_SERIES`, `docs/plan-rest.md` §5.4,
# §7), only for a game whose pieces rest, and `engine/engine.py`, `engine/state.py` and
# `engine/__init__.py` carry `Engine.rest_turn` and its `RestTurn` record. The library's rest
# game is breather (and the previews that rest); its components read the same games, so no
# number should move. It was "7557cf49f08b4705" after Q999c; the library's re-score owed after
# Q983b re-scores under it.
#
# **Moved 2026-10-23 by Q1007c (the command effect)**: `spec/complexity.py` charges half a rule
# for a command's `captures_own` (`_SIGHT_TURNCOAT`), and `engine/engine.py`, `moves.py` and
# `state.py` carry `MOVE_COMMANDED`, the `~` of `describe_move` and `Effects.commanded`. No
# library game declares a command, so no number should move. It was "a4f5513c93f74731" after
# Q983b; R31's replay re-scores under it.
#
# **Moved 2026-10-23 by Q1007b (spelling 31, portal moves on the group)**: `spec/complexity.py`
# counts each distinct `rules.portals[].moves` list as a piece type with its moves
# (`_portal_movement`), the stand-in piece every portal game declared before the re-spelling,
# so every portal game's counts and rule mass are what they were (`docs/plan-schema-31.md`
# §4.3). No number should move. It was "6a665ced2b90fe0a" after Q1007c; R31's replay
# re-scores under it.
#
# **Moved 2026-10-23 by Q1008d**: `metrics/mutants.py`'s `idle_wanderer` no longer raises
# `schema_version`, which no gate and no name reads since Q1008c
# (`docs/reference/schema-reboot-assessment.md` §10). Every library file is stamped 31, so the
# line wrote nothing anywhere, and no number can move. It was "f9bd681ef9fbd903" after Q1007b;
# R31's replay re-scores under it (`measure --check` reads the same 104 replays as before).
#
# **Moved 2026-10-23 by Q1007d (command in the tallies)**: `tournament/analysis.py` records
# `COMMAND_SERIES` beside `SIGHT_SERIES`, only for a game whose sight commands, and
# `_SIGHT_BITS` knows the command bits 8 and 16 (`docs/plan-schema-31.md` §3.9). No library
# game declares a command, and a sight game without one writes the keys it did, so no number
# should move. It was "8e8d8c6ea96b7471" after Q1008d; the library replay after the detached
# training campaign re-scores under it.
FROZEN_METRICS = "0d041f41fc5bfba1"


class TestTheMetricsFingerprintIsWhereTheLibraryLeftIt:
    """A score's basis is only as good as its most recent silent edit.

    `metrics` is the fingerprint every run on disk carries, and a change to it stales all
    284 of them. `docs/archive/plan-four-seeds.md` §9.2 puts that move in one stage and
    forbids it in the three after, and this is where the forbidding happens -- an adoption
    path, a CLI flag or a page is not allowed to reach into `metrics/` on its way past.
    """

    def test_it_reads_what_the_library_was_scored_under(self) -> None:
        assert measurement.source_fingerprint("metrics") == FROZEN_METRICS, (
            "the metrics fingerprint moved; every run in runs/ now reads stale. If that "
            "was the point, re-score them and put the new value in FROZEN_METRICS with "
            "the date and the change that moved it"
        )

    def test_the_new_adoption_path_is_in_no_fingerprint_set(self) -> None:
        """`tournament/adopt.py` decides no game and computes no component.

        It reads a schedule, plays through the runner and copies three files. Putting it
        in a set would charge every measurement in `runs/` for an edit to a file no score
        depends on.
        """
        every = (
            measurement.ENGINE_SOURCES
            + measurement.METRICS_SOURCES
            + measurement.ANALYSIS_SOURCES
        )
        found = {path for pattern in every for path in measurement.ROOT.glob(pattern)}
        assert measurement.ROOT / "chess_variant_sandbox" / "tournament" / "adopt.py" not in found

    def test_the_two_sides_are_not_the_same_files(self) -> None:
        """Engine and metrics have to be separable, or the cheap repair is unreachable.

        If a metrics-only change registered as an engine change, every weights bump
        would replay the whole corpus, and `refresh` would be `measure` with extra
        steps.
        """
        engine = {p for pattern in measurement.ENGINE_SOURCES
                  for p in measurement.ROOT.glob(pattern)}
        metrics = {p for pattern in measurement.METRICS_SOURCES
                   for p in measurement.ROOT.glob(pattern)}
        assert engine and metrics
        assert not (engine & metrics)

    def test_the_complexity_count_is_in_the_metrics_set(self) -> None:
        """`rule_yield` divides by `rule_mass`, and `spec/complexity.py` computes it.

        Outside the set, a change to what a rule is charged moves a stored score while
        its basis reads current.
        """
        complexity = measurement.ROOT / "chess_variant_sandbox" / "spec" / "complexity.py"
        assert complexity in measurement.fingerprinted("metrics")
        assert complexity not in measurement.fingerprinted("engine")

    def test_the_reachability_walk_is_in_the_metrics_set(self) -> None:
        """`piece_utilisation` counts only the types `pieces_a_player_can_hold` returns.

        The walk lives in its own file so the rest of `spec/schema.py`, which changes at
        every schema bump, stays outside the fingerprint. The metric has to import it from
        the file that is fingerprinted.
        """
        from chess_variant_sandbox.metrics import components

        walk = measurement.ROOT / "chess_variant_sandbox" / "spec" / "reachability.py"
        schema = measurement.ROOT / "chess_variant_sandbox" / "spec" / "schema.py"
        assert walk in measurement.fingerprinted("metrics")
        assert walk not in measurement.fingerprinted("engine")
        assert schema not in measurement.fingerprinted("metrics")
        assert components.pieces_a_player_can_hold.__module__ == "chess_variant_sandbox.spec.reachability"

    def test_the_game_record_is_in_the_metrics_set(self) -> None:
        """Every metric reads a game back through `GameRecord`."""
        record = measurement.ROOT / "chess_variant_sandbox" / "tournament" / "record.py"
        assert record in measurement.fingerprinted("metrics")
        assert record not in measurement.fingerprinted("engine")

    def test_the_schedule_and_the_game_loop_are_in_the_engine_set(self) -> None:
        """`runner.py` fixes every game's seed, and `play.py` hands the core its values.

        Either one changed means the stored games are not the games that would be played
        now, so the repair is a replay.
        """
        for name in ("runner.py", "play.py"):
            path = measurement.ROOT / "chess_variant_sandbox" / "tournament" / name
            assert path in measurement.fingerprinted("engine"), name
            assert path not in measurement.fingerprinted("metrics"), name

    def test_the_python_side_of_the_boundary_is_in_the_analysis_set(self) -> None:
        """The analysis pass decodes moves and reads positions through these four files.

        They decide no game, because the tournament runs whole inside the core, so they
        charge a re-analysis. They sit inside `metrics` as `analysis.py` does, and outside
        `engine`.
        """
        for name in ("__init__.py", "engine.py", "moves.py", "state.py"):
            path = measurement.ROOT / "chess_variant_sandbox" / "engine" / name
            assert path in measurement.fingerprinted("analysis"), name
            assert path in measurement.fingerprinted("metrics"), name
            assert path not in measurement.fingerprinted("engine"), name

    def test_a_fingerprint_moves_when_a_file_does(self, tmp_path) -> None:
        one = tmp_path / "a.py"
        one.write_text("x = 1", encoding="utf-8")
        before = measurement._hash_files([one])
        one.write_text("x = 2", encoding="utf-8")
        assert measurement._hash_files([one]) != before

    def test_it_does_not_depend_on_the_order_it_is_given(self, tmp_path) -> None:
        one = tmp_path / "a.py"
        two = tmp_path / "b.py"
        one.write_text("x = 1", encoding="utf-8")
        two.write_text("y = 2", encoding="utf-8")
        assert measurement._hash_files([one, two]) == measurement._hash_files([two, one])

    def test_the_name_is_part_of_it(self, tmp_path) -> None:
        """Two files with the same body are not the same set of sources."""
        one = tmp_path / "a.py"
        two = tmp_path / "b.py"
        one.write_text("same", encoding="utf-8")
        two.write_text("same", encoding="utf-8")
        assert measurement._hash_files([one]) != measurement._hash_files([two])


# What `source_fingerprint("analysis")` read when `analysis.BLUNDER_SAMPLE` was raised
# from 24 to 48, 2026-09-24 (Q388, `v11` A5, repair E; see `FROZEN_METRICS`'s own comment
# on the same move). It was "cff5777bf713d819" from before that move until then. `analysis.py`
# sits in both `ANALYSIS_SOURCES` and `METRICS_SOURCES`, so the one edit moved both pins at
# once; this is the tripwire that says whether the analysis *pass* itself moved, which costs
# `reanalyse`, distinct from a plain `metrics` move that only costs `rescore`.
#
# **A real change to the pass is allowed to move it.** When it does, the repair is to
# re-analyse every run, put the new value here with the date, and say in the commit which
# change moved it. What is not allowed is for it to move quietly.
#
# **Moved 2026-09-24 by Q383 (the package rename, `docs/archive/plan-rename.md` stage 4)**: the
# imports name the package, so the code reading moved while the pass did not. It was
# "25a187bfef3c04b7" after Q388; `FROZEN_METRICS`'s comment has the rest.
#
# **Moved 2026-09-26 by Q471**: the probe reads the immunity term, a real change to the
# pass on six specs. It was "39b9d85481a2febe" after Q383; `FROZEN_METRICS`'s comment has
# the rest, and Q460's replay is the re-analysis.
#
# **Moved 2026-10-02 by Q631**: `agents/evaluation.py` joined the set when `agents/*.py`
# left `engine`. It was "2c8e4e21096c5fd5" after Q471; `FROZEN_METRICS`'s comment has the
# rest, and the replay Q631 owes is the re-analysis.
#
# **Moved 2026-10-04 by Q708**: the probe's ruler reads the seal (see `FROZEN_METRICS`).
# It was "3d6002464c646c75" after Q631; the replay batched with Q696 is the re-analysis.
#
# **Moved 2026-10-09 by Q540**: `engine.py`'s move text names schema 17's captures (see
# `FROZEN_METRICS`). It was "4007c61e39ad76c0" after Q708; the replay Q540 owes is the
# re-analysis.
#
# **Moved 2026-10-09 by Q786**: the boundary's `goal_left_by` (see `FROZEN_METRICS`). It was
# "8a435f46ce02cd4d" after Q540; the batched replay Q803 is the re-analysis.
#
# **Moved 2026-10-10 by Q771**: schema 19's transits in `engine.py`'s move text and
# `agents/minimax.py`'s loud test (see `FROZEN_METRICS`). It was "a8d24b3ad16d45ac" after
# Q786; the replay Q771 owes is the re-analysis.
#
# **Moved 2026-10-10 by Q807**: schema 20's bar crosses the boundary in `engine.py` and
# `state.py` (see `FROZEN_METRICS`). It was "a0ed8e3487536646" after Q771; the replay Q807
# owes is the re-analysis.
#
# **Moved 2026-10-11 by Q843c**: schema 22's folds cross the boundary in `engine.py` and
# `state.py`, and `engine.py`'s move text names a fold's moves (see `FROZEN_METRICS`). It was
# "bc59672c7e718716" after Q807; the replay batched with Q843a's (Q843r) is the re-analysis.
#
# **Moved 2026-10-11 by Q846**: `agents/evaluation.py`'s immunity term skips a portal end's
# byte (see `FROZEN_METRICS`); it moves no number the core computes. It was
# "84d3505a6a0baa2c" after Q843c; Q847 re-analysed the library under it.
#
# **Moved 2026-10-13 by Q867b**: schema 24's move text through a fold in `engine.py` (see
# `FROZEN_METRICS`). It was "2aa42b5d8a26bb74" after Q846; the replay Q867b owes (Q867r) is
# the re-analysis.
#
# **Moved 2026-10-13 by Q873**: `tournament/analysis.py` reads a fold's own move to the end
# of the game (see `FROZEN_METRICS`). It was "f6b69603d9aa0c1c" after Q867b; Q867r's replay
# is the re-analysis.
#
# **Moved 2026-10-13 by Q875**: `tournament/analysis.py` reads a rotation as no capture and
# a momentum end as no instance (see `FROZEN_METRICS`). It was "8c26b090cb82ab85" after
# Q873; Q867r's replay is the re-analysis.
#
# **Moved 2026-10-15 by Q918a**: `engine/state.py` and `engine/engine.py` (see
# `FROZEN_METRICS`). It was "3c06f1e7c404993c" after Q875.
#
# **Moved 2026-10-15 by Q918b**: `agents/evaluation.py` and `engine/engine.py` (see
# `FROZEN_METRICS`). It was "3c32bdf6773b231e" after Q918a; Q929's replay is the re-analysis.
#
# **Moved 2026-10-15 by Q918d (the pass reads games in rounds)**: `tournament/analysis.py`
# walks a round game a round at a time, probes each player's moves against the opponent's
# equilibrium mix, forces round blunders and counts the resolutions (see `FROZEN_METRICS`). An
# alternating game takes the path it took, byte for byte (checked on 16 games and their
# trials of each of gardner, dobutsu, intransitive, crazyhouse and atomic). It was
# "7dfe04cc79c30b1f" after Q918b; the library's re-analysis under it is owed (the refresher).
#
# **Moved 2026-10-16 by Q942a**: `engine/state.py` and `engine/engine.py` (see
# `FROZEN_METRICS`). It was "2a75bbd0f846bec7" after Q918d; the replay after Q935a is the
# re-analysis.
#
# **Moved 2026-10-16 by Q935a**: `engine/engine.py`, `engine/state.py` and
# `engine/__init__.py` (see `FROZEN_METRICS`). It was "aa0316d1dc871057" after Q942a; the
# replay after Q935a is the re-analysis.
#
# **Moved 2026-10-16 by Q942b**: `agents/evaluation.py`, the probe's ruler, reads the rested-
# royal term under `rest` (see `FROZEN_METRICS`). It was "413c68553fd35509" after Q935a; the
# replay batched after Q950 is the re-analysis.
#
# **Moved 2026-10-17 by Q935b**: `agents/evaluation.py` and `engine/engine.py` (see
# `FROZEN_METRICS`). It was "64fd6dfab6c53bd9" after Q942b; the owed replay is the
# re-analysis.
#
# **Moved 2026-10-17 by Q935d (the pass reads a spy game)**: `tournament/analysis.py` walks a
# spy game's record by the phase the engine reports, probes its turns in Python with a round
# as a leaf, reads an announced round by the maximin and the best reply, counts `SPY_COUNTS`
# and forces a spy game's blunders (see `FROZEN_METRICS`). A game without a spy takes the
# path it took, byte for byte. It was "87e95a857dae4b4a" after Q935b; the owed replay is the
# re-analysis.
#
# **Moved 2026-10-17 by Q935h**: `agents/evaluation.py`, the probe's ruler, reads the spy term
# in a spy game (see `FROZEN_METRICS`). It was "b1de98c45e315f78" after Q935d and Q952's
# replay; the next library replay is the re-analysis.
#
# **Moved 2026-10-17 by Q947e**: `engine/state.py` and `engine/__init__.py` lose the dead
# `PLAYER_ZERO`/`PLAYER_ONE` (see `FROZEN_METRICS`). It was "12d9bfabaa75f542" after Q935h;
# the next library replay is the re-analysis.
#
# **Moved 2026-10-19 by Q943b**: `agents/evaluation.py` and `tournament/analysis.py` (see
# `FROZEN_METRICS`); a game without a speed is analysed byte for byte as before. It was
# "031be810615f81e7" after Q947e and Q966's replay; the next library replay is the re-analysis.
#
# **Moved 2026-10-20 by Q983a**: `agents/evaluation.py` (the probe's ruler) and
# `engine/state.py` (see `FROZEN_METRICS`). It was "8a7d2819c34b14a5" after Q943b and Q968's
# replay; Q983c's replay is the re-analysis.
#
# **Moved 2026-10-20 by Q986b**: `agents/evaluation.py` (the probe's ruler; the rested royal is
# now `RESTED_ROYAL_PIECES` mean pieces, see `FROZEN_METRICS`). It was "6aaf204a1094d816" after
# Q983a; the replay batched after Q986b (Q558r) is the re-analysis.
#
# **Moved 2026-10-22 by Q999a**: `engine/engine.py`, `engine/state.py` and `engine/__init__.py`
# (see `FROZEN_METRICS`); a game without a sight is analysed byte for byte as before. It was
# "a0aa597a07b63b94" after Q558r's replay; the next library replay is the re-analysis.
#
# **Moved 2026-10-22 by Q999c**: `tournament/analysis.py`'s sight tallies (see `FROZEN_METRICS`);
# a game without a sight is analysed byte for byte as before. It was "2e9d152d5ca7f07b" after
# Q999a; the replay R30 owed for Q999a is the re-analysis.
#
# **Moved 2026-10-22 by Q983b**: `tournament/analysis.py`'s rest tallies and `Engine.rest_turn`
# (see `FROZEN_METRICS`); a game without a rest is analysed byte for byte as before
# (`tests/test_rest_players.py`). It was "59c2e597b2f08d51" after Q999c; the library's
# `reanalyse` owed after Q983b is the re-analysis.
#
# **Moved 2026-10-23 by Q1007c**: `engine/engine.py`, `moves.py` and `state.py`'s command (see
# `FROZEN_METRICS`); a game without a command is analysed byte for byte as before. It was
# "d375f16fdb8ee837" after Q983b; R31's replay is the re-analysis.
#
# **Moved 2026-10-23 by Q1007d**: `tournament/analysis.py`'s command tallies (`COMMAND_SERIES`,
# see `FROZEN_METRICS`); a game without a command is analysed byte for byte as before
# (`tests/test_command_page.py`, `tests/test_sight_page.py`). It was "e54e6675c520c48c" after
# Q1007c; the library replay after the detached training campaign is the re-analysis.
FROZEN_ANALYSIS = "c7d6042d1d057fc2"


class TestTheAnalysisFingerprintIsWhereTheLibraryLeftIt:
    """The other half of a score's basis, pinned the same way.

    `analysis` is the fingerprint every run's `analysis.json` was written under, and a
    change to it stales all of them by the stronger repair. Before this pin, an undeclared
    move showed only as one more *re-analyse* line in `chess-variant-sandbox measure --check`'s output --
    invisible next to a corpus that is already stale by re-analyse for a declared reason.
    """

    def test_it_reads_what_the_library_was_analysed_under(self) -> None:
        assert measurement.source_fingerprint("analysis") == FROZEN_ANALYSIS, (
            "the analysis fingerprint moved; every run in runs/ now reads stale by "
            "re-analyse. If that was the point, re-analyse them and put the new value in "
            "FROZEN_ANALYSIS with the date and the change that moved it"
        )


class TestLineEndingsAreNotPartOfIt:
    """The same commit in two checkouts has to be one fingerprint.

    `core.autocrlf=true` on the machine this is developed on and the committed blobs are
    LF, so a fresh `.claude/worktrees/` checkout holds CRLF where the main checkout holds
    LF. Hashing raw bytes made identical code fingerprint differently in the two, and once
    the drift reached `rust/src/` it landed inside the *build* fingerprint, whose only
    repair is a replay -- so a campaign run in a worktree could not be delivered as
    current at all. `docs/archive/bugs-resolved.md` has the 60-spec measurement.

    The tests below pin both halves of the fold. It has to erase CRLF against LF, and it
    has to leave everything else exactly where it was -- a fingerprint that stops noticing
    a real difference is worse than the bug it was fixing.
    """

    def test_crlf_and_lf_are_one_fingerprint(self, tmp_path) -> None:
        crlf = tmp_path / "crlf"
        lf = tmp_path / "lf"
        crlf.mkdir()
        lf.mkdir()
        body = b"x = 1\ny = 2\nz = 3\n"
        (crlf / "a.py").write_bytes(body.replace(b"\n", b"\r\n"))
        (lf / "a.py").write_bytes(body)
        assert measurement._hash_files([crlf / "a.py"]) == measurement._hash_files([lf / "a.py"])

    def test_an_lf_file_hashes_to_what_it_always_did(self, tmp_path) -> None:
        """The pin that says the fold moved no existing number.

        Taken from the byte-at-a-time loop over `b"a.py" + NUL + body + NUL`, which is what
        `_hash_files` fed FNV-1a before the fold existed and still feeds it for LF input.
        Every fingerprint in `runs/` taken over LF sources is this arithmetic.
        """
        one = tmp_path / "a.py"
        one.write_bytes(b"x = 1\ny = 2\n")
        assert measurement._hash_files([one]) == "4c2b08865f63a3c3"

    def test_a_lone_carriage_return_is_not_folded(self, tmp_path) -> None:
        """Only the pair is git's doing, so only the pair is erased.

        A bare CR is a real difference between two files, and no checkout on this machine
        produces one. Folding it would collapse them for nothing.
        """
        one = tmp_path / "a.py"
        one.write_bytes(b"x = 1\ry = 2\r")
        assert measurement._hash_files([one]) == "3f890ad2a095c27f"
        assert measurement._hash_files([one]) != "4c2b08865f63a3c3"

    def test_the_fold_is_outside_the_hash(self) -> None:
        """`_fnv1a` is still the plain FNV-1a `absorb` is, CR and all.

        The fold belongs to the file reader because only a file body carries a line ending.
        `tables_fingerprint` hands `_fnv1a` a `json.dumps` blob, which escapes a CR rather
        than emitting one, and moving the fold inside would put a second rule between the
        two languages for no gain.
        """
        assert measurement._fnv1a([b"a\r\nb"]) != measurement._fnv1a([b"a\nb"])


class TestTheHashIsAbsorbByteForByte:
    """`_fnv1a` is `absorb` from `rust/build.rs`, run four bytes a block.

    `absorb` xors a byte in, multiplies by the prime and keeps the low 64 bits, once per
    byte. `_fnv1a` keeps the low 64 bits once per four bytes, because in pure Python the
    mask is an allocation and the reduction commutes with both steps -- item 16 of
    `docs/archive/performance-opportunities.md`. Same residue by construction, and construction
    is not what a fingerprint is trusted on: one bit out and every score in `runs/` is
    stale at once, with the build check reporting a rebuild that never catches up.

    So the reference below is `absorb`'s loop transcribed, kept here rather than imported
    so the module under test cannot define its own correctness. The fixed values are what
    the byte-at-a-time implementation gave before the change, and the two short ones are
    the published FNV-1a 64-bit test vectors. The third leg -- that this agrees with what
    the Rust side compiled in for the current `rust/src/` -- is `TestTheBuildFingerprint`
    below.
    """

    @staticmethod
    def _absorb(chunks: list[bytes]) -> str:
        digest = 0xCBF29CE484222325
        for chunk in chunks:
            for byte in chunk:
                digest ^= byte
                digest = (digest * 0x00000100000001B3) & ((1 << 64) - 1)
        return f"{digest:016x}"

    @pytest.mark.parametrize(
        ("chunks", "digest"),
        [
            # Nothing absorbed is the offset basis.
            ([], "cbf29ce484222325"),
            ([b""], "cbf29ce484222325"),
            # The published vectors, one byte and six.
            ([b"a"], "af63dc4c8601ec8c"),
            ([b"foobar"], "85944171f73967e8"),
            # The same six bytes with the chunk boundaries somewhere else.
            ([b"foo", b"", b"bar"], "85944171f73967e8"),
            # Every byte value once, in order: a whole number of blocks.
            ([bytes(range(256))], "4242dc5249c33625"),
        ],
    )
    def test_fixed_inputs_hash_to_what_they_always_did(self, chunks, digest) -> None:
        assert measurement._fnv1a(chunks) == digest
        assert self._absorb(chunks) == digest

    def test_every_block_phase_agrees_with_the_byte_at_a_time_loop(self) -> None:
        """Lengths 0 through 63: every remainder mod 4, and blocks before each tail.

        The tail is the part a blocked implementation gets wrong -- a byte dropped or
        absorbed twice at the boundary -- and it is invisible to any test whose inputs
        happen to be whole blocks.
        """
        rng = random.Random(16)
        for length in range(64):
            for _ in range(4):
                data = rng.randbytes(length)
                assert measurement._fnv1a([data]) == self._absorb([data]), length

    def test_chunk_boundaries_are_invisible(self) -> None:
        """`_hash_files` hands over names, separators and bodies as separate chunks.

        The digest is of the concatenation and nothing else, which is also what
        `build.rs` computes with four `absorb` calls per file.
        """
        rng = random.Random(16)
        for _ in range(64):
            parts = [rng.randbytes(rng.randrange(40)) for _ in range(rng.randrange(1, 6))]
            assert measurement._fnv1a(parts) == measurement._fnv1a([b"".join(parts)])
            assert measurement._fnv1a(parts) == self._absorb(parts)


class TestTheBuildFingerprint:
    """The one number computed twice, in two languages, that has to agree.

    `rust/build.rs` hashes `rust/src/` at compile time and PyO3 exposes the result;
    `measurement._hash_files` computes the same thing from the same files at read time.
    Comparing them is what turns "you edited the core and did not rebuild" -- the footgun
    `CLAUDE.md` sets in bold -- from a silently wrong measurement into an error.

    If FNV-1a is ever changed on one side, this is the test that fails, and it fails
    loudly rather than by quietly reporting a permanent rebuild-needed.

    Both sides now fold CRLF to LF before hashing, and that fold has to match too. A
    checkout whose `rust/src/` holds CRLF is the case that proves it -- the two numbers
    agree there only if `build.rs::fold_crlf` and `measurement._fold_crlf` do the same
    thing -- so the first test below says which kind of checkout it just ran in.
    """

    @staticmethod
    def _crlf_sources() -> list[str]:
        src = measurement.ROOT / "rust" / "src"
        return sorted(p.name for p in src.glob("*.rs") if b"\r\n" in p.read_bytes())

    def test_the_rust_and_python_hashes_agree(self) -> None:
        on_disk = measurement.rust_source_fingerprint()
        if on_disk is None:
            pytest.skip("no rust/src beside this install")
        crlf = self._crlf_sources()
        witness = f"{len(crlf)} CRLF sources in this checkout: {', '.join(crlf) or 'none'}"
        assert on_disk == measurement.built_fingerprint(), witness

    def test_the_compiled_number_is_the_lf_reading_of_the_sources(self, tmp_path) -> None:
        """The cross-checkout claim, on the real sources rather than a fixture.

        Copy `rust/src/` out twice, once all-LF and once all-CRLF, and hash both. Equal to
        each other says the fold works; equal to what the loaded core carries says the
        number baked in at compile time is the token reading (Q630) of the LF form, which
        is `_rust_reading` -- and that the reading is the property that
        lets a measurement taken in a CRLF worktree be delivered into an LF checkout.
        """
        src = measurement.ROOT / "rust" / "src"
        if not src.is_dir():
            pytest.skip("no rust/src beside this install")

        lf, crlf = tmp_path / "lf", tmp_path / "crlf"
        lf.mkdir()
        crlf.mkdir()
        for path in src.glob("*.rs"):
            body = path.read_bytes().replace(b"\r\n", b"\n")
            (lf / path.name).write_bytes(body)
            (crlf / path.name).write_bytes(body.replace(b"\n", b"\r\n"))

        reading = measurement._hash_files(list(lf.glob("*.rs")), measurement._rust_reading)
        assert measurement._hash_files(
            list(crlf.glob("*.rs")), measurement._rust_reading
        ) == reading
        assert measurement.built_fingerprint() == reading

    def test_a_matching_build_is_not_reported_as_stale(self) -> None:
        if measurement.rust_source_fingerprint() is None:
            pytest.skip("no rust/src beside this install")
        assert measurement.stale_build() is None


_RUST_SRC = measurement.ROOT / "rust" / "src"
_CORPUS = measurement.ROOT / "rust" / "reading-corpus"

_RUST = b"""//! A module.
use std::fmt;

/// Adds one.
pub fn add(x: u32) -> u32 {
    let s = "a // b"; // the comment
    let c = '/';
    x + 1
}

#[cfg(test)]
mod tests {
    #[test]
    fn adds() { assert_eq!(super::add(1), 2); }
}
"""


def _rust_alike(tmp_path: Path, before: bytes, after: bytes) -> bool:
    one = tmp_path / "before" / "a.rs"
    two = tmp_path / "after" / "a.rs"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_bytes(before)
    two.write_bytes(after)
    return measurement._hash_files([one], measurement._rust_reading) == (
        measurement._hash_files([two], measurement._rust_reading)
    )


class TestTheRustTokenReading:
    """Q630: a comment, whitespace or test-only edit in `rust/src/` moves no fingerprint.

    `rust/build.rs` (through `rust/token_reading.rs`) and `measurement._rust_reading`
    compute one reading in two languages. Both are held to `rust/reading-corpus/`, a
    hand-written list of awkward inputs and their readings: `cargo test`'s
    `tests/token_reading.rs` on the Rust side, the first test here on this one. The third
    leg, that the number compiled into the loaded core is this reading of the sources on
    disk, is `TestTheBuildFingerprint` above.
    """

    def test_every_corpus_case_reads_as_written(self) -> None:
        inputs = sorted(_CORPUS.glob("*.in"))
        assert len(inputs) >= 15
        for path in inputs:
            expected = path.with_suffix(".out").read_bytes()
            assert measurement._rust_reading(path.read_bytes()) == expected, path.name

    def test_crlf_reads_as_lf(self) -> None:
        assert measurement._rust_reading(_RUST.replace(b"\n", b"\r\n")) == (
            measurement._rust_reading(_RUST)
        )

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (b"// the comment", b"// a reworded comment"),
            (b"// the comment", b""),
            (b"/// Adds one.\n", b"/// Adds one,\n/// and says so at length.\n"),
            (b"//! A module.\n", b""),
            (b"x + 1\n", b"x + 1 /* a block /* nested */ comment */\n"),
            (b"    let c", b"\n\n\t\tlet c"),
            (b"pub fn add(x: u32) -> u32 {", b"pub fn add(x: u32)\n    -> u32\n{"),
            (b"assert_eq!(super::add(1), 2);", b"assert_eq!(super::add(2), 3);"),
            (b"    fn adds()", b"    // a new test\n    fn adds()"),
        ],
    )
    def test_a_comment_whitespace_or_test_edit_leaves_it(self, tmp_path, old, new) -> None:
        assert old in _RUST
        assert _rust_alike(tmp_path, _RUST, _RUST.replace(old, new))

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (b"x + 1", b"x + 2"),
            # Whitespace collapses but is never removed, so space added where there
            # was none moves it: `a & &b` and `a && b` are different programs.
            (b"x + 1", b"x+1"),
            (b"add(x", b"add( x"),
            # Inside a literal, `//` and whitespace are the value.
            (b'"a // b"', b'"a // c"'),
            (b'"a // b"', b'"a //  b"'),
            (b"'/'", b"'|'"),
            (b"use std::fmt;", b"use std::fmt as f;"),
            # The test module's attribute is what makes it test-only.
            (b"#[cfg(test)]\n", b""),
        ],
    )
    def test_a_code_edit_moves_it(self, tmp_path, old, new) -> None:
        assert old in _RUST
        assert not _rust_alike(tmp_path, _RUST, _RUST.replace(old, new))

    def test_every_core_source_is_read_as_tokens(self) -> None:
        """No real source falls back to bytes, or a comment there would still stale."""
        if not _RUST_SRC.is_dir():
            pytest.skip("no rust/src beside this install")
        for path in _RUST_SRC.glob("*.rs"):
            body = measurement._fold_crlf(path.read_bytes())
            assert measurement._rust_token_reading(body) != body, path.name

    def test_commenting_every_source_leaves_the_fingerprint(self, tmp_path) -> None:
        """The whole of `rust/src/`, copied and commented throughout, hashes the same."""
        if not _RUST_SRC.is_dir():
            pytest.skip("no rust/src beside this install")
        plain, commented = tmp_path / "plain", tmp_path / "commented"
        plain.mkdir()
        commented.mkdir()
        for path in _RUST_SRC.glob("*.rs"):
            body = measurement._fold_crlf(path.read_bytes())
            (plain / path.name).write_bytes(body)
            lines = []
            for line in body.split(b"\n"):
                stripped = line.lstrip()
                if stripped.startswith((b"fn ", b"pub fn ", b"pub(crate) fn ")):
                    lines.append(b"/// Q630: a doc line that reaches no build.")
                if stripped.startswith(b"let ") and line.endswith(b";"):
                    line += b"  // Q630: a trailing remark"
                lines.append(line)
            text = b"\n".join(lines)
            (commented / path.name).write_bytes(
                b"/* Q630: a header. */\n\n" + text + b"\n\n// Q630: a footer.\n"
            )
            assert text != body, path.name
        reading = measurement._hash_files(list(plain.glob("*.rs")), measurement._rust_reading)
        assert reading == measurement._hash_files(
            list(commented.glob("*.rs")), measurement._rust_reading
        )
        assert reading == measurement.rust_source_fingerprint()
        # And the bytes would have moved: the reading, not the copy, is what held still.
        assert measurement._hash_files(list(plain.glob("*.rs"))) != (
            measurement._hash_files(list(commented.glob("*.rs")))
        )


_SOURCE = b'''"""Module docstring."""

import os  # why os


class A:
    """Class docstring."""

    # a comment line
    x = "kept # not a comment"

    def f(self):
        """Function docstring,
        on two lines."""
        text = """first

last   """
        return text  # trailing
'''

_READING = b'''import os
class A:
    x = "kept # not a comment"
    def f(self):
        text = """first

last   """
        return text'''


def _reads_alike(tmp_path, before: bytes, after: bytes) -> bool:
    one = tmp_path / "before" / "a.py"
    two = tmp_path / "after" / "a.py"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_bytes(before)
    two.write_bytes(after)
    return measurement._hash_files([one], measurement._code_reading) == (
        measurement._hash_files([two], measurement._code_reading)
    )


class TestTheCodeReading:
    """Prose edits leave a Python fingerprint alone, and code edits move it.

    `292d7ae` reworded one docstring and one glossary string, and all 61 stored scores read
    stale. The re-score moved 0 of 73 indices. The dangerous direction is still the
    permissive one, so most cases below are edits that must move the fingerprint.
    """

    def test_the_reading_keeps_code_and_drops_prose(self) -> None:
        """Pinned as text, so a Python upgrade that changes the reading fails here."""
        assert measurement._code_reading(_SOURCE) == _READING

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (b"Module docstring.", b"A reworded module docstring."),
            (b"Class docstring.", b"Class docstring,\n    now on two lines."),
            (b"on two lines.", b"on two lines, reworded."),
            (b'        """Function docstring,\n        on two lines."""\n', b""),
            (b"# a comment line", b"# a different comment"),
            (b"    # a comment line\n", b""),
            (b"import os  # why os", b"import os"),
            (b"return text  # trailing", b"return text  # other"),
            (b"\n\nclass A:", b"\n\n\n\n\nclass A:"),
        ],
    )
    def test_a_prose_edit_leaves_it(self, tmp_path, old, new) -> None:
        assert old in _SOURCE
        assert _reads_alike(tmp_path, _SOURCE, _SOURCE.replace(old, new))

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (b"return text  #", b"return None  #"),
            (b"kept # not a comment", b"kept # changed"),
            # Blank lines and trailing whitespace inside a string are its value.
            (b"first\n\nlast", b"first\nlast"),
            (b"last   \"\"\"", b"last\"\"\""),
            # A string statement after the first is not a docstring.
            (b"        return text", b'        "not a docstring"\n        return text'),
            (b"import os", b"import sys"),
            (b"    def f(self):", b"    async def f(self):"),
        ],
    )
    def test_a_code_edit_moves_it(self, tmp_path, old, new) -> None:
        assert old in _SOURCE
        assert not _reads_alike(tmp_path, _SOURCE, _SOURCE.replace(old, new))

    def test_blank_lines_inside_an_f_string_are_its_value(self, tmp_path) -> None:
        assert not _reads_alike(
            tmp_path, b'x = f"""a\n\n{1}"""\n', b'x = f"""a\n{1}"""\n'
        )

    def test_crlf_and_lf_read_alike(self, tmp_path) -> None:
        assert _reads_alike(tmp_path, _SOURCE, _SOURCE.replace(b"\n", b"\r\n"))

    @pytest.mark.parametrize(
        "source",
        [
            b"def broken(:\n    return 1  # a comment\n",
            b"# -*- coding: latin-1 -*-\nx = 1  # a comment\n",
            b"x = 1  # a comment\ry = 2\n",
        ],
    )
    def test_anything_in_doubt_is_read_as_bytes(self, tmp_path, source) -> None:
        """A file that does not parse, declares a coding, or holds a lone CR."""
        assert measurement._code_reading(source) == source
        assert not _reads_alike(tmp_path, source, source.replace(b"a comment", b"other"))

    def test_every_fingerprinted_source_is_read_as_code(self) -> None:
        """No real source falls back to bytes, or prose edits there would still stale.

        Each body is read once (Q1009): the three kinds share most of their files, and
        reading every one afresh for each kind and each question cost 2.3 s. The reading
        is a function of the body, so a body seen before gets the answer it got.
        """
        readings: dict[bytes, bytes] = {}

        def reading(body: bytes) -> bytes:
            if body not in readings:
                readings[body] = measurement._code_reading(body)
            return readings[body]

        for kind in ("engine", "metrics", "analysis"):
            paths = measurement.fingerprinted(kind)
            assert paths, kind
            for path in paths:
                body = measurement._fold_crlf(path.read_bytes())
                assert reading(body) != body, path.name
            assert measurement.source_fingerprint(kind) == measurement._hash_files(
                paths, reading
            )

    @staticmethod
    def _fingerprinted_trees() -> list[tuple[str, object]]:
        import ast

        paths = {
            path
            for kind in ("engine", "metrics", "analysis")
            for path in measurement.fingerprinted(kind)
        }
        return [
            (path.name, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(paths)
        ]

    def test_no_fingerprinted_source_reads_a_docstring(self) -> None:
        """The premise for dropping docstrings: they reach no result."""
        import ast

        for name, tree in self._fingerprinted_trees():
            for node in ast.walk(tree):  # type: ignore[arg-type]
                word = getattr(node, "attr", None) or getattr(node, "id", None)
                assert word not in ("__doc__", "getdoc"), name


class TestEveryImportIsFingerprintedOrCoveredElsewhere:
    """A fingerprinted source that imports an unfingerprinted module is blind to it.

    `spec/complexity.py` fed `rule_yield` from outside every set until Q269, and Q274 found
    more modules like it. This walks every `chess_variant_sandbox` import a fingerprinted source makes,
    lazy and type-only ones included. Each module it reaches is in a source set, or named
    in `COVERED` with what covers it instead. The record's Q274 table gives the reasons.
    """

    # (module, name) -> what covers it. A name of "*" covers the whole module.
    COVERED: dict[tuple[str, str], str] = {
        ("chess_variant_sandbox/engine/compile.py", "*"): "tables: build_tables flattens every field read",
        ("chess_variant_sandbox/engine/bridge.py", "*"): "tables: tables_fingerprint hashes its output",
        ("chess_variant_sandbox/spec/canonical.py", "*"): "ruleset_hash: its mechanics bytes are the hash",
        ("chess_variant_sandbox/spec/__init__.py", "spec_hash"): "ruleset_hash",
        ("chess_variant_sandbox/spec/schema.py", "format_square"): "a mutant's own ruleset_hash",
        ("chess_variant_sandbox/spec/schema.py", "parse_square"): "a mutant's own ruleset_hash",
        ("chess_variant_sandbox/spec/transform.py", "*"): "a mutant's own ruleset_hash",
        # Since Q631 the agents are read by what they hand the core, not by their source.
        ("chess_variant_sandbox/agents/__init__.py", "native_spec"): (
            "players: players_fingerprint hashes every seat's native spec"
        ),
        ("chess_variant_sandbox/agents/__init__.py", "parse_agent"): (
            "players: a name reaches the core only as its native spec"
        ),
        ("chess_variant_sandbox/agents/__init__.py", "build_agent"): (
            "agents_source: a seat with no native form makes players unreadable"
        ),
        ("chess_variant_sandbox/agents/base.py", "AgentConfig"): "a type, and players",
        # The heuristic pass plays its own games, which no basis stamps; nothing heuristic
        # is scored (`.claude/rules/agents-and-eval.md`).
        ("chess_variant_sandbox/agents/features.py", "VOCABULARY_VERSION"): "the heuristic pass",
        ("chess_variant_sandbox/agents/heuristics.py", "Heuristic"): "the heuristic pass",
        ("chess_variant_sandbox/agents/heuristics.py", "HeuristicSet"): "the heuristic pass",
    }

    @staticmethod
    def _source(dotted: str) -> Path | None:
        """The file a dotted module name imports, or ``None`` for a name inside a module.

        Matched against the directory listing, because a case-insensitive filesystem
        would otherwise read the class `chess_variant_sandbox.engine.Engine` as a module `Engine.py`.
        """
        *parents, last = dotted.split(".")
        folder = measurement.ROOT.joinpath(*parents)
        if not folder.is_dir():
            return None
        names = {entry.name for entry in folder.iterdir()}
        if f"{last}.py" in names:
            return folder / f"{last}.py"
        if last in names and (folder / last / "__init__.py").is_file():
            return folder / last / "__init__.py"
        return None

    @staticmethod
    def _sources() -> set[Path]:
        return {
            path
            for kind in ("engine", "metrics", "analysis")
            for path in measurement.fingerprinted(kind)
        }

    def _reached(self) -> list[tuple[str, Path | None, str]]:
        """Every `(importer, module file, name)` a fingerprinted source imports."""
        import ast

        found: list[tuple[str, Path | None, str]] = []
        for path in sorted(self._sources()):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.level == 0, f"{path.name} imports relatively"
                    pairs = [(node.module or "", alias.name) for alias in node.names]
                elif isinstance(node, ast.Import):
                    pairs = [(alias.name, "") for alias in node.names]
                else:
                    continue
                for module, name in pairs:
                    if module.split(".")[0] != "chess_variant_sandbox":
                        continue

                    # The built extension. `Basis.engine` covers it through `built`.
                    if "chess_variant_sandbox._core" in (module, f"{module}.{name}"):
                        continue

                    whole = self._source(f"{module}.{name}" if name else module)
                    if whole is not None:
                        found.append((path.name, whole, "*"))
                    else:
                        found.append((path.name, self._source(module), name))
        return found

    def test_every_module_a_score_reads_is_fingerprinted_or_covered(self) -> None:
        sources = self._sources()
        blind = []
        for importer, target, name in self._reached():
            assert target is not None, f"{importer} imports {name} from nowhere this can find"
            if target in sources:
                continue
            module = target.relative_to(measurement.ROOT).as_posix()
            if (module, "*") not in self.COVERED and (module, name) not in self.COVERED:
                blind.append(f"{importer} imports {name} from {module}")
        assert not blind, blind

    def test_every_covered_entry_is_still_imported(self) -> None:
        """An entry nothing imports any more is a claim nobody checks. Delete it."""
        reached: set[tuple[str, str]] = set()
        for _importer, target, name in self._reached():
            if target is None:
                continue
            module = target.relative_to(measurement.ROOT).as_posix()
            reached |= {(module, name), (module, "*")}
        assert set(self.COVERED) <= reached, sorted(set(self.COVERED) - reached)

    def test_no_canonical_roster_seats_a_fairy_stockfish_player(self) -> None:
        """What `COVERED` rests on for `fairy/`. Seating one would change the roster."""
        from chess_variant_sandbox.corpus import CANONICAL_AGENTS

        assert not any(agent.split(":")[0] == "fairystockfish" for agent in CANONICAL_AGENTS)


class TestTheGlossaryIsNotFingerprinted:
    """Display prose beside the metrics, left out of the code reading.

    Its `how` strings are constants rather than docstrings, so the code reading alone
    would still charge a corpus re-score for rewording one.
    """

    def test_it_is_outside_every_code_reading_and_inside_the_bytes_reading(self) -> None:
        glossary = measurement.ROOT / "chess_variant_sandbox" / "metrics" / "glossary.py"
        assert glossary.exists()
        assert glossary in set(measurement._sources(*measurement.METRICS_SOURCES))
        for kind in ("engine", "metrics", "analysis"):
            assert glossary not in measurement.fingerprinted(kind), kind

    def test_no_fingerprinted_source_imports_it(self) -> None:
        """The premise for leaving it out. A score that read it would be blind to it."""
        import ast

        for name, tree in TestTheCodeReading._fingerprinted_trees():
            for node in ast.walk(tree):  # type: ignore[arg-type]
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                assert not any("glossary" in word for word in names), name


class TestAStampFromBeforeTheCodeReading:
    """Every stored basis on the day the code reading landed holds the bytes reading.

    Such a stamp still clears where today's bytes match it, because unchanged bytes are
    unchanged code. That is exactly as strict as the bytes reading was, and it spares the
    corpus a replay for a hasher change. Once any byte moves, the stamp is stale.
    """

    @pytest.mark.parametrize(
        ("kind", "needs"),
        [("engine", "replay"), ("metrics", "rescore"), ("analysis", "reanalyse")],
    )
    def test_it_clears_while_its_bytes_match_and_not_after(self, kind, needs) -> None:
        stored = _basis(**{kind: "bytes-reading"})
        assert stored.stale_against(_basis(**{f"{kind}_bytes": "bytes-reading"})).current
        stale = stored.stale_against(_basis(**{f"{kind}_bytes": "moved"}))
        assert stale.needs == needs

    def test_an_empty_bytes_reading_clears_nothing(self) -> None:
        assert not _basis(metrics="").stale_against(_basis(metrics_bytes="")).current

    def test_the_bytes_reading_is_neither_stored_nor_compared(self) -> None:
        current = Basis.current("abc123", "v9", ("random",), 4, (0,), tables="t")
        built = measurement.built_fingerprint()
        assert current.engine_bytes == measurement.bytes_fingerprint("engine") + "-" + built
        assert current.metrics_bytes == measurement.bytes_fingerprint("metrics")
        assert current.analysis_bytes == measurement.bytes_fingerprint("analysis")
        assert "metrics_bytes" not in current.as_dict()
        assert Basis.from_dict(current.as_dict()) == current


class TestWhatMakesAMeasurementStale:
    def test_a_basis_matches_itself(self) -> None:
        basis = _basis()
        assert basis.stale_against(basis).current
        assert not basis.stale_against(basis)

    @pytest.mark.parametrize(
        ("field", "value", "needs"),
        [
            # The games themselves are about a different match.
            ("ruleset_hash", "different", "replay"),
            ("agents", ("random",), "replay"),
            ("games_per_pairing", 8, "replay"),
            ("seeds", (1,), "replay"),
            ("engine", "moved", "replay"),
            ("tables", "moved", "replay"),
            # The games stand but the file the pass wrote from them does not.
            ("analysis", "moved", "reanalyse"),
            # The games stand; only the number derived from them moved.
            ("weights_version", "v7", "rescore"),
            # `metrics` is held by `test_a_change_to_the_scorer_alone_is_a_rescore`, which
            # asserts the same `needs` and the reason's words as well (Q754).
        ],
    )
    def test_each_field_is_compared_and_costs_what_it_should(
        self, field, value, needs
    ) -> None:
        stale = _basis(**{field: value}).stale_against(_basis())
        assert stale.needs == needs, field
        assert stale.reasons, field

    def test_a_replay_is_never_downgraded_to_a_rescore(self) -> None:
        """The `needs or` in `stale_against`, which is deliberate and easy to get wrong.

        When the engine *and* the weights have both moved, the answer is replay: the
        cheap repair would re-score games that should not have been kept. Writing this
        as an unordered assignment would silently pick whichever check ran last.
        """
        stale = _basis(engine="moved", weights_version="v7").stale_against(_basis())
        assert stale.needs == "replay"
        assert len(stale.reasons) == 2

    def test_a_reanalysis_is_never_downgraded_to_a_rescore(self) -> None:
        """The same property one rung down: the pass and the weights both moved.

        A re-score would read the stale `analysis.json` and stamp the result current --
        which is the defect `docs/archive/bugs-resolved.md` (`v5`) demonstrates, and the reason the
        weights check runs *after* the analysis check rather than before it.
        """
        stale = _basis(analysis="moved", weights_version="v7").stale_against(_basis())
        assert stale.needs == "reanalyse"
        assert len(stale.reasons) == 2

    def test_a_replay_is_never_downgraded_to_a_reanalysis(self) -> None:
        stale = _basis(engine="moved", analysis="moved").stale_against(_basis())
        assert stale.needs == "replay"

    def test_a_change_to_the_pass_is_named_once(self) -> None:
        """An edit to `analysis.py` moves `metrics` too, because the file is in both sets.

        Reporting "the metrics changed" beside "the analysis pass changed" would be the
        same fact twice, the second time less precisely. The pass is the reason given.
        """
        stale = _basis(analysis="moved", metrics="moved").stale_against(_basis())
        assert stale.needs == "reanalyse"
        assert stale.reasons == ("the analysis pass changed since it was run",)

    def test_a_change_to_the_scorer_alone_is_a_rescore(self) -> None:
        """`metrics/`, the ladder or the heuristics moved and the pass did not.

        This is the cheap repair the split exists to keep reachable: the stored
        `analysis.json` is still the file the current pass would write, so it is read
        rather than rewritten.
        """
        stale = _basis(metrics="moved").stale_against(_basis())
        assert stale.needs == "rescore"
        assert stale.reasons == ("the metrics changed since it was scored",)

    def test_a_roster_that_changed_players_says_which(self) -> None:
        """Since v9 a roster can differ with the same shape, so shape is not the reason.

        A net ladder plays a learned evaluator on its searching rungs, which makes
        `minimax:nodes=6400` and `minimax:net=P,nodes=6400` the same count at the same
        games a pairing and different players. Reporting only the shape rendered that as
        "different roster (2 agents at 4 games a pairing, not 2 at 4)", which reads as a
        bug rather than as something a person can act on.
        """
        stale = _basis(agents=("random", "minimax:nodes=400")).stale_against(
            _basis(agents=("random", "minimax:net=abc,nodes=400"))
        )
        assert stale.needs == "replay"
        summary = stale.summary()
        assert "minimax:nodes=400" in summary
        assert "minimax:net=abc,nodes=400" in summary
        # The shape is the same, so quoting it would be the confusing half.
        assert "2 agents at 4 games a pairing, not 2 at 4" not in summary
        # `random` is in both rosters and is not what changed.
        assert "random" not in summary

    def test_a_roster_that_changed_shape_still_says_the_shape(self) -> None:
        """The other branch: different counts are described by their counts."""
        stale = _basis(agents=("random",)).stale_against(_basis())
        assert "1 agents at 4 games a pairing, not 2 at 4" in stale.summary()

    def test_the_reason_names_both_versions(self) -> None:
        """A reason a person can act on says what it has and what it wants."""
        stale = _basis(weights_version="v5").stale_against(_basis())
        assert "v5" in stale.summary()
        assert "v8" in stale.summary()


class TestAMeasurementThatCannotSayWhatItRestsOn:
    """The common case on first contact: everything measured before this existed.

    Reported as stale rather than as current, and that direction is the whole design.
    Trusting an unstamped measurement is exactly the default that produced the failure;
    refusing it costs one re-measure, once.
    """

    def test_a_missing_basis_is_unknown_and_needs_a_replay(self, tmp_path) -> None:
        path = tmp_path / "canonical.json"
        path.write_text(json.dumps({"hash": "abc123", "index": 0.8}), encoding="utf-8")

        stored = measurement.load(path, _basis())
        assert stored is not None
        assert not stored.current
        assert stored.staleness.needs == "replay"
        assert "before this was tracked" in stored.staleness.summary()

    @pytest.mark.parametrize(
        "basis",
        [None, {}, {"weights_version": "v8"}, "not a dict", {"agents": "not a list"}],
    )
    def test_a_basis_it_cannot_parse_is_no_basis(self, basis) -> None:
        assert Basis.from_dict(basis) is None

    def test_a_basis_survives_a_round_trip(self) -> None:
        basis = _basis()
        assert Basis.from_dict(basis.as_dict()) == basis

    def test_an_unreadable_file_is_absent_rather_than_an_error(self, tmp_path) -> None:
        path = tmp_path / "canonical.json"
        path.write_text("{not json", encoding="utf-8")
        assert measurement.load(path, _basis()) is None
        assert measurement.load(tmp_path / "nothing.json", _basis()) is None


class TestABasisStampedBeforeThePassHadAFingerprint:
    """Every measurement in `runs/` on the day `analysis` joined the basis.

    Those bases recorded no `analysis` field, and they must neither be thrown away nor
    trusted blindly. What they *did* record is a `metrics` fingerprint taken over a set
    that included `analysis.py`, so the pass is covered at a coarser grain: an unmoved
    `metrics` proves the pass did not move, and a moved one cannot say whether the pass
    was among what moved -- for which the only honest answer is the stronger repair.
    Reading the absence as stale would have replayed the whole corpus for an edit nobody
    made; reading it as current would be the mistake `tables` was built to refuse.
    """

    @staticmethod
    def _old_shape(**overrides) -> Basis:
        data = _basis(**overrides).as_dict()
        del data["analysis"]
        parsed = Basis.from_dict(data)
        assert parsed is not None
        return parsed

    def test_it_still_parses_and_records_no_pass(self) -> None:
        assert self._old_shape().analysis == ""

    def test_it_is_current_while_its_metrics_match(self) -> None:
        assert self._old_shape().stale_against(_basis()).current

    def test_a_moved_metrics_takes_the_stronger_repair_and_says_why(self) -> None:
        stale = self._old_shape(metrics="moved").stale_against(_basis())
        assert stale.needs == "reanalyse"
        assert len(stale.reasons) == 1
        assert "does not record whether the analysis pass" in stale.summary()

    def test_a_moved_engine_is_still_a_replay(self) -> None:
        stale = self._old_shape(engine="moved", metrics="moved").stale_against(_basis())
        assert stale.needs == "replay"

    def test_a_basis_stamped_now_records_the_pass(self) -> None:
        current = Basis.current("abc123", "v9", ("random",), 4, (0,), tables="t")
        assert current.analysis
        assert current.analysis == measurement.source_fingerprint("analysis")
        assert Basis.from_dict(current.as_dict()) == current


class TestStalenessReadsAsWhatItIs:
    def test_a_current_one_says_nothing(self) -> None:
        assert Staleness().summary() == ""
        assert Staleness().current
        assert not Staleness()

    def test_it_names_the_cost_in_the_summary(self) -> None:
        assert "re-score" in Staleness(("a reason",), "rescore").summary()
        assert "re-analyse" in Staleness(("a reason",), "reanalyse").summary()
        assert "replay" in Staleness(("a reason",), "replay").summary()


class TestTheTablesAreWhatAGameRestsOn:
    """The per-ruleset fingerprint, and the replay it exists to stop being charged.

    `ENGINE_SOURCES` used to be `chess_variant_sandbox/engine/*.py` plus `chess_variant_sandbox/agents/*.py`, so any
    edit anywhere under `engine/` invalidated every stored measurement in `runs/`.
    Schema 6 paid that in full -- forty-seven games replayed and rescored to reproduce
    numbers its own inertness evidence had already said were correct, because
    `rules.obstacles` has no entry in `build_tables` at all.

    What the architecture actually claims is that a game is a function of the flattened
    tables plus the code that walks them. These tests hold that claim to its word, and
    the sharp direction is again the permissive one: a fingerprint that failed to notice
    a table changing would report a game as current after its rules moved, which is worse
    than any number of spurious replays.
    """

    @staticmethod
    def _spec(name: str) -> dict:
        from chess_variant_sandbox.spec import load_spec

        return load_spec(measurement.ROOT / "rulesets" / f"{name}.json")

    def test_the_compiler_is_not_in_either_source_set(self) -> None:
        """The whole change, stated as the property it buys.

        `engine/compile.py` interprets the rule DSL, and what it interprets it *into* is
        fingerprinted per ruleset instead. If it ever reappears in a source set, every
        edit to the compiler charges a corpus replay again, and this test is the only
        thing that would say so.
        """
        compiler = measurement.ROOT / "chess_variant_sandbox" / "engine" / "compile.py"
        assert compiler.exists()
        for pattern in (
            measurement.ENGINE_SOURCES
            + measurement.METRICS_SOURCES
            + measurement.ANALYSIS_SOURCES
        ):
            assert compiler not in set(measurement.ROOT.glob(pattern)), pattern

    def test_every_boundary_key_is_classified_as_inert_or_meaningful(self) -> None:
        """A new field on the boundary is a named failure here, not a silent replay.

        This is the invariant that makes the canonicalisation trustworthy rather than
        merely present. `_INERT_DEFAULTS` and `_ALWAYS_MEANINGFUL` must partition what
        `build_tables` returns: a key in neither would be hashed unconditionally, so
        adding it at its inert default would move every ruleset's fingerprint at once
        and hand back the corpus replay this is built to remove.
        """
        from chess_variant_sandbox.engine import bridge
        from chess_variant_sandbox.engine.compile import compile_spec

        keys = set(bridge.build_tables(compile_spec(self._spec("chess"))))
        inert = set(bridge._INERT_DEFAULTS)
        assert not (inert & bridge._ALWAYS_MEANINGFUL), "a key cannot be both"
        assert keys == inert | bridge._ALWAYS_MEANINGFUL

    def test_a_new_key_at_its_inert_default_moves_no_fingerprint(self) -> None:
        """The property, exercised the way it will actually happen.

        Stage B adds fields for facing and stage C adds them for NPCs. Each arrives at a
        default that means "this ruleset does not use it", and every ruleset that does
        not use it has to keep the fingerprint it had.
        """
        from chess_variant_sandbox.engine import bridge
        from chess_variant_sandbox.engine.compile import compile_spec

        compiled = compile_spec(self._spec("chess"))
        before = bridge.canonical_tables(compiled)

        real_build = bridge.build_tables
        try:
            bridge.build_tables = lambda spec: {
                **real_build(spec), "has_facing": False, "facing_frames": [],
            }
            bridge._INERT_DEFAULTS["has_facing"] = False
            bridge._INERT_DEFAULTS["facing_frames"] = []
            assert bridge.canonical_tables(compiled) == before
        finally:
            bridge.build_tables = real_build
            del bridge._INERT_DEFAULTS["has_facing"]
            del bridge._INERT_DEFAULTS["facing_frames"]

    def test_a_rule_that_changes_the_tables_changes_the_fingerprint(self) -> None:
        """The other direction, which is the one that must never be permissive.

        Two rules, chosen because they reach the tables by opposite routes: a wall is
        resolved away into shorter rays, and a strike crosses as its own array.
        """
        chess = self._spec("chess")
        plain = measurement.tables_fingerprint(chess)

        walled = {**chess, "schema_version": 6,
                  "rules": {**chess["rules"],
                            "obstacles": {"squares": ["d4", "e5"]}}}
        assert measurement.tables_fingerprint(walled) != plain

        striking = {**chess, "schema_version": 6,
                    "rules": {**chess["rules"],
                              "strike": {"effect": "demote", "position": "stay",
                                         "immune": ["K"],
                                         "chain": [{"from": "Q", "to": "R"}]}}}
        assert measurement.tables_fingerprint(striking) != plain

    def test_a_spelled_out_default_compiles_to_the_game_that_was_already_there(self) -> None:
        """`{"effect": "capture"}` compiles to `None`, and the tables say so.

        The Zobrist draw is seeded from the game's identity, which since Q1008a is the
        hash of its normal form (or the name it already had): the spelled-out default is
        dropped there, so the two spellings are one game with one name, and their tables
        are identical down to the draw. The claim is checked one layer under the seed
        first -- what the compiler *did* with the rule -- and then whole.
        """
        from chess_variant_sandbox.engine.bridge import canonical_tables
        from chess_variant_sandbox.engine.compile import compile_spec
        from chess_variant_sandbox.spec.canonical import spec_hash

        chess = self._spec("chess")
        spelled = {**chess, "schema_version": 6,
                   "rules": {**chess["rules"], "strike": {"effect": "capture"}}}
        assert spec_hash(spelled) == spec_hash(chess)
        assert compile_spec(spelled).strike is None

        def without_the_draw(spec: dict) -> dict:
            return {k: v for k, v in canonical_tables(compile_spec(spec)).items()
                    if not k.startswith("zobrist_")}

        assert without_the_draw(spelled) == without_the_draw(chess)
        assert canonical_tables(compile_spec(spelled)) == canonical_tables(compile_spec(chess))
        assert measurement.tables_fingerprint(spelled) == measurement.tables_fingerprint(chess)

    def test_it_is_the_same_answer_twice(self) -> None:
        """Determinism, which a dict of arrays does not get for free.

        Computed from a fresh compile rather than from the cache, or this would be
        testing the dictionary.
        """
        chess = self._spec("chess")
        first = measurement.tables_fingerprint(chess)
        measurement._TABLES.clear()
        assert measurement.tables_fingerprint(chess) == first

    def test_two_games_are_two_fingerprints(self) -> None:
        assert measurement.tables_fingerprint(
            self._spec("chess")
        ) != measurement.tables_fingerprint(self._spec("gardner"))


class TestNothingIsAnAnswer:
    """The third thing `needs` can say, and the one that was unreachable before.

    Every change under `engine/` used to mean "replay", for every ruleset, whether or not
    it changed what any of them compiles to. "Nothing" is now the expected answer to a
    compiler change that moves no table, and it is the whole return on this item.
    """

    def test_a_basis_that_matches_needs_nothing(self) -> None:
        assert _basis().stale_against(_basis()).needs == ""

    def test_a_basis_written_before_the_tables_were_fingerprinted_is_unknown(self) -> None:
        """Absent is not equal to inert, and must not be read as it.

        Every measurement stored before this field existed lacks it. Defaulting it to
        the current value would let exactly the runs this was built to distinguish pass
        as current, so the field is required and its absence is `UNKNOWN` -- replay.
        """
        older = {k: v for k, v in _basis().as_dict().items() if k != "tables"}
        assert Basis.from_dict(older) is None


def _committed_specs() -> list[dict]:
    """Every committed and archived ruleset, as `corpus_hashes` finds them."""
    from chess_variant_sandbox.spec import load_spec

    found = []
    for directory in ("rulesets", "archive"):
        for path in sorted((measurement.ROOT / directory).glob("*.json")):
            try:
                spec = load_spec(path)
            except (OSError, ValueError):
                continue
            if isinstance(spec, dict):
                found.append(spec)
    return found


class _Compiles:
    """`compile_spec`, counted, standing in wherever `tables_fingerprint` reaches it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import chess_variant_sandbox.engine.compile as compile_module

        self.calls = 0
        real = compile_module.compile_spec

        def counted(spec):
            self.calls += 1
            return real(spec)

        monkeypatch.setattr(compile_module, "compile_spec", counted)


def _cold() -> None:
    """What a fresh process holds: nothing in memory, the file on disk untouched."""
    measurement._TABLES.clear()
    measurement._DISK.update(path=None, key=None, tables={})
    measurement._VALUES.clear()
    measurement._VALUES_DISK.update(path=None, key=None, tables={})
    measurement._LAST_COMPILED.clear()


class TestTheTablesCacheOnDisk:
    """R9 of `docs/reference/perf-survey-2026-09.md`: compile once per source tree, not per process.

    `measure --check` and a cold listing compiled every ruleset in the corpus to answer
    "did anything change", which was 91% of the check. The cache is only worth having if it
    can never answer for a compiler that is not the one on disk, so most of what is held
    here is that direction.
    """

    @pytest.fixture(autouse=True)
    def own_cache(self, tmp_path, monkeypatch) -> Path:
        path = tmp_path / "tables-fingerprints.json"
        monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(path))
        _cold()
        yield path
        _cold()

    @staticmethod
    def _spec(name: str) -> dict:
        return json.loads((measurement.ROOT / "rulesets" / f"{name}.json").read_text())

    def test_a_cold_process_compiles_nothing_the_last_one_compiled(
        self, own_cache, monkeypatch
    ) -> None:
        specs = [self._spec(name) for name in ("chess", "gardner", "dull")]
        compiles = _Compiles(monkeypatch)
        first = [measurement.tables_fingerprint(spec) for spec in specs]
        assert compiles.calls == len(specs)
        assert own_cache.exists()

        _cold()
        again = [measurement.tables_fingerprint(spec) for spec in specs]
        assert again == first
        assert compiles.calls == len(specs)

    def test_measure_check_compiles_nothing_when_nothing_changed(
        self, own_cache, tmp_path, monkeypatch, capsys
    ) -> None:
        """The acceptance, end to end: the command, twice, the second from cold."""
        from chess_variant_sandbox import cli, corpus
        from chess_variant_sandbox.spec import spec_hash

        runs = tmp_path / "runs"
        path = measurement.ROOT / "rulesets" / "gardner.json"
        spec = self._spec("gardner")
        digest = spec_hash(spec)
        out = corpus.measurement_dir(digest, runs)
        out.mkdir(parents=True)
        basis = corpus.current_basis(digest, spec=spec).as_dict()
        (out / "canonical.json").write_text(json.dumps({
            "hash": digest, "name": "gardner", "agents": list(corpus.CANONICAL_AGENTS),
            "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(basis["seeds"]),
            "games": 4, "index": 0.5, "totals_used": 1, "sd": 0.0, "viable": True,
            "violations": [], "weights_version": basis["weights_version"],
            "families": {}, "standings": [], "runs": [], "basis": basis,
        }), encoding="utf-8")
        argv = ["measure", str(path), "--check", "--runs", str(runs)]

        own_cache.unlink()
        _cold()
        compiles = _Compiles(monkeypatch)
        first = cli.main(argv)
        said = capsys.readouterr().out
        assert compiles.calls == 1

        _cold()
        assert cli.main(argv) == first
        assert capsys.readouterr().out == said
        assert compiles.calls == 1

    def test_an_answer_under_other_sources_is_never_read(
        self, own_cache, monkeypatch
    ) -> None:
        """What an edit to the compiler does: the key moves, and the stored answer is dead.

        The stored answer is planted wrong on purpose, so being served it is visible.
        """
        from chess_variant_sandbox.spec import spec_hash

        spec = self._spec("gardner")
        right = measurement.tables_fingerprint(spec)
        own_cache.write_text(json.dumps({
            "sources": "some other tree", "tables": {spec_hash(spec): "0" * 16},
        }))
        _cold()
        compiles = _Compiles(monkeypatch)
        assert measurement.tables_fingerprint(spec) == right
        assert compiles.calls == 1

    def test_a_planted_answer_under_these_sources_is_what_is_read(
        self, own_cache
    ) -> None:
        """The control for the test above: the file is really consulted."""
        from chess_variant_sandbox.spec import spec_hash

        spec = self._spec("gardner")
        own_cache.write_text(json.dumps({
            "sources": measurement.compile_source_key(),
            "tables": {spec_hash(spec): "f" * 16},
        }))
        _cold()
        assert measurement.tables_fingerprint(spec) == "f" * 16

    def test_an_unreadable_file_is_a_miss_and_is_rewritten(self, own_cache) -> None:
        own_cache.write_text("{half a file")
        spec = self._spec("gardner")
        answer = measurement.tables_fingerprint(spec)
        stored = json.loads(own_cache.read_text())
        assert stored["sources"] == measurement.compile_source_key()
        assert answer in stored["tables"].values()

    def test_the_sources_are_every_file_under_engine_and_spec(self) -> None:
        sources = set(measurement.compile_sources())
        for directory in ("engine", "spec"):
            for path in (measurement.ROOT / "chess_variant_sandbox" / directory).rglob("*.py"):
                assert path.resolve() in sources, path
        # The entry points, the package they run first, and the file that turns tables
        # into a number.
        package = measurement.ROOT / "chess_variant_sandbox"
        for name in ("engine/compile.py", "engine/bridge.py", "__init__.py", "measurement.py"):
            assert (package / name).resolve() in sources, name

    def test_the_closure_follows_an_import_out_of_the_two_directories(
        self, tmp_path, monkeypatch
    ) -> None:
        """A helper the compiler reads from somewhere new is followed, not missed."""
        package = tmp_path / "chess_variant_sandbox"
        (package / "engine").mkdir(parents=True)
        (package / "spec").mkdir()
        (package / "helpers").mkdir()
        (package / "__init__.py").write_text("")
        (package / "engine" / "compile.py").write_text(
            "def f():\n    from chess_variant_sandbox.helpers import grid, other\n"
        )
        (package / "engine" / "bridge.py").write_text(
            "import chess_variant_sandbox.helpers.far\n"
        )
        (package / "helpers" / "__init__.py").write_text("")
        (package / "helpers" / "grid.py").write_text("")
        (package / "helpers" / "far.py").write_text("")
        (package / "helpers" / "unread.py").write_text("")
        monkeypatch.setattr(measurement, "ROOT", tmp_path)

        names = {path.relative_to(tmp_path).as_posix() for path in measurement.compile_sources()}
        assert {
            "chess_variant_sandbox/helpers/__init__.py",
            "chess_variant_sandbox/helpers/grid.py",
            "chess_variant_sandbox/helpers/far.py",
        } <= names
        assert "chess_variant_sandbox/helpers/unread.py" not in names

    def test_editing_any_source_moves_the_key(self, tmp_path, monkeypatch) -> None:
        """Every file the key reads, one byte at a time: each one misses the cache."""
        import shutil

        real_root = measurement.ROOT
        files = [path.relative_to(real_root) for path in measurement.compile_sources()]
        for relative in files:
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(real_root / relative, target)
        monkeypatch.setattr(measurement, "ROOT", tmp_path)

        def key() -> str:
            measurement.compile_source_key.cache_clear()
            return measurement.compile_source_key()

        try:
            before = key()
            assert [p.relative_to(tmp_path) for p in measurement.compile_sources()] == files
            for relative in files:
                target = tmp_path / relative
                body = target.read_bytes()
                target.write_bytes(body + b"\n")
                assert key() != before, relative
                target.write_bytes(body)
            assert key() == before
        finally:
            monkeypatch.undo()
            measurement.compile_source_key.cache_clear()

    @pytest.mark.slow
    def test_cached_equals_fresh_for_every_committed_and_archived_ruleset(
        self, own_cache, monkeypatch
    ) -> None:
        specs = _committed_specs()
        assert len(specs) > 40
        fresh = [measurement.tables_fingerprint(spec) for spec in specs]
        _cold()
        compiles = _Compiles(monkeypatch)
        assert [measurement.tables_fingerprint(spec) for spec in specs] == fresh
        assert compiles.calls == 0


def _ast_module(dotted: str) -> Path | None:
    """The file a dotted name imports, spelled exactly, or ``None`` for a name in a module.

    The independent half of `TestTheImportClosure`: `os.listdir`, not `measurement`'s
    listing, so the two share no code.
    """
    *parents, last = dotted.split(".")
    folder = measurement.ROOT.joinpath(*parents)
    try:
        names = set(os.listdir(folder))
    except OSError:
        return None
    if f"{last}.py" in names:
        return (folder / f"{last}.py").resolve()
    if last in names and "__init__.py" in os.listdir(folder / last):
        return (folder / last / "__init__.py").resolve()
    return None


def _ast_imports(path: Path) -> set[Path]:
    """Every package file one source's import statements reach, read off its syntax tree,
    the parent packages each import runs included."""
    package = path.parent.relative_to(measurement.ROOT.resolve()).parts
    found: set[Path] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = ".".join(package[: len(package) - node.level + 1])
                module = f"{base}.{module}" if module else base
            names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for dotted in names:
            parts = dotted.split(".")
            if parts[0] != "chess_variant_sandbox":
                continue
            for end in range(1, len(parts) + 1):
                target = _ast_module(".".join(parts[:end]))
                if target is not None:
                    found.add(target)
    return found


def _ast_closure(entries: tuple[str, ...]) -> set[Path]:
    queue = []
    for entry in entries:
        path = (measurement.ROOT / entry).resolve()
        queue.append(path)
        parts = path.parent.relative_to(measurement.ROOT.resolve()).parts
        queue += [
            target
            for end in range(1, len(parts) + 1)
            if (target := _ast_module(".".join(parts[:end]))) is not None
        ]
    found: set[Path] = set()
    while queue:
        path = queue.pop()
        if path not in found:
            found.add(path)
            queue += _ast_imports(path)
    return found


class TestTheImportClosure:
    """`_closure` finds every package import a file makes, as the syntax tree reads them.

    Q887 found the old pattern's `[\\w\\s,]+` after `import` running on across the newline,
    so of two consecutive `from chess_variant_sandbox... import` lines it saw only the
    first, and `values_sources` lacked twelve files (Q889).
    """

    def _tree(self, tmp_path, monkeypatch, files: dict[str, str]) -> set[str]:
        for name, body in files.items():
            target = tmp_path / "chess_variant_sandbox" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        monkeypatch.setattr(measurement, "ROOT", tmp_path)
        found = measurement._closure(("chess_variant_sandbox/entry.py",))
        assert found == _ast_closure(("chess_variant_sandbox/entry.py",))
        package = (tmp_path / "chess_variant_sandbox").resolve()
        return {path.relative_to(package).as_posix() for path in found}

    def test_every_form_of_import_is_followed(self, tmp_path, monkeypatch) -> None:
        found = self._tree(tmp_path, monkeypatch, {
            "__init__.py": "",
            "entry.py": (
                "from chess_variant_sandbox.a import one\n"
                "from chess_variant_sandbox.b import two\n"
                "from chess_variant_sandbox.c import (\n"
                "    three,  # a comment, with a comma\n"
                "    four as quatre,\n"
                ")\n"
                "import chess_variant_sandbox.d.deep as deep, chess_variant_sandbox.e\n"
                "import os; from chess_variant_sandbox.f import six\n"
                "from chess_variant_sandbox.g import \\\n    seven\n"
                "def later():\n"
                "    from chess_variant_sandbox.h import eight\n"
                "    from . import nine\n"
                "from .pkg import ten\n"
            ),
            "a.py": "", "b.py": "",
            "c/__init__.py": "", "c/three.py": "", "c/four.py": "",
            "d/__init__.py": "", "d/deep.py": "", "e.py": "", "f.py": "",
            "g/__init__.py": "", "g/seven.py": "", "h.py": "", "nine.py": "",
            "pkg/__init__.py": "from .inner import x\nfrom .. import eleven\n",
            "pkg/inner.py": "", "eleven.py": "",
            "unread.py": "",
        })
        assert found == {
            "__init__.py", "entry.py", "a.py", "b.py",
            "c/__init__.py", "c/three.py", "c/four.py", "d/__init__.py", "d/deep.py",
            "e.py", "f.py", "g/__init__.py", "g/seven.py", "h.py", "nine.py",
            "pkg/__init__.py", "pkg/inner.py", "eleven.py",
        }

    def test_a_name_is_a_module_only_spelled_exactly(self, tmp_path, monkeypatch) -> None:
        """`Path.is_file` is case-blind on Windows: `import Grid` (a class) is no `grid.py`."""
        found = self._tree(tmp_path, monkeypatch, {
            "__init__.py": "",
            "entry.py": "from chess_variant_sandbox.helpers import Grid\n",
            "helpers/__init__.py": "", "helpers/grid.py": "",
        })
        assert found == {"__init__.py", "entry.py", "helpers/__init__.py"}

    def test_prose_and_comments_import_nothing(self, tmp_path, monkeypatch) -> None:
        found = self._tree(tmp_path, monkeypatch, {
            "__init__.py": "",
            "entry.py": (
                "# from chess_variant_sandbox import a\n"
                "x = 1  # import chess_variant_sandbox.b\n"
            ),
            "a.py": "", "b.py": "",
        })
        assert found == {"__init__.py", "entry.py"}

    @pytest.mark.parametrize("entries", ["_COMPILE_ENTRIES", "_VALUES_ENTRIES"])
    def test_every_closure_is_the_syntax_trees(self, entries: str) -> None:
        """Every closure `measurement` builds, against the walk of the package's syntax."""
        starts = getattr(measurement, entries)
        assert measurement._closure(starts) == _ast_closure(starts)

    def test_the_values_sources_hold_what_the_old_pattern_missed(self) -> None:
        names = {path.relative_to(measurement.ROOT).as_posix()
                 for path in measurement.values_sources()}
        for missed in ("agents/heuristics.py", "agents/minimax.py", "agents/objective.py",
                       "fairy/variant.py", "fairy/verify.py"):
            assert f"chess_variant_sandbox/{missed}" in names, missed

    @pytest.mark.slow
    def test_every_package_file_imports_what_its_syntax_tree_says(self) -> None:
        """Not only the closures' files: any file a future entry reaches (about 5 s here)."""
        disagree = []
        for path in sorted((measurement.ROOT / "chess_variant_sandbox").rglob("*.py")):
            path = path.resolve()
            ours = {
                target.resolve()
                for dotted in measurement.package_imports(
                    path.read_bytes(), measurement._package_of(path)
                )
                if (target := measurement._module_file(dotted)) is not None
            }
            if ours != _ast_imports(path):
                disagree.append(path.relative_to(measurement.ROOT).as_posix())
        assert not disagree, disagree
