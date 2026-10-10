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
from chess_variant_sandbox.agents import fitted
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.analysis import analyse_corpus
from chess_variant_sandbox.tournament.heuristics import head_to_head
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

REPOSITORY = Path(__file__).resolve().parents[1]
RULESETS = REPOSITORY / "rulesets"

# The two specs the acceptance names, picked by what `fitted/` says about them: the first
# accepts its fitted objective and dull refuses it, so between them they exercise both
# arms of the verdict every later unit is handed.
#
# **The accepting one was dobutsu until Q437.** The refits `v11` A6 forced
# (`docs/reference/v11-calibration.md`, B1) refused dobutsu's fit at 0.1250 and left no
# committed spec under 64 squares with an accepted one. Kingofthehill is 8x8 and picked
# over the other two that accept (chess, racingkings) on cost and on margin. Its games are
# short (29 plies against chess's 82), so one seed of `corpus.measure`, fit included, took
# 31.3 s at one worker on core `0596a87d9f1daaab`, where dobutsu took 9.5 s. The smallest
# spec that accepts, `mutant-gardner-shuffle` (5x5), took 31.0 s and accepts at 0.5938.
# Kingofthehill accepts at 0.7083 over 48 games, about three standard errors clear of the
# bar rather than one, and a fresh fit on that core reached the verdict its committed store
# made on an older one. So it is the accepting spec least likely to flip under the next
# core change. If one does flip it, the assertion that it accepts says so, and `fitted/`
# lists the verdicts to choose from.
#
# **Kingofthehill flipped too**, at the Q281a6 refit on core `2468e338` (0.7083 to 0.2917,
# `docs/reference/replay-2026-09-29.md`), and the slow tier stayed red on it for a week
# because nothing read it (Q743). So the accepting spec is no longer a name: it is read from
# `fitted/` when the measurement runs (`_accepting`), the committed ruleset with the
# smallest board whose committed fit under today's procedure accepts, the larger margin
# breaking a tie. On 2026-10-05 that is gardner (5x5, 0.6146). A refit that flips it moves
# the choice to the next one rather than turning this red; a fresh fit that disagrees with
# the committed store still fails, below and in `test_the_sliced_verdict_is_...`.
ACCEPTS = "accepts"
REFUSES = "dull"


def _accepting() -> str:
    """The committed ruleset, by file stem, the slices test measures as the accepting one."""
    from chess_variant_sandbox.spec.canonical import spec_hash

    procedure = fitted.procedure_digest()
    found = []
    for path in sorted(RULESETS.glob("*.json")):
        spec = load_spec(path)
        verdict = fitted.load_fit(spec_hash(spec), procedure)
        if verdict is not None and verdict.accepted:
            area = int(spec["board"]["width"]) * int(spec["board"]["height"])
            found.append((area, -verdict.accept_score, path.stem))
    assert found, f"no committed ruleset accepts its fit under procedure {procedure}"
    return min(found)[2]


def _named(role: str) -> str:
    return _accepting() if role == ACCEPTS else role

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
# **Re-opened 2026-10-02 by Q631** (the players digest, `plan-compute.md` §2.3): Q281a6 and
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


class TestTheFitIsAnInputToEveryLaterUnit:
    """No `fitted/` store in a browser, so the verdict travels in the request."""

    @staticmethod
    def _unfitted(dull_spec) -> tuple[dict, str]:
        """A game with no verdict anywhere: dull with one piece worth something else.

        A piece value is inside the mechanics hash, so this is a different game to every
        reader, and `fitted/` has no file for it. That is the browser's situation for
        every draft, which is the case the request format exists for.
        """
        from chess_variant_sandbox.spec.canonical import spec_hash

        spec = dict(dull_spec)
        spec["pieces"] = [dict(piece) for piece in spec["pieces"]]
        spec["pieces"][0]["value"] = 7.5
        return spec, spec_hash(spec)

    def test_a_roster_carrying_a_fit_cannot_be_played_without_the_verdict(
        self, dull_spec, monkeypatch
    ) -> None:
        monkeypatch.setattr(fitted, "_FITTED", {})
        spec, digest = self._unfitted(dull_spec)
        assert not fitted.store_path(digest, fitted.procedure_digest()).exists()
        config = TournamentConfig(
            agents=(f"minimax:fit={fitted.procedure_digest()},nodes=100",),
            games_per_pairing=1,
        )
        unit = slices.playing_units(slices.TOURNAMENT, config, 1)[0]
        with pytest.raises(fitted.FitError):
            slices.run_request(slices.request(unit, spec, config=config))

    def test_the_same_unit_runs_once_the_verdict_is_in_the_request(
        self, dull_spec, monkeypatch
    ) -> None:
        monkeypatch.setattr(fitted, "_FITTED", {})
        spec, digest = self._unfitted(dull_spec)
        procedure = fitted.procedure_digest()
        verdict = fitted.FittedObjective(
            ruleset_hash=digest,
            procedure=procedure,
            vector=(0.0,) * len(fitted.VOCABULARY),
            accepted=False,
            accept_score=0.0,
            samples=0,
            loss=0.0,
        )
        config = TournamentConfig(
            agents=(f"minimax:fit={procedure},nodes=100",), games_per_pairing=1
        )
        unit = slices.playing_units(slices.TOURNAMENT, config, 1)[0]
        text = slices.run_request(
            slices.request(unit, spec, fit=verdict, config=config)
        )
        assert len(slices.parse_records(text)) == 1
        assert fitted.registered_fits()[(procedure, digest)] == verdict


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


@pytest.mark.slow
class TestTheAcceptanceMatchTotal:
    """The one piece of arithmetic this module restates rather than calls."""

    def test_it_equals_the_total_head_to_head_takes(self, dull_spec) -> None:
        config = TournamentConfig(
            agents=("greedy", "random"),
            games_per_pairing=2,
            seed=5,
            include_self_play=False,
            collect_series=False,
        )
        records = run_tournament(dull_spec, config).records
        assert slices.match_score(records, "greedy") == head_to_head(
            dull_spec, "greedy", "random", 2, 5, 0
        )


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """`corpus.measure` on each of the two rulesets, run once and read four times.

    The real canonical roster, at one thread, and at the one seed the comparison reads.
    A committed ruleset is measured at `EXPECTED_LIBRARY_SEEDS` by default, and until Q437
    this played all four and compared one. Both games are cheap to measure (`ACCEPTS` says
    how cheap), which is the only reason B0's acceptance can be a test rather than a run
    recorded in a document.
    """
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.spec.canonical import spec_hash

    work = tmp_path_factory.mktemp("slices-measured")
    done = {}
    for name in (ACCEPTS, REFUSES):
        spec = load_spec(RULESETS / f"{_named(name)}.json")
        digest = spec_hash(spec)
        corpus.measure(
            corpus.Candidate(
                hash=digest, name=str(spec["name"]), spec=spec,
                path=RULESETS / f"{_named(name)}.json", stage="committed",
            ),
            runs_dir=work / name,
            workers=1,
            seeds=(corpus.CANONICAL_SEED,),
        )
        done[name] = (
            spec,
            work / name / f"{digest}-{corpus.CANONICAL_SEED}",
            fitted.load_fit(digest, fitted.procedure_digest(), fitted.fit_store()),
        )
    return work, done


@pytest.mark.slow
@pytest.mark.parametrize("name", [ACCEPTS, REFUSES])
@pytest.mark.parametrize("count", [1, 3, 4])
def test_a_sliced_measurement_writes_what_corpus_measure_writes(
    name, count, measured
) -> None:
    """B0's acceptance: the canonical roster, one ruleset that accepts a fit and one
    that refuses, at one slice, three and four.

    Every phase runs, the fit included, so what is compared is a whole measurement and
    not a tournament with the hard part handed to it. `games.jsonl` is compared with
    `duration_ms` dropped, because a slice is timed separately from the whole; the other
    three files are compared as bytes.
    """
    work, done = measured
    spec, whole, fitted_verdict = done[name]
    assert fitted_verdict is not None
    assert fitted_verdict.accepted is (name == ACCEPTS), (
        f"{spec['name']}'s fit {'accepted' if fitted_verdict.accepted else 'refused'} at "
        f"{fitted_verdict.accept_score} on this core, where its committed store in fitted/ "
        "says otherwise (ACCEPTS is read from it): this core re-derives a different "
        "verdict, so the store owes a refit (`fit --check`) before this test means anything"
    )

    sliced = work / f"sliced-{name}-{count}"
    gathered = slices.measure_in_slices(
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

    # The fit phases reach the verdict `fit_ruleset` reached, which is the half of the
    # measurement no file above would have caught: a wrong vector plays a different
    # roster and would have shown up as different games, but a wrong *accept score* on
    # the right vector would not.
    assert gathered.fit.as_dict() == fitted_verdict.as_dict()

    # A sliced measurement is provisional: the file every reader trusts is not its to
    # write, and the basis stamp belongs where the result is re-derived.
    assert not (sliced / "canonical.json").exists()


@pytest.mark.slow
@pytest.mark.parametrize("name", [ACCEPTS, REFUSES])
def test_the_sliced_verdict_is_the_one_the_repository_committed(name, measured) -> None:
    """The fit reproduced from games, against the store `fitted/` tracks.

    `build` is left out: a committed verdict names the core it was made on, and this one
    names the core in this checkout. Everything a player reads is compared.
    """
    _work, done = measured
    spec, _whole, made = done[name]
    from chess_variant_sandbox.spec.canonical import spec_hash

    committed = fitted.load_fit(spec_hash(spec), fitted.procedure_digest())
    if committed is None:
        pytest.skip(f"{name} has no committed fit under this procedure")
    assert made is not None
    assert {k: v for k, v in made.as_dict().items() if k != "build"} == {
        k: v for k, v in committed.as_dict().items() if k != "build"
    }
