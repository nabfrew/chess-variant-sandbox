"""Tier one, the legality job: a browser's measurement walked, re-derived and scored.

`docs/plan-client-compute.md` §8, the row *Tier one, the legality job*. Its acceptance, and
what holds each part:

* a submission with one illegal move, one move after a terminal position, one wrong
  outcome, one missing seed or one altered fit vector is refused by name --
  `TestTheFiveRefusals`, and `TestTheOtherRefusals` for the rest of the names;
* an honest submission is scored to the server's own number -- `TestTheHonestCase` on a
  small roster, and `test_an_honest_chess_submission_scores_to_the_servers_number`, `slow`,
  on chess's whole canonical measurement;
* the walk rebuilds the series and analysis fields as the core and the analysis pass do --
  `TestTheWalkReadsAsTheCoreAndThePassDo`;
* the slim upload is B0's gathered text, written and read here and nowhere else --
  `TestTheSlimForm`;
* the job moves a provisional result to `legal` and no further, and a refusal records
  nothing -- `TestTheJob`, which needs a real Postgres and skips with its reason without
  one, as every store test does.

The honest submission is made the way a browser makes one: every phase's units run through
B0's request text and their replies gathered in part order. `dull` is small and its fit
reaches the acceptance match, so both halves of the verdict are walked.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import site_store

from chess_variant_sandbox import corpus, measurement
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.fitted import FittedObjective, procedure_digest
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import LEGAL, Basis
from chess_variant_sandbox.site import db, legality, provisional
from chess_variant_sandbox.site.legality import Refused
from chess_variant_sandbox.site.provisional import ProvisionalError
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.tournament.runner import TournamentConfig

REPOSITORY = Path(__file__).resolve().parents[1]
RULESETS = REPOSITORY / "rulesets"
FILES = ("games.jsonl", "tournament.json", "analysis.json")


# --------------------------------------------------------- a browser's measurement


@dataclass
class Measured:
    """A measurement as a browser holds it when it finishes, and what it scored to."""

    spec: dict[str, Any]
    roster: TournamentConfig
    current: Basis
    verdict: FittedObjective
    phases: dict[str, str]
    index: float
    viable: bool
    files: dict[str, str]

    @property
    def player(self) -> str:
        """Which objective the fitted rungs played: the digest of the browser's own fit."""
        return corpus.player_digest(self.verdict)

    @property
    def stamp(self) -> Basis:
        """What the browser stamps: the server's basis, played in a browser, with its fit."""
        return replace(self.current, fit=self.player).in_browser()

    def upload(self, **changed: Any) -> str:
        """The slim upload, with any phase's text (by its name), the `verdict` or the
        `basis` replaced."""
        phases = {**self.phases, **{k: v for k, v in changed.items() if k in slices.PHASES}}
        return legality.upload(
            changed.get("basis", self.stamp),
            changed.get("verdict", self.verdict),
            phases,
        )

    def rows(self, phase: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.phases[phase].splitlines() if line]

    def check(self, text: str) -> legality.Checked:
        return legality.check(self.spec, legality.read_upload(text), self.current)


def lines(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def measure_as_a_browser(
    spec: dict[str, Any], roster: TournamentConfig, out: Path, parts: int = 2
) -> Measured:
    """Every phase's units through B0's request text, gathered in part order."""

    def phase(units: list[slices.Unit], fit: Any = None, records: Any = None) -> str:
        return "".join(
            slices.run_request(
                slices.request(unit, spec, fit=fit, records=records, config=roster)
            )
            for unit in units
        )

    fit_text = phase(slices.playing_units(slices.FIT, slices.fit_config(), parts))
    provisional_fit = slices.gather_fit(spec, slices.parse_records(fit_text))
    accept_text = ""
    verdict = provisional_fit
    if slices.needs_a_match(provisional_fit):
        accept_text = phase(
            slices.playing_units(
                slices.ACCEPT, slices.accept_config(provisional_fit.procedure), parts
            ),
            fit=provisional_fit,
        )
        verdict = slices.gather_verdict(provisional_fit, slices.parse_records(accept_text))
    tournament_text = phase(slices.playing_units(slices.TOURNAMENT, roster, parts), fit=verdict)
    records = slices.parse_records(tournament_text)
    analysis_text = phase(slices.analysis_units(records, parts), fit=verdict, records=records)
    blunder_text = phase(
        slices.blunder_units(records, parts, seed=roster.seed), fit=verdict, records=records
    )
    gathered = slices.gather_measurement(
        spec,
        out,
        verdict,
        records,
        slices.gather_analysis(
            slices.parse_analyses(analysis_text), slices.parse_blunders(blunder_text)
        ),
        config=roster,
    )
    digest = spec_hash(spec)
    current = Basis.current(
        ruleset_hash=digest,
        weights_version=gathered.result.weights_version,
        agents=roster.agents,
        games_per_pairing=roster.games_per_pairing,
        seeds=(roster.seed,),
        tables=measurement.tables_fingerprint(spec),
        verdict_rule=corpus.VERDICT_RULE,
    )
    return Measured(
        spec=spec,
        roster=roster,
        current=current,
        verdict=verdict,
        phases={
            slices.FIT: fit_text,
            slices.ACCEPT: accept_text,
            slices.TOURNAMENT: tournament_text,
            slices.ANALYSIS: analysis_text,
            slices.BLUNDER: blunder_text,
        },
        index=gathered.index,
        viable=bool(gathered.result.viable),
        files={path.name: path.read_text(encoding="utf-8") for path in out.iterdir()},
    )


def small_roster() -> TournamentConfig:
    """Both kinds of seat the pass tells apart, and a `fit=` player, two games a pairing."""
    return slices.canonical_config(
        agents=("random", "greedy", f"minimax:fit={procedure_digest()},nodes=100"),
        games_per_pairing=2,
        seed=0,
        workers=1,
    )


@pytest.fixture(scope="module")
def honest(tmp_path_factory: pytest.TempPathFactory) -> Measured:
    """Dull, measured once as a browser would, on a roster small enough to walk often."""
    spec = load_spec(RULESETS / "dull.json")
    return measure_as_a_browser(spec, small_roster(), tmp_path_factory.mktemp("browser"))


def refused(measured: Measured, text: str) -> Refused:
    with pytest.raises(Refused) as caught:
        measured.check(text)
    return caught.value


def a_long_game(measured: Measured) -> int:
    """The first tournament game long enough to change a move in its middle."""
    return next(i for i, row in enumerate(measured.rows(slices.TOURNAMENT)) if len(row["moves"]) > 6)


# ------------------------------------------------------------------ the honest case


class TestTheHonestCase:
    def test_it_scores_to_the_number_the_browser_gathered(self, honest: Measured) -> None:
        checked = honest.check(honest.upload())
        scored = legality.score(checked, honest.current.weights_version)
        assert scored.headline == honest.index
        assert scored.viable == honest.viable
        assert not scored.unscored

    def test_the_servers_files_are_the_browsers_to_the_byte(self, honest: Measured) -> None:
        """Every field the slim form left out came back, rebuilt from the positions."""
        scored = legality.score(honest.check(honest.upload()), honest.current.weights_version)
        index = f"index-{honest.current.weights_version}.json"
        for name in (*FILES, index):
            assert scored.files[name] == honest.files[name], name

    def test_the_fit_travels_beside_the_four_files(self, honest: Measured) -> None:
        scored = legality.score(honest.check(honest.upload()), honest.current.weights_version)
        assert FittedObjective.from_dict(json.loads(scored.files[provisional.FIT])) == honest.verdict
        assert scored.files[provisional.FIT_GAMES] == honest.phases[slices.FIT]
        assert scored.files[provisional.ACCEPT_GAMES] == honest.phases[slices.ACCEPT]
        assert all(provisional.RECORD_FILES.match(name) for name in scored.files)
        assert "canonical.json" not in scored.files

    def test_the_verdict_it_re_derives_is_the_browsers(self, honest: Measured) -> None:
        checked = honest.check(honest.upload())
        assert checked.fit == honest.verdict
        assert checked.accept_records, "dull's fit reaches its acceptance match"


# ---------------------------------------------------------------- the five refusals


class TestTheFiveRefusals:
    """The row's acceptance: each of the five is refused by its own name."""

    def test_one_illegal_move(self, honest: Measured) -> None:
        rows = honest.rows(slices.TOURNAMENT)
        game = a_long_game(honest)
        ply = 3
        engine = Engine(honest.spec)
        state = engine.initial_state()
        for move in rows[game]["moves"][:ply]:
            state = engine.apply(state, move)
        legal = engine.legal_moves(state)
        rows[game]["moves"][ply] = next(m for m in itertools.count() if m not in legal)
        found = refused(honest, honest.upload(tournament=lines(rows)))
        assert found.name == legality.ILLEGAL_MOVE
        assert (found.phase, found.game, found.ply) == (slices.TOURNAMENT, game, ply)
        assert f"at ply {ply}" in str(found)

    def test_one_move_after_a_terminal_position(self, honest: Measured) -> None:
        rows = honest.rows(slices.TOURNAMENT)
        ended = len(rows[0]["moves"])
        rows[0]["moves"].append(rows[0]["moves"][-1])
        rows[0]["plies"] += 1
        found = refused(honest, honest.upload(tournament=lines(rows)))
        assert found.name == legality.AFTER_TERMINAL
        assert (found.game, found.ply) == (0, ended)
        assert "after the game ended" in str(found)

    def test_one_wrong_outcome(self, honest: Measured) -> None:
        for change in ("winner", "reason"):
            rows = honest.rows(slices.TOURNAMENT)
            game = rows[1]
            if change == "winner":
                game["winner"] = 0 if game["winner"] is None else None
            else:
                game["reason"] = "resignation"
            found = refused(honest, honest.upload(tournament=lines(rows)))
            assert found.name == legality.WRONG_OUTCOME, change
            assert found.game == 1
            assert "the submission says" in str(found)

    def test_one_missing_seed(self, honest: Measured) -> None:
        for phase in (slices.TOURNAMENT, slices.FIT, slices.ACCEPT):
            rows = honest.rows(phase)
            gone = rows.pop(3)
            found = refused(honest, honest.upload(**{phase: lines(rows)}))
            assert found.name == legality.MISSING_SEED, phase
            assert (found.phase, found.seed) == (phase, gone["seed"])
            assert f"missing seed {gone['seed']}" in str(found)

    def test_one_altered_fit_vector(self, honest: Measured) -> None:
        data = honest.verdict.as_dict()
        term = max(range(len(VOCABULARY)), key=lambda i: abs(data["weights"][i]))
        data["weights"][term] = round(data["weights"][term] + 0.000001, 6)
        found = refused(honest, honest.upload(verdict=data))
        assert found.name == legality.FIT_VECTOR
        assert VOCABULARY[term] in str(found)


class TestTheOtherRefusals:
    def test_moves_that_stop_before_the_game_ends(self, honest: Measured) -> None:
        rows = honest.rows(slices.TOURNAMENT)
        game = a_long_game(honest)
        rows[game]["moves"].pop()
        found = refused(honest, honest.upload(tournament=lines(rows)))
        assert found.name == legality.UNFINISHED and found.game == game

    def test_a_fit_game_is_walked_as_a_tournament_game_is(self, honest: Measured) -> None:
        rows = honest.rows(slices.FIT)
        rows[5]["winner"] = 0 if rows[5]["winner"] is None else None
        found = refused(honest, honest.upload(fit=lines(rows)))
        assert (found.name, found.phase, found.game) == (legality.WRONG_OUTCOME, slices.FIT, 5)

    def test_a_game_the_schedule_does_not_hold(self, honest: Measured) -> None:
        rows = honest.rows(slices.TOURNAMENT)
        rows[2]["seed"] = 10_000
        assert refused(honest, honest.upload(tournament=lines(rows))).name == legality.UNSCHEDULED
        rows = honest.rows(slices.TOURNAMENT)
        rows.append(rows[4])
        found = refused(honest, honest.upload(tournament=lines(rows)))
        assert found.name == legality.UNSCHEDULED and "both have seed" in str(found)

    def test_other_players_and_another_order(self, honest: Measured) -> None:
        rows = honest.rows(slices.TOURNAMENT)
        first = next(i for i, row in enumerate(rows) if row["agents"][0] != row["agents"][1])
        rows[first]["agents"].reverse()
        found = refused(honest, honest.upload(tournament=lines(rows)))
        assert (found.name, found.game) == (legality.WRONG_PLAYERS, first)
        rows = honest.rows(slices.TOURNAMENT)
        rows[0], rows[1] = rows[1], rows[0]
        assert refused(honest, honest.upload(tournament=lines(rows))).name == legality.OUT_OF_ORDER

    def test_a_verdict_that_is_not_the_matchs(self, honest: Measured) -> None:
        for key, value in (("accepted", not honest.verdict.accepted), ("accept_score", 0.99)):
            data = {**honest.verdict.as_dict(), key: value}
            found = refused(honest, honest.upload(verdict=data))
            assert found.name == legality.FIT_VERDICT and key in str(found)

    def test_an_acceptance_game_is_walked_too(self, honest: Measured) -> None:
        rows = honest.rows(slices.ACCEPT)
        rows[0]["moves"].append(rows[0]["moves"][-1])
        found = refused(honest, honest.upload(accept=lines(rows)))
        assert (found.name, found.phase) == (legality.AFTER_TERMINAL, slices.ACCEPT)

    def test_an_analysis_of_other_games_or_impossible_counts(self, honest: Measured) -> None:
        rows = honest.rows(slices.ANALYSIS)
        assert refused(honest, honest.upload(analysis=lines(rows[1:]))).name == legality.ANALYSIS
        rows = honest.rows(slices.ANALYSIS)
        rows[0]["near_best"][0] = 10_000
        found = refused(honest, honest.upload(analysis=lines(rows)))
        assert found.name == legality.ANALYSIS and "probe counts" in str(found)
        rows = honest.rows(slices.ANALYSIS)
        rows[0]["near_best"].append(1)
        assert refused(honest, honest.upload(analysis=lines(rows))).name == legality.ANALYSIS

    def test_trials_the_sampler_did_not_draw(self, honest: Measured) -> None:
        rows = honest.rows(slices.BLUNDER)
        assert rows, "the small roster's games are long enough to be sampled"
        assert refused(honest, honest.upload(blunder=lines(rows[1:]))).name == legality.BLUNDER
        rows = honest.rows(slices.BLUNDER)
        rows[0]["seed"] += 1
        assert refused(honest, honest.upload(blunder=lines(rows))).name == legality.BLUNDER
        rows = honest.rows(slices.BLUNDER)
        rows[0]["new_winner"] = 2
        assert refused(honest, honest.upload(blunder=lines(rows))).name == legality.BLUNDER

    def test_a_stale_module_is_refused_by_its_build_before_any_walk(
        self, honest: Measured
    ) -> None:
        sources = honest.current.engine.rpartition("-")[0]
        stale = replace(honest.stamp, engine=f"{sources}-0123456789abcdef")
        found = refused(honest, honest.upload(basis=stale))
        assert found.name == legality.STALE_BUILD
        assert "build 0123456789abcdef" in str(found) and "fetches the current module" in str(found)

    def test_a_basis_for_another_roster_is_refused(self, honest: Measured) -> None:
        other = replace(honest.stamp, games_per_pairing=4)
        assert refused(honest, honest.upload(basis=other)).name == legality.BASIS

    def test_what_cannot_be_read(self, honest: Measured) -> None:
        text = honest.upload()
        data = json.loads(text)
        cases = {
            "not JSON": "{",
            "form": json.dumps({**data, "schema": 2}),
            "lacks the phases": json.dumps({**data, "phases": {"fit": ""}}),
            "no measurement has": json.dumps(
                {**data, "phases": {**data["phases"], "extra": ""}}
            ),
        }
        for said, bad in cases.items():
            with pytest.raises(Refused, match=said) as caught:
                legality.read_upload(bad)
            assert caught.value.name == legality.UNREADABLE
        rows = honest.rows(slices.TOURNAMENT)
        rows[0]["moves"][0] = "e2e4"
        found = refused(honest, honest.upload(tournament=lines(rows)))
        assert found.name == legality.UNREADABLE and "whole numbers" in str(found)
        rows = honest.rows(slices.TOURNAMENT)
        rows[0]["winner"] = True
        assert refused(honest, honest.upload(tournament=lines(rows))).name == legality.UNREADABLE

    def test_every_name_is_listed(self) -> None:
        assert len(set(legality.NAMES)) == len(legality.NAMES)
        assert {
            legality.ILLEGAL_MOVE,
            legality.AFTER_TERMINAL,
            legality.WRONG_OUTCOME,
            legality.MISSING_SEED,
            legality.FIT_VECTOR,
        } <= set(legality.NAMES)


# ------------------------------------------------------------------- the stamp's fit

OTHER_PLAYER = "fit:0123456789ab"
"""A `Basis.fit` no regression of these games gives."""


class TestTheStampsFitIsCompared:
    """Q468: `Basis.fit` is held to what the server would play, as every field is."""

    def test_the_honest_stamp_carries_the_player_its_fit_gives(self, honest: Measured) -> None:
        checked = honest.check(honest.upload())
        assert checked.basis is not None
        assert checked.basis.fit == honest.player == corpus.player_digest(checked.fit)
        assert json.loads(honest.upload())["basis"]["fit"] == honest.player

    def test_a_stamp_that_does_not_say_is_refused_before_any_walk(
        self, honest: Measured
    ) -> None:
        silent = replace(honest.stamp, fit="")
        found = refused(honest, honest.upload(basis=silent))
        assert found.name == legality.BASIS and found.phase is None
        assert "which objective its fitted players played" in str(found)

    def test_a_stamp_whose_fit_its_games_do_not_give_is_refused(
        self, honest: Measured
    ) -> None:
        """A visitor's design has no fit store, so the regression is the only answer."""
        assert honest.current.fit_reading is None
        found = refused(honest, honest.upload(basis=replace(honest.stamp, fit=OTHER_PLAYER)))
        assert found.name == legality.BASIS and found.phase == slices.FIT
        said = str(found)
        assert f"it played the fitted objective {OTHER_PLAYER}" in said
        assert "plays now" in said

    def test_a_game_with_a_store_holds_the_stamp_to_it_before_any_walk(
        self, honest: Measured
    ) -> None:
        core = measurement.built_fingerprint()
        stored = replace(
            honest.current,
            fit=OTHER_PLAYER,
            fit_reading=measurement.FitReading(
                player=OTHER_PLAYER, made_on=core, verified_on=frozenset({core})
            ),
        )
        with pytest.raises(Refused) as caught:
            legality.check(honest.spec, legality.read_upload(honest.upload()), stored)
        assert caught.value.name == legality.BASIS and caught.value.phase is None
        assert f"the fitted objective {OTHER_PLAYER} plays now" in str(caught.value)
        # The stamp matches the store, and the walk's own regression still refuses it.
        matching = replace(honest.stamp, fit=OTHER_PLAYER)
        with pytest.raises(Refused) as caught:
            legality.check(honest.spec, legality.read_upload(honest.upload(basis=matching)), stored)
        assert caught.value.name == legality.BASIS and caught.value.phase == slices.FIT


# -------------------------------------------------------------------- the slim form


class TestTheSlimForm:
    def test_the_upload_leaves_out_what_the_walk_rebuilds(self, honest: Measured) -> None:
        data = json.loads(honest.upload())
        assert data["schema"] == legality.SCHEMA
        assert set(data["phases"]) == set(slices.PHASES)
        for line in data["phases"][slices.TOURNAMENT].splitlines():
            assert not set(legality.REBUILT_RECORD_FIELDS) & set(json.loads(line))
        for line in data["phases"][slices.ANALYSIS].splitlines():
            row = json.loads(line)
            assert not set(legality.REBUILT_ANALYSIS_FIELDS) & set(row)
            assert "near_best" in row
        assert data["phases"][slices.BLUNDER] == honest.phases[slices.BLUNDER]
        assert data["fit"] == honest.verdict.as_dict()

    def test_a_slim_line_is_still_b0s_line(self, honest: Measured) -> None:
        """No second parser: B0's own readers read the slim text."""
        data = json.loads(honest.upload())
        slim = slices.parse_records(data["phases"][slices.TOURNAMENT])
        full = slices.parse_records(honest.phases[slices.TOURNAMENT])
        assert [(r.seed, r.moves, r.winner, r.reason) for r in slim] == [
            (r.seed, r.moves, r.winner, r.reason) for r in full
        ]
        assert [a.near_best for a in slices.parse_analyses(data["phases"][slices.ANALYSIS])] == [
            a.near_best for a in slices.parse_analyses(honest.phases[slices.ANALYSIS])
        ]

    def test_a_line_that_keeps_the_rebuilt_fields_reads_the_same(
        self, honest: Measured
    ) -> None:
        data = json.loads(honest.upload())
        data["phases"] = dict(honest.phases)
        full = legality.score(
            honest.check(json.dumps(data)), honest.current.weights_version
        ).files
        slim = legality.score(honest.check(honest.upload()), honest.current.weights_version).files
        assert full == slim

    def test_forged_copies_of_rebuilt_fields_are_ignored(self, honest: Measured) -> None:
        """The walk's copies stand; a browser's copies of them never reach the score."""
        data = json.loads(honest.upload())
        tournament = honest.rows(slices.TOURNAMENT)
        for row in tournament:
            row["material"] = [99.0] * len(row["material"])
            row["branching"] = [1] * len(row["branching"])
        analyses = honest.rows(slices.ANALYSIS)
        for row in analyses:
            row["position_keys"] = [7] * len(row["position_keys"])
            row["revisited_plies"] = 0
        data["phases"] = {
            **honest.phases,
            slices.TOURNAMENT: lines(tournament),
            slices.ANALYSIS: lines(analyses),
        }
        scored = legality.score(honest.check(json.dumps(data)), honest.current.weights_version)
        assert scored.headline == honest.index
        assert scored.files["games.jsonl"] == honest.files["games.jsonl"]

    def test_the_slim_form_is_the_smaller(self, honest: Measured) -> None:
        """What it leaves out is most of the two phases that carry positions."""
        slim = json.loads(honest.upload())["phases"]
        rebuilt = (slices.TOURNAMENT, slices.ANALYSIS)
        assert sum(len(slim[p]) for p in rebuilt) < 0.6 * sum(
            len(honest.phases[p]) for p in rebuilt
        )


# ------------------------------------------------------------------------ the walk


class TestTheWalkReadsAsTheCoreAndThePassDo:
    """The walk restates two readings, and each is held to the one it restates."""

    @pytest.mark.parametrize("name", ["dobutsu", "dull"])
    def test_series_analysis_and_trials_match_the_core_and_the_pass(self, name: str) -> None:
        """Dobutsu drops pieces, so the drop half of the piece counting is read too."""
        spec = load_spec(RULESETS / f"{name}.json")
        engine = Engine(spec)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        epsilon = analysis_pass._epsilon_for(engine)
        records = [
            play_game(engine, pair, seed)
            for seed, pair in enumerate(
                [("greedy", "minimax:nodes=100"), ("minimax:nodes=100", "greedy")] * 3
            )
        ]
        for index, record in enumerate(records):
            walked = legality.walk(
                engine, evaluator, record, phase=slices.TOURNAMENT, game=index, series=True
            )
            assert walked.record.as_dict() == record.as_dict()
            own = analysis_pass._analyse_game(engine, evaluator, record, epsilon)
            assert replace(walked.positions, near_best=own.near_best) == own
            if record.plies >= 12:
                trial = analysis_pass._one_blunder(
                    engine, record, analysis_pass.blunder_rng(0, record)
                )
                assert (walked.blunder_choices >= 2) == (trial is not None)
                if trial is not None:
                    assert walked.blunder_side == trial.blundering_side
                    assert int(record.plies * analysis_pass.BLUNDER_FRACTION) == trial.ply

    def test_a_game_without_series_rebuilds_none(self, honest: Measured) -> None:
        checked = honest.check(honest.upload())
        assert all(not r.branching and not r.material for r in checked.fit_records)
        assert all(r.branching for r in checked.records if r.moves)


class TestTheModuleIsDownstream:
    def test_it_is_in_no_source_set(self) -> None:
        """No stamped fingerprint reads it, so building it staled no score."""
        itself = (REPOSITORY / "chess_variant_sandbox" / "site" / "legality.py").resolve()
        for kind in ("engine", "metrics", "analysis"):
            assert itself not in {p.resolve() for p in measurement.fingerprinted(kind)}


# -------------------------------------------------------------------------- the job


@dataclass
class Store:
    connection: Any
    runs: Path
    game: str
    designer: int
    other: int
    measured: Measured
    url: str = ""
    schema: str = ""

    def basis(self, game: str, spec: dict[str, Any]) -> Basis:
        assert spec_hash(spec) == game
        return self.measured.current

    def file(self, text: str | None = None, submitter: int | None = None) -> int:
        return legality.file(
            self.connection,
            self.runs,
            game=self.game,
            submitter=self.designer if submitter is None else submitter,
            text=self.measured.upload() if text is None else text,
            basis=self.basis,
        )

    def run(self, job: int) -> legality.Finished:
        finished = legality.run(self.connection, self.runs, job, basis=self.basis)
        self.connection.commit()
        return finished

    def job(self, job: int) -> tuple[str, str | None, str | None, str | None]:
        with self.connection.cursor() as cursor:
            cursor.execute("select state, error, log, fingerprint from jobs where id = %s", (job,))
            return tuple(cursor.fetchone())  # type: ignore[return-value]

    def results(self) -> list[tuple[int, str, int | None]]:
        with self.connection.cursor() as cursor:
            cursor.execute("select id, state, job from provisional order by id")
            return [(int(r[0]), str(r[1]), r[2]) for r in cursor.fetchall()]


@pytest.fixture
def store(tmp_path: Path, honest: Measured) -> Iterator[Store]:
    url, schema = site_store.emptied("no database to queue legality jobs in")
    connection = db.connect(url)
    digest = spec_hash(honest.spec)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
            ids = []
            for handle in ("ada", "bo"):
                cursor.execute(
                    "insert into users (handle, email) values (%s, %s) returning id",
                    (handle, f"{handle}@example.org"),
                )
                ids.append(int(cursor.fetchone()[0]))
            cursor.execute(
                "insert into games (hash, spec, name, owner, visibility, origin)"
                " values (%s, %s::jsonb, %s, %s, 'public', 'designed')",
                (digest, json.dumps(honest.spec), honest.spec["name"], ids[0]),
            )
        connection.commit()
        (tmp_path / "runs").mkdir()
        yield Store(
            connection, tmp_path / "runs", digest, ids[0], ids[1], honest, url, schema
        )
    finally:
        connection.rollback()
        connection.close()


class TestTheJob:
    def test_an_honest_submission_becomes_legal_with_the_servers_number(
        self, store: Store
    ) -> None:
        job = store.file()
        store.connection.commit()
        assert store.job(job)[0] == legality.QUEUED
        assert legality.upload_path(store.runs, job).is_file()

        assert legality.claim(store.connection) == job
        state, _, _, fingerprint = store.job(job)
        assert state == legality.RUNNING and fingerprint == measurement.built_fingerprint()

        finished = store.run(job)
        assert finished.state == legality.DONE and finished.refusal is None
        result = finished.result
        assert result is not None and result.state == LEGAL and result.job == job
        assert result.headline == store.measured.index
        assert result.viable == store.measured.viable
        assert result.basis == replace(store.measured.stamp, checked=LEGAL)
        assert result.basis.fit == store.measured.player
        state, error, log, _ = store.job(job)
        assert (state, error) == (legality.DONE, None)
        assert log == f"provisional result {result.id}: {store.measured.index:.4f}"

        where = store.runs / "provisional" / str(result.id)
        for name in FILES:
            assert (where / name).read_text(encoding="utf-8") == store.measured.files[name]
        assert (where / provisional.FIT).is_file()
        assert not list(store.runs.rglob("canonical.json"))

        assert legality.sweep(store.connection, store.runs) == [job]
        assert not legality.upload_path(store.runs, job).exists()

    def test_a_refusal_records_nothing_and_leaves_the_standing_result(
        self, store: Store
    ) -> None:
        first = store.run(store.file())
        assert first.state == legality.DONE and first.result is not None
        rows = store.measured.rows(slices.TOURNAMENT)
        rows[2]["winner"] = 0 if rows[2]["winner"] is None else None
        job = store.file(store.measured.upload(tournament=lines(rows)))
        finished = store.run(job)
        assert finished.state == legality.REFUSED and finished.result is None
        assert finished.refusal is not None and finished.refusal.name == legality.WRONG_OUTCOME
        state, error, _, _ = store.job(job)
        assert state == legality.REFUSED
        assert error is not None and error.startswith(f"{legality.WRONG_OUTCOME}: game 2 ")
        # The earlier result still stands, and no directory was made for the refused one.
        assert store.results() == [(first.result.id, LEGAL, first.job)]
        assert sorted(p.name for p in (store.runs / "provisional").iterdir()) == sorted(
            [str(first.result.id), legality.UPLOADS]
        )

    def test_a_stamp_whose_fit_differs_is_refused_and_records_nothing(
        self, store: Store
    ) -> None:
        """Q468. Filed (a visitor's design has no store to refuse it by), then refused by
        the walk's own regression; the honest stamp beside it is taken."""
        wrong = replace(store.measured.stamp, fit=OTHER_PLAYER)
        job = store.file(store.measured.upload(basis=wrong))
        finished = store.run(job)
        assert finished.state == legality.REFUSED and finished.result is None
        assert finished.refusal is not None and finished.refusal.name == legality.BASIS
        state, error, _, _ = store.job(job)
        assert state == legality.REFUSED
        assert error is not None and f"the fitted objective {OTHER_PLAYER}" in error
        assert store.results() == []

        taken = store.run(store.file())
        assert taken.state == legality.DONE and taken.result is not None
        assert taken.result.basis.fit == store.measured.player

    def test_a_newer_pass_supersedes_the_standing_result(self, store: Store) -> None:
        first = store.run(store.file())
        second = store.run(store.file())
        assert first.result is not None and second.result is not None
        assert store.results() == [
            (first.result.id, provisional.SUPERSEDED, first.job),
            (second.result.id, LEGAL, second.job),
        ]

    def test_what_needs_no_walk_is_refused_at_once(self, store: Store) -> None:
        sources = store.measured.current.engine.rpartition("-")[0]
        stale = replace(store.measured.stamp, engine=f"{sources}-0123456789abcdef")
        with pytest.raises(Refused, match="fetches the current module") as caught:
            store.file(store.measured.upload(basis=stale))
        assert caught.value.name == legality.STALE_BUILD
        with pytest.raises(Refused) as caught:
            store.file("{")
        assert caught.value.name == legality.UNREADABLE
        with pytest.raises(ProvisionalError, match="only the person who made a game"):
            store.file(submitter=store.other)
        store.connection.rollback()
        with store.connection.cursor() as cursor:
            cursor.execute("select count(*) from jobs")
            assert cursor.fetchone()[0] == 0

    def test_one_submission_at_a_time(self, store: Store) -> None:
        store.file()
        with pytest.raises(legality.LegalityError, match="still being checked"):
            store.file()

    def test_a_fault_is_recorded_on_the_job_and_nothing_is_half_kept(
        self, store: Store
    ) -> None:
        job = store.file()
        store.connection.commit()
        legality.upload_path(store.runs, job).unlink()
        finished = store.run(job)
        assert finished.state == legality.FAILED and "upload" in finished.error
        assert store.job(job)[0] == legality.FAILED
        assert store.results() == []
        with pytest.raises(legality.LegalityError, match="already failed"):
            legality.run(store.connection, store.runs, job, basis=store.basis)

    def test_the_oldest_job_is_claimed_first_and_an_empty_queue_says_so(
        self, store: Store
    ) -> None:
        assert legality.claim(store.connection) is None
        first = store.file()
        store.connection.commit()
        with store.connection.cursor() as cursor:
            cursor.execute(
                "insert into jobs (kind, hash, requester, state) values (%s, %s, %s, %s)"
                " returning id",
                ("canonical", store.game, store.other, legality.QUEUED),
            )
        store.connection.commit()
        assert legality.claim(store.connection) == first
        assert legality.claim(store.connection) is None


# ------------------------------------------------------- the site's route (B10)


@pytest.fixture
def site(store: Store, tmp_path: Path) -> Iterator[Any]:
    """The site, over the store, with the runner held to the test's small roster."""
    from chess_variant_sandbox.site.server import SiteApp

    for name in ("rulesets", "archive"):
        (tmp_path / name).mkdir()

    def open_one() -> Any:
        connection = db.connect(store.url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {store.schema}")
        return connection

    app = SiteApp(
        tmp_path / "rulesets", store.runs, archive_dir=tmp_path / "archive", connect=open_one
    )
    app.checks.basis = store.basis
    yield app
    app.checks.stop()


def _as(app: Any, viewer: int | None) -> Any:
    app.viewer = viewer
    return app


class TestTheSubmissionRoute:
    """`docs/plan-client-compute.md` B10: the page's upload filed, walked on the runner's
    own thread, and answered with the server's number and state."""

    def test_an_honest_upload_becomes_legal_with_the_servers_own_number(
        self, site: Any, store: Store
    ) -> None:
        sent = _as(site, store.designer).post(
            "/api/provisional", {"game": store.game, "upload": store.measured.upload()}
        )
        assert sent["state"] == legality.QUEUED and sent["game"] == store.game
        job = sent["job"]
        assert site.get(f"/api/provisional/{job}")["state"] == legality.QUEUED

        # Tier one, then the spot check it filed (Q503e), on the same runner.
        finished, checked = site.checks.drain()
        assert finished.job == job and finished.state == legality.DONE
        assert checked.job == finished.sample and checked.state == legality.DONE
        answer = site.get(f"/api/provisional/{job}")
        assert answer["state"] == legality.DONE and answer["seconds"] >= 0
        assert answer["spotCheck"] == {"state": "done"}
        shown = answer["result"]
        assert shown["state"] == "spot-checked" and shown["yours"] is True
        assert shown["index"] == store.measured.index
        assert f"{shown['index']:.4f}" == f"{store.measured.index:.4f}"
        assert shown["seeds"] == list(store.measured.current.seeds)
        # The runner swept the upload once the result held the server's own copies.
        assert not legality.upload_path(store.runs, job).exists()

    def test_a_stale_build_is_refused_by_name_and_nothing_is_filed(
        self, site: Any, store: Store
    ) -> None:
        sources = store.measured.current.engine.rpartition("-")[0]
        stale = replace(store.measured.stamp, engine=f"{sources}-0123456789abcdef")
        answer = _as(site, store.designer).post(
            "/api/provisional",
            {"game": store.game, "upload": store.measured.upload(basis=stale)},
        )
        assert answer["state"] == legality.REFUSED and "job" not in answer
        assert answer["refusal"]["name"] == legality.STALE_BUILD
        assert "fetches the current module" in answer["refusal"]["sentence"]
        assert site.checks.drain() == []
        with store.connection.cursor() as cursor:
            cursor.execute("select count(*) from jobs")
            assert cursor.fetchone()[0] == 0

    def test_a_refusal_the_walk_finds_is_named_on_the_status(
        self, site: Any, store: Store
    ) -> None:
        rows = store.measured.rows(slices.TOURNAMENT)
        rows[2]["winner"] = 0 if rows[2]["winner"] is None else None
        text = store.measured.upload(tournament=lines(rows))
        job = _as(site, store.designer).post(
            "/api/provisional", {"game": store.game, "upload": text}
        )["job"]
        site.checks.drain()
        answer = site.get(f"/api/provisional/{job}")
        assert answer["state"] == legality.REFUSED and "result" not in answer
        assert answer["refusal"]["name"] == legality.WRONG_OUTCOME
        assert answer["refusal"]["sentence"].startswith("game 2 ")

    def test_only_the_sender_reads_it_and_a_visitor_neither_sends_nor_reads(
        self, site: Any, store: Store
    ) -> None:
        from chess_variant_sandbox.site.server import SignInFirst

        job = _as(site, store.designer).post(
            "/api/provisional", {"game": store.game, "upload": store.measured.upload()}
        )["job"]
        with pytest.raises(KeyError, match="no such submission"):
            _as(site, store.other).get(f"/api/provisional/{job}")
        with pytest.raises(KeyError, match="no such submission"):
            _as(site, store.designer).get("/api/provisional/nine")
        with pytest.raises(SignInFirst):
            _as(site, None).get(f"/api/provisional/{job}")
        with pytest.raises(SignInFirst):
            _as(site, None).post("/api/provisional", {"game": store.game, "upload": "{}"})
        # Somebody else's game is refused with the sentence the page shows.
        with pytest.raises(ProvisionalError, match="only the person who made a game"):
            _as(site, store.other).post(
                "/api/provisional", {"game": store.game, "upload": store.measured.upload()}
            )

    def test_a_draft_not_saved_on_the_site_says_so(self, site: Any, store: Store) -> None:
        with pytest.raises(legality.LegalityError, match="save it, then send its score again"):
            _as(site, store.designer).post(
                "/api/provisional", {"game": "0" * 16, "upload": store.measured.upload()}
            )

    def test_the_runner_thread_walks_what_is_filed(self, site: Any, store: Store) -> None:
        """`serve` starts it; a submission wakes it. A job a stopped process left
        running is queued again first, because this is the only runner."""
        with store.connection.cursor() as cursor:
            cursor.execute(
                "insert into jobs (kind, hash, requester, state) values (%s, %s, %s, %s)"
                " returning id",
                (legality.KIND, store.game, store.other, legality.RUNNING),
            )
            left = int(cursor.fetchone()[0])
        store.connection.commit()
        assert site.checks.requeue() == 1
        with store.connection.cursor() as cursor:
            cursor.execute("delete from jobs where id = %s", (left,))
        store.connection.commit()

        site.checks.every = 30.0
        site.checks.start()
        job = _as(site, store.designer).post(
            "/api/provisional", {"game": store.game, "upload": store.measured.upload()}
        )["job"]
        deadline = time.monotonic() + 60
        answer = site.get(f"/api/provisional/{job}")
        while answer["state"] in (legality.QUEUED, legality.RUNNING):
            assert time.monotonic() < deadline, answer
            time.sleep(0.05)
            answer = site.get(f"/api/provisional/{job}")
        assert answer["state"] == legality.DONE
        assert answer["result"]["index"] == store.measured.index


# ------------------------------------------------------------- chess, on its roster


@pytest.mark.slow
def test_an_honest_chess_submission_scores_to_the_servers_number(tmp_path: Path) -> None:
    """The row's acceptance on chess's whole canonical measurement, fit included.

    A few minutes at four threads: the fit, its acceptance match at 6,400 nodes, 256
    games, the analysis and the blunder trials, all through B0's request text.
    """
    spec = load_spec(RULESETS / "chess.json")
    roster = slices.canonical_config(workers=4)
    measured = measure_as_a_browser(spec, roster, tmp_path / "browser", parts=4)
    assert measured.current == corpus.current_basis(
        spec_hash(spec), spec=spec, seeds=corpus.ONE_SEED
    )
    started = time.perf_counter()
    checked = measured.check(measured.upload())
    scored = legality.score(checked, measured.current.weights_version)
    assert time.perf_counter() - started < 120
    assert scored.headline == measured.index
    assert scored.viable == measured.viable
    for name in (*FILES, f"index-{measured.current.weights_version}.json"):
        assert scored.files[name] == measured.files[name], name
