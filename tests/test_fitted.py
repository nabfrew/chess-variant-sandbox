"""The fitted objective: the one thing allowed to break the per-ruleset rule, and why.

Stage 28 revises a stated invariant -- *a heuristic is per-ruleset by construction and
so must never enter the rated roster* -- and the revision is narrow enough that most of
these tests are about the narrowness rather than about the fit:

* the agent's **name** must be the same string on every board, because that is the whole
  of the argument for letting it into the roster;
* a **stated** vector must still be unable to get in, because nothing about stage 28
  changes what is wrong with a hand-chosen digest;
* a fit that does not beat the objective it replaces must **fall back and say so**,
  because a silent bad fit is a differently-blind ladder wearing a better name;
* the built-in path must be untouched, because every committed number in the project was
  produced by it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chess_variant_sandbox.agents import (
    NATIVE_VALUES,
    NATIVE_WEIGHTS,
    build_agent,
    native_spec,
    parse_agent,
)
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator, feature_context
from chess_variant_sandbox.agents.fitted import (
    ACCEPT_NODES,
    FitError,
    FittedObjective,
    fitted_weights,
    logistic_fit,
    procedure_digest,
    register_fit,
    resolve_fit,
    store_path,
    unit,
)
from chess_variant_sandbox.agents.heuristics import Heuristic, register_heuristic
from chess_variant_sandbox.agents.objective_options import objective_weights
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.fitting import fit_ruleset, harvest
from chess_variant_sandbox.tournament.play import play_game, play_game_python

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def vector(**terms: float) -> tuple[float, ...]:
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return tuple(weights)


def a_fit(ruleset_hash: str, accepted: bool = True, **terms: float) -> FittedObjective:
    return FittedObjective(
        ruleset_hash=ruleset_hash,
        procedure=procedure_digest(),
        vector=unit(vector(**terms) if terms else vector(material=1.0, centrality=0.3)),
        accepted=accepted,
        accept_score=0.75 if accepted else 0.25,
        accept_games=48,
        samples=500,
        loss=0.5,
    )


class TestTheNameIsTheProcedure:
    """28.2, and it is the whole argument rather than a detail of spelling."""

    def test_two_rulesets_fitted_players_share_one_agent_string(self) -> None:
        """A rung has to be one rung.

        Every skill component reads a ladder of agent *names* out of a rating table. A
        name that differed per ruleset would put a different player on each board's
        ladder, which is exactly the objection to letting a stated heuristic in -- so if
        this ever stops holding, the invariant stops being revised and starts being
        broken.
        """
        procedure = procedure_digest()
        register_fit(a_fit("aaaaaaaaaaaa"), replace=True)
        register_fit(a_fit("bbbbbbbbbbbb", material=1.0, territory_margin=0.9), replace=True)

        name = f"minimax:fit={procedure},nodes=6400"
        first = native_spec(name, "aaaaaaaaaaaa")
        second = native_spec(name, "bbbbbbbbbbbb")
        assert first is not None and second is not None
        # One name, two vectors: the per-ruleset half is derived, exactly as the piece
        # values already are.
        assert first[NATIVE_WEIGHTS] != second[NATIVE_WEIGHTS]

    def test_the_procedure_digest_says_nothing_about_any_ruleset(self) -> None:
        assert procedure_digest() == procedure_digest()
        assert len(procedure_digest()) == 12

    def test_changing_the_procedure_changes_the_name(self) -> None:
        """A bump is a new player, never a quiet edit of an old one."""
        import chess_variant_sandbox.agents.fitted as fitted

        before = procedure_digest()
        original = fitted.FIT_GAMES_PER_PAIRING
        try:
            fitted.FIT_GAMES_PER_PAIRING = original + 1
            assert procedure_digest() != before
        finally:
            fitted.FIT_GAMES_PER_PAIRING = original
        assert procedure_digest() == before


class TestRefusingAFitThatDidNotWork:
    """28.3. The measurement that makes the revision answerable rather than asserted."""

    def test_a_refused_fit_plays_the_built_in_objective(self) -> None:
        register_fit(a_fit("cccccccccccc", accepted=False), replace=True)
        weights = fitted_weights({"fit": procedure_digest()}, "cccccccccccc")
        # The empty tuple is the signal every agent already understands. A refused fit is
        # the built-in agent, by name and in the record.
        assert weights == ()

    def test_the_vector_is_still_written_down(self) -> None:
        """A failure that leaves no file is a failure nobody can read."""
        refused = a_fit("cccccccccccc", accepted=False)
        assert any(refused.vector)
        assert refused.weights == ()
        assert FittedObjective.from_dict(refused.as_dict()).accepted is False

    def test_the_verdict_carries_the_resolution_of_the_test_that_produced_it(self) -> None:
        """A binary verdict off a finite match needs its own error bar beside it.

        Not used to decide anything -- a margin applied here would be a threshold chosen
        after seeing which rulesets it excludes -- but a reader has to be able to tell
        0.93 over 48 games from 0.52 over 48 games, and only one of those is a result.
        """
        assert a_fit("dddddddddddd").accept_error == pytest.approx(math.sqrt(0.25 / 48))
        assert FittedObjective.from_dict(
            {**a_fit("d" * 12).as_dict(), "accept_games": 0}
        ).accept_error == 0.0


    def test_a_ruleset_too_short_to_sample_is_refused_rather_than_raising(self) -> None:
        """A fit that cannot be *made* is a refusal, not an exception.

        `mutant-cramped-board` and `mutant-gardner-uniform` are over in fewer plies than
        the harvest skips, so they yield no rows at all. Raising there would mean a
        roster carrying `fit=` could not play a corpus containing one such entry, and a
        measurement that covers whichever entries the tooling can reach is not a
        measurement of the corpus.
        """
        from chess_variant_sandbox.metrics.mutants import build_mutants
        from chess_variant_sandbox.spec.canonical import load_spec
        from chess_variant_sandbox.tournament.fitting import fit_ruleset

        bases = {}
        for path in sorted(Path("rulesets").glob("*.json")):
            spec = load_spec(path)
            bases[spec["name"]] = spec
        short = next(
            spec for mutant, spec in build_mutants(bases)
            if mutant.name == "mutant-cramped-board"
        )
        from chess_variant_sandbox.measurement import built_fingerprint

        fit = fit_ruleset(short, workers=1)
        assert fit.samples == 0
        assert fit.accepted is False
        # And it still names the core, though it never reached a match: the games that
        # yielded no rows were played by one, and "which engine found nothing" is the
        # question a reader of this refusal has.
        assert fit.build == built_fingerprint()
        # And therefore the built-in objective, by the same route every other refusal
        # takes rather than by a special case downstream.
        assert fit.weights == ()


class TestResolutionNeverDefaults:
    def test_an_unregistered_pair_raises(self) -> None:
        with pytest.raises(FitError, match="no fit for procedure"):
            resolve_fit("nosuchprocedure", "eeeeeeeeeeee")

    def test_a_caller_with_no_ruleset_raises_rather_than_guessing(self) -> None:
        """The one thing a fitted agent cannot do is resolve without a board.

        `native_spec` defaults the hash to empty for the callers that ask about agents
        which cannot depend on the ruleset. A fitted agent reaching that default is a
        caller that has not been threaded through, and it has to fail loudly there rather
        than play something.
        """
        register_fit(a_fit("ffffffffffff"), replace=True)
        with pytest.raises(FitError, match="which ruleset it is playing"):
            native_spec(f"minimax:fit={procedure_digest()},nodes=100")

    def test_a_digest_that_looks_like_a_number_survives_the_option_parser(self) -> None:
        """The same trap `h=` fell into: `parse_agent` coerces anything numeric-looking."""
        for digest in ("096717939346", "37625e727351"):
            config = parse_agent(f"minimax:fit={digest},nodes=100")
            assert config.options["fit"] == digest
            assert str(config) == f"minimax:fit={digest},nodes=100"

    def test_a_store_from_another_vocabulary_is_refused(self) -> None:
        payload = a_fit("gggggggggggg").as_dict()
        payload["vocabulary_version"] = payload["vocabulary_version"] + 1
        with pytest.raises(FitError, match="refit"):
            FittedObjective.from_dict(payload)

    def test_a_store_from_another_procedure_version_is_refused(self) -> None:
        payload = a_fit("hhhhhhhhhhhh").as_dict()
        payload["procedure_version"] = payload["procedure_version"] + 1
        with pytest.raises(FitError, match="refit"):
            FittedObjective.from_dict(payload)

    def test_a_renamed_file_is_refused_rather_than_believed(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.agents.fitted import load_fit

        fit = a_fit("iiiiiiiiiiii")
        wrong = tmp_path / store_path("jjjjjjjjjjjj", fit.procedure).name
        wrong.write_text(json.dumps(fit.as_dict()), encoding="utf-8")
        with pytest.raises(FitError, match="has been renamed"):
            load_fit("jjjjjjjjjjjj", fit.procedure, tmp_path)


class TestTheStoreNamesTheCoreThatMadeIt:
    """Q46. The engine decides the games and nothing in the store used to say which one.

    A fit is a verdict about games twice over: the vector is regressed from records the
    core played, and the accept/refuse comes from a match the core plays. Neither the
    procedure digest nor the ruleset hash covers `rust/src/`, so a store written before
    this field could not tell a reader whether the roster was about to play a vector
    accepted by an engine that no longer exists -- under an agent string that would not
    change either way. `agents/valuations.py` had already made the same argument for the
    searched piece values; these are its tests in the second place.
    """

    def test_a_verdict_names_the_core_it_was_made_on(self) -> None:
        """Absent reads as *unknown*, never as agreement -- the safe direction, exactly
        as an absent `measurement.Basis` is treated as stale."""
        from chess_variant_sandbox.agents.fitted import stale_verdict
        from chess_variant_sandbox.measurement import built_fingerprint

        unstamped = a_fit("aaaaaaaaaaaa")
        assert unstamped.build == ""
        assert "records no core build" in (stale_verdict(unstamped) or "")

        current = FittedObjective(**{**unstamped.__dict__, "build": built_fingerprint()})
        assert stale_verdict(current) is None

        other = FittedObjective(**{**unstamped.__dict__, "build": "0000000000000000"})
        drifted = stale_verdict(other) or ""
        assert "0000000000000000" in drifted and "may no longer hold" in drifted

    def test_the_build_survives_the_file_round_trip(self, tmp_path: Path) -> None:
        original = FittedObjective(**{**a_fit("bbbbbbbbbbbb").__dict__, "build": "abc123"})
        assert FittedObjective.load(original.save(tmp_path / "f.json")).build == "abc123"

    def test_a_store_written_before_the_field_is_unknown_and_not_the_current_core(
        self,
    ) -> None:
        """The `Basis.tables` mistake, not made twice.

        Defaulting an absent stamp to whatever is loaded now would let exactly the stores
        this field exists to distinguish pass as agreeing with the engine about to play
        them, which is the one reading with no evidence behind it.
        """
        from chess_variant_sandbox.agents.fitted import stale_verdict

        payload = a_fit("cccccccccccc").as_dict()
        del payload["build"]
        assert FittedObjective.from_dict(payload).build == ""
        assert stale_verdict(FittedObjective.from_dict(payload)) is not None

    def test_the_one_constructor_stamps_the_core_it_ran_on(self) -> None:
        """`_record` is where every verdict is built, provisional and final alike, so the
        stamp cannot be present on one and missing on the other."""
        from chess_variant_sandbox.measurement import built_fingerprint
        from chess_variant_sandbox.tournament.fitting import _record

        made = _record("dddddddddddd", procedure_digest(), vector(material=1.0),
                       True, 0.75, 48, 500, 0.5)
        assert made.build == built_fingerprint()

    def test_reusing_a_stored_verdict_from_another_core_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The site the review named: `ensure_fit` reuses without re-deriving.

        The reuse is right -- refusing would break every command that touches an older
        ruleset, and refitting unasked would spend minutes on a caller that only wanted a
        number. What was wrong is that it happened in silence.
        """
        from chess_variant_sandbox.tournament.fitting import ensure_fit

        spec = load_spec(RULESETS / "gardner.json")
        digest = Engine(spec).spec.hash
        stored = FittedObjective(
            **{**a_fit(digest).__dict__, "build": "0000000000000000"}
        )
        stored.save(store_path(digest, procedure_digest(), tmp_path))

        # No games: the store is read, and that is the whole of the call.
        reused = ensure_fit(spec, store=str(tmp_path))
        assert reused.vector == stored.vector
        assert "may no longer hold" in capsys.readouterr().err

    def test_a_verdict_made_on_this_core_is_reused_without_a_word(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An anti-false-positive: a warning on every reuse is a warning nobody reads."""
        from chess_variant_sandbox.measurement import built_fingerprint
        from chess_variant_sandbox.tournament.fitting import ensure_fit

        spec = load_spec(RULESETS / "gardner.json")
        digest = Engine(spec).spec.hash
        FittedObjective(
            **{**a_fit(digest).__dict__, "build": built_fingerprint()}
        ).save(store_path(digest, procedure_digest(), tmp_path))

        ensure_fit(spec, store=str(tmp_path))
        assert capsys.readouterr().err == ""

    def test_no_committed_fit_this_checkout_can_play_hides_which_core_made_it(
        self,
    ) -> None:
        """Files under a superseded digest are out of scope: they cannot be loaded, let
        alone played, so what they are evidence *of* is the procedure that wrote them.

        **One exemption, named rather than papered over.**
        `b88e98dab286-98f172aeac49.json` was written by Q10 hours before this field
        existed. `docs/reference/schema-8.md` records it as made on `e7cbecd650a965c9`,
        which is a contemporaneous note and not a guess -- and it is still not written
        into the file, because a stamp put there by a later hand is indistinguishable to
        every future reader from an invented one. `CLAUDE.md` already says this about
        hand-writing a `canonical.json`; a store is the same kind of artefact. The
        assertion is a subset, so a refit that removes the last unstamped file passes
        rather than having to be edited.
        """
        from chess_variant_sandbox.agents.fitted import STORE_DIR

        root = Path(__file__).resolve().parents[1] / STORE_DIR
        unstamped = {
            path.name
            for path in sorted(root.glob(f"*-{procedure_digest()}.json"))
            if not FittedObjective.load(path).build
        }
        assert unstamped <= {"b88e98dab286-98f172aeac49.json"}


class TestNothingWritesIntoTheStoreByAccident:
    """Q91. The store is tracked, and the suite used to write into it.

    `corpus.measure` fits a candidate on the way to measuring it, and `ensure_fit` is the
    only writer whose caller names no path -- so `tests/test_corpus.py`'s five-square
    fixture left `fitted/ee26d2322244-*.json` in the directory git watches, and somebody
    committed one. A fit is a record of what an opponent was given to play one game with,
    so a store anything can write into by accident is a store whose files mean nothing.
    """

    def test_the_default_store_is_the_repository_until_the_environment_moves_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from chess_variant_sandbox.agents.fitted import STORE_DIR, STORE_ENV, fit_store

        monkeypatch.delenv(STORE_ENV, raising=False)
        assert fit_store() == STORE_DIR
        monkeypatch.setenv(STORE_ENV, str(tmp_path))
        assert fit_store() == str(tmp_path)

    def test_ensure_fit_reads_the_environment_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The behavioural half, and it is a *read* rather than a write on purpose.

        Writing would mean playing the fit, which is minutes. Pre-storing the verdict and
        watching `ensure_fit` come back with it -- having played nothing -- proves the
        same thing: the call resolved its store from the environment. If the default were
        still bound into the signature it would miss, and fit `gardner` for real.
        """
        from chess_variant_sandbox.agents.fitted import STORE_ENV
        from chess_variant_sandbox.tournament.fitting import ensure_fit

        spec = load_spec(RULESETS / "gardner.json")
        digest = Engine(spec).spec.hash
        stored = a_fit(digest, centrality=0.9)
        stored.save(store_path(digest, procedure_digest(), tmp_path))

        monkeypatch.setenv(STORE_ENV, str(tmp_path))
        assert ensure_fit(spec).vector == stored.vector

    def test_an_explicit_store_still_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The environment is the default, never an override: `chess-variant-sandbox fit --store` and
        the tests below pass a path and must get that path."""
        from chess_variant_sandbox.agents.fitted import STORE_ENV
        from chess_variant_sandbox.tournament.fitting import ensure_fit

        spec = load_spec(RULESETS / "gardner.json")
        digest = Engine(spec).spec.hash
        asked = tmp_path / "asked"
        asked.mkdir()
        a_fit(digest, mobility=0.8).save(store_path(digest, procedure_digest(), asked))

        monkeypatch.setenv(STORE_ENV, str(tmp_path / "ignored"))
        assert ensure_fit(spec, store=str(asked)).vector == unit(vector(mobility=0.8))

    def test_the_session_is_pointed_away_from_the_repository(self) -> None:
        """The fixture in `conftest.py` is what actually closes the defect, so it is
        asserted rather than trusted: any fit this run makes lands outside the store."""
        from chess_variant_sandbox.agents.fitted import STORE_DIR, fit_store

        where = Path(fit_store()).resolve()
        assert where != (Path(__file__).resolve().parents[1] / STORE_DIR)
        assert not where.is_relative_to(Path(__file__).resolve().parents[1])


class TestAStatedVectorStillCannotGetIn:
    def test_naming_a_heuristic_and_a_fit_at_once_is_refused(self) -> None:
        """Two answers to one question, and a precedence rule would hide the wrong one."""
        digest = register_heuristic(Heuristic("m", "", vector(material=1.0)))
        register_fit(a_fit("kkkkkkkkkkkk"), replace=True)
        with pytest.raises(ValueError, match="h and fit at once"):
            objective_weights({"h": digest, "fit": procedure_digest()}, "kkkkkkkkkkkk")

    def test_a_heuristic_still_resolves_without_a_ruleset(self) -> None:
        """The distinction the whole stage turns on, asserted rather than described.

        A stated digest *is* the vector, so it needs no board and is per-ruleset only in
        what it means. A fitted procedure is the opposite on both counts.
        """
        weights = vector(material=1.0, centrality=0.4)
        digest = register_heuristic(Heuristic("hold", "", weights))
        assert objective_weights({"h": digest}) == weights


class TestTheBuiltInPathIsUntouched:
    def test_no_fit_option_means_the_original_evaluator(self, chess: Engine) -> None:
        assert objective_weights({}, chess.spec.hash) == ()
        assert isinstance(build_agent("minimax:nodes=100", chess).evaluator, Evaluator)

    @pytest.mark.parametrize("agent", ["random", "greedy", "minimax:nodes=400", "mcts:nodes=100"])
    def test_every_built_in_still_sends_an_empty_weight_vector(
        self, agent: str, chess: Engine
    ) -> None:
        spec = native_spec(agent, chess.spec.hash)
        assert spec is not None
        assert spec[NATIVE_WEIGHTS] == () and spec[NATIVE_VALUES] == ()


class TestTheRegression:
    """Convex, deterministic, and it has to put a term nobody's ruleset declares at zero."""

    def test_a_term_that_is_always_zero_is_fitted_to_exactly_zero(self) -> None:
        """What the fixed-length vocabulary means by "absent", arrived at by the solver.

        Ridge does it: a column of zeros contributes nothing to the gradient or the
        Hessian, so the penalty is all that acts on that weight and it decays to the
        origin. Not approximately -- `FeatureEvaluator` skips a zero weight rather than
        multiplying by it, so "absent" has to be exact.
        """
        samples = [
            ((1.0, 0.0), 1.0),
            ((0.8, 0.0), 1.0),
            ((-1.0, 0.0), 0.0),
            ((-0.7, 0.0), 0.0),
        ]
        weights, _intercept, _loss = logistic_fit(samples)
        assert weights[1] == 0.0
        assert weights[0] > 0.0

    def test_it_finds_the_sign_of_a_feature_that_predicts_the_result(self) -> None:
        samples = [((x / 10.0,), 1.0 if x > 0 else 0.0) for x in range(-10, 11) if x]
        weights, _intercept, _loss = logistic_fit(samples)
        assert weights[0] > 0.0
        flipped, _i, _l = logistic_fit([(features, 1.0 - y) for features, y in samples])
        assert flipped[0] < 0.0

    def test_a_draw_is_evidence_for_neither_side(self) -> None:
        """Fractional targets, not a third class and not a dropped row."""
        weights, intercept, _loss = logistic_fit([((1.0,), 0.5), ((-1.0,), 0.5)])
        assert weights[0] == pytest.approx(0.0, abs=1e-6)
        assert intercept == pytest.approx(0.0, abs=1e-6)

    def test_the_same_sample_fits_the_same_vector_twice(self) -> None:
        samples = [((x / 7.0, (x % 3) / 3.0), 1.0 if x > 0 else 0.0) for x in range(-9, 10)]
        assert logistic_fit(samples) == logistic_fit(samples)

    def test_an_empty_sample_is_an_error_rather_than_a_zero_vector(self) -> None:
        with pytest.raises(FitError, match="needs positions"):
            logistic_fit([])


class TestHarvesting:
    def test_every_sample_is_the_whole_vocabulary_and_a_result(self, chess: Engine) -> None:
        record = play_game(chess, ("greedy", "minimax:nodes=200"), seed=3)
        samples = harvest(chess, [record])
        assert samples
        assert all(len(features) == len(VOCABULARY) for features, _ in samples)
        assert {label for _, label in samples} == {record.score_for(0)}

    def test_the_opening_is_skipped_and_the_rest_is_strided(self, chess: Engine) -> None:
        from chess_variant_sandbox.agents.fitted import FIT_SKIP_PLIES, FIT_STRIDE

        record = play_game(chess, ("greedy", "minimax:nodes=200"), seed=4)
        expected = len(range(FIT_SKIP_PLIES, record.plies, FIT_STRIDE))
        assert len(harvest(chess, [record])) == expected

    def test_a_term_that_is_dead_on_this_ruleset_reads_exactly_zero(
        self, chess: Engine
    ) -> None:
        """The premise the test below rests on, checked rather than assumed.

        Chess declares no facing, so `facing_alignment` has to be identically `0.0` over
        a whole harvested sample -- not small, exactly zero -- or the argument that the
        vocabulary bump costs the corpus nothing beyond bookkeeping is not available.
        """
        record = play_game(chess, ("greedy", "minimax:nodes=200"), seed=3)
        column = VOCABULARY.index("facing_alignment")
        values = {features[column] for features, _ in harvest(chess, [record])}
        assert values == {0.0}


def test_adding_a_dead_column_moves_no_other_weight_to_the_bit() -> None:
    """What says a vocabulary bump owes the corpus a refit and not a different answer.

    `facing_alignment` was appended to a fixed-length vocabulary, so every stored
    fourteen-weight vector is refused by its own length and the whole corpus owes a
    refit. The question that decides how much M1 has to worry about that is whether the
    refit *lands somewhere else*, and this is the answer: on a ruleset where the new
    column is identically zero, the other fourteen weights come out **bit-identical** to
    what a fourteen-column fit produces from the same rows.

    It is a property of the solver rather than a coincidence. `_dot` skips a zero
    feature; the gradient loop skips it; the Hessian entries that pair it with anything
    else are `variance * xi * 0.0`, which leaves an exact `0.0`; its diagonal carries the
    ridge alone, so partial pivoting never chooses its row for another column and
    elimination against it is skipped on a zero factor; and back-substitution subtracts
    `rows[c][14] * out[14]`, which is `0.0 * 0.0`. Adding an exact zero to a float is the
    identity, so nothing rounds.

    Sampled synthetically rather than from a game, because the claim is about the solver
    and a fixed table is what makes the test cheap and exact.
    """
    from chess_variant_sandbox.agents.fitted import logistic_fit

    rows = []
    for x in range(-9, 10):
        wide = tuple(
            [x / 7.0, (x % 3) / 3.0, (x % 5) / 5.0] + [0.0] * (len(VOCABULARY) - 3)
        )
        rows.append((wide, 1.0 if x > 0 else 0.0))
    narrow = [(features[:-1], label) for features, label in rows]

    wide_weights, wide_intercept, wide_loss = logistic_fit(rows)
    narrow_weights, narrow_intercept, narrow_loss = logistic_fit(narrow)

    assert wide_weights[:-1] == narrow_weights
    assert wide_weights[-1] == 0.0
    assert wide_intercept == narrow_intercept
    assert wide_loss == narrow_loss


@pytest.mark.slow
class TestTheFitPlaysTheSameGameOnBothSides:
    """A dense vector is where the port is most likely to disagree, and it is silent.

    Every hand-written heuristic in the project weights two or three terms. A fit weights
    whatever the data says, so it routinely turns on `advance`, `centrality` and
    `prey_weight` at once -- all of them means over the pieces, which is the shape
    `dispersion` was removed for. A last-bit disagreement in one of them makes the Python
    oracle and the core different players, and every perft still passes.
    """

    @pytest.mark.parametrize("name", ["gardner", "dobutsu", "rps-v2"])
    def test_a_dense_fitted_vector_plays_identically_in_both_backends(self, name: str) -> None:
        engine = Engine(load_spec(RULESETS / f"{name}.json"))
        fit = fit_ruleset(load_spec(RULESETS / f"{name}.json"), workers=1, verify=False)
        register_fit(fit, replace=True)
        agents = (f"minimax:fit={fit.procedure},nodes=300", "greedy")

        native = play_game(engine, agents, seed=7)
        in_python = play_game_python(engine, agents, seed=7)
        assert native.moves == in_python.moves
        assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
        assert native.nodes == in_python.nodes
        assert native.depth_sum == in_python.depth_sum


@pytest.mark.slow
class TestEndToEnd:
    def test_a_fit_is_a_pure_function_of_the_ruleset(self) -> None:
        """Two runs, one vector. What makes a fitted player reproducible from the repo.

        `verify=False` because the acceptance match is the expensive half and it is not
        what this asserts -- the claim is about the procedure being deterministic, and
        the match is played by the ordinary tournament runner, which has its own test.
        """
        spec = load_spec(RULESETS / "gardner.json")
        first = fit_ruleset(spec, workers=1, verify=False)
        second = fit_ruleset(spec, workers=1, verify=False)
        assert first.vector == second.vector
        assert first.samples == second.samples

    def test_a_fitted_agent_is_a_feature_evaluator_and_a_refused_one_is_not(self) -> None:
        spec = load_spec(RULESETS / "gardner.json")
        engine = Engine(spec)
        fit = fit_ruleset(spec, workers=1, verify=False)

        register_fit(fit, replace=True)
        agent = build_agent(f"minimax:fit={fit.procedure},nodes=100", engine)
        assert isinstance(agent.evaluator, FeatureEvaluator)

        refused = FittedObjective.from_dict({**fit.as_dict(), "accepted": False})
        register_fit(refused, replace=True)
        fallen_back = build_agent(f"minimax:fit={fit.procedure},nodes=100", engine)
        assert isinstance(fallen_back.evaluator, Evaluator)


def test_the_acceptance_budget_is_the_rung_the_index_reads() -> None:
    """6,400 nodes is the top of the canonical roster, and that is not a coincidence.

    The bar is "beats the built-in objective at equal depth", and the depth that matters
    is the one every skill component is computed at. Checking at a budget nobody rates
    would be checking a player nobody plays.
    """
    from chess_variant_sandbox.corpus import CANONICAL_AGENTS

    # Read as a budget rather than as a whole name: since v9 the roster's searching
    # rungs carry `fit=<procedure>` too, and what has to match is the amount of
    # thinking, not the string.
    top = CANONICAL_AGENTS[-1]
    assert top.startswith("minimax:")
    assert top.endswith(f"nodes={ACCEPT_NODES}")


def test_the_fitted_vector_reads_the_same_vocabulary_the_heuristics_do() -> None:
    """One vocabulary, so `heuristic_agreement` can score a fit in its own terms."""
    context = feature_context(Engine(load_spec(RULESETS / "chess.json")).spec)
    reader = FeatureEvaluator(context, (1.0,) * len(VOCABULARY))
    engine = Engine(load_spec(RULESETS / "chess.json"))
    features = reader.features(engine, engine.initial_state(), 0, 20)
    assert len(features) == len(VOCABULARY)



def test_the_procedure_digest_is_where_the_vocabulary_bump_put_it() -> None:
    """`corpus.CANONICAL_AGENTS` embeds this string, and a literal is the point.

    `procedure_digest` hashes a JSON blob, so a key changed in it changes the roster,
    which moves every stored `Basis`, which marks every measurement stale and forces a
    weights re-anchor. Stage 30B added a second procedure by putting its key in the blob
    *only* when it was selected, precisely so the default would not move; it lost its
    measurement and was deleted.

    **The default has moved exactly once, and this is the record of it.** Q10 added
    `facing_alignment` to `VOCABULARY`, which bumped `VOCABULARY_VERSION` to 2, which is
    a key in this blob -- so `16a4528b0d53` became `98f172aeac49`, the 92 verdicts in
    `fitted/` filed under the old digest are the previous procedure's evidence rather
    than this one's, and the corpus owes a refit. That was not a side effect: a fifteenth
    term makes every fourteen-weight vector refuse to load by its own length, so the bump
    was owed by the shape of the vector before anything read a facing.
    """
    assert procedure_digest() == "98f172aeac49"
