"""`fit <spec> --check`: reproduce a stored fit on the loaded core, or refit it (Q80b).

`docs/reference/refit-repair.md` is the design and `fit-reproduction.md` (Q231) the three
tests. A check on a store made on the loaded core plays nothing; one on a foreign store
re-derives the fit and either records the loaded core beside the store (it reproduces) or
replaces the store (it does not).

Most arms swap `fitting.fit_ruleset` for a canned answer, because the real procedure is a
48-game match at 6,400 nodes and what is under test here is what the check *does* with
the answer. `TestOnRealGames` runs the real procedure on a ruleset too short to sample,
which is the one fit that costs no match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import cli
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.fitted import (
    FittedObjective,
    load_all,
    procedure_digest,
    store_path,
    unit,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import built_fingerprint, fingerprinted
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import fitting
from chess_variant_sandbox.tournament.fitting import (
    FITTED,
    MADE_HERE,
    REFIT,
    REPRODUCED,
    REPRODUCED_BEFORE,
    check_fit,
    differences,
    ensure_fit,
    reproduced_on,
    sidecar_path,
    store_digest,
    verified_cores,
)

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
FOREIGN = "0000000000000000"


def vector(**terms: float) -> tuple[float, ...]:
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return unit(tuple(weights))


@pytest.fixture
def spec() -> dict[str, Any]:
    return load_spec(RULESETS / "gardner.json")


@pytest.fixture
def digest(spec: dict[str, Any]) -> str:
    return Engine(spec).spec.hash


def a_fit(digest: str, build: str, **changes: Any) -> FittedObjective:
    fields: dict[str, Any] = {
        "ruleset_hash": digest,
        "procedure": procedure_digest(),
        "vector": vector(material=1.0, centrality=0.3),
        "accepted": True,
        "accept_score": 0.625,
        "accept_games": 48,
        "samples": 500,
        "loss": 0.5,
        "build": build,
    }
    return FittedObjective(**{**fields, **changes})


class Fitter:
    """Stands in for `fit_ruleset`: answers with ``fresh`` and counts the calls."""

    def __init__(self, fresh: FittedObjective) -> None:
        self.fresh = fresh
        self.calls = 0

    def __call__(self, spec: dict[str, Any], workers: int = 0) -> FittedObjective:
        self.calls += 1
        return self.fresh


@pytest.fixture
def fitter(monkeypatch: pytest.MonkeyPatch, digest: str) -> Fitter:
    """The loaded core's answer, by default the same fit as `a_fit`'s at a moved score."""
    answer = Fitter(a_fit(digest, built_fingerprint(), accept_score=0.6042))
    monkeypatch.setattr(fitting, "fit_ruleset", answer)
    return answer


def stored(digest: str, root: Path, build: str = FOREIGN, **changes: Any) -> FittedObjective:
    fit = a_fit(digest, build, **changes)
    fit.save(store_path(digest, fit.procedure, root))
    return fit


class TestTheThreeTests:
    """Q231's criterion: samples identical, vector within 1e-9, the same verdict."""

    def test_an_identical_fit_reproduces_whatever_its_score(self, digest: str) -> None:
        """A score that stays on one side of the bar plays the same objective."""
        then = a_fit(digest, FOREIGN)
        assert differences(then, a_fit(digest, "x", accept_score=0.5104)) == []

    def test_float_noise_is_absorbed_and_nothing_more(self, digest: str) -> None:
        then = a_fit(digest, FOREIGN)
        noisy = list(then.vector)
        noisy[0] += 1e-10
        assert differences(then, a_fit(digest, "x", vector=tuple(noisy))) == []
        noisy[0] += 1e-6
        assert "vector moves" in differences(then, a_fit(digest, "x", vector=tuple(noisy)))[0]

    def test_each_test_fails_on_its_own(self, digest: str) -> None:
        then = a_fit(digest, FOREIGN)
        assert differences(then, a_fit(digest, "x", samples=501)) == ["samples 500 -> 501"]
        flipped = differences(then, a_fit(digest, "x", accepted=False, accept_score=0.4792))
        assert flipped == ["verdict accepted 0.6250 -> refused 0.4792"]


class TestTheCheck:
    def test_a_store_made_on_the_loaded_core_is_a_no_op(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        fit = stored(digest, tmp_path, build=built_fingerprint())
        before = store_path(digest, fit.procedure, tmp_path).read_bytes()

        found = check_fit(spec, store=tmp_path)

        assert found.outcome == MADE_HERE and found.fit == fit
        assert fitter.calls == 0
        assert store_path(digest, fit.procedure, tmp_path).read_bytes() == before
        assert not (tmp_path / fitting.REPRODUCED_DIR).exists()

    def test_a_foreign_store_that_reproduces_gains_the_core(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        fit = stored(digest, tmp_path)
        before = store_path(digest, fit.procedure, tmp_path).read_bytes()
        core = built_fingerprint()

        found = check_fit(spec, store=tmp_path)

        assert found.outcome == REPRODUCED and fitter.calls == 1
        # The store is untouched, its score included: its `build` still names the core
        # whose games the numbers came from.
        assert store_path(digest, fit.procedure, tmp_path).read_bytes() == before
        assert found.fit == fit
        record = json.loads(sidecar_path(digest, fit.procedure, tmp_path).read_text(encoding="utf-8"))
        assert record == {
            "store_digest": store_digest(fit),
            "made_on": FOREIGN,
            "reproduced_on": [core],
        }
        assert verified_cores(fit, tmp_path) == {FOREIGN, core}

    def test_a_second_check_on_the_same_core_plays_nothing(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        stored(digest, tmp_path)
        check_fit(spec, store=tmp_path)
        again = check_fit(spec, store=tmp_path)
        assert again.outcome == REPRODUCED_BEFORE
        assert fitter.calls == 1
        assert reproduced_on(again.fit, tmp_path) == (built_fingerprint(),)

    def test_a_later_core_is_appended_rather_than_replacing_the_first(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        fit = stored(digest, tmp_path)
        sidecar = sidecar_path(digest, fit.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(fit), "made_on": FOREIGN,
            "reproduced_on": ["1111111111111111"],
        }))
        check_fit(spec, store=tmp_path)
        assert reproduced_on(fit, tmp_path) == ("1111111111111111", built_fingerprint())

    def test_a_perturbed_vector_is_refit(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        fit = stored(digest, tmp_path, vector=vector(material=1.0, centrality=0.31))
        found = check_fit(spec, store=tmp_path)

        assert found.outcome == REFIT
        assert found.failures and "vector moves" in found.failures[0]
        on_disk = FittedObjective.load(store_path(digest, fit.procedure, tmp_path))
        assert on_disk == fitter.fresh and on_disk.build == built_fingerprint()

    def test_a_refit_takes_the_old_sidecar_with_it(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        """The sidecar vouched for a store that no longer exists."""
        fit = stored(digest, tmp_path, samples=499)
        sidecar = sidecar_path(digest, fit.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(fit), "made_on": FOREIGN,
            "reproduced_on": ["1111111111111111"],
        }))
        assert check_fit(spec, store=tmp_path).outcome == REFIT
        assert not sidecar.exists()

    def test_a_verdict_flip_is_refit_even_when_the_games_agree(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        """Q231: the match is deeper than the fit games and can move on its own."""
        stored(digest, tmp_path, accepted=False, accept_score=0.4896)
        found = check_fit(spec, store=tmp_path)
        assert found.outcome == REFIT
        assert found.failures == ("verdict refused 0.4896 -> accepted 0.6042",)
        assert found.fit.accepted

    def test_an_unstamped_store_is_checked_rather_than_trusted(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        fit = stored(digest, tmp_path, build="")
        assert verified_cores(fit, tmp_path) == frozenset()
        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED
        assert verified_cores(fit, tmp_path) == {built_fingerprint()}

    def test_no_store_is_fitted(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        found = check_fit(spec, store=tmp_path)
        assert found.outcome == FITTED and found.stored is None
        assert FittedObjective.load(found.path) == fitter.fresh


class TestTheSidecar:
    def test_it_is_invisible_to_the_store_reader(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        """`load_all` globs `<store>/*.json`, which is why the sidecar sits one down."""
        stored(digest, tmp_path)
        check_fit(spec, store=tmp_path)
        assert sidecar_path(digest, procedure_digest(), tmp_path).exists()
        assert load_all(tmp_path) == 1

    def test_a_sidecar_for_another_store_vouches_for_nothing(
        self, digest: str, tmp_path: Path
    ) -> None:
        """A store refit by hand after the check leaves a sidecar about a different file."""
        fit = stored(digest, tmp_path)
        sidecar = sidecar_path(digest, fit.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        other = a_fit(digest, FOREIGN, loss=0.4)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(other), "made_on": FOREIGN,
            "reproduced_on": [built_fingerprint()],
        }))
        assert reproduced_on(fit, tmp_path) == ()
        assert verified_cores(fit, tmp_path) == {FOREIGN}

    def test_the_digest_covers_the_whole_store(self, digest: str) -> None:
        fit = a_fit(digest, FOREIGN)
        assert store_digest(fit) == store_digest(a_fit(digest, FOREIGN))
        assert store_digest(a_fit(digest, "1111111111111111")) != store_digest(fit)
        for change in ({"accept_score": 0.6}, {"loss": 0.4}):
            assert store_digest(a_fit(digest, FOREIGN, **change)) != store_digest(fit)

    def test_a_reproduced_store_is_reused_without_the_warning(
        self,
        spec: dict[str, Any],
        digest: str,
        fitter: Fitter,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`ensure_fit`'s stderr line says the verdict may not hold; after a check it does."""
        stored(digest, tmp_path)
        ensure_fit(spec, store=tmp_path)
        assert "may no longer hold" in capsys.readouterr().err
        check_fit(spec, store=tmp_path)
        ensure_fit(spec, store=tmp_path)
        assert capsys.readouterr().err == ""


class TestTheCommand:
    def test_check_reports_and_exits_zero_whatever_the_verdict(
        self,
        spec: dict[str, Any],
        digest: str,
        fitter: Fitter,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stored(digest, tmp_path, accepted=False, accept_score=0.25)
        fitter.fresh = a_fit(digest, built_fingerprint(), accepted=False, accept_score=0.3)
        ruleset = str(RULESETS / "gardner.json")

        assert cli.main(["fit", ruleset, "--store", str(tmp_path), "--check"]) == 0
        out = capsys.readouterr().out
        assert f"made on core {FOREIGN}, reproduces on {built_fingerprint()}" in out
        assert "score 0.2500 -> 0.3000" in out
        assert "NOTE" not in out

        # The plain command reads the sidecar too, rather than warning about a verdict a
        # check has just shown holds.
        assert cli.main(["fit", ruleset, "--store", str(tmp_path)]) == 1
        assert "NOTE" not in capsys.readouterr().out

    def test_a_refit_names_what_moved(
        self,
        digest: str,
        fitter: Fitter,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stored(digest, tmp_path, samples=277)
        ruleset = str(RULESETS / "gardner.json")
        assert cli.main(["fit", ruleset, "--store", str(tmp_path), "--check"]) == 0
        out = capsys.readouterr().out
        assert "does NOT reproduce" in out and "samples 277 -> 500" in out

    def test_check_and_force_are_two_answers_to_one_question(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cli.main(["fit", str(RULESETS / "gardner.json"), "--store", str(tmp_path),
                      "--check", "--force"])


class TestOnRealGames:
    """The real procedure, on the one ruleset where it costs no acceptance match."""

    @pytest.fixture
    def short(self) -> dict[str, Any]:
        from chess_variant_sandbox.metrics.mutants import build_mutants

        bases = {}
        for path in sorted(RULESETS.glob("*.json")):
            base = load_spec(path)
            bases[base["name"]] = base
        return next(
            spec for mutant, spec in build_mutants(bases)
            if mutant.name == "mutant-cramped-board"
        )

    def test_a_foreign_store_reproduces_and_a_perturbed_one_is_refit(
        self, short: dict[str, Any], tmp_path: Path
    ) -> None:
        made = check_fit(short, store=tmp_path, workers=1)
        assert made.outcome == FITTED and made.fit.samples == 0

        # The same verdict filed as another core's: the games reproduce it.
        foreign = FittedObjective(**{**made.fit.__dict__, "build": FOREIGN})
        foreign.save(made.path)
        assert check_fit(short, store=tmp_path, workers=1).outcome == REPRODUCED
        assert verified_cores(foreign, tmp_path) == {FOREIGN, built_fingerprint()}

        # A vector no game on this core produces: refit, and back to the empty one.
        bent = FittedObjective(**{**foreign.__dict__, "vector": vector(mobility=1.0)})
        bent.save(made.path)
        refit = check_fit(short, store=tmp_path, workers=1)
        assert refit.outcome == REFIT
        assert FittedObjective.load(made.path) == made.fit


def test_the_check_moves_no_fingerprint() -> None:
    """The design's claim: the check's code sits in no fingerprint set, so landing it
    stales no measurement (`refit-repair.md` section 3)."""
    root = Path(fitting.__file__).resolve().parents[2]
    for kind in ("engine", "metrics", "analysis"):
        read = {path.resolve() for path in fingerprinted(kind)}
        assert (root / "chess_variant_sandbox" / "tournament" / "fitting.py") not in read
        assert (root / "chess_variant_sandbox" / "cli.py") not in read
