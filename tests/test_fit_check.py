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
    record_made,
    reproduced_on,
    reproduction_basis,
    save_fit,
    sidecar_path,
    store_digest,
    unverified_reason,
    verified_cores,
    verified_under,
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
        fit = a_fit(digest, built_fingerprint())
        save_fit(fit, spec, tmp_path)
        before = store_path(digest, fit.procedure, tmp_path).read_bytes()
        sidecar = sidecar_path(digest, fit.procedure, tmp_path).read_bytes()

        found = check_fit(spec, store=tmp_path)

        assert found.outcome == MADE_HERE and found.fit == fit
        assert fitter.calls == 0
        assert store_path(digest, fit.procedure, tmp_path).read_bytes() == before
        assert sidecar_path(digest, fit.procedure, tmp_path).read_bytes() == sidecar

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
            "verified_under": [reproduction_basis(spec)],
        }
        # The maker is not counted: nothing records the tables its games were played on.
        assert verified_cores(fit, tmp_path, spec) == {core}

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
        found = check_fit(spec, store=tmp_path)
        assert found.outcome == REFIT
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert record["store_digest"] == store_digest(found.fit)
        assert record["reproduced_on"] == []
        assert reproduced_on(fit, tmp_path) == ()
        # The new store is its maker's answer and needs no second fit.
        assert verified_under(found.fit, tmp_path) == (reproduction_basis(spec),)
        assert check_fit(spec, store=tmp_path).outcome == MADE_HERE
        assert fitter.calls == 1

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
        assert verified_cores(fit, tmp_path, spec) == frozenset()
        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED
        assert verified_cores(fit, tmp_path, spec) == {built_fingerprint()}

    def test_no_store_is_fitted(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        found = check_fit(spec, store=tmp_path)
        assert found.outcome == FITTED and found.stored is None
        assert FittedObjective.load(found.path) == fitter.fresh
        assert verified_cores(found.fit, tmp_path, spec) == {built_fingerprint()}


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
        self, spec: dict[str, Any], digest: str, tmp_path: Path
    ) -> None:
        """A store refit by hand after the check leaves a sidecar about a different file."""
        fit = stored(digest, tmp_path)
        sidecar = sidecar_path(digest, fit.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        other = a_fit(digest, FOREIGN, loss=0.4)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(other), "made_on": FOREIGN,
            "reproduced_on": [built_fingerprint()],
            "verified_under": [reproduction_basis(spec)],
        }))
        assert reproduced_on(fit, tmp_path) == ()
        assert verified_under(fit, tmp_path) == ()
        assert verified_cores(fit, tmp_path, spec) == frozenset()

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


class TestTheBasis:
    """Q892: a sidecar vouches for the core *and* what else the fit's games read.

    Q880 moved *quarry*'s compiled tables with no core change, and its store, listed on
    the core, went unrederived until a new core arrived. A part of the basis moving on
    its own is simulated by moving the fingerprint the basis reads, as a compiler edit
    moves `tables_fingerprint` and nothing else.
    """

    @staticmethod
    def _move(monkeypatch: pytest.MonkeyPatch, part: str) -> None:
        from chess_variant_sandbox import measurement

        name = {
            "tables": "tables_fingerprint",
            "players": "players_fingerprint",
            "runner": "source_fingerprint",
        }[part]
        real = getattr(measurement, name)

        def moved(*args: Any) -> str:
            return "moved-" + real(*args)

        monkeypatch.setattr(measurement, name, moved)

    def test_matching_core_and_tables_keep_the_store_verified(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        fit = a_fit(digest, built_fingerprint())
        save_fit(fit, spec, tmp_path)
        assert verified_cores(fit, tmp_path, spec) == {built_fingerprint()}
        assert unverified_reason(fit, tmp_path, spec) is None
        assert check_fit(spec, store=tmp_path).outcome == MADE_HERE
        assert fitter.calls == 0

    def test_a_tables_change_on_the_same_core_unverifies_the_store(
        self,
        spec: dict[str, Any],
        digest: str,
        fitter: Fitter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fit = a_fit(digest, built_fingerprint())
        save_fit(fit, spec, tmp_path)
        before = reproduction_basis(spec)
        self._move(monkeypatch, "tables")
        after = reproduction_basis(spec)
        assert after["core"] == before["core"] and after["tables"] != before["tables"]

        assert verified_cores(fit, tmp_path, spec) == frozenset()
        reason = unverified_reason(fit, tmp_path, spec)
        assert reason is not None and "under other tables," in reason

        found = check_fit(spec, store=tmp_path)
        assert found.outcome == REPRODUCED and fitter.calls == 1
        assert verified_under(fit, tmp_path) == (before, after)
        # Made here, so the history of other cores gains nothing.
        assert reproduced_on(fit, tmp_path) == ()
        assert verified_cores(fit, tmp_path, spec) == {built_fingerprint()}
        assert check_fit(spec, store=tmp_path).outcome == MADE_HERE
        assert fitter.calls == 1

    @pytest.mark.parametrize("part", ["players", "runner"])
    def test_every_other_part_unverifies_it_too(
        self,
        spec: dict[str, Any],
        digest: str,
        fitter: Fitter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        part: str,
    ) -> None:
        fit = stored(digest, tmp_path)
        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED
        self._move(monkeypatch, part)
        assert verified_cores(fit, tmp_path, spec) == frozenset()
        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED
        assert fitter.calls == 2
        # One core in the history, two bases verified on it.
        assert reproduced_on(fit, tmp_path) == (built_fingerprint(),)
        assert len(verified_under(fit, tmp_path)) == 2

    def test_a_sidecar_from_before_the_basis_is_checked_once(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        """The loaded core listed in an old sidecar is history, not a verification:
        nothing recorded the tables that check ran on."""
        fit = stored(digest, tmp_path)
        sidecar = sidecar_path(digest, fit.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(fit), "made_on": FOREIGN,
            "reproduced_on": ["1111111111111111", built_fingerprint()],
        }))
        assert verified_cores(fit, tmp_path, spec) == frozenset()

        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED
        assert fitter.calls == 1
        assert reproduced_on(fit, tmp_path) == ("1111111111111111", built_fingerprint())
        assert verified_under(fit, tmp_path) == (reproduction_basis(spec),)
        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED_BEFORE
        assert fitter.calls == 1

    def test_a_store_made_here_with_no_record_is_checked_once(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        """A store saved before Q892, or by hand, says its core and not its tables."""
        fit = stored(digest, tmp_path, build=built_fingerprint())
        assert verified_cores(fit, tmp_path, spec) == frozenset()
        reason = unverified_reason(fit, tmp_path, spec)
        assert reason is not None and "nothing records the tables" in reason
        assert check_fit(spec, store=tmp_path).outcome == REPRODUCED
        assert check_fit(spec, store=tmp_path).outcome == MADE_HERE
        assert fitter.calls == 1

    def test_no_spec_is_verified_nowhere(
        self, spec: dict[str, Any], digest: str, tmp_path: Path
    ) -> None:
        fit = a_fit(digest, built_fingerprint())
        save_fit(fit, spec, tmp_path)
        assert verified_cores(fit, tmp_path, None) == frozenset()

    def test_recording_a_maker_replaces_what_vouched_for_the_old_store(
        self, spec: dict[str, Any], digest: str, tmp_path: Path
    ) -> None:
        old = stored(digest, tmp_path)
        sidecar = sidecar_path(digest, old.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(old), "made_on": FOREIGN,
            "reproduced_on": ["1111111111111111"],
        }))
        new = a_fit(digest, built_fingerprint(), samples=600)
        new.save(store_path(digest, new.procedure, tmp_path))
        record_made(new, spec, tmp_path)
        assert reproduced_on(new, tmp_path) == ()
        assert verified_cores(new, tmp_path, spec) == {built_fingerprint()}

    def test_the_plain_command_records_its_maker(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path
    ) -> None:
        ruleset = str(RULESETS / "gardner.json")
        assert cli.main(["fit", ruleset, "--store", str(tmp_path), "--force"]) == 0
        fit = FittedObjective.load(store_path(digest, procedure_digest(), tmp_path))
        assert verified_cores(fit, tmp_path, spec) == {built_fingerprint()}


class TestWhatDoesNotMatch:
    """Q894: *stale (refit)* and `fit --check` name the part of the basis that differs.

    Until Q892 the sentence said "made on core X and has not been re-derived on this one"
    whatever the reason, which is false when X is the loaded core and the tables, players
    or runner moved, or when nothing recorded them.
    """

    @staticmethod
    def _summary(spec: dict[str, Any], digest: str, root: Path) -> str:
        from dataclasses import replace

        from chess_variant_sandbox.corpus import fit_reading, player_digest
        from chess_variant_sandbox.measurement import Basis

        reading = fit_reading(digest, root, spec)
        assert reading is not None
        current = Basis.current(
            ruleset_hash=digest, weights_version="v12", agents=("random", "greedy"),
            games_per_pairing=4, seeds=(0,), tables="t", fit=reading,
        )
        # A stamp taken on this core that recorded the store's player.
        stamp = replace(current, fit=player_digest(stored_fit(digest, root)))
        return stamp.stale_against(current).summary()

    def test_a_tables_change_names_the_tables(
        self, spec: dict[str, Any], digest: str, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        save_fit(a_fit(digest, built_fingerprint()), spec, tmp_path)
        TestTheBasis._move(monkeypatch, "tables")
        assert self._summary(spec, digest, tmp_path) == (
            "stale (refit): its fit was last verified where the tables differ, and has "
            "not been re-derived on this one"
        )

    def test_a_core_change_names_the_core_that_made_it(
        self, spec: dict[str, Any], digest: str, tmp_path: Path
    ) -> None:
        fit = stored(digest, tmp_path)
        other = {**reproduction_basis(spec), "core": FOREIGN}
        sidecar = sidecar_path(digest, fit.procedure, tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({
            "store_digest": store_digest(fit), "made_on": FOREIGN,
            "reproduced_on": [], "verified_under": [other],
        }))
        assert self._summary(spec, digest, tmp_path) == (
            f"stale (refit): its fit was made on core {FOREIGN} and has not been "
            "re-derived on this one"
        )

    def test_no_record_says_nothing_records_the_basis(
        self, spec: dict[str, Any], digest: str, tmp_path: Path
    ) -> None:
        stored(digest, tmp_path, build=built_fingerprint())
        assert self._summary(spec, digest, tmp_path) == (
            f"stale (refit): its fit was made on this core ({built_fingerprint()}), but "
            "nothing records the tables, players and runner its games read, so it has not "
            "been re-derived under them"
        )

    def test_the_check_line_for_a_store_made_here_with_no_record(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stored(digest, tmp_path, build=built_fingerprint())
        ruleset = str(RULESETS / "gardner.json")
        assert cli.main(["fit", ruleset, "--store", str(tmp_path), "--check"]) == 0
        out = capsys.readouterr().out
        assert (
            f"made on core {built_fingerprint()}, the loaded one, with no record of the "
            "tables, players and runner it read, reproduces on "
        ) in out

    def test_the_check_line_for_a_tables_change(
        self, spec: dict[str, Any], digest: str, fitter: Fitter, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        save_fit(a_fit(digest, built_fingerprint()), spec, tmp_path)
        TestTheBasis._move(monkeypatch, "tables")
        ruleset = str(RULESETS / "gardner.json")
        assert cli.main(["fit", ruleset, "--store", str(tmp_path), "--check"]) == 0
        out = capsys.readouterr().out
        assert (
            f"made on core {built_fingerprint()}, the loaded one, last verified where "
            "the tables differ, reproduces on "
        ) in out


def stored_fit(digest: str, root: Path) -> FittedObjective:
    return FittedObjective.load(store_path(digest, procedure_digest(), root))


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
        assert verified_cores(foreign, tmp_path, short) == {built_fingerprint()}

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
