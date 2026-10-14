"""`Basis.fit`: which objective the roster's fitted rungs played (Q80c).

`docs/reference/refit-repair.md` is the design. A fit is stamped with the core that made
it and nothing else, so the basis records the *player* -- the built-in objective, or a
digest of the procedure and the vector -- and a store neither made nor reproduced on the
loaded core reads `stale (refit)`. `tests/test_corpus.py` holds what `refresh` does with it
(Q80d); these hold the reading.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.fitted import FittedObjective, procedure_digest, store_path, unit
from chess_variant_sandbox.corpus import fit_reading, player_digest
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import (
    BUILTIN_PLAYER,
    REPAIR_NAMES,
    SINGLE_RUN_FORM,
    UNVERIFIED,
    Basis,
    FitReading,
    built_fingerprint,
    fingerprinted,
    load,
)
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import fitting

FOREIGN = "0000000000000000"
RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
ACCEPTED = "fit:aaaaaaaaaaaa"
OTHER = "fit:bbbbbbbbbbbb"


def now(reading: FitReading | None, **changes: Any) -> Basis:
    """The basis a measurement taken now would carry, told about this store."""
    basis = Basis.current(
        ruleset_hash="0123456789ab",
        weights_version="v11",
        agents=("random", "greedy"),
        games_per_pairing=4,
        seeds=(0,),
        tables="t",
        fit=reading,
    )
    return replace(basis, **changes) if changes else basis


def here(player: str = ACCEPTED, *also: str) -> FitReading:
    """A store made on the loaded core, or reproduced on it."""
    return FitReading(player, built_fingerprint(), frozenset({built_fingerprint(), *also}))


def elsewhere(player: str = ACCEPTED) -> FitReading:
    """A store made on another core and never reproduced on this one."""
    return FitReading(player, FOREIGN, frozenset({FOREIGN}))


def stamp(fit: str, reading: FitReading | None = None) -> Basis:
    """A stored stamp, taken on the loaded core, recording `fit` (`""` records none)."""
    return replace(now(reading or here()), fit=fit)


class TestTheReading:
    def test_a_fit_made_here_holds(self) -> None:
        assert stamp(ACCEPTED).stale_against(now(here())).current

    def test_a_fit_made_elsewhere_reads_stale_refit_and_nothing_else(self) -> None:
        current = now(elsewhere())
        assert current.fit == UNVERIFIED

        staleness = stamp(ACCEPTED).stale_against(current)

        assert staleness.needs == "refit" and staleness.refit
        assert staleness.cost() == "refit"
        assert staleness.summary() == (
            f"stale (refit): its fit was made on core {FOREIGN} and has not been "
            "re-derived on this one"
        )

    def test_a_refit_is_owed_beside_a_stronger_repair_and_names_none_of_it(self) -> None:
        stored = replace(stamp(ACCEPTED), metrics="moved")
        staleness = stored.stale_against(now(elsewhere()))
        assert staleness.needs == "rescore"
        assert staleness.refit

    def test_a_store_with_no_core_says_so(self) -> None:
        staleness = stamp(ACCEPTED).stale_against(
            now(FitReading(ACCEPTED, "", frozenset()))
        )
        assert "records no core" in staleness.summary()

    def test_a_reproduced_fit_holds_for_a_stamp_that_records_none(self) -> None:
        """The absent reading: the store decides where the stamp's core made or reproduced it."""
        reproduced = FitReading(ACCEPTED, FOREIGN, frozenset({FOREIGN, built_fingerprint()}))
        assert stamp("").stale_against(now(reproduced)).current

    def test_a_silent_stamp_on_an_unverified_fit_owes_the_refit(self) -> None:
        staleness = stamp("").stale_against(now(elsewhere()))
        assert staleness.needs == "refit"

    def test_a_silent_stamp_from_another_core_is_replayed_as_its_core_says(self) -> None:
        stored = replace(stamp(""), engine="sources-" + FOREIGN)
        staleness = stored.stale_against(now(here()))
        assert staleness.needs == "replay"
        assert any("does not record which objective played" in r for r in staleness.reasons)

    def test_an_accepted_vector_that_changed_is_a_replay(self) -> None:
        staleness = stamp(ACCEPTED).stale_against(now(here(OTHER)))
        assert staleness.needs == "replay" and not staleness.refit
        assert f"it played the fitted objective {ACCEPTED}" in staleness.summary()
        assert f"the fitted objective {OTHER} plays now" in staleness.summary()

    def test_a_fit_that_became_refused_is_a_replay(self) -> None:
        staleness = stamp(ACCEPTED).stale_against(now(here(BUILTIN_PLAYER)))
        assert staleness.needs == "replay"
        assert "the built-in objective plays now" in staleness.summary()

    def test_a_stamp_recording_unverified_is_a_replay_once_the_fit_is_verified(self) -> None:
        """Nothing says what such a measurement played, so the refit cannot clear it."""
        assert stamp(UNVERIFIED).stale_against(now(here())).needs == "replay"

    def test_a_basis_told_nothing_checks_nothing(self) -> None:
        current = now(None)
        assert current.fit == ""
        assert stamp(ACCEPTED).stale_against(current).current
        assert stamp("").stale_against(current).current

    def test_the_single_run_form_owes_a_re_score_over_a_refit(self, tmp_path: Path) -> None:
        path = tmp_path / "canonical.json"
        path.write_text(json.dumps({"hash": "x", "basis": stamp(ACCEPTED).as_dict()}))
        found = load(path, now(elsewhere()))
        assert found is not None
        assert found.staleness.needs == "rescore" and found.staleness.refit
        assert SINGLE_RUN_FORM in found.staleness.reasons

    def test_refit_is_the_weakest_repair_named(self) -> None:
        assert next(iter(REPAIR_NAMES)) == "refit"


class TestTheStamp:
    def test_the_field_round_trips_and_is_written_only_when_it_says_something(self) -> None:
        written = stamp(ACCEPTED).as_dict()
        assert written["fit"] == ACCEPTED
        back = Basis.from_dict(written)
        assert back is not None and back.fit == ACCEPTED
        assert "fit" not in stamp("").as_dict()

    def test_a_stamp_from_before_the_field_reads_as_recording_none(self) -> None:
        written = stamp(ACCEPTED).as_dict()
        del written["fit"]
        back = Basis.from_dict(written)
        assert back is not None and back.fit == ""

    def test_no_fingerprinted_source_is_touched(self) -> None:
        """The note's claim: the code is in no fingerprint set (`agents/fitted.py` moves only
        in docstrings, which the code reading drops)."""
        for kind in ("engine", "analysis", "metrics"):
            names = {path.name for path in fingerprinted(kind)}
            assert "fitting.py" not in names
            assert "corpus.py" not in names
            assert "measurement.py" not in names


def a_fit(build: str = FOREIGN, **changes: Any) -> FittedObjective:
    weights = [0.0] * len(VOCABULARY)
    weights[VOCABULARY.index("material")] = 1.0
    weights[VOCABULARY.index("centrality")] = 0.3
    fields: dict[str, Any] = {
        "ruleset_hash": "0123456789ab", "procedure": procedure_digest(),
        "vector": unit(tuple(weights)), "accepted": True, "accept_score": 0.625,
        "accept_games": 48, "samples": 500, "loss": 0.5, "build": build,
    }
    return FittedObjective(**{**fields, **changes})


class TestThePlayer:
    def test_an_accepted_fit_is_named_by_its_procedure_and_vector(self) -> None:
        digest = player_digest(a_fit())
        assert digest.startswith("fit:") and len(digest) == len("fit:") + 12
        # Score, samples, loss and core do not change the player.
        assert player_digest(a_fit("x", accept_score=0.9, samples=7, loss=0.1)) == digest
        moved = list(a_fit().vector)
        moved[0] += 1e-6
        assert player_digest(a_fit(vector=tuple(moved))) != digest

    def test_a_refused_fit_is_the_built_in_objective_whatever_its_vector(self) -> None:
        refused = a_fit(accepted=False)
        assert player_digest(refused) == BUILTIN_PLAYER
        drifted = a_fit(accepted=False, vector=tuple(reversed(refused.vector)))
        assert player_digest(drifted) == BUILTIN_PLAYER


class TestTheStoreReading:
    def test_no_store_is_no_reading(self, tmp_path: Path) -> None:
        assert fit_reading("0123456789ab", tmp_path) is None

    def test_the_reading_names_the_maker_and_every_core_verified_under_this_game(
        self, tmp_path: Path
    ) -> None:
        """Q892: a core counts only where the sidecar records it with this game's tables,
        players and runner as they are now, so the reading needs the spec to say so."""
        spec = load_spec(RULESETS / "gardner.json")
        fit = a_fit(ruleset_hash=Engine(spec).spec.hash)
        fit.save(store_path(fit.ruleset_hash, fit.procedure, tmp_path))
        reading = fit_reading(fit.ruleset_hash, tmp_path, spec)
        assert reading == FitReading(player_digest(fit), FOREIGN, frozenset(), recorded=False)

        fitting._add_basis(fit, fitting.reproduction_basis(spec), tmp_path)
        reading = fit_reading(fit.ruleset_hash, tmp_path, spec)
        assert reading is not None
        assert reading.verified_on == {built_fingerprint()}
        assert now(reading).fit == player_digest(fit)
        # Without the spec the tables cannot be compared: verified nowhere.
        blind = fit_reading(fit.ruleset_hash, tmp_path)
        assert blind is not None and blind.verified_on == frozenset()

    def test_an_unreadable_store_is_no_reading(self, tmp_path: Path) -> None:
        path = store_path("0123456789ab", procedure_digest(), tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocabulary_version": -1}))
        assert fit_reading("0123456789ab", tmp_path) is None


@pytest.mark.parametrize("reading", [None, "store"])
def test_current_basis_finds_the_fit_store_by_hash_and_verifies_it_with_the_spec(
    reading: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.agents.fitted import STORE_ENV

    monkeypatch.setenv(STORE_ENV, str(tmp_path))
    # No rated net, or gardner's roster would be its net ladder and record no fit (Q502h).
    monkeypatch.setattr(corpus, "NETS_DIR", tmp_path / "no-nets")
    spec = load_spec(RULESETS / "gardner.json")
    digest = Engine(spec).spec.hash
    fit = a_fit(built_fingerprint(), ruleset_hash=digest)
    if reading:
        fitting.save_fit(fit, spec, tmp_path)
    basis = corpus.current_basis(digest, spec=spec)
    assert basis.fit == (player_digest(fit) if reading else "")
    blind = corpus.current_basis(digest)
    assert blind.fit == (UNVERIFIED if reading else "")
