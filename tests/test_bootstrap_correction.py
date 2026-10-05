"""A stratified resample reads each stratum as its own population, and the sd is short.

Drawing n games with replacement from a stratum of n estimates that stratum's variance
by the plug-in formula, which is (n-1)/n of the sample variance. So every spread the
bootstrap reports is short by sqrt(n/(n-1)) -- 1.155 at the canonical four games a
pairing -- and `bootstrap_index` multiplies it back.

Four replays of the whole library measured the published sd 14% short and found nothing
beyond this factor (`docs/reference/run-to-run-spread.md` §1).

The fixtures are synthetic strata and synthetic records on ``rulesets/dull.json``, so
nothing here reads chess.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import pytest

from chess_variant_sandbox.metrics import bootstrap
from chess_variant_sandbox.metrics.bootstrap import bootstrap_index, stratum_correction
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.record import GameRecord

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
AGENTS = ("minimax:nodes=100", "minimax:nodes=200", "minimax:nodes=400")
RESAMPLES = 60


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec(RULESETS / "dull.json")


def _corpus(games_per_pairing: dict[tuple[str, str], int]) -> list[GameRecord]:
    """Synthetic records with random move lists, one stratum per ordered pairing."""
    rng = random.Random(11)
    records: list[GameRecord] = []
    seed = 0
    for pairing, count in games_per_pairing.items():
        for _ in range(count):
            records.append(
                GameRecord(
                    ruleset_hash="fixture", ruleset_name="fixture", seed=seed,
                    agents=pairing, moves=[rng.randrange(50) for _ in range(30)],
                    winner=rng.choice([0, 1, None]), reason="fixture", plies=30,
                    branching=[20] * 30, material=[0.0] * 30,
                )
            )
            seed += 1
    return records


def _even(count: int) -> dict[tuple[str, str], int]:
    return {(first, second): count for first in AGENTS for second in AGENTS}


def _bootstrap(spec: dict, records: list[GameRecord]) -> bootstrap.BootstrapResult:
    result = bootstrap_index(
        spec, records, load_weights("v2"), resamples=RESAMPLES, seed=0,
        components_of_interest=("stall_rate", "ply_spread"),
    )
    assert result is not None
    return result


class TestTheFactorItself:
    def test_four_games_a_stratum_owes_the_measured_one_one_five_five(self) -> None:
        assert stratum_correction([4] * 9) == pytest.approx((4 / 3) ** 0.5, abs=1e-12)
        assert stratum_correction([4] * 9) == pytest.approx(1.1547, abs=5e-5)

    def test_a_bigger_stratum_owes_less(self) -> None:
        owed = [stratum_correction([n]) for n in (2, 4, 8, 64)]
        assert owed == sorted(owed, reverse=True)
        assert owed[0] == pytest.approx(2**0.5)
        assert owed[-1] < 1.01

    def test_mixed_sizes_pool_by_stratum_size(self) -> None:
        """Three strata of four and one of two, weighted by the games in each."""
        expected = ((3 * 4 * 4 / 3 + 2 * 2 / 1) / (3 * 4 + 2)) ** 0.5
        assert stratum_correction([4, 4, 4, 2]) == pytest.approx(expected)
        assert min(stratum_correction([4]), stratum_correction([2])) <= expected
        assert expected <= max(stratum_correction([4]), stratum_correction([2]))

    def test_a_stratum_of_one_cannot_be_corrected_and_is_left_out(self) -> None:
        """One game resamples to itself. It adds no variance and n/(n-1) is undefined,
        so it changes no factor -- and the interval stays short by what it carries."""
        assert stratum_correction([4, 4, 1, 1]) == stratum_correction([4, 4])
        assert stratum_correction([1] * 20) == 1.0
        assert stratum_correction([]) == 1.0


class TestWhatTheFactorRepairs:
    """The identity the correction rests on, on numbers rather than on games."""

    def test_it_puts_a_stratified_mean_back_on_the_sample_variance(self) -> None:
        rng = random.Random(3)
        strata = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(9)]
        total = sum(len(values) for values in strata)

        plug_in = sum(
            len(values) * statistics.pvariance(values) for values in strata
        ) / total**2
        unbiased = sum(
            len(values) * statistics.variance(values) for values in strata
        ) / total**2

        factor = stratum_correction(len(values) for values in strata)
        assert plug_in * factor**2 == pytest.approx(unbiased)

    def test_the_resample_really_does_report_the_plug_in_spread(self) -> None:
        """Monte Carlo, so that the identity above is checked against drawing, not only
        against the algebra it was derived from."""
        rng = random.Random(5)
        strata = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(9)]
        total = sum(len(values) for values in strata)
        drawn = [
            sum(sum(rng.choices(values, k=len(values))) for values in strata) / total
            for _ in range(20000)
        ]
        plug_in = sum(
            len(values) * statistics.pvariance(values) for values in strata
        ) / total**2
        assert statistics.pstdev(drawn) == pytest.approx(plug_in**0.5, rel=0.03)


Pair = tuple[bootstrap.BootstrapResult, bootstrap.BootstrapResult]


class TestTheIndexIntervalCarriesIt:
    @pytest.fixture(scope="class")
    @classmethod
    def pair(cls, spec: dict) -> Pair:
        """One even schedule, bootstrapped with the factor and without it.

        Computed once for the three tests that read it, where each used to compute both
        (Q377).
        """
        records = _corpus(_even(4))
        corrected = _bootstrap(spec, records)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(bootstrap, "stratum_correction", lambda sizes: 1.0)
            plain = _bootstrap(spec, records)
        return corrected, plain

    def test_the_published_sd_is_the_resampled_sd_times_the_factor(self, pair: Pair) -> None:
        corrected, plain = pair
        assert corrected.stratum_correction == pytest.approx(1.1547, abs=5e-5)
        assert plain.stratum_correction == 1.0
        assert plain.sd > 0.0
        assert corrected.sd == pytest.approx(plain.sd * 1.1547, abs=1e-4)

    def test_it_widens_the_interval_and_moves_no_location(self, pair: Pair) -> None:
        corrected, plain = pair

        assert corrected.mean == plain.mean
        assert corrected.component_mean == plain.component_mean
        assert corrected.width > plain.width
        assert corrected.ci_low < plain.ci_low
        assert corrected.ci_high > plain.ci_high
        for end in (corrected.ci_low, corrected.ci_high):
            assert abs(end - corrected.mean) == pytest.approx(
                1.1547 * abs(
                    (plain.ci_low if end < corrected.mean else plain.ci_high)
                    - plain.mean
                ),
                abs=1e-4,
            )

    def test_every_tracked_component_is_corrected_the_same_way(self, pair: Pair) -> None:
        corrected, plain = pair
        assert set(corrected.component_sd) == set(plain.component_sd)
        for name, sd in plain.component_sd.items():
            assert corrected.component_sd[name] == pytest.approx(sd * 1.1547, abs=1e-4)

    def test_a_schedule_with_single_game_pairings_corrects_only_the_rest(
        self, spec: dict
    ) -> None:
        """The pairings of one carry no resampling variance, so they set no factor."""
        counts = _even(4)
        singles = list(counts)[:3]
        for pairing in singles:
            counts[pairing] = 1
        result = _bootstrap(spec, _corpus(counts))
        assert result.stratum_correction == pytest.approx(1.1547, abs=5e-5)
