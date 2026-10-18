"""How `v3` to `v11` placed their curve ends, kept so those ends can be re-derived.

Since `v12` no end is placed by the band (`docs/archive/plan-evolving-score.md` §4), so nothing in
the scoring path calls these. They stay for the record: `docs/reference/negatives-widened.py`,
`v10-worst-half-noise.py`, `v11-acceptance.py` and `stage26-partial-write.py` run them, and
`.claude/rules/metrics-and-calibration.md` names `component_noise` as a new component's
precision test. None of them had run in the suite (archive/plan-tests T7, F9; Q751), so these pin what
they do, each from a fact a weights file or a reference record states:

* `calibration.held_out` is the `v8`-salted slice `docs/reference/v11-calibration.md` lists.
* `calibration.anchor_curves` follows the rules its docstring calls not negotiable, and puts
  `v11`'s two notable ends where `weights/v11.json`'s notes say it did.
* `bootstrap.component_noise` is the corpus spread over the mean resampled sd.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import pytest

from chess_variant_sandbox.metrics import calibration
from chess_variant_sandbox.metrics.bootstrap import bootstrap_index, component_noise
from chess_variant_sandbox.metrics.calibration import (
    HEADROOM,
    MIN_SPAN_IN_BANDS,
    MIN_SPAN_IN_NOISE,
    Measured,
    anchor_curves,
    held_out,
)
from chess_variant_sandbox.metrics.components import compute_components
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.ratings import rate
from chess_variant_sandbox.tournament.record import GameRecord

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


class TestTheHeldOutSlice:
    """`docs/reference/v11-calibration.md` §*The membership*: which of `runs/v11`'s
    negatives were held out of every curve, and which were anchored on."""

    HELD = (
        "dull", "rps annihilation", "rps both endings", "rps repaint",
        "mutant-uniform-army", "mutant-bare-kings", "mutant-idle-army",
        "mutant-shogi-uniform", "mutant-torus-uniform", "mutant-territory-no-captures",
        "null-reordered-gardner", "probe-one-piece-rules", "probe-running-king",
        "probe-mirrored-setup", "probe-no-promotion",
    )
    ANCHORED = (
        "bastion", "rps race", "hierarchy chess", "rps territory",
        "mutant-all-pawns", "mutant-no-captures", "mutant-royal-and-pawns",
        "mutant-vast-board", "mutant-cramped-board", "mutant-instant-promotion",
        "mutant-gardner-shuffle", "mutant-gardner-uniform", "mutant-gardner-no-captures",
        "mutant-xiangqi-uniform",
    )

    def test_it_is_the_slice_the_v11_anchor_held_out(self) -> None:
        assert [name for name in self.HELD if not held_out(name)] == []
        assert [name for name in self.ANCHORED if held_out(name)] == []

    def test_it_is_derived_from_the_name_and_the_salt_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names = [f"entry-{n}" for n in range(600)]
        share = sum(map(held_out, names)) / len(names)
        assert abs(share - calibration.HELDOUT_SHARE) < 0.06
        before = [held_out(name) for name in names]
        monkeypatch.setattr(calibration, "HELDOUT_SALT", "v9")
        after = [held_out(name) for name in names]
        # A new salt redraws the whole slice; there is no moving one entry alone.
        assert sum(a != b for a, b in zip(before, after, strict=True)) > 150


def _corpus(name: str, human: list[float], negatives: list[float],
            ladder: float = 1.0) -> list[Measured]:
    return [
        *(Measured(f"h{n}", "human", {name: v, "ladder_monotonicity": ladder})
          for n, v in enumerate(human)),
        *(Measured(f"w{n}", "worse", {name: v}) for n, v in enumerate(negatives)),
    ]


HUMAN = [10.0, 11.0, 12.0, 13.0, 14.0, 16.0]


class TestTheAnchorsRules:
    """`anchor_curves`' own rules, on a corpus small enough to read by eye."""

    def test_a_higher_better_curve_runs_past_the_whole_band(self) -> None:
        points, _ = anchor_curves(_corpus("mean_branching", HUMAN, [1.0, 2.0, 3.0]),
                                  {"mean_branching": {"curve": "higher_better"}})
        width = 16.0 - 10.0
        assert points["mean_branching"]["good_at"] == round(16.0 + HEADROOM * width, 4)
        # The zero end is the negatives' median, unless that is too close to the good end.
        assert points["mean_branching"]["zero_at"] == min(
            2.0, round(16.0 + HEADROOM * width - MIN_SPAN_IN_BANDS * width, 4))

    def test_a_lower_better_curve_starts_at_the_bands_best(self) -> None:
        points, _ = anchor_curves(_corpus("ply_spread", HUMAN, [40.0, 50.0, 60.0]),
                                  {"ply_spread": {"curve": "lower_better"}})
        assert points["ply_spread"] == {"good_at": 10.0, "zero_at": 50.0}

    def test_the_noise_floor_keeps_a_curve_no_sharper_than_its_measurement(self) -> None:
        corpus = _corpus("ply_spread", HUMAN, [20.0])
        config = {"ply_spread": {"curve": "lower_better"}}
        quiet, _ = anchor_curves(corpus, config)
        noisy, _ = anchor_curves(corpus, config, noise={"ply_spread": 8.0})
        assert quiet["ply_spread"]["zero_at"] == 22.0  # two band widths past good_at
        assert noisy["ply_spread"]["zero_at"] == 10.0 + MIN_SPAN_IN_NOISE * 8.0

    def test_a_band_is_the_human_range_with_its_zero_ends_a_floor_apart(self) -> None:
        points, _ = anchor_curves(_corpus("mean_branching", HUMAN, [1.0]),
                                  {"mean_branching": {"curve": "band"}})
        assert points["mean_branching"] == {
            "good_low": 10.0, "good_high": 16.0, "zero_low": 4.0, "zero_high": 22.0,
        }

    def test_an_end_never_leaves_the_quantitys_domain(self) -> None:
        points, _ = anchor_curves(
            _corpus("stall_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.9], [0.5]),
            {"stall_rate": {"curve": "band"}})
        band = points["stall_rate"]
        assert band["zero_low"] == 0.0 and band["zero_high"] == 1.0

    def test_a_ladder_the_roster_cannot_play_is_left_out_and_named(self) -> None:
        clean = [Measured(f"h{n}", "human", {"skill_gradient": v, "mean_branching": v,
                                             "ladder_monotonicity": 1.0})
                 for n, v in enumerate(HUMAN)]
        crooked = Measured("crooked", "human", {"skill_gradient": 99.0, "mean_branching": 99.0,
                                                "ladder_monotonicity": 0.9})
        config = {"skill_gradient": {"curve": "higher_better"},
                  "mean_branching": {"curve": "higher_better"}}
        points, dropped = anchor_curves([*clean, crooked], config)
        # A skill-derived curve demands a perfectly clean ladder, and 0.9 is not one; any
        # other curve asks only that the ladder be measurable, which 0.9 is.
        assert dropped == {"skill_gradient": ["crooked"]}
        assert points["skill_gradient"]["good_at"] == round(16.0 + HEADROOM * 6.0, 4)
        assert points["mean_branching"]["good_at"] == round(99.0 + HEADROOM * 89.0, 4)

    def test_under_six_human_games_places_no_end(self) -> None:
        points, _ = anchor_curves(_corpus("mean_branching", HUMAN[:5], [1.0]),
                                  {"mean_branching": {"curve": "higher_better"}})
        assert points == {}


class TestTheV11EndsTheNotesName:
    """Two ends `weights/v11.json`'s notes explain, re-derived from the bands they quote."""

    @staticmethod
    def _band(low: float, high: float, n: int = 28) -> list[float]:
        return [low + (high - low) * k / (n - 1) for k in range(n)]

    def test_position_divergence_clamps_to_its_domain_at_both_ends(self) -> None:
        """'its band runs 0.214-0.983, and twice that width clamps both ends'."""
        points, _ = anchor_curves(
            _corpus("position_divergence", self._band(0.214, 0.983), [0.1, 0.2]),
            {"position_divergence": {"curve": "higher_better"}})
        assert points["position_divergence"] == {"good_at": 1.0, "zero_at": 0.0}

    def test_skill_gradients_zero_end_is_the_span_floors_not_the_negatives(self) -> None:
        """'its band widened to 93.99-245.32 over 28 clean ladders, so twice its width
        reaches past good_at and the span floor, not the negatives' median of 14.61,
        placed it' -- at 0.0, its domain."""
        points, _ = anchor_curves(
            _corpus("skill_gradient", self._band(93.99, 245.32), [10.0, 14.61, 20.0]),
            {"skill_gradient": {"curve": "higher_better"}})
        assert points["skill_gradient"]["zero_at"] == 0.0


AGENTS = ("minimax:nodes=100", "minimax:nodes=200", "minimax:nodes=400")
# Built to differ in decisiveness and nothing else; no game in them ever stalls.
NOISED = ("decisive_rate", "ply_spread", "stall_rate")


def _records(seed: int, decisive: float) -> list[GameRecord]:
    """Two games per ordered pairing, synthetic, a share ``decisive`` of them won."""
    rng = random.Random(seed)
    out = []
    for first in AGENTS:
        for second in AGENTS:
            for _ in range(2):
                plies = rng.randrange(20, 80)
                out.append(GameRecord(
                    ruleset_hash="fixture", ruleset_name="fixture", seed=len(out),
                    agents=(first, second), moves=[rng.randrange(50) for _ in range(plies)],
                    winner=rng.choice([0, 1]) if rng.random() < decisive else None,
                    reason="fixture", plies=plies, branching=[20] * plies,
                    material=[0.0] * plies,
                ))
    return out


class TestComponentNoise:
    """The corpus spread of each component over its mean resampled sd."""

    @pytest.fixture(scope="class")
    @classmethod
    def measured(cls) -> tuple[list, dict]:
        spec = load_spec(RULESETS / "dull.json")
        corpus = [(spec, _records(n, share), None)
                  for n, share in enumerate((0.1, 0.4, 0.7, 0.9))]
        return corpus, component_noise(corpus, load_weights("v2"), NOISED, resamples=12)

    def test_it_is_the_spread_over_the_resampled_error(self, measured: tuple[list, dict]) -> None:
        corpus, noise = measured
        weights = load_weights("v2")
        for name in NOISED:
            raws = [compute_components(spec, records, rate(records)).raw[name]
                    for spec, records, _ in corpus]
            sds = []
            for spec, records, _ in corpus:
                drawn = bootstrap_index(spec, records, weights, resamples=12, seed=0,
                                        components_of_interest=NOISED)
                assert drawn is not None
                sds.append(drawn.component_sd[name])
            assert noise[name]["corpus_sd"] == round(statistics.pstdev(raws), 4)
            assert noise[name]["noise_sd"] == round(statistics.fmean(sds), 4)
            if statistics.fmean(sds) > 0:
                assert noise[name]["ratio"] == round(
                    statistics.pstdev(raws) / statistics.fmean(sds), 2)

    def test_it_tells_real_spread_from_resampling(self, measured: tuple[list, dict]) -> None:
        _, noise = measured
        # Four corpora built 0.1 to 0.9 decisive differ by more than resampling moves one;
        # their lengths are drawn alike, so that spread is mostly the resampling's own.
        assert noise["decisive_rate"]["ratio"] > 2.0
        assert noise["ply_spread"]["ratio"] < 1.0

    def test_a_component_nothing_moves_reads_infinite_rather_than_dividing_by_zero(
        self, measured: tuple[list, dict]
    ) -> None:
        _, noise = measured
        assert noise["stall_rate"] == {"corpus_sd": 0.0, "noise_sd": 0.0, "ratio": float("inf")}

    def test_under_four_games_it_says_nothing(self, measured: tuple[list, dict]) -> None:
        corpus, _ = measured
        assert component_noise(corpus[:3], load_weights("v2"), NOISED, resamples=4) == {}
