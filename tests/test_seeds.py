"""Several runs of one game, read as one measurement.

The failures this holds down are all of one shape -- a number published that nothing
measured. A placeholder 0.0 averaged in as though it were a score, a mean over the lucky
half of a corpus that could not be read, a total recomputed from mean components rather
than taken from the runs, a file that differs by a digit depending on which worker
finished first, and a file written before any of this existed read as though it were one
of these.

The arithmetic is checked against fixtures rather than against any game. `combine` never
sees a spec, and nothing here plays anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox import measurement
from chess_variant_sandbox.metrics import seeds
from chess_variant_sandbox.metrics.index import IndexResult
from chess_variant_sandbox.metrics.seeds import (
    MIN_TOTALS,
    VOTE_POOLED_SEAT,
    VOTE_TIE_ADMITS,
    VOTE_TIE_REFUSES,
    RunReading,
    combine,
    verdict,
)


def _interval(mean: float, sd: float = 0.02, spread: float = 0.04) -> dict[str, float]:
    return {
        "resamples": 1024,
        "mean": mean,
        "sd": sd,
        "ci_low": round(mean - spread, 4),
        "ci_high": round(mean + spread, 4),
        "deterministic_pairings": 0.125,
        "stratum_correction": 1.02,
    }


def _run(
    seed: int,
    total: float = 0.7,
    viable: bool = True,
    sd: float = 0.02,
    violations: tuple[str, ...] = (),
    rating: float = 100.0,
) -> RunReading:
    """A run that was scored, with one component and a two-rung ladder."""
    return RunReading(
        seed=seed,
        scored=True,
        total=total,
        sd=sd,
        viable=viable,
        violations=violations or (() if viable else (f"a gate {seed} failed",)),
        scores={"depth": total},
        raw={"depth": total * 10},
        contributions={"depth": total},
        families={"depth": total},
        interval=_interval(total, sd),
        standings=(
            {"name": "strong", "rating": rating, "games": 16},
            {"name": "weak", "rating": -rating, "games": 16},
        ),
        games=256,
    )


def _placeholder(seed: int) -> RunReading:
    """A run the roster could not read: components, no total, and a reason."""
    return RunReading(
        seed=seed,
        scored=False,
        total=None,
        sd=None,
        viable=False,
        violations=(f"this roster cannot read this ruleset at seed {seed}",),
        unreadable=("skill_gradient",),
        scores={"depth": 0.1},
        raw={"depth": 1.0},
        contributions={},
        families={},
        interval=None,
        standings=({"name": "strong", "rating": 0.0, "games": 16},),
        games=256,
    )


def _combined(runs, rule: str = VOTE_TIE_REFUSES):
    return combine("abc123", "a fixture", "v10", runs, rule)


class TestARunEitherHasATotalOrHasASentence:
    """The eight placeholder runs on disk, and the rule that keeps them out of a mean.

    `IndexResult` gives 0.0 a total in two refusals, and neither is a score. A mean that
    took one in would publish a number nothing measured, and the game it happened to
    would sink to the bottom of a leaderboard for a reason about the instrument.
    """

    def test_a_real_score_is_scored_and_both_refusals_are_not(self) -> None:
        scored = IndexResult(
            ruleset_hash="abc", ruleset_name="x", weights_version="v10", total=0.7,
            viable=True, violations=[], scores={"depth": 0.7},
            contributions={"depth": 0.7}, raw={}, families={"depth": 0.7},
        )
        assert scored.scored
        # Both placeholder constructors leave `contributions` empty, because there is no
        # combination to report. That emptiness is the whole signal.
        withheld = IndexResult(
            ruleset_hash="abc", ruleset_name="x", weights_version="v10", total=0.0,
            viable=False, violations=["unreadable"], scores={"depth": 0.1},
            contributions={}, raw={}, families={},
        )
        assert not withheld.scored

    def test_a_placeholder_is_never_averaged(self) -> None:
        runs = [_run(0, 0.8), _run(1000, 0.6), _run(2000, 0.7), _placeholder(3000)]
        found = _combined(runs)

        assert found.totals_used == 3
        assert found.index == pytest.approx(0.7)
        # And not the mean of four with a zero in it, which is the failure.
        assert found.index != pytest.approx((0.8 + 0.6 + 0.7 + 0.0) / 4)
        assert [entry["index"] for entry in found.runs] == [0.8, 0.6, 0.7, None]

    def test_a_run_whose_scorer_raised_carries_the_sentence_and_no_number(self) -> None:
        raised = RunReading.unscored(3000, "weights v10 expect components not present")
        found = _combined([_run(0), _run(1000), _run(2000), raised])

        assert found.totals_used == 3
        assert found.runs[3]["index"] is None
        assert "expect components not present" in found.runs[3]["violations"][0]
        # It votes to refuse, and the other three runs are untouched.
        assert found.verdict.refuse == 1
        assert found.index == pytest.approx(0.7)

    def test_a_run_with_no_total_votes_to_refuse(self) -> None:
        assert not verdict([_placeholder(0)], VOTE_TIE_REFUSES).viable
        assert not verdict([_placeholder(0)], VOTE_TIE_ADMITS).viable


class TestHowManyRunsAMeanNeeds:
    """`MIN_TOTALS` of four, for the reason a verdict needs three.

    A run is withheld when its ladder came out unordered, which is the same draw that
    would have scored low on depth. So the runs that survive lean high, and a mean of two
    is a mean of the lucky half.
    """

    def test_three_of_four_is_a_number_and_two_of_four_is_not(self) -> None:
        assert MIN_TOTALS == 3
        three = _combined([_run(0), _run(1000), _run(2000), _placeholder(3000)])
        two = _combined([_run(0), _run(1000), _placeholder(2000), _placeholder(3000)])

        assert three.index is not None and three.sd is not None
        assert two.index is None and two.sd is None
        assert two.ci_low is None and two.ci_high is None
        assert two.totals_used == 2

    def test_one_run_must_itself_be_scored(self) -> None:
        """A visitor's design is one run, and there is no majority to take."""
        assert seeds.minimum_totals(1) == 1
        assert _combined([_run(0, 0.55)]).index == pytest.approx(0.55)
        assert _combined([_placeholder(0)]).index is None

    def test_a_mean_of_none_has_no_interval_either(self) -> None:
        found = _combined([_placeholder(0)])
        assert (found.sd, found.ci_low, found.ci_high) == (None, None, None)
        assert found.detail["interval"] is None
        assert found.detail["total"] is None


class TestTheVerdictIsAVote:
    """One function, because a verdict combined in two places is two rules.

    Both rules are implemented although only one is in force, so the answer to the open
    question in `docs/decisions.md` is a one-line edit and a re-score rather than a stage.
    """

    @pytest.mark.parametrize(
        ("accepting", "refuses", "admits"),
        [(4, True, True), (3, True, True), (2, False, True), (1, False, False),
         (0, False, False)],
    )
    def test_four_runs_vote_and_the_tie_is_the_only_difference(
        self, accepting, refuses, admits
    ) -> None:
        runs = [
            _run(seed, viable=index < accepting)
            for index, seed in enumerate((0, 1000, 2000, 3000))
        ]
        assert verdict(runs, VOTE_TIE_REFUSES).viable is refuses
        assert verdict(runs, VOTE_TIE_ADMITS).viable is admits
        assert verdict(runs, VOTE_TIE_REFUSES).accept == accepting

    def test_at_one_run_every_rule_returns_that_run_s_own_verdict(self) -> None:
        for viable in (True, False):
            for rule in seeds.VOTE_RULES:
                assert verdict([_run(0, viable=viable)], rule).viable is viable
                assert verdict([_run(0, viable=viable)], rule).split == ""

    def test_agreement_has_nothing_to_say_about_a_split(self) -> None:
        agreed = [_run(seed) for seed in (0, 1000, 2000, 3000)]
        assert verdict(agreed, VOTE_TIE_REFUSES).split == ""
        refused = [_run(seed, viable=False) for seed in (0, 1000, 2000, 3000)]
        assert verdict(refused, VOTE_TIE_REFUSES).split == ""

    @pytest.mark.parametrize(
        ("accepting", "said"),
        [
            (3, "Three of its four runs accept it and one refuses it, so it stays in."),
            (2, "Two of its four runs accept it and two refuse it. A tie refuses, so it "
                "is ruled out."),
            (1, "One of its four runs accepts it and three refuse it, so it is ruled "
                "out."),
        ],
    )
    def test_the_split_reads_as_a_sentence(self, accepting, said) -> None:
        runs = [
            _run(seed, viable=index < accepting)
            for index, seed in enumerate((0, 1000, 2000, 3000))
        ]
        assert verdict(runs, VOTE_TIE_REFUSES).split == said

    def test_a_tie_that_admits_says_so(self) -> None:
        runs = [_run(seed, viable=index < 2)
                for index, seed in enumerate((0, 1000, 2000, 3000))]
        assert "A tie admits, so it stays in." in verdict(runs, VOTE_TIE_ADMITS).split

    def test_an_unknown_rule_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(ValueError):
            verdict([_run(0)], "whatever-feels-right")

    def test_the_refusing_runs_are_named_by_seed_when_there_are_several(self) -> None:
        runs = [_run(0), _run(1000), _run(2000, viable=False), _run(3000, viable=False)]
        found = _combined(runs)

        assert not found.viable
        assert found.violations[0] == found.verdict.split
        assert "run 2000: a gate 2000 failed" in found.violations
        assert "run 3000: a gate 3000 failed" in found.violations

    def test_one_run_s_violations_stand_unprefixed(self) -> None:
        """What a visitor's score has always said, and what the seed-0 files hold."""
        found = _combined([_run(0, viable=False)])
        assert found.violations == ("a gate 0 failed",)


class TestTheMeanIsTakenFromTheTotals:
    def test_the_total_is_never_recomputed_from_the_mean_scores(self) -> None:
        """Families combine geometrically, so the two differ in the third digit.

        Only one of them is what the runs measured. The fixture's component means differ
        from the mean of the totals on purpose.
        """
        runs = [_run(0, 0.9), _run(1000, 0.5), _run(2000, 0.8), _run(3000, 0.4)]
        found = _combined(runs)

        assert found.index == pytest.approx(0.65)
        assert found.detail["total"] == found.index
        # The component means are their own quantity, and the total is not read off them.
        found.detail["scores"]["depth"] = 0.0
        assert found.detail["total"] == pytest.approx(0.65)

    def test_the_interval_at_one_run_is_that_run_s_own_to_the_digit(self) -> None:
        """S1 moves no published interval, and this is the arithmetic that says so."""
        one = _run(0, 0.7783, sd=0.0184)
        found = _combined([one])

        assert one.interval is not None
        assert found.sd == one.interval["sd"]
        assert found.ci_low == one.interval["ci_low"]
        assert found.ci_high == one.interval["ci_high"]
        assert found.index == one.total

    def test_four_runs_halve_the_sd(self) -> None:
        """Four equal errors added in quadrature and divided by four is one half."""
        runs = [_run(seed, 0.7, sd=0.02) for seed in (0, 1000, 2000, 3000)]
        assert _combined(runs).sd == pytest.approx(0.01)

    def test_a_lopsided_interval_stays_lopsided(self) -> None:
        run = RunReading(
            seed=0, scored=True, total=0.7, sd=0.02, viable=True,
            contributions={"depth": 0.7},
            interval={"resamples": 8, "mean": 0.7, "sd": 0.02, "ci_low": 0.62,
                      "ci_high": 0.73, "deterministic_pairings": 0.0,
                      "stratum_correction": 1.0},
        )
        found = _combined([run])
        assert found.index is not None and found.ci_low is not None
        assert found.ci_high is not None
        assert 0.7 - found.ci_low > found.ci_high - 0.7

    def test_the_deterministic_share_is_the_mean_and_not_the_worst(self) -> None:
        """The maximum would refuse more intervals as seeds are added without any game
        changing, which is the defect pooling had."""
        runs = [_run(seed, 0.7) for seed in (0, 1000, 2000)]
        loud = RunReading(
            seed=3000, scored=True, total=0.7, sd=0.02, viable=True,
            contributions={"depth": 0.7},
            interval={**_interval(0.7), "deterministic_pairings": 0.925},
        )
        found = _combined([*runs, loud])
        share = found.detail["interval"]["deterministic_pairings"]
        assert share == pytest.approx((0.125 * 3 + 0.925) / 4, abs=5e-4)
        assert share < 0.925

    def test_the_ladder_is_the_per_agent_mean_with_its_games_summed(self) -> None:
        runs = [_run(0, rating=100.0), _run(1000, rating=300.0)]
        found = _combined(runs)

        assert found.standings == [
            {"name": "strong", "rating": pytest.approx(200.0), "games": 32},
            {"name": "weak", "rating": pytest.approx(-200.0), "games": 32},
        ]

    def test_a_withheld_run_still_rates_its_players(self) -> None:
        """It played its games. Only its score was withheld."""
        found = _combined([_run(0), _run(1000), _run(2000), _placeholder(3000)])
        games = {entry["name"]: entry["games"] for entry in found.standings}
        assert games["strong"] == 64

    def test_a_component_three_runs_read_is_not_withheld_by_the_fourth(self) -> None:
        found = _combined([_run(0), _run(1000), _run(2000), _placeholder(3000)])
        assert found.detail["unreadable"] == []


class TestTheFileIsAPropertyOfTheRuns:
    def test_the_same_bytes_whatever_order_the_runs_were_combined_in(self) -> None:
        """Floating-point addition is not associative, so a file that differed by a
        digit depending on which worker finished first would be a determinism bug wearing
        a rounding error."""
        runs = [_run(0, 0.9137), _run(1000, 0.5311), _run(2000, 0.8029),
                _placeholder(3000)]
        first = _combined(runs)
        shuffled = _combined([runs[2], runs[3], runs[0], runs[1]])

        assert json.dumps(first.detail, sort_keys=True) == json.dumps(
            shuffled.detail, sort_keys=True
        )
        assert first == shuffled
        assert first.seeds == (0, 1000, 2000, 3000)

    def test_a_measurement_needs_at_least_one_run(self) -> None:
        with pytest.raises(ValueError):
            _combined([])


class TestTheOldSingleRunForm:
    """Every `canonical.json` written before a game could have more than one run.

    Two rules rather than one, so that neither rests on an accident. A file at the old
    path is stale, and so is any file with no `runs` list wherever it sits -- a reader
    that trusted the path would trust a copy.
    """

    @staticmethod
    def _basis() -> measurement.Basis:
        return measurement.Basis(
            ruleset_hash="abc123", weights_version="v10", agents=("random",),
            games_per_pairing=4, seeds=(0,), engine="e", metrics="m", tables="t",
            analysis="a", verdict_rule="vote-tie-refuses",
        )

    def _write(self, path: Path, **extra) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "hash": "abc123", "index": 0.8, "sd": 0.02, "viable": True,
            "basis": self._basis().as_dict(), **extra,
        }), encoding="utf-8")
        return path

    def test_a_file_with_no_runs_list_never_yields_a_number(self, tmp_path) -> None:
        path = self._write(tmp_path / "abc123" / "canonical.json")
        found = measurement.load(path, self._basis())

        assert found is not None
        assert not found.current
        assert measurement.SINGLE_RUN_FORM in found.staleness.summary()
        assert found.staleness.needs == "rescore"

    def test_a_file_at_the_old_path_never_yields_a_number(self, tmp_path) -> None:
        path = self._write(
            tmp_path / "abc123-0" / "canonical.json",
            runs=[{"seed": 0, "index": 0.8, "sd": 0.02, "viable": True,
                   "violations": [], "unreadable": [], "origin": "played"}],
        )
        found = measurement.load(path, self._basis(), old_path=True)

        assert found is not None
        assert not found.current
        assert measurement.SINGLE_RUN_FORM in found.staleness.summary()

    def test_a_file_in_the_new_shape_at_the_new_path_is_current(self, tmp_path) -> None:
        """The control. Without it the two rules above could pass by refusing everything."""
        path = self._write(
            tmp_path / "abc123" / "canonical.json",
            runs=[{"seed": 0, "index": 0.8, "sd": 0.02, "viable": True,
                   "violations": [], "unreadable": [], "origin": "played"}],
        )
        found = measurement.load(path, self._basis())

        assert found is not None and found.current

    def test_a_stored_seed_reads_as_one_run_from_that_seed(self) -> None:
        """So the staleness names the true repair rather than a replay of everything."""
        stored = {**self._basis().as_dict(), "seed": 0}
        del stored["seeds"]
        rebuilt = measurement.Basis.from_dict(stored)
        assert rebuilt is not None and rebuilt.seeds == (0,)


class TestWhatTheSeedsMakeStale:
    @staticmethod
    def _basis(**overrides) -> measurement.Basis:
        base = {
            "ruleset_hash": "abc123", "weights_version": "v10", "agents": ("random",),
            "games_per_pairing": 4, "seeds": (0,), "engine": "e", "metrics": "m",
            "tables": "t", "analysis": "a", "verdict_rule": "vote-tie-refuses",
        }
        return measurement.Basis(**{**base, **overrides})

    def test_a_seed_the_reader_expects_is_replayed_alone(self) -> None:
        stale = self._basis().stale_against(
            self._basis(seeds=(0, 1000, 2000, 3000))
        )
        assert stale.missing_seeds == (1000, 2000, 3000)
        assert stale.needs == "replay"
        # The run already in the file owes nothing, and the line says so.
        assert stale.stored_needs == ""
        assert stale.summary() == (
            "stale (replay 3 of 4 runs): not yet measured on seeds 1000, 2000 and 3000"
        )

    def test_a_metrics_move_on_top_counts_both_halves(self) -> None:
        stale = self._basis(metrics="old").stale_against(
            self._basis(seeds=(0, 1000, 2000, 3000))
        )
        assert stale.cost() == "replay 3 of 4 runs, re-score 1"
        assert stale.needs == "replay"
        assert stale.stored_needs == "rescore"

    def test_more_seeds_than_expected_is_never_stale(self) -> None:
        """A retired game and a candidate tested before promotion keep what they have."""
        stale = self._basis(seeds=(0, 1000, 2000, 3000)).stale_against(self._basis())
        assert stale.current

    def test_a_seed_set_the_project_does_not_use_is_stale_by_its_seeds(self) -> None:
        current = self._basis(
            seeds=(0,), known_seed_sets=((0,), (0, 1000, 2000, 3000))
        )
        stale = self._basis(seeds=(0, 500)).stale_against(current)
        assert stale.needs == "replay"
        assert any("does not use" in reason for reason in stale.reasons)

    def test_the_verdict_rule_is_compared_only_where_there_was_a_vote(self) -> None:
        four = (0, 1000, 2000, 3000)
        voted = self._basis(seeds=four, verdict_rule="vote-tie-admits").stale_against(
            self._basis(seeds=four)
        )
        assert voted.needs == "rescore"
        assert any("vote-tie-admits" in reason for reason in voted.reasons)

        alone = self._basis(verdict_rule="vote-tie-admits").stale_against(self._basis())
        assert alone.current


_SEAT_RUNS = iter(range(10**9))


def _seat(wins: int, losses: int, draws: int = 0) -> dict:
    """One run's pair-band seat diagnostics, the shape `metrics/seat.py` writes.

    Every decided game a different game, and different from every other call's.
    """
    run = next(_SEAT_RUNS)
    return {
        "top_rung_seat": {
            "rungs": [1600, 6400],
            "games": wins + losses + draws,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "distinct": wins + losses,
            "transcripts": [f"{run}-{game}" for game in range(wins + losses)],
        }
    }


def _seat_refusal(seed: int) -> str:
    """A run's own seat sentence, as `metrics/seat.py` writes it into a run's file."""
    return (
        f"seat_decided_chance=1.00e-04 below minimum 0.001: the first seat went 15-0 "
        f"with 1 draws over the 16 games between the two strongest rungs at seed {seed}"
    )


class TestTheSeatIsDecidedOverTheRunsPooled:
    """The user's decision of 2026-09-19, *sweeps only*.

    A run's own seat reading is 16 games, on which the tolerated share cannot refuse even
    a clean sweep, so it cannot vote. The four runs' games are read once, together, and
    every other check still votes the way it did. `docs/reference/seat-effect-size.md`
    §3 and §13 are the design, and §16 the campaign on fresh seeds that tested it.
    """

    def _runs(self, splits, viable=True, violations=None):
        """One run per split, each carrying its own seat sentence when it refused."""
        return [
            RunReading(
                seed=seed * 1000,
                scored=True,
                total=0.7,
                sd=0.02,
                viable=viable and violations is None,
                violations=violations or (),
                scores={"depth": 0.7},
                raw={"depth": 7.0},
                contributions={"depth": 0.7},
                families={"depth": 0.7},
                interval=_interval(0.7),
                games=256,
                seat=_seat(*split),
            )
            for seed, split in enumerate(splits)
        ]

    def test_a_sweep_nobody_run_could_refuse_is_refused_on_the_four(self) -> None:
        """Sixteen decided games a run, 64 together. No run's own reading reaches the
        pair's alpha and the pooled one does, which is the whole point of pooling."""
        runs = self._runs([(16, 0, 0)] * 4)
        assert _combined(runs, VOTE_POOLED_SEAT).viable is False
        assert _combined(runs, VOTE_TIE_REFUSES).viable is True

    def test_a_lean_two_runs_refused_is_ranked_with_the_lean_printed(self) -> None:
        """`weathervane half-turn`: 10-51 pooled, five decided games in six to the second
        seat, refused today on a two-against-two vote. Under the restated check it is
        back in the rankings and the page says what it leans by."""
        runs = self._runs([(3, 12, 1), (2, 13, 1)], violations=(_seat_refusal(0),))
        runs += self._runs([(3, 13, 0), (2, 13, 1)])

        assert _combined(runs, VOTE_TIE_REFUSES).viable is False
        combined = _combined(runs, VOTE_POOLED_SEAT)
        assert combined.viable is True
        assert combined.violations == ()
        assert combined.seat["refuses"] is False
        assert "second seat took 51 of the 61 decided games" in combined.seat["lean"]

    def test_every_other_check_still_votes(self) -> None:
        """Only the seat leaves the vote. A game two of whose four runs failed some
        other gate is ruled out on the tie exactly as before, and the seat's own
        sentences are not quoted beside that."""
        runs = self._runs([(8, 7, 1), (8, 7, 1)], violations=("mean_plies=6.0 too low",))
        runs += self._runs([(8, 7, 1), (8, 7, 1)])
        combined = _combined(runs, VOTE_POOLED_SEAT)
        assert combined.viable is False
        assert combined.verdict.accept == 2
        assert any("mean_plies" in line for line in combined.violations)

    def test_a_run_refused_only_by_its_own_seat_votes_to_accept(self) -> None:
        """Otherwise the seat would be counted twice -- once per run and once pooled --
        and four runs could refuse a game the pooled reading accepts."""
        runs = self._runs([(15, 0, 1)] * 3, violations=(_seat_refusal(0),))
        runs += self._runs([(2, 13, 1)])
        combined = _combined(runs, VOTE_POOLED_SEAT)
        assert combined.verdict.accept == 4
        assert combined.verdict.split == ""

    def test_a_run_with_no_total_still_votes_to_refuse(self) -> None:
        """A check a run never reached is not a check it passed, which is what
        `unscored` has always meant. Its missing seat games are simply not pooled."""
        runs = self._runs([(8, 7, 1)] * 2) + [_placeholder(3000)]
        combined = _combined(runs, VOTE_POOLED_SEAT)
        assert combined.verdict.accept == 2
        assert combined.verdict.refuse == 1
        assert combined.seat["bands"]["top_rung_seat"]["runs"] == 2

    def test_one_run_keeps_the_built_form(self) -> None:
        """A visitor's design is one run of 16 pair games, on which the tolerated share
        refuses nothing at all. That measurement reads exactly as it did before: its own
        seat gate decides, and no pooled reading is published."""
        run = self._runs([(15, 0, 1)], violations=(_seat_refusal(0),))
        assert _combined(run, VOTE_POOLED_SEAT).viable is False
        assert _combined(run, VOTE_POOLED_SEAT).seat == {}
        assert _combined(run, VOTE_POOLED_SEAT).violations == _combined(
            run, VOTE_TIE_REFUSES
        ).violations

    def test_the_pooled_refusal_replaces_the_runs_own_seat_sentences(self) -> None:
        """One refusal, over 64 games, in place of four each describing 16."""
        runs = self._runs([(16, 0, 0)] * 4, violations=(_seat_refusal(0),))
        combined = _combined(runs, VOTE_POOLED_SEAT)
        assert len(combined.violations) == 1
        assert "over 4 runs" in combined.violations[0]
        assert combined.detail["viable"] is False
        assert combined.detail["violations"] == list(combined.violations)

    def test_an_unknown_rule_is_named_rather_than_guessed_at(self) -> None:
        with pytest.raises(ValueError, match="unknown verdict rule"):
            verdict(self._runs([(8, 7, 1)]), "vote-by-coin")
