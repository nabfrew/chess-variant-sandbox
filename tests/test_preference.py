"""The pairwise judge, and the ordering it adds up to.

An absolute 0-1 score from a model was blended into the headline at a quarter weight,
which made it the ranker inside a band the objective index resolves to about 0.05 -- an
accident of scale compression rather than a decision. Asking which of two games is better
is a different and much better-behaved question, and these tests hold down the three
things that make its answer usable: it is asked in both orders, a flip is recorded as a
non-result, and the ordering is not allowed to decide anything until enough comparisons
support it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chess_variant_sandbox.llm.client import JUDGE_MODEL
from chess_variant_sandbox.llm.compare import compare_rulesets
from chess_variant_sandbox.loop.preference import MIN_COMPARISONS, Comparison, PreferenceLog, fit
from chess_variant_sandbox.spec import load_spec, spec_hash

PREFERS_FIRST = {"better": "first", "because": "more to decide"}
PREFERS_SECOND = {"better": "second", "because": "more to decide"}
NO_PREFERENCE = {"better": "neither", "because": "very close"}


class QueuedModel:
    """Returns queued answers and records what it was shown."""

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        # A stub stands for `Claude`, and `Claude` knows which model it is: whether a
        # prefix earns a cache breakpoint is a fact about the model, so `cached_system`
        # asks. This is the judge, and the judge's brief is far below Haiku's minimum --
        # which is the branch these comparisons therefore exercise.
        self.model = JUDGE_MODEL

    def structured(self, system, user, schema, max_tokens=16000) -> dict[str, Any]:
        self.prompts.append(user)
        return self.answers.pop(0) if self.answers else dict(NO_PREFERENCE)


def _plain(name: str) -> dict[str, Any]:
    """A minimal valid spec with no prose of its own to confuse a text assertion."""
    return {
        "schema_version": 1,
        "name": name,
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "squares": ["c1"]}, {"piece": "R", "squares": ["a1", "e1"]}]},
        "rules": {"win": "royal_capture", "max_plies": 120, "no_progress_draw": 40},
    }


@pytest.fixture(scope="module")
def pair():
    left = load_spec("rulesets/chess.json")
    right = load_spec("rulesets/gardner.json")
    return left, right


class TestComparing:
    def test_a_preference_that_survives_the_swap_is_a_preference(self, pair) -> None:
        left, right = pair
        # Prefers the game shown first, then the game shown second: both times, chess.
        model = QueuedModel([dict(PREFERS_FIRST), dict(PREFERS_SECOND)])

        comparison = compare_rulesets(left, right, [], [], client=model)

        assert len(model.prompts) == 2, "a comparison has to be asked both ways round"
        assert comparison.agreed
        assert comparison.winner == spec_hash(left)
        assert comparison.score_for_first == 1.0

    def test_a_preference_that_flips_is_a_tie(self, pair) -> None:
        """Position bias is well documented, and this is the whole defence against it.

        Answering "the first one" twice means the model preferred chess and then
        preferred gardner, which is not a weak preference -- it is an answer about
        position rather than about the games.
        """
        left, right = pair
        model = QueuedModel([dict(PREFERS_FIRST), dict(PREFERS_FIRST)])

        comparison = compare_rulesets(left, right, [], [], client=model)

        assert not comparison.agreed
        assert comparison.winner is None
        assert comparison.score_for_first == 0.5

    def test_neither_is_a_real_answer(self, pair) -> None:
        left, right = pair
        model = QueuedModel([dict(NO_PREFERENCE), dict(NO_PREFERENCE)])
        comparison = compare_rulesets(left, right, [], [], client=model)
        assert comparison.agreed
        assert comparison.winner is None

    def test_no_number_from_the_index_reaches_the_judge(self) -> None:
        """The same guarantee the proposer has, for the same reason.

        A judge shown the component scores restates them, and an ordering that restates
        the index and is then used to break the index's ties is the index breaking its
        own ties.

        Checked against specs whose own prose is plain, so that the assertion is about
        what this module puts in the prompt. A ruleset is free to describe itself using
        the word "score" -- chess's committed description does -- and that is the
        author's prose about the game, not a measurement leaking in.
        """
        from chess_variant_sandbox.metrics import load_weights

        model = QueuedModel([dict(NO_PREFERENCE), dict(NO_PREFERENCE)])
        compare_rulesets(_plain("one"), _plain("two"), [], [], client=model)

        weights = load_weights()
        assert model.prompts
        for prompt in model.prompts:
            lowered = prompt.lower()
            for component in weights.components():
                assert component not in lowered
                assert component.replace("_", " ") not in lowered
            for term in ("index", "weights", "viable", "score", "interval", "bootstrap"):
                assert term not in lowered


class TestOrdering:
    def _comparisons(self, wins: int, ties: int = 0) -> list[Comparison]:
        made = [
            Comparison(first="aaa", second="bbb", winner="aaa", agreed=True)
            for _ in range(wins)
        ]
        made += [
            Comparison(first="aaa", second="bbb", winner=None, agreed=False)
            for _ in range(ties)
        ]
        return made

    def test_a_consistent_winner_rates_above_a_loser(self) -> None:
        table = fit(self._comparisons(wins=6))
        assert table.rating("aaa") > table.rating("bbb")
        assert table.counts["aaa"] == 6

    def test_one_opinion_decides_nothing(self) -> None:
        """A ranker that is confidently arbitrary is worse than none: it looks like signal."""
        table = fit(self._comparisons(wins=1))
        assert not table.confident("aaa")
        assert not table.beats("aaa", "bbb")

    def test_a_contested_pair_does_not_separate(self) -> None:
        """Six comparisons split down the middle are six comparisons that found nothing."""
        made = self._comparisons(wins=3) + [
            Comparison(first="aaa", second="bbb", winner="bbb", agreed=True)
            for _ in range(3)
        ]
        table = fit(made)
        assert table.confident("aaa")
        assert not table.beats("aaa", "bbb")
        assert not table.beats("bbb", "aaa")

    def test_a_clear_pair_does_separate(self) -> None:
        table = fit(self._comparisons(wins=MIN_COMPARISONS + 5))
        assert table.beats("aaa", "bbb")
        assert not table.beats("bbb", "aaa")

    def test_the_flip_rate_is_reported(self) -> None:
        """The honest headline for a pairwise judge: at 0.5 it is answering by position."""
        table = fit(self._comparisons(wins=4, ties=4))
        assert table.flip_rate == pytest.approx(0.5)

    def test_an_unrated_candidate_sorts_last_not_first(self) -> None:
        table = fit(self._comparisons(wins=6))
        assert table.ranked(["bbb", "aaa", "ccc"]) == ["aaa", "bbb", "ccc"]


class TestLog:
    def test_comparisons_survive_a_restart(self, tmp_path) -> None:
        log = PreferenceLog(tmp_path)
        log.record(Comparison(first="aaa", second="bbb", winner="aaa", agreed=True))
        log.record(Comparison(first="aaa", second="ccc", winner=None, agreed=False))

        again = PreferenceLog(tmp_path)
        assert len(again.comparisons) == 2
        assert again.seen("bbb", "aaa"), "a pair is a pair whichever way round it is named"
        assert not again.seen("bbb", "ccc")

    def test_a_damaged_line_does_not_lose_the_rest(self, tmp_path) -> None:
        log = PreferenceLog(tmp_path)
        log.record(Comparison(first="aaa", second="bbb", winner="aaa", agreed=True))
        with open(log.path, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        log.record(Comparison(first="aaa", second="ccc", winner="ccc", agreed=True))

        assert len(PreferenceLog(tmp_path).comparisons) == 2

    def test_the_table_reads_back_as_json(self, tmp_path) -> None:
        log = PreferenceLog(tmp_path)
        for _ in range(4):
            log.record(Comparison(first="aaa", second="bbb", winner="aaa", agreed=True))
        payload = json.dumps(log.table().as_dict())
        assert "ratings" in payload and "flip_rate" in payload
