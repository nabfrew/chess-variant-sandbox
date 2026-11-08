"""`metrics/repeats.py`: the share of different games beside a score (Q953a).

`docs/reference/repeated-games.md` §1 fixed the reading before any library record was read
and §6 made it part of every score: a game's identity is its ordered players and its whole
move list, shares are pooled over the runs, the distinct re-score keeps each identity once
within its run, and a score rests on repeated games at two published sds. Inline records,
no engine, no scorer: the scorer is stood in for where the reading calls it.
"""

from __future__ import annotations

import pytest

from chess_variant_sandbox.metrics import repeats
from chess_variant_sandbox.tournament.analysis import BlunderTrial, CorpusAnalysis, GameAnalysis
from chess_variant_sandbox.tournament.record import GameRecord

ROSTER = ("random", "greedy", "minimax:nodes=400", "minimax:nodes=1600", "minimax:nodes=6400")
TOP = ("minimax:nodes=1600", "minimax:nodes=6400")


def _game(agents: tuple[str, str], moves: list[int], seed: int = 0) -> GameRecord:
    return GameRecord(
        ruleset_hash="h", ruleset_name="repeats", seed=seed, agents=agents, moves=moves,
        winner=0, reason="checkmate", plies=len(moves),
        branching=[20] * len(moves), material=[0.0] * len(moves), captures=[0] * len(moves),
    )


class TestAGamesIdentity:
    def test_the_seed_is_not_part_of_it_and_the_players_order_is(self) -> None:
        """Two seeds whose players play the same moves are one game played twice; the same
        moves with the seats swapped are two games."""
        a = _game(TOP, [1, 2, 3], seed=0)
        b = _game(TOP, [1, 2, 3], seed=1000)
        swapped = _game((TOP[1], TOP[0]), [1, 2, 3], seed=0)
        assert repeats.identity(a) == repeats.identity(b)
        assert repeats.identity(a) != repeats.identity(swapped)
        assert repeats.distinct_share([a, b]) == 0.5
        assert repeats.distinct_share([a, b, swapped]) == pytest.approx(0.6667)
        assert repeats.distinct_share([]) is None

    def test_the_three_bands(self) -> None:
        """All games; both seats searching; both seats one of the two largest budgets."""
        games = [
            _game(("random", "greedy"), [1]),
            _game(("random", "greedy"), [1]),
            _game(("minimax:nodes=400", "minimax:nodes=1600"), [2]),
            _game(TOP, [3]),
            _game(TOP, [3]),
            _game((TOP[1], TOP[1]), [4]),
        ]
        found = repeats.shares(games, ROSTER)
        assert found == {"all": round(4 / 6, 4), "searching": 0.75, "top_two": round(2 / 3, 4)}


class TestTheDistinctRecords:
    def test_each_identity_once_the_first_in_order_with_its_analyses(self) -> None:
        records = [
            _game(TOP, [1, 2], seed=0),
            _game(TOP, [1, 2], seed=1),
            _game(TOP, [1, 3], seed=2),
        ]
        analysis = CorpusAnalysis(
            games=[
                GameAnalysis(seed=seed, agents=TOP) for seed in (0, 1, 2)
            ],
            blunders=[
                BlunderTrial(seed=1, agents=TOP, ply=0, original_winner=0,
                             blundering_side=0, new_winner=1),
                BlunderTrial(seed=2, agents=TOP, ply=0, original_winner=0,
                             blundering_side=0, new_winner=1),
            ],
        )
        kept, kept_analysis = repeats.distinct_records(records, analysis)
        assert [record.seed for record in kept] == [0, 2]
        assert kept_analysis is not None
        assert [game.seed for game in kept_analysis.games] == [0, 2]
        assert [trial.seed for trial in kept_analysis.blunders] == [2]


class TestTheRule:
    @pytest.mark.parametrize(
        ("distinct", "index", "sd", "rests"),
        [
            (0.6888, 0.6534, 0.0089, True),   # gardner, §4.1: 4.0 sd
            (0.6300, 0.6028, 0.0107, True),   # gardner with spies resting: 2.5 sd
            (0.6200, 0.6000, 0.0100, True),   # exactly two sds is at the line, and in
            (0.6199, 0.6000, 0.0100, False),
            (0.5801, 0.6000, 0.0100, False),  # either direction
            (0.5800, 0.6000, 0.0100, True),
            (None, 0.6, 0.01, False),
            (0.7, None, 0.01, False),
            (0.7, 0.6, None, False),
        ],
    )
    def test_two_published_sds(self, distinct, index, sd, rests) -> None:
        assert repeats.REST_SD == 2.0
        assert repeats.rests(distinct, index, sd) is rests


class TestTheReading:
    def test_the_distinct_mean_is_over_the_runs_the_index_averages(self, monkeypatch) -> None:
        """A run the index does not average is not re-scored, and a run whose distinct
        re-score produced no combination is no total: neither is ever averaged as 0.0,
        which is the defect Q953's reading script had on sweeping bishop chess."""
        totals = {0: 0.70, 1000: None, 2000: 0.60, 3000: 0.62}
        asked: list[int] = []

        def distinct_total(spec, records, analysis, weights):
            asked.append(records[0].seed)
            return totals[records[0].seed], len(records)

        monkeypatch.setattr(repeats, "distinct_total", distinct_total)
        runs = [
            repeats.Run(seed=seed, records=[_game(TOP, [seed], seed=seed)], analysis=None,
                        counted=seed != 3000)
            for seed in (3000, 0, 1000, 2000)
        ]
        block = repeats.reading({}, runs, ROSTER, "v12", index=0.60, sd=0.01, minimum=2)
        assert asked == [0, 1000, 2000]
        assert [run["seed"] for run in block["runs"]] == [0, 1000, 2000, 3000]
        assert block["distinct_index"] == 0.65
        assert block["rests"] is True
        assert block["games"] == 4 and block["all"] == 1.0

        short = repeats.reading({}, runs, ROSTER, "v12", index=0.60, sd=0.01, minimum=3)
        assert short["distinct_index"] is None and short["rests"] is False


class TestTheBandEdges:
    def test_a_one_node_rung_searches_and_three_budgets_have_a_top_two(self) -> None:
        """A budget of one node is a searching rung; a game whose two seats are exactly the
        searching set is in the band; with three budgets the top two are the upper two."""
        roster = ("greedy", "minimax:nodes=1", "minimax:nodes=400")
        games = [
            _game(("minimax:nodes=1", "minimax:nodes=400"), [1]),
            _game(("minimax:nodes=1", "minimax:nodes=400"), [1]),
            _game(("minimax:nodes=400", "minimax:nodes=400"), [2]),
            _game(("greedy", "minimax:nodes=400"), [3]),
        ]
        assert repeats.shares(games, roster) == {
            "all": 0.75, "searching": 0.6667, "top_two": 0.6667,
        }


class _Scored:
    def __init__(self, scored: bool, headline: float) -> None:
        self.scored = scored
        self.headline = headline


class TestTheDistinctTotal:
    """The scorer and the ratings stood in for: what the re-score is asked and returns."""

    def _stand_in(self, monkeypatch, outcome) -> list[dict]:
        from chess_variant_sandbox.metrics import index
        from chess_variant_sandbox.tournament import ratings

        asked: list[dict] = []

        def score_ruleset(spec, records, rated, **kwargs):
            asked.append({"games": [r.seed for r in records], **kwargs})
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(index, "score_ruleset", score_ruleset)
        monkeypatch.setattr(ratings, "rate", lambda records: None)
        return asked

    def _records(self) -> list[GameRecord]:
        return [_game(TOP, [1], seed=0), _game(TOP, [1], seed=1), _game(TOP, [2], seed=2)]

    def test_each_game_once_no_interval_four_places(self, monkeypatch) -> None:
        asked = self._stand_in(monkeypatch, _Scored(True, 0.123456))
        assert repeats.distinct_total({}, self._records(), None, "v12") == (0.1235, 2)
        assert asked == [
            {"games": [0, 2], "weights": "v12", "analysis": None, "interval": False},
        ]

    def test_no_combination_is_no_total(self, monkeypatch) -> None:
        self._stand_in(monkeypatch, _Scored(False, 0.5))
        assert repeats.distinct_total({}, self._records(), None, "v12") == (None, 2)

    def test_a_refusal_is_no_total_and_never_fails_the_measurement(self, monkeypatch) -> None:
        self._stand_in(monkeypatch, ValueError("refused"))
        assert repeats.distinct_total({}, self._records(), None, "v12") == (None, 2)

    def test_no_games_is_no_total_and_nothing_kept(self, monkeypatch) -> None:
        asked = self._stand_in(monkeypatch, _Scored(True, 0.5))
        assert repeats.distinct_total({}, [], None, "v12") == (None, 0)
        assert asked == []


class TestTheValues:
    def test_a_run_is_immutable(self) -> None:
        """The reading is pure in its runs; a run is a value, not a handle to edit."""
        import dataclasses

        run = repeats.Run(seed=0, records=[], analysis=None, counted=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.counted = False  # type: ignore[misc]

    def test_the_distinct_mean_is_published_to_four_places(self, monkeypatch) -> None:
        totals = {0: 0.1234, 1: 0.1235, 2: 0.1237}
        monkeypatch.setattr(
            repeats, "distinct_total",
            lambda spec, records, analysis, weights: (totals[records[0].seed], 1),
        )
        runs = [
            repeats.Run(seed=s, records=[_game(TOP, [s], seed=s)], analysis=None, counted=True)
            for s in totals
        ]
        block = repeats.reading({}, runs, ROSTER, "v12", index=None, sd=None, minimum=1)
        assert block["distinct_index"] == 0.1235

    @pytest.mark.parametrize(
        ("distinct", "index", "sd", "rests"),
        [
            (0.61996, 0.6, 0.01, True),     # the difference rounds up to the line
            (0.6208, 0.6, 0.0104, True),    # the line is 0.0208, not 0.021
            (0.62, 0.6, 0.010004, True),    # the line is 0.0200, not 0.02001
        ],
    )
    def test_both_sides_are_read_at_four_places(self, distinct, index, sd, rests) -> None:
        assert repeats.rests(distinct, index, sd) is rests


class TestTheSummary:
    def test_the_line_and_its_rests_clause(self) -> None:
        block = {
            "all": 0.5, "searching": None, "top_two": 0.25, "distinct_index": 0.65344,
            "rests": True,
        }
        assert repeats.summary(block) == (
            "  different games: all 50%, top two 25%; counted once, 0.653"
            " (rests on repeated games)"
        )
        assert repeats.summary({**block, "rests": False}) == (
            "  different games: all 50%, top two 25%; counted once, 0.653"
        )
        assert repeats.summary(None) == ""
