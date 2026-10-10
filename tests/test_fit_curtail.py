"""Exact curtailment of the fit's acceptance match (plan-compute §4.2, Q800).

The match's 48 games are unpaired, so ``duel.settled`` reads them one game at a time. Held
here: that reading is exactly "every completion of the played games agrees" (brute force
over game completions), a curtailed match gives the full match's verdict and keeps exactly
its prefix at 1 and 4 workers, and ``fit_ruleset(curtail=True)`` stores the full fit's
verdict with the games it was decided after.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_corpus import SPEC

from chess_variant_sandbox.tournament import fitting
from chess_variant_sandbox.tournament.duel import settled
from chess_variant_sandbox.tournament.fitting import (
    acceptance,
    beats,
    fit_ruleset,
    reading_order,
)

RESULTS = (0.0, 0.5, 1.0)
GARDNER = str(Path(__file__).resolve().parent.parent / "rulesets" / "gardner.json")


def _brute(prefix: Sequence[float], planned: int) -> bool | None:
    calls = {
        beats([*prefix, *rest])
        for rest in itertools.combinations_with_replacement(RESULTS, planned - len(prefix))
    }
    return calls.pop() if len(calls) == 1 else None


@pytest.mark.parametrize("planned", [1, 2, 5, 8])
def test_settled_over_games_is_exactly_every_game_completion_agreeing(planned: int) -> None:
    seen = set()
    for played in range(planned + 1):
        for prefix in itertools.product(RESULTS, repeat=played):
            found = settled(prefix, planned, beats)
            assert found == _brute(prefix, planned), (prefix, planned)
            seen.add(found)
    if planned > 1:
        assert seen == {True, False, None}


def test_a_curtailed_score_never_sits_across_the_bar_from_its_verdict() -> None:
    """What `Acceptance` promises a reader of a curtailed store: the score over a decided
    prefix is on the verdict's side of the bar. Its argument does not depend on the size."""
    planned = 12
    for points in range(0, 2 * planned + 1):
        for played in range(planned + 1):
            prefix = [1.0] * (points // 2) + [0.5] * (points % 2)
            if len(prefix) > played:
                continue
            prefix += [0.0] * (played - len(prefix))
            call = settled(prefix, planned, beats)
            if call is not None:
                assert fitting._match_score(prefix) > 0.5 if call else (
                    fitting._match_score(prefix) <= 0.5
                ), (points, played)


def test_the_reading_order_alternates_the_colours() -> None:
    tasks = [(seed, "a", "b") for seed in range(3)] + [(seed, "b", "a") for seed in range(3, 6)]
    assert [seed for seed, _, _ in reading_order(tasks)] == [0, 3, 1, 4, 2, 5]
    with pytest.raises(ValueError, match="both colours"):
        reading_order(tasks[:5])


# ------------------------------------------------------------------ real matches

# A mixed match (wins, draws and losses) and a lopsided one, on the five-square game.
MATCHES = {
    "mixed": ("minimax:nodes=24", "random"),
    "ceiling": ("minimax:nodes=24", "minimax:nodes=192"),
}
GAMES = 6
SEED = 70


@pytest.fixture(scope="module")
def full() -> dict[str, fitting.Acceptance]:
    return {name: acceptance(SPEC, *players, GAMES, SEED, 1) for name, players in MATCHES.items()}


@pytest.mark.parametrize("name", sorted(MATCHES))
def test_a_curtailed_match_gives_the_full_verdict_at_any_worker_count(
    name: str, full: dict[str, fitting.Acceptance]
) -> None:
    whole = full[name]
    one, four = (
        acceptance(SPEC, *MATCHES[name], GAMES, SEED, workers, curtail=True)
        for workers in (1, 4)
    )
    assert whole.decided_after == whole.planned == 2 * GAMES
    assert one.accepted == whole.accepted
    assert one == four, "threads changed where the match stopped"
    k = one.decided_after
    assert one.planned == 2 * GAMES and 1 <= k <= one.planned
    assert [(r.seed, r.agents, r.moves) for r in one.records] == [
        (r.seed, r.agents, r.moves) for r in four.records
    ]

    # The kept games are the full schedule's, read in its order, and no more.
    every = fitting.run_tournament(
        SPEC,
        fitting.TournamentConfig(
            agents=MATCHES[name], games_per_pairing=GAMES, seed=SEED, workers=1,
            include_self_play=False, collect_series=False,
        ),
    ).records
    by_task = {(r.seed, *r.agents): r for r in every}
    order = reading_order([(r.seed, *r.agents) for r in every])
    expected = [by_task[task] for task in order[:k]]
    assert [(r.seed, r.agents, r.moves) for r in one.records] == [
        (r.seed, r.agents, r.moves) for r in expected
    ]
    scores = [r.score_for(0 if r.agents[0] == MATCHES[name][0] else 1) for r in expected]
    assert one.score == fitting._match_score(scores)
    # The stopping point is the first prefix `settled` answers, and no earlier one.
    assert all(settled(scores[:j], 2 * GAMES, beats) is None for j in range(k))


def test_the_fixtures_stop_early(full: dict[str, fitting.Acceptance]) -> None:
    """Otherwise the comparison above would hold of a match that never curtails."""
    stops = {
        acceptance(SPEC, *players, GAMES, SEED, 1, curtail=True).decided_after
        for players in MATCHES.values()
    }
    assert min(stops) < 2 * GAMES


def test_a_batch_that_comes_back_as_other_games_is_refused() -> None:
    other = fitting.run_tournament(
        SPEC,
        fitting.TournamentConfig(
            agents=("random", "greedy"), games_per_pairing=2, seed=900, workers=1,
            include_self_play=False, collect_series=False,
        ),
    ).records

    def wrong(block: list[tuple[int, str, str]]) -> list:
        return list(other[: len(block)])

    with pytest.raises(ValueError, match="other games"):
        acceptance(SPEC, "random", "greedy", 2, 0, 2, curtail=True, play=wrong)


def test_fit_ruleset_curtailed_stores_the_full_verdict(monkeypatch) -> None:
    """The fit end to end, its acceptance match shrunk to the fixture's size."""
    monkeypatch.setattr(fitting, "ACCEPT_NODES", 24)
    monkeypatch.setattr(fitting, "ACCEPT_GAMES_PER_PAIRING", GAMES)
    whole = fit_ruleset(SPEC, workers=1)
    assert whole.accept_games == 2 * GAMES
    for workers in (1, 4):
        cut = fit_ruleset(SPEC, workers=workers, curtail=True)
        assert cut.accepted == whole.accepted
        assert cut.vector == whole.vector and cut.samples == whole.samples
        assert 1 <= cut.accept_games <= 2 * GAMES
        assert fitting.differences(whole, cut) == []


# ------------------------------------------------------------------ `train --curtail`


@pytest.mark.parametrize(("flags", "curtail"), [([], False), (["--curtail"], True)])
def test_train_curtails_net_acceptance_only_when_asked(
    flags: list[str], curtail: bool, monkeypatch, tmp_path
) -> None:
    """Off by default: D2's score and error are measurements the teacher plan's gates read."""
    pytest.importorskip("numpy")
    from chess_variant_sandbox.agents.training import store
    from chess_variant_sandbox.cli import main

    seen: dict[str, object] = {}

    class Stop(Exception):
        pass

    def train_ruleset(*_args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(store, "train_ruleset", train_ruleset)
    with pytest.raises(Stop):
        main(["train", GARDNER, "--store", str(tmp_path), *flags])
    assert seen["curtail"] is curtail


def test_train_refuses_curtail_without_a_duel(capsys) -> None:
    from chess_variant_sandbox.cli import main

    assert main(["train", GARDNER, "--curtail", "--no-duel"]) == 2
    assert "--curtail" in capsys.readouterr().err
