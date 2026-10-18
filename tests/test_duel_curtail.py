"""Exact curtailment of a stamped duel (reference/compute-economy.md §4.2, Q632).

``settled`` must say a verdict exactly when every completion of the played pairs gives it,
which is checked here against brute force; and a curtailed duel must give the full duel's
verdict, keep exactly the full duel's prefix, and stop at the same pair at any worker count.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence

import pytest
from test_corpus import SPEC

from chess_variant_sandbox.agents.training.store import accepts
from chess_variant_sandbox.tournament.duel import (
    Verdict,
    curtailed_duel,
    duel,
    floor_verdict,
    settled,
)

QUARTERS = (0.0, 0.25, 0.5, 0.75, 1.0)
VERDICTS: dict[str, Verdict] = {
    "accept": accepts,
    "floor 0.20": floor_verdict(0.20),
    "floor 0.30": floor_verdict(0.30),
    "floor 0.375": floor_verdict(0.375),
    "floor 0.50": floor_verdict(0.50),
}


def _brute(prefix: Sequence[float], planned: int, verdict: Verdict) -> bool | None:
    calls = {
        verdict([*prefix, *rest])
        for rest in itertools.combinations_with_replacement(QUARTERS, planned - len(prefix))
    }
    return calls.pop() if len(calls) == 1 else None


@pytest.mark.parametrize("name", sorted(VERDICTS))
@pytest.mark.parametrize("planned", [1, 4, 6])
def test_settled_is_exactly_every_completion_agreeing(name: str, planned: int) -> None:
    verdict = VERDICTS[name]
    seen = set()
    for played in range(planned + 1):
        for prefix in itertools.combinations_with_replacement(QUARTERS, played):
            for order in (prefix, prefix[::-1]):
                found = settled(order, planned, verdict)
                assert found == _brute(order, planned, verdict), (name, order, planned)
                seen.add(found)
    # Each verdict family both decides early and stays open somewhere at these sizes.
    if planned > 1:
        assert seen == {True, False, None}


def test_a_full_duel_is_always_settled_and_a_graded_pair_is_refused() -> None:
    assert settled([1.0, 0.75], 2, accepts) is True
    assert settled([0.5, 0.5], 2, accepts) is False
    with pytest.raises(ValueError, match="quarters"):
        settled([0.6], 4, accepts)
    with pytest.raises(ValueError, match="played of a duel"):
        settled([1.0] * 3, 2, accepts)


def test_the_acceptance_bar_at_its_full_size_is_quick_to_read() -> None:
    """D2's 200 pairs: a clear duel decides well short of the end, in well under a second
    a look (§4.2 counts on the reading being cheap beside the games)."""
    from chess_variant_sandbox.agents.training.procedure import ACCEPT_PAIRS

    for pair, expect in ((0.0, False), (1.0, True)):
        played: list[float] = []
        started = time.perf_counter()
        while (call := settled(played, ACCEPT_PAIRS, accepts)) is None:
            played.append(pair)
        assert call is expect
        assert len(played) < ACCEPT_PAIRS
    assert time.perf_counter() - started < 5.0


# ------------------------------------------------------------------ real duels

# Two small fixtures on the five-square game: a mixed duel (its pairs are 1, 1/2 and 0) and
# the ceiling duel's shape, a player against itself at eight times the nodes (all lost).
DUELS = {
    "mixed": ("minimax:nodes=24", "random"),
    "ceiling": ("minimax:nodes=24", "minimax:nodes=192"),
}
PAIRS = 8
SEED = 40


@pytest.fixture(scope="module")
def full() -> dict[str, list[float]]:
    return {
        name: list(duel(SPEC, *players, PAIRS, seed=SEED, workers=1).pairs)
        for name, players in DUELS.items()
    }


@pytest.mark.parametrize("fixture", sorted(DUELS))
@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_a_curtailed_duel_gives_the_full_verdict_at_any_worker_count(
    fixture: str, name: str, full: dict[str, list[float]]
) -> None:
    verdict, pairs = VERDICTS[name], full[fixture]
    one, four = (
        curtailed_duel(SPEC, *DUELS[fixture], PAIRS, verdict, seed=SEED, workers=workers)
        for workers in (1, 4)
    )
    assert one.verdict == verdict(pairs)
    assert one == four, "threads changed where the duel stopped"
    k = one.decided_after
    assert one.planned == PAIRS and 1 <= k <= PAIRS
    assert list(one.found.pairs) == pairs[:k]
    assert len(one.records) == 2 * k == one.found.games
    assert [(r.seed, r.agents, r.moves) for r in one.records] == [
        (r.seed, r.agents, r.moves) for r in four.records
    ]
    assert {record.seed for record in one.records} == set(range(SEED, SEED + k))
    assert one.as_dict()["decided_after"] == k and one.as_dict()["planned_pairs"] == PAIRS
    # The stopping point is the first prefix `settled` answers, and no earlier one.
    assert all(settled(pairs[:j], PAIRS, verdict) is None for j in range(k))


def test_the_fixtures_both_stop_early_and_play_to_the_end(full: dict[str, list[float]]) -> None:
    """Otherwise the comparisons above would hold of a runner that never curtails, or of
    one that always stops at once."""
    assert len(set(full["mixed"])) == 3
    stops = {
        settled_at(pairs, verdict)
        for pairs in full.values()
        for verdict in VERDICTS.values()
    }
    assert min(stops) < PAIRS and max(stops) == PAIRS


def settled_at(pairs: Sequence[float], verdict: Verdict) -> int:
    return next(k for k in range(PAIRS + 1) if settled(pairs[:k], PAIRS, verdict) is not None)
