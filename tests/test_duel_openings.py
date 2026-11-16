"""R-P1: one public duel path, opened by default; an unopened duel only by name.

What this file holds (``tournament/duel.py``'s docstring, ``tournament/openings.py``,
``.claude/rules/agents-and-eval.md`` *tournament/duel.py*):

- the default duel opens every seed-pair with random plies drawn from its seed, both games
  of a pair from the same opening with the seats swapped, and reports its distinct openings
  and distinct games;
- ``unopened=True`` is the old duel, game for game (the ordinary paired runner);
- an unopened duel cannot happen by omission: ``opening_plies=0`` is refused, and so is
  naming plies beside ``unopened``;
- the answer does not depend on the worker count, and a block from a later seed is the tail
  of the longer duel, opened or not;
- the openings moved out of ``agents/training/data.py`` are the same functions under both
  names, so no stored dataset or acceptance duel moved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_corpus import SPEC

from chess_variant_sandbox.tournament import openings
from chess_variant_sandbox.tournament.checkpoint import run_tournament
from chess_variant_sandbox.tournament.duel import (
    DUEL_OPENING_PLIES,
    curtailed_duel,
    duel,
    duel_records,
    floor_verdict,
)
from chess_variant_sandbox.tournament.runner import TournamentConfig

PLAYERS = ("minimax:nodes=12", "minimax:nodes=24")
PAIRS = 6
SEED = 40


def _key(records: list) -> list:  # type: ignore[type-arg]
    return [(r.seed, r.agents, r.moves) for r in records]


def test_the_default_duel_is_opened_and_both_games_of_a_pair_share_their_opening() -> None:
    records = duel_records(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=1)
    assert len(records) == 2 * PAIRS
    by_seed: dict[int, list] = {}  # type: ignore[type-arg]
    for record in records:
        by_seed.setdefault(record.seed, []).append(record)
    assert sorted(by_seed) == list(range(SEED, SEED + PAIRS))
    for seed, games in by_seed.items():
        assert {g.agents for g in games} == {PLAYERS, PLAYERS[::-1]}
        k = openings.opened_plies(seed, DUEL_OPENING_PLIES)
        assert games[0].moves[:k] == games[1].moves[:k], "a pair's games opened apart"
    assert any(openings.opening_plies_for(s, DUEL_OPENING_PLIES) for s in by_seed)

    found = duel(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=1)
    assert found.opening_plies == DUEL_OPENING_PLIES == 8
    assert found.openings == openings.count_openings(records, DUEL_OPENING_PLIES)
    assert found.distinct_games == openings.count_games(records)
    report = found.as_dict()
    assert {"opening_plies", "openings", "distinct_games", "identical", "games"} <= set(report)
    assert report["opening_plies"] == 8 and report["games"] == 2 * PAIRS


def test_unopened_by_name_is_the_old_duel_game_for_game() -> None:
    """The duel before R-P1 was the ordinary paired runner from the start position."""
    old = run_tournament(
        SPEC,
        TournamentConfig(agents=PLAYERS, games_per_pairing=PAIRS, seed=SEED, workers=1,
                         include_self_play=False, collect_series=False, paired=True),
    ).records
    unopened = duel_records(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=1, unopened=True)
    assert _key(unopened) == _key(old)
    found = duel(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=1, unopened=True)
    assert (found.opening_plies, found.openings) == (0, 1)
    assert found.distinct_games == openings.count_games(old)


def test_an_unopened_duel_is_never_asked_for_by_omission() -> None:
    for call in (duel, duel_records):
        with pytest.raises(ValueError, match="unopened=True"):
            call(SPEC, *PLAYERS, 2, opening_plies=0)
        with pytest.raises(ValueError, match="asks for both"):
            call(SPEC, *PLAYERS, 2, opening_plies=4, unopened=True)
    with pytest.raises(ValueError, match="unopened=True"):
        curtailed_duel(SPEC, *PLAYERS, 2, floor_verdict(0.5), opening_plies=-1)
    with pytest.raises(ValueError, match="material series"):
        duel(SPEC, *PLAYERS, 2, margin_weight=0.1)


@pytest.mark.parametrize("unopened", [False, True])
def test_threads_and_blocks_never_change_a_duel(unopened: bool) -> None:
    one = duel_records(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=1, unopened=unopened)
    three = duel_records(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=3, unopened=unopened)
    assert _key(one) == _key(three)
    tail = duel_records(SPEC, *PLAYERS, PAIRS - 2, seed=SEED + 2, workers=2, unopened=unopened)
    assert sorted(_key(tail)) == sorted(k for k in _key(one) if k[0] >= SEED + 2)


def test_a_curtailed_duel_is_opened_as_the_full_duel_is() -> None:
    full = duel(SPEC, *PLAYERS, PAIRS, seed=SEED, workers=1)
    cut = curtailed_duel(SPEC, *PLAYERS, PAIRS, floor_verdict(0.5), seed=SEED, workers=2)
    k = cut.decided_after
    assert list(cut.found.pairs) == list(full.pairs[:k])
    assert cut.found.opening_plies == DUEL_OPENING_PLIES


def test_the_opening_counts_read_rounds_and_the_start_position() -> None:
    games = [SimpleNamespace(seed=s, moves=[1, 2, 3, 4, 5, 6, 7, 8, 9]) for s in range(20)]
    assert openings.count_openings(games, 0) == 1
    assert openings.count_games(games) == 1
    for seed in range(20):
        plies = openings.opening_plies_for(seed, 8)
        assert openings.opened_plies(seed, 8) == plies
        assert openings.opened_plies(seed, 8, rounds=True) == 2 * (plies // 2)
    # Identical move lists: one game; distinct openings are the distinct prefix lengths.
    assert openings.count_openings(games, 8) == len(
        {openings.opened_plies(s, 8) for s in range(20)}
    )


def test_the_moved_openings_are_the_same_functions_under_the_old_names() -> None:
    from chess_variant_sandbox.agents.training import data

    assert data.opening_plies_for is openings.opening_plies_for
    assert data.opening_seed is openings.opening_seed
    assert data.diverse_game is openings.diverse_game
    assert data._play_tasks is openings.play_opened


def test_a_seat_without_a_native_form_is_finished_by_the_mixed_loop() -> None:
    assert openings.native_seats(PLAYERS, "")
    assert not openings.native_seats(("minimax:nodes=12", "fairystockfish:nodes=100"), "")
