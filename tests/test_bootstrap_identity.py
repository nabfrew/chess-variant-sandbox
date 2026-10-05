"""A bootstrap resample draws games with replacement, and a copy is not a repeat.

Two components count distinct games: ``distinct_game_rate`` and ``opening_diversity``.
Under the identity rule they read each game identity once per resample, so duplication
cannot manufacture repeats. Every other component averages or sums over games, and the
rule must not touch it.

Three more read a distance to the nearest *other* game (`docs/archive/plan-variety.md` §3):
``prefix_divergence``, ``position_divergence`` and ``event_divergence``. They average
over the draws like any other component, but their neighbours are identities, so a copy
is never a neighbour at distance zero. The same shape holds them.

The fixtures are synthetic records on ``rulesets/dull.json`` with random move lists, so
the component reads only what the records say.
"""

from __future__ import annotations

import contextlib
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chess_variant_sandbox.metrics import components
from chess_variant_sandbox.metrics.bootstrap import BootstrapResult, bootstrap_index
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, GameAnalysis
from chess_variant_sandbox.tournament.ratings import rate
from chess_variant_sandbox.tournament.record import GameRecord

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
AGENTS = ("minimax:nodes=100", "minimax:nodes=200", "minimax:nodes=400", "minimax:nodes=800")
COUNTS = ("distinct_game_rate", "opening_diversity")
AVERAGES = ("decisive_rate", "stall_rate", "ply_spread", "first_player_advantage")
RESAMPLES = 60


def _corpus(replayed: bool) -> list[GameRecord]:
    """Four games per ordered pairing, every move list random.

    With ``replayed`` each pairing's second game replays its first move for move under
    its own seed, so the corpus holds genuine repeats.
    """
    rng = random.Random(7)
    records: list[GameRecord] = []
    seed = 0
    for first in AGENTS:
        for second in AGENTS:
            for game in range(4):
                moves = [rng.randrange(50) for _ in range(30)]
                if replayed and game == 1:
                    moves = list(records[-1].moves)
                records.append(
                    GameRecord(
                        ruleset_hash="fixture", ruleset_name="fixture", seed=seed,
                        agents=(first, second), moves=moves,
                        winner=rng.choice([0, 1, None]), reason="fixture", plies=30,
                        branching=[20] * 30, material=[0.0] * 30,
                    )
                )
                seed += 1
    return records


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec(RULESETS / "dull.json")


def _bootstrap(spec: dict, records: list[GameRecord]) -> BootstrapResult:
    result = bootstrap_index(
        spec, records, load_weights("v2"), resamples=RESAMPLES, seed=0,
        components_of_interest=COUNTS + AVERAGES,
    )
    assert result is not None
    return result


def _point(spec: dict, records: list[GameRecord]) -> dict[str, float]:
    return components.compute_components(spec, records, rate(records)).raw


def _half_samples(records: list[GameRecord], draws: int = 200) -> list[float]:
    """``distinct_game_rate`` on two of each pairing's four games, without replacement."""
    strata: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for record in records:
        strata[record.agents].append(record)
    rng = random.Random(0)
    return [
        components._distinct_game_rate(
            [g for games in strata.values() for g in rng.sample(games, len(games) // 2)]
        )
        for _ in range(draws)
    ]


Drawn = Callable[..., BootstrapResult]


@contextlib.contextmanager
def _without_the_rule() -> Iterator[None]:
    """The same draws with every copy read as a game of its own."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(components, "_one_per_game", lambda games: list(games))
        yield


@pytest.fixture(scope="module")
def drawn(spec: dict) -> Drawn:
    """Each bootstrap the tests below read, computed once for the module (Q377).

    Four of them -- with or without genuine repeats, with or without the identity rule
    -- where the tests used to compute seven. The worker-count test still computes its
    own, since dividing the work is what it is about.
    """
    kept: dict[tuple[bool, bool], BootstrapResult] = {}

    def get(replayed: bool, ruled: bool = True) -> BootstrapResult:
        if (replayed, ruled) not in kept:
            if ruled:
                kept[replayed, ruled] = _bootstrap(spec, _corpus(replayed))
            else:
                with _without_the_rule():
                    kept[replayed, ruled] = _bootstrap(spec, _corpus(replayed))
        return kept[replayed, ruled]

    return get


def test_a_stored_corpus_reads_the_same_under_the_rule(spec: dict) -> None:
    """Every identity in a played corpus is unique, so the point value cannot move."""
    for replayed in (False, True):
        records = _corpus(replayed)
        assert components._one_per_game(records) == records


def test_copies_of_one_game_count_once() -> None:
    records = _corpus(False)[:4]
    doubled = records + records
    assert components._distinct_game_rate(doubled) == 1.0
    assert components._opening_diversity(doubled) == components._opening_diversity(records)


def test_a_count_no_longer_sits_below_its_point_when_every_game_differs(
    spec: dict, drawn: Drawn
) -> None:
    point = _point(spec, _corpus(False))
    result = drawn(False)
    for name in COUNTS:
        assert point[name] == 1.0
        assert result.component_mean[name] == pytest.approx(point[name], abs=1e-9)
        assert result.component_sd[name] == 0.0


def test_the_rule_is_what_removed_the_bias(drawn: Drawn) -> None:
    """The same draws without the rule read the copies as repeats."""
    plain = drawn(False, ruled=False)
    assert plain.component_mean["distinct_game_rate"] < 0.8
    assert plain.component_mean["opening_diversity"] < 0.95


@pytest.mark.parametrize("replayed", [False, True])
def test_averaging_components_are_untouched(drawn: Drawn, replayed: bool) -> None:
    ruled, plain = drawn(replayed), drawn(replayed, ruled=False)
    for name in AVERAGES:
        assert ruled.component_sd[name] == plain.component_sd[name]
        assert ruled.component_mean[name] == plain.component_mean[name]


def test_the_interval_does_not_depend_on_the_worker_count(spec: dict) -> None:
    """The seed is fixed per call, so dividing the work cannot change an answer.

    Two threads divide four corpora as surely as four did, on fewer cores than that
    (Q377c: a worker of the booked fast suite is one core)."""
    corpora = [_corpus(False), _corpus(True)] * 2
    answers = []
    for workers in (1, 2):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers.append(list(pool.map(lambda r: _bootstrap(spec, r), corpora)))
    assert answers[0] == answers[1]
    assert answers[0][0] == answers[0][2]


def test_the_bootstrap_agrees_with_a_half_sample_without_replacement(
    spec: dict, drawn: Drawn
) -> None:
    """Cross-check on a corpus with genuine repeats.

    A share of distinct games rises as the sample shrinks, because fewer games have
    fewer chances to collide. A resample holds about 68% of the games as identities
    and a half-sample holds 50%, so the bootstrap mean must lie between the point value
    and the half-sample mean. The two spreads must agree within a factor of 1.5.

    The comparison is against what the resample *drew*, so the published sd's stratum
    correction is divided back out. Both scalings here are rough -- the half-sample
    divides by root two and ignores its own finite-population factor -- and a band of
    1.5 is what either supports. `test_bootstrap_correction.py` holds the factor.
    """
    records = _corpus(True)
    point = _point(spec, records)["distinct_game_rate"]
    result = drawn(True)
    halves = _half_samples(records)
    half_mean = statistics.fmean(halves)
    half_sd = statistics.pstdev(halves) / 2**0.5

    assert point < result.component_mean["distinct_game_rate"] < half_mean
    drawn_sd = result.component_sd["distinct_game_rate"] / result.stratum_correction
    ratio = drawn_sd / half_sd
    assert 1 / 1.5 <= ratio <= 1.5


# ------------------------------------------------ the nearest other game (§3.1 to §3.3)

NEAREST = ("prefix_divergence", "position_divergence", "event_divergence")


def _keyed_corpus(replayed: bool) -> tuple[list[GameRecord], CorpusAnalysis]:
    """Four games per ordered pairing, with position keys and events.

    A position is how many of each of three moves have been played, so games share
    positions in part and transpose, and every distance lands between its ends. A move
    of kind 0 cannot be undone. With ``replayed`` each pairing's second game replays its
    first under its own seed, so the corpus holds genuine repeats.
    """
    rng = random.Random(11)
    records: list[GameRecord] = []
    games: list[GameAnalysis] = []
    seed = 0
    for first in AGENTS:
        for second in AGENTS:
            for game in range(4):
                moves = [rng.randrange(3) for _ in range(24)]
                if replayed and game == 1:
                    moves = list(records[-1].moves)
                counts = [0, 0, 0]
                keys = []
                for move in moves:
                    counts[move] += 1
                    keys.append(counts[0] * 10_000 + counts[1] * 100 + counts[2])
                records.append(
                    GameRecord(
                        ruleset_hash="fixture", ruleset_name="fixture", seed=seed,
                        agents=(first, second), moves=moves,
                        winner=rng.choice([0, 1, None]), reason="fixture", plies=24,
                        branching=[20] * 24, material=[0.0] * 24,
                        captures=[int(move == 0) for move in moves],
                    )
                )
                games.append(
                    GameAnalysis(
                        seed=seed, agents=(first, second), walked_plies=24,
                        position_keys=keys,
                    )
                )
                seed += 1
    return records, CorpusAnalysis(games=games)


def _keyed_bootstrap(
    spec: dict, records: list[GameRecord], analysis: CorpusAnalysis
) -> BootstrapResult:
    result = bootstrap_index(
        spec, records, load_weights("v2"), resamples=RESAMPLES, seed=0,
        components_of_interest=NEAREST, analysis=analysis,
    )
    assert result is not None
    return result


def _keyed_point(
    spec: dict, records: list[GameRecord], analysis: CorpusAnalysis
) -> dict[str, float]:
    return components.compute_components(spec, records, rate(records), analysis).raw


def _keyed_half_samples(
    spec: dict, records: list[GameRecord], analysis: CorpusAnalysis, draws: int = 200
) -> dict[str, list[float]]:
    """Each nearest distance on two of each pairing's four games, without replacement."""
    strata: dict[tuple[str, str], list[GameRecord]] = defaultdict(list)
    for record in records:
        strata[record.agents].append(record)
    rng = random.Random(0)
    readings: dict[str, list[float]] = {name: [] for name in NEAREST}
    for _ in range(draws):
        half = [g for games in strata.values() for g in rng.sample(games, len(games) // 2)]
        # Every seat searches, so the graded games are all of them.
        raw = {"prefix_divergence": components._prefix_divergence(half)}
        raw.update(components._keyed_components(half, analysis, {}))
        for name in NEAREST:
            readings[name].append(raw[name])
    return readings


@pytest.fixture(scope="module")
def keyed(spec: dict) -> Drawn:
    """The nearest-distance bootstraps, computed once for the module like `drawn`'s."""
    kept: dict[tuple[bool, bool], BootstrapResult] = {}

    def get(replayed: bool, ruled: bool = True) -> BootstrapResult:
        if (replayed, ruled) not in kept:
            records, analysis = _keyed_corpus(replayed)
            if ruled:
                kept[replayed, ruled] = _keyed_bootstrap(spec, records, analysis)
            else:
                with _without_the_rule():
                    kept[replayed, ruled] = _keyed_bootstrap(spec, records, analysis)
        return kept[replayed, ruled]

    return get


@pytest.mark.parametrize("replayed", [False, True])
def test_copies_of_one_game_are_never_neighbours(spec: dict, replayed: bool) -> None:
    records, analysis = _keyed_corpus(replayed)
    doubled = CorpusAnalysis(games=analysis.games + analysis.games)
    once = _keyed_point(spec, records, analysis)
    twice = _keyed_point(spec, records + records, doubled)
    for name in NEAREST:
        assert 0.0 < once[name] < 1.0, name
        assert twice[name] == once[name], name


def test_a_nearest_distance_sits_near_its_point(spec: dict, keyed: Drawn) -> None:
    """A resample holds fewer identities, so a neighbour can only be as near or farther.

    On a corpus of distinct games the mean sits a little above its point, never below it
    by more than noise, and §4's bootstrap-honesty arm allows at most 0.10 above. A
    corpus of replays is the exception and is not held to this: a replay's twin drops
    out of a resample a third of the time, and the replay then reads as far as any other
    game. With every pairing's second game a replay, the mean sits 0.2 above its point.
    """
    records, analysis = _keyed_corpus(replayed=False)
    point = _keyed_point(spec, records, analysis)
    result = keyed(False)
    for name in NEAREST:
        gap = result.component_mean[name] - point[name]
        assert -0.01 <= gap <= 0.10, (name, gap)
        assert result.component_sd[name] > 0.0, name


@pytest.mark.parametrize("replayed", [False, True])
def test_the_rule_is_what_keeps_a_copy_from_being_a_neighbour(
    keyed: Drawn, replayed: bool
) -> None:
    """The same draws without the rule find each drawn copy at distance zero."""
    ruled, plain = keyed(replayed), keyed(replayed, ruled=False)
    for name in NEAREST:
        assert plain.component_mean[name] < 0.7 * ruled.component_mean[name], name


def test_a_nearest_interval_does_not_depend_on_the_worker_count(spec: dict) -> None:
    """The distance tables are shared between threads and must never change an answer."""
    corpora = [_keyed_corpus(False), _keyed_corpus(True)] * 2
    answers = []
    for workers in (1, 2):  # two share the tables as surely as four (Q377c)
        components._TABLES.clear()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers.append(list(pool.map(lambda c: _keyed_bootstrap(spec, *c), corpora)))
    assert answers[0] == answers[1]
    assert answers[0][:2] == answers[0][2:]


@pytest.mark.parametrize("replayed", [False, True])
def test_a_nearest_bootstrap_agrees_with_a_half_sample_without_replacement(
    spec: dict, keyed: Drawn, replayed: bool
) -> None:
    """As for `distinct_game_rate`: a nearest distance rises as the sample shrinks, so the
    bootstrap mean lies between the point value and the half-sample mean, and the two
    spreads agree within a factor of 1.5."""
    records, analysis = _keyed_corpus(replayed)
    point = _keyed_point(spec, records, analysis)
    result = keyed(replayed)
    halves = _keyed_half_samples(spec, records, analysis)
    for name in NEAREST:
        half_mean = statistics.fmean(halves[name])
        half_sd = statistics.pstdev(halves[name]) / 2**0.5
        assert point[name] < result.component_mean[name] < half_mean, name
        drawn_sd = result.component_sd[name] / result.stratum_correction
        assert 1 / 1.5 <= drawn_sd / half_sd <= 1.5, (name, drawn_sd / half_sd)
