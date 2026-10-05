"""The rule-space fuzzer, run on a fixed seed set.

Generality is otherwise tested by example. Twenty-five rule keys is a combination space
nothing enumerates, and the project's own lesson is that rules that cannot interact in
chess can interact in a variant -- `obstacles` x `royal_facing` was caught in schema 6
and `obstacles` x `castling` was not, and the difference was whether a person thought to
ask.

What runs here is a fixed, small seed set, so it is a regression test rather than a
search: the same specs every time, and a failure is reproducible from the seed it names.
The search is `python -m chess_variant_sandbox.fuzz --cases N`, which is where the coverage report and
the rare combinations come from. See `chess_variant_sandbox/fuzz.py` for why the oracle list needed a
kind of oracle it did not have.
"""

from __future__ import annotations

import pytest

from chess_variant_sandbox import fuzz


def test_the_generator_actually_produces_specs() -> None:
    """The control, and it is not idle.

    Every assertion below is of the form "nothing went wrong", which a fuzzer that
    silently generates nothing satisfies perfectly. This is the one test here that fails
    if the generator stops working.
    """
    built = [fuzz.case(seed) for seed in range(20)]
    made = [case for case in built if case is not None]
    assert len(made) >= 18, "the generator is refusing most seeds"
    assert sum(1 for case in made if case.operators) >= 15, "most cases were not mutated"
    assert len({case.spec.get("name") for case in made}) > 5, "the cases are all alike"


def test_a_seed_rebuilds_exactly_the_same_spec() -> None:
    """A failure is a reproduction, or it is a rumour.

    The whole value of a seeded fuzzer is that the case it failed on can be rebuilt. If
    the generator drew from anywhere but its own seeded stream, a reported failure would
    name a spec nobody could get back.
    """
    first, second = fuzz.case(7), fuzz.case(7)
    assert first is not None and second is not None
    assert first.spec == second.spec
    assert first.operators == second.operators


def _hexagons(seeds: range) -> list[fuzz.Case]:
    built = [fuzz.case(seed) for seed in seeds]
    return [
        case for case in built
        if case is not None and case.spec["board"].get("shape") == "hexagon"
    ]


def test_a_seeded_batch_builds_hexagons_joined_and_unjoined() -> None:
    """Schema 16 (Q281a5). No operator adds a shape, so the hexagons come in as parents.

    Every committed ruleset is a grid, so without its own draw the fuzzer would never
    build one. Both kinds must turn up in a small batch: a hexagon with a seam and one
    without, since the landing function takes a different path for each.
    """
    hexagons = _hexagons(range(64))
    assert len(hexagons) >= 4, [case.describe() for case in hexagons]
    joined = {"wrap" in fuzz.active_rules(case.spec) for case in hexagons}
    assert joined == {True, False}
    assert all("board.shape" in fuzz.active_rules(case.spec) for case in hexagons)
    assert "board.shape" in fuzz.Coverage().report()


def test_the_obstacle_oracles_see_a_hexagon_s_corners() -> None:
    """Why the two obstacle oracles need no change: the corners are in `obstacles`."""
    from chess_variant_sandbox.engine.compile import compile_spec

    for case in _hexagons(range(64)):
        compiled = compile_spec(case.spec)
        assert compiled.absent, case.describe()
        assert compiled.absent <= compiled.obstacles, case.describe()


def test_a_hexagon_walks_under_every_oracle() -> None:
    """One of each kind walked here; the whole batch is in the slow run below."""
    hexagons = _hexagons(range(64))
    joined = next(case for case in hexagons if "wrap" in fuzz.active_rules(case.spec))
    unjoined = next(case for case in hexagons if "wrap" not in fuzz.active_rules(case.spec))
    for case in (joined, unjoined):
        assert fuzz.walk(case.spec, case.seed) > 0, case.describe()


@pytest.mark.slow
def test_every_hexagon_in_the_fixed_seed_set_walks_under_every_oracle() -> None:
    hexagons = _hexagons(range(200))
    assert len(hexagons) >= 15
    for case in hexagons:
        fuzz.walk(case.spec, case.seed)


@pytest.mark.slow
def test_the_oracles_hold_over_a_fixed_seed_set() -> None:
    """Two hundred generated rulesets, forty plies each, every oracle at every ply.

    Deliberately a *fixed* range rather than a random one: a test that draws new seeds
    every run is a test that fails in CI for reasons nobody can reproduce, and this
    project has enough of those in the measurement layer already.
    """
    coverage = fuzz.run(range(200))
    assert coverage.walked >= 150, f"too few cases survived generation: {coverage.report()}"


@pytest.mark.slow
def test_the_run_reaches_combinations_the_corpus_does_not_have() -> None:
    """The coverage claim, asserted rather than printed.

    A fuzzer nobody can read the coverage of is a fuzzer that gets deleted -- and one
    whose coverage is *asserted* is one that cannot quietly stop exploring. The bar is
    set below what a run currently reaches, because the number to defend is "it still
    crosses rules the corpus never crosses", not the exact figure of the day.
    """
    coverage = fuzz.run(range(200))
    assert len(coverage.pairs) >= 100, coverage.report()
    # The report has to name what was *not* reached, which is the half a future bump
    # actually needs. Reading it is the point; an empty list would mean the run had
    # covered everything, which would be the surprising outcome.
    report = coverage.report()
    assert "pairs never reached" in report
    assert "rules never active in any case" in report


def test_an_oracle_failure_names_the_seed_that_reproduces_it() -> None:
    """What a failure has to carry to be worth anything.

    The message is the interface: it is what a person reads at 400 cases in, and it has
    to say which seed to hand back to `fuzz.case`.
    """
    def always_fails(engine: object, state: object) -> None:
        raise fuzz.OracleFailure("a deliberately broken oracle")

    original = fuzz.RULE_OBEDIENCE
    try:
        fuzz.RULE_OBEDIENCE = (always_fails,)  # type: ignore[assignment]
        fuzz.ORACLES = fuzz.SELF_CONSISTENCY + fuzz.RULE_OBEDIENCE  # type: ignore[assignment]
        with pytest.raises(fuzz.OracleFailure) as caught:
            fuzz.run(range(3), plies=2)
    finally:
        fuzz.RULE_OBEDIENCE = original  # type: ignore[assignment]
        fuzz.ORACLES = fuzz.SELF_CONSISTENCY + original  # type: ignore[assignment]

    message = str(caught.value)
    assert "seed" in message
    assert "a deliberately broken oracle" in message
    seed = int(message.split("seed ")[1].split(":")[0])
    assert fuzz.case(seed) is not None, "the seed in the message has to rebuild the case"


def test_a_piece_may_not_be_dropped_onto_a_wall() -> None:
    """The fuzzer's first find, pinned as an ordinary regression test.

    `generate_drops` picks the empty squares, and a wall is empty -- `_compile_square`
    dissolved it, which is the whole design of the rule. So a piece could be dropped
    onto one. It is the fourth instance of the root cause item 1 enumerates and the
    second in `generate_drops`, whose *other* blindness -- `no_doubled` bucketing by
    file -- is genuinely fine.

    Fixed in the compiler rather than in the core, which is where the rest of the rule
    is answered: a piece with no drop region of its own gets one when there are walls,
    because "anywhere" has to stop meaning "including on a wall".
    """
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.engine.moves import MOVE_DROP, move_to, move_type

    spec = {
        "schema_version": 6, "name": "drop onto a wall",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0,
             "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            {"id": "S", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "S", "player": 1, "squares": ["d5"]},
        ]},
        # No drop region at all, which is the case the validator cannot refuse: there is
        # no square named for it to object to.
        "rules": {"win": "royal_capture", "drops": {"from_captures": True},
                  "obstacles": {"squares": ["c3"], "mirror": False}},
    }
    engine = Engine(spec)
    state = engine.initial_state().with_(hands=((0, 1), (0, 1)))
    drops = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_DROP]
    assert drops, "the position is meant to offer drops"
    assert not [m for m in drops if move_to(m) in engine.spec.obstacles]


def test_a_drop_region_that_names_a_wall_is_refused_by_name() -> None:
    """And the half the validator *can* refuse, which it had never actually checked.

    `_check_obstacle_coherence` claimed four rules that name a square a piece stands on.
    `rules.drops.region` is a map from piece id to squares, not a list of squares, and
    the helper returns immediately on anything that is not a list -- so the fourth of
    the four had never once run. Three rules were being checked and the comment said
    four.
    """
    from chess_variant_sandbox.spec.schema import validate_spec

    spec = {
        "schema_version": 6, "name": "drop region on a wall",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0,
             "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            {"id": "S", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "S", "player": 1, "squares": ["d5"]},
        ]},
        "rules": {"win": "royal_capture",
                  "drops": {"from_captures": True, "region": {"S": ["c3", "c4"]}},
                  "obstacles": {"squares": ["c3"], "mirror": False}},
    }
    errors = validate_spec(spec)
    assert any("rules.drops.region" in error and "obstacle" in error for error in errors), errors
