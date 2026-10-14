"""The paired estimator and the searched valuation, and what each of them may not do.

Two things arrive together here and only one of them is about piece values.

``tournament/duel.py`` is a way of comparing two players, and the property the rest of
the stage rests on is exact rather than statistical: **two spellings of one player score
exactly 0.5 with exactly zero spread**. That is not an approximation that happens to hold
on a corpus; it follows from the seeds being matched across the two colours and each
seat's stream being derived from the seed and the seat, and a test that let it become
approximate would let the whole estimator become a slightly biased win rate.

``agents/valuations.py`` is a piece-value table found by hill-climbing that estimator.
Most of what is tested about it is the *refusals*, for the same reason ``test_fitted.py``
is mostly about narrowness: a searched table that shipped without beating the derived one
would be a differently-priced player wearing a better name, and the store is what makes
that impossible to do quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox.agents import native_spec, parse_agent
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import builtin_vector, feature_context
from chess_variant_sandbox.agents.heuristics import Heuristic, register_heuristic, resolve
from chess_variant_sandbox.agents.objective_options import objective_values, objective_weights
from chess_variant_sandbox.agents.valuations import (
    SEARCH,
    SearchProcedure,
    Valuation,
    ValuationError,
    load_valuation,
    normalise_procedure,
    register_valuation,
    resolve_valuation,
    searched_values,
    searched_weights,
    store_path,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.duel import _graded, duel, grade
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig, schedule

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# Two real agent names. `schedule` parses every name before it hands out a seed, so a
# placeholder would fail there rather than in the scheduling this is about.
A = "minimax:nodes=100"
B = "minimax:nodes=200"
A_B = (A, B)


def compiled(name: str):
    spec = load_spec(RULESETS / f"{name}.json")
    return spec, Engine(spec).spec


def identity_table(spec) -> tuple[float, ...]:
    """The derived table with royal slots explicit. An override equal to no override."""
    derived = Evaluator.for_spec(spec).values
    royal = set(spec.royal_types)
    return tuple(0.0 if i in royal else v for i, v in enumerate(derived))


def valuation_for(spec, vector, accepted=True, procedure="p" * 12) -> Valuation:
    return Valuation(
        ruleset_hash=spec.hash,
        procedure=procedure,
        piece_ids=tuple(spec.piece_ids),
        vector=tuple(vector),
        derived=tuple(Evaluator.for_spec(spec).values),
        weights=builtin_vector(feature_context(spec)),
        accepted=accepted,
        accept_score=0.6 if accepted else 0.4,
        accept_error=0.04,
        accept_games=96,
    )


def record(seed: int, agents: tuple[str, str], winner: int | None, moves=(1, 2)) -> GameRecord:
    return GameRecord(
        ruleset_hash="h",
        ruleset_name="n",
        seed=seed,
        agents=agents,
        moves=list(moves),
        winner=winner,
        reason="test",
        plies=len(moves),
    )


# --------------------------------------------------------------- the paired schedule


def test_the_paired_schedule_gives_both_colours_of_a_pairing_the_same_seeds():
    config = TournamentConfig(
        agents=A_B, games_per_pairing=3, include_self_play=False, paired=True
    )
    tasks = schedule(config)
    forward = sorted(seed for seed, first, _ in tasks if first == A)
    backward = sorted(seed for seed, first, _ in tasks if first == B)
    assert forward == backward
    assert len(forward) == 3


def test_an_unpaired_schedule_is_exactly_what_it_always_was():
    """Every committed run was scheduled this way; renumbering it would replay a
    different corpus under an unchanged name."""
    config = TournamentConfig(agents=A_B, games_per_pairing=2)
    assert schedule(config) == [
        (0, A, A), (1, A, A),
        (2, A, B), (3, A, B),
        (4, B, A), (5, B, A),
        (6, B, B), (7, B, B),
    ]


def test_a_paired_schedule_still_covers_self_play_once():
    config = TournamentConfig(agents=A_B, games_per_pairing=2, paired=True)
    tasks = schedule(config)
    assert sum(1 for _seed, first, second in tasks if first == second == A) == 2
    # Every seed is used by exactly the two orderings of a pairing, or once by a
    # self-pairing, so nothing is scheduled twice under one seed and one ordering.
    assert len(set(tasks)) == len(tasks)


# --------------------------------------------------------------- the estimator


def test_a_null_arm_scores_exactly_a_half_with_exactly_no_spread():
    """The property the whole stage rests on, and it is exact rather than statistical.

    An override equal to the derived table is the identity under ``rescaled``, so these
    two digests are one player. Matched seeds then make the two games of every pair each
    other's mirror, and mirrored results cancel to the bit.
    """
    spec, spec_c = compiled("gardner")
    weights = builtin_vector(feature_context(spec_c))
    plain = Heuristic(name="plain", plan="", weights=weights)
    echoed = Heuristic(name="echoed", plan="", weights=weights, values=identity_table(spec_c))
    assert plain.digest != echoed.digest
    for heuristic in (plain, echoed):
        # A digest is taken over the unit-length weights, but the core plays the raw
        # weights and is not bitwise scale-invariant. So a shorter spelling registered
        # earlier in this process would be the player these strings name: Q431, 0.4167
        # with 3 of 6 pairs identical. Since Q433 registration refuses a second spelling
        # of a held digest, and conftest clears registrations between tests; this says so
        # if one from a wider scope ever gets here first.
        held = resolve(register_heuristic(heuristic))
        assert (held.weights, held.values) == (heuristic.weights, heuristic.values), (
            f"{heuristic.digest} is already registered as another vector, so "
            f"{heuristic.name!r} is not the player its digest names in this process"
        )

    result = duel(
        spec,
        f"minimax:h={echoed.digest},nodes=400",
        f"minimax:h={plain.digest},nodes=400",
        pairs=6,
        seed=0,
        workers=1,
    )
    assert result.score == 0.5
    assert result.error == 0.0
    assert result.identical == 6
    assert result.games == 12
    assert not result.decided


def test_a_duel_refuses_one_player_named_twice():
    spec, _ = compiled("gardner")
    with pytest.raises(ValueError, match="two spellings of one player"):
        duel(spec, "minimax:nodes=100", "minimax:nodes=100", pairs=2)


def test_grading_refuses_a_schedule_that_was_not_paired():
    """An unpaired half-sample would produce a number of the right shape with none of
    the variance reduction, which is the worst failure available here."""
    records = [record(0, ("a", "b"), 0), record(1, ("b", "a"), 0)]
    with pytest.raises(ValueError, match="the schedule was not paired"):
        grade(records, "a", "b")


def test_grading_refuses_a_pair_that_is_not_a_mirror():
    records = [record(0, ("a", "b"), 0), record(0, ("a", "b"), 1)]
    with pytest.raises(ValueError, match="not each other's mirror"):
        grade(records, "a", "b")


def test_a_mirrored_pair_cancels_and_an_unmirrored_one_does_not():
    mirrored = [record(0, ("a", "b"), 0), record(0, ("b", "a"), 1)]
    assert grade(mirrored, "a", "b").score == 1.0
    split = [record(0, ("a", "b"), 0), record(0, ("b", "a"), 0)]
    assert grade(split, "a", "b").score == 0.5


def test_the_measured_error_is_the_samples_own_and_not_a_binomial_constant():
    even = [record(s, ("a", "b"), 0) for s in range(4)]
    even += [record(s, ("b", "a"), 1) for s in range(4)]
    result = grade(even, "a", "b")
    # Every pair said the same thing, so there is no spread to report -- where a
    # binomial formula would have claimed sqrt(0.25 / 8) = 0.177.
    assert result.score == 1.0
    assert result.error == 0.0
    assert result.decided


def test_a_margin_blend_can_never_reorder_a_result_below_a_third():
    """The bound that makes the blend a *grading* of the result rather than a vote
    against it, and the bound is a third rather than the half it first looked like.

    Worst win is ``1 - w`` and best draw ``0.5 + 0.5 w``; they cross at ``w = 1/3``, and
    the draw-against-loss pair crosses at the same place. Every weight the calibration
    swept is below it, and the weight that shipped is zero.
    """
    drawn_with_a_lead = record(0, ("a", "b"), None)
    drawn_with_a_lead.material = [40.0]
    drawn_from_behind = record(0, ("a", "b"), None)
    drawn_from_behind.material = [-40.0]
    won_from_behind = record(0, ("a", "b"), 0)
    won_from_behind.material = [-40.0]
    lost_with_a_lead = record(0, ("a", "b"), 1)
    lost_with_a_lead.material = [40.0]

    for weight in (0.1, 0.25, 0.32):
        worst_win = _graded(won_from_behind, 0, 10.0, weight)
        best_draw = _graded(drawn_with_a_lead, 0, 10.0, weight)
        worst_draw = _graded(drawn_from_behind, 0, 10.0, weight)
        best_loss = _graded(lost_with_a_lead, 0, 10.0, weight)
        assert worst_win > best_draw > worst_draw > best_loss

    # And it really does break above the bound, which is why the bound is stated.
    assert _graded(won_from_behind, 0, 10.0, 0.4) < _graded(drawn_with_a_lead, 0, 10.0, 0.4)


def test_the_margin_is_switched_off_and_that_is_the_finding():
    """Zero by measurement: the blend lowers the error and lowers the signal faster.

    Pinned so a future edit has to be a deliberate one with a sweep behind it.
    """
    from chess_variant_sandbox.tournament import duel as module

    assert module.MARGIN_WEIGHT == 0.0
    plain = record(0, ("a", "b"), 0)
    plain.material = [-40.0]
    assert _graded(plain, 0, 10.0, module.MARGIN_WEIGHT) == 1.0


# --------------------------------------------------------------- the procedure


def test_the_search_procedure_digest_is_pinned():
    """`store_path` and every agent name carry this string, so an accidental edit to a
    constant above would orphan the store rather than announce itself."""
    assert SEARCH.digest == "976d0dc3670c"


def test_the_procedure_digest_moves_when_the_procedure_does():
    assert SearchProcedure(nodes=400).digest != SEARCH.digest
    assert SearchProcedure(accept_seed=7).digest != SEARCH.digest
    assert SearchProcedure().digest == SEARCH.digest


def test_a_procedure_digest_survives_the_agent_string_round_trip():
    """One digest in sixteen starts with a zero and `parse_agent` coerces what looks
    like a number; the repair is the same one heuristics needed."""
    assert normalise_procedure(96717939346) == "096717939346"
    assert normalise_procedure("976d0dc3670c") == "976d0dc3670c"
    options = parse_agent("minimax:values=976d0dc3670c,nodes=400").options
    assert normalise_procedure(options["values"]) == "976d0dc3670c"


# --------------------------------------------------------------- the store


def test_a_valuation_round_trips_through_its_file(tmp_path):
    _spec, spec_c = compiled("gardner")
    original = valuation_for(spec_c, identity_table(spec_c))
    path = original.save(tmp_path / "v.json")
    assert Valuation.load(path) == original


def test_a_valuation_refuses_a_table_written_for_another_game():
    _spec, spec_c = compiled("gardner")
    with pytest.raises(ValuationError, match="different game"):
        valuation_for(spec_c, identity_table(spec_c) + (1.0,))


def test_a_valuation_refuses_labels_that_do_not_match_the_numbers():
    _spec, spec_c = compiled("gardner")
    good = valuation_for(spec_c, identity_table(spec_c))
    with pytest.raises(ValuationError, match="labels and the numbers disagree"):
        Valuation(**{**good.__dict__, "piece_ids": good.piece_ids[:-1]})


def test_a_valuation_from_another_procedure_version_is_refused_rather_than_reused(tmp_path):
    _spec, spec_c = compiled("gardner")
    path = valuation_for(spec_c, identity_table(spec_c)).save(tmp_path / "v.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["procedure_version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValuationError, match="search again rather than reusing it"):
        Valuation.load(path)


def test_a_weight_vector_from_another_vocabulary_is_refused_at_load(tmp_path):
    """Q430. Every store searched before `facing_alignment` carried fourteen weights, and
    the loader let them through: the procedure digest does not cover the vocabulary, so
    the file sat at the current path, and the core refused it on the first move of a
    game against it. Refused here, accepted or not, as a fit store is. Run on an accepted
    table only: `from_dict` measures the weights before it reads `accepted` at all, so the
    `[False]` case ran the same lines (Q754)."""
    from chess_variant_sandbox.agents.features import VOCABULARY
    from chess_variant_sandbox.agents.valuations import load_all

    _spec, spec_c = compiled("gardner")
    current = valuation_for(spec_c, identity_table(spec_c), accepted=True)
    assert len(current.weights) == len(VOCABULARY)
    stale = Valuation(**{**current.__dict__, "weights": current.weights[:-1]})
    path = stale.save(store_path(spec_c.hash, stale.procedure, tmp_path))

    message = f"carries {len(VOCABULARY) - 1} weights where .* has {len(VOCABULARY)} terms"
    with pytest.raises(ValuationError, match=message):
        Valuation.load(path)
    with pytest.raises(ValuationError, match=message):
        load_valuation(spec_c.hash, stale.procedure, tmp_path)
    # Skipped by the bulk loader, which reads a directory of mixed vintages.
    assert load_all(tmp_path) == 0
    # And a current one beside it still loads.
    current.save(path)
    assert Valuation.load(path) == current


def test_a_renamed_store_file_is_refused(tmp_path):
    _spec, spec_c = compiled("gardner")
    valuation = valuation_for(spec_c, identity_table(spec_c))
    valuation.save(store_path("wrong-hash", valuation.procedure, tmp_path))
    with pytest.raises(ValuationError, match="the file has been renamed"):
        load_valuation("wrong-hash", valuation.procedure, tmp_path)


def test_a_verdict_names_the_core_it_was_measured_on():
    """The engine decides the games and neither the procedure digest nor the ruleset hash
    covers it, so a store without this is a set of numbers with no build behind them.

    Absent reads as *unknown*, never as agreement -- the safe direction, exactly as an
    absent `measurement.Basis` is treated as stale.
    """
    from chess_variant_sandbox.agents.valuations import stale_verdict
    from chess_variant_sandbox.measurement import built_fingerprint

    _spec, spec_c = compiled("gardner")
    table = identity_table(spec_c)

    unstamped = valuation_for(spec_c, table)
    assert unstamped.build == ""
    assert "records no core build" in (stale_verdict(unstamped) or "")

    current = Valuation(**{**unstamped.__dict__, "build": built_fingerprint()})
    assert stale_verdict(current) is None

    other = Valuation(**{**unstamped.__dict__, "build": "0000000000000000"})
    assert "may no longer hold" in (stale_verdict(other) or "")


def test_the_build_survives_the_file_round_trip(tmp_path):
    _spec, spec_c = compiled("gardner")
    original = Valuation(
        **{**valuation_for(spec_c, identity_table(spec_c)).__dict__, "build": "abc123"}
    )
    assert Valuation.load(original.save(tmp_path / "v.json")).build == "abc123"


def test_every_committed_valuation_names_the_core_that_measured_it():
    """A store shipped without one cannot be told from a store measured on this core, and
    the second is the only one whose number means anything here."""
    from chess_variant_sandbox.agents.valuations import STORE_DIR

    root = Path(__file__).resolve().parents[1] / STORE_DIR
    missing = [
        path.name
        for path in sorted(root.glob("*.json"))
        if not Valuation.load(path).build
    ]
    assert not missing


def test_a_missing_valuation_is_absent_rather_than_an_error(tmp_path):
    assert load_valuation("nothing", SEARCH.digest, tmp_path) is None


def test_resolution_never_defaults():
    with pytest.raises(ValuationError, match="no valuation for procedure"):
        resolve_valuation("ffffffffffff", "not-a-ruleset")


def test_resolution_refuses_to_guess_which_board_it_is_on():
    """A searched table is per-ruleset; a caller that has not passed its hash through is
    a bug, and defaulting would hide it behind a plausible player."""
    with pytest.raises(ValuationError, match="has not passed its ruleset hash through"):
        resolve_valuation("ffffffffffff", "")


# --------------------------------------------------------------- the refusal


def test_a_refused_search_falls_all_the_way_back_to_the_ordinary_agent():
    """Both halves empty, not the built-in vector with no values: the first is
    `minimax:nodes=N` to the bit and the second only agrees with it."""
    _spec, spec_c = compiled("gardner")
    refused = valuation_for(spec_c, [v * 2 for v in identity_table(spec_c)], accepted=False)
    register_valuation(refused, replace=True)
    options = {"values": refused.procedure}
    assert searched_values(options, spec_c.hash) == ()
    assert searched_weights(options, spec_c.hash) == ()

    plain = native_spec("minimax:nodes=400", spec_c.hash)
    refused_agent = native_spec(
        f"minimax:values={refused.procedure},nodes=400", spec_c.hash
    )
    assert refused_agent == plain


def test_an_accepted_search_reaches_the_agent_as_both_halves():
    _spec, spec_c = compiled("gardner")
    table = tuple(v * 3 for v in identity_table(spec_c))
    accepted = valuation_for(spec_c, table, procedure="a" * 12)
    register_valuation(accepted, replace=True)
    spec = native_spec(f"minimax:values={accepted.procedure},nodes=400", spec_c.hash)
    assert spec is not None
    from chess_variant_sandbox.agents import NATIVE_VALUES, NATIVE_WEIGHTS

    assert spec[NATIVE_VALUES] == table
    assert spec[NATIVE_WEIGHTS] == builtin_vector(feature_context(spec_c))


# --------------------------------------------------------------- the exclusions


@pytest.mark.parametrize(
    "agent",
    [
        "minimax:h=aaaaaaaaaaaa,values=bbbbbbbbbbbb,nodes=400",
        "minimax:fit=aaaaaaaaaaaa,values=bbbbbbbbbbbb,nodes=400",
        "minimax:h=aaaaaaaaaaaa,fit=bbbbbbbbbbbb,nodes=400",
    ],
)
def test_two_claims_about_where_the_numbers_came_from_are_refused(agent):
    options = parse_agent(agent).options
    with pytest.raises(ValueError, match="at once"):
        objective_weights(options, "hash")
    with pytest.raises(ValueError, match="at once"):
        objective_values(options, "hash")


def test_the_rated_roster_carries_no_searched_valuation():
    """Stage C's danger, held down by a test rather than by care.

    Every skill component reads `CANONICAL_AGENTS` to ask *does thinking harder win?*
    If a rung also priced the pieces differently it would beat the rung below it for two
    reasons and the measurement would stop being about the game. Putting one on the
    ladder is a `v10` re-anchor with a calibration campaign behind it, never an edit.
    """
    from chess_variant_sandbox.corpus import CANONICAL_AGENTS

    assert not any("values=" in agent for agent in CANONICAL_AGENTS)


def test_no_committed_heuristic_set_prices_the_pieces():
    """The proposer must not write piece values, and this is the door it would come
    through: a stated set may carry a `values` array and none may.

    Sharper now than when it was written. On a chess-like ruleset a model writes
    1/3/3/5/9 from memory, and `runs/stage30/pilots.txt` measures those as *worse* than
    the derived table at every budget the rated ladder reaches.
    """
    from chess_variant_sandbox.agents.heuristics import HeuristicSet

    root = Path(__file__).resolve().parents[1] / "heuristics"
    offenders = []
    for path in sorted(root.rglob("*.json")):
        stored = HeuristicSet.from_dict(json.loads(path.read_text(encoding="utf-8")))
        offenders += [
            f"{path.name}:{h.name}" for h in stored.heuristics if h.values
        ]
    assert not offenders


# ---------------------------------------------------------------- the search itself
#
# `tournament/valuing.py` and `chess-variant-sandbox values`, its one caller, had never run in
# the suite (Q751, plan-tests F9). `SEARCH` plays thousands of games, so each test runs a
# procedure small enough to finish in a fraction of a second: 200 nodes, two seed-pairs, one
# coarse step. Gardner, because it has five priced pieces and a royal one.

def _tiny(accept_sigma: float, max_comparisons: int) -> SearchProcedure:
    return SearchProcedure(nodes=200, pairs=2, steps=(4.0,), sweeps=1,
                           max_comparisons=max_comparisons, accept_sigma=accept_sigma,
                           accept_nodes=200, accept_pairs=2)


# Any candidate whose games differ from the incumbent's is kept: the bar is a billion
# standard errors *below* a half. It makes a search that takes steps, which is the path
# under test, without depending on whether a 200-node step happens to help.
EAGER = _tiny(-1e9, 3)


def test_the_command_searches_writes_the_verdict_and_then_reuses_it(
    tmp_path, capsys, monkeypatch
):
    from chess_variant_sandbox import cli
    from chess_variant_sandbox.agents import valuations
    from chess_variant_sandbox.measurement import built_fingerprint
    from chess_variant_sandbox.tournament import valuing

    monkeypatch.setattr(valuations, "SEARCH", EAGER)
    argv = ["values", str(RULESETS / "gardner.json"), "--store", str(tmp_path),
            "--workers", "1", "--quiet"]
    code = cli.main(argv)
    printed = capsys.readouterr().out
    spec = Engine(load_spec(RULESETS / "gardner.json")).spec
    path = store_path(spec.hash, EAGER.digest, tmp_path)
    found = Valuation.load(path)  # written whether accepted or refused
    # The exit code is the verdict, and the verdict is strictly better or nothing.
    assert code == (0 if found.accepted else 1)
    assert found.accepted == (found.accept_score > 0.5)

    derived = tuple(Evaluator.for_spec(spec).values)
    assert found.derived == derived and found.piece_ids == spec.piece_ids
    # Coordinate ascent: one piece at a time, by one step up or down, and a piece the
    # search kept a step for is one accepted step. Five priced pieces take more than the
    # three comparisons allowed, so the search runs out, and says so.
    ratios = [v / d if d else None for v, d in zip(found.vector, derived, strict=True)]
    assert all(r in (None, 1.0, 4.0, 0.25) for r in ratios), ratios
    assert found.accepted_steps == sum(r not in (None, 1.0) for r in ratios)
    assert (found.comparisons, found.exhausted) == (3, True)
    assert "(ran out of comparisons)" in printed
    # The royal piece is pinned at zero and never tried.
    assert found.vector[spec.piece_ids.index("K")] == 0.0
    # A pawn worth four pawns plays different games at 200 nodes, so a step is kept and
    # the verdict is a held-out match that was played, with the ladder number beside it.
    assert found.accepted_steps >= 1
    assert found.accept_games == 2 * EAGER.accept_pairs
    assert set(found.plies) == {str(b) for b in valuing.PROBE_BUDGETS}
    assert found.build == built_fingerprint()
    verdict = "accepted" if found.accepted else "REFUSED, falls back to the derived table"
    assert f"over {found.accept_games} games -- {verdict}" in printed

    # A stored verdict is read, not searched again.
    def no_search(*_: object, **__: object) -> None:
        raise AssertionError("searched again")

    monkeypatch.setattr(valuing, "timed_search", no_search)
    assert cli.main(argv) == code
    assert f"reusing {path}" in capsys.readouterr().out


def test_a_search_that_moves_nothing_is_refused_without_a_match():
    """It *is* the derived table, so the match would report exactly 0.5 for a hundred
    games, and a refusal that played would look like a search that tried and lost."""
    from chess_variant_sandbox.agents.valuations import registered_valuations
    from chess_variant_sandbox.tournament.valuing import search_valuation

    # An infinite bar: no candidate clears it, not even one that won every game.
    stubborn = _tiny(float("inf"), 2)
    found = search_valuation(load_spec(RULESETS / "gardner.json"), stubborn, workers=1)
    assert found.vector == found.derived
    assert (found.comparisons, found.accepted_steps) == (2, 0)
    assert not found.accepted
    assert (found.accept_games, found.accept_score, found.ladder_score) == (0, 0.5, 0.5)
    # Nothing was registered under the procedure, so nothing can play it.
    assert (stubborn.digest, found.ruleset_hash) not in registered_valuations()


def test_the_search_is_a_function_of_the_spec_and_the_procedure_alone():
    """The same table at one worker or two: the schedule fixes every seed first."""
    from chess_variant_sandbox.agents.valuations import registered_valuations
    from chess_variant_sandbox.tournament.valuing import search_valuation

    spec = load_spec(RULESETS / "gardner.json")
    one = search_valuation(spec, EAGER, workers=1)
    # A search that played its acceptance match registers its verdict by its real name.
    assert registered_valuations()[(EAGER.digest, one.ruleset_hash)] == one
    assert search_valuation(spec, EAGER, workers=2) == one
