"""TS1 (`docs/plan-tessabi.md` §1): scoring goals, `rules.reach.removes`, `.count` and `.both`.

The bank is played in the core (`Game::write_bank`, undone by `unmake_bank`), the count and the
both-at-once reading in `Game::scoring_goal`. Every question here is asked of the engine; the
test only builds positions and reads what the core answers.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def _crossings(**reach: Any) -> dict[str, Any]:
    spec = load_spec(ROOT / "rulesets" / "crossings.json")
    spec["rules"]["reach"].update(reach)
    for key, value in list(spec["rules"]["reach"].items()):
        if value is None:
            del spec["rules"]["reach"][key]
    return spec


# ------------------------------------------------------------------------ inert


def test_the_keys_at_their_off_values_are_the_goal_without_them() -> None:
    plain = load_spec(ROOT / "rulesets" / "kingofthehill.json")
    written = copy.deepcopy(plain)
    written["rules"]["reach"].update({"removes": False, "count": 1, "both": "first"})
    assert validate_spec(written) == []
    assert spec_hash(written) == spec_hash(plain)
    assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(plain))


def test_a_goal_without_the_keys_draws_no_bank_keys() -> None:
    compiled = compile_spec(load_spec(ROOT / "rulesets" / "runners.json"))
    assert compiled.reach is not None and compiled.reach.bank_zobrist == ()
    assert not compiled.reach.counts


# ------------------------------------------------------------------------ the bank


def test_an_arrival_is_banked_and_taken_off() -> None:
    engine = Engine(_crossings())
    state = engine.make_state({"c6": ("R", 0), "e2": ("R", 1), "a7": ("R", 1)})
    banked = engine.apply(state, engine.find_move(state, "Rc6-c7"))
    assert banked.banked == (1, 0)
    assert banked.board[engine._square_index("c7")] == 0
    assert banked.key == engine._compute_hash(banked)
    # A bank is material leaving the board: the repetition window and the clock start again.
    assert banked.halfmove == 0 and banked.repetition == (banked.key,)
    assert engine.terminal(banked) is None


def test_the_count_decides_and_the_bank_counts_towards_it() -> None:
    engine = Engine(_crossings())
    state = engine.make_state({"c6": ("R", 0), "e2": ("R", 1), "a7": ("R", 1)}, banked=(2, 0))
    after = engine.apply(state, engine.find_move(state, "Rc6-c7"))
    assert after.banked == (3, 0)
    outcome = engine.terminal(after)
    assert outcome is not None and (outcome.winner, outcome.reason) == (0, "reached")


def test_banked_key_tells_two_banks_apart() -> None:
    engine = Engine(_crossings())
    placement = {"c4": ("R", 0), "e4": ("R", 1)}
    one, two = (engine.make_state(placement, banked=(n, 0)) for n in (1, 2))
    assert one.board == two.board and one.key != two.key


def test_a_game_banks_and_its_key_stays_the_core_s_own() -> None:
    """Random games: every ply's incremental key is `compute_hash`'s, and the banks only grow."""
    engine = Engine(_crossings())
    seen_bank = False
    for seed in range(6):
        rng = random.Random(seed)
        state = engine.initial_state()
        while engine.terminal(state) is None:
            before = state.banked
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
            assert state.key == engine._compute_hash(state)
            assert state.banked[0] >= before[0] and state.banked[1] >= before[1]
            seen_bank |= state.banked != (0, 0)
    assert seen_bank


# ------------------------------------------------------------------------ both at once


@pytest.mark.parametrize(
    ("both", "winner"), [(None, 0), ("draw", None), ("mover", 1)],
)
def test_a_goal_both_reach_at_once_is_read_by_both(both: str | None, winner: int | None) -> None:
    """A position where both players stand two runners on their goal under a count of two
    (built by hand: no move of crossings brings both home at once), player 0 to move, so the
    side whose move made it is player 1."""
    engine = Engine(_crossings(removes=None, count=2, both=both))
    state = engine.make_state(
        {"a7": ("R", 0), "b7": ("R", 0), "a1": ("R", 1), "b1": ("R", 1), "d4": ("R", 0)},
    )
    outcome = engine.terminal(state)
    assert outcome is not None and (outcome.winner, outcome.reason) == (winner, "reached")


def test_a_count_without_a_bank_counts_the_standing_pieces() -> None:
    engine = Engine(_crossings(removes=None, count=2))
    one = engine.make_state({"a7": ("R", 0), "d4": ("R", 0), "d5": ("R", 1)})
    assert engine.terminal(one) is None
    stood =engine.make_state({"a7": ("R", 0), "b6": ("R", 0), "d5": ("R", 1)})
    after = engine.apply(stood, engine.find_move(stood, "Rb6-b7"))
    assert after.banked == (0, 0) and after.board[engine._square_index("b7")] != 0
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0


# ------------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    ("edit", "words"),
    [
        (lambda s: s["rules"]["reach"].update(on="leave"), 'on "leave"'),
        (lambda s: s["rules"]["reach"].update(safe=True), "beside rules.reach.safe"),
        (lambda s: s["rules"].update(win="checkmate") or s["pieces"][0].update(royal=True),
         "royal type"),
        (lambda s: s["rules"].update(forbid_immobility=True), "forbid_immobility"),
        (lambda s: s["rules"].update(extinction={"pieces": "any"}), "rules.extinction"),
        (lambda s: s["rules"]["reach"].update(count=15), "can never be reached"),
        (lambda s: s["rules"].update(simultaneous={}), "rules.simultaneous"),
    ],
    ids=["leave", "safe", "royal", "immobility", "extinction", "supply", "rounds"],
)
def test_a_scoring_goal_is_refused_where_it_has_no_reading(edit: Any, words: str) -> None:
    spec = _crossings()
    edit(spec)
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


def test_the_core_holds_a_bank_beside_check_shut() -> None:
    """`rust_supports` refuses what the validator refuses, for a spec that got past it."""
    compiled = replace(compile_spec(_crossings()), forbid_check=True)
    assert "check" in (rust_supports(compiled) or "")
    assert rust_supports(compile_spec(_crossings())) is None
