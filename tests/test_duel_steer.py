"""The steering rule: its boundaries, its error rates, and that it never reaches a verdict.

``tournament/duel.steer`` is a group-sequential rule for duels that only choose the next
experiment (``docs/reference/compute-economy.md`` §4.3, C4). It was pre-registered in
``docs/reference/learned-evaluator.md`` §9 with the numbers frozen below, before anything
used it, so a change here that moves one of them is a new rule and goes back through that
section first.

The error rates are proved exactly, not sampled: the one-sided error is the worst over
every pattern of tied pairs, because under no difference each decisive pair is a fair
coin whatever the ties do. The simulation test only checks that the exact calculator
models the function a caller actually runs.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from chess_variant_sandbox.tournament import duel as duel_module
from chess_variant_sandbox.tournament.duel import (
    CANDIDATE,
    CONTINUE,
    INCONCLUSIVE,
    REFERENCE,
    STEER_ALPHA,
    STEER_BOUND,
    STEER_FUTILITY,
    STEER_LOOKS,
    TOO_CLOSE,
    Duel,
    steer,
    steer_call,
    steer_characteristics,
    steering_duel,
)

PACKAGE = Path(duel_module.__file__).resolve().parents[1]

# §9.2: the fewest pairs the candidate must win outright, of ``m`` decisive pairs, for its
# call. Under seven decisive pairs no result is a call.
REGISTERED_WINS_NEEDED = {
    7: 7, 8: 8, 9: 9, 10: 9, 11: 10, 12: 11, 13: 12, 14: 12, 15: 13, 16: 13, 17: 14, 18: 15,
    19: 15, 20: 16, 21: 17, 22: 17, 23: 18, 24: 19,
}

# §9.3: the worst one-sided error over every tie pattern, futility ignored.
REGISTERED_WORST_ALPHA = 0.0210

# §9.4: (tie rate, decisive pairs the candidate wins) -> (P candidate, P reference,
# P too close, expected pairs), futility obeyed.
REGISTERED_CHARACTERISTICS = {
    (0.0, 0.5): (0.0142, 0.0142, 0.6243, 16.5),
    (0.0, 0.7): (0.3155, 0.0001, 0.2447, 18.4),
    (0.0, 0.8): (0.7179, 0.0000, 0.0630, 16.9),
    (0.0, 0.9): (0.9774, 0.0000, 0.0049, 13.0),
    (0.3, 0.5): (0.0103, 0.0103, 0.7457, 14.5),
    (0.3, 0.7): (0.2101, 0.0001, 0.4389, 17.2),
    (0.3, 0.8): (0.5298, 0.0000, 0.2001, 18.4),
    (0.3, 0.9): (0.8805, 0.0000, 0.0450, 17.1),
    (0.6, 0.5): (0.0064, 0.0064, 0.6896, 16.0),
    (0.6, 0.9): (0.5339, 0.0000, 0.1213, 20.6),
}


def test_the_design_is_the_registered_one() -> None:
    assert STEER_LOOKS == (8, 16, 24)
    assert STEER_BOUND == 2.5
    assert STEER_FUTILITY == (0.5, 1.0)
    assert STEER_ALPHA == 0.025


@pytest.mark.parametrize("decisive", range(1, 25))
def test_the_boundary_is_the_registered_table(decisive: int) -> None:
    """The fewest outright wins that make the call, at every look that count can reach."""
    for look, size in enumerate(STEER_LOOKS):
        if decisive > size:
            continue
        called = [w for w in range(decisive + 1) if steer_call(look, w, decisive - w) == CANDIDATE]
        assert min(called, default=None) == REGISTERED_WINS_NEEDED.get(decisive)
        for wins in range(decisive + 1):
            # The mirror: the same count of losses is the reference's call.
            mirrored = steer_call(look, decisive - wins, wins) == REFERENCE
            assert mirrored == (steer_call(look, wins, decisive - wins) == CANDIDATE)


def test_steer_reads_the_looks_in_order() -> None:
    assert steer([]).call == CONTINUE and steer([]).next_look == 8
    assert steer([1.0] * 7).next_look == 8
    # A sweep stops at the first look.
    assert steer([1.0] * 8).call == CANDIDATE
    assert steer([0.0] * 8).call == REFERENCE
    # Quarter pairs count as wins and losses: the sign, not the size.
    assert steer([0.75] * 7 + [0.5]).call == CANDIDATE
    # Eight ties is too close to be worth sixteen more.
    assert steer([0.5] * 8).call == TOO_CLOSE
    # 6-2 is neither: play to sixteen.
    six_two = [1.0] * 6 + [0.0] * 2
    assert steer(six_two).call == CONTINUE and steer(six_two).next_look == 16
    # 12 wins, 2 losses, 2 ties at sixteen: z = 10 / sqrt(14) = 2.67.
    assert steer(six_two + [1.0] * 6 + [0.5] * 2).call == CANDIDATE
    # Unresolved at the last look.
    # Unresolved at the last look: z 1.13, 1.60, then 1.28.
    block = [1.0] * 5 + [0.0] * 2 + [0.5]
    assert steer(block * 2 + [1.0] * 4 + [0.0] * 4).call == INCONCLUSIVE


def test_pairs_past_the_deciding_look_are_never_read() -> None:
    decided = steer([1.0] * 8)
    for tail in ([0.0] * 8, [0.0] * 16, [0.5] * 3):
        assert steer([1.0] * 8 + tail) == decided


def test_more_than_the_last_look_is_refused() -> None:
    with pytest.raises(ValueError, match="at most 24"):
        steer([0.5] * 25)


def _pairs(look: int, wins: int, losses: int) -> list[float]:
    return [1.0] * wins + [0.0] * losses + [0.5] * (look - wins - losses)


def _conditional_alpha(path: tuple[int, int, int]) -> float:
    """P(candidate call | no difference, ``path`` decisive pairs by each look), futility off.

    Written apart from ``steer_characteristics`` on purpose: a binomial walk over the
    decisive pairs only, so the proof does not lean on the calculator it checks.
    """
    states = {0: 1.0}
    seen = 0
    called = 0.0
    for decisive in path:
        while seen < decisive:
            grown: dict[int, float] = {}
            for lead, chance in states.items():
                grown[lead + 1] = grown.get(lead + 1, 0.0) + chance / 2
                grown[lead - 1] = grown.get(lead - 1, 0.0) + chance / 2
            states = grown
            seen += 1
        going = {}
        for lead, chance in states.items():
            if decisive and lead / math.sqrt(decisive) >= STEER_BOUND:
                called += chance
            else:
                going[lead] = chance
        states = going
    return called


def test_the_error_holds_under_every_tie_pattern() -> None:
    """The registered alpha, exactly: every way the ties can fall across the three looks."""
    worst = max(
        (_conditional_alpha((a, a + b, a + b + c)), (a, a + b, a + b + c))
        for a in range(9)
        for b in range(9)
        for c in range(9)
    )
    assert worst[0] <= STEER_ALPHA
    assert round(worst[0], 4) == REGISTERED_WORST_ALPHA
    assert worst[1] == (7, 10, 16)


@pytest.mark.parametrize(("ties", "share"), sorted(REGISTERED_CHARACTERISTICS))
def test_the_registered_operating_characteristics(ties: float, share: float) -> None:
    exact = steer_characteristics(share * (1 - ties), (1 - share) * (1 - ties))
    candidate, reference, too_close, pairs = REGISTERED_CHARACTERISTICS[(ties, share)]
    assert round(exact.candidate, 4) == candidate
    assert round(exact.reference, 4) == reference
    assert round(exact.too_close, 4) == too_close
    assert round(exact.expected_pairs, 1) == pairs
    total = exact.candidate + exact.reference + exact.too_close + exact.inconclusive
    assert total == pytest.approx(1.0, abs=1e-12)
    # The futility stop is non-binding: ignoring it can only add calls, never remove one.
    played_on = steer_characteristics(share * (1 - ties), (1 - share) * (1 - ties), futility=False)
    assert played_on.candidate >= exact.candidate
    if share == 0.5:
        assert played_on.candidate <= STEER_ALPHA


@pytest.mark.parametrize(("win", "loss"), [(0.5, 0.5), (0.45, 0.15), (0.2, 0.2)])
def test_the_calculator_models_the_function(win: float, loss: float) -> None:
    """A seeded simulation through ``steer`` lands within four errors of the exact value."""
    rng = np.random.default_rng(633)
    trials = 3000
    draws = rng.random((trials, STEER_LOOKS[-1]))
    pairs = np.where(draws < win, 1.0, np.where(draws < win + loss, 0.0, 0.5))
    calls = [steer(row.tolist()).call for row in pairs]
    exact = steer_characteristics(win, loss)
    for call, chance in ((CANDIDATE, exact.candidate), (TOO_CLOSE, exact.too_close)):
        seen = calls.count(call) / trials
        error = math.sqrt(max(chance * (1 - chance), 1e-6) / trials)
        assert abs(seen - chance) <= 4 * error, (call, seen, chance)


def test_steering_duel_plays_one_duels_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each look plays the next seeds, so the pairs are those of one 24-pair duel's prefix."""
    played: list[tuple[int, int]] = []
    outcome = [1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0] + [1.0] * 8 + [0.5] * 8

    def fake_duel(spec, candidate, reference, pairs, seed=0, workers=0, margin_weight=0.0):  # type: ignore[no-untyped-def]
        played.append((seed, pairs))
        block = tuple(outcome[seed - 100 : seed - 100 + pairs])
        return Duel(candidate, reference, 0.5, 0.0, block, identical=0)

    monkeypatch.setattr(duel_module, "duel", fake_duel)
    result = steering_duel({}, "a", "b", seed=100, workers=3)
    assert played == [(100, 8), (108, 8)]
    assert result.estimates == tuple(outcome[:16])
    assert result.steer == steer(outcome)
    assert result.steer.call == CANDIDATE and result.steer.pairs == 16
    assert result.games == 32


# The modules that write a stamp, an acceptance or a score. None may reach the rule.
STAMPED = (
    "premise.py",
    "measurement.py",
    "history.py",
    "agents/training/store.py",
    "agents/training/procedure.py",
    "agents/valuations.py",
    "tournament/valuing.py",
)

# Callers of the rule outside ``duel.py``. Empty today. A steering caller joins it only
# if it writes no stamp, and the test below then also holds that no stamped module imports
# it.
STEERING_CALLERS: frozenset[str] = frozenset()

RULE_NAMES = frozenset({
    "steer", "steering_duel", "steer_characteristics", "Steer", "SteeringDuel",
    "SteerCharacteristics", "STEER_BOUND", "STEER_FUTILITY", "STEER_LOOKS", "STEER_ALPHA",
})


def _references_the_rule(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in RULE_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in RULE_NAMES:
            return True
        if isinstance(node, ast.ImportFrom) and any(a.name in RULE_NAMES for a in node.names):
            return True
    return False


def _module_name(relative: str) -> str:
    return "chess_variant_sandbox." + relative.removesuffix(".py").replace("/", ".")


def test_the_rule_is_never_wired_to_a_stamped_verdict() -> None:
    texts = {
        path.relative_to(PACKAGE).as_posix(): path.read_text(encoding="utf-8")
        for path in PACKAGE.rglob("*.py")
    }
    for stamped in STAMPED:
        assert stamped in texts, f"{stamped} moved; point this test at where it went"
    # Parse only what could name the rule (every name in it starts so), and the stamped
    # modules, whose imports are read below.
    sources = {
        relative: ast.parse(text)
        for relative, text in texts.items()
        if relative in STAMPED or "steer" in text.lower()
    }

    callers = {
        relative
        for relative, tree in sources.items()
        if relative != "tournament/duel.py" and _references_the_rule(tree)
    }
    assert callers <= STEERING_CALLERS, (
        f"{sorted(callers - STEERING_CALLERS)} use the steering rule; it is for duels that "
        "are never stamped (reference/compute-economy.md §4.3)"
    )
    for stamped in STAMPED:
        assert stamped not in callers
        imported = {
            name
            for node in ast.walk(sources[stamped])
            if isinstance(node, ast.ImportFrom) and node.module
            for name in [node.module]
        } | {
            alias.name
            for node in ast.walk(sources[stamped])
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not imported & {_module_name(caller) for caller in STEERING_CALLERS}, stamped
