"""R-S4b's reading of the flag (`docs/reference/strength-check-all-games.md` §2.3) on fixtures.

`docs/reference/strength-check-all-games.py` plays the check; its `clauses` is the whole rule
and is a pure function of c, f1, whether Fairy-Stockfish plays, and the score's file. These
cases hold it to the registration: each clause at its threshold, the any-of, and the words.
"""

from __future__ import annotations

from typing import Any

import private_paths
import pytest

SCRIPT = "docs/reference/strength-check-all-games.py"
check = private_paths.script(SCRIPT, "strength_check_all_games")
pytestmark = private_paths.needs(SCRIPT)


def _score(unreadable: list[bool], measured: bool = True) -> dict[str, Any]:
    """A `canonical.json` cut to the fields clause C reads."""
    return {
        "runs": [{"seed": 1000 * i, "unreadable": ["x"] if u else []}
                 for i, u in enumerate(unreadable)],
        "verdict": {"measured": measured},
    }


READABLE = _score([False, False, False, False])


def test_the_registered_thresholds() -> None:
    assert (check.PAIRS, check.SEED, check.OPENING_PLIES, check.MULTIPLE) == (24, 0, 8, 4)
    assert (check.FLAT_TOP, check.FAIRY_FLOOR, check.UNREADABLE_SHARE) == (0.40, 0.30, 0.5)
    assert (check.DEEP, check.DRAWN_OUT, check.THIN) == (0.10, 0.80, 12)


@pytest.mark.parametrize(("c", "a"), [(0.40, True), (0.3999, False), (0.55, True), (0.0, False)])
def test_clause_a_is_c_at_or_above_point_four(c: float, a: bool) -> None:
    found = check.clauses(c, None, False, READABLE)
    assert found["A"] is a
    assert found["word"] == (check.DEPENDS if a else check.HELD_OWN)


@pytest.mark.parametrize(("f1", "b"), [(0.30, False), (0.2999, True), (0.0, True), (None, False)])
def test_clause_b_is_f1_below_point_three(f1: float | None, b: bool) -> None:
    found = check.clauses(0.2, f1, True, READABLE)
    assert found["B"] is b
    if f1 is None:
        assert found["word"] == check.NOT_READ  # Fairy plays, its duel is not in
    else:
        assert found["word"] == (check.DEPENDS if b else check.HELD)


@pytest.mark.parametrize(
    ("score", "c_holds"),
    [
        (_score([True, True, False, False]), True),  # half
        (_score([True, False, False, False]), False),  # a quarter
        (_score([True, True, True, False]), True),
        (_score([False, False, False, False], measured=False), True),
        (_score([True], measured=True), True),  # a one-run score: one of one
        (None, False),
    ],
)
def test_clause_c_reads_the_score_file(score: dict[str, Any] | None, c_holds: bool) -> None:
    assert check.clauses(0.2, None, False, score)["C"] is c_holds


def test_clause_c_alone_flags_a_game_nobody_has_played() -> None:
    assert check.clauses(None, None, False, _score([True, True]))["word"] == check.DEPENDS
    assert check.clauses(None, None, False, READABLE)["word"] == check.NOT_READ


def test_any_clause_flags_and_the_reading_names_which() -> None:
    found = check.clauses(0.45, 0.1, True, _score([True, True, False, False]))
    assert found["holding"] == ["A", "B", "C"]
    assert found["word"] == check.DEPENDS
    assert check.clauses(0.2, 0.5, True, READABLE)["word"] == check.HELD
    assert check.clauses(0.2, None, False, READABLE)["word"] == check.HELD_OWN


def test_elo_and_its_edges() -> None:
    assert check.elo(0.5, 48) == {"elo": 0.0, "beyond": False}
    assert check.elo(0.40, 48)["elo"] == pytest.approx(-70.4, abs=0.1)
    top = check.elo(1.0, 48)
    assert top["beyond"] and top["elo"] == pytest.approx(791.0, abs=1.0)
    assert check.elo(0.0, 48)["elo"] == pytest.approx(-791.0, abs=1.0)


def test_beside_decides_nothing() -> None:
    duel = {"duel": "self", "score": 0.08, "drawn": 0.85, "openings": 11}
    fairy = {"duel": "fairy1", "score": 0.5, "drawn": 0.0, "openings": 20}
    found = check.beside(duel, fairy, None)
    assert found == {"deep": True, "drawn_out": True, "thin": ["self"]}
