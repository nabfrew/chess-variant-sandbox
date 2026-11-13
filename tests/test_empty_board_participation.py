"""`docs/reference/empty-board-participation.py` (R-S14b): what its record says of the validator,
the engine and the scorer, held on the record's own three specs and on hand-built runs. No
stored run is read and no game is played."""

from __future__ import annotations

import private_paths
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.metrics.components import Components
from chess_variant_sandbox.metrics.index import score_components
from chess_variant_sandbox.metrics.scoring import WeightsError
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.schema import SpecError

SCRIPT = "docs/reference/empty-board-participation.py"
record = private_paths.script(SCRIPT, "empty_board_participation")
pytestmark = private_paths.needs(SCRIPT)


def test_the_rule_language_refuses_an_empty_board() -> None:
    """The record's first finding: nothing on the board is one refusal, and only that."""
    assert validate_spec(record.SPECS["muster"]) == [record.REFUSAL]
    with pytest.raises(SpecError):
        Engine(record.SPECS["muster"])


@pytest.mark.parametrize("name", ["muster with kings set", "muster with stones set"])
def test_two_pieces_on_the_board_is_a_valid_game(name: str) -> None:
    spec = record.SPECS[name]
    assert validate_spec(spec) == []
    assert record.opening(spec)[0] == 2


def test_the_stones_are_the_pieces_that_never_move() -> None:
    assert record.opening(record.SPECS["muster with stones set"]) == (2, 2)
    assert record.opening(record.SPECS["muster with kings set"]) == (2, 0)


def test_how_a_piece_enters_is_read_from_the_rules() -> None:
    assert record.enters(record.SPECS["muster"]) == "drops (placement phase)"
    assert record.enters({"rules": {"drops": {"gating": {"ranks": [0]}}}}) == "gating"
    assert record.enters({"rules": {"purse": {"market": ["N"]}}}) == "market"
    assert record.enters({"rules": {}}) == ""


def _run(raw: dict[str, float]) -> dict:
    return {"index": {"raw": raw, "diagnostics": {}}}


def _full_raw(weights) -> dict[str, float]:
    """A raw value at each curve's best end, and a ladder the roster reads."""
    raw = {}
    for name, curve in weights.curves.items():
        points = curve.points
        raw[name] = points.get("good_at", points.get("good_low", 0.0))
    raw["ladder_monotonicity"] = 1.0
    return raw


def test_a_run_without_the_dial_is_no_score_today_and_one_once_optional() -> None:
    """Option (b): `v13` gives a run that lacks the dial no total; with the dial optional the
    same run is scored on its family's other dials, and a run that has the dial is unmoved."""
    v13, optional, _ = record.weights_pair()
    raw = _full_raw(v13)
    game = {"hash": "0" * 12, "name": "hand-built", "runs": [_run(dict(raw)) for _ in range(4)]}
    whole = record.rescore(game, v13)
    assert whole is not None
    assert record.rescore(game, optional) == whole

    absent = [None, None, None, None]
    assert record.rescore(game, v13, absent) is None
    assert record.rescore(game, optional, absent) is not None


GATES_PASSED = {"search_advantage": 0.9, "stall_rate_worst_half": 0.0,
                "decisive_rate_worst_half": 0.9, "mean_plies_worst_half": 40.0,
                "quick_kill_rate_worst_half": 0.0}


def test_the_scorer_has_two_answers_for_a_run_without_the_dial() -> None:
    """The record's §1.3. A run that passes every gate raises, naming the dial and the pass
    that would never supply it; a run that fails a gate is refused for the gate, with *could
    not be measured* beside it and a placeholder total."""
    v13, _, _ = record.weights_pair()
    raw = _full_raw(v13) | GATES_PASSED
    del raw[record.DIAL]
    with pytest.raises(WeightsError, match=record.DIAL) as raised:
        score_components("0" * 12, "hand-built", Components(raw=raw, diagnostics={}), v13)
    assert "analyse" in str(raised.value)

    refused = raw | {"search_advantage": 0.5}
    result = score_components("0" * 12, "hand-built", Components(raw=refused, diagnostics={}), v13)
    assert not result.scored and not result.viable
    assert f"could not be measured on this corpus: {record.DIAL}" in result.violations
