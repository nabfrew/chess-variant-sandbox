"""The loop's operators for b6's and b7's push keys (Q1074bo, `docs/plan-push.md`).

`bar-the-push-back` toggles `rules.push.no_push_back`, `crush-on-the-walls` toggles
`rules.push.walls` between `"block"` and `"crush"` on a board with walls. Each declines
where nothing pushes, and where the validator would refuse the child.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.schema import push_rule

ROOT = Path(__file__).parents[1]


def _apply(name: str, spec: dict[str, Any], seed: int = 1) -> tuple[str | None, dict[str, Any]]:
    operator = next(op for op in OPERATORS if op.name == name)
    child = copy.deepcopy(spec)
    return operator.apply(child, random.Random(seed)), child


def _pushing(push: Any = True, walls: list[str] | None = None, **rule: Any) -> dict[str, Any]:
    """Chess whose rooks push, with ``rule`` as `rules.push` and walls at ``walls``."""
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    for piece in spec["pieces"]:
        if piece["id"] == "R":
            for move in piece["moves"]:
                move["push"] = copy.deepcopy(push)
    if rule:
        spec["rules"]["push"] = rule
    if walls:
        spec["rules"]["obstacles"] = {"squares": walls}
    assert validate_spec(spec) == []
    return spec


def test_both_decline_where_nothing_pushes() -> None:
    chess = load_spec(ROOT / "rulesets" / "chess.json")
    for name in ("bar-the-push-back", "crush-on-the-walls"):
        said, child = _apply(name, chess)
        assert said is None and child == chess


def test_bar_the_push_back_toggles_the_bar() -> None:
    spec = _pushing()
    said, child = _apply("bar-the-push-back", spec)
    assert said is not None and validate_spec(child) == []
    assert child["rules"]["push"] == {"no_push_back": True}
    said, back = _apply("bar-the-push-back", child)
    assert said is not None and "push" not in back["rules"]
    assert back == spec


def test_bar_the_push_back_declines_a_bar_nothing_can_meet() -> None:
    # Rooks that push only their own pieces: no push of an enemy and no long turn, so the
    # validator refuses the bar as never binding (Q879's reading, Q1074b6).
    spec = _pushing({"whose": "own"})
    trial = copy.deepcopy(spec)
    trial["rules"]["push"] = {"no_push_back": True}
    assert validate_spec(trial)
    assert _apply("bar-the-push-back", spec)[0] is None


def test_crush_on_the_walls_toggles_the_crush() -> None:
    spec = _pushing(walls=["d4", "e5"], chain="row")
    said, child = _apply("crush-on-the-walls", spec)
    assert said is not None and validate_spec(child) == []
    assert child["rules"]["push"] == {"chain": "row", "walls": "crush"}
    rule = push_rule(child)
    assert rule is not None and rule["walls"] == "crush"
    said, back = _apply("crush-on-the-walls", child)
    assert said is not None and back == spec


def test_crush_on_the_walls_declines_without_a_wall() -> None:
    assert _apply("crush-on-the-walls", _pushing())[0] is None


def test_push_instead_can_give_a_walled_game_its_crush() -> None:
    """A game with walls and no push reaches `walls: "crush"` in one edit."""
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    spec["rules"]["obstacles"] = {"squares": ["d4", "e5"]}
    assert validate_spec(spec) == []
    crushed = []
    for seed in range(200):
        said, child = _apply("push-instead", spec, seed)
        rule = push_rule(child) if said else None
        if rule is not None and rule["walls"] == "crush":
            assert validate_spec(child) == []
            crushed.append(seed)
    assert crushed
