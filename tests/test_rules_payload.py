"""CP5 (PY14): every page that draws a game carries the whole of `rules_payload`.

The play board's ruleset block, the offline replay's payload and the still preview each
used to pick their own subset of the compiled rules, and drifted (the replay carried no
blast, speed, rest or purse; the still no blast). Each now takes `rules_payload` whole;
this holds all three to every key and value, over rulesets that between them put every
rule the payload draws in force (`test_the_rulesets_put_every_rule_in_force`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_portal_board import _game

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.viz.html import render_position
from chess_variant_sandbox.viz.rules_payload import rules_payload
from chess_variant_sandbox.web.card_rules import preview
from chess_variant_sandbox.web.play import Session

ROOT = Path(__file__).resolve().parent.parent

# Between them, every rule `rules_payload` has a key for is in force in one of these.
RULESETS = (
    "rulesets/chess.json",
    "rulesets/atomic.json",  # explosion
    "rulesets/dobutsu.json",  # reach, drops
    "archive/kyoto.json",  # drops either face
    "archive/jeson-mor.json",  # a goal left rather than reached
    "archive/chess-oriented-pawns.json",  # facing
    "archive/divide.json",  # walls
    "archive/hex-glinski-hexagon.json",  # a hexagon's absent corners
    "archive/the-hydra.json",  # pounce
    "rulesets/torus.json",  # wrap
    "rulesets/rps-v2.json",  # territory
    "archive/carousel.json",  # portals
    "archive/the-shepherds-duck.json",  # duck
    "archive/breather.json",  # rest
    "tests/_purse_market.json",  # purse
    "tests/_sight_speed.json",  # speed
    "tests/_gating_seirawan.json",  # gating (Q1059a)
)


def _spec(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _replay_payload(spec: dict[str, Any], engine: Engine, tmp_path: Path) -> dict[str, Any]:
    page = render_position(spec, engine.initial_state(), tmp_path / "replay.html", icons=False)
    line = next(text for text in page.read_text(encoding="utf-8").splitlines()
                if text.startswith("const DATA = "))
    return dict(json.loads(line[len("const DATA = "):].rstrip(";")))


@pytest.mark.parametrize("path", RULESETS)
def test_every_page_carries_every_compiled_rule(path: str, tmp_path: Path) -> None:
    spec = _spec(path)
    engine = Engine(spec)
    expected = _jsonable(rules_payload(engine))
    session = Session(id="s1", game=_game(spec), seats=("human", "human"), seed=0,
                      state=engine.initial_state())
    pages = {
        "play": _jsonable(session.snapshot()["ruleset"]),
        "replay": _replay_payload(spec, engine, tmp_path),
        "still": _jsonable(preview(engine, engine.initial_state(), spec)),
    }
    for page, payload in pages.items():
        missing = sorted(set(expected) - set(payload))
        assert not missing, f"{page} lacks {missing}"
        differ = sorted(key for key in expected if payload[key] != expected[key])
        assert not differ, f"{page} disagrees on {differ}"


def test_the_rulesets_put_every_rule_in_force() -> None:
    """Each key that is empty without its rule says something for one of `RULESETS`, so the
    test above compares a drawn rule and not only a ``None``."""
    payloads = [rules_payload(Engine(_spec(path))) for path in RULESETS]
    idle = [key for key in payloads[0] if not any(_says(payload[key]) for payload in payloads)]
    assert not idle, f"no ruleset puts these in force: {idle}"


def _says(value: Any) -> bool:
    """Whether a value says anything: a list of all ``False``, zero or ``None`` says nothing."""
    if isinstance(value, list):
        return any(_says(item) for item in value)
    if isinstance(value, dict):
        return any(_says(item) for item in value.values())
    return bool(value)
