"""The pick hues (`--pick-1` to `--pick-3`) are told apart on the score history page (SH3,
`docs/plan-score-history.md` §2.4).

The floor is Q1040's, fixed before any hue was tried (`test_kind_hues.py`): **20 CIE76 ΔE**
between every two picks, and between each pick and `--muted`, `--ink`, `--bg` and `--panel`,
in both themes; and 3 : 1 against the panel (`tokens.FLOORS`, read over the paper's worst
fleck by `test_style.py::TestContrast`). A pick is a 2px line and an 8px marker among a
hundred faint traces, so it must not read as the ink, the grey or the ground. `--trace`, a
game not picked, is the ink at a low opacity and never one of them.
"""

from __future__ import annotations

import re

import pytest
from test_style import _delta_e, _ratio, _rgb

from chess_variant_sandbox.web.page import style, tokens

FLOOR = 20.0
NEUTRALS = ("muted", "ink", "bg", "panel")
PICKS = ("pick-1", "pick-2", "pick-3")
THEMES = {"light": tokens.LIGHT, "dark": {**tokens.LIGHT, **tokens.DARK}}


def too_close(theme: dict[str, str]) -> list[tuple[str, str, float]]:
    """Every pair, pick against pick or pick against a neutral, nearer than the floor."""
    near = []
    for at, one in enumerate(PICKS):
        for two in PICKS[at + 1:] + NEUTRALS:
            apart = _delta_e(_rgb(theme[one]), _rgb(theme[two]))
            if apart < FLOOR:
                near.append((one, two, round(apart, 1)))
    return near


@pytest.mark.parametrize("name", THEMES)
def test_every_pick_is_told_apart(name: str) -> None:
    assert too_close(THEMES[name]) == []


@pytest.mark.parametrize("name", THEMES)
def test_every_pick_holds_three_to_one_on_the_panel(name: str) -> None:
    theme = THEMES[name]
    for pick in PICKS:
        assert _ratio(_rgb(theme[pick]), _rgb(theme["panel"])) >= 3.0, pick
        assert (pick, "panel", 3.0) in tokens.FLOORS


def test_the_floor_would_catch_a_twin() -> None:
    """A pick a shade off another, or off the grey, is caught: the check is not vacuous."""
    for name, theme in THEMES.items():
        twin = {**theme, "pick-3": theme["pick-1"], "pick-2": theme["muted"]}
        near = {(one, two) for one, two, _ in too_close(twin)}
        assert ("pick-1", "pick-3") in near and ("pick-2", "muted") in near, name


def test_the_four_tokens_are_in_both_themes() -> None:
    for token in (*PICKS, "trace"):
        assert token in tokens.LIGHT and token in tokens.DARK, token


def test_the_trace_is_the_ink_faint() -> None:
    """`--trace` is the ink's own colour at a low opacity, in each theme."""
    for name, theme in THEMES.items():
        found = re.fullmatch(r"rgba\((\d+), (\d+), (\d+), (\.\d+)\)", theme["trace"])
        assert found, name
        ink = tuple(round(part * 255) for part in _rgb(theme["ink"]))
        assert tuple(int(part) for part in found.groups()[:3]) == ink, name
        assert float(found.group(4)) <= 0.25, name


def test_nothing_but_the_history_wears_them() -> None:
    """One colour, one meaning: the stylesheet reads the picks only in the history's rules,
    and no page script but `history.js` names them."""
    from chess_variant_sandbox.web.page import SCRIPTS

    for line in style.RULES.splitlines():
        if "var(--pick-" in line or "var(--trace)" in line:
            assert "history" in line, line
    for route, path in SCRIPTS.items():
        if path.name == "history.js":
            continue
        text = path.read_text(encoding="utf-8")
        assert "--pick-" not in text and "--trace" not in text, route
