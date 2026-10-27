"""The kind hues (`--kind-*`) are told apart wherever a kind of rule is named (Q1040,
`docs/plan-visual.md` §3.2, choice E1).

The floor, chosen before the colours were picked: **20 CIE76 ΔE** between any two kind hues,
and between each kind hue and `--muted`, `--ink`, `--bg` and `--panel`, in both themes. A kind
is drawn small (a map link's stroke, a 10px swatch, a chip's 3px inset edge, a 600-weight
`+`), and a small field shrinks the difference the eye sees, so the floor is about twice the
~10 that reads as *different* on a large patch. It is also the wall rows' floor already
(`tests/test_landing_wall.py`, Q1038), so one number holds both. `--muted` is the grey a kind
falls back to when the stylesheet does not know it (`mapKindColour`), so a kind near it reads
as *no kind*. The old `--kind-valuation` sat 18.7 / 12.7 from `--kind-hand`, and `--kind-turn`
7.2 / 4.1 from `--muted`.
"""

from __future__ import annotations

import re

import pytest
from test_style import _delta_e, _ratio, _rgb

from chess_variant_sandbox.spec import relate
from chess_variant_sandbox.web.page import style, tokens

FLOOR = 20.0
NEUTRALS = ("muted", "ink", "bg", "panel")
KINDS = tuple(f"kind-{kind}" for kind in relate.KINDS) + ("kind-sight",)
THEMES = {"light": tokens.LIGHT, "dark": {**tokens.LIGHT, **tokens.DARK}}

# The two hues Q1040 moved, as they stood before, for the check that the test would have failed.
BEFORE = {"light": {"kind-valuation": "#5a4fb5", "kind-turn": "#6f6558"},
          "dark": {"kind-valuation": "#9a93ef", "kind-turn": "#b0a48f"}}


def too_close(theme: dict[str, str]) -> list[tuple[str, str, float]]:
    """Every pair, kind against kind or kind against a neutral, nearer than the floor."""
    near = []
    for at, one in enumerate(KINDS):
        for two in KINDS[at + 1:] + NEUTRALS:
            apart = _delta_e(_rgb(theme[one]), _rgb(theme[two]))
            if apart < FLOOR:
                near.append((one, two, round(apart, 1)))
    return near


@pytest.mark.parametrize("name", THEMES)
def test_every_kind_hue_is_told_apart(name: str) -> None:
    assert too_close(THEMES[name]) == []


@pytest.mark.parametrize("name", THEMES)
def test_the_old_hues_would_fail(name: str) -> None:
    near = {(one, two) for one, two, _ in too_close({**THEMES[name], **BEFORE[name]})}
    assert near == {("kind-hand", "kind-valuation"), ("kind-turn", "muted")}


def test_every_kind_is_in_both_themes() -> None:
    for kind in KINDS:
        assert kind in tokens.LIGHT and kind in tokens.DARK, kind


def test_the_door_glyph_holds_text_contrast() -> None:
    """The one place a kind hue is text, the wall's `+` door, is the family's hue mixed toward
    the ink; that mix holds AA for every kind and for `--muted` on the panel and the page."""
    found = re.search(r"a\.wallDoor \{[^}]*?color: color-mix\(in srgb, var\(--family\) "
                      r"(\d+)%, var\(--ink\)\); \}", style.STYLE)
    assert found, "the door's colour is no longer the family mixed toward the ink"
    share = int(found.group(1)) / 100
    for theme in THEMES.values():
        ink = _rgb(theme["ink"])
        for family in KINDS + ("muted",):
            mixed = tuple(share * hue + (1 - share) * dark
                          for hue, dark in zip(_rgb(theme[family]), ink, strict=True))
            for ground in ("panel", "bg"):
                ratio = _ratio(mixed, _rgb(theme[ground]))  # type: ignore[arg-type]
                assert ratio >= 4.5, (family, ground, round(ratio, 2))
