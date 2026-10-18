"""The portal pairs' palette (Q838b, `docs/archive/plan-portal-look.md` §2.1): colour means the pair,
so the four pair tokens must stay apart for the readers colour is hardest for.

The check is `docs/reference/visual/q838a-palette.py`'s, as a test: CIEDE2000 distances between
the pairs' hues as seen with normal vision and as simulated for protanopia, deuteranopia and
tritanopia (Machado, Oliveira and Fernandes 2009, severity 1.0, in linear RGB). Every two pairs
stay at least `FLOOR` apart under each, and pairs 1 and 2, the two every two-pair game uses,
at least `FIRST_TWO`. The number on the glyph is the second cue; this is the first.
"""

from __future__ import annotations

import itertools
import math

import pytest

from chess_variant_sandbox.web.page import tokens

FLOOR = 15.0
FIRST_TWO = 35.0
PAIRS = 4

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}
Colour = tuple[float, float, float]


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _unlin(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _rgb(hexa: str) -> Colour:
    h = hexa.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return r, g, b


def _seen(colour: Colour, vision: str) -> Colour:
    if vision == "normal":
        return colour
    m = MACHADO[vision]
    li = [_lin(c) for c in colour]
    r, g, b = (_unlin(sum(m[row][k] * li[k] for k in range(3))) for row in range(3))
    return r, g, b


def _lab(colour: Colour) -> Colour:
    r, g, b = (_lin(c) for c in colour)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _de2000(a: Colour, b: Colour) -> float:
    l1, a1, b1 = a
    l2, a2, b2 = b
    cm = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cm ** 7 / (cm ** 7 + 25 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dl, dc = l2 - l1, c2p - c1p
    dh = h2p - h1p
    if c1p * c2p == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    big_dh = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh) / 2)
    lm, cmp_ = (l1 + l2) / 2, (c1p + c2p) / 2
    hm = h1p + h2p
    if c1p * c2p != 0:
        hm = (h1p + h2p + 360) / 2 if abs(h1p - h2p) > 180 else (h1p + h2p) / 2
    t = (1 - 0.17 * math.cos(math.radians(hm - 30)) + 0.24 * math.cos(math.radians(2 * hm))
         + 0.32 * math.cos(math.radians(3 * hm + 6)) - 0.20 * math.cos(math.radians(4 * hm - 63)))
    sl = 1 + 0.015 * (lm - 50) ** 2 / math.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + 0.045 * cmp_, 1 + 0.015 * cmp_ * t
    rt = (-2 * math.sqrt(cmp_ ** 7 / (cmp_ ** 7 + 25 ** 7))
          * math.sin(math.radians(60 * math.exp(-(((hm - 275) / 25) ** 2)))))
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (big_dh / sh) ** 2
                     + rt * (dc / sc) * (big_dh / sh))


def _distance(x: str, y: str, vision: str) -> float:
    return _de2000(_lab(_seen(_rgb(x), vision)), _lab(_seen(_rgb(y), vision)))


def _hues() -> dict[int, str]:
    return {n: tokens.LIGHT[f"pair-{n}"] for n in range(1, PAIRS + 1)}


class TestThePalette:
    def test_the_measure_is_the_published_one(self) -> None:
        """A known value: black against white is 100 under normal vision."""
        assert _distance("#000000", "#ffffff", "normal") == pytest.approx(100, abs=0.01)

    @pytest.mark.parametrize("vision", ["normal", "protan", "deutan", "tritan"])
    def test_every_two_pairs_stay_apart(self, vision: str) -> None:
        hues = _hues()
        for a, b in itertools.combinations(hues, 2):
            assert _distance(hues[a], hues[b], vision) >= FLOOR, (vision, a, b)

    @pytest.mark.parametrize("vision", ["normal", "protan", "deutan", "tritan"])
    def test_the_first_two_pairs_are_far_apart(self, vision: str) -> None:
        hues = _hues()
        assert _distance(hues[1], hues[2], vision) >= FIRST_TWO, vision

    def test_the_pairs_are_one_value_for_both_themes(self) -> None:
        """The board does not invert (S8): a pair's colour is the same in the dark."""
        for n in range(1, PAIRS + 1):
            assert f"pair-{n}" not in tokens.DARK
