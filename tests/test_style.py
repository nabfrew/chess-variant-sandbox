"""The stylesheet's palette, type and textures (plan-visual.md §3, task V1).

Spec-side tests of the stylesheet as text: every colour is a token named by its meaning, set
for both themes; the rules name no colour; the contrast floors of §3.2 hold in light and
dark, over the paper's worst fleck; the typefaces are the vendored files, each with its
system fallback; the paper is small inline SVG. Nothing here plays a game.
"""

from __future__ import annotations

import hashlib
import json
import re

from chess_variant_sandbox.web.page import shell, style, tokens

_HEX = re.compile(r"^#([0-9a-f]{6})$")
_RGBA = re.compile(r"^rgba\((\d+), (\d+), (\d+), ([.\d]+)\)$")


def _rgb(value: str) -> tuple[float, float, float]:
    found = _HEX.match(value)
    assert found, f"not a solid colour: {value}"
    digits = found.group(1)
    return tuple(int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _luminance(rgb: tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(one: tuple[float, float, float], two: tuple[float, float, float]) -> float:
    a, b = sorted((_luminance(one), _luminance(two)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _over(top: tuple[float, float, float], alpha: float,
          under: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(alpha * t + (1 - alpha) * u for t, u in zip(top, under, strict=True))  # type: ignore[return-value]


def _theme(dark: bool) -> dict[str, str]:
    return {**tokens.LIGHT, **tokens.DARK} if dark else dict(tokens.LIGHT)


def _grain(square: str) -> tuple[str, tuple[float, float, float], float, float, int, int, int]:
    return tokens.GRAIN_LT if square == "sq-lt" else tokens.GRAIN_DK


def _grained(dark: bool, square: str) -> tuple[float, float, float]:
    """A square under its grain's darkest (or, on the dark squares, lightest) streak: the
    tile's peak alpha at the theme's opacity, the worst a piece or a mark can sit on."""
    theme = _theme(dark)
    spec = _grain(square)
    alpha = float(theme["grain-" + square[3:] + "-a"]) * tokens.grain_most(spec)
    return _over(spec[1], alpha, _rgb(theme[square]))


def _lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(u: float) -> float:
        return u ** (1 / 3) if u > 0.008856 else 7.787 * u + 16 / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


# Machado, Oliveira and Fernandes (2009), severity 1, applied to linear RGB: the colours a
# reader with protanopia or deuteranopia sees.
_CVD: dict[str, tuple[tuple[float, float, float], ...]] = {
    "protanopia": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
    "deuteranopia": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
                     (-0.011820, 0.042940, 0.968881)),
}


def _seen(rgb: tuple[float, float, float],
          matrix: tuple[tuple[float, float, float], ...] | None) -> tuple[float, float, float]:
    """`rgb` as a reader with that colour blindness sees it (`None`: as it is)."""
    if matrix is None:
        return rgb

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def gamma(c: float) -> float:
        c = min(1.0, max(0.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    flat = [linear(c) for c in rgb]
    return tuple(gamma(sum(m * c for m, c in zip(row, flat, strict=True)))  # type: ignore[return-value]
                 for row in matrix)


def _delta_e(one: tuple[float, float, float], two: tuple[float, float, float]) -> float:
    """CIE76: the straight distance between two colours in CIELAB."""
    return float(sum((p - q) ** 2 for p, q in zip(_lab(one), _lab(two), strict=True)) ** 0.5)


class TestTheTokens:
    def test_the_rules_name_no_colour(self) -> None:
        """A colour outside the tokens is one dark mode and the direction cannot reach."""
        literal = re.compile(
            r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|"
            r"(?<![-\w])(?:white|black|red|green|blue|gray|grey|orange|yellow|purple|pink)(?![-\w])"
        )
        rules = re.sub(r"/\*.*?\*/", "", style.RULES, flags=re.S)
        assert literal.findall(rules) == []

    def test_every_token_the_rules_read_is_set(self) -> None:
        read = set(re.findall(r"var\(--([a-z0-9-]+)", style.RULES))
        set_ = set(tokens.LIGHT) | set(tokens.ALIASES) | set(tokens.SIZES)
        set_ |= {"f-head", "f-body", "f-mono"}
        # Not colours: where a board's walnut starts, set on its frame by `wood.js` and
        # read with a fallback.
        set_ |= {"wood-x", "wood-y"}
        # Nor these: where a square's name stands off the board (V4), the unit step outwards
        # from its edge that `drawCoords` sets on each, and how far it goes.
        set_ |= {"co-out", "dx", "dy"}
        # Nor these: a library card's own sheet of paper (Q684), set on each card by
        # `wood.js`'s `paperSheet` and read with a fallback.
        set_ |= {"sheet-x", "sheet-y", "sheet-s", "sheet-a", "sheet-t"}
        # Nor this: where a designer column sticks (F11, Q580), set on each by `design.js`'s
        # `fitSticky` and read with a fallback.
        set_ |= {"stick-top"}
        # Nor this: how many lines the status strip keeps (Q725), set by its own rules.
        set_ |= {"strip-lines"}
        # Nor this: a family's colour on the landing's wall of words (LP2), set on each row
        # by its family's class, which names a token (`landing.BLOCKS`).
        set_ |= {"family"}
        # Nor this: a quality bar's length on the landing's story (LP4), set by its own rule
        # to nothing and on each bar by `landing.js` from its number.
        set_ |= {"v"}
        assert read - set_ == set()

    def test_dark_sets_only_what_light_sets(self) -> None:
        assert set(tokens.DARK) - set(tokens.LIGHT) == set()

    def test_the_dark_block_reaches_a_viewer_who_picked_dark_and_one_whose_system_is(self) -> None:
        """The footer's switch wins over the system; the system wins over nothing."""
        dark = tokens.block(tokens.DARK).strip()
        assert ':root[data-theme="dark"] {' in style.STYLE
        assert ':root:not([data-theme="light"]) {' in style.STYLE
        assert style.STYLE.count("--bg: " + tokens.DARK["bg"] + ";") == 2
        assert dark.splitlines()[0] in style.STYLE

    def test_one_meaning_one_colour(self) -> None:
        """S4: the action colour is none of the board's highlights, in either theme, and
        the alarm is its own red."""
        for dark in (False, True):
            theme = _theme(dark)
            board = {theme[name] for name in ("hl-last", "hl-sel", "dot", "sq-lt", "sq-dk")}
            assert theme["act"] not in board, dark
            assert theme["alarm"] not in {theme["bad"], theme["loss"], theme["act"]}, dark

    def test_the_alarm_is_read_only_where_danger_is(self) -> None:
        """Test 3 of the meta-game: no red dots. `--alarm` is the check pill's and the
        resign button's, and the stylesheet reads it nowhere else."""
        uses = [line.strip() for line in style.RULES.splitlines() if "var(--alarm)" in line]
        assert uses and all(line.startswith((".pill.alarm", "button.danger")) for line in uses), uses


class TestContrast:
    """§3.2's floors, measured from the tokens. Text on a surface the paper covers is read
    over the paper's worst fleck as well as the bare surface (the fleck can never cover
    more than `paper_most`), so the texture cannot pull a ratio under its floor."""

    def test_the_floors_hold_in_both_themes(self) -> None:
        for dark in (False, True):
            theme = _theme(dark)
            fibre = tokens.PAPER_DARK if dark else tokens.PAPER_LIGHT
            most = tokens.paper_most(fibre)
            for fore, back, floor in tokens.FLOORS:
                ground = _rgb(theme[back])
                grounds = [ground]
                if back in tokens.PAPERED:
                    grounds.append(_over(fibre[0], most, ground))
                for under in grounds:
                    ratio = _ratio(_rgb(theme[fore]), under)
                    assert ratio >= floor, (dark, fore, back, round(ratio, 2), floor)

    def test_a_black_piece_reads_on_a_dark_square_in_both_themes(self) -> None:
        """S8: the board dims in dark and never inverts."""
        for dark in (False, True):
            ratio = _ratio((0.0, 0.0, 0.0), _rgb(_theme(dark)["sq-dk"]))
            assert ratio >= tokens.PIECE_FLOOR, (dark, round(ratio, 2))

    def test_the_board_does_not_invert(self) -> None:
        for name in ("sq-lt", "sq-dk"):
            light, dark = _luminance(_rgb(tokens.LIGHT[name])), _luminance(_rgb(tokens.DARK[name]))
            assert 0.6 * light <= dark <= light, name

    def test_the_fibre_is_faint(self) -> None:
        """The user asked for it subtler than Q629 drew it (whose flecks reached .26 and
        .11): a fleck covers at most a fifth of the ground in light, a tenth in dark."""
        assert tokens.paper_most(tokens.PAPER_LIGHT) <= 0.2
        assert tokens.paper_most(tokens.PAPER_DARK) <= 0.1

    def test_the_tile_is_small_inline_svg(self) -> None:
        for theme in (tokens.LIGHT, tokens.DARK):
            tile = theme["paper"]
            assert tile.startswith('url("data:image/svg+xml;utf8,<svg')
            assert len(tile.encode("utf-8")) < 1024
            assert "#" not in tile and '"' not in tile[5:-2]

    def test_a_piece_reads_over_the_worst_streak(self) -> None:
        """S8 over the wood: the darkest streak a square's grain can lay under a piece
        (the tile's peak alpha at the theme's opacity) leaves a black piece, and the black
        outline a white piece reads by, at the floor on both squares in both themes."""
        for dark in (False, True):
            for square in ("sq-lt", "sq-dk"):
                ratio = _ratio((0.0, 0.0, 0.0), _grained(dark, square))
                assert ratio >= tokens.PIECE_FLOOR, (dark, square, round(ratio, 2))

    def test_a_legal_move_dot_reads_over_the_worst_streak(self) -> None:
        for dark in (False, True):
            theme = _theme(dark)
            for square in ("sq-lt", "sq-dk"):
                under = _grained(dark, square)
                dot = _over(_rgb(theme["dot"]), dict(tokens.BOARD_MARKS)["dot"], under)
                ratio = _ratio(dot, under)
                assert ratio >= tokens.DOT_FLOOR, (dark, square, round(ratio, 2))

    def test_every_mark_stays_as_distinct_as_it_was(self) -> None:
        """The highlights are told apart by hue, which a luminance ratio cannot see, so this
        reads the CIELAB distance between a square with the mark and without it: over the
        worst streak at least `MARK_KEPT` of what it is over the bare square, and more than
        the streak's own distance from the bare square, so no streak reads as a mark."""
        for dark in (False, True):
            theme = _theme(dark)
            for square in ("sq-lt", "sq-dk"):
                bare = _rgb(theme[square])
                under = _grained(dark, square)
                streak = _delta_e(under, bare)
                for name, alpha in tokens.BOARD_MARKS:
                    was = _delta_e(_over(_rgb(theme[name]), alpha, bare), bare)
                    now = _delta_e(_over(_rgb(theme[name]), alpha, under), under)
                    assert now >= tokens.MARK_KEPT * was, (dark, square, name, now, was)
                    assert now > streak, (dark, square, name, now, streak)

    def test_a_coming_arrow_reads_over_the_worst_streak(self) -> None:
        """Q660: a wild piece's coming step is a graphic, 3 : 1 at the opacity it is drawn
        at, on both woods, bare and under their worst streak, in both themes."""
        alpha = dict(tokens.BOARD_MARKS)["coming"]
        for dark in (False, True):
            theme = _theme(dark)
            for square in ("sq-lt", "sq-dk"):
                for under in (_rgb(theme[square]), _grained(dark, square)):
                    arrow = _over(_rgb(theme["coming"]), alpha, under)
                    ratio = _ratio(arrow, under)
                    assert ratio >= tokens.DOT_FLOOR, (dark, square, round(ratio, 2))

    def test_a_held_move_reads_on_both_woods(self) -> None:
        """Q733: a move picked and not sent is drawn in `--held` edged in `--held-halo`, so
        the dashes read 3 : 1 on both squares, bare and under their worst streak, in both
        themes, and the halo stands off the dashes at text contrast."""
        for dark in (False, True):
            theme = _theme(dark)
            held, halo = _rgb(theme["held"]), _rgb(theme["held-halo"])
            assert _ratio(held, halo) >= 4.5, dark
            for square in ("sq-lt", "sq-dk"):
                for under in (_rgb(theme[square]), _grained(dark, square)):
                    ratio = _ratio(held, under)
                    assert ratio >= tokens.DOT_FLOOR, (dark, square, round(ratio, 2))

    def test_a_held_move_is_not_the_last_move(self) -> None:
        """A sent move (the last move's amber fill) and an unsent one (dark dashes) are two
        kinds of mark, far apart in colour in both themes, and neither is the action colour."""
        for dark in (False, True):
            theme = _theme(dark)
            assert theme["held"] != theme["act"], dark
            apart = _delta_e(_rgb(theme["held"]), _rgb(theme["hl-last"]))
            assert apart >= 40, (dark, round(apart, 1))

    def test_a_standoff_is_not_a_held_move(self) -> None:
        """Q940: a held move, a void run's link and a barred pair's link are three marks in
        three colours, each pair far apart in both themes, as seen and under simulated
        protanopia and deuteranopia (Machado et al. 2009, severity 1), so neither a reader nor
        the look check takes a held move for a standoff, or one standoff for the other."""
        names = ("held", "standoff-run", "standoff-bar")
        for dark in (False, True):
            theme = _theme(dark)
            for one, two in ((a, b) for i, a in enumerate(names) for b in names[i + 1:]):
                for seen in (None, *_CVD.values()):
                    apart = _delta_e(_seen(_rgb(theme[one]), seen), _seen(_rgb(theme[two]), seen))
                    assert apart >= 30, (dark, one, two, seen is not None, round(apart, 1))

    def test_each_mark_kind_paints_its_own_token(self) -> None:
        """Q940: the look check knows a mark kind by the token it paints in, so no two of
        `MARK_KINDS`' rows share one, but for the wild piece's two coming steps (one arrow
        drawn fainter) and a piece's sight (schema 30, Q999c), which tints what it sees in the
        colour of the effect it would put there, so its row's key shares the freeze's token;
        and none paints in a token the board draws something that is not a kind in (the held
        move's `--held`)."""
        from page_text import WRAPPED

        table = WRAPPED[WRAPPED.index("const MARK_KINDS = {"):]
        table = table[:table.index("\n};\n")]
        rows = dict(re.findall(r"^  (\w+): \{ colour: 'var\((--[\w-]+)\)'", table, re.M))
        assert {"voidRun", "barred", "check", "goal", "coming"} <= rows.keys()
        shared = {"coming", "comingThen", "seen"}
        seen: dict[str, str] = {}
        for kind, token in rows.items():
            assert token not in ("--held", "--held-halo"), kind
            if kind in shared:
                continue
            assert token not in seen, (kind, seen.get(token), token)
            seen[token] = kind
        for token in set(rows.values()):
            assert token[2:] in tokens.LIGHT, token

    def test_a_coming_arrow_is_none_of_the_other_marks(self) -> None:
        """One colour, one meaning: the arrow is told apart from every other thing the
        board draws on or between squares -- the dot, the last move, the selection, the
        check, the wild piece's own ochre and trail, the rings, seams and claims -- by a
        clear CIELAB distance in both themes."""
        others = ("dot", "hl-last", "hl-sel", "alarm", "p2", "goal", "zone", "seam",
                  "claim0", "claim1")
        for dark in (False, True):
            theme = _theme(dark)
            for name in others:
                apart = _delta_e(_rgb(theme["coming"]), _rgb(theme[name]))
                assert apart >= 20, (dark, name, round(apart, 1))

    def test_a_walk_reads_on_both_woods(self) -> None:
        """Q852: a wild piece's walk is drawn in its hue on a casing of `--walk-edge`; the
        casing reads 3 : 1 on both squares, bare and under their worst streak, in both
        themes (each hue holds 3 : 1 on the casing, `FLOORS`)."""
        for dark in (False, True):
            theme = _theme(dark)
            edge = _rgb(theme["walk-edge"])
            for square in ("sq-lt", "sq-dk"):
                for under in (_rgb(theme[square]), _grained(dark, square)):
                    ratio = _ratio(edge, under)
                    assert ratio >= tokens.DOT_FLOOR, (dark, square, round(ratio, 2))

    def test_the_walks_are_told_apart(self) -> None:
        """Four walks on one board (*wolves*: two patrols, each mirrored) are four hues, none
        of them the coming arrow's violet or the wild piece's own ochre."""
        hues = [f"walk-{n}" for n in range(1, 5)]
        for dark in (False, True):
            theme = _theme(dark)
            for at, name in enumerate(hues):
                for other in [*hues[at + 1:], "coming", "p2"]:
                    apart = _delta_e(_rgb(theme[name]), _rgb(theme[other]))
                    assert apart >= 20, (dark, name, other, round(apart, 1))

    def test_rgba_tokens_are_well_formed(self) -> None:
        for theme in (tokens.LIGHT, tokens.DARK):
            for name, value in theme.items():
                if value.startswith("rgba"):
                    found = _RGBA.match(value)
                    assert found and 0 <= float(found.group(4)) <= 1, name


def _blocks(css: str) -> list[tuple[str, str]]:
    """Every innermost `selector { declarations }` of a stylesheet, comments gone. A rule
    inside an `@media` comes back with its own selector, not the query's."""
    bare = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", bare)]


class TestTheControls:
    """V3 (plan-visual.md §3.5): the button kinds, one height scale, the focus ring, the
    coarse pointer's targets, segmented switches, chips, the fields, the (i), the site's
    forms. The stylesheet as text; the look is the look check's."""

    CONTROL = re.compile(r"\bbutton\b|\bselect\b|\binput\b|\.chip\b|\.crumb\b|\.linkRow\b")

    def test_a_coarse_pointer_gets_44_px_targets(self) -> None:
        """§3.5: 40 px at a desk, 44 px under a coarse pointer, for a button, a field and a
        compact control alike; a chip drawn smaller reaches 44 through `--hit`."""
        assert tokens.SIZES["ctl-h"] == "40px"
        for name in ("ctl-h", "ctl-h-sm", "hit"):
            assert int(tokens.COARSE[name].removesuffix("px")) >= tokens.TARGET, name
        assert set(tokens.COARSE) <= set(tokens.SIZES)
        coarse = re.search(r"@media \(pointer: coarse\) \{\s*:root \{([^}]*)\}", style.STYLE)
        assert coarse and "--ctl-h: 44px;" in coarse.group(1) and "--hit: 44px;" in coarse.group(1)

    def test_one_height_scale(self) -> None:
        """Every control's height is a token (or zero, for one drawn smaller that reaches
        `--hit`), never a pixel value of its own. `--ctl-h-lg` is *Send your move*'s (Q733)."""
        allowed = re.compile(r"^(?:0|var\(--(?:ctl-h|ctl-h-sm|ctl-h-lg|chip-h|hit)\)|"
                             r"max\(var\(--hit\), calc\(var\(--(?:ctl-h|ctl-h-sm)\) - 2px\)\))$")
        seen = 0
        for selector, body in _blocks(style.RULES):
            # The control is what the rule styles: the last compound of one of its
            # selectors (`.chip .dot` styles the dot in a chip, not the chip).
            if not any(self.CONTROL.search(part.split()[-1]) for part in selector.split(",")):
                continue
            for prop, value in re.findall(r"(?<![-\w])(min-height|height)\s*:\s*([^;]+);", body):
                if selector.startswith("textarea") or "info" in selector or ".menu" in selector:
                    continue
                seen += 1
                assert allowed.match(value.strip()), (selector, prop, value)
        assert seen >= 10
        info = dict(_blocks(style.RULES))["button.info"]
        assert "min-height: 0" in info

    def test_the_four_button_kinds(self) -> None:
        rules = dict(_blocks(style.RULES))
        primary = rules["button.primary"]
        assert "var(--act)" in primary and "var(--act-ink)" in primary
        quiet = rules["button.quiet"]
        assert "transparent" in quiet and "color: var(--act)" in quiet
        danger = rules["button.danger, button.grave"]
        assert "border-color: var(--alarm)" in danger and "color: var(--alarm)" in danger
        base = rules["button"]
        assert "min-height: var(--ctl-h)" in base
        assert "opacity: .45" in rules["button:disabled"]

    def test_the_focus_ring_is_the_actions_and_never_removed(self) -> None:
        rings = [body for selector, body in _blocks(style.RULES) if ":focus-visible" in selector]
        assert any("outline: 2px solid var(--focus)" in body for body in rings)
        assert "outline: none" not in style.STYLE and "outline: 0" not in style.STYLE
        assert tokens.LIGHT["focus"] == tokens.LIGHT["act"]
        assert tokens.DARK["focus"] == tokens.DARK["act"]

    def test_the_small_controls_reach_a_target(self) -> None:
        """The (i) keeps its 15 px circle and a chip its pill; each reaches `--hit`."""
        rules = dict(_blocks(style.RULES))
        reach = rules["button.info::after, button.chip::after"]
        assert "max(100%, var(--hit))" in reach and "position: absolute" in reach
        assert "position: relative" in rules["button.info, .chip"]
        assert "min-height: var(--chip-h)" in rules[".chip"]

    def test_segmented_switches_fill_the_chosen_one_in_ink(self) -> None:
        """§3.5: *list · map*, *moves · rules · pieces* and the theme switch; the chosen
        one in ink, not the accent."""
        rules = dict(_blocks(style.RULES))
        chosen = rules['.viewSwitch button.on, .tabs button.on, .themePick button[aria-pressed="true"]']
        assert "background: var(--ink)" in chosen and "color: var(--panel)" in chosen
        assert "var(--act)" not in chosen

    def test_a_chosen_option_is_the_actions_tint(self) -> None:
        rules = dict(_blocks(style.RULES))
        on = next(body for selector, body in rules.items() if selector.startswith(".chip.on,"))
        assert "background: var(--act-tint)" in on and "border-color: var(--act)" in on

    def test_a_fields_edge_is_a_control_boundary(self) -> None:
        """WCAG 1.4.11: a field's edge at 3 : 1 (`FLOORS`), where `--line` is faint."""
        rules = dict(_blocks(style.RULES))
        field = next(body for selector, body in rules.items() if selector.startswith("select, input"))
        assert "border-color: var(--edge)" in field and "min-height: var(--ctl-h)" in field
        assert ("edge", "panel", 3.0) in tokens.FLOORS and ("edge", "bg", 3.0) in tokens.FLOORS

    def test_text_on_the_actions_tint_reads(self) -> None:
        """A chosen chip's ink over the action's tint, a hovered quiet button's green over
        the sunk tint, each laid on the panel and the ground, at AA in both themes, over the
        paper's worst fleck too."""
        for dark in (False, True):
            theme = _theme(dark)
            fibre = tokens.PAPER_DARK if dark else tokens.PAPER_LIGHT
            for fore, tint_name, floor in tokens.TINTED:
                tint = _RGBA.match(theme[tint_name])
                assert tint, tint_name
                colour = tuple(int(tint.group(i)) / 255 for i in (1, 2, 3))
                alpha = float(tint.group(4))
                for ground in ("panel", "bg"):
                    bare = _rgb(theme[ground])
                    for under in (bare, _over(fibre[0], tokens.paper_most(fibre), bare)):
                        tinted = _over(colour, alpha, under)  # type: ignore[arg-type]
                        ratio = _ratio(_rgb(theme[fore]), tinted)
                        assert ratio >= floor, (dark, fore, tint_name, ground, round(ratio, 2))

    def test_the_frames_bar_is_quiet_and_resign_the_one_danger(self) -> None:
        """§3.3: take back and flip as quiet buttons, resign as the one danger button."""
        from page_text import BOARD

        for name in ("undo", "flip", "stopPlay", "draw", "cancel"):
            assert re.search(rf'<button id="{name}" class="quiet"', BOARD), name
        assert '<button id="resign" class="danger"' in BOARD
        assert BOARD.count('class="danger"') == 1

    def test_the_sites_forms_take_the_controls(self) -> None:
        """Sign-in and the invitation's join: the fields on a sheet of their own, the
        site's own rules naming no colour, a revoke the danger kind."""
        from chess_variant_sandbox.site import pages

        for page in (pages.signin(), pages.join("code")):
            assert '<div class="panel form">' in page
            assert page.index('<div class="panel form">') < page.index('id="handle"')
            assert page.index('id="go"') < page.index("</div>", page.index('id="go"'))
        own = re.sub(r"/\*.*?\*/", "", pages._OWN, flags=re.S)
        assert re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(", own) == []
        assert "button.grave" not in own
        assert '<button id="revoke-all" class="danger">' in pages.admin("someone")


class TestThePlayFrame:
    """V4 (plan-visual.md §3.3): the frame's look. Its order and markup are
    `tests/test_web.py::TestThePlayFrame`'s; the names' placement `tests/test_viz.py`'s."""

    def test_check_is_a_filled_pill_whose_words_read(self) -> None:
        rules = dict(_blocks(style.RULES))
        pill = rules[".pill.alarm"]
        assert "background: var(--alarm)" in pill and "color: var(--alarm-ink)" in pill
        assert ("alarm-ink", "alarm", 4.5) in tokens.FLOORS
        for dark in (False, True):
            theme = _theme(dark)
            assert _ratio(_rgb(theme["alarm-ink"]), _rgb(theme["alarm"])) >= 4.5, dark

    def test_the_strip_is_ink_with_the_side_to_move_as_a_dot(self) -> None:
        rules = dict(_blocks(style.RULES))
        assert "color: var(--ink)" in rules[".frame-status"]
        assert rules[".seat-dot.s0"].strip() == "background: var(--p0);"
        assert rules[".seat-dot.s1"].strip() == "background: var(--p1);"

    def test_the_strip_keeps_one_height_whatever_it_says(self) -> None:
        """Q725: on a phone the board jumped up and down as *thinking* or *in check* took
        a line more or less. The strip's height is fixed, in lines of its own 22 px, never
        its content's; no state sets a height, a padding or a border width of its own, and
        none hides it. The look check's `S1 steady` drives the states in a browser."""
        blocks = _blocks(style.RULES)
        rules = dict(blocks)
        strip = rules[".frame-status"]
        assert "height: calc(var(--strip-lines) * 22px + 14px)" in strip
        assert "line-height: 22px" in strip and "min-height" not in strip
        for held in ("box-sizing: border-box", "overflow: hidden", "padding: 6px 0",
                     "border: 1px solid transparent", "--strip-lines: 2"):
            assert held in strip, held
        # The other states may only change how many lines are kept, or colour the box in.
        assert rules[".frame-status.narrow, .frame-status.between"].strip() == "--strip-lines: 3;"
        assert rules[".frame-status.between.narrow"].strip() == "--strip-lines: 4;"
        ended = rules[".frame-status.ended"]
        assert "padding: 6px 12px" in ended and "border-color: var(--ink)" in ended
        assert "height" not in ended and "border:" not in ended and "border-width" not in ended
        for selector, body in blocks:
            if ".frame-status" not in selector or selector == ".frame-status":
                continue
            assert "display: none" not in body and "visibility" not in body, selector
            if "#status" not in selector and ".frame-note" not in selector:
                assert not re.search(r"(^|[\s;])(min-|max-)?height\s*:", body), selector
        # Words longer than the room end in an ellipsis rather than growing the box.
        words = rules[".frame-status #status"]
        assert "-webkit-line-clamp: var(--strip-lines)" in words and "overflow: hidden" in words

    def test_the_names_stand_outside_in_the_numbers_face(self) -> None:
        rules = dict(_blocks(style.RULES))
        name = rules[".coords .co"]
        assert "var(--f-mono)" in name and "color: var(--muted)" in name
        assert "var(--co-out)" in name and "var(--dx)" in name and "var(--dy)" in name
        # The panel makes room for them, and the frame no longer clips them.
        assert "padding-left" in rules[".board-panel.coords-out"]
        assert "overflow: visible" in rules[".board-panel .board-frame"]

    def test_the_banner_is_a_result_never_an_alarm(self) -> None:
        """A result is `--win` or `--loss` (§3.2), never the alarm red, and both read on
        the banner's panel over its worst fleck (`FLOORS`)."""
        banner = "\n".join(body for selector, body in _blocks(style.RULES)
                           if selector.startswith(".frame-ended"))
        assert "var(--win)" in banner and "var(--loss)" in banner
        assert "var(--alarm)" not in banner
        assert ("win", "panel", 4.5) in tokens.FLOORS and ("loss", "panel", 4.5) in tokens.FLOORS

    def test_on_a_phone_the_bar_is_icons_with_a_word_under_each(self) -> None:
        bare = re.sub(r"/\*.*?\*/", "", style.RULES, flags=re.S)
        assert ".frame-controls .ico { display: none; }" in bare
        phone = bare[bare.index("@media (max-width: 480px)", bare.index(".frame-controls .ico")):]
        phone = phone[:phone.index("\n  }\n")]
        assert ".frame-controls .ico { display: block; }" in phone
        bar = phone[phone.index(".frame-controls > button {"):]
        bar = bar[:bar.index("}")]
        assert "flex-direction: column" in bar and "min-height: var(--ctl-h)" in bar

    def test_every_icon_is_one_stroke_on_one_grid(self) -> None:
        """Original drawings, one 24-unit grid, the button's own colour, no fill; every
        icon the bar names is drawn."""
        from page_text import BOARD

        icons = BOARD[BOARD.index("const FRAME_ICONS = {"):]
        icons = icons[:icons.index("};")]
        names = set(re.findall(r"^  (\w+): '", icons, flags=re.M))
        used = set(re.findall(r"barWords\('(\w+)'", BOARD))
        assert used and used <= names
        assert "fill" not in icons.replace('fill="none"', "")
        maker = BOARD[BOARD.index("function frameIcon(name) {"):]
        maker = maker[:maker.index("\n}\n")]
        assert 'viewBox="0 0 24 24"' in maker and 'stroke="currentColor"' in maker
        assert 'fill="none"' in maker and 'stroke-width="1.8"' in maker


class TestTheWood:
    """The board's wood as tokens (Q588, plan-visual.md §9's third answer). The drawing,
    each square's own offset, is `tests/test_wood.py`'s."""

    TILES = ("grain-lt", "grain-dk", "walnut")

    def test_the_tiles_are_small_inline_svg_set_once_for_both_themes(self) -> None:
        """The board is an object (S8): one set of tiles, and only how strongly they show
        moves with the theme. Under about 2 KB together, no image file."""
        total = 0
        for name in self.TILES:
            tile = tokens.LIGHT[name]
            assert tile.startswith('url("data:image/svg+xml;utf8,<svg') and "feTurbulence" in tile
            assert "#" not in tile and '"' not in tile[5:-2]
            assert name not in tokens.DARK
            total += len(tile.encode("utf-8"))
        assert total < 2048, total

    def test_light_and_dark_squares_are_two_woods(self) -> None:
        """The second refinement: the two squares' grains differ in frequency (scale), in
        tone, in seed, and in how large a window of each a board draws from."""
        light, dark = tokens.GRAIN_LT, tokens.GRAIN_DK
        lf = [float(v) for v in light[0].split()]
        df = [float(v) for v in dark[0].split()]
        assert lf != df and df[1] < lf[1] and df[0] > lf[0]
        assert light[1] != dark[1] and light[5] != dark[5]
        from page_text import view

        tile = re.search(r"const WOOD_TILE = \{ lt: ([\d.]+), dk: ([\d.]+) \}", view("wood"))
        assert tile and float(tile.group(1)) != float(tile.group(2))

    def test_subtler_than_the_mock(self) -> None:
        """The first refinement: Q629 laid its grain over every square at .32; every
        square's grain here is at most .25 of its tile, in both themes, and fainter in dark."""
        for name in ("grain-lt-a", "grain-dk-a"):
            light, dark = float(tokens.LIGHT[name]), float(tokens.DARK[name])
            assert 0 < dark < light <= 0.25, name

    def test_the_frame_is_walnut_and_the_grain_reads_the_theme(self) -> None:
        rules = style.RULES
        frame = re.search(r"\.board-frame \{[^}]*\}", rules)
        assert frame and "var(--walnut) var(--wood-x, 0) var(--wood-y, 0)" in frame.group(0)
        assert ".grain.lt { opacity: var(--grain-lt-a); }" in rules
        assert ".grain.dk { opacity: var(--grain-dk-a); }" in rules

    def test_the_marks_the_contrast_test_reads_are_the_ones_drawn(self) -> None:
        """`BOARD_MARKS`' opacities are the board's own, so the floors read what is drawn."""
        from page_text import BOARD, WRAPPED

        marks = dict(tokens.BOARD_MARKS)
        assert f"mark(x, y, cell, 'var(--hl-last)', {marks['hl-last']})" in BOARD
        assert f"mark(x, y, cell, 'var(--hl-sel)', {marks['hl-sel']})" in BOARD
        assert re.search(r"fill=\"\$\{piece \? 'none' : 'var\(--dot\)'\}\"\s+stroke=\"var\(--dot\)\" "
                         rf"stroke-width=\"\$\{{cell \* 0\.05\}}\" opacity=\"{marks['dot']}\"", BOARD)
        assert f'fill="${{kind.colour}}" opacity="{marks["alarm"]}"' in WRAPPED
        assert "check: { colour: 'var(--alarm)', shape: 'glow'" in WRAPPED
        assert f"const COMING_OPACITY = {marks['coming']};" in WRAPPED
        assert "coming: { colour: 'var(--coming)', shape: 'arrow'" in WRAPPED


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _toned(rgb: tuple[float, float, float], tone: float) -> tuple[float, float, float]:
    """The panel colour as a card of this tone draws it: `oklch(from var(--panel) calc(l +
    t * tone-l) calc(c - t * tone-c) h)`, out of gamut clamped as the browser maps it."""
    r, g, b = (_linear(c) for c in rgb)
    lms = [(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
           (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
           (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)]
    lightness = 0.2104542553 * lms[0] + 0.7936177850 * lms[1] - 0.0040720468 * lms[2]
    a = 1.9779984951 * lms[0] - 2.4285922050 * lms[1] + 0.4505937099 * lms[2]
    b2 = 0.0259040371 * lms[0] + 0.7827717662 * lms[1] - 0.8086757660 * lms[2]
    chroma = (a * a + b2 * b2) ** 0.5
    lightness += tone * tokens.SHEET["tone-l"]
    toned = max(0.0, chroma - tone * tokens.SHEET["tone-c"])
    a, b2 = (a * toned / chroma, b2 * toned / chroma) if chroma else (0.0, 0.0)
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b2) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b2) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b2) ** 3
    return (_gamma(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_),
            _gamma(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_),
            _gamma(-0.0040960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_))


class TestTheSheets:
    """Each library card's own sheet of paper (Q684): offset, size, turn and tone hashed from
    the game (`wood.js`'s `paperSheet`, tested for determinism in `tests/test_wood.py`),
    within `tokens.SHEET`'s ranges. Here: the ranges are the stylesheet's and the script's,
    and the floors hold at both ends of them."""

    def _card(self) -> str:
        found = re.search(r"\.panel\.card \{.*?\}\s*\.panel\.card::before \{.*?\}", style.RULES, re.S)
        assert found
        return found.group(0)

    def test_the_ranges_are_small(self) -> None:
        """Subtle by construction: a few percent of size, a few degrees, under a
        hundredth of lightness."""
        sheet = tokens.SHEET
        assert sheet["offset"] == 1.0 and sheet["tone"] == 1.0
        assert 0 < sheet["scale"] <= 0.06 and 0 < sheet["angle"] <= 3
        assert 0 < sheet["tone-l"] <= 0.015 and 0 < sheet["tone-c"] <= 0.005

    def test_the_stylesheet_and_the_script_read_the_same_ranges(self) -> None:
        from page_text import view

        card = self._card()
        assert f"calc(l + var(--sheet-t, 0) * {tokens.SHEET['tone-l']:g})".replace("0.", ".") in card
        assert f"calc(c - var(--sheet-t, 0) * {tokens.SHEET['tone-c']:g})".replace("0.", ".") in card
        script = re.search(r"const SHEET = \{ tile: (\d+), scale: ([\d.]+), angle: ([\d.]+) \};",
                           view("wood"))
        assert script
        assert float(script.group(2)) == tokens.SHEET["scale"]
        assert float(script.group(3)) == tokens.SHEET["angle"]
        # The tile it cuts from is the panels' own, at the size the panels draw it.
        assert f"width='{script.group(1)}'" in tokens.LIGHT["paper"]
        assert f"calc({script.group(1)}px * var(--sheet-s, 1))" in card
        assert f"/ {script.group(1)}px, var(--panel)" in style.RULES

    def test_the_card_keeps_its_box_and_its_plain_fallback(self) -> None:
        """No reflow: the sheet is a clipped layer under the text, and a browser without
        relative colours draws the plain panel. No motion, no image file."""
        card = self._card()
        assert "background: var(--panel);\n" in card and "position: absolute" in card
        assert "z-index: -1" in card and "isolation: isolate" in card and "overflow: clip" in card
        for moving in ("transition", "animation", "url(", "padding", "margin", "width", "height"):
            assert moving not in card, moving

    def test_a_turned_sheet_still_covers_its_card(self) -> None:
        """The layer reaches `inset` past each edge; a card turned by the largest angle needs
        half its other side's sine of that past each edge, which the inset covers for any
        card up to 1200 px on its long side."""
        import math

        inset = re.search(r"inset: -(\d+)px", self._card())
        assert inset
        turn = math.radians(tokens.SHEET["angle"])
        for width, height in ((258, 1200), (1200, 258), (343, 900), (1200, 1200)):
            need_x = (width * (math.cos(turn) - 1) + height * math.sin(turn)) / 2
            need_y = (height * (math.cos(turn) - 1) + width * math.sin(turn)) / 2
            assert max(need_x, need_y) <= int(inset.group(1)), (width, height)

    def test_the_floors_hold_at_both_ends_of_the_tone_over_the_worst_fleck(self) -> None:
        """Every floor whose ground is the panel, and the chosen chip's tint over it, read on
        the panel toned to either end and under the fibre's strongest fleck, in both themes.
        Scale and turn only move where a fleck lies: the tile's peak alpha is the same."""
        for dark in (False, True):
            theme = _theme(dark)
            fibre = tokens.PAPER_DARK if dark else tokens.PAPER_LIGHT
            most = tokens.paper_most(fibre)
            for tone in (-tokens.SHEET["tone"], 0.0, tokens.SHEET["tone"]):
                sheet = _toned(_rgb(theme["panel"]), tone)
                for under in (sheet, _over(fibre[0], most, sheet)):
                    for fore, back, floor in tokens.FLOORS:
                        if back != "panel":
                            continue
                        ratio = _ratio(_rgb(theme[fore]), under)
                        assert ratio >= floor, (dark, tone, fore, round(ratio, 3), floor)
                    for fore, tint_name, floor in tokens.TINTED:
                        tint = _RGBA.match(theme[tint_name])
                        assert tint
                        colour = tuple(int(tint.group(i)) / 255 for i in (1, 2, 3))
                        tinted = _over(colour, float(tint.group(4)), under)  # type: ignore[arg-type]
                        ratio = _ratio(_rgb(theme[fore]), tinted)
                        assert ratio >= floor, (dark, tone, fore, tint_name, round(ratio, 3))

    def test_the_tone_is_a_touch(self) -> None:
        """Visible beside a neighbour only when looked for: the two ends of the tone are at
        most four units of CIELAB apart, so either end is within about one just-noticeable
        step (2.3) of the middle, the size of the ±1.5% of lightness the user's example
        named; and at least half a unit, or it would not be there at all."""
        for dark in (False, True):
            panel = _rgb(_theme(dark)["panel"])
            apart = _delta_e(_toned(panel, -1.0), _toned(panel, 1.0))
            assert 0.5 <= apart <= 4.0, (dark, round(apart, 2))


class TestTheTypefaces:
    def test_every_face_is_a_vendored_file_with_its_digest(self) -> None:
        for name, entry in shell.fonts().items():
            body = (shell.FONT_DIR / name).read_bytes()
            assert body[:4] == b"wOF2", name
            assert len(body) == entry["bytes"], name
            assert hashlib.sha256(body).hexdigest() == entry["sha256"], name
            assert entry["licence"] == "OFL-1.1"
            assert "SIL Open Font License" in (shell.FONT_DIR / entry["licenceFile"]).read_text(
                encoding="utf-8"
            )

    def test_the_faces_fit_the_budget(self) -> None:
        """§3.1: at most four files and about 120 KB per direction."""
        files = shell.fonts()
        assert len(files) <= 4
        assert sum(entry["bytes"] for entry in files.values()) <= 120_000

    def test_nothing_else_lies_in_the_font_directory(self) -> None:
        manifest = json.loads((shell.FONT_DIR / "manifest.json").read_text(encoding="utf-8"))
        named = {entry["file"] for entry in manifest["files"]}
        named |= {entry["licenceFile"] for entry in manifest["files"]}
        on_disk = {path.name for path in shell.FONT_DIR.iterdir()} - {"manifest.json"}
        assert on_disk == named

    def test_each_family_is_declared_with_its_system_fallback(self) -> None:
        """A page whose fonts are blocked is the page as it was."""
        for role, fallback in (("headings", "serif"), ("text", "sans-serif"),
                               ("numbers and moves", "monospace")):
            assert shell.stack(role).endswith(fallback), role
        assert "--f-head: 'Fraunces'" in style.STYLE
        assert style.STYLE.count("@font-face") == len(shell.fonts())
        assert "font-display: swap" in style.STYLE

    def test_no_face_is_fetched_from_another_origin(self) -> None:
        assert "googleapis" not in style.STYLE and "gstatic" not in style.STYLE
        bare = re.sub(r'url\("data:[^"]*"\)', "", style.STYLE)
        assert re.findall(r"url\((?!/static/fonts/)", bare) == []
