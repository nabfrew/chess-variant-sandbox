"""The palette: every colour the pages draw, named by what it means (plan-visual.md §3.2).

The club room (direction B, the user's choice of 2026-10-02, §9): warm paper, baize green
for what can be done, a walnut-and-maple board, with Q629's paper fibre in the ground and
the panels. Each token is set once for light and once for dark; nothing else in the
stylesheet names a colour (`tests/test_style.py` holds that), so a page that wants a
colour asks for a meaning, and a meaning added here reaches both themes, both servers and
every page at once.

A theme is chosen three ways, in this order: the viewer's own pick from the footer's switch
(`data-theme` on `<html>`, set by `web/static/shell.js` before the first paint), else the
system's `prefers-color-scheme`, else light. `TOKENS` is the CSS that does that: the light
block on `:root`, the dark block under the media query unless the viewer picked light,
and the dark block again for a viewer who picked dark.

**One colour, one meaning** (§2 S4). `--act` is *you can do this* and never appears on the
board; the board's highlights (`--hl-last`, `--hl-sel`, `--dot`) are their own values;
`--alarm` is *danger, now* and is used for nothing that is not. The board is an object,
not part of the page, so its squares dim a little in dark and never invert (S8): a black
piece stays readable on a dark square in both themes.
"""

from __future__ import annotations


def _uri(svg: str) -> str:
    """An SVG as a CSS `url()`, quoted for the inline stylesheet."""
    escaped = svg.replace('"', "'").replace("%", "%25").replace("#", "%23")
    return "url(\"data:image/svg+xml;utf8," + escaped + "\")"


def noise(freq: str, rgb: tuple[float, float, float], gain: float, bias: float, size: int,
          seed: int, octaves: int = 3) -> str:
    """A tile of fractal noise turned into one colour at varying alpha.

    The alpha at a point is `gain * noise + bias`, the noise being in [0, 1], so the most it
    can ever cover is `gain + bias`, clamped to 1. An even `freq` draws paper; a stretched
    one (`"0.006 0.26"`: slow along x, fast along y) draws a straight grain along x.
    """
    r, g, b = rgb
    return _uri(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'>"
        "<filter id='n' x='0' y='0' width='100%' height='100%'>"
        f"<feTurbulence type='fractalNoise' baseFrequency='{freq}' numOctaves='{octaves}' "
        f"seed='{seed}' stitchTiles='stitch'/>"
        f"<feColorMatrix values='0 0 0 0 {r} 0 0 0 0 {g} 0 0 0 0 {b} {gain} 0 0 0 {bias}'/>"
        "</filter><rect width='100%' height='100%' filter='url(#n)'/></svg>"
    )


def paper(rgb: tuple[float, float, float], gain: float, bias: float, seed: int = 3) -> str:
    """A tile of paper fibre.

    The most a fleck can ever cover is `gain + bias` (`paper_most`): the contrast test reads
    text over the panel blended with the fleck's colour at that alpha, the worst a reader can
    meet. Q629's tile (`docs/reference/visual-clubroom-twist-2.py`), made subtler as the user
    asked: the flecks show over less of the tile and at about two thirds the strength.
    """
    return noise("0.55", rgb, gain, bias, 180, seed)


# (colour as 0..1 rgb, gain, bias) of each theme's fibre: dark flecks on light paper, light
# flecks on dark. At twice the dark strength the fibre read as felt noise (Q629), so dark
# is close to invisible on purpose.
PAPER_LIGHT = ((0.35, 0.25, 0.15), 0.3, -0.15)
PAPER_DARK = ((1.0, 0.95, 0.85), 0.14, -0.065)


# How far one library card's sheet may differ from the next (Q684): `wood.js`'s
# `paperSheet` draws each from the game's hash and `style.py` lays it. `offset` is anywhere
# in the tile (a fraction of it); `scale` the tile's size within 6% either way; `angle` a
# turn within 3 degrees; `tone` one unit either way, which moves the card's panel colour by
# `tone-l` of oklch lightness and `tone-c` of chroma against it (lighter and greyer, or
# darker and creamier, like two sheets from different reams). Tone is the one that can
# cost contrast, and the floors leave little: at 1.5% of lightness the 3 : 1 pairs (a
# note's rule in light, a field's edge in dark) fell under, so it is .6%, which
# `tests/test_style.py` reads at both ends over the worst fleck. Scale and angle move where
# a fleck lies and never how strong it is.
SHEET: dict[str, float] = {
    "offset": 1.0, "scale": 0.06, "angle": 3.0, "tone": 1.0, "tone-l": 0.006, "tone-c": 0.002,
}


def paper_most(spec: tuple[tuple[float, float, float], float, float]) -> float:
    """The largest alpha a fleck of this fibre can reach."""
    _, gain, bias = spec
    return max(0.0, min(1.0, gain + bias))


# The board's wood (Q588: Q629's *wood and paper*, chosen with four refinements, plan-visual
# §9's third answer). Each row is `noise`'s arguments: (frequency, colour, gain, bias, tile
# px, seed, octaves). Light and dark squares are cut from two different woods (the second
# refinement): the light squares a fine, close maple grain; the dark squares a coarser,
# wider and browner one, at a little more than half the light grain's frequency along y and
# about twice its frequency along x, so its streaks are broader and shorter. Each tile's
# alpha reaches 1 at its darkest streak (`gain + bias > 1`), so how far a streak can darken a
# square is the theme's `grain-lt-a` / `grain-dk-a` and nothing else: that opacity is the
# worst case the contrast tests read a piece or a highlight over.
GRAIN_LT = ("0.006 0.26", (0.30, 0.17, 0.07), 2.4, -1.0, 256, 11, 4)
GRAIN_DK = ("0.011 0.15", (0.96, 0.84, 0.66), 2.4, -1.0, 256, 29, 3)
# The frame's walnut (the fourth refinement): Q629's walnut tile as drawn, over `--frame`.
WALNUT = ("0.01 0.5", (0.12, 0.06, 0.02), 3.0, -1.1, 240, 5, 4)


def grain_most(spec: tuple[str, tuple[float, float, float], float, float, int, int, int]) -> float:
    """The largest alpha a streak of this tile can reach, before the theme's opacity."""
    return max(0.0, min(1.0, spec[2] + spec[3]))


LIGHT: dict[str, str] = {
    # --- the page ---
    "bg": "#f6f1e7", "panel": "#fffdf8", "ink": "#2b2420", "muted": "#5f544b",
    "line": "#e1d7c7",
    # The edge of a field you type in or choose from (an input, a select, a text box): a
    # control's boundary, so 3 : 1 against the ground and the panel (WCAG 1.4.11), where
    # `--line` is a rule between things and deliberately faint (V3).
    "edge": "#827362",
    # What a line, a dot or a name on the map fades to while something else is held up. A
    # colour rather than an opacity, so a faded hairline is still exactly a hairline.
    "faint": "#d3c8b6",
    # A faint tint for a hovered row or a code span, and the striping of long lists.
    "sunk": "rgba(60, 40, 20, .05)", "stripe": "rgba(60, 40, 20, .045)",
    # The move list's current row: neutral, never the last move's yellow (§3.2).
    "now": "rgba(43, 36, 32, .1)",
    "shadow": "0 1px 2px rgba(60, 40, 20, .08), 0 4px 14px rgba(60, 40, 20, .06)",
    "shadow-pop": "0 8px 26px rgba(40, 25, 10, .18)",
    "shadow-toast": "0 6px 24px rgba(40, 25, 10, .3)",
    "shadow-bar": "0 -5px 18px rgba(40, 25, 10, .22)",
    "paper": paper(*PAPER_LIGHT),
    # --- you can do this: primary buttons, links, the active tab, a chosen option ---
    "act": "#1d6a48", "act-ink": "#ffffff", "act-tint": "rgba(29, 106, 72, .12)",
    # The keyboard's focus ring (V3): solid, so it clears WCAG's 3 : 1 for a focus
    # indicator against the ground and the panel (`FLOORS`); the action's own green.
    "focus": "#1d6a48",
    # --- how to read the number beside it: the exhibit's and the premise's rule, the
    # verdict's edge, a royal piece's tag. Neither good nor bad.
    "note": "#a0782f",
    # --- a measurement drawn as a length or a shade: score bars, the map's score fill ---
    "meter": "#b8873a",
    # --- a result or a verdict as text ---
    "good": "#2a6d42", "bad": "#944434",
    "win": "#1a6e40", "loss": "#8c3b2e",
    # --- danger, now: a royal piece attacked, the *in check* pill, resign's edge ---
    # A shade darker than V1 set it (#b8221a), so a danger button reads at AA on the page
    # itself, over the paper's worst fleck, and not only on a panel (V3).
    "alarm": "#b4201a",
    # The words on the alarm: the *in check* pill is filled red (V4), its words this.
    "alarm-ink": "#ffffff",
    # --- the board: maple and walnut, the board everybody knows (D3: one quiet board) ---
    "sq-lt": "#f0d9b5", "sq-dk": "#b58863", "frame": "#6b4a32", "wall": "#5b4636",
    # The wood (Q588). The tiles are the same in both themes (the board is an object, S8);
    # `web/static/page/wood.js` reads `--grain-lt` and `--grain-dk` and lays each square
    # its own window of them, and the frame takes `--walnut` (`style.py`). The opacities
    # are how strongly a grain shows: Q629's mock multiplied its grain in at .32 over
    # every square; the user asked for a bit more subtle, so .24 on the light squares and
    # .2 on the dark ones, which already carry more colour.
    "grain-lt": noise(*GRAIN_LT), "grain-dk": noise(*GRAIN_DK), "walnut": noise(*WALNUT),
    "grain-lt-a": ".24", "grain-dk-a": ".2",
    # What just happened, what you are holding, where it may go. Solid colours: the board
    # draws each at its own opacity over the square.
    "hl-last": "#f2c94c", "hl-sel": "#2e8b65", "dot": "#2b2420",
    # A move picked between two people and not sent yet (Q733): dashed outlines and a
    # dashed arrow in a near-black pine, each edged in a cream halo so it reads on both
    # woods. Not the last move's amber, never a fill: a sent move and an unsent one are
    # two kinds of mark, not two shades of one.
    "held": "#0c2e20", "held-halo": "#fff8ea",
    # Two pieces of a game in rounds in a standoff (schema 26, Q918c; Q940): a link between
    # their squares, cased in `held-halo` as the held move's arrow is, but never the held
    # move's pine, so a held move and a standoff are two marks. An umber for one void round
    # between them (a second in a row bars the pair) and a cobalt once the pair is barred:
    # apart from the pine and from each other in lightness and on the blue-yellow axis, so
    # under every simulated colour blindness too; each holds 4.5 : 1 on the casing, where
    # the run's *1* is written (`FLOORS`).
    "standoff-run": "#8a4a00", "standoff-bar": "#1d3f8a",
    # Schema 27 (Q942b): a swap of a game in rounds that met (`swap: "collision"`), a burst
    # between the two squares in a wine of its own, and a rested piece's pause sign
    # (`rest`), a slate disc with two bars in the casing's colour: a pause is a neutral fact,
    # not a danger. Both are drawn on `held-halo`, at 4.5 : 1 (`FLOORS`).
    "clash": "#8e1b5c", "rest": "#474d57",
    # Schema 28 (Q943b): the order disc of a round between two speeds, numbered and ringed in a
    # deep indigo of its own on `held-halo` (`FLOORS`): apart from the held move's pine.
    "order": "#2a2350",
    # Schema 27 (Q935c): the move announced in a spy game's announced round, a solid arrow
    # in a deep teal, cased in `held-halo`: apart from the held move's pine in hue and
    # weight, and from the standoffs' umber and cobalt and the clash's wine (`FLOORS`).
    "announced": "#00606e",
    # Generation 42 (N4c, `docs/plan-npc.md` §4.6): the duck's flight on the last ply, a solid
    # arrow in a deep olive cased in `held-halo`: apart from the announced teal, the held
    # pine, the standoffs' umber and cobalt and the clash's wine (`FLOORS`). It rings the
    # duck's square in duck mode too.
    "duck": "#4a5200",
    # Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.6): a piece under a line of sight's effect
    # wears a badge on `held-halo`, and a piece's sight tints the squares it sees in the same
    # colour. A deep ice blue for frozen (a snowflake), a rust for disarmed (a broken blade),
    # a deep green for shielded (a shield): apart in hue and in shape, and from the rest's
    # slate and the order's indigo; each holds 4.5 : 1 on the casing (`FLOORS`).
    "frozen": "#0d4f80", "disarmed": "#8a3510", "shielded": "#1b6236",
    # Q1007d (`docs/plan-schema-31.md` §3.9): a commanded piece's badge (a cross and strings)
    # and a command sight's tint, a violet of its own: apart from the three effects' blue,
    # rust and green, and lighter and redder than the wild piece's coming purple; 4.5 : 1 on
    # the casing (`FLOORS`).
    "commanded": "#6b2d8f",
    # The board dimmed behind a choice drawn over it (Q765: what a promoting piece
    # becomes), so the card is the one lit thing in the reader's eye line.
    "scrim": "rgba(30, 20, 10, .6)",
    # A capture's flash and a shield's glow, the board's two animations.
    "fx-fire": "#f59e42", "fx-flare": "#e8852e", "fx-spark": "#f5c242", "fx-shield": "#5bc0de",
    # --- the armies, for the lettered disc a piece with no drawing is ---
    "p0": "#ffffff", "p0-ink": "#16171a", "p1": "#26282c", "p1-ink": "#f4f2ee",
    # A third owner (schema 9): a wild piece is neither player's, so it is drawn in neither
    # player's colour. Ochre because the two armies are a light/dark pair whatever the
    # theme, and a third piece has to read as *off* that axis rather than as a shade
    # between them. It never takes an army's icon, so a wild piece is always the disc.
    "p2": "#c08a3e", "p2-ink": "#16171a",
    # A portal end (schema 19, `docs/archive/plan-portals.md` §4, D3, D9; drawn since Q838b as the
    # vortex of `docs/archive/plan-portal-look.md` §2.3). No pair's hue can hold 3 : 1 against both
    # woods and still be its hue, so the well is ringed in `portal-edge`, a near-black that
    # does, and falls to it in the eye. `portal-stone` is the ground of the designer's end
    # being put down. The board does not invert (S8), so dark sets only the stone.
    "portal-edge": "#1e140c", "portal-stone": "#b3aca1",
    # Q838b (`docs/archive/plan-portal-look.md` §2.1): colour means the pair. Each pair's well, at
    # both its ends, the trail through it and the dots where it lets a piece out, in one
    # hue: violet, yellow, blue, orange, in that order because every two-pair game uses the
    # first two, the most distinct pair (violet against yellow, the blue-yellow axis that
    # survives red-green colour blindness, and far apart in lightness too). Each holds 3 : 1
    # against `portal-edge`, which rings the well (`FLOORS`), and every two stay apart under
    # simulated colour blindness (`tests/test_portal_palette.py`). The board does not invert,
    # so dark sets none. The designer draws a pair in these too (Q838d): its chip, its card's
    # icon, the end being put down and the ways on its card.
    "pair-1": "#7b4fd6", "pair-2": "#efc12b", "pair-3": "#22b8cf", "pair-4": "#d55e00",
    # A fold portal's pool (schema 22, Q843d, `docs/archive/plan-portals-2.md` §5.3): the light in its
    # middle, near white, through the group's hue to its rim, so a fold is bright in the middle
    # where the vortex is dark, told apart in a grey picture too. On the board, which does not
    # invert, so dark sets none.
    "fold-light": "#fffbe8",
    # Where a wild piece steps after the next move (Q660): a dotted arrow, in a deep violet
    # that is none of the board's other marks (not the wild piece's ochre, which its trail
    # already draws in) and reads as a graphic, 3 : 1, on both woods in both themes.
    "coming": "#3d1f6e",
    # Its name in the legend line, which sits on the page rather than the wood: the same
    # violet in light, a lighter one on a dark page (AA in `FLOORS`).
    "coming-ink": "#3d1f6e",
    # Every wild piece's walk, in order (Q852, `viz/walks.py`): a path of arrows per walk,
    # each walk in its own hue, cased in `walk-edge`, a near-black that holds 3 : 1 on both
    # woods, as a portal's well is ringed, since no four hues can hold it on both woods and
    # still be four hues. Each hue holds 3 : 1 against the casing (`FLOORS`). Blue, orange,
    # magenta, gold: none is the coming arrow's violet or the wild piece's ochre. The board
    # does not invert, so dark sets the same values (the page's views read a token set in
    # both themes).
    "walk-1": "#648fff", "walk-2": "#fe6100", "walk-3": "#dc267f", "walk-4": "#ffb000",
    "walk-edge": "#1e140c",
    # --- a rule's ground on the board ---
    "claim0": "#4f8a5b", "claim1": "#7a5ea8",
    # A square the rules end the game on when it is reached. Teal because every other
    # thing the board draws on a square has a colour already.
    "goal": "#1f8f86",
    # A seam a piece may cross one way and not the other, and the squares a piece promotes
    # on. Blue and rose because neither is any army's colour under either theme.
    "seam": "#3466ac", "zone": "#b0468a",
    # --- the map: one colour per kind of rule, for its links. The names are
    # `spec/relate.py`'s `KINDS`, read as `var(--kind-<kind>, var(--muted))`, so a kind
    # added to the schema draws in grey rather than not at all.
    "kind-board": "#3466ac", "kind-pieces": "#b0468a", "kind-setup": "#1f8f86",
    "kind-capture": "#c0503a", "kind-ending": "#4f8a5b", "kind-hand": "#7a5ea8",
    "kind-wild": "#c08a3e", "kind-turn": "#6f6558", "kind-clock": "#7d7b2a",
    "kind-valuation": "#5a4fb5",
    # One colour per community on the map, in the payload's order: eight categorical hues
    # validated for colour-blind separation against the panel a dot sits on, in both
    # themes (`docs/archive/plan-graph-map.md` S3). A ninth keeps the neutral dot.
    "community-1": "#2a78d6", "community-2": "#eb6834", "community-3": "#1baf7a",
    "community-4": "#eda100", "community-5": "#e87ba4", "community-6": "#008300",
    "community-7": "#4a3aa7", "community-8": "#e34948",
    # A member's mark (`marks.py`, R2 and R3 of archive/plan-return.md §6.3): the eight hues it is
    # drawn in, the club room's inks: claret, cobalt, violet, brass, walnut, petrol, olive
    # and plum. Each is a graphic on the paper, so 3 : 1 on the ground and the panel over
    # the worst fleck (`FLOORS`); each stays clear of the action's green and the alarm's
    # red (S4) and of the other seven (`tests/test_marks.py`).
    "mk-1": "#8c2b4a", "mk-2": "#2c4f9e", "mk-3": "#673a8c", "mk-4": "#8a5f12",
    "mk-5": "#6e4a33", "mk-6": "#1d6377", "mk-7": "#5f6a12", "mk-8": "#7a3f6e",
}

DARK: dict[str, str] = {
    "bg": "#1b1714", "panel": "#241f1b", "ink": "#efe7db", "muted": "#a99d8e",
    "line": "#3a322b", "edge": "#857868", "faint": "#4a4038",
    "sunk": "rgba(255, 240, 220, .05)", "stripe": "rgba(255, 240, 220, .04)",
    "now": "rgba(239, 231, 219, .12)",
    "shadow": "0 1px 2px rgba(0, 0, 0, .3)",
    "shadow-pop": "0 8px 26px rgba(0, 0, 0, .45)",
    "shadow-toast": "0 6px 24px rgba(0, 0, 0, .55)",
    "shadow-bar": "0 -5px 18px rgba(0, 0, 0, .45)",
    "paper": paper(*PAPER_DARK),
    "act": "#4fae84", "act-ink": "#0e1a14", "act-tint": "rgba(79, 174, 132, .16)",
    "focus": "#4fae84",
    "note": "#d3ab62", "meter": "#c99a4a",
    "good": "#6fae7c", "bad": "#e0846f",
    "win": "#5cc48a", "loss": "#e0846f",
    "alarm": "#ff6f5e", "alarm-ink": "#1b0f0d",
    # The board dims; it does not invert (S8).
    "sq-lt": "#dcc49d", "sq-dk": "#a27b56", "frame": "#4a3322", "wall": "#2a1f17",
    # Fainter on the dimmed squares, where a dark streak costs a black piece more.
    "grain-lt-a": ".2", "grain-dk-a": ".16",
    "dot": "#1e140c",
    "held": "#0a2519", "held-halo": "#f3e9d6",
    "standoff-run": "#7f4400", "standoff-bar": "#1a3a80",
    "clash": "#851a56", "rest": "#40464f",
    "order": "#251f48",
    "announced": "#005a66",
    "duck": "#434a00",
    "frozen": "#0b4774", "disarmed": "#7e300e", "shielded": "#185a31",
    "commanded": "#622984",
    "scrim": "rgba(0, 0, 0, .6)",
    "p0": "#f0ede7", "p1": "#3b3f45", "p2": "#8f6a2f", "p2-ink": "#f4f2ee",
    "portal-stone": "#9b948a",
    "coming": "#341a5e", "coming-ink": "#b9a3ec",
    # A walk's hues and casing, on the board, which does not invert: the light ones (Q852).
    "walk-1": "#648fff", "walk-2": "#fe6100", "walk-3": "#dc267f", "walk-4": "#ffb000",
    "walk-edge": "#1e140c",
    "claim0": "#6fae7c", "claim1": "#a98fd6",
    "goal": "#5cc9bd", "seam": "#74a6ef", "zone": "#e58cc4",
    "kind-board": "#74a6ef", "kind-pieces": "#e58cc4", "kind-setup": "#5cc9bd",
    "kind-capture": "#e0836a", "kind-ending": "#6fae7c", "kind-hand": "#a98fd6",
    "kind-wild": "#d9a441", "kind-turn": "#b0a48f", "kind-clock": "#c3bf55",
    "kind-valuation": "#9a93ef",
    "community-1": "#3987e5", "community-2": "#d95926", "community-3": "#199e70",
    "community-4": "#c98500", "community-5": "#d55181", "community-6": "#008300",
    "community-7": "#9085e9", "community-8": "#e66767",
    "mk-1": "#e88aa6", "mk-2": "#8ea7f2", "mk-3": "#c09df0", "mk-4": "#d9ab52",
    "mk-5": "#d0a585", "mk-6": "#78bfd6", "mk-7": "#bfc35e", "mk-8": "#e39bd2",
}

# Names kept for what still reads them by their old names: `viz/wrapped.js` draws a
# square in `--light` or `--dark`, and the offline replay (`viz/html.py`, which carries a
# palette of its own) shares that file. They point at the board's tokens, so they follow
# the theme with them; renaming them inside `viz/` is the replay's own task.
ALIASES: dict[str, str] = {"light": "var(--sq-lt)", "dark": "var(--sq-dk)"}

# The contrast floors of plan-visual.md §3.2 and §2 (S1, S4, S8), each (foreground,
# background, floor). A pair whose background is a surface the paper lies on is read over
# the paper's worst fleck too (`tests/test_style.py`). 4.5 is AA for text; 3 for a
# control's edge.
FLOORS: tuple[tuple[str, str, float], ...] = (
    ("ink", "bg", 4.5),
    ("ink", "panel", 4.5),
    ("muted", "panel", 4.5),
    ("muted", "bg", 4.5),
    ("act-ink", "act", 4.5),
    ("act", "panel", 4.5),
    ("act", "bg", 4.5),
    ("alarm", "panel", 4.5),
    ("good", "panel", 4.5),
    ("bad", "panel", 4.5),
    ("note", "panel", 3.0),
    # The controls (V3, §3.5): a quiet or danger button is its text on whatever it sits on,
    # the page or a panel; a field's edge and the focus ring are a control's boundary.
    ("alarm", "bg", 4.5),
    ("edge", "panel", 3.0),
    ("edge", "bg", 3.0),
    ("focus", "panel", 3.0),
    ("focus", "bg", 3.0),
    # The chosen segment of a switch: the panel's colour on an ink fill.
    ("panel", "ink", 4.5),
    # The play frame (V4): the filled *in check* pill, the ended banner's result in the
    # reader's win or loss colour on the banner's panel, and the square names beside the
    # board, which are muted text on the panel (above).
    ("alarm-ink", "alarm", 4.5),
    ("win", "panel", 4.5),
    ("loss", "panel", 4.5),
    # The legend's name for a wild piece's coming step (Q660), on the page and a panel.
    ("coming-ink", "panel", 4.5),
    ("coming-ink", "bg", 4.5),
    # A portal (Q772b): the keyline on the stone of the designer's end being put down. The
    # keyline against the woods is read over their worst streak in `TestContrast`.
    ("portal-edge", "portal-stone", 4.5),
    # A pair's well (Q838b): its hue against the keyline that rings it.
    *((f"pair-{n}", "portal-edge", 3.0) for n in range(1, 5)),
    # A standoff's link (Q940): each colour against the casing it is drawn on, where the
    # run's *1* is written in it.
    ("standoff-run", "held-halo", 4.5),
    ("standoff-bar", "held-halo", 4.5),
    # A swap's burst and a rested piece's sign (Q942b), on the same casing; the sign's bars
    # are the casing's colour on the disc.
    ("clash", "held-halo", 4.5),
    ("held-halo", "rest", 4.5),
    # The order disc's numeral and ring (Q943b), on the same casing.
    ("order", "held-halo", 4.5),
    # An announced move's arrow (Q935c), on the same casing.
    ("announced", "held-halo", 4.5),
    # The duck's flight (N4c), on the same casing.
    ("duck", "held-halo", 4.5),
    # A line of sight's badges (Q999c): each glyph is the casing's colour on its disc.
    ("held-halo", "frozen", 4.5),
    ("held-halo", "disarmed", 4.5),
    ("held-halo", "shielded", 4.5),
    # A commanded piece's badge (Q1007d), the same way round.
    ("held-halo", "commanded", 4.5),
    # A wild piece's walk (Q852): each hue against the casing it is drawn on.
    *((f"walk-{n}", "walk-edge", 3.0) for n in range(1, 5)),
    # A member's mark (R3): each of its eight hues, a graphic, on the page and a panel.
    *((f"mk-{n}", back, 3.0) for n in range(1, 9) for back in ("panel", "bg")),
)
# A chosen chip or toggle: text over the action's translucent tint, each (text, tint,
# floor), read composed over the panel and the ground. (A quiet button under the pointer
# is underlined rather than tinted: any tint on the paper's worst fleck took its green
# under AA.)
TINTED: tuple[tuple[str, str, float], ...] = (
    ("ink", "act-tint", 4.5),
)

# The controls' sizes (plan-visual.md §3.4, §3.5): not colours, so one set for both themes.
# `ctl-h` is a button's, an input's and a select's height; `ctl-h-sm` a compact control's,
# in a dense row (a card's drop, the designer's palette, a maintainer's row action, the
# footer's theme switch); `chip-h` a chip's. `hit` is how far a control drawn smaller than
# that (the (i), a chip, a link-like button) reaches for a pointer: 24 px at a desk, WCAG's
# minimum target. Two radii, controls' and panels'.
SIZES: dict[str, str] = {
    "ctl-h": "40px", "ctl-h-sm": "30px", "chip-h": "28px", "hit": "24px",
    # The one control a frame makes larger than the rest: *Send your move* under a board
    # between two people (Q733), the step a person must not miss.
    "ctl-h-lg": "52px",
    "r-ctl": "8px", "r-panel": "10px",
}
# Under a coarse pointer (a finger) every target is at least 44 px (§3.5): a button, an
# input and a select at 44, a compact control at 44, a chip drawn at 36 and reaching 44.
COARSE: dict[str, str] = {"ctl-h": "44px", "ctl-h-sm": "44px", "chip-h": "36px", "hit": "44px",
                          "ctl-h-lg": "56px"}
TARGET = 44
# A black piece on a dark square (S8): the library's black pieces are filled #000.
PIECE_FLOOR = 4.5
# The marks the play board lays over a square, (token, opacity as `board.js` and
# `viz/wrapped.js` draw it): a legal-move dot, the last move, the selection, the check's
# glow, a wild piece's coming step. The grain lies under all of them, and
# `tests/test_style.py` reads each over the
# worst streak: it must stay at least `MARK_KEPT` as distinct from its square as it is from
# the bare square, and more distinct than the streak itself is.
BOARD_MARKS: tuple[tuple[str, float], ...] = (
    ("dot", 0.8), ("hl-last", 0.34), ("hl-sel", 0.55), ("alarm", 0.55), ("coming", 0.85),
    # What a piece sees (schema 30, Q999c), each effect's tint at `SEEN_OPACITY`.
    ("frozen", 0.3), ("disarmed", 0.3), ("shielded", 0.3),
    # A command sight's (Q1007d), at the same opacity.
    ("commanded", 0.3),
)
MARK_KEPT = 0.8
# A legal-move dot is a control's mark, not text: WCAG's 3 : 1 for a graphic.
DOT_FLOOR = 3.0
PAPERED = ("bg", "panel")


def block(tokens: dict[str, str], indent: str = "    ") -> str:
    """One theme's custom properties, one declaration a line."""
    return "\n".join(f"{indent}--{name}: {value};" for name, value in tokens.items())


TOKENS = f"""
  :root {{
{block(LIGHT)}
{block(ALIASES)}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
{block(DARK, "      ")}
    }}
  }}
  :root[data-theme="dark"] {{
{block(DARK)}
  }}
  :root {{ color-scheme: light; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; }} }}
  :root[data-theme="dark"] {{ color-scheme: dark; }}
  :root {{
{block(SIZES)}
  }}
  @media (pointer: coarse) {{
    :root {{
{block(COARSE, "      ")}
    }}
  }}
"""
