"""Draw the library's own pieces in Cburnett's grammar (plan-visual.md §7, decision D4).

    python assets/pieces/draw.py

Every piece here is composed from a few shared parts, so a board of them reads as one set
with the familiar six beside them: a 45 x 45 frame, a 1.5 black stroke with round joins and
caps, the first player's pieces filled white and the second's black, a detail line black on
a white piece and white on a black one (Cburnett's own rule), one shared base, transparent
ground. A piece is a *body* standing on that base and a *head* on the body; what kind of
piece it is shows in the body and what piece it is shows in the head:

    leaper    a slim stem and collar, the head an emblem standing on it
    rider     a tall shaft (a rider goes far), the emblem on top
    compound  a rider's own top (the rook's battlements, the bishop's mitre, the queen's
              coronet) with a leaper's emblem rising from it, as the chancellor is drawn
    royal     the king's wide body, a crown of its own on top
    pawn      the pawn's low skirt, a small head
    oddity    a picture standing on the base alone

Every part is original geometry, drawn here and released CC0, except where a drawing uses
Cburnett's knight head (`KNIGHT`), which keeps his CC BY-SA 3.0 and credits him, as the
chancellor and archbishop do. `DRAWN` is the table `fetch.py` reads, so the credits and
the manifest name each drawing's licence from what it is made of.

The files are written, not hand-edited: change a part here and every piece that uses it
changes with it. `check.py` then sorts the whole library by silhouette at 24 px.
"""

# compute-lease: light
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SVG_DIR = ROOT / "svg"

# The grammar's constants: the frame, the ink, the two seats.
FRAME = 45
STROKE = 1.5
INK = "#000000"
SEATS = {"white": ("#ffffff", "#000000"), "black": ("#000000", "#ffffff")}  # fill, detail
CX = 22.5  # the frame's centre line, which every symmetric piece is drawn about


@dataclass(frozen=True)
class Mark:
    """One shape. `kind` says how a seat paints it:

    body   filled in the seat's colour, outlined in black
    line   a detail line inside a body: black on a white piece, white on a black one
    dot    a filled detail (an eye, a jewel): the detail colour, no outline
    solid  black on both seats, outlined (a cannon's bore, a bomb's fuse)
    """

    kind: str
    d: str
    extra: str = ""
    # Where the shape is drawn: a point (x, y) of `d` lands at (dx + k x, dy + k y). The
    # stroke is divided by `k`, so a grown head keeps the grammar's one stroke width.
    k: float = 1.0
    dx: float = 0.0
    dy: float = 0.0


Part = list[Mark]


def _n(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def poly(points: Iterable[tuple[float, float]], close: bool = True) -> str:
    pts = list(points)
    head = f"M {_n(pts[0][0])},{_n(pts[0][1])}"
    rest = " ".join(f"L {_n(x)},{_n(y)}" for x, y in pts[1:])
    return f"{head} {rest}{' Z' if close else ''}"


def circle(cx: float, cy: float, r: float) -> str:
    return (
        f"M {_n(cx - r)},{_n(cy)} A {_n(r)},{_n(r)} 0 1 1 {_n(cx + r)},{_n(cy)} "
        f"A {_n(r)},{_n(r)} 0 1 1 {_n(cx - r)},{_n(cy)} Z"
    )


def ellipse(cx: float, cy: float, rx: float, ry: float) -> str:
    return (
        f"M {_n(cx - rx)},{_n(cy)} A {_n(rx)},{_n(ry)} 0 1 1 {_n(cx + rx)},{_n(cy)} "
        f"A {_n(rx)},{_n(ry)} 0 1 1 {_n(cx - rx)},{_n(cy)} Z"
    )


def star(cx: float, cy: float, outer: float, inner: float, points: int,
         turn: float = -90.0) -> str:
    pts = []
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        a = math.radians(turn + i * 180.0 / points)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return poly(pts)


def rect(x: float, y: float, w: float, h: float) -> str:
    return poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def mirror(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """A half outline drawn left of the centre line, closed by its mirror image."""
    return points + [(2 * CX - x, y) for x, y in reversed(points)]


# ------------------------------------------------------------------ the shared parts

# The base every piece stands on: a plinth, its top corners eased.
BASE: Part = [Mark("body", "M 9.5,39.5 L 35.5,39.5 L 35.5,37 C 35.5,36 34.5,35.5 33.5,35.5 "
                           "L 11.5,35.5 C 10.5,35.5 9.5,36 9.5,37 Z")]

# Where two parts of a body meet, a seam: invisible on a white piece, whose outline is
# already there, and the white line that keeps a black piece from reading as one blot
# (Cburnett's black pieces are drawn the same way). Drawn after the body, before the head.
BASE_SEAM = Mark("line", "M 11.5,35.5 L 33.5,35.5")

# Bodies. Each ends at a top line `y` where its head stands, returned beside the marks.
def stem() -> tuple[Part, float]:
    """The leaper's body: a short waisted stem and a collar, the head standing at y 28.5,
    so the head has most of the frame to say what the piece is."""
    return [
        Mark("body", "M 14.5,35.5 C 16.5,33.5 18.5,32.5 19,30.5 L 26,30.5 C 26.5,32.5 "
                     "28.5,33.5 30.5,35.5 Z"),
        Mark("body", "M 16.5,30.5 L 28.5,30.5 L 28.5,29.5 C 28.5,29 28,28.5 27.5,28.5 "
                     "L 17.5,28.5 C 17,28.5 16.5,29 16.5,29.5 Z"),
        Mark("line", "M 18,30.5 L 27,30.5"),
    ], 28.5


def shaft() -> tuple[Part, float]:
    """The rider's body: a straight shaft (a rider goes far) on a step, its collar higher
    than the leaper's, the head standing at y 24.5."""
    return [
        Mark("body", "M 12.5,35.5 L 12.5,34 C 12.5,33.5 13,33 13.5,33 L 31.5,33 "
                     "C 32,33 32.5,33.5 32.5,34 L 32.5,35.5 Z"),
        Mark("body", rect(20.5, 26, 4, 7)),
        Mark("body", "M 17,26 L 28,26 L 28,25.5 C 28,25 27.5,24.5 27,24.5 L 18,24.5 "
                     "C 17.5,24.5 17,25 17,25.5 Z"),
        Mark("line", "M 14,33 L 31,33 M 19,26 L 26,26"),
    ], 24.5


def barrel() -> tuple[Part, float]:
    """The royal body: the king's wide barrel, its band a detail line, crowned at y 21."""
    return [
        Mark("body", "M 11,35.5 C 9.5,32 9.5,27 11.5,23.5 L 33.5,23.5 C 35.5,27 35.5,32 "
                     "34,35.5 Z"),
        Mark("line", "M 11.5,31 C 18,29.5 27,29.5 33.5,31"),
        Mark("body", "M 11.5,23.5 L 33.5,23.5 L 32.5,21 L 12.5,21 Z"),
        Mark("line", "M 13,23.5 L 32,23.5"),
    ], 21


def skirt() -> tuple[Part, float]:
    """The pawn's body: a low bell skirt, the head standing at y 23.5."""
    return [
        Mark("body", "M 12.5,35.5 C 13,30.5 16.5,27.5 19,26 L 26,26 C 28.5,27.5 32,30.5 "
                     "32.5,35.5 Z"),
        Mark("body", "M 17.5,26 L 27.5,26 C 28,25 27.5,24 26.5,23.5 L 18.5,23.5 "
                     "C 17.5,24 17,25 17.5,26 Z"),
        Mark("line", "M 19,26 L 26,26"),
    ], 23.5


def plinth() -> tuple[Part, float]:
    """The oddity's body: nothing but the base, the picture standing at y 35.5."""
    return [], 35.5


# The tops a compound borrows from the familiar riders, drawn here in their outline and
# not copied: the rook's battlements, the bishop's mitre, the queen's coronet, each on a
# straight column as broad as the rook's (so a compound is never the bishop's slim waist)
# and lower than the rider's own, so the leaper's emblem has room above it.
COLUMN: Part = [
    Mark("body", "M 12,35.5 L 12,33 L 33,33 L 33,35.5 Z"),
    Mark("body", "M 14,33 L 15,26.5 L 30,26.5 L 31,33 Z"),
    Mark("line", "M 13.5,33 L 31.5,33 M 16.5,26.5 L 28.5,26.5"),
]


def battlements() -> tuple[Part, float]:
    return COLUMN + [
        Mark("body", "M 15,26.5 L 12,23.5 L 33,23.5 L 30,26.5 Z"),
        Mark("body", "M 12,23.5 L 12,18.5 L 16,18.5 L 16,20.5 L 20.5,20.5 L 20.5,18.5 "
                     "L 24.5,18.5 L 24.5,20.5 L 29,20.5 L 29,18.5 L 33,18.5 L 33,23.5 Z"),
        Mark("line", "M 13.5,23.5 L 31.5,23.5"),
    ], 20.5


def mitre() -> tuple[Part, float]:
    return COLUMN + [
        Mark("body", "M 15,26.5 C 11,22.5 14,17.5 22.5,12.5 C 31,17.5 34,22.5 30,26.5 Z"),
        Mark("line", "M 22.5,17 L 22.5,23 M 19.5,20 L 25.5,20"),
    ], 13.5


def coronet() -> tuple[Part, float]:
    return COLUMN + [
        Mark("body", "M 15,26.5 L 10.5,16.5 L 16.5,21.5 L 19,15 L 22.5,20.5 L 26,15 "
                     "L 28.5,21.5 L 34.5,16.5 L 30,26.5 Z"),
    ], 19.5


BODIES: dict[str, Callable[[], tuple[Part, float]]] = {
    "stem": stem, "shaft": shaft, "barrel": barrel, "skirt": skirt, "plinth": plinth,
    "battlements": battlements, "mitre": mitre, "coronet": coronet,
}


# ------------------------------------------------------------------ the heads
#
# A head is drawn standing on the line `top`, centred on the frame. `room` is how tall
# it may be; most fill it.

Head = Callable[[float], Part]


def _ball(r: float) -> Head:
    return lambda top: [Mark("body", circle(CX, top - r + 0.5, r))]


def _star(points: int, outer: float, inner: float) -> Head:
    return lambda top: [Mark("body", star(CX, top - outer * 0.95, outer, inner, points))]


def trefoil(top: float) -> Part:
    r = 4.0
    return [
        Mark("body", circle(CX - 4.2, top - 4.6, r)),
        Mark("body", circle(CX + 4.2, top - 4.6, r)),
        Mark("body", circle(CX, top - 11.0, r)),
        Mark("dot", circle(CX, top - 6.5, 1.0)),
    ]


def triangle(top: float) -> Part:
    return [
        Mark("body", poly([(CX - 9.5, top + 0.5), (CX, top - 16), (CX + 9.5, top + 0.5)])),
        Mark("line", poly([(CX - 4, top - 3), (CX, top - 10), (CX + 4, top - 3)])),
    ]


def diamond(top: float) -> Part:
    return [
        Mark("body", poly([(CX, top - 15), (CX + 11, top - 7.25), (CX, top + 0.5),
                           (CX - 11, top - 7.25)])),
        Mark("dot", circle(CX, top - 7.25, 1.6)),
    ]


def block(top: float) -> Part:
    return [
        Mark("body", rect(CX - 7, top - 14, 14, 14.5)),
        Mark("line", rect(CX - 3.5, top - 10.5, 7, 7.5)),
    ]


def ring(top: float) -> Part:
    return [
        Mark("body", circle(CX, top - 8.5, 8.5) + " " + circle(CX, top - 8.5, 5),
             'fill-rule="evenodd"'),
    ]


def cross(top: float) -> Part:
    a, b = 2.6, 8.0
    return [
        Mark("body", poly([
            (CX - a, top + 0.5), (CX - a, top - 8 + a), (CX - b, top - 8 + a),
            (CX - b, top - 8 - a), (CX - a, top - 8 - a), (CX - a, top - 16),
            (CX + a, top - 16), (CX + a, top - 8 - a), (CX + b, top - 8 - a),
            (CX + b, top - 8 + a), (CX + a, top - 8 + a), (CX + a, top + 0.5),
        ])),
    ]


def saltire(top: float) -> Part:
    c, w, r = top - 9.5, 2.0, 11.5
    arms = []
    for angle in (45, 135):
        a = math.radians(angle)
        dx, dy = math.cos(a), math.sin(a)
        px, py = -dy * w, dx * w
        arms.append(Mark("body", poly([
            (CX - dx * r + px, c - dy * r + py), (CX + dx * r + px, c + dy * r + py),
            (CX + dx * r - px, c + dy * r - py), (CX - dx * r - px, c - dy * r - py),
        ])))
    return arms


def crescent_up(top: float) -> Part:
    """Two horns rising, the bowl standing on the collar."""
    return [Mark("body", f"M {_n(CX - 9)},{_n(top - 14)} C {_n(CX - 10)},{_n(top - 3)} "
                         f"{_n(CX + 10)},{_n(top - 3)} {_n(CX + 9)},{_n(top - 14)} "
                         f"C {_n(CX + 9.5)},{_n(top + 2)} {_n(CX - 9.5)},{_n(top + 2)} "
                         f"{_n(CX - 9)},{_n(top - 14)} Z")]


def spear(top: float) -> Part:
    return [
        Mark("body", f"M {_n(CX)},{_n(top - 19)} C {_n(CX + 5)},{_n(top - 13)} "
                     f"{_n(CX + 5)},{_n(top - 7)} {_n(CX + 1.5)},{_n(top - 3)} "
                     f"L {_n(CX + 1.5)},{_n(top + 0.5)} L {_n(CX - 1.5)},{_n(top + 0.5)} "
                     f"L {_n(CX - 1.5)},{_n(top - 3)} C {_n(CX - 5)},{_n(top - 7)} "
                     f"{_n(CX - 5)},{_n(top - 13)} {_n(CX)},{_n(top - 19)} Z"),
        Mark("line", f"M {_n(CX)},{_n(top - 15)} L {_n(CX)},{_n(top - 6)}"),
    ]


def trident(top: float) -> Part:
    t = top
    return [Mark("body", poly([
        (CX - 1.5, t + 0.5), (CX - 1.5, t - 5), (CX - 7, t - 6), (CX - 9, t - 15),
        (CX - 6.5, t - 13), (CX - 5.5, t - 8.5), (CX - 1.5, t - 8), (CX - 1.5, t - 15),
        (CX, t - 18), (CX + 1.5, t - 15), (CX + 1.5, t - 8), (CX + 5.5, t - 8.5),
        (CX + 6.5, t - 13), (CX + 9, t - 15), (CX + 7, t - 6), (CX + 1.5, t - 5),
        (CX + 1.5, t + 0.5),
    ]))]


def flame(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX)},{_n(t + 0.5)} C {_n(CX - 9)},{_n(t)} {_n(CX - 9)},{_n(t - 9)} "
                     f"{_n(CX - 4)},{_n(t - 13)} C {_n(CX - 4)},{_n(t - 10)} {_n(CX - 2)},"
                     f"{_n(t - 9)} {_n(CX - 1)},{_n(t - 9)} C {_n(CX - 3)},{_n(t - 14)} "
                     f"{_n(CX + 1)},{_n(t - 18)} {_n(CX + 3)},{_n(t - 20)} C {_n(CX + 3)},"
                     f"{_n(t - 15)} {_n(CX + 9)},{_n(t - 11)} {_n(CX + 8)},{_n(t - 5)} "
                     f"C {_n(CX + 7.5)},{_n(t - 1)} {_n(CX + 4)},{_n(t + 0.5)} {_n(CX)},"
                     f"{_n(t + 0.5)} Z"),
        Mark("line", f"M {_n(CX)},{_n(t - 2)} C {_n(CX - 3)},{_n(t - 3)} {_n(CX - 2)},"
                     f"{_n(t - 7)} {_n(CX + 1)},{_n(t - 9)}"),
    ]


def heart(top: float) -> Part:
    t = top
    return [Mark("body", f"M {_n(CX)},{_n(t + 0.5)} C {_n(CX - 4)},{_n(t - 4)} "
                         f"{_n(CX - 10)},{_n(t - 7)} {_n(CX - 9.5)},{_n(t - 12)} "
                         f"C {_n(CX - 9)},{_n(t - 17)} {_n(CX - 2)},{_n(t - 17.5)} "
                         f"{_n(CX)},{_n(t - 12.5)} C {_n(CX + 2)},{_n(t - 17.5)} "
                         f"{_n(CX + 9)},{_n(t - 17)} {_n(CX + 9.5)},{_n(t - 12)} "
                         f"C {_n(CX + 10)},{_n(t - 7)} {_n(CX + 4)},{_n(t - 4)} "
                         f"{_n(CX)},{_n(t + 0.5)} Z")]


def bell(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX - 9)},{_n(t - 1)} C {_n(CX - 6)},{_n(t - 4)} {_n(CX - 7)},"
                     f"{_n(t - 15)} {_n(CX)},{_n(t - 16)} C {_n(CX + 7)},{_n(t - 15)} "
                     f"{_n(CX + 6)},{_n(t - 4)} {_n(CX + 9)},{_n(t - 1)} Z"),
        Mark("body", circle(CX, t - 17.5, 1.8)),
        Mark("body", circle(CX, t - 0.5, 1.8)),
    ]


def hourglass(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX - 8, t - 2, 16, 2.5)),
        Mark("body", rect(CX - 8, t - 19, 16, 2.5)),
        Mark("body", f"M {_n(CX - 6)},{_n(t - 16.5)} C {_n(CX - 6)},{_n(t - 12)} "
                     f"{_n(CX - 1)},{_n(t - 11)} {_n(CX - 1)},{_n(t - 9.5)} "
                     f"C {_n(CX - 1)},{_n(t - 8)} {_n(CX - 6)},{_n(t - 7)} {_n(CX - 6)},"
                     f"{_n(t - 2)} L {_n(CX + 6)},{_n(t - 2)} C {_n(CX + 6)},{_n(t - 7)} "
                     f"{_n(CX + 1)},{_n(t - 8)} {_n(CX + 1)},{_n(t - 9.5)} "
                     f"C {_n(CX + 1)},{_n(t - 11)} {_n(CX + 6)},{_n(t - 12)} {_n(CX + 6)},"
                     f"{_n(t - 16.5)} Z"),
    ]


def wings(top: float) -> Part:
    """Two wings spread from a small body, feathers as detail lines."""
    t = top
    left = [(CX - 2, t + 0.5), (CX - 6, t - 2), (CX - 11, t - 5), (CX - 14, t - 10),
            (CX - 13, t - 15), (CX - 9, t - 12), (CX - 6, t - 14.5), (CX - 4, t - 10),
            (CX - 2, t - 12)]
    return [
        Mark("body", poly(mirror(left))),
        Mark("line", f"M {_n(CX - 4)},{_n(t - 4)} L {_n(CX - 10)},{_n(t - 8)} "
                     f"M {_n(CX + 4)},{_n(t - 4)} L {_n(CX + 10)},{_n(t - 8)}"),
        Mark("body", circle(CX, t - 14, 2.8)),
    ]


def sun(top: float) -> Part:
    c = top - 9
    return [
        Mark("body", star(CX, c, 10, 4.8, 8, turn=-90 + 22.5)),
        Mark("body", circle(CX, c, 4.6)),
    ]


def eye(top: float) -> Part:
    c = top - 7.5
    return [
        Mark("body", f"M {_n(CX - 11)},{_n(c)} C {_n(CX - 6)},{_n(c - 8)} {_n(CX + 6)},"
                     f"{_n(c - 8)} {_n(CX + 11)},{_n(c)} C {_n(CX + 6)},{_n(c + 8)} "
                     f"{_n(CX - 6)},{_n(c + 8)} {_n(CX - 11)},{_n(c)} Z"),
        Mark("line", circle(CX, c, 3.6)),
        Mark("dot", circle(CX, c, 1.6)),
    ]


def arrow(top: float) -> Part:
    t = top
    return [Mark("body", poly([
        (CX - 2, t + 0.5), (CX - 2, t - 10), (CX - 8, t - 9), (CX, t - 19),
        (CX + 8, t - 9), (CX + 2, t - 10), (CX + 2, t + 0.5),
    ]))]


def chevron(top: float) -> Part:
    t = top
    return [Mark("body", poly([
        (CX - 10, t - 2), (CX, t - 13), (CX + 10, t - 2), (CX + 10, t - 7.5),
        (CX, t - 18.5), (CX - 10, t - 7.5),
    ]))]


def bolt(top: float) -> Part:
    t = top
    return [Mark("body", poly([
        (CX - 2, t + 0.5), (CX + 9, t - 9), (CX + 2, t - 9), (CX + 7, t - 19),
        (CX - 9, t - 7), (CX - 2, t - 7.5), (CX - 6, t + 0.5),
    ]))]


def leaf(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX)},{_n(t + 0.5)} C {_n(CX - 12)},{_n(t - 6)} {_n(CX - 6)},"
                     f"{_n(t - 16)} {_n(CX + 6)},{_n(t - 19)} C {_n(CX + 9)},{_n(t - 10)} "
                     f"{_n(CX + 6)},{_n(t - 2)} {_n(CX)},{_n(t + 0.5)} Z"),
        Mark("line", f"M {_n(CX)},{_n(t - 1)} C {_n(CX)},{_n(t - 8)} {_n(CX + 2)},"
                     f"{_n(t - 13)} {_n(CX + 5)},{_n(t - 17)}"),
    ]


def shield(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX - 8.5)},{_n(t - 17)} L {_n(CX + 8.5)},{_n(t - 17)} "
                     f"L {_n(CX + 8.5)},{_n(t - 9)} C {_n(CX + 8.5)},{_n(t - 4)} "
                     f"{_n(CX + 4)},{_n(t - 1)} {_n(CX)},{_n(t + 0.5)} C {_n(CX - 4)},"
                     f"{_n(t - 1)} {_n(CX - 8.5)},{_n(t - 4)} {_n(CX - 8.5)},{_n(t - 9)} Z"),
        Mark("line", f"M {_n(CX)},{_n(t - 15)} L {_n(CX)},{_n(t - 2)} "
                     f"M {_n(CX - 6.5)},{_n(t - 10)} L {_n(CX + 6.5)},{_n(t - 10)}"),
    ]


def anchor(top: float) -> Part:
    t = top
    return [
        Mark("body", circle(CX, t - 17, 2.5)),
        Mark("body", rect(CX - 1.5, t - 14.5, 3, 13)),
        Mark("body", rect(CX - 6, t - 12.5, 12, 2.5)),
        Mark("body", f"M {_n(CX - 10)},{_n(t - 7)} C {_n(CX - 9)},{_n(t)} {_n(CX + 9)},"
                     f"{_n(t)} {_n(CX + 10)},{_n(t - 7)} L {_n(CX + 7)},{_n(t - 5.5)} "
                     f"C {_n(CX + 5)},{_n(t - 2.5)} {_n(CX - 5)},{_n(t - 2.5)} "
                     f"{_n(CX - 7)},{_n(t - 5.5)} Z"),
    ]


def rose(top: float) -> Part:
    c = top - 8.5
    petals = [Mark("body", circle(CX + 5 * math.cos(math.radians(-90 + 72 * i)),
                                  c + 5 * math.sin(math.radians(-90 + 72 * i)), 4.2))
              for i in range(5)]
    return petals + [Mark("body", circle(CX, c, 3.0)), Mark("dot", circle(CX, c, 1.1))]


def acorn(top: float) -> Part:
    t = top
    return [
        Mark("body", ellipse(CX, t - 6.5, 6.5, 7)),
        Mark("body", f"M {_n(CX - 9)},{_n(t - 11)} C {_n(CX - 9)},{_n(t - 18)} {_n(CX + 9)},"
                     f"{_n(t - 18)} {_n(CX + 9)},{_n(t - 11)} Z"),
        Mark("body", rect(CX - 1, t - 20.5, 2, 3.5)),
    ]


def horns(top: float) -> Part:
    """Two long horns in a V, an antelope's."""
    t = top
    return [
        Mark("body", ellipse(CX, t - 4, 5, 4.5)),
        Mark("body", f"M {_n(CX - 3)},{_n(t - 7)} C {_n(CX - 7)},{_n(t - 12)} {_n(CX - 9)},"
                     f"{_n(t - 16)} {_n(CX - 10)},{_n(t - 21)} C {_n(CX - 6)},{_n(t - 17)} "
                     f"{_n(CX - 3)},{_n(t - 13)} {_n(CX)},{_n(t - 8)} Z"),
        Mark("body", f"M {_n(CX + 3)},{_n(t - 7)} C {_n(CX + 7)},{_n(t - 12)} {_n(CX + 9)},"
                     f"{_n(t - 16)} {_n(CX + 10)},{_n(t - 21)} C {_n(CX + 6)},{_n(t - 17)} "
                     f"{_n(CX + 3)},{_n(t - 13)} {_n(CX)},{_n(t - 8)} Z"),
    ]


def bull(top: float) -> Part:
    """A broad head with short horns curving out, a bison's."""
    t = top
    return [
        Mark("body", f"M {_n(CX - 6)},{_n(t - 11)} C {_n(CX - 12)},{_n(t - 11)} "
                     f"{_n(CX - 14)},{_n(t - 15)} {_n(CX - 13)},{_n(t - 19)} "
                     f"C {_n(CX - 11)},{_n(t - 16)} {_n(CX - 9)},{_n(t - 15)} {_n(CX - 5)},"
                     f"{_n(t - 15)} L {_n(CX + 5)},{_n(t - 15)} C {_n(CX + 9)},{_n(t - 15)} "
                     f"{_n(CX + 11)},{_n(t - 16)} {_n(CX + 13)},{_n(t - 19)} "
                     f"C {_n(CX + 14)},{_n(t - 15)} {_n(CX + 12)},{_n(t - 11)} {_n(CX + 6)},"
                     f"{_n(t - 11)} Z"),
        Mark("body", f"M {_n(CX - 6.5)},{_n(t - 13)} C {_n(CX - 7)},{_n(t - 5)} {_n(CX - 4)},"
                     f"{_n(t + 0.5)} {_n(CX)},{_n(t + 0.5)} C {_n(CX + 4)},{_n(t + 0.5)} "
                     f"{_n(CX + 7)},{_n(t - 5)} {_n(CX + 6.5)},{_n(t - 13)} Z"),
        Mark("dot", circle(CX - 3, t - 9, 1.0)),
        Mark("dot", circle(CX + 3, t - 9, 1.0)),
    ]


def frog(top: float) -> Part:
    t = top
    return [
        Mark("body", ellipse(CX, t - 5.5, 10, 6)),
        Mark("body", circle(CX - 6, t - 11, 3.4)),
        Mark("body", circle(CX + 6, t - 11, 3.4)),
        Mark("dot", circle(CX - 6, t - 11.5, 1.3)),
        Mark("dot", circle(CX + 6, t - 11.5, 1.3)),
        Mark("line", f"M {_n(CX - 5)},{_n(t - 4)} C {_n(CX - 2)},{_n(t - 2)} {_n(CX + 2)},"
                     f"{_n(t - 2)} {_n(CX + 5)},{_n(t - 4)}"),
    ]


def coronet_small(top: float) -> Part:
    """A small three-pointed crown, for a crowned piece."""
    t = top
    return [Mark("body", poly([
        (CX - 6.5, t + 0.5), (CX - 7.5, t - 7), (CX - 3.5, t - 3.5), (CX, t - 9),
        (CX + 3.5, t - 3.5), (CX + 7.5, t - 7), (CX + 6.5, t + 0.5),
    ]))]


def orb(top: float) -> Part:
    t = top
    return [
        Mark("body", circle(CX, t - 6, 6.5)),
        Mark("line", f"M {_n(CX - 6.5)},{_n(t - 6)} L {_n(CX + 6.5)},{_n(t - 6)}"),
        Mark("body", poly([
            (CX - 1.2, t - 12), (CX - 1.2, t - 14.5), (CX - 3.5, t - 14.5), (CX - 3.5, t - 16.8),
            (CX - 1.2, t - 16.8), (CX - 1.2, t - 19), (CX + 1.2, t - 19), (CX + 1.2, t - 16.8),
            (CX + 3.5, t - 16.8), (CX + 3.5, t - 14.5), (CX + 1.2, t - 14.5), (CX + 1.2, t - 12),
        ])),
    ]


def helmet(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX - 6)},{_n(t)} C {_n(CX - 6.5)},{_n(t - 10)} {_n(CX + 6.5)},"
                     f"{_n(t - 10)} {_n(CX + 6)},{_n(t)} Z"),
        Mark("body", rect(CX - 9, t - 1.5, 18, 2)),
        Mark("body", f"M {_n(CX - 1)},{_n(t - 8)} C {_n(CX - 1)},{_n(t - 12)} {_n(CX + 4)},"
                     f"{_n(t - 13)} {_n(CX + 6)},{_n(t - 12)} C {_n(CX + 3)},{_n(t - 11)} "
                     f"{_n(CX + 1.5)},{_n(t - 10)} {_n(CX + 1.5)},{_n(t - 7.5)} Z"),
    ]


def cap(top: float) -> Part:
    """A flat-topped drum, a sergeant's cap."""
    t = top
    return [
        Mark("body", f"M {_n(CX - 5)},{_n(t + 0.5)} L {_n(CX - 7)},{_n(t - 8)} L {_n(CX + 7)},"
                     f"{_n(t - 8)} L {_n(CX + 5)},{_n(t + 0.5)} Z"),
        Mark("body", rect(CX - 8.5, t - 10, 17, 2.5)),
        Mark("line", f"M {_n(CX - 3)},{_n(t - 5.5)} L {_n(CX)},{_n(t - 3)} L {_n(CX + 3)},"
                     f"{_n(t - 5.5)}"),
    ]


def onion(top: float) -> Part:
    t = top
    return [Mark("body", f"M {_n(CX - 2)},{_n(t + 0.5)} C {_n(CX - 6)},{_n(t - 2)} "
                         f"{_n(CX - 5)},{_n(t - 8)} {_n(CX)},{_n(t - 19)} C {_n(CX + 5)},"
                         f"{_n(t - 8)} {_n(CX + 6)},{_n(t - 2)} {_n(CX + 2)},{_n(t + 0.5)} Z")]


def twin(top: float) -> Part:
    """Two balls, one above the other."""
    return [Mark("body", circle(CX, top - 3.5, 4.0)), Mark("body", circle(CX, top - 10, 3.2))]


def cube(top: float) -> Part:
    t = top
    return [
        Mark("body", poly([(CX - 6, t + 0.5), (CX - 6, t - 8), (CX - 2, t - 11), (CX + 7, t - 11),
                           (CX + 7, t - 2.5), (CX + 3, t + 0.5)])),
        Mark("line", poly([(CX - 6, t - 8), (CX + 3, t - 8), (CX + 7, t - 11)], close=False)),
        Mark("line", f"M {_n(CX + 3)},{_n(t - 8)} L {_n(CX + 3)},{_n(t + 0.5)}"),
    ]


def tower_small(top: float) -> Part:
    t = top
    return [Mark("body", poly([
        (CX - 4, t + 0.5), (CX - 4, t - 6), (CX - 5.5, t - 6), (CX - 5.5, t - 10),
        (CX - 3, t - 10), (CX - 3, t - 8.5), (CX - 1, t - 8.5), (CX - 1, t - 10),
        (CX + 1, t - 10), (CX + 1, t - 8.5), (CX + 3, t - 8.5), (CX + 3, t - 10),
        (CX + 5.5, t - 10), (CX + 5.5, t - 6), (CX + 4, t - 6), (CX + 4, t + 0.5),
    ]))]


def sword(top: float) -> Part:
    t = top
    return [
        Mark("body", poly([(CX - 1.9, t - 6), (CX - 1.9, t - 17), (CX, t - 20.5),
                           (CX + 1.9, t - 17), (CX + 1.9, t - 6)])),
        Mark("body", rect(CX - 7, t - 6.8, 14, 2.4)),
        Mark("body", rect(CX - 1.3, t - 4.4, 2.6, 4.9)),
    ]


def hammer(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX - 1.4, t - 13, 2.8, 13.5)),
        Mark("body", "M " + _n(CX - 9) + "," + _n(t - 13) + " L " + _n(CX - 9) + "," + _n(t - 17)
             + " C " + _n(CX - 9) + "," + _n(t - 18.5) + " " + _n(CX - 8) + "," + _n(t - 19.5)
             + " " + _n(CX - 6.5) + "," + _n(t - 19.5) + " L " + _n(CX + 9) + "," + _n(t - 19.5)
             + " L " + _n(CX + 9) + "," + _n(t - 13) + " Z"),
    ]


def flag(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX - 6.5, t - 20, 2.4, 20.5)),
        Mark("body", f"M {_n(CX - 4.1)},{_n(t - 19.5)} C {_n(CX)},{_n(t - 21)} {_n(CX + 4)},"
                     f"{_n(t - 17)} {_n(CX + 10)},{_n(t - 18)} L {_n(CX + 8)},{_n(t - 13.5)} "
                     f"L {_n(CX + 10)},{_n(t - 9)} C {_n(CX + 4)},{_n(t - 8)} {_n(CX)},"
                     f"{_n(t - 12)} {_n(CX - 4.1)},{_n(t - 10.5)} Z"),
    ]


def axe(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX + 0.5, t - 20, 2.4, 20.5)),
        Mark("body", f"M {_n(CX + 0.5)},{_n(t - 17)} C {_n(CX - 3)},{_n(t - 18)} {_n(CX - 7)},"
                     f"{_n(t - 20)} {_n(CX - 10)},{_n(t - 21)} C {_n(CX - 12)},{_n(t - 16)} "
                     f"{_n(CX - 12)},{_n(t - 11)} {_n(CX - 10)},{_n(t - 6.5)} C {_n(CX - 7)},"
                     f"{_n(t - 8)} {_n(CX - 3)},{_n(t - 10)} {_n(CX + 0.5)},{_n(t - 10.5)} Z"),
    ]


def key(top: float) -> Part:
    t = top
    return [
        Mark("body", circle(CX, t - 15.5, 5) + " " + circle(CX, t - 15.5, 2.2),
             'fill-rule="evenodd"'),
        Mark("body", poly([(CX - 1.3, t + 0.5), (CX - 1.3, t - 10.6), (CX + 1.3, t - 10.6),
                           (CX + 1.3, t - 6), (CX + 5.5, t - 6), (CX + 5.5, t - 3.5),
                           (CX + 1.3, t - 3.5), (CX + 1.3, t - 2.5), (CX + 4, t - 2.5),
                           (CX + 4, t), (CX + 1.3, t), (CX + 1.3, t + 0.5)])),
    ]


def mushroom(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX - 2.5, t - 9, 5, 9.5)),
        Mark("body", f"M {_n(CX - 11.5)},{_n(t - 8.5)} C {_n(CX - 12)},{_n(t - 21)} "
                     f"{_n(CX + 12)},{_n(t - 21)} {_n(CX + 11.5)},{_n(t - 8.5)} Z"),
        Mark("dot", circle(CX - 5, t - 12.5, 1.6)),
        Mark("dot", circle(CX + 3, t - 15.5, 1.6)),
        Mark("dot", circle(CX + 6.5, t - 11, 1.2)),
    ]


def duck(top: float) -> Part:
    """A duck, facing left: Duck Chess's blocker."""
    t = top
    return [
        Mark("body", f"M {_n(CX - 9)},{_n(t - 9)} C {_n(CX - 10)},{_n(t - 2)} {_n(CX - 5)},"
                     f"{_n(t + 0.5)} {_n(CX + 2)},{_n(t + 0.5)} C {_n(CX + 9)},{_n(t + 0.5)} "
                     f"{_n(CX + 12)},{_n(t - 4)} {_n(CX + 13)},{_n(t - 13)} C {_n(CX + 10)},"
                     f"{_n(t - 10)} {_n(CX + 6)},{_n(t - 11)} {_n(CX + 1)},{_n(t - 10)} "
                     f"C {_n(CX - 3)},{_n(t - 9.5)} {_n(CX - 6)},{_n(t - 10)} {_n(CX - 9)},"
                     f"{_n(t - 9)} Z"),
        Mark("body", circle(CX - 5.5, t - 16, 5)),
        Mark("body", f"M {_n(CX - 10)},{_n(t - 17)} C {_n(CX - 13)},{_n(t - 17)} "
                     f"{_n(CX - 15)},{_n(t - 16)} {_n(CX - 15.5)},{_n(t - 14.5)} "
                     f"C {_n(CX - 13.5)},{_n(t - 14)} {_n(CX - 11.5)},{_n(t - 14)} "
                     f"{_n(CX - 10)},{_n(t - 14.5)} Z"),
        Mark("dot", circle(CX - 6.5, t - 17, 1.1)),
        Mark("line", f"M {_n(CX - 2)},{_n(t - 6)} C {_n(CX + 2)},{_n(t - 3)} {_n(CX + 6)},"
                     f"{_n(t - 4)} {_n(CX + 8)},{_n(t - 7)}"),
    ]


def bomb(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX + 1.5, t - 19.5, 4.5, 4)),
        Mark("body", circle(CX - 1, t - 8.5, 8.5)),
        Mark("body", star(CX + 7, t - 21.5, 3.5, 1.4, 6)),
        Mark("line", f"M {_n(CX - 5.5)},{_n(t - 12)} C {_n(CX - 4.5)},{_n(t - 14)} "
                     f"{_n(CX - 3)},{_n(t - 15)} {_n(CX - 1)},{_n(t - 15.3)}"),
    ]


def crown(top: float) -> Part:
    t = top
    return [
        Mark("body", poly([(CX - 10, t + 0.5), (CX - 12, t - 13), (CX - 6, t - 6),
                           (CX, t - 16), (CX + 6, t - 6), (CX + 12, t - 13), (CX + 10, t + 0.5)])),
        Mark("body", circle(CX - 12, t - 14.5, 1.9)),
        Mark("body", circle(CX, t - 17.5, 1.9)),
        Mark("body", circle(CX + 12, t - 14.5, 1.9)),
        Mark("line", f"M {_n(CX - 10.5)},{_n(t - 3)} L {_n(CX + 10.5)},{_n(t - 3)}"),
    ]


def gem(top: float) -> Part:
    t = top
    return [
        Mark("body", poly([(CX - 11, t - 13), (CX - 6.5, t - 18), (CX + 6.5, t - 18),
                           (CX + 11, t - 13), (CX, t + 0.5)])),
        Mark("line", poly([(CX - 11, t - 13), (CX + 11, t - 13)], close=False)),
        Mark("line", poly([(CX - 4, t - 13), (CX, t - 1), (CX + 4, t - 13)], close=False)),
    ]


def fir(top: float) -> Part:
    t = top
    return [
        Mark("body", rect(CX - 1.8, t - 4, 3.6, 4.5)),
        Mark("body", poly([(CX - 10, t - 4), (CX - 5, t - 10), (CX - 7.5, t - 10),
                           (CX - 3, t - 16), (CX - 5, t - 16), (CX, t - 22), (CX + 5, t - 16),
                           (CX + 3, t - 16), (CX + 7.5, t - 10), (CX + 5, t - 10),
                           (CX + 10, t - 4)])),
    ]


def ghost(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX - 9)},{_n(t + 0.5)} L {_n(CX - 9)},{_n(t - 10)} "
                     f"C {_n(CX - 9)},{_n(t - 21)} {_n(CX + 9)},{_n(t - 21)} {_n(CX + 9)},"
                     f"{_n(t - 10)} L {_n(CX + 9)},{_n(t + 0.5)} L {_n(CX + 6)},{_n(t - 2.5)} "
                     f"L {_n(CX + 3)},{_n(t + 0.5)} L {_n(CX)},{_n(t - 2.5)} L {_n(CX - 3)},"
                     f"{_n(t + 0.5)} L {_n(CX - 6)},{_n(t - 2.5)} Z"),
        Mark("dot", ellipse(CX - 3.5, t - 12, 1.6, 2.2)),
        Mark("dot", ellipse(CX + 3.5, t - 12, 1.6, 2.2)),
    ]


def drop(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX)},{_n(t - 21)} C {_n(CX + 4)},{_n(t - 14)} {_n(CX + 8.5)},"
                     f"{_n(t - 10)} {_n(CX + 8.5)},{_n(t - 7)} C {_n(CX + 8.5)},{_n(t - 2)} "
                     f"{_n(CX + 4.5)},{_n(t + 0.5)} {_n(CX)},{_n(t + 0.5)} C {_n(CX - 4.5)},"
                     f"{_n(t + 0.5)} {_n(CX - 8.5)},{_n(t - 2)} {_n(CX - 8.5)},{_n(t - 7)} "
                     f"C {_n(CX - 8.5)},{_n(t - 10)} {_n(CX - 4)},{_n(t - 14)} {_n(CX)},"
                     f"{_n(t - 21)} Z"),
        Mark("line", f"M {_n(CX + 4.5)},{_n(t - 7)} C {_n(CX + 4.5)},{_n(t - 4.5)} "
                     f"{_n(CX + 3)},{_n(t - 3)} {_n(CX + 1)},{_n(t - 3)}"),
    ]


def moon_star(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX + 2)},{_n(t - 20)} C {_n(CX - 10)},{_n(t - 20)} "
                     f"{_n(CX - 12)},{_n(t + 0.5)} {_n(CX + 3)},{_n(t + 0.5)} C {_n(CX - 5)},"
                     f"{_n(t - 2)} {_n(CX - 6)},{_n(t - 17)} {_n(CX + 2)},{_n(t - 20)} Z"),
        Mark("body", star(CX + 5, t - 10, 5, 2.1, 5)),
    ]


def fish(top: float) -> Part:
    t = top
    return [
        Mark("body", f"M {_n(CX - 12)},{_n(t - 8)} C {_n(CX - 8)},{_n(t - 15)} {_n(CX + 2)},"
                     f"{_n(t - 15)} {_n(CX + 6)},{_n(t - 8)} L {_n(CX + 12)},{_n(t - 13)} "
                     f"L {_n(CX + 12)},{_n(t - 1)} L {_n(CX + 6)},{_n(t - 6)} C {_n(CX + 2)},"
                     f"{_n(t + 0.5)} {_n(CX - 8)},{_n(t + 0.5)} {_n(CX - 12)},{_n(t - 7)} Z"),
        Mark("dot", circle(CX - 7, t - 8.5, 1.2)),
        Mark("line", f"M {_n(CX - 3)},{_n(t - 12)} C {_n(CX - 1)},{_n(t - 9)} {_n(CX - 1)},"
                     f"{_n(t - 6)} {_n(CX - 3)},{_n(t - 3)}"),
    ]


def snowflake(top: float) -> Part:
    return [Mark("body", star(CX, top - 10, 10.5, 2.6, 6))]


def _path(*steps: object) -> str:
    """Path data from commands and (x, y) points given as offsets from (CX, top)."""
    out = []
    for step in steps:
        if isinstance(step, str):
            out.append(step)
        else:
            x, y = step  # type: ignore[misc]
            out.append(f"{_n(x)},{_n(y)}")
    return " ".join(out)


def elephant(top: float) -> Part:
    """An elephant's head, facing out: wide ears behind a domed brow, the trunk hanging
    to the collar, two tusks curving out. The alfil, which began as the elephant."""
    t = top

    def p(x: float, y: float) -> tuple[float, float]:
        return (CX + x, t + y)

    ears = [
        Mark("body", _path("M", p(-4, -15.5), "C", p(-9, -19.5), p(-13.5, -16.5), p(-13, -11),
                           "C", p(-12.5, -6.5), p(-8.5, -4.5), p(-4.5, -8), "Z")),
        Mark("body", _path("M", p(4, -15.5), "C", p(9, -19.5), p(13.5, -16.5), p(13, -11),
                           "C", p(12.5, -6.5), p(8.5, -4.5), p(4.5, -8), "Z")),
        Mark("line", _path("M", p(-5.5, -14), "C", p(-9, -16.5), p(-11.5, -13.5),
                           p(-10.5, -10), "M", p(5.5, -14), "C", p(9, -16.5),
                           p(11.5, -13.5), p(10.5, -10))),
    ]
    head = [
        Mark("body", _path(
            "M", p(-2, -1.5), "C", p(-2, -4), p(-2.5, -6.5), p(-4, -9),
            "C", p(-6.5, -11.5), p(-6.5, -17), p(0, -18.5),
            "C", p(6.5, -17), p(6.5, -11.5), p(4, -9),
            "C", p(2.5, -6.5), p(2, -4), p(2, -1.5),
            "Q", p(2.5, 0), p(3.5, 0.5), "L", p(-3.5, 0.5), "Q", p(-2.5, 0), p(-2, -1.5),
            "Z")),
        # The cheeks' seam, where the head lies over the ears: the white line that keeps
        # a black elephant from reading as one blot, as BASE_SEAM does for the base.
        Mark("line", _path("M", p(-2.4, -6), "C", p(-2.8, -7.2), p(-3.3, -8.2), p(-4, -9),
                           "C", p(-5.5, -10.5), p(-6.2, -13), p(-5.8, -15),
                           "M", p(2.4, -6), "C", p(2.8, -7.2), p(3.3, -8.2), p(4, -9),
                           "C", p(5.5, -10.5), p(6.2, -13), p(5.8, -15))),
    ]
    tusk_left = _path("M", p(-3, -8.5), "C", p(-3.5, -5.5), p(-5.5, -3.5), p(-8, -4.5),
                      "C", p(-6.5, -5.5), p(-5.5, -7), p(-5, -9.5), "Z")
    tusk_right = _path("M", p(3, -8.5), "C", p(3.5, -5.5), p(5.5, -3.5), p(8, -4.5),
                       "C", p(6.5, -5.5), p(5.5, -7), p(5, -9.5), "Z")
    # A tusk is ivory on either seat: the body in the seat's colour, its edge a detail line
    # inside the outline, so a black piece's tusks still read as tusks.
    tusks = [Mark("body", tusk_left), Mark("body", tusk_right),
             Mark("line", tusk_left), Mark("line", tusk_right)]
    eyes = [Mark("dot", circle(CX - 2.6, t - 12.5, 0.9)),
            Mark("dot", circle(CX + 2.6, t - 12.5, 0.9))]
    return ears + head + tusks + eyes


def camel(top: float) -> Part:
    """A camel standing on the collar, seen from the side and facing left: one hump, the
    long neck raised, the head held level."""
    t = top

    def p(x: float, y: float) -> tuple[float, float]:
        return (CX + x, t + y)

    return [
        Mark("body", _path(
            "M", p(-5, 0.5), "L", p(-5.2, -5.5),
            "C", p(-7.2, -7), p(-8.2, -10), p(-8.7, -13),
            "L", p(-11.7, -13.5), "C", p(-13.5, -13.8), p(-13.5, -16.2), p(-11.7, -16.5),
            "C", p(-10.7, -17.8), p(-9.2, -18.5), p(-7.9, -18),
            "L", p(-7.5, -19.8), "L", p(-6.5, -17.5),
            "C", p(-5.7, -16), p(-5.2, -13), p(-3.2, -11.5),
            "C", p(-1.7, -17.5), p(4.5, -18), p(6, -11.5),
            "C", p(7.8, -10.5), p(8, -8.5), p(7, -5.5),
            "L", p(6, 0.5), "L", p(3.8, 0.5), "L", p(3.8, -4.5),
            "L", p(-2.6, -4.5), "L", p(-2.8, 0.5), "Z")),
        Mark("dot", circle(CX - 10, t - 16, 0.9)),
    ]


def wheel(top: float) -> Part:
    """A ring with a plus across it, the wazir's plus in a disc as the fairy fonts drew it
    (and the astronomers' earth), the disc opened between the arms so its outline is its
    own: the ferz is a whole disc with a cross on it."""
    big, small, w = 9.5, 6.5, 1.5
    c = top - big + 0.5
    reach = small + 0.5
    return [
        Mark("body", circle(CX, c, big) + " " + circle(CX, c, small), 'fill-rule="evenodd"'),
        Mark("body", poly([
            (CX - w, c - reach), (CX + w, c - reach), (CX + w, c - w), (CX + reach, c - w),
            (CX + reach, c + w), (CX + w, c + w), (CX + w, c + reach), (CX - w, c + reach),
            (CX - w, c + w), (CX - reach, c + w), (CX - reach, c - w), (CX - w, c - w),
        ])),
    ]


# Cburnett's knight head, as it is drawn in Chess_nlt45.svg (CC BY-SA 3.0): the neck and
# the head, an eye and a nostril. A head that uses it makes its drawing his work
# rearranged, credited to him under his licence (`Drawing.cburnett`).
KNIGHT_NECK = "M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"
KNIGHT_HEAD = (
    "M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 "
    "12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 "
    "C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 "
    "L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"
)


def _nostril() -> str:
    """Cburnett's nostril, an ellipse turned 30 degrees, as a polygon."""
    a = math.radians(30)
    pts = []
    for i in range(16):
        u = 2 * math.pi * i / 16
        x, y = 1.25 * math.cos(u), 2.25 * math.sin(u)
        pts.append((14.5 + x * math.cos(a) - y * math.sin(a), 15.5 + x * math.sin(a)
                    + y * math.cos(a)))
    # The ellipse is placed by Cburnett's matrix(0.866,0.5,-0.5,0.866,9.693,-5.173).
    return poly([(0.866 * x - 0.5 * y + 9.693, 0.5 * x + 0.866 * y - 5.173) for x, y in pts])


def _knight(top: float, k: float, before: Part = (), after: Part = ()) -> Part:
    """The knight's head standing on `top`, `k` of its own size, with marks drawn in its
    own coordinates behind (`before`) and over (`after`) it."""
    dx, dy = CX - 26.5 * k, top + 0.5 - 39 * k

    def place(m: Mark) -> Mark:
        return Mark(m.kind, m.d, m.extra, k, dx, dy)

    return [place(m) for m in (
        list(before)
        + [Mark("body", KNIGHT_NECK), Mark("body", KNIGHT_HEAD),
           Mark("dot", circle(9, 25.5, 1.25)), Mark("dot", _nostril())]
        + list(after)
    )]


def knight_plain(top: float) -> Part:
    return _knight(top, 0.6)


def knight_striped(top: float) -> Part:
    stripes = [Mark("line", "M 25,13.5 L 29,19"), Mark("line", "M 30,21 L 35,24"),
               Mark("line", "M 24,28 L 36,30"), Mark("line", "M 19.5,35 L 37,35.5")]
    return _knight(top, 0.6, after=stripes)


def knight_horned(top: float) -> Part:
    horn = [Mark("body", poly([(12.5, 12.5), (1.5, -1.5), (16, 9.5)]))]
    return _knight(top, 0.6, before=horn)


def knight_winged(top: float) -> Part:
    wing = [Mark("body", poly([(26, 20), (36, 2), (39, 10), (46, 6), (44, 16), (50, 15),
                              (44, 25), (37, 28)]))]
    return _knight(top, 0.6, before=wing)


def knight_moon(top: float) -> Part:
    moon = [Mark("body", "M 30,-2 C 46,-2 50,22 34,28 C 42,20 41,4 30,-2 Z")]
    return _knight(top, 0.6, before=moon)


def knight_crowned(top: float) -> Part:
    crest = [Mark("body", poly([(13.5, 8.5), (12, -1), (16, 3.5), (18.5, -3), (21, 3.5),
                               (25.5, -1), (23, 8.5)]))]
    return _knight(top, 0.5, before=crest)


KNIGHT_HEADS = ("knight", "knight-striped", "knight-horned", "knight-winged",
                "knight-crowned", "knight-moon")


HEADS: dict[str, Head] = {
    "knight": knight_plain, "knight-striped": knight_striped,
    "knight-horned": knight_horned, "knight-winged": knight_winged,
    "knight-crowned": knight_crowned, "knight-moon": knight_moon,
    "sword": sword, "hammer": hammer, "flag": flag, "axe": axe, "key": key,
    "mushroom": mushroom, "duck": duck, "bomb": bomb, "crown": crown, "gem": gem,
    "fir": fir, "ghost": ghost, "drop": drop, "moon-star": moon_star, "fish": fish,
    "snowflake": snowflake, "elephant": elephant, "camel": camel,
    "wheel": wheel,
    "ball": _ball(5.0), "pawnball": _ball(4.6), "star5": _star(5, 9.5, 4.0),
    "star4": _star(4, 9.5, 3.2), "star6": _star(6, 9.0, 5.0), "star8": _star(8, 9.5, 5.5),
    "trefoil": trefoil, "triangle": triangle, "diamond": diamond, "block": block,
    "ring": ring, "cross": cross, "saltire": saltire, "crescent-up": crescent_up,
    "spear": spear, "trident": trident, "flame": flame, "heart": heart, "bell": bell,
    "hourglass": hourglass, "wings": wings, "sun": sun, "eye": eye, "arrow": arrow,
    "chevron": chevron, "bolt": bolt, "leaf": leaf, "shield": shield, "anchor": anchor,
    "rose": rose, "acorn": acorn, "horns": horns, "bull": bull, "frog": frog,
    "coronet-small": coronet_small, "orb": orb, "helmet": helmet, "cap": cap,
    "onion": onion, "twin": twin, "cube": cube, "tower-small": tower_small,
    "none": lambda top: [],
}


# ------------------------------------------------------------------ the pieces

@dataclass(frozen=True)
class Drawing:
    """One piece in the library: what it is called, how it is drawn, how it is found."""

    name: str
    category: str
    body: str
    head: str
    says: str  # one line for the palette and the credits: what the drawing shows
    words: tuple[str, ...] = ()  # more names a designer might type
    scale: float = 1.0
    parts: tuple[str, ...] = field(default=())  # anything drawn beside body and head

    @property
    def cburnett(self) -> bool:
        """Whether the drawing uses Cburnett's own knight head, and so keeps his licence."""
        return self.head in KNIGHT_HEADS


CATEGORIES = ("leapers", "riders", "compounds", "royals", "pawns", "oddities")

DRAWN: dict[str, Drawing] = {}


def piece(slug: str, *args: object, **kwargs: object) -> None:
    DRAWN[slug] = Drawing(*args, **kwargs)  # type: ignore[arg-type]


# Leapers: an emblem on the stem.
piece("threeleaper", "threeleaper", "leapers", "stem", "trefoil",
      "a trefoil on a stem", ("(3,0) leaper", "three squares straight", "H"))
piece("tripper", "tripper", "leapers", "stem", "triangle",
      "a triangle on a stem", ("(3,3) leaper", "three squares diagonal", "G"), scale=0.95)
piece("kirin", "kirin", "leapers", "stem", "flame",
      "a flame on a stem", ("ferz and dabbaba", "fd"))
piece("phoenix", "phoenix", "leapers", "stem", "wings",
      "spread wings on a stem", ("wazir and alfil", "wa"))
piece("frog", "frog", "leapers", "stem", "frog",
      "a frog's head on a stem", ("ferz and threeleaper", "fh"))
piece("squirrel", "squirrel", "leapers", "stem", "acorn",
      "an acorn on a stem", ("dabbaba, alfil and knight", "adn"))
piece("antelope", "antelope", "leapers", "stem", "horns",
      "long horns on a stem", ("(3,4) leaper", "antilope"), scale=0.95)
piece("bison", "bison", "leapers", "stem", "bull",
      "a bull's head on a stem", ("camel and zebra", "cz"))
piece("waffle", "waffle", "leapers", "stem", "block",
      "a square on a stem", ("wazir and alfil", "pheonix", "wa"))
piece("fad", "fad", "leapers", "stem", "diamond",
      "a diamond on a stem", ("ferz, alfil and dabbaba", "fad"), scale=1.12)
piece("hopper", "chevron", "leapers", "stem", "chevron",
      "a chevron on a stem", ("jumper", "leaper"))
piece("star-leaper", "star leaper", "leapers", "stem", "star5",
      "a five-pointed star on a stem", ("star", "general leaper"), scale=0.88)
# The three the fairy fonts drew (Quivira's glyphs until Q557b), redrawn in the grammar.
piece("alfil", "alfil", "leapers", "stem", "elephant",
      "an elephant's head on a stem", ("elephant", "(2,2) leaper", "khon", "A"))
piece("camel", "camel", "leapers", "stem", "camel",
      "a camel on a stem", ("(1,3) leaper", "dromedary", "L"))
piece("wazir", "wazir", "leapers", "stem", "wheel",
      "a ring crossed by a plus, on a stem", ("one orthogonal step", "vizier", "W"),
      scale=1.06)

# Riders: an emblem on the tall shaft.
piece("spear", "spear", "riders", "shaft", "spear",
      "a spearhead on a shaft", ("lance", "forward rider", "western lance"), scale=0.88)
piece("trident", "trident", "riders", "shaft", "trident",
      "a trident on a shaft", ("three-way rider", "sea"))
piece("rose", "rose", "riders", "shaft", "rose",
      "a rose on a shaft", ("circular knight rider", "qq"), scale=0.88)
piece("grasshopper", "grasshopper", "riders", "shaft", "chevron",
      "a chevron on a shaft", ("hopper", "g", "jumps one piece"))
piece("arrow", "arrow", "riders", "shaft", "arrow",
      "an arrowhead on a shaft", ("archer", "forward", "shooter"))
piece("bolt", "lightning", "oddities", "plinth", "bolt",
      "a lightning bolt", ("bolt", "fast", "zigzag", "storm"))
piece("ring-rider", "ring", "riders", "shaft", "ring",
      "a ring on a shaft", ("circle", "orbit", "loop"), scale=1.12)
piece("saltire-rider", "saltire", "riders", "shaft", "saltire",
      "a diagonal cross on a shaft", ("x", "diagonal rider", "cross"))

# Compounds: a rider's top with a leaper's emblem rising from it.
piece("crowned-rook", "crowned rook", "compounds", "battlements", "coronet-small",
      "the rook's battlements wearing a small crown",
      ("dragon king", "rook and king", "rook and ferz", "promoted rook"), scale=1.06)
piece("crowned-bishop", "crowned bishop", "compounds", "mitre", "crown",
      "the bishop's mitre wearing a small crown",
      ("dragon horse", "bishop and king", "bishop and wazir", "promoted bishop"), scale=0.8)
piece("queen-star", "queen and star", "compounds", "coronet", "star6",
      "the queen's coronet with a six-pointed star",
      ("queen plus", "empress plus", "super queen"), scale=0.8)
piece("rook-diamond", "rook and diamond", "compounds", "battlements", "diamond",
      "the rook's battlements with a diamond",
      ("rook and alfil", "rook and fad"), scale=1.12)

# Royals: the king's barrel, a crown of its own.
piece("prince", "prince", "royals", "barrel", "coronet-small",
      "a wide body with a small crown", ("crown prince", "heir", "royal commoner"))
piece("emperor", "emperor", "royals", "barrel", "cross",
      "a wide body with a broad cross", ("monarch", "regent", "royal"))
piece("shield-king", "warden", "royals", "barrel", "shield",
      "a wide body with a shield", ("guard", "shield", "defender"), scale=0.95)

# Pawns: the low skirt, a small head.
piece("soldier", "soldier", "pawns", "skirt", "helmet",
      "a pawn in a crested helmet", ("footman", "infantry", "western soldier"), scale=0.8)
piece("sergeant", "sergeant", "pawns", "skirt", "cap",
      "a pawn in a flat cap with a chevron", ("drill pawn", "corporal"))
piece("berolina", "berolina pawn", "pawns", "skirt", "onion",
      "a pawn with a pointed head", ("berolina", "diagonal pawn"), scale=0.88)

# Oddities: a picture on the base.
piece("anchor", "anchor", "oddities", "plinth", "anchor",
      "an anchor", ("immobile", "stuck", "fixed"))
piece("hourglass", "hourglass", "oddities", "plinth", "hourglass",
      "an hourglass", ("timer", "clock", "sand"))
piece("bell", "bell", "oddities", "plinth", "bell",
      "a bell", ("alarm", "signal"), scale=1.2)
piece("leaf", "leaf", "oddities", "plinth", "leaf",
      "a leaf", ("plant", "green", "growth"))
piece("heart", "heart", "oddities", "plinth", "heart",
      "a heart", ("life", "love"))
piece("eye", "eye", "oddities", "plinth", "eye",
      "an eye", ("watcher", "sight", "spy"))
piece("flame", "flame", "oddities", "plinth", "flame",
      "a flame", ("fire", "burn", "bomb"))
piece("duck", "duck", "oddities", "plinth", "duck",
      "a duck", ("duck chess", "blocker", "bird"), scale=0.95)
piece("bomb", "bomb", "oddities", "plinth", "bomb",
      "a bomb with a lit fuse", ("explode", "atomic", "kamikaze"), scale=0.95)
piece("crown", "crown", "oddities", "plinth", "crown",
      "a crown", ("royal", "throne", "regalia"))
piece("tree", "tree", "oddities", "plinth", "fir",
      "a fir tree", ("forest", "obstacle", "fir"), scale=1.06)
piece("ghost", "ghost", "oddities", "plinth", "ghost",
      "a ghost", ("phantom", "spirit", "passes through"), scale=1.28)
piece("drop", "drop", "oddities", "plinth", "drop",
      "a drop of water", ("water", "rain", "lake"))
piece("hammer", "hammer", "oddities", "plinth", "hammer",
      "a hammer", ("smith", "builder", "breaker"))

# The knight's family: Cburnett's own knight head on the grammar's bodies, which makes
# these his work rearranged (`Drawing.cburnett`). They replace Commons' zebra, nightrider,
# unicorn and dragon, each a whole knight with a mark on it, which the sort test could not
# tell from the knight at 24 px (plan-visual.md §7's amendment).
piece("zebra", "zebra", "leapers", "stem", "knight-striped",
      "a striped knight's head on a stem", ("(2,3) leaper", "z"))
piece("nightrider", "nightrider", "riders", "shaft", "knight-moon",
      "a knight's head before a crescent moon, on a shaft", ("knight rider", "nn"), scale=0.8)
piece("unicorn", "unicorn", "riders", "shaft", "knight-horned",
      "a horned knight's head on a shaft", ("bishop and nightrider", "bnn"), scale=0.8)
piece("dragon", "dragon", "leapers", "stem", "knight-winged",
      "a winged knight's head on a stem", ("wyvern", "drake"), scale=0.95)
piece("royal-knight", "royal knight", "royals", "barrel", "knight-crowned",
      "a wide body with a crowned knight's head", ("royal knight", "knight king"), scale=0.8)

# More riders, leapers and royals.
piece("sword", "sword", "riders", "shaft", "sword",
      "a sword on a shaft", ("blade", "swordsman", "duelist"), scale=0.95)
piece("banner", "banner", "riders", "shaft", "flag",
      "a banner on a shaft", ("flag", "herald", "standard"), scale=0.95)
piece("key", "key", "riders", "shaft", "key",
      "a key on a shaft", ("gate", "unlock", "keeper"), scale=0.95)
piece("mushroom", "mushroom", "leapers", "stem", "mushroom",
      "a mushroom on a stem", ("toad", "spore", "fungus"))
piece("fish", "fish", "leapers", "stem", "fish",
      "a fish on a stem", ("swimmer", "carp", "sea"))
piece("night", "night", "leapers", "stem", "moon-star",
      "a crescent and a star on a stem", ("moon", "star", "night leaper"), scale=0.95)
piece("gem", "gem", "leapers", "stem", "gem",
      "a cut gem on a stem", ("jewel", "diamond", "treasure"))
piece("snowflake", "snowflake", "leapers", "stem", "snowflake",
      "a snowflake on a stem", ("ice", "frost", "six ways"), scale=0.88)
piece("axe", "axe", "leapers", "stem", "axe",
      "an axe on a stem", ("berserker", "woodcutter", "executioner"), scale=0.95)


# ------------------------------------------------------------------ writing the files

# How large a head is drawn on each body, so it fills the room the body leaves it: the
# heads are drawn about 18 units tall.
HEAD_SCALE = {"stem": 1.3, "shaft": 1.15, "barrel": 1.15, "skirt": 1.0, "plinth": 1.4,
              "battlements": 0.9, "mitre": 0.7, "coronet": 0.95}


def scaled(part: Part, scale: float, top: float) -> Part:
    """A head grown about the point it stands on."""
    if abs(scale - 1.0) < 1e-9:
        return part
    ox, oy = CX * (1 - scale), top * (1 - scale)
    return [
        Mark(m.kind, m.d, m.extra, m.k * scale, ox + scale * m.dx, oy + scale * m.dy)
        for m in part
    ]


def marks(drawing: Drawing) -> Part:
    body, top = BODIES[drawing.body]()
    head = scaled(HEADS[drawing.head](top), HEAD_SCALE[drawing.body] * drawing.scale, top)
    return BASE + [BASE_SEAM] + body + head


def paint(mark: Mark, seat: str) -> str:
    fill, detail = SEATS[seat]
    width = _n(STROKE / mark.k)
    if mark.kind == "body":
        style = f'fill="{fill}" stroke="{INK}"'
    elif mark.kind == "line":
        style = f'fill="none" stroke="{detail}"'
    elif mark.kind == "dot":
        style = f'fill="{detail}" stroke="none"'
    elif mark.kind == "solid":
        style = f'fill="{INK}" stroke="{INK}"'
    else:
        raise ValueError(mark.kind)
    extra = (" " + mark.extra) if mark.extra else ""
    if abs(mark.k - 1.0) > 1e-9 or mark.dx or mark.dy:
        extra += (f' transform="translate({_n(mark.dx)},{_n(mark.dy)}) '
                  f'scale({mark.k:.4f})"')
    return f'  <path d="{mark.d}" {style} stroke-width="{width}"{extra}/>'


def licence_words(drawing: Drawing) -> str:
    """What the file says of itself in its opening comment."""
    if drawing.cburnett:
        return ("Drawn for the chess variant sandbox (assets/pieces/draw.py) from "
                "Cburnett's knight (Wikimedia Commons, CC BY-SA 3.0); released under the "
                "same licence.")
    return ("Drawn for the chess variant sandbox in Cburnett's grammar "
            "(assets/pieces/draw.py) and released CC0.")


def svg(slug: str, seat: str) -> str:
    drawing = DRAWN[slug]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- {drawing.name}: {drawing.says}. {licence_words(drawing)} -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FRAME}" height="{FRAME}" '
        f'viewBox="0 0 {FRAME} {FRAME}">',
        '<g stroke-linecap="round" stroke-linejoin="round" fill-rule="nonzero">',
    ]
    lines += [paint(m, seat) for m in marks(drawing)]
    lines += ["</g>", "</svg>", ""]
    return "\n".join(lines)


def main() -> int:
    SVG_DIR.mkdir(exist_ok=True)
    for slug in sorted(DRAWN):
        for seat in SEATS:
            (SVG_DIR / f"{slug}-{seat}.svg").write_text(svg(slug, seat), encoding="utf-8",
                                                       newline="\n")
    print(f"wrote {len(DRAWN) * 2} files ({len(DRAWN)} pieces) to {SVG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
