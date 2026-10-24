"""Check that the library sorts -- no two western pieces alike in silhouette at 24 px,
every black piece legible on the dark square (plan-visual.md §7) -- and, with `frames`,
that every icon actually draws inside its own frame.

    uv run python assets/pieces/check.py           # the sort and the contrast (resvg)
    uv run python assets/pieces/check.py frames    # and the frames, in a headless browser

Half the source files declare no ``viewBox``, so ``fetch.py`` derives one from
``width``/``height``. That is right only if the art was authored at the origin,
and nothing else in the pipeline would notice if it were not: the files parse,
the licences check out, the manifest looks sane, and the page still renders
something. A file that parses is not a file that draws. This measures what a
browser actually paints -- the bounding box of the art against the frame it is
supposed to fill -- and flags anything clipped, empty, or lost in a corner.

It is deliberately a separate script rather than part of ``fetch.py``: it needs a
browser, and the fetch has to work on a machine without one.

The frame check needs a Chrome or Edge binary. Without one it reports that it could not
check rather than passing silently. The sort and the contrast need resvg (the `site`
extra) and the package's tokens, so the script is run with `uv run`;
`tests/test_piece_library.py` pins both.
"""

# compute-lease: heavy when frames

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
]

# Fraction of the frame the art must cover, and how far outside it may stray.
# The bounds are loose: they are meant to catch art that is clipped or invisible,
# not to police composition.
MIN_COVERAGE = 0.25
MAX_OVERFLOW = 0.02

PROBE = """<meta charset="utf-8"><body><script>
const SOURCES = %s;
const FILES = Object.keys(SOURCES);
async function bbox(file) {
  const text = SOURCES[file];
  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:0;top:0;width:400px;height:400px';
  host.innerHTML = text;
  document.body.appendChild(host);
  const svg = host.querySelector('svg');
  if (!svg) { host.remove(); return {error: 'no <svg> element'}; }
  const vb = svg.getAttribute('viewBox');
  let frame;
  if (vb) {
    const n = vb.split(/[\\s,]+/).map(Number);
    frame = {x: n[0], y: n[1], w: n[2], h: n[3]};
  } else {
    frame = {x: 0, y: 0, w: parseFloat(svg.getAttribute('width')),
             h: parseFloat(svg.getAttribute('height'))};
  }
  let box;
  try { box = svg.getBBox(); } catch (e) { host.remove(); return {error: String(e)}; }
  host.remove();
  return {frame, box: {x: box.x, y: box.y, w: box.width, h: box.height}};
}
(async () => {
  const out = {};
  for (const f of FILES) {
    try { out[f] = await bbox(f); } catch (e) { out[f] = {error: String(e)}; }
  }
  document.title = 'DONE';
  const pre = document.createElement('pre');
  pre.id = 'result';
  pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script></body>"""


# ----------------------------------------------------------------- the sort test (§7)
#
# What `docs/plan-visual.md` §7 asks of the library, read without a browser: resvg (the
# `site` extra, the same renderer the link pictures use) rasterises each file, and a small
# PNG reader here turns the answer into pixels. The rule was fixed in the plan before
# anything was measured: a silhouette is a piece's white form at 24 x 24 px, a pixel
# counted where it is more than half opaque; two silhouettes overlap by intersection over
# union in their shared frame; and the threshold is the largest overlap between any two of
# Cburnett's six standard pieces, the familiar set's own worst pair, which every chess
# player already tells apart. No two western pieces may overlap above it.

SILHOUETTE_PX = 24
STANDARD = ("king", "queen", "rook", "bishop", "knight", "pawn")
# Families told apart by the character on a tile or a disc, one outline by design (§7
# *families kept apart*): not sorted against each other, and not against the western set.
OWN_FAMILIES = ("shogi-", "xiangqi-")
# The floor a black piece's body keeps against the dark square under its worst streak.
BODY_FLOOR = 4.5


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def pixels(png: bytes) -> tuple[int, int, list[bytes]]:
    """An 8-bit RGBA PNG (what resvg writes) as its width, height and rows of RGBA bytes."""
    import struct
    import zlib

    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    at, data, width, height = 8, b"", 0, 0
    while at < len(png):
        (length,) = struct.unpack(">I", png[at : at + 4])
        kind = png[at + 4 : at + 8]
        body = png[at + 8 : at + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, colour, interlace) != (8, 6, 0):
                raise ValueError("expected 8-bit RGBA, not interlaced")
        elif kind == b"IDAT":
            data += body
        at += 12 + length
    raw = zlib.decompress(data)
    stride, rows, prev = width * 4, [], bytes(width * 4)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1 : start + 1 + stride])
        for i in range(stride):
            left = line[i - 4] if i >= 4 else 0
            up = prev[i]
            corner = prev[i - 4] if i >= 4 else 0
            add = (0, left, up, (left + up) // 2, _paeth(left, up, corner))[kind]
            line[i] = (line[i] + add) & 0xFF
        rows.append(bytes(line))
        prev = bytes(line)
    return width, height, rows


def sized(stem: str, px: int, manifest: dict | None = None) -> str:
    """One file's markup, drawn at `px` x `px` and scaled to the frame its manifest row
    names. The opening tag keeps its namespace declarations (resvg reads XML, and an
    Inkscape file's `inkscape:` attributes need theirs) and loses only its size and frame."""
    import re

    manifest = manifest or json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest[stem]
    text = (ROOT / entry["file"]).read_text(encoding="utf-8", errors="replace")
    opening = re.search(r"<svg\b[^>]*>", text, flags=re.S)
    if opening is None:
        raise ValueError(f"{stem}: no <svg> element")
    tag = re.sub(
        r'\s(?:width|height|viewBox|preserveAspectRatio)\s*=\s*"[^"]*"', "", opening.group(0)
    )
    tag = tag.replace(
        "<svg", f'<svg width="{px}" height="{px}" viewBox="{entry["viewBox"]}"', 1
    )
    return text[: opening.start()] + tag + text[opening.end() :]


def render(stem: str, px: int, manifest: dict | None = None) -> list[bytes]:
    """One file at `px` x `px` as rows of RGBA bytes, through resvg."""
    import resvg_py

    text = sized(stem, px, manifest)
    width, height, rows = pixels(bytes(resvg_py.svg_to_bytes(svg_string=text)))
    if (width, height) != (px, px):
        raise ValueError(f"{stem}: rendered {width}x{height}, not {px}x{px}")
    return rows


def silhouette(stem: str, manifest: dict | None = None) -> int:
    """The piece's filled outline at 24 px as a bit mask, one bit a pixel."""
    mask = 0
    for y, row in enumerate(render(stem, SILHOUETTE_PX, manifest)):
        for x in range(SILHOUETTE_PX):
            if row[x * 4 + 3] > 127:
                mask |= 1 << (y * SILHOUETTE_PX + x)
    return mask


_FULL = (1 << SILHOUETTE_PX * SILHOUETTE_PX) - 1
# Every pixel but the last column, and every pixel but the first: a shift one pixel along
# a row must not wrap into the next row.
_NOT_LAST = sum(
    1 << (y * SILHOUETTE_PX + x) for y in range(SILHOUETTE_PX) for x in range(SILHOUETTE_PX - 1)
)
_NOT_FIRST = _NOT_LAST << 1


def _shift(mask: int, dx: int, dy: int) -> int:
    """The mask moved by (dx, dy) pixels, losing what leaves the frame."""
    if dx > 0:
        mask = (mask & _NOT_LAST) << 1
    elif dx < 0:
        mask = (mask & _NOT_FIRST) >> 1
    if dy > 0:
        mask = mask << SILHOUETTE_PX
    elif dy < 0:
        mask = mask >> SILHOUETTE_PX
    return mask & _FULL


def contour(mask: int) -> int:
    """The silhouette's pixels with a 4-neighbour outside it (the frame's edge counts)."""
    inner = mask
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= _shift(mask, dx, dy)
    return mask & ~inner


def _near(mask: int) -> int:
    """The mask grown by one pixel in all eight directions."""
    out = mask
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out |= _shift(mask, dx, dy)
    return out


def overlap(one: int, two: int) -> float:
    """How far two silhouettes' outlines coincide: the boundary F-score at one pixel.

    The fraction of each contour that has the other's within a pixel, and their
    harmonic mean. Plan-visual §7's amendment says why the outlines and not the areas:
    the area overlap of any two squat pieces centred in one frame is high.
    """
    a, b = contour(one), contour(two)
    if not a or not b:
        return 1.0 if a == b else 0.0
    p = (a & _near(b)).bit_count() / a.bit_count()
    r = (b & _near(a)).bit_count() / b.bit_count()
    return 2 * p * r / (p + r) if p + r else 0.0


def pieces(manifest: dict) -> list[str]:
    """The western family's pieces, each once: its white form (or its only one)."""
    found = []
    for stem in sorted(manifest):
        if stem.startswith(OWN_FAMILIES) or stem.endswith("-black"):
            continue
        found.append(stem)
    return found


def threshold(masks: dict[str, int]) -> float:
    """The familiar set's worst pair: the most two standard pieces overlap."""
    return max(
        overlap(masks[a + "-white"], masks[b + "-white"])
        for i, a in enumerate(STANDARD)
        for b in STANDARD[i + 1 :]
    )


def sort_test(manifest: dict | None = None) -> tuple[float, list[tuple[float, str, str]]]:
    """The threshold, and every pair of western pieces at or above it, worst first.

    A pair *at* the threshold is the standard set's own worst pair, which is the rule's
    anchor, so it is reported only when it is not that pair.
    """
    manifest = manifest or json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    stems = pieces(manifest)
    masks = {stem: silhouette(stem, manifest) for stem in stems}
    limit = threshold(masks)
    pairs = []
    for i, a in enumerate(stems):
        for b in stems[i + 1 :]:
            value = overlap(masks[a], masks[b])
            if value > limit:
                pairs.append((value, a, b))
    return limit, sorted(pairs, reverse=True)


def luminance(rgb: tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def body(stem: str, manifest: dict | None = None) -> tuple[float, float, float]:
    """A piece's body colour: the median, by luminance, of its fully opaque pixels at 45 px."""
    opaque = [
        (row[i] / 255, row[i + 1] / 255, row[i + 2] / 255)
        for row in render(stem, 45, manifest)
        for i in range(0, len(row), 4)
        if row[i + 3] == 255
    ]
    if not opaque:
        raise ValueError(f"{stem}: no opaque pixel")
    return sorted(opaque, key=luminance)[len(opaque) // 2]


def ratio(one: tuple[float, float, float], two: tuple[float, float, float]) -> float:
    a, b = sorted((luminance(one), luminance(two)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def find_browser() -> Path | None:
    return next((p for p in BROWSERS if p.exists()), None)


def measure(browser: Path, files: list[str]) -> dict:
    # The sources are embedded rather than fetched: a file:// page cannot fetch
    # siblings, and inlining keeps the probe independent of where it is written.
    sources = {f: (ROOT / f).read_text(encoding="utf-8", errors="replace") for f in files}
    payload = json.dumps(sources).replace("</", "<\\/")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "probe.html"
        page.write_text(PROBE % payload, encoding="utf-8")
        # --dump-dom runs the page and prints the settled DOM, which carries the
        # <pre> the probe fills in.
        proc = subprocess.run(
            [
                str(browser),
                "--headless",
                "--disable-gpu",
                "--virtual-time-budget=15000",
                "--allow-file-access-from-files",
                "--dump-dom",
                page.as_uri(),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
    html = proc.stdout
    start = html.find('<pre id="result">')
    if start < 0:
        raise SystemExit("probe did not produce a result; browser said:\n" + proc.stderr[-2000:])
    start = html.index(">", start) + 1
    end = html.index("</pre>", start)
    return json.loads(html[start:end].replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))


def worst_dark_squares() -> dict[str, tuple[float, float, float]]:
    """The board's dark square under its worst streak, in each theme: the darkest a black
    piece's ground can be (`tests/test_style.py`'s `_grained`, read the same way)."""
    from chess_variant_sandbox.web.page import tokens

    out = {}
    for name, theme in (("light", tokens.LIGHT), ("dark", {**tokens.LIGHT, **tokens.DARK})):
        digits = theme["sq-dk"].lstrip("#")
        square = [int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        spec = tokens.GRAIN_DK
        alpha = float(theme["grain-dk-a"]) * tokens.grain_most(spec)
        r, g, b = (alpha * c + (1 - alpha) * s for c, s in zip(spec[1], square, strict=True))
        out[name] = (r, g, b)
    return out


def contrast(manifest: dict) -> list[tuple[float, str, str]]:
    """Every western black piece's body against the worst dark square, per theme."""
    grounds = worst_dark_squares()
    rows = []
    for stem in sorted(manifest):
        if stem.startswith(OWN_FAMILIES) or not stem.endswith("-black"):
            continue
        colour = body(stem, manifest)
        for theme, ground in grounds.items():
            rows.append((ratio(colour, ground), stem, theme))
    return rows


def sort_and_contrast(manifest: dict) -> bool:
    limit, pairs = sort_test(manifest)
    print(f"sorted {len(pieces(manifest))} western pieces at {SILHOUETTE_PX} px; "
          f"threshold {limit:.3f} (the standard set's worst pair)")
    for value, a, b in pairs:
        print(f"  too alike: {a} and {b} ({value:.3f})")
    rows = contrast(manifest)
    low = [row for row in rows if row[0] < BODY_FLOOR]
    print(f"black bodies on the worst dark square: the least is {min(rows)[0]:.2f} : 1 "
          f"({min(rows)[1]}, {min(rows)[2]} theme)")
    for value, stem, theme in low:
        print(f"  {stem} reads {value:.2f} : 1 on the dark square ({theme} theme)")
    return not pairs and not low


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    sorted_ok = sort_and_contrast(manifest)
    if sys.argv[1:] != ["frames"]:
        return 0 if sorted_ok else 1
    browser = find_browser()
    if browser is None:
        print("no Chrome or Edge found; cannot check rendering", file=sys.stderr)
        return 2

    files = [entry["file"] for entry in manifest.values()]
    by_file = {entry["file"]: slug for slug, entry in manifest.items()}
    results = measure(browser, files)

    problems: list[str] = []
    for file, res in sorted(results.items()):
        slug = by_file.get(file, file)
        if "error" in res:
            problems.append(f"{slug:<32} {res['error']}")
            continue
        frame, box = res["frame"], res["box"]
        if not frame["w"] or not frame["h"]:
            problems.append(f"{slug:<32} frame has no size")
            continue
        if box["w"] <= 0 or box["h"] <= 0:
            problems.append(f"{slug:<32} draws nothing")
            continue
        coverage = (box["w"] * box["h"]) / (frame["w"] * frame["h"])
        over = max(
            frame["x"] - box["x"],
            frame["y"] - box["y"],
            (box["x"] + box["w"]) - (frame["x"] + frame["w"]),
            (box["y"] + box["h"]) - (frame["y"] + frame["h"]),
        )
        overflow = over / max(frame["w"], frame["h"])
        if overflow > MAX_OVERFLOW:
            problems.append(
                f"{slug:<32} art falls {overflow * 100:.0f}% outside its frame (clipped)"
            )
        elif coverage < MIN_COVERAGE:
            problems.append(
                f"{slug:<32} art fills only {coverage * 100:.0f}% of its frame"
            )

    print(f"checked {len(results)} icons with {browser.name}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print("  " + line)
        return 1
    print("all icons draw inside their frames")
    return 0 if sorted_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
