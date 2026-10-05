"""Render every icon in the library onto one page, for eyeballing.

    python assets/pieces/contact_sheet.py && open assets/pieces/index.html

Reads manifest.json, so it cannot drift from what fetch.py actually downloaded.
Icons are inlined rather than linked so the page works straight off the disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Cburnett's light pieces are white with a black outline, invisible against a
# white page, so every swatch gets a board-square background behind it.
PAGE = """<!doctype html>
<meta charset="utf-8">
<title>piece icons</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; background: #fafaf8; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  .grid {{ display: grid; gap: .75rem; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); }}
  figure {{ margin: 0; text-align: center; }}
  .sq {{ display: grid; place-items: center; height: 72px; border-radius: 4px;
         background: #b58863; border: 1px solid #00000022; }}
  .sq.light {{ background: #f0d9b5; }}
  .sq svg {{ width: 56px; height: 56px; }}
  figcaption {{ font-size: 11px; margin-top: .3rem; word-break: break-word; color: #555; }}
  code {{ font-size: 11px; }}
  p.note {{ color: #666; max-width: 46rem; }}
</style>
<h1>Piece icon library</h1>
<p class="note">{count} files. White pieces sit on a dark square and black pieces on a
light one, so the outlines read either way. The shogi tiles are single-colour and
shown once &mdash; that set distinguishes the two players by rotating a piece 180&deg;,
not by recolouring it. Xiangqi does recolour, and also uses a different character
per side, so it has real pairs; its <code>-white</code> is the red side.
Licensing per file is in <code>CREDITS.md</code>.</p>
{sections}
"""


def inline(svg_path: Path, view_box: str) -> str:
    """Strip the XML prologue and force a known viewBox and size.

    The opening tag is matched wherever it is rather than anchored at the start:
    Inkscape writes a ``<!-- Created with Inkscape -->`` comment ahead of it, and
    an anchored pattern silently left every such file un-normalised -- keeping its
    own ``width``/``height`` and no ``viewBox``, so it drew at natural size instead
    of scaling to the swatch.
    """
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<\?xml.*?\?>", "", text, flags=re.S)
    text = re.sub(r"<!DOCTYPE.*?>", "", text, flags=re.S)
    text = text.strip()
    opening = re.search(r"<svg\b[^>]*>", text, flags=re.S)
    if opening is None:
        raise SystemExit(f"{svg_path.name}: no <svg> element")
    replacement = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}"'
        ' preserveAspectRatio="xMidYMid meet">'
    )
    return text[: opening.start()] + replacement + text[opening.end() :]


def group_of(slug: str) -> str:
    if slug.startswith("shogi-"):
        return "Shogi (westernised)"
    if slug.startswith("xiangqi-"):
        return "Xiangqi"
    if slug.startswith("inverted-"):
        return "Inverted forms"
    if slug.rsplit("-", 1)[0] in {"king", "queen", "rook", "bishop", "knight", "pawn"}:
        return "Standard chess"
    return "Fairy pieces"


ORDER = ["Standard chess", "Fairy pieces", "Inverted forms", "Shogi (westernised)", "Xiangqi"]


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = {name: [] for name in ORDER}
    for slug in sorted(manifest):
        groups.setdefault(group_of(slug), []).append(slug)

    sections = []
    for name in ORDER:
        slugs = groups.get(name) or []
        if not slugs:
            continue
        cells = []
        for slug in slugs:
            entry = manifest[slug]
            svg = inline(ROOT / entry["file"], entry["viewBox"])
            square = "light" if slug.endswith("-black") else ""
            cells.append(
                f'<figure><div class="sq {square}">{svg}</div>'
                f"<figcaption>{slug}</figcaption></figure>"
            )
        joined = "\n".join(cells)
        sections.append(
            f"<h2>{name} <small>({len(slugs)})</small></h2>\n"
            f'<div class="grid">{joined}</div>'
        )

    out = ROOT / "index.html"
    out.write_text(
        PAGE.format(count=len(manifest), sections="\n".join(sections)), encoding="utf-8"
    )
    print(f"wrote {out} ({len(manifest)} icons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
