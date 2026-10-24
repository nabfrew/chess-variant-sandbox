"""Render every drawing in the library onto one page, for eyeballing.

    uv run python assets/pieces/contact_sheet.py && open assets/pieces/index.html

Each drawing is shown as the board shows it (plan-visual.md §7): both seats on the board's
light and dark square, in the light theme and the dark one, the squares' colours read from
`web/page/tokens.py` so the sheet cannot drift from the board. Reads `catalogue.json` and
`manifest.json`, so it cannot drift from what the library holds either. The pictures are
linked by relative path, so the page works straight off the disk.
"""

# compute-lease: light
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>piece icons</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; background: #fafaf8; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  .grid {{ display: grid; gap: 1rem .75rem; grid-template-columns: repeat(auto-fill, minmax(176px, 1fr)); }}
  figure {{ margin: 0; }}
  .board {{ display: grid; grid-template-columns: repeat(4, 40px); width: 160px;
           border: 1px solid #00000022; }}
  .board span {{ display: grid; place-items: center; width: 40px; height: 40px; }}
  .board img {{ width: 36px; height: 36px; }}
  .board img.turned {{ transform: rotate(180deg); }}
  .light-lt {{ background: {light_lt}; }} .light-dk {{ background: {light_dk}; }}
  .dark-lt {{ background: {dark_lt}; }} .dark-dk {{ background: {dark_dk}; }}
  figcaption {{ font-size: 12px; margin-top: .3rem; color: #333; }}
  figcaption small {{ display: block; color: #666; font-size: 11px; }}
  code {{ font-size: 11px; }}
  p.note {{ color: #555; max-width: 52rem; }}
</style>
<h1>Piece icon library</h1>
<p class="note">{count} drawings in {files} files. Each is drawn as the board draws it: the
top row in the light theme, the bottom row in the dark one, each row the first player's
piece on the light and the dark square and then the second player's. Shogi tiles are one
file, the second player's turned round. The western pieces are one set in Cburnett's
grammar; shogi and xiangqi are their own families. Licensing per file is in
<code>CREDITS.md</code>; how the new drawings are made is in <code>draw.py</code>, and
<code>check.py</code> sorts them by silhouette at 24 px.</p>
{sections}
"""

TITLES = {
    "leapers": "Leapers", "riders": "Riders", "compounds": "Compounds", "royals": "Royals",
    "pawns": "Pawns", "oddities": "Oddities", "shogi": "Shogi (westernised)",
    "xiangqi": "Xiangqi",
}


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


def squares() -> dict[str, str]:
    """The board's two squares in each theme, from the stylesheet's own tokens."""
    from chess_variant_sandbox.web.page import tokens

    dark = {**tokens.LIGHT, **tokens.DARK}
    return {"light_lt": tokens.LIGHT["sq-lt"], "light_dk": tokens.LIGHT["sq-dk"],
            "dark_lt": dark["sq-lt"], "dark_dk": dark["sq-dk"]}


def figure(slug: str, entry: dict, manifest: dict) -> str:
    if entry["seats"] == ["one"]:
        files = [(manifest[slug]["file"], False), (manifest[slug]["file"], True)]
    else:
        files = [(manifest[slug + "-white"]["file"], False),
                 (manifest[slug + "-black"]["file"], False)]
    cells = []
    for theme in ("light", "dark"):
        for path, turned in files:
            for square in ("lt", "dk"):
                cls = ' class="turned"' if turned else ""
                cells.append(
                    f'<span class="{theme}-{square}"><img{cls} src="{path}" alt=""></span>'
                )
    says = entry.get("says") or ", ".join(entry["words"][:3])
    return (f'<figure><div class="board">{"".join(cells)}</div><figcaption>'
            f'{html.escape(entry["name"])} <code>{slug}</code>'
            f"<small>{html.escape(says)}</small></figcaption></figure>")


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    catalogue = json.loads((ROOT / "catalogue.json").read_text(encoding="utf-8"))
    sections = []
    for category in catalogue["categories"]:
        slugs = sorted(s for s, e in catalogue["pieces"].items() if e["category"] == category)
        if not slugs:
            continue
        cells = "\n".join(figure(s, catalogue["pieces"][s], manifest) for s in slugs)
        sections.append(f"<h2>{TITLES[category]} <small>({len(slugs)})</small></h2>\n"
                        f'<div class="grid">{cells}</div>')
    out = ROOT / "index.html"
    out.write_text(
        PAGE.format(count=len(catalogue["pieces"]), files=len(manifest),
                    sections="\n".join(sections), **squares()),
        encoding="utf-8", newline="\n",
    )
    print(f"wrote {out} ({len(catalogue['pieces'])} drawings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
