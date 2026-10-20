"""Remake the game pictures' two typefaces (G6, `web/page/game_picture.py`) from the site's own.

The site serves its faces as WOFF2 (`web/static/fonts/`), which resvg cannot read, and the
two it carries for the picture are variable: Source Sans 3's default instance is its
ExtraLight, and Fraunces' optical size is the browser's to pick by the text's size. So the
picture's faces are those same files, decompressed to TrueType and fixed at one instance
each, renamed to the family the picture asks for:

- `fraunces-v38-latin-600-opsz72.ttf`: Fraunces SemiBold at optical size 72, the size a
  game's name is set at;
- `source-sans-3-v20-latin-600.ttf`: Source Sans 3 SemiBold, for the line under the name
  and the letter on a piece that has no drawing.

Both stay under the SIL Open Font License 1.1, whose texts are beside them. The tool is
fontTools with Brotli, which nothing else here needs, so it is run once and not installed:

    uvx --from "fonttools[woff]" python -I chess_variant_sandbox/web/page/fonts/make.py \
        chess_variant_sandbox/web/static/fonts chess_variant_sandbox/web/page/fonts

A new file is a new picture for every game, since the faces are part of the picture's key.
"""

import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

MADE = (
    ("fraunces-v38-latin-600.woff2", {"opsz": 72}, "Fraunces", "SemiBold",
     "fraunces-v38-latin-600-opsz72.ttf"),
    ("source-sans-3-v20-latin.woff2", {"wght": 600}, "Source Sans 3", "SemiBold",
     "source-sans-3-v20-latin-600.ttf"),
)


def rename(font: TTFont, family: str, style: str) -> None:
    """Name the instance as one family and one style, on the Windows platform only."""
    table = font["name"]
    full = f"{family} {style}"
    for name_id, value in ((1, full), (2, "Regular"), (4, full), (6, full.replace(" ", "-")),
                           (16, family), (17, style)):
        table.setName(value, name_id, 3, 1, 0x409)
    table.removeNames(platformID=1)


def main(source: Path, out: Path) -> None:
    for woff, axes, family, style, made in MADE:
        font = TTFont(source / woff)
        font.flavor = None
        font = instancer.instantiateVariableFont(font, axes)
        font["OS/2"].usWeightClass = 600
        rename(font, family, style)
        font.save(out / made)
        print(made, (out / made).stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
