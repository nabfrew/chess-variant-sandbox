"""The piece library as the designer's palette offers it (plan-visual.md §7, V9).

`GET /api/design?part=pieces` answers with every drawing in `assets/pieces/`: its slug
(what `appearance.icons` names), its name, its category, the words it is found by, and the
first seat's picture as SVG text for the palette's thumbnail. The page searches and sorts
these and writes the slug a designer picks into the draft; it never decides what a piece
does from its picture, because a picture never changes a game's rules or its identity
(`appearance` is outside `canonical.MECHANIC_KEYS`).

It is a part of the design kit fetched when the palette is first opened rather than a
field of the kit itself: the pictures are about a quarter of a megabyte of text, and the
kit is fetched by every visit to the designer.

The catalogue is `assets/pieces/catalogue.json` (`catalogue.py` writes it). A checkout
without the library answers an empty palette, as `viz/pieces.py` answers no icon.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from chess_variant_sandbox.viz.pieces import ASSET_ROOT

# The order the palette's category chips are offered in, and what each is called on them.
CATEGORY_WORDS = {
    "leapers": "leapers", "riders": "riders", "compounds": "compounds", "royals": "royals",
    "pawns": "pawns", "oddities": "oddities", "shogi": "shogi tiles",
    "xiangqi": "xiangqi discs",
}


def _read(name: str) -> Any:
    try:
        return json.loads((ASSET_ROOT / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def palette() -> dict[str, Any]:
    """Every drawing, in the catalogue's category order and then by name."""
    catalogue, manifest = _read("catalogue.json"), _read("manifest.json")
    if not isinstance(catalogue, dict) or not isinstance(manifest, dict):
        return {"categories": [], "pieces": []}
    order = {name: i for i, name in enumerate(catalogue["categories"])}
    pieces = []
    for slug, entry in catalogue["pieces"].items():
        # The first seat's drawing (or the one file of a single-colour set) is the
        # thumbnail; a two-seat drawing says so, because it changes which picture the
        # second player's pieces wear.
        stem = slug + "-white" if "white" in entry["seats"] else slug
        row = manifest.get(stem)
        if not isinstance(row, dict):
            continue
        try:
            svg = (ASSET_ROOT / row["file"]).read_text(encoding="utf-8")
        except OSError:
            continue
        pieces.append({
            "slug": slug,
            "name": entry["name"],
            "category": entry["category"],
            "words": entry.get("words", []),
            "says": entry.get("says", ""),
            "pair": entry["seats"] != ["one"],
            "licence": row.get("licence", ""),
            "svg": svg,
        })
    pieces.sort(key=lambda p: (order.get(p["category"], len(order)), p["name"]))
    return {
        "categories": [
            {"key": key, "label": CATEGORY_WORDS.get(key, key)}
            for key in catalogue["categories"]
        ],
        "pieces": pieces,
    }
