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

PF: a drawing whose movement tradition settles also carries `ready`, the piece that moves
that way, out of `assets/pieces/ready.json` (`tests/test_ready_pieces.py` holds each entry
to its reading in `docs/plan-pieces-moves.md` §1). The page lets a designer take it whole --
drawing, name and moves -- and says in the entry's own words what game-wide rule it needs.
The file is data: nothing here reads what a move means beyond saying it in words.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from chess_variant_sandbox.spec.describe import move_sentence
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


# What of a ready entry travels into a draft as the piece itself.
_READY_PIECE_KEYS = ("name", "value", "royal", "moves", "rest", "sight", "facing")


def _ready(entry: Any) -> dict[str, Any] | None:
    """One drawing's ready piece, as the page takes it, or None."""
    if not isinstance(entry, dict) or not isinstance(entry.get("moves"), list):
        return None
    lattice = entry.get("lattice", "square")
    return {
        "id": entry.get("id", ""),
        "piece": {key: entry[key] for key in _READY_PIECE_KEYS if key in entry},
        "reading": entry.get("reading", ""),
        "says": [move_sentence(move, lattice == "hex") for move in entry["moves"]],
        "lattice": lattice,
        "rules": entry.get("rules"),
        "rulesSay": entry.get("rules_say", ""),
        "note": entry.get("note", ""),
    }


@lru_cache(maxsize=1)
def palette() -> dict[str, Any]:
    """Every drawing, in the catalogue's category order and then by name."""
    catalogue, manifest = _read("catalogue.json"), _read("manifest.json")
    ready_file = _read("ready.json")
    ready = ready_file.get("pieces", {}) if isinstance(ready_file, dict) else {}
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
            "ready": _ready(ready.get(slug)),
        })
    pieces.sort(key=lambda p: (order.get(p["category"], len(order)), p["name"]))
    return {
        "categories": [
            {"key": key, "label": CATEGORY_WORDS.get(key, key)}
            for key in catalogue["categories"]
        ],
        "pieces": pieces,
    }
