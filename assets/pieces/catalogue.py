"""The library as a designer looks for a piece in it: each drawing's name, its category,
the words it is found by. Writes `catalogue.json`, which the designer's piece palette reads.

    python assets/pieces/catalogue.py

The categories are plan-visual.md §7's: leapers, riders, compounds, royals, pawns,
oddities, and the two families kept apart, shogi and xiangqi. A category says what kind of
piece a drawing is *for*, never what a piece wearing it does: a picture never changes a
game's rules (`viz/pieces.py`). The drawings made here carry their own entries in
`draw.py`; the downloaded ones are described below, by hand, as `aliases.json` is.
"""

# compute-lease: light
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CATEGORIES = ("leapers", "riders", "compounds", "royals", "pawns", "oddities", "shogi",
              "xiangqi")

# The downloaded and derived drawings: slug -> (name, category, words).
KNOWN: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "king": ("king", "royals", ("royal", "shah", "general")),
    "queen": ("queen", "riders", ("rook and bishop", "vizier")),
    "rook": ("rook", "riders", ("chariot", "castle", "tower")),
    "bishop": ("bishop", "riders", ("diagonal rider", "courier")),
    "knight": ("knight", "leapers", ("(1,2) leaper", "horse")),
    "pawn": ("pawn", "pawns", ("soldier", "footman")),
    "amazon": ("amazon", "compounds", ("queen and knight", "empress plus")),
    "centaur": ("centaur", "compounds", ("king and knight", "crowned knight")),
    "chancellor": ("chancellor", "compounds", ("rook and knight", "marshal", "empress")),
    "archbishop": ("archbishop", "compounds", ("bishop and knight", "cardinal", "princess")),
    "champion": ("champion", "compounds", ("wazir, alfil and dabbaba",)),
    "moon": ("moon", "compounds", ("wizard", "ferz and camel", "crescent")),
    "ferz": ("ferz", "leapers", ("met", "advisor", "one diagonal step")),
    "dabbaba": ("dabbaba", "leapers", ("(2,0) leaper", "siege engine")),
    "giraffe": ("giraffe", "leapers", ("(1,4) leaper",)),
    "man": ("man", "royals", ("mann", "non-royal king")),
    "commoner": ("commoner", "royals", ("non-royal king", "guard")),
    "ship": ("ship", "riders", ("boat", "sail")),
    "short-rook": ("short rook", "riders", ("limited rook",)),
    "inverted-king": ("king, upside down", "royals", ("inverted", "promoted")),
    "inverted-queen": ("queen, upside down", "riders", ("inverted", "promoted")),
    "inverted-pawn": ("pawn, upside down", "pawns", ("inverted", "promoted")),
    "rock": ("rock", "oddities", ("rock paper scissors", "boulder")),
    "paper": ("paper", "oddities", ("rock paper scissors", "sheet")),
    "scissors": ("scissors", "oddities", ("rock paper scissors", "cut")),
}

SHOGI = {
    "king": "king", "rook": "rook", "bishop": "bishop", "gold": "gold general",
    "silver": "silver general", "knight": "knight", "lance": "lance", "pawn": "pawn",
    "promoted-rook": "dragon king", "promoted-bishop": "dragon horse",
    "promoted-silver": "promoted silver", "promoted-knight": "promoted knight",
    "promoted-lance": "promoted lance", "promoted-pawn": "tokin",
}
XIANGQI = {
    "general": "general", "advisor": "advisor", "elephant": "elephant", "horse": "horse",
    "chariot": "chariot", "cannon": "cannon", "soldier": "soldier",
}


def slugs(manifest: dict) -> list[str]:
    """One slug a drawing: the manifest's stems without their seat."""
    out = set()
    for stem in manifest:
        out.add(stem.rsplit("-", 1)[0] if stem.endswith(("-white", "-black")) else stem)
    return sorted(out)


def build(manifest: dict) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from draw import DRAWN

    pieces: dict[str, dict] = {}
    for slug in slugs(manifest):
        seats = ["white", "black"] if slug + "-white" in manifest else ["one"]
        if slug in DRAWN:
            d = DRAWN[slug]
            entry = {"name": d.name, "category": d.category, "words": list(d.words),
                     "says": d.says}
        elif slug in KNOWN:
            name, category, words = KNOWN[slug]
            entry = {"name": name, "category": category, "words": list(words)}
        elif slug.startswith("shogi-"):
            entry = {"name": "shogi " + SHOGI[slug[6:]], "category": "shogi",
                     "words": [slug[6:].replace("-", " ")]}
        elif slug.startswith("xiangqi-pictorial-"):
            piece = slug[len("xiangqi-pictorial-"):]
            entry = {"name": "xiangqi " + XIANGQI[piece] + ", drawn", "category": "xiangqi",
                     "words": [piece, "pictorial"]}
        elif slug.startswith("xiangqi-"):
            piece = slug[len("xiangqi-"):]
            entry = {"name": "xiangqi " + XIANGQI[piece], "category": "xiangqi",
                     "words": [piece, "character"]}
        else:
            raise SystemExit(f"{slug}: no catalogue entry; add it to KNOWN or draw.py")
        if entry["category"] not in CATEGORIES:
            raise SystemExit(f"{slug}: unknown category {entry['category']!r}")
        pieces[slug] = {**entry, "seats": seats}
    return {"categories": list(CATEGORIES), "pieces": pieces}


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    out = build(manifest)
    (ROOT / "catalogue.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote catalogue.json ({len(out['pieces'])} drawings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
