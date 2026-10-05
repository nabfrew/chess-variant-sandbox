"""Pick piece icons from ``assets/pieces`` for a spec, and inline them.

The icon library is deliberately optional. It is a hand-maintained judgement about
what a piece *looks* like (``aliases.json``), it does not cover every name, and a
checkout may not have fetched it at all -- so every lookup here can fail, and the
caller falls back to drawing the spec letter. Nothing in this module raises on a
missing file, a missing alias or a missing library.

Icons are inlined as ``data:`` URIs rather than as SVG ``<symbol>`` elements. Four
of the library files carry their own ``<style>`` block with class names like
``.s0``, and several reuse ids like ``path1``; dropping them into one document
would let one icon restyle another. A data URI is a separate document, so the
files travel bit-for-bit unmodified, which is also what their licences ask for.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "pieces"

# Player 0 is drawn light, player 1 dark, matching the board palette. Single-colour
# sets (shogi, xiangqi) distinguish the players by rotation instead, so a slug with
# no -white/-black pair is used for both and turned 180 degrees for player 1.
_SUFFIX = ("white", "black")


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    try:
        raw = json.loads((ASSET_ROOT / "aliases.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        key.lower(): value
        for key, value in raw.items()
        if not key.startswith("_") and isinstance(value, str)
    }


@lru_cache(maxsize=1)
def _manifest() -> dict[str, dict[str, Any]]:
    """Per-file provenance, if ``fetch.py`` has run. Only used for attribution."""
    try:
        raw = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=256)
def _data_uri(stem: str) -> str | None:
    path = ASSET_ROOT / "svg" / f"{stem}.svg"
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    return "data:image/svg+xml;base64," + base64.b64encode(blob).decode("ascii")


def _slug(name: str) -> str:
    key = name.strip().lower()
    return _aliases().get(key, key.replace(" ", "-"))


# What an uploaded icon looks like when it arrives here. ``appearance.icons`` maps a
# piece id to a *string*, which the schema requires to be non-empty and leaves otherwise
# free -- so an inlined drawing needed no schema change and no new field, and a spec
# carrying one still hashes to the same game as the same spec without it, because
# appearance is outside `canonical.MECHANIC_KEYS`.
_INLINE = "data:image/svg+xml"


def icon_for(name: str, player: int) -> dict[str, Any] | None:
    """The icon for a piece named ``name`` as played by ``player``, or ``None``.

    ``None`` is the ordinary case for an invented piece, and means "draw the letter".

    A name that is itself a drawing is used as it stands. One upload serves both players,
    turned around for the second, which is exactly what a single-colour library set does
    -- and it makes ``side_names`` answer "first" and "second" rather than claiming a
    colour the picture does not have.
    """
    if name.startswith(_INLINE):
        return {
            "href": name,
            "rotate": player % 2 == 1,
            # Identity is the picture. Two pieces given the same upload collide, which is
            # what `tests/test_viz.py` is checking for -- two pieces that draw alike is
            # worse than neither drawing at all -- and two different uploads do not.
            "stem": "uploaded-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12],
        }
    slug = _slug(name)
    stem = f"{slug}-{_SUFFIX[player % 2]}"
    href = _data_uri(stem)
    if href is not None:
        return {"href": href, "rotate": False, "stem": stem}
    href = _data_uri(slug)
    if href is not None:
        return {"href": href, "rotate": player % 2 == 1, "stem": slug}
    return None


def icons_for(
    names: Sequence[str],
    ids: Sequence[str] = (),
    overrides: dict[str, str | Sequence[str]] | None = None,
) -> list[list[dict[str, Any] | None]]:
    """One entry per piece type, each a ``[player 0, player 1]`` pair.

    ``overrides`` is a spec's ``appearance.icons``, keyed by piece id: it says which
    library icon a piece should wear when its *name* is not something the library has
    heard of, which is most invented pieces. A named icon that does not exist is not an
    error -- the lookup falls back to the piece's name and then to drawing its letter,
    exactly as it does for a piece with no entry at all.
    """
    chosen = overrides or {}
    resolved: list[list[dict[str, Any] | None]] = []
    for index, name in enumerate(names):
        piece_id = ids[index] if index < len(ids) else ""
        wanted = chosen.get(piece_id)
        # One name serves both players; a pair names them separately, which is what a
        # piece with a facing needs -- see `_validate_appearance`. A pair of drawings is
        # never turned round for the second player, because the drawings already differ.
        if isinstance(wanted, (list, tuple)) and len(wanted) == 2:
            pair = [
                (
                    {**found, "rotate": False}
                    if (found := icon_for(str(wanted[player]), player))
                    else None
                )
                for player in (0, 1)
            ]
        else:
            pair = [
                icon_for(str(wanted), player) if wanted else None for player in (0, 1)
            ]
        if not any(pair):
            pair = [icon_for(name, player) for player in (0, 1)]
        resolved.append(pair)
    return resolved


def side_names(icons: Sequence[Sequence[dict[str, Any] | None]]) -> tuple[str, str]:
    """What to call the two players on a board drawn with these icons.

    "white" and "black" are the palette's own claim rather than a chess assumption:
    player 0 is drawn light and player 1 dark, and the letter fallback follows the same
    two colours, so they are the names a person can check against the screen.

    They are a lie on a single-colour set. Shogi and dobutsu tell the players apart by
    turning a piece around, and both armies are the same colour -- so a player told they
    are black looks for black pieces and finds the opponent's as well. One rotated piece
    is enough for that, which is why this is all-or-nothing rather than per piece: the
    names label a *side*, and a side cannot be black for two piece types and not for the
    other three.

    A checkout that never fetched the icon library draws every piece as a letter, in the
    same two colours -- so the answer there is "white" and "black" for every ruleset, and
    it is true.
    """
    rotated = any(
        len(pair) > 1 and pair[1] is not None and pair[1].get("rotate") for pair in icons
    )
    return ("first", "second") if rotated else ("white", "black")


def credits_for(icons: Sequence[Sequence[dict[str, Any] | None]]) -> list[str]:
    """Attribution lines for the icons actually used, for the page footer.

    The CC BY and CC BY-SA files require it. Per-file authorship lives in
    ``manifest.json``, which only exists once ``fetch.py`` has run; without it the
    page says where the icons came from and points at ``CREDITS.md``.
    """
    stems = sorted(
        {
            icon["stem"]
            for pair in icons
            for icon in pair
            # An uploaded drawing is the spec author's own and has no library licence to
            # honour. Crediting Wikimedia for it would be a false attribution, which is
            # the one thing an attribution line must not be.
            if icon is not None and not str(icon["stem"]).startswith("uploaded-")
        }
    )
    if not stems:
        return []
    manifest = _manifest()
    if not manifest:
        return [
            "Piece icons from Wikimedia Commons, redistributed unmodified under "
            "their original free licences; see assets/pieces/CREDITS.md."
        ]
    lines = []
    for stem in stems:
        entry = manifest.get(stem)
        if not isinstance(entry, dict):
            continue
        author = entry.get("author") or "unknown"
        licence = entry.get("licence") or "?"
        lines.append(f"{stem} - {author} ({licence})")
    return lines
