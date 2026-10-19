"""Specs retired from the library that tests still build on, rebuilt from their twins.

The library audit (Q868, 2026-10-12, `docs/archive/plan-library-audit.md`) retired near copies
that tests had been written against. Each is one key from a game still in the library,
so it is rebuilt here from that game rather than kept as a copy, which is how Q280 kept its
tests when it retired the square-grid hex spec: from what is in the tree, not from what
left it. Every rebuild reproduces the retired game's mechanics hash, which
`tests/test_retired_specs.py` holds.

`archive/hex-glinski-promoting.json` (`279f92a97530`) was retired for its native twin
`archive/hex-glinski-hexagon.json`, which is the same game on a board that has no corner
cells instead of thirty walls (Q281e). Many tests were written against the walled spelling
because it was the only hex spec there was, and what they hold -- the walled rhombus, the
obstacle path on a hex lattice, the twin equality -- is still worth holding.

`divide-two-way` (`2fa712feb218`) and `divide-fixed` (`31ead85da009`) were Divide's other
two door rules, kept as the record that no door rule fixed its draws, and
`wormhole-adrift` (`22cc8982bc90`) was `wormhole` with ends either player moves; the portal
tests still read the rules they exercise.

The audit's second pass (Q870, 2026-10-13) retired four more variants of
`hex-glinski-hexagon` for `archive/pinwheel.json`, a game designed around what they were
kept to show. `hex-glinski-flanks` (`5ba53615c752`) and `hex-glinski-flanks-one-way`
(`2edcd9a30022`) are the hexagon with its empty flanks joined, both ways and one way per
player; the board drawing, the designer and the seam tests still read them.
`hex-glinski-turned` (`30c1003b9a61`) and `hex-glinski-turned-mirrored` (`461e29677d6d`)
are the walled game with every officer's move written once and turned `rotate6` or
`dihedral12`; the symmetry tests still read them against their long form.

Q876 (2026-10-13) retired `wolves-immune` (`a2193fb37097`) for `archive/pitchfork.json`, a game
built around the key it was kept to show. It was `wolves` with the five `captures` lists that
say *only a knight may take a wolf* written as one key, `npcs[].immune: ["N"]`, and played it
move for move; the site's twin tests still read the two spellings of one game.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"

# The retired specs' mechanics hashes, which the rebuilds must reproduce.
RETIRED_HASHES = {
    "hex-glinski-promoting": "279f92a97530",
    "divide-two-way": "2fa712feb218",
    "divide-fixed": "31ead85da009",
    "wormhole-adrift": "22cc8982bc90",
    "hex-glinski-flanks": "5ba53615c752",
    "hex-glinski-flanks-one-way": "2edcd9a30022",
    "hex-glinski-turned": "30c1003b9a61",
    "hex-glinski-turned-mirrored": "461e29677d6d",
    "wolves-immune": "a2193fb37097",
}

# The fifteen walls of one corner pair, mirrored: thirty cells of the 11x11 rhombus that
# are not cells of Glinski's hexagon.
_WALLS = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "c1", "c2", "c3", "d1", "d2", "e1"]


def _archived(name: str) -> dict[str, Any]:
    return copy.deepcopy(json.loads((ARCHIVE / f"{name}.json").read_text(encoding="utf-8")))


def hex_glinski_promoting() -> dict[str, Any]:
    """`hex-glinski-hexagon` with its shape spelled as thirty walls: the retired spec."""
    spec = _archived("hex-glinski-hexagon")
    spec["name"] = "hex glinski promoting"
    spec["schema_version"] = 14
    spec["provenance"] = {"added": "2026-09-15"}
    del spec["board"]["shape"]
    spec["rules"]["obstacles"] = {"squares": list(_WALLS), "mirror": True}
    return spec


def divide_two_way() -> dict[str, Any]:
    """`divide` with the pair's `passage` removed: either player may go in at either end."""
    spec = _archived("divide")
    spec["name"] = "divide two way"
    del spec["rules"]["portals"][0]["passage"]
    return spec


def divide_fixed() -> dict[str, Any]:
    """`divide_two_way` with the pair's `moved_by` removed: nobody moves an end."""
    spec = divide_two_way()
    spec["name"] = "divide fixed"
    del spec["rules"]["portals"][0]["moved_by"]
    return spec


def wormhole_adrift() -> dict[str, Any]:
    """`wormhole` with both ends moved by either player."""
    spec = _archived("wormhole")
    spec["name"] = "wormhole adrift"
    spec["rules"]["portals"][0]["moved_by"] = {"orange": [0, 1], "blue": [0, 1]}
    return spec


def hex_glinski_flanks() -> dict[str, Any]:
    """`hex-glinski-hexagon` with its two empty flanks joined both ways."""
    spec = _archived("hex-glinski-hexagon")
    spec["name"] = "hex glinski flanks"
    spec["rules"]["wrap"] = {"nw_se": "both"}
    return spec


def hex_glinski_flanks_one_way() -> dict[str, Any]:
    """The same seam one way per player, symmetric under the half turn."""
    spec = _archived("hex-glinski-hexagon")
    spec["name"] = "hex glinski flanks one way"
    spec["rules"]["wrap"] = {"nw_se_by_player": {"0": "se", "1": "nw"}}
    return spec


def _turned(symmetry: str, knight: list[list[int]], name: str) -> dict[str, Any]:
    spec = hex_glinski_promoting()
    spec["name"] = name
    spec["schema_version"] = 18
    spec["provenance"] = {"added": "2026-10-10"}
    officers = {
        "N": ("leap", knight),
        "B": ("slide", [[1, 1]]),
        "R": ("slide", [[1, 0]]),
        "Q": ("slide", [[1, 0], [1, 1]]),
        "K": ("leap", [[1, 0], [1, 1]]),
    }
    for piece in spec["pieces"]:
        if piece["id"] in officers:
            kind, vectors = officers[piece["id"]]
            piece["moves"] = [
                {"kind": kind, "mode": "both", "symmetry": symmetry, "vectors": vectors}
            ]
    return spec


def hex_glinski_turned() -> dict[str, Any]:
    """The walled game with every officer's move written once and turned `rotate6`. The
    knight needs both hands written, which is why it played Glinski's game move for move."""
    return _turned("rotate6", [[1, 2], [2, 1]], "hex glinski turned")


def hex_glinski_turned_mirrored() -> dict[str, Any]:
    """The same with `dihedral12`, where the knight is one jump."""
    return _turned("dihedral12", [[1, 2]], "hex glinski turned mirrored")


def wolves_immune() -> dict[str, Any]:
    """`wolves` with its five `captures` lists written as one key: only a knight may take a wolf."""
    spec = _archived("wolves")
    spec["name"] = "wolves immune"
    spec["schema_version"] = 10
    spec["provenance"] = {"added": "2026-09-07"}
    for piece in spec["pieces"]:
        if piece["id"] != "N":
            piece.pop("captures", None)
    for npc in spec["rules"]["npcs"]:
        npc["immune"] = ["N"]
    return spec


REBUILT = {
    "hex-glinski-promoting": hex_glinski_promoting,
    "divide-two-way": divide_two_way,
    "divide-fixed": divide_fixed,
    "wormhole-adrift": wormhole_adrift,
    "hex-glinski-flanks": hex_glinski_flanks,
    "hex-glinski-flanks-one-way": hex_glinski_flanks_one_way,
    "hex-glinski-turned": hex_glinski_turned,
    "hex-glinski-turned-mirrored": hex_glinski_turned_mirrored,
    "wolves-immune": wolves_immune,
}


# Q962b (2026-10-18, `docs/reference/previews-2026-10-18.md`) took six previews that did not
# work well out of `archive/`. Their files moved here unchanged (`git mv`), named
# `_retired_<stem>.json`, because the showcase and spy tests still build on them and three of
# them have no kept game one key away to rebuild from; the site retires each at the next deploy.
RETIRED_PREVIEWS = {
    "gardner-in-rounds": "e043fce6968e",
    "gardner-in-rounds-with-meek-pawns": "ff1ccdaaf3cf",
    "gardner-in-rounds-with-the-initiative": "6278607c721b",
    "gardner-in-rounds-where-swaps-collide": "7985900696d7",
    "gardner-in-rounds-scored": "a90a8db7e2dd",
    "gardner-with-spies": "f4c62a33798d",
    # Q962c (2026-10-19): a near copy of `archive/breather.json`, the rest without its two endings.
    "gardner-in-rounds-resting": "4da669bb3d50",
}

HERE = Path(__file__).resolve().parent


def retired_preview_path(name: str) -> Path:
    """Where the retired preview `name` (its old `archive/` stem) is kept now."""
    assert name in RETIRED_PREVIEWS, f"{name} is not a retired preview"
    return HERE / f"_retired_{name.replace('-', '_')}.json"


def retired_preview(name: str) -> dict[str, Any]:
    """The retired preview `name`, as it stood in `archive/`."""
    return json.loads(retired_preview_path(name).read_text(encoding="utf-8"))
