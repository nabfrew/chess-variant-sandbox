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
    # Q1005 (2026-10-23, `docs/reference/preview-call-2026-10.md`): the user asked for a call on
    # every preview. Six left the same way; the showcase, player and net tests still read them.
    "gardner-with-weary-heavies": "1d1d3c79ce19",
    "breather-by-weight": "7ea0357fa8de",
    "gardner-in-rounds-with-a-swift-king": "d8bb879440ce",
    "gardner-in-rounds-with-swift-knights": "bf230a962d10",
    "watering-hole-by-speed": "423a29997264",
    "gardner-with-spies-resting": "8ce37482d0de",
    # Q1007e (2026-10-24, `docs/archive/plan-schema-31.md` §6.1): Q999b's colour-bound game, a library
    # game since Q1005, replaced under its name by R1 (riders as rook and queen) by the user's
    # rule; the showcase and lame-rider tests still read it.
    "opposite-colours": "690631b0e4eb",
    # Q1007f (2026-10-24, `docs/archive/plan-schema-31.md` §6.3): the command showcases K1 and K2, built,
    # measured and retired before they ever joined: each missed V2 (the effect is offered too
    # rarely), and the call registered before their first measured game retires a miss. Never
    # previews; kept here because the V1 mechanism tests read them
    # (`tests/test_showcases_q1007.py`).
    "chess-with-commanding-knights": "adc1a91b0a83",
    "chess-with-a-spellbinding-king": "6a8120390853",
    # I2 (2026-10-25, `docs/archive/plan-pieces-moves.md` §4.7): the showcase of pieces that never move,
    # chess with an immobile royal flag, built, measured and retired before it ever joined: the
    # index missed V3's 0.800 (0.790), and the call registered before its first measured game
    # retires a miss. Kept here because its V1 mechanism tests read it
    # (`tests/test_showcase_capture_the_flag.py`).
    "capture-the-flag": "407af145c5d6",
    # N8 (2026-10-26, `docs/archive/plan-npc.md` §6.3): the showcase of a wild piece's sight (freeze,
    # disarm, shield) and of the duck that walks by its moves from a start square, chess on 8 by
    # 9 with a herd of six effect-carrying wild pieces and a walking duck, built, measured and
    # retired before it ever joined: the index missed V3's 0.840 (0.816), and the call
    # registered before its first measured game retires a miss. Kept here because its V1
    # mechanism tests read it (`tests/test_showcase_menagerie.py`).
    "the-menagerie": "3ab2ce41d4c9",
    # TC9 (2026-10-30, `docs/plan-turncoat.md` §8.2): the showcase of bounties, the purse and the
    # market, chess whose captures pay coins that buy a pawn, knight or bishop onto the back
    # rank, built, measured and retired before it ever joined: V2's snowball bar missed (the
    # first capturer won 0.754 of decided graded games, the bar 0.70), and the call registered
    # before its first measured game retires a miss. Kept here because its V1 mechanism tests
    # read it (`tests/test_showcase_war_chest.py`).
    "the-war-chest": "0495d0f9f2d2",
    # TC10 (2026-10-30, `docs/plan-turncoat.md` §8.3): the showcase of hiring and bribing, chess
    # with four mercenaries nobody owns hired for coins that captures pay and bribed away at
    # twice the price, built, measured and retired before it ever joined: V2's bars missed (a
    # hire in 0.323 of graded games, the bar 0.60; a bribe in 0.098, the bar 0.20; the
    # mercenaries were mostly taken for their bounty), and the call registered before its first
    # measured game retires a miss. Kept here because its V1 mechanism tests read it
    # (`tests/test_showcase_condottieri.py`).
    "the-condottieri": "80134a4bced2",
    # TC9b (2026-10-30, `docs/plan-turncoat.md` §8.4): the war chest's successor, chess whose
    # captures pay one coin each whatever they took, and coins buy a pawn, knight or rook at
    # their value onto the back rank, built, measured and retired before it ever joined: V2's
    # snowball bar at strength missed (the first capturer won 0.618 of decided top-three games,
    # the bar chess's 0.492 + 0.10), and the call registered before its first measured game
    # retires a miss. Kept here because its V1 mechanism tests read it
    # (`tests/test_showcase_levy.py`).
    "the-levy": "8693c6554669",
    # TC9c (2026-10-30, `docs/plan-turncoat.md` §8.6): the showcase of bounties paid to the
    # loser (`rules.purse.bounty_to`), the levy's market with each piece's bounty paid to the
    # player who lost it, built, measured and retired before it ever joined: V2's market clause
    # missed (rooks 0.758 of graded buys, the bar 0.70), though both snowball clauses and the
    # buying from behind held, and the call registered before its first measured game retires
    # a miss. Kept here because its V1 mechanism tests read it
    # (`tests/test_showcase_indemnity.py`).
    "the-indemnity": "d9787f06e362",
    # TC9d (2026-10-30, `docs/plan-turncoat.md` §8.7): the indemnity's successor, its bounties
    # and payee with the bishop added to the market and the bishop and rook priced a coin over
    # their value, built, measured and retired before it ever joined: V2's snowball clause at
    # strength missed (the first capturer 0.719 of decided top-three games, the bar chess's
    # 0.492 + 0.10), though the market clause held (rooks 0.563 of graded buys), and the call
    # registered before its first measured game retires a miss. Kept here because its V1
    # mechanism tests read it (`tests/test_showcase_quartermaster.py`).
    "the-quartermaster": "4d02af0670b2",
}

# Retired previews kept byte for byte though the validator now refuses them, so a test can hold
# the refusal: Q962c's near copy of `archive/breather.json` (2026-10-19, the rest without its two
# endings) writes schema 27's `rules.simultaneous.rest`, which schema 29 retired (Q983a,
# `docs/plan-rest.md` §4.4, §4.5). Its rewrite is `tests/_simultaneous_resting.json`
# (`tests/test_rest_equivalence.py`); its site row is rewritten by the store's migration.
REFUSED_PREVIEWS = {
    "gardner-in-rounds-resting": "4da669bb3d50",
}

HERE = Path(__file__).resolve().parent


def retired_preview_path(name: str) -> Path:
    """Where the retired preview `name` (its old `archive/` stem) is kept now."""
    assert name in RETIRED_PREVIEWS or name in REFUSED_PREVIEWS, f"{name} is not a retired preview"
    return HERE / f"_retired_{name.replace('-', '_')}.json"


def retired_preview(name: str) -> dict[str, Any]:
    """The retired preview `name`, as it stood in `archive/`."""
    return json.loads(retired_preview_path(name).read_text(encoding="utf-8"))
