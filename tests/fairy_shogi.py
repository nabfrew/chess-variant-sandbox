"""Two shogis derived from the committed one, for the tests of what shogi's shape needs.

The committed shogi translates since Q454 (Q450's B3): all six of its promotion entries
carry schema 15's `on_leaving`, so its silver, bishop and rook promote on a move *out* of
the zone here as they do over there. (Q438's second refusal, a pawn dropped to mate, came
off in Q453: it is a move that loses over there and no move here, and `verify` accepts
that.)

`shogi_arriving` is the game B3 replaced (`630828e31147`, promotion on arriving only),
kept as the case Fairy-Stockfish's options cannot say and so still refuse.

`shogi_forward` is the nearest shogi that asks neither rule. Promotion as a map, drops, a
promoted piece marked in the FEN, one-way pieces that are not the pawn slot, and a hand
that holds a piece that is not one-way are still worth testing on their own. Only the
pawn, lance and knight promote, and they move forward only, so none can leave its zone
and their flags compile away. `drops.no_mate` is left out, so that nothing needs the
tolerance; the tests that want it put it back. A walk of 40 random games of 400 plies
found no position where the two engines' moves differ (record, Q438).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# The names the fairy tests ask `_spec` for. Neither is a file in `rulesets/`.
SHOGI_FORWARD = "shogi-forward"
SHOGI_ARRIVING = "shogi-arriving"

# The pieces shogi promotes that can move out of their zone, and what each becomes.
_LEAVING = {"S": "Z", "B": "H", "R": "D"}


def shogi_arriving() -> dict[str, Any]:
    """Shogi promoting on arriving in the zone only: the committed file without the flag."""
    spec = load_spec(RULESETS / "shogi.json")
    spec["name"] = "shogi, promoting on arrival"
    for rule in spec["rules"]["promotion"]:
        rule.pop("on_leaving", None)
    return spec


def shogi_forward() -> dict[str, Any]:
    """Shogi where only the forward-moving pieces promote and a pawn may drop to mate."""
    spec = load_spec(RULESETS / "shogi.json")
    gone = set(_LEAVING.values())
    spec["name"] = "shogi, promoting forward"
    spec["pieces"] = [piece for piece in spec["pieces"] if piece["id"] not in gone]
    spec["rules"]["promotion"] = [
        rule for rule in spec["rules"]["promotion"] if rule["piece"] not in _LEAVING
    ]
    del spec["rules"]["drops"]["no_mate"]
    icons = spec["appearance"]["icons"]
    spec["appearance"]["icons"] = {key: icon for key, icon in icons.items() if key not in gone}
    return spec
