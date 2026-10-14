"""A draw offered to the computer, and judged by the strongest search the game has (Q790).

`docs/reference/intransitive-quiet-endgame.md` §e.1 is the rule, pre-registered before any
judgement ran; this module is that rule and nothing more.

**Rule 3.** Whether a position is drawn is a question about the game, so Python does not
answer it. The core does: `Core.search_values` (Q665) is the playing search's own root value
of a position, for the side to move, at the last depth of deepening it completed. What is
here is the policy over that number -- *level within half the game's cheapest piece* --
and the bookkeeping of when an offer may be made. Nothing here reads a square.

**The judge** is one player on every game, whatever sits in the computer's seat: the
site's strongest built-in budget (*hardest*, 200,000 nodes) with its mobility term off.
Mobility adds 0.05 a legal move of the side to move at the leaf, so its sign follows the
parity of the depth the search stopped on; it is a tie-break between quiet moves, and with
it off a position nobody can win reads 0. Not the learned net where a game has one: its
value is not centred on a draw (§e.1 gives the reading), and it is never rated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.agents.base import native_spec
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import eval_values

JUDGE = "minimax:nodes=200000,mobility=0"
"""Who judges an offer: *hardest*'s search, mobility off (§e.1)."""

EARLIEST = 40
"""The first ply at which the computer considers a draw: twenty moves each (§e.1).

Earlier it declines without searching. A level opening is not a draw, and an offer at
move one against the strongest player of a rated game would buy a rating for nothing."""

GAP = 10
"""A side makes at most one offer in any ten plies, five of its own moves (§e.1)."""

BAND_SHARE = 0.5
"""Level means `|value| < BAND_SHARE` of the game's cheapest piece (§e.1)."""

ACCEPTED_SAID = "The computer accepts the draw: its strongest search sees a level game."
DECLINED_SAID = "The computer declines the draw: it does not see a level game."
TOO_EARLY_SAID = (
    "The computer considers a draw once each side has made "
    f"{EARLIEST // 2} moves."
)
OWN_TURN_SAID = "A draw is offered on your own turn."


def band(spec: Any) -> float:
    """Half the cheapest piece in the game's derived table, among pieces that are not royal.

    The unit is the table's, so *level* means the same share of a piece on every game:
    half a pawn in chess, half a rock in *Intransitive*. A table with no positive non-royal
    value (no game the library has) is read as one piece.
    """
    values = eval_values(spec)
    royal = set(spec.royal_types)
    pieces = [value for kind, value in enumerate(values) if value > 0 and kind not in royal]
    return BAND_SHARE * (min(pieces) if pieces else 1.0)


@dataclass(frozen=True)
class Judgement:
    """The core's answer to *is this level?*, and the verdict the rule reads off it.

    `value` is the judge's root value for the side to move, `depth` the depth it completed;
    both None where the core had no value (no legal move, a finished position, no depth
    completed), which the rule reads as a decline.
    """

    accepted: bool
    value: float | None
    depth: int | None
    band: float

    def as_entry(self, seat: int, ply: int) -> dict[str, Any]:
        """The offer as the store keeps it: who, when, and what the core said."""
        return {
            "seat": seat,
            "ply": ply,
            "accepted": self.accepted,
            "value": self.value,
            "depth": self.depth,
            "band": self.band,
        }


def judge(engine: Any, state: GameState, seed: int) -> Judgement:
    """Ask the core whether this position is level, and read the rule off its answer.

    `seed` is the stream the search shuffles its root with, `match seed + ply` as the seat's
    own moves are seeded, so the same offer in the same position gets the same answer.
    """
    limit = band(engine.spec)
    native = native_spec(JUDGE, engine.spec.hash)
    if engine.core is None or native is None:
        return Judgement(accepted=False, value=None, depth=None, band=limit)
    answers = engine.core.search_values(
        [_pack(state)], native, eval_values(engine.spec), [seed]
    )
    answer = answers[0] if answers else None
    if answer is None:
        return Judgement(accepted=False, value=None, depth=None, band=limit)
    value, _move, depth = answer
    level = math.isfinite(value) and abs(value) < limit
    return Judgement(accepted=level, value=float(value), depth=int(depth), band=limit)


def last_offer(offers: Sequence[dict[str, Any]], seat: int) -> dict[str, Any] | None:
    """This seat's most recent offer in the log, or None."""
    for entry in reversed(offers):
        if int(entry.get("seat", -1)) == seat:
            return entry
    return None


def gap_refusal(offers: Sequence[dict[str, Any]], seat: int, ply: int) -> str | None:
    """Why this seat may not offer at this ply under the one-in-ten rule, or None.

    Measured as the distance between plies either way, so a take-back to before an offer
    does not open a second one beside it.
    """
    for entry in offers:
        if int(entry.get("seat", -1)) != seat:
            continue
        apart = abs(ply - int(entry.get("ply", 0)))
        if apart < GAP:
            left = math.ceil((GAP - apart) / 2)
            when = "just now" if apart == 0 else f"{apart} {'ply' if apart == 1 else 'plies'} ago"
            return (
                f"You offered a draw {when}; you can offer again after {left} more "
                f"{'move' if left == 1 else 'moves'} of your own."
            )
    return None


def against_computer_refusal(
    offers: Sequence[dict[str, Any]], seat: int, ply: int, to_move: int
) -> str | None:
    """Why the person in `seat` may not offer the computer a draw now, or None."""
    if to_move != seat:
        return OWN_TURN_SAID
    if ply < EARLIEST:
        return TOO_EARLY_SAID
    return gap_refusal(offers, seat, ply)


def answer_said(entry: dict[str, Any] | None) -> str | None:
    """The sentence the board shows for the computer's answer to an offer."""
    if entry is None:
        return None
    return ACCEPTED_SAID if entry.get("accepted") else DECLINED_SAID
