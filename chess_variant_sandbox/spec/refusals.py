"""The refused pairs, by name (PY9, `docs/plan-smells-python.md` §1; box CP4).

Each row is one plain two-key refusal the validator still makes: two rules (or a rule and a
property, or a rule and the lattice) that may not stand in one game, refused whenever both
are on, with one fixed message. Its id is the row of `docs/plan-schema-next.md` §2 that
inventories it (an id can hold several rows when the plan's row names several keys: K1, K3,
R10), ``first`` and ``second`` are the two key paths, and ``message`` is the validator's text,
word for word. `spec/schema.py` emits each through `refusal`, so a pair is stated once, a test
asks `refused_ids` for the id instead of matching prose, and a lift deletes the row.

What each key path being *on* means (a live strike, a turn longer than one move, a contact
table with links) is read where the refusal is checked, as it was before the table; the table
holds what is refused and what is said. A key path names a value where only that value is
refused (``rules.royal_facing: "forbidden"``), and a narrowed pair names the narrowing
(``rules.strike (keeping its victim)``). Refusals whose condition or message is more than a
pair (a message naming a piece or an index, a third key, a dependency refused by absence)
stay inline in `schema.py`.

Kept rows only: what SR2 and SR2b lifted (L5, L6, T2 and T5 in part, H1, H2, H6, W9, N20,
V5 and V6 in part) is not here, or is here as the narrower case still refused.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple


class Refusal(NamedTuple):
    """One refused pair: its plan row, its two key paths, and the validator's words."""

    id: str
    first: str
    second: str
    message: str


_SWEEP = "pieces[].moves[].kind: \"sweep\""
_LONG_TURN = "rules.turn (moves or ramp over 1)"
_COMMAND = "pieces[].sight effect \"command\""
_CAPTURES_OWN = "pieces[].sight captures_own"
_NOBODY = "setup.placements player \"nobody\""
_MAJORITY = "pieces[].control: \"majority\""


def _rounds(rid: str, second: str, what: str, why: str) -> Refusal:
    return Refusal(rid, "rules.simultaneous", second, f"rules.simultaneous beside {what}: {why}")


def _sweep(rid: str, key: str, why: str) -> Refusal:
    return Refusal(rid, _SWEEP, f"rules.{key}", f"rules.{key} is refused beside a piece that sweeps: {why}")


def _command(rid: str, key: str, why: str) -> Refusal:
    return Refusal(
        rid, _COMMAND, f"rules.{key}",
        f"a command (pieces[].sight effect 'command') beside rules.{key}: {why}; leave one out",
    )


def _turncoat(rid: str, key: str) -> Refusal:
    return Refusal(
        rid, _CAPTURES_OWN, f"rules.{key}",
        f"captures_own (a commanded piece taking its own side) beside rules.{key}: "
        "what a commanded own-side capture leaves behind would be a second rule; "
        "leave one out",
    )


def _duck_check(second: str, rule: str) -> Refusal:
    return Refusal(
        "K1", "rules.duck", second,
        f"rules.duck beside {rule}: a move's legality would depend on the duck move after "
        'it; the duck is for a game without check ("win": "royal_capture", "annihilation" '
        'or "none")',
    )


def _duck_taken(key: str) -> Refusal:
    return Refusal(
        "K3", "rules.duck", f"rules.{key}",
        f"rules.duck beside rules.{key}: an uncapturable piece in a blast or a strike "
        "would be a second rule",
    )


def _portals(rid: str, key: str, why: str) -> Refusal:
    return Refusal(rid, "rules.portals", f"rules.{key}", f"rules.portals is refused beside rules.{key}: {why}")


def _nobody(rid: str, key: str, why: str) -> Refusal:
    return Refusal(rid, _NOBODY, f"rules.{key}", f"a placement for nobody is refused beside rules.{key}: {why}")


def _majority(rid: str, key: str, why: str) -> Refusal:
    return Refusal(rid, _MAJORITY, f"rules.{key}", f"majority control is refused beside rules.{key}: {why}")


def _purse(rid: str, key: str, why: str) -> Refusal:
    return Refusal(rid, "rules.purse", f"rules.{key}", f"rules.purse is refused beside rules.{key}: {why}")


_TAKEN_BY_CONTACT = "an explosion destroys what a contact just transformed"
_NO_CHECK_AT_ONCE = "there is no check when both players move at once"
_SECOND_LANDING = "what a capture leaves behind beside an exit is a second landing rule"

REFUSALS: tuple[Refusal, ...] = (
    # §2.2 Long turns.
    Refusal(
        "L1", "rules.en_passant", _LONG_TURN,
        "rules.en_passant beside a turn longer than one move: the window a double "
        "step opens is still open on the mover's own next move, so a player could "
        "take their own pawn in passing; use one move per turn, or drop en passant",
    ),
    Refusal(
        "L2", "pieces[].rest", _LONG_TURN,
        "pieces[].rest beside a turn longer than one move (rules.turn moves or ramp): "
        "inside a turn of several moves a rest would need a second rule, for a piece that "
        "moved earlier in the same turn; use one move per turn, or leave the rest out",
    ),
    Refusal(
        "L3", "rules.npcs", _LONG_TURN,
        "rules.npcs beside a turn longer than one move: a wild piece walks between "
        "turns, and a legality filter would have to know whether the move it is "
        "testing ends the turn; use one move per turn",
    ),
    _rounds("L4", _LONG_TURN, "a turn longer than one move", "a round is one secret move each"),
    Refusal("L7", "rules.duck", "rules.turn", "rules.duck beside rules.turn: a turn already has its two halves"),
    # §2.3 Rounds. R1 (its message says whether the win is stated), R14 (three keys) and R16
    # (it names the npc) stay inline.
    _rounds("R2", "rules.checkmate", "rules.checkmate", "there is no checkmate when both players move at once"),
    _rounds("R3", "rules.forbid_check", "rules.forbid_check", _NO_CHECK_AT_ONCE),
    _rounds("R4", "rules.check_limit", "rules.check_limit", _NO_CHECK_AT_ONCE),
    _rounds(
        "R5", 'rules.royal_facing: "forbidden"', "rules.royal_facing 'forbidden'",
        "two legal moves made together can open the file, which neither did alone",
    ),
    _rounds("R6", "rules.castling", "rules.castling", "castling moves two pieces in one move, and a round has one each"),
    _rounds(
        "R7", "rules.en_passant", "rules.en_passant",
        "it reads the last move, and the pawn that stepped twice may move again this round",
    ),
    _rounds(
        "R8", _SWEEP, "a sweep",
        "which pieces a sweep removes depends on who stood on its path when, and a "
        "round has no such time",
    ),
    _rounds(
        "R9", "rules.portals[] moved_by", "a portal a player moves (moved_by)",
        "a piece going through a portal moved in the same round needs a third timing rule",
    ),
    _rounds(
        "R10", 'rules.strike.position: "choose"', "rules.strike.position 'choose'",
        "the square is chosen before the other player's move is known",
    ),
    _rounds(
        "R10", 'rules.contacts.position: "choose"', "rules.contacts.position 'choose'",
        "the square is chosen before the other player's move is known",
    ),
    _rounds(
        "R11", 'rules.strike.position: "stay"', "rules.strike.position 'stay'",
        "a captor that collided did move, so staying would mean going back",
    ),
    _rounds(
        "R12", 'rules.contacts (an own link, position "swap")',
        "an own rules.contacts link with position 'swap'",
        "the mover's own piece would be put on its origin before the meetings, "
        "where the other player's piece may arrive",
    ),
    _rounds("R13", "rules.drops.no_mate", "rules.drops.no_mate", "it is about checkmate, and there is none"),
    Refusal(
        "R15", "rules.duck", "rules.simultaneous",
        "rules.duck in a game in rounds (rules.simultaneous): both players move at once, "
        "and nobody is the mover",
    ),
    _command("R17", "simultaneous", "two players could move one piece in one round"),
    _nobody("R18", "simultaneous", "a round's collisions were not written for a piece nobody owns"),
    _purse("R19", "simultaneous", "two buyers in one round is a rule not written"),
    # §2.4 What a taken piece becomes.
    Refusal(
        "T1", "rules.strike", "rules.explode_on_capture",
        "rules.strike and rules.explode_on_capture both say what happens to a taken "
        "piece; an explosion destroys what a strike just transformed",
    ),
    Refusal(
        "T2", "rules.strike (keeping its victim)", "rules.drops",
        "rules.strike and rules.drops both say what happens to a taken piece; a struck "
        "piece cannot both stand on the board and sit in a hand (beside a hand only a "
        "strike that removes its victim is allowed: one that captures and stays put, "
        "turning no capturer to the other side)",
    ),
    Refusal(
        "T3", "rules.contacts", "rules.strike",
        "rules.contacts and rules.strike are two spellings of one rule; a strike is "
        "a contact table with no mover column, so write one or the other",
    ),
    Refusal(
        "T4", "rules.contacts", "rules.explode_on_capture",
        "rules.contacts and rules.explode_on_capture both say what happens to a taken "
        f"piece; {_TAKEN_BY_CONTACT}",
    ),
    Refusal(
        "T5", "rules.contacts (a link keeping its target)", "rules.drops",
        "rules.contacts and rules.drops both say what happens to a taken piece; a "
        "transformed piece cannot both stand on the board and sit in a hand (beside a "
        "hand only links that remove their target, naming nothing it becomes, are "
        "allowed)",
    ),
    Refusal(
        "T6", "rules.contacts", "rules.npcs",
        "rules.contacts and rules.npcs both say what happens to a taken piece; a wild piece "
        "eats what it steps onto, and a contact says what that meeting leaves behind",
    ),
    Refusal(
        "T7", "rules.npcs", "rules.strike",
        "rules.npcs and rules.strike both say what happens when a piece is taken; a "
        "struck wild piece would have to become a wild piece of another type, or a "
        "player's, and neither is a rule anybody wrote",
    ),
    Refusal(
        "T8", "rules.npcs", "rules.explode_on_capture",
        "rules.npcs and rules.explode_on_capture both say what happens when a piece "
        "is taken; a blast that destroys a wild piece pays nobody its bounty, and a "
        "wild piece that eats does not explode",
    ),
    Refusal(
        "T9", "rules.contacts.links[].own", "rules.must_capture",
        "rules.contacts has a link onto an own piece beside rules.must_capture; "
        "whether moving onto your own piece counts as a capture is not settled",
    ),
    Refusal(
        "T10", "rules.strike.by", "rules.strike.switches_sides",
        "rules.strike.by beside rules.strike.switches_sides: rules.strike.switches_sides "
        "already names the capturers it changes, and two lists of capturers in one strike "
        "would make which one a switch reads a second rule; leave one out",
    ),
    # §2.5 Promotion, the hand and extinction. H4 (its message names the position) stays inline.
    Refusal(
        "H3", "rules.promotion.only_lost", "rules.promote_on_entry",
        "rules.promotion.only_lost beside rules.promote_on_entry: a piece that enters "
        "its zone takes the first target listed, with no move in which to withhold it; "
        "drop one of the two",
    ),
    Refusal(
        "H5", "rules.drops.either_face", "rules.strike.switches_sides",
        "rules.drops.either_face beside rules.strike.switches_sides: a capture "
        "already changes a piece's side, so what a taken piece becomes has two "
        "answers; drop one of the two",
    ),
    Refusal(
        "H11", 'rules.extinction.per_type: "own"', "rules.drops",
        'rules.extinction.per_type "own" beside rules.drops: a kind that starts in hand '
        "has no opening count on the board, and whether a held kind counts is not "
        "settled; drop one of the two",
    ),
    # §2.6 Sweeps, in `_check_sweep_coherence`'s order. W7 (it names the piece) stays inline.
    _sweep(
        "W1", "strike",
        "a strike says what one taken piece becomes, a sweep takes several, and "
        "ordering the two would choose a game silently",
    ),
    _sweep(
        "W2", "contacts",
        "a contact says what one taken piece becomes, a sweep takes several, and "
        "ordering the two would choose a game silently",
    ),
    _sweep(
        "W3", "explode_on_capture",
        "two multi-square clears on one move would be undone in an order nobody stated",
    ),
    _sweep("W4", "drops", "a sweep over several pieces would put several in a hand at once"),
    _sweep(
        "W5", "npcs",
        "a bounty is paid per victim, and a swept wild piece would be a second "
        "victim with nowhere to be paid",
    ),
    _sweep(
        "W6", "wrap",
        "round a seam two rays can reach one square by two paths, and a move names only its target",
    ),
    Refusal(
        "W8", "rules.portals", _SWEEP,
        "rules.portals is refused beside a 'sweep' move: what a sweep through a portal "
        "removes is not settled",
    ),
    Refusal(
        "W10", "pieces[].sight effect \"shield\"", _SWEEP,
        "a shield (pieces[].sight effect 'shield') beside a sweep move: whether a sweep "
        "passing over a shielded piece removes it or is stopped by it would be a second "
        "rule; leave one out",
    ),
    # §2.7 Lines of sight. V10 to V15 name a piece or a location and stay inline.
    Refusal(
        "V1", "pieces[].sight", "rules.portals",
        "pieces[].sight beside rules.portals: an attack through a portal would need the "
        "sight's reading of attack too, and a sight through a portal a rule of its own; "
        "leave one out",
    ),
    Refusal(
        "V2", "pieces[].sight", "rules.explode_on_capture",
        "pieces[].sight beside rules.explode_on_capture: what a blast does to a shielded "
        "piece, and how its legality reads a frozen attacker, are rules of their own; leave "
        "one out",
    ),
    _command("V3", "reach", "whether a commanded arrival wins for the mover or the owner would be a second rule"),
    _command(
        "V4", "territory",
        "whether a commanded arrival paints for the mover or the owner would be a second rule",
    ),
    _turncoat("V5", "strike"),
    _turncoat("V6", "contacts"),
    Refusal(
        "V7", _COMMAND, "pieces[].rest",
        "a command (pieces[].sight effect 'command') beside pieces[].rest: whose turns a "
        "commanded piece's rest counts would be a second rule; leave one out",
    ),
    Refusal(
        "V8", _COMMAND, _SWEEP,
        "a command (pieces[].sight effect 'command') beside a sweep move: what a commanded "
        "sweep clears would be a second rule; leave one out",
    ),
    Refusal(
        "V9", _CAPTURES_OWN, "rules.drops.from_captures",
        "captures_own (a commanded piece taking its own side) beside "
        "rules.drops.from_captures: whose hand a turncoat capture fills would be a second "
        "rule; leave one out",
    ),
    # §2.9 The duck. K4 (it names the clock) and K5 to K11 (one rule's own shape) stay inline.
    _duck_check("rules.win", '"win": "checkmate"'),
    _duck_check("rules.forbid_check", '"forbid_check": true'),
    _duck_check("rules.check_limit", "rules.check_limit"),
    Refusal(
        "K2", "rules.duck", "rules.portals",
        "rules.duck beside rules.portals: a portal end and the duck share the third "
        "owner's block",
    ),
    _duck_taken("explode_on_capture"),
    _duck_taken("strike"),
    _duck_taken("contacts"),
    # §2.10 Portals. P7, P8 and P11 to P15 name a piece, a move or an index and stay inline.
    Refusal(
        "P1", "rules.portals", 'board.lattice: "hex"',
        "rules.portals is refused on a hex lattice in this schema: the direction a piece "
        "comes out in is read from the sign of its vector, which on six neighbours does "
        "not name one of them",
    ),
    _portals("P2", "npcs", "a wild piece walks a fixed route and a portal moves"),
    _portals("P3", "strike", _SECOND_LANDING),
    _portals("P4", "contacts", _SECOND_LANDING),
    _portals("P5", "explode_on_capture", "a blast beside an exit is a second landing rule"),
    _portals("P6", "territory", "which square a transit colours is not settled"),
    _purse("P9", "portals", "a portal end is nobody's and its own mover"),
    _nobody("P10", "portals", "portal ends share nobody's block"),
    # §2.11 Pieces nobody owns, the purse, bounties and prices, in the validator's order.
    _nobody("O1", "npcs", "the phase and the bounty tally read nobody's block as wild pieces"),
    _nobody("O2", "duck", "the duck shares nobody's block, and its half reads the block as the duck's"),
    _nobody("O3", "strike", "what a capture leaves of a piece nobody owns is a second rule"),
    _nobody("O4", "contacts", "what a contact makes of a piece nobody owns is a second rule"),
    _majority("O10", "territory", "who claims a square a nobody's piece steps on is a second rule"),
    _majority("O11", "reach", "whether a nobody's piece reaching a goal wins is a second rule"),
    _majority("O12", "explode_on_capture", "who a held piece's blast is charged to is a second rule"),
    _purse("O22", "explode_on_capture", "who is paid for a blast's victims is a second rule"),
    # §2.14 Lattice and shape. X2 to X4, X7 and X9 to X11, X13 name a location or a value.
    Refusal(
        "X5", 'setup.orientation (not "rotate")', 'board.lattice: "hex"',
        'setup.orientation must be "rotate" on a hex lattice, written out: the rank '
        "reflection sends a neighbour to a cell that is not one, and the half turn is "
        "the hex symmetry that carries one army onto the other",
    ),
    Refusal(
        "X6", "rules.reach.mirror (true, the default)", 'board.lattice: "hex"',
        'rules.reach.mirror must be "rotate" or false on a hex lattice: true, the '
        "default, reflects the goal by rank, which is not a hex symmetry",
    ),
    Refusal(
        "X12", 'board.shape: "hexagon"', 'board.lattice: "square"',
        'board.shape "hexagon" needs board.lattice "hex": a hexagon of cells is '
        "cut from a hex rhombus, and a square lattice has no hexagon to cut",
    ),
    # §2.15 Royalty and facing.
    Refusal(
        "Z3", 'rules.win: "annihilation"', "pieces[].royal",
        "rules.win is 'annihilation' but a piece is marked royal",
    ),
    # §2.16 Inside one rule.
    Refusal(
        "I1", "rules.reach.safe", 'rules.reach.on: "leave"',
        'rules.reach.safe asks whether the piece on the goal square could be taken '
        'there, and under "on": "leave" the piece that decides the game has just left '
        "it; leave safe out",
    ),
)


def _index() -> dict[tuple[str, str], Refusal]:
    by: dict[tuple[str, str], Refusal] = {}
    for row in REFUSALS:
        assert (row.id, row.second) not in by, (row.id, row.second)
        by[row.id, row.second] = row
    return by


_BY_ID_AND_KEY = _index()
_BY_MESSAGE = {row.message: row for row in REFUSALS}


def refusal(rid: str, second: str | None = None) -> str:
    """The message of the refused pair ``rid`` (beside ``second``, for an id of several rows)."""
    rows = [row for row in REFUSALS if row.id == rid and (second is None or row.second == second)]
    if len(rows) != 1:
        raise KeyError(f"{rid!r} beside {second!r} names {len(rows)} refused pairs")
    return rows[0].message


def refusal_rows(rid: str) -> tuple[Refusal, ...]:
    """Every row the id ``rid`` holds, in the table's order (empty for an id not in it)."""
    return tuple(row for row in REFUSALS if row.id == rid)


def refused_ids(errors: Iterable[str]) -> set[str]:
    """The ids of the refused pairs among ``errors``, a validator's messages."""
    return {_BY_MESSAGE[error].id for error in errors if error in _BY_MESSAGE}
