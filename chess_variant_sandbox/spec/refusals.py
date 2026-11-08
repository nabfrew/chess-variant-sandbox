"""The refused pairs, by name (PY9, `docs/archive/plan-smells-python.md` §1; box CP4).

Each row is one plain two-key refusal the validator still makes: two rules (or a rule and a
property, or a rule and the lattice) that may not stand in one game, refused whenever both
are on, with one fixed message. Its id is the row of `docs/plan-schema-next.md` §2 that
inventories it (an id can hold several rows when the plan's row names several keys: K1,
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
V5 and V6 in part) and the core lifts of SR3d (K2), SR3e (K3), SR3f (O1, O2), SR3g (P9,
P10), SR4b (W4), SR4c (W5), SR4d (H11), SR4e (T6, T7), SR4f (T8), SR4h (V5, V6, V9), SR4i (T9), SR4j (T10), SR5a (T1, T4),
SR5b (V2), SR5c and SR5c2 (V1), SR5d (V3, V4), SR5g (P1), SR5h (P5), SR5i (P6, P7), SR5j (O12), SR5k (O10, O11), SR6a (W1), SR6b (W2), SR6c (W3), SR6d (W6), SR7a (L1), SR7b (L3), SR7d (L2), SR7e (R11), SR7f (R18), SR7g (R19), SR7n (R20) is not here, or is here as the
narrower case still refused.
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


def _rounds(rid: str, second: str, what: str, why: str) -> Refusal:
    return Refusal(rid, "rules.simultaneous", second, f"rules.simultaneous beside {what}: {why}")


def _command(rid: str, key: str, why: str) -> Refusal:
    return Refusal(
        rid, _COMMAND, f"rules.{key}",
        f"a command (pieces[].sight effect 'command') beside rules.{key}: {why}; leave one out",
    )


def _duck_check(second: str, rule: str) -> Refusal:
    return Refusal(
        "K1", "rules.duck", second,
        f"rules.duck beside {rule}: a move's legality would depend on the duck move after "
        'it; the duck is for a game without check ("win": "royal_capture", "annihilation" '
        'or "none")',
    )


def _portals(rid: str, key: str, why: str) -> Refusal:
    return Refusal(rid, "rules.portals", f"rules.{key}", f"rules.portals is refused beside rules.{key}: {why}")


_NO_CHECK_AT_ONCE = "there is no check when both players move at once"

REFUSALS: tuple[Refusal, ...] = (
    # §2.2 Long turns.
    # L1 (en passant beside a long turn) lifted by SR7a: the window is the opponent's, opened
    # by the mover's last double step of the turn and open for the opponent's first move only.
    # L2 (rest beside a long turn) lifted by SR7d: a piece that moved this turn is not offered
    # again this turn, then rests its `rest` turns; the list is written per move, counted per turn.
    # L3 (wild pieces beside a long turn) lifted by SR7b: a phase follows every move, as in a
    # turn of one, so no legality filter needs to know whether a move ends the turn.
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
    # R11 (a strike placed "stay" in rounds) lifted by SR7e: a shot does not move the shooter,
    # so it never collides, and shots resolve before the round's moves (rounds.md §4 step 0).
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
    # R18 (a placement for nobody in rounds) lifted by SR7f for a piece that stands, and its
    # narrower row (a held piece, majority control; R20) by SR7n: the holder is read on the
    # round's board and its move is an arrival as any other.
    # R19 (the purse in rounds) lifted by SR7g for the market: a buy is an arrival with the
    # round's moves, and two buys onto one square, or a buy and a move, collide by `beats`
    # and the tie, each price paid. A purse that hires or bribes stays refused (the auction).
    Refusal(
        "R19", "rules.purse (a hire or rules.purse.bribe)", "rules.simultaneous",
        "rules.purse that can hire (a priced piece placed for nobody) or bribe "
        "(rules.purse.bribe) beside rules.simultaneous: two players recruiting one piece on "
        "the board in one round is a rule not written; a purse with a market alone is allowed",
    ),
    # §2.4 What a taken piece becomes.
    # SR5a (T1, ruled 2026-10-27: a switch, `explode_on_capture.strikes`) lifted the pair;
    # what stays refused is the switch where it cannot be read, and the pair where its
    # order is a rule not written.
    Refusal(
        "T1", "rules.explode_on_capture.strikes", 'rules.strike.position (not "stay" or "home")',
        "rules.explode_on_capture.strikes other than 'none' needs rules.strike.position "
        "'stay' or 'home': the blast destroys the capturer, and only those two place a "
        "struck piece without it",
    ),
    Refusal(
        "T1", "rules.explode_on_capture.strikes", 'rules.strike (effect "capture", position "stay")',
        "rules.explode_on_capture.strikes other than 'none' beside a strike of effect "
        "'capture' placed 'stay' (rifle): such a strike only removes, as the blast already "
        "does, so the switch changes nothing; leave it out",
    ),
    Refusal(
        "T1", 'rules.strike.position: "choose"', "rules.explode_on_capture",
        "rules.strike.position 'choose' beside rules.explode_on_capture: a capture that "
        "explodes strikes nothing a chosen square could hold, so the chosen square would "
        "be a second move to the same position",
    ),
    _rounds(
        "T1", "rules.strike", "rules.strike beside rules.explode_on_capture",
        "the order of a round's strikes and blasts is a rule not written",
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
    # SR5a (T4, ruled 2026-10-27 as T1's default): a capture that explodes makes no contact;
    # the switch beside contacts, a choosing contact and rounds stay refused.
    Refusal(
        "T4", "rules.contacts", "rules.explode_on_capture.strikes",
        "rules.explode_on_capture.strikes other than 'none' beside rules.contacts: a capture "
        "that explodes makes no contact, so the switch would strike with a table whose "
        "links are about the capturer too; leave it 'none'",
    ),
    Refusal(
        "T4", 'rules.contacts.position: "choose"', "rules.explode_on_capture",
        "rules.contacts.position 'choose' beside rules.explode_on_capture: a capture that "
        "explodes makes no contact a chosen square could hold, so the chosen square would "
        "be a second move to the same position",
    ),
    _rounds(
        "T4", "rules.contacts", "rules.contacts beside rules.explode_on_capture",
        "the order of a round's contacts and blasts is a rule not written",
    ),
    Refusal(
        "T5", "rules.contacts (a link keeping its target)", "rules.drops",
        "rules.contacts and rules.drops both say what happens to a taken piece; a "
        "transformed piece cannot both stand on the board and sit in a hand (beside a "
        "hand only links that remove their target, naming nothing it becomes, are "
        "allowed)",
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
    # §2.6 Sweeps, in `_check_sweep_coherence`'s order. W7 (it names the piece) stays inline.
    # W1 lifted by SR6a (the sweep switch, `moves[].effects`) but for a chosen square.
    Refusal(
        "W1", _SWEEP, 'rules.strike.position: "choose"',
        "rules.strike.position 'choose' is refused beside a piece that sweeps: a sweep's "
        "move names no landing square for the struck piece, so the choice would be a "
        "second rule; place it 'stay', 'swap' or 'home'",
    ),
    # W2 lifted by SR6b (the sweep switch beside contacts) but for a chosen square.
    Refusal(
        "W2", _SWEEP, 'rules.contacts.position: "choose"',
        "rules.contacts.position 'choose' is refused beside a piece that sweeps: a sweep's "
        "move names no landing square for the piece its contact keeps, so the choice would "
        "be a second rule; place it 'stay' or 'swap'",
    ),
    # W3 lifted by SR6c (the sweep switch beside a blast) but beside a strike or contacts.
    Refusal(
        "W3", _SWEEP, "rules.explode_on_capture beside rules.strike or rules.contacts",
        "rules.explode_on_capture beside rules.strike or rules.contacts is refused beside a "
        "piece that sweeps: explode_on_capture.strikes orders a capture's blast and strike, "
        "not a sweep's, so the order of a sweep's blasts and its strikes or contacts would "
        "be a second rule; leave one of them out",
    ),
    # W6 (a sweep beside wrap) lifted by SR6d: where two rays of a piece reach one target, the
    # move word names its ray (`moves.move_ray`), and the text tells them apart by what each sweeps.
    # W8 (a sweep beside portals) lifted by SR6e: a fold propagates a sweep (its landing on an
    # empty fold may go through, sweeping only before it), and a momentum end ends the ray.
    # W10 (a shield beside a sweep) lifted by SR6g: a shielded piece may not be captured, so it
    # ends a sweep's ray (never passed, never landed on, and no check past it), read off the board.
    # §2.7 Lines of sight. V10 to V15 name a piece or a location and stay inline.
    # V1 (a sight beside portals) lifted by SR5c for freeze, disarm and shield: a momentum end
    # stops a line as a wall does, a fold lets it see the far end's square, and an attack
    # through either reads the effects. SR5c2 lifted the command: a commanded piece passes a
    # portal as its owner's piece would.
    # V2 (a sight beside a blast) lifted by SR5b: a shielded piece survives a blast as if
    # immune to it, and a frozen or disarmed piece makes no capture, so sets off none.
    # V3, V4 (a command beside reach, territory) lifted by SR5d: a commanded piece arrives,
    # leaves a goal and claims for its owner.
    # V7 (a command beside rest) lifted by SR5e: a commanded move rests its piece on its
    # owner's turns, and a resting piece is moved by neither player.
    # V8 (a command beside a sweep) lifted by SR6f: a commanded sweep moves as its owner's
    # sweep does, and removes only its owner's non-royal, unshielded pieces its captures allow,
    # under `captures_own`; any other piece ends the ray.
    # §2.9 The duck. K4 (it names the clock) and K5 to K11 (one rule's own shape) stay inline.
    _duck_check("rules.win", '"win": "checkmate"'),
    _duck_check("rules.forbid_check", '"forbid_check": true'),
    _duck_check("rules.check_limit", "rules.check_limit"),
    # §2.10 Portals. P8 and P11 to P15 name a piece, a move or an index and stay inline.
    # P1 (portals on a hex lattice) lifted by SR5g: a transit comes out one hex king's step
    # beyond the exit end, along the one of the twelve nearest the angle it travelled.
    _portals("P2", "npcs", "a wild piece walks a fixed route and a portal moves"),
    # P3 and P4 (a strike, contacts) lifted by SR4m: the exit is the capture square, read as
    # anywhere (rifle leaves the mover where it was, swap puts the struck piece there).
    # P5 (a blast) lifted by SR5h: the blast centres on the exit, where the capture happened;
    # a portal end in it stands, and a fold in it loses the piece standing in it and stays.
    # P6 (territory) lifted by SR5i: a transit claims the square it ends on, its exit, and
    # never a portal end's; a momentum end's own move claims nothing.
    # §2.11 Pieces nobody owns, the purse, bounties and prices, in the validator's order.
    # O3 and O4 (a strike, contacts) lifted by SR4k: a struck or linked nobody's piece stays
    # nobody's, and a held piece strikes as its holder's piece would.
    # O10, O11 (territory, reach) lifted by SR5k: a held move claims for the mover, and a held
    # arrival (or departure) is read by the goal's own rule as if the mover owned the piece.
    # O12 (a blast) lifted by SR5j: a held piece's blast is the mover's, so one that would take
    # the mover's royal is illegal as any, and one that takes the other royal wins for the mover.
    # O22 (the purse beside a blast) lifted by SR4l: the mover is paid every enemy victim's
    # bounty, nothing for its own.
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


# Q1074b1 (`docs/plan-push.md` §3.14): the push rule's refused pairs, by that plan's row ids,
# each prefixed ``push.`` so they never collide with `docs/plan-schema-next.md` §2's (which has
# its own R1, X2, W4 ...). A table of their own, beside `REFUSALS`, because that one is held
# row for row to `plan-schema-next.md` §2 (`docs_tests/test_refused_pair_ids.py`); the lookups
# below read both. A lift deletes the row, as there.
_PUSH = "pieces[].moves[].push"


def _push(rid: str, second: str, why: str) -> Refusal:
    return Refusal(
        f"push.{rid}", _PUSH, second, f"a push (pieces[].moves[].push) beside {second}: {why}"
    )


PUSH_REFUSALS: tuple[Refusal, ...] = (
    # Refused for now, each with its reason (§3.1 D6 and D7 are a move's own shape and stay
    # inline in `schema._push_entry_refusals`).
    _push(
        "W4", "rules.portals",
        "whether a pushed piece goes through a portal end, and whether a push transits one, "
        "are rules not written",
    ),
    _push(
        "X2", "rules.strike",
        "a strike acts on a capture landing, and a push that removes lands nowhere",
    ),
    _push(
        "X2", "rules.contacts",
        "a contact acts on a capture landing, and a push that removes lands nowhere",
    ),
    _push(
        "X3", "rules.explode_on_capture",
        "a push that removes has no capture square on the board to centre a blast on",
    ),
    _push(
        "S1", "rules.simultaneous",
        "two pushes in one round can move each other's pushers, and a round has no order "
        "for a moved bystander",
    ),
    # Round 1's refusals that round 2 lifted (§3.14; Q1074b8), each kept where its reading
    # stops: rest, reach, territory and the purse were lifted whole.
    _push(
        "E2", "rules.en_passant",
        "the en passant piece has a pushing capture entry whose push replaces its capture, so "
        "what its en passant would push is unstated (give that push 'captures': true)",
    ),
    _push(
        "N4", "pieces[].sight",
        "a command that captures_own moves a commanded piece onto its owner's own piece, and "
        "whether that landing pushes is unstated",
    ),
)


# H1 (`docs/plan-hunger.md` §2.4): what a move's cost is refused beside in its first build, each
# prefixed ``upkeep.`` as the push's are. A lift deletes the row, and the plan's §2.4 says what
# reading each lift would need.
_UPKEEP = "rules.purse.upkeep"


def _upkeep(rid: str, second: str, why: str) -> Refusal:
    return Refusal(
        f"upkeep.{rid}", _UPKEEP, second,
        f"rules.purse.upkeep beside {second}: {why} (first build)",
    )


UPKEEP_REFUSALS: tuple[Refusal, ...] = (
    _upkeep(
        "U1", "rules.simultaneous",
        "a round's collisions would need a rule for refunding a move that never arrived",
    ),
    _upkeep(
        "U2", _LONG_TURN,
        "whether a later move of the turn is paid from the purse the earlier ones left is unbuilt",
    ),
    _upkeep(
        "U3", "rules.portals",
        "a transit's exit and the square the move named disagree about where the piece arrives",
    ),
    _upkeep("U4", "rules.npcs", "a wild piece has no purse, and the pair is untested"),
    _upkeep(
        "U5", _COMMAND,
        "a commanded piece's move is a foreign move, and whose purse pays for it is unsettled",
    ),
    _upkeep(
        "U6", 'pieces[].control "majority" or "shared"',
        "a held or shared piece's move is a foreign move, and whose purse pays for it is unsettled",
    ),
    _upkeep(
        "U7", "rules.drops",
        "a drop arrives and so pays for its square, but what a drop costs beside a placement "
        "phase and a hand filled by captures is a design of its own",
    ),
    _upkeep(
        "U8", 'board.lattice "hex"',
        "rows from a player's first row need one row direction for both sides, which a hex "
        "board does not have",
    ),
    _upkeep(
        "U9", "rules.contacts",
        "a contact may change the moving piece, and which type pays for its square is unsettled",
    ),
    _upkeep(
        "U10", "rules.strike.switches_sides",
        "a capturer that changes sides ends as the opponent's piece, and whose rows it pays "
        "for is unsettled",
    ),
)

# H3 (`docs/plan-hunger.md` §4): what the pass is refused beside in its first build, each
# prefixed ``pass.`` as the upkeep's are. A lift deletes the row; §R3 names the later values.
_PASS = "rules.pass"


def _pass(rid: str, second: str, why: str) -> Refusal:
    return Refusal(f"pass.{rid}", _PASS, second, f"rules.pass beside {second}: {why} (first build)")


PASS_REFUSALS: tuple[Refusal, ...] = (
    _pass(
        "P1", "rules.simultaneous",
        "a stand-still move in a round is a later value of the rule, unbuilt",
    ),
    _pass(
        "P2", "rules.duck",
        "the duck's move is half of every turn, and whether a pass skips it is unsettled",
    ),
    _pass(
        "P3", "rules.turn (more than one move, or parts)",
        "whether a pass ends a long turn early or spends one of its moves is a later value of "
        "the rule, unbuilt",
    ),
    _pass(
        "P4", "rules.must_capture",
        "a pass would be the way out of every forced capture",
    ),
    _pass(
        "P5", "rules.drops.placement",
        "a placement phase allows nothing but a drop",
    ),
    _pass(
        "P6", "rules.npcs",
        "whether a pass sets the wild pieces' phase going is unsettled",
    ),
    _pass(
        "P7", "rules.portals",
        "a straight-back bar and a fold's own move each read the move before, and whether a "
        "pass spends a bar is unsettled",
    ),
    _pass(
        "P8", "pieces[].moves[].push",
        "the push's bar lasts for the very next move, and whether a pass spends it (Go's ko "
        "does) is unsettled",
    ),
)

# H2 (`docs/plan-hunger.md` §3, §R2): what income is refused beside in its first build, each
# prefixed ``income.`` as the upkeep's and the pass's are. A lift deletes the row.
_INCOME = "rules.income"


def _income(rid: str, second: str, why: str) -> Refusal:
    return Refusal(
        f"income.{rid}", _INCOME, second, f"rules.income beside {second}: {why} (first build)"
    )


INCOME_REFUSALS: tuple[Refusal, ...] = (
    _income(
        "I1", 'rules.scoring.count "points"',
        "income into the points tally is a later value of the rule, and the tally is not in "
        "the position's key, so a holder shuffling in place would repeat the position and draw "
        "while it banked points",
    ),
    _income(
        "I2", "rules.simultaneous",
        "a round has no start of one player's turn to pay at, so when income lands in a round "
        "is unsettled",
    ),
)

_ALL = REFUSALS + PUSH_REFUSALS + UPKEEP_REFUSALS + PASS_REFUSALS + INCOME_REFUSALS


def _index() -> dict[tuple[str, str], Refusal]:
    by: dict[tuple[str, str], Refusal] = {}
    for row in _ALL:
        assert (row.id, row.second) not in by, (row.id, row.second)
        by[row.id, row.second] = row
    return by


_BY_ID_AND_KEY = _index()
_BY_MESSAGE = {row.message: row for row in _ALL}


def refusal(rid: str, second: str | None = None) -> str:
    """The message of the refused pair ``rid`` (beside ``second``, for an id of several rows)."""
    rows = [
        row for row in _ALL
        if row.id == rid and (second is None or row.second == second)
    ]
    if len(rows) != 1:
        raise KeyError(f"{rid!r} beside {second!r} names {len(rows)} refused pairs")
    return rows[0].message


def refusal_rows(rid: str) -> tuple[Refusal, ...]:
    """Every row the id ``rid`` holds, in the table's order (empty for an id not in it)."""
    return tuple(row for row in _ALL if row.id == rid)


def refused_ids(errors: Iterable[str]) -> set[str]:
    """The ids of the refused pairs among ``errors``, a validator's messages."""
    return {_BY_MESSAGE[error].id for error in errors if error in _BY_MESSAGE}
