"""A declared test oracle (CLAUDE.md, architecture rule 3): the capture list played on a board
in Python, by `docs/reference/capture-effects-design.md` §2.2 and §2.3 (R-L4b stage 2).

It answers what a move leaves on the board, in the hands, the points and the purse, from the
board before the move, the move word and a compiled spec carrying a list
(`compile.with_capture_list`). It shares no code with the core: part one (the step, a
promotion, en passant's victim, a sweep's path, a push's shift, a gated piece entering, the
move's cost) and the list's walk are written here again, from the record, so a test that asks
both and compares them checks the core's walk.

**Its scope, declared.** One facing (no headings), no `promote_on_entry`, no region, no sight
(so no shield), no wild piece, no portal, a mover moving its own piece (no command, no held
piece), no promoting sweep, no `either_face` hand. `supported(compiled)` says whether a spec is
in it, and `play` raises `Unsupported` for a move outside it (a castling, a drop): the caller
skips those.

Part one (§2.2, §3.2): the step; a promotion; en passant's victim; a sweep's path cleared; a
push's line shifted one step (`CompiledSpec.push_line`'s walk: a piece thrown off the edge or
crushed on a wall is the push's taking, from the square it left, whoever owns it); a gated
piece entering the origin from the hand; and under `rules.purse.upkeep` the move's cost, for
the type the move made the piece (its promotion), on the square it stands on when the list has
run (its origin where a mover half returned it there, else its target, the square a piece the
list took away last stood on), paid before any bounty.

The walk (§2.2): the move's takings in taking order (the landing's, then a sweep's path, then a
push's); each entry from the top over the round's takings; a blast's victims are the next
round's. Rule 1: a taking leaves limbo once. Rule 2: a placement needs an empty square. Rule 3:
a placed piece is settled (no blast clears it), and a blast fires once on a square. Each
`strike` entry reads the strike table and each `contacts` entry the contact table where the
spec states both (T3, `CompiledSpec.contacts`). A meeting with the mover's own piece is offered
only by a `contacts` entry reading the landing (§2.5), so only then does it set off no blast.

**A round** (R-L4b5, `rounds.md` §4 steps 5 and 6): `walk_round` walks the list over a round's
takings, in the order the round decided them (the shots, then the meetings in seat order), with
the same three rules. The round's own board writes -- a bounty, a hand, which captures set off a
blast and what it clears, a strike's mover and victim halves, a shooter's change -- are the
caller's (`tests/simultaneous_oracle.py`, through `RoundHooks`); what the list means (which
entry reads which taking, in what order, which victim is still in limbo, which square is
settled, which centre has fired, a blast's victims walked as the next round, one record per
captor whose blast reached the piece) is here. A list derived from the old keys
(`default_round_list`, the core's `CaptureProgram::derive`) reads a collision's loser with its
strike or contacts entry where `SimRule.collision_strikes` says so, as schema 26 did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from chess_variant_sandbox.engine.compile import (
    BLAST_STRIKES_TARGET,
    BLAST_STRIKES_VICTIMS,
    CaptureEntry,
    CompiledSpec,
    StrikeRule,
)
from chess_variant_sandbox.engine.moves import (
    MOVE_EN_PASSANT,
    MOVE_NORMAL,
    MOVE_PROMOTION,
    MOVE_PUSH,
    MOVE_SWEEP,
    move_from,
    move_gate,
    move_landing,
    move_promotion,
    move_push,
    move_to,
    move_type,
)

PURSE_MAX = 99


class Unsupported(Exception):
    """A move or a spec outside the oracle's declared scope."""


def supported(compiled: CompiledSpec) -> bool:
    """Whether ``compiled`` is inside the oracle's scope (the module's docstring)."""
    return (
        compiled.capture is not None
        and compiled.n_facings == 1
        and compiled.entry_promotion is None
        and not compiled.region_stand
        and compiled.sight is None
        and compiled.npcs is None
        and compiled.portals is None
        and compiled.neutrals is None
        and compiled.duck is None
        and not (compiled.drops is not None and compiled.drops.faces)
    )


@dataclass
class Taking:
    square: int
    victim: int
    cause: str
    limbo: bool = True


@dataclass
class Outcome:
    board: list[int]
    hands: list[list[int]]
    points: list[float]
    purse: list[int]


@dataclass
class _Walk:
    taker_type: int
    moving: int
    player: int
    origin: int
    target: int
    landing_index: int
    taker_at: int | None
    last_at: int
    settled: set[int] = field(default_factory=set)
    fired: set[int] = field(default_factory=set)
    # Each bounty as `(payee, amount)`, paid after the move's cost (§3.2).
    paid: list[tuple[int, float]] = field(default_factory=list)


class _Board:
    def __init__(self, compiled: CompiledSpec) -> None:
        self.c = compiled
        self.n = compiled.n_types

    def owner(self, value: int) -> int:
        return (value - 1) // self.n

    def type_of(self, value: int) -> int:
        return (value - 1) % self.n

    def value(self, owner: int, piece_type: int) -> int:
        return owner * self.n + piece_type + 1


def _table(c: CompiledSpec, entry: CaptureEntry) -> StrikeRule:
    """The strike table ``entry`` reads: T3's contact table for a ``contacts`` entry where the
    spec states one beside its strike, else the one strike table."""
    if entry.do == "contacts" and c.contacts is not None:
        return c.contacts
    assert c.strike is not None
    return c.strike


def _meets_own(c: CompiledSpec) -> bool:
    """§2.5: whether the list offers a meeting with the mover's own piece (the first
    ``contacts`` entry reading the landing has own links)."""
    assert c.capture is not None
    for entry in c.capture:
        if entry.do == "contacts" and "landing" in entry.on:
            return _table(c, entry).meets_own
    return False


def play(
    compiled: CompiledSpec, board: bytes, side: int, move: int, ep_victim: int,
    hands: tuple[tuple[int, ...], ...], points: tuple[float, float], purse: tuple[int, int],
) -> Outcome:
    """The position ``move`` leaves, as the list says (see the module's docstring)."""
    assert compiled.capture is not None
    b = _Board(compiled)
    out = Outcome(list(board), [list(hand) for hand in hands], list(points), list(purse))
    kind, origin, target = move_type(move), move_from(move), move_to(move)
    if kind not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_EN_PASSANT, MOVE_SWEEP, MOVE_PUSH):
        raise Unsupported(f"move kind {kind}")
    moving = out.board[origin]
    if moving == 0 or b.owner(moving) != side:
        raise Unsupported("a piece the mover does not own")
    taker_type = b.type_of(moving)
    # Part one: the step and what its kind adds; the takings in taking order.
    takings: list[Taking] = []
    captured, captured_square = out.board[target], target
    swept: list[tuple[int, int]] = []
    promoted: int | None = None
    if kind == MOVE_SWEEP:
        if move_promotion(move):
            raise Unsupported("a promoting sweep")
        swept = [(sq, out.board[sq]) for sq in compiled.sweep_path(board, move, side)]
    out.board[origin] = 0
    out.board[target] = moving
    if kind == MOVE_PROMOTION:
        promoted = move_promotion(move)
        out.board[target] = b.value(side, promoted)
    elif kind == MOVE_EN_PASSANT:
        captured_square = ep_victim
        captured = out.board[ep_victim]
        out.board[ep_victim] = 0
    elif kind == MOVE_PUSH:
        if move_promotion(move):
            promoted = move_promotion(move) - 1
            out.board[target] = b.value(side, promoted)
        captured, captured_square = _push_shift(compiled, out.board, move, board[target])
    for square, _ in swept:
        out.board[square] = 0
    _gate(compiled, b, out, side, move, origin)
    if captured:
        cause = "push" if kind == MOVE_PUSH else "landing"
        takings.append(Taking(captured_square, captured, cause))
    takings.extend(Taking(sq, value, "path") for sq, value in swept if value)
    walk = _Walk(taker_type, moving, side, origin, target, move_landing(move), target, target)
    if takings:
        mover_done = [False] * len(compiled.capture)
        rounds = takings
        while rounds:
            following: list[Taking] = []
            for index, entry in enumerate(compiled.capture):
                if entry.do == "pay":
                    _pay(compiled, b, entry, walk, rounds)
                elif entry.do == "hand":
                    _hand(compiled, b, entry, walk, rounds, out)
                elif entry.do == "blast":
                    _blast(compiled, b, entry, walk, rounds, following, out)
                else:
                    mover_done[index] = _strike(
                        compiled, b, entry, walk, rounds, mover_done[index], out,
                    )
            rounds = following
    out.purse[side] -= _cost(compiled, side, taker_type, promoted, walk.last_at)
    for payee, amount in walk.paid:
        out.points[payee] += amount
        if compiled.purse is not None and amount > 0:
            out.purse[payee] = min(out.purse[payee] + int(amount), PURSE_MAX)
    return out


def _push_shift(c: CompiledSpec, board: list[int], move: int, first: int) -> tuple[int, int]:
    """A push's line shifted one step from the target (the pusher already there): each piece
    carried onto the next square until one was empty; the piece pushed off the edge or onto a
    crushing wall is returned with the square it left, ``(0, sink)`` where none fell."""
    assert c.push is not None
    following = c.push.next[move_push(move)]
    carry, square = first, move_to(move)
    for _ in range(c.n_squares):
        behind = following[square]
        if behind < 0:
            return carry, square
        there = board[behind]
        board[behind] = carry
        if not there:
            return 0, behind
        carry, square = there, behind
    raise AssertionError("a push line with no end")


def _gate(c: CompiledSpec, b: _Board, out: Outcome, side: int, move: int, origin: int) -> None:
    """§3.2: a gated piece enters the origin from the hand in part one (Seirawan's gating;
    a castle's rook-square gate is outside the scope, being a castle)."""
    if c.drops is None or c.drops.gates is None or not move_gate(move):
        return
    piece_type = move_gate(move) - 1
    out.board[origin] = b.value(side, piece_type)
    if not (c.drops.unlimited and c.drops.unlimited[piece_type]):
        out.hands[side][piece_type] -= 1


def _cost(c: CompiledSpec, side: int, mover: int, promoted: int | None, square: int) -> int:
    """§3.2 (U9, U10): the move's own cost under ``rules.purse.upkeep``, for the type the move
    made the piece, on the square it last stood on when the list had run; 0 without it."""
    purse = c.purse
    if purse is None or not purse.upkeep:
        return 0

    def at(piece_type: int) -> int:
        return purse.move_cost(side, piece_type, square, c.n_squares)

    if promoted is None:
        return at(mover)
    return at(promoted) if purse.upkeep_new else at(mover) + at(promoted)


def _pay(c: CompiledSpec, b: _Board, entry: CaptureEntry, w: _Walk, rounds: list[Taking]) -> None:
    for taking in rounds:
        if not entry.reads(taking.cause, w.taker_type):
            continue
        owner = b.owner(taking.victim)
        if owner == w.player or not c.bounty:
            continue
        paid = c.bounty[b.type_of(taking.victim)]
        if not paid:
            continue
        payee = owner if c.purse is not None and c.purse.to_loser else w.player
        if payee >= 2:
            continue
        w.paid.append((payee, paid))


def _hand(c: CompiledSpec, b: _Board, entry: CaptureEntry, w: _Walk, rounds: list[Taking],
          out: Outcome) -> None:
    assert c.drops is not None
    for taking in rounds:
        if not taking.limbo or not entry.reads(taking.cause, w.taker_type):
            continue
        if b.owner(taking.victim) == w.player:
            continue
        taking.limbo = False
        slot = c.demotions[b.type_of(taking.victim)]
        if not (c.drops.unlimited and c.drops.unlimited[slot]):
            out.hands[w.player][slot] += 1


def _sets_off(c: CompiledSpec, b: _Board, w: _Walk, victim: int) -> bool:
    explosion = c.explosion
    assert explosion is not None
    if not victim:
        return False
    if explosion.by_types is not None and w.taker_type not in explosion.by_types:
        return False
    if explosion.of_types is not None and b.type_of(victim) not in explosion.of_types:
        return False
    # SR5a (T4): a player's own contact link is a meeting, not a capture (§2.5: offered only
    # by a contacts entry reading the landing).
    return not (b.owner(victim) == w.player and _meets_own(c))


def _blast(c: CompiledSpec, b: _Board, entry: CaptureEntry, w: _Walk, rounds: list[Taking],
           following: list[Taking], out: Outcome) -> None:
    explosion = c.explosion
    assert explosion is not None
    centres: list[int] = []
    at_taker = set_off = False
    for taking in rounds:
        if not entry.reads(taking.cause, w.taker_type) or not _sets_off(c, b, w, taking.victim):
            continue
        set_off = True
        if entry.spread and taking.cause in ("path", "push", "blast"):
            centres.append(taking.square)
        else:
            at_taker = True
        if not entry.spares_taken:
            taking.limbo = False
    if not set_off:
        return
    if at_taker:
        centres.append(w.taker_at if w.taker_at is not None else w.target)
    fresh: list[int] = []
    for centre in centres:
        if centre not in w.fired and centre not in fresh:
            fresh.append(centre)
    if not fresh:
        return
    if w.taker_at is not None:
        out.board[w.taker_at] = 0
        w.taker_at = None
    for centre in fresh:
        w.fired.add(centre)
        for square in explosion.blast[centre]:
            value = out.board[square]
            if value and b.type_of(value) not in explosion.immune_types and square not in w.settled:
                out.board[square] = 0
                following.append(Taking(square, value, "blast"))


def _strike(c: CompiledSpec, b: _Board, entry: CaptureEntry, w: _Walk, rounds: list[Taking],
            mover_done: bool, out: Outcome) -> bool:
    rule = _table(c, entry)
    n = rule.n_types
    mover_owner = b.owner(w.moving)

    def link(victim: int) -> tuple[int, int, bool]:
        cell = w.taker_type * n + b.type_of(victim)
        if b.owner(victim) == mover_owner:
            to = rule.own_onto_to[cell] if rule.own_onto_to else -1
            mover_to = rule.own_mover_to[cell] if rule.own_mover_to else -1
            return to, mover_to, False
        flips = bool(rule.mover_flips) and rule.mover_flips[cell]
        return rule.onto_to[cell], rule.mover_to[cell], flips

    if not mover_done and w.taker_at is not None:
        change: tuple[int, bool, bool] | None = None
        if entry.reads("landing", w.taker_type):
            for taking in rounds:
                if taking.cause != "landing":
                    continue
                to, mover_to, flips = link(taking.victim)
                stay = (to != -1 or mover_to >= 0) and rule.position == "stay"
                if mover_to >= 0 or flips or stay:
                    change = (mover_to, flips, stay)
        if change is None and entry.reads("path", w.taker_type):
            for taking in reversed(rounds):
                if taking.cause == "path" and link(taking.victim)[1] >= 0:
                    change = (link(taking.victim)[1], False, False)
                    break
        if change is not None:
            mover_done = True
            mover_to, flips, stay = change
            at = w.taker_at
            square = w.origin if stay and at != w.origin and out.board[w.origin] == 0 else at
            byte = out.board[at]
            owner = b.owner(byte)
            if flips:
                kind = mover_to if mover_to >= 0 else b.type_of(byte)
                now = b.value(1 - owner, kind)
            elif mover_to >= 0:
                now = b.value(owner, mover_to)
            else:
                now = byte
            out.board[at] = 0
            out.board[square] = now
            w.taker_at = w.last_at = square
    for taking in rounds:
        if not taking.limbo or not entry.reads(taking.cause, w.taker_type):
            continue
        to, _, _ = link(taking.victim)
        if to < 0:
            continue
        owner = b.owner(taking.victim)
        at: int | None
        if rule.position == "home":
            index = (owner * n + to) * c.n_squares + taking.square
            at = rule.rebirth[index] if rule.rebirth[index] >= 0 else None
        elif taking.cause == "landing":
            if rule.position == "stay":
                at = taking.square
            elif rule.position == "swap":
                at = w.origin
            elif rule.position == "choose":
                landings = rule.landing[w.player][taking.square]
                at = landings[w.landing_index] if w.landing_index < len(landings) else None
            else:
                at = None
        elif rule.position == "choose":
            at = None
        else:
            at = taking.square
        if at is None or out.board[at] != 0:
            continue
        out.board[at] = b.value(owner, to)
        w.settled.add(at)
        taking.limbo = False
    return mover_done


# ------------------------------------------------------------------------------ a round


@dataclass
class RoundTaking:
    """One taking of a round (R-L4b5): the captor's seat and type as it set out, the victim's
    byte and square, its cause (``"landing"``, ``"collision"``, ``"blast"``), whether a derived
    list's strike reads a collision's loser (``strikes``), whether the captor is gone (a tie, a
    shooter shot), a shot, the move word, the captor's block, where its blast centres, and a
    shot's victim already placed at step 0 (out of limbo)."""

    captor: int
    captor_type: int
    victim: int
    square: int
    origin: int | None
    cause: str
    strikes: bool
    gone: bool
    shot: bool
    m: int
    block: int
    centre: int
    placed: bool = False
    id: int = 0


class RoundHooks(Protocol):
    """The round's board writes `walk_round` asks for (`tests/simultaneous_oracle.py`)."""

    def pay(self, taken: RoundTaking) -> None: ...

    def file(self, taken: RoundTaking) -> None: ...

    def fires(self, taken: RoundTaking) -> bool: ...

    def reach(self, centre: int) -> tuple[int, ...]: ...

    def clear(self, centres: set[int], reach: set[int], settled: set[int]) -> list[tuple[int, int]]:
        """Clear the centres and every reached square the blast clears (not a settled one),
        ascending; return each cleared ``(square, value)`` that is not a centre."""
        ...

    def strike(self, rule: StrikeRule, taken: RoundTaking, in_limbo: bool) -> int | None:
        """A strike or contacts entry's two halves; the square a victim was placed on."""
        ...

    def shooter(self, taken: RoundTaking) -> None: ...


def default_round_list(c: CompiledSpec) -> list[CaptureEntry]:
    """The list a game's old keys say (the core's `CaptureProgram::derive`): ``pay``, ``hand``
    (a capture hand), ``blast``, ``strike`` or ``contacts``, with each kind's default causes,
    a blast that spares its taken piece and a strike that reads the blast's victims where
    ``explode_on_capture.strikes`` says so. A ``pay`` entry pays nothing where no bounty is."""
    entries = [CaptureEntry("pay", frozenset({"landing", "path", "blast", "push", "collision"}))]
    if c.drops is not None and c.drops.from_captures:
        entries.append(CaptureEntry("hand", frozenset({"landing", "path", "push", "collision"})))
    if c.explosion is not None:
        entries.append(CaptureEntry(
            "blast", frozenset({"landing", "path", "collision"}),
            spares_taken=bool(c.explosion.strikes & BLAST_STRIKES_TARGET),
        ))
    if c.strike is not None:
        on = {"landing", "collision"}
        if c.explosion is not None and c.explosion.strikes & BLAST_STRIKES_VICTIMS:
            on.add("blast")
        entries.append(CaptureEntry("strike", frozenset(on)))
    return entries


def walk_round(c: CompiledSpec, takings: list[RoundTaking], hooks: RoundHooks) -> None:
    """The capture list over a round's takings (the module's docstring)."""
    general = c.capture is not None
    entries = list(c.capture) if c.capture is not None else default_round_list(c)
    b = _Board(c)
    limbo: list[bool] = []
    for index, taken in enumerate(takings):
        taken.id = index
        limbo.append(not taken.placed)
    settled: set[int] = set()
    fired: set[int] = set()

    def reads(entry: CaptureEntry, taken: RoundTaking) -> bool:
        if taken.cause == "collision" and not general and entry.do in ("strike", "contacts"):
            return taken.strikes
        return taken.cause in entry.on

    rounds = takings
    while rounds:
        following: list[RoundTaking] = []
        for entry in entries:
            if entry.do == "pay":
                for taken in rounds:
                    if reads(entry, taken):
                        hooks.pay(taken)
            elif entry.do == "hand":
                for taken in rounds:
                    if limbo[taken.id] and reads(entry, taken) and b.owner(taken.victim) != taken.captor:
                        limbo[taken.id] = False
                        hooks.file(taken)
            elif entry.do == "blast":
                sources: list[tuple[RoundTaking, int]] = []
                for taken in rounds:
                    if not reads(entry, taken) or not hooks.fires(taken):
                        continue
                    if not entry.spares_taken:
                        limbo[taken.id] = False
                    centre = taken.square if taken.cause == "blast" and entry.spread else taken.centre
                    if centre not in fired:
                        sources.append((taken, centre))
                if not sources:
                    continue
                centres = {centre for _, centre in sources}
                reach = {square for centre in centres for square in hooks.reach(centre)}
                for square, value in hooks.clear(centres, reach, settled):
                    index = len(limbo)
                    limbo.append(True)
                    for source, centre in sources:
                        if square in hooks.reach(centre):
                            following.append(RoundTaking(
                                source.captor, source.captor_type, value, square, source.origin,
                                "blast", source.strikes, source.gone, False, source.m, source.block,
                                centre, False, index,
                            ))
                fired |= centres
            else:
                rule = _table(c, entry)
                for taken in rounds:
                    if not reads(entry, taken):
                        continue
                    if taken.shot:
                        hooks.shooter(taken)
                        continue
                    placed = hooks.strike(rule, taken, limbo[taken.id])
                    if placed is not None:
                        limbo[taken.id] = False
                        settled.add(placed)
        rounds = following
