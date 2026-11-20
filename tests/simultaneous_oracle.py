"""The declared Python oracle for a simultaneous round (schema 26, Q918a).

`docs/archive/plan-simultaneous.md` §6.1: resolution is checked by this oracle alone. It resolves
one round from the compiled tables and the two move words, in the plan's order (§2, §4),
and returns the position the core's `make_round` must reach -- board, hands, colouring,
tally, rights, clocks, the initiative, the void run, the bars and the resting pieces (schema
29's `pieces[].rest`, by `tests/rest_oracle.py`, the rest's own oracle); schema 27's swap
meeting is resolved here too, and `legal_for` filters a player's moves by the rest. Schema 28 (Q943a, `docs/archive/plan-speed.md` §2.4, §4.2): a round
between unequal speeds is two moves in order, each through the same step, the faster first,
the endings read between them, and the slower made, void or lost. CLAUDE.md rule 3 allows
it: it is a declared test oracle. It asks the engine only questions the engine's own tests
answer elsewhere -- the moves a player has on a board (for `lands_if_empty`, §5.3, and the
slower move's word, `offered`), what the wild pieces' phase does on a board
(`coming_phases`), whether a board ends the game (`terminal_prefix`, for the ordered round's
middle) and the slow definition of the key (`_compute_hash`) -- and never asks it to
resolve anything.

SR7e (R11, `docs/reference/rounds.md` §4 step 0): under `rules.strike` placed `"stay"` a
capture is a **shot**. The shooter stands whatever its victim chose, so it never collides;
both shots read their victims on the round's board and are written before anything
departs; a piece a shot strikes does not make its own move (it is *lost*); in step 5 the
shooter, if nothing struck or took it, becomes what its move made it (a promotion, a side
switch). In an ordered round the slower shot whose victim the faster move took away is void.

SR7g (R19, ruled 2026-10-27: *a collision like any other*): under ``rules.purse`` a buy is an
arrival with the round's moves, as a drop is. Its price leaves the buyer's purse at the
departures, whether or not the bought piece survives; two buys onto one square, or a buy and a
move, collide by ``beats`` and the tie, the bought piece by its own type. Every capture's
bounty (a wild piece's route bounty, a type's ``pieces[].bounty`` for any piece but the
captor's own) is paid to the taker, or under ``bounty_to: "loser"`` to the piece's owner
(nobody, for a nobody's piece), into the points and the purse (capped at ``PURSE_MAX``), a
blast's victims each to the captor whose blast reached them.

SR7n (R20): a held piece (``control: "majority"``) in a round. Its holder is read on the round's
board (the core's generator, asked as any player's moves are), and its move is that player's one
move of the round, an arrival as any other: it collides by ``beats`` and the tie, and is taken
as a nobody's piece is. It stays nobody's whatever it meets (its *block*, ``WILD``): a strike or
a link changes its type in nobody's block and never its side. Its claim is its holder's, and its
arrival on its holder's goal is read off the round's moves into ``goal_left_by``, as ``make``'s
``held_arrival`` reads one move: still standing nobody's where it moved, of a type the goal
admits, ``safe`` asked of the other player's captures onto it on the round's resolved board
(the oracle's reading of a threat: a capture move, so a held goal with ``safe`` is checked here
only beside no wild piece and no sight).
"""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import (
    CONTROL_MAJORITY,
    CONTROL_NEUTRAL,
    PURSE_MAX,
    WILD,
    CompiledSpec,
    StrikeRule,
)
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_FOLD,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    decode_move,
    move_fold,
    move_foreign,
    move_portal,
)
from chess_variant_sandbox.engine.state import GameState
from tests.capture_oracle import RoundTaking, walk_round
from tests.rest_oracle import after_turn, as_rest, legal, lists_of

NO_PAIR = (-1, -1)


@cache
def _tables(engine: Engine) -> dict[str, Any]:
    return build_tables(engine.spec)


def _entering(spec: CompiledSpec, player: int, piece_type: int) -> int:
    facing = spec.facing_initial[piece_type] if spec.facing_initial else 0
    return spec.encode_piece(player, piece_type, facing)


def _promoted(spec: CompiledSpec, player: int, move: int, piece_type: int) -> int:
    """What a piece ``player`` moved with ``move`` becomes promoting to ``piece_type``: the
    player's own piece entering play, or (SR8j, O15) for a held piece's word (foreign) a piece
    that stays nobody's, entering play as nobody's at its type's initial facing (nobody's
    frame), whoever held it."""
    return _entering(spec, WILD if move_foreign(move) else player, piece_type)


Tally = tuple[tuple[int, ...], tuple[int, ...]]


def _tally(spec: CompiledSpec, board: bytes) -> Tally:
    """SR8a (H3): each player's pieces of each type on ``board``, what an entering piece
    counts beside ``only_lost`` (`Game::tally_entries`): the round's start board."""
    counts = [[0] * spec.n_types, [0] * spec.n_types]
    for value in board:
        if value and spec.owner_of(value) < 2:
            counts[spec.owner_of(value)][spec.type_of(value)] += 1
    return tuple(counts[0]), tuple(counts[1])


def _arriving(
    spec: CompiledSpec, tally: Tally | None, player: int, piece_type: int, square: int
) -> int:
    """What enters play (`Game::arriving`): schema 9's first target, or beside ``only_lost``
    (SR8a, H3) the first target the player has fewer of in ``tally`` than it started with,
    and itself where none is."""
    if spec.entry_promotion is not None:
        becomes = spec.entry_promotion.at(player, piece_type, square)
        if becomes >= 0:
            rule = next(
                (r for r in spec.promotions if r.piece_type == piece_type and r.only_lost), None
            )
            if rule is None:
                return _entering(spec, player, becomes)
            assert tally is not None, "an only_lost entry needs the round's tally"
            for target in rule.targets[player]:
                if tally[player][target] < rule.opening[player][target]:
                    return _entering(spec, player, target)
    return _entering(spec, player, piece_type)


def _castling_partner(
    spec: CompiledSpec, board: bytearray, player: int, origin: int, target: int, promotion: int
) -> tuple[int, int, int]:
    """SR8d (R6): a castling's partner, ``(its square, its castled square, its value)``, read
    as `Game::castling_rook_squares` reads it: under castling by file the partner's file is
    the move's promotion field and its castled file the rule's for that side; by steps, the
    first of ``player``'s partner type outward from the king, which lands on the square the
    king crossed last."""
    castling = spec.castling
    assert castling is not None
    width = spec.width
    base = (origin // width) * width
    if castling.king_files is not None:
        assert castling.rook_files is not None
        kingside = promotion > origin % width
        frm = base + promotion
        to = base + castling.rook_files[1 if kingside else 0]
        return frm, to, board[frm]
    step = 1 if target > origin else -1
    file = origin % width + step
    while 0 <= file < width:
        square = base + file
        value = board[square]
        if value and spec.owner_of(value) == player and spec.type_of(value) == castling.rook_type:
            return square, target - step, value
        file += step
    raise AssertionError("a castling with no partner")


def _in_block(
    spec: CompiledSpec, tally: Tally | None, block: int, piece_type: int, square: int
) -> int:
    """SR7n: a piece of ``piece_type`` entering ``block``: a player's arriving value, or a
    nobody's byte (facing 0, no zone), as `Tables::arriving_value` reads the third owner."""
    if block == WILD:
        return spec.encode_piece(WILD, piece_type, 0)
    return _arriving(spec, tally, block, piece_type, square)


def _rehome(
    spec: CompiledSpec, tally: Tally | None, value: int, new_owner: int, square: int
) -> int:
    """TC2 (`docs/plan-turncoat.md` §6): a piece changing sides keeps its heading, unless it
    has one orientation or `promote_on_entry` turns it into another type there."""
    piece_type = spec.type_of(value)
    arrived = _arriving(spec, tally, new_owner, piece_type, square)
    if not spec.has_facing or spec.facing_axes[piece_type] < 2 or spec.type_of(arrived) != piece_type:
        return arrived
    kept = spec.coat_facing[spec.owner_of(value)][spec.facing_of(value)]
    return spec.encode_piece(new_owner, piece_type, kept)


def _meeting(spec: CompiledSpec, mover_type: int, victim: int, mover_owner: int) -> tuple[int, int, bool]:
    if spec.strike is None:
        return -1, -1, False
    return _meeting_in(spec, spec.strike, mover_type, victim, mover_owner)


def _meeting_in(
    spec: CompiledSpec, strike: StrikeRule, mover_type: int, victim: int, mover_owner: int
) -> tuple[int, int, bool]:
    """`_meeting` read off ``strike``: the strike's table, or (R-L4b5, T3) a contact table."""
    owner = spec.owner_of(victim)
    cell = mover_type * strike.n_types + spec.type_of(victim)
    if owner == mover_owner:
        # A blast's victim may be the captor's own piece (R-L4b5): no own link, no change.
        return (
            strike.own_onto_to[cell] if strike.own_onto_to else -1,
            strike.own_mover_to[cell] if strike.own_mover_to else -1,
            False,
        )
    flips = owner != WILD and bool(strike.mover_flips) and strike.mover_flips[cell]
    return strike.onto_to[cell], strike.mover_to[cell], flips


def lands_if_empty(engine: Engine, state: GameState, board: bytes, player: int, move: int) -> bool:
    """§5.3: the move's origin and destination are offered with the destination empty."""
    origin, target, _, _ = decode_move(move)
    emptied = bytearray(board)
    emptied[target] = 0
    probe = state.with_(board=bytes(emptied), side_to_move=player)
    for other in engine.pseudo_legal_moves(probe):
        o, t, kind, _ = decode_move(other)
        if kind not in (MOVE_DROP, MOVE_BUY, MOVE_ROTATE) and o == origin and t == target:
            return True
    return False


def _standoff_after(state: GameState, void: list[bool], origins: list[int | None], touched: set[int]):
    run = NO_PAIR
    bars = set(state.bars)
    barred = False
    if void[0] and void[1]:
        pair = (origins[0], origins[1])
        if state.void_run == pair:
            bars.add(pair)
            barred = True
        else:
            run = pair
    if run != NO_PAIR and (run[0] in touched or run[1] in touched):
        run = NO_PAIR
    bars = {bar for bar in bars if bar[0] not in touched and bar[1] not in touched}
    return run, tuple(sorted(bars)), barred


def _moves_portal_end(spec: CompiledSpec, plan: dict[str, Any]) -> bool:
    """SR8e (R9): whether a round's move moves a portal end -- a fold's own move, or a move of
    a momentum end's byte."""
    if spec.portals is None:
        return False
    mover = plan["mover"]
    return plan["kind"] == MOVE_FOLD or (
        plan["origin"] is not None and mover != 0 and spec.owner_of(mover) == WILD
        and spec.type_of(mover) in spec.portals.end_types
    )


def _fold_slot(spec: CompiledSpec, end: int) -> int:
    """SR8e: where fold end ``end`` stands in `GameState.folds`, the fold ends in end order."""
    assert spec.portals is not None
    return [e for e, kind in enumerate(spec.portals.kinds) if kind == "fold"].index(end)


def _end_squares(spec: CompiledSpec, board: bytes | bytearray) -> dict[int, int]:
    """SR8e: where each momentum end stands on ``board``, by end."""
    rule = spec.portals
    assert rule is not None
    out: dict[int, int] = {}
    for square, value in enumerate(board):
        if value and spec.owner_of(value) == WILD and spec.type_of(value) in rule.end_types:
            out.setdefault(rule.end_types.index(spec.type_of(value)), square)
    return out


def _read_through(
    spec: CompiledSpec, player: int, move: int, ends_before: dict[int, int],
    ends_now: dict[int, int], folds_before: list[int], folds_now: list[int],
) -> int | None:
    """SR8e (R9): ``move`` read through the ends where they now stand: a transit whose far end
    moved comes out one step beyond where it stands now, in the direction it would have come
    out before; a go-through whose far fold moved, on that fold's square; ``None`` where that
    is off the board. Every other word is itself."""
    rule = spec.portals
    assert rule is not None
    _, target, kind, _ = decode_move(move)

    def retarget(square: int) -> int:
        return (move & ~(0xFF << 8)) | (square << 8)

    if kind == MOVE_PORTAL:
        far = rule.end_types.index(rule.dest[move_portal(move)])
        before, now = ends_before.get(far), ends_now.get(far)
        if before == now:
            return move
        if before is None or now is None:
            return None
        exits = rule.exits[player]
        for direction, square in enumerate(exits[before]):
            if square == target:
                exit_now = exits[now][direction]
                return None if exit_now < 0 else retarget(exit_now)
        return None
    if rule.folds is not None and kind in (MOVE_NORMAL, MOVE_PROMOTION) and move_fold(move):
        far = _fold_slot(spec, rule.end_types.index(rule.dest[rule.end_types[move_fold(move) - 1]]))
        if folds_now[far] == folds_before[far]:
            return move
        return None if folds_now[far] < 0 else retarget(folds_now[far])
    return move


def legal_for(engine: Engine, state: GameState, player: int) -> list[int]:
    """`player`'s moves in a round under the rest (schema 29, `docs/plan-rest.md` §3.7; schema
    27's round rest before it): `tests/rest_oracle.py::legal`, the moves with nothing resting
    less the resting pieces', unless that would leave none (the lift). Without a rest the
    position rests nothing and this is the core's list."""
    return legal(engine, state, player)


def offered(engine: Engine, state: GameState, player: int, move: int) -> bool:
    """Schema 28 (`docs/archive/plan-speed.md` §2.5): the word ``move`` is among the moves
    ``player``'s generator offers on this position, without the list-level filters."""
    return move in engine.pseudo_legal_moves(state.with_(side_to_move=player))


def resolve_round(engine: Engine, state: GameState, first: int, second: int) -> GameState:
    """The position after one round, resolved by the plan's rule (§2, §4).

    Schema 28 (Q943a, `docs/archive/plan-speed.md` §2.4): when the two movers' speeds differ, the
    faster move is resolved alone, the endings are read, and the slower is resolved alone on
    what is left if its piece still stands unchanged and its word is still offered."""
    spec = engine.spec
    sim = spec.simultaneous
    assert sim is not None
    tables = _tables(engine)
    n_types = spec.n_types
    board = bytearray(state.board)
    hands = [list(state.hands[0]), list(state.hands[1])]
    colour = bytearray(state.colour)
    points = list(state.points)
    purse = list(state.purse)
    unmoved = state.unmoved
    initiative = state.initiative
    moves = (first, second)
    # SR8a (H3): what an entering piece counts beside `only_lost`, the round's start board.
    tally = _tally(spec, state.board)

    portal_rule = spec.portals
    folds = list(state.folds)

    def make_plan(player: int, move: int) -> dict[str, Any]:
        origin, target, kind, promotion = decode_move(move)
        if kind in (MOVE_DROP, MOVE_BUY):
            plan: dict[str, Any] = {
                "m": move, "kind": kind, "origin": None, "target": target, "promotion": promotion,
                "mover": _arriving(spec, tally, player, promotion, target), "type": promotion,
            }
        elif kind == MOVE_FOLD:
            # SR8e (R9): a fold's own move writes no board byte; its piece is the fold's end.
            assert portal_rule is not None
            plan = {
                "m": move, "kind": kind, "origin": origin, "target": target, "promotion": promotion,
                "mover": 0, "type": portal_rule.end_types[move_fold(move)],
            }
        else:
            plan = {
                "m": move, "kind": kind, "origin": origin, "target": target, "promotion": promotion,
                "mover": board[origin], "type": spec.type_of(board[origin]),
            }
        # SR8d (R6): a castling's partner on the round's board, as `castling_rook_squares`.
        plan["partner"] = (
            _castling_partner(spec, board, player, origin, target, promotion)
            if kind == MOVE_CASTLE else None
        )
        # SR7n (R20): the block the moving piece stands in, WILD for a held piece.
        plan["block"] = (
            WILD if plan["origin"] is not None and spec.owner_of(plan["mover"]) == WILD
            else player
        )
        return plan

    plans = [make_plan(player, move) for player, move in enumerate(moves)]

    touched: set[int] = set()
    flags = {"material": False, "promoted": False, "one_way": False, "kind": False,
             "passed": False, "laid": False}

    wild_types = {t for t in spec.npcs.piece if t >= 0} if spec.npcs is not None else set()

    def pay(victim: int, square: int, captor: int) -> None:
        """SR7g: what taking ``victim`` pays, as `Tables::bounty_paid` and `bounty_payee` say."""
        owner = spec.owner_of(victim)
        if owner == captor:
            return
        paid = 0.0
        if owner == WILD and spec.type_of(victim) in wild_types:
            paid += tables["npc_bounty"][square]
        if spec.bounty:
            paid += spec.bounty[spec.type_of(victim)]
        payee = owner if spec.purse is not None and spec.purse.to_loser else captor
        if paid == 0 or payee >= 2:
            return
        points[payee] += paid
        if spec.purse is not None and paid > 0:
            purse[payee] = min(purse[payee] + int(paid), PURSE_MAX)

    def worth(value: int) -> float:
        return -1.0 if value == 0 else spec.piece_values[spec.type_of(value)]

    def stays_put(plan: dict[str, Any], player: int) -> bool:
        strike = spec.strike
        if (
            strike is None or strike.position != "stay" or plan["origin"] is None
            or plan["kind"] == MOVE_CASTLE
        ):
            return False
        victim = board[plan["target"]]
        if victim == 0 or plan["origin"] == plan["target"]:
            return False
        # SR4j (T10): a switch alone advances; only a meeting the strike reaches stays.
        becomes, mover_becomes, _ = _meeting(spec, plan["type"], victim, player)
        return becomes != -1 or mover_becomes >= 0

    def contact_in_place(player: int, plan: dict[str, Any]) -> bool:
        origin, target = plan["origin"], plan["target"]
        victim, mover = board[target], board[origin]
        becomes, mover_becomes, flips = _meeting(spec, plan["type"], victim, player)
        flips = flips and plan["block"] != WILD
        as_type = mover_becomes if mover_becomes >= 0 else spec.type_of(mover)
        if flips:
            now = _entering(spec, player, as_type) if mover_becomes >= 0 else mover
            attacker = _rehome(spec, tally, now, 1 - player, origin)
        elif mover_becomes >= 0:
            attacker = _in_block(spec, tally, plan["block"], as_type, origin)
        else:
            attacker = mover
        survivor = _arriving(spec, tally, spec.owner_of(victim), becomes, target) if becomes >= 0 else 0
        board[origin] = attacker
        board[target] = survivor
        touched.update((origin, target))
        return worth(attacker) != worth(mover) or worth(survivor) != worth(victim)

    def here() -> GameState:
        return state.with_(
            board=bytes(board), hands=(tuple(hands[0]), tuple(hands[1])), colour=bytes(colour),
            points=(points[0], points[1]), unmoved=unmoved, initiative=initiative,
            purse=(purse[0], purse[1]), folds=tuple(folds),
        )

    def step(
        active: tuple[bool, bool],
    ) -> tuple[list[bool], list[bool], list[bool], list[bool]]:
        """§2.2 to §4 step 7 for the players ``active`` names (both at equal speeds)."""
        nonlocal unmoved, initiative
        raw = [active[i] and plans[i]["origin"] != plans[i]["target"] for i in (0, 1)]
        stays = [active[i] and stays_put(plans[i], i) for i in (0, 1)]
        # SR7e (R11): a shot stands, and strikes its victim before anything moves.
        shot = [shoots and stays[i] for i in (0, 1)]
        struck = [
            active[j] and shot[1 - j] and not shot[j]
            and plans[j]["origin"] == plans[1 - j]["target"]
            for j in (0, 1)
        ]
        standing = [False, False]
        for i in (0, 1):
            j = 1 - i
            there = board[plans[i]["target"]]
            own_target = stays[i] and there != 0 and spec.owner_of(there) == i
            vacated = raw[j] and not stays[j] and plans[j]["origin"] == plans[i]["target"]
            standing[i] = active[i] and not struck[i] and (
                plans[i]["kind"] == MOVE_ROTATE or shot[i]
                or (stays[i] and (own_target or not vacated))
            )
        moving = [raw[i] and not standing[i] and not struck[i] for i in (0, 1)]
        # Schema 27 (Q942a, R2): a swap meets as a collision, so neither move is void.
        swapped = (
            sim.swap_collides and moving[0] and moving[1]
            and plans[0]["origin"] is not None
            and plans[1]["origin"] == plans[0]["target"]
            and plans[0]["origin"] == plans[1]["target"]
        )
        step_board = bytes(board)
        void = [False, False]
        for i in (0, 1):
            j = 1 - i
            void[i] = (
                not swapped and moving[i] and moving[j] and plans[i]["origin"] is not None
                and plans[j]["origin"] == plans[i]["target"]
                and not lands_if_empty(engine, state, step_board, i, plans[i]["m"])
            )
        arriving = [moving[i] and not void[i] for i in (0, 1)]

        # SR7f (R18): two moves that stand and strike one nobody's piece (two shots, two
        # contacts that stay) meet as a collision: `beats` between the strikers, then the
        # tie; the winner's strike alone is made, the loser (both, with no winner) misses.
        missed = [False, False]
        target = plans[0]["target"]
        victim = board[target]
        if (
            standing[0] and standing[1] and plans[1]["target"] == target
            and victim != 0 and spec.owner_of(victim) == WILD
            and all(plans[i]["kind"] != MOVE_ROTATE and plans[i]["origin"] != target for i in (0, 1))
        ):
            types = [plans[0]["type"], plans[1]["type"]]
            beats = [
                sim.collision_allowed[types[0] * n_types + types[1]],
                sim.collision_allowed[types[1] * n_types + types[0]],
            ]
            if beats == [True, False]:
                contest: int | None = 0
            elif beats == [False, True]:
                contest = 1
            elif sim.tie == 1:
                contest = initiative
                initiative = 1 - initiative
                flags["passed"] = True
            else:
                contest = None
            missed = [contest != i for i in (0, 1)]
            shot = [shot[i] and not missed[i] for i in (0, 1)]

        for i in (0, 1):
            if arriving[i] or (standing[i] and not missed[i]):
                kind, promotion = plans[i]["kind"], plans[i]["promotion"]
                flags["one_way"] |= plans[i]["type"] in spec.one_way_types
                plain_portal = kind == MOVE_PORTAL and promotion == 0
                flags["kind"] |= kind not in (MOVE_NORMAL, MOVE_ROTATE) and not plain_portal
                flags["promoted"] |= (
                    kind in (MOVE_PROMOTION, MOVE_BUY) or (kind == MOVE_PORTAL and promotion != 0)
                )

        # Step 0 (SR7e): the shots, both read on the round's board, then written together.
        takings: list[dict[str, Any]] = []
        shot_writes = []
        for i in (0, 1):
            if not shot[i]:
                continue
            plan = plans[i]
            victim = board[plan["target"]]
            becomes, _, _ = _meeting(spec, plan["type"], victim, i)
            survivor = _arriving(spec, tally, spec.owner_of(victim), becomes, plan["target"]) if becomes >= 0 else 0
            shot_writes.append((plan["target"], survivor))
            takings.append({
                "captor": i, "captor_type": plan["type"], "victim": victim,
                "square": plan["target"], "origin": plan["origin"], "strikes": False,
                "gone": shot[1 - i] and plan["origin"] == plans[1 - i]["target"],
                "shot": True, "m": plan["m"], "block": plan["block"],
                "cause": "landing", "placed": survivor != 0,
            })
        for square, value in shot_writes:
            board[square] = value
            touched.add(square)
            flags["material"] = True

        # Step 2: departures.
        for i in (0, 1):
            if not arriving[i]:
                continue
            if plans[i]["origin"] is not None:
                board[plans[i]["origin"]] = 0
                touched.add(plans[i]["origin"])
                # SR8d: a file-based king landing on its partner lifts it here.
                partner = plans[i]["partner"]
                if partner is not None and partner[0] == plans[i]["target"]:
                    board[partner[0]] = 0
                    touched.add(partner[0])
            elif plans[i]["kind"] == MOVE_BUY:
                # SR7g: the price is paid whether or not the bought piece survives.
                rule = spec.purse
                assert rule is not None
                purse[i] -= rule.price[rule.market.index(plans[i]["type"])]
            else:
                drops = spec.drops
                assert drops is not None
                slot = drops.hand_slot(plans[i]["type"])
                if not (drops.unlimited and drops.unlimited[slot]):
                    hands[i][slot] -= 1

        # Step 3: each move's own consequences.
        arrival = [0, 0]
        for i in (0, 1):
            if not active[i] or struck[i] or shot[i] or missed[i]:
                continue
            plan = plans[i]
            if plan["kind"] == MOVE_ROTATE:
                board[plan["origin"]] = spec.encode_piece(i, plan["type"], plan["promotion"])
                touched.add(plan["origin"])
                continue
            if standing[i]:
                there = board[plan["target"]]
                if there != 0 and spec.owner_of(there) == i:
                    flags["material"] |= contact_in_place(i, plan)
                continue
            if not arriving[i]:
                continue
            value = plan["mover"]
            if plan["kind"] == MOVE_PROMOTION:
                value = _promoted(spec, i, plan["m"], plan["promotion"])
            elif plan["kind"] == MOVE_PORTAL and plan["promotion"] != 0:
                value = _promoted(spec, i, plan["m"], plan["promotion"] - 1)
            occupant = board[plan["target"]]
            if occupant != 0 and spec.owner_of(occupant) == i:
                _, mover_becomes, _ = _meeting(spec, plan["type"], occupant, i)
                if mover_becomes >= 0:
                    value = _arriving(spec, tally, i, mover_becomes, plan["target"])
                board[plan["target"]] = 0
                touched.add(plan["target"])
                flags["material"] = True
            arrival[i] = value

        # Step 4: the meetings.

        def taking(captor: int, victim: int, square: int, strikes: bool) -> dict[str, Any]:
            return {
                "captor": captor, "captor_type": plans[captor]["type"], "victim": victim,
                "square": square, "origin": plans[captor]["origin"],
                "strikes": strikes and spec.strike is not None, "gone": False,
                "shot": False, "m": plans[captor]["m"], "block": plans[captor]["block"],
                "cause": "landing", "placed": False,
            }

        def meet() -> tuple[int | None, list[bool]]:
            """Who wins a meeting of the two arrivals (`beats`, then the tie rule)."""
            nonlocal initiative
            types = [spec.type_of(arrival[0]), spec.type_of(arrival[1])]
            beats = [
                sim.collision_allowed[types[0] * n_types + types[1]],
                sim.collision_allowed[types[1] * n_types + types[0]],
            ]
            if beats == [True, False]:
                return 0, beats
            if beats == [False, True]:
                return 1, beats
            if sim.tie == 1:
                holder = initiative
                initiative = 1 - initiative
                flags["passed"] = True
                return holder, beats
            return None, beats

        collided = arriving[0] and arriving[1] and plans[0]["target"] == plans[1]["target"]
        if swapped:
            # Both destinations were emptied by the departures; each is taken where it would
            # have landed, on the other's origin.
            flags["material"] = True
            touched.update((plans[0]["target"], plans[1]["target"]))
            winner, beats = meet()
            if winner is not None:
                square = plans[winner]["target"]
                board[square] = arrival[winner]
                loser = taking(winner, arrival[1 - winner], square, False)
                loser["strikes"] = sim.collision_strikes
                loser["cause"] = "collision"
                takings.append(loser)
            elif beats[0] and beats[1]:
                for w in (0, 1):
                    each = taking(w, arrival[1 - w], plans[w]["target"], False)
                    each["strikes"] = sim.collision_strikes
                    each["cause"] = "collision"
                    each["gone"] = True
                    takings.append(each)
        elif collided:
            square = plans[0]["target"]
            touched.add(square)
            flags["material"] = True
            winner, beats = meet()
            there = board[square]
            if winner is not None:
                if there != 0:
                    takings.append(taking(winner, there, square, True))
                board[square] = arrival[winner]
                loser = taking(winner, arrival[1 - winner], square, False)
                loser["strikes"] = sim.collision_strikes
                loser["cause"] = "collision"
                takings.append(loser)
            else:
                board[square] = 0
                if beats[0] and beats[1]:
                    for w in (0, 1):
                        each = taking(w, arrival[1 - w], square, False)
                        each["strikes"] = sim.collision_strikes
                        each["cause"] = "collision"
                        each["gone"] = True
                        takings.append(each)
        else:
            for i in (0, 1):
                if not arriving[i]:
                    continue
                square = plans[i]["target"]
                touched.add(square)
                occupant = board[square]
                if occupant != 0:
                    flags["material"] = True
                    takings.append(taking(i, occupant, square, True))
                board[square] = arrival[i]
        for i in (0, 1):
            plan = plans[i]
            if not standing[i] or plan["kind"] == MOVE_ROTATE or shot[i] or missed[i]:
                continue
            victim = board[plan["target"]]
            if victim == 0 or spec.owner_of(victim) == i or board[plan["origin"]] != plan["mover"]:
                continue
            flags["material"] |= contact_in_place(i, plan)

        # SR8d (R6, ruled 2026-10-27): the castling partners, after the meetings. The partner
        # moves to its castled square if that is empty, else stays; one the king landed on
        # (lifted above) goes there if empty, else it is removed. Both rights go, as `make`
        # spends them.
        for i in (0, 1):
            partner = plans[i]["partner"]
            if partner is None or not arriving[i]:
                continue
            frm, to, value = partner
            if frm == plans[i]["target"]:
                if board[to] == 0:
                    board[to] = value
                    touched.add(to)
                else:
                    flags["material"] = True
            elif board[frm] == value and board[to] == 0:
                board[frm], board[to] = 0, value
                touched.update((frm, to))
            unmoved &= ~(1 << frm) & ~(1 << to)

        # Steps 5 and 6 (R-L4b5): the capture list walked over the round's takings
        # (`tests/capture_oracle.py::walk_round`); the round's board writes are these hooks.
        explosion = spec.explosion

        def hook_pay(taken: RoundTaking) -> None:
            pay(taken.victim, taken.square, taken.captor)

        def hook_file(taken: RoundTaking) -> None:
            drops = spec.drops
            assert drops is not None
            slot = spec.demotions[spec.type_of(taken.victim)]
            if not (drops.unlimited and drops.unlimited[slot]):
                hands[taken.captor][slot] += 1

        def hook_fires(taken: RoundTaking) -> bool:
            assert explosion is not None
            return (explosion.by_types is None or taken.captor_type in explosion.by_types) and (
                explosion.of_types is None or spec.type_of(taken.victim) in explosion.of_types
            )

        def hook_reach(centre: int) -> tuple[int, ...]:
            assert explosion is not None
            return tuple(explosion.blast[centre])

        def hook_clear(centres: set[int], reach: set[int], settled: set[int]) -> list[tuple[int, int]]:
            assert explosion is not None
            cleared = []
            for square in range(spec.n_squares):
                value = board[square]
                if value == 0 or square in settled:
                    continue
                if square not in centres and not (
                    square in reach and spec.type_of(value) not in explosion.immune_types
                ):
                    continue
                if square not in centres:
                    cleared.append((square, value))
                board[square] = 0
                touched.add(square)
                flags["material"] = True
            return cleared

        def hook_strike(rule: StrikeRule, taken: RoundTaking, in_limbo: bool) -> int | None:
            nonlocal unmoved
            becomes, mover_becomes, flips = _meeting_in(spec, rule, taken.captor_type, taken.victim, taken.captor)
            flips = flips and taken.block != WILD
            square, captor = taken.square, taken.captor
            there = board[square]
            if taken.cause != "blast" and not taken.gone and there != 0 and spec.owner_of(there) == taken.block and (
                mover_becomes >= 0 or flips
            ):
                as_type = mover_becomes if mover_becomes >= 0 else spec.type_of(there)
                if flips:
                    now = _entering(spec, captor, as_type) if mover_becomes >= 0 else there
                    board[square] = _rehome(spec, tally, now, 1 - captor, square)
                else:
                    board[square] = _in_block(spec, tally, taken.block, as_type, square)
            if becomes < 0 or not in_limbo:
                return None
            owner = spec.owner_of(taken.victim)
            place = None
            if rule.position == "home":
                home = rule.rebirth[(owner * n_types + becomes) * spec.n_squares + square]
                place = home if home >= 0 else None
            elif rule.position == "choose":
                place = None
            elif taken.cause == "blast":
                place = square
            elif rule.position == "swap":
                place = taken.origin
            if place is None or board[place] != 0:
                return None
            board[place] = _arriving(spec, tally, owner, becomes, place)
            touched.add(place)
            if rule.position == "home":
                unmoved |= 1 << place
            else:
                unmoved &= ~(1 << place)
            return place

        def hook_shooter(taken: RoundTaking) -> None:
            shooter_after({
                "origin": taken.origin, "captor": taken.captor, "gone": taken.gone,
                "block": taken.block, "m": taken.m, "captor_type": taken.captor_type,
                "victim": taken.victim,
            })

        if takings:
            hooks = SimpleNamespace(
                pay=hook_pay, file=hook_file, fires=hook_fires, reach=hook_reach, clear=hook_clear,
                strike=hook_strike, shooter=hook_shooter,
            )
            walk_round(spec, [
                RoundTaking(
                    t["captor"], t["captor_type"], t["victim"], t["square"], t["origin"], t["cause"],
                    t["strikes"], t["gone"], t["shot"], t["m"], t["block"], t["square"], t["placed"],
                )
                for t in takings
            ], hooks)  # type: ignore[arg-type]

        # Step 7: claims.
        territory = spec.territory
        if territory is not None:
            for i in (0, 1):
                if active[i] and not struck[i] and plans[i]["kind"] == MOVE_ROTATE:
                    square = plans[i]["origin"]
                elif arriving[i]:
                    square = plans[i]["target"]
                else:
                    continue
                value = board[square]
                if value == 0 or spec.owner_of(value) != plans[i]["block"]:
                    continue
                want, before = i + 1, colour[square]
                if before != want and (territory.mode == "repaint" or before == 0):
                    colour[square] = want
                    flags["laid"] |= territory.mode != "repaint"
        return arriving, [standing[i] and not missed[i] for i in (0, 1)], void, struck

    def shooter_after(taken: dict[str, Any]) -> None:
        """SR7e, §4 step 5 for a shot: the shooter on its origin becomes what its move made
        it (the promoted piece, then a side switch), unless struck or taken this round."""
        origin, captor = taken["origin"], taken["captor"]
        there = board[origin]
        if taken["gone"] or there == 0 or spec.owner_of(there) != taken["block"]:
            return
        _, _, kind, promotion = decode_move(taken["m"])
        value = there
        if kind == MOVE_PROMOTION:
            value = _promoted(spec, captor, taken["m"], promotion)
        elif kind == MOVE_PORTAL and promotion != 0:
            value = _promoted(spec, captor, taken["m"], promotion - 1)
        _, mover_becomes, flips = _meeting(spec, taken["captor_type"], taken["victim"], captor)
        flips = flips and taken["block"] != WILD
        if mover_becomes >= 0:
            value = _in_block(spec, tally, taken["block"], mover_becomes, origin)
        if flips:
            now = _entering(spec, captor, mover_becomes) if mover_becomes >= 0 else value
            value = _rehome(spec, tally, now, 1 - captor, origin)
        if value != there:
            board[origin] = value
            touched.add(origin)

    def goal_left(arriving: list[bool]) -> int:
        reach = spec.reach
        assert reach is not None
        left = [False, False]
        for i in (0, 1):
            origin = plans[i]["origin"]
            if origin is None:
                continue
            value = board[plans[i]["target"]]
            left[i] = (
                arriving[i] and origin in reach.squares[i]
                and reach.admits(plans[i]["type"])
                and value != 0 and spec.owner_of(value) == plans[i]["block"]
            )
        return {(True, True): 2, (True, False): 0, (False, True): 1}.get(tuple(left), -1)

    def held_arrivals(arriving: list[bool]) -> int:
        """SR7n (R20, O11's reading): which players' held pieces arrived on their goals."""
        reach = spec.reach
        assert reach is not None and spec.neutrals is not None
        kind = spec.neutrals.kind
        arrived = [False, False]
        for i in (0, 1):
            plan = plans[i]
            standing = board[plan["target"]]
            if not (
                arriving[i] and plan["block"] == WILD
                and kind[spec.type_of(plan["mover"])] == CONTROL_MAJORITY
                and standing != 0 and spec.owner_of(standing) == WILD
                and kind[spec.type_of(standing)] in (CONTROL_MAJORITY, CONTROL_NEUTRAL)
                and plan["target"] in reach.squares[i]
                and reach.admits(spec.type_of(standing))
            ):
                continue
            if reach.safe:
                assert spec.npcs is None and spec.sight is None, "a threat read as a capture"
                probe = here().with_(side_to_move=1 - i)
                if any(
                    decode_move(m)[1] == plan["target"]
                    and decode_move(m)[2] not in (MOVE_DROP, MOVE_BUY)
                    for m in engine.pseudo_legal_moves(probe)
                ):
                    continue
            arrived[i] = True
        return {(True, True): 2, (True, False): 0, (False, True): 1}.get(tuple(arrived), -1)

    leaves = spec.reach is not None and spec.reach.leaves
    reach_held = (
        spec.reach is not None and not spec.reach.leaves
        and spec.neutrals is not None and spec.neutrals.has_majority
    )
    shoots = bool(sim.collision_strikes) and spec.strike is not None and spec.strike.position == "stay"
    speeds = [sim.speed[plan["type"]] for plan in plans] if sim.speed else [0, 0]
    gone = [False, False]
    ended = False
    goal_left_by = state.goal_left_by
    portal = [_moves_portal_end(spec, plan) for plan in plans]
    if portal[0] or portal[1]:
        # SR8e (R9, ruled 2026-10-27): the round's portal moves are made first, and the one
        # piece move then reads the ends where they now stand, resolved alone (as an ordered
        # round's slower move) if its word, read through the moved ends, is offered there. A
        # portal end rests nothing (`gone`).
        assert portal_rule is not None
        ends_before = _end_squares(spec, board)
        folds_before = list(folds)
        made = list(portal)
        if portal[0] and portal[1]:
            same_end = plans[0]["type"] == plans[1]["type"]
            same_square = plans[0]["target"] == plans[1]["target"]
            if same_end and same_square:
                made[1] = False
            elif same_end or same_square:
                made = [False, False]
        for i in (0, 1):
            if not made[i]:
                continue
            origin, target = plans[i]["origin"], plans[i]["target"]
            if plans[i]["kind"] == MOVE_FOLD:
                folds[_fold_slot(spec, move_fold(plans[i]["m"]))] = target
            else:
                board[target], board[origin] = board[origin], 0
                unmoved &= ~(1 << origin) & ~(1 << target)
            touched.update((origin, target))
        arriving, standing, void = [False, False], [False, False], [False, False]
        gone = list(portal)
        if portal[0] != portal[1]:
            s = 1 if portal[0] else 0
            read = _read_through(
                spec, s, plans[s]["m"], ends_before, _end_squares(spec, board), folds_before, folds
            )
            if read is not None and offered(engine, here(), s, read):
                plans[s] = make_plan(s, read)
                made_s, stood, _, _ = step((s == 0, s == 1))
                arriving[s], standing[s] = made_s[s], stood[s]
    elif speeds[0] == speeds[1]:
        arriving, standing, void, gone = step((True, True))
    else:
        # Schema 28: two moves in order, the faster first.
        f = 0 if speeds[0] > speeds[1] else 1
        s = 1 - f
        arriving, standing, void = [False, False], [False, False], [False, False]
        slower_shot = shoots and stays_put(plans[s], s)
        made, stood, _, _ = step((f == 0, f == 1))
        arriving[f], standing[f] = made[f], stood[f]
        if leaves:
            goal_left_by = goal_left(arriving)
        elif reach_held:
            goal_left_by = held_arrivals(arriving)
        # §2.6: the endings after the faster move, asked of the position it left with the
        # clocks and the repetition window set aside (they are read at the round's end).
        middle = here().with_(goal_left_by=goal_left_by, halfmove=0, ply=0, repetition=())
        ended = engine.terminal_prefix(middle) is not None
        origin = plans[s]["origin"]
        if ended or origin is not None and (origin in touched or board[origin] != plans[s]["mover"]):
            gone[s] = True
        elif offered(engine, here(), s, plans[s]["m"]) and not (
            slower_shot and not stays_put(plans[s], s)
        ):
            made, stood, _, _ = step((s == 0, s == 1))
            arriving[s], standing[s] = made[s], stood[s]

    # Step 8: the wild pieces, as the engine's phase would walk them on this board -- not
    # after a round that ended with its faster move.
    if spec.npcs is not None and not ended:
        phase = engine.core.coming_phases(_pack(here()), 1)[0]
        for origin, target, eaten, kind in phase:
            if kind == 1:
                # Generation 36 (N1): a return, the piece that stood on its start at ply zero.
                board[target] = engine.initial_state().board[target]
                unmoved &= ~(1 << target)
                touched.add(target)
                flags["material"] |= eaten != 0
                continue
            board[target] = board[origin]
            board[origin] = 0
            unmoved &= ~(1 << origin) & ~(1 << target)
            touched.update((origin, target))
            flags["material"] |= eaten != 0

    # Schema 29 (Q983a, `pieces[].rest`): a round is a turn of each player, so each
    # player's rests count down, and each moved piece rests where it ended the round (its
    # origin if it stood, turned or was void), while that square holds its player's piece; a
    # slower move lost or unplayed rests nothing (schema 28). `tests/rest_oracle.py`.
    rest = state.rest
    if spec.rest:
        lists = lists_of(state)
        for i in (0, 1):
            end = None if gone[i] else plans[i]["target"] if arriving[i] else plans[i]["origin"]
            lists = after_turn(engine, lists, bytes(board), i, end)
        rest = as_rest(lists)

    # Step 9: the void run and the bars.
    run, bars, _ = _standoff_after(state, void, [plans[0]["origin"], plans[1]["origin"]], touched)

    # Bookkeeping.
    for i in (0, 1):
        if arriving[i] or standing[i]:
            if plans[i]["origin"] is not None:
                unmoved &= ~(1 << plans[i]["origin"])
            unmoved &= ~(1 << plans[i]["target"])
    rights = unmoved != state.unmoved
    if leaves:
        goal_left_by = goal_left(arriving)
    elif reach_held:
        goal_left_by = held_arrivals(arriving)
    progress = flags["material"] or flags["promoted"] or flags["one_way"] or flags["laid"]
    irreversible = progress or flags["kind"] or rights or flags["passed"]
    after = here().with_(
        halfmove=0 if progress else state.halfmove + 2,
        ply=state.ply + 2,
        void_run=run,
        bars=bars,
        rest=rest,
        goal_left_by=goal_left_by,
    )
    key = engine._compute_hash(after)
    repetition = (key,) if irreversible else (*state.repetition, key)
    return after.with_(key=key, repetition=repetition)
