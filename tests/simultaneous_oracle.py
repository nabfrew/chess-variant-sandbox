"""The declared Python oracle for a simultaneous round (schema 26, Q918a).

`docs/plan-simultaneous.md` §6.1: resolution is checked by this oracle alone. It resolves
one round from the compiled tables and the two move words, in the plan's order (§2, §4),
and returns the position the core's `make_round` must reach -- board, hands, colouring,
tally, rights, clocks, the initiative, the void run, the bars and (schema 27, Q942a) the
rested squares; schema 27's swap meeting is resolved here too, and `legal_for` filters a
player's moves by the rest. CLAUDE.md rule 3 allows
it: it is a declared test oracle. It asks the engine only questions the engine's own tests
answer elsewhere -- the moves a player has on a board (for `lands_if_empty`, §5.3), what the
wild pieces' phase does on a board (`coming_phases`) and the slow definition of the key
(`_compute_hash`) -- and never asks it to resolve anything.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import WILD, CompiledSpec
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.moves import (
    MOVE_DROP,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    decode_move,
)
from chess_variant_sandbox.engine.state import GameState

NO_PAIR = (-1, -1)


@cache
def _tables(engine: Engine) -> dict[str, Any]:
    return build_tables(engine.spec)


def _entering(spec: CompiledSpec, player: int, piece_type: int) -> int:
    facing = spec.facing_initial[piece_type] if spec.facing_initial else 0
    return spec.encode_piece(player, piece_type, facing)


def _arriving(spec: CompiledSpec, player: int, piece_type: int, square: int) -> int:
    if spec.entry_promotion is not None:
        becomes = spec.entry_promotion.at(player, piece_type, square)
        if becomes >= 0:
            return _entering(spec, player, becomes)
    return _entering(spec, player, piece_type)


def _meeting(spec: CompiledSpec, mover_type: int, victim: int, mover_owner: int) -> tuple[int, int, bool]:
    strike = spec.strike
    if strike is None:
        return -1, -1, False
    owner = spec.owner_of(victim)
    cell = mover_type * strike.n_types + spec.type_of(victim)
    if owner == mover_owner:
        return strike.own_onto_to[cell], strike.own_mover_to[cell], False
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
        if kind not in (MOVE_DROP, MOVE_ROTATE) and o == origin and t == target:
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


def legal_for(engine: Engine, state: GameState, player: int) -> list[int]:
    """Schema 27 (Q942a, `docs/plan-round-chase.md` §3 R1): `player`'s moves under the rest.

    The core's moves on the same board with nothing resting, less the rested piece's,
    unless that would leave none (the lift); in the core's order. Without `rest` the
    position's rest is `(-1, -1)` and this is the core's list.
    """
    unrested = engine.legal_moves_for(state.with_(rest=NO_PAIR), player)
    rested = state.rest[player]
    if rested < 0:
        return unrested

    def from_rested(move: int) -> bool:
        origin, _, kind, _ = decode_move(move)
        return kind != MOVE_DROP and origin == rested

    kept = [move for move in unrested if not from_rested(move)]
    return kept if kept else unrested


def resolve_round(engine: Engine, state: GameState, first: int, second: int) -> GameState:
    """The position after one round, resolved by the plan's rule (§2, §4)."""
    spec = engine.spec
    sim = spec.simultaneous
    assert sim is not None
    tables = _tables(engine)
    n_types = spec.n_types
    board = bytearray(state.board)
    hands = [list(state.hands[0]), list(state.hands[1])]
    colour = bytearray(state.colour)
    points = list(state.points)
    unmoved = state.unmoved
    initiative = state.initiative
    moves = (first, second)

    plans = []
    for player, move in enumerate(moves):
        origin, target, kind, promotion = decode_move(move)
        if kind == MOVE_DROP:
            plans.append({
                "m": move, "kind": kind, "origin": None, "target": target, "promotion": promotion,
                "mover": _arriving(spec, player, promotion, target), "type": promotion,
            })
        else:
            plans.append({
                "m": move, "kind": kind, "origin": origin, "target": target, "promotion": promotion,
                "mover": board[origin], "type": spec.type_of(board[origin]),
            })

    def stays_put(plan: dict[str, Any], player: int) -> bool:
        strike = spec.strike
        if strike is None or strike.position != "stay" or plan["origin"] is None:
            return False
        victim = board[plan["target"]]
        if victim == 0 or plan["origin"] == plan["target"]:
            return False
        becomes, mover_becomes, flips = _meeting(spec, plan["type"], victim, player)
        return becomes != -1 or mover_becomes >= 0 or flips

    raw = [plan["origin"] != plan["target"] for plan in plans]
    stays = [stays_put(plans[0], 0), stays_put(plans[1], 1)]
    standing = [False, False]
    for i in (0, 1):
        j = 1 - i
        there = board[plans[i]["target"]]
        own_target = stays[i] and there != 0 and spec.owner_of(there) == i
        vacated = raw[j] and not stays[j] and plans[j]["origin"] == plans[i]["target"]
        standing[i] = plans[i]["kind"] == MOVE_ROTATE or (stays[i] and (own_target or not vacated))
    moving = [raw[i] and not standing[i] for i in (0, 1)]
    # Schema 27 (Q942a, R2): a swap meets as a collision, so neither move is void.
    swapped = (
        sim.swap_collides and moving[0] and moving[1]
        and plans[0]["origin"] is not None
        and plans[1]["origin"] == plans[0]["target"]
        and plans[0]["origin"] == plans[1]["target"]
    )
    round_board = bytes(board)
    void = [False, False]
    for i in (0, 1):
        j = 1 - i
        void[i] = (
            not swapped and moving[i] and moving[j] and plans[i]["origin"] is not None
            and plans[j]["origin"] == plans[i]["target"]
            and not lands_if_empty(engine, state, round_board, i, plans[i]["m"])
        )
    arriving = [moving[i] and not void[i] for i in (0, 1)]

    touched: set[int] = set()
    material = promoted = one_way = irreversible_kind = passed = False
    for i in (0, 1):
        if arriving[i] or standing[i]:
            kind, promotion = plans[i]["kind"], plans[i]["promotion"]
            one_way |= plans[i]["type"] in spec.one_way_types
            plain_portal = kind == MOVE_PORTAL and promotion == 0
            irreversible_kind |= kind not in (MOVE_NORMAL, MOVE_ROTATE) and not plain_portal
            promoted |= kind == MOVE_PROMOTION or (kind == MOVE_PORTAL and promotion != 0)

    def worth(value: int) -> float:
        return -1.0 if value == 0 else spec.piece_values[spec.type_of(value)]

    def contact_in_place(player: int, plan: dict[str, Any]) -> bool:
        origin, target = plan["origin"], plan["target"]
        victim, mover = board[target], board[origin]
        becomes, mover_becomes, flips = _meeting(spec, plan["type"], victim, player)
        as_type = mover_becomes if mover_becomes >= 0 else spec.type_of(mover)
        if flips:
            attacker = _arriving(spec, 1 - player, as_type, origin)
        elif mover_becomes >= 0:
            attacker = _arriving(spec, player, as_type, origin)
        else:
            attacker = mover
        survivor = _arriving(spec, spec.owner_of(victim), becomes, target) if becomes >= 0 else 0
        board[origin] = attacker
        board[target] = survivor
        touched.update((origin, target))
        return worth(attacker) != worth(mover) or worth(survivor) != worth(victim)

    # Step 2: departures.
    for i in (0, 1):
        if not arriving[i]:
            continue
        if plans[i]["origin"] is not None:
            board[plans[i]["origin"]] = 0
            touched.add(plans[i]["origin"])
        else:
            drops = spec.drops
            assert drops is not None
            slot = drops.hand_slot(plans[i]["type"])
            if not (drops.unlimited and drops.unlimited[slot]):
                hands[i][slot] -= 1

    # Step 3: each move's own consequences.
    arrival = [0, 0]
    for i in (0, 1):
        plan = plans[i]
        if plan["kind"] == MOVE_ROTATE:
            board[plan["origin"]] = spec.encode_piece(i, plan["type"], plan["promotion"])
            touched.add(plan["origin"])
            continue
        if standing[i]:
            there = board[plan["target"]]
            if there != 0 and spec.owner_of(there) == i:
                material |= contact_in_place(i, plan)
            continue
        if not arriving[i]:
            continue
        value = plan["mover"]
        if plan["kind"] == MOVE_PROMOTION:
            value = _entering(spec, i, plan["promotion"])
        elif plan["kind"] == MOVE_PORTAL and plan["promotion"] != 0:
            value = _entering(spec, i, plan["promotion"] - 1)
        occupant = board[plan["target"]]
        if occupant != 0 and spec.owner_of(occupant) == i:
            _, mover_becomes, _ = _meeting(spec, plan["type"], occupant, i)
            if mover_becomes >= 0:
                value = _arriving(spec, i, mover_becomes, plan["target"])
            board[plan["target"]] = 0
            touched.add(plan["target"])
            material = True
        arrival[i] = value

    # Step 4: the meetings.
    takings: list[dict[str, Any]] = []

    def taking(captor: int, victim: int, square: int, strikes: bool) -> dict[str, Any]:
        return {
            "captor": captor, "captor_type": plans[captor]["type"], "victim": victim,
            "square": square, "origin": plans[captor]["origin"],
            "strikes": strikes and spec.strike is not None, "gone": False,
        }

    def meet() -> tuple[int | None, list[bool]]:
        """Who wins a meeting of the two arrivals (`beats`, then the tie rule)."""
        nonlocal initiative, passed
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
            passed = True
            return holder, beats
        return None, beats

    collided = arriving[0] and arriving[1] and plans[0]["target"] == plans[1]["target"]
    if swapped:
        # Both destinations were emptied by the departures; each is taken where it would
        # have landed, on the other's origin.
        material = True
        touched.update((plans[0]["target"], plans[1]["target"]))
        winner, beats = meet()
        if winner is not None:
            square = plans[winner]["target"]
            board[square] = arrival[winner]
            loser = taking(winner, arrival[1 - winner], square, False)
            loser["strikes"] = sim.collision_strikes
            takings.append(loser)
        elif beats[0] and beats[1]:
            for w in (0, 1):
                each = taking(w, arrival[1 - w], plans[w]["target"], False)
                each["strikes"] = sim.collision_strikes
                each["gone"] = True
                takings.append(each)
    elif collided:
        square = plans[0]["target"]
        touched.add(square)
        material = True
        winner, beats = meet()
        there = board[square]
        if winner is not None:
            if there != 0:
                takings.append(taking(winner, there, square, True))
            board[square] = arrival[winner]
            loser = taking(winner, arrival[1 - winner], square, False)
            loser["strikes"] = sim.collision_strikes
            takings.append(loser)
        else:
            board[square] = 0
            if beats[0] and beats[1]:
                for w in (0, 1):
                    each = taking(w, arrival[1 - w], square, False)
                    each["strikes"] = sim.collision_strikes
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
                material = True
                takings.append(taking(i, occupant, square, True))
            board[square] = arrival[i]
    for i in (0, 1):
        plan = plans[i]
        if not standing[i] or plan["kind"] == MOVE_ROTATE:
            continue
        victim = board[plan["target"]]
        if victim == 0 or spec.owner_of(victim) == i or board[plan["origin"]] != plan["mover"]:
            continue
        material |= contact_in_place(i, plan)

    # Step 5: what the captures cause.
    for taken in takings:
        victim, captor = taken["victim"], taken["captor"]
        drops = spec.drops
        if drops is not None and drops.from_captures and spec.owner_of(victim) != captor:
            slot = spec.demotions[spec.type_of(victim)]
            if not (drops.unlimited and drops.unlimited[slot]):
                hands[captor][slot] += 1
        if spec.npcs is not None and spec.owner_of(victim) == WILD:
            points[captor] += tables["npc_bounty"][taken["square"]]
        if not taken["strikes"]:
            continue
        strike = spec.strike
        assert strike is not None
        becomes, mover_becomes, flips = _meeting(spec, taken["captor_type"], victim, captor)
        square = taken["square"]
        there = board[square]
        if not taken["gone"] and there != 0 and spec.owner_of(there) == captor and (
            mover_becomes >= 0 or flips
        ):
            as_type = mover_becomes if mover_becomes >= 0 else spec.type_of(there)
            board[square] = _arriving(spec, 1 - captor if flips else captor, as_type, square)
        if becomes < 0:
            continue
        owner = spec.owner_of(victim)
        place = None
        if strike.position == "home":
            home = strike.rebirth[(owner * n_types + becomes) * spec.n_squares + square]
            place = home if home >= 0 else None
        elif strike.position == "swap":
            place = taken["origin"]
        if place is not None and board[place] == 0:
            board[place] = _arriving(spec, owner, becomes, place)
            touched.add(place)
            if strike.position == "home":
                unmoved |= 1 << place
            else:
                unmoved &= ~(1 << place)

    # Step 6: explosions, all at once.
    explosion = spec.explosion
    if explosion is not None:
        centres: set[int] = set()
        blast: set[int] = set()
        for taken in takings:
            capturer, victim_type = taken["captor_type"], spec.type_of(taken["victim"])
            if (explosion.by_types is None or capturer in explosion.by_types) and (
                explosion.of_types is None or victim_type in explosion.of_types
            ):
                centres.add(taken["square"])
                blast.update(explosion.blast[taken["square"]])
        for square in range(spec.n_squares):
            value = board[square]
            if value == 0:
                continue
            if square in centres or (
                square in blast and spec.type_of(value) not in explosion.immune_types
            ):
                board[square] = 0
                touched.add(square)
                material = True

    # Step 7: claims.
    laid = False
    territory = spec.territory
    if territory is not None:
        for i in (0, 1):
            if plans[i]["kind"] == MOVE_ROTATE:
                square = plans[i]["origin"]
            elif arriving[i]:
                square = plans[i]["target"]
            else:
                continue
            value = board[square]
            if value == 0 or spec.owner_of(value) != i:
                continue
            want, before = i + 1, colour[square]
            if before != want and (territory.mode == "repaint" or before == 0):
                colour[square] = want
                laid |= territory.mode != "repaint"

    # Step 8: the wild pieces, as the engine's phase would walk them on this board.
    if spec.npcs is not None:
        here = state.with_(board=bytes(board), colour=bytes(colour), unmoved=unmoved)
        phase = engine.core.coming_phases(_pack(here), 1)[0]
        for origin, target, eaten in phase:
            board[target] = board[origin]
            board[origin] = 0
            unmoved &= ~(1 << origin) & ~(1 << target)
            touched.update((origin, target))
            material |= eaten != 0

    # Schema 27 (Q942a, R1): each moved piece rests where it ended the round (its origin if
    # it stood, turned or was void), while that square holds its player's piece.
    rest = state.rest
    if sim.rest:
        ends = []
        for i in (0, 1):
            end = plans[i]["target"] if arriving[i] else plans[i]["origin"]
            ok = end is not None and board[end] != 0 and spec.owner_of(board[end]) == i
            ends.append(end if ok else -1)
        rest = (ends[0], ends[1])

    # Step 9: the void run and the bars.
    run, bars, _ = _standoff_after(state, void, [plans[0]["origin"], plans[1]["origin"]], touched)

    # Bookkeeping.
    for i in (0, 1):
        if arriving[i] or standing[i]:
            if plans[i]["origin"] is not None:
                unmoved &= ~(1 << plans[i]["origin"])
            unmoved &= ~(1 << plans[i]["target"])
    rights = unmoved != state.unmoved
    goal_left_by = state.goal_left_by
    reach = spec.reach
    if reach is not None and reach.leaves:
        left = [False, False]
        for i in (0, 1):
            origin = plans[i]["origin"]
            if origin is None:
                continue
            value = board[plans[i]["target"]]
            left[i] = (
                arriving[i] and origin in reach.squares[i]
                and (reach.piece_type < 0 or plans[i]["type"] == reach.piece_type)
                and value != 0 and spec.owner_of(value) == i
            )
        goal_left_by = {(True, True): 2, (True, False): 0, (False, True): 1}.get(tuple(left), -1)
    progress = material or promoted or one_way or laid
    irreversible = progress or irreversible_kind or rights or passed
    after = state.with_(
        board=bytes(board),
        hands=(tuple(hands[0]), tuple(hands[1])),
        colour=bytes(colour),
        points=(points[0], points[1]),
        unmoved=unmoved,
        halfmove=0 if progress else state.halfmove + 2,
        ply=state.ply + 2,
        initiative=initiative,
        void_run=run,
        bars=bars,
        rest=rest,
        goal_left_by=goal_left_by,
    )
    key = engine._compute_hash(after)
    repetition = (key,) if irreversible else (*state.repetition, key)
    return after.with_(key=key, repetition=repetition)
