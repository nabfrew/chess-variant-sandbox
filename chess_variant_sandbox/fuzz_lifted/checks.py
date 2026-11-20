"""Rule-obedience checks for every pair of rules the validator lifted (`[[lifted]]`).

Each check states, from the spec alone, something the engine's play must obey when its rule is
present, and reads a `Record` (the engine's own positions and moves) to see that it did. It
runs on every case whose spec holds its keys, so it watches its rule beside whatever partner
the case put in, and it names the `[[lifted]]` rows it was written for (`pairs`): those are the
pairs the coverage report counts as *checked*.

**What a check may do** (CLAUDE.md rule 3): read the spec and the compiled tables, read the
positions and the move words the engine produced, and ask the engine questions about a position
(its effects, whether a rest was lifted, its verdict, its phase). It never generates a move or
works out an effect itself. So a check can see the engine break a stated rule; it cannot see a
rule the engine and the check both misread, and it says nothing about a move that was never
offered (an *under*-generation). The record (`docs/reference/fuzz-lifted-pairs.md`) lists what
each check does not prove.

Every check fails on a planted fault by test (`tests/test_fuzz_lifted.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.engine.compile import (
    CONTROL_DUCK,
    CONTROL_MAJORITY,
    CONTROL_NEUTRAL,
    CONTROL_PORTAL,
    CONTROL_ROUTE,
    MODE_CAPTURE,
    MODE_MOVE,
    SCREEN_ENEMY,
    SCREEN_OWN,
    SCREEN_TYPES,
    SIGHT_COMMAND,
    SIGHT_DISARM,
    SIGHT_FREEZE,
    SIGHT_SHIELD,
    WILD,
)
from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    MOVE_BUY,
    MOVE_CASTLE,
    MOVE_DISPLACE,
    MOVE_DROP,
    MOVE_DUCK,
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_FOREIGN,
    MOVE_HIRE,
    MOVE_KEEP,
    MOVE_NORMAL,
    MOVE_PASS,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_PUSH,
    MOVE_ROTATE,
    MOVE_SWEEP,
    move_from,
    move_gate,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.moves import move_gate as _move_gate
from chess_variant_sandbox.fuzz_lifted.keys import holds
from chess_variant_sandbox.fuzz_lifted.record import Record, Step
from chess_variant_sandbox.spec.capture import (
    CAPTURE_CAUSES,
    capture_as_old_keys,
    capture_list,
    needs_the_walk,
)

NO_SQUARE = 255
NO_WINDOW = -1  # `GameState.ep_target` with no window open

# A failure whose message starts with this names an engine fault already found and reported
# (`docs/reference/fuzz-lifted-pairs.md` §5, each with an xfail test): the run files it apart,
# so a soak still sees it recur and the gate still fails on anything new.
KNOWN_PREFIX = "[known: "
# R-L6c (2026-11-15) fixed the four the first soaks found (F1 to F4), and R-L6e (2026-11-15)
# R-L6b's F7 (a sweep's ray read as a gate); a recurrence of any of them now fails the run.
KNOWN_FAULTS: dict[str, str] = {}


class CheckFailure(AssertionError):
    """A check said no. The message names the step."""


# ---------------------------------------------------------------------------- helpers

class View:
    """The compiled spec's readings a check needs, once per record."""

    def __init__(self, record: Record) -> None:
        self.record = record
        # R-L4b stage 3: a capture list the old keys say is read as them (the normal form's
        # write-back), so every check written against the old keys reads either spelling.
        self.spec = capture_as_old_keys(record.spec) or record.spec
        self.engine = record.engine
        self.c = record.engine.spec
        c = self.c
        kinds = c.control.kind if c.control is not None else ()
        self.kind = lambda t: kinds[t] if t < len(kinds) else 0
        self.duck = c.duck.piece_type if c.duck is not None else -1
        self.wild_types = {t for _, t in c.npcs.starts} if c.npcs is not None else set()
        self.ids = {name: index for index, name in enumerate(c.piece_ids)}
        # R-L4b stage 3: the capture list (either spelling), and whether the core walks it (a
        # list the old keys cannot say, or a pair only the list plays): the checks written for
        # the old keys' order judge only where it does not.
        self.entries = capture_list(self.spec) or []
        self.walked = needs_the_walk(self.spec)
        # How many events of its rule the check judged: a check that ran on a record and
        # met nothing to judge (no castle played, no buy made) proved nothing there.
        self.judged = 0

    def owner(self, value: int) -> int:
        return self.c.owner_of(value)

    def type(self, value: int) -> int:
        return self.c.type_of(value)

    def nobody_kind(self, value: int) -> int:
        """The control kind of a byte in the third block, 0 for a player's piece."""
        if not value or self.owner(value) != WILD:
            return 0
        return self.kind(self.type(value))

    def is_wild_piece(self, value: int) -> bool:
        return self.nobody_kind(value) == CONTROL_ROUTE or (
            bool(value) and self.owner(value) == WILD and self.type(value) in self.wild_types
        )

    def is_capture(self, board: bytes, move: int, side: int) -> bool:
        """Whether a move word takes something, by Q122's reading (the board says what stands
        on its target; the word says its kind)."""
        kind = move_type(move)
        if kind in (MOVE_EN_PASSANT, MOVE_SWEEP):
            return True
        if kind in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL):
            target = board[move_to(move)]
            return bool(target) and self.owner(target) != side
        return False

    def ids_of(self, names: Iterable[Any]) -> set[int]:
        return {self.ids[str(n)] for n in names if str(n) in self.ids}

    def rules(self) -> dict[str, Any]:
        rules = self.spec.get("rules")
        return rules if isinstance(rules, dict) else {}


def _fail(step: Step | None, message: str) -> None:
    where = f"step {step.index}" if step is not None else "the opening position"
    raise CheckFailure(f"{where}: {message}")


def _alternating(record: Record) -> list[Step]:
    return [step for step in record.steps if not step.round]


@dataclass(frozen=True)
class Check:
    """One rule-obedience check: the keys it runs under, the rows it was written for."""

    name: str
    keys: tuple[str, ...]  # every one must be held for the check to run (any of `any_keys`)
    pairs: tuple[str, ...]
    run: Callable[[View], None]
    any_keys: tuple[str, ...] = ()
    unless: tuple[str, ...] = ()

    def applies(self, spec: dict[str, Any]) -> bool:
        if not all(holds(spec, key) for key in self.keys):
            return False
        if self.any_keys and not any(holds(spec, key) for key in self.any_keys):
            return False
        return not any(holds(spec, key) for key in self.unless)

    def __call__(self, record: Record) -> int:
        """Run the check on ``record``; how many events of its rule it judged."""
        view = View(record)
        self.run(view)
        return view.judged


# ------------------------------------------------------------------------- the duck

def duck_stands_alone_and_moves_last(v: View) -> None:
    """The duck is never taken, never moved but by a duck move, lands only on an empty square,
    and each duck move follows a piece move of the same player, the side changing only after
    it. Beside a long turn (L7) the turn is its piece moves and then one duck move."""
    duck = v.duck

    def ducks(board: bytes) -> list[int]:
        return [sq for sq, value in enumerate(board)
                if value and v.owner(value) == WILD and v.type(value) == duck]

    previous: Step | None = None
    for step in v.record.steps:
        before, after = ducks(step.before.board), ducks(step.after.board)
        if len(after) > 1:
            _fail(step, f"{len(after)} ducks stand")
        if before and not after:
            _fail(step, f"the duck on square {before[0]} left the board")
        move = step.moves[0]
        kind = move_type(move)
        if kind == MOVE_DUCK:
            v.judged += 1
            if move_promotion(move) != DUCK_PASS and step.before.board[move_to(move)]:
                _fail(step, f"the duck landed on occupied square {move_to(move)}")
            if previous is None or move_type(previous.moves[0]) == MOVE_DUCK:
                _fail(step, "a duck move that follows no piece move")
            elif previous.mover != step.mover:
                _fail(step, "the duck was moved by the player who did not just move a piece")
        elif before and after != before:
            _fail(step, f"a piece move carried the duck from {before} to {after}")
        if (previous is not None and kind != MOVE_DUCK
                and move_type(previous.moves[0]) != MOVE_DUCK and previous.mover != step.mover):
            _fail(step, "the side changed with no duck move after the piece move")
        previous = step


# ------------------------------------------------------------------------ long turns

def _turn_lengths(rules: dict[str, Any]) -> tuple[list[int], int]:
    turn = rules.get("turn") or {}
    if turn.get("parts"):
        return [], len(turn["parts"])
    return list(turn.get("ramp", []) or []), int(turn.get("moves", 1) or 1)


def a_turn_is_as_long_as_the_schedule(v: View) -> None:
    """Each player's turn is the declared number of moves (the ramp, then `moves`, or one per
    part), counted in piece moves; beside the duck (L7) one duck move closes each turn."""
    ramp, moves = _turn_lengths(v.rules())
    steps = v.record.steps
    if any(step.round for step in steps):
        return
    runs: list[list[Step]] = []
    for step in steps:
        if runs and runs[-1][0].mover == step.mover:
            runs[-1].append(step)
        else:
            runs.append([step])
    for index, run in enumerate(runs):
        expected = ramp[index] if index < len(ramp) else moves
        v.judged += 1
        pieces = [s for s in run if move_type(s.moves[0]) != MOVE_DUCK]
        complete = index < len(runs) - 1
        if len(pieces) > expected or (complete and len(pieces) != expected):
            _fail(run[0], f"turn {index} was {len(pieces)} piece moves, the schedule says {expected}")
        if v.duck >= 0 and complete:
            ducks = [s for s in run if move_type(s.moves[0]) == MOVE_DUCK]
            if len(ducks) != 1 or move_type(run[-1].moves[0]) != MOVE_DUCK:
                _fail(run[0], f"turn {index} ended with {len(ducks)} duck moves, not one last")


# ------------------------------------------------------------------------ en passant

def en_passant_only_on_the_turn_s_first_move(v: View) -> None:
    """In a turn of several moves the window is the opponent's first move only (L1): an en
    passant capture is a turn's first move, takes the opponent's promoting piece on the
    recorded victim square, and is never offered mid-turn."""
    promoting = {rule.piece_type for rule in v.c.promotions}
    previous: Step | None = None
    for step in _alternating(v.record):
        mid_turn = previous is not None and previous.mover == step.mover
        v.judged += int(mid_turn) + int(move_type(step.moves[0]) == MOVE_EN_PASSANT)
        if mid_turn and any(move_type(m) == MOVE_EN_PASSANT for m in step.offered):
            _fail(step, "en passant offered on a move after the turn's first")
        move = step.moves[0]
        if move_type(move) == MOVE_EN_PASSANT:
            victim = step.before.ep_victim
            value = step.before.board[victim] if victim != NO_SQUARE else 0
            if not value or v.owner(value) != 1 - step.mover or v.type(value) not in promoting:
                _fail(step, f"en passant took no opponent's promoting piece on square {victim}")
            capturer = step.before.board[move_from(move)]
            struck = bool(value) and bool(capturer) and _strike_reaches(v, v.type(capturer), value)
            # R-L6b's soak (seed 2130750): a strike placed "stay" with a chain leaves the pawn
            # taken in passing standing as its link, its owner's still.
            if not struck and v.owner(step.after.board[victim] or 0) == 1 - step.mover and step.after.board[victim]:
                _fail(step, f"the piece taken in passing still stands on square {victim}")
        previous = step


# ------------------------------------------------------------------------------ rest

def a_resting_piece_is_not_moved(v: View) -> None:
    """A piece with `rest` that moved (or was commanded, V7) is moved by neither player until
    its owner has played `rest` more turns, nor again in the same turn of several moves (L2),
    unless the engine says the rest was lifted (every move the player had was a resting
    piece's) or the piece was taken, moved or changed. A push (push.N5) ends the rest of
    every piece on its line and the pusher rests."""
    if any(step.round for step in v.record.steps) or v.c.strike is not None:
        return
    rest = v.c.rest
    entries: dict[int, tuple[int, int, int, int]] = {}  # square -> (byte, owner, left, turn)
    turn = 0
    for step in v.record.steps:
        board = step.before.board
        entries = {sq: e for sq, e in entries.items() if board[sq] == e[0]}
        move = step.moves[0]
        kind = move_type(move)
        origin, target = move_from(move), move_to(move)
        # A fold's own move moves the fold, never the piece standing in it (schema 22).
        moves_a_piece = kind not in (
            MOVE_DROP, MOVE_BUY, MOVE_KEEP, MOVE_PASS, MOVE_DUCK, MOVE_HIRE, MOVE_FOLD,
        )
        v.judged += len(entries)  # each rest in force is a constraint this step is held to
        if (moves_a_piece and origin in entries
                and not v.engine.rest_turn(step.before, step.mover).lifted):
            _fail(step, f"the piece on square {origin} moved while it rests")
        if kind == MOVE_PUSH:
            # push.N5 (Q1074b8): a pushed piece rests no more, so a piece of the same code
            # shoved onto a resting square does not inherit its rest; the pusher rests.
            for square, _ in v.c.push_line(board, move):
                entries.pop(square, None)
        # Where the moved piece now stands, and whose it is.
        if kind not in (MOVE_KEEP, MOVE_PASS, MOVE_DUCK, MOVE_HIRE):
            mover_owner = step.mover if kind in (MOVE_DROP, MOVE_BUY) else v.owner(board[origin] or 1)
            after = step.after.board
            landing = None
            if kind == MOVE_ROTATE:
                landing = origin
            elif after[target] and v.owner(after[target]) == mover_owner:
                landing = target
            if landing is not None and mover_owner in (0, 1):
                left = rest[v.type(after[landing])] if rest else 0
                if left:
                    entries[landing] = (after[landing], mover_owner, left, turn)
        if step.after.side_to_move != step.before.side_to_move:
            ended = step.before.side_to_move
            kept = {}
            for sq, (byte, owner, left, made) in entries.items():
                if owner == ended and made != turn:
                    left -= 1
                if left > 0:
                    kept[sq] = (byte, owner, left, made)
            entries = kept
            turn += 1


# --------------------------------------------------------------------- wild pieces

def _route_key(npcs: Any, square: int) -> int:
    from chess_variant_sandbox.fuzz import _route_key as key
    return key(npcs, square)


def wild_pieces_keep_their_routes(v: View) -> None:
    """`fuzz.every_npc_stands_on_its_own_route` over every position of the record: beside a
    strike or contacts (T6, T7: a wild type is immune), a blast (T8), a long turn (L3: a phase
    after every move), a sweep (W5), a hand (N17), nobody's pieces (O1) or a sight (V11); and on
    a line two pieces walk head-on (N4: never two on one square, never more than started); and
    beside a pouncer that comes back (N7: never more of its type than its entries start)."""
    from chess_variant_sandbox.fuzz import OracleFailure, every_npc_stands_on_its_own_route

    for step in v.record.steps:
        v.judged += 1
        try:
            every_npc_stands_on_its_own_route(v.engine, step.after)
        except OracleFailure as failure:
            _fail(step, str(failure))


def a_wild_piece_comes_back_on_its_first_free_start(v: View) -> None:
    """A wild piece that comes back stands on a start of a respawning entry of its own type,
    and no start before it was free on the board the phase left -- empty, or a player's
    non-royal piece the type may take. A pouncer's entry (N7, SR8g: *a pouncer respawns on its
    entry's first free start*) is one count over its mirrored starts too, read in the mover's
    reading order; any other entry's half in the order written. Not judged beside a sight (a
    shield holds a start, and which squares it shields is the engine's to say)."""
    npcs = v.c.npcs
    if npcs is None or not npcs.respawn or v.c.sight is not None:
        return
    type_at = dict(npcs.starts)
    pouncing = {t for t, kind in enumerate(v.c.pounce.kind) if kind} if v.c.pounce is not None else set()
    unit_of = {sq: unit for unit, (_, starts) in enumerate(npcs.respawn) for sq in starts}
    for step in _alternating(v.record):
        returns = [i for i, s in enumerate(step.phase) if s.kind == "return"]
        if not returns:
            continue
        board = bytearray(step.after.board)
        for i in reversed(returns):
            back = step.phase[i]
            square = back.target
            if square not in unit_of:
                _fail(step, f"a wild piece came back on square {square}, no respawning entry's start")
            starts = list(npcs.respawn[unit_of[square]][1])
            piece_type = type_at[starts[0]]
            if v.type(board[square]) != piece_type or not v.is_wild_piece(board[square]):
                _fail(step, f"what came back on square {square} is not its entry's wild type")
            board[square] = back.eaten
            if piece_type in pouncing and v.c.pounce is not None:
                reading = v.c.pounce.reading_rank[step.mover]
                starts.sort(key=lambda sq: reading[sq])
            v.judged += 1
            for earlier in starts[:starts.index(square)]:
                value = board[earlier]
                held = bool(value) and (
                    v.owner(value) == WILD or v.type(value) in v.c.royal_types
                    or not v.c.capture_allowed[piece_type][v.type(value)]
                )
                if not held:
                    _fail(step, f"a wild piece came back on square {square} while its entry's "
                                f"start {earlier}, earlier in its order, was free")


def a_royal_is_never_left_where_a_wild_piece_steps(v: View) -> None:
    """Beside a line (N5, SR8h: *its attack is the square ahead after the turn*), a move never
    leaves the mover's royal where the next phase, played on the position the move left, would
    step a wild piece -- under checkmate, so a move into a turning piece's way (ahead, or behind
    it where it would turn at a block) is not offered -- and under `forbid_check` never the
    opponent's either. The phase is the engine's own (`coming_phase`, the board's drawing), so
    the check holds the threat the legality filter reads to the walk the phase plays. Not judged
    beside the duck (a piece half runs no phase) or after a move that took the other royal
    (legal whatever it leaves)."""
    from chess_variant_sandbox.engine.coming import coming_phase

    c = v.c
    if c.npcs is None or c.duck is not None or not c.royal_types:
        return
    rules = v.rules()
    mate = rules.get("win", "checkmate") == "checkmate"
    forbid = rules.get("forbid_check") is True
    royal = set(c.royal_types)
    for step in _alternating(v.record):
        after = step.after
        if not any(v.is_wild_piece(value) for value in after.board):
            continue
        # A move that takes the other royal wins on the spot, whatever it leaves (the
        # engine's verdict is not asked: it would call a royal left in check an illegal
        # position, which is the fault this looks for).
        if not any(value and v.owner(value) == 1 - step.mover and v.type(value) in c.royal_types
                   for value in after.board):
            continue
        stepped = {s.target for s in coming_phase(v.engine, after) if s.kind != "return"}
        v.judged += 1
        for square in stepped:
            value = after.board[square]
            if not value or v.owner(value) == WILD or v.type(value) not in royal:
                continue
            owner = v.owner(value)
            if mate and owner == step.mover:
                _fail(step, f"the mover's royal on square {square} is left where a wild piece steps next")
            if forbid and owner != step.mover:
                _fail(step, f"the move left the other royal on square {square} where a wild piece "
                            "steps next, which forbid_check counts as giving check")


def a_safe_goal_is_not_where_a_wild_piece_steps(v: View) -> None:
    """Beside a line (N5, SR8h) under a safe arrival goal (no `removes`, `count` or `leave`), a
    game the engine ends as reached by a player has, of that player's pieces the goal admits on
    its goal squares, at least one the next phase does not step onto: a piece standing where a
    wild piece steps after its turn is threatened, so it reaches nothing. The phase is the
    engine's own (`coming_phase`); a player's attack is the engine's to read, so the check holds
    only the wild half."""
    from chess_variant_sandbox.engine.coming import coming_phase

    rule = v.c.reach
    if v.c.npcs is None or v.c.duck is not None or rule is None or not rule.safe:
        return
    if rule.leaves or rule.removes or rule.count != 1:
        return
    for step in _alternating(v.record):
        after = step.after
        verdict = v.engine.terminal(after)
        if verdict is None or verdict.reason != "reached" or verdict.winner not in (0, 1):
            continue
        if rule.result != "win":
            continue
        side = verdict.winner
        standing = [
            sq for sq in rule.squares[side]
            if after.board[sq] and v.owner(after.board[sq]) == side
            and (not rule.piece_types or v.type(after.board[sq]) in rule.piece_types)
        ]
        if not standing:
            continue  # reached by an arrival the record holds elsewhere (a held piece's)
        stepped = {s.target for s in coming_phase(v.engine, after) if s.kind != "return"}
        v.judged += 1
        if all(sq in stepped for sq in standing):
            _fail(step, f"player {side} reached a safe goal whose every piece on it, on "
                        f"{sorted(standing)}, a wild piece steps onto next")


def an_immune_wild_piece_falls_only_to_its_takers(v: View) -> None:
    """A wild piece whose entry lists `immune` is taken only by those types, by a capture or a
    sweep, whatever else the game holds (N17)."""
    immune: dict[int, set[int] | None] = {}
    for entry in v.rules().get("npcs", []) or []:
        piece = v.ids.get(str(entry.get("piece")))
        if piece is None:
            continue
        listed = v.ids_of(entry["immune"]) if entry.get("immune") else None
        if piece in immune and immune[piece] != listed:
            immune[piece] = None  # entries of one type disagree: not checked
        else:
            immune.setdefault(piece, listed)
    for step in _alternating(v.record):
        move = step.moves[0]
        board = step.before.board
        kind = move_type(move)
        if kind not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL, MOVE_SWEEP) or not board[move_from(move)]:
            continue
        mover = v.type(board[move_from(move)])
        squares = [move_to(move)]
        if kind == MOVE_SWEEP:
            squares += list(v.c.sweep_path(board, move, step.mover))
        for sq in squares:
            value = board[sq]
            if value and v.is_wild_piece(value):
                takers = immune.get(v.type(value))
                v.judged += bool(takers)
                if takers and mover not in takers:
                    _fail(step, f"a {v.c.piece_ids[mover]!r} took the wild piece on square {sq}, "
                                f"immune to all but {sorted(v.c.piece_ids[t] for t in takers)}")


# ---------------------------------------------------------- nobody's pieces and majority

def a_standing_piece_stands(v: View) -> None:
    """A piece nobody owns of control "stands" never moves: after any move or round, every
    square holding one held a nobody's piece before (it may be taken, struck in place or eaten,
    never moved) -- beside rounds (R18), wild pieces (O1), the duck (O2), a strike or contacts
    (O3, O4), portals (P10), a sight (V15) and a spy (O6)."""
    for step in v.record.steps:
        if any(move_type(m) == MOVE_DISPLACE for m in step.moves):
            continue  # a displacement shifts every jumped piece one back, a nobody's too (TS4)
        before = {sq for sq, value in enumerate(step.before.board)
                  if v.nobody_kind(value) in (CONTROL_NEUTRAL, CONTROL_MAJORITY)}
        # A held or shared piece that meets a link becomes what it says, nobody's still,
        # where its move took it (O4): the one way a standing type arrives somewhere new.
        before |= {move_to(m) for m in step.moves
                   if v.nobody_kind(step.before.board[move_from(m)]) == CONTROL_MAJORITY}
        # A strike placed "choose" or "home" puts a struck nobody's piece on the landing chosen
        # or its home, nobody's still (O3, R-L6b's soak): one standing piece may arrive there
        # when the move took one.
        placed = sum(
            1 for m in step.moves
            if _places_its_survivor(v) and v.nobody_kind(step.before.board[move_to(m)]) == CONTROL_NEUTRAL
        )
        v.judged += bool(before)
        for sq, value in enumerate(step.after.board):
            if v.nobody_kind(value) == CONTROL_NEUTRAL and sq not in before:
                if placed:
                    placed -= 1
                    continue
                _fail(step, f"a standing piece nobody owns appeared on square {sq}")


def _places_its_survivor(v: View) -> bool:
    """Whether a strike or contact table puts what it strikes somewhere other than where it
    stood (`choose`, `home`)."""
    return v.c.strike is not None and _strike_spelling(v).get("position") in ("choose", "home")


def no_piece_joins_a_side_from_nowhere(v: View) -> None:
    """A side's piece count rises only by its own drops and buys: a held piece moved by a
    player stays nobody's (L5, R20, O10, O11, O12), and a piece nobody owns never becomes a
    player's but by a hire. Not run where a strike or contact can change a piece's side."""
    if v.c.strike is not None:
        return
    for step in v.record.steps:
        if any(move_type(m) == MOVE_HIRE for m in step.moves):
            continue
        v.judged += 1
        counts = []
        for board in (step.before.board, step.after.board):
            tally = [0, 0]
            for value in board:
                if value and v.owner(value) in (0, 1):
                    tally[v.owner(value)] += 1
            counts.append(tally)
        for player in (0, 1):
            mine = [step.moves[player]] if step.round else (
                list(step.moves) if step.mover == player else [])
            entered = sum(int(move_type(m) in (MOVE_DROP, MOVE_BUY)) + int(bool(move_gate(m)))
                          for m in mine)
            if counts[1][player] > counts[0][player] + entered:
                _fail(step, f"player {player} went from {counts[0][player]} pieces to "
                            f"{counts[1][player]} with {entered} entering")


def a_held_piece_moves_in_its_holders_frame(v: View) -> None:
    """A held piece moves as its holder's own piece of its type pointing the same way would
    (SR8i, `docs/plan-schema-next.md` §2.11 *the mover's reading*: O13, O14, O17, O18, O19;
    O9's placed facing): every held move the engine offered or played lands on a square the
    compiled tables give that type from its square in the **holder's** frame -- the holder's
    player and the facing of the holder's that points where nobody's byte points, read off
    `facing_headings`, never the compiler's own `held_frames` -- by an entry whose first-move
    condition the piece's own mark meets, and a hop only over a screen of the owner its word
    names, read for the holder (`own` the holder's piece, `enemy` the other player's, a
    nobody's piece neither); a turn
    leaves it nobody's at the facing its word names, and a step leaves it nobody's at the
    facing it had. A transit or a go-through (a portal game's tagged words) is not judged."""
    c = v.c
    if c.neutrals is None or not c.neutrals.has_majority:
        return
    n = c.n_facings
    headings = c.facing_headings()

    def frame(holder: int, facing: int) -> int:
        if n == 1:
            return holder
        return holder * n + headings[holder].index(headings[0][facing])

    def screens(holder: int, code: int, occupant: int) -> bool:
        # O19: `own` the holder's pieces, `enemy` the other player's; nobody's neither.
        if code == SCREEN_OWN:
            return v.owner(occupant) == holder
        if code == SCREEN_ENEMY:
            return v.owner(occupant) == 1 - holder
        return True

    def reach(state: Any, origin: int, value: int, holder: int) -> set[int]:
        piece_type = v.type(value)
        # A type that does not face has no arrow: the holder's forward (SR8j's fix).
        at = frame(holder, c.facing_of(value)) if c.facing_axes[piece_type] > 1 else holder * n
        unmoved = bool(state.unmoved >> origin & 1)
        squares: set[int] = set()
        # A sweep over an empty line is the slide's word (`MOVE_NORMAL`, R-L6d's trial: 135
        # held sweepers' plain moves read as unreachable before its table was read here).
        for entries in (c.leaps[at][piece_type][origin], c.slides[at][piece_type][origin],
                        c.sweeps[at][piece_type][origin]):
            for entry in entries:
                ray, modes, first_move = entry[0], entry[1], entry[2]
                if first_move and not unmoved:
                    continue
                squares |= {sq for sq, mode in zip(ray, modes, strict=True) if mode}
        for hop in c.hops[at][piece_type][origin]:
            path, path_modes, needs_first, code = hop[0], hop[1], hop[2], hop[4]
            if needs_first and not unmoved:
                continue
            screen = next((i for i, sq in enumerate(path) if state.board[sq]), None)
            if screen is None or not (code >= SCREEN_TYPES or screens(
                holder, code, state.board[path[screen]]
            )):
                continue
            squares |= {sq for sq, mode in zip(path[screen + 1:], path_modes[screen + 1:], strict=True)
                        if mode}
        return squares

    for step in v.record.steps:
        played = (list(enumerate(step.moves)) if step.round
                  else [(step.mover, m) for m in step.moves])
        judged = list(played)
        if not step.round:
            judged += [(step.mover, m) for m in step.offered if m not in step.moves]
        for holder, move in judged:
            origin = move_from(move)
            value = step.before.board[origin]
            if not move & MOVE_FOREIGN or v.nobody_kind(value) != CONTROL_MAJORITY:
                continue
            kind = move_type(move)
            is_played = (holder, move) in played
            if kind == MOVE_ROTATE:
                v.judged += 1
                after = step.after.board[origin]
                if is_played and not step.round and (
                    v.owner(after) != WILD or v.type(after) != v.type(value)
                    or c.facing_of(after) != move_promotion(move)
                ):
                    _fail(step, f"a held turn {move:#x} left square {origin} holding {after}")
                continue
            if kind != MOVE_NORMAL or (c.portals is not None and move >> 25 & 0x1F):
                continue
            v.judged += 1
            target = move_to(move)
            if target not in reach(step.before, origin, value, holder):
                _fail(step, f"a held move {move:#x} by player {holder} lands on square {target}, "
                            "which its holder's own piece of the type could not reach")
            if (is_played and not step.round and c.strike is None
                    and not _rules_of(v).get("contacts") and not _rules_of(v).get("explode_on_capture")):
                after = step.after.board[target]
                if v.owner(after) != WILD or c.facing_of(after) != c.facing_of(value):
                    _fail(step, f"a held move {move:#x} left square {target} holding {after}, "
                                "not the piece nobody owns as it pointed")


def a_held_piece_promotes_as_its_holder_would(v: View) -> None:
    """A held piece promotes in its **holder's** zone, to a target its holder's promotion
    offers, never a royal, a wild type or the duck, and stays nobody's as the new type (SR8j,
    `docs/plan-schema-next.md` §2.11 O15, *the mover's reading*). Every held promotion the
    engine offered or played ends in the holder's zone (or, under `on_leaving`, starts there)
    and names such a target; a played one leaves the square holding a nobody's piece of that
    type (where no strike, contact or blast acts on the move); and under a mandatory
    promotion no held arrival in the holder's zone is offered unpromoted. Read off the
    compiled promotion rules' zones and targets (`PromotionRule.squares`, `.targets`), never
    the core's `neutral_held_promotion`. A transit or a go-through is not judged."""
    c = v.c
    if c.neutrals is None or not c.neutrals.has_majority or not c.promotions:
        return
    rules = {rule.piece_type: rule for rule in c.promotions}
    never = set(c.royal_types) | v.wild_types | ({v.duck} if v.duck >= 0 else set())
    acts_on_captures = c.strike is not None or c.explosion is not None or bool(_rules_of(v).get("contacts"))
    for step in _alternating(v.record):
        holder = step.mover
        for move in set(step.offered) | set(step.moves):
            origin = move_from(move)
            value = step.before.board[origin]
            if not move & MOVE_FOREIGN or v.nobody_kind(value) != CONTROL_MAJORITY:
                continue
            rule = rules.get(v.type(value))
            kind = move_type(move)
            if rule is None or kind not in (MOVE_NORMAL, MOVE_PROMOTION):
                continue
            if c.portals is not None and move >> 25 & 0x1F:
                continue
            target = move_to(move)
            zone = rule.squares[holder]
            in_zone = target in zone or (rule.on_leaving and origin in zone)
            v.judged += 1
            if kind == MOVE_NORMAL:
                if in_zone and rule.mandatory:
                    _fail(step, f"a held move {move:#x} by player {holder} entered its zone on "
                                f"square {target} without promoting")
                continue
            promoted = move_promotion(move)
            if not in_zone:
                _fail(step, f"a held promotion {move:#x} by player {holder} outside its zone")
            if promoted in never or promoted not in rule.targets[holder]:
                _fail(step, f"a held promotion {move:#x} by player {holder} into type {promoted}, "
                            "which its holder's promotion does not offer a held piece")
            if move in step.moves and not acts_on_captures and target not in step.phase_targets:
                after = step.after.board[target]
                if v.owner(after) != WILD or v.type(after) != promoted:
                    _fail(step, f"a held promotion {move:#x} left square {target} holding {after}, "
                                "not a piece nobody owns of the type it named")


def a_held_sweep_removes_only_what_its_holder_could_capture(v: View) -> None:
    """A held sweep removes what its holder could capture with it (SR8k,
    `docs/plan-schema-next.md` §2.11 O16, *the mover's reading*): every held sweep the engine
    offered or played passes over only the other player's pieces, none royal, none shielded,
    and lands on an empty square or one of those; a played one (where no strike, contact or
    blast acts on it) leaves every swept square empty and its landing holding a piece nobody
    owns. The path is the compiled sweep table's ray in the holder's frame
    (`CompiledSpec.sweep_path` given the holder), the shields `Engine.effects`'."""
    c = v.c
    if c.neutrals is None or not c.neutrals.has_majority or not c.has_sweep:
        return
    acts = c.strike is not None or c.explosion is not None or bool(_rules_of(v).get("contacts"))
    royals = set(c.royal_types)
    for step in _alternating(v.record):
        holder = step.mover
        board = step.before.board
        bits: Any = None
        for move in set(step.offered) | set(step.moves):
            origin = move_from(move)
            if (
                move_type(move) != MOVE_SWEEP or not move & MOVE_FOREIGN
                or v.nobody_kind(board[origin]) != CONTROL_MAJORITY
            ):
                continue
            v.judged += 1
            path = c.sweep_path(board, move, holder)
            target = move_to(move)
            if not path:
                _fail(step, f"a held sweep {move:#x} whose path the holder's frame cannot read")
            if bits is None:
                bits = v.engine.effects(step.before).bits if c.sight is not None else ()
            for sq in [sq for sq in path if board[sq]] + ([target] if board[target] else []):
                value = board[sq]
                if v.owner(value) != 1 - holder or v.type(value) in royals:
                    _fail(step, f"a held sweep {move:#x} by player {holder} removed square {sq}'s "
                                f"piece {value}, which its holder could not capture")
                if bits and bits[sq] & SIGHT_SHIELD:
                    _fail(step, f"a held sweep {move:#x} passed a shielded piece on square {sq}")
            if move in step.moves and not acts and target not in step.phase_targets:
                after = step.after.board
                if any(after[sq] for sq in path if board[sq] and sq not in step.phase_targets):
                    _fail(step, f"a held sweep {move:#x} left a swept piece standing")
                if v.owner(after[target]) != WILD:
                    _fail(step, f"a held sweep {move:#x} left square {target} holding "
                                f"{after[target]}, not a piece nobody owns")


def _rules_of(v: View) -> dict[str, Any]:
    rules = v.spec.get("rules")
    return rules if isinstance(rules, dict) else {}


# ------------------------------------------------------------------- purse and bounty

def a_purse_stays_between_nothing_and_its_cap(v: View) -> None:
    """A purse is never negative and never past 99 (L6, R19, O22, P9, O27)."""
    for step in v.record.steps:
        v.judged += 1
        for player, coins in enumerate(step.after.purse):
            if not 0 <= coins <= 99:
                _fail(step, f"player {player}'s purse holds {coins}")


def a_buy_pays_its_price_onto_its_region(v: View) -> None:
    """A buy lands on an empty square of its region (never a portal end, P9), and spends its
    type's price; in a round (R19) the price is spent whether or not the piece survives."""
    purse = v.c.purse
    if purse is None:
        return
    rules = v.rules()
    bounties = any(v.c.bounty) or v.c.npcs is not None
    if rules.get("income") or (rules.get("purse") or {}).get("upkeep"):
        return
    for step in v.record.steps:
        players = (0, 1) if step.round else (step.mover,)
        for player in players:
            move = step.moves[player] if step.round else step.moves[0]
            if move_type(move) != MOVE_BUY:
                continue
            v.judged += 1
            bought, target = move_promotion(move), move_to(move)
            if bought not in purse.market:
                _fail(step, f"a {v.c.piece_ids[bought]!r} was bought; the market sells none")
            slot = purse.market.index(bought)
            if step.before.board[target]:
                _fail(step, f"a piece was bought onto occupied square {target}")
            if target not in purse.squares[player][slot]:
                _fail(step, f"a {v.c.piece_ids[bought]!r} was bought onto square {target}, "
                            "outside its region")
            if not bounties:
                spent = step.before.purse[player] - step.after.purse[player]
                if spent != purse.price[slot]:
                    _fail(step, f"a buy spent {spent} coins; the price is {purse.price[slot]}")


def _bounty_on(v: View, square: int, value: int) -> float:
    if v.is_wild_piece(value):
        return float(v.c.npcs.bounty[square]) if v.c.npcs is not None else 0.0
    bounty = v.c.bounty
    return float(bounty[v.type(value)]) if bounty else 0.0


def a_capture_pays_the_bounty_of_what_it_removed(v: View) -> None:
    """The mover's points rise by exactly the bounties of the pieces not its own that its move
    removed -- a capture, each swept piece (W5), each piece a blast destroyed (O22, T8), the
    piece a push throws off (push.purse, Q1074b8: *a pushed-off bounty fills the purse*) --
    and the other side's not at all. Read off the board: what stood before, less what stands after,
    less what the wild phase ate (which pays nobody)."""
    rules = v.rules()
    if v.c.strike is not None or (rules.get("purse") or {}).get("bounty_to") == "loser":
        return
    if any(e.get("do") == "pay" and (e.get("to") == "loser" or set(e["on"]) != set(CAPTURE_CAUSES))
           for e in v.entries):
        return  # a pay entry that reads only some causes pays only for those
    for entry in rules.get("npcs", []) or []:
        if entry.get("respawn") or entry.get("pounce") or entry.get("split"):
            return
    if (rules.get("reach") or {}).get("removes"):
        return
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) in (MOVE_HIRE,):
            continue
        mover = step.mover

        def worth(board: bytes, mover: int = mover) -> float:
            return sum(_bounty_on(v, sq, value) for sq, value in enumerate(board)
                       if value and v.owner(value) != mover)

        eaten = sum(_bounty_on(v, s.target, s.eaten) for s in step.phase
                    if s.eaten and v.owner(s.eaten) != mover)
        removed = worth(step.before.board) - worth(step.after.board) - eaten
        paid = step.after.points[mover] - step.before.points[mover]
        other = step.after.points[1 - mover] - step.before.points[1 - mover]
        v.judged += bool(removed or paid)
        if abs(paid - removed) > 1e-6 or abs(other) > 1e-6:
            _fail(step, f"the move removed {removed:g} of bounty and paid the mover {paid:g} "
                        f"(the other side {other:g})")


# ------------------------------------------------------------------------- rounds

def _castling_squares(v: View, board: bytes, move: int, player: int) -> tuple[int, int, int] | None:
    """(partner's square, partner's castled square, king's origin) for a castle, or None
    where the spec's castling is not the plain two-step kind this reads."""
    rule = v.c.castling
    if rule is None or rule.king_files is not None or rule.king_steps < 2:
        return None
    origin, target = move_from(move), move_to(move)
    width = v.c.width
    rank, file, to_file = origin // width, origin % width, target % width
    step = 1 if to_file > file else -1
    partner = None
    f = file + step
    while 0 <= f < width:
        sq = v.c.square_index(f, rank)
        value = board[sq]
        if value and v.owner(value) == player and v.type(value) == rule.rook_type:
            partner = sq
            break
        f += step
    if partner is None:
        return None
    if rule.rook_files is not None:
        dest_file = rule.rook_files[1 if step > 0 else 0]
    else:
        dest_file = to_file - step
    return partner, v.c.square_index(dest_file, rank), origin


def a_castle_moves_both_its_pieces(v: View) -> None:
    """A castle puts its castling piece on its target and its partner beside it on the side it
    came from (Z2: the castling piece need not be royal). In a round (R6) the partner moves
    after the meetings, onto its castled square if that is empty, else stays; both rights are
    spent either way."""
    rule = v.c.castling
    if rule is None:
        return
    for step in v.record.steps:
        players = (0, 1) if step.round else (step.mover,)
        for player in players:
            move = step.moves[player] if step.round else step.moves[0]
            if move_type(move) != MOVE_CASTLE:
                continue
            squares = _castling_squares(v, step.before.board, move, player)
            if squares is None:
                continue
            partner, castled, origin = squares
            v.judged += 1
            rook = step.before.board[partner]
            after = step.after.board
            if step.round:
                other = step.moves[1 - player]
                if move_to(other) in (partner, castled) or step.phase_targets & {partner, castled}:
                    continue
                if after[castled] != rook and after[partner] != rook:
                    _fail(step, f"the castling partner from square {partner} is neither on "
                                f"{castled} nor where it stood")
                if after[partner] == rook and not step.before.board[castled] and not after[castled]:
                    _fail(step, f"the partner stayed on {partner} with {castled} empty")
                if (step.after.unmoved >> partner) & 1 or (step.after.unmoved >> origin) & 1:
                    _fail(step, "a castling right survived the castle")
            else:
                if step.phase:
                    continue
                king = after[move_to(move)]
                if not king or v.type(king) != rule.king_type or v.owner(king) != player:
                    _fail(step, f"the castling piece is not on square {move_to(move)}")
                if after[castled] != rook:
                    _fail(step, f"the castling partner is not on square {castled}")


def _strike_spelling(v: View) -> dict[str, Any]:
    """The strike as its old key's object, read off the capture list (either spelling)."""
    return _rule_of(v, "strike")


def _rule_of(v: View, kind: str) -> dict[str, Any]:
    """A kind's rule (its first entry's keys) in the capture list, ``{}`` if none."""
    for entry in v.entries:
        if entry.get("do") == kind:
            return {k: val for k, val in entry.items() if k not in ("do", "on", "path_by")}
    return {}


def _strike_reaches(v: View, mover_type: int, victim: int) -> bool:
    """Whether a capture by ``mover_type`` of ``victim`` strikes, read off the spelling: the
    mover in `by` (if any), the victim not immune (listed, royal, wild, the duck)."""
    strike = _strike_spelling(v)
    by = v.ids_of(strike.get("by", []) or []) if strike.get("by") else None
    if by is not None and mover_type not in by:
        return False
    victim_type = v.type(victim)
    if strike.get("effect", "capture") != "capture" and strike.get("position") != "home":
        # A chain strike reaches only a piece with a link (SR4j: a meeting it does not reach
        # is an ordinary capture, the capturer advancing).
        linked = v.ids_of(link.get("from") for link in strike.get("chain", []) or [])
        if victim_type not in linked:
            return False
    immune = v.ids_of(strike.get("immune", []) or [])
    return not (
        victim_type in immune or victim_type in v.c.royal_types
        or victim_type in v.wild_types or victim_type == v.duck or v.nobody_kind(victim) in (
            CONTROL_ROUTE, CONTROL_DUCK, CONTROL_PORTAL)
    )


def _explodes(v: View, mover_type: int, victim_type: int) -> bool:
    blast = v.c.explosion
    if blast is None:
        return False
    if blast.by_types is not None and mover_type not in blast.by_types:
        return False
    return blast.of_types is None or victim_type in blast.of_types


def a_shot_leaves_its_shooter_where_it_stood(v: View) -> None:
    """Under a strike placed "stay", a capture the strike reaches leaves the capturer on its
    own square (beside a hand T2, wild pieces T7, a portal's exit P3, nobody's pieces O3, a
    command V5, a spy Y4, a list of shooters T10), and one it does not reach (an immune
    victim) is an ordinary capture. In a round (R11) a shooter stays unless the other move
    shot it."""
    if _strike_spelling(v).get("position") != "stay":
        return
    for step in v.record.steps:
        players = (0, 1) if step.round else (step.mover,)
        for player in players:
            move = step.moves[player] if step.round else step.moves[0]
            kind = move_type(move)
            if kind not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL):
                continue
            board = step.before.board
            origin, target = move_from(move), move_to(move)
            shooter, victim = board[origin], board[target]
            if not shooter or not victim or v.owner(victim) == v.owner(shooter):
                continue
            if v.owner(victim) == player and move & MOVE_FOREIGN:
                continue
            if _explodes(v, v.type(shooter), v.type(victim)):
                continue
            if step.phase_targets & {origin, target}:
                continue
            after = step.after.board[origin]
            v.judged += 1
            if _strike_reaches(v, v.type(shooter), victim):
                if step.round:
                    other = step.moves[1 - player]
                    if move_to(other) == origin:
                        continue
                    if v.c.portals is not None and board[move_from(other)] and \
                            v.type(board[move_from(other)]) in v.c.portals.end_types and \
                            v.owner(board[move_from(other)]) == WILD:
                        # R-L6b's soak (seed 3178451): a portal end moved first in the round, so
                        # the transit is read through where it now stands and may take nothing.
                        continue
                if not after:
                    _fail(step, f"the shooter left square {origin} on a shot at {target}")
            elif not step.round and after == shooter:
                _fail(step, f"a capture the strike does not reach left its capturer on {origin}")


def a_struck_piece_becomes_its_link(v: View) -> None:
    """Under a strike placed "stay" with a chain, a struck piece becomes its chain link where it
    stands, still its owner's (a nobody's piece stays nobody's, O3; a link into a spy is the
    owner's spy, Y4), or is removed where it has none."""
    strike = _strike_spelling(v)
    if strike.get("position") != "stay" or strike.get("effect", "capture") == "capture":
        return
    if v.walked:
        return  # the list's order decides first who has the piece: `a_taken_piece_is_in_one_place`
    chain = {v.ids[str(link["from"])]: v.ids[str(link["to"])]
             for link in strike.get("chain", []) or [] if str(link.get("from")) in v.ids}
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL):
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        shooter, victim = board[origin], board[target]
        if not shooter or not victim or v.owner(victim) == v.owner(shooter):
            continue
        if not _strike_reaches(v, v.type(shooter), victim) or _explodes(v, v.type(shooter), v.type(victim)):
            continue
        if step.phase_targets & {target}:
            continue
        if step.after.banked != step.before.banked:
            continue  # R-L6b's soak: a link left on its goal was banked off it (as the meeting check)
        after = step.after.board[target]
        v.judged += 1
        becomes = chain.get(v.type(victim))
        allowed = {becomes}
        entry = v.c.entry_promotion
        owner = v.owner(victim)
        if becomes is not None and entry is not None and owner in (0, 1) and entry.at(owner, becomes, target) >= 0:
            # R-L6b's soak: a link that leaves a promoting piece standing in its zone promotes
            # under `promote_on_entry` (RULES.md: *left standing there by a strike*), to the
            # first target, or under `only_lost` the first lost or none.
            allowed |= {entry.at(owner, becomes, target)}
            allowed |= {t for rule in v.c.promotions if rule.piece_type == becomes
                        for t in rule.targets[owner]}
        if becomes is None:
            if after:
                _fail(step, f"a struck piece with no link still stands on square {target}")
        elif not after or v.type(after) not in allowed or v.owner(after) != v.owner(victim):
            _fail(step, f"the piece struck on square {target} did not become its "
                        f"{v.c.piece_ids[becomes]!r}, its owner's")


def a_switcher_changes_sides(v: View) -> None:
    """A capture by a type `switches_sides` lists makes it the other player's piece where the
    capture leaves it (on the square it took, or on its own under a shot), read apart from
    `by` (T10); a spy that switches is the new side's spy (Y5). A held or commanded capturer
    never switches."""
    strike = _strike_spelling(v)
    switchers = v.ids_of(strike.get("switches_sides", []) or [])
    if not switchers:
        return
    stays = strike.get("position") == "stay"
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL) or move & MOVE_FOREIGN:
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        mover, victim = board[origin], board[target]
        if not mover or not victim or v.owner(mover) != step.mover or v.owner(victim) == step.mover:
            continue
        if v.type(mover) not in switchers or _explodes(v, v.type(mover), v.type(victim)):
            continue
        reached = _strike_reaches(v, v.type(mover), victim)
        if not reached and v.type(victim) in set(v.c.royal_types) | v.wild_types:
            continue
        if step.phase_targets & {origin, target}:
            continue
        if step.after.banked != step.before.banked:
            continue  # R-L6b's soak (seed 160964): the switched piece stood on its new goal, banked
        lands = origin if (stays and reached) else target
        v.judged += 1
        after = step.after.board[lands]
        if not after or v.owner(after) != 1 - step.mover:
            _fail(step, f"the switcher's capture left square {lands} not the other player's")


# ------------------------------------------------------------------------------ spies

def the_phase_follows_who_holds_a_spy(v: View) -> None:
    """The spy phase is read off the board both ways: both players with a spy on the board
    alternate; one with a spy is a reply or an announced round; neither, blind rounds. A spy
    nobody owns (O6), in a hand (Y7, Y8), or bought (O27) counts for nobody until it stands as
    a player's own spy type; one gained by a promotion, a chain, a switch or a link (Y3-Y6)
    counts at once."""
    spy = v.c.spy
    if spy is None:
        return
    for index, state in enumerate(v.record.positions()):
        holding = [
            any(value and v.owner(value) == player and v.type(value) in spy.types[player]
                for value in state.board)
            for player in (0, 1)
        ]
        name = v.engine.phase(state).name
        v.judged += 1
        expected = ("alternating",) if all(holding) else ("blind",) if not any(holding) else (
            "reply", "announced")
        if name not in expected:
            step = v.record.steps[index - 1] if index else None
            _fail(step, f"the phase is {name!r} with spies held {holding}")


# ------------------------------------------------------------------------------ sweeps

def a_sweep_clears_only_what_it_may(v: View) -> None:
    """A sweep never passes the duck (W9), a portal end (W8) or a shielded piece (W10), never
    passes a square twice or its own (W6); beside no strike, contacts or blast every piece on its
    path is gone after it (beside a strike or contacts on `landing`, likewise; on `path` each
    swept square keeps only its owner's piece, W1, W2); a commanded sweep (V8) removes none of
    the commander's pieces and no royal; and under a capture hand every swept piece not the
    mover's goes to the mover's hand (W4)."""
    if v.walked:
        return  # under the walk: `a_sweep_follows_the_list`
    rules = v.rules()
    drops = v.c.drops
    blast = v.c.explosion is not None
    struck = v.c.strike is not None
    path_effects = any(
        m.get("effects") == "path" for p in v.spec.get("pieces", []) for m in p.get("moves", [])
    )
    unlimited = drops is not None and drops.unlimited and any(drops.unlimited)
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) != MOVE_SWEEP:
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        path = v.c.sweep_path(board, move, step.mover)
        v.judged += 1
        if not path:
            _fail(step, "a sweep word whose path the tables cannot read")
        if origin in path or len(set(path)) != len(path):
            _fail(step, f"the sweep's path {path} passes a square twice or its own")
        bits = v.engine.effects(step.before).bits if v.c.sight is not None else ()
        # SR8k (O16): a held sweep is judged by its own check, below; this one reads a
        # commanded sweep's (V8).
        foreign = bool(move & MOVE_FOREIGN) and v.nobody_kind(board[origin]) != CONTROL_MAJORITY
        swept = [sq for sq in path if board[sq]]
        for sq in swept:
            value = board[sq]
            kind = v.nobody_kind(value)
            if kind in (CONTROL_DUCK, CONTROL_PORTAL) or v.type(value) == v.duck:
                _fail(step, f"the sweep passed the duck or a portal end on square {sq}")
            if bits and bits[sq] & SIGHT_SHIELD:
                _fail(step, f"the sweep passed a shielded piece on square {sq}")
            if foreign and (v.owner(value) == step.mover or v.type(value) in v.c.royal_types):
                _fail(step, f"a commanded sweep removed the commander's or a royal piece on {sq}")
        if foreign and swept and not any(
            s.get("captures_own") for p in v.spec.get("pieces", []) for s in p.get("sight", []) or []
        ):
            _fail(step, "a commanded sweep without captures_own removed pieces")
        if blast:
            continue
        after = step.after.board
        # R-L6e's fuzz hour (seeds 5116012, 5132971): under a strike on `landing`, the landing's
        # victim is struck after the path is cleared, so circe (`position: "home"`) may put it
        # back on its home where the sweep just emptied a square: one swept square may hold a
        # piece of the victim's owner (`Rd1xd4 (sweeps d2) (reborn d2)`).
        reborn = board[target] if struck and not path_effects else 0
        for sq in swept:
            if sq in step.phase_targets:
                continue
            if struck and path_effects:
                # R-L4b3's soak (seed 105284): placed "home", a struck piece is reborn on its home,
                # which may be another swept square, so a swept square may then hold another's.
                if after[sq] and v.owner(after[sq]) != v.owner(board[sq]) and (
                        _strike_spelling(v).get("position") != "home"):
                    _fail(step, f"a swept square {sq} holds a piece of another owner")
            elif after[sq] and reborn and v.owner(after[sq]) == v.owner(reborn):
                reborn = 0
            elif after[sq]:
                _fail(step, f"a swept piece still stands on square {sq}")
        if drops is not None and drops.from_captures and not struck and not unlimited:
            gained = sum(step.after.hands[step.mover]) - sum(step.before.hands[step.mover])
            taken = [sq for sq in swept if v.owner(board[sq]) != step.mover]
            if board[target] and v.owner(board[target]) != step.mover:
                taken.append(target)
            if gained != len(taken) and rules.get("drops", {}).get("from_captures", True):
                _fail(step, f"the sweep took {len(taken)} pieces not the mover's and the hand "
                            f"gained {gained}")


# -------------------------------------------------------------------------- blasts

def a_capture_that_explodes_takes_the_capturer(v: View) -> None:
    """A capture that explodes destroys its capturer, whatever it is -- held (O12), commanded,
    beside a strike (T1) or contacts (T4) or portals (P5) -- and, beside no shield, strike
    placement or sweep, every non-immune piece on the blast's squares (the duck, K3, and a
    portal end are immune; a wild piece is not, T8). Under a capture hand only the piece taken
    reaches it (H6)."""
    blast = v.c.explosion
    if blast is None:
        return
    if v.walked and not _reads(v, "blast", "landing"):
        return  # a list's blast that reads no landing capture sets none off
    shields = v.c.sight is not None and v.c.sight.has_shield
    drops = v.c.drops
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL):
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        capturer, victim = board[origin], board[target]
        if not capturer or not victim or v.owner(victim) == v.owner(capturer):
            continue
        if not _explodes(v, v.type(capturer), v.type(victim)):
            continue
        if step.phase_targets & {origin}:
            continue
        v.judged += 1
        after = step.after.board
        # R-L4b (gating.blast): a gated piece of the capturer's own type may stand on its origin.
        gated = bool(v.c.drops is not None and v.c.drops.gates is not None and _move_gate(move))
        if (after[origin] == capturer and not gated) or (
                after[target] and v.owner(after[target]) == v.owner(capturer) and after[target] == capturer):
            _fail(step, f"the capturer from square {origin} survived its own blast at {target}")
        if v.walked:
            continue  # what the list places after the blast: `a_spared_piece_stands_after_its_blast`
        if not shields and not blast.strikes:
            for sq in blast.blast[target]:
                value = board[sq]
                if (value and sq != origin and v.type(value) not in blast.immune_types
                        and after[sq] and sq not in step.phase_targets):
                    _fail(step, f"a piece on square {sq} survived the blast at {target}")
        if drops is not None and drops.from_captures and v.c.strike is None and v.type(victim) not in v.c.royal_types:
            slot_unlimited = bool(drops.unlimited) and any(drops.unlimited)
            gained = sum(step.after.hands[step.mover]) - sum(step.before.hands[step.mover])
            if not slot_unlimited and gained != 1:
                _fail(step, f"an exploding capture put {gained} pieces in the mover's hand")


# ------------------------------------------------------------------------- contacts

def a_meeting_follows_its_link(v: View) -> None:
    """A capture that meets a contact link makes both pieces what the link says (advance: the
    mover on the target as `mover_to`; stay: the mover in place, the target as `onto_to` or
    gone), its owner's still (a held mover stays nobody's, O4); one with no link, or of an
    immune target (a wild type, T6), is an ordinary capture; an own link moves onto one's own
    piece (T9); a commanded capture of the owner's own piece reads no link (V6). At a portal's
    exit (P4) the exit is the capture square; on a sweep (W2) the landing's meeting."""
    contacts = v.rules().get("contacts")
    if not isinstance(contacts, dict) or v.walked:
        return  # under the walk: `two_tables_each_read_their_own_meetings`
    position = contacts.get("position", "advance")
    if position not in ("advance", "stay"):
        return
    immune = v.ids_of(contacts.get("immune", []) or []) | set(v.c.royal_types) | v.wild_types
    links = {}
    for link in contacts.get("links", []) or []:
        key = (v.ids.get(str(link.get("mover"))), v.ids.get(str(link.get("onto"))), bool(link.get("own")))
        links[key] = (v.ids.get(str(link.get("mover_to"))) if link.get("mover_to") else None,
                      v.ids.get(str(link.get("onto_to"))) if link.get("onto_to") else None)
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) not in (MOVE_NORMAL, MOVE_PORTAL):
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        mover, victim = board[origin], board[target]
        if not mover or not victim or (move_type(move) == MOVE_PORTAL and move_promotion(move)):
            continue
        if _explodes(v, v.type(mover), v.type(victim)) or step.phase_targets & {origin, target}:
            continue
        if step.after.banked != step.before.banked:
            continue  # U2's fuzz: a piece the meeting left on its goal was banked off it
        own = v.owner(victim) == v.owner(mover)
        v.judged += 1
        if own and move & MOVE_FOREIGN:
            link = None  # V6: an ordinary removal
        elif v.type(victim) in immune or v.nobody_kind(victim) in (CONTROL_DUCK, CONTROL_PORTAL):
            link = None
        else:
            link = links.get((v.type(mover), v.type(victim), own))
            if own and link is None:
                _fail(step, f"a move onto its own piece on square {target} with no own link")
        after = step.after.board
        mover_to, onto_to = link if link is not None else (None, None)
        becomes = v.type(mover) if mover_to is None else mover_to
        lands = origin if (link is not None and position == "stay") else target
        stands = after[lands]
        if not stands or v.owner(stands) != v.owner(mover) or v.type(stands) != becomes:
            _fail(step, f"the mover from {origin} is not a {v.c.piece_ids[becomes]!r} of its "
                        f"owner on square {lands}")
        if link is not None and position == "stay":
            left = after[target]
            if onto_to is None and left:
                _fail(step, f"the target on square {target} stands with no onto_to")
            if onto_to is not None and (not left or v.type(left) != onto_to or v.owner(left) != v.owner(victim)):
                _fail(step, f"the target on square {target} did not become its onto_to")


def must_capture_offers_only_captures(v: View) -> None:
    """Under `must_capture`, where the engine offers a capture it offers nothing else; a move
    onto one's own piece by an own link is no capture (T9)."""
    if v.c.push is not None:
        return
    for step in _alternating(v.record):
        offered = step.offered
        board = step.before.board
        captures = [m for m in offered if v.is_capture(board, m, step.mover)]
        v.judged += bool(captures)
        if captures and len(captures) != len(offered):
            others = [m for m in offered if m not in captures]
            _fail(step, f"{len(captures)} captures offered beside {len(others)} other moves "
                        f"(first {others[0]:#x})")


# ------------------------------------------------------------------------ promotion

def a_promotion_takes_only_a_lost_target(v: View) -> None:
    """Under `only_lost` a piece promotes only into a target its side has fewer of on the board
    than it started with (beside a capture hand H1 or circe H2, the board is counted); entering
    its zone under `promote_on_entry` (H3) it becomes the first such target, counted before the
    move, or stays itself if none is."""
    rules = {rule.piece_type: rule for rule in v.c.promotions if rule.only_lost}
    if not rules:
        return
    entry = v.c.entry_promotion

    def count(board: bytes, player: int, piece_type: int) -> int:
        return sum(1 for value in board
                   if value and v.owner(value) == player and v.type(value) == piece_type)

    for step in _alternating(v.record):
        move = step.moves[0]
        kind = move_type(move)
        board, after = step.before.board, step.after.board
        player = step.mover
        origin, target = move_from(move), move_to(move)
        if kind == MOVE_DROP:
            dropped = move_promotion(move)
            rule = rules.get(dropped)
            if rule is None or entry is None or entry.at(player, dropped, target) < 0:
                continue
            if target in step.phase_targets or not after[target]:
                continue
            v.judged += 1
            lost = [t for t in rule.targets[player] if count(board, player, t) < rule.opening[player][t]]
            expected = lost[0] if lost else dropped
            if v.type(after[target]) != expected:
                _fail(step, f"a {v.c.piece_ids[dropped]!r} entering on square {target} became "
                            f"{v.c.piece_ids[v.type(after[target])]!r}, not "
                            f"{v.c.piece_ids[expected]!r}")
            continue
        if move & MOVE_FOREIGN or not board[origin] or kind in (MOVE_CASTLE, MOVE_ROTATE):
            continue
        rule = rules.get(v.type(board[origin]))
        if rule is None or v.owner(board[origin]) != player:
            continue
        if kind != MOVE_PROMOTION:
            continue  # a link's or a chain's change of type is no promotion
        became = move_promotion(move)
        v.judged += 1
        if became != rule.piece_type and count(board, player, became) >= rule.opening[player][became]:
            _fail(step, f"promoted into {v.c.piece_ids[became]!r}, which player {player} "
                        "has not lost")


def a_promoting_sweeper_promotes(v: View) -> None:
    """A promoting sweeper that lands in its zone under a mandatory rule promotes there, after
    clearing its line (W7); a commanded sweeper never promotes."""
    rules = {rule.piece_type: rule for rule in v.c.promotions
             if rule.mandatory and not rule.only_lost}
    if not rules or v.c.explosion is not None or v.c.strike is not None:
        return
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) != MOVE_SWEEP:
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        piece = board[origin]
        rule = rules.get(v.type(piece)) if piece else None
        if rule is None or target in step.phase_targets:
            continue
        if move & MOVE_FOREIGN and v.owner(piece) == WILD:
            continue  # a held sweeper promotes as its holder's would (O15, O16): the held-promotion check's
        lands = step.after.board[target]
        v.judged += 1
        if move & MOVE_FOREIGN:
            if lands and v.type(lands) != v.type(piece):
                _fail(step, f"a commanded sweeper promoted on square {target}")
            continue
        if target in rule.squares[step.mover] and (not lands or v.type(lands) not in rule.targets[step.mover]):
            _fail(step, f"a sweeper landed in its zone on square {target} and did not promote")


# ------------------------------------------------------------------------- portals

def portal_ends_are_never_taken(v: View) -> None:
    """Every momentum portal end stands on the board exactly once in every position, and the
    folds stay as many: no capture, blast (P5), strike (P3), contact (P4), sweep (W8), buy (P9),
    the duck (K2) or round (R9) removes or doubles one; a moved end stands where it went."""
    portals = v.c.portals
    if portals is None:
        return
    kinds = portals.kinds or ("",) * len(portals.end_types)
    momentum = [t for t, kind in zip(portals.end_types, kinds, strict=True) if kind != "fold"]
    folds = len(v.record.positions()[0].folds)
    for index, state in enumerate(v.record.positions()):
        step = v.record.steps[index - 1] if index else None
        v.judged += 1
        for end in momentum:
            standing = sum(1 for value in state.board
                           if value and v.owner(value) == WILD and v.type(value) == end)
            if standing != 1:
                _fail(step, f"portal end {v.c.piece_ids[end] if end < len(v.c.piece_ids) else end} "
                            f"stands {standing} times")
        if len(state.folds) != folds:
            _fail(step, f"{len(state.folds)} folds where the game began with {folds}")
    for step in v.record.steps:
        moves = step.moves
        ends = [m for m in moves if step.before.board[move_from(m)]
                and v.nobody_kind(step.before.board[move_from(m)]) == CONTROL_PORTAL
                and move_type(m) == MOVE_NORMAL]
        if len(ends) == 1 and not any(move_type(m) == MOVE_FOLD for m in moves):
            move = ends[0]
            if step.round and move_to(moves[0]) == move_to(moves[1]):
                continue
            if step.after.board[move_to(move)] != step.before.board[move_from(move)]:
                _fail(step, f"a moved portal end is not on square {move_to(move)}")


def a_piece_stays_in_its_region(v: View) -> None:
    """A player's piece of a type with a `region` stands only in it, however it arrived: an
    exit outside it is no move (P7)."""
    from chess_variant_sandbox.engine.compile import _region_squares

    width, height = v.c.width, v.c.height
    orientation = str(v.spec.get("setup", {}).get("orientation", "mirror"))
    regions: dict[tuple[int, int], frozenset[int]] = {}
    for piece in v.spec.get("pieces", []):
        if piece.get("region") and str(piece["id"]) in v.ids:
            for player in (0, 1):
                squares = _region_squares(piece["region"], player, width, height, orientation)
                if squares is not None:
                    regions[(player, v.ids[str(piece["id"])])] = squares
    positions = v.record.positions()
    for index, state in enumerate(positions):
        step = v.record.steps[index - 1] if index else None
        for sq, value in enumerate(state.board):
            if not value or v.owner(value) not in (0, 1):
                continue
            if step is not None and step.before.board[sq] == value:
                continue  # it stood here before this step: judged where it arrived
            region = regions.get((v.owner(value), v.type(value)))
            if region is None:
                continue
            v.judged += 1
            if sq in region:
                continue
            _fail(step, f"a {v.c.piece_ids[v.type(value)]!r} stands on square {sq}, "
                        "outside its region")


# ---------------------------------------------------------------- territory and goals

def _arrival(v: View, step: Step) -> tuple[int, int] | None:
    """(square, claimant) for a move whose piece now stands where the word says it went: the
    owner of a player's piece (a commanded one too), the mover for a held one."""
    move = step.moves[0]
    kind = move_type(move)
    if kind not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL, MOVE_SWEEP):
        return None
    board = step.before.board
    origin, target = move_from(move), move_to(move)
    piece = board[origin]
    if not piece or v.nobody_kind(piece) in (CONTROL_PORTAL, CONTROL_DUCK, CONTROL_ROUTE):
        return None
    after = step.after.board[target]
    if not after or v.owner(after) != v.owner(piece) or target in step.phase_targets:
        return None
    if step.after.board[origin] == piece:
        return None  # it did not leave: a shot
    claimant = step.mover if v.owner(piece) == WILD else v.owner(piece)
    return target, claimant


def a_move_claims_where_it_lands(v: View) -> None:
    """Under territory a move claims the square its piece ends on -- through a portal the exit
    (P6) -- for the piece's owner (a commanded piece's owner, V4) or, for a held piece, the
    player who moved it (O10); a permanent claim is never taken back."""
    rule = v.c.territory
    if rule is None:
        return
    for step in _alternating(v.record):
        arrival = _arrival(v, step)
        if arrival is None or not step.after.colour:
            continue
        square, claimant = arrival
        v.judged += 1
        before, after = step.before.colour[square], step.after.colour[square]
        if rule.mode == "permanent" and before:
            if after != before:
                _fail(step, f"a permanent claim on square {square} changed hands")
        elif after != claimant + 1:
            _fail(step, f"square {square} is claimed by {after - 1}, not by player {claimant}")


def an_arrival_on_a_goal_decides(v: View) -> None:
    """Under an arrival goal (no `safe`, `removes`, `count` or `leave`), a piece its rule admits
    arriving on its claimant's goal ends the game, for the claimant where the ending is the
    goal's (a commanded piece's owner, V3; a held piece's mover, O11)."""
    rule = v.c.reach
    if rule is None or rule.safe or rule.leaves or rule.removes or rule.count != 1:
        return
    for step in _alternating(v.record):
        arrival = _arrival(v, step)
        if arrival is None:
            continue
        square, claimant = arrival
        piece = step.after.board[square]
        if rule.piece_types and v.type(piece) not in rule.piece_types:
            continue
        if square not in rule.squares[claimant]:
            continue
        v.judged += 1
        verdict = v.engine.terminal(step.after)
        if verdict is None:
            _fail(step, f"a piece arrived on player {claimant}'s goal {square} and the game went on")
            return
        if verdict.reason == "reached" and rule.result == "win" and verdict.winner not in (claimant, None):
            other = 1 - claimant
            if not any(step.after.board[sq] and v.owner(step.after.board[sq]) == other
                       for sq in rule.squares[other]):
                _fail(step, f"player {claimant}'s arrival on {square} won for player {other}")


# ----------------------------------------------------------------------------- sight

def a_frozen_piece_makes_no_move(v: View) -> None:
    """No move the engine offers is a frozen piece's, nor a capture by a disarmed one, and a
    commanded move moves an enemy piece the command binds, never a royal, onto an empty square
    or (captures_own) onto its owner's own non-royal unshielded piece, and never castles, takes
    in passing or promotes -- beside portals (V1), a blast (V2), goals and territory (V3, V4),
    rest (V7), a sweep (V8, W10), nobody's sight (V15) and a strike, contacts or hand (V5, V6,
    V9)."""
    if v.c.sight is None:
        return
    captures_own = any(
        s.get("captures_own") for p in v.spec.get("pieces", []) for s in p.get("sight", []) or []
    )
    for step in _alternating(v.record):
        board = step.before.board
        bits = v.engine.effects(step.before).bits
        v.judged += sum(1 for b in bits if b)  # each piece under an effect is a constraint held
        for move in step.offered:
            kind = move_type(move)
            if kind in (MOVE_DROP, MOVE_BUY, MOVE_KEEP, MOVE_PASS, MOVE_DUCK, MOVE_HIRE, MOVE_FOLD):
                continue
            origin, target = move_from(move), move_to(move)
            piece = board[origin]
            if not piece or v.nobody_kind(piece) in (CONTROL_PORTAL, CONTROL_DUCK):
                continue
            if bits[origin] & SIGHT_FREEZE:
                _fail(step, f"a move {move:#x} of the frozen piece on square {origin} was offered")
            if bits[origin] & SIGHT_DISARM and v.is_capture(board, move, step.mover):
                _fail(step, f"a capture {move:#x} by the disarmed piece on square {origin} was offered")
            if move & MOVE_FOREIGN and v.owner(piece) == 1 - step.mover:
                if not bits[origin] & SIGHT_COMMAND:
                    _fail(step, f"the piece on square {origin} was commanded with no command on it")
                if v.type(piece) in v.c.royal_types:
                    _fail(step, f"a royal on square {origin} was commanded")
                if kind in (MOVE_CASTLE, MOVE_EN_PASSANT, MOVE_PROMOTION, MOVE_PUSH):
                    _fail(step, f"a commanded move {move:#x} castles, takes in passing or promotes")
                victim = board[target] if kind in (MOVE_NORMAL, MOVE_PORTAL) and target != origin else 0
                if victim:
                    if not captures_own or v.owner(victim) != v.owner(piece):
                        _fail(step, f"a commanded move {move:#x} takes a piece it may not")
                    elif v.type(victim) in v.c.royal_types or bits[target] & SIGHT_SHIELD:
                        _fail(step, f"a commanded move {move:#x} takes a royal or shielded piece")


# ------------------------------------------------------------------------ extinction

def running_out_of_one_s_own_kind_ends_it(v: View) -> None:
    """Under `extinction.per_type: "own"` beside a hand (H11), a side short of a kind it started
    with, counting what it holds in hand, has lost the game: the engine ends it there; and an
    extinction ending names such a side."""
    rule = v.c.extinction
    if rule is None or not rule.own:
        return
    for index, state in enumerate(v.record.positions()):
        step = v.record.steps[index - 1] if index else None
        short = []
        for player in (0, 1):
            for piece_type in rule.side_types[player]:
                standing = sum(1 for value in state.board
                               if value and v.owner(value) == player and v.type(value) == piece_type)
                held = state.hands[player][piece_type] if state.hands and state.hands[player] else 0
                if standing + held < rule.min_count:
                    short.append(player)
        v.judged += 1
        verdict = v.engine.terminal(state)
        if short and verdict is None:
            _fail(step, f"player(s) {sorted(set(short))} ran out of a kind and the game went on")
        if verdict is not None and verdict.reason == "extinction" and not short:
            _fail(step, "an extinction ending with nobody short of a kind")


# ------------------------------------------------------- U2: nine pairs the core played

def income_fills_the_purse_alone(v: View) -> None:
    """Income (income.I1) is paid at the start of a player's turn into the purse, never into
    the points: on every step that hands the turn over, the player now to move gains in the
    purse exactly ``per_piece`` for each of their pieces on their region on the board the step
    left (capped at 99), and nothing in the points."""
    rule = v.c.income
    rules = v.rules()
    if rule is None or v.c.purse is None or (rules.get("purse") or {}).get("bounty_to") == "loser":
        return
    for step in _alternating(v.record):
        nxt = step.after.side_to_move
        if nxt == step.before.side_to_move:
            continue  # mid-turn: no income lands
        v.judged += 1
        standing = sum(
            1 for square in rule.squares[nxt]
            if step.after.board[square] and v.owner(step.after.board[square]) == nxt
        )
        want = min(99, step.before.purse[nxt] + rule.per_piece * standing)
        if step.after.purse[nxt] != want:
            _fail(step, f"player {nxt}'s purse went {step.before.purse[nxt]} -> "
                        f"{step.after.purse[nxt]}, and their income makes it {want}")
        if step.after.points[nxt] != step.before.points[nxt]:
            _fail(step, f"player {nxt}'s points moved at the start of their turn "
                        f"({step.before.points[nxt]:g} -> {step.after.points[nxt]:g})")


def a_banked_piece_is_counted_off_the_board(v: View) -> None:
    """Beside a bank (goals.extinction, goals.bare_royal) both endings count the board, which a
    banked piece has left; a bank that reaches the count wins first. So: an extinction or
    bare-royal verdict never stands where a side has banked its count, and where no side has,
    a side short on the board (or bare there) has ended the game."""
    reach = v.c.reach
    if reach is None or not reach.removes:
        return
    extinction = v.c.extinction
    plain = extinction is not None and not extinction.per_type
    for index, state in enumerate(v.record.positions()):
        step = v.record.steps[index - 1] if index else None
        verdict = v.engine.terminal(state)
        counted = any(state.banked[p] >= reach.count for p in (0, 1))
        v.judged += 1
        if verdict is not None and verdict.reason in ("extinction", "bare_royal") and counted:
            _fail(step, f"a {verdict.reason} verdict where a bank reached the count "
                        f"({state.banked}), which wins first")
        if counted or verdict is not None:
            continue
        for player in (0, 1):
            mine = [v.type(value) for value in state.board if value and v.owner(value) == player]
            if plain and extinction is not None:
                listed = [t for t in mine if extinction.any_piece or t in extinction.piece_types]
                if len(listed) < extinction.min_count:
                    _fail(step, f"player {player} has {len(listed)} of the listed kinds on the "
                                "board and the game went on")
            if v.c.bare_royal and mine and all(t in v.c.royal_types for t in mine):
                _fail(step, f"player {player} stands bare and the game went on")


def only_a_player_s_unmoved_piece_gates(v: View) -> None:
    """A gate (gating.control, gating.duck) rides only on the mover's own piece leaving a gate
    square of theirs on its first move: never a held or shared piece's move, never a duck
    half; and a gated move leaves the gated piece, the mover's, on the square it named."""
    drops = v.c.drops
    if drops is None or drops.gates is None:
        return
    for step in _alternating(v.record):
        mover = step.mover
        board = step.before.board
        for move in step.offered:
            gate = move_gate(move)
            if not gate:
                continue
            v.judged += 1
            kind = move_type(move)
            origin = move_from(move)
            if move & MOVE_FOREIGN:
                _fail(step, f"a foreign move {move:#x} gates")
            if kind == MOVE_DUCK:
                _fail(step, f"a duck move {move:#x} gates")
            if kind == MOVE_CASTLE:
                continue  # the partner's square is the castle's own reading
            piece = board[origin]
            if not piece or v.owner(piece) != mover:
                _fail(step, f"a gate on {move:#x} from a square that is not the mover's piece")
            if origin not in drops.gates[mover] or not (step.before.unmoved >> origin) & 1:
                _fail(step, f"a gate on {move:#x} from a square that is no unmoved gate")
        move = step.moves[0]
        gate = move_gate(move)
        if move_from(move) in step.phase_targets:
            continue  # R-L6b's soak (seed 1198608): a pouncer took the gated piece after the move
        if move_type(move) == MOVE_SWEEP:
            # R-L6e (F7): a sweep never gates and never lands on its origin (a ray ends on a
            # square it has visited), and gating is refused beside a strike, so its origin is
            # empty after it. Read off the board, not off `move_gate`, which reads a sweep's tag
            # as no gate: a core that read its ray as a gate again would leave a piece here.
            v.judged += 1
            if step.after.board[move_from(move)]:
                _fail(step, f"a sweep {move:#x} left a piece on its origin")
            continue
        if gate and move_type(move) != MOVE_CASTLE:
            target = step.before.board[move_to(move)]
            if target and v.c.explosion is not None and _explodes(
                    v, v.type(step.before.board[move_from(move)]), v.type(target)):
                continue  # R-L4b (gating.blast): `a_gated_piece_enters_before_the_blast`
            left = step.after.board[move_from(move)]
            if not left or v.owner(left) != mover or v.type(left) != gate - 1:
                _fail(step, f"the gated {v.c.piece_ids[gate - 1]!r} is not on the square left")


def a_displaced_piece_claims_nothing(v: View) -> None:
    """Under territory a displacing hop (run.territory) claims its landing for the mover and
    nothing else: the squares its jumped pieces are shifted onto keep their colour. A push
    alike (push.G2, `plan-push.md` §3.10: *territory is claimed by a move's landing*): the
    pusher claims its target, a pushed piece nothing."""
    rule = v.c.territory
    if rule is None:
        return
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) not in (MOVE_DISPLACE, MOVE_PUSH) or not step.after.colour:
            continue
        if move_type(move) == MOVE_PUSH and move & MOVE_FOREIGN:
            continue  # a held pusher claims for its mover; its own reading is O10's
        v.judged += 1
        target = move_to(move)
        for square, (before, after) in enumerate(zip(step.before.colour, step.after.colour, strict=True)):
            if square != target and before != after:
                _fail(step, f"a displacement changed square {square}'s colour")
        kept = rule.mode == "permanent" and step.before.colour[target]
        if not kept and step.after.colour[target] != step.mover + 1:
            _fail(step, f"a displacement's landing {target} is not the mover's")


def a_window_opens_only_over_an_empty_square(v: View) -> None:
    """An en passant window (run.en_passant) is opened only by a step over a square that was
    empty: a hop over a run, or a displacement, opens none."""
    for step in _alternating(v.record):
        target = step.after.ep_target
        move = step.moves[0]
        if move_type(move) == MOVE_DISPLACE:
            v.judged += 1
            if target not in (NO_WINDOW, NO_SQUARE):
                _fail(step, "a displacement opened an en passant window")
        if target in (NO_WINDOW, NO_SQUARE) or target == step.before.ep_target:
            continue
        v.judged += 1
        if step.before.board[target] or step.after.board[target]:
            _fail(step, f"an en passant window opened over occupied square {target}")


def a_hire_makes_the_piece_the_hirer_s(v: View) -> None:
    """A hire or a bribe (shared.price) makes the piece the hirer's own, of its type, and costs
    coins: a shared piece hired is the hirer's piece from then on."""
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) != MOVE_HIRE:
            continue
        square = move_to(move)
        if square in step.phase_targets:
            continue  # the wild phase stepped onto it after the hire
        v.judged += 1
        before, after = step.before.board[square], step.after.board[square]
        if not after or v.owner(after) != step.mover or v.type(after) != v.type(before):
            _fail(step, f"the hired piece on {square} is not the hirer's {v.c.piece_ids[v.type(before)]!r}")
        if step.after.purse[step.mover] >= step.before.purse[step.mover] and not v.c.income:
            _fail(step, "a hire cost nothing")


# --------------------------------- R-L6d: the lifted pairs outside batches A to F

def _reach_on(v: View, frame: int, piece_type: int, origin: int, board: bytes,
              unmoved: bool, side: int) -> dict[int, int]:
    """Every square the compiled tables give ``piece_type`` from ``origin`` in ``frame`` on
    ``board``, to the modes it may land there by: a leap's landings as they stand, a slide's
    (a turning path's, a lame ride's) up to and including the first square holding anything
    -- a leg's square has no mode and still ends the ride -- and a hop's past its first piece
    (an owner-named screen read for ``side``) up to the next. A hop over a run is not read
    (its landing is not the plain hop's); its type is not judged."""
    c = v.c
    found: dict[int, int] = {}
    for targets, modes, first_move in c.leaps[frame][piece_type][origin]:
        if first_move and not unmoved:
            continue
        for sq, mode in zip(targets, modes, strict=True):
            found[sq] = found.get(sq, 0) | mode
    # A sweep over an empty line, onto an empty square or a piece it takes, is the slide's
    # word (`MOVE_NORMAL`), so its table is read as a slide's.
    rides = [entry[:3] for entry in c.slides[frame][piece_type][origin]]
    rides += [entry[:3] for entry in c.sweeps[frame][piece_type][origin]]
    for ray, modes, first_move in rides:
        if first_move and not unmoved:
            continue
        for sq, mode in zip(ray, modes, strict=True):
            if mode:
                found[sq] = found.get(sq, 0) | mode
            if board[sq]:
                break
    for hop in c.hops[frame][piece_type][origin]:
        path, path_modes, needs_first, code = hop[0], hop[1], hop[2], hop[4]
        if needs_first and not unmoved:
            continue
        screen = next((i for i, sq in enumerate(path) if board[sq]), None)
        if screen is None:
            continue
        owner = v.owner(board[path[screen]])
        # The core's reading (`tables.rs::screen_matches`): `own` the side's pieces, `enemy`
        # every piece not the side's, a nobody's or wild piece too (R-L6b's soak, seed 3186030).
        if code == SCREEN_OWN and owner != side or code == SCREEN_ENEMY and owner == side:
            continue
        for sq, mode in zip(path[screen + 1:], path_modes[screen + 1:], strict=True):
            if mode:
                found[sq] = found.get(sq, 0) | mode
            if board[sq]:
                break
    return found


def _lands_by(v: View, board: bytes, target: int, mover: int, modes: int) -> bool:
    """Whether a landing of ``modes`` may land on ``target`` as ``mover``'s: onto an empty
    square by a move mode, onto anyone else's piece by a capture mode."""
    victim = board[target]
    if not victim:
        return bool(modes & MODE_MOVE)
    return v.owner(victim) != mover and bool(modes & MODE_CAPTURE)


def _ride_types(v: View) -> set[int]:
    """The types with a turning path, a lame jump or a leap under `blocks_leaps`, and no hop
    over a run (whose landing the plain reading does not give)."""
    blocks = bool((_rules_of(v).get("obstacles") or {}).get("blocks_leaps"))
    found = set()
    for piece in v.spec.get("pieces", []):
        moves = [m for m in piece.get("moves", []) or [] if isinstance(m, dict)]
        if any(m.get("run") for m in moves):
            continue
        rides = any(m.get("path", "straight") != "straight" or m.get("kind") == "lame"
                    or (blocks and m.get("kind") == "leap" and m.get("legs")) for m in moves)
        if rides and str(piece["id"]) in v.ids:
            found.add(v.ids[str(piece["id"])])
    return found


def a_ride_lands_only_where_its_tables_reach(v: View) -> None:
    """A turning path (on a hex lattice, PA2; on a lame rider, PA4), a lame jump stating its
    legs on a hex lattice (X1) and a leap stating its legs under `blocks_leaps` on hex (X8)
    land only where the compiled table lets them: every plain move the engine offers a
    player's own piece of such a type lands on a square some entry of its frame reaches on
    that board -- a ride ended at the first square holding anything, its legs' squares
    included, a jump whose stated leg is a wall absent -- by a mode the square allows (an
    empty square a move, the other side's piece a capture). The tables are the compiler's
    reading of the rule (`compile._turning_rides`, `_lame_ride`, the leap's wall test); the
    check holds the core to them on every board the game passed. A transit, a go-through, a
    held or commanded move, a push and a hop over a run are not judged."""
    c = v.c
    types = _ride_types(v)
    if not types:
        return
    for step in _alternating(v.record):
        board = step.before.board
        side = step.mover
        reaches: dict[int, dict[int, int]] = {}
        for move in step.offered:
            kind = move_type(move)
            if kind not in (MOVE_NORMAL, MOVE_PROMOTION) or move & MOVE_FOREIGN:
                continue
            if c.portals is not None and move >> 25 & 0x1F:
                continue
            origin, target = move_from(move), move_to(move)
            value = board[origin]
            if not value or v.owner(value) != side or v.type(value) not in types:
                continue
            if board[target] and v.owner(board[target]) == side:
                continue  # an own link's meeting, contacts' to read
            if origin not in reaches:
                unmoved = bool(step.before.unmoved >> origin & 1)
                reaches[origin] = _reach_on(v, c.frame_at(value), v.type(value), origin, board,
                                            unmoved, side)
            v.judged += 1
            modes = reaches[origin].get(target)
            if modes is None or not _lands_by(v, board, target, side, modes):
                _fail(step, f"a {v.c.piece_ids[v.type(value)]!r} on square {origin} was offered "
                            f"{move:#x} to square {target}, which no entry of its table reaches there")


def a_pounce_follows_its_pouncer_s_moves(v: View) -> None:
    """A pouncer on its prey (N20, SR2: a turning path on a prey pouncer) moves onto a piece
    of the mover's its type could take from where it stands: the compiled tables in the
    first frame (where a wild piece reads its moves) reach the prey's square by a capture
    mode, its path -- a turning ride's every square before it -- empty on the board the
    phase played on. That board is the step's last, the phase undone; the pounces are then
    walked forward in order. A step whose phase holds a copy is not judged."""
    c = v.c
    if c.pounce is None or c.npcs is None:
        return
    for step in _alternating(v.record):
        if not any(s.kind == "pounce" for s in step.phase) or any(s.kind == "copy" for s in step.phase):
            continue
        board = bytearray(step.after.board)
        for s in reversed(step.phase):
            if s.kind == "return":
                board[s.target] = s.eaten
            elif s.origin != s.target:
                board[s.origin], board[s.target] = board[s.target], s.eaten
        for s in step.phase:
            if s.kind == "pounce":
                pouncer = board[s.origin]
                v.judged += 1
                if not pouncer or not v.is_wild_piece(pouncer):
                    _fail(step, f"a pounce from square {s.origin}, where no wild piece stood")
                prey = board[s.target]
                if not prey or v.owner(prey) != step.mover:
                    _fail(step, f"a pounce onto square {s.target}, which held no piece of the mover's")
                reach = _reach_on(v, 0, v.type(pouncer), s.origin, bytes(board), False, WILD)
                if not reach.get(s.target, 0) & MODE_CAPTURE:
                    _fail(step, f"a pouncer on square {s.origin} took square {s.target}, which its "
                                "moves do not reach there")
            if s.kind == "return":
                board[s.target] = step.after.board[s.target]
            elif s.origin != s.target:
                board[s.target], board[s.origin] = board[s.origin], 0


def a_push_moves_no_shielded_or_bound_piece(v: View) -> None:
    """A push beside a sight (push.sight, Q1074b8, `plan-push.md` §3.9 N4): no push the engine
    offers moves a shielded piece anywhere on its line (the target and every piece behind
    it), is made by a frozen or a disarmed pusher, or is a commanded piece's (a command lands
    on empty squares only). The line is `CompiledSpec.push_line`, the effects `Engine.effects`'."""
    c = v.c
    if c.push is None or c.sight is None:
        return
    for step in _alternating(v.record):
        pushes = [m for m in step.offered if move_type(m) == MOVE_PUSH]
        if not pushes:
            continue
        board = step.before.board
        bits = v.engine.effects(step.before).bits
        for move in pushes:
            v.judged += 1
            origin = move_from(move)
            if move & MOVE_FOREIGN:
                _fail(step, f"a commanded piece on square {origin} was offered the push {move:#x}")
            if bits[origin] & (SIGHT_FREEZE | SIGHT_DISARM):
                _fail(step, f"the frozen or disarmed piece on square {origin} was offered the push {move:#x}")
            for square, _ in c.push_line(board, move):
                if bits[square] & SIGHT_SHIELD:
                    _fail(step, f"the push {move:#x} moves the shielded piece on square {square}")


def a_pushed_piece_standing_on_its_goal_decides(v: View) -> None:
    """Beside an arrival goal (push.G1, `plan-push.md` §3.10: *standing decides*), a push that
    leaves a piece its goal admits standing on its own side's goal squares -- the pusher, or
    a piece of either side pushed there -- has ended the game. Read under the plain goal (no
    `safe`, `leave`, `removes` or `count`)."""
    rule = v.c.reach
    if v.c.push is None or rule is None or rule.safe or rule.leaves or rule.removes or rule.count != 1:
        return
    for step in _alternating(v.record):
        if move_type(step.moves[0]) != MOVE_PUSH:
            continue
        after = step.after.board
        standing = [
            (side, sq) for side in (0, 1) for sq in rule.squares[side]
            if after[sq] and v.owner(after[sq]) == side
            and (not rule.piece_types or v.type(after[sq]) in rule.piece_types)
        ]
        v.judged += 1
        if standing and v.engine.terminal(step.after) is None:
            side, sq = standing[0]
            _fail(step, f"after a push player {side}'s piece stands on its goal {sq} and the game went on")


# ------------------------------------------------------------------- the lean copy

def full_copy_engine(spec: dict[str, Any]) -> Any:
    """``spec``'s engine built on the full copy of the core's common path, whatever its
    tables (R-L5c2, `docs/reference/common-path-plan.md` §2.5 item 4): the test-only switch
    is set for this one build and put back."""
    from chess_variant_sandbox import _core
    from chess_variant_sandbox.engine import Engine

    before = _core.force_full_path(True)
    try:
        return Engine(spec)
    finally:
        _core.force_full_path(before)


def the_lean_copy_offers_what_the_full_copy_offers(v: View) -> None:
    """R-L5c2 (§2.5 item 5): a case the core plays on its lean copy (no rare rule in use) is
    offered, at every position of its alternating play, exactly the moves the full copy offers
    there, in the same order (order is tie-break order). The lean copy is the full one with
    the rare arms compiled out, so a difference is a gate whose family `rare_in_use` forgets.
    Judges nothing on a case that plays the full copy."""
    if not v.engine.core.lean_path():
        return
    full = full_copy_engine(v.spec)
    if full.core.lean_path():
        _fail(None, "the forced-full switch built a lean engine")
    for step in _alternating(v.record):
        v.judged += 1
        offered = tuple(full.legal_moves(step.before))
        if offered != step.offered:
            missing = sorted(set(offered) - set(step.offered))
            extra = sorted(set(step.offered) - set(offered))
            _fail(step, f"the lean copy offered {len(step.offered)} moves, the full copy "
                        f"{len(offered)} (only on the full: {missing[:4]}; only on the lean: {extra[:4]})")


# ---------------------------------------------------------------------------- the list

# ------------------------------------------------- R-L4b stage 3: the capture list

def _index(v: View, kind: str) -> int:
    """Where the first entry of ``kind`` stands in the capture list, -1 if none."""
    return next((i for i, e in enumerate(v.entries) if e.get("do") == kind), -1)


def _reads(v: View, kind: str, cause: str, taker: int | None = None) -> bool:
    """Whether some entry of ``kind`` reads takings of ``cause`` (by ``taker``, for a path)."""
    for entry in v.entries:
        if entry.get("do") != kind or cause not in entry["on"]:
            continue
        if (cause == "path" and taker is not None and "path_by" in entry
                and v.c.piece_ids[taker] not in entry["path_by"]):
            continue
        return True
    return False


def _plain_capture(v: View, step: Step) -> tuple[int, int, int, int] | None:
    """A move's own capture of the other side's piece on its target (a step, a promotion, a
    transit's exit), with no wild step or bank on its squares: (origin, target, taker, victim)."""
    move = step.moves[0]
    if move_type(move) not in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL):
        return None
    board = step.before.board
    origin, target = move_from(move), move_to(move)
    taker, victim = board[origin], board[target]
    if not taker or not victim or v.owner(victim) == v.owner(taker):
        return None
    if step.phase or step.after.banked != step.before.banked:
        return None
    return origin, target, taker, victim


def _material(state: Any) -> int:
    """Every piece on the board and in the two hands."""
    return sum(1 for value in state.board if value) + sum(map(sum, state.hands))


_ENTERS = (MOVE_DROP, MOVE_BUY, MOVE_HIRE, MOVE_DUCK, MOVE_PASS, MOVE_KEEP)


def a_taken_piece_is_in_one_place(v: View) -> None:
    """Under a list the core walks, a taken piece is in one place (§2.2 rule 1): no move but an
    entry into play adds a piece to the board and the hands together, so a piece a hand files
    is not also left standing, struck or linked (T2, T5, H4, H5, gating.captures). Where the
    hand reads the landing ahead of every strike and contact table, and no blast stands in the
    list, a move's own capture files exactly one piece (the gate's spend counted back)."""
    if not v.walked or _index(v, "hand") < 0:
        return
    hand_first = _reads(v, "hand", "landing") and _index(v, "blast") < 0 and all(
        _index(v, kind) < 0 or _index(v, "hand") < _index(v, kind) for kind in ("strike", "contacts")
    )
    drops = v.c.drops
    unlimited = drops is not None and bool(drops.unlimited) and any(drops.unlimited)
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) in _ENTERS or step.phase or step.after.banked != step.before.banked:
            continue
        v.judged += 1
        # A gate spends from the hand, but an unlimited slot is never spent (soak seed 33806):
        # the gated piece then enters play as a drop does.
        entered = 1 if (_move_gate(move) and unlimited) else 0
        if _material(step.after) > _material(step.before) + entered:
            _fail(step, f"the move left {_material(step.after) - _material(step.before)} more "
                        "pieces on the board and in the hands than stood there")
        capture = _plain_capture(v, step)
        if not hand_first or capture is None or unlimited:
            continue
        victim = capture[3]
        if v.type(victim) in v.c.royal_types or v.owner(victim) not in (0, 1):
            continue
        side = step.mover
        spent = 1 if _move_gate(move) else 0
        gained = sum(step.after.hands[side]) - sum(step.before.hands[side]) + spent
        if gained != 1:
            _fail(step, f"a capture the hand reads first filed {gained} pieces in the mover's hand")


def a_spared_piece_stands_after_its_blast(v: View) -> None:
    """Under a list the core walks, a capture whose blast reads it takes its capturer (T1.position,
    T4.switch: a link's mover half finds nothing to change); a choosing strike or contact behind
    a blast that destroys the taken piece first offers one word for the capture, not one per
    square (T1.choose, T4.choose); a blast that spares the taken piece ahead of a strike placed
    "stay" that reaches it leaves it standing, struck, where it was taken (T1.position)."""
    blast = v.c.explosion
    if not v.walked or blast is None or not _reads(v, "blast", "landing"):
        return
    shields = v.c.sight is not None and v.c.sight.has_shield
    spares = any(e.get("do") == "blast" and e.get("spares_taken") for e in v.entries)
    strike = _strike_spelling(v)
    choosing = [k for k in ("strike", "contacts")
                if _rule_of(v, k).get("position") == "choose" and _index(v, "blast") < _index(v, k)]
    for step in _alternating(v.record):
        capture = _plain_capture(v, step)
        if capture is None or step.moves[0] & MOVE_FOREIGN:
            continue
        origin, target, taker, victim = capture
        if not _explodes(v, v.type(taker), v.type(victim)):
            continue
        v.judged += 1
        after = step.after.board
        gated = bool(v.c.drops is not None and v.c.drops.gates is not None and _move_gate(step.moves[0]))
        if (after[origin] == taker and not gated) or after[target] == taker:
            _fail(step, f"the capturer from square {origin} survived the blast its capture set off")
        if choosing and not spares and step.offered:
            words = [m for m in step.offered if move_from(m) == origin and move_to(m) == target
                     and move_type(m) == move_type(step.moves[0])]
            if len(words) > 1:
                _fail(step, f"a capture the blast takes first offered {len(words)} landings for "
                            "the piece it destroys")
        placed = {"stay": target, "swap": origin}.get(strike.get("position", "advance"))
        if (spares and not shields and placed is not None and _index(v, "blast") < _index(v, "strike")
                and strike.get("effect", "capture") != "capture"
                and _reads(v, "strike", "landing") and _strike_reaches(v, v.type(taker), victim)
                and _index(v, "hand") < 0 and not _move_gate(step.moves[0])):
            linked = {link.get("from") for link in strike.get("chain", []) or []}
            if v.c.piece_ids[v.type(victim)] in linked and (
                    not after[placed] or v.owner(after[placed]) != v.owner(victim)):
                _fail(step, f"the spared piece placed on square {placed} does not stand after its blast")


def two_tables_each_read_their_own_meetings(v: View) -> None:
    """A strike and a contact table in one list (T3): a capture whose pair a link names with a
    ``mover_to`` leaves the capturer as that type on the square it took, whichever entry comes
    first, while the strike sends the struck piece elsewhere (placed home, swap or choose)."""
    contacts, strike = _rule_of(v, "contacts"), _strike_spelling(v)
    if not v.walked or not contacts or not strike:
        return
    if contacts.get("position", "advance") != "advance" or strike.get("position") not in ("home", "swap", "choose"):
        return
    if strike.get("switches_sides") or not _reads(v, "contacts", "landing"):
        return
    if _index(v, "blast") >= 0:
        return
    links = {(v.ids.get(str(link.get("mover"))), v.ids.get(str(link.get("onto")))): v.ids.get(str(link["mover_to"]))
             for link in contacts.get("links", []) or [] if link.get("mover_to") and not link.get("own")}
    immune = v.ids_of(contacts.get("immune", []) or []) | set(v.c.royal_types) | v.wild_types
    for step in _alternating(v.record):
        capture = _plain_capture(v, step)
        if capture is None or step.moves[0] & MOVE_FOREIGN or move_type(step.moves[0]) != MOVE_NORMAL:
            continue
        origin, target, taker, victim = capture
        becomes = links.get((v.type(taker), v.type(victim)))
        if becomes is None or v.type(victim) in immune:
            continue
        v.judged += 1
        stands = step.after.board[target]
        if not stands or v.owner(stands) != v.owner(taker) or v.type(stands) != becomes:
            _fail(step, f"the capturer on square {target} is not the "
                        f"{v.c.piece_ids[becomes]!r} its contact link names")


def a_sweep_follows_the_list(v: View) -> None:
    """Under a list the core walks: a sweep whose swept pieces no board entry reads (no blast;
    no strike or contact entry reading its sweeper's path, `path_by`) leaves every swept square
    empty (W1.disagree); a sweep that takes a piece setting off a blast reading its path or
    landing takes the sweeper with it (W3.both)."""
    if not v.walked:
        return
    shields = v.c.sight is not None and v.c.sight.has_shield
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) != MOVE_SWEEP or move & MOVE_FOREIGN or step.phase:
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        sweeper = board[origin]
        if not sweeper or v.owner(sweeper) != step.mover:
            continue
        kind = v.type(sweeper)
        path = [sq for sq in v.c.sweep_path(board, move, step.mover) if sq != target and board[sq]]
        if not path:
            continue
        reads_path = any(_reads(v, k, "path", kind) for k in ("strike", "contacts"))
        after = step.after.board
        if _index(v, "blast") < 0 and not reads_path:
            v.judged += 1
            for sq in path:
                if after[sq]:
                    _fail(step, f"the swept piece on square {sq} still stands, and no entry reads "
                                "its sweeper's path")
        elif _index(v, "blast") >= 0 and not shields and _reads(v, "blast", "path", kind):
            taken = [board[sq] for sq in path if v.owner(board[sq]) != step.mover]
            if board[target] and v.owner(board[target]) != step.mover:
                taken.append(board[target])
            if any(_explodes(v, kind, v.type(victim)) for victim in taken):
                v.judged += 1
                if after[target] == sweeper or after[origin] == sweeper:
                    _fail(step, f"the sweeper from square {origin} survived the blast its sweep set off")


def a_pushed_off_piece_is_taken_as_the_list_says(v: View) -> None:
    """Under a list the core walks, a pushed-off piece is a taking from the square it left: where
    no board entry reads ``"push"`` (none does by default: push.X2, push.X3), a push removes at
    most the one piece it throws off and the pusher stands on its target; where a blast reads it
    and the piece thrown off sets it off, the pusher goes."""
    if not v.walked or v.c.push is None:
        return
    readers = [k for k in ("blast", "strike", "contacts") if _reads(v, k, "push")]
    for step in _alternating(v.record):
        move = step.moves[0]
        if move_type(move) != MOVE_PUSH or step.phase or step.after.banked != step.before.banked:
            continue
        board = step.before.board
        origin, target = move_from(move), move_to(move)
        pusher = board[origin]
        before = sum(1 for value in board if value)
        after_count = sum(1 for value in step.after.board if value)
        if not readers:
            v.judged += 1
            if after_count < before - 1:
                _fail(step, f"a push no entry reads removed {before - after_count} pieces")
            if step.after.board[target] != pusher and move_promotion(move) == 0:
                _fail(step, f"the pusher did not stand on square {target} after a push no entry reads")
        elif readers == ["blast"]:
            thrown = v.c.push_taken(board, move)
            if thrown and _explodes(v, v.type(pusher), v.type(thrown)) and not (
                    v.c.sight is not None and v.c.sight.has_shield):
                v.judged += 1
                if step.after.board[target] == pusher or step.after.board[origin] == pusher:
                    _fail(step, f"the pusher from square {origin} survived the blast its push set off")


def a_gated_piece_enters_before_the_blast(v: View) -> None:
    """A gating capture that sets off a blast enters its gated piece first (§3.2, gating.blast):
    the piece stands on the square the capturer left unless the blast reaches that square and
    does not spare its type."""
    blast = v.c.explosion
    if blast is None or v.c.drops is None or v.c.drops.gates is None:
        return
    if v.walked and not _reads(v, "blast", "landing"):
        return
    if v.c.sight is not None and v.c.sight.has_shield:
        return
    for step in _alternating(v.record):
        capture = _plain_capture(v, step)
        gate = _move_gate(step.moves[0])
        if capture is None or not gate:
            continue
        origin, target, taker, victim = capture
        if not _explodes(v, v.type(taker), v.type(victim)):
            continue
        v.judged += 1
        gated = gate - 1
        stands = step.after.board[origin]
        reached = origin in blast.blast[target] and gated not in blast.immune_types
        if reached and stands:
            _fail(step, f"the piece gated onto square {origin} survived a blast that reaches it")
        if not reached and (not stands or v.type(stands) != gated or v.owner(stands) != step.mover):
            _fail(step, f"the piece gated onto square {origin} is not there after the blast")


def a_move_pays_for_what_it_made_itself(v: View) -> None:
    """Under a move's cost, a capture pays for the type the move made, on the square it took,
    from the mover's purse (U9, U10): a link that changes the capturer, or a switch that turns it,
    after the move is not charged. Judged where nothing else fills a purse on the move (no
    bounty, no income) and the capturer advances."""
    purse = v.c.purse
    if purse is None or not purse.upkeep or v.c.income is not None:
        return
    if v.c.bounty and any(v.c.bounty):
        return
    if v.c.npcs is not None and any(v.c.npcs.bounty):
        return
    for kind in ("strike", "contacts"):
        if _rule_of(v, kind).get("position", "advance") == "stay":
            return
    n_types, n_squares = v.c.n_types, v.c.n_squares
    for step in _alternating(v.record):
        capture = _plain_capture(v, step)
        if capture is None or move_type(step.moves[0]) != MOVE_NORMAL:
            continue
        origin, target, taker, victim = capture
        if _explodes(v, v.type(taker), v.type(victim)):
            continue
        side = step.mover
        cost = purse.upkeep[(side * n_types + v.type(taker)) * n_squares + target]
        paid = step.before.purse[side] - step.after.purse[side]
        v.judged += 1
        if paid != cost:
            _fail(step, f"a capture onto square {target} cost the mover {paid}, not the {cost} its "
                        "own type's arrival there costs")


def a_round_follows_the_list(v: View) -> None:
    """R-L4b5 (T1.rounds, T4.rounds): a round walks the capture list over its takings. No round
    leaves more pieces on the board and in the hands than stood there, a piece entering play
    (a buy, a drop) aside (rule 1: a taken piece is in one place); and an arrival that takes a standing enemy piece
    (no collision on its square, its victim not moving) whose capture sets off the blast leaves
    no piece of the capturer's side on that square, whatever stands ahead of the blast in the
    list (a blast takes its taker where it stands)."""
    if v.c.simultaneous is None or v.c.explosion is None:
        return
    shields = v.c.sight is not None and v.c.sight.has_shield
    for step in v.record.steps:
        if not step.round or step.phase:
            continue
        before, after = step.before, step.after
        # A buy enters play, and so does a drop from an unlimited slot (its count never moves:
        # R-L4b5's soak, seeds 16758 and 88595); a counted drop moves one piece hand to board.
        entered = sum(1 for m in step.moves if move_type(m) in (MOVE_BUY, MOVE_DROP))
        v.judged += 1
        if _material(after) > _material(before) + entered:
            _fail(step, f"the round left {_material(after) - _material(before)} more pieces on the "
                        "board and in the hands than stood there")
        if shields:
            continue
        for i in (0, 1):
            move, other = step.moves[i], step.moves[1 - i]
            if move_type(move) not in (MOVE_NORMAL, MOVE_PROMOTION) or move & MOVE_FOREIGN:
                continue
            origin, target = move_from(move), move_to(move)
            taker, victim = before.board[origin], before.board[target]
            if not taker or not victim or origin == target or v.owner(taker) != i or v.owner(victim) != 1 - i:
                continue
            if move_type(other) in _ENTERS or move_from(other) == target or move_to(other) == target:
                continue
            if not _explodes(v, v.type(taker), v.type(victim)):
                continue
            stands = after.board[target]
            if stands and v.owner(stands) == i:
                _fail(step, f"player {i}'s capture onto square {target} set off its blast and a "
                            "piece of that player still stands there")


CHECKS: tuple[Check, ...] = (
    Check("duck_stands_alone_and_moves_last", ("rules.duck",),
          ("L7", "K2", "K3", "O2", "W9", "V11"), duck_stands_alone_and_moves_last),
    Check("a_turn_is_as_long_as_the_schedule", ("rules.turn",),
          ("L1", "L2", "L3", "L5", "L6", "L7"), a_turn_is_as_long_as_the_schedule),
    Check("en_passant_only_on_the_turn_s_first_move", ("rules.en_passant",), ("L1",),
          en_passant_only_on_the_turn_s_first_move),
    Check("a_resting_piece_is_not_moved", ("pieces[].rest",), ("L2", "V7", "push.N5"),
          a_resting_piece_is_not_moved),
    Check("wild_pieces_keep_their_routes", ("rules.npcs",),
          ("L3", "T6", "T7", "T8", "N17", "O1", "W5", "V11", "N4", "N7"), wild_pieces_keep_their_routes),
    Check("a_wild_piece_comes_back_on_its_first_free_start", ("rules.npcs[].respawn",), ("N7",),
          a_wild_piece_comes_back_on_its_first_free_start),
    Check("a_royal_is_never_left_where_a_wild_piece_steps", ("rules.npcs[].turns_back",),
          ("N5", "N5.forbid_check"), a_royal_is_never_left_where_a_wild_piece_steps,
          any_keys=('rules.win: "checkmate"', "rules.forbid_check")),
    Check("a_safe_goal_is_not_where_a_wild_piece_steps",
          ("rules.npcs[].turns_back", "rules.reach.safe"), ("N5.safe",),
          a_safe_goal_is_not_where_a_wild_piece_steps),
    Check("an_immune_wild_piece_falls_only_to_its_takers", ("rules.npcs[].immune",), ("N17",),
          an_immune_wild_piece_falls_only_to_its_takers),
    Check("a_standing_piece_stands", ('setup.placements[].player: "nobody"',),
          ("R18", "O1", "O2", "O3", "O4", "O6", "P10", "V15"), a_standing_piece_stands),
    Check("no_piece_joins_a_side_from_nowhere", (),
          ("L5", "R20", "O10", "O11", "O12"), no_piece_joins_a_side_from_nowhere,
          any_keys=('pieces[].control: "majority"', 'setup.placements[].player: "nobody"')),
    Check("a_held_piece_moves_in_its_holders_frame", ('pieces[].control: "majority" or "shared"',),
          ("O9", "O13", "O14", "O17", "O18", "O19"), a_held_piece_moves_in_its_holders_frame),
    Check("a_held_piece_promotes_as_its_holder_would",
          ('pieces[].control: "majority" or "shared"', "rules.promotion"), ("O15",),
          a_held_piece_promotes_as_its_holder_would),
    Check("a_held_sweep_removes_only_what_its_holder_could_capture",
          ('pieces[].control: "majority" or "shared"', 'pieces[].moves[].kind: "sweep"'), ("O16",),
          a_held_sweep_removes_only_what_its_holder_could_capture),
    Check("a_purse_stays_between_nothing_and_its_cap", ("rules.purse",),
          ("L6", "R19", "O22", "P9", "O27", "push.purse"), a_purse_stays_between_nothing_and_its_cap),
    Check("a_buy_pays_its_price_onto_its_region", ("rules.purse.market",),
          ("R19", "P9", "O24", "O27"), a_buy_pays_its_price_onto_its_region),
    Check("a_capture_pays_the_bounty_of_what_it_removed", (),
          ("O22", "T8", "W5", "push.purse"), a_capture_pays_the_bounty_of_what_it_removed,
          any_keys=("rules.purse", "rules.npcs")),
    Check("a_castle_moves_both_its_pieces", ("rules.castling",), ("R6", "Z2"),
          a_castle_moves_both_its_pieces),
    Check("a_round_follows_the_list", ("rules.simultaneous", "rules.explode_on_capture"),
          ("T1.rounds", "T4.rounds"), a_round_follows_the_list),
    Check("a_shot_leaves_its_shooter_where_it_stood", ('rules.strike.position: "stay"',),
          ("R11", "T2", "T7", "P3", "O3", "V5", "Y4", "T10"), a_shot_leaves_its_shooter_where_it_stood),
    Check("a_struck_piece_becomes_its_link", ("rules.strike.chain",), ("O3", "Y4", "P3", "T7"),
          a_struck_piece_becomes_its_link),
    Check("a_switcher_changes_sides", ("rules.strike.switches_sides",), ("T10", "Y5"),
          a_switcher_changes_sides),
    Check("the_phase_follows_who_holds_a_spy", ("pieces[].spy",),
          ("O6", "O24", "O27", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"), the_phase_follows_who_holds_a_spy),
    Check("a_sweep_clears_only_what_it_may", ('pieces[].moves[].kind: "sweep"',),
          ("W1", "W2", "W3", "W4", "W5", "W6", "W8", "W9", "W10", "V8"), a_sweep_clears_only_what_it_may),
    Check("a_capture_that_explodes_takes_the_capturer", ("rules.explode_on_capture",),
          ("T1", "T4", "T8", "V2", "O12", "O22", "P5", "W3", "H6", "K3"),
          a_capture_that_explodes_takes_the_capturer),
    Check("a_meeting_follows_its_link", ("rules.contacts",),
          ("T5", "T6", "T9", "V6", "P4", "O4", "W2", "Y6"), a_meeting_follows_its_link),
    Check("must_capture_offers_only_captures", ("rules.must_capture",), ("T9",),
          must_capture_offers_only_captures),
    Check("a_promotion_takes_only_a_lost_target", ("rules.promotion.only_lost",), ("H1", "H2", "H3"),
          a_promotion_takes_only_a_lost_target),
    Check("a_promoting_sweeper_promotes", ('pieces[].moves[].kind: "sweep"', "rules.promotion"),
          ("W7", "V8"), a_promoting_sweeper_promotes),
    Check("portal_ends_are_never_taken", ("rules.portals",),
          ("K2", "P1", "P3", "P4", "P5", "P6", "P7", "P9", "P10", "V1", "W8", "R9"),
          portal_ends_are_never_taken),
    Check("a_piece_stays_in_its_region", ("pieces[].region",), ("P7",), a_piece_stays_in_its_region),
    Check("a_move_claims_where_it_lands", ("rules.territory",), ("P6", "V4", "O10"),
          a_move_claims_where_it_lands),
    Check("an_arrival_on_a_goal_decides", ("rules.reach",), ("V3", "O11"), an_arrival_on_a_goal_decides),
    Check("a_frozen_piece_makes_no_move", ("pieces[].sight",),
          ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V15", "W10"),
          a_frozen_piece_makes_no_move),
    Check("running_out_of_one_s_own_kind_ends_it", ('rules.extinction.per_type: "own"',), ("H11",),
          running_out_of_one_s_own_kind_ends_it),
    # U2 (`unheld-pairs-plan.md` §2.2).
    Check("income_fills_the_purse_alone", ("rules.income",), ("income.I1",),
          income_fills_the_purse_alone),
    Check("a_banked_piece_is_counted_off_the_board", ("rules.reach.removes",),
          ("goals.extinction", "goals.bare_royal"), a_banked_piece_is_counted_off_the_board,
          any_keys=("rules.extinction", "rules.bare_royal")),
    Check("only_a_player_s_unmoved_piece_gates", ("rules.drops.gating",),
          ("gating.control", "gating.duck"), only_a_player_s_unmoved_piece_gates),
    Check("a_displaced_piece_claims_nothing", ("rules.territory",),
          ("run.territory", "push.G2"), a_displaced_piece_claims_nothing,
          any_keys=('pieces[].moves[].displace: "back"', "pieces[].moves[].push")),
    Check("a_window_opens_only_over_an_empty_square", ("rules.en_passant",),
          ("run.en_passant",), a_window_opens_only_over_an_empty_square),
    Check("a_hire_makes_the_piece_the_hirer_s", ("rules.purse",), ("shared.price",),
          a_hire_makes_the_piece_the_hirer_s),
    # R-L6d: the lifted pairs outside batches A to F (SR1a, SR1c, SR2, SR11a, Q1074b8).
    Check("a_ride_lands_only_where_its_tables_reach", (), ("PA2", "PA4", "X1", "X8"),
          a_ride_lands_only_where_its_tables_reach,
          any_keys=("pieces[].moves[].path", 'pieces[].moves[].kind: "lame" (stating its legs)',
                    'moves[].kind: "lame" (a rider)',
                    "rules.obstacles.blocks_leaps (leaps stating their legs)")),
    Check("a_pounce_follows_its_pouncer_s_moves", ('rules.npcs[].pounce: "prey"',), ("N20",),
          a_pounce_follows_its_pouncer_s_moves),
    Check("a_push_moves_no_shielded_or_bound_piece", ("pieces[].moves[].push", "pieces[].sight"),
          ("push.sight",), a_push_moves_no_shielded_or_bound_piece),
    Check("a_pushed_piece_standing_on_its_goal_decides", ("pieces[].moves[].push", "rules.reach"),
          ("push.G1",), a_pushed_piece_standing_on_its_goal_decides),
    # R-L4b stage 3: the capture list (`capture-effects-design.md` §6 stage 3's seven checks).
    Check("a_taken_piece_is_in_one_place", ("rules.capture: a hand",),
          ("T2.keeping", "T5.keeping", "H4", "H5", "gating.captures"), a_taken_piece_is_in_one_place),
    Check("a_spared_piece_stands_after_its_blast", ("rules.capture: a blast",),
          ("T1.position", "T1.choose", "T4.switch", "T4.choose"), a_spared_piece_stands_after_its_blast),
    Check("two_tables_each_read_their_own_meetings", ("rules.capture: a strike", "rules.capture: contacts"),
          ("T3",), two_tables_each_read_their_own_meetings),
    Check("a_sweep_follows_the_list", ('pieces[].moves[].kind: "sweep"',),
          ("W1.disagree", "W3.both"), a_sweep_follows_the_list),
    Check("a_pushed_off_piece_is_taken_as_the_list_says", ("pieces[].moves[].push",),
          ("push.X2.strike", "push.X2.contacts", "push.X3"), a_pushed_off_piece_is_taken_as_the_list_says,
          any_keys=("rules.capture: a blast", "rules.capture: a strike", "rules.capture: contacts")),
    Check("a_gated_piece_enters_before_the_blast", ("rules.drops.gating", "rules.capture: a blast"),
          ("gating.blast",), a_gated_piece_enters_before_the_blast),
    Check("a_move_pays_for_what_it_made_itself", ("rules.purse.upkeep",),
          ("upkeep.U9", "upkeep.U10"), a_move_pays_for_what_it_made_itself,
          any_keys=("rules.capture: contacts", "rules.capture: a strike that switches sides")),
    # R-L5c2: on every case; judges only one the core plays on its lean copy.
    Check("the_lean_copy_offers_what_the_full_copy_offers", (), (),
          the_lean_copy_offers_what_the_full_copy_offers),
)


# A `[[lifted]]` row with no check says why here (R-L6d): an id with a reason a reader can
# weigh, never an empty string. Every row lifted by 2026-11-14 (107) has a check, so it is
# empty; `tests/test_fuzz_lifted_coverage.py` fails on a new row in neither place.
UNCHECKED: dict[str, str] = {}


def checked_pairs() -> set[str]:
    """Every `[[lifted]]` id some check was written for."""
    return {pair for check in CHECKS for pair in check.pairs}
