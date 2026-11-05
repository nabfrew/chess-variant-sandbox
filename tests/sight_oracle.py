"""The declared Python oracle for lines of sight (schema 30, Q999a, `docs/archive/plan-passive.md` §2, §3.6).

`pieces[].sight`: every player's piece standing in another piece's line of sight is under its
effect -- **freeze** (it may not move, and attacks nothing), **disarm** (it may not capture, and
attacks nothing), **shield** (nothing may capture it, and a shielded royal is not in check) --
read from the board as it stands. This module works the rule out a second way and the tests
hold the core to it, move list by move list. CLAUDE.md rule 3 allows it: it is a declared test
oracle.

**The effects** come from the compiler's forward lines (`SightRule.lines`, what each type sees
from each square on an empty board), walked here over the board: a ray stops at the first piece
it meets, an aura does not; a square nearer than ``min_range`` is unseen but still stops a ray.
The core walks the reverse table instead (`Game::effects_at`), so the two agree only if both are
right.

**Attack** is §2.4's definition stated as a list: ``by`` attacks a square when one of the
*sight-stripped twin's* pseudo-legal captures of ``by``'s lands on it (with a stand-in of the
victim's type on an empty square) from a piece the effects leave free to capture, and the piece
on the square is not shielded. The twin is the same game with every sight removed, and, under
checkmate, read as ``royal_capture`` so that its castling asks no royal safety: the oracle asks
it here, under the effects.

**The legal list** is the twin's pseudo-legal list less the moves the effects forbid, less (under
checkmate) the moves that leave the mover's royal attacked on the board they make, and (under a
rest) less the resting pieces' moves unless that leaves none. The engine is asked only for the
twin's pseudo-legal lists and for the board a move makes; it is never asked what an effect does.

**Command** (Q1007c, `docs/archive/plan-schema-31.md` §3.2, §3.3, §3.7): the side to move may also move
each opponent's piece a command binds and no freeze does. Its quiet moves are the owner's own
pseudo-legal moves of that piece, asked of the twin with the owner to move (no window, no hand),
that land on an empty square or turn in place: never a castling, an en passant, a drop or a
promotion (a move that must promote is a promotion word only, so it falls out). Its turncoat
captures (`captures_own`, not disarmed) are read with each of the owner's non-royal pieces in
turn recoloured to the commander's, so that the twin lists the capture of it; a shielded victim
is barred as every capture's is. Each candidate is legal when the board it makes leaves the
commander's royal unattacked. The core's commanded moves follow its own; their order among
themselves is the core's walk, so `normalise` sorts that segment before a comparison. Beside
portals (SR5c2, V1's command arm) the owner's twin list is the reading itself: a commanded
piece goes through a momentum end (a ``MOVE_PORTAL`` with no promotion) or a fold (a plain move
to the far end) exactly where its owner's piece would, by the owner's ``enter``, ``carries``
and wrap, landing on an empty square, or under ``captures_own`` on a recoloured own piece.

**Sweep** (schema 13, `RULES.md`'s `sweep`): a sweep that passes over a piece (the twin lists it
as `MOVE_SWEEP`; one over an empty path is a plain landing) takes every piece it passes,
friendly included, and so is a capture wherever the sight asks whether a move captures: a
disarmed sweeper may not make it, and under `must_capture` it counts. It attacks, for the
attack test, the square it lands on *and* any square it passes over (a sweep over the enemy
royal is check), from a piece the effects leave free. SR6g (W10): a shielded piece may not be
captured, so a twin sweep that would remove one (passing it or landing on it) is no move and
reaches nothing for the attack test, the attacked square's own shield read first.
SR6f (V8): a commanded sweep is the owner's twin's sweep of the commanded piece, kept where
every piece it removes (swept, or under ``captures_own`` taken on the landing, read with the
landing recoloured as a turncoat capture is) is the owner's, non-royal and unshielded, and only
under ``captures_own`` without disarm; without it, only a sweep over an empty line onto an empty
square (a plain landing) is a commanded move. The board a sweep makes is the engine's, as for
every move; a sweep over the mover's only royal (friendly pieces are swept too) leaves it
without one, and under checkmate that move is refused rather than played.

**Wild pieces** (generation 37, N2, `docs/archive/plan-npc.md` §4.2): a wild piece's sight (``whose``
``"any"``) reads the third owner's frame of the forward lines and touches both players' pieces;
effects never touch a wild piece. **A wild piece never takes a shielded piece**: the oracle plays
the phase itself, on the board the move leaves (the twin's move with the twin's phase taken back
off its log): each wild piece, ascending, steps to its route's next square when that is empty or
holds a player's piece its type may capture that was not shielded at the phase's start, and
waits otherwise. A royal on a wild piece's next square that it may take and that is not shielded
is attacked. The oracle reads one wild piece per route and no returns (the walked specs have
neither ``also_at`` nor ``respawn``).

**A sight on a wild piece** (generation 84, SR5f, `docs/plan-schema-next.md` §2.7 V11): a sight
whose ``affects`` names a wild type, under ``whose`` ``"any"``, binds the wild pieces of that type
too (``SightEntry.tame``). The phase reads them at its start: a frozen wild piece does not step,
a disarmed one steps onto an empty square only; a frozen or disarmed wild piece attacks nothing,
so a royal on its next square is not attacked by it; a shielded one is taken by nobody. **A
nobody's piece** (V15) sees as a wild piece does, through the third owner's frame.

**Portals** (SR5c, `docs/plan-schema-next.md` §2.7 row V1, ruled 2026-10-27): a momentum end
stops a line, ray or aura, as a wall does -- the walk sees nothing at or past it. A fold is floor
to a line, and where the walk sees an empty fold that lets the seeing piece in (`admitted`), it
also sees the square of the end that fold leads to and the piece there, never its own piece,
and the walk goes on along the line, not from the exit (one fold per line). An attack through
either kind is the twin's transit (``MOVE_PORTAL``) or go-through (a plain move to the far end)
landing on the square, from a piece the effects leave free; a fold's own move (``MOVE_FOLD``)
moves no piece and takes nothing, so no effect bars it. A commanded piece passes a portal as its
owner's would (**Command** above).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import (
    SIGHT_COMMAND,
    SIGHT_COMMAND_OWN,
    SIGHT_DISARM,
    SIGHT_FREEZE,
    SIGHT_SHIELD,
)
from chess_variant_sandbox.engine.engine import _pack, _unpack
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_FOREIGN,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    MOVE_SWEEP,
    decode_move,
)
from chess_variant_sandbox.engine.state import GameState, PhaseStep

NO_SQUARE = -1


def normalise(moves: list[int]) -> list[int]:
    """A legal list with its commanded moves (Q1007c) sorted at the end, the rest in order."""
    return [m for m in moves if not m & MOVE_FOREIGN] + sorted(
        m for m in moves if m & MOVE_FOREIGN
    )


def twin_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game with every sight removed and, under checkmate, its castling made to ask
    no royal safety (`royal_capture`), which the oracle asks itself."""
    twin = copy.deepcopy(spec)
    for piece in twin["pieces"]:
        piece.pop("sight", None)
    rules = twin.setdefault("rules", {})
    if rules.get("win", "checkmate") == "checkmate":
        rules["win"] = "royal_capture"
    return twin


class SightOracle:
    """The rule worked out from the forward lines and the twin's lists."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.engine = Engine(spec)
        self.twin = Engine(twin_spec(spec))
        self.compiled = self.engine.spec
        assert self.compiled.sight is not None, "the oracle is for a game with a sight"
        self.checkmate = self.compiled.win == "checkmate" and bool(self.compiled.royal_types)
        castling = self.compiled.castling
        self.castling = castling

    # ------------------------------------------------------------------ effects

    def fold_exits(self, state: GameState) -> dict[int, tuple[int, int]]:
        """SR5c: each fold end standing on ``state``'s board, by its square: ``(the type of
        that end, the square of the end it leads to)``; empty without folds. An end that
        leads nowhere is left out."""
        spec = self.compiled
        portals = spec.portals
        if portals is None or portals.folds is None:
            return {}
        where = {
            end: self.engine.fold_square(state, end) for end in range(len(portals.end_types))
        }
        found: dict[int, tuple[int, int]] = {}
        for end, square in where.items():
            end_type = portals.end_types[end]
            to = portals.dest[end_type]
            if square is None or to < 0:
                continue
            far = where[portals.end_types.index(to)]
            if far is not None:
                found[square] = (end_type, far)
        return found

    def effects(
        self, board: bytes, folds: dict[int, tuple[int, int]] | None = None,
    ) -> list[int]:
        """Each square's effect bits on the piece standing there, for its owner. ``folds`` is
        `fold_exits` of the position (SR5c); a momentum end stops every line."""
        spec = self.compiled
        sight = spec.sight
        assert sight is not None
        portals = spec.portals
        folds = folds or {}
        bits = [0] * spec.n_squares

        def bind(square: int, owner: int, entry: Any) -> None:
            occupant = board[square]
            victim_owner = spec.owner_of(occupant)
            if victim_owner <= 1 and (
                entry.whose == "any" or owner > 1
                or (entry.whose == "enemy") == (victim_owner != owner)
            ) and entry.affects >> spec.type_of(occupant) & 1:
                bits[square] |= entry.effect
            elif victim_owner == 2 and entry.tame >> spec.type_of(occupant) & 1:
                bits[square] |= entry.effect  # SR5f: a wild piece or the duck it names

        for origin, value in enumerate(board):
            if not value:
                continue
            owner = spec.owner_of(value)
            piece_type = spec.type_of(value)
            entries = sight.entries[piece_type]
            if not entries or (owner > 1 and spec.frame_at(value) >= len(sight.lines)):
                continue
            for number, ray, seen in sight.lines[spec.frame_at(value)][piece_type][origin]:
                entry = entries[number]
                for index, square in enumerate(ray):
                    occupant = board[square]
                    if (
                        occupant and portals is not None and spec.owner_of(occupant) > 1
                        and portals.is_end(spec.type_of(occupant))
                    ):
                        break  # a momentum end: a wall to every line
                    if seen[index] and occupant:
                        bind(square, owner, entry)
                    if seen[index] and not occupant and square in folds and owner <= 1:
                        assert portals is not None
                        end_type, far = folds[square]
                        if (
                            far != origin and board[far]
                            and portals.admitted(end_type, owner, piece_type)
                        ):
                            bind(far, owner, entry)
                    if occupant and not entry.through:
                        break
        return bits

    # ------------------------------------------------------------------- attack

    def attacked(
        self, state: GameState, by: int, square: int, victim_type: int | None = None
    ) -> bool:
        """Whether ``by`` attacks ``square`` on ``state``'s board under its effects (§2.4)."""
        spec = self.compiled
        board = state.board
        effects = self.effects(board, self.fold_exits(state))
        occupant = board[square]
        if occupant and effects[square] & SIGHT_SHIELD:
            return False
        probe = bytearray(board)
        if not occupant:
            stand_in = victim_type if victim_type is not None else 0
            probe[square] = spec.encode_piece(1 - by, stand_in)
        probe_board = bytes(probe)
        asked = dataclasses.replace(
            state, board=probe_board, side_to_move=by, ep_target=NO_SQUARE,
            ep_victim=NO_SQUARE,
        )
        for move in self.twin.pseudo_legal_moves(asked):
            origin, target, kind, _ = decode_move(move)
            if kind == MOVE_SWEEP:
                reached = (
                    target == square or square in spec.sweep_path(probe_board, move)
                ) and not self._sweeps_a_shield(probe_board, move, effects, square)
            else:
                reached = target == square and kind in (MOVE_NORMAL, MOVE_PROMOTION, MOVE_PORTAL)
            if reached and not effects[origin] & (SIGHT_FREEZE | SIGHT_DISARM):
                return True
        return False

    def _sweeps_a_shield(
        self, board: bytes, move: int, effects: list[int], besides: int | None,
    ) -> bool:
        """SR6g (W10): whether the sweep ``move`` would remove a shielded piece on ``board`` --
        one it passes or the one it lands on -- other than the one on ``besides`` (the square
        an attack test asks about, whose own shield it reads first). Such a sweep is no move:
        a shielded piece may not be captured, so it ends the ray. ``effects`` are the real
        board's, read before the attack test's stand-in."""
        target = decode_move(move)[1]
        removed = [square for square, _ in self.compiled.sweep_victims(board, move)]
        if board[target]:
            removed.append(target)
        return any(
            effects[square] & SIGHT_SHIELD for square in removed if square != besides
        )

    def wild_attacked(self, board: bytes, square: int, victim_type: int | None = None) -> bool:
        """Whether a wild piece steps onto ``square`` next and would take what stands there
        (or a piece of ``victim_type`` on an empty square), under the shields of ``board``."""
        spec = self.compiled
        npcs = spec.npcs
        if npcs is None:
            return False
        occupant = board[square]
        effects = self.effects(board)
        if occupant:
            if spec.owner_of(occupant) > 1 or effects[square] & SIGHT_SHIELD:
                return False
            victim_type = spec.type_of(occupant)
        for origin, value in enumerate(board):
            if (
                value and spec.owner_of(value) > 1 and npcs.next[origin] == square
                and origin != square
                and not effects[origin] & (SIGHT_FREEZE | SIGHT_DISARM)
                and (victim_type is None
                     or spec.capture_allowed[spec.type_of(value)][victim_type])
            ):
                return True
        return False

    def after(self, state: GameState, move: int) -> GameState:
        """The position ``move`` makes: the twin's, with the wild phase played here (a
        shielded piece is never taken) in a game with wild pieces."""
        spec = self.compiled
        # The twin never generates a commanded word, so it plays the word unvetted
        # (`Core.apply`'s `offered_only`, CS6): the oracle builds its words itself.
        packed, phase = self.twin.core.apply(_pack(state), move, False)
        after = _unpack(packed)
        if spec.npcs is None:
            return after
        steps = tuple(PhaseStep.from_core(step) for step in phase)
        board = bytearray(after.board)
        for step in reversed(steps):
            assert step.kind == "step", "the oracle reads no returns"
            board[step.origin] = board[step.target]
            board[step.target] = step.eaten
        shields = self.effects(bytes(board))
        npcs = spec.npcs
        # Listed before anything moves, so a piece steps once.
        standing = [
            origin for origin, value in enumerate(board) if value and spec.owner_of(value) > 1
        ]
        for origin in standing:
            value = board[origin]
            target = npcs.next[origin]
            if target < 0 or target == origin or shields[origin] & SIGHT_FREEZE:
                continue
            there = board[target]
            if there and (
                spec.owner_of(there) > 1
                or shields[origin] & SIGHT_DISARM
                or shields[target] & SIGHT_SHIELD
                or not spec.capture_allowed[spec.type_of(value)][spec.type_of(there)]
            ):
                continue
            board[origin] = 0
            board[target] = value
        return dataclasses.replace(after, board=bytes(board))

    def _royals(self, state: GameState, player: int) -> list[int]:
        spec = self.compiled
        return [
            square for square, value in enumerate(state.board)
            if value and spec.owner_of(value) == player and spec.type_of(value) in spec.royal_types
        ]

    def royal_attacked(self, state: GameState, player: int) -> bool:
        """Whether any of ``player``'s royals stands attacked under the effects."""
        spec = self.compiled
        for square, value in enumerate(state.board):
            if (
                value and spec.owner_of(value) == player
                and spec.type_of(value) in spec.royal_types
                and (self.attacked(state, 1 - player, square)
                     or self.wild_attacked(state.board, square))
            ):
                return True
        return False

    # ------------------------------------------------------------------ the list

    def barred(self, state: GameState, move: int, effects: list[int]) -> bool:
        """Whether the effects on ``state``'s board forbid ``move`` (§2.4, §2.7)."""
        board = state.board
        origin, target, kind, promotion = decode_move(move)
        if kind in (MOVE_DROP, MOVE_FOLD):
            return False
        if effects[origin] & SIGHT_FREEZE:
            return True
        if kind == MOVE_ROTATE:
            return False
        if kind == MOVE_CASTLE:
            rook = self._castling_rook(state, origin, target)
            return rook is not None and bool(effects[rook] & SIGHT_FREEZE)
        if kind == MOVE_SWEEP:
            # It takes what it passes and lands on: a disarm bars it, and (SR6g, W10) so does a
            # shield on any piece it would remove, since a shielded piece ends the ray.
            return bool(effects[origin] & SIGHT_DISARM) or self._sweeps_a_shield(
                board, move, effects, None,
            )
        if kind == MOVE_EN_PASSANT:
            victim = state.ep_victim
        elif board[target]:
            victim = target
        else:
            return False
        if effects[origin] & SIGHT_DISARM:
            return True
        return bool(effects[victim] & SIGHT_SHIELD)

    def _castling_rook(self, state: GameState, origin: int, target: int) -> int | None:
        """The rook a step-based castling moves: the first one outward from the king."""
        spec = self.compiled
        assert self.castling is not None
        width = spec.width
        step = 1 if target > origin else -1
        file = origin % width + step
        while 0 <= file < width:
            square = origin - origin % width + file
            value = state.board[square]
            if (
                value and spec.owner_of(value) == state.side_to_move
                and spec.type_of(value) == self.castling.rook_type
            ):
                return square
            file += step
        return None

    def _castling_safe(self, state: GameState, move: int) -> bool:
        """Castling's royal safety, under the effects: the king not in check and no square it
        crosses (its destination included) attacked, as a piece of the king's type."""
        spec = self.compiled
        origin, target, _, _ = decode_move(move)
        player = state.side_to_move
        king_type = spec.type_of(state.board[origin])
        if self.attacked(state, 1 - player, origin) or self.wild_attacked(state.board, origin):
            return False
        step = 1 if target > origin else -1
        square = origin + step
        while True:
            if self.attacked(state, 1 - player, square, king_type):
                return False
            if not state.board[square] and self.wild_attacked(state.board, square, king_type):
                return False
            if square == target:
                return True
            square += step

    def legal_moves(self, state: GameState) -> list[int]:
        """The alternating legal list (§3.6)."""
        assert self.castling is None or self.castling.king_files is None, (
            "the oracle reads step-based castling only"
        )
        effects = self.effects(state.board, self.fold_exits(state))
        moves = [
            move for move in self.twin.pseudo_legal_moves(state)
            if not self.barred(state, move, effects)
        ]
        moves += self.commanded(state, effects)
        if self.checkmate:
            player = state.side_to_move
            kept = []
            for move in moves:
                if decode_move(move)[2] == MOVE_CASTLE and not self._castling_safe(state, move):
                    continue
                after = self.after(state, move)
                if self._royals(state, 1 - player) and not self._royals(after, 1 - player):
                    # A move that removes the opponent's royal (a blast) wins on the spot,
                    # whatever it leaves of the mover's own (`leaves_royal_attacked`).
                    kept.append(move)
                    continue
                if self._royals(state, player) and not self._royals(after, player):
                    continue  # a sweep over its own last royal loses it: refused, not played
                if not self.royal_attacked(after, player):
                    kept.append(move)
            moves = kept
        return self._must_capture(state, self._rest(state, state.side_to_move, moves))

    def commanded(self, state: GameState, effects: list[int]) -> list[int]:
        """The side to move's commanded moves before royal safety (§3.2, §3.3), sorted."""
        spec = self.compiled
        sight = spec.sight
        assert sight is not None
        if not sight.has_command:
            return []
        player = state.side_to_move
        owner = 1 - player
        drops = spec.drops
        if drops is not None and drops.placement and any(state.hands[player]):
            return []
        board = state.board
        bound = [
            square for square, value in enumerate(board)
            if value and spec.owner_of(value) == owner
            and effects[square] & SIGHT_COMMAND and not effects[square] & SIGHT_FREEZE
        ]
        if not bound:
            return []
        hands = tuple(
            tuple(0 for _ in hand) if side == owner else hand
            for side, hand in enumerate(state.hands)
        )
        asked = dataclasses.replace(
            state, side_to_move=owner, ep_target=NO_SQUARE, ep_victim=NO_SQUARE, hands=hands,
        )
        found: set[int] = set()

        def sweeps_own(move: int, origin: int) -> bool:
            # SR6f (V8): a commanded sweep removes only its owner's non-royal, unshielded
            # pieces, and only under `captures_own` without disarm; the twin's sweep of
            # anything else is no commanded move.
            if not effects[origin] & SIGHT_COMMAND_OWN or effects[origin] & SIGHT_DISARM:
                return False
            return all(
                spec.owner_of(value) == owner and spec.type_of(value) not in spec.royal_types
                and not effects[square] & SIGHT_SHIELD
                for square, value in spec.sweep_victims(board, move)
            )

        for move in self.twin.pseudo_legal_moves(asked):
            origin, target, kind, promotion = decode_move(move)
            if origin in bound and (
                kind == MOVE_ROTATE
                or (kind in (MOVE_NORMAL, MOVE_PORTAL) and not promotion and not board[target])
                or (kind == MOVE_SWEEP and not board[target] and sweeps_own(move, origin))
            ):
                found.add(move | MOVE_FOREIGN)
        stride = spec.n_facings * spec.n_types
        for origin in bound:
            if not effects[origin] & SIGHT_COMMAND_OWN or effects[origin] & SIGHT_DISARM:
                continue
            for victim, value in enumerate(board):
                if (
                    victim == origin or not value or spec.owner_of(value) != owner
                    or spec.type_of(value) in spec.royal_types
                    or effects[victim] & SIGHT_SHIELD
                ):
                    continue
                probe = bytearray(board)
                probe[victim] = value + (player - owner) * stride
                recoloured = dataclasses.replace(asked, board=bytes(probe))
                for move in self.twin.pseudo_legal_moves(recoloured):
                    if decode_move(move)[:3] in (
                        (origin, victim, MOVE_NORMAL), (origin, victim, MOVE_PORTAL),
                    ) and not decode_move(move)[3]:
                        found.add(move | MOVE_FOREIGN)
                    elif decode_move(move)[:3] == (origin, victim, MOVE_SWEEP) and sweeps_own(
                        move, origin,
                    ):
                        # SR6f: a sweep landing on its own side's piece; the path read on the
                        # real board, where the landing is not on it.
                        found.add(move | MOVE_FOREIGN)
        return sorted(found)

    def _must_capture(self, state: GameState, moves: list[int]) -> list[int]:
        """``must_capture`` after the effects: a capture is forced only among those left."""
        if not self.compiled.must_capture:
            return moves
        captures = [
            move for move in moves
            if decode_move(move)[2] in (MOVE_EN_PASSANT, MOVE_SWEEP)
            or (decode_move(move)[2] in (MOVE_NORMAL, MOVE_PROMOTION)
                and state.board[decode_move(move)[1]])
        ]
        return captures or moves

    def legal_moves_for(self, state: GameState, player: int) -> list[int]:
        """A round's list for ``player`` (§2.8): the twin's round list (no check in rounds)
        less the moves the effects of the round's board forbid that player."""
        effects = self.effects(state.board, self.fold_exits(state))
        asked = dataclasses.replace(state, side_to_move=player)
        moves = [
            move for move in self.twin.pseudo_legal_moves(asked)
            if not self.barred(asked, move, effects)
        ]
        return self._must_capture(state, self._rest(state, player, moves))

    def _rest(self, state: GameState, player: int, moves: list[int]) -> list[int]:
        """Schema 29's rest after the effects: a resting piece's moves go, unless that leaves
        none (the lift lifts the rest only, never an effect)."""
        if not self.compiled.rest:
            return moves
        resting = {square for square, _ in state.rest[player]}
        kept = [
            move for move in moves
            if decode_move(move)[2] == MOVE_DROP or decode_move(move)[0] not in resting
        ]
        return kept or moves
