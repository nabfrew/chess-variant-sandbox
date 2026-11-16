"""The declared Python oracle for wild pieces that turn back (generation 38, N5, `docs/archive/plan-npc.md` §4.7).

`rules.npcs[].turns_back`: the entry's route is a **line**, walked first to last and back. Each
piece on it has a heading, forward at the start. In the phase it looks at the next square of its
heading; empty, or holding a player's piece its type may take, it steps there. Otherwise it turns
round and looks once the other way: free, it steps; not, it waits, turned. A piece that lands on
an end faces back into the line; one blocked on an end waits facing in. A cyclic route's piece
steps to its route's next square when that is free, and waits otherwise. CLAUDE.md rule 3 allows
this module: it is a declared test oracle, and it reads the rule off the spec, never off the
compiled tables.

**The phase is played here**, on the board the move leaves: the core's move with its phase taken
back off the phase log (mechanically, newest entry first), then every wild piece standing on it,
stepped by the rule above in the order below, headings first cleared under the pieces a player
took. The oracle answers the board, the headings and the phase log, and the tests hold the core
to all three. It reads no shields and no returns (the walked specs have no sight; a game in
which a return falls is skipped from the comparison at that ply and said so).

**Beside check** (generation 125, SR8h, N5, ruled 2026-10-27: *its attack is the square ahead
after the turn*): a wild piece threatens the square the phase would step it to, played on the
board as it stands -- for a line piece the square ahead, or, where that holds a piece it may not
take, the one behind after it turns. ``legal_moves`` is the core's pseudo-legal list less the
moves that leave the mover's royal eaten by the phase or, on the board the phase leaves (with the
headings it leaves), attacked by the opponent's pseudo-legal captures or by the next phase's
steps; under ``forbid_check`` less the moves that leave the other royal so attacked.

**The order** (generation 123, SR8f, N4, ruled 2026-10-27), for several pieces on one route
(``also_at``): passes over the pieces still to step, ascending; in a pass a piece whose square
ahead (by its heading on a line, its route's next on a cycle) holds a piece still to step waits
for a later pass, unless the two are head-on (each ahead of the other) and it is the one with
the lower place on its route (nearer the route's first square). A pass in which nothing steps
ends the phase, every piece left waiting where it stands.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import move_to
from chess_variant_sandbox.engine.state import NO_SQUARE
from chess_variant_sandbox.spec.schema import npc_routes, npc_turns_back

WILD = 2


class TurningOracle:
    def __init__(self, raw: dict[str, Any]) -> None:
        engine = Engine(raw)
        spec = engine.spec
        self.engine = engine
        self.raw = raw
        rules = self.raw["rules"]
        orientation = self.raw.get("setup", {}).get("orientation", "mirror")
        self.width, self.height = spec.width, spec.height
        self.routes: dict[int, tuple[tuple[int, ...], bool]] = {}
        for label, squares in npc_routes(rules, spec.width, spec.height, orientation):
            index = int(label.split("[")[1].split("]")[0])
            line = npc_turns_back(rules["npcs"][index])
            for square in squares:
                self.routes[square] = (squares, line)
        self.captures = {
            piece["id"]: piece.get("captures") for piece in self.raw["pieces"]
        }
        self.ids = list(spec.piece_ids)
        self.royal = {piece["id"] for piece in self.raw["pieces"] if piece.get("royal")}

    def _owner_type(self, value: int) -> tuple[int, str]:
        spec = self.engine.spec
        owner, rest = divmod(value - 1, spec.n_facings * spec.n_types)
        return owner, self.ids[rest % spec.n_types]

    def _may_enter(self, board: bytearray, wild: int, target: int) -> bool:
        standing = board[target]
        if not standing:
            return True
        owner, victim = self._owner_type(standing)
        if owner == WILD:
            return False
        listed = self.captures[self._owner_type(wild)[1]]
        return listed is None or victim in listed

    def _ahead(self, square: int, back: bool) -> int | None:
        squares, _ = self.routes[square]
        at = squares.index(square) + (-1 if back else 1)
        return squares[at] if 0 <= at < len(squares) else None

    def _ahead_of(self, square: int, heading: set[int]) -> int | None:
        """The square ahead of the piece on ``square`` as the phase starts."""
        squares, line = self.routes[square]
        if line:
            return self._ahead(square, square in heading)
        if len(squares) == 1:
            return None
        return squares[(squares.index(square) + 1) % len(squares)]

    def _order(self, wild: list[int], heading: set[int]) -> list[int]:
        """The stepping order, read off the start headings alone (the module's *order*)."""
        pending = set(wild)
        order: list[int] = []
        while pending:
            stepped = False
            for square in sorted(pending):
                if square not in pending:
                    continue
                ahead = self._ahead_of(square, heading)
                if ahead is not None and ahead in pending:
                    squares, _ = self.routes[square]
                    head_on = self._ahead_of(ahead, heading) == square
                    if not (head_on and squares.index(square) < squares.index(ahead)):
                        continue
                pending.discard(square)
                order.append(square)
                stepped = True
            if not stepped:
                break
        return order

    def _landed(self, to: int, back: bool) -> bool:
        return (not back) if self._ahead(to, back) is None else back

    def phase(
        self, board: bytearray, heading: set[int]
    ) -> list[tuple[int, int, int, str]]:
        """Play the phase on ``board`` and ``heading`` in place; return its log."""
        wild = [sq for sq in range(len(board)) if board[sq] and self._owner_type(board[sq])[0] == WILD]
        heading &= set(wild)
        log: list[tuple[int, int, int, str]] = []
        for square in self._order(wild, set(heading)):
            value = board[square]
            squares, line = self.routes[square]
            if not line:
                target = squares[(squares.index(square) + 1) % len(squares)]
                if target != square and self._may_enter(board, value, target):
                    log.append((square, target, board[target], "step"))
                    board[target], board[square] = value, 0
                continue
            back = square in heading
            heading.discard(square)
            to, after, turned = square, back, False
            ahead = self._ahead(square, back)
            if ahead is None:
                other = self._ahead(square, not back)
                if other is not None and self._may_enter(board, value, other):
                    to, after = other, self._landed(other, not back)
                else:
                    after = not back
            elif self._may_enter(board, value, ahead):
                to, after = ahead, self._landed(ahead, back)
            else:
                other = self._ahead(square, not back)
                if other is not None:
                    turned = True
                    if self._may_enter(board, value, other):
                        to, after = other, self._landed(other, not back)
                    else:
                        after = not back
            if after:
                heading.add(to)
            kind = "turn" if turned else "step"
            if to != square:
                log.append((square, to, board[to], kind))
                board[to], board[square] = value, 0
            elif turned:
                log.append((square, square, 0, "turn"))
        return log

    def check(self, state: GameState, move: int) -> GameState | None:
        """Play ``move`` in the core and compare its phase with the oracle's. Returns the
        position after, or None where a return fell (not this oracle's to read)."""
        after, phase = self.engine.apply_with_phase(state, move)
        if any(step.kind == "return" for step in phase):
            return None
        board = bytearray(after.board)
        for step in reversed(phase):
            value = board[step.target]
            board[step.target] = step.eaten
            board[step.origin] = value
        heading = set(state.heading_back)
        log = self.phase(board, heading)
        core_log = [(s.origin, s.target, s.eaten, s.kind) for s in phase]
        assert log == core_log, (self.engine.describe_move(state, move), log, core_log)
        assert bytes(board) == after.board
        assert tuple(sorted(heading)) == after.heading_back, (heading, after.heading_back)
        return after

    # ------------------------------------------------------------------ beside check

    def _royals(self, board: bytes | bytearray, player: int) -> list[int]:
        found = []
        for sq, value in enumerate(board):
            if value:
                owner, piece = self._owner_type(value)
                if owner == player and piece in self.royal:
                    found.append(sq)
        return found

    def _attacked(
        self, base: GameState, board: bytes | bytearray, heading: set[int], square: int, by: int
    ) -> bool:
        """``by``'s pseudo-legal captures onto ``square``, or a step of the next phase onto it,
        played on a copy of ``board`` and ``heading``."""
        asked = base.with_(
            board=bytes(board), side_to_move=by, unmoved=0, ep_target=NO_SQUARE, ep_victim=NO_SQUARE,
        )
        if any(move_to(move) == square for move in self.engine.pseudo_legal_moves(asked)):
            return True
        steps = self.phase(bytearray(board), set(heading))
        return any(target == square and origin != target for origin, target, _, _ in steps)

    def _after_phase(self, state: GameState, move: int) -> tuple[bytearray, set[int]]:
        """The board and headings ``move`` and its phase leave, the phase played here."""
        after, phase = self.engine.apply_with_phase(state, move)
        board = bytearray(after.board)
        for step in reversed(phase):
            value = board[step.target]
            board[step.target] = step.eaten
            board[step.origin] = value
        heading = set(state.heading_back)
        self.phase(board, heading)
        return board, heading

    def legal_moves(self, state: GameState) -> list[int]:
        """The legal list beside check (the module's *beside check*), or the pseudo-legal list
        where nothing asks check."""
        rules = self.raw["rules"]
        mate = rules.get("win", "checkmate") == "checkmate" and bool(self.royal)
        forbid = rules.get("forbid_check") is True and bool(self.royal)
        moves = self.engine.pseudo_legal_moves(state)
        if not (mate or forbid):
            return moves
        side = state.side_to_move
        before = len(self._royals(state.board, side))
        legal = []
        for move in moves:
            board, heading = self._after_phase(state, move)
            mine = self._royals(board, side)
            if mate and (len(mine) < before or any(
                self._attacked(state, board, heading, r, 1 - side) for r in mine
            )):
                continue
            if forbid and any(
                self._attacked(state, board, heading, r, side) for r in self._royals(board, 1 - side)
            ):
                continue
            legal.append(move)
        return legal
