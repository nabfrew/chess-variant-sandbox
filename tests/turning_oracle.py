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
ascending, stepped by the rule above, headings first cleared under the pieces a player took. The
oracle answers the board, the headings and the phase log, and the tests hold the core to all
three. It reads one piece per route, no shields and no returns (the walked spec has neither
``also_at``, a sight nor, on the walked entries, a ``respawn`` that is due: a game in which a
return falls is skipped from the comparison at that ply and said so).
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.engine import Engine, GameState
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

    def _landed(self, to: int, back: bool) -> bool:
        return (not back) if self._ahead(to, back) is None else back

    def phase(
        self, board: bytearray, heading: set[int]
    ) -> list[tuple[int, int, int, str]]:
        """Play the phase on ``board`` and ``heading`` in place; return its log."""
        wild = [sq for sq in range(len(board)) if board[sq] and self._owner_type(board[sq])[0] == WILD]
        heading &= set(wild)
        log: list[tuple[int, int, int, str]] = []
        for square in wild:
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
