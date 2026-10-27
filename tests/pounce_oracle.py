"""The declared Python oracle for wild pieces that pounce (generation 39, N3b, `docs/archive/plan-npc.md` §3, §4.3).

`rules.npcs[].pounce: "prey"`: after a move, before the wild pieces step, each pouncer takes a
piece **of the player who has just moved** that it attacks by its own type's capture moves,
moving onto it; with several, the first in the **prey order** (a royal, then the higher declared
value, then the nearer by king's distance, then the mover's reading order: the square nearest the
mover's own first rank, then the one furthest to the mover's left, as the second player reads the
reflection of the first player's squares). The pouncers go in the mover's reading order of their
squares, each reading the board the earlier ones left. Then every wild piece that did not pounce
steps, front first, to its route's next square when the square it stands on is on a route of its
own type and the next square is empty or holds a player's piece its type may take. Check, for the
side to move, is a player's piece attacking its royal, or the phase, played with the side to move
as the mover, taking it (the pass test).

`"attackers"` (generation 40, N3c, §4.4): such a pouncer takes instead a piece of the mover's
that attacks it -- one of the mover's pseudo-legal captures onto its square, from a type the
pouncer's type may take -- by the same prey order, the distance and the reading read off the
attacker's square.

`split` (generation 41, N3d, §4.5): a splitting `"attackers"` pouncer takes all its attackers in
prey order, a copy of itself growing on each attacker's square while room lasts (room: the
type's cap less the wild pieces of its type on the board as the phase has left it), then, if
attackers remain, moves onto the next as plain `"attackers"` does; the rest are spared. A copy
acts from the next phase. CLAUDE.md rule 3 allows this module: it is a declared
test oracle, and it reads the rule off the spec, never off the compiled tables.

**What the engine is asked.** Only pseudo-legal lists and the board a move makes: what a pouncer
attacks is the pseudo-legal captures of a stand-in, a piece of its type belonging to the mover's
opponent on the pouncer's square (no first-move rights, no en passant window), onto the mover's
pieces; what attacks a royal is the opponent's pseudo-legal captures onto it, or a wild piece on
a route square of its own type whose next square the royal stands on. The board a move makes is
the core's with its phase taken back off the phase log, newest entry first; the oracle plays the
phase itself and holds the core's board, phase log, legal list and check answer to its own.

It reads no shield, no respawn and no line (refused beside a pounce: respawn and turning back;
the walked spec has no sight), and the order among the steppers is the queue rule's (front first).
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.engine.state import NO_SQUARE
from chess_variant_sandbox.spec.schema import npc_pounce, npc_routes, npc_split

WILD = 2


class PounceOracle:
    def __init__(self, raw: dict[str, Any]) -> None:
        engine = Engine(raw)
        spec = engine.spec
        self.engine = engine
        self.raw = raw
        rules = raw["rules"]
        self.orientation = raw.get("setup", {}).get("orientation", "mirror")
        self.width, self.height = spec.width, spec.height
        self.ids = list(spec.piece_ids)
        by_id = {piece["id"]: piece for piece in raw["pieces"]}
        self.value = [float(by_id[pid].get("value", 1)) for pid in self.ids]
        self.royal = [bool(by_id[pid].get("royal", False)) for pid in self.ids]
        self.captures = [by_id[pid].get("captures") for pid in self.ids]
        self.pounces = {self.ids.index(npc["piece"]) for npc in rules["npcs"] if npc_pounce(npc)}
        self.kind = {
            self.ids.index(npc["piece"]): npc_pounce(npc) for npc in rules["npcs"] if npc_pounce(npc)
        }
        self.split = {
            self.ids.index(npc["piece"]): npc_split(npc) for npc in rules["npcs"] if npc_split(npc)
        }
        # square -> (the route's next square, the route's type)
        self.route: dict[int, tuple[int, int]] = {}
        for label, squares in npc_routes(rules, self.width, self.height, self.orientation):
            index = int(label.split("[")[1].split("]")[0])
            piece_type = self.ids.index(rules["npcs"][index]["piece"])
            for position, square in enumerate(squares):
                self.route[square] = (squares[(position + 1) % len(squares)], piece_type)

    # ------------------------------------------------------------------ the board

    def owner_type(self, value: int) -> tuple[int, int]:
        spec = self.engine.spec
        owner, rest = divmod(value - 1, spec.n_facings * spec.n_types)
        return owner, rest % spec.n_types

    def may_capture(self, attacker: int, victim: int) -> bool:
        listed = self.captures[attacker]
        return listed is None or self.ids[victim] in listed

    def reading(self, mover: int, square: int) -> tuple[int, int]:
        file, rank = square % self.width, square // self.width
        if mover == 0:
            return rank, file
        if self.orientation == "rotate":
            return self.height - 1 - rank, self.width - 1 - file
        return self.height - 1 - rank, file

    def distance(self, a: int, b: int) -> int:
        return max(abs(a % self.width - b % self.width), abs(a // self.width - b // self.width))

    def _state(self, base: GameState, board: bytearray | bytes, side: int) -> GameState:
        return base.with_(
            board=bytes(board), side_to_move=side, unmoved=0, ep_target=NO_SQUARE, ep_victim=NO_SQUARE,
        )

    # ------------------------------------------------------------------ the rule

    def prey(self, base: GameState, board: bytearray, square: int, mover: int) -> int | None:
        """The square of the piece the pouncer on ``square`` takes, or None."""
        _, piece_type = self.owner_type(board[square])
        if self.kind[piece_type] == "attackers":
            return self.attacker(base, board, square, mover)
        stand_in = bytearray(board)
        stand_in[square] = self.engine.spec.encode_piece(1 - mover, piece_type)
        asked = self._state(base, stand_in, 1 - mover)
        victims = set()
        for move in self.engine.pseudo_legal_moves(asked):
            target = move_to(move)
            if move_from(move) != square or not board[target]:
                continue
            owner, victim = self.owner_type(board[target])
            if owner == mover and self.may_capture(piece_type, victim):
                victims.add(target)
        if not victims:
            return None

        def key(target: int) -> tuple[Any, ...]:
            victim = self.owner_type(board[target])[1]
            return (
                not self.royal[victim], -self.value[victim], self.distance(square, target),
                self.reading(mover, target),
            )

        return min(victims, key=key)

    def attacker(self, base: GameState, board: bytearray, square: int, mover: int) -> int | None:
        """The square of the piece an ``"attackers"`` pouncer on ``square`` takes, or None: the
        mover's pseudo-legal captures onto it, from a type it may take, by the prey order."""
        found = self.attackers(base, board, square, mover)
        return found[0] if found else None

    def attackers(self, base: GameState, board: bytearray, square: int, mover: int) -> list[int]:
        """Every piece an ``"attackers"`` pouncer on ``square`` may take, in prey order."""
        _, piece_type = self.owner_type(board[square])
        asked = self._state(base, board, mover)
        found = set()
        for move in self.engine.pseudo_legal_moves(asked):
            if move_to(move) != square:
                continue
            origin = move_from(move)
            owner, attacker = self.owner_type(board[origin])
            if owner == mover and self.may_capture(piece_type, attacker):
                found.add(origin)
        def key(origin: int) -> tuple[Any, ...]:
            attacker = self.owner_type(board[origin])[1]
            return (
                not self.royal[attacker], -self.value[attacker], self.distance(square, origin),
                self.reading(mover, origin),
            )

        return sorted(found, key=key)

    def step(self, board: bytearray, square: int) -> int | None:
        if square not in self.route:
            return None
        nxt, route_type = self.route[square]
        _, piece_type = self.owner_type(board[square])
        if nxt == square or route_type != piece_type:
            return None
        if not board[nxt]:
            return nxt
        owner, victim = self.owner_type(board[nxt])
        if owner == WILD or not self.may_capture(piece_type, victim):
            return None
        return nxt

    def _order(self, pending: set[int]) -> list[int]:
        """The queue rule: passes over the pending pieces, ascending, a piece whose route's next
        square holds a piece still pending going after it; stopped by a pass that defers
        nobody, or one in which nobody went (a full route)."""
        pending = set(pending)
        order: list[int] = []
        while True:
            deferred = stepped = False
            for square in sorted(pending):
                nxt = self.route.get(square, (square, -1))[0]
                if nxt != square and nxt in pending:
                    deferred = True
                    continue
                pending.discard(square)
                stepped = True
                order.append(square)
            if not deferred or not stepped:
                return order

    def phase(
        self, base: GameState, board: bytearray, mover: int
    ) -> list[tuple[int, int, int, str]]:
        """Play the phase on ``board`` in place, ``mover`` having just moved; return its log."""
        wild = [sq for sq in range(len(board)) if board[sq] and self.owner_type(board[sq])[0] == WILD]
        pouncers = sorted(
            (sq for sq in wild if self.owner_type(board[sq])[1] in self.pounces),
            key=lambda sq: self.reading(mover, sq),
        )
        log: list[tuple[int, int, int, str]] = []
        steppers = set(wild)
        for square in pouncers:
            piece_type = self.owner_type(board[square])[1]
            if piece_type in self.split:
                found = self.attackers(base, board, square, mover)
                if not found:
                    continue
                standing = sum(
                    1 for value in board if value and self.owner_type(value) == (WILD, piece_type)
                )
                room = max(0, self.split[piece_type] - standing)
                for origin in found[:room]:
                    log.append((square, origin, board[origin], "copy"))
                    board[origin] = board[square]
                if len(found) > room:
                    victim = found[room]
                    log.append((square, victim, board[victim], "pounce"))
                    board[victim], board[square] = board[square], 0
                steppers.discard(square)
                continue
            victim = self.prey(base, board, square, mover)
            if victim is None:
                continue
            log.append((square, victim, board[victim], "pounce"))
            board[victim], board[square] = board[square], 0
            steppers.discard(square)
        for square in self._order(steppers):
            target = self.step(board, square)
            if target is not None:
                log.append((square, target, board[target], "step"))
                board[target], board[square] = board[square], 0
        return log

    # ------------------------------------------------------------------ check and legality

    def royals(self, board: bytes | bytearray, player: int) -> list[int]:
        return [
            sq for sq in range(len(board))
            if board[sq] and self.owner_type(board[sq])[0] == player
            and self.royal[self.owner_type(board[sq])[1]]
        ]

    def attacked_by_tables(self, base: GameState, board: bytes | bytearray, square: int, by: int) -> bool:
        """The tables' reading: ``by``'s pseudo-legal captures onto ``square``, or a wild piece
        on a route square of its own type whose next square it is."""
        asked = self._state(base, board, by)
        if any(move_to(move) == square for move in self.engine.pseudo_legal_moves(asked)):
            return True
        _, victim = self.owner_type(board[square])
        for origin, (nxt, route_type) in self.route.items():
            value = board[origin]
            if nxt != square or nxt == origin or not value:
                continue
            owner, piece_type = self.owner_type(value)
            if owner == WILD and piece_type == route_type and self.may_capture(piece_type, victim):
                return True
        return False

    def in_check(self, state: GameState) -> bool:
        """The side to move's check: a player's piece on its royal, or the pass test."""
        side = state.side_to_move
        board = bytearray(state.board)
        asked = self._state(state, board, 1 - side)
        targets = {move_to(move) for move in self.engine.pseudo_legal_moves(asked)}
        if any(royal in targets for royal in self.royals(board, side)):
            return True
        before = len(self.royals(board, side))
        self.phase(state, board, side)
        return len(self.royals(board, side)) < before

    def after(self, state: GameState, move: int) -> tuple[GameState, list[tuple[int, int, int, str]], bytearray]:
        """The core's position and log for ``move``, and the board of the move alone."""
        after, phase = self.engine.apply_with_phase(state, move)
        board = bytearray(after.board)
        for step in reversed(phase):
            value = board[step.target]
            board[step.target] = step.eaten
            if step.kind != "copy":
                board[step.origin] = value
        return after, [(s.origin, s.target, s.eaten, s.kind) for s in phase], board

    def legal_moves(self, state: GameState) -> list[int]:
        """The pseudo-legal list less the moves that leave the mover's royal taken by the phase
        or attacked on the board the phase leaves (checkmate only)."""
        moves = self.engine.pseudo_legal_moves(state)
        if self.raw["rules"].get("win", "checkmate") != "checkmate":
            return moves
        side = state.side_to_move
        legal = []
        for move in moves:
            _, _, board = self.after(state, move)
            self.phase(state, board, side)
            royals = self.royals(board, side)
            if not royals or any(self.attacked_by_tables(state, board, r, 1 - side) for r in royals):
                continue
            legal.append(move)
        return legal

    def check(self, state: GameState, move: int) -> GameState:
        """Play ``move`` in the core and hold its phase, board and log to the oracle's."""
        after, core_log, board = self.after(state, move)
        log = self.phase(state, board, state.side_to_move)
        assert log == core_log, (self.engine.describe_move(state, move), log, core_log)
        assert bytes(board) == after.board
        return after
