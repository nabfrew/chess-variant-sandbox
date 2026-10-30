"""The declared Python oracle for a sweep with two lengths (Q1063).

A sweep moves up to ``range`` steps (m, 0 unlimited) along each of its lines and clears up
to ``sweep_range`` (n, default m). Over steps 1..n it may land on any square, empty or the
other side's piece, and removes every piece it passes, its own included. Over steps n+1..m
it moves as a slide: the first piece on the line ends it, taken if it is the other side's.

This module states that again from the spec, by its own means: directions expanded by hand
and squares worked out by file and rank, never through a compiled table or the core. It
plays the chess-shaped test specs only (leaps, straight slides and sweeps; a mirrored
setup; a pawn's first-rank double step, its capture leap, en passant and promotion on the
last rank; castling two steps with the outermost rook), which is why its users' names say
chess. Check is a capture's definition: a royal is in check when some pseudo-legal move of
the other side would take it, as a landing or as a piece swept. A move that leaves the
mover's own royal in check, or sweeps it away, is illegal. CLAUDE.md rule 3 allows it: it
is a declared test oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# (origin, target, promotion type or -1, kind) where kind is "plain", "ep" or "castle".
Move = tuple[int, int, int, str]


def directions(move: dict[str, Any]) -> set[tuple[int, int]]:
    """A descriptor's directions for the first player, its symmetry expanded by hand."""
    out: set[tuple[int, int]] = set()
    symmetry = move.get("symmetry", "none")
    for x, y in move["vectors"]:
        base = {(x, y)}
        if symmetry == "lateral":
            base = {(x, y), (-x, y)}
        if symmetry in ("rotate4", "dihedral8"):
            base = {(x, y), (-y, x), (-x, -y), (y, -x)}
        if symmetry == "dihedral8":
            base |= {(a, -b) for a, b in base}
        out |= base
    return out


@dataclass
class Position:
    """A board of ``(player, type)`` or ``None``, the side to move, the squares whose piece
    has not moved, and the en passant square with its victim."""

    board: list[tuple[int, int] | None]
    side: int
    unmoved: set[int]
    ep: tuple[int, int] | None = None
    history: list[Move] = field(default_factory=list)


class Oracle:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.width = spec["board"]["width"]
        self.height = spec["board"]["height"]
        self.ids = [piece["id"] for piece in spec["pieces"]]
        self.royal = {i for i, piece in enumerate(spec["pieces"]) if piece.get("royal")}
        rules = spec.get("rules", {})
        promotion = rules.get("promotion") or {}
        self.promoter = self.ids.index(promotion["piece"]) if promotion else -1
        self.promote_to = [self.ids.index(p) for p in promotion.get("to", [])]
        castling = rules.get("castling") or {}
        self.castle_king = self.ids.index(castling["king"]) if castling else -1
        self.castle_rook = self.ids.index(castling["rook"]) if castling else -1
        self.en_passant = bool(rules.get("en_passant"))
        for piece in spec["pieces"]:
            for move in piece["moves"]:
                assert move["kind"] in ("leap", "slide", "sweep"), move
                assert "min_range" not in move or move.get("condition"), move
                assert move.get("path", "straight") == "straight", move

    # ------------------------------------------------------------------ geometry

    def _step(self, square: int, dx: int, dy: int) -> int | None:
        file, rank = square % self.width + dx, square // self.width + dy
        if 0 <= file < self.width and 0 <= rank < self.height:
            return rank * self.width + file
        return None

    def _relative_rank(self, square: int, side: int) -> int:
        rank = square // self.width
        return rank if side == 0 else self.height - 1 - rank

    # ------------------------------------------------------------------ moves

    def pseudo(self, pos: Position, side: int) -> list[tuple[Move, tuple[int, ...]]]:
        """Every pseudo-legal move of ``side`` but castling, with the squares it takes from
        (its landing if occupied, every piece it sweeps, an en passant victim)."""
        out: list[tuple[Move, tuple[int, ...]]] = []
        for origin, held in enumerate(pos.board):
            if held is None or held[0] != side:
                continue
            piece_type = held[1]
            for move in self.spec["pieces"][piece_type]["moves"]:
                condition = move.get("condition") or {}
                if "from_rank" in condition and (
                    self._relative_rank(origin, side) != condition["from_rank"]
                ):
                    continue
                mode = move.get("mode", "both")
                quiet, takes = mode != "capture", mode != "move"
                kind = move["kind"]
                for dx, dy in sorted(directions(move)):
                    if side == 1:
                        dy = -dy
                    if kind == "leap":
                        target = self._step(origin, dx, dy)
                        if target is None:
                            continue
                        other = pos.board[target]
                        if other is None and quiet:
                            out.append(((origin, target, -1, "plain"), ()))
                        elif other is not None and other[0] != side and takes:
                            out.append(((origin, target, -1, "plain"), (target,)))
                        elif (
                            other is None and takes and pos.ep is not None
                            and target == pos.ep[0] and piece_type == self.promoter
                        ):
                            out.append(((origin, target, -1, "ep"), (pos.ep[1],)))
                        continue
                    reach = move.get("range", 0) or self.width * self.height
                    least = move.get("min_range", 1)
                    clears = 0
                    if kind == "sweep":
                        clears = move.get("sweep_range", 0) or reach
                    square, passed = origin, []
                    for step in range(1, reach + 1):
                        nxt = self._step(square, dx, dy)
                        if nxt is None:
                            break
                        square = nxt
                        other = pos.board[square]
                        if step >= least:
                            if other is None and quiet:
                                out.append(((origin, square, -1, "plain"), tuple(passed)))
                            elif other is not None and other[0] != side and takes:
                                out.append(
                                    ((origin, square, -1, "plain"), (*passed, square))
                                )
                        if other is not None:
                            if step > clears:
                                break
                            passed.append(square)
        return self._promote(pos, out)

    def _promote(
        self, pos: Position, moves: list[tuple[Move, tuple[int, ...]]]
    ) -> list[tuple[Move, tuple[int, ...]]]:
        out = []
        for (origin, target, _, kind), taken in moves:
            held = pos.board[origin]
            assert held is not None
            if held[1] == self.promoter and self._relative_rank(target, held[0]) == (
                self.height - 1
            ):
                out.extend(((origin, target, to, kind), taken) for to in self.promote_to)
            else:
                out.append(((origin, target, -1, kind), taken))
        return out

    def in_check(self, pos: Position, side: int) -> bool:
        royals = [
            square for square, held in enumerate(pos.board)
            if held is not None and held[0] == side and held[1] in self.royal
        ]
        if not royals:
            return True
        return any(
            square in taken for _, taken in self.pseudo(pos, 1 - side) for square in royals
        )

    def _attacked(self, pos: Position, square: int, side: int) -> bool:
        """Whether the other side could take a ``side`` royal standing on ``square``."""
        board = list(pos.board)
        board[square] = (side, next(iter(self.royal)))
        probe = Position(board, pos.side, pos.unmoved, pos.ep)
        return any(square in taken for _, taken in self.pseudo(probe, 1 - side))

    def castles(self, pos: Position) -> list[Move]:
        side, out = pos.side, []
        if self.castle_king < 0:
            return out
        for origin, held in enumerate(pos.board):
            if held != (side, self.castle_king) or origin not in pos.unmoved:
                continue
            if self.in_check(pos, side):
                return out
            for dx in (-1, 1):
                square, between = origin, []
                while True:
                    nxt = self._step(square, dx, 0)
                    if nxt is None:
                        break
                    square = nxt
                    if pos.board[square] is not None:
                        break
                    between.append(square)
                rook = square
                if pos.board[rook] != (side, self.castle_rook) or rook not in pos.unmoved:
                    continue
                passed, target = self._step(origin, dx, 0), self._step(origin, 2 * dx, 0)
                if target is None or passed is None or len(between) < 2:
                    continue
                if self._attacked(pos, passed, side):
                    continue
                out.append((origin, target, -1, "castle"))
        return out

    def make(self, pos: Position, move: Move, taken: tuple[int, ...] = ()) -> Position:
        origin, target, promotion, kind = move
        board = list(pos.board)
        held = board[origin]
        assert held is not None
        side = held[0]
        unmoved = set(pos.unmoved) - {origin, target}
        ep = None
        for square in taken:
            board[square] = None
        board[origin] = None
        board[target] = (side, promotion) if promotion >= 0 else held
        if kind == "castle":
            dx = 1 if target > origin else -1
            square = target
            while board[square] != (side, self.castle_rook):
                square += dx
            board[square] = None
            board[origin + dx] = (side, self.castle_rook)
            unmoved.discard(square)
        if (
            held[1] == self.promoter and self.en_passant
            and abs(target // self.width - origin // self.width) == 2
            and target % self.width == origin % self.width
        ):
            ep = ((origin + target) // 2, target)
        return Position(board, 1 - side, unmoved, ep, [*pos.history, move])

    def legal(self, pos: Position) -> list[Move]:
        out = []
        for move, taken in self.pseudo(pos, pos.side):
            if not self.in_check(self.make(pos, move, taken), pos.side):
                out.append(move)
        for move in self.castles(pos):
            if not self.in_check(self.make(pos, move), pos.side):
                out.append(move)
        return out

    def play(self, pos: Position, move: Move) -> Position:
        for candidate, taken in self.pseudo(pos, pos.side):
            if candidate == move:
                return self.make(pos, move, taken)
        return self.make(pos, move)

    def perft(self, pos: Position, depth: int) -> int:
        moves = self.legal(pos)
        if depth == 1:
            return len(moves)
        return sum(self.perft(self.play(pos, move), depth - 1) for move in moves)


def position_of(board: bytes, side: int, unmoved: int, n_types: int,
                ep_target: int | None = None, ep_victim: int | None = None) -> Position:
    """An oracle position from the core's board bytes (one facing): byte ``v`` is player
    ``(v - 1) // n_types`` and type ``(v - 1) % n_types``."""
    cells: list[tuple[int, int] | None] = [
        None if value == 0 else ((value - 1) // n_types, (value - 1) % n_types)
        for value in board
    ]
    squares = {square for square in range(len(board)) if unmoved >> square & 1}
    ep = (ep_target, ep_victim) if ep_target is not None and ep_victim is not None else None
    return Position(cells, side, squares, ep)
