"""The declared Python oracle for a hop over a run and displacement (TS4, `docs/plan-tessabi.md` §4).

`pieces[].moves[].run` and `pieces[].moves[].displace` on a hop of ``mode: "move"``. This module
states, from the spec and by its own means, what the core must offer and play:

* **the moves a player may make** (``moves``): every step (a ``leap`` or a ``slide`` of range 1)
  onto an empty square, and every hop. A hop with ``run`` jumps the run of occupied squares that
  starts on the square next to the hopper along its direction, every piece of it one its
  ``screen`` allows and the run no longer than its cap, landing on the first square past the
  run if that square is on the board, not a wall, within ``range`` and at least ``min_range``
  away. A hop without ``run`` jumps the first piece on its line (any empty squares before it)
  and lands on each empty square past it up to ``screen_range`` (0 the rest of the line), until
  the next piece. Under ``displace: "back"`` the move is ``"displace"``, else ``"normal"``;
* **what a move leaves on the board** (``after``): the hopper on the target, and under
  ``displace`` every square strictly between origin and target taking what stood on the next
  one, walked from the origin, the last emptying: every jumped piece one square back, the
  nearest onto the origin.

Squares are worked out by file and rank here, never through a compiled table: square boards,
walls named by square (``rules.obstacles.squares``, mirrored where ``mirror`` says), the
mirrored setup, the four square symmetries and ``none``, no condition, no promotion. A board
byte is read only to name its owner and type (``decode``). CLAUDE.md rule 3 allows it: it is a
declared test oracle.
"""

from __future__ import annotations

from typing import Any

Step = tuple[int, int]
# (origin, target, "normal" | "displace")
Move = tuple[int, int, str]


def _square(name: str, width: int) -> int:
    return (int(name[1:]) - 1) * width + "abcdefghijklmnop".index(name[0])


def directions(move: dict[str, Any]) -> set[Step]:
    """A move's directions for the first player, its symmetry expanded by hand."""
    out: set[Step] = set()
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


class Geometry:
    """The spec's board by file and rank: its edges and walls."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.width, self.height = spec["board"]["width"], spec["board"]["height"]
        obstacles = spec.get("rules", {}).get("obstacles") or {}
        walls = {_square(name, self.width) for name in obstacles.get("squares", [])}
        if obstacles.get("mirror"):
            walls |= {
                (self.height - 1 - square // self.width) * self.width + square % self.width
                for square in walls
            }
        self.walls = walls

    def step(self, square: int, dx: int, dy: int) -> int | None:
        """Where one step lands, ``None`` off the board or on a wall."""
        file, rank = square % self.width + dx, square // self.width + dy
        if not (0 <= file < self.width and 0 <= rank < self.height):
            return None
        landed = rank * self.width + file
        return None if landed in self.walls else landed

    def line(self, origin: int, step: Step, limit: int) -> list[int]:
        """The squares a ray from ``origin`` walks, up to ``limit`` (0 the whole board)."""
        out: list[int] = []
        at = origin
        for _ in range(limit or max(self.width, self.height)):
            nxt = self.step(at, *step)
            if nxt is None:
                break
            out.append(nxt)
            at = nxt
        return out


def _screens(screen: Any, player: int, owner: int, piece: str) -> bool:
    if isinstance(screen, list):
        return piece in screen
    if screen == "own":
        return owner == player
    if screen == "enemy":
        return owner != player
    return True


def moves(spec: dict[str, Any], decode: Any, board: bytes, player: int) -> set[Move]:
    """Every move ``player`` may make on ``board`` by this module's reading (the pseudo-legal
    list; the specs it is asked of have no check). ``decode(value)`` answers ``(owner, piece
    id)`` for a board byte."""
    geometry = Geometry(spec)
    pieces = {piece["id"]: piece for piece in spec["pieces"]}
    found: set[Move] = set()
    for origin, value in enumerate(board):
        if not value or decode(value)[0] != player:
            continue
        piece = pieces[decode(value)[1]]
        for move in piece["moves"]:
            assert "condition" not in move and move.get("mode") == "move", move
            for dx, dy in directions(move):
                step = (dx, -dy) if player == 1 else (dx, dy)
                if move["kind"] in ("leap", "slide"):
                    assert move["kind"] == "leap" or move.get("range") == 1, move
                    target = geometry.step(origin, *step)
                    if target is not None and not board[target]:
                        found.add((origin, target, "normal"))
                    continue
                assert move["kind"] == "hop", move
                kind = "displace" if move.get("displace") == "back" else "normal"
                line = geometry.line(origin, step, move.get("range", 0))
                least = move.get("min_range", 1)
                screen = move.get("screen", "any")
                run = move.get("run", False)
                if run is not False:
                    cap = len(line) if run is True else run
                    index = 0
                    while index < len(line) and board[line[index]]:
                        owner, name = decode(board[line[index]])
                        if not _screens(screen, player, owner, name) or index >= cap:
                            index = -1
                            break
                        index += 1
                    if index > 0 and index < len(line) and index + 1 >= least:
                        found.add((origin, line[index], kind))
                    continue
                index = 0
                while index < len(line) and not board[line[index]]:
                    index += 1
                if index >= len(line):
                    continue
                owner, name = decode(board[line[index]])
                if not _screens(screen, player, owner, name):
                    continue
                index += 1
                reach = move.get("screen_range", 0)
                stop = len(line) if not reach else min(len(line), index + reach)
                while index < stop and not board[line[index]]:
                    if index + 1 >= least:
                        found.add((origin, line[index], kind))
                    index += 1
    return found


def after(spec: dict[str, Any], board: bytes, move: Move) -> bytes:
    """The board after ``move`` (``moves``' shape): the hopper on the target and, for a
    displacement, the squares strictly between shifted one back toward the origin. The squares
    between are the straight line from origin to target, found by file and rank."""
    origin, target, kind = move
    width = spec["board"]["width"]
    out = bytearray(board)
    out[target] = board[origin]
    out[origin] = 0
    if kind != "displace":
        return bytes(out)
    fx, fy = origin % width, origin // width
    tx, ty = target % width, target // width
    distance = max(abs(tx - fx), abs(ty - fy))
    sx, sy = (tx - fx) // distance, (ty - fy) // distance
    between = [(fy + sy * k) * width + fx + sx * k for k in range(1, distance)]
    back = origin
    for square in between:
        out[back] = board[square]
        back = square
    out[back] = 0
    return bytes(out)
