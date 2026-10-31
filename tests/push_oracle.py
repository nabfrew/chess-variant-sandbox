"""The declared Python oracle for the push rule's tables (Q1074b1, `docs/plan-push.md` §1 to §3).

`pieces[].moves[].push`, `pieces[].pushes`, `pieces[].weight`, `pieces[].strength` and
`rules.push`: a move that lands on a piece it may push moves it one step on. This module states, from the spec and by
its own means, what the compiler must have put in its tables:

* **which landings push**, and whose (§1.3): every landing of a pushing entry on an empty
  board, the enemy bit for ``whose`` ``enemy``/``any``, the own bit for ``own``/``any``, and
  the capture bit gone where an enemy is pushed and ``captures`` is false;
* **the step a pushed piece takes** (§3.1): a ray's own step (D1), a leap's vector over the gcd
  of its components (D2), or its rank or file component's sign (D3), in the mover's frame;
* **where that step goes** from a square (`next`): off the board at an edge or onto a
  hexagon's absent cell (W2), stopped by a wall (W1), round a seam where the mover's wrap opens
  one (H3);
* **who may push whom** (§1.4, §2): the lists, the weight shorthand under ``one`` and ``row``,
  never a wild type or the duck; the weights under ``weight``, and each pusher's strength (its
  ``strength``, else its weight; Q1074b4s); and where a type may stand (W5);
* **the pushes a player may make on a board** (Q1074b2, ``push_moves``): the single push of
  the chain ``"one"``, which `tests/test_push_moves.py` holds the core's ``MOVE_PUSH`` words to,
  and since Q1074b4 the rows of ``"row"`` and ``"weight"`` (`tests/test_push_rows.py`);
* **what a push leaves on the board** (Q1074b4, ``after_push``): the pusher on the target, the
  line one step on, its last piece gone where the line meets the edge;
* **the bar on pushing straight back** (Q1074b6, K5, ``bar_after``, and ``push_moves``'
  ``bar``): after a push that moved the piece landed on from ``x`` to ``y`` and did not throw
  it off, no push on the next move carries that piece from ``y`` back onto ``x``, whoever
  pushes and wherever in the line it stands (Q1078); every other move leaves no bar;
* **crushing walls and rings** (Q1074b7, W1, H4): under ``rules.push.walls: "crush"`` a piece
  pushed against a wall is removed, as one pushed off the board (never the pusher's own
  royal); on a wrapped board a line that comes back round to its target is a full ring and
  cannot move, and one that reaches the pusher's origin ends there.

Squares are worked out by file and rank here, never through a compiled table or
`compile._land`, and a hexagon's cells by their own distance test. It covers what the tests ask
of it: leaps and straight slides with no condition, the four square symmetries and ``none``,
the ``files``/``ranks`` wrap, walls named by square, and a hexagon without seams. CLAUDE.md
rule 3 allows it: it is a declared test oracle.
"""

from __future__ import annotations

from math import gcd
from typing import Any

PUSH_OFF = -1
PUSH_STOP = -2
PUSH_CRUSH = -3
ENEMY = 4
OWN = 8
CAPTURE = 2

Step = tuple[int, int]


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


class Board:
    """The spec's board as file and rank: its edges, wrap, walls and hexagon cells."""

    def __init__(self, spec: dict[str, Any]) -> None:
        board = spec["board"]
        self.width, self.height = board["width"], board["height"]
        rules = spec.get("rules", {})
        wrap = rules.get("wrap") or {}
        self.wrap_files = wrap.get("files", "none") == "both"
        self.wrap_ranks = wrap.get("ranks", "none") == "both"
        self.walls = {
            _square(name, self.width)
            for name in (rules.get("obstacles") or {}).get("squares", [])
        }
        self.hexagon = board.get("shape") == "hexagon"
        self.radius = (self.width - 1) // 2

    def is_cell(self, file: int, rank: int) -> bool:
        if not (0 <= file < self.width and 0 <= rank < self.height):
            return False
        if not self.hexagon:
            return True
        q, r = file - self.radius, rank - self.radius
        return max(abs(q), abs(r), abs(q + r)) <= self.radius

    def step(self, square: int, dx: int, dy: int) -> int | None:
        """Where a step from ``square`` lands, ``None`` off the board (a wall is a square)."""
        file, rank = square % self.width + dx, square // self.width + dy
        if self.wrap_files:
            file %= self.width
        if self.wrap_ranks:
            rank %= self.height
        if not self.is_cell(file, rank):
            return None
        return rank * self.width + file


def frame_step(step: Step, player: int, orientation: str = "mirror") -> Step:
    x, y = step
    if player == 0:
        return x, y
    return (-x, -y) if orientation == "rotate" else (x, -y)


def push_of(move: dict[str, Any]) -> dict[str, Any] | None:
    push = move.get("push")
    if push is True:
        push = {}
    if not isinstance(push, dict):
        return None
    return {"whose": "enemy", "captures": False, "along": "move", **push}


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def pushed_step(step: Step, push: dict[str, Any], leap: bool) -> Step:
    x, y = step
    sign = _sign
    if push["along"] == "rank":
        return sign(x), 0
    if push["along"] == "file":
        return 0, sign(y)
    if leap:
        g = gcd(abs(x), abs(y))
        return x // g, y // g
    return x, y


def push_bits(move: dict[str, Any], push: dict[str, Any]) -> int:
    mode = {"move": 1, "capture": 2, "both": 3}[move.get("mode", "both")]
    if push["whose"] in ("enemy", "any"):
        mode |= ENEMY
        if not push["captures"]:
            mode &= ~CAPTURE
    if push["whose"] in ("own", "any"):
        mode |= OWN
    return mode


def landings(
    spec: dict[str, Any], piece_id: str, player: int, origin: int
) -> set[tuple[int, int, Step]]:
    """Every landing a pushing entry of ``piece_id`` reaches from ``origin`` on an empty board:
    its mode (the push bits and what is left of the capture bit) and the pushed piece's step."""
    board = Board(spec)
    orientation = spec.get("setup", {}).get("orientation", "mirror")
    piece = next(p for p in spec["pieces"] if p["id"] == piece_id)
    found: set[tuple[int, int, Step]] = set()
    for move in piece["moves"]:
        push = push_of(move)
        if push is None:
            continue
        mode = push_bits(move, push)
        kind = move["kind"]
        for direction in directions(move):
            step = frame_step(direction, player, orientation)
            pushed = pushed_step(step, push, kind == "leap")
            if kind == "leap":
                target = board.step(origin, *step)
                if target is not None and target not in board.walls and target != origin:
                    found.add((target, mode, pushed))
                continue
            limit = move.get("range", 0) or board.width * board.height
            at, seen = origin, {origin}
            for _ in range(limit):
                nxt = board.step(at, *step)
                if nxt is None or nxt in seen or nxt in board.walls:
                    break
                seen.add(nxt)
                found.add((nxt, mode, pushed))
                at = nxt
    return found


def next_square(spec: dict[str, Any], square: int, step: Step) -> int:
    """Where a piece on ``square`` goes when pushed by ``step``: a wall crushes it under
    ``walls: "crush"`` (Q1074b7) and stops it otherwise."""
    board = Board(spec)
    target = board.step(square, *step)
    if target is None:
        return PUSH_OFF
    if target in board.walls:
        crush = (spec.get("rules", {}).get("push") or {}).get("walls") == "crush"
        return PUSH_CRUSH if crush else PUSH_STOP
    if target == square:
        return PUSH_STOP
    return target


def may_push(spec: dict[str, Any], pusher: str, victim: str) -> bool:
    """Whether ``pusher`` may push ``victim`` by type (its owner is the landing's business)."""
    rules = spec.get("rules", {})
    nobody = {npc["piece"] for npc in rules.get("npcs", [])}
    if "duck" in rules:
        nobody.add(rules["duck"]["piece"])
    pieces = {p["id"]: p for p in spec["pieces"]}
    if victim in nobody:
        return False
    piece = pieces[pusher]
    if not any(push_of(move) for move in piece["moves"]):
        return False
    chain = (rules.get("push") or {}).get("chain", "one")
    if "pushes" in piece:
        return victim in piece["pushes"]
    if chain != "weight" and "weight" in piece:
        return pieces[victim].get("weight", 99) <= piece["weight"]
    return True


def stands(spec: dict[str, Any], piece_id: str, player: int, square: int) -> bool:
    """Whether a pushed ``piece_id`` of ``player`` may stand on ``square`` (W5)."""
    board = Board(spec)
    piece = next(p for p in spec["pieces"] if p["id"] == piece_id)
    region = piece.get("region")
    if region is None:
        return True
    squares = {_square(name, board.width) for name in region}
    if player == 1:
        squares = {
            (board.height - 1 - s // board.width) * board.width + s % board.width
            for s in squares
        }
    return square in squares


# ---------------------------------------------------------------- the moves (Q1074b2)

PushMove = tuple[int, int, Step, int]


def _promotions(spec: dict[str, Any], piece_id: str, player: int, target: int) -> list[int]:
    """The promotion field a pusher's push to ``target`` carries: each target type plus one, or
    ``[0]`` where it does not promote. Reads ``rules.promotion`` with ``rank: "last"`` only,
    which is all the tests here ask of it."""
    rule = spec.get("rules", {}).get("promotion")
    if rule is None or rule["piece"] != piece_id:
        return [0]
    assert rule.get("rank") == "last" and set(rule) <= {"piece", "rank", "to"}, rule
    height, width = spec["board"]["height"], spec["board"]["width"]
    last = height - 1 if player == 0 else 0
    if target // width != last:
        return [0]
    ids = [piece["id"] for piece in spec["pieces"]]
    return [ids.index(to) + 1 for to in rule["to"]]


def push_moves(
    spec: dict[str, Any], decode: Any, board: bytes, player: int,
    bar: tuple[int, int] = (-1, -1),
) -> set[PushMove]:
    """Every push ``player`` may make on ``board``, by this module's own reading of §1.3, §2
    and §3: ``(origin, target, the pushed piece's step, the promotion field)``. Under
    ``"one"`` the line is the piece landed on; under ``"row"`` and ``"weight"`` it runs on to
    the first empty square (the pusher's origin counts as empty), every piece in it one the
    move may push, and under ``"weight"`` their weights summed no more than the pusher's. ``decode(value)`` answers ``(owner, piece id)`` for a board byte (the compiled
    spec's owner-major byte, read only to name the piece). Leaps and straight slides with no
    condition, the mirrored setup, no region; the pseudo-legal list, before any check filter.
    ``bar`` is ``bar_after``'s ``(y, x)``: no push from ``y`` onto ``x``, and none whose line
    carries the piece on ``y`` onto ``x`` (K5, Q1078)."""
    geometry = Board(spec)
    orientation = spec.get("setup", {}).get("orientation", "mirror")
    pieces = {piece["id"]: piece for piece in spec["pieces"]}
    rule = spec.get("rules", {}).get("push") or {}
    edge_removes = rule.get("edge", "removes") == "removes"
    chain = rule.get("chain", "one")
    found: set[PushMove] = set()
    for origin, value in enumerate(board):
        if not value or decode(value)[0] != player:
            continue
        pusher = decode(value)[1]
        for move in pieces[pusher]["moves"]:
            push = push_of(move)
            if push is None:
                continue
            assert "condition" not in move and move["kind"] in ("leap", "slide"), move
            leap = move["kind"] == "leap"
            for direction in directions(move):
                step = frame_step(direction, player, orientation)
                target = _first_piece(geometry, board, origin, step, move, leap)
                if target is None or (origin, target) == bar:
                    continue
                pushed = pushed_step(step, push, leap)
                load = 0
                square = target
                moves = False
                for _ in range(len(board)):
                    owner, victim = decode(board[square])
                    whose = (
                        "own" if owner == player else "enemy" if owner == 1 - player else None
                    )
                    if whose is None or push["whose"] not in (whose, "any"):
                        break
                    if not may_push(spec, pusher, victim):
                        break
                    load += pieces[victim].get("weight", 0)
                    behind = next_square(spec, square, pushed)
                    if behind == PUSH_STOP:
                        break
                    if behind in (PUSH_OFF, PUSH_CRUSH):
                        moves = (behind == PUSH_CRUSH or edge_removes) and not (
                            whose == "own" and pieces[victim].get("royal")
                        )
                        break
                    if (square, behind) == bar:
                        break  # carries the piece just pushed straight back (K5, Q1078)
                    if behind == target:
                        break  # a full ring (H4): no square of it is empty
                    if behind == origin or not board[behind]:
                        moves = True
                        break
                    if chain == "one":
                        break
                    square = behind
                strength = pieces[pusher].get("strength", pieces[pusher].get("weight", 0))
                if not moves or (chain == "weight" and load > strength):
                    continue
                for promoted in _promotions(spec, pusher, player, target):
                    found.add((origin, target, pushed, promoted))
    return found


def _first_piece(
    geometry: Board, board: bytes, origin: int, step: Step, move: dict[str, Any], leap: bool
) -> int | None:
    """The occupied square a leap lands on, or the first piece a slide reaches in its range
    (none before ``min_range`` may stand in its way), or ``None``."""
    if leap:
        target = geometry.step(origin, *step)
        if target is None or target in geometry.walls or target == origin or not board[target]:
            return None
        return target
    limit = move.get("range", 0) or geometry.width * geometry.height
    at, seen = origin, {origin}
    for distance in range(1, limit + 1):
        nxt = geometry.step(at, *step)
        if nxt is None or nxt in seen or nxt in geometry.walls:
            return None
        if board[nxt]:
            return nxt if distance >= move.get("min_range", 1) else None
        seen.add(nxt)
        at = nxt
    return None


def bar_after(
    spec: dict[str, Any], decode: Any, before: bytes, push: PushMove | None, player: int
) -> tuple[int, int]:
    """K5 (Q1074b6): the bar ``(y, x)`` the move ``push`` by ``player`` leaves on ``before``
    (``None`` for a move that is not a push), or ``(-1, -1)``. Only under
    ``rules.push.no_push_back``; alternating turns only. The piece landed on, at ``x``, goes
    one ``step`` on to ``y``; off the board it leaves no bar. Whose it is does not matter
    (Q1078): any push may carry it back."""
    rule = spec.get("rules", {}).get("push") or {}
    assert "turn" not in spec.get("rules", {}) and "duck" not in spec.get("rules", {})
    if push is None or not rule.get("no_push_back"):
        return (-1, -1)
    _origin, target, step, _promoted = push
    standing = next_square(spec, target, step)
    assert standing != PUSH_STOP, push
    if standing in (PUSH_OFF, PUSH_CRUSH):
        return (-1, -1)
    return (standing, target)


def line_squares(spec: dict[str, Any], board: bytes, push: PushMove) -> list[int]:
    """Q1074b8: the squares of the line ``push`` moves, on ``board`` before it: the target and
    every occupied square behind it along the pushed step, up to the first empty square (the
    pusher's origin counts as empty), the edge or a wall. Squares by file and rank."""
    origin, target, step, _promoted = push
    line, square = [target], target
    while True:
        behind = next_square(spec, square, step)
        if behind < 0 or behind in (origin, target) or not board[behind]:
            return line
        line.append(behind)
        square = behind


def after_push(spec: dict[str, Any], board: bytes, push: PushMove) -> bytes:
    """The board after ``push`` (``push_moves``' shape) with no promotion: the pusher on the
    target, every piece of the line one step on along the pushed step, and the line's last
    piece gone where the step from it leaves the board or meets a crushing wall. Squares by
    file and rank."""
    origin, target, step, promoted = push
    assert promoted == 0, push
    out = bytearray(board)
    out[origin] = 0
    carry, square = board[target], target
    out[target] = board[origin]
    while True:
        behind = next_square(spec, square, step)
        assert behind != PUSH_STOP, push
        if behind in (PUSH_OFF, PUSH_CRUSH):
            break
        there = out[behind]
        out[behind] = carry
        if not there:
            break
        carry, square = there, behind
    return bytes(out)
