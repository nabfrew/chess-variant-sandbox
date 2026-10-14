"""An independent reading of schema 19's portals, for tests only (Q771; schema 21, Q843a).

A declared test oracle under CLAUDE.md rule 3: it answers which moves go *through* a portal
in a position, and whether a square is attacked through one, and nothing outside `tests/`
imports it. Fairy-Stockfish cannot play a portal (`fairy/variant.py` refuses the rule), so
this is the second implementation the core's transits are held against
(`docs/plan-portals.md` §5); `tests/hex_geometry.py` is the precedent.

It shares no code with `engine/compile.py` or the core. It reads the spec's own move
descriptors -- vectors, symmetry, mode, range, min_range, screens, the first-move and rank
conditions -- and walks them over the board square by square with its own wrap and wall
stepping, where the compiler resolves them once into tables and the core walks those. Only
the spelling helpers of `spec/schema.py` are shared (`wrap_stated`, `promotion_squares`,
`obstacle_squares`, `portal_groups`): what a spec *says* is the validator's to read, and what
it *means* is what is being checked. Schema 21 (`docs/plan-portals-2.md` §2): each end leads
to the end its ``to`` names (a pair's to each other), or nowhere, and lets in a player's
piece only when the player is in its ``enter`` and the piece in its ``carries``.

A position is ``{square: (piece id, owner)}`` with owner 0 or 1 for the players and
``"portal"`` for a portal end, whose id is its end's name (``1o``, ``1b``, ``2a``, ...). Squares are
``rank * width + file``. Not supported, because no portal game can have them: a hex lattice,
facings, regions, strikes, sweeps, and drops.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.spec.schema import (
    obstacle_squares,
    parse_square,
    portal_end_ids,
    portal_groups,
    promotion_squares,
    wrap_stated,
)

Position = dict[int, tuple[str, Any]]
# A transit: (origin, entry square, exit square, promoted piece id or None).
Transit = tuple[int, int, int, "str | None"]

_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _symmetric(vector: list[int], symmetry: str) -> set[tuple[int, int]]:
    """Every direction a base vector implies, written out case by case."""
    x, y = vector
    found = {(x, y)}
    if symmetry == "lateral":
        found.add((-x, y))
    if symmetry in ("rotate4", "dihedral8"):
        found |= {(-y, x), (-x, -y), (y, -x)}
    if symmetry == "dihedral8":
        found |= {(a, -b) for a, b in list(found)}
    return found


class PortalOracle:
    """One spec's portals, read straight off the spec."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        board = spec["board"]
        self.width, self.height = board["width"], board["height"]
        rules = spec.get("rules", {})
        self.orientation = spec.get("setup", {}).get("orientation", "mirror")
        self.pieces = {piece["id"]: piece for piece in spec["pieces"]}
        self.walls = obstacle_squares(rules, self.width, self.height, self.orientation)
        stated = wrap_stated(rules.get("wrap"))
        self.files = [
            (value in ("left", "both"), value in ("right", "both")) for value in stated["files"]
        ]
        self.ranks = [
            (value in ("bottom", "both"), value in ("top", "both")) for value in stated["ranks"]
        ]
        self.groups = portal_groups(rules)
        names = iter(portal_end_ids(self.groups))
        # end name -> (its group, its place in it), and the name of the end it leads to.
        self.where: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        named: dict[tuple[int, int], str] = {}
        for group in self.groups:
            for place, end in enumerate(group["ends"]):
                name = next(names)
                self.where[name] = (group, end)
                named[(group["number"], place)] = name
        self.dest: dict[str, str | None] = {
            name: None if end["to"] is None else named[(group["number"], end["to"])]
            for name, (group, end) in self.where.items()
        }
        promotion = rules.get("promotion")
        entries = promotion if isinstance(promotion, list) else ([promotion] if promotion else [])
        self.promotion: dict[str, tuple[Any, Any]] = {}
        for entry in entries:
            zones = promotion_squares(entry, self.width, self.height, self.orientation)
            targets = entry.get("to", [])
            self.promotion[entry["piece"]] = (zones, (targets, bool(entry.get("mandatory", True))))

    # -- geometry -------------------------------------------------------------------

    def step(self, square: int, dx: int, dy: int, player: int) -> int | None:
        """Where one step from ``square`` lands for ``player``, or ``None`` off the board."""
        file, rank = square % self.width + dx, square // self.width + dy
        low, high = self.files[player]
        if file < 0:
            if not low:
                return None
            file %= self.width
        elif file >= self.width:
            if not high:
                return None
            file %= self.width
        low, high = self.ranks[player]
        if rank < 0:
            if not low:
                return None
            rank %= self.height
        elif rank >= self.height:
            if not high:
                return None
            rank %= self.height
        landed = rank * self.width + file
        return None if landed in self.walls else landed

    def _directions(self, move: dict[str, Any], player: int) -> set[tuple[int, int]]:
        found: set[tuple[int, int]] = set()
        for vector in move["vectors"]:
            found |= _symmetric(vector, move.get("symmetry", "none"))
        if player == 0:
            return found
        if self.orientation == "rotate":
            return {(-dx, -dy) for dx, dy in found}
        return {(dx, -dy) for dx, dy in found}

    # -- reaching a portal ----------------------------------------------------------

    def _open(self, position: Position, square: int, player: int, pid: str) -> bool:
        """Whether ``player``'s piece ``pid`` may go in by the end on ``square``."""
        occupant = position.get(square)
        if occupant is None or occupant[1] != "portal":
            return False
        _, end = self.where[occupant[0]]
        if self.dest[occupant[0]] is None or player not in end["enter"]:
            return False
        return end["carries"] is None or pid in end["carries"]

    def entries(
        self, position: Position, player: int, unmoved: set[int], origin: int
    ) -> dict[int, set[tuple[int, str]]]:
        """Each open portal end the piece on ``origin`` can land on, with every
        ``(direction, mode)`` that reaches it; mode is ``"move"`` or ``"capture"``."""
        pid, owner = position[origin]
        assert owner == player
        found: dict[int, set[tuple[int, str]]] = {}
        relative_rank = (
            origin // self.width if player == 0 else self.height - 1 - origin // self.width
        )
        for move in self.pieces[pid]["moves"]:
            condition = move.get("condition") or {}
            if condition.get("first_move") and origin not in unmoved:
                continue
            if "from_rank" in condition and condition["from_rank"] != relative_rank:
                continue
            kind = move["kind"]
            mode = move.get("mode", "both")
            modes = {"move", "capture"} if mode == "both" else {mode}
            limit = move.get("range", 1 if kind in ("leap", "lame") else 0)
            least = move.get("min_range", 1)
            for dx, dy in sorted(self._directions(move, player)):
                sign = _DIRECTIONS.index((_sign(dx), _sign(dy)))

                def hit(
                    square: int, distance: int = 1, sign: int = sign, least: int = least,
                    modes: set[str] = modes,
                ) -> None:
                    if distance >= least and self._open(position, square, player, pid):
                        found.setdefault(square, set()).update((sign, m) for m in modes)

                if kind == "leap":
                    target = self.step(origin, dx, dy, player)
                    if target is not None and target != origin:
                        hit(target)
                elif kind == "lame":
                    if abs(dx) == abs(dy):
                        leg_x, leg_y = dx // 2, dy // 2
                    elif abs(dx) > abs(dy):
                        leg_x, leg_y = _sign(dx), 0
                    else:
                        leg_x, leg_y = 0, _sign(dy)
                    leg = self.step(origin, leg_x, leg_y, player)
                    target = self.step(origin, dx, dy, player)
                    if leg is None or target is None or leg == target or leg in position:
                        continue
                    hit(target)
                elif kind == "slide":
                    self._walk_slide(position, origin, dx, dy, player, limit, hit)
                elif kind == "hop":
                    self._walk_hop(position, origin, dx, dy, player, limit, move, hit)
        return found

    def _ray(self, origin: int, dx: int, dy: int, player: int, limit: int) -> list[int]:
        ray: list[int] = []
        seen = {origin}
        square = origin
        bound = limit if limit > 0 else self.width * self.height
        while len(ray) < bound:
            stepped = self.step(square, dx, dy, player)
            if stepped is None or stepped in seen:
                break
            ray.append(stepped)
            seen.add(stepped)
            square = stepped
        return ray

    def _walk_slide(self, position: Position, origin: int, dx: int, dy: int, player: int,
                    limit: int, hit: Any) -> None:
        for distance, square in enumerate(self._ray(origin, dx, dy, player, limit), start=1):
            if square in position:
                hit(square, distance)
                return

    def _walk_hop(self, position: Position, origin: int, dx: int, dy: int, player: int,
                  limit: int, move: dict[str, Any], hit: Any) -> None:
        ray = self._ray(origin, dx, dy, player, limit)
        screen = next((i for i, square in enumerate(ray) if square in position), None)
        if screen is None:
            return
        owner = position[ray[screen]][1]
        wanted = move.get("screen", "any")
        if (wanted == "own" and owner != player) or (wanted == "enemy" and owner == player):
            return
        reach = move.get("screen_range", 0)
        last = len(ray) if reach == 0 else min(len(ray), screen + 1 + reach)
        for index in range(screen + 1, last):
            if ray[index] in position:
                hit(ray[index], index + 1)
                return

    # -- coming out -------------------------------------------------------------------

    def _partner_square(self, position: Position, entry: int) -> int | None:
        """Where the end the entry end leads to stands, or ``None``."""
        dest = self.dest[position[entry][0]]
        return next(
            (sq for sq, (pid, owner) in position.items() if owner == "portal" and pid == dest),
            None,
        )

    def _may_take(self, attacker: str, victim: str) -> bool:
        allowed = self.pieces[attacker].get("captures")
        return allowed is None or victim in allowed

    def transits(self, position: Position, player: int, unmoved: set[int]) -> set[Transit]:
        """Every pseudo-legal move through a portal of ``player``'s, in no order."""
        found: set[Transit] = set()
        for origin, (pid, owner) in position.items():
            if owner != player:
                continue
            for entry, ways in self.entries(position, player, unmoved, origin).items():
                partner = self._partner_square(position, entry)
                if partner is None:
                    continue
                for direction, mode in ways:
                    exit_square = self.step(partner, *_DIRECTIONS[direction], player)
                    if exit_square is None or exit_square == origin:
                        continue
                    occupant = position.get(exit_square)
                    if occupant is None:
                        if mode != "move":
                            continue
                    elif (
                        mode != "capture" or occupant[1] in (player, "portal")
                        or not self._may_take(pid, occupant[0])
                    ):
                        continue
                    for promoted in self._promotions(pid, player, exit_square):
                        found.add((origin, entry, exit_square, promoted))
        return found

    def _promotions(self, pid: str, player: int, square: int) -> list[str | None]:
        rule = self.promotion.get(pid)
        if rule is None or square not in rule[0][player]:
            return [None]
        targets, mandatory = rule[1]
        return list(targets) + ([] if mandatory else [None])

    def attacked_through(
        self, position: Position, by: int, square: int, unmoved: set[int] | None = None
    ) -> bool:
        """Whether a piece of ``by``'s could take on ``square`` by going in by a portal."""
        unmoved = set(position) if unmoved is None else unmoved
        victim = position.get(square)
        for origin, (pid, owner) in position.items():
            if owner != by:
                continue
            if victim is not None and not self._may_take(pid, victim[0]):
                continue
            for entry, ways in self.entries(position, by, unmoved, origin).items():
                partner = self._partner_square(position, entry)
                if partner is None:
                    continue
                for direction, mode in ways:
                    if mode == "capture" and origin != square and self.step(
                        partner, *_DIRECTIONS[direction], by
                    ) == square:
                        return True
        return False

    def portal_moves(
        self, position: Position, player: int, last: tuple[int, int] | None = None,
    ) -> set[tuple[int, int]]:
        """``(origin, target)`` for every move of a portal end ``player`` may move.

        ``last`` is the move just played, as ``(origin, target)``, or ``None``. Schema 20
        (Q807): when it carried an end of a pair that sets ``no_straight_back``, that end
        may not go straight back -- read here from the move and the spec, not from any
        state the core keeps.
        """
        moved: set[tuple[int, int]] = set()
        for origin, (pid, owner) in position.items():
            if owner != "portal":
                continue
            group, end = self.where[pid]
            if player not in end["moved_by"]:
                continue
            for move in self.pieces[group["piece"]]["moves"]:
                limit = move.get("range", 1 if move["kind"] == "leap" else 0)
                for dx, dy in self._directions(move, player):
                    if move["kind"] == "leap":
                        target = self.step(origin, dx, dy, player)
                        if target is not None and target != origin and target not in position:
                            moved.add((origin, target))
                    elif move["kind"] == "slide":
                        for square in self._ray(origin, dx, dy, player, limit):
                            if square in position:
                                break
                            moved.add((origin, square))
        if last is not None and position.get(last[1], ("", None))[1] == "portal":
            pid = position[last[1]][0]
            if self.where[pid][0]["no_straight_back"]:
                moved.discard((last[1], last[0]))
        return moved


def square_of(name: str, width: int, height: int) -> int:
    coords = parse_square(name, width, height)
    assert coords is not None
    return coords[1] * width + coords[0]
