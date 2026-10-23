"""An independent reading of fold portals, for tests only (Q843c; rewritten for schema 24, Q867b).

A declared test oracle under CLAUDE.md rule 3: it answers, from the spec and by brute force,
which moves a position has once folds are on the board -- every leap, slide and hop of the
side to move with the fold rules applied, a fold's own moves, and the momentum transits beside
them -- and so whether a square is attacked. Nothing outside `tests/` imports it.
Fairy-Stockfish cannot play a portal, so this is the second implementation the core is held
against (`docs/archive/plan-portals-2.md` §6); `tests/portal_oracle.py` is the precedent and the base it
builds on.

It shares no code with `engine/compile.py` or the core. It reads each piece's own move
descriptors one at a time (the compiler merges them into rays and the core walks those) and
steps them square by square with `PortalOracle`'s wrap and wall stepping. Only the spelling
helpers of `spec/schema.py` are shared (`portal_groups`, `portal_end_ids` through the base).

The rules, as `docs/archive/plan-portals-2.md` §3R.1 writes them:

* A fold is not a piece and is floor to every move that does not end on it: a slide or hop
  runs on past it, a leap's path and a lame leap's leg cross it, a momentum transit's ray
  reaches a portal beyond it. The ordinary moves are `PortalOracle`'s walks with no fold.
* A **landing step** -- a square one of the piece's moves may end on, with that step's modes
  -- onto an **empty** fold end that admits the piece's player (``enter``) and type
  (``carries``) and leads somewhere is offered twice: the stay (only under a *move* mode, as
  on any square), and the go-through, judged as a landing on the far end's square instead:
  onto it empty under a *move* mode, onto an enemy piece it may take under a *capture* mode,
  never onto the mover's own piece, never when the far end is the mover's own square. It
  promotes by the far square. One fold per move: the go-through ends there.
* Where one target is reached both directly and through a fold the move is the direct one;
  between folds, the lowest end's.
* A square is attacked when some capture move of the attacker ends on it: directly, through a
  fold, or through a momentum portal.

A position is ``{square: (piece id, owner)}`` with owner 0, 1 or ``"portal"`` for a momentum
end (as `PortalOracle`'s), and ``folds`` is ``{end name: square}`` for every fold end.
"""

from __future__ import annotations

from typing import Any

from portal_oracle import PortalOracle, Position

# An ordinary move: (origin, target, promoted piece id or None, fold end name or None).
Move = tuple[int, int, "str | None", "str | None"]
Folds = dict[str, int]


class FoldOracle(PortalOracle):
    """One spec's folds (and momentum portals), read straight off the spec."""

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        self.order = list(self.where)
        self.fold_names = [
            name for name, (group, _) in self.where.items() if group["kind"] == "fold"
        ]

    # -- going through ------------------------------------------------------------------

    def _fold_on(self, folds: Folds, square: int) -> str | None:
        return next((name for name, at in folds.items() if at == square), None)

    def _way_through(
        self, folds: Folds, square: int, pid: str, player: int,
    ) -> tuple[str, int] | None:
        """``(end, far square)`` when a fold stands on ``square`` that lets ``player``'s
        ``pid`` go through: it admits it and leads somewhere. Says nothing of what stands on
        either square."""
        name = self._fold_on(folds, square)
        if name is None:
            return None
        _, end = self.where[name]
        dest = self.dest[name]
        if dest is None or player not in end["enter"]:
            return None
        if end["carries"] is not None and pid not in end["carries"]:
            return None
        return name, folds[dest]

    # -- walking --------------------------------------------------------------------

    def _landings(
        self, position: Position, origin: int, player: int, unmoved: set[int],
    ) -> list[tuple[int, set[str]]]:
        """Every landing step of the piece on ``origin``: ``(square, modes)`` for each square
        one of its moves may end on, empty or not, with the modes that step has there. A
        slide or hop stops at the first piece (the piece's square is still a landing step);
        nothing about a fold changes the walk."""
        pid, _ = position[origin]
        found: list[tuple[int, set[str]]] = []
        rank = origin // self.width if player == 0 else self.height - 1 - origin // self.width
        for move in self.pieces[pid]["moves"]:
            condition = move.get("condition") or {}
            if condition.get("first_move") and origin not in unmoved:
                continue
            if "from_rank" in condition and condition["from_rank"] != rank:
                continue
            kind = move["kind"]
            mode = move.get("mode", "both")
            modes = {"move", "capture"} if mode == "both" else {mode}
            limit = move.get("range", 1 if kind in ("leap", "lame") else 0)
            least = move.get("min_range", 1)
            for dx, dy in sorted(self._directions(move, player)):
                if kind == "leap":
                    target = self.step(origin, dx, dy, player)
                    if target is not None and target != origin and least <= 1:
                        found.append((target, modes))
                elif kind == "lame":
                    if abs(dx) == abs(dy):
                        leg_x, leg_y = dx // 2, dy // 2
                    elif abs(dx) > abs(dy):
                        leg_x, leg_y = (1 if dx > 0 else -1), 0
                    else:
                        leg_x, leg_y = 0, (1 if dy > 0 else -1)
                    leg = self.step(origin, leg_x, leg_y, player)
                    target = self.step(origin, dx, dy, player)
                    if leg is None or target is None or leg == target or target == origin:
                        continue
                    if leg not in position and least <= 1:
                        found.append((target, modes))
                elif kind == "slide":
                    for distance, square in enumerate(
                        self._ray(origin, dx, dy, player, limit), start=1,
                    ):
                        if distance >= least:
                            found.append((square, modes))
                        if square in position:
                            break
                elif kind == "hop":
                    ray = self._ray(origin, dx, dy, player, limit)
                    screen = next((i for i, square in enumerate(ray) if square in position), None)
                    if screen is None:
                        continue
                    screen_owner = position[ray[screen]][1]
                    wanted = move.get("screen", "any")
                    if (wanted == "own" and screen_owner != player) or (
                        wanted == "enemy" and screen_owner == player
                    ):
                        continue
                    reach = move.get("screen_range", 0)
                    last = len(ray) if reach == 0 else min(len(ray), screen + 1 + reach)
                    for index in range(screen + 1, last):
                        if index + 1 >= least:
                            found.append((ray[index], modes))
                        if ray[index] in position:
                            break
        return found

    def _lands(self, position: Position, square: int, pid: str, player: int,
               modes: set[str]) -> bool:
        """Whether a step with ``modes`` may end on ``square``, as it stands."""
        occupant = position.get(square)
        if occupant is None:
            return "move" in modes
        return (
            "capture" in modes and occupant[1] not in (player, "portal")
            and self._may_take(pid, occupant[0])
        )

    def _moves_from(
        self, position: Position, folds: Folds, origin: int, player: int, unmoved: set[int],
    ) -> list[tuple[int, str | None]]:
        """``(target, fold end gone through or None)`` for every move of the piece on
        ``origin``: each landing step as it lands, and, onto an empty fold that lets it
        through, the same step judged on the far end's square."""
        pid, _ = position[origin]
        found: list[tuple[int, str | None]] = []
        for square, modes in self._landings(position, origin, player, unmoved):
            if self._lands(position, square, pid, player, modes):
                found.append((square, None))
            if square in position:
                continue
            through = self._way_through(folds, square, pid, player)
            if through is None:
                continue
            name, far = through
            if far != origin and self._lands(position, far, pid, player, modes):
                found.append((far, name))
        return found

    def ordinary_moves(
        self, position: Position, folds: Folds, player: int, unmoved: set[int],
    ) -> set[Move]:
        """Every leap, slide and hop of ``player``'s pieces, folds applied and duplicates
        folded (direct first, else the lowest end), in no order. No en passant, castling
        or transit: those are other move kinds."""
        found: set[Move] = set()
        for origin, (pid, owner) in position.items():
            if owner != player:
                continue
            best: dict[int, str | None] = {}
            for target, name in self._moves_from(position, folds, origin, player, unmoved):
                if target not in best or best[target] is not None and (
                    name is None or self.order.index(name) < self.order.index(best[target])
                ):
                    best[target] = name
            for target, name in best.items():
                for promoted in self._promotions(pid, player, target, origin):
                    found.add((origin, target, promoted, name))
        return found

    def fold_moves(
        self, position: Position, folds: Folds, player: int,
        last: tuple[str, int, int] | None = None,
    ) -> set[tuple[str, int, int]]:
        """``(end, origin, target)`` for every move of a fold ``player`` may move: onto an
        empty square or a piece of either side, never onto another portal end of either
        kind; a slide stops at the first piece, a hop at the first past its screen. ``last`` is the fold move just played, for
        ``no_straight_back``."""
        taken = set(folds.values())
        moved: set[tuple[str, int, int]] = set()
        for name in self.fold_names:
            group, end = self.where[name]
            if player not in end["moved_by"]:
                continue
            origin = folds[name]

            def lands(square: int) -> bool:
                return square not in taken and position.get(square, ("", None))[1] != "portal"

            for move in group["moves"]:
                limit = move.get("range", 1 if move["kind"] == "leap" else 0)
                for dx, dy in self._directions(move, player):
                    if move["kind"] == "leap":
                        target = self.step(origin, dx, dy, player)
                        if target is not None and target != origin and lands(target):
                            moved.add((name, origin, target))
                    elif move["kind"] == "slide":
                        for square in self._ray(origin, dx, dy, player, limit):
                            if lands(square):
                                moved.add((name, origin, square))
                            if square in position:
                                break
                    elif move["kind"] == "hop":
                        moved |= {
                            (name, origin, square)
                            for square in self._past_screen(position, origin, dx, dy, player,
                                                            limit, move)
                            if lands(square)
                        }
        if last is not None and self.where[last[0]][0]["no_straight_back"]:
            moved.discard((last[0], last[2], last[1]))
        return moved

    # -- momentum portals beside folds ---------------------------------------------------

    def momentum_transits(
        self, position: Position, folds: Folds, player: int, unmoved: set[int],
    ) -> set[Any]:
        """`PortalOracle.transits`: a fold is floor to a ray on its way to a momentum end,
        and a transit may come out onto a fold, where it stays (rule 7)."""
        del folds
        return self.transits(position, player, unmoved)

    def momentum_moves(
        self, position: Position, folds: Folds, player: int,
        last: tuple[int, int] | None = None, unmoved: set[int] | None = None,
    ) -> set[tuple[int, int]]:
        """A momentum end's own moves: never onto a fold (rule 6)."""
        return {
            (origin, target)
            for origin, target in self.portal_moves(position, player, last, unmoved)
            if target not in folds.values()
        }

    # -- attack -----------------------------------------------------------------------

    def attacked(
        self, position: Position, folds: Folds, by: int, square: int, unmoved: set[int],
    ) -> bool:
        """Whether a piece of ``by``'s could take on ``square``: directly, through a fold
        onto its far end, or through a momentum portal. Asked of an occupied square."""
        if any(
            target == square for _, target, _, _ in self.ordinary_moves(position, folds, by, unmoved)
        ):
            return True
        return self.attacked_through(position, by, square, unmoved)
