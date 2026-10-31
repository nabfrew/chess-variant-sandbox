"""The declared Python oracle for neutral pieces and majority control (TC3, `docs/plan-turncoat.md` §4).

`setup.placements[].player: "nobody"` puts pieces in nobody's block; `pieces[].control:
"majority"` lets the player with **strictly more** of their own pieces next to one move it, as
one of their moves, in frame 0, onto an empty square or taking a non-royal piece of the other
player its type may take. A tie, none against none included, means nobody moves it. This module
works the rule out a second way and the tests hold the core to it, move list by move list.
CLAUDE.md rule 3 allows it: it is a declared test oracle.

**Next to** is the oracle's own reading of the geometry: the control's vectors (or the king's
step), expanded by their symmetry, added to the square's file and rank, kept where they land on
the board and on no wall or absent cell. The core reads `compile._compile_square`'s table
instead, so the two agree only if both are right. The oracle reads rectangles without wrap; a
wrap is held by a hand position in `tests/test_neutral_pieces.py`.

**The holder** is counted here from the board's bytes: every piece of each player on a
neighbour square, royals included.

**A held neutral's moves** come from the *frame-0 twin*: the same position with the neutral's
byte recoloured to the holder's piece of the same type at facing 0, asked of the core's
ordinary generator (`pseudo_legal_moves`, the holder to move), kept where they start on the
neutral's square, are not themselves foreign, and land on an empty square or on a non-royal
piece of the holder's opponent (never on another nobody's piece). Each is marked
`MOVE_FOREIGN`. The core's own walk (`foreign.rs::push_held`) never sees the twin.

**Hires and bribes** (TC6, `docs/plan-turncoat.md` §5.5) are worked out here too: under
`rules.purse`, a nobody's piece of a priced type (a hire, at its price) or, under
`rules.purse.bribe`, the opponent's piece of a priced type (a bribe, at twice its price), whose
cost the side to move's purse holds and which stands next to one of the mover's pieces, by the
same reading of *next to* (the type's control geometry, or the king's step). Each is a
`MOVE_HIRE` on the piece's square. The core's `generate_hires` is never asked.

**The legal list** is the core's pseudo-legal own moves (less its hires) plus the oracle's held
moves and hires, each kept
(under `win: "checkmate"`) when the board the core makes of it leaves the mover's royal
unattacked. The engine is asked for the twin's lists, the board a move makes and whether a royal
is attacked (check is unchanged by the rule: a neutral attacks nothing); it is never asked who
holds a neutral or what a held neutral may do.

**A held neutral that takes a royal** (Q1077, `pieces[].takes_royal`, `docs/plan-turncoat.md`
§4.7) may also take the royal of the player it is held against, and so attacks it. The oracle
reads that attack off the same frame-0 twin: a royal is attacked when the core, asked of the
spec **without** the key, says so, or when some such neutral the other player holds has a twin
move onto the royal's square. The core's own reading (`foreign.rs::held_attacks`, over the
third block's attack sources) is never asked.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_FOREIGN,
    MOVE_HIRE,
    MOVE_NORMAL,
    encode_move,
    move_foreign,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.schema import (
    absent_squares,
    control_geometry,
    obstacle_squares,
)

NOBODY_BLOCK = 2


class NeutralOracle:
    """The rule for one spec, asked of positions."""

    def __init__(self, spec: dict[str, Any], engine: Engine) -> None:
        self.spec = spec
        self.engine = engine
        compiled = engine.spec
        self.width = compiled.width
        self.height = compiled.height
        self.n_types = compiled.n_types
        self.n_facings = compiled.n_facings
        self.royal = set(compiled.royal_types)
        self.ids = [piece["id"] for piece in spec["pieces"]]
        board = spec["board"]
        rules = spec.get("rules", {})
        orientation = spec.get("setup", {}).get("orientation", "mirror")
        blocked = obstacle_squares(rules, self.width, self.height, orientation) | absent_squares(board)
        assert "wrap" not in rules, "the oracle reads rectangles without wrap"

        def next_to(piece: dict[str, Any]) -> list[tuple[int, ...]]:
            vectors, symmetry = control_geometry(piece)
            steps: set[tuple[int, int]] = set()
            for vector in vectors:
                steps |= expand_symmetry(vector, symmetry)
            per_square: list[tuple[int, ...]] = []
            for square in range(self.width * self.height):
                file, rank = square % self.width, square // self.width
                near = []
                for dx, dy in steps:
                    f, r = file + dx, rank + dy
                    if 0 <= f < self.width and 0 <= r < self.height and r * self.width + f not in blocked:
                        near.append(r * self.width + f)
                per_square.append(tuple(sorted(near)))
            return per_square

        self.majority: dict[int, list[tuple[int, ...]]] = {}
        for piece_type, piece in enumerate(spec["pieces"]):
            control = piece.get("control", "stands")
            if control == "stands":
                continue
            self.majority[piece_type] = next_to(piece)
        # TC6: what a hire and a bribe cost, by type, and what *next to* means for them.
        purse = rules.get("purse") if isinstance(rules.get("purse"), dict) else None
        nobodys = {
            placement["piece"] for placement in spec.get("setup", {}).get("placements", [])
            if placement.get("player") == "nobody" and placement.get("squares")
        }
        self.hire: dict[int, int] = {}
        self.bribe: dict[int, int] = {}
        self.near: dict[int, list[tuple[int, ...]]] = {}
        for piece_type, piece in enumerate(spec["pieces"]):
            price = piece.get("price", 0)
            if purse is None or not price:
                continue
            if piece["id"] in nobodys:
                self.hire[piece_type] = price
            if purse.get("bribe") is True:
                self.bribe[piece_type] = 2 * price
            if piece_type in self.hire or piece_type in self.bribe:
                self.near[piece_type] = next_to(piece)
        self.checkmate = rules.get("win", "checkmate") == "checkmate"
        # Q1077: the types whose held pieces take royals, and the game without the key, whose
        # check is the rest of the attack (the boards of the two read alike: same types).
        self.takers = {
            piece_type for piece_type, piece in enumerate(spec["pieces"])
            if piece.get("takes_royal") is True and piece_type in self.majority
        }
        self.plain = engine
        if self.takers:
            without = copy.deepcopy(spec)
            for piece in without["pieces"]:
                piece.pop("takes_royal", None)
            self.plain = Engine(without)

    # ---------------------------------------------------------------- the byte

    def owner(self, value: int) -> int:
        return (value - 1) // (self.n_facings * self.n_types)

    def type_of(self, value: int) -> int:
        return (value - 1) % self.n_types

    def byte(self, player: int, piece_type: int) -> int:
        return (player * self.n_facings) * self.n_types + piece_type + 1

    # ---------------------------------------------------------------- the rule

    def holder(self, state: GameState, square: int) -> int:
        """0, 1, or -1: who may move the neutral on ``square`` (-1 for a tie or no neutral)."""
        value = state.board[square]
        if not value or self.owner(value) != NOBODY_BLOCK:
            return -1
        near = self.majority.get(self.type_of(value))
        if near is None:
            return -1
        count = [0, 0]
        for next_square in near[square]:
            occupant = state.board[next_square]
            if occupant and self.owner(occupant) < 2:
                count[self.owner(occupant)] += 1
        if count[0] == count[1]:
            return -1
        return 0 if count[0] > count[1] else 1

    def _twin_targets(self, state: GameState, square: int, player: int) -> list[int]:
        """Where the neutral on ``square`` could land as ``player``'s piece of its type at
        facing 0 (the frame-0 twin), by the plain game's ordinary generator."""
        board = bytearray(state.board)
        board[square] = self.byte(player, self.type_of(state.board[square]))
        twin = dataclasses.replace(state, board=bytes(board), side_to_move=player)
        return [
            move_to(move) for move in self.plain.pseudo_legal_moves(twin)
            if not move_foreign(move) and move_from(move) == square and move_type(move) == MOVE_NORMAL
        ]

    def held_moves(self, state: GameState) -> list[int]:
        """The side to move's held neutrals' pseudo-legal moves, from the frame-0 twin."""
        player = state.side_to_move
        found: list[int] = []
        for square, value in enumerate(state.board):
            if not value or self.holder(state, square) != player:
                continue
            takes_royal = self.type_of(value) in self.takers
            for target in self._twin_targets(state, square, player):
                occupant = state.board[target]
                if occupant and (
                    self.owner(occupant) != 1 - player
                    or (self.type_of(occupant) in self.royal and not takes_royal)
                ):
                    continue
                found.append(encode_move(square, target, MOVE_NORMAL, 0) | MOVE_FOREIGN)
        return sorted(set(found))

    def attacked(self, state: GameState, player: int) -> bool:
        """Whether ``player``'s royal is attacked: by the plain game's reading, or (Q1077) by a
        neutral of a type that takes royals, held by the other player, whose twin lands on it."""
        if self.plain.in_check(state, player):
            return True
        if not self.takers:
            return False
        royals = {
            square for square, value in enumerate(state.board)
            if value and self.owner(value) == player and self.type_of(value) in self.royal
        }
        for square, value in enumerate(state.board):
            if (
                value and self.owner(value) == NOBODY_BLOCK
                and self.type_of(value) in self.takers
                and self.holder(state, square) == 1 - player
                and royals.intersection(self._twin_targets(state, square, 1 - player))
            ):
                return True
        return False

    def cost(self, state: GameState, square: int) -> int:
        """TC6: what buying the piece on ``square`` costs the side to move, 0 where it cannot:
        its price from nobody (a hire), twice it from the opponent (a bribe)."""
        value = state.board[square]
        if not value:
            return 0
        owner, piece_type = self.owner(value), self.type_of(value)
        player = state.side_to_move
        if owner == NOBODY_BLOCK:
            return self.hire.get(piece_type, 0)
        if owner == 1 - player:
            return self.bribe.get(piece_type, 0)
        return 0

    def purchases(self, state: GameState) -> list[int]:
        """TC6: the side to move's hires and bribes: every affordable piece for hire or bribe
        standing next to one of the mover's pieces."""
        player = state.side_to_move
        found: list[int] = []
        for square, value in enumerate(state.board):
            cost = self.cost(state, square)
            if not cost or cost > state.purse[player]:
                continue
            near = self.near[self.type_of(value)][square]
            if any(
                state.board[n] and self.owner(state.board[n]) == player for n in near
            ):
                found.append(encode_move(square, square, MOVE_HIRE, 0))
        return found

    def legal_moves(self, state: GameState) -> list[int]:
        """The whole legal list, sorted, as the oracle reads it."""
        own = [
            m for m in self.plain.pseudo_legal_moves(state)
            if not move_foreign(m) and move_type(m) != MOVE_HIRE
        ]
        candidates = own + self.held_moves(state) + self.purchases(state)
        if not self.checkmate:
            return sorted(candidates)
        player = state.side_to_move
        return sorted(
            m for m in candidates
            if not self.attacked(self.plain.apply(state, m), player)
            and not self._castles_through_a_held_attack(state, m)
        )

    def _castles_through_a_held_attack(self, state: GameState, move: int) -> bool:
        """Q1077: a castling king may not start on, pass or land on a square a held neutral
        that takes royals attacks. The plain game's generator already asks its own attackers
        of those squares; this asks the held ones, the king stood on each square in turn (the
        oracle's reading of castling along a rank, which is how the games it holds castle)."""
        if not self.takers or move_type(move) != MOVE_CASTLE:
            return False
        player = state.side_to_move
        origin, target = move_from(move), move_to(move)
        step = 1 if target > origin else -1
        for square in range(origin, target + step, step):
            board = bytearray(state.board)
            board[origin] = 0
            board[square] = state.board[origin]
            if self.attacked(dataclasses.replace(state, board=bytes(board)), player):
                return True
        return False
