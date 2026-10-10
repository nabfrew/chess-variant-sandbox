"""Game state and outcomes.

State is immutable and made of flat, primitive values — ``bytes`` for the board, ints
for everything else. No objects per square, no per-piece records. This was a deliberate
constraint and it paid twice: it keeps a state hashable and safely shareable, and it is
what made the layout translatable to the Rust core at all. A position crosses the
language boundary as these nine fields and nothing else.

Nothing here is on the hot path any more. A search inside the core plays a move and
takes it back rather than building one of these per node; what this type is for is the
positions Python still holds — the ones a test asserts on, the visualiser replays, and
the CLI prints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

PLAYER_ZERO = 0
PLAYER_ONE = 1
DRAW = None

NO_SQUARE = -1


@dataclass(frozen=True, slots=True)
class GameState:
    """A complete, self-contained position.

    ``board`` encodes each square as ``0`` for empty, otherwise
    ``player * n_types + type_index + 1``.

    ``unmoved`` is a bitmask over squares whose current occupant has never moved.
    One mask serves both castling rights and any ``first_move`` move condition, which
    is why neither needs its own field: a move clears the bit at both its origin and
    its destination, so rights are lost exactly when a piece leaves or is displaced.

    ``repetition`` holds position hashes since the last irreversible move. It lives in
    the state rather than alongside it so that ``Engine.terminal(state)`` stays a pure
    function of its argument — a bare state can be handed to anything and reach the same
    verdict as whatever produced it. Inside the core the same window is a pair of
    indices into one history, which is the same idea without the copy per ply.

    ``key`` is the Zobrist hash of the position. It is carried on the state rather
    than recomputed on demand because search asks for it at every node, and a full
    recompute is a loop over every square. ``Engine.apply`` maintains it with a
    handful of XORs; ``Engine._compute_hash`` is the slow definition it must agree
    with, and a test holds them to that.
    """

    board: bytes
    side_to_move: int
    unmoved: int
    ep_target: int = NO_SQUARE
    ep_victim: int = NO_SQUARE
    halfmove: int = 0
    ply: int = 0
    key: int = 0
    repetition: tuple[int, ...] = ()
    # Checks delivered so far by each player, for rulesets that end on a count of them.
    # Carried on the state rather than tallied by the game loop for the same reason the
    # repetition history is: ``terminal`` has to stay a pure function of its argument.
    # Maintained only when ``check_limit`` is set, so it costs nothing otherwise.
    checks: tuple[int, int] = (0, 0)
    # Pieces held off the board, per player, counted by piece type. Empty for every
    # ruleset without drops. It needs no place in the Zobrist key: a capture or a drop
    # is an irreversible move, which clears the repetition window, so two positions
    # ever compared for repetition always share the same hands.
    hands: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    # Who owns each square: 0 nobody, 1 the first player, 2 the second. Empty for every
    # ruleset without a territory rule.
    #
    # A *permanent* colouring needs no Zobrist term, for the same reason ``hands`` needs
    # none and by the same argument: laying a colour down is irreversible, so it clears
    # the repetition window, so two positions the repetition rule ever compares always
    # share the same colouring. A repainting one does need a term, and pays for it.
    colour: bytes = b""
    # Moves left in the current turn, and how many turns have been taken. 1 and a
    # running count for every ruleset without a turn rule, where the side flips on every
    # move -- which is what made the counter invisible until a rule wanted otherwise.
    #
    # This one *does* need a Zobrist term, unlike ``hands`` and unlike a permanent
    # colouring: two positions with the same pieces and different moves-remaining are
    # genuinely different positions, one side being about to move again, and nothing
    # about the difference is irreversible.
    remaining: int = 1
    turns_taken: int = 0
    # What each player has been paid for taking a wild piece (schema 9). Compared at the
    # clocks under ``scoring.count: "points"``, and ``(0.0, 0.0)`` for every ruleset
    # without ``rules.npcs``.
    #
    # It needs no Zobrist term, for the reason ``hands`` needs none and a permanent
    # colouring needs none: a bounty is awarded only when a wild piece is taken, which is
    # a material change, which is irreversible, which clears the repetition window -- so
    # two positions the repetition rule ever compares always share the tally.
    points: tuple[float, float] = (0.0, 0.0)
    # Schema 18 (Q786, ``reach.on: "leave"``): the side whose last move took the named
    # piece off one of its goal squares, or -1. Nothing on the board says what the last
    # move was, so ``terminal`` could not read it there and stay a pure function of the
    # state; -1 for every ruleset without the rule.
    #
    # It needs no Zobrist term: a position where it is set is terminal, so no search and
    # no repetition rule ever compares it with another position.
    goal_left_by: int = -1
    # Schema 20 (Q807, ``no_straight_back``): ``(standing, left)`` when the last move carried
    # a barring portal end from ``left`` to ``standing``, so it may not go straight back on
    # this move; ``(-1, -1)`` for none, and for every ruleset without the rule. Nothing on
    # the board says what the last move was, so it is state, and it **is** in the Zobrist
    # key, as the en passant file is: the same board with and without a bar has different
    # legal moves.
    portal_bar: tuple[int, int] = (-1, -1)

    def with_(self, **changes: Any) -> GameState:
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class Outcome:
    """How a game ended. ``winner`` is ``None`` for a draw."""

    winner: int | None
    reason: str

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    def score_for(self, player: int) -> float:
        """Result from ``player``'s point of view: 1.0 win, 0.5 draw, 0.0 loss."""
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == player else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"winner": self.winner, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class PhaseStep:
    """One wild piece's step in the phase the rules run after a move (schema 9).

    Not part of a position. A ply with wild pieces in it makes two board changes -- the
    mover's move and then the phase -- and what the phase did is a fact about the
    transition, which the end position no longer carries.

    ``eaten`` is the board byte the destination square held at the instant the wild piece
    wrote over it: zero for a quiet step, the piece a promotion had already become where
    the phase ate it on the square it promoted onto, and the rook on its new square where
    a castling had just moved it. A caller diffing two positions has to guess at those
    last two.
    """

    origin: int
    target: int
    eaten: int

    def as_dict(self) -> dict[str, Any]:
        return {"from": self.origin, "to": self.target, "eaten": self.eaten}
