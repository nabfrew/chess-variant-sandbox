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
    # What each player has been paid for taking a wild piece (schema 9), or (TC4) a piece
    # of a type with a ``bounty``. Compared at the clocks under ``scoring.count: "points"``,
    # and ``(0.0, 0.0)`` for every ruleset with neither.
    #
    # It needs no Zobrist term, for the reason ``hands`` needs none and a permanent
    # colouring needs none: a bounty is awarded only when a piece is taken, which is
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
    # Schema 22 (Q843c, fold portals): the square each fold end stands on, in end order.
    # A fold stands on no square of ``board`` -- a piece may stand in it -- so where it is
    # is state of its own, and it **is** in the Zobrist key. ``()`` for every ruleset
    # without folds; a hand-built position that leaves it ``()`` has its folds where they
    # start.
    folds: tuple[int, ...] = ()
    # Schema 26 (Q918a, simultaneous moves): who holds the initiative, -1 for every game
    # whose tie is not the initiative; the void run, ``(player 0's square, player 1's
    # square)`` of a pair whose one void round was the last round, ``(-1, -1)`` for none;
    # and the barred pairs, ascending. All three are state the board does not show, and
    # all three **are** in the Zobrist key (`docs/reference/rounds.md` §2.7, §5.3).
    initiative: int = -1
    void_run: tuple[int, int] = (-1, -1)
    bars: tuple[tuple[int, int], ...] = ()
    # Schema 29 (Q983a, ``pieces[].rest``, `docs/plan-rest.md` §4.2): each player's resting
    # pieces, ``((square, turns left), ...)`` by square, player 0's first. A piece that made
    # its player's move may not make it again while it rests (unless nothing else can, the
    # lift); ``turns left`` counts that player's own turns, alternating turns or rounds.
    # ``((), ())`` for every game without a rest. In the Zobrist key, since the same board
    # with a different piece resting has different legal moves. (Schema 27's
    # ``rules.simultaneous.rest`` kept ``(square, square)`` here; schema 29 retired it.)
    rest: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = ((), ())
    # Generation 36 (N1, ``rules.npcs[].respawn``, `docs/archive/plan-npc.md` §4.1): one clock per
    # respawning half (an entry that comes back, then its mirrored half), 0 idle, else the
    # phases until a taken piece's return is tried. ``()`` for every game without the rule,
    # and read as every clock idle. In the Zobrist key: the same board with a different
    # clock has a different future.
    respawn: tuple[int, ...] = ()
    # Generation 38 (N5, ``rules.npcs[].turns_back``, `docs/archive/plan-npc.md` §4.7): the squares,
    # ascending, of the wild pieces walking their line back; a piece on a line and not listed
    # walks it forward. ``()`` for every game without the rule. In the Zobrist key: the same
    # board with a piece facing the other way has a different future.
    heading_back: tuple[int, ...] = ()
    # TC4 (`docs/plan-turncoat.md` §5.2): each player's coins, 0 to ``PURSE_MAX`` (99),
    # ``(0, 0)`` for every game without a purse (every game until TC5's ``rules.purse``; a
    # test hook compiles one). A bounty paid under the rule also fills the payee's purse,
    # capped. In the Zobrist key, unlike ``points``: repetition alone would not need it (a
    # purse changes only beside a material change), but a transposition table does, since
    # one board can be reached holding different coins.
    purse: tuple[int, int] = (0, 0)

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


# The core's phase-log kinds (`game::PHASE_STEP`, `PHASE_RETURN`, `PHASE_TURN`,
# `PHASE_POUNCE`, `PHASE_COPY`), by code.
PHASE_KINDS = ("step", "return", "turn", "pounce", "copy")


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
    # What the entry is (generation 36, N1): ``"step"``, a wild piece's step along its route,
    # or ``"return"``, a taken wild piece coming back onto ``target`` (``origin`` is the same
    # square: it came from nowhere on the board); or ``"turn"`` (generation 38, N5), a
    # turning-back piece that found its way blocked and turned round: a step the other way, or
    # ``origin == target`` where it waits, turned; or ``"pounce"`` (generation 39, N3b), a
    # pouncer moving from ``origin`` onto the piece of the mover's it took on ``target``; or
    # ``"copy"`` (generation 41, N3d), a splitting pouncer on ``origin`` growing a copy of
    # itself on ``target``, eating the attacker there (the parent stays where it is).
    kind: str = "step"

    @classmethod
    def from_core(cls, step: tuple[int, int, int, int]) -> PhaseStep:
        """One entry of the core's phase log, ``(from, to, eaten, kind)``."""
        origin, target, eaten, kind = step
        return cls(origin, target, eaten, PHASE_KINDS[kind])

    def as_dict(self) -> dict[str, Any]:
        """``{from, to, eaten}``, with ``kind`` only on an entry that is not a step, so the
        page reads every game before the rule as it always did."""
        out: dict[str, Any] = {"from": self.origin, "to": self.target, "eaten": self.eaten}
        if self.kind != "step":
            out["kind"] = self.kind
        return out


@dataclass(frozen=True, slots=True)
class SpyPhase:
    """A spy game's phase at a position (schema 27, Q935a; `docs/plan-spy.md` §1.3), as the
    core reports it (`Engine.phase`). Not part of a position: a function of who has a spy
    and whose turn it is, both already in it.

    ``name`` is ``alternating``, ``reply``, ``announced`` or ``blind``; ``announcer`` is the
    player who chooses first in an announced round, the one without a spy, and -1 in every
    other phase.
    """

    name: str
    announcer: int

    @property
    def is_round(self) -> bool:
        """Whether this step is a round (both players move) rather than one move."""
        return self.name in ("announced", "blind")


@dataclass(frozen=True, slots=True)
class Effects:
    """The effects of lines of sight on one board (schema 30, Q999a; `docs/archive/plan-passive.md`
    §3.4), as the core reports them (`Engine.effects`). Not part of a position: an effect is a
    function of the board.

    ``bits[square]`` is the effect bits on the piece standing there, for its owner
    (``compile.SIGHT_FREEZE`` 1, ``SIGHT_DISARM`` 2, ``SIGHT_SHIELD`` 4), 0 on an empty
    square. ``frozen``, ``disarmed`` and ``shielded`` count the side to move's moves the
    effects removed for each reason (a move barred two ways counts once, under the first).
    ``suppressed`` is whether the side to move's royal stands under an attack the effects
    suppressed: a frozen or disarmed attacker, or a shielded royal. Q1007c
    (`docs/archive/plan-schema-31.md` §3.7): ``bits`` also carry ``SIGHT_COMMAND`` 8 (the piece's
    opponent may move it) and ``SIGHT_COMMAND_OWN`` 16 (and make it take its own side), and
    ``commanded`` counts the side to move's legal commanded moves, 0 without a command.
    """

    bits: tuple[int, ...]
    frozen: int
    disarmed: int
    shielded: int
    suppressed: bool
    commanded: int = 0
    # Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): each square's bits from the wild pieces'
    # sights alone, on a player's piece; empty in a game without wild pieces.
    wild: tuple[int, ...] = ()

    @property
    def binds(self) -> bool:
        """Whether the effects removed any move of the side to move."""
        return self.frozen + self.disarmed + self.shielded > 0


@dataclass(frozen=True, slots=True)
class RestTurn:
    """What the rest does to one player's choice on one board (schema 29, Q983b;
    `docs/plan-rest.md` §5.6, §7), as the core reports it (`Engine.rest_turn`). Not part of a
    position: it is a question about one.

    ``binds``: the player's legal moves differ from those on the same board with nothing
    resting, so the rest took away at least one move. ``lifted``: they do not, yet a resting
    piece's move is among them, which the rule allows only when the rest would have left no
    move at all (§3.7). Never both; both false in a game without a rest.
    """

    binds: bool
    lifted: bool
