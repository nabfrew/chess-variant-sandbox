"""Move encoding.

A move is a plain ``int``. Not a dataclass, not a tuple: move lists are the hottest
allocation in the engine, and they cross the boundary into agents and game records
where a compact, JSON-friendly, hashable value is worth far more than attribute
access. The layout is fixed here so the Rust core can reproduce it exactly.

    bits  0-7   origin square index
    bits  8-15  destination square index
    bits 16-19  move type (see the MOVE_* constants)
    bits 20-24  promotion piece type index, meaningful only for MOVE_PROMOTION,
                the dropped piece type for MOVE_DROP (and the bought one for
                MOVE_BUY, TC5), and the new facing for
                MOVE_ROTATE
    bits 25-31  landing index (schema 12): which entry of the compiled landing list a
                struck piece is placed on under ``position: "choose"``, zero otherwise;
                under MOVE_PORTAL (schema 19) the internal type of the portal end the
                piece went in by, and bits 20-24 then hold the promoted type plus one;
                under MOVE_FOLD (schema 22) the fold end that moves (its index among
                the portal ends), and on an ordinary move in a fold game the fold end
                it went through plus one (``move_fold``), zero for a direct move;
                bit 30 (``MOVE_FOREIGN``; Q1007c's ``MOVE_COMMANDED``, renamed by TC1)
                marks a foreign move, the side to move moving a piece that is not its
                own -- today a commanded one, the opponent's (every other use of these
                bits is refused beside a sight)

Square indices are ``rank * width + file``, so the layout supports boards up to
16x16 (256 squares), matching ``spec.schema.MAX_BOARD_DIM``.
"""

from __future__ import annotations

MOVE_NORMAL = 0
MOVE_EN_PASSANT = 1
MOVE_CASTLE = 2
MOVE_PROMOTION = 3
# A drop names no origin square: the piece comes from its owner's hand, and the piece
# type travels in the promotion field. Origin is set equal to the destination so that
# every consumer which clears "the origin has moved" stays correct without a special
# case -- a dropped piece has indeed never moved, but neither has it kept any right.
MOVE_DROP = 4
# A rotation names no destination: the piece turns where it stands, and the *new facing*
# travels in the promotion field. Origin is set equal to the destination for the same
# reason a drop does it -- every consumer that clears "the origin has moved" stays
# correct without a special case, and a piece that has turned has indeed moved.
#
# This is the second move whose origin and target being equal means something, and the
# first whose meaning is not "there was no origin". Anything that tells a drop apart by
# `origin == target` alone is now wrong, which is why `move_type` is the test everywhere
# and this note is here rather than in a commit message.
MOVE_ROTATE = 5
# Schema 13: a sweep whose path held a piece. The path is recovered from the compiled
# sweep table by origin and target. A sweep over an empty path is ``MOVE_NORMAL``.
MOVE_SWEEP = 6
# Schema 19: a transit through a portal. The target is the exit square, where the piece
# ends up; bits 25-31 hold the internal type of the portal end it went in by
# (``move_portal``), and the promotion field holds the promoted type *plus one*, 0 for
# none, because the kind field cannot say both "portal" and "promotion".
MOVE_PORTAL = 7
# Schema 22: a fold's own move. Origin and target are the fold's squares and nothing on the
# board byte changes, since a fold stands on no square of it (``GameState.folds``); bits
# 25-31 hold the end (its index among the portal ends, ``move_fold``).
MOVE_FOLD = 8
# Generation 42 (N4b, `docs/archive/plan-npc.md` §4.6): the duck's move, the second half of every turn
# under ``rules.duck``. The promotion field says how (``DUCK_STEP``, ``DUCK_PLACE``,
# ``DUCK_PASS``); a placement and a pass have origin equal to target, as a drop does. Takes
# nothing, on any reading of the board.
MOVE_DUCK = 9
# TC5 (`docs/plan-turncoat.md` §5.4): a piece bought from the market (``rules.purse``) onto
# an empty square of the buyer's region. Origin equals target, as a drop's; the promotion
# field holds the bought type, and its price leaves the buyer's purse. Takes nothing.
MOVE_BUY = 10
# TC6 (`docs/plan-turncoat.md` §5.5): a piece hired (nobody's, at its price) or bribed (the
# opponent's, at twice it) where it stands, next to one of the buyer's pieces. Origin equals
# target, the bought piece's square; the promotion field is 0. It becomes the buyer's and the
# coins leave the buyer's purse. Takes nothing.
MOVE_HIRE = 11
DUCK_STEP = 0
DUCK_PLACE = 1
DUCK_PASS = 2

# Q1007c (`docs/archive/plan-schema-31.md` §3.7), renamed by TC1 (`docs/plan-turncoat.md`
# §3.3): a flag, not a kind. A foreign move is a ``MOVE_NORMAL`` (or ``MOVE_ROTATE``) made by
# the side to move with a piece that is not its own -- today one of the opponent's that a
# ``command`` sight binds; the core sets the bit (`foreign.rs::push_foreign`), and
# ``describe_move`` writes the move with a ``~`` before the piece.
MOVE_FOREIGN = 1 << 30

MOVE_TYPE_NAMES = {
    MOVE_NORMAL: "normal",
    MOVE_EN_PASSANT: "en_passant",
    MOVE_CASTLE: "castle",
    MOVE_PROMOTION: "promotion",
    MOVE_DROP: "drop",
    MOVE_ROTATE: "rotate",
    MOVE_SWEEP: "sweep",
    MOVE_PORTAL: "portal",
    MOVE_FOLD: "fold",
    MOVE_DUCK: "duck",
    MOVE_BUY: "buy",
    MOVE_HIRE: "hire",
}


def encode_move(
    origin: int,
    target: int,
    move_type: int = MOVE_NORMAL,
    promotion: int = 0,
    landing: int = 0,
) -> int:
    """Pack a move into a single integer."""
    return origin | (target << 8) | (move_type << 16) | (promotion << 20) | (landing << 25)


def move_from(move: int) -> int:
    return move & 0xFF


def move_to(move: int) -> int:
    return (move >> 8) & 0xFF


def move_type(move: int) -> int:
    return (move >> 16) & 0xF


def move_promotion(move: int) -> int:
    return (move >> 20) & 0x1F


def move_landing(move: int) -> int:
    """The landing index of a strike under ``position: "choose"``; zero for any other move."""
    return (move >> 25) & 0x7F


def move_portal(move: int) -> int:
    """The internal type of the portal end a ``MOVE_PORTAL`` went in by (schema 19)."""
    return (move >> 25) & 0x7F


def move_fold(move: int) -> int:
    """Schema 22: bits 25-31 read as a fold end. A ``MOVE_FOLD``'s end; on an ordinary move in
    a fold game, the end it went through plus one, 0 for a direct move."""
    return (move >> 25) & 0x7F


def decode_move(move: int) -> tuple[int, int, int, int]:
    """Unpack a move into ``(origin, target, move_type, promotion)``."""
    return (move & 0xFF, (move >> 8) & 0xFF, (move >> 16) & 0xF, (move >> 20) & 0x1F)


def move_foreign(move: int) -> bool:
    """Whether ``move`` is a foreign move (TC1): the side to move moving a piece that is not
    its own, today a commanded one (Q1007c)."""
    return bool(move & MOVE_FOREIGN)
