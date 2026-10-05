"""Move encoding.

A move is a plain ``int``. Not a dataclass, not a tuple: move lists are the hottest
allocation in the engine, and they cross the boundary into agents and game records
where a compact, JSON-friendly, hashable value is worth far more than attribute
access. The layout is fixed here so the Rust core can reproduce it exactly.

    bits  0-7   origin square index
    bits  8-15  destination square index
    bits 16-19  move type (see the MOVE_* constants)
    bits 20-24  promotion piece type index, meaningful only for MOVE_PROMOTION,
                the dropped piece type for MOVE_DROP, and the new facing for
                MOVE_ROTATE
    bits 25-31  landing index (schema 12): which entry of the compiled landing list a
                struck piece is placed on under ``position: "choose"``, zero otherwise

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

MOVE_TYPE_NAMES = {
    MOVE_NORMAL: "normal",
    MOVE_EN_PASSANT: "en_passant",
    MOVE_CASTLE: "castle",
    MOVE_PROMOTION: "promotion",
    MOVE_DROP: "drop",
    MOVE_ROTATE: "rotate",
    MOVE_SWEEP: "sweep",
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


def decode_move(move: int) -> tuple[int, int, int, int]:
    """Unpack a move into ``(origin, target, move_type, promotion)``."""
    return (move & 0xFF, (move >> 8) & 0xFF, (move >> 16) & 0xF, (move >> 20) & 0x1F)
