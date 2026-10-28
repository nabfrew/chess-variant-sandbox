"""Positions and moves, in the words Fairy-Stockfish uses for them.

Two directions, and they are deliberately not symmetrical.

**A position goes out as a FEN.** That is the whole of what the opponent is told, so
everything the rules depend on has to be in it -- the board and which of its pieces
count as promoted, the side, castling rights,
the en passant window, the two clocks, a pocket where there are drops, a check count
where they are counted and a `*` for a wall. A field this gets wrong is a field the
opponent plays without.

**A move comes back as text and is looked up, never parsed into one.** `uci_of` is
applied to every move our engine says is legal and the answer is matched against what
the engine sent; an unmatched string is an error rather than a move. Building a move
from the text instead would mean this module deciding what is legal, which is the one
thing `CLAUDE.md` says Python may not do -- and it would turn a translation bug into an
illegal move played in a rated game rather than into a refusal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    MOVE_DROP,
    MOVE_PROMOTION,
    decode_move,
)
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.fairy.variant import board_letters
from chess_variant_sandbox.spec.schema import format_square

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chess_variant_sandbox.engine.compile import CompiledSpec
    from chess_variant_sandbox.fairy.variant import Variant


class MoveMismatch(Exception):
    """The opponent named a move our engine does not have.

    Always a translation defect rather than a bad engine: the variant it was given and
    the ruleset we are playing have come apart somewhere. Raised rather than absorbed,
    because the alternative is a game that quietly stops being the game being rated.
    """


def fen(
    compiled: CompiledSpec,
    state: GameState,
    chars: list[str] | tuple[str, ...],
    king: int | None,
    *,
    drops: bool = False,
    check_limit: int = 0,
    castling: bool = False,
) -> str:
    """``state`` as a Fairy-Stockfish FEN under the translated variant."""
    board = _board_field(compiled, state, chars)
    if drops:
        board += f"[{_pocket_field(compiled, state, chars)}]"
    fields = [
        board,
        "w" if their_side(compiled, state) == 0 else "b",
        _castling_field(compiled, state, king) if castling else "-",
        format_square(state.ep_target % compiled.width, state.ep_target // compiled.width) if state.ep_target != NO_SQUARE else "-",
    ]
    if check_limit:
        # Remaining checks, first player first, which is the order the board field is
        # written in. A count that has already passed the limit cannot occur in a
        # position anyone is asked about -- the game ended on the ply that reached it --
        # but it is clamped anyway, because a negative field is a parse error there and
        # an exception thrown while choosing a move is worse than a rounded number.
        remaining = [max(check_limit - state.checks[player], 0) for player in (0, 1)]
        fields.append(f"{remaining[0]}+{remaining[1]}")
    fields.append(str(state.halfmove))
    # Generation 42 (N4c): a duck game's turn is two of our plies and one move of theirs.
    fields.append(str(state.ply // (4 if compiled.duck is not None else 2) + 1))
    return " ".join(fields)


def their_side(compiled: CompiledSpec, state: GameState) -> int:
    """The side to move as their engine sees ``state`` (generation 42, N4c).

    Ours, except in a duck game ended on a piece move, before the duck's half our engine
    never asks for: their move is both halves, so there the turn is over and the other side
    is to move. The result check reads both engines' verdicts from this side.
    """
    if compiled.duck is not None and state.remaining == 1:
        return 1 - state.side_to_move
    return state.side_to_move


def _board_field(
    compiled: CompiledSpec, state: GameState, chars: list[str] | tuple[str, ...]
) -> str:
    """The board, top rank first, as Fairy-Stockfish reads it.

    Our square 0 is a1 and a FEN starts at the far rank, so the ranks are walked
    backwards. Runs of empty squares are written as a number that may be two digits --
    a twelve-file board has a row that is simply ``12``, which the parser handles and
    a hand-rolled reader would not.

    A piece whose type demotes when captured is written promoted, ``+P`` or ``H~``
    (`variant.board_letters`, Q406). Without the mark the engine counts it as never
    promoted and its search puts the wrong piece in the captor's hand.

    A wall (`rules.obstacles`) is written ``*``, which is a square nothing stands on there
    (Q703). The translator refuses the walls that would mean something else
    (`variant._check_walls`), and it refuses a hexagon, whose absent cells the compiler
    folds into the same set.
    """
    letters = board_letters(compiled, chars)
    walls = set(compiled.obstacles)
    # Generation 42 (N4c, `docs/archive/plan-npc.md` §4.6): the duck is a wall there (`wallingRule =
    # duck`), written `*` where it stands; off the board it is not written at all.
    duck = compiled.duck
    if duck is not None:
        value = compiled.encode_piece(2, duck.piece_type)
        walls.update(sq for sq, held in enumerate(state.board) if held == value)
    rows: list[str] = []
    for rank in range(compiled.height - 1, -1, -1):
        row = ""
        empty = 0
        for file in range(compiled.width):
            square = rank * compiled.width + file
            occupant = compiled.piece_at(state.board, square)
            if square in walls:
                if empty:
                    row += str(empty)
                    empty = 0
                row += "*"
                continue
            if occupant is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            player, piece = occupant
            row += letters[piece].upper() if player == 0 else letters[piece].lower()
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def _pocket_field(
    compiled: CompiledSpec, state: GameState, chars: list[str] | tuple[str, ...]
) -> str:
    """What each side holds off the board, first player's in upper case."""
    held = ""
    for player in (0, 1):
        for piece, count in enumerate(state.hands[player]):
            letter = chars[piece].upper() if player == 0 else chars[piece].lower()
            held += letter * count
    return held


def _castling_field(compiled: CompiledSpec, state: GameState, king: int | None) -> str:
    """Which castlings are still available, in the usual four letters.

    Derived from the `unmoved` mask rather than tracked, because that mask *is* the
    right, and the derivation is the same one `generate_castling` makes: a side may
    still castle towards a rook when neither that rook nor its king has moved. A rook
    that was captured on its own square is gone from the board and so is its right,
    which is why the occupant is checked and not only the bit.

    It reads the *castling rule*, never the king slot. Atomic is why: there the royal
    piece is pseudo-royal rather than a king, so a field keyed on the king slot came
    back empty and the translated game silently could not castle at all. Whether a
    piece is royal and whether it castles are two questions.
    """
    rule = compiled.castling
    if rule is None:
        return "-"
    text = ""
    for player, letters in ((0, "KQ"), (1, "kq")):
        kings = _unmoved_squares(compiled, state, player, rule.king_type)
        if not kings:
            continue
        king_square = kings[0]
        rank = king_square // compiled.width
        king_file = king_square % compiled.width
        rooks = [
            square % compiled.width
            for square in _unmoved_squares(compiled, state, player, rule.rook_type)
            if square // compiled.width == rank
        ]
        if any(file > king_file for file in rooks):
            text += letters[0]
        if any(file < king_file for file in rooks):
            text += letters[1]
    return text or "-"


def _unmoved_squares(
    compiled: CompiledSpec, state: GameState, player: int, piece: int
) -> list[int]:
    wanted = compiled.encode_piece(player, piece)
    return [
        square
        for square, value in enumerate(state.board)
        if value == wanted and state.unmoved >> square & 1
    ]


# ------------------------------------------------------------------------------ moves


def uci_of(
    compiled: CompiledSpec,
    state: GameState,
    move: int,
    chars: list[str] | tuple[str, ...],
    promotion_suffix: str = "piece",
) -> str:
    """One of our moves, spelled the way the opponent spells it.

    Castling is written king-to-destination rather than king-to-rook, which is what
    Fairy-Stockfish expects while `chess960` is off -- and it is off, because the
    translated variant states the two rook files outright.

    ``promotion_suffix`` is the one place the two spellings of a promotion have to be
    told apart. Where the ruleset offers a choice of piece the move names it, and where
    each promoting piece has exactly one target there is nothing to name and the move
    carries a bare ``+``. `variant._promotion_suffix` decides which, from the same fact
    that decided how the promotion was written into the config.
    """
    origin, target, kind, promoted = decode_move(move)
    width = compiled.width
    if kind == MOVE_DROP:
        face = promoted % compiled.n_types
        drops = compiled.drops
        slot = face if drops is None else drops.hand_slot(face)
        # Schema 18's `either_face` (Q785): a piece dropped as its other face is spent
        # from its base's slot, and `dropPromoted` writes that drop as the base marked
        # promoted, `+P@c3`.
        mark = "+" if slot != face else ""
        return f"{mark}{chars[slot].upper()}@{format_square(target % width, target // width)}"
    text = f"{format_square(origin % width, origin // width)}{format_square(target % width, target // width)}"
    if kind == MOVE_PROMOTION:
        if promotion_suffix == "plus":
            # A turn back to the piece's `promoted_from` is `pieceDemotion`'s move there,
            # written with a minus (Kyoto shogi's flip, `variant._flip_pairs`).
            mover = compiled.type_of(state.board[origin])
            back = compiled.demotions[mover] == promoted and promoted != mover
            text += "-" if back else "+"
        else:
            text += chars[promoted].lower()
    return text


def turn_word(
    compiled: CompiledSpec,
    state: GameState,
    move: int,
    duck_move: int | None,
    chars: list[str] | tuple[str, ...],
    promotion_suffix: str = "piece",
) -> str:
    """A duck game's turn as their one word (generation 42, N4c, `docs/archive/plan-npc.md` §4.6).

    Their move is both halves: our piece move, then ``,``, its target and the square our duck
    move puts the duck on (``e2e4,e4d5``; castling ``e1g1,g1b1``). ``duck_move`` None is a
    piece move that ended our game, which has no duck half here: its word is the prefix
    alone, which every word of theirs for that piece move starts with (`verify` matches it
    so). Spelled from our two moves, never parsed from theirs.
    """
    head = uci_of(compiled, state, move, chars, promotion_suffix)
    if duck_move is None:
        return head
    width = compiled.width
    _, target, _, _ = decode_move(move)
    _, duck_target, _, how = decode_move(duck_move)
    lands = target if how == DUCK_PASS else duck_target
    return f"{head},{format_square(target % width, target // width)}{format_square(lands % width, lands // width)}"


def move_of(
    compiled: CompiledSpec,
    state: GameState,
    legal: list[int],
    text: str,
    chars: list[str] | tuple[str, ...],
    promotion_suffix: str = "piece",
) -> int:
    """The one legal move the opponent's ``text`` names.

    Raises `MoveMismatch` when nothing matches, and *also* when more than one does.
    Two of our moves sharing a spelling means the opponent could not have told them
    apart, so picking either would be picking for it.
    """
    matches = [
        move
        for move in legal
        if uci_of(compiled, state, move, chars, promotion_suffix) == text
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise MoveMismatch(
            f"the opponent played {text!r}, which is not one of "
            f"{sorted(uci_of(compiled, state, move, chars, promotion_suffix) for move in legal)}"
        )
    raise MoveMismatch(
        f"{text!r} names {len(matches)} different moves of ours; the translated variant "
        "cannot tell them apart"
    )


def variant_fen(variant: Variant, compiled: CompiledSpec, state: GameState) -> str:
    """``state`` as a FEN under ``variant``, with every field the variant needs.

    The one spelling the agent, `verify` and the result check all hand over, so the
    three cannot describe one position three ways.
    """
    return fen(
        compiled,
        state,
        variant.chars,
        variant.king,
        drops=variant.drops,
        check_limit=_limit(compiled, variant),
        castling=variant.castling,
    )


def _limit(compiled: CompiledSpec, variant: Variant) -> int:
    return compiled.check_limit if variant.check_counting else 0

