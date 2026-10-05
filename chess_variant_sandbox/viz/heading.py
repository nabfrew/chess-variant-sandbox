"""Which pieces a board draws pointing, and which it draws upright.

A piece with a facing carries an orientation it can spend a move changing, so where it
points is a fact about the position and the board has to draw it. Every other piece is
drawn exactly as its icon arrives, for both seats -- a bishop standing on its head says
nothing true, and a reader spends the game decoding it.

Both renderers used to turn every one of the second player's icons by
``facing_headings()[1][0]``, which is 180 degrees on every ruleset. ``facing_headings``
answers *which way does this facing point*, so asking it about a piece with no facing
gets the army's answer rather than the piece's. Which pieces may ask is a separate
question, and this module is it.

Ownership is a different matter and lives in ``viz/pieces.py``: a single-colour set
turns the second player's tile 180 because both armies are the same colour, and that is
not a heading.
"""

from __future__ import annotations

from chess_variant_sandbox.engine.compile import CompiledSpec


def directional_types(spec: CompiledSpec) -> tuple[bool, ...]:
    """Per piece type: did the compiler give this piece a facing?

    The whole test. A facing is a direction by construction -- the validator refuses one
    whose turn changes no move -- and it moves during the game rather than following the
    seat, so it is the only thing that turns a drawing. A pawn without one draws upright
    for both armies even though they advance opposite ways, because the icon arrives
    pointing up and up is what it stays.
    """
    return tuple(axes > 1 for axes in spec.facing_axes)
