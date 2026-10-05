"""Board and movement visualisation.

The visualiser reads finished game records and never drives the engine. That keeps it
out of the simulation path entirely -- a rendering bug cannot corrupt a tournament,
and the renderer stays honest, since it can only show what was actually recorded.
"""

from chess_variant_sandbox.viz.html import render_game, render_position

__all__ = ["render_game", "render_position"]
