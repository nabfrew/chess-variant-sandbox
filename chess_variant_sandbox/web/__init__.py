"""The local application: browse the library, play a game, start a design run.

A game that has never been played by a person is a number rather than a game. This
package is the part of the project a person can use directly, and it is deliberately
thin: it holds positions and shows measurements, and every question about the rules goes
to the engine, exactly as CLAUDE.md requires of anything above the boundary.
"""

from chess_variant_sandbox.web.library import Game, detail, discover
from chess_variant_sandbox.web.play import Session, Sessions
from chess_variant_sandbox.web.server import App, PortInUse, Server, serve

__all__ = [
    "App",
    "Game",
    "PortInUse",
    "Server",
    "Session",
    "Sessions",
    "detail",
    "discover",
    "serve",
]
