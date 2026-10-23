"""The rule interpreter.

Downstream code (agents, tournament, metrics) imports only from this package's public
surface, never from its internals. That boundary is what made the Rust core a drop-in
replacement, and what let the Python interpreter it replaced be deleted without a single
caller noticing — see CLAUDE.md.
"""

from chess_variant_sandbox.engine.compile import CompiledSpec, compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    decode_move,
    encode_move,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import (
    DRAW,
    Effects,
    GameState,
    Outcome,
    PhaseStep,
    RestTurn,
    SpyPhase,
)

__all__ = [
    "DRAW",
    "CompiledSpec",
    "Effects",
    "Engine",
    "GameState",
    "Outcome",
    "PhaseStep",
    "RestTurn",
    "SpyPhase",
    "compile_spec",
    "decode_move",
    "encode_move",
    "move_from",
    "move_promotion",
    "move_to",
    "move_type",
]
