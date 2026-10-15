"""Players, all behind one protocol.

Importing this package registers the built-in agents. Anything that resolves an agent
by name must import from here rather than from a submodule, or the registry will be
half-populated and a tournament will fail on an agent name that is perfectly valid.
"""

from chess_variant_sandbox.agents.base import (
    NATIVE_VALUES,
    NATIVE_WEIGHTS,
    Agent,
    AgentConfig,
    NativeSpec,
    SearchStats,
    available_agents,
    build_agent,
    native_spec,
    parse_agent,
    register,
    register_native,
)
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.fairy_stockfish import FairyStockfishAgent
from chess_variant_sandbox.agents.matrix import MatrixAgent
from chess_variant_sandbox.agents.mcts import MCTSAgent
from chess_variant_sandbox.agents.minimax import MinimaxAgent
from chess_variant_sandbox.agents.simple import GreedyAgent, RandomAgent

__all__ = [
    "NATIVE_VALUES",
    "NATIVE_WEIGHTS",
    "Agent",
    "AgentConfig",
    "Evaluator",
    "FairyStockfishAgent",
    "GreedyAgent",
    "MCTSAgent",
    "MatrixAgent",
    "MinimaxAgent",
    "NativeSpec",
    "RandomAgent",
    "SearchStats",
    "available_agents",
    "build_agent",
    "native_spec",
    "parse_agent",
    "register",
    "register_native",
]
