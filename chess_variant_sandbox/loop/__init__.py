"""The automatic design loop and its supporting machinery.

Cheap gates first, expensive evidence last. Everything tried is archived, failures
included -- a rejected proposal is the most useful thing the next prompt can carry.
"""

from chess_variant_sandbox.loop.archive import Archive, Entry
from chess_variant_sandbox.loop.gates import SmokeResult, smoke_test
from chess_variant_sandbox.loop.grid import AXES, CELLS, Grid, describe_cell, descriptor
from chess_variant_sandbox.loop.operators import OPERATORS, Mutation, Operator, mutate
from chess_variant_sandbox.loop.orchestrator import LoopConfig, evolve
from chess_variant_sandbox.loop.preference import PreferenceLog, PreferenceTable
from chess_variant_sandbox.loop.screen import ScreenResult, screen

__all__ = [
    "AXES",
    "CELLS",
    "OPERATORS",
    "Archive",
    "Entry",
    "Grid",
    "LoopConfig",
    "Mutation",
    "Operator",
    "PreferenceLog",
    "PreferenceTable",
    "ScreenResult",
    "SmokeResult",
    "describe_cell",
    "descriptor",
    "evolve",
    "mutate",
    "screen",
    "smoke_test",
]
