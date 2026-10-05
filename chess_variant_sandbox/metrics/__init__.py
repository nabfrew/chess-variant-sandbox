"""The interestingness index and its components.

Measurement (``components``) is kept separate from judgement (``scoring``) so that a
corpus can be re-scored under new weights without re-simulating a tournament, and so
that a change to the weights is visibly a change to the weights.
"""

from chess_variant_sandbox.metrics.components import Components, compute_components
from chess_variant_sandbox.metrics.index import IndexResult, score_components, score_ruleset
from chess_variant_sandbox.metrics.scoring import (
    DEFAULT_WEIGHTS,
    Curve,
    Gate,
    Weights,
    WeightsError,
    load_weights,
    parse_weights,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "Components",
    "Curve",
    "Gate",
    "IndexResult",
    "Weights",
    "WeightsError",
    "compute_components",
    "load_weights",
    "parse_weights",
    "score_components",
    "score_ruleset",
]
