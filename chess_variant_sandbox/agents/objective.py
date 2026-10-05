"""Which objective an agent maximises: the built-in one, or a heuristic's.

One function, and it exists to keep a promise rather than to save a line. An empty
weight vector must reach ``agents/evaluation.Evaluator`` and nothing else -- not a
feature evaluator configured to behave like it, not a vector that happens to reproduce
it. Every committed number in this project was produced by that class, and float
addition is not associative, so "equivalent arithmetic" is not the same claim as "the
same arithmetic". The built-in agents therefore keep their own code path, byte for byte,
and the heuristic path is a branch beside it rather than a generalisation of it.

``features.builtin_vector`` does express the built-in objective in the vocabulary, and
that vector anchors index 0 of every heuristic set. It reproduces the built-in's
*decisions*, not its bits -- dividing and multiplying by the same float is not the
identity -- which is exactly why it is an anchor and not a replacement.
"""

from __future__ import annotations

from typing import Protocol

from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, Evaluator
from chess_variant_sandbox.agents.features import FeatureEvaluator
from chess_variant_sandbox.engine import Engine, GameState


class Objective(Protocol):
    """What a search needs from whatever it is maximising.

    ``values`` is read for capture ordering, which is not the objective: ordering only
    decides which equally-legal move is examined first, so it keeps the spec-derived
    piece values whatever a heuristic says about them. Declared read-only, because
    ``Evaluator`` is a frozen dataclass and a settable attribute would exclude it --
    nothing assigns to it, so the weaker requirement is the accurate one.
    """

    @property
    def values(self) -> tuple[float, ...]: ...

    def evaluate(
        self, engine: Engine, state: GameState, player: int, move_count: int | None = ...
    ) -> float: ...


def objective_for(
    engine: Engine,
    mobility: float,
    goal: float = GOAL_PIECES,
    weights: tuple[float, ...] = (),
    values: tuple[float, ...] = (),
    immune: float = IMMUNE_PIECES,
    net: bytes = b"",
) -> Objective:
    """The built-in evaluator, or a heuristic's, depending on whether one was named.

    ``goal`` is stage 17A's weight on a non-capture ending and reaches only the built-in
    evaluator. A heuristic has no use for it: its vector already says what the ending is
    worth, term by term, which is the whole point of expressing an objective as weights.

    ``values`` is a piece-value override and travels with the weights because it is half
    of one objective -- an opponent's digest covers both. It reaches the built-in
    evaluator never: that evaluator *is* the spec-derived valuation, and a settable one
    would be a second thing calling itself the built-in objective.
    """
    if net:
        # Q502e: a net is the whole objective (`nets.NetEvaluator`, the oracle for
        # `Objective::Net`); capture ordering keeps the derived table, as the core's does.
        if weights or values:
            raise ValueError("a net was given beside a weight vector or piece values")
        from chess_variant_sandbox.agents.nets import NetEvaluator

        derived = Evaluator.for_engine(engine, mobility_weight=mobility).values
        return NetEvaluator(engine, net, derived)
    if not weights:
        if values:
            raise ValueError(
                "piece values were given with no weight vector, which would name the "
                "built-in objective and then quietly not be used by it; an override is "
                "half of a heuristic and needs the other half"
            )
        return Evaluator.for_engine(
            engine, mobility_weight=mobility, goal_pieces=goal, immune_pieces=immune
        )
    return FeatureEvaluator.for_engine(engine, tuple(weights), tuple(values))
