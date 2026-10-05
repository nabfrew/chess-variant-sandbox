"""Rule proposal and qualitative judging.

Optional: the engine, tournament and metrics all run without this package's
dependency or credentials. Only the proposal and judging stages need a model.

The one rule that spans both halves: the proposer never sees the index, and the judge
never sees its scores. ``feedback.py`` explains why.

The judge is asked which of two games is better rather than how good one is --
``compare.py`` -- because an absolute score from a model is noisy and poorly calibrated,
and blending one into the headline made it the ranker inside a band the objective index
cannot resolve. ``judge.py`` keeps the absolute form, which is still the right question
when a single ruleset is being read on its own, but the loop no longer scores with it.
"""

from chess_variant_sandbox.llm.client import (
    JUDGE_MODEL,
    PROPOSAL_MODEL,
    Claude,
    LLMUnavailable,
    Usage,
)
from chess_variant_sandbox.llm.compare import Comparison, compare_rulesets
from chess_variant_sandbox.llm.feedback import describe
from chess_variant_sandbox.llm.judge import Judgement, judge_ruleset
from chess_variant_sandbox.llm.propose import Proposal, propose_ruleset

__all__ = [
    "JUDGE_MODEL",
    "PROPOSAL_MODEL",
    "Claude",
    "Comparison",
    "Judgement",
    "LLMUnavailable",
    "Proposal",
    "Usage",
    "compare_rulesets",
    "describe",
    "judge_ruleset",
    "propose_ruleset",
]
