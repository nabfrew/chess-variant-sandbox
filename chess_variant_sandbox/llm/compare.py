"""Asking the model which of two games is better, rather than how good one is.

An absolute 0-1 score from a model is noisy and poorly calibrated: it drifts with
phrasing, it clusters, and nothing anchors it between calls. Blending one into the index
at a fixed weight made it the *de facto* ranker inside the human band, which was an
accident of scale compression rather than a decision.

Pairwise preference is a different question and a much better-behaved one. "Which of
these two would you rather play?" needs no scale, and the comparisons fit into a
Bradley-Terry ordering with the same maths the agent ladder already uses.

Three properties this module is built around:

**Both orders, every time.** Position bias in pairwise LLM judging is well documented.
A preference that flips when the two candidates are swapped is not a weak preference, it
is a non-result, and it is recorded as a tie. Agreement across the two orders is the only
thing that counts as a preference here.

**No numbers reach it.** Same rule as the proposer, for the same reason. A judge shown
the component scores restates them, and an ordering that restates the index and is then
used to break the index's ties is the index breaking its own ties.

**It judges the rules, and it is told so.** The games shown were played by programs, some
deliberately weak. A blunder is evidence about the player.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.llm.client import JUDGE_MODEL, Claude, cached_system
from chess_variant_sandbox.llm.judge import _render_game, _sample_games
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.tournament.record import GameRecord

# Two games each is enough to show what a ruleset does and short enough that the pair
# fits in one call without the transcripts crowding out the rules.
GAMES_SHOWN = 2

COMPARISON_BRIEF = """\
You are an experienced player and designer of abstract strategy games. You are shown \
two newly invented two-player games: the rules of each, and a couple of complete games \
played under them by computer opponents of differing strength.

Say which of the two you would rather play, and why. What matters:

- Which offers the more interesting decisions -- real trade-offs rather than an obvious \
  best move.
- Which set of rules hangs together better, as opposed to having a piece or exception \
  bolted on.
- Which is more likely to hide a degeneracy: a dominant opening, a piece that \
  trivialises the game, a position type that always locks up.
- Which you would rather explain to someone, and play a second time.

Judge the *rules*, not the play. The games were played by programs, some of them \
deliberately weak; a blunder is evidence about the player, not about the design.

If the two are genuinely close, say so -- "neither" is a real answer and a far more \
useful one than a coin flip. Do not try to be even-handed across many comparisons: each \
one is on its own.
"""

COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "better": {
            "type": "string",
            "enum": ["first", "second", "neither"],
            "description": "Which game you would rather play, or 'neither' if genuinely close.",
        },
        "because": {
            "type": "string",
            "description": "Two or three sentences on what decides it.",
        },
    },
    "required": ["better", "because"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Comparison:
    """One pair, judged in both orders."""

    first: str
    second: str
    # The hash of the preferred ruleset, or ``None`` for a tie -- which includes the
    # case where the two orders disagreed.
    winner: str | None
    # Whether the two orders said the same thing. A comparison that flipped is stored
    # as a tie *and* flagged, because the rate of flipping is the honest measure of how
    # much any of these preferences are worth.
    agreed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            "winner": self.winner,
            "agreed": self.agreed,
            "reasons": list(self.reasons),
        }

    @property
    def score_for_first(self) -> float:
        """Bradley-Terry score for the first name: 1, 0.5 or 0."""
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == self.first else 0.0


def compare_rulesets(
    left: dict[str, Any],
    right: dict[str, Any],
    left_records: list[GameRecord],
    right_records: list[GameRecord],
    client: Claude | None = None,
    games_shown: int = GAMES_SHOWN,
) -> Comparison:
    """Ask which of two rulesets is the better game, in both orders.

    Two calls, not one. The cost is real -- this is the expensive half of a generation --
    and it buys the only thing that makes the answer usable: a preference that survives
    swapping the two candidates round.
    """
    client = client or Claude(model=JUDGE_MODEL, effort="medium")
    left_hash, right_hash = spec_hash(left), spec_hash(right)

    forward = _ask(client, left, right, left_records, right_records, games_shown)
    backward = _ask(client, right, left, right_records, left_records, games_shown)

    # ``forward`` names a position; translate both answers into a ruleset before
    # comparing them, which is the whole point of asking twice.
    preferred = {
        "first": left_hash,
        "second": right_hash,
        "neither": None,
    }[forward[0]]
    preferred_back = {
        "first": right_hash,
        "second": left_hash,
        "neither": None,
    }[backward[0]]

    agreed = preferred == preferred_back
    return Comparison(
        first=left_hash,
        second=right_hash,
        winner=preferred if agreed else None,
        agreed=agreed,
        reasons=(forward[1], backward[1]),
    )


def _ask(
    client: Claude,
    first: dict[str, Any],
    second: dict[str, Any],
    first_records: list[GameRecord],
    second_records: list[GameRecord],
    games_shown: int,
) -> tuple[str, str]:
    system = cached_system(client.model, COMPARISON_BRIEF)
    user = "\n\n".join(
        [
            "GAME ONE",
            describe_spec(first),
            _transcripts(first, first_records, games_shown),
            "GAME TWO",
            describe_spec(second),
            _transcripts(second, second_records, games_shown),
            "Which would you rather play?",
        ]
    )
    data = client.structured(
        system=system, user=user, schema=COMPARISON_SCHEMA, max_tokens=2_000
    )
    return str(data.get("better", "neither")), str(data.get("because", ""))


def _transcripts(spec: dict[str, Any], records: list[GameRecord], count: int) -> str:
    if not records:
        return "(no games available)"
    engine = Engine(spec)
    return "\n\n".join(
        _render_game(engine, record) for record in _sample_games(records, count)
    )
