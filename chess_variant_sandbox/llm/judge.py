"""The written verdict: the only thing here that says *why* a game is worth playing.

This module used to produce a 0-1 score as well, blended into the index at
``llm_weight``. Two things happened to that number. ``v4`` set the weight to zero, and
the design loop replaced the blend with a pairwise ordering, because an absolute score
from a model is noisy and poorly calibrated while "which of these two is better" is
neither. What was left was a scoring path nothing called -- which is how a second
ranker returns quietly -- so the score is gone and the prose is what remains.

The prose was always the more valuable half. Every other output of this system is a
measurement: an index, an interval, a rating, a rate. None of them can tell a person
browsing the library what a game *is* and whether it is any good. That is what a
verdict is for, and it is why it belongs beside ``spec/describe.py``'s rendering in
the library rather than anywhere near ``IndexResult.total``.

The judge is still kept ignorant of every number the objective side produced. That
independence was the point while the verdict was being blended, and it is still the
point now: a judge shown the component scores would mostly restate them, and a
restatement is not a second opinion.

Sampling favours games between the strongest players available, and covers both a
decisive game and a drawn one where the corpus has both. A judge shown only wins
concludes the game is decisive; shown only draws, that it is dead. Neither is a read
on the rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.llm.client import JUDGE_MODEL, Claude, cached_system
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.tournament.record import GameRecord

MAX_MOVES_SHOWN = 60
DEFAULT_SAMPLE = 4

JUDGE_BRIEF = """\
You are an experienced player and designer of abstract strategy games. You are shown \
the rules of a newly invented two-player game and a few complete games played by \
computer opponents of differing strength.

Judge whether the game is worth playing. What you are looking for:

- Are the decisions interesting? Does a position offer real trade-offs, or is the \
  right move usually obvious?
- Do the rules hang together, or is there a piece or exception that feels bolted on?
- Is there a visible degeneracy -- a dominant opening, a piece that trivialises the \
  game, a position type that always locks up?
- Could you explain this game to someone in a minute, and would they want a second game?

Judge the *rules*, not the play. The games you are shown were played by programs, some \
of them deliberately weak; a blunder in a game is evidence about the player, not about \
the design. What a game does tell you is what the rules make possible.

Write for someone deciding whether to play this. They can already see the rules and \
the measurements, and need neither repeated back. Do not rate the game on a scale, and \
do not hedge into one -- no "a solid seven", no "middling". Say what the game is, say \
whether it works, and be specific about what you think would go wrong. Most inventions \
are not good, and saying so plainly is more useful than being encouraging.
"""

# No numeric field, deliberately. A number here has exactly one place to go -- back into
# the ranking, alongside an ordering derived from the same opinion -- and a test holds
# this schema to prose so it cannot grow one back.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "description": "Two or three sentences: what this game is, and whether it works.",
        },
        "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "weaknesses": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "suspected_degeneracy": {
            "type": "string",
            "description": "The specific way this game most likely breaks, or an empty string.",
        },
        "would_play_again": {"type": "boolean"},
    },
    "required": [
        "verdict",
        "strengths",
        "weaknesses",
        "suspected_degeneracy",
        "would_play_again",
    ],
    "additionalProperties": False,
}


@dataclass
class Judgement:
    """What a reader gets: prose, and nothing that can be averaged into a ranking."""

    verdict: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suspected_degeneracy: str = ""
    would_play_again: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def notes(self) -> str:
        parts = [self.verdict]
        if self.suspected_degeneracy:
            parts.append(f"Suspected degeneracy: {self.suspected_degeneracy}")
        return " ".join(parts)


def judge_ruleset(
    spec: dict[str, Any],
    records: list[GameRecord],
    client: Claude | None = None,
    sample: int = DEFAULT_SAMPLE,
) -> Judgement:
    """Read a ruleset and some of its games, and say in prose whether it is any good."""
    client = client or Claude(model=JUDGE_MODEL, effort="medium")
    engine = Engine(spec)

    system = cached_system(
        client.model, "\n\n".join([JUDGE_BRIEF, "The rules:", describe_spec(spec)])
    )
    transcripts = [
        _render_game(engine, record) for record in _sample_games(records, sample)
    ]
    user = (
        "Here are complete games played under these rules.\n\n"
        + "\n\n".join(transcripts)
        + "\n\nJudge the game."
    )

    data = client.structured(system=system, user=user, schema=VERDICT_SCHEMA, max_tokens=4_000)
    return Judgement(
        verdict=str(data["verdict"]),
        strengths=list(data.get("strengths", [])),
        weaknesses=list(data.get("weaknesses", [])),
        suspected_degeneracy=str(data.get("suspected_degeneracy", "")),
        would_play_again=bool(data.get("would_play_again", False)),
    )


def _sample_games(records: list[GameRecord], sample: int) -> list[GameRecord]:
    """Pick games that show the range of what happens, strongest players first."""
    if not records:
        return []

    def strength(record: GameRecord) -> int:
        return sum(0 if agent.startswith("random") else 1 for agent in record.agents)

    ranked = sorted(records, key=lambda r: (-strength(r), r.seed))
    decisive = [r for r in ranked if r.winner is not None]
    drawn = [r for r in ranked if r.winner is None]

    chosen: list[GameRecord] = []
    # Interleave so the sample never becomes all one kind of ending.
    for index in range(sample):
        pool = decisive if index % 2 == 0 else drawn
        fallback = drawn if pool is decisive else decisive
        source = pool or fallback
        while source:
            candidate = source.pop(0)
            if candidate not in chosen:
                chosen.append(candidate)
                break
    return chosen


def _render_game(engine: Engine, record: GameRecord) -> str:
    """A readable transcript. Long games are shown from both ends, not truncated flat."""
    state = engine.initial_state()
    texts: list[str] = []
    for move in record.moves:
        texts.append(engine.describe_move(state, move))
        state = engine.apply(state, move)

    if len(texts) > MAX_MOVES_SHOWN:
        head = texts[: MAX_MOVES_SHOWN // 2]
        tail = texts[-(MAX_MOVES_SHOWN // 2) :]
        shown = head + [f"... {len(texts) - len(head) - len(tail)} moves omitted ..."] + tail
    else:
        shown = texts

    outcome = (
        f"draw by {record.reason}"
        if record.winner is None
        else f"player {record.winner} wins by {record.reason}"
    )
    header = (
        f"Game: {record.agents[0]} (first) vs {record.agents[1]} (second) -- "
        f"{outcome} after {record.plies} plies."
    )
    return header + "\n" + " ".join(shown)
