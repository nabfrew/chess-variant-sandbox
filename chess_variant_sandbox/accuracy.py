"""How much of their winning chances each side gave away, move by move, in plain words.

The visitor's half of Q701 (`docs/reference/win-probability.md`). It reads one finished
game's positions (``winprob.read_game``) through a fitted map (``winprob.WinFit``) and
says, for every move, what the side to move's chances were with the best move the
computer found, what they were after the move played, and how much was lost. A side's
accuracy is the mean over its moves that had a choice of ``100 * exp(-loss / SCALE)``.

It never changes anything: no score, rating, verdict or player reads it. Its words are
held to a list (``FORBIDDEN``) so the evaluator's units and search sizes stay out of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.winprob import CALIBRATED, TOO_FEW, GameReading, WinFit

# Giving away this much of the chances (0.10 = ten points) scores 37 out of 100.
SCALE = 0.10

# A loss under each bound gets its word; at the last bound and over, the last word.
WORDS: tuple[tuple[float, str], ...] = (
    (0.02, "best or as good"),
    (0.05, "good"),
    (0.10, "a slip"),
    (0.20, "a mistake"),
)
WORST = "a blunder"
ONLY_MOVE = "the only move"

SIDES = ("first player", "second player")

# Words a visitor's report never says: the evaluator's units, the search's size, and
# chess's own vocabulary for this (rule 5: nothing hardcodes chess).
FORBIDDEN = (
    "centipawn",
    "pawn",
    "node",
    "evaluator",
    "evaluation",
    "eval",
    "depth",
    "inaccuracy",
    "white",
    "black",
    "logistic",
)


def word_for(loss: float) -> str:
    for bound, word in WORDS:
        if loss < bound:
            return word
    return WORST


def move_accuracy(loss: float) -> float:
    return 100.0 * math.exp(-max(loss, 0.0) / SCALE)


@dataclass(frozen=True)
class Judgement:
    """One move: the chances before and after, and what that cost."""

    ply: int
    mover: int
    before: float
    after: float
    forced: bool
    played: str = ""
    best: str = ""

    @property
    def loss(self) -> float:
        return 0.0 if self.forced else max(0.0, self.before - self.after)

    @property
    def word(self) -> str:
        return ONLY_MOVE if self.forced else word_for(self.loss)

    @property
    def accuracy(self) -> float | None:
        return None if self.forced else move_accuracy(self.loss)


@dataclass(frozen=True)
class Report:
    """A finished game, judged. ``fit`` travels with it: no chance without its basis."""

    moves: tuple[Judgement, ...]
    winner: int | None
    fit: WinFit

    def accuracy(self, side: int) -> float | None:
        """A side's mean move accuracy over its moves that had a choice, or ``None``."""
        scores = [m.accuracy for m in self.moves if m.mover == side and m.accuracy is not None]
        return sum(scores) / len(scores) if scores else None

    def count(self, side: int, word: str) -> int:
        return sum(1 for m in self.moves if m.mover == side and m.word == word)

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "accuracy": [self.accuracy(0), self.accuracy(1)],
            "moves": [
                {
                    "ply": m.ply,
                    "side": m.mover,
                    "played": m.played,
                    "best": m.best,
                    "before": m.before,
                    "after": m.after,
                    "loss": m.loss,
                    "word": m.word,
                    "accuracy": m.accuracy,
                }
                for m in self.moves
            ],
            "fit": self.fit.as_dict(),
        }


def judge(
    reading: GameReading, fit: WinFit, texts: Sequence[tuple[str, str]] | None = None
) -> Report:
    """Turn a game's readings into a report through ``fit``.

    ``texts`` is (played, best) per position, as the engine names the moves; the report
    is complete without it.
    """
    moves = []
    for i, position in enumerate(reading.positions):
        played, best = texts[i] if texts is not None else ("", "")
        moves.append(
            Judgement(
                ply=position.ply,
                mover=position.mover,
                before=fit.chances(position.best),
                after=fit.chances(position.played),
                forced=position.forced,
                played=played,
                best=best,
            )
        )
    return Report(moves=tuple(moves), winner=reading.winner, fit=fit)


def move_texts(engine: Any, record: GameRecord, reading: GameReading) -> list[tuple[str, str]]:
    """The engine's own name for the move played and the best move, per position read."""
    wanted = {p.ply: p for p in reading.positions}
    out: list[tuple[str, str]] = []
    state = engine.initial_state()
    for ply, move in enumerate(record.moves):
        position = wanted.get(ply)
        if position is not None:
            out.append(
                (engine.describe_move(state, move), engine.describe_move(state, position.best_move))
            )
        state = engine.apply(state, move)
    return out


def _percent(p: float) -> str:
    return f"{round(100 * p)}%"


def _side_line(report: Report, side: int) -> str:
    accuracy = report.accuracy(side)
    if accuracy is None:
        return f"The {SIDES[side]} never had a choice to make."
    counts = []
    for word, label in ((WORST, "blunder"), ("a mistake", "mistake"), ("a slip", "slip")):
        n = report.count(side, word)
        counts.append(f"{n} {label}{'' if n == 1 else 's'}")
    return f"The {SIDES[side]}: accuracy {accuracy:.0f} out of 100 ({', '.join(counts)})."


def render(report: Report, name: str = "") -> str:
    """The report as a visitor reads it."""
    lines: list[str] = []
    title = f"How the game went{f': {name}' if name else ''}"
    lines.append(title)
    lines.append("")
    if report.winner is None:
        lines.append("The game was drawn.")
    else:
        lines.append(f"The {SIDES[report.winner]} won.")
    verdict = report.fit.calibration.verdict
    if verdict == TOO_FEW:
        lines.append(
            "Too few games of this kind have been read to check these chances: treat them "
            "as a rough guide only."
        )
    elif verdict != CALIBRATED:
        lines.append(
            "On games kept aside to check them, these chances were off by more than we "
            "allow: treat them as a rough guide."
        )
    lines.append(
        "Winning chances are what the computer, thinking briefly, gave the player about to "
        "move, with a draw counted as half. A move's loss is how far the move played fell "
        "short of the best move the computer found."
    )
    lines.append("")
    lines.append(_side_line(report, 0))
    lines.append(_side_line(report, 1))
    lines.append("")
    for m in report.moves:
        played = f" {m.played}" if m.played else ""
        head = f"Move {m.ply + 1}, {SIDES[m.mover]}{played}:"
        if m.forced:
            lines.append(f"{head} the only move.")
            continue
        text = (
            f"{head} {m.word}; winning chances {_percent(m.before)} with the best move, "
            f"{_percent(m.after)} after this one"
        )
        if m.loss >= 0.005:
            text += f", {round(100 * m.loss)} points lost"
            if m.best and m.best != m.played:
                text += f" (the best was {m.best})"
        lines.append(text + ".")
    lines.append("")
    basis = report.fit.basis
    lines.append(
        f"These chances come from {basis.fit_games} games of this kind and were checked on "
        f"{basis.held_games} more ({report.fit.calibration.verdict})."
    )
    return "\n".join(lines)
