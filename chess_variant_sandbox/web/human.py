"""What happened when a person actually played: the only ground truth in the project.

Everything else here compares the ladder against itself. `skill_gradient` asks whether a
bigger budget beats a smaller one, `ladder_monotonicity` asks whether the ordering holds,
`heuristic_agreement` asks whether a different objective beats the built-in one -- and all
three are answered by the same agents that are being measured. The one measurement that
is not circular is a person sitting down and losing, and until now the project had the
surface for it (`chess-variant-sandbox serve`) and no record of a single result.

Three decisions, and the third is the one that keeps this honest.

**Recording is automatic, not a button.** A result you have to ask to keep is a result you
keep when you win. The store is written whenever a session with exactly one human seat
reaches a terminal position, and the person is never asked.

**A started game is written down too.** The bias automatic recording cannot remove is
the tab closed halfway through a loss. So a session is noted when it starts and updated
when it ends, and the difference -- games begun and never finished -- is reported beside
the results rather than silently dropped. It is the number a reader should look at first.

**Nothing is fitted to it.** The anchor places *one* player against ratings that were
computed without it: the opponents' Elo comes from the picker's own table, is held fixed,
and a single number is solved for. No component reads this file, no weights version
scores it, and no curve is anchored on it -- `tests/test_web.py` asserts the first of
those directly. One player's results are a calibration anchor and a sanity check, never a
target; the mutant corpus already caught this project fitting an instrument to a label
that turned out to be wrong.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.tournament.ratings import PRIOR_GAMES, expected_score

HUMAN = "human"

# What the site writes in `reason` when a leaderboard match is resigned -- by leaving,
# or by the idle sweep. Named here because the anchor's reading counts it and the local
# app, which has no modes, never writes it.
RESIGNED = "resigned"

# The bracket the single-parameter fit is bisected over. Wide enough that no real result
# reaches an end, which is why "unbounded" below is decided by the *results* -- a clean
# sweep is a bound rather than a rating -- and not by where the arithmetic stopped.
UNBOUNDED = 4000.0


@dataclass(frozen=True)
class HumanGame:
    """One game a person played, finished or not.

    ``result`` is the *human's* score -- 1.0, 0.5, 0.0 -- and is ``None`` while the game
    is still in progress or was abandoned. ``opponent`` is the full agent string, so a
    node budget and a rule of thumb are distinguishable without a second field.

    ``moves`` is the game as played, the engine's move integers from the opening, so a
    person's win can be replayed through the engine and studied (Q502a). A row written
    before it was kept, and every row of a game still running, has none. Nothing that
    scores or rates reads it.
    """

    session: str
    ruleset_hash: str
    ruleset_name: str
    opponent: str
    human_seat: int
    seed: int
    started: float
    plies: int = 0
    result: float | None = None
    winner: int | None = None
    reason: str = ""
    finished: float | None = None
    moves: tuple[int, ...] = ()
    # Which net the opponent really played, `net:<12 hex>`, where it is a learned player
    # (Q709, `Session.played_by`); "" for every other opponent and every older row.
    played: str = ""

    @property
    def complete(self) -> bool:
        return self.result is not None

    def as_dict(self) -> dict[str, Any]:
        found = asdict(self)
        found["moves"] = list(self.moves)
        return found

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanGame:
        fields = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        if "moves" in fields:
            fields["moves"] = tuple(int(move) for move in fields["moves"])
        return cls(**fields)


class HumanRecord:
    """The store, append-only, with the last line for a session winning.

    Append-only because a session is written twice -- once when it starts and once when
    it ends -- and rewriting a file to update one row is how a crash mid-write loses
    every result before it. Folding by session id at read time costs nothing at this
    size and cannot corrupt anything.
    """

    def __init__(self, runs_dir: str | Path) -> None:
        self.path = Path(runs_dir) / "human" / "games.jsonl"
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- writing

    def note_start(
        self,
        session: str,
        ruleset_hash: str,
        ruleset_name: str,
        opponent: str,
        human_seat: int,
        seed: int,
        played: str = "",
    ) -> None:
        self._append(
            HumanGame(
                session=session,
                ruleset_hash=ruleset_hash,
                ruleset_name=ruleset_name,
                opponent=opponent,
                human_seat=human_seat,
                seed=seed,
                started=time.time(),
                played=played,
            )
        )

    def note_result(
        self,
        session: str,
        ruleset_hash: str,
        ruleset_name: str,
        opponent: str,
        human_seat: int,
        seed: int,
        plies: int,
        winner: int | None,
        reason: str,
        moves: tuple[int, ...] | list[int] = (),
        played: str = "",
    ) -> None:
        """Write the verdict and the moves that reached it. Idempotent: a page that asks
        twice records once, and the moves are the first finish's.

        The page polls, and a finished position answers the same way every time, so this
        is called again on every refresh of an ended game. Dropping the repeat here
        rather than deduplicating at read time keeps the file a log of what happened
        rather than a log of how often it was looked at.
        """
        existing = {game.session: game for game in self.games()}.get(session)
        if existing is not None and existing.complete:
            return
        result = 0.5 if winner is None else (1.0 if winner == human_seat else 0.0)
        self._append(
            HumanGame(
                session=session,
                ruleset_hash=ruleset_hash,
                ruleset_name=ruleset_name,
                opponent=opponent,
                human_seat=human_seat,
                seed=seed,
                started=existing.started if existing else time.time(),
                plies=plies,
                result=result,
                winner=winner,
                reason=reason,
                finished=time.time(),
                moves=tuple(int(move) for move in moves),
                played=played or (existing.played if existing else ""),
            )
        )

    def _append(self, game: HumanGame) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(game.as_dict(), separators=(",", ":")) + "\n")

    # ---------------------------------------------------------------- reading

    def games(self, ruleset_hash: str = "") -> list[HumanGame]:
        """Every session, latest row per session. A truncated final line is skipped."""
        latest: dict[str, HumanGame] = {}
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    game = HumanGame.from_dict(data)
                except (TypeError, ValueError):
                    continue
                latest[game.session] = game
        found = sorted(latest.values(), key=lambda g: g.started)
        return [g for g in found if not ruleset_hash or g.ruleset_hash == ruleset_hash]

    def summary(self, ruleset_hash: str, ratings: dict[str, float] | None) -> dict[str, Any]:
        """One ruleset's human record, and where it puts the person on the ladder."""
        return summarise(self.games(ruleset_hash), ruleset_hash, ratings)


def summarise(
    games: list[HumanGame], ruleset_hash: str, ratings: dict[str, float] | None
) -> dict[str, Any]:
    """One ruleset's human record, out of whatever holds the rows.

    ``ratings`` is the picker's own table for this ruleset, opponent to Elo. Absent or
    empty means the games are still worth reporting and the anchor is not computable,
    which is a different statement from "the person is unrated".

    A function rather than a method because the site keeps the same rows in `matches`
    and reads them back through `site/matches.py`. What a human record *means* is one
    reading, wherever the rows are kept.
    """
    finished = [g for g in games if g.complete]

    per_opponent: dict[str, dict[str, Any]] = {}
    for game in finished:
        entry = per_opponent.setdefault(
            game.opponent,
            {"agent": game.opponent, "games": 0, "score": 0.0, "rating": None},
        )
        entry["games"] += 1
        entry["score"] += game.result or 0.0
    for entry in per_opponent.values():
        entry["score"] = round(entry["score"] / entry["games"], 4)
        if ratings is not None:
            entry["rating"] = ratings.get(entry["agent"])

    anchor = (
        anchor_rating([(g.opponent, g.result or 0.0) for g in finished], ratings or {})
        if ratings
        else None
    )
    return {
        "hash": ruleset_hash,
        "name": games[0].ruleset_name if games else "",
        "started": len(games),
        "finished": len(finished),
        # Begun and never finished. The bias automatic recording cannot remove, and
        # the first number a reader should look at: a person who abandons the games
        # they are losing has a rating this file cannot bound from below.
        "abandoned": len(games) - len(finished),
        # The same number where a mode makes it answerable. A leaderboard match left
        # mid-game is resigned rather than dropped, so on the site the hole above closes
        # and this is what it became.
        "resigned": sum(1 for g in finished if g.reason == RESIGNED),
        "score": (
            round(sum(g.result or 0.0 for g in finished) / len(finished), 4)
            if finished
            else None
        ),
        "byOpponent": sorted(per_opponent.values(), key=lambda e: -(e["rating"] or -9e9)),
        "anchor": anchor,
    }


def anchor_rating(
    results: list[tuple[str, float]],
    ratings: dict[str, float],
    prior_games: float = PRIOR_GAMES,
) -> dict[str, Any] | None:
    """Where one player sits on a ladder that is *not* re-fitted around them.

    A single-parameter Bradley-Terry fit: every opponent keeps the rating it already had
    and one number is solved for, so a person's results cannot move the ladder they are
    being placed on. That is 29.3 in code rather than in a comment -- `rate()` would
    re-fit everybody, and then the anchor and the thing it anchors would be the same
    measurement twice.

    Monotone in the rating, so bisection is exact enough and cannot fail to converge.
    The prior is the same handful of virtual drawn games `ratings.py` uses, and for the
    same reason: a player who has won everything has no upper bound at all, and the
    honest report of that is the flag below rather than a large number.
    """
    played = [(agent, score) for agent, score in results if agent in ratings]
    if not played:
        return None

    mean = sum(ratings.values()) / len(ratings)
    scored = sum(score for _, score in played)
    observed = scored + prior_games / 2.0

    def expected(rating: float) -> float:
        total = prior_games * expected_score(rating, mean)
        for agent, _score in played:
            total += expected_score(rating, ratings[agent])
        return total

    low, high = -UNBOUNDED, UNBOUNDED
    for _ in range(200):
        middle = (low + high) / 2.0
        if expected(middle) < observed:
            low = middle
        else:
            high = middle
    rating = (low + high) / 2.0

    ordered = sorted(ratings.items(), key=lambda item: -item[1])
    # A person who beat everything, or lost to everything, is off the end of the ladder
    # rather than at a number: their results are a *bound*, and the prior is the only
    # thing making the arithmetic finite. Reported as a flag so the interface can draw an
    # arrow rather than a rating nothing in the data supports -- and with the direction,
    # because an arrow needs one and it cannot be read off the mean score: that average
    # covers games against opponents this ladder has no rating for.
    bounded = 0.0 < scored < float(len(played))
    return {
        "rating": round(rating, 1),
        "games": len(played),
        "opponents": len(ratings),
        "bounded": bounded,
        "swept": "" if bounded else ("up" if scored else "down"),
        # What the answer is *for*: how far the top rated rung sits above or below the
        # person. A roster whose top rung is below a mediocre player on one ruleset and
        # above one on another is a roster whose rungs are not the same rung, which is
        # what stage 27 says from the other side.
        "top": ordered[0][0] if ordered else "",
        "top_rating": round(ordered[0][1], 1) if ordered else 0.0,
        "top_above_human": round(ordered[0][1] - rating, 1) if ordered else 0.0,
        "expected_against_top": (
            round(expected_score(rating, ordered[0][1]), 4) if ordered else 0.0
        ),
    }


def picker_ratings(table: dict[str, Any] | None) -> dict[str, float]:
    """Opponent to Elo, out of a stored picker table. Empty if there is not one.

    The picker's table and never the rated ladder's, for a reason
    `web/opponents.py` states at length: the two are different quantities and are never
    mixed in one column. The picker's is the right one here because it is the table that
    rates *what a person was actually offered*, rules of thumb and the 200,000-node tier
    included, and the rated ladder's contains none of those by construction.
    """
    if not table:
        return {}
    out = {}
    for standing in table.get("standings", []):
        name = standing.get("name")
        rating = standing.get("rating")
        if isinstance(name, str) and isinstance(rating, (int, float)):
            out[name] = float(rating)
    return out
