"""Badges, kept once earned: the row, the writer, and the rule for a current score.

Box E1 of `docs/archive/plan-meta-game.md`, against its §2.9. A
milestone is a live reading over match rows, so it reopens when a ladder or the library
grows. A badge is the opposite: it is the record of an event, written once and never
updated, because three of a badge's inputs do not survive. A vote is replaced when a
person votes again and `votes` keeps no history, a score goes stale when the weights or
the sources move, and the count of available rules grows with each schema version.

**No badge kind is defined here.** The kinds, their thresholds and their sentences live in
`site/badges.py`, which box E3 builds. This module knows how to write a row, how to read
one back, and what a score badge is allowed to quote.

**Nothing commits.** The caller owns the transaction, as `matches`, `rating` and `votes`
do, and that is the point of the writer rather than a convention it happens to follow: a
badge is granted inside the transaction of the event that completed it, so the event and
the badge land together or neither does.

## The rule for a current score

`score_evidence` is §2.9's first answer to *how does a kept badge square with a stale
score showing no number*. A badge that quotes an index is granted only while that game's
measurement reads current; a stale one earns nothing and waits. What the badge then keeps
is a dated quotation -- the figure, its range, the weights version and the whole basis
stamp as the measurement held it -- which enters no ordering and is never compared with a
live score. So a page can carry a kept badge above a score panel that reads *out of date*,
and both statements are true.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

CURRENT = "current"
"""The one `measurements.state` a score badge may be granted on. `load.CURRENT` is the
same string from the writing end; a constant here keeps the reader from importing the
loader for one word."""

_COLUMNS = "kind, tier, holder, game, earned, evidence, rule"


class AwardError(ValueError):
    """An award cannot be written or read as asked.

    A `ValueError` like `votes.VoteError`, so a route answers with the sentence rather
    than with a constraint violation from Postgres.
    """


@dataclass(frozen=True)
class Award:
    """One badge earned, as the row holds it.

    `tier` is the step of a series -- the *n* in *beat easy on n games*, the 80 in *over
    0.80* -- and is null for a badge with no series. `holder` is null when the badge sits
    on a library game, and `game` is null for a badge earned across the library or once
    per person. Those three nulls are part of the key, which is why the unique constraint
    is declared `nulls not distinct`.

    `earned` is when the completing event landed. Left unset, the row takes `now()`; a
    reader granting a badge an older match completed passes that match's date instead.

    `evidence` holds every figure the badge's sentence quotes, as read at that moment, and
    `rule` holds the thresholds in force, so a row stays readable after a constant moves.
    Neither ever holds a handle: a person who leaves reads as *a former player* on an
    award as on a match.
    """

    kind: str
    tier: int | None = None
    holder: int | None = None
    game: str | None = None
    earned: datetime | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    rule: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int | None, int | None, str | None]:
        """What the store is unique on, for a reader deciding what is already held."""
        return (self.kind, self.tier, self.holder, self.game)


@dataclass(frozen=True)
class ScoreEvidence:
    """What a score badge may quote, and the date it is quoted under is the row's.

    Built only from a measurement that reads current. There is no shape of this value
    that means *stale*, because a stale measurement produces no value at all.
    """

    game: str
    headline: float
    ci_low: float | None
    ci_high: float | None
    weights: str
    spec_hash: str
    basis: dict[str, Any]

    @property
    def ranged(self) -> bool:
        """Whether both ends of the range are known.

        A measurement taken before the interval was written down has a headline and no
        ends. A badge that tests the whole range (U1's *under 0.65*, N5's *over the
        median*) cannot be granted on one of those, and this is how it asks.
        """
        return self.ci_low is not None and self.ci_high is not None

    def as_dict(self) -> dict[str, Any]:
        """The `evidence` payload §2.9 lists for a badge that reads the index."""
        return {
            "headline": self.headline,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "weights": self.weights,
            "spec_hash": self.spec_hash,
            "basis": self.basis,
        }


# ------------------------------------------------------------------------- the writer


def grant(connection: Any, awards: Iterable[Award]) -> list[Award]:
    """Write every award not already held, and return the ones that landed.

    `on conflict do nothing` is the whole of the *once* rule, and it is the store's
    answer rather than the caller's: two events completing the same badge in the same
    second cannot both write it, and a reader that recomputes what a person is due on
    every finish writes nothing the second time. A conflict is not an error and produces
    no row, so the returned list is exactly what is new -- which is what a caller that
    wants to say *you have just earned this* should read.

    Nothing commits. The caller's transaction decides whether the event and its badges
    both happened.
    """
    written: list[Award] = []
    for award in awards:
        if not str(award.kind).strip():
            raise AwardError("an award has no kind, so nothing could ever read it back")
        with connection.cursor() as cursor:
            cursor.execute(
                f"insert into awards ({_COLUMNS})"
                f" values (%s, %s, %s, %s, coalesce(%s, now()), %s::jsonb, %s::jsonb)"
                f" on conflict do nothing returning {_COLUMNS}",
                (
                    str(award.kind),
                    None if award.tier is None else int(award.tier),
                    None if award.holder is None else int(award.holder),
                    award.game,
                    award.earned,
                    json.dumps(award.evidence or {}),
                    json.dumps(award.rule or {}),
                ),
            )
            found = cursor.fetchone()
        if found is not None:
            written.append(_to_award(found))
    return written


# ------------------------------------------------------------------------- the readers


def held(
    connection: Any, holder: int | None = None, game: str | None = None
) -> list[Award]:
    """Every award on this person, this game, or both, oldest first.

    At least one of the two is required. An unfiltered read is the whole table, which no
    page wants and which would be a slow way to find out that a filter was forgotten.

    Asking by game returns the badges that sit on it whether or not they have a holder,
    because a library game's badges have none and a designed game's sit on its owner.
    """
    if holder is None and game is None:
        raise AwardError("held needs a holder or a game; it is not a listing of the table")
    clauses: list[str] = []
    values: list[Any] = []
    if holder is not None:
        clauses.append("holder = %s")
        values.append(int(holder))
    if game is not None:
        clauses.append("game = %s")
        values.append(game)
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_COLUMNS} from awards where {' and '.join(clauses)}"
            f" order by earned, kind, tier",
            tuple(values),
        )
        return [_to_award(row) for row in cursor.fetchall()]


def score_evidence(connection: Any, game: str) -> ScoreEvidence | None:
    """What this game's index may be quoted as, or nothing at all.

    The rule that a badge is granted only while the game's measurement reads current.
    Every other state -- stale by any reading, never measured, no row -- answers `None`,
    and a caller cannot get a number out of one by accident, because there is none in the
    value to get.

    A row whose state says current and whose headline is null is treated as no answer.
    The loader writes the two together and neither without the other, so a row that
    disagrees with itself is one the site should not be quoting from.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select state, headline, ci_low, ci_high, basis"
            " from measurements where hash = %s",
            (game,),
        )
        found = cursor.fetchone()
    if found is None:
        return None
    state, headline, ci_low, ci_high, basis = found
    if str(state) != CURRENT or headline is None:
        return None
    stamp = dict(basis) if isinstance(basis, dict) else {}
    return ScoreEvidence(
        game=str(game),
        headline=float(headline),
        ci_low=None if ci_low is None else float(ci_low),
        ci_high=None if ci_high is None else float(ci_high),
        weights=str(stamp.get("weights_version") or ""),
        spec_hash=str(stamp.get("ruleset_hash") or game),
        basis=stamp,
    )


def _to_award(found: Sequence[Any]) -> Award:
    return Award(
        kind=str(found[0]),
        tier=None if found[1] is None else int(found[1]),
        holder=None if found[2] is None else int(found[2]),
        game=None if found[3] is None else str(found[3]),
        earned=found[4],
        evidence=dict(found[5] or {}),
        rule=dict(found[6] or {}),
    )
