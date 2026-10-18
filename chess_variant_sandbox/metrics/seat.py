"""The seat gate: a game the strongest rungs win from one seat is refused.

``first_player_advantage`` reads 32 self-play games and cannot tell an even seat from a
0.28 lean. A game the top rungs won from the first seat every time passed every ``v10``
gate and scored 0.98 on balance. This is the refusal the user decided on 2026-09-14: read
the games between the strongest rungs, and if one seat's split of them is one a fair seat
produces rarely enough, the ruleset is out, the way a game that never ends is out.

Two readings stand side by side. The top pair's 16 games at ``p <= 0.001`` catch a sweep
and nothing narrower. The top three rungs' 36 games at ``p <= 0.0005`` catch a strong lean
as well, which is what a game like ``gardner`` looks like. Each threshold was set from the
binomial before the corpus it reads was counted, and each refusal carries its own text.

**A library game is read over its four runs pooled, against a tolerated share.** One run's
16 pair games make ``p <= 0.001`` a sweep check by construction, but the same p-value
pooled over four runs refuses a 72% split, and over sixteen a 61% one -- so the built form
refuses more of the library every time the library is measured again. ``share_test``
replaces the even seat in the null with a *share*, ``POOLED_TOLERATED_SHARE``, and the
refused set then converges instead of growing: a game above the share is refused in the
end, a game at or under it is refused at most alpha of the time at any sample size.
``pooled_seat_reading`` is that reading, ``metrics/seeds.py`` takes it once over a game's
runs, and a one-run measurement keeps the built form above.

**A game played in rounds** (schema 26) has no first mover: both players choose on the same
board and are resolved together. The check still runs and still means something, because the
seats still differ: a lean there is an asymmetry of the setup or, under ``tie:
"initiative"``, the edge of holding the token at the start (`docs/archive/plan-simultaneous.md`
§6.3). "The first seat" is seat 0, the player listed first; the alphas and the share are
the same, since the question -- does one seat win more than a fair one would -- is.

**Each different game counts once** (Q953a, ``docs/reference/repeated-games.md`` §6). A
band's sign test, per run and pooled, reads every distinct decided transcript once: players
whose draws are inert replay one line many times, and a line played sixteen times is one
observation of its pairing. The split as played is kept beside it, and the lean reads that.
The cost is pinned in ``docs/bugs.md``: a band that plays one forced line cannot be refused
by this check however real the lean, so only the other gates can refuse such a game.

``docs/reference/metrics-and-calibration-record.md`` (Q235) has the first reading's design.
``docs/reference/seat-fresh-2026-09.md`` has the fresh-games campaign that tested the
second, ``docs/reference/seat-effect-size.md`` is the restated check and the two campaigns
that validated it, and ``docs/archive/bugs-resolved.md`` *A seat-decided game* holds the decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import blake2b
from math import comb, exp, lgamma, log
from typing import Any

from chess_variant_sandbox.metrics.scoring import Gate

SEAT_DECIDED_CHANCE = "seat_decided_chance"
TOP_RUNG_FIRST_SEAT_SCORE = "top_rung_first_seat_score"
TOP_RUNG_SEAT_DETAIL = "top_rung_seat"

TOP_THREE_SEAT_DECIDED_CHANCE = "top_three_seat_decided_chance"
TOP_THREE_FIRST_SEAT_SCORE = "top_three_first_seat_score"
TOP_THREE_SEAT_DETAIL = "top_three_seat"

# Two-sided, per ruleset. With fair seats everywhere, one false refusal across 42 rulesets
# in about 4% of measurements; the power of ten below Bonferroni's 0.05 / 42. On the
# canonical top pair's 16 games only a clean or near-clean sweep reaches it.
SEAT_DECIDED_ALPHA = 0.001

# The wider reading's own threshold, half the pair's because it runs beside it. On 36
# games it refuses 29-7 and on 29 decided games 25-4, so it reads a strong lean and not
# only a sweep. Together the two cost about one false refusal per 1,000 measurements of a
# fair ruleset, 10% over a corpus of 71.
TOP_THREE_ALPHA = 0.0005

SEAT_GATE = Gate(
    component=SEAT_DECIDED_CHANCE,
    minimum=SEAT_DECIDED_ALPHA,
    reason=(
        "a game the two strongest rungs win from one seat nearly every time, whoever "
        "sits in it, is decided by the seat and not by play"
    ),
)

TOP_THREE_SEAT_GATE = Gate(
    component=TOP_THREE_SEAT_DECIDED_CHANCE,
    minimum=TOP_THREE_ALPHA,
    reason=(
        "a game the three strongest rungs win from one seat far more often than the "
        "other, whoever sits in it, is decided by the seat and not by play"
    ),
)

# Each gate with the diagnostics key whose split its refusal quotes.
_READINGS: tuple[tuple[Gate, str, str], ...] = (
    (SEAT_GATE, TOP_RUNG_SEAT_DETAIL, "two strongest rungs"),
    (TOP_THREE_SEAT_GATE, TOP_THREE_SEAT_DETAIL, "three strongest rungs"),
)


def sign_test(wins: int, losses: int) -> float:
    """The chance of a split at least this lopsided under a fair seat, two-sided.

    Exact: ``2 * P(Bin(n, 1/2) >= max(wins, losses))`` capped at one, with ``n`` the
    decided games. Draws are not counted, because a draw says nothing about which seat
    wins, and conditioning on ``n`` is what makes the figure exact. A fair coin is the
    least favourable null for any strength structure a both-orders schedule can hold,
    so the figure is conservative.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def transcript_digest(moves: Sequence[int]) -> str:
    """A short, stable name for one game's move list, so identical games compare equal.

    What a run stores for each decided game in a seat band, so the pooled reading can
    count distinct games across runs without the runs' records. Sixteen hex digits: a
    band holds tens of games, and a collision would undercount by one.
    """
    text = ",".join(str(move) for move in moves)
    return blake2b(text.encode(), digest_size=8).hexdigest()


def seat_violations(raw: dict[str, float], diagnostics: dict[str, Any]) -> list[str]:
    """Every seat refusal these measurements carry, pair first.

    Silent where a reading is absent, like ``withheld_components``: a hand-built
    ``Components`` has not measured the seat, and a roster whose third rung searches
    nothing emits no wide reading at all. Absence is not a verdict.
    """
    violations = []
    for gate, detail_key, rung_phrase in _READINGS:
        chance = raw.get(gate.component)
        if chance is None or gate.check(chance) is None:
            continue
        violations.append(_refusal(gate, chance, diagnostics.get(detail_key) or {}, rung_phrase))
    return violations


def seat_details(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Just the seat splits out of a run's diagnostics, dropping the rest.

    A run's diagnostics describe the corpus that run was taken over and stay in that
    run's own index file. These two keys are the exception, because the pooled reading
    adds them up across runs.
    """
    return {
        key: dict(diagnostics[key])
        for _gate, key, _phrase in _READINGS
        if isinstance(diagnostics.get(key), Mapping)
    }


def is_seat_violation(text: str) -> bool:
    """Whether a stored violation sentence came from a seat gate.

    Read by the combiner, which drops a run's own seat refusal from that run's vote and
    takes the pooled reading instead. Keyed on the component name each sentence opens
    with, so a run read off a file written months ago is classified the same way a run
    just scored is.
    """
    return any(text.startswith(f"{gate.component}=") for gate, _detail, _phrase in _READINGS)


# The share of decided games one seat may take before the check refuses. Selected by the
# criterion in `docs/reference/seat-effect-size.md` §5 -- the highest of a closed list of
# candidates at which a game swept nine times in ten is still caught four times in five,
# on the pair alone, at the library's median count of decided pair games. The criterion
# read no game's split, and the choice is not reopened by a larger sample.
POOLED_TOLERATED_SHARE = 2 / 3


def share_test(wins: int, losses: int, tolerated: float = POOLED_TOLERATED_SHARE) -> float:
    """The chance of a split this lopsided when the favoured seat takes `tolerated`.

    Exact and two-sided, draws dropped, the same conditioning `sign_test` uses. The null
    is composite -- every share at or under `tolerated`, for either seat -- and its least
    favourable point is the boundary, so both boundary tails are summed and the figure is
    an upper bound on the false-refusal rate. At ``tolerated = 0.5`` it is `sign_test` to
    the digit, so the built form is one member of this family.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    most = max(wins, losses)
    return min(1.0, _upper_tail(n, most, tolerated) + _upper_tail(n, most, 1.0 - tolerated))


def _upper_tail(n: int, k: int, share: float) -> float:
    """``P(Bin(n, share) >= k)``, summed from the far end so the small terms land first."""
    total = 0.0
    for i in range(n, k - 1, -1):
        total += exp(
            lgamma(n + 1)
            - lgamma(i + 1)
            - lgamma(n - i + 1)
            + i * log(share)
            + (n - i) * log(1.0 - share)
        )
    return total


def pooled_seat_reading(
    details: Sequence[Mapping[str, Any]],
    tolerated: float = POOLED_TOLERATED_SHARE,
) -> dict[str, Any]:
    """Both bands over several runs' games at once, with the refusals they carry.

    ``details`` is one run's seat diagnostics per run, in seed order. A band is pooled
    over the runs that measured it: a roster whose third rung searches nothing emits no
    wide band, and a run whose scorer raised emits neither, so each band records how many
    runs it rests on. A band no run measured is absent, and absence is not a verdict.

    Draws are summed and dropped from the test, exactly as one run's reading drops them.
    """
    bands: dict[str, Any] = {}
    for gate, detail_key, rung_phrase in _READINGS:
        found = [detail for detail in details if detail.get(detail_key)]
        if not found:
            continue
        bands[detail_key] = _band(
            [dict(detail[detail_key]) for detail in found], gate, rung_phrase, tolerated
        )
    refuses = any(band["refuses"] for band in bands.values())
    return {
        "tolerated_share": round(tolerated, 4),
        "bands": bands,
        "refuses": refuses,
        "violations": [band["refusal"] for band in bands.values() if band["refuses"]],
        # A refused game has no lean, whatever its other band read. The lean is the news
        # about a game that is *ranked*, and one sentence saying a refused game leans
        # 69% on its wider band is a thing a reader will one day print beside a refusal.
        "lean": "" if refuses else next(
            (band["lean"] for band in bands.values() if band["lean"]), ""
        ),
    }


def _band(
    per_run: list[dict[str, Any]], gate: Gate, rung_phrase: str, tolerated: float
) -> dict[str, Any]:
    """One band pooled over the runs that measured it."""
    wins = sum(int(run.get("wins", 0)) for run in per_run)
    losses = sum(int(run.get("losses", 0)) for run in per_run)
    draws = sum(int(run.get("draws", 0)) for run in per_run)
    rungs = sorted({int(rung) for run in per_run for rung in run.get("rungs", ())})

    decided = wins + losses
    distinct = _distinct(per_run)
    split = _distinct_split(per_run)
    # The test reads each different decided game once (Q953a): a line replayed sixteen
    # times is one observation of what its pairing does. A run stored before the split
    # was kept cannot say it, and the band is then tested as played, and says so.
    tested_wins, tested_losses = split if split is not None else (wins, losses)
    tested = tested_wins + tested_losses
    chance = share_test(tested_wins, tested_losses, tolerated)
    refuses = gate.check(chance) is not None
    share = max(wins, losses) / decided if decided else 0.5
    seat = "first" if wins >= losses else "second"
    tested_share = max(tested_wins, tested_losses) / tested if tested else 0.5
    tested_seat = "first" if tested_wins >= tested_losses else "second"

    band = {
        "component": gate.component,
        "rungs": rungs,
        "runs": len(per_run),
        "games": sum(int(run.get("games", 0)) for run in per_run),
        # The split as played, which the lean reads.
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decided": decided,
        "distinct": distinct,
        "favoured": seat if decided else "",
        "share": round(share, 4),
        # The split the test reads: each different decided game once, or as played.
        "counted": "distinct" if split is not None else "as played",
        "distinct_wins": tested_wins if split is not None else None,
        "distinct_losses": tested_losses if split is not None else None,
        "tested_favoured": tested_seat if tested else "",
        "tested_share": round(tested_share, 4),
        "chance": chance,
        "alpha": gate.minimum,
        "refuses": refuses,
    }
    band["refusal"] = _pooled_refusal(band, gate, rung_phrase, tolerated) if refuses else ""
    band["lean"] = "" if refuses else _lean(band, gate, rung_phrase, tolerated)
    return band


def _distinct(per_run: list[dict[str, Any]]) -> int | None:
    """How many of the band's decided games are different games, over every run.

    Counted on the union of the runs' transcripts, since two runs at different seeds can
    play the same game move for move and summing each run's own count would count it
    twice. ``None`` when a run predates the transcripts: a count it cannot take is not
    one to guess, and the lean that reads it then says nothing.
    """
    if any(not isinstance(run.get("transcripts"), list) for run in per_run):
        return None
    return len({str(digest) for run in per_run for digest in run["transcripts"]})


def _distinct_split(per_run: list[dict[str, Any]]) -> tuple[int, int] | None:
    """The first seat's wins and losses over the band's different decided games, pooled.

    Taken on the union of the runs' transcripts, as `_distinct` counts them, and split by
    the union of the transcripts each run's first seat won: a move list has one ending, so
    a transcript two runs both played is won by the same seat in both. ``None`` when a run
    stored no split, which every run scored before Q953a is.
    """
    if any(
        not isinstance(run.get("transcripts"), list)
        or not isinstance(run.get("first_seat_transcripts"), list)
        for run in per_run
    ):
        return None
    every = {str(digest) for run in per_run for digest in run["transcripts"]}
    won = {str(digest) for run in per_run for digest in run["first_seat_transcripts"]} & every
    return len(won), len(every) - len(won)


def _pooled_refusal(
    band: dict[str, Any], gate: Gate, rung_phrase: str, tolerated: float
) -> str:
    rungs = ", ".join(str(rung) for rung in band["rungs"])
    if band["counted"] == "distinct":
        tested = int(band["distinct_wins"]) + int(band["distinct_losses"])
        games = (
            f"{tested} decided games, each different game counted once "
            f"({band['decided']} as played),"
        )
    else:
        games = f"{band['decided']} decided games"
    return (
        f"{gate.component}={band['chance']:.2e} below minimum {gate.minimum}: the "
        f"{band['tested_favoured']} seat took {band['tested_share']:.0%} of the {games} "
        f"among the {band['games']} played between the {rung_phrase} "
        f"({rungs} nodes) over {band['runs']} runs, which rules out a seat held to "
        f"{tolerated:.0%} or less ({gate.reason})"
    )


def _lean(band: dict[str, Any], gate: Gate, rung_phrase: str, tolerated: float) -> str:
    """What a game leans by when the check does not refuse it, or nothing to say.

    Printed beside the score and read by no gate. Two conditions, and neither is a number
    of its own. The share has to be past what the check tolerates, because a game under
    that is inside what the check allows and has nothing to report. And the band has to
    hold enough decided games that a clean sweep of them *would* have been refused,
    because otherwise the lean says only that the game draws -- a band of three decided
    games reads 67% and means nothing. Both follow from `tolerated` and the band's alpha,
    which are fixed elsewhere.

    The floor counts *distinct* decided games (`seat-effect-size.md` §15). Deterministic
    players replay one game, so 200 decided games can be eleven games played many times,
    and a lean off eleven games is one the band has not earned however often they were
    replayed. A band whose distinct count is unknown prints nothing. The lean quotes the
    split as played; the refusal, since Q953a, tests the distinct split
    (`docs/reference/repeated-games.md` §6), so a band can lean on many replays of a few
    games without being refused for them.
    """
    # The split itself, not the stored `share`: that is rounded to four places, and 42 of
    # 63 rounds to 0.6667, past 2 / 3, so a split exactly at the line read as a lean. The
    # null is every share *at or under* the tolerated one, so the line itself is inside.
    # `took / decided` and `2 / 3` are the same correctly rounded double for every split
    # of exactly two in three.
    took = max(int(band["wins"]), int(band["losses"]))
    if not band["decided"] or took / band["decided"] <= tolerated:
        return ""
    distinct = band.get("distinct")
    if distinct is None or gate.check(share_test(distinct, 0, tolerated)) is None:
        return ""
    games = f"{band['decided']} decided games"
    if distinct < band["decided"]:
        games += f" ({distinct} of them distinct)"
    return (
        f"the {band['favoured']} seat took {took} of the {games} "
        f"between the {rung_phrase} ({band['share']:.0%}), more than the "
        f"{tolerated:.0%} the seat check tolerates, though not by enough to refuse "
        f"the game"
    )


def _refusal(gate: Gate, chance: float, detail: dict[str, Any], rung_phrase: str) -> str:
    rungs = ", ".join(str(rung) for rung in detail.get("rungs", ()))
    distinct = ""
    if "distinct_wins" in detail:
        distinct = (
            f", {detail['distinct_wins']}-{detail['distinct_losses']} with each different "
            f"game counted once"
        )
    return (
        f"{gate.component}={chance:.2e} below minimum {gate.minimum}: the first seat "
        f"went {detail.get('wins', '?')}-{detail.get('losses', '?')} with "
        f"{detail.get('draws', '?')} draws over the {detail.get('games', '?')} games "
        f"between the {rung_phrase} ({rungs} nodes){distinct}, one chance in "
        f"{1 / chance:,.0f} under a fair seat ({gate.reason})"
    )
