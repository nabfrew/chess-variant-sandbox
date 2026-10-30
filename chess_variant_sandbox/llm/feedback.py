"""Translating measurements into design observations the proposer can act on.

This module exists to hold one line: **the proposing model sees observations, never
scores.** It is told "a player allowed to think four times as long wins barely more
often" -- it is never told that this is the ``skill_gradient`` component, that it
scored 0.07, or that it carries the largest weight in the index.

The distinction is not squeamishness. A model shown the objective function optimises
the objective function, and the objective function is a proxy. Handed the weights, it
will discover whatever a component rewards cheaply -- under `v10`, that a game with
three pieces and no special rules maxed elegance and variety outright -- and start
proposing that. Handed observations, it has to reason
about what would actually make the game better -- which is the thing the index is
trying to approximate in the first place.

Raw facts about what happened are fair game and necessary; the scoring function is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.metrics.components import held_type_indices
from chess_variant_sandbox.metrics.index import IndexResult
from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_CHANCE

if TYPE_CHECKING:  # pragma: no cover - a type, and an import cycle if it were real
    from chess_variant_sandbox.loop.archive import Entry

# A piece type played less often than this share of all moves is worth naming. Not a
# threshold on quality -- a piece can be rare and decisive -- but a share this small in
# a corpus of competent games usually means the piece is not part of the game.
IDLE_SHARE = 0.02

# Read from the verdict, never from the first-player figure. That figure is 32 self-play
# games wide and cannot tell an even seat from a 0.28 lean, so no sentence about the seat
# is keyed on it in either direction. The seat gate reads the strongest rungs' games and
# fires only on a sweep, and this is what the sweep sounds like.
_SEAT_DECIDED = (
    "Between the two strongest players tested, the same seat won nearly every game "
    "whoever sat in it. Who moves first decides the result before play begins."
)

# Each rule reads one raw measurement and, when it fires, states what was observed in
# the language a designer would use. Ordered by how badly it usually needs fixing.
#
# None reads `rule_mass` or `rule_yield`, in either list. The user took rule yield out of
# the score with no limit on rule mass in its place (2026-09-23, `docs/archive/plan-variety.md`
# §5.4, A2), so the designer is neither told its rules are few nor that they are many.
# The last to go (Q400) was "the rules are heavy", which fired past a mass of 110: the
# ceiling the user declined, said as advice.
_OBSERVATIONS: list[tuple[str, Any, str]] = [
    (
        "search_advantage",
        lambda v: v < 0.6,
        "A player that searches ahead barely outperforms one choosing at random. "
        "Whatever decides these games, it is not the quality of the decisions.",
    ),
    (
        "skill_gradient",
        lambda v: v < 10,
        "Letting a player think twice as long gains it almost nothing. The positions "
        "do not seem to reward calculation -- either the good move is obvious or it "
        "does not matter which move is played.",
    ),
    (
        "stall_rate",
        lambda v: v > 0.3,
        "Most games run out of time or stop making progress rather than reaching a "
        "conclusion. The pieces cannot force a result against each other.",
    ),
    (
        "decisive_rate",
        lambda v: v < 0.35,
        "Competent players almost always draw. There is no way to convert an "
        "advantage into a win.",
    ),
    (
        "mean_branching",
        lambda v: v < 8,
        "Players typically have only a handful of legal moves. There is very little "
        "to choose between at any given moment.",
    ),
    (
        "mean_branching",
        lambda v: v > 70,
        "Players face an enormous number of legal moves every turn. Most of them are "
        "probably irrelevant, which makes the position hard to read rather than deep.",
    ),
    (
        "mean_plies",
        lambda v: v < 20,
        "Games are over almost immediately. There is no room for a plan to develop.",
    ),
    (
        "mean_plies",
        lambda v: v > 200,
        "Games drag on for a very long time, usually without resolving.",
    ),
    (
        "quick_kill_rate",
        lambda v: v > 0.15,
        "A large share of games are decided within the first few moves, which "
        "suggests an opening trap that ends the game before it starts.",
    ),
    (
        "opening_diversity",
        lambda v: v < 0.5,
        "Strong players open the same way nearly every game. The first moves appear "
        "to be forced or obviously best.",
    ),
    (
        "distinct_game_rate",
        lambda v: v < 0.6,
        "Many games repeat each other move for move. The play has one groove and "
        "falls into it.",
    ),
    (
        "lead_changes",
        lambda v: v < 1.0,
        "Once one side is ahead on material it essentially never changes hands. "
        "There is no comeback mechanism.",
    ),
    (
        "material_volatility",
        lambda v: v < 0.03,
        "Material almost never changes. Pieces rarely trade, so positions do not "
        "evolve.",
    ),
    (
        "gradient_persistence",
        lambda v: v < 0.4,
        "Thinking harder helps at first and then stops helping. Beyond a certain "
        "point players seem to have found everything there is to find, which suggests "
        "the game runs out of things to work out sooner than it runs out of moves.",
    ),
    (
        "gradient_persistence",
        lambda v: v > 2.0,
        "Weak players are barely distinguishable from each other while strong ones "
        "pull far apart. Whatever the game rewards, it is invisible until quite late, "
        "which makes the early experience of playing it close to arbitrary.",
    ),
    (
        "draw_slope",
        lambda v: v < 0.35,
        "Games that are decisive between weaker players peter out between stronger "
        "ones. Once both sides know what they are doing, neither can make progress.",
    ),
    (
        "decision_criticality",
        lambda v: v < 0.15,
        "Almost every legal move is about as good as every other. Players have many "
        "options and no reason to prefer any of them.",
    ),
    (
        "decision_criticality",
        lambda v: v > 0.9,
        "Nearly every legal move loses material immediately. There is usually one "
        "playable move, so players are following a line rather than choosing one.",
    ),
    (
        "piece_utilisation",
        lambda v: v < 0.55,
        "Some piece types barely move all game. Whatever they were added for is not "
        "happening, and a player would not miss them.",
    ),
    (
        "blunder_recoverability",
        lambda v: v < 0.12,
        "A single careless move loses the game almost every time. The winner is "
        "whoever slips second rather than whoever plays better.",
    ),
    (
        "blunder_recoverability",
        lambda v: v > 0.9,
        "Forcing a random move into a game between strong players changes almost "
        "nothing. Whatever decides these games, it is not the moves.",
    ),
    (
        "ladder_monotonicity",
        lambda v: v < 0.6,
        "Players given more time to think did not reliably beat players given less. "
        "That is unusual enough to suspect the goal of this game is hard to judge from "
        "a position -- if what counts as progress cannot be recognised part-way "
        "through, calculating further ahead does not help.",
    ),
]

_STRENGTHS: list[tuple[str, Any, str]] = [
    ("skill_gradient", lambda v: v > 45, "Thinking longer reliably wins, which is the main thing working here."),
    ("opening_diversity", lambda v: v > 0.8, "Strong players open in genuinely varied ways."),
    ("lead_changes", lambda v: 2.0 < v < 12.0, "The advantage changes hands often enough to keep games live."),
    (
        "gradient_persistence",
        lambda v: 0.8 < v < 1.7,
        "Thinking harder keeps paying off all the way up, so there is no obvious "
        "ceiling on how well this could be played.",
    ),
    (
        "piece_utilisation",
        lambda v: v > 0.9,
        "Every kind of piece gets used. Nothing on the board is decoration.",
    ),
    (
        "blunder_recoverability",
        lambda v: 0.25 < v < 0.6,
        "One bad move hurts without automatically ending the game, so there is "
        "something to play for after a mistake.",
    ),
]


def describe(result: IndexResult, spec: dict[str, Any] | None = None) -> str:
    """Prose feedback on a tested ruleset, with no scores or weights in it.

    ``spec`` is optional and only ever used for *names*: a count of moves per piece type
    is a fact about the games, but "your elephant moved eleven times in 144 games" is a
    design observation and "type 6 moved eleven times" is a puzzle.
    """
    lines: list[str] = []
    raw = result.raw
    diagnostics = result.diagnostics

    lines.append(
        f"Tested over {diagnostics.get('games', '?')} games between players of "
        f"differing strength."
    )
    lines.append(
        f"Games ran {raw.get('mean_plies', 0):.0f} plies on average with about "
        f"{raw.get('mean_branching', 0):.0f} legal moves per turn, and "
        f"{raw.get('decisive_rate', 0) * 100:.0f}% of them produced a winner."
    )

    reasons = diagnostics.get("outcome_reasons") or {}
    if reasons:
        breakdown = ", ".join(
            f"{reason} {count}" for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"Endings: {breakdown}.")

    lines.extend(_strength_lines(diagnostics))
    lines.extend(_piece_lines(diagnostics, spec))
    lines.extend(_blunder_lines(diagnostics))

    problems = _fire(_OBSERVATIONS, raw)
    if any(text.startswith(SEAT_DECIDED_CHANCE) for text in result.violations):
        problems.insert(0, _SEAT_DECIDED)
    if problems:
        lines.append("\nProblems observed:")
        lines.extend(f"- {text}" for text in problems)

    strengths = _fire(_STRENGTHS, raw)
    if strengths:
        lines.append("\nWorking well:")
        lines.extend(f"- {text}" for text in strengths)

    if result.unreadable:
        # Not the same message, and the difference matters more than it looks. A ruleset
        # the roster cannot read is not a bad game: the opponents played it towards the
        # wrong objective, so every number that came out of their ratings describes
        # them. Telling a designer to change something structural on that evidence would
        # be asking them to fix the instrument.
        lines.append(
            "\nThe programs that played this could not work out how to win it: giving "
            "them more time to think did not make them play any better. Nothing here is "
            "a judgement about the design -- the game may be fine and simply need a "
            "different kind of opponent. Read the observations above as being about "
            "what happened, not about what is possible."
        )
        if result.measured:
            # A blind run that still failed a test the players could read (plan-teacher
            # §10.16): that test is a fact about the game, whatever the ladder did.
            lines.append(
                "It was still rejected on a test the programs could read, and that part "
                "is about the design: something structural has to change."
            )
        else:
            lines.append("So it is not measured: there is no verdict either way.")
    elif not result.viable:
        lines.append(
            "\nThis ruleset was rejected outright. It is not a near miss to be tuned; "
            "something structural has to change."
        )

    return "\n".join(lines)


def _strength_lines(diagnostics: dict[str, Any]) -> list[str]:
    """How the result changed as the players got better.

    Two games with the same overall decisive rate can be completely different: one where
    the defence holds once both sides can see two moves ahead, and one where the result
    never stops being decisive because skill never converts into control. A designer can
    act on that difference; a single average hides it.
    """
    tiers = diagnostics.get("decisive_by_tier") or {}
    low, high = tiers.get("low_decisive"), tiers.get("high_decisive")
    if low is None or high is None:
        return []
    return [
        f"Between the weaker players {low * 100:.0f}% of games produced a winner; "
        f"between the strongest, {high * 100:.0f}%."
    ]


def _piece_lines(diagnostics: dict[str, Any], spec: dict[str, Any] | None) -> list[str]:
    """Which pieces actually got played, by name.

    The most common way an invented ruleset goes wrong is material added for the sake of
    novelty: a piece that exists in the rules, sits on its home square, and would not be
    missed. Nothing else in this feedback would tell the model that, and it is exactly
    the kind of thing it can fix.
    """
    totals = diagnostics.get("moves_by_type")
    if not totals or sum(totals) <= 0:
        return []

    # The pieces a player holds, read as the score reads them: `moves_by_type` follows the
    # compiled types, which on a portal game run past the spec's piece list with one per
    # portal end, and a portal belongs to the board (Q877, after Q875). Leaving it out also
    # stops the proposer being told to give a portal a job.
    names = _piece_names(spec, len(totals))
    held = held_type_indices(spec or {}, len(totals))
    names = [names[index] for index in held]
    totals = [totals[index] for index in held]
    played = sum(totals)
    if played <= 0:
        return []
    ranked = sorted(zip(names, totals, strict=False), key=lambda pair: pair[1])

    lines = [
        "Moves played by piece: "
        + ", ".join(
            f"{name} {count} ({count / played * 100:.0f}%)"
            for name, count in sorted(ranked, key=lambda pair: -pair[1])
        )
        + "."
    ]
    idle = [name for name, count in ranked if count / played < IDLE_SHARE]
    if idle:
        listed = ", ".join(idle)
        lines.append(
            f"Barely used: {listed}. Over the whole corpus these were hardly ever "
            "worth moving, so a player would not miss them."
        )
    return lines


def _blunder_lines(diagnostics: dict[str, Any]) -> list[str]:
    """What one careless move costs.

    Both ends of this are worth knowing and neither is visible from a result list: a
    game where an arbitrary move always loses rewards not erring rather than playing
    well, and one where it never matters is not responding to move quality at all.
    """
    trials = diagnostics.get("blunder_trials")
    changed = diagnostics.get("blunder_changed_result")
    if not trials or changed is None:
        return []
    return [
        f"One arbitrary legal move was forced into {trials} games between strong "
        f"players a third of the way in; it changed who won in {changed * 100:.0f}% "
        "of them."
    ]


def _piece_names(spec: dict[str, Any] | None, count: int) -> list[str]:
    pieces = (spec or {}).get("pieces") or []
    names = [str(piece.get("name") or piece.get("id", "?")) for piece in pieces]
    if len(names) < count:
        names.extend(f"piece {index}" for index in range(len(names), count))
    return names[:count]


def describe_failures(entries: Sequence[Entry]) -> str:
    """What was already tried from here and did not work.

    The single largest unused signal in the archive. Rejected candidates are kept
    precisely because a proposal that failed its gate names a specific way this design
    space goes wrong -- and until now the proposer saw only a list of hashes to avoid,
    which tells it that something was tried and nothing about what happened.

    Same rule as everything else in this module: what failed and why, never how it
    scored.
    """
    if not entries:
        return ""

    lines: list[str] = []
    for entry in entries:
        reason = (entry.notes or "").strip().splitlines()
        headline = reason[0] if reason else "no reason recorded"
        if entry.status == "invalid":
            what = "did not satisfy the schema"
        elif entry.status == "rejected":
            what = "was thrown out before it could be played"
        elif not entry.measured:
            # *Not measured* (plan-teacher §10.16): no verdict, so not a disqualification.
            what = ("was played, but the programs could not work out how to win it, so "
                    "there is no verdict either way")
        else:
            what = "was played and disqualified"
        lines.append(f"- {entry.name}: {what} -- {headline}")
    return "\n".join(lines)


def _fire(rules: list[tuple[str, Any, str]], raw: dict[str, float]) -> list[str]:
    fired: list[str] = []
    for component, predicate, text in rules:
        value = raw.get(component)
        if value is not None and predicate(value) and text not in fired:
            fired.append(text)
    return fired
