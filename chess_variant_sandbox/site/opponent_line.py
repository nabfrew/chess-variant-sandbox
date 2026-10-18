"""The opponent line: how far people's play has tested a game's strongest computer player.

Q653, B2 of the launch-set plan, as built (`docs/reference/opponent-line.md` §1). The
trust line says how far to trust a game's *score*; this line says whether its strongest
computer player has stood up to the people who played it. The person test was scrapped by
the user on 2026-10-03, so the line reports natural play and never a pass: one of three
states, read live from the counts `site/people.py` already folds for the trust line.

- **beaten by people**: `BEATEN_PEOPLE` or more people have beaten the strongest computer
  player (`unbeaten`'s own clears, so the line and the *Unbeaten* tag cannot disagree);
- **held so far**: nobody has, over `HELD_GAMES` or more rated games against it by
  `HELD_PEOPLE` or more people;
- **not enough games yet**: everything else, a game nobody has played included.

The thresholds were fixed on 2026-10-04 before any count was read, and are written in the
record (§1); 20 games by 5 people are the trust line's own (`trust.RULE`), so the line adds
no new number. A state is never inferred from a score. Nothing here moves a score, a
fingerprint or the trust level: this module reads `trust.People` and nothing reads it back.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox import trust

# Pre-registered 2026-10-04 (`docs/reference/opponent-line.md` §1); never re-aimed.
BEATEN_PEOPLE = 1
HELD_GAMES = 20
HELD_PEOPLE = 5

BEATEN = "beaten by people"
HELD = "held so far"
EARLY = "not enough games yet"
STATES = (BEATEN, HELD, EARLY)

_SAID = {
    BEATEN: "People have beaten the strongest computer player here. A win against it still "
            "counts.",
    HELD: "Nobody has beaten the strongest computer player here in the games people have "
          "played against it.",
    EARLY: "Too few people have played the strongest computer player here to say how hard "
           "it is to beat.",
}

RULE_SAID = (
    f"It reads held so far once {HELD_GAMES} rated games against it by {HELD_PEOPLE} different "
    "people bring no win, and beaten by people from the first person's win."
)


def state(people: trust.People) -> str:
    """The state people's counts say, by the thresholds above and nothing else."""
    if people.cleared >= BEATEN_PEOPLE:
        return BEATEN
    if people.games >= HELD_GAMES and people.people >= HELD_PEOPLE:
        return HELD
    return EARLY


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def shown(people: trust.People | None, *, rated: bool = True) -> dict[str, Any]:
    """The line as the page draws it: ``state``, ``said`` and ``more``.

    ``rated`` false is a game whose computer players have no ratings on the site, so no
    game against them is rated and none can be counted: it reads *not enough games yet*,
    and *more* says why rather than counting nothing.
    """
    found = people if people is not None else trust.People(0, 0, 0)
    if not rated:
        return {
            "state": EARLY,
            "said": _SAID[EARLY],
            "more": ["Its computer players have no ratings here yet, so no game against "
                     "them is rated."],
        }
    more: list[str] = []
    if found.games:
        when = trust.month(found.latest)
        more.append(
            f"{_plural(found.games, 'rated game', 'rated games')} against the strongest "
            f"computer player, by {_plural(found.people, 'person', 'people')}"
            + (f", the latest in {when}." if when else ".")
        )
    else:
        more.append("No rated games against the strongest computer player yet.")
    if found.cleared:
        more.append(f"{_plural(found.cleared, 'person has', 'people have')} beaten it.")
    elif found.games:
        more.append("Nobody has beaten it yet.")
    more.append(RULE_SAID)
    now = state(found)
    return {"state": now, "said": _SAID[now], "more": more}
