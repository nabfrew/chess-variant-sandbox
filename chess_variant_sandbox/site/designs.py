"""The store's rows the designer milestones read, D1 to D4 of `docs/archive/plan-meta-game.md`.

`site/milestones.py` reads rows and opens no connection, so the two queries it needs are
here: the designed games around a person or a game (`games.owner` and `games.parent`,
by hash), and the signed-in seats on those games (`matches`, by game).

## Which rows

A **design** is a `games` row with `origin = 'designed'`. A killed or deleted one stays:
its designer kept it and somebody forked or played it, and none of that is undone by the
rules being taken away (the seven tests' first, nothing lapses). A game loaded from the
tree is not a design, and a fork of one has no designer upstream to tell.

A **try** is a signed-in seat in a match on a design, in either mode. Sandbox rows are read
by no statistic (`site/matches.py`), and this is the one stated exception, the user's
decision B of 2026-09-14: practice games started by other signed-in people are counted, as
a plain number of people on the game's page, with no names and nothing else reading them.
A visitor's seat names nobody, and a watch seats no person, so neither is a try. Rows the
local application loaded (`source = 'local-app'`) predate the site and are left out.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site.milestones import Design, Try

DESIGNED = "designed"

_COLUMNS = "hash, owner, parent, created, screen_passed"


def _design(found: Any) -> Design:
    return Design(
        hash=str(found[0]),
        owner=None if found[1] is None else int(found[1]),
        parent=None if found[2] is None else str(found[2]),
        created=found[3],
        screen_passed=bool(found[4]),
    )


def of(connection: Any, owner: int) -> tuple[Design, ...]:
    """This person's designs, and every design forked from one of them."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_COLUMNS} from games where origin = %s and ("
            " owner = %s or parent in"
            " (select hash from games where origin = %s and owner = %s))"
            " order by created, hash",
            (DESIGNED, owner, DESIGNED, owner),
        )
        return tuple(_design(found) for found in cursor.fetchall())


def around(connection: Any, digest: str) -> tuple[Design | None, tuple[Design, ...]]:
    """One design and the designs forked from it, or `None` for a game that is not one."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_COLUMNS} from games where origin = %s and (hash = %s or parent = %s)"
            " order by created, hash",
            (DESIGNED, digest, digest),
        )
        found = [_design(row) for row in cursor.fetchall()]
    game = next((design for design in found if design.hash == digest), None)
    if game is None:
        return None, ()
    return game, tuple(design for design in found if design.parent == digest)


def tries(connection: Any, games: Iterable[str]) -> tuple[Try, ...]:
    """Every signed-in seat in a match on these games, either mode, oldest first."""
    wanted = sorted(set(games))
    if not wanted:
        return ()
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, seat0_user, seat1_user, started, id from matches"
            " where game = any(%s) and (source is null or source <> %s)"
            " and (seat0_user is not null or seat1_user is not null)"
            # A game between people called off before both moved is nobody's try (Q570).
            " and coalesce(reason, '') <> %s"
            " order by started, id",
            (wanted, match_rows.LOCAL_APP, match_rows.CANCELLED),
        )
        rows = cursor.fetchall()
    found: list[Try] = []
    for game, first, second, started, match in rows:
        for user in dict.fromkeys(seat for seat in (first, second) if seat is not None):
            found.append(Try(game=str(game), user=int(user), started=started, match=str(match)))
    return tuple(found)
