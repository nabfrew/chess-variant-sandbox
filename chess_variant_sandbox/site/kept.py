"""The store's own copy of a game's rules, for the page a retired game keeps.

`docs/plan-website.md`, *A game that leaves the corpus*. The local application draws a
game's page from the tree, so a spec that left `rulesets/` and `archive/` used to take
the page with it and a retired game answered 404. The user, 2026-09-18, asked for the
page to stay. The rules are still in `games.spec` -- the loader wrote them there and the
retirement did not touch them -- so the page is drawn from the row instead.

The row is a fallback and never a second library. `SiteApp.game` asks the tree first, so
every game the tree holds is read exactly as it was, and only a hash the tree has no spec
for reaches this module. What comes back is the same `library.Game` a committed spec
makes, with `source` reading *retired*, so `library.detail` describes it in the one place
rules are turned into sentences.

The measurement is still read from `runs/`. `canonical.json` is the single source for a
score here as everywhere else (`.claude/rules/viz-and-web.md`), and a retirement removes
nothing from disk -- the game was played and scored, and those files say so.

Nothing here computes anything about the game. The spec goes to `spec/describe.py` and
the engine through `library.detail`, which is the same path a committed spec takes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_variant_sandbox import corpus
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.web import library


def rules_of(connection: Any, digest: str) -> tuple[str, dict[str, Any]] | None:
    """The stored name and spec of one game, or `None` when there is nothing to draw.

    `None` covers both a hash the store has never held and a game whose rules were
    deleted, because `0010_deleted.sql` leaves the deleted row with a null spec. Neither
    has a page, and the caller answers both with the 404 an unknown hash gets.
    """
    with connection.cursor() as cursor:
        cursor.execute("select name, spec from games where hash = %s", (digest,))
        found = cursor.fetchone()
    if found is None or found[1] is None:
        return None
    return str(found[0]), dict(found[1])


def game_from(
    digest: str,
    name: str,
    spec: dict[str, Any],
    runs_dir: Path,
    weights: str = DEFAULT_WEIGHTS,
) -> library.Game:
    """One `library.Game` built from the store's copy of a spec.

    `path` is the measurement's own directory rather than a spec file, because there is
    no file: the rules came from a row. Nothing on the game's page reads it -- `keeping`
    resolves a game through `corpus.find`, which answers *unknown* for a hash in neither
    directory, and that is the true answer for a game that has left the tree.
    """
    return library.Game(
        hash=digest,
        name=name,
        description=str(spec.get("description", "")),
        path=corpus.measurement_dir(digest, runs_dir),
        source="retired",
        spec=spec,
        measurement=corpus.read_measurement(digest, runs_dir, weights, spec),
    )
