"""The hashes committed files held before a mechanics edit, and the hash each holds now.

A mechanics edit in place is a new game (CLAUDE.md, Conventions): the file keeps its path and
its human name, takes a new `spec_hash`, and the old hash's runs stay where they are. Every
record that named the old hash goes on naming it -- a child's `provenance.parents`, a design
run's archive, a stored game's parent -- and that is right, because it is what the child was
made from on the day. What it is not is a game on today's map. `EDITED` is the edit's own
record, read by whatever turns a recorded parent into a game drawn today (Q1075, 2026-10-31,
`docs/reference/lineage-through-edits.md`).

Keyed by the old hash, valued by the hash the same file held right after the edit; a file
edited twice is two rows, and `current` follows the chain. Every row matches a `was <hash>`
comment on that file's row of `tests/test_schema_nine.py::FROZEN`, which
`tests/test_edits.py` holds both ways: an edit re-frozen without a row here fails, and so
does a provenance parent that resolves to no game.

Add a row in the same change as the re-freeze. Never rewrite a child's provenance instead:
the old hash is the true record, and it is what keys the old runs.
"""

from __future__ import annotations

EDITED: dict[str, str] = {
    # Q454, 2026-09-25: shogi made traditional shogi (promotion on leaving the zone).
    "630828e31147": "54c731ca6e95",  # rulesets/shogi.json
    # Q869, 2026-10-13: the rear pawn respelled as a pawn facing backwards (schema 25).
    "ab865f3f954a": "9598e603241d",  # archive/crazyhouse-torus.json
    "3194d3a2251b": "8d9fb1b4c0b4",  # archive/torus-less-extended.json
    # Q1069a, 2026-10-30: `rules.insufficient_material` switched on.
    "9c26a04f1ca5": "1605b8619844",  # rulesets/almost.json
    "14a126b38a1a": "d06cfc337d9c",  # rulesets/berolina.json
    "57c73975c905": "7cddf4ba316b",  # archive/chess-oriented-pawns.json
    "4c1922ffae26": "831c77e75650",  # rulesets/chess.json
    "9f538ab85a77": "accd659f2c76",  # archive/sardines.json
    "69f1e486af87": "e1bb4740516e",  # archive/chess-with-galloping-knights.json
    "91c2c24305e5": "bafc4a7011a3",  # archive/chess-with-preaching-bishops.json
    "df2fbba0ca76": "a5e2a7e05c9a",  # archive/perpetual.json
    "8fc3939243d5": "6784c35550d8",  # archive/weathervane-half-turn.json
    "1442513d38d7": "241fcd286e26",  # archive/weathervane.json
    "fee1ab560013": "b48942ad079f",  # archive/chess-with-a-weary-queen.json
    "59a118c8ee38": "38ba9c25e715",  # archive/madrasi.json
    "c2b124298181": "3fdad4d69d87",  # archive/chess-with-daunting-knights.json
    "cc5e28e9ed0f": "ce17f9cdcd55",  # archive/chess-with-nightriders.json
    "23f6613e9fa3": "7e57cab1fd7e",  # archive/crooked-bishops.json
    "379efba235c5": "4bf09661fdd9",  # archive/circling-queen.json
    # Q1074bd, 2026-10-31: dohyo becomes the corner ring.
    "242c02248dd2": "559e947e585a",  # rulesets/dohyo.json
}


def current(digest: str) -> str:
    """The hash the file that once held `digest` holds now; `digest` itself if never edited.

    Follows a chain of edits to its end. A cycle would mean an edit undone by a later one,
    and stops where it would revisit a hash, at the latest hash before the repeat.
    """
    seen = {digest}
    while digest in EDITED and EDITED[digest] not in seen:
        digest = EDITED[digest]
        seen.add(digest)
    return digest
