"""Schema 29 (Q983a): the store's migration rewrites every stored copy of the retired rest.

`docs/plan-rest.md` §4.6: once the validator and the loader refuse `rules.simultaneous.rest`, a
retired game's page and its match replays, built from `games.spec`, would fail. The migration
`0039_rest_on_pieces.sql` rewrites each stored spec that carries the old key into the new
spelling, keeping the row's hash. Asked of a store that really was at `0038`: the migrations
before it run, old-form games are stored as the deploy of that time stored them, then `0039`
runs, as the deploy will. Every rewritten spec must be the files' rewrite
(`tests/rest_equivalence.py::rewrite`) exactly, validate, build an engine and hash to the
rewritten file's game; every other row is untouched.

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from rest_equivalence import REWRITE_HISTORY, rewrite
from test_site_links import _scratch_store

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.site import matches, migrate
from chess_variant_sandbox.spec import spec_hash, validate_spec

HERE = Path(__file__).parent
ROOT = HERE.parent
MIGRATION = "0039_rest_on_pieces.sql"


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _unwritten(spec: dict[str, Any]) -> dict[str, Any]:
    """A rewritten file as it stood before the bump: the rests off, the switch on, the
    version it declared, the history's sentence taken back."""
    old = copy.deepcopy(spec)
    old["schema_version"] = 27
    for piece in old["pieces"]:
        piece.pop("rest", None)
    old["rules"]["simultaneous"]["rest"] = True
    provenance = old.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("history"), str):
        provenance["history"] = provenance["history"].removesuffix(f" {REWRITE_HISTORY}")
    return old


def _wild_resting() -> dict[str, Any]:
    """A round game with a wild piece and the old rest: the wild piece never made a
    player's move, so it is the one piece the rewrite does not rest."""
    spec = _load(HERE / "_simultaneous_wild.json")
    spec["schema_version"] = 27
    spec["rules"]["simultaneous"]["rest"] = True
    return spec


# Old-form specs as the store holds them, each with the hash its row carries, and the file
# whose game its rewrite must be (None: checked against `rewrite` alone).
BREATHER = _unwritten(_load(ROOT / "archive" / "breather.json"))
SPIES = _unwritten(_load(ROOT / "tests" / "_retired_gardner_with_spies_resting.json"))
RESTING = _load(HERE / "_retired_gardner_in_rounds_resting.json")
OLD_FORM = {
    "632c199e4589": (BREATHER, ROOT / "archive" / "breather.json"),
    "1d798bedbf66": (SPIES, ROOT / "tests" / "_retired_gardner_with_spies_resting.json"),
    "4da669bb3d50": (RESTING, HERE / "_simultaneous_resting.json"),
    spec_hash(_wild_resting()): (_wild_resting(), None),
}
UNTOUCHED = _load(ROOT / "archive" / "watering-hole.json")


def _knight_twice() -> list[int]:
    """White's knight out, a Black pawn step, the same knight again: legal in the spy game's
    alternating phase under schema 27 (SP1, `_retired_gardner_with_spies.json`, plays its
    alternating phase as SP2 did), refused under schema 29, where the knight rests."""
    from retired_specs import retired_preview

    from chess_variant_sandbox.engine.moves import decode_move

    engine = Engine(retired_preview("gardner-with-spies"))
    state = engine.initial_state()
    played = []
    for origin, target in (("b1", "c3"), ("e4", "e3"), ("c3", "b5")):
        word = next(
            move for move in engine.legal_moves(state)
            if decode_move(move)[:2] == (engine._square_index(origin), engine._square_index(target))
        )
        played.append(word)
        state = engine.apply(state, word)
    return played


def test_the_old_forms_are_the_old_games() -> None:
    """The specs this test stores are the games the live rows hold: the old hashes, refused
    now (the validator knows no `rules.simultaneous.rest`)."""
    for digest, (spec, _) in OLD_FORM.items():
        assert spec_hash(spec) == digest
        assert any("rest" in error for error in validate_spec(spec)), digest


def test_the_migration_rewrites_every_stored_old_rest_and_nothing_else(tmp_path: Path) -> None:
    older = tmp_path / "migrations-before-0039"
    older.mkdir()
    through = tmp_path / "migrations-through-0039"  # later ones (`0041`) re-spell again
    through.mkdir()
    for migration in migrate.available():
        if migration.number < 39:
            (older / migration.name).write_text(migration.sql, encoding="utf-8")
        if migration.number <= 39:
            (through / migration.name).write_text(migration.sql, encoding="utf-8")

    with _scratch_store() as store:
        migrate.run(store, older)
        rows = [(digest, spec) for digest, (spec, _) in OLD_FORM.items()]
        rows.append((spec_hash(UNTOUCHED), UNTOUCHED))
        with store.cursor() as cursor:
            for digest, spec in rows:
                cursor.execute(
                    "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                    (digest, json.dumps(spec), spec["name"], "corpus"),
                )
            # A deleted game keeps its row and loses its spec (`0010_deleted.sql`).
            cursor.execute(
                "insert into games (hash, spec, name, origin, deleted)"
                " values ('000000000000', null, 'gone', 'designed', now())"
            )
        store.commit()

        report = migrate.run(store, through)
        store.commit()
        assert report.applied == (MIGRATION,), report.summary()

        with store.cursor() as cursor:
            cursor.execute("select hash, spec from games")
            stored = {str(digest): spec for digest, spec in cursor.fetchall()}

        # A finished spy match whose knight moved on two turns running, which schema 27 let
        # it do and schema 29 does not: its board cannot come back, and it says so.
        twice = _knight_twice()
        with store.cursor() as cursor:
            for match_id, game, moves in (
                ("stale-spy-match", "1d798bedbf66", twice),
                ("old-breather-match", "632c199e4589", []),
            ):
                cursor.execute(
                    "insert into matches (id, game, mode, seat0_agent, seat1_agent, seed, moves,"
                    " result, reason, finished) values (%s, %s, 'sandbox', 'greedy', 'greedy',"
                    " 0, %s::jsonb, 1.0, 'royal_captured', now())",
                    (match_id, game, json.dumps(moves)),
                )
        store.commit()
        with pytest.raises(matches.MatchError, match="no longer replay"):
            matches.session(store, "stale-spy-match")
        assert matches.session(store, "old-breather-match").state.ply == 0

    assert stored["000000000000"] is None
    assert stored[spec_hash(UNTOUCHED)] == UNTOUCHED
    for digest, (old, path) in OLD_FORM.items():
        migrated = stored[digest]
        assert migrated == rewrite(old), digest
        assert "rest" not in migrated["rules"]["simultaneous"]
        assert validate_spec(migrated) == [], digest
        Engine(migrated)
        if path is not None:
            assert spec_hash(migrated) == spec_hash(_load(path)), digest
    wild = stored[spec_hash(_wild_resting())]
    npcs = {npc["piece"] for npc in wild["rules"]["npcs"]}
    assert all(("rest" in piece) == (piece["id"] not in npcs) for piece in wild["pieces"])
