"""The store's specs in the one current spelling, every row keeping its key (Q1008b).

`0041_one_spelling.sql` rewrites `games.spec` in place as `spec/spelling.py::current_spelling`
rewrites a file (`docs/reference/schema-reboot-assessment.md` §5 step 3). The deploy's names
step (`deploy/names.py`) compares stored specs as they are *before* migrations run, so it does
not see this one; what it would have guarded is held here instead (§7): after the migration
every row is named by its own key, except the rows `0039` re-spelt in place, which keep the
name they already had apart from their key.

Asked of a store built as the live one was: the migrations before `0039` run, the old rest
games are stored as that deploy stored them, `0039` and `0040` run, then the library (each game
in other spellings `tests/respell.py` writes, keyed by its name), the retired games, a few
designed spellings and a deleted row are stored, and `0041` runs. Each stored spec must then
be exactly the Python converter's answer, valid, and named as before.

Since spelling 31 (Q1007b) the converter also writes a portal group's moves on the group, which
`0042_portal_moves.sql` does to the store after this one; so here a portal game is stored in
the stand-in spelling it had then (`schema.portal_stand_in_shape`), `0041` alone runs, and its
answer is the converter's with that one rewrite left out
(`tests/test_site_portal_moves.py` holds `0042`).

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from respell import RESPELLINGS, every_respelling
from retired_specs import REBUILT, RETIRED_HASHES, RETIRED_PREVIEWS, retired_preview
from test_new_spec_refusals import RETIRED, RETIRED_HASH
from test_schema_nine import FROZEN
from test_site_links import _scratch_store
from test_site_rest_migration import OLD_FORM

from chess_variant_sandbox.site import migrate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import portal_stand_in_shape, stored_spec_errors
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = "0041_one_spelling.sql"
STAMP = 30  # the version the migration writes: `SCHEMA_VERSION` when it was written
# The rows `0039` re-spelt in place under their old keys, so named apart from their key ever
# since (`schema-reboot-assessment.md` §7, `deploy/names.py`).
NAMED_APART = {"1d798bedbf66", "4da669bb3d50", "632c199e4589"}


def _designed() -> list[dict[str, Any]]:
    """Spellings a draft may hold that no library file does: each rewrite's input once."""
    chess = load_spec(ROOT / "rulesets" / "chess.json")
    out = []
    for index, rules in enumerate((
        {"promotion": [{"piece": "P", "rank": 5, "to_by_player": {"0": ["Q"], "1": ["Q"]}}]},
        {"promotion": [
            {"piece": "P", "ranks": [7, 6, 7], "to": ["Q"], "mandatory": False},
            {"piece": "Q", "rank": "last", "to": ["K"]},
        ], "win": "royal_capture"},
        {"wrap": {"files_by_player": {"0": "both", "1": "both"}, "ranks": "none"}},
        {"wrap": {"files_by_player": {"0": "both", "1": "none"}, "ranks": "none"}},
        {"turn": {"moves": 1, "ramp": [1]}, "strike": {"effect": "capture"}, "drops": {},
         "forbid_check": False, "scoring": None, "en_passant": False},
        {"turn": {"moves": 2}, "must_capture": False, "reach": None},
    )):
        spec = copy.deepcopy(chess)
        spec["name"] = f"designed {index}"
        spec["schema_version"] = 6
        spec["rules"].update(rules)
        if isinstance(rules.get("promotion"), list) or rules.get("turn", {}).get("moves", 1) > 1:
            spec["rules"].pop("en_passant", None)  # refused beside either
        out.append(spec)
    return out


def _rows() -> dict[str, dict[str, Any]]:
    """Every row stored after `0040`, by key: the key is the name each was stored under."""
    rows: dict[str, dict[str, Any]] = {}
    for index, (path, (digest, _)) in enumerate(sorted(FROZEN.items())):
        spec = load_spec(ROOT / path)
        respell = list(RESPELLINGS.values())[index % len(RESPELLINGS)]
        # A portal game with the stand-in the store held then (spelling 30), as the file
        # declared it (a re-spelling writes the army's keys out, never the stand-in's).
        rows[digest] = portal_stand_in_shape(every_respelling(spec) if index % 2 else respell(spec))
    for name, build in REBUILT.items():
        rows[RETIRED_HASHES[name]] = portal_stand_in_shape(build())
    for name, digest in RETIRED_PREVIEWS.items():
        rows[digest] = retired_preview(name)
    rows[RETIRED_HASH] = load_spec(RETIRED)
    for spec in _designed():
        assert validate_spec(spec) == [], (spec["name"], validate_spec(spec))
        rows[spec_hash(spec)] = spec
    for digest, spec in rows.items():
        assert spec_hash(spec) == digest, (digest, spec.get("name"))
    return rows


def _stored(store: Any) -> dict[str, Any]:
    with store.cursor() as cursor:
        cursor.execute("select hash, spec from games")
        return {str(digest): spec for digest, spec in cursor.fetchall()}


def test_the_migration_re_spells_every_row_and_keeps_every_name(tmp_path: Path) -> None:
    before_0039, before_0041 = tmp_path / "to-0038", tmp_path / "to-0040"
    through_0041 = tmp_path / "to-0041"
    for folder, below in ((before_0039, 39), (before_0041, 41), (through_0041, 42)):
        folder.mkdir()
        for migration in migrate.available():
            if migration.number < below:
                (folder / migration.name).write_text(migration.sql, encoding="utf-8")

    def store_rows(store: Any, rows: dict[str, dict[str, Any]]) -> None:
        with store.cursor() as cursor:
            for digest, spec in rows.items():
                cursor.execute(
                    "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                    (digest, json.dumps(spec), spec["name"], "corpus"),
                )
        store.commit()

    with _scratch_store() as store:
        migrate.run(store, before_0039)
        store_rows(store, {digest: spec for digest, (spec, _) in OLD_FORM.items()})
        migrate.run(store, before_0041)
        store_rows(store, _rows())
        with store.cursor() as cursor:
            # A deleted game keeps its row and loses its spec (`0010_deleted.sql`).
            cursor.execute(
                "insert into games (hash, spec, name, origin, deleted)"
                " values ('000000000000', null, 'gone', 'designed', now())"
            )
            # A spec that states no pieces is no game, and is not the migration's.
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values ('ffffffffffff', '{}'::jsonb, 'empty', 'designed')"
            )
        store.commit()
        before = _stored(store)

        report = migrate.run(store, through_0041)
        store.commit()
        assert report.applied == (MIGRATION,), report.summary()
        after = _stored(store)

        # Run again by hand: a re-spelt store is already in the one spelling.
        with store.cursor() as cursor:
            cursor.execute(migrate._read(migrate.MIGRATIONS / MIGRATION))
        again = _stored(store)
        store.rollback()
        assert again == after

    assert after.keys() == before.keys()
    assert after.pop("000000000000") is None
    assert after.pop("ffffffffffff") == {}
    assert len(after) == len(_rows()) + len(OLD_FORM)

    apart = {key for key, spec in before.items() if spec and spec_hash(spec) != key}
    assert NAMED_APART <= apart <= set(OLD_FORM), apart
    for key, spec in after.items():
        expected = portal_stand_in_shape(dict(current_spelling(before[key]), schema_version=STAMP))
        assert spec == expected, key
        # Valid once `0042` writes a portal group's moves on the group.
        respelt = current_spelling(spec)
        assert stored_spec_errors(respelt) == [], (key, stored_spec_errors(respelt))
        # Each row's name: its key, or for a row 0039 re-spelt, the name it already had.
        assert spec_hash(spec) == spec_hash(before[key]), key
        assert (spec_hash(spec) == key) is (key not in apart), key
