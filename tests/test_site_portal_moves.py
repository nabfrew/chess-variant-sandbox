"""The store's portal games in spelling 31, every row keeping its key (Q1007b).

`0042_portal_moves.sql` rewrites `games.spec` in place as `spec/spelling.py::current_spelling`
rewrites a file (`docs/plan-schema-31.md` §4.2): a group that names a stand-in piece takes its
moves as its own `moves`, the stand-in leaves `pieces`, and every stored spec is stamped 31. The
deploy's names step (`deploy/names.py`) compares stored specs *before* migrations run, so it
cannot see a rename made here; what it would have guarded is held here instead: after the
migration every row is exactly the Python converter's answer, valid, and named by its own key
(`schema-reboot-assessment.md` §7's rule for an in-place re-spelling).

Asked of a store built as the live one stands after `0041`: the migrations before `0042` run,
then each game is stored as `0041` left it (the portal games in the stand-in spelling every
portal game had, `schema.portal_stand_in_shape`), and `0042` runs. A stored portal game of any
other shape, which the normal form could not name as it was, stops the migration with nothing
written.

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_schema_nine import FROZEN
from test_site_links import _scratch_store

from chess_variant_sandbox.site import migrate
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import portal_stand_in_shape
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = "0042_portal_moves.sql"
STAMP = 31  # the version the migration writes: `SCHEMA_VERSION` when it was written
STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


def _as_0041_left_it(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` as the store held it after `0041`: in that spelling, its portal moves on the
    stand-in piece, stamped 30."""
    old = portal_stand_in_shape(current_spelling(spec))
    return {**old, "schema_version": 30}


def _designed() -> list[dict[str, Any]]:
    """Portal games a member's draft could hold that the library does not: a pair nobody moves
    (a stand-in with no moves), a fold beside a momentum map, two groups that move the same
    way, and a game that declares its own `O`, whose stand-in the designer called `V`."""
    chess = load_spec(ROOT / "rulesets" / "chess.json")
    out = []
    for index, portals in enumerate((
        [{"orange": "c4", "blue": "f5"}],
        [{"moves": [STEP], "orange": "c4", "blue": "f5", "kind": "fold",
          "moved_by": {"orange": [0, 1], "blue": [0, 1]}},
         {"moves": [STEP], "ends": [{"at": "a3", "to": 1}, {"at": "d3", "to": 2},
                                    {"at": "h6", "to": 0}]}],
        [{"moves": [STEP], "orange": "b4", "blue": "g5",
          "moved_by": {"orange": [0, 1], "blue": [0, 1]}, "no_straight_back": True}],
    )):
        spec = copy.deepcopy(chess)
        spec["name"] = f"designed portals {index}"
        spec["rules"]["portals"] = portals
        out.append(spec)
    own = copy.deepcopy(chess)
    own["name"] = "designed portals with an O"
    for piece in own["pieces"]:
        if piece["id"] == "N":
            piece["id"] = "O"
    for placement in own["setup"]["placements"]:
        if placement["piece"] == "N":
            placement["piece"] = "O"
    for entry in [own["rules"]["promotion"]]:
        entry["to"] = ["O" if pid == "N" else pid for pid in entry["to"]]
    own["rules"]["portals"] = [{"moves": [STEP], "orange": "c4", "blue": "f5",
                                "moved_by": {"orange": [0]}}]
    out.append(own)
    for spec in out:
        assert validate_spec(spec) == [], (spec["name"], validate_spec(spec))
    return out


def _rows() -> dict[str, dict[str, Any]]:
    """Every row as `0041` left it, by key: each portal game of the library, the test fixtures
    and the designs, and three games without portals, whose rows only take the stamp."""
    games = [load_spec(ROOT / path) for path in sorted(FROZEN)]
    games += [load_spec(ROOT / "tests" / name) for name in (
        "_simultaneous_portals.json", "_simultaneous_speed_portals.json")]
    games += _designed()
    portal = [spec for spec in games if spec.get("rules", {}).get("portals")]
    plain = [spec for spec in games if not spec.get("rules", {}).get("portals")][:3]
    rows = {spec_hash(spec): _as_0041_left_it(spec) for spec in portal + plain}
    # A stand-in that writes out a key the normal form drops is still the one shape.
    rows[spec_hash(_designed()[3])]["pieces"][-1]["royal"] = False
    for digest, spec in rows.items():
        assert spec_hash(spec) == digest, (digest, spec.get("name"))
    assert sum(1 for spec in rows.values() if spec["rules"].get("portals")) == len(portal) >= 16
    return rows


def _below(tmp_path: Path, number: int) -> Path:
    """A folder of the migrations numbered below ``number``."""
    folder = tmp_path / f"to-{number - 1:04d}"
    folder.mkdir()
    for migration in migrate.available():
        if migration.number < number:
            (folder / migration.name).write_text(migration.sql, encoding="utf-8")
    return folder


def _store(store: Any, rows: dict[str, dict[str, Any]]) -> None:
    with store.cursor() as cursor:
        for digest, spec in rows.items():
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps(spec), spec["name"], "designed"),
            )
    store.commit()


def _stored(store: Any) -> dict[str, Any]:
    with store.cursor() as cursor:
        cursor.execute("select hash, spec from games")
        return {str(digest): spec for digest, spec in cursor.fetchall()}


def test_a_stored_portal_game_holds_the_stand_in_every_portal_game_had() -> None:
    """The store's old spelling, as this file builds it, is the shape the migration expects:
    one stand-in `O` (or `V`) named by every group, last in `pieces`."""
    for spec in _rows().values():
        groups = spec["rules"].get("portals")
        if not groups:
            continue
        named = {group["piece"] for group in groups}
        assert len(named) == 1 and spec["pieces"][-1]["id"] in named
        stand = spec["pieces"][-1]
        assert stand["name"] == "portal" and stand["value"] == 0
        assert set(stand) <= {"id", "name", "value", "moves", "royal"}
        assert all(move["mode"] == "move" for move in stand["moves"])
        assert validate_spec(spec), "the old spelling is refused"


def test_the_migration_re_spells_every_portal_row_and_keeps_every_name(tmp_path: Path) -> None:
    rows = _rows()
    with _scratch_store() as store:
        migrate.run(store, _below(tmp_path, 42))
        _store(store, rows)
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

        report = migrate.run(store, _below(tmp_path, 43))
        store.commit()
        assert report.applied == (MIGRATION,), report.summary()
        after = _stored(store)

        # Run again by hand: a re-spelt store is already in spelling 31.
        with store.cursor() as cursor:
            cursor.execute(migrate._read(migrate.MIGRATIONS / MIGRATION))
        again = _stored(store)
        store.rollback()
        assert again == after

    assert after.keys() == before.keys()
    assert after.pop("000000000000") is None
    assert after.pop("ffffffffffff") == {}
    assert len(after) == len(rows)
    for key, spec in after.items():
        assert spec == dict(current_spelling(before[key]), schema_version=STAMP), key
        assert validate_spec(spec) == [], (key, validate_spec(spec))
        assert "piece" not in json.dumps(spec["rules"].get("portals", []))
        # Each row's name is its key, and is the name its old spelling had.
        assert spec_hash(spec) == key == spec_hash(before[key]), key


@pytest.mark.parametrize("odd", ["not last", "renamed", "a second stand-in", "a move that takes"])
def test_a_stored_portal_game_of_another_shape_stops_it(tmp_path: Path, odd: str) -> None:
    """A stand-in the normal form would not write back as it was: re-spelling the row would
    rename it, so the migration refuses, names the key, and writes nothing."""
    spec = _as_0041_left_it(_designed()[2])
    stand = spec["pieces"][-1]
    if odd == "not last":
        spec["pieces"] = [stand, *spec["pieces"][:-1]]
    elif odd == "renamed":
        stand["name"] = "door"
    elif odd == "a second stand-in":
        spec["pieces"].append({**copy.deepcopy(stand), "id": "V"})
        spec["rules"]["portals"].append({"piece": "V", "orange": "a4", "blue": "h5"})
    else:
        stand["moves"][0]["mode"] = "both"
    with _scratch_store() as store:
        migrate.run(store, _below(tmp_path, 42))
        _store(store, {"0123456789ab": spec, "ba9876543210": _as_0041_left_it(_designed()[0])})
        before = _stored(store)
        with pytest.raises(Exception, match="of another shape.*0123456789ab"):
            migrate.run(store, _below(tmp_path, 43))
        store.rollback()
        assert _stored(store) == before
