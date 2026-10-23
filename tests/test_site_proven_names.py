"""Q993c: proven positions named by what they are, and the store's marks and tries moved.

`docs/plan-proven.md` §5.3. A record names a position by where it was found (`seed-game-ply`)
and `0037` keyed a person's marks (`proven_marks.position`) and a try (`matches.puzzle`) by
that, so a re-proof from re-played runs could have handed a person's tick to another
position. The site now names a position by its replayed state (`puzzles.name`), and
`0040_proven_names.sql` renames what was stored, once, from a table written by replaying the
seven delivered records (`puzzles.renames`).

Proved here: the table is one to one within each game and is exactly the replay of the
records on disk (skipped where they are not, as the other record tests are; the gate re-runs
it where they are); and, in a scratch schema migrated to `0039` and filled as the live store
is shaped (people, marks of both ways on every game, the matches the converted marks name,
tries finished and not, ordinary matches, a member's game, a mark on an id no record holds),
`0040` renames every mark and every try on a listed id, loses none, and touches nothing else.
The store test needs a real Postgres and skips with a reason without one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from test_site_links import _scratch_store

from chess_variant_sandbox import puzzles
from chess_variant_sandbox.corpus import RUNS_DIR, find
from chess_variant_sandbox.site import migrate, proven
from chess_variant_sandbox.web.library import engine_for

NUMBER = 40
MIGRATION = "0040_proven_names.sql"
ROW = re.compile(r"^\s+\('([0-9a-f]{12})', '(\d+-\d+-\d+)', '([0-9a-f]{16})'\)[,;]?$")


def _table() -> dict[tuple[str, str], str]:
    """The migration's own table: `(game, old id)` to the new name."""
    text = (migrate.MIGRATIONS / MIGRATION).read_text(encoding="utf-8")
    found: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        match = ROW.match(line)
        if match is not None:
            game, old, new = match.groups()
            assert (game, old) not in found, (game, old)
            found[(game, old)] = new
    return found


TABLE = _table()
GAMES = sorted({game for game, _ in TABLE})


def test_the_migration_is_the_next_number_and_its_table_is_whole() -> None:
    names = {migration.number: migration.name for migration in migrate.available()}
    # The next number when it was written; `0041_one_spelling.sql` came after it.
    assert names[NUMBER] == MIGRATION and NUMBER - 1 in names
    assert len(TABLE) == 810 and len(GAMES) == 7
    for game in GAMES:
        news = [new for (held, _), new in TABLE.items() if held == game]
        assert len(news) == len(set(news)), f"{game}: two positions share a name"
        assert all(len(new) == puzzles.NAME_LENGTH for new in news)


@pytest.mark.parametrize("game", GAMES)
def test_the_table_is_the_replay_of_the_delivered_records(game: str) -> None:
    """Every position of each record, its old id and the name its replayed prefix reaches
    today, which is the name the loader gives it (`test_site_proven`'s delivered-record
    test): the migration and the site agree. Skips where the record is not on disk (the
    gate re-runs it there)."""
    path = Path(RUNS_DIR) / proven.DIRECTORY / f"{game}.json"
    if not path.exists():
        pytest.skip("no runs/puzzles/ record here")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "near_min" in (data.get("basis") or {}):
        # Proved again by the upkeep (`puzzles.make`, Q993d): not the record 0040 moved.
        pytest.skip(f"{game}: its record was proved again since the migration")
    pairs = puzzles.renames(engine_for(find(game).spec), data)
    assert len(pairs) == len(data["positions"]), f"{game}: a prefix no longer replays"
    assert dict(pairs) == {old: new for (held, old), new in TABLE.items() if held == game}


# --------------------------------------------------------------------- the store


def _store_before(store: Any, tmp_path: Path) -> None:
    older = tmp_path / "migrations-before-0040"
    older.mkdir()
    for migration in migrate.available():
        if migration.number < NUMBER:
            (older / migration.name).write_text(migration.sql, encoding="utf-8")
    migrate.run(store, older)
    store.commit()


def _fill(store: Any) -> dict[str, Any]:
    """The live store's shape, at `0039`: who marked what, which tries there are."""
    by_game: dict[str, list[str]] = {}
    for game, old in TABLE:
        by_game.setdefault(game, []).append(old)
    people: list[int] = []
    marks: list[tuple[int, str, str, str, str | None]] = []
    tries: list[tuple[str, str, str]] = []
    with store.cursor() as cursor:
        for handle in ("ada", "bo", "cy", "di"):
            cursor.execute("insert into users (handle, email) values (%s, %s) returning id",
                           (handle, f"{handle}@example.invalid"))
            people.append(cursor.fetchone()[0])
        for game in [*GAMES, "0123456789ab"]:
            spec = find(game).spec if game in GAMES else {"name": "a member's game"}
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (game, json.dumps(spec), spec["name"], "corpus" if game in GAMES else "designed"),
            )
            # An ordinary match on every game, which no rename may touch.
            cursor.execute(
                "insert into matches (id, game, mode, seat0_user, seat1_agent, seed)"
                " values (%s, %s, 'sandbox', %s, 'random', 1)",
                (f"plain-{game}", game, people[0]),
            )
        for number, game in enumerate(GAMES):
            ids = by_game[game]
            for seat, person in enumerate(people):
                # Each person on a few positions of each game, some both ways.
                for step in range(seat, len(ids), 37 + 5 * seat):
                    old = ids[step]
                    match = f"try-{game}-{person}-{step}"
                    finished = (step + number) % 3 != 0
                    cursor.execute(
                        "insert into matches (id, game, mode, seat0_user, seat1_agent, seed,"
                        " source, puzzle, puzzle_ply, result, reason, finished) values"
                        " (%s, %s, 'sandbox', %s, %s, 2, 'puzzle', %s, %s, %s, %s,"
                        " case when %s then now() end)",
                        (match, game, person, proven.OPPONENT, old, 10 + step % 7,
                         1.0 if finished else None, "royal_captured" if finished else None,
                         finished),
                    )
                    tries.append((match, game, old))
                    if finished:
                        marks.append((person, game, old, proven.CONVERTED, match))
                    if step % 2 == 0:
                        marks.append((person, game, old, proven.FOUND, None))
        # A mark on an id no record holds: kept as it is, and counted by nothing.
        marks.append((people[1], GAMES[0], "9-9-9", proven.FOUND, None))
        for mark in marks:
            cursor.execute(
                "insert into proven_marks (person, game, position, way, match)"
                " values (%s, %s, %s, %s, %s)", mark,
            )
    store.commit()
    return {"marks": marks, "tries": tries}


def test_the_migration_renames_every_mark_and_try_and_loses_none(tmp_path: Path) -> None:
    with _scratch_store() as store:
        _store_before(store, tmp_path)
        before = _fill(store)
        with store.cursor() as cursor:
            cursor.execute("select count(*) from matches")
            matches_before = cursor.fetchone()[0]
            cursor.execute("select id, puzzle_ply, result from matches order by id")
            columns_before = cursor.fetchall()

        report = migrate.run(store)
        store.commit()
        assert report.applied[0] == MIGRATION, report.summary()

        with store.cursor() as cursor:
            cursor.execute("select person, game, position, way, match, marked from proven_marks")
            marks_after = cursor.fetchall()
            cursor.execute("select id, game, puzzle from matches")
            tries_after = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
            cursor.execute("select id, puzzle_ply, result from matches order by id")
            columns_after = cursor.fetchall()
            cursor.execute("select to_regclass('proven_names')")
            leftover = cursor.fetchone()[0]

    expected = {
        (person, game, TABLE.get((game, old), old), way, match)
        for person, game, old, way, match in before["marks"]
    }
    assert len(before["marks"]) > 50 and len(before["tries"]) > 40
    assert len(marks_after) == len(before["marks"]), "a mark was lost"
    assert {tuple(row[:5]) for row in marks_after} == expected
    assert all(row[5] is not None for row in marks_after)
    renamed = [row for row in marks_after if row[2] != "9-9-9"]
    assert all(len(row[2]) == puzzles.NAME_LENGTH for row in renamed)
    assert (before["marks"][-1][0], GAMES[0], "9-9-9") in {tuple(row[:3]) for row in marks_after}
    assert {proven.CONVERTED, proven.FOUND} == {row[3] for row in marks_after}
    assert {row[1] for row in marks_after} == set(GAMES)

    assert len(tries_after) == matches_before
    for match, game, old in before["tries"]:
        assert tries_after[match] == (game, TABLE[(game, old)]), match
    for game in [*GAMES, "0123456789ab"]:
        assert tries_after[f"plain-{game}"] == (game, None)
    # Only the names moved: every match's other columns are as they were, and the
    # migration's working table went with its transaction.
    assert columns_after == columns_before
    assert leftover is None
