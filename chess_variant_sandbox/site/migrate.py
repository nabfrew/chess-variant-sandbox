"""Numbered SQL files, applied in order, one transaction each, recorded when they run.

**A migration is never edited once it has been applied.** The database cannot re-read a
file it has already run, so an edit means the schema on the box and the schema in the
repository have quietly parted. The runner makes that loud instead. It records a checksum
beside every applied file and refuses to do anything at all when a file on disk no longer
matches what ran. The repair is a new numbered file, always.

Postgres runs DDL inside transactions, so a migration that fails part way leaves nothing
behind. Each file gets its own transaction rather than the whole run getting one, because
a run that stops half way should keep the files that succeeded -- the next run then has
less to do, not the same amount.

Checksums are taken over text read with universal newlines, so a checkout with CRLF
endings and one with LF agree. A hash that disagreed by line ending would report drift on
every Windows checkout and teach people to ignore the refusal.

No schema is named, here or in the files, so everything lands wherever ``search_path``
points. That is what lets a test run a whole migration inside a scratch schema it drops.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIGRATIONS = Path(__file__).parent / "migrations"

# What the runner writes to remember itself. Created by `run`, never by a migration,
# because the runner has to read it before the first file has run.
LEDGER = "schema_migrations"

LEDGER_DDL = f"""
create table if not exists {LEDGER} (
    name     text primary key,
    checksum text not null,
    applied  timestamptz not null default now()
)
"""

FILENAME = re.compile(r"^(\d{4})_[a-z0-9]+(?:_[a-z0-9]+)*\.sql$")


class MigrationError(ValueError):
    """The migrations on disk and the ones in the database do not agree.

    A `ValueError` like `db.ConfigError`, because `cli.main` prints those as one line and
    exits 2. Every case is something a person did to the files, not a fault.
    """


@dataclass(frozen=True)
class Migration:
    """One numbered file, its text and the checksum that pins it."""

    number: int
    name: str
    sql: str
    checksum: str


@dataclass(frozen=True)
class Report:
    """What a run did, or what a dry run would do."""

    applied: tuple[str, ...]
    pending: tuple[str, ...]
    already: tuple[str, ...]
    dry_run: bool = False

    def summary(self) -> str:
        """One line, and it says outright when nothing changed."""
        if self.dry_run:
            if not self.pending:
                return f"nothing to apply; {_count(self.already)} already recorded"
            return f"would apply {_count(self.pending)}: {', '.join(self.pending)}"
        if not self.applied:
            return (
                f"nothing to apply; {_count(self.already)} already recorded "
                f"and no rows changed"
            )
        return f"applied {_count(self.applied)}: {', '.join(self.applied)}"


def _count(names: tuple[str, ...]) -> str:
    return f"{len(names)} migration{'' if len(names) == 1 else 's'}"


def _read(path: Path) -> str:
    """The file's text with line endings normalised.

    ``newline=None`` is the universal-newlines default and is written out because the
    checksum depends on it: git hands this repository CRLF on Windows and LF elsewhere,
    and the two checkouts have to agree on what a migration hashes to.
    """
    with path.open("r", encoding="utf-8", newline=None) as handle:
        return handle.read()


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def available(directory: Path = MIGRATIONS) -> tuple[Migration, ...]:
    """Every migration on disk, in number order.

    A file that does not read as `NNNN_name.sql` is refused rather than skipped, because
    a migration nobody notices is worse than one that stops the run. Two files sharing a
    number is the shape two worktrees produce, so it is refused by name.
    """
    found: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = FILENAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"{path.name} is not a migration name; "
                f"migrations are numbered like 0001_store.sql"
            )
        number = int(match.group(1))
        if number in found:
            raise MigrationError(
                f"{path.name} and {found[number].name} share the number {number:04d}; "
                f"renumber one of them"
            )
        text = _read(path)
        found[number] = Migration(number, path.name, text, _checksum(text))
    return tuple(found[number] for number in sorted(found))


def recorded(connection: Any) -> dict[str, str]:
    """Which migrations the database says have run, name to checksum.

    Empty when the ledger does not exist yet, which is an empty database rather than a
    problem. `to_regclass` answers through `search_path`, so a scratch schema sees its
    own ledger and not another one.
    """
    with connection.cursor() as cursor:
        cursor.execute("select to_regclass(%s)", (LEDGER,))
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return {}
        cursor.execute(f"select name, checksum from {LEDGER}")
        return {name: checksum for name, checksum in cursor.fetchall()}


def drift(files: tuple[Migration, ...], have: dict[str, str]) -> str | None:
    """The reason the two disagree, or None when they do not.

    Three ways to disagree. An applied file was edited, an applied file was deleted, or a
    file was slipped in under a number that has already been passed. All three mean the
    schema in the database is not the one the repository describes.
    """
    by_name = {migration.name: migration for migration in files}
    for name, checksum in sorted(have.items()):
        migration = by_name.get(name)
        if migration is None:
            return (
                f"{name} has been applied to this database but is not in "
                f"{MIGRATIONS.name}/; a migration is never removed once applied"
            )
        if migration.checksum != checksum:
            return (
                f"{name} has changed since it was applied ({checksum[:12]} became "
                f"{migration.checksum[:12]}); a migration is never edited once applied, "
                f"so put the change in a new numbered file"
            )
    highest = max((by_name[name].number for name in have), default=-1)
    for migration in files:
        if migration.name not in have and migration.number < highest:
            return (
                f"{migration.name} has not been applied but a higher-numbered migration "
                f"has; renumber it above {highest:04d} so it runs in order"
            )
    return None


def run(connection: Any, directory: Path = MIGRATIONS, dry_run: bool = False) -> Report:
    """Apply what has not run, and say what happened.

    The drift check comes before anything is written, so a database whose files have been
    edited is left exactly as it was. A dry run writes nothing at all, the ledger
    included -- it is asked on a database somebody is deciding whether to touch.
    """
    files = available(directory)
    have = recorded(connection)
    reason = drift(files, have)
    if reason is not None:
        raise MigrationError(reason)

    pending = tuple(migration for migration in files if migration.name not in have)
    already = tuple(migration.name for migration in files if migration.name in have)
    if dry_run:
        return Report((), tuple(m.name for m in pending), already, dry_run=True)

    if pending:
        _apply(connection, LEDGER_DDL, None)
        for migration in pending:
            _apply(connection, migration.sql, (migration.name, migration.checksum))
    return Report(tuple(m.name for m in pending), (), already)


def _apply(connection: Any, sql: str, record: tuple[str, str] | None) -> None:
    """One statement group and its ledger row, committed together or not at all."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            if record is not None:
                cursor.execute(
                    f"insert into {LEDGER} (name, checksum) values (%s, %s)", record
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
