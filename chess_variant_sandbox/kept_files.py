"""Reading a kept file whose generation may be packed (D1, `docs/reference/previous-archives.md`).

A replaced measurement is kept under `runs/<hash>/previous/<n>/` (`history.keep`); an old
generation's bulky run files may be packed into `previous/<n>/packed.zip`
(`history.pack_generation`). These two readers answer for a kept file either way. They are
here, not in `history`, so that a reader of records (`measurement.records_digest`) reads a
packed run without importing the history, which no metric or trust rule may
(`tests/test_volatility.py`). Pure file reading: nothing here makes or moves a number.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PREVIOUS_DIR = "previous"
PACK_FILE = "packed.zip"


def generation_of(path: Path) -> Path | None:
    """The `previous/<n>/` directory a path lies in, or ``None``."""
    for parent in Path(path).parents:
        if parent.name.isdigit() and parent.parent.name == PREVIOUS_DIR:
            return parent
    return None


def member(generation: Path, path: Path) -> str:
    """A kept file's name inside its generation's archive."""
    return Path(path).relative_to(generation).as_posix()


def kept_bytes(path: Path) -> bytes | None:
    """A kept file's bytes: the file itself, or, where its generation is packed, its member
    of `packed.zip`. ``None`` when neither holds it."""
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    generation = generation_of(path)
    if generation is None or not (generation / PACK_FILE).is_file():
        return None
    with zipfile.ZipFile(generation / PACK_FILE) as archive:
        try:
            return archive.read(member(generation, path))
        except KeyError:
            return None


def kept_exists(path: Path) -> bool:
    """Whether a kept file is on disk, plain or packed."""
    path = Path(path)
    if path.is_file():
        return True
    generation = generation_of(path)
    if generation is None or not (generation / PACK_FILE).is_file():
        return False
    with zipfile.ZipFile(generation / PACK_FILE) as archive:
        return member(generation, path) in set(archive.namelist())
