"""Proven positions on the site: the fourth challenge before the ten (R9b, Q605).

`docs/plan-website.md`, *Proven positions*. R9a (`chess_variant_sandbox/puzzles.py`,
`docs/reference/proven-positions.md`) proved positions from the library's own recorded
games won for the side to move, offline, and wrote them as `runs/puzzles/<hash>.json`
(`puzzles.FORMAT` 4; 1 to 3 are still read). This module reads that record and offers each
position two ways:

- **Convert it**: a real game from the position against *hardest*, the strongest
  computer player every board offers. It is a practice match started from the record's
  move prefix (`matches.start(..., puzzle=...)`, source `puzzle`), and it succeeds when
  the game ends in the person's win -- the engine's own outcome of the game, read off the
  match row at its finish (`converted`). Nothing new judges it.
- **Find the move**, only where the proof shows a single winning move: the person picks
  a move and the answer is the stored score of that root move, looked up (`find`). A move
  whose proof is *open* is answered *not settled*, never *wrong*: only a move the proof
  found lost is called one that loses.

**Rule 3.** Python holds a position and asks the engine; it never answers. The loader
replays each prefix through the engine, asking at every step whether the move is legal,
and drops (and counts) a position whose prefix no longer replays, whose position has
ended, whose side to move is not the record's, or whose legal moves are not the record's
root moves -- a record proved under rules or a core that no longer hold. It never
re-judges one. A row the record proves *lost* for the side to move is passed over and not
counted as dropped (Q993i): it is a true proof, never a position to offer, and `dropped`
counts exactly what `puzzles.owes` re-proves a record for. *Find the move* compares a move
number with the record's numbers.

**A position's name is what it is** (Q993c, `docs/archive/plan-proven.md` §5.3). The loader names
each position it keeps by the state its replay reached (`puzzles.name`: the core's key and
what the key leaves out), not by the record's `seed-game-ply`, and a second row reaching a
position already named is dropped and counted. The page, the API, a try (`matches.puzzle`)
and a mark (`proven_marks.position`) all use that name; `0040_proven_names.sql` moved the
marks and tries stored before it.

**Too easy** (Q993a, Q993f, Q993g, `docs/archive/plan-proven.md` §3). The loader also leaves out,
and counts, a position the test calls trivial (`puzzles.trivial`): a win in one with more
than one winning move (T1), half the moves or more proved to win (T2), in a `FORMAT` 2 to 4
record T3's stored answer that nothing the defender does changes the win (`none`), in a
`FORMAT` 3 or 4 record T4's, that the defender can only make the loss come sooner
(`none`), and in a `FORMAT` 4 record T5's, that the computer's two quickest players played
a winning move (`found`). It reads the record's numbers and decides nothing; the basis line
says how many and why, and a game left with none offered says that in a sentence (`NONE_LEFT`).

**Its own game** (Q1084). A record sifted by `puzzles.refilter` or `puzzles.make` holds
only positions whose proven line shows something its game has and chess lacks; the rest are
kept in the record's `set_aside` and never offered. The basis line counts them, and a game
left with none says so (`NONE_SHOWN`).

**The seven tests** (`docs/reference/no-dark-patterns.md`). No timer and no streak; a
person's marks (`proven_marks`) are read on their own page alone and no count across
people is read anywhere; nothing expires; a failed try writes nothing and is a game the
person lost; and the record's basis is on the page (`basis_said`).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import ROOT, built_fingerprint, fold_crlf, tables_cache_path
from chess_variant_sandbox.puzzles import (
    FORMATS,
    LOST,
    OPEN,
    WON,
    basis_owes,
    certificate,
    holding,
    trivial,
    work_basis,
)
from chess_variant_sandbox.web import card_rules, library, opponents
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.page import game as game_drawing
from chess_variant_sandbox.web.page import game_picture, proven_page

DIRECTORY = "puzzles"
"""Where the records sit under `runs/`, as R9a writes them."""

PAGE_ROUTE = "GET /game/<hash>/proven"
"""The positions' addresses' row in `usage_hours` (G10): the list and each position, every
read a visitor's, since the page reads no cookie."""

PICTURE_ROUTE = "GET /picture/proven/<hash>"
"""A position's picture's row in `usage_hours` (G10)."""

SOURCE = "puzzle"
"""`matches.source` on a try at converting a proven position (`0037_proven_positions.sql`)."""

CONVERTED = "converted"
FOUND = "found"
WAYS = (CONVERTED, FOUND)

OPPONENT = next(agent for agent, name, _tail in opponents.RUNGS if name == "hardest")
"""*Convert it* is played against *hardest*, the ladder's strongest rung on every board
that alternates (`minimax:nodes=200000`). Its seat is rationed like any other
(`quota.check_new_match`), and it is for people with an account (L15)."""

NOT_HERE = "There is no proven position at this address on this game."
NO_TAKE_BACK = (
    "a game from a proven position has no take-back: it is decided by how the game ends. "
    "Start it again from the position instead"
)
SIGN_IN_TO_CONVERT = (
    "Sign in to convert a position: it is played against hardest, which is for people "
    "with an account. Finding the move needs no account."
)


# ------------------------------------------------------------------------ the record


@dataclass(frozen=True)
class Position:
    """One proved position, as the record holds it and the loader replayed it.

    `number` is its place in the order the page lists them, shortest proved win first;
    `id` is its name (`puzzles.name`, Q993c): a digest of the position the loader's replay
    reached, never the record's `seed-game-ply`, so the store's marks and tries, keyed by
    it, stay on the position whatever runs a later record is proved from. `scores` are the record's root moves in the engine's order, each with
    its `move`, `text` and `proof` (`won`, `lost` or `open`) and, when proved, `ends_in`.
    """

    id: str
    number: int
    moves: tuple[int, ...]
    side: int
    ends_in: int
    pieces: int
    single: bool
    others_lost: bool
    scores: tuple[dict[str, Any], ...]

    @property
    def ply(self) -> int:
        return len(self.moves)

    def score_of(self, move: int) -> dict[str, Any] | None:
        return next((entry for entry in self.scores if entry["move"] == move), None)

    def winning(self) -> list[dict[str, Any]]:
        return [entry for entry in self.scores if entry["proof"] == WON]


@dataclass(frozen=True)
class Record:
    """A game's proven positions, those still offered, and what the record says of itself."""

    game: str
    made: str
    basis: dict[str, Any]
    counts: dict[str, int]
    positions: tuple[Position, ...]
    dropped: int
    too_easy: int = 0
    """Positions that replay but the triviality test leaves out (T1 to T5)."""
    set_aside: int = 0
    """Proved positions the sift set aside (Q1084): their line showed nothing the game has
    and chess lacks. Kept in the record, never offered."""
    defence: bool = False
    """Whether the record carries T3's answers (`FORMAT` 2 or 3 with `defence_nodes`)."""
    hasten: bool = False
    """Whether the record carries T4's answers (`FORMAT` 3 or 4 with `hasten_nodes`)."""
    shallow: bool = False
    """Whether the record carries T5's answers (`FORMAT` 4 with `shallow_nodes`)."""
    provisional: str = ""
    """Why the record does not stand on the engine now running, or "" where it does
    (Q1090: `puzzles.basis_owes`, the record proved, or last re-certified, on another core
    build or compile of the rules, or by older code). Such a record is shown, marked
    `PROVISIONAL`, never as current, until it is re-certified or proved again."""
    certified: dict[str, Any] | None = None
    """The sample that re-certified the record on a core that did not prove it
    (`puzzles.certificate`), or None."""

    @property
    def tests(self) -> str:
        """The last part of the triviality test this record could apply: `T2` (T1 and T2,
        read off the proofs), `T3`, `T4` or `T5`, which names the basis line's clause."""
        if self.shallow:
            return "T5"
        return "T4" if self.hasten else "T3" if self.defence else "T2"

    def find(self, position_id: str) -> Position | None:
        return next((found for found in self.positions if found.id == position_id), None)

    @property
    def singles(self) -> int:
        return sum(1 for found in self.positions if found.single)


def path(runs_dir: Path | str, digest: str) -> Path:
    """Where a game's record is, whether or not it exists."""
    return Path(runs_dir) / DIRECTORY / f"{digest}.json"


_CACHE: dict[tuple[str, str], tuple[tuple[int, int, str], Record | None]] = {}
_LOCK = threading.Lock()
_FLIGHTS: dict[tuple[str, str], threading.Lock] = {}
"""One lock a record (Q1093, `docs/reference/library-cold-read.md` fix 1). The first reader
of a version `_CACHE` does not hold replays it; every reader arriving meanwhile waits on this
lock and takes that answer rather than replaying the record alongside. On 2026-11-02 eleven
library reads each replayed every stale record at once: 12 in flight, up to 357 s."""


def load(runs_dir: Path | str, digest: str, spec: dict[str, Any] | None) -> Record | None:
    """The game's record, its positions replayed through today's rules, or None.

    None where there is no record (every member's game), where the record proved nothing
    (the upkeep's empty record for a game with no small ending: three of the launch set),
    where it is not a format it reads (`FORMATS`) or not this game's, and where no
    spec is given. A
    record is read once a process for each version of the file on disk and these rules,
    since the replay costs a few tenths of a second on a record of 150, and readers asking
    for the same version at once share that one replay (`_FLIGHTS`).
    """
    if spec is None:
        return None
    where = path(runs_dir, digest)
    try:
        held = where.stat()
    except OSError:
        return None
    stamp = (held.st_mtime_ns, held.st_size, json.dumps(spec, sort_keys=True))
    key = (str(where), digest)
    with _LOCK:
        cached = _CACHE.get(key)
        flight = _FLIGHTS.setdefault(key, threading.Lock())
    if cached is not None and cached[0] == stamp:
        return cached[1]
    # Single flight: one replay a version of the record, however many readers ask at once.
    # A reader that waited here finds the first one's answer in the cache and returns it.
    with flight:
        with _LOCK:
            cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            body = where.read_bytes()
        except OSError:
            body = b""
        found = _replayed(body, digest, spec)
        with _LOCK:
            _CACHE[key] = (stamp, found)
        return found


def _replayed(body: bytes, digest: str, spec: dict[str, Any]) -> Record | None:
    """The record these bytes hold, replayed (`_read`), or None where they hold none."""
    try:
        data = json.loads(body.decode("utf-8"))
    except ValueError:
        data = None
    return _read(data, digest, spec) if isinstance(data, dict) else None


_COUNTED: dict[tuple[str, str], tuple[tuple[int, int, str], int]] = {}
"""`offered`'s answers this process took from the store (`ProofAnswers`) or worked out, by the
version of the record on disk they answer, as `_CACHE` keeps whole records."""


def _known(key: tuple[str, str], stamp: tuple[int, int, str]) -> int | None:
    """`offered`'s answer for this version of the record if this process has it; under `_LOCK`."""
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return 0 if cached[1] is None else len(cached[1].positions)
    counted = _COUNTED.get(key)
    return counted[1] if counted is not None and counted[0] == stamp else None


def offered(runs_dir: Path | str, digest: str, spec: dict[str, Any] | None) -> int:
    """How many proven positions the game's page offers: nought where it has no section.

    The games list's *proven positions* filter reads this (Q972), the same replay and the
    same positions the game page's section draws, so a card the filter keeps is exactly a
    game whose page has the section. A game with no record costs one `stat`.

    **Kept on disk** (Q1094, `docs/reference/library-cold-read.md` fix 2). The count is a
    pure function of the record's bytes, the spec and the code that replays it, so it is
    stored under exactly those (`ProofAnswers`, `proof_answers_key`) and a new process reads
    it back instead of replaying the record: a restart's library read hashes each record
    and replays only those whose bytes, spec or replaying code it has no answer for. A
    replay done here also leaves its `Record` in `_CACHE` for the game page. Readers asking
    for the same version at once share one replay or one lookup (`_FLIGHTS`).
    """
    if spec is None:
        return 0
    where = path(runs_dir, digest)
    try:
        held = where.stat()
    except OSError:
        return 0
    stamp = (held.st_mtime_ns, held.st_size, json.dumps(spec, sort_keys=True))
    key = (str(where), digest)
    with _LOCK:
        known = _known(key, stamp)
        flight = _FLIGHTS.setdefault(key, threading.Lock())
    if known is not None:
        return known
    with flight:
        with _LOCK:
            known = _known(key, stamp)
        if known is not None:
            return known
        try:
            body = where.read_bytes()
        except OSError:
            return 0
        record_digest = hashlib.sha256(body).hexdigest()
        spec_digest = hashlib.sha256(stamp[2].encode("utf-8")).hexdigest()
        answer = _PROOFS.get(digest, record_digest, spec_digest)
        if answer is None:
            found = _replayed(body, digest, spec)
            answer = 0 if found is None else len(found.positions)
            with _LOCK:
                _CACHE[key] = (stamp, found)
            _PROOFS.put(digest, record_digest, spec_digest, answer, _listing(found))
            _PROOFS.save()
        with _LOCK:
            _COUNTED[key] = (stamp, answer)
        return answer


Listing = tuple[str, tuple[str, ...]]
"""What the sitemap reads of a record (`listed`): the day it was made as the record says it,
and the offered positions' ids in the page's order."""


def _listing(found: Record | None) -> Listing | None:
    return None if found is None else (found.made, tuple(one.id for one in found.positions))


_LISTED: dict[tuple[str, str], tuple[tuple[int, int, str], Listing | None]] = {}
"""`listed`'s answers this process took from the store or from a replay, by record version."""


def listed(runs_dir: Path | str, digest: str, spec: dict[str, Any] | None) -> Listing | None:
    """The record's `made` and its offered positions' ids, or None where `load` is None.

    Exactly `load`'s record cut down to what the sitemap prints, and kept on disk beside
    `offered`'s count under the same key (C15f, `docs/plan-performance.md`): a process's
    first `/sitemap.xml` used to replay every record, 7,626 positions and 102 compiles, 31 s,
    and now reads the answers a library listing (`offered`) or an earlier process stored. A
    record the store has no listing for is replayed once (`load`) and its listing stored.
    """
    if spec is None:
        return None
    where = path(runs_dir, digest)
    try:
        held = where.stat()
    except OSError:
        return None
    stamp = (held.st_mtime_ns, held.st_size, json.dumps(spec, sort_keys=True))
    key = (str(where), digest)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return _listing(cached[1])
        known = _LISTED.get(key)
        if known is not None and known[0] == stamp:
            return known[1]
    try:
        body = where.read_bytes()
    except OSError:
        return None
    record_digest = hashlib.sha256(body).hexdigest()
    spec_digest = hashlib.sha256(stamp[2].encode("utf-8")).hexdigest()
    stored = _PROOFS.listing(digest, record_digest, spec_digest)
    if stored is not None:
        answer = stored[0]
    else:
        # These bytes, as `offered` replays them, so what is stored answers the digest it is
        # stored under; the replay is left for the game page as `offered` leaves one.
        found = _replayed(body, digest, spec)
        with _LOCK:
            _CACHE[key] = (stamp, found)
        answer = _listing(found)
        _PROOFS.put(digest, record_digest, spec_digest,
                    0 if found is None else len(found.positions), answer)
        _PROOFS.save()
    with _LOCK:
        _LISTED[key] = (stamp, answer)
    return answer


# ------------------------------------------------------- the replay's answers' store
#
# Q1094. Every cold `/api/library` replayed every proof record to count what `offered`
# says: 7,574 rows and 1.2 million plies stepped from Python, plus each game compiled
# again, about two thirds of the deploy's ~125 s warm on the box (`library-cold-read.md`).
# The count is kept on disk as Q887 keeps the cards' answers (`library.CardAnswers`).

PROOF_ANSWERS_NAME = "library-proven.json"
# Moves when the file's shape does, what a stored answer means, or how the key is read.
_PROOF_FORMAT = 2  # 2 (C15f): an entry carries `listed`, the sitemap's reading
_HERE = "chess_variant_sandbox/site/proven.py"
_PUZZLES = "chess_variant_sandbox/puzzles.py"


def proof_answers_path() -> Path:
    """Beside the card answers and the tables cache (`measurement.tables_cache_path`): on
    the box `/opt/chess-variant-sandbox/runs/`, the app user's and kept across a deploy's
    swap; in a test run the session's own directory, never the checkout's `runs/`."""
    return tables_cache_path().with_name(PROOF_ANSWERS_NAME)


def _disk(name: str) -> bytes:
    try:
        return (ROOT / name).read_bytes()
    except OSError:
        return b"<unreadable>"


def module_imports(body: bytes) -> bytes:
    """The import statements a module runs when it is imported, and none it defers into a
    function or a class: the source of every import outside a `def` or `class` body."""
    text = body.decode("utf-8")
    heads: list[str] = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Import, ast.ImportFrom)):
                heads.append(ast.get_source_segment(text, inner) or "")
    return "\n".join(heads).encode("utf-8")


@lru_cache(maxsize=1)
def proof_files() -> tuple[tuple[str, bytes], ...]:
    """Every file whose content can decide a stored count, by repository path, with its bytes.

    The count is `len(_read(...).positions)`: `_read`'s gate (`FORMATS`, the record's own
    hash), the engine compiled from the spec (`Engine`), `puzzles.holding` (which replays
    each won row and names the position it reaches) and `puzzles.trivial`. So the key holds

    - this file whole (`_read`, `_kept`, `offered`, this store);
    - `puzzles.py` whole;
    - the import closure of what `puzzles.py` imports when it is imported (`module_imports`),
      deferred imports in those files followed too: the engine, its compiler and bridge,
      `agents`, `tournament`, `spec`, `fairy` and `measurement`, 47 files on 2026-11-04.

    Left out on purpose: `puzzles.py`'s own deferred imports, which reach `web.library` and
    so the whole package, but sit in functions the count never calls (`spec.relate`,
    `corpus`, the record writer's `rules_beyond`); and this file's other imports (`web`, the
    page drawing), which `_read` never calls. `tests/test_site_proven.py` holds both: an AST
    walk from `holding` and `trivial` reaches no deferred import, the names `_read` and
    `_kept` call come only from the key's modules, and a traced replay runs no package file
    outside it. With the core's build fingerprint (`_core.BUILD_FINGERPRINT`, Rust) beside
    them, nothing that moves the count is outside the key: more would only replay again
    after an unrelated edit, fewer could serve a stale count."""
    found = {_HERE: _disk(_HERE), _PUZZLES: _disk(_PUZZLES)}
    entries = library.imported_paths(module_imports(found[_PUZZLES]))
    for name, body in library.package_files(entries):
        found.setdefault(name, body)
    return tuple(sorted(found.items()))


def proof_files_in(
    read: Callable[[str], bytes | None], listing: Callable[[str], frozenset[str]]
) -> tuple[tuple[str, bytes], ...]:
    """`proof_files` in any tree (Q1095): ``read`` gives a file's bytes by repository path,
    or ``None``, and ``listing`` the exact names of the files in a directory, as
    `library.card_files_in` takes them. `deploy/proven.py` reads a git ref's blobs through
    this, to know the key the box will run a ref under before it is deployed; over this
    checkout's disk it is `proof_files` (`tests/test_deploy_proven.py` holds the two equal)."""
    found = {name: read(name) or b"<unreadable>" for name in (_HERE, _PUZZLES)}
    entries = library.imported_paths(module_imports(found[_PUZZLES]), listing)
    for name, body in library.card_files_in(read, listing, (), entries=entries):
        found.setdefault(name, body)
    return tuple(sorted(found.items()))


def proof_key(core: str, files: Iterable[tuple[str, bytes]]) -> str:
    """SHA-256 over `_PROOF_FORMAT`, a core build fingerprint and ``files`` by repository
    path and bytes read LF: `proof_answers_key` for any core and tree. Folded CRLF to LF
    as `library.card_key` is, so a CRLF checkout keys its commit as `deploy/proven.py
    key_at` reads it from the blobs (Q1151)."""
    digest = hashlib.sha256(f"library-proven {_PROOF_FORMAT}\0core {core}\0".encode())
    for name, body in files:
        digest.update(name.encode("utf-8") + b"\0" + fold_crlf(body) + b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def proof_answers_key() -> str:
    """The key the stored counts are read and written under in this process: `proof_key`
    of the loaded core's build fingerprint and `proof_files`. Cached for the process:
    neither moves under it."""
    return proof_key(built_fingerprint(), proof_files())


class ProofAnswers:
    """`offered`'s count for each game's record, read from disk once and written back.

    The file holds one key's answers, ``{"sources": key, "proven": {game hash: {"record":
    sha256 of the record's bytes, "spec": sha256 of the spec's sorted JSON, "offered":
    count}}}``. A count is read back only when all three match: the code and core
    (`proof_answers_key`), the record's exact bytes (a digest, not the mtime, so a copy
    that changes nothing invalidates nothing) and the spec. One entry a game, so the file
    stays the library's size whatever the upkeep rewrites. Anything malformed, a partial
    write or another key reads as no answer, and the record is replayed.

    `save` writes to a temporary file renamed over the old one, after folding in what
    another process stored meanwhile under the same key. A `runs/` that cannot be written
    leaves the answers in memory: the worst a failure costs is a replay.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._fixed = path
        self._lock = threading.Lock()
        self._read_as: tuple[Path, str] | None = None
        self._proven: dict[str, dict[str, Any]] = {}
        self._added = False

    def _path(self) -> Path:
        return self._fixed if self._fixed is not None else proof_answers_path()

    @staticmethod
    def _valid(entry: Any) -> bool:
        return (isinstance(entry, dict) and isinstance(entry.get("record"), str)
                and isinstance(entry.get("spec"), str)
                and type(entry.get("offered")) is int and entry["offered"] >= 0)

    @classmethod
    def _stored(cls, path: Path, key: str) -> dict[str, dict[str, Any]]:
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(held, dict) or held.get("sources") != key:
            return {}
        proven = held.get("proven")
        if not isinstance(proven, dict):
            return {}
        return {str(game): dict(entry) for game, entry in proven.items() if cls._valid(entry)}

    def _current(self) -> dict[str, dict[str, Any]]:
        """The answers for this path and key, read at most once per pair; under the lock."""
        place = (self._path(), proof_answers_key())
        if self._read_as != place:
            self._proven = self._stored(*place)
            self._read_as = place
            self._added = False
        return self._proven

    def get(self, game: str, record: str, spec: str) -> int | None:
        with self._lock:
            entry = self._current().get(game)
        if entry is None or entry["record"] != record or entry["spec"] != spec:
            return None
        return int(entry["offered"])

    def listing(self, game: str, record: str, spec: str) -> tuple[Listing | None] | None:
        """`listed`'s answer stored for these bytes and spec, in a one-tuple (its answer can
        be None), or None where none is stored (an entry from before C15f has none)."""
        with self._lock:
            entry = self._current().get(game)
        if entry is None or entry["record"] != record or entry["spec"] != spec:
            return None
        if "listed" not in entry:
            return None
        held = entry["listed"]
        if held is None:
            return (None,)
        if (isinstance(held, dict) and isinstance(held.get("made"), str)
                and isinstance(held.get("ids"), list)
                and all(isinstance(one, str) for one in held["ids"])
                and len(held["ids"]) == entry["offered"]):
            return ((held["made"], tuple(held["ids"])),)
        return None

    def put(self, game: str, record: str, spec: str, offered: int,
            listed: Listing | None = None) -> None:
        with self._lock:
            self._current()[game] = {
                "record": record, "spec": spec, "offered": int(offered),
                "listed": None if listed is None
                else {"made": listed[0], "ids": list(listed[1])},
            }
            self._added = True

    def save(self) -> None:
        with self._lock:
            if not self._added or self._read_as is None:
                return
            path, key = self._read_as
            merged = {**self._stored(path, key), **self._proven}
            self._proven = merged
            payload = json.dumps({"sources": key, "proven": dict(sorted(merged.items()))},
                                 sort_keys=True, separators=(",", ":"))
            temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, path)
            except OSError:
                with contextlib.suppress(OSError):
                    temporary.unlink()
                return
            self._added = False


_PROOFS = ProofAnswers()


def _kept(
    engine: Engine, rows: list[Any]
) -> tuple[list[tuple[int, int, str, dict[str, Any]]], int, int]:
    """The rows a record offers, in the page's order, with how many were too easy and how
    many did not hold. Everything `offered`'s count is decided by past `_read`'s gate."""
    kept: list[tuple[int, int, str, dict[str, Any]]] = []
    too_easy = 0
    # The won rows that hold, by name (`puzzles.holding`, which `puzzles.owes` reads too). A
    # second row reaching a position already named is the same position twice: the first
    # is kept, as a record from the runs never holds one (`puzzles.candidates`). A row the
    # record proves lost is passed over and not dropped (Q993i): a true proof, never a
    # position to offer, and nothing the upkeep would re-prove.
    holds, dropped = holding(engine, rows)
    for order, called, row in holds:
        if trivial(row):
            too_easy += 1
            continue
        kept.append((int(row.get("ends_in") or 0), order, called, row))
    # Shortest proved win first, then the record's own order: a fixed order the server
    # decides, so the page sorts nothing.
    kept.sort(key=lambda entry: (entry[0], entry[1]))
    return kept, too_easy, dropped


def _read(data: dict[str, Any], digest: str, spec: dict[str, Any]) -> Record | None:
    if data.get("format") not in FORMATS or data.get("ruleset_hash") != digest:
        return None
    if not data.get("positions") and not data.get("set_aside"):
        # The empty record (Q993d): the upkeep asked, and the game's runs held no small
        # ending it could prove. It is there so the game is asked once; to a reader it is
        # no record, and the game has no section, as before it was asked. One whose every
        # proof was set aside (Q1084) is a record, which says why it offers none.
        return None
    # The engine is compiled here, not through `library.engine_for`, so the code that
    # decides the count is in `proof_files` (`Engine`'s closure), not `web/library.py`.
    engine = Engine(spec)
    kept, too_easy, dropped = _kept(engine, list(data.get("positions") or []))
    positions = tuple(
        Position(
            id=called,
            number=number,
            moves=tuple(int(move) for move in row["moves"]),
            side=int(row["side"]),
            ends_in=int(row.get("ends_in") or 0),
            pieces=int(row.get("pieces") or 0),
            single=bool(row.get("single")),
            others_lost=bool(row.get("others_lost")),
            scores=tuple(dict(entry) for entry in row["scores"]),
        )
        for number, (_ends, _order, called, row) in enumerate(kept, start=1)
    )
    return Record(
        game=digest,
        made=str(data.get("made") or ""),
        basis=dict(data.get("basis") or {}),
        counts={key: int(value) for key, value in (data.get("counts") or {}).items()},
        positions=positions,
        dropped=dropped,
        too_easy=too_easy,
        set_aside=len(data.get("set_aside") or []),
        defence=data.get("format") != 1 and bool((data.get("basis") or {}).get("defence_nodes")),
        hasten=data.get("format") not in (1, 2)
        and bool((data.get("basis") or {}).get("hasten_nodes")),
        shallow=data.get("format") not in (1, 2, 3)
        and bool((data.get("basis") or {}).get("shallow_nodes")),
        provisional=basis_owes(engine, data, work_basis(spec)) or "",
        certified=certificate(data),
    )


def state_of(spec: dict[str, Any], position: Position) -> Any:
    """The position itself, reached by replaying its prefix through the engine."""
    engine = engine_for(spec)
    return engine.play(engine.initial_state(), position.moves)


# --------------------------------------------------------------------- what is said


def _thousands(number: int) -> str:
    return f"{number:,}"


def _millions(number: int) -> str:
    if number % 1_000_000 == 0:
        return f"{number // 1_000_000} million"
    return _thousands(number)


def basis_said(record: Record) -> str:
    """The proof's basis, in a visitor's words (test 7): where the positions came from,
    how they were proved, how many were asked and proved, and when."""
    basis = record.basis
    counts = record.counts
    runs = len(basis.get("runs") or [])
    near = basis.get("near")
    pieces = basis.get("max_pieces")
    asked = counts.get("candidates", 0)
    won = counts.get("won", 0)
    lost = counts.get("lost", 0)
    unsettled = counts.get("open", 0)
    where = (
        f"Positions from the games the computer players played in this game's {runs} runs"
        if runs else "Positions from the games the computer players played in this game"
    )
    which = []
    if pieces:
        which.append(f"with {pieces} pieces or fewer left")
    near_min = basis.get("near_min")
    if near and isinstance(near_min, int) and near_min > 1:
        # The upkeep's sample (Q993d, `puzzles.NEAR_MIN`) skips the last plies of a game.
        which.append(f"from {near_min} to {near} turns before the end of a game one side "
                     "won, with the winner to move")
    elif near:
        which.append(f"within {near} turns of the end of a game one side won, with the "
                     "winner to move")
    said = where + (", " + ", ".join(which) if which else "") + ". "
    nodes = basis.get("nodes")
    said += (
        "The computer searched every move from each, "
        + (f"looking at up to {_millions(int(nodes))} positions, " if nodes else "")
        + "and a position is here only where that search found a forced win for the "
        "side to move. "
    )
    if asked:
        forced = counts.get("forced", 0)
        said += (
            f"Of {_thousands(asked)} positions asked it proved {_thousands(won)}"
            # The winner of the recorded game was to move, and still a proof can find them
            # lost: the other side missed the win later (Q993i). Said, so every one asked
            # is accounted for.
            + (f"; {_thousands(lost)} it proved lost for the side to move, who won the game "
               "played only because the other side then missed its win" if lost else "")
            + (f"; it could not settle {_thousands(unsettled)}" if unsettled else "")
            + (f", and {_thousands(forced)} had only one move to play" if forced else "")
            + ". "
        )
    if record.made:
        said += f"Proved on {record.made}."
    found = record.certified
    if found is not None and not record.provisional:
        # Q1090: kept current by a sample, as a library score is (C6), and said so.
        said += (
            f" Checked again on {found.get('made')}, on a later version of the computer's "
            f"engine, by proving {_thousands(int(found.get('sample') or 0))} of its "
            f"{_thousands(int(found.get('rows') or 0))} positions again: every one came out "
            "the same."
        )
    if record.too_easy:
        said += (f" Of the {_thousands(won)} it proved, " if won else " ") + (
            f"{_thousands(record.too_easy)} {TOO_EASY[record.tests]}")
    if record.set_aside:
        one = record.set_aside == 1
        said += (
            f" {_thousands(record.set_aside)} more {'was' if one else 'were'} set aside: "
            f"nothing in {'it' if one else 'them'}, from the pieces on the board to the moves "
            "of the win, is anything this game has that chess does not."
        )
    if record.dropped:
        said += (
            f" {record.dropped} more no longer replay under these rules, so they are "
            "not offered."
        )
    if record.provisional:
        # Q1090: last, so it closes the account of how they were proved.
        said += " " + PROVISIONAL
    return said


PROVISIONAL = (
    "These were proved on an earlier version of the computer's engine and have not yet been "
    "checked on the one now running, so they are provisional until they are."
)
"""What the page says of a record that does not stand on the engine now running (Q1090,
`Record.provisional`): shown, and marked, never as current."""

TOO_EASY = {
    "T5": (
        "were left out as too easy: the win was the very next move with more than one way "
        "to make it, half the moves or more won, nothing the defender could do put the win "
        "off, or a quick look found it: greedy or easy, two of the weakest computer players, "
        "played a winning move."
    ),
    "T4": (
        "were left out as too easy: the win was the very next move with more than one way "
        "to make it, half the moves or more won, or nothing the defender could do put the "
        "win off, only bring it sooner."
    ),
    "T3": (
        "were left out as too easy: the win was the very next move with more than one way "
        "to make it, half the moves or more won, or nothing the defender could do changed "
        "it."
    ),
    "T2": (
        "were left out as too easy: the win was the very next move with more than one way "
        "to make it, or half the moves or more won."
    ),
}
"""The basis line's clause for the positions the triviality test leaves out (test 7),
keyed by the last part the record could apply (`Record.tests`): it names only the tests
applied. T4's clause covers T3's, since a defender who cannot change the win cannot put it
off either; T5's adds the quick players, by their names on the site's list of players."""

NONE_LEFT = (
    "None to solve on this game. The computer proved some positions won here, and every "
    "one was too easy to set as a challenge; how they were proved, and why each was left "
    "out, is below."
)
"""What the game's page says where a record is there but offers no position (Q993g): the
reason in a sentence, never an empty list."""

NONE_SHOWN = (
    "None to solve on this game. The computer proved some positions won here, and each was "
    "either too easy to set as a challenge or showed nothing this game has that chess does "
    "not; how they were proved, and why each was left out, is below."
)
"""`NONE_LEFT` where the sift set some aside (Q1084): a game's positions must show what
makes it its own game, and this says that is why none is left."""


def none_said(record: Record) -> str:
    """Why a record offers no position: `NONE_SHOWN` where the sift set some aside, else
    `NONE_LEFT`."""
    return NONE_SHOWN if record.set_aside else NONE_LEFT


def _turns(count: int) -> str:
    return f"{count} turn{'' if count == 1 else 's'}"


TRY_SAID = {
    "said": (
        "A proven position: the computer proved that the side to move, which is you, wins "
        "from here with best play. Play it out against hardest. You convert it if the "
        "game ends in your win. There is no take-back and no clock."
    ),
    "won": "You converted it: the game ended in your win.",
    "lost": (
        "This try did not end in your win. It is a game you lost and nothing more: start "
        "it again from the position whenever you like."
    ),
}
"""What the board says on a try at converting a position: while it runs, and at each
ending. The page shows `won` where the engine's outcome names the person's side the
winner and `lost` at any other ending: it reads the outcome the board already draws, and
decides nothing itself."""


def try_snapshot(
    position_id: str, ply: int, side: int, converted: bool = False
) -> dict[str, Any]:
    """The `puzzle` block of a try's snapshot: which position, where its prefix ends, the
    person's side, the board's sentences, and whether the person holds the position's
    `converted` mark, as the finish left it (Q993b): the reply to the move that ends a
    won try carries the mark the finish wrote, so the positions' page ticks its row from
    the reply without asking again."""
    return {
        "id": position_id, "from": ply, "side": side, "converted": converted, **TRY_SAID,
    }


def converted_by(connection: Any, match_id: str) -> bool:
    """Whether the person at this try holds its position's `converted` mark: the row
    `converted` writes at a won finish, or one an earlier try wrote. Read for the try's
    snapshot (`matches._puzzle_block`)."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select exists (select 1 from proven_marks p join matches m"
            " on p.game = m.game and p.position = m.puzzle"
            " where m.id = %s and p.way = %s"
            " and p.person in (m.seat0_user, m.seat1_user))",
            (match_id, CONVERTED),
        )
        found = cursor.fetchone()
    return bool(found and found[0])


# ---------------------------------------------------------- the positions' own page

FOUND_FOR_VISIT = "found, for this visit"
"""A visitor's right pick, ticked for the visit and kept nowhere (§2.2 *A visitor*)."""

SOLVED_WORDS = {
    (True, False): "converted",
    (False, True): "found",
    (True, True): "converted and found",
}
"""A row's state on the right of the list, a word with an ink tick, never a colour (S4)."""

SINGLE = "one winning move"

PAGE_WORDS = {
    "play": "Play it out against hardest",
    "next": "next position",
    "nextShort": "next",
    "previous": "previous",
    "again": "try it again from the position",
    "playing": "playing",
    "single": SINGLE,
    "keep": "Sign in to keep what you solve.",
    "marked": "It is marked on your page.",
    "leave": (
        "Leave this game unfinished? It is a practice game from the position, and nothing "
        "is kept of a try left before its end."
    ),
    "empty": "No position on the list passes the filters that are on.",
    "alone": "No other position on the list passes the filters that are on.",
    "open": "open them",
    "find": "Find the move",
    "findSaid": "Only one move here is proved to win. Pick it.",
    "show": "show the proved move",
    "converted": SOLVED_WORDS[(True, False)],
    "found": SOLVED_WORDS[(False, True)],
    "both": SOLVED_WORDS[(True, True)],
    "visit": FOUND_FOR_VISIT,
    "gone": (
        "The position that address named is not offered here any more, so the list opens "
        "at its first."
    ),
    "copy": "copy link",
    "copyTitle": "copy this position's address, to send to somebody",
}
"""The positions' page's own words (Q993b, `docs/archive/plan-proven.md` §2.2), the server's, so
the guard test reads them: the view's one primary, which names its result (S6), the quiet
*previous* and *next*, the ended banner's *next position* and *try it again*, the line a
visitor reads above the list, a row's state, and the confirmation a try in progress asks
before it is left."""

CHIPS = (
    ("unsolved", "not solved yet"),
    ("single", SINGLE),
)
"""The chips over the list (§2.2): each narrows it in place, and *previous* and *next* walk
the list as narrowed. Their counts are the server's (`page`); a chip that would narrow
nothing (every position, or none, has one winning move) is not sent."""


def share_said(solved: int, of: int) -> str:
    """A share solved as a percentage (the user, 2026-10-20, `docs/archive/plan-proven.md` §4: *"each
    game's %"*). Rounded down, so *100%* is said only when every position is solved, and
    *under 1%* rather than *0%* for a share that is not nought."""
    if of <= 0:
        return ""
    whole = (100 * solved) // of
    if solved and whole == 0:
        return "under 1%"
    return f"{whole}%"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _heading(ends_in: int) -> str:
    return f"Wins in {_turns(ends_in)}"


def solved_ids(record: Record, marks: dict[str, set[str]] | None) -> set[str]:
    """The positions offered now that these marks solve, either way."""
    mine = marks or {}
    return {found.id for found in record.positions if mine.get(found.id)}


def progress_said(record: Record, marks: dict[str, set[str]]) -> str:
    """The progress line (§2.2), signed in: a count of the reader's own marks against what
    is offered now, with the share as a percentage. With no mark it is the count of
    positions alone, never *0 of n* (*absent is not nought*)."""
    count = len(record.positions)
    solved = solved_ids(record, marks)
    if not solved:
        return f"{_plural(count, 'position')} to solve."
    converted = sum(1 for found in record.positions if CONVERTED in marks.get(found.id, set()))
    found_ = sum(
        1 for found in record.positions if found.single and FOUND in marks.get(found.id, set())
    )
    said = (
        f"You have solved {len(solved)} of {count} ({share_said(len(solved), count)}): "
        f"converted {converted}"
    )
    if record.singles:
        said += (
            f", found the move in {found_} of the {record.singles} with one winning move"
        )
    return said + ". Only you see this."


def _visit_said(record: Record, found: set[str]) -> str:
    """A visitor's line: what they found on this visit, kept nowhere."""
    count = len(record.positions)
    if not found:
        return ""
    return (
        f"On this visit you have found the move in {len(found)} of {count} "
        f"({share_said(len(found), count)}). Nothing is kept."
    )


def page(
    record: Record,
    marks: dict[str, set[str]] | None,
    visit: set[str] | None = None,
) -> dict[str, Any]:
    """`GET /api/proven/<hash>`: the positions' own page (Q993b, `docs/archive/plan-proven.md` §2).

    The list in the loader's order (shortest proved win first), grouped by turns to the
    win, each row with the reader's state in words; the chips' counts; the progress line;
    the basis; and the page's own words. `marks` is the reader's own (`marks_on`), None for
    a visitor, whose right picks this visit (`visit`, ids the page holds and sends back)
    are counted for the visit and kept nowhere. Nothing here counts anybody else, and
    nothing here judges a move: every answer is the record's.
    """
    signed_in = marks is not None
    mine = marks or {}
    offered = {found.id for found in record.positions}
    seen = (set(visit or ()) & offered) if not signed_in else set()
    rows: list[dict[str, Any]] = []
    for found in record.positions:
        ways = mine.get(found.id, set())
        done = SOLVED_WORDS.get((CONVERTED in ways, FOUND in ways), "")
        if not signed_in and found.id in seen:
            done = FOUND_FOR_VISIT
        rows.append({
            "id": found.id,
            "number": found.number,
            "side": found.side,
            "endsIn": found.ends_in,
            "single": found.single,
            "converted": CONVERTED in ways,
            "found": FOUND in ways or found.id in seen,
            "solved": bool(done),
            "done": done,
        })
    groups: list[dict[str, Any]] = []
    for row in rows:
        if not groups or groups[-1]["endsIn"] != row["endsIn"]:
            groups.append({"endsIn": row["endsIn"], "heading": _heading(row["endsIn"]),
                           "ids": []})
        groups[-1]["ids"].append(row["id"])
    unsolved = sum(1 for row in rows if not row["solved"])
    solved = len(rows) - unsolved
    return {
        "game": record.game,
        "basis": basis_said(record),
        "count": len(rows),
        "singles": record.singles,
        "positions": rows,
        "groups": groups,
        "chips": [
            {"key": key, "label": label,
             "count": unsolved if key == "unsolved" else record.singles}
            for key, label in CHIPS
            if key == "unsolved" or 0 < record.singles < len(rows)
        ],
        "progress": progress_said(record, mine) if signed_in else _visit_said(record, seen),
        "share": (
            {"solved": solved, "of": len(rows), "said": share_said(solved, len(rows))}
            if rows and solved else None
        ),
        "signedIn": signed_in,
        "signInToConvert": "" if signed_in else SIGN_IN_TO_CONVERT,
        "keep": "" if signed_in else PAGE_WORDS["keep"],
        "none": "" if rows else none_said(record),
        "provisional": PROVISIONAL if record.provisional else "",
        "words": PAGE_WORDS,
    }


def summary(record: Record, marks: dict[str, set[str]] | None) -> dict[str, Any]:
    """*Proven positions* on the game's page since Q993b: the count, the reader's progress
    where there is any, and the way to the positions' own page. The list is that page's
    (`page`), so a game page no longer ships a row a position. A record that offers nothing
    keeps its sentence and its basis (`none_said`)."""
    count = len(record.positions)
    solved = solved_ids(record, marks)
    said = (
        f"{_plural(count, 'position')} from this game that the computer proved won for the "
        "side to move, to play out against hardest"
        + (", or to find the one move that wins" if record.singles else "")
        + ". There is no clock, and nothing here expires."
    ) if count else ""
    return {
        "count": count,
        "singles": record.singles,
        "said": said,
        "yours": progress_said(record, marks) if marks is not None and solved else "",
        "none": "" if count else none_said(record),
        "basis": "" if count else basis_said(record),
        "provisional": PROVISIONAL if record.provisional else "",
        "open": PAGE_WORDS["open"],
    }


# ------------------------------------------------------------------ one position


def view(record: Record, spec: dict[str, Any], position_id: str) -> dict[str, Any]:
    """One position, drawn: the still board, whose move it is, and, where the proof shows
    a single winning move, the moves to pick from -- each move and its words, never its
    score, which is the answer."""
    found = record.find(position_id)
    if found is None:
        raise KeyError(NOT_HERE)
    engine = engine_for(spec)
    state = state_of(spec, found)
    return {
        "id": found.id,
        "number": found.number,
        "side": found.side,
        "endsIn": found.ends_in,
        "single": found.single,
        "preview": card_rules.preview(engine, state, spec),
        "moves": (
            [{"move": entry["move"], "text": entry["text"]} for entry in found.scores]
            if found.single else []
        ),
        "said": (
            "The computer proved a win from "
            f"here in {_turns(found.ends_in)} with best play on both sides."
        ),
    }


def answer(record: Record, position_id: str, move: int | None, show: bool = False) -> dict[str, Any]:
    """*Find the move*: the stored proof of the move picked, looked up, in plain words.

    `proof` is the record's word for that root move. `right` is true only for the move the
    proof names; a move proved lost is said to lose, and an *open* one is said to be not
    the move the computer proved, which it could not settle either way -- never *wrong*.
    `show` gives the proved move without a pick, and marks nothing.
    """
    found = record.find(position_id)
    if found is None:
        raise KeyError(NOT_HERE)
    if not found.single:
        raise ValueError(
            "this position has more than one winning move, so it is offered to convert "
            "rather than to find the move"
        )
    best = found.winning()[0]
    if show:
        return {
            "proof": None,
            "right": False,
            "shown": best["text"],
            "said": (
                f"The move the computer proved wins is {best['text']}, in "
                f"{_turns(int(best.get('ends_in') or found.ends_in))} with best play."
            ),
        }
    entry = found.score_of(int(move)) if move is not None else None
    if entry is None:
        raise ValueError("that is not one of the moves in this position")
    proof = str(entry["proof"])
    text = str(entry["text"])
    if proof == WON:
        said = (
            f"Yes: {text} is the move the computer proved wins, in "
            f"{_turns(int(entry.get('ends_in') or found.ends_in))} with best play."
        )
    elif proof == LOST:
        said = f"No: the computer proved {text} loses."
        if found.others_lost:
            said += " Every move here but one loses."
    elif proof == OPEN:
        said = (
            f"{text} is not the move the computer proved. It could not settle this one "
            "either way, so it may win too, or it may not."
        )
    else:
        raise ValueError(f"the record says {proof!r} of that move, which is no proof")
    return {"proof": proof, "right": proof == WON, "move": text, "said": said}


# ------------------------------------------------------------------- a person's marks


def mark(
    connection: Any, person: int, game: str, position: str, way: str, match: str | None = None
) -> bool:
    """A person's solved mark, once. Whether it is new. Does not commit."""
    if way not in WAYS:
        raise ValueError(f"a mark is {CONVERTED} or {FOUND}, not {way!r}")
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into proven_marks (person, game, position, way, match)"
            " values (%s, %s, %s, %s, %s) on conflict do nothing",
            (person, game, position, way, match),
        )
        return bool(cursor.rowcount)


def converted(connection: Any, match_id: str) -> bool:
    """At a finish: mark the position converted if this try's game ended in the person's win.

    Called by `matches._finish` in the finish's own transaction, after the verdict is on
    the row. The verdict is the row's `result`, which is the engine's outcome of the game
    (or a resignation or an agreed draw, which are never the person's win). A visitor's
    try has nobody to mark, and a game nobody sat at is no try.
    """
    won = won_by(connection, match_id)
    if won is None:
        return False
    person, game, puzzle = won
    return mark(connection, person, game, puzzle, CONVERTED, match_id)


def won_by(connection: Any, match_id: str) -> tuple[int, str, str] | None:
    """Who won this try, on which game and position, or None where it is no person's win.

    The person, the game and the position (`matches.puzzle`) where the row is a try whose
    game the engine ended in the person's win; None for any other row or ending. What
    `converted` marks, and what tells the site whose proven-position badges to grant
    (`badges.grant_for_proof`).
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, puzzle, seat0_user, seat1_user, seat0_agent, seat1_agent, result"
            " from matches where id = %s",
            (match_id,),
        )
        found = cursor.fetchone()
    if found is None or found[1] is None or found[6] is None:
        return None
    game, puzzle, user0, user1, agent0, agent1, result = found
    if agent0 is None and user0 is not None and agent1 is not None:
        person, score = int(user0), float(result)
    elif agent1 is None and user1 is not None and agent0 is not None:
        person, score = int(user1), 1.0 - float(result)
    else:
        return None
    if score != 1.0:
        return None
    return person, str(game), str(puzzle)


def marks_on(connection: Any, person: int, game: str) -> dict[str, set[str]]:
    """This person's marks on one game: position id to the ways they solved it."""
    found: dict[str, set[str]] = {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select position, way from proven_marks where person = %s and game = %s",
            (person, game),
        )
        for position, way in cursor.fetchall():
            found.setdefault(str(position), set()).add(str(way))
    return found


def games_marked(connection: Any, person: int) -> dict[str, dict[str, set[str]]]:
    """Every game this person has a mark on, each with `marks_on`'s answer."""
    found: dict[str, dict[str, set[str]]] = {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, position, way from proven_marks where person = %s", (person,)
        )
        for game, position, way in cursor.fetchall():
            found.setdefault(str(game), {}).setdefault(str(position), set()).add(str(way))
    return found


def forget(connection: Any, person: int) -> None:
    """Leaving takes a person's marks with their address (`admin.forget`)."""
    with connection.cursor() as cursor:
        cursor.execute("delete from proven_marks where person = %s", (person,))


def profile_lines(
    marked: dict[str, dict[str, set[str]]],
    records: dict[str, Record | None],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    """The person's own page: on each game they have a mark on, how many of its positions
    they converted and how many moves they found, of how many it offers.

    Counted against the positions offered now, so a mark on a position a later record no
    longer offers is not counted. A game with no record left, or none it still offers,
    reads its marks alone.
    """
    lines = []
    for game in sorted(marked, key=lambda digest: (names.get(digest, "").lower(), digest)):
        record = records.get(game)
        if record is not None and not record.positions:
            record = None
        offered = {found.id for found in record.positions} if record is not None else None
        singles = (
            {found.id for found in record.positions if found.single}
            if record is not None else None
        )
        by_position = marked[game]
        done = [pid for pid, ways in by_position.items() if CONVERTED in ways]
        found_ = [pid for pid, ways in by_position.items() if FOUND in ways]
        if offered is not None:
            done = [pid for pid in done if pid in offered]
            found_ = [pid for pid in found_ if singles is not None and pid in singles]
        solved = len(set(done) | set(found_))
        lines.append({
            "hash": game,
            "name": names.get(game, ""),
            "converted": len(done),
            "found": len(found_),
            "of": None if record is None else len(record.positions),
            "singles": None if record is None else record.singles,
            # Q993b: each game's share solved, as a percentage, against what it offers now.
            "solved": None if record is None else solved,
            "share": None if record is None else share_said(solved, len(record.positions)),
        })
    return lines


def overall(
    marked: dict[str, dict[str, set[str]]], records: dict[str, Record | None]
) -> dict[str, Any] | None:
    """The share of every proven position on the site the person has solved, across every
    game that offers some (the user, 2026-10-20, `docs/archive/plan-proven.md` §4: *"aggregate milestone
    tracking % of proved positions of all games solved"*). Live: counted against what each
    game offers now, so a re-proof that adds positions moves it down and takes nothing kept
    away. None where nothing is offered or nothing is solved (*absent is not nought*)."""
    offered = solved = games = 0
    for game, record in records.items():
        if record is None or not record.positions:
            continue
        games += 1
        offered += len(record.positions)
        solved += len(solved_ids(record, marked.get(game)))
    if not offered or not solved:
        return None
    return {
        "solved": solved, "of": offered, "games": games, "share": share_said(solved, offered),
        "said": (
            f"Across all {_plural(games, 'game')} with proven positions you have solved "
            f"{solved} of {offered} ({share_said(solved, offered)})."
        ),
    }


def everywhere(runs_dir: Path | str, specs: dict[str, dict[str, Any]]) -> dict[str, Record | None]:
    """Every record on disk for a game the site knows, loaded (and cached) as its page
    loads it: the denominator of `overall`. A file for a game the site does not know is
    not read."""
    where = Path(runs_dir) / DIRECTORY
    try:
        names = sorted(item.stem for item in where.glob("*.json"))
    except OSError:
        return {}
    return {digest: load(runs_dir, digest, specs[digest]) for digest in names if digest in specs}


def seats_for(position: Position) -> list[str]:
    """The two seats of a try: the person on the side to move, *hardest* on the other."""
    from chess_variant_sandbox.web.human import HUMAN

    seats = [OPPONENT, OPPONENT]
    seats[position.side] = HUMAN
    return seats


# ------------------------------------------------- the positions' own addresses (G10)
#
# `docs/plan-growth.md` §2.5: each proven position has an address a crawler can read, a
# title, a description and a picture of its own, and a control to copy its address. The
# words say whose move it is and what to find, and never the answer: no move is named, and
# no count of solvers or name of one is anywhere on the page.


def side_said(sides: Sequence[str], side: int) -> str:
    """The side to move by the game's own name for it, capitalised, else by its place
    (`proven.js` `provenSide` says the same)."""
    named = sides[side] if 0 <= side < len(sides) else ""
    said = named or ("the first player" if side == 0 else "the second player")
    return said[:1].upper() + said[1:]


def position_task(position: Position) -> str:
    """What there is to do from a position, in three or four words."""
    return "find the winning move" if position.single else "play the win out"


def position_title(name: str, position: Position) -> str:
    """A position's title, without the site's name (*‹game›, position 3: find the winning
    move*): the plan's, with the position's place so no two of a game's share one."""
    return f"{name or 'An unnamed game'}, position {position.number}: {position_task(position)}"


def position_description(sides: Sequence[str], position: Position) -> str:
    """A position's line for a search result and a preview: whose move it is, that the
    computer proved a win, and what to find. Never the move."""
    side = side_said(sides, position.side)
    turns = _turns(position.ends_in)
    if position.single:
        return (f"{side} to move, and one move here wins, in {turns} with best play on both "
                "sides, as the computer proved. Find it free online in your browser.")
    return (f"{side} to move, and the computer proved a win from here in {turns} with best "
            "play on both sides. Play it out free online in your browser.")


def picture_words(sides: Sequence[str], position: Position) -> tuple[str, ...]:
    """What a position's picture says under the game's name: whose move, and what to find."""
    task = position_task(position)
    return f"{side_said(sides, position.side)} to move and win", task[:1].upper() + task[1:]


def drawing(name: str, spec: dict[str, Any], position: Position, sides: Sequence[str]) -> str:
    """A position's picture as SVG (`game_picture.svg`): the position itself, the core's
    replay of the record's prefix (`state_of`) put into the still board's payload
    (`card_rules.preview`), turned for the second player as the page turns it. Rule 3: the
    engine says where every piece stands; nothing here moves one."""
    engine = engine_for(spec)
    shape = card_rules.preview(engine, state_of(spec, position), spec)
    return game_picture.svg(name, shape, picture_words(sides, position),
                            flipped=position.side == 1)


def positions_title(name: str) -> str:
    """The list's title, as `proven.js` names the view."""
    return f"Proven positions on {name or 'an unnamed game'}"


def positions_description(name: str, record: Record) -> str:
    """The list's line for a search result and a preview."""
    count = len(record.positions)
    if not count:
        return game_drawing.first_sentence(none_said(record))
    what = ("find the winning move, or play the win out against the computer"
            if record.singles else "play each win out against the computer")
    # Q1068: *free online in your browser* ahead of what to do, so a long name cuts the
    # task short rather than the words a search result should carry.
    return game_drawing.first_sentence(
        f"{_plural(count, 'position')} from {name or 'an unnamed game'} that the computer "
        f"proved won for the side to move, free online in your browser: {what}.")


def _row_said(sides: Sequence[str], position: Position) -> str:
    said = f"Position {position.number}: {side_said(sides, position.side)} to move"
    return said + (f", {SINGLE}" if position.single else "")


def positions_drawn(
    record: Record, name: str, sides: Sequence[str], *, base: str, indexable: bool,
    picture: str = "", picture_alt: str = "",
) -> proven_page.Drawn:
    """What `/game/<hash>/proven` draws: the list, a link a position, by length of win.
    `picture` is the game's own picture where one is made."""
    groups: list[proven_page.Group] = []
    for found in record.positions:
        row = proven_page.Row(id=found.id, number=found.number,
                              said=_row_said(sides, found))
        heading = _heading(found.ends_in)
        if groups and groups[-1].heading == heading:
            groups[-1] = proven_page.Group(heading, (*groups[-1].rows, row))
        else:
            groups.append(proven_page.Group(heading, (row,)))
    said = summary(record, None)
    return proven_page.Drawn(
        hash=record.game,
        title=positions_title(name),
        description=positions_description(name, record),
        path=game_drawing.proven_path(record.game),
        indexable=indexable and bool(record.positions),
        heading=positions_title(name),
        back=game_drawing.path(record.game),
        back_said=name or "An unnamed game",
        said=tuple(line for line in (said["said"], said["none"]) if line),
        groups=tuple(groups),
        basis=basis_said(record),
        base=base,
        picture=picture,
        picture_alt=picture_alt or game_picture.alt(name),
    )


def position_drawn(
    record: Record, position: Position, name: str, sides: Sequence[str], *, base: str,
    indexable: bool, picture: str = "", picture_alt: str = "",
) -> proven_page.Drawn:
    """What `/game/<hash>/proven/<position>` draws: whose move, what to find, the proof's
    length and basis, and the way back to the list. `picture` is the position's own where
    one is made, else the game's (with `picture_alt` saying which), else nothing."""
    side = side_said(sides, position.side)
    task = (
        "Only one move here is proved to win: find it. Then play the win out against "
        "hardest, the strongest computer player."
        if position.single else
        "More than one move wins here. Play the win out against hardest, the strongest "
        "computer player."
    )
    return proven_page.Drawn(
        hash=record.game,
        title=position_title(name, position),
        description=position_description(sides, position),
        path=game_drawing.proven_path(record.game, position.id),
        indexable=indexable,
        heading=position_title(name, position)[:1].upper() + position_title(name, position)[1:],
        back=game_drawing.proven_path(record.game),
        back_said=positions_title(name),
        said=(
            f"{side} to move. The computer proved a win from here in "
            f"{_turns(position.ends_in)} with best play on both sides.",
            task,
        ),
        basis=basis_said(record),
        game=f"About {name or 'this game'}, and playing it from the start",
        base=base,
        picture=picture,
        picture_alt=picture_alt,
    )
