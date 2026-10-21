"""Each game's link-preview picture on the site: made once, kept on disk, served for a year
(G6 of `docs/plan-growth.md`, §2.2).

`web/page/game_picture.py` draws; this decides when, and keeps what it drew. A picture is
the file `runs/pictures/<hash>-<key>.png`, its key the digest of its drawing
(`game_picture.key`), so it is made once per drawing: once per game and card key, since the
card's still board is what the drawing is drawn from, and again only when the drawing moves
(a rename, an icon, a token, a face, a new renderer). The directory is the app's, like the
rest of `runs/`, and nothing else writes it.

- **A game's own address names its picture only once the file is there**
  (`Pictures.address`). Until then it names the site's picture, never a broken link, and
  asks for the game's picture. Everything a picture costs -- the still board, the drawing,
  the PNG -- is paid on one thread of its own, one game at a time, so a page read never
  waits for one (it pays a lookup) and a burst of new games is a queue rather than a crowd.
  The next read of the page names the picture.
- **`/game` asks for every listed game's picture** once per listing (`Pictures.want_all`),
  so the deploy's warm read of `/game` draws the library's pictures before anybody shares
  one.
- **A picture is drawn from what the page is drawn from**: the game as `SiteApp.game` reads
  it and its still board (`library.opening`), the core's own opening position.
- **A game that cannot be drawn** (no resvg in this venv, rules that do not compile to a
  still board, an error in the renderer) stays on the site's picture for the life of the
  process and is said once on the log; it is not tried again until a restart.
- When a game's picture is written, the game's older pictures are removed: an unfurler that
  fetched an old address has its copy, and the page names only the newest.

The route (`SiteApp.game_picture`) answers only a game a visitor can open, and only a file
this directory holds: a killed or deleted game's picture is the 404 its page is.

**A proven position's picture** (G10) is made the same way on the same thread: its own
position, the core's replay of the record's prefix (`proven.drawing`), drawn as a game's
starting position is, into `runs/pictures/proven/<hash>/<position>-<key>.png`. The key is the
drawing's digest again, so a position is drawn once per content name (`puzzles.name`) and
again only when what the picture shows moves. It is asked for when the position's address
or its drawing (`GET /api/proven/<hash>/<position>`) is read, never for a whole list, and
`SiteApp.proven_picture` serves it only while the position is offered.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import game as drawing
from chess_variant_sandbox.web.page import game_picture

DIRECTORY = "pictures"
"""The pictures' directory under the app's `runs/`."""

ROUTE = "GET /picture/game/<hash>"
"""A picture's row in `usage_hours`: what serving one costs, counted like the page."""

MEMO_MOST = 4096
"""How many games' keys are remembered before the memory starts again."""

IDLE_SECONDS = 60.0
"""How long the drawing thread waits for work before it ends; the next request starts one."""

Memo = tuple[str, ...]
"""What a picture is drawn from, as the memory of keys knows it: for a game its hash, its
name and the digest of its rules; for a position (G10) those and the position's name and
the words its picture carries."""


def rules_digest(game: library.Game) -> str:
    return hashlib.sha256(json.dumps(game.spec, sort_keys=True).encode("utf-8")).hexdigest()


def memo_of(digest: str, game: library.Game) -> Memo:
    return digest, game.name, rules_digest(game)


@dataclass(frozen=True)
class Wanted:
    """One picture to look up or make: what it is drawn from (`memo`), where its files are
    under the directory (`stem`, a key short of a file name), how to draw it (the SVG, or
    None where it cannot be drawn), and its address for a key."""

    memo: Memo
    stem: str
    draw: Callable[[], str | None]
    address: Callable[[str], str]


def _game_drawing(game: library.Game) -> str | None:
    shape = library.opening(game)
    if shape is None:
        return None
    return game_picture.svg(game.name, shape, drawing.board_words(game.spec))


class Pictures:
    """The pictures one site holds, and the one thread that makes them. Thread-safe."""

    def __init__(
        self,
        directory: Path,
        can_draw: bool | None = None,
        make: Callable[[str], bytes] = game_picture.png,
    ):
        self.directory = directory
        self.can_draw = game_picture.available() if can_draw is None else can_draw
        self.make = make
        self._lock = threading.Lock()
        # Each game's picture key, or None where it has no picture; absent until worked out.
        self._keys: dict[Memo, str | None] = {}
        self._pending: set[Memo] = set()
        self._failed: set[Memo] = set()
        self._queue: queue.Queue[Wanted] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._listing: object | None = None

    def file(self, digest: str, key: str) -> Path:
        return self.directory / game_picture.file_name(digest, key)

    def position_file(self, digest: str, position: str, key: str) -> Path:
        return self.directory / game_picture.position_file_name(digest, position, key)

    def address(self, digest: str, game: library.Game) -> str | None:
        """The path of the game's picture where it is on disk; else `None` (the page names
        the site's picture), having asked for it. A lookup, never a drawing."""
        return self._address(Wanted(
            memo=memo_of(digest, game), stem=digest,
            draw=lambda: _game_drawing(game),
            address=lambda key: game_picture.path(digest, key),
        ))

    def position_address(
        self, digest: str, game: library.Game, position: str, words: tuple[str, ...],
        draw: Callable[[], str | None],
    ) -> str | None:
        """G10: the path of a proven position's picture where it is on disk; else `None`
        (its page names the game's picture), having asked for it. `words` are what the
        picture says beside the game's name, part of what it is drawn from, and `draw` makes
        its SVG from the core's replay (`proven.drawing`). A lookup, never a drawing."""
        if not game_picture.POSITION_NAME.match(position):
            return None
        return self._address(Wanted(
            memo=("proven", *memo_of(digest, game), position, *words),
            stem=f"proven/{digest}/{position}", draw=draw,
            address=lambda key: game_picture.position_path(digest, position, key),
        ))

    def _address(self, wanted: Wanted) -> str | None:
        if not self.can_draw:
            return None
        with self._lock:
            known, key = wanted.memo in self._keys, self._keys.get(wanted.memo)
        if known and key is None:
            return None
        if key is not None and (self.directory / f"{wanted.stem}-{key}.png").is_file():
            return wanted.address(key)
        self._want(wanted)
        return None

    def want_all(self, listing: object, games: Iterable[tuple[str, library.Game]]) -> None:
        """Ask for every game's picture, once per `listing` (the listed games' tuple, which
        `ListedCache` keeps for its five minutes)."""
        with self._lock:
            if listing is self._listing:
                return
            self._listing = listing
        for digest, game in games:
            self.address(digest, game)

    def served(self, digest: str, key: str) -> bytes | None:
        """A picture's bytes, or `None` where this directory holds no such file."""
        try:
            return self.file(digest, key).read_bytes()
        except OSError:
            return None

    def position_served(self, digest: str, position: str, key: str) -> bytes | None:
        """A position's picture's bytes (G10), or `None` where there is no such file."""
        try:
            return self.position_file(digest, position, key).read_bytes()
        except OSError:
            return None

    def settle(self) -> None:
        """Wait until every picture asked for is made or has failed."""
        self._queue.join()

    def held(self) -> int:
        """How many pictures the directory holds."""
        try:
            return sum(1 for _ in self.directory.glob("*.png"))
        except OSError:
            return 0

    # ----------------------------------------------------------------- the one thread

    def _want(self, wanted: Wanted) -> None:
        with self._lock:
            if wanted.memo in self._pending or wanted.memo in self._failed:
                return
            self._pending.add(wanted.memo)
            self._queue.put(wanted)
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, name="game-pictures", daemon=True)
                self._worker.start()

    def _work(self) -> None:
        """Make what is asked for, one at a time, and end after `IDLE_SECONDS` with nothing
        asked: a site holds no thread for pictures while it has none to make."""
        while True:
            try:
                wanted = self._queue.get(timeout=IDLE_SECONDS)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            try:
                self._make(wanted)
            except Exception as error:  # noqa: BLE001 - the site's picture stands in
                with self._lock:
                    self._failed.add(wanted.memo)
                _say(f"no picture for {wanted.stem}: {type(error).__name__}: {error}")
            finally:
                with self._lock:
                    self._pending.discard(wanted.memo)
                self._queue.task_done()

    def _make(self, wanted: Wanted) -> None:
        """Work out the picture's drawing and its key, and make the file if it is not there:
        a line on the log says what each made picture cost this thread, which is the box's
        own reading of a picture's CPU (`docs/plan-growth.md` G6)."""
        began = time.thread_time()
        drawn = wanted.draw()
        if drawn is None:
            self._remember(wanted.memo, None)
            _say(f"no picture for {wanted.stem}: its rules give no still board")
            return
        key = game_picture.key(drawn)
        target = self.directory / f"{wanted.stem}-{key}.png"
        if not target.is_file():
            made = self.make(drawn)
            target.parent.mkdir(parents=True, exist_ok=True)
            scratch = target.parent / f".{target.name}.{os.getpid()}.part"
            scratch.write_bytes(made)
            os.replace(scratch, target)
            stem = wanted.stem.rsplit("/", 1)[-1]
            for old in target.parent.glob(f"{stem}-*.png"):
                if old.name != target.name:
                    old.unlink(missing_ok=True)
            _say(f"made {wanted.stem}-{key}.png in {time.thread_time() - began:.3f} CPU-s, "
                 f"{len(made)} bytes")
        self._remember(wanted.memo, key)

    def _remember(self, memo: Memo, key: str | None) -> None:
        with self._lock:
            if len(self._keys) >= MEMO_MOST:
                self._keys.clear()
            self._keys[memo] = key


def _say(line: str) -> None:
    print(f"game pictures: {line}", file=sys.stderr, flush=True)
