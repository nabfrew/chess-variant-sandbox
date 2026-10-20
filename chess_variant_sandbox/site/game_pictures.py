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

Memo = tuple[str, str, str]
"""A game as a picture sees it: its hash, its name and the digest of its rules."""


def memo_of(digest: str, game: library.Game) -> Memo:
    rules = hashlib.sha256(json.dumps(game.spec, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, game.name, rules


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
        self._queue: queue.Queue[tuple[Memo, library.Game]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._listing: object | None = None

    def file(self, digest: str, key: str) -> Path:
        return self.directory / game_picture.file_name(digest, key)

    def address(self, digest: str, game: library.Game) -> str | None:
        """The path of the game's picture where it is on disk; else `None` (the page names
        the site's picture), having asked for it. A lookup, never a drawing."""
        if not self.can_draw:
            return None
        memo = memo_of(digest, game)
        with self._lock:
            known, key = memo in self._keys, self._keys.get(memo)
        if known and key is None:
            return None
        if key is not None and self.file(digest, key).is_file():
            return game_picture.path(digest, key)
        self._want(memo, game)
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

    def _want(self, memo: Memo, game: library.Game) -> None:
        with self._lock:
            if memo in self._pending or memo in self._failed:
                return
            self._pending.add(memo)
            self._queue.put((memo, game))
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, name="game-pictures", daemon=True)
                self._worker.start()

    def _work(self) -> None:
        """Make what is asked for, one at a time, and end after `IDLE_SECONDS` with nothing
        asked: a site holds no thread for pictures while it has none to make."""
        while True:
            try:
                memo, game = self._queue.get(timeout=IDLE_SECONDS)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            try:
                self._make(memo, game)
            except Exception as error:  # noqa: BLE001 - the site's picture stands in
                with self._lock:
                    self._failed.add(memo)
                _say(f"no picture for {memo[0]}: {type(error).__name__}: {error}")
            finally:
                with self._lock:
                    self._pending.discard(memo)
                self._queue.task_done()

    def _make(self, memo: Memo, game: library.Game) -> None:
        """Work out the game's drawing and its key, and make the file if it is not there:
        a line on the log says what each made picture cost this thread, which is the box's
        own reading of a picture's CPU (`docs/plan-growth.md` G6)."""
        digest = memo[0]
        began = time.thread_time()
        shape = library.opening(game)
        if shape is None:
            self._remember(memo, None)
            _say(f"no picture for {digest}: its rules give no still board")
            return
        drawn = game_picture.svg(game.name, shape, drawing.board_words(game.spec))
        key = game_picture.key(drawn)
        name = game_picture.file_name(digest, key)
        if not (self.directory / name).is_file():
            made = self.make(drawn)
            self.directory.mkdir(parents=True, exist_ok=True)
            scratch = self.directory / f".{name}.{os.getpid()}.part"
            scratch.write_bytes(made)
            os.replace(scratch, self.directory / name)
            for old in self.directory.glob(f"{digest}-*.png"):
                if old.name != name:
                    old.unlink(missing_ok=True)
            _say(f"made {name} in {time.thread_time() - began:.3f} CPU-s, {len(made)} bytes")
        self._remember(memo, key)

    def _remember(self, memo: Memo, key: str | None) -> None:
        with self._lock:
            if len(self._keys) >= MEMO_MOST:
                self._keys.clear()
            self._keys[memo] = key


def _say(line: str) -> None:
    print(f"game pictures: {line}", file=sys.stderr, flush=True)
