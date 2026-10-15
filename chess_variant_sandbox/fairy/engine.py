"""The Fairy-Stockfish process, and how to find one.

**Nothing is bundled and nothing is downloaded.** The binary is discovered on the
machine or the opponent is simply not offered, which is why `web/opponents.py` can call
`fairy_stockfish_for` unconditionally and get an empty list. A research harness that
fetched and ran an executable to fill in a menu would be making a decision that is not
its to make, and a missing opponent is a far better failure than a surprising one.

Discovery, in order:

1. ``CHESS_VARIANT_SANDBOX_FAIRY_STOCKFISH`` -- a path to the binary. The escape hatch, and the one
   that works when the engine lives somewhere odd or is a wrapper script.
2. the usual names on ``PATH``: ``fairy-stockfish``, ``fairy-stockfish-largeboard``,
   ``fairystockfish``, and the ``.exe`` spellings.

**Selecting a variant the engine does not have is a segfault, not an error.**
`uci.cpp`'s `position` does `variants.find(Options["UCI_Variant"])->second` and
dereferences the result without checking it against `end()`, so a `UCI_Variant` that
never loaded takes the process down on the next `position` command -- and the commonest
way to get there is a `VariantPath` pointing at a file that is not there, which the
engine reports by saying nothing at all. So the config file is checked for existence
before it is named, and the handshake below confirms the engine actually took the
variant rather than trusting that it did.

**The engine does not admit to knowing a loaded variant, and that is not the same as
not having it.** A config loaded through `VariantPath` goes into the engine's variant
map but the `UCI_Variant` combo advertised by `uci` is not rebuilt until the *next*
`uci`, so a name-in-list guard would refuse every translated ruleset. Nor does `d`
help on its own: it prints the position the engine is holding, and selecting a variant
does not re-set that position -- `d` reports the previous game until a `position`
command arrives. The check that works is `position startpos` and *then* `d`.

A binary that is on the path but is *not* a large-board build answers questions about a
9x9 board with nonsense rather than with an error, so the handshake asks: `variants`
must include a nine-file variant, and a build that fails that is reported as unusable
rather than used. That check is the reason this module talks UCI at all before it is
asked anything -- the alternative is discovering it four moves into a rated game.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading
from typing import TYPE_CHECKING

from chess_variant_sandbox.fairy.unread import parse_check

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chess_variant_sandbox.fairy.variant import Variant

BINARY_ENV = "CHESS_VARIANT_SANDBOX_FAIRY_STOCKFISH"

# The option that takes a variant configuration. A path, never the text of one.
VARIANT_PATH_OPTION = "VariantPath"

# The names a Fairy-Stockfish build is usually installed under. `stockfish` is
# deliberately absent: an ordinary Stockfish answers `uci` perfectly well and cannot
# play a single variant in this corpus, so finding one would be worse than finding none.
BINARY_NAMES = (
    "fairy-stockfish-largeboard_x86-64",
    "fairy-stockfish-largeboard",
    "fairy-stockfish",
    "fairystockfish",
)

# How long any one exchange may take before the engine is declared broken. Generous:
# `go perft 5` on a big board is seconds, and the failure this guards against is a
# process that has died with its pipe still open, which would otherwise hang a request
# for ever.
TIMEOUT = 120.0

# A variant only a large-board build has. Asking for it is how a build that silently
# truncates a nine-file board is caught before it plays one.
_LARGE_BOARD_PROBE = "xiangqi"


class FairyEngineError(Exception):
    """The engine could not be started, or stopped answering."""


def binary_path() -> str | None:
    """Where the Fairy-Stockfish binary is, or ``None``.

    An explicit ``CHESS_VARIANT_SANDBOX_FAIRY_STOCKFISH`` that does not exist is *not* silently
    replaced by one on the path: somebody who named a binary meant that binary, and
    quietly running a different one is how a measurement comes to describe a program
    nobody chose.
    """
    named = os.environ.get(BINARY_ENV)
    if named:
        return named if os.path.isfile(named) else None
    for name in BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def available() -> bool:
    """Whether there is a Fairy-Stockfish this project can play.

    Memoised on the resolved path, because the answer costs a process start and the
    picker asks it on every request. Installing an engine mid-session therefore needs a
    restart, which is the same bargain `chess_variant_sandbox._core` already makes.
    """
    path = binary_path()
    if path is None:
        return False
    verdict = _USABLE.get(path)
    if verdict is None:
        verdict, _NAMES[path] = _probe(path)
        _USABLE[path] = verdict
    return verdict


def engine_name() -> str:
    """What the installed engine calls itself -- ``Fairy-Stockfish 14 LB``, say.

    Read off the same handshake `available` already pays for, so it costs nothing
    extra and is memoised with it. It exists because *which build* is installed is the
    whole answer to why a ruleset that translates is not offered: the newest tagged
    release is from 2021 and cannot play six of the corpus's translations, and a page
    that says "your engine disagreed" without naming the engine leaves the reader to
    guess whether the translation or the download is at fault. Empty where there is no
    usable engine.
    """
    if not available():
        return ""
    path = binary_path()
    return _NAMES.get(path or "", "")


_USABLE: dict[str, bool] = {}
_NAMES: dict[str, str] = {}


def _probe(path: str) -> tuple[bool, str]:
    """Start the engine once and ask whether it is the build we need, and its name."""
    try:
        with UciEngine(path) as engine:
            return _LARGE_BOARD_PROBE in engine.variants(), engine.name
    except (FairyEngineError, OSError):
        return False, ""


class UciEngine:
    """One Fairy-Stockfish process, spoken to over UCI.

    Deliberately not thread-safe and deliberately not shared. A UCI engine is a
    conversation with state -- the loaded variant, the position, the search that is
    running -- and two callers interleaving on one process produce a move for the wrong
    position rather than an error. `agents/fairy_stockfish.py` owns one per agent, and
    the tournament runner builds one agent per seat per game.
    """

    def __init__(self, path: str | None = None) -> None:
        resolved = path or binary_path()
        if resolved is None:
            raise FairyEngineError(
                f"no Fairy-Stockfish binary found; set {BINARY_ENV} or put one on PATH"
            )
        self.path = resolved
        self._lock = threading.Lock()
        try:
            self._process = subprocess.Popen(  # noqa: S603 - the path is the user's own
                [resolved],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            raise FairyEngineError(f"could not start {resolved!r}: {error}") from error
        self._loaded: str | None = None
        self._variants: tuple[str, ...] = ()
        # The build's own name for itself, from the `id name` line of the handshake.
        self.name: str = ""
        self._handshake()

    # ------------------------------------------------------------------ lifecycle

    def _handshake(self) -> None:
        for line in self._exchange("uci", "uciok"):
            if line.startswith("id name "):
                self.name = line[len("id name "):].strip()
            if line.startswith("option name UCI_Variant"):
                # The variant list arrives inside the option declaration, as
                # `... type combo default chess var chess var 3check var ...`.
                parts = line.split(" var ")
                self._variants = tuple(part.strip() for part in parts[1:])

    def variants(self) -> tuple[str, ...]:
        """Every variant this build knows, including any loaded from a config."""
        return self._variants

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        # An engine that died after the poll above (a variant that crashed it, Q703) has a
        # broken pipe: the `quit` fails and so does flushing stdin on close. Neither is
        # news by then, and raising it out of `__exit__` would hide the verdict.
        try:
            self._send("quit")
            process.wait(timeout=5)
        except (OSError, FairyEngineError, subprocess.TimeoutExpired):
            process.kill()
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()

    def __enter__(self) -> UciEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        with contextlib.suppress(Exception):
            self.close()

    # -------------------------------------------------------------------- talking

    def _send(self, command: str) -> None:
        process = self._process
        if process.poll() is not None or process.stdin is None:
            raise FairyEngineError(f"{self.path} has exited")
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except OSError as error:
            raise FairyEngineError(f"{self.path} closed its input: {error}") from error

    def _exchange(self, command: str, until: str) -> list[str]:
        """Send ``command`` and collect output up to the line starting ``until``.

        The timeout is enforced by a watchdog that kills the process rather than by a
        read deadline, because a pipe read has none: an engine that has stopped talking
        would otherwise block the thread for ever, and a hung tournament looks like a
        slow one.
        """
        self._send(command)
        process = self._process
        assert process.stdout is not None
        timer = threading.Timer(TIMEOUT, process.kill)
        timer.start()
        lines: list[str] = []
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    raise FairyEngineError(
                        f"{self.path} stopped answering during {command!r}"
                    )
                line = line.strip()
                if line:
                    lines.append(line)
                if line.startswith(until):
                    return lines
        finally:
            timer.cancel()

    # ------------------------------------------------------------------- variants

    def load(self, variant: Variant, config_path: str) -> None:
        """Teach the engine one translated ruleset and select it.

        The configuration reaches the engine as a *file*: there is no option that takes
        the text of one. The file is the caller's to place and to clean up, which is why
        the path is an argument rather than a temporary made here -- the engine reads it
        when told to and the path has to still be valid then.
        """
        if self._loaded == variant.name:
            return
        if not os.path.isfile(config_path):
            raise FairyEngineError(
                f"the variant configuration {config_path!r} does not exist; naming a "
                "variant the engine could not load segfaults it on the next position"
            )
        self._set_option(VARIANT_PATH_OPTION, config_path)
        self._set_option("UCI_Variant", variant.name)
        self.new_game()
        self._confirm(variant)
        self._loaded = variant.name

    def _confirm(self, variant: Variant) -> None:
        """Check the engine is playing the ruleset it was asked for.

        An unknown variant name is taken in silence and the engine keeps the game it
        had, which would be a whole tournament of chess reported as xiangqi -- and then
        a crash somewhere in the middle of it. There is no line anywhere in the output
        that names the current variant, so the opening array is what gets compared;
        two rulesets that share one are the same game to a first move, which is as far
        as this check claims to go.
        """
        self._send("position startpos")
        shown = self._describe()
        expected = _start_fen(variant.config)
        if expected and _opening_field(shown) != _opening_field(expected):
            raise FairyEngineError(
                f"the engine did not take variant {variant.name!r}: it opens from "
                f"{shown!r} rather than {expected!r}"
            )

    def _describe(self) -> str:
        """The FEN the engine currently holds.

        Terminated on ``readyok`` rather than on the ``Fen:`` line itself: ``d`` prints
        four more lines after it, and stopping at the interesting one leaves them in the
        pipe to be read as the answer to the *next* question.
        """
        self._send("d")
        for line in self._exchange("isready", "readyok"):
            if line.startswith("Fen:"):
                return line[len("Fen:"):].strip()
        return ""

    def _set_option(self, name: str, value: str) -> None:
        """Set one UCI option and wait for the engine to have applied it.

        The wait is not optional. ``setoption`` produces no reply of its own, so without
        an ``isready`` in between, the next command races the option -- and the two
        options this sends are the ones that decide which game is about to be played.
        """
        self._send(f"setoption name {name} value {value}")
        self._exchange("isready", "readyok")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._exchange("isready", "readyok")

    # -------------------------------------------------------------------- playing

    def best_move(self, fen: str, nodes: int) -> str | None:
        """The move the engine plays here, searching ``nodes`` nodes.

        Nodes rather than time, matching the rest of the roster and for the same
        reason: a budget in seconds makes a rating a statement about the machine that
        took it. ``None`` where the engine reports no move, which is a finished
        position and the caller's business rather than an error.
        """
        with self._lock:
            self._send(f"position fen {fen}")
            for line in self._exchange(f"go nodes {nodes}", "bestmove"):
                if line.startswith("bestmove"):
                    parts = line.split()
                    move = parts[1] if len(parts) > 1 else "(none)"
                    return None if move in ("(none)", "0000") else move
        return None

    # ------------------------------------------------------------------ verifying

    def perft(self, fen: str, depth: int) -> int:
        """Leaf count below ``fen`` at ``depth``."""
        return sum(self.divide(fen, depth).values())

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        """Leaf count per root move. At depth 1 this is the legal move list.

        The engine prints ``e2e4: 20`` per root move and then a total; the total line
        has no colon in the same shape, so the parse keys on a move-looking left side
        rather than on position in the output.
        """
        with self._lock:
            self._send(f"position fen {fen}")
            lines = self._exchange(f"go perft {depth}", "Nodes searched")
        counts: dict[str, int] = {}
        for line in lines:
            head, separator, tail = line.partition(":")
            if not separator or " " in head.strip() or not head.strip():
                continue
            try:
                counts[head.strip()] = int(tail.strip())
            except ValueError:
                continue
        return counts

    def mated_after(self, fen: str, move: str) -> bool:
        """Whether ``move`` leaves the side that replies in check with no legal move.

        `verify.Backend.mated_after`. The reply count is a perft of one from the
        position after the move, and the check is the ``Checkers:`` line of ``d``, which
        lists the pieces giving check and is empty when there are none.
        """
        with self._lock:
            self._send(f"position fen {fen} moves {move}")
            for line in self._exchange("go perft 1", "Nodes searched"):
                if line.startswith("Nodes searched") and line.split(":")[-1].strip() != "0":
                    return False
            self._send("d")
            for line in self._exchange("isready", "readyok"):
                if line.startswith("Checkers:"):
                    return bool(line[len("Checkers:"):].strip())
        return False

    def unread_options(self, config_path: str) -> frozenset[str] | None:
        """The keys of ``config_path`` this build does not take, by its own account.

        Asked of a second process, ``<binary> check <file>``, which parses the file as a
        ``VariantPath`` is parsed and names each key it skipped (`unread.parse_check`).
        Loading never says: a key the build does not know is dropped in silence, which
        is how the 2021 release came to play racingkings with no goal (Q915). ``None``
        where the build could not be asked; memoised on the binary and the file's text.
        """
        try:
            with open(config_path, encoding="utf-8") as stream:
                text = stream.read()
        except OSError:
            return None
        key = (self.path, text)
        if key not in _UNREAD:
            try:
                ran = subprocess.run(  # noqa: S603 - the path is the user's own
                    [self.path, "check", config_path],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            _UNREAD[key] = parse_check(ran.stdout + ran.stderr)
        return _UNREAD[key]


_UNREAD: dict[tuple[str, str], frozenset[str] | None] = {}


def _start_fen(config: str) -> str:
    """The `startFen` a translated config declares, for the handshake check above."""
    for line in config.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "startFen":
            return value.strip()
    return ""


def _opening_field(fen: str) -> tuple[str, str]:
    """A FEN's board field, with its pocket read as a multiset.

    The engine writes a hand in its own piece order (king first) and ours writes it in
    type order, so a stocked opening hand (schema 10's finite stock, written into the
    startFen since Q784) is the same hand spelled two ways. The board is compared as
    written.
    """
    first = fen.split()[0] if fen.split() else ""
    board, _, pocket = first.partition("[")
    return board, "".join(sorted(pocket.rstrip("]")))
