"""The local application: one page, a small JSON API, and the standard library.

No web framework, on purpose. The whole surface is a dozen routes over data this package
already produces, and a dependency that has to be installed before a person can play a
game they just invented is a dependency in the wrong place. ``ThreadingHTTPServer`` also
matters: an agent's search happens inside a request, and a single-threaded server would
stop serving the page while it thought.

Bound to localhost. Sessions are in memory, a design run spends money, and nothing here
authenticates anybody -- this is a tool for the person sitting at the machine.

``Server`` turns ``allow_reuse_address`` off on Windows and leaves it on everywhere else,
which is the one place this module disagrees with the standard library. The option means
two different things on the two platforms, so no single constant is right for both, and
``Server``'s docstring has the measurements.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import webbrowser
from collections.abc import Iterable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, unquote, urlparse

from chess_variant_sandbox import corpus, history
from chess_variant_sandbox.measurement import stale_build
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import describe, spec_hash, validate_spec
from chess_variant_sandbox.spec import reference as rule_reference
from chess_variant_sandbox.web import (
    candidacy,
    cards,
    changelog,
    designing,
    launch,
    library,
    opponents,
    refreshing,
)
from chess_variant_sandbox.web.evolving import Runs
from chess_variant_sandbox.web.human import HUMAN, HumanGame, HumanRecord, picker_ratings
from chess_variant_sandbox.web.keeping import Measurements
from chess_variant_sandbox.web.opponents import Ratings
from chess_variant_sandbox.web.page import SCRIPTS, assets, page, picture, rules_page, shell
from chess_variant_sandbox.web.page import game as game_page
from chess_variant_sandbox.web.play import Session, Sessions, seed_asked

# The header every state-changing request from the page carries. A browser lets a
# cross-site form POST, but only a same-origin script may set a custom header, so its
# absence marks a request no page of ours sent.
REQUEST_HEADER = "X-Requested-With"
REQUEST_VALUE = "chess-variant-sandbox"

# How much of a refused POST's body is read and thrown away before its connection closes.
# Closing a socket that still holds unread bytes aborts the connection rather than ending
# it, and an abort discards whatever the client has not yet read -- including the refusal
# itself. Draining first makes the close clean. The length is the caller's claim, so it is
# capped: a page's own POST never reaches here at all, and a forged one that declares more
# than this gets the close it used to get.
DRAIN_LIMIT = 64 * 1024

# The games the map draws. A design run's candidates and an unpublished draft are not on
# it: the map is one payload served the same to everybody, and a link spells out what two
# games differ in. Which games a *gap* may end on is a different question, and a
# per-person one -- `App.findable` answers it.
MAPPED = ("committed", "archived")

# The games the map draws no dot for because they are somebody's drafts: one built by hand
# and one a design run found. A retired game is off the map too, and is nobody's draft.
# A draft's own page is where it is offered *make the game in between* (`App._nearest`).
DRAFTED = ("designed", "evolved")

# Third-party files, served byte for byte from inside the package.
#
# One directory, one manifest, and the route serves what the manifest names and nothing
# else -- so a path cannot walk out of the directory and an untracked file dropped beside
# the vendored ones is not reachable. The answer is `immutable` because the version is in
# the file's own name: a new version is a new name, never new bytes under the old one.
VENDOR_PREFIX = "/static/vendor/"
VENDOR_DIR = Path(__file__).resolve().parent / "static" / "vendor"

# A year, and never asked about again: for a file whose name changes whenever its bytes do.
IMMUTABLE = "public, max-age=31536000, immutable"

# The browser's measuring runtime: Pyodide and this package's wheel, which
# `deploy/pyodide/build.sh` writes to `build/browser/` (`docs/archive/plan-client-compute.md`,
# *What B1 built*). The directory is found under the working directory, where `serve`
# finds `rulesets/` and `runs/`, and `App.browser_dir` points one app at another. On the
# box Caddy answers `/browser/*` itself (B7); this route is the local application's, and
# the site's fallback.
#
# The manifest is the whole of what the route serves, as the vendored manifest is above:
# a path is a lookup in its `files`, not a path to resolve. So `..` and an untracked file
# dropped into the directory are not reachable, and neither is anything while there is no
# manifest -- which is also what a build whose checks failed leaves, because
# `deploy/pyodide/manifest.py` writes nothing then.
BROWSER_PREFIX = "/browser/"
BROWSER_DIR = Path("build") / "browser"
BROWSER_MANIFEST = "manifest.json"

# By extension. The standard library's `mimetypes` reads the Windows registry, which can
# say `text/plain` for `.js`, and a module script or `WebAssembly.instantiateStreaming`
# refuses the file over a wrong type. A wheel is a zip.
BROWSER_TYPES = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
    ".zip": "application/zip",
    ".whl": "application/zip",
}

# The page's own scripts (`web.page.SCRIPTS`, route -> file) and its stylesheets are
# answered under versioned names, `<route stem>.<digest>.<ext>`, for a year, and only the
# names `web/page/assets.py` holds (G12). The plain route of a script is still answered,
# `text/javascript`, `no-store`, for a tool or a test that asks by it; no page names it.

# B3's worker script, one real file inside the package. `no-store`, because its name does
# not change when its bytes do: it is loaded only when a draft is measured in the browser,
# never by a visit to a page, so no visit pays for it.
WORKER_ROUTE = "/static/measure-worker.js"

# The rule language's reference, `RULES.md`, as the code this server runs renders it (Q905):
# the designer links it (`reference.PUBLISHED` is this path), so the page a reader opens
# describes the validator that will read their file, never an older export. A constant of
# the running code, served to anybody on both servers; cached briefly, since a deploy can
# change it under the same address.
RULES_ROUTE = rule_reference.PUBLISHED
RULES_TYPE = "text/markdown; charset=utf-8"
RULES_CACHE = "public, max-age=300"

# The same reference as a web page (G8, `web/page/rules_page.py`), drawn from the same
# `render_reference()`, so the two cannot drift. The local application answers it with the
# page under its own header; the site answers it in its pages' shell before dispatch.
RULES_PAGE_ROUTE = rule_reference.PAGE


def rules_markdown() -> bytes:
    """The reference's bytes, rendered once per process: the schema cannot change under a
    running server."""
    held = getattr(rules_markdown, "held", None)
    if held is None:
        held = rule_reference.render_reference().encode("utf-8")
        rules_markdown.held = held  # type: ignore[attr-defined]
    return held


# The addresses the single page answers (Q723). `/` is the landing page and `/game` the
# games, for everybody; the page itself tells them apart (`web/static/page/core.js`'s
# `render`), and every hash route works on either, so a link from before `/game` existed
# still opens what it named.
GAMES_PATH = "/game"
PAGE_PATHS = ("/", "/index.html", GAMES_PATH)
WORKER_FILE = Path(__file__).resolve().parent / "static" / "measure-worker.js"

# What a browser may run, load and send, on every answer this application or the site
# gives (Q514, `docs/reference/page-javascript.md` §7 has each directive's reason). The
# point is `script-src` without `'unsafe-inline'`: an injected `<script>` or `on...=`
# attribute is inert text. `'wasm-unsafe-eval'` is the in-browser measurement's: Pyodide
# compiles WebAssembly in the worker, and the worker takes its policy from its own
# script's answer, which is this one. `style-src 'self'` is the same rule for style (G12):
# every stylesheet is a file (`web/page/assets.py`), no page carries a `<style>` or a
# `style=` attribute, and a value a script works out is a rule it adds through the CSSOM
# (`styleClass` in `viz/wrapped.js`), which the policy does not govern. An injected `<style>`
# or `style=` is inert, as an injected `<script>` is.
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self' 'wasm-unsafe-eval'",
        "worker-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    )
)


def vendored() -> dict[str, dict[str, Any]]:
    """The manifest, by file name. Read once and kept, because it never changes."""
    held = getattr(vendored, "held", None)
    if held is None:
        manifest = json.loads((VENDOR_DIR / "manifest.json").read_text(encoding="utf-8"))
        held = {entry["file"]: entry for entry in manifest["files"]}
        vendored.held = held  # type: ignore[attr-defined]
    return held


def browser_file(root: Path, asked: str) -> tuple[Path, str, str] | None:
    """What `/browser/<asked>` answers: a file, its type and its cache header, or `None`.

    Read afresh on every request, because a rebuild replaces the directory under a
    running server. `manifest.json` is `no-store` for the same reason. A file under one
    of the two directories the manifest names -- its `runtime` and the directory of its
    `wheel`, `pyodide-<version>/` and `wheel-<digest>/` -- is immutable, because a new
    version or a new wheel is a new directory. Anything else it names is `no-store`.

    `None` when there is no manifest, when it is not a JSON object, when it does not
    name `asked`, or when what it names resolves outside `root`. The last cannot happen
    to a manifest `deploy/pyodide/manifest.py` wrote, and is checked because the lookup
    is only as good as the file it reads.
    """
    try:
        manifest = json.loads((root / BROWSER_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if asked == BROWSER_MANIFEST:
        return root / BROWSER_MANIFEST, "application/json", "no-store"
    files = manifest.get("files")
    named = {
        entry.get("path")
        for entry in (files if isinstance(files, list) else [])
        if isinstance(entry, dict)
    }
    if asked not in named:
        return None
    target = (root / asked).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    lasting = {
        str(name).split("/", 1)[0]
        for name in (manifest.get("runtime"), manifest.get("wheel"))
        if name
    }
    directory, _, rest = asked.partition("/")
    cache = IMMUTABLE if rest and directory in lasting else "no-store"
    kind = BROWSER_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return target, kind, cache


def _seed(asked: Any) -> int:
    """A seed from a query string, refused rather than guessed at.

    A remix is the same draw the list of steps made, which only holds if the seed is the
    one the list gave. Anything that is not a number is a request nothing sent.
    """
    try:
        return int(str(asked))
    except (TypeError, ValueError) as error:
        raise designing.DesignError(
            f"a design step is drawn under a seed, and {asked!r} is not one"
        ) from error


def _seed_set(asked: Any) -> tuple[int, ...] | None:
    """`one` or `four` from a request body, as the seeds `corpus` names.

    Two words rather than a list of integers, and the same two the command line takes.
    Anything else -- absent, empty, a number a page invented -- is ``None``, which means
    whatever the game's own rung expects, so a browser cannot ask for a seed set the
    project does not use.
    """
    return {"one": corpus.ONE_SEED, "four": corpus.LIBRARY_SEEDS}.get(str(asked or ""))

class SessionStore(Protocol):
    """Where a game in progress lives. In memory here; a `matches` row on the site.

    The local application's `Sessions` is the whole of this, and `chess_variant_sandbox/site/` supplies
    the same two calls over the store. Written down because it is the one seam a public
    site needed in a module the local tool owns -- everything else the site changes is a
    route it declines to register.
    """

    def start(self, *args: Any, **kwargs: Any) -> Session: ...

    def get(self, session_id: str) -> Session: ...


class AnchorStore(Protocol):
    """Where human results live. A JSONL file here; leaderboard rows on the site."""

    def note_start(self, *args: Any, **kwargs: Any) -> None: ...

    def note_result(self, *args: Any, **kwargs: Any) -> None: ...

    def games(self, ruleset_hash: str = "") -> list[HumanGame]: ...

    def summary(self, ruleset_hash: str, ratings: dict[str, float] | None) -> dict[str, Any]: ...


class App:
    """Everything the routes need, and the only mutable state in the process."""

    def __init__(
        self,
        ruleset_dir: Path,
        runs_dir: Path,
        heuristics_dir: Path | str = "heuristics",
        archive_dir: Path | str = corpus.ARCHIVE_DIR,
        browser_dir: Path | str = BROWSER_DIR,
    ) -> None:
        self.ruleset_dir = Path(ruleset_dir)
        self.runs_dir = Path(runs_dir)
        self.heuristics_dir = Path(heuristics_dir)
        self.archive_dir = Path(archive_dir)
        # Where `/browser/<path>` is answered from; see `BROWSER_DIR`.
        self.browser_dir = Path(browser_dir)
        self.measurements = Measurements(runs_dir, self.ruleset_dir, self.archive_dir)
        self.humans: AnchorStore = HumanRecord(runs_dir)
        self.sessions: SessionStore = Sessions()
        self.runs = Runs()
        self.ratings = Ratings(runs_dir)
        self._games: dict[str, library.Game] = {}
        # Not read here. Building the application used to read the whole corpus, and on
        # the box that was a 21 s outage every restart: `chess_variant_sandbox/site/server.py` builds
        # the app, then binds, so the port stayed closed for the length of the read.
        # Nothing needs the listing until a request wants it, and `serve` warms it on a
        # thread once the socket is bound.
        self._read = False
        self._reading = threading.RLock()
        # Guards `library_payload` end to end, not only the `discover` step `_reading`
        # covers. See that method for why: without this, `warm()`'s background pass and a
        # request landing while it still runs could each pay the smoke gate and the
        # Fairy-Stockfish translation for the same spec (Q418).
        self._warming = threading.Lock()

    def known(self) -> dict[str, library.Game]:
        """The last listing, read now if none has been.

        Every reader of the snapshot goes through here rather than touching `_games`,
        so "the library has not been read yet" is a moment that lasts until the first
        request and is invisible to all of them.
        """
        if not self._read:
            self.refresh()
        return self._games

    def roster(self, ruleset_hash: str) -> list[opponents.Opponent]:
        """Who you can play in this ruleset, with its heuristic sets registered.

        Every caller goes through here, including the one that starts a session, because
        listing an opponent and being able to build it are the same act: `resolve`
        refuses an unknown digest rather than defaulting to one.
        """
        game = self.known().get(ruleset_hash)
        spec = game.spec if game is not None else None
        return opponents.for_game(ruleset_hash, self.heuristics_dir, spec=spec)

    def refresh(self) -> list[library.Game]:
        """Re-read the library from disk.

        Called on every listing rather than cached for the process's lifetime, because a
        design run running *in this process* writes new rulesets, and a library that
        cannot see them is a library that goes stale while you watch it. `discover` is
        cheap when nothing has moved and a full re-read when something has, so what this
        costs is what changed.

        One reader at a time. Requests are threads, and two of them reading the same
        unmoved tree at once would each pay the whole read; the second waits and is
        served by the first. The lock is only ever held by this method.
        """
        with self._reading:
            games = library.discover(self.ruleset_dir, self.runs_dir, self.archive_dir)
            self._games = {game.hash: game for game in games}
            self._read = True
            return games

    def library_payload(self) -> dict[str, Any]:
        """Everything `GET /api/library` answers: the listing, banded, with the coverage
        grid, the sort legend and the opponent ladder.

        One lock around the whole of it, `_warming`, and not only around `refresh`'s own
        `discover` step. Per game this also runs the smoke gate and attempts the
        Fairy-Stockfish translation (`Game.summary()`, through `library._playability` and
        `library._fairy`) -- memoised on the spec's content in `web/library.py`, but that
        memo holds no lock of its own around calling the gate or the translator, so two
        callers racing on a spec neither has seen yet would each pay for it once.

        `warm()` runs this, off the request path, on a thread started right after the
        socket binds (Q418). A request for `/api/library` that lands while that thread is
        still in here waits on `_warming` instead of starting a second pass, and by the
        time it can proceed every spec `warm()` reached is already cached -- so it costs
        what a second listing always cost, about 0.03 s over the whole corpus, seconds
        cheaper than the 7-13 s a cold first listing pays (Q415, in
        `docs/reference/fast-suite-review.md`).

        Those per-game answers (and the card's still board) also outlive the process:
        `library.CardAnswers` keeps them on disk under the code that decides them, so a
        restart that changed none of it lists without compiling a game (Q887).
        """
        with self._warming:
            everything = self.refresh()
            # Q931: the previews are listed apart, in `previews`, and every scored surface
            # below -- the bands, the provisional count, the coverage, the newest games --
            # reads the rest. `games` is the library and nothing else.
            games = library.scored(everything)
            previews = [game for game in everything if game.source == library.PREVIEW]
            # Where each game's score stands against every other measured game's. On the
            # card rather than in `summary()` because it is the one thing on a card that
            # is not a fact about that game alone -- it takes the whole listing to work
            # out, and the server works it out so that no page compares two scores.
            band = library.bands(games)
            # Each card's thumbnail, the opening position (V6), with its pieces' drawings
            # sent once for the whole listing rather than once a card.
            cards = [{**game.summary(), "band": band.get(game.hash),
                      "still": library.still(game)} for game in games]
            art: dict[str, str] = {}
            for game in everything:
                art.update(library.piece_art(game))
            # The cards' rule answers this listing worked out, written once for the next
            # process to read instead of compiling every game again (Q887).
            library.save_card_answers()
            return {"games": cards,
                    # A preview's card has no band: it is in no comparison (Q931).
                    "previews": [{**game.summary(), "band": None,
                                  "still": library.still(game)} for game in previews],
                    "pieceArt": art,
                    # The leaderboard's header (EV3): how many of those numbers are
                    # provisional, or null when none is and the header says nothing.
                    "provisionalScores": library.provisional_count(cards),
                    "coverage": library.coverage(games),
                    # What the list can be ordered by, in words. Sent once with the
                    # listing rather than fetched when the menu opens: it is four
                    # kilobytes and the menu is the first thing a person touches.
                    "metrics": library.metric_legend(),
                    # Q788: the landing page's newest games and its spotlight, over
                    # what the listing holds (the site narrows both to what it offers).
                    "newGames": launch.new_cards(games),
                    "spotlight": launch.spotlight_card(games),
                    "opponents": [
                        entry.as_dict() for entry in opponents.ladder()
                    ]}

    def game(self, digest: str) -> library.Game:
        """One game, its own facts re-read from disk rather than looked up unchanged.

        `_games` is a snapshot of *derived facts* -- a `Game` carries the measurement,
        the verdict and the design score it was built with, and memoises its score on the
        instance -- and this process writes those facts. It used to refresh only when the
        hash was unknown, which meant it could see a new ruleset appear and could not see
        an existing one's measurement change.

        That is the whole of the bug in `docs/archive/bugs-resolved.md`: `/api/test`
        finishes, the page drops its own cache and asks for the game again, and the
        answer is the `Game` built before the tournament ran -- `score: null`,
        `staleness: null`, so the panel that started the run renders "nobody has measured
        this one". The library was right about the same game in the same second because
        `/api/library` re-reads on every listing, which is what made it look like a page
        bug and is why walking back to the list "fixed" it.

        The fix for *that* bug was a full `refresh()` -- the whole corpus re-read on
        every open of one game's page, performance item 20. This keeps the same
        freshness at the cost of the one game it is about: an already-known hash is
        answered by `library.reread_one`, which re-reads only this game's own measurement
        and verdict (`corpus.read_measurement` already resolves those with the spec in
        hand, at a handful of file opens, never a walk of `runs/`). A hash `known()` has
        never seen -- a brand-new ruleset a design run just wrote, or one dropped into
        `rulesets/` since the last listing -- still gets the full `refresh()` that finds
        it, because nothing cheaper can.
        """
        game = self.known().get(digest)
        if game is None:
            self.refresh()
            game = self.known().get(digest)
            if game is None:
                raise KeyError(f"no ruleset with hash {digest}")
            return game
        # The same lock `refresh()` holds around its own read, so a concurrent full
        # listing cannot swap `_games` for a new dict out from under this one entry's
        # update -- cheap here because this read touches one game's own files, never
        # the whole tree.
        with self._reading:
            fresh = library.reread_one(game, self.runs_dir)
            self._games[digest] = fresh
        return fresh

    def _parent_spec(self, digest: Any) -> dict[str, Any] | None:
        """The spec of the game a draft was forked from, or None when it has none.

        Looked up among the games already listed rather than through `game()`: the check
        runs on every pause in typing, and a parent the library cannot find is a draft
        with nothing to compare against, not a reason to re-read the corpus.
        """
        if not isinstance(digest, str) or not digest:
            return None
        game = self.known().get(digest)
        return game.spec if game is not None else None

    def graph(self) -> dict[str, Any]:
        """The library as a map: games, the lines of likeness, communities, lineage, places.

        **The corpus, not the listing.** A node is a committed or archived ruleset and
        nothing else. A design run's candidates are hundreds of near-identical children of
        one parent, which would bury the library they were grown from; and on the public
        site the same listing holds drafts nobody has published, whose rules a map would
        spell out to everybody. `graphing` reads whatever it is handed, so the day the
        site has a links table of its own with the visibility that needs
        (G7 of `docs/archive/plan-game-graph.md`), it hands it more.

        The first call with no `runs/graph/map.json` reads every game and draws the
        picture -- several seconds -- and writes that file; a later call settles from it,
        and one over the same library is answered from memory (`graphing._MAPS`).
        """
        from chess_variant_sandbox.web import graphing

        return graphing.graph(self.mapped(self.refresh()), self.runs_dir)

    def mapped(self, games: Iterable[library.Game]) -> list[library.Game]:
        """The games the map draws, out of this listing: the committed and archived ones.

        The one population for the map and for every list that names a game as near or
        like another (`near`), so the two are always drawn over the same library (M5 of
        `docs/plan-map.md`). The site leaves out what an admin took off the library
        (`site/server.SiteApp.mapped`).
        """
        return [game for game in games if game.source in MAPPED]

    def near(self, game: library.Game) -> dict[str, Any]:
        """The games most like this one on the map, and what each pair is
        (`library.neighbours`): a game's page names these, and nothing else names near."""
        return library.neighbours(game, self.mapped(self.known().values()), self.runs_dir)

    # ---------------------------------------------------------------- endpoints

    def get(self, path: str, query: dict[str, str] | None = None) -> Any:
        """One GET, dispatched by path.

        `query` is the request's query string, already flattened. No listing here reads
        it -- every listing the local application draws is narrowed and ordered in the
        browser, over a payload it already holds, and the site's own listing takes `sort`
        and `filter` server-side (§5). Two routes do: a draft's progress page, and the
        designer's starting points, where it names which derived start is wanted.
        """
        parts = [part for part in path.strip("/").split("/") if part]
        if parts == ["api", "me"]:
            # The header's identity probe (Q199). This application has no store and
            # nobody is ever signed in, so the answer is the constant every page gets --
            # 200 rather than the 404 `/api/profile` gives, because a header asking a
            # question that makes no sense here still deserves an answer rather than a
            # console error on every page load. `accounts: False` is how the header
            # tells this apart from the site's own visitor, who gets the same shape with
            # `accounts: True`.
            return {"signedIn": False, "handle": None, "admin": False, "accounts": False}
        if parts == ["api", "library"]:
            return self.library_payload()
        if parts == ["api", "graph"]:
            return self.graph()
        if parts == ["api", "changelog"]:
            # `docs/changelog.md` as blocks and inline runs. Read on each request, which
            # costs about a millisecond, and answered with a sentence rather than an
            # error when the file is not there.
            return changelog.payload()
        if len(parts) == 3 and parts[:2] == ["api", "game"]:
            digest = parts[2]
            # Read once and answered from, rather than asked for four times. `game` now
            # re-reads the library, and a view assembled from four reads of disk is a
            # view that can disagree with itself about the game it is describing.
            game = self.game(digest)
            refused = opponents.refusals(digest, game.path, heuristics=self.heuristics_dir)
            # The games like this one, on the map's relation (M5 of `docs/plan-map.md`):
            # the page's list, and the draft's gap button's nearest library game.
            near = self.near(game)
            return {
                **library.detail(game, self.known().values()),
                **near,
                # The card's band, from the listing `self.game` just refreshed, so the
                # page and the card cannot answer differently about the same game.
                "band": library.bands(self.known().values()).get(digest),
                "opponents": [entry.as_dict() for entry in self.roster(digest)],
                # Why the outside engine is, or is not, among those opponents: the
                # verdict the roster was just built from, so the two cannot disagree.
                "fairyCheck": opponents.fairy_check(game.spec),
                # A searched table the roster just left out as stale, one
                # sentence each with the command that repairs it (Q142), and a built
                # opponent it left out as a second spelling of a held plan (Q446). The
                # strength panel shows them, because rating the picker is where one used
                # to fail.
                "refusedStores": [said for _, said in refused],
                # The same players by name alone, which is all the site says of them (G19).
                "refusedNames": [name for name, _ in refused],
                "pickerRatings": self.ratings.stored(digest),
                # K3 (`docs/archive/plan-understanding.md` §4.2 step 3): the opponent a person's
                # first game here seats, the gentlest on this game's ladder above random,
                # read off its rating table as the front page's first step reads it, and
                # why the game is far from chess, where it is.
                "firstGame": self._first_game(game, digest),
                "keeping": library.keeping(
                    game, self.runs_dir, self.ruleset_dir, self.archive_dir
                ),
                # Why the measure button on this page is off, when it is. The same
                # sentence the library's panel shows, from the same check.
                "build": stale_build() or "",
                # A draft's own gap button: the library game nearest it, and the game in
                # between the two or why there is none. Null on a library game, and on a
                # draft this person may not build from.
                "between": self._nearest(game, near["links"]),
            }
        if len(parts) == 3 and parts[:2] == ["api", "rating"]:
            run = self.ratings.get(parts[2])
            if run is not None:
                return run.as_dict()
            stored = self.ratings.stored(parts[2])
            return {"hash": parts[2], "state": "finished" if stored else "absent",
                    "table": stored, "played": 0, "total": 0, "log": []}
        if len(parts) == 3 and parts[:2] == ["api", "test"]:
            measuring = self.measurements.get(parts[2])
            if measuring is not None:
                return measuring.as_dict()
            measured = self.measurements.stored(parts[2])
            return {"hash": parts[2], "state": "finished" if measured else "absent",
                    "result": measured, "seconds": 0, "log": []}
        if len(parts) == 3 and parts[:2] == ["api", "session"]:
            return self.sessions.get(parts[2]).snapshot()
        if parts == ["api", "human"]:
            return {"rulesets": [
                self.human_summary(game.hash)
                for game in self.refresh()
                if self.humans.games(game.hash)
            ]}
        if len(parts) == 3 and parts[:2] == ["api", "human"]:
            return self.human_summary(parts[2])
        if len(parts) == 3 and parts[:2] == ["api", "run"]:
            return self.runs.get(parts[2]).as_dict()
        if parts == ["api", "runs"]:
            return {"runs": self.runs.all()}
        if parts == ["api", "scoring"]:
            return self.scoring()
        if parts == ["api", "design"]:
            # Everything the editor needs before it can draw anything: what a new game
            # can start from, and what an opponent can be told to care about.
            corpus_specs = cards.corpus_specs(self.ruleset_dir, self.archive_dir)
            return {
                "templates": designing.templates(),
                "vocabulary": designing.vocabulary(),
                "schema": designing.schema_note(),
                "cards": cards.card_model(corpus_specs),
                # What a rule is called and what it does, in words. The schema describes
                # thirteen of the twenty-eight keys and names all of them, so a menu
                # built from the schema alone can only show the key.
                "ruleWords": describe.rule_words(),
                "ruleChoices": describe.rule_choices(),
                # Every distinct piece the library holds, with its moves, for the tray
                # to lift one out of a game that already plays.
                "palette": designing.piece_palette(corpus_specs),
                "rungs": candidacy.rungs(self.measurements.last_seconds),
                "nominations": candidacy.nominations(
                    self.runs_dir, self.ruleset_dir, self.archive_dir),
                # For writing a game outside the designer (Q830b): where the rule language
                # is published, and the two controls `RULES.md` names, one spelling each.
                "ruleFile": designing.rule_file_kit(),
            }
        if len(parts) == 4 and parts[:3] == ["api", "design", "standing"]:
            return self._draft_standing(parts[3], query or {})
        if len(parts) == 3 and parts[:2] == ["api", "design"]:
            # An existing game as a starting point -- and what it *cost* to know what
            # this game is, because the editor is about to change it and a measurement
            # does not travel to the game that comes out. A provisional number is
            # withheld here: the editor compares a draft with its parent, and a
            # provisional estimate enters no comparison (`Game.current_score`).
            game = self.game(parts[2])
            state = library.keeping(
                game, self.runs_dir, self.ruleset_dir, self.archive_dir
            )
            return {
                "hash": game.hash,
                "name": game.name,
                "spec": game.spec,
                "source": game.source,
                "stage": state["stage"],
                "index": (game.current_score or {}).get("index"),
                # How many runs that number is the mean of, under the card's own key: the
                # editor says it beside the number (`measuredHtml`), and without this it
                # printed the number alone.
                "seeds": list((game.current_score or {}).get("seeds") or []),
                "staleness": game.staleness,
                "roster": state["roster"],
                "gamesPerPairing": state["gamesPerPairing"],
                **self._remixes(game, query or {}),
            }
        raise KeyError(f"no such endpoint: /{'/'.join(parts)}")

    def _first_game(self, game: library.Game, digest: str) -> dict[str, Any] | None:
        ladder = [entry.as_dict() for entry in self.roster(digest) if entry.store == "ladder"]
        stored = self.ratings.stored(digest)
        table = stored if stored and stored.get("standings") else game.ratings
        chosen = launch.opponents(ladder, table)
        if chosen is None:
            return None
        return {**chosen, "far": library.far_from_chess(game)}

    def _remixes(self, game: library.Game, query: dict[str, str]) -> dict[str, Any]:
        """A starting point derived from this game, when the query asks for one.

        Three questions, all of them about *where a draft could start from here*, which is
        what this route has always answered. Asked for by query rather than by three
        routes of their own, because the answer is about the game in the path and the site
        already decides who may reach that game (`site/server._playable`).

        `steps` lists the design steps that apply; `step` and `seed` draw one of them;
        `between` names another game and asks for the one in between. None of them is
        computed unless it is asked for -- listing the steps costs a twentieth of a second
        and *make your own from this* must not pay it.

        Nothing here widens who may keep a draft. Every answer is a spec in a reply, and
        `POST /api/design/save` is still the only way one is written down.
        """
        answer: dict[str, Any] = {}
        if query.get("steps"):
            answer["steps"] = designing.design_steps(game.spec)
        step = query.get("step")
        if step:
            answer["remix"] = designing.remix(
                game.spec, step, _seed(query.get("seed")), game.name
            )
        other = query.get("between")
        if other:
            answer["between"] = self._between(game, other)
        return answer

    def _between(self, game: library.Game, other: str) -> dict[str, Any]:
        """The game between this one and another, by hash.

        Both ends must be games this person may be shown (`findable`), which is what a
        link's two ends already are: a gap is a link, and it spells out what the two
        differ in. So a person's own draft can be one end, and somebody else's unlisted
        one cannot be either.

        Both ends are asked, not only the query's. The query's hash is named by the
        browser, and the path's is gated by whoever serves the route for *may this be
        opened* -- which on the site lets anybody open an unlisted draft by its link.
        Opening a game is not being allowed to find it (`site/finding.py`), and building
        from it is the second.

        The refusal is worded as it was when the rule was the map's, because a game
        somebody may not be shown is never one the map draws. It is the answer a
        stranger was given before a person's own game could be an end, word for word.
        """
        far = self.game(other)
        shown = self.findable((game, far))
        for end in (game, far):
            if end.hash not in shown:
                raise designing.DesignError(
                    f"{end.name} is not one of the games the map draws, so there is no "
                    "gap between these two to build a game in"
                )
        return designing.in_between(game.spec, far.spec, (game.name, far.name))

    def findable(self, games: Iterable[library.Game]) -> set[str]:
        """Which of these games the person asking may be shown, by hash.

        Here, all of them. The local application has one person: every draft and every
        candidate under `runs/` is theirs, and its game page already names each of them
        as a neighbour. The site has many and asks its store (`site/server.SiteApp`),
        where an unlisted draft is its owner's to find and nobody else's.
        """
        return {game.hash for game in games}

    def _nearest(
        self, game: library.Game, links: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        """*Make the game in between* on a draft's own page, against the nearest library game.

        The map offers the button on a gap and draws no drafts, so the game between a
        person's own draft and the library had a route (`_between`) and no page. A draft's
        page is where it goes, and it needs a second end: the library game most like the
        draft, which is the first of the page's own *games like this one* (`links`, from
        `near`) that this person may be shown. So *nearest* means here what it means on
        the page and on the map: the map's likeness (M5 of `docs/plan-map.md`).
        `likeness` on the answer says how alike, and `units` how many rules apart.

        Answered only where the press would be: both ends findable, the question
        `_between` asks when the button is pressed. On the site that is the owner of an
        unlisted draft and not a stranger who opened it by its link (`SiteApp.findable`);
        here it is every draft. The game in between is `designing.in_between`'s, the one
        place the map's gaps are decided too, and what travels is the map's shape: which
        end it is built on (`from`), which end the rule comes from (`taken`) and that
        rule's phrase -- or `why` there is none. A library game one rule from the draft
        has nothing between them, which is said in those words rather than as whichever
        of `in_between`'s refusals a one-rule pair happens to meet.

        None on a library game, on a draft this person may not build from, and where the
        library holds nothing this person may be shown.
        """
        if game.source not in DRAFTED:
            return None
        known = self.known()
        listed = [known[str(link["hash"])] for link in links if str(link["hash"]) in known]
        shown = self.findable([game, *listed])
        if game.hash not in shown:
            return None
        link = next((one for one in links if str(one["hash"]) in shown), None)
        if link is None:
            return None
        other = known[str(link["hash"])]
        answer: dict[str, Any] = {
            "hash": other.hash,
            "name": other.name,
            "units": link["units"],
            "likeness": link["likeness"],
        }
        if link["units"] == 1:
            answer["why"] = (
                f"{other.name} is one rule from this game, so no game stands between them."
            )
            return answer
        made = designing.in_between(game.spec, other.spec, (game.name, other.name))
        if "spec" not in made:
            answer["why"] = str(made["why"])
            return answer
        ends = (game.hash, other.hash)
        answer["from"] = ends[int(made["on"])]
        answer["taken"] = ends[1 - int(made["on"])]
        answer["words"] = str(made["words"])
        return answer

    def post(self, path: str, body: dict[str, Any]) -> Any:
        parts = [part for part in path.strip("/").split("/") if part]
        if parts == ["api", "session"]:
            return self._start_session(body)
        if len(parts) == 4 and parts[:2] == ["api", "session"]:
            return self._session_action(parts[2], parts[3], body)
        if parts == ["api", "run"]:
            return self._start_run(body)
        if parts == ["api", "rate"]:
            return self._start_rating(body)
        if parts == ["api", "test"]:
            return self._start_test(body)
        if parts == ["api", "scoring"]:
            return self._start_sweep(body)
        if parts == ["api", "scoring", "stop"]:
            return self._stop_sweep()
        if parts == ["api", "archive"]:
            return self._keep(body, promote=False)
        if parts == ["api", "promote"]:
            return self._keep(body, promote=True)
        if parts == ["api", "design", "check"]:
            return designing.check(body.get("spec"), self._parent_spec(body.get("parent")))
        if parts == ["api", "design", "load"]:
            # A rule file somebody wrote, read and validated in memory (Q830b): a valid one
            # comes back as the draft, an invalid one with the validator's lines.
            return designing.read_rule_file(body.get("text"))
        if parts == ["api", "design", "moves"]:
            return designing.move_map(
                body.get("spec") or {},
                int(body.get("piece", 0)),
                int(body.get("square", -1)),
                # Two different questions, and the caller says which: what the descriptor
                # means on an empty board, or what the piece may do from the opening
                # position as it stands.
                alone=body.get("alone", True) is not False,
            )
        if parts == ["api", "design", "teach"]:
            return designing.teach(
                body.get("spec") or {},
                int(body.get("piece", 0)),
                int(body.get("square", -1)),
                [int(at) for at in (body.get("targets") or [])],
                str(body.get("mean", "both")),
                from_rank=bool(body.get("fromRank")),
            )
        if parts == ["api", "design", "hex"]:
            # A draft whose cells were just made hexagons, with the square lattice's
            # readings rewritten as their hex twins (Q640). Nothing is kept.
            return designing.onto_hex(body.get("spec"))
        if parts == ["api", "design", "rules"]:
            # Which absent rules this draft could not take, tried by the validator. The
            # candidate cards are the browser's; nothing here decides what one holds.
            return designing.refusals(body.get("spec"), body.get("candidates") or {})
        if parts == ["api", "design", "turns"]:
            # The draft played another way (one after the other, in rounds, several moves
            # a turn), with every change that way needs, asked of the validator (Q995b).
            return designing.take_turns(body.get("spec"), str(body.get("way", "")),
                                        body.get("card"))
        if parts == ["api", "design", "rest"]:
            # Every piece resting the same, written out on each piece, with what the
            # validator lets take a rest decided by the validator (Q995d).
            return designing.rest_every_piece(body.get("spec"), body.get("rest"))
        if parts == ["api", "design", "steps"]:
            return self._draft_steps_or_one(body)
        if parts == ["api", "design", "save"]:
            return self._save_design(body)
        if parts == ["api", "design", "hint"]:
            # Optional (Q396): which sentences of a saved game's description its rules
            # seem not to support, asked of a third-party model only when a key is set.
            # The page asks after a save has answered; the save never reaches this, and
            # the import is here so nothing on the save path loads the module.
            from chess_variant_sandbox.web import jev_hint

            return jev_hint.for_saved(self.runs_dir, str(body.get("hash") or ""))
        if parts == ["api", "design", "play"]:
            return self._play_draft(body)
        if parts == ["api", "design", "screen"]:
            return candidacy.screen_draft(body.get("spec"))
        if parts == ["api", "design", "measure"]:
            return self._measure_design(body)
        if parts == ["api", "design", "nominate"]:
            return self._nominate(body)
        if parts == ["api", "design", "icon"]:
            return {"href": designing.icon_data_uri(str(body.get("svg", "")))}
        if parts == ["api", "opponent"]:
            return self._build_opponent(body)
        if parts == ["api", "opponent", "forget"]:
            return designing.forget_opponent(
                str(body["hash"]), str(body["digest"]), self.heuristics_dir
            )
        raise KeyError(f"no such endpoint: /{'/'.join(parts)}")

    # --------------------------------------------------------- building one by hand

    def _draft_steps_or_one(self, body: dict[str, Any]) -> Any:
        """The design steps that apply to a draft in hand, and one of them pressed.

        The POST twin of `_remixes`, and the same two questions asked of a spec in the
        body rather than of a hash in the path -- because the draft in the editor is
        usually not a game the library holds and has no hash anybody could name.

        **Nothing is written and nothing is named.** The draft arrives in the request, is
        read in memory and goes back changed, which is what `POST /api/design/check` and
        `POST /api/design/play` already do with a draft; `POST /api/design/save` is still
        the one route that keeps one.
        """
        spec = body.get("spec") or {}
        step = str(body.get("step") or "")
        if step:
            return {"remix": designing.draft_remix(spec, step, _seed(body.get("seed")))}
        # `back` is the draft the last press was made on: a step landing there is undo.
        back = body.get("back")
        return designing.draft_steps(spec, [back] if isinstance(back, dict) else [])

    def _save_design(self, body: dict[str, Any], restate: bool = True) -> Any:
        """Keep a hand-built game as a candidate, and let the library see it at once.

        The parent travels as a *spec* and not only as a hash, because the refusal that
        matters is "you changed nothing" and answering it needs the parent's own settled
        hash rather than the one its filename was derived from. A parent this library has
        never heard of is not an error: the draft is simply not a fork of anything it can
        compare against, and `designing.save` falls back to the hash it was given.

        `restate` is whether a re-save of kept rules may give them its name and
        description (Q595). Here it always may: the local application is one person's.
        The site says no for rules somebody else saved first and for a published game.
        """
        parent = str(body.get("parent", ""))
        known = self.known().get(parent) if parent else None
        kept = designing.save(
            body.get("spec") or {},
            runs_dir=self.runs_dir,
            parent=parent,
            note=str(body.get("note", "")),
            parent_spec=None if known is None else known.spec,
            parent_name="" if known is None else known.name,
            restate=restate,
        )
        self.refresh()
        return kept

    def _draft_standing(self, digest: str, query: dict[str, str]) -> Any:
        """Where a draft is on the rungs, and the family it belongs to.

        Re-read, because a measurement or a save may have landed since the last listing.
        """
        games = {game.hash: game for game in self.refresh()}
        run = self.measurements.get(digest)
        return candidacy.standing(
            games,
            digest,
            self.runs_dir,
            self.ruleset_dir,
            self.archive_dir,
            measuring=None if run is None else run.as_dict(),
            parent=str(query.get("parent", "")),
            name=str(query.get("name", "")),
            last_seconds=self.measurements.last_seconds,
            build=stale_build() or "",
        )

    def _measure_design(self, body: dict[str, Any]) -> Any:
        """Queue a saved draft's canonical measurement.

        The same queue as a game page's button and a sweep, so it waits its turn behind
        either. Refused on a stale build before it is queued.
        """
        behind = stale_build()
        if behind is not None:
            raise ValueError(behind)
        return self.measurements.start(self._candidate(str(body["hash"]))).as_dict()

    def _nominate(self, body: dict[str, Any]) -> Any:
        """Mark a candidate for a maintainer. Nothing is written outside `runs/`."""
        kept = candidacy.nominate(
            self._candidate(str(body["hash"])), self.runs_dir, str(body.get("note", ""))
        )
        return {
            **kept,
            "nominations": candidacy.nominations(
                self.runs_dir, self.ruleset_dir, self.archive_dir),
        }

    def _play_draft(self, body: dict[str, Any]) -> Any:
        """Play a draft without keeping it.

        The loop this closes is the one that matters most: the only honest answer to
        *is this rule interesting* is playing it, and until now the way to find out was
        to write the idea into `runs/designed/` first. Ten moves of something you throw
        away should not leave a candidate behind.

        Nothing is written. The session holds the draft's own spec, the engine answers
        every question about it exactly as it does for a committed game, and the
        `scratch` flag is what keeps it out of the human-anchor record -- a game against
        a ruleset nobody else can look up is not evidence about where the ladder sits.
        """
        spec = designing.settled(body.get("spec") or {})
        problems = validate_spec(spec)
        if problems:
            raise designing.DesignError(
                "These rules aren't a complete game yet: " + "; ".join(problems[:4])
            )
        digest = spec_hash(spec)
        draft = library.Game(
            hash=digest,
            name=str(spec.get("name") or "this draft"),
            description=str(spec.get("description", "")),
            path=Path(),
            # Not "designed": that word means a candidate somebody kept, and this one
            # exists only for as long as the tab is open.
            source="draft",
            spec=spec,
        )
        # `show` opens the board part-way into a game the engine played: a rule at work
        # (`{"rule": key}`) or a game that did not finish (`"unfinished"`). Both seats
        # are people then, so nothing plays on until somebody moves.
        show = body.get("show")
        demo = None
        if show == "unfinished":
            demo = designing.unfinished(spec)
        elif isinstance(show, dict):
            demo = designing.demonstrate(spec, str(show.get("rule", "")))
        elif show is not None:
            raise ValueError('show must be "unfinished" or {"rule": <key>}')
        default = [HUMAN, HUMAN] if demo else [HUMAN, "minimax:nodes=1600"]
        roster = opponents.for_game(digest, self.heuristics_dir, spec=spec)
        seats = [opponents.seat_for(str(seat), roster) for seat in body.get("seats") or default]
        agents = [entry.agent for entry in roster]
        if len(seats) != 2 or any(seat not in agents for seat in seats):
            raise ValueError(f"seats must be two of: {', '.join(agents)}")
        session = self.sessions.start(
            draft, (str(seats[0]), str(seats[1])), seed=seed_asked(body.get("seed")),
            scratch=True,
        )
        # The rules and pieces panels beside the board are `library.detail`'s, fetched by
        # hash for every other game. A draft has no hash to fetch by, so its detail
        # travels with the position -- the same call, on the same spec, in one request.
        if demo is None:
            return {
                **session.snapshot(),
                "detail": library.detail(draft),
                "watching": _watching(seats, roster),
            }
        # Replayed as a stored game is, by the engine's step kinds: a round's two moves are
        # in seat order, which a spy game's announced round does not choose in (Q935c).
        for move in demo["moves"]:
            session.replay(move)
        return {
            **session.snapshot(),
            "detail": library.detail(draft),
            "replay": designing.timeline(session),
            "demo": {"start": demo["start"], "says": demo["says"]},
        }

    def _build_opponent(self, body: dict[str, Any]) -> Any:
        """Turn a set of weights into a playable opponent for one ruleset.

        The ruleset has to exist, and is looked up rather than trusted: a heuristic is
        per-ruleset by construction -- the feature vocabulary normalises by board size,
        and a piece-value table is as long as the ruleset has pieces -- so a digest filed
        under a hash nothing here holds is an opponent that can never be built.
        """
        digest = str(body["hash"])
        game = self.game(digest)
        return designing.build_opponent(
            ruleset_hash=digest,
            name=str(body.get("name", "")),
            plan=str(body.get("plan", "")),
            weights=list(body.get("weights") or []),
            # Multipliers of what the rules imply, so the spec has to travel with them:
            # what is stored is an absolute valuation, and turning one into the other
            # needs the derived table.
            values=list(body.get("values") or []),
            spec=game.spec,
            root=self.heuristics_dir,
        )

    # ------------------------------------------------------- the human anchor

    def human_summary(self, ruleset_hash: str) -> Any:
        """What a person's games say about where the ladder sits, for one ruleset.

        The opponents' ratings come from the picker's own stored table and are held
        fixed: only the person is fitted. Absent means the picker has not been rated
        here, which leaves the games reportable and the anchor not computable -- a
        different thing from a person with no rating.
        """
        return self.humans.summary(
            ruleset_hash, picker_ratings(self.ratings.stored(ruleset_hash))
        )

    def _note_human(self, session: Any, snapshot: dict[str, Any]) -> None:
        """Record a human game, when there is exactly one person in it.

        Called on every session action rather than on a "game over" event, because there
        is no such event: the page discovers the game is over by looking at the position
        it got back. Both writes are idempotent, so looking twice records once.

        Two humans, or none, is not an anchor -- a person playing themselves says
        nothing about the ladder, and two agents playing each other is a tournament with
        extra steps. Neither is a draft: `web/human.py` is keyed by ruleset hash and read
        back by hash, so a game against rules that were never saved would leave a row
        nothing can ever name.
        """
        if session.scratch:
            return
        seats = list(session.seats)
        if seats.count(HUMAN) != 1:
            return
        seat = seats.index(HUMAN)
        opponent = seats[1 - seat]
        outcome = snapshot.get("outcome")
        if outcome is None:
            return
        self.humans.note_result(
            session=session.id,
            ruleset_hash=session.game.hash,
            ruleset_name=session.game.name,
            opponent=opponent,
            human_seat=seat,
            seed=session.seed,
            plies=int(snapshot.get("ply", 0)),
            winner=outcome.get("winner"),
            reason=str(outcome.get("reason", "")),
            moves=list(session.played),
            played=session.played_by[1 - seat],
        )

    # ------------------------------------------------- what becomes of a candidate

    def _candidate(self, digest: str) -> corpus.Candidate:
        return corpus.find(
            digest,
            runs_dir=self.runs_dir,
            ruleset_dir=self.ruleset_dir,
            archive_dir=self.archive_dir,
        )

    def _start_test(self, body: dict[str, Any]) -> Any:
        """Measure one game on the roster a committed score means, on a thread.

        Does only the work the stored basis says is missing -- seconds for a weights
        bump, minutes for a replay -- unless `force`, which replays regardless.

        `seeds` is `one` or `four`, and absent means whatever the game's rung expects.
        The promotion panel asks for four, because that is what `promotion_blocker`
        demands and a panel that offered a rung the next one refuses would be a trap.
        """
        return self.measurements.start(
            self._candidate(str(body["hash"])),
            force=bool(body.get("force")),
            seeds=_seed_set(body.get("seeds")),
        ).as_dict()

    # ------------------------------------------------------- the scores, as a whole

    def scoring(self) -> dict[str, Any]:
        """What every corpus measurement rests on, and whether a sweep is running.

        `chess-variant-sandbox measure --check` for the browser, and the same `corpus.survey` behind
        it, so the panel cannot come to a different conclusion from the command line
        about what is out of date.
        """
        statuses = corpus.survey(
            runs_dir=self.runs_dir,
            ruleset_dir=self.ruleset_dir,
            archive_dir=self.archive_dir,
        )
        sweep = self.measurements.sweep()
        return {
            "games": [status.as_dict() for status in statuses],
            "current": sum(1 for status in statuses if status.state == "current"),
            "stale": sum(1 for status in statuses if status.state == "stale"),
            "absent": sum(1 for status in statuses if status.state == "absent"),
            # One count per repair, and all three, because `Staleness.needs` has three
            # answers. Counting only two put every `reanalyse` in with the replays and
            # told a reader that games nothing has invalidated have to be played again.
            "rescore": sum(1 for status in statuses if status.needs == "rescore"),
            "reanalyse": sum(1 for status in statuses if status.needs == "reanalyse"),
            "replay": sum(1 for status in statuses if status.needs == "replay"),
            "roster": list(corpus.CANONICAL_AGENTS),
            "gamesPerPairing": corpus.CANONICAL_GAMES,
            "weights": DEFAULT_WEIGHTS,
            # The footgun in bold in `CLAUDE.md`, asked before a button is offered
            # rather than after twenty minutes have been spent: a core built from older
            # sources does not fail, it answers with the previous build.
            "build": stale_build() or "",
            "sweep": None if sweep is None else sweep.as_dict(),
            # Q705. The refresher, if one runs over this `runs/` (`web/refreshing.py`):
            # the maintainer's, like the whole panel.
            "refresher": refreshing.view(self.runs_dir),
            # EV6 (Q690). How much each kept game's stored score has moved between its
            # revisions, beside its interval (`history.volatility`), in words.
            "volatility": history.volatility_rows(
                self.runs_dir, {status.hash: status.name for status in statuses}),
        }

    def _start_sweep(self, body: dict[str, Any]) -> Any:
        """Bring several measurements up to date, on one thread, in order.

        Scope is `stale` or `all`, and the difference is what it costs: `stale` runs
        only what the survey says is out of date -- often a re-score of seconds -- while
        `all` replays every game in the corpus and is measured in hours. Named hashes
        narrow it further, which is what the game page's own button uses when a person
        wants one game re-measured out of the queue's order.
        """
        scope = str(body.get("scope", "stale"))
        if scope not in ("stale", "all"):
            raise ValueError("scope must be 'stale' or 'all'")
        wanted = [str(digest) for digest in (body.get("hashes") or [])]
        statuses = corpus.survey(
            wanted or None,
            runs_dir=self.runs_dir,
            ruleset_dir=self.ruleset_dir,
            archive_dir=self.archive_dir,
        )
        if scope == "stale":
            statuses = [status for status in statuses if not status.current]
        if not statuses:
            raise ValueError("there is nothing to measure, because every score is current")
        # The build check is the server's, not the page's, so the command line and the
        # browser refuse the same run for the same reason.
        behind = stale_build()
        if behind is not None:
            raise ValueError(behind)
        queue = [
            (self._candidate(status.hash), status.needs or "replay") for status in statuses
        ]
        return self.measurements.start_sweep(
            queue, scope=scope, force=scope == "all"
        ).as_dict()

    def _stop_sweep(self) -> Any:
        sweep = self.measurements.stop_sweep()
        return {"sweep": None if sweep is None else sweep.as_dict()}

    def _keep(self, body: dict[str, Any], promote: bool) -> Any:
        """Write the spec into `archive/` or `rulesets/`, and re-read the library.

        The refusals are `corpus`'s, not this file's -- a browser must not be able to
        put a game in the corpus that the command line would decline to.
        """
        candidate = self._candidate(str(body["hash"]))
        name = str(body.get("name", "")).strip()
        if promote:
            path = corpus.promote(
                candidate,
                reason=str(body.get("reason", "")),
                ruleset_dir=self.ruleset_dir,
                runs_dir=self.runs_dir,
                name=name,
            )
        else:
            path = corpus.archive_game(
                candidate,
                note=str(body.get("note", "")),
                archive_dir=self.archive_dir,
                runs_dir=self.runs_dir,
                name=name,
            )
        # The library is keyed by hash and the hash has not changed -- but the game now
        # lives somewhere else, and the panel that offered this action reads that.
        self.refresh()
        return {
            "hash": candidate.hash,
            "stage": "committed" if promote else "archived",
            "path": str(path),
            "name": name or candidate.name,
        }

    def _start_rating(self, body: dict[str, Any]) -> Any:
        """Rate everything in this ruleset's picker, on a thread.

        Its own table, in its own file, and never the one the index reads -- see
        `web/opponents.py` for why that separation is not a tidiness preference.
        """
        digest = str(body["hash"])
        game = self.game(digest)
        run = self.ratings.start(
            spec=game.spec,
            ruleset_hash=digest,
            name=game.name,
            roster=self.roster(digest),
            games=max(1, min(int(body.get("games", 2)), 6)),
            seed=int(body.get("seed", 0)),
        )
        return run.as_dict()

    def _start_session(self, body: dict[str, Any]) -> Any:
        roster = self.roster(str(body["hash"]))
        # A game played in rounds seats its own rung of the name asked for (Q918c).
        seats = [
            opponents.seat_for(str(seat), roster)
            for seat in body.get("seats") or ["human", "minimax:nodes=1600"]
        ]
        agents = [entry.agent for entry in roster]
        if len(seats) != 2 or any(seat not in agents for seat in seats):
            raise ValueError(f"seats must be two of: {', '.join(agents)}")
        session = self.sessions.start(
            self.game(str(body["hash"])),
            (str(seats[0]), str(seats[1])),
            # A fresh seed unless the request names one (Q987): `play.seed_asked`.
            seed=seed_asked(body.get("seed")),
        )
        # Noted before a move is played, so a game abandoned halfway through still
        # leaves a row. The gap between started and finished is the one bias automatic
        # recording cannot remove, and it is reported rather than hidden.
        if list(session.seats).count(HUMAN) == 1:
            seat = list(session.seats).index(HUMAN)
            self.humans.note_start(
                session=session.id,
                ruleset_hash=session.game.hash,
                ruleset_name=session.game.name,
                opponent=session.seats[1 - seat],
                human_seat=seat,
                seed=session.seed,
                played=session.played_by[1 - seat],
            )
        return session.snapshot()

    def _session_action(self, session_id: str, action: str, body: dict[str, Any]) -> Any:
        session = self.sessions.get(session_id)
        if action == "move":
            snapshot = session.play(int(body["move"]))
        elif action == "agent":
            snapshot = session.play_agent()
        elif action == "undo":
            # Deliberately not recorded and deliberately not un-recorded. Taking back a
            # move is how a person explores a position; a finished game that was then
            # rewound still happened, and deleting the row would make the store a record
            # of games nobody chose to forget.
            return session.undo()
        else:
            raise KeyError(f"no such action: {action}")
        self._note_human(session, snapshot)
        return snapshot

    def _start_run(self, body: dict[str, Any]) -> Any:
        game = self.game(str(body["hash"]))
        run_id = secrets.token_urlsafe(6)
        run = self.runs.start(
            run_id=run_id,
            spec_path=game.path,
            parent_name=game.name,
            theme=str(body.get("theme", "")).strip(),
            directory=self.runs_dir / "design" / run_id,
            generations=max(1, min(int(body.get("generations", 4)), 40)),
            judge=bool(body.get("judge", True)),
        )
        return run.as_dict()


def _watching(seats: list[Any], roster: list[opponents.Opponent]) -> str:
    """Who is playing a draft, in plain words, or "" when a person is at the board.

    A game nobody is sitting at has to say who is playing it, because weak play by a
    weak player looks exactly like a broken game from the other side of the screen --
    which is what *watch it play itself* looked like on a draft where every piece could
    turn (Q265). What is named is the *budget*, because a budget is part of an agent's
    identity -- `minimax:nodes=400` and `minimax:nodes=6400` are two rated players, and
    the first sees about half as far into a wide game.

    The words are the picker's own name and plain words for each seat
    (`Opponent.name`, `Opponent.about`), so a rung relabelled there is relabelled here
    and the page never holds a second opinion about what a seat is. A seat the roster
    does not name is *the computer*, never its agent string.
    """
    if any(seat == HUMAN for seat in seats):
        return ""
    found = {entry.agent: entry for entry in roster}

    def named(agent: Any) -> str:
        entry = found.get(agent)
        if entry is None:
            return opponents.name_of(str(agent))
        return f"{entry.name} ({entry.about})" if entry.about else entry.name

    if seats[0] == seats[1]:
        return f"Both sides are played by {named(seats[0])}."
    return f"{named(seats[0])} is playing {named(seats[1])}."


def handler_for(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "chess-variant-sandbox"

        # Keep-alive, so a click costs one connection rather than one per request. What
        # makes it safe is that every response goes through `_send`, which always sets
        # Content-Length: that is how a reused connection finds the end of a body.
        # Measured 2026-09-11, 40 requests down one connection -- 63 ms and 43 sockets
        # left on the port at HTTP/1.0, against 12 ms and 2 here.
        protocol_version = "HTTP/1.1"

        # A kept-alive connection holds one of this server's threads, so an idle one has
        # to be let go of. Only reads and writes are timed, and an agent thinking inside
        # a request does neither, so this bounds a dead peer and never a slow search.
        timeout = 30

        # Whether the request in hand is a HEAD: `_send` then writes the headers a GET
        # would have and leaves the body off. The handler outlives a request on a
        # kept-alive connection, so `do_HEAD` sets it and puts it back.
        _head_only = False

        def do_HEAD(self) -> None:  # noqa: N802 - the stdlib's naming
            """The answer a GET would give, without its body (Q792: a monitor's
            ``HEAD /`` was a 501). The same dispatch, so every GET route has a HEAD and
            the two cannot drift apart; every response goes through `_send`, which is
            where the body is dropped. ``Content-Length`` stays the body's, as HTTP says
            a HEAD answers."""
            self._head_only = True
            try:
                self.do_GET()
            finally:
                self._head_only = False

        def do_GET(self) -> None:  # noqa: N802 - the stdlib's naming
            parsed = urlparse(self.path)
            # A game's own address (G3) is the page too; the site draws the game into it
            # for a reader without script, and here the router draws it as it always has.
            if parsed.path in PAGE_PATHS or game_page.digest_of(parsed.path) is not None:
                # Drawn per request: it names each script by its current bytes' digest.
                self._send(200, page().encode("utf-8"), "text/html; charset=utf-8")
                return
            held = assets.answer(parsed.path)
            if held is not None:
                # A script or stylesheet under its versioned name (G12): a year, because
                # the name changes whenever the bytes do.
                self._send(200, held[0], held[1], cache=IMMUTABLE)
                return
            if parsed.path.startswith(VENDOR_PREFIX):
                self._vendor(parsed.path[len(VENDOR_PREFIX):])
                return
            if parsed.path.startswith(shell.FONT_PREFIX):
                self._font(parsed.path[len(shell.FONT_PREFIX):])
                return
            if parsed.path.startswith(picture.PREFIX):
                self._picture(parsed.path[len(picture.PREFIX):])
                return
            if parsed.path in shell.icons():
                body, kind = shell.icons()[parsed.path]
                self._send(200, body, kind, cache=shell.ICON_CACHE)
                return
            if parsed.path == shell.SHELL_SCRIPT_ROUTE:
                self._script(shell.SHELL_SCRIPT_FILE)
                return
            if parsed.path == WORKER_ROUTE:
                self._worker()
                return
            if parsed.path == RULES_ROUTE:
                self._send(200, rules_markdown(), RULES_TYPE, cache=RULES_CACHE)
                return
            if parsed.path == RULES_PAGE_ROUTE:
                self._send(200, rules_page.local_bytes(), "text/html; charset=utf-8",
                           cache=RULES_CACHE)
                return
            if parsed.path in SCRIPTS:
                self._script(SCRIPTS[parsed.path])
                return
            if parsed.path.startswith(BROWSER_PREFIX):
                self._browser(unquote(parsed.path[len(BROWSER_PREFIX):]))
                return
            # Flattened, last value wins, so a repeated parameter is one answer rather
            # than a list every reader would have to unwrap.
            query = {
                name: values[-1]
                for name, values in parse_qs(parsed.query).items()
                if values
            }
            self._json(lambda: app.get(parsed.path, query))

        def _vendor(self, name: str) -> None:
            """One vendored file, or 404 for every other name.

            The manifest is the whole of what this route will serve, so the name is a
            lookup rather than a path to resolve. `..` and a subdirectory are simply not
            in it.
            """
            entry = vendored().get(name)
            if entry is None:
                self._send_json(404, {"error": f"no such file: {name}"})
                return
            body = (VENDOR_DIR / name).read_bytes()
            self._send(200, body, entry["type"], cache=IMMUTABLE)

        def _font(self, name: str) -> None:
            """One of the page's typefaces (`web/page/shell.py`), or 404 for any other name.

            The vendored files' rule: the font manifest is the whole of what this route
            serves, a lookup and never a path, and the version is in the file's name, so
            the answer is cached for a year.
            """
            entry = shell.fonts().get(name)
            if entry is None:
                self._send_json(404, {"error": f"no such file: {name}"})
                return
            self._send(200, (shell.FONT_DIR / name).read_bytes(), entry["type"], cache=IMMUTABLE)

        def _picture(self, name: str) -> None:
            """The link-preview picture (G5, `web/page/picture.py`), or 404 for any other
            name: the fonts' rule, a lookup in its manifest and never a path, and a year's
            cache, since a new picture is a new file name."""
            entry = picture.pictures().get(name)
            if entry is None:
                self._send_json(404, {"error": f"no such file: {name}"})
                return
            self._send(200, (picture.DIR / name).read_bytes(), entry["type"], cache=IMMUTABLE)

        def _script(self, path: Path) -> None:
            """One of the page's own scripts by its plain route, served as the worker
            script is.

            `SCRIPTS` is the whole of what reaches here, a lookup and never a path to
            resolve. `no-store`, because this name does not change when the bytes do; the
            pages name the versioned one (`assets.answer`), which is cached for a year.
            """
            self._send(200, path.read_bytes(), "text/javascript")

        def _worker(self) -> None:
            """The worker script, or 404 while the package carries none."""
            try:
                body = WORKER_FILE.read_bytes()
            except OSError:
                self._send_json(404, {"error": "this build has no measuring worker"})
                return
            self._send(200, body, "text/javascript")

        def _browser(self, asked: str) -> None:
            """One file of the browser build, or 404 for anything `browser_file` refuses.

            The two refusals are worded apart: the first says nothing is built here, the
            second that the build has no such name. Neither names a directory on the
            machine, since the site answers this route too.
            """
            found = browser_file(app.browser_dir, asked)
            if found is None and asked == BROWSER_MANIFEST:
                # The designer asks for the manifest on every open, afresh, because a
                # build can appear under a running server. With none here the answer is
                # "nothing", not "missing": a 404 is a console error on every page load
                # of a machine that simply has no build (B10). The page offers nothing.
                self._send(204, b"", "application/json")
                return
            if found is None:
                if not (app.browser_dir / BROWSER_MANIFEST).is_file():
                    error = "no browser build here: deploy/pyodide/build.sh makes one"
                else:
                    error = f"no such file in the browser build: {asked}"
                self._send_json(404, {"error": error})
                return
            target, kind, cache = found
            try:
                body = target.read_bytes()
            except OSError:
                # Replaced between the lookup and the read, by a rebuild.
                self._send_json(404, {"error": f"no such file in the browser build: {asked}"})
                return
            self._send(200, body, kind, cache=cache)

        def _forged(self) -> bool:
            """Refuse a POST that lacks the page's header, and say whether it was refused.

            A cross-site form can POST here, and the body is read as JSON whatever its
            Content-Type says. It cannot set a custom header, so the refusal comes before
            anything reads the body or the cookie. The body is then discarded unread, so
            that closing the connection delivers the refusal rather than aborting it.
            """
            if self.headers.get(REQUEST_HEADER) == REQUEST_VALUE:
                return False
            self.close_connection = True
            self._send_json(403, {"error": f"a POST must carry {REQUEST_HEADER}"})
            self._discard_body()
            return True

        def _discard_body(self, limit: int = DRAIN_LIMIT) -> None:
            """Read a refused request's body and throw it away, before the close.

            Measured on Windows 2026-09-19: a body still in the socket when the handler
            closes turns the close into an abort, and the client reading a moment later
            gets WinError 10053 instead of the 403 already written to it. Reproduced
            20 times in 20 by sending the headers and the body as two writes, which is
            what the suite hit on a busy machine when its own read came late.

            Only a body whose declared length is at most `limit` is drained; a longer
            claim is closed on unread, as before. It is read in pieces of at most
            `DRAIN_LIMIT` and none is kept, so a caller may pass a limit larger than it
            would ever hold, and a sender that stops short ends the drain.
            """
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return
            if not 0 < length <= limit:
                return
            while length > 0:
                piece = self.rfile.read(min(length, DRAIN_LIMIT))
                if not piece:
                    return
                length -= len(piece)

        def do_POST(self) -> None:  # noqa: N802 - the stdlib's naming
            if self._forged():
                return
            if self.headers.get("Transfer-Encoding"):
                # Content-Length is the only framing read here, and on a kept-alive
                # connection a body read wrong is worse than a wrong answer, because the
                # bytes left unread become the next request line. Refused and closed
                # instead, and not drained the way `_forged` drains: where this body ends
                # is exactly what is not known, so this answer may still be lost to the
                # abort described there.
                self.close_connection = True
                self._send_json(411, {"error": "send a body with Content-Length"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            path = urlparse(self.path).path
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                self._send_json(400, {"error": "body must be JSON"})
                return
            self._json(lambda: app.post(path, body))

        def _json(self, produce: Any) -> None:
            """Run an endpoint and answer, turning the three expected failures into
            statuses a page can act on rather than a stack trace it cannot."""
            try:
                self._send_json(200, produce())
            except KeyError as error:
                self._send_json(404, {"error": str(error).strip("'")})
            except (ValueError, TypeError) as error:
                self._send_json(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 - a local tool, one reader
                self._send_json(500, {"error": f"{type(error).__name__}: {error}"})

        def _send_json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send(
            self, status: int, body: bytes, content_type: str, cache: str = "no-store"
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            if self.close_connection:
                # Whoever decided this connection ends here -- an HTTP/1.0 client, a
                # `Connection: close`, or the refusal above -- the client is told so
                # rather than left to discover it on the next request.
                self.send_header("Connection", "close")
            self.end_headers()
            if not self._head_only:
                self.wfile.write(body)

        def end_headers(self) -> None:
            """The one policy, on every answer: a page, a script, JSON, a redirect, an
            error. The stdlib calls this last for each of them, so no route can miss it,
            and the site's own `end_headers` reaches it through `super()`."""
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            super().end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            """Quiet by default: a click is several requests and the console is where
            the design run's own report goes."""

    return Handler


class PortInUse(ValueError):
    """The port is already served by something else.

    A ``ValueError`` on purpose: `cli.main` prints those as one line and exits 2, and a
    busy port is a thing the person at the keyboard did, not a stack trace.
    """


class Server(ThreadingHTTPServer):
    """The project's server: threaded, and reusing a busy address everywhere but Windows.

    ``SO_REUSEADDR`` is not one behaviour under one name, so one constant cannot be right
    on both platforms.

    On Windows it lets a second process bind a port another process is **listening** on.
    Both stay alive, neither says anything, and only one is ever reached -- measured
    2026-09-07 across two processes, the second server binding in 8 trials of 8 and every
    request reaching the first. Off there, and the second bind fails with WinError 10048,
    which `listen` turns into a sentence. Windows pays nothing for that: a *connected*
    socket in TIME_WAIT does not hold the listening port, so a rebind straight after a
    close is already granted (measured 2026-09-11 with four server-side TIME_WAITs on the
    port, granted with the option on and off alike).

    On Linux it only reuses a socket left in TIME_WAIT, and a live listener still refuses
    the bind -- so the refusal above survives while the cost does not. Off, that cost the
    deployed site up to a minute of 502s per restart, 65 s from a SIGTERM and 66 s from a
    SIGKILL, measured on the box 2026-09-11.

    ``SO_EXCLUSIVEADDRUSE`` is the Windows option for *making* an exclusion binding on
    other processes, and it is deliberately not set. It buys nothing here -- with
    ``SO_REUSEADDR`` off, another process that sets it is already refused with WSAEACCES,
    measured with and without, 3 trials each -- and the two options are mutually
    exclusive, so setting both raises WinError 10022.
    """

    allow_reuse_address = os.name != "nt"


def listen(app: App, host: str = "127.0.0.1", port: int = 8765) -> Server:
    """Bind a server for this app, turning a busy port into a sentence.

    `chess_variant_sandbox/site/server.py` binds the same server around a shorter route table and has
    to refuse a busy port in the same words, so the refusal lives here rather than
    inside `serve`.
    """
    try:
        return Server((host, port), handler_for(app))
    except OSError as error:
        raise PortInUse(
            f"{host}:{port} is already in use, so this server did not start -- "
            f"another `chess-variant-sandbox serve` is the usual reason ({error}). "
            f"Stop it, choose another port with --port, or use --port 0 to be given a "
            f"free one; the port it gets is printed."
        ) from error


def warm(app: App) -> threading.Thread:
    """Pay for a first `/api/library`, on a thread, with the socket already bound.

    The corpus is sixty-odd games over several thousand files, and reading it *before*
    binding is what made a restart an outage rather than a pause: Q190 measured 21 s
    from stop to the first 200 on the box, 19.7 s of it this read, with the port closed
    throughout. Bound first, the same read happens while the server answers, so a person
    arriving in the middle of it gets the page and waits only for the listing.

    Runs the whole of `library_payload`, not only `refresh`. A process's first
    `/api/library` also runs the smoke gate and attempts the Fairy-Stockfish translation
    over every ruleset -- 7-13 s of the 71-ruleset corpus depending on how many cores the
    box gives this process (Q415) -- and until Q418 `warm` left both for whoever asked
    first. `library_payload`'s own lock (`App._warming`) is what makes this safe to run
    concurrently with a request: a listing that arrives while this thread is still inside
    it waits there rather than starting a second pass over the same specs, and finds
    every cache this thread reached already warm.

    The thread is a daemon because it holds nothing a shutdown has to wait for.
    """

    def read() -> None:
        started = time.monotonic()
        try:
            payload = app.library_payload()
        except Exception as error:  # noqa: BLE001 - a bad listing must not kill the server
            print(f"the library could not be read: {error}")
            return
        count = len(payload["games"])
        print(f"library: {count} rulesets in {time.monotonic() - started:.1f}s")
        # K2: each game's pieces alone on an empty board, once, so no game page waits on
        # its first visitor's engine questions (`library.piece_maps`, memoised by spec).
        known = list(app.known().values())
        reference = library.reference_of(known)
        for game in known:
            library.piece_maps(game, reference)

    thread = threading.Thread(target=read, name="library", daemon=True)
    thread.start()
    return thread


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    ruleset_dir: Path = library.RULESET_DIR,
    runs_dir: Path = library.RUNS_DIR,
    open_browser: bool = False,
    heuristics_dir: Path | str = "heuristics",
    archive_dir: Path | str = corpus.ARCHIVE_DIR,
    browser_dir: Path | str = BROWSER_DIR,
) -> None:
    """Run the application until interrupted."""
    app = App(ruleset_dir, runs_dir, heuristics_dir, archive_dir, browser_dir)
    server = listen(app, host, port)
    url = f"http://{host}:{server.server_port}{GAMES_PATH}"
    print(f"Chess variant sandbox is playable at {url}")
    print("press ctrl-c to stop")
    warm(app)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
