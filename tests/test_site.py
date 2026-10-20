"""The public site: which routes it registers, where its database comes from, and the store.

Tasks 1.1 and 1.2 of `docs/plan-website.md`. The rule under test is §2's -- a public site cannot
inherit the routes that spend money, write the repository, or queue minutes of work -- and
the way it is held is an allowlist in the app rather than a refusal at the reverse proxy.
So these tests ask the app what it will dispatch. Caddy refuses the paid and
repository-writing paths at the edge as well and that stays; two doors is the design, and
this is the one that cannot be got round by reaching the port directly. One test reads the
Caddyfile, to hold that nothing the app serves is refused at the edge.

The store's tests need a real Postgres and skip with a reason without one, so the file
still passes on a machine that has none. Each takes a scratch schema and drops it, which
is why they can be pointed at a server that has other things on it.

Getting one on this development machine, which has PostgreSQL 15 installed and its own
cluster behind a password nobody here has -- a cluster of your own, on a port of your own,
trusting whoever connects, deleted afterwards:

    "$PGBIN/initdb.exe" -D "$SCRATCH/pgdata" -U postgres --auth=trust --encoding=UTF8
    "$PGBIN/pg_ctl.exe" -D "$SCRATCH/pgdata" -l "$SCRATCH/pg.log" \\
        -o "-p 55433 -c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off \\
            -c full_page_writes=off" start
    "$PGBIN/psql.exe" -h 127.0.0.1 -p 55433 -U postgres -c "create database chess_variant_sandbox"
    CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55433/chess_variant_sandbox uv run pytest

`$PGBIN` is `/c/Program Files/PostgreSQL/15/bin` and `$SCRATCH` is anywhere outside the
repository. `pg_ctl ... stop -m fast` and delete the directory when the run is done. The
three settings that stop Postgres waiting on the disk are safe only because the cluster is
thrown away; they took a sixth off two of these modules (Q377). The site tests need the
`site` extra too (`uv sync --extra site`), or they skip for want of `psycopg`.
"""

from __future__ import annotations

import ast
import contextlib
import html
import importlib
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from http.client import HTTPConnection
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import node_session
import page_names
import page_text
import pytest
import site_store

from chess_variant_sandbox import cli
from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.site import (
    admin,
    awards,
    badges,
    browse,
    daily,
    db,
    identity,
    mail,
    migrate,
    milestones,
    quota,
    rating,
    routes,
    screening,
    unbeaten,
    usage,
    votes,
)
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import designing, library
from chess_variant_sandbox.web.page import marks
from chess_variant_sandbox.web.server import (
    CONTENT_SECURITY_POLICY,
    REQUEST_HEADER,
    REQUEST_VALUE,
    App,
    Server,
    handler_for,
)

# `serve_forever` looks for its stop flag every half second by default, so each test
# that stopped a server waited out that half second in `shutdown()`. 20 ms stops it at once.
SERVE_FAST = {"poll_interval": 0.02}

REPOSITORY = Path(__file__).resolve().parents[1]
# The deploy files are not in the public export (`docs/reference/public-fork.md`), so the
# tests that read them stand down there; in this repository they always run.
NEEDS_DEPLOY = pytest.mark.skipif(not (REPOSITORY / "deploy").is_dir(),
                                  reason="the public export carries no deploy/")
RULESETS = library.RULESET_DIR

# Where a database to connect to is named, when there is one. Unset on a development
# machine with no Postgres, which is why every test that needs one skips with a reason.
TEST_URL_ENV = "CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL"

# What §2 says a public site cannot inherit, spelled the way the page asks for it. The
# paid route and the repository-writing routes are the ones Caddy also refuses at the
# edge; the rest are the facilities the site does not have yet -- a measurement queue, a
# rating queue, a per-user opponent store, the human anchor.
FORBIDDEN = (
    ("POST", "/api/run"),
    ("GET", "/api/run/abc123def456"),
    ("GET", "/api/runs"),
    ("GET", "/api/scoring"),
    ("POST", "/api/scoring"),
    ("POST", "/api/scoring/stop"),
    ("POST", "/api/archive"),
    ("POST", "/api/promote"),
    ("POST", "/api/design/icon"),
    ("POST", "/api/design/measure"),
    ("POST", "/api/design/nominate"),
    ("POST", "/api/test"),
    ("GET", "/api/test/abc123def456"),
    ("POST", "/api/rate"),
    ("GET", "/api/rating/abc123def456"),
    ("POST", "/api/opponent"),
    ("POST", "/api/opponent/forget"),
    ("GET", "/api/human"),
    ("GET", "/api/human/abc123def456"),
)

# The designer's cheap steps, served to a signed-in person since Q221: the save, the
# playability check and the progress page. The optional description hint (Q396) follows a
# save, so it asks for the same sign-in.
MEMBERS = (
    ("POST", "/api/design/save"),
    ("POST", "/api/design/hint"),
    ("POST", "/api/design/screen"),
    ("GET", "/api/design/standing/abc123def456"),
)

# What stays withheld from a signed-in person too. The designer's measurement and
# nomination wait on `docs/plan-usage.md`'s first month; the game page's measurement and
# the picker rating behind *work them out again* are minutes of tournament.
STILL_WITHHELD = (
    ("POST", "/api/design/measure"),
    ("POST", "/api/design/nominate"),
    ("POST", "/api/test"),
    ("POST", "/api/rate"),
    ("GET", "/api/rating/abc123def456"),
)


# --------------------------------------------------------- reading the local app's table


def _parts(node: ast.expr) -> tuple[str, ...] | None:
    """A list of string constants, as a tuple, or None for anything else."""
    if not isinstance(node, ast.List):
        return None
    values = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return tuple(values)


def _equality(node: ast.AST) -> ast.Compare | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return node
    return None


def _exact(node: ast.AST) -> tuple[str, ...] | None:
    """`parts == ["api", "library"]`."""
    compare = _equality(node)
    if compare is None or not isinstance(compare.left, ast.Name) or compare.left.id != "parts":
        return None
    return _parts(compare.comparators[0])


def _prefix(node: ast.AST) -> tuple[str, ...] | None:
    """`parts[:2] == ["api", "game"]`."""
    compare = _equality(node)
    if compare is None or not isinstance(compare.left, ast.Subscript):
        return None
    target = compare.left
    if not isinstance(target.slice, ast.Slice):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "parts":
        return None
    return _parts(compare.comparators[0])


def _length(node: ast.AST) -> int | None:
    """`len(parts) == 3`."""
    compare = _equality(node)
    if compare is None or not isinstance(compare.left, ast.Call):
        return None
    call = compare.left
    if not isinstance(call.func, ast.Name) or call.func.id != "len":
        return None
    value = compare.comparators[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return value.value
    return None


def _local_routes(method: str) -> set[tuple[str, ...]]:
    """Every route `web/server.py` dispatches, read out of the source.

    The site's table is a classification of the local application's, so a route added
    there and never classified here would be silently refused rather than considered.
    Reading the source is what notices. A variable segment comes back as `<>`, which is
    what the site's own paths are compared as.
    """
    source = (REPOSITORY / "chess_variant_sandbox" / "web" / "server.py").read_text(encoding="utf-8")
    app = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "App"
    )
    function = next(
        node
        for node in app.body
        if isinstance(node, ast.FunctionDef) and node.name == method.lower()
    )
    found: set[tuple[str, ...]] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.BoolOp):
            lengths = [length for length in map(_length, node.values) if length]
            prefixes = [prefix for prefix in map(_prefix, node.values) if prefix]
            if lengths and prefixes:
                found.add(prefixes[0] + ("<>",) * (lengths[0] - len(prefixes[0])))
        exact = _exact(node)
        if exact:
            found.add(exact)
    return found


def _shape(route: routes.Route) -> tuple[str, ...]:
    return tuple("<>" if part.startswith("<") else part for part in route.parts)


class TestTheRouteTable:
    def test_every_local_route_is_classified(self) -> None:
        for method in ("GET", "POST"):
            local = _local_routes(method)
            assert len(local) > 5, f"the source reader found {len(local)} {method} routes"
            mine = {
                _shape(route)
                for route in routes.ALLOWED + routes.WITHHELD
                if route.method == method
            }
            missing = sorted("/".join(parts) for parts in local - mine)
            invented = sorted("/".join(parts) for parts in mine - local)
            assert not missing, f"{method} routes the site has not classified: {missing}"
            assert not invented, f"{method} routes the local app does not have: {invented}"

    def test_the_sites_own_routes_are_not_the_local_apps(self) -> None:
        """`OWN` is the site's, and it stays out of the classification above.

        A vote and a profile are rows the local application has no table for, so there is
        no local handler to allow or withhold. Putting one in `ALLOWED` would make the
        comparison against `web/server.py` report it as a route the local app does not
        have, and the honest fix is a separate list rather than a looser test.
        """
        assert routes.OWN
        for method in ("GET", "POST"):
            local = _local_routes(method)
            mine = {_shape(route) for route in routes.OWN if route.method == method}
            assert not (mine & local), f"{method}: the local app has one of these"
        assert routes.find("POST", "/api/vote") is not None
        assert routes.find("GET", "/api/profile") is not None

    def test_no_route_is_both_served_and_withheld(self) -> None:
        served = {(route.method, _shape(route)) for route in routes.ALLOWED}
        withheld = {(route.method, _shape(route)) for route in routes.WITHHELD}
        assert not served & withheld

    def test_every_route_says_what_it_is_for(self) -> None:
        assert all(route.note for route in routes.ALLOWED + routes.WITHHELD)

    @pytest.mark.parametrize(("method", "path"), FORBIDDEN)
    def test_the_forbidden_routes_are_unregistered(self, method: str, path: str) -> None:
        assert routes.find(method, path) is None

    def test_the_routes_that_remain_are_the_ones_a_person_plays_with(self) -> None:
        """Listing, one game, a session and the designer's read-only half."""
        for method, path in (
            ("GET", "/api/library"),
            ("GET", "/api/game/abc123def456"),
            ("GET", "/api/graph"),
            ("POST", "/api/session"),
            ("POST", "/api/session/xyz/move"),
            ("POST", "/api/design/check"),
            ("POST", "/api/design/load"),
        ):
            assert routes.find(method, path) is not None, f"{method} {path}"

    def test_the_designers_cheap_steps_are_the_routes_that_ask_for_a_sign_in(self) -> None:
        """Q221. Each carries the sentence a visitor is refused with, and no other served
        route of the local application's asks for one."""
        asking = {route for route in routes.ALLOWED if route.visitor}
        assert asking == {routes.find(method, path) for method, path in MEMBERS}
        assert all(route.visitor.startswith("Sign in") for route in asking)

    @pytest.mark.parametrize(("method", "path"), STILL_WITHHELD)
    def test_the_long_steps_and_the_nomination_stay_withheld(
        self, method: str, path: str
    ) -> None:
        assert routes.find(method, path) is None
        parts = routes.split(path)
        assert any(route.matches(method, parts) for route in routes.WITHHELD)

    @NEEDS_DEPLOY
    def test_the_vendored_files_are_classified_and_served_to_anybody(
        self, designer: SiteApp, recorded: FakeConnection
    ) -> None:
        """`docs/archive/plan-game-graph.md` §8.1: the explorer's one third-party file.

        Not a route -- the local application's handler answers it from inside the
        package before any dispatch, the way it answers `/`, and the site inherits that
        handler. So it is classified beside `PAGES` rather than in `ALLOWED`, and what
        this checks is that a signed-out visitor gets it, that the store is never asked,
        and that Caddy would not refuse it on the box.
        """
        assert "/static/vendor/" in routes.STATIC
        assert routes.find("GET", "/static/vendor/cytoscape-3.34.3.min.js") is None

        caddyfile = (REPOSITORY / "deploy" / "remote" / "Caddyfile").read_text(
            encoding="utf-8"
        )
        (pattern,) = re.findall(r"^\s*@forbidden path_regexp (\S+)\s*$", caddyfile, re.M)
        assert not re.compile(pattern).match("/static/vendor/cytoscape.min.js")

        server = Server(("127.0.0.1", 0), site_server.site_handler_for(designer))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            answer = browser.get("/static/vendor/cytoscape-3.34.3.min.js")
            assert answer.status == 200
            assert answer.body.startswith(b"/**")
            assert "immutable" in answer.headers["Cache-Control"]
            assert browser.get("/static/vendor/anything-else.js").status == 404
        finally:
            server.shutdown()
            server.server_close()
        # A constant read from the package: nobody's name, and no connection opened.
        assert recorded.executed == []

    def test_the_browser_build_and_its_worker_are_served_to_anybody(
        self,
        designer: SiteApp,
        recorded: FakeConnection,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`docs/archive/plan-client-compute.md` B4: `/browser/<path>` and the worker script.

        Classified in `STATIC` beside the vendored files, for the same reason: the local
        handler answers both before any dispatch and the site inherits it. A signed-out
        visitor gets them with B4's headers, the store is never asked, and Caddy would not
        refuse them. The build is B1's layout in miniature, since the route reads the
        layout and not Pyodide's bytes.
        """
        from chess_variant_sandbox.web import server as local_server

        assert "/static/measure-worker.js" in routes.STATIC
        assert "/browser/" in routes.STATIC
        assert routes.find("GET", "/browser/manifest.json") is None
        assert routes.find("GET", "/static/measure-worker.js") is None

        runtime = "pyodide-0.29.5"
        wasm = b"\x00asm\x01\x00\x00\x00"
        root = tmp_path / "build" / "browser"
        (root / runtime).mkdir(parents=True)
        (root / runtime / "pyodide.asm.wasm").write_bytes(wasm)
        (tmp_path / "build" / "secret.txt").write_text("not the build's", encoding="utf-8")
        (root / "manifest.json").write_text(
            json.dumps({
                "schema": 1,
                "runtime": runtime,
                "wheel": "wheel-0123456789ab/a.whl",
                "files": [{"path": f"{runtime}/pyodide.asm.wasm", "bytes": len(wasm)}],
            }),
            encoding="utf-8",
        )
        designer.browser_dir = root
        worker = tmp_path / "measure-worker.js"
        worker.write_bytes(b"self.onmessage = () => {};\n")
        monkeypatch.setattr(local_server, "WORKER_FILE", worker)

        server = Server(("127.0.0.1", 0), site_server.site_handler_for(designer))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            manifest = browser.get("/browser/manifest.json")
            assert manifest.status == 200
            assert manifest.headers["Cache-Control"] == "no-store"
            assert manifest.payload["runtime"] == runtime

            module = browser.get(f"/browser/{runtime}/pyodide.asm.wasm")
            assert module.status == 200
            assert module.body == wasm
            assert module.headers["Content-Type"] == "application/wasm"
            assert module.headers["Cache-Control"] == "public, max-age=31536000, immutable"

            script = browser.get("/static/measure-worker.js")
            assert script.status == 200
            assert script.headers["Content-Type"] == "text/javascript"
            assert script.body == worker.read_bytes()

            climbed = browser.get("/browser/../secret.txt")
            assert climbed.status == 404
            assert b"not the build's" not in climbed.body

            (root / "manifest.json").unlink()
            # No build is a quiet "nothing" for the manifest the designer always asks
            # for (B10), and a 404 for everything else.
            assert browser.get("/browser/manifest.json").status == 204
            assert browser.get(f"/browser/{runtime}/pyodide.asm.wasm").status == 404
            worker.unlink()
            assert browser.get("/static/measure-worker.js").status == 404
        finally:
            server.shutdown()
            server.server_close()
        # Constants read from files: nobody's name, and no connection opened.
        assert recorded.executed == []

    def test_the_pages_scripts_are_served_to_anybody(
        self, designer: SiteApp, recorded: FakeConnection
    ) -> None:
        """Q513: the application page's scripts and the site pages' own, as the worker
        script is -- `text/javascript`, `no-store`, the file's bytes -- to a signed-out
        visitor, with the store never asked, and nothing under either prefix that the
        tables do not name."""
        from chess_variant_sandbox.web import page as app_page

        assert app_page.SCRIPT_PREFIX in routes.STATIC
        assert site_pages.SCRIPT_PREFIX in routes.STATIC
        served = {**app_page.SCRIPTS, **site_pages.SCRIPTS}
        assert all(routes.find("GET", route) is None for route in served)

        server = Server(("127.0.0.1", 0), site_server.site_handler_for(designer))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            for route, file in served.items():
                answer = browser.get(route)
                assert answer.status == 200, route
                assert answer.headers["Content-Type"] == "text/javascript", route
                assert answer.headers["Cache-Control"] == "no-store", route
                assert answer.body == file.read_bytes(), route
            for route in ("/static/site/nothing.js", "/static/site/../pages.py",
                          "/static/page/nothing.js"):
                assert browser.get(route).status == 404, route
        finally:
            server.shutdown()
            server.server_close()
        assert recorded.executed == []

    @NEEDS_DEPLOY
    def test_caddy_refuses_no_static_path_the_site_serves(self) -> None:
        """The same two doors as the routes below, for what is served ahead of them."""
        caddyfile = (REPOSITORY / "deploy" / "remote" / "Caddyfile").read_text(encoding="utf-8")
        (pattern,) = re.findall(r"^\s*@forbidden path_regexp (\S+)\s*$", caddyfile, re.M)
        refused = re.compile(pattern)
        served = [path + "file.js" if path.endswith("/") else path for path in routes.STATIC]
        assert [path for path in served if refused.match(path)] == []

    @NEEDS_DEPLOY
    def test_caddy_refuses_no_route_the_site_serves(self) -> None:
        """A route works on the box only when both doors are open.

        Caddy answers before the app is asked, so a route served here and refused there
        is dead on the box. That is what the user hit saving a draft on 2026-09-14.
        """
        caddyfile = (REPOSITORY / "deploy" / "remote" / "Caddyfile").read_text(encoding="utf-8")
        (pattern,) = re.findall(r"^\s*@forbidden path_regexp (\S+)\s*$", caddyfile, re.M)
        refused = re.compile(pattern)
        served = [
            "/" + "/".join("abc123" if part.startswith("<") else part for part in route.parts)
            for route in routes.ALLOWED + routes.OWN
        ]
        assert [path for path in served if refused.match(path)] == []

    def test_a_sloppy_path_is_read_the_way_the_app_reads_it(self) -> None:
        """`//api//library/` reaches the same handler, so it must match the same route.

        The app drops empty parts before it dispatches, so a matcher that only knew the
        tidy spelling would leave a door open beside the one it shut. It is why the
        Caddyfile's refusal is a regexp rather than a list of paths, and this is the same
        property one layer in.
        """
        tidy = routes.find("GET", "/api/library")
        assert tidy is not None
        for sloppy in ("//api//library", "/api/library/", "//api//library//"):
            assert routes.find("GET", sloppy) is tidy
        for sloppy in ("//api//run", "/api/run/", "//api//design//measure"):
            assert routes.find("POST", sloppy) is None


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(RULESETS, library.RUNS_DIR)


class TestTheApp:
    def test_a_withheld_route_answers_as_one_that_never_existed(self, app: SiteApp) -> None:
        with pytest.raises(KeyError) as withheld:
            app.post("/api/promote", {"hash": "abc123def456"})
        with pytest.raises(KeyError) as invented:
            app.post("/api/nonsense", {})
        assert "no such endpoint: /api/promote" in str(withheld.value)
        assert "no such endpoint: /api/nonsense" in str(invented.value)

    def test_no_handler_runs_for_a_withheld_route(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allowlist gates before dispatch, not after it.

        `corpus.promote` writing the repository and then being refused is not a refusal,
        so the local application's dispatcher is replaced with something that fails, and
        every forbidden route still has to come back as a missing endpoint.
        """
        def reached(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("the local application's dispatcher was reached")

        monkeypatch.setattr(App, "get", reached)
        monkeypatch.setattr(App, "post", reached)
        for method, path in FORBIDDEN:
            with pytest.raises(KeyError):
                if method == "GET":
                    app.get(path)
                else:
                    app.post(path, {})

    def test_an_allowed_route_answers(self, app: SiteApp) -> None:
        """The allowlist lets a route through, not only refuses the others.

        `/api/design` rather than `/api/library`, because task 1.5 put the store's own
        columns on every card and the listing now opens a connection. That route is
        tested against a real cluster in `tests/test_site_unbeaten.py`; this one only has
        to prove the gate is not shut on everything.
        """
        payload = app.get("/api/design")
        assert payload["templates"] and payload["vocabulary"]

    def test_the_changelog_is_served_to_whoever_asks(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """*What's new* on the site: the same route the local application serves, over
        the same file, with nobody's name in it.

        A visitor reaches it -- `app.viewer` is nobody here -- because the changelog is a
        tracked file that says what this project can do, and the page is the way in. The
        site files the Baseline's moves in from the store (Q503n,
        `tests/test_site_resting.py`); with none, the file's days pass through unchanged.
        """
        from contextlib import nullcontext

        from chess_variant_sandbox.site import rounds, stages

        monkeypatch.setattr(app, "_connection", lambda: nullcontext(None))
        monkeypatch.setattr(stages, "moves", lambda connection: [])
        # Q503h's round lines are filed the same way; `test_site_rounds.py` has them.
        monkeypatch.setattr(rounds, "sweep", lambda connection, wait=True: None)
        monkeypatch.setattr(rounds, "news", lambda connection: [])
        payload = app.get("/api/changelog")
        assert payload["note"] == "" and payload["days"]
        assert payload["days"][0]["entries"][0]["heading"]
        # W8 (Q615): an entry marked for maintainers is not served on the site.
        served = [entry for day in payload["days"] for entry in day["entries"]]
        assert not [entry["heading"] for entry in served if entry.get("maintainers")]
        assert "Invite people from the Maintenance page" not in {e["heading"] for e in served}

    def test_the_mode_control_offers_a_visitor_sandbox_alone(self) -> None:
        """The page's half of §4's *a sandbox game needs no account*.

        `SiteApp._standing` sends `signedIn` and the mode control is its one reader, so
        the two are pinned together here: a visitor is shown a sentence about signing in
        rather than a menu whose second entry the server refuses.
        """
        from page_text import PAGE

        picker = PAGE[
            PAGE.index("function modePicker(") : PAGE.index("function standingHtml(")
        ]
        assert "standing.signedIn === false" in picker
        assert "Sign in to play a game" in picker
        assert picker.index("signedIn") < picker.index('<select id="mode"')

    def test_the_page_is_the_local_apps_page(self, app: SiteApp) -> None:
        """One page, served by one handler. The site is a route table, not a fork."""
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/") as response:
                assert b"<title>Chess variant sandbox</title>" in response.read()
            for method, path in (("GET", "/api/runs"), ("POST", "/api/promote")):
                asked = Request(
                    base + path,
                    data=b"{}" if method == "POST" else None,
                    headers={REQUEST_HEADER: REQUEST_VALUE},
                )
                with pytest.raises(HTTPError) as error:
                    urlopen(asked)
                assert error.value.code == 404, f"{method} {path}"
        finally:
            server.shutdown()
            server.server_close()

    def test_head_answers_like_get_without_a_body(self, app: SiteApp) -> None:
        """Q792: ``HEAD /`` was a 501. The site's own pages, its 404 and the local
        handler's routes it falls through to all answer a HEAD as their GET, bodiless."""
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            seen: set[int] = set()
            for path in ("/", "/privacy", "/no-such-page"):
                connection.request("GET", path)
                got = connection.getresponse()
                body = got.read()
                connection.request("HEAD", path)
                head = connection.getresponse()
                assert head.read() == b"", path
                assert head.status == got.status, path
                assert head.getheader("Content-Type") == got.getheader("Content-Type")
                assert head.getheader("Content-Length") == str(len(body)), path
                seen.add(got.status)
            assert 200 in seen and 404 in seen
            connection.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_a_cross_site_post_is_refused_before_dispatch(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§6's custom header. A form on another site can POST a `text/plain` body that
        parses as JSON, and `SameSite=Lax` only keeps the cookie off it. Without the
        header nothing is dispatched; with it, the same request reaches the app."""
        reached: list[str] = []

        def dispatched(path: str, body: object) -> dict[str, bool]:
            reached.append(path)
            return {"ok": True}

        monkeypatch.setattr(app, "post", dispatched)
        # Nobody is signed in, and no store is needed to say so.
        monkeypatch.setattr(app, "identify", lambda token: None)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        body = json.dumps({"handle": "ada", "password": PASSWORD}).encode("utf-8")
        try:
            forged = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            forged.request(
                "POST", "/api/signin", body=body, headers={"Content-Type": "text/plain"}
            )
            refused = forged.getresponse()
            assert refused.status == 403
            assert REQUEST_HEADER in json.loads(refused.read())["error"]
            forged.close()
            assert reached == []

            ours = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            ours.request(
                "POST",
                "/api/signin",
                body=body,
                headers={"Content-Type": "application/json", REQUEST_HEADER: REQUEST_VALUE},
            )
            answered = ours.getresponse()
            assert answered.status == 200
            # Read before closing. A client that closes on an unread body resets the
            # connection, and the handler waiting for the next request on it would go
            # out through `handle_error` and print a traceback this run has no use for.
            answered.read()
            ours.close()
            assert reached == ["/api/signin"]
        finally:
            server.shutdown()
            server.server_close()

    def test_a_refusal_survives_a_body_that_arrives_after_its_headers(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal has to *reach* the sender, not only be written to the socket.

        `_forged` answers before reading the body, and bytes still in the socket when
        the handler closes turn that close into an abort rather than an end. The abort
        discards what the client has not read yet, including the 403 -- WinError 10053,
        measured 20 times in 20 before the drain went in. This test forces the ordering
        by sending the headers and the body as two writes; the suite met it by chance,
        twice in about six runs, whenever the machine was busy enough for its own read
        to come late. Nothing is dispatched either way, which is the other half.
        """
        reached: list[str] = []
        monkeypatch.setattr(app, "post", lambda path, body: reached.append(path))
        monkeypatch.setattr(app, "identify", lambda token: None)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        body = json.dumps({"handle": "ada", "password": PASSWORD}).encode("utf-8")
        headers = (
            b"POST /api/signin HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n"
        )
        answer = b""
        try:
            client = socket.create_connection(("127.0.0.1", server.server_port), timeout=10)
            try:
                client.sendall(headers)
                # The gap is the whole point: the refusal is written, and the handler is
                # on its way to closing, while these bytes are still in flight.
                time.sleep(0.05)
                client.sendall(body)
                while True:
                    more = client.recv(4096)
                    if not more:
                        break
                    answer += more
            except OSError as error:
                pytest.fail(f"the refusal was lost: {type(error).__name__}: {error}")
            finally:
                client.close()
        finally:
            server.shutdown()
            server.server_close()
        assert answer.startswith(b"HTTP/1.1 403"), answer[:64]
        assert REQUEST_HEADER.encode("ascii") in answer
        assert reached == []


@pytest.fixture
def recorded() -> FakeConnection:
    """The store a designer test writes to. It records each statement and returns no rows."""
    return FakeConnection()


@pytest.fixture
def designer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: FakeConnection
) -> SiteApp:
    """A site over an empty library of its own, so a save lands in `tmp_path`.

    The playability check is a stub. What it decides is `test_candidacy.py`'s and what
    bounds it is `test_site_screening.py`'s; here only who reaches it is under test. The
    row a save writes is `tests/test_site_admin.py`'s, against a real store.
    """
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
    return SiteApp(
        tmp_path / "rulesets",
        tmp_path / "runs",
        archive_dir=tmp_path / "archive",
        connect=lambda: recorded,
    )


def _draft() -> dict[str, Any]:
    spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
    spec["name"] = "a draft somebody kept"
    return spec


def _ask(app: SiteApp, method: str, path: str) -> Any:
    return app.get(path) if method == "GET" else app.post(path, {"spec": _draft()})


def _unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the local application's dispatcher with one that fails the test."""

    def reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the local application's dispatcher was reached")

    monkeypatch.setattr(App, "get", reached)
    monkeypatch.setattr(App, "post", reached)


class TestTheCheapStepsNeedASignedInPerson:
    """Q221. The designer's save, playability check and progress page are served to a
    signed-in person and refused to a visitor in a sentence. The long steps stay withheld
    from everybody.

    None of this needs Postgres. The store is `recorded`, and these tests set the viewer
    themselves rather than reading a cookie.
    """

    def test_a_signed_in_person_saves_checks_and_reads_the_progress(
        self, designer: SiteApp, tmp_path: Path, recorded: FakeConnection
    ) -> None:
        designer.viewer = 7
        kept = designer.post("/api/design/save", {"spec": _draft()})
        # Q503a: a save keeps a draft, unlisted whatever the check says.
        assert kept["created"] is True and kept["listed"] is False
        assert (tmp_path / "runs" / "designed" / "specs" / f"{kept['hash']}.json").is_file()
        # hash, spec, name, owner, visibility, parent, origin, screen_passed
        [row] = recorded.rows("games")
        assert row[0] == kept["hash"]
        assert row[3:] == (7, "unlisted", None, "designed", True)

        standing = designer.get(f"/api/design/standing/{kept['hash']}")
        assert standing["saved"] is True and standing["hash"] == kept["hash"]
        assert designer.post("/api/design/screen", {"spec": _draft()}) == {"passed": True}

    def test_the_save_asks_which_badges_it_completed(
        self, designer: SiteApp, recorded: FakeConnection
    ) -> None:
        """Box E6. The designer badges are read in the save's own transaction, after the
        row. What they answer is `tests/test_badges.py`'s; that they are asked is this.
        """
        designer.viewer = 7
        kept = designer.post("/api/design/save", {"spec": _draft()})
        statements = [sql for sql, _params in recorded.executed]
        assert any("insert into games" in sql for sql in statements)
        read = next(index for index, sql in enumerate(statements) if "from awards" in sql)
        wrote = next(index for index, sql in enumerate(statements) if "insert into games" in sql)
        assert read > wrote, "the badges are read after the row they read"
        assert kept["created"] is True

    def test_a_visitor_is_asked_to_sign_in_before_anything_runs(
        self,
        designer: SiteApp,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        recorded: FakeConnection,
    ) -> None:
        _unreachable(monkeypatch)
        designer.viewer = None
        for method, path in MEMBERS:
            with pytest.raises(site_server.SignInFirst, match="^Sign in"):
                _ask(designer, method, path)
        assert list((tmp_path / "runs").iterdir()) == []
        assert recorded.executed == []

    def test_the_long_steps_are_withheld_whoever_asks(
        self, designer: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unreachable(monkeypatch)
        for viewer in (7, None):
            designer.viewer = viewer
            for method, path in STILL_WITHHELD:
                with pytest.raises(KeyError, match="no such endpoint"):
                    _ask(designer, method, path)

    def test_over_http_a_member_gets_200_a_visitor_400_and_a_long_step_404(
        self, designer: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signed_in: list[int | None] = [7]

        def identify(token: str | None) -> None:
            designer.viewer = signed_in[0]

        monkeypatch.setattr(designer, "identify", identify)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(designer))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            kept = browser.post("/api/design/save", {"spec": _draft()})
            assert kept.status == 200, kept.text
            digest = kept.payload["hash"]
            progress = browser.get(f"/api/design/standing/{digest}")
            assert progress.status == 200 and progress.payload["saved"] is True
            assert browser.post("/api/design/screen", {"spec": _draft()}).status == 200

            signed_in[0] = None
            for method, path in MEMBERS:
                answer = browser.get(path) if method == "GET" else browser.post(path, {})
                assert answer.status == 400, f"{method} {path}"
                assert answer.payload["error"].startswith("Sign in"), answer.text

            for viewer in (7, None):
                signed_in[0] = viewer
                for method, path in STILL_WITHHELD:
                    answer = browser.get(path) if method == "GET" else browser.post(path, {})
                    assert answer.status == 404, f"{method} {path}"
                    assert answer.payload["error"] == f"no such endpoint: {path}"
        finally:
            server.shutdown()
            server.server_close()


class TestTheDescriptionHintOnTheSite:
    """Q396. The optional hint reaches the site through the local application's route,
    behind the same sign-in as the save, and touches neither the save nor the store.

    What it asks and how it fails is `tests/test_jev_hint.py`'s; here only the site's
    door and the store are under test. No request leaves the machine: `_send` is a stub.
    """

    SAVED = {"hash", "name", "created", "path", "listed", "visibility", "screenPassed", "said",
             "rung", "rungPassed", "restated", "unchanged", "keptAsIs"}

    def test_with_no_key_the_save_is_unchanged_and_the_hint_is_off(
        self, designer: SiteApp, recorded: FakeConnection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chess_variant_sandbox.web import jev_hint

        monkeypatch.delenv(jev_hint.KEY_VAR, raising=False)
        monkeypatch.setattr(jev_hint, "_send", _no_request)
        designer.viewer = 7
        kept = designer.post("/api/design/save", {"spec": _draft()})
        assert set(kept) == self.SAVED
        before = len(recorded.executed)
        assert designer.post("/api/design/hint", {"hash": kept["hash"]}) == {"off": True}
        assert len(recorded.executed) == before

    def test_with_a_key_a_member_gets_the_hint_and_the_store_is_not_asked(
        self,
        designer: SiteApp,
        recorded: FakeConnection,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from chess_variant_sandbox.web import jev_hint

        monkeypatch.setenv(jev_hint.KEY_VAR, "test-key-not-a-real-one")
        monkeypatch.delenv(jev_hint.CAP_VAR, raising=False)
        monkeypatch.setattr(jev_hint, "_send", _one_contradiction)
        designer.viewer = 7
        spec = _draft()
        spec["description"] = "Kings fly across the board. A draft kept for a test."
        kept = designer.post("/api/design/save", {"spec": spec})
        assert set(kept) == self.SAVED
        before = len(recorded.executed)
        hint = designer.post("/api/design/hint", {"hash": kept["hash"]})
        assert [one["claim"] for one in hint["claims"]] == ["Kings fly across the board."]
        assert len(recorded.executed) == before
        assert (tmp_path / "runs" / "jev-hint" / f"{kept['hash']}.json").is_file()

    def test_a_visitor_is_refused_before_the_hint_runs(
        self, designer: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chess_variant_sandbox.web import jev_hint

        monkeypatch.setenv(jev_hint.KEY_VAR, "test-key-not-a-real-one")
        monkeypatch.setattr(jev_hint, "_send", _no_request)
        designer.viewer = None
        with pytest.raises(site_server.SignInFirst, match="^Sign in"):
            designer.post("/api/design/hint", {"hash": "abc123def456"})

    def test_the_privacy_notice_follows_the_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chess_variant_sandbox.web import jev_hint

        monkeypatch.delenv(jev_hint.KEY_VAR, raising=False)
        assert "TypeSafe" not in site_pages.privacy()
        monkeypatch.setenv(jev_hint.KEY_VAR, "test-key-not-a-real-one")
        notice = site_pages.privacy()
        assert "<h2>The description hint</h2>" in notice
        assert "sends that game's description and its rules to TypeSafe" in notice
        # The notice's own count is of what is kept about a person, which this is not.
        assert "seven things and nothing else" in notice


def _no_request(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("a request was sent to the hint's service")


def _one_contradiction(request: dict[str, Any], secret: str) -> dict[str, Any]:
    """The service's answer shape: the first sentence contradicted, the rest agreeing."""
    answers = {
        key: {"type": "choice", "choice": "contradicts" if key == "claim_0" else "agrees",
              "probabilities": {"contradicts": 0.8 if key == "claim_0" else 0.1,
                                "agrees": 0.1 if key == "claim_0" else 0.8,
                                "not_checkable": 0.1},
              "confidence": 0.7}
        for key in request["questions"]
    }
    return {"model": request["model"], "answers": answers,
            "usage": {"input_tokens": 900, "output_tokens": 0}}


class TestTheExtra:
    def test_the_driver_is_named_only_under_the_site_extra(self) -> None:
        """What a plain `uv sync` installs, read off the manifest rather than the venv.

        A test cannot ask what a sync *would* install without running one, so it asks the
        two things that decide it: the project has no runtime dependencies at all, and
        `psycopg` is named once, under `site`.
        """
        text = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r"^dependencies = \[\]\s*$", text, re.MULTILINE)
        assert text.count("psycopg") == 1
        assert re.search(r"^site = \[[^\]]*psycopg", text, re.MULTILINE)


class TestTheIndexBandReachesTheSite:
    """The band is the local application's, and the site adds nothing to it.

    Box B3 of `docs/archive/plan-meta-game.md`. The store has a game's headline and not
    its interval, so the three counts can only come from the files -- the site's listing
    is the local application's, decorated and ordered, and both of those steps have to
    leave a key they have never heard of alone. A `Columns` key that shadowed it, or a
    copy that dropped it, would leave the site's cards blank while the local
    application's were fine.
    """

    BAND = {"clears": 40, "clearedBy": 12, "close": 18, "of": 70}

    def test_decorating_and_ordering_leave_it_alone(self) -> None:
        cards = browse.decorate(
            [{"hash": "aaaa", "name": "one", "index": 0.5, "band": self.BAND},
             {"hash": "bbbb", "name": "two", "index": 0.9, "band": None}],
            {},
        )
        assert [card["band"] for card in cards] == [self.BAND, None]

        ordered = browse.order(cards, browse.INDEX)
        assert [card["hash"] for card in ordered] == ["bbbb", "aaaa"]
        assert ordered[1]["band"] == self.BAND

    def test_no_store_column_is_called_band(self) -> None:
        """The one way the site could overwrite it without anybody noticing."""
        columns = browse.Columns(
            game="aaaa", stars=votes.NONE, human_games=0,
            tag=unbeaten.tags_from(("aaaa",), (), {}, {})["aaaa"],
        )
        assert "band" not in columns.as_dict()


class TestTheDatabaseUrl:
    def test_the_site_refuses_without_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(db.ENV_VAR, raising=False)
        with pytest.raises(db.ConfigError) as error:
            db.database_url()
        message = str(error.value)
        assert "\n" not in message
        assert db.ENV_VAR in message and db.SOCKET_URL in message

    def test_the_refusal_comes_before_anything_is_built_or_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def reached(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("the site started before it knew where its database was")

        monkeypatch.delenv(db.ENV_VAR, raising=False)
        monkeypatch.setattr(site_server, "SiteApp", reached)
        monkeypatch.setattr(site_server, "listen", reached)
        with pytest.raises(db.ConfigError):
            site_server.serve()

    def test_a_blank_value_is_no_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(db.ENV_VAR, "   ")
        with pytest.raises(db.ConfigError):
            db.database_url()

    def test_only_one_module_reads_the_environment_for_it(self) -> None:
        readers = sorted(
            path.relative_to(REPOSITORY).as_posix()
            for path in (REPOSITORY / "chess_variant_sandbox").rglob("*.py")
            if db.ENV_VAR in path.read_text(encoding="utf-8")
        )
        assert readers == ["chess_variant_sandbox/site/db.py"]

    def test_a_password_never_reaches_the_journal(self) -> None:
        assert db.redacted(db.SOCKET_URL) == db.SOCKET_URL
        assert (
            db.redacted("postgresql://chess_variant_sandbox:hunter2@localhost:5432/chess_variant_sandbox")
            == "postgresql://chess_variant_sandbox:***@localhost:5432/chess_variant_sandbox"
        )
        assert "hunter2" not in db.redacted("postgresql:///chess_variant_sandbox?host=/tmp&password=hunter2")


@contextlib.contextmanager
def _without_psycopg():
    """Import psycopg as if it had never been installed, wherever this runs."""
    class Blocker:
        def find_spec(self, name: str, path: object = None, target: object = None) -> None:
            if name == "psycopg" or name.startswith("psycopg."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    blocker = Blocker()
    saved = sys.modules.pop("psycopg", None)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        if saved is not None:
            sys.modules["psycopg"] = saved


class TestTheDriver:
    def test_the_module_body_imports_no_driver(self) -> None:
        """A checkout that never installed the extra must still import the site.

        The module is re-executed with psycopg unimportable, which is the property: the
        driver is imported inside `connect` and nowhere else. Otherwise `chess-variant-sandbox --help`
        in a plain checkout is a stack trace about a database.
        """
        with _without_psycopg():
            importlib.reload(db)
            assert db.database_url is not None

    def test_connecting_without_a_driver_names_the_extra(self) -> None:
        with _without_psycopg(), pytest.raises(db.ConfigError) as error:
            db.connect("postgresql:///nothing")
        assert "uv sync --extra site" in str(error.value)

    def test_it_opens_a_connection(self) -> None:
        """The one thing 1.1 owes the database: that the connection point works.

        No schema and no tables -- those are 1.2's. `select 1` is the whole query.
        """
        url = os.environ.get(TEST_URL_ENV, "").strip()
        if not url:
            pytest.skip(f"{TEST_URL_ENV} is unset: no database to open a connection to")
        pytest.importorskip("psycopg", reason="the site extra is not installed here")
        with db.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute("select 1")
            assert cursor.fetchone() == (1,)

    @pytest.mark.parametrize("zone", ["America/Los_Angeles", "Pacific/Kiritimati"])
    def test_a_stamp_reads_back_on_utcs_calendar_whatever_the_session_zone(
        self, zone: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q744. A connection asked for a zone either side of UTC still reads stamps in UTC.

        Midnight UTC is the evening before in Los Angeles and the afternoon of the same
        day at UTC+14; the page prints a stamp's first ten characters, so either zone
        leaking into the session would name a day the rest of the site does not.
        `PGTZ` is libpq's, read when the connection opens.
        """
        url = os.environ.get(TEST_URL_ENV, "").strip()
        if not url:
            pytest.skip(f"{TEST_URL_ENV} is unset: no database to open a connection to")
        pytest.importorskip("psycopg", reason="the site extra is not installed here")
        monkeypatch.setenv("PGTZ", zone)
        with db.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute("select '2026-09-02 00:00:00+00'::timestamptz")
            (stamp,) = cursor.fetchone()
            connection.rollback()
            cursor.execute("show timezone")
            assert cursor.fetchone() == ("UTC",)
        assert stamp.isoformat() == "2026-09-02T00:00:00+00:00"


# ------------------------------------------------------------------- 1.2, the store

# Every table the store's first migration makes, with its columns in the order the DDL
# declares them. It is written out here rather than read from the file so that §3 of the
# plan and the schema cannot part without a test saying so: a column added to the
# migration and not to the plan fails here, and so does the reverse.
#
# `matches.left` is the plan's own name, quoted in the DDL because Postgres refuses it
# bare; `information_schema` reports the unquoted spelling, which is why it reads plainly
# here. `votes.voter` and `votes.note` are §3's `user` and `text`, renamed -- the
# migration's header says why each was handled the way it was.
STORE: dict[str, tuple[str, ...]] = {
    # `forgotten` is `0005_leaving.sql`, task 1.8's: when somebody left, which is not the
    # same as when somebody was killed. Last in the tuple because that is where `alter
    # table` puts it. The same migration drops `email`'s `not null`, so that leaving can
    # actually empty the column rather than writing a sentinel address into it.
    "users": (
        "id",
        "handle",
        "email",
        "password",
        "invited_by",
        "created",
        "role",
        "revoked",
        "forgotten",
        # `0024_move_email.sql`, Q572. The move email's box, off by default, and the
        # token of its one-click off.
        "move_email",
        "mail_off",
        # `0026_member_page.sql`, Q596. *Show my five-star games on my page*, off by
        # default.
        "show_favourites",
        # `0027_takes_challengers.sql`, Q606. *The maker takes challengers*, off by
        # default.
        "takes_challengers",
        # `0028_invites_to_give.sql`, Q719. How many invitations this person may still
        # send themselves, set on `/admin`, zero by default.
        "invites_to_give",
        # `0030_spot_check.sql`, Q503e. When a spot check found a game the site's replay
        # did not reproduce; null for everybody until one does.
        "flagged",
    ),
    # `email` is `0004_invite_email.sql`, task 1.3's: an invitation goes to one address,
    # so the code in the link belongs to it and the join asks for a handle and a password
    # and nothing else. Last in the tuple because that is where `alter table` puts it.
    "invites": ("code", "issuer", "uses_left", "expiry", "created", "email"),
    "sessions": ("token", "user_id", "created", "seen", "expires"),
    "games": (
        "hash",
        "spec",
        "name",
        "owner",
        "visibility",
        "parent",
        "origin",
        "created",
        "nominated",
        "revoked",
        # `retired` is `0007_retired.sql`, Q287's: when the spec left the tree. Last in
        # the tuple because that is where `alter table` puts it. A separate column from
        # `revoked` because an admin's kill switch and the loader's reading of the tree
        # are separate acts that must not overwrite each other.
        "retired",
        # `deleted` is `0010_deleted.sql`, Q303's: when the creator or an admin took the
        # rules away for good. A third column for the same reason, and the strongest of
        # the three -- the same migration drops `spec`'s `not null` and holds the pair
        # together with a check, so a deleted row cannot carry rules and a live one
        # cannot lack them.
        "deleted",
        # `screen_passed` and `published` are `0016_drafts.sql`, Q503a's: a save keeps a
        # draft and records the check's verdict here, and *Publish* stamps when the game
        # went on trial.
        "screen_passed",
        "published",
        # `rested` and `woken` are `0020_resting.sql`, Q503i's: when the sweep took a trial
        # game nobody plays off the listing, and when its designer published it again.
        "rested",
        "woken",
        # `terms_accepted` and `terms_version` are `0029_terms.sql`, Q722's: when the
        # designer agreed to the terms at *Publish*, and which version. Both or neither.
        "terms_accepted",
        "terms_version",
        # `publish_asked` and `publish_stopped` are `0031_publish_waits.sql`, Q742's: a
        # press of *Publish* waiting for the site's check of the score, and why the last
        # one was called off.
        "publish_asked",
        "publish_stopped",
    ),
    "measurements": (
        "hash",
        "directory",
        "state",
        "basis",
        "headline",
        "ladder",
        "top",
        "top_rating",
        "updated",
        # `0008_awards.sql`, box E1 of `docs/archive/plan-meta-game.md`. Both ends of the
        # interval, which lived only in each run's `index-<weights>.json`: a badge that
        # tests a game's whole range does so inside a vote's transaction, which should
        # not be opening files. Last in the tuple because that is where `alter table`
        # puts them.
        "ci_low",
        "ci_high",
    ),
    "ratings": (
        "id",
        "game",
        "player_user",
        "player_agent",
        "rating",
        "deviation",
        "volatility",
        "matches",
        "established",
        "updated",
    ),
    "matches": (
        "id",
        "game",
        "mode",
        "seat0_user",
        "seat0_agent",
        "seat1_user",
        "seat1_agent",
        "seed",
        "moves",
        "result",
        "reason",
        "left",
        "clock",
        "started",
        "moved",
        "finished",
        "source",
        "top",
        "seat0_rating_before",
        "seat0_deviation_before",
        "seat0_rating_after",
        "seat0_deviation_after",
        "seat1_rating_before",
        "seat1_deviation_before",
        "seat1_rating_after",
        "seat1_deviation_after",
        # `0002_top_group.sql`. The tied top this match was started against, beside the
        # bit that says whether its own opponent was in it -- task 1.5 found that one
        # boolean cannot say both *the top* and *which of the tied*.
        "top_players",
        # `0009_rematch.sql`. The match this one followed, written only by the board's
        # rematch control, so null means *not a rematch* rather than *unknown*.
        "rematch_of",
        # `0014_depths.sql`, Q424. How far the machine looked before each move, one entry
        # per move beside `moves`; null on a row that never kept any.
        "depths",
        # `0018_finishes.sql`, Q502a. Take-backs since the opening position, which a
        # finish copies so a win found with the undo button says so.
        "takebacks",
        # `0022_correspondence.sql`, Q570. A standing draw offer between two people (the
        # seat and the ply it was made at), and when each move landed, for those games.
        "draw_offer",
        "draw_offer_ply",
        "times",
        # `0024_move_email.sql`, Q572. When each seat last read the running game, so a
        # move made while it is open sends no email.
        "seen0",
        "seen1",
        # `0025_played.sql`, Q709. Which net a learned seat really played, `net:<12 hex>`,
        # null for every other seat.
        "seat0_played",
        "seat1_played",
        # `0032_human_data.sql`, Q767 (H1 to H3 of `docs/reference/human-data.md`). Think time per
        # move against a computer player, what each take-back undid, and each side's rating
        # when a practice game started.
        "thinks",
        "undone",
        "standing",
        # `0033_pinned_and_alarm.sql`, Q768 (H4). Who the computer seats were beyond their
        # agent strings: the core's build, the tables and the values, per identity.
        "pinned",
        # `0035_draw_offers.sql`, Q790. Every draw offer made on the row, with the core's
        # judgement where the computer answered it.
        "draw_offers",
        # `0036_sealed_moves.sql`, Q918f. The one move sealed in a round between two
        # people not yet revealed: its seat, the move and when; read by
        # `site/correspondence.py` alone.
        "sealed_move",
        # `0037_proven_positions.sql`, Q605 (R9b). A try at converting a proven position:
        # its id in the game's record and the length of the prefix `moves` starts with.
        "puzzle",
        "puzzle_ply",
    ),
    # `0037_proven_positions.sql`, Q605 (R9b). A person's own solved marks, one a position
    # a way; read by that person's page alone.
    "proven_marks": ("person", "game", "position", "way", "match", "marked"),
    "votes": ("voter", "game", "stars", "tags", "note", "updated"),
    "jobs": (
        "id",
        "kind",
        "hash",
        "requester",
        "state",
        "worker",
        "fingerprint",
        "created",
        "claimed",
        "finished",
        "error",
        "log",
        # `0030_spot_check.sql`, Q503e: the result a spot check replays, its draw, and the
        # processor time every check spent.
        "result",
        "sample",
        "cpu_seconds",
        # `0033_pinned_and_alarm.sql`, Q768 (H9): who asked for a `train` job, and the
        # counts a people's entry was raised on.
        "cause",
        "evidence",
    ),
    # `0006_usage.sql`, `docs/plan-usage.md` §2. Counts by the hour, and no column that
    # could name a person: `TestUsageIsCounted` reads these tuples for exactly that.
    "usage_hours": (
        "hour",
        "route",
        "variant",
        "status",
        "kind",
        "who",
        "count",
        "cpu",
        "wall",
        "wall_max",
    ),
    "usage_people": (
        "hour",
        "route",
        "variant",
        "people",
        "refused",
        "peak",
        "one",
        "two_to_five",
        "six_to_ten",
        "eleven_to_twenty",
        "twenty_one_to_fifty",
        "over_fifty",
        "partial",
    ),
    # `0008_awards.sql`, box E1. One row per badge earned, written once and never
    # updated. Three of the four key columns are nullable and each null means something,
    # so the unique constraint is `nulls not distinct`; `TestTheAwardsTable` in
    # `test_awards.py` is what holds that down.
    "awards": ("kind", "tier", "holder", "game", "earned", "evidence", "rule"),
    # `0011_suggestions.sql`, Q279's. The suggestion box: a short text, one of two kinds
    # and who wrote it. `author` does not cascade, for `awards.holder`'s reason -- leaving
    # empties a person out of their rows rather than removing them, so a suggestion by
    # somebody who has gone reads as *a former player*.
    "suggestions": ("id", "author", "kind", "body", "created", "done"),
    # `0012_links.sql`, G7 of `docs/archive/plan-game-graph.md`. One row per unordered pair
    # of games, `a < b`, with the count of rules that differ and -- within three -- the
    # reading. `relate` is the version the row was taken under, so a reading that moves
    # is recomputed rather than trusted, and both ends cascade so no row can name a
    # `games` row that is not there.
    "links": (
        "a",
        "b",
        "units",
        "kind",
        "kinds",
        "unit",
        "sentence",
        "differences",
        "relate",
        "computed",
        # `0013_similarity.sql`, T4 of the same archive copy's §9.5. The similarity index
        # and its six counts, so the store orders a game's links as `web/library.py` does. Both
        # nullable: a row from before the migration has neither and is compared again.
        "similarity",
        "counts",
    ),
    "usage_samples": (
        "at",
        "cpu_busy",
        "app_cpu",
        "load1",
        "mem_available",
        "app_memory",
        "disk_free",
        "in_flight_max",
        "strong_seats_max",
    ),
    # `0015_provisional.sql`, Q312. A score a visitor's browser measured and the server
    # checked without measuring it: its own table, so no reader that ranks or awards can
    # reach it. `tests/test_site_provisional.py` holds the states and who sees what.
    "provisional": (
        "id",
        "game",
        "submitter",
        "basis",
        "headline",
        "viable",
        "state",
        "job",
        "directory",
        "created",
        "updated",
        # `0030_spot_check.sql`, Q503e: the shortlist's 30-day grace starts here.
        "stale_since",
    ),
    # `0017_reports.sql`, Q503d. One report a person a game, which is the primary key;
    # `reporter` does not cascade, for `suggestions.author`'s reason.
    "reports": ("reporter", "game", "reason", "note", "created", "dismissed"),
    # `0034_invite_asks.sql`, Q831. A visitor's request for an invitation: no user, since
    # the person asking has none; deleted `quota.ASK_KEPT` after it is answered.
    "invite_asks": ("id", "email", "note", "created", "answered", "granted"),
    # `0018_finishes.sql`, Q502a. Every finish a person sat at, with its moves, which no
    # take-back or restart rewrites; it names its match and nothing about who played.
    "finishes": (
        "match", "ordinal", "moves", "result", "reason", "takebacks", "finished",
        # `0032_human_data.sql`, Q767: the three copied from the match row at the finish.
        "thinks", "undone", "standing",
    ),
    # `0020_resting.sql`, Q503n. Each move of a game between the Library and the Baseline
    # that `site load` found, with the maintainer's reason; no route writes it.
    "level_moves": ("id", "game", "was", "became", "reason", "moved"),
    # `0021_rounds.sql`, Q503h. The monthly round: its entrants (with the tally written at
    # the close, or carried with a passed game), the members' yes and no, and what became
    # of each entrant.
    "rounds": ("day", "opens", "closes", "closed"),
    "round_entrants": ("round", "game", "nominated", "carried", "yes", "no"),
    "round_votes": ("round", "game", "person", "yes", "voted"),
    "round_decisions": ("round", "game", "decision", "reason", "decided", "decider"),
    # `0022_correspondence.sql`, Q570. `/admin`'s on/off settings, on while a row exists;
    # the first is the pause on new games between people.
    "switches": ("name", "set", "set_by"),
    # `0023_challenges.sql`, Q571. A challenge, named or a link, and what became of it;
    # and a member's *open to a game of this* boxes, one row a ticked game.
    "challenges": (
        "id", "challenger", "challenged", "token", "game", "seat", "rematch_of", "made",
        "answer", "answered", "match",
    ),
    "open_to": ("person", "game", "since"),
    # `0024_move_email.sql`, Q572. One row a move email: what the daily cap counts, and
    # one move never sends two.
    "move_emails": ("id", "match", "recipient", "ply", "sent", "failed"),
}

# The indexes §3's notes ask for by name, beside the two that make "one row per player
# per game" true. Pinned because each one is a query the plan names -- the game page, the
# fold over a game's matches in finish order, a person's match list, the top of a table.
INDEXES = (
    "matches_by_game",
    "matches_by_seat0",
    "matches_by_seat1",
    "ratings_by_game",
    "ratings_one_row_per_agent",
    "ratings_one_row_per_person",
    "sessions_by_user",
    "jobs_queue",
    # `0004_invite_email.sql`, task 1.3's. Sign-in matches a handle case-insensitively,
    # so `ada` and `Ada` being two accounts would be two accounts one password reaches.
    "users_one_handle",
    # `0008_awards.sql`, box E1's two reads: a person's badges on their profile and a
    # game's badges on its page. Both partial, because a badge with a null holder is on
    # no profile and one with a null game is on no game page.
    "awards_by_holder",
    "awards_by_game",
    # `0011_suggestions.sql`, Q279's two reads by author: this person's own list, and the
    # rolling day the daily allowance counts over.
    "suggestions_by_author",
    # `0012_links.sql`, G7's one read asked from either end: this game's links, nearest
    # first. Both columns carry it, because a pair is stored once in sort order.
    "links_by_a",
    "links_by_b",
    # `0015_provisional.sql`, Q312. The one standing result a game has, which is every
    # reader's question and which the index makes at most one; and a game's history.
    "provisional_one_standing",
    "provisional_by_game",
    # `0017_reports.sql`, Q503d. The rolling day `quota.reports_today` counts over.
    "reports_by_reporter",
    # `0020_resting.sql`, Q503n. A game's latest move (its page) and the latest moves
    # (*What's new*).
    "level_moves_by_game",
    "level_moves_by_moved",
)


@contextlib.contextmanager
def _scratch_schema() -> Iterator[tuple[Any, str]]:
    """A connection pointed at an empty schema of its own, dropped afterwards.

    Every migration names its objects unqualified, so a schema plus a `search_path` is a
    whole empty database as far as the runner can tell. That is what lets these tests run
    against a real server -- the box's, over a tunnel, or a scratch cluster on a
    development machine -- and leave nothing behind on it.

    For the runner's own tests, which need a schema nothing has migrated. A test that only
    needs a migrated store takes `_migrated`, the session's shared one.
    """
    url = site_store.database("no database to run migrations against")
    name = f"migrate_test_{secrets.token_hex(6)}"
    connection = db.connect(url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"create schema {name}")
            cursor.execute(f"set search_path to {name}")
        connection.commit()
        yield connection, name
    finally:
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(f"drop schema if exists {name} cascade")
        connection.commit()
        connection.close()


@contextlib.contextmanager
def _migrated() -> Iterator[Any]:
    """A connection on the session's migrated store, emptied first (`tests/site_store.py`)."""
    url, schema = site_store.emptied("no database to run migrations against")
    connection = db.connect(url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
        connection.commit()
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _tables(connection: Any, schema: str) -> dict[str, tuple[str, ...]]:
    """What the database says it has, read back rather than assumed."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select table_name, column_name from information_schema.columns"
            " where table_schema = %s order by table_name, ordinal_position",
            (schema,),
        )
        found: dict[str, list[str]] = {}
        for table, column in cursor.fetchall():
            found.setdefault(table, []).append(column)
    return {table: tuple(columns) for table, columns in found.items()}


def _ledger(connection: Any) -> list[tuple[Any, ...]]:
    with connection.cursor() as cursor:
        cursor.execute(
            f"select name, checksum, applied from {migrate.LEDGER} order by name"
        )
        return list(cursor.fetchall())


class TestTheMigrationFiles:
    """What the runner can work out without a database in reach."""

    def test_the_store_is_the_first_migration(self) -> None:
        files = migrate.available()
        assert files[0].name == "0001_store.sql"
        assert files[0].number == 1
        assert [file.number for file in files] == list(range(1, len(files) + 1))

    def test_a_checksum_ignores_line_endings(self, tmp_path: Path) -> None:
        """CRLF and LF checkouts have to agree on what a migration hashes to.

        This repository is checked out with CRLF on Windows and LF elsewhere. A checksum
        that saw the difference would report drift on every Windows machine, and a
        refusal that cries wolf is one people learn to pass `--force` at.
        """
        body = "create table a (b text);\n"
        (tmp_path / "0001_a.sql").write_bytes(body.encode())
        lf = migrate.available(tmp_path)[0].checksum
        (tmp_path / "0001_a.sql").write_bytes(body.replace("\n", "\r\n").encode())
        assert migrate.available(tmp_path)[0].checksum == lf

    def test_an_unnumbered_file_stops_the_run(self, tmp_path: Path) -> None:
        (tmp_path / "store.sql").write_text("select 1")
        with pytest.raises(migrate.MigrationError) as error:
            migrate.available(tmp_path)
        assert "0001_store.sql" in str(error.value)

    def test_two_files_sharing_a_number_stop_the_run(self, tmp_path: Path) -> None:
        """The shape two worktrees produce, and the one the ledger cannot untangle."""
        (tmp_path / "0002_left.sql").write_text("select 1")
        (tmp_path / "0002_right.sql").write_text("select 1")
        with pytest.raises(migrate.MigrationError) as error:
            migrate.available(tmp_path)
        assert "share the number 0002" in str(error.value)

    def test_a_file_slipped_in_below_the_highest_applied_is_refused(self) -> None:
        """Two worktrees again, this time noticed after one of them has deployed."""
        files = (
            migrate.Migration(1, "0001_store.sql", "", "aaa"),
            migrate.Migration(2, "0002_late.sql", "", "bbb"),
        )
        reason = migrate.drift(files, {"0002_late.sql": "bbb"})
        assert reason is not None and "renumber it above 0002" in reason

    def test_a_deleted_migration_is_refused(self) -> None:
        reason = migrate.drift((), {"0001_store.sql": "aaa"})
        assert reason is not None and "never removed once applied" in reason


def _every_migration() -> tuple[str, ...]:
    """Every file the runner would apply, in order.

    Read rather than written out, because `STORE` above is already the pin on *what the
    schema is*: a second list of filenames would only say how many files it took to get
    there, and would have to be edited by every migration that changes nothing here.
    """
    return tuple(file.name for file in migrate.available())


class TestTheRunner:
    def test_it_creates_every_table_on_an_empty_database(self) -> None:
        with _scratch_schema() as (connection, schema):
            report = migrate.run(connection)
            assert report.applied == _every_migration()
            found = _tables(connection, schema)
            assert found == {**STORE, migrate.LEDGER: ("name", "checksum", "applied")}

    def test_it_creates_the_indexes_the_plan_names(self) -> None:
        with _scratch_schema() as (connection, schema):
            migrate.run(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    "select indexname from pg_indexes where schemaname = %s", (schema,)
                )
                names = {row[0] for row in cursor.fetchall()}
        assert set(INDEXES) <= names

    def test_a_second_run_changes_nothing_and_says_so(self) -> None:
        """Idempotent, and loud about it, because the deploy script runs it every time."""
        with _scratch_schema() as (connection, schema):
            migrate.run(connection)
            before = _ledger(connection)
            tables_before = _tables(connection, schema)

            report = migrate.run(connection)

            assert report.applied == ()
            assert report.already == _every_migration()
            assert "no rows changed" in report.summary()
            # The ledger's `applied` timestamp would move if the file had been re-run.
            assert _ledger(connection) == before
            assert _tables(connection, schema) == tables_before

    def test_an_edited_migration_is_refused_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """The one the checksum exists for. The repair is a new file, never an edit."""
        original = migrate.MIGRATIONS / "0001_store.sql"
        copy = tmp_path / "0001_store.sql"
        copy.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")

        with _scratch_schema() as (connection, schema):
            migrate.run(connection, tmp_path)
            before = _ledger(connection)
            tables_before = _tables(connection, schema)

            copy.write_text(
                copy.read_text(encoding="utf-8") + "\n-- a later thought\n",
                encoding="utf-8",
            )
            with pytest.raises(migrate.MigrationError) as error:
                migrate.run(connection, tmp_path)

            message = str(error.value)
            assert "0001_store.sql has changed since it was applied" in message
            assert "never edited once applied" in message
            assert "new numbered file" in message
            assert _ledger(connection) == before
            assert _tables(connection, schema) == tables_before

    def test_a_dry_run_writes_nothing_at_all(self) -> None:
        """Asked on a database somebody is deciding whether to touch, ledger included."""
        with _scratch_schema() as (connection, schema):
            report = migrate.run(connection, dry_run=True)
            pending = _every_migration()
            assert report.pending == pending
            assert f"would apply {len(pending)} migration" in report.summary()
            assert _tables(connection, schema) == {}

    def test_a_failing_migration_leaves_no_half_a_schema(self, tmp_path: Path) -> None:
        """Postgres runs DDL in a transaction, and the runner leans on that."""
        (tmp_path / "0001_broken.sql").write_text(
            "create table a (b text);\ncreate table a (b text);\n"
        )
        with _scratch_schema() as (connection, schema):
            with pytest.raises(Exception, match="already exists"):
                migrate.run(connection, tmp_path)
            assert "a" not in _tables(connection, schema)


class TestTheMigrateCommand:
    def test_it_refuses_without_a_database_url(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`chess-variant-sandbox site migrate` reaches `db.database_url` and no further."""
        monkeypatch.delenv(db.ENV_VAR, raising=False)
        assert cli.main(["site", "migrate"]) == 2
        assert db.ENV_VAR in capsys.readouterr().err

    def test_the_dry_run_flag_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deploy script's way of asking before it decides (§7).

        A flag the parser did not know would leave `SystemExit` rather than a return, so
        reaching the handler's own refusal is the thing being checked.
        """
        monkeypatch.delenv(db.ENV_VAR, raising=False)
        assert cli.main(["site", "migrate", "--dry-run"]) == 2


# --------------------------------------------------- 1.3, invites, users and sign-in

# The whole path the row is accepted on -- invite, emailed link, handle and password,
# back by password -- is one test below, driven over HTTP so the cookie is the real
# thing. Everything else here is one way that path can be refused, plus the two
# disciplines §6 asks for: the revoke, and a handle that never reaches a page raw.
#
# No message leaves this machine. `Sent` is the fake every test passes where the site
# takes a sender, and `mail.send` itself is only ever asked what it does without a key.


class Sent:
    """A sender that keeps the message instead of posting it."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    def __call__(self, message: Any) -> None:
        self.messages.append(message)

    @property
    def last(self) -> Any:
        assert self.messages, "nothing was sent"
        return self.messages[-1]


class Answer:
    """One HTTP answer, as these tests read it."""

    def __init__(self, status: int, body: bytes, headers: Any) -> None:
        self.status = status
        self.body = body
        self.headers = headers

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    @property
    def payload(self) -> Any:
        return json.loads(self.body or b"{}")

    @property
    def set_cookie(self) -> str:
        return self.headers.get("Set-Cookie", "")


class Browser:
    """A client that keeps its cookie, which is what makes a session a session.

    `forwarded` is what Caddy puts in front of the app: a test that wants to see the
    `Secure` attribute asks for `https`, and one that wants a development machine leaves
    it out.

    `client` is the whole `X-Forwarded-For` header, so a test can write what Caddy would
    -- one address, or a forged list with the real one appended (Q369).
    """

    def __init__(self, port: int, forwarded: str = "", client: str = "") -> None:
        self.port = port
        self.forwarded = forwarded
        self.client = client
        self.cookie: str | None = None

    def get(self, path: str) -> Answer:
        return self._ask("GET", path, None)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Answer:
        return self._ask("POST", path, payload or {})

    def _ask(self, method: str, path: str, payload: dict[str, Any] | None) -> Answer:
        headers: dict[str, str] = {}
        if self.cookie is not None:
            headers["Cookie"] = f"{site_pages.COOKIE}={self.cookie}"
        if self.forwarded:
            headers["X-Forwarded-Proto"] = self.forwarded
        if self.client:
            headers["X-Forwarded-For"] = self.client
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers[REQUEST_HEADER] = REQUEST_VALUE
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, path, body=body, headers=headers)
            found = connection.getresponse()
            answer = Answer(found.status, found.read(), found.headers)
        finally:
            connection.close()
        self._keep(answer.set_cookie)
        return answer

    def _keep(self, header: str) -> None:
        if not header:
            return
        jar = SimpleCookie()
        jar.load(header)
        morsel = jar.get(site_pages.COOKIE)
        if morsel is None:
            return
        self.cookie = None if str(morsel["max-age"]) == "0" else morsel.value


class Site:
    """A running site with an empty store under it."""

    def __init__(self, app: SiteApp, store: Any, port: int) -> None:
        self.app = app
        self.store = store
        self.port = port

    def browser(self, forwarded: str = "", client: str = "") -> Browser:
        return Browser(self.port, forwarded, client)

    def issuer(self, handle: str = "maintainer") -> int:
        with self.store.cursor() as cursor:
            cursor.execute(
                "insert into users (handle, email, role) values (%s, %s, 'admin')"
                " returning id",
                (handle, f"{handle}@example.invalid"),
            )
            found = cursor.fetchone()
        self.store.commit()
        return int(found[0])

    def invite(self, email: str, sender: Sent, issuer: int | None = None) -> Any:
        """What `chess-variant-sandbox site invite` does, with the sending function passed in."""
        found, _ = identity.send_invitation(
            self.store, email, "https://chess-variant-sandbox.example", sender, issuer=issuer
        )
        self.store.commit()
        return found

    def sessions(self) -> int:
        with self.store.cursor() as cursor:
            cursor.execute("select count(*) from sessions")
            return int(cursor.fetchone()[0])


@contextlib.contextmanager
def _running_site(root: Path | None = None) -> Iterator[Site]:
    """The site on a port, over the session's store, emptied first.

    Over HTTP rather than through `SiteApp` directly, because the cookie is what is under
    test and a cookie only exists between a header and a header.

    `root` is a checkout to serve instead of this one. The map's tests want a library of
    four specs rather than the committed seventy, because what they ask about is the
    shape of an answer and not what any game happens to be -- and comparing seventy-one
    specs takes twenty seconds.
    """
    url, name = site_store.emptied("no database to sign anybody in against")

    def connect() -> Any:
        connection = db.connect(url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        return connection

    store = connect()
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    store.commit()
    app = (
        SiteApp(RULESETS, library.RUNS_DIR, connect=connect)
        if root is None
        else SiteApp(
            root / "rulesets", root / "runs", archive_dir=root / "archive", connect=connect
        )
    )
    server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    try:
        yield Site(app, store, server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        store.close()


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


PASSWORD = "a-long-enough-one"


def _user_id(where: Site, handle: str) -> int:
    with where.store.cursor() as cursor:
        cursor.execute("select id from users where handle = %s", (handle,))
        return int(cursor.fetchone()[0])


def _joined(where: Site, handle: str) -> Browser:
    """One person through the whole join, for a test that starts after it."""
    invited = where.invite(f"{handle}@example.com", Sent())
    browser = where.browser()
    made = browser.post(
        "/api/join", {"code": invited.code, "handle": handle, "password": PASSWORD}
    )
    assert made.status == 200, made.text
    return browser


class TestThePathAPersonTakes:
    def test_invite_link_handle_and_password_then_back_by_password(
        self, site: Site
    ) -> None:
        """The row's own acceptance, end to end and over HTTP.

        Five steps: a maintainer invites an address, the mail carries a link, the link is
        a form, the form makes the account and signs it in, and the password brings the
        person back in a browser that has never seen them.
        """
        sender = Sent()
        invited = site.invite("Ada@example.com", sender, issuer=site.issuer())

        # What the message carries is the link, and nothing the person has to work out.
        assert sender.last.to == "ada@example.com"
        link = identity.link("https://chess-variant-sandbox.example", invited.code)
        assert link in sender.last.body

        # The link is a page, and the code is in it rather than in the script.
        browser = site.browser()
        joining = browser.get(f"/join?code={invited.code}")
        assert joining.status == 200
        assert invited.code in joining.text

        made = browser.post(
            "/api/join", {"code": invited.code, "handle": "ada", "password": PASSWORD}
        )
        assert made.status == 200
        assert made.payload == {"handle": "ada"}
        assert browser.cookie, "joining did not set a session cookie"

        # Signed in: the profile answers, and it is this person's.
        profile = browser.get("/api/profile")
        assert profile.status == 200
        assert profile.payload["handle"] == "ada"

        # The account is the invitation's address, and the invitation is spent.
        with site.store.cursor() as cursor:
            cursor.execute(
                "select email, invited_by, password from users where handle = 'ada'"
            )
            email, invited_by, password = cursor.fetchone()
            cursor.execute(
                "select uses_left from invites where code = %s", (invited.code,)
            )
            left = cursor.fetchone()[0]
        assert email == "ada@example.com"
        assert invited_by is not None
        assert password.startswith("scrypt$") and PASSWORD not in password
        assert left == 0

        # And back, in a browser with no cookie: the password is the whole of it.
        returning = site.browser()
        assert returning.get("/api/profile").status == 400
        back = returning.post("/api/signin", {"handle": "ada", "password": PASSWORD})
        assert back.status == 200
        assert returning.get("/api/profile").payload["handle"] == "ada"

    def test_signing_out_takes_the_cookie_and_the_row_with_it(self, site: Site) -> None:
        browser = _joined(site, "bob")
        assert site.sessions() == 1

        out = browser.post("/api/signout")
        assert out.status == 200
        assert browser.cookie is None
        assert site.sessions() == 0

    def test_the_handle_is_matched_whatever_its_capitals(self, site: Site) -> None:
        """One account, one handle. The unique index on `lower(handle)` is the other
        half of that; this is the half a person meets."""
        _joined(site, "Cass")
        browser = site.browser()
        back = browser.post("/api/signin", {"handle": "cass", "password": PASSWORD})
        assert back.status == 200


class TestTheAppNeverAsksForTheSharedPassword:
    """Q641. The browser's own password box (the trial's shared door) appears only for a
    401 carrying `WWW-Authenticate`. Caddy's `basic_auth` writes that for any request that
    reaches it without the shared credentials, on every path alike; the app must never
    write it itself, or a refused sign-in or a signed-out read would open the box for
    someone already through the door. So every request a page makes -- the page, each
    `src`, `href` and `url(...)` it names, and the routes its scripts read -- is asked as a
    visitor and as a member, and none answers with the header."""

    PAGES = ("/", "/profile", "/players", "/suggestions", "/signin", "/join", "/admin")
    READS = ("/api/me", "/api/profile", "/api/players", "/api/suggestions", "/api/admin",
             "/api/challenge/link/nobody", "/static/fonts/none.woff2", "/favicon.png")

    def test_no_route_a_page_asks_answers_with_the_password_box(self, site: Site) -> None:
        asked: dict[str, int] = {}
        for browser in (site.browser(), _joined(site, "erin")):
            routes = set(self.READS)
            for path in self.PAGES:
                page = browser.get(path)
                asked[path] = page.status
                assert "WWW-Authenticate" not in page.headers, path
                routes.update(re.findall(r'(?:src|href)="(/[^"#?]*)', page.text))
                routes.update(re.findall(r"url\((/[^)]*)\)", page.text))
            for path in sorted(routes):
                answer = browser.get(path)
                asked[path] = answer.status
                assert "WWW-Authenticate" not in answer.headers, path
            refused = browser.post("/api/signin", {"handle": "erin", "password": "wrong"})
            assert refused.status == 401
            assert "WWW-Authenticate" not in refused.headers
        # The fonts, the icons and the scripts were among them, and were found.
        assert any(path.startswith("/static/fonts/") and status == 200
                   for path, status in asked.items())
        for path in ("/favicon.svg", "/favicon.ico", "/apple-touch-icon.png"):
            assert asked[path] == 200, path


class TestEveryWayItIsRefused:
    def test_a_wrong_password_is_a_401_with_no_cookie(self, site: Site) -> None:
        _joined(site, "dana")
        browser = site.browser()
        refused = browser.post(
            "/api/signin", {"handle": "dana", "password": "not-the-one"}
        )
        assert refused.status == 401
        assert refused.payload["error"] == "that handle and password do not match"
        assert browser.cookie is None
        assert site.sessions() == 1  # the join's, untouched

    def test_an_unknown_handle_is_refused_in_the_same_words(self, site: Site) -> None:
        """Which of the two was wrong is information for whoever is guessing."""
        refused = site.browser().post(
            "/api/signin", {"handle": "nobody", "password": PASSWORD}
        )
        assert refused.status == 401
        assert refused.payload["error"] == "that handle and password do not match"

    def test_a_cookie_nobody_issued_is_a_visitor(self, site: Site) -> None:
        browser = site.browser()
        browser.cookie = secrets.token_urlsafe(32)
        assert browser.get("/api/profile").status == 400
        assert browser.get("/profile").status == 303

    def test_an_expired_session_is_a_visitor(self, site: Site) -> None:
        """The row is there and the cookie is right; the date is what refuses."""
        browser = _joined(site, "erin")
        with site.store.cursor() as cursor:
            cursor.execute("update sessions set expires = now() - interval '1 day'")
        site.store.commit()
        assert browser.get("/api/profile").status == 400
        assert site.sessions() == 1

    def test_a_spent_invitation_cannot_be_spent_again(self, site: Site) -> None:
        invited = site.invite("fay@example.com", Sent())
        site.browser().post(
            "/api/join", {"code": invited.code, "handle": "fay", "password": PASSWORD}
        )
        again = site.browser().post(
            "/api/join", {"code": invited.code, "handle": "fay2", "password": PASSWORD}
        )
        assert again.status == 401
        assert "expired or has already been used" in again.payload["error"]

    def test_an_expired_invitation_is_refused(self, site: Site) -> None:
        invited = identity.invite(
            site.store, "gus@example.com", lifetime=timedelta(days=-1)
        )
        site.store.commit()
        refused = site.browser().post(
            "/api/join", {"code": invited.code, "handle": "gus", "password": PASSWORD}
        )
        assert refused.status == 401

    def test_a_handle_somebody_has_leaves_the_invitation_unspent(
        self, site: Site
    ) -> None:
        """A form that refuses must not cost the person their invitation."""
        _joined(site, "hana")
        invited = site.invite("iris@example.com", Sent())
        taken = site.browser().post(
            "/api/join", {"code": invited.code, "handle": "HANA", "password": PASSWORD}
        )
        assert taken.status == 401
        assert "taken" in taken.payload["error"]

        with site.store.cursor() as cursor:
            cursor.execute(
                "select uses_left from invites where code = %s", (invited.code,)
            )
            assert cursor.fetchone()[0] == 1
        kept = site.browser().post(
            "/api/join", {"code": invited.code, "handle": "iris", "password": PASSWORD}
        )
        assert kept.status == 200

    @pytest.mark.parametrize(
        "handle", ["ab", "a" * 25, "-nope", "with space", "<script>", "ünïcode"]
    )
    def test_a_handle_outside_the_set_never_reaches_the_store(
        self, site: Site, handle: str
    ) -> None:
        """The first of §6's two defences for a name printed on somebody else's page.
        The second is the escaping, below."""
        invited = site.invite(f"{secrets.token_hex(4)}@example.com", Sent())
        refused = site.browser().post(
            "/api/join", {"code": invited.code, "handle": handle, "password": PASSWORD}
        )
        assert refused.status == 401
        assert "3 to 24 characters" in refused.payload["error"]

    def test_a_short_password_is_refused_by_length(self, site: Site) -> None:
        invited = site.invite("jody@example.com", Sent())
        refused = site.browser().post(
            "/api/join", {"code": invited.code, "handle": "jody", "password": "short"}
        )
        assert refused.status == 401
        assert "at least 10 characters" in refused.payload["error"]

    def test_an_account_with_no_password_cannot_sign_in(self, site: Site) -> None:
        """The loader's maintainer row, which owns the corpus and is not a login."""
        site.issuer("maintainer")
        refused = site.browser().post(
            "/api/signin", {"handle": "maintainer", "password": ""}
        )
        assert refused.status == 401


class TestWhichAddressAnAttemptIsCountedAgainst:
    """Q369. The app is behind Caddy, so the client is a header, and a header a client
    can write is a header a client can forge.

    Caddy **appends** the connecting address to whatever `X-Forwarded-For` arrived, so
    the last entry is the one it wrote and everything before it is the client's. Reading
    the first would hand a guessing run a fresh allowance per attempt.
    """

    def test_the_last_entry_is_the_one_the_front_server_wrote(self) -> None:
        assert quota.source_address("127.0.0.1", "198.51.100.5") == "198.51.100.5"

    def test_a_forged_list_does_not_change_the_answer(self) -> None:
        forged = "203.0.113.9, 10.0.0.1, 198.51.100.5"
        assert quota.source_address("127.0.0.1", forged) == "198.51.100.5"

    def test_a_peer_that_is_not_the_proxy_is_read_instead_of_its_header(self) -> None:
        """Nothing forwards to the app from off the box, so a header from off the box is
        written by whoever is asking."""
        assert quota.source_address("198.51.100.5", "127.0.0.1") == "198.51.100.5"

    def test_no_header_is_the_peer(self) -> None:
        assert quota.source_address("127.0.0.1", None) == "127.0.0.1"
        assert quota.source_address("127.0.0.1", "  ,  ") == "127.0.0.1"


class TestTheSignInLimitItself:
    """Q369's counter, driven on a clock of its own so a window can pass in a test."""

    def _gate(self) -> tuple[quota.SignIns, list[float]]:
        now = [0.0]
        return quota.SignIns(clock=lambda: now[0]), now

    def test_the_limit_trips_at_the_number_and_not_before(self) -> None:
        gate, _ = self._gate()
        for _ in range(quota.SIGNIN_TRIES_A_SOURCE):
            gate.check("198.51.100.5")
            gate.failed("198.51.100.5")
        with pytest.raises(quota.Spent) as refused:
            gate.check("198.51.100.5")
        assert "Too many failed sign-in attempts" in str(refused.value)

    def test_it_resets_when_the_window_has_passed(self) -> None:
        gate, now = self._gate()
        for _ in range(quota.SIGNIN_TRIES_A_SOURCE):
            gate.failed("198.51.100.5")
        with pytest.raises(quota.Spent):
            gate.check("198.51.100.5")
        now[0] += quota.SIGNIN_WINDOW.total_seconds() + 1
        gate.check("198.51.100.5")

    def test_one_address_cannot_spend_anothers(self) -> None:
        gate, _ = self._gate()
        for _ in range(quota.SIGNIN_TRIES_A_SOURCE * 2):
            gate.failed("198.51.100.5", "ada")
        with pytest.raises(quota.Spent):
            gate.check("198.51.100.5", "ada")
        gate.check("203.0.113.9", "bea")

    def test_a_success_clears_what_was_failed(self) -> None:
        gate, _ = self._gate()
        for _ in range(quota.SIGNIN_TRIES_A_SOURCE - 1):
            gate.failed("198.51.100.5", "ada")
        gate.passed("198.51.100.5", "ada")
        for _ in range(quota.SIGNIN_TRIES_A_SOURCE - 1):
            gate.check("198.51.100.5", "ada")
            gate.failed("198.51.100.5", "ada")

    def test_the_handle_is_counted_across_addresses(self) -> None:
        """The looser of the two limits, and the only one a spread of addresses meets.

        It is also the one a stranger could shut an owner out with, which is why it sits
        at three times the per-address number rather than at it.
        """
        gate, _ = self._gate()
        for source in range(quota.SIGNIN_TRIES_AN_ACCOUNT // quota.SIGNIN_TRIES_A_SOURCE):
            where = f"198.51.100.{source}"
            for _ in range(quota.SIGNIN_TRIES_A_SOURCE):
                gate.failed(where, "ada")
        with pytest.raises(quota.Spent):
            gate.check("203.0.113.9", "ada")
        gate.check("203.0.113.9", "bea")

    def test_a_handle_nobody_has_trips_at_the_same_attempt_in_the_same_words(
        self,
    ) -> None:
        """Only failures are counted, and a made-up handle fails the same way a real one
        does, so the limit says nothing about which accounts exist."""
        gate, _ = self._gate()
        said = []
        for source, handle in (("198.51.100.5", "ada"), ("203.0.113.9", "nobody-at-all")):
            for _ in range(quota.SIGNIN_TRIES_A_SOURCE):
                gate.failed(source, handle)
            with pytest.raises(quota.Spent) as refused:
                gate.check(source, handle)
            said.append(str(refused.value))
        assert said[0] == said[1]

    def test_the_table_stays_under_its_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A table keyed by what a stranger sends is a table a stranger can grow."""
        monkeypatch.setattr(quota, "SIGNIN_TRACKED", 8)
        gate, _ = self._gate()
        for source in range(40):
            gate.failed(f"198.51.100.{source}")
        assert len(gate._failures) <= 8


class TestSigningInIsRationed:
    """Q369, over HTTP: the limit the sign-in route has, since the front door has none.

    Caddy 2.11 ships no request limiter and the firewall counts connections, so before
    this a source could spend 240 password guesses a minute on `/api/signin`.
    """

    def _spend(self, site: Site, client: str, handle: str = "nobody") -> None:
        """Every attempt one address is allowed, all of them refused."""
        for attempt in range(quota.SIGNIN_TRIES_A_SOURCE):
            answer = site.browser(client=client).post(
                "/api/signin", {"handle": handle, "password": f"wrong-{attempt}"}
            )
            assert answer.status == 401, answer.text

    def test_the_attempt_after_the_limit_is_a_429_that_says_when(self, site: Site) -> None:
        self._spend(site, "198.51.100.5")
        refused = site.browser(client="198.51.100.5").post(
            "/api/signin", {"handle": "nobody", "password": "wrong-again"}
        )
        assert refused.status == 429
        assert "Too many failed sign-in attempts" in refused.payload["error"]
        assert "minute" in refused.payload["error"]

    def test_another_address_is_untouched(self, site: Site) -> None:
        _joined(site, "mara")
        self._spend(site, "198.51.100.5", "mara")
        signed_in = site.browser(client="203.0.113.9").post(
            "/api/signin", {"handle": "mara", "password": PASSWORD}
        )
        assert signed_in.status == 200, signed_in.text

    def test_a_forged_forwarding_header_does_not_dodge_the_limit(
        self, site: Site
    ) -> None:
        """Caddy appends the real address, so a client writing its own entries only ever
        writes what comes before it."""
        for attempt in range(quota.SIGNIN_TRIES_A_SOURCE):
            forged = f"203.0.113.{attempt}, 198.51.100.5"
            answer = site.browser(client=forged).post(
                "/api/signin", {"handle": "nobody", "password": f"wrong-{attempt}"}
            )
            assert answer.status == 401, answer.text
        refused = site.browser(client="203.0.113.99, 198.51.100.5").post(
            "/api/signin", {"handle": "nobody", "password": "wrong-again"}
        )
        assert refused.status == 429

    def test_an_ordinary_sign_in_is_never_refused_and_clears_the_count(
        self, site: Site
    ) -> None:
        """Somebody who mistyped their way to one attempt short still gets in, and their
        next bad evening starts from zero."""
        _joined(site, "nell")
        for attempt in range(quota.SIGNIN_TRIES_A_SOURCE - 1):
            site.browser(client="198.51.100.5").post(
                "/api/signin", {"handle": "nell", "password": f"wrong-{attempt}"}
            )
        browser = site.browser(client="198.51.100.5")
        assert browser.post(
            "/api/signin", {"handle": "nell", "password": PASSWORD}
        ).status == 200
        assert browser.cookie is not None
        again = site.browser(client="198.51.100.5").post(
            "/api/signin", {"handle": "nell", "password": "wrong-once-more"}
        )
        assert again.status == 401, "the count was cleared, so this is not the limit"

    def test_guessing_an_invitation_code_meets_the_same_limit(self, site: Site) -> None:
        """A code is a credential too, and the only one that hands out an account."""
        for attempt in range(quota.SIGNIN_TRIES_A_SOURCE):
            answer = site.browser(client="198.51.100.5").post(
                "/api/join",
                {"code": f"guess-{attempt}", "handle": "opal", "password": PASSWORD},
            )
            assert answer.status == 401, answer.text
        refused = site.browser(client="198.51.100.5").post(
            "/api/join", {"code": "guess-again", "handle": "opal", "password": PASSWORD}
        )
        assert refused.status == 429

    def test_the_maintenance_handle_gets_the_ordinary_taken_answer(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q365b left the handle free until the account's first sign-in, so a stranger
        could join under it, and their next sign-in was a 400 naming `ADMIN_ACCOUNT_NAME`
        -- which told them both that there is such an account and what its key is called."""
        monkeypatch.setenv(admin.NAME_VAR, "keeper-of-keys")
        invited = site.invite("pia@example.com", Sent())

        taken = site.browser().post(
            "/api/join",
            {"code": invited.code, "handle": "Keeper-Of-Keys", "password": PASSWORD},
        )

        assert taken.status == 401
        assert taken.payload["error"] == "the handle Keeper-Of-Keys is taken; choose another"
        assert admin.NAME_VAR not in taken.payload["error"]
        with site.store.cursor() as cursor:
            cursor.execute("select uses_left from invites where code = %s", (invited.code,))
            assert cursor.fetchone()[0] == 1, "a refused form must not cost the invitation"

    def test_no_form_tells_the_maintenance_handle_from_a_held_one(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same answer as somebody's handle, whatever else the form gets wrong.

        The first cut of the fix refused the maintenance handle ahead of the code and the
        password, so a stranger with no invitation at all could send it with a junk code
        and read *taken* where anybody else's handle reads *expired*. Every form below
        must answer the two handles alike, word for word but the handle itself.
        """
        monkeypatch.setenv(admin.NAME_VAR, "keeper-of-keys")
        _joined(site, "mara")
        invited = site.invite("pia@example.com", Sent())
        forms = {
            "a good code": {"code": invited.code, "password": PASSWORD},
            "a made-up code": {"code": "not-a-code", "password": PASSWORD},
            "a short password": {"code": invited.code, "password": "short"},
        }
        for form, fields in forms.items():
            answers = []
            for handle in ("mara", "keeper-of-keys"):
                answer = site.browser(client="198.51.100.7").post(
                    "/api/join", {**fields, "handle": handle}
                )
                answers.append((answer.status, answer.payload["error"].replace(handle, "X")))
            assert answers[0] == answers[1], form
            assert admin.NAME_VAR not in answers[1][1], form

    def test_the_maintenance_handle_is_taken_once_its_row_is_written(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the account's first sign-in the row answers for itself, the same way."""
        monkeypatch.setenv(admin.NAME_VAR, "keeper-of-keys")
        monkeypatch.setenv(admin.PASSWORD_VAR, "a-maintenance-password")
        signed_in = site.browser().post(
            "/api/signin", {"handle": "keeper-of-keys", "password": "a-maintenance-password"}
        )
        assert signed_in.status == 200, signed_in.text
        invited = site.invite("pia@example.com", Sent())

        taken = site.browser().post(
            "/api/join",
            {"code": invited.code, "handle": "keeper-of-keys", "password": PASSWORD},
        )

        assert taken.status == 401
        assert taken.payload["error"] == "the handle keeper-of-keys is taken; choose another"


MAINTENANCE = ("keeper-of-keys", "a-maintenance-password")
"""The maintenance account the next class configures: its handle, then its password."""


class TestEveryAttemptCostsOneHash:
    """Q379. The limit caps how often a source may ask; this is what each answer says.

    Q369's record left a timing leak behind it. A handle nobody held cost no scrypt, a
    held one cost one, and the maintenance handle cost two -- the reconcile's, then the
    check's -- so how long `/api/signin` took said whether a handle existed and which one
    was the admin's. Wall-clock time is too noisy to hold in a test, so these count the
    work instead: every scrypt goes through `identity._scrypt`, and each attempt below
    has to cost what every other one does.
    """

    def _counting(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """A tally of every scrypt the site makes from here on, and of requests sent."""
        made = [0, 0]
        real = identity._scrypt

        def counted(*args: Any) -> bytes:
            made[0] += 1
            return real(*args)

        monkeypatch.setattr(identity, "_scrypt", counted)
        return made

    def _cost(
        self, made: list[int], site: Site, path: str, body: dict[str, str]
    ) -> tuple[int, int]:
        """One request, from an address of its own so the limit never answers instead,
        and what it cost in scrypts beside the status it got."""
        made[0] = 0
        made[1] += 1
        answer = site.browser(client=f"198.51.100.{made[1]}").post(path, body)
        return made[0], answer.status

    def _configured(self, site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
        """A maintenance account, a person, and the loader's row with no password."""
        handle, password = MAINTENANCE
        monkeypatch.setenv(admin.NAME_VAR, handle)
        monkeypatch.setenv(admin.PASSWORD_VAR, password)
        admin.account_hash()  # what `serve` does before it binds
        _joined(site, "mara")
        site.issuer("maintainer")

    def test_a_handle_costs_the_same_whether_it_is_nobodys_anybodys_or_the_admins(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Signing in and joining, each across every kind of handle.

        Twice round for signing in, because the maintenance account's row is written by
        the first attempt that meets it and the second finds it there. Joining hashes the
        chosen password before it looks the handle up, so a held handle, the reserved one
        and a free one cost alike there too; a short password is refused before any hash,
        for all three alike.
        """
        self._configured(site, monkeypatch)
        made = self._counting(monkeypatch)
        kinds = {
            "the maintenance account's": MAINTENANCE[0],
            "a person's": "mara",
            "the loader's, with no password": "maintainer",
            "nobody's": "nobody-at-all",
        }
        signing_in = {
            (round_, kind): self._cost(
                made, site, "/api/signin", {"handle": handle, "password": "wrong-password"}
            )
            for round_ in (1, 2)
            for kind, handle in kinds.items()
        }
        assert set(signing_in.values()) == {(1, 401)}, signing_in

        joining: dict[tuple[str, str], int] = {}
        for form in ("a made-up code", "a short password", "a good code"):
            for kind, handle in (
                ("the maintenance account's", MAINTENANCE[0]),
                ("a person's", "mara"),
                ("nobody's", "opal"),
            ):
                code = "not-a-code"
                if form != "a made-up code":
                    code = site.invite(f"{secrets.token_hex(4)}@example.com", Sent()).code
                password = "short" if form == "a short password" else PASSWORD
                cost, _ = self._cost(
                    made, site, "/api/join",
                    {"code": code, "handle": handle, "password": password},
                )
                joining[(form, kind)] = cost
        for form, each in (("a made-up code", 1), ("a short password", 0), ("a good code", 1)):
            costs = {kind: cost for (at, kind), cost in joining.items() if at == form}
            assert set(costs.values()) == {each}, (form, costs)

    def test_a_right_password_costs_what_a_wrong_one_does(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For a person and for the maintenance account: one scrypt, the check itself."""
        self._configured(site, monkeypatch)
        made = self._counting(monkeypatch)
        attempts = {
            (who, which): self._cost(
                made, site, "/api/signin", {"handle": handle, "password": password}
            )
            for who, handle, right in (
                ("a person", "mara", PASSWORD),
                ("the maintenance account", *MAINTENANCE),
            )
            for which, password in (("right", right), ("wrong", "wrong-password"))
        }

        assert {status for _, status in attempts.values()} == {200, 401}, attempts
        assert all(
            status == (200 if which == "right" else 401)
            for (_, which), (_, status) in attempts.items()
        ), attempts
        assert {cost for cost, _ in attempts.values()} == {1}, attempts


class TestRevokingEverybody:
    def test_one_command_and_no_session_works(self, site: Site) -> None:
        """§8's *revoke everybody at once on unexpected traffic*, and its other half:
        every account survives it and everybody can sign in again."""
        first = _joined(site, "kim")
        second = _joined(site, "lee")
        assert first.get("/api/profile").status == 200

        gone = identity.revoke_sessions(site.store)
        site.store.commit()

        assert gone == 2
        assert site.sessions() == 0
        assert first.get("/api/profile").status == 400
        assert second.get("/api/profile").status == 400

        again = site.browser()
        back = again.post("/api/signin", {"handle": "kim", "password": PASSWORD})
        assert back.status == 200

    def test_a_revoked_account_cannot_come_back(self, site: Site) -> None:
        """1.8's per-user switch, honoured by the cookie the day the cookie exists."""
        browser = _joined(site, "moss")
        with site.store.cursor() as cursor:
            cursor.execute("update users set revoked = now() where handle = 'moss'")
        site.store.commit()
        assert browser.get("/api/profile").status == 400
        refused = site.browser().post(
            "/api/signin", {"handle": "moss", "password": PASSWORD}
        )
        assert refused.status == 401


class TestTheCookie:
    def test_it_is_lax_and_kept_from_the_page(self, site: Site) -> None:
        """§6's CSRF answer. `Lax` is what stops a form on somebody else's site posting
        as this person; `HttpOnly` is what stops a script on this one reading it."""
        browser = site.browser()
        invited = site.invite("nell@example.com", Sent())
        made = browser.post(
            "/api/join", {"code": invited.code, "handle": "nell", "password": PASSWORD}
        )
        header = made.set_cookie
        assert "SameSite=Lax" in header
        assert "HttpOnly" in header
        assert "Path=/" in header
        assert "Max-Age=2592000" in header

    def test_secure_is_set_when_caddy_says_the_request_arrived_over_tls(
        self, site: Site
    ) -> None:
        """On the box every request carries that header; on a development machine none
        does, and a `Secure` cookie would never come back."""
        plain = site.browser()
        invited = site.invite("owen@example.com", Sent())
        made = plain.post(
            "/api/join", {"code": invited.code, "handle": "owen", "password": PASSWORD}
        )
        assert "Secure" not in made.set_cookie

        tls = site.browser(forwarded="https")
        back = tls.post("/api/signin", {"handle": "owen", "password": PASSWORD})
        assert "Secure" in back.set_cookie

    def test_the_row_holds_a_digest_rather_than_the_cookie(self, site: Site) -> None:
        """A dump of the store is not a drawer of live cookies."""
        browser = _joined(site, "pat")
        with site.store.cursor() as cursor:
            cursor.execute("select token from sessions")
            stored = cursor.fetchone()[0]
        assert browser.cookie is not None
        assert stored != browser.cookie
        assert stored == identity.digest_of(browser.cookie)

    def test_two_people_at_once_are_two_people(self, site: Site) -> None:
        """`SiteApp.viewer` is per request on a threaded server, so what this pins is one
        person's identity answering another person's request."""
        first = _joined(site, "quin")
        second = _joined(site, "rex")
        assert first.get("/api/profile").payload["handle"] == "quin"
        assert second.get("/api/profile").payload["handle"] == "rex"
        assert first.get("/api/profile").payload["handle"] == "quin"


class TestTheCheapIdentityProbe:
    """`/api/me` -- Q199. The header used to call `/api/profile` on every page load,
    which reads this person's record on every rated game; this answers the three things
    a header needs (signed in or not, the handle, the role) plus `accounts`, which is
    how the header tells this site apart from the local application that has none, and
    costs nothing beyond the cookie `identify` already resolved.
    """

    def test_a_visitor_is_told_nobody_and_that_the_site_has_accounts(
        self, site: Site
    ) -> None:
        answer = site.browser().get("/api/me")
        assert answer.status == 200
        assert answer.payload == {
            "signedIn": False, "handle": None, "mark": None, "admin": False, "accounts": True,
            "players": False, "round": False,
        }

    def test_a_signed_in_person_is_named_and_not_an_admin(self, site: Site) -> None:
        browser = _joined(site, "uma")
        me = browser.get("/api/me").payload
        assert me.pop("mark") == marks.mark(_user_id(site, "uma"))  # R2: their own mark
        assert me == {
            "signedIn": True, "handle": "uma", "admin": False, "accounts": True, "players": False,
            "round": False,
        }

    def test_an_admin_is_told_so(self, site: Site) -> None:
        browser = _joined(site, "vic")
        with site.store.cursor() as cursor:
            cursor.execute("update users set role = 'admin' where handle = 'vic'")
        site.store.commit()
        me = browser.get("/api/me").payload
        assert me.pop("mark") == marks.mark(_user_id(site, "vic"))
        assert me == {
            "signedIn": True, "handle": "vic", "admin": True, "accounts": True, "players": False,
            "round": False,
        }

    def test_it_is_registered_and_the_profile_still_answers_too(self) -> None:
        """`/api/me` is new; `/api/profile` is unmoved -- the profile page still reads
        its own heavier route."""
        assert routes.find("GET", "/api/me") is not None
        assert routes.find("GET", "/api/profile") is not None


def _every_page() -> dict[str, str]:
    """The ten pages, by the path each is served at."""
    return {
        "/join": page_text.inlined(site_pages.join("code-1")),
        "/signin": page_text.inlined(site_pages.signin()),
        "/profile": page_text.inlined(site_pages.profile("ada")),
        "/players": page_text.inlined(site_pages.players("ada")),
        "/admin": page_text.inlined(site_pages.admin("boss")),
        "/credits": site_pages.credits(),
        "/privacy": site_pages.privacy(),
        "/terms": site_pages.terms_page(),
        "/badges": site_pages.badges(),
        "/request-invitation": page_text.inlined(site_pages.request_invitation()),
    }


def _in_node(source: str, runner: str) -> Any:
    """Run a slice of a page's script, and read back what it printed.

    `tests/test_web.py` does the same for the long page. A sort written in JavaScript is
    tested by sorting something with it; asserting that the page string contains a
    comparator would only say the line is there.

    It runs in the session's one node (`tests/node_session.py`), whose channel is JSON,
    so an em dash arrives as one whatever the console's code page is.
    """
    done = node_session.run(source + "\n" + runner)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _script_of(page: str) -> str:
    """The program a site page runs: its files' text in load order (`page_text`), after
    one line that fills the page's JSON data tag where it has one, since the stubbed DOM
    below has no markup to find it in."""
    script = page_text.script_of(page)
    assert script, "this page carries no script"
    data = page_text.data_of(page)
    if data:
        script = (
            f"document.getElementById('page-data').textContent = {json.dumps(data)};\n"
            + script
        )
    return script


class TestOneShell:
    """D1 of `docs/archive/plan-ux.md`. The site's own pages and the application's were two
    looks -- grey ground, no header, Title-case buttons, blue links at the foot -- and a
    person moving between them crossed a seam that says nothing true about the site."""

    def test_every_page_draws_the_applications_header(self) -> None:
        for path, page in _every_page().items():
            assert '<h1 class="brand"><a href="/"><svg class="mark"' in page, path
            assert '<span class="word">Chess variant sandbox</span></a></h1>' in page, path
            assert site_pages.TAGLINE in page, path
            assert '<div class="wrap">' in page, path
            assert '<span class="muted small" id="who">' in page, path

    def test_every_page_is_painted_from_the_applications_stylesheet(self) -> None:
        """Imported, not copied: one palette, and a token added there reaches here."""
        from chess_variant_sandbox.web.page.style import STYLE as app_style

        for path, page in _every_page().items():
            assert app_style in page, path

    def test_the_foot_links_are_the_applications_footer(self) -> None:
        for path, page in _every_page().items():
            assert "<nav>" not in page, path
            assert '<footer class="muted small">' in page, path
            foot = page[page.index('<footer class="muted small">'):]
            assert (path != "/privacy") == ('href="/privacy"' in foot), path
            assert (path != "/credits") == ('href="/credits"' in foot), path
            assert (path != "/terms") == ('href="/terms"' in foot), path

    def test_the_way_to_the_players_waits_for_five_people(self) -> None:
        """L3 of `docs/archive/plan-launch.md` §3.3. Below `milestones.LISTED_FROM` people no
        footer links `/players`; from then on the pages a signed-in person gets do, the
        same for every reader. A constant page reads no store and never links it."""
        for path, page in _every_page().items():
            assert 'href="/players">players' not in page, path
        for page in (site_pages.profile("ada", True), site_pages.suggestions("ada", True),
                     site_pages.admin("boss", True)):
            assert 'href="/players">players' in page

    def test_the_corner_names_whoever_the_page_knows_about(self) -> None:
        """The application fills it from `/api/me`; these pages are rendered by the
        server, which hands the handle to the two pages that have one. Credits and
        privacy read no cookie at all, so their corner is empty on purpose."""
        assert '<span class="muted small" id="who">ada</span>' in page_text.inlined(site_pages.profile("ada"))
        assert f'{site_pages._member_button("boss")}</span>' in page_text.inlined(site_pages.admin("boss"))
        assert f'{site_pages._member_button("ada")}</span>' in page_text.inlined(site_pages.players("ada"))
        for path in ("/signin", "/join", "/credits", "/privacy", "/terms", "/badges",
                     "/request-invitation"):
            assert '<span class="muted small" id="who"></span>' in _every_page()[path], path

    def test_the_way_to_your_page_is_a_button_that_says_where_it_goes(self) -> None:
        """Q731: a bare handle in the corner read as a label saying who is signed in. It
        is an outlined button, named for where it goes and for whom, as tall as any other
        control, its handle dropped on a phone; both servers build the same element."""
        from page_text import PAGE

        button =site_pages._member_button("<ada>")
        assert button.startswith('<a class="memberButton" href="/profile"')
        assert 'aria-label="Your page, signed in as &lt;ada&gt;"' in button
        assert '<span class="memberName">&lt;ada&gt;</span>' in button
        assert '<span class="memberGo">your page<span aria-hidden="true"> ›</span></span>' in button
        style = site_pages.STYLE
        rule = style[style.index("header a.memberButton {"):]
        rule = rule[: rule.index("}")]
        for held in ("min-height: var(--ctl-h)", "border: 1px solid var(--edge)",
                     "border-radius: var(--r-ctl)", "direction: ltr"):
            assert held in rule, held
        assert "header a.memberButton .memberName { display: none; }" in style
        assert "memberButton(VIEWER, VIEWER_MARK)" in PAGE
        assert "anchor.className = 'memberButton';" in PAGE
        assert "`Your page, signed in as ${handle}`" in PAGE
        assert "go.append('your page', chevron);" in PAGE

    def test_an_admin_reading_their_profile_is_offered_the_maintainers_page(self) -> None:
        page = page_text.inlined(site_pages.profile("ada"))
        assert "'/api/me'" in page and "me.admin" in page
        assert "link.textContent = 'Maintenance';" in page

    def test_the_buttons_are_in_the_applications_case(self) -> None:
        """Lowercase, as every button on the application's own page is. A heading is
        still a sentence, which is also what the application does with one."""
        pages = _every_page()
        for label in ("create the account", "sign in", "sign out",
                      "sign everybody out", "send the invitation"):
            assert any(f">{label}</button>" in page for page in pages.values()), label
        for label in ("'kill'", "'forget'", "'let back in'", "'take off'", "'put back'"):
            assert f"button({label}," in pages["/admin"], label
        written = [text for page in pages.values()
                   for text in re.findall(r"<button[^>]*>([^<]+)</button>", page)]
        assert written and all(text == text.lower() for text in written), written

    def test_every_script_on_the_site_parses(self) -> None:
        """`tests/test_web.py` holds the long page to this; these are shorter and
        assembled the same way, out of Python strings. Found rather than listed: a list
        written here missed `/suggestions` from the day it arrived (Q511)."""
        scripted = {path: page for path, page in page_names.site_pages().items()
                    if page_text.script_of(page)}
        assert set(scripted) >= {"/join", "/signin", "/profile", "/players",
                                 "/suggestions", "/admin"}
        for path, page in scripted.items():
            done = node_session.run(_script_of(page), check=True)
            assert done.returncode == 0, f"{path}: {done.stderr}"


class TestTheLinksOpenTheGame:
    """D0. The router reads `#/game/<hash>`; `/#<hash>` fell through to the library, so
    every game named on the profile and on the maintainer's page opened the wrong page.
    G3: every game now has its own address, `/game/<hash>`, which is what they write."""

    def test_both_pages_write_the_routers_own_address(self) -> None:
        for page in (page_text.inlined(site_pages.profile("ada")), page_text.inlined(site_pages.admin("boss"))):
            assert "const GAME = '/game/';" in page
            assert "link.href = GAME + " in page
            assert "'/#' + game.hash" not in page
        # Every link the profile draws -- a table row, a milestone line, a game named
        # under *still open* -- comes out of the one builder, so there is one place the
        # address can be wrong.
        profile = _script_of(page_text.inlined(site_pages.profile("ada")))
        assert profile.count("link.href = GAME") == 1
        assert "function gameLink(hash, name)" in profile

    def test_the_address_is_one_the_application_routes(self) -> None:
        """Against the router itself rather than against a string written twice."""
        from page_text import CORE

        assert "parts[0] === 'game' && parts[1]" in CORE
        assert "parts = ['game', pathGame()];" in CORE
        assert page_text.text("/static/site/game-link.js").splitlines()[-1] == "const GAME = '/game/';"


class TestTheProfileIsInOrder:
    """D2. The payload arrives in store order, which put the game somebody played this
    morning wherever the library happened to load it."""

    def _ordered(self, games: list[dict[str, Any]]) -> list[str]:
        script = _script_of(page_text.inlined(site_pages.profile("ada")))
        found = re.search(
            r"const UNNAMED = .*?\nfunction inOrder\(games\) \{.*?\n\}\n", script, re.S
        )
        assert found is not None
        return _in_node(
            found.group(0),
            "console.log(JSON.stringify(inOrder("
            + json.dumps(games)
            + ").map(named)));",
        )

    def test_played_games_come_first_and_each_part_is_by_name(self) -> None:
        games = [
            {"hash": "a" * 12, "name": "zebra", "played": 2},
            {"hash": "b" * 12, "name": "atomic", "played": 0},
            {"hash": "c" * 12, "name": "courier", "played": 5},
            {"hash": "d" * 12, "name": "berolina", "played": 0},
        ]
        assert self._ordered(games) == ["courier", "zebra", "atomic", "berolina"]

    def test_a_game_with_no_name_is_an_unnamed_game_and_never_its_hash(self) -> None:
        """S12. The hash still keeps two unnamed games in a fixed order, unseen."""
        games = [
            {"hash": "ffff" * 3, "name": "", "played": 0},
            {"hash": "0000" * 3, "name": None, "played": 0},
            {"hash": "aaaa" * 3, "name": "alpha", "played": 0},
        ]
        assert self._ordered(games) == ["alpha", "an unnamed game", "an unnamed game"]
        assert self._ordered(games[::-1]) == self._ordered(games)

    def test_the_table_fits_a_phone(self) -> None:
        """Eight columns do not fit 375px, so the last three -- the best opponent
        beaten, the rungs and the distance to the top -- are columns on a desk and lines
        under the game's name on a phone. Nothing is hidden at either width; the cell
        and the line carry the same sentence."""
        page = page_text.inlined(site_pages.profile("ada"))
        assert '<table class="record" id="record">' in page
        assert "<th>game</th><th>played</th><th>won</th>" in page
        assert "@media (max-width: 760px)" in page
        for column, reading in (
            ("best", "rung(game)"),
            ("beat", "rungs(game)"),
            ("dist", "distance(game)"),
        ):
            assert f"cell(row, {reading}, '{column}" in page, column
            assert (
                f"table.record th.{column}, table.record td.{column}" in page
            ), column
        for label, reading in (
            ("best beaten", "rung(game)"),
            ("computer players beaten", "rungs(game)"),
            ("your rating", "distance(game)"),
        ):
            assert f"narrow(first, '{label}', {reading});" in page, label


class TestTheMaintainersPageOnAPhone:
    """D3. It laid out at 664px, so a phone showed it zoomed out and scrolling
    sideways."""

    def test_every_table_scrolls_inside_its_own_box(self) -> None:
        page = page_text.inlined(site_pages.admin("boss"))
        assert page.count("<table") == page.count('<div class="scroller">')
        assert page.count("<table") == 20, (
            "status, people, invites, requests for an invitation, games, earlier scores, how much scores moved, people "
            "beating the strongest, the training queue, computer players on people's "
            "scale, the refresher's refusals, the round, "
            "reports, dismissed reports, flagged accounts, suggestions, limits, by day, "
            "two of usage"
        )
        assert ".scroller { overflow-x: auto;" in page

    def test_signing_everybody_out_is_the_first_thing_under_the_heading(self) -> None:
        page = page_text.inlined(site_pages.admin("boss"))
        sessions = page.index("<h2>Sessions</h2>")
        assert page.index('id="revoke-all"') > sessions
        assert page.index('id="revoke-all"') < page.index("<h2>People</h2>")
        assert page.index('id="revoke-all"') < page.index("leaves every account intact")


class TestThePages:
    def test_a_handle_with_a_script_tag_in_it_never_reaches_the_page_raw(
        self, site: Site
    ) -> None:
        """The discipline task 1.10 audits the long page for, held here for the short ones.

        `identity` refuses this handle at the door, so it is written straight into the
        store to get past that: what is under test is the second defence, not the first.
        """
        raw = "<script>alert('q153')</script>"
        with site.store.cursor() as cursor:
            cursor.execute(
                "insert into users (handle, email, password) values (%s, %s, %s)",
                (raw, "sam@example.com", identity.hash_password(PASSWORD)),
            )
        site.store.commit()

        browser = site.browser()
        signed = browser.post("/api/signin", {"handle": raw, "password": PASSWORD})
        assert signed.status == 200
        page = browser.get("/profile")

        assert page.status == 200
        assert raw not in page.text
        assert html.escape(raw) in page.text

    def test_the_profile_draws_the_counts_and_the_rungs_through_text_nodes(self) -> None:
        """No `innerHTML` anywhere in the site's own pages (§6)."""
        page = page_text.inlined(site_pages.profile("ada"))
        assert "innerHTML" not in page
        assert "textContent" in page
        for count in ("games tried", "games won", "strongest beaten"):
            assert count in page

    def test_a_profile_with_nobody_signed_in_is_the_door(self, site: Site) -> None:
        answer = site.browser().get("/profile")
        assert answer.status == 303
        assert answer.headers["Location"] == "/signin"

    def test_the_sign_in_page_says_what_an_account_is_for(self) -> None:
        """L7 (`docs/archive/plan-launch.md` §4): what an account adds, in full, and what needs none."""
        page = " ".join(page_text.inlined(site_pages.signin()).split())
        assert site_pages.ACCOUNT_ADDS in page
        assert "playing practice games, need no account" in page

    def test_what_an_account_adds_is_said_in_a_players_words(self) -> None:
        """L7, with `docs/archive/plan-understanding.md` §1.3's one change of word: the strongest
        computer player on a game, never *the top*, and every clause a thing the site does."""
        said = site_pages.ACCOUNT_ADDS
        assert said.startswith("With an account your games count:")
        assert "the strongest computer player on a game" in said
        assert "the top" not in said
        for clause in ("a rating on each game", "save and publish", "by correspondence",
                       "star and vote"):
            assert clause in said, clause

    def test_the_sign_in_page_says_accounts_are_by_invitation(self) -> None:
        """Q578, the first-visit study's F5: a reader looked here for a way to make an
        account, found none, and nothing said why."""
        page = page_text.inlined(site_pages.signin())
        assert "Accounts are by invitation for now" in page

    def test_the_join_page_says_what_the_site_is_and_that_this_is_a_trial(self) -> None:
        """L8 (Q567): the invitation page carries `docs/archive/plan-understanding.md` §1.1's
        statement of purpose as written, and says the person is joining a trial."""
        page = " ".join(page_text.inlined(site_pages.join("code-1")).split())
        assert "You have been invited to the trial" in page
        assert ("<b>Chess variant sandbox is a place to find chess-like games worth "
                "playing.</b>") in page
        assert "The people who play here are the test the computer cannot replace." in page
        assert site_pages.ACCOUNT_ADDS in page  # L7: what an account adds, in full

    def test_an_empty_handle_asks_for_one_and_sends_nothing(self) -> None:
        """Q578, the study's N3: the server's answer to an empty handle is its format
        rule, which read as though something typed was wrong. The page notices the empty
        field first; a handle that breaks the rule is still the server's to refuse."""
        script = page_text.script_of(page_text.inlined(site_pages.join("code-1")))
        stub = """
          const __fields = { code: 'code-1', handle: '   ', password: 'long enough pw',
                             note: '' }, __sent = [];
          let __click = null;
          global.window = { location: { href: '' } };
          global.document = { getElementById: id => ({
            get value() { return __fields[id]; },
            set textContent(v) { __fields[id] = v; },
            addEventListener: (kind, handler) => { __click = handler; },
          }) };
          global.fetch = (path, options) => { __sent.push(path); return new Promise(() => {}); };
        """
        runner = """
          __click().then(() => {
            const empty = { said: __fields.note, sent: __sent.slice() };
            __fields.handle = 'ada';
            __click();
            console.log(JSON.stringify({ empty, named: __sent }));
          });
        """
        found = _in_node(stub + script, runner)
        assert found["empty"]["sent"] == []
        assert found["empty"]["said"].startswith("Choose a handle first")
        assert found["named"] == ["/api/join"]

    def test_the_sites_own_pages_are_listed_in_one_place(self) -> None:
        """Eleven pages, listed beside the routes, so what the site serves is one table
        rather than a search through a handler. Six of them read the cookie, and one of
        those six is reached only by an admin. The badge list, the terms (Q722) and the
        request for an invitation (Q831) are not among the six on purpose -- each reads
        the same for everybody (§3, test 7)."""
        assert routes.PAGES == (
            "/join", "/signin", "/profile", "/players", "/suggestions", "/credits",
            "/privacy", "/terms", "/badges", "/admin", "/request-invitation",
        )
        assert routes.PERSONAL == (
            "/join", "/signin", "/profile", "/players", "/suggestions", "/admin"
        )
        assert set(routes.PERSONAL) <= set(routes.PAGES)
        assert set(routes.ADMIN_ONLY) <= set(routes.PAGES) | {
            route.path for route in routes.OWN
        }


class TestTheCreditsAndTheNotice:
    """Task 1.9. Two pages the site owes a reader, and neither asks who is reading.

    Every test here runs on a machine with no Postgres, which is the point rather than a
    convenience: the pages are constants, so serving one opens no connection.
    """

    @contextlib.contextmanager
    def _served(self, app: SiteApp) -> Iterator[Browser]:
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            yield Browser(server.server_port)
        finally:
            server.shutdown()
            server.server_close()

    def test_both_pages_are_served_and_neither_reads_the_cookie(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`routes.PERSONAL` is what keeps the store out of a constant page, so the way
        to test it is to make the lookup fail and ask for both pages anyway."""

        def reached(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("the cookie was read to serve a constant page")

        monkeypatch.setattr(app, "identify", reached)
        with self._served(app) as browser:
            attributions = browser.get("/credits")
            notice = browser.get("/privacy")
            agreed = browser.get("/terms")
        assert agreed.status == 200 and "<h1>Terms</h1>" in agreed.text
        assert attributions.status == 200
        assert "<h1>Credits</h1>" in attributions.text
        assert notice.status == 200
        assert "<h1>Privacy</h1>" in notice.text

    def test_the_notice_names_the_three_things_kept(self) -> None:
        """The row's acceptance. The email, the password hash and the match records are
        what `site/identity.py` stores, and the session digest is the fourth."""
        page = site_pages.privacy()
        assert "Your email address" in page
        assert "scrypt hash" in page
        assert "Every match you play while signed in" in page
        assert "SHA-256" in page

    def test_the_notice_says_the_site_keeps_a_dated_record_of_badges(self) -> None:
        """Box E3 of `docs/archive/plan-meta-game.md`. An award row is a fifth thing the
        store holds about a person and a suggestion is the sixth (Q279), and this notice is
        where the site says what it holds."""
        page = site_pages.privacy()
        assert "seven things and nothing else" in page
        assert "A dated record of each badge you earn" in page
        assert "nothing counts how many anybody holds" in page

    def test_the_notice_says_what_a_sandbox_game_keeps(self) -> None:
        """Q192's answer, in the words a player can check against the game page."""
        page = site_pages.privacy()
        assert "A practice game keeps no person at all" in page
        assert "needs no account" in page

    def test_the_notice_says_the_usage_counts_name_nobody(self) -> None:
        """`docs/plan-usage.md` §3.3. The structural half is `STORE`'s usage tables."""
        page = site_pages.privacy()
        assert "hold no name, no address and no cookie" in page
        assert "how many different signed-in people" in page

    def test_the_notice_says_what_leaving_does(self) -> None:
        """The retention answer of 2026-09-11, in the phrase it asked for."""
        page = site_pages.privacy()
        assert "Your games stay and your name goes" in page
        assert "a former player" in page

    def _lines(self) -> list[str]:
        """What the library asks to be credited, recomputed from the asset directory
        rather than read off the page's own helper, so a page that credits a subset
        fails."""
        from chess_variant_sandbox.viz import pieces

        stems = sorted(path.stem for path in pieces.ASSET_ROOT.glob("svg/*.svg"))
        lines = pieces.credits_for([[{"stem": stem}] for stem in stems])
        assert len(lines) > 50, "this checkout has no icon library to credit"
        return lines

    def test_the_credits_page_carries_every_attribution_the_library_asks_for(
        self,
    ) -> None:
        """What the CC BY and CC BY-SA files require, for every drawing the site ships.

        Grouped by artist (D4), so what a line used to say whole is now a file under a
        heading naming the person and the licence. All three parts are still on the page
        for every file.
        """
        page = site_pages.credits()
        for line in self._lines():
            stem, _, rest = line.partition(" - ")
            artist, _, licence = rest.rpartition(" (")
            assert f"<li>{html.escape(stem)}</li>" in page, line
            assert html.escape(artist) in page, line
            assert html.escape(licence[:-1]) in page, line
        assert "Wikimedia Commons" in page
        # S18: the full record is pointed at in a visitor's words, not as a path.
        assert "The full record, with a link to every original file" in page
        assert "CREDITS.md" not in page

    def test_the_page_has_one_heading_per_artist_and_licence(self) -> None:
        """A hundred lines of `afil-black - Cburnett (CC BY-SA 4.0)` read as a file
        listing. The artists are few, so they are the headings and the files sit under
        them."""
        pairs = set()
        for line in self._lines():
            _, _, rest = line.partition(" - ")
            artist, _, licence = rest.rpartition(" (")
            pairs.add((artist, licence[:-1]))

        page = site_pages.credits()
        assert len(pairs) * 3 < len(self._lines()), "grouping has to shorten the page"
        assert page.count('<h2 class="artist">') == len(pairs)
        assert page.count('<ul class="art">') == len(pairs)
        for artist, licence in pairs:
            assert (
                f'<h2 class="artist">{html.escape(artist)} '
                f'<span class="licence muted">{html.escape(licence)}</span></h2>' in page
            ), artist

    def test_the_artists_who_drew_most_of_the_library_are_at_the_top(self) -> None:
        """The tail of the list is derivative chains that name everybody in them, and a
        heading is a paragraph long there. Ordering by how many files each drew puts the
        page's useful half first."""
        groups, _ = site_pages._grouped(self._lines())
        counts = [len(files) for _, _, files in groups]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] > 10

    def test_a_line_the_grouping_cannot_read_is_kept_whole(self) -> None:
        """The no-manifest fallback is one sentence rather than `stem - who (licence)`,
        and a sentence dropped for not parsing would be an attribution dropped."""
        sentence = "Piece icons from Wikimedia Commons; see assets/pieces/CREDITS.md."
        groups, loose = site_pages._grouped(["mace-white - Sunny3113 (CC BY-SA 3.0)", sentence])
        assert groups == [("Sunny3113", "CC BY-SA 3.0", ["mace-white"])]
        assert loose == [sentence]

    def test_neither_prose_page_carries_a_script(self) -> None:
        """§6's discipline, kept cheap: a page with nothing to run runs nothing."""
        for page in (site_pages.credits(), site_pages.privacy()):
            assert "<script>" not in page
            assert "innerHTML" not in page


class Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


SHOW = {"spec": {}, "show": {"rule": "promotion"}}
STUCK = {"spec": {}, "show": "unfinished"}


class TestShowMeIsRationed:
    """Q210 and Q214. The designer's *show me* plays 32 games a press and *show the stuck
    game* 16, so the site allows `quota.SHOW_ME_AN_HOUR` presses of either per account in
    a rolling hour and refuses visitors.

    None of this needs Postgres. The count lives in memory, and the draft itself is
    replaced by a stub so fifty presses cost nothing.
    """

    @pytest.fixture
    def clock(self, app: SiteApp, monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
        clock = Clock()
        monkeypatch.setattr(app, "presses", quota.Presses(clock))
        monkeypatch.setattr(App, "_play_draft", lambda self, body: {"played": True})
        yield clock
        app.viewer = None

    def _spend(self, app: SiteApp, user: int) -> None:
        app.viewer = user
        for _ in range(quota.SHOW_ME_AN_HOUR):
            assert app.post("/api/design/play", SHOW) == {"played": True}

    def test_the_press_past_the_limit_is_refused_with_when(
        self, app: SiteApp, clock: Clock
    ) -> None:
        self._spend(app, 7)
        clock.now += 60 * 20
        with pytest.raises(quota.Spent) as spent:
            app.post("/api/design/play", SHOW)
        assert "50 times in the last hour" in str(spent.value)
        assert "available in 40 minutes" in str(spent.value)

    def test_the_count_is_per_account(self, app: SiteApp, clock: Clock) -> None:
        self._spend(app, 7)
        app.viewer = 8
        assert app.post("/api/design/play", SHOW) == {"played": True}

    def test_the_hour_rolls(self, app: SiteApp, clock: Clock) -> None:
        """Presses leave the count one at a time, an hour after each was made."""
        app.viewer = 7
        app.post("/api/design/play", SHOW)
        clock.now += 60 * 30
        for _ in range(quota.SHOW_ME_AN_HOUR - 1):
            app.post("/api/design/play", SHOW)
        with pytest.raises(quota.Spent):
            app.post("/api/design/play", SHOW)
        clock.now += 60 * 30
        assert app.post("/api/design/play", SHOW) == {"played": True}
        with pytest.raises(quota.Spent, match="available in 30 minutes"):
            app.post("/api/design/play", SHOW)

    def test_a_refused_press_is_not_counted(self, app: SiteApp, clock: Clock) -> None:
        self._spend(app, 7)
        for _ in range(5):
            with pytest.raises(quota.Spent):
                app.post("/api/design/play", SHOW)
        clock.now += quota.SHOW_ME_WINDOW.total_seconds()
        self._spend(app, 7)

    def test_a_plain_draft_game_is_not_counted(self, app: SiteApp, clock: Clock) -> None:
        self._spend(app, 7)
        assert app.post("/api/design/play", {"spec": {}}) == {"played": True}

    def test_a_stuck_game_press_counts(self, app: SiteApp, clock: Clock) -> None:
        """Q214. *Show the stuck game* plays 16 games a press and counts like *show me*."""
        app.viewer = 7
        assert app.post("/api/design/play", STUCK) == {"played": True}
        assert app.presses.counts() == [1]

    def test_the_two_buttons_share_one_budget(self, app: SiteApp, clock: Clock) -> None:
        app.viewer = 7
        half = quota.SHOW_ME_AN_HOUR // 2
        for _ in range(half):
            app.post("/api/design/play", SHOW)
        for _ in range(quota.SHOW_ME_AN_HOUR - half):
            app.post("/api/design/play", STUCK)
        for body in (SHOW, STUCK):
            with pytest.raises(quota.Spent, match="50 times in the last hour"):
                app.post("/api/design/play", body)

    def test_a_visitor_is_asked_to_sign_in(self, app: SiteApp, clock: Clock) -> None:
        app.viewer = None
        for body in (SHOW, STUCK):
            with pytest.raises(quota.QuotaError, match="Sign in to use show me"):
                app.post("/api/design/play", body)
        assert app.post("/api/design/play", {"spec": {}}) == {"played": True}

    def test_the_site_answers_429_and_a_visitor_400(
        self, app: SiteApp, clock: Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signed_in: list[int | None] = [7]

        def identify(token: str | None) -> None:
            app.viewer = signed_in[0]

        monkeypatch.setattr(app, "identify", identify)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            for _ in range(quota.SHOW_ME_AN_HOUR):
                assert browser.post("/api/design/play", STUCK).status == 200
            for body in (SHOW, STUCK):
                spent = browser.post("/api/design/play", body)
                assert spent.status == 429
                assert "Your next press is available in 60 minutes." in spent.payload["error"]
            signed_in[0] = None
            for body in (SHOW, STUCK):
                visitor = browser.post("/api/design/play", body)
                assert visitor.status == 400
                assert "Sign in to use show me" in visitor.payload["error"]
        finally:
            server.shutdown()
            server.server_close()

    def test_the_local_app_has_no_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(App, "_play_draft", lambda self, body: {"played": True})
        local = App(RULESETS, library.RUNS_DIR)
        server = Server(("127.0.0.1", 0), handler_for(local))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            for _ in range(quota.SHOW_ME_AN_HOUR + 10):
                for body in (SHOW, STUCK):
                    assert browser.post("/api/design/play", body).status == 200
        finally:
            server.shutdown()
            server.server_close()

    def test_the_designer_prints_the_refusal_beside_the_board(self) -> None:
        """The page shows any refused draft's `error` in the designer's own box."""
        from page_text import PAGE

        play = PAGE[PAGE.index("function playDraft(") : PAGE.index("function showDraft(")]
        assert "getElementById('dSaved')" in play
        # A refusal's sentence, as `errorWords` gives it (Q682: a page fault is reworded).
        assert "errorWords(error)" in play
        core = PAGE[PAGE.index("async function api(") :]
        assert "throw new Error((answer.data && answer.data.error)" in core[: core.index("\n}\n")]

    def test_the_admin_page_lists_the_limit(self) -> None:
        assert str(quota.SHOW_ME_AN_HOUR) in [value for _, value in quota.summary()]


# A clock hour, as a Unix time.
HOUR = 1_789_200_000.0 - 1_789_200_000.0 % 3600
LIBRARY = "GET /api/library"


class TestUsageIsCounted:
    """`docs/plan-usage.md` boxes 1-4: the site counts its use by the hour, naming nobody."""

    def test_no_usage_table_has_a_column_that_names_a_person(self) -> None:
        naming = {"user", "voter", "email", "address", "token", "handle", "cookie", "ip"}
        tables = {name: cols for name, cols in STORE.items() if name.startswith("usage_")}
        assert set(tables) == {"usage_hours", "usage_people", "usage_samples"}
        for name, columns in tables.items():
            for column in columns:
                assert not set(column.split("_")) & naming, f"{name}.{column}"

    def test_a_request_is_named_by_its_template(self) -> None:
        assert usage.template("GET", "/api/game/abc123") == "GET /api/game/<hash>"
        assert usage.template("POST", "/api/test") == "POST /api/test"
        assert usage.template("POST", "/api/nonsense") == usage.OTHER
        assert usage.default_variant(usage.SESSION_ACTION, "/api/session/x/agent") == "agent"

    def test_the_buckets(self) -> None:
        assert usage.buckets([1, 2, 5, 6, 10, 11, 20, 21, 50, 51]) == (1, 2, 2, 2, 2, 1)

    def test_requests_add_up_by_hour_route_and_outcome(self) -> None:
        clock = Clock()
        clock.now = HOUR + 60
        counter = usage.Counter(clock=clock)
        counter.record(LIBRARY, "", 200, "", None, 0.01, 0.02)
        counter.record(LIBRARY, "", 200, "", None, 0.03, 0.05)
        counter.record(LIBRARY, "", 200, "", 7, 0.01, 0.01)
        assert counter.drain().hours == {}, "an unfinished hour stays in memory"
        clock.now = HOUR + 3601
        hour = usage.hour_of(HOUR)
        drained = counter.drain()
        visitors = drained.hours[(hour, LIBRARY, "", 200, "", usage.VISITOR)]
        assert visitors.count == 2
        assert visitors.cpu == pytest.approx(0.04)
        assert visitors.wall_max == pytest.approx(0.05)
        assert drained.hours[(hour, LIBRARY, "", 200, "", usage.MEMBER)].count == 1
        assert counter.drain().hours == {}

    def test_people_are_numbers_and_refusals_are_people(self) -> None:
        clock = Clock()
        clock.now = HOUR - 10
        counter = usage.Counter(clock=clock)
        clock.now = HOUR + 10
        counter.record("POST /api/session", "", 200, "", 7, 0.0, 0.0)
        counter.record("POST /api/session", "", 200, "", 7, 0.0, 0.0)
        counter.record("POST /api/session", "", 400, "QuotaError", 8, 0.0, 0.0)
        clock.now = HOUR + 3700
        [row] = counter.drain().rows
        assert (row.people, row.refused, row.peak) == (2, 1, 2)
        assert row.buckets == (1, 1, 0, 0, 0, 0)
        assert not row.partial, "the process saw the whole hour"

    def test_an_hour_the_process_did_not_see_whole_is_partial(self) -> None:
        clock = Clock()
        clock.now = HOUR + 10
        counter = usage.Counter(clock=clock)
        counter.record(LIBRARY, "", 200, "", 7, 0.0, 0.0)
        clock.now = HOUR + 3610
        counter.record(LIBRARY, "", 200, "", 7, 0.0, 0.0)
        [started_inside] = counter.drain().rows
        assert started_inside.partial
        [ended_early] = counter.drain(final=True).rows
        assert ended_early.partial and ended_early.hour == usage.hour_of(HOUR + 3600)

    def test_show_me_people_are_read_off_the_presses(self) -> None:
        presses = quota.Presses(Clock())
        for _ in range(3):
            presses.press(7)
        presses.press(8)
        clock = Clock()
        clock.now = HOUR + 10
        counter = usage.Counter(lambda: presses, clock=clock)
        counter.record(*usage.SHOW_ME, 200, "", 7, 0.3, 0.3)
        [row] = counter.drain(final=True).rows
        assert (row.people, row.peak, row.buckets) == (2, 3, (1, 1, 0, 0, 0, 0))

    def test_the_most_requests_at_once(self) -> None:
        counter = usage.Counter()
        counter.enter()
        counter.enter()
        counter.leave()
        assert counter.in_flight_max() == 2
        assert counter.in_flight_max() == 1, "the one still running"
        counter.leave()
        assert counter.in_flight_max() == 1
        assert counter.in_flight_max() == 0

    def test_the_ledger_shares_the_process_clock_by_wall_time(self) -> None:
        """Q970. Between two moments the process's CPU is split evenly among what ran."""
        clock = Clock()
        clock.now = 0.0
        ledger = usage.Ledger(clock)
        clock.now = 5.0  # nothing running: the idle floor, charged to nobody
        first = ledger.enter()
        clock.now = 7.0  # the first alone: all of it
        second = ledger.enter()
        clock.now = 10.0  # both: half each
        assert ledger.leave(first) == pytest.approx(2.0 + 1.5)
        clock.now = 11.0  # the second alone again
        with ledger.sharing():  # a walk beside it takes its half and is charged to nobody
            clock.now = 13.0
        assert ledger.leave(second) == pytest.approx(1.5 + 1.0 + 1.0)

    def test_a_request_is_charged_for_work_it_runs_on_other_threads(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q970. The playability check plays its smoke games on the core's threads, which
        the request's own thread clock never saw (`usage-first-month.md` §2: 0.48)."""
        def spin() -> None:
            ends = time.thread_time() + 0.4
            while time.thread_time() < ends:
                pass

        def elsewhere(self: Any, body: Any) -> Any:
            worker = threading.Thread(target=spin)
            worker.start()
            worker.join()
            return {"played": True}

        timings: list[usage.Timing] = []
        timed = usage.Timing

        def keep(produce: Any) -> usage.Timing:
            timings.append(timed(produce))
            return timings[-1]

        monkeypatch.setattr(app, "identify", lambda token: None)
        monkeypatch.setattr(App, "_play_draft", elsewhere)
        monkeypatch.setattr(app, "usage", usage.Counter())
        monkeypatch.setattr(usage, "Timing", keep)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            assert Browser(server.server_port).post(
                "/api/design/play", {"spec": {}}).status == 200
        finally:
            server.shutdown()
            server.server_close()
        [timing] = timings
        assert timing.thread < 0.1, "the request's own thread only waited"
        [sums] = app.usage.drain(final=True).hours.values()
        assert sums.cpu == pytest.approx(timing.cpu)
        assert sums.cpu > 0.3, "the worker thread's 0.4 CPU-s is the request's"

    def test_a_drafts_computer_move_notes_its_budget(self, app: SiteApp) -> None:
        """Q970. A draft's session is answered by the local handler, which noted no budget,
        so its computer moves landed on the `agent` row (`usage-first-month.md` §3.2)."""
        spec = json.loads((RULESETS / "chess.json").read_text(encoding="utf-8"))
        session = app.drafts.start(app.game(spec_hash(spec)), ("minimax:nodes=400", "human"))
        usage.take()
        app.post(f"/api/session/{session.id}/agent", {})
        assert usage.take() == "nodes=400"

    def test_a_server_error_is_counted_and_written_to_the_journal_with_its_traceback(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Q721. The local handler's request log is off, so this line is the only trace
        on the box of why a route answered 500."""
        def broken(self: Any, body: Any) -> Any:
            raise RuntimeError("the core fell over")

        monkeypatch.setattr(app, "identify", lambda token: None)
        monkeypatch.setattr(App, "_play_draft", broken)
        monkeypatch.setattr(app, "usage", usage.Counter())
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            assert Browser(server.server_port).post(
                "/api/design/play", {"spec": {}}).status == 500
        finally:
            server.shutdown()
            server.server_close()
        err = capfd.readouterr().err
        assert "server error 500 on POST /api/design/play" in err
        assert "Traceback" in err and "RuntimeError: the core fell over" in err
        counts = {key[3:5]: sums.count for key, sums in app.usage.drain(final=True).hours.items()}
        assert counts == {(500, "RuntimeError"): 1}

    def test_the_handler_counts_served_refused_withheld_and_invented_requests(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signed_in: list[int | None] = [7]

        def identify(token: str | None) -> None:
            app.viewer = signed_in[0]

        clock = Clock()
        clock.now = HOUR + 60
        monkeypatch.setattr(app, "identify", identify)
        monkeypatch.setattr(App, "_play_draft", lambda self, body: {"played": True})
        monkeypatch.setattr(app, "presses", quota.Presses(Clock()))
        monkeypatch.setattr(app, "usage", usage.Counter(lambda: app.presses, clock=clock))
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            assert browser.post("/api/design/play", SHOW).status == 200
            assert browser.post("/api/design/play", {"spec": {}}).status == 200
            for _ in range(quota.SHOW_ME_AN_HOUR - 1):
                assert browser.post("/api/design/play", STUCK).status == 200
            assert browser.post("/api/design/play", STUCK).status == 429
            signed_in[0] = None
            assert browser.post("/api/design/play", SHOW).status == 400
            assert browser.post("/api/design/play", STUCK).status == 400
            assert browser.post("/api/test", {}).status == 404
            assert browser.get("/api/no/such/thing").status == 404
        finally:
            server.shutdown()
            server.server_close()
            app.viewer = None
        drained = app.usage.drain(final=True)
        counts = {key[1:]: sums.count for key, sums in drained.hours.items()}
        play = "POST /api/design/play"
        stuck = "show=unfinished"
        assert counts == {
            (play, "show=rule", 200, "", usage.MEMBER): 1,
            (play, "", 200, "", usage.MEMBER): 1,
            (play, stuck, 200, "", usage.MEMBER): quota.SHOW_ME_AN_HOUR - 1,
            (play, stuck, 429, "Spent", usage.MEMBER): 1,
            (play, "show=rule", 400, "QuotaError", usage.VISITOR): 1,
            (play, stuck, 400, "QuotaError", usage.VISITOR): 1,
            ("POST /api/test", "", 404, "KeyError", usage.VISITOR): 1,
            (usage.OTHER, "", 404, "KeyError", usage.VISITOR): 1,
        }

    def test_a_flush_writes_a_restart_adds_and_old_rows_go(self) -> None:
        with _migrated() as connection:
            hour = usage.hour_of(HOUR)
            with connection.cursor() as cursor:
                cursor.execute(
                    "insert into usage_hours values (%s, 'old', '', 200, '', 'member',"
                    " 1, 0, 0, 0)",
                    (hour - timedelta(days=401),),
                )
                cursor.execute(
                    "insert into usage_samples (at) values (%s)",
                    (hour - timedelta(days=91),),
                )
            clock = Clock()
            clock.now = HOUR - 10
            first = usage.Counter(clock=clock)
            clock.now = HOUR + 10
            first.record(LIBRARY, "", 200, "", 7, 0.5, 1.0)
            first.flush(connection, final=True)
            second = usage.Counter(clock=clock)
            second.record(LIBRARY, "", 200, "", 8, 0.25, 2.0)
            clock.now = HOUR + 3700
            second.flush(connection)
            with connection.cursor() as cursor:
                cursor.execute("select route, count, cpu, wall, wall_max from usage_hours")
                assert cursor.fetchall() == [(LIBRARY, 2, 0.75, 3.0, 2.0)]
                cursor.execute("select people, peak, one, partial from usage_people")
                assert cursor.fetchall() == [(2, 1, 2, True)]
                cursor.execute("select count(*) from usage_samples")
                assert cursor.fetchone() == (0,)


class FakeCursor:
    """Records what was executed; every query answers with no rows."""

    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = ()) -> None:
        if self._connection.broken:
            raise OSError("the database is down")
        self._connection.executed.append((sql, params))

    def executemany(self, sql: str, rows: Any) -> None:
        for row in rows:
            self.execute(sql, row)

    def fetchall(self) -> list[Any]:
        return []

    def fetchone(self) -> Any:
        # A count always has a row, and here it is nought: the draft cap (Q503a) counts a
        # person's drafts before a save runs the check.
        executed = self._connection.executed
        last = executed[-1][0] if executed else ""
        return (0,) if last.lstrip().startswith("select count(") else None


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.broken = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None

    def rows(self, table: str) -> list[Any]:
        return [params for sql, params in self.executed if f"insert into {table}" in sql]

    @contextlib.contextmanager
    def connect(self) -> Iterator[FakeConnection]:
        yield self


def _fake_machine(root: Path, idle: int, busy: int, usec: int) -> None:
    """A `/proc` and a cgroup as a Linux box writes them, under `root`."""
    (root / "proc" / "self").mkdir(parents=True, exist_ok=True)
    (root / "proc" / "stat").write_text(
        f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 2 3 4\n", encoding="ascii"
    )
    (root / "proc" / "loadavg").write_text("0.42 0.30 0.20 1/100 999\n", encoding="ascii")
    (root / "proc" / "meminfo").write_text(
        "MemTotal:  3900000 kB\nMemAvailable:  2048000 kB\n", encoding="ascii"
    )
    (root / "proc" / "self" / "cgroup").write_text(
        "0::/system.slice/chess-variant-sandbox.service\n", encoding="ascii"
    )
    unit = root / "sys" / "fs" / "cgroup" / "system.slice" / "chess-variant-sandbox.service"
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "cpu.stat").write_text(
        f"usage_usec {usec}\nuser_usec 1\nsystem_usec 1\n", encoding="ascii"
    )
    (unit / "memory.current").write_text(f"{300 * 2**20}\n", encoding="ascii")


class TestTheSampler:
    """`docs/plan-usage.md` box 5: a row every five minutes, and the counts reach the store."""

    def test_the_machine_is_read_from_proc_and_the_cgroup(self, tmp_path: Path) -> None:
        _fake_machine(tmp_path, idle=1000, busy=1000, usec=5_000_000)
        before = usage.read_machine(tmp_path, tmp_path)
        _fake_machine(tmp_path, idle=1300, busy=1100, usec=7_500_000)
        after = usage.read_machine(tmp_path, tmp_path)
        assert (after.load1, after.mem_available, after.app_memory) == (0.42, 2000, 300)
        assert after.disk_free is not None and after.disk_free > 0
        assert usage.between(before, after) == (pytest.approx(0.25), pytest.approx(2.5))
        assert usage.between(None, after) == (None, None)

    def test_missing_proc_files_leave_nulls_and_the_counts_still_flush(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        clock.now = HOUR + 10
        counter = usage.Counter(clock=clock)
        counter.record(LIBRARY, "", 200, "", 7, 0.1, 0.2)
        store = FakeConnection()
        sampler = usage.Sampler(counter, store.connect, disk=tmp_path, root=tmp_path,
                                clock=clock)
        sampler.start_reading()
        clock.now = HOUR + 3610
        sampler.tick()
        [sample] = store.rows("usage_samples")
        assert sample[1:6] == (None, None, None, None, None), "no /proc, no cgroup"
        assert sample[6] is not None, "the disk is still read"
        [hours] = store.rows("usage_hours")
        assert hours[1:4] == (LIBRARY, "", 200)
        [people] = store.rows("usage_people")
        assert people[-1] is True, "the process started inside the hour"

    def test_a_tick_samples_the_interval_and_flushes_finished_hours(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        clock.now = HOUR - 10
        counter = usage.Counter(clock=clock)
        store = FakeConnection()
        sampler = usage.Sampler(counter, store.connect, disk=tmp_path, root=tmp_path,
                                clock=clock)
        _fake_machine(tmp_path, idle=1000, busy=1000, usec=0)
        sampler.start_reading()
        clock.now = HOUR + 10
        counter.enter()
        counter.leave()
        counter.record(LIBRARY, "", 200, "", 7, 0.1, 0.2)
        _fake_machine(tmp_path, idle=1100, busy=1100, usec=1_000_000)
        sampler.tick()
        [sample] = store.rows("usage_samples")
        assert sample[1:3] == (pytest.approx(0.5), pytest.approx(1.0))
        assert sample[7:] == (1, 0)
        assert store.rows("usage_hours") == [], "the hour has not ended"
        clock.now = HOUR + 3610
        sampler.tick()
        [people] = store.rows("usage_people")
        assert people[-1] is False, "the process saw the whole hour"
        assert any(sql.startswith("delete from usage_samples") for sql, _ in store.executed)

    def test_a_database_that_is_down_costs_nothing_but_the_sample(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        clock.now = HOUR + 10
        counter = usage.Counter(clock=clock)
        counter.record(LIBRARY, "", 200, "", None, 0.1, 0.2)
        store = FakeConnection()
        sampler = usage.Sampler(counter, store.connect, disk=tmp_path, root=tmp_path,
                                clock=clock)
        clock.now = HOUR + 3610
        store.broken = True
        sampler.tick()
        store.broken = False
        sampler.tick()
        [hours] = store.rows("usage_hours")
        assert hours[6] == 1, "the counts waited in memory for the next flush"

    def test_stopping_samples_and_flushes_the_hour_in_progress(self, tmp_path: Path) -> None:
        clock = Clock()
        clock.now = HOUR + 10
        counter = usage.Counter(clock=clock)
        store = FakeConnection()
        sampler = usage.Sampler(counter, store.connect, disk=tmp_path, root=tmp_path,
                                clock=clock, every=3600.0)
        sampler.start()
        counter.record(LIBRARY, "", 200, "", 7, 0.1, 0.2)
        sampler.stop(timeout=5.0)
        assert len(store.rows("usage_samples")) == 1
        [hours] = store.rows("usage_hours")
        assert hours[1] == LIBRARY
        [people] = store.rows("usage_people")
        assert people[-1] is True, "the hour had not ended"

    def test_against_the_store_a_restart_is_partial_and_old_samples_go(
        self, tmp_path: Path
    ) -> None:
        with _migrated() as connection:

            @contextlib.contextmanager
            def connect() -> Iterator[Any]:
                yield connection
                connection.commit()

            with connection.cursor() as cursor:
                cursor.execute(
                    "insert into usage_samples (at) values (%s)",
                    (usage.hour_of(HOUR) - timedelta(days=91),),
                )
            clock = Clock()
            clock.now = HOUR - 10
            first = usage.Counter(clock=clock)
            clock.now = HOUR + 10
            first.record(LIBRARY, "", 200, "", 7, 0.5, 1.0)
            usage.Sampler(first, connect, disk=tmp_path, root=tmp_path, clock=clock).tick(
                final=True
            )
            second = usage.Counter(clock=clock)
            second.record(LIBRARY, "", 200, "", 8, 0.25, 2.0)
            clock.now = HOUR + 3700
            usage.Sampler(second, connect, disk=tmp_path, root=tmp_path, clock=clock).tick()
            with connection.cursor() as cursor:
                cursor.execute("select count, people, partial from usage_hours"
                               " join usage_people using (hour, route, variant)")
                assert cursor.fetchall() == [(2, 2, True)]
                cursor.execute("select strong_seats_max, load1 from usage_samples order by at")
                assert cursor.fetchall() == [(0, None), (0, None)]


class Tables:
    """A connection whose selects answer with fake rows, by the table they read."""

    def __init__(self, **rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.asked: list[tuple[str, Any]] = []
        self._last = ""

    def cursor(self) -> Tables:
        return self

    def __enter__(self) -> Tables:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = ()) -> None:
        self.asked.append((sql, params))
        self._last = sql

    def fetchall(self) -> list[tuple[Any, ...]]:
        found = re.search(r"\bfrom (\w+)", self._last)
        return list(self.rows.get(found.group(1), [])) if found else []

    def fetchone(self) -> tuple[Any, ...] | None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


def _at(hours: float) -> Any:
    return usage.hour_of(HOUR) + timedelta(hours=hours)


PLAY = "POST /api/design/play"

# Two whole hours and one partial one, the partial after a restart.
HOUR_ROWS = [
    (_at(0), LIBRARY, "", 200, "", usage.VISITOR, 30, 0.6, 3.0, 0.4),
    (_at(0), LIBRARY, "", 200, "", usage.MEMBER, 10, 0.2, 1.0, 0.2),
    (_at(0), PLAY, "show=rule", 200, "", usage.MEMBER, 12, 6.0, 6.5, 0.9),
    (_at(0), PLAY, "show=rule", 429, "Spent", usage.MEMBER, 2, 0.0, 0.01, 0.01),
    (_at(0), PLAY, "show=rule", 400, "QuotaError", usage.VISITOR, 3, 0.0, 0.01, 0.01),
    (_at(1), PLAY, "show=rule", 200, "", usage.MEMBER, 8, 4.0, 4.2, 0.7),
    (_at(2), PLAY, "show=rule", 200, "", usage.MEMBER, 5, 2.0, 2.1, 0.5),
]
PEOPLE_ROWS = [
    (_at(0), LIBRARY, "", 4, 0, 5, 2, 2, 0, 0, 0, 0, False),
    (_at(0), PLAY, "show=rule", 3, 1, 51, 1, 0, 0, 1, 0, 1, False),
    (_at(1), PLAY, "show=rule", 2, 0, 12, 0, 1, 0, 1, 0, 0, False),
    (_at(2), PLAY, "show=rule", 4, 0, 30, 1, 1, 1, 0, 1, 0, True),
]
SAMPLE_ROWS = [
    (_at(0), 0.10, 20.0, 0.3, 2500, 300, 30000, 2, 0),
    (_at(0.5), 0.85, 240.0, 1.9, 900, 700, 30000, 9, 3),
    (_at(1), 0.40, 90.0, 0.8, 1800, 400, 29990, 4, 1),
]


class TestTheReadings:
    """`docs/plan-usage.md` box 6: the report, the samples, the admin key and the command."""

    def test_the_report_adds_up_whole_hours_and_keeps_a_partial_hour_apart(self) -> None:
        store = Tables(usage_hours=HOUR_ROWS, usage_people=PEOPLE_ROWS)
        found = usage.report(store, _at(0), _at(3))
        assert found.hours == 3
        assert found.partial == (_at(2),)
        uses = {(one.route, one.variant): one for one in found.routes}

        library_use = uses[(LIBRARY, "")]
        assert (library_use.requests, library_use.by_visitors) == (40, 30)
        assert library_use.cpu == pytest.approx(0.8)
        assert library_use.wall_mean == pytest.approx(0.1)
        assert library_use.wall_max == pytest.approx(0.4)

        show = uses[usage.SHOW_ME]
        assert (show.requests, show.refused, show.by_visitors) == (30, 5, 3)
        assert (show.people_largest, show.people_hours) == (3, 5), "whole hours only"
        assert (show.partial_hours, show.partial_people) == (1, 4)

        me = found.show_me
        assert (me.presses, me.spent, me.asked_by_visitors) == (25, 2, 3)
        assert me.buckets == (1, 1, 0, 2, 0, 1), "the partial hour is not summed in"
        assert me.partial_buckets == (1, 1, 1, 0, 1, 0)
        assert (me.hours, me.partial_hours, me.peak) == (2, 1, 51)
        assert (me.over(10), me.over(20), me.over(50)) == (3, 1, 1)
        said = " ".join(me.said())
        assert "1 partial hour(s) are left out" in said
        assert found.as_dict()["partial"] == [_at(2).isoformat()]

    def test_the_report_names_what_it_cannot_say_and_counts_no_visitor(self) -> None:
        found = usage.report(Tables(usage_hours=HOUR_ROWS, usage_people=PEOPLE_ROWS),
                             _at(0), _at(3))
        cannot = " ".join(found.cannot())
        assert "never as a person" in cannot
        assert "not how many different people" in cannot
        assert "rolling hour" in cannot and "partial hour" in cannot
        keys = json.dumps(found.as_dict()).lower()
        assert not re.search(r'"(visitors|visitorpeople|distinctvisitors|uniquevisitors)"', keys)

    def test_the_route_filter_reaches_the_query(self) -> None:
        store = Tables()
        usage.report(store, _at(0), _at(3), route=LIBRARY)
        assert all(params == (_at(0), _at(3), LIBRARY) for _, params in store.asked)

    def test_samples_over_a_range(self) -> None:
        store = Tables(usage_samples=SAMPLE_ROWS)
        box = usage.samples(store, _at(0), _at(2))
        [(sql, params)] = store.asked
        assert "at >= %s and at < %s" in sql and params == (_at(0), _at(2))
        assert box.newest is not None and box.newest.at == _at(1)
        assert box.busiest is not None and box.busiest.at == _at(0.5)
        assert box.leanest is not None and box.leanest.mem_available == 900
        assert (box.under_a_gigabyte, box.in_flight_max, box.strong_seats_seen) == (1, 9, 3)
        empty = usage.samples(Tables(), _at(0), _at(2))
        assert empty.newest is None and empty.said()[0].startswith("No samples")

    def test_strong_seats_are_never_called_a_maximum(self) -> None:
        """Q218: `strong_seats_max` is the seats live at each sample, not a five-minute
        maximum, so no reading of it may say so."""
        box = usage.samples(Tables(usage_samples=SAMPLE_ROWS), _at(0), _at(2))
        payload = box.as_dict()
        assert payload["strongSeatsAtASample"] == 3
        assert "strongSeatsAtSample" in payload["newest"]
        for line in box.said():
            if "strong seats" in line.lower():
                assert "at a sample" in line or "live" in line
                assert "maximum" not in line.lower()
        assert any("not the most at once" in line for line in payload["cannot"])

    def test_the_admin_key(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = Tables(usage_hours=HOUR_ROWS, usage_people=PEOPLE_ROWS,
                       usage_samples=SAMPLE_ROWS)
        monkeypatch.setattr(app, "connect", lambda: store)
        # The round's panel reads its own tables (`tests/test_site_rounds.py`), and the
        # by-day table its own rows (`tests/test_site_daily.py`).
        monkeypatch.setattr(app, "_admin_round", lambda connection: None)
        monkeypatch.setattr(site_server.daily, "read", lambda connection: daily.Days(()))
        payload = app._admin_view()["usage"]
        assert set(payload) == {"day", "month", "box", "kept"}
        assert "400 days" in payload["kept"] and "90 days" in payload["kept"]
        assert {"routes", "showMe", "partial", "cannot", "said"} <= set(payload["month"])
        page = page_text.inlined(site_pages.admin("boss"))
        for mark in ("use-day", "use-month", "use-show-me", "use-box", "use-cannot"):
            assert f'id="{mark}"' in page
        assert "usage(payload.usage, payload.days)" in page and "innerHTML" not in page

    def test_the_command_prints_the_readings(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = Tables(usage_hours=HOUR_ROWS, usage_people=PEOPLE_ROWS,
                       usage_samples=SAMPLE_ROWS)
        monkeypatch.setattr(db, "database_url", lambda: "postgresql:///scratch")
        monkeypatch.setattr(db, "connect", lambda url: store)
        assert cli.main(["site", "usage", "--days", "3", "--samples"]) == 0
        out = capsys.readouterr().out
        assert "POST /api/design/play show=rule" in out
        assert "1h/4" in out, "the partial hour is printed apart"
        assert "Person-hours over 10: 3, 20: 1, 50: 1" in out
        assert "What these numbers cannot say" in out
        assert out.count("strong seats live") >= 3, "the samples are listed"

        assert cli.main(["site", "usage", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["report"]["showMe"]["partialHours"] == 1
        assert "samples" not in payload

    def test_a_member_and_a_visitor_reach_no_reading(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chess_variant_sandbox.site import admin

        read: list[bool] = []
        monkeypatch.setattr(usage, "admin_view", lambda connection: read.append(True) or {})
        monkeypatch.setattr(app, "connect", lambda: Tables())
        monkeypatch.setattr(app, "_admin_round", lambda connection: None)
        role: list[str | None] = [admin.MEMBER]

        def identify(token: str | None) -> None:
            app.viewer = None if role[0] is None else 7
            app._request.role = role[0] or admin.MEMBER

        monkeypatch.setattr(app, "identify", identify)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        browser = Browser(server.server_port)
        try:
            assert browser.get("/api/admin").status == 404
            role[0] = None
            assert browser.get("/api/admin").status == 404
            assert read == [], "nothing was read for a non-admin"
            role[0] = admin.ADMIN
            answer = browser.get("/api/admin")
            assert answer.status == 200 and "usage" in answer.payload
            assert read == [True]
        finally:
            server.shutdown()
            server.server_close()
            app.viewer = None
            app._request.role = admin.MEMBER

    def test_the_command_refuses_without_the_database(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Whoever can run it holds the store; nobody else has anything to read."""
        monkeypatch.delenv(db.ENV_VAR, raising=False)
        assert cli.main(["site", "usage"]) == 2
        assert db.ENV_VAR in capsys.readouterr().err

    def test_against_the_store(self) -> None:
        with _migrated() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    "insert into usage_hours values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    HOUR_ROWS,
                )
                cursor.executemany(
                    "insert into usage_people values"
                    " (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    PEOPLE_ROWS,
                )
                cursor.executemany(
                    "insert into usage_samples values (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    SAMPLE_ROWS,
                )
            found = usage.report(connection, _at(0), _at(3))
            assert found.show_me.buckets == (1, 1, 0, 2, 0, 1)
            assert found.partial == (_at(2),)
            box = usage.samples(connection, _at(0), _at(1))
            assert len(box.rows) == 2 and box.strong_seats_seen == 3


class TestTheSender:
    def test_the_invitation_carries_the_link_and_little_else(self) -> None:
        message = mail.invitation(
            "ada@example.com", "https://chess-variant-sandbox.example/join?code=abc"
        )
        assert message.to == "ada@example.com"
        assert "https://chess-variant-sandbox.example/join?code=abc" in message.body
        assert "<" not in message.body, "the invitation is plain text"
        # The site's name as the user settled it: no article, the capital kept.
        assert message.subject == "You're invited to try Chess variant sandbox"
        assert message.body.startswith("Somebody has invited you to Chess variant sandbox:")

    def test_the_invitation_says_who_what_and_the_one_promise(self) -> None:
        """L8 of `docs/plan-launch.md` (§5.3, Q567): who sent it, what the site is (§1.1 of
        `docs/archive/plan-understanding.md`), that it is a trial and what is asked, and that no
        other email comes unless the person switches on the move email."""
        message = mail.invitation("bo@example.com", "https://x.example/join?code=c", "ada")
        body = " ".join(message.body.split())
        assert message.subject == "ada invited you to try Chess variant sandbox"
        assert body.startswith("ada has invited you to Chess variant sandbox: a place to "
                               "find chess-like games worth playing.")
        for said in ("This is a trial:", "tell ada what confused you", "It costs nothing",
                     "nothing on the site counts down",
                     "The site sends you no other email unless you ask it to",
                     "switch it off in one click",
                     "The link works once and expires in two weeks"):
            assert said in body, said
        assert "<" not in message.body and "https://x.example/join?code=c" in message.body

    def test_the_invitation_says_how_long_its_link_lasts(self) -> None:
        for days, said in ((14, "two weeks"), (7, "a week"), (1, "a day"), (30, "30 days")):
            body = " ".join(mail.invitation("bo@example.com", "l", None, days).body.split())
            assert f"expires in {said};" in body, days

    def test_sending_without_a_key_names_the_variable_and_the_way_round_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing reaches the network: the refusal comes before the request is built."""
        monkeypatch.delenv(mail.KEY_VAR, raising=False)
        with pytest.raises(mail.MailError) as error:
            mail.send(mail.invitation("ada@example.com", "https://chess-variant-sandbox.example/join"))
        assert mail.KEY_VAR in str(error.value)
        assert "--no-send" in str(error.value)

    def test_the_request_names_itself_rather_than_urllib(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cloudflare in front of the service refuses urllib's default User-Agent (1010)."""
        sent: list[Request] = []

        class Accepted:
            status = 200

            def read(self) -> bytes:
                return b"{}"

            def __enter__(self) -> Accepted:
                return self

            def __exit__(self, *_: object) -> None:
                return None

        def capture(request: Request, timeout: float) -> Accepted:
            sent.append(request)
            return Accepted()

        monkeypatch.setenv(mail.KEY_VAR, "key")
        monkeypatch.setenv(mail.FROM_VAR, "site@chess-variant-sandbox.example")
        monkeypatch.setattr(mail.urllib.request, "urlopen", capture)
        mail.send(mail.invitation("ada@example.com", "https://chess-variant-sandbox.example/join"))
        [request] = sent
        agent = request.get_header("User-agent")
        assert agent == mail.USER_AGENT
        assert "urllib" not in agent.lower()

    def test_a_link_needs_the_sites_own_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The app behind Caddy never sees the address it is reached at, so an invitation
        cannot work it out from a request."""
        monkeypatch.delenv(mail.URL_VAR, raising=False)
        with pytest.raises(mail.MailError) as error:
            mail.site_url()
        assert mail.URL_VAR in str(error.value)

    def test_the_service_refusal_comes_back_as_a_sentence(self) -> None:
        with pytest.raises(mail.MailError) as error:
            mail._accepted(422, b'{"message": "the domain is not verified"}')
        assert "the domain is not verified" in str(error.value)
        assert "422" in str(error.value)


class TestPasswordsThemselves:
    def test_a_password_verifies_and_a_wrong_one_does_not(self, real_scrypt: int) -> None:
        """At the site's own cost. Every other test hashes at a cheap one
        (`conftest.py`'s `scrypt_is_cheap`), so this is where n=2**14 is still run."""
        stored = identity.hash_password(PASSWORD)
        assert int(stored.split("$")[1]) == real_scrypt
        assert identity.verify(stored, PASSWORD)
        assert not identity.verify(stored, PASSWORD + "!")

    def test_the_same_password_hashes_differently_every_time(self) -> None:
        """The salt. Two people with one password have two rows that look unalike."""
        assert identity.hash_password(PASSWORD) != identity.hash_password(PASSWORD)

    def test_the_stored_string_carries_its_own_cost(self) -> None:
        """So the cost can be raised without a migration and old rows still verify."""
        stored = identity.hash_password(PASSWORD)
        kind, n, r, p, _salt, _digest = stored.split("$")
        assert (kind, int(n), int(r), int(p)) == ("scrypt", identity.SCRYPT_N, 8, 1)

    def test_a_row_with_no_password_verifies_nothing(self) -> None:
        assert not identity.verify(None, "")
        assert not identity.verify("", "")
        assert not identity.verify("not-a-scrypt-string", "anything")

    def test_every_check_costs_one_hash_at_todays_cost_whatever_it_is_handed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q379. No row costs what a row costs, so the time a check takes names nothing.

        The stand-in is hashed at the same n, r and p and with a salt of the same length
        as a row `hash_password` writes, which is what makes it cost the same.
        """
        stored = identity.hash_password(PASSWORD)
        made: list[tuple[Any, ...]] = []
        real = identity._scrypt

        def counted(*args: Any) -> bytes:
            made.append(args)
            return real(*args)

        monkeypatch.setattr(identity, "_scrypt", counted)
        for handed in (stored, None, "", "not-a-scrypt-string", "scrypt$1$2$3$zz$00"):
            identity.verify(handed, PASSWORD)

        assert len(made) == 5
        assert {
            (len(salt), n, r, p) for _password, salt, n, r, p in made
        } == {(identity.SALT_BYTES, identity.SCRYPT_N, identity.SCRYPT_R, identity.SCRYPT_P)}


class TestTheMaintainerCommands:
    """The parser, as far as the database refusal: both reach `db.database_url` and stop."""

    def test_invite_refuses_without_a_database_url(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv(db.ENV_VAR, raising=False)
        assert cli.main(["site", "invite", "ada@example.com"]) == 2
        assert db.ENV_VAR in capsys.readouterr().err

    def test_revoke_refuses_without_a_database_url(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv(db.ENV_VAR, raising=False)
        assert cli.main(["site", "revoke"]) == 2
        assert db.ENV_VAR in capsys.readouterr().err


# --------------------------------------------------- B2's two filters, `chess_variant_sandbox/site/browse.py`


def _row(
    game: str,
    user: int,
    *,
    agent: str = "minimax:nodes=400",
    result: float | None = 1.0,
    finished: bool = True,
    source: str | None = None,
) -> match_rows.Row:
    """A leaderboard row with a person against a computer, built here rather than
    through a store -- `browse._untried` is pure over exactly this shape, the same way
    `browse._human_games` already is over `Row`.
    """
    return match_rows.Row(
        id=f"{game}-{user}-{agent}",
        game=game,
        mode=match_rows.LEADERBOARD,
        seat0=rating.person(user),
        seat1=rating.computer(agent),
        seed=0,
        plies=10,
        result=result,
        reason="",
        left=False,
        top=True,
        top_players=(),
        before=(None, None),
        source=source,
        started=datetime(2026, 1, 1),
        moved=None,
        finished=datetime(2026, 1, 1) if finished else None,
    )


class TestUntriedIsPureOverTheRows:
    """Box B2 of `docs/archive/plan-meta-game.md`. `browse._untried` reads the same rows
    `unbeaten.record_from` does, so it is proved the same way -- rows built by hand,
    no store -- and by the same rule: sandbox and the local application's rows are
    `unbeaten.py`'s to keep untested twice, not this file's.
    """

    def test_empty_with_no_viewer(self) -> None:
        """Nobody to ask about, so nothing is marked either way (§2.1's *absent is not
        zero*) -- `order`'s filter then keeps nothing for a visitor rather than refusing
        the whole listing."""
        rows = (_row("alpha", user=1),)
        assert browse._untried(rows, ["alpha"], None) == {}

    def test_a_game_this_viewer_has_a_finished_row_on_is_not_untried(self) -> None:
        rows = (_row("alpha", user=1),)
        assert browse._untried(rows, ["alpha"], 1) == {"alpha": False}

    def test_a_game_with_no_row_for_this_viewer_is_untried(self) -> None:
        rows = (_row("alpha", user=1),)
        assert browse._untried(rows, ["alpha", "beta"], 1) == {
            "alpha": False, "beta": True,
        }

    def test_another_persons_row_leaves_it_untried_for_this_viewer(self) -> None:
        rows = (_row("alpha", user=2),)
        assert browse._untried(rows, ["alpha"], 1) == {"alpha": True}

    def test_a_draw_or_a_loss_still_counts_as_tried(self) -> None:
        rows = (_row("alpha", user=1, result=0.5), _row("beta", user=1, result=0.0))
        assert browse._untried(rows, ["alpha", "beta"], 1) == {
            "alpha": False, "beta": False,
        }

    def test_a_sandbox_or_local_app_row_leaves_it_untried(self) -> None:
        """Neither counts as a try, the same rule `unbeaten.record_from` reads off the
        same rows for the profile's own *untried* state."""
        rows = (_row("alpha", user=1, source=match_rows.LOCAL_APP),)
        assert browse._untried(rows, ["alpha"], 1) == {"alpha": True}


class TestTheTwoNewFilters:
    """`order`'s half of B2, over cards built directly -- `unbeaten` is proved the same
    way against a live store in `tests/test_site_unbeaten.py`, and these two need no
    store to prove: everything they read is already a plain field on the card.
    """

    def _cards(self) -> list[dict[str, Any]]:
        return [
            {"name": "alpha", "untried": True, "humanGames": 0},
            {"name": "beta", "untried": False, "humanGames": 3},
            {"name": "gamma", "untried": None, "humanGames": 0},
        ]

    def test_it_filters_on_untried(self) -> None:
        kept = browse.order(self._cards(), keep=browse.UNTRIED)
        assert [card["name"] for card in kept] == ["alpha"]

    def test_untried_keeps_nothing_with_no_viewer_signed_in(self) -> None:
        """Every card's `untried` is null with nobody to ask (`browse._untried`), so the
        filter empties the listing rather than refusing it outright -- the same *absent
        is not zero* `unbeaten` already lives by for an unrated game."""
        cards = [{"name": name, "untried": None} for name in ("alpha", "beta")]
        assert browse.order(cards, keep=browse.UNTRIED) == []

    def test_it_filters_on_unplayed(self) -> None:
        kept = browse.order(self._cards(), keep=browse.UNPLAYED)
        assert [card["name"] for card in kept] == ["alpha", "gamma"]

    def test_an_unknown_filter_is_refused_by_name(self) -> None:
        with pytest.raises(browse.BrowseError) as error:
            browse.order(self._cards(), keep="stale")
        message = str(error.value)
        assert browse.UNTRIED in message and browse.UNPLAYED in message

    def test_untried_and_unplayed_are_two_different_games(self) -> None:
        """A game can be untried by you and played by others, or played by nobody and
        already tried by you -- the two filters read different columns and neither
        implies the other."""
        cards = [
            {"name": "tried-by-nobody", "untried": True, "humanGames": 0},
            {"name": "tried-by-someone-else", "untried": True, "humanGames": 2},
        ]
        assert [c["name"] for c in browse.order(cards, keep=browse.UNPLAYED)] == [
            "tried-by-nobody"
        ]
        assert [c["name"] for c in browse.order(cards, keep=browse.UNTRIED)] == [
            "tried-by-nobody", "tried-by-someone-else",
        ]


def _played(
    game: str,
    *,
    seat0: rating.PlayerKey,
    seat1: rating.PlayerKey,
    when: datetime,
    result: float | None = 1.0,
    source: str | None = None,
) -> match_rows.Row:
    """A finished leaderboard row with either kind of player in either seat."""
    return match_rows.Row(
        id=f"{game}-{when.isoformat()}",
        game=game,
        mode=match_rows.LEADERBOARD,
        seat0=seat0,
        seat1=seat1,
        seed=0,
        plies=10,
        result=result,
        reason="",
        left=False,
        top=True,
        top_players=(),
        before=(None, None),
        source=source,
        started=when,
        moved=None,
        finished=when,
    )


def _rated(user: int, score: float, deviation: float = 60.0) -> rating.Player:
    return rating.Player(
        key=rating.person(user), rating=score, deviation=deviation, volatility=0.06
    )


class TestTheGamePagesThreeLines:
    """Box B1's readings (`docs/archive/plan-meta-game.md`), pure over rows and a table.

    The page draws what `SiteApp._standing` sends and works nothing out itself, so what
    the lines say is settled here. *First played by* is M5 and *the highest-rated person
    here* is S2; the ordered clears are `unbeaten.Tag.reached`, which the last two hold.

    Rows are built by hand rather than through a store, the way `browse._untried`'s are:
    every rule these read is a rule about one row.
    """

    ADA = 1
    GRACE = 2
    EASY = "minimax:nodes=400"
    HARD = "minimax:nodes=6400"
    NAMED = {ADA: "ada", GRACE: "grace"}
    WHEN = datetime(2026, 9, 14, 12)

    def _row(
        self, user: int | None = None, *, when: datetime | None = None, **rest: Any
    ) -> match_rows.Row:
        person = rating.person(user) if user is not None else rating.computer(self.EASY)
        return _played(
            "alpha",
            seat0=person,
            seat1=rating.computer(self.HARD),
            when=when or self.WHEN,
            **rest,
        )

    # ------------------------------------------------------------- first played by

    def test_nobody_has_played_it_says_nothing(self) -> None:
        """§2.1's *absent is not zero*: the page leaves the line out rather than
        drawing an empty one."""
        assert site_server._first_played((), self.NAMED) is None

    def test_the_first_counted_row_names_its_person_and_its_date(self) -> None:
        rows = (
            self._row(self.ADA),
            self._row(self.GRACE, when=datetime(2026, 9, 15, 12)),
        )
        assert site_server._first_played(rows, self.NAMED) == {
            "handles": ["ada"], "when": self.WHEN.isoformat(),
        }

    def test_two_computer_players_are_not_a_human_game(self) -> None:
        """The reading M5 and the *nobody has played this* filter share: a row the
        rating run wrote is not somebody playing."""
        rows = (self._row(), self._row(self.ADA, when=datetime(2026, 9, 15, 12)))
        found = site_server._first_played(rows, self.NAMED)
        assert found is not None and found["handles"] == ["ada"]

    def test_a_row_loaded_from_the_local_application_is_not_one(self) -> None:
        """§3 keeps those rows for the anchor and counts them in no leaderboard."""
        rows = (self._row(self.ADA, source=match_rows.LOCAL_APP),)
        assert site_server._first_played(rows, self.NAMED) is None

    def test_an_unfinished_row_is_not_one(self) -> None:
        rows = (self._row(self.ADA, result=None),)
        assert site_server._first_played(rows, self.NAMED) is None

    def test_both_people_hold_a_game_between_two_of_them(self) -> None:
        """Phase 3's rows, which nothing writes yet. The milestone is theirs jointly,
        so the line names both rather than picking one."""
        rows = (
            _played(
                "alpha",
                seat0=rating.person(self.ADA),
                seat1=rating.person(self.GRACE),
                when=self.WHEN,
            ),
        )
        found = site_server._first_played(rows, self.NAMED)
        assert found is not None and found["handles"] == ["ada", "grace"]

    def test_somebody_who_left_is_named_as_the_store_names_them(self) -> None:
        """`unbeaten.handles` answers `identity.FORMER` for a withdrawn account, which
        is the privacy notice's promise that the games stay and the name goes."""
        rows = (self._row(self.ADA),)
        found = site_server._first_played(rows, {self.ADA: identity.FORMER})
        assert found is not None and found["handles"] == [identity.FORMER]

    # -------------------------------------------------------- highest-rated person

    def test_a_table_with_nobody_established_has_no_highest(self) -> None:
        """Nobody holds it until their rating has settled, so a newcomer two games in
        is not shown as top of a table of one."""
        table = {rating.person(self.ADA): _rated(self.ADA, 1700, deviation=300.0)}
        assert site_server._highest(table, self.NAMED) is None

    def test_the_computer_players_are_not_people(self) -> None:
        table = {
            rating.computer(self.HARD): rating.Player(
                key=rating.computer(self.HARD), rating=2300, deviation=30.0,
                volatility=0.06,
            )
        }
        assert site_server._highest(table, self.NAMED) is None

    def test_the_established_person_at_the_top_is_named_with_their_rating(self) -> None:
        table = {
            rating.person(self.ADA): _rated(self.ADA, 1700),
            rating.person(self.GRACE): _rated(self.GRACE, 1620),
        }
        assert site_server._highest(table, self.NAMED) == {
            "handles": ["ada"], "rating": 1700,
        }

    def test_everybody_level_at_the_top_holds_it_together(self) -> None:
        table = {
            rating.person(self.ADA): _rated(self.ADA, 1700),
            rating.person(self.GRACE): _rated(self.GRACE, 1700),
        }
        found = site_server._highest(table, self.NAMED)
        assert found is not None and found["handles"] == ["ada", "grace"]

    # --------------------------------------------------------------- the clears

    def _cleared(self, user: int, when: datetime, seat: int = 0) -> match_rows.Row:
        person, computer = rating.person(user), rating.computer(self.HARD)
        return _played(
            "alpha",
            seat0=person if seat == 0 else computer,
            seat1=computer if seat == 0 else person,
            when=when,
            result=1.0 if seat == 0 else 0.0,
        )

    def _tag(self, rows: tuple[match_rows.Row, ...]) -> unbeaten.Tag:
        table = {
            rating.computer(self.HARD): rating.Player(
                key=rating.computer(self.HARD), rating=2300, deviation=30.0,
                volatility=0.06,
            )
        }
        return unbeaten.tag_from("alpha", rows, table, self.NAMED)

    def test_an_unbeaten_game_has_no_clears_to_list(self) -> None:
        assert self._tag(()).reached == ()

    def test_every_clear_is_kept_in_the_order_they_were_made(self) -> None:
        """And the first clear and the count are readings of that list, so the card and
        the page cannot disagree about who was first."""
        rows = (
            self._cleared(self.GRACE, datetime(2026, 9, 14, 12), seat=1),
            self._cleared(self.ADA, datetime(2026, 9, 15, 12)),
        )
        tag = self._tag(rows)
        assert [(clear.handle, clear.seat) for clear in tag.reached] == [
            ("grace", 1), ("ada", 0),
        ]
        assert tag.first is not None and tag.first.handle == "grace"
        assert tag.clears == 2 and tag.cleared is True

    def test_one_person_clearing_twice_is_one_clear(self) -> None:
        rows = (
            self._cleared(self.ADA, datetime(2026, 9, 14, 12)),
            self._cleared(self.ADA, datetime(2026, 9, 15, 12), seat=1),
        )
        assert len(self._tag(rows).reached) == 1


# ------------------------------------------ A2, the profile shows the meta-game


class TestTheProfilesTwoColumns:
    """Box A2's two table columns, pure over rows and tables.

    A2's box is in `docs/archive/plan-meta-game.md`. *n of k rungs* is M4 counted rather
    than dated, and *distance to the top* is T3. Both live in `site/milestones.py` so that
    the page draws them and works nothing out, and both are proved here the way B1's three
    lines are, over rows built by hand.
    """

    ADA = 1
    GRACE = 2
    WEAK = "random"
    TOP = "minimax:nodes=6400"
    GONE = "greedy"
    WHEN = datetime(2026, 9, 14, 12)

    def _ladder(self, *people: rating.Player) -> dict[rating.PlayerKey, rating.Player]:
        table: dict[rating.PlayerKey, rating.Player] = {
            rating.computer(agent): rating.Player(
                key=rating.computer(agent), rating=score, deviation=30.0, volatility=0.06
            )
            for agent, score in ((self.WEAK, 1200.0), (self.TOP, 2300.0))
        }
        return {**table, **{player.key: player for player in people}}

    def _win(
        self, agent: str, *, user: int = ADA, result: float = 1.0, when: int = 0
    ) -> match_rows.Row:
        return _played(
            "alpha",
            seat0=rating.person(user),
            seat1=rating.computer(agent),
            when=self.WHEN + timedelta(minutes=when),
            result=result,
        )

    # ----------------------------------------------------------------- n of k rungs

    def test_an_untried_game_has_reached_none_of_its_rungs(self) -> None:
        """Still *of 2*, because the ladder is what the column is counting against."""
        found = milestones.rungs_from(self.ADA, (), {"alpha": self._ladder()})
        assert found == {"alpha": milestones.Rungs(beaten=0, of=2)}

    def test_each_computer_player_beaten_counts_once(self) -> None:
        rows = [
            self._win(self.WEAK),
            self._win(self.WEAK, when=1),
            self._win(self.TOP, when=2),
        ]
        found = milestones.rungs_from(self.ADA, rows, {"alpha": self._ladder()})
        assert found == {"alpha": milestones.Rungs(beaten=2, of=2)}

    def test_a_draw_or_a_loss_reaches_no_rung(self) -> None:
        rows = [self._win(self.WEAK, result=0.5), self._win(self.TOP, result=0.0, when=1)]
        found = milestones.rungs_from(self.ADA, rows, {"alpha": self._ladder()})
        assert found["alpha"].beaten == 0

    def test_somebody_elses_win_is_not_yours(self) -> None:
        rows = [self._win(self.WEAK, user=self.GRACE)]
        found = milestones.rungs_from(self.ADA, rows, {"alpha": self._ladder()})
        assert found["alpha"].beaten == 0

    def test_a_win_over_a_player_the_table_no_longer_carries_counts_for_nothing(
        self,
    ) -> None:
        """The column reads the table as it stands now, so it reopens with M4 when a
        rung is added and closes with it when one is dropped."""
        rows = [self._win(self.GONE)]
        found = milestones.rungs_from(self.ADA, rows, {"alpha": self._ladder()})
        assert found["alpha"] == milestones.Rungs(beaten=0, of=2)

    def test_a_game_with_no_ladder_has_no_rungs_to_be_n_of(self) -> None:
        assert milestones.rungs_from(self.ADA, (), {"alpha": {}}) == {}

    def test_a_sandbox_row_reaches_no_rung(self) -> None:
        rows = [replace(self._win(self.TOP), mode=match_rows.SANDBOX)]
        found = milestones.rungs_from(self.ADA, rows, {"alpha": self._ladder()})
        assert found["alpha"].beaten == 0

    # ---------------------------------------------------------- distance to the top

    def test_a_game_you_have_no_rating_on_is_absent_rather_than_a_gap_of_nought(
        self,
    ) -> None:
        assert milestones.distance_from(self.ADA, {"alpha": self._ladder()}) == {}

    def test_the_gap_is_the_top_less_your_own_rating(self) -> None:
        table = self._ladder(_rated(self.ADA, 2100.0))
        found = milestones.distance_from(self.ADA, {"alpha": table})
        assert found == {
            "alpha": milestones.Distance(points=200.0, established=True, rating=2100.0)
        }

    def test_a_rating_above_the_top_is_a_negative_gap(self) -> None:
        table = self._ladder(_rated(self.ADA, 2400.0))
        assert milestones.distance_from(self.ADA, {"alpha": table})["alpha"].points == -100.0

    def test_a_provisional_rating_says_so(self) -> None:
        """The game's own table does not rank one, so the gap travels with the fact."""
        table = self._ladder(_rated(self.ADA, 2100.0, deviation=300.0))
        assert not milestones.distance_from(self.ADA, {"alpha": table})["alpha"].established

    def test_a_game_with_no_computer_players_has_no_top_to_be_below(self) -> None:
        table = {rating.person(self.ADA): _rated(self.ADA, 2100.0)}
        assert milestones.distance_from(self.ADA, {"alpha": table}) == {}


# A DOM small enough to render one list and read it back, so what a line on the profile
# actually says is under test rather than whether its source string is in the page. The
# profile's script runs whole over it -- `fetch` answers *not ok*, which is the path a
# signed-out reader takes, so the two calls at the foot of the script settle quietly and
# the drawing functions are then called by hand.
_DOM = """
function element(tag) {
  return {
    tag, className: '', own: '', children: [],
    set textContent(said) { this.own = said; this.children = []; },
    get textContent() {
      return this.own + this.children.map(child => child.textContent).join('');
    },
    appendChild(child) { this.children.push(child); return child; },
    replaceChildren() { this.own = ''; this.children = []; },
    addEventListener() {},
    prepend() {},
  };
}
const BY_ID = {};
const document = {
  createElement: element,
  createTextNode(said) { return {textContent: said}; },
  getElementById(id) { return BY_ID[id] || (BY_ID[id] = element('div')); },
  addEventListener() {},
};
const window = {location: {}};
async function fetch() { return {ok: false, json: async () => ({})}; }
"""


def _rendered(runner: str) -> Any:
    """Run the profile's whole script over that DOM and read back what it printed."""
    return _in_node(_DOM + _script_of(page_text.inlined(site_pages.profile("ada"))), runner)


def _done(lines: list[dict[str, Any]], held: list[dict[str, Any]] | None = None) -> Any:
    """What the dated list reads, milestones and badges together."""
    return _rendered(
        f"drawDone({json.dumps(lines)}, {json.dumps(held or [])});"
        "console.log(JSON.stringify(document.getElementById('done').textContent));"
    )


class TestTheProfileDrawsTheMetaGame:
    """A2's own half. The dated lines and *still open* under the table, and the two new
    columns in it, all drawn from `GET /api/profile` through text nodes.

    Each part is absent when there is nothing to say. An empty heading would read as a
    nought, and §2.1's third rule is that absent is not zero.
    """

    LINES = [
        {"kind": "both-seats", "rule": "A win from each seat of this game.",
         "when": "2026-09-14T12:00:00+00:00", "game": "aaa", "name": "alpha", "seat": 1},
        {"kind": "every-game", "rule": "A game on every game rated now.",
         "when": "2026-09-13T12:00:00+00:00", "game": "bbb", "name": "beta", "seat": 0},
    ]

    def test_every_milestone_kind_the_reader_sends_has_a_line_here(self) -> None:
        """The page names the kinds as wire strings, so this is what keeps the two sets
        together when a milestone is added."""
        assert set(site_pages._MILESTONE_LINES) == set(milestones.KINDS)

    def test_nothing_on_the_page_is_written_as_markup(self) -> None:
        assert "innerHTML" not in page_text.inlined(site_pages.profile("ada"))

    def test_a_line_names_its_milestone_its_game_its_date_and_its_seat(self) -> None:
        said = _done(self.LINES)
        assert said.startswith("What you have done")
        assert "Both sides, on alpha, 2026-09-14, playing second." in said
        assert "The tour, completed on beta, 2026-09-13, playing first." in said

    def test_a_line_carries_its_own_rule(self) -> None:
        """§3's seventh test. A page that named a rule only once you had met it would be
        hiding every rule you had not."""
        said = _done(self.LINES)
        for line in self.LINES:
            assert line["rule"] in said

    def test_a_person_with_no_milestones_gets_no_heading(self) -> None:
        assert _done([]) == ""

    def test_still_open_names_a_kind_its_counts_and_a_few_of_the_untried_games(
        self,
    ) -> None:
        axes = [{"axis": "board", "kinds": [
            {"bin": "small", "tried": False,
             "untried": ["aaa", "bbb", "ccc", "ddd", "eee"],
             "unwon": ["aaa"], "unbeaten": ["fff"]},
        ]}]
        said = _rendered(
            "NAMES = {aaa: 'alpha', bbb: 'beta', ccc: 'gamma', ddd: 'delta',"
            " eee: 'epsilon'};"
            f"drawStillOpen({json.dumps(axes)});"
            "console.log(JSON.stringify(document.getElementById('open').textContent));"
        )
        assert "Still open for you" in said
        assert "board" in said
        assert (
            "small (a kind you have not tried) — 5 untried, 1 not yet won, "
            "1 still Unbeaten. alpha, beta, gamma, delta and 1 more."
        ) in said

    def test_still_open_speaks_in_the_kind_cards_words(self) -> None:
        """S6. The heading is the question and the line the kind, both the server's,
        rather than the grid's own keys (*ending*, *ground*)."""
        axes = [{"axis": "ending", "asks": "How the game is won", "kinds": [
            {"bin": "ground", "words": "won by claiming enough of the board",
             "tried": True, "untried": ["aaa"], "unwon": [], "unbeaten": []},
        ]}]
        said = _rendered(
            "NAMES = {aaa: 'alpha'};"
            f"drawStillOpen({json.dumps(axes)});"
            "console.log(JSON.stringify(document.getElementById('open').textContent));"
        )
        assert "How the game is won" in said
        assert "won by claiming enough of the board — 1 untried" in said
        assert "ending" not in said and "ground —" not in said
        assert "behaviour grid" not in said

    def test_a_kind_you_have_tried_is_not_marked_as_one_you_have_not(self) -> None:
        axes = [{"axis": "board", "kinds": [
            {"bin": "small", "tried": True, "untried": [], "unwon": ["aaa"],
             "unbeaten": []},
        ]}]
        said = _rendered(
            "NAMES = {aaa: 'alpha'};"
            f"drawStillOpen({json.dumps(axes)});"
            "console.log(JSON.stringify(document.getElementById('open').textContent));"
        )
        assert "a kind you have not tried" not in said
        assert "small — 1 not yet won." not in said
        assert "small — 1 not yet won" in said

    def test_a_bin_with_nothing_open_is_left_out(self) -> None:
        axes = [{"axis": "board", "kinds": [
            {"bin": "small", "tried": True, "untried": [], "unwon": [], "unbeaten": []},
            {"bin": "large", "tried": True, "untried": ["aaa"], "unwon": [],
             "unbeaten": []},
        ]}]
        said = _rendered(
            "NAMES = {aaa: 'alpha'};"
            f"drawStillOpen({json.dumps(axes)});"
            "console.log(JSON.stringify(document.getElementById('open').textContent));"
        )
        assert "small" not in said
        assert "large" in said

    def test_an_axis_with_nothing_open_draws_no_section(self) -> None:
        """Both ways of having nothing to say. A bin list that is empty, and bins whose
        every reading is closed, leave out the heading rather than draw an empty one."""
        for kinds in (
            [],
            [{"bin": "small", "tried": True, "untried": [], "unwon": [],
              "unbeaten": []}],
        ):
            axes = [{"axis": "army", "kinds": kinds}]
            said = _rendered(
                f"drawStillOpen({json.dumps(axes)});"
                "console.log(JSON.stringify("
                "document.getElementById('open').textContent));"
            )
            assert said == "", kinds

    def test_the_two_columns_word_their_own_readings(self) -> None:
        said = _rendered(
            "console.log(JSON.stringify(["
            "rungs({rungs: {beaten: 2, of: 5}}), rungs({rungs: {beaten: 0, of: 5}}),"
            " rungs({}),"
            " distance({distance: {points: 118.4, rating: 1612.4, established: true}}),"
            " distance({distance: {points: -23.6, rating: 1754.6, established: true}}),"
            " distance({distance: {points: 0, rating: 1731, established: true}}),"
            " distance({distance: {points: 40, rating: 1691, established: false}}),"
            " distance({})]));"
        )
        # F8: the person's own rating first, then how far it sits from the strongest.
        assert said == [
            "2 of 5", "none of 5", "—",
            "1612, 118 below the strongest", "1755, 24 above the strongest",
            "1731, equal to the strongest", "1691, 40 below the strongest (still settling)",
            "—",
        ]

    def test_topping_no_game_shows_no_count_of_games_topped(self) -> None:
        """S2's count. A person who tops nothing is on no list rather than shown a
        nought (§3, test 5)."""
        counts = {"rated": 40, "tried": 3, "won": 1, "top": 0, "highest": 0}
        said = _rendered(
            f"drawCounts({json.dumps(counts)});"
            "console.log(JSON.stringify(document.getElementById('counts').textContent));"
        )
        assert "top rated" not in said
        assert "games tried" in said

    def test_topping_a_game_shows_the_count(self) -> None:
        counts = {"rated": 40, "tried": 3, "won": 1, "top": 1, "highest": 2}
        said = _rendered(
            f"drawCounts({json.dumps(counts)});"
            "console.log(JSON.stringify(document.getElementById('counts').textContent));"
        )
        assert "2games where you are top rated" in said


class TestAnEmptyProfileSaysWhatToDo:
    """L2 of `docs/archive/plan-launch.md` §3.2. A profile with nothing on it says what a person
    can do, never three noughts over a table of *none yet*; once a rated game is finished
    the counts and the table are back and the line is gone."""

    @staticmethod
    def _game(name: str, **record: Any) -> dict[str, Any]:
        return {"hash": name * 4, "name": name, "played": 0, "won": 0, "drawn": 0,
                "lost": 0, "unbeaten": True, "retired": False, **record}

    def _drawn(self, counts: dict[str, Any], games: list[dict[str, Any]]) -> Any:
        return _rendered(
            f"const COUNTS = {json.dumps(counts)}, GAMES = {json.dumps(games)};"
            "drawCounts(COUNTS); drawStart(COUNTS, GAMES);"
            "drawGames(GAMES, COUNTS.tried > 0);"
            "console.log(JSON.stringify({"
            " counts: document.getElementById('counts').textContent,"
            " start: document.getElementById('start').textContent,"
            " hidden: [document.getElementById('record').hidden,"
            "          document.getElementById('record-under').hidden]}));"
        )

    NONE = {"rated": 4, "tried": 0, "won": 0, "top": 0, "highest": 0}

    def test_nothing_played_says_what_to_do_and_draws_no_nought(self) -> None:
        said = self._drawn(self.NONE, [
            self._game("alpha"), self._game("beta"),
            self._game("gamma", unbeaten=False), self._game("delta", retired=True),
        ])
        assert said["counts"] == ""
        assert said["start"].startswith(
            "Nothing here yet. Play a rated game and it starts here."
        )
        # The unbeaten games, as an offer: the beaten one and the retired one are not.
        assert "2 games nobody has beaten yet: alpha, beta." in said["start"]
        assert "gamma" not in said["start"] and "delta" not in said["start"]
        assert said["hidden"] == [True, True]

    def test_the_line_links_to_a_game_to_play(self) -> None:
        script = _script_of(page_text.inlined(site_pages.profile("ada")))
        # The games (Q723): `/` is the landing page now.
        assert "const START = '/game';" in script
        assert "play.href = START;" in script

    def test_with_nothing_unbeaten_the_line_stands_alone(self) -> None:
        said = self._drawn(self.NONE, [self._game("alpha", unbeaten=False)])
        assert said["start"] == "Nothing here yet. Play a rated game and it starts here."

    def test_a_long_list_names_a_few_and_counts_the_rest(self) -> None:
        games = [self._game(name) for name in ("aa", "bb", "cc", "dd", "ee", "ff")]
        said = self._drawn(self.NONE, games)
        assert "6 games nobody has beaten yet: aa, bb, cc, dd and 2 more." in said["start"]

    def test_a_finished_game_brings_the_counts_and_the_table_back(self) -> None:
        counts = {"rated": 4, "tried": 1, "won": 0, "top": 0, "highest": 0}
        said = self._drawn(counts, [self._game("alpha", played=1, lost=1)])
        assert said["start"] == ""
        assert "1games tried" in said["counts"] and "0games won" in said["counts"]
        assert said["hidden"] == [False, False]

    def test_nothing_on_it_counts_down_or_nags(self) -> None:
        said = self._drawn(self.NONE, [self._game("alpha")])["start"].lower()
        for word in ("streak", "days left", "hurry", "don't miss", "before it"):
            assert word not in said, word


def _badge(
    kind: str,
    when: str = "2026-10-03T12:00:00+00:00",
    tier: int | None = None,
    game: str | None = None,
    name_of: str = "",
    **evidence: Any,
) -> dict[str, Any]:
    """One held badge, as `/api/profile` and the game's answer send it.

    Built through `badges.line`, so a test asserts against what a page is really given
    rather than against a shape written out twice.
    """
    award = awards.Award(
        kind=kind, tier=tier, holder=1, game=game,
        earned=datetime.fromisoformat(when), evidence=evidence,
    )
    return {**badges.line(award), "name_of": name_of}


class TestTheProfileDrawsBadges:
    """Box E4 of `docs/archive/plan-meta-game.md`. A held badge is a dated line among the
    milestones, with its own rule under it and the figures the row kept on the day.

    Nothing here is a total. Only badges somebody holds arrive, and a page that said
    *4 of 17* would be counting them, which §2.1's first rule refuses.
    """

    MILESTONE = [
        {"kind": "both-seats", "rule": "A win from each seat of this game.",
         "when": "2026-09-14T12:00:00+00:00", "game": "aaa", "name": "alpha", "seat": 1},
    ]

    def test_a_badge_is_a_dated_sentence_among_the_milestones(self) -> None:
        held = [_badge(badges.LONG_WAY_ROUND, when="2026-09-15T09:00:00+00:00")]
        said = _done(self.MILESTONE, held)
        assert said.startswith("What you have done")
        assert "The long way round, 2026-09-15." in said
        assert badges.BADGES[badges.LONG_WAY_ROUND].sentence in said
        assert "Both sides, on alpha" in said

    def test_the_list_is_newest_first_whichever_kind_of_record_it_is(self) -> None:
        held = [_badge(badges.LONG_WAY_ROUND, when="2026-09-13T09:00:00+00:00")]
        said = _done(self.MILESTONE, held)
        assert said.index("Both sides") < said.index("The long way round")
        later = _done(
            self.MILESTONE,
            [_badge(badges.LONG_WAY_ROUND, when="2026-09-16T09:00:00+00:00")],
        )
        assert later.index("The long way round") < later.index("Both sides")

    def test_a_badge_carries_its_own_rule(self) -> None:
        """§3's seventh test, the same answer the milestones give."""
        said = _done([], [_badge(badges.GIANT_KILLER, game="aaa", name_of="alpha")])
        assert badges.BADGES[badges.GIANT_KILLER].rule in said

    def test_a_series_line_names_its_step(self) -> None:
        said = _done([], [_badge(badges.BEAT[0], tier=5)])
        assert "5 different games" in said

    def test_a_badge_on_a_game_names_the_game_and_one_across_the_library_does_not(
        self,
    ) -> None:
        on_a_game = _done([], [_badge(badges.GIANT_KILLER, game="aaa", name_of="alpha")])
        assert "Giant-killer, on alpha, 2026-10-03." in on_a_game
        across = _done([], [_badge(badges.GRAND_TOUR)])
        assert "The grand tour, 2026-10-03." in across

    def test_a_score_badge_quotes_the_day_its_range_and_its_weights(self) -> None:
        """§2.9's dated quotation. The figures are the row's, so nothing since moves
        them, and they are compared with no live score."""
        said = _done([], [_badge(
            badges.CLANKER, game="aaa", name_of="alpha",
            mean=4.4, voters=9, people=12,
            score={"headline": 0.52, "ci_low": 0.47, "ci_high": 0.58, "weights": "v9",
                   "basis": {"seeds": [0, 1000, 2000, 3000]}},
        )])
        assert "Proved the clanker wrong, on alpha, 2026-10-03." in said
        assert (
            "On that day, a score of 0.52 (likely 0.47 to 0.58; the average of four runs,"
            " scoring rules version 9), 4.4 stars from 9 people and 12 people who had"
            " played it."
        ) in said

    def test_a_score_badge_with_no_run_count_quotes_no_score(self) -> None:
        """Q339. A kept basis that names no seeds cannot say which claim its number
        was, so the line quotes the stars and the players and no index at all."""
        said = _done([], [_badge(
            badges.CLANKER, game="aaa", name_of="alpha",
            mean=4.4, voters=9, people=12,
            score={"headline": 0.52, "ci_low": 0.47, "ci_high": 0.58, "weights": "v9"},
        )])
        assert "an index score of" not in said and "0.52" not in said
        assert (
            "On that day, 4.4 stars from 9 people and 12 people who had played it."
        ) in said

    def test_a_profile_with_no_badge_reads_as_it_did_without_them(self) -> None:
        """A person with none sees no heading, no empty slot and no greyed list."""
        assert _done(self.MILESTONE, []) == _done(self.MILESTONE)
        assert _done([], []) == ""

    def test_no_page_counts_badges(self) -> None:
        """§3, test 5 and §2.1's first rule. A count would turn a record into a score."""
        held = [_badge(badges.LONG_WAY_ROUND), _badge(badges.GRAND_TOUR)]
        said = _done([], held)
        for counted in (" of 17", "2 badges", "badges earned"):
            assert counted not in said

    def test_the_way_to_the_list_of_rules_is_on_the_page_whatever_you_hold(self) -> None:
        """The link is markup rather than something the script draws, so a person with
        no badge is not a person with no way to the rules (§3, test 7)."""
        page = page_text.inlined(site_pages.profile("ada"))
        assert 'href="/badges"' in page
        assert "drawDone" in page


class TestTheListOfEveryBadge:
    """§2.9's one plain list, which is what makes test 7 pass for a badge nobody holds.

    It reads the same for everybody. Nothing on it is marked earned, nothing shows how
    close anybody is, and it opens no connection -- `routes.PERSONAL` leaves it out, so
    the page cannot know who is asking even if it wanted to.
    """

    def test_every_badge_is_on_it_with_the_sentence_from_the_table(self) -> None:
        page = site_pages.badges()
        for badge in badges.BADGES.values():
            assert html.escape(badge.name) in page, badge.kind
            assert html.escape(badge.rule) in page, badge.kind

    def test_a_series_says_its_steps_in_the_unit_it_counts_in(self) -> None:
        page = site_pages.badges()
        assert "In steps, at 1, 5, 10 and 25." in page
        assert "In steps, at 50%, 75% and 100%." in page

    def test_it_holds_one_entry_per_badge_and_nothing_else(self) -> None:
        """One line each, whoever is reading. A greyed row, a tick or a *not yet* would
        be the list telling somebody where they stand, which is what it is not for."""
        page = site_pages.badges()
        found = re.search(r'<ul class="lines">(.*?)</ul>', page, re.S)
        assert found is not None
        assert found.group(1).count("<li>") == len(badges.BADGES)
        for mark in ("not yet", "✓", "you hold", "so far", "0 of", "locked"):
            assert mark not in page.lower()

    def test_nothing_on_it_is_a_script_or_a_name(self) -> None:
        """A constant page with no script cannot come to know who is reading it."""
        page = site_pages.badges()
        assert "<script>" not in page
        assert "/api/" not in page

    def test_it_reads_the_same_for_everybody(self) -> None:
        """A constant, and the page asks nobody's name to build it."""
        assert site_pages.badges() == site_pages.badges()
        assert "/badges" not in routes.PERSONAL
        assert "/badges" in routes.PAGES

    def test_it_is_served_without_reading_the_cookie(
        self, app: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """As credits and the notice are. The lookup is made to fail and the page is
        asked for anyway, which is the only way to prove nothing reached for it."""

        def reached(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("the cookie was read to serve the list of badges")

        monkeypatch.setattr(app, "identify", reached)
        server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
        try:
            answer = Browser(server.server_port).get("/badges")
        finally:
            server.shutdown()
            server.server_close()
        assert answer.status == 200
        assert "<h1>The badges</h1>" in answer.text


class TestTheProfileRouteCarriesTheMetaGame:
    """A2's payload over a real store, which is the only place its queries are proved.

    The handler reads every leaderboard row on the site rather than the viewer's,
    because *the first game played here* and the *Unbeaten* tag are decided by other
    people's games, and it reads each rated game's grid cell out of `games.spec`. Rows
    built by hand say nothing about either, so these run against PostgreSQL.
    """

    WEAK = "random"
    TOP = "minimax:nodes=6400"
    WHEN = datetime(2026, 9, 14, 12)

    def _game(self, where: Site, ruleset: str) -> str:
        """One rated game in the store: its spec, its name and a two-rung ladder."""
        spec = json.loads((RULESETS / f"{ruleset}.json").read_text(encoding="utf-8"))
        digest = spec_hash(spec)
        with where.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, %s::jsonb, %s, 'corpus')",
                (digest, json.dumps(spec), spec["name"]),
            )
        rating.store(
            where.store,
            digest,
            [
                rating.Player(
                    key=rating.computer(agent), rating=score, deviation=30.0,
                    volatility=0.0, matches=40,
                )
                for agent, score in ((self.WEAK, 1200.0), (self.TOP, 2300.0))
            ],
        )
        where.store.commit()
        return digest

    def _win(
        self, where: Site, game: str, user: int, agent: str, *, seat: int, minute: int
    ) -> None:
        """A finished leaderboard win for that person, from that seat."""
        when = self.WHEN + timedelta(minutes=minute)
        columns = (
            ("seat0_user", "seat1_agent", user, agent, 1.0)
            if seat == 0
            else ("seat0_agent", "seat1_user", agent, user, 0.0)
        )
        with where.store.cursor() as cursor:
            cursor.execute(
                f"insert into matches (id, game, mode, seed, {columns[0]}, {columns[1]},"
                " moves, result, reason, top, started, finished)"
                " values (%s, %s, %s, 0, %s, %s, '[]'::jsonb, %s, 'checkmate', false,"
                " %s, %s)",
                (f"m{secrets.token_hex(4)}", game, match_rows.LEADERBOARD,
                 columns[2], columns[3], columns[4], when, when),
            )
        where.store.commit()

    def _rate(self, where: Site, game: str, user: int, score: float) -> None:
        rating.store(where.store, game, [_rated(user, score)])
        where.store.commit()

    def _me(self, where: Site, handle: str) -> int:
        with where.store.cursor() as cursor:
            cursor.execute("select id from users where handle = %s", (handle,))
            return int(cursor.fetchone()[0])

    def test_the_payload_carries_both_columns_the_lines_and_what_is_open(
        self, site: Site
    ) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        played = self._game(site, "chess")
        untried = self._game(site, "dobutsu")
        self._win(site, played, ada, self.WEAK, seat=0, minute=0)
        self._win(site, played, ada, self.WEAK, seat=1, minute=1)
        self._rate(site, played, ada, 2100.0)

        payload = browser.get("/api/profile").payload
        rows = {game["hash"]: game for game in payload["games"]}
        assert set(rows) == {played, untried}

        # M4's column counts the rungs beaten of the rungs there are, and T3's is the
        # gap to the top. An untried game still has rungs to be *none of*, and a game
        # with no rating is absent rather than a gap of nought.
        assert rows[played]["rungs"] == {"beaten": 1, "of": 2}
        assert rows[untried]["rungs"] == {"beaten": 0, "of": 2}
        # F8: the person's own rating travels with the gap, so the page shows the number
        # first and works nothing out.
        assert rows[played]["distance"] == {
            "points": 200.0, "rating": 2100.0, "established": True
        }
        assert rows[untried]["distance"] is None
        # S1: the strongest beaten by its name, never its agent string.
        assert rows[played]["rung"]["agent"] == self.WEAK
        assert rows[played]["rung"]["label"] == "random"

        # A win from each seat is M3, and each line carries the rule it was read by.
        kinds = {line["kind"]: line for line in payload["milestones"]}
        assert milestones.BOTH_SEATS in kinds
        line = kinds[milestones.BOTH_SEATS]
        assert line["game"] == played
        assert line["name"] and line["rule"] == milestones.RULES[milestones.BOTH_SEATS]

        # The untried game is what is still open, under the axes its own spec puts it
        # on. The played game is not, and no line names a game the payload cannot name.
        opened = {
            hash_
            for axis in payload["stillOpen"]
            for kind in axis["kinds"]
            for hash_ in kind["untried"]
        }
        assert opened == {untried}
        assert payload["stillOpen"], "a person with an untried game has something open"
        assert {axis["axis"] for axis in payload["stillOpen"]} <= {
            axis.name for axis in grid.AXES
        }
        # S6: each axis carries its question and each kind its words, the game page's
        # own kind cards', so nothing reads as a grid key.
        for axis in payload["stillOpen"]:
            assert axis["asks"][0].isupper() and axis["asks"] != axis["axis"]
            for kind in axis["kinds"]:
                assert kind["words"] and " " in kind["words"], kind

    def test_the_game_page_says_how_far_each_computer_player_sits_from_you(
        self, site: Site
    ) -> None:
        """`docs/reference/plain-words.md` §3.3 (the audit's P6): signed in and rated on a game,
        the picker says *about 200 above you*, and the subtraction is the server's. Nobody
        rated there, or nobody signed in, gets no distances at all."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        assert "yourGaps" not in browser.get(f"/api/game/{game}").payload["standing"]
        self._rate(site, game, ada, 2100.0)
        standing = browser.get(f"/api/game/{game}").payload["standing"]
        assert standing["yourGaps"] == {self.WEAK: -900.0, self.TOP: 200.0}
        assert standing["siteRatings"] == {self.WEAK: 1200.0, self.TOP: 2300.0}
        visitor = site.browser().get(f"/api/game/{game}").payload["standing"]
        assert "yourGaps" not in visitor

    def test_another_persons_game_is_read_without_being_listed_as_yours(
        self, site: Site
    ) -> None:
        """The handler passes every row on the site, so *the first game played here*
        can be somebody else's. Ada's own table still holds only Ada's record."""
        browser = _joined(site, "ada")
        _joined(site, "bob")
        ada, bob = self._me(site, "ada"), self._me(site, "bob")
        game = self._game(site, "chess")
        self._win(site, game, bob, self.WEAK, seat=0, minute=0)
        self._win(site, game, ada, self.WEAK, seat=0, minute=1)

        payload = browser.get("/api/profile").payload
        row = next(game_ for game_ in payload["games"] if game_["hash"] == game)
        assert row["played"] == 1, "bob's match is read, and it is not on Ada's row"
        kinds = {line["kind"] for line in payload["milestones"]}
        assert milestones.FIRST_HERE not in kinds, "bob got here first"

    # ------------------------------------------------------------- E4's own payload

    def _hold(self, where: Site, award: awards.Award) -> None:
        awards.grant(where.store, [award])
        where.store.commit()

    def test_a_held_badge_arrives_with_its_sentence_its_rule_and_its_figures(
        self, site: Site
    ) -> None:
        """Box E4 (`docs/archive/plan-meta-game.md`). The page works nothing out, so what a
        line says is settled here."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        self._hold(site, awards.Award(
            kind=badges.CLANKER, holder=ada, game=game, earned=self.WHEN,
            evidence={"mean": 4.4, "voters": 9, "people": 12,
                      "score": {"headline": 0.52, "ci_low": 0.47, "ci_high": 0.58,
                                "weights": "v9", "basis": {"seeds": [0]}}},
        ))

        held = browser.get("/api/profile").payload["badges"]
        assert len(held) == 1
        line = held[0]
        assert line["kind"] == badges.CLANKER
        assert line["name"] == badges.BADGES[badges.CLANKER].name
        assert line["rule"] == badges.BADGES[badges.CLANKER].rule
        # `awards.earned` is a `timestamptz`, so it comes back with the session's zone
        # on it. The page reads the day off the front, which is what is asserted.
        assert str(line["when"]).startswith("2026-09-14")
        assert line["game"] == game and line["name_of"]
        assert line["quoted"] == (
            "a score of 0.52 (likely 0.47 to 0.58; one run, scoring rules version 9), "
            "4.4 stars from 9 people and 12 people who had played it"
        )

    def test_a_badge_on_a_game_no_ladder_holds_is_still_named(self, site: Site) -> None:
        """The names are asked for the awards' games as well as the rated ones, so a
        badge on a game with no rating table reads as a name rather than a hash."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        with site.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values ('unrated0000', '{}'::jsonb, 'Nine holes', 'designed')"
            )
        self._hold(site, awards.Award(
            kind=badges.DESCENDANT, holder=ada, game="unrated0000", earned=self.WHEN,
        ))
        held = browser.get("/api/profile").payload["badges"]
        assert [line["name_of"] for line in held] == ["Nine holes"]

    def test_a_person_with_no_badge_carries_an_empty_list(self, site: Site) -> None:
        """Absent rather than a nought (§2.1's third rule), and no count anywhere."""
        browser = _joined(site, "ada")
        payload = browser.get("/api/profile").payload
        assert payload["badges"] == []
        assert "badgeCount" not in payload and "badges" not in payload["counts"]

    def test_somebody_elses_badge_is_not_on_this_profile(self, site: Site) -> None:
        browser = _joined(site, "ada")
        _joined(site, "bob")
        game = self._game(site, "chess")
        self._hold(site, awards.Award(
            kind=badges.LONG_WAY_ROUND, holder=self._me(site, "bob"), game=game,
        ))
        assert browser.get("/api/profile").payload["badges"] == []

    def test_a_games_own_badges_arrive_with_its_page(self, site: Site) -> None:
        """The other half of E4. A library game's badge has no holder at all, so the
        game's answer reads by game and never by who is looking."""
        browser = _joined(site, "ada")
        game = self._game(site, "chess")
        self._hold(site, awards.Award(
            kind=badges.CROWD_PLEASER, game=game, earned=self.WHEN,
            evidence={"mean": 4.4, "voters": 9, "people": 12},
        ))
        standing = browser.get(f"/api/game/{game}").payload["standing"]
        assert [line["kind"] for line in standing["badges"]] == [badges.CROWD_PLEASER]
        assert standing["badges"][0]["quoted"] == (
            "4.4 stars from 9 people and 12 people who had played it"
        )
        assert standing["badges"][0]["rule"] == badges.BADGES[badges.CROWD_PLEASER].rule

    def test_a_players_own_badge_on_the_game_is_not_the_games(self, site: Site) -> None:
        """N2 and N3 are filed under the game with the player as holder, as M2 and M3
        are. They are that person's line on their own profile: the sentence is written
        to whoever is reading, and one per player would be a count of them (§3, test 6).
        """
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        self._hold(site, awards.Award(
            kind=badges.GIANT_KILLER, holder=ada, game=game, earned=self.WHEN,
        ))
        assert browser.get(f"/api/game/{game}").payload["standing"]["badges"] == []
        assert [line["kind"] for line in browser.get("/api/profile").payload["badges"]] \
            == [badges.GIANT_KILLER]

    def test_a_designed_games_own_badge_sits_on_its_owner_and_is_still_the_games(
        self, site: Site
    ) -> None:
        """*Descendant* and *Family tree* name the owner as holder, and both are about
        the game. `origin` is what tells a designed game from a library one, because the
        loader gives every library game the maintainer's account."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        mine = self._game(site, "chess")
        with site.store.cursor() as cursor:
            cursor.execute(
                "update games set origin = 'designed', owner = %s, visibility = 'public'"
                " where hash = %s",
                (ada, mine),
            )
        self._hold(site, awards.Award(
            kind=badges.DESCENDANT, holder=ada, game=mine, earned=self.WHEN,
        ))
        standing = browser.get(f"/api/game/{mine}").payload["standing"]
        assert [line["kind"] for line in standing["badges"]] == [badges.DESCENDANT]

    def test_a_game_with_no_badge_carries_an_empty_list(self, site: Site) -> None:
        browser = _joined(site, "ada")
        game = self._game(site, "chess")
        assert browser.get(f"/api/game/{game}").payload["standing"]["badges"] == []

    def test_a_badge_on_another_game_is_not_on_this_ones_page(self, site: Site) -> None:
        browser = _joined(site, "ada")
        game, other = self._game(site, "chess"), self._game(site, "dobutsu")
        self._hold(site, awards.Award(kind=badges.CROWD_PLEASER, game=other))
        assert browser.get(f"/api/game/{game}").payload["standing"]["badges"] == []


# --------------------------------------------- A3, the overall page at `/players`


def _drawn(runner: str) -> Any:
    """Run the overall page's whole script over the small DOM and read back a section."""
    return _in_node(_DOM + _script_of(page_text.inlined(site_pages.players("ada"))), runner)


def _unclaimed(open_now: dict[str, Any]) -> Any:
    """What the page's opening section reads, lead and list together."""
    return _drawn(
        f"drawUnclaimed({json.dumps(open_now)});"
        "console.log(JSON.stringify(document.getElementById('open-lead').textContent"
        " + ' | ' + document.getElementById('unclaimed').textContent));"
    )


def _people(rows: list[dict[str, Any]], rated: int = 40, you: int | None = None) -> Any:
    """What the page's table reads, rows and the sentence under them."""
    return _drawn(
        (f"YOU = {json.dumps(you)};" if you is not None else "")
        + f"drawPeople({json.dumps(rows)}, {rated});"
        "console.log(JSON.stringify(document.getElementById('people').textContent"
        " + ' | ' + document.getElementById('people-lead').textContent));"
    )


def _game(name: str) -> dict[str, str]:
    return {"hash": name * 4, "name": name}


def _here(members: int, listed: bool = False) -> Any:
    """The people section below the threshold: its heading, its sentence, its link."""
    here = {"listed": listed, "members": members, "listedFrom": 5}
    return _drawn(
        f"drawPeople([], 40, {json.dumps(here)});"
        "const lead = document.getElementById('people-lead');"
        "console.log(JSON.stringify({"
        " head: document.getElementById('people-head').textContent,"
        " lead: lead.textContent,"
        " links: lead.children.filter(child => child.href).map(child => child.href),"
        " table: document.getElementById('people-table').className,"
        " rule: document.getElementById('people-rule').hidden}));"
    )


class TestThePlayersPageBelowFivePeople:
    """L3 of `docs/archive/plan-launch.md` §3.3. Reached by its address before five people have
    finished a rated game, the page says who is here and what puts a person on the list,
    and points at the unclaimed firsts it opens with. Nothing ranked is drawn."""

    def test_it_says_who_is_here_and_when_the_list_starts(self) -> None:
        said = _here(3)
        assert said["head"] == "Who is here"
        assert said["lead"].startswith("3 members are here.")
        assert "once they finish a rated game" in said["lead"]
        assert "The list starts when 5 people have." in said["lead"]
        assert said["links"] == ["#unclaimed"]
        assert said["table"] == "people empty"
        assert said["rule"] is True

    def test_one_member_reads_as_one(self) -> None:
        assert _here(1)["lead"].startswith("1 member is here.")

    def test_from_five_on_the_list_is_drawn(self) -> None:
        said = _drawn(
            "drawPeople([{user: 1, handle: 'ada', tried: 1, won: 1, top: 0}], 40,"
            " {listed: true, members: 5, listedFrom: 5});"
            "console.log(JSON.stringify({"
            " head: document.getElementById('people-head').textContent,"
            " rows: document.getElementById('people').textContent,"
            " table: document.getElementById('people-table').className,"
            " rule: document.getElementById('people-rule').hidden}));"
        )
        assert said == {"head": "Who has played", "rows": "ada110", "table": "people",
                        "rule": False}


class TestTheOverallPageOpensWithWhatIsUnclaimed:
    """The user's decision of 2026-09-14, option A. What leads the page is a set of
    games to go and take, not a table of people ahead of you (§3, test 5)."""

    def test_the_unclaimed_section_is_above_the_people(self) -> None:
        page = page_text.inlined(site_pages.players("ada"))
        assert page.index("Still unclaimed") < page.index("Who has played")
        assert page.index('id="unclaimed"') < page.index('id="people"')

    def test_a_line_says_how_many_names_a_few_and_carries_its_rule(self) -> None:
        said = _unclaimed({
            "unplayed": [_game(name) for name in ("aa", "bb", "cc", "dd", "ee", "ff", "gg")],
            "unbeaten": [],
            "offered": 12,
        })
        assert "Firsts nobody has taken yet, across the 12 games you can start today" in said
        assert "7 games nobody has played yet. aa, bb, cc, dd, ee, ff and 1 more." in said
        assert "Whoever finishes the first one is the first to have played it here." in said

    def test_both_lists_are_drawn_and_each_names_what_it_reads(self) -> None:
        """§3's seventh test, on the page rather than in a plan."""
        said = _unclaimed({
            "unplayed": [_game("aa")], "unbeaten": [_game("aa"), _game("bb")],
            "offered": 3,
        })
        assert "1 game nobody has played yet. aa." in said
        assert "2 games nobody has beaten the strongest computer player on. aa, bb." in said
        assert "One win, from either side, is enough." in said

    def test_nothing_unclaimed_is_a_sentence_and_never_two_noughts(self) -> None:
        """Absent is not zero (§2.1's third rule), so the section says what happened
        rather than drawing a pair of empty lists."""
        said = _unclaimed({"unplayed": [], "unbeaten": [], "offered": 4})
        assert said.endswith(" | ")
        assert ("Every game you can start has been played, and on every one somebody has "
                "beaten the strongest computer player") in said
        assert "0 games" not in said

    def test_nothing_on_the_page_counts_down_or_runs_out(self) -> None:
        """§3's first and second tests. The lists are scarce because there is one
        library; nothing about them is timed."""
        said = _unclaimed({"unplayed": [_game("aa")], "unbeaten": [], "offered": 2})
        assert "Nothing here expires and nobody loses one by being away." in said
        page = page_text.inlined(site_pages.players("ada")).lower()
        for word in ("deadline", "expires in", "days left", "hurry", "last chance"):
            assert word not in page, word


class TestTheOverallPageListsPeopleAndNotPositions:
    def test_a_row_is_a_handle_and_three_counts(self) -> None:
        said = _people([{"user": 1, "handle": "ada", "tried": 7, "won": 4, "top": 2}])
        assert said.startswith("ada742")
        assert "Of 40 rated games: how many each person has finished a game on" in said

    def test_no_position_is_drawn_beside_anybody(self) -> None:
        """A standing, not a league table. §2.1's fifth rule refuses a rank number and
        §3's fifth test refuses anything that reads as one."""
        rows = [
            {"user": 1, "handle": "ada", "tried": 7, "won": 4, "top": 2},
            {"user": 2, "handle": "bob", "tried": 3, "won": 1, "top": 0},
        ]
        said = _people(rows).split(" | ")[0]
        assert said == "ada742bob310", "a handle and three counts, and nothing beside them"

    def test_your_own_row_is_marked_and_nobody_elses(self) -> None:
        rows = [
            {"user": 1, "handle": "ada", "tried": 7, "won": 4, "top": 2},
            {"user": 2, "handle": "bob", "tried": 3, "won": 1, "top": 0},
        ]
        said = _people(rows, you=2)
        assert "bob you" in said and "ada you" not in said

    def test_an_empty_site_says_so_instead_of_drawing_an_empty_table(self) -> None:
        """A heading row over no rows is a list of noughts nobody is on, so the table
        itself goes and the sentence stays."""
        said = _people([])
        assert said.startswith(" | Nobody has finished a rated game yet.")
        assert "The first person to finish one is the first name here." in said
        assert _drawn(
            "drawPeople([], 4);"
            "console.log(JSON.stringify(document.getElementById('people-table')"
            ".className));"
        ) == "people empty"
        assert "table.people.empty { display: none; }" in site_pages.STYLE

    def test_nothing_on_the_page_is_written_as_markup(self) -> None:
        assert "innerHTML" not in page_text.inlined(site_pages.players("ada"))

    def test_the_page_names_what_each_column_reads(self) -> None:
        """§3's seventh test again, for the table. The sentence is on the page whether
        or not anybody is on the list."""
        page = page_text.inlined(site_pages.players("ada"))
        assert "Nobody here is given a place, and nothing here counts matches." in page
        assert ("The list is ordered by the games where each person beat the strongest "
                "computer player, then games won, then games tried.") in page
        assert "Somebody who has not finished a game is on no list" in page

    def test_it_fits_a_phone_without_hiding_a_column(self) -> None:
        """Four columns, so nothing has to become a line under the name the way the
        profile's eight do."""
        page = page_text.inlined(site_pages.players("ada"))
        assert '<table class="people" id="people-table">' in page
        assert len(re.findall(r"<th[ >]", page)) == 4
        assert "narrow(" not in _script_of(page)


class TestTheOverallPageNeedsAnAccount:
    """The same rule a profile follows. The page gathers what people have done into one
    place, and the site is invite-only."""

    def test_the_route_refuses_a_visitor_with_a_sentence(self, app: SiteApp) -> None:
        app.viewer = None
        with pytest.raises(site_server.SignInFirst) as raised:
            _ask(app, "GET", "/api/players")
        assert "Sign in to see the players" in str(raised.value)

    def test_a_visitor_asking_for_the_page_is_sent_to_the_door(self, site: Site) -> None:
        answer = site.browser().get("/players")
        assert answer.status == 303
        assert answer.headers["Location"] == "/signin"

    def test_a_signed_in_person_gets_the_page(self, site: Site) -> None:
        answer = _joined(site, "ada").get("/players")
        assert answer.status == 200
        assert "<h1>Players</h1>" in answer.text

    def test_a_handle_with_a_script_tag_in_it_is_escaped_in_the_corner(self) -> None:
        raw = "<script>alert('q251')</script>"
        page = page_text.inlined(site_pages.players(raw))
        assert raw not in page and html.escape(raw) in page


class TestThePlayersRouteOverAStore:
    """A3's payload, where its queries are proved. Which games can still be started is
    the store's answer, and so is who has left."""

    WEAK = "random"
    TOP = "minimax:nodes=6400"
    WHEN = datetime(2026, 9, 14, 12)
    LISTED_FROM = milestones.LISTED_FROM

    @pytest.fixture(autouse=True)
    def _listed_from_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The list's own readings are proved with one person; L3's threshold has its
        own tests below, at the real constant."""
        monkeypatch.setattr(milestones, "LISTED_FROM", 1)

    def _game(self, where: Site, ruleset: str) -> str:
        spec = json.loads((RULESETS / f"{ruleset}.json").read_text(encoding="utf-8"))
        digest = spec_hash(spec)
        with where.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, %s::jsonb, %s, 'corpus')",
                (digest, json.dumps(spec), spec["name"]),
            )
        rating.store(
            where.store,
            digest,
            [
                rating.Player(
                    key=rating.computer(agent), rating=score, deviation=30.0,
                    volatility=0.0, matches=40,
                )
                for agent, score in ((self.WEAK, 1200.0), (self.TOP, 2300.0))
            ],
        )
        where.store.commit()
        return digest

    def _win(
        self, where: Site, game: str, user: int, agent: str, *, minute: int,
        result: float = 1.0,
    ) -> None:
        when = self.WHEN + timedelta(minutes=minute)
        with where.store.cursor() as cursor:
            cursor.execute(
                "insert into matches (id, game, mode, seed, seat0_user, seat1_agent,"
                " moves, result, reason, top, started, finished)"
                " values (%s, %s, %s, 0, %s, %s, '[]'::jsonb, %s, 'checkmate', %s,"
                " %s, %s)",
                (f"m{secrets.token_hex(4)}", game, match_rows.LEADERBOARD, user, agent,
                 result, agent == self.TOP, when, when),
            )
        where.store.commit()

    def _me(self, where: Site, handle: str) -> int:
        with where.store.cursor() as cursor:
            cursor.execute("select id from users where handle = %s", (handle,))
            return int(cursor.fetchone()[0])

    def test_an_empty_site_lists_nobody_and_leaves_every_game_unclaimed(
        self, site: Site
    ) -> None:
        """A newcomer's first sight of the page: two lists of games and no table."""
        browser = _joined(site, "ada")
        played = self._game(site, "chess")
        other = self._game(site, "dobutsu")

        payload = browser.get("/api/players").payload
        assert payload["players"] == []
        assert payload["rated"] == 2
        open_now = payload["unclaimed"]
        assert open_now["offered"] == 2
        assert {game["hash"] for game in open_now["unplayed"]} == {played, other}
        assert {game["hash"] for game in open_now["unbeaten"]} == {played, other}
        assert all(game["name"] for game in open_now["unplayed"])

    def test_a_finished_game_puts_its_player_on_the_list_and_takes_the_game_off(
        self, site: Site
    ) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        played = self._game(site, "chess")
        self._game(site, "dobutsu")
        self._win(site, played, ada, self.WEAK, minute=0)

        payload = browser.get("/api/players").payload
        assert payload["players"] == [
            {"user": ada, "handle": "ada", "mark": marks.mark(ada), "page": "/players/ada",
             "tried": 1, "won": 1, "top": 0}
        ]
        assert [game["hash"] for game in payload["unclaimed"]["unplayed"]] != [played]
        assert played in {game["hash"] for game in payload["unclaimed"]["unbeaten"]}

    def test_beating_the_top_takes_the_game_off_both_lists(self, site: Site) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        played = self._game(site, "chess")
        self._win(site, played, ada, self.TOP, minute=0)

        payload = browser.get("/api/players").payload
        assert payload["unclaimed"] == {"unplayed": [], "unbeaten": [], "offered": 1}
        assert payload["players"][0]["top"] == 1

    def test_the_strongest_record_comes_first_and_nobody_idle_is_listed(
        self, site: Site
    ) -> None:
        browser = _joined(site, "ada")
        _joined(site, "bob")
        _joined(site, "cat")
        ada, bob = self._me(site, "ada"), self._me(site, "bob")
        game = self._game(site, "chess")
        self._win(site, game, ada, self.WEAK, minute=0)
        self._win(site, game, bob, self.TOP, minute=1)

        payload = browser.get("/api/players").payload
        assert [row["handle"] for row in payload["players"]] == ["bob", "ada"]
        assert "cat" not in {row["handle"] for row in payload["players"]}

    @pytest.mark.parametrize(
        "taken_off",
        [
            "revoked = now()",
            "retired = now()",
            # A deleted row holds no spec at all, which the store's own constraint says.
            "spec = null, name = 'a deleted game', deleted = now()",
        ],
    )
    def test_a_game_nobody_can_start_is_on_neither_unclaimed_list(
        self, site: Site, taken_off: str
    ) -> None:
        """The same three states the library leaves out. A game taken off the site is
        not a first anybody can go and take."""
        browser = _joined(site, "ada")
        game = self._game(site, "chess")
        with site.store.cursor() as cursor:
            cursor.execute(
                f"update games set {taken_off} where hash = %s", (game,)
            )
        site.store.commit()

        payload = browser.get("/api/players").payload
        assert payload["unclaimed"] == {"unplayed": [], "unbeaten": [], "offered": 0}
        assert payload["rated"] == 1, "the counts still read every rated game"

    def test_somebody_who_left_is_never_named(self, site: Site) -> None:
        """The privacy notice's promise: the games stay and the name goes."""
        browser = _joined(site, "ada")
        _joined(site, "bob")
        bob = self._me(site, "bob")
        game = self._game(site, "chess")
        self._win(site, game, bob, self.WEAK, minute=0)
        admin.forget_id(site.store, bob)
        site.store.commit()

        payload = browser.get("/api/players").payload
        assert [row["handle"] for row in payload["players"]] == [identity.FORMER]
        assert "bob" not in json.dumps(payload)

    def test_the_counts_are_the_ones_the_profile_shows(self, site: Site) -> None:
        """One reader behind both pages, so a person cannot read two ways."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        played = self._game(site, "chess")
        self._game(site, "dobutsu")
        self._win(site, played, ada, self.WEAK, minute=0)
        self._win(site, played, ada, self.WEAK, minute=1, result=0.0)

        mine = browser.get("/api/players").payload["players"][0]
        counts = browser.get("/api/profile").payload["counts"]
        assert (mine["tried"], mine["won"], mine["top"]) == (
            counts["tried"], counts["won"], counts["top"]
        )
        assert mine["tried"] == 1, "two matches on one game is one game tried"

    def _crowd(self, where: Site, handles: tuple[str, ...]) -> Browser:
        browser = _joined(where, handles[0])
        for handle in handles[1:]:
            _joined(where, handle)
        return browser

    def test_below_five_people_nothing_ranked_is_sent_or_linked(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L3: four people with a finished game and one without. The page is told who
        is here and when the list starts, and is sent no rows; no footer links it."""
        monkeypatch.setattr(milestones, "LISTED_FROM", self.LISTED_FROM)
        handles = ("ada", "bob", "cat", "dan", "eve")
        browser = self._crowd(site, handles)
        game = self._game(site, "chess")
        for minute, handle in enumerate(handles[: self.LISTED_FROM - 1]):
            self._win(site, game, self._me(site, handle), self.WEAK, minute=minute)

        payload = browser.get("/api/players").payload
        assert payload["players"] == []
        assert payload["listed"] is False
        assert payload["listedFrom"] == self.LISTED_FROM == 5
        assert payload["members"] == 5
        assert payload["unclaimed"]["offered"] == 1
        assert browser.get("/api/me").payload["players"] is False
        # The footer does not link it; the header's *People* (Q760) is the way to find a
        # member, and the page then sends no rows of counts.
        assert 'href="/players">players' not in browser.get("/profile").text

    def test_the_fifth_person_starts_the_list_and_the_link(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(milestones, "LISTED_FROM", self.LISTED_FROM)
        handles = ("ada", "bob", "cat", "dan", "eve")
        browser = self._crowd(site, handles)
        game = self._game(site, "chess")
        for minute, handle in enumerate(handles):
            self._win(site, game, self._me(site, handle), self.WEAK, minute=minute)

        payload = browser.get("/api/players").payload
        assert payload["listed"] is True
        assert sorted(row["handle"] for row in payload["players"]) == sorted(handles)
        assert browser.get("/api/me").payload["players"] is True
        assert 'href="/players">players' in browser.get("/profile").text
        assert 'href="/players">players' in browser.get("/suggestions").text

    def test_members_are_the_people_still_here(self, site: Site) -> None:
        """*‹n› members*: an account somebody shut out, or one that left, is not here."""
        browser = self._crowd(site, ("ada", "bob", "cat"))
        assert browser.get("/api/players").payload["members"] == 3
        with site.store.cursor() as cursor:
            cursor.execute("update users set revoked = now() where handle = 'bob'")
        site.store.commit()
        admin.forget_id(site.store, self._me(site, "cat"))
        site.store.commit()
        assert browser.get("/api/players").payload["members"] == 1


# ------------------------------------------------- B2: a game you left running


def _carry(entries: list[dict[str, Any]]) -> Any:
    """What the profile's *games you left running* section draws, and where it links.

    The links are read back as well as the words, because the whole point of the section
    is the address: an entry the store will not hand the rules back for draws no link at
    all, and nothing but the href tells the two apart.
    """
    return _rendered(
        f"drawCarryOn({json.dumps(entries)});"
        "const box = document.getElementById('carry');"
        "const links = [];"
        "(function walk(node) {"
        "  if (node.href !== undefined) links.push({href: node.href,"
        "    text: node.textContent});"
        "  (node.children || []).forEach(walk);"
        "})(box);"
        "console.log(JSON.stringify({said: box.textContent, links}));"
    )


class TestTheProfileOffersAnOpenGameBack:
    """`docs/archive/plan-ux.md` B2, the profile's half.

    A rated game somebody walked away from stayed open in the store and was named on no
    page at all, so the only way back into it was a URL nobody had kept. The section
    above the table is that way back, and it is the one part of this page about something
    unfinished.
    """

    ENTRY = {
        "id": "sessionid01", "game": "aaa", "name": "alpha",
        "opponent": "minimax:nodes=1600", "label": "medium", "seat": 0, "plies": 12,
        "started": "2026-09-17T09:00:00+00:00", "moved": "2026-09-18T11:30:00+00:00",
        "openable": True,
    }

    def test_the_line_links_to_the_board_and_says_what_game_it_is(self) -> None:
        drawn = _carry([self.ENTRY])
        assert "Games you left running" in drawn["said"]
        assert drawn["links"] == [{"href": "/#/play/sessionid01", "text": "alpha"}]
        assert (
            "alpha — against medium, playing first, 12 turns played, "
            "last moved 2026-09-18."
        ) in drawn["said"]

    def test_an_opponent_the_ladder_does_not_name_reads_as_the_computer(
        self,
    ) -> None:
        """S8, P16. The server names a player the ladder has no row for *the computer*
        (`opponents.name_of`), and a line that arrives with no label says the same: an
        agent string is never a name."""
        said = _carry([{**self.ENTRY, "label": ""}])["said"]
        assert "against the computer" in said
        assert "minimax:" not in said

    def test_the_address_is_one_the_application_routes(self) -> None:
        """Against the router itself rather than against a string written twice."""
        from page_text import CORE

        assert "parts[0] === 'play' && parts[1]" in CORE
        assert page_text.text("/static/site/play-link.js").splitlines()[-1] == "const PLAY = '/#/play/';"

    def test_a_game_nobody_has_moved_in_says_so(self) -> None:
        said = _carry([{**self.ENTRY, "plies": 0, "moved": None}])["said"]
        assert "0 turns played, not moved yet." in said

    def test_the_seat_is_read_off_the_entry(self) -> None:
        said = _carry([{**self.ENTRY, "seat": 1}])["said"]
        assert "playing second" in said and "playing first" not in said

    def test_a_game_the_store_will_not_open_draws_no_link(self) -> None:
        """The rules are gone, so there is nothing to replay the moves through and no
        board can be drawn. The line still names the game, because the row is this
        person's and the idle sweep will resign it like any other."""
        drawn = _carry([{**self.ENTRY, "name": "deleted game", "openable": False}])
        assert drawn["links"] == []
        assert "deleted game — against medium" in drawn["said"]
        assert "These rules are gone, so the board cannot be drawn." in drawn["said"]

    def test_nothing_running_draws_no_heading(self) -> None:
        """Absent rather than an empty section, which would read as a nought."""
        assert _carry([])["said"] == ""

    def test_the_section_sits_above_the_table(self) -> None:
        page = page_text.inlined(site_pages.profile("ada"))
        assert page.index('id="carry"') < page.index('<table class="record" id="record">')
        assert "drawCarryOn(payload.carryOn || []);" in page

    def test_nothing_on_it_is_written_as_markup(self) -> None:
        """A game's name is whatever somebody called it, and it reaches the page as a
        text node like every other name here."""
        assert "innerHTML" not in page_text.inlined(site_pages.profile("ada"))


class TestTheModeMenuSaysWhatHappens:
    """B1's last sentence. The menu said *leaving resigns*, and leaving does not.

    Walking away leaves the row open. What ends it is the idle sweep or starting another
    rated game on the same game, and both are in the option's own words now. The number
    of days is `site/quota.py`'s, so the sentence is held against that rather than
    against itself.
    """

    def test_the_option_names_both_ways_an_unfinished_game_ends(self) -> None:
        from page_text import DETAIL

        assert "leaving resigns" not in DETAIL
        assert (
            "rated, no undo; an unfinished game is resigned after thirty idle days or "
            "when you start another"
        ) in DETAIL

    def test_thirty_is_the_limit_the_sweep_actually_runs_on(self) -> None:
        from page_text import DETAIL

        assert quota.IDLE_LIMIT.days == 30, "the page says thirty in words"
        assert "thirty idle days" in DETAIL

    def test_the_menu_is_clamped_to_the_column_it_sits_in(self) -> None:
        """A select is as wide as its widest option, and the rated one is now a
        sentence. Measured at 375px: the menu took 714px and the whole page scrolled
        sideways, which is what C7 got rid of."""
        from page_text import DETAIL

        assert '<select id="mode" style="max-width:100%">' in DETAIL


class TestWhatTheTwoPayloadsSayAboutAnOpenGame:
    """B2's server half, over a real store: which rows count as *carry on* and which not.

    The rows are written by `matches.start`, because what an open leaderboard row looks
    like -- the seat columns, the mode, the empty finish -- is that call's business and a
    row built by hand here would be a second opinion about it.
    """

    AGENT = "minimax:nodes=1600"

    def _game(self, where: Site, ruleset: str) -> str:
        """One rated game in the store, which is what leaderboard mode needs."""
        spec = json.loads((RULESETS / f"{ruleset}.json").read_text(encoding="utf-8"))
        digest = spec_hash(spec)
        with where.store.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin)"
                " values (%s, %s::jsonb, %s, 'corpus')",
                (digest, json.dumps(spec), spec["name"]),
            )
        rating.store(
            where.store,
            digest,
            [rating.Player(key=rating.computer(self.AGENT), rating=1800.0,
                           deviation=30.0, volatility=0.0, matches=40)],
        )
        where.store.commit()
        return digest

    def _me(self, where: Site, handle: str) -> int:
        with where.store.cursor() as cursor:
            cursor.execute("select id from users where handle = %s", (handle,))
            return int(cursor.fetchone()[0])

    def _start(self, where: Site, game: str, user: int, seats: list[str]) -> str:
        found = match_rows.start(
            where.store, game=game, mode=match_rows.LEADERBOARD, seats=seats, user=user
        )
        where.store.commit()
        return found.id

    def test_the_game_page_offers_this_persons_open_game_back(self, site: Site) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        started = self._start(site, game, ada, ["human", self.AGENT])

        open_ = browser.get(f"/api/game/{game}").payload["standing"]["yourOpen"]
        assert [entry["id"] for entry in open_] == [started]
        assert open_[0]["opponent"] == self.AGENT
        assert open_[0]["label"] == "medium", "the ladder's word for it"
        assert open_[0]["seat"] == 0 and open_[0]["plies"] == 0
        assert open_[0]["moved"] is None and open_[0]["started"]

    def test_the_seat_is_the_one_the_person_actually_sat_at(self, site: Site) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        self._start(site, game, ada, [self.AGENT, "human"])
        open_ = browser.get(f"/api/game/{game}").payload["standing"]["yourOpen"]
        assert open_[0]["seat"] == 1 and open_[0]["opponent"] == self.AGENT

    def test_a_finished_game_is_not_offered_back(self, site: Site) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        started = self._start(site, game, ada, ["human", self.AGENT])
        match_rows.resign(site.store, started, by=ada)
        site.store.commit()
        assert browser.get(f"/api/game/{game}").payload["standing"]["yourOpen"] == []

    def test_a_sandbox_game_is_not_offered_back(self, site: Site) -> None:
        """Nothing counts, nothing is resigned, and the row is read by no statistic."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        found = match_rows.start(
            site.store, game=game, mode=match_rows.SANDBOX,
            seats=["human", self.AGENT], user=ada,
        )
        site.store.commit()
        assert found.mode == match_rows.SANDBOX
        assert browser.get(f"/api/game/{game}").payload["standing"]["yourOpen"] == []

    def test_somebody_elses_open_game_is_not_offered_to_you(self, site: Site) -> None:
        browser = _joined(site, "ada")
        _joined(site, "bob")
        game = self._game(site, "chess")
        self._start(site, game, self._me(site, "bob"), ["human", self.AGENT])
        assert browser.get(f"/api/game/{game}").payload["standing"]["yourOpen"] == []

    def test_a_visitor_is_offered_nothing_and_asked_nothing(self, site: Site) -> None:
        """A rated game needs an account, so a visitor has none to have left."""
        game = self._game(site, "chess")
        standing = site.browser().get(f"/api/game/{game}").payload["standing"]
        assert "yourOpen" not in standing

    def test_the_profile_names_every_open_game_with_the_way_into_it(
        self, site: Site
    ) -> None:
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        one, two = self._game(site, "chess"), self._game(site, "dobutsu")
        first = self._start(site, one, ada, ["human", self.AGENT])
        second = self._start(site, two, ada, ["human", self.AGENT])

        carry = browser.get("/api/profile").payload["carryOn"]
        # Newest first, so the game you were last at is the first one offered back.
        assert [entry["id"] for entry in carry] == [second, first]
        assert {entry["game"] for entry in carry} == {one, two}
        assert all(entry["name"] for entry in carry), "a name rather than a hash"
        assert all(entry["openable"] for entry in carry)

    def test_starting_another_rated_game_here_resigns_the_one_you_left(
        self, site: Site
    ) -> None:
        """§4's rule, read off the payload: one open game per person per game, and the
        *carry on* entry is the one that is still running."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        self._start(site, game, ada, ["human", self.AGENT])
        second = self._start(site, game, ada, ["human", self.AGENT])
        carry = browser.get("/api/profile").payload["carryOn"]
        assert [entry["id"] for entry in carry] == [second]

    def test_a_deleted_games_open_row_stays_listed_and_is_not_openable(
        self, site: Site
    ) -> None:
        """Q303 left a row that can never draw a board: its rules are gone, so there is
        nothing to replay the moves through. It is still this person's unfinished game
        and the sweep will resign it, so the profile names it and offers no link."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        started = self._start(site, game, ada, ["human", self.AGENT])
        with site.store.cursor() as cursor:
            cursor.execute(
                "update games set retired = now() where hash = %s", (game,)
            )
        admin.delete_game(site.store, game, viewer=None, admin=True)
        site.store.commit()

        carry = browser.get("/api/profile").payload["carryOn"]
        assert [entry["id"] for entry in carry] == [started]
        assert carry[0]["openable"] is False
        assert carry[0]["name"] == admin.DELETED_NAME
        # And the page it would have linked to is a 404, which is why it draws no link.
        assert browser.get(f"/api/game/{game}").status == 404

    def test_a_retired_games_open_row_can_still_be_carried_on(self, site: Site) -> None:
        """A retired game keeps its rules in its row, so the position rebuilds and the
        game can be finished. Only starting a new one is refused."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        started = self._start(site, game, ada, ["human", self.AGENT])
        with site.store.cursor() as cursor:
            cursor.execute("update games set retired = now() where hash = %s", (game,))
        site.store.commit()

        carry = browser.get("/api/profile").payload["carryOn"]
        assert [entry["openable"] for entry in carry] == [True]
        # And the route the *carry on* link goes to answers with the position.
        found = browser.get(f"/api/session/{started}")
        assert found.status == 200 and found.payload["id"] == started

    def test_the_session_route_hands_back_the_position_it_left(self, site: Site) -> None:
        """What the router now fetches on a fresh load of `#/play/<id>`. The moves are
        replayed through the engine, so the board comes back where it was."""
        browser = _joined(site, "ada")
        ada = self._me(site, "ada")
        game = self._game(site, "chess")
        started = self._start(site, game, ada, ["human", self.AGENT])
        opening = browser.get(f"/api/session/{started}").payload
        moved = browser.post(
            f"/api/session/{started}/move", {"move": opening["moves"][0]["move"]}
        )
        assert moved.status == 200, moved.text

        again = browser.get(f"/api/session/{started}").payload
        assert again["id"] == started
        assert again["mode"] == match_rows.LEADERBOARD
        assert again["ply"] == moved.payload["ply"]
        assert again["board"] == moved.payload["board"]

    def test_a_person_with_nothing_running_carries_an_empty_list(
        self, site: Site
    ) -> None:
        browser = _joined(site, "ada")
        self._game(site, "chess")
        assert browser.get("/api/profile").payload["carryOn"] == []


# ------------------------------------------- the people's numbers, over the map's nodes


def _dull(name: str, plies: int) -> dict[str, Any]:
    """`dull` under a new name with its ply limit moved, which is one rule of difference."""
    spec = json.loads((RULESETS / "dull.json").read_text(encoding="utf-8"))
    spec.pop("provenance", None)
    spec["name"] = name
    spec["rules"]["max_plies"] = plies
    return spec


def _spelt_out(made: dict[str, Any]) -> dict[str, Any]:
    """The same game with who-may-take-whom written out, which compiles to the same tables.

    Two files, one node -- the shape `wolves` and `wolves-immune` have in the corpus, and
    the one the overlay has to fold before anybody sees a dot.
    """
    twin = json.loads(json.dumps(made))
    twin["name"] = made["name"] + "-spelt"
    twin["schema_version"] = 3
    for piece in twin["pieces"]:
        piece["captures"] = [other["id"] for other in twin["pieces"]]
    return twin


PLAIN = _dull("plain", 40)
SPELT = _spelt_out(PLAIN)
OTHER = _dull("other", 60)
KILLED = _dull("killed", 80)
MAPPED_SPECS = (PLAIN, SPELT, OTHER, KILLED)


class MapSite:
    """A running site over four spec files, with the store's games rows already in."""

    TOP = "minimax:nodes=6400"
    WHEN = datetime(2026, 9, 19, 12)

    def __init__(self, site: Site) -> None:
        self.site = site

    def hash_of(self, spec: dict[str, Any]) -> str:
        return spec_hash(spec)

    def person(self, handle: str) -> tuple[Browser, int]:
        browser = _joined(self.site, handle)
        with self.site.store.cursor() as cursor:
            cursor.execute("select id from users where handle = %s", (handle,))
            return browser, int(cursor.fetchone()[0])

    def rate(self, spec: dict[str, Any]) -> None:
        """One computer player at the top, so the game is rated and can carry the tag."""
        rating.store(
            self.site.store,
            self.hash_of(spec),
            [rating.Player(
                key=rating.computer(self.TOP), rating=2300.0, deviation=30.0,
                volatility=0.0, matches=40,
            )],
        )
        self.site.store.commit()

    def vote(self, spec: dict[str, Any], voter: int, stars: int) -> None:
        votes.cast(self.site.store, voter, self.hash_of(spec), stars)
        self.site.store.commit()

    def played(self, spec: dict[str, Any], user: int, minute: int = 0) -> None:
        """A finished leaderboard game with a person in seat 0, lost to the top."""
        when = self.WHEN + timedelta(minutes=minute)
        with self.site.store.cursor() as cursor:
            cursor.execute(
                "insert into matches (id, game, mode, seed, seat0_user, seat1_agent,"
                " moves, result, reason, top, started, finished)"
                " values (%s, %s, %s, 0, %s, %s, '[]'::jsonb, 0.0, 'checkmate', true,"
                " %s, %s)",
                (f"m{secrets.token_hex(4)}", self.hash_of(spec), match_rows.LEADERBOARD,
                 user, self.TOP, when, when),
            )
        self.site.store.commit()

    def kill(self, spec: dict[str, Any]) -> None:
        admin.kill_game(self.site.store, self.hash_of(spec))
        self.site.store.commit()


@pytest.fixture
def map_tree(tmp_path: Path) -> Path:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    for spec in MAPPED_SPECS:
        (tmp_path / "rulesets" / f"{spec['name']}.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
    return tmp_path


@pytest.fixture
def map_site(map_tree: Path) -> Iterator[MapSite]:
    with _running_site(map_tree) as running:
        loader.load(running.store, map_tree)
        running.store.commit()
        yield MapSite(running)


class TestThePeoplesNumbersOverTheMap:
    """`docs/archive/plan-game-graph.md` §8.3 and §8.5 G12.

    What a checkout cannot know about the games it draws -- what people scored them out
    of five, and how many leaderboard games they have finished on them -- reaching the
    map as an overlay keyed by the nodes the map already drew.

    Three claims are held here and each is a door somebody could open by accident. The
    overlay names **the same nodes** `/api/graph` drew, so a killed game and a draft are
    out of it without this route deciding anything about either. A node is **a game, not
    a file**, so two spellings are one record with their votes and their matches added.
    And `/api/graph` itself is **one answer for everybody**, which is what lets it stay
    warm while this is answered per request.
    """

    def _people(self, browser: Browser) -> dict[str, Any]:
        answer = browser.get("/api/graph/people")
        assert answer.status == 200, answer.text
        return dict(answer.payload)

    def _nodes(self, browser: Browser) -> dict[str, Any]:
        answer = browser.get("/api/graph")
        assert answer.status == 200, answer.text
        return {str(node["hash"]): node for node in answer.payload["nodes"]}

    # ------------------------------------------------------ what the two payloads say

    def test_the_map_says_there_is_an_overlay_and_a_checkout_says_there_is_not(
        self, map_site: MapSite
    ) -> None:
        """`overlays` is the caller's constant, which is what makes a page with no store
        behind it ask for nothing rather than ask and be refused."""
        from chess_variant_sandbox.web import graphing

        answer = map_site.site.browser().get("/api/graph")
        assert answer.payload["overlays"] == [browse.OVERLAY]
        assert graphing.build([], map_site.site.app.runs_dir).as_dict()["overlays"] == []

    def test_a_visitor_gets_stars_votes_and_played_for_every_node_and_no_untried(
        self, map_site: MapSite
    ) -> None:
        """Anybody may ask, as anybody may ask for the listing, and the two columns are
        the ones a signed-out visitor already sees there. *Untried* reads this person's
        own record, so a visitor has none and is not offered the filter."""
        _, ada = map_site.person("ada")
        map_site.vote(PLAIN, ada, 4)
        map_site.played(PLAIN, ada)

        payload = self._people(map_site.site.browser())

        assert payload["signedIn"] is False
        drawn = self._nodes(map_site.site.browser())
        assert set(payload["games"]) == set(drawn)
        for record in payload["games"].values():
            assert set(record) == {
                "stars", "votes", "played", "unbeaten", "untried", "unplayed", "size"
            }
            assert record["untried"] is None
        assert [one["filter"] for one in payload["filters"]] == [
            browse.UNBEATEN, browse.UNPLAYED
        ]

    def test_a_member_is_offered_untried_and_it_reads_their_own_record(
        self, map_site: MapSite
    ) -> None:
        browser, ada = map_site.person("ada")
        _, grace = map_site.person("grace")
        map_site.played(PLAIN, grace)

        payload = self._people(browser)

        assert payload["signedIn"] is True
        assert browse.UNTRIED in [one["filter"] for one in payload["filters"]]
        games = payload["games"]
        # Somebody else's game leaves it untried for this person, and a game with no row
        # at all is untried for everybody.
        assert games[map_site.hash_of(PLAIN)]["untried"] is True
        assert games[map_site.hash_of(OTHER)]["untried"] is True

        map_site.played(PLAIN, ada)
        assert self._people(browser)["games"][map_site.hash_of(PLAIN)]["untried"] is False

    # ------------------------------------------------------------- one game, two files

    def test_a_twin_nodes_votes_and_games_are_its_two_spellings_added(
        self, map_site: MapSite
    ) -> None:
        """`plain` and `plain-spelt` say one game two ways, and the map draws one dot.

        The server adds the matches and the votes and takes the mean over all the votes,
        so the page is handed one record and averages nothing.
        """
        _, ada = map_site.person("ada")
        _, grace = map_site.person("grace")
        map_site.vote(PLAIN, ada, 5)
        map_site.vote(SPELT, grace, 3)
        map_site.played(PLAIN, ada, minute=0)
        map_site.played(SPELT, grace, minute=1)
        map_site.played(SPELT, ada, minute=2)

        drawn = self._nodes(map_site.site.browser())
        node = next(one for one in drawn.values() if len(one["hashes"]) > 1)
        assert sorted(node["names"]) == ["plain", "plain-spelt"]

        record = self._people(map_site.site.browser())["games"][node["hash"]]
        assert (record["votes"], record["stars"]) == (2, 4.0)
        assert record["played"] == 3
        # And only one record: the other spelling is not a key of its own.
        assert map_site.hash_of(SPELT) not in self._people(map_site.site.browser())["games"]

    def test_a_twin_is_untried_only_while_this_person_has_tried_neither_spelling(
        self, map_site: MapSite
    ) -> None:
        browser, ada = map_site.person("ada")
        drawn = self._nodes(browser)
        twin = next(one["hash"] for one in drawn.values() if len(one["hashes"]) > 1)
        assert self._people(browser)["games"][twin]["untried"] is True

        map_site.played(SPELT, ada)

        assert self._people(browser)["games"][twin]["untried"] is False

    # ------------------------------------------------------------ which games, and why

    def test_a_killed_game_is_in_neither_payload_and_a_draft_is_in_neither(
        self, map_site: MapSite, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The route takes no hash from the request: it asks the map which nodes it drew
        and reads the store about those. So G7's visibility rule is untouched, because
        this route makes no visibility decision of its own."""
        monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
        browser, _ = map_site.person("ada")
        draft = _dull("a draft somebody kept", 100)
        kept = browser.post("/api/design/save", {"spec": draft, "parent": ""})
        assert kept.status == 200, kept.text
        map_site.kill(KILLED)

        drawn = self._nodes(browser)
        games = self._people(browser)["games"]

        for gone in (map_site.hash_of(KILLED), map_site.hash_of(draft)):
            assert gone not in drawn
            assert gone not in games
        assert map_site.hash_of(OTHER) in games

    def test_the_base_payload_is_the_same_bytes_for_two_viewers(
        self, map_site: MapSite
    ) -> None:
        """One answer for everybody, warm, with nobody's name in it. Everything that
        depends on who is asking is in the overlay, which is answered per request.

        Warmed first, because the first build of a payload writes the pair cache and the
        next reads it back with its keys sorted -- the same JSON, spelled in a different
        order. That is a difference between a cold server and a warm one, not between
        two people, and this test is about the second.
        """
        browser, ada = map_site.person("ada")
        map_site.played(PLAIN, ada)
        map_site.site.browser().get("/api/graph")

        visitor = map_site.site.browser().get("/api/graph")
        member = browser.get("/api/graph")

        assert visitor.body == member.body

    # --------------------------------------------------------------- absent is not zero

    def test_an_unvoted_game_has_no_stars_and_an_unplayed_one_has_a_true_nought(
        self, map_site: MapSite
    ) -> None:
        """An unvoted game is not a game everybody hated. A game nobody has finished has
        genuinely been finished no times, so that one is a number."""
        _, ada = map_site.person("ada")
        map_site.vote(OTHER, ada, 2)

        games = self._people(map_site.site.browser())["games"]
        quiet = games[map_site.hash_of(PLAIN)]

        assert (quiet["stars"], quiet["votes"]) == (None, 0)
        assert quiet["played"] == 0 and quiet["unplayed"] is True
        assert quiet["size"][browse.STARS_SIZE] is None
        assert games[map_site.hash_of(OTHER)]["stars"] == 2.0

    def test_unbeaten_is_null_until_the_game_is_rated(self, map_site: MapSite) -> None:
        """The card's own rule: an unrated game says *unrated* rather than carrying a tag
        nothing can be beaten on."""
        games = self._people(map_site.site.browser())["games"]
        assert games[map_site.hash_of(OTHER)]["unbeaten"] is None

        map_site.rate(OTHER)

        again = self._people(map_site.site.browser())["games"]
        assert again[map_site.hash_of(OTHER)]["unbeaten"] is True

    # ------------------------------------------------------------------- the two sizes

    def test_the_two_keys_arrive_in_the_shape_the_size_menu_already_reads(
        self, map_site: MapSite
    ) -> None:
        """G11's mechanism, extended: the same seven fields, so the menu, the legend's
        sentence and a dot's radius are the ones that stage built."""
        _, ada = map_site.person("ada")
        map_site.vote(PLAIN, ada, 4)
        map_site.played(PLAIN, ada, minute=0)
        map_site.played(OTHER, ada, minute=1)
        map_site.played(OTHER, ada, minute=2)

        payload = self._people(map_site.site.browser())
        sizes = {one["key"]: one for one in payload["sizes"]}

        assert list(sizes) == [browse.STARS_SIZE, browse.PLAYED_SIZE]
        for one in sizes.values():
            assert set(one) == {
                "key", "label", "words", "low", "high", "lowWords", "highWords"
            }
        # The star scale is the one people voted on, and the played scale is the
        # library's own busiest game.
        assert (sizes[browse.STARS_SIZE]["low"], sizes[browse.STARS_SIZE]["high"]) == (1, 5)
        assert sizes[browse.PLAYED_SIZE]["high"] == 2

        games = payload["games"]
        assert games[map_site.hash_of(OTHER)]["size"][browse.PLAYED_SIZE] == 1.0
        assert games[map_site.hash_of(PLAIN)]["size"][browse.PLAYED_SIZE] == 0.5
        assert games[map_site.hash_of(PLAIN)]["size"][browse.STARS_SIZE] == 0.75
        for record in games.values():
            for value in record["size"].values():
                assert value is None or 0.0 <= value <= 1.0

    def test_a_key_nothing_can_answer_is_left_out_of_the_menu(
        self, map_site: MapSite
    ) -> None:
        """G11's rule for `score` on a library with no ramp: an entry that sized every
        game the same is a control that does nothing."""
        payload = self._people(map_site.site.browser())

        assert payload["sizes"] == []
        for record in payload["games"].values():
            assert record["size"] == {browse.STARS_SIZE: None, browse.PLAYED_SIZE: None}

    def test_the_route_is_classified_and_reachable_by_anybody(self) -> None:
        """`tests/test_site.py`'s route table fails on a route it has not heard of, and
        a route with no `visitor` sentence is one a signed-out person may reach."""
        found = routes.find("GET", "/api/graph/people")
        assert found is not None and found.visitor == ""
        assert found in routes.OWN


class TestEveryPageRefusesInlineScript:
    """Q514: the site sends the policy that refuses inline script on every page route.

    `docs/reference/page-javascript.md` §7. The header is written in the local handler's
    `end_headers`, which the site's own reaches through `super()`, so the test asks the
    site itself: every route in `routes.PAGES` and the application's page, as nobody, as
    a member and as an admin, whatever each answers (a page, the door's 303, the 404 a
    member gets for `/admin`).
    """

    def test_every_page_route_carries_the_policy(self, site: Site) -> None:
        member = _joined(site, "pia")
        chief = _joined(site, "quin")
        with site.store.cursor() as cursor:
            cursor.execute("update users set role = 'admin' where handle = 'quin'")
        site.store.commit()
        asked = ("/", "/index.html", *routes.PAGES)
        for who, browser in (("nobody", site.browser()), ("member", member), ("admin", chief)):
            for path in asked:
                answer = browser.get(path)
                assert answer.status in (200, 303, 404), (who, path, answer.status)
                assert answer.headers.get_all("Content-Security-Policy") == [
                    CONTENT_SECURITY_POLICY
                ], (who, path)
