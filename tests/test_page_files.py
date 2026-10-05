"""The pages' JavaScript is plain files, and no page a server sends carries a script body.

Q513, variant A of `docs/reference/page-javascript.md` §4.1. The page's views, the site's
pages' scripts and the replay's were Python strings; they are `.js` files now, loaded by
`<script src>` and answered from an allowlist (`web.page.SCRIPTS`, `site.pages.SCRIPTS`)
the way the measuring worker is. What is held here is what the move promised:

- **No served page has an inline script** (§4.1, *What it prevents*): every `<script>` is
  a `src` the server answers, or a JSON data tag, which never runs. That is what a policy
  refusing inline script (`script-src 'self'`, Q514) needs, and it is why the markup may
  carry no `on...=` handler either.
- **The files are the whole of it**: each table names files that exist, every file in the
  directories is named, and the page loads the table in its order.
- **Each file parses on its own**, found by a glob rather than listed, so a new file
  cannot go unparsed as `/suggestions` once did.
- **They run as separate scripts.** The suite's Node runs read the files joined into one
  program (`page_text`); a browser runs them one after another in one global scope, where
  a function declared in a later file is not yet there. Loading them one by one holds that
  nothing at load reaches forward.
- **The routes answer them**: `text/javascript`, `no-store`, the file's bytes, and a 404
  for any name the table does not hold.
"""

from __future__ import annotations

import json
import re
import threading
from http.client import HTTPConnection
from pathlib import Path

import node_session
import page_names
import page_text
import pytest

from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import routes
from chess_variant_sandbox.web import library, page
from chess_variant_sandbox.web.page import shell
from chess_variant_sandbox.web.server import CONTENT_SECURITY_POLICY, App, Server, handler_for

PACKAGE = Path(page.__file__).resolve().parents[2]
RULESETS = PACKAGE.parent / "rulesets"
SERVE_FAST = {"poll_interval": 0.02}

# Where script files live, each directory with the table that serves it.
DIRECTORIES = (
    (PACKAGE / "web" / "static" / "page", page.SCRIPTS),
    (PACKAGE / "site" / "static", site_pages.SCRIPTS),
)

_TAG = re.compile(r"<script([^>]*)>(.*?)</script>", re.S)


def _served_pages() -> dict[str, str]:
    """Every HTML page a server sends: the application's, and each of the site's."""
    return {"/": page.PAGE, **page_names.site_pages()}


class TestNoServedPageCarriesAScriptBody:
    def test_every_script_tag_is_a_served_file_or_data(self) -> None:
        for path, html in _served_pages().items():
            for attributes, body in _TAG.findall(html):
                if 'type="application/json"' in attributes:
                    # Data never runs, and nothing in it can close the tag.
                    json.loads(body)
                    assert "</" not in body, path
                    continue
                source = re.fullmatch(r' src="([^"]+)"', attributes)
                assert source is not None, f"{path}: <script{attributes}>"
                assert source[1] in page_text.SERVED, f"{path}: {source[1]}"
                assert body == "", f"{path}: a tag with a src and a body"

    def test_no_tag_carries_an_event_handler(self) -> None:
        """An `onclick=` in the markup is inline script too, and the policy refuses it."""
        for path, html in _served_pages().items():
            markup = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
            assert re.findall(r"<[^>]*\son[a-z]+\s*=", markup) == [], path

    def test_a_page_with_a_script_still_has_one(self) -> None:
        """The pages that ran a script still run one, a challenge link's (Q571) among
        them, and credits, privacy and the badges still run none. Every page carries the
        shell's script once, in its head (V2), which is not counted here."""
        shell_tag = f'<script src="{shell.SHELL_SCRIPT_ROUTE}"></script>'
        for path, html in _served_pages().items():
            assert html.count(shell_tag) == 1, path
            assert html.index(shell_tag) < html.index("</head>"), path
        scripted = {
            path for path, html in _served_pages().items()
            if "<script" in html.replace(shell_tag, "")
        }
        assert scripted == {"/", "/join", "/signin", "/profile", "/players", "/suggestions",
                            "/admin", "/challenge_link", "/players/ada"}


class TestTheFilesAreTheWholeOfIt:
    def test_every_file_a_table_names_exists(self) -> None:
        for route, file in page_text.SERVED.items():
            assert file.is_file(), route
            assert file.suffix == ".js", route

    def test_every_file_in_the_directories_is_served(self) -> None:
        for directory, table in DIRECTORIES:
            named = {file.resolve() for file in table.values()}
            assert sorted(directory.glob("*.js")) and all(
                file.resolve() in named for file in directory.glob("*.js")
            ), directory

    def test_the_page_loads_the_table_in_its_order(self) -> None:
        loaded = re.findall(r'<script src="([^"]+)"></script>', page.PAGE)
        assert loaded[0] == shell.SHELL_SCRIPT_ROUTE
        loaded = loaded[1:]
        assert loaded == list(page.SCRIPTS)
        assert loaded[0].endswith("/core.js") and loaded[-1].endswith("/boot.js")

    def test_each_table_lives_under_its_own_prefix(self) -> None:
        assert all(route.startswith(page.SCRIPT_PREFIX) for route in page.SCRIPTS)
        assert all(route.startswith(site_pages.SCRIPT_PREFIX) for route in site_pages.SCRIPTS)
        assert page.SCRIPT_PREFIX in routes.STATIC
        assert site_pages.SCRIPT_PREFIX in routes.STATIC

    def test_the_replay_carries_its_files_inline(self) -> None:
        """The offline replay is a file opened from disk, not an answer: it keeps one
        inline script, made of `wrapped.js` and `replay.js`."""
        from chess_variant_sandbox.viz import html as replay

        viz = PACKAGE / "viz"
        for name in ("wrapped.js", "replay.js"):
            assert (viz / name).read_text(encoding="utf-8").strip() in replay._PAGE, name


class TestEachFileParses:
    """Found by a glob, so a script added later is parsed without anybody listing it."""

    def test_every_script_file_parses_on_its_own(self) -> None:
        files = [
            *(PACKAGE / "web" / "static" / "page").glob("*.js"),
            *(PACKAGE / "site" / "static").glob("*.js"),
            *(PACKAGE / "viz").glob("*.js"),
            shell.SHELL_SCRIPT_FILE,
        ]
        assert len(files) >= 25
        for file in files:
            done = node_session.run(file.read_text(encoding="utf-8"), check=True)
            assert done.returncode == 0, f"{file.name}: {done.stderr}"


# A browser's DOM, as little of it as loading the page needs. The same stub the page's
# own Node checks use (`tests/test_web.py::_in_the_page`), written for a context of its own.
_PAGE_STUB = """
const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                           set: () => true });
globalThis.document = { getElementById: () => el, querySelector: () => null,
                        querySelectorAll: () => [], addEventListener: () => {},
                        createElement: () => el, body: el };
globalThis.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                      addEventListener: () => {}, scrollTo: () => {}, innerWidth: 1200 };
globalThis.location = { pathname: '/game', hash: '' };
globalThis.fetch = () => new Promise(() => {});
"""

# The site's pages read their data tag and draw with `textContent`; the stub hands the tag
# its JSON and every other element an inert one.
_SITE_STUB = """
function element() {
  return { className: '', textContent: '', children: [], style: {}, dataset: {},
           appendChild(child) { return child; }, replaceChildren() {}, prepend() {},
           addEventListener() {}, setAttribute() {}, append() {}, remove() {} };
}
const BY_ID = {};
globalThis.document = {
  createElement: element, createTextNode: said => ({ textContent: said }),
  getElementById: id => BY_ID[id] || (BY_ID[id] = element()),
  addEventListener() {}, querySelector: () => null, querySelectorAll: () => [],
};
globalThis.window = { location: {}, addEventListener() {} };
globalThis.fetch = () => new Promise(() => {});
"""

# Each file in its own `vm.runInContext`, in one context: separate classic scripts sharing
# one global scope, `let` and `const` included, as a browser loads them.
_ONE_BY_ONE = """
const vm = require('vm');
const context = vm.createContext({ console, setTimeout, clearTimeout, setInterval,
                                   clearInterval, queueMicrotask, Promise });
vm.runInContext(%(stub)s, context, { filename: 'stub.js' });
const failed = [];
for (const [name, source] of %(files)s) {
  try { vm.runInContext(source, context, { filename: name }); }
  catch (error) { failed.push(name + ': ' + String(error)); }
}
console.log(JSON.stringify(failed));
"""


def _one_by_one(stub: str, files: list[tuple[str, str]]) -> list[str]:
    done = node_session.run(_ONE_BY_ONE % {"stub": json.dumps(stub), "files": json.dumps(files)})
    assert done.returncode == 0, done.stderr
    failed: list[str] = json.loads(done.stdout)
    return failed


class TestTheFilesRunAsSeparateScripts:
    def test_the_page_loads_file_by_file(self) -> None:
        files = [(route, page_text.text(route)) for route in page.SCRIPTS]
        assert _one_by_one(_PAGE_STUB, files) == []

    def test_a_call_into_a_later_file_is_caught(self) -> None:
        """What the check is for: one program joined up would hoist the later function,
        and two scripts do not."""
        files = [("a.js", "const drawn = later();"), ("b.js", "function later() { return 1; }")]
        assert _one_by_one(_PAGE_STUB, files) == ["a.js: ReferenceError: later is not defined"]
        assert node_session.run(files[0][1] + "\n" + files[1][1]).returncode == 0

    def test_each_site_page_loads_file_by_file(self) -> None:
        for path, html in page_names.site_pages().items():
            routes_loaded = [
                route for route in re.findall(r'<script src="([^"]+)"></script>', html)
                if route != shell.SHELL_SCRIPT_ROUTE
            ]
            if not routes_loaded:
                continue
            data = page_text.data_of(html)
            stub = _SITE_STUB + (
                f"BY_ID['page-data'] = {{ textContent: {json.dumps(data)} }};\n" if data else ""
            )
            files = [(route, page_text.text(route)) for route in routes_loaded]
            assert _one_by_one(stub, files) == [], path


def _get(app: App, *paths: str) -> list[tuple[int, dict[str, str], bytes]]:
    """Each path asked of a running local application."""
    server = Server(("127.0.0.1", 0), handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    answers = []
    try:
        for path in paths:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                answers.append((response.status, dict(response.getheaders()), response.read()))
            finally:
                connection.close()
    finally:
        server.shutdown()
        server.server_close()
    return answers


class TestTheApplicationServesThem:
    @pytest.fixture()
    def app(self, tmp_path: Path) -> App:
        return App(RULESETS, library.RUNS_DIR, browser_dir=tmp_path / "absent")

    def test_each_file_as_the_worker_script_is(self, app: App) -> None:
        answers = _get(app, *page.SCRIPTS)
        for (route, file), (status, headers, body) in zip(page.SCRIPTS.items(), answers, strict=True):
            assert status == 200, route
            assert headers["Content-Type"] == "text/javascript", route
            assert headers["Cache-Control"] == "no-store", route
            assert body == file.read_bytes(), route

    def test_nothing_else_under_the_prefix(self, app: App) -> None:
        asked = [
            page.SCRIPT_PREFIX + "nothing.js",
            page.SCRIPT_PREFIX + "../server.py",
            page.SCRIPT_PREFIX + "%2e%2e/server.py",
            page.SCRIPT_PREFIX,
        ]
        for route, (status, _, body) in zip(asked, _get(app, *asked), strict=True):
            assert status == 404, route
            assert b"def " not in body, route

    def test_the_page_names_only_what_is_served(self, app: App) -> None:
        [(status, headers, body)] = _get(app, "/")
        assert status == 200 and headers["Cache-Control"] == "no-store"
        assert body == page.PAGE.encode("utf-8")
        assert "<script>" not in body.decode("utf-8")


def _directives() -> dict[str, list[str]]:
    policy: dict[str, list[str]] = {}
    for part in CONTENT_SECURITY_POLICY.split(";"):
        name, *values = part.split()
        assert name not in policy, name
        policy[name] = values
    return policy


class TestThePolicyRefusesInlineScript:
    """Q514, `docs/reference/page-javascript.md` §7: what the move to files was for.

    The site's pages are held to it in `tests/test_site.py`
    (`TestEveryPageRefusesInlineScript`); these are the local application's routes and
    the policy's own shape.
    """

    def test_script_is_the_origins_files_and_nothing_inline(self) -> None:
        policy = _directives()
        assert policy["script-src"] == ["'self'", "'wasm-unsafe-eval'"]
        assert policy["worker-src"] == ["'self'"]
        assert policy["default-src"] == ["'self'"]
        assert policy["object-src"] == ["'none'"] and policy["base-uri"] == ["'none'"]
        for values in policy.values():
            # No inline script, no eval, no host of anybody else's.
            assert "'unsafe-eval'" not in values
            assert not any(":" in value and value != "data:" for value in values), values
        inline = [name for name, values in policy.items() if "'unsafe-inline'" in values]
        # The stylesheet is still a `<style>` in each page, and the page writes `style=`.
        assert inline == ["style-src"]

    def test_every_route_of_the_application_carries_it(self, tmp_path: Path) -> None:
        app = App(RULESETS, library.RUNS_DIR, browser_dir=tmp_path / "absent")
        asked = [
            "/",
            "/index.html",
            *page.SCRIPTS,
            "/static/measure-worker.js",
            "/browser/manifest.json",
            "/static/vendor/nothing.js",
            "/api/nowhere",
        ]
        for path, (status, headers, _) in zip(asked, _get(app, *asked), strict=True):
            assert status in (200, 204, 404), path
            assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY, path
