"""Q670: an answer a site page cannot read is a sentence in the site's voice, never the browser's.

Q659 gave the application's page one request helper (`tests/test_page_requests.py`); the
site's own pages (profile, players, suggestions, the challenge link, the maintainer's
page, signing in and joining) are a separate set of scripts, and they still read
`response.json()` unguarded, so a deploy showed Safari's *The string did not match the
expected pattern* there too. Their helper is in `site/static/post.js`, the prelude every
site page loads first. These tests run it in node with a `fetch` that answers from a
script of replies -- an HTML page, a 502, a dropped connection, a refusal the server
wrote, a normal answer -- and then run two pages' own scripts on top of it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import node_session
import site_words

from chess_variant_sandbox.site import pages as site_pages

STATIC = Path(site_pages.__file__).resolve().parent / "static"

_STUB = """
  const __KNOWN = {};
  function __node(id) {
    return { id, textContent: '', className: '', children: [], options: [], disabled: false,
      value: '', style: {}, dataset: {},
      appendChild(kid) { this.children.push(kid); return kid; }, append() {}, prepend() {},
      replaceChildren(...kids) { this.children = kids; }, addEventListener() {},
      setAttribute() {}, classList: { toggle() {}, add() {}, remove() {} } };
  }
  global.document = {
    getElementById: id => __KNOWN[id] || (__KNOWN[id] = __node(id)),
    createElement: tag => __node(tag), createTextNode: text => ({ textContent: text }),
    querySelector: () => null, querySelectorAll: () => [], addEventListener() {},
  };
  global.window = { location: { href: '' }, confirm: () => false, addEventListener() {} };
  global.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
  const REPLIES = {
    ok: () => ({ ok: true, status: 200, json: async () => ({ answer: 42 }) }),
    html: () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError(
      'The string did not match the expected pattern.'); } }),
    bad502: () => ({ ok: false, status: 502, json: async () => { throw new SyntaxError(
      'Unexpected token <'); } }),
    bad503json: () => ({ ok: false, status: 503,
                         json: async () => ({ error: 'Every seat at that level is taken.' }) }),
    refused: () => ({ ok: false, status: 401, json: async () => ({ error: 'Sign in first.' }) }),
    down: () => { throw new TypeError('Load failed'); },
  };
  let __ASKED = [];
  function serve(script) {
    __ASKED = [];
    let at = 0;
    global.fetch = async (path, options) => {
      __ASKED.push(((options && options.method) || 'GET') + ' ' + path);
      return REPLIES[script[at++] || 'ok']();
    };
  }
  serve([]);
"""

# label -> (method, the replies in the order they are given).
_CASES: list[tuple[str, str, list[str]]] = [
    ("get ok", "GET", ["ok"]),
    ("post ok", "POST", ["ok"]),
    ("get html", "GET", ["html", "html"]),
    ("get 502 twice", "GET", ["bad502", "bad502"]),
    ("get 502 then ok", "GET", ["bad502", "ok"]),
    ("get down", "GET", ["down", "down"]),
    ("get refused", "GET", ["refused", "ok"]),
    ("get 503 with a sentence", "GET", ["bad503json", "ok"]),
    ("post html", "POST", ["html", "ok"]),
    ("post 502", "POST", ["bad502", "ok"]),
    ("post down", "POST", ["down", "ok"]),
    ("post refused", "POST", ["refused", "ok"]),
]

_HELPER_RUNNER = """
  ASK_RETRY_PAUSE = 0;
  (async () => {
    const seen = {};
    for (const [label, method, script] of CASES) {
      serve(script);
      let got;
      try {
        const answer = method === 'GET' ? await read('/api/thing') : await post('/api/thing', { a: 1 });
        got = method === 'GET' ? { ok: answer.ok, data: answer.data } : { data: answer };
      } catch (error) { got = { error: error.message, unavailable: !!error.unavailable }; }
      seen[label] = { ...got, asked: __ASKED.length };
    }
    console.log(JSON.stringify(seen));
  })();
"""

_RAN: dict[str, Any] = {}


def _script(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _helper() -> dict[str, Any]:
    if not _RAN:
        cases = "const CASES = " + json.dumps(_CASES) + ";\n"
        done = node_session.run(_STUB + _script("post.js") + cases + _HELPER_RUNNER)
        assert done.returncode == 0, done.stderr
        _RAN.update(json.loads(done.stdout))
    return _RAN


def _page(page: str, replies: list[str], prelude: str = "") -> dict[str, Any]:
    """Run `post.js` and a page's script with `fetch` answering `replies`; what the page
    wrote where it speaks, and what it asked."""
    runner = """
      setTimeout(() => {
        const said = id => (__KNOWN[id] ? __KNOWN[id].textContent : '');
        const kids = id => (__KNOWN[id] ? __KNOWN[id].children.map(k => k.textContent) : []);
        console.log(JSON.stringify({ note: said('note'), challenge: kids('challenge'),
                                     asked: __ASKED }));
      }, 30);
    """
    source = (_STUB + "serve(" + json.dumps(replies) + ");\n" + _script("post.js")
              + "ASK_RETRY_PAUSE = 0;\n" + prelude + _script(page) + runner)
    done = node_session.run(source)
    assert done.returncode == 0, done.stderr
    result: dict[str, Any] = json.loads(done.stdout)
    return result


SAID = "The site is updating or busy and did not answer. Try again in a moment."
BROWSER_SAID = ("did not match the expected pattern", "Unexpected token", "Load failed",
                "Failed to fetch", "Bad Gateway")


class TestTheHelper:
    def test_a_normal_answer_is_returned_as_it_was(self) -> None:
        seen = _helper()
        assert seen["get ok"] == {"ok": True, "data": {"answer": 42}, "asked": 1}
        assert seen["post ok"] == {"data": {"answer": 42}, "asked": 1}

    def test_a_web_page_a_502_or_a_dropped_connection_is_one_plain_sentence(self) -> None:
        seen = _helper()
        for label in ("get html", "get 502 twice", "get down", "post html", "post 502",
                      "post down"):
            got = seen[label]
            assert got["error"] == SAID, (label, got)
            assert got["unavailable"] is True
            assert not any(raw in got["error"] for raw in BROWSER_SAID)

    def test_a_read_is_asked_once_more_and_a_post_never(self) -> None:
        """No POST a site page sends is retried: a suggestion, an invitation or an accepted
        challenge would be made twice, and a sign-in counts against its limit."""
        seen = _helper()
        for label in ("get html", "get 502 twice", "get down"):
            assert seen[label]["asked"] == 2, label
        for label in ("post html", "post 502", "post down", "post refused"):
            assert seen[label]["asked"] == 1, label

    def test_a_read_that_answers_the_second_time_is_an_answer(self) -> None:
        assert _helper()["get 502 then ok"] == {"ok": True, "data": {"answer": 42}, "asked": 2}

    def test_a_refusal_the_server_wrote_is_an_answer_and_not_repeated(self) -> None:
        seen = _helper()
        assert seen["get refused"] == {"ok": False, "data": {"error": "Sign in first."},
                                       "asked": 1}
        assert seen["get 503 with a sentence"] == {
            "ok": False, "data": {"error": "Every seat at that level is taken."}, "asked": 1}
        assert seen["post refused"] == {"error": "Sign in first.", "unavailable": False,
                                        "asked": 1}

    def test_the_sentence_passes_the_words_guard(self) -> None:
        assert site_words.hits(SAID) == []


class TestThePages:
    def test_a_page_that_reads_on_load_says_the_sentence_after_two_tries(self) -> None:
        got = _page("suggestions.js", ["bad502", "html"])
        assert got["asked"] == ["GET /api/suggestions", "GET /api/suggestions"]
        assert got["note"] == SAID

    def test_a_refusal_still_reads_as_the_page_says_it(self) -> None:
        got = _page("players.js", ["refused"],
                    "document.getElementById('page-data').textContent = '{}';\n")
        assert got["note"] == "Sign in to see the players."

    def test_the_challenge_link_says_the_sentence_or_the_servers_own(self) -> None:
        prelude = ("document.getElementById('page-data').textContent = "
                   "JSON.stringify({ token: 'abc' });\n")
        got = _page("challenge.js", ["down", "down"], prelude)
        assert got["challenge"] == [SAID]
        got = _page("challenge.js", ["refused"], prelude)
        assert got["challenge"] == ["Sign in first."]


class TestOneHelper:
    def test_only_the_helper_asks_or_reads_a_body(self) -> None:
        """Every request a site page makes is `askOnce`'s, in `post.js`."""
        for route, file in site_pages.SCRIPTS.items():
            text = file.read_text(encoding="utf-8")
            code = re.sub(r"//[^\n]*", "", text)
            expected = 1 if file.name == "post.js" else 0
            assert len(re.findall(r"\bfetch\(", code)) == expected, route
            assert code.count(".json()") == expected, route

    def test_every_page_loads_the_helper_first(self) -> None:
        for name in ("join", "signin", "profile", "players", "suggestions", "admin",
                     "challenge"):
            assert site_pages.SCRIPT_PREFIX + name + ".js" in site_pages.SCRIPTS
        source = Path(site_pages.__file__).read_text(encoding="utf-8")
        lists = re.findall(r'scripts = \(([^)]*)\)', source)
        lists += re.findall(r', \(("[^)]*"challenge")\)', source)
        assert len(lists) == 7
        for scripts in lists:
            assert scripts.startswith('"post"'), scripts
