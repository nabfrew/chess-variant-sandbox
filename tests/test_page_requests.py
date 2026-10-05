"""Q659: an answer the page cannot read is a sentence in the site's voice, never the browser's.

The user, 2026-10-02, on the live site: *"Any move I try to make gives an error message.
'The string did not match the expected pattern.' Reloading the page made it go away."* That
is Safari's sentence for `response.json()` on a body that is not JSON: the proxy answering
a move with its own error page while the site restarted.

Every request the page reads goes through `api` (`web/static/page/core.js`). These tests run
the page's own script in node with a `fetch` that answers from a script of replies -- an
HTML page, a 502, a dropped connection, a refusal the server wrote, a normal answer -- and
read what `api` and the move path (`send`, `board.js`) did with each. Nothing here is
about any one game: the move is a number and the position is a history length.
"""

from __future__ import annotations

import json
import re
from typing import Any

import node_session
import site_words

_STUB = """
  function __node() {
    const node = {
      className: '', children: [], parent: null, style: {}, dataset: {}, hidden: false,
      innerHTML: '', textContent: '',
      append() {}, appendChild(kid) { return kid; }, prepend() {}, remove() {},
      closest() { return null; }, querySelectorAll() { return []; },
      querySelector() { return null; }, addEventListener() {}, setAttribute() {},
      classList: { toggle() {}, add() {}, remove() {} },
    };
    return node;
  }
  const __KNOWN = {};
  global.document = {
    getElementById: id => __KNOWN[id] || (__KNOWN[id] = __node()),
    querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
    createElement: () => __node(), body: __node(), visibilityState: 'visible',
  };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 375, scrollY: 0,
                    scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  // Nothing answers the boot, so the runner below is the only thing driving the page.
  global.fetch = () => new Promise(() => {});
"""

# The replies a fake server can give, by name. `html` is what a proxy sends while the
# application behind it is down; `refused` is a refusal the application wrote itself.
_RUNNER = """
  const REPLIES = {
    ok: () => ({ ok: true, status: 200, statusText: 'OK',
                 json: async () => ({ answer: 42 }) }),
    html: () => ({ ok: true, status: 200, statusText: 'OK',
                   json: async () => { throw new SyntaxError(
                     'The string did not match the expected pattern.'); } }),
    bad502: () => ({ ok: false, status: 502, statusText: 'Bad Gateway',
                     json: async () => { throw new SyntaxError('Unexpected token <'); } }),
    bad503json: () => ({ ok: false, status: 503, statusText: 'Service Unavailable',
                         json: async () => ({ error: 'Every seat at that level is taken.' }) }),
    refused: () => ({ ok: false, status: 400, statusText: 'Bad Request',
                      json: async () => ({ error: 'that move is not legal in this position' }) }),
    down: () => { throw new TypeError('Load failed'); },
  };
  // Each request takes the next reply in `script`; `asked` is what was requested.
  function serve(script, routes) {
    const asked = [];
    let at = 0;
    global.fetch = async (path, options) => {
      asked.push(((options && options.method) || 'GET') + ' ' + path);
      const name = routes ? routes(path, options) : script[at++];
      return REPLIES[name]();
    };
    return asked;
  }
  async function outcome(call) {
    try { return { data: await call() }; }
    catch (error) { return { error: error.message, unavailable: !!error.unavailable }; }
  }
  API_RETRY_PAUSE = 0;
  (async () => {
    const seen = {};
    for (const accounts of [true, false]) {
      ACCOUNTS = accounts;
      for (const [label, method, script] of CASES) {
        const asked = serve(script);
        const body = method === 'POST' ? { move: 7 } : undefined;
        const got = await outcome(() => api('/api/thing', body));
        seen[(accounts ? 'site ' : 'local ') + label] = { ...got, asked: asked.length };
      }
    }
    // The move path. `fail` and the two drawers are recorded, not drawn.
    ACCOUNTS = true;
    const calls = [];
    fail = (error, near) => { calls.push(['fail', error.message, near]); };
    drawBoard = () => { calls.push(['drawBoard']); };
    applySnapshot = s => { calls.push(['applySnapshot', s.history.length]); };
    const moves = {};
    // Reading the game after a lost move: as it was, or one move on.
    for (const [name, history] of [['same', [1, 2]], ['moved', [1, 2, 3]]]) {
      REPLIES[name] = () => ({ ok: true, status: 200, statusText: 'OK',
                               json: async () => ({ id: 'g1', history }) });
    }
    for (const [label, moveReply, sessionReplies] of MOVES) {
      calls.length = 0;
      SESSION = { id: 'g1', history: [1, 2], correspondence: null };
      let reads = 0;
      const asked = serve(null, (path, options) => {
        if (options && options.method === 'POST') return moveReply;
        return sessionReplies[reads++];
      });
      send(7);
      await new Promise(done => setTimeout(done, 20));
      moves[label] = { calls: calls.map(c => c.slice()), asked };
    }
    console.log(JSON.stringify({ seen, moves }));
  })();
"""

# label -> (method, the replies in the order they are given).
_CASES: list[tuple[str, str, list[str]]] = [
    ("get ok", "GET", ["ok"]),
    ("post ok", "POST", ["ok"]),
    ("post html", "POST", ["html", "ok"]),
    ("get html", "GET", ["html", "html"]),
    ("get 502 twice", "GET", ["bad502", "bad502"]),
    ("get 502 then ok", "GET", ["bad502", "ok"]),
    ("post 502", "POST", ["bad502", "ok"]),
    ("get down", "GET", ["down", "down"]),
    ("post down", "POST", ["down", "ok"]),
    ("post refused", "POST", ["refused", "ok"]),
    ("get 503 with a sentence", "GET", ["bad503json", "ok"]),
]

# label -> (the move's reply, the replies to reading the game afterwards).
_MOVES: list[tuple[str, str, list[str]]] = [
    ("made", "moved", []),
    ("refused", "refused", []),
    ("lost, game unchanged", "html", ["same"]),
    ("lost, game moved on", "bad502", ["moved"]),
    ("lost, game unreadable", "down", ["down", "down"]),
]

_RAN: dict[str, Any] = {}


def _run() -> dict[str, Any]:
    if not _RAN:
        from page_text import PAGE

        script = re.search(r"<script>(.*)</script>", PAGE, re.S)
        assert script is not None
        cases = "const CASES = " + json.dumps(_CASES) + ";\n"
        moves = "const MOVES = " + json.dumps(_MOVES) + ";\n"
        done = node_session.run(_STUB + script.group(1) + "\n" + cases + moves + _RUNNER)
        assert done.returncode == 0, done.stderr
        _RAN.update(json.loads(done.stdout))
    return _RAN


SITE_SAID = "The site is updating or busy and did not answer. Try again in a moment."
LOCAL_SAID = "The application did not answer. Try again in a moment."
BROWSER_SAID = ("did not match the expected pattern", "Unexpected token", "Load failed",
                "Bad Gateway")


class TestAnAnswerThePageCannotRead:
    def test_a_normal_answer_is_returned_as_it_was(self) -> None:
        """The happy path does not change: one request, the server's data."""
        seen = _run()["seen"]
        for where in ("site", "local"):
            for label in ("get ok", "post ok"):
                assert seen[f"{where} {label}"] == {"data": {"answer": 42}, "asked": 1}

    def test_a_web_page_a_502_or_a_dropped_connection_is_one_plain_sentence(self) -> None:
        seen = _run()["seen"]
        for label in ("post html", "get html", "get 502 twice", "post 502", "get down",
                      "post down"):
            for where, said in (("site", SITE_SAID), ("local", LOCAL_SAID)):
                got = seen[f"{where} {label}"]
                assert got["error"] == said, (where, label, got)
                assert got["unavailable"] is True
                assert not any(raw in got["error"] for raw in BROWSER_SAID)

    def test_a_read_is_asked_once_more_and_a_post_never(self) -> None:
        """A GET changes nothing, so asking it twice is safe. A POST is never repeated:
        a move carries no position, so a repeat after a first that landed would be put to
        the next one."""
        seen = _run()["seen"]
        for label in ("get html", "get 502 twice", "get down"):
            assert seen[f"site {label}"]["asked"] == 2, label
        for label in ("post html", "post 502", "post down", "post refused"):
            assert seen[f"site {label}"]["asked"] == 1, label

    def test_a_read_that_answers_the_second_time_is_an_answer(self) -> None:
        seen = _run()["seen"]
        assert seen["site get 502 then ok"] == {"data": {"answer": 42}, "asked": 2}

    def test_a_sentence_the_server_wrote_still_wins(self) -> None:
        """A refusal with a JSON body is an answer, whatever its status, and is not
        repeated."""
        seen = _run()["seen"]
        assert seen["site post refused"] == {
            "error": "that move is not legal in this position", "unavailable": False,
            "asked": 1}
        assert seen["site get 503 with a sentence"] == {
            "error": "Every seat at that level is taken.", "unavailable": False,
            "asked": 1}

    def test_the_sentences_pass_the_words_guard(self) -> None:
        moves = _run()["moves"]
        said = [SITE_SAID, LOCAL_SAID] + [
            call[1] for move in moves.values() for call in move["calls"]
            if call[0] == "fail"]
        for sentence in said:
            assert site_words.hits(sentence) == [], sentence


class TestAMoveWhoseAnswerWasLost:
    """`send`: the move is never sent twice; the game is read instead, and the board is
    drawn as the game is."""

    def test_a_move_that_is_made_is_drawn(self) -> None:
        made = _run()["moves"]["made"]
        assert made["asked"] == ["POST /api/session/g1/move"]
        assert made["calls"] == [["applySnapshot", 3]]

    def test_a_refusal_is_said_as_a_toast_never_in_the_status_strip(self) -> None:
        """Q725: a box beside the status grew the strip and pushed the board down; no
        control is named, so `fail` puts the sentence in a toast at the foot."""
        refused = _run()["moves"]["refused"]
        assert refused["asked"] == ["POST /api/session/g1/move"]
        assert refused["calls"] == [
            ["fail", "that move is not legal in this position", None]]

    def test_a_lost_answer_on_an_unchanged_game_redraws_the_board_and_says_so(self) -> None:
        lost = _run()["moves"]["lost, game unchanged"]
        assert lost["asked"] == ["POST /api/session/g1/move", "GET /api/session/g1"]
        assert lost["calls"] == [
            ["drawBoard"],
            ["fail", "The site is updating or busy, so that move was not made. "
                     "Try it again in a moment.", None]]

    def test_a_lost_answer_on_a_game_that_moved_on_draws_the_game_and_says_nothing(
        self,
    ) -> None:
        lost = _run()["moves"]["lost, game moved on"]
        assert lost["asked"] == ["POST /api/session/g1/move", "GET /api/session/g1"]
        assert lost["calls"] == [["applySnapshot", 3]]

    def test_a_game_that_cannot_be_read_either_says_only_what_is_known(self) -> None:
        """Whether the move landed is not known, so the sentence does not say."""
        lost = _run()["moves"]["lost, game unreadable"]
        assert lost["asked"] == ["POST /api/session/g1/move", "GET /api/session/g1",
                                 "GET /api/session/g1"]
        assert lost["calls"] == [["drawBoard"], ["fail", SITE_SAID, None]]


class TestOneHelper:
    def test_only_the_helper_reads_a_json_body(self) -> None:
        """Every `.json()` in the page is `askOnce`'s, except the measuring view's
        manifest, which checks `ok`, catches, and shows nothing for an unreadable one."""
        from page_text import PAGE

        assert PAGE.count(".json();") == 2
        start = PAGE.index("async function askOnce(")
        assert ".json();" in PAGE[start:PAGE.index("function unansweredWords(", start)]
        start = PAGE.index("async function hereManifest(")
        body = PAGE[start:PAGE.index("\n}\n", start)]
        assert ".json();" in body and "catch (error)" in body and "!response.ok" in body

    def test_every_other_request_is_the_helpers(self) -> None:
        from page_text import PAGE

        script = re.search(r"<script>(.*)</script>", PAGE, re.S)
        assert script is not None
        found = re.findall(r"\bfetch\(", script.group(1))
        # `askOnce` and the manifest.
        assert len(found) == 2
