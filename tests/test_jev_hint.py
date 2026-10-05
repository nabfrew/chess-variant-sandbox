"""The optional description hint (Q396): what it asks, and that nothing relies on it.

`chess_variant_sandbox/web/jev_hint.py` asks TypeSafe's Jev which sentences of a saved game's
description its rules seem not to support. No test here reaches the service. The service
is a local HTTP server that answers the way the real one does, so the module's own client
-- the request it builds, the timeout, the status it refuses -- is what runs, and only the
address is swapped.

What is held down:
- With no key there is no hint, no request, and the save is the save it always was.
- A stubbed service's answer becomes the sentences the page names, and a second ask of
  the same saved description is answered from the stored file.
- An error and a timeout are no hint, and are counted.
- Past the day's cap nothing is sent.
- The page draws a save the same way whether the hint answers, fails or is gone.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import node_session
import pytest

from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import designing, jev_hint
from chess_variant_sandbox.web.server import App

REPOSITORY = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "Six soldiers a side on a six by six board. Soldiers may step backwards to take. "
    "It was invented on a wet afternoon."
)
FALSE = "Soldiers may step backwards to take."


def _archived(name: str) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(
        (REPOSITORY / "archive" / f"{name}.json").read_text(encoding="utf-8"))
    return spec


def _draft(description: str = DESCRIPTION, width: int = 6) -> dict[str, Any]:
    spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
    spec["name"] = "a draft with a description"
    spec["description"] = description
    spec["board"]["width"] = width
    return spec


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(jev_hint.KEY_VAR, raising=False)
    monkeypatch.delenv(jev_hint.CAP_VAR, raising=False)


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jev_hint.KEY_VAR, "test-key-not-a-real-one")
    monkeypatch.delenv(jev_hint.CAP_VAR, raising=False)


@pytest.fixture
def app(tmp_path: Path) -> App:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    return App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")


def _saved(app: App, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(app.post("/api/design/save", {"spec": spec or _draft(), "parent": ""}))


class Service:
    """A local stand-in for TypeSafe's endpoint, speaking its request and answer shapes.

    `mode` is `answer`, `error` (a 500), `refused` (a 401) or `slow` (answers after
    `delay` seconds). An answer reads each question's claim out of the state and says
    *contradicts* for the one with "backwards" in it.
    """

    def __init__(self) -> None:
        self.mode = "answer"
        self.delay = 1.0
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                service.requests.append(body)
                service.headers.append(dict(self.headers))
                if service.mode == "slow":
                    time.sleep(service.delay)
                if service.mode in ("error", "refused"):
                    self.send_response(500 if service.mode == "error" else 401)
                    self.end_headers()
                    self.wfile.write(b'{"error": "no"}')
                    return
                payload = json.dumps(service.answer(body)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                # The client may have given up first, which is the timeout test.
                with contextlib.suppress(OSError):
                    self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/systemone"

    @staticmethod
    def answer(body: dict[str, Any]) -> dict[str, Any]:
        claims = body["state"]["claims"]
        answers = {}
        for key in body["questions"]:
            claim = claims[int(key.split("_")[1])]
            wrong = "backwards" in claim
            answers[key] = {
                "type": "choice",
                "choice": "contradicts" if wrong else "agrees",
                "probabilities": {"agrees": 0.08 if wrong else 0.85,
                                  "contradicts": 0.9 if wrong else 0.05,
                                  "not_checkable": 0.02 if wrong else 0.10},
                "confidence": 0.8,
            }
        return {"model": body["model"], "answers": answers,
                "usage": {"input_tokens": 1234, "output_tokens": 0}}


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[Service]:
    stub = Service()
    threading.Thread(
        target=stub.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    ).start()
    monkeypatch.setattr(jev_hint, "URL", stub.url)
    try:
        yield stub
    finally:
        stub.server.shutdown()
        stub.server.server_close()


def _never_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything reaches for the network."""
    import urllib.request

    def refused(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a request was sent")

    monkeypatch.setattr(jev_hint, "_send", refused)
    monkeypatch.setattr(urllib.request, "urlopen", refused)


# ------------------------------------------------------------------------ no key


class TestWithNoKeyNothingHappens:
    def test_the_hint_says_it_is_off_and_sends_nothing(
        self, app: App, no_key: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _never_sent(monkeypatch)
        kept = _saved(app)
        assert app.post("/api/design/hint", {"hash": kept["hash"]}) == {"off": True}
        assert not (tmp_path / "runs" / jev_hint.HINT_DIR).exists()

    def test_the_save_does_not_load_the_module_or_change_its_answer(
        self, app: App, no_key: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The save path never imports the hint, with a key or without one, and says the
        same four things it said before the hint existed."""
        monkeypatch.delitem(sys.modules, "chess_variant_sandbox.web.jev_hint", raising=False)
        plain = _saved(app)
        assert "chess_variant_sandbox.web.jev_hint" not in sys.modules
        assert set(plain) == {"hash", "name", "created", "path"}

        monkeypatch.setenv(jev_hint.KEY_VAR, "test-key-not-a-real-one")
        _never_sent(monkeypatch)
        monkeypatch.delitem(sys.modules, "chess_variant_sandbox.web.jev_hint", raising=False)
        again = _saved(app, _draft(width=7))
        assert "chess_variant_sandbox.web.jev_hint" not in sys.modules
        assert set(again) == set(plain)

    def test_the_privacy_notice_says_nothing_about_it(self, no_key: None) -> None:
        from chess_variant_sandbox.site import pages

        assert "TypeSafe" not in pages.privacy()
        assert jev_hint.privacy_paragraph() == ""


# ------------------------------------------------------------------ a stubbed service


class TestAStubbedServiceAnswers:
    def test_the_contradicted_sentence_is_named(
        self, app: App, keyed: None, service: Service, tmp_path: Path
    ) -> None:
        kept = _saved(app)
        hint = app.post("/api/design/hint", {"hash": kept["hash"]})
        assert hint["hash"] == kept["hash"]
        assert hint["checked"] == 3
        assert [one["claim"] for one in hint["claims"]] == [FALSE]
        assert hint["claims"][0]["contradicts"] == pytest.approx(0.9)

        day = next(iter(jev_hint.calls(tmp_path / "runs").values()))
        assert day["calls"] == 1 and day["tokens"] == 1234 and day["failed"] == 0
        assert day["dollars"] == pytest.approx(1234 * jev_hint.PRICE_PER_TOKEN, abs=1e-8)

    def test_it_asks_claim_by_claim_with_the_counts_and_without_the_description(
        self, app: App, keyed: None, service: Service
    ) -> None:
        kept = _saved(app)
        app.post("/api/design/hint", {"hash": kept["hash"]})
        [request] = service.requests
        assert service.headers[0]["Authorization"] == "Bearer test-key-not-a-real-one"
        assert request["model"] == jev_hint.MODEL
        state = request["state"]
        assert state["claims"] == jev_hint.claims(DESCRIPTION)
        assert set(request["questions"]) == {f"claim_{i}" for i in range(3)}
        for index, question in enumerate(request["questions"].values()):
            assert question["type"] == "choice"
            assert f"`claims[{index}]`" in question["instructions"]
            assert set(question["criteria"]) == {"agrees", "contradicts", "not_checkable"}
        # The rules are blind: neither the description nor the name answers for itself.
        assert "wet afternoon" not in state["rules"]
        assert "a draft with a description" not in state["rules"]
        facts = state["facts"]
        assert facts["first player's army at the start (White, moves first)"] == {
            "each kind": {"soldier": 6}, "in all": 6}
        assert facts["pieces on the board at the start, both sides together"] == 12

    def test_a_second_ask_is_answered_from_the_stored_answer(
        self, app: App, keyed: None, service: Service, tmp_path: Path
    ) -> None:
        kept = _saved(app)
        first = app.post("/api/design/hint", {"hash": kept["hash"]})
        second = app.post("/api/design/hint", {"hash": kept["hash"]})
        assert first == second
        assert len(service.requests) == 1
        stored = json.loads(
            (tmp_path / "runs" / jev_hint.HINT_DIR / f"{kept['hash']}.json").read_text(encoding="utf-8"))
        assert stored["description"] == DESCRIPTION
        assert [row["choice"] for row in stored["claims"]] == [
            "agrees", "contradicts", "agrees"]

    def test_nothing_is_asked_about_a_game_nobody_saved(
        self, app: App, keyed: None, service: Service
    ) -> None:
        unsaved = spec_hash(_draft())
        for asked in (unsaved, "", "../../etc", "ZZZ"):
            assert app.post("/api/design/hint", {"hash": asked}) == {}
        assert service.requests == []

    def test_a_game_with_no_description_is_not_asked_about(
        self, app: App, keyed: None, service: Service
    ) -> None:
        kept = _saved(app, _draft(description=""))
        assert app.post("/api/design/hint", {"hash": kept["hash"]}) == {}
        assert service.requests == []

    def test_the_privacy_notice_says_descriptions_go_to_typesafe(self, keyed: None) -> None:
        from chess_variant_sandbox.site import pages

        notice = pages.privacy()
        assert "TypeSafe" in notice and "description" in notice
        assert "not your handle, your email address or your cookie" in notice


# --------------------------------------------------------------- an error and a timeout


class TestAFailureIsNoHint:
    @pytest.mark.parametrize("mode", ["error", "refused"])
    def test_an_error_is_no_hint_and_is_counted(
        self, app: App, keyed: None, service: Service, tmp_path: Path, mode: str
    ) -> None:
        service.mode = mode
        kept = _saved(app)
        assert app.post("/api/design/hint", {"hash": kept["hash"]}) == {}
        day = next(iter(jev_hint.calls(tmp_path / "runs").values()))
        assert day["calls"] == 1 and day["failed"] == 1 and day["tokens"] == 0
        # Nothing stored, so a later ask tries again rather than showing a failure.
        assert not (tmp_path / "runs" / jev_hint.HINT_DIR / f"{kept['hash']}.json").exists()

    def test_a_timeout_is_no_hint_and_does_not_wait_for_the_service(
        self,
        app: App,
        keyed: None,
        service: Service,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A service ten seconds slow, and a bound of five: a client that waited for the
        # answer fails by a wide margin, and one that gave up at 0.2 s passes by one no
        # busy machine closes. It was 1.5 s and 1.0 s, 0.8 s of slack beside three other
        # test workers (Q377c). The handler thread is a daemon; nothing waits for it.
        service.mode, service.delay = "slow", 10.0
        monkeypatch.setattr(jev_hint, "TIMEOUT", 0.2)
        kept = _saved(app)
        started = time.monotonic()
        assert app.post("/api/design/hint", {"hash": kept["hash"]}) == {}
        assert time.monotonic() - started < 5.0
        day = next(iter(jev_hint.calls(tmp_path / "runs").values()))
        assert day["failed"] == 1

    def test_a_malformed_answer_is_no_hint(
        self, app: App, keyed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(jev_hint, "_send", lambda request, secret: {"answers": {}})
        kept = _saved(app)
        assert app.post("/api/design/hint", {"hash": kept["hash"]}) == {}


# --------------------------------------------------------------------------- the cap


class TestTheDailyCap:
    def test_past_the_cap_nothing_is_sent(
        self,
        app: App,
        keyed: None,
        service: Service,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(jev_hint.CAP_VAR, "1")
        first = _saved(app)
        second = _saved(app, _draft(width=7))
        assert app.post("/api/design/hint", {"hash": first["hash"]})["claims"]
        assert app.post("/api/design/hint", {"hash": second["hash"]}) == {}
        assert len(service.requests) == 1
        day = next(iter(jev_hint.calls(tmp_path / "runs").values()))
        assert day["calls"] == 1 and day["capped"] == 1
        # A stored answer is not a request, so it is still shown past the cap.
        assert app.post("/api/design/hint", {"hash": first["hash"]})["claims"]

    @pytest.mark.parametrize(("stated", "cap"), [
        ("", jev_hint.DEFAULT_CAP), ("7", 7), ("0", 0), ("-3", 0), ("lots", 0), ("2.5", 0)])
    def test_the_cap_reads_a_whole_number_and_anything_else_as_none(
        self, monkeypatch: pytest.MonkeyPatch, stated: str, cap: int
    ) -> None:
        monkeypatch.setenv(jev_hint.CAP_VAR, stated)
        assert jev_hint.daily_cap() == cap

    def test_a_cap_of_zero_switches_the_requests_off(
        self,
        app: App,
        keyed: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(jev_hint.CAP_VAR, "0")
        _never_sent(monkeypatch)
        kept = _saved(app)
        assert app.post("/api/design/hint", {"hash": kept["hash"]}) == {}


# ------------------------------------------------------------------ what is asked


class TestWhatIsAsked:
    def test_a_description_is_split_into_sentences_and_clauses(self) -> None:
        text = ("A game on a torus.  Pawns step one square; kings step one too. "
                "Hi. \"Quoted\" opening sentence here. Numbers 3 of them, 2 more here.")
        assert jev_hint.claims(text) == [
            "A game on a torus.",
            "Pawns step one square",
            "kings step one too.",
            "\"Quoted\" opening sentence here.",
            "Numbers 3 of them, 2 more here.",
        ]

    def test_a_long_description_is_read_from_its_start(self) -> None:
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        found = jev_hint.claims(text)
        assert len(found) == jev_hint.MAX_CLAIMS
        assert found[0] == "Sentence number 0 is here."
        assert all(len(one) <= jev_hint.MAX_CLAIM_CHARS for one in jev_hint.claims("x " * 500))

    def test_the_facts_count_each_side_and_say_who_takes_whom(self) -> None:
        """An inline spec with a restricted capture relation, a piece only one side has
        and a royal piece: the facts are the compiled opening board, counted."""
        spec = _draft()
        spec["schema_version"] = 3
        spec["pieces"] = [
            {**spec["pieces"][0], "id": "S", "name": "soldier", "captures": ["G"]},
            {**spec["pieces"][0], "id": "G", "name": "guard", "captures": []},
            {**spec["pieces"][0], "id": "K", "name": "chief", "royal": True},
        ]
        spec["setup"] = {"mirror": False, "placements": [
            {"piece": "S", "squares": ["a1", "b1", "c1"], "player": 0},
            {"piece": "K", "squares": ["f1"], "player": 0},
            {"piece": "S", "squares": ["a6", "b6", "c6"], "player": 1},
            {"piece": "G", "squares": ["d6"], "player": 1},
            {"piece": "K", "squares": ["f6"], "player": 1},
        ]}
        spec["rules"] = {**spec["rules"], "win": "checkmate"}
        facts = jev_hint.facts(spec)
        assert facts["first player's army at the start (White, moves first)"] == {
            "each kind": {"soldier": 3, "chief": 1}, "in all": 4}
        assert facts["second player's army at the start (Black)"]["in all"] == 5
        assert facts["pieces on the board at the start, both sides together"] == 9
        assert facts["royal pieces (losing them loses the game)"] == ["chief"]
        assert facts["which enemy pieces each piece may take"]["soldier"] == ["guard"]
        assert facts["which enemy pieces each piece may take"]["guard"] == "none"
        assert facts["a player who is checkmated"] == (
            "loses, so checkmating the opponent wins")

    @pytest.mark.parametrize(("stalemate", "said"), [
        ("draw", "draws, so leaving the opponent with no legal move (stalemating them) draws"),
        ("loss", "loses, so leaving the opponent with no legal move (stalemating them) wins"),
        ("win", "wins, so leaving the opponent with no legal move (stalemating them) loses"),
    ])
    def test_the_end_with_no_legal_move_is_said_from_both_sides(
        self, stalemate: str, said: str
    ) -> None:
        """The live check's one false reading: "stalemating the opponent wins" against
        "a player with no legal move loses" read as a contradiction until the facts said
        it from the side of the player who does it."""
        spec = _draft()
        spec["rules"] = {**spec["rules"], "stalemate": stalemate}
        facts = jev_hint.facts(spec)
        assert facts["a player left with no legal move"] == said
        assert "a player who is checkmated" not in facts  # nothing is royal

    @pytest.mark.parametrize(("name", "held"), [("stones", "unlimited"), ("three-stones", 3)])
    def test_a_stock_is_said_as_the_rules_state_it(self, name: str, held: Any) -> None:
        """An unlimited slot is stored as a constant 1, and the facts used to print that
        1 (Q132): *"The supply is unlimited"* read as contradicted at 0.47."""
        facts = jev_hint.facts(_archived(name))
        assert facts["held in hand by each player at the start"] == {"stone": held}

    @pytest.mark.parametrize(("name", "board"), [
        ("hex-glinski-hexagon",
         "a regular hexagon of 91 cells, 6 on a side; every cell touches six others"),
        ("hex-glinski-promoting",
         "hexagonal: 91 cells on a 11 by 11 hex grid, 30 of them walled off so nothing"
         " may stand there"),
    ])
    def test_a_hexagons_corners_are_not_walls(self, name: str, board: str) -> None:
        """Q281b: a native hexagon's absent corners are folded into the compiled walls
        for the tables, and the facts count them as neither walls nor cells; its walled
        twin still says what it always said."""
        assert jev_hint.facts(_archived(name))["board"] == board

    @pytest.mark.parametrize("name", ["stones", "three-stones"])
    def test_a_piece_with_no_capturing_move_is_said_to_take_nothing(self, name: str) -> None:
        """The repaired stones: every capturing piece's list leaves the stone out, and the
        stone's own list is absent -- a permission to take everything that it has no move
        to use. Listing that permission told Jev a stone takes stones (Q132)."""
        takes = jev_hint.facts(_archived(name))["which enemy pieces each piece may take"]
        assert takes["stone"] == jev_hint.NO_CAPTURE
        assert takes["pawn"] == ["pawn", "knight", "bishop", "rook", "queen", "king"]

    def test_with_no_capture_lists_the_piece_with_no_capturing_move_is_still_named(
        self,
    ) -> None:
        """The stones before their repair: no `captures` list anywhere, so every piece may
        take anything -- except that the stone takes nothing, and the facts said
        "every piece" of it too."""
        spec = _archived("stones")
        for piece in spec["pieces"]:
            piece.pop("captures", None)
        takes = jev_hint.facts(spec)["which enemy pieces each piece may take"]
        assert isinstance(takes, dict), takes
        assert takes["stone"] == jev_hint.NO_CAPTURE
        assert "stone" in takes["pawn"]

    def test_the_rules_sent_do_not_say_a_stone_nothing_may_take_is_taken(self) -> None:
        """The third input Q132 found: the rules text itself said the stone stays
        *"until it is taken"*, and with the two facts corrected Jev still sided with it."""
        rules = jev_hint.rules_text(_archived("stones"))
        assert "until it is taken" not in rules
        assert "for the rest of the game" in rules

    @pytest.mark.slow
    def test_every_committed_ruleset_has_facts(self) -> None:
        """About two seconds: a compile of each of the committed rulesets."""
        for path in sorted((REPOSITORY / "rulesets").glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            facts = jev_hint.facts(spec)
            assert facts["pieces on the board at the start, both sides together"] > 0, path


# ------------------------------------------------------------ the call site, disabled


class TestTheCallSiteDisabled:
    """What is left when the hint's route is gone: the save and the page, unchanged."""

    def test_the_save_works_when_the_hint_route_is_gone(
        self, app: App, keyed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = App.post

        def without_the_hint(self: App, path: str, body: dict[str, Any]) -> Any:
            if path.strip("/") == "api/design/hint":
                raise KeyError("no such endpoint: /api/design/hint")
            return original(self, path, body)

        monkeypatch.setattr(App, "post", without_the_hint)
        _never_sent(monkeypatch)
        kept = _saved(app)
        assert kept["created"] is True
        assert set(kept) == {"hash", "name", "created", "path"}
        with pytest.raises(KeyError, match="no such endpoint"):
            app.post("/api/design/hint", {"hash": kept["hash"]})
        assert app.get(f"/api/design/standing/{kept['hash']}")["saved"] is True

    def test_the_route_is_one_call_site_and_the_save_path_never_names_the_module(
        self,
    ) -> None:
        source = (REPOSITORY / "chess_variant_sandbox" / "web" / "server.py").read_text(encoding="utf-8")
        assert source.count("jev_hint") == 2  # the import and the call, in one branch
        save = source.split("def _save_design", 1)[1].split("\n    def ", 1)[0]
        assert "jev_hint" not in save
        assert not re.search(r"^from chess_variant_sandbox\.web import .*jev_hint", source, re.M)


def _page_run(hint: str) -> str:
    """Save a draft on the page, in node, with the hint route answering `hint`.

    `hint` is a JavaScript expression for what `api('/api/design/hint')` returns. The
    save answers, the game's own fetch never does (so the verdict is not redrawn), and
    what is printed is the saved message and how often the hint was asked.
    """
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      global.document = { getElementById: () => el, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {}, innerWidth: 1200,
                        getComputedStyle: () => ({ position: 'static' }) };
      global.location = { pathname: '/game', hash: '' };
      global.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
      global.fetch = () => new Promise(() => {});
    """
    run = """
      let ASKED = 0;
      DRAFT = { name: 'a draft', description: 'x' };
      api = (path, body) => {
        if (path === '/api/design/save') {
          return Promise.resolve({ hash: 'abc123def456', name: 'a draft', created: true,
                                   path: 'runs/designed/specs/abc123def456.json' });
        }
        if (path === '/api/design/hint') { ASKED += 1; return (HINT_ANSWER); }
        return new Promise(() => {});
      };
      (async () => {
        // Not awaited: the save's own chain ends in the game's fetch, which never answers.
        saveDraft();
        await new Promise(resolve => setTimeout(resolve, 20));
        saveDraft();
        await new Promise(resolve => setTimeout(resolve, 20));
        console.log(JSON.stringify({ saved: DRAFT_SAVED, asked: ASKED, off: HINT_OFF }));
      })();
    """.replace("HINT_ANSWER", hint)
    done = node_session.run(stub + script.group(1) + "\n" + run)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


class TestThePage:
    def test_a_save_is_drawn_the_same_when_the_hint_route_is_gone(self) -> None:
        shown = json.loads(_page_run(
            "Promise.reject(new Error('no such endpoint: /api/design/hint'))"))
        assert "Saved" in shown["saved"] and "a draft" in shown["saved"]
        assert "data-hint" not in shown["saved"]
        assert shown["asked"] == 2

    def test_a_server_with_no_key_is_asked_once(self) -> None:
        shown = json.loads(_page_run("Promise.resolve({ off: true })"))
        assert "Saved" in shown["saved"] and "data-hint" not in shown["saved"]
        assert shown["asked"] == 1 and shown["off"] is True

    def test_nothing_to_say_draws_nothing(self) -> None:
        shown = json.loads(_page_run("Promise.resolve({})"))
        assert "Saved" in shown["saved"] and "data-hint" not in shown["saved"]

    def test_a_named_sentence_is_drawn_under_the_save(self) -> None:
        shown = json.loads(_page_run(
            "Promise.resolve({ checked: 3, claims: [{ claim: 'Kings <b>fly</b>.', "
            "contradicts: 0.9 }] })"))
        saved = shown["saved"]
        assert saved.index("Saved") < saved.index('data-hint="description"')
        assert "Kings &lt;b&gt;fly&lt;/b&gt;." in saved
        assert "can be wrong" in saved and "saved either way" in saved
