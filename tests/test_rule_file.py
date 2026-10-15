"""Q830b: the designer links to the rule language and takes a rule file somebody wrote.

`docs/plan-website.md`'s *LLM co-design* decisions of 2026-10-10: the site offers no AI;
the rule language is `RULES.md`, linked from the designer, for people writing their own
game files by hand or with their own AI, and the designer loads such a file. Since Q905
the link is the running server's own `/rules.md`, rendered from the code it runs, so it
names the schema version the validator reading the file accepts; the public repository's
copy is a snapshot that can lag.

Four claims. **The reading is the server's**: `designing.read_rule_file` holds a file to its
size and depth, parses it, and asks the validator and the compiler the designer's check
asks; a valid file comes back as the settled draft, an invalid one as the validator's own
lines, the ones `validate` prints. **The page decides nothing**: it sends the text, takes
a valid answer as the draft and leaves the draft alone otherwise, and sends no more of a
large file than the refusal needs. **The link is one string**, `reference.PUBLISHED`, sent
with the kit and drawn to open in a new tab, beside the button `RULES.md` names, and both
servers answer it with the reference. **On the site** the route is anybody's, writes
nothing, and sits under the design POSTs' body cap, and `/rules.md` is anybody's too.
"""

from __future__ import annotations

import copy
import json
import re
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest
from test_shell import _get
from test_site_admin import Site, _template
from test_site_admin import drafting as drafting  # noqa: F401  (the fixture, used by name)
from test_web import _game_payload, _page_on
from test_web import answers as answers  # noqa: F401  (the fixture, used by name)

from chess_variant_sandbox.site import quota, routes
from chess_variant_sandbox.spec import reference
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, validate_spec
from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web import server as web_server
from chess_variant_sandbox.web.server import App

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "chess_variant_sandbox"
HASH = "abc123def456"


def _written(index: int = 1, name: str = "a game somebody wrote") -> dict[str, Any]:
    """A game as somebody would write it: a template's rules under their own name."""
    spec = copy.deepcopy(designing.templates()[index]["spec"])
    spec["name"] = name
    spec["description"] = "written by hand"
    return spec


def _text(spec: dict[str, Any]) -> str:
    return json.dumps(spec, indent=2)


def _broken() -> dict[str, Any]:
    spec = _written()
    spec["pieces"][0]["speed"] = 2
    spec["pieces"][0]["moves"][0]["kind"] = "teleport"
    return spec


# ------------------------------------------------------------------- the reading


class TestTheReading:
    def test_a_valid_file_comes_back_as_the_settled_draft(self) -> None:
        spec = _written()
        read = designing.read_rule_file(_text(spec))
        assert read == {"valid": True, "errors": [], "spec": designing.settled(spec)}
        assert validate_spec(read["spec"]) == []

    def test_an_invalid_file_is_refused_in_the_validators_own_lines(self) -> None:
        spec = _broken()
        read = designing.read_rule_file(_text(spec))
        assert read["valid"] is False and "spec" not in read
        # Every line the validator gives, as it gives them: the paths a person who wrote
        # the file can find, not the form's words.
        assert read["errors"] == validate_spec(designing.settled(spec))
        assert any(line.startswith("pieces[0] has unknown keys: speed")
                   for line in read["errors"]), read["errors"]
        assert any("got 'teleport'" in line for line in read["errors"]), read["errors"]

    def test_a_file_over_the_size_is_refused_by_its_size(self) -> None:
        spec = _written()
        spec["description"] = "x" * designing.RULE_FILE_BYTES
        read = designing.read_rule_file(_text(spec))
        assert read["valid"] is False
        assert read["errors"] == [
            f"That is more than {designing.RULE_FILE_BYTES // 1024} KB, and a rule file may "
            f"be at most {designing.RULE_FILE_BYTES // 1024} KB. The largest game in the "
            "library is about 10 KB."
        ]

    def test_a_file_nested_too_deep_is_refused_before_it_is_parsed(self) -> None:
        deep = "[" * 5000 + "]" * 5000
        read = designing.read_rule_file(deep)
        assert read["valid"] is False
        assert read["errors"][0].startswith("That nests 5000 levels deep")

    def test_brackets_inside_a_string_are_not_nesting(self) -> None:
        spec = _written()
        spec["description"] = "[{" * 100 + '\\"' + "}]" * 100
        assert designing.read_rule_file(_text(spec))["valid"] is True

    @pytest.mark.parametrize(
        ("text", "said"),
        [
            ('<svg xmlns="http://www.w3.org/2000/svg"></svg>', "That is not JSON:"),
            ("\x89PNG\r\n\x1a\n\x00\x00", "That is not JSON:"),
            ('{"name": "half a game",', "That is not JSON:"),
            ("[1, 2, 3]", "A rule file is one JSON object, { ... }, and this is a list."),
            ('"a game"', "A rule file is one JSON object, { ... }, and this is a single value."),
            ("   ", "That file is empty."),
        ],
    )
    def test_a_file_of_the_wrong_kind_is_refused_by_what_it_is(
        self, text: str, said: str
    ) -> None:
        read = designing.read_rule_file(text)
        assert read["valid"] is False
        assert read["errors"][0].startswith(said), read["errors"]

    def test_the_parse_error_names_its_line_and_column(self) -> None:
        read = designing.read_rule_file('{\n  "name": "x",\n  oops\n}')
        assert read["errors"] == [
            "That is not JSON: Expecting property name enclosed in double quotes at line 3,"
            " column 3."
        ]

    def test_text_that_is_not_text_is_refused(self) -> None:
        for sent in (None, 5, {"name": "x"}, ["x"]):
            assert designing.read_rule_file(sent) == {
                "valid": False, "errors": ["That file could not be read as text."]}

    def test_a_byte_order_mark_is_not_part_of_the_file(self) -> None:
        assert designing.read_rule_file("﻿" + _text(_written()))["valid"] is True

    def test_every_game_in_the_library_fits_inside_the_limits(self) -> None:
        """The limits are for files nobody wrote by hand, never for a game: every committed
        and archived file is under the size with room to spare, and under the depth."""
        files = sorted((ROOT / "rulesets").glob("*.json")) + sorted(
            (ROOT / "archive").glob("*.json"))
        assert files
        for path in files:
            text = path.read_text(encoding="utf-8")
            assert 3 * len(text.encode("utf-8")) < designing.RULE_FILE_BYTES * 1.2, path.name
            assert designing._nesting(text) <= designing.RULE_FILE_DEPTH // 2, path.name

    def test_a_file_at_the_limit_fits_the_sites_body_cap_as_the_page_sends_it(self) -> None:
        """The page sends the text inside a JSON body, where every quote and line end is
        escaped. A file the load would take must not meet the body cap's 413 first."""
        spec = _written()
        spec["description"] = "x" * (designing.RULE_FILE_BYTES - len(_text(spec)) - 64)
        text = _text(spec)
        assert len(text.encode("utf-8")) <= designing.RULE_FILE_BYTES
        body = json.dumps({"text": text}).encode("utf-8")
        assert len(body) <= quota.DESIGN_BODY_BYTES


class TestTheLocalRoute:
    @pytest.fixture
    def app(self, tmp_path: Path) -> App:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        return App(tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive")

    def test_the_route_reads_a_file_and_writes_nothing(self, app: App, tmp_path: Path) -> None:
        spec = _written()
        assert app.post("/api/design/load", {"text": _text(spec)}) == {
            "valid": True, "errors": [], "spec": designing.settled(spec)}
        refused = app.post("/api/design/load", {"text": _text(_broken())})
        assert refused["valid"] is False and refused["errors"]
        assert not any((tmp_path / "runs").iterdir())

    def test_the_kit_carries_the_link_and_the_controls_names(self, app: App) -> None:
        assert app.get("/api/design")["ruleFile"] == {
            "rulesUrl": reference.PUBLISHED,
            "load": reference.LOAD_CONTROL,
            "paste": reference.PASTE_CONTROL,
            "bytes": designing.RULE_FILE_BYTES,
        }

    def test_the_server_serves_the_reference_its_code_renders(self) -> None:
        """Q905: `/rules.md` is `render_reference()` of the running code, as Markdown, and
        its first line names the schema version this validator writes and accepts."""
        ((status, headers, body),) = _get(reference.PUBLISHED)
        assert status == 200
        assert headers["Content-Type"] == "text/markdown; charset=utf-8"
        assert headers["Cache-Control"] == "public, max-age=300"
        assert headers["Content-Security-Policy"] == web_server.CONTENT_SECURITY_POLICY
        text = body.decode("utf-8")
        assert text == reference.render_reference()
        assert text.splitlines()[0] == f"# The rule language, schema version {SCHEMA_VERSION}"


# --------------------------------------------------------------------- the link


class TestTheLink:
    def test_it_is_the_servers_own_route_not_the_public_repositorys_copy(self) -> None:
        """Q905: a path on whichever server drew the page, so on the site it is the site's
        `/rules.md` and on a checkout the checkout's, each the reference for the validator
        behind it. The public repository's copy is an export, and lags until refreshed."""
        assert reference.PUBLISHED == "/rules.md"
        assert web_server.RULES_ROUTE == reference.PUBLISHED
        assert reference.PUBLISHED in routes.STATIC
        assert designing.rule_file_kit()["rulesUrl"] == reference.PUBLISHED

    def test_it_is_spelt_in_one_place(self) -> None:
        """Nowhere in the package but `reference.py`, and the site's table of everything it
        serves (`routes.STATIC`, which imports nothing; the test above holds it to the
        constant). The page draws the kit's and the servers answer the constant. And no
        link to the public repository's copy is left."""
        sources = [path for path in PACKAGE.rglob("*")
                   if path.suffix in {".py", ".js", ".html", ".css"}]
        spelt = [path.relative_to(PACKAGE).as_posix() for path in sources
                 if '"/rules.md"' in path.read_text(encoding="utf-8")]
        # `rglob` lists in the filesystem's order (Windows: sorted; Linux: directory order),
        # so the comparison is of sets of paths, not of a sequence.
        assert sorted(spelt) == ["site/routes.py", "spec/reference.py"]
        linked = [path.relative_to(PACKAGE).as_posix() for path in sources
                  if "chess-variant-sandbox/blob" in path.read_text(encoding="utf-8")]
        assert sorted(linked) == []

    def test_rules_md_names_the_button_and_the_box_the_designer_draws(self) -> None:
        trying = (ROOT / "RULES.md").read_text(encoding="utf-8").split("## Trying a game")[1]
        trying = trying.split("\n## ")[0]
        assert f"press *{reference.LOAD_CONTROL}*" in trying
        assert f"*{reference.PASTE_CONTROL}*" in trying


# --------------------------------------------------------------------- the page


def _kit_answers(answers: dict, **extra: Any) -> dict:  # noqa: F811
    kit = {**answers["/api/design"], "ruleFile": designing.rule_file_kit()}
    return {**answers, "/api/design": kit, **extra}


LANDED = 'id="dLoadFile"'

# A file as the browser hands it over: a name, a size, and `slice` for the part read.
FAKE_FILE = """
  const __file = (name, text, size) => ({
    name, size: size === undefined ? text.length : size,
    text: () => Promise.resolve(text),
    slice: (start, end) => ({ text: () => Promise.resolve(text.slice(start, end)) }),
  });
"""

# Once the load has answered, what the draft is and what the page said, for the report.
REPORT = (".then(() => window.__scrolled.push({ draft: JSON.stringify(DRAFT),"
          " before: window.__before || '',"
          " said: (document.getElementById('dLoadSaid') || {})._html || '',"
          " pasted: (document.getElementById('dPasteSaid') || {})._html || '' }));")


class TestThePage:
    @pytest.mark.parametrize("site", [True, False])
    def test_the_button_and_the_link_are_drawn_from_the_kit(
        self, answers, site: bool  # noqa: F811
    ) -> None:
        seen = _page_on(site, "#/design", _game_payload(HASH, site=site),
                        _kit_answers(answers), landed=LANDED)
        assert seen["errors"] == []
        html = seen["html"]
        assert html.count('id="dLoadFile"') == 1
        assert f">{reference.LOAD_CONTROL}</button>" in html
        link = re.search(r'<a href="([^"]+)" target="_blank" rel="noopener"[^>]*>'
                         r"the rule language</a>", html)
        assert link is not None, html
        assert link.group(1) == reference.PUBLISHED
        assert "by hand, or with your own AI" in html
        # Above the board, where *start from* is.
        assert html.index('id="dLoadFile"') < html.index('class="designer bench"')
        assert f"{reference.PASTE_CONTROL} (JSON)" in seen["every"]

    def test_a_valid_file_becomes_the_draft(self, answers) -> None:  # noqa: F811
        spec = _written()
        loaded = designing.read_rule_file(_text(spec))
        press = (FAKE_FILE + "loadRuleFile(__file('mine.json', "
                 + json.dumps(_text(spec)) + "))" + REPORT)
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False),
                        _kit_answers(answers, **{"/api/design/load": loaded}),
                        landed=LANDED, press=press)
        assert seen["errors"] == []
        sent = [body for path, body in seen["sent"] if path == "/api/design/load"]
        assert sent == [{"text": _text(spec)}]
        after = seen["scrolled"][-1]
        assert json.loads(after["draft"]) == loaded["spec"]
        assert "<b>mine.json</b> is your draft now" in after["said"]

    def test_an_invalid_file_is_refused_in_its_words_and_the_draft_kept(
        self, answers  # noqa: F811
    ) -> None:
        refused = designing.read_rule_file(_text(_broken()))
        press = (FAKE_FILE + "window.__before = JSON.stringify(DRAFT);"
                 "loadRuleFile(__file('mine.json', '{}'))" + REPORT)
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False),
                        _kit_answers(answers, **{"/api/design/load": refused}),
                        landed=LANDED, press=press)
        assert seen["errors"] == []
        after = seen["scrolled"][-1]
        assert after["before"] and after["draft"] == after["before"]
        assert "<b>mine.json</b> was not loaded" in after["said"]
        for line in refused["errors"]:
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(
                ">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
            assert f"<code>{escaped}</code>" in after["said"], line

    def test_a_large_file_is_sent_only_as_far_as_its_refusal_needs(
        self, answers  # noqa: F811
    ) -> None:
        refused = designing.read_rule_file("x" * (designing.RULE_FILE_BYTES + 1))
        press = (FAKE_FILE + "loadRuleFile(__file('huge.json', 'x'.repeat(200000), 50000000))"
                 + REPORT)
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False),
                        _kit_answers(answers, **{"/api/design/load": refused}),
                        landed=LANDED, press=press)
        sent = [body for path, body in seen["sent"] if path == "/api/design/load"]
        assert [len(body["text"]) for body in sent] == [designing.RULE_FILE_BYTES + 1]
        assert "is more than 32 KB" in seen["scrolled"][-1]["said"]

    def test_the_paste_box_goes_through_the_same_reading(self, answers) -> None:  # noqa: F811
        refused = designing.read_rule_file("[1]")
        press = ("document.getElementById('dJson').value = '[1]';"
                 "document.getElementById('dJsonApply').onclick()" + REPORT)
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False),
                        _kit_answers(answers, **{"/api/design/load": refused}),
                        landed=LANDED, press=press)
        sent = [body for path, body in seen["sent"] if path == "/api/design/load"]
        assert sent == [{"text": "[1]"}]
        pasted = seen["scrolled"][-1]["pasted"]
        assert "That text was not loaded" in pasted
        assert "one JSON object" in pasted


# --------------------------------------------------------------------- the site


class TestOnTheSite:
    def test_anybody_may_load_a_file_and_nothing_is_kept(self, drafting: Site) -> None:  # noqa: F811
        visitor = drafting.browser()
        spec = _template(1, "a game somebody wrote")
        answer = visitor.post("/api/design/load", {"text": _text(spec)})
        assert answer.status == 200, answer.text
        assert answer.payload == {"valid": True, "errors": [],
                                  "spec": designing.settled(spec)}
        # An ordinary draft in the page: no row, no file, nothing listed.
        assert drafting.count("games") == 0
        assert not any((Path(drafting.app.runs_dir)).rglob("*.json"))

    def test_an_invalid_file_is_refused_in_the_validators_lines(
        self, drafting: Site  # noqa: F811
    ) -> None:
        member = drafting.signed_in("ada")
        answer = member.post("/api/design/load", {"text": _text(_broken())})
        assert answer.status == 200
        assert answer.payload["valid"] is False
        assert answer.payload["errors"] == validate_spec(designing.settled(_broken()))
        assert drafting.count("games") == 0

    def test_a_file_over_the_limit_is_refused_by_the_load(self, drafting: Site) -> None:  # noqa: F811
        answer = drafting.browser().post(
            "/api/design/load", {"text": "x" * (designing.RULE_FILE_BYTES + 1)})
        assert answer.status == 200
        assert answer.payload["errors"][0].startswith("That is more than 32 KB")

    def test_a_body_over_the_design_cap_is_refused_before_it_is_read(
        self, drafting: Site  # noqa: F811
    ) -> None:
        answer = drafting.browser().post(
            "/api/design/load", {"text": "x" * quota.DESIGN_BODY_BYTES})
        assert answer.status == 413
        assert f"at most {quota.DESIGN_BODY_BYTES // 1024} KB" in answer.payload["error"]

    def test_the_site_serves_its_own_reference_to_a_visitor(
        self, drafting: Site  # noqa: F811
    ) -> None:
        """Q905: no cookie, no account; the reference of the code the site runs, as
        Markdown, cached briefly, under the one policy, and nothing written."""
        connection = HTTPConnection("127.0.0.1", drafting.port, timeout=30)
        try:
            connection.request("GET", reference.PUBLISHED)
            found = connection.getresponse()
            body = found.read()
        finally:
            connection.close()
        assert found.status == 200
        assert found.headers["Content-Type"] == "text/markdown; charset=utf-8"
        assert found.headers["Cache-Control"] == "public, max-age=300"
        assert found.headers.get_all("Content-Security-Policy") == [
            web_server.CONTENT_SECURITY_POLICY]
        assert found.headers.get("Set-Cookie") is None
        text = body.decode("utf-8")
        assert text == reference.render_reference()
        assert text.splitlines()[0] == f"# The rule language, schema version {SCHEMA_VERSION}"
        assert drafting.count("games") == 0

    def test_the_sites_kit_carries_the_link(self, drafting: Site) -> None:  # noqa: F811
        kit = drafting.browser().get("/api/design").payload
        assert kit["ruleFile"]["rulesUrl"] == reference.PUBLISHED
        assert kit["ruleFile"]["load"] == reference.LOAD_CONTROL
