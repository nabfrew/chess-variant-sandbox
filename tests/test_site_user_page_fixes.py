"""The user page's fixes from its cold read (UP7, `docs/reference/user-page-cold-read.md` §3).

Four findings marked *fix*, one class each:

1. *open in the designer* on your own draft opened a copy of it (*two stones variant*, its
   description emptied). It opens the draft as itself now, and a save keeps its name, its
   description and its hash; a library game or somebody else's still opens as a copy.
2. The draft's next step, a score measured on your computer, was a sentence; it is a link
   that opens the draft at the designer's *measure it here*, scrolled to and focused, with
   one line saying measuring takes a few minutes and is easier on a computer.
3. *Your game on trial* over one of two games on trial says *One of your 2 games on trial*.
4. `/profile` held its frame back only after it had drawn it: the frame now waits for
   `/api/profile`, so nothing shifts under a thumb.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from typing import Any

import node_session
import page_text
import pytest
from test_site import _DOM, _in_node, _rendered, _script_of
from test_site_admin import DULL, DULL_HASH, Site, _saved, _screened, drafting
from test_site_correspondence import app, store

from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import stages, workshop

__all__ = ["app", "drafting", "store"]


def _kept(level: str, reading: str = "none", screened: bool = True) -> workshop.Kept:
    return workshop.Kept(hash="h" + level[:3], name=level, spec={}, parent="", screened=screened,
                         level=level, waking=False, reading=reading)


@pytest.fixture
def workshop_site(drafting: Site, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """The drafting site with `dull`'s row, so a draft made from it keeps its parent."""
    _screened(monkeypatch, passed=True)
    drafting.game()
    yield drafting


def _adas_draft() -> dict[str, Any]:
    spec = copy.deepcopy(DULL)
    spec["name"] = "two stones"
    spec["description"] = "Ada's own words for it."
    spec["rules"]["max_plies"] = 80
    return spec


# The designer's whole script in node, every question it asks answered from `ANSWERS` (what
# the site said to the same reader), as `test_design_doors.py` runs a door.
_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.history = { replaceState: () => {} };
  global.fetch = () => new Promise(() => {});
"""

_OPEN = """
api = async (path) => {
  if (!Object.prototype.hasOwnProperty.call(ANSWERS, path)) throw new Error('unasked ' + path);
  return JSON.parse(JSON.stringify(ANSWERS[path]));
};
(async () => {
  await openGame(FROM, AS_DRAFT);
  stampParent();
  // What `saveDraft` sends, and whether it would first ask for a name of its own.
  console.log(JSON.stringify({ spec: DRAFT, parent: DRAFT_FROM, own: DRAFT_OWN,
                               asks: borrowedName(), heldParent: (DRAFT_PARENT || {}).hash }));
})();
"""


def _opened(answers: dict[str, Any], digest: str, as_draft: bool,
            library: list[dict[str, Any]]) -> dict[str, Any]:
    """The designer's draft after `#/design/<digest>` (`?as=draft` when `as_draft`)."""
    script = re.search(r"<script>(.*)</script>", page_text.PAGE, re.S)
    assert script is not None
    body = (_STUB + script.group(1) + "\n"
            + f"LIBRARY = {json.dumps(library)};\n"
            + f"const ANSWERS = {json.dumps(answers)};\n"
            + f"const FROM = {json.dumps(digest)}; const AS_DRAFT = {json.dumps(as_draft)};\n"
            + _OPEN)
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return dict(json.loads(done.stdout))


class TestYourOwnDraftOpensAsItself:
    """Finding 1."""

    def test_the_server_says_whose_the_rules_are(self, workshop_site: Site) -> None:
        ada = workshop_site.signed_in("ada")
        bob = workshop_site.signed_in("bob")
        digest = _saved(ada, _adas_draft(), parent=DULL_HASH).payload["hash"]
        assert ada.get(f"/api/design/{digest}").payload["yours"] == {
            "name": "two stones", "description": "Ada's own words for it.",
            "parent": DULL_HASH, "restates": True,
        }
        # Somebody else's draft, and a game nobody on the site owns, are not yours.
        assert bob.get(f"/api/design/{digest}").payload["yours"] is None
        assert ada.get(f"/api/design/{DULL_HASH}").payload["yours"] is None

    def test_a_save_keeps_the_drafts_name_description_and_identity(
        self, workshop_site: Site
    ) -> None:
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _adas_draft(), parent=DULL_HASH).payload["hash"]
        answers = {f"/api/design/{digest}": ada.get(f"/api/design/{digest}").payload,
                   f"/api/design/{DULL_HASH}": ada.get(f"/api/design/{DULL_HASH}").payload}
        # The library lists the draft under its own name, which is not a borrowed one.
        library = [{"hash": digest, "name": "two stones"}, {"hash": DULL_HASH, "name": "dull"}]
        opened = _opened(answers, digest, True, library)

        assert opened["spec"]["name"] == "two stones"
        assert opened["spec"]["description"] == "Ada's own words for it."
        assert opened["own"] == digest
        assert opened["parent"] == DULL_HASH and opened["heldParent"] == DULL_HASH
        assert opened["spec"]["provenance"]["parents"] == [DULL_HASH]
        assert opened["asks"] == ""

        saved = _saved(ada, opened["spec"], parent=opened["parent"]).payload
        assert saved["hash"] == digest
        assert saved["created"] is False and saved["restated"] == []
        assert saved["name"] == "two stones"
        assert ada.get(f"/api/design/{digest}").payload["yours"] == {
            "name": "two stones", "description": "Ada's own words for it.",
            "parent": DULL_HASH, "restates": True,
        }
        # One game, made from dull, on the profile: no copy came of it.
        games = [game for group in ada.get("/api/profile").payload["yourGames"]
                 for game in group["games"]]
        assert [game["hash"] for game in games] == [digest]
        assert games[0]["parent"]["hash"] == DULL_HASH

        # New words on the same rules are a re-save of the same game (Q595), not a copy that
        # changed no rule, since its parent is still the game it was made from.
        worded = {**opened["spec"], "description": "Better words."}
        again = _saved(ada, worded, parent=opened["parent"]).payload
        assert again["hash"] == digest and again["restated"] == ["description"]

    def test_a_library_game_or_somebody_elses_draft_still_opens_as_a_copy(
        self, workshop_site: Site
    ) -> None:
        ada = workshop_site.signed_in("ada")
        bob = workshop_site.signed_in("bob")
        digest = _saved(ada, _adas_draft(), parent=DULL_HASH).payload["hash"]
        library = [{"hash": digest, "name": "two stones"}, {"hash": DULL_HASH, "name": "dull"}]
        theirs = {f"/api/design/{digest}": bob.get(f"/api/design/{digest}").payload}
        opened = _opened(theirs, digest, True, library)
        assert opened["spec"]["name"] == "two stones variant"
        assert opened["spec"]["description"] == ""
        assert opened["own"] == "" and opened["parent"] == digest
        dull = {f"/api/design/{DULL_HASH}": ada.get(f"/api/design/{DULL_HASH}").payload}
        forked = _opened(dull, DULL_HASH, True, library)
        assert forked["spec"]["name"] == "dull variant" and forked["parent"] == DULL_HASH
        # Your own draft asked for without `?as=draft` (*Try another change*) is a copy too.
        own = {f"/api/design/{digest}": ada.get(f"/api/design/{digest}").payload}
        assert _opened(own, digest, False, library)["spec"]["name"] == "two stones variant"

    def test_the_workshop_opens_your_draft_as_itself(self) -> None:
        drawn = _rendered(
            f"drawYourGames({json.dumps([{'level': 'Draft', 'games': [GAME]}])},"
            f" {json.dumps(COLUMNS)}, true);"
            "const links = [];"
            "(function walk(node) {"
            "  if (node.href !== undefined) links.push([node.href, node.textContent]);"
            "  (node.children || []).forEach(walk);"
            "})(document.getElementById('mine'));"
            "console.log(JSON.stringify(links));"
        )
        assert ["/#/design/aaaa?as=draft", "open in the designer"] in drawn
        # *Try another change* is still the way to a copy of the game it was made from.
        assert ["/#/design/pppp", "three stones"] in drawn

    def test_the_router_hands_the_designer_its_question(self) -> None:
        assert ("showDesigner(parts[1] || '', routeQuery.get('rule'),\n"
                "      { as: routeQuery.get('as'), at: routeQuery.get('at') })") in page_text.PAGE


GAME = {"hash": "aaaa", "name": "two stones", "inDesigner": True, "tagCounts": [],
        "still": None, "column": "Draft",
        "step": "Next: a score measured on your computer, from the designer.",
        "stepAt": workshop.MEASURE, "stepNote": workshop.MEASURE_NOTE,
        "change": ["Rule changed: three is now two."],
        "rungs": [{"key": "validate", "name": "complete", "state": "done", "says": "yes",
                   "next": False}],
        "parent": {"hash": "pppp", "name": "three stones", "level": "Library", "score": None}}
COLUMNS = [{"word": word, "says": "its line", "games": ["aaaa"] if word == "Draft" else []}
           for word in workshop.COLUMNS]


class TestTheNextStepIsALinkToItsControl:
    """Finding 2."""

    def test_only_a_draft_waiting_on_its_score_is_taken_at_the_measuring_control(self) -> None:
        said = stages.NEXT[stages.DRAFT]
        assert workshop.step_at(_kept(stages.DRAFT), said) == workshop.MEASURE
        assert workshop.step_at(_kept(stages.DRAFT, screened=False), said) == ""
        assert workshop.step_at(_kept(stages.DRAFT, "checking"), said) == ""
        assert workshop.step_at(_kept(stages.DRAFT, "checked"), said) == ""
        assert workshop.step_at(_kept(stages.DRAFT), stages.WAITING) == ""
        assert workshop.step_at(_kept(stages.ON_TRIAL), stages.NEXT[stages.ON_TRIAL]) == ""

    def test_the_profile_sends_where_and_the_one_line(self, workshop_site: Site) -> None:
        ada = workshop_site.signed_in("ada")
        digest = _saved(ada, _adas_draft(), parent=DULL_HASH).payload["hash"]
        game = next(game for group in ada.get("/api/profile").payload["yourGames"]
                    for game in group["games"] if game["hash"] == digest)
        assert game["stepAt"] == workshop.MEASURE
        assert game["stepNote"] == workshop.MEASURE_NOTE
        assert "few minutes" in game["stepNote"] and "easier on a computer" in game["stepNote"]

    def test_the_card_links_the_step_to_the_measuring_control_with_its_line(self) -> None:
        drawn = _rendered(
            f"drawYourGames({json.dumps([{'level': 'Draft', 'games': [GAME]}])},"
            f" {json.dumps(COLUMNS)}, true);"
            "const found = [];"
            "(function walk(node) {"
            "  if (node.className === 'stepGo') found.push([node.href, node.textContent]);"
            "  if (node.className === 'under') found.push(['', node.textContent]);"
            "  (node.children || []).forEach(walk);"
            "})(document.getElementById('mine'));"
            "console.log(JSON.stringify(found));"
        )
        assert drawn == [["/#/design/aaaa?as=draft&at=measure", GAME["step"]],
                         ["", workshop.MEASURE_NOTE]]

    def test_a_step_taken_elsewhere_stays_a_sentence(self) -> None:
        other = {**GAME, "step": "Next: publish it.", "stepAt": "", "stepNote": ""}
        drawn = _rendered(
            f"drawYourGames({json.dumps([{'level': 'Draft', 'games': [other]}])},"
            f" {json.dumps(COLUMNS)}, true);"
            "const mine = document.getElementById('mine');"
            "const hrefs = [];"
            "(function walk(node) {"
            "  if (node.href !== undefined) hrefs.push(node.href);"
            "  (node.children || []).forEach(walk);"
            "})(mine);"
            "console.log(JSON.stringify([mine.textContent.includes('Next: publish it.'),"
            " hrefs.some(h => h.includes('at=measure'))]));"
        )
        assert drawn == [True, False]

    def test_the_designer_reaches_the_control_once_it_stands_where_it_stays(self) -> None:
        script = re.search(r"<script>(.*)</script>", page_text.PAGE, re.S)
        assert script is not None
        runner = """
          const did = [];
          const listened = {};
          window.innerHeight = 800;
          window.addEventListener = (kind, call) => { listened[kind] = call; };
          const made = (name, top) => ({ name, top,
            getBoundingClientRect() { return { top: this.top, bottom: this.top + 40 }; },
            scrollIntoView(how) { did.push(['scroll', name, how.block]); this.top = 380; },
            focus() { did.push(['focus', name]); document.activeElement = this; } });
          let shown = {};
          document.getElementById = id => shown[id] || null;
          DRAFT_AT = 'measure'; DRAFT_HASH = 'h';
          // Before the standing for this draft: the control may yet move into *Publish*.
          shown = { dHereGo: made('first', 6000) }; DRAFT_STANDING = null; reachAsked();
          const before = did.length;
          DRAFT_STANDING = { hash: 'h' }; reachAsked();
          // The panel drawn again: a new button, the focus gone with the old one. It is
          // given the focus back where it stands, and scrolled to only once it is away.
          shown = { dHereGo: made('again', 380) }; document.activeElement = null; reachAsked();
          shown = { dHereGo: made('moved', 1900) }; reachAsked();
          const held = DRAFT_AT;
          // A touch lets go: nothing more is scrolled or focused.
          listened.pointerdown();
          shown = { dHereGo: made('after', 5000) }; document.activeElement = null; reachAsked();
          console.log(JSON.stringify({ before, did, held, at: DRAFT_AT,
                                       kinds: Object.keys(listened).sort() }));
        """
        done = node_session.run(_STUB + script.group(1) + "\n" + runner)
        assert done.returncode == 0, done.stderr
        found = json.loads(done.stdout)
        assert found == {
            "before": 0,
            "did": [["scroll", "first", "center"], ["focus", "first"],
                    ["focus", "again"],
                    ["scroll", "moved", "center"], ["focus", "moved"]],
            "held": "held", "at": "",
            "kinds": ["keydown", "pointerdown", "touchstart", "wheel"],
        }

    def test_the_control_is_asked_for_where_it_is_wired(self) -> None:
        body = page_text.PAGE[page_text.PAGE.index("function wireHere()"):]
        body = body[: body.index("\n}\n")]
        assert "reachAsked()" in body
        assert "DRAFT_AT = asked.at === 'measure' ? 'measure' : '';" in page_text.PAGE


class TestTheKickerSaysWhenItIsOneOfSeveral:
    """Finding 3."""

    def test_one_game_on_trial_is_your_game_on_trial(self) -> None:
        assert workshop.featured(stages.ON_TRIAL, 1) == "Your game on trial"

    def test_two_are_one_of_your_two(self) -> None:
        assert workshop.featured(stages.ON_TRIAL, 2) == "One of your 2 games on trial"
        assert workshop.featured(stages.LIBRARY, 3) == "One of your 3 games in the library"
        assert workshop.featured(stages.SHORTLISTED, 2) == "One of your 2 games shortlisted"
        assert workshop.featured(stages.DRAFT, 4) == ""

    def test_the_columns_carry_the_counted_kicker(self) -> None:
        groups = [{"games": [{"hash": "a", "column": stages.ON_TRIAL},
                             {"hash": "b", "column": stages.ON_TRIAL},
                             {"hash": "c", "column": stages.LIBRARY}]}]
        kickers = {column["word"]: column["feature"] for column in workshop.columns(groups)}
        assert kickers[stages.ON_TRIAL] == "One of your 2 games on trial"
        assert kickers[stages.LIBRARY] == "Your game in the library"


class TestTheFrameWaitsForItsPayload:
    """Finding 5 (the fourth fix)."""

    def test_the_frame_is_drawn_held_back_with_one_line_in_its_place(self) -> None:
        page = site_pages.profile("ada")
        assert '<div id="own-view" class="waits">' in page
        assert 'id="own-wait" role="status">Reading your page…</p>' in page
        # Held back only where a script will bring it in; without one it is drawn at once.
        assert ".js #own-view.waits > * { display: none; }" in site_pages._OWN
        assert "#own-wait { display: none; }" in site_pages._OWN

    def test_the_frame_comes_in_once_the_payload_has_drawn_or_failed(self) -> None:
        script = _script_of(page_text.site_own(site_pages.profile("ada")))
        seeded = ("BY_ID['own-view'] = Object.assign(element('div'), {classList: {gone: [],"
                  " remove(name) { this.gone.push(name); }}});\n")
        found = _in_node(
            _DOM + seeded + script,
            "setTimeout(() => console.log(JSON.stringify("
            "BY_ID['own-view'].classList.gone)), 20);",
        )
        assert found == ["waits"]
        assert "draw().catch(" in script and ".finally(frameDrawn);" in script
