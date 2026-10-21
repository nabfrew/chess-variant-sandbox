"""Q642: a new game started from an existing one.

The game page's *make your own from this* and the designer's *start again from* a library
game. The button had been in the game page's left column since the designer arrived, and
from Q256 C6 (the play column first on a phone) it sat some ten thousand pixels down a
phone's page; the chooser had only ever listed templates.

Four claims. **The button** is above both columns, on the site and locally, and opens the
designer on that game. **The copy** is that game's rules under a new name that no library
game has, with an empty description and `provenance.parents` naming the game it came
from; the answer it was made from is untouched, and its mechanics hash (worked out by the
existing Python) is the original's until a rule changes. **The chooser** keeps the
templates first and the first template the default, and lists every library game after
them. **The save** says in `provenance.parents` what the draft's parent is at the save,
whatever undo left in the draft.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_web import _game_payload, _page_on
from test_web import answers as answers  # noqa: F401  (the fixture, used by name)

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.web import designing

HASH = "abc123def456"
PARENT = "0123456789ab"
LANDED = 'data-rung="save" data-state="done"'


def _parent_spec() -> dict[str, Any]:
    """The game being copied: a template's rules under a library name."""
    spec = copy.deepcopy(designing.templates()[1]["spec"])
    spec["name"] = "a parent game"
    spec["description"] = "the parent's own sentence"
    spec["provenance"] = {"origin": "human", "author": "somebody", "parents": ["ffffffffffff"]}
    return spec


def _library(*games: tuple[str, str]) -> dict[str, Any]:
    return {
        "games": [{"hash": digest, "name": name} for digest, name in games],
        "opponents": _game_payload(HASH, site=False)["opponents"],
        "coverage": {"axes": []}, "metrics": None,
    }


def _with(answers: dict, **extra: Any) -> dict:
    spec = _parent_spec()
    return {
        **answers,
        "/api/library": _library(
            (PARENT, "a parent game"), ("bbbbbbbbbbbb", "A parent game variant"),
            ("cccccccccccc", "an unrelated game"),
        ),
        f"/api/design/{PARENT}": {"spec": spec, "name": spec["name"], "hash": PARENT},
        **extra,
    }


def _checked(seen: dict) -> dict:
    """The last draft the designer sent to be checked."""
    bodies = [body for path, body in seen["sent"] if path == "/api/design/check"]
    assert bodies, seen["fetched"]
    return bodies[-1]["spec"]


class TestTheButton:
    @pytest.mark.parametrize("site", [True, False])
    def test_it_is_above_both_columns_and_opens_the_designer(self, site: bool) -> None:
        seen = _page_on(site, f"#/game/{HASH}", _game_payload(HASH, site=site),
                        press="document.getElementById('fork').onclick();"
                              "window.__scrolled.push(location.hash);")
        assert seen["errors"] == []
        html = seen["html"]
        assert html.count('id="fork"') == 1
        assert ">make your own from this</button>" in html
        # Above the two columns, so a phone, which puts the play column first, still
        # shows it under the name rather than at the end of the page.
        assert html.index('id="fork"') < html.index('class="gameTop"')
        assert seen["scrolled"][-1] == f"#/design/{HASH}"


class TestTheCopy:
    def test_the_copy_is_the_games_rules_under_a_new_name(self, answers) -> None:  # noqa: F811
        seen = _page_on(False, f"#/design/{PARENT}", _game_payload(HASH, site=False),
                        _with(answers), landed=LANDED,
                        press=f"window.__scrolled.push(__answers['/api/design/{PARENT}']);")
        assert seen["errors"] == []
        draft = _checked(seen)
        parent = _parent_spec()
        # A name no library game has: *a parent game variant* is taken, in another case.
        assert draft["name"] == "a parent game variant 2"
        assert draft["description"] == ""
        # The parent's own provenance (its author, its parents) is not this draft's.
        assert draft["provenance"] == {"parents": [PARENT]}
        # Rules unchanged, so the same game by the existing hash until a rule moves.
        assert spec_hash(draft) == spec_hash(parent)
        assert validate_spec(draft) == []
        # The answer the copy was made from is as it came.
        assert seen["scrolled"][-1]["spec"] == parent
        assert "Started from <b>a parent game</b>" in seen["html"]

    def test_a_copy_of_any_template_validates_and_keeps_its_hash(self) -> None:
        """The draft's shape, as the schema reads it, for every starting point the
        designer offers before a library is in hand."""
        for template in designing.templates():
            spec = template["spec"]
            draft = copy.deepcopy(spec)
            draft.update(name=spec["name"] + " variant", description="",
                         provenance={"parents": [PARENT]})
            assert validate_spec(designing.settled(draft)) == [], template["key"]
            assert spec_hash(draft) == spec_hash(spec), template["key"]

    def test_the_save_keeps_the_parent_in_the_spec(self, tmp_path: Path) -> None:
        spec = copy.deepcopy(designing.templates()[0]["spec"])
        spec["provenance"] = {"parents": [PARENT]}
        kept = designing.save(spec, tmp_path, parent=PARENT, parent_spec=_parent_spec())
        written = json.loads(Path(kept["path"]).read_text(encoding="utf-8"))
        assert written["provenance"] == {"parents": [PARENT]}
        assert kept["hash"] == spec_hash(spec)


class TestTheChooser:
    def test_templates_stay_first_and_the_default(self, answers) -> None:  # noqa: F811
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False),
                        _with(answers), landed=LANDED,
                        press="document.getElementById('dTemplate').onchange("
                              f"{{ target: {{ value: 'game:{PARENT}' }} }});"
                              "window.__scrolled.push(location.hash);")
        assert seen["errors"] == []
        every = seen["every"]
        templates = every.index('<optgroup label="templates">')
        games = every.index('<optgroup label="games in the library">')
        assert templates < games
        for key in [t["key"] for t in designing.templates()]:
            assert templates < every.index(f'<option value="{key}"') < games, key
        # Every library game, by name.
        order = [every.index(f'value="game:{digest}"')
                 for digest in (PARENT, "bbbbbbbbbbbb", "cccccccccccc")]
        assert all(games < at for at in order)
        assert order == sorted(order)
        # The default draft is still the first template.
        assert _checked(seen)["name"] == designing.templates()[0]["spec"]["name"]
        # Choosing a game opens it the way its page's button does.
        assert seen["scrolled"][-1] == f"#/design/{PARENT}"

    def test_with_no_library_there_are_templates_alone(self, answers) -> None:  # noqa: F811
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False),
                        {**answers, "/api/library": _library()}, landed=LANDED)
        assert seen["errors"] == []
        assert '<optgroup label="templates">' in seen["every"]
        assert "games in the library" not in seen["every"]


class TestTheSave:
    def test_the_save_says_the_parent_the_draft_has_now(self, answers) -> None:  # noqa: F811
        """Undo restores a whole draft and not the game it started from, so `parents`
        can be stale in the draft. The save puts it right, and keeps what else a person
        wrote in `provenance`."""
        press = (
            # The harness's window draws nothing, and a save's answer asks how it is drawn.
            "window.getComputedStyle = () => ({ position: 'static' });"
            # The default draft is the first template, still under its name, and a save
            # under a template's name asks for one of its own first (Q995c).
            "DRAFT.name = 'a game of my own';"
            "DRAFT_FROM = '';"
            "DRAFT.provenance = { parents: ['ffffffffffff'], author: 'me' };"
            "saveDraft();"
            "DRAFT_FROM = 'aaaaaaaaaaaa'; DRAFT.provenance = { parents: ['ffffffffffff'] };"
            "saveDraft();"
            "DRAFT_FROM = ''; DRAFT.provenance = { parents: ['ffffffffffff'] };"
            "saveDraft();"
        )
        seen = _page_on(False, "#/design", _game_payload(HASH, site=False), answers,
                        landed=LANDED, press=press)
        saves = [body for path, body in seen["sent"] if path == "/api/design/save"]
        assert len(saves) == 3
        assert saves[0]["spec"]["provenance"] == {"author": "me"}
        assert saves[1]["spec"]["provenance"] == {"parents": ["aaaaaaaaaaaa"]}
        assert saves[1]["parent"] == "aaaaaaaaaaaa"
        assert "provenance" not in saves[2]["spec"]
