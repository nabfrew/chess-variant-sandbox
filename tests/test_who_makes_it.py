"""L20 (`docs/plan-launch.md` R17): the site says who makes it.

A visitor meets one plain sentence on the front page saying that AI agents design, build and
maintain the site under one person's direction; each game's page says whether a person or one
of the project's AI agents designed it, read off the store and the file (`site/made_by.py`),
and says *not recorded* where the file cannot tell; the invitation says it too, and its trial
paragraph is gone. Specs here are inline, never a named game.
"""

from __future__ import annotations

import html
import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from test_web import _in_page

from chess_variant_sandbox.site import made_by as m
from chess_variant_sandbox.site import mail
from chess_variant_sandbox.web import landing

LINK = "https://chess-variant-sandbox.example/join?code=c"


def _visible(markup: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def _made(origin: Any = None, source: str = "committed", **provenance: Any) -> dict[str, str]:
    found: dict[str, Any] = {"source": source, "provenance": {**provenance}}
    if origin is not None:
        found["provenance"]["origin"] = origin
    return m.made_by(found)


class TestWhoDesignedAGame:
    def test_a_design_run_is_an_agent(self) -> None:
        for origin in ("generated", "evolved"):
            assert _made(origin, author="stage 25") == {"kind": m.AGENT, "said": m.DESIGN_RUN_SAID}

    def test_a_game_written_for_the_project_is_an_agents_unless_it_names_a_person(self) -> None:
        for origin in ("designed", "hand", "control"):
            assert _made(origin)["kind"] == m.AGENT
        assert _made("designed", author="the user") == {"kind": m.PERSON, "said": m.OWNER_SAID}

    def test_a_published_game_names_its_people(self) -> None:
        assert _made("human", author="Ann Example")["said"] == "designed by a person, Ann Example"
        assert _made("human", author="A. One and B. Two")["said"] == (
            "designed by people: A. One and B. Two")
        assert _made("human", year="1920s") == {"kind": m.PERSON, "said": m.BEFORE_SAID}

    def test_what_the_file_cannot_tell_is_said_to_be_unrecorded(self) -> None:
        """*human* with no author and no year is on published games and on the project's own
        showcases alike, so the page claims neither."""
        for found in (_made("human"), _made(None), _made("unrecorded"), _made("something")):
            assert found == {"kind": m.UNRECORDED, "said": m.UNRECORDED_SAID}
        assert "a person" in m.UNRECORDED_SAID and "AI agents" in m.UNRECORDED_SAID

    def test_a_members_game_is_a_persons_by_their_handle(self) -> None:
        found = m.made_by({"source": "designed", "designer": {"by": "ada"},
                           "provenance": {"origin": "evolved"}})
        assert found == {"kind": m.PERSON, "said": "designed by a member of this site, ada"}
        assert m.made_by({"source": "designed"})["kind"] == m.UNRECORDED

    def test_the_owner_is_never_named(self) -> None:
        assert "the user" not in _made("designed", author="the user")["said"]


class TestTheFrontPageSaysIt:
    def test_the_sentence_says_agents_build_it_under_one_person(self) -> None:
        for said in ("AI agents", "design, build and maintain", "one person's direction"):
            assert said in landing.MADE_BY

    def test_the_hero_carries_it_under_its_ways_in(self) -> None:
        """Seen without a press: in the hero the script draws and in the one the server draws
        before any script runs."""
        hero = landing.html_page()
        intro = landing.drawn_intro()
        for drawn in (hero, intro):
            ways = drawn.index("landingWays")
            assert drawn.index(html.escape(landing.MADE_BY), ways) > ways

    def test_the_foot_no_longer_leaves_the_agents_out(self) -> None:
        assert "made and run by one person" not in landing.CONTACT_LINE
        assert landing.MADE_BY in landing.CONTACT_LINE
        assert landing.CONTACT in landing.CONTACT_LINE


class TestTheInvitationSaysIt:
    def test_it_says_who_makes_the_site(self) -> None:
        body = " ".join(mail.invitation("bo@example.invalid", LINK, "ada").body.split())
        assert mail.MAKERS in body
        assert "AI agents design, build and maintain the site" in mail.MAKERS

    def test_the_trial_paragraph_is_gone(self) -> None:
        body = mail.invitation("bo@example.invalid", LINK, "ada").body
        assert "trial" not in body.lower() and "rough" not in body

    def test_it_leaves_the_inviters_handle_unexplained(self) -> None:
        """Whether to say the inviter's name is their handle on the site is the owner's
        question (`first-visit-study.md` §6.5 item 7), not this task's."""
        body = mail.invitation("bo@example.invalid", LINK, "ada").body
        assert "handle on the site" not in body and "their handle" not in body


class TestTheGamePageSaysIt:
    QUIET = "globalThis.setTimeout = () => 0; ACCOUNTS = true;"

    def test_under_the_name_an_agents_game_says_so(self) -> None:
        game = {"source": "committed", "madeBy": {"kind": "agent", "said": m.AGENT_SAID}}
        drawn = _in_page(self.QUIET, f"designerByHtml({json.dumps(game)})")
        assert _visible(drawn) == "Designed by one of this project's AI agents"

    def test_under_the_name_an_unrecorded_game_says_nothing(self) -> None:
        game = {"source": "committed", "madeBy": {"kind": "unrecorded", "said": m.UNRECORDED_SAID}}
        assert _in_page(self.QUIET, f"designerByHtml({json.dumps(game)})") == ""

    def test_where_it_came_from_says_it_in_full_and_claims_no_person(self) -> None:
        provenance = {"origin": "human", "source": "a place"}
        game = {"source": "committed", "madeBy": {"kind": "unrecorded", "said": m.UNRECORDED_SAID}}
        drawn = _visible(_in_page(
            self.QUIET, f"provenanceHtml({json.dumps(provenance)}, {json.dumps(game)})"))
        assert "who designed it" in drawn and "Not recorded" in drawn
        assert "invented by a person" not in drawn


class TestTheSiteSendsIt:
    def test_the_game_route_answers_before_the_records_are_stripped(self) -> None:
        """The fast half: `/api/game` puts `madeBy` on the payload and only then strips the
        maintainer's records, so a design run's `author` is read and never sent."""
        import inspect

        from chess_variant_sandbox.site.server import SiteApp

        source = inspect.getsource(SiteApp.get)
        route = source.index('parts[:2] == ("api", "game")')
        said = source.index('found["madeBy"] = made_by.made_by(found)', route)
        assert route < said < source.index("return _without_records(found)", route)

    @pytest.mark.slow
    def test_the_game_payload_carries_who_designed_it(self, tmp_path: Path) -> None:
        """Against the running site: a design run's file and a file naming its author, found
        by their provenance rather than by name, served from a library of those two alone."""
        from test_site import RULESETS, _running_site

        from chess_variant_sandbox.spec import spec_hash

        found: dict[str, str] = {}
        for directory in ("rulesets", "archive", "runs"):
            (tmp_path / directory).mkdir()
        for path in sorted(RULESETS.glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            provenance = spec.get("provenance") or {}
            kind = ("agent" if provenance.get("origin") == "generated" else
                    "person" if provenance.get("origin") == "human" and provenance.get("author")
                    else None)
            if kind and kind not in found:
                found[kind] = spec_hash(spec)
                (tmp_path / "rulesets" / path.name).write_bytes(path.read_bytes())
        assert set(found) == {"agent", "person"}
        with _running_site(tmp_path) as site:
            for kind, digest in found.items():
                payload = site.browser().get(f"/api/game/{digest}").payload
                assert payload["madeBy"]["kind"] == kind, kind
                assert unescape(payload["madeBy"]["said"]).startswith("designed by")
            # A design run's internal author is still not sent.
            agent = site.browser().get(f"/api/game/{found['agent']}").payload
            assert "author" not in agent["provenance"]
