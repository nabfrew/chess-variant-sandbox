"""The terms for games people make (Q722, `site/terms.py`).

The page is a constant and needs no database; the publish step's acceptance and its record
do, and skip without `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` (`tests/test_site.py` has
the recipe for a scratch cluster).

What is held: the page says each of the brief's points in plain words with the legal
wording beside it and invents no contact address; it is linked from the footer and the
privacy notice; a publish without the box ticked, or agreeing to an older version, is
refused with a sentence and changes nothing; a publish that goes through records when and
which version; games published before the terms carry no record and keep working.
"""

from __future__ import annotations

import re
from typing import Any

import page_text
import pytest
import site_store
import site_words
from test_site_admin import (  # noqa: F401 -- `drafting` is a fixture
    Site,
    _saved,
    _screened,
    _template,
    _user_id,
    drafting,
)

from chess_variant_sandbox.site import drafts, routes, terms
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.web import landing


class TestThePage:
    def test_it_says_every_point_plainly_with_the_legal_wording_beside_it(self) -> None:
        page = site_pages.terms_page()
        assert "<h1>Terms</h1>" in page
        assert f"Version {terms.VERSION}" in page
        keys = [point.key for point in terms.points()]
        assert keys == ["dedication", "yours", "as-is", "members", "takedown", "changes"]
        for point in terms.points():
            section = page[page.index(f'<section class="point" id="{point.key}">'):]
            section = section[:section.index("</section>")]
            assert '<div class="plain">' in section and '<div class="legal">' in section
            assert point.plain and point.legal

    def test_the_dedication_is_cc0_with_its_fallback_and_keeps_nothing_for_the_site(
        self,
    ) -> None:
        dedication, = (p for p in terms.points() if p.key == "dedication")
        assert "CC0 1.0" in dedication.legal and terms.CC0 in dedication.legal
        assert "section 3 of CC0 1.0" in dedication.legal and "Fallback" in dedication.legal
        assert "No right, title, licence or credit" in dedication.legal
        assert "copy, change, share or sell" in dedication.plain
        assert "claims no credit" in dedication.plain
        assert "public domain under CC0 1.0" in terms.DEDICATION
        assert "copy, change, share or sell" in terms.DEDICATION

    def test_the_site_comes_as_is_and_answers_for_nobody(self) -> None:
        said = {point.key: point for point in terms.points()}
        assert '"as is"' in said["as-is"].legal and "without warranty" in said["as-is"].legal
        assert "not responsible for what members make, say or do" in said["members"].plain

    def test_what_a_designer_puts_in_is_theirs_and_a_licence_taken_stays(self) -> None:
        yours = next(point for point in terms.points() if point.key == "yours")
        assert "generally not" in yours.plain
        for thing in ("names", "drawings", "trade marks", "licence"):
            assert thing in yours.plain
        assert "remains subject to its licence" in yours.legal

    def test_the_way_to_ask_for_a_take_down_names_the_contact_and_no_email(self) -> None:
        """Q720: the landing page's contact line, the user's GitHub profile. No email
        address is named anywhere, and none is made up."""
        down = next(point for point in terms.points() if point.key == "takedown")
        assert terms.CONTACT == landing.CONTACT == "https://github.com/nabfrew"
        assert "@" not in down.plain + down.legal
        assert "report a game from its page" in down.plain
        assert f"on GitHub, at {terms.CONTACT_SAID}" in down.plain
        page = site_pages.terms_page()
        assert f'<a href="{terms.CONTACT}" rel="noopener">{terms.CONTACT_SAID}</a>' in page
        assert "@" not in site_words.visible(page)

    def test_naming_the_contact_left_the_legal_wording_of_version_one_alone(self) -> None:
        """The legal wording agreed to under `VERSION` 1 already sent notices to "the
        contact the operator names on this page"; naming it kept those words, so the
        version did not move."""
        down = next(point for point in terms.points() if point.key == "takedown")
        assert terms.VERSION == 1
        assert down.legal.startswith(
            "Notices of claimed infringement or unlawful content may be sent through the "
            "report link on the game's page, or through the contact the operator names on "
            "this page, identifying the material,")

    def test_it_says_nothing_about_legal_advice_or_the_review(self) -> None:
        """The lawyer's list is the user's (`docs/reference/terms-review.md`), not a
        visitor's."""
        page = site_pages.terms_page().lower()
        assert "legal advice" not in page and "lawyer" not in page

    def test_it_is_a_constant_page_in_the_sites_shell_at_a_stable_address(self) -> None:
        assert terms.PATH == "/terms"
        assert terms.PATH in routes.PAGES and terms.PATH not in routes.PERSONAL
        page = site_pages.terms_page()
        assert "/static/site/" not in page and "page-data" not in page, "no script of its own"
        assert '<main class="prose wide">' in page
        assert "@media (max-width: 760px) { .point { grid-template-columns: 1fr; } }" in (
            page_text.inlined(page))

    def test_the_footer_and_the_privacy_notice_link_it(self) -> None:
        for page in (site_pages.privacy(), site_pages.credits(), site_pages.badges(),
                     site_pages.signin()):
            foot = page[page.index("<footer"):]
            assert 'href="/terms">terms</a>' in foot
            assert f'href="{terms.CONTACT}">contact</a>' in foot, "Q720"
        notice = site_pages.privacy()
        body = notice[:notice.index("<footer")]
        assert body.count('href="/terms"') == 2
        assert "which version of the terms you agreed to" in body
        assert "Games published before the terms existed carry no such record" in body
        own = site_pages.terms_page()
        assert 'href="/terms"' not in own[own.index("<footer"):], "not the page you are on"


STAND_IN = site_store.STAND_IN_READING


def _recorded(site: Site, digest: str) -> tuple[Any, Any]:
    with site.store.cursor() as cursor:
        cursor.execute("select terms_accepted, terms_version from games where hash = %s",
                       (digest,))
        found = cursor.fetchone()
    site.store.commit()
    return found[0], found[1]


class TestThePublishStep:
    @pytest.fixture
    def ready(self, drafting: Site, monkeypatch: pytest.MonkeyPatch) -> tuple[Site, Any, str]:  # noqa: F811
        """ada's passed draft, with a reading attached as Q503e will attach one, so the
        terms are the last thing between the press and the trial."""
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        monkeypatch.setattr(drafts, "attached", lambda connection, digest, *rest: STAND_IN)
        return drafting, ada, digest

    def test_a_publish_without_the_box_is_refused_and_nothing_moves(
        self, ready: tuple[Site, Any, str]
    ) -> None:
        site, ada, digest = ready
        for body in ({"hash": digest}, {"hash": digest, "terms": True},
                     {"hash": digest, "terms": str(terms.VERSION)},
                     {"hash": digest, "terms": None}):
            answer = ada.post("/api/design/publish", body)
            assert answer.status == 400, body
            assert answer.payload["error"] == terms.NOT_ACCEPTED
        assert digest not in site.listed()
        assert _recorded(site, digest) == (None, None)

    def test_a_publish_agreeing_to_other_terms_is_told_to_reload(
        self, ready: tuple[Site, Any, str]
    ) -> None:
        site, ada, digest = ready
        answer = ada.post("/api/design/publish", {"hash": digest, "terms": terms.VERSION + 1})
        assert answer.status == 400 and answer.payload["error"] == terms.STALE
        assert digest not in site.listed()

    def test_a_publish_with_the_box_records_when_and_which_version(
        self, ready: tuple[Site, Any, str]
    ) -> None:
        site, ada, digest = ready
        answer = ada.post("/api/design/publish", {"hash": digest, "terms": terms.VERSION})
        assert answer.status == 200, answer.payload
        assert digest in site.listed()
        when, version = _recorded(site, digest)
        assert when is not None and version == terms.VERSION

    def test_the_standing_carries_the_sentence_only_with_the_button(
        self, ready: tuple[Site, Any, str]
    ) -> None:
        site, ada, digest = ready
        panel = ada.get(f"/api/design/standing/{digest}").payload["publish"]
        assert panel["publishable"] is True
        assert panel["terms"] == {"version": terms.VERSION, "says": terms.DEDICATION,
                                  "href": terms.PATH}
        bea = site.signed_in("bea")
        assert bea.get(f"/api/design/standing/{digest}").payload["publish"]["terms"] is None
        ada.post("/api/design/publish", {"hash": digest, "terms": terms.VERSION})
        assert ada.get(f"/api/design/standing/{digest}").payload["publish"]["terms"] is None

    def test_a_game_published_before_the_terms_keeps_working_unmarked(
        self, drafting: Site, monkeypatch: pytest.MonkeyPatch  # noqa: F811
    ) -> None:
        """Published as the store held it before `0029_terms.sql`: listed, played, and its
        record left empty, since nobody agreed to anything for it."""
        _screened(monkeypatch, passed=True)
        ada = drafting.signed_in("ada")
        digest = _saved(ada, _template()).payload["hash"]
        with drafting.store.cursor() as cursor:
            cursor.execute("update games set visibility = 'public', published = now()"
                           " where hash = %s", (digest,))
        drafting.store.commit()
        assert digest in drafting.listed()
        assert _recorded(drafting, digest) == (None, None)
        assert ada.get(f"/api/game/{digest}").status == 200
        again = drafts.publish(drafting.store, digest, _user_id(drafting, "ada"), STAND_IN)
        assert again.already, "already on trial: no box asked for, nothing stamped"
        drafting.store.rollback()
        assert _recorded(drafting, digest) == (None, None)

    def test_the_store_refuses_a_stamp_without_its_version(
        self, ready: tuple[Site, Any, str]
    ) -> None:
        site, _, digest = ready
        refused = pytest.raises(Exception, match=re.compile("games_terms_both|check", re.I))
        with refused, site.store.cursor() as cursor:
            cursor.execute("update games set terms_accepted = now() where hash = %s",
                           (digest,))
        site.store.rollback()
