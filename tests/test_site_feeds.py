"""Feeds, and a *what's new* a crawler can read (G11 of `docs/plan-growth.md`).

- **`/whats-new`** is drawn on the server: the changelog's entries are in the served HTML,
  outside any script, each an element a link can name, escaped, and an entry marked *For
  maintainers.* (`changelog.MAINTAINERS`) is not among them.
- **Two Atom feeds**, `/feeds/games.atom` (the listed games that joined, library and
  published, newest first) and `/feeds/whats-new.atom` (the entries the page shows), parsed
  here as Atom 1.0 and checked against RFC 4287's required elements: a stable absolute id,
  an `updated` in RFC 3339, absolute links, a feed `updated` that is its newest entry's. A
  *For maintainers.* entry is in neither the page nor the feed.
- **Both feeds are linked from the head** of every page that is for the index, and from no
  page that says `noindex`.

Unit tests need no server; the rest run over HTTP against a running site and inline specs
(`test_site_doors`' walks, never chess) and need a real Postgres (`tests/site_store.py`).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime
from html import escape, unescape
from pathlib import Path
from typing import Any

import page_names
import pytest
from test_site_doors import PUBLIC
from test_site_game_page import (
    DISTANT,
    DRAFT,
    DRAFT_HASH,
    LONG,
    LONG_HASH,
    MEDIUM,
    SCORE,
    SHORT,
    SHORT_HASH,
    SITE_URL,
    _fetch,
    _measure,
)
from test_site_links import Site, _load, _running_site, _saved

from chess_variant_sandbox.site import admin, feeds, mail, pages, screening, seo
from chess_variant_sandbox.web import changelog
from chess_variant_sandbox.web.page import front, news_page

ATOM = "{http://www.w3.org/2005/Atom}"
RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")

CHANGELOG = """\
# Changelog

Header the site never shows.

---

## 2026-10-21

### A visitor's news

Something a visitor *would* notice, with `code` in it.

### Maintainer chatter

*For maintainers.*

Run `secret-command` in the folder `docs/secret/`.

## 2026-10-20

### A visitor's news

The same heading on another day is another entry.

### Two of a kind

First.

### Two of a kind

Second, under the same heading on the same day.

### <script>alert(1)</script> & friends

Text with <b>tags</b> & an ampersand.

## 2026-10-18

### The only news of its day

Older.
"""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """`test_site_game_page`'s library: four walks, the short one measured."""
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    for filename, spec in (("short.json", SHORT), ("medium.json", MEDIUM),
                           ("long.json", LONG), ("distant.json", DISTANT)):
        (tmp_path / "rulesets" / filename).write_text(json.dumps(spec), encoding="utf-8")
    _measure(tmp_path, SHORT, SCORE)
    return tmp_path


@pytest.fixture
def site(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
    with _running_site(tree) as running:
        _load(running, tree)
        yield running


def _payload(text: str = CHANGELOG) -> dict[str, Any]:
    """The changelog as the server reads it, *before* the site's filter."""
    days = changelog.parse(text)
    return {"days": days, "note": ""}


def _atom(text: str) -> ET.Element:
    root = ET.fromstring(text.encode("utf-8"))
    assert root.tag == f"{ATOM}feed"
    return root


def _text(parent: ET.Element, tag: str) -> str:
    found = parent.find(f"{ATOM}{tag}")
    assert found is not None and found.text, tag
    return found.text


def _problems(text: str) -> list[str]:
    """What RFC 4287 §4.1 requires of a feed and its entries that this one breaks: empty when
    it validates. Not a schema engine; the elements a feed reader refuses a document without,
    and the ones the plan names (stable ids, updated times, absolute links)."""
    found: list[str] = []
    try:
        root = _atom(text)
    except (ET.ParseError, AssertionError) as error:
        return [f"not an Atom feed: {error}"]

    def whole(address: str | None) -> bool:
        return bool(address) and re.match(r"^https?://[^/\s]+", address or "") is not None

    for tag in ("id", "title", "updated"):
        if root.find(f"{ATOM}{tag}") is None:
            found.append(f"feed has no {tag}")
    feed_author = root.find(f"{ATOM}author") is not None
    selves = [one for one in root.findall(f"{ATOM}link") if one.get("rel") == "self"]
    if len(selves) != 1:
        found.append("a feed has one rel=self link")
    for one in root.findall(f"{ATOM}link"):
        if not whole(one.get("href")):
            found.append(f"feed link is not absolute: {one.get('href')}")
    if not whole(_text(root, "id")):
        found.append("feed id is not an absolute address")
    ids: list[str] = []
    stamps: list[str] = []
    for entry in root.findall(f"{ATOM}entry"):
        for tag in ("id", "title", "updated"):
            if entry.find(f"{ATOM}{tag}") is None:
                found.append(f"entry has no {tag}")
        if not feed_author and entry.find(f"{ATOM}author") is None:
            found.append("entry has no author and the feed has none")
        entry_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if not whole(entry_id):
            found.append(f"entry id is not an absolute address: {entry_id}")
        ids.append(entry_id)
        when = (entry.findtext(f"{ATOM}updated") or "").strip()
        if not RFC3339.match(when):
            found.append(f"entry updated is not RFC 3339: {when}")
        stamps.append(when)
        alternates = [one for one in entry.findall(f"{ATOM}link")
                      if one.get("rel") in (None, "alternate")]
        if not alternates and entry.find(f"{ATOM}content") is None:
            found.append("entry has neither an alternate link nor content")
        for one in entry.findall(f"{ATOM}link"):
            if not whole(one.get("href")):
                found.append(f"entry link is not absolute: {one.get('href')}")
    if len(ids) != len(set(ids)):
        found.append("entry ids repeat")
    when = (root.findtext(f"{ATOM}updated") or "").strip()
    if not RFC3339.match(when):
        found.append(f"feed updated is not RFC 3339: {when}")
    elif stamps and when != max(stamps):
        found.append("feed updated is not its newest entry's")
    elif stamps:
        parsed = [datetime.fromisoformat(one.replace("Z", "+00:00")) for one in stamps]
        if parsed != sorted(parsed, reverse=True):
            found.append("entries are not newest first")
    return found


# --------------------------------------------------------------------- the validator itself


class TestTheValidatorCatchesWhatItShould:
    """A check that cannot fail checks nothing."""

    GOOD = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><id>https://x.example/f</id><title>t</title>'
        "<updated>2026-10-21T00:00:00Z</updated><author><name>a</name></author>"
        '<link rel="self" href="https://x.example/f"/>'
        "<entry><id>https://x.example/e</id><title>e</title>"
        "<updated>2026-10-21T00:00:00Z</updated>"
        '<link rel="alternate" href="https://x.example/e"/></entry></feed>'
    )

    def test_a_good_feed_has_no_problems(self) -> None:
        assert _problems(self.GOOD) == []

    @pytest.mark.parametrize("broken", [
        ("<title>t</title>", ""),
        ('href="https://x.example/e"', 'href="/e"'),
        ("2026-10-21T00:00:00Z</updated><author>", "2026-10-21</updated><author>"),
        ('<link rel="self" href="https://x.example/f"/>', ""),
        ("<id>https://x.example/e</id>", "<id>e</id>"),
    ])
    def test_each_break_is_named(self, broken: tuple[str, str]) -> None:
        assert _problems(self.GOOD.replace(*broken))

    def test_text_that_is_not_xml_is_named(self) -> None:
        assert _problems("<feed")


# ------------------------------------------------------------------ the page's words, alone


class TestTheWhatsNewPage:
    def test_it_draws_every_visitor_entry_and_none_for_maintainers(self) -> None:
        html = news_page.body(_payload())
        assert "A visitor&#x27;s news" in html
        assert "<code>code</code>" in html and "<em>would</em>" in html
        assert "Maintainer chatter" not in html
        assert "secret-command" not in html and "docs/secret" not in html
        assert "Header the site never shows" not in html

    def test_nothing_in_the_file_becomes_markup(self) -> None:
        html = news_page.body(_payload())
        assert "<script>" not in html and "<b>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; friends" in html
        assert "&lt;b&gt;tags&lt;/b&gt; &amp; an ampersand" in html

    def test_each_entry_has_an_anchor_that_names_it_and_that_never_repeats(self) -> None:
        found = news_page.items(_payload())
        anchors = [one.anchor for one in found]
        assert len(anchors) == len(set(anchors))
        assert "2026-10-21-a-visitor-s-news" in anchors
        assert "2026-10-20-a-visitor-s-news" in anchors
        assert "2026-10-20-two-of-a-kind" in anchors and "2026-10-20-two-of-a-kind-2" in anchors
        for anchor in anchors:
            assert f'id="{anchor}"' in news_page.body(_payload())

    def test_an_anchor_does_not_move_when_a_newer_entry_is_filed_above_it(self) -> None:
        before = {one.heading: one.anchor for one in news_page.items(_payload())}
        newer = CHANGELOG.replace(
            "## 2026-10-21", "## 2026-10-22\n\n### Brand new\n\nNews.\n\n## 2026-10-21", 1)
        after = {one.heading: one.anchor for one in news_page.items(_payload(newer))}
        assert after["Two of a kind"] == before["Two of a kind"]
        assert "Brand new" in after

    def test_an_empty_changelog_says_so_rather_than_drawing_nothing(self) -> None:
        html = news_page.body({"days": [], "note": "This server has no copy."})
        assert "This server has no copy." in html and "newsEntry" not in html

    def test_the_page_links_both_feeds_in_its_body(self) -> None:
        html = news_page.body(_payload())
        assert f'href="{front.NEWS_FEED}"' in html and f'href="{front.GAMES_FEED}"' in html

    def test_it_is_one_of_the_site_pages_that_say_what_they_are(self) -> None:
        drawn = page_names.site_pages()["/whats-new"]
        assert 'id="2026-10-21-a-change"' in drawn
        assert f"<title>{escape(news_page.TITLE)}</title>" in drawn


# ------------------------------------------------------------------------ the feeds, alone


class TestTheChangelogFeed:
    def test_it_validates_as_atom(self) -> None:
        assert _problems(feeds.news(SITE_URL, _payload())) == []

    def test_an_entry_for_maintainers_is_in_no_feed_even_from_an_unfiltered_payload(self) -> None:
        text = feeds.news(SITE_URL, _payload())
        assert "Maintainer chatter" not in text
        assert "secret-command" not in text and "docs/secret" not in text
        assert len(_atom(text).findall(f"{ATOM}entry")) == 6  # seven entries, one maintainers'

    def test_ids_are_the_pages_addresses_and_do_not_move(self) -> None:
        first = _atom(feeds.news(SITE_URL, _payload()))
        again = _atom(feeds.news(SITE_URL, _payload()))
        ids = [_text(one, "id") for one in first.findall(f"{ATOM}entry")]
        assert ids == [_text(one, "id") for one in again.findall(f"{ATOM}entry")]
        assert f"{SITE_URL}/whats-new#2026-10-21-a-visitor-s-news" in ids
        # And they name an element of the page.
        page = news_page.body(_payload())
        for one in ids:
            assert f'id="{one.split("#", 1)[1]}"' in page

    def test_a_newer_entry_changes_no_older_entrys_id_or_time(self) -> None:
        old = {_text(one, "id"): _text(one, "updated") for one in
               _atom(feeds.news(SITE_URL, _payload())).findall(f"{ATOM}entry")}
        newer = CHANGELOG.replace(
            "## 2026-10-21", "## 2026-10-22\n\n### Brand new\n\nNews.\n\n## 2026-10-21", 1)
        root = _atom(feeds.news(SITE_URL, _payload(newer)))
        now = {_text(one, "id"): _text(one, "updated") for one in root.findall(f"{ATOM}entry")}
        assert old.items() <= now.items()
        assert _text(root, "updated") == "2026-10-22T00:00:00Z"
        assert _problems(feeds.news(SITE_URL, _payload(newer))) == []

    def test_updated_is_the_day_and_the_feed_is_newest_first(self) -> None:
        root = _atom(feeds.news(SITE_URL, _payload()))
        stamps = [_text(one, "updated") for one in root.findall(f"{ATOM}entry")]
        assert stamps[0] == "2026-10-21T00:00:00Z" and stamps[-1] == "2026-10-18T00:00:00Z"
        assert _text(root, "updated") == stamps[0]

    def test_the_content_is_the_pages_markup_and_hostile_text_stays_text(self) -> None:
        root = _atom(feeds.news(SITE_URL, _payload()))
        contents = {_text(one, "id"): one.find(f"{ATOM}content") for one in
                    root.findall(f"{ATOM}entry")}
        news = contents[f"{SITE_URL}/whats-new#2026-10-21-a-visitor-s-news"]
        assert news is not None and news.get("type") == "html"
        assert "<em>would</em>" in (news.text or "") and "<code>code</code>" in (news.text or "")
        hostile = contents[f"{SITE_URL}/whats-new#2026-10-20-script-alert-1-script-friends"]
        assert hostile is not None
        assert "<script>" not in (hostile.text or "") and "&lt;b&gt;" in (hostile.text or "")

    def test_a_day_without_a_date_has_no_time_and_is_left_out_of_the_feed(self) -> None:
        text = CHANGELOG.replace("## 2026-10-18", "## Some Tuesday")
        root = _atom(feeds.news(SITE_URL, _payload(text)))
        assert "The only news of its day" not in feeds.news(SITE_URL, _payload(text))
        assert _problems(feeds.news(SITE_URL, _payload(text))) == []
        assert root.find(f"{ATOM}entry") is not None
        assert "Some Tuesday" in news_page.body(_payload(text))

    def test_an_empty_changelog_is_still_a_valid_feed(self) -> None:
        text = feeds.news(SITE_URL, {"days": [], "note": "none"})
        assert _problems(text) == []
        assert _text(_atom(text), "updated") == feeds.EPOCH

    def test_the_real_changelog_has_no_maintainers_entry_in_the_page_or_the_feed(self) -> None:
        payload = changelog.payload(Path(__file__).resolve().parents[1] / "docs" / "changelog.md")
        hidden = [entry["heading"] for day in payload["days"] for entry in day["entries"]
                  if entry.get("maintainers")]
        assert hidden, "docs/changelog.md has no entry marked for maintainers to test against"
        page = news_page.body(payload)
        feed = feeds.news(SITE_URL, payload)
        seen = {one.heading for one in news_page.items(payload)}
        for heading in hidden:
            # A heading a visitor entry shares is not a leak; one only a maintainer has is.
            if heading in seen:
                continue
            assert f"<h4>{heading}</h4>" not in page
            assert heading not in {_text(one, "title") for one in
                                   _atom(feed).findall(f"{ATOM}entry")}
        assert _problems(feed) == []


class TestTheGamesFeedAlone:
    def _listed(self) -> list[front.Listed]:
        return [
            front.Listed("a" * 12, "Old walk", "2026-10-01", "2026-10-01T08:00:00Z",
                         "library", "An old game. It has more."),
            front.Listed("b" * 12, "New <walk> & more", "2026-10-20", "2026-10-20T09:30:00Z",
                         "published", "A newer game, published. Then a second sentence."),
            front.Listed("c" * 12, "No row", "", "", "", "Has no arrival, so no news."),
        ]

    def test_it_validates_lists_newest_first_and_leaves_out_a_game_with_no_arrival(self) -> None:
        text = feeds.games(SITE_URL, self._listed())
        assert _problems(text) == []
        root = _atom(text)
        entries = root.findall(f"{ATOM}entry")
        assert [_text(one, "title") for one in entries] == ["New <walk> & more", "Old walk"]
        assert _text(entries[0], "id") == f"{SITE_URL}/game/{'b' * 12}"
        assert _text(entries[0], "summary") == "A newer game, published."
        category = entries[0].find(f"{ATOM}category")
        assert category is not None and category.get("term") == "published"
        assert _text(root, "updated") == "2026-10-20T09:30:00Z"

    def test_it_carries_the_newest_fifty(self) -> None:
        many = [front.Listed(f"{n:012x}", f"Game {n}", "", f"2026-01-{1 + n % 28:02d}T00:{n % 60:02d}:00Z",
                             "library", "x.") for n in range(feeds.FEED_LIMIT + 7)]
        root = _atom(feeds.games(SITE_URL, many))
        stamps = [_text(one, "updated") for one in root.findall(f"{ATOM}entry")]
        assert len(stamps) == feeds.FEED_LIMIT
        assert stamps == sorted(stamps, reverse=True)
        assert _problems(feeds.games(SITE_URL, many)) == []

    def test_a_name_with_a_control_character_still_parses(self) -> None:
        text = feeds.games(SITE_URL, [front.Listed(
            "d" * 12, "Bad\x01 name\x0b", "", "2026-10-20T09:30:00Z", "library", "x\x02y.")])
        assert _problems(text) == []

    def test_no_games_is_still_a_valid_feed(self) -> None:
        text = feeds.games(SITE_URL, [])
        assert _problems(text) == [] and _text(_atom(text), "updated") == feeds.EPOCH

    def test_without_any_address_the_links_are_still_whole(self) -> None:
        assert _problems(feeds.games("", self._listed())) == []
        assert _problems(feeds.news("", _payload())) == []


class TestTheHeadLinks:
    def test_a_page_for_the_index_links_both_feeds(self) -> None:
        head = front.head_lines("T", "D", "https://x.example/t")
        for path, title in front.FEEDS:
            assert (f'<link rel="alternate" type="application/atom+xml" '
                    f'title="{escape(title, quote=True)}" href="{path}">') in head
        assert head.count('rel="alternate"') == 2

    def test_a_page_that_says_noindex_links_none(self) -> None:
        assert 'rel="alternate"' not in front.head_lines("T", "D", None, index=False)

    def test_the_two_titles_differ_and_name_the_site(self) -> None:
        titles = [title for _, title in front.FEEDS]
        assert len(set(titles)) == 2 and all("Chess variant sandbox" in t for t in titles)
        assert feeds.GAMES_TITLE != feeds.NEWS_TITLE


# --------------------------------------------------------------------------- over HTTP


def _entries(port: int, path: str) -> list[ET.Element]:
    found = _fetch(port, path)
    assert found.status == 200
    return _atom(found.html).findall(f"{ATOM}entry")


@pytest.fixture
def written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The changelog the site reads, replaced by a file of known entries."""
    path = tmp_path / "changelog-under-test.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.setattr(changelog, "CHANGELOG", path)
    return path


class TestWhatsNewOverHttp:
    def test_the_served_html_holds_the_entries_without_a_script(
        self, site: Site, written: Path
    ) -> None:
        found = _fetch(site.port, "/whats-new")
        assert found.status == 200
        assert found.headers["content-type"] == "text/html; charset=utf-8"
        # Readable with every script, template and style removed, as a crawler reads it.
        words = found.readable
        assert "What's new" in words
        assert "Something a visitor would notice, with code in it." in words
        assert "The same heading on another day is another entry." in words
        assert "Older." in words
        assert "21 October 2026" in words
        assert found.title == news_page.TITLE
        assert found.canonical == SITE_URL + "/whats-new"
        assert found.meta("description") == news_page.DESCRIPTION
        assert found.meta("robots") is None

    def test_an_entry_for_maintainers_is_not_on_the_served_page(
        self, site: Site, written: Path
    ) -> None:
        found = _fetch(site.port, "/whats-new")
        for leak in ("Maintainer chatter", "secret-command", "docs/secret", "For maintainers"):
            assert leak not in found.html, leak
        # And the data the script's view draws from agrees with the server's drawing.
        api = _fetch(site.port, "/api/changelog")
        assert "Maintainer chatter" not in api.html

    def test_the_script_less_page_and_the_script_view_read_one_payload(
        self, site: Site, written: Path
    ) -> None:
        payload = json.loads(_fetch(site.port, "/api/changelog").html)
        drawn = _fetch(site.port, "/whats-new").html
        found = news_page.items(payload)
        assert len(found) == 6
        for item in found:
            assert f'id="{item.anchor}"' in drawn
            assert escape(item.heading) in drawn

    def test_it_is_in_the_sitemap_and_the_footer_and_llms_txt(
        self, site: Site, written: Path
    ) -> None:
        assert "/whats-new" in seo.PAGES
        assert f"<loc>{SITE_URL}/whats-new</loc>" in _fetch(site.port, "/sitemap.xml").html
        assert 'href="/whats-new"' in _fetch(site.port, "/credits").html
        llms = _fetch(site.port, "/llms.txt").html
        assert f"({SITE_URL}/feeds/games.atom)" in llms
        assert f"({SITE_URL}/feeds/whats-new.atom)" in llms and f"({SITE_URL}/whats-new)" in llms

    def test_the_nav_to_the_single_page_view_still_opens(self, site: Site) -> None:
        """`#/whats-new` is a fragment of `/`, which the server never sees; `/` still answers."""
        assert _fetch(site.port, "/").status == 200


class TestFeedsOverHttp:
    def test_the_changelog_feed_validates_and_hides_the_maintainers_entry(
        self, site: Site, written: Path
    ) -> None:
        found = _fetch(site.port, "/feeds/whats-new.atom")
        assert found.status == 200
        assert found.headers["content-type"] == "application/atom+xml; charset=utf-8"
        assert "max-age" in found.headers["cache-control"]
        assert _problems(found.html) == []
        assert "Maintainer chatter" not in found.html and "secret-command" not in found.html
        root = _atom(found.html)
        assert _text(root, "id") == SITE_URL + "/feeds/whats-new.atom"
        selves = [one.get("href") for one in root.findall(f"{ATOM}link")
                  if one.get("rel") == "self"]
        assert selves == [SITE_URL + "/feeds/whats-new.atom"]
        assert len(root.findall(f"{ATOM}entry")) == 6

    def test_the_games_feed_lists_the_listed_games_and_validates(self, site: Site) -> None:
        found = _fetch(site.port, "/feeds/games.atom")
        assert found.status == 200
        assert found.headers["content-type"] == "application/atom+xml; charset=utf-8"
        assert _problems(found.html) == []
        ids = {_text(one, "id") for one in _atom(found.html).findall(f"{ATOM}entry")}
        assert ids == {f"{SITE_URL}/game/{digest}" for digest in
                       {one.hash for one in site.app.listed_pages()}}
        assert f"{SITE_URL}/game/{SHORT_HASH}" in ids

    def test_each_entry_links_the_games_own_page_which_answers(self, site: Site) -> None:
        for one in _entries(site.port, "/feeds/games.atom"):
            link = one.find(f"{ATOM}link")
            assert link is not None
            address = link.get("href") or ""
            assert address.startswith(SITE_URL + "/game/")
            page = _fetch(site.port, address[len(SITE_URL):])
            assert page.status == 200 and page.canonical == address

    def test_a_library_game_is_new_on_the_day_the_loader_made_it_and_a_published_one_on_its_day(
        self, site: Site
    ) -> None:
        owner = site.signed_in("bo")
        _saved(owner, DRAFT)
        with site.store.cursor() as cursor:
            cursor.execute(
                "update games set visibility = 'public',"
                " published = '2026-10-19T10:00:00+00:00' where hash = %s", (DRAFT_HASH,))
            cursor.execute(
                "update games set created = '2026-10-01T00:00:00+00:00' where hash = %s",
                (SHORT_HASH,))
        site.store.commit()
        site.app.listing.clear()

        root = _atom(_fetch(site.port, "/feeds/games.atom").html)
        by_id = {_text(one, "id"): one for one in root.findall(f"{ATOM}entry")}
        published = by_id[f"{SITE_URL}/game/{DRAFT_HASH}"]
        library = by_id[f"{SITE_URL}/game/{SHORT_HASH}"]
        assert _text(published, "updated") == "2026-10-19T10:00:00Z"
        assert _text(library, "updated") == "2026-10-01T00:00:00Z"
        assert published.find(f"{ATOM}category").get("term") == "published"  # type: ignore[union-attr]
        assert library.find(f"{ATOM}category").get("term") == "library"  # type: ignore[union-attr]
        stamps = [_text(one, "updated") for one in root.findall(f"{ATOM}entry")]
        assert stamps == sorted(stamps, reverse=True)
        assert _problems(_fetch(site.port, "/feeds/games.atom").html) == []

    def test_a_game_taken_off_the_listing_leaves_the_feed_like_the_sitemap(
        self, site: Site
    ) -> None:
        with site.store.cursor() as cursor:
            cursor.execute("update games set rested = now() where hash = %s", (LONG_HASH,))
        admin.kill_game(site.store, SHORT_HASH)
        site.store.commit()
        site.app.listing.clear()
        ids = {_text(one, "id") for one in _entries(site.port, "/feeds/games.atom")}
        assert not {f"{SITE_URL}/game/{LONG_HASH}", f"{SITE_URL}/game/{SHORT_HASH}"} & ids
        sitemap = _fetch(site.port, "/sitemap.xml").html
        for one in ids:
            assert one in sitemap

    def test_a_saved_draft_is_in_no_feed(self, site: Site) -> None:
        _saved(site.signed_in("ada"), DRAFT)
        site.app.listing.clear()
        assert DRAFT_HASH not in _fetch(site.port, "/feeds/games.atom").html

    def test_a_head_request_answers_without_a_body(self, site: Site) -> None:
        for path in ("/feeds/games.atom", "/feeds/whats-new.atom", "/whats-new"):
            found = _fetch(site.port, path, "HEAD")
            assert found.status == 200 and found.html == "", path

    def test_an_unknown_feed_is_the_not_found_page(self, site: Site) -> None:
        assert _fetch(site.port, "/feeds/nothing.atom").status == 404
        assert _fetch(site.port, "/feeds/").status == 404


class TestLinkedFromTheHead:
    LINKED = tuple(path for path in PUBLIC) + ("/whats-new", f"/game/{SHORT_HASH}")

    def test_every_public_page_links_both_feeds(self, site: Site, written: Path) -> None:
        for path in self.LINKED:
            head = _fetch(site.port, path).html.split("</head>")[0]
            links = re.findall(
                r'<link rel="alternate" type="application/atom\+xml" title="([^"]*)" '
                r'href="([^"]*)">', head)
            assert [href for _, href in links] == [front.GAMES_FEED, front.NEWS_FEED], path
            assert [unescape(title) for title, _ in links] == [t for _, t in front.FEEDS], path

    def test_the_feed_links_resolve_to_feeds_that_answer(self, site: Site) -> None:
        head = _fetch(site.port, "/").html.split("</head>")[0]
        for href in re.findall(r'rel="alternate" type="application/atom\+xml"[^>]*href="([^"]*)"',
                               head):
            assert _fetch(site.port, href).status == 200, href

    def test_a_page_robots_keeps_out_links_no_feed(self, site: Site) -> None:
        for path in ("/signin", "/join?code=abc", "/nowhere"):
            assert 'rel="alternate"' not in _fetch(site.port, path).html, path

    def test_the_feeds_are_open_to_crawlers(self, site: Site) -> None:
        for path in (front.GAMES_FEED, front.NEWS_FEED, front.NEWS_PATH):
            assert not seo.disallowed(path)

    def test_the_pages_unit_draws_the_same_head(self) -> None:
        assert 'rel="alternate"' in pages.credits()
