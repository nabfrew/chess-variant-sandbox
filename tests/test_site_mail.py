"""What the site's mail looks like to a mail program's junk filter (Q639).

The trial's first invitation went to junk on the user's phone (`docs/reference/
first-visit-study.md` §1.2a, item 4). The domain's records were in order; what this side
could change is the message: a sender with a name, an HTML part that says what the text
says with no link hiding behind other words, and a line asking the person to mark it
*not junk*. Every test passes its own `urlopen`: no mail leaves the machine.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

import pytest
import site_words

from chess_variant_sandbox.site import mail

SITE = "https://chess-variant-sandbox.example"
LINK = f"{SITE}/join?code=abc&x=1"


class _Accepted:
    status = 200

    def read(self) -> bytes:
        return b"{}"

    def __enter__(self) -> _Accepted:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _posted(monkeypatch: pytest.MonkeyPatch, sender: str, message: mail.Message) -> dict[str, Any]:
    posted: list[dict[str, Any]] = []

    def caught(request: Any, timeout: float) -> _Accepted:
        posted.append(json.loads(request.data))
        return _Accepted()

    monkeypatch.setenv(mail.KEY_VAR, "a-key-that-goes-nowhere")
    monkeypatch.setenv(mail.FROM_VAR, sender)
    monkeypatch.setattr(mail.urllib.request, "urlopen", caught)
    mail.send(message)
    [fields] = posted
    return fields


def test_a_bare_sender_address_gets_the_sites_name(monkeypatch: pytest.MonkeyPatch) -> None:
    fields = _posted(monkeypatch, "site@example.invalid", mail.invitation("bo@example.invalid", LINK))
    assert fields["from"] == "Chess variant sandbox <site@example.invalid>"


def test_a_sender_that_names_itself_is_left_as_written(monkeypatch: pytest.MonkeyPatch) -> None:
    fields = _posted(monkeypatch, "Ada <site@example.invalid>", mail.invitation("bo@example.invalid", LINK))
    assert fields["from"] == "Ada <site@example.invalid>"


def test_the_html_part_says_what_the_text_says(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same words, no styles, no images, nothing remote; the one link's visible text is
    its own address, so what a reader sees is where it goes."""
    message = mail.invitation("bo@example.invalid", LINK, "ada")
    fields = _posted(monkeypatch, "site@example.invalid", message)
    assert fields["text"] == message.body
    page = fields["html"]
    for absent in ("<img", "style", "<script", "src=", "<link"):
        assert absent not in page, absent
    hrefs = re.findall(r'<a href="([^"]*)">([^<]*)</a>', page)
    assert hrefs == [(html.escape(LINK), html.escape(LINK))]
    said = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())
    assert said == " ".join(message.body.split())


def test_the_move_email_has_its_html_part_too(monkeypatch: pytest.MonkeyPatch) -> None:
    message = mail.your_move("bo@example.invalid", "Shogi", "ada", "Pe4", f"{SITE}/g/1", f"{SITE}/mail/off/t")
    page = _posted(monkeypatch, "site@example.invalid", message)["html"]
    assert re.findall(r'<a href="([^"]*)">', page) == [f"{SITE}/g/1", f"{SITE}/mail/off/t"]


def test_the_invitation_ends_with_the_junk_tip_in_plain_words() -> None:
    body = mail.invitation("bo@example.invalid", LINK, "ada").body
    assert body.rstrip().endswith(mail.JUNK_TIP)
    assert site_words.hits(mail.JUNK_TIP) == []
    assert mail.JUNK_TIP.upper() != mail.JUNK_TIP and len(mail.JUNK_TIP.split()) < 25


def test_the_invitation_carries_one_link() -> None:
    """One link, on the site's own host: a sign-in mail whose links point elsewhere reads
    as phishing."""
    body = mail.invitation("bo@example.invalid", LINK, "ada").body
    assert re.findall(r"https?://\S+", body) == [LINK]


def test_the_invitation_names_where_its_link_goes_in_words() -> None:
    """PF2 (`docs/reference/first-visit-study.md` §6.5 item 7): a cold reader could not place
    the site and read *choose a handle and a password* as phishing. The mail names the site's
    address in plain words beside the link, as text and never as a second link, and says the
    site has no other way to reach the reader."""
    body = " ".join(mail.invitation("bo@example.invalid", LINK, "ada").body.split())
    assert mail.link_host(LINK) == "chess-variant-sandbox.example"
    assert ("Open this link to choose a handle and a password on "
            "chess-variant-sandbox.example, the site's own address:") in body
    assert re.findall(r"https?://\S+", body) == [LINK]
    assert "it has no other way to reach you: it never asks for a phone number" in body
    assert mail.link_host("https://www.x.example/join") == "x.example"
    assert mail.link_host("l") == ""
    bare = " ".join(mail.invitation("bo@example.invalid", "l").body.split())
    assert "Open this link to choose a handle and a password, the site's own address:" in bare


def test_the_invitation_says_its_sender_goes_by_that_name_on_the_site() -> None:
    """DN1 (the owner, 2026-11-20): a member is known by the name they chose, never by their
    address. The invitation names its sender by that handle and says in one plain sentence
    that it is the name they go by on the site; an unsigned invitation says nothing of it."""
    body = mail.invitation("bo@example.invalid", LINK, "ada").body
    assert "ada is the name they go by on the site.\n" in body
    assert body.index("ada has invited you") < body.index("ada is the name") < body.index(
        mail.MAKERS)
    assert site_words.hits(mail.NAMED) == []
    assert len(mail.NAMED.split()) < 15
    bare = mail.invitation("bo@example.invalid", LINK).body
    assert "the name they go by" not in bare
