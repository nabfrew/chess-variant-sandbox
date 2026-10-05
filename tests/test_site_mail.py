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
