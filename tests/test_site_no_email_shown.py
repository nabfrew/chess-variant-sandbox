"""No member's email address reaches anybody but that member and the owner's admin page (DN1).

The owner, 2026-11-20: *"The users should get to chose a display-name, the email should not
be shown."* A member is known by the handle they chose when they joined. This walks a running
site over a real store with a seeded member whose address, and the addresses tied to them (a
friend they invited, a stranger who asked for an invitation), carry one invented marker, and
searches every page, every payload and every mail somebody else receives for it:

- **a signed-out visitor and another member** get every page the site serves and every GET
  route the site registers, with each `<hash>`, `<id>` and `<token>` filled from the seeded
  activity (a draft, a library game, a challenge, the match it started, a challenge link),
  and none of their answers carries the marker; a route this cannot fill is named in
  `UNFILLED` with the reason, so a new route cannot slip past the walk;
- **the mail** another person receives (the friend's invitation) names the member by handle
  and carries no address of theirs;
- **the admin page's payload does carry them**, which is what shows the seeding worked and
  the search would have found a leak.

Audit and reasoning: `docs/reference/site-as-built.md` §6, *Who sees an email address*.
Needs a real Postgres and skips with a reason without one (`tests/site_store.py`).
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_links import (
    DRAFT,
    FAR,
    LEFT,
    LEFT_HASH,
    MIDDLE,
    RIGHT,
    SERVE_FAST,
    Site,
    _load,
    _saved,
)

from chess_variant_sandbox import fairy
from chess_variant_sandbox.site import admin, feeds, mail, migrate, routes, screening, seo
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web.server import Server

MARK = "zq7vh3"
"""The invented string every seeded address carries, and nothing else on the site does."""

MEMBER = "quillfeather"
OTHER = "tallowmere"
ADDRESS = f"qa-{MARK}-member@unseen.invalid"
OTHER_ADDRESS = f"qa-{MARK}-other@unseen.invalid"
FRIEND = f"qa-{MARK}-friend@unseen.invalid"
ASKER = f"qa-{MARK}-asker@unseen.invalid"
OWNER = "the-owner@owner.invalid"

UNFILLED = {
    "/api/round/<date>": "a day's round; `/api/round` is walked and names the same rows",
    "/api/provisional/<job>": "a browser measurement's job, which this walk never starts",
    "/api/proven/<hash>/<position>": "a proven position of a game none of this seeds",
}
"""GET routes the walk does not fill, each with why it cannot carry an address the walk
would miss."""


def _shows(text: str) -> list[str]:
    """Every address in `text` carrying the marker, or the marker anywhere."""
    found = re.findall(rf"[^\s\"'<>]*{MARK}[^\s\"'<>]*", text, flags=re.IGNORECASE)
    return found


@contextlib.contextmanager
def _own_site(root: Path) -> Iterator[Site]:
    """The site on a port of its own, over a schema of its own: the session's shared schema
    is emptied by every other site test, and under `-n` xdist may hand this worker another
    module's test between two of these, which would empty the seeding under them."""
    url = site_store.database("no store to run the site on")
    name = f"no_email_{secrets.token_hex(6)}"

    def open_one() -> Any:
        connection = connect(url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {name}")
        return connection

    with connect(url) as making:
        with making.cursor() as cursor:
            cursor.execute(f"create schema {name}")
            cursor.execute(f"set search_path to {name}")
        making.commit()
        migrate.run(making)
        making.commit()
    store = open_one()
    store.commit()
    app = SiteApp(root / "rulesets", root / "runs", archive_dir=root / "archive",
                  connect=open_one)
    server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    try:
        yield Site(app, store, server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        store.close()
        with connect(url) as dropping:
            with dropping.cursor() as cursor:
                cursor.execute(f"drop schema {name} cascade")
            dropping.commit()


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One running site, seeded once, read by every test here: the walk is many requests,
    and splitting it keeps each test inside the fast tier's time."""
    root = tmp_path_factory.mktemp("no-email")
    for name in ("rulesets", "runs", "archive"):
        (root / name).mkdir()
    for name, spec in (("left", LEFT), ("middle", MIDDLE), ("right", RIGHT), ("far", FAR)):
        (root / "rulesets" / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch, _own_site(root) as site:
        monkeypatch.setenv(mail.URL_VAR, "https://chess-variant-sandbox.example")
        monkeypatch.setenv(mail.OWNER_VAR, OWNER)
        # The playability check plays games, and a game's page asks Fairy-Stockfish about
        # its opponents: seconds each, and neither holds an address.
        monkeypatch.setattr(screening, "check", lambda spec: {"passed": True})
        monkeypatch.setattr(fairy, "available", lambda: False)
        monkeypatch.setattr(fairy, "verdict", lambda spec: None)
        yield _seeded(site, root)


def _seeded(site: Site, root: Path) -> dict[str, Any]:
    """The member, another member, an admin, and what each did, through the site."""
    _load(site, root)
    member_browser = site.signed_in(MEMBER)
    other_browser = site.signed_in(OTHER)
    admin_browser = site.signed_in("keeperof", admin.ADMIN)
    with site.store.cursor() as cursor:
        cursor.execute("update users set email = %s where handle = %s", (ADDRESS, MEMBER))
        cursor.execute(
            "update users set email = %s where handle = %s", (OTHER_ADDRESS, OTHER)
        )
        cursor.execute("update users set invites_to_give = 2 where handle = %s", (MEMBER,))
    site.store.commit()
    sent: list[mail.Message] = []
    site.app.send = sent.append

    draft = _saved(member_browser, DRAFT).payload["hash"]
    invited = member_browser.post("/api/invite", {"email": FRIEND})
    assert invited.status == 200, invited.text
    asked = site.browser().post("/api/ask-invitation", {"email": ASKER, "note": "a note"})
    assert asked.status == 200, asked.text
    said = member_browser.post("/api/suggest", {"kind": "site", "body": "a suggestion"})
    assert said.status == 200, said.text

    challenged = member_browser.post(
        "/api/challenge", {"hash": LEFT_HASH, "seat": "first", "handle": OTHER}
    )
    assert challenged.status == 200, challenged.text
    accepted = other_browser.post(f"/api/challenge/{challenged.payload['id']}/accept")
    assert accepted.status == 200, accepted.text
    linked = member_browser.post("/api/challenge", {"hash": LEFT_HASH, "link": True})
    assert linked.status == 200, linked.text
    token = re.search(r"/c/([A-Za-z0-9_-]+)", linked.text)
    assert token is not None, linked.text
    return {
        "site": site,
        "member": member_browser,
        "other": other_browser,
        "admin": admin_browser,
        "sent": sent,
        "fill": {
            "hash": [LEFT_HASH, draft],
            "id": [str(accepted.payload["id"])],
            "token": [token.group(1)],
        },
    }


def _paths(fill: dict[str, list[str]]) -> list[str]:
    """Every page the site serves and every GET route it registers, filled in."""
    pages = ["/", "/game", "/rules", "/rules.md", "/whats-new", "/score-history",
             seo.ROBOTS_PATH, seo.SITEMAP_PATH, seo.LLMS_PATH, feeds.GAMES_PATH,
             feeds.NEWS_PATH, f"{routes.MEMBERS}{MEMBER}", f"{routes.MEMBERS}{OTHER}"]
    pages += list(routes.PAGES)
    pages += [f"/game/{digest}" for digest in fill["hash"]]
    pages += [f"/c/{token}" for token in fill["token"]]
    walked = []
    for route in routes.ALLOWED + routes.OWN:
        if route.method != "GET" or route.path in UNFILLED:
            continue
        holes = re.findall(r"<(\w+)>", route.path)
        assert all(hole in fill for hole in holes), (
            f"{route.path} has a part this walk cannot fill; fill it or name it in UNFILLED")
        filled = [route.path]
        for hole in holes:
            filled = [one.replace(f"<{hole}>", value, 1) for one in filled
                      for value in fill[hole]]
        walked += filled
    return pages + walked


VIEWERS = ("a visitor", "another member")


def _viewer(seeded: dict[str, Any], who: str) -> Any:
    return seeded["site"].browser() if who == "a visitor" else seeded["other"]


@pytest.mark.parametrize("who", VIEWERS)
def test_no_page_shows_a_members_address_to_anybody_else(
    seeded: dict[str, Any], who: str
) -> None:
    paths = [path for path in _paths(seeded["fill"]) if not path.startswith("/api/")]
    assert len(paths) > 20
    browser = _viewer(seeded, who)
    for path in paths:
        answer = browser.get(path)
        assert _shows(answer.text) == [], f"{who} reading {path} ({answer.status})"


@pytest.mark.parametrize("who", VIEWERS)
def test_no_payload_carries_a_members_address_to_anybody_else(
    seeded: dict[str, Any], who: str
) -> None:
    """A payload carrying an address the page does not draw still shows it."""
    paths = [path for path in _paths(seeded["fill"]) if path.startswith("/api/")]
    assert len(paths) > 20
    browser = _viewer(seeded, who)
    for path in paths:
        answer = browser.get(path)
        assert _shows(answer.text) == [], f"{who} reading {path} ({answer.status})"


def test_the_mail_another_person_gets_names_the_member_by_handle(
    seeded: dict[str, Any]
) -> None:
    """The friend's invitation names its sender by handle, says that is the name they go
    by, and carries no address but the friend's own; the one other message, the request for
    an invitation, goes to the owner alone."""
    [invitation] = [one for one in seeded["sent"] if one.to == FRIEND]
    assert invitation.subject.startswith(MEMBER)
    assert mail.NAMED.format(inviter=MEMBER) in invitation.body
    assert _shows(invitation.subject + invitation.body) == []
    assert sorted(one.to for one in seeded["sent"]) == sorted([FRIEND, OWNER])


def test_the_owners_admin_page_does_carry_them(seeded: dict[str, Any]) -> None:
    """The seeding worked, so the searches above would have found a leak."""
    shown = seeded["admin"].get("/api/admin").text
    for address in (ADDRESS, OTHER_ADDRESS, FRIEND, ASKER):
        assert address in shown
