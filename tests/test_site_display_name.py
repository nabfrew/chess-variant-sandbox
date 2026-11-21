"""A member changes their display name (DN2, `docs/reference/site-as-built.md` §6).

The owner, 2026-11-20: *"The users should get to chose a display-name, the email should not
be shown."* A member chose a handle at `/join`; this is the way to change it afterwards, from
their own account page, and what holds each decision the brief made:

- the change runs joining's format rule and uniqueness check, with the maintenance name
  reserved as at joining;
- **a name once used is never free for somebody else**: nobody can take a member's former
  name, and the member can go back to one of their own;
- **old links keep working**: `/players/<old>` sends a member to the page under the current
  name, and no page shows another member a *formerly* line;
- one change in thirty days, the first free, said on the page before the change;
- the sign-in limiter's per-account count follows the account through a rename;
- the migration is safe on a store that already has members.

That no address appears anywhere new is `tests/test_site_no_email_shown.py`'s walk, which
renames its member and walks the new routes. Invented names and addresses only. Needs a real
Postgres and skips without one, as every site test does.
"""

from __future__ import annotations

import secrets
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import site_store
from test_site import (
    PASSWORD,
    Answer,
    Browser,
    Sent,
    Site,
    _joined,
    _running_site,
    _user_id,
)

from chess_variant_sandbox.site import admin, identity, members, migrate, pages, quota, routes
from chess_variant_sandbox.site.db import connect

MIGRATION = "0045_former_handles.sql"

PROFILE_JS = Path(__file__).resolve().parents[1] / "chess_variant_sandbox/site/static/profile.js"


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


def _rename(browser: Browser, handle: str) -> Answer:
    return browser.post("/api/handle", {"handle": handle})


def _former(site: Site, handle: str) -> list[str]:
    with site.store.cursor() as cursor:
        cursor.execute(
            "select f.handle from former_handles f join users u on u.id = f.user_id"
            " where u.handle = %s order by f.handle",
            (handle,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _age_the_change(site: Site, handle: str, days: int) -> None:
    """Move a member's last change `days` into the past, so the limit can be crossed."""
    with site.store.cursor() as cursor:
        cursor.execute(
            "update users set handle_changed = now() - make_interval(days => %s)"
            " where handle = %s",
            (days, handle),
        )
    site.store.commit()


class TestRenaming:
    def test_a_member_changes_their_name_and_every_page_follows(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        changed = _rename(ada, "ada-wren")
        assert changed.status == 200, changed.text
        assert changed.payload["handle"] == "ada-wren"
        assert changed.payload["page"] == "/players/ada-wren"
        assert ada.get("/api/me").payload["handle"] == "ada-wren"
        profile = ada.get("/api/profile").payload
        assert profile["handle"] == "ada-wren" and profile["page"] == "/players/ada-wren"
        assert profile["name"]["former"] == ["ada-lark"]
        bea = _joined(site, "bea-moss")
        page = bea.get("/players/ada-wren")
        assert page.status == 200 and "ada-wren" in page.text
        assert "ada-lark" not in page.text, "no page shows another member a former name"
        directory = bea.get("/api/players").payload["directory"]
        assert [row["handle"] for row in directory] == ["ada-wren", "bea-moss"]

    def test_the_format_rule_is_joinings(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        for bad in ("ab", "-ada", "a" * 25, "ada lark", "ada@lark", ""):
            refused = _rename(ada, bad)
            assert refused.status in (400, 401), bad
            assert "3 to 24 characters" in refused.payload["error"], bad
        assert ada.get("/api/me").payload["handle"] == "ada-lark"

    def test_the_same_name_is_not_a_change(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        refused = _rename(ada, "ada-lark")
        assert refused.status in (400, 401)
        assert "already your name" in refused.payload["error"]

    def test_a_visitor_cannot_rename_anybody(self, site: Site) -> None:
        refused = _rename(site.browser(), "nobody-here")
        assert refused.status in (400, 401)
        assert routes.find("POST", "/api/handle") in routes.OWN


class TestNamesAreNeverFreed:
    def test_nobody_takes_another_members_current_name_in_any_case(
        self, site: Site
    ) -> None:
        _joined(site, "ada-lark")
        bea = _joined(site, "bea-moss")
        refused = _rename(bea, "ADA-lark")
        assert refused.status in (400, 401)
        assert refused.payload["error"] == "the handle ADA-lark is taken; choose another"

    def test_nobody_takes_a_members_former_name(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        bea = _joined(site, "bea-moss")
        refused = _rename(bea, "Ada-Lark")
        assert refused.status in (400, 401)
        assert refused.payload["error"] == "the handle Ada-Lark is taken; choose another", (
            "a former name is refused in the very words a current one is")
        assert bea.get("/api/me").payload["handle"] == "bea-moss"

    def test_nobody_joins_under_a_members_former_name(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        invited = site.invite("cleo@example.com", Sent())
        refused = site.browser().post(
            "/api/join", {"code": invited.code, "handle": "ada-lark", "password": PASSWORD}
        )
        assert refused.status == 401
        assert refused.payload["error"] == "the handle ada-lark is taken; choose another"

    def test_a_member_goes_back_to_their_own_former_name(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        _age_the_change(site, "ada-wren", 31)
        back = _rename(ada, "Ada-Lark")
        assert back.status == 200, back.text
        assert back.payload["handle"] == "Ada-Lark"
        assert _former(site, "Ada-Lark") == ["ada-wren"], (
            "the name taken back leaves the former names; the one given up joins them")

    def test_the_maintenance_name_is_reserved_as_at_joining(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(admin.NAME_VAR, "keeper-of-keys")
        ada = _joined(site, "ada-lark")
        refused = _rename(ada, "Keeper-Of-Keys")
        assert refused.status in (400, 401)
        assert refused.payload["error"] == "the handle Keeper-Of-Keys is taken; choose another"

    def test_the_maintenance_account_takes_no_members_former_name(
        self, site: Site, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        monkeypatch.setenv(admin.NAME_VAR, "ada-lark")
        assert admin.ensure_account(site.store) is None
        site.store.commit()
        assert identity.holder_of(site.store, "ada-lark") == (
            _user_id(site, "ada-wren"), "ada-wren")

    def test_leaving_takes_the_former_names_with_it(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        admin.forget(site.store, "ada-wren")
        site.store.commit()
        with site.store.cursor() as cursor:
            cursor.execute("select count(*) from former_handles")
            assert cursor.fetchone()[0] == 0


class TestOldLinksFollow:
    def test_the_old_page_sends_a_member_to_the_new_one(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        bea = _joined(site, "bea-moss")
        moved = bea.get("/players/ADA-LARK")
        assert moved.status == 303, "never a 301: a browser keeps one, and names come back"
        assert moved.headers["Location"] == "/players/ada-wren"
        assert bea.get("/players/ada-wren").status == 200

    def test_a_visitor_still_goes_to_the_door(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        door = site.browser().get("/players/ada-lark")
        assert door.status == 303 and door.headers["Location"] == "/signin"

    def test_a_name_nobody_had_is_still_not_found(self, site: Site) -> None:
        bea = _joined(site, "bea-moss")
        lost = bea.get("/players/never-taken")
        assert lost.status == 404 and lost.text == pages.not_found()

    def test_a_shut_out_member_is_not_found_under_any_name(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        with site.store.cursor() as cursor:
            cursor.execute("update users set revoked = now() where handle = 'ada-wren'")
        site.store.commit()
        bea = _joined(site, "bea-moss")
        assert bea.get("/players/ada-lark").status == 404
        assert bea.get("/players/ada-wren").status == 404

    def test_a_former_name_still_signs_its_member_in(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        again = site.browser()
        signed = again.post("/api/signin", {"handle": "ada-lark", "password": PASSWORD})
        assert signed.status == 200, signed.text
        assert signed.payload["handle"] == "ada-wren"

    def test_the_privacy_notice_says_earlier_names_are_kept_and_go(self) -> None:
        notice = pages.privacy()
        assert "keeps your earlier handles" in notice
        assert "other members see only the handle you have now" in notice
        assert "leaving deletes the earlier ones" in notice

    def test_the_address_helper_and_the_prefix_agree(self) -> None:
        assert members.address("ada-wren") == routes.MEMBERS + "ada-wren"


class TestTheLimit:
    def test_the_first_change_is_free_and_the_next_waits_thirty_days(
        self, site: Site
    ) -> None:
        ada = _joined(site, "ada-lark")
        assert ada.get("/api/profile").payload["name"]["canChange"] is True
        assert _rename(ada, "ada-wren").status == 200, "the first change, straight away"
        box = ada.get("/api/profile").payload["name"]
        assert box["canChange"] is False and box["from"]
        refused = _rename(ada, "ada-finch")
        assert refused.status in (400, 401)
        assert "less than 30 days ago" in refused.payload["error"]
        assert box["from"] in refused.payload["error"]
        _age_the_change(site, "ada-wren", 29)
        assert _rename(ada, "ada-finch").status in (400, 401)
        _age_the_change(site, "ada-wren", 31)
        assert _rename(ada, "ada-finch").status == 200

    def test_the_rule_is_said_before_the_change(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        box = ada.get("/api/profile").payload["name"]
        assert box["rule"] == identity.RENAME_RULE
        for said in ("once every 30 days", "first change", "keep working",
                     "nobody else can ever take one", "go back to one of them"):
            assert said in identity.RENAME_RULE, said
        assert identity.RENAME_EVERY.days == 30
        profile_page = ada.get("/profile").text
        assert 'id="your-name"' in profile_page

    def test_the_page_draws_the_rule_above_the_form(self) -> None:
        script = PROFILE_JS.read_text(encoding="utf-8")
        drawing = script[script.index("function drawYourName"):]
        drawing = drawing[:drawing.index("\n}\n")]
        assert drawing.index("box.rule") < drawing.index("createElement('input')")
        assert "'/api/handle'" in drawing


class TestTheLimiterFollowsTheAccount:
    @pytest.fixture(autouse=True)
    def _small_allowance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An account allowance below one address's, so a few attempts trip it and no
        address's own limit is what trips (each attempt is a 50 ms scrypt)."""
        monkeypatch.setattr(quota, "SIGNIN_TRIES_AN_ACCOUNT", 4)

    def _spend_on(self, site: Site, handle: str) -> None:
        """The per-account allowance against one name."""
        for attempt in range(quota.SIGNIN_TRIES_AN_ACCOUNT):
            answer = site.browser(client="198.51.100.4").post(
                "/api/signin", {"handle": handle, "password": f"wrong-{attempt}"}
            )
            assert answer.status == 401, answer.text

    def test_renaming_does_not_clear_the_count(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        self._spend_on(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        for name in ("ada-wren", "ada-lark"):
            refused = site.browser(client="203.0.113.7").post(
                "/api/signin", {"handle": name, "password": PASSWORD}
            )
            assert refused.status == 429, name

    def test_an_old_name_is_no_fresh_allowance(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        self._spend_on(site, "ada-lark")
        refused = site.browser(client="203.0.113.7").post(
            "/api/signin", {"handle": "ada-wren", "password": PASSWORD}
        )
        assert refused.status == 429

    def test_the_key_is_the_account_for_every_name(self, site: Site) -> None:
        ada = _joined(site, "ada-lark")
        assert _rename(ada, "ada-wren").status == 200
        user = _user_id(site, "ada-wren")
        for name in ("ada-wren", "ADA-WREN", "ada-lark"):
            assert identity.limit_key(site.store, name) == f"user {user}"
        assert identity.limit_key(site.store, "nobody-here") == "nobody-here"
        assert not identity.HANDLE.match(f"user {user}"), "no handle can meet an account key"


class TestTheMigration:
    def test_it_is_safe_on_a_store_with_members(self, tmp_path: Path) -> None:
        """A store migrated to the step before, with members in it, takes `0045`: every
        member keeps their name, starts with no former names and their first change still
        to make, and can rename at once."""
        url = site_store.database("no store to migrate")
        schema = f"dn2_{secrets.token_hex(6)}"
        before = tmp_path / "before"
        before.mkdir()
        for one in migrate.MIGRATIONS.glob("*.sql"):
            if one.name < MIGRATION:
                shutil.copy(one, before / one.name)
        with connect(url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"create schema {schema}")
                cursor.execute(f"set search_path to {schema}")
            try:
                migrate.run(connection, before)
                with connection.cursor() as cursor:
                    for handle in ("ada-lark", "bea-moss"):
                        cursor.execute(
                            "insert into users (handle, email) values (%s, %s)",
                            (handle, f"{handle}@example.invalid"),
                        )
                report = migrate.run(connection)
                assert report.applied == (MIGRATION,)
                with connection.cursor() as cursor:
                    cursor.execute("select handle, handle_changed from users order by handle")
                    assert cursor.fetchall() == [("ada-lark", None), ("bea-moss", None)]
                    cursor.execute("select count(*) from former_handles")
                    assert cursor.fetchone()[0] == 0
                    cursor.execute("select id from users where handle = 'ada-lark'")
                    ada = int(cursor.fetchone()[0])
                assert identity.rename(connection, ada, "ada-wren") == "ada-wren"
            finally:
                connection.rollback()
                with connection.cursor() as cursor:
                    cursor.execute(f"drop schema if exists {schema} cascade")
                connection.commit()
