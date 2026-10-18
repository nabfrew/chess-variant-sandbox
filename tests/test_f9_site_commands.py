"""`chess-variant-sandbox site admin | forget | load`, against a real PostgreSQL (archive/plan-tests F9, Q751b).

The three commands the deploy and the maintainer run against the site's store. Each is driven
through `cli.main`, with the store reached the way the box reaches it, by the database URL in
the environment (here the session's scratch schema, named in the URL's `search_path`), and
judged by what it prints and by what the rows say afterwards.

Skips, with the reason, without `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and `psycopg`
(`uv sync --extra dev --extra site`); `tests/test_site.py` has the recipe for a scratch cluster.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import pytest
import site_store

from chess_variant_sandbox import cli
from chess_variant_sandbox.site import admin, db, identity

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"
DECOY_WORD = "hunter2-not-a-real-one"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The emptied session schema, named by the environment's URL, and a connection to it."""
    url, schema = site_store.emptied("the site commands run against a real store")
    # Given a password the cluster ignores or needs, so the commands' printed address can be
    # held to hiding it. (A scratch cluster on trust takes any; the box's own is not on this URL.)
    parts = urlsplit(url)
    if "@" in parts.netloc and ":" not in parts.netloc.split("@")[0]:
        user, _, host = parts.netloc.partition("@")
        url = urlunsplit(parts._replace(netloc=f"{user}:{DECOY_WORD}@{host}"))
    joiner = "&" if "?" in url else "?"
    monkeypatch.setenv(db.ENV_VAR, f"{url}{joiner}options={quote(f'-csearch_path={schema}')}")
    connection = db.connect()
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def person(connection: Any, handle: str, role: str = admin.MEMBER) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into users (handle, email, password, role) values (%s, %s, %s, %s)"
            " returning id",
            (handle, f"{handle}@example.invalid", identity.hash_password("a-long-enough-one"),
             role),
        )
        found = cursor.fetchone()
    connection.commit()
    return int(found[0])


def one(connection: Any, sql: str, *args: Any) -> Any:
    with connection.cursor() as cursor:
        cursor.execute(sql, args)
        row = cursor.fetchone()
    connection.commit()
    return row[0] if row else None


class TestSiteAdmin:
    def test_promotes_to_admin_and_says_where_the_page_is(
        self, store: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        person(store, "ada")
        assert cli.main(["site", "admin", "ada"]) == 0
        assert capsys.readouterr().out == "ada is an admin; the maintenance page is at /admin\n"
        assert one(store, "select role from users where handle = 'ada'") == admin.ADMIN

    def test_remove_puts_them_back_to_an_ordinary_member(
        self, store: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        person(store, "ada", admin.ADMIN)
        assert cli.main(["site", "admin", "ada", "--remove"]) == 0
        assert capsys.readouterr().out == "ada is an ordinary member again\n"
        assert one(store, "select role from users where handle = 'ada'") == admin.MEMBER

    def test_the_handle_is_matched_without_regard_to_case(self, store: Any) -> None:
        person(store, "ada")
        assert cli.main(["site", "admin", "ADA"]) == 0
        assert one(store, "select role from users where handle = 'ada'") == admin.ADMIN

    def test_nobody_by_that_name_is_a_refusal_and_changes_nothing(
        self, store: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        person(store, "ada")
        assert cli.main(["site", "admin", "nobody"]) == 2
        assert capsys.readouterr().err == "error: no user with the handle 'nobody'\n"
        assert one(store, "select role from users where handle = 'ada'") == admin.MEMBER


class TestSiteForget:
    def test_yes_deletes_the_address_and_the_hash_and_keeps_the_row(
        self, store: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        user = person(store, "ada")
        assert cli.main(["site", "forget", "ada", "--yes"]) == 0
        assert capsys.readouterr().out == (
            "ada has been forgotten; their games stay and read 'a former player'\n"
        )
        with store.cursor() as cursor:
            cursor.execute("select email, password, handle, forgotten from users where id = %s",
                           (user,))
            email, password, handle, forgotten = cursor.fetchone()
        store.commit()
        assert (email, password, handle) == (None, None, None) and forgotten is not None

    def test_without_yes_it_asks_and_a_no_changes_nothing(
        self, store: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        person(store, "ada")
        asked: list[str] = []
        monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "n")
        assert cli.main(["site", "forget", "ada"]) == 0
        assert capsys.readouterr().out == "nothing was changed\n"
        assert "Forget ada?" in asked[0] and "cannot be undone" in asked[0]
        assert one(store, "select email from users where handle = 'ada'") == "ada@example.invalid"

    @pytest.mark.parametrize("answer", ["y", "YES", " yes "])
    def test_a_yes_at_the_prompt_forgets_them(
        self, store: Any, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        person(store, "ada")
        monkeypatch.setattr("builtins.input", lambda prompt: answer)
        assert cli.main(["site", "forget", "ada"]) == 0
        assert one(store, "select count(*) from users where forgotten is not null") == 1

    def test_anything_but_yes_at_the_prompt_is_a_no(
        self, store: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        person(store, "ada")
        monkeypatch.setattr("builtins.input", lambda prompt: "maybe")
        assert cli.main(["site", "forget", "ada"]) == 0
        assert one(store, "select count(*) from users where forgotten is not null") == 0


class TestSiteLoad:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        """A checkout with one committed game and nothing measured."""
        (tmp_path / "rulesets").mkdir()
        shutil.copy(RULESETS / "dull.json", tmp_path / "rulesets" / "dull.json")
        return tmp_path

    def games(self, store: Any) -> int:
        return int(one(store, "select count(*) from games"))

    def test_loads_the_library_names_the_store_without_its_password_and_says_what_moved(
        self, store: Any, tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["site", "load", "--root", str(tree)]) == 0
        out = capsys.readouterr().out
        assert out.startswith(f"loading {tree} into ")
        assert "dry run" not in out
        assert "row(s) written" in out and "games        1 row(s): 1 added, 0 updated" in out
        assert self.games(store) == 1
        assert one(store, "select name from games") == "dull"
        # the maintainer the library is owned by exists now
        assert one(store, "select count(*) from users where role = %s", admin.ADMIN) == 1

    def test_a_second_load_over_unmoved_files_writes_nothing_and_says_so(
        self, store: Any, tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["site", "load", "--root", str(tree)])
        capsys.readouterr()
        assert cli.main(["site", "load", "--root", str(tree)]) == 0
        assert "nothing changed" in capsys.readouterr().out
        assert self.games(store) == 1

    def test_dry_run_reports_the_same_and_rolls_it_back(
        self, store: Any, tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["site", "load", "--root", str(tree), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "1 added" in out and out.rstrip().endswith("dry run: nothing was committed")
        assert self.games(store) == 0
        assert one(store, "select count(*) from users") == 0

    def test_the_maintainer_option_names_who_owns_the_library(
        self, store: Any, tree: Path
    ) -> None:
        assert cli.main(["site", "load", "--root", str(tree), "--maintainer", "curator"]) == 0
        assert one(store, "select handle from users where role = %s", admin.ADMIN) == "curator"

    def test_the_address_it_names_does_not_carry_the_password(
        self, store: Any, tree: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        if DECOY_WORD not in db.database_url():
            pytest.skip("the test database's URL has no user to give a password to")
        assert cli.main(["site", "load", "--root", str(tree), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert DECOY_WORD not in out
        assert out.startswith(f"loading {tree} into postgresql://")
