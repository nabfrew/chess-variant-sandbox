"""Q276: the wall clock on the playability check a save runs.

`docs/plan-website.md` phase 2, row 1 -- *validate -> smoke -> screen on save,
synchronous, capped* -- and §6's *compute exhaustion* row. Q221 measured the check with
no cap at all and Q273 made every save run it, so a draft could hold a request thread and
a core on the box for as long as it liked.

`_slow_draft` is that draft: a 16 by 16 board of sliders under a ten-thousand-ply limit,
a perfectly valid ruleset whose check costs seven and a half seconds at one worker on the
development machine and far more on a small box. These tests give it a short clock
and prove three things -- the check is refused, the refusal says what to change, and no
file and no row are left behind.

What the check *decides* is `test_candidacy.py`'s and `test_screen.py`'s. Nothing here
asserts a verdict on a real game.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.site import screening
from chess_variant_sandbox.site import server as site_server
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE, Server

# `serve_forever` looks for its stop flag every half second by default, so each test
# that stopped a server waited out that half second in `shutdown()`. 20 ms stops it at once.
SERVE_FAST = {"poll_interval": 0.02}

# Long enough that the clock runs out inside the screen rather than during the child's
# half second of interpreter start, short enough that the test costs under two seconds:
# the test waits out the whole clock, so the clock is its time. It was two seconds, 2.02 s
# in the gate; one and a half (Q749) still gives the screen a second past the start.
CLOCK = 1.5

FILES = "abcdefghijklmnop"


def _slow_draft() -> dict[str, Any]:
    """§6's compute-exhaustion spec, written as a draft somebody could really save."""
    return {
        "schema_version": 2,
        "name": "a draft built to run long",
        "description": "",
        "board": {"width": 16, "height": 16},
        "pieces": [
            {
                "id": "Q",
                "name": "ranger",
                "value": 9,
                "moves": [
                    {
                        "kind": "slide",
                        "vectors": [[0, 1], [1, 0], [1, 1]],
                        "symmetry": "dihedral8",
                        "mode": "both",
                    }
                ],
            }
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {
                    "piece": "Q",
                    "squares": [f"{f}{rank}" for rank in (1, 2, 3, 4) for f in FILES],
                }
            ],
        },
        "rules": {
            "win": "annihilation",
            "stalemate": "draw",
            "max_plies": 10000,
            "repetition_draw": 10,
            "no_progress_draw": 1000,
        },
    }


def _boxed_in() -> dict[str, Any]:
    """A valid ruleset nobody can move in: a full board of forward-only soldiers."""
    return {
        "schema_version": 2,
        "name": "a draft nobody can move in",
        "description": "",
        "board": {"width": 4, "height": 4},
        "pieces": [
            {
                "id": "S",
                "name": "soldier",
                "value": 1,
                "moves": [
                    {
                        "kind": "leap",
                        "vectors": [[0, 1]],
                        "symmetry": "none",
                        "mode": "move",
                    }
                ],
            }
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {"piece": "S", "squares": ["a1", "b1", "c1", "d1", "a2", "b2", "c2", "d2"]}
            ],
        },
        "rules": {
            "win": "annihilation",
            "stalemate": "draw",
            "max_plies": 100,
            "repetition_draw": 3,
            "no_progress_draw": 40,
        },
    }


def _template() -> dict[str, Any]:
    spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
    spec["name"] = "a draft somebody kept"
    return spec


class Store:
    """A connection that records every statement and returns no rows.

    A count is the one statement that always has a row, so it answers nought: the draft
    cap (Q503a) counts a person's drafts before the check runs.
    """

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []

    @contextlib.contextmanager
    def cursor(self) -> Iterator[Store]:
        yield self

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> tuple[int] | None:
        last = self.executed[-1][0] if self.executed else ""
        return (0,) if last.lstrip().startswith("select count(") else None

    def fetchall(self) -> list[Any]:
        return []

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def designer(tmp_path: Path, store: Store) -> SiteApp:
    """A site over an empty library of its own, running the *real* check.

    Every other site test stubs `screening.check`; this file is the one that does not.
    """
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    app = SiteApp(
        tmp_path / "rulesets",
        tmp_path / "runs",
        archive_dir=tmp_path / "archive",
        connect=lambda: store,
    )
    app.viewer = 7
    return app


class TestTheCheckHasAWallClock:
    def test_a_draft_built_to_run_long_is_refused_and_says_what_to_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        children: list[subprocess.Popen[str]] = []

        class Watched(subprocess.Popen):  # type: ignore[type-arg]
            """The real child, remembered so its end can be read."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                children.append(self)

        monkeypatch.setattr(screening.subprocess, "Popen", Watched)
        with pytest.raises(screening.TooSlow) as refused:
            screening.check(_slow_draft(), seconds=CLOCK)

        # The clock is what stopped it, not the draft finishing: the child was killed. One
        # that ran to its end exits 0 with a verdict, and one the kill missed would be
        # waited for until it did. This used to be a bound on the wall time, three clocks,
        # which a machine busy with three other test workers can pass (Q377c).
        [child] = children
        assert child.returncode not in (0, None), f"the child exited {child.returncode}"
        said = str(refused.value)
        assert f"longer than {CLOCK:g} seconds" in said
        assert "nothing was saved" in said
        assert "smaller board" in said

    def test_nothing_is_saved_or_listed_from_a_check_that_did_not_finish(
        self, designer: SiteApp, store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the save does with a refusal. The refusal itself -- the real clock, the
        real kill -- is the test above, so the check here raises it at once rather than
        waiting out a second clock (Q377)."""

        def outran(spec: Any) -> dict[str, Any]:
            raise screening.TooSlow(screening.refusal(CLOCK))

        monkeypatch.setattr(screening, "check", outran)

        with pytest.raises(screening.TooSlow):
            designer.post("/api/design/save", {"spec": _slow_draft()})

        assert list((tmp_path / "runs").iterdir()) == []
        # The draft cap reads the store first (Q503a); nothing is written.
        assert [sql for sql, _ in store.executed if not sql.lstrip().startswith("select")] == []

    def test_a_draft_that_finishes_the_check_is_saved_as_a_draft_that_passed(
        self, designer: SiteApp, store: Store, tmp_path: Path
    ) -> None:
        """The check moved in front of the save, so the ordinary path is worth one pass
        through the real thing rather than through a stub."""
        kept = designer.post("/api/design/save", {"spec": _template()})

        # Q503a: a save lists nothing, whatever the check says; it records the pass.
        assert kept["listed"] is False and kept["visibility"] == "unlisted"
        assert kept["screenPassed"] is True
        # Q275: the page says what the save did, so the answer says how far the check
        # got. Both rungs ran and the draft passed the last of them.
        assert (kept["rung"], kept["rungPassed"]) == ("screen", True)
        assert "passes the playability check" in kept["said"]
        assert "measured its score on your computer" in kept["said"]
        assert (tmp_path / "runs" / "designed" / "specs" / f"{kept['hash']}.json").is_file()
        assert [
            params for sql, params in store.executed if "insert into games" in sql
        ], store.executed

    def test_the_cap_is_a_number_the_site_owns(self) -> None:
        """The plan names `screening.CHECK_SECONDS` as where the cap lives, and the
        smoke rung's own share has to fit inside it."""
        assert screening.CHECK_SECONDS > 0
        assert screening.SMOKE_SECONDS < screening.CHECK_SECONDS
        assert screening.WORKERS == 1


class TestTheRefusalReachesThePage:
    """A `ValueError` is the 400 the page prints where the click was. The slow draft is
    not needed to see that, so these two raise the refusal directly and cost nothing."""

    def test_a_save_that_outran_the_clock_answers_400_with_the_sentence(
        self, designer: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        said = screening.refusal(screening.CHECK_SECONDS)

        def refuse(spec: Any) -> dict[str, Any]:
            raise screening.TooSlow(said)

        def identify(token: str | None) -> None:
            # `viewer` is thread-local, and the server answers on a thread of its own.
            designer.viewer = 7

        monkeypatch.setattr(screening, "check", refuse)
        monkeypatch.setattr(designer, "identify", identify)
        status, payload = _over_http(designer, "/api/design/save", {"spec": _slow_draft()})
        assert status == 400
        assert payload["error"] == said

    def test_the_check_route_runs_the_capped_check_too(
        self, designer: SiteApp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/api/design/screen` is the same check asked for on its own, so it goes
        through the same cap rather than the local application's uncapped handler."""
        asked: list[Any] = []

        def capped(spec: Any) -> dict[str, Any]:
            asked.append(spec)
            return {"passed": True}

        monkeypatch.setattr(screening, "check", capped)
        assert designer.post("/api/design/screen", {"spec": _template()}) == {"passed": True}
        assert asked == [_template()]


class TestTheSmokeRung:
    """Phase 2's row asks for a smoke rung of its own between validate and screen. It is
    the cheap one, so a draft it refuses never costs the screen's eight games."""

    def test_a_draft_nobody_can_move_in_is_refused_by_the_smoke_rung(self) -> None:
        verdict = screening.check(_boxed_in())

        assert verdict["passed"] is False
        assert verdict["reasons"] == [
            "the first player has no legal move from the starting position"
        ]
        # The screen was never reached, so it left no numbers behind and the rung the
        # draft got to is the smoke one.
        assert verdict["stats"] == {}
        assert verdict["smoke"]["passed"] is False
        assert verdict["rung"] == "smoke"

    def test_a_template_passes_both_rungs_and_carries_both_sets_of_numbers(self) -> None:
        verdict = screening.check(_template())

        assert verdict["passed"] is True
        assert verdict["reasons"] == []
        assert verdict["stats"]["games"] == 8.0
        assert verdict["smoke"]["stats"]["games"] == float(screening.SMOKE_GAMES)
        assert verdict["hash"] and verdict["seconds"] > 0
        assert verdict["limit"] == screening.CHECK_SECONDS
        assert verdict["rung"] == "screen"

    def test_the_rungs_a_verdict_names_are_the_ladders_own(self) -> None:
        """`rung` is a key from `candidacy.rungs`, which is where the page reads the
        names it prints. A key the ladder does not have would print as itself."""
        from chess_variant_sandbox.web import candidacy

        keys = {rung["key"] for rung in candidacy.rungs()}
        assert {"validate", "smoke", "screen"} <= keys

    def test_an_unlisted_save_says_which_rung_stopped_it(
        self, designer: SiteApp, store: Store
    ) -> None:
        """Q275. The answer the page draws its sentence from, for a draft that was kept
        and left off the listing: the rung it stopped at, and why it is not listed."""
        kept = designer.post("/api/design/save", {"spec": _boxed_in()})

        assert kept["listed"] is False and kept["visibility"] == "unlisted"
        assert (kept["rung"], kept["rungPassed"]) == ("smoke", False)
        assert "did not pass the playability check" in kept["said"]
        assert "no legal move" in kept["said"]


class TestAnUnvalidatableDraftCostsNoProcess:
    def test_a_draft_that_is_not_a_ruleset_is_refused_before_anything_plays(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("a child was started for a draft that does not validate")

        monkeypatch.setattr(screening, "_in_a_child", unreachable)
        spec = _template()
        spec["board"] = {"width": 99, "height": 99}
        with pytest.raises(designing.DesignError, match="aren't a complete game yet"):
            screening.check(spec)


def _over_http(app: SiteApp, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """One POST through the site's own handler, so the status mapping is the real one."""
    server = Server(("127.0.0.1", 0), site_server.site_handler_for(app))
    threading.Thread(target=server.serve_forever, kwargs=SERVE_FAST, daemon=True).start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", REQUEST_HEADER: REQUEST_VALUE},
        )
        found = connection.getresponse()
        return found.status, json.loads(found.read() or b"{}")
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
