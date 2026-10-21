"""From a draft to a number: rungs, the measurement queue, lineage and nomination."""

from __future__ import annotations

import inspect
import json
import re
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import node_session
import pytest
import site_words

from chess_variant_sandbox import corpus
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import candidacy, designing, keeping
from chess_variant_sandbox.web import server as server_module
from chess_variant_sandbox.web.server import App

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"


def _variant(plies: int) -> dict[str, Any]:
    """The first template with a different move limit, which is a different game."""
    spec = json.loads(json.dumps(designing.templates()[0]["spec"]))
    spec["rules"]["max_plies"] = plies
    spec["name"] = f"skirmish {plies}"
    return spec


@pytest.fixture
def world(tmp_path: Path) -> tuple[App, Path]:
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    spec = load_spec(RULESETS / "dull.json")
    (rulesets / "dull.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "archive").mkdir()
    return App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive"), tmp_path


def _save(app: App, plies: int, parent: str = "") -> str:
    return str(app.post("/api/design/save", {"spec": _variant(plies), "parent": parent})["hash"])


def _store(
    tmp_path: Path,
    digest: str,
    weights: str = "",
    seeds: tuple[int, ...] = corpus.LIBRARY_SEEDS,
) -> None:
    """A measurement on the current basis, or on an older weights version.

    On the library's four seeds by default, because the rung this fixture stands before
    is promotion, and `promotion_blocker` asks for all four. `corpus.ONE_SEED` is what
    *measure it* on a draft's page writes.
    """
    candidate = corpus.find(
        digest, runs_dir=tmp_path / "runs", ruleset_dir=tmp_path / "rulesets",
        archive_dir=tmp_path / "archive",
    )
    basis = corpus.current_basis(digest, spec=candidate.spec, seeds=seeds)
    if weights:
        basis = replace(basis, weights_version=weights)
    out = corpus.measurement_dir(digest, tmp_path / "runs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "canonical.json").write_text(json.dumps({
        "hash": digest, "name": "measured", "index": 0.5, "sd": 0.02, "viable": True,
        "violations": [], "weights_version": basis.weights_version, "families": {},
        "standings": [], "games": 4 * len(seeds), "seeds": list(seeds),
        "totals_used": len(seeds),
        "runs": [{"seed": seed, "index": 0.5, "sd": 0.02, "viable": True,
                  "violations": [], "unreadable": [], "origin": "played"}
                 for seed in seeds],
        "basis": basis.as_dict(),
    }), encoding="utf-8")


class TestTheRungsAreOnScreen:
    def test_every_rung_from_validate_to_nomination_states_its_cost(self, world) -> None:
        app, _ = world
        rungs = app.get("/api/design")["rungs"]
        assert [r["key"] for r in rungs] == [
            "validate", "smoke", "screen", "save", "canonical", "nominate"]
        assert all(r["cost"] and r["what"] and r["runs"] for r in rungs)

    def test_the_long_rung_names_this_machines_last_timing_once_there_is_one(self) -> None:
        assert "not yet timed" in candidacy.rungs(None)[4]["cost"]
        assert "5 minutes the last time" in candidacy.rungs(300.0)[4]["cost"]

    def test_the_cheap_rungs_run_inline_and_are_timed(self, world) -> None:
        app, _ = world
        spec = _variant(170)
        check = app.post("/api/design/check", {"spec": spec})
        assert check["smokeSeconds"] >= 0
        screen = app.post("/api/design/screen", {"spec": spec})
        assert screen["hash"] == check["hash"]
        assert isinstance(screen["passed"], bool) and screen["seconds"] >= 0
        assert not (world[1] / "runs" / "designed").exists(), "a screen writes nothing"

    def test_a_screen_refuses_a_draft_that_is_not_a_ruleset(self, world) -> None:
        app, _ = world
        with pytest.raises(designing.DesignError):
            app.post("/api/design/screen", {"spec": {"name": "broken"}})


class TestTheLongMeasurementIsQueued:
    def test_measurements_run_one_game_at_a_time_whoever_asked(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        release = threading.Event()
        inside: list[str] = []
        most = [0]

        class Done:
            def as_dict(self) -> dict[str, Any]:
                return {"index": 0.5}

        def refresh(candidate, **_: Any) -> Done:
            inside.append(candidate.hash)
            most[0] = max(most[0], len(inside))
            release.wait(5)
            inside.remove(candidate.hash)
            return Done()

        monkeypatch.setattr(keeping.corpus, "refresh", refresh)
        queue = keeping.Measurements(tmp_path)
        spec = _variant(160)
        first = queue.start(corpus.Candidate("aaa", "a", spec, tmp_path, "candidate"))
        deadline = time.monotonic() + 5
        while not inside and time.monotonic() < deadline:
            time.sleep(0.01)
        second = queue.start(corpus.Candidate("bbb", "b", spec, tmp_path, "candidate"))
        while not second.waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert second.as_dict()["waiting"] is True
        assert second.as_dict()["state"] == "running", "a page polling for running keeps polling"

        release.set()
        while second.state == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (first.state, second.state) == ("finished", "finished")
        assert most[0] == 1
        assert queue.last_seconds is not None

    def test_a_saved_draft_is_queued_from_its_own_page(self, world, monkeypatch) -> None:
        app, _ = world
        digest = _save(app, 180)
        started: list[str] = []

        def start(candidate, force=False):
            started.append(candidate.hash)
            return keeping.Measurement(ruleset_hash=candidate.hash, name=candidate.name)

        monkeypatch.setattr(server_module, "stale_build", lambda: None)
        monkeypatch.setattr(app.measurements, "start", start)
        assert app.post("/api/design/measure", {"hash": digest})["hash"] == digest
        assert started == [digest]

    def test_a_stale_build_refuses_before_anything_is_queued(self, world, monkeypatch) -> None:
        app, _ = world
        digest = _save(app, 180)
        monkeypatch.setattr(server_module, "stale_build", lambda: "the core is behind")
        with pytest.raises(ValueError, match="behind"):
            app.post("/api/design/measure", {"hash": digest})
        assert app.measurements.get(digest) is None

    def test_a_number_is_shown_only_on_a_current_basis(self, world) -> None:
        app, tmp_path = world
        digest = _save(app, 190)
        assert app.get(f"/api/design/standing/{digest}")["measured"] == {"state": "unscored"}

        _store(tmp_path, digest)
        now = app.get(f"/api/design/standing/{digest}")["measured"]
        assert now["state"] == "current" and now["index"] == 0.5

        _store(tmp_path, digest, weights="v0")
        stale = app.get(f"/api/design/standing/{digest}")["measured"]
        assert stale["state"] == "stale" and "index" not in stale
        assert stale["needs"] == "rescore"


def _walk(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [node, *[n for child in node["children"] for n in _walk(child)]]


class TestTheLineageIsALineage:
    def test_four_saves_of_one_idea_are_four_games_in_one_tree(self, world) -> None:
        app, _ = world
        dull = spec_hash(load_spec(RULESETS / "dull.json"))
        first = _save(app, 161, parent=dull)
        second = _save(app, 162, parent=first)
        third = _save(app, 163, parent=second)
        beside = _save(app, 164, parent=first)

        lineage = app.get(f"/api/design/standing/{third}")["lineage"]
        assert lineage["line"] == [dull, first, second, third]
        nodes = {n["hash"]: n for n in _walk(lineage["root"])}
        assert set(nodes) == {dull, first, second, third, beside}
        assert lineage["root"]["hash"] == dull and lineage["root"]["source"] == "committed"
        assert [c["hash"] for c in nodes[first]["children"]] == [second, beside]
        assert nodes[third]["here"] and not nodes[beside]["onLine"]

    def test_an_unsaved_draft_hangs_under_the_game_it_came_from(self, world) -> None:
        app, _ = world
        first = _save(app, 171)
        draft = spec_hash(designing.settled(_variant(172)))
        lineage = app.get(
            f"/api/design/standing/{draft}", {"parent": first, "name": "idea"})["lineage"]
        assert lineage["line"] == [first, draft]
        (child,) = lineage["root"]["children"]
        assert child["draft"] and child["name"] == "idea"

    def test_a_stale_member_carries_no_number(self, world) -> None:
        app, tmp_path = world
        first = _save(app, 181)
        second = _save(app, 182, parent=first)
        _store(tmp_path, first)
        _store(tmp_path, second, weights="v0")
        nodes = {n["hash"]: n for n in _walk(
            app.get(f"/api/design/standing/{second}")["lineage"]["root"])}
        assert nodes[first]["measured"]["index"] == 0.5
        assert "index" not in nodes[second]["measured"]

    def test_the_page_draws_every_node_and_no_stale_number(self) -> None:
        tree = {
            "hash": "aaa", "name": "root", "known": True, "here": False, "draft": False,
            "measured": {"state": "current", "index": 0.61, "seeds": [0]},
            "nominated": False,
            "children": [
                {"hash": "bbb", "name": "fork", "known": True, "here": True, "draft": False,
                 "measured": {"state": "stale", "needs": "replay"}, "nominated": True,
                 "children": []},
                {"hash": "ccc", "name": "idea", "known": False, "here": False, "draft": True,
                 "measured": {"state": "unscored"}, "nominated": False, "children": []},
            ],
        }
        html = _run_page("", f"lineageHtml({json.dumps(tree)})")
        assert all(f'data-lineage="{h}"' in html for h in ("aaa", "bbb", "ccc"))
        assert "0.61" in html and "needs a replay" in html and "not saved" in html
        assert "<b>0.61</b> as the one run" in html


class TestAMeasuredNumberSaysHowManyRuns:
    """Q339. The measurement rung and the lineage print a number, and the number says
    what it rests on, in the words a card's hover uses. A current measurement with no
    number reads *no score*, never the 0.00 `Number(null)` prints."""

    def test_the_standing_carries_the_seeds_beside_the_number(self, world) -> None:
        app, tmp_path = world
        digest = _save(app, 195)
        _store(tmp_path, digest)
        now = app.get(f"/api/design/standing/{digest}")["measured"]
        assert now["index"] == 0.5
        assert now["seeds"] == list(corpus.LIBRARY_SEEDS)

    def test_a_stale_measurement_carries_neither(self, world) -> None:
        app, tmp_path = world
        digest = _save(app, 196)
        _store(tmp_path, digest, weights="v0")
        stale = app.get(f"/api/design/standing/{digest}")["measured"]
        assert "index" not in stale and "seeds" not in stale

    @staticmethod
    def _said(measured: dict[str, Any]) -> str:
        return str(_run_page("", f"measuredNumberHtml({json.dumps(measured)})"))

    def test_four_runs_and_one_run_read_as_the_card_reads_them(self) -> None:
        four = self._said({"state": "current", "index": 0.5, "seeds": [0, 1, 2, 3]})
        assert four == "<b>0.50</b> as the mean of four runs"
        one = self._said({"state": "current", "index": 0.5, "seeds": [0]})
        assert one == "<b>0.50</b> as the one run"

    def test_no_number_and_no_count_both_read_no_score(self) -> None:
        for measured in (
            {"state": "current", "index": None, "seeds": [0, 1, 2, 3]},
            {"state": "current", "index": 0.5, "seeds": []},
            {"state": "current", "index": 0.5},
        ):
            said = self._said(measured)
            assert "no score" in said and "0.50" not in said and "0.00" not in said

    def test_the_measurement_rung_says_the_runs(self) -> None:
        standing = {
            "saved": True, "stage": "candidate",
            "measured": {"state": "current", "index": 0.5, "weights": "v10",
                         "seeds": [0, 1, 2, 3], "viable": True},
        }
        states = _run_page(
            "", f"rungStates({{ playable: true, smokeSeconds: 0.1, hash: 'h' }},"
                f" {json.dumps(standing)}, null)")
        said = states["canonical"]["says"]
        assert "<b>0.50</b> as the mean of four runs" in said and "under weights v10" in said

    def test_the_game_a_draft_starts_from_carries_its_seeds(self, world) -> None:
        """`measuredHtml` said the count when there was one, and the route never sent it,
        so the editor printed the number alone."""
        app, tmp_path = world
        digest = _save(app, 197)
        _store(tmp_path, digest)
        source = app.get(f"/api/design/{digest}")
        assert source["index"] == 0.5
        assert source["seeds"] == list(corpus.LIBRARY_SEEDS)

    def test_the_game_a_draft_starts_from_says_its_runs_or_no_score(self) -> None:
        current = {"current": True, "needs": ""}
        roster = ["random", "greedy"]

        def said(**parent: Any) -> str:
            return str(_run_page("", f"measuredHtml({json.dumps(parent)})"))

        four = said(index=0.5, seeds=[0, 1, 2, 3], roster=roster, staleness=current)
        assert "<b>0.50</b>" in four and "on 2 players, mean of four runs" in four
        for missing in ({"index": None, "seeds": [0, 1, 2, 3]}, {"index": 0.5, "seeds": []}):
            bare = said(**missing, roster=roster, staleness=current)
            assert "no score" in bare and "0.50" not in bare, missing
            assert "never measured" not in bare, "it was measured"
        assert "never measured" in said(index=None, seeds=[], roster=roster, staleness=None)


class TestOneRunIsEnoughForACandidate:
    """Q488. A candidate owes one run and a library game four (`corpus.expected_seeds`).

    The web reader used to expect four of every game, so a draft measured from its own
    page read *out of date* on the rung, in its family, on its page and on its card,
    while `corpus.refresh` found the same file current and did nothing when *measure it*
    was pressed again. Every web surface is asked here, because each reached the file
    by its own path through `library`.
    """

    def test_a_candidates_one_run_reads_current_everywhere_and_says_one_run(
        self, world
    ) -> None:
        app, tmp_path = world
        parent = _save(app, 201)
        digest = _save(app, 202, parent=parent)
        _store(tmp_path, digest, seeds=corpus.ONE_SEED)

        standing = app.get(f"/api/design/standing/{digest}")
        assert standing["measured"]["state"] == "current"
        assert standing["measured"]["index"] == 0.5
        assert standing["measured"]["seeds"] == list(corpus.ONE_SEED)

        nodes = {n["hash"]: n for n in _walk(standing["lineage"]["root"])}
        assert nodes[digest]["measured"]["index"] == 0.5
        assert nodes[digest]["measured"]["seeds"] == list(corpus.ONE_SEED)

        page = app.get(f"/api/game/{digest}")
        assert page["index"] == 0.5 and page["seeds"] == list(corpus.ONE_SEED)
        assert page["staleness"]["current"]
        kept = page["keeping"]
        assert kept["stage"] == "candidate" and kept["measurement"]["staleness"]["current"]

        (card,) = [g for g in app.get("/api/library")["games"] if g["hash"] == digest]
        assert card["index"] == 0.5 and card["seeds"] == list(corpus.ONE_SEED)
        assert card["staleness"]["current"]

        # And *measure it* has nothing left to do, so pressing it again is not a loop:
        # `corpus.refresh` and the page agree the number is current.
        found = corpus.find(
            digest, runs_dir=tmp_path / "runs", ruleset_dir=tmp_path / "rulesets",
            archive_dir=tmp_path / "archive",
        ).measurement(tmp_path / "runs")
        assert found is not None and found.current

    def test_a_library_games_one_run_still_reads_stale(self, world) -> None:
        app, tmp_path = world
        dull = spec_hash(load_spec(RULESETS / "dull.json"))
        _store(tmp_path, dull, seeds=corpus.ONE_SEED)

        page = app.get(f"/api/game/{dull}")
        # Its number is shown, marked provisional (EV2), and never as current.
        assert page["index"] == 0.5 and page["status"] == "provisional"
        assert not page["staleness"]["current"]
        assert page["staleness"]["needs"] == "replay"
        assert page["keeping"]["stage"] == "committed"
        assert not page["keeping"]["measurement"]["staleness"]["current"]

        (card,) = [g for g in app.get("/api/library")["games"] if g["hash"] == dull]
        assert card["status"] == "provisional" and not card["staleness"]["current"]

        # Four runs of the same game read current, so it is the count that staled it.
        _store(tmp_path, dull)
        assert app.get(f"/api/game/{dull}")["index"] == 0.5


class TestNominationNotPromotion:
    def test_a_candidate_is_marked_once_and_nothing_enters_rulesets(self, world) -> None:
        app, tmp_path = world
        before = sorted(p.name for p in (tmp_path / "rulesets").iterdir())
        digest = _save(app, 191)

        marked = app.post("/api/design/nominate", {"hash": digest, "note": "it races"})
        assert marked["created"] is True
        again = app.post("/api/design/nominate", {"hash": digest})
        assert again["created"] is False and again["note"] == "it races"

        (listed,) = app.get("/api/design")["nominations"]
        assert listed["hash"] == digest and listed["stage"] == "candidate"
        assert "chess-variant-sandbox test" in listed["blocker"]
        assert app.get(f"/api/design/standing/{digest}")["nominated"] is True
        assert sorted(p.name for p in (tmp_path / "rulesets").iterdir()) == before
        assert list((tmp_path / "archive").iterdir()) == []

    def test_a_game_already_in_the_yardstick_is_not_nominated(self, world) -> None:
        app, _ = world
        dull = spec_hash(load_spec(RULESETS / "dull.json"))
        with pytest.raises(designing.DesignError, match="already part"):
            app.post("/api/design/nominate", {"hash": dull})

    def test_the_editor_offers_no_way_to_archive_or_promote(self) -> None:
        from page_text import DESIGN

        assert "/api/promote" not in DESIGN and "/api/archive" not in DESIGN
        assert "/api/design/nominate" in DESIGN and "/api/design/measure" in DESIGN


_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.fetch = () => new Promise(() => {});
"""


def _run_page(setup: str, expression: str) -> Any:
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    body = _STUB + script.group(1) + "\n" + setup + "\n"
    body += "Promise.resolve(" + expression + ").then(v => console.log(JSON.stringify(v)));"
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _words(html: str) -> str:
    """A rendered panel as one line of words, whatever the source's line breaks."""
    return " ".join(str(html).split())


class TestTheMeasuringRungInPlainWords:
    """Q489: what a person tripped on driving *round king hunt* (`c5e5b57cab2b`) from a
    template to a measured number on the designer's page alone (Q421)."""

    def test_the_long_rungs_time_names_its_unit_in_the_unit_a_person_would_use(
        self,
    ) -> None:
        """An eight-second run read *the last one here took 0.1*: minutes, unnamed."""
        cost = candidacy.rungs(8.2)[4]["cost"]
        assert "8 seconds" in cost and "0.1" not in cost
        assert candidacy.took(1.0) == "1 second"
        assert candidacy.took(60.0) == "1 minute"
        assert candidacy.took(150.0) == "2.5 minutes"
        assert candidacy.took(1800.0) == "30 minutes"

    def test_a_run_queued_behind_another_says_so_from_the_press(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The press's own answer is the run before its thread has asked for the turn,
        and it read *measuring* even while another measurement held the turn."""
        asked = threading.Event()

        class Done:
            def as_dict(self) -> dict[str, Any]:
                return {"index": 0.5}

        monkeypatch.setattr(keeping.corpus, "refresh", lambda candidate, **_: Done())
        queue = keeping.Measurements(tmp_path)
        measure = queue._measure

        def held(*args: Any) -> None:
            # The thread does not look at the turn until the press's answer is read, so
            # what that answer says is `start`'s own doing.
            asked.wait(5)
            measure(*args)

        queue._measure = held  # type: ignore[method-assign]
        spec = _variant(161)
        queue._turn.acquire()
        try:
            run = queue.start(corpus.Candidate("ccc", "c", spec, tmp_path, "candidate"))
            assert run.as_dict()["waiting"] is True
            assert run.as_dict()["state"] == "running"
            asked.set()
        finally:
            asked.set()
            queue._turn.release()
        deadline = time.monotonic() + 5
        while run.state == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert run.state == "finished" and run.waiting is False

    def test_the_rung_says_queued_starting_or_how_long_it_has_run(self) -> None:
        said = _run_page("", "[measuringWords({ waiting: true, seconds: 4, log: [] }),"
                             " measuringWords({ waiting: false, seconds: 0, log: [] }),"
                             " measuringWords({ waiting: false, seconds: 5,"
                             "   log: ['c: no score yet', '  scoring'] })]")
        assert said[0].startswith("queued behind another measurement")
        assert said[1] == "starting"
        assert said[2] == "measuring for 5 seconds so far; last step: scoring"
        from page_text import DESIGN

        assert "measuring, at" not in DESIGN

    def test_what_promotion_needs_is_said_without_a_command(self, world) -> None:
        """Before measuring it named a terminal command beside the button that does the
        job; after one run it said *out of date -- stale (replay 3 of 4 runs)*, when one
        run is exactly what the rung owes and promotion wants four."""
        app, tmp_path = world
        digest = _save(app, 203)
        before = app.get(f"/api/design/standing/{digest}")
        assert "no score yet" in before["promotion"]["says"]
        assert "four runs" in before["promotion"]["says"]

        _store(tmp_path, digest, seeds=corpus.ONE_SEED)
        after = app.get(f"/api/design/standing/{digest}")
        assert after["measured"]["state"] == "current"
        says = after["promotion"]["says"]
        assert says.startswith("One run is measured") and "four runs" in says
        assert "the other three are played first" in says
        for standing in (before, after):
            words = standing["promotion"]["says"]
            assert "`" not in words and "uv run" not in words and "stale" not in words
            assert standing["promotion"]["command"] == (
                f"uv run chess-variant-sandbox test {digest}")
        # `corpus`'s own refusal is unchanged: the command line still names the command.
        assert "chess-variant-sandbox test" in after["blocker"]

        _store(tmp_path, digest)
        assert app.get(f"/api/design/standing/{digest}")["promotion"] is None

    def test_a_stored_measurement_is_read_against_the_runs_its_rung_owes(
        self, world
    ) -> None:
        """`GET /api/test/<hash>` (`Measurements.stored`) read every game against the
        library's four runs. The page reads only its state today, but a candidate's one
        run is current and a library game's one run is not, here as everywhere."""
        app, tmp_path = world
        digest = _save(app, 204)
        _store(tmp_path, digest, seeds=corpus.ONE_SEED)
        candidate = app.get(f"/api/test/{digest}")["result"]
        assert candidate["staleness"]["current"] is True, candidate["staleness"]

        dull = spec_hash(load_spec(RULESETS / "dull.json"))
        _store(tmp_path, dull, seeds=corpus.ONE_SEED)
        library = app.get(f"/api/test/{dull}")["result"]
        assert library["staleness"]["current"] is False
        assert library["staleness"]["missingSeeds"], library["staleness"]

    def test_the_command_is_folded_away_and_never_drawn_on_the_site(self) -> None:
        standing = json.dumps({"stage": "candidate", "promotion": {
            "says": "One run is measured.", "command": "uv run x test abc"}})
        local, site = (_words(_run_page(setup, f"promotionHtml({standing})")) for setup in (
            "", "ACCOUNTS = true;"))
        assert "One run is measured." in local and "<details" in local
        assert "uv run x test abc" in local
        # Q576: on the site promotion is the track's, so none of the sentence is drawn.
        assert site == ""
        from page_text import DESIGN

        assert "esc(s.blocker)" not in DESIGN, "the page shows corpus's refusal again"


def _visible(html: str) -> str:
    """What a reader sees of a rendered panel: its words, with every tag dropped."""
    return _words(re.sub(r"<[^>]+>", " ", str(html)))


# The maintainer's sentences the first-visit study's readers met on the site (Q576, B2 and
# F6 of `docs/reference/first-visit-study.md`). Each is right on a checkout and never drawn
# on the site, where `ACCOUNTS` is true.
_MAINTAINERS = ("command line", "runs/", "git ignores", "yardstick", "256 games",
                "games a pairing", "PATH", "CHESS_VARIANT_SANDBOX_FAIRY_STOCKFISH")


class TestTheSitesWordsAreAVisitors:
    """Q576: the site's words split from the local application's, by the one switch the
    page tells the two apart by (`ACCOUNTS`). The checkout keeps every sentence."""

    _HASH = "c5e5b57cab2b"
    _OTHER = "0a1b2c3d4e5f"

    def _standing(self, **more: Any) -> str:
        promotion = {"says": "It has no score yet. Measuring it plays one run of 256 games.",
                     "command": "uv run chess-variant-sandbox test " + self._HASH}
        standing = {
            "hash": self._HASH, "saved": True, "stage": "candidate",
            "measured": {"state": "unscored"}, "nominated": False, "build": "",
            "rungs": candidacy.rungs(None), "promotion": promotion,
            "lineage": {"cut": False, "root": {
                "hash": self._OTHER, "name": "the parent", "known": True,
                "measured": {"state": "unscored"}, "children": [
                    {"hash": self._HASH, "name": "the draft", "known": True, "here": True,
                     "measured": {"state": "unscored"}, "children": []}]}},
            **more,
        }
        return (
            f"DRAFT = {{ name: 'the draft', pieces: [] }}; DRAFT_HASH = '{self._HASH}';"
            f"DRAFT_PARENT = {{ hash: '{self._OTHER}', name: 'the parent', index: null,"
            "  seeds: [], roster: [], staleness: null };"
            f"DRAFT_STANDING = {json.dumps(standing)};"
            f"DESIGN_KIT = {{ rungs: DRAFT_STANDING.rungs, nominations: [{{ name: 'n',"
            f"  hash: '{self._OTHER}', stage: 'candidate', command: 'uv run x promote' }}] }};"
        )

    def _panels(self, setup: str) -> str:
        check = f"{{ hash: '{self._HASH}', valid: true, playable: true, schemaVersion: 2 }}"
        return " ".join(_run_page(setup, f"[ladderHtml({check}), identityHtml({check}),"
                                         " familyHtml(), promotionHtml(DRAFT_STANDING)]"))

    def test_the_designers_panel_speaks_to_a_visitor_on_the_site(self) -> None:
        """F6: *runs/designed/, which git ignores*, *command line*, *yardstick*, the hash and
        *256 games* on the designer's right-hand panel, each kept on the checkout."""
        local = self._panels(self._standing())
        site = self._panels("ACCOUNTS = true;" + self._standing())
        for said in ("command line", "runs/designed/, which git ignores", "yardstick",
                     "256 games", "games a pairing"):
            assert said in local, said
        assert self._HASH in _visible(local) and self._OTHER in _visible(local)
        for said in _MAINTAINERS:
            assert said not in site, said
        assert self._HASH not in _visible(site) and self._OTHER not in _visible(site)
        # The same rungs in the same order, in a visitor's words.
        assert "kept on this site as your draft" in site
        assert "measured against every computer player" in site
        assert "A game is its rules" in site and "the parent's rules" in site

    def test_a_baseline_games_last_rung_reads_in_a_visitors_words(self) -> None:
        local = self._panels(self._standing(stage="committed"))
        site = self._panels("ACCOUNTS = true;" + self._standing(stage="committed"))
        assert "already part of the yardstick" in local
        assert "yardstick" not in site and "already one of them" in site

    def test_each_rung_carries_a_visitors_words_beside_the_maintainers(self) -> None:
        """The words differ and nothing else: the keys, costs and order are one list."""
        for rung in candidacy.rungs(300.0):
            assert set(rung["site"]) <= {"name", "what", "cost", "runs"}, rung["key"]
            site = {**rung, **rung["site"]}
            for said in _MAINTAINERS:
                assert said not in site["name"] + site["what"], (rung["key"], said)
        assert set(candidacy.SITE_WORDS) <= {r["key"] for r in candidacy.rungs(None)}

    def test_the_designers_sentence_says_where_a_save_goes_on_each(self) -> None:
        from page_text import DESIGN

        assert "'A saved game is kept on this site as your draft.'" in DESIGN
        assert "invented${info('candidate')}" in DESIGN

    def test_an_absent_outside_engine_is_not_explained_on_the_site(self) -> None:
        """B2: *no Fairy-Stockfish is installed on this machine. Put one on PATH...* is the
        maintainer's remedy. On the site an engine nobody installed is simply absent."""
        for translates in (True, False):
            game = json.dumps({"fairyCheck": {"engine": None}, "fairyTranslates": translates,
                               "fairyReasons": ["no counterpart"]})
            local, site = (_run_page(setup, f"fairyNote({game}, [])")
                           for setup in ("", "ACCOUNTS = true;"))
            assert "is not offered" in local, translates
            assert site == "", translates
        assert "Put one on PATH" in _words(_run_page("", "fairyNote({ fairyCheck: {},"
                                                         " fairyTranslates: true }, [])"))
        # An engine the site does have, and plays differently, is still said, in a
        # visitor's words and without the position the two engines disagreed on (G8).
        differs = json.dumps({"fairyCheck": {"engine": "fsf", "ok": False, "reasons": ["x"],
                                             "fen": "8/8/8/8/8/8/8/8 w - - 0 1"},
                              "fairyTranslates": True})
        site = _visible(_run_page("ACCOUNTS = true;", f"fairyNote({differs}, [])"))
        assert "plays these rules a little differently, so it isn't offered" in site
        assert "8/8/8" not in site and "did not play the same game" not in site
        assert "did not play the same game" in _words(_run_page("", f"fairyNote({differs}, [])"))
        # Rules it can't play: one sentence, no reasons (G7); a check not run: nothing (G9).
        cannot = json.dumps({"fairyCheck": {"engine": "fsf"}, "fairyTranslates": False,
                             "fairyReasons": ["piece.captures: no such property there"]})
        site = _visible(_run_page("ACCOUNTS = true;", f"fairyNote({cannot}, [])"))
        assert site == "Fairy-Stockfish i can't play these rules.", site
        unrun = json.dumps({"fairyCheck": {"engine": "fsf"}, "fairyTranslates": True})
        assert _run_page("ACCOUNTS = true;", f"fairyNote({unrun}, [])") == ""
        assert "has not run" in _words(_run_page("", f"fairyNote({unrun}, [])"))

    # ---- W0 (Q607): the words guard. `site_words.py` is the list; `test_site_words.py` reads
    # what a running site sends. This is the page's own script, run with `ACCOUNTS` true.

    def test_the_pages_own_words_hold_no_phrase_the_audit_has_removed(self) -> None:
        found = served_by_the_page()
        assert len(found) > 40, "the guard read almost nothing"
        trouble = [f"{where}: {phrase!r} in {text[:90]!r}"
                   for where, text in found for phrase in site_words.hits(text)]
        assert not trouble, "\n".join(trouble[:40])

    def test_a_phrase_added_back_to_an_enforced_spot_fails_the_guard(self) -> None:
        """Every enforced phrase is caught in a sentence, and a word that only contains one
        is not (`digit` is not `git`), and a lone hash is data, not prose."""
        assert set(site_words.ENFORCED) | set(site_words.NOT_YET) == set(site_words.PHRASES)
        samples = {
            "runs/": "Kept in runs/designed.", "git": "Which git ignores.",
            "command line": "The command line prints it.",
            "chess-variant-sandbox ": "Run chess-variant-sandbox test now.",
            "PATH": "Put one on PATH.", "minimax:": "It beat minimax:h=2.",
            "nodes=": "Played at nodes=400.", "a twelve-hex hash": "It is c5e5b57cab2b here.",
            "plies": "Drawn after 100 plies.", "weights v": "Under weights v11.",
            "checkout": "Not by this checkout.", "corpus": "Measured on the corpus.",
            "schema": "The rule schema.", "candidate": "Kept as a candidate.",
            "yardstick": "Part of the yardstick.", "leaderboard": "A leaderboard game.",
            "sandbox mode": "Watch it in sandbox mode.",
            "searches N positions": "easy - searches 400 positions",
            "lookahead": "greedy - one move of lookahead",
            "search depth": "Grouped by search depth.",
            "playouts": "tree search - 1,600 random playouts",
            "positions a move": "Two searching players at 6,400 positions a move.",
            "an engine somebody else wrote": "Group: an engine somebody else wrote.",
            "rungs": "A rung added later reopens it.",
            "tops beaten": "The order is tops beaten.",
            "behaviour grid": "By the kinds the behaviour grid reads.",
            "idle sweep": "Until the idle sweep resigns it.",
            "maintainer": "The maintainer's reason: why.",
            "beat the top": "Nobody has beaten the top on it.",
            "a game that counts": "Sign in to play a game that counts.",
            "the sweep": "A game the sweep closed.",
            "does not offer yet": "A tournament the site does not offer yet.",
            "measure or nominate": "It does not measure or nominate games.",
            "a reading from a computer": "A reading from your computer, checked.",
            "refresh it": "Only its designer can refresh it.",
            "similarity index": "Their similarity index is 3.",
            "the cut": "Drawn from the cut at 2 up.",
            "design step": "One design step from it makes this.",
            "fifty-five features": "By a count of fifty-five features.",
            "human games": "It has 12 human games.",
            "the gate says": "Playable: the gate says no.",
            "nobody's played this": "A card saying nobody's played this.",
            "revisions deep": "Ordered by how many revisions deep.",
            "translates to Fairy-Stockfish": "A tag: translates to Fairy-Stockfish.",
            "rule set version": "Written against rule set version 2.",
            "the engine answered": "The engine answered it, not the page.",
            "searching player": "Played by a searching player.",
            "not a ruleset yet": "This draft is not a ruleset yet.",
            "not an object": "The draft is not an object.",
            "play-tested": "It could not be play-tested: boom.",
            "makes nothing of this game": "It makes nothing of this game under seed 3.",
            "asking the engine": "Now asking the engine...",
            "a path into the game's text":
                "setup.placements[0].squares entry 'z9' is not a square on this board.",
            "does not compile": "It does not compile: boom.",
            "run out of clock": "93% of random games run out of clock.",
            "one seat": "One seat wins 90% more often.",
            "under random play": "Decided within 6 turns even under random play.",
            "the clock cut short": "A sample the clock cut short is not graded.",
            "a rule's key": "Changed max plies from 200 to 300.",
            "a declared piece": "'Q' is not a declared piece id.",
            "a non-empty string": "The name must be a non-empty string.",
            "ceiling": "It scored 0.43 in the ceiling duel.",
            "floor": "0.30 is the floor.",
            "verdict": "The check's verdict is fails.",
            "Fairy-Stockfish's own terms": "Betza has no ride that starts past its first square.",
        }
        assert set(samples) == set(site_words.PHRASES)
        for name, sentence in samples.items():
            assert site_words.hits(sentence, (name,)) == [name], name
        assert site_words.hits("A digit and a target.", ("git",)) == []
        assert site_words.hits("Ralph Betza's chancellor.", ("Fairy-Stockfish's own terms",)) == []
        # *Not a verdict* says what a score is not, in the purpose's own words, and stays.
        assert site_words.hits("The best estimate today, not a verdict.", ("verdict",)) == []
        assert list(site_words.strings({"hash": "c5e5b57cab2b", "says": "Two words"})) == [
            (".says", "Two words")]
        assert set(site_words.NOT_YET) <= set(site_words.PHRASES)
        assert all(re.search(r"\b(?:[WT]\d+[ab]?|Q\d+)\b", why)
                   for why in site_words.NOT_YET.values()), "each names the task that removes it"

    def test_the_game_pages_hash_is_the_checkouts_alone(self) -> None:
        """F6 on a draft's game page: the hash closing the line under its name."""
        from page_text import PAGE

        start = PAGE.index("complexity ${game.ruleMass}")
        line = PAGE[start:PAGE.index("</div>", start)]
        assert "${ACCOUNTS ? '' : `<span>&middot;</span>" in line
        assert "${esc(game.hash)}</span>`}" in line

    def test_the_editors_trial_says_the_trial_is_not_recorded(self) -> None:
        """N4: *watch it on its own page* called a saved game *not saved*. The trial is the
        thing that is not kept, so that is what it says, on both."""
        from page_text import PAGE

        assert "this trial isn't recorded" in PAGE
        assert "which is not saved" not in PAGE

    def test_a_saved_draft_is_one_row_that_reads_its_fresh_measurement(self) -> None:
        """After a save the draft *is* its parent: one hash, listed once, and the number
        the standing has, not the *never measured* fetched before it was measured."""
        setup = (
            "DRAFT = { name: 'round king hunt', pieces: [] };"
            "DRAFT_PARENT = { hash: 'c5e5b57cab2b', name: 'round king hunt', index: null,"
            "  seeds: [], roster: [], staleness: null };"
            "DRAFT_STANDING = { hash: 'c5e5b57cab2b', saved: true, measured:"
            "  { state: 'current', index: 0.6388, seeds: [0], viable: true } };"
        )
        said = _words(_run_page(
            setup, "identityHtml({ hash: 'c5e5b57cab2b', schemaVersion: 2 })"))
        assert said.count("c5e5b57cab2b") == 1, said
        assert "never measured" not in said
        assert "<b>0.64</b> as the one run" in said
        running = _words(_run_page(
            setup + "DRAFT_STANDING.measuring = { state: 'running', waiting: false };",
            "identityHtml({ hash: 'c5e5b57cab2b', schemaVersion: 2 })"))
        assert "being measured now" in running

    def test_every_template_brings_its_own_name_and_sentence(self) -> None:
        """Choosing a template kept the name *new game* and an empty sentence."""
        from chess_variant_sandbox.spec import validate_spec

        names = set()
        for template in designing.templates():
            spec = template["spec"]
            assert spec["name"] and spec["name"] != "new game", template["key"]
            assert len(spec["description"]) > 40, template["key"]
            assert validate_spec(spec) == [], template["key"]
            names.add(spec["name"])
        assert len(names) == len(designing.templates())

    def test_a_chosen_rule_survives_the_menu_being_redrawn(self) -> None:
        """The refusals land about a second after any change and redraw the menu. The
        choice used to go with it, leaving *add it* off, so the first press did nothing
        and the second, after choosing again, worked (reproduced by hand, Q489)."""
        setup = """
          const tOption = value => ({ value });
          const tMenuOf = value => ({ value, options: [tOption(''), tOption('wrap')],
                                     onchange: null });
          let tMenu = tMenuOf('wrap'), tGo = { disabled: false };
          const tBox = {
            set innerHTML(html) { tMenu = tMenuOf(''); tGo = { disabled: true }; },
            querySelector: id => (id === '#dAddRule' ? tMenu : tGo),
          };
          const tForm = { querySelector: () => tBox };
          document.getElementById = id => (id === 'form' ? tForm : null);
          ruleMenuHtml = () => '';
          wireCards = () => { tMenu.onchange = () => { tGo.disabled = !tMenu.value; }; };
        """
        after = _run_page(setup, "(redrawRuleMenu(), [tMenu.value, tGo.disabled])")
        assert after == ["wrap", False]


# The (i)s `docs/archive/plan-understanding.md` §3.1 confirms are never drawn on the site: they sit
# behind `!ACCOUNTS`, a withheld route, or have no caller. The local application keeps them.
_NOT_DRAWN_ON_THE_SITE = frozenset({
    "unscored", "lineage", "candidate", "archived", "committed", "rulesetHash",
    "designRun", "thumb",
})


def _score_panel(site: bool, ruled_out: bool = False) -> str:
    """A game's score panel, drawn by the page over the current scoring rules' own guide
    as the game page is sent it (`library.score_guide`), with its band, one refusing run of
    four, a lean toward one side and a part's roster check. `ruled_out` draws a score the
    tests ruled out instead, each refusal as the measurement filed it."""
    from chess_variant_sandbox.metrics.glossary import explain
    from chess_variant_sandbox.web import library, sitescore

    version = explain()["version"]
    guide = library.score_guide({"weights_version": version})
    assert guide is not None
    names = sorted(guide["components"])
    guide["components"][names[0]] = {**guide["components"][names[0]], "roster_rho": 0.62}
    unreadable = ("this roster cannot read this ruleset: the rating ladder orders the agents on"
                  " only 40% of its budget steps, so lead_conversion, skill_gradient would"
                  " describe the players rather than the game")
    gate = ("quick_kill_rate_worst_half=0.538 above maximum 0.5 (a game where half of all"
            " contests are over in a handful of moves is decided by an opening trap)")
    band = {"component": "seat_decided_chance", "wins": 41, "losses": 20, "decided": 61,
            "favoured": "first", "share": 0.6721,
            "lean": "the first seat took 41 of the 61 decided games between the two strongest"
                    " rungs (67%), more than the 67% the seat check tolerates"}
    score = {
        "total": 0.61, "viable": not ruled_out, "weights_version": version,
        "interval": {"sd": 0.02, "ci_low": 0.57, "ci_high": 0.65},
        "seeds": [0, 1000, 2000, 3000], "totals_used": 3, "games": 1024,
        "violations": ["run 1000: " + gate, "run 2000: " + unreadable] if ruled_out else [],
        "runs": [{"seed": seed, "index": None if seed == 2000 else 0.6,
                  "viable": seed != 2000 and not (ruled_out and seed == 1000),
                  "violations": [unreadable] if seed == 2000
                  else [gate] if ruled_out and seed == 1000 else []}
                 for seed in (0, 1000, 2000, 3000)],
        "verdict": {"accept": 3, "refuse": 1, "split": "Three of its four runs accept it and"
                                                       " one refuses it, so it stays in."},
        "seat": {"tolerated_share": 0.6667, "refuses": False, "lean": band["lean"],
                 "bands": {"top_rung_seat": band}},
        "families": {family: 0.5 for family in guide["families"]},
        "scores": {name: 0.5 for name in names},
        "contributions": {name: guide["components"][name]["weight"] for name in names},
        "raw": {name: 1.0 for name in names},
    }
    worded = sitescore.worded(score)
    band_of = {"of": 70, "clears": 3, "clearedBy": 12, "close": 55}
    return str(_run_page(("ACCOUNTS = true;" if site else ""),
                         f"scoreHtml({json.dumps(worded)}, {json.dumps(guide)},"
                         f" {json.dumps(band_of)})"))


# A game page's own views, as `served_by_the_page` and the W3 tests render them: one game
# payload with a stale score, a ladder table of the canonical players that scored it, a
# picker table, a refused store, a provenance from a design run, a link and a draft's marks.
_LADDER = ("random", "greedy", "minimax:nodes=800", "minimax:nodes=6400")


def _game_for_views(stage: str = "Baseline") -> dict[str, Any]:
    from chess_variant_sandbox.site import stages
    from chess_variant_sandbox.web import opponents

    roster = [entry.as_dict() for entry in opponents.ladder()]
    return {
        "hash": "abc123def456", "name": "a test game", "source": "committed",
        "staleness": {"current": False, "needs": "rescore",
                      "summary": "stale (re-score): the metrics fingerprint moved"},
        "provisional": None, "score": None, "keeping": {}, "build": "",
        "opponents": roster,
        "ratings": {"games": 1024, "seeds": [0, 1000, 2000, 3000], "standings": [
            {"name": agent, "rating": 100.0 * index, "games": 224,
             "label": opponents.table_name(agent)} for index, agent in enumerate(_LADDER)]},
        "pickerRatings": None,
        "refusedStores": [opponents.unavailable("own piece values")],
        "provenance": {"origin": "evolved", "run": "run-2026-09-01", "generation": 3,
                       "theme": "more drops", "parents": ["0a1b2c3d4e5f"]},
        "links": [{"hash": "0a1b2c3d4e5f", "name": "near", "units": 1, "likeness": 0.62,
                   "shares": ["the camel"], "line": True,
                   "kinds": ["rule"], "differences": [{"sentence": "a draw clock"}],
                   "similarity": {"words": "1 rule, 28 squares, 1 clock or price",
                                  "site": "1 rule, 28 starting squares, 1 limit or price"},
                   "gap": None}],
        "onMap": True, "community": {"id": "f1", "name": "near chess"}, "open": 5,
        "designer": {"passesScreen": True, "tried": 3, "forkedBy": ["bo"],
                     "rules": {"passesScreen": "Its rules passed the playability check."}},
        "shortlist": {"says": "On trial.", "checks": [], "mine": True, "nominable": True},
        **stages.shown(stage),
    }


# W2 (rows B1-B23, M1-M13): the library's controls, legend and cards, and the map's panels,
# as the site draws them. Every card state the audit names, hovers included.
_LIBRARY_SETUP = """
ACCOUNTS = true;
SITE = { sorts: [], tabs: [] };
COVERAGE = { axes: [{ axis: 'hand', asks: 'whether captured pieces come back',
  bins: [{ bin: 'none', means: 'captured pieces are gone for good' }] }] };
const SPLIT = { split: 'x', accept: 3, refuse: 1,
  site: { split: 'Three of its four runs pass it and one rules it out, so it stays in.' } };
const RUNS = [{ seed: 0, viable: true }, { seed: 1000, viable: true },
  { seed: 2000, viable: true },
  { seed: 3000, viable: false, violations: ['top_rung_first_seat_score=1.2e-04 below'],
    site: { violations: ['In one run, the strongest computer players won from one side'
      + ' almost every time.'] } }];
const CARD = { hash: 'aaaaaaaaaaaa', name: 'a game', description: 'A game.',
  source: 'committed', level: 'Baseline', levelGloss: 'baseline', board: '8x8',
  pieceTypes: 6, ruleMass: 52, squares: 64, descent: 2, cell: '0', cellBins: ['none'],
  index: 0.612, viable: true, playable: true, seeds: [0, 1000, 2000, 3000],
  weights: 'v11', staleness: { current: true }, families: {}, components: {},
  family: { hash: 'bbbbbbbbbbbb', name: 'its parent' }, run: '', theme: '',
  runVerdict: SPLIT, runs: RUNS, humanGames: 12, votes: 2, stars: 4.5,
  fairyTranslates: true, rated: true, unbeaten: true,
  top: ['minimax:h=2,nodes=6400'] };
LIBRARY = [CARD];
"""

_LIBRARY_CARDS = (
    "[CARD,"
    " { ...CARD, staleness: { current: false, summary: 'stale (replay 1 of 4 runs,"
    " re-score 3): the core, the agents or the game loop changed' } },"
    " { ...CARD, index: null, totalsUsed: 2 },"
    " { ...CARD, index: null, provisional: { state: 'spot-checked', sample: 32,"
    "   yours: false, index: 0.6, viable: true, seeds: [0], weights: 'v11' } },"
    " { ...CARD, playable: false, playableReasons: ['the rules can never be won'] },"
    " { ...CARD, unbeaten: false, firstClear: { handle: 'ada' }, clears: 3 },"
    " { ...CARD, rated: false },"
    " { ...CARD, humanGames: 0, untried: true }]"
)

_MAP_SETUP = """
MAP_DATA = {
  counts: { nodes: 70, specs: 71, twins: 1, lines: 120, lineage: 9, communities: 6,
            unusual: 4 },
  mutual: 5,
  scale: { low: 0.1, high: 0.8 }, components: { skill_gradient: 'does thinking longer win' },
  kinds: [{ kind: 'rules', words: 'how it is won', lines: 3 }],
  communities: [{ id: 'f1', name: 'the chess family', words: 'the castling move', size: 5,
                  names: ['a game'], members: ['aaaaaaaaaaaa'], centre: 'aaaaaaaaaaaa',
                  parts: [['aaaaaaaaaaaa'], ['cccccccccccc']] }],
  unusual: [{ hash: 'cccccccccccc', distance: 0.8 }],
  neighbours: [{ source: 'aaaaaaaaaaaa', target: 'cccccccccccc', likeness: 0.42,
                 shares: ['the castling move'], units: 4, kind: '', kinds: [],
                 sentence: '', differences: [], gap: null }],
  lineage: [],
  nodes: [
    { hash: 'aaaaaaaaaaaa', name: 'a game', names: ['a game', 'a twin'], degree: 1,
      unusual: false, board: '8x8', community: 'f1', score: null, staleness: null,
      like: [{ hash: 'cccccccccccc', likeness: 0.42, shares: ['the castling move'],
               line: true }] },
    { hash: 'cccccccccccc', name: 'another', names: ['another'], degree: 1, unusual: true,
      board: '8x8', community: null, score: null, staleness: null,
      like: [{ hash: 'aaaaaaaaaaaa', likeness: 0.42, shares: [], line: true }] }],
};
MAP_AT = Object.fromEntries(MAP_DATA.nodes.map(node => [node.hash, node]));
"""


def _library_and_map_views() -> list[tuple[str, str]]:
    """What the library and the map draw on the site (W2, rows B1-B23 and M1-M13)."""
    from chess_variant_sandbox.web import library

    setup = _LIBRARY_SETUP + f"METRICS = {json.dumps(library.metric_legend())};"
    found: list[tuple[str, str]] = []

    def view(where: str, call: str, more: str = "") -> None:
        found.append((where, site_words.visible(str(_run_page(setup + more, call)))))

    view("the library's controls", "controlsHtml()")
    view("the library's legend", "pillLegendHtml()")
    view("the library's cards", f"{_LIBRARY_CARDS}.map(cardHtml).join(' ')")
    view("the library's cards, ordered by a part",
         "cardHtml({ ...CARD, components: { stall_rate: 0.1 } })",
         "BROWSE.sort = 'component:stall_rate';")
    for call in ("mapNothingPanel()", "mapNodePanel(MAP_AT.aaaaaaaaaaaa)",
                 "mapNodePanel(MAP_AT.cccccccccccc)", "mapLinePanel(MAP_DATA.neighbours[0])",
                 "mapCommunityPanel(MAP_DATA.communities[0])",
                 "mapLegendHtml()", "mapLikeHtml(MAP_AT.aaaaaaaaaaaa)",
                 "mapMovedHtml({ moved: [{ key: 'skill_gradient', index: 0.4,"
                 " otherIndex: 0.6 }] })",
                 "mapArrowPanel({ source: 'cccccccccccc', target: 'aaaaaaaaaaaa', units: 1,"
                 " run: 'evolve-20260930', generation: 3 })",
                 "[{ state: 'asking' }, { state: 'done', steps: [] }, { state: 'done',"
                 " steps: [{ operator: 'o', seed: 1, words: 'removed the pawn' }] }]"
                 ".map(held => (MAP_TRY.aaaaaaaaaaaa = held,"
                 " mapTryHtml(MAP_AT.aaaaaaaaaaaa))).join(' ')"):
        view(f"the map ({call})", call, _MAP_SETUP)
    return found


# W5a (rows D1-D39): the designer's own words, as the site draws them: the ladder in every
# state, the save's answer, which game this is, the family, the move marks, a taught move,
# the rule cards' refusals and the browser's own measurement.
def _designer_views() -> list[tuple[str, str]]:
    words = TestTheSitesWordsAreAVisitors()
    check = (f"{{ hash: '{words._HASH}', valid: true, playable: true, smokeSeconds: 0.12,"
             " schemaVersion: 2, settledVersion: true, declaredVersion: null }")
    base = "ACCOUNTS = true; VIEWER = { handle: 'ada' };"
    found: list[tuple[str, str]] = []

    def view(where: str, setup: str, call: str) -> None:
        found.append((where, site_words.visible(str(_run_page(base + setup, call)))))

    measured = {"state": "current", "index": 0.62, "viable": False, "weights": "v11",
                "seeds": [0, 1000, 2000, 3000]}
    standings = {
        "measured": {"measured": measured},
        "stale": {"measured": {"state": "stale", "needs": "replay"}},
        "nominated": {"nominated": True},
        "a library game": {"stage": "committed", "name": "chess"},
        "held": {"build": "rebuild with uv sync", "measured": {"state": "unscored"}},
    }
    for state, more in standings.items():
        setup = words._standing(**more) + (
            f"DRAFT_SCREEN = {{ hash: '{words._HASH}', passed: true, seconds: 0.4 }};")
        view(f"the designer's ladder ({state})", setup, f"ladderHtml({check})")
    parent = ("{ hash: '0a1b2c3d4e5f', name: 'the parent', index: 0.61, seeds: [0, 1, 2, 3],"
              " roster: [1, 2, 3, 4, 5, 6, 7, 8, 9], staleness: { current: true } }")
    for state, draft_parent in (("a fork", parent), ("no parent", "null"),
                                ("an out-of-date parent", "{ ...P, index: null,"
                                 " staleness: { current: false, needs: 'replay' } }")):
        view(f"which game this is ({state})",
             words._standing() + f"const P = {parent}; DRAFT_PARENT = {draft_parent};",
             f"identityHtml({check}) + familyHtml()")
    view("the save's answer", words._standing(),
         "savedHtml({ created: true, name: 'n', hash: 'aaaaaaaaaaaa', rung: 'screen',"
         " rungPassed: true })")
    view("the move marks", "DRAFT = { pieces: [{ id: 'K', name: 'king' }] }; DRAFT_PIECE = 0;",
         "[(DRAFT_MAP = { width: 8, from: 3, live: true, standing: [3] }, moveKeyHtml()),"
         " (DRAFT_MAP = { width: 8, from: 3, live: false, captureOnly: [] }, moveKeyHtml())]"
         ".join(' ')")
    view("a taught move", "",
         "[{ targets: [1], busy: true, offers: [] }, { targets: [1], offers: [] },"
         " { targets: [1], offers: [{ says: 'leaps', exact: true, extra: [] }] }]"
         ".map(t => (TEACHING = t, teachOffersHtml(8))).join(' ')")
    view("the rule cards' refusals", "DRAFT_CHECK = { valid: false };",
         "refusedRulesHtml(['x'], { x: { label: 'x' } })")
    # Q995b: the reasons themselves, as the server words them, list opened. From chess, a
    # game in rounds is refused for three rules at once.
    chess = json.loads((RULESETS / "chess.json").read_text(encoding="utf-8"))
    refused = designing.refusals(chess, {"simultaneous": {}, "promote_on_entry": True})
    keys = json.dumps(sorted(refused["refused"]))
    view("the rule cards' refusals, every reason", f"RULE_REFUSED = {json.dumps(refused['refused'])};",
         f"refusedRulesHtml({keys}, Object.fromEntries({keys}.map(k => [k, {{ label: k }}])))")
    # And how the players take turns, with what a choice changed and what refused one.
    turned = designing.take_turns(chess, "rounds")
    sweeping = json.loads(json.dumps(chess))
    sweeping["pieces"][2]["moves"][0]["kind"] = "sweep"
    stuck = designing.take_turns(sweeping, "rounds")
    view("how the players take turns", f"DRAFT = {json.dumps(turned['draft'])};",
         f"[turnSaidHtml({{ way: 'rounds', changes: {json.dumps(turned['changes'])} }}),"
         f" turnSaidHtml({{ way: 'rounds', errors: {json.dumps(stuck['errors'])} }}),"
         " turnsHtml()].join(' ')")
    # Q995c: a piece's turns on its card, in rounds and out; finding a rule by a word, and
    # what a search says beside the menu; a save under a template's name.
    from chess_variant_sandbox.spec import describe
    from chess_variant_sandbox.web import cards

    templates = designing.templates()
    kit = (f"DESIGN_KIT = {json.dumps({'cards': cards.card_model([]), 'templates': templates, 'ruleWords': describe.rule_words(), 'ruleChoices': describe.rule_choices()})};")
    rounds = json.loads(json.dumps(turned["draft"]))
    rounds["pieces"][1]["speed"] = 1
    view("how a piece takes its turns", kit + f"DRAFT = {json.dumps(rounds)};",
         "[pieceEditor(DRAFT.pieces[1], 1), paletteHtml(), pieceTrayHtml()].join(' ')")
    view("how a piece takes its turns, not in rounds", kit + f"DRAFT = {json.dumps(chess)};",
         "pieceEditor(DRAFT.pieces[1], 1)")
    view("finding a rule by a word", kit + f"DRAFT = {json.dumps(chess)};"
         f" RULE_REFUSED = {json.dumps(refused['refused'])};",
         "[ruleMenuHtml(), ruleChosenHtml('drops'), ruleChosenHtml('turn')]"
         ".concat(['rounds', 'atomic', 'castling', 'zzzz', 'piece']"
         ".map(word => (RULE_FIND = word, ruleFoundHtml() + ruleOptionsHtml())))"
         ".join(' ')")
    view("a save under a template's name", kit + f"DRAFT = {json.dumps(templates[0]['spec'])};",
         "nameAskHtml('template') + nameAskHtml('game')")
    view("trying small changes", "TRY_OPEN = false;", "tryHtml()")
    view("measuring in the browser", "DRAFT_CHECK = { hash: 'a' };",
         "[hereOwnHtml({ result: { index: 0.5, games: 256, weights: 'v11' } }, null),"
         " hereCheckHtml({ state: 'running' }), hereCheckHtml({ state: 'done' }),"
         " hereTimesHtml({ seconds: 125, workers: 4, times: [], heaps: [25165824] })]"
         ".join(' ')")
    return found


def _game_page_views() -> list[tuple[str, str]]:
    """What a game's page draws on the site, view by view (W3, rows G1-G43)."""
    game = json.dumps(_game_for_views())
    setup = f"ACCOUNTS = true; var GAME = {game}; ROSTER = GAME.opponents;"
    found: list[tuple[str, str]] = []

    def view(where: str, call: str, more: str = "") -> None:
        found.append((where, site_words.visible(str(_run_page(setup + more, call)))))

    # L2 (`docs/archive/plan-leaderboard.md`): the leaderboard in each of its states, where the
    # site drew the strength table before. A row the picker no longer offers carries the
    # server's name for it (`opponents.table_name`, G17), never its agent string.
    from chess_variant_sandbox.web import opponents

    gone = json.dumps(opponents.table_name("minimax:nodes=12800"))
    row = ("{ name: %s, computer: %s, rating: %d, settling: %s, games: %d, top: %s,"
           " gone: %s, you: %s }")
    rows = ", ".join((
        row % ('"hardest"', "true", 2302, "false", 4, "true", "false", "false"),
        row % ('"ada"', "false", 1712, "true", 3, "false", "false", "true"),
        row % (gone, "true", 1650, "false", 1, "false", "true", "false"),
        row % ('"random"', "true", 820, "false", 2, "false", "false", "false"),
    ))
    for where, board in (
        ("the leaderboard", f"{{ rated: true, people: 1, library: true, rows: [{rows}],"
         " waiting: [], unrated: ['tree search'] }"),
        ("the leaderboard, nobody yet", "{ rated: true, people: 0, library: true, rows:"
         " [{ name: 'hard', computer: true, rating: 1946, settling: false, games: 0,"
         " top: true, gone: false, you: false }], waiting: [], unrated: [] }"),
        ("the leaderboard, not rated yet", "{ rated: false, people: 0, library: true,"
         " rows: [], waiting: ['random', 'greedy', 'easy'], unrated: [] }"),
        ("the leaderboard, not rated yet", "{ rated: false, people: 0, library: false,"
         " rows: [], waiting: ['random'], unrated: [] }"),
    ):
        view(where, "leaderboardHtml(GAME)",
             f"GAME.standing = {{ leaderboard: {board} }};")
    view("the measuring panel", "measuringHtml(GAME)")
    view("the provisional score", "measuringHtml(GAME) + provisionalHtml(GAME)",
         "GAME.staleness = null; GAME.provisional = { yours: true, state: 'spot-checked',"
         " index: 0.5, seeds: [0], weights: 'v11', sample: 2 };")
    view("where it stands", "whereHtml(GAME)")
    # Its parent links are wired on a timer that reads the drawn view, which a stub has not.
    view("where it came from", "provenanceHtml(GAME.provenance, GAME)",
         "const __later = global.setTimeout; global.setTimeout = (f, ...rest) =>"
         " (String(f).includes('data-hash') ? 0 : __later(f, ...rest));")
    view("the games near it", "linksHtml(GAME)")
    view("a royal piece", "piecesHtml({ name: 'king', id: 'K', royal: true, moves: [] })")
    view("the report link", "reportBody({ reportReasons: [] })"
         " + reportBody({ yourReport: { label: 'spam' } })")
    standings = (
        "{ rated: false, signedIn: true }",
        "{ rated: true, signedIn: false, unbeaten: true, top: ['minimax:nodes=6400'] }",
        "{ rated: true, signedIn: true, unbeaten: true, top: ['minimax:nodes=6400'],"
        " topRating: 1946, yourRecord: { state: 'played', won: 1, drawn: 0, lost: 2,"
        " played: 3, rung: { agent: 'minimax:nodes=1600', rating: 1700, seat: 0 } },"
        " highest: { handles: ['ada'], rating: 1712 } }",
        "{ rated: true, signedIn: true, unbeaten: false, top: ['minimax:nodes=6400',"
        " 'minimax:nodes=200000'], topRating: 1946, firstClear: { handle: 'ada', seat: 1 },"
        " clearedBy: [{ handle: 'ada', seat: 1 }, { handle: 'bo', seat: 0 }] }",
    )
    for standing in standings:
        view("Unbeaten and the mode", "standingHtml(GAME) + modePicker(GAME)",
             f"GAME.standing = {standing};")
    view("the start button", "startWords(['human', 'minimax:nodes=1600'], 'sandbox')")
    view("a game left running", "carryOnLine({ opponent: 'minimax:nodes=1600', seat: 0,"
                                " plies: 12, moved: null })")
    for score in (False, True):
        found.append(("the score panel", site_words.visible(_score_panel(True, score))))
    return [(where, text) for where, text in found if text]


class TestTheLibraryAndTheMapSpeakToAVisitor:
    """W2 (`docs/archive/plan-understanding.md` §3.3, B1-B23 and M1-M13): the library's cards,
    controls and legend, and the map's panels, in the site's words."""

    @staticmethod
    def _views() -> dict[str, str]:
        return dict(_library_and_map_views())

    def test_the_cards_say_the_audits_replacements(self) -> None:
        views = self._views()
        cards = views["the library's cards"]
        for said in ("12 games played by people", "beaten by 3 people", "not played here yet",
                     "2 versions on from its parent",
                     "Nobody has beaten its strongest computer player yet",
                     "the average of four runs, under scoring rules version 11",
                     "The computer players or the scoring changed since it was measured;"
                     " it will be measured again.",
                     "Too few of its four runs could be scored, so it has no score rather"
                     " than a low one.",
                     "spot-checked by the site", "came out exactly the same",
                     "A Baseline game", "Nobody has finished a rated game on this yet",
                     "A practice game leaves no mark", "ruled out"):
            assert said in cards, said
        for gone in ("minimax", "human game", "clears", "Fairy-Stockfish", "server",
                     "cannot be won", "replay 1 of 4", "A mean needs"):
            assert gone not in cards, gone

    def test_the_controls_and_legend_say_them_too(self) -> None:
        views = self._views()
        controls, legend = views["the library's controls"], views["the library's legend"]
        for said in ("how many versions it has been through", "games that never finish",
                     "does thinking longer win", "works as a game", "ruled out",
                     "unrated (no computer players yet)", "whether captured pieces come back"):
            assert said in controls, said
        for gone in ("the gate says", "Fairy-Stockfish", "the rules translate",
                     "skill gradient", "stall rate"):
            assert gone not in controls, gone
        assert "Tap or point at one for what it means" in legend
        assert "Where it stands" in legend and "Baseline: a well-known game" in legend

    def test_the_local_application_keeps_its_own_cards(self) -> None:
        local = _run_page(_LIBRARY_SETUP.replace("ACCOUNTS = true;", ""),
                          f"{_LIBRARY_CARDS}.map(cardHtml).join(' ')")
        assert "translates to Fairy-Stockfish" in local
        assert "mean of four runs (scoring rules v11)" in local
        assert "replay 1 of 4 runs" in local and "A mean needs three" in local

    def test_the_map_says_alike_communities_and_small_changes(self) -> None:
        """M4 of `docs/plan-map.md`: the slider and the rule-step lines left the map, and
        so did their words; a line now says how alike two games are and what they share."""
        views = self._views()
        said = " ".join(text for where, text in views.items() if where.startswith("the map"))
        for words in ("A line joins two games when each is among the other's 5 nearest",
                      "A community i is a group of games more like each other than like"
                      " the rest", "unusual games are in no community",
                      "Two of the games are the same rules written two ways, shown as one"
                      " dot.", "also listed as a twin: the same rules, written two ways",
                      "small changes to the rules", "The computer invented it, after 3"
                      " rounds of changes.", "What changed most, of the eleven things"
                      " measured", "Games like this one, most alike first",
                      "where 1 is the same game", "Both have:", "Drawn in 2 places",
                      "What marks them out:", "4 rules apart, too many to list"):
            assert words in said, words
        for gone in ("similarity index", "the cut", "this cut", "design step", "design run",
                     "generation", "fifty-five", "compile", "two files", "slider",
                     "faint line", "rules away, fewest first"):
            assert gone not in said, gone

    def test_the_maps_controls_and_its_size_menu(self) -> None:
        from page_text import EXPLORER

        from chess_variant_sandbox.web import graphing

        assert "arrows to the" in EXPLORER and "games made from each" in EXPLORER
        assert "every game back where it" in EXPLORER and "server put it" not in EXPLORER
        assert "tap one for the games like it" in EXPLORER
        for gone in ("join games this alike or closer", "lines between games one to three",
                     "move the slider"):
            assert gone not in EXPLORER, gone
        assert "'Not played here yet.'" in EXPLORER
        assert "one scored under older scoring rules" in inspect.getsource(graphing)


class TestTheDesignerSpeaksToAVisitor:
    """W5a (`docs/archive/plan-understanding.md` §3.3, D1-D39): the designer's own words, on the
    site; the local application keeps the maintainer's where the row says so."""

    @staticmethod
    def _views() -> dict[str, str]:
        return dict(_designer_views())

    def test_the_ladder_says_when_each_step_happens_and_who_decides(self) -> None:
        views = self._views()
        ladder = views["the designer's ladder (measured)"]
        for said in ("From a draft to a score", "a computer player can win it",
                     "two small computer players play eight games", "saved as your draft",
                     "yes, as your draft", "yes (0.12 seconds)", "Takes milliseconds;"
                     " happens as you edit", "when you press save", "not on this site yet",
                     "decided by the person who runs the site", "scored 0.62 , the average"
                     " of four runs (scoring rules version 11); ruled out"):
            assert said in ladder, said
        for gone in ("candidate", "maintainer", "Costs", "runs inline", "weights",
                     "not viable", "s here"):
            assert gone not in ladder, gone
        assert "measured, but out of date: no score is shown until it is measured again" in (
            views["the designer's ladder (stale)"])
        assert "put forward; the person who runs the site decides" in (
            views["the designer's ladder (nominated)"])
        assert "yes: these rules are already a library game, chess" in (
            views["the designer's ladder (a library game)"])
        signed_out = _run_page("ACCOUNTS = true; VIEWER = null;"
                               + TestTheSitesWordsAreAVisitors()._standing(saved=False),
                               f"ladderHtml({self._check()})")
        assert "sign in to save it" in _words(signed_out)
        held = views["the designer's ladder (held)"]
        assert "Measuring is paused while the site is updated." in held
        assert "uv sync" not in held

    def test_which_game_this_is_says_a_score_and_no_version(self) -> None:
        views = self._views()
        fork = views["which game this is (a fork)"]
        assert "score 0.61 , the average of four runs" in fork and "players" not in fork
        assert ("This game has no score yet: a score from the parent's rules says nothing"
                " about these. No score is not a score of zero.") in fork
        assert "A score belongs to the exact rules it was measured on" in fork
        assert "version" not in fork and "You sent" not in fork
        assert "This game has no score yet. No score is not a score of zero." in (
            views["which game this is (no parent)"])
        stale = views["which game this is (an out-of-date parent)"]
        assert "measured, but out of date" in stale and "needs a" not in stale

    def test_the_rest_of_the_designer(self) -> None:
        views = self._views()
        assert "ran all three of its steps, and these rules passed every one" in (
            views["the save's answer"])
        assert "engine" not in views["the move marks"]
        taught = views["a taught move"]
        assert "working it out" in taught and "the editor can describe moves" in taught
        assert "Each is a move this editor can already write." in taught
        assert "the last time the draft was a complete game" in views["the rule cards' refusals"]
        assert "Small changes the site knows how to make" in views["trying small changes"]
        browser = views["measuring in the browser"]
        for said in ("measured on your computer", "an older version of this page",
                     "every game replayed to check it was legal, and scored again by the site",
                     "the site has changed how it scores games since", "on four cores"):
            assert said in browser, said
        assert "MiB" not in browser and "worker" not in browser

    def test_the_page_text_and_the_local_application(self) -> None:
        from page_text import DESIGN

        for said in ("A game here is a board, some pieces, where they start and how it ends.",
                     "Each rule has a card, and the sentence under each control says what",
                     # The box's name is the kit's since Q830b (`reference.PASTE_CONTROL`),
                     # and a text with a mistake is refused in the server's words.
                     "(JSON), to copy out or paste in",
                     "Paste a game's text here to edit it.",
                     "was not loaded, and the game you were editing is as it was:",
                     "played by a computer", "An automatic check by an AI model",
                     "last, or a row number (your own back row is 0)", "> none</label>",
                     "the editor doesn't know ("):
            assert said in DESIGN, said
        local = _words(_run_page(self._local(), f"ladderHtml({self._check()})"))
        assert "Costs milliseconds; runs inline" in local and "kept as a candidate" in local

    @staticmethod
    def _check() -> str:
        return ("{ hash: 'c5e5b57cab2b', valid: true, playable: true, smokeSeconds: 0.12,"
                " schemaVersion: 2 }")

    @staticmethod
    def _local() -> str:
        return TestTheSitesWordsAreAVisitors()._standing()

    def test_the_servers_sentences(self) -> None:
        from chess_variant_sandbox.site import drafts, provisional

        assert "only written differently" in drafts.Twin("h", "chess", "in the Library").said
        assert provisional.INCOMPLETE == "The score arrived incomplete; measure it again."
        assert "changed how it scores games" in provisional.CHANGED
        with pytest.raises(designing.DesignError, match="That draft could not be read."):
            candidacy.screen_draft([])
        with pytest.raises(designing.DesignError, match="aren't a complete game yet"):
            candidacy.screen_draft({"name": "x"})


class TestTheShellAndItsInfosSpeakToAVisitor:
    """W1 (`docs/archive/plan-understanding.md` §3.3, A1-A22): every (i) and the score panel in the
    site's words where the page is the site, the local application's sentence kept where
    the maintainer still wants it, and the site's vocabulary defined once."""

    def test_the_site_reads_its_own_entry_and_the_checkout_keeps_its(self) -> None:
        keys = _run_page("", "Object.keys(SITE_GLOSS)")
        local = _run_page("", "Object.keys(SITE_GLOSS).map(key => glossEntry(key))")
        site = _run_page("ACCOUNTS = true;",
                         "Object.keys(SITE_GLOSS).map(key => glossEntry(key))")
        shared = _run_page("", "Object.keys(SITE_GLOSS).map(key => GLOSS[key])")
        assert {"index", "stale", "roster", "elo", "siteScale", "kind", "differs",
                "fairy", "moveRule"} <= set(keys)
        assert local == shared, "the local application reads the glossary as written"
        for key, mine, theirs in zip(keys, site, local, strict=True):
            assert mine[1] != theirs[1], key
        stale = dict(zip(keys, site, strict=True))["stale"][1]
        assert "checkout" not in stale and "It will be measured again." in stale
        assert "checkout" in dict(zip(keys, local, strict=True))["stale"][1]

    def test_the_score_says_one_paragraph_and_folds_two_behind_more(self) -> None:
        site = _run_page("ACCOUNTS = true;", "glossBubble('index', '')")
        local = _run_page("", "glossBubble('index', '')")
        assert "<details><summary>more</summary>" in site
        first, folded = site.split("<details>")
        assert "how interesting a game looks when the computer plays it" in first
        assert "Every score says what it rests on" in folded
        assert "best estimate available today, not a verdict" in folded
        assert "<details>" not in local and "four more seeds" in local
        # The title, a hover, carries what is shown and not what is folded.
        title = _run_page("ACCOUNTS = true;", "infoBits('index').title")
        assert "Every score says" not in title

    def test_a_badge_says_where_every_badge_is_listed_with_a_link(self) -> None:
        said = _run_page("ACCOUNTS = true;", "glossBubble('badge', '')")
        assert '<a href="/badges">the badges page</a>' in said
        assert "/badges," not in _visible(said)

    def test_rated_here_is_one_scale_on_the_site(self) -> None:
        site = _run_page("ACCOUNTS = true;", "[glossEntry('siteScale'), GLOSS.ratedHere]")
        assert site[0] == site[1] and site[0][0] == "Rated here"
        assert "starts at 1500" in site[0][1]

    def test_the_vocabulary_is_defined_once_for_the_tasks_after(self) -> None:
        """§3.2's kept words, each an (i) a later word task points at."""
        said = _run_page("", "Object.fromEntries(Object.entries(GLOSS)"
                             ".map(([key, entry]) => [key, entry[1]]))")
        for key in ("run", "spotChecked", "settling", "baseline", "practice", "ratedGame",
                    "turns", "strongest", "nominate", "community", "playability",
                    "ratedHere", "ruleMass", "royal"):
            assert said.get(key), key
        assert "such as chess" in said["baseline"]
        assert "one turn" in said["turns"]

    def test_a_refusing_run_is_named_by_its_place_on_the_site(self) -> None:
        call = ("splitSentences({ split: 'Three of four.' }, [{ seed: 0, viable: true },"
                " { seed: 1000, viable: true }, { seed: 2000, viable: false,"
                " violations: ['why'] }, { seed: 3000, viable: true }])")
        assert _run_page("ACCOUNTS = true;", call) == ["Three of four.", "run 3 of 4: why"]
        assert _run_page("", call) == ["Three of four.", "run 2000: why"]

    def test_the_band_is_clearly_above_and_clearly_below(self) -> None:
        said = _visible(_run_page("", "bandHtml({ of: 70, clears: 3, clearedBy: 12,"
                                      " close: 55 })"))
        assert ("Against the other 70 scored games: clearly above 3 i , clearly below 12 i ,"
                " too close to call with 55 i .") in said

    def test_a_withheld_route_is_refused_in_a_visitors_words(self) -> None:
        said = _run_page("ACCOUNTS = true;",
                         "api('/api/promote', {}).then(() => 'sent', error => error.message)")
        assert said == "That isn't available on this site."

    def test_a_move_rules_info_carries_no_raw_rule_on_the_site(self) -> None:
        piece = json.dumps({"name": "p", "id": "p", "moves": ["moves one step"],
                            "descriptors": ["step (0,1) range 1"]})
        local = _run_page("", f"piecesHtml({piece})")
        site = _run_page("ACCOUNTS = true;", f"piecesHtml({piece})")
        assert "step (0,1) range 1" in local
        assert "step (0,1) range 1" not in site and 'data-gloss="moveRule"' in site

    def test_the_score_panel_reads_the_sites_words_on_the_site(self) -> None:
        site, local = _score_panel(site=True), _score_panel(site=False)
        assert "How it is measured." in local and "How it is measured." not in site
        assert "command line" in local and "command line" not in site
        assert "how the score is worked out" in site
        # The parts' own names, and (W3, G29) the tests' names under them.
        assert "<span>does thinking longer win</span>" in site
        assert "<span>skill gradient</span>" not in site
        assert "<span>skill gradient</span>" in local
        assert "rating points each time thinking time doubles" in site
        assert "Each version of the scoring rules is its own" in site
        assert "weighted geometric mean" in local and "weighted geometric mean" not in site
        assert "run 3 of 4: the computer players didn't get clearly stronger" in (
            site_words.visible(site))
        assert "run 2000: this roster cannot read this ruleset" in local
        assert "clearly above" in site

    def test_a_games_page_says_each_row_as_the_plan_replaces_it(self) -> None:
        """W3 (`docs/archive/plan-understanding.md` §3.3, G rows): the views of a game's page as
        the site draws them, one sentence per row."""
        views: dict[str, str] = {}
        for where, text in _game_page_views():
            views[where] = views.get(where, "") + " " + text
        rows = {
            "a royal piece": ["the piece the game is about losing, like a king"],  # G2
            "Unbeaten and the mode": [
                "This is a practice game",  # G5
                "nothing counts towards a rating or Unbeaten",  # G6
                "Its computer players have no ratings yet, so there is no rated play and no"
                " Unbeaten.",  # G10
                "Its strongest computer player is rated 1946 here",  # G11
                "rated 1700 here at the time",  # G12
            ],
            "a game left running": ["12 turns played"],  # G13
            # G14 and G16's table is the leaderboard since L2: the scale said once, and
            # the scoring run's own table no longer drawn on the site at all.
            "the leaderboard": [
                "Everyone who has finished a rated game here, and every computer player,"
                " strongest first, on the scale people are rated on.",  # G14, L2
                "A number with a ? is still settling",  # L2, §2.1
                "the 12,800-position player (no longer offered)",  # G17, L2
                "One computer player, own piece values, is unavailable on this game for"
                " now.",  # G19
                "tree search has no rating on this game yet, so it is not on it.",  # L2
            ],
            "the leaderboard, nobody yet": [
                "Nobody has played a rated game here yet. Your first one puts you on this"
                " table, starting at 1500.",  # L2, §2.5
            ],
            "the leaderboard, not rated yet": [
                "This game's computer players have not been rated yet, so every game here"
                " is practice for now. They are rated on our own machine; this usually takes"
                " a day.",  # L2, §2.5
            ],
            "the measuring panel": ["This game's score is out of date"],  # G20
            "the provisional score": [
                "The site has not measured this one itself yet. The score below was measured"
                " on your computer.",  # G21
                "scoring rules version 11",  # G22
            ],
            "where it stands": [
                "Passes the playability check",  # G32
                "Built on by bo",  # G33
                "A Baseline game",  # G41
                "put it forward for this month's members' vote",  # G43
            ],
            "the report link": ["The person who runs the site reads every report."],  # G35
            "where it came from": [
                "invented by the computer here", "after 3 rounds of changes",  # G36
                "it was asked to try", "an earlier game no longer listed",  # G36, G38
            ],
            "the games near it": [
                "what differs: 1 rule, 28 starting squares, 1 limit or price"],  # G40
        }
        for where, sentences in rows.items():
            for sentence in sentences:
                assert sentence in views[where], (where, sentence)
        assert "stale (re-score)" not in views["the measuring panel"]  # G20
        assert "run-2026-09-01" not in views["where it came from"]  # G36
        assert "0a1b2c3d4e5f" not in views["where it came from"]  # G38
        assert "+" not in views["the leaderboard"]  # G14
        # G42: the level's sentence, the server's.
        from chess_variant_sandbox.site import stages

        assert stages.SAYS[stages.LIBRARY] == (
            "In the library: scored by the site and listed for everyone.")
        assert stages.shown(stages.BASELINE)["levelGloss"] == "baseline"
        assert stages.shown(stages.LIBRARY)["levelGloss"] == ""

    def test_the_rules_count_turns_on_both_servers(self) -> None:
        """G4: the rules' own writer serves both, in one voice."""
        from chess_variant_sandbox.spec import load_spec
        from chess_variant_sandbox.spec.describe import describe_rules, describe_spec
        from chess_variant_sandbox.web import library

        spec = load_spec(library.RULESET_DIR / "chess.json")
        said = " ".join(describe_rules(spec)) + " " + describe_spec(spec)
        assert "turns without a capture or other lasting change" in said
        assert "turns in all" in said and "plies" not in said

    def test_the_local_application_keeps_the_maintainers_sentences(self) -> None:
        """What the site rewords stays as it was on a checkout, where its reader runs it."""
        game = json.dumps(_game_for_views())
        local = _visible(_run_page(f"var GAME = {game}; ROSTER = GAME.opponents;",
                                   "strengthHtml(GAME) + measuringHtml(GAME)"))
        assert "in the tournament" in local and "standard opponents" in local
        assert "stale (re-score): the metrics fingerprint moved" in _run_page(
            f"var GAME = {game};", "measuringHtml(GAME)")

    def test_the_score_panels_tests_runs_and_lean_in_a_visitors_words(self) -> None:
        """W3 (G23, G26-G31): what the score rests on, the side it leans to, the tests a
        game must pass and the scoring's own check, each as the site says it."""
        site, local = _score_panel(site=True), _score_panel(site=False)
        said, kept = site_words.visible(site), site_words.visible(local)
        # G23: two short sentences, each run's score behind *more*; the local wording kept.
        assert "The average of four runs i . The ± is how far chance alone could move it." \
            in said
        assert "sampling error" not in said and "sampling error of that mean" in kept
        assert "<details><summary>more</summary>Four runs of 256 games each" in site
        assert "could not be scored, so the average is over the other three" in said
        # G26, G31: the lean in sides and computer players, counted.
        assert "One side wins more often, and it is ranked anyway." in said
        assert "the two strongest computer players won 41 of their 61 decided games from" \
               " the first side" in said
        assert "It leans toward one seat" in kept and "rungs" in kept
        # G27: a family's name opens what it asks on a tap too.
        assert 'data-gloss="family"' in site and "Tap or hover a name for what it asks" in said
        # G28: the roster check is not drawn on the site.
        assert "ρ" not in said and "&rho;" not in site and "Roster dependence" not in said
        assert "&rho;" in local and "Roster dependence" in kept
        # G29: the tests, with *worse half* defined once.
        assert "Tests a game must pass" in said and "What rules a game out entirely" in kept
        assert said.count("whichever half of the computer players, weaker or stronger") == 2
        for name in ("thinking beats random moves", "won by one side", "games that never"
                     " finish, worse half", "does thinking longer win must be at least 8"):
            assert name in said, name
        assert "search beats chance" in kept and "seat-decided chance" in kept
        assert "search beats chance" not in said and "seat-decided" not in said
        # The split, in *pass it* and *rule it out*.
        assert "Three of its four runs pass it and one rules it out, so it stays in." in said
        # A score the tests ruled out lists each refusal in the same words, by its place.
        out = site_words.visible(_score_panel(site=True, ruled_out=True))
        assert "run 2 of 4: quick kills, worse half came to 0.538, over the 0.5" in out
        assert "quick_kill_rate" not in out and "run 1000" not in out


def served_by_the_page() -> list[tuple[str, str]]:
    """`(where, words)` for the prose the page's own script draws where it is the site.

    The script is run with `ACCOUNTS` true: every (i)'s title, text and folded paragraphs
    as the site reads them (`glossEntry`, so `SITE_GLOSS` where it has the key), the
    designer's right-hand panel for a draft and for a baseline game, the rungs as the
    site words them, the Fairy-Stockfish note for an absent and a differing engine, (W1) a
    game's score panel over the current scoring rules' own guide, with its band, and (W3)
    the rest of a game's page: the strength table, Unbeaten, the mode, the measuring and
    provisional lines, where it stands and came from, its neighbours and its marks, and
    (W6) the library card's tags, which W2 rewrites.
    A view the audit has not reached is not read here yet; a word task that rewrites one
    adds it, beside the phrases it moves to `site_words.ENFORCED`.
    """
    found: list[tuple[str, str]] = []
    glossary = _run_page("ACCOUNTS = true;",
                         "[...new Set([...Object.keys(GLOSS), ...Object.keys(SITE_GLOSS)])]"
                         ".map(key => [key, glossEntry(key)])")
    for key, entry in glossary:
        if key in _NOT_DRAWN_ON_THE_SITE:
            continue
        said = [part for part in entry if isinstance(part, str)]
        said += [text for part in entry if isinstance(part, dict)
                 for text in part.get("more", [])]
        found.extend((f"GLOSS.{key}", _words(part)) for part in said)
    # W3 (G1-G43): a game's page, view by view, with the score panel among them.
    found.extend(_game_page_views())
    # W2 (B1-B23, M1-M13): the library and the map.
    found.extend(_library_and_map_views())
    # W5a (D1-D39): the designer's own words.
    found.extend(_designer_views())
    for rung in candidacy.rungs(300.0):
        site = {**rung, **rung["site"]}
        found.extend((f"rung {rung['key']}", _words(site[part]))
                     for part in ("name", "what", "cost", "runs"))
    words = TestTheSitesWordsAreAVisitors()
    for stage in ("candidate", "committed"):
        found.append((f"the designer's panel ({stage})", _visible(words._panels(
            "ACCOUNTS = true;" + words._standing(stage=stage)))))
    for engine in ({"engine": None}, {"engine": "fsf", "ok": False, "reasons": ["x"]}):
        game = json.dumps({"fairyCheck": engine, "fairyTranslates": True})
        found.append(("the Fairy-Stockfish note",
                      _visible(_run_page("ACCOUNTS = true;", f"fairyNote({game}, [])"))))
    # Q579 (F7; P1-P6, P14-P19): the opponent picker as the site draws it, to a visitor
    # and to somebody rated on the game, with what goes under it; a searching player's
    # move in the move list; a game's end; the controls of a board without edges.
    from chess_variant_sandbox.web import opponents

    roster = json.dumps([entry.as_dict() for entry in opponents.ladder()])
    ratings = {agent: 1500.0 + 100 * index for index, (agent, _) in enumerate(opponents.LADDER)}
    for standing in ({"siteRatings": ratings},
                     {"siteRatings": ratings, "yourGaps": {a: -240.0 for a in ratings}}):
        game = json.dumps({"standing": standing})
        found.append(("the opponent picker", site_words.visible(_run_page(
            f"ACCOUNTS = true; ROSTER = {roster};",
            f"seatSelect('seat1', ROSTER, 'human', null, {game})"
            " + ROSTER.map(o => planFor(o.agent)).join(' ')",
        ))))
    found.append(("the move list", site_words.visible(_run_page(
        "ACCOUNTS = true;", "lookedHtml({ depth: 3 }) + lookedHtml({ depth: 0 })"))))
    # W6 (Q613): the library card's tags, every state, hovers included. W2 rewrites them
    # (B15-B18, B22); they are read here so the phrases they still hold stay listed as
    # `NOT_YET` honestly rather than enforced over a view nobody rendered.
    found.append(("the library card's tags", site_words.visible(_run_page(
        "ACCOUNTS = true;",
        "[unbeatenTag({ rated: false }),"
        " unbeatenTag({ unbeaten: true, top: ['minimax:nodes=6400'] }),"
        " unbeatenTag({ unbeaten: false, firstClear: { handle: 'ada' }, clears: 3 }),"
        " untriedTag({ untried: true }), unplayedTag({ humanGames: 0 })].join(' ')"))))
    found.append(("a game's end", _words(_run_page(
        "ACCOUNTS = true; SESSION = { seats: ['human', 'minimax:nodes=1600'],"
        " ruleset: { sides: ['white', 'black'] } };",
        "['extinction', 'bare_royal', 'something_new']"
        ".map(reason => verdictText({ winner: 1, reason })).join(' ')"))))
    # Q993b: a game's proven positions on their own page, signed in and as a visitor, and
    # the game page's panel that leads there, each over the server's own words.
    found.extend(_proven_views())
    return [(where, text) for where, text in found if text]


def _proven_views() -> list[tuple[str, str]]:
    """The proven positions' own page and the game page's panel (Q993b), as the site draws
    them from what `site/proven.py` sends."""
    from test_site_proven import (
        BOTH,
        DIGEST,
        LOSES,
        ONE,
        ONE_ID,
        SPEC,
        _load,
        _record,
        _write,
    )

    from chess_variant_sandbox.site import proven

    found: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as folder:
        runs = Path(folder)
        _write(runs, _record([ONE, LOSES, BOTH]))
        record = _load(runs)
        view = proven.view(record, SPEC, ONE_ID)
        states = ((None, set()), (None, {ONE_ID}), ({}, set()),
                  ({ONE_ID: {proven.CONVERTED}}, set()))
        pages = [{**proven.page(record, marks, visit), "name": "proven test",
                  "sides": ["white", "black"]} for marks, visit in states]
        games = [{"hash": DIGEST, "standing": {"proven": proven.summary(record, marks)}}
                 for marks, _visit in states]
        # One run of the page for every state, each drawn as the view draws it.
        drawn = _run_page(
            f"ACCOUNTS = true; const PAGES = {json.dumps(pages)};"
            f" const GAMES = {json.dumps(games)};",
            f"PAGES.map(data => {{ PROVEN = {{ hash: '{DIGEST}', data, id: '{ONE_ID}',"
            " playing: null }; return provenViewHtml(data) + provenListHtml()"
            f" + provenFrameHtml({json.dumps(view)}); }})"
            ".concat(GAMES.map(game => provenHtml(game)))",
        )
        for index, html in enumerate(drawn):
            where = ("the proven positions' page" if index < len(states)
                     else "the game page's proven positions")
            found.append((where, site_words.visible(str(html))))
    return found


class TestAForkIsToldWhatItChanged:
    """Q491: the designer says, in the rules' own sentences, what a fork changed from the
    game it was forked from -- asked of the server with the parent's hash, drawn by the
    page from the answer."""

    @staticmethod
    def _fork(app: App) -> tuple[str, dict[str, Any], dict[str, Any]]:
        (digest,) = app.known()
        parent = app.get(f"/api/design/{digest}")["spec"]
        child = json.loads(json.dumps(parent))
        child["name"] = "a fork"
        child["pieces"][0]["moves"][0]["range"] = 2
        child["rules"]["max_plies"] = 100
        return digest, parent, child

    def test_the_check_names_the_changed_move_and_rule_and_nothing_else(self, world) -> None:
        app, _ = world
        digest, parent, child = self._fork(app)
        check = app.post("/api/design/check", {"spec": child, "parent": digest})
        assert check["valid"], check["errors"]
        changes = check["changes"]
        assert len(changes) == 2, changes
        assert changes[0].startswith(parent["pieces"][0]["name"]), changes
        assert "100" in changes[1] and "120" in changes[1], changes
        # Without a parent there is nothing to compare, which is not "nothing changed".
        assert app.post("/api/design/check", {"spec": child})["changes"] is None
        assert app.post("/api/design/check", {"spec": child, "parent": "nope"})[
            "changes"] is None

    def test_the_page_shows_them_under_the_parents_name(self, world) -> None:
        app, _ = world
        digest, _, child = self._fork(app)
        check = app.post("/api/design/check", {"spec": child, "parent": digest})
        drawn = _run_page(
            f"DRAFT = {json.dumps(child)}; DRAFT_FROM = {json.dumps(digest)};"
            " DRAFT_PARENT = { name: 'dull', hash: " + json.dumps(digest) + " };",
            f"[changesHtml({json.dumps(check)}),"
            f" changesHtml({{ changes: [] }}),"
            f" (DRAFT_FROM = '', changesHtml({json.dumps(check)}))]",
        )
        said = _words(drawn[0])
        assert "What you changed from dull" in said
        for line in check["changes"]:
            assert _words(line) in said, line
        assert "Nothing yet" in _words(drawn[1])
        assert drawn[2] == "", "a draft with no parent has nothing to be compared with"


class TestTheFirstStepSaysWhatItChecks:
    """Q681: the first step was *the rules make sense*, a judgement the check does not
    make. It is *the rules are a complete game*, the words the refusal uses (D28), and a
    draft it refuses is told why in the designer's words, never by a path."""

    def test_the_step_is_named_for_what_it_checks(self) -> None:
        first = candidacy.rungs(None)[0]
        assert first["key"] == "validate"
        assert first["name"] == "the rules are a complete game"
        site = {**first, **first["site"]}
        assert "make sense" not in site["name"] + site["what"]
        from page_text import PAGE

        assert "rules make sense" not in PAGE

    def test_a_refused_draft_is_told_in_the_designers_words(self) -> None:
        spec = load_spec(Path(__file__).resolve().parents[1] / "rulesets" / "chess.json")
        spec["setup"]["placements"][0]["squares"] = ["z9"]
        with pytest.raises(designing.DesignError) as refused:
            candidacy.screen_draft(spec)
        said = str(refused.value)
        assert said.startswith("These rules aren't a complete game yet: "), said
        assert "z9 isn't a square on this board" in said, said
        assert not site_words.hits(said, tuple(site_words.PHRASES)), said
