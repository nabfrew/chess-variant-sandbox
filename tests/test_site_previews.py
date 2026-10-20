"""Previews: new kinds of game on the site, playable and outside every scored surface (Q931).

The user, 2026-10-15: *"get the showcase games against computer up for testing on the site
as soon as possible. label them testing/previews if necessary"*. The games played in
rounds that Q918e archived said so in their own files, and Q942c's and Q935f's after them
(`provenance.preview`, outside the mechanics hash), and from that one sentence:

- **the listing sends them apart**: `/api/library`'s `games` is the library and nothing
  else, and the previews come in `previews`, at the level *Preview*, under their own tab;
- **they are in no scored surface**: no band, no trust level, no neighbour or link either
  way, no newest-games shelf, no measurement row, no seeded rating table, no leaderboard and
  no rated play;
- **they play**: a visitor starts a practice game against a round opponent through the web
  API and plays a round, and *watch it play itself* seats the round ladder's player.

Taking the sentence off is how a preview joins the library (Q934: the watering hole and
breather, then named Gardner in rounds against the chase; Q936t: riders and leapers; Q943f: the
watering hole by speed), and such a game is then on every scored surface.

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from retired_specs import retired_preview_path
from test_site_admin import DULL, Site, _running_site

from chess_variant_sandbox.site import correspondence, matches, stages
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.server import App

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"
RULESETS = Path(__file__).resolve().parent.parent / "rulesets"

ARCHIVED_PREVIEWS = {
    # Q935f's SP2 (`plan-spy.md` §5): a spy game starts by turns and ends in rounds once a
    # spy falls.
    "gardner-with-spies-resting.json": "8ce37482d0de",  # rewritten by Q983a, was 1d798bedbf66
    # Q943c's two Gardner speed showcases (`plan-speed.md` §6.1), played in rounds: not
    # viable as Q943d measured them (§6.5), so they stay previews.
    "gardner-in-rounds-with-a-swift-king.json": "d8bb879440ce",
    "gardner-in-rounds-with-swift-knights.json": "bf230a962d10",
}
"""The previews in `archive/`, by file and by the mechanics hash each was or will be measured
under: the round and spy games less the four that joined the library (`JOINED`), the six that
Q962b retired and the one Q962c retired (`docs/reference/previews-2026-10-18.md`; their files
are `tests/_retired_*.json`), and Q943c's two Gardner speed showcases."""

ROUND_PREVIEW = "0c7c0191c850"
ROUND_PREVIEW_FILE = "gardner-in-rounds-resting.json"
"""A preview played in rounds, for the tests that need one. `archive/` has none since Q962c
retired S1 (`docs/archive/plan-round-chase.md` §5) as a near copy of breather, so the test
checkout takes the retired file, which still carries its preview sentence: the site's handling
of a round preview is still code that has to hold. Since schema 29 (Q983a) the retired file is
the old rest's refusal fixture, so the checkout takes its rewrite (`"rest": 1` on every piece,
`tests/rest_equivalence.py::rewrite`), the same game under the hash `0c7c0191c850`."""

PREVIEWS = {**ARCHIVED_PREVIEWS, ROUND_PREVIEW_FILE: ROUND_PREVIEW}
"""The previews of the test checkout (`_root`)."""

JOINED_ROUNDS = {
    "watering-hole.json": "5c0b017473f5",
    # Named gardner in rounds against the chase until Q962; rewritten by Q983a, was 632c199e4589.
    "breather.json": "b8687024cfc6",
    # Q943c's T3 (`plan-speed.md` §6.1), viable as Q943d measured it (§6.5); joined by Q943f.
    "watering-hole-by-speed.json": "423a29997264",
}
"""Three round games that were previews and joined the library by taking the mark off (Q934,
`docs/archive/plan-library-audit.md`; Q943f, `docs/plan-speed.md`): Library games played in
rounds, on every scored surface."""

JOINED_TURNS = {"riders-and-leapers.json": "26636602fae0"}
"""A game played by turns that was a preview (Q936s: `"per_type": "own"`) and joined the same
way (Q936t): a Library game offered the alternating ladder (`opponents.RUNGS`)."""

JOINED = {**JOINED_ROUNDS, **JOINED_TURNS}

SPY_PREVIEWS = frozenset({"gardner-with-spies-resting.json"})
"""The preview that starts by turns (`library` `rounds` is false for a spy game, Q935c) and
offer the spy ladder, which has no round player (`opponents.SPY_RUNGS`)."""

PLAIN = "kyoto.json"
"""An archived game that is no preview, for the library's side of every comparison."""

ROUND_AGENT = "matrix:nodes=6400"


def _root(tmp_path: Path) -> Path:
    """A checkout of the dull Baseline game, five Library games (three of them played in
    rounds) and the four previews."""
    (tmp_path / "rulesets").mkdir()
    (tmp_path / "archive").mkdir()
    (tmp_path / "runs").mkdir()
    shutil.copy(RULESETS / "dull.json", tmp_path / "rulesets" / "dull.json")
    for name in (*PREVIEWS, *JOINED, PLAIN):
        (tmp_path / "archive" / name).write_text(
            json.dumps(_source(name), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return tmp_path


def _source(name: str) -> dict[str, Any]:
    """The test checkout's file `name`: from `archive/`, or the retired round preview's
    schema-29 rewrite."""
    if name == ROUND_PREVIEW_FILE:
        from rest_equivalence import rewrite

        return rewrite(load_spec(retired_preview_path("gardner-in-rounds-resting")))
    return load_spec(ARCHIVE / name)


# ----------------------------------------------------------------- the files themselves


class TestTheFiles:
    def test_the_four_say_they_are_previews_and_keep_their_hashes(self) -> None:
        for name, digest in PREVIEWS.items():
            spec = _source(name)
            assert spec_hash(spec) == digest, "the mark is outside the mechanics hash"
            assert validate_spec(spec) == []
            said = library.preview_sentence(spec)
            assert said is not None and "preview" in said.lower()
            assert "not yet trusted" in said

    def test_no_baseline_game_is_a_preview_and_only_the_round_games_are(self) -> None:
        for path in sorted(RULESETS.glob("*.json")):
            assert library.preview_sentence(load_spec(path)) is None, path.name
        marked = {
            path.name for path in ARCHIVE.glob("*.json")
            if library.preview_sentence(load_spec(path)) is not None
        }
        assert marked == set(ARCHIVED_PREVIEWS)

    def test_the_four_that_joined_say_nothing_and_keep_their_hashes(self) -> None:
        for name, digest in JOINED.items():
            spec = load_spec(ARCHIVE / name)
            assert spec_hash(spec) == digest, "taking the mark off moved no game"
            assert validate_spec(spec) == []
            assert library.preview_sentence(spec) is None
            assert ("simultaneous" in spec["rules"]) == (name in JOINED_ROUNDS)

    def test_an_empty_sentence_is_refused(self) -> None:
        spec = _source(ROUND_PREVIEW_FILE)
        spec["provenance"]["preview"] = "  "
        assert any("provenance.preview" in problem for problem in validate_spec(spec))
        spec["provenance"]["preview"] = 3
        assert any("provenance.preview" in problem for problem in validate_spec(spec))


# ------------------------------------------------------------- the local application


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> App:
    root = _root(tmp_path_factory.mktemp("previews"))
    return App(root / "rulesets", root / "runs", archive_dir=root / "archive")


class TestTheListing:
    def test_games_is_the_library_and_previews_holds_the_four(self, app: App) -> None:
        listing = app.get("/api/library")
        assert {card["hash"] for card in listing["games"]} == {
            spec_hash(DULL), spec_hash(load_spec(ARCHIVE / PLAIN)), *JOINED.values()
        }
        previews = {card["hash"]: card for card in listing["previews"]}
        assert set(previews) == set(PREVIEWS.values())
        for card in previews.values():
            assert card["source"] == library.PREVIEW
            assert card["level"] == stages.PREVIEW
            assert card["preview"] and card["band"] is None and card["trust"] is None
            # The quick check plays a game in rounds as it plays any game (Q932), so the
            # card carries its verdict: all four play to a finish under random play.
            assert card["playable"] is True and card["playableReasons"] == []
        assert all(card["preview"] is None for card in listing["games"])

    def test_no_preview_is_on_the_newest_games_shelf(self, app: App) -> None:
        listing = app.get("/api/library")
        assert not {card["hash"] for card in listing["newGames"]} & set(PREVIEWS.values())

    def test_a_previews_page_says_so_and_compares_it_with_nothing(self, app: App) -> None:
        for name, digest in PREVIEWS.items():
            game = app.get(f"/api/game/{digest}")
            assert game["preview"] and game["level"] == stages.PREVIEW
            assert game["levelSays"] == stages.SAYS[stages.PREVIEW]
            assert game["links"] == [] and game["band"] is None and game["trust"] is None
            agents = {entry["agent"] for entry in game["opponents"]}
            if name in SPY_PREVIEWS:
                assert game["rounds"] is False
                assert ROUND_AGENT not in agents and any(a.startswith("minimax:") for a in agents)
            else:
                assert game["rounds"] is True
                assert agents >= {ROUND_AGENT}

    def test_a_round_game_that_joined_is_a_library_page_played_in_rounds(self, app: App) -> None:
        for digest in JOINED_ROUNDS.values():
            game = app.get(f"/api/game/{digest}")
            assert game["preview"] is None and game["level"] != stages.PREVIEW
            assert game["rounds"] is True
            assert {entry["agent"] for entry in game["opponents"]} >= {ROUND_AGENT}

    def test_a_turn_game_that_joined_is_a_library_page_played_by_turns(self, app: App) -> None:
        for digest in JOINED_TURNS.values():
            game = app.get(f"/api/game/{digest}")
            assert game["preview"] is None and game["level"] != stages.PREVIEW
            assert game["rounds"] is False
            agents = {entry["agent"] for entry in game["opponents"]}
            assert ROUND_AGENT not in agents and any(a.startswith("minimax:") for a in agents)

    def test_no_library_page_names_a_preview(self, app: App) -> None:
        for digest in (
            spec_hash(DULL), spec_hash(load_spec(ARCHIVE / PLAIN)), *JOINED.values()
        ):
            links = app.get(f"/api/game/{digest}")["links"]
            assert not {link["hash"] for link in links} & set(PREVIEWS.values())

    def test_the_map_leaves_them_out(self, app: App) -> None:
        drawn = json.dumps(app.get("/api/graph"))
        assert not any(digest in drawn for digest in PREVIEWS.values())


class TestNoBand:
    def test_a_preview_with_a_current_score_is_in_no_band(
        self, app: App, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every game given the same current score: the library's six are banded against
        each other, and the four previews neither have a band nor move the others'."""
        score = {"weights_version": library.DEFAULT_WEIGHTS, "combined": 0.5,
                 "interval": {"ci_low": 0.4, "ci_high": 0.6, "deterministic_pairings": 1}}
        monkeypatch.setattr(library, "current_only", lambda _score: score)
        bands = library.bands(app.known().values())
        assert set(bands) == {
            spec_hash(DULL), spec_hash(load_spec(ARCHIVE / PLAIN)), *JOINED.values()
        }
        alone = library.bands(library.scored(app.known().values()))
        assert bands == alone, "the previews move no other game's band"


class TestThePage:
    def test_the_card_and_the_page_draw_the_badge_and_the_sentence(self) -> None:
        from test_web import _in_page

        said = "A preview: a new kind of game, still being tested."
        card = _in_page(
            "", "cardHtml(" + json.dumps({
                "hash": ROUND_PREVIEW, "name": "gardner in rounds resting", "description": "",
                "level": "Preview", "levelGloss": "", "preview": said, "board": "5x5",
                "pieceTypes": 6, "index": 0.503, "status": "current", "rated": False,
            }) + ")",
        )
        assert 'class="tag level preview">Preview</span>' in card
        assert f'<div class="previewNote">{said}</div>' in card
        assert "0.503" not in card, "a preview's card carries no score"


# --------------------------------------------------------------------------- the site


@pytest.fixture
def previews_site(tmp_path: Path) -> Iterator[Site]:
    root = _root(tmp_path)
    with _running_site(root) as running:
        report = loader.load(running.store, root)
        running.store.commit()
        assert not report.skipped, report.skipped
        yield running


def _origins(site: Site) -> dict[str, str]:
    with site.store.cursor() as cursor:
        cursor.execute("select hash, origin from games")
        return {str(row[0]): str(row[1]) for row in cursor.fetchall()}


class TestTheLoad:
    def test_the_four_load_as_previews_and_nothing_scores_them(
        self, tmp_path: Path
    ) -> None:
        root = _root(tmp_path)
        # A ladder on disk for one of them: a preview is seeded from none, whatever is there.
        ladder = {"standings": [{"name": "random", "rating": 1200.0},
                                {"name": ROUND_AGENT, "rating": 1500.0}]}
        (root / "runs" / "opponents").mkdir()
        (root / "runs" / "opponents" / f"{ROUND_PREVIEW}.json").write_text(
            json.dumps(ladder), encoding="utf-8"
        )
        with _running_site(root) as site:
            loader.load(site.store, root)
            site.store.commit()
            origins = _origins(site)
            assert {origins[digest] for digest in PREVIEWS.values()} == {loader.PREVIEW}
            assert origins[spec_hash(load_spec(ARCHIVE / PLAIN))] == loader.ARCHIVED
            assert {origins[digest] for digest in JOINED.values()} == {loader.ARCHIVED}
            rows = stages.read(site.store, list(PREVIEWS.values()))
            assert {row["level"] for row in rows.values()} == {stages.PREVIEW}
            listed = ", ".join(f"'{digest}'" for digest in PREVIEWS.values())
            assert site.count("measurements", f"hash in ({listed})") == 0
            assert site.count("ratings", f"game in ({listed})") == 0
            # A second load writes nothing, and taking the mark off makes it a Library game.
            assert not loader.load(site.store, root).written
            path = root / "archive" / ROUND_PREVIEW_FILE
            spec = load_spec(path)
            del spec["provenance"]["preview"]
            path.write_text(json.dumps(spec), encoding="utf-8")
            report = loader.load(site.store, root)
            site.store.commit()
            assert _origins(site)[ROUND_PREVIEW] == loader.ARCHIVED
            assert any("joined the Library from preview" in note for note in report.notes)
            assert site.count("level_moves") == 0


class TestTheSiteListing:
    def test_previews_come_apart_under_their_own_tab(self, previews_site: Site) -> None:
        listing = previews_site.browser().get("/api/library").payload
        assert not {card["hash"] for card in listing["games"]} & set(PREVIEWS.values())
        assert {card["hash"] for card in listing["previews"]} == set(PREVIEWS.values())
        assert {card["level"] for card in listing["previews"]} == {stages.PREVIEW}
        assert all(card["rated"] is False for card in listing["previews"])
        tabs = {tab["key"]: tab for tab in listing["site"]["tabs"]}
        assert tabs["previews"]["count"] == len(PREVIEWS)
        assert tabs["library"]["count"] == 2 + len(JOINED)
        for key in ("newGames", "launchSet"):
            assert not {card["hash"] for card in listing.get(key) or []} & set(
                PREVIEWS.values()
            )
        assert (listing.get("firstStep") or {}).get("hash") not in PREVIEWS.values()

    def test_a_preview_taken_off_the_site_is_not_listed(self, previews_site: Site) -> None:
        with previews_site.store.cursor() as cursor:
            cursor.execute(f"update games set revoked = now() where hash = '{ROUND_PREVIEW}'")
        previews_site.store.commit()
        listing = previews_site.browser().get("/api/library").payload
        assert ROUND_PREVIEW not in {card["hash"] for card in listing["previews"]}

    def test_the_game_page_has_no_leaderboard_and_no_links(
        self, previews_site: Site
    ) -> None:
        game = previews_site.browser().get(f"/api/game/{ROUND_PREVIEW}").payload
        assert game["preview"] and game["level"] == stages.PREVIEW
        assert game["levelSays"] == stages.SAYS[stages.PREVIEW]
        assert game["links"] == [] and game["band"] is None
        board = game["standing"]["leaderboard"]
        assert board["rated"] is False and board["rows"] == []
        assert game["standing"]["rated"] is False

    def test_no_preview_is_on_the_players_page(self, previews_site: Site) -> None:
        member = previews_site.signed_in("ada")
        answer = member.get("/api/players")
        assert answer.status == 200, answer.text
        assert not any(digest in answer.text for digest in PREVIEWS.values())


def _round(browser: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Play the first offered move of one round, the computer sealing first."""
    sid = snapshot["id"]
    if snapshot["awaitingAgent"]:
        snapshot = browser.post(f"/api/session/{sid}/agent", {}).payload
    move = snapshot["moves"][0]["move"]
    answer = browser.post(f"/api/session/{sid}/move", {"move": move})
    assert answer.status == 200, answer.text
    return dict(answer.payload)


ROUND_PREVIEWS = sorted(
    digest for name, digest in PREVIEWS.items() if name not in SPY_PREVIEWS
)
SPY_PREVIEW_HASHES = sorted(PREVIEWS[name] for name in SPY_PREVIEWS)
TURN_JOINED_HASHES = sorted(JOINED_TURNS.values())
SPY_AGENT = "minimax:nodes=400"
"""A rung of the spy ladder (`opponents.SPY_RUNGS`): a spy game offers no round player."""


class TestPlayingOne:
    @pytest.mark.parametrize("digest", ROUND_PREVIEWS)
    def test_a_visitor_plays_a_round_against_the_computer(
        self, previews_site: Site, digest: str
    ) -> None:
        visitor = previews_site.browser()
        started = visitor.post(
            "/api/session", {"hash": digest, "seats": ["human", ROUND_AGENT]}
        )
        assert started.status == 200, started.text
        after = _round(visitor, started.payload)
        assert after["round"]["number"] == 2, "one round was revealed"
        assert len(after["history"]) == 2

    @pytest.mark.parametrize("digest", TURN_JOINED_HASHES)
    def test_a_visitor_plays_a_turn_game_that_joined_against_the_ladder(
        self, previews_site: Site, digest: str
    ) -> None:
        """A game that alternates (Q936s, joined by Q936t) is played as any alternating game is."""
        visitor = previews_site.browser()
        started = visitor.post("/api/session", {"hash": digest, "seats": ["human", SPY_AGENT]})
        assert started.status == 200, started.text
        snapshot = dict(started.payload)
        assert not snapshot.get("round")
        sid = snapshot["id"]
        moved = visitor.post(f"/api/session/{sid}/move", {"move": snapshot["moves"][0]["move"]})
        assert moved.status == 200, moved.text
        replied = visitor.post(f"/api/session/{sid}/agent", {})
        assert replied.status == 200, replied.text
        assert len(replied.payload["history"]) == 2

    @pytest.mark.parametrize("digest", SPY_PREVIEW_HASHES)
    def test_a_visitor_plays_a_spy_preview_by_turns_against_the_spy_ladder(
        self, previews_site: Site, digest: str
    ) -> None:
        """A spy game starts by turns (Q935c), so its first move is a turn, not a round."""
        visitor = previews_site.browser()
        started = visitor.post("/api/session", {"hash": digest, "seats": ["human", SPY_AGENT]})
        assert started.status == 200, started.text
        snapshot = dict(started.payload)
        assert not snapshot.get("round"), "a spy game opens by turns"
        sid = snapshot["id"]
        moved = visitor.post(f"/api/session/{sid}/move", {"move": snapshot["moves"][0]["move"]})
        assert moved.status == 200, moved.text
        assert moved.payload["awaitingAgent"], "the spy ladder's player is to move"
        replied = visitor.post(f"/api/session/{sid}/agent", {})
        assert replied.status == 200, replied.text
        assert len(replied.payload["history"]) == 2

    def test_watch_it_play_itself_seats_the_round_ladders_players(
        self, previews_site: Site
    ) -> None:
        answer = previews_site.browser().post(
            "/api/session",
            {"hash": ROUND_PREVIEW, "seats": ["minimax:nodes=6400", "minimax:nodes=6400"]},
        )
        assert answer.status == 200, answer.text
        assert answer.payload["seats"] == ["matrix:nodes=102400", "matrix:nodes=102400"]

    def test_rated_play_is_refused_in_a_sentence(self, previews_site: Site) -> None:
        member = previews_site.signed_in("ada")
        answer = member.post(
            "/api/session",
            {"hash": ROUND_PREVIEW, "seats": ["human", ROUND_AGENT], "mode": "leaderboard"},
        )
        assert answer.status == 400
        assert "is a preview" in answer.text and "practice" in answer.text

    def test_rated_play_is_refused_even_with_a_rating_table(
        self, previews_site: Site
    ) -> None:
        """Belt and braces: a table seated by hand still makes no preview rated."""
        with previews_site.store.cursor() as cursor:
            cursor.execute(
                "insert into ratings (game, player_agent, rating, deviation, volatility)"
                " values (%s, %s, 1500, 100, 0.06)",
                (ROUND_PREVIEW, ROUND_AGENT),
            )
        previews_site.store.commit()
        with previews_site.connect() as connection:
            assert correspondence.mode_for(connection, ROUND_PREVIEW) == (
                correspondence.PRACTICE
            )
            with pytest.raises(matches.MatchError, match="is a preview"):
                matches.start(
                    connection, ROUND_PREVIEW, matches.LEADERBOARD,
                    ["human", ROUND_AGENT], user=previews_site.person("bo"),
                )

