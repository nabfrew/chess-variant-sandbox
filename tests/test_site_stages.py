"""One level word per game, decided in one place (Q503b, `docs/plan-game-pipeline.md` §5).

`site/stages.py` names where a game stands -- Draft, On trial, Shortlisted, Library,
Baseline, Resting, Retired -- from its row on the site (`stage`) and from where its file
is on a checkout (`local`). The acceptance is that every page naming a level reads that
one function, so the claims here are three.

**The words.** `stage` over each combination of the four columns it reads, a column a row
does not carry reading as absent; `local` over `corpus.find`'s and `library.discover`'s
sources.

**Every reader asks `stage`.** The site runs over a scratch store with a game at each
level, and `stages.stage` is replaced by a spy that marks its answer. The card, the tabs,
the designer's *Where this game stands*, the game page's *Where it stands* and the
profile's *Your games* then each carry the mark, so none of them decided a level itself.
The same for `local` on a checkout's card and game page.

**The pages draw what they are sent.** No page script holds a level word; the card's
pill, the game page's panel, the designer's track and the library's tabs are run in node
over a payload and draw its words.

The store tests need a real Postgres and skip with a reason without one;
`tests/test_site.py` has the recipe for a throwaway cluster.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_admin import (
    DULL,
    DULL_HASH,
    Site,
    _running_site,
    _saved,
    _screened,
    _template,
)
from test_web import _FIXTURE, _in_page

from chess_variant_sandbox.site import stages
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import library

REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVED = load_spec(REPOSITORY / "archive" / "merger.json")
ARCHIVED_HASH = spec_hash(ARCHIVED)


# ------------------------------------------------------------------------ the words


class TestTheWords:
    def test_a_drafts_next_step_says_its_link_works(self) -> None:
        """Q578, the first-visit study's B3: the profile sent a designer back to a Publish
        they could not use and never said the draft's link already worked. Publishing
        stays the next step (its own sentences are Q563's)."""
        said = stages.NEXT[stages.DRAFT]
        assert "link works now" in said and "copy link" in said
        assert "Publish it from the designer" in said
        assert "opens soon" not in said and "coming" not in said

    @pytest.mark.parametrize(
        ("row", "level"),
        [
            ({"origin": "archive", "visibility": "public"}, stages.LIBRARY),
            ({"origin": "corpus", "visibility": "public"}, stages.BASELINE),
            ({"origin": "corpus", "retired": "2026-09-01"}, stages.RETIRED),
            ({"origin": "archive", "retired": "2026-09-01"}, stages.RETIRED),
            ({"origin": "designed", "visibility": "unlisted"}, stages.DRAFT),
            ({"origin": "designed", "visibility": "private"}, stages.DRAFT),
            ({"origin": "designed", "visibility": "public"}, stages.ON_TRIAL),
            (
                {"origin": "designed", "visibility": "public", "nominated": "2026-09-02"},
                stages.SHORTLISTED,
            ),
            (
                {"origin": "designed", "visibility": "public", "rested": "2026-09-03"},
                stages.RESTING,
            ),
            # A nomination on a draft is not a shortlisting: only a listed game is put
            # forward, so an unlisted row is a Draft whatever else it carries.
            (
                {"origin": "designed", "visibility": "unlisted", "nominated": "2026-09-02"},
                stages.DRAFT,
            ),
        ],
    )
    def test_the_row_decides(self, row: dict[str, Any], level: str) -> None:
        assert stages.stage(row) == level

    def test_a_column_the_row_does_not_carry_reads_as_absent(self) -> None:
        """A caller holding fewer columns -- no `rested`, or none at all -- still gets a
        level rather than an error."""
        assert stages.stage({}) == stages.DRAFT
        assert stages.stage({"origin": "designed", "visibility": "public"}) == stages.ON_TRIAL
        assert stages.stage({"origin": "corpus"}) == stages.BASELINE

    @pytest.mark.parametrize(
        ("source", "level"),
        [
            ("committed", stages.BASELINE),
            ("archived", stages.LIBRARY),
            ("retired", stages.RETIRED),
            ("preview", stages.PREVIEW),
            ("candidate", stages.DRAFT),
            ("designed", stages.DRAFT),
            ("evolved", stages.DRAFT),
        ],
    )
    def test_a_checkout_reads_where_the_file_is(self, source: str, level: str) -> None:
        assert stages.local(source) == level

    def test_every_word_is_one_of_the_eight(self) -> None:
        # Q931 added *Preview*, beside the track: a level with a tab and no step.
        assert set(stages.LEVELS) == {
            "Draft", "On trial", "Shortlisted", "Library", "Baseline", "Preview", "Resting",
            "Retired",
        }
        assert set(stages.SAYS) == set(stages.LEVELS)
        assert {level for tab in stages.TABS for level in tab["levels"]} == {
            stages.LIBRARY, stages.BASELINE, stages.SHORTLISTED, stages.ON_TRIAL,
            stages.PREVIEW,
        }
        assert stages.PREVIEW not in stages.TRACK

    def test_the_track_lights_the_level_and_puts_the_baseline_in_librarys_place(
        self,
    ) -> None:
        words = [step["word"] for step in stages.track(stages.ON_TRIAL)]
        assert words == ["Draft", "On trial", "Shortlisted", "Library"]
        assert [step["lit"] for step in stages.track(stages.ON_TRIAL)] == [
            False, True, False, False
        ]
        baseline = stages.track(stages.BASELINE)
        assert [step["word"] for step in baseline] == ["Draft", "On trial", "Shortlisted", "Baseline"]
        assert baseline[-1]["lit"] is True
        assert not any(step["lit"] for step in stages.track(stages.RETIRED))


# ----------------------------------------------------------------- a site with each level


class Levels:
    """A running site with a game at each level the store can hold today."""

    def __init__(self, site: Site, ada: Any, draft: str, trial: str, shortlisted: str) -> None:
        self.site = site
        self.ada = ada
        self.draft = draft
        self.trial = trial
        self.shortlisted = shortlisted

    @property
    def expected(self) -> dict[str, str]:
        return {
            DULL_HASH: stages.BASELINE,
            ARCHIVED_HASH: stages.LIBRARY,
            self.draft: stages.DRAFT,
            self.trial: stages.ON_TRIAL,
            self.shortlisted: stages.SHORTLISTED,
        }


@pytest.fixture
def levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Levels]:
    """`dull` in `rulesets/` and `merger` in `archive/`, each with its loaded row, and
    three of ada's saves: one kept a draft, one put on trial, one shortlisted by hand
    (nominating is Q503f's, so the stamp is written as that task will write it)."""
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    (tmp_path / "rulesets" / "dull.json").write_text(json.dumps(DULL), encoding="utf-8")
    (tmp_path / "archive" / "merger.json").write_text(json.dumps(ARCHIVED), encoding="utf-8")
    _screened(monkeypatch, passed=True)
    with _running_site(tmp_path) as site:
        with site.store.cursor() as cursor:
            for digest, spec, origin in (
                (DULL_HASH, DULL, "corpus"), (ARCHIVED_HASH, ARCHIVED, "archive")
            ):
                cursor.execute(
                    "insert into games (hash, spec, name, origin, visibility)"
                    " values (%s, %s::jsonb, %s, %s, 'public')",
                    (digest, json.dumps(spec), spec["name"], origin),
                )
        site.store.commit()
        ada = site.signed_in("ada")
        draft = _saved(ada, _template(0, "ada's draft")).payload["hash"]
        trial = _saved(ada, _template(1, "ada's trial")).payload["hash"]
        shortlisted = _saved(ada, _template(2, "ada's pick")).payload["hash"]
        site_store.on_trial(site.store, trial)
        with site.store.cursor() as cursor:
            cursor.execute(
                "update games set visibility = 'public', published = now() - interval '40 days',"
                " nominated = now() where hash = %s",
                (shortlisted,),
            )
        site.store.commit()
        yield Levels(site, ada, draft, trial, shortlisted)


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> Callable[[str], str]:
    """Replace `stages.<name>` with one that marks its answer, and return the marker."""
    real = getattr(stages, name)
    monkeypatch.setattr(stages, name, lambda arg: f"asked:{real(arg)}")
    return lambda level: f"asked:{level}"


class TestTheStoreReadsEachLevel:
    def test_read_names_every_row_by_its_level(self, levels: Levels) -> None:
        read = stages.read(levels.site.store, list(levels.expected))
        assert {digest: row["level"] for digest, row in read.items()} == levels.expected

    def test_a_retired_row_is_retired(self, levels: Levels) -> None:
        with levels.site.store.cursor() as cursor:
            cursor.execute("update games set retired = now() where hash = %s", (DULL_HASH,))
        levels.site.store.commit()
        assert stages.read(levels.site.store, [DULL_HASH])[DULL_HASH]["level"] == stages.RETIRED

    def test_a_hash_with_no_row_is_left_out(self, levels: Levels) -> None:
        assert stages.read(levels.site.store, ["0" * 64]) == {}
        assert stages.read(levels.site.store, []) == {}


class TestEveryReaderAsksTheOneFunction:
    """With `stages.stage` marked, every page's level carries the mark."""

    def test_the_card_and_the_tabs(self, levels: Levels, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = levels.site.browser().get("/api/library").payload
        cards = {card["hash"]: card for card in payload["games"]}
        # Unmarked first: the listing's levels, and a draft is never listed.
        assert levels.draft not in cards
        for digest in (DULL_HASH, ARCHIVED_HASH, levels.trial, levels.shortlisted):
            assert cards[digest]["level"] == levels.expected[digest], digest
        tabs = {tab["key"]: tab for tab in payload["site"]["tabs"]}
        assert (tabs["library"]["count"], tabs["trial"]["count"]) == (3, 1)
        assert cards[levels.trial]["onTrialSince"] is not None

        marked = _spy(monkeypatch, "stage")
        payload = levels.site.browser().get("/api/library").payload
        for card in payload["games"]:
            assert card["level"] == marked(levels.expected[card["hash"]]), card["name"]
        # The tabs file cards by the same word, so a marked word is in no tab; a tab
        # past the first that holds nothing is not sent (L4).
        assert [tab["count"] for tab in payload["site"]["tabs"]] == [0]

    def test_every_card_carries_the_day_it_joined(self, levels: Levels) -> None:
        """Q826: a library file's day is its `provenance.added`, a member's game's the day
        it was published, and *date added* (`sort=added`) puts the newest first."""
        from datetime import timezone

        from chess_variant_sandbox.site import browse

        payload = levels.site.browser().get("/api/library").payload
        cards = {card["hash"]: card for card in payload["games"]}
        assert cards[DULL_HASH]["added"] == DULL["provenance"]["added"]
        assert cards[ARCHIVED_HASH]["added"] == ARCHIVED["provenance"]["added"]
        with levels.site.store.cursor() as cursor:
            cursor.execute("select published from games where hash = %s", (levels.trial,))
            (published,) = cursor.fetchone()
        assert cards[levels.trial]["added"] == (
            published.astimezone(timezone.utc).date().isoformat())
        # Published today, so newest; the library's two files follow by their own days.
        ordered = browse.order(payload["games"], browse.ADDED)
        assert ordered[0]["hash"] in (levels.trial, levels.shortlisted)
        days = [card["added"] for card in ordered if card["added"] is not None]
        assert days == sorted(days, reverse=True)
        named = browse.order(levels.site.browser().get(
            "/api/library?sort=added").payload["games"])
        assert [card["hash"] for card in named] == [card["hash"] for card in ordered]

    def test_the_trial_tab_is_absent_until_a_game_is_on_trial(self, levels: Levels) -> None:
        """L4 of `docs/archive/plan-launch.md` §3.4: absent, not empty. So is the order by
        *newest on trial*, which would order nothing."""
        def site_block() -> dict[str, Any]:
            block: dict[str, Any] = levels.site.browser().get("/api/library").payload["site"]
            return block
        shown = site_block()
        assert [tab["key"] for tab in shown["tabs"]] == ["library", "trial"]
        assert "onTrialSince" in {sort["key"] for sort in shown["sorts"]}
        with levels.site.store.cursor() as cursor:
            cursor.execute(
                "update games set visibility = 'private' where hash = %s", (levels.trial,)
            )
        levels.site.store.commit()
        shown = site_block()
        assert [tab["key"] for tab in shown["tabs"]] == ["library"]
        assert "onTrialSince" not in {sort["key"] for sort in shown["sorts"]}

    def test_the_game_page(self, levels: Levels, monkeypatch: pytest.MonkeyPatch) -> None:
        for digest, level in levels.expected.items():
            page = levels.ada.get(f"/api/game/{digest}").payload
            assert page["level"] == level, digest
            if level == stages.SHORTLISTED:
                # Q503h: a nominee's sentence names the round it waits for.
                assert page["levelSays"].startswith("Shortlisted: members vote on it at ")
            else:
                assert page["levelSays"] == stages.SAYS[level]
            assert [step["lit"] for step in page["track"]].count(True) == 1

        marked = _spy(monkeypatch, "stage")
        for digest, level in levels.expected.items():
            page = levels.ada.get(f"/api/game/{digest}").payload
            assert page["level"] == marked(level), digest

    def test_the_designers_panel(self, levels: Levels, monkeypatch: pytest.MonkeyPatch) -> None:
        for digest, level in levels.expected.items():
            found = levels.ada.get(f"/api/design/standing/{digest}").payload
            assert found["level"] == level, digest
            if digest in (levels.draft, levels.trial, levels.shortlisted):
                assert found["publish"]["level"] == level
            else:
                assert found["publish"] is None
        draft = levels.ada.get(f"/api/design/standing/{levels.draft}").payload["publish"]
        assert draft["publishable"] is True
        trial = levels.ada.get(f"/api/design/standing/{levels.trial}").payload["publish"]
        assert trial["publishable"] is False

        marked = _spy(monkeypatch, "stage")
        for digest, level in levels.expected.items():
            found = levels.ada.get(f"/api/design/standing/{digest}").payload
            assert found["level"] == marked(level), digest
            if found["publish"] is not None:
                assert found["publish"]["level"] == marked(level)

    def test_the_profiles_games(self, levels: Levels, monkeypatch: pytest.MonkeyPatch) -> None:
        mine = levels.ada.get("/api/profile").payload["yourGames"]
        assert [group["level"] for group in mine] == [
            stages.DRAFT, stages.ON_TRIAL, stages.SHORTLISTED
        ]
        assert [[game["hash"] for game in group["games"]] for group in mine] == [
            [levels.draft], [levels.trial], [levels.shortlisted]
        ]
        assert [group["games"][0]["inDesigner"] for group in mine] == [True, False, False]

        marked = _spy(monkeypatch, "stage")
        mine = levels.ada.get("/api/profile").payload["yourGames"]
        assert {group["level"] for group in mine} == {
            marked(stages.DRAFT), marked(stages.ON_TRIAL), marked(stages.SHORTLISTED)
        }

    def test_the_same_game_refusal_names_the_level_by_it_too(
        self, levels: Levels, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`drafts.twin` words the game a copy copies by its level, and asks `stage`."""
        from chess_variant_sandbox.site import drafts

        copy = json.loads(json.dumps(ARCHIVED))
        copy["name"] = "merger again"
        copy["pieces"] = list(reversed(copy["pieces"]))
        found = drafts.twin(levels.site.store, spec_hash(copy), copy)
        assert found is not None and found.level == drafts.LEVELS[stages.LIBRARY]

        calls: list[dict[str, Any]] = []
        real = stages.stage
        monkeypatch.setattr(stages, "stage", lambda row: calls.append(dict(row)) or real(row))
        drafts.twin(levels.site.store, spec_hash(copy), copy)
        assert calls, "twin read no level through stages.stage"


class TestACheckoutAsksTheOneFunction:
    """The local application has no rows: its card and game page read `stages.local`."""

    def test_the_card_and_the_game_page(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("rulesets", "runs", "archive"):
            (tmp_path / name).mkdir()
        (tmp_path / "rulesets" / "dull.json").write_text(json.dumps(DULL), encoding="utf-8")
        (tmp_path / "archive" / "merger.json").write_text(json.dumps(ARCHIVED), encoding="utf-8")
        found = library.discover(
            tmp_path / "rulesets", tmp_path / "runs", archive_dir=tmp_path / "archive"
        )
        games = {game.hash: game for game in found}
        assert games[DULL_HASH].summary()["level"] == stages.BASELINE
        assert games[ARCHIVED_HASH].summary()["level"] == stages.LIBRARY
        page = library.detail(games[ARCHIVED_HASH])
        assert page["level"] == stages.LIBRARY and page["track"][-1]["lit"] is True

        marked = _spy(monkeypatch, "local")
        assert games[DULL_HASH].summary()["level"] == marked(stages.BASELINE)
        assert library.detail(games[DULL_HASH])["level"] == marked(stages.BASELINE)


# ------------------------------------------------------------ the pages draw what they get


class TestThePagesHoldNoLevelWord:
    def test_no_page_script_spells_a_level(self) -> None:
        """Every script the two servers answer, read for a level word as a string."""
        import page_text

        words = "|".join(re.escape(word) for word in stages.LEVELS)
        pattern = re.compile("['\"`](" + words + ")['\"`]")
        for route, path in page_text.SERVED.items():
            if not str(path).endswith(".js") or "vendor" in str(path):
                continue
            found = pattern.findall(path.read_text(encoding="utf-8"))
            assert not found, f"{route} spells {found}"

    def test_the_card_draws_the_level_it_is_sent(self) -> None:
        drawn = _in_page(
            _FIXTURE + "LIBRARY[0].level = 'Word <sent>';", "cardHtml(LIBRARY[0])"
        )
        assert '<span class="tag level">Word &lt;sent&gt;</span>' in drawn
        assert 'class="tag level"' not in _in_page(_FIXTURE, "cardHtml(LIBRARY[2])")

    def test_the_tabs_file_cards_by_their_level(self) -> None:
        setup = _FIXTURE + """
          LIBRARY[0].level = 'A'; LIBRARY[1].level = 'B'; LIBRARY[2].level = 'C';
          SITE = { tabs: [
            { key: 'one', label: 'One', levels: ['A', 'C'], count: 2, sort: 'index' },
            { key: 'two', label: 'Two', levels: ['B'], count: 1, sort: 'onTrialSince' } ] };
        """
        first = _in_page(setup, "narrowed().map(game => game.name)")
        assert sorted(first) == ["alpha", "gamma"]
        second = _in_page(setup + "BROWSE.tab = 'two';", "narrowed().map(game => game.name)")
        assert second == ["beta"]
        drawn = _in_page(setup, "tabsHtml()")
        assert "One (2)" in drawn and "Two (1)" in drawn
        # A checkout is sent no tabs and lists everything.
        assert len(_in_page(_FIXTURE, "narrowed()")) == 3
        assert _in_page(_FIXTURE, "tabsHtml()") == ""

    def test_the_game_page_draws_where_it_stands(self) -> None:
        game = {
            "level": "Word", "levelSays": "What the word means.",
            "track": [{"word": "First", "lit": False}, {"word": "Word", "lit": True}],
        }
        drawn = " ".join(_in_page("", f"whereHtml({json.dumps(game)})").split())
        assert "Where it stands: Word" in drawn
        assert '<li class="lit" >Word</li>' in drawn
        assert '<li class="" >First</li>' in drawn
        assert "What the word means." in drawn
        # A game sent no level has no heading; a checkout still draws its keeping actions.
        assert "Where it stands" not in _in_page("", "whereHtml({})")

    def test_a_phone_keeps_the_track_on_one_line(self) -> None:
        """Q517, archive/plan-game-pipeline §3.1 *Phone*: at 375 px the four steps wrapped and put
        *Library* on a line of its own, on the designer and the game page alike."""
        from chess_variant_sandbox.web.page.style import STYLE

        narrow = STYLE[STYLE.index("@media (max-width: 480px)"):]
        narrow = narrow[: narrow.index("\n  }\n")]
        assert ".track { flex-wrap: nowrap;" in narrow
        assert ".track li { white-space: nowrap; }" in narrow

    def test_the_designer_draws_the_track_for_a_game_with_no_publish_panel(self) -> None:
        standing = {
            "level": "Word", "levelSays": "What the word means.", "publish": None,
            "track": [{"word": "First", "lit": False}, {"word": "Word", "lit": True}],
        }
        drawn = " ".join(_in_page("", f"publishHtml({json.dumps(standing)})").split())
        assert "Where this game stands: Word" in drawn
        assert 'class="track"' in drawn and "What the word means." in drawn
        assert "dPublishGo" not in drawn
