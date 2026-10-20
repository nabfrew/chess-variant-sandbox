"""The landing page's *Made by members* (Q741, `site/showcase.py`).

The user, 2026-10-05: *"Switch up the main page to highlight user-created games"*, then
*"just ones that have been published"*. Three claims.

**What qualifies.** Over a scratch site with a game at each level, `/api/library`'s
`madeHere` holds the games members designed and published (On trial, Shortlisted), newest
published first, at most `showcase.SHOWN`, with how many qualify; never a draft, a game the
site came with (Baseline, Library), or a resting one; and the maker as their row reads now.

**The page.** The section is in the landing page's markup for the site only
(`data-needs="accounts"`), and `landing.js`'s `fillMembersShelf`, run in node over a
payload, draws each game's name, maker and Play, the invitation when the shelf is empty,
the count when there are more, and removes the section when there is no listing.

**Nothing moves.** The shelf's height is fixed in the stylesheet, which is what lets it
fill without moving the page; `look.mjs`'s `shelf steady` row measures it in a browser.

The store tests need a real Postgres and skip with a reason without one;
`tests/test_site.py` has the recipe for a throwaway cluster.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_admin import DULL, DULL_HASH, Site, _running_site, _saved, _screened, _template
from test_web import _in_page

from chess_variant_sandbox.site import identity, showcase, stages
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import landing

REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVED = load_spec(REPOSITORY / "archive" / "merger.json")
ARCHIVED_HASH = spec_hash(ARCHIVED)


class Shelf:
    def __init__(self, site: Site, people: dict[str, Any], games: dict[str, str]) -> None:
        self.site = site
        self.people = people
        self.games = games

    def made(self) -> dict[str, Any]:
        found: dict[str, Any] = self.site.browser().get("/api/library").payload["madeHere"]
        return found

    def sql(self, statement: str, *params: Any) -> None:
        with self.site.store.cursor() as cursor:
            cursor.execute(statement, params)
        self.site.store.commit()


def _spec(index: int, name: str) -> dict[str, Any]:
    """A template with its own mechanics, so each save is a game of its own."""
    spec = _template(index % 3, name)
    spec["rules"]["max_plies"] = 150 + 10 * index
    return spec


@pytest.fixture
def shelf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Shelf]:
    """`dull` in the Baseline and `merger` in the Library, and two members' saves: ada's
    draft, ada's trial game published through `drafts.publish`, and grace's game
    shortlisted by hand, published an hour before ada's."""
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
        ada, grace = site.signed_in("ada"), site.signed_in("grace")
        games = {
            "draft": _saved(ada, _spec(0, "ada's draft")).payload["hash"],
            "trial": _saved(ada, _spec(1, "ada's trial")).payload["hash"],
            "pick": _saved(grace, _spec(2, "grace's pick")).payload["hash"],
        }
        site_store.on_trial(site.store, games["trial"])
        made = Shelf(site, {"ada": ada, "grace": grace}, games)
        made.sql(
            "update games set visibility = 'public', published = now() - interval '1 hour',"
            " nominated = now() where hash = %s", games["pick"],
        )
        yield made


class TestWhatIsOnTheShelf:
    def test_published_members_games_newest_first_and_nothing_else(self, shelf: Shelf) -> None:
        made = shelf.made()
        assert [game["hash"] for game in made["games"]] == [
            shelf.games["trial"], shelf.games["pick"]
        ]
        assert made["total"] == 2
        trial, pick = made["games"]
        assert (trial["name"], trial["by"], trial["level"]) == (
            "ada's trial", "ada", stages.ON_TRIAL)
        assert (pick["name"], pick["by"], pick["level"]) == (
            "grace's pick", "grace", stages.SHORTLISTED)
        assert trial["mark"] is not None and trial["published"]
        # Never the draft, nor the games the site came with.
        shown = {game["hash"] for game in made["games"]}
        assert not shown & {shelf.games["draft"], DULL_HASH, ARCHIVED_HASH}

    def test_empty_until_somebody_publishes(self, shelf: Shelf) -> None:
        shelf.sql(
            "update games set visibility = 'unlisted', nominated = null where hash = any(%s)",
            [shelf.games["trial"], shelf.games["pick"]],
        )
        assert shelf.made() == {"games": [], "total": 0}

    def test_a_resting_game_and_a_library_game_leave_it(self, shelf: Shelf) -> None:
        shelf.sql("update games set rested = now() where hash = %s", shelf.games["trial"])
        # The round confirmed grace's game: the loader takes the row over as the library's.
        shelf.sql("update games set origin = 'archive' where hash = %s", shelf.games["pick"])
        assert shelf.made() == {"games": [], "total": 0}

    def test_at_most_the_shelfs_worth_and_how_many_there_are(
        self, shelf: Shelf
    ) -> None:
        ada = shelf.people["ada"]
        more = [_saved(ada, _spec(3 + n, f"ada's game {n}")).payload["hash"]
                for n in range(showcase.SHOWN)]
        for n, digest in enumerate(more):
            shelf.sql(
                "update games set visibility = 'public', published = now() - %s * interval"
                " '1 minute' where hash = %s", n + 1, digest,
            )
        made = shelf.made()
        assert made["total"] == showcase.SHOWN + 2
        # The trial game was published just now; the four follow a minute apart.
        assert [game["hash"] for game in made["games"]] == [shelf.games["trial"], *more[:3]]

    def test_somebody_who_left_reads_as_a_former_player(self, shelf: Shelf) -> None:
        shelf.sql("update users set handle = null where handle = 'grace'")
        pick = shelf.made()["games"][1]
        assert (pick["by"], pick["mark"]) == (identity.FORMER, None)

    def test_the_maker_renamed_reads_as_their_handle_now(self, shelf: Shelf) -> None:
        shelf.sql("update users set handle = 'ada_lovelace' where handle = 'ada'")
        assert shelf.made()["games"][0]["by"] == "ada_lovelace"


# ----------------------------------------------------------------------------- the page


def _fill(payload: Any, viewer: str | None = None) -> dict[str, Any]:
    """Run `fillMembersShelf` over `payload` against the few nodes it touches."""
    setup = """
      const NODES = {};
      for (const id of ['landing-members', 'membersShelf', 'membersEmpty', 'membersAll',
                        'membersCount']) {
        NODES[id] = { id, hidden: id === 'membersEmpty' || id === 'membersAll',
          innerHTML: '', textContent: '', removed: false, busy: true,
          remove() { this.removed = true; }, removeAttribute() { this.busy = false; } };
      }
      document.getElementById = id => NODES[id] || null;
    """ + f"VIEWER = {json.dumps(viewer)};"
    found: dict[str, Any] = _in_page(
        setup, f"(fillMembersShelf({json.dumps(payload)}), NODES)"
    )
    return found


GAME = {"hash": "abc123", "name": "Ada's <game>", "by": "ada", "mark": {"d": "", "hue": 1},
        "level": "On trial", "published": "2026-10-05T10:00:00+00:00"}


class TestThePage:
    def test_the_section_is_the_sites_and_comes_straight_after_the_board(self) -> None:
        page = landing.HTML
        section = re.search(r'<section [^>]*id="landing-members"[^>]*>', page)
        assert section is not None and 'data-needs="accounts"' in section.group(0)
        assert page.index('id="landing-members"') < page.index('id="landing-intent"')
        assert page.index("landingHero") < page.index('id="landing-members"')
        assert landing.MEMBERS_HEADING in page and "Build your own" in page

    def test_a_filled_shelf_names_the_game_its_maker_and_a_way_to_play(self) -> None:
        nodes = _fill({"games": [GAME], "madeHere": {"games": [GAME], "total": 1}})
        drawn = nodes["membersShelf"]["innerHTML"]
        assert "Ada&#39;s &lt;game&gt;" in drawn or "Ada's &lt;game&gt;" in drawn
        assert "by " in drawn and "ada" in drawn
        assert drawn.count('href="/game/abc123"') == 2 and ">Play</a>" in drawn
        assert nodes["membersEmpty"]["hidden"] and nodes["membersAll"]["hidden"]
        assert not nodes["landing-members"]["removed"] and not nodes["membersShelf"]["busy"]
        # A visitor reads the maker's name; a member can open their page.
        assert "/players/" not in drawn
        member = _fill({"games": [GAME], "madeHere": {"games": [GAME], "total": 1}}, "grace")
        assert 'href="/players/ada"' in member["membersShelf"]["innerHTML"]

    def test_more_than_the_shelf_says_how_many(self) -> None:
        nodes = _fill({"games": [], "madeHere": {"games": [GAME], "total": 7}})
        assert not nodes["membersAll"]["hidden"] and nodes["membersCount"]["textContent"] == "7"

    def test_an_empty_shelf_shows_the_invitation(self) -> None:
        nodes = _fill({"games": [], "madeHere": {"games": [], "total": 0}})
        assert not nodes["membersEmpty"]["hidden"]
        assert nodes["membersShelf"]["innerHTML"] == ""
        assert not nodes["landing-members"]["removed"]

    def test_no_listing_no_section(self) -> None:
        assert _fill(None)["landing-members"]["removed"]
        # The local application sends no `madeHere`.
        assert _fill({"games": []})["landing-members"]["removed"]

    def test_the_shelf_holds_one_height_whatever_it_holds(self) -> None:
        from page_text import PAGE

        rule = re.search(r"\.membersShelf \{([^}]*)\}", PAGE)
        assert rule is not None
        assert re.search(r"(?<!-)height: \d+px", rule.group(1))
        assert "grid-auto-flow: column" in rule.group(1)
        assert "overflow-y: hidden" in rule.group(1)

    def test_the_shelf_takes_the_listings_drawings_itself(self) -> None:
        """Q762: `showLanding` filled the shelf before `landingGames` took the listing's
        drawings, so on a first visit every piece on the shelf was its letter."""
        from page_text import PAGE

        shelf = PAGE[PAGE.index("function fillMembersShelf(data)"):]
        shelf = shelf[:shelf.index("\n}\n")]
        assert shelf.index("takePieceArt(data)") < shelf.index("thumbHtml(game)")
        assert PAGE.count("PIECE_ART = ") == 2  # the declaration and `takePieceArt`

    def test_the_days_game_is_named_as_the_days(self) -> None:
        assert 'data-today="Today&#39;s game"' in landing.HTML
        named = _in_page(
            "", "landingGames({ games: [{ hash: 'a', name: 'A', still: {} },"
            " { hash: 'b', name: 'B', still: {} }], firstStep: { hash: 'b', sentence: 's' },"
            " launchSet: [{ hash: 'a', sentence: 't' }] }).map(e => [e.game.hash, e.today])",
        )
        assert named == [["b", True], ["a", False]]


# ------------------------------------------------------------ the pieces the shelf draws

# A piece on a still board is its drawing (`<image href>`) or, where it has none, its letter
# in a disc. The two are told apart here and nothing else is read.
_GLYPH = re.compile(r'<image href="([^"]*)"|<text [^>]*font-weight="600"[^>]*>([^<]*)</text>')

# A drawing of our own, as the designer's upload stores it: one SVG serves both seats.
_UPLOADED = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZp"
    "ZXdCb3g9IjAgMCA0NSA0NSI+PGNpcmNsZSBjeD0iMjIuNSIgY3k9IjIyLjUiIHI9IjE1Ii8+PC9zdmc+"
)


def _glyphs(drawn: str) -> list[str]:
    # The wood's grain is an image too, inside its pattern; a piece never is.
    pieces = re.sub(r"<pattern\b.*?</pattern>", "", drawn, flags=re.S)
    return [href or f"letter {letter.strip()}" for href, letter in _GLYPH.findall(pieces)]


def _custom(name: str, index: int = 0) -> dict[str, Any]:
    """A designer's template with its first piece drawn by an upload of our own."""
    spec = _template(index, name)
    spec["appearance"] = {"icons": {spec["pieces"][0]["id"]: _UPLOADED}}
    return spec


def _drawn(listing: dict[str, Any], openings: dict[str, Any]) -> dict[str, Any]:
    """Each shelf game's pieces as the landing page's shelf draws them on a first visit
    (nothing taken before it), as the games page's card draws them, and as the game's own
    page draws its opening position."""
    setup = """
      const NODES = {};
      for (const id of ['landing-members', 'membersShelf', 'membersEmpty', 'membersAll',
                        'membersCount']) {
        NODES[id] = { id, hidden: true, innerHTML: '', textContent: '',
          remove() {}, removeAttribute() {} };
      }
      document.getElementById = id => NODES[id] || null;
    """ + f"const LISTING = {json.dumps(listing)}, OPENINGS = {json.dumps(openings)};"
    expression = """(() => {
      PIECE_ART = {};
      fillMembersShelf(LISTING);
      const shelf = NODES.membersShelf.innerHTML.split('class="memberGame panel"').slice(1);
      PIECE_ART = {};
      takePieceArt(LISTING);
      const byHash = new Map(LISTING.games.map(game => [game.hash, game]));
      return LISTING.madeHere.games.map((entry, n) => ({
        hash: entry.hash,
        shelf: shelf[n],
        card: thumbHtml(byHash.get(entry.hash)),
        page: boardPreview(Object.assign({}, OPENINGS[entry.hash], { hash: entry.hash }), null),
      }));
    })()"""
    found: dict[str, Any] = _in_page(setup, expression)
    return found


def _same_pieces(drawn: Any, openings: dict[str, Any]) -> None:
    for game in drawn:
        page = _glyphs(game["page"])
        # One glyph a piece standing in the opening position, and nothing else read.
        assert len(page) == sum(1 for value in openings[game["hash"]]["board"] if value)
        assert _glyphs(game["shelf"]) == _glyphs(game["card"]) == page, game["hash"]


class TestTheShelfDrawsTheGamesOwnPieces:
    """Q762: the shelf's boards carry the same pieces as the game's card and its own page,
    square for square, on a first visit to the landing page."""

    @pytest.mark.slow  # four cold cards and their pages, 3.1 s on one core (Q749)
    def test_on_the_local_listing(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.web.server import App

        rulesets, archive = tmp_path / "rulesets", tmp_path / "archive"
        rulesets.mkdir()
        archive.mkdir()
        # The default drawings (chess), a hand and fairy pieces (dobutsu), drawings a
        # ruleset carries (weathervane), and an inline spec with an upload of its own.
        for name in ("chess.json", "dobutsu.json"):
            (rulesets / name).write_text(
                (REPOSITORY / "rulesets" / name).read_text(encoding="utf-8"), encoding="utf-8")
        (archive / "weathervane.json").write_text(
            (REPOSITORY / "archive" / "weathervane.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        (rulesets / "drawn.json").write_text(json.dumps(_custom("drawn by hand")),
                                             encoding="utf-8")
        app = App(rulesets, tmp_path / "runs", archive_dir=archive)
        listing = app.get("/api/library")
        cards = listing["games"]
        assert len(cards) == 4
        listing["madeHere"] = {"games": [
            {"hash": card["hash"], "name": card["name"], "by": "ada", "mark": None}
            for card in cards], "total": len(cards)}
        openings = {card["hash"]: app.get(f"/api/game/{card['hash']}")["opening"]
                    for card in cards}
        drawn = _drawn(listing, openings)
        _same_pieces(drawn, openings)
        # Every piece of every one of these games has a drawing: none is a letter.
        assert not any(glyph.startswith("letter") for game in drawn
                       for glyph in _glyphs(game["shelf"]))
        uploaded = next(game for game in drawn
                        if game["hash"] == next(card["hash"] for card in cards
                                                if card["name"] == "drawn by hand"))
        assert _UPLOADED in _glyphs(uploaded["shelf"])

    def test_on_the_site_for_games_made_in_the_designer(self, shelf: Shelf) -> None:
        ada = shelf.people["ada"]
        drawn_hash = _saved(ada, _custom("ada's drawn game", 1)).payload["hash"]
        shelf.sql("update games set visibility = 'public', published = now()"
                  " where hash = %s", drawn_hash)
        browser = shelf.site.browser()
        listing = browser.get("/api/library").payload
        shown = [game["hash"] for game in listing["madeHere"]["games"]]
        assert shown == [drawn_hash, shelf.games["trial"], shelf.games["pick"]]
        openings = {digest: browser.get(f"/api/game/{digest}").payload["opening"]
                    for digest in shown}
        drawn = _drawn(listing, openings)
        _same_pieces(drawn, openings)
        assert _UPLOADED in _glyphs(drawn[0]["shelf"])
