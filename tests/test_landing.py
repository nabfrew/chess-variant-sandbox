"""The landing page (Q718): what a visitor meets at `/` before the games.

`web/landing.py` makes the page's markup from the scoring itself, and `landing.js` draws its
opening board. These hold three things: the markup is what the queue asked for, in order;
its words pass the plain-words guard, on the page and on the wire; and the score's
ingredients agree with the weights file, read here on its own, so the page cannot describe
a scoring rule that is not the one in force.

`TestOnTheWire` reads a running site and so needs the test database (`tests/test_site.py`
has the recipe); it skips without one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import node_session
import page_text
import pytest
import site_words
from page_text import CORE, PAGE, view

from chess_variant_sandbox import corpus
from chess_variant_sandbox.metrics import scoring
from chess_variant_sandbox.web import landing, page

LANDING = view("landing")


def _weights() -> dict:
    path = Path(scoring.WEIGHTS_DIR) / f"{scoring.DEFAULT_WEIGHTS}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ingredient(name: str) -> str:
    found = re.search(rf'<li class="ingredient" data-ingredient="{name}">(.*?)</li>',
                      landing.HTML, re.S)
    assert found, name
    return found[1]


class TestTheMarkup:
    def test_the_page_carries_it_as_a_template(self) -> None:
        assert page.PAGE.count('<template id="landingPage">') == 1
        assert landing.HTML in page.PAGE

    def test_its_script_loads_before_the_boot(self) -> None:
        routes = list(page.SCRIPTS)
        assert routes.index("/static/page/landing.js") == routes.index("/static/page/boot.js") - 1

    def test_the_sections_in_the_order_asked(self) -> None:
        keys = re.findall(r'<section class="landingPart" id="landing-([a-z]+)"', landing.HTML)
        assert keys == ["intent", "palette", "play", "people", "viable", "score", "more"]
        assert landing.HTML.index('class="landingHero"') < landing.HTML.index('id="landing-intent"')

    def test_the_intention_in_the_users_words(self) -> None:
        assert "vibe-coded playground for designing chess-like strategy games" in landing.HTML

    def test_the_score_is_an_estimate_and_its_ingredients_start_shut(self) -> None:
        score = landing.HTML[landing.HTML.index('id="landing-score"'):]
        assert "preliminary, evolving measure" in score
        assert "not to be treated as a definitive judgement" in score
        assert "provisional" in score
        details = re.search(r'<details[^>]*id="landingIngredients"[^>]*>', landing.HTML)
        assert details and " open" not in details[0]

    def test_the_games_are_one_tap_away(self) -> None:
        hero = landing.HTML[:landing.HTML.index('id="landing-intent"')]
        assert '<a class="button primary" href="/game">See the games</a>' in hero
        assert '<li><a href="/game">The library</a>' in landing.HTML

    def test_the_third_way_in_is_the_readers_own(self) -> None:
        """Q723: everybody meets the page, so *Sign in* is a visitor's and a member is
        offered their own page in its place; `landing.js` removes the one that does not
        apply, and both on the local application, which has no accounts."""
        hero = landing.HTML[:landing.HTML.index('id="landing-intent"')]
        assert '<a class="button" href="/signin" data-needs="visitor">Sign in</a>' in hero
        assert '<a class="button" href="/profile" data-needs="member">Your page</a>' in hero
        assert "visitor: ACCOUNTS && !VIEWER, member: ACCOUNTS && !!VIEWER" in LANDING

    def test_the_contact_line_points_at_the_github_profile_and_names_no_email(self) -> None:
        """Q720: the user's words, *"Refer to my github"*. A profile link, no address to
        harvest, nothing fetched: a plain link the visitor follows or not."""
        assert landing.CONTACT == "https://github.com/nabfrew"
        foot = re.search(rf'<div class="landingFoot" id="{landing.CONTACT_ID}">(.*?)</div>',
                         landing.HTML, re.S)
        assert foot, "the contact line sits in the place Q718 kept for it"
        assert f'<a href="{landing.CONTACT}" rel="noopener">{landing.CONTACT_SAID}</a>' in foot[1]
        assert "@" not in foot[1] and "<img" not in foot[1] and "<script" not in foot[1]
        assert site_words.hits(site_words.visible(foot[1])) == []

    def test_the_footer_script_links_the_same_contact(self) -> None:
        core = (Path(page.__file__).parent.parent / "static" / "page" / "core.js").read_text(
            encoding="utf-8")
        assert f"const CONTACT = '{landing.CONTACT}';" in core
        assert "link(CONTACT, 'contact')" in core

    def test_the_site_only_parts_say_so(self) -> None:
        assert 'id="landing-people" data-needs="accounts"' in landing.HTML
        assert '<li data-needs="round"><a href="#/round">' in landing.HTML


# The router, run under Node with the views stubbed: what each address draws, and what
# the address bar says after (Q723). `core.js`'s `render` and its two helpers alone, so
# nothing here depends on a view's own drawing.
_ROUTER = """
const __el = () => ({ textContent: '', style: {}, dataset: {}, hidden: false,
  addEventListener() {}, append() {}, appendChild() {}, remove() {}, setAttribute() {},
  removeAttribute() {},
  querySelector: () => null, querySelectorAll: () => [], classList: { add() {}, remove() {},
  toggle() {} } });
global.document = { getElementById: () => __el(), querySelector: () => null,
  querySelectorAll: () => [], addEventListener() {}, body: __el(), createElement: __el };
global.window = { addEventListener() {}, scrollTo() {}, innerWidth: 1200,
  matchMedia: () => ({ matches: false, addEventListener() {} }) };
global.location = { pathname: PATH, hash: HASH,
  get href() { return 'https://site.example' + this.pathname + this.hash; } };
const __moved = [];
const __to = (kind, url) => {
  __moved.push(kind + ' ' + url);
  const at = new URL(url, location.href);
  location.pathname = at.pathname; location.hash = at.hash;
};
global.history = { replaceState: (s, t, url) => __to('replace', url),
                   pushState: (s, t, url) => __to('push', url) };
"""

_VIEWS = """
const __shown = [];
const __show = (what) => { __shown.push(what); return Promise.resolve(); };
showLanding = () => __show('landing');
showLibrary = () => __show('library');
showGame = (hash) => __show('game ' + hash);
showExplorer = () => __show('map');
showWhatsNew = () => __show('whats-new');
showRound = (day) => __show('round ' + day);
showDesigner = (from) => __show('design ' + from);
showBoard = (session) => __show('play ' + session.id);
api = (path) => Promise.resolve({ id: path.split('/').pop() });
pending = () => {};
VIEWER = WHO; ADMIN = BOSS; ACCOUNTS = true;
"""


def _route(path: str, hash_: str, then: str = "", who: str = "visitor") -> dict:
    """Draw `path` + `hash_` as `who` (*visitor*, *member*, *admin*), then run `then`."""
    viewer = {"visitor": "null", "member": "'ada'", "admin": "'root'"}[who]
    head = (_ROUTER.replace("PATH", json.dumps(path)).replace("HASH", json.dumps(hash_)))
    stubs = _VIEWS.replace("WHO", viewer).replace("BOSS", str(who == "admin").lower())
    # The views are other files; `core.js` reads them only when a route is drawn.
    names = ["showLanding", "showLibrary", "showGame", "showExplorer", "showWhatsNew", "showRound", "showDesigner", "showBotBuilder", "showBoard", "showRun", "stopPolling"]
    declared = "".join(f"var {name} = () => Promise.resolve();\n" for name in names)
    source = (head + declared + CORE + stubs + "render();\n" + then
              + "\nsetTimeout(() => console.log(JSON.stringify({ shown: __shown, "
              "moved: __moved, at: location.pathname + location.hash })), 20);\n")
    done = node_session.run(source)
    assert done.returncode == 0, done.stderr
    said: dict = json.loads(done.stdout.strip().splitlines()[-1])
    return said


VIEWERS = ("visitor", "member", "admin")


class TestTheRoute:
    """Q723, the user's words (2026-10-04): *"Make it the landing page for all, with a
    separate `/game` page to get to the games."*"""

    @pytest.mark.parametrize("who", VIEWERS)
    def test_everybody_meets_the_landing_page_at_the_bare_address(self, who: str) -> None:
        for path in ("/", "/index.html"):
            assert _route(path, "", who=who) == {"shown": ["landing"], "moved": [], "at": path}

    @pytest.mark.parametrize("who", VIEWERS)
    def test_everybody_finds_the_games_at_game(self, who: str) -> None:
        assert _route("/game", "", who=who) == {"shown": ["library"], "moved": [],
                                                "at": "/game"}

    @pytest.mark.parametrize("path", ["/", "/game"])
    def test_a_saved_hash_root_is_the_games_said_as_game(self, path: str) -> None:
        assert _route(path, "#/") == {"shown": ["library"], "moved": ["replace /game"],
                                      "at": "/game"}

    @pytest.mark.parametrize("path", ["/", "/game"])
    def test_about_is_the_landing_page_anywhere(self, path: str) -> None:
        assert _route(path, "#/about")["shown"] == ["landing"]

    @pytest.mark.parametrize("hash_, shown", [
        ("#/game/abc123", "game abc123"), ("#/play/s42", "play s42"), ("#/map", "map"),
        ("#/design", "design "), ("#/design/abc123", "design abc123"),
        ("#/whats-new", "whats-new"), ("#/round", "round "),
    ])
    @pytest.mark.parametrize("path", ["/", "/game"])
    def test_deep_links_from_before_game_still_open_what_they_named(
        self, path: str, hash_: str, shown: str
    ) -> None:
        """Emails, challenge links, bookmarks and shared game links all say `/#/...`."""
        assert _route(path, hash_) == {"shown": [shown], "moved": [], "at": path + hash_}

    def test_every_way_back_to_the_library_lands_on_game(self) -> None:
        """*Back* on a game, a deleted game, the map's *list*: each says `go('#/')`."""
        said = _route("/", "#/game/abc123", "location.hash = '#/'; render();")
        assert said == {"shown": ["game abc123", "library"], "moved": ["replace /game"],
                        "at": "/game"}

    def test_the_name_goes_to_the_landing_page_and_back_and_forward_redraw(self) -> None:
        said = _route("/game", "", "goPath('/');"
                      "location.pathname = '/game'; onHistory();"
                      "location.pathname = '/'; onHistory();")
        assert said == {"shown": ["library", "landing", "library", "landing"],
                        "moved": ["push /"], "at": "/"}
        assert "document.getElementById('home').onclick = () => goPath('/');" in PAGE

    def test_a_step_that_changes_the_hash_is_drawn_once(self) -> None:
        said = _route("/", "", "location.hash = '#/map'; onHistory();")
        assert said["shown"] == ["landing"]

    def test_the_footer_links_the_games_and_the_landing_page_on_both_servers(self) -> None:
        assert "link(GAMES_PATH, 'games'), ' · ', link('/', 'about')" in PAGE


class TestTheWayIn:
    """Where the site sends a person to the games, decided by its meaning (Q723)."""

    def test_signing_in_and_accepting_an_invitation_go_to_the_games(self) -> None:
        post = page_text.text("/static/site/post.js")
        assert "return held && LINK_SHAPE.test(held) ? held : '/game';" in post
        for name in ("signin", "join"):
            assert "window.location.href = afterwards();" in page_text.text(
                f"/static/site/{name}.js")

    def test_the_sites_pages_link_the_games_and_the_name_goes_home(self) -> None:
        from chess_variant_sandbox.site import pages

        for drawn in (pages.privacy(), pages.credits(), pages.terms_page(), pages.not_found()):
            assert '<h1 class="brand"><a href="/">' in drawn
            assert '<a href="/game">Games</a>' in drawn
            assert '<a href="/game">games</a>' in drawn
        assert '<a class="button primary" href="/game">See the games</a>' in pages.not_found()
        assert 'href="/#/"' not in pages.not_found()

    def test_the_application_header_says_games_plainly(self) -> None:
        assert '<a href="/game">Games</a>' in page.PAGE

    def test_a_challenge_put_off_and_an_empty_profile_offer_the_games(self) -> None:
        assert "later.href = '/game';" in page_text.text("/static/site/challenge.js")
        assert "const START = '/game';" in page_text.text("/static/site/profile.js")

    def test_both_servers_answer_game_with_the_page(self) -> None:
        from chess_variant_sandbox.web import server

        assert server.PAGE_PATHS == ("/", "/index.html", "/game")


class TestTheBoard:
    def test_it_draws_with_the_cards_own_still_board(self) -> None:
        assert "thumbHtml(entry.game)" in LANDING
        assert "function boardPreview" not in LANDING and "squareXY" not in LANDING

    def test_it_stands_still_under_reduced_motion_and_offers_a_pause(self) -> None:
        assert "const moving = !STILL.matches && named.length > 1;" in LANDING
        assert 'id="landingPause"' in landing.HTML
        from chess_variant_sandbox.web.page.style import STYLE

        reduced = STYLE[STYLE.index("@media (prefers-reduced-motion: reduce)"):]
        assert ".landingBoard .settle { opacity: 1; animation: none; }" in reduced

    def test_it_plays_no_move(self) -> None:
        for word in ("/api/session", "legal", "apply"):
            assert word not in LANDING


class TestTheWords:
    def test_the_page_passes_the_words_guard(self) -> None:
        assert site_words.hits(site_words.visible(landing.HTML)) == []

    def test_the_script_passes_the_words_guard(self) -> None:
        for text in site_words.literals(LANDING):
            assert site_words.hits(text) == [], text

    def test_the_correspondence_rule_is_the_sites(self) -> None:
        from chess_variant_sandbox.site import quota

        assert landing.CORRESPONDENCE_IDLE_SAID == quota.CORRESPONDENCE_IDLE_SAID
        assert landing.CORRESPONDENCE_IDLE_SAID in landing.HTML

    def test_the_players_and_games_are_the_measuring_runs(self) -> None:
        count = len(corpus.CANONICAL_AGENTS)
        assert f"{landing._say(count)} computer players play it" in landing.HTML
        assert f"{count * count * corpus.CANONICAL_GAMES:,} games a run" in landing.HTML


class TestTheIngredientsAgreeWithTheWeights:
    def test_every_scored_ingredient_and_nothing_else(self) -> None:
        listed = set(re.findall(r'data-ingredient="([a-z_]+)"', landing.HTML))
        assert listed == set(_weights()["components"])

    def test_each_weight_is_its_share_of_the_file(self) -> None:
        components = _weights()["components"]
        total = sum(entry["weight"] for entry in components.values())
        for name, entry in components.items():
            assert f"weight {entry['weight'] / total * 100:.1f}%" in _ingredient(name), name

    def test_each_quality_is_its_share_of_the_file(self) -> None:
        components = _weights()["components"]
        total = sum(entry["weight"] for entry in components.values())
        families: dict[str, float] = {}
        for entry in components.values():
            families[entry["family"]] = families.get(entry["family"], 0.0) + entry["weight"]
        for family, weight in families.items():
            block = re.search(rf'data-quality="{family}"><h4>(.*?)</h4>', landing.HTML)
            assert block, family
            assert f"{weight / total * 100:.1f}% of the weight" in block[1]

    def test_each_scale_names_the_files_ends(self) -> None:
        for name, entry in _weights()["components"].items():
            said = landing.INGREDIENTS[name]
            text = _ingredient(name)
            for value in entry["points"].values():
                assert landing.written(float(value), said.unit, said.noun) in text, (name, value)

    def test_every_test_that_rules_a_game_out_with_its_number(self) -> None:
        start = landing.HTML.index('class="ruleOuts"')
        tests = landing.HTML[start:landing.HTML.index("</ul>", start)]
        gates = _weights()["gates"]
        assert tests.count("<li>") == len(gates)
        for gate in gates:
            bound = gate.get("minimum", gate.get("maximum"))
            words, unit = landing.RULE_OUTS[gate["component"]]
            assert words.format(n=landing.written(float(bound), unit)) in tests

    def test_the_version_and_the_least_a_quality_counts(self) -> None:
        number = _weights()["version"].removeprefix("v")
        assert f"the scoring rules in force now, version {number}." in landing.HTML
        assert f"No quality counts as lower than {scoring.FLOOR:g}" in landing.HTML

    def test_the_numbers_are_written_as_read(self) -> None:
        assert landing.written(0.735, "number") == "0.735"
        assert landing.written(0.2954, "share") == "29.5%"
        assert landing.written(112.7434, "number", "moves") == "112.7 moves"
        assert landing.written(0.5, "seat") == "75% of the points"
        assert landing.written(6400, "number") == "6,400"

    def test_a_scored_ingredient_without_words_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        words = dict(landing.INGREDIENTS)
        words.pop("stall_rate")
        monkeypatch.setattr(landing, "INGREDIENTS", words)
        with pytest.raises(landing.LandingWordsMissing, match="stall_rate"):
            landing.ingredients()


class TestOnTheWire:
    def test_the_site_sends_it_at_the_bare_address_in_plain_words(self) -> None:
        from test_site_admin import _running_site

        with _running_site() as site:
            answer = site.browser().get("/")
            assert answer.status == 200
            assert '<template id="landingPage">' in answer.text
            assert "/static/page/landing.js" in answer.text
            words = site_words.visible(answer.text)
            assert "The score: a best estimate, not a judgement" in words
            assert site_words.hits(words) == []
            script = site.browser().get("/static/page/landing.js")
            assert script.status == 200

    def test_the_three_viewers_get_the_page_at_both_addresses(self) -> None:
        """Q723: a visitor, a member and a maintainer are each answered the one page at `/`
        and at `/game` (which of the two it draws is the router's, `TestTheRoute`), and an
        address beside them is the not-found page with its way back to the games."""
        from test_site_admin import _running_site

        from chess_variant_sandbox.site import admin, pages
        from chess_variant_sandbox.web.page import PAGE as SERVED

        with _running_site() as site:
            browsers = {"visitor": site.browser(), "member": site.signed_in("ada"),
                        "admin": site.signed_in("root", admin.ADMIN)}
            for who, browser in browsers.items():
                for path in ("/", "/game", "/game?from=mail"):
                    answer = browser.get(path)
                    assert (answer.status, answer.text == SERVED) == (200, True), (who, path)
                for path in ("/games", "/game/abc123", "/landing"):
                    answer = browser.get(path)
                    assert (answer.status, answer.text) == (404, pages.not_found()), (who, path)
            assert '<a class="button primary" href="/game">See the games</a>' in (
                pages.not_found())
