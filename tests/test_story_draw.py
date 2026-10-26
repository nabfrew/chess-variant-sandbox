"""A story at random on each load (`docs/plan-landing.md` box LP6, §4.1).

The server sends every visitor the same page, holding every story that still holds, with the
pool on `<html data-stories>`; `web/static/story.js`, in the head after the shell's script,
draws one before the first paint. Held here: the page's bytes do not depend on the draw; the
head script is in the digest table and served by both servers; every story in the pool is
in the markup and the drawn headline; a story that stops holding leaves both; the script's
draw (`?story=`, a reload or a step back keeping the tab's story, any other load drawing
among the others, refused storage, no pool); and `/llms.txt`'s stable paragraph. The browser
half (twenty loads, the scroll, reduced motion) is the look check's and LP6's record.
"""

from __future__ import annotations

import copy
import json
import re
from html import escape
from pathlib import Path
from typing import Any

import node_session
import pytest
from test_stories import LIBRARY_STORIES, Roots, made_score

from chess_variant_sandbox.site import routes, seo
from chess_variant_sandbox.web import landing, page, stories
from chess_variant_sandbox.web.page import assets, front, shell
from chess_variant_sandbox.web.page.style import STYLE

KEYS = [story.key for story in stories.STORIES]


def _render(tmp_path: Path, broken: str = "") -> dict[str, Any]:
    """The stories rendered with each library pair reading as told, except `broken`'s,
    whose ruled-out game now stays in."""
    made = {}
    for story in LIBRARY_STORIES:
        made[story.wrong.game.hash] = made_score(story.key == broken)
        made[story.right.game.hash] = made_score(True)
    saved = stories.read_score
    stories.read_score = lambda app, digest, *_: copy.deepcopy(made.get(digest))  # type: ignore[assignment]
    try:
        return stories.render(Roots(tmp_path))
    finally:
        stories.read_score = saved  # type: ignore[assignment]


@pytest.fixture
def drawn(tmp_path: Path) -> Any:
    """The landing drawn from the four holding stories, as a server keeps it, and put back."""
    rendered = _render(tmp_path)
    landing.use(rendered["stories"])
    yield rendered
    landing.use([])


def _pool(html: str) -> list[str]:
    found = re.search(r'<html lang="en" data-stories="([^"]*)">', html)
    return found[1].split() if found else []


# ------------------------------------------------------------------ the bytes


def test_the_pages_bytes_do_not_depend_on_the_draw(drawn: dict[str, Any]) -> None:
    assert page.page() == page.page()
    assert front.home("https://site.example") == front.home("https://site.example")
    # Nothing the server writes names a draw: the pool, never a chosen story, is on `<html>`.
    html = front.home("https://site.example")
    assert _pool(html) == KEYS
    assert "data-story-at" not in html.split("<body>")[0]
    assert not re.search(r'<html[^>]*\sdata-story=', html)


def test_one_canonical_and_one_set_of_preview_tags(drawn: dict[str, Any]) -> None:
    """`/?story=<key>` is answered by the same bytes as `/`, whose head names `/` and carries
    the front page's preview once, never a story's."""
    html = front.home("https://site.example")
    head = html.split("</head>")[0]
    assert head.count('<link rel="canonical" href="https://site.example/">') == 1
    for tag in ("og:title", "og:description", "og:image", "twitter:title"):
        assert head.count(f'"{tag}"') == 1, tag
    assert escape(front.HOME_DESCRIPTION, quote=True) in head


# ------------------------------------------------------------------ the head script


def test_the_head_script_is_in_the_digest_table() -> None:
    named = assets.name_of(shell.STORY_SCRIPT_ROUTE)
    assert named != shell.STORY_SCRIPT_ROUTE
    assert assets.answer(named) == (shell.STORY_SCRIPT_FILE.read_bytes(), "text/javascript")
    assert shell.HEAD_SCRIPTS[shell.STORY_SCRIPT_ROUTE] == shell.STORY_SCRIPT_FILE
    assert shell.STORY_SCRIPT_ROUTE in routes.STATIC
    head = page.PAGE.split("</head>")[0]
    tags = re.findall(r'<script src="([^"]+)"></script>', head)
    assert tags == [assets.name_of(shell.SHELL_SCRIPT_ROUTE), named]
    # Before the stylesheet, so the drawn story is on `<html>` before anything is painted.
    assert head.index(named) < head.index('rel="stylesheet"')


def test_the_head_script_names_no_story() -> None:
    script = shell.STORY_SCRIPT_FILE.read_text(encoding="utf-8")
    for key in (*KEYS, *(s.key for s in stories.EXAMPLES)):
        assert f"'{key}'" not in script and f'"{key}"' not in script, key


# ------------------------------------------------------------------ the markup


def test_every_story_in_the_pool_is_in_the_markup(drawn: dict[str, Any]) -> None:
    html = front.home("https://site.example")
    pool = _pool(html)
    assert pool == [s["key"] for s in drawn["stories"]] == KEYS
    template = html[html.index('<template id="landingPage">'):]
    assert re.findall(r'<div class="storyPart" data-story="(\w+)">', template) == pool
    # The headline drawn outside the template holds each ending, in the pool's order.
    view = html[html.index('<div id="view">'):html.index('<template id="landingPage">')]
    assert re.findall(r'<span class="clause [\w-]+ on" data-story="(\w+)">', view) == pool


def test_a_story_that_stops_holding_leaves_the_pool_and_the_markup(tmp_path: Path) -> None:
    rendered = _render(tmp_path, broken="door")
    assert [d["key"] for d in rendered["dropped"]] == ["door"]
    try:
        landing.use(rendered["stories"])
        html = front.home("https://site.example")
    finally:
        landing.use([])
    assert _pool(html) == [k for k in KEYS if k != "door"]
    assert 'data-story="door"' not in html
    assert "/?story=door" not in html


def test_no_story_holding_carries_no_pool() -> None:
    landing.use([])
    assert _pool(page.page()) == []
    assert "data-stories" not in page.page().split("<head>")[0]


def test_the_stylesheet_shows_the_drawn_ending_by_its_place() -> None:
    assert len(stories.STORIES) <= landing.STORY_SLOTS
    assert ".drawnClauses > .clause { display: none; }" in STYLE
    assert "html:not([data-story-at]) .drawnClauses > .clause:first-child" in STYLE
    for at in range(landing.STORY_SLOTS):
        assert (f'html[data-story-at="{at}"] .drawnClauses > .clause:nth-child({at + 1})'
                in STYLE)


# ------------------------------------------------------------------ the draw, in node


_DOM = """
const attrs = {}; const store = {};
const root = { getAttribute: k => (k in attrs ? attrs[k] : null),
               setAttribute: (k, v) => { attrs[k] = String(v); } };
globalThis.document = { documentElement: root };
globalThis.window = globalThis;
let HIDE = [];
window.addEventListener = (kind, fn) => { if (kind === 'pagehide') HIDE.push(fn); };
let SHOWN = null;
document.querySelector = () => (SHOWN ? { getAttribute: () => SHOWN } : null);
window.scrollY = 0;
globalThis.location = { pathname: '/', hash: '', search: '' };
let NAV = 'navigate';
globalThis.performance = { getEntriesByType: () => [{ type: NAV }] };
const STORE = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; },
                removeItem: k => { delete store[k]; } };
const REFUSED = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); },
                  removeItem() { throw new Error('denied'); } };
// The reader leaves the page scrolled to `y` with the drawn story shown.
function leave(y) { SHOWN = attrs['data-story']; window.scrollY = y; HIDE.forEach(fn => fn()); SHOWN = null; }
Object.defineProperty(globalThis, 'sessionStorage', { get: () => STORAGE, configurable: true });
// A seeded stream in place of the browser's, so a run is the same every time.
let SEED = 12345;
Math.random = () => { SEED = (SEED * 1103515245 + 12345) % 2147483648; return SEED / 2147483648; };
const SCRIPT = SOURCE;
// One load: a fresh document holding `pool`, at `where`, arrived by `nav`.
function load(pool, nav, where) {
  for (const k of Object.keys(attrs)) delete attrs[k];
  if (pool !== null) attrs['data-stories'] = pool;
  NAV = nav || 'navigate';
  Object.assign(location, { pathname: '/', hash: '', search: '' }, where || {});
  window.landingStory = undefined;
  HIDE = [];
  (0, eval)(SCRIPT);
  return { story: attrs['data-story'] || null, at: attrs['data-story-at'] || null };
}
"""


def _run(body: str, storage: str = "STORE") -> Any:
    script = shell.STORY_SCRIPT_FILE.read_text(encoding="utf-8")
    source = (_DOM.replace("STORAGE", storage).replace("SOURCE", json.dumps(script))
              + body)
    done = node_session.run(source)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


POOL = " ".join(KEYS)


def test_twenty_plain_loads_show_every_story_and_never_one_twice_running() -> None:
    seen = _run(f"const seen = []; for (let i = 0; i < 20; i++) seen.push(load({POOL!r}).story);"
                "console.log(JSON.stringify(seen));")
    assert len(seen) == 20 and set(seen) <= set(KEYS)
    assert all(a != b for a, b in zip(seen, seen[1:], strict=False))
    # Each load draws among the three the last did not show (a seeded stream here).
    assert set(seen) == set(KEYS)


def test_the_place_in_the_pool_goes_with_the_key() -> None:
    said = _run(f"const one = load({POOL!r}); console.log(JSON.stringify(one));")
    assert KEYS.index(said["story"]) == int(said["at"])


def test_a_named_story_is_kept() -> None:
    said = _run(f"const a = load({POOL!r}, 'navigate', {{ search: '?story=blast' }});"
                f"const b = load({POOL!r}, 'reload');"
                "console.log(JSON.stringify([a.story, a.at, b.story]));")
    assert said == ["blast", str(KEYS.index("blast")), "blast"]


def test_a_reload_and_a_step_back_keep_the_tabs_story() -> None:
    said = _run(f"const a = load({POOL!r}); const b = load({POOL!r}, 'reload');"
                f"const c = load({POOL!r}, 'back_forward'); const d = load({POOL!r});"
                "console.log(JSON.stringify([a.story, b.story, c.story, d.story]));")
    assert said[0] == said[1] == said[2]
    assert said[3] != said[0]


def test_a_reload_comes_back_where_the_reader_was() -> None:
    said = _run(f"const a = load({POOL!r}); leave(2200);"
                f"const b = load({POOL!r}, 'reload'); const first = window.landingScroll();"
                "const again = window.landingScroll(); leave(900);"
                f"const c = load({POOL!r}, 'back_forward'); const back = window.landingScroll();"
                f"leave(500); const d = load({POOL!r}); const fresh = window.landingScroll();"
                "console.log(JSON.stringify([a.story === b.story, first, again, back, fresh]));")
    assert said == [True, 2200, None, 900, None]


def test_refused_storage_comes_back_at_the_top() -> None:
    said = _run(f"load({POOL!r}); leave(2200); load({POOL!r}, 'reload');"
                "console.log(JSON.stringify(window.landingScroll()));", storage="REFUSED")
    assert said is None


def test_an_unknown_story_is_a_draw() -> None:
    said = _run(f"const a = load({POOL!r}, 'navigate', {{ search: '?story=nothing' }});"
                "console.log(JSON.stringify(a));")
    assert said["story"] in KEYS


def test_refused_storage_still_draws() -> None:
    said = _run(f"const seen = []; for (let i = 0; i < 8; i++) seen.push(load({POOL!r}).story);"
                f"const r = load({POOL!r}, 'reload').story;"
                "console.log(JSON.stringify({ seen, r }));", storage="REFUSED")
    assert all(key in KEYS for key in [*said["seen"], said["r"]])


def test_no_pool_draws_nothing() -> None:
    said = _run("const a = load(null); const b = load('');"
                "console.log(JSON.stringify([a, b, window.landingStory()]));")
    assert said == [{"story": None, "at": None}, {"story": None, "at": None}, None]


def test_a_load_of_the_games_spends_no_draw() -> None:
    """`/game` and a game's address draw only when the landing is opened from them, so a
    trip to the games and a step back leave the tab's story where it was."""
    said = _run(f"const a = load({POOL!r});"
                f"const g = load({POOL!r}, 'navigate', {{ pathname: '/game' }});"
                f"const h = load({POOL!r}, 'navigate', {{ pathname: '/game/0123456789ab' }});"
                f"const back = load({POOL!r}, 'back_forward');"
                f"const about = load({POOL!r}, 'navigate', {{ pathname: '/game', hash: '#/about' }});"
                "console.log(JSON.stringify([a.story, g.story, h.story, back.story, about.story]));")
    assert said[1] is None and said[2] is None
    assert said[3] == said[0]
    assert said[4] in KEYS and said[4] != said[0]


def test_the_landing_keeps_the_drawn_part() -> None:
    script = (Path(page.SCRIPT_DIR) / "landing.js").read_text(encoding="utf-8")
    body = script[script.index("function chooseStory"):script.index("function dropUnoffered")]
    assert "window.landingStory()" in body
    # Without the head script it falls back to `?story=` and then the first.
    assert "get('story')" in body and "parts[0]" in body


# ------------------------------------------------------------------ llms.txt


def test_llms_names_the_pairs_in_the_pools_order(tmp_path: Path) -> None:
    rendered = _render(tmp_path)
    text = seo.llms("https://site.example", [], rendered["stories"])
    assert text == seo.llms("https://site.example", [], rendered["stories"])
    lines = [line for line in text.splitlines() if line.startswith("- ") and "ruled out" in line]
    assert len(lines) == len(KEYS)
    for line, story in zip(lines, rendered["stories"], strict=True):
        assert line.startswith(f"- {story['title']} ")
        for side in ("failed", "fixed"):
            assert (f"]({'https://site.example'}{front.game_path(str(story[side]['hash']))})"
                    in line), (story["key"], side)
    assert "## The front page's stories" in text
    assert text.index("## The front page's stories") < text.index("## The games")
    assert "random" in text  # said once, as what the page does; the file itself is not


def test_llms_says_an_example_has_no_address(tmp_path: Path) -> None:
    ground = stories.render_story(stories.GROUND, Roots(tmp_path))
    text = "\n".join(seo.story_lines("https://site.example", [ground]))
    assert "an example made for the page" in text
    assert "](" not in text


def test_llms_without_stories_has_no_paragraph() -> None:
    assert "front page's stories" not in seo.llms("https://site.example", [])
