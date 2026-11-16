"""R-W1 (`docs/roadmap-2026-11.md`): one press on the front page lands on a board.

A front-page play door's own address starts the game: `?play=<agent>` on a game's page starts
a practice game at once (`detail.js` `askedPlay`), so a press that follows the link (a new
tab, a press before the page's script is ready) is not left on the opponent picker. The
opponent is chosen by one rule for every game (`defaultOpponent`): the one asked for when the
game's picker offers it, else the game's gentlest above random (`firstGame`, read by
`launch.opponents` off the rating table, or the ladder's order without one), else the first
ladder rung above random. The picker stays one press from the board (`change opponent`).
"""

from __future__ import annotations

import json
import re
from typing import Any

import node_session
from page_text import PAGE

from chess_variant_sandbox.web import landing, launch

_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game/abcdef012345', hash: '', search: '',
                      href: 'http://x/game/abcdef012345' };
  global.history = { state: null, replaceState: (s, t, url) => { global.REPLACED = url; } };
  global.fetch = () => new Promise(() => {});
"""


def _run(setup: str, body: str) -> Any:
    """Run `body` (async, its value printed) against the page's own functions, in node."""
    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    runner = (f"\n{setup}\n(async () => {{ {body} }})()"
              ".then(v => console.log(JSON.stringify(v)),"
              " e => console.log(JSON.stringify({ error: String(e) })));")
    done = node_session.run(_STUB + script.group(1) + runner)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# A game as `/api/game/<hash>` sends it: the picker's roster weakest first, and the
# gentlest the server named for a first game.
_GAME = {
    "hash": "abcdef012345", "name": "a game", "sides": ["white", "black"],
    "opponents": [
        {"agent": "human", "store": "ladder"},
        {"agent": "random", "store": "ladder"},
        {"agent": "greedy", "store": "ladder"},
        {"agent": "minimax:nodes=400", "store": "ladder"},
        {"agent": "minimax:nodes=1600", "store": "ladder"},
        {"agent": "plan:x", "store": "plans"},
    ],
    "firstGame": {"gentle": {"agent": "minimax:nodes=400", "name": "easy"},
                  "harder": None, "basis": "ratings"},
}


def test_the_default_opponent_is_one_rule_for_every_game() -> None:
    game = json.dumps(_GAME)
    got = _run(f"const G = {game};", """
      const without = Object.assign({}, G, { firstGame: null });
      const lost = Object.assign({}, G, { firstGame: { gentle: { agent: 'gone' } } });
      const people = Object.assign({}, G, { opponents: [{ agent: 'human' }] });
      const plans = Object.assign({}, G, { firstGame: null,
        opponents: [{ agent: 'human' }, { agent: 'plan:x', store: 'plans' }] });
      return [
        defaultOpponent(G, 'minimax:nodes=1600'),  // asked for and offered
        defaultOpponent(G, 'human'),               // a person is never the computer
        defaultOpponent(G, 'nobody'),              // not offered: the game's gentlest
        defaultOpponent(G, 'gentlest'),
        defaultOpponent(without, 'gentlest'),      // no rating read: the ladder above random
        defaultOpponent(lost, null),               // a gentlest the picker lost
        defaultOpponent(plans, null),              // no ladder: the first offered
        defaultOpponent(people, null),             // nobody to play
      ];""")
    assert got == ["minimax:nodes=1600", "minimax:nodes=400", "minimax:nodes=400",
                   "minimax:nodes=400", "greedy", "greedy", "plan:x", None]


def test_the_address_asks_for_a_start_once() -> None:
    got = _run("", """
      const out = [];
      for (const search of ['', '?seat=white', '?seat=white&play=minimax%3Anodes%3D400',
                            '?play=', '?play=gentlest']) {
        location.search = search;
        out.push(askedPlay());
      }
      location.href = 'http://x/game/abcdef012345?seat=black&play=greedy';
      dropAskedPlay();
      out.push(REPLACED);
      out.push(playPath('abcdef012345', null, 'minimax:nodes=400'));
      out.push(playPath('abcdef012345', 'black', null));
      return out;""")
    assert got == [None, None, "minimax:nodes=400", "gentlest", "gentlest",
                   "/game/abcdef012345?seat=black",
                   "/game/abcdef012345?play=minimax%3Anodes%3D400",
                   "/game/abcdef012345?seat=black&play=gentlest"]


def test_a_door_followed_as_a_link_starts_the_game_in_its_seat() -> None:
    """The game's page, opened at a door's address, starts the practice game before it draws
    anything, and takes `?play=` out of the address so a step back lands on the picker."""
    game = json.dumps(_GAME)
    setup = (f"const G = {game}; let STARTED = null; let DREW = false;"
             " detailFor = async () => G; roster = async () => G.opponents;"
             " humanFor = async () => null; setTitle = () => {};"
             " startPractice = async (hash, agent, seat) => { STARTED = [hash, agent, seat]; };"
             " leadHtml = () => { DREW = true; return ''; };")
    got = _run(setup, """
      const out = [];
      for (const search of ['?seat=black&play=minimax%3Anodes%3D1600', '?seat=white&play=gentlest']) {
        location.search = search;
        location.href = 'http://x/game/abcdef012345' + search;
        STARTED = null; DREW = false;
        await showGame('abcdef012345');
        out.push([STARTED, DREW, REPLACED]);
      }
      // Without `?play=` the page is the picker, and nothing starts.
      location.search = '?seat=white'; STARTED = null; DREW = false;
      try { await showGame('abcdef012345'); } catch (stubbed) { /* the fake DOM ends it */ }
      out.push([STARTED, DREW]);
      return out;""")
    assert got == [
        [["abcdef012345", "minimax:nodes=1600", 1], False, "/game/abcdef012345?seat=black"],
        [["abcdef012345", "minimax:nodes=400", 0], False, "/game/abcdef012345?seat=white"],
        [None, True],
    ]


def test_the_front_pages_doors_carry_the_start_in_their_address() -> None:
    """A story's door names its opponent, or the game's default; *Play today's game* is given
    the day's game's address starting against the gentlest the server named for it."""
    assert landing.play_link("abcdef012345", "white") == (
        "/game/abcdef012345?seat=white&play=gentlest")
    assert landing.play_link("abcdef012345", "black", "minimax:nodes=400") == (
        "/game/abcdef012345?seat=black&play=minimax%3Anodes%3D400")
    assert landing.PLAY_DEFAULT == "gentlest" and "const PLAY_DEFAULT = 'gentlest';" in PAGE
    assert "if (today && step && step.hash) today.href = playPath(step.hash);" in PAGE


def test_the_front_page_names_the_game_pages_own_default() -> None:
    """`first_step` takes the game page's `firstGame` when given it (the site passes
    `App._first_game`), so the front page and the game's page never name two players."""
    from datetime import date

    game = type("G", (), {"hash": launch.by_name(launch.FEATURED[0]).hash,
                          "name": launch.FEATURED[0], "ratings": None})()
    ladder = [{"agent": a, "name": a} for a in ("human", "random", "greedy", "mcts:nodes=1600")]
    page_says = {"gentle": {"agent": "mcts:nodes=1600", "name": "tree search"},
                 "harder": None, "basis": "ratings"}
    day = launch.ROTATION_START
    assert isinstance(day, date)
    plain = launch.first_step([game], ladder, day)
    told = launch.first_step([game], ladder, day, chosen=lambda g: page_says)
    assert plain is not None and told is not None
    assert plain["opponents"]["gentle"]["agent"] == "greedy"
    assert told["opponents"] == page_says


def test_the_days_game_starts_against_a_low_rung_in_either_state() -> None:
    """`firstStep`'s opponent, scores current (a rating table) or provisional (none): never a
    person, the random mover or the hardest, and the ladder's order without a table."""
    ladder = [{"agent": a, "name": a} for a in
              ("human", "random", "greedy", "minimax:nodes=400", "minimax:nodes=200000")]
    provisional = launch.opponents(ladder, None)
    assert provisional is not None
    assert provisional["gentle"]["agent"] == "greedy" and provisional["basis"] == "ladder"
    table = {"standings": [{"name": "greedy", "rating": 1300.0},
                           {"name": "minimax:nodes=400", "rating": 1200.0}]}
    current = launch.opponents(ladder, table)
    assert current is not None
    assert current["gentle"]["agent"] == "minimax:nodes=400" and current["basis"] == "ratings"
