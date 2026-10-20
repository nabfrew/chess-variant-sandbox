"""The site's own short pages -- joining, signing in, a profile, the players, the
suggestion box, the badges, credits, privacy and the maintainer's page.

Tasks 1.3, 1.9 and 1.8, plus Q279's box. Not the local application's page --
`chess_variant_sandbox/web/page/` is one long page about a board and these are short ones about a
person and what is owed them. They follow the same convention
(`.claude/rules/viz-and-web.md`): markup in Python strings, assembled here and served by
the site's handler, and script in plain files under `site/static/`, which no page
carries inline (`SCRIPTS` below).

They wear the application's shell rather than one of their own: `_shell` draws its
header and its footer and the page imports its stylesheet, so the ground, the ink, the
buttons and the links are the same on both sides of a sign-in (box D1 of
`docs/archive/plan-ux.md`; the bare box ids B2 and D0 to D4 below are that file's too).
What the corner of the header says is the one difference. The application asks
`/api/me` after it boots; these pages are rendered by the server, which hands each the
handle it already knows, and credits and privacy get nothing because they read no
cookie.

Two rules run through the forms, and they are §6's.

**Every user-supplied string is escaped or a text node.** The handle rendered into the
profile and the admin page goes through `html.escape`; everything drawn from `GET
/api/profile` and `GET /api/admin` is set with `textContent`. There is no `innerHTML` in
these pages' scripts, which is what task 1.10 is going to audit the long page for.

**A form posts JSON to the API and reads the sentence back.** No redirect carries an
error, so nothing about a failed sign-in lands in a URL or a log.

Credits and privacy are 1.9's and are prose. Neither carries a script, neither asks who
is reading, and neither opens a connection -- `routes.PERSONAL` is what keeps the cookie
lookup off them. The notice says what `site/identity.py` actually stores and what the
retention answer promises, so an edit to either belongs here in the same commit.

`admin` is 1.8's and is the one page a member cannot reach. It draws nothing itself: the
readings are `site/admin.py`'s and the rationed numbers are `site/quota.py`'s, so the page
has no number of its own to fall behind.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from chess_variant_sandbox.site import asking, quota, routes, seo, terms
from chess_variant_sandbox.site import badges as site_badges
from chess_variant_sandbox.viz import pieces
from chess_variant_sandbox.web.page import front, marks, shell
from chess_variant_sandbox.web.page.style import STYLE as APP_STYLE

COOKIE = "chess_variant_sandbox_session"

# What these pages add to the application's stylesheet, which they import rather than
# answer with a second palette (`docs/archive/plan-ux.md` D1). Everything shared -- the
# ground, the ink, the header rule, buttons, links, `.row`, `.panel`, `td.num` -- is the
# app's and is not restated here. What is left is the four things the app has no use for: a
# form, the three counts, the maintainer's tables and a page of prose.
_OWN = """
  header h1 a { color: var(--ink); border-bottom: none; }
  main { max-width: 32rem; }
  main.wide { max-width: 54rem; }
  main.sheet { max-width: none; }
  h1 { cursor: default; }
  main h1 { font-size: 28px; line-height: 34px; margin: 0 0 8px; }
  main h2 { font-size: 20px; line-height: 26px; margin: 26px 0 6px; }
  p.lead { margin: 0 0 22px; color: var(--muted); }
  p.under { margin: 0 0 10px; font-size: 13px; color: var(--muted); }
  /* A form (V3, plan-visual.md §3.5): the app's fields and buttons at the controls' one
     height, each field named above it, the sign-in and the invitation's fields on a sheet
     of their own (`.panel.form`). */
  label { display: block; margin: 0 0 14px; }
  label > span { display: block; font-size: 13px; line-height: 18px; font-weight: 600;
                 color: var(--muted); margin-bottom: 6px; }
  input:where(:not([type=checkbox], [type=radio], [type=range])) { width: 100%; }
  input[type=checkbox] { width: 18px; height: 18px; margin: 0 6px 0 0; vertical-align: -3px; }
  .panel.form { padding: 20px 20px 8px; margin: 0 0 20px; }
  .panel.form .row { margin-top: 4px; }
  .panel.form .note:empty { min-height: 12px; }
  .note { min-height: 20px; margin: 12px 0 0; color: var(--bad); }
  .note.done { color: var(--muted); }
  .hint { font-size: 13px; color: var(--muted); margin: -8px 0 16px; }
  form.invite { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;
                max-width: 36rem; }
  form.invite label { flex: 1 1 16rem; margin: 0; }
  /* The counts (V7): each a small sheet of the panel's paper, the number in the mono face
     and its words under it, side by side and wrapping on a phone. */
  .counts { display: flex; gap: 12px; flex-wrap: wrap; margin: 0 0 22px; }
  .counts:empty { display: none; }
  .counts div { flex: 1 1 7.5rem; max-width: 12rem; padding: 10px 14px 9px;
                background: var(--paper) 0 0 / 180px, var(--panel); border: 1px solid var(--line);
                border-radius: var(--r-panel); box-shadow: var(--shadow); }
  .counts b { display: block; font: 600 26px/32px var(--f-mono); color: var(--ink); }
  .counts span { font-size: 13px; line-height: 18px; color: var(--muted); }

  /* The maintainer's rows, the profile's and the players' (V7): a table is a sheet of the
     panel's paper, as a panel is, its head a quiet band and its rows ruled, the numbers in
     the mono face. A table too wide for a phone scrolls inside its own sheet (`.scroller`),
     never the page. */
  main table { margin-bottom: 4px; border-collapse: separate; border-spacing: 0;
               background: var(--paper) 0 0 / 180px, var(--panel); border: 1px solid var(--line);
               border-radius: var(--r-panel); box-shadow: var(--shadow); font-size: 14px; }
  main th { font-weight: 600; font-size: 13px; color: var(--muted); text-align: left;
            background: var(--sunk); }
  main th, main td { padding: 7px 12px; border-bottom: 1px solid var(--line);
                     vertical-align: top; }
  main tbody tr:last-child td { border-bottom: none; }
  main td.num { font: 14px/20px var(--f-mono); color: var(--ink); }
  /* A row's own action, the compact size; a grave one is the app's danger button. */
  button.act { min-height: var(--ctl-h-sm); padding: 3px 10px; font-size: 13px; line-height: 18px;
               margin: 0 5px 4px 0; }
  td.state { font-size: 13px; }
  td.state.killed, td.state.retired, td.state.deleted, td.state.left,
  td.state.expired { color: var(--muted); }
  span.state { font-size: 12px; color: var(--muted); }
  /* Q831: a visitor's note keeps its line breaks and gets room to be read. */
  td.ask-note { white-space: pre-line; min-width: 14rem; }
  code.hash { font-size: 12px; color: var(--muted); }
  /* Every table on the maintainer's page scrolls inside its own box, so nine columns of
     usage cannot widen the page itself (D3). */
  .scroller { overflow-x: auto; margin-bottom: 14px; border-radius: var(--r-panel); }
  .scroller > table { min-width: 30rem; margin-bottom: 0; }
  /* A row's buttons stay on one line, so a row is one line high and the sheet scrolls. */
  .scroller td:has(> button) { white-space: nowrap; }
  .scroller > table.use { min-width: 52rem; }

  /* The round's table (Q537). Seven columns, the last a reason box and two buttons, do
     not fit a phone even scrolled, so below 760px each entrant is a block of labelled
     lines (`admin.js` copies each heading into its cell's `data-label`) and nothing
     scrolls sideways. The `archive` command breaks between words and never inside a
     flag; only the long hash may break anywhere. */
  code.command { overflow-wrap: anywhere; }
  code.command .flag { white-space: nowrap; }
  @media (max-width: 760px) {
    .scroller > table.round { min-width: 0; }
    table.round thead { display: none; }
    table.round, table.round tbody, table.round tr, table.round td { display: block; }
    table.round tr { border-bottom: 1px solid var(--line); padding: 6px 0 8px; }
    table.round td { border-bottom: none; padding: 2px 0; }
    table.round td:empty { display: none; }
    table.round td[data-label]::before { content: attr(data-label) ": ";
                                         color: var(--muted); font-size: 12px; }
    table.round td.num { text-align: left; }
    table.round td.acts input { display: block; margin: 4px 0 6px; }
  }

  /* Your page's settings (V7): the move email's box is a sheet of its own, as a form is. */
  #move-email:not(:empty) { max-width: 40rem; padding: 2px 16px 4px; margin: 6px 0 18px;
                            background: var(--paper) 0 0 / 180px, var(--panel);
                            border: 1px solid var(--line); border-radius: var(--r-panel);
                            box-shadow: var(--shadow); }
  /* A way on drawn as a button (the not-found page): the app's two kinds, on a link. */
  a.button { display: inline-flex; align-items: center; min-height: var(--ctl-h);
             padding: 7px 14px; border: 1px solid var(--line); border-radius: var(--r-ctl);
             background: var(--panel); color: var(--ink); text-decoration: none; font-weight: 500; }
  a.button:hover { border-color: var(--muted); }
  a.button.primary { background: var(--act); border-color: var(--act); color: var(--act-ink);
                     font-weight: 600; }
  /* The not-found page (V7): the mark large and quiet over one sentence and two ways on. */
  main.lost { padding: 32px 0 8px; }
  .lostMark { color: var(--muted); margin: 0 0 18px; }
  .lostMark .mark { display: block; }

  /* A page of prose (V7): a reading measure, the body at 16/24, each section under a
     heading with the page's rule above it. */
  main.prose { max-width: 42rem; }
  main.prose.wide { max-width: 54rem; }
  main.prose p { margin: 0 0 14px; }
  main.prose ul { margin: 0 0 14px; padding-left: 18px; }
  main.prose li { margin: 0 0 7px; }
  main.prose h2 { padding-top: 18px; border-top: 1px solid var(--line); }
  /* Credits, by artist: a heading per group and its files under it. */
  .artist { margin: 20px 0 4px; font-size: 14px; }
  .artist .licence { font-weight: 400; }
  ul.art { list-style: none; padding: 0; margin: 0; font-size: 13px; color: var(--muted);
           columns: 15rem 3; column-gap: 24px; }
  ul.art li { margin: 0 0 2px; break-inside: avoid; }
  /* The terms (Q722): each point in plain words with its legal form beside it, two
     columns on a desk and the legal form under the plain words on a phone. */
  .point { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 28px; }
  .point h2 { grid-column: 1 / -1; }
  .point .legal { font-size: 13px; line-height: 1.55; color: var(--muted); }
  .point .legal p { margin: 0 0 10px; }
  .point .says { font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
                 color: var(--muted); margin: 0 0 4px; }
  @media (max-width: 760px) { .point { grid-template-columns: 1fr; } }
  p.dedication { border-left: 3px solid var(--act); padding: 2px 0 2px 12px; margin: 0 0 18px;
                 font-style: italic; }

  /* The meta-game's dated lines and *still open*, under the table. Each line carries its
     own rule sentence, which is the profile's answer to §3's seventh test. */
  main h3 { font-size: 13px; font-weight: 600; color: var(--muted); margin: 15px 0 3px; }
  ul.lines { list-style: none; padding: 0; margin: 0 0 6px; }
  ul.lines li { margin: 0 0 10px; }
  ul.lines .rule { display: block; font-size: 12px; color: var(--muted); }
  ul.open { list-style: none; padding: 0; margin: 0; font-size: 13px; }
  ul.open li { margin: 0 0 4px; }
  span.mark { color: var(--muted); }

  /* The overall page. Four columns, so it fits a phone without hiding anything. The
     table keeps its own width rather than the page's, so three small numbers are not
     spread across a desk, and the counts are right-aligned under their headings. */
  main th.num { text-align: right; }
  table.people { max-width: 30rem; }
  table.people.empty { display: none; }

  /* Q760, finding a member: one large search field, then everybody as a sheet each, two
     or three across on a desk and one on a phone. Each sheet is wholly the link to the
     member's page, so the target is the whole sheet, never a small word; the handle is
     large, and the rest is in the body size rather than the small print, for older eyes. */
  label.find { max-width: 30rem; }
  label.find input { min-height: 46px; font-size: 17px; padding: 8px 12px; }
  ul.directory { list-style: none; padding: 0; margin: 0 0 8px; display: grid; gap: 10px;
                 grid-template-columns: repeat(auto-fill, minmax(15.5rem, 1fr)); }
  ul.directory li { margin: 0; }
  ul.directory a.person { display: grid; grid-template-columns: 40px 1fr; gap: 2px 12px;
                          align-items: center; min-height: 64px; padding: 10px 14px;
                          background: var(--paper) 0 0 / 180px, var(--panel);
                          border: 1px solid var(--line); border-radius: var(--r-panel);
                          box-shadow: var(--shadow); color: var(--ink); text-decoration: none; }
  ul.directory a.person:hover, ul.directory a.person:focus-visible { border-color: var(--muted); }
  ul.directory .personMark { grid-row: 1 / span 3; align-self: start; }
  ul.directory .personName { font-size: 18px; line-height: 24px; font-weight: 600;
                             overflow-wrap: anywhere; }
  ul.directory .personName .you { font-size: 14px; font-weight: 400; color: var(--muted); }
  ul.directory .personSince, ul.directory .personMade { font-size: 14px; line-height: 20px;
                                                       color: var(--muted); }
  p.found { font-size: 15px; margin: 0 0 12px; }
  /* A member's page (Q760): the sections' lines a little larger than the small print. */
  main.member ul.lines li { font-size: 15px; line-height: 22px; }
  main.member select { max-width: 100%; }

  /* The profile's record, narrow. The last three columns -- the best opponent beaten,
     the rungs and the distance to the top -- are columns on a desk and lines under the
     game's name below 760px, so eight columns never have to fit a phone. */
  .narrow { display: none; }
  @media (max-width: 760px) {
    .narrow { display: block; font-size: 12px; color: var(--muted); }
    table.record th.best, table.record td.best,
    table.record th.beat, table.record td.beat,
    table.record th.dist, table.record td.dist { display: none; }
    main th, main td { padding-right: 7px; }
  }
"""

STYLE = APP_STYLE + _OWN

# The pages' scripts: plain files under `site/static/`, served at `SCRIPT_PREFIX` by the
# site's handler ahead of dispatch, as the application serves its own (Q513). A page lists
# its files in load order; they are classic scripts sharing one global scope, so the three
# preludes -- `post.js` (post JSON, and put the error where the person is looking),
# `game-link.js` and `play-link.js` (where a game and a game in progress live on the
# application's page) -- are declared once and read by the page's own file after them.
# Nothing a page's script needs from Python is spliced into it: such a value is handed
# over as data, one JSON tag the script reads (`_shell`'s `data`).
SCRIPT_PREFIX = "/static/site/"
SCRIPT_DIR = Path(__file__).resolve().parent / "static"
SCRIPTS: dict[str, Path] = {
    SCRIPT_PREFIX + name + ".js": SCRIPT_DIR / f"{name}.js"
    for name in (
        "post", "game-link", "play-link",
        "join", "signin", "profile", "players", "suggestions", "admin", "challenge",
        "member", "ask",
    )
}


TAGLINE = shell.TAGLINE  # the application's header line, one spelling for both

# What each milestone is called on the page, and how its line joins the game it was
# completed on. Box A2 of `docs/archive/plan-meta-game.md`. The keys are the kinds
# `/api/profile` sends, which `site/milestones.py` names; a test holds the two sets
# together, and a kind with no entry here draws under its own key rather than
# disappearing.
#
# The preposition is the difference between a milestone on a game -- *Both seats, on
# courier* -- and one across the library, where the game is only where the last piece of
# it landed.
_MILESTONE_LINES: dict[str, tuple[str, str]] = {
    "both-seats": ("Both sides", "on"),
    "whole-ladder": ("Every computer player beaten", "on"),
    "first-here": ("The first game played here", "on"),
    "every-game": ("The tour", "completed on"),
    "every-kind": ("One of every kind", "completed on"),
    "first-person": ("Against a person", "on"),
    "person-both-seats": ("Both sides against people", "completed on"),
    # D1 to D4, the designer's lines: each is the first of its kind, on the game it was.
    "first-draft": ("First draft kept", "on"),
    "passes-screen": ("Passes the playability check", "on"),
    "played-by-others": ("Played by somebody else", "on"),
    "forked": ("Built on by somebody else", "on"),
}

# How many untried games one *still open* line names before it says how many are left.
# The line is there so a person can pick one, not so the library can be listed twice.
_SHOWN = 4

# The same for the overall page's two unclaimed lists, which have one line each rather
# than one per grid bin and so have room for a few more names.
_SHOWN_GAMES = 6


def _foot(here: str = "", players: bool = False) -> str:
    """The application's footer -- what's new, credits and privacy -- minus the page you
    are on.

    These used to be blue underlined links at the foot of the page's own `<nav>`, in a
    different place and a different case from the app's (D1). The way back to the library
    is the header's name, as it is on every other page.

    *What's new* is a route inside the application rather than a page of its own, so it is
    an address into it and is never the page being read here.

    *Suggestions* is in the same row on every page, signed in or not. It needs an
    account, so a visitor who follows it is sent to the door -- the answer a visitor
    asking for a profile already gets -- rather than being shown a link that comes and
    goes with the reader.

    *Players* is linked only once the page lists `milestones.LISTED_FROM` people (L3 of
    `docs/archive/plan-launch.md` §3.3), which is the site's fact, not the reader's: `players` is
    what the server read for the pages a signed-in person gets. A constant page (credits,
    privacy, the badges, the door) reads no store, and so never links it; the page itself
    answers by its address whatever the count.
    """
    links = (
        ("/game", "games"),
        ("/", "about"),
        ("/#/whats-new", "what's new"),
        *((("/players", "players"),) if players else ()),
        ("/suggestions", "suggestions"),
        ("/credits", "credits"),
        ("/privacy", "privacy"),
        (terms.PATH, "terms"),
        (terms.CONTACT, "contact"),
    )
    return shell.footer(
        " · ".join(f'<a href="{where}">{text}</a>' for where, text in links if where != here)
    )


def _shell(
    title: str,
    body: str,
    scripts: tuple[str, ...] = (),
    kind: str = "",
    corner: str = "",
    here: str = "",
    data: dict[str, Any] | None = None,
    players: bool = False,
    *,
    description: str,
    path: str | None = None,
) -> str:
    """One page, in the application's shell.

    Every page says what it is in a description (G4 of `docs/plan-growth.md`). A page with
    an address of its own (`path`) names it as its canonical, on the site's address
    (`seo.site_base`), and says `noindex` when `robots.txt` keeps crawlers off it
    (`seo.disallowed`); a page without one (a token's, the not-found page) says `noindex`.

    The header and the footer are the app's, drawn here in the markup its own stylesheet
    expects -- `chess_variant_sandbox/web/page/style.py`, imported above rather than answered with a
    second palette. The app fills the corner from `/api/me` after it boots; these pages
    are rendered by the server and are handed the corner instead, which is empty on the
    three a visitor can be reading (`routes.PERSONAL` keeps the cookie off credits and
    privacy, so those two cannot know who is asking without asking).

    A page with no script gets no `<script>` tag at all, and no page gets a script body:
    `scripts` names files under `site/static/`, loaded in order, and `data` is what the
    server hands them, as JSON in a tag that never runs. `<` is written `\\u003c` there,
    so no value can close the tag.
    """
    tail = "".join(f'<script src="{SCRIPT_PREFIX}{name}.js"></script>\n' for name in scripts)
    head = front.head_lines(
        description,
        None if path is None else seo.site_base() + path,
        index=path is not None and not seo.disallowed(path),
    )
    if data is not None:
        held = json.dumps(data).replace("<", "\\u003c")
        tail = f'<script type="application/json" id="page-data">{held}</script>\n' + tail
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n{head}{shell.head_links()}<style>{STYLE}</style>\n"
        "</head>\n"
        '<body>\n<div class="wrap">\n'
        '  <header class="top">\n'
        f'    <h1 class="brand"><a href="/">{shell.brand()}</a></h1>\n'
        f'    <span class="muted tagline">{TAGLINE}</span>\n'
        '    <span class="grow"></span>\n'
        # Q760: *People* for whoever is signed in, which is every page with a corner.
        f"    {shell.sections('/', people=bool(corner))}\n"
        f'    <span class="muted small" id="who">{corner}</span>\n'
        "  </header>\n"
        f'<main class="{kind}">\n{body}\n</main>\n'
        f"{_foot(here, players)}\n</div>\n{tail}</body>\n</html>\n"
    )


# What each page says in a search result (G4): one sentence, under 160 characters, never the
# same for two pages (`tests/test_site_doors.py` holds both).
NOT_FOUND_SAYS = "There is nothing at this address on Chess variant sandbox."
MAIL_OFF_SAYS = "Switch off the emails that say it is your move in a correspondence game."
JOIN_SAYS = (
    "Join Chess variant sandbox with the invitation you were sent: choose a handle and a "
    "password, and the account is yours."
)
SIGNIN_SAYS = (
    "Sign in to Chess variant sandbox to play members by correspondence, keep a rating on "
    "each game and publish the games you design."
)
ASK_SAYS = (
    "Accounts on Chess variant sandbox are by invitation: leave your email address, and the "
    "person who runs the site can send you one."
)
CHALLENGE_SAYS = "A challenge to a game by correspondence on Chess variant sandbox."
PROFILE_SAYS = "Your own page on Chess variant sandbox: your games, your ratings and your badges."
MEMBER_SAYS = "{handle}'s page on Chess variant sandbox: the games they made and played."
PLAYERS_SAYS = "Find the members of Chess variant sandbox by name, and challenge one to a game."
SUGGESTIONS_SAYS = (
    "Write down an idea for a rule or for Chess variant sandbox, and read the ones you sent."
)
ADMIN_SAYS = "The maintenance page of Chess variant sandbox."
CREDITS_SAYS = (
    "Who drew the pieces on Chess variant sandbox, under which free licence, and the "
    "typefaces the site is set in."
)
BADGES_SAYS = (
    "Every badge on Chess variant sandbox and the one rule that earns it: the same list "
    "for everybody."
)
PRIVACY_SAYS = (
    "What Chess variant sandbox keeps about you and why: an account, your games and "
    "ratings, and what leaving does."
)
TERMS_SAYS = (
    "The terms of Chess variant sandbox: a game published here is for everybody, the site "
    "claims no rights in it, and what it does not promise."
)


LINK = re.compile(r"^/c/[A-Za-z0-9_-]{1,64}$")
"""A challenge link's address (`site/challenges.py`), the one place sign-in returns to."""


MAIL_OFF = re.compile(r"^/mail/off/([A-Za-z0-9_-]{16,64})$")
"""The one-click *switch them off* address in every move email (Q572,
`site/move_email.py`): the person's own token, no sign-in."""


def mail_off(said: str) -> str:
    """What `/mail/off/<token>` says once it has done its one thing (Q572).

    A constant page but for the sentence, which is `move_email.OFF_SAID` or
    `OFF_UNKNOWN`: the box was switched off by opening the link, and there is nothing to
    press. A mail program that opens it ahead of the person does the safe thing.
    """
    body = (
        "<h1>Move emails</h1>\n"
        f'<p class="lead">{escape(said)}</p>\n'
        '<p class="under"><a href="/profile">Your page</a> has the box, for switching '
        "them on or off.</p>"
    )
    return _shell("Move emails", body, kind="prose", description=MAIL_OFF_SAYS)


def not_found() -> str:
    """What an address the site has no page for answers, with a 404 (V7).

    A constant page: it reads no cookie and names nothing about the address asked for,
    so a member asking for `/admin` gets these bytes exactly as an invented path does and
    cannot tell the page is there (task 1.8). A route under `/api/` keeps its JSON answer,
    which is what a script reads.
    """
    body = (
        f'<div class="lostMark">{shell.mark(64)}</div>\n'
        "<h1>No page here</h1>\n"
        '<p class="lead">There is nothing at this address. It may have been mistyped, or '
        "the page may have moved.</p>\n"
        '<div class="row"><a class="button primary" href="/game">See the games</a>'
        '<a class="button" href="/#/whats-new">What\'s new</a></div>'
    )
    return _shell("No page here", body, kind="lost", description=NOT_FOUND_SAYS)


def _who(handle: str, user: int | None, linked: bool = False) -> str:
    """The header's corner on a signed-in page: the reader's mark (R2) and handle, on
    every page but their own one a button to it (Q731: a bare name read as a label saying
    who is signed in, not as a way somewhere). `_member_button` is the one markup; the
    application's header builds the same element in `core.js`'s `paintIdentity`."""
    mark = "" if user is None else marks.svg(user, 20)
    if linked:
        return _member_button(handle, mark)
    return f"{mark}{escape(handle)}"


def _member_button(handle: str, mark: str = "") -> str:
    """The header's way to your own page: your mark and handle, then *your page* and a
    chevron, in one outlined button (`style.py`'s `.memberButton`). Its name says both
    what it is and who is signed in; on a phone the handle goes and *your page* stays."""
    named = escape(handle)
    return (
        f'<a class="memberButton" href="/profile" aria-label="Your page, signed in as {named}">'
        f'{mark}<span class="memberName">{named}</span>'
        '<span class="memberGo">your page<span aria-hidden="true"> ›</span></span></a>'
    )


def _your_mark(user: int | None) -> str:
    """Your page's heading carries your mark large, beside your handle (R2)."""
    return "" if user is None else marks.svg(user, 40)


def _then(asked: str | None) -> str | None:
    """Where sign-in or joining goes afterwards: a challenge link's address, or nowhere.

    Only a challenge link (Q571, `docs/archive/plan-return.md` §1.4): a link opened by somebody
    who then signs in or joins works for them, and the site has no other return to offer.
    Anything else is dropped, so the parameter is never a way off the site.
    """
    if asked is None or not LINK.match(asked):
        return None
    return asked


PURPOSE = (
    "<b>Chess variant sandbox is a place to find chess-like games worth playing.</b> People "
    "invent games here, in a designer that needs no programming, and anyone can play them "
    "in the browser, against the computer or against each other. What no other site does "
    "is measure them: the computer plays every game against itself hundreds of times, at "
    "several strengths, and scores how interesting the play is &mdash; whether skill pays, "
    "whether the first player has it easy, whether games go somewhere or stall. That score "
    "is the best estimate available today, not a verdict: it says what it rests on and how "
    "far to trust it, and it is measured again as the computer players get stronger and as "
    "people play. The people who play here are the test the computer cannot replace."
)
"""`docs/archive/plan-understanding.md` §1.1's statement of purpose, as written (tone A), for the
invitation page. The invitation email carries its shorter form (`mail.PURPOSE`)."""


ACCOUNT_ADDS = (
    "With an account your games count: you get a rating on each game, you can beat the "
    "strongest computer player on a game and be named for it, you can save and publish "
    "games you design, play other members by correspondence, and star and vote."
)
"""What an account adds (L7, `docs/archive/plan-launch.md` §4, with `docs/archive/plan-understanding.md`
§1.3's *strongest computer player on a game* for *the top*): in full on the invitation page
and the sign-in page. The application's header says it in one line beside *sign in*
(`core.js` `paintIdentity`). Every clause is a thing the site does today."""


def join(code: str, after: str | None = None) -> str:
    """Choose a handle and a password, against the invitation in the link.

    The code goes into a hidden field rather than into the script, so nothing from the
    URL is ever parsed as JavaScript. `after` is a challenge link to go to once the
    account exists (`_then`), handed to the script as data.
    """
    body = (
        "<h1>Welcome to Chess variant sandbox</h1>\n"
        '<p class="lead">You have been invited to the trial: a few people, before anybody '
        "else. Choose a handle and a password and the account is yours.</p>\n"
        f'<p class="under">{PURPOSE}</p>\n'
        f'<p class="under">{ACCOUNT_ADDS}</p>\n'
        f'<input type="hidden" id="code" value="{escape(code, quote=True)}">\n'
        '<div class="panel form">\n'
        '<label><span>handle</span><input id="handle" autocomplete="username" '
        'autocapitalize="off" spellcheck="false" autofocus></label>\n'
        '<p class="hint">3 to 24 characters: letters, digits, hyphen, underscore. '
        "Other people see it.</p>\n"
        '<label><span>password</span><input id="password" type="password" '
        'autocomplete="new-password"></label>\n'
        '<p class="hint">At least 10 characters. Nothing else is needed.</p>\n'
        '<div class="row"><button class="primary" id="go">create the account</button></div>\n'
        '<p class="note" id="note"></p>\n'
        "</div>\n"
        '<p class="small muted"><a href="/signin">I already have an account</a></p>'
    )
    scripts = ("post", "join")
    return _shell("Join Chess variant sandbox", body, scripts, data={"then": _then(after)},
                  description=JOIN_SAYS, path="/join")


def signin(after: str | None = None) -> str:
    """Come back with the password chosen at the join, and then to `after` if it is a
    challenge link (`_then`)."""
    body = (
        "<h1>Sign in</h1>\n"
        f'<p class="lead">{ACCOUNT_ADDS} Looking, and playing practice games, need no '
        "account.</p>\n"
        '<div class="panel form">\n'
        '<label><span>handle</span><input id="handle" autocomplete="username" '
        'autocapitalize="off" spellcheck="false" autofocus></label>\n'
        '<label><span>password</span><input id="password" type="password" '
        'autocomplete="current-password"></label>\n'
        '<div class="row"><button class="primary" id="go">sign in</button></div>\n'
        '<p class="note" id="note"></p>\n'
        "</div>\n"
        # The first-visit study's F5: a reader looked here for a way to make an account
        # and found none, and nothing said why. Q831: and now the way to ask for one.
        '<p class="small muted">Accounts are by invitation for now: an invitation\'s '
        "link is where you choose a handle and a password.</p>\n"
        f"{_ask_button()}"
    )
    scripts = ("post", "signin")
    return _shell(
        "Sign in to Chess variant sandbox", body, scripts, data={"then": _then(after)},
        description=SIGNIN_SAYS, path="/signin",
    )


def _ask_button() -> str:
    """The way to the request form, wherever a visitor meets the invitation wall (Q831)."""
    return (
        f'<div class="row"><a class="button" id="ask-invitation" href="{asking.PATH}">'
        f"{asking.BUTTON}</a></div>"
    )


def request_invitation() -> str:
    """Ask for an invitation (Q831, `site/asking.py`): an address, an optional note, send.

    A constant page, like the credits: no cookie read and no store, so a visitor's look at
    it costs nothing. The script posts the form to `/api/ask-invitation` and, on an
    answer, puts the one sentence the server sends where the form was. The note's hint is
    the user's own suggestion, that they say how they found the site.
    """
    body = (
        f"<h1>{asking.BUTTON}</h1>\n"
        '<p class="lead">Accounts here are by invitation. Leave your email address and the '
        "person who runs the site can send you one. Looking at the games, and playing "
        "practice games, need no account.</p>\n"
        '<form class="panel form" id="ask-form">\n'
        '<label><span>email address</span><input id="ask-email" type="email" '
        'autocomplete="email" autocapitalize="off" spellcheck="false" '
        f'maxlength="{quota.ASK_EMAIL_LENGTH}" required autofocus></label>\n'
        '<p class="hint">Where the invitation would go. The site sends nothing else to '
        "it.</p>\n"
        '<label><span>a note, if you like</span><textarea id="ask-note" rows="3" '
        f'maxlength="{quota.ASK_NOTE_LENGTH}"></textarea></label>\n'
        f'<p class="hint">{asking.NOTE_HINT}</p>\n'
        '<div class="row"><button class="primary" id="ask-send" type="submit">'
        "send the request</button></div>\n"
        '<p class="note" id="note"></p>\n'
        "</form>\n"
        '<p class="small muted"><a href="/signin">I already have an account</a> · '
        '<a href="/privacy">what the site keeps</a></p>'
    )
    return _shell(asking.BUTTON, body, ("post", "ask"), description=ASK_SAYS, path=asking.PATH)


def challenge_link(token: str) -> str:
    """A challenge link, `/c/<token>` (Q571, s8 of the mock).

    A constant page: the token goes to the script as data and the script asks
    `/api/challenge/link/<token>` whose challenge it is. A member is offered *Accept and
    play*; somebody without an account is told whose challenge it is, that accounts are by
    invitation, and the way to sign in, which comes back here. The link is never an
    invitation.
    """
    body = (
        "<h1>A challenge</h1>\n"
        '<div id="challenge"><p class="muted">Loading the challenge…</p></div>\n'
        '<p class="note" id="note"></p>'
    )
    return _shell(
        "A challenge", body, ("post", "game-link", "play-link", "challenge"),
        data={"token": token}, description=CHALLENGE_SAYS,
    )


def profile(handle: str, players: bool = False, *, user: int | None = None) -> str:
    """The three counts, this person's record on every rated game, and the meta-game.

    The handle is the one string here that somebody chose, and it is escaped. Everything
    else is drawn from `GET /api/profile` through text nodes, so a game named after a
    script tag is a game with a silly name and nothing else.

    Box A2 of `docs/archive/plan-meta-game.md` adds three things. The table gains *n of k
    rungs* (M4) and *distance to the top* (T3); under it come the dated milestone lines,
    newest first, and *still open for you* grouped by the behaviour grid's axes. Each part
    is absent when there is nothing to say, because an empty heading would read as a
    nought and `docs/archive/plan-meta-game.md` §2.1's third rule is that absent is not zero.

    Above the table sits *games you left running* (`docs/archive/plan-ux.md` B2), which is
    the only part of the page about something unfinished. Each line links to the board,
    and a line whose rules the store will not hand back draws no link.
    """
    body = (
        f"<h1>{_your_mark(user)}{escape(handle)}</h1>\n"
        '<p class="lead">What you have played, and how many of each game\'s computer '
        "players you have beaten, weakest to strongest.</p>\n"
        '<div class="counts" id="counts"></div>\n'
        # L2 of `docs/archive/plan-launch.md` §3.2: in the counts' place until the first is
        # non-zero, what a person can do here rather than three noughts.
        '<div id="start"></div>\n'
        # B2, above the table: the one part of this page that is an unfinished thing
        # rather than a record of finished ones.
        '<div id="carry"></div>\n'
        # Q572, s6 of the mock: *Correspondence*. Q570's games between people, running,
        # and the line of what this person finished; Q571's challenges for you and sent
        # by you, and the games you are open to; and the one email's box.
        '<div id="correspondence"><h2>Correspondence</h2>\n'
        '<div id="between"></div>\n'
        '<div id="challenges"></div>\n'
        '<div id="move-email"></div>\n'
        # Q606 (R10): the maker's one switch, drawn only for somebody with a published
        # game of their own.
        '<div id="takes-challengers"></div></div>\n'
        # Q719: *Invite somebody*, drawn only for a person `/admin` gave invitations to.
        '<div id="invite"></div>\n'
        # Q596: your page as other members see it, and its one box (R1).
        '<div id="your-page"></div>\n'
        # Q503b: the games this person made, grouped by level, each with its next step.
        '<div id="mine"></div>\n'
        # Q503a: the games a save had listed, back to Draft, with the reason.
        '<div id="drafts"></div>\n'
        '<table class="record" id="record"><thead><tr><th>game</th><th>played</th><th>won</th>'
        '<th>drawn</th><th>lost</th><th class="best">best beaten</th>'
        '<th class="beat">computer players beaten</th><th class="dist">your rating</th>'
        "</tr></thead>\n"
        '<tbody id="games"></tbody></table>\n'
        '<p class="under" id="record-under"><em>computer players beaten</em> is how many of a game\'s '
        "computer players you have beaten at least once, of how many it has. If one is "
        "added later, there is a new one to beat. <em>your rating</em> is your rating on "
        "that game, then how far it sits from the game's strongest computer player, and "
        "it is blank until you have played a rated game there.</p>\n"
        '<div id="done"></div>\n'
        # R9b (Q605): this person's own solved marks on proven positions, by game.
        '<div id="proven"></div>\n'
        '<div id="open"></div>\n'
        # Always here, badges or none. A rule shown only to whoever has met it is a
        # hidden rule (§3, test 7), so the way to the list cannot depend on the reader.
        '<p class="under">Badges are dated records of things people do here. '
        '<a href="/badges">Every badge and what earns it</a> is on one page, the same '
        "for everybody.</p>\n"
        '<p class="note" id="note"></p>\n'
        '<div class="row"><button id="out">sign out</button></div>'
    )
    scripts = ("post", "game-link", "play-link", "profile")
    data = {"LINES": _MILESTONE_LINES, "SHOWN": _SHOWN}
    # The corner names the reader, as the application's header does. Here it is text
    # rather than a link, because the page it would link to is this one.
    return _shell(
        f"{handle} on Chess variant sandbox", body, scripts, kind="wide", corner=_who(handle, user),
        data=data, players=players, description=PROFILE_SAYS, path="/profile",
    )


# ------------------------------------------------------------------ a member's page

# The milestones' rules as a member's page says them: about the person, not to the reader.
# Every rule in `milestones.RULES` that addresses *you* does so as *you designed*.
def _theirs(rule: str) -> str:
    return rule.replace("you designed", "they designed")


def _member_link(handle: str, user: int) -> str:
    """A member's name, with their mark, linked to their page (R1)."""
    drawn = marks.svg(user, 18)
    return f'<a href="/players/{escape(handle, quote=True)}">{drawn}{escape(handle)}</a>'


def _game_link(digest: str, name: str) -> str:
    return f'<a href="/game/{escape(digest, quote=True)}">{escape(name or "an unnamed game")}</a>'


def _day(when: Any) -> str:
    return f"{when.day} {when:%b %Y}" if hasattr(when, "day") else escape(str(when)[:10])


_RESULT_WORDS = {"won": "Beat", "lost": "Lost to", "drew": "Drew with"}


def _against_html(against: dict[str, Any]) -> str:
    """Who sat across: a computer player's name, a member linked to their page, or *a
    former player*, who has no page."""
    if "computer" in against:
        return escape(str(against["computer"]))
    if against.get("id") is not None:
        return _member_link(str(against["person"]), int(against["id"]))
    return escape(str(against.get("person", "")))


def _played_html(game: dict[str, Any]) -> str:
    """One line of *Games ‹handle› played lately* (Q760): the result, who sat across
    (linked to their page when they have one), the game, and what kind of game it was."""
    link = _game_link(str(game["game"]), str(game.get("name", "")))
    kind = "correspondence" if game.get("between") else "rated"
    if game.get("between") and game.get("rated"):
        kind = "correspondence, rated"
    return (
        f"{_RESULT_WORDS[str(game['result'])]} "
        f"{_against_html(game['against'])} at {link}"
        f' <span class="mark">· {kind} · {_day(game["when"])}</span>'
    )


def _made_html(game: dict[str, Any]) -> str:
    """One line of *Games ‹handle› made* (Q760): the game, its level, the day it was
    published, and how many other people have tried it (a count, never who)."""
    parts = [escape(str(game["level"]))]
    if game.get("published") is not None:
        parts.append(f"published {_day(game['published'])}")
    if game.get("tried"):
        parts.append(
            f'{game["tried"]} {"person has" if game["tried"] == 1 else "people have"} tried it'
        )
    return (
        f'{_game_link(str(game["game"]), str(game["name"]))} '
        f'<span class="mark">· {" · ".join(parts)}</span>'
    )


def _section(heading: str, inner: str, lead: str = "") -> str:
    """One of the page's sections, or nothing: a section with nothing in it is absent
    (test 5; `docs/plan-launch.md` R5)."""
    if not inner:
        return ""
    under = f'<p class="under">{lead}</p>\n' if lead else ""
    return f"<h2>{escape(heading)}</h2>\n{under}{inner}\n"


def _items(lines: list[str]) -> str:
    if not lines:
        return ""
    return '<ul class="lines">\n' + "".join(f"<li>{line}</li>\n" for line in lines) + "</ul>"


def _options(games: list[dict[str, Any]]) -> str:
    return "".join(
        f'<option value="{escape(str(g["game"]), quote=True)}">'
        f'{escape(str(g["name"]) or "an unnamed game")}</option>'
        for g in games
    )


def _challenge_html(handle: str, challenge: dict[str, Any] | None) -> str:
    """*Challenge ‹handle›* (§6.2's header, Q571's challenge by handle), for another
    member. A sheet the page's script opens: a game, a side, and send. The games they made
    come first, under their own heading (Q760), then the rated games. Nothing counts down
    and nothing is sent to anybody but the challenge itself (tests 2, 3)."""
    if challenge is None:
        return ""
    name = escape(handle)
    if challenge.get("paused"):
        return f'<p class="under">{escape(str(challenge.get("pausedSays", "")))}</p>\n'
    theirs = challenge.get("theirs") or []
    games = challenge.get("games") or []
    if not games and not theirs:
        return (
            f'<p class="under">To challenge {name}, open a game and use <em>Play a '
            "person</em> on its page.</p>\n"
        )
    if theirs and games:
        options = (
            f'<optgroup label="Games {escape(handle, quote=True)} made">{_options(theirs)}'
            f'</optgroup><optgroup label="Rated games">{_options(games)}</optgroup>'
        )
    else:
        options = _options(theirs or games)
    return (
        '<div class="row"><button class="primary" id="challenge-open">'
        f"Challenge {name}</button></div>\n"
        '<div class="panel form" id="challenge-sheet" hidden>\n'
        f'<label><span>game</span><select id="challenge-game">{options}</select></label>\n'
        '<label><span>you play</span><select id="challenge-seat">'
        '<option value="either">either, drawn when it starts</option>'
        '<option value="first">moving first</option>'
        '<option value="second">moving second</option></select></label>\n'
        '<p class="hint">By correspondence: you move whenever you like and the game waits '
        f"for you. The challenge waits until {name} accepts or declines it, and nothing is "
        f"sent to {name} but the challenge, which they find on their page. For a game "
        "not listed here, use <em>Play a person</em> on its page.</p>\n"
        '<div class="row"><button class="primary" id="challenge-send">send the '
        'challenge</button><button id="challenge-cancel">cancel</button></div>\n'
        "</div>\n"
    )


DONE_LEAD = (
    "Milestones and badges, newest first: the only achievements the site keeps, and "
    'nobody is scored against anybody else. What earns each badge is on <a href="/badges">'
    "the badges page</a>, the same for everybody."
)
"""What *What ‹handle› has done* says it holds, and what the site does not keep (Q760)."""


def _not_yet(handle: str, reading: dict[str, Any]) -> str:
    """Q760: what this page has nothing of yet, said plainly in one paragraph rather than
    as empty headings or noughts (test 5: absent is not zero, and nothing here says
    *last*). Only the three sections every member could have; the opt-in ones are not
    named, so the page does not say whether somebody ticked a box."""
    name = escape(handle)
    said = []
    if not reading.get("designed"):
        said.append(f"{name} has not published a game yet.")
    if not reading.get("played"):
        said.append(
            "No finished rated game or game against a person yet; practice games against "
            "the computer are not listed."
        )
    if not reading.get("milestones") and not reading.get("badges"):
        said.append("No milestones or badges yet.")
    if not said:
        return ""
    return f'<p class="under" id="not-yet">{" ".join(said)}</p>\n'


def member(
    reading: dict[str, Any], reader: str, players: bool = False, *, viewer: int | None = None
) -> str:
    """`/players/<handle>`, a member's page (R1, `docs/archive/plan-return.md` §6.2; Q760).

    Drawn whole by the server from `members.read`, every string escaped. Only a member
    reads it (a visitor is sent to the door before this runs), and every member reads the
    same page about a person, the person included; their own copy adds one line pointing
    at their own page, where the settings are, and drops *Challenge*.

    Q760's order: the header with *Challenge*, the games they made (newest first), the
    games they played lately, what they have done, then the two opt-in or derived
    sections; and one plain paragraph naming what there is nothing of yet.
    """
    handle = str(reading["handle"])
    user = int(reading["user"])
    name = escape(handle)
    people = reading.get("playsWith") or {}
    header = (
        f"<h1>{marks.svg(user, 40)}{name}</h1>\n"
        f'<p class="lead">Member since {escape(str(reading["since"]))}.'
        + (f" {escape(str(reading['finished']))}" if reading.get("finished") else "")
        + "</p>\n"
    )
    if reading.get("own"):
        header += (
            '<p class="under">This is your page as other members see it. Visitors '
            'cannot see it. <a href="/profile">Your own page</a> has your settings, your '
            "games in progress and your challenges.</p>\n"
        )
    # The note sits under the button, so the word that a challenge went is where the
    # eye already is, not at the foot of a long page.
    header += _challenge_html(handle, reading.get("challenge")) + '<p class="note" id="note"></p>\n'
    designed = _items([_made_html(game) for game in reading.get("designed", [])])
    played = _items([_played_html(game) for game in reading.get("played", [])])
    done_lines: list[tuple[str, str]] = []
    for line in reading.get("milestones", []):
        label, join = _MILESTONE_LINES.get(str(line["kind"]), (str(line["kind"]), "on"))
        done_lines.append((
            str(line["when"]),
            f"<b>{escape(label)}</b>, {escape(join)} "
            f'{_game_link(str(line["game"]), str(line["name"]))}, {_day(line["when"])}.'
            f'<span class="rule">{escape(_theirs(str(line["rule"])))}</span>',
        ))
    for badge in reading.get("badges", []):
        on = (
            f", on {_game_link(str(badge['game']), str(badge.get('name_of', '')))}"
            if badge.get("game") else ""
        )
        done_lines.append((
            str(badge.get("when") or ""),
            f"<b>{escape(str(badge['name']))}</b>{on}"
            + (f", {escape(str(badge['when'])[:10])}" if badge.get("when") else "")
            + ".",
        ))
    done_lines.sort(key=lambda pair: pair[0], reverse=True)
    done = _items([line for _, line in done_lines])
    favourites = _items([
        _game_link(str(game["game"]), str(game["name"]))
        for game in reading.get("favourites", [])
    ])
    with_people = [
        _member_link(str(person["handle"]), int(person["id"]))
        for person in people.get("people", [])
    ]
    with_lines = []
    if with_people:
        with_lines.append("People: " + ", ".join(with_people) + ".")
    if people.get("computers"):
        with_lines.append(
            "Computer players: " + ", ".join(escape(str(c)) for c in people["computers"]) + "."
        )
    body = (
        header
        + _section(
            f"Games {handle} made",
            designed,
            "Published games, newest first. A draft is never shown.",
        )
        + _section(
            f"Games {handle} played lately",
            played,
            "Finished rated games and games against a person, newest first, the last "
            "eight at most. Practice games against the computer are not listed.",
        )
        + _section(f"What {handle} has done", done, DONE_LEAD)
        + _section(
            "Favourite games", favourites, f"The games {name} gave five stars, shown by their choice."
        )
        + _section(
            "Plays most with",
            _items(with_lines),
            "The people they have finished most games with, and the computer players "
            "they have finished most rated games against, most first.",
        )
        + _not_yet(handle, reading)
        + '<p class="under"><a href="/players">Find another member</a></p>'
    )
    return _shell(
        f"{handle} on Chess variant sandbox",
        body,
        ("post", "member") if reading.get("challenge") else (),
        kind="wide member",
        corner=_who(reader, viewer, linked=True),
        data={"handle": handle} if reading.get("challenge") else None,
        players=players,
        description=MEMBER_SAYS.format(handle=handle),
        path=routes.MEMBERS + handle,
    )


# ------------------------------------------------------------------ the overall page


def players(handle: str, *, user: int | None = None, asked: str = "") -> str:
    """S3, the overall page. Box A3 of `docs/archive/plan-meta-game.md`.

    The user decided its shape on 2026-09-14, option A: it lists only people who have
    finished a game, and it opens with what is unclaimed. Both halves are why it passes
    §3's fifth test. A person who has done nothing here is on no list rather than last on
    one, and the first thing anybody reads is a set of games to go and take rather than a
    table of people ahead of them.

    Nothing on the page is a position. The rows are ordered strongest first, by tops
    beaten and then wins and then games tried, and no number is drawn beside a name. The
    three columns count games and never matches, so playing one game a hundred times
    moves no column past one (§3, test 6).

    The handle in the corner is escaped; everything from `GET /api/players` is a text
    node, so a name full of tags is a name full of tags. Somebody who has left reads as
    *a former player*, which is the handle the store hands back for them.

    Q760 puts *Find a member* first (the user, 2026-10-05: *"It should not be so hard to
    find other players"*): a search field over every member, A to Z, each the way to their
    page, where *Challenge* is. It holds no number, so it ranks nobody and leaves the
    decision of 2026-09-14 (nothing ranked leads the page) standing. `asked` is the
    address's `?q=`, filled into the field so a search can be linked to.
    """
    body = (
        "<h1>Players</h1>\n"
        '<p class="lead">Find another member and open their page: the games they made and '
        "played, their badges, and a button to challenge them. Then what nobody has claimed "
        "yet, and who has played.</p>\n"
        '<h2 id="find-head">Find a member</h2>\n'
        '<label class="find"><span>search by name, or by a game they made</span>'
        '<input type="search" id="find" autocomplete="off" autocapitalize="off" '
        'spellcheck="false" enterkeyhint="search" aria-controls="directory" '
        f'value="{escape(asked[:40], quote=True)}"></label>\n'
        '<p class="found" id="found" aria-live="polite"></p>\n'
        '<ul class="directory" id="directory" aria-labelledby="find-head"></ul>\n'
        '<p class="under" id="find-said"></p>\n'
        "<h2>Still unclaimed</h2>\n"
        '<p class="under" id="open-lead"></p>\n'
        '<ul class="lines" id="unclaimed"></ul>\n'
        '<h2 id="people-head">Who has played</h2>\n'
        '<table class="people" id="people-table"><thead><tr><th>player</th>'
        '<th class="num">tried</th>'
        '<th class="num">won</th><th class="num">beat the strongest</th></tr></thead>\n'
        '<tbody id="people"></tbody></table>\n'
        '<p class="under" id="people-lead"></p>\n'
        '<p class="under" id="people-rule">Nobody here is given a place, and nothing here '
        "counts matches. "
        "The list is ordered by the games where each person beat the strongest computer "
        "player, then games won, then games tried. Somebody who has "
        "not finished a game is on no list rather than at the bottom of this one, and "
        "the list grows when the library does.</p>\n"
        '<p class="note" id="note"></p>'
    )
    scripts = ("post", "game-link", "players")
    data = {"SHOWN": _SHOWN_GAMES}
    # The corner is the way back to your own record, as it is on the maintainer's page.
    return _shell(
        "Players on Chess variant sandbox", body, scripts, kind="wide",
        corner=_who(handle, user, linked=True), here="/players", data=data,
        description=PLAYERS_SAYS, path="/players",
    )


# ----------------------------------------------------------------- the suggestion box


def suggestions(handle: str, players: bool = False, *, user: int | None = None) -> str:
    """Somewhere to write down an idea for a rule or an idea for the site. Q279.

    The user's inbox, 2026-09-17: *"Add a place to add ideas of game-rules or other
    website ideas that could be added."*

    The form is the page and the person's own suggestions are under it. Nobody else's are
    here: the maintainer reads the whole box on `/admin`, and a page that showed
    everybody's would be a decision about whether the box is public, which is the user's
    to make.

    Every suggestion is drawn with `textContent`, this person's own included, so a
    suggestion that is nothing but a script tag reads as a script tag and runs as
    nothing. The kinds and the limits come from the server rather than being spelt here,
    so the form cannot offer a kind the column refuses or a length the store does.
    """
    body = (
        "<h1>Suggestions</h1>\n"
        '<p class="lead">An idea for a rule, or an idea for the site. Write it here and '
        "the person who runs the site reads it.</p>\n"
        '<label><span>what kind of idea</span>\n'
        '<select id="kind"></select></label>\n'
        '<label><span>the idea</span>\n'
        '<textarea id="body" rows="6" autofocus></textarea></label>\n'
        '<p class="hint" id="limits"></p>\n'
        '<div class="row"><button class="primary" id="go">post it</button></div>\n'
        '<p class="note" id="note"></p>\n'
        "<h2>What you have suggested</h2>\n"
        '<p class="under">Only you and the person who runs the site read these. Nothing '
        "here is answered automatically and no mail is sent; <em>done</em> means they "
        "have marked it so.</p>\n"
        '<ul class="lines" id="mine"></ul>\n'
        '<p class="under" id="empty"></p>'
    )
    scripts = ("post", "suggestions")
    return _shell(
        "Suggestions on Chess variant sandbox",
        body,
        scripts,
        corner=_who(handle, user, linked=True),
        here="/suggestions",
        players=players,
        description=SUGGESTIONS_SAYS,
        path="/suggestions",
    )


# --------------------------------------------------------------- the maintainer's page


_USE_TABLE = (
    '<div class="scroller"><table class="use"><thead><tr><th>route</th><th>requests</th>'
    "<th>refused</th><th>by visitors</th><th>CPU s</th><th>mean wall s</th>"
    "<th>slowest s</th><th>most people in an hour</th><th>people-hours</th></tr></thead>\n"
    '<tbody id="{id}"></tbody></table></div>\n'
)


def admin(handle: str, players: bool = False, *, user: int | None = None) -> str:
    """Users, invitations, games, reports, suggestions, the kill switches and the numbers.

    Task 1.8, with Q503d's reports.

    Reachable only by a `users.role` of `admin`, which a maintainer sets with `chess-variant-sandbox
    site admin <handle>`; everybody else gets the 404 an invented path gets, so the page
    does not announce itself to a member who cannot use it.

    Every cell here is a text node. Handles, game names and email addresses are the three
    user-supplied strings on the page and all three arrive through `textContent`, so the
    file keeps `pages.py`'s rule of having no `innerHTML` in it.

    The two destructive acts ask first. Killing is reversible and asks once; forgetting
    is not reversible at all, and its question names the person.
    """
    body = (
        "<h1>Maintenance</h1>\n"
        f'<p class="lead">Signed in as {escape(handle)}. Everything on this page takes '
        "effect at once.</p>\n"
        "<h2>Sessions</h2>\n"
        # First thing under the heading, and the sentence beneath it: an admin who has
        # opened this page on a phone is usually here for this one button (D3).
        '<div class="row"><button id="revoke-all" class="danger">sign everybody out</button></div>\n'
        '<p class="note" id="note"></p>\n'
        '<p class="under">Signing everybody out leaves every account intact; they come '
        "back with the password they already have.</p>\n"
        # Q813. The box's disk and the queues, each against the age that makes it a
        # problem; the same lines `deploy/status.sh` prints from the maintainer's machine.
        "<h2>Status</h2>\n"
        '<p class="under" id="status-said"></p>\n'
        '<p class="under" id="status-disk"></p>\n'
        '<div class="scroller"><table><thead><tr><th>queue</th><th>waiting</th>'
        "<th>oldest</th><th>a problem past</th><th>why that limit</th></tr></thead>\n"
        '<tbody id="status-queues"></tbody></table></div>\n'
        '<p class="under">A queue reads as a problem when its oldest has waited past the '
        "limit; depth alone only means slower.</p>\n"
        "<h2>People</h2>\n"
        '<p class="under">A killed account cannot sign in and its open sessions stop '
        "working at once. Forgetting keeps the privacy notice's promise and cannot be "
        "undone. The address and the password go, and the games stay. <em>can invite</em> "
        "is how many invitations a person may still send themselves, from their own page; "
        "it is 0 until you set it, and each one they send takes one off.</p>\n"
        '<div class="scroller"><table><thead><tr><th>handle</th><th>email</th><th>role</th>'
        "<th>state</th><th>sessions</th><th>invites</th><th>can invite</th><th>matches</th>"
        "<th></th>"
        '</tr></thead>\n<tbody id="people"></tbody></table></div>\n'
        "<h2>Invitations</h2>\n"
        '<p class="under">An invitation mails a link to one address, and the link makes '
        "one account. Inviting an address again mails a fresh link and leaves the earlier "
        "one working. An address that already has an account is refused. The codes "
        "themselves are not printed here.</p>\n"
        '<form class="invite" id="invite-form">\n'
        '<label><span>email address</span><input id="invite-email" type="email" '
        'autocomplete="off" autocapitalize="off" spellcheck="false" required></label>\n'
        '<button id="invite-send" type="submit">send the invitation</button>\n'
        "</form>\n"
        '<p class="note" id="invite-note"></p>\n'
        '<div class="scroller"><table><thead><tr><th>address</th><th>sent</th><th>from</th>'
        "<th>expires</th><th>accepted</th></tr></thead>\n"
        '<tbody id="invites"></tbody></table></div>\n'
        # Q831. What visitors asked for from the sign-in page, oldest first.
        "<h2>Requests for an invitation</h2>\n"
        '<p class="under">Visitors ask from the sign-in page, with an address and, if '
        "they like, a note. <em>send an invitation</em> sends the ordinary invitation, from "
        "you; <em>decline</em> sends nothing. Either way the person is told nothing but "
        "the invitation, if you send one. A request from an address that already has an "
        "account, or that already asked, is not kept, and the person is not told so. An "
        f"answered request is deleted after {quota.ASK_KEPT.days} days, and the same "
        "address cannot ask again before then.</p>\n"
        '<p class="note" id="asks-note"></p>\n'
        '<div class="scroller"><table><thead><tr><th>address</th><th>asked</th>'
        "<th>note</th><th>invited before</th><th></th></tr></thead>\n"
        '<tbody id="asks"></tbody></table></div>\n'
        '<p class="under" id="asks-empty"></p>\n'
        "<h2>Games</h2>\n"
        '<p class="under">A killed game is off the listing, off its own page and cannot '
        "be started. The matches already played on it stay and still count. A "
        "<em>retired</em> game is one whose rules have left the corpus. The last deploy "
        "took it off the listing and it takes no new game, while its page, its row and "
        "its matches are read as before; putting its rules back in the tree is what "
        "brings it back. A <em>deleted</em> game is one whose creator or an admin "
        "deleted its rules for good, from the game's own page. There is no way back and "
        "no button here, and the matches, ratings and votes on it stay where they are "
        "and name it a deleted game.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>origin</th>'
        "<th>matches</th><th>state</th><th></th></tr></thead>\n"
        '<tbody id="games"></tbody></table></div>\n'
        # Q649. The scores a new measurement replaced, from each game's `history.jsonl`:
        # the maintainer's *was* lines, with the basis and the reason. Visitors see none.
        "<h2>Earlier scores</h2>\n"
        '<p class="under">Each score a new measurement replaced, newest first: what it '
        "was, when it was measured and replaced, what it was measured with, and why it "
        "was replaced. The current score is the one on the game's page; these are kept "
        "under <code>runs/&lt;hash&gt;/previous/</code> and are never shown as a score. "
        "Visitors do not see them.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>was</th>'
        "<th>measured</th><th>replaced</th><th>measured with</th><th>why</th>"
        "<th>convergence</th></tr></thead>\n"
        '<tbody id="score-history"></tbody></table></div>\n'
        '<p class="under" id="score-history-empty"></p>\n'
        # EV6 (Q690). How much each stored score has moved between its revisions at one
        # weights version, beside its interval (`history.volatility`). Visitors see none.
        "<h2>How much scores have moved</h2>\n"
        '<p class="under">For each game, how far its stored score has moved across its '
        "last few re-measurements under the current scoring rules, beside the interval "
        "of the score it has now. No line is drawn between steady and unsteady: compare "
        "the two numbers. This changes no score and no trust level. Visitors do not see "
        "it.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>scores compared</th>'
        "<th>moved</th><th>largest change</th><th>interval</th><th>in words</th>"
        "</tr></thead>\n"
        '<tbody id="score-volatility"></tbody></table></div>\n'
        '<p class="under" id="score-volatility-empty"></p>\n'
        # H8 and H9 of `docs/reference/human-data.md` (§3.3): games whose strongest computer
        # player people beat, against the bar fixed before any game was read, and the
        # training queue a game over it enters. Visitors see none of it.
        "<h2>People beating the strongest</h2>\n"
        '<p class="under" id="bar-said"></p>\n'
        '<p class="under">Counted over rated games against the computer players that are '
        "the strongest on the game now. A game over the bar is queued for a stronger "
        "computer player by itself, after every entry the strength-premise check made; the "
        "player it gets is accepted by games between computers only, and no person's game "
        "reaches a score. Visitors do not see this.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>over the bar</th>'
        "<th>the counts</th><th>training</th></tr></thead>\n"
        '<tbody id="bar"></tbody></table></div>\n'
        '<p class="under" id="bar-empty"></p>\n'
        "<h3>The training queue</h3>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>asked by</th>'
        "<th>state</th><th>queued</th></tr></thead>\n"
        '<tbody id="training-queue"></tbody></table></div>\n'
        '<p class="under" id="training-queue-empty"></p>\n'
        # H7 (§3.2): each computer player's place on people's scale, per game.
        "<h2>Computer players on people's scale</h2>\n"
        '<p class="under">Each computer player\'s rating on a game\'s table, where '
        "people's rated results have moved it from the computer-only ladder it was seeded "
        "from, and how many of the people with an established rating there it is above. "
        "A game's page shows these lines under <em>more</em> once five people are "
        "established on it. It moves no score.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>established</th>'
        "<th>computer player</th><th>rating</th><th>seeded at</th><th>games with people"
        "</th><th>above</th></tr></thead>\n"
        '<tbody id="people-scale"></tbody></table></div>\n'
        '<p class="under" id="people-scale-empty"></p>\n'
        # Q705. The refresher's status file (`web/refreshing.py`): what it is doing, the
        # game, when it last wrote, what it refused and why. Visitors see none of it.
        "<h2>The refresher</h2>\n"
        '<p class="under">The refresher keeps the library\'s scores up to date by itself, '
        "one game at a time, from its own copy of the code "
        "(<code>chess-variant-sandbox refresh --library --watch</code>), and once no score "
        "is owed it rates the games' computer players, one ladder at a time. A game or "
        "ladder it refused "
        "is not tried again until its measurement changes or "
        "<code>refresh --forget</code> clears it. Visitors do not see this.</p>\n"
        '<ul id="refresher-said"></ul>\n'
        '<div class="scroller"><table><thead><tr><th>refused game</th><th>repair</th>'
        "<th>why</th><th>what it said</th><th>when</th></tr></thead>\n"
        '<tbody id="refresher-refused"></tbody></table></div>\n'
        '<p class="under" id="refresher-refused-empty"></p>\n'
        # Q570. Games between people (`docs/archive/plan-return.md` §1.7): the pause on new ones,
        # how many are running, and the maintainer's own way to seat two members; members
        # find each other by challenges (Q571).
        "<h2>Games between people</h2>\n"
        '<p class="under" id="between-said"></p>\n'
        '<div class="row"><button id="between-switch"></button></div>\n'
        '<p class="under">Pausing refuses new games between people and leaves every '
        "running game to carry on and finish. Seating two members starts a game between "
        "them on a game's rules, the first handle moving first; it is rated where the "
        "game has rated computer players and practice elsewhere.</p>\n"
        '<form class="invite" id="pair-form">\n'
        '<label><span>game (hash)</span><input id="pair-hash" autocomplete="off" '
        'spellcheck="false" required></label>\n'
        '<label><span>moves first</span><input id="pair-seat0" autocomplete="off" '
        'spellcheck="false" required></label>\n'
        '<label><span>moves second</span><input id="pair-seat1" autocomplete="off" '
        'spellcheck="false" required></label>\n'
        '<button id="pair-send" type="submit">seat them</button>\n'
        "</form>\n"
        '<p class="note" id="between-note"></p>\n'
        # Q503h. The latest round: each entrant with what the decision needs, and the
        # confirm and veto, each with its reason, on an entrant the members passed.
        "<h2>Round</h2>\n"
        '<p class="under" id="round-said"></p>\n'
        '<p class="under">Members vote for a week from the 1st; a game passes with more '
        "yes than no and at least four yes. Confirm or veto each game that passed, with a "
        "reason either way: it is shown on the game's page and in What's new. At most four "
        "are confirmed a round, in the order listed; a passed game past them is carried to "
        "the next round with its vote. Confirming records the decision and shows the "
        "<code>archive</code> command to run on your own machine, with the game's rules "
        "from the box in <code>runs/</code>; the game joins the library when that commit is "
        "deployed and <code>site load</code> finds it.</p>\n"
        '<div class="scroller"><table class="round" id="round-table"><thead><tr>'
        "<th>game</th><th>score</th>"
        "<th>premise</th><th>library cost</th><th>vote</th><th>decision</th><th></th>"
        '</tr></thead>\n<tbody id="round-entrants"></tbody></table></div>\n'
        '<p class="note" id="round-note"></p>\n'
        # Q503d. What people reported from a game's page, newest first, each with the
        # game's own kill switch beside it. A report hides nothing by itself.
        "<h2>Reports</h2>\n"
        '<p class="under">Games people reported from the game\'s page, newest first. A '
        "report takes nothing off the site by itself; the button beside it is the same "
        "kill switch as in the games table above. Each person can report a game once, and "
        "dismissing a report takes it off this list but keeps that: the same person "
        "still cannot report the game again.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>reason</th>'
        "<th>note</th><th>from</th><th>when</th><th></th></tr></thead>\n"
        '<tbody id="reports"></tbody></table></div>\n'
        '<p class="under" id="reports-empty"></p>\n'
        # Q517: what a press on a report row did, said beside the rows rather than in the
        # note at the top of the page, a long scroll away from the button.
        '<p class="note" id="reports-note"></p>\n'
        # Q518. The dismissed ones, folded away under the open list, so a dismissal can
        # be looked back on. Read only: a dismissed report's row is what keeps its
        # reporter from filing the same game again.
        '<details id="dismissed-reports"><summary id="dismissed-summary">Dismissed '
        "reports</summary>\n"
        '<p class="under">Reports taken off the list above, most recently dismissed '
        "first. Each row stays, so its reporter still cannot report the game again.</p>\n"
        '<div class="scroller"><table><thead><tr><th>game</th><th>reason</th>'
        "<th>note</th><th>from</th><th>reported</th><th>dismissed</th></tr></thead>\n"
        '<tbody id="dismissed"></tbody></table></div>\n'
        '<p class="under" id="dismissed-said"></p></details>\n'
        # Q730. The accounts a spot check flagged (Q503e, `site/spotcheck.py`), newest
        # flag first, each with what its check found and the button that clears it.
        "<h2>Flagged accounts</h2>\n"
        '<p class="under">An account is flagged when a score sent from its computer did '
        "not match the site's own replay of one of its games. A flagged account cannot "
        "publish, and its scores are not spot-checked until the flag is cleared. "
        "<em>found</em> names the game, seed, move and both builds, which is what tells a "
        "bug in the site's code from a forged score. Clearing lets the account publish "
        "again; the mismatched score stays on record.</p>\n"
        '<div class="scroller"><table><thead><tr><th>handle</th><th>flagged</th>'
        "<th>found</th><th></th></tr></thead>\n"
        '<tbody id="flagged"></tbody></table></div>\n'
        '<p class="under" id="flagged-empty"></p>\n'
        '<p class="note" id="flagged-note"></p>\n'
        "<h2>Suggestions</h2>\n"
        '<p class="under">What people have written in the suggestion box, open ones '
        "first. Only you and whoever posted one can read them. Nothing here is answered "
        "automatically and no mail was sent; <em>done</em> is your own mark and can be "
        "put back. Deleting one removes the text for good and tells nobody.</p>\n"
        '<div class="scroller"><table><thead><tr><th>kind</th><th>from</th><th>when</th>'
        "<th>suggestion</th><th>state</th><th></th></tr></thead>\n"
        '<tbody id="suggestions"></tbody></table></div>\n'
        '<p class="under" id="suggestions-empty"></p>\n'
        "<h2>What the site limits</h2>\n"
        '<p class="under">These are all the limits the site sets. They are read from '
        "<code>site/quota.py</code>, the only place they are written down.</p>\n"
        '<div class="scroller"><table><thead><tr><th>limit</th><th>value</th></tr></thead>\n'
        '<tbody id="limits"></tbody></table></div>\n'
        "<h2>Use</h2>\n"
        '<p class="under">Counted by the hour from the site\'s own tables. A visitor is '
        "counted as requests and never as a person. People-hours count a person once per "
        "hour they were active. A partial hour is shown apart, after a slash.</p>\n"
        '<p class="under" id="use-kept"></p>\n'
        # Q721. Who came, what they played, who joined and what mail went, a row a day.
        "<h3>By day, the last fortnight</h3>\n"
        '<p class="under">UTC days, newest first. <em>practice</em> is games visitors '
        "started; <em>with members</em> is games with a signed-in person in a seat; "
        "<em>players</em> is how many different people sat in one. <em>mail refused</em> "
        "is invitations the mail service would not take.</p>\n"
        '<div class="scroller"><table class="use"><thead><tr><th>day</th>'
        "<th>requests</th><th>by visitors</th><th>server errors</th><th>practice</th>"
        "<th>with members</th><th>players</th><th>joined</th><th>invited</th>"
        "<th>mail refused</th><th>move emails</th><th>failed</th></tr></thead>\n"
        '<tbody id="days"></tbody></table></div>\n'
        '<ul id="days-said"></ul>\n'
        "<h3>The last day, by route</h3>\n"
        '<p class="under" id="use-day-said"></p>\n'
        + _USE_TABLE.format(id="use-day")
        + "<h3>The last month, by route</h3>\n"
        '<p class="under" id="use-month-said"></p>\n'
        + _USE_TABLE.format(id="use-month")
        + "<h3>Show me, the last month</h3>\n"
        '<ul id="use-show-me"></ul>\n'
        "<h3>The box, the last seven days</h3>\n"
        '<ul id="use-box"></ul>\n'
        "<h3>What these numbers cannot say</h3>\n"
        '<ul id="use-cannot"></ul>'
    )
    scripts = ("post", "game-link", "admin")
    # Only an admin is served this page, so the corner names the reader and links to the
    # profile. The Maintenance link the application's header carries would point here.
    return _shell(
        "Chess variant sandbox maintenance",
        body,
        scripts,
        kind="sheet",
        corner=_who(handle, user, linked=True),
        players=players,
        description=ADMIN_SAYS,
        path="/admin",
    )


# ------------------------------------------------------- what the site owes the reader


@lru_cache(maxsize=1)
def _artists() -> list[str]:
    """An attribution line for every piece drawing the site ships.

    The whole library rather than the icons one game happens to wear. A reader meets the
    drawings across the corpus and again in the designer, which can dress a piece in any
    of them, so a page built from the games loaded today would fall behind the first game
    that reached for a new one. `viz.pieces.credits_for` is the one place a file is
    joined to its artist, and it wants icons, so the shipped files are handed to it as
    the stems they are. A checkout that never fetched the library has no files and gets
    an empty list, which the page says in a sentence.
    """
    stems = sorted(path.stem for path in pieces.ASSET_ROOT.glob("svg/*.svg"))
    return pieces.credits_for([[{"stem": stem}] for stem in stems])


# `credits_for` writes one line per file, `<stem> - <artist> (<licence>)`, and the page
# wants the same lines under a heading each (D4). The shape is that one function's own,
# so reading it back here keeps the licence text where it is written rather than joining
# a second reader to the manifest. A line that does not match is kept whole, which is
# what the no-manifest fallback sentence is.
def _grouped(lines: list[str]) -> tuple[list[tuple[str, str, list[str]]], list[str]]:
    """(artist, licence, files) groups, and whatever did not parse.

    Most files come from a handful of people and the tail is derivative chains that name
    everybody in them, so the groups are ordered by how many files each drew. That puts
    the six artists who drew most of the library at the top and the one-file chains,
    whose names are a paragraph each, at the bottom.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    loose: list[str] = []
    for line in lines:
        stem, _, rest = line.partition(" - ")
        artist, bracket, licence = rest.rpartition(" (")
        if not rest or not bracket or not licence.endswith(")"):
            loose.append(line)
            continue
        groups.setdefault((artist, licence[:-1]), []).append(stem)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    return ([(artist, licence, sorted(files)) for (artist, licence), files in ordered], loose)


def credits() -> str:
    """Who drew the pieces. Task 1.9, and what the CC BY and CC BY-SA files require."""
    groups, loose = _grouped(_artists())
    if groups or loose:
        art = (
            "<p>The drawings are grouped by the person who made them, with the licence "
            "that person's files carry beside the name. A piece the library has no "
            "drawing for is drawn as its letter instead, so an invented piece needs no "
            "artwork to be played.</p>\n"
            + "".join(
                f'<h2 class="artist">{escape(artist)} '
                f'<span class="licence muted">{escape(licence)}</span></h2>\n'
                '<ul class="art">\n'
                + "\n".join(f"<li>{escape(stem)}</li>" for stem in files)
                + "\n</ul>\n"
                for artist, licence, files in groups
            )
            + "".join(f"<p>{escape(line)}</p>\n" for line in loose)
            + "<p>The full record, with a link to every original file, is kept with the "
            "site's source.</p>\n"
        )
    else:
        art = "<p>Every piece is drawn as its letter.</p>\n"
    faces = "".join(
        f"<li>{escape(entry['family'])}, for {escape(entry['role'])} "
        f'<span class="muted">{escape(entry["copyright"])}</span></li>\n'
        for entry in shell.fonts().values()
    )
    body = (
        "<h1>Credits</h1>\n"
        '<p class="lead">The pieces on every board are drawings from Wikimedia Commons, '
        "redistributed unmodified under their own free licences.</p>\n"
        f"{art}"
        "<h2>Typefaces</h2>\n"
        "<p>The site's type is set in three families under the SIL Open Font License "
        "1.1, served from this site rather than from a font service:</p>\n"
        f"<ul>\n{faces}</ul>\n"
    )
    return _shell("Credits", body, kind="prose wide", here="/credits",
                  description=CREDITS_SAYS, path="/credits")


def badges() -> str:
    """Every badge and the one sentence saying what earns it. Box E4, against §2.9.

    Box E4 and §2.9 are in `docs/archive/plan-meta-game.md`. An unearned badge is drawn
    on nobody's profile, so a profile alone would keep every rule a person has not met
    out of sight, and the seventh test (`docs/reference/no-dark-patterns.md`) refuses a
    hidden rule.
    This page is the answer. It reads the same for everybody: nothing here is marked
    earned, nothing shows how close anybody is, and no badge is counted.

    Every sentence comes from `site/badges.BADGES` and from nowhere else, which is what
    holds this list and the line beside a held badge to one text.
    """
    rows = "".join(
        "<li>\n"
        f"<b>{escape(badge.name)}</b>\n"
        f"<span class=\"rule\">{escape(badge.rule)}</span>\n"
        + (
            f'<span class="rule">In steps, at '
            f"{escape(_listed(badge.steps, badge.kind))}.</span>\n"
            if badge.steps
            else ""
        )
        + "</li>\n"
        for badge in site_badges.BADGES.values()
    )
    body = (
        "<h1>The badges</h1>\n"
        '<p class="lead">A badge is a dated record of something somebody did here. It is '
        "written once and kept, so it still reads true after the numbers behind it have "
        "moved.</p>\n"
        "<p>This is all of them, with what earns each one. Nothing on this page says who "
        "holds what or how far along anybody is, and no page on the site counts badges "
        "or ranks people by them. A badge you have earned appears on your profile and on "
        "the page other members see of you, and a badge about a game appears on that "
        "game's page.</p>\n"
        f'<ul class="lines">\n{rows}</ul>\n'
        "<p class=\"under\">Only rated games count &mdash; a practice game, a game "
        f"somebody resigned and a game resigned after {quota.IDLE_SAID} count for none "
        "of these. Nothing here expires and nothing is taken back.</p>\n"
    )
    return _shell("The badges on Chess variant sandbox", body, kind="prose", here="/badges",
                  description=BADGES_SAYS, path="/badges")


def _listed(steps: tuple[int, ...], kind: str) -> str:
    """A series' steps as a sentence says them, with the unit the badge counts in.

    A share reads as a percentage and a count reads as a bare number, which is the
    difference between *50%, 75% and 100%* and *1, 5, 10 and 25*. `Badge.thresholds` is
    what says which: a badge whose steps are shares stores them under `shares`.
    """
    mark = "%" if "shares" in site_badges.BADGES[kind].thresholds else ""
    said = [f"{step}{mark}" for step in steps]
    return ", ".join(said[:-1]) + " and " + said[-1] if len(said) > 1 else said[0]


def privacy() -> str:
    """What is kept about a person, what a sandbox game keeps, and what leaving does.

    Task 1.9, with the badge of box E3 (`docs/archive/plan-meta-game.md`) and Q279's
    suggestion added. Seven things are named because the store holds seven about a person:
    `site/identity.py`'s three, the match rows, the dated award rows of
    `docs/archive/plan-meta-game.md` §2.9, anything they wrote in the suggestion box, and
    their solved marks on proven positions (R9b, `0037_proven_positions.sql`). Leaving
    is the retention answer of 2026-09-11 in the words it asked for -- your games stay,
    your name goes.
    """
    body = (
        "<h1>Privacy</h1>\n"
        '<p class="lead">The site is small and invite-only, and it keeps about you only '
        "what an account and a rating need.</p>\n"
        "<h2>What the site keeps</h2>\n"
        "<p>It keeps seven things and nothing else.</p>\n"
        "<ul>\n"
        "<li>Your email address &mdash; the invitation you joined by was a link sent to "
        "it. If you tick the box on your own page that asks for it, the site also sends "
        "one email there each time it becomes your move in a game against another "
        "person, saying whose move, in which game, with a link to it; the box is off "
        "unless you tick it, and every such email has a link that switches it off with "
        "one click, as unticking the box does. The site keeps when each was sent, for "
        "its daily limit. No other mail goes there, ever, and the address is passed to "
        "nobody.</li>\n"
        "<li>Your password as a scrypt hash &mdash; the password itself is never stored, "
        "and a hash cannot be turned back into the password it was made from. Signing in "
        "hashes what you type and compares the two.</li>\n"
        "<li>Your session as a hash of the cookie &mdash; your browser holds a random "
        "string and the site stores its SHA-256, so a copy of the database is a list of "
        "digests rather than a drawer of working sessions. Signing out deletes the "
        "row.</li>\n"
        "<li>Every match you play while signed in &mdash; the moves, the result, who or "
        "what played each side, and the rating the game moved. Each game's ratings are "
        "worked out from these records, which is why they are kept. Against a computer "
        "player it also keeps how long you took over each move and which moves a "
        "take-back undid, and in practice the ratings each side had on that game when it "
        "started; no page shows these, and they are read with the game's moves, never "
        "added up per person.</li>\n"
        "<li>A dated record of each badge you earn &mdash; which badge it was, when the "
        "game that earned it finished, and the figures its sentence quotes. A badge is "
        "written once and kept, so it still reads true after the numbers behind it have "
        "moved, and nothing counts how many anybody holds.</li>\n"
        "<li>Anything you post in the suggestion box &mdash; the text you wrote, which of "
        "the two kinds you picked, and the day you posted it. Your suggestions are read "
        "by you and by the person who runs the site and by nobody else, and no other page "
        "on the site shows them. They can mark one done or delete it; deleting it removes "
        "the text for good.</li>\n"
        # R9b (Q605): a person's solved marks on proven positions, `proven_marks`.
        "<li>Which proven positions you have solved &mdash; for each, the game, the "
        "position, whether you converted it or found the move, and when. Only your own "
        "page shows them, nothing counts them across people, and a try that did not "
        "succeed is recorded only as the game it was.</li>\n"
        "</ul>\n"
        "<p>Nothing else is asked of you when you join, your page shows other members "
        "only what the next part says, and the session cookie is the only cookie the "
        "site sets.</p>\n"
        "<h2>What other members see</h2>\n"
        "<p>Signed-in members, and nobody else, can open your page: your handle and your "
        "mark, the month you joined, how many of your games against other people you "
        "finished, your last few finished rated games and games against people with "
        "their results and opponents, your firsts, the games you designed that are "
        "published (never a draft), your milestones and badges, and the people and "
        "computer players you have finished most games with. Visitors are sent to the "
        "sign-in page instead. Nothing records when you visit, so no page says when you "
        "were last here.</p>\n"
        "<p>The players page lists every member, A to Z, so members can find each other "
        "by name: your mark, your handle, the month you joined and the names of up to "
        "three of your published games, each the way to your page. It shows members "
        "nothing that your page does not, puts nobody in an order but the alphabet's, "
        "and visitors cannot see it.</p>\n"
        "<p>Your votes are private. A game's page shows how many people voted, the "
        "average, and how many votes chose each of the fixed tags, never who. The games "
        "you gave five stars appear on your page only if you tick the box for it on your "
        "own page. A vote is stars and up to three fixed tags, with no text, so nothing "
        "you write reaches a game's designer.</p>\n"
        "<p>When you publish a game you designed, you give it to the public domain, as the "
        f'<a href="{terms.PATH}">terms</a> say, and the site records on the game that '
        "you agreed and which version of the terms you agreed to. That is a tick and a "
        "number, with the time; nothing you type is kept with it. Games published before "
        "the terms existed carry no such record.</p>\n"
        "<p>If you designed a published game, you may switch on, on your own page, being "
        "listed on your games as taking challengers. It is off unless you switch it on. "
        "Then members, and nobody else, see your handle first under Play a person on each "
        "of your published games, and can challenge you there. A challenge holds no "
        "message.</p>\n"
        # Q831: a visitor's request for an invitation, which is not one of the six: the
        # person asking has no account, and the row is deleted once it is answered.
        "<h2>Asking for an invitation</h2>\n"
        "<p>If you ask for an invitation, the site keeps the email address and the note "
        "you wrote, and when, for the person who runs the site to read; nobody else sees "
        "them. The address is used for one thing, the invitation, if they send you one, "
        "and no other mail goes to it. A request is deleted "
        f"{quota.ASK_KEPT.days} days after it is answered.</p>\n"
        f"{_ask_button()}\n"
        "<h2>A game played without signing in</h2>\n"
        "<p>A practice game keeps no person at all. The moves are stored so that the tab "
        "can be reopened, with how long each of your moves took and which moves a "
        "take-back undid, and neither side names anybody. A game that reaches its end "
        "is also kept, move by move with its result, after a take-back or a restart, so "
        "that the ways people beat the computer players can be studied; it too names "
        "nobody. A practice game counts for "
        "nothing and appears in no table, which is why it needs no account.</p>\n"
        "<h2>Leaving</h2>\n"
        "<p>Your games stay and your name goes. Ask the person who invited you, and your "
        "email address and your password hash are deleted and your name on every match "
        "becomes <em>a former player</em>. The matches themselves stay, because each "
        "game's ratings are worked out by replaying its matches from the first, and "
        "removing rows would move other people's ratings along with yours. Which proven "
        "positions you solved is deleted. Anything you "
        "posted in the suggestion box stays under the same name, because a suggestion is "
        "about the site rather than about you; ask and the person who runs the site will "
        "delete it.</p>\n"
        + _description_hint()
        + "<h2>What the site does not do</h2>\n"
        "<p>There are no analytics, no advertising and no third-party scripts, and the "
        "page loads nothing from another domain. The web server in front keeps an "
        "ordinary request log, which records the address a request came from, and only "
        "the person who runs the site reads it. The site itself keeps a running count of "
        "how it is "
        "used &mdash; for each hour, how many requests each kind of page and move "
        "received, how much of the machine's time they took, and how many different "
        "signed-in people made them. Those counts hold no name, no address and no "
        "cookie, so nothing in them can be traced to you, and they are what the site's "
        "limits are set from.</p>\n"
        "<h2>Backups</h2>\n"
        "<p>The database is copied once a night onto the same machine and the copies are "
        "kept for a fortnight. Something deleted today is gone from the site at once and "
        "out of the last copy two weeks later. The copies do not leave the "
        "machine.</p>\n"
        f'<p class="under">The <a href="{terms.PATH}">terms</a> say what publishing a game '
        "gives away and what the site does not promise.</p>\n"
    )
    return _shell("Privacy", body, kind="prose", here="/privacy",
                  description=PRIVACY_SAYS, path="/privacy")


def terms_page() -> str:
    """The site's terms (Q722): what publishing a game gives away, and what the site does
    not promise, each point in plain words with its legal wording beside it.

    A constant page, like credits and privacy: no script, nobody's name read, no store
    touched. Every word is `site/terms.py`'s, so the sentence the publish step shows and
    the version it records are the ones written here.
    """
    rows = "".join(
        f'<section class="point" id="{point.key}">\n'
        f"<h2>{escape(point.heading)}</h2>\n"
        f'<div class="plain"><p class="says">In plain words</p>\n<p>{escape(point.plain)}</p></div>\n'
        f'<div class="legal"><p class="says">The legal wording</p>\n<p>{escape(point.legal)}</p></div>\n'
        "</section>\n"
        for point in terms.points()
    )
    body = (
        "<h1>Terms</h1>\n"
        '<p class="lead">Games made here are for everybody to share and enjoy. The site '
        "keeps no rights in them and claims no credit for them.</p>\n"
        f"<p>Version {terms.VERSION}, {escape(terms.DATED)}. These terms apply to "
        "using the site and to every game published on it. Each point is said in plain "
        "words, with the wording that applies beside it. When you publish a game, you "
        "agree to this sentence:</p>\n"
        f'<p class="dedication">{escape(terms.DEDICATION)}</p>\n'
        f"{rows}"
        '<p class="under">What the site keeps about you is in the '
        '<a href="/privacy">privacy notice</a>. CC0 1.0\'s full text is at '
        f'<a href="{terms.CC0}">creativecommons.org</a>. The person who runs the site is '
        f'on GitHub at <a href="{terms.CONTACT}" rel="noopener">'
        f"{escape(terms.CONTACT_SAID)}</a>.</p>\n"
    )
    return _shell("Terms", body, kind="prose wide", here=terms.PATH,
                  description=TERMS_SAYS, path=terms.PATH)


def _description_hint() -> str:
    """What the optional description hint sends, said only while it is switched on (Q396).

    The module owns both the switch and the sentence, so the notice cannot say the hint
    is off while the site sends descriptions, or the other way round. Imported here, when
    the notice is drawn, rather than with this module.
    """
    from chess_variant_sandbox.web import jev_hint

    return jev_hint.privacy_paragraph()
