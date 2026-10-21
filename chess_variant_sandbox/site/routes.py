"""Which of the local application's routes a public site may register.

`web/server.py` dispatches by splitting a path and comparing the parts; this is that same
set of routes written down as data, each one marked served or withheld, and a withheld
route carries the sentence from §2 of `docs/plan-website.md` that says why.

The site matches against `ALLOWED` and `OWN` and nothing else, so a withheld route is not
a route: it gets the 404 an invented path gets, from the same message, before any handler
is reached. Caddy refuses the paid and repository-writing routes at the edge as well
(`deploy/remote/Caddyfile`), and that stays -- two doors that must both be open is the
point. This one is the door, and a route served here must not be one Caddy refuses.

A served route with a `visitor` sentence needs a signed-in person. A visitor gets that
sentence as a 400, the answer a vote or *show me* already gives, and the handler never
runs.

`ALLOWED` and `WITHHELD` together are the local application's whole table, which is what
the test compares. `OWN` is the short list of routes the site has and the local
application has not -- a vote, a profile -- and it is kept separate so that comparison
stays exact.

Adding a route to the local application means classifying it here; `tests/test_site.py`
reads `web/server.py` and fails on a route this file has never heard of. Default is
withheld, because the failure direction that matters is a repository-writing route
quietly becoming public.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    """One route, as a method and a path whose `<...>` segments are variables.

    `visitor` is empty for a route anybody may reach. Otherwise it is the sentence a
    visitor who is not signed in is refused with.
    """

    method: str
    path: str
    note: str
    visitor: str = ""

    @property
    def parts(self) -> tuple[str, ...]:
        return split(self.path)

    def matches(self, method: str, parts: Sequence[str]) -> bool:
        own = self.parts
        if method != self.method or len(own) != len(parts):
            return False
        return all(
            mine.startswith("<") or mine == theirs
            for mine, theirs in zip(own, parts, strict=True)
        )


def split(path: str) -> tuple[str, ...]:
    """A path as the parts the handlers compare against.

    Empty parts dropped, exactly as `App.get` and `App.post` do it, so `//api//run/` is
    matched as `api/run` here too. A matcher that only knew the tidy spelling would be a
    door standing open beside the one that is shut -- which is why the Caddyfile's own
    refusal is a regexp rather than a list of paths.
    """
    return tuple(part for part in path.strip("/").split("/") if part)


def find(method: str, path: str) -> Route | None:
    """The registered route this request is, or `None` for every other request."""
    parts = split(path)
    for route in ALLOWED + OWN:
        if route.matches(method, parts):
            return route
    return None


# What the site serves, and the page itself, which `web/server.py`'s handler answers ahead
# of any of these.
#
# The last three are the designer's cheap steps (Q221). Each needs a signed-in person, and
# the two that play games are capped by `site/screening.py` rather than by hope. Between
# the save and the check sits the optional description hint (Q396), which needs a
# signed-in person too and is capped by the day rather than by the person.
ALLOWED: tuple[Route, ...] = (
    Route(
        "GET",
        "/api/me",
        "signed in or not, the handle, the role -- nothing rated, and cheap (Q199)",
    ),
    Route("GET", "/api/library", "the listing, with the roster and the metric legend"),
    Route("GET", "/api/game/<hash>", "one game: rules, pieces, measurement, opponents"),
    Route(
        "GET",
        "/api/graph",
        "the library as a map: which games are alike, their communities, and where each "
        "sits. The "
        "handler draws the committed corpus alone, so no draft and no unlisted game is "
        "on it; a user's game joins when the site has a links table of its own "
        "(G7 of `docs/archive/plan-game-graph.md`)",
    ),
    Route(
        "GET",
        "/api/changelog",
        "`docs/changelog.md` as days of entries, for the *what's new* page. A tracked "
        "file the deploy already ships, read on each request, with the Baseline's moves "
        "that `site load` recorded filed in by date (Q503n); nobody's name, and the same "
        "answer for everybody",
    ),
    Route("GET", "/api/session/<id>", "the position a game is in"),
    Route(
        "POST",
        "/api/session",
        "seat a person against a rated agent; with `puzzle`, a practice game from one of "
        "the game's proven positions against hardest (R9b, `site/proven.py`)",
    ),
    Route(
        "POST",
        "/api/session/<id>/<action>",
        "move, agent, undo, restart, resign, draw; undo and restart are sandbox mode's "
        "alone, and draw is a draw offered to the computer, which the core judges (Q790); "
        "a game between two people (Q570) takes move, resign, draw, accept, decline and "
        "cancel from its own players only",
    ),
    Route("GET", "/api/design", "what a draft can start from, and the vocabulary"),
    Route("GET", "/api/design/<hash>", "an existing game as a starting point"),
    Route(
        "POST",
        "/api/design/check",
        "validate a draft; like every design POST, refused at 413 past "
        "quota.DESIGN_BODY_BYTES before the body is read (Q503a)",
    ),
    Route(
        "POST",
        "/api/design/load",
        "a rule file a person wrote, by hand or with their own AI, read and validated as "
        "the designer's next draft (Q830b); in memory, writes nothing, milliseconds. Its "
        "text is refused past designing.RULE_FILE_BYTES and RULE_FILE_DEPTH, and the body "
        "past quota.DESIGN_BODY_BYTES like every design POST. A valid file becomes an "
        "ordinary draft in the page, which only a save keeps",
    ),
    Route("POST", "/api/design/moves", "the move-map preview for one piece on one square"),
    Route(
        "POST",
        "/api/design/hex",
        "a draft whose cells were just made hexagons, with the square lattice's readings "
        "rewritten as their hex twins (Q640); in memory, writes nothing, milliseconds",
    ),
    Route(
        "POST",
        "/api/design/rules",
        "which absent rules this draft cannot take, and every reason for each in plain "
        "words; one validate per rule and about forty milliseconds for all of them",
    ),
    Route(
        "POST",
        "/api/design/turns",
        "the draft played another way -- one after the other, in rounds, or several moves "
        "a turn -- with every change that way needs and a sentence for each (Q995b); in "
        "memory, writes nothing, a few validates",
    ),
    Route(
        "POST",
        "/api/design/rest",
        "the draft with every piece resting the same number of turns after it moves, or "
        "none, written out on each piece; a piece the validator refuses a rest on is left "
        "without and said (Q995d); in memory, writes nothing, a few validates",
    ),
    Route(
        "POST",
        "/api/design/teach",
        "what a click on a square could mean: every way of moving the rule language has "
        "that reaches it, as sentences. A compile per candidate, capped at 64 and about "
        "a tenth of a second on chess",
    ),
    Route(
        "POST",
        "/api/design/steps",
        "the design steps that apply to the draft in hand, and one of them pressed. The "
        "draft is read in the request and nothing is written; a draw is seeded, so the "
        "same press on the same draft makes the same game. The optional `back` is the "
        "draft the last press was made on: a step whose draw only lands there again, "
        "which is an undo, is skipped and the next seed tried. About a twentieth of a "
        "second to list them, and a few milliseconds to make one",
    ),
    Route(
        "POST",
        "/api/design/play",
        "play a draft without keeping it; in memory, writes nothing",
    ),
    Route(
        "POST",
        "/api/design/save",
        "keep a draft in runs/designed/ and as an unlisted games row, recording whether "
        "it passed the playability check, which the save runs first and under a wall "
        "clock (Q276). A save lists nothing since Q503a; *Publish* does. Saving kept rules "
        "again gives the saver's own unpublished draft the new name and description, and "
        "nothing else (Q595, drafts.restatable). Capped at "
        "quota.DRAFTS_A_PERSON rows and quota.SAVES_A_DAY saves",
        visitor="Sign in to save a draft. Playing it without saving needs no account.",
    ),
    Route(
        "POST",
        "/api/design/hint",
        "optional (Q396): the sentences of a saved game's description its rules seem not "
        "to support, asked of TypeSafe's Jev only when CHESS_VARIANT_SANDBOX_JEV_KEY is set and under "
        "CHESS_VARIANT_SANDBOX_JEV_DAILY_CAP; one request per saved description, stored and not asked "
        "again. With no key it answers that it is off, and sends nothing",
        visitor="Sign in to see a hint about a saved game's description.",
    ),
    Route(
        "POST",
        "/api/design/screen",
        "the playability check: random playouts and then eight games between two small "
        "searches, one core, refused past site/screening.CHECK_SECONDS",
        visitor="Sign in to run the playability check. It plays games on the server.",
    ),
    Route(
        "GET",
        "/api/design/standing/<hash>",
        "the progress page: a draft's rungs, lineage and blockers, read from files",
        visitor="Sign in to see a draft's progress.",
    ),
)

# What the site has and the local application does not. These are not a classification of
# anything -- there is no local handler to allow or withhold -- so they are kept apart
# from `ALLOWED`, and `tests/test_site.py` reads the local application's table against
# that tuple alone.
#
# The first two need a signed-in person, because each writes or reads a row that names
# one. The three identity routes are how a person comes to be signed in, so they are the
# ones a visitor may reach; each answers with a cookie or takes one away.
OWN: tuple[Route, ...] = (
    Route(
        "GET",
        "/api/graph/people",
        "the site's own two numbers for every game the map draws: the star rating and "
        "how many leaderboard games people have finished on it, plus the listing's "
        "three filters. Anybody may reach it, as `/api/library` may be; it takes no "
        "hash from the request and answers about the nodes `/api/graph` drew, so no "
        "draft and no killed game is in it (`docs/archive/plan-game-graph.md` §8.3)",
    ),
    Route("POST", "/api/vote", "one to five stars on a game, replacing this person's last"),
    Route("GET", "/api/profile", "the three counts and this person's record on every rated game"),
    Route(
        "GET",
        "/api/players",
        "the overall page: every member A to Z for finding one by name (Q760: the mark, "
        "the handle, member since and a few of their published games), what nobody has "
        "claimed yet, and everybody who has finished "
        "a game beside their three counts, once five people have (below that, how many "
        "members there are and no rows)",
        visitor=(
            "Sign in to see the players. The library and every game's page are open to "
            "everybody."
        ),
    ),
    Route(
        "POST",
        "/api/suggest",
        "one suggestion: a short text and one of two kinds, rule or site. Read by the "
        "person who posted it and by an admin, and by nobody else",
        visitor="Sign in to post a suggestion. The box is invite-only, like the rest of the site.",
    ),
    Route(
        "GET",
        "/api/suggestions",
        "the suggestions this person has posted. An admin reads every suggestion on "
        "`/api/admin` instead, which is the one place the admin gate is",
        visitor="Sign in to see your suggestions.",
    ),
    Route("POST", "/api/join", "spend an invitation: a handle, a password, and a session"),
    Route("POST", "/api/signin", "a handle and a password back, for a session"),
    Route("POST", "/api/signout", "drop this session and the cookie with it"),
    # Q831, a visitor asks for an invitation (`site/asking.py`). For anybody: it is the
    # one write a stranger makes, capped by the connection and by the site's day.
    Route(
        "POST",
        "/api/ask-invitation",
        "a request for an invitation: an address and an optional note, kept for `/admin` "
        "to grant or decline. One answer whatever happened to it; nothing is mailed",
    ),
    Route(
        "POST",
        "/api/game/<hash>/delete",
        "delete a retired game's rules for good; its creator or an admin, confirmed in "
        "the browser and irreversible. The rows that reference it stay and name it a "
        "deleted game, so nobody's statistics move",
        visitor="Sign in to delete a game. Only the person who made it, or an admin, can.",
    ),
    Route(
        "POST",
        "/api/game/<hash>/report",
        "report a game: one of three reasons (spam, offensive, a copy) and an optional "
        "short note, filed as a row the maintainer reads on `/admin` beside the kill "
        "switch. One report a person a game, quota.REPORTS_A_DAY a day; it hides "
        "nothing by itself (`docs/plan-game-pipeline.md` Q503d)",
        visitor="Sign in to report a game.",
    ),
    Route(
        "POST",
        "/api/game/<hash>/nominate",
        "put this person's trial game on the shortlist, once it meets the bar "
        "(quota.SHORTLIST_DAYS on trial, quota.SHORTLIST_FINISHERS other finishers, "
        "quota.SHORTLIST_VOTES qualifying votes averaging quota.SHORTLIST_STARS) and under "
        "quota.SHORTLISTED_A_DESIGNER and quota.SHORTLIST_SIZE; it stamps games.nominated "
        "(`docs/plan-game-pipeline.md` Q503f)",
        visitor="Sign in to nominate a game. Only the person who made it can.",
    ),
    Route(
        "GET",
        "/api/round",
        "the latest monthly round: its entrants, each with its four-run score, stars and "
        "players, and this person's own vote; counts only after the close "
        "(`docs/plan-game-pipeline.md` Q503h)",
    ),
    Route(
        "GET",
        "/api/round/<date>",
        "one round by the 1st of its month, as `/api/round` answers the latest",
    ),
    Route(
        "POST",
        "/api/round/<date>/vote",
        "this person's yes or no on one entrant, while the round's week is open, by a "
        "member who finished a game on it, is not its designer and holds an account "
        "quota.VOTER_ACCOUNT_AGE old; changed until the close (Q503h)",
        visitor="Sign in to vote at the round. Members who have finished a game on a "
        "nominee vote on it.",
    ),
    Route(
        "POST",
        "/api/provisional",
        "a measurement this person's browser made of their own game, in tier one's slim "
        "form, filed for the server to walk and score itself "
        "(`docs/archive/plan-client-compute.md` B10), then a sample of it replayed exactly "
        "(tier two, `site/spotcheck.py`); refused at once by name when its basis is "
        "stale, `stale-build` being the one the page repairs by fetching the manifest, "
        "or when the person's day of checking is spent (quota.CHECK_CPU_A_DAY)",
        visitor="Sign in to have the site check a score measured on your computer.",
    ),
    Route(
        "POST",
        "/api/design/publish",
        "put one of this person's drafts on trial: the check passed, a spot-checked "
        "reading on today's basis attached (`drafts.attached`, Q503e), the account not "
        "flagged by a spot check, and under quota.PUBLISHED_AT_ONCE and "
        "quota.PUBLISHES_A_DAY (`docs/plan-game-pipeline.md` Q503a)",
        visitor="Sign in to publish a game. Only the person who saved it can.",
    ),
    Route(
        "GET",
        "/api/provisional/<job>",
        "where one of this person's submissions stands: queued, running, refused by "
        "name, or done with the server's own number, its state and its spot check's "
        "(`spotCheck`); nobody else's",
        visitor="Sign in to see a score you sent for checking.",
    ),
    # Q571, finding an opponent (`docs/archive/plan-return.md` §1.4). Each needs a member except
    # reading a challenge link, which says whose challenge it is to anybody holding it.
    Route(
        "GET",
        "/api/challenges",
        "this person's challenges: waiting on them, sent by them, and the games they "
        "are open to a game of",
        visitor="Sign in to see challenges. Playing a person needs an account.",
    ),
    Route(
        "POST",
        "/api/challenge",
        "send a challenge to a member by handle, make a challenge link, or offer a "
        "rematch; capped a day and at once (`quota.CHALLENGES_A_DAY`, "
        "`quota.CHALLENGES_WAITING`)",
        visitor="Sign in to challenge somebody. Playing a person needs an account.",
    ),
    Route(
        "POST",
        "/api/challenge/<id>/<action>",
        "accept or decline a challenge sent to you, or withdraw one you sent",
        visitor="Sign in to answer a challenge.",
    ),
    Route(
        "GET",
        "/api/challenge/link/<token>",
        "whose challenge a link is and to which game, for whoever holds the link; it is "
        "never an invitation",
    ),
    Route(
        "POST",
        "/api/challenge/link/<token>/accept",
        "accept a challenge link: the game starts and the link is used up",
        visitor="Sign in to accept this challenge. Accounts are by invitation for now.",
    ),
    # Q572, the one email (`docs/archive/plan-return.md` §1.5).
    Route(
        "POST",
        "/api/move-email",
        "tick or untick the move email on your own page: off by default, one email each "
        "time it becomes your move in a game between people, and nothing else",
        visitor="Sign in to change your email setting.",
    ),
    # Q596, a member's page (`docs/archive/plan-return.md` §6.2, R1).
    Route(
        "POST",
        "/api/favourites",
        "tick or untick *show my five-star games on my page*: off by default, and then "
        "only members see them, on `/players/<handle>`",
        visitor="Sign in to change what your page shows.",
    ),
    # Q719, a member's own invitations (`docs/archive/plan-launch.md` §5.3a).
    Route(
        "POST",
        "/api/invite",
        "send one of the invitations `/admin` gave this person, to one address: the "
        "ordinary invitation with them as its issuer; refused in a sentence at zero",
        visitor="Sign in to invite somebody.",
    ),
    # Q606, the maker takes challengers (`docs/archive/plan-return.md` §3 C4, R10).
    Route(
        "POST",
        "/api/takes-challengers",
        "switch *list me on the games I made as taking challengers* on or off: off by "
        "default, and refused to a person with no published game of their own",
        visitor="Sign in to change what your page shows.",
    ),
    Route(
        "POST",
        "/api/game/<hash>/open",
        "tick or untick *open to a game of this*: per person, per game, off by default",
        visitor="Sign in to say you are open to a game. Playing a person needs an account.",
    ),
    # R9b (Q605), proven positions (`docs/plan-website.md`, `site/proven.py`). For anybody:
    # a visitor finds the move as a member does, and only a member's right pick is kept.
    Route(
        "GET",
        "/api/proven/<hash>",
        "a game's proven positions, its own page (Q993b): the list shortest proved win "
        "first, grouped by turns to the win, the chips' counts and the reader's own "
        "progress; a visitor's finds this visit are sent back and kept nowhere",
    ),
    Route(
        "GET",
        "/api/proven/<hash>/<position>",
        "one of a game's proven positions, drawn, with whose move it is and, where the "
        "proof shows one winning move, the moves to pick from without their scores",
    ),
    Route(
        "POST",
        "/api/proven/<hash>/<position>/find",
        "find the move: the stored proof of the move picked, looked up and never worked "
        "out; a signed-in person's right pick is marked on their own page, a wrong one "
        "writes nothing",
    ),
    Route(
        "GET",
        "/api/admin",
        "users, invites, games, suggestions, reports and the quota numbers; admins only",
    ),
    Route(
        "POST",
        "/api/admin",
        "one administrative act: an invitation, a kill switch, revoke-all, forget, a "
        "round's confirm or veto",
    ),
)

# The site's own pages, which are not routes. The handler answers them with markup before
# any dispatch, the way `web/server.py` answers `/`. Listed here so the one table says
# everything the site serves. `STATIC` below is the other such entry.
PAGES: tuple[str, ...] = (
    "/join", "/signin", "/profile", "/players", "/suggestions", "/credits", "/privacy",
    "/terms", "/badges", "/admin", "/request-invitation",
)

# The site's pages with a variable part, by prefix: a challenge link, `/c/<token>`
# (Q571, `docs/archive/plan-return.md` §1.4). Like credits it is a constant page that reads no
# cookie; its script asks `/api/challenge/link/<token>` whose challenge it is.
# `pages.LINK` is the whole shape a token may take.
LINKS: tuple[str, ...] = ("/c/",)

# A member's page, `/players/<handle>` (Q596, `docs/archive/plan-return.md` §6.2): by prefix, like a
# challenge link, because the handle is the variable part. It reads the cookie: a visitor
# is sent to the door whatever the handle, so the page cannot be used to learn who is a
# member, and a member asking for a handle with no page gets the not-found page.
# `members.PATH` is the same string.
MEMBERS = "/players/"

# The one-click *switch them off* address in every move email, `/mail/off/<token>` (Q572,
# `docs/archive/plan-return.md` §1.5): the person's own token and no cookie. Opening it switches
# the box off and says so; a POST to it does the same, which is the one-click
# `List-Unsubscribe-Post` a mail program sends (RFC 8058), and the one POST the site takes
# without the page's header, since a mail program cannot send it and the address can do
# nothing but switch something off. `pages.MAIL_OFF` is its whole shape.
MAIL_OFF = "/mail/off/"

# The files the handler answers ahead of dispatch, which are not routes either. An entry
# ending in `/` is a prefix and the others are one path each. `web/server.py`'s handler
# answers them before any dispatch, the way it answers `/`, and the site inherits that
# handler -- so the site serves them by being the local application, not by registering
# anything.
#
# - `/static/vendor/`, the vendored third-party files, from
#   `chess_variant_sandbox/web/static/vendor/`. Only what its `manifest.json` names.
# - `/static/page/`, the application page's own scripts, from
#   `chess_variant_sandbox/web/static/page/` (and `viz/wrapped.js`). Only what
#   `web.page.SCRIPTS` names (Q513).
# - `/static/site/`, the site's pages' scripts, from `chess_variant_sandbox/site/static/`.
#   Only what `pages.SCRIPTS` names; the site's handler answers these, beside its pages.
# - `/static/fonts/`, the pages' typefaces, from `chess_variant_sandbox/web/static/fonts/`.
#   Only what its `manifest.json` names (`web/page/shell.py`).
# - `/static/shell.js`, the shell's script (the theme before the first paint, the theme
#   switch, the phone's menu), and the mark as icons: `/favicon.svg`, `/favicon.ico`,
#   `/apple-touch-icon.png`, made from the drawing in `web/page/shell.py`.
# - `/static/measure-worker.js`, the browser's measuring worker, from the package
#   (`docs/archive/plan-client-compute.md` B3). A 404 while the package carries none.
# - `/rules.md`, the rule language's reference (`RULES.md`) as the running code renders it,
#   `text/markdown`, cached five minutes (Q905). The designer links it (`reference.PUBLISHED`),
#   so a person writing a game reads the version this site's validator accepts.
# - `/browser/`, the browser build -- Pyodide and this package's wheel -- from the
#   app's `browser_dir`, `build/browser/` (B1, B4). Only what that build's
#   `manifest.json` names, and nothing while it has none. On the box Caddy answers
#   these paths itself once B7 ships the build, and this is the fallback behind it.
# - `/static/picture/`, the link-preview picture every page's head names (G5,
#   `web/page/picture.py`), from `chess_variant_sandbox/web/static/picture/`. Only what its
#   `manifest.json` names, cached for a year: a new picture is a new file name.
# - `/robots.txt` and `/sitemap.xml`, the doors for crawlers (G4, `site/seo.py`), and
#   `/llms.txt`, the same for an assistant (G8): the site's handler answers them, beside `/`
#   and `/game` with their own heads.
# - `/rules`, the rule language as a page (G8, `web/page/rules_page.py`): drawn from the same
#   `render_reference()` as `/rules.md`, in the site's shell here and under the application's
#   header on the local application. A constant, cached five minutes like `/rules.md`.
#
# - `/whats-new`, *What's new* drawn on the server (G11, `web/page/news_page.py`), and
#   `/feeds/`, the two Atom feeds (`site/feeds.py`): the changelog's entries and the new
#   games. Read from the changelog file and the store, never from a cookie.
# - `/static/style/`, the pages' stylesheets, which the code writes (G12,
#   `web/page/assets.py`). Every script above under `/static/page/`, `/static/site/` and
#   `/static/shell.js` is answered under a versioned name too, `<stem>.<digest>.js`, which is
#   the one the pages name: both for a year, and only a name `assets` holds.
#
# **Served to anybody.** They are the same bytes for every viewer and nobody's name is
# read; all but the last (which reads the changelog and the store) touch no store. A path cannot reach out of any directory, because each
# is served by a lookup in its manifest or table and never by resolving what was asked.
STATIC: tuple[str, ...] = (
    "/static/vendor/", "/static/page/", "/static/site/", "/static/measure-worker.js", "/browser/",
    "/static/fonts/", "/static/shell.js", "/favicon.svg", "/favicon.ico", "/apple-touch-icon.png",
    "/static/picture/", "/rules.md", "/robots.txt", "/sitemap.xml", "/llms.txt", "/rules",
    "/whats-new", "/feeds/", "/static/style/",
)

# The pages whose answer depends on who is asking, and so the only ones the cookie is
# read for. Credits, the privacy notice and the list of badges are constants; a database
# connection to serve a constant would be a connection per page view, and the three would
# need a store on a machine that has none. The badge list is deliberately one of them: it
# reads the same for everybody, which is what makes it §3's answer to a hidden rule.
#
# `/players` is here because it needs a signed-in person. Its markup is the same for
# everybody, but it gathers what people have done into one place, and the site is
# invite-only; a visitor asking for it is sent to the door, as one asking for a profile is.
# `/suggestions` is here for the same reason: posting one names a person and the list
# under the form is that person's own.
PERSONAL: tuple[str, ...] = (
    "/join", "/signin", "/profile", "/players", "/suggestions", "/admin",
)

# The page and the two routes behind it that only an admin may reach. Everybody else gets
# the 404 an invented path gets, from the same message, rather than a 403 -- a member who
# is told *forbidden* has been told the facility exists and what it is called.
ADMIN_ONLY: tuple[str, ...] = ("/admin", "/api/admin")

# What the site does not register, and why. Every one of these is a handler the local
# application has and the site declines to reach. `docs/plan-website.md` §2 lists them
# by reason.
WITHHELD: tuple[Route, ...] = (
    Route("POST", "/api/run", "a design run spends money; admin only, or a brought key (§8)"),
    Route("GET", "/api/run/<id>", "the state of a design run the site cannot start"),
    Route("GET", "/api/runs", "the same, as a listing"),
    Route("GET", "/api/scoring", "a corpus sweep is the maintainer's; admin only (§2)"),
    Route("POST", "/api/scoring", "the same sweep, started"),
    Route("POST", "/api/scoring/stop", "the same sweep, stopped"),
    Route(
        "POST",
        "/api/archive",
        "writes the repository; a maintainer's CLI act, never the site's (§3 rule 3)",
    ),
    Route("POST", "/api/promote", "the same, into the calibration corpus"),
    Route(
        "POST",
        "/api/design/icon",
        "cheap, but an uploaded SVG waits on §6's sanitising policy",
    ),
    Route("POST", "/api/test", "a canonical measurement: minutes of tournament (1.2, 1.7)"),
    Route("GET", "/api/test/<hash>", "polls a measurement the site cannot start"),
    Route(
        "POST",
        "/api/rate",
        "rates a picker by tournament: minutes, 1838 CPU-seconds on the box (§2)",
    ),
    Route(
        "POST",
        "/api/design/measure",
        "queues a draft's canonical measurement; withheld until plan-usage's first month",
    ),
    Route(
        "POST",
        "/api/design/nominate",
        "writes runs/designed/nominations.jsonl, the local application's; the site "
        "nominates a trial game with POST /api/game/<hash>/nominate (Q503f)",
    ),
    Route("GET", "/api/rating/<hash>", "polls a rating run the site cannot start"),
    Route(
        "POST",
        "/api/opponent",
        "writes heuristics/custom/; nothing a user does reaches the repository (§3 rule 3)",
    ),
    Route("POST", "/api/opponent/forget", "the same store, unwritten"),
    Route(
        "GET",
        "/api/human",
        "the anchor is the CLI's offline reading; the site's table is §5's (1.4, 1.11)",
    ),
    Route("GET", "/api/human/<hash>", "the same anchor, per ruleset"),
)
