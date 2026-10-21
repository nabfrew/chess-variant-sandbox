"""The public site: the local application, with a shorter route table.

Not a second application. `SiteApp` is `web.server.App` with two things changed -- which
paths it will dispatch, and where a game in progress lives -- so every handler the site
serves is the handler the local app serves, and a fix to one is a fix to both. What the
site adds beyond that arrives as its own modules (the store, identity, the queue), never
as a second reading of a route the local app already answers.

The second change is task 1.4's. A game in progress is a row in `matches` rather than a
dictionary that dies with the process, so the two session routes come through
`site/matches.py` and every other route is the local application's untouched. Each
request opens a connection, commits on its way out and closes; a pool is task 1.7's, when
there is a load to size it against.

The page is the local app's page, served by the local app's handler. It still offers
buttons for routes the site withholds, and those get a 404 the way they got a 403 from
Caddy during the trial week; the page's own trimming is phase 1's later rows.

The third change is task 1.3's, and it is the handler rather than the app. Who is asking
arrives as a cookie, which is HTTP and not a route, so `site_handler_for` wraps the local
application's handler with two things: it fills `SiteApp.viewer` from the cookie before
any dispatch, and it sets or clears that cookie afterwards when the request was a sign-in.
The site's own pages -- joining, signing in, a profile, 1.9's credits and privacy notice,
and 1.8's maintenance page -- are answered there too, ahead of dispatch, the way the local
handler answers `/`.

The fourth change is task 1.8's, and it runs through both. `routes.ADMIN_ONLY` names the
page and the two routes a `users.role` of `admin` reaches and nobody else does, and the
refusal is the 404 an invented path gets rather than a 403. The per-game kill switch is
three doors in this file -- the listing drops a killed game, its own page and the
designer's routes answer 404, and `matches.start` refuses to seat anybody at it.

A draft saved on the site gets its `games` row in the same request (`site/drafts.py`), so
both switches and a match reach it the way they reach a corpus game.
"""

from __future__ import annotations

import re
import sys
import threading
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from chess_variant_sandbox import corpus
from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.measurement import Basis
from chess_variant_sandbox.site import (
    admin,
    alarm,
    asking,
    awards,
    badges,
    browse,
    challenges,
    correspondence,
    daily,
    db,
    depths,
    designs,
    drafts,
    finding,
    game_page,
    game_pictures,
    identity,
    inviting,
    kept,
    leaderboard,
    legality,
    mail,
    matches,
    members,
    milestones,
    move_email,
    opponent_line,
    pages,
    people_scale,
    proven,
    provisional,
    quota,
    rating,
    reports,
    rounds,
    routes,
    screening,
    seo,
    shortlist,
    showcase,
    spine,
    stages,
    suggestions,
    terms,
    training,
    unbeaten,
    usage,
    votes,
)
from chess_variant_sandbox.site import people as trust_people
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.canonical import load_spec
from chess_variant_sandbox.web import (
    changelog,
    designing,
    launch,
    library,
    opponents,
    play,
    refreshing,
)
from chess_variant_sandbox.web.page import SCRIPTS as PAGE_SCRIPTS
from chess_variant_sandbox.web.page import front, game_picture, picture, rules_page, shell
from chess_variant_sandbox.web.page import game as game_drawing
from chess_variant_sandbox.web.server import (
    BROWSER_PREFIX,
    IMMUTABLE,
    PAGE_PATHS,
    RULES_CACHE,
    RULES_ROUTE,
    VENDOR_PREFIX,
    WORKER_ROUTE,
    App,
    handler_for,
    listen,
    warm,
)
from chess_variant_sandbox.web.server import _watching as watching_line

# Not 8765. The local application has that, and a person running both wants two ports
# rather than a refusal. The box's unit names its port outright, so this default is a
# convenience and never a deployment's answer.
DEFAULT_PORT = 8766

DESIGN_DRAIN_BYTES = 2 * 1024 * 1024
"""How much of a designer body refused for its size is read and thrown away (Q841).

The box's Caddy admits no body over 2 MB (`deploy/remote/Caddyfile`'s `max_size`), so on
the box every refused body is drained and its 413 arrives; a body the drain would not
cover never reaches the application there. Off the box a claim over this is closed on
unread. Drained in pieces and kept nowhere, so it bounds bandwidth, not memory: the cap
`quota.DESIGN_BODY_BYTES` still refuses before anything is parsed or held."""

CHALLENGE_FIELDS = frozenset({"hash", "seat", "handle", "link", "rematch"})
"""Every field `/api/challenge` takes. Anything else is refused, so no field can carry a
message (Q606)."""


@dataclass(frozen=True)
class Grant:
    """What the answer to this request should do to the cookie.

    A token to set one, `None` to take it away. The app decides -- it is the thing that
    opened or closed the session -- and the handler writes the header, because the
    attributes that make the cookie safe are properties of the connection rather than of
    the session.
    """

    token: str | None


# W7 (Q614, `docs/archive/plan-understanding.md` §3.3 G39): what a spec's provenance keeps for the
# maintainer. `history` and `notes` are the record of how a game was made and changed (runs,
# hashes, schema versions, who decided what), and `run` names a design run. The local
# application draws them; the site neither draws nor sends them.
RECORDS = ("history", "notes", "run")

# A `source` that names a file in the project's tree (`docs/game-ideas.md`,
# `archive/wolves`) or a queue task (`Q281`) is the maintainer's pointer, not a citation a
# visitor can follow; a country, a book or a web address stays.
_IN_THE_TREE = re.compile(r"(?<![:/\w])(?:docs|archive|rulesets|runs)/|\bQ\d+\b")


def visitors_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """A provenance as the site sends it: without the maintainer's records (`RECORDS`), a
    machine's run name for its author, or a `source` that points into the project's tree."""
    kept = {key: value for key, value in provenance.items() if key not in RECORDS}
    # A machine's game has no author but the machine, and its `author` is the internal name
    # of the run that made it (`stage 25`, a command line); the origin already says *the
    # computer*.
    if kept.get("origin") in ("generated", "evolved"):
        kept.pop("author", None)
    if _IN_THE_TREE.search(str(kept.get("source") or "")):
        kept.pop("source")
    return kept


def _without_records(found: dict[str, Any]) -> dict[str, Any]:
    """A game's page as the site sends it: its provenance, and the spec's own copy of it,
    are `visitors_provenance`'s. The provenance is outside the mechanics hash, so the spec
    still names the same game."""
    if isinstance(found.get("provenance"), dict):
        found["provenance"] = visitors_provenance(found["provenance"])
    spec = found.get("spec")
    if isinstance(spec, dict) and isinstance(spec.get("provenance"), dict):
        found["spec"] = {**spec, "provenance": visitors_provenance(spec["provenance"])}
    return found


def history_shown(found: dict[str, Any], is_admin: bool) -> list[dict[str, Any]]:
    """A game page's score history as the site sends it (Q649).

    The admin is sent every line with its basis, as the local application is. Anybody else
    is sent none here: a visitor reads each earlier score as a *was* line under the trust
    line's *more* (`library.trust_line`, `history.for_visitors`; the user's answer to
    `docs/archive/plan-confidence.md` §5 Q1, A, on 2026-10-03), dated and labelled with the
    players that made it, and never with its basis.
    """
    if not is_admin:
        return []
    return [line for line in found.get("history") or [] if isinstance(line, dict)]


class SignInFirst(ValueError):
    """A visitor asked for a route that needs a signed-in person.

    A `ValueError` like `quota.QuotaError`, so the answer is a 400 carrying the sentence
    the page prints where the click was.
    """


class SiteApp(App):
    """The local application behind an allowlist, playing into the store.

    A withheld route raises the `KeyError` an invented path raises, with the same text,
    so the site cannot be read from the outside for what it is declining to serve. The
    handler turns that into a 404.
    """

    def __init__(
        self,
        ruleset_dir: Path,
        runs_dir: Path,
        heuristics_dir: Path | str = "heuristics",
        archive_dir: Path | str = corpus.ARCHIVE_DIR,
        connect: Any = None,
        send: mail.Sender | None = None,
    ) -> None:
        super().__init__(ruleset_dir, runs_dir, heuristics_dir, archive_dir)
        self.connect = connect or db.connect
        # What an invitation from `/admin` is mailed through. Tests pass a recorder.
        self.send: mail.Sender = mail.send if send is None else send
        # Both of these are the local application's own objects replaced by readings of
        # `matches`. `sessions` answers `GET /api/session/<id>` from the row, and
        # `humans` is the anchor over leaderboard rows -- the site never writes
        # `runs/human/games.jsonl`, and `matches.Record` refuses rather than letting a
        # second store drift out of step with the first.
        # Drafts are the one game that is not a row: `POST /api/design/play` plays rules
        # nobody has kept, so there is no `games` row to point a match at. They live in
        # the local application's own in-memory store, behind the same session routes.
        self.drafts = play.Sessions()
        self.sessions = matches.Sessions(self._connection, drafts=self.drafts)
        self.humans = matches.Record(self._connection)
        # The designer's *show me* presses, counted per account for the last hour.
        self.presses = quota.Presses()
        # Each person's saves in the last rolling day (Q503a): a save runs the check.
        self.saves = quota.Saves()
        # Failed sign-ins and failed invitation codes, by address and by handle (Q369).
        # One object for the whole process, so the count holds across the threads the
        # server answers requests on.
        self.signins = quota.SignIns()
        # Requests for an invitation, by network address (Q831): `quota.ASKS_A_SOURCE` a
        # day from one, so one source cannot spend the site's `ASKS_A_DAY`.
        self.asks = quota.Asks()
        # The owner's mails about those requests (Q979), each sent on a thread of its own
        # after the row is committed, so the visitor never waits on the sender and how
        # long the answer took says nothing about the address. `settle` waits for them.
        self._telling: list[threading.Thread] = []
        self._telling_lock = threading.Lock()
        # Every dispatched request, counted by the hour with nobody named
        # (`docs/plan-usage.md`). *Show me*'s people are read off `presses`.
        self.usage = usage.Counter(lambda: self.presses)
        # Who is playing, and what this answer should do to their cookie -- both per
        # request, and this server answers requests on a thread each. One `viewer`
        # attribute shared between threads would be one person's identity answering
        # another person's request, which is the failure that has to be impossible rather
        # than unlikely.
        self._request = threading.local()
        # Tier one's runner (B10): the one thread that walks a submitted measurement,
        # started by `serve` and woken by each submission. A test drains it by hand.
        self.checks = legality.Checker(self._connection, self.runs_dir)
        # G3: each game's near list for its own address, reused for a few minutes.
        self.near_cache = game_page.NearCache()
        # G4: the games a search engine may index, for the sitemap and `/game` without script.
        self.listing = game_page.ListedCache()
        # G6: each game's link-preview picture, made once on a thread of its own and kept in
        # runs/pictures/.
        self.pictures = game_pictures.Pictures(self.runs_dir / game_pictures.DIRECTORY)

    @property
    def viewer(self) -> int | None:
        """Who is asking, or `None` for a visitor who has not signed in.

        Filled from the cookie by the handler, before anything dispatches. A visitor
        plays in sandbox mode, which counts for nothing and needs no identity (§4);
        leaderboard mode, a vote and a profile all name a person and are refused until
        one is known.
        """
        return getattr(self._request, "viewer", None)

    @viewer.setter
    def viewer(self, user: int | None) -> None:
        self._request.viewer = user

    @property
    def source(self) -> str:
        """Which address this request is counted against, written by the handler.

        `quota.source_address` reads it off the socket and the forwarding header; the app
        only holds it, per request like `viewer`. The empty string is nobody's address
        and stands for a caller that reached the app without a socket, which is a test.
        """
        return str(getattr(self._request, "source", ""))

    @source.setter
    def source(self, address: str) -> None:
        self._request.source = address

    @property
    def grant(self) -> Grant | None:
        """What this request did to the session, for the handler to write as a cookie."""
        return getattr(self._request, "grant", None)

    @grant.setter
    def grant(self, given: Grant | None) -> None:
        self._request.grant = given

    @property
    def role(self) -> str:
        """What this person is allowed to reach, from the same read as the cookie.

        `member` for everybody an invitation makes and for a visitor, `admin` for a
        maintainer. Kept beside the id rather than asked for again, so the admin page
        costs the same query the cookie already paid for.
        """
        return str(getattr(self._request, "role", admin.MEMBER))

    def identify(self, token: str | None) -> identity.Viewer | None:
        """Read the cookie, on the one connection that costs, and remember the answer.

        Called once per dispatched request. The `Viewer` goes back to the handler, which
        needs the handle for the profile page; the app itself keeps the id and the role,
        because every route that names a person names them by the first and the admin
        routes turn on the second.
        """
        with self._connection() as connection:
            found = identity.resolve(connection, token)
        self.viewer = None if found is None else found.id
        self._request.role = admin.MEMBER if found is None else found.role
        self._request.token = token
        self.grant = None
        return found

    def get(self, path: str, query: dict[str, str] | None = None) -> Any:
        self._registered("GET", path)
        asked = query or {}
        parts = routes.split(path)
        if parts == ("api", "library"):
            return self._library(asked)
        if parts == ("api", "me"):
            return self._me()
        if parts == ("api", "profile"):
            return self._profile()
        if parts == ("api", "challenges"):
            return self._challenges()
        if len(parts) == 4 and parts[:3] == ("api", "challenge", "link"):
            with self._connection() as connection:
                return challenges.link_view(connection, parts[3], self.viewer)
        if parts == ("api", "players"):
            return self._players()
        if parts == ("api", "graph", "people"):
            return self._graph_people()
        if parts == ("api", "suggestions"):
            return self._suggestions()
        if parts == ("api", "admin"):
            return self._admin_view()
        if parts == ("api", "changelog"):
            return self._whats_new()
        if parts == ("api", "round"):
            return self._round(None)
        if len(parts) == 3 and parts[:2] == ("api", "round"):
            return self._round(rounds.parse_day(parts[2]))
        if len(parts) == 3 and parts[:2] == ("api", "provisional"):
            return self._submission(parts[2])
        if len(parts) == 3 and parts[:2] == ("api", "session") and not self.drafts.find(
            parts[2]
        ):
            return self._session_read(parts[2])
        if len(parts) == 3 and parts[:2] == ("api", "game"):
            self._playable(parts[2])
            found = {**super().get(path, asked), **self._standing(parts[2])}
            # A stored player the picker left out is named and no more (G19): the file it
            # is in and the command that repairs it are the maintainer's.
            found["refusedStores"] = [
                opponents.unavailable(name) for name in found.pop("refusedNames", [])
            ]
            found["history"] = history_shown(found, self.role == admin.ADMIN)
            found["trust"] = self._trust_line(parts[2], found.get("trust"))
            return _without_records(found)
        if len(parts) == 3 and parts[:2] == ("api", "proven"):
            # Q993b: the positions' own page, the list with the reader's marks.
            self._playable(parts[2])
            return self._proven_page(parts[2], asked)
        if len(parts) == 4 and parts[:2] == ("api", "proven"):
            # R9b: one proven position, drawn, and where it has one winning move its moves.
            self._playable(parts[2])
            record, game = self._proven_record(parts[2])
            return proven.view(record, game.spec, parts[3])
        if len(parts) >= 3 and parts[:2] == ("api", "design"):
            # A killed game is no designer's starting point and has no progress page.
            self._playable(parts[-1])
        if len(parts) == 4 and parts[:3] == ("api", "design", "standing"):
            found = super().get(path, asked)
            with self._connection() as connection:
                found["publish"] = drafts.standing(
                    connection, parts[3], self.viewer, self._current
                )
                # Q518. The shortlist's bar on the designer's own track too, the game
                # page's lines and its *nominate it* (`shortlist.panel`); only a game on
                # trial has one, and only its designer is sent the button.
                found["shortlist"] = shortlist.panel(
                    connection, parts[3], self.viewer, self._counted(connection, parts[3])
                )
                row = stages.read(connection, [parts[3]]).get(parts[3])
                says = None if row is None else stages.said(connection, row, self.viewer)
            # *Where this game stands* (Q503b): the level and its track, for any game the
            # store holds. Rules nobody has saved have no row, and are a Draft.
            found.update(stages.shown(stages.DRAFT if row is None else row["level"], says))
            return found
        return super().get(path, asked)

    def post(self, path: str, body: dict[str, Any]) -> Any:
        self._registered("POST", path)
        parts = routes.split(path)
        if parts == ("api", "vote"):
            return self._vote(body)
        if parts == ("api", "suggest"):
            return self._suggest(body)
        if parts == ("api", "join"):
            return self._join(body)
        if parts == ("api", "signin"):
            return self._signin(body)
        if parts == ("api", "signout"):
            return self._signout()
        if parts == ("api", "ask-invitation"):
            return self._ask_invitation(body)
        if parts == ("api", "admin"):
            return self._admin_act(body)
        if parts == ("api", "challenge"):
            return self._challenge(body)
        if parts == ("api", "move-email"):
            # Q572. The box on a person's own page: tick or untick the move email.
            with self._connection() as connection:
                viewer = self._signed_in()
                move_email.set_on(connection, viewer, bool(body.get("on")))
                return _move_box(connection, viewer)
        if parts == ("api", "favourites"):
            return self._favourites(body)
        if parts == ("api", "invite"):
            # Q719. One of the invitations `/admin` gave this person, sent from their page.
            with self._connection() as connection:
                viewer = self._signed_in()
                sent, _left = inviting.send(
                    connection, viewer, str(body.get("email") or ""), mail.site_url(),
                    self.send,
                )
                return {"said": f"Invited {sent.email}. The link is on its way.",
                        "box": inviting.box(connection, viewer)}
        if parts == ("api", "takes-challengers"):
            # Q606. The maker's one switch on their own page.
            with self._connection() as connection:
                viewer = self._signed_in()
                challenges.set_takes_challengers(connection, viewer, bool(body.get("on")))
                return challenges.maker_box(connection, viewer)
        if len(parts) == 5 and parts[:3] == ("api", "challenge", "link") and parts[4] == "accept":
            with self._connection() as connection:
                made = challenges.accept_link(connection, parts[3], self._signed_in())
            return {"id": made.id}
        if len(parts) == 4 and parts[:2] == ("api", "challenge"):
            return self._challenge_answer(parts[2], parts[3])
        if len(parts) == 4 and parts[:2] == ("api", "game") and parts[3] == "open":
            with self._connection() as connection:
                on = challenges.set_open(
                    connection, self._signed_in(), parts[2], bool(body.get("on"))
                )
                return {"open": on, "people": challenges.people(
                    connection, parts[2], self._signed_in()
                )}
        if len(parts) == 5 and parts[:2] == ("api", "proven") and parts[4] == "find":
            return self._find_the_move(parts[2], parts[3], body)
        if parts == ("api", "provisional"):
            return self._submit(body)
        if parts == ("api", "design", "publish"):
            return self._publish(body)
        if len(parts) == 4 and parts[:2] == ("api", "game") and parts[3] == "delete":
            return self._delete_game(parts[2])
        if len(parts) == 4 and parts[:2] == ("api", "game") and parts[3] == "report":
            return self._report(parts[2], body)
        if len(parts) == 4 and parts[:2] == ("api", "game") and parts[3] == "nominate":
            return self._put_forward(parts[2])
        if len(parts) == 4 and parts[:2] == ("api", "round") and parts[3] == "vote":
            return self._round_vote(rounds.parse_day(parts[2]), body)
        if parts == ("api", "design", "screen"):
            # The same capped check the save runs, asked for on its own. The local
            # application's handler runs it uncapped in the process that serves the
            # request, which is the maintainer's machine and its own affair.
            return screening.check(body.get("spec"))
        if parts == ("api", "design", "play") and body.get("show"):
            # *Show me* plays 32 searched games per press and *show the stuck game* 16
            # random ones, so the site counts both against one per-account budget before
            # any are played. A plain draft game passes. Noted first, so a refused press
            # lands on the same usage row.
            show = body["show"]
            if show == "unfinished":
                usage.note("show=unfinished")
                self.presses.press(self.viewer)
            elif isinstance(show, dict):
                usage.note("show=rule")
                self.presses.press(self.viewer)
        return super().post(path, body)

    def _registered(self, method: str, path: str) -> None:
        """Whether this request is a route at all, for whoever is asking.

        An admin route is not a route to anybody else: it raises the `KeyError` an
        invented path raises, with the same text, so a member cannot learn from a status
        code that the site has a maintenance page (task 1.8).

        A route that needs a signed-in person refuses a visitor with its own sentence,
        before the local application's handler runs.
        """
        parts = routes.split(path)
        missing = KeyError(f"no such endpoint: /{'/'.join(parts)}")
        found = routes.find(method, path)
        if found is None:
            raise missing
        if f"/{'/'.join(parts)}" in routes.ADMIN_ONLY and self.role != admin.ADMIN:
            raise missing
        if found.visitor and self.viewer is None:
            raise SignInFirst(found.visitor)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """One connection for one unit of work, committed at the end or rolled back.

        `site/matches.py` and `site/rating.py` both refuse to commit, so that a finish
        and the ratings it moved land together. This is where together is decided.
        """
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    # ------------------------------------------------ the games page, with the store on it

    def _library(self, query: dict[str, str]) -> Any:
        """The local application's listing with the site's own columns and orderings.

        Task 1.5. The cards are the library's, read off the files exactly as the local
        application reads them; what the store adds is the three things §5 lets a person
        sort and filter by -- stars, human games played and the *Unbeaten* tag -- plus
        whether the game is rated at all, which is what decides between offering
        leaderboard mode and offering sandbox with a reason (§4). Box B2 of
        `docs/archive/plan-meta-game.md` adds two more filters over the same columns:
        *untried*, this viewer's own (a visitor has none to read), and *unplayed*, nobody's.

        `sort` and `filter` are the query's, and an unrecognised value is refused by name
        rather than ignored. With neither, the order is the library's own and unchanged.

        A game the maintainer has killed is dropped here, before the columns are read.
        That is 1.8's per-game switch on the listing side; `matches.start` is the other
        side, and `_playable` is the game's own page. A card from `runs/` is dropped too
        unless the store holds a public row for it (`drafts.offered`).

        A game the loader retired is dropped by the same door, which is why `gone` is
        `unlisted_games` rather than `revoked_games`. The two states part on the other
        two doors: a retired game's row and its matches are still read.
        """
        payload = super().get("/api/library")
        with self._connection() as connection:
            public = drafts.listed(connection)
            gone = admin.unlisted_games(connection)
            payload["games"] = drafts.offered(payload["games"], shown=public, gone=gone)
            # Q931. The previews, apart from the library and through the same door: a
            # preview the store has no public row for, or one taken off the site, is not
            # offered. Nothing below that ranks, bands, counts or features a game reads
            # them; they get their row's level, the store's columns and their own tab.
            previews = drafts.offered(payload.get("previews") or [], shown=public, gone=gone)
            preview_rows = stages.read(connection, [str(card["hash"]) for card in previews])
            preview_columns = browse.columns(
                connection, [str(card["hash"]) for card in previews], viewer=self.viewer
            )
            hashes = [str(game["hash"]) for game in payload["games"]]
            found = browse.columns(connection, hashes, viewer=self.viewer)
            checked = self._provisional(connection, hashes)
            rows = stages.read(connection, hashes)
            people_counts = trust_people.counts_of(connection, hashes)
            # Q741: the landing page's *Made by members*, the newest games members
            # published, over the cards this listing offers (so never a draft, a resting
            # game or one taken off the site).
            payload["madeHere"] = showcase.made_here(connection, payload["games"], rows)
        known = self.known()
        # Q564 (L6, `docs/reference/opponent-line.md` §3.1) and Q654 (B5): the front page's first
        # step, the day's featured game with its gentlest opponent and the next one up,
        # and the launch set in order, both over the games this listing offers.
        offered = [known[str(card["hash"])] for card in payload["games"]
                   if str(card["hash"]) in known]
        payload["firstStep"] = launch.first_step(
            offered, payload.get("opponents") or [], launch.today(),
            preview=lambda game: designing.opening_preview(game.spec),
        )
        payload["launchSet"] = launch.launch_cards(offered)
        # Q603 (C2, `docs/archive/plan-return.md` §3): this month's game, by the calendar month.
        payload["monthGame"] = launch.month_game(offered, launch.today())
        # Q788: the landing page's newest games and spotlight, over what this listing
        # offers, so a game taken off the site is never highlighted.
        payload["newGames"] = launch.new_cards(offered)
        payload["spotlight"] = launch.spotlight_card(offered)
        # Q647. The card's trust chip with people's counts folded in (they may move the
        # level, never the score: `trust.RULE.people_move_level`).
        for game in payload["games"]:
            listed = known.get(str(game["hash"]))
            if game.get("trust") is not None and listed is not None:
                game["trust"] = library.trust_card(
                    listed, people_counts.get(str(game["hash"]))
                )
        # Q503b. The card's level is the row's (`stages.stage`), over the file's word the
        # library put there, and a listed game carries the day it was published for the
        # *On trial* tab's newest-first order.
        for game in payload["games"]:
            row = rows.get(str(game["hash"]))
            if row is not None:
                game["level"] = row["level"]
                game["levelGloss"] = stages.GLOSSES.get(row["level"], "")
                published = row["published"]
                game["onTrialSince"] = None if published is None else published.isoformat()
                # Q826: a member's game joined the site the day its designer published
                # it (`games.published`, UTC), the day the *date added* order reads. A
                # library file's day is its own (`library.added`) and is left alone.
                if row.get("origin") == stages.DESIGNED:
                    game["added"] = None if published is None else (
                        published.astimezone(timezone.utc).date().isoformat())
        cards = browse.decorate(payload["games"], found)
        # The header's count over the cards this listing offers, drafts and retired games
        # already dropped and before a filter narrows it: the leaderboard's, not the view's.
        payload["provisionalScores"] = library.provisional_count(cards)
        # A number measured on somebody's own machine rides on the card under its own
        # key, never `index`, so the ordering below cannot read it (Q312).
        for card in cards:
            card["provisional"] = checked.get(str(card["hash"]))
            # Q972: how many proven positions the game's page offers, the record its
            # section reads (`proven.offered`), for the *proven positions* filter.
            listed = known.get(str(card["hash"]))
            card["provenPositions"] = proven.offered(
                self.runs_dir, str(card["hash"]), None if listed is None else listed.spec
            )
        payload["games"] = browse.order(
            cards, query.get("sort", ""), query.get("filter", "")
        )
        # Q931. A preview's level is its row's, as every card's is; a preview with no row
        # was dropped by `drafts.offered` above. By name, never by a score.
        shown_previews = browse.decorate(previews, preview_columns)
        for card in shown_previews:
            row = preview_rows.get(str(card["hash"]))
            if row is not None:
                card["level"] = row["level"]
                card["levelGloss"] = stages.GLOSSES.get(row["level"], "")
        payload["previews"] = sorted(
            shown_previews, key=lambda card: str(card["name"]).lower()
        )
        # L4 of `docs/archive/plan-launch.md` §3.4: a tab past the first is sent only when it
        # holds a card. *On trial* is absent, not empty, until somebody publishes; so are
        # the previews (Q931).
        counted = [
            {
                **tab,
                "count": sum(
                    1 for card in [*payload["games"], *payload["previews"]]
                    if card.get("level") in tab["levels"]
                ),
            }
            for tab in stages.TABS
        ]
        tabs = counted[:1] + [tab for tab in counted[1:] if tab["count"]]
        # What this listing can be sorted and filtered by, sent with it rather than
        # assumed by the page, so a key added here reaches the menu without a page change.
        # `sort` is what the query takes and `key` is what the card carries, because the
        # page orders the payload it already holds rather than asking again.
        payload["site"] = {
            # Who is asking, for a header that has a link only some people should see.
            # `signedIn` on the game page is the same answer for the same reason (Q192);
            # this is the listing's, because the header sits above the listing too. The
            # link itself is `chess_variant_sandbox/web/page/`'s, which is Q197's.
            "signedIn": self.viewer is not None,
            "admin": self.role == admin.ADMIN,
            "sorts": [
                {"sort": browse.STARS, "key": "stars", "label": "star rating"},
                {
                    "sort": browse.HUMAN_GAMES,
                    "key": "humanGames",
                    "label": "games played by people",
                },
                # Only while a game is on trial: an order by nothing is an empty tab.
                *(
                    [{"sort": browse.NEWEST, "key": "onTrialSince",
                      "label": "newest on trial"}]
                    if any(tab["key"] == "trial" for tab in tabs) else []
                ),
            ],
            # §3.2's tabs (Q503b): which levels each lists, how many cards it holds and
            # the order it opens in. The page files each card by its `level` and holds no
            # level word itself; a level in no tab is never listed.
            "tabs": tabs,
            "filters": [
                {
                    "filter": browse.UNBEATEN,
                    "key": "unbeaten",
                    "label": "nobody has beaten its strongest computer player yet",
                },
                {
                    "filter": browse.UNTRIED,
                    "key": "untried",
                    "label": "you haven't tried this yet",
                },
                {
                    "filter": browse.UNPLAYED,
                    "key": "unplayed",
                    "label": "not played here yet",
                },
                {
                    "filter": browse.PROVEN,
                    "key": "provenPositions",
                    "label": "has proven positions to solve",
                },
            ],
        }
        return payload

    def _trust_line(self, digest: str, sent: Any) -> Any:
        """A game page's trust line on the site (Q647, `docs/archive/plan-confidence.md` §2.1).

        The library's reading of the files with this game's people's counts folded in
        (`site/people.py`), so people's results move the level the way §2.3 says and the
        score not at all. A visitor is sent the level, its sentence and *more*; the admin
        the maintainer's numbers too, as the local application shows them. A game the
        listing does not hold keeps what was sent, less the numbers.
        """
        is_admin = self.role == admin.ADMIN
        game = self.known().get(digest)
        if game is None:
            if isinstance(sent, dict) and not is_admin:
                return {key: value for key, value in sent.items() if key != "numbers"}
            return sent
        with self._connection() as connection:
            counted = trust_people.counts(connection, digest)
        return library.trust_line(game, counted, maintainer=is_admin)

    def _standing(self, digest: str) -> dict[str, Any]:
        """The game page's half of §5: the tag, the target, and this person's record.

        The target is named whether or not the tag is still on, because *Unbeaten* with
        nothing to beat is a mystery rather than a motivator. `yourRecord` is absent until
        1.3's cookie says who is asking.

        `signedIn` is what the page's mode control reads. A visitor is offered sandbox
        and told in one sentence that a game which counts needs signing in, rather than
        being offered a choice the server would refuse.

        `siteRatings` is the opponent table's second column (A6 of
        `docs/archive/plan-ux.md`): every computer player's live rating on this scale,
        keyed by the agent string the picker ladder names it by, so the page can put the
        two numbers for one player side by side. Only the computer players: the opponent
        picker reads it. People are on `leaderboard` (L2 of `docs/archive/plan-leaderboard.md`),
        the whole table as `site/leaderboard.py` orders and marks it.

        `firstPlayed`, `highest` and `clearedBy` are the meta-game's three lines (box B1
        of `docs/archive/plan-meta-game.md`), and each is null or empty when there is
        nothing to say, so the page leaves the line out rather than drawing a blank. They
        are worked out here because the page computes nothing: *highest-rated person* is
        `milestones.highest_people`, the same reading the profile's count is, and never a
        sort the page does over ratings.
        """
        with self._connection() as connection:
            found = browse.columns(connection, [digest])[digest]
            standing = found.as_dict()
            table = rating.load(connection, digest)
            named = unbeaten.handles(connection)
            standing["siteRatings"] = {
                player.key.agent: player.rating
                for player in rating.computers(table)
            }
            # L2 (`docs/archive/plan-leaderboard.md` §2.1, §2.5): the whole table as the page's
            # *Leaderboard* draws it -- computer players by the picker's name, people by
            # handle, strongest first, the marks decided here -- so the page sorts nothing.
            roster = self.roster(digest)
            # A library game is one read from `rulesets/` or `archive/`: the listing on
            # disk also holds the drafts a save wrote under `runs/designed/`, which are
            # members' games, rated on the site at publish (L5, `box`).
            card = self.known().get(digest)
            standing["leaderboard"] = leaderboard.shown(
                table,
                named=named,
                roster=roster,
                viewer=self.viewer,
                library=card is not None and card.source in drafts.CORPUS_SOURCES,
                box=spine.state(connection, digest),
            )
            # §3.3 of `docs/reference/plain-words.md` (the audit's P6): signed in, each computer
            # player's distance from the reader, so the picker says *about 240 above
            # you* rather than a signed number. The subtraction is here, never on the
            # page; positive is above the reader. Absent until the reader has a rating
            # on this game.
            reader = (
                table.get(rating.person(self.viewer)) if self.viewer is not None else None
            )
            if reader is not None:
                standing["yourGaps"] = {
                    player.key.agent: player.rating - reader.rating
                    for player in rating.computers(table)
                }
            rated_rows = matches.leaderboard_matches(connection, digest)
            standing["firstPlayed"] = _first_played(rated_rows, named)
            standing["highest"] = _highest(table, named)
            standing["clearedBy"] = [
                {
                    "handle": clear.handle,
                    "when": None if clear.when is None else clear.when.isoformat(),
                    "seat": clear.seat,
                }
                for clear in found.tag.reached
            ]
            # Q653 (`docs/reference/opponent-line.md` §1): the opponent line, read live from the
            # people's counts the trust line reads, and moving neither it nor the score.
            standing["opponentLine"] = opponent_line.shown(
                trust_people.counts(connection, digest), rated=bool(found.tag.rated)
            )
            # H7 (`docs/reference/human-data.md` §3.2): each computer player's place among the
            # people rated here, drawn in the opponent line's *more*, and only once five
            # people are established on this table (`people_scale.MINIMUM_ESTABLISHED`).
            standing["peopleScale"] = people_scale.shown(
                people_scale.places_from(
                    digest,
                    rated_rows,
                    table,
                    rating.seeds_for(connection, digest),
                )
            )
            standing["signedIn"] = self.viewer is not None
            standing["admin"] = self.role == admin.ADMIN
            # R9b (`docs/plan-website.md` *Proven positions*): the game's proven positions
            # and this reader's own marks on them, or None where the game has no record.
            standing["proven"] = self._proven_section(connection, digest)
            # Q599 (`docs/archive/plan-return.md` §5.1): the fixed tags a vote may carry, in the
            # page's order and words, and how many votes carry each, for everybody. A
            # count of a quality and never who; no text is kept with a vote.
            standing["voteTags"] = [
                {"tag": key, "words": words} for key, words in votes.TAGS.items()
            ]
            standing["mostTags"] = votes.MOST_TAGS
            standing["tagCounts"] = votes.tag_counts(connection, digest)
            # L11 of `docs/archive/plan-launch.md` (§5.2): the players a start would be refused
            # right now because every strong seat is taken, each with the words the
            # opponent list puts after its name, so the list says *busy* before anybody
            # picks it. Only the ladder's rungs reach `quota.STRONG_NODES`; every plan and
            # searched table plays at 6,400. Empty when a seat is free.
            standing["busy"] = (
                {
                    entry.agent: quota.BUSY_SAID
                    for entry in opponents.ladder()
                    if quota.is_strong(entry.agent)
                }
                if quota.busy(connection)
                else {}
            )
            # L15 (`docs/archive/plan-launch.md` §3.6): the players a visitor's start would be
            # refused, because only signed-in people play anything stronger than *hard*,
            # each with the words the opponent list puts after its name, so the list says
            # so before anybody picks it. Empty for a signed-in person.
            standing["signInFor"] = (
                {
                    entry.agent: matches.SIGN_IN_TO_PLAY
                    for entry in roster
                    if quota.members_only(entry.agent)
                }
                if self.viewer is None
                else {}
            )
            # Whether this game's rules have left the tree, and whether the person
            # reading may take them away for good. The page draws a *retired* label and
            # no play button from the first, and the delete button from the second; both
            # are the server's answer, because `admin.GameState.may_delete` is the one
            # place who-may-delete is decided and the route asks it again on the press.
            state = admin.state_of(connection, digest)
            standing["retired"] = state is not None and state.retired is not None
            standing["canDelete"] = state is not None and state.may_delete(
                self.viewer, self.role == admin.ADMIN
            )
            # Box E4 of `docs/archive/plan-meta-game.md`. The badges this game itself
            # holds. A player's own badge is filed under the game too and is that
            # person's line on their own profile, so `on_the_game` is what keeps it off a
            # page about the game. Empty on a game with none, and the page then draws
            # nothing.
            standing["badges"] = [
                badges.line(award)
                for award in badges.on_the_game(
                    awards.held(connection, game=digest),
                    badges.designer_of(connection, digest),
                )
            ]
            if self.viewer is not None:
                mine = unbeaten.record(connection, digest, self.viewer)
                standing["yourRecord"] = _record(mine)
                # R7 (`docs/archive/plan-return.md` §3 C3): every computer player on the table,
                # weakest first, with this person's best result against it. Empty on a
                # game with no ratings, and the page then draws no row.
                standing["yourLadder"] = [
                    {"agent": agent, "name": opponents.table_name(agent), "mark": mark}
                    for agent, mark in unbeaten.ladder_from(
                        digest,
                        self.viewer,
                        matches.leaderboard_matches(connection, digest, user=self.viewer),
                        table,
                    )
                ]
                # B2 of `docs/archive/plan-ux.md`. This person's unfinished rated games
                # on this game, so the page can offer each one back. Every one of them
                # opens: a killed or a deleted game has no page for this list to be on.
                standing["yourOpen"] = [
                    _carry_on(found, self.viewer)
                    for found in matches.leaderboard_matches(
                        connection, digest, user=self.viewer, running=True
                    )
                    if found.running and not correspondence.between_people(found)
                ]
                vote = votes.of(connection, self.viewer, digest)
                standing["yourVote"] = None if vote is None else {
                    "stars": vote.stars, "tags": list(vote.tags),
                }
                # Q503d. The quiet *report this game* link under the rules, or the line
                # saying this person already did; one report a person a game.
                filed = reports.of(connection, self.viewer, digest)
                standing["yourReport"] = None if filed is None else {
                    "reason": filed.reason, "label": filed.label,
                }
                standing["reportReasons"] = reports.reasons()
                # Q571. *Play a person*: who is open to a game of this one, whether the
                # viewer is, the members to suggest, and whether it would be rated.
                if not standing["retired"]:
                    standing["people"] = challenges.people(connection, digest, self.viewer)
            # Q503f. §3.3's sentence beside the stars: whether this person's vote counts
            # toward the shortlist. Empty off a trial game and for a visitor.
            standing["voteCounts"] = shortlist.vote_counts(connection, digest, self.viewer)
            checked = self._provisional(connection, [digest])
            # Q503h. A round that is due opens or closes on the way, so a nominee's
            # sentence is never a week behind when no timer runs `site sweep`.
            rounds.sweep(connection, wait=False)
            row = stages.read(connection, [digest]).get(digest)
            # Q503f. The shortlist's bar, counted from rows, under *Where it stands*; only a
            # trial game has one.
            bar = shortlist.panel(
                connection, digest, self.viewer, self._counted(connection, digest)
            )
            # Q503i and Q503n. The level's sentence knowing the game: since when a game
            # rests and whether it can come back, and since when a game is in the Baseline
            # and the maintainer's reason.
            says = None if row is None else stages.said(connection, row, self.viewer)
            # Q503h. The yes and no, on a nominee while members vote on it.
            voting = rounds.on_the_game(connection, digest, self.viewer)
            designer = _designer_marks(connection, digest, named)
        # Beside `score` rather than inside `standing`, because it is the other answer to
        # *how it scored* and the page draws it in that panel. The level (Q503b) sits at
        # the top, over the file's word `library.detail` put there, because the panel
        # that draws it, *Where it stands*, reads the same keys on a checkout.
        answer = {
            "standing": standing,
            "provisional": checked.get(digest),
            "shortlist": bar,
            "round": voting,
            "designer": designer,
        }
        if row is not None:
            answer.update(stages.shown(row["level"], says))
        return answer

    def _whats_new(self) -> Any:
        """*What's new*: `docs/changelog.md`, with the Baseline's moves filed in (Q503n).

        A game joins or leaves the Baseline by the maintainer's hand in the tree, and the
        site learns of it at the next `site load` (`stages.moves`). The file cannot say
        when that load ran, so the store's rows are filed among the file's days by date.
        """
        # An entry marked for maintainers (a command, a folder) is not a visitor's news (W8).
        payload = changelog.for_visitors(super().get("/api/changelog"))
        with self._connection() as connection:
            rounds.sweep(connection, wait=False)
            found: list[stages.Notice] = [*stages.moves(connection), *rounds.news(connection)]
        return stages.whats_new(payload, found)

    def _round(self, day: Any) -> Any:
        """The round page's answer (Q503h, §3.3): one round, or the latest with `None`.

        Anybody may read it, as anybody may read the library. A due round opens or closes
        first. Counts are in it only after the close, and this person's own vote only.
        """
        with self._connection() as connection:
            rounds.sweep(connection, wait=False)
            return rounds.page(connection, day, self.viewer)

    def _round_vote(self, day: Any, body: dict[str, Any]) -> Any:
        """This member's yes or no on one entrant, answered with the page redrawn."""
        viewer = self._viewer()
        yes = body.get("yes")
        if not isinstance(yes, bool):
            raise rounds.RoundError("A vote is yes or no.")
        with self._connection() as connection:
            rounds.sweep(connection, wait=False)
            said = rounds.vote(connection, day, str(body.get("hash") or ""), viewer, yes)
        with self._connection() as connection:
            page = rounds.page(connection, day, viewer)
        return {**page, "said": said}

    def _provisional(
        self, connection: Any, digests: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """The provisional results this person may see on these games (Q312).

        Judged against the basis the server would stamp on its own one-run measurement
        of each game now, which is what a visitor's browser measures.
        """
        return provisional.shown(connection, digests, self.viewer, self._current)

    def _current(self, digest: str, spec: Any) -> Basis | None:
        """The basis a browser's reading of this game is judged against now.

        The runner's (`checks.basis`), so the number the page shows, the reading
        *Publish* needs and the stamp tier one held a submission to are one basis; a test
        holds all three to its small roster by replacing that one function. `None` where
        the store holds no rules, as `_one_run_basis` says.
        """
        if not isinstance(spec, dict):
            return None
        return self.checks.basis(digest, spec)

    def _counted(self, connection: Any, digest: str) -> Any:
        """The reading the shortlist's bar counts: spot-checked and current, or stale for
        less than `quota.STALE_GRACE_DAYS` (`provisional.counted`, §2.4's grace)."""
        return provisional.counted(connection, digest, self._current, quota.STALE_GRACE_DAYS)

    def _playable(self, digest: str) -> None:
        """Refuse a killed or a deleted game its own page, with the 404 an unknown hash gets.

        The listing drops it and `matches.start` refuses it, and this is the door between
        them: a hash somebody bookmarked opens nothing. A `KeyError`, so the answer is
        the one the local application gives for a ruleset it does not have.

        A deleted game is refused here for a different reason from a killed one. Killing
        is a policy an admin can undo; deleting took the rules away, so there is no page
        to draw and no starting point for a design. A retired game still passes: its row
        holds its rules and `self.game` draws its page from them. One query for both, as
        `admin.unopenable_games`, because this runs on every game page.
        """
        with self._connection() as connection:
            if digest in admin.unopenable_games(connection):
                raise KeyError(f"no ruleset with hash {digest}")

    def _report(self, digest: str, body: dict[str, Any]) -> Any:
        """File this person's report of a game, from its page (Q503d).

        A killed or deleted game is the 404 an unknown one is, as on its page: there is
        nothing left on the site to report. So is a hash the store has no row for. The
        report itself does nothing but file the row; `/admin` lists it beside the kill
        switch, and killing stays the maintainer's act.
        """
        viewer = self._viewer()
        self._playable(digest)
        with self._connection() as connection:
            if admin.state_of(connection, digest) is None:
                raise KeyError(f"no ruleset with hash {digest}")
            filed = reports.file(
                connection,
                viewer,
                digest,
                str(body.get("reason") or ""),
                str(body.get("note") or ""),
            )
        return {
            "hash": digest,
            "reason": filed.reason,
            "label": filed.label,
            "said": "Reported. The person who runs the site reads every report; thank you.",
        }

    def _put_forward(self, digest: str) -> Any:
        """Put this person's trial game on the shortlist, or say why not (Q503f).

        A killed or deleted game is the 404 an unknown one is, as on its page. The reading
        is the one the bar counts (`_counted`): spot-checked on today's basis, or stale for
        less than the 30 days §2.4 allows. The answer carries the new level and its track,
        so the page redraws *Where it stands* from the server's words.
        """
        viewer = self._viewer()
        self._playable(digest)
        with self._connection() as connection:
            done = shortlist.nominate(connection, digest, viewer, self._counted(connection, digest))
            row = stages.read(connection, [digest]).get(digest)
        level = stages.SHORTLISTED if row is None else str(row["level"])
        return {"hash": done.hash, "said": done.said, "shortlist": None, **stages.shown(level)}

    def _delete_game(self, digest: str) -> Any:
        """Take a retired game's rules away for good, for its creator or an admin.

        The button is on the game's own page rather than on `/admin`, because a creator
        never sees `/admin` and the page is where both of them are already looking. The
        press confirms in the browser and this is the only server-side act; there is no
        undo route, and `admin.delete_game` is where every refusal is worded.

        The answer carries the name the game had, which is the last time it is said.
        """
        with self._connection() as connection:
            was = admin.delete_game(
                connection, digest, self.viewer, self.role == admin.ADMIN
            )
        return {"hash": digest, "was": was, "name": admin.DELETED_NAME}

    def game(self, digest: str) -> library.Game:
        """One game, from the tree if it is there and from the store if it is not.

        The fallback is a retired game and nothing else. Its spec has left `rulesets/`
        and `archive/`, so the local application's reading of the tree raises, and the
        store's copy of the rules is what the page is drawn from (`site/kept.py`). A hash
        neither holds keeps the `KeyError` it already had.
        """
        try:
            return super().game(digest)
        except KeyError:
            with self._connection() as connection:
                found = kept.rules_of(connection, digest)
            if found is None:
                raise
            name, spec = found
            return kept.game_from(digest, name, spec, self.runs_dir)

    def game_page(self, digest: str) -> str | None:
        """G3: the page a game's own address answers, or `None` for the 404.

        The same bytes for everybody, so no cookie is read: `site/game_page.py` decides what
        a visitor is shown and `web/page/game.py` draws it.
        """
        with self._connection() as connection:
            found = game_page.drawn(
                connection, digest, self.known(), self.game, self.near_cache,
                seo.site_base(), self._near_names, self.pictures.address,
            )
        return None if found is None else game_drawing.draw(found)

    def game_picture(self, digest: str, key: str) -> bytes | None:
        """G6: a game's link-preview picture, or `None` for the 404: a game a visitor cannot
        open (`game_page.drawn`'s doors), or a picture this site has not made."""
        with self._connection() as connection:
            row = game_page.row_of(connection, digest)
        if (row is None and digest not in self.known()) or (
                row is not None and not row.openable):
            return None
        return self.pictures.served(digest, key)

    def _near_names(self, game: library.Game) -> tuple[game_drawing.Near, ...]:
        """The games a game's own address names as near it: `near`'s, by name alone."""
        return tuple(
            game_drawing.Near(hash=digest, name=name)
            for digest, name in library.near_names(
                game, self.mapped(self.known().values()), self.runs_dir
            )
        )

    # ------------------------------------------------------- the doors for crawlers (G4)

    def listed_pages(self) -> tuple[front.Listed, ...]:
        """The games whose own address is indexable, reused `game_page.LISTED_SECONDS`."""

        def read() -> tuple[front.Listed, ...]:
            with self._connection() as connection:
                return game_page.listed_pages(connection, self.known())

        return self.listing.get(read)

    def home_page(self) -> str:
        """`/`: the single page, with its head and the landing's headline drawn in."""
        return front.home(seo.site_base())

    def games_page(self) -> str:
        """`/game`: the single page, with its head and every listed game drawn in for a
        reader without script."""
        listed = self.listed_pages()
        known = self.known()
        # G6: every listed game's picture, asked for once per listing, so the deploy's warm
        # read of this page draws the library's pictures before anybody shares one.
        self.pictures.want_all(listed, (
            (one.hash, known[one.hash]) for one in listed if one.hash in known))
        return front.games(seo.site_base(), listed)

    def sitemap(self, base: str) -> str:
        """`sitemap.xml`, its addresses written on `base`."""
        return seo.sitemap(base, self.listed_pages())

    def llms(self, base: str) -> str:
        """`llms.txt` (G8), its addresses written on `base`."""
        return seo.llms(base, self.listed_pages())

    def rules_page(self) -> bytes:
        """`/rules` (G8): the rule language in the site's shell. The reference inside it is
        drawn once a process (`rules_page.drawn`); the shell around it is a few lines."""
        return pages.rules().encode("utf-8")

    def findable(self, games: Iterable[library.Game]) -> set[str]:
        """Which of these games this person may be shown: the store's answer, not the tree's.

        What `App._between` asks of a gap's two ends. It used to ask whether each was a
        game the map draws, which kept everybody's drafts out -- the owner's included --
        and let a game an admin had killed in. `finding.findable` is the store's predicate:
        the library, this person's own games whatever state they are in, every live game
        for an admin, and a killed or deleted game for nobody. So the game between a person's own draft and a library game can be built,
        and nobody can build one out of somebody else's unlisted draft.
        """
        with self._connection() as connection:
            return finding.findable(
                connection,
                self.viewer,
                self.role == admin.ADMIN,
                among=[game.hash for game in games],
            )

    def graph(self) -> dict[str, Any]:
        """The map, without the games the store has taken off the library.

        `web/server.py` draws the committed and archived rulesets and reads them from the
        tree, which knows nothing of the kill switch. So a game an admin killed kept its
        dot, its name and every rule difference on a public map -- the first of the two
        things earlier stages of `docs/archive/plan-game-graph.md` left for G7. The store
        is asked now. A retired or deleted game has no spec in the tree and was never
        drawn; a killed one was, and is not any more.

        A person's own draft is still not here. Putting one on the map means a per-viewer
        layout, a per-viewer cache and a payload with somebody's rules in it, which is a
        stage of its own -- *What is left* in `docs/archive/plan-game-graph.md` says so, and the
        table this stage builds is what it will be built on.
        """
        from chess_variant_sandbox.web import graphing

        return graphing.graph(
            self.mapped(self.refresh()), self.runs_dir, overlays=[browse.OVERLAY]
        )

    def mapped(self, games: Iterable[library.Game]) -> list[library.Game]:
        """The games the map draws, less what the store has taken off the library.

        The one population for the map and for every near list on the site -- a game's
        page, its own address and a draft's gap button (M5 of `docs/plan-map.md`) -- so a
        killed game is named on none of them, and the map is one answer for everybody: no
        list names a person's unlisted draft to anybody but, on its own page, its owner.
        """
        with self._connection() as connection:
            gone = admin.unlisted_games(connection)
        return [game for game in super().mapped(games) if game.hash not in gone]

    def _graph_people(self) -> dict[str, Any]:
        """The site's own two numbers for every game the map draws.

        `docs/archive/plan-game-graph.md` §8.3, G12. `/api/graph` is one answer for
        everybody and warm; this is what a checkout cannot know -- what people scored the
        games out of five and how many leaderboard games they have finished on them -- and
        it is answered per request, because *untried* is this person's own.

        **Which games is not this route's choice.** It asks `graph` which nodes it drew
        and reads the store about those and nothing else, so a killed game and a draft
        are out of it because they were never on the map. No hash comes in with the
        request, so nobody can ask it about a game they may not be shown, and G7's
        visibility rule is untouched.

        **A node is a game, not a file.** A node carries every spelling merged into it,
        and `browse.people` adds their matches and their votes. The page is handed one
        record per dot and averages nothing.
        """
        drawn = self.graph()
        nodes = {
            str(node["hash"]): [str(one) for one in node["hashes"]]
            for node in drawn["nodes"]
        }
        spellings = sorted({one for names in nodes.values() for one in names})
        with self._connection() as connection:
            found = browse.columns(connection, spellings, viewer=self.viewer)
        payload = browse.people(nodes, found, viewer=self.viewer)
        return {"signedIn": self.viewer is not None, **payload}

    def roster(self, ruleset_hash: str) -> list[opponents.Opponent]:
        """Who plays this game, read against the same rules the page is drawn from.

        The local application resolves the spec out of the last listing, which a retired
        game is not in. Without it the stored opponents and the outside engine would be
        judged against no rules at all, and the panel would name players this game cannot
        offer. Nobody can start a match on a retired game -- `matches.start` refuses it --
        so this is about the page telling the truth rather than about seating anybody.
        """
        if ruleset_hash in self.known():
            return super().roster(ruleset_hash)
        with self._connection() as connection:
            found = kept.rules_of(connection, ruleset_hash)
        spec = None if found is None else found[1]
        return opponents.for_game(ruleset_hash, self.heuristics_dir, spec=spec)

    def _me(self) -> Any:
        """What the header asks on every page load: signed in or not, the handle, the role.

        Q199. The header used to call `_profile`, which reads this person's record on
        every rated game -- a query the header has no use for. This is the same cookie
        `identify` already resolved, so a visitor costs nothing further and a signed-in
        person costs the one query for their handle. `accounts` is the one thing this
        adds beyond the three the row asked for: `web/server.py`'s local application has
        no store and no sign-in, and answers with the same shape and `accounts: False`,
        which is how the header tells the two apart without a 404 to read.

        `players` is whether the footer links `/players` (L3 of `docs/archive/plan-launch.md`
        §3.3): only once it lists `milestones.LISTED_FROM` people. It and `round` are the
        readings here a visitor costs, two queries, because each link is the
        same for everybody who reads the footer. `round` is the same kind of fact (L4,
        §3.4): the footer links the round only while the latest round has an entrant.
        """
        handle = None
        with self._connection() as connection:
            if self.viewer is not None:
                handle = identity.handle_of(connection, self.viewer)
            players = milestones.players_listed(matches.leaderboard_matches(connection))
            round_on = rounds.announced(connection)
        return {
            "signedIn": self.viewer is not None,
            "handle": handle,
            # The reader's own mark (R2), for the header's corner.
            "mark": identity.mark_of(self.viewer, handle),
            "admin": self.role == admin.ADMIN,
            "accounts": True,
            "players": players,
            "round": round_on,
        }

    def players_listed(self) -> bool:
        """Whether `/players` lists people yet, and so whether a footer links it (L3 of
        `docs/archive/plan-launch.md` §3.3, `milestones.LISTED_FROM`)."""
        with self._connection() as connection:
            return milestones.players_listed(matches.leaderboard_matches(connection))

    def _profile(self) -> Any:
        """§5's three counts, this person's record on every rated game, and box A2's lines.

        The page that draws it is 1.3's, because a profile needs somebody signed in; the
        reading is here so that page has nothing left to work out.

        A game that has since been retired keeps its row here, marked. Somebody's games
        are theirs, and dropping them would make a profile disagree with the counts above
        it; the mark is what explains a game the library no longer offers. A game whose
        rules were deleted keeps its row for the same reason and reads as *deleted game*,
        which is the name the store holds for it -- nobody's counts move when a game's
        rules go.

        `milestones` and `stillOpen` are the meta-game's, box A2 of
        `docs/archive/plan-meta-game.md`.
        `milestones.read_from` wants every leaderboard row on the site rather than this
        person's, because *the first human game here* and the *Unbeaten* tag are decided
        by other people's games, so the whole set is read once here and handed to the tag
        reader as well. `rungs` and `distance` are the table's two new columns, and both
        read this person's rows alone.

        Every game named under `stillOpen` is a rated game, and `games` carries every
        rated game, so the page resolves a hash to a name from the payload it already
        holds rather than being sent each name twice.

        `carryOn` is B2 of `docs/archive/plan-ux.md`: this person's unfinished rated
        games, anywhere on the site, each with the address that opens it. `openable` is
        false where the store will not hand the rules back -- a killed game or a deleted
        one, the same set `_playable` refuses a page to -- and the page then names the
        game and draws no link, because a board cannot be drawn from rules that are not
        there. A retired game is openable: its row still holds its rules, so the game can
        be finished.
        """
        viewer = self._viewer()
        with self._connection() as connection:
            tables = rating.load_many(connection)
            rated = milestones._rated(tables)
            everyone = matches.leaderboard_matches(connection)
            mine = matches.leaderboard_matches(connection, user=viewer)
            running = [
                row
                for row in matches.leaderboard_matches(
                    connection, user=viewer, running=True
                )
                # A game against another person is listed under its own heading (Q570).
                if row.running and not correspondence.between_people(row)
            ]
            clock = correspondence.now(connection)
            between = correspondence.started_order([
                _between_entry(connection, row, viewer, clock)
                for row in correspondence.running_for(connection, viewer)
            ])
            finished_line = correspondence.finished_line(connection, viewer)
            asked = _challenges_of(connection, viewer, waiting=False)
            move_box = _move_box(connection, viewer)
            favourites_box = members.favourites_box(connection, viewer)
            maker_box = challenges.maker_box(connection, viewer)
            invites_box = inviting.box(connection, viewer)
            found = {game: unbeaten.record_from(game, viewer, mine) for game in rated}
            counts = unbeaten.counts_from(found.values())
            people = unbeaten.handles(connection)
            tags = unbeaten.tags_from(rated, everyone, tables, people)
            # D1 to D4: this person's designs and the forks of them, and the signed-in
            # seats on their designs (`site/designs.py`).
            made = designs.of(connection, viewer)
            reading = milestones.read_from(
                viewer, everyone, tables, badges.cells(connection, rated),
                made,
                designs.tries(connection, [d.hash for d in made if d.owner == viewer]),
            )
            handle = identity.handle_of(connection, viewer)
            held = awards.held(connection, holder=viewer)
            # A badge can sit on a game the ladder does not, and so can an open match on
            # a game whose ladder has since gone, so the names are asked for the three
            # sets together rather than per reading.
            named = browse.names(
                connection,
                sorted({
                    *rated,
                    *(a.game for a in held if a.game),
                    *(row.game for row in running),
                    *(entry["game"] for entry in between),
                    *(line.game for line in reading.milestones),
                }),
            )
            retired = admin.retired_games(connection)
            gone = admin.deleted_games(connection)
            shut = admin.unopenable_games(connection)
            back = drafts.returned(connection, viewer)
            designed = stages.yours(connection, viewer)
            # Q599. Each of this person's games with the fixed tags its voters chose,
            # counted; never who, and no text.
            tagged = votes.tag_counts_by_game(
                connection, [game["hash"] for group in designed for game in group["games"]]
            )
            for group in designed:
                for game in group["games"]:
                    game["tagCounts"] = tagged.get(game["hash"], [])
            # R9b: this person's own solved marks on proven positions, by game.
            marked = proven.games_marked(connection, viewer)
            proven_names = browse.names(connection, sorted(marked))
        known = self.known()
        proven_records = {
            game: proven.load(self.runs_dir, game, known[game].spec if game in known else None)
            for game in marked
        }
        proven_lines = proven.profile_lines(marked, proven_records, proven_names)
        # Q993b: the share of every proven position on the site this person has solved,
        # across every game that offers some, counted live against what each offers now.
        proven_share = proven.overall(
            marked, proven.everywhere(self.runs_dir, {d: g.spec for d, g in known.items()})
        ) if marked else None
        return {
            "user": viewer,
            # R9b. On each game this person has solved a proven position on, how many they
            # converted and how many moves they found, of how many it offers. Theirs alone.
            "proven": proven_lines,
            "provenShare": proven_share,
            # Q503b. *Your games*: the games this person made, grouped by level, each
            # with its next step (`stages.yours`).
            "yourGames": designed,
            # Q503a. This person's games a save had listed, which went back to Draft when
            # saving stopped listing; the page says why under its own heading.
            "backToDraft": [
                {"hash": game.hash, "name": game.name, "when": game.when.isoformat()}
                for game in back
            ],
            # The games whose rules were deleted, as hashes. The page prints their name
            # rather than a link, because there is no page left to link to; the rows
            # themselves are this person's and stay exactly where they were.
            "deleted": sorted(gone),
            "handle": handle,
            "counts": {
                "rated": counts.rated,
                "tried": counts.tried,
                "won": counts.won,
                "top": counts.top,
                # S2's count, the games where this person is the highest-rated person.
                "highest": reading.highest,
            },
            "games": [
                {"hash": game, "name": named.get(game, ""), **_record(record),
                 "unbeaten": tags[game].unbeaten, "top": list(tags[game].top),
                 "retired": game in retired,
                 "rungs": _rungs(reading.rungs.get(game)),
                 "distance": _distance(reading.distance.get(game))}
                for game, record in sorted(found.items())
            ],
            "milestones": [
                {
                    "kind": line.kind,
                    "rule": milestones.RULES[line.kind],
                    "when": line.when.isoformat(),
                    "game": line.game,
                    "name": named.get(line.game, ""),
                    # A designer's line (D1 to D4) has no seat.
                    "seat": line.seat if line.seat >= 0 else None,
                    # D4's *by ‹handle›*; somebody who has left reads `identity.FORMER`.
                    **({"by": people.get(line.by, identity.FORMER)}
                       if line.by is not None else {}),
                }
                for line in reading.milestones
            ],
            # Box E4 of `docs/archive/plan-meta-game.md`. What this person holds, each
            # with its sentence, its rule and the figures the row kept. Unearned badges
            # are not here at all: the plain list at /badges is where a rule is read
            # before it is met (`docs/archive/plan-meta-game.md` §2.9).
            "badges": [
                {**badges.line(award), "name_of": named.get(award.game or "", "")}
                for award in held
            ],
            # B2 of `docs/archive/plan-ux.md`. The games this person left running, newest
            # first, so the one they were last at is the first one offered back.
            "carryOn": [
                {
                    **_carry_on(row, viewer),
                    "name": named.get(row.game, ""),
                    "openable": row.game not in shut,
                }
                for row in sorted(running, key=lambda row: row.started, reverse=True)
            ],
            # Q570, Q572. Games between people still being played, each with whose move
            # it is (the engine's side to move), in `correspondence.started_order`, and
            # the line of what this person finished (§1.6), absent until there is a
            # finished one. The page draws them under *Correspondence*: *Your move*,
            # then *Their move*.
            "between": {
                "running": [
                    {**entry, "name": named.get(entry["game"], "")} for entry in between
                ],
                "finished": finished_line,
                "rule": correspondence.RULES[-1],
            },
            # Q571. *A challenge for you*, *Challenges you sent* and *Open to a game of*
            # (`docs/archive/plan-return.md` §1.4, s6 of the mock).
            "challenges": asked,
            # Q572. The one email's box, off by default, and its sentences.
            "moveEmail": move_box,
            # Q596. Your page as members see it, and its one box, off by default.
            "page": members.address(handle) if handle and handle != identity.FORMER else None,
            "favourites": favourites_box,
            # Q606. *List me on the games I made as taking challengers*, off by default;
            # drawn only for somebody with a published game of their own.
            "takesChallengers": maker_box,
            # Q719. *Invite somebody*: how many invitations `/admin` left this person to
            # send, absent for somebody never given any.
            "invites": invites_box,
            # S6 of `docs/archive/plan-understanding.md` §3.3: each axis headed by the question
            # it asks and each kind in the game page's own words (`kind_words`), the
            # grid's bin keys kept beside them for the reader that wants a key.
            "stillOpen": [
                {
                    "axis": axis.axis,
                    "asks": kind_words(axis.axis)[0],
                    "kinds": [
                        {"bin": kind.bin, "words": kind_words(axis.axis, kind.bin)[1],
                         "tried": kind.tried,
                         "untried": list(kind.untried), "unwon": list(kind.unwon),
                         "unbeaten": list(kind.unbeaten)}
                        for kind in axis.kinds
                    ],
                }
                for axis in reading.still_open
            ],
        }

    def member(self, handle: str, viewer: int) -> dict[str, Any] | None:
        """What `/players/<handle>` draws (R1, Q596), or None for a handle with no page."""
        with self._connection() as connection:
            return members.read(connection, handle, viewer)

    def _favourites(self, body: dict[str, Any]) -> Any:
        """Tick or untick *show my five-star games on my page* (R1); the box as it stands."""
        viewer = self._signed_in()
        with self._connection() as connection:
            members.set_favourites(connection, viewer, bool(body.get("on")))
            return members.favourites_box(connection, viewer)

    def _players(self) -> Any:
        """S3's overall page, box A3 of `docs/archive/plan-meta-game.md`.

        Two readings and nothing else. `unclaimed` is what nobody has taken -- the games
        with no counted game on them and the games still *Unbeaten* -- and it leads the
        page, which is half of what makes the listing pass §3's fifth test. `players` is
        everybody with a finished game beside their three counts, and somebody with none
        is absent rather than shown at nought.

        The counts are read over every rated game, exactly as a profile's are, so one
        person reads the same on both pages. What is unclaimed is read over the games
        that can still be started instead: a killed, retired or deleted game is on
        nobody's list of what to go and claim, which is `admin.unlisted_games`, the same
        set the library leaves out.

        Names come with the unclaimed hashes, because the page holds hashes and nothing
        else and a person picking a game to play should read *Courier chess*.

        L3 of `docs/archive/plan-launch.md` §3.3: below `milestones.LISTED_FROM` people the rows
        are not sent at all, so nothing ranked reaches the page; `members` and `listedFrom`
        are what it says instead -- who is here, and when the list starts.
        """
        viewer = self._viewer()
        with self._connection() as connection:
            tables = rating.load_many(connection)
            rated = milestones._rated(tables)
            everyone = matches.leaderboard_matches(connection)
            unlisted = admin.unlisted_games(connection)
            offered = tuple(game for game in rated if game not in unlisted)
            open_now = milestones.unclaimed_from(everyone, tables, offered)
            standings = milestones.standings_from(
                everyone, tables, unbeaten.handles(connection)
            )
            named = browse.names(
                connection, sorted({*open_now.unplayed, *open_now.unbeaten})
            )
            members_here = identity.members(connection)
            paged = members.with_pages(connection)
            # Q760: every member, A to Z, for finding one by name.
            everybody = members.directory(connection, viewer)
        listed = len(standings) >= milestones.LISTED_FROM
        return {
            "you": viewer,
            "directory": everybody,
            "directorySaid": members.DIRECTORY_SAID,
            "rated": len(rated),
            "listed": listed,
            "listedFrom": milestones.LISTED_FROM,
            "members": members_here,
            "unclaimed": {
                "unplayed": [_named(game, named) for game in open_now.unplayed],
                "unbeaten": [_named(game, named) for game in open_now.unbeaten],
                "offered": open_now.offered,
            },
            "players": [
                {
                    "user": standing.user,
                    "handle": standing.handle,
                    "mark": identity.mark_of(standing.user, standing.handle),
                    # Their page (R1); none for somebody who left.
                    "page": (
                        members.address(standing.handle)
                        if standing.user in paged else None
                    ),
                    "tried": standing.counts.tried,
                    "won": standing.counts.won,
                    "top": standing.counts.top,
                }
                for standing in standings
            ] if listed else [],
        }

    # ----------------------------------------------------------------- who is playing

    def _join(self, body: dict[str, Any]) -> Any:
        """Spend an invitation and sign the new person in, in one transaction.

        A handle somebody has, or a password too short, leaves the invitation unspent:
        the refusal comes back before the row is written and the whole unit rolls back.

        An invitation code is a guessable credential, so this route meets the same
        per-address limit `/api/signin` does (Q369). By address alone: the code names no
        account, and counting against the handle being claimed would let a stranger spend
        somebody else's allowance by choosing their name.
        """
        self.signins.check(self.source)
        try:
            with self._connection() as connection:
                # The maintenance account's handle is spoken for whether or not its row
                # has been written yet. `accept` refuses it where and as it refuses a
                # held handle -- after the code and the password -- because an earlier or
                # a different answer would tell a stranger which handle to guess at.
                viewer, token = identity.accept(
                    connection,
                    code=str(body.get("code") or ""),
                    handle=str(body.get("handle") or ""),
                    password=str(body.get("password") or ""),
                    reserved=(admin.account_name(),),
                )
        except identity.Refused:
            self.signins.failed(self.source)
            raise
        self.signins.passed(self.source)
        self.viewer = viewer.id
        self.grant = Grant(token)
        return {"handle": viewer.handle}

    def _signin(self, body: dict[str, Any]) -> Any:
        """A handle and a password back, for a session.

        The maintainer's account is the environment's (`admin.ensure_account`), and it is
        reconciled here, and committed, before the password is checked. Committed first
        because a refused sign-in rolls the rest back: a box whose `ADMIN_PASSWORD` has
        been taken away has to lose the stored hash even when the attempt that prompted
        it fails, or the old password would keep opening the door.

        **Every attempt costs the same, whichever handle it names** (Q379). The reconcile
        runs for all of them and hashes nothing, and `identity.sign_in` hashes once
        whether or not the handle is anybody's. Before, the maintenance handle alone was
        reconciled, at a second scrypt, and a handle nobody held cost none, so the time
        an answer took said which was which.

        The limit is met before the connection is opened (Q369), so a refused attempt
        costs neither a reconcile nor the 50 ms of scrypt the check itself is.
        """
        handle = str(body.get("handle") or "")
        self.signins.check(self.source, handle)
        try:
            with self._connection() as connection:
                admin.ensure_account(connection)
                connection.commit()
                viewer, token = identity.sign_in(
                    connection,
                    handle=handle,
                    password=str(body.get("password") or ""),
                )
        except identity.Refused:
            self.signins.failed(self.source, handle)
            raise
        self.signins.passed(self.source, handle)
        self.viewer = viewer.id
        self.grant = Grant(token)
        return {"handle": viewer.handle}

    def _signout(self) -> Any:
        """Drop this session. Signing out twice is signing out once."""
        with self._connection() as connection:
            identity.sign_out(connection, getattr(self._request, "token", None))
        self.viewer = None
        self.grant = Grant(None)
        return {"signedIn": False}

    def _ask_invitation(self, body: dict[str, Any]) -> Any:
        """A visitor asks for an invitation (Q831, `site/asking.py`).

        The form is checked first, which costs nothing; then the connection's day
        (`quota.Asks`, a 429); then the site's day and the row, in one transaction. The
        answer is `asking.CONFIRMED` whether a row was written or not, so it says nothing
        about the address. A row that was written is forwarded to the owner (Q979) once
        it is committed, off this thread; nothing is mailed to the visitor.
        """
        address, note = asking.checked(body.get("email"), body.get("note"))
        self.asks.take(self.source)
        with self._connection() as connection:
            written = asking.ask(connection, address, note)
        if written:
            self._tell_owner(address, note)
        return {"said": asking.CONFIRMED}

    def _tell_owner(self, address: str, note: str | None) -> None:
        """Mail the owner a request that was just kept (Q979), on a thread of its own.

        No owner's address in the settings means nobody is told. The request is already
        committed, so whatever the sender does -- refuses, raises, hangs until its
        timeout -- costs it nothing: the failure is printed, without the owner's address,
        and the visitor's answer has already gone or goes regardless. Nothing retries;
        the request waits on the admin panel either way.
        """
        owner = mail.owner_address()
        if owner is None:
            return

        def tell() -> None:
            try:
                link = f"{mail.site_url()}{asking.GRANTED_AT}"
                self.send(mail.asked_for_invitation(owner, address, note, link))
            except Exception as error:  # noqa: BLE001 - any failure of the sender is the same
                said = str(error).replace(owner, f"${mail.OWNER_VAR}")
                print(f"a request for an invitation was kept, but the owner was not "
                      f"mailed: {said}", file=sys.stderr)

        thread = threading.Thread(target=tell, name="tell-owner", daemon=True)
        with self._telling_lock:
            self._telling = [one for one in self._telling if one.is_alive()]
            # Started under the lock, so `settle` never joins a thread not yet started.
            thread.start()
            self._telling.append(thread)

    def settle(self, timeout: float = 30.0) -> None:
        """Wait for the owner's mails in flight to be handed to the sender. For tests,
        and for anything that must not stop the process with one half sent."""
        with self._telling_lock:
            waiting = list(self._telling)
        for thread in waiting:
            thread.join(timeout)

    # ----------------------------------------------------------- the maintainer's page

    def _admin_view(self) -> Any:
        """Everything the admin page draws, in one answer.

        Many readings and one connection. The page redraws the whole thing after every
        act, because an act changes more than the row it was clicked on -- killing
        somebody empties their session count -- and a few small queries are cheaper than
        working out which cells moved. `usage` is `docs/plan-usage.md` §4.3's.

        `suggestions` is Q279's, and it is here rather than on a route of its own because
        this is where the admin gate already is. A member who asks for this path gets the
        404 an invented path gets, which is what keeps the box out of everybody's reach
        but its author's.
        """
        with self._connection() as connection:
            return {
                # Q813. The box's disk and every queue's depth and oldest age, each against
                # its threshold: the numbers `deploy/status.sh` prints (`site/depths.sql`).
                "status": depths.view(connection, self.runs_dir),
                "users": [person.as_dict() for person in admin.people(connection)],
                "invites": [one.as_dict() for one in admin.invitations(connection)],
                # Q831. Requests for an invitation nobody has answered yet, oldest
                # first, and how many were granted and declined in the kept window.
                "asks": [one.as_dict() for one in asking.pending(connection)],
                "asksAnswered": asking.answered(connection),
                "games": [game.as_dict() for game in admin.games(connection)],
                "suggestions": [
                    one.as_dict() for one in suggestions.every(connection)
                ],
                # Q503d. Every report, newest first, each with the game's kill switch.
                "reports": [one.as_dict() for one in reports.every(connection)],
                # Q518. The dismissed ones, most recently dismissed first, folded away
                # under the open list; `dismissedCount` says when only the newest show.
                **self._dismissed_reports(connection),
                "limits": [
                    {"label": label, "value": value} for label, value in quota.summary()
                ],
                "usage": usage.admin_view(connection),
                # Q721. The last fortnight by UTC day: requests, errors, games, players,
                # joins, invitations and mail, all counts from tables already kept.
                "days": daily.read(connection).as_dict(),
                # Q649. Every library score a new measurement replaced, with its basis
                # and why: the maintainer's *was* lines (`history.py`).
                "history": self._score_history(),
                # EV6 (Q690). How much each stored score has moved between revisions,
                # beside its interval (`history.volatility`): the maintainer's only.
                "volatility": self._score_volatility(),
                # Q705. The refresher's status file, in words (`web/refreshing.py`).
                "refresher": refreshing.view(self.runs_dir),
                # Q503h. The latest round, each entrant with its confirm and veto.
                "round": self._admin_round(connection),
                # Q503e. Accounts a spot check flagged, with what it found; `clear-flag`.
                "flagged": provisional.flagged_accounts(connection),
                # Q570. The pause on new games between people and how many are running
                # (`docs/archive/plan-return.md` §1.7); Q572, the move emails of the last rolling
                # day against the site-wide cap.
                "correspondence": {
                    "paused": correspondence.paused(connection),
                    "running": correspondence.running_count(connection),
                    "pausedSays": correspondence.PAUSED_SAID,
                    "mailed": move_email.sent_today(connection),
                    "mailCap": quota.MOVE_EMAILS_A_DAY,
                },
                # H7 (`docs/reference/human-data.md` §3.2). Each computer player's place among
                # the people rated on each game, and whether the game page shows it yet.
                "peopleScale": people_scale.admin_rows(connection),
                # H8 and H9 (§3.3). Every game with a rated game against its strongest,
                # against the bar, with the training queue's entry for it.
                "bar": self._admin_bar(connection),
            }

    @staticmethod
    def _admin_bar(connection: Any) -> dict[str, Any]:
        found = alarm.readings(connection)
        held = training.held(connection)
        queue = training.entries(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "select hash, name from games where hash = any(%s)",
                ([one.game for one in found] + [entry.game for entry in queue],),
            )
            names = {str(row[0]): str(row[1]) for row in cursor.fetchall()}
        return {
            "said": alarm.BAR_SAID,
            "games": [
                {
                    "hash": one.game,
                    "name": names.get(one.game, one.game[:12]),
                    "over": one.over,
                    "said": alarm.said(one),
                    "queued": None if one.game not in held else held[one.game].as_dict(),
                }
                for one in found
            ],
            "queue": [
                {**entry.as_dict(), "name": names.get(entry.game, entry.game[:12])}
                for entry in queue
            ],
        }

    def _score_history(self) -> list[dict[str, Any]]:
        """`/admin`'s *Earlier scores*: one row a game whose score replaced another."""
        from chess_variant_sandbox import history

        names = {digest: game.name for digest, game in self.known().items()}
        return history.admin_rows(self.runs_dir, names)

    def _score_volatility(self) -> list[dict[str, Any]]:
        """`/admin`'s *How much scores have moved*: every game with a stored measurement."""
        from chess_variant_sandbox import history

        names = {digest: game.name for digest, game in self.known().items()
                 if (self.runs_dir / digest).is_dir()}
        return history.volatility_rows(self.runs_dir, names)

    @staticmethod
    def _dismissed_reports(connection: Any) -> dict[str, Any]:
        shown, count = reports.dismissed(connection)
        return {
            "dismissedReports": [one.as_dict() for one in shown],
            "dismissedCount": count,
        }

    def _admin_round(self, connection: Any) -> Any:
        rounds.sweep(connection, wait=False)
        # The runs directory, so an entrant's premise check (Q502d) is read where one is.
        return rounds.admin_view(connection, runs_dir=self.runs_dir)

    def _admin_act(self, body: dict[str, Any]) -> Any:
        """One administrative act, named in the body, answered with what it did.

        A sentence back rather than a bare success, because every one of these is a thing
        done to somebody and the page shows what happened. An unknown act is a `KeyError`
        and so a 404, like any other name the site does not have.
        """
        act = str(body.get("act") or "")
        with self._connection() as connection:
            if act == "invite":
                return self._invite(connection, str(body.get("email") or ""))
            if act in ("grant-ask", "decline-ask"):
                return self._answer_ask(connection, act, body.get("ask"))
            if act == "set-invites":
                # Q719. How many invitations a member may send themselves.
                try:
                    handle, count = inviting.set_to_give(
                        connection, int(body["user"]), body.get("count")
                    )
                except inviting.InviteRefused as error:
                    raise admin.AdminError(str(error)) from error
                return {"said": f"{handle} can send {count} invitation(s) now"}
            if act == "revoke-sessions":
                gone = identity.revoke_sessions(connection)
                return {"said": f"signed out {gone} session(s); every account is intact"}
            if act == "kill-user":
                name = admin.kill_user(connection, int(body["user"]))
                return {"said": f"{name} can no longer sign in, and their sessions are gone"}
            if act == "revive-user":
                name = admin.revive_user(connection, int(body["user"]))
                return {"said": f"{name} can sign in again with the password they had"}
            if act == "forget-user":
                name = admin.forget_id(connection, int(body["user"]))
                return {"said": f"{name} has been forgotten; their games stay"}
            if act in ("pause-correspondence", "resume-correspondence"):
                paused = act == "pause-correspondence"
                correspondence.set_paused(connection, paused, by=self.viewer)
                return {
                    "said": "new games between people are paused; running ones carry on"
                    if paused
                    else "new games between people can be started again"
                }
            if act == "pair":
                # Q570's door for the maintainer, beside challenges (Q571): seat two
                # members at a game, first named moving first.
                players = [
                    correspondence.member_named(connection, str(body.get(seat) or ""))
                    for seat in ("seat0", "seat1")
                ]
                made = correspondence.start(
                    connection, str(body.get("hash") or ""), players, by=self.viewer
                )
                return {"said": f"started a game between two people at #/play/{made.id}",
                        "id": made.id}
            if act == "clear-flag":
                # Q503e. The account may publish again and its scores are spot-checked
                # again; the mismatched result stays, with its records, for the log.
                user = int(body["user"])
                if not provisional.clear_flag(connection, user):
                    raise admin.AdminError("that account is not flagged")
                return {"said": "the flag is cleared: the account can publish again"}
            if act == "kill-game":
                name = admin.kill_game(connection, str(body["hash"]))
                return {"said": f"{name} is off the site; its matches are unchanged"}
            if act == "revive-game":
                name = admin.revive_game(connection, str(body["hash"]))
                return {"said": f"{name} is back on the site"}
            # Q503o. The row stays, so the same person still cannot report the game again.
            if act == "dismiss-report":
                handle = reports.dismiss(
                    connection, int(body["reporter"]), str(body["hash"])
                )
                return {"said": f"dismissed the report from {handle}; they cannot report it again"}
            # Q279's two. Marking is reversible and deleting is not, so the page confirms
            # the second and this side only carries it out.
            if act == "done-suggestion":
                suggestions.mark(connection, int(body["suggestion"]), done=True)
                return {"said": "marked done; it stays on this page"}
            if act == "open-suggestion":
                suggestions.mark(connection, int(body["suggestion"]), done=False)
                return {"said": "put back as open"}
            # Q503h. The maintainer's confirm or veto of an entrant that passed, each with
            # its reason; the confirm answers with the `archive` command to run.
            if act in ("confirm-entrant", "veto-entrant"):
                done = rounds.decide(
                    connection,
                    str(body.get("hash") or ""),
                    act == "confirm-entrant",
                    str(body.get("reason") or ""),
                    self.viewer,
                )
                return {"said": done.said, "command": done.command}
            if act == "drop-suggestion":
                dropped = suggestions.drop(connection, int(body["suggestion"]))
                return {"said": f"deleted the {dropped.label} from {dropped.handle}"}
        raise KeyError(f"no such act: {act}")

    def _answer_ask(self, connection: Any, act: str, asked: Any) -> Any:
        """Grant or decline one request for an invitation (Q831).

        Granting sends the ordinary invitation with the admin who pressed as its issuer,
        as `_invite` does; an address that has gained an account since it asked is refused
        in `identity`'s words, as a 400, and the request stays pending. Declining sends
        nothing. The visitor is told neither.
        """
        if isinstance(asked, bool) or not isinstance(asked, int):
            raise admin.AdminError("name the request by its number")
        try:
            if act == "decline-ask":
                return {"said": f"declined the request from {asking.decline(connection, asked)}; "
                                "nothing was sent"}
            sent = asking.grant(connection, asked, mail.site_url(), self.send, self.viewer)
        except (identity.Refused, asking.AskGone) as error:
            raise admin.AdminError(str(error)) from error
        return {"said": f"invited {sent}: the link is on its way"}

    def _invite(self, connection: Any, email: str) -> Any:
        """Invite one address from the page, through the command's own function.

        The admin who pressed the button is the issuer. A refused address answers 400 like
        every other refused act, not the 401 a refused sign-in gets, because nobody's
        credentials were wrong. A send that fails raises, so `_connection` rolls the row
        back.
        """
        try:
            made, _ = identity.send_invitation(
                connection, email, mail.site_url(), self.send, issuer=self.viewer
            )
        except identity.Refused as error:
            raise admin.AdminError(str(error)) from error
        return {"said": f"invited {made.email}: the link is on its way"}

    # ------------------------------------------------------------ a draft somebody kept

    def _save_design(self, body: dict[str, Any], restate: bool = True) -> Any:
        """The playability check, then the local application's save, then the row.

        The designer badges are granted last, in the same transaction as the row they
        read, so the save and whatever it completed land together or neither does. The
        answer says nothing about them; a badge is learned from the profile and the game
        page (§3 test 3).

        The check goes first and under `screening.CHECK_SECONDS`, so a draft that outruns
        the clock leaves no file under `runs/designed/` and no row -- nothing the listing
        or a match can reach comes from a check that did not finish. It also runs before
        a connection opens, because it plays games for about a second.

        The answer says what the save did: whether the draft was kept (`created`, from
        the local application), whether it was listed and why (`listed`, `said`), and
        which rung the check reached (`rung`, keyed as `candidacy.rungs` keys them, with
        `rungPassed`). The designer prints all three beside the save button.

        A re-save of kept rules gives them its name and description only where
        `drafts.restatable` allows it -- the designer's own unpublished draft (Q595); the
        answer's `restated` names what changed, and otherwise `unchanged` names the words
        that stayed and `keptAsIs` says why.
        """
        viewer = self._viewer()
        digest = _draft_hash(body.get("spec"))
        # The two caps before the check, so a refused save costs the box nothing: room
        # for another draft (rules the store already holds need none), then the day's
        # count, which a refusal for room does not spend.
        with self._connection() as connection:
            quota.check_new_draft(connection, viewer, digest)
            frozen = drafts.restatable(connection, digest, viewer) if digest else ""
        restate = restate and not frozen
        self.saves.save(viewer)
        screened = screening.check(body.get("spec"))
        kept = super()._save_design(body, restate=restate)
        spec = load_spec(Path(str(kept["path"])))
        with self._connection() as connection:
            row = drafts.keep(
                connection,
                str(kept["hash"]),
                spec,
                owner=viewer,
                parent=str(body.get("parent") or ""),
                passed=bool(screened.get("passed")),
                restate=restate,
            )
            badges.grant_for_save(connection, viewer, row.hash)
        return {
            **kept,
            "keptAsIs": frozen if kept.get("unchanged") else "",
            "listed": row.listed,
            "visibility": row.visibility,
            "screenPassed": row.screen_passed,
            "said": drafts.why(row, screened),
            "rung": str(screened.get("rung") or ""),
            "rungPassed": bool(screened.get("passed")),
        }

    def _publish(self, body: dict[str, Any]) -> Any:
        """Put one of this person's drafts on trial, or say why not (Q503a).

        A killed or deleted game is the 404 an unknown one is, as on its page. The reading
        is the draft's spot-checked score on today's basis (`drafts.attached`, Q503e): a
        press that gets past the owner, the account's flag, the check, the same-game
        refusal (`drafts.twin`, Q503c) and the two caps without one is refused with
        `drafts.NEEDS_A_READING`.

        `terms` is the version of the terms the designer's ticked box stood for (Q722), a
        whole number and nothing else; anything else counts as no tick, which
        `drafts.publish` refuses with `terms.NOT_ACCEPTED`.

        Q742: while the site is still checking the draft's score (`provisional.checking`)
        the press is taken and waits, `listed` false and `waiting` true, and the runner is
        woken so that `drafts.settle` puts the game on trial as soon as the check passes.
        """
        viewer = self._viewer()
        digest = str(body.get("hash") or "")
        if not digest:
            raise drafts.PublishError("Say which game to publish.")
        self._playable(digest)
        with self._connection() as connection:
            reading = drafts.attached(connection, digest, self._current)
            done = drafts.publish(
                connection, digest, viewer, reading, _agreed(body.get("terms")),
                checking=reading is None and provisional.checking(connection, digest),
            )
        if done.waiting or done.rating:
            # The check may have ended between the read above and this commit; the runner
            # settles every waiting publish on each pass, so a wake is all it needs. A game
            # listed now has its five standard players to rate (L5), which the same wake
            # starts once the transaction above has committed the job.
            self.checks.wake()
        return {
            "hash": done.hash, "listed": not done.waiting, "waiting": done.waiting,
            "said": done.said,
        }

    def _vote(self, body: dict[str, Any]) -> Any:
        """One to five stars on a game, with up to three fixed tags, replacing whatever
        this person said last.

        No text (Q599, `docs/archive/plan-return.md` R4): `votes.checked_body` refuses a request
        carrying anything but the game, the stars and the tags, and `votes.fixed_tags`
        stores the module's own key for each tag chosen. The answer carries the tag
        counts the page draws under the stars.

        The badges go last and in the same transaction as the row they read, so the
        rating and whatever it completed land together or neither does. The answer says
        nothing about them and the star control shows no progress toward one, which is
        the condition §3 puts on the voting badge existing at all.
        """
        votes.checked_body(body)
        digest = str(body["hash"])
        asked = body.get("tags") or []
        if not isinstance(asked, list):
            raise votes.VoteError("tags are a list of the fixed choices")
        viewer = self._viewer()
        with self._connection() as connection:
            cast = votes.cast(
                connection,
                voter=viewer,
                game=digest,
                stars=int(body["stars"]),
                tags=[str(tag) for tag in asked],
            )
            found = votes.stars(connection, digest)
            counted = votes.tag_counts(connection, digest)
            badges.grant_for_vote(connection, viewer, digest)
        return {
            "hash": digest, "stars": found.mean, "votes": found.count,
            "tags": list(cast.tags), "tagCounts": counted,
        }

    # ------------------------------------------------------------ the suggestion box

    def _suggest(self, body: dict[str, Any]) -> Any:
        """One idea written down, by whoever is signed in. Q279.

        No badge, no mail and no queue: a suggestion is read by its author and by an
        admin, and what becomes of it is somebody's judgement rather than a mechanism.
        The daily allowance and the length cap are `site/quota.py`'s and are checked in
        `suggestions.post`, so a refusal is one sentence in the place the button was.

        The whole row comes back, so the page can put it at the top of the person's own
        list without asking again.
        """
        viewer = self._viewer()
        with self._connection() as connection:
            made = suggestions.post(
                connection,
                author=viewer,
                kind=str(body.get("kind") or ""),
                body=str(body.get("body") or ""),
            )
        return {"suggestion": made.as_dict()}

    def _suggestions(self) -> Any:
        """This person's own suggestions, and nobody else's.

        An admin reads every suggestion through `GET /api/admin`, which is the one place
        the admin gate is written down (`routes.ADMIN_ONLY`). Answering an admin with the
        whole box here would be a second gate to keep in step with the first.

        `limits` travels with the list so the form can say how long a suggestion may be
        and how many are left today before anybody presses anything.
        """
        viewer = self._viewer()
        with self._connection() as connection:
            posted = suggestions.mine(connection, viewer)
            today = quota.suggestions_today(connection, viewer)
        return {
            "kinds": [
                {"kind": kind, "label": suggestions.LABELS[kind]}
                for kind in suggestions.KINDS
            ],
            "suggestions": [one.as_dict() for one in posted],
            "limits": {
                "length": quota.SUGGESTION_LENGTH,
                "aDay": quota.SUGGESTIONS_A_DAY,
                "left": max(0, quota.SUGGESTIONS_A_DAY - today),
            },
        }

    # -------------------------------------------------------- a game, as a match row

    def _start_session(self, body: dict[str, Any]) -> Any:
        """Seat a player in one of the two modes, and write the row.

        The body is the local application's plus `mode`, which defaults to sandbox,
        because a result that counts is something a player chooses and never something
        they get by leaving a field out.

        A visitor who has not signed in gets a sandbox row with nobody in their seat.
        That is §4's sandbox exactly -- undo and restart, counted by nothing -- and
        `matches.start` is what refuses them leaderboard mode.

        `rematch_of` is the board's rematch control saying which game this one follows,
        and it is the only thing that sends it: the game page's start button and the
        designer's play button leave it out, so their rows are null (C1). It is a match
        id and nothing about a player, and `matches.start` checks it belongs to the
        player asking before writing it.

        A game nobody sits at says who is playing it, in the same sentence the designer's
        watch has carried since Q265 and from the same function -- weak play by a weak
        player looks exactly like a broken game, and a site watch raises that question as
        much as a draft's does. It is "" for every game a person is in.
        """
        digest = str(body["hash"])
        if body.get("puzzle"):
            return self._start_try(digest, str(body["puzzle"]), body)
        roster = self.roster(digest)
        # A game played in rounds seats its own rung of the name asked for, as the local
        # application does (Q918c): the page's *watch it play itself* asks for the
        # alternating ladder's player, and a preview in rounds must answer it (Q931).
        seats = [
            opponents.seat_for(str(seat), roster)
            for seat in body.get("seats") or ["human", "minimax:nodes=1600"]
        ]
        agents = [entry.agent for entry in roster]
        if len(seats) != 2 or any(seat not in agents for seat in seats):
            raise ValueError(f"seats must be two of: {', '.join(agents)}")
        if any(quota.is_strong(str(seat)) for seat in seats):
            usage.note("strong")
        with self._connection() as connection:
            found = matches.start(
                connection,
                game=digest,
                mode=str(body.get("mode") or matches.SANDBOX),
                seats=[str(seats[0]), str(seats[1])],
                user=self.viewer,
                # Fresh unless the request names one, drawn here and kept in the row
                # (Q987): no page names one, and a rematch is a fresh match.
                seed=play.seed_asked(body.get("seed")),
                rematch_of=str(body["rematch_of"]) if body.get("rematch_of") else None,
            )
            return {
                **matches.session(connection, found.id).snapshot(),
                "watching": watching_line(list(seats), roster),
            }

    # ------------------------------------------------ proven positions (R9b, Q605)

    def _proven_record(self, digest: str) -> tuple[proven.Record, library.Game]:
        """The game's proven positions, or the 404 an address with none gets."""
        game = self.known().get(digest)
        record = proven.load(self.runs_dir, digest, None if game is None else game.spec)
        if game is None or record is None:
            raise KeyError(proven.NOT_HERE)
        return record, game

    def _proven_section(self, connection: Any, digest: str) -> dict[str, Any] | None:
        """*Proven positions* on the game's page, with the reader's own marks, or None.

        A record that offers nothing because every position was too easy still has its
        section, which says so (`proven.NONE_LEFT`); one whose positions no longer replay has
        none."""
        game = self.known().get(digest)
        record = proven.load(self.runs_dir, digest, None if game is None else game.spec)
        if record is None or not (record.positions or record.too_easy):
            return None
        marks = (
            None if self.viewer is None else proven.marks_on(connection, self.viewer, digest)
        )
        return proven.summary(record, marks)

    def _proven_page(self, digest: str, asked: dict[str, str]) -> dict[str, Any]:
        """Q993b: `GET /api/proven/<hash>`, the positions' own page (`proven.page`): the
        list, its groups and chips, and the reader's progress. A visitor's right picks this
        visit come back as `found` (ids, comma-separated), held in the page and kept
        nowhere; a signed-in reader's are their stored marks, and `found` is not read."""
        record, game = self._proven_record(digest)
        marks = None
        if self.viewer is not None:
            with self._connection() as connection:
                marks = proven.marks_on(connection, self.viewer, digest)
        visit = {part for part in str(asked.get("found") or "").split(",") if part}
        return {**proven.page(record, marks, visit), "name": game.name,
                "sides": list(library.sides(game))}

    def _start_try(self, digest: str, position_id: str, body: dict[str, Any]) -> Any:
        """*Convert it*: a practice game from a proven position against *hardest*.

        The seats are the position's: the person on the side to move, *hardest* on the
        other. Nothing else in the body is read but the game and the position, so a try
        cannot be rated, seated otherwise or called a rematch. `matches.start` rations
        the strong seat and refuses a visitor *hardest* (L15) as it does for any start.
        """
        stray = sorted(set(body) - {"hash", "puzzle", "seats", "mode"})
        if stray or (body.get("mode") or matches.SANDBOX) != matches.SANDBOX:
            raise matches.MatchError(matches.PUZZLE_IS_PRACTICE)
        self._playable(digest)
        record, _game = self._proven_record(digest)
        position = record.find(position_id)
        if position is None:
            raise KeyError(proven.NOT_HERE)
        if proven.OPPONENT not in {entry.agent for entry in self.roster(digest)}:
            raise matches.MatchError(
                "hardest is not offered on this game, so a proven position cannot be "
                "converted here"
            )
        usage.note("strong")
        with self._connection() as connection:
            found = matches.start(
                connection,
                game=digest,
                mode=matches.SANDBOX,
                seats=proven.seats_for(position),
                user=self.viewer,
                puzzle=(position.id, position.moves),
            )
            return {**matches.session(connection, found.id).snapshot(), "watching": ""}

    def _find_the_move(self, digest: str, position_id: str, body: dict[str, Any]) -> Any:
        """*Find the move*: the stored proof of the move picked, and a mark for a right one.

        The answer is the record's, looked up (`proven.answer`). A signed-in person who
        picks the move the proof names gets it marked on their own page; a visitor's pick
        is answered and kept nowhere, and a pick that is not the move writes nothing.
        """
        self._playable(digest)
        record, _game = self._proven_record(digest)
        show = bool(body.get("show"))
        move = body.get("move")
        picked = move if isinstance(move, int) and not isinstance(move, bool) else None
        if not show and picked is None:
            raise ValueError("pick one of the moves in this position")
        said = proven.answer(record, position_id, None if show else picked, show=show)
        said["marked"] = False
        if said["right"] and self.viewer is not None:
            with self._connection() as connection:
                proven.mark(connection, self.viewer, digest, position_id, proven.FOUND)
            said["marked"] = True
        return said

    def _session_action(self, session_id: str, action: str, body: dict[str, Any]) -> Any:
        """Move, let the computer player answer, take a move back, start again, resign,
        or offer a draw (Q790, answered by the core at once).

        `undo` and `restart` are sandbox's; `matches` refuses them on a leaderboard row
        in a sentence a player can read. A finish -- by the position ending or by a
        resignation -- moves the game's table inside this one transaction.

        Every action but `resign` can take a strongest-opponent seat back, for a game
        that sat past `quota.SEAT_ACTIVE` or ended, so `matches` checks the cap first and
        refuses with a 400 saying to come back in a few minutes or start against a
        smaller opponent (Q410). The check holds a lock until this transaction commits,
        which is why the action and its commit share one connection.

        A draft's session is in memory rather than in the store, so it is answered by the
        local application's own handler on the way past.
        """
        draft = self.drafts.find(session_id)
        if draft is not None:
            if action == "agent":
                matches.note_budget(list(draft.seats))
            return super()._session_action(session_id, action, body)
        due: move_email.Due | None = None
        with self._connection() as connection:
            found = matches.row(connection, session_id)
            if not correspondence.between_people(found):
                answer = self._against_computer(connection, session_id, action, body)
                if found.source == matches.PUZZLE:
                    self._proof_badges(connection, session_id)
                return answer
            answer = self._between_people(connection, found, action, body)
            if action == "move":
                # Q572. Whether this move sends the one email, written down in the move's
                # own transaction; it goes to the sender after the commit.
                due = move_email.due(
                    connection, found.id, correspondence.seat_of(found, self.viewer), answer,
                    before=found.plies,
                )
        self._mail_move(due)
        return answer

    def _proof_badges(self, connection: Any, match_id: str) -> None:
        """Q993e: a try that ended in the person's win may finish a game's proven
        positions, so its badges are granted here, in the finish's own transaction and
        after `proven.converted` wrote the mark (`badges.grant_for_proof`). Any other
        action, ending or person grants nothing. The records are this site's, as its
        pages load them, so the badge counts exactly the positions the page offers."""
        won = proven.won_by(connection, match_id)
        if won is None:
            return
        known = self.known()

        def record(digest: str) -> proven.Record | None:
            game = known.get(digest)
            return proven.load(self.runs_dir, digest, None if game is None else game.spec)

        badges.grant_for_proof(connection, won[0], record)

    def _against_computer(
        self, connection: Any, session_id: str, action: str, body: dict[str, Any]
    ) -> Any:
        if action == "move":
            return matches.play(connection, session_id, int(body["move"]))
        if action == "agent":
            return matches.play_agent(connection, session_id)
        if action == "undo":
            return matches.undo(connection, session_id)
        if action == "restart":
            return matches.restart(connection, session_id)
        if action == "resign":
            # `by` names a seat when there could be two people in the game, which
            # phase 1 never has. A visitor has no id to name and is the only person
            # at the board, so `matches` finds their seat itself.
            matches.resign(connection, session_id, by=self.viewer)
            return matches.session(connection, session_id).snapshot()
        if action == "draw":
            # Q790: the computer answers at once, by the strongest search the game has,
            # asked of the core; a draw it accepts is finished and rated in this
            # transaction, like any other finish.
            return matches.offer_draw(connection, session_id, by=self.viewer)
        raise KeyError(f"no such action: {action}")

    def _mail_move(self, due: move_email.Due | None) -> None:
        """Hand a move email to the sender, after the move it is for has committed.

        A failed send never blocks a move (Q572): whatever the sender raises is written on
        the email's row and printed, and the person who moved gets their answer as if
        nothing had been sent. Nothing retries.
        """
        if due is None:
            return
        try:
            self.send(due.message)
        except Exception as error:  # noqa: BLE001 - any failure of the sender is the same
            print(f"move email {due.id} not sent: {error}", file=sys.stderr)
            try:
                with self._connection() as connection:
                    move_email.failed(connection, due.id, str(error))
            except Exception as again:  # noqa: BLE001 - the note is best effort
                print(f"move email {due.id}: the failure was not written: {again}",
                      file=sys.stderr)

    # --------------------------------------------- finding an opponent (Q571, §1.4)

    def _signed_in(self) -> int:
        """The signed-in person's id. Every challenge route refuses a visitor earlier
        (`routes.OWN`'s sentence), so this is the type's answer, not a second door."""
        if self.viewer is None:
            raise SignInFirst("Sign in to play a person.")
        return int(self.viewer)

    def _challenges(self) -> Any:
        """The landing page's *A challenge for you*, and the rest of this person's own.

        The landing page draws the first list above its first step and nothing at all
        when it is empty (§1.5: no empty heading, no count).
        """
        with self._connection() as connection:
            return _challenges_of(connection, self._signed_in())

    def _challenge(self, body: dict[str, Any]) -> Any:
        """Send a challenge: to a member by handle, as a link, or as a rematch.

        `seat` is `first`, `second` or `either`, the challenger's. A rematch names the
        finished game between two people it is offered from, and its seats are that
        game's swapped. The answer is the challenge as *Challenges you sent* lists it.
        """
        viewer = self._signed_in()
        # A challenge is a game, a seat and a person or a link, and nothing else: no
        # field can carry a message (Q606, the user's 2026-10-02 *no comments*).
        stray = sorted(set(body) - CHALLENGE_FIELDS)
        if stray:
            raise matches.MatchError(
                "a challenge names a game, a seat and a member or a link, and nothing else"
            )
        with self._connection() as connection:
            if body.get("rematch"):
                made = challenges.rematch(connection, str(body["rematch"]), viewer)
            else:
                seat = str(body.get("seat") or "either")
                if seat not in challenges.SEATS:
                    raise matches.MatchError("the seat is first, second or either")
                made = challenges.challenge(
                    connection,
                    viewer,
                    str(body.get("hash") or ""),
                    challenges.SEATS[seat],
                    str(body.get("handle") or "") or None,
                    link=bool(body.get("link")),
                )
            listed = [
                entry for entry in challenges.sent_by(connection, viewer)
                if entry["id"] == made.id
            ]
        return listed[0]

    def _challenge_answer(self, challenge: str, action: str) -> Any:
        """Accept, decline or withdraw one challenge, by its id.

        Accepting answers with the match it started, which the page opens; the other two
        answer with this person's challenges as they now stand.
        """
        viewer = self._signed_in()
        try:
            number = int(challenge)
        except ValueError:
            raise KeyError(f"there is no challenge {challenge}") from None
        with self._connection() as connection:
            if action == "accept":
                made = challenges.accept(connection, number, viewer)
                return {"id": made.id}
            if action == "decline":
                challenges.decline(connection, number, viewer)
            elif action == "withdraw":
                challenges.withdraw(connection, number, viewer)
            else:
                raise KeyError(f"no such action on a challenge: {action}")
            return _challenges_of(connection, viewer)

    def _session_read(self, session_id: str) -> Any:
        """A match's position, with a game between two people's own block beside it.

        A running game between two people is its two players' alone; anybody else is
        answered as a wrong address is (Q570, `site/correspondence.py`).
        """
        with self._connection() as connection:
            found = matches.row(connection, session_id)
            if not correspondence.may_read(found, self.viewer):
                raise KeyError(correspondence.NOT_HERE)
            if correspondence.between_people(found):
                # Q572. A player reading their running game has it open, so a move made
                # while they watch sends them no email.
                move_email.seen(connection, found, self.viewer)
            return self._snapshot(connection, found)

    def _snapshot(self, connection: Any, found: matches.Row) -> dict[str, Any]:
        # Drawn for this reader: in a game in rounds between two people, a move the other
        # player sealed is in it only as *sealed* (Q918f, `correspondence.snapshot_for`).
        return correspondence.snapshot_for(connection, found, self.viewer)

    def _between_people(
        self, connection: Any, found: matches.Row, action: str, body: dict[str, Any]
    ) -> Any:
        """One player's act in a game between two people (Q570).

        Every act names its player, who is whoever is signed in; `correspondence`
        refuses somebody who is not one of the two. A move is sent only when the person
        presses *Send*, so what arrives here is final. `draw` offers a draw (or accepts
        the other's standing offer), `accept` accepts one, `decline` declines it (Q790),
        `cancel` calls the game off
        before both have moved. There is no take-back, restart or computer reply.
        """
        if not correspondence.may_read(found, self.viewer):
            raise KeyError(correspondence.NOT_HERE)
        viewer = self.viewer
        if action == "move":
            correspondence.move(connection, found.id, viewer, int(body["move"]))
        elif action == "resign":
            correspondence.resign(connection, found.id, viewer)
        elif action == "draw":
            correspondence.offer_draw(connection, found.id, viewer)
        elif action == "accept":
            correspondence.accept_draw(connection, found.id, viewer)
        elif action == "decline":
            correspondence.decline_draw(connection, found.id, viewer)
        elif action == "cancel":
            correspondence.cancel(connection, found.id, viewer)
        elif action in ("undo", "restart"):
            matches._sandbox_only(connection, found.id, "undone" if action == "undo" else "restarted")
        else:
            raise KeyError(f"no such action in a game between people: {action}")
        return self._snapshot(connection, found)

    def _submit(self, body: dict[str, Any]) -> Any:
        """File a measurement this person's browser made of their own game (B10).

        `upload` is the text the gathering half wrote under Pyodide, carried as a string
        the page never parsed. A refusal that needs no walk is an answer rather than an
        error, with its name, because `stale-build` is one the page acts on: it fetches
        the current manifest and offers to measure again. What is filed is walked on
        `checks`' thread, never this one.
        """
        viewer = self._viewer()
        game = str(body.get("game") or "")
        text = body.get("upload")
        if not isinstance(text, str) or not text:
            raise ValueError("a submission carries the measurement's upload, as text")
        try:
            with self._connection() as connection:
                job = legality.file(
                    connection,
                    self.runs_dir,
                    game=game,
                    submitter=viewer,
                    text=text,
                    basis=self.checks.basis,
                )
        except legality.Refused as refusal:
            return {
                "game": game,
                "state": legality.REFUSED,
                "refusal": {"name": refusal.name, "sentence": refusal.sentence},
            }
        self.checks.wake()
        return {"job": job, "game": game, "state": legality.QUEUED}

    def _submission(self, job: str) -> Any:
        """Where one of this person's submissions stands (`legality.status`)."""
        if not job.isdigit():
            raise KeyError(f"no such submission: {job}")
        with self._connection() as connection:
            return legality.status(connection, int(job), self.viewer, self.checks.basis)

    def _viewer(self) -> int:
        """Who is asking, for the routes that have no answer without one.

        A vote and a profile are one person's, so both are refused until 1.3's cookie
        says who that is. Playing is not on this list: a sandbox game names nobody.
        """
        if self.viewer is None:
            raise matches.MatchError("sign in to keep a star rating and a profile")
        return self.viewer


def _agreed(said: Any) -> int | None:
    """The terms' version a publish request says its box agreed to, or `None` (Q722).

    A whole number only: a boolean, a string or anything else is no tick, so a body cannot
    smuggle text into the record.
    """
    if isinstance(said, bool) or not isinstance(said, int):
        return None
    return said


def _draft_hash(spec: Any) -> str:
    """The hash a save would give this draft, or `""` when it is not a ruleset yet.

    Worked out as `designing.save` works it out -- settled, then hashed -- so the draft cap
    can wave through rules the store already holds. A draft that does not validate gets
    `""`, and the save refuses it in its own words further on.
    """
    try:
        ready = designing.settled(spec)
        if not isinstance(ready, dict) or validate_spec(ready):
            return ""
        return str(spec_hash(ready))
    except Exception:  # noqa: BLE001 - any failure here is the save's to report
        return ""


def _one_run_basis(digest: str, spec: Any) -> Basis | None:
    """What the server would stamp on its own one-run measurement of this game now.

    `None` where the store holds no rules to compile, which shows nothing: without the
    tables fingerprint there is nothing to match a stamp against.
    """
    if not isinstance(spec, dict):
        return None
    return corpus.current_basis(digest, spec=spec, seeds=corpus.ONE_SEED)


DESIGNER_RULES = {
    "passesScreen": "Its rules passed the playability check at a save.",
    "tried": (
        "Other signed-in people who have started a game of it here, practice or rated. "
        "Its designer is not counted, and nobody is named."
    ),
    "forkedBy": "People other than its designer who kept a game of their own built from this one.",
}
"""What each of the game page's designer marks reads, so the page names it (test 7)."""


def _designer_marks(
    connection: Any, digest: str, named: Mapping[int, str]
) -> dict[str, Any] | None:
    """D2 to D4 on a designed game's page, or `None` on a game nobody designed here.

    *Passes the screen*, *n people have tried this* and *forked by ‹handle›*, each with
    the sentence that says what it reads. The count is a number and never who (the user's
    decision B of 2026-09-14). The page draws a mark only where there is something to
    say, so a game nobody has tried reads as nothing rather than as nought.

    `by` is its designer's handle, for the page's *where it came from* (the first-visit
    study's F9): a saved draft was credited to *a design run here … found in round 0*,
    the save's machinery, and not to the person who made it. It is the owner's row read
    now, so a renamed account reads as its new handle and one that left as
    `identity.FORMER`; nothing is written into the rules, so no game's hash moves.
    """
    game, forks = designs.around(connection, digest)
    if game is None:
        return None
    marks = milestones.marks_from(game, forks, designs.tries(connection, [digest]))
    return {
        "by": None if game.owner is None else named.get(game.owner, identity.FORMER),
        "passesScreen": marks.passes_screen,
        "tried": marks.tried,
        "forkedBy": [named.get(user, identity.FORMER) for user in marks.forked_by],
        "rules": DESIGNER_RULES,
    }


def _first_played(
    rows: Sequence[matches.Row], named: Mapping[int, str]
) -> dict[str, Any] | None:
    """M5, this game's first human game: who was in it and when it finished.

    The rows arrive in finish order, so the first one that counts is the answer. A row
    counts under the same rule the milestone reader applies -- finished, with a result,
    both seats held by an account, not loaded from the local application, and not two
    computer players. Everyone seated in that match is named, so a game between two people
    reads *first played by ada and grace* rather than picking one of them.
    """
    for row in rows:
        if row.source is not None or row.result is None:
            continue
        if row.seat0 is None or row.seat1 is None:
            continue
        people = [row.player(seat).user for seat in (0, 1) if not row.is_computer(seat)]
        if not people:
            continue
        return {
            "handles": [named.get(int(user), "") for user in people if user is not None],
            "when": None if row.finished is None else row.finished.isoformat(),
        }
    return None


def _highest(
    table: Mapping[rating.PlayerKey, rating.Player], named: Mapping[int, str]
) -> dict[str, Any] | None:
    """S2, the highest-rated person on this game's table, or None while nobody is.

    `milestones.highest_people` is the reading, so the line here and the count on a
    profile answer the same question. Nobody holds it until their rating is established,
    and everyone tied at the top holds it together.
    """
    people = milestones.highest_people(table)
    if not people:
        return None
    return {
        "handles": [
            named.get(int(player.key.user), "")
            for player in people
            if player.key.user is not None
        ],
        "rating": people[0].rating,
    }


def _named(game: str, names: Mapping[str, str]) -> dict[str, Any]:
    """One game as the overall page names it, hash and name together.

    The name can be empty -- a game the store knows by hash alone -- and the page then
    shows the short hash, as every other page holding hashes does.
    """
    return {"hash": game, "name": names.get(game, "")}


def _record(found: unbeaten.Record) -> dict[str, Any]:
    """One person's record on one game, as the page reads it.

    The rung is the highest-rated computer player they have beaten, carrying the rating it
    held going into that match and the seat they won from -- §5's three, and all three are
    on the row rather than read off a table human play keeps moving.
    """
    best = found.best
    return {
        "state": found.state,
        "played": found.played,
        "won": found.won,
        "drawn": found.drawn,
        "lost": found.lost,
        "rung": None if best is None else {
            "agent": best.agent,
            # What the profile calls it (S1): it has no roster to look a player up in.
            "label": _opponent_label(best.agent),
            "rating": best.rating,
            "seat": best.seat,
            "top": best.top,
            "when": None if best.when is None else best.when.isoformat(),
        },
    }


def _between_entry(
    connection: Any, found: matches.Row, user: int, at: datetime
) -> dict[str, Any]:
    """One running game between two people, as the profile lists it (Q570).

    Whose move it is is the replayed position's side to move, the engine's answer. When
    the last move landed is a fact, and so is the time the player to move has left
    (Q627): `left` in seconds for `correspondence.started_order`, `toMove` in words.
    """
    seat = found.seat_of(rating.person(user))
    other = found.player(1 - seat).user or 0
    # Whether it is this person's move: the engine's side to move, or in a game in rounds
    # whether their move for the round is still to seal (Q918f).
    yours = seat in correspondence.owing(connection, found)
    left = correspondence.time_left(found, at)
    opponent = identity.handle_of(connection, other)
    return {
        "id": found.id,
        "game": found.game,
        "opponent": opponent,
        "opponentMark": identity.mark_of(other, opponent),
        "seat": seat,
        "yourMove": yours,
        "rated": found.mode == matches.LEADERBOARD,
        "plies": found.plies,
        "moved": (found.moved or found.started).isoformat(),
        "left": left.total_seconds(),
        "toMove": correspondence.to_move_said(left, "You" if yours else "They"),
    }


def _challenges_of(connection: Any, user: int, waiting: bool = True) -> dict[str, Any]:
    """This person's challenges and ticks, as the landing page and their own page read
    them (Q571): waiting on them, sent by them, and the games they are open to.

    `waiting` adds Q572's *Games waiting on your move* for the landing page; the profile
    has the whole list already and asks without it.
    """
    paused = correspondence.paused(connection)
    return {
        "forYou": challenges.for_you(connection, user),
        "sent": challenges.sent_by(connection, user),
        "open": challenges.open_games(connection, user),
        "pausedSays": correspondence.PAUSED_SAID if paused else None,
        **({"waiting": _waiting_on(connection, user)} if waiting else {}),
    }


def _waiting_on(connection: Any, user: int) -> list[dict[str, Any]]:
    """*Games waiting on your move* (Q572, `docs/archive/plan-return.md` §1.5, s3 of the mock).

    This person's running games between people where the engine's side to move is their
    seat, in `correspondence.started_order` (Q627), each with its game's name, the
    opponent, when the last move landed and the time left to move in the server's words.
    Games waiting on the other person are on the person's own page, not here. Empty when
    nothing waits, and the landing page then draws nothing at all.
    """
    clock = correspondence.now(connection)
    ordered = correspondence.started_order([
        _between_entry(connection, row, user, clock)
        for row in correspondence.running_for(connection, user)
    ])
    mine = [entry for entry in ordered if entry["yourMove"]]
    named = browse.names(connection, sorted({entry["game"] for entry in mine}))
    return [{**entry, "name": named.get(entry["game"], "")} for entry in mine]


def _move_box(connection: Any, user: int) -> dict[str, Any]:
    """The move email's box on a person's own page (Q572): whether it is ticked, and
    the sentences that say exactly what it does."""
    return {
        "on": move_email.is_on(connection, user),
        "said": move_email.BOX_SAID,
        "under": move_email.BOX_UNDER,
    }


def _carry_on(found: matches.Row, user: int) -> dict[str, Any]:
    """One unfinished rated game of this person's, as a *carry on* entry reads it.

    B2 of `docs/archive/plan-ux.md`. The id is the whole point -- `#/play/<id>` is the
    address the router now fetches -- and the rest is what tells two open games apart: who
    is opposite, which side this person is on, and how far in it got.

    The seat is read off the row rather than assumed, because either side can be the
    person's. A leaderboard row always names one, so a seat of -1 cannot happen here; it
    is written as-is if it ever does, rather than guessed at.

    `label` is the opponent in words, off the ladder every board shares. The profile has
    no roster to look a player up in and would otherwise print `minimax:nodes=1600`; the
    game page has its own roster and uses that instead, which also names the plans and
    the searched tables this list cannot.
    """
    seat = found.seat_of(rating.person(user))
    other = found.seat1 if seat == 0 else found.seat0
    agent = (other.agent or "") if other is not None else ""
    return {
        "id": found.id,
        "game": found.game,
        "opponent": agent,
        "label": _opponent_label(agent),
        "seat": seat,
        "plies": found.plies,
        "started": found.started.isoformat(),
        "moved": None if found.moved is None else found.moved.isoformat(),
    }


def kind_words(axis: str, bin_name: str = "") -> tuple[str, str]:
    """One axis of the kind-of-game grid as a visitor reads it: its question, capitalised
    (*How the game is won*), and one bin's meaning cut to its first clause (*won by
    claiming enough of the board*).

    The words are the game page's own kind cards' (`loop/grid.axis_vocabulary`, which a
    game's *What kind of game it is* panel draws), so the profile and the game page say a
    kind one way (S6). The cut is at the first clause because a list line is not the
    card: the card's whole sentence says why a kind matters, and a line names it. An axis
    or bin the grid does not have reads as its key, as it did before.
    """
    for entry in grid.axis_vocabulary():
        if entry["axis"] != axis:
            continue
        asks = str(entry["asks"])
        means = next((str(b["means"]) for b in entry["bins"] if b["bin"] == bin_name), "")
        for cut in (" -- ", ", so ", ", where "):
            means = means.split(cut)[0]
        return asks[:1].upper() + asks[1:], means or bin_name
    return axis, bin_name


def _opponent_label(agent: str) -> str:
    """That player's name off the ladder, and *the computer* where the ladder has none.

    The name the page's own `shortName` reads (`Opponent.name`), so one player reads the
    same on a game's page and on a profile, and never an agent string (S8, P16).
    """
    return opponents.name_of(agent)


def _rungs(found: milestones.Rungs | None) -> dict[str, Any] | None:
    """M4's column. Null on a game with no ladder, which the profile lists none of."""
    return None if found is None else {"beaten": found.beaten, "of": found.of}


def _distance(found: milestones.Distance | None) -> dict[str, Any] | None:
    """T3's column, null until this person has a rating on the game.

    `points` is how far below the top they are, so a negative number means above it.
    `rating` is the person's own rating there, the number F8 found the profile hid. The
    page words the direction; nothing here is rounded, because rounding is a drawing.
    """
    return (
        None
        if found is None
        else {"points": found.points, "rating": found.rating,
              "established": found.established}
    )


def _reraise() -> Any:
    """Raise the exception being handled, so another mapping can answer it.

    `site_handler_for` puts one case in front of the local handler's status mapping and
    hands the rest back to it. Re-running the endpoint to do that would run it twice, so
    the exception is carried across instead of the call.
    """
    raise


def served_elsewhere(path: str) -> bool:
    """Whether a GET for this path is answered by the local application's handler.

    The API, the page itself and the files it loads. Every other address the site has no
    page for gets the not-found page (V7) rather than a line of JSON a browser shows raw.
    """
    return (
        path.startswith(
            ("/api/", VENDOR_PREFIX, shell.FONT_PREFIX, picture.PREFIX, BROWSER_PREFIX)
        )
        or path in (*PAGE_PATHS, shell.SHELL_SCRIPT_ROUTE, WORKER_ROUTE, RULES_ROUTE)
        or path in shell.icons()
        or path in PAGE_SCRIPTS
    )


def site_handler_for(app: SiteApp) -> type[Any]:
    """The local application's handler, with the cookie around it.

    Three things are added and nothing is replaced. Every dispatched request has
    `app.viewer` filled from the cookie before it runs; an answer that opened or closed a
    session carries the `Set-Cookie` that says so; and the site's own pages and their
    scripts are served ahead of dispatch, the way the local handler serves `/` and its.

    CSRF has two answers from §6. The cookie is `SameSite=Lax`, so the browser does not
    send it on a cross-site POST and a form on somebody else's page arrives as a visitor.
    And the local handler refuses any POST without `web.server.REQUEST_HEADER`, which
    only the page's own script can set, so such a form is not dispatched at all. `HttpOnly` keeps it out of any script on the page, and `Secure` is set
    when Caddy says the request arrived over TLS -- which is always, on the box, and never
    on a development machine serving `http://127.0.0.1`.
    """
    base = handler_for(app)

    class Handler(base):  # type: ignore[valid-type, misc]
        def do_GET(self) -> None:  # noqa: N802 - the stdlib's naming
            self._note_source()
            parsed = urlparse(self.path)
            if parsed.path in routes.PAGES:
                self._page(parsed.path, parsed.query)
                return
            if pages.MAIL_OFF.match(parsed.path):
                self._mail_off(parsed.path)
                return
            if parsed.path.startswith(routes.MEMBERS):
                self._member(parsed.path[len(routes.MEMBERS):])
                return
            if pages.LINK.match(parsed.path):
                # Q571. A challenge link is a constant page: the script asks whose it is.
                self._html(pages.challenge_link(parsed.path[len(routes.LINKS[0]):]))
                return
            if parsed.path in pages.SCRIPTS:
                self._script(pages.SCRIPTS[parsed.path])
                return
            digest = game_drawing.digest_of(parsed.path) or game_drawing.proven_digest_of(
                parsed.path
            )
            if digest is not None:
                self._game_page(digest)
                return
            shown = game_picture.ROUTE.match(parsed.path)
            if shown is not None:
                self._game_picture(shown.group(1), shown.group(2))
                return
            if self._front(parsed.path):
                return
            if not served_elsewhere(parsed.path):
                self._lost()
                return
            self._identify(parsed.path)
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - the stdlib's naming
            self._note_source()
            if pages.MAIL_OFF.match(urlparse(self.path).path):
                self._discard_body()
                self._mail_off(urlparse(self.path).path)
                return
            if self._forged():
                return
            if self._too_big():
                return
            self._identify(urlparse(self.path).path)
            super().do_POST()

        def _mail_off(self, path: str) -> None:
            """The one-click off in a move email (Q572): switch the box off, say so.

            By the token alone, never the cookie, so it works from any mail program and
            in a browser signed in as somebody else; the token can do nothing else.
            """
            found = pages.MAIL_OFF.match(path)
            token = "" if found is None else found.group(1)
            with app._connection() as connection:
                known = move_email.turn_off(connection, token)
            self._html(pages.mail_off(move_email.OFF_SAID if known else move_email.OFF_UNKNOWN))

        def _too_big(self) -> bool:
            """Refuse a designer POST whose body is over `quota.DESIGN_BODY_BYTES` (Q503a).

            Read off `Content-Length` before the body is, so a large body costs no memory:
            nothing of it is parsed, kept or identified. Once the 413 is written the body
            is drained in pieces and thrown away, up to `DESIGN_DRAIN_BYTES`, and the
            connection closed. Without the drain the close finds the body still in the
            socket and becomes an abort, which discards the 413 before the sender reads
            it (WinError 10053 on Windows, a reset elsewhere; Q841): the designer would
            see a failed request instead of the sentence saying how big a design may be.
            A claim longer than the drain is closed on unread, as before.
            """
            parts = routes.split(urlparse(self.path).path)
            if parts[:2] != ("api", "design"):
                return False
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return False
            if length <= quota.DESIGN_BODY_BYTES:
                return False
            self.close_connection = True
            self._send_json(
                413,
                {
                    "error": f"A design may be at most {quota.DESIGN_BODY_BYTES // 1024} KB "
                    f"as the designer sends it; this one is {length // 1024} KB."
                },
            )
            self._discard_body(DESIGN_DRAIN_BYTES)
            return True

        def _note_source(self) -> None:
            """Which address this request is counted against, before anything dispatches.

            First thing in both verbs, because the sign-in limit is the reason the
            address is read at all and a route that skips `_identify` must not skip this.
            """
            app.source = quota.source_address(
                self.client_address[0], self.headers.get("X-Forwarded-For")
            )

        # ------------------------------------------------------------ the site's pages

        def _page(self, path: str, query: str) -> None:
            """One of the site's own short pages, or the way to one.

            Credits and the privacy notice come first because they are constants and ask
            nobody's name. A profile with nobody signed in is the sign-in page rather
            than a refusal -- the person asked to see their profile, and being sent to
            the door is the answer to that.
            """
            if path == "/credits":
                self._html(pages.credits())
                return
            if path == "/privacy":
                self._html(pages.privacy())
                return
            if path == terms.PATH:
                self._html(pages.terms_page())
                return
            if path == asking.PATH:
                # Q831. The form for asking for an invitation: a constant page, so a
                # visitor's request for it opens no connection and reads no cookie.
                self._html(pages.request_invitation())
                return
            if path == "/badges":
                # The plain list of every badge and its rule. It asks nobody's name on
                # purpose: a rule shown only to whoever has met it is a hidden rule (§3,
                # test 7), so this page reads the same for everybody.
                self._html(pages.badges())
                return
            viewer = self._identify(path)
            after = (parse_qs(query).get("then") or [""])[-1] or None
            if path == "/signin":
                self._html(pages.signin(after))
            elif path == "/join":
                asked = parse_qs(query).get("code") or [""]
                self._html(pages.join(asked[-1], after))
            elif viewer is None:
                self._redirect("/signin")
            elif path == "/players":
                # Q760: `?q=` fills the search field, so a search can be linked to.
                searched = (parse_qs(query).get("q") or [""])[-1]
                self._html(pages.players(viewer.handle, user=viewer.id, asked=searched))
            elif path == "/suggestions":
                self._html(pages.suggestions(viewer.handle, app.players_listed(), user=viewer.id))
            elif path == "/admin":
                # A member gets the answer an invented path gets, rather than a 403 that
                # would tell them the page is there. Somebody not signed in at all has
                # already been sent to the door above, because they may well be an admin.
                if viewer.role != admin.ADMIN:
                    self._lost()
                else:
                    self._html(pages.admin(viewer.handle, app.players_listed(), user=viewer.id))
            else:
                self._html(pages.profile(viewer.handle, app.players_listed(), user=viewer.id))

        def _member(self, handle: str) -> None:
            """A member's page (R1, Q596). A visitor goes to the door, whatever the handle,
            as one asking for `/players` does; a member asking for a handle with no page
            (never taken, left, or shut out) gets the not-found page, as for any address
            with nothing at it."""
            viewer = self._identify(routes.MEMBERS)
            if viewer is None:
                self._redirect("/signin")
                return
            reading = app.member(handle, viewer.id)
            if reading is None:
                self._lost()
                return
            self._html(
                pages.member(reading, viewer.handle, app.players_listed(), viewer=viewer.id)
            )

        def _game_page(self, digest: str) -> None:
            """A game's own address (G3): its page drawn on the server, or the not-found
            page for a game a visitor cannot open. Timed and counted like an API route, as
            `game_page.ROUTE`, so the hourly rows say what a read of it costs."""
            timed = usage.Timing(lambda: app.game_page(digest))
            app.usage.enter()
            try:
                body = timed()
            finally:
                app.usage.leave()
            status = 404 if body is None else 200
            app.usage.record(game_page.ROUTE, "", status, "", None, timed.cpu, timed.wall)
            if body is None:
                self._lost()
            else:
                self._html(body)

        def _game_picture(self, digest: str, key: str) -> None:
            """A game's link-preview picture (G6), cached for a year since a new picture is a
            new address; the not-found page for anything else. Timed and counted as
            `game_pictures.ROUTE`."""
            timed = usage.Timing(lambda: app.game_picture(digest, key))
            app.usage.enter()
            try:
                body = timed()
            finally:
                app.usage.leave()
            status = 404 if body is None else 200
            app.usage.record(game_pictures.ROUTE, "", status, "", None, timed.cpu, timed.wall)
            if body is None:
                self._lost()
            else:
                self._send(200, body, game_picture.TYPE, cache=IMMUTABLE)

        def _front(self, path: str) -> bool:
            """G4: the front page, the games, `robots.txt` and `sitemap.xml`, each with what a
            crawler reads; G8: `llms.txt` and the rule language as a page. The same bytes for everybody: no cookie is read. Whether it
            answered."""
            if path in ("/", "/index.html"):
                self._html(app.home_page())
            elif path == front.GAMES_PATH:
                self._html(app.games_page())
            elif path == seo.ROBOTS_PATH:
                base = seo.base_for(self.headers.get("Host"))
                self._send(200, seo.robots(base).encode("utf-8"), seo.ROBOTS_TYPE,
                           cache=seo.CACHE)
            elif path == seo.SITEMAP_PATH:
                base = seo.base_for(self.headers.get("Host"))
                self._send(200, app.sitemap(base).encode("utf-8"), seo.SITEMAP_TYPE,
                           cache=seo.CACHE)
            elif path == seo.LLMS_PATH:
                base = seo.base_for(self.headers.get("Host"))
                self._send(200, app.llms(base).encode("utf-8"), seo.LLMS_TYPE,
                           cache=seo.CACHE)
            elif path == rules_page.PATH:
                # G8: the rule language as a page, in the site's shell; the local handler's
                # own drawing of it (under the application's header) is never reached here.
                self._send(200, app.rules_page(), "text/html; charset=utf-8",
                           cache=RULES_CACHE)
            else:
                return False
            return True

        def _html(self, body: str) -> None:
            self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")

        def _lost(self) -> None:
            """The not-found page (V7), the one answer for an address with no page."""
            self._send(404, pages.not_found().encode("utf-8"), "text/html; charset=utf-8")

        def _redirect(self, where: str) -> None:
            self.send_response(303)
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.end_headers()

        # ------------------------------------------------------------------ the cookie

        def _identify(self, path: str) -> Any:
            """Fill in who is asking, once, before anything runs.

            Only for the paths that are answered from the store. The local application's
            page is a constant string, and so are credits and the privacy notice; a
            database connection to serve a constant would be a connection per page view.
            """
            if (
                not path.startswith(("/api/", routes.MEMBERS))
                and path not in routes.PERSONAL
            ):
                return None
            jar = SimpleCookie()
            jar.load(self.headers.get("Cookie", ""))
            morsel = jar.get(pages.COOKIE)
            return app.identify(None if morsel is None else morsel.value)

        def end_headers(self) -> None:
            """Where the session cookie is written, if this answer opened or closed one.

            The stdlib calls this last for every response, which is what makes it the one
            place a header can be added without repeating the local handler's `_send`.
            """
            grant = app.grant
            if grant is not None:
                app.grant = None
                self.send_header("Set-Cookie", self._cookie(grant))
            super().end_headers()

        def _cookie(self, grant: Grant) -> str:
            secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
            if grant.token is None:
                return f"{pages.COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax{secure}"
            age = int(identity.SESSION_LIFETIME.total_seconds())
            return (
                f"{pages.COOKIE}={grant.token}; Path=/; Max-Age={age}; "
                f"HttpOnly; SameSite=Lax{secure}"
            )

        # ------------------------------------------------------------------ the answer

        def _json(self, produce: Any) -> None:
            """The answer, timed and counted by route template (`docs/plan-usage.md`).

            Who is asking is read before the endpoint runs, so a sign-in counts as the
            visitor who made it. `_send_json` records the row, with the status it writes,
            before the answer leaves.
            """
            path = urlparse(self.path).path
            timed = usage.Timing(produce)
            self._counted: tuple[str, str, int | None, usage.Timing] | None = (
                usage.template(self.command, path), path, app.viewer, timed
            )
            app.usage.enter()
            try:
                self._answer(timed)
            finally:
                app.usage.leave()

        def _send_json(self, status: int, payload: Any) -> None:
            counted = getattr(self, "_counted", None)
            if counted is not None:
                self._counted = None
                route, path, viewer, timed = counted
                app.usage.record(
                    route,
                    timed.variant or usage.default_variant(route, path),
                    status,
                    "" if status == 200 else timed.kind,
                    viewer,
                    timed.cpu,
                    timed.wall,
                )
                if status >= 500:
                    # Q721. The hourly counts say a route failed; this line, in the
                    # serve unit's journal, says why. The local handler's request log is
                    # off, so without it a 500 left no trace on the box but Caddy's.
                    print(f"server error {status} on {route}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            super()._send_json(status, payload)

        def _answer(self, produce: Any) -> None:
            """The local handler's statuses, with a refused sign-in in front of them.

            A wrong password is a 401 rather than the 400 a `ValueError` gets, because a
            page has to tell *these credentials are wrong* from *this request made no
            sense*, and so does anything reading the access log for a guessing run.
            A spent hourly allowance is a 429, whose sentence says when to come back.
            """
            try:
                payload = produce()
            except identity.Refused as error:
                self._send_json(401, {"error": str(error)})
            except quota.Spent as error:
                self._send_json(429, {"error": str(error)})
            except Exception:  # noqa: BLE001 - handed straight back to the local mapping
                super()._json(_reraise)
            else:
                self._send_json(200, payload)

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    ruleset_dir: Path = library.RULESET_DIR,
    runs_dir: Path = library.RUNS_DIR,
    heuristics_dir: Path | str = "heuristics",
    archive_dir: Path | str = corpus.ARCHIVE_DIR,
) -> None:
    """Run the site until interrupted.

    The database is read before anything binds, so a unit missing its environment fails
    in one line rather than serving a site whose first request cannot be answered.
    Nothing here opens a connection: no route touches the database until task 1.2, and a
    connection opened to prove a point is a connection to leak.

    Localhost, always. TLS and the public address are Caddy's, in front
    (`deploy/remote/Caddyfile`), and an app that could be reached directly would be an
    app whose client addresses are whatever a client says they are. The limit on guessing
    a password is this process's own (`quota.SignIns`), because Caddy 2.11 ships no
    request limiter and the firewall counts connections rather than attempts.
    """
    database = db.database_url()
    # Whether the environment gives this box a maintenance account, checked before the
    # bind and without a connection. A name that cannot be a handle, or a password under
    # the site's own minimum, is one line here and no account -- never a refusal to
    # start, which is what took the site down on 2026-09-19 (Q365b). The row itself is
    # written at the next sign-in (`SiteApp._signin`), from the hash made here: made
    # before the bind, so no attempt pays a second scrypt for it (Q379).
    maintenance, refused = admin.account_setting()
    admin.account_hash()
    app = SiteApp(ruleset_dir, runs_dir, heuristics_dir, archive_dir)
    server = listen(app, host, port)
    # The site's handler in place of the local application's. The same dispatch, plus the
    # cookie and the site's own pages. `listen` owns the busy-port refusal, so the
    # class is swapped after the bind rather than a second server being built here.
    server.RequestHandlerClass = site_handler_for(app)
    url = f"http://{host}:{server.server_port}/"
    print(f"Chess variant sandbox site is at {url}")
    print(f"serving {len(routes.ALLOWED)} routes; {len(routes.WITHHELD)} are withheld")
    print(f"database: {db.redacted(database)}")
    # Whether an invitation can be sent from here, said at start rather than found out by
    # a maintainer whose `chess-variant-sandbox site invite` fails after writing the row.
    print(
        f"invitations: sending as {mail.sender_address()}"
        if mail.configured()
        else f"invitations: no {mail.KEY_VAR}, so `chess-variant-sandbox site invite --no-send` "
        f"prints the link"
    )
    # Q979. Whether a request for an invitation reaches the owner by mail; the address is
    # the owner's own and is never printed, only whether the setting is there.
    if mail.owner_address() is None:
        print(f"requests for an invitation: no {mail.OWNER_VAR}, so they wait on /admin unmailed")
    else:
        print(f"requests for an invitation: mailed to the address {mail.OWNER_VAR} names")
    # The handle is not printed. It is the one half of the maintenance account that a
    # journal would carry in clear, and this line only has to say whether there is one.
    # A refusal is the same line with its reason, and the site serves either way.
    if refused is not None:
        print(f"maintenance: {refused}; no maintenance account can sign in")
    else:
        print(
            f"maintenance: the account {admin.NAME_VAR} names can sign in"
            if maintenance
            else f"maintenance: no {admin.PASSWORD_VAR}, so no maintenance account can sign in"
        )
    print("press ctrl-c to stop")
    # The corpus is read here, after the bind, and not while the port is still closed.
    # `warm` has the measurement; the journal gets the line it prints when it finishes.
    warm(app)
    # Samples the machine and flushes the usage counts every five minutes, and once more
    # on the way out (`docs/plan-usage.md` §2.4).
    sampler = usage.Sampler(app.usage, app._connection, disk=Path(runs_dir))
    sampler.start()
    # Tier one's runner: a submitted browser measurement is walked here, off the request
    # threads (B10). What a stopped process left running is queued again first.
    app.checks.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        app.checks.stop()
        sampler.stop()
