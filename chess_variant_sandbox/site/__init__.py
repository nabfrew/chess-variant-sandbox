"""The public site: a second front door onto the same package, not a fork of the app.

`chess_variant_sandbox/web/` is the tool for the person sitting at the machine, and every route in it
was written on the assumption that the person at the keyboard owns the repository. The
site is that application with the routes that assumption paid for taken away, plus what a
public site needs and a local one does not: a store, identity, and a queue in front of
anything that costs minutes.

Eighteen things live here for now. `routes` says which of the local application's routes
the site registers; `server` is the app and the server, both the local ones with the
shorter table; `db` is the single place the database URL is read; `migrate` applies the
numbered SQL files under `migrations/` that the store is made of; `rating` is the one Elo
table a game has, a Glicko-2 fold over its leaderboard matches; `matches` is a game in
progress as a row, in the two modes of §4; `load` fills `games`, `measurements`, `ratings`
and `matches` from the repository's own files, idempotently and by hash; `votes` is a
person's one to five stars on a game; `unbeaten` is the tag a game carries until somebody
beats its top computer player, with the per-person record and the three profile counts
under it; `browse` is the games page's own columns and the three orderings §5 offers over
them; `screening` is the playability check a save runs, in a child process the site kills
at a wall clock; `identity` is the invitation, the account and the session behind the
cookie; `mail` is the one message the site sends; `admin` is what a maintainer can see and
do about people and games, leaving included; `quota` is every rationed number in one place and the
check a new match passes; `suggestions` is the box people write ideas into, read by its
author and by an admin; and `pages` is the site's own short pages.
`provisional` is a score a visitor's browser measured, held while the server checks it and
read by nothing that ranks or awards (`docs/archive/plan-client-compute.md` §5), and `legality` is
tier one's job, which walks such a measurement, re-derives its fit and scores it itself.
`stages` is the one word for where a game stands, Draft to Baseline, that every page
names it by (`docs/plan-game-pipeline.md`, Q503b), and `shortlist` is the bar a trial
game meets before its designer may nominate it, and the press (Q503f). `resting` takes a
trial game nobody plays off the listing (Q503i). The rest of the queries over those tables
are tasks 1.7 onward -- see §10 of
`docs/plan-website.md`.

The table modules keep their names behind their modules rather than flattening them in
here, because `load`, `seed`, `top`, `finish`, `start`, `sweep` and `cast` say what they
mean beside a table and nothing at all beside a site.

Installed under the `site` extra: `uv sync --extra site`. A plain `uv sync` installs no
database driver, which is why `db.connect` imports one only when it is called.
"""

from chess_variant_sandbox.site import (
    admin,
    browse,
    identity,
    legality,
    load,
    mail,
    matches,
    pages,
    provisional,
    quota,
    rating,
    suggestions,
    unbeaten,
    votes,
)
from chess_variant_sandbox.site.admin import AdminError
from chess_variant_sandbox.site.browse import BrowseError
from chess_variant_sandbox.site.db import ConfigError, connect, database_url
from chess_variant_sandbox.site.identity import Invite, Refused, Viewer
from chess_variant_sandbox.site.mail import MailError, Message
from chess_variant_sandbox.site.matches import MatchError
from chess_variant_sandbox.site.migrate import MigrationError, Report
from chess_variant_sandbox.site.quota import QuotaError
from chess_variant_sandbox.site.rating import Player, PlayerKey, RatingError
from chess_variant_sandbox.site.routes import ALLOWED, OWN, PAGES, WITHHELD, Route, find
from chess_variant_sandbox.site.server import SiteApp, serve
from chess_variant_sandbox.site.suggestions import SuggestionError
from chess_variant_sandbox.site.unbeaten import Tag
from chess_variant_sandbox.site.votes import VoteError

__all__ = [
    "ALLOWED",
    "OWN",
    "PAGES",
    "AdminError",
    "BrowseError",
    "ConfigError",
    "Invite",
    "MailError",
    "MatchError",
    "Message",
    "MigrationError",
    "Player",
    "QuotaError",
    "PlayerKey",
    "RatingError",
    "Refused",
    "Report",
    "Route",
    "SiteApp",
    "SuggestionError",
    "Tag",
    "Viewer",
    "VoteError",
    "WITHHELD",
    "admin",
    "browse",
    "connect",
    "database_url",
    "find",
    "identity",
    "legality",
    "load",
    "mail",
    "matches",
    "pages",
    "provisional",
    "quota",
    "rating",
    "serve",
    "suggestions",
    "unbeaten",
    "votes",
]
