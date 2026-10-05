"""A score a visitor's browser measured, which the server has checked and not measured itself.

Q312, the first row of `docs/plan-client-compute.md` §8 after the probe. The browser runs the
whole package under Pyodide, plays and scores a draft, and sends the records. The server
checks them in three tiers (§5): it walks every game for legal play, it replays a sample of
them byte for byte, and on request it measures the game itself. This module is where a
result waits between those tiers. It walks nothing and replays nothing -- those are the
tiers' own jobs, and tier one's is `legality` (Q457) -- and it is the one place a
provisional number is stored, moved from state to state, and read.

## Official keeps its one meaning

A score is *official* when the server produced it from games it played itself, which is an
ordinary `measurements` row read off an ordinary `runs/<hash>/canonical.json`. Nothing here
writes either. A provisional result has a table of its own and records under
`runs/provisional/`, so the index sort, *Unbeaten*, the score badges and nomination, which
read only `measurements` and `canonical.json`, cannot reach it. `Basis.played` is the
second door: a browser-played stamp is never current against the server's own basis, so an
official reader handed one would show no number.

## The five states, and who sees each

| state | reached by | who sees it |
|---|---|---|
| `legal` | tier one: every game walked legal, scored by the server | its designer alone |
| `spot-checked` | tier two: a sample replayed exactly | everybody who can see the game |
| `confirmed` | tier three: the server's own measurement landed | nobody here; the official score shows |
| `mismatch` | a check that failed | nobody |
| `superseded` | a newer result for the game | nobody |

`spot-checked` reaches everybody by the user's answer A of 2026-09-18, which they kept on
2026-09-25 after T2 found that a few forged games can move chess by several times its own
error (`docs/reference/forgery-shift.md`). The label carries that: the number says it was
measured on somebody's own machine and spot-checked, never that it is an official score.

**The designer is the game's owner, and only the owner sends a result.** So *its designer*
means the same person in every sentence here. A game kept in the library has no owner and
is measured by the server alone.

## A result shows only under a matching basis

The stored stamp is the server's, taken when the result was recorded: the same game,
roster, seed, weights and fingerprints it would stamp for its own one-run measurement, with
`played` saying a browser played the games and `checked` saying how far the server looked.
A reader judges it against today's basis in the same shape, and a new weights version, a
core rebuild or changed tables hide it as they hide every other stale score. The server does
not replay a provisional result, so nothing but the designer's browser repairs one.

**The same comparison refuses a result on its way in.** A browser holding an old cached
module sends an old build fingerprint, and `refusal` names it, so the page can fetch the
current module and measure again.

Nothing here commits. The caller owns the transaction, the rule `votes` and `matches`
follow. Deleting records from disk is `sweep`'s, run after the commit, so a rolled-back
transaction never takes files with it.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from chess_variant_sandbox.measurement import (
    BROWSER,
    LEGAL,
    SAMPLE,
    Basis,
    FitReading,
    read_check,
    sample_check,
)

SPOT_CHECKED = "spot-checked"
CONFIRMED = "confirmed"
MISMATCH = "mismatch"
SUPERSEDED = "superseded"
STATES: tuple[str, ...] = (LEGAL, SPOT_CHECKED, CONFIRMED, MISMATCH, SUPERSEDED)
"""The five states of §5, as the table's check constraint spells them. `LEGAL` is
`measurement.LEGAL`, the same word the stamp's `checked` carries."""

STANDING: tuple[str, ...] = (LEGAL, SPOT_CHECKED)
"""The states a result can be shown in. At most one result a game is in one of them."""

FROM: dict[str, tuple[str, ...]] = {
    SPOT_CHECKED: (LEGAL,),
    CONFIRMED: (LEGAL, SPOT_CHECKED),
    MISMATCH: (LEGAL, SPOT_CHECKED),
    SUPERSEDED: (LEGAL, SPOT_CHECKED),
}
"""Which states each state is reached from. `legal` is where a result starts, and the
last three are where it ends."""

DIRECTORY = "provisional"
"""Where the records live, under the runs root. Never `runs/<hash>-<seed>/`."""

RECORD_FILES = re.compile(
    r"^(games\.jsonl|tournament\.json|analysis\.json|index-v\d+\.json"
    r"|fit\.json|fit\.jsonl|accept\.jsonl)$"
)
"""The files a result may carry: `slices.measure_in_slices`' four, the fit beside them, and
never the fifth. `canonical.json` is what every official reader trusts, and only the
server's own run writes it."""

GAMES = "games.jsonl"

FIT = "fit.json"
"""The fit's verdict as tier one re-derived it, in `fitted/`'s own shape. The tournament's
`fit=` players cannot be replayed without its vector, so tier two needs it."""

FIT_GAMES = "fit.jsonl"
"""The games the fit was regressed on, as tier one walked them."""

ACCEPT_GAMES = "accept.jsonl"
"""The acceptance match's games, as tier one walked them; empty when the fit was refused
before its match. Tier two samples these and the fit's at the tournament's rate (§5)."""


INCOMPLETE = "The score arrived incomplete; measure it again."
CHANGED = ("The site has changed how it scores games since; this score is out of date. "
           "Measure it again.")
"""The two refusals a person who measured a game in their browser reads (D37). Which part
of the stamp did not match is the maintainer's, and goes to the server's log."""


class ProvisionalError(ValueError):
    """A provisional result cannot be stored or moved as asked.

    A `ValueError` like `votes.VoteError`, so a route answers 400 with the sentence.
    """


@dataclass(frozen=True)
class Result:
    """One provisional result, as its row holds it."""

    id: int
    game: str
    submitter: int
    basis: Basis
    headline: float | None
    viable: bool | None
    state: str
    job: int | None = None
    directory: str | None = None
    created: datetime | None = None
    stale_since: datetime | None = None

    @property
    def sample(self) -> int | None:
        """How many games tier two replayed, or `None` before it has."""
        found = read_check(self.basis.checked)
        return found[1] if found is not None and found[0] == SAMPLE else None

    def visible_to(self, viewer: int | None) -> bool:
        """Whether this person may see the number at all, before any basis is asked.

        A standing result is always its designer's to see. Anybody else sees one only
        when the row says `spot-checked` and the stamp names its sample, so a row and a
        stamp that disagree fall to the narrower audience.
        """
        if self.state not in STANDING:
            return False
        if viewer is not None and viewer == self.submitter:
            return True
        return self.state == SPOT_CHECKED and self.sample is not None

    def payload(self, viewer: int | None) -> dict[str, Any]:
        """What the page draws: the number, the state it was checked to, and whose it is.

        Its own key on a card, `provisional`, never `index`: the listing sorts on `index`,
        and a number under another key is one no sort reads. The page words the two
        labels from `state` and `yours`.
        """
        return {
            "index": self.headline,
            "viable": self.viable,
            "state": self.state,
            "sample": self.sample,
            "yours": viewer is not None and viewer == self.submitter,
            "seeds": list(self.basis.seeds),
            "weights": self.basis.weights_version,
        }


def refusal(submitted: Mapping[str, Any] | None, current: Basis) -> str | None:
    """Why the server will not take this result, or `None` when it matches.

    `submitted` is the stamp the browser made, under Pyodide, from the wheel's own
    sources. `current` is the server's basis for this game's one-run measurement, as it
    would stamp its own. The build fingerprint is asked first and named on its own,
    because it is the common case and it has a remedy the page acts on: a browser holding
    an old cached module fetches the current one.

    `fit`, which objective the fitted rungs played, must be stamped, and it is compared
    wherever `current` knows the answer: from the fit store before the walk, when the game
    has one, and from the server's own regression after it (`refitted`).
    """
    stamp = Basis.from_dict(dict(submitted) if isinstance(submitted, Mapping) else None)
    if stamp is None:
        return "it carries no basis the server can read, so nothing says what it measured"
    if stamp.played != BROWSER:
        return "its basis does not say a browser played the games"
    if stamp.checked:
        return "its basis says how far the server checked it, which only the server says"
    there, here = _build(stamp.engine), _build(current.engine)
    if there != here:
        return (
            f"its games were played by a core built from other sources (build {there}; "
            f"this server runs {here}), so the page fetches the current module and "
            "measures again"
        )
    if not stamp.fit:
        # A browser re-derives the fit on every measurement, so its stamp can always say
        # which objective the fitted rungs played. Absent on a server's stamp means *made
        # before the field*, and is read through the fit store; nothing a browser sends
        # was made before it, so absence here says nothing and is refused (Q468).
        return "its basis does not say which objective its fitted players played"
    staleness = stamp.stale_against(current.in_browser())
    if staleness.needs:
        return "it was not measured on what the server measures now: " + "; ".join(
            staleness.reasons
        )
    return None


def other_build(submitted: Mapping[str, Any] | None, current: Basis) -> bool:
    """Whether this stamp's games were played by a core built from other sources.

    The case `refusal` asks first and names on its own, as a predicate, so a caller can
    tell the page to fetch the current manifest rather than just print the sentence.
    """
    stamp = Basis.from_dict(dict(submitted) if isinstance(submitted, Mapping) else None)
    if stamp is None or stamp.played != BROWSER or stamp.checked:
        return False
    return _build(stamp.engine) != _build(current.engine)


def _build(engine: str) -> str:
    """The core's build fingerprint out of an `engine` stamp, `<sources>-<build>`."""
    return engine.rpartition("-")[2]


def refitted(current: Basis, player: str) -> Basis:
    """`current` with the objective the server's own regression of the fit games plays.

    `player` is `corpus.player_digest` of the verdict tier one re-derived. The server's
    one-run basis reads `Basis.fit` from the fit store, and a visitor's design has none
    (the browser re-derives its fit and the server never stores it), so before the walk
    there is nothing to compare the stamp's `fit` with. After it, this is what the server
    would play, derived on this core, and `refusal` against it compares `fit` as it
    compares every other field. It is also the stamp `record` stores (Q468).
    """
    core = _build(current.engine)
    return replace(
        current,
        fit=player,
        fit_reading=FitReading(player=player, made_on=core, verified_on=frozenset({core})),
    )


def record(
    connection: Any,
    runs_dir: Path,
    *,
    game: str,
    submitter: int,
    submitted: Mapping[str, Any] | None,
    current: Basis,
    headline: float | None,
    viable: bool | None,
    files: Mapping[str, str],
    job: int | None = None,
) -> Result:
    """Store a result that has passed tier one, as `legal`, with its records.

    Tier one is the caller: it has walked every game for legal moves and outcomes and
    scored the records with the server's own metrics, so `headline` and `viable` are the
    server's numbers and never the browser's. What is checked here is what this module
    owns: the stamp matches the server's basis (`refusal`), the game is the submitter's
    own, and the files are the four a measurement writes and the fit's three
    (`RECORD_FILES`).

    The game's standing result, if it has one, is superseded in the same transaction.
    Its records stay on disk until `sweep` runs after the commit.

    The new records are written before the caller commits, so that a failed write
    rolls the row back with it. A transaction the caller rolls back after this returns
    leaves a directory no row names; nothing reads it, and `sweep` deletes only what a
    committed row ties it to, so it can never race a result still being recorded.
    """
    # What the person who measured it reads is one of two sentences (D37); the reason in
    # the stamp's own terms goes to the server's log, for whoever runs the site.
    reason = refusal(submitted, current)
    if reason is not None:
        print(f"provisional: the result for {game} was refused: {reason}", file=sys.stderr)
        raise ProvisionalError(
            CHANGED if reason.startswith(("it was not measured on what", "its games were"))
            else INCOMPLETE
        )
    if current.ruleset_hash != game:
        print(f"provisional: the basis is for the game {current.ruleset_hash}, not {game}",
              file=sys.stderr)
        raise ProvisionalError(INCOMPLETE)
    unknown = sorted(name for name in files if not RECORD_FILES.match(name))
    if unknown:
        print(f"provisional: a result for {game} also had {', '.join(unknown)}",
              file=sys.stderr)
        raise ProvisionalError(INCOMPLETE)
    if GAMES not in files:
        raise ProvisionalError("a result without its games cannot be checked")
    check_owner(connection, game, submitter)

    stamp = current.in_browser(LEGAL)
    withdraw(connection, game)
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into provisional (game, submitter, basis, headline, viable, state, job)"
            " values (%s, %s, %s::jsonb, %s, %s, %s, %s) returning id, created",
            (game, submitter, json.dumps(stamp.as_dict()), headline, viable, LEGAL, job),
        )
        found = cursor.fetchone()
        number, created = int(found[0]), found[1]
        where = f"{DIRECTORY}/{number}"
        cursor.execute(
            "update provisional set directory = %s where id = %s", (where, number)
        )
    target = Path(runs_dir) / DIRECTORY / str(number)
    try:
        target.mkdir(parents=True, exist_ok=False)
        for name in sorted(files):
            (target / name).write_text(files[name], encoding="utf-8", newline="")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return Result(
        id=number,
        game=game,
        submitter=submitter,
        basis=stamp,
        headline=headline,
        viable=viable,
        state=LEGAL,
        job=job,
        directory=where,
        created=created,
    )


def check_owner(connection: Any, game: str, submitter: int) -> None:
    """Refuse anyone but the game's owner, and a game that is gone or has no owner.

    `record` asks it, and tier one's `legality.file` asks it before a walk is queued.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select owner, revoked, deleted from games where hash = %s", (game,)
        )
        found = cursor.fetchone()
    if found is None:
        raise ProvisionalError("This game isn't on the site.")
    owner, revoked, deleted = found
    if deleted is not None or revoked is not None:
        raise ProvisionalError("this game has been taken off the site")
    if owner is None:
        raise ProvisionalError(
            "a game kept in the library is measured by the server alone"
        )
    if int(owner) != int(submitter):
        raise ProvisionalError(
            "only the person who made a game can send a score for it from their machine"
        )


def withdraw(connection: Any, game: str) -> int:
    """Supersede the game's standing result, when its rules are taken away for good.

    `admin.delete_game` calls it. The next `sweep` then deletes the records, which are
    games of rules nobody can read any more. Returns how many rows it moved.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update provisional set state = %s, updated = now()"
            " where game = %s and state = any(%s)",
            (SUPERSEDED, game, list(STANDING)),
        )
        return int(cursor.rowcount or 0)


def spot_checked(connection: Any, result: int, n: int, job: int | None = None) -> Result:
    """Tier two passed: n games replayed byte for byte. The stamp names n."""
    return _move(connection, result, SPOT_CHECKED, checked=sample_check(n), job=job)


def mismatched(connection: Any, result: int, job: int | None = None) -> Result:
    """A check failed. The result is never shown again, and its records are kept."""
    return _move(connection, result, MISMATCH, job=job)


def confirmed(connection: Any, result: int, job: int | None = None) -> Result:
    """Tier three's own measurement landed; the official score shows from now on."""
    return _move(connection, result, CONFIRMED, job=job)


def _move(
    connection: Any, result: int, to: str, checked: str | None = None, job: int | None = None
) -> Result:
    with connection.cursor() as cursor:
        cursor.execute(f"select {_COLUMNS} from provisional where id = %s for update", (result,))
        found = cursor.fetchone()
    if found is None:
        raise ProvisionalError(f"no provisional result numbered {result}")
    now = _to_result(found)
    if now.state not in FROM[to]:
        raise ProvisionalError(f"a result that is {now.state} cannot become {to}")
    stamp = now.basis if checked is None else now.basis.in_browser(checked)
    with connection.cursor() as cursor:
        cursor.execute(
            "update provisional set state = %s, basis = %s::jsonb,"
            " job = coalesce(%s, job), updated = now() where id = %s"
            f" returning {_COLUMNS}",
            (to, json.dumps(stamp.as_dict()), job, result),
        )
        return _to_result(cursor.fetchone())


def standing(connection: Any, games: Collection[str]) -> dict[str, tuple[Result, Any]]:
    """Each game's standing result with the game's rules, keyed by game.

    The rules come back beside the result because judging its basis needs the tables
    fingerprint, which is a reading of the spec, and the store already holds the spec.
    """
    wanted = sorted({str(game) for game in games})
    if not wanted:
        return {}
    columns = ", ".join(f"p.{name.strip()}" for name in _COLUMNS.split(","))
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {columns}, g.spec from provisional p join games g on g.hash = p.game"
            " where p.game = any(%s) and p.state = any(%s)"
            " and g.revoked is null and g.deleted is null",
            (wanted, list(STANDING)),
        )
        rows = cursor.fetchall()
    answer: dict[str, tuple[Result, Any]] = {}
    for row in rows:
        # A row whose stamp cannot be read shows nothing, rather than taking the whole
        # listing down with it.
        read = _read(row[:-1])
        if read is not None:
            answer[read.game] = (read, row[-1])
    return answer


def shown(
    connection: Any,
    games: Collection[str],
    viewer: int | None,
    current: Callable[[str, Any], Basis | None],
) -> dict[str, dict[str, Any]]:
    """What this person is shown for each game, keyed by game; absent is nothing.

    `current(game, spec)` is the basis the server would stamp on its own one-run
    measurement of that game now. A result whose stamp does not match it -- in the same
    shape, played in a browser -- is hidden, as every stale score is, and a result this
    person may not see is absent the same way, so a reader cannot tell the two apart
    from outside.
    """
    answer: dict[str, dict[str, Any]] = {}
    for game, (result, spec) in standing(connection, games).items():
        if not result.visible_to(viewer):
            continue
        now = current(game, spec)
        if now is None:
            continue
        if result.basis.stale_against(now.in_browser(result.basis.checked)).needs:
            continue
        answer[game] = result.payload(viewer)
    return answer


Current = Callable[[str, Any], Basis | None]
"""The server's basis for a game's one-run measurement now, from its hash and its rules."""


def one_run(game: str, spec: Any) -> Basis | None:
    """What the server would stamp on its own one-run measurement of this game now.

    `None` where there are no rules to compile, which shows nothing: without the tables
    fingerprint there is nothing to match a stamp against. The default `Current`.
    """
    if not isinstance(spec, dict):
        return None
    from chess_variant_sandbox import corpus  # the site's readers need it; the store's do not

    return corpus.current_basis(game, spec=spec, seeds=corpus.ONE_SEED)


def attached(connection: Any, game: str, current: Current) -> dict[str, Any] | None:
    """The spot-checked reading on today's basis this game carries, or `None` (Q503e).

    What *Publish* needs (`docs/plan-game-pipeline.md` §2.2, answer Q5): a result the
    designer's browser measured, tier one walked and tier two replayed in part, whose
    stamp matches what the server would stamp now. A `legal` result is not one -- nobody
    has replayed a game of it -- and neither is a stale one. It is read as a visitor reads
    it (`shown` for nobody), so it is exactly the number everybody can see.
    """
    found = shown(connection, [game], None, current).get(game)
    return found if found is not None and found.get("state") == SPOT_CHECKED else None


def counted(
    connection: Any, game: str, current: Current, grace_days: int, now: datetime | None = None
) -> dict[str, Any] | None:
    """The reading the shortlist's bar counts: `attached`, or a stale one in its grace.

    `docs/plan-game-pipeline.md` §2.4: only the designer's browser repairs a trial game's
    reading, so a reading a core rebuild or a weights version made stale still counts for
    `grace_days` from `stale_since`, and then stops until it is measured again. A stale
    reading the server had not yet noticed is stamped now (`mark_stale`), so its clock
    starts no later than the first time anybody asks. The payload says `stale: true` and
    `staleSince`, so the page can say whose move it is.
    """
    found = attached(connection, game, current)
    if found is not None:
        return found
    result_and_spec = standing(connection, [game]).get(game)
    if result_and_spec is None:
        return None
    result, spec = result_and_spec
    if result.state != SPOT_CHECKED or result.sample is None:
        return None
    since = result.stale_since
    if since is None:
        basis = current(game, spec)
        if basis is None:
            return None
        if not result.basis.stale_against(basis.in_browser(result.basis.checked)).needs:
            return None
        since = _stamp_stale(connection, result.id)
    moment = now if now is not None else _now(connection)
    if (moment - since).total_seconds() >= grace_days * 86400:
        return None
    return {**result.payload(None), "stale": True, "staleSince": since.isoformat()}


def mark_stale(connection: Any, current: Current) -> int:
    """Stamp `stale_since` on every standing result whose stamp no longer matches, now.

    The runner calls it each time it starts, which is each deploy, which is when a basis
    moves; `counted` stamps the ones it meets in between. A result that matches again (a
    deploy rolled back) has its stamp cleared. Returns how many rows it stamped.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select p.id, p.basis, p.stale_since, p.game, g.spec from provisional p"
            " join games g on g.hash = p.game where p.state = any(%s)",
            (list(STANDING),),
        )
        rows = cursor.fetchall()
    stamped = 0
    for number, stamp, since, game, spec in rows:
        basis = Basis.from_dict(stamp if isinstance(stamp, dict) else None)
        now_basis = current(str(game), spec)
        if basis is None or now_basis is None:
            continue
        stale = basis.stale_against(now_basis.in_browser(basis.checked)).needs
        if stale and since is None:
            _stamp_stale(connection, int(number))
            stamped += 1
        elif not stale and since is not None:
            with connection.cursor() as cursor:
                cursor.execute(
                    "update provisional set stale_since = null where id = %s", (int(number),)
                )
    return stamped


def _stamp_stale(connection: Any, number: int) -> datetime:
    with connection.cursor() as cursor:
        cursor.execute(
            "update provisional set stale_since = coalesce(stale_since, now())"
            " where id = %s returning stale_since",
            (number,),
        )
        return cast(datetime, cursor.fetchone()[0])


def _now(connection: Any) -> datetime:
    with connection.cursor() as cursor:
        cursor.execute("select now()")
        return cast(datetime, cursor.fetchone()[0])


# ------------------------------------------------------------- the account's flag


FLAGGED = (
    "A score sent from your computer did not match the site's own replay of it, so your "
    "account can't publish until the person who runs the site has looked at it."
)
"""What a flagged designer reads at *Publish*, and beside its disabled button."""


def flag(connection: Any, user: int) -> None:
    """Flag this account after a mismatch. Its first flag's date stands."""
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set flagged = coalesce(flagged, now()) where id = %s", (int(user),)
        )


def flagged(connection: Any, user: int | None) -> bool:
    """Whether a spot check has flagged this account and no admin has cleared it."""
    if user is None:
        return False
    with connection.cursor() as cursor:
        cursor.execute("select flagged from users where id = %s", (int(user),))
        found = cursor.fetchone()
    return found is not None and found[0] is not None


def flagged_accounts(connection: Any) -> list[dict[str, Any]]:
    """Every flagged account for `/admin`, newest flag first, with what its check found.

    `found` is the newest failed spot check's error: the phase, game, seed, ply and both
    build fingerprints (`spotcheck`), which is what tells a core bug from a forgery.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select u.id, u.handle, u.flagged, (select j.error from jobs j where"
            " j.requester = u.id and j.kind = 'sample' and j.state = 'refused'"
            " order by j.id desc limit 1) from users u where u.flagged is not null"
            " order by u.flagged desc, u.id"
        )
        rows = cursor.fetchall()
    return [
        {
            "user": int(row[0]),
            "handle": None if row[1] is None else str(row[1]),
            "flagged": row[2].isoformat() if row[2] is not None else None,
            "found": None if row[3] is None else str(row[3]),
        }
        for row in rows
    ]


def clear_flag(connection: Any, user: int) -> bool:
    """An admin's act: the account may publish, and its scores are spot-checked again.

    The mismatched result stays `mismatch`, with its records, for the log. Returns whether
    there was a flag to clear.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "update users set flagged = null where id = %s and flagged is not null",
            (int(user),),
        )
        return bool(cursor.rowcount)


def sweep(connection: Any, runs_dir: Path) -> list[str]:
    """Delete the records of every result that no longer needs them, after the commit.

    A confirmed result has the server's own records, and a superseded one has a newer
    result's. A mismatch keeps its records for the admin log. Returns the directories
    removed, relative to the runs root, and clears each row's `directory`.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, directory from provisional"
            " where state = any(%s) and directory is not null order by id",
            ([CONFIRMED, SUPERSEDED],),
        )
        found = [(int(row[0]), str(row[1])) for row in cursor.fetchall()]
    root = Path(runs_dir)
    for number, where in found:
        # The column's check constraint already holds this shape; asking again here
        # means a hand-edited row can never point the delete outside the directory.
        if where != f"{DIRECTORY}/{number}":
            raise ProvisionalError(f"result {number} names its records as {where!r}")
        shutil.rmtree(root / DIRECTORY / str(number), ignore_errors=True)
    if found:
        with connection.cursor() as cursor:
            cursor.execute(
                "update provisional set directory = null where id = any(%s)",
                ([number for number, _ in found],),
            )
    return [where for _, where in found]


_COLUMNS = (
    "id, game, submitter, basis, headline, viable, state, job, directory, created,"
    " stale_since"
)


def _to_result(found: Any) -> Result:
    read = _read(found)
    if read is None:
        # The table's checks hold `played` and `checked` to what this module writes, so
        # an unreadable stamp is a row somebody wrote by hand. It is refused by name
        # rather than read as some basis it does not say.
        raise ProvisionalError(f"provisional result {found[0]} has no readable basis")
    return read


def _read(found: Any) -> Result | None:
    """The row as a result, or `None` when its stamp cannot be read, which shows nothing."""
    stamp = Basis.from_dict(found[3] if isinstance(found[3], dict) else None)
    if stamp is None:
        return None
    return Result(
        id=int(found[0]),
        game=str(found[1]),
        submitter=int(found[2]),
        basis=stamp,
        headline=None if found[4] is None else float(found[4]),
        viable=None if found[5] is None else bool(found[5]),
        state=str(found[6]),
        job=None if found[7] is None else int(found[7]),
        directory=None if found[8] is None else str(found[8]),
        created=found[9],
        stale_since=found[10],
    )
