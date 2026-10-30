"""From a draft to a number: the rungs, the lineage and the nomination.

The designer's side of `corpus`. A draft climbs the rungs every candidate climbs, and this
module says which rung it is on, what the next one costs, which games it descends from and
whether somebody has asked a maintainer to promote it.

Nothing here decides a rule question or scores a game. The screen is `loop/screen.py`'s,
the measurement is `corpus.refresh` through `web/keeping.Measurements`, and a promotion
blocker is `corpus.promotion_blocker`. A number leaves this module only when its basis
matches, as it does everywhere else in the interface.

**The editor never writes to `rulesets/` or `archive/`.** A nomination is a line in
`runs/designed/nominations.jsonl`, which git ignores, and a maintainer promotes with
`chess-variant-sandbox promote` and a reason.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chess_variant_sandbox import corpus
from chess_variant_sandbox.loop.archive import Archive
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.plain import plain_problems
from chess_variant_sandbox.web.designing import DESIGNED_RUN, DesignError, settled

NOMINATIONS = "nominations.jsonl"

# How far a lineage tree is drawn. A tree past these is cut, and says so.
MAX_DEPTH = 12
MAX_NODES = 60


# ------------------------------------------------------------------ the rungs


def took(seconds: float) -> str:
    """A wall-clock time in the unit a person would say it in, the unit named.

    The long rung used to print minutes with no unit, so an eight-second measurement
    read *took 0.1*.
    """
    if seconds < 59.5:
        whole = max(1, round(seconds))
        return f"{whole} second{'' if whole == 1 else 's'}"
    minutes = seconds / 60
    if minutes < 9.95:
        shown = f"{minutes:.1f}".removesuffix(".0")
        return f"{shown} minute{'' if shown == '1' else 's'}"
    return f"{round(minutes)} minutes"


# When a rung happens, in a visitor's words (D15): *happens as you edit*, *when you press
# save*, *not on this site yet*, *decided by the person who runs the site*.
_AS_YOU_EDIT = "happens as you edit"

SITE_WORDS: dict[str, dict[str, str]] = {
    "validate": {"what": "every part is filled in and the parts fit together, the same "
                         "check every game here has passed",
                 "runs": _AS_YOU_EDIT},
    "smoke": {"runs": _AS_YOU_EDIT},
    "screen": {"name": "a computer player can win it",
               "what": "two small computer players play eight games",
               "runs": _AS_YOU_EDIT},
    "save": {"name": "saved as your draft",
             "what": "kept on this site as your draft; the rules never change under it",
             "runs": "when you press save"},
    "canonical": {"name": "measured against every computer player",
                  "what": "each computer player plays every other, then the games are "
                          "analysed and scored",
                  "cost": "several minutes of computer time",
                  "runs": "not on this site yet: the site does not measure designed games "
                          "yet"},
    "nominate": {"name": "made one of the games every score is measured against",
                 "what": "the person who runs the site reads it and decides, with a reason",
                 "cost": "a person's time",
                 "runs": "decided by the person who runs the site"},
}
"""The rungs in a site visitor's words (Q576, the first-visit study's F6). The checkout's
maintainer has a terminal, a `runs/` directory and the calibration corpus, and the local
sentences name them; a visitor has none of the three. Only the words differ: the keys,
costs and order are the same rungs. Each rung carries its own under `site`, and the page
draws them where it is the site (`ACCOUNTS`, the one switch it tells the two apart by)."""


def rungs(last_seconds: float | None = None) -> list[dict[str, Any]]:
    """Every rung between a draft and the corpus, in order, with what each costs here.

    The cheap costs are measured on each call and arrive with the answer they belong to
    (`check.smokeSeconds`, a screen's `seconds`). The long rung has no stored timing, so
    its cost is this process's last measurement when there is one and an estimate named
    as an estimate when there is not. A rung's `site` is `SITE_WORDS`' words for it.
    """
    games = len(corpus.CANONICAL_AGENTS) ** 2 * corpus.CANONICAL_GAMES
    if last_seconds is None:
        long_cost = (
            "minutes, by an estimate of 5 to 10 on a small computer; not yet timed on "
            "this one"
        )
    else:
        long_cost = f"{took(last_seconds)} the last time one ran here"
    return [{**rung, "site": SITE_WORDS.get(rung["key"], {})}
            for rung in _local_rungs(games, long_cost)]


def _local_rungs(games: int, long_cost: str) -> list[dict[str, Any]]:
    """The rungs in the maintainer's words, the ones the local application shows."""
    return [
        # Named for what it checks, on both servers: *make sense* promised a judgement
        # the check does not make (Q681). The site's refusal says *not a complete game
        # yet* (D28), so the step says the same.
        {"key": "validate", "name": "the rules are a complete game",
         "what": "the rule schema, the same check the command line runs",
         "cost": "milliseconds", "runs": "inline"},
        {"key": "smoke", "name": "it plays and ends",
         "what": "150 games of random moves: can anybody move, does anybody win",
         "cost": "about a fifth of a second", "runs": "inline"},
        {"key": "screen", "name": "a searching player can win it",
         "what": "two small searches, eight games between them",
         "cost": "under a second for the templates and chess", "runs": "inline"},
        {"key": "save", "name": "kept as a candidate",
         "what": "written to runs/designed/, which git ignores; the rules never change "
                 "under it",
         "cost": "milliseconds", "runs": "on request"},
        {"key": "canonical", "name": "measured on the yardstick's roster",
         "what": f"{len(corpus.CANONICAL_AGENTS)} players, {corpus.CANONICAL_GAMES} games "
                 f"a pairing, {games} games, then the analysis and the score",
         "cost": long_cost, "runs": "queued, one game at a time"},
        {"key": "nominate", "name": "nominated for the yardstick",
         "what": "a maintainer reads it and promotes it with a reason, or does not",
         "cost": "a person's decision", "runs": "by a maintainer"},
    ]


def screen_draft(spec: Any) -> dict[str, Any]:
    """The middle rung, run on a draft that need not be saved."""
    from chess_variant_sandbox.loop.screen import screen

    if not isinstance(spec, dict):
        raise DesignError("That draft could not be read.")
    spec = settled(spec)
    problems = validate_spec(spec)
    if problems:
        said = plain_problems(problems, spec)
        raise DesignError("These rules aren't a complete game yet: " + " ".join(said[:4]))
    started = time.perf_counter()
    result = screen(spec)
    return {
        **result.as_dict(),
        "hash": spec_hash(spec),
        "seconds": round(time.perf_counter() - started, 3),
    }


# ------------------------------------------------------------------ what a game measures


def measured(game: Any) -> dict[str, Any]:
    """A game's measurement in the interface's three states.

    `index` is present only when the measurement is current. A stale one carries its
    repair and no number: a draft is compared against it, and a provisional estimate is
    never compared (`library.current_only`). `seeds` is how many runs the number is the
    mean of, under the card's own key (`library.Game.summary`), because a mean of four runs and one run are
    not the same claim and the page says which beside the number.
    """
    from chess_variant_sandbox.web.library import current_only

    score = current_only(game.score)
    if score is not None:
        return {"state": "current", "index": score.get("index"),
                "viable": score.get("viable"), "measured": score.get("measured") is not False,
                "weights": score.get("weights_version"),
                "seeds": list(score.get("seeds") or [])}
    staleness = game.staleness
    if staleness is not None:
        return {"state": "stale", "needs": staleness.get("needs") or "replay",
                "summary": staleness.get("summary", "")}
    return {"state": "unscored"}


# ------------------------------------------------------------------ lineage


def _designed_parents(runs_dir: Path) -> dict[str, str]:
    """Each hand-built game's parent as the archive recorded it, parents outside the
    archive included. `library._lineage` walks only parents inside the archive, so a
    first save forked from a committed game would otherwise have no parent."""
    archive = Archive(Path(runs_dir) / DESIGNED_RUN)
    return {entry.hash: entry.parent for entry in archive.entries if entry.parent}


def _parent_of(game: Any, designed: dict[str, str]) -> str | None:
    parent = game.ancestry().get("parent") or {}
    return parent.get("hash") or designed.get(game.hash)


def lineage(
    games: dict[str, Any],
    digest: str,
    runs_dir: Path,
    parent: str = "",
    name: str = "",
) -> dict[str, Any]:
    """The family a game belongs to, drawn from its oldest known ancestor down.

    Every save is its own node with its own hash and its own measurement, so four saves
    of one idea read as four games. An unsaved draft is shown under `parent` and marked
    as a draft. A parent the library cannot name ends the walk as a bare hash.
    """
    designed = _designed_parents(runs_dir)
    nominated = nominated_hashes(runs_dir)
    parents = {h: _parent_of(g, designed) for h, g in games.items()}
    children: dict[str, list[str]] = {}
    for child, of in parents.items():
        if of:
            children.setdefault(of, []).append(child)

    draft = digest not in games
    if draft and parent:
        children.setdefault(parent, []).append(digest)

    line = [digest]
    here = parent if draft else parents.get(digest)
    while here and here not in line:
        line.insert(0, here)
        here = parents.get(here)
    if draft and not parent:
        line = [digest]

    count = 0
    cut = False

    def node(at: str, depth: int, seen: frozenset[str]) -> dict[str, Any]:
        nonlocal count, cut
        count += 1
        game = games.get(at)
        entry: dict[str, Any] = {
            "hash": at,
            "here": at == digest,
            "onLine": at in line,
            "draft": draft and at == digest,
            "known": game is not None,
            "name": game.name if game is not None else (name if at == digest else ""),
            "source": game.source if game is not None else "",
            "measured": measured(game) if game is not None else {"state": "unscored"},
            "nominated": at in nominated,
            "children": [],
        }
        below = sorted(
            (c for c in children.get(at, []) if c not in seen),
            key=lambda c: (games[c].name.lower() if c in games else "", c),
        )
        if below and (depth >= MAX_DEPTH or count >= MAX_NODES):
            cut = True
            return entry
        for child in below:
            if count >= MAX_NODES:
                cut = True
                break
            entry["children"].append(node(child, depth + 1, seen | {child}))
        return entry

    root = line[0]
    return {"root": node(root, 0, frozenset({root})), "line": line, "cut": cut}


# ------------------------------------------------------------------ nomination


def _nominations_path(runs_dir: Path) -> Path:
    return Path(runs_dir) / DESIGNED_RUN / NOMINATIONS


def _read_nominations(runs_dir: Path) -> list[dict[str, Any]]:
    path = _nominations_path(runs_dir)
    if not path.exists():
        return []
    found: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("hash"):
            found.append(row)
    return found


def nominated_hashes(runs_dir: Path) -> set[str]:
    return {str(row["hash"]) for row in _read_nominations(runs_dir)}


def nominate(candidate: corpus.Candidate, runs_dir: Path, note: str = "") -> dict[str, Any]:
    """Mark a candidate for a maintainer. Append-only, and a second mark is the first.

    No measurement is demanded: the list shows each nomination's promotion blocker, and
    the maintainer who reads it is the one who decides.
    """
    if candidate.stage == "committed":
        raise DesignError(f"{candidate.name} is already part of the yardstick")
    for row in _read_nominations(runs_dir):
        if row["hash"] == candidate.hash:
            return {**row, "created": False}
    row = {
        "hash": candidate.hash,
        "name": candidate.name,
        "note": note.strip(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = _nominations_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")
    return {**row, "created": True}


def nominations(
    runs_dir: Path, ruleset_dir: Path, archive_dir: Path
) -> list[dict[str, Any]]:
    """Every nomination, with where the game sits now and what stands in its way."""
    listed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _read_nominations(runs_dir):
        digest = str(row["hash"])
        if digest in seen:
            continue
        seen.add(digest)
        try:
            candidate = corpus.find(
                digest, runs_dir=runs_dir, ruleset_dir=ruleset_dir, archive_dir=archive_dir
            )
        except corpus.CorpusError:
            listed.append({**row, "stage": "missing", "blocker": "the game is gone from runs/"})
            continue
        listed.append({
            **row,
            "stage": candidate.stage,
            "blocker": corpus.promotion_blocker(candidate, runs_dir, ruleset_dir),
            "command": f'uv run chess-variant-sandbox promote {digest} --reason "..."',
        })
    return listed


# ------------------------------------------------------------------ what promotion needs

_COUNT = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"]


def promotion(
    candidate: corpus.Candidate, blocker: str | None, runs_dir: Path
) -> dict[str, str] | None:
    """What stands between a candidate and promotion, in words a visitor can read.

    `corpus.promotion_blocker` is the judgement, and it is shared with `promote` on the
    command line, so it names a terminal command. That is right in a terminal and wrong
    under a button on a web page, where it also said *out of date* about a candidate
    whose one run is exactly what its rung owes. This says the same thing in the page's
    words: how many runs are measured, what they give the game here, and that promotion
    wants four. `command` is the maintainer's way to the four, which the page keeps out
    of a visitor's sight. ``None`` when nothing stands in the way.
    """
    if blocker is None or candidate.stage == "committed":
        return None
    library = corpus.LIBRARY_SEEDS
    four = len(library)
    runs_dir = Path(runs_dir)
    whole = corpus.read_measurement(
        candidate.hash, runs_dir, spec=candidate.spec, seeds=library
    )
    own = corpus.read_measurement(
        candidate.hash, runs_dir, spec=candidate.spec,
        seeds=corpus.expected_seeds(candidate.stage),
    )
    command = f"uv run chess-variant-sandbox test {candidate.hash}"
    games = len(corpus.CANONICAL_AGENTS) ** 2 * corpus.CANONICAL_GAMES
    if whole is None:
        return {
            "says": f"It has no score yet. Measuring it plays one run of {games} games, "
                    f"which gives it a number here. A maintainer promotes a game only on "
                    f"{_COUNT[four]} runs, because one run can put a game on the wrong "
                    "side of the line.",
            "command": command,
        }
    if whole.current:
        if whole.stored.get("measured") is False:
            # The third verdict (plan-teacher §10.16): no verdict, so nothing to promote
            # on, and nothing ruled it out either.
            return {"says": f"Its {_COUNT[four]} runs reached no verdict: its computer "
                            "players could not tell a stronger search from a weaker one "
                            "on most of them. So it is not measured, and is not promoted.",
                    "command": ""}
        if not whole.stored.get("viable"):
            return {"says": f"Its {_COUNT[four]} runs rule it out as not viable, so it "
                            "is not promoted.", "command": ""}
        return {"says": "It is not ready to be promoted yet.", "command": ""}
    measured = len(whole.stored.get("seeds") or [])
    if own is not None and own.current and 0 < measured < four:
        runs = f"{_COUNT[measured].capitalize()} run{'' if measured == 1 else 's'}"
        return {
            "says": f"{runs} {'is' if measured == 1 else 'are'} measured, which gives it "
                    f"its number here. A maintainer promotes a game only on "
                    f"{_COUNT[four]} runs, so the other {_COUNT[four - measured]} "
                    f"{'is' if four - measured == 1 else 'are'} played first.",
            "command": command,
        }
    return {
        "says": "Its score is out of date, so it is measured again before it can be "
                "promoted.",
        "command": command,
    }


# ------------------------------------------------------------------ one draft's standing


def standing(
    games: dict[str, Any],
    digest: str,
    runs_dir: Path,
    ruleset_dir: Path,
    archive_dir: Path,
    measuring: dict[str, Any] | None = None,
    parent: str = "",
    name: str = "",
    last_seconds: float | None = None,
    build: str = "",
) -> dict[str, Any]:
    """Where one draft is on the rungs: saved or not, measured or not, nominated or not."""
    # Imported here, as `library.detail` does: the site's package imports this server.
    from chess_variant_sandbox.site import stages

    game = games.get(digest)
    stage = "draft"
    blocker = None
    needs = None
    if game is not None:
        try:
            candidate = corpus.find(
                digest, runs_dir=runs_dir, ruleset_dir=ruleset_dir, archive_dir=archive_dir
            )
            stage = candidate.stage
            blocker = corpus.promotion_blocker(candidate, runs_dir, ruleset_dir)
            needs = promotion(candidate, blocker, runs_dir)
        except corpus.CorpusError:
            stage = "unknown"
    return {
        "hash": digest,
        "saved": game is not None,
        "stage": stage,
        # The game's own name, for the save rung's *already a library game, ‹name›* (D20).
        "name": game.name if game is not None else "",
        "measured": measured(game) if game is not None else {"state": "unscored"},
        "measuring": measuring,
        "nominated": digest in nominated_hashes(runs_dir),
        # `corpus`'s refusal, which names a terminal command, and the same in the
        # page's words (`promotion`), which is the one a page shows.
        "blocker": blocker,
        "promotion": needs,
        "build": build,
        "rungs": rungs(last_seconds),
        "lineage": lineage(games, digest, runs_dir, parent=parent, name=name),
        # *Where this game stands* (Q503b): the level word, its sentence and the track,
        # from where the file is (`stages.local`), as the checkout's card and game page
        # have them. The site replaces all three with the row's.
        **stages.shown(stages.local(stage)),
    }
