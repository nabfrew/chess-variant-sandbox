"""How far to trust a score: the level, and the evidence behind it (Q646).

`docs/plan-confidence.md` §2.3, **the rule pre-registered 2026-10-02**, and nothing else.
A score is one number its measuring players produced; this module reads, beside it, what
else has checked those players on the game, and says one of three words:

- **questioned** when any piece of evidence *finds more* than the measuring players see,
  current **or stale**: a finding made on older players is not undone by the players
  changing (the asymmetry `premise.Premise.marked` already keeps);
- else **checked** when any piece *agrees* and is **current**: a stale agreement counts
  for nothing;
- else **early estimate**.

**What it reads.** Persisted records only, never an engine and never the site's
database: the score's `runs/<hash>/canonical.json` (the ladder), `runs/<hash>/premise.json`
(the self-duel and Fairy-Stockfish, judged by `premise.read`), the learned evaluator's
stores `nets/<hash>-<procedure>.json` against the score's `Basis.net`, and
`runs/<hash>/convergence.json` (T5 writes it; its shape is `CONVERGENCE_FILE`'s note).
People's counts are handed in as `People` (T3 folds them), never read here.

**What it moves: nothing.** The level is beside the score and never in it. Nothing under
`metrics/` imports this file and it is in no fingerprint set (`tests/test_trust.py`), so
no score, basis, weights version or fingerprint moves when evidence arrives, and a change
to `RULE` stales no measurement: the level is read live, each time, from the evidence.

**What it refuses.** A score that is out of date or absent carries no level (`level` is
``None``): *out of date* is already the whole message (§2.1). A record that cannot be read
reads `UNREADABLE` and *silent*, never a guessed reading.

**People move the level, never the score.** The user answered §5 Q3 A on 2026-10-03
(`docs/archive/decisions.md`): two distinct people clearing the top question a game, and
twenty games by five people with no clear check it. `RULE.people_move_level` is the switch,
``True`` since that answer; turning it back is a dated edit of `RULE`, nothing more.

**The words** (Q647, T2; `plan-confidence.md` §2.1 and §2.2) are here too, beside the rule
whose readings they say: `card` is a library card's chip (the level word and its sentence
for the (i)), `shown` is a game page's line (the level word, one sentence, and the *more*
list, each piece with at most one plain count and a date), and `with_people` folds the
site's counts into a reading already made from the files. The level is a word only (§5 Q2,
answered A): no count is on the line, and the counts live under *more*. A visitor's words
carry no floor, no score out of 1 and no player's spelling; the maintainer's numbers ride
in `numbers`, which the site sends to an admin alone.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The level words (§2.2). Never *verified* or *certified*: an estimate is never final.
EARLY = "early estimate"
CHECKED = "checked"
QUESTIONED = "questioned"

# Evidence kinds (§1), in the order the page lists them.
LADDER = "ladder"  # E-L
SELF_DUEL = "self-duel"  # E-C
FAIRY = "fairy-stockfish"  # E-F
NET = "trained player"  # E-N
CONVERGENCE = "convergence"  # E-V
PEOPLE = "people"  # E-H
KINDS = (LADDER, SELF_DUEL, FAIRY, NET, CONVERGENCE, PEOPLE)

# An item's state against its own basis.
CURRENT = "current"
STALE = "stale"
ABSENT = "absent"
UNREADABLE = "unreadable"

# What an item says.
FINDS_MORE = "finds more"
AGREES = "agrees"
SILENT = "silent"
REPORTED = "reported"  # the self-duel: shown, never counted (the user, 2026-10-01)

CONVERGENCE_FILE = "convergence.json"
"""`runs/<hash>/convergence.json`, T5's record. Read here as: ``move`` (the index's
change between the two measurements), ``sd`` (the outgoing four-run standard deviation)
and ``to``, the `measurement_digest` of the basis of the measurement it compared *into*.
It is current while ``to`` is the score's own basis; the reading is applied here, under
`RULE`, from ``move`` and ``sd``, never taken from a stored verdict."""


@dataclass(frozen=True)
class Rule:
    """§2.3's thresholds, spelled. Pre-registered 2026-10-02; never re-aimed.

    The ladder's own threshold is the index's (`calibration.MEASURABLE_LADDER` and a
    negative `skill_gradient`): the ladder finds more exactly when the stored score
    withholds its ladder components, so it is read from the score, not re-derived.
    """

    fairy_floor: float = 0.30  # general-player.md §3.2 (c); existing
    convergence_sds: float = 2.0  # general-player.md §3.1.4; existing
    people_clears: int = 2  # new, 2026-10-02
    people_games: int = 20  # new, 2026-10-02
    people_people: int = 5  # new, 2026-10-02
    # §5 Q3, answered A by the user on 2026-10-03: people's results move the level, never
    # the score. False reports them under *more* and counts them for nothing.
    people_move_level: bool = True

    def spelled(self) -> str:
        from chess_variant_sandbox.metrics.calibration import MEASURABLE_LADDER

        return (
            f"ladder: finds more when the score withholds it (monotonicity below "
            f"{MEASURABLE_LADDER:g} or a negative gradient); self-duel: reported only; "
            f"fairy-stockfish: finds more below {self.fairy_floor:g}, agrees at or above; "
            "trained player: finds more when an accepted net is not in the score's basis; "
            f"convergence: finds more at a move of {self.convergence_sds:g} sd or more, "
            f"agrees below; people: finds more at {self.people_clears} or more people "
            f"clearing the top, agrees at {self.people_games} or more games by "
            f"{self.people_people} or more people with no clear"
            + ("" if self.people_move_level else ", reported only for now")
            + "; questioned on any finding, current or stale; else checked on a current "
            "agreement; else early estimate"
        )


RULE = Rule()


@dataclass(frozen=True)
class Count:
    """One plain count a sentence may carry: *6 of 10 games*."""

    n: int
    of: int
    unit: str

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "of": self.of, "unit": self.unit}


@dataclass(frozen=True)
class People:
    """People's results against a game's top computer group, as T3 folds them.

    ``games``: finished leaderboard games against the top group; ``people``: distinct
    people who played them; ``cleared``: distinct people who cleared the top (`unbeaten`'s
    own fold); ``latest``: the date of the newest, ISO, or ``None``.
    """

    games: int
    people: int
    cleared: int
    latest: str | None = None


@dataclass(frozen=True)
class Item:
    """One kind of evidence, read: §4 T1's fields."""

    kind: str
    state: str
    reading: str
    count: Count | None = None
    date: str | None = None
    # What T2 turns into words (`plan-confidence.md` §2.2, `plan-understanding.md` §2.2).
    sentence: str = ""
    # Whether this item may move the level (only people's can be held back).
    counted: bool = True
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state": self.state,
            "reading": self.reading,
            "count": self.count.as_dict() if self.count else None,
            "date": self.date,
            "sentence": self.sentence,
            "counted": self.counted,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Trust:
    """A score's level and its evidence. ``level`` is ``None`` when the score itself is
    not current: no number, so no level (§2.1)."""

    hash: str
    score: str  # CURRENT | STALE | ABSENT
    level: str | None
    items: tuple[Item, ...]
    rule: str

    @property
    def sentence(self) -> str:
        """The lead sentence's key (§2.2), ``""`` with no level."""
        return {EARLY: "level.early", CHECKED: "level.checked",
                QUESTIONED: "level.questioned"}.get(self.level or "", "")

    def item(self, kind: str) -> Item:
        return next(item for item in self.items if item.kind == kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "score": self.score,
            "level": self.level,
            "sentence": self.sentence,
            "items": [item.as_dict() for item in self.items],
            "rule": self.rule,
        }


def level_of(items: Iterable[Item]) -> str:
    """§2.3's three clauses, in order, over the counted items."""
    counted = [item for item in items if item.counted]
    if any(item.reading == FINDS_MORE for item in counted):
        return QUESTIONED
    if any(item.reading == AGREES and item.state == CURRENT for item in counted):
        return CHECKED
    return EARLY


def measurement_digest(basis: Mapping[str, Any]) -> str:
    """Twelve hex naming one measurement by its stamped basis: what `CONVERGENCE_FILE`'s
    ``to`` holds."""
    canonical = json.dumps(dict(basis), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ reading files


def _date(path: Path) -> str | None:
    """When a record was written: its file's date, UTC. Reported, never read by the rule."""
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).date().isoformat()


def _json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _games(score: Any, games: Any) -> Count | None:
    try:
        played = int(games)
        return Count(round(float(score) * played), played, "games") if played > 0 else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ the evidence


def _ladder(stored: dict[str, Any] | None, state: str, date: str | None) -> Item:
    """E-L: finds more when the score withholds its ladder; never agrees."""
    if stored is None:
        return Item(LADDER, state, SILENT, sentence="ladder.absent")
    runs = stored.get("runs")
    if not isinstance(runs, list) or not runs or not all(isinstance(r, dict) for r in runs):
        return Item(LADDER, UNREADABLE, SILENT, date=date, sentence="ladder.unreadable")
    # The score's withheld components are the runs' intersection (`seeds._unreadable`).
    withheld: set[str] | None = None
    for run in runs:
        named = {str(name) for name in run.get("unreadable") or ()}
        withheld = named if withheld is None else withheld & named
    blind = sum(1 for run in runs if run.get("unreadable"))
    finds = bool(withheld)
    return Item(
        LADDER, state, FINDS_MORE if finds else SILENT,
        count=Count(blind, len(runs), "runs"), date=date,
        sentence="ladder.withheld" if finds else "ladder.ordered",
        detail={"withheld": sorted(withheld or ())},
    )


def _premise_items(
    digest: str, runs_dir: Path, spec: dict[str, Any] | None, rule: Rule
) -> tuple[Item, Item]:
    """E-C and E-F, both from `premise.json`, judged by `premise.read`."""
    from chess_variant_sandbox import premise

    path = premise.premise_path(digest, runs_dir)
    if not path.exists():
        return (Item(SELF_DUEL, ABSENT, REPORTED, sentence="self-duel.absent"),
                Item(FAIRY, ABSENT, SILENT, sentence="fairy.absent"))
    found = premise.read(digest, runs_dir, spec)
    date = _date(path)
    if found is None or found.basis is None:
        return (Item(SELF_DUEL, UNREADABLE, REPORTED, date=date, sentence="self-duel.unreadable"),
                Item(FAIRY, UNREADABLE, SILENT, date=date, sentence="fairy.unreadable"))
    state = CURRENT if found.current else STALE
    reasons = {"reasons": list(found.reasons)} if found.reasons else {}
    ceiling = found.stored.get("ceiling") or {}
    fairy = found.stored.get("fairy") or {}
    ceiling_score = ceiling.get("score") if isinstance(ceiling, dict) else None
    self_duel = Item(
        SELF_DUEL, state, REPORTED,
        count=_games(ceiling_score, ceiling.get("games")) if isinstance(ceiling, dict) else None,
        date=date, sentence="self-duel.reported",
        detail={"score": ceiling_score, **reasons},
    )
    if not isinstance(fairy, dict) or not fairy.get("played"):
        why = fairy.get("why", "") if isinstance(fairy, dict) else ""
        return self_duel, Item(FAIRY, state, SILENT, date=date, sentence="fairy.not-played",
                               detail={"why": why, **reasons})
    try:
        score = float(fairy["score"])
    except (KeyError, TypeError, ValueError):
        return self_duel, Item(FAIRY, UNREADABLE, SILENT, date=date, sentence="fairy.unreadable")
    finds = score < rule.fairy_floor
    return self_duel, Item(
        FAIRY, state, FINDS_MORE if finds else AGREES,
        count=_games(score, fairy.get("games")), date=date,
        sentence="fairy.beats-ours" if finds else "fairy.held-its-own",
        detail={"score": score, **reasons},
    )


def _net(digest: str, played: str, nets_dir: Path) -> Item:
    """E-N: an accepted net the score's basis does not play finds more; nothing agrees."""
    from chess_variant_sandbox.agents.nets import ACCEPTED, INPUTS_VERSION, REFUSED

    paths = sorted(Path(nets_dir).glob(f"{digest}-*.json"))
    if not paths:
        return Item(NET, ABSENT, SILENT, sentence="net.absent")
    playing = {part.strip() for part in played.split(",") if part.strip()}
    findings: list[Item] = []
    refused: list[Item] = []
    unreadable = 0
    for path in paths:
        store = _json(path) or {}
        procedure = str(store.get("procedure") or "")
        verdict = store.get("verdict")
        weights = str(store.get("weights_sha256") or "")
        if (store.get("ruleset_hash") != digest or not procedure
                or verdict not in (ACCEPTED, REFUSED) or (verdict == ACCEPTED and not weights)):
            unreadable += 1
            continue
        # A store trained on another input layout is no player this checkout can sit
        # down: its verdict was made on older players, and stands as a stale reading.
        state = CURRENT if store.get("inputs_version") == INPUTS_VERSION else STALE
        acceptance = store.get("acceptance")
        acceptance = acceptance if isinstance(acceptance, dict) else {}
        count = _games(acceptance.get("score"), acceptance.get("games"))
        detail = {"procedure": procedure, "score": acceptance.get("score"),
                  "error": acceptance.get("error")}
        if verdict == REFUSED:
            refused.append(Item(NET, state, SILENT, count=count, date=_date(path),
                                sentence="net.refused", detail=detail))
            continue
        if f"{procedure}=net:{weights[:12]}" in playing:
            # Measured with it: E-V carries the reading now (§2.3).
            refused.append(Item(NET, state, SILENT, count=count, date=_date(path),
                                sentence="net.in-the-score", detail=detail))
            continue
        findings.append(Item(NET, state, FINDS_MORE, count=count, date=_date(path),
                             sentence="net.beats-ours", detail=detail))
    # One item per kind: a finding first, a current one before a stale one.
    if findings:
        return sorted(findings, key=lambda item: item.state != CURRENT)[0]
    if refused:
        return refused[0]
    return Item(NET, UNREADABLE, SILENT, sentence="net.unreadable",
                detail={"unreadable": unreadable})


def _convergence(
    digest: str, runs_dir: Path, basis: Mapping[str, Any] | None, rule: Rule
) -> Item:
    """E-V: moved by `convergence_sds` sd or more finds more; less agrees."""
    from chess_variant_sandbox.corpus import measurement_dir

    path = measurement_dir(digest, runs_dir) / CONVERGENCE_FILE
    if not path.exists():
        return Item(CONVERGENCE, ABSENT, SILENT, sentence="convergence.absent")
    stored = _json(path)
    date = _date(path)
    try:
        move = abs(float((stored or {})["move"]))
        sd = float((stored or {})["sd"])
        into = str((stored or {})["to"])
    except (KeyError, TypeError, ValueError):
        move, sd, into = 0.0, -1.0, ""
    # A negative or NaN spread, or a missing field, is no reading at all.
    if not sd >= 0.0 or move != move:
        return Item(CONVERGENCE, UNREADABLE, SILENT, date=date, sentence="convergence.unreadable")
    state = CURRENT if basis is not None and into == measurement_digest(basis) else STALE
    finds = move >= rule.convergence_sds * sd
    return Item(
        CONVERGENCE, state, FINDS_MORE if finds else AGREES, date=date,
        sentence="convergence.moved" if finds else "convergence.held",
        detail={"move": move, "sd": sd},
    )


def _people(people: People | None, rule: Rule) -> Item:
    """E-H: two people clearing the top finds more; twenty games by five, no clear, agrees."""
    counted = rule.people_move_level
    if people is None:
        return Item(PEOPLE, ABSENT, SILENT, sentence="people.absent", counted=counted)
    count = Count(people.cleared, people.people, "people")
    if people.cleared >= rule.people_clears:
        reading, sentence = FINDS_MORE, "people.cleared"
    elif (people.cleared == 0 and people.games >= rule.people_games
          and people.people >= rule.people_people):
        reading, sentence = AGREES, "people.unbeaten"
    else:
        reading, sentence = SILENT, "people.counts"
    return Item(PEOPLE, CURRENT, reading, count=count, date=people.latest,
                sentence=sentence, counted=counted,
                detail={"games": people.games, "people": people.people,
                        "cleared": people.cleared})


# ------------------------------------------------------------------ the reading


def read(
    digest: str,
    runs_dir: Path,
    spec: dict[str, Any] | None,
    people: People | None = None,
    *,
    nets_dir: str | Path | None = None,
    rule: Rule = RULE,
) -> Trust:
    """How far to trust this game's library score: one item per kind, and the level.

    The score is read as `corpus.read_measurement` reads it; a score that is absent or
    out of date carries no level, though its evidence is still listed for `/admin`.
    """
    from chess_variant_sandbox.agents.nets import STORE_DIR
    from chess_variant_sandbox.corpus import read_measurement

    runs_dir = Path(runs_dir)
    measured = None
    with contextlib.suppress(OSError, ValueError):
        measured = read_measurement(digest, runs_dir, spec=spec)
    if measured is None:
        score, stored, basis, date = ABSENT, None, None, None
    else:
        score = CURRENT if measured.current else STALE
        stored = measured.stored
        basis = measured.basis.as_dict() if measured.basis is not None else None
        date = _date(measured.directory / "canonical.json")
    self_duel, fairy = _premise_items(digest, runs_dir, spec, rule)
    items = (
        _ladder(stored, score, date),
        self_duel,
        fairy,
        _net(digest, str((basis or {}).get("net") or ""),
             Path(STORE_DIR if nets_dir is None else nets_dir)),
        _convergence(digest, runs_dir, basis, rule),
        _people(people, rule),
    )
    return Trust(
        hash=digest,
        score=score,
        level=level_of(items) if score == CURRENT else None,
        items=items,
        rule=rule.spelled(),
    )


# ------------------------------------------------------------------ the words (Q647, T2)

# The lead sentence of each level (§2.2's table). *Checked* and *questioned* open with the
# sentence of the evidence that made them, then say this.
EARLY_SAID = (
    "Measured by the site's own computer players only. Nothing else has checked them on "
    "this game yet. If you beat the strongest computer player here, that tells us something."
)
CHECKED_TAIL = ("It is still an estimate: it is measured again as stronger players arrive and "
                "as people play.")
QUESTIONED_TAIL = ("Read the score as a first guess; it will be measured again when a stronger "
                   "player is ready.")

# What a kind of evidence says, by its sentence key, in a visitor's words
# (`plan-understanding.md` §2.2, with §2.2 of the plan's two changes). `{n}`, `{of}` are the
# item's one plain count; a sentence whose count is missing uses its `bare` form. A key
# missing here says nothing: an absent record is not a piece of evidence.
_SAID: dict[str, str] = {
    "ladder.ordered": ("Its computer players get steadily stronger with more thinking time, "
                       "which is what the score needs."),
    "ladder.withheld": ("Its computer players don't get clearly stronger with more thinking "
                        "time here, so parts of the score describe the players more than "
                        "the game."),
    "self-duel.reported": ("A copy of the strongest computer player, given much longer to "
                           "think, played it. That shows how much more is in the game, not "
                           "how far to trust the players, so it is not counted."),
    "fairy.held-its-own": ("An engine written elsewhere (Fairy-Stockfish) played ours on this "
                           "game, and ours held its own."),
    "fairy.beats-ours": ("An engine written elsewhere (Fairy-Stockfish) beats ours on this "
                         "game, so ours are missing something."),
    "fairy.not-played": ("An engine written elsewhere (Fairy-Stockfish) has not been tried "
                         "on this game: it cannot play its rules."),
    "net.beats-ours": ("A computer player that learned this game from its own play beats the "
                       "measuring players: the score will be measured again with it."),
    "net.refused": ("A computer player that learned this game from its own play was tried "
                    "and did not clearly beat the measuring players."),
    "net.in-the-score": ("A computer player that learned this game from its own play is one "
                         "of the players that measured this score."),
    "convergence.held": ("Stronger computer players measured this game again and the score "
                         "stayed where it was."),
    "convergence.moved": ("Stronger computer players measured this game again and the score "
                          "moved, so it may move again."),
    "people.cleared": ("{n} people have beaten the strongest computer player here. If that is "
                       "easy, the score may be reading too little into this game."),
    "people.unbeaten": ("People have played the strongest computer player here {games} times "
                        "and nobody has beaten it."),
}

# The count a sentence carries under *more*, said after it.
_COUNTED: dict[str, str] = {
    "self-duel.reported": "The strongest player scored {n} of {of} games against the copy.",
    "fairy.held-its-own": "Ours scored {n} of {of} games against it.",
    "fairy.beats-ours": "Ours scored {n} of {of} games against it.",
    "net.beats-ours": "It scored {n} of {of} games against them.",
    "net.refused": "It scored {n} of {of} games against them.",
}

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")

# People's two counted readings as the line says them, with no count (§5 Q2); *more*
# carries the count.
PEOPLE_CLEARED_BARE = ("More than one person has beaten the strongest computer player here. "
                       "If that is easy, the score may be reading too little into this game.")
PEOPLE_UNBEATEN_BARE = ("Many people have played the strongest computer player here and "
                        "nobody has beaten it.")

FAIRY_NOT_TRIED = ("An engine written elsewhere (Fairy-Stockfish) has not been tried on this "
                   "game yet.")
NOBODY_YET = "Nobody has played the strongest computer player here yet."
EARLIER = "This was found with earlier computer players."


def month(day: str | None) -> str:
    """*October 2026* from an ISO date, ``""`` from anything else."""
    try:
        when = datetime.date.fromisoformat(str(day)[:10])
    except ValueError:
        return ""
    return f"{_MONTHS[when.month - 1]} {when.year}"


def _people_said(item: Item) -> str:
    """People's line, which is said even when it is silent: the counts are the evidence."""
    games = int(item.detail.get("games") or 0)
    seen = int(item.detail.get("people") or 0)
    cleared = int(item.detail.get("cleared") or 0)
    if item.sentence == "people.cleared":
        return _SAID["people.cleared"].format(n=cleared)
    if item.sentence == "people.unbeaten":
        return _SAID["people.unbeaten"].format(games=games)
    if not games:
        return NOBODY_YET
    who = "1 person has" if seen == 1 else f"{seen} people have"
    if not cleared:
        return f"{who} played the strongest computer player here; none has beaten it yet."
    return f"{who} played the strongest computer player here, and one has beaten it."


def said(item: Item, *, counted: bool = False, fairy_translates: bool | None = None) -> str:
    """One piece of evidence in a visitor's words, ``""`` where it has nothing to say.

    ``counted`` adds its one plain count and its month, as the *more* list shows it; the
    lead sentence is said without either (the level is a word only, §5 Q2).
    ``fairy_translates`` lets an absent Fairy-Stockfish reading say whether the engine
    could play the rules at all (`library._fairy`, the cheap half).
    """
    if item.kind == PEOPLE:
        if item.state == ABSENT:
            return ""
        bare = {"people.cleared": PEOPLE_CLEARED_BARE,
                "people.unbeaten": PEOPLE_UNBEATEN_BARE}.get(item.sentence)
        text = bare if bare and not counted else _people_said(item)
    elif item.kind == FAIRY and item.state == ABSENT:
        if fairy_translates is None:
            return ""
        text = FAIRY_NOT_TRIED if fairy_translates else _SAID["fairy.not-played"]
    else:
        text = _SAID.get(item.sentence, "")
    if not text:
        return ""
    if item.state == STALE:
        text += " " + EARLIER
    if counted:
        if item.count is not None and item.sentence in _COUNTED:
            text += " " + _COUNTED[item.sentence].format(n=item.count.n, of=item.count.of)
        when = month(item.date)
        if when:
            text = f"{text[:-1]} ({when}){text[-1]}" if text.endswith(".") else text
    return text


def lead(found: Trust) -> str:
    """The level's sentence: §2.2's, opened by the evidence that made the level."""
    if found.level is None:
        return ""
    if found.level == EARLY:
        return EARLY_SAID
    counted = [item for item in found.items if item.counted]
    if found.level == QUESTIONED:
        made = sorted((item for item in counted if item.reading == FINDS_MORE),
                      key=lambda item: item.state != CURRENT)
        tail = QUESTIONED_TAIL
    else:
        made = [item for item in counted if item.reading == AGREES and item.state == CURRENT]
        tail = CHECKED_TAIL
    opening = next((text for text in (said(item) for item in made) if text), "")
    return f"{opening} {tail}" if opening else tail


def with_people(found: Trust, people: People | None, rule: Rule = RULE) -> Trust:
    """The same reading with people's counts in place of whatever it had: what the site
    does to a reading the library made from the files, with no file read again."""
    items = tuple(_people(people, rule) if item.kind == PEOPLE else item
                  for item in found.items)
    return Trust(hash=found.hash, score=found.score,
                 level=level_of(items) if found.score == CURRENT else None,
                 items=items, rule=rule.spelled())


def card(found: Trust | None) -> dict[str, Any] | None:
    """A library card's chip: the level word, and its sentence for the (i). ``None``
    where the score is absent or out of date, which says so itself."""
    if found is None or found.level is None:
        return None
    return {"level": found.level, "said": lead(found)}


def _numbers(found: Trust, premise_words: str = "") -> list[str]:
    """The maintainer's lines: each reading with its numbers, then the rule."""
    lines = []
    for item in found.items:
        bits = [f"{item.kind}: {item.reading}, {item.state}"]
        if item.count is not None:
            bits.append(f"{item.count.n} of {item.count.of} {item.count.unit}")
        for key in ("score", "error", "move", "sd", "procedure", "withheld", "why", "games",
                    "people", "cleared", "reasons", "unreadable"):
            value = item.detail.get(key)
            if value not in (None, "", [], ()):
                bits.append(f"{key} {value}")
        if item.date:
            bits.append(item.date)
        if not item.counted:
            bits.append("not counted")
        lines.append("; ".join(str(bit) for bit in bits))
    if premise_words:
        lines.append(f"strength check: {premise_words}")
    lines.append(f"rule: {found.rule}")
    return lines


def shown(
    found: Trust | None,
    *,
    was: Iterable[Mapping[str, Any]] = (),
    fairy_translates: bool | None = None,
    premise_words: str = "",
    maintainer: bool = True,
) -> dict[str, Any] | None:
    """A game page's trust line: ``level``, ``said`` (one sentence) and ``more`` (one line
    per piece of evidence there is, current or out of date, then each earlier score,
    dated and labelled with its players: §5 Q1, answered A). ``numbers`` is the
    maintainer's, and absent when ``maintainer`` is false. ``None`` with no level."""
    if found is None or found.level is None:
        return None
    more = [text for text in (said(item, counted=True, fairy_translates=fairy_translates)
                              for item in found.items) if text]
    for line in was:
        index = line.get("index")
        if not isinstance(index, (int, float)):
            continue
        # A score never shows without what it rests on (`runsShort` on the page).
        runs = line.get("runs")
        rests = f" ({runs} run{'' if runs == 1 else 's'})" if isinstance(runs, int) and runs else ""
        when = month(line.get("measured"))
        more.append(f"Was {float(index):.3f}{rests} when measured by the earlier computer "
                    "players" + (f" ({when})." if when else "."))
    payload: dict[str, Any] = {"level": found.level, "said": lead(found), "more": more}
    if maintainer:
        payload["numbers"] = _numbers(found, premise_words)
    return payload
