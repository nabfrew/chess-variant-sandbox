"""The strength-premise check (Q502d): are a game's players strong enough for its score?

A library score reads differences between the roster's rungs, and every rung shares one
evaluator. Where that evaluator is blind to how the game is won, the score is "premised
on very poor play as if it is advanced level play" (the user, 2026-09-28), and
`docs/reference/general-player.md` §6.4 measured that on four of six games the score moves
when every player gets stronger. So each library game carries a second, separate reading
beside its score: whether the players who produced the score are good at the game.

**What is played.** Two duels, both through the core's ordinary tournament runner, which
fixes every seed before a thread starts (`tournament.duel`):

- *The ceiling duel* (§3.1.1): the roster's top rung against the same player given `Rule
  .multiple` times its nodes. How much the top rung holds is how much play is left above
  the players that scored the game. Both sides share the evaluator, so on its own it is
  circular; it is still the one yardstick every game has.
- *Fairy-Stockfish* (§3.1.3), where the game translates, the installed build plays it
  as ours does (`fairy.checked`), and a binary is on the machine: the top rung against
  `fairystockfish:nodes=N` at its own node budget, an engine nobody here wrote.

**The verdict.** `passes` when the top rung scores at least `Rule.ceiling_floor` in the
ceiling duel and, where Fairy-Stockfish played, at least `Rule.fairy_floor` against it
(§3.2 (b) and (c), the record's proposed thresholds). Otherwise `fails`.

**What it decides: nothing.** The user's decision A of 2026-09-28 held a game whose verdict
failed out of the band, the set of real games that sets the scale; the user withdrew it on
2026-10-01 (`docs/reference/general-player.md` §4.2), and Q645 (`docs/plan-confidence.md`
§2.4, T0) removed the hold: every real game with a current score is in the band whatever
its verdict. The check stays as one piece of evidence for the trust line that replaces the
mark (`plan-confidence.md` T1, T2); until then a failing verdict marks its score on the
page, and nothing reads it as a gate.
**It changes no score.** Nothing under `metrics/` reads it, this file is in no fingerprint
set, and no score's `Basis` names it.

**Its own basis.** `runs/<hash>/premise.json`, beside `canonical.json`, stamped with a
`PremiseBasis`: who played (the top rung's spelling, the core, the tables, the fit store's
player, the nets), the rule by value, this check's own code, and Fairy-Stockfish's
identity. A mismatch is *stale*, with a reason, never a verdict. A failing verdict that has
gone stale keeps its mark (`Premise.marked`): a stale fail is not a pass.

A stronger player (a `net=` roster, Q502h) moves the roster, which moves the top rung,
which stales every verdict; the re-check is then played by the stronger player.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from chess_variant_sandbox.measurement import Basis, code_fingerprint, tables_fingerprint

PREMISE_FILE = "premise.json"

PASSES = "passes"
FAILS = "fails"

# The check's own source, stamped by its code reading. `tournament/duel.py` grades the
# duels, so a change there is a different reading of the same games.
PREMISE_SOURCES = (
    "chess_variant_sandbox/premise.py",
    "chess_variant_sandbox/tournament/duel.py",
)
# The translator, stamped only where Fairy-Stockfish is part of the verdict: a change to
# `fairy/` stales the verdicts it can reach and no other.
FAIRY_SOURCES = ("chess_variant_sandbox/fairy/*.py",)

# `PremiseBasis.fairy_engine` where the game translates and the machine that ran the check
# had no Fairy-Stockfish. Read now, it means *not told*: a reader with no binary cannot say
# what one would play, so it does not stale a verdict that one played.
FAIRY_ABSENT = "absent"


@dataclass(frozen=True)
class Rule:
    """How the check is played and graded. Stamped by value (`spelled`).

    The floors are §3.2's proposals ((b) and (c)); they are the user's to confirm, and
    changing one stales every verdict and no score. Sixteen seed-pairs is E8's size
    (§6.6), at a seed-pair error of about 0.05.

    ``curtailed`` (plan-compute §4.2, Q632): each duel stops at the first schedule-order
    prefix of seed-pairs after which no result of the rest could change its verdict. The
    verdict is the sixteen pairs' by construction; the stored score is the prefix's, with
    ``decided_after`` of ``planned_pairs`` beside it.
    """

    multiple: int = 8
    pairs: int = 16
    seed: int = 0
    ceiling_floor: float = 0.20
    fairy_pairs: int = 16
    fairy_floor: float = 0.30
    curtailed: bool = True

    def spelled(self) -> str:
        return (
            f"ceiling x{self.multiple}, {self.pairs} pairs, seed {self.seed}, "
            f"floor {self.ceiling_floor:g}; fairy-stockfish at equal nodes, "
            f"{self.fairy_pairs} pairs, floor {self.fairy_floor:g}"
            + ("; curtailed" if self.curtailed else "")
        )


RULE = Rule()


# ------------------------------------------------------------------ the players

_NODES = re.compile(r"(?<=nodes=)\d+")


def _nodes(agent: str) -> int:
    found = _NODES.search(agent)
    return int(found.group(0)) if found else 0


def top_rung(agents: tuple[str, ...]) -> str:
    """The roster's strongest searching player: the one with the most nodes."""
    searching = [agent for agent in agents if _nodes(agent) > 0]
    if not searching:
        raise ValueError(f"no searching player in the roster {list(agents)}")
    return max(searching, key=_nodes)


def ceiling_of(agent: str, multiple: int) -> str:
    """The same player spelled with ``multiple`` times its nodes."""
    return _NODES.sub(str(_nodes(agent) * multiple), agent, count=1)


def fairy_of(agent: str) -> str:
    """Fairy-Stockfish at the same node budget as ``agent``."""
    return f"fairystockfish:nodes={_nodes(agent)}"


# ------------------------------------------------------------------ Fairy-Stockfish


def _fairy_translates(spec: dict[str, Any]) -> bool:
    from chess_variant_sandbox import fairy

    return fairy.supports(spec)


@lru_cache(maxsize=4)
def _binary_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def fairy_engine() -> str:
    """The installed Fairy-Stockfish as a stamp, `<name>@<12 hex of the binary>`, or
    `FAIRY_ABSENT`. A node-limited search is reproducible on one build and not across two,
    so the binary itself is the identity (`agents/fairy_stockfish.py`)."""
    from chess_variant_sandbox import fairy

    if not fairy.available():
        return FAIRY_ABSENT
    path = fairy.binary_path()
    if path is None:
        return FAIRY_ABSENT
    try:
        return f"{fairy.engine_name() or 'Fairy-Stockfish'}@{_binary_digest(path)}"
    except OSError:
        return FAIRY_ABSENT


def _fairy_plays(spec: dict[str, Any]) -> str:
    """``""`` where the installed build plays these rules as ours does, else why not."""
    from chess_variant_sandbox import fairy

    if fairy.checked(spec) is not None:
        return ""
    found = fairy.verdict(spec)
    said = "; ".join(found.reasons[:1]) if found is not None and found.reasons else ""
    return "the installed Fairy-Stockfish does not play these rules as ours does" + (
        f" ({said})" if said else ""
    )


# ------------------------------------------------------------------ the basis


@dataclass(frozen=True)
class PremiseBasis:
    """Everything a premise verdict depends on. Compared field by field.

    The player fields (`engine`, `tables`, `fit`, `net`) are taken from `Basis.current`,
    the score's own, so the verdict is about the players the score was played by and
    stales with them. `fairy_code` is `""` where the game has no Fairy-Stockfish
    counterpart; `fairy_engine` is `""` there too, `FAIRY_ABSENT` where it translates and
    no binary was on the machine, and the binary's stamp where one was.
    """

    ruleset_hash: str
    player: str
    engine: str
    tables: str
    fit: str
    net: str
    rule: str
    code: str
    fairy_code: str = ""
    fairy_engine: str = ""

    @classmethod
    def current(
        cls,
        digest: str,
        spec: dict[str, Any] | None,
        agents: tuple[str, ...] | None = None,
        rule: Rule = RULE,
    ) -> PremiseBasis:
        """The basis a check taken now would carry. ``spec`` ``None`` cannot compare
        tables and reads stale, as `corpus.current_basis` does."""
        from chess_variant_sandbox import corpus

        roster = tuple(corpus.CANONICAL_AGENTS if agents is None else agents)
        score = Basis.current(
            ruleset_hash=digest,
            weights_version="",
            agents=roster,
            games_per_pairing=0,
            seeds=(),
            tables=tables_fingerprint(spec) if spec is not None else "",
            fit=corpus.fit_reading(digest, spec=spec),
        )
        translates = spec is not None and _fairy_translates(spec)
        return cls(
            ruleset_hash=digest,
            player=top_rung(roster),
            # The agents' source beside the core, the rule `engine` held until Q631: a
            # premise check is not read through `Basis.players`.
            engine=f"{score.engine}+{score.agents_source}",
            tables=score.tables,
            fit=score.fit,
            net=score.net,
            rule=rule.spelled(),
            code=code_fingerprint(*PREMISE_SOURCES),
            fairy_code=code_fingerprint(*FAIRY_SOURCES) if translates else "",
            fairy_engine=fairy_engine() if translates else "",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ruleset_hash": self.ruleset_hash,
            "player": self.player,
            "engine": self.engine,
            "tables": self.tables,
            "fit": self.fit,
            "net": self.net,
            "rule": self.rule,
            "code": self.code,
            "fairy_code": self.fairy_code,
            "fairy_engine": self.fairy_engine,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PremiseBasis | None:
        """``None`` for anything this module did not write: an unreadable stamp is stale."""
        if not isinstance(data, dict):
            return None
        try:
            return cls(**{key: str(data[key]) for key in (
                "ruleset_hash", "player", "engine", "tables", "fit", "net", "rule",
                "code", "fairy_code", "fairy_engine",
            )})
        except (KeyError, TypeError):
            return None

    def stale_against(self, current: PremiseBasis) -> tuple[str, ...]:
        """Why this verdict is not the one a check taken now would give; empty if it is."""
        reasons: list[str] = []
        if self.ruleset_hash != current.ruleset_hash:
            reasons.append(f"it checked a different game ({self.ruleset_hash})")
        if self.player != current.player:
            reasons.append(
                f"its strongest player was {self.player}, and {current.player} is now"
            )
        if self.engine != current.engine:
            reasons.append("the core, the agents or the game loop changed since it was checked")
        if self.tables != current.tables:
            reasons.append("this game compiles to different tables than it was checked on")
        if self.fit != current.fit:
            reasons.append(
                f"its players' fit was {self.fit or 'none'}, and is {current.fit or 'none'} now"
            )
        if self.net != current.net:
            reasons.append(
                f"its nets were {self.net or 'none'}, and are {current.net or 'none'} now"
            )
        if self.rule != current.rule:
            reasons.append(f"it was graded under another rule ({self.rule})")
        if self.code != current.code:
            reasons.append("the check itself changed since it ran")
        if self.fairy_code != current.fairy_code:
            reasons.append(
                "Fairy-Stockfish's translation of this game changed since it was checked"
                if self.fairy_code and current.fairy_code
                else "whether Fairy-Stockfish has a counterpart for this game changed"
            )
        # A reader with no binary cannot say what one would play: not told, not stale.
        if self.fairy_engine != current.fairy_engine and current.fairy_engine != FAIRY_ABSENT:
            reasons.append(
                "it was checked with no Fairy-Stockfish, and one is installed now"
                if self.fairy_engine == FAIRY_ABSENT
                else f"it was checked against {self.fairy_engine or 'no Fairy-Stockfish'}, "
                f"and {current.fairy_engine or 'none'} is here now"
            )
        return tuple(reasons)


# ------------------------------------------------------------------ the check


def check(
    spec: dict[str, Any],
    agents: tuple[str, ...] | None = None,
    rule: Rule = RULE,
    workers: int = 0,
) -> dict[str, Any]:
    """Play the premise check on one game and return what `write` stores.

    Python orchestrates: each duel is one call into the tournament runner, which plays the
    whole schedule in the core (or, for Fairy-Stockfish, over the core's moves).
    """
    from chess_variant_sandbox.spec.canonical import spec_hash

    digest = spec_hash(spec)
    basis = PremiseBasis.current(digest, spec, agents, rule)
    top = basis.player
    ceiling = ceiling_of(top, rule.multiple)
    ceiling_part = _duel_part(
        spec, top, ceiling, rule.pairs, rule.ceiling_floor, rule, workers
    )

    fairy_part: dict[str, Any]
    if not basis.fairy_code:
        fairy_part = {"played": False, "why": "Fairy-Stockfish has no counterpart for these rules"}
    elif basis.fairy_engine == FAIRY_ABSENT:
        fairy_part = {
            "played": False,
            "why": "no Fairy-Stockfish was installed on the machine that ran the check",
        }
    else:
        refused = _fairy_plays(spec)
        if refused:
            fairy_part = {"played": False, "why": refused}
        else:
            fairy_part = {
                "played": True,
                **_duel_part(
                    spec, top, fairy_of(top), rule.fairy_pairs, rule.fairy_floor, rule, workers
                ),
            }

    passes = ceiling_part["passes"] and (not fairy_part["played"] or fairy_part["passes"])
    return {
        "hash": digest,
        "name": str(spec.get("name", digest)),
        "verdict": PASSES if passes else FAILS,
        "ceiling": ceiling_part,
        "fairy": fairy_part,
        "basis": basis.as_dict(),
    }


def _duel_part(
    spec: dict[str, Any],
    top: str,
    against: str,
    pairs: int,
    floor: float,
    rule: Rule,
    workers: int,
) -> dict[str, Any]:
    """One duel of the check as stored: its grade, its pairs, the floor and whether it held.

    Curtailed under ``rule.curtailed``: ``passes`` is then the full duel's verdict, and the
    score and pairs are the deciding prefix's (a decided pass reads at or above the floor
    and a decided fail below it, so the words never contradict the verdict).
    """
    from chess_variant_sandbox.tournament.duel import curtailed_duel, duel, floor_verdict

    if rule.curtailed:
        played = curtailed_duel(
            spec, top, against, pairs, floor_verdict(floor), seed=rule.seed, workers=workers
        )
        return {
            **played.as_dict(),
            "pairs": list(played.found.pairs),
            "floor": floor,
            "passes": played.verdict,
        }
    held = duel(spec, top, against, pairs, seed=rule.seed, workers=workers)
    return {
        **held.as_dict(),
        "pairs": list(held.pairs),
        "floor": floor,
        "passes": held.score >= floor,
    }


def premise_path(digest: str, runs_dir: Path) -> Path:
    from chess_variant_sandbox.corpus import measurement_dir

    return measurement_dir(digest, runs_dir) / PREMISE_FILE


def write(payload: dict[str, Any], runs_dir: Path) -> Path:
    """Store a check beside the game's score, whole or not at all."""
    path = premise_path(str(payload["hash"]), runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


# ------------------------------------------------------------------ reading it


@dataclass(frozen=True)
class Premise:
    """One stored check, judged against the basis that holds now."""

    hash: str
    stored: dict[str, Any]
    basis: PremiseBasis | None
    reasons: tuple[str, ...]

    @property
    def current(self) -> bool:
        return self.basis is not None and not self.reasons

    @property
    def recorded(self) -> str:
        """What the file says, current or not: `PASSES`, `FAILS`, or ``""``."""
        said = self.stored.get("verdict")
        return said if said in (PASSES, FAILS) else ""

    @property
    def verdict(self) -> str | None:
        """The verdict, and ``None`` while it is stale: a mismatch is never a value."""
        return self.recorded or None if self.current else None

    @property
    def marked(self) -> bool:
        """Whether the page marks this game's score: its last verdict failed, current or
        stale, since only a passing check lifts the mark. It holds nothing out of the band
        (withdrawn 2026-10-01, Q645)."""
        return self.recorded == FAILS

    @property
    def state(self) -> str:
        return (self.recorded or "stale") if self.current else "stale"

    def words(self) -> str:
        """What the check says, in plain words, for a page or `/admin`."""
        if not self.current:
            why = "; ".join(self.reasons) if self.reasons else "it records nothing to match"
            if self.recorded == FAILS:
                return (
                    "Failed the strength check when it was last run, and that check is out "
                    f"of date ({why}). The score still shows, marked, until a check passes."
                )
            return f"The strength check is out of date ({why}), so it says nothing yet."
        ceiling = self.stored.get("ceiling") or {}
        fairy = self.stored.get("fairy") or {}
        held = (
            f"its strongest player scored {float(ceiling.get('score', 0.0)):.2f} against "
            f"itself given {_multiple(self.stored)} times the search "
            f"({float(ceiling.get('floor', 0.0)):.2f} is enough)"
        )
        if fairy.get("played"):
            held += (
                f", and {float(fairy.get('score', 0.0)):.2f} against Fairy-Stockfish, an "
                "engine written elsewhere, at the same search "
                f"({float(fairy.get('floor', 0.0)):.2f} is enough)"
            )
        if self.recorded == PASSES:
            return f"Checked: the players who scored this game are strong enough for it: {held}."
        return (
            f"Scored by players not yet strong enough for this game: {held}. There is more "
            "to it than they could see. The score still shows, marked, until a stronger "
            "player passes this check."
        )

    def site_words(self) -> str:
        """The same check in a visitor's words: *How far to trust it*, a level word and why.

        `docs/plan-understanding.md` §2.2 (T1): no floor, no number, no *scale*, and the
        evidence said as §2.2's table says it, until Q544's line replaces this. The level
        words are §2.2's: *questioned* when a longer search or an outside engine beats the
        players that scored the game, *checked* when an outside engine agrees, *early
        estimate* when nothing outside the site's own players has, and *not known* when the
        check was made on older players. `words` keeps the numbers for `/admin` and the
        local application.
        """
        if not self.current:
            return ("How far to trust it: not known. The last check was made on older "
                    "computer players.")
        ceiling = self.stored.get("ceiling") or {}
        fairy = self.stored.get("fairy") or {}
        longer = (f"A copy of the strongest computer player, given {_times(self.stored)} "
                  "longer to think,")
        if self.recorded == FAILS:
            found = []
            if float(ceiling.get("score", 0.0)) < float(ceiling.get("floor", 0.0)):
                found.append(f"{longer} beats it easily, so there is more in this game than "
                             "the players that scored it can see.")
            if fairy.get("played") and float(fairy.get("score", 0.0)) < float(
                    fairy.get("floor", 0.0)):
                found.append("An engine written elsewhere (Fairy-Stockfish) beats ours on "
                             "this game, so ours are missing something.")
            if not found:
                found.append("The players that scored it can see less of this game than it "
                             "holds.")
            return ("How far to trust it: questioned. " + " ".join(found) + " Read the score "
                    "as a first guess; it will be measured again when a stronger player is "
                    "ready.")
        if fairy.get("played"):
            return ("How far to trust it: checked. An engine written elsewhere "
                    "(Fairy-Stockfish) plays this game about as well as ours.")
        return (f"How far to trust it: early estimate. {longer} does not beat it easily, but "
                "nothing from outside the site's own computer players has checked them on "
                "this game yet.")

    def as_dict(self) -> dict[str, Any]:
        """What a page renders: `state`, whether the score is marked, and the words.

        The mark travels as ``held``, the name `web/static/page/core.js`'s `premiseHtml`
        reads until the trust line replaces it (`docs/plan-confidence.md` T2).
        """
        return {
            "state": self.state,
            "current": self.current,
            "recorded": self.recorded,
            "held": self.marked,
            "words": self.words(),
            "site": self.site_words(),
            "reasons": list(self.reasons),
        }


_TIMES = {2: "twice", 4: "four times", 8: "eight times", 16: "sixteen times"}


def _times(stored: dict[str, Any]) -> str:
    """The ceiling duel's multiple as a visitor reads it: *eight times*, or *much*."""
    found = re.search(r"ceiling x(\d+)", str((stored.get("basis") or {}).get("rule", "")))
    return _TIMES.get(int(found.group(1)), "much") if found else "much"


def _multiple(stored: dict[str, Any]) -> str:
    found = re.search(r"ceiling x(\d+)", str((stored.get("basis") or {}).get("rule", "")))
    return found.group(1) if found else "several"


def read(
    digest: str,
    runs_dir: Path,
    spec: dict[str, Any] | None = None,
    agents: tuple[str, ...] | None = None,
    rule: Rule = RULE,
) -> Premise | None:
    """The stored check for this game, judged now, or ``None`` if none was stored.

    A hash with no spec is found the way `corpus.read_measurement` finds one; one that
    cannot be found compares no tables and reads stale.
    """
    path = premise_path(digest, runs_dir)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict):
        return None
    if spec is None:
        from chess_variant_sandbox.corpus import CorpusError, find

        with contextlib.suppress(CorpusError, OSError, ValueError):
            spec = find(digest, runs_dir=runs_dir).spec
    basis = PremiseBasis.from_dict(stored.get("basis"))
    if basis is None:
        reasons: tuple[str, ...] = ("it does not record what it was checked under",)
    else:
        reasons = basis.stale_against(PremiseBasis.current(digest, spec, agents, rule))
    return Premise(hash=digest, stored=stored, basis=basis, reasons=reasons)

