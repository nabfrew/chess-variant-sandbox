"""The convergence record: did the score hold when stronger players measured it again? (Q650)

`docs/archive/plan-confidence.md` §3.2 step 4 (T5), the rule of `general-player.md` §3.1.4: **a
game's score converges when a re-measurement by other players moves it by less than twice
its four-run standard deviation.** After `history` keeps a replaced measurement, this
module compares the two and, where they differ *only in their players*, writes
`runs/<hash>/convergence.json`, which `trust.read` reads as E-V.

**Only the players.** The two stamps must agree on the game, its tables, the seeds, the
games a pairing, who played (`played`) and the core (`engine`); they must differ in at least
one of `history.PLAYER_FIELDS`. A pair that also moved anything in the first list says
nothing about the players (a move whose cause is mixed is not evidence), and no verdict is
written. A pair that also moved the scoring (`weights_version`, `metrics`, `analysis`,
`verdict_rule`) is read after the outgoing runs are **re-scored under the current metrics**
from their kept games (`corpus.rescore_kept`), re-analysed first where the pass moved, so
both sides are scored by the same code and only the players differ.

**The record** (`trust.CONVERGENCE_FILE`): ``move`` (new index minus old), ``sd`` (the
sample standard deviation of the outgoing runs' totals, as E4 took it), ``to`` (the
`trust.measurement_digest` of the new measurement's basis, which is what makes it current in
`trust`), ``from``, both indexes, the rule and its verdict, whether and under what the old
runs were re-scored, and its own ``basis`` (both digests and the rule). `trust` re-applies
the rule from ``move`` and ``sd`` and never reads the stored verdict. A record already there
is kept under the replaced measurement's `previous/<n>/` before it is overwritten.

Nothing under `metrics/` imports this file and it is in no fingerprint set
(`tests/test_convergence.py`): a record moves no score.
"""

from __future__ import annotations

import datetime
import json
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.history import PLAYER_FIELDS, previous_root
from chess_variant_sandbox.trust import CONVERGENCE_FILE, RULE, Rule, measurement_digest

# Moved, these leave the games standing and are repaired by a re-score of the old runs.
RESCORE_FIELDS = ("weights_version", "metrics", "analysis", "verdict_rule")
# Moved, these make the pair mixed: no verdict.
SAME_FIELDS = ("ruleset_hash", "games_per_pairing", "seeds", "engine", "tables",
               "played", "checked")

HELD = "held"
MOVED = "moved"
NO_VERDICT = "no verdict"

# What a re-score of the kept runs returns: each run's total, ``None`` where it had none.
Rescore = Callable[[Path, Sequence[int], str, bool], Sequence[float | None]]


@dataclass(frozen=True)
class Pairing:
    """How two stamps differ, sorted into the three kinds this rule cares about."""

    players: tuple[str, ...]
    rescore: tuple[str, ...]
    mixed: tuple[str, ...]


def compare(
    old: Mapping[str, Any], new: Mapping[str, Any], moved: Callable[[str], bool] | None = None
) -> Pairing:
    """Sort the fields two stamps differ in. ``moved(name)`` overrides plain inequality for
    a source fingerprint, so a stamp under the older bytes reading of an unchanged source is
    not read as moved (`measurement._moved`)."""

    def differs(name: str) -> bool:
        if moved is not None and name in ("engine", "metrics", "analysis"):
            return moved(name)
        return old.get(name) != new.get(name)

    players = [name for name in ("agents", "fit", "net") if differs(name)]
    if old.get("players") and new.get("players"):
        if differs("players"):
            players.append("players")
    elif differs("agents_source"):
        # What the core is handed cannot be read on one side: the agents' code decides,
        # as it does in `Basis.stale_against`.
        players.append("agents_source")
    rescore = [name for name in RESCORE_FIELDS if differs(name)]
    known = set(PLAYER_FIELDS) | set(RESCORE_FIELDS)
    mixed = [name for name in SAME_FIELDS if differs(name)]
    mixed += sorted(name for name in set(old) | set(new)
                    if name not in known and name not in SAME_FIELDS and differs(name))
    return Pairing(tuple(players), tuple(rescore), tuple(mixed))


@dataclass(frozen=True)
class Outcome:
    """What the read made of a pair: the word for the history line, and the record if one
    was written."""

    said: str
    record: dict[str, Any] | None = None


def _none(why: str) -> Outcome:
    return Outcome(f"{NO_VERDICT}: {why}")


def _totals(stored: Mapping[str, Any]) -> list[float | None]:
    return [None if run.get("index") is None else float(run["index"])
            for run in stored.get("runs") or [] if isinstance(run, Mapping)]


def judge(
    digest: str,
    runs_dir: Path,
    number: int,
    outgoing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    rescore: Rescore | None = None,
    moved: Callable[[str], bool] | None = None,
    rule: Rule = RULE,
) -> Outcome:
    """Read one replacement: write `convergence.json` where the pair differs only in its
    players, else say why not.

    ``outgoing`` and ``incoming`` are the two `canonical.json` contents, each with its
    ``basis``; ``number`` is the `previous/<n>` the outgoing one was kept as; ``rescore``
    re-scores its kept runs under the current metrics and weights.
    """
    from chess_variant_sandbox.metrics.seeds import minimum_totals

    old, new = outgoing.get("basis"), incoming.get("basis")
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        return _none("a measurement records no basis")
    pairing = compare(old, new, moved)
    if not pairing.players:
        return _none("the players did not change")
    if pairing.mixed:
        return _none("more than the players changed (" + ", ".join(pairing.mixed) + ")")

    kept = previous_root(digest, runs_dir) / str(number)
    runs = _totals(outgoing)
    rescored: dict[str, Any] | None = None
    if pairing.rescore:
        if rescore is None:
            return _none("the outgoing runs owe a re-score under the current metrics")
        reanalyse = "analysis" in pairing.rescore or (
            not old.get("analysis") and "metrics" in pairing.rescore)
        seeds = [int(seed) for seed in outgoing.get("seeds") or old.get("seeds") or ()]
        runs = [None if total is None else float(total)
                for total in rescore(kept, seeds, str(new.get("weights_version")), reanalyse)]
        rescored = {
            "moved": list(pairing.rescore),
            "reanalysed": reanalyse,
            "weights_version": new.get("weights_version"),
            "metrics": new.get("metrics"),
            "analysis": new.get("analysis"),
        }
    usable = [total for total in runs if total is not None]
    if len(usable) < max(2, minimum_totals(len(runs))):
        return _none(f"the outgoing measurement has {len(usable)} scored runs of {len(runs)}")
    if incoming.get("index") is None:
        return _none("the new measurement has no index")
    before = round(statistics.fmean(usable), 4) if rescored else outgoing.get("index")
    if before is None:
        return _none("the outgoing measurement has no index")
    after = float(incoming["index"])
    sd = round(statistics.stdev(usable), 4)
    move = round(after - float(before), 4)
    verdict = MOVED if abs(move) >= rule.convergence_sds * sd else HELD
    record = {
        "hash": digest,
        "from": measurement_digest(old),
        "to": measurement_digest(new),
        "from_index": float(before),
        "to_index": after,
        "move": move,
        "sd": sd,
        "runs": len(usable),
        "players": list(pairing.players),
        "rule": {
            "convergence_sds": rule.convergence_sds,
            "spelled": f"held when the index moved by less than {rule.convergence_sds:g} "
                       "times the outgoing runs' standard deviation, else moved",
        },
        "verdict": verdict,
        "rescored": rescored,
        "previous": f"previous/{number}",
        "written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        # The record's own basis: which two measurements, under which rule. A record whose
        # `to` is not the score's basis reads stale in `trust`.
        "basis": {"from": measurement_digest(old), "to": measurement_digest(new),
                  "rule": rule.convergence_sds, "rescored": rescored},
    }
    write(digest, runs_dir, number, record)
    return Outcome(verdict, record)


def write(digest: str, runs_dir: Path, number: int, record: Mapping[str, Any]) -> Path:
    """Write the record, keeping one already there under `previous/<number>/` first."""
    path = Path(runs_dir) / digest / CONVERGENCE_FILE
    if path.exists():
        kept = previous_root(digest, runs_dir) / str(number) / CONVERGENCE_FILE
        if not kept.exists():
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, kept)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2), encoding="utf-8")
    return path
