"""The refresher's status, in words, for `/admin` and the local application (Q705).

`chess-variant-sandbox refresh --library --watch` (`refresher.py`, EV5) writes
`runs/refresh-status.json` after every step; this module reads it through
`refresher.read_status` and says what it means: what the refresher is doing now, the
game, how long ago it last wrote, what it refused and why, and how many games are owed
a repair. Both pages draw the sentences and rows it returns and decide nothing.

It is the maintainer's: `/api/admin` and `/api/scoring`, both admin-only on the site, are
its only routes, so a visitor is never sent it. A missing or unreadable file is not an
error, it is a refresher that is not running here.

A status file says what the refresher was doing when it last wrote, and it writes at
every pass of its loop. One that has gone quiet for longer than a pass can take is a
refresher that stopped without saying so (a closed terminal, a lost machine), whatever
its last state says. `QUIET_AFTER_S` is that limit: three idle passes at the default
interval (`Refresher.interval`, 600 s), or, while a game is being repaired, four hours,
longer than any one game's replay has taken.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chess_variant_sandbox import refresher

QUIET_AFTER_S = {"working": 4 * 3600}
QUIET_AFTER_DEFAULT_S = 3 * 600

# What each repair does, in the local application's words (`upkeep.js`' `REPAIRS`).
REPAIRS = {
    "refit": "checking its fit",
    "rescore": "scoring again",
    "reanalyse": "analysing and scoring again",
    "replay": "playing again",
}

# Why a game was refused, in plain words, by `refresher`'s refusal kinds.
REFUSALS = {
    refresher.STALE_BUILD: "the game engine changed and was not compiled again",
    refresher.FIT_NOT_REPRODUCED: "its fitted player could not be reproduced on this build",
    refresher.GATE_NEWLY_FAILED: "it passed its checks before and fails them now",
    refresher.COMMIT_MOVED: "the code changed while it was being measured",
    refresher.CHANGED_COMMITTED_FILES: "measuring it changed files kept in the repository",
    refresher.ERROR: "measuring it failed",
}


def ago(seconds: float | None) -> str:
    """`seconds` as a person says it: *just now*, *4 minutes ago*, *2 days ago*."""
    if seconds is None:
        return "at a time it did not record"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "just now"
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            count = seconds // size
            return f"{count} {unit}{'' if count == 1 else 's'} ago"
    return "just now"  # pragma: no cover - the loop always returns


def _game(entry: Mapping[str, Any]) -> str:
    return str(entry.get("name") or str(entry.get("hash") or "")[:12] or "a game")


def _refusal(entry: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(entry.get("kind") or "")
    return {
        "hash": str(entry.get("hash") or ""),
        "name": _game(entry),
        "needs": REPAIRS.get(str(entry.get("needs") or ""), ""),
        "why": REFUSALS.get(kind, kind or "refused"),
        "reason": str(entry.get("reason") or ""),
        "at": str(entry.get("at") or ""),
    }


def state_of(status: Mapping[str, Any] | None) -> str:
    """`none`, `quiet` (stopped writing), or the state the file names."""
    if not status:
        return "none"
    state = str(status.get("state") or "")
    age = status.get("age_s")
    limit = QUIET_AFTER_S.get(state, QUIET_AFTER_DEFAULT_S)
    if state != "stopped" and (age is None or age > limit):
        return "quiet"
    return state or "quiet"


def view(runs_dir: Path) -> dict[str, Any]:
    """Everything a page draws about the refresher, read from `runs_dir`.

    `running` is whether it is running now; `said` is the headline and the lines under
    it, in order; `refused` is every standing refusal, newest first; `owed` and
    `owedCounts` are what it found owed at its last survey, refused games apart.
    """
    status = refresher.read_status(runs_dir)
    state = state_of(status)
    if status is None:
        return {"state": "none", "running": False, "said": [
            "The refresher is not running here: it has never written a status."],
            "ageS": None, "doing": None, "owed": 0, "owedCounts": {}, "refusedOwed": 0,
            "refused": [], "lastDone": None, "lastRefusal": None}

    age = status.get("age_s")
    doing = status.get("doing") if isinstance(status.get("doing"), Mapping) else None
    head = {
        "quiet": f"The refresher is not running: it last wrote {ago(age)}, and it writes at "
        "every step, so it stopped without saying so.",
        "stopped": f"The refresher is not running: it stopped {ago(age)}.",
        "starting": f"The refresher is starting (written {ago(age)}).",
        "idle": f"The refresher is running and idle: nothing it can repair is owed "
        f"(written {ago(age)}).",
        "waiting": f"The refresher is running and waiting for "
        f"{status.get('waiting_for') or 'the machine'} (written {ago(age)}).",
        "halted": "The refresher is running and takes no game until the code moves on: "
        f"{(status.get('last_refusal') or {}).get('reason') or 'it halted'} "
        f"(written {ago(age)}).",
    }.get(state)
    if head is None and state == "working":
        head = (f"The refresher is working on {_game(doing)}: "
                f"{REPAIRS.get(str(doing.get('needs')), str(doing.get('needs')))}, "
                f"since {ago(age)}." if doing else
                f"The refresher is running (written {ago(age)}).")
    if head is None:
        head = f"The refresher last said it was {state} ({ago(age)})."
    said = [head]

    counts = {k: int(v) for k, v in (status.get("owed_counts") or {}).items()
              if k in REPAIRS and v}
    owed = sum(counts.values())
    refused_owed = int(status.get("refused_owed") or 0)
    if owed:
        parts = [f"{counts[k]} {REPAIRS[k]}" for k in refresher.REPAIR_ORDER if counts.get(k)]
        said.append(f"{owed} game{'' if owed == 1 else 's'} out of date at its last look: "
                    + ", ".join(parts) + ".")
    else:
        said.append("No game was out of date at its last look.")
    if refused_owed:
        said.append(f"{refused_owed} more {'is' if refused_owed == 1 else 'are'} out of date "
                    "and refused, so it does not try them again.")

    last = status.get("last_done") if isinstance(status.get("last_done"), Mapping) else None
    if last:
        index = last.get("index")
        number = "no score" if index is None else f"{float(index):.4f}"
        said.append(f"Last repaired: {_game(last)}, "
                    f"{REPAIRS.get(str(last.get('needs')), str(last.get('needs')))}, "
                    f"now {number}.")
    merged = status.get("owed_merge") or []
    if merged:
        said.append(f"Waiting to be merged into main from branch {refresher.BRANCH}: "
                    + ", ".join(str(m) for m in merged) + ".")

    standing = [r for r in status.get("refusals") or [] if isinstance(r, Mapping)]
    refused = sorted((_refusal(r) for r in standing), key=lambda r: r["at"], reverse=True)
    raw = status.get("last_refusal")
    return {
        "state": state,
        "running": state not in ("none", "quiet", "stopped"),
        "said": said,
        "ageS": age,
        "doing": ({"hash": str(doing.get("hash") or ""), "name": _game(doing),
                   "needs": REPAIRS.get(str(doing.get("needs")), "")}
                  if doing and state == "working" else None),
        "owed": owed,
        "owedCounts": counts,
        "refusedOwed": refused_owed,
        "refused": refused,
        "lastDone": dict(last) if last else None,
        "lastRefusal": _refusal(raw) if isinstance(raw, Mapping) else None,
    }
