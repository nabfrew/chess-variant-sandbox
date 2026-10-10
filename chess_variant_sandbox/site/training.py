"""The pipeline's training queue, and people's games' one way into it (route C, H9).

`docs/plan-game-pipeline.md` trains a game's computer player on the 16-core machine, today
only where the computer-only strength-premise check says the built-in player is weak
(Q503k). The user answered `docs/plan-human-data.md` §6 **B** on 2026-10-08: people's games
may decide *where the computer players look next*, never *what is found*. So a game over
the bar of §3.3 (`site/alarm.py`) is queued for a stronger computer player automatically,
the trained player is still accepted only by computer-only duels, and nothing a person
played reaches a score (rule 5).

**The queue** is `jobs` rows of kind `train`, one game each, with `cause` saying who asked
(`0033_pinned_and_alarm.sql`): `premise`, the premise check's own entry, or `people`, the
bar's. **Every premise entry is served before any people entry**, oldest first within each
(`entries`, `claim`): the computer-only check's own verdict is the queue's first business,
and people only say which game is next after it.

**One open entry a game.** A game with a train job queued or running gets no second one,
whoever asks. A people entry is raised once per top group: the evidence keeps the group it
was read against, so the same strongest player beaten again queues nothing more, while a
new top group that people beat as well can be queued again. A failed entry does not count,
so the next finish over the bar asks again.

**What is not built here.** The machine that pulls a `train` job, trains, and has the net
accepted by computer-only duels is the pipeline lane (Q503g) and Q503k; neither exists yet,
so an entry waits in the queue until they do. `claim` is the contract they pull by.

Nothing here commits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chess_variant_sandbox.site import alarm

KIND = "train"

PREMISE = "premise"
PEOPLE = "people"
CAUSES = (PREMISE, PEOPLE)
"""Who asked, in the order the queue serves them."""

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
OPEN = (QUEUED, RUNNING)


class TrainingError(ValueError):
    """A queue entry the queue cannot take."""


@dataclass(frozen=True)
class Entry:
    id: int
    game: str
    cause: str
    state: str
    created: datetime
    evidence: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hash": self.game,
            "cause": self.cause,
            "state": self.state,
            "created": self.created.isoformat(),
            "evidence": self.evidence,
        }


_COLUMNS = "id, hash, cause, state, created, evidence"
# Premise entries first, then people's, each oldest first; `id` breaks a tie.
_ORDER = f"case cause when '{PREMISE}' then 0 else 1 end, created, id"


def _entry(found: Any) -> Entry:
    return Entry(
        id=int(found[0]),
        game=str(found[1]),
        cause=str(found[2]),
        state=str(found[3]),
        created=found[4],
        evidence=None if found[5] is None else dict(found[5]),
    )


def queue(
    connection: Any, game: str, cause: str, evidence: dict[str, Any] | None = None
) -> int | None:
    """Queue `game` for training, or None where the queue already holds it.

    None when a train job of the game is queued or running, and for a people entry when one
    that has not failed was raised against the same top group (`evidence["top"]`).
    """
    if cause not in CAUSES:
        raise TrainingError(f"a training entry is asked for by {' or '.join(CAUSES)}")
    if cause == PEOPLE and not (evidence and evidence.get("top")):
        raise TrainingError("people's entry names the top group it was read against")
    with connection.cursor() as cursor:
        # One game's entries are decided one at a time: two finishes over the bar
        # together queue it once.
        cursor.execute("select pg_advisory_xact_lock(hashtext(%s))", (f"train:{game}",))
        cursor.execute(
            "select 1 from jobs where kind = %s and hash = %s and state = any(%s)",
            (KIND, game, list(OPEN)),
        )
        if cursor.fetchone() is not None:
            return None
        if cause == PEOPLE and evidence is not None:
            cursor.execute(
                "select 1 from jobs where kind = %s and hash = %s and cause = %s"
                " and state <> %s and evidence -> 'top' = %s::jsonb",
                (KIND, game, PEOPLE, FAILED, json.dumps(sorted(evidence["top"]))),
            )
            if cursor.fetchone() is not None:
                return None
        stored = None
        if evidence is not None:
            stored = dict(evidence)
            if "top" in stored:
                stored["top"] = sorted(stored["top"])
        cursor.execute(
            "insert into jobs (kind, hash, state, cause, evidence)"
            " values (%s, %s, %s, %s, %s::jsonb) returning id",
            (KIND, game, QUEUED, cause,
             None if stored is None else json.dumps(stored)),
        )
        found = cursor.fetchone()
    return int(found[0])


def entries(connection: Any, states: tuple[str, ...] = OPEN) -> list[Entry]:
    """The queue's entries in the order the pipeline serves them."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select {_COLUMNS} from jobs where kind = %s and state = any(%s) order by {_ORDER}",
            (KIND, list(states)),
        )
        return [_entry(found) for found in cursor.fetchall()]


def claim(connection: Any, worker: str, fingerprint: str) -> Entry | None:
    """Take the next queued entry for `worker`, premise entries first, or None.

    The pull the pipeline lane (Q503g) makes; one atomic update, as the check lane's
    claims are, carrying the worker's build in `fingerprint`.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"update jobs set state = %s, claimed = now(), worker = %s, fingerprint = %s"
            f" where id = (select id from jobs where kind = %s and state = %s"
            f" order by {_ORDER} limit 1 for update skip locked)"
            f" returning {_COLUMNS}",
            (RUNNING, worker, fingerprint, KIND, QUEUED),
        )
        found = cursor.fetchone()
    return None if found is None else _entry(found)


def alarm_for(connection: Any, game: str) -> int | None:
    """Queue `game` as people's entry if it is over the bar now; the job id, else None.

    What a rated finish against the top calls (`matches._finish`).
    """
    found = alarm.read(connection, game)
    if not found.over:
        return None
    return queue(connection, game, PEOPLE, found.evidence())


def held(connection: Any) -> dict[str, Entry]:
    """Each game's latest train entry of any state, for `/admin`'s line beside the bar."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"select distinct on (hash) {_COLUMNS} from jobs where kind = %s"
            f" order by hash, created desc, id desc",
            (KIND,),
        )
        return {str(found[1]): _entry(found) for found in cursor.fetchall()}
