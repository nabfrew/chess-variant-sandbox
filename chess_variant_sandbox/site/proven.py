"""Proven positions on the site: the fourth challenge before the ten (R9b, Q605).

`docs/plan-website.md`, *Proven positions*. R9a (`chess_variant_sandbox/puzzles.py`,
`docs/reference/proven-positions.md`) proved positions from the library's own recorded
games won for the side to move, offline, and wrote them as `runs/puzzles/<hash>.json`
(`puzzles.FORMAT` 2; 1 is still read). This module reads that record and offers each
position two ways:

- **Convert it**: a real game from the position against *hardest*, the strongest
  computer player every board offers. It is a practice match started from the record's
  move prefix (`matches.start(..., puzzle=...)`, source `puzzle`), and it succeeds when
  the game ends in the person's win -- the engine's own outcome of the game, read off the
  match row at its finish (`converted`). Nothing new judges it.
- **Find the move**, only where the proof shows a single winning move: the person picks
  a move and the answer is the stored score of that root move, looked up (`find`). A move
  whose proof is *open* is answered *not settled*, never *wrong*: only a move the proof
  found lost is called one that loses.

**Rule 3.** Python holds a position and asks the engine; it never answers. The loader
replays each prefix through the engine, asking at every step whether the move is legal,
and drops (and counts) a position whose prefix no longer replays, whose position has
ended, whose side to move is not the record's, or whose legal moves are not the record's
root moves -- a record proved under rules or a core that no longer hold. It never
re-judges one. *Find the move* compares a move number with the record's numbers.

**Too easy** (Q993a, `docs/plan-proven.md` §3). The loader also leaves out, and counts, a
position the pre-registered test calls trivial (`puzzles.trivial`): a win in one with more
than one winning move (T1), half the moves or more proved to win (T2), or, in a `FORMAT` 2
record, T3's stored answer that nothing the defender does changes the win (`none`). It
reads the record's numbers and decides nothing; the basis line says how many and why.

**The seven tests** (`docs/reference/no-dark-patterns.md`). No timer and no streak; a
person's marks (`proven_marks`) are read on their own page alone and no count across
people is read anywhere; nothing expires; a failed try writes nothing and is a game the
person lost; and the record's basis is on the page (`basis_said`).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.puzzles import FORMATS, LOST, OPEN, WON, trivial
from chess_variant_sandbox.web import card_rules, opponents
from chess_variant_sandbox.web.library import engine_for

DIRECTORY = "puzzles"
"""Where the records sit under `runs/`, as R9a writes them."""

SOURCE = "puzzle"
"""`matches.source` on a try at converting a proven position (`0037_proven_positions.sql`)."""

CONVERTED = "converted"
FOUND = "found"
WAYS = (CONVERTED, FOUND)

OPPONENT = next(agent for agent, name, _tail in opponents.RUNGS if name == "hardest")
"""*Convert it* is played against *hardest*, the ladder's strongest rung on every board
that alternates (`minimax:nodes=200000`). Its seat is rationed like any other
(`quota.check_new_match`), and it is for people with an account (L15)."""

NOT_HERE = "There is no proven position at this address on this game."
NO_TAKE_BACK = (
    "a game from a proven position has no take-back: it is decided by how the game ends. "
    "Start it again from the position instead"
)
SIGN_IN_TO_CONVERT = (
    "Sign in to convert a position: it is played against hardest, which is for people "
    "with an account. Finding the move needs no account."
)


# ------------------------------------------------------------------------ the record


@dataclass(frozen=True)
class Position:
    """One proved position, as the record holds it and the loader replayed it.

    `number` is its place in the order the page lists them, shortest proved win first;
    `id` is the record's own name for it (`seed-game-ply`), which the store keys a mark
    and a try by. `scores` are the record's root moves in the engine's order, each with
    its `move`, `text` and `proof` (`won`, `lost` or `open`) and, when proved, `ends_in`.
    """

    id: str
    number: int
    moves: tuple[int, ...]
    side: int
    ends_in: int
    pieces: int
    single: bool
    others_lost: bool
    scores: tuple[dict[str, Any], ...]

    @property
    def ply(self) -> int:
        return len(self.moves)

    def score_of(self, move: int) -> dict[str, Any] | None:
        return next((entry for entry in self.scores if entry["move"] == move), None)

    def winning(self) -> list[dict[str, Any]]:
        return [entry for entry in self.scores if entry["proof"] == WON]


@dataclass(frozen=True)
class Record:
    """A game's proven positions, those still offered, and what the record says of itself."""

    game: str
    made: str
    basis: dict[str, Any]
    counts: dict[str, int]
    positions: tuple[Position, ...]
    dropped: int
    too_easy: int = 0
    """Positions that replay but the triviality test leaves out (T1, T2 or T3)."""
    defence: bool = False
    """Whether the record carries T3's answers (`FORMAT` 2 with `defence_nodes`)."""

    def find(self, position_id: str) -> Position | None:
        return next((found for found in self.positions if found.id == position_id), None)

    @property
    def singles(self) -> int:
        return sum(1 for found in self.positions if found.single)


def path(runs_dir: Path | str, digest: str) -> Path:
    """Where a game's record is, whether or not it exists."""
    return Path(runs_dir) / DIRECTORY / f"{digest}.json"


_CACHE: dict[tuple[str, str], tuple[tuple[int, int, str], Record | None]] = {}
_LOCK = threading.Lock()


def load(runs_dir: Path | str, digest: str, spec: dict[str, Any] | None) -> Record | None:
    """The game's record, its positions replayed through today's rules, or None.

    None where there is no record (three of the launch set have none, and every member's
    game), where it is not a format it reads (`FORMATS`) or not this game's, and where no
    spec is given. A
    record is read once a process for each version of the file on disk and these rules,
    since the replay costs a few tenths of a second on a record of 150.
    """
    if spec is None:
        return None
    where = path(runs_dir, digest)
    try:
        held = where.stat()
    except OSError:
        return None
    stamp = (held.st_mtime_ns, held.st_size, json.dumps(spec, sort_keys=True))
    key = (str(where), digest)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        data = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    found = _read(data, digest, spec) if isinstance(data, dict) else None
    with _LOCK:
        _CACHE[key] = (stamp, found)
    return found


def offered(runs_dir: Path | str, digest: str, spec: dict[str, Any] | None) -> int:
    """How many proven positions the game's page offers: nought where it has no section.

    The games list's *proven positions* filter reads this (Q972), the same `load` and the
    same positions the game page's section draws, so a card the filter keeps is exactly a
    game whose page has the section. A game with no record costs one `stat`.
    """
    record = load(runs_dir, digest, spec)
    return 0 if record is None else len(record.positions)


def _read(data: dict[str, Any], digest: str, spec: dict[str, Any]) -> Record | None:
    if data.get("format") not in FORMATS or data.get("ruleset_hash") != digest:
        return None
    engine = engine_for(spec)
    kept: list[tuple[int, int, dict[str, Any]]] = []
    dropped = 0
    too_easy = 0
    for order, row in enumerate(data.get("positions") or []):
        if not isinstance(row, dict) or row.get("result") != WON or not _replays(engine, row):
            dropped += 1
            continue
        if trivial(row):
            too_easy += 1
            continue
        kept.append((int(row.get("ends_in") or 0), order, row))
    # Shortest proved win first, then the record's own order: a fixed order the server
    # decides, so the page sorts nothing.
    kept.sort(key=lambda entry: (entry[0], entry[1]))
    positions = tuple(
        Position(
            id=str(row["id"]),
            number=number,
            moves=tuple(int(move) for move in row["moves"]),
            side=int(row["side"]),
            ends_in=int(row.get("ends_in") or 0),
            pieces=int(row.get("pieces") or 0),
            single=bool(row.get("single")),
            others_lost=bool(row.get("others_lost")),
            scores=tuple(dict(entry) for entry in row["scores"]),
        )
        for number, (_ends, _order, row) in enumerate(kept, start=1)
    )
    return Record(
        game=digest,
        made=str(data.get("made") or ""),
        basis=dict(data.get("basis") or {}),
        counts={key: int(value) for key, value in (data.get("counts") or {}).items()},
        positions=positions,
        dropped=dropped,
        too_easy=too_easy,
        defence=data.get("format") != 1 and bool((data.get("basis") or {}).get("defence_nodes")),
    )


def _replays(engine: Any, row: dict[str, Any]) -> bool:
    """Whether the engine still reaches the position the record proved, as the record says.

    Every prefix move is put to `legal_moves` before it is applied, the position must not
    have ended, the side to move must be the record's, and the engine's legal moves must
    be the record's root moves in the same order. The engine answers each question.
    """
    try:
        moves = [int(move) for move in row["moves"]]
        side = int(row["side"])
        scores = [int(entry["move"]) for entry in row["scores"]]
    except (KeyError, TypeError, ValueError):
        return False
    state = engine.initial_state()
    for move in moves:
        if move not in engine.legal_moves(state):
            return False
        state = engine.apply(state, move)
    legal = engine.legal_moves(state)
    if engine.terminal(state, legal) is not None or state.side_to_move != side:
        return False
    return list(legal) == scores


def state_of(spec: dict[str, Any], position: Position) -> Any:
    """The position itself, reached by replaying its prefix through the engine."""
    engine = engine_for(spec)
    return engine.play(engine.initial_state(), position.moves)


# --------------------------------------------------------------------- what is said


def _thousands(number: int) -> str:
    return f"{number:,}"


def _millions(number: int) -> str:
    if number % 1_000_000 == 0:
        return f"{number // 1_000_000} million"
    return _thousands(number)


def basis_said(record: Record) -> str:
    """The proof's basis, in a visitor's words (test 7): where the positions came from,
    how they were proved, how many were asked and proved, and when."""
    basis = record.basis
    counts = record.counts
    runs = len(basis.get("runs") or [])
    near = basis.get("near")
    pieces = basis.get("max_pieces")
    asked = counts.get("candidates", 0)
    won = counts.get("won", 0)
    unsettled = counts.get("open", 0)
    where = (
        f"Positions from the games the computer players played in this game's {runs} runs"
        if runs else "Positions from the games the computer players played in this game"
    )
    which = []
    if pieces:
        which.append(f"with {pieces} pieces or fewer left")
    if near:
        which.append(f"within {near} turns of the end of a game one side won, with the "
                     "winner to move")
    said = where + (", " + ", ".join(which) if which else "") + ". "
    nodes = basis.get("nodes")
    said += (
        "The computer searched every move from each, "
        + (f"looking at up to {_millions(int(nodes))} positions, " if nodes else "")
        + "and a position is here only where that search found a forced win for the "
        "side to move. "
    )
    if asked:
        forced = counts.get("forced", 0)
        said += (
            f"Of {_thousands(asked)} positions asked it proved {_thousands(won)}"
            + (f"; it could not settle {_thousands(unsettled)}" if unsettled else "")
            + (f", and {_thousands(forced)} had only one move to play" if forced else "")
            + ". "
        )
    if record.made:
        said += f"Proved on {record.made}."
    if record.too_easy:
        said += (f" Of the {_thousands(won)} it proved, " if won else " ") + (
            f"{_thousands(record.too_easy)} {TOO_EASY[record.defence]}")
    if record.dropped:
        said += (
            f" {record.dropped} more no longer replay under these rules, so they are "
            "not offered."
        )
    return said


TOO_EASY = {
    True: (
        "were left out as too easy: the win was the very next move with more than one way "
        "to make it, half the moves or more won, or nothing the defender could do changed "
        "it."
    ),
    False: (
        "were left out as too easy: the win was the very next move with more than one way "
        "to make it, or half the moves or more won."
    ),
}
"""The basis line's clause for the positions the triviality test leaves out (test 7),
keyed by whether the record carries T3's answers: it names only the tests applied."""


def _turns(count: int) -> str:
    return f"{count} turn{'' if count == 1 else 's'}"


TRY_SAID = {
    "said": (
        "A proven position: the computer proved that the side to move, which is you, wins "
        "from here with best play. Play it out against hardest. You convert it if the "
        "game ends in your win. There is no take-back and no clock."
    ),
    "won": "You converted it: the game ended in your win.",
    "lost": (
        "This try did not end in your win. It is a game you lost and nothing more: start "
        "it again from the position whenever you like."
    ),
}
"""What the board says on a try at converting a position: while it runs, and at each
ending. The page shows `won` where the engine's outcome names the person's side the
winner and `lost` at any other ending: it reads the outcome the board already draws, and
decides nothing itself."""


def try_snapshot(position_id: str, ply: int, side: int) -> dict[str, Any]:
    """The `puzzle` block of a try's snapshot: which position, where its prefix ends, the
    person's side and the board's sentences."""
    return {"id": position_id, "from": ply, "side": side, **TRY_SAID}


# ------------------------------------------------------------------ the game's page


def section(
    record: Record,
    marks: dict[str, set[str]] | None,
    signed_in: bool,
) -> dict[str, Any]:
    """*Proven positions* on the game's page: the basis, the list, and the reader's marks.

    `marks` is the reader's own (`marks_on`), position id to the ways they solved it, and
    None for a visitor, who has none. Nothing here counts anybody else.
    """
    mine = marks or {}
    positions = [
        {
            "id": found.id,
            "number": found.number,
            "side": found.side,
            "endsIn": found.ends_in,
            "single": found.single,
            "converted": CONVERTED in mine.get(found.id, set()),
            "found": FOUND in mine.get(found.id, set()),
        }
        for found in record.positions
    ]
    yours = None
    if marks is not None:
        yours = {
            "converted": sum(1 for entry in positions if entry["converted"]),
            "found": sum(1 for entry in positions if entry["found"]),
        }
    return {
        "basis": basis_said(record),
        "count": len(record.positions),
        "singles": record.singles,
        "positions": positions,
        "yours": yours,
        "signInToConvert": "" if signed_in else SIGN_IN_TO_CONVERT,
    }


def view(record: Record, spec: dict[str, Any], position_id: str) -> dict[str, Any]:
    """One position, drawn: the still board, whose move it is, and, where the proof shows
    a single winning move, the moves to pick from -- each move and its words, never its
    score, which is the answer."""
    found = record.find(position_id)
    if found is None:
        raise KeyError(NOT_HERE)
    engine = engine_for(spec)
    state = state_of(spec, found)
    return {
        "id": found.id,
        "number": found.number,
        "side": found.side,
        "endsIn": found.ends_in,
        "single": found.single,
        "preview": card_rules.preview(engine, state, spec),
        "moves": (
            [{"move": entry["move"], "text": entry["text"]} for entry in found.scores]
            if found.single else []
        ),
        "said": (
            "The computer proved a win from "
            f"here in {_turns(found.ends_in)} with best play on both sides."
        ),
    }


def answer(record: Record, position_id: str, move: int | None, show: bool = False) -> dict[str, Any]:
    """*Find the move*: the stored proof of the move picked, looked up, in plain words.

    `proof` is the record's word for that root move. `right` is true only for the move the
    proof names; a move proved lost is said to lose, and an *open* one is said to be not
    the move the computer proved, which it could not settle either way -- never *wrong*.
    `show` gives the proved move without a pick, and marks nothing.
    """
    found = record.find(position_id)
    if found is None:
        raise KeyError(NOT_HERE)
    if not found.single:
        raise ValueError(
            "this position has more than one winning move, so it is offered to convert "
            "rather than to find the move"
        )
    best = found.winning()[0]
    if show:
        return {
            "proof": None,
            "right": False,
            "shown": best["text"],
            "said": (
                f"The move the computer proved wins is {best['text']}, in "
                f"{_turns(int(best.get('ends_in') or found.ends_in))} with best play."
            ),
        }
    entry = found.score_of(int(move)) if move is not None else None
    if entry is None:
        raise ValueError("that is not one of the moves in this position")
    proof = str(entry["proof"])
    text = str(entry["text"])
    if proof == WON:
        said = (
            f"Yes: {text} is the move the computer proved wins, in "
            f"{_turns(int(entry.get('ends_in') or found.ends_in))} with best play."
        )
    elif proof == LOST:
        said = f"No: the computer proved {text} loses."
        if found.others_lost:
            said += " Every move here but one loses."
    elif proof == OPEN:
        said = (
            f"{text} is not the move the computer proved. It could not settle this one "
            "either way, so it may win too, or it may not."
        )
    else:
        raise ValueError(f"the record says {proof!r} of that move, which is no proof")
    return {"proof": proof, "right": proof == WON, "move": text, "said": said}


# ------------------------------------------------------------------- a person's marks


def mark(
    connection: Any, person: int, game: str, position: str, way: str, match: str | None = None
) -> bool:
    """A person's solved mark, once. Whether it is new. Does not commit."""
    if way not in WAYS:
        raise ValueError(f"a mark is {CONVERTED} or {FOUND}, not {way!r}")
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into proven_marks (person, game, position, way, match)"
            " values (%s, %s, %s, %s, %s) on conflict do nothing",
            (person, game, position, way, match),
        )
        return bool(cursor.rowcount)


def converted(connection: Any, match_id: str) -> bool:
    """At a finish: mark the position converted if this try's game ended in the person's win.

    Called by `matches._finish` in the finish's own transaction, after the verdict is on
    the row. The verdict is the row's `result`, which is the engine's outcome of the game
    (or a resignation or an agreed draw, which are never the person's win). A visitor's
    try has nobody to mark, and a game nobody sat at is no try.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, puzzle, seat0_user, seat1_user, seat0_agent, seat1_agent, result"
            " from matches where id = %s",
            (match_id,),
        )
        found = cursor.fetchone()
    if found is None or found[1] is None or found[6] is None:
        return False
    game, puzzle, user0, user1, agent0, agent1, result = found
    if agent0 is None and user0 is not None and agent1 is not None:
        person, score = int(user0), float(result)
    elif agent1 is None and user1 is not None and agent0 is not None:
        person, score = int(user1), 1.0 - float(result)
    else:
        return False
    if score != 1.0:
        return False
    return mark(connection, person, str(game), str(puzzle), CONVERTED, match_id)


def marks_on(connection: Any, person: int, game: str) -> dict[str, set[str]]:
    """This person's marks on one game: position id to the ways they solved it."""
    found: dict[str, set[str]] = {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select position, way from proven_marks where person = %s and game = %s",
            (person, game),
        )
        for position, way in cursor.fetchall():
            found.setdefault(str(position), set()).add(str(way))
    return found


def games_marked(connection: Any, person: int) -> dict[str, dict[str, set[str]]]:
    """Every game this person has a mark on, each with `marks_on`'s answer."""
    found: dict[str, dict[str, set[str]]] = {}
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, position, way from proven_marks where person = %s", (person,)
        )
        for game, position, way in cursor.fetchall():
            found.setdefault(str(game), {}).setdefault(str(position), set()).add(str(way))
    return found


def forget(connection: Any, person: int) -> None:
    """Leaving takes a person's marks with their address (`admin.forget`)."""
    with connection.cursor() as cursor:
        cursor.execute("delete from proven_marks where person = %s", (person,))


def profile_lines(
    marked: dict[str, dict[str, set[str]]],
    records: dict[str, Record | None],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    """The person's own page: on each game they have a mark on, how many of its positions
    they converted and how many moves they found, of how many it offers.

    Counted against the positions offered now, so a mark on a position a later record no
    longer offers is not counted. A game with no record left reads its marks alone.
    """
    lines = []
    for game in sorted(marked, key=lambda digest: (names.get(digest, "").lower(), digest)):
        record = records.get(game)
        offered = {found.id for found in record.positions} if record is not None else None
        singles = (
            {found.id for found in record.positions if found.single}
            if record is not None else None
        )
        by_position = marked[game]
        done = [pid for pid, ways in by_position.items() if CONVERTED in ways]
        found_ = [pid for pid, ways in by_position.items() if FOUND in ways]
        if offered is not None:
            done = [pid for pid in done if pid in offered]
            found_ = [pid for pid in found_ if singles is not None and pid in singles]
        lines.append({
            "hash": game,
            "name": names.get(game, ""),
            "converted": len(done),
            "found": len(found_),
            "of": None if record is None else len(record.positions),
            "singles": None if record is None else record.singles,
        })
    return lines


def seats_for(position: Position) -> list[str]:
    """The two seats of a try: the person on the side to move, *hardest* on the other."""
    from chess_variant_sandbox.web.human import HUMAN

    seats = [OPPONENT, OPPONENT]
    seats[position.side] = HUMAN
    return seats
