"""Proven positions: small endgames from a game's recorded runs that the core has proved.

`docs/plan-website.md`, *Proven positions* (R9a); the campaign and its numbers are
`docs/reference/proven-positions.md`, driven by the script beside it.

A position is carried as the move prefix that reaches it, so there is no new position
format: the site replays the prefix through the engine, as it replays any match. The record
calls it by where it was found (`id`, `seed-game-ply`, a game of a library run and a ply);
the site names it by what it is (`name`, Q993c), a digest of the replayed state, so a
re-proof from other runs can neither move a person's mark onto another position nor lose it. Its verdict is the core's
analysis probe (`Engine.root_scores`, a full-window search under a node budget), never
Python's. This module holds positions, asks the core, and reads the numbers back: a root
move whose score is at least `PROVEN` was searched to a forced win, and one at most
`-PROVEN` to a forced loss, because the probe is full width and scores an ending as
`WIN_SCORE - ply` while no evaluation comes near that. Anything between is *open*: the
budget ran out first, and the move is neither won nor lost as far as anyone knows.

The record, `runs/puzzles/<hash>.json` (`FORMAT` 4), is what the site loads (R9b). Each
position carries its prefix, the side to move, its result (`won` or `lost` for the side to
move), the plies to the ending under best play as far as the proof saw it, and every root
move's score and proof, so *find the move* is a lookup.

**Too easy** (`docs/archive/plan-proven.md` §3.1, pre-registered before anything was counted). A
position is trivial under T1 (a win in one with two or more winning moves), T2 (half the
legal moves or more proved won), both read off the record's own proofs (`trivial`), or T3
(the defence has no say: after the shortest winning move, every reply leaves a win exactly
two plies shorter), which asks the core again (`defence`). T4 (`docs/archive/plan-proven.md` §3.6,
adopted after the look that motivated it): the defence can only hasten it, every reply
leaves a win at most two plies shorter, asked the same way (`hasten`). T5 (`docs/archive/plan-proven.md`
§3.7, pre-registered): seen at a glance, the core's two quickest players, asked from eight
fixed seeds, play a move that wins (`shallow`). `FORMAT` 2 adds each asked position's T3
answer and the basis it was asked on, `FORMAT` 3 T4's beside it, `FORMAT` 4 T5's; `FORMAT`
1 records carry none of them.

**Upkeep** (Q993d, `docs/archive/plan-proven.md` §5). Nobody runs a driver: `survey` reads which
Library and Baseline games are owed a record and why (`owes`: none yet, positions that no
longer replay, or a record older than `FORMAT` or `PROOF_VERSION`), and `make` writes one
whole, every stage resumable by its unit file under `runs/puzzles/work/<hash>/`: the count,
the sample (`NEAR_MIN`, §3.5), the proofs, T3, T4 and T5. A game whose runs hold no small
ending gets a record with none, so it is asked once. The refresher's proofs tier calls both
(`refresher.py` point 9), and `chess-variant-sandbox proofs [--check]` is the same by hand.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, WIN_SCORE
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_EN_PASSANT,
    MOVE_FOREIGN,
    MOVE_NORMAL,
    MOVE_PROMOTION,
    MOVE_TYPE_NAMES,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import choose_move, eval_values
from chess_variant_sandbox.tournament.record import GameRecord

FORMAT = 4
FORMATS = (1, 2, 3, 4)
"""Every record format a reader accepts: 1 has no T3 answers, 2 has them, 3 adds T4's, 4
adds T5's."""
NODES = 20_000_000
# T3's budget a reply (`docs/archive/plan-proven.md` §3.1): a tenth of the proof's, since each of 20 to
# 30 replies is asked and the position after a reply is two plies nearer its end.
DEFENCE_NODES = 2_000_000
MAX_DEPTH = 400
MAX_PIECES = 6
# Positions proved per game, at most. E2 found 241 on *Intransitive*'s predecessor and the
# plan's cost (3-4 core-hours a game at 20 M nodes) is that many; today's runs hold
# thousands per game (`docs/reference/proven-positions.md` §1), so they are sampled.
CAP = 240
# How far before a won game's recorded end a position is asked. A 20 M-node full-width
# probe proves about eight plies ahead on *Intransitive*, and nothing proved further back
# in the first, evenly spread sample (`proven-positions.md` §2), so twelve leaves room.
NEAR = 12
# And how near is too near (`docs/archive/plan-proven.md` §3.5, Q993d): a position fewer than this many
# plies before its game's recorded end is a win in one or two by construction, which T1
# mostly removes, so the sample skips it. Like `NEAR` it decides where an expensive search
# points, never what a position is worth, and the basis names it (`near_min`).
NEAR_MIN = 3
# The probe's own version: a fix to the proof itself (`verdict`, the probe's budget or how a
# score is read) that could change what was proved bumps it, and every record is owed again
# (`owes`). A record written before it existed carries none and reads as version 1, the
# probe as R9a ran it; a change to where the sample looks (`NEAR_MIN`) is not one.
PROOF_VERSION = 1
# A score this close to a win is the probe having reached an ending, not an evaluation:
# no evaluator reads within a thousand of `WIN_SCORE`, and no game's clock runs that long.
PROVEN = WIN_SCORE - 1000

WON, LOST, OPEN = "won", "lost", "open"

# T3's answers, a position's `defence` in a `FORMAT` 2 or 3 record (absent where T3 does not
# apply): no reply changed the shortest win; some reply was proved to change it; or none
# was proved to, but at least one the probe could not settle, which counts as a say.
NONE, SOME, UNSETTLED = "none", "some", "unsettled"
DEFENCES = (NONE, SOME, UNSETTLED)
# T4's answers, a position's `hasten` in a `FORMAT` 3 record, are the same three words
# read against *at most* n - 2: `none` is that no reply did anything but keep pace or lose
# faster; `some`, that a reply was proved to put the win off or escape it.
T1, T2, T3, T4, T5 = "T1", "T2", "T3", "T4", "T5"

# T5 (`docs/archive/plan-proven.md` §3.7): the core's two quickest players a person meets, at their own
# budgets on the built-in objective (*greedy* and *easy* on the site), each asked from these
# fixed seeds. A position's `shallow` is `found` where any choice wins and `missed` where none
# does; there is no third answer, since a choice the core cannot prove won counts as a miss.
SHALLOW_AGENTS = ("greedy", "minimax:nodes=400")
SHALLOW_SEEDS = tuple(range(8))
FOUND, MISSED = "found", "missed"
SHALLOWS = (FOUND, MISSED)
# The record fields a later test adds to a row; what was asked before it never reads them.
ANSWERS = ("defence", "hasten", "shallow", "shallow_moves", "shows")


@dataclass(frozen=True)
class Candidate:
    """A position reached in a recorded game: run seed, game number, ply and its prefix."""

    seed: int
    game: int
    ply: int
    pieces: int
    moves: tuple[int, ...]

    @property
    def id(self) -> str:
        return f"{self.seed}-{self.game}-{self.ply}"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "seed": self.seed, "game": self.game, "ply": self.ply,
                "pieces": self.pieces, "moves": list(self.moves)}


def pieces_in_play(state: GameState) -> int:
    """Every piece on the board and in either hand: what a small endgame is small in.

    A hand counts because a piece in hand is a piece in play; a drop game with six on the
    board and four in hand is not a six-piece ending.
    """
    return sum(1 for cell in state.board if cell) + sum(sum(hand) for hand in state.hands)


# ------------------------------------------------------------------ a position's name

NAME_LENGTH = 16
"""Hex characters of a position's name: 64 bits, against at most `CAP` positions a game."""


def content(state: GameState) -> tuple[Any, ...]:
    """What a position *is*, as the engine returned it (Q993c, `docs/archive/plan-proven.md` §5.3).

    The core's position key (`state.key`, the Zobrist key the core maintains and seeds from
    the rules' hash) covers the board, the side to move, the en passant file and every part
    of the state a repetition can return to. It leaves out, by design, what only an
    irreversible move changes, since repetition never compares across it: the hands, the
    check counts, a permanent colouring and the points. Those are here beside it, since two
    positions apart in any of them can end differently.

    Left out, as the key leaves them out: the counters (`ply`, `halfmove`, `turns_taken`,
    the repetition window), which say how the position was reached, and the unmoved mask,
    which says which pieces ever moved: a royal that stepped out and back makes it differ
    on one board, and two prefixes reaching one position must name it once. Python
    computes none of it; every field is the engine's.
    """
    return (
        state.key,
        tuple(tuple(hand) for hand in state.hands),
        tuple(state.checks),
        state.colour.hex(),
        tuple(float(points) for points in state.points),
    )


def name(state: GameState) -> str:
    """A position's name: a digest of its `content`, the same wherever and however it is
    reached, in any run, under any build that keeps the core's key. The site keys a
    person's marks and tries by it (`proven_marks.position`, `matches.puzzle`), so a
    re-proof that finds the same position again finds their marks with it, and one that
    finds another position can never inherit them."""
    held = content(state)
    text = json.dumps(
        [held[0], [list(hand) for hand in held[1]], list(held[2]), held[3], list(held[4])],
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:NAME_LENGTH]


def reached(engine: Engine, moves: Iterable[Any]) -> GameState | None:
    """The position a prefix reaches, every move put to the engine's `legal_moves` before
    it is applied, or None where one is not legal (or not a move at all)."""
    state = engine.initial_state()
    try:
        prefix = [int(move) for move in moves]
    except (TypeError, ValueError):
        return None
    for move in prefix:
        if move not in engine.legal_moves(state):
            return None
        state = engine.apply(state, move)
    return state


def replays(engine: Engine, row: dict[str, Any]) -> GameState | None:
    """The position a record's row proved, if the engine still reaches it as the row says;
    None where it does not.

    Every prefix move is put to `legal_moves` before it is applied (`reached`), the position
    must not have ended, the side to move must be the row's, and the engine's legal moves
    must be the row's root moves in the same order. The engine answers each question. The
    site's loader keeps a position only where this holds (`site/proven.py`), and a record
    with a row that fails it is owed again (`owes`).
    """
    try:
        side = int(row["side"])
        scores = [int(entry["move"]) for entry in row["scores"]]
        state = reached(engine, row["moves"])
    except (KeyError, TypeError, ValueError):
        return None
    if state is None:
        return None
    legal = engine.legal_moves(state)
    if engine.terminal(state, legal) is not None or state.side_to_move != side:
        return None
    return state if list(legal) == scores else None


def holding(
    engine: Engine, rows: Iterable[Any]
) -> tuple[list[tuple[int, str, dict[str, Any]]], int]:
    """A record's won rows that still hold, and how many rows do not (Q993i).

    Each kept row is `(order, name, row)`, in the record's order: a won row that `replays`,
    named by the state it reaches (`name`), the first to reach that name. A row the record
    proves *lost* for the side to move is passed over and counted as neither: a true proof,
    kept in the record, and never a position to offer. Every other row is counted as not
    holding: a won row that no longer replays or reaches a position already named, and a
    row that is no proof the writer writes. The site's loader offers from the kept rows and
    says how many did not hold (`site/proven.py`), and the upkeep re-proves a record with
    any (`owes`), so the two never disagree about a record.
    """
    kept: list[tuple[int, str, dict[str, Any]]] = []
    named: set[str] = set()
    failing = 0
    for order, row in enumerate(rows):
        if isinstance(row, dict) and row.get("result") == LOST:
            continue
        won = isinstance(row, dict) and row.get("result") == WON
        state = replays(engine, row) if won else None
        called = None if state is None else name(state)
        if called is None or called in named:
            failing += 1
            continue
        named.add(called)
        kept.append((order, called, row))
    return kept, failing


def renames(engine: Engine, data: dict[str, Any]) -> list[tuple[str, str]]:
    """`(old, new)` for every position of a record: its `seed-game-ply` id and the name
    its replayed prefix reaches (`name`), in the record's order. A prefix that no longer
    replays has no new name and is left out. What `0040_proven_names.sql` was written from,
    and what its test reads it against."""
    found = []
    for row in data.get("positions") or []:
        state = reached(engine, row.get("moves") or [])
        if state is not None:
            found.append((str(row["id"]), name(state)))
    return found


def candidates(
    engine: Engine,
    runs: Iterable[tuple[int, Iterable[GameRecord]]],
    max_pieces: int = MAX_PIECES,
    near: int | None = None,
    near_min: int = 1,
) -> Iterator[Candidate]:
    """Every distinct non-terminal position with at most `max_pieces` in play.

    `runs` is `(seed, records)` in a fixed order; the first time a position is reached
    is divided. Distinct is by the position's `content`, its key with what the key
    leaves out because repetition never compares across it (the hands, the check counts,
    a permanent colouring, the points), which is what the site names it by (`name`).

    With `near`, only the positions a proof can reach: in a game the record says was won,
    with its winner to move, at most `near` plies before the recorded end. The record's
    result is read, not judged; whether the position is won is still the probe's answer.
    `near_min` (with `near`) also skips those fewer than `near_min` plies before the end
    (`NEAR_MIN`, §3.5); the default, 1, skips nothing, since no position is 0 plies before.
    """
    if engine.simultaneous or engine.spy:
        raise ValueError("a game of rounds has no position to prove move by move")
    seen: set[tuple[Any, ...]] = set()
    for seed, records in runs:
        for number, record in enumerate(records):
            if near is not None and record.winner is None:
                continue
            length = len(record.moves)
            state = engine.initial_state()
            for ply, move in enumerate([*record.moves, None]):
                if near is None or (
                    near_min <= length - ply <= near and state.side_to_move == record.winner
                ):
                    pieces = pieces_in_play(state)
                    same = content(state)
                    if (pieces <= max_pieces and same not in seen
                            and engine.terminal(state) is None):
                        seen.add(same)
                        yield Candidate(seed, number, ply, pieces, tuple(record.moves[:ply]))
                if move is None:
                    break
                state = engine.apply(state, move)


def sample(found: list[Candidate], cap: int = CAP) -> list[Candidate]:
    """At most `cap` of the candidates, whole games at a time, evenly by their order.

    Every `k`-th game that has any, for the smallest `k` that keeps the total within
    `cap`: the positions spread over many games, each game keeps every depth it had, and
    the choice is fixed by the list alone.
    """
    games: dict[tuple[int, int], list[Candidate]] = {}
    for candidate in found:
        games.setdefault((candidate.seed, candidate.game), []).append(candidate)
    groups = list(games.values())
    stride = 1
    while sum(len(group) for group in groups[::stride]) > cap:
        stride += 1
    return [candidate for group in groups[::stride] for candidate in group][:cap]


def proof_of(score: float) -> str:
    """What one root score proves for the side to move."""
    if score >= PROVEN:
        return WON
    if score <= -PROVEN:
        return LOST
    return OPEN


def ends_in(score: float, ply: int) -> int:
    """Plies from the position to the ending the proof reached, for a proven score."""
    return int(round(WIN_SCORE - abs(score))) - ply


def prove(
    engine: Engine, candidate: Candidate, nodes: int = NODES, max_depth: int = MAX_DEPTH
) -> dict[str, Any]:
    """The candidate's prefix replayed, and `verdict` of the position it reaches."""
    state = engine.play(engine.initial_state(), candidate.moves)
    return {**candidate.as_dict(), **verdict(engine, state, nodes, max_depth)}


def verdict(
    engine: Engine, state: GameState, nodes: int = NODES, max_depth: int = MAX_DEPTH
) -> dict[str, Any]:
    """Ask the core for every root move's score in this position and read them back.

    A position with one legal move is not asked: the probe does not score a forced move,
    and a position whose only move is forced is no question to set. It is returned as
    `forced` and counted, never offered.
    """
    legal = engine.legal_moves(state)
    row: dict[str, Any] = {"side": state.side_to_move}
    if len(legal) < 2:
        row["result"] = "forced"
        return row
    scores = engine.root_scores(
        state, eval_values(engine.spec), nodes, max_depth, GOAL_PIECES, IMMUNE_PIECES
    )
    moves = []
    for move, score in zip(legal, scores, strict=True):
        proof = proof_of(score)
        entry: dict[str, Any] = {
            "move": move, "text": engine.describe_move(state, move),
            "score": score, "proof": proof,
        }
        if proof != OPEN:
            entry["ends_in"] = ends_in(score, state.ply)
        moves.append(entry)
    best = max(scores)
    winning = sum(1 for entry in moves if entry["proof"] == WON)
    if proof_of(best) == WON:
        row["result"] = WON
    elif all(entry["proof"] == LOST for entry in moves):
        row["result"] = LOST
    else:
        row["result"] = OPEN
    if row["result"] != OPEN:
        row["ends_in"] = ends_in(best, state.ply)
    row["winning"] = winning
    row["single"] = row["result"] == WON and winning == 1
    row["others_lost"] = row["single"] and all(
        entry["proof"] == LOST for entry in moves if entry["proof"] != WON
    )
    row["scores"] = moves
    return row


def counts(
    rows: Iterable[dict[str, Any]], defence: bool = False, hasten: bool = False,
    shallow: bool = False,
) -> dict[str, int]:
    """The record's tally: what was asked, what was proved, how many set one move, and,
    where T3 was asked (`defence`), its answers and how many positions are too easy to
    offer under T1, T2, T3 and, where T4 was asked (`hasten`) or T5 (`shallow`), their
    answers too."""
    tally = {"candidates": 0, "forced": 0, "won": 0, "lost": 0, "open": 0,
             "single": 0, "others_lost": 0}
    defences = dict.fromkeys(DEFENCES, 0)
    hastens = dict.fromkeys(DEFENCES, 0)
    shallows = dict.fromkeys(SHALLOWS, 0)
    easy = 0
    for row in rows:
        tally["candidates"] += 1
        tally[row["result"]] += 1
        tally["single"] += bool(row.get("single"))
        tally["others_lost"] += bool(row.get("others_lost"))
        if row.get("defence") in defences:
            defences[row["defence"]] += 1
        if row.get("hasten") in hastens:
            hastens[row["hasten"]] += 1
        if row.get("shallow") in shallows:
            shallows[row["shallow"]] += 1
        easy += bool(trivial(row))
    if defence:
        tally.update({f"defence_{answer}": count for answer, count in defences.items()})
    if hasten:
        tally.update({f"hasten_{answer}": count for answer, count in hastens.items()})
    if shallow:
        tally.update({f"shallow_{answer}": count for answer, count in shallows.items()})
    if defence or hasten or shallow:
        tally["trivial"] = easy
    return tally


# ------------------------------------------------------------------- too easy (§3.1)


def winning_moves(row: dict[str, Any]) -> int:
    """W: the root moves the record proves won. An *open* move is not won."""
    return sum(1 for entry in row.get("scores") or [] if entry.get("proof") == WON)


def trivial(row: dict[str, Any]) -> tuple[str, ...]:
    """The parts of the pre-registered test (`docs/archive/plan-proven.md` §3.1) a won position fails.

    Empty where the position is kept. Read off the record alone, deciding nothing: `n` is
    the record's `ends_in`, `L` its number of root moves, `W` those proved won.

    - **T1**, over at once with a choice: n = 1 and W >= 2;
    - **T2**, a wasted move still wins: W >= L / 2;
    - **T3**, the defence has no say: the record's stored `defence` answer is `none`
      (`defence` asks the core for it; a `FORMAT` 1 record has none to read);
    - **T4**, the defence can only hasten it (§3.6): the stored `hasten` answer is `none`
      (`hasten` asks the core; a `FORMAT` 3 or 4 record has it);
    - **T5**, seen at a glance (§3.7): the stored `shallow` answer is `found` (`shallow`
      asks the core; only a `FORMAT` 4 record has it).

    A position that is not a proved win is not offered at all, so it fails nothing here.
    """
    if row.get("result") != WON:
        return ()
    legal = len(row.get("scores") or [])
    won = winning_moves(row)
    parts = []
    if int(row.get("ends_in") or 0) == 1 and won >= 2:
        parts.append(T1)
    if legal and 2 * won >= legal:
        parts.append(T2)
    if row.get("defence") == NONE:
        parts.append(T3)
    if row.get("hasten") == NONE:
        parts.append(T4)
    if row.get("shallow") == FOUND:
        parts.append(T5)
    return tuple(parts)


def key_move(row: dict[str, Any]) -> dict[str, Any] | None:
    """The attacker's shortest winning move: among the root moves proved won, those with
    the least `ends_in`, and of those the first in the engine's order (the record's)."""
    won = [entry for entry in row.get("scores") or [] if entry.get("proof") == WON]
    if not won:
        return None
    shortest = min(int(entry["ends_in"]) for entry in won)
    return next(entry for entry in won if int(entry["ends_in"]) == shortest)


def asks_defence(row: dict[str, Any]) -> bool:
    """Whether T3 and T4 are put to this position: a proved win with two or more winning
    moves that T1 and T2 keep (`docs/archive/plan-proven.md` §3.3: those already left out are not
    asked). T4 is asked of the same positions as T3 (§3.6), whatever T3 answered."""
    return row.get("result") == WON and winning_moves(row) >= 2 and not trivial(_unasked(row))


def _unasked(row: dict[str, Any]) -> dict[str, Any]:
    """The row as the proofs left it, before any engine-asked test's answer."""
    return {key: value for key, value in row.items() if key not in ANSWERS}


def asks_shallow(row: dict[str, Any]) -> bool:
    """Whether T5 is put to this position (`docs/archive/plan-proven.md` §3.7): a proved win that T1 to
    T4 keep, read with T3's and T4's stored answers and without T5's own."""
    return row.get("result") == WON and not trivial(
        {key: value for key, value in row.items() if key not in ("shallow", "shallow_moves")}
    )


def defence(
    engine: Engine,
    row: dict[str, Any],
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> str | None:
    """T3 (`docs/archive/plan-proven.md` §3.1): does anything the defender does change the count?

    None where T3 does not apply (not a proved win, or fewer than two winning moves).
    Otherwise the position is reached (see `state`), the attacker's shortest winning move
    (`key_move`) is applied, and for **every** legal reply the core is asked
    (`Engine.root_scores` at `nodes`, the analysis evaluator and the given weights) about
    the position it leaves. A reply has no say only when the core proves that position won
    for the attacker with a shortest win of exactly n - 2 plies (or the game has ended
    there, in the attacker's win, with n = 2). Answers: `none` (no reply had a say),
    `some` (a reply was proved to change it: a longer or shorter win, or no win), or
    `unsettled` (none was proved to, but the probe could not settle at least one, which
    counts as a say). The first reply proved to have a say decides the answer, so the
    rest are not asked; the answer is the same either way.

    `state` is the position itself where the caller holds it; otherwise the record's
    prefix is replayed. Rule 3: every legal move, outcome and score is the core's; this
    compares the numbers.
    """
    return _says(engine, row, nodes, max_depth, goal, immune, state, faster=False)


def hasten(
    engine: Engine,
    row: dict[str, Any],
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> str | None:
    """T4 (`docs/archive/plan-proven.md` §3.6): can the defender do anything but hasten the loss?

    T3 exactly as `defence` asks it, with one difference: a reply has no say when the core
    proves the position it leaves won for the attacker with a shortest win of **at most**
    n - 2 plies, not exactly n - 2 (or the game has ended there in the attacker's win). So
    a reply that walks into a quicker loss has no say; one proved to put the win off, to
    escape it, or to lose it is a say, and one the probe cannot settle counts as one.
    The answers are T3's three words: `none` is that every reply keeps pace or loses
    faster. Asked of the same positions as T3, and on its own pass, because T3 stops at
    the first reply that changes the count and so cannot answer this.
    """
    return _says(engine, row, nodes, max_depth, goal, immune, state, faster=True)


def _says(
    engine: Engine,
    row: dict[str, Any],
    nodes: int,
    max_depth: int,
    goal: float,
    immune: float,
    state: GameState | None,
    faster: bool,
) -> str | None:
    """T3 (`faster` false: exactly n - 2) or T4 (`faster`: at most n - 2), for `defence`
    and `hasten`. The core answers every move, outcome and score; this compares.

    A turn of several plies (a duck's second half, `rules.turn` of two moves or more) is
    read whole, as §3.1 means a move and a reply: where the key move leaves the attacker
    still to move, the attacker's turn goes on by its shortest proved win at `nodes` (the
    first in the engine's order, as `key_move` chose the key), and a reply is every way
    the defender can play out its whole turn, each a sequence of plies until the attacker
    is to move again or the game ends. "n - 2" is n less the plies the move and the reply
    took, which is two where every turn is one ply. A continuation the probe cannot prove
    leaves the answer `unsettled`, a say, as an unsettled reply does.
    """
    if row.get("result") != WON or winning_moves(row) < 2:
        return None
    key = key_move(row)
    assert key is not None
    n = int(row["ends_in"])
    attacker = int(row["side"])
    if state is None:
        state = engine.play(engine.initial_state(), [int(move) for move in row["moves"]])
    if state.side_to_move != attacker:
        raise ValueError(f"{row.get('id')}: the record's side is not the side to move")
    values = eval_values(engine.spec)
    root = state.ply
    after = engine.apply(state, int(key["move"]))
    while True:
        replies = engine.legal_moves(after)
        ended = engine.terminal(after, replies)
        if ended is not None:
            # The attacker's turn ends the game: there is no reply to make.
            return NONE if ended.winner == attacker else SOME
        if after.side_to_move != attacker:
            break
        # The attacker's turn has more plies: its shortest proved win goes on.
        scores = engine.root_scores(after, values, nodes, max_depth, goal, immune)
        won = [(ends_in(score, after.ply), index) for index, score in enumerate(scores)
               if proof_of(score) == WON]
        if not won:
            return UNSETTLED
        after = engine.apply(after, replies[min(won)[1]])

    def no_say(there: GameState, plies: int) -> bool:
        want = n - (there.ply - root)
        return plies <= want if faster else plies == want

    said = NONE
    for there, outcome in _turns(engine, after, attacker):
        if outcome is not None:
            answer = NONE if outcome.winner == attacker and no_say(there, 0) else SOME
        else:
            best = max(engine.root_scores(there, values, nodes, max_depth, goal, immune))
            proof = proof_of(best)
            if proof == WON:
                answer = NONE if no_say(there, ends_in(best, there.ply)) else SOME
            elif proof == LOST:
                answer = SOME
            else:
                answer = UNSETTLED
        if answer == SOME:
            return SOME
        if answer == UNSETTLED:
            said = UNSETTLED
    return said


def _turns(engine: Engine, position: GameState, attacker: int) -> Iterator[tuple[GameState, Any]]:
    """Every whole turn the defender can play from `position`, depth first in the engine's
    order: each the position it leaves (the attacker to move, or the game over) and the
    core's outcome there (None while the game goes on). The core answers every ply."""
    for move in engine.legal_moves(position):
        there = engine.apply(position, move)
        outcome = engine.terminal(there, engine.legal_moves(there))
        if outcome is None and there.side_to_move != attacker:
            yield from _turns(engine, there, attacker)
        else:
            yield there, outcome


def shallow(
    engine: Engine,
    row: dict[str, Any],
    agents: tuple[str, ...] = SHALLOW_AGENTS,
    seeds: tuple[int, ...] = SHALLOW_SEEDS,
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> tuple[str, dict[str, list[int]]] | None:
    """T5 (`docs/archive/plan-proven.md` §3.7): do the core's quick players play a move that wins?

    None where the position is not a proved win. Otherwise each of `agents` is asked from
    each of `seeds` what it plays here (`choose_move`: the core's own player, its seat
    stream fixed by the seed), and each move chosen is read against the record: proved
    `won` wins, proved `lost` does not, and an `open` one is asked again of the position
    it leaves (`Engine.terminal`, else `Engine.root_scores` at `nodes` with the given
    weights), winning only where the core proves that position won for the mover. The
    answer is `found` when any choice wins and `missed` when none does, with the moves
    each agent played in seed order. The first choice proved to win decides it, so the
    rest are not asked again; the answer is the same either way.

    `state` is the position itself where the caller holds it; otherwise the record's
    prefix is replayed. Rule 3: every move, outcome and score is the core's; this compares.
    """
    if row.get("result") != WON:
        return None
    mover = int(row["side"])
    if state is None:
        state = engine.play(engine.initial_state(), [int(move) for move in row["moves"]])
    if state.side_to_move != mover:
        raise ValueError(f"{row.get('id')}: the record's side is not the side to move")
    proofs = {int(entry["move"]): entry["proof"] for entry in row.get("scores") or []}
    played: dict[str, list[int]] = {}
    for agent in agents:
        moves = []
        for seed in seeds:
            move = choose_move(engine, agent, state, seed=seed)
            if move is None or move not in proofs:
                raise ValueError(f"{row.get('id')}: {agent} played no root move of the record")
            moves.append(move)
        played[agent] = moves
    values = eval_values(engine.spec)

    def wins(move: int) -> bool:
        if proofs[move] != OPEN:
            return proofs[move] == WON
        after = engine.apply(state, move)
        legal = engine.legal_moves(after)
        outcome = engine.terminal(after, legal)
        if outcome is not None:
            return outcome.winner == mover
        best = proof_of(max(engine.root_scores(after, values, nodes, max_depth, goal, immune)))
        return best == (WON if after.side_to_move == mover else LOST)

    chosen = list(dict.fromkeys(move for moves in played.values() for move in moves))
    answer = FOUND if any(wins(move) for move in chosen) else MISSED
    return answer, played


# --------------------------------------------- what its game has and chess lacks (Q1084)

REFERENCE = Path("rulesets") / "chess.json"
"""The game every other is compared with (`library.REFERENCE_FILE`, the calibration
baseline), read as data: nothing here names one of its pieces."""
SHOWS_VERSION = 1
"""The sift's own version: a change to what a position must show bumps it, and every record
sifted before is owed a re-filter (`FILTER`), which asks no proof again."""
NOTHING: list[str] = []
"""A sifted row's `shows` where its line showed nothing its game has and chess lacks."""
CHESS_KINDS = frozenset({MOVE_NORMAL, MOVE_EN_PASSANT, MOVE_CASTLE, MOVE_PROMOTION})
"""The move words chess plays. Any other (a drop, a shove, a hire, the duck's half, a step
through a portal) is a rule chess lacks, played."""
PIECE_RULES = ("captures", "beats", "spy", "speed", "rest", "bounty", "price", "control",
               "takes_royal", "sight", "pushes", "weight", "strength", "facing", "region")
"""Keys of a piece that are rules of the piece (`relate`'s piece identity and key): whom it
takes, its speed, rest, bounty, price, control, line of sight, push, facing, confinement.
Chess's pieces state none, so a piece stating one is its game's own."""
_OFF = (None, False, 0, "stands", "straight")
WIN_REASONS = {"checkmate": "checkmate", "royal_capture": "royal_captured",
               "annihilation": "annihilated"}
"""The ending's reason (`Outcome.reason`, the core's word) for each `rules.win`: how the
reference game ends a game won, which a line won otherwise shows is not its."""
BOARD_RULES = {"board.walls": "obstacles", "board.wrap": "wrap"}
"""The board's rules `library.rules_beyond` names, by the rule key a twin takes out."""


@dataclass(frozen=True, eq=False)
class Distinction:
    """What a game has and the reference game lacks, as a position can show it (Q1084).

    `same`: the game is the reference in mechanics (one normal form), so every position
    shows it. `pieces`: the compiled types (by index) chess lacks, each with why: one whose
    move map on an empty board is no chess piece's (`library.maps_against`, the engine's
    answer the game page shows), one that shares a chess piece's name or letter and moves
    otherwise, one royal where the chess piece it moves like is not (or not royal where it
    is), or one stating a rule of the piece (`PIECE_RULES`, a shove, a sweep, a turning
    path). `twins`: for each rule the game states otherwise than chess (its rules beyond
    chess's, `library.rules_beyond`, and every rule key whose value differs), the game
    compiled with that rule put back to chess's (or taken out where chess has none), where
    it compiles to the same position encoding; the line is read against each. `unread`
    names the rules that did not, which only the move words and the pieces can show.
    `wins` is how chess ends a game won (its `rules.win`)."""

    same: bool
    pieces: dict[int, str]
    twins: tuple[tuple[str, Engine], ...]
    unread: tuple[str, ...]
    wins: str


def reference_spec(root: Path | str = ".") -> dict[str, Any]:
    """The reference game's spec under ``root``, or this checkout's where ``root`` has none
    (a test's library laid out in a temporary directory)."""
    from chess_variant_sandbox.spec.canonical import load_spec

    where = Path(root) / REFERENCE
    if not where.is_file():
        where = Path(__file__).resolve().parents[1] / REFERENCE
    spec = load_spec(where)
    if not isinstance(spec, dict):
        raise ValueError(f"{REFERENCE} is not a ruleset")
    return spec


def distinction(spec: dict[str, Any], reference: dict[str, Any]) -> Distinction:
    """`Distinction` of ``spec`` against ``reference``, memoised on the two specs."""
    return _distinction(json.dumps(spec, sort_keys=True), json.dumps(reference, sort_keys=True))


@lru_cache(maxsize=64)
def _distinction(blob: str, reference_blob: str) -> Distinction:
    from chess_variant_sandbox.spec import relate
    from chess_variant_sandbox.spec.spelling import current_spelling
    from chess_variant_sandbox.web.library import rules_beyond

    spec, reference = json.loads(blob), json.loads(reference_blob)
    win = str((reference.get("rules") or {}).get("win", "checkmate"))
    wins = WIN_REASONS.get(win, win)
    form, theirs = relate.normal_form(spec), relate.normal_form(reference)
    if form.key == theirs.key:
        return Distinction(True, {}, (), (), wins)
    engine = Engine(spec)

    # The rules: the keys `rules_beyond` names, every rule written otherwise than chess
    # writes it (a clock aside: no won line runs into one), the board's own, and the rules
    # a piece states (`piece.captures`, `piece.sight`, ...), each read by a twin.
    beyond = rules_beyond(spec, reference)
    keys = {key for key in beyond if not key.startswith(("piece.", "board."))}
    keys |= {key for key, value in form.rules.items()
             if key not in relate._CLOCK_KEYS and theirs.rules.get(key) != value}
    keys |= {BOARD_RULES[key] for key in beyond if key in BOARD_RULES}
    keys |= {key for key in beyond if key.startswith("piece.") and key[6:] in PIECE_RULES}
    written, chess_written = current_spelling(spec), current_spelling(reference)
    twins: list[tuple[str, Engine]] = []
    unread: list[str] = []
    for key in sorted(keys):
        twin = json.loads(json.dumps(written))
        if key.startswith("piece."):
            for piece in twin.get("pieces") or []:
                piece.pop(key[6:], None)
        else:
            rules = twin.setdefault("rules", {})
            if key in (chess_written.get("rules") or {}):
                rules[key] = json.loads(json.dumps(chess_written["rules"][key]))
            else:
                rules.pop(key, None)
        try:
            other = Engine(twin)
        except Exception:  # noqa: BLE001 - a twin that does not compile is a rule unread
            unread.append(key)
            continue
        if _same_encoding(engine, other):
            twins.append((key, other))
        else:
            unread.append(key)
    read = {key[6:] for key, _twin in twins if key.startswith("piece.")}

    # The pieces: by how each moves once out (`piece_out`), against chess's of its name or
    # letter, then against every chess piece; and a rule of the piece no twin reads.
    compiled = engine.spec
    usual = {piece_out(theirs, pid) for pid in theirs.pieces}
    namesakes = {str(word).lower(): pid for pid, piece in theirs.pieces.items()
                 for word in (pid, piece.get("name", pid))}
    pieces: dict[int, str] = {}
    for index, pid in enumerate(compiled.piece_ids):
        piece = form.pieces.get(pid)
        if piece is None:
            continue  # an internal type (a portal end): nobody's, read off the board
        mine = piece_out(form, pid)
        namesake = namesakes.get(str(pid).lower()) or namesakes.get(
            str(piece.get("name", pid)).lower())
        own = [key for key in PIECE_RULES if key not in read
               and (piece.get(key) is not None if key == "captures"
                    else piece.get(key) not in _OFF)]
        if namesake is not None and piece_out(theirs, namesake) != mine:
            pieces[index] = (
                f"moves unlike chess's {theirs.pieces[namesake].get('name', namesake)}")
        elif mine not in usual:
            pieces[index] = "moves like no piece of chess"
        elif own:
            pieces[index] = f"has its own {own[0]}"
    return Distinction(False, pieces, tuple(twins), tuple(unread), wins)


def piece_out(form: Any, pid: str) -> tuple[Any, ...]:
    """What a piece is once out of where it starts (Q1084): `relate.piece_identity` with
    the moves a starting square conditions left out (a pawn's first double step), and a
    shove left to the push rule's own reading (the move word, `CHESS_KINDS`). Two pieces
    alike in it move alike on the board of a proven ending, wherever they started: the
    first step is the opening's, and a small board without it is its setup's."""
    from chess_variant_sandbox.spec import relate

    unconditioned = relate._frozen({})
    movement, *rest = relate.piece_identity(form, pid)
    merged: dict[Any, frozenset[Any]] = {}
    for attrs, dirs in movement:
        if attrs[4] != unconditioned:
            continue
        plain = tuple(part for part in attrs
                      if not (isinstance(part, tuple) and part and part[0] == "push"))
        merged[plain] = merged.get(plain, frozenset()) | dirs
    return (frozenset(merged.items()), *rest)

def _same_encoding(engine: Engine, other: Engine) -> bool:
    """Whether a position of `engine` is a position of `other`: the same squares, types,
    facings and owners, and an opening state of the same shape field by field."""
    one, two = engine.spec, other.spec
    if (one.n_squares, one.n_types, one.n_facings, tuple(one.piece_ids)) != (
            two.n_squares, two.n_types, two.n_facings, tuple(two.piece_ids)):
        return False
    first, second = engine.initial_state(), other.initial_state()
    for field_ in dataclasses.fields(first):
        a, b = getattr(first, field_.name), getattr(second, field_.name)
        if isinstance(a, (bytes, tuple)) and isinstance(b, (bytes, tuple)):
            if len(a) != len(b):
                return False
        elif type(a) is not type(b):
            return False
    return first.board == second.board


def _what(state: GameState) -> tuple[Any, ...]:
    """What a move leaves that a rule acting on it can change: the board, the hands, the
    purses and the points (a blast, a shot, a demotion, a bounty paid)."""
    return (state.board, state.hands, state.purse, state.points)


def _asked(call: Any, *args: Any) -> Any:
    """A twin's answer, or None where the core refuses a position of the other game."""
    try:
        return call(*args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 - a core panic is a twin that cannot answer
        return None


def asks_shows(row: dict[str, Any]) -> bool:
    """Whether the sift is put to this position: a proved win T1 to T5 keep, read with
    their stored answers. A row left out as too easy is never offered, so it is not asked."""
    return row.get("result") == WON and not trivial(row)


def shows(
    engine: Engine,
    row: dict[str, Any],
    seen: Distinction,
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> list[str]:
    """What the position and its proven line show that the game has and chess lacks (Q1084).

    Empty (`NOTHING`) where nothing. The line starts with the key move (`key_move`) and goes
    on, each ply, by the side to move's best move as the core scores it at `nodes` (the
    attacker's quickest proved win, the defender's longest proved loss), until the game ends,
    the proof no longer holds, or `ends_in` plies are played. At each position on it: a piece
    chess lacks (`Distinction.pieces`), a piece nobody owns (a neutral, a wild piece, the
    duck, a portal's end) or a piece in hand. At each move: a word chess has no kind for
    (`CHESS_KINDS`, or a move of the opponent's piece), and for each twin, a move it does not
    allow (the rule gave it), a move it allows that the game does not (the rule took it, as a
    wall or a ban on check does), another board after the move (the rule acted on it: a blast,
    a shot), or another ending (a goal reached, a stalemate won). And a line won otherwise
    than chess wins. The first thing found is the answer. Rule 3: every move, board, ending
    and score is the core's, or a twin's; this compares them.
    """
    if seen.same:
        return ["the game is chess"]
    if row.get("result") != WON:
        return []
    key = key_move(row)
    if key is None:
        return []
    compiled = engine.spec
    if state is None:
        state = engine.play(engine.initial_state(), [int(move) for move in row["moves"]])
    attacker = int(row["side"])
    values = eval_values(compiled)

    def standing(position: GameState) -> list[str]:
        for value in position.board:
            if not value:
                continue
            owner, kind = compiled.owner_of(value), compiled.type_of(value)
            if owner > 1:
                return [f"{compiled.piece_names[kind]}, nobody's piece"]
            if kind in seen.pieces:
                return [f"{compiled.piece_names[kind]}, which {seen.pieces[kind]}"]
        if any(sum(hand) for hand in position.hands):
            return ["a piece in hand"]
        return []

    position, move = state, int(key["move"])
    found = standing(position)
    for _ply in range(max(1, int(row.get("ends_in") or 1))):
        if found:
            return found
        legal = engine.legal_moves(position)
        kind = move_type(move)
        if move & MOVE_FOREIGN:
            return ["a move of a piece the mover does not own"]
        if kind not in CHESS_KINDS:
            return [f"a {MOVE_TYPE_NAMES.get(kind, 'new')} move"]
        after = engine.apply(position, move)
        ended = engine.terminal(after)
        for rule, twin in seen.twins:
            allowed = _asked(twin.legal_moves, position)
            if allowed is None:
                continue
            if move not in allowed:
                return [f"the rule {rule}, played"]
            if set(allowed) - set(legal):
                return [f"the rule {rule}, which barred a move"]
            there = _asked(twin.apply, position, move)
            if there is not None and _what(there) != _what(after):
                return [f"the rule {rule}, acting on the move"]
            their_end = _asked(twin.terminal, after)
            if (None if ended is None else (ended.winner, ended.reason)) != (
                    None if their_end is None else (their_end.winner, their_end.reason)):
                return [f"the rule {rule}, ending the game"]
        if ended is not None:
            if ended.winner is not None and ended.reason != seen.wins:
                return [f"a win by {ended.reason}"]
            return standing(after)
        found = standing(after)
        if found:
            return found
        replies = engine.legal_moves(after)
        if len(replies) == 1:
            # A forced move: the probe scores none, and the line plays it.
            position, move = after, replies[0]
            continue
        scores = engine.root_scores(after, values, nodes, max_depth, goal, immune)
        best = max(range(len(scores)), key=lambda at: (scores[at], -at))
        if proof_of(scores[best]) != (WON if after.side_to_move == attacker else LOST):
            return []
        position, move = after, replies[best]
    return found


def record(
    engine: Engine,
    name: str,
    seeds: Iterable[int],
    rows: list[dict[str, Any]],
    built: str,
    nodes: int = NODES,
    max_pieces: int = MAX_PIECES,
    made: datetime.date | None = None,
    distinct: int | None = None,
    near: int | None = NEAR,
    cap: int = CAP,
    defence_nodes: int | None = None,
    defence_built: str | None = None,
    defence_made: datetime.date | None = None,
    hasten_nodes: int | None = None,
    hasten_built: str | None = None,
    hasten_made: datetime.date | None = None,
    shallow_nodes: int | None = None,
    shallow_built: str | None = None,
    shallow_made: datetime.date | None = None,
    shallow_agents: tuple[str, ...] = SHALLOW_AGENTS,
    shallow_seeds: tuple[int, ...] = SHALLOW_SEEDS,
    near_min: int | None = None,
    shows_nodes: int | None = None,
    shows_built: str | None = None,
    shows_made: datetime.date | None = None,
) -> dict[str, Any]:
    """The record the site loads: the basis, the tally, and the proved positions only.

    Open and forced positions are counted and left out, since nothing can be offered from
    them; lost ones stay, as a proof the side to move cannot escape is still a proof.
    `distinct` is how many positions the runs held before `sample` took `cap` of them.

    Rows that carry T3's answer (`defence`) keep it, and `defence_nodes` names the budget
    it was asked at, with the core and the day (`defence_built`, `defence_made`): the
    basis of every `none` the site leaves out; T4's answers (`hasten`) likewise with
    `hasten_nodes`, `hasten_built` and `hasten_made`; T5's (`shallow`, `shallow_moves`) with
    `shallow_nodes` (the second ask's budget), the players and seeds asked, `shallow_built`
    and `shallow_made`. Positions too easy to offer stay in
    the record; the reader leaves them out (`trivial`), so the record holds every proof.
    The basis names the probe's version (`PROOF_VERSION`) and, where the sample skipped the
    last plies of each game (`NEAR_MIN`), `near_min`.

    Where the sift was asked (`shows_nodes`, Q1084), a won row whose line showed nothing its
    game has and chess lacks (`shows` empty) is set aside (`sifted`): out of `positions`,
    kept whole in `set_aside`, and the basis names the sift's version, budget, core and day.
    """
    ordered = sorted(rows, key=lambda row: (row["seed"], row["game"], row["ply"]))
    basis: dict[str, Any] = {
        "probe": "Engine.root_scores",
        "proof_version": PROOF_VERSION,
        "nodes": nodes,
        "max_depth": MAX_DEPTH,
        "proven_at": PROVEN,
        "goal": GOAL_PIECES,
        "immune": IMMUNE_PIECES,
        "built": built,
        "runs": [f"{engine.spec.hash}-{seed}" for seed in seeds],
        "max_pieces": max_pieces,
        "distinct": len(ordered) if distinct is None else distinct,
        "near": near,
        "cap": cap,
    }
    if near_min is not None:
        basis["near_min"] = near_min
    if defence_nodes is not None:
        basis["defence_nodes"] = defence_nodes
        basis["defence_built"] = defence_built
        basis["defence_made"] = (defence_made or datetime.date.today()).isoformat()
    if hasten_nodes is not None:
        basis["hasten_nodes"] = hasten_nodes
        basis["hasten_built"] = hasten_built
        basis["hasten_made"] = (hasten_made or datetime.date.today()).isoformat()
    if shallow_nodes is not None:
        basis["shallow_agents"] = list(shallow_agents)
        basis["shallow_seeds"] = list(shallow_seeds)
        basis["shallow_nodes"] = shallow_nodes
        basis["shallow_built"] = shallow_built
        basis["shallow_made"] = (shallow_made or datetime.date.today()).isoformat()
    data = {
        "format": FORMAT,
        "ruleset_hash": engine.spec.hash,
        "name": name,
        "made": (made or datetime.date.today()).isoformat(),
        "basis": basis,
        "counts": counts(ordered, defence=defence_nodes is not None,
                         hasten=hasten_nodes is not None,
                         shallow=shallow_nodes is not None),
        "positions": [row for row in ordered if row["result"] in (WON, LOST)],
    }
    if shows_nodes is not None:
        sifted(data, shows_nodes, shows_built or built, shows_made)
    return data


def sifted(
    data: dict[str, Any], nodes: int, built: str, made: datetime.date | None = None,
) -> dict[str, Any]:
    """The record with every row the sift found showing nothing set aside (Q1084), in place.

    A row of `positions` whose `shows` is empty moves to `set_aside`, and one there whose
    `shows` is no longer empty comes back, each list in the record's order;
    `counts["set_aside"]` is how many are aside, and the basis is stamped with the sift's
    version and how it was asked. Moves rows; deletes none and asks nothing."""
    rows = [*(data.get("positions") or []), *(data.get("set_aside") or [])]
    rows.sort(key=lambda row: (row.get("seed", 0), row.get("game", 0), row.get("ply", 0)))
    data["positions"] = [row for row in rows if row.get("shows") != NOTHING]
    data["set_aside"] = [row for row in rows if row.get("shows") == NOTHING]
    data.setdefault("counts", {})["set_aside"] = len(data["set_aside"])
    data.setdefault("basis", {}).update({
        "shows_version": SHOWS_VERSION, "shows_nodes": nodes, "shows_built": built,
        "shows_made": (made or datetime.date.today()).isoformat(),
    })
    return data


def is_sifted(data: Any) -> bool:
    """Whether a record was sifted by this code's sift (`SHOWS_VERSION`) or a later one."""
    basis = data.get("basis") if isinstance(data, dict) else None
    version = basis.get("shows_version") if isinstance(basis, dict) else None
    return isinstance(version, int) and version >= SHOWS_VERSION


def write(path: Path, data: dict[str, Any]) -> None:
    """Whole or not at all: a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------- upkeep (Q993d, §5)

DIRECTORY = "puzzles"
"""Where the records sit under `runs/`, and their work under `work/<hash>/`."""
PROVE, REPROVE, FILTER = "prove", "re-prove", "re-filter"
NEEDS = (PROVE, REPROVE, FILTER)
"""What a game is owed: a record it has never had, then one it has that no longer holds,
then one that holds but was never sifted for what its game has and chess lacks (Q1084),
which `refilter` mends without proving anything again."""
STORED = "proofs:"
"""A game's record file, hashed under this prefix: what a refusal of its proofs is keyed
on, never to be mistaken for its `canonical.json` digest or its ladder's."""
PROVE_BATCH = 8
TEST_BATCH = 8
"""Positions a unit file holds: what a killed run loses at most, per stage."""
WORKERS = 8
"""Threads a game is proved on: a `heavy` lease's slots, less two (the slow tier's eight)."""
TESTS = ("defence", "hasten", "shallow", "shows")
"""The engine-asked tests in the order they are asked: T3, T4, then T5 read with both, then
the sift (Q1084) of what T1 to T5 keep."""


def record_path(runs_dir: Path | str, digest: str) -> Path:
    return Path(runs_dir) / DIRECTORY / f"{digest}.json"


def work_dir(runs_dir: Path | str, digest: str) -> Path:
    return Path(runs_dir) / DIRECTORY / "work" / digest


def stored_of(runs_dir: Path | str, digest: str) -> str:
    """The game's record file, hashed, or `proofs:absent`: what a refusal is keyed on."""
    try:
        body = record_path(runs_dir, digest).read_bytes()
    except OSError:
        return STORED + "absent"
    return STORED + hashlib.sha256(body).hexdigest()[:16]


@dataclass(frozen=True)
class Standing:
    """Where one Library or Baseline game's proven positions stand. `needs` is `""` when
    nothing is owed, and `reason` then says why (current, a game of rounds, no runs)."""

    hash: str
    name: str
    needs: str
    reason: str
    stored: str
    made: str = ""

    @property
    def owed(self) -> bool:
        return bool(self.needs)


def library(root: Path | str = ".") -> list[tuple[str, str, dict[str, Any]]]:
    """Every Library and Baseline game as `(hash, name, spec)`: `rulesets/` (the Baseline)
    then `archive/` (the Library), each hash once. An archived preview is left out, as
    the site leaves it out of every scored surface and `ladders_check.library` does."""
    from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
    from chess_variant_sandbox.web.library import preview_sentence

    found: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for directory, archived in (("rulesets", False), ("archive", True)):
        folder = Path(root) / directory
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            try:
                spec = load_spec(path)
            except (OSError, ValueError):
                continue
            if not isinstance(spec, dict):
                continue
            digest = spec_hash(spec)
            if digest in found or (archived and preview_sentence(spec) is not None):
                continue
            found[digest] = (digest, str(spec.get("name") or path.stem), spec)
    return list(found.values())


def owes(engine: Engine, data: Any) -> str | None:
    """Why a game's record is owed a re-proof, or None where it holds (§5.2 (ii), (iii)).

    A record that is not this game's or cannot be read; one in a format older than the
    code's (`FORMAT`); one proved by an older probe (`PROOF_VERSION`; a record without the
    field is version 1); or one with a proved position the engine no longer reaches as the
    record says (`replays`), or reaches twice, or a row that is no proof (`holding`, which
    the site's loader reads: a record is owed exactly where the loader drops a row). A row
    proved lost for the side to move holds (Q993i). A record from newer code is not owed: this
    checkout cannot tell it is wrong. The engine answers every replay; this counts.
    """
    if not isinstance(data, dict) or data.get("ruleset_hash") != engine.spec.hash:
        return "the record cannot be read as this game's"
    written = data.get("format")
    if not isinstance(written, int) or isinstance(written, bool):
        return "the record names no format"
    if written < FORMAT:
        return f"its record is format {written}, and the code writes {FORMAT}"
    basis = data.get("basis")
    version = basis.get("proof_version", 1) if isinstance(basis, dict) else 1
    if isinstance(version, int) and version < PROOF_VERSION:
        return f"it was proved by probe version {version}, and the code's is {PROOF_VERSION}"
    rows = [row for row in data.get("positions") or []
            if not (isinstance(row, dict) and row.get("result") == LOST)]
    _kept, dropped = holding(engine, rows)
    if dropped:
        return f"{dropped} of its {len(rows)} proved positions no longer replay"
    return None


def runs_missing(runs_dir: Path | str, digest: str, seeds: Iterable[int]) -> list[int]:
    """The library seeds whose run has no games on disk."""
    return [seed for seed in seeds
            if not (Path(runs_dir) / f"{digest}-{seed}" / "games.jsonl").is_file()]


def standing(
    digest: str, game: str, spec: dict[str, Any], runs_dir: Path | str,
    seeds: Iterable[int] | None = None,
) -> Standing:
    """One game's `Standing`: what it is owed, if anything, and why. Reads; proves nothing.

    Owed (§5.2) where the game alternates and its library runs are on disk, and it has
    (i) no record (`PROVE`) or (ii, iii) one `owes` finds wanting (`REPROVE`). A game of
    rounds is never owed (a round has no position to prove move by move), nor one whose
    runs are not all there yet: `measure` writes them, and then it is.
    """
    from chess_variant_sandbox.corpus import LIBRARY_SEEDS

    seeds = tuple(LIBRARY_SEEDS if seeds is None else seeds)
    stored = stored_of(runs_dir, digest)

    def stands(needs: str, reason: str, made: str = "") -> Standing:
        return Standing(digest, game, needs, reason, stored, made)

    engine = Engine(spec)
    if engine.simultaneous or engine.spy:
        return stands("", "a game of rounds: no position to prove move by move")
    path = record_path(runs_dir, digest)
    data: Any = None
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        made = str(data.get("made") or "") if isinstance(data, dict) else ""
        why = owes(engine, data)
        if why is None and not is_sifted(data):
            # Q1084: a record that holds but was never sifted is mended by `refilter`,
            # which proves nothing again, so it needs no runs on disk.
            return stands(FILTER, "its positions were never sifted for what the game has "
                                  "and chess lacks", made)
        if why is None:
            rows = [row for row in data.get("positions") or []
                    if isinstance(row, dict) and row.get("result") == WON]
            offered = sum(1 for row in rows if not trivial(row))
            aside = len(data.get("set_aside") or [])
            kept = f"current: {len(rows)} proved, {offered} offered" + (
                f", {aside} set aside as showing nothing of the game's own" if aside else "")
            return stands("", kept if rows or aside
                          else "current: its runs held no small ending to prove", made)
        needs, reason = REPROVE, why
    else:
        made = ""
        needs, reason = PROVE, "no record"
    missing = runs_missing(runs_dir, digest, seeds)
    if missing:
        return stands("", f"{reason}, and its library runs are not all on disk (seeds "
                          f"{', '.join(str(seed) for seed in missing)}): measure writes them",
                      made)
    return stands(needs, reason, made)


def survey(
    runs_dir: Path | str, root: Path | str = ".", seeds: Iterable[int] | None = None,
) -> list[Standing]:
    """Every Library and Baseline game's `Standing`, in the library's order. Reads only."""
    return [standing(digest, game, spec, runs_dir, seeds)
            for digest, game, spec in library(root)]


def retire(
    runs_dir: Path | str, root: Path | str = ".", today: datetime.date | None = None,
) -> list[Path]:
    """Move every delivered record of a game that is not in ``root``'s library out of what
    is offered, and return where each went. Moves; proves and deletes nothing.

    A game leaves the library when it is edited in place (its mechanics re-hashed, as
    Dohyo's walled corners re-hashed it in Q1074bd) or taken out of it. Its old runs are
    kept, and so is its record: beside its work as `work/<hash>/record-<day>-retired.json`,
    as a record a format replaced is kept, never over a file already there. The site reads
    only a library game's record, and `runs/puzzles/` holds only those (`tests/test_puzzles.py`).
    """
    today = today or datetime.date.today()
    folder = Path(runs_dir) / DIRECTORY
    if not folder.is_dir():
        return []
    games = {digest for digest, _name, _spec in library(root)}
    moved: list[Path] = []
    for path in sorted(folder.glob("*.json")):
        if path.stem in games:
            continue
        work = work_dir(runs_dir, path.stem)
        work.mkdir(parents=True, exist_ok=True)
        to = work / f"record-{today.isoformat()}-retired.json"
        to = _aside(to, today) if to.exists() else to
        path.replace(to)
        moved.append(to)
    return moved


def _file_digest(path: Path) -> str:
    """The first 16 hex digits of a file's SHA-256, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _aside(path: Path, today: datetime.date) -> Path:
    """A free name beside `path` for what is moved out of the way, dated, never over one."""
    for n in range(1, 1000):
        candidate = path.with_name(
            f"{path.stem}.before-{today.isoformat()}{'' if n == 1 else f'-{n}'}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free name beside {path}")


def _standing_units(
    expected: list[tuple[Path, list[str]]], built: str, nodes: int,
) -> dict[Path, dict[str, Any]]:
    """The unit files on disk that stand, by path.

    A unit stands only where it covers the positions it should (its rows' ids, in order)
    and was asked on this build at this budget; the rest are asked again, so a record never
    mixes two builds or budgets in one stage. A whole stage proved on another core no
    longer stands either (Q1082): the core decides every answer, and Q1078 changed a game's
    legal moves under work that was then read back as current. `make` moves work from
    another build aside before it reads a unit (`work_basis`); this is the check per unit.
    """
    loaded: dict[Path, dict[str, Any]] = {}
    for path, ids in expected:
        try:
            unit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(unit, dict) and [row.get("id") for row in unit.get("rows") or []] == ids:
            loaded[path] = unit
    return {path: unit for path, unit in loaded.items()
            if unit.get("built") == built and unit.get("nodes") == nodes}


WORK_WORDS = {"built": "core build", "tables": "compile of its rules"}
"""What `work_basis` keys, in the words `make` names a moved-aside work's reason in."""


def work_basis(spec: dict[str, Any]) -> dict[str, str]:
    """What decides a proof beside the sampling (Q1082), as a library run's basis has it
    (`measurement`): the core build (`built_fingerprint`, the half of a run's `engine` stamp
    the core owns) and what this game compiles to (`tables_fingerprint`, the per-ruleset
    half). The core answers every replay, proof and test from those two; the probe's own
    Python is `PROOF_VERSION`'s, read off the record by `owes`."""
    from chess_variant_sandbox.measurement import built_fingerprint, tables_fingerprint

    return {"built": built_fingerprint(), "tables": tables_fingerprint(spec)}


def _ask_units(
    units: list[tuple[Path, list[Any]]],
    standing: dict[Path, dict[str, Any]],
    ask: Any,
    unit_of: Any,
    workers: int,
    say: Any,
    label: str,
) -> None:
    """Ask every missing unit's positions on one pool, and write each unit as it is whole,
    in order: the slowest position of one unit never idles the other threads, and a killed
    run loses only the units not yet written. The answers are the same however the work is
    divided, since each position is asked alone."""
    from concurrent.futures import ThreadPoolExecutor

    missing = [(path, items) for path, items in units if path not in standing]
    if not missing:
        return
    with ThreadPoolExecutor(max(1, workers)) as pool:
        futures = [(path, items, [pool.submit(ask, item) for item in items])
                   for path, items in missing]
        for done, (path, items, asked) in enumerate(futures, start=1):
            answers = [future.result() for future in asked]
            write(path, unit_of(items, answers))
            say(f"  {label} {done} of {len(futures)} written")


def make(
    spec: dict[str, Any],
    runs_dir: Path | str,
    workers: int = WORKERS,
    say: Any = None,
    seeds: Iterable[int] | None = None,
    nodes: int = NODES,
    defence_nodes: int = DEFENCE_NODES,
    today: datetime.date | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one game's record whole (§5.2): count, sample, prove, T3, T4, T5, the sift,
    record.

    The work is R9a's driver's, moved into the package: the candidates of the game's four
    library runs (`candidates`, at most `MAX_PIECES` in play, `NEAR_MIN` to `NEAR` plies
    before a won game's end, the winner to move), whole games sampled to `CAP` (`sample`),
    each proved at `nodes` (`prove`), T3 and T4 asked of the proved wins T1 and T2 keep
    with two or more winning moves, T5 of every proved win T1 to T4 keep, and the record
    written (`record`) into `runs_dir`'s `puzzles/`, replacing any there, which is kept
    beside the work as `record-<day>.json`. Each stage writes unit files under
    `work/<hash>/` and a run picks up from them (`_standing_units`); a work directory
    sampled another way, or made on another core build or compile of the rules
    (`work_basis`, Q1082), is moved aside first, never written over, and done again. A game whose runs hold
    no small ending gets a record with no position. Rule 3: every candidate, proof and
    choice is the core's; this holds positions, asks, and counts.

    Returns what it did: the counts, how many positions are offered, and the seconds each
    stage took in this call (a stage picked up whole from disk took none).
    """
    import time

    from chess_variant_sandbox.corpus import LIBRARY_SEEDS
    from chess_variant_sandbox.tournament.record import read_records

    def note(line: str) -> None:
        if say is not None:
            say(line)

    today = today or datetime.date.today()
    seeds = tuple(LIBRARY_SEEDS if seeds is None else seeds)
    engine = Engine(spec)
    digest = engine.spec.hash
    game = str(spec.get("name") or digest)
    if engine.simultaneous or engine.spy:
        raise ValueError("a game of rounds has no position to prove move by move")
    runs_dir = Path(runs_dir)
    missing = runs_missing(runs_dir, digest, seeds)
    if missing:
        raise FileNotFoundError(f"{game}: no library run on disk for seeds {missing}")
    basis = work_basis(spec)
    built = basis["built"]
    seconds: dict[str, float] = {}

    # The count and the sample, keyed by how it was taken.
    work = work_dir(runs_dir, digest)
    # The runs by their bytes as well as their names (Q1032): a re-measured game keeps its
    # run directories' names, so work keyed on the names alone picked up the old runs'
    # sample and re-proved the record it was meant to replace. And by what answers every
    # question asked of them (Q1082): Q1078 moved the push rule's legal moves under work
    # keyed on the sampling alone, and the old core's proofs were written again as new.
    sampling = {"runs": [f"{digest}-{seed}" for seed in seeds], "max_pieces": MAX_PIECES,
                "near": NEAR, "near_min": NEAR_MIN, "cap": CAP,
                "games": [_file_digest(runs_dir / f"{digest}-{seed}" / "games.jsonl")
                          for seed in seeds], **basis}
    listed_path = work / "candidates.json"
    listed: dict[str, Any] | None = None
    try:
        listed = json.loads(listed_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        listed = None
    if listed is not None and listed.get("sampling") != sampling:
        was = listed.get("sampling")
        was = was if isinstance(was, dict) else {}
        moved = [key for key in basis if was.get(key) != basis[key]]
        why = (f"was made on another {' and '.join(WORK_WORDS[key] for key in moved)}"
               if moved else "was sampled another way")
        aside = _aside(work, today)
        work.rename(aside)
        note(f"{game}: its work {why}; moved aside to {aside.name}")
        listed = None
    elif listed is None and work.exists() and any(work.iterdir()):
        aside = _aside(work, today)
        work.rename(aside)
        note(f"{game}: its work had no count; moved aside to {aside.name}")
    if listed is None:
        began = time.monotonic()
        games = 0

        def records(seed: int) -> Iterator[GameRecord]:
            nonlocal games
            for one in read_records(runs_dir / f"{digest}-{seed}" / "games.jsonl"):
                games += 1
                yield one

        distinct = sum(1 for _ in candidates(engine, ((seed, records(seed)) for seed in seeds)))
        near = list(candidates(
            engine, ((seed, read_records(runs_dir / f"{digest}-{seed}" / "games.jsonl"))
                     for seed in seeds), near=NEAR, near_min=NEAR_MIN))
        taken = sample(near)
        listed = {"ruleset_hash": digest, "sampling": sampling, "games": games,
                  "distinct": distinct, "near_count": len(near),
                  "near_games": len({(c.seed, c.game) for c in near}),
                  "candidates": [c.as_dict() for c in taken]}
        write(listed_path, listed)
        seconds["count"] = round(time.monotonic() - began, 1)
        note(f"{game}: {games} games, {distinct} small positions, {len(near)} near a won "
             f"end, {len(taken)} taken ({seconds['count']:.0f} s)")
    taken_rows = list(listed["candidates"])

    # The proofs.
    began = time.monotonic()
    units = [(work / f"batch-{start:04d}.json", taken_rows[start:start + PROVE_BATCH])
             for start in range(0, len(taken_rows), PROVE_BATCH)]
    kept = _standing_units([(path, [row["id"] for row in rows]) for path, rows in units],
                           built, nodes)

    def prove_row(row: dict[str, Any]) -> dict[str, Any]:
        return prove(engine, Candidate(row["seed"], row["game"], row["ply"], row["pieces"],
                                       tuple(row["moves"])), nodes=nodes)

    _ask_units(units, kept, prove_row,
               lambda _rows, proved: {"built": built, "nodes": nodes,
                                      "made": today.isoformat(), "rows": proved},
               workers, note, f"{game}: proofs, batch")
    batches = [json.loads(path.read_text(encoding="utf-8")) for path, _rows in units]
    if len({batch["built"] for batch in batches}) > 1:
        raise RuntimeError(f"{game}: proofs from two builds")
    rows = sorted((row for batch in batches for row in batch["rows"]),
                  key=lambda row: (row["seed"], row["game"], row["ply"]))
    proved_on = batches[0]["built"] if batches else built
    made = max((datetime.date.fromisoformat(batch["made"]) for batch in batches
                if batch.get("made")), default=today)
    seconds["prove"] = round(time.monotonic() - began, 1)

    # T3, T4, then T5, each read with the answers before it.
    asked: dict[str, Any] = {}
    asks = {"defence": asks_defence, "hasten": asks_defence, "shallow": asks_shallow,
            "shows": asks_shows}
    # The sift (Q1084) reads the game against the reference, `rulesets/chess.json` here
    # unless the caller hands one.
    seen = distinction(spec, reference if reference is not None else reference_spec())

    def sift(engine: Engine, row: dict[str, Any], nodes: int) -> list[str]:
        return shows(engine, row, seen, nodes=nodes)

    questions = {"defence": defence, "hasten": hasten, "shallow": shallow, "shows": sift}
    for test in TESTS:
        began = time.monotonic()
        wanted = [row for row in rows if asks[test](row)]
        test_units = [(work / f"{test}-{start:04d}.json", wanted[start:start + TEST_BATCH])
                      for start in range(0, len(wanted), TEST_BATCH)]
        kept = _standing_units(
            [(path, [row["id"] for row in unit]) for path, unit in test_units],
            built, defence_nodes)
        question = questions[test]

        def put(row: dict[str, Any], question: Any = question) -> Any:
            return question(engine, row, nodes=defence_nodes)

        def unit_of(unit: list[dict[str, Any]], answers: list[Any],
                    test: str = test) -> dict[str, Any]:
            if test == "shallow":
                said = [{"id": row["id"], "shallow": answer[0], "shallow_moves": answer[1]}
                        for row, answer in zip(unit, answers, strict=True)]
                extra: dict[str, Any] = {"agents": list(SHALLOW_AGENTS),
                                         "seeds": list(SHALLOW_SEEDS)}
            else:
                said = [{"id": row["id"], test: answer}
                        for row, answer in zip(unit, answers, strict=True)]
                extra = {}
            return {"built": built, "nodes": defence_nodes, "made": today.isoformat(),
                    **extra, "rows": said}

        _ask_units(test_units, kept, put, unit_of, workers, note, f"{game}: {test}, batch")
        loaded = [json.loads(path.read_text(encoding="utf-8")) for path, _unit in test_units]
        answers = {row["id"]: {key: value for key, value in row.items() if key != "id"}
                   for unit in loaded for row in unit["rows"]}
        rows = [{**row, **answers[row["id"]]} if row["id"] in answers else row for row in rows]
        # Stamped even when it asked no position: the test was applied, to none.
        asked |= {f"{test}_nodes": defence_nodes,
                  f"{test}_built": loaded[0]["built"] if loaded else built,
                  f"{test}_made": max((datetime.date.fromisoformat(unit["made"])
                                       for unit in loaded), default=today)}
        seconds[test] = round(time.monotonic() - began, 1)
        note(f"{game}: {test} asked of {len(wanted)} ({seconds[test]:.0f} s)")

    data = record(engine, game, seeds, rows, proved_on, nodes=nodes, made=made,
                  distinct=int(listed["distinct"]), near_min=NEAR_MIN, **asked)
    out = record_path(runs_dir, digest)
    if out.exists():
        work.mkdir(parents=True, exist_ok=True)
        old = work / f"record-{today.isoformat()}.json"
        out.replace(_aside(old, today) if old.exists() else old)
    write(out, data)
    offered = sum(1 for row in data["positions"]
                  if row.get("result") == WON and not trivial(row))
    note(f"{game}: recorded {len(data['positions'])} proved, {offered} offered")
    return {"hash": digest, "name": game, "counts": data["counts"],
            "proved": len(data["positions"]), "offered": offered, "seconds": seconds,
            "built": built, "set_aside": len(data.get("set_aside") or [])}


def refilter(
    spec: dict[str, Any],
    runs_dir: Path | str,
    workers: int = WORKERS,
    say: Any = None,
    nodes: int = DEFENCE_NODES,
    today: datetime.date | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sift one game's delivered record (Q1084): ask `shows` of every row T1 to T5 keep and
    set aside each that shows nothing its game has and chess lacks (`sifted`). Proves
    nothing again: a record that holds keeps every proof, and the record it replaces is kept
    beside the work as `record-<day>.json`, as `make` keeps one, never written over.

    Rows already answered under this `SHOWS_VERSION` are not asked again. Returns what it
    did: how many rows it asked, how many stay offered and how many are set aside, and the
    seconds it took.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    began = time.monotonic()
    today = today or datetime.date.today()
    engine = Engine(spec)
    digest = engine.spec.hash
    game = str(spec.get("name") or digest)
    path = record_path(runs_dir, digest)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("ruleset_hash") != digest:
        raise ValueError(f"{game}: its record cannot be read as this game's")
    fresh = not is_sifted(data)
    seen = distinction(spec, reference if reference is not None else reference_spec())
    rows = [*(data.get("positions") or []), *(data.get("set_aside") or [])]
    wanted = [row for row in rows if asks_shows(row) and (fresh or "shows" not in row)]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        answers = list(pool.map(lambda row: shows(engine, row, seen, nodes=nodes), wanted))
    for row, answer in zip(wanted, answers, strict=True):
        row["shows"] = answer
    if fresh:
        # A row a sift of another version answered, and this one did not ask (a row T1 to
        # T5 leave out), keeps no answer: it is not offered either way.
        asked = {id(row) for row in wanted}
        for row in rows:
            if id(row) not in asked:
                row.pop("shows", None)
    old = json.loads(path.read_text(encoding="utf-8"))
    sifted(data, nodes, work_basis(spec)["built"], today)
    work = work_dir(runs_dir, digest)
    work.mkdir(parents=True, exist_ok=True)
    kept_as = work / f"record-{today.isoformat()}.json"
    kept_as = _aside(kept_as, today) if kept_as.exists() else kept_as
    write(kept_as, old)
    write(path, data)
    offered = sum(1 for row in data["positions"] if row.get("result") == WON and not trivial(row))
    aside = len(data["set_aside"])
    if say is not None:
        say(f"{game}: sifted {len(wanted)}, {offered} offered, {aside} set aside "
            f"(the record before kept as {kept_as.name})")
    return {"hash": digest, "name": game, "asked": len(wanted), "offered": offered,
            "set_aside": aside, "seconds": round(time.monotonic() - began, 1)}
