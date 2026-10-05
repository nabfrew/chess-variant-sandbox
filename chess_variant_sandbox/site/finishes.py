"""Every game a person finished on the site, kept move by move so it can be replayed.

Q502a, `docs/reference/general-player.md` §4.2, for its E6 (*where a person wins*). A
loss to a person is the one strength reading that is not the ladder measuring itself,
and a loss that cannot be replayed cannot be studied: which ply lost it, and whether a
bigger search would have seen it, are questions put to a position.

A leaderboard match already kept its moves and verdict for good. A sandbox match did
not: a take-back truncates its move list, a restart empties it, and both clear the
verdict, so a sandbox win followed by *play again* left nothing behind. So a finish is
also written here, by `site/matches.py::_finish`, in the transaction that writes the
verdict, and nothing changes or deletes it afterwards (`0018_finishes.sql`).

**The row names its match and nothing more.** Who sat where, the game, the seed and the
mode are the match row's, where they already were. A visitor's match row has nobody in
their seat, so a visitor's finish is a move list, a result, a count of take-backs and a
time, attached to no person. `recorded` reads the join and hands out no identity either:
a reader who wants one person's games names that person's handle, and gets their games
back without it.

**Nothing here decides anything about a position.** `replay` puts every stored move to
the engine, which says whether it is legal, applies it and gives the verdict at the end
(CLAUDE.md rule 3). A list the engine refuses raises rather than drawing a board that
never happened.

**Nothing reads this for a score.** Like `web/human.py`, it is a record of what a person
did, not an input to any metric, rating or weight. The rating table folds over
`matches`, as it did.

How it gets out: `chess-variant-sandbox site games`, one JSON line per finish, on the
box that holds the store -- `site usage`'s shape, since whoever can run it already holds
the database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from chess_variant_sandbox.engine.state import GameState, Outcome
from chess_variant_sandbox.web.library import engine_for


def record(connection: Any, match_id: str) -> bool:
    """Keep this match's finish, if a person sat at it. Whether a row was written.

    Called by `matches._finish` once the verdict is on the match row, and copies it from
    there with the moves, so the two cannot disagree. A watch -- a computer player in
    both seats -- writes nothing. Does not commit: the caller's transaction holds the
    verdict, the rating update and this row together.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "insert into finishes (match, ordinal, moves, result, reason, takebacks)"
            " select m.id,"
            "  coalesce((select max(f.ordinal) from finishes f where f.match = m.id), 0) + 1,"
            "  m.moves, m.result, coalesce(m.reason, ''), m.takebacks"
            " from matches m"
            " where m.id = %s and m.result is not null"
            " and not (m.seat0_agent is not null and m.seat1_agent is not null)",
            (match_id,),
        )
        return bool(cursor.rowcount)


@dataclass(frozen=True)
class Finish:
    """One finished game with a person in it, as a reader outside the site wants it.

    `person_seat` is the seat the person sat in; `opponent` the computer player in the
    other one. `result` is the *person's* score -- 1, 0.5 or 0 -- and `winner` the
    winning seat, or None for a draw, which are `web/human.py`'s own conventions.
    `signed_in` says whether the seat named an account (always, in leaderboard mode)
    without saying whose.
    """

    match: str
    ordinal: int
    game: str
    name: str
    mode: str
    person_seat: int
    opponent: str
    signed_in: bool
    seed: int
    moves: tuple[int, ...]
    result: float
    winner: int | None
    reason: str
    takebacks: int
    finished: str

    @property
    def person_won(self) -> bool:
        return self.result == 1.0

    def as_dict(self) -> dict[str, Any]:
        found = asdict(self)
        found["moves"] = list(self.moves)
        return found


def recorded(
    connection: Any,
    game: str = "",
    handle: str | None = None,
    mode: str | None = None,
    won: bool = False,
) -> list[Finish]:
    """The finishes kept, oldest first, narrowed as asked.

    `game` is a hash or a prefix of one. `handle` keeps one person's games and never a
    visitor's. `won` keeps the ones the person won, which are the ones E6 marks.
    """
    where = ["true"]
    values: list[Any] = []
    if game:
        where.append("left(m.game, length(%s)) = %s")
        values.extend((game, game))
    if mode is not None:
        where.append("m.mode = %s")
        values.append(mode)
    if handle is not None:
        where.append(
            "exists (select 1 from users u where lower(u.handle) = lower(%s)"
            " and u.id in (m.seat0_user, m.seat1_user))"
        )
        values.append(handle)
    with connection.cursor() as cursor:
        cursor.execute(
            "select m.id, f.ordinal, m.game, g.name, m.mode, m.seat0_agent, m.seat1_agent,"
            " m.seat0_user, m.seat1_user, m.seed, f.moves, f.result, f.reason, f.takebacks,"
            " f.finished"
            " from finishes f join matches m on m.id = f.match join games g on g.hash = m.game"
            f" where {' and '.join(where)} order by f.finished, m.id, f.ordinal",
            tuple(values),
        )
        rows = cursor.fetchall()
    found = [_to_finish(row) for row in rows]
    return [finish for finish in found if finish.person_won] if won else found


def _to_finish(row: Any) -> Finish:
    (match, ordinal, game, name, mode, agent0, agent1, user0, user1, seed, moves,
     result, reason, takebacks, finished) = row
    # A finish is kept only with somebody at the board, so at least one seat has no
    # agent. The site seats one person at a time, so the other seat is the opponent.
    seat = 0 if agent0 is None else 1
    opponent = agent1 if seat == 0 else agent0
    seat0_score = float(result)
    score = seat0_score if seat == 0 else 1.0 - seat0_score
    winner = None if seat0_score == 0.5 else (0 if seat0_score == 1.0 else 1)
    return Finish(
        match=str(match),
        ordinal=int(ordinal),
        game=str(game),
        name=str(name),
        mode=str(mode),
        person_seat=seat,
        opponent=str(opponent or ""),
        signed_in=(user0 if seat == 0 else user1) is not None,
        seed=int(seed),
        moves=tuple(int(move) for move in moves),
        result=score,
        winner=winner,
        reason=str(reason),
        takebacks=int(takebacks),
        finished=finished.isoformat(),
    )


@dataclass(frozen=True)
class Replayed:
    """Where a stored move list leads, as the engine says: the final position, and the
    verdict the engine gives there (None for a game that ended off the board, as a
    resignation does)."""

    state: GameState
    outcome: Outcome | None
    positions: tuple[GameState, ...]


def replay(spec: dict[str, Any], moves: tuple[int, ...] | list[int]) -> Replayed:
    """Play a stored move list through the engine, from the opening position.

    Each move is asked of the engine before it is applied, as `web.play.Session.replay`
    asks, so a list that could not have happened raises `ValueError` naming the ply.
    `positions` holds every position on the way, the opening first, which is what a
    reader marking the ply a game was lost at walks.
    """
    engine = engine_for(spec)
    state = engine.initial_state()
    positions = [state]
    for ply, move in enumerate(moves):
        if engine.terminal(state) is not None:
            raise ValueError(f"ply {ply}: the game was already over")
        if int(move) not in engine.legal_moves(state):
            raise ValueError(f"ply {ply}: move {move} is not legal in this position")
        state = engine.apply(state, int(move))
        positions.append(state)
    return Replayed(state=state, outcome=engine.terminal(state), positions=tuple(positions))
