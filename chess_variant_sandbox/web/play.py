"""A game in progress, held for a browser to play.

The rule this file exists to honour is the one in CLAUDE.md: *Python may hold a position
and ask the engine questions; Python may not answer them.* A session holds a `GameState`
and a move list. Every question about that position -- what is legal, what a move does,
whether the game is over, what the opponent plays -- goes to the engine, and the
opponent's move goes through the core's own agent, the one `choose_move` runs and the
tournament rates, asked to say how deep it looked as well. Playing a game to find out what it is like and then reading its index would
otherwise be exercising two different players.

Sessions live in memory and die with the process. That is the right lifetime for a local
tool: a game in progress is not a result, and anything worth keeping is a ruleset, which
is already on disk.

The public site wants the other lifetime, and `chess_variant_sandbox/site/matches.py` gets it from
`played` -- the moves that made this position, in order. Replaying them through `replay`
rebuilds the whole session, history and all, because every question on the way is put to
the engine again. So a match is a list of integers in a row and a restart costs a replay.
Beside them the row keeps `depths`, how far the machine looked for each move, because
that is the one thing on a move the engine cannot be asked again: it was the search's
answer at the time, and a replayed move is not searched (Q424).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.agents.base import AgentConfig, native_spec, parse_agent
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.engine import _square_name as square_name
from chess_variant_sandbox.engine.moves import (
    MOVE_DROP,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    MOVE_TYPE_NAMES,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import choose_move, eval_values
from chess_variant_sandbox.viz.heading import directional_types
from chess_variant_sandbox.viz.pieces import icons_for, side_names
from chess_variant_sandbox.viz.wrapped import (
    goal_payload,
    portal_payload,
    portal_transit,
    walls_payload,
    wrap_payload,
    zone_payload,
)
from chess_variant_sandbox.web.library import Game, engine_for

HUMAN = "human"

# The agent kinds whose search in the core deepens one move at a time, and so the only
# ones whose reported depth says how far they looked. The core reports zero for every
# other kind (`rust/src/agents.rs::Stats::depth`), which from a searching player would
# mean it played without looking ahead at all -- so a zero is only passed on from these.
DEEPENING = frozenset({"minimax"})


def strength(seat: str) -> str:
    """How hard a seat thinks, in words somebody who has never seen this project reads.

    Off the node budget the seat plays at, because the budget *is* the player: 400 and
    6,400 nodes are two rated players under one kind of search. The picker's label names
    the rung -- "hard" -- and the seat panel used to stop there, so two machines read
    *hard vs hard* and nothing on the board said what hard meant (Q288).

    Empty for a person, and for a kind this does not know how to describe: a player
    registered from outside says nothing rather than something made up for it.
    """
    if seat == HUMAN:
        return ""
    try:
        config = parse_agent(seat)
    except ValueError:
        return ""
    if config.kind == "random":
        return "plays anything legal"
    # In turns, the unit the move list says how far a searching player looked in
    # (`board.js::lookedHtml`), so the picker and the board count one way (F7).
    if config.kind == "greedy":
        return "takes what it can see, one turn ahead"
    budget = _budget(config)
    if budget is None:
        return ""
    if config.kind == "mcts":
        return f"tries {budget:,} random games before each move"
    if config.kind in ("minimax", "fairystockfish"):
        return f"looks at {budget:,} positions before each move"
    return ""


def _budget(config: AgentConfig) -> int | None:
    """The node budget this seat plays at: the one it names, else its kind's default.

    The default is read off the core's own parameters for the kind with no options, so a
    seat written without a budget is described at the budget it is actually given. Only
    the kind goes in: resolving a rule of thumb's digest needs its store loaded, and the
    budget does not depend on it.
    """
    named = config.options.get("nodes")
    if isinstance(named, (int, float)):
        return int(named)
    native = native_spec(AgentConfig(config.kind))
    return native[1] if native is not None and native[1] > 0 else None


def _reply(
    engine: Any, seat: str, state: GameState, seed: int, choices: int
) -> tuple[int | None, int | None]:
    """The seated agent's move, and how many moves ahead its search got, from the core.

    The move is ``choose_move``'s: the same parameters and the same seed go to the same
    agent, and ``select_move_depth`` is ``select_move`` returning what the search reached
    beside what it chose. Nothing here works out a depth -- a search's depth is a
    property of the position and the budget together, and only the core knows it.

    The depth is ``None`` where there is nothing to report: an agent that does not look
    ahead by deepening, one the core does not run, and a position with one legal move,
    where no search happens.
    """
    native = native_spec(seat, engine.spec.hash)
    if engine.core is None or native is None:
        return choose_move(engine, seat, state, seed=seed), None
    move, depth = engine.select_move_depth(state, native, eval_values(engine.spec), seed)
    if native[0] not in DEEPENING or choices < 2:
        return move, None
    return move, depth


def _coming(engine: Any, state: GameState) -> dict[str, list[dict[str, Any]]]:
    """Where each wild piece steps next from ``state``, and after that, as the core plays it.

    `engine.coming.coming_phases` is the two phases the rules would run on the board as it
    stands (Q660, Q764), each in the shape a ply's own ``phase`` has: ``coming`` after the
    next move, ``comingThen`` after the reply, which is every step a wild piece takes
    before the player now to move moves again. The page draws each as an arrow per wild
    piece, the second fainter, and decides nothing: which piece walks where, whether it
    waits and what it would eat are the core's. Wild pieces walk fixed routes, so nothing
    here is drawn at random; the one thing it cannot know is the coming moves, which may
    block, clear or take a wild piece.
    """
    first, then = coming_phases(engine, state)
    return {
        "coming": [step.as_dict() for step in first],
        "comingThen": [step.as_dict() for step in then],
    }


def drop_faces_payload(drops: Any) -> list[int] | None:
    """Each hand slot's other face under ``drops.either_face`` (schema 18e, Q785), or ``None``.

    The compiled pairs, so the page offers a piece in hand as both faces without reading
    the spec's ``promoted_from``.
    """
    if drops is None or not drops.faces:
        return None
    return list(drops.faces)


@dataclass
class Session:
    """One game, one browser tab."""

    id: str
    game: Game
    seats: tuple[str, str]
    seed: int
    state: GameState
    history: list[dict[str, Any]] = field(default_factory=list)
    states: list[GameState] = field(default_factory=list)

    # The moves that made this position, as the engine encodes them. `history` describes
    # a move for a reader and cannot be replayed; this can, which is what the site
    # persists and what an undo truncates.
    played: list[int] = field(default_factory=list)
    # A game on a draft nobody kept. The engine answers it exactly as it answers any
    # other -- the flag says nothing about the rules -- but the rest of the application
    # is keyed by a ruleset hash the library can resolve, and a draft's cannot be. So:
    # nothing is recorded about it, and the page it came from is the editor rather than
    # a game page that does not exist.
    scratch: bool = False

    # Which net each learned seat really plays, `net:<12 hex>` (`nets.played_by`), "" for
    # every other seat (Q709). Read when the game starts, which is also where a learned
    # seat whose net cannot be loaded refuses: the agent string names a training
    # procedure, the same on every board, and this names the net that sat down.
    played_by: tuple[str, str] = ("", "")

    # What the store calls this game -- `sandbox` or `leaderboard`, `site/matches.py`'s
    # own two words. Empty for the local application, which keeps nothing and has no
    # modes, and the board reads it to say whether the game it is drawing counts.
    mode: str = ""

    # A finish the position does not show: a resignation, or the sweep resigning for
    # somebody who closed the tab. `{"winner": int | None, "reason": str}`, as the row
    # recorded it. Giving a game up is not a fact about the position, so the engine has
    # no answer to it and this is not the page working one out -- it is the store saying
    # what happened, and it is used only where the engine reports no outcome of its own.
    verdict: dict[str, Any] | None = None

    @property
    def engine(self) -> Any:
        return engine_for(self.game.spec)

    @property
    def depths(self) -> list[int | None]:
        """How far the machine looked for each move in `played`, in the same order.

        `None` for a move with nothing to report. What the site stores beside `played`,
        and what `replay` takes back, so a rebuilt game says what the played one said.
        """
        return [entry.get("depth") for entry in self.history]

    # ------------------------------------------------------------------ reading

    def snapshot(self) -> dict[str, Any]:
        """Everything the page needs to draw the position and offer a move."""
        engine = self.engine
        legal = engine.legal_moves(self.state)
        outcome = engine.terminal(self.state, legal)
        mover = self.state.side_to_move
        # How this game ended, if it has. The engine's answer where the position has
        # one, and the row's where it does not, because a game somebody gave up is over
        # in a position that is still playable.
        ended = (
            {"winner": outcome.winner, "reason": outcome.reason}
            if outcome is not None
            else self.verdict
        )
        art = icons_for(
            engine.spec.piece_names,
            engine.spec.piece_ids,
            (self.game.spec.get("appearance") or {}).get("icons"),
        )

        return {
            "id": self.id,
            "ruleset": {
                "hash": self.game.hash,
                "name": self.game.name,
                # Whether this ruleset can be looked up at all. Every link off the board
                # -- back to its page, a rematch, the other side -- goes through the
                # library by hash, and a draft is not in it.
                "scratch": self.scratch,
                "width": engine.spec.width,
                "height": engine.spec.height,
                # `"hex"` draws each cell as a hexagon on a sheared grid (schema 14).
                "lattice": engine.spec.lattice,
                "pieceIds": list(engine.spec.piece_ids),
                "pieceNames": list(engine.spec.piece_names),
                # Schema 18e (Q785): each hand slot's other face, or -1, compiled; `None`
                # where a held piece comes back as itself only. The hand draws a button
                # per face from it, so the picker never reads `promoted_from`.
                "dropFaces": drop_faces_payload(engine.spec.drops),
"nTypes": engine.spec.n_types,
                # A whole army's worth of board values, the headings a facing piece is
                # drawn at, and which pieces may be turned to a heading at all. A piece
                # with no direction of its own is drawn upright for both seats.
                "sideStride": engine.spec.side_stride,
                "facingAngles": [list(row) for row in engine.spec.facing_headings()],
                "directional": list(directional_types(engine.spec)),
                "nFacings": engine.spec.n_facings,
                # What the colouring on this board is *for*, which decides whether it
                # is scenery or the scoreboard. Read from the spec rather than inferred
                # from a non-empty colour array: a game may claim ground and be won on
                # something else entirely.
                "territory": (self.game.spec.get("rules") or {}).get("territory"),
                "icons": art,
                # What to call the two players, which is what the board is drawing and
                # therefore not something the page can settle for itself. "white" and
                # "black" are wrong for a single-colour set.
                "sides": list(side_names(art)),
                # The *compiled* wrap rule, per player and per axis. The spec spells its
                # permissions as edge names and `engine/compile.py` is the one place
                # that means anything by them, so what crosses is the resolved pair of
                # booleans and the page never reads the DSL.
                "wrap": wrap_payload(engine.spec.wrap, engine.spec.copies),
                # Squares a game ends on when a piece reaches one, compiled, per
                # player. The spec spells them as names and mirrors them by a keyword,
                # so what crosses is the resolved pair of sets -- `None` for a game
                # with no such rule, and a board that never drew them asked a player to
                # find the winning square by reading the rules panel.
                "goals": goal_payload(engine.spec.reach),
                # Schema 18 (Q786): whether the game is decided by moving off one of
                # those squares rather than standing on one, so the ended banner says
                # which (`reasonWords`). Off the compiled rule, as `goals` is.
                "goalLeaves": bool(engine.spec.reach and engine.spec.reach.leaves),
                # Squares a piece promotes on, compiled, per player. The spec states
                # the zone once from the first player's side and the schema reflects
                # it, so what crosses is the resolved pair of bands -- `None` for a
                # game with no promotion, which is what the board tests to draw no
                # band and offer no toggle.
                "zones": zone_payload(engine.spec.promotions),
                # Which piece types are royal, compiled, one flag per type. The engine
                # says *whether* the side to move is in check (`inCheck`); this is how
                # the board finds the square to draw it on, by the piece standing there,
                # without deciding anything about attacks.
                "royal": [t in engine.spec.royal_types for t in range(engine.spec.n_types)],
                # Explosion metadata: the page needs to know *which* disappearances
                # are blast damage so it can animate them differently from a plain
                # capture, and which piece types survive the blast (the "resistant"
                # indicator).  Constant per ruleset, so sent once in the ruleset block.
                "explosion": self._explosion_info(engine),
                # Schema 19: each portal end's pair, colour, movers and doors, compiled
                # (`viz.wrapped.portal_payload`); `None` without portals.
                "portals": portal_payload(engine.spec),
            },
            "seats": list(self.seats),
            # How hard each seat thinks, in plain words off the budget it plays at, for
            # the panel that names the seats. Empty for a person.
            "strength": [strength(seat) for seat in self.seats],
            # Whether this game counts, which the board draws a mark for and which
            # decides whether it offers a take-back or a resignation. Empty where there
            # are no modes at all, and the board then draws neither mark.
            "mode": self.mode,
            "board": list(self.state.board),
            # Counts per piece type, which is what the state carries -- not a list of
            # pieces. Empty for every ruleset without drops.
            "hands": [list(self.state.hands[0]), list(self.state.hands[1])],
            # Who owns each square: 0 nobody, 1 the first player, 2 the second. Empty
            # for every ruleset without a territory rule -- and three committed rulesets
            # are *won* on this, so a board that never drew it was hiding the score of
            # the game being played on it.
            "colour": list(self.state.colour),
            # Squares nothing may occupy. Empty for every ruleset without a wall. A
            # player who cannot see one is being asked why the rook stopped there.
            **walls_payload(engine.spec),
            # Moves left in this turn. 1 for every ruleset without a turn rule, where
            # the side flips on every move; a player owed a second move needs telling.
            "remaining": self.state.remaining,
            "sideToMove": mover,
            "ply": self.state.ply,
            "inCheck": engine.in_check(self.state, mover) if outcome is None else False,
            "moves": [] if ended else self._offer(engine, legal),
            "history": self.history,
            "lastMove": self.history[-1] if self.history else None,
            # Where each wild piece steps after the coming move, if that move leaves its
            # way as it is, and after the reply (`comingThen`): the core's answer for this
            # position (`engine.coming.coming_phases`), `[{from, to, eaten}]` each, and
            # empty for every ruleset without `rules.npcs`. A finished game has no coming
            # move, so nothing is coming.
            **({"coming": [], "comingThen": []} if ended else _coming(engine, self.state)),
            "outcome": ended,
            # The page uses this to decide whether to ask for the opponent's reply. It
            # is deliberately not the same call: a human move should land on the board
            # before an agent starts searching, or the interface freezes for the length
            # of the search.
            "awaitingAgent": ended is None and self.seats[mover] != HUMAN,
        }

    @staticmethod
    def _explosion_info(engine: Any) -> dict[str, Any] | None:
        """Explosion metadata for the page's animation layer.

        Returns ``None`` when the ruleset has no explosion rule, so the page can
        skip the check entirely.  Otherwise returns the blast neighbourhood per
        square and a boolean-per-type array saying which types are immune.

        ``by`` and ``of`` are schema 10 and are ``None`` when every capture explodes.
        They have to cross for the reason the blast does: **the page cannot work out from
        two positions which board change was which**, so a rule that narrows *when* the
        blast fires has to reach it or the animation lies. Without them a capture by a
        piece that is not a bomb would still be treated as a blast centre and the shield
        shimmer would draw survivors of an explosion that never happened.
        """
        ex = engine.spec.explosion
        if ex is None:
            return None
        n_types = engine.spec.n_types

        def selection(types: frozenset[int] | None) -> list[bool] | None:
            return None if types is None else [t in types for t in range(n_types)]

        return {
            "blast": [list(neighbours) for neighbours in ex.blast],
            "immune": [t in ex.immune_types for t in range(n_types)],
            "by": selection(ex.by_types),
            "of": selection(ex.of_types),
        }

    def _offer(self, engine: Any, legal: list[int]) -> list[dict[str, Any]]:
        """The legal moves, described well enough for a person to pick one.

        Promotions and drops are the two cases where a square pair is not a move: several
        moves share (from, to) and differ only in what the piece becomes, and a drop has
        no origin square at all. Both are handed to the page as separate options rather
        than resolved here, because which one a player wants is a question for the player.
        """
        offered = []
        for move in legal:
            kind = move_type(move)
            entry = {
                "move": move,
                "from": None if kind == MOVE_DROP else move_from(move),
                "to": move_to(move),
                "text": engine.describe_move(self.state, move),
                "kind": MOVE_TYPE_NAMES.get(kind, "normal"),
            }
            if kind == MOVE_DROP:
                # What stands on the square afterwards, and (schema 18e) the hand slot
                # the drop spends, which differ only for a piece dropped as its other face.
                entry["drop"] = move_promotion(move) % engine.spec.n_types
                drops = engine.spec.drops
                entry["hand"] = entry["drop"] if drops is None else drops.hand_slot(entry["drop"])
            elif kind == MOVE_PROMOTION:  # what the piece becomes travels with it
                entry["promotion"] = move_promotion(move) % engine.spec.n_types
            elif kind == MOVE_ROTATE:
                entry["rotation"] = move_promotion(move)
            elif kind == MOVE_PORTAL:
                # Schema 19: the end it goes in by, the end it comes out beside and the
                # way it comes out, named here (`viz.wrapped.portal_transit`); the target
                # is the exit, and the promotion field carries the new piece plus one.
                entry.update(portal_transit(engine, self.state, move) or {})
                if move_promotion(move):
                    entry["promotion"] = move_promotion(move) - 1
            # Schema 12: a strike under `choose` is one move per landing square, so the
            # page offers the square, named here rather than worked out there.
            landing = engine.landing_square(self.state, move)
            if landing is not None:
                entry["landing"] = landing
                entry["landingName"] = square_name(landing, engine.spec.width)
            offered.append(entry)
        return offered

    # ------------------------------------------------------------------ writing

    def play(self, move: int) -> dict[str, Any]:
        """Apply one move, which must be legal in the current position."""
        engine = self.engine
        if move not in engine.legal_moves(self.state):
            raise ValueError("that move is not legal in this position")
        self._apply(engine, move, by=self.seats[self.state.side_to_move])
        return self.snapshot()

    def replay(self, move: int, depth: int | None = None) -> None:
        """Apply a move somebody already played, with the depth the core reported for it.

        How a stored game comes back (`site/matches.py::session`). The move is put to the
        engine exactly as `play` puts it, so a stored list that could not have happened
        fails here rather than drawing a board that never existed. It is not searched
        again: the move was chosen already, and a search per stored move would cost a
        whole game's thinking on every request.

        So its depth is not worked out either. It is what the core said when the move was
        chosen, which the store kept beside it (Q424), and `None` where it kept nothing --
        a game stored before it kept any shows none. A person's move never has one,
        whatever the store holds. No snapshot is made, since nothing reads one per move.
        """
        engine = self.engine
        if move not in engine.legal_moves(self.state):
            raise ValueError("that move is not legal in this position")
        by = self.seats[self.state.side_to_move]
        self._apply(engine, move, by=by, depth=None if by == HUMAN else depth)

    def play_agent(self) -> dict[str, Any]:
        """Let the seated agent answer.

        The agent is the *rated* one: ``choose_move`` runs the core's implementation
        wherever the core has one, which is every built-in kind. An opponent built from
        the Python agents would be a different player from the one the index measured.

        The move keeps how far the search looked to choose it, as the core reported it,
        and the board shows it beside the move (Q288). This used to be thrown away, so a
        machine that saw one move ahead of a wide position and one that saw four of a
        narrow one looked the same from the other side of the board.
        """
        engine = self.engine
        seat = self.seats[self.state.side_to_move]
        if seat == HUMAN:
            raise ValueError("it is a person's turn")
        legal = engine.legal_moves(self.state)
        if engine.terminal(self.state, legal) is not None:
            return self.snapshot()

        move, depth = _reply(
            engine, seat, self.state, self.seed + self.state.ply, len(legal)
        )
        if move is None:
            return self.snapshot()
        self._apply(engine, move, by=seat, depth=depth)
        return self.snapshot()

    def undo(self) -> dict[str, Any]:
        """Take back to the last position a person was asked to move in.

        One ply is not enough when an agent is seated opposite: taking back your own
        move and leaving the reply on the board is not a position you were ever in.

        A game nobody is sitting at has no such position, so it steps back one ply. The
        alternative is unwinding the whole game, since the loop would never find a seat
        to stop at.
        """
        # A game taken back is a game going on again, so a finish the row recorded does
        # not survive the take-back that undid the position it was recorded in. The
        # store's row is reopened by the same call.
        self.verdict = None
        watched = HUMAN not in self.seats
        while self.states:
            self.state = self.states.pop()
            self.history.pop()
            self.played.pop()
            if watched or self.seats[self.state.side_to_move] == HUMAN:
                break
        return self.snapshot()

    def _apply(self, engine: Any, move: int, by: str, depth: int | None = None) -> None:
        self.states.append(self.state)
        self.played.append(move)
        state, phase = engine.apply_with_phase(self.state, move)
        self.history.append(
            {
                "ply": self.state.ply,
                "text": engine.describe_move(self.state, move),
                "from": None if move_type(move) == MOVE_DROP else move_from(move),
                "to": move_to(move),
                "by": by,
                "player": self.state.side_to_move,
                # What the wild-piece phase did after this move (schema 9), from the
                # core, which is the only thing that knows. One submitted move makes two
                # board changes and the second leaves no trace in the position, so the
                # page draws the phase from this rather than working it out from a diff.
                # Empty for every ruleset without `rules.npcs`.
                "phase": [step.as_dict() for step in phase],
                # Schema 19: a transit's two ends and the way it came out, read off the
                # position it was played in (`viz.wrapped.portal_transit`), so the trail is
                # drawn in two legs and the move animated in two beats, on a position
                # stepped back to as on the live one. Absent for every other move.
                **(portal_transit(engine, self.state, move) or {}),
                # How many moves ahead the machine's search got before it chose this
                # one, counting both sides', from the core. `None` for a person's move
                # and wherever `_reply` has nothing to report. A game rebuilt from its
                # stored moves -- the site's, on every request -- replays them rather
                # than searching again, so a rebuilt move has the depth the store kept
                # beside it (`replay`), and none where the store kept none.
                "depth": depth,
                # Whether this move left the side then to move in check, from the core
                # (the question `inCheck` asks of the position as it stands, asked of
                # each position as it is reached). The move list writes a check sign
                # from it, and a position stepped back to draws its check from it -- the
                # page never works one out (the first-visit study's B1).
                "check": engine.in_check(state, state.side_to_move),
                # Where each wild piece goes next from the position this ply left, and
                # after that (`coming`, `comingThen`), from the core
                # (`engine.coming.coming_phases`), so a position stepped back to shows what
                # was coming *then*. Empty for every ruleset without `rules.npcs`.
                **_coming(engine, state),
            }
        )
        self.state = state


def seats_played(seats: tuple[str, str], ruleset_hash: str) -> tuple[str, str]:
    """`Session.played_by` for these seats: each learned seat's net, "" for any other.

    Raises `nets.NetError` in a sentence for a learned seat whose net cannot be loaded, so
    the game is refused before it starts rather than played by the built-in evaluation
    under the learned player's name (Q709).
    """
    from chess_variant_sandbox.agents.nets import played_by

    first, second = ("" if seat == HUMAN else played_by(seat, ruleset_hash) for seat in seats)
    return first, second


class Sessions:
    """The games currently being played, keyed by an unguessable id."""

    def __init__(self, limit: int = 32) -> None:
        self._sessions: dict[str, Session] = {}
        self._limit = limit

    def start(
        self, game: Game, seats: tuple[str, str], seed: int = 0, scratch: bool = False
    ) -> Session:
        engine = engine_for(game.spec)
        session = Session(
            id=secrets.token_urlsafe(9),
            game=game,
            seats=seats,
            seed=seed,
            state=engine.initial_state(),
            scratch=scratch,
            played_by=seats_played(seats, engine.spec.hash),
        )
        self._sessions[session.id] = session
        # Oldest first, so a long-lived server does not accumulate abandoned games.
        while len(self._sessions) > self._limit:
            self._sessions.pop(next(iter(self._sessions)))
        return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("This game is no longer running; start a new one.")
        return session

    def find(self, session_id: str) -> Session | None:
        """This id's session, or None where `get` would refuse.

        For a caller that holds two stores and is deciding which one an id belongs to,
        which is the site with a draft in memory and every other game in a row.
        """
        return self._sessions.get(session_id)
