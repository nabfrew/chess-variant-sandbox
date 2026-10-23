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
    fold_payload,
    fold_transit,
    goal_payload,
    portal_payload,
    portal_transit,
    walls_payload,
    wrap_payload,
    zone_payload,
)
from chess_variant_sandbox.web import draw_offer
from chess_variant_sandbox.web.library import Game, engine_for

HUMAN = "human"

SEEDS = 2**31
"""How many seeds a game can be given: `fresh_seed` draws below it, and the store's
`seed` column (`bigint`) and the core's `u64` take any of them."""


def fresh_seed() -> int:
    """A seed for a game nobody named one for, drawn from the operating system (Q987).

    Every game on either server used to run with seed 0 when the request named none, so
    the computer chose the same move in the same position at the same ply in every game
    anyone played (a round game's mixed strategy is drawn from `seed + ply`;
    `docs/reference/round-hardest-weakness.md` §1). The seed is drawn once, when the
    game starts, and kept with it -- the match row, or the session -- so a game is still a
    pure function of `(ruleset_hash, seed, agent_configs)` and replays exactly. `secrets`,
    not `random`: nothing here may share a stream with anything else.
    """
    return secrets.randbelow(SEEDS)


def seed_asked(asked: Any) -> int:
    """The seed a request named, or a fresh one when it named none.

    `asked` is the body's `seed`, absent or null for a page that does not name one, which
    every page does today. A named seed is kept exactly, whatever it is, because a test or
    a reproduction that names one depends on it; anything that is not a whole number is
    refused rather than guessed at.
    """
    if asked is None:
        return fresh_seed()
    if isinstance(asked, bool) or (isinstance(asked, float) and not asked.is_integer()):
        raise ValueError(f"a seed is a whole number, and {asked!r} is not one")
    try:
        return int(asked)
    except (TypeError, ValueError):
        raise ValueError(f"a seed is a whole number, and {asked!r} is not one") from None


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
    if config.kind == "matrix":
        # Schema 26 (Q918b): a node is one round resolved, both players' moves at once.
        return f"weighs {budget:,} rounds before each move"
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


def _portal_bar(state: GameState) -> dict[str, int] | None:
    """Schema 20: the state's straight-back bar as `{standing, left}`, or ``None``."""
    standing, left = state.portal_bar
    return {"standing": standing, "left": left} if standing >= 0 and left >= 0 else None


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


def effects_payload(engine: Any, state: GameState) -> dict[str, Any]:
    """Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.6): the effects on a position a game has
    passed through, for a step back to draw its marks, as ``{"effects": [[square, bits], ...]}``
    with only the squares whose piece is under an effect (1 frozen, 2 disarmed, 4 shielded);
    the core's answer (`Engine.effects`). Empty in a game without a sight, so a history entry
    of every other game is what it was."""
    if engine.spec.sight is None:
        return {}
    bits = engine.effects(state).bits
    return {"effects": [[square, value] for square, value in enumerate(bits) if value]}


def resting_payload(state: GameState) -> list[dict[str, int]]:
    """Schema 29 (`pieces[].rest`): every resting piece of a position, ``{player, square,
    left}``, player 0's first: a piece that made its player's move and may not make it for
    that player's next ``left`` turns unless it is all that player can move. The core's state
    (``GameState.rest``), read and never worked out; ``[]`` without a rest."""
    return [
        {"player": player, "square": square, "left": left}
        for player in (0, 1) for square, left in state.rest[player]
    ]


def rest_entry(engine: Any, before: GameState, after: GameState, seat: int) -> dict[str, Any]:
    """Schema 29 (Q983b, `docs/plan-rest.md` §5.6): what a history entry says of the rest, in a
    game whose pieces rest: the resting pieces of the position it left (``resting``), so a step
    back badges its own position, and ``lifted`` where the move was chosen under a lift (every
    move ``seat`` had was a resting piece's), the core's answer (`Engine.rest_turn`). Empty in
    every other game, so a history entry of every other game is what it was."""
    if not engine.spec.rest:
        return {}
    out: dict[str, Any] = {"resting": resting_payload(after)}
    if engine.rest_turn(before, seat).lifted:
        out["lifted"] = True
    return out


# Schema 29 (Q983b, `docs/plan-rest.md` §5.6): a round chosen under a lift, in the round's
# sentences; `{side}` is what the board calls the player.
LIFTED_WORDS = "every move {side} had was a resting piece's, so the rest was lifted for {side} this round"


def sights_payload(engine: Any, state: GameState) -> list[dict[str, Any]]:
    """Schema 30 (Q999c, §4.6): what each piece with a line of sight sees on this board, for
    the board's hover, as ``[{"square", "seen": [[square, bits], ...]}]``, square order: the
    core's answer for each (`Engine.sight_of`), a ray's stopping piece included and nothing
    past it. The page computes no square of it. Only the pieces whose type declares a sight
    are asked, and a piece that sees nothing on this board is left out."""
    spec = engine.spec
    if spec.sight is None:
        return []
    carriers = {kind for kind, entries in enumerate(spec.sight.entries) if entries}
    out = []
    for square, value in enumerate(state.board):
        if not value or spec.owner_of(value) > 1 or spec.type_of(value) not in carriers:
            continue
        seen = engine.sight_of(state, square)
        if seen:
            out.append({"square": square, "seen": [[target, bits] for target, bits in seen]})
    return out


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

    # Every draw offer made in this game, as the store logged it (Q790,
    # `0035_draw_offers.sql`): `seat` and `ply`, and against the computer what the core
    # said. Read for when the next offer may be made and what the last one was answered;
    # empty in the local application, which keeps no result and offers no draw.
    draw_offers: list[dict[str, Any]] = field(default_factory=list)

    # Schema 26 (Q918c, `docs/archive/plan-simultaneous.md` §6.6): the moves sealed in the round
    # now open, by seat, and the depth each computer seat reported beside its own. Never in
    # a snapshot: a sealed move reaches the page only as `round.sealed`'s one boolean, and
    # its move only at the reveal, as history. Empty of moves in every alternating game.
    pending: list[int | None] = field(default_factory=lambda: [None, None])
    pending_depths: list[int | None] = field(default_factory=lambda: [None, None])

    # Schema 26 by correspondence (Q918f, `site/correspondence.py::snapshot_for`): the seat
    # this snapshot is drawn for, when each of two people reads the game at their own
    # screen; -1 for a reader in neither seat, None everywhere else (one screen, the local
    # application, a game against the computer). The board then offers that seat's moves
    # alone, whatever the other has done. `sealed_unseen` is each seat whose move the store
    # holds sealed and did not give this session: the reader learns it is in, never which.
    viewer: int | None = None
    sealed_unseen: tuple[bool, bool] = (False, False)

    # A try at converting a proven position (R9b, `site/proven.py::try_snapshot`): which
    # position, where its prefix ends, the person's side and the board's sentences. The
    # store's to set, None everywhere else; the board then offers no take-back.
    puzzle: dict[str, Any] | None = None

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
        # Schema 27 (Q935c, `docs/plan-spy.md` §3.6): a spy game's phase here, the core's
        # answer (`Engine.phase`); None in every other game.
        spy = engine.phase(self.state) if engine.spy else None
        # Schema 26: a game in rounds has no side to move. Whose choice the board offers,
        # its moves, the verdict and whether a computer seat still owes its sealed move are
        # `_round_view`'s; everything else in the snapshot is drawn as for any game. A spy
        # game is one by turns until a spy is lost, and in rounds after.
        in_rounds = spy.is_round if spy is not None else engine.simultaneous
        rounds = self._round_view(engine, spy) if in_rounds else None
        if rounds is None:
            legal = engine.legal_moves(self.state)
            outcome = engine.terminal(self.state, legal)
            mover = self.state.side_to_move
        else:
            legal, outcome, mover = rounds["legal"], rounds["outcome"], rounds["mover"]
        # How this game ended, if it has. The engine's answer where the position has
        # one, and the row's where it does not, because a game somebody gave up is over
        # in a position that is still playable.
        ended = (
            {"winner": outcome.winner, "reason": outcome.reason}
            if outcome is not None
            else self.verdict
        )
        # Schema 27 (Q942b, `both`): an ending that fired for both players in one round, in
        # the words the page ends its sentence with in place of the reason's.
        if outcome is not None and rounds is not None and ended is not None:
            how = both_fired_words(engine, self.state, outcome)
            if how:
                ended = {**ended, "how": how}
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
                # Schema 28 (Q943b, `docs/plan-speed.md` §4.5): each type's speed, compiled
                # (`SimRule.speed`), for the legend's line of ranks; empty in every game
                # without a speed, which draws no line.
                "speed": list(engine.spec.simultaneous.speed)
                if engine.spec.simultaneous is not None else [],
                # Schema 29 (Q983b, `docs/plan-rest.md` §5.6): each type's rest, compiled
                # (`CompiledSpec.rest`), so a game whose longest rest is over 1 numbers its
                # pause signs with the turns left; empty in every game without a rest.
                "rest": list(engine.spec.rest),
            },
            "seats": list(self.seats),
            # How hard each seat thinks, in plain words off the budget it plays at, for
            # the panel that names the seats. Empty for a person.
            "strength": [strength(seat) for seat in self.seats],
            # Whether this game counts, which the board draws a mark for and which
            # decides whether it offers a take-back or a resignation. Empty where there
            # are no modes at all, and the board then draws neither mark.
            "mode": self.mode,
            # R9b: a try from a proven position, which the board names and words; absent
            # on every other game.
            **({"puzzle": self.puzzle} if self.puzzle is not None else {}),
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
            # Schema 20 (`no_straight_back`): the end the last move carried, standing on
            # `standing`, may not go straight back to `left` this move; `None` where nothing
            # is barred. The state's own bar (`GameState.portal_bar`), read and never worked
            # out: the engine already leaves the move out, and the page only says so (Q838c).
            "portalBar": _portal_bar(self.state),
            # Schema 22: where each fold stands (`viz.wrapped.fold_payload`), `[]` without
            # folds. A fold is not on the board byte, so the page reads it here, and draws
            # it under the pieces.
            "folds": fold_payload(engine, self.state),
            # Schema 29 (Q983a, `pieces[].rest`): every resting piece, `{player, square,
            # left}`, player 0's first: a piece that made its player's move and may not make
            # it for that player's next `left` turns unless it is all that player can move.
            # The core's state, in every game, alternating or in rounds; `[]` without a rest.
            "resting": resting_payload(self.state),
            # Schema 30 (Q999a, `docs/archive/plan-passive.md` §4.6): the effect bits on each square
            # (1 frozen, 2 disarmed, 4 shielded, on the piece standing there), as the core
            # answers them (`Engine.effects`), for the page's marks. Absent in every game
            # without a sight, whose block is what it always was.
            # And (Q999c) what each piece with a sight sees on this board, for the board's
            # hover (`sights_payload`); absent in every game without a sight.
            **(
                {"effects": list(engine.effects(self.state).bits),
                 "sights": sights_payload(engine, self.state)}
                if engine.spec.sight is not None else {}
            ),
            "ply": self.state.ply,
            # A game in rounds has no check (§2.5): a royal under attack may simply move.
            "inCheck": (
                engine.in_check(self.state, mover)
                if outcome is None and rounds is None else False
            ),
            "moves": [] if ended else self._offer(engine, legal, mover),
            # Schema 26 (§6.6): the round now open, as this game's viewers may see it --
            # who has sealed (a boolean a seat, never the move), whose choice the board
            # offers, the initiative, the void run and the bars. Absent in a game that
            # alternates.
            **({"round": rounds["block"]} if rounds is not None else {}),
            # Schema 27 (Q935c, §3.6): a spy game's phase as the core answers it --
            # `alternating`, `reply`, `announced` or `blind` -- and in an announced round the
            # announcer, the player without a spy, who chooses first and in the open; None
            # in every other phase. The page words the phase line from it and works nothing
            # out. Absent in every game without a spy.
            **(
                {"spy": {"phase": spy.name,
                         "announcer": spy.announcer if spy.name == "announced" else None}}
                if spy is not None else {}
            ),
            "history": self.history,
            "lastMove": self.history[-1] if self.history else None,
            # Where each wild piece steps after the coming move, if that move leaves its
            # way as it is, and after the reply (`comingThen`): the core's answer for this
            # position (`engine.coming.coming_phases`), `[{from, to, eaten}]` each, and
            # empty for every ruleset without `rules.npcs`. A finished game has no coming
            # move, so nothing is coming.
            **({"coming": [], "comingThen": []} if ended else _coming(engine, self.state)),
            "outcome": ended,
            # Whether the person may offer the computer a draw now, why not, and the
            # computer's answer to an offer made in this position (Q790); `None` where no
            # such offer is made: the local application, a watch, a game between two
            # people (whose own block says it) and a finished game.
            # A game in rounds offers none yet: the computer's answer is the alternating
            # search's (`web/draw_offer.py`), which does not play rounds.
            # A spy game offers none in any phase, for the same reason: its rounds follow.
            "drawOffer": self._draw_offer(ended) if rounds is None and spy is None else None,
            # The page uses this to decide whether to ask for the opponent's reply. It
            # is deliberately not the same call: a human move should land on the board
            # before an agent starts searching, or the interface freezes for the length
            # of the search. In a round, it asks for the computer's sealed move, which is
            # chosen on the round's board before the person's is accepted.
            "awaitingAgent": (
                ended is None and self.seats[mover] != HUMAN
                if rounds is None else ended is None and rounds["awaiting"]
            ),
        }

    def _draw_offer(self, ended: dict[str, Any] | None) -> dict[str, Any] | None:
        """A draw offer to the computer as the page offers it (Q790), or None.

        `may` and `why` are `draw_offer`'s rule (own turn, from ply 40, one in ten plies),
        `said` the sentence for the core's answer to an offer made at this very ply -- a
        decline, since an accepted offer ends the game and the banner says so.
        """
        if not self.mode or ended is not None or list(self.seats).count(HUMAN) != 1:
            return None
        seat = list(self.seats).index(HUMAN)
        ply = self.state.ply
        refusal = draw_offer.against_computer_refusal(
            self.draw_offers, seat, ply, self.state.side_to_move
        )
        last = draw_offer.last_offer(self.draw_offers, seat)
        answered = last if last is not None and int(last.get("ply", -1)) == ply else None
        return {
            "may": refusal is None,
            "why": refusal,
            "said": draw_offer.answer_said(answered),
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

    def _offer(self, engine: Any, legal: list[int], mover: int) -> list[dict[str, Any]]:
        """The legal moves, described well enough for a person to pick one.

        Promotions and drops are the two cases where a square pair is not a move: several
        moves share (from, to) and differ only in what the piece becomes, and a drop has
        no origin square at all. Both are handed to the page as separate options rather
        than resolved here, because which one a player wants is a question for the player.

        `mover` is whose moves these are. In a game that alternates it is the side to
        move; in a round (schema 26) it is the seat choosing, which the position does not
        carry, so each move is described on the position as that seat sees it.
        """
        state = _as_seat(self.state, mover)
        offered = []
        for move in legal:
            kind = move_type(move)
            entry = {
                "move": move,
                "from": None if kind == MOVE_DROP else move_from(move),
                "to": move_to(move),
                "text": engine.describe_move(state, move),
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
                entry.update(portal_transit(engine, state, move) or {})
                if move_promotion(move):
                    entry["promotion"] = move_promotion(move) - 1
            # Schema 22: a move through a fold, its way in and out and the group's colour
            # (`viz.wrapped.fold_transit`), for the dot where only a fold reaches (Q843d).
            entry.update(fold_transit(engine, state, move) or {})
            # Schema 12: a strike under `choose` is one move per landing square, so the
            # page offers the square, named here rather than worked out there.
            landing = engine.landing_square(state, move)
            if landing is not None:
                entry["landing"] = landing
                entry["landingName"] = square_name(landing, engine.spec.width)
            offered.append(entry)
        return offered

    # ------------------------------------------------------------------ writing

    def play(self, move: int) -> dict[str, Any]:
        """Apply one move, which must be legal in the current position.

        In a game in rounds (schema 26) the move is a person's sealed choice for the round
        now open (`seal`), and the round is revealed once every seat's is in.
        """
        engine = self.engine
        if self._in_rounds(engine):
            return self.seal(move)
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

        A game in rounds stores both moves of a round in seat order (§6.3's record), so a
        replayed move is player 0's, sealed, or player 1's, which reveals the round.
        A spy game's record is the same flat list, one move a turn and two a round in seat
        order whoever announced (`docs/plan-spy.md` §2.5), the step's kind asked of the
        engine at each position.
        """
        engine = self.engine
        if self._in_rounds(engine):
            seat = 0 if self.pending[0] is None else 1
            if move not in engine.legal_moves_for(self.state, seat):
                raise ValueError("that move is not legal in this position")
            self.pending[seat] = move
            self.pending_depths[seat] = None if self.seats[seat] == HUMAN else depth
            if seat == 1:
                self._reveal(engine)
            return
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
        if self._in_rounds(engine):
            # A round: every computer seat seals its move on the round's board, blind to
            # anything a person has chosen, and a round with no person in it is revealed.
            # An announced round (a spy game's) asks one choice a call, in its order: the
            # announcer's, shown at once, and the answerer's once the announcement is in.
            if engine.terminal(self.state) is None:
                self._seal_computers(engine, one=True)
                if None not in self.pending:
                    self._reveal(engine)
            return self.snapshot()
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
        if self.engine.spy:
            return self._undo_step()
        if self.engine.simultaneous:
            # A round is taken back whole (§6.6, *undo pops a round*), and with it any move
            # sealed in the round that was open: the computer seals again on the board the
            # take-back restores, as it would on any round's.
            self.pending = [None, None]
            self.pending_depths = [None, None]
            if self.states:
                self.state = self.states[-2]
                del self.states[-2:], self.history[-2:], self.played[-2:]
            return self.snapshot()
        watched = HUMAN not in self.seats
        while self.states:
            self.state = self.states.pop()
            self.history.pop()
            self.played.pop()
            if watched or self.seats[self.state.side_to_move] == HUMAN:
                break
        return self.snapshot()

    def _undo_step(self) -> dict[str, Any]:
        """A spy game's take-back (schema 27, `docs/plan-spy.md` §3.6): step by step, a turn
        one move and a round whole, back to the last position a person was asked to choose
        in -- a turn of a person's, or any round with a person in it -- or one step in a
        game nobody is sitting at. A round's moves are taken back with it, a sealed or
        announced one of the round that was open too; a computer seat chooses again on the
        board the take-back restores, as it would on any."""
        self.pending = [None, None]
        self.pending_depths = [None, None]
        engine = self.engine
        watched = HUMAN not in self.seats
        while self.states:
            steps = 2 if "round" in self.history[-1] else 1
            self.state = self.states[-steps]
            del self.states[-steps:], self.history[-steps:], self.played[-steps:]
            if watched:
                break
            if self._in_rounds(engine) or self.seats[self.state.side_to_move] == HUMAN:
                break
        return self.snapshot()

    def _in_rounds(self, engine: Any) -> bool:
        """Whether the step now open is a round: always in a game in rounds, by the phase
        (the core's) in a spy game, never in a game that alternates."""
        return bool(engine.is_round(self.state))

    def _announcer(self, engine: Any) -> int | None:
        """The announcer of the round now open, if it is a spy game's announced round (the
        core's phase): the player without a spy, who chooses first and in the open."""
        if not engine.spy:
            return None
        phase = engine.phase(self.state)
        return phase.announcer if phase.name == "announced" else None

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
                # Schema 22 (Q843d): a move through a fold, its way in and out
                # (`viz.wrapped.fold_transit`), for the ripple and the trail; and where the
                # folds stand once it is made (`fold_payload`), so a position stepped back to
                # draws them where they were then. Absent and empty without folds.
                **(fold_transit(engine, self.state, move) or {}),
                **({"folds": folds} if (folds := fold_payload(engine, state)) else {}),
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
                # Schema 30 (Q999c): the effects on the position this ply left, so a step
                # back draws its marks (`effects_payload`); absent without a sight.
                **effects_payload(engine, state),
                # Schema 29 (Q983b): the resting pieces this ply left, and whether it was
                # chosen under a lift (`rest_entry`); absent without a rest.
                **rest_entry(engine, self.state, state, self.state.side_to_move),
            }
        )
        self.state = state

    # ------------------------------------------------------------------ rounds
    #
    # Schema 26 (Q918c, `docs/archive/plan-simultaneous.md` §6.6). A round is both seats choosing on
    # one board, sealed, then revealed and resolved together by the core
    # (`Engine.apply_round_with_phase`). The order this keeps is the whole of the rule's
    # honesty at a board: **a computer seat seals before a person's move is accepted**,
    # chosen by the core's own round player (`Engine.select_round`) from the round's
    # position and nothing else, so it cannot answer a move it never saw. The page asks for
    # it as the round opens (`awaitingAgent`, `play_agent`); `seal` asks again first, so a
    # page that never asked still cannot get its move in ahead of the computer's.

    def _choosing(self, announcer: int | None = None) -> int | None:
        """The seat whose move the board offers: the first person yet to seal, else None.

        Against the computer that is the person, every round. Between two people at one
        screen it is the first player, then the second, and neither is shown the other's.
        Between two people at two screens (`viewer`, Q918f) it is the reader, while their
        own move is not in, and nobody once it is.

        In an announced round (schema 27, `announcer` given) the order is the rule's: the
        announcer first, then the answerer once the announcement is in. A seat still owed
        by the computer comes first in that order, so nobody is offered a move then. At two
        screens (Q935e) the answerer is offered nothing until the announcement is in.
        """
        if self.viewer is not None:
            seat = self.viewer
            if seat not in (0, 1) or self.pending[seat] is not None:
                return None
            if announcer is not None and seat != announcer and self.pending[announcer] is None:
                return None
            return None if self.sealed_unseen[seat] else seat
        if announcer is not None:
            for seat in (announcer, 1 - announcer):
                if self.pending[seat] is None:
                    return seat if self.seats[seat] == HUMAN else None
            return None
        for seat in (0, 1):
            if self.seats[seat] == HUMAN and self.pending[seat] is None:
                return seat
        return None

    def _owing(self, announcer: int | None) -> list[int]:
        """The computer seats whose move the round now open waits on: every one yet to
        seal, or in an announced round the next seat in the rule's order, if a computer's."""
        if announcer is None:
            return [
                seat for seat in (0, 1)
                if self.seats[seat] != HUMAN and self.pending[seat] is None
            ]
        for seat in (announcer, 1 - announcer):
            if self.pending[seat] is None:
                return [seat] if self.seats[seat] != HUMAN else []
        return []

    def _rounds_played(self) -> int:
        """Rounds revealed so far: two history entries each. A spy game's turns before its
        rounds are one entry each and are not rounds."""
        return sum(1 for entry in self.history if "round" in entry) // 2

    def _round_view(self, engine: Any, spy: Any = None) -> dict[str, Any]:
        """What `snapshot` reads for a game in rounds: the choosing seat's moves, the
        verdict, who the strip names, whether a computer seat owes its sealed move, and
        the `round` block the page draws the round from. `spy` is a spy game's phase."""
        outcome = engine.terminal(self.state)
        announcer = spy.announcer if spy is not None and spy.name == "announced" else None
        choosing = self._choosing(announcer)
        owing = self._owing(announcer)
        # The seat the strip names: a computer still choosing first, since the board waits
        # on it, then the person choosing. A reader at their own screen whose move is in
        # is waiting on the other seat, which the strip names instead (Q918f).
        mover = owing[0] if owing else choosing if choosing is not None else 0
        reader = self.viewer if self.viewer in (0, 1) else None
        if reader is not None and choosing is None:
            mover = 1 - reader
        legal = (
            engine.legal_moves_for(self.state, choosing)
            if choosing is not None and not owing else []
        )
        run = self.state.void_run
        sim = engine.spec.simultaneous
        mine = self.pending[reader] if reader is not None else None
        block = {
            # Rounds revealed so far, and this one's number, counted from one.
            "number": self._rounds_played() + 1,
            # Who has sealed this round. A boolean a seat: the move itself is never sent
            # before the reveal, to anybody (§6.6, *the computer commits blind*), but its
            # own seat's reader (`yours`, below).
            "sealed": [
                move is not None or unseen
                for move, unseen in zip(self.pending, self.sealed_unseen, strict=True)
            ],
            "choosing": choosing,
            # The tie rule (§3): `both`, `initiative` or `impossible`, and who holds the
            # initiative (-1 under any other rule), the core's state.
            "tie": _TIE_WORDS.get(sim.tie, "both") if sim is not None else "both",
            "initiative": self.state.initiative,
            # The void run (§2.7): the two squares of a pair whose one void round was the
            # last, player 0's first, or None; and every barred pair. The core's state,
            # drawn and never worked out.
            "voidRun": list(run) if run[0] >= 0 and run[1] >= 0 else None,
            "bars": [list(bar) for bar in self.state.bars],
        }
        if reader is not None and mine is not None and reader != announcer:
            # Q918f: the reader's own sealed move, described as an offered move is, so the
            # board can draw it ghosted where it would land. Only ever the reader's seat.
            # An announcement is not one (Q935e): it is `announced`, below, for both.
            block["yours"] = self._offer(engine, [mine], reader)[0]
        if announcer is not None:
            # Schema 27 (Q935c, `docs/plan-spy.md` §1.4, §3.6): an announced round's
            # announcer, and once it is in, the announced move, described as an offered
            # move is: it is meant to be seen, by both players, and the board draws it as
            # its own arrow. Absent from every other round.
            shown = self.pending[announcer]
            block["announcer"] = announcer
            block["announced"] = (
                self._offer(engine, [shown], announcer)[0] if shown is not None else None
            )
        return {
            "legal": legal,
            "outcome": outcome,
            "mover": mover,
            "awaiting": bool(owing),
            "block": block,
        }

    def seal(self, move: int) -> dict[str, Any]:
        """A person's move for the round now open, sealed; the round revealed once every
        seat's move is in.

        The computer's is sealed first, on the round's board, if it is not in already. A
        move cannot be sealed twice: against the computer the round is revealed the moment
        the person's is in, and between two people at one screen the second player is
        choosing by then.
        """
        engine = self.engine
        if engine.terminal(self.state) is not None:
            raise ValueError("this game is over")
        announcer = self._announcer(engine)
        if announcer is not None:
            # An announced round (schema 27): a computer announcer announces before a
            # person answers, and a computer answerer is asked nothing before the person's
            # announcement is in (`_seal_computers` stops at the first person in the order).
            self._seal_computers(engine)
        seat = self._choosing(announcer)
        if seat is None:
            raise ValueError("no move is owed by a person in this round")
        if move not in engine.legal_moves_for(self.state, seat):
            raise ValueError("that move is not legal in this position")
        if announcer is None:
            self._seal_computers(engine)
        self.pending[seat] = move
        self.pending_depths[seat] = None
        if None not in self.pending:
            self._reveal(engine)
        return self.snapshot()

    def _seal_computers(self, engine: Any, one: bool = False) -> None:
        """Every computer seat's move for this round, chosen by the core on the round's
        board and sealed. Blind by construction: `select_round` is handed the position and
        the seat, and a person's sealed move is not in the position.

        In an announced round (schema 27, `docs/plan-spy.md` §1.4) the seats are asked in
        the rule's order and nothing past a person yet to choose: a computer announcer is
        asked with no announcement, a computer answerer with the move announced, and never
        before it is in. `one` asks one choice only, so a board between two computers shows
        the announcement before the answer to it."""
        announcer = self._announcer(engine)
        order = (0, 1) if announcer is None else (announcer, 1 - announcer)
        for seat in order:
            agent = self.seats[seat]
            if self.pending[seat] is not None:
                continue
            if agent == HUMAN:
                if announcer is not None:
                    return
                continue
            native = native_spec(agent, engine.spec.hash)
            if native is None:
                raise ValueError(f"{agent} does not play a game in rounds here")
            seed = self.seed + self.state.ply
            values = eval_values(engine.spec)
            if announcer is None:
                move, _strategy, _nodes, depth = engine.select_round(
                    self.state, seat, native, values, seed
                )
            else:
                shown = None if seat == announcer else self.pending[announcer]
                move, _strategy, _nodes, depth = engine.select_round(
                    self.state, seat, native, values, seed, announced=shown
                )
            if move is None:
                return  # a player with no move: the verdict says so (§2.6)
            # How far it looked, in rounds, for the players that deepen: every step after a
            # spy game's first round is a round too, so its minimax counts rounds here.
            choices = len(engine.legal_moves_for(self.state, seat))
            self.pending[seat] = move
            deepens = native[0] == "matrix" or (engine.spy and native[0] in DEEPENING)
            self.pending_depths[seat] = depth if deepens and choices > 1 else None
            if one and announcer is not None:
                return

    def _reveal(self, engine: Any) -> None:
        """Both sealed moves, resolved together by the core, written as two history entries
        in seat order -- the round's notation, `12. Nd5 | Nd5` -- the second carrying the
        reveal: what each move did and the round's events in words (`round_words`)."""
        first, second = self.pending[0], self.pending[1]
        assert first is not None and second is not None
        before = self.state
        announcer = self._announcer(engine)
        number = self._rounds_played()
        after, phase, events = engine.apply_round_with_phase(before, first, second)
        art = icons_for(
            engine.spec.piece_names,
            engine.spec.piece_ids,
            (self.game.spec.get("appearance") or {}).get("icons"),
        )
        sides = side_names(art)
        reveal = round_reveal(engine, before, after, (first, second), events, sides)
        # Schema 29 (Q983b, `docs/plan-rest.md` §5.6): a seat whose every move was a resting
        # piece's chose under a lift, which the round's sentences say (the core's answer).
        rests = [rest_entry(engine, before, after, seat) for seat in (0, 1)]
        for seat in (0, 1):
            if rests[seat].get("lifted"):
                line = LIFTED_WORDS.format(side=sides[seat]) + "."
                reveal["words"].append(line[:1].upper() + line[1:])
        if announcer is not None:
            # Schema 27 (Q935c): who announced, so the move list marks the announced move.
            reveal["announcer"] = announcer
        for seat, move in enumerate((first, second)):
            seen = _as_seat(before, seat)
            entry: dict[str, Any] = {
                "ply": before.ply + seat,
                "text": engine.describe_move(seen, move),
                "from": None if move_type(move) == MOVE_DROP else move_from(move),
                "to": move_to(move),
                "by": self.seats[seat],
                "player": seat,
                # The wild pieces walk once a round, after both moves: the phase is the
                # round's, on its second entry, which is the one `lastMove` names.
                "phase": [step.as_dict() for step in phase] if seat == 1 else [],
                **(portal_transit(engine, seen, move) or {}),
                **(fold_transit(engine, seen, move) or {}),
                "depth": self.pending_depths[seat],
                "check": False,
                # What this move came to: `void`, `won` or `lost` a collision, `both lost`.
                "result": reveal["results"][seat],
                "round": number,
                **rests[seat],
            }
            if seat == 1:
                entry["reveal"] = reveal
                entry.update(_coming(engine, after))
                entry.update(effects_payload(engine, after))
                if folds := fold_payload(engine, after):
                    entry["folds"] = folds
            self.states.append(before)
            self.played.append(move)
            self.history.append(entry)
        self.state = after
        self.pending = [None, None]
        self.pending_depths = [None, None]


# `SimRule.tie` in the words the page and the round's sentences use.
_TIE_WORDS = {0: "both", 1: "initiative", 2: "impossible"}

# `Engine.apply_round_with_phase`'s event kinds (`rust/src/rounds.rs`).
_EVENT_COLLISION, _EVENT_VOID, _EVENT_INITIATIVE, _EVENT_BARRED = 0, 1, 2, 3
# Schema 27 (Q942a, `swap: "collision"`): two pieces that swapped squares met; the square is
# the first player's destination. Q942b draws the clash between the two squares.
_EVENT_SWAP = 4
# Schema 28 (Q943a, `docs/plan-speed.md` §4.2): an ordered round -- the faster player (5,
# the square its move ended on), a slower move lost with its piece (6, its origin) or never
# played because the faster move ended the game (7). Q943b draws the two beats.
_EVENT_ORDER, _EVENT_LOST, _EVENT_UNPLAYED = 5, 6, 7
_EVENT_NAMES = ("collision", "void", "initiative", "barred", "swap", "order", "lost", "unplayed")


# `ScoringRule.count` as the score's name in the both-fire sentence: what a side was ahead
# on, and what was level.
_SCORE_WORDS = {
    "material": ("on material", "the material"),
    "pieces": ("on pieces left", "the pieces left"),
    "region": ("on the squares held", "the squares held"),
    "territory": ("on ground held", "the ground held"),
    "points": ("on points", "the points"),
}


def both_fired_words(engine: Any, state: GameState, outcome: Any) -> str | None:
    """How a round game ended when its ending fired for both players in the last round,
    as the end of the sentence *‹side› wins ...* or *Drawn ...*, or None for any other end.

    Schema 27 (Q942a's `both: "score"`, `docs/archive/plan-round-chase.md` §3 X1): a both-fire is
    settled by `rules.scoring` under the switch and is a draw without it. The verdict is the
    core's (`outcome`); this only says it, from two facts of the final position (neither side
    has a royal left, or neither a piece) and the rule. Without it the page would say the
    winner won *by capturing the royal piece* when both royals went."""
    spec = engine.spec
    n_squares = spec.n_squares

    def holds(side: int, royal: bool) -> bool:
        for value in state.board[:n_squares]:
            if value and spec.owner_of(value) == side and (
                not royal or spec.type_of(value) in spec.royal_types
            ):
                return True
        hand = state.hands[side] if spec.drops is not None else ()
        kinds = spec.royal_types if royal else range(len(hand))
        return any(kind < len(hand) and hand[kind] > 0 for kind in kinds)

    if outcome.reason in ("royal_captured", "royal_destroyed") and spec.royal_types:
        if holds(0, True) or holds(1, True):
            return None
        what = "both royal pieces were taken in the same round"
    elif outcome.reason == "annihilated":
        if holds(0, False) or holds(1, False):
            return None
        what = "both sides lost their last piece in the same round"
    else:
        return None
    sim = spec.simultaneous
    scoring = spec.scoring if sim is not None and sim.both_score else None
    count = scoring.count if scoring is not None else ""
    ahead, level = _SCORE_WORDS.get(count, ("on the score", "the score"))
    scored = scoring is not None
    if outcome.winner is not None:
        return f"{ahead}, as {what}"
    if scored:
        return f"as {what} and {level} was level"
    return f"as {what}"


def _as_seat(state: GameState, seat: int) -> GameState:
    """The position as one seat sees it to choose a move: the same position, with that
    seat named as the one moving, so a move of the second seat in a round (where the
    state names nobody) is described from its own side. Unchanged where it already is."""
    return state if state.side_to_move == seat else state.with_(side_to_move=seat)


def round_reveal(
    engine: Any,
    before: GameState,
    after: GameState,
    moves: tuple[int, int],
    events: tuple[tuple[int, int, int], ...],
    sides: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """A revealed round, as the page draws it and a reader reads it.

    Every fact here is the core's: the moves both seats sealed, and the events
    `apply_round_with_phase` logged as it resolved them -- a collision and its winner, a
    void move, a tie the initiative decided, a pair barred. This only names them: which
    piece moved from where to where, and a sentence per event in the game's own names for
    the pieces and the sides. Nothing here decides an outcome; a move with no event did
    what it said, which the boards before and after show.

    `moves` is each seat's `{from, to, void}`, `events` each event as `{kind, square,
    player}`, `results` each seat's move in one word (`void`, `won`, `lost`, `both lost`,
    schema 28's `unplayed`, or None), `words` the sentences, in the order the events came.
    `sides` is what the board calls the two players (`viz.pieces.side_names`).
    """
    spec = engine.spec
    width = spec.width

    def piece(seat: int) -> str:
        move = moves[seat]
        if move_type(move) == MOVE_DROP:
            kind = move_promotion(move) % spec.n_types
        else:
            kind = spec.type_of(before.board[move_from(move)])
        return f"{sides[seat]}'s {spec.piece_names[kind]}"

    def named(square: int) -> str:
        return square_name(square, width)

    def said(seat: int) -> str:
        """The move as its seat sealed it, on the round's board."""
        return str(engine.describe_move(_as_seat(before, seat), moves[seat]))

    origins = [None if move_type(m) == MOVE_DROP else move_from(m) for m in moves]
    targets = [move_to(m) for m in moves]
    voids = {player for kind, _square, player in events if kind == _EVENT_VOID}
    ordered = any(kind == _EVENT_ORDER for kind, _square, _player in events)
    results: list[str | None] = [None, None]
    words: list[str] = []
    if len(voids) == 2:
        words.append(
            f"{piece(0)} and {piece(1)} each went for the other, which had moved: "
            f"both moves were void."
        )
    for kind, square, player in events:
        if kind == _EVENT_VOID:
            results[player] = "void"
            if ordered:
                words.append(
                    f"{piece(player)} could no longer go to {named(targets[player])} on the "
                    f"board the faster move left: its move was void and it stayed."
                )
            elif len(voids) == 1:
                words.append(
                    f"{piece(player)} on {named(square)} went for {named(targets[player])}, "
                    f"which had moved: its move was void and it stayed."
                )
        elif kind == _EVENT_COLLISION:
            meeting = f"{piece(0)} and {piece(1)} met on {named(square)}"
            if player < 0:
                results[0] = results[1] = "both lost"
                words.append(f"{meeting}: both were lost.")
            else:
                results[player], results[1 - player] = "won", "lost"
                words.append(f"{meeting}: {piece(player)} won, and {piece(1 - player)} was lost.")
        elif kind == _EVENT_SWAP:
            meeting = (
                f"{piece(0)} and {piece(1)} swapped squares and met between "
                f"{named(targets[1])} and {named(targets[0])}"
            )
            if player < 0:
                results[0] = results[1] = "both lost"
                words.append(f"{meeting}: both were lost.")
            else:
                results[player], results[1 - player] = "won", "lost"
                words.append(f"{meeting}: {piece(player)} won, and {piece(1 - player)} was lost.")
        elif kind == _EVENT_INITIATIVE:
            words.append(
                f"The tie went to {sides[player]}, who held the initiative; "
                f"it passes to {sides[1 - player]}."
            )
        elif kind == _EVENT_ORDER:
            # Schema 28 (Q943b, `docs/plan-speed.md` §4.5): the round's sentence leads with
            # the order, and the faster move as it was sealed.
            words.append(f"{piece(player)} was faster: {said(player)}.")
        elif kind == _EVENT_LOST:
            results[player] = "lost"
            faster, stood = 1 - player, origins[player]
            if stood is not None and targets[faster] == stood:
                words.append(
                    f"{piece(faster)} took {piece(player)} on {named(stood)} before "
                    f"it could move: its move was lost."
                )
            else:
                words.append(
                    f"{piece(player)} was taken or changed before it could move: its move "
                    f"was lost."
                )
        elif kind == _EVENT_UNPLAYED:
            results[player] = "unplayed"
            words.append(
                f"The faster move ended the game, so {piece(player)}'s move was not played."
            )
        elif kind == _EVENT_BARRED:
            pair = next((bar for bar in after.bars if bar[0] == square), None)
            there = f"{named(pair[0])} and {named(pair[1])}" if pair else named(square)
            words.append(
                f"A second void round in a row for {there}: neither may take the other that "
                f"way until one of the two squares changes."
            )
    if ordered:
        # The slower move, made: the board the faster left offered it (§2.5). Onto the
        # square the faster piece just took, it takes that piece there (case C).
        first = next(player for kind, _square, player in events if kind == _EVENT_ORDER)
        slower = 1 - first
        if results[slower] is None and slower not in voids:
            if targets[slower] == targets[first] and origins[slower] is not None:
                words.append(f"{piece(slower)} then took it on {named(targets[slower])}.")
            else:
                words.append(f"{piece(slower)} then made its move: {said(slower)}.")
    swapped = (
        origins[0] is not None and origins[1] is not None
        and targets[0] == origins[1] and targets[1] == origins[0] and not voids
        and not ordered
        and not any(kind == _EVENT_SWAP for kind, _square, _player in events)
    )
    if swapped:
        words.insert(
            0, f"{piece(0)} and {piece(1)} swapped squares: each left the square the other took."
        )
    run = after.void_run
    if run[0] >= 0 and run[1] >= 0 and voids:
        words.append(
            f"One void round for {named(run[0])} and {named(run[1])}: a second in a row "
            f"bars those moves."
        )
    return {
        "moves": [
            {"from": origins[seat], "to": targets[seat], "player": seat, "void": seat in voids}
            for seat in (0, 1)
        ],
        "events": [
            {"kind": _EVENT_NAMES[kind], "square": square, "player": player}
            for kind, square, player in events
        ],
        "results": results,
        # A sentence starts with a capital, whatever the sides are called (`white`).
        "words": [line[:1].upper() + line[1:] for line in words],
    }


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
        self,
        game: Game,
        seats: tuple[str, str],
        seed: int | None = None,
        scratch: bool = False,
    ) -> Session:
        """Hold a new game; `seed` None (nobody named one) draws a `fresh_seed`."""
        engine = engine_for(game.spec)
        session = Session(
            id=secrets.token_urlsafe(9),
            game=game,
            seats=seats,
            seed=fresh_seed() if seed is None else seed,
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
