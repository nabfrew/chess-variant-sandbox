"""The rule interpreter: a thin façade over the Rust core.

Everything downstream talks to this class and nothing else in the package. The public
methods below are the boundary described in CLAUDE.md, and they are unchanged from the
pure-Python engine this replaced -- that stability is the whole reason the swap was
possible without touching agents, tournament or metrics -- and it is why this file is
a facade rather than a dispatcher: there is one engine now, and every committed ruleset
plays on it.

Two things are worth knowing about the shape of this file.

**Positions still cross the boundary as ``GameState``.** A tournament never pays for
that: whole games, whole schedules and whole perft trees are single calls, and the
position never comes back to Python until the game is over. What the per-move methods
buy is that a test, the visualiser and the CLI can still hold a position, print it, and
poke at it in Python, which is worth far more than the microseconds it costs them.

**Compilation stays in Python.** ``compile_spec`` interprets the DSL; the core only
walks the tables it produces. One interpretation means the two sides cannot disagree
about what a rule means. A new movement primitive still changes both languages: its
meaning goes in ``spec/schema.py`` and ``engine/compile.py``, and the core
(``rust/src/game.rs``) has to walk its tables. Skill ``new-move-primitive`` lists the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.engine.bridge import (
    build_tables,
    core_boundary_mismatch,
    core_table_mismatch,
    rust_supports,
)
from chess_variant_sandbox.engine.compile import (
    FACING_NAMES,
    REMOVED_IN_PLACE,
    WILD,
    CompiledSpec,
    compile_spec,
    facing_quarter_turns,
)
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_COMMANDED,
    MOVE_DROP,
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_NORMAL,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    MOVE_SWEEP,
    decode_move,
    move_fold,
    move_landing,
    move_portal,
)
from chess_variant_sandbox.engine.state import (
    NO_SQUARE,
    Effects,
    GameState,
    Outcome,
    PhaseStep,
    RestTurn,
    SpyPhase,
)

if TYPE_CHECKING:
    # A stub-only alias: the compiled module exports no such name at runtime.
    from chess_variant_sandbox._core import State

try:
    import chess_variant_sandbox._core as _core
except ImportError as error:  # pragma: no cover - a build problem, not a code path
    raise ImportError(
        "chess_variant_sandbox._core is not built. The engine is a Rust extension module; run "
        "`uv sync --extra dev` (or `maturin develop`) to build it."
    ) from error


# ``unmoved`` is an arbitrary-width Python integer and a 16x16 board needs 256 bits of
# it, so it crosses as bytes rather than as a machine word.
_MASK_BYTES = 32


def _pack(state: GameState) -> tuple[Any, ...]:
    return (
        state.board,
        state.side_to_move,
        state.unmoved.to_bytes(_MASK_BYTES, "little"),
        state.ep_target,
        state.ep_victim,
        state.halfmove,
        state.ply,
        state.key,
        state.repetition,
        state.checks,
        (list(state.hands[0]), list(state.hands[1])),
        # Everything schema 4 added, grouped: the colouring, the moves left in this turn
        # and the turns taken. Grouped because PyO3's tuple conversions stop at twelve
        # elements, so the boundary has a hard ceiling and the next field joins a group.
        # Schema 9's bounty tally is the first field to meet that ceiling, so it joins
        # this group rather than taking a slot of its own, and schema 18's side that has
        # just left a goal square (Q786), schema 20's straight-back bar (Q807) and schema
        # 22's folds (Q843c) join it the same way.
        (
            state.colour, state.remaining, state.turns_taken, state.points,
            state.goal_left_by, state.portal_bar, list(state.folds),
            # Schema 26 (Q918a): the initiative, the void run and the bars.
            (state.initiative, state.void_run, [tuple(bar) for bar in state.bars]),
            # Schema 29 (Q983a): each player's resting pieces, a field of every position.
            ([tuple(entry) for entry in state.rest[0]], [tuple(entry) for entry in state.rest[1]]),
            # Generation 36 (N1): the respawn clocks, one per respawning half, as bytes
            # (PyO3 crosses a `Vec<u8>` so).
            bytes(state.respawn),
            # Generation 38 (N5): the squares of the wild pieces walking their line back.
            list(state.heading_back),
        ),
    )


def _unpack(packed: State) -> GameState:
    # Typed as ``State`` and not ``tuple[Any, ...]``: this is the only consumer of the
    # declared boundary shape, so it is the only place mypy can notice that the stub
    # and the core have drifted. Loosely typed, the stub went two fields stale through
    # the whole of schema 4 and every check still passed.
    return GameState(
        board=packed[0],
        side_to_move=packed[1],
        unmoved=int.from_bytes(packed[2], "little"),
        ep_target=packed[3],
        ep_victim=packed[4],
        halfmove=packed[5],
        ply=packed[6],
        key=packed[7],
        repetition=tuple(packed[8]),
        checks=packed[9],
        hands=(tuple(packed[10][0]), tuple(packed[10][1])),
        colour=packed[11][0],
        remaining=packed[11][1],
        turns_taken=packed[11][2],
        points=packed[11][3],
        goal_left_by=packed[11][4],
        portal_bar=(packed[11][5][0], packed[11][5][1]),
        folds=tuple(packed[11][6]),
        initiative=packed[11][7][0],
        void_run=(packed[11][7][1][0], packed[11][7][1][1]),
        bars=tuple((bar[0], bar[1]) for bar in packed[11][7][2]),
        rest=(
            tuple((square, left) for square, left in packed[11][8][0]),
            tuple((square, left) for square, left in packed[11][8][1]),
        ),
        respawn=tuple(packed[11][9]),
        heading_back=tuple(packed[11][10]),
    )


class Engine:
    """A compiled ruleset, ready to play.

    Construct once per ruleset and reuse: compilation builds a table for every
    (player, piece type, square) triple, which is cheap once and wasteful per game.

    ``spec`` is the compiled tables as Python sees them, and everything downstream that
    needs to know the shape of the game reads it. ``core`` is the same tables inside the
    Rust extension; it is public because the callers that run *many* games -- the
    tournament runner, the smoke gate, ``bench`` -- go straight to it rather than
    driving this class a move at a time.
    """

    __slots__ = ("spec", "core")

    # Annotated rather than assigned: ``__slots__`` supplies the descriptors and the
    # values are set below. Without these, every downstream ``engine.spec`` is an
    # attribute error to a type checker.
    spec: CompiledSpec
    core: Any

    def __init__(self, spec: dict[str, Any] | CompiledSpec) -> None:
        """Compile a ruleset and hand its tables to the core.

        This used to be a dispatch: ``Engine(spec)`` returned a Rust-backed engine, or
        the pure-Python interpreter for rules the core could not yet express. The core
        expresses all of them now -- the last gap was a bounded screen range, closed by
        giving every attack source a ``prefix_len`` -- so there is nothing to dispatch on
        and the interpreter is gone.

        ``rust_supports`` is still asked, because being wrong in the permissive direction
        is silent: a primitive that lands in the compiler before it lands in the core
        would otherwise be played under rules nobody wrote. It refuses nothing today, and
        a new primitive is expected to add its check there in the same change that adds
        its row to ``MOVE_KINDS``.

        ``core_boundary_mismatch`` asks the other half of that question -- not what the
        core can express, but whether the extension on this machine and this package
        describe the same position. It is asked here, once, so that a stale build says so
        instead of surfacing as an ``IndexError`` in whichever consumer indexed a tuple
        first; ``_pack`` and ``_unpack`` are handed over as the boundary it probes. Not
        memoised on purpose: it costs one opening position where this constructor already
        pays for a whole compile, three orders of magnitude more.

        ``core_table_mismatch`` asks the same question of the *tables*, and has to run
        before they cross rather than after: a core that never reads a key ignores it in
        silence, so there is nothing to catch on the far side. It compares the package's
        keys with ``_core.TABLE_KEYS``, which the core reports off the registry its own
        readers are spelled out of.
        """
        compiled = spec if isinstance(spec, CompiledSpec) else compile_spec(spec)
        unsupported = rust_supports(compiled)
        if unsupported is not None:  # no committed ruleset reaches this; a test stages it
            raise NotImplementedError(f"the core cannot play this ruleset: {unsupported}")
        self.spec = compiled
        tables = build_tables(compiled)
        unread = core_table_mismatch(tables, _core.TABLE_KEYS)
        if unread is not None:  # no current build reaches this; a test stages it
            raise RuntimeError(unread)
        self.core = _core.Core(tables)
        mismatch = core_boundary_mismatch(self.core, _unpack, _pack)
        if mismatch is not None:  # no current build reaches this; a test stages it
            raise RuntimeError(mismatch)

    # ------------------------------------------------------------------ lifecycle

    def initial_state(self) -> GameState:
        return _unpack(self.core.initial_state())

    # ----------------------------------------------------------------- generation

    def legal_moves(self, state: GameState) -> list[int]:
        """Every move the side to move may play."""
        return self.core.legal_moves(_pack(state))

    def pseudo_legal_moves(self, state: GameState) -> list[int]:
        """Moves that satisfy the piece rules, without the self-check filter."""
        return self.core.pseudo_legal_moves(_pack(state))

    # ------------------------------------------------------------- simultaneous

    @property
    def simultaneous(self) -> bool:
        """Whether this game is played in rounds (schema 26, ``rules.simultaneous``)."""
        return self.spec.simultaneous is not None

    @property
    def spy(self) -> bool:
        """Whether this is a spy game (schema 27, ``pieces[].spy``): it alternates while both
        players have a spy and plays rounds once one is lost (`docs/plan-spy.md` §1)."""
        return self.spec.spy is not None

    def phase(self, state: GameState) -> SpyPhase:
        """A spy game's phase here, asked of the core (`docs/plan-spy.md` §1.3):
        ``alternating``, ``reply`` (one ordinary turn by the player who has just lost their
        spy), ``announced`` (a round the announcer, the player without a spy, chooses first
        and in the open) or ``blind`` (a round chosen in secret)."""
        name, announcer = self.core.phase(_pack(state))
        return SpyPhase(name=name, announcer=announcer)

    def is_round(self, state: GameState) -> bool:
        """Whether the next step here is a round (both players move) rather than one move:
        always in a round game, by the phase in a spy game, never otherwise."""
        if self.spy:
            return self.phase(state).is_round
        return self.simultaneous

    def steps(
        self, state: GameState, moves: Sequence[int]
    ) -> Iterator[tuple[GameState, tuple[int, ...]]]:
        """Walk a record's flat move list from ``state``, one step at a time: each yields
        the position the step was chosen on and its moves (one for a turn, two in seat order
        for a round), the step's kind asked of the engine at every position. The reader a
        mixed record needs (schema 27's spy game), and the plain one for every other game."""
        index = 0
        while index < len(moves):
            if self.is_round(state):
                pair = (moves[index], moves[index + 1])
                yield state, pair
                state = self.apply_round(state, *pair)
                index += 2
            else:
                yield state, (moves[index],)
                state = self.apply(state, moves[index])
                index += 1

    def legal_moves_for(self, state: GameState, player: int) -> list[int]:
        """Every move ``player`` may choose in a round on this board (schema 26)."""
        return self.core.legal_moves_for(_pack(state), player)

    def apply_round(self, state: GameState, first: int, second: int) -> GameState:
        """The position after one round: ``first`` by player 0, ``second`` by player 1,
        revealed and resolved together (`docs/reference/rounds.md` §2, §4)."""
        return _unpack(self.core.apply_round(_pack(state), first, second)[0])

    def board_alone(self, state: GameState, player: int, move: int) -> bytes:
        """Q1020: the board after ``player``'s round move ``move`` resolved alone, every piece
        of the other player standing (`docs/reference/rounds.md` §4 steps 2-7, as an ordered
        round's faster step), as the core answers it. What one move of a round does by
        itself, for move text; a board, not a position to play on."""
        return bytes(self.core.board_alone(_pack(state), player, move))

    def apply_round_with_phase(
        self, state: GameState, first: int, second: int
    ) -> tuple[GameState, tuple[PhaseStep, ...], tuple[tuple[int, int, int], ...]]:
        """``apply_round``, what the wild pieces did, and what the meetings did: one
        ``(event, square, player)`` per collision (0, the winner or -1), void move (1),
        tie the initiative decided (2), pair barred (3), under schema 27's
        `swap: "collision"` swap that met (4, the winner or -1), and under schema 28's
        speeds a round made in order (5, the faster move's destination and player), a slower
        move lost with its piece (6, its origin) or never played because the faster move
        ended the game (7), for a page to draw; a slower move the faster one made void is
        a void move (1)."""
        packed, phase, events = self.core.apply_round(_pack(state), first, second)
        return (
            _unpack(packed),
            tuple(PhaseStep.from_core(step) for step in phase),
            tuple((event, square, player) for event, square, player in events),
        )

    # ---------------------------------------------------------------- application

    def apply(self, state: GameState, move: int) -> GameState:
        """Return the position after ``move``. ``state`` is never mutated."""
        return _unpack(self.core.apply(_pack(state), move)[0])

    def apply_with_phase(
        self, state: GameState, move: int
    ) -> tuple[GameState, tuple[PhaseStep, ...]]:
        """``apply``, and what the wild-piece phase did on the way (schema 9).

        A ply with wild pieces in it makes two board changes: the mover's move, and then
        the phase the rules run after it inside the same ``make``. The end position
        carries only the sum, so a caller that has to *show* the ply -- the board in the
        web app, and the move list beside it -- would otherwise be working the phase out
        from a diff and the compiled route table. It asks instead. Empty for every
        ruleset without ``rules.npcs``, and for most plies of the ones with it.
        """
        packed, phase = self.core.apply(_pack(state), move)
        return _unpack(packed), tuple(PhaseStep.from_core(step) for step in phase)

    # -------------------------------------------------------------------- verdict

    def terminal(self, state: GameState, moves: list[int] | None = None) -> Outcome | None:
        """Return the outcome if the game is over, else ``None``.

        ``moves`` is an optional already-generated legal move list. A game loop has one
        in hand every ply, and generation is the expensive half of this call.
        """
        verdict = self.core.terminal(_pack(state), moves)
        if verdict is None:
            return None
        winner, reason = verdict
        return Outcome(winner=winner, reason=reason)

    def terminal_prefix(self, state: GameState) -> Outcome | None:
        """The half of the verdict that does not need a move list.

        Every ending except checkmate and stalemate, which are the two that mean
        "and there is nothing legal to play". A search wants exactly this half before it
        consults its transposition table: a cached value is a value for a position still
        being played, and generating moves to find out whether the game is over would
        undo the table's whole purpose.
        """
        verdict = self.core.terminal_prefix(_pack(state))
        if verdict is None:
            return None
        winner, reason = verdict
        return Outcome(winner=winner, reason=reason)

    def in_check(self, state: GameState, player: int | None = None) -> bool:
        """Whether ``player`` (default: the side to move) has a royal under attack."""
        player = state.side_to_move if player is None else player
        return self.core.in_check(_pack(state), player)

    # --------------------------------------------------------------------- sight

    def effects(self, state: GameState) -> Effects:
        """The effects on this board (schema 30, `docs/archive/plan-passive.md` §3.4), as the core
        answers them: each square's effect bits (``SIGHT_FREEZE``, ``SIGHT_DISARM``,
        ``SIGHT_SHIELD`` on the piece there, for its owner), the side to move's moves the
        effects removed by reason, and whether its royal stands under a suppressed attack.
        All zeros in a game without a sight. Q1007c: ``commanded`` counts the side to move's
        legal commanded moves (`docs/plan-schema-31.md` §3.7). Generation 37 (N2): ``wild``,
        each square's bits from the wild pieces' sights alone (empty without wild pieces)."""
        bits, (freeze, disarm, shield), suppressed, commanded, wild = self.core.effects(
            _pack(state)
        )
        return Effects(tuple(bits), freeze, disarm, shield, suppressed, commanded, tuple(wild))

    def rest_turn(self, state: GameState, player: int) -> RestTurn:
        """What the rest does to ``player``'s choice on this board (schema 29, Q983b,
        `docs/plan-rest.md` §5.6, §7), as the core answers it: whether it took away a legal
        move (``binds``), and whether it was lifted because it would have left none
        (``lifted``). The list asked of is the one the game offers here: ``player``'s in a
        round, else the turn's, as ``player`` would see it. Both false without a rest."""
        binds, lifted = self.core.rest_turn(_pack(state), player)
        return RestTurn(binds, lifted)

    def sight_of(self, state: GameState, square: int) -> list[tuple[int, int]]:
        """The squares the piece on ``square`` sees on this board, each with the effect bits
        of the sights that see it (schema 30), ascending; empty for an empty square."""
        return [(target, bits) for target, bits in self.core.sight_of(_pack(state), square)]

    # ------------------------------------------------------------------- players

    # The three calls below take an agent as the core describes it -- a ``NativeSpec``
    # tuple and a list of piece values -- rather than as a name. Resolving a name into
    # those is an agent concern and lives in ``tournament/play.py``; the engine's job is
    # to pack a position and get out of the way.

    def select_move(
        self, state: GameState, agent: tuple[Any, ...], values: list[float], seed: int = 0
    ) -> int | None:
        """What this agent would play in this position, or ``None`` if it cannot move.

        The one-move counterpart to ``play_game``. Without it, anything needing a single
        decision -- an interactive opponent, a position probe -- has to run the Python
        agents, which is a different implementation from the one whose rating the index
        reads: playing a game to find out what it is like and then scoring it would be
        exercising two different players.
        """
        return self.core.select_move(_pack(state), agent, values, seed)

    def select_round(
        self,
        state: GameState,
        seat: int,
        agent: tuple[Any, ...],
        values: list[float],
        seed: int = 0,
        announced: int | None = None,
    ) -> tuple[int | None, list[float], int, int]:
        """Schema 26 (Q918b): what this agent plays for ``seat`` in a round on this board,
        the mixed strategy over that seat's legal moves it drew it from (empty for
        ``random``), the nodes it spent and its last finished pass. ``None`` for the move
        where either player has none.

        Schema 27 (Q935b): in a spy game's announced round the announcer is asked with
        ``announced`` absent and the answerer with the move the announcer showed; both
        choices are pure, so the strategy is the chosen move alone (`docs/plan-spy.md` §3.2)."""
        return self.core.select_round(_pack(state), seat, agent, values, seed, announced)

    def select_move_depth(
        self,
        state: GameState,
        agent: tuple[Any, ...],
        values: list[float],
        seed: int = 0,
    ) -> tuple[int | None, int]:
        """What the agent would play here, and how deep it got to decide it."""
        return self.core.select_move_depth(_pack(state), agent, values, seed)

    def root_scores(
        self,
        state: GameState,
        values: list[float],
        nodes: int,
        max_depth: int,
        goal: float = 0.0,
        immune: float = 0.0,
    ) -> list[float]:
        """Score every legal move at a full window, in generation order.

        Not a player: a playing search returns a bound for every move except the best,
        which is fine for choosing and useless for asking how close the alternatives
        were. See ``rust/src/probe.rs``.

        ``goal`` is the evaluator's goal weight, in mean non-royal pieces. It is passed
        rather than defaulted inside the core because there is one definition of that
        constant and it lives in ``agents/evaluation.py``. ``immune`` is the immunity
        term's weight (``evaluation.IMMUNE_PIECES``), passed for the same reason.
        """
        return self.core.root_scores(_pack(state), values, nodes, max_depth, goal, immune)

    def play_from(
        self,
        state: GameState,
        agents: tuple[tuple[Any, ...], tuple[Any, ...]],
        values: list[float],
        seed: int,
        collect_series: bool = False,
    ) -> tuple[Any, ...]:
        """Finish a game already in progress, and report it as ``play_game`` would."""
        return self.core.play_from(_pack(state), agents, values, seed, collect_series)

    # ---------------------------------------------------------------------- hash

    def hash(self, state: GameState) -> int:
        """Zobrist hash of the position, maintained incrementally by ``apply``."""
        return state.key

    def _compute_hash(self, state: GameState) -> int:
        """The slow definition of the hash: a full scan, and the reference the
        incremental update in ``apply`` is tested against."""
        return self.core.compute_hash(_pack(state))

    # --------------------------------------------------------------- convenience

    def perft(
        self,
        state: GameState,
        depth: int,
        threads: int = 1,
        stop_at_terminal: bool = False,
    ) -> int:
        """Count leaf nodes at ``depth``. The engine's correctness workhorse.

        ``threads`` splits the root moves across a thread pool. The count is a sum, so
        the answer does not depend on how it was divided -- but the default stays
        serial, because the callers that matter are tests comparing against published
        numbers and they run many small perfts at once.

        ``stop_at_terminal`` stops descending once a position is already decided. It
        makes no difference to a game that ends only when someone runs out of moves,
        which is why chess never needed it -- but a ruleset that ends on arrival or on a
        count of checks keeps generating moves in positions that are already over, and
        every other implementation has stopped.
        """
        return self.core.perft(
            _pack(state), depth, stop_at_terminal=stop_at_terminal, threads=threads
        )

    def describe_move(self, state: GameState, move: int) -> str:
        """Human-readable move text, used by records and the visualiser."""
        if move & MOVE_COMMANDED:
            # Q1007c (`docs/plan-schema-31.md` §3.7): a commanded move is the same move
            # written with a ``~`` before the piece, which is the opponent's (``~Nb8-c6``);
            # the origin names the piece, so the text stays unique.
            return "~" + self.describe_move(state, move & ~MOVE_COMMANDED)
        spec = self.spec
        origin, target, kind, promoted = decode_move(move)
        if kind == MOVE_DROP:
            # A drop has no origin: the piece comes from a hand, and the encoding sets
            # origin equal to destination. Reading the board there finds an empty square,
            # which used to name the piece by wrapping -1 around the type count -- every
            # crazyhouse and shogi drop was reported as the last piece type moving from
            # a square to itself.
            # ``@`` rather than ``*``, which is how the interpreter this replaced wrote
            # it: a record written before the port and one written after should not need
            # different readers.
            return (
                f"{spec.piece_ids[promoted % spec.n_types]}@"
                f"{_square_name(target, spec.width)}"
            )
        if kind == MOVE_FOLD:
            # Schema 22: a fold's own move names its group and end, as a momentum end's
            # does (``2a:e3-f3``); a piece may be standing on either square.
            portals = spec.portals
            assert portals is not None
            end = portals.end_types[move_fold(move)]
            return (
                f"{spec.piece_ids[end]}:{_square_name(origin, spec.width)}-"
                f"{_square_name(target, spec.width)}"
            )
        piece = spec.piece_ids[spec.type_of(state.board[origin])]
        if kind in (MOVE_NORMAL, MOVE_PROMOTION) and move_fold(move) and self._has_folds:
            return self._fold_text(state, move)
        if kind == MOVE_ROTATE:
            # A turn goes nowhere, so the "origin-dash-target" shape would read as a
            # piece moving to the square it is already on. It is written as the piece,
            # its square, and the direction it now points -- which is what a player
            # would say out loud, and what a reader of the record needs in order to know
            # what every later move of that piece means.
            return (
                f"{piece}{_square_name(origin, spec.width)}"
                f"^{FACING_NAMES[facing_quarter_turns(promoted, spec.n_facings)]}"
            )
        if kind == MOVE_PORTAL:
            return self._transit_text(state, move)
        if spec.portals is not None and spec.owner_of(state.board[origin]) == WILD:
            # Schema 19 (N1): a portal's own move names its pair and end, so two portals of
            # one type are told apart: ``1o:e3-f3``.
            return (
                f"{piece}:{_square_name(origin, spec.width)}-"
                f"{_square_name(target, spec.width)}"
            )
        text = f"{piece}{_square_name(origin, spec.width)}"
        # A castling takes nothing, and under file-based castling the king's destination
        # can be its own rook's square -- so the occupant of the target is not the test.
        took = kind in (MOVE_EN_PASSANT, MOVE_SWEEP) or (
            kind != MOVE_CASTLE and state.board[target]
        )
        text += "x" if took else "-"
        text += _square_name(target, spec.width)
        if kind == MOVE_SWEEP:
            # Schema 13: the pieces a sweep passed over, which two positions cannot name.
            swept = [
                _square_name(square, spec.width)
                for square in spec.sweep_path(state.board, origin, target)
                if state.board[square]
            ]
            text += f" (sweeps {', '.join(swept)})"
        # A strike is generated as a capture and written as one, so "Rh8xf8" would say
        # the rook moved to f8 when under `stay` it never left h8 -- and a reader with
        # no way to tell the two apart cannot follow the game at all. The suffix names
        # what the capture actually did, and it is appended rather than replacing the
        # capture text because the move *is* the same move.
        # Q539: a contact that turns the attacker into something else overrides a
        # promotion on the same move (`strike_writes` reads the promoted piece and replaces
        # it), so the "=X" a promotion word would print is not what stands there.
        promotion_overridden = False
        if spec.strike is not None and state.board[target]:
            # Keyed by the pair since schema 10: a contact's outcome depends on who is
            # arriving as well as on who is standing there, and a `strike` fills the
            # attacker's column with -1, so this reads schema 6 exactly as it did.
            mover = spec.type_of(state.board[origin])
            victim = spec.type_of(state.board[target])
            cell = mover * spec.strike.n_types + victim
            # Schema 11: a friendly target reads the own tables.
            own = spec.owner_of(state.board[origin]) == spec.owner_of(state.board[target])
            onto_table = spec.strike.own_onto_to if own else spec.strike.onto_to
            mover_table = spec.strike.own_mover_to if own else spec.strike.mover_to
            becomes = onto_table[cell] if onto_table else -1
            mover_becomes = mover_table[cell] if mover_table else -1
            flips = not own and bool(spec.strike.mover_flips) and spec.strike.mover_flips[cell]
            if spec.strike.position == "home" and becomes >= 0:
                # Schema 17 (Q540, circe). Where the piece went, and whether its home was
                # free, is the core's answer: the move is played and the square read back.
                if becomes != victim:
                    text += f"->{spec.piece_ids[becomes]}"
                text += self._reborn(state, move, becomes)
            elif becomes == REMOVED_IN_PLACE:
                text += " (stays)"  # schema 17: rifle
            elif becomes >= 0 or mover_becomes >= 0:
                if becomes >= 0:
                    text += f"->{spec.piece_ids[becomes]}"
                if mover_becomes >= 0:
                    promotion_overridden = True
                    text += f"/{spec.piece_ids[mover]}->{spec.piece_ids[mover_becomes]}"
                if spec.strike.position == "stay":
                    text += " (stands fast)"
                elif spec.strike.position == "swap":
                    text += " (swap)"
                elif spec.strike.position == "choose" and becomes >= 0:
                    text += self._placed_on(state, move)
            if flips:
                text += " (changes sides)"  # schema 17: Andernach
        elif (
            kind == MOVE_EN_PASSANT
            and spec.strike is not None
            and spec.strike.position == "home"
        ):
            victim = spec.type_of(state.board[state.ep_victim])
            cell = spec.type_of(state.board[origin]) * spec.strike.n_types + victim
            if spec.strike.onto_to[cell] >= 0:
                text += self._reborn(state, move, spec.strike.onto_to[cell])
        elif (
            kind == MOVE_EN_PASSANT
            and spec.strike is not None
            and spec.strike.position == "choose"
        ):
            # Schema 12: en passant's victim stands off the target, and the landing moves
            # differ only in where it is placed.
            victim = state.board[state.ep_victim]
            cell = spec.type_of(state.board[origin]) * spec.strike.n_types + spec.type_of(victim)
            if spec.strike.onto_to[cell] >= 0:
                text += f"->{spec.piece_ids[spec.strike.onto_to[cell]]}"
                text += self._placed_on(state, move)
        if kind == MOVE_PROMOTION and not promotion_overridden:
            text += f"={spec.piece_ids[promoted]}"
        elif kind == MOVE_CASTLE:
            text += " (castle)"
        elif kind == MOVE_EN_PASSANT:
            text += " e.p."
        return text

    def _transit_text(self, state: GameState, move: int) -> str:
        """A move through a portal (schema 19): origin, the end it went in by, then the exit.

        ``Bc1-d2>e3``, with a capture ``Bc1-d2>xe3`` and a promotion ``Pe7-e8>f8=Q``.
        Each exit is its own move with its own text, so ``find_move`` stays unique.
        """
        spec = self.spec
        origin, target, _, promoted = decode_move(move)
        piece = spec.piece_ids[spec.type_of(state.board[origin])]
        entry = self.portal_entry(state, move)
        text = f"{piece}{_square_name(origin, spec.width)}-"
        text += _square_name(entry, spec.width) if entry is not None else "?"
        text += ">x" if state.board[target] else ">"
        text += _square_name(target, spec.width)
        if promoted:
            text += f"={spec.piece_ids[promoted - 1]}"
        return text

    @property
    def _has_folds(self) -> bool:
        return self.spec.portals is not None and self.spec.portals.folds is not None

    def _start_folds(self) -> tuple[int, ...]:
        """Schema 22: where the fold ends start, in end order; ``()`` without folds."""
        portals = self.spec.portals
        if portals is None or portals.folds is None:
            return ()
        return tuple(square for square in portals.folds.starts if square >= 0)

    def fold_square(self, state: GameState, end: int) -> int | None:
        """Schema 22: where fold end ``end`` (its index among the portal ends) stands in
        ``state``, or ``None`` for an end that is not a fold. Read off ``state.folds``, the
        fold ends' squares in end order (their starts where a hand-built state left it
        empty)."""
        portals = self.spec.portals
        if portals is None or portals.folds is None or portals.folds.starts[end] < 0:
            return None
        squares = state.folds or self._start_folds()
        return squares[sum(1 for square in portals.folds.starts[:end] if square >= 0)]

    def fold_route(self, state: GameState, move: int) -> tuple[int, int] | None:
        """Schema 22: ``(way in, way out)`` -- the squares of the fold a move went through
        and of the end it leads to, which is where the move ends (schema 24) -- or ``None``
        for a move that went through none."""
        portals = self.spec.portals
        _, _, kind, _ = decode_move(move)
        tag = move_fold(move)
        if portals is None or portals.folds is None or not tag or kind not in (
            MOVE_NORMAL, MOVE_PROMOTION,
        ):
            return None
        end = tag - 1
        way_in = self.fold_square(state, end)
        dest = portals.dest[portals.end_types[end]]
        way_out = self.fold_square(state, portals.end_types.index(dest)) if dest >= 0 else None
        if way_in is None or way_out is None:
            return None
        return way_in, way_out

    def _fold_text(self, state: GameState, move: int) -> str:
        """A move through a fold (schema 24, `docs/archive/plan-portals-2.md` §3R.2): its way in, and
        its way out, which is where it ends. ``Ra3-b3~e4`` (in at b3, out on e4); a capture
        ``Ra3-b3~xe4``; a promotion ``Pc6-c7~f8=Q``. A stay is written as any move,
        ``Ra3-b3``. Every text stays distinct, so ``find_move`` keeps working."""
        spec = self.spec
        origin, target, kind, promoted = decode_move(move)
        piece = spec.piece_ids[spec.type_of(state.board[origin])]
        route = self.fold_route(state, move)
        way_in = route[0] if route is not None else origin
        text = f"{piece}{_square_name(origin, spec.width)}-{_square_name(way_in, spec.width)}~"
        text += "x" if state.board[target] else ""
        text += _square_name(target, spec.width)
        if kind == MOVE_PROMOTION:
            text += f"={spec.piece_ids[promoted]}"
        return text

    def portal_entry(self, state: GameState, move: int) -> int | None:
        """The square of the portal end a ``MOVE_PORTAL`` goes in by, read off the board.

        The move word names the end by its type, which is unique on the board, so a stored
        transit means the same move wherever that end stands (schema 19).
        """
        spec = self.spec
        if spec.portals is None:
            return None
        value = spec.encode_piece(WILD, move_portal(move))
        square = state.board.find(bytes((value,)))
        return square if square >= 0 else None

    def landing_square(self, state: GameState, move: int) -> int | None:
        """Where a strike under ``position: "choose"`` places its survivor, or ``None``.

        Read off the compiled landing table by the move's landing index (schema 12).
        """
        spec = self.spec
        strike = spec.strike
        if strike is None or strike.position != "choose":
            return None
        origin, target, kind, _ = decode_move(move)
        struck = state.ep_victim if kind == MOVE_EN_PASSANT else target
        if kind in (MOVE_DROP, MOVE_CASTLE, MOVE_ROTATE) or not state.board[struck]:
            return None
        mover, victim = state.board[origin], state.board[struck]
        cell = spec.type_of(mover) * strike.n_types + spec.type_of(victim)
        own = spec.owner_of(mover) == spec.owner_of(victim)
        onto = strike.own_onto_to if own else strike.onto_to
        if not onto or onto[cell] < 0:
            return None
        return strike.landing[spec.owner_of(mover)][struck][move_landing(move)]

    def _reborn(self, state: GameState, move: int, becomes: int) -> str:
        """Where a struck piece was reborn under ``position: "home"`` (schema 17), as text.

        The home square is the compiled table's; whether the piece stands on it is read
        off the position the core leaves, so this says what happened rather than working
        it out a second time.

        Q1020: in a round (schema 26) one move is not a position, so the core resolves the
        move alone, every piece of the other player standing (``board_alone``), and the text
        says what the move does by itself; what the other move of its round changed is the
        round's to show, not the move's.
        """
        spec = self.spec
        strike = spec.strike
        assert strike is not None
        origin, target, kind, _ = decode_move(move)
        struck = state.ep_victim if kind == MOVE_EN_PASSANT else target
        owner = spec.owner_of(state.board[struck])
        home = strike.rebirth[(owner * spec.n_types + becomes) * spec.n_squares + struck]
        if home >= 0:
            if self.is_round(state):
                mover = spec.owner_of(state.board[origin])
                value = self.board_alone(state, mover, move)[home]
            else:
                value = self.apply(state, move).board[home]
            if value and spec.owner_of(value) == owner and spec.type_of(value) == becomes:
                return f" (reborn {_square_name(home, spec.width)})"
        return " (removed: no free home)"

    def _placed_on(self, state: GameState, move: int) -> str:
        square = self.landing_square(state, move)
        return "" if square is None else f" (placed on {_square_name(square, self.spec.width)})"

    def play(self, state: GameState, moves: Iterable[int]) -> GameState:
        for move in moves:
            state = self.apply(state, move)
        return state

    def find_move(self, state: GameState, text: str) -> int:
        """Look up a legal move by the text ``describe_move`` would give it.

        Tests and the CLI both need to name a move without knowing the bit layout.
        Raises ``LookupError`` rather than returning ``None`` so a typo in a test
        fixture fails loudly instead of quietly testing the null move.
        """
        for move in self.legal_moves(state):
            if self.describe_move(state, move) == text:
                return move
        raise LookupError(f"no legal move {text!r}; legal: {self.move_texts(state)}")

    def move_texts(self, state: GameState) -> list[str]:
        return [self.describe_move(state, move) for move in self.legal_moves(state)]

    def make_state(
        self,
        placement: dict[str, tuple[str, int]],
        side_to_move: int = 0,
        moved: Iterable[str] = (),
        ep_target: str | None = None,
        ep_victim: str | None = None,
        halfmove: int = 0,
        ply: int = 0,
        folds: Iterable[str] = (),
        respawn: Iterable[int] = (),
        heading_back: Iterable[str] = (),
    ) -> GameState:
        """Build a position directly, for tests, puzzles, and CLI setup.

        ``placement`` maps square names to ``(piece id, player)``. Everything placed
        counts as unmoved unless named in ``moved`` -- the convenient default for
        castling tests, and the reason this takes a list of exceptions rather than a
        list of inclusions. ``folds`` (schema 22) names where each fold end stands, in end
        order; left empty, the folds stand where they start. ``respawn`` (generation 36) is
        each respawning half's clock, in ``npcs.respawn`` order; left empty, every clock is
        idle, and a half standing short starts its clock in the next phase.
        ``heading_back`` (generation 38) names the turning-back wild pieces walking their line
        back; the rest walk it forward. A square holding no such piece is dropped.
        """
        spec = self.spec
        board = bytearray(spec.n_squares)
        index_of = {pid: i for i, pid in enumerate(spec.piece_ids)}
        unmoved = 0

        for square_name, (piece_id, player) in placement.items():
            square = self._square_index(square_name)
            board[square] = spec.encode_piece(player, index_of[piece_id])
            unmoved |= 1 << square
        for square_name in moved:
            unmoved &= ~(1 << self._square_index(square_name))

        state = GameState(
            board=bytes(board),
            side_to_move=side_to_move,
            unmoved=unmoved,
            ep_target=self._square_index(ep_target) if ep_target else NO_SQUARE,
            ep_victim=self._square_index(ep_victim) if ep_victim else NO_SQUARE,
            halfmove=halfmove,
            ply=ply,
            # Sized whenever the ruleset has a territory rule, and empty otherwise. A
            # hand-built position with no colouring at all is the one place a
            # zero-length array could reach the core, and the core indexes it per square.
            colour=bytes(spec.n_squares) if spec.territory is not None else b"",
            folds=tuple(self._square_index(name) for name in folds) or self._start_folds(),
            # Schema 26: the initiative where it starts, under that tie rule.
            initiative=self.initial_state().initiative if self.simultaneous else -1,
            respawn=tuple(respawn) or self.initial_state().respawn,
            heading_back=tuple(sorted(
                square for square in map(self._square_index, heading_back)
                if board[square] and spec.owner_of(board[square]) == 2
                and spec.npcs is not None and spec.npcs.on_line(square)
            )),
        )
        key = self._compute_hash(state)
        return state.with_(key=key, repetition=(key,))

    def _square_index(self, name: str) -> int:
        file = ord(name[0].lower()) - ord("a")
        rank = int(name[1:]) - 1
        if not (0 <= file < self.spec.width and 0 <= rank < self.spec.height):
            raise ValueError(
                f"{name!r} is not a square on a {self.spec.width}x{self.spec.height} board"
            )
        return rank * self.spec.width + file


def _square_name(square: int, width: int) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"
