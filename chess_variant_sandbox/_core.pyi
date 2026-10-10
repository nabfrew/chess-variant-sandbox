"""Type stubs for the Rust core.

The extension module is built from ``rust/`` and has no Python source to read, so this
file is where the boundary is written down. It is not documentation that can go stale
quietly: ``mypy chess_variant_sandbox`` checks every call in ``engine/engine.py``,
``tournament/play.py`` and ``loop/gates.py`` against it.

A position crosses as a plain tuple of twelve slots, with ``unmoved`` as 32 bytes
because Python's is arbitrary-width and a 16x16 board needs 256 bits.

**Twelve slots, fourteen fields.** PyO3 converts tuples up to twelve elements and no
further, which is a hard ceiling rather than a lint, so slot 11 groups everything
schema 4 added: the colouring, the moves left in this turn, and the turns taken. Schema
9's bounty tally is the first field to meet that ceiling and joins the same group.

The alias only bites where a return value is consumed, which is ``engine.py``'s
``_unpack`` -- so that function takes ``State`` rather than ``tuple[Any, ...]``. It
did not, and the stub sat two fields stale through the whole of schema 4 while mypy
reported success: every parameter here is ``tuple[Any, ...]`` by necessity (``_pack``
builds one), and with nothing consuming the declared shape there was nothing to check
it against.
"""

from typing import Any

# (board, side_to_move, unmoved, ep_target, ep_victim, halfmove, ply, key,
#  repetition, checks, hands,
#  (colour, remaining, turns_taken, points, goal_left_by, portal_bar))
State = tuple[
    bytes, int, bytes, int, int, int, int, int, list[int],
    tuple[int, int], tuple[list[int], list[int]],
    tuple[bytes, int, int, tuple[float, float], int, tuple[int, int]],
]

# What the wild-piece phase did on the way to a position (schema 9): one
# ``(from, to, eaten)`` per wild piece that walked, oldest step first. Empty for every
# ruleset without ``rules.npcs`` and for most plies of the ones with it.
Phase = list[tuple[int, int, int]]

# (kind, nodes, mobility, max_depth, rollout_depth, exploration, goal, weights, values)
#
# Nine slots. The last is stage 30's piece-value override, ruleset-shaped and empty for
# the table the rules imply -- and it is *beside* the ``values`` argument every call
# below takes, not a replacement for it: that one is the derived table, which normalises
# material and orders captures for every player whatever its opinion of the pieces. Only
# this alias changed shape; no signature here did, which is exactly the sort of edit the
# docstring above says mypy reports success on. It was checked by reading.
# Ten and eleven: Q466's immunity weight and Q502e's net store bytes (empty for none).
NativeSpec = tuple[
    str,
    int,
    float,
    int,
    int,
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    float,
    bytes,
]

# (moves, winner, reason, branching, material, captures, nodes, depth_sum, searches,
#  duration_ms). The last three are the search's cost per seat: nodes visited, plies
# completed summed over the game's searches, and how many searches that was.
Record = tuple[
    list[int],
    int | None,
    str,
    list[int],
    list[float],
    bytes,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    float,
]

# (plies, winner, reason, branching total, branching samples)
Playout = tuple[int, int | None, str, int, int]

class Core:
    def __init__(self, tables: dict[str, Any]) -> None: ...
    def initial_state(self) -> State: ...
    def legal_moves(self, state: tuple[Any, ...]) -> list[int]: ...
    def pseudo_legal_moves(self, state: tuple[Any, ...]) -> list[int]: ...
    def apply(self, state: tuple[Any, ...], move: int) -> tuple[State, Phase]: ...
    def coming_phases(self, state: tuple[Any, ...], count: int) -> list[Phase]: ...
    def terminal(
        self, state: tuple[Any, ...], moves: list[int] | None = ...
    ) -> tuple[int | None, str] | None: ...
    def terminal_prefix(
        self, state: tuple[Any, ...]
    ) -> tuple[int | None, str] | None: ...
    def in_check(self, state: tuple[Any, ...], player: int) -> bool: ...
    def compute_hash(self, state: tuple[Any, ...]) -> int: ...
    def perft(self, state: tuple[Any, ...], depth: int, stop_at_terminal: bool = ..., threads: int = ...) -> int: ...
    def play_game(
        self,
        agents: tuple[NativeSpec, NativeSpec],
        values: list[float],
        seed: int,
        collect_series: bool = ...,
    ) -> Record: ...
    # The three calls that answer a question about *one* position. Everything else here
    # is a whole unit of work; these exist because an interactive opponent and the
    # analysis probe are not, and driving the Python agents for them would mean playing
    # a different implementation from the one the index rates.
    def select_move(
        self,
        state: tuple[Any, ...],
        agent: NativeSpec,
        values: list[float],
        seed: int = ...,
    ) -> int | None: ...
    # `select_move`, plus how deep the search got. Depth is a property of the position
    # and the budget together and cannot be computed from outside, so the heuristic
    # pass's depth control has to read it here.
    def select_move_depth(
        self,
        state: tuple[Any, ...],
        agent: NativeSpec,
        values: list[float],
        seed: int = ...,
    ) -> tuple[int | None, int]: ...
    # Q502e: a net's raw output along a line of moves -- (incremental, inputs, fresh) per
    # position -- for the oracle in `agents/nets.py`; and the layout's input count.
    def net_line(
        self, state: tuple[Any, ...], net: bytes, moves: list[int]
    ) -> list[tuple[int, list[int], int]]: ...
    def net_inputs(self) -> int: ...
    def root_scores(
        self,
        state: tuple[Any, ...],
        values: list[float],
        nodes: int,
        max_depth: int,
        goal: float = ...,
    ) -> list[float]: ...
    # Q665: the minimax player's own root value, move and completed depth on each state,
    # one stream per state from `seeds` (as `select_move`'s seed); a training label's
    # teacher, never called by play. The answer does not depend on `threads`.
    def search_values(
        self,
        states: list[tuple[Any, ...]],
        agent: NativeSpec,
        values: list[float],
        seeds: list[int],
        threads: int = ...,
    ) -> list[tuple[float, int, int] | None]: ...
    def play_from(
        self,
        state: tuple[Any, ...],
        agents: tuple[NativeSpec, NativeSpec],
        values: list[float],
        seed: int,
        collect_series: bool = ...,
    ) -> Record: ...
    def run_tournament(
        self,
        agents: list[NativeSpec],
        values: list[float],
        tasks: list[tuple[int, int, int]],
        collect_series: bool = ...,
        threads: int = ...,
    ) -> list[Record]: ...
    def random_playouts(
        self,
        games: int,
        seed: int = ...,
        time_budget: float = ...,
        threads: int = ...,
    ) -> list[Playout]: ...

def native_agent_kinds() -> list[str]: ...

# Hash of the `rust/src/` sources this extension was compiled from, baked in by
# `build.rs`. `chess_variant_sandbox.measurement` compares it against the same hash of the source on
# disk, which is how a build that predates an edit to the core becomes an error rather
# than a wrong number.
BUILD_FINGERPRINT: str
NET_INPUTS_VERSION: int

# Whether the extension was built with `-C debug-assertions=on`, which turns on the
# checks `game.rs` carries against the two things it memoises. False for every ordinary
# build, including the release build every measurement is taken on.
DEBUG_ASSERTIONS: bool

# Every key `tables.rs` reads out of the dict `bridge.build_tables` sends, off the one
# registry its readers are spelled out of. `bridge.core_table_mismatch` compares it with
# what the package writes, which is how a core built before a table was added becomes an
# error rather than a game played under the rules it already knew.
TABLE_KEYS: tuple[str, ...]
