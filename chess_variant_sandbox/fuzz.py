"""Walk the rule space with the generator and the oracles this repository already has.

Generality is otherwise tested by example: hand-picked un-chess-like rulesets, plus a
short-game walk over the corpus. Twenty-five rule keys is a combination space nothing
enumerates, and the project's own hard-won lesson is that **rules that cannot interact in
chess can interact in a variant**. Schema 6 is that lesson in miniature: `obstacles` x
`royal_facing` was caught, `obstacles` x `castling` was not, and the difference was
whether a person thought to ask.

So: compose *n* random operators onto a random corpus seed, validate, compile, and walk a
game against every oracle available. Everything it needs already existed --
`loop.operators.mutate` is a valid-spec generator with 25 operators, and the oracles are
the invariants the test suite already holds engines to.

**Seeded, so a failure is a reproduction.** A run reports the seed of every case it
walked, and `case(seed)` rebuilds exactly that spec from exactly that seed.

## The oracles, and the one this item had to add

The item that asked for this listed four oracles already in the repo: the incremental key
against the slow definition, the make/unmake round trip, perft under a thread count, and
the wrap cyclic-shift isomorphism. Its acceptance test was that the fuzzer should find the
castling-onto-a-wall defect with the fix reverted.

**Those four do not find it, and that is the most useful thing this module has to say.**
All four ask whether the engine agrees *with itself*. A king standing on a wall is
perfectly self-consistent: the key is right, unmake restores it, the perft is the same on
any number of threads, and no geometry is being computed twice. What catches it is an
oracle of a different kind -- **the position obeys a rule the spec stated** -- and the
list did not have one. `no_piece_stands_on_an_obstacle` is that oracle, and it fails on
the reverted tree within a handful of cases.

The general lesson is worth more than the fix: *self-consistency oracles cannot see a rule
that was never enforced.* Every rule that names squares nobody may stand on owes an oracle
here, and the next one to arrive should add its own.

## Coverage

A fuzzer nobody can read the coverage of is a fuzzer that gets deleted. `Coverage` counts
**pairs of simultaneously active rules**, which is the unit the lesson above is stated in
-- 25 keys is 300 pairs, most of them unreachable because no operator can produce them
together and some genuinely refused by the validator. What matters is not the percentage
but the list of pairs a run did *not* reach, because that is what the next bump has to
check by hand.

## Hexagons

Schema 16's `board.shape: "hexagon"` is not a rule, so no operator adds it: `_CANNOT_ENTER`
says only the model crosses it. Every committed ruleset is a grid, so a fuzzer drawing its
parents from `rulesets/` alone would never build one. So a share of the seeds
(`HEXAGON_SHARE`) draws its parent from the archive's hexagons instead, the unjoined one
and the two with seams, and the operators take it from there (`join-the-edges` both joins
a clear pair and removes every seam). The decision has its own stream, so every other seed
builds the case it built before hexagons were added. The obstacle oracles read
`engine.spec.obstacles`, which holds a hexagon's corners, so they cover absent cells with
no change; `board.shape` is counted in the coverage like `piece.facing`.

Run a soak with `python -m chess_variant_sandbox.fuzz --cases 2000`, which prints the report.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD, CompiledSpec, compile_spec
from chess_variant_sandbox.loop.operators import OPERATORS, mutate
from chess_variant_sandbox.spec.canonical import load_spec
from chess_variant_sandbox.spec.complexity import _RULE_ACTIVE, _RULE_COSTS, _RULE_DEFAULTS
from chess_variant_sandbox.spec.schema import validate_spec

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "rulesets"
# Where the hexagons come from: every spec here with `board.shape: "hexagon"`.
HEXAGONS = ROOT / "archive"

# One seed in eight starts from a hexagon. A hexagon case walks in a few tenths of a
# second, so the share costs little, and at 200 seeds it is about twenty-five hexagons,
# joined and unjoined.
HEXAGON_SHARE = 1 / 8

# How many operators to compose onto one seed. One edit is the loop's own step size and
# reaches almost nothing new. Drawn per case, so a run covers the range rather than one
# point of it -- and the range runs to **eight**, which is well past anything the design
# loop would do in one step and is the point. The interesting combinations need two
# specific operators to land in the right order: a wall on a rank a king castles along
# needs the back rank emptied *and then* walled, and at four edits that is rare enough
# that 400 cases do not find it. Measured: the deep pairs arrive several times faster at
# eight, and a case costs the same either way, because the cost is compiling and walking
# rather than mutating.
EDITS = (1, 2, 3, 4, 6, 8)

# How far to walk each case. Forty plies is the item's figure and it is enough to reach a
# capture on every board in the corpus; the effects that matter -- a strike, a blast, a
# castling -- need something to have happened first.
PLIES = 40


class OracleFailure(AssertionError):
    """An oracle said no. Carries the seed, because a seed is a reproduction."""


@dataclass
class Case:
    """One generated ruleset, and how to get it back."""

    seed: int
    parent: str
    operators: tuple[str, ...]
    spec: dict[str, Any]

    def describe(self) -> str:
        return (
            f"seed {self.seed}: {self.parent} + " + " + ".join(self.operators)
            if self.operators
            else f"seed {self.seed}: {self.parent} unmutated"
        )


# Exceptions a player has to be taught that live on a *piece* rather than in `rules`, and
# are therefore absent from `_RULE_COSTS`. Named here so the coverage report can say
# whether the fuzzer ever built one, which is the difference between "this rule is fine"
# and "this rule was never tried".
PIECE_RULES = ("piece.facing",)
# The same for the board: schema 16's shape is a `board` key, and a hexagon is exactly the
# kind of case whose pairs the report should be able to say were never tried.
BOARD_RULES = ("board.shape",)


@dataclass
class Coverage:
    """Which pairs of rules were seen active together, and which were not."""

    pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    singles: Counter[str] = field(default_factory=Counter)
    cases: int = 0
    walked: int = 0
    refused: Counter[str] = field(default_factory=Counter)

    def observe(self, active: set[str]) -> None:
        self.cases += 1
        for rule in active:
            self.singles[rule] += 1
        for pair in itertools.combinations(sorted(active), 2):
            self.pairs[pair] += 1

    def report(self, limit: int = 12) -> str:
        """What was and was not exercised, for the next bump to read."""
        # `_RULE_COSTS` plus the piece properties that are exceptions in their own
        # right. The table is over `rules`, and a universe read straight off it would
        # report full coverage of a feature it never generated.
        every = sorted([*_RULE_COSTS, *PIECE_RULES, *BOARD_RULES])
        possible = list(itertools.combinations(every, 2))
        reached = [pair for pair in possible if self.pairs[pair]]
        missing = [pair for pair in possible if not self.pairs[pair]]
        never = [rule for rule in every if not self.singles[rule]]

        lines = [
            f"{self.cases} specs generated, {self.walked} walked to completion",
            f"rule pairs seen active together: {len(reached)} of {len(possible)}",
        ]
        if never:
            lines.append(
                f"rules never active in any case ({len(never)}): " + ", ".join(never)
            )
        if self.refused:
            lines.append(
                "specs the validator or the engine refused: "
                + ", ".join(f"{why} x{n}" for why, n in self.refused.most_common(6))
            )
        lines.append("most-seen pairs: " + ", ".join(
            f"{a}+{b} x{n}" for (a, b), n in self.pairs.most_common(limit)
        ))
        # The list that matters. Truncated by count rather than dropped, because a
        # reader who wants all of it can ask for a bigger limit.
        shown = missing[:limit]
        lines.append(
            f"pairs never reached ({len(missing)}), first {len(shown)}: "
            + ", ".join(f"{a}+{b}" for a, b in shown)
        )
        return "\n".join(lines)


def active_rules(spec: dict[str, Any]) -> set[str]:
    """Which costed rules this spec actually turns on.

    The same reading `rule_mass` uses, and deliberately so: a rule spelled at its default
    is not a rule, and counting one would make every spec look like it exercised
    `strike` and `wrap`.
    """
    rules = spec.get("rules", {})
    active: set[str] = set()
    # `facing` is a piece property and `_RULE_COSTS` is a table over `rules`, so it would
    # otherwise be invisible here -- and a coverage report that never mentions a rule
    # reads exactly like one that always reaches it.
    if any(
        isinstance(piece, dict) and isinstance(piece.get("facing"), dict)
        for piece in spec.get("pieces", []) or []
    ):
        active.add("piece.facing")
    board = spec.get("board")
    if isinstance(board, dict) and board.get("shape") == "hexagon":
        active.add("board.shape")
    for key in _RULE_COSTS:
        if key not in rules:
            continue
        value = rules[key]
        if key in _RULE_DEFAULTS:
            if value != _RULE_DEFAULTS[key]:
                active.add(key)
        elif key in _RULE_ACTIVE:
            if _RULE_ACTIVE[key](value):
                active.add(key)
        elif value:
            active.add(key)
    return active


@functools.cache
def hexagons(source: Path = HEXAGONS) -> tuple[Path, ...]:
    """The specs in `source` with `board.shape: "hexagon"`, in name order."""
    return tuple(
        path
        for path in sorted(source.glob("*.json"))
        if load_spec(path).get("board", {}).get("shape") == "hexagon"
    )


def case(seed: int, corpus: Path = CORPUS, hexagon_source: Path = HEXAGONS) -> Case | None:
    """Build one case from one seed, or `None` if the draw produced nothing valid.

    `None` is a real answer: `mutate` gives up after its own attempt budget on a spec
    with few local edits available, and a case that does not exist is not a failure.

    Whether the parent is a hexagon is drawn from a stream of its own, so adding the
    hexagons changed no other seed's case.
    """
    rng = random.Random(seed)
    paths = sorted(corpus.glob("*.json"))
    if not paths:
        return None
    if random.Random(seed ^ 0x4E6A).random() < HEXAGON_SHARE:
        paths = list(hexagons(hexagon_source)) or paths
    parent = rng.choice(paths)
    spec = load_spec(parent)
    applied: list[str] = []
    for _ in range(rng.choice(EDITS)):
        child = mutate(spec, rng, operators=OPERATORS)
        if child is None:
            break
        spec = child.spec
        applied.append(child.operator)
    return Case(seed=seed, parent=parent.stem, operators=tuple(applied), spec=spec)


# ------------------------------------------------------------------------ the oracles
#
# Each takes a live engine and a position and raises on a violation. Split into named
# functions rather than inlined so that a failure names the invariant it broke, and so
# that the two *kinds* -- does the engine agree with itself, does the position obey the
# spec -- can be told apart in the list.


def legal_is_a_subset_of_pseudo_legal(engine: Engine, state: Any) -> None:
    legal, pseudo = set(engine.legal_moves(state)), set(engine.pseudo_legal_moves(state))
    extra = legal - pseudo
    if extra:
        raise OracleFailure(f"{len(extra)} legal moves are not pseudo-legal")


def the_key_matches_the_slow_definition(engine: Engine, state: Any) -> None:
    if state.key != engine._compute_hash(state):
        raise OracleFailure("the incremental key drifted from a full recompute")


def the_verdict_does_not_depend_on_being_handed_the_moves(engine: Engine, state: Any) -> None:
    if engine.terminal(state) != engine.terminal(state, engine.legal_moves(state)):
        raise OracleFailure("terminal disagrees with itself when handed a move list")


def unmake_restores_what_make_changed(engine: Engine, state: Any) -> None:
    """A perft crossed against the same count taken from fresh states.

    `perft` walks with make/unmake inside the core and reuses one position; `apply`
    builds a new one per move and can carry no damage forward. If unmake left anything
    behind, the two disagree -- which is the only way to see it from this side of the
    boundary, since `unmake` is not exposed.
    """
    moves = engine.legal_moves(state)
    if not moves:
        return
    fresh = sum(len(engine.legal_moves(engine.apply(state, move))) for move in moves)
    if engine.perft(state, 2) != fresh:
        raise OracleFailure("perft over make/unmake disagrees with a walk over apply")


def threads_do_not_change_a_perft(engine: Engine, state: Any) -> None:
    if engine.perft(state, 2, threads=1) != engine.perft(state, 2, threads=4):
        raise OracleFailure("a perft depends on how the work was divided")


def no_piece_stands_on_an_obstacle(engine: Engine, state: Any) -> None:
    """The oracle the four self-consistency ones cannot replace.

    A wall is somewhere nothing may *be*. `_compile_square` dissolves it into shorter
    rays and dropped leaps, so nothing downstream can see it -- and anything in the core
    that works a square out by arithmetic instead of reading a table is blind to it. That
    is exactly how a king came to castle onto one.
    """
    obstacles: frozenset[int] = getattr(engine.spec, "obstacles", frozenset())
    if not obstacles:
        return
    standing = [square for square in obstacles if state.board[square]]
    if standing:
        raise OracleFailure(
            f"a piece is standing on an obstacle at square(s) {sorted(standing)}"
        )


def no_move_targets_an_obstacle(engine: Engine, state: Any) -> None:
    """The same rule one step earlier, on the move rather than on its effect.

    Worth having separately: a move generated onto a wall and then filtered out for some
    other reason would leave the position oracle satisfied, and the generator would still
    be wrong.
    """
    obstacles: frozenset[int] = getattr(engine.spec, "obstacles", frozenset())
    if not obstacles:
        return
    bad = [move for move in engine.legal_moves(state) if ((move >> 8) & 0xFF) in obstacles]
    if bad:
        raise OracleFailure(f"{len(bad)} legal moves land on an obstacle")


def every_piece_is_in_an_orientation_it_may_be_in(engine: Engine, state: Any) -> None:
    """No piece is standing in a facing its type cannot turn to.

    Item 5's finding applied to schema 8. `facing` does not name squares, so the plan
    expected it to owe no oracle -- but it names *states*, which is the same class one
    level up: a piece with two axes has a forward and a backward and no sideways, and a
    piece with no facing at all has exactly one orientation. Every one of the four places
    a piece can acquire a value it did not have -- a promotion, a drop, a strike survivor,
    a rotation -- could write one outside that set, and none of it moves a perft, because
    a piece in an impossible orientation still generates a legal-looking move list.

    Cheap when nothing turns: one integer comparison and out.
    """
    spec = engine.spec
    if not spec.has_facing:
        return
    bad = []
    for square in range(spec.n_squares):
        value = state.board[square]
        if value == 0:
            continue
        facing = spec.facing_of(value)
        piece_type = spec.type_of(value)
        if facing % spec.facing_step(piece_type):
            bad.append((square, spec.piece_ids[piece_type], facing))
    if bad:
        raise OracleFailure(
            f"{len(bad)} piece(s) stand in an orientation their type cannot be in: {bad[:3]}"
        )


def every_npc_stands_on_its_own_route(engine: Engine, state: Any) -> None:
    """Every wild piece is on a route square, and no route holds two.

    Schema 9's `rules.npcs` names squares a piece stands on -- the class item 5 said owes
    an oracle here -- and the phase that walks them is the first code in the core that
    writes the board when nobody is to move. Three things it could get wrong that no
    self-consistency oracle would see: a wild piece stepping off its route, two wild
    pieces on one loop (one stepped onto the other rather than waiting), and a wild
    byte appearing where no route runs. Cheap when nothing is wild: one `is None`.
    """
    spec = engine.spec
    npcs = spec.npcs
    if npcs is None:
        return
    routes_seen: dict[int, int] = {}
    for square in range(spec.n_squares):
        value = state.board[square]
        if value == 0 or spec.owner_of(value) != WILD:
            continue
        if npcs.next[square] < 0:
            raise OracleFailure(f"a wild piece stands on square {square}, where no route runs")
        if spec.type_of(value) != npcs.piece[square]:
            raise OracleFailure(
                f"the wild piece on square {square} is a "
                f"{spec.piece_ids[spec.type_of(value)]!r}, and that route belongs to a "
                f"{spec.piece_ids[npcs.piece[square]]!r}"
            )
        # Walk to a canonical square of this route -- its lowest -- so two wild pieces on
        # the same loop meet in one key whatever squares they stand on.
        lowest = square
        cursor = npcs.next[square]
        while cursor != square:
            lowest = min(lowest, cursor)
            cursor = npcs.next[cursor]
        if lowest in routes_seen:
            raise OracleFailure(
                f"two wild pieces walk one route, on squares {routes_seen[lowest]} and {square}"
            )
        routes_seen[lowest] = square


def the_points_are_the_bounties_of_the_npcs_that_are_gone(engine: Engine, state: Any) -> None:
    """The two tallies add up to the bounties of the routes with no wild piece on them.

    A wild piece leaves the board by being taken and by nothing else -- not by a blast
    (refused beside `npcs`), not by a strike (refused), never by another wild piece -- and
    every taking pays exactly its route's bounty to somebody. So the sum of the tallies is
    the sum of the bounties of the empty routes, whoever collected them. Which player
    collected which is not recoverable from the board, and that is why the tally is state.

    **Fails loudly, not quietly, on a state with no tally.** Q9b lifts the core's refusal
    and lands `points` in the same change; a wild ruleset reaching this oracle on a state
    that has no `points` is exactly the intermediate state that must not pass.
    """
    spec = engine.spec
    npcs = spec.npcs
    if npcs is None:
        return
    points = getattr(state, "points", None)
    if points is None:
        raise OracleFailure("a wild ruleset is being played on a state that carries no tally")
    standing: set[int] = set()
    for square in range(spec.n_squares):
        value = state.board[square]
        if value and spec.owner_of(value) == WILD:
            standing.add(square)
    owed = 0.0
    for start, _ in npcs.starts:
        route = [start]
        cursor = npcs.next[start]
        while cursor != start:
            route.append(cursor)
            cursor = npcs.next[cursor]
        if not any(square in standing for square in route):
            owed += npcs.bounty[start]
    if abs(sum(points) - owed) > 1e-9:
        raise OracleFailure(
            f"the tallies sum to {sum(points):g} and the empty routes are worth {owed:g}"
        )


def a_piece_that_entered_its_promotion_zone_has_promoted(engine: Engine, state: Any) -> None:
    """Under `promote_on_entry`, no piece with a mandatory promotion rule stands in its own
    zone unless the opening array put it there and it has not moved since.

    Schema 9's second key names *squares*, the class item 5 said owes an oracle: once the
    flag is on, a zone is somewhere a piece of the promoting type cannot stand after
    arriving by any route -- a move in promotes (the rule is mandatory), and an entry
    promotes (the flag). What a position alone cannot tell apart is a piece that arrived
    from one the setup placed there, so the `unmoved` bit decides: every route onto a
    square clears it -- an ordinary arrival, a drop, a strike survivor -- and a piece that
    still carries it is standing where it started. A piece the setup put in its zone and
    later turned in place would read as a violation, so a square the opening board holds
    the same owner and type on is exempt; that is the one blind spot, and a narrow one.

    Optional rules are skipped: a pawn that moved into a shogi zone and declined stands
    there by right, and nothing on the board says how it got there. Cheap when the flag is
    off: one `is None`.
    """
    spec = engine.spec
    entry = spec.entry_promotion
    if entry is None:
        return
    mandatory = {rule.piece_type for rule in spec.promotions if rule.mandatory}
    initial = spec.initial_board
    bad = []
    for square in range(spec.n_squares):
        value = state.board[square]
        if value == 0 or spec.owner_of(value) == WILD:
            continue
        player, piece_type = spec.owner_of(value), spec.type_of(value)
        if piece_type not in mandatory or entry.at(player, piece_type, square) < 0:
            continue
        if state.unmoved >> square & 1:
            continue
        before = initial[square]
        if before and spec.owner_of(before) == player and spec.type_of(before) == piece_type:
            continue
        bad.append((square, spec.piece_ids[piece_type], player))
    if bad:
        raise OracleFailure(
            f"{len(bad)} piece(s) stand unpromoted in a promotion zone they entered: {bad[:3]}"
        )


# Two kinds, and the distinction is the finding this module exists to record. The first
# five ask whether the engine agrees with itself; the rest ask whether the position
# obeys a rule the spec stated. A rule that was never enforced is invisible to the first
# kind, however many of them there are.
SELF_CONSISTENCY = (
    legal_is_a_subset_of_pseudo_legal,
    the_key_matches_the_slow_definition,
    the_verdict_does_not_depend_on_being_handed_the_moves,
    unmake_restores_what_make_changed,
    threads_do_not_change_a_perft,
)
RULE_OBEDIENCE = (
    no_piece_stands_on_an_obstacle,
    no_move_targets_an_obstacle,
    every_piece_is_in_an_orientation_it_may_be_in,
    every_npc_stands_on_its_own_route,
    the_points_are_the_bounties_of_the_npcs_that_are_gone,
    a_piece_that_entered_its_promotion_zone_has_promoted,
)
ORACLES = SELF_CONSISTENCY + RULE_OBEDIENCE


def walk(spec: dict[str, Any], seed: int, plies: int = PLIES) -> int:
    """Play `spec` for `plies`, checking every oracle at every ply. Returns plies played.

    The expensive oracles -- the two perfts -- run on a sample of plies rather than all
    of them, because each is a depth-2 search and a soak that runs them everywhere walks
    a tenth as far for no more coverage.
    """
    engine = Engine(spec)
    rng = random.Random(seed ^ 0x5EED)
    state = engine.initial_state()
    expensive = (unmake_restores_what_make_changed, threads_do_not_change_a_perft)
    played = 0
    for ply in range(plies):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None or not moves:
            break
        for oracle in ORACLES:
            if oracle in expensive and ply % 8:
                continue
            oracle(engine, state)
        state = engine.apply(state, rng.choice(moves))
        played += 1
    # The final position is checked too: the last move's effects are only visible here,
    # and a strike or a blast is exactly the kind of thing that lands on the last ply.
    for oracle in RULE_OBEDIENCE:
        oracle(engine, state)
    return played


def run(seeds: range | list[int], plies: int = PLIES, corpus: Path = CORPUS) -> Coverage:
    """Walk every seed, raising on the first oracle failure. Returns what was covered."""
    coverage = Coverage()
    for seed in seeds:
        built = case(seed, corpus)
        if built is None:
            continue
        errors = validate_spec(built.spec)
        if errors:
            # A generated spec the validator refuses is the system working. Recorded so
            # a run that refuses everything is visible as such rather than as a clean
            # sheet -- which is what a fuzzer silently generating nothing looks like.
            coverage.refused["invalid"] += 1
            continue
        try:
            compiled: CompiledSpec = compile_spec(built.spec)
        except Exception as refusal:  # noqa: BLE001 - any refusal is a legitimate answer
            coverage.refused[type(refusal).__name__] += 1
            continue
        del compiled
        coverage.observe(active_rules(built.spec))
        try:
            coverage.walked += 1
            walk(built.spec, seed, plies)
        except OracleFailure as failure:
            raise OracleFailure(f"{built.describe()}\n  {failure}") from failure
        except NotImplementedError:
            # `bridge.rust_supports` holding an intermediate state shut: the compiler
            # knows a rule the core does not yet. Counted as a refusal rather than raised,
            # for the reason an invalid spec is -- the system is working -- and named so
            # that a run which refuses every wild spec is visible as such.
            coverage.walked -= 1
            coverage.refused["core cannot play"] += 1
    return coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=200, help="how many seeds to walk")
    parser.add_argument("--start", type=int, default=0, help="the first seed")
    parser.add_argument("--plies", type=int, default=PLIES)
    parser.add_argument("--limit", type=int, default=12, help="pairs to list either way")
    args = parser.parse_args(argv)

    try:
        coverage = run(range(args.start, args.start + args.cases), args.plies)
    except OracleFailure as failure:
        print(f"FAILED\n{failure}", file=sys.stderr)
        return 1
    print(coverage.report(args.limit))
    return 0


if __name__ == "__main__":  # pragma: no cover - a soak runner, not a library path
    raise SystemExit(main())
