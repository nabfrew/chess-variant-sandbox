"""The free half of the operator set: local edits, a cheap screen, and a distance.

The loop had one operator and it was the expensive one, which meant there was no notion
of a *local* move in spec space and therefore no landscape at all -- only a sequence of
samples from a conditioned prior. These tests hold down the three properties that make
the mechanical half usable: every child is valid and different, the screen keeps every
real game, and the distance is zero only for descriptions that really are identical.
"""

from __future__ import annotations

import copy
import functools
import itertools
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from retired_specs import hex_glinski_promoting

from chess_variant_sandbox.loop.operators import OPERATORS, SAME_GAME, mutate, refusal
from chess_variant_sandbox.loop.screen import screen
from chess_variant_sandbox.spec import load_spec, schema, spec_hash, transform, validate_spec
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.schema import (
    CLOCK_BOUNDS,
    REPETITION_NEVER,
    SCHEMA_VERSION,
    SCORING_COUNTS,
    is_simultaneous,
    opening_counts,
)

ROOT = Path(__file__).resolve().parents[1]
RULESETS = ROOT / "rulesets"
CORPUS = sorted(path.stem for path in RULESETS.glob("*.json"))

# Everything committed and playable, which is what a design run actually meets: the
# calibration corpus *and* the archive. Every spec with `rules.npcs` is in `archive/`
# (the seven `wolves` specs), and three of the four operator defects Q85 fixed were only
# visible on specs `CORPUS` does not hold, so the sweep below reads both directories.
ALL_SPECS = sorted(
    (path for folder in ("rulesets", "archive") for path in (ROOT / folder).glob("*.json")),
    key=lambda path: path.stem,
)

# Fixed, and six of them, because the point is a deterministic answer over the whole
# corpus rather than a sample: 27,876 children over 101 specs and 46 operators at Q746,
# which made it one test per spec (`test_every_operator_on_each_committed_spec`); 11,257 children over 70 specs and 34 operators at Q483, 63
# refused (all `resize-the-board`'s) and 249 as the parent's game -- twelve fewer than Q436's
# 11,269, all `promote-what-arrives` on `stones` and `three-stones`, whose flag could never
# fire beside a hand only the stone's stock fills; before that, the same 11,362 children every run over 71 specs and 33
# operators at Q327, 64 refused by the validator and all `resize-the-board`'s (67 before
# Q102 moved a promotion `ranks` band with the far edge), and 253
# refused as the parent's game, all unchanged by Q110's inert-rule settle (11,055 over 70 specs and 32 operators at Q224; 9,275 before later
# showcases and Q224's `merge-your-own`, 8,197 before Q101's
# seven showcases joined the corpus, 7,381 before Q108's three new operators, 7,401 before
# Q98 taught `toggle-a-rule` which piece en passant is about). `docs/reference/bugs-investigations.md`, "The
# mutation operators were losing rather a lot", is the table those numbers belong to, and
# it carries the two denominators -- operators and specs -- that move them.
_SWEEP_SEEDS = range(6)

# Committed on purpose and not games: `dull` has no strategy in it, and hierarchy chess
# cannot be won at all, because nothing outranks the king. A gate that kept them would
# not be a gate.
#
# `rps-v1` is deliberately *not* here. Its designer rejected it for draws and the smoke
# gate agrees with him, but this rung does not: searching players win half of its eight
# games. Both readings are right about different players, which is what having two rungs
# is for -- and the disagreement is recorded rather than tuned away.
_NOT_REAL_GAMES = {"dull", "rps-hierarchy-chess"}

# The current bump's inventory collector, resolved *by version* rather than imported by
# name, so that the next schema inherits `TestEveryFeatureOfThisSchemaIsReachable` below
# instead of re-deriving it. The naming convention is `_schema_<number word>_uses` and
# `new-move-primitive` states it; `test_this_schema_has_an_inventory_collector` is what
# fails if a bump spells it something else.
#
# `None` rather than an `AttributeError` at import, because a collection error would take
# the refusal tests down with the reachability ones and they are about different things.
_NUMBER_WORDS = {
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 21: "twenty_one", 22: "twenty_two", 23: "twenty_three", 24: "twenty_four",
    25: "twenty_five", 26: "twenty_six", 27: "twenty_seven",
}
_COLLECTOR = f"_schema_{_NUMBER_WORDS.get(SCHEMA_VERSION, SCHEMA_VERSION)}_uses"
_uses: Callable[[dict], set[str]] | None = getattr(schema, _COLLECTOR, None)


def _inventory(spec: dict) -> set[str]:
    """What this spec uses of the current bump, or nothing if the collector is missing."""
    return _uses(spec) if _uses is not None else set()


@pytest.fixture(scope="module")
def specs() -> dict:
    return {name: load_spec(RULESETS / f"{name}.json") for name in CORPUS}


@dataclass(frozen=True)
class _Child:
    """One operator applied to one spec at one seed, and what came of it.

    ``errors`` is `operators.refusal`'s answer, the whole of what `mutate` would discard
    the child for: the validator's errors, or ``SAME_GAME`` alone for a child that is its
    parent's game again. ``invalid`` and ``same_game`` tell the two apart.

    ``gained`` and ``lost`` are the current bump's inventory delta -- what the collector
    resolved above says about the child that it did not say about the parent, and the
    reverse. They ride along here rather than in a sweep of their own because the
    expensive half is `validate_spec` and the reachability audit wants exactly the answers
    the refusal tests already paid for.
    """

    operator: str
    spec: str
    seed: int
    summary: str | None
    errors: tuple[str, ...]
    raised: str | None
    gained: frozenset[str] = frozenset()
    lost: frozenset[str] = frozenset()
    survived: frozenset[str] = frozenset()
    same_army: bool = False

    @property
    def same_game(self) -> bool:
        """Refused for being the parent's game again, by `mutate` and not the validator."""
        return self.errors == (SAME_GAME,)

    @property
    def invalid(self) -> bool:
        """Refused by the validator."""
        return bool(self.errors) and not self.same_game


def _rotates(spec: dict) -> bool:
    """Whether `relate.identity_key` compiles this spec's children to read their orientation.

    `relate._orientation_played` compiles a ``"rotate"`` spec as written and with every
    square stated (one and a half compiles since Q866, `compile.compile_two_readings`) to
    ask whether rotating is a rule or a spelling, and returns at once for any other
    orientation. On a drop or hex game that is 0.2-0.4 s a child. Q746
    measured the corpus sweep at 217 s of one core over 101 specs, about 180 s of it the
    13 rotating parents' `identity_key`; without it those 13 cost 7 s.
    """
    return (spec.get("setup") or {}).get("orientation") == "rotate"


def _refusal_without_the_compile(parent: dict, child: dict) -> list[str]:
    """`operators.refusal` less its `relate.identity_key` half: the validator, then the hash.

    What the fast suite asks of a rotating parent (`_rotates`); the slow tier asks the
    whole refusal (`test_the_rotating_specs_under_the_whole_refusal`). It refuses a
    subset of what `refusal` refuses -- a child `identity_key` alone would call the parent's
    game reads as valid -- so the validator's verdicts, the raises and every total loss the
    validator or the hash causes are the same; only a parent handed back under another
    spelling waits for the slow tier. `test_the_refusal_mutate_applies_is_the_one_the_sweep_counts`
    pins the two halves this repeats.
    """
    errors = validate_spec(child)
    if errors:
        return errors
    return [SAME_GAME] if spec_hash(child) == spec_hash(parent) else []


def _sweep_over(
    parents: dict[str, dict],
    refuse: Callable[[dict, dict], list[str]] = refusal,
) -> list[_Child]:
    """Every operator against every one of ``parents``, at the fixed seeds."""
    children: list[_Child] = []
    for name, parent in parents.items():
        before = _inventory(parent)
        army = {piece.get("id") for piece in parent.get("pieces", [])}
        for operator in OPERATORS:
            for seed in _SWEEP_SEEDS:
                child = copy.deepcopy(parent)
                try:
                    summary = operator.apply(child, random.Random(seed))
                except Exception as exc:  # noqa: BLE001 -- the point is to report any of them
                    children.append(
                        _Child(operator.name, name, seed, None, (), f"{type(exc).__name__}: {exc}")
                    )
                    continue
                errors = () if summary is None else tuple(refuse(parent, child))
                after = _inventory(child) if summary is not None and not errors else before
                children.append(
                    _Child(
                        operator.name,
                        name,
                        seed,
                        summary,
                        errors,
                        None,
                        gained=frozenset(after - before),
                        lost=frozenset(before - after),
                        # Which rules the child still carries, so "crossed this feature's
                        # axis" can be told apart from "deleted the rule it rides on".
                        survived=frozenset(
                            # Schema 26: `"simultaneous": {}` is a whole rule, not an empty one.
                            key for key, value in (child.get("rules") or {}).items()
                            if value or (key == "simultaneous" and isinstance(value, dict))
                        )
                        if summary is not None and not errors
                        else frozenset(),
                        same_army={piece.get("id") for piece in child.get("pieces", [])} == army,
                    )
                )
    return children


# The corpus sweep, one committed spec at a time (Q746). It was one module fixture of
# every operator over every committed spec: 27,876 children over 101 specs at Q746, 220 s
# in one xdist worker that finished 215 s after the other three, holding every child. Now
# each spec is its own test, so xdist deals them across the workers and nothing outlives
# its spec; the questions that are about the whole corpus at once (is the documented
# refusal still made somewhere, can an operator enter a feature anywhere) search it and
# stop at the first answer. The parametrisation reads file names only, so collection
# loads no spec.
_SPEC_PATHS = {path.stem: path for path in ALL_SPECS}


@functools.cache
def _cheap_first() -> list[tuple[str, dict]]:
    """Every committed spec, the ones whose children are cheap to refuse first.

    The order the whole-corpus questions search in. Order changes what a search costs,
    never what it finds. Loaded once per worker and shared, so a reader deep-copies
    before it edits, as `_sweep_over` and `_first_child` do.
    """
    loaded = [(path.stem, load_spec(path)) for path in ALL_SPECS]
    return sorted(loaded, key=lambda pair: (_rotates(pair[1]), pair[0]))


def _children_of(name: str, parent: dict, exact: bool = True) -> list[_Child]:
    """Every operator against one parent at the fixed seeds, refused as `mutate` would.

    ``exact=False`` refuses a rotating parent's children without `identity_key`'s compile
    (`_refusal_without_the_compile`); any other parent is refused exactly either way, since
    only a rotating spec pays for that half.
    """
    refuse = refusal if exact or not _rotates(parent) else _refusal_without_the_compile
    return _sweep_over({name: parent}, refuse)


def _first_child(
    found: Callable[[dict, dict], bool],
    operators: tuple[Any, ...] = OPERATORS,
    likely: Callable[[dict], bool] = lambda spec: False,
) -> str | None:
    """Where one of ``operators`` first makes a child ``found`` holds of, corpus-wide.

    Every committed spec (`_cheap_first`'s order, the ``likely`` ones before the rest) at
    the sweep's seeds, stopping at the first ``found(parent, child)``, so an existence
    question costs what it takes to answer and a missing answer costs the whole corpus.
    ``None`` when no child satisfies it.
    """
    for name, parent in sorted(_cheap_first(), key=lambda pair: not likely(pair[1])):
        for operator in operators:
            for seed in _SWEEP_SEEDS:
                child = copy.deepcopy(parent)
                if operator.apply(child, random.Random(seed)) is None:
                    continue
                if found(parent, child):
                    return f"{operator.name} on {name} seed {seed}"
    return None


class TestMutation:
    @pytest.mark.parametrize(
        "name",
        [
            "chess",
            "xiangqi",
            "gardner",
            "dull",
            # The rotating case. Shogi's forty children cost 6 s, each refused through two
            # compiles (`relate._orientation_played`); dobutsu's 0.1 s (Q749). Shogi's
            # stays, in the slow tier, and the per-spec sweep above asks the validator of
            # every operator on shogi in the fast suite.
            "dobutsu",
            pytest.param("shogi", marks=pytest.mark.slow),
        ],
    )
    def test_every_child_is_valid_and_different(self, specs, name) -> None:
        """The contract. An operator that emits invalid specs is a way of wasting a
        validation; one that emits the parent again is a way of wasting a tournament."""
        rng = random.Random(7)
        parent = specs[name]
        parent_hash = spec_hash(parent)
        made = 0
        for _ in range(40):
            child = mutate(parent, rng)
            if child is None:
                continue
            made += 1
            assert validate_spec(child.spec) == []
            assert child.hash != parent_hash
            assert child.operator
            assert child.summary
        assert made >= 30, f"only {made} of 40 edits of {name} produced a child"

    def test_the_parent_is_never_modified(self, specs) -> None:
        """A mutation is a new spec, not an edit in place: the archive holds the parent."""
        parent = specs["chess"]
        before = spec_hash(parent)
        rng = random.Random(1)
        for _ in range(20):
            mutate(parent, rng)
        assert spec_hash(parent) == before

    def test_the_same_seed_gives_the_same_child(self, specs) -> None:
        """Determinism, for the same reason everything else here has it: a run has to be
        reproducible from its seed, and an operator drawing from module-level randomness
        would quietly break that."""
        first = mutate(specs["chess"], random.Random(11))
        second = mutate(specs["chess"], random.Random(11))
        assert first is not None and second is not None
        assert first.hash == second.hash
        assert first.summary == second.summary

    def test_the_operators_are_reachable(self, specs) -> None:
        """A named operator that never fires is dead weight pretending to be variety.

        `mutate` draws its operator uniformly, so what can leave one out is an operator
        that never produces a child `mutate` keeps. Each operator is asked on its own,
        parent by parent and seed by seed, until it fires (Q749). It used to be one
        shared draw of 300 children a parent until all had fired, 8 s in the gate, most
        of it shogi's refusals; asked this way it is the same question, does not move
        when an operator is added before another, and stops at each operator's first
        child.
        """
        # Schema 21's three portal operators reshape or filter portals a game already has
        # (`open-a-portal` is the one that puts them on), so the archived Wormhole is
        # their parent here. Schema 23's `share-a-walk` adds a wild piece to a walk a game
        # already has (Wolves). `promote-on-leaving` needs a piece that can leave its
        # zone (Makruk's pawn); `either-face`, `move-a-placement` and
        # `promote-what-arrives` a game that rotates (Dobutsu, then Shogi). Schema 25's
        # `fix-a-facing` fixes or frees a facing a game already has (chess with turning
        # pawns).
        parents = [specs[name] for name in ("chess", "crazyhouse", "gardner")]
        parents += [load_spec(ROOT / "archive" / f"{name}.json") for name in ("wormhole", "wolves")]
        parents += [specs[name] for name in ("makruk", "dobutsu", "shogi")]
        parents.append(load_spec(RULESETS / "chess-oriented-pawns.json"))
        # Schema 26's `set-a-collision` and `break-the-tie` work on a game played in rounds,
        # and none is committed until Q918e: the Gardner test spec stands in.
        parents.append(load_spec(ROOT / "tests" / "_simultaneous_gardner.json"))

        def fires(operator) -> bool:
            return any(
                mutate(parent, random.Random(seed), operators=(operator,), attempts=1)
                is not None
                for parent in parents
                for seed in range(10)
            )

        missing = sorted(operator.name for operator in OPERATORS if not fires(operator))
        assert not missing, f"never fired: {missing}"

    @pytest.mark.parametrize(
        "name",
        [
            "chess",
            "xiangqi",
            "gardner",
            # The rotating case: each identity is two compiles. Shogi's thirty children
            # were 10 s of the test's 11 (Q749), dobutsu's are 0.3 s; shogi's run in the
            # slow tier, beside `test_the_rotating_specs_under_the_whole_refusal`.
            "dobutsu",
            pytest.param("shogi", marks=pytest.mark.slow),
        ],
    )
    def test_a_child_is_never_its_parent_under_another_spelling(self, specs, name) -> None:
        """A new hash is not a new game, so `mutate` asks `relate.identity_key` as well.

        `change-symmetry` can write a symmetry the vectors already have -- *none to
        lateral* on a pawn whose captures are already a mirrored pair, *rotate4 to
        dihedral8* on a bishop -- and `add-a-direction` a step the symmetry already
        reaches. Each child hashes differently and compiles to its parent's tables, and
        before Q327 `mutate` compared only the hash, so it handed the loop its own parent
        to play a tournament on: 65 of 210 `change-symmetry` children in G1's reading.
        """
        from chess_variant_sandbox.spec.relate import identity_key

        restyle = tuple(
            operator
            for operator in OPERATORS
            if operator.name in ("change-symmetry", "add-a-direction")
        )
        parent = specs[name]
        mine = identity_key(parent)
        returned = 0
        for seed in range(30):
            child = mutate(parent, random.Random(seed), operators=restyle, attempts=1)
            if child is None:
                continue
            returned += 1
            assert identity_key(child.spec) != mine, (
                f"{name} seed {seed}: {child.operator} returned the parent's game "
                f"({child.summary})"
            )
        # 23-24 of 30 on each parent (Q749); the four together asked for over 40.
        assert returned > 10, f"only {returned} children of {name}; the test no longer reaches"

    def test_the_refusal_mutate_applies_is_the_one_the_sweep_counts(self, specs) -> None:
        """`refusal` is the whole of what `mutate` discards a drawn child for, stated once
        so the corpus sweep below counts exactly what a design run would throw away."""
        from chess_variant_sandbox.spec.transform import rename_everything

        parent = specs["gardner"]
        assert refusal(parent, copy.deepcopy(parent)) == [SAME_GAME]
        twin = rename_everything(copy.deepcopy(parent))
        assert spec_hash(twin) != spec_hash(parent)
        assert refusal(parent, twin) == [SAME_GAME]

        broken = copy.deepcopy(parent)
        broken["board"]["width"] = 0
        assert refusal(parent, broken) == validate_spec(broken) != []

        child = copy.deepcopy(parent)
        was = parent["rules"].get("stalemate")
        child["rules"]["stalemate"] = "loss" if was != "loss" else "win"
        assert refusal(parent, child) == []

    def test_a_known_game_is_not_rediscovered(self, specs) -> None:
        """A local edit can easily reproduce something the archive already holds.

        Measured on a ten-generation run before this existed: one edit -- changing the
        win condition -- was rediscovered five times, and each cost a whole generation to
        the duplicate check. Drawing again costs a validation.
        """
        rng = random.Random(4)
        parent = specs["chess"]
        first = mutate(parent, rng)
        assert first is not None

        for _ in range(20):
            again = mutate(parent, rng, avoid={first.hash})
            assert again is None or again.hash != first.hash

    def test_an_edit_can_be_aimed_at_new_ground(self, specs) -> None:
        """Most edits are intra-cell by construction; three cross an axis on their own.

        Aiming does not *cost* a generation when it misses: every valid child is kept as
        the fallback, so the caller gets the local edit it would have got anyway.
        """
        from chess_variant_sandbox.loop.grid import descriptor

        rng = random.Random(9)
        parent = specs["gardner"]
        here = descriptor(parent)
        # Every cell one step away on the ending axis or the hand axis.
        targets = [
            here[:2] + (1,) + here[3:],
            here[:3] + (1,) + here[4:],
            here[:3] + (2,) + here[4:],
        ]
        landed = 0
        for _ in range(20):
            child = mutate(parent, rng, targets=targets)
            assert child is not None, "aiming must not lose the fallback"
            if descriptor(child.spec) in [tuple(t) for t in targets]:
                landed += 1
        assert landed >= 5, f"only {landed} of 20 aimed edits reached new ground"

    def test_every_grid_axis_has_an_operator_that_can_cross_it(self, specs) -> None:
        """The general form of a bug this project has now had twice.

        `loop/grid.py` declares six axes and files every candidate by all six, and an
        axis no mechanical edit can move is a dimension the *free* operator is blind to
        -- reachable only by a model call, which is the expensive one. The coverage
        number does not notice: it reports which cells are filled, not which ones could
        ever be reached for nothing.

        It has caught two axes so far. The ending axis grew a `score` bin in stage 14
        and a `ground` bin in stage 16, neither with an operator, so a third of it was
        model-only. And the army axis had never had one at all: every edit that touched
        the piece list *removed* a type and nothing added one, so a free run could shrink
        an army forever and never grow one.

        Written against `AXES` rather than against a list of names, so a seventh axis
        fails here until something can reach it.
        """
        import copy

        from chess_variant_sandbox.loop.grid import AXES, descriptor

        parents = ["gardner", "chess", "courier", "shogi", "losalamos"]
        crossed: dict[str, set[str]] = {axis.name: set() for axis in AXES}
        here = {name: descriptor(specs[name]) for name in parents}
        # Forty draws of each operator on each parent, each pair from its own
        # `Random(4)`: the same children as ever, drawn round by round rather than pair by
        # pair, so the search stops as soon as every axis has a crosser (Q746 stopped at
        # whole operators). Every axis is crossed in the second round; drawn pair by pair
        # it took 44 of 46 operators' forty draws, 13 s in the gate (Q749). A blind axis
        # still costs the whole search before it fails.
        draws = {(operator.name, name): random.Random(4) for operator in OPERATORS for name in parents}
        for _ in range(40):
            for operator in OPERATORS:
                for name in parents:
                    if all(crossed.values()):
                        break
                    child = copy.deepcopy(specs[name])
                    rng = draws[operator.name, name]
                    if operator.apply(child, rng) is None or validate_spec(child):
                        continue
                    # Once per child: each call measures the spec's complexity.
                    there = descriptor(child)
                    for index, axis in enumerate(AXES):
                        if here[name][index] != there[index]:
                            crossed[axis.name].add(operator.name)

        blind = sorted(name for name, names in crossed.items() if not names)
        assert not blind, (
            f"no mechanical operator can cross {blind}; the free half of the search "
            f"cannot reach half of {'that axis' if len(blind) == 1 else 'those axes'}"
        )

    def test_what_a_mechanical_edit_picks_is_a_rule_not_a_number(self, specs) -> None:
        """The part of stage 18 that is not mechanical, held to what it says it does.

        `territory.win_at` is 41 on a 9x9 board because a designer chose a majority. An
        operator has no designer, so it needs a rule -- and a rule is testable where a
        number somebody liked is not.
        """
        import copy

        from chess_variant_sandbox.loop.operators import (
            _claim_the_ground,
            _lengthen_the_turn,
            _score_the_clock,
        )

        for name in ("gardner", "chess", "courier"):
            board = specs[name]["board"]
            squares = board["width"] * board["height"]

            spec = copy.deepcopy(specs[name])
            assert _claim_the_ground(spec, random.Random(2))
            territory = spec["rules"]["territory"]
            # A bare majority: the only threshold both sides cannot hold at once, so the
            # rule decides the game by construction rather than by a lucky number.
            assert territory["win_at"] == squares // 2 + 1
            assert territory["win_at"] <= squares, "a threshold the board cannot supply"
            assert validate_spec(spec) == []

            spec = copy.deepcopy(specs[name])
            assert _lengthen_the_turn(spec, random.Random(2))
            # Never without the ramp. Stage 16 measured a two-move turn giving the first
            # seat 0.931 and the 1-2-2-2 ramp putting it back to 0.597, and
            # `first_player_advantage` is scored -- so the unramped version would be an
            # operator manufacturing candidates the index is guaranteed to mark down.
            assert spec["rules"]["turn"] == {"moves": 2, "ramp": [1]}
            assert validate_spec(spec) == []

            spec = copy.deepcopy(specs[name])
            assert _score_the_clock(spec, random.Random(2))
            # Never `region`: it needs a named set of squares, and an operator inventing
            # one is inventing the interesting half of the rule.
            assert spec["rules"]["scoring"]["count"] in ("material", "pieces")
            assert validate_spec(spec) == []

    def test_the_new_endings_undo_cleanly(self, specs) -> None:
        """Each of the three is add-or-remove, in the shape `toggle-a-rule` already uses.

        The one that can get this wrong is territory: a `scoring` rule counting ground is
        a rule with nothing to count once the ground is gone, and the validator rejects
        it by name -- so an operator that removed one without the other would spend a
        draw on a candidate that could never be played.
        """
        import copy

        from chess_variant_sandbox.loop.operators import (
            _claim_the_ground,
            _lengthen_the_turn,
            _score_the_clock,
        )

        for edit in (_claim_the_ground, _lengthen_the_turn, _score_the_clock):
            spec = copy.deepcopy(specs["gardner"])
            before = spec_hash(spec)
            assert edit(spec, random.Random(2))
            assert spec_hash(spec) != before
            assert edit(spec, random.Random(2))
            assert validate_spec(spec) == []
            assert spec_hash(spec) == before, f"{edit.__name__} does not undo itself"

        # Territory first, then a count of it, then territory away again.
        spec = copy.deepcopy(specs["gardner"])
        _claim_the_ground(spec, random.Random(2))
        _score_the_clock(spec, random.Random(2))
        assert spec["rules"]["scoring"]["count"] == "territory"
        _claim_the_ground(spec, random.Random(2))
        assert "scoring" not in spec["rules"]
        assert validate_spec(spec) == []


    def test_the_board_axis_is_reachable_from_a_board_that_is_not_near_an_edge(
        self, specs
    ) -> None:
        """The free operator and the archive it fills must agree about what a step is.

        `resize-the-board` used to change one dimension by one, which moves the area by
        a whole row and sounds local because it is -- but the grid bins boards at 40 and
        80 squares, and from an 8x8 the three reachable shapes are 64 plus or minus
        eight, sixteen short of the nearest edge. Measured over five parents at 200
        draws: only Los Alamos, which starts four squares from an edge, ever crossed.
        A model-free run could not reach a differently sized game however long it ran,
        while the coverage number said the axis was being explored.

        Chess is the case to hold down, not Los Alamos: a parent that cannot get there
        by accident.
        """
        import copy

        from chess_variant_sandbox.loop.grid import descriptor
        from chess_variant_sandbox.loop.operators import _resize

        for name in ("chess", "gardner", "courier"):
            parent = specs[name]
            here = descriptor(parent)[0]
            rng = random.Random(4)
            reached = set()
            for _ in range(200):
                child = copy.deepcopy(parent)
                if _resize(child, rng) is None or validate_spec(child):
                    continue
                reached.add(descriptor(child)[0])
            assert reached - {here}, f"{name} cannot leave board bin {here}"
            # ...and the local step is still there. An operator that *always* crossed
            # would have stopped being local, which is the other half of the contract.
            assert here in reached, f"{name} lost its intra-cell resize"

    def test_a_crossing_resize_is_the_smallest_one_that_crosses(self) -> None:
        """A local move that lands in a neighbouring cell, not a regenerated board.

        The bin edges are read off `grid.AXES` rather than repeated in the operator, so
        this also fails if the two definitions of "large" ever drift apart.
        """
        from chess_variant_sandbox.loop.grid import AXES
        from chess_variant_sandbox.loop.operators import _across_a_board_bin

        bin_of = AXES[0].of
        for width, height in ((8, 8), (5, 5), (12, 8), (9, 9)):
            here = bin_of({"board": {"width": width, "height": height}})
            found = _across_a_board_bin(width, height, random.Random(0))
            assert found is not None, f"{width}x{height} can reach no other bin"
            new_width, new_height = found
            assert bin_of({"board": {"width": new_width, "height": new_height}}) != here
            # One dimension, and no further than it had to go.
            changed = (new_width != width) + (new_height != height)
            assert changed == 1, f"{width}x{height} -> {found} changed both dimensions"
            step = abs(new_width - width) + abs(new_height - height)
            for closer in range(1, step):
                for shape in (
                    (width + closer, height), (width - closer, height),
                    (width, height + closer), (width, height - closer),
                ):
                    if not all(3 <= value <= 16 for value in shape):
                        continue
                    assert bin_of({"board": {"width": shape[0], "height": shape[1]}}) == here, (
                        f"{shape} crosses in {closer} steps but {found} took {step}"
                    )

    def test_a_child_says_what_changed(self, specs) -> None:
        """The archive and the library both show this, and "mutant 47" tells nobody
        anything."""
        child = mutate(specs["chess"], random.Random(5))
        assert child is not None
        assert child.spec["name"].startswith("chess (")
        assert child.summary in child.spec["description"]


class TestEnPassantFollowsTheDoubleStep:
    """The rule is about *the promoting* piece, on both sides of the edit (Q98).

    En passant answers a two-square move by the piece that promotes, so an operator that
    adds the rule has to find one there, and an edit that takes the move away has to
    take the rule with it. Neither is visible in the corpus sweep -- no committed spec
    has a double-stepping rook beside a single-stepping pawn, and none is one clip away
    from losing its double step -- so both are asserted directly here. Both ask
    `schema.can_double_step`, which is the function the validator's own refusal asks.
    """

    @staticmethod
    def _no_double_step(spec: dict) -> dict:
        """Clip every pawn slide to one square. Mutates and returns `spec`."""
        pawn = next(piece for piece in spec["pieces"] if piece["id"] == "P")
        for move in pawn["moves"]:
            if move["kind"] == "slide":
                move["range"] = 1
                move.pop("min_range", None)
        return spec

    def test_the_rule_is_not_offered_to_a_piece_that_cannot_double_step(
        self, specs
    ) -> None:
        """The trap case 1 names (`docs/reference/bugs-investigations.md`, "Rules that declare nothing"), in its own example.

        The guard used to read *any* piece's `range: 2` slide, so a spec whose rook
        double-steps and whose pawn does not was handed an en passant with an empty
        `ep_passes` behind it -- the child then refused and `mutate` drew again in
        silence, 12 of 213 children. So the rook here is given exactly that `range: 2`:
        without it the old guard would decline for the right answer by accident, and the
        test would pass against the defect it is for.
        """
        from chess_variant_sandbox.loop.operators import _toggle_a_rule

        spec = self._no_double_step(copy.deepcopy(specs["chess"]))
        del spec["rules"]["en_passant"]
        next(piece for piece in spec["pieces"] if piece["id"] == "R")["moves"][0]["range"] = 2
        assert validate_spec(spec) == []
        for seed in range(60):
            child = copy.deepcopy(spec)
            _toggle_a_rule(child, random.Random(seed))
            assert not child["rules"].get("en_passant"), f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}"

        # And with the double step back it is offered again, so this is a guard rather
        # than an operator that has quietly stopped reaching the rule at all.
        spec = copy.deepcopy(specs["chess"])
        del spec["rules"]["en_passant"]
        assert any(
            _toggle_a_rule(copy.deepcopy(spec), random.Random(seed)) == "added en passant"
            for seed in range(60)
        )

    def test_the_rule_is_not_offered_where_two_pieces_promote(self, specs) -> None:
        """The other half of the guard, and the refusal that made the trap live.

        `rules.en_passant needs a single promoting piece to be defined against` is
        already in the tree, so it -- not the inert-rule clause -- is what was actually
        refusing those 12 children. Both questions are asked before the rule is offered.
        """
        from chess_variant_sandbox.loop.operators import _toggle_a_rule

        spec = copy.deepcopy(specs["chess"])
        del spec["rules"]["en_passant"]
        spec["rules"]["promotion"] = [
            {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
            {"piece": "N", "rank": "last", "to": ["Q"]},
        ]
        assert validate_spec(spec) == []
        for seed in range(60):
            child = copy.deepcopy(spec)
            _toggle_a_rule(child, random.Random(seed))
            assert not child["rules"].get("en_passant"), f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}"

    def test_the_rule_is_not_offered_inside_a_turn_of_more_than_one_move(
        self, specs
    ) -> None:
        """Q341's half of the pair, on the side that adds the rule.

        With two moves a turn the mover can answer their own double step, so the
        validator refuses the combination and the operator must not offer it. The other
        half is `set_turn`, which drops the rule rather than declining the edit.
        """
        from chess_variant_sandbox.loop.operators import _toggle_a_rule

        spec = copy.deepcopy(specs["chess"])
        del spec["rules"]["en_passant"]
        spec["rules"]["turn"] = {"moves": 2, "ramp": [1]}
        spec["schema_version"] = max(int(spec.get("schema_version", 1)), 4)
        assert validate_spec(spec) == []
        for seed in range(60):
            child = copy.deepcopy(spec)
            _toggle_a_rule(child, random.Random(seed))
            assert not child["rules"].get("en_passant"), f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}"

    def test_lengthening_the_turn_takes_en_passant_with_it(self, specs) -> None:
        """The same pair from the other side, where declining would cost the operator most.

        45 of the 71 library specs declare en passant, so a `_lengthen_the_turn` that
        refused them all would be an operator with almost nothing to do -- which is the
        shape `TestEveryOperatorOverTheWholeCorpus` exists to catch. `set_turn` settles
        the rule away instead, and the sentence the operator returns says so.
        """
        from chess_variant_sandbox.loop.operators import _lengthen_the_turn

        spec = copy.deepcopy(specs["chess"])
        assert spec["rules"]["en_passant"] is True
        sentence = _lengthen_the_turn(spec, random.Random(0))
        assert sentence is not None
        assert "en passant" in sentence
        assert "en_passant" not in spec["rules"]
        assert spec["rules"]["turn"] == {"moves": 2, "ramp": [1]}
        assert validate_spec(spec) == []

    def test_an_edit_that_takes_the_double_step_takes_the_rule(self, specs) -> None:
        """`transform.settle_en_passant`: the repair `drop_piece` already makes, one
        level down.

        A promotion whose piece is gone takes en passant with it; a double step that is
        gone has to do the same, or the child is refused and `mutate` draws again in
        silence -- the class `TestEveryOperatorOverTheWholeCorpus` exists to catch.
        Asked of the promoting piece only: clipping the *rook* changes nothing.
        """
        from chess_variant_sandbox.spec.transform import settle_en_passant

        spec = self._no_double_step(copy.deepcopy(specs["chess"]))
        assert spec["rules"]["en_passant"] is True
        assert validate_spec(settle_en_passant(spec)) == []
        assert "en_passant" not in spec["rules"]

        untouched = copy.deepcopy(specs["chess"])
        rook = next(piece for piece in untouched["pieces"] if piece["id"] == "R")
        rook["moves"][0]["range"] = 1
        settle_en_passant(untouched)
        assert untouched["rules"]["en_passant"] is True

    def test_slide_becomes_leap_takes_the_rule_with_the_move(self, specs) -> None:
        """The operator that reaches it on chess itself.

        A leap passes over nothing, so turning the pawn's double step into one leaves an
        `ep_passes` with no entries -- exactly what `chess-oriented-pawns` was charged
        1.0 for until Q98 repaired it.
        """
        from chess_variant_sandbox.loop.operators import _slide_becomes_leap
        from chess_variant_sandbox.spec.schema import can_double_step

        cleared = 0
        for seed in range(60):
            child = copy.deepcopy(specs["chess"])
            assert child["rules"]["en_passant"] is True
            if _slide_becomes_leap(child, random.Random(seed)) is None:
                continue
            pawn = next(piece for piece in child["pieces"] if piece["id"] == "P")
            if not can_double_step(pawn):
                cleared += 1
                assert "en_passant" not in child["rules"], f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}"
        assert cleared, "never turned the pawn's double step into a leap in 60 seeds"

    def test_clip_a_slider_takes_the_rule_with_the_move(self, specs) -> None:
        """The same, for the operator chess cannot demonstrate.

        Chess spells its double step `range: 2, min_range: 2`, and `_clip_a_slider`
        never goes below a move's own minimum -- so on chess this operator cannot reach
        the rule at all, which is why the corpus sweep never saw it. The other legal
        spelling of the same move is a plain `range: 2`, and that one clips to 1.
        """
        from chess_variant_sandbox.loop.operators import _clip_a_slider
        from chess_variant_sandbox.spec.schema import can_double_step

        base = copy.deepcopy(specs["chess"])
        double = next(
            move
            for move in next(p for p in base["pieces"] if p["id"] == "P")["moves"]
            if move.get("range") == 2
        )
        del double["min_range"]
        assert validate_spec(base) == []

        cleared = 0
        for seed in range(60):
            child = copy.deepcopy(base)
            if _clip_a_slider(child, random.Random(seed)) is None:
                continue
            pawn = next(piece for piece in child["pieces"] if piece["id"] == "P")
            if not can_double_step(pawn):
                cleared += 1
                assert "en_passant" not in child["rules"], f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}"
        assert cleared, "never clipped the pawn's double step in 60 seeds"


class TestAResizedPromotionFollowsTheFarEdge:
    """A promotion zone spelled as ranks keeps its distance from the far edge (Q102).

    `transform.resize_board` used to put a lone rule's ``rank`` on the last rank whatever
    it had been, even when only the width changed, and to leave every other rule where it
    was. So shogi's band of the far three ranks was refused on a board one rank shorter
    (3 of the 67 refusals the corpus sweep counted), makruk's sixth rank became the
    eighth when the board gained a file, and a list rule's far rank fell off a shorter
    board in silence -- the validator bounds a ``ranks`` entry but not a ``rank``, so
    that child was valid and could never promote.

    Everything a spec names by *square* still stays put, and a shrink that strands one is
    still refused: `test_only_the_one_documented_operator_is_ever_refused` says why.
    """

    _STEPS = [(dw, dh) for dw in (-1, 0, 1) for dh in (-1, 0, 1) if (dw, dh) != (0, 0)]

    @staticmethod
    def _rules(spec: dict) -> list[dict]:
        promotion = (spec.get("rules") or {}).get("promotion")
        rules = promotion if isinstance(promotion, list) else [promotion]
        return [rule for rule in rules if isinstance(rule, dict)]

    @classmethod
    def _from_the_far_edge(cls, spec: dict) -> list[list[int]]:
        """Per rule, the first player's zone as distances from the far edge."""
        width, height = spec["board"]["width"], spec["board"]["height"]
        return [
            sorted(
                {
                    height - 1 - square // width
                    for square in schema.promotion_squares(rule, width, height)[0]
                }
            )
            for rule in cls._rules(spec)
        ]

    def test_no_zone_spelled_by_rank_moves_against_the_far_edge(self) -> None:
        """Every committed spec whose promotion is spelled by rank, one step every way.

        Equal distances say all three things at once: a band still ends where it did, a
        width change moves no rank, and a rank on the board stays on it.
        """
        from chess_variant_sandbox.spec.transform import resize_board

        checked = []
        for path in ALL_SPECS:
            parent = load_spec(path)
            rules = self._rules(parent)
            if not rules or any("squares" in rule for rule in rules):
                continue
            before = self._from_the_far_edge(parent)
            for dw, dh in self._STEPS:
                width = parent["board"]["width"] + dw
                height = parent["board"]["height"] + dh
                if not (3 <= width <= 16 and 3 <= height <= 16):
                    continue
                child = resize_board(copy.deepcopy(parent), width, height)
                where = f"{path.stem} at {width}x{height}"
                # A distance the shorter board does not have is gone: `einstein`'s zone is
                # every rank (Q781), and on a board one rank shorter it still is.
                expected = [[d for d in zone if d < height] for zone in before]
                assert self._from_the_far_edge(child) == expected, where
                refused = [error for error in validate_spec(child) if "promotion" in error]
                assert not refused, f"{where}: {refused}"
            checked.append(path.stem)
        # A lone rank, a list of bands, and a list of lone ranks one of which used to
        # fall off: the three spellings the old code got wrong, each on a committed game.
        assert {"makruk", "shogi", "torus-less-extended"} <= set(checked), checked

    def test_the_refused_band_is_a_game_now(self, specs) -> None:
        """The three refusals themselves: shogi one rank shorter, the far three ranks."""
        from chess_variant_sandbox.spec.transform import resize_board

        parent = specs["shogi"]
        assert {tuple(rule["ranks"]) for rule in self._rules(parent)} == {(6, 7, 8)}
        child = resize_board(copy.deepcopy(parent), 9, 8)
        assert validate_spec(child) == []
        assert {tuple(rule["ranks"]) for rule in self._rules(child)} == {(5, 6, 7)}

    def test_a_lone_last_is_written_out_as_it_always_was(self) -> None:
        """Load-bearing spelling. `metrics/mutants.py` builds its board mutants through
        `resize_board`, and the hash reads ``"last"`` and ``15`` as two games, so this
        writes ``"last"`` out as the new last rank exactly as before Q102 -- which is also
        where moving it with the far edge puts it."""
        from chess_variant_sandbox.spec.transform import resize_board

        seen = 0
        for path in ALL_SPECS:
            parent = load_spec(path)
            promotion = (parent.get("rules") or {}).get("promotion")
            if not isinstance(promotion, dict) or promotion.get("rank") != "last":
                continue
            child = resize_board(copy.deepcopy(parent), 10, 12)
            assert child["rules"]["promotion"]["rank"] == 11, path.stem
            seen += 1
        assert seen > 20


class TestEveryOperatorOverTheWholeCorpus:
    """The class test. Three workers each doing something else found three defects in
    `loop/operators.py` on one day, none of them looking for it, and every one of them
    had the same shape: **a child the validator refuses is indistinguishable from an
    operator that had nothing to do.** `mutate` discards an invalid child and draws
    again, which is the right behaviour and is also perfect camouflage -- an operator
    that has never once produced a playable child on half the corpus reads, from outside,
    as an operator that simply did not apply.

    So the question is asked of the whole corpus at once rather than of the five parents
    the tests above use. It found seven more defects than the three it was written for,
    including the two worst: `hand-the-pieces-back` had never emitted a valid child from
    a schema-1 parent (chess among them) because it did not settle the schema stamp, and
    `remove-a-piece` -- the only operator that can shrink an army -- was a total loss on
    25 of the 53 specs because `drop_piece` left six kinds of dangling reference behind.

    Deliberately not "no operator ever emits an invalid child": that is false on purpose
    and the module says why, at the top. A shrunk board can strand a placement no
    transform can rescue, and repairing it by parsing the validator's English is the
    thing this project refuses to build. The three questions below are the ones that
    separate that from a bug.
    """

    # Which operators may be refused, and which may hand back the parent's game: exact in
    # both directions. The subset half is asked of each spec by
    # `test_every_operator_on_each_committed_spec`, the other half by the two tests that
    # carry each entry's reason.
    _REFUSED = frozenset({"resize-the-board"})
    _HANDS_BACK = frozenset({"change-symmetry", "add-a-direction", "clip-a-slider"})

    @staticmethod
    def _no_operator_raises(children: list[_Child]) -> None:
        """The loudest failure and the cheapest to prevent.

        `mutate` does not guard `operator.apply`, so an exception is not a lost candidate
        -- it ends the generation, and in a design run it ends the run. That is
        deliberate: a `try` there would turn every future operator bug into a draw that
        vanishes quietly, which is the entire class this file is about. The guard is this
        test instead.

        `_score_the_clock` raised `KeyError('points')` on any spec with wild pieces (Q57)
        because its table of prose subjects was written before schema 9 added the count
        `transform.set_scoring` picks for exactly those specs.
        """
        raised = [child for child in children if child.raised]
        assert not raised, "operators raised:\n" + "\n".join(
            f"  {child.operator} on {child.spec} seed {child.seed}: {child.raised}"
            for child in raised[:20]
        )

    @staticmethod
    def _no_operator_is_a_total_loss(children: list[_Child]) -> None:
        """The module's own contract, in a test: *"what is checked here is only what
        would make **every** draw a rejection"* -- written twice in `operators.py` and
        true of neither operator that says it, before this.

        An operator that applies to a spec and whose every child that spec produces is
        refused is not being frugal, it is dead on that spec, and it spends a draw of
        `mutate`'s sixteen every time it is picked. The seeds are fixed, so this is a
        determinate fact about the corpus rather than a sample that might be unlucky.

        Refused means either refusal `mutate` makes: an operator whose every child on a
        spec is that spec again is as dead there as one the validator always refuses.
        """
        applied: dict[tuple[str, str], int] = defaultdict(int)
        valid: dict[tuple[str, str], int] = defaultdict(int)
        for child in children:
            if child.summary is None:
                continue
            applied[(child.operator, child.spec)] += 1
            if not child.errors:
                valid[(child.operator, child.spec)] += 1

        dead = sorted(pair for pair in applied if not valid[pair])
        assert not dead, (
            "these operators applied and were refused every time, which is a bug wearing "
            "the costume of an operator that had nothing to do:\n"
            + "\n".join(
                f"  {operator} on {spec}: {applied[(operator, spec)]} draws, 0 valid"
                for operator, spec in dead[:20]
            )
        )

    @classmethod
    def _refused_only_as_documented(cls, children: list[_Child]) -> None:
        """No operator outside `_REFUSED` is refused by the validator (the subset half)."""
        refused = {child.operator for child in children if child.invalid}
        assert refused <= cls._REFUSED, (
            f"operators refused by the validator: {sorted(refused)}; documented: "
            f"{sorted(cls._REFUSED)}. Newly refused {sorted(refused - cls._REFUSED)} is a "
            "defect (see `test_only_the_one_documented_operator_is_ever_refused`)."
        )

    @classmethod
    def _hands_back_only_as_documented(cls, children: list[_Child]) -> None:
        """No operator outside `_HANDS_BACK` hands back the parent's game (the subset half)."""
        found = {child.operator for child in children if child.same_game}
        assert found <= cls._HANDS_BACK, (
            f"operators handing back the parent's game: {sorted(found)}; documented: "
            f"{sorted(cls._HANDS_BACK)}. New {sorted(found - cls._HANDS_BACK)} (see "
            "`test_only_the_documented_operators_hand_back_the_parent`)."
        )

    @classmethod
    def _assert_clean(cls, children: list[_Child]) -> None:
        cls._no_operator_raises(children)
        cls._no_operator_is_a_total_loss(children)
        cls._refused_only_as_documented(children)
        cls._hands_back_only_as_documented(children)

    @pytest.mark.parametrize("name", sorted(_SPEC_PATHS))
    def test_every_operator_on_each_committed_spec(self, name: str) -> None:
        """Every operator at six seeds on one committed spec, and the four questions.

        No operator raises, none is a total loss on the spec, and only the documented
        operators are refused by the validator or hand back the parent's game. Each is a
        question about one spec, so each spec is its own test and xdist deals them across
        its workers (Q746; one module fixture of every spec ran 220 s in one worker).

        A rotating parent's children are refused here without `identity_key`'s compile
        (`_rotates`, `_refusal_without_the_compile`): every verdict but *the parent under
        another spelling* is the same, and that one is
        `test_the_rotating_specs_under_the_whole_refusal`'s, in the slow tier.
        """
        self._assert_clean(_children_of(name, load_spec(_SPEC_PATHS[name]), exact=False))

    @pytest.mark.slow
    def test_the_rotating_specs_under_the_whole_refusal(self) -> None:
        """The four questions on every rotating spec, refused exactly as `mutate` refuses.

        The half the fast suite leaves out (Q746): a child of a rotating parent that only
        `relate.identity_key` calls the parent's game. Asking it compiles each child twice,
        about 180 s of one core over the 13 rotating specs of Q746's corpus.
        """
        rotating = [(name, spec) for name, spec in _cheap_first() if _rotates(spec)]
        assert rotating, "no committed spec rotates; this test asks nothing"
        children = [child for name, spec in rotating for child in _children_of(name, spec)]
        self._assert_clean(children)

    def test_only_the_one_documented_operator_is_ever_refused(self) -> None:
        """Which operators may be refused at all, stated rather than counted.

        The one entry is the validator doing its job on a shape the operator declines to
        pre-check, and it has a written reason:

        * **`resize-the-board`** -- the module docstring's own example. A shrunk board can
          strand what the spec names by square: a placement, a region, a reach goal, a
          wild route, a hex board's promotion squares. A named square means that square,
          and clipping or moving it would change what the rule means, so the child is
          refused and `mutate` draws again (Q102 read all 67 and found one kind that was
          not this: a promotion zone spelled as *ranks* now follows the far edge, see
          `TestAResizedPromotionFollowsTheFarEdge`, which left 64).

        **`toggle-a-rule` was the second entry and is now clean (Q98)**, which is this
        test's own instruction carried out: its en passant guard asked whether *any*
        piece double-steps where the rule is defined against *the promoting* one, and it
        cost 12 of 213 children. It now asks `schema.can_double_step` of the promoting
        piece -- the same function the validator's refusal asks -- so the two readings
        cannot drift. Two operators that could *remove* a double step, `clip-a-slider`
        and `slide-becomes-leap`, take the rule with them through
        `transform.settle_en_passant` rather than becoming the next entry here.

        Exact rather than a subset, in both directions. A new operator that starts
        emitting refused children fails `test_every_operator_on_each_committed_spec` on the
        spec it is refused on; a fix that makes this one clean fails here -- at which
        point the entry is stale and deleting it is the change. This half searches the
        corpus at the sweep's seeds and stops at the first refusal.
        """
        for name in sorted(self._REFUSED):
            operator = next(op for op in OPERATORS if op.name == name)
            assert _first_child(lambda parent, child: bool(validate_spec(child)), (operator,)), (
                f"{name} is documented as refused, and no committed spec refuses it at "
                "the sweep's seeds: the entry is stale, and deleting it is the change"
            )

    @pytest.mark.parametrize("operator", sorted(_HANDS_BACK))
    def test_only_the_documented_operators_hand_back_the_parent(self, operator: str) -> None:
        """Which operators can write an edit that changes no game, stated, with why.

        `mutate` refuses such a child (`operators.SAME_GAME`), so it costs a draw rather
        than a tournament. It is still a draw spent on nothing, so the set is pinned the
        way the validator's is, exactly and in both directions:

        * **`change-symmetry`** -- a symmetry the vectors already have: *none to lateral*
          on a pawn whose captures are already a mirrored pair, *rotate4 to dihedral8* on
          a bishop. The hash moves, which is why this took `relate.identity_key` (Q327).
        * **`add-a-direction`** -- a step the move's symmetry already reaches, such as a
          knight's 2,1 beside its 1,2 under `dihedral8`. The operator compares the vector
          with the written list, not with the directions the list expands to.
        * **`clip-a-slider`** -- a slide clipped to the range it already has, a pawn's
          double step clipped to two. The hash does not move either, so `mutate` has
          always refused this one.

        Measured at Q327 over the corpus at six seeds: 119 of 420, 70 of 415 and 64 of
        378. An operator that learns to skip these deletes its row; a new one that hands
        back its parent adds a row or is fixed. The new one fails
        `test_every_operator_on_each_committed_spec`; this is the other half, each row
        still handing back some committed spec at the sweep's seeds, searched until the
        first one does.
        """
        chosen = (next(op for op in OPERATORS if op.name == operator),)
        assert _first_child(lambda parent, child: refusal(parent, child) == [SAME_GAME], chosen), (
            f"{operator} no longer hands back any committed spec at the sweep's seeds: "
            "its row in `_HANDS_BACK` is stale, and deleting it is the change"
        )


# ------------------------------------------------ the inert-rule refusals, operator by operator
#
# Q110. `schema._check_inert_rules` refuses twelve spellings of a rule that declares nothing,
# and an edit that takes away what one of them acts on hands `mutate` a child it refuses and
# draws again for, in silence. The corpus sweep above cannot see these: at a hundred seeds
# over all 71 specs the only inert refusal any operator produced was a castling king a shrunk
# board had already stranded. So each pair the audit found is built here on a small parent,
# gardner with one thing changed, and `docs/reference/bugs-investigations.md` ("Q110: the
# operators against the twelve clauses") has the table these rows come from.


def _gardner() -> dict:
    spec = load_spec(RULESETS / "gardner.json")
    spec.pop("provenance", None)
    return spec


def _piece(spec: dict, pid: str) -> dict:
    return next(piece for piece in spec["pieces"] if piece["id"] == pid)


def _stamped(spec: dict) -> dict:
    from chess_variant_sandbox.spec.transform import settle_schema_version

    return settle_schema_version(spec)


def _takes_only(spec: dict, allowed: dict[str, list[str]]) -> dict:
    """Every piece captures nothing, except what ``allowed`` says. Mutates ``spec``."""
    for piece in spec["pieces"]:
        piece["captures"] = list(allowed.get(piece["id"], []))
    return spec


def _without_royalty(spec: dict) -> dict:
    """Antichess's ending on gardner: no royal, and the first to lose everything wins."""
    for piece in spec["pieces"]:
        piece.pop("royal", None)
    spec["rules"].update(
        win="none", stalemate="win", extinction={"pieces": "any", "min_count": 1, "result": "win"}
    )
    return spec


def _one_pair_left(rule: str, value: object) -> Callable[[], dict]:
    """Only the knight takes, and only a pawn: one edit from nothing being takeable."""

    def build() -> dict:
        spec = _takes_only(_gardner(), {"N": ["P"]})
        if rule == "must_capture":
            _without_royalty(spec)
        spec["rules"][rule] = value
        return _stamped(spec)

    return build


def _only_a_sweep_takes() -> dict:
    """Every move only moves, except the bishop's sweep, whose path is a capture site."""
    spec = _without_royalty(_gardner())
    for piece in spec["pieces"]:
        for move in piece["moves"]:
            move["mode"] = "move"
    _piece(spec, "B")["moves"][0]["kind"] = "sweep"
    spec["rules"]["must_capture"] = True
    return _stamped(spec)


def _stocked_but_nothing_taken() -> dict:
    """A hand that fills from captures and from a stock, in a game where nothing is taken."""
    spec = _takes_only(_gardner(), {})
    spec["pieces"].append({"id": "S", "name": "stone", "moves": []})
    spec["rules"]["drops"] = {"from_captures": True, "stock": {"S": "unlimited"}}
    return _stamped(spec)


def _counted_but_forbidden() -> dict:
    """``check_limit`` beside ``forbid_check``, live only because a blast can open a line."""
    spec = _gardner()
    spec["rules"].update(check_limit=3, forbid_check=True, explode_on_capture={"radius": 1})
    return _stamped(spec)


def _castling_rook_split_off() -> dict:
    """The castling rook's one square on the king's rank is the one a split takes (``b1``)."""
    spec = _gardner()
    spec["board"] = {"width": 8, "height": 8}
    spec["setup"]["placements"] = [
        {"piece": "P", "squares": ["c3", "d3", "e3", "f3"]},
        {"piece": "R", "squares": ["a2", "h1"]},
        {"piece": "K", "squares": ["e1"]},
    ]
    spec["rules"]["castling"] = {"king": "K", "rook": "R", "king_steps": 2}
    return _stamped(spec)


def _zed(route: Callable[[dict], None], promotes: bool = True) -> Callable[[], dict]:
    """A piece ``Z`` placed nowhere, which stands only by ``route``, and a rule about it."""

    def build() -> dict:
        spec = _gardner()
        spec["pieces"].append(
            {"id": "Z", "name": "zed", "value": 4, "moves": copy.deepcopy(_piece(spec, "N")["moves"])}
        )
        if promotes:
            spec["rules"]["promotion"] = [
                spec["rules"]["promotion"],
                {"piece": "Z", "rank": "last", "to": ["Q"]},
            ]
        else:
            spec["rules"]["reach"] = {"piece": "Z", "squares": ["c3"], "mirror": False, "result": "win"}
        route(spec)
        return _stamped(spec)

    return build


def _by_strike(spec: dict) -> None:
    spec["rules"]["strike"] = {
        "effect": "promote", "position": "stay", "chain": [{"from": "N", "to": "Z"}], "immune": ["K"],
    }


def _by_contact(spec: dict) -> None:
    spec["rules"]["contacts"] = {
        "position": "advance", "links": [{"mover": "N", "onto": "B", "mover_to": "Z"}], "immune": ["K"],
    }


def _by_own_link(spec: dict) -> None:
    spec["rules"]["contacts"] = {
        "position": "advance",
        "links": [{"mover": "P", "onto": "N", "own": True, "mover_to": "Z"}],
        "immune": ["K"],
    }


def _by_stock(spec: dict) -> None:
    spec["rules"]["drops"] = {"from_captures": True, "stock": {"Z": 1}}


def _by_promotion(spec: dict) -> None:
    promotion = spec["rules"]["promotion"]
    first = promotion[0] if isinstance(promotion, list) else promotion
    first["to"] = ["Z", "R", "B", "N"]


def _wild_and_named() -> dict:
    """`archive/wolves` with a second promotion rule, about the wild wolf."""
    spec = load_spec(ROOT / "archive" / "wolves.json")
    spec.pop("provenance", None)
    spec["rules"].pop("en_passant", None)
    spec["rules"]["promotion"] = [
        spec["rules"]["promotion"],
        {"piece": "W", "rank": "last", "to": ["Q"]},
    ]
    return _stamped(spec)


# (clause, parent, operator): every pair Q110 found that reaches a clause by *taking away*
# what the rule acts on. The first element is the case number in the reference record.
_TAKES_IT_AWAY = [
    ("3", _one_pair_left("must_capture", True), "restrict-a-capture"),
    ("3", _one_pair_left("must_capture", True), "remove-a-piece"),
    ("3", _only_a_sweep_takes, "sweep-a-slider"),
    ("4", _one_pair_left("explode_on_capture", {"radius": 1}), "restrict-a-capture"),
    ("4", _one_pair_left("explode_on_capture", {"radius": 1}), "remove-a-piece"),
    ("5", _one_pair_left("drops", {"from_captures": True}), "restrict-a-capture"),
    ("5", _one_pair_left("drops", {"from_captures": True}), "remove-a-piece"),
    ("5", _stocked_but_nothing_taken, "place-a-stone"),
    ("5", _stocked_but_nothing_taken, "remove-a-piece"),
    ("11", _counted_but_forbidden, "set-the-blast"),
    ("12", _castling_rook_split_off, "add-a-piece"),
    ("13", _zed(_by_strike), "soften-the-capture"),
    ("13", _zed(_by_strike), "remove-a-piece"),
    ("13", _zed(_by_contact), "become-what-you-take"),
    ("13", _zed(_by_contact), "remove-a-piece"),
    ("13", _zed(_by_own_link), "merge-your-own"),
    ("13", _zed(_by_stock), "hand-the-pieces-back"),
    ("13", _zed(_by_promotion), "remove-a-piece"),
    ("13", _zed(_by_promotion, promotes=False), "remove-a-piece"),
    ("13", _wild_and_named, "set-a-patrol"),
]


class TestNoEditLeavesARuleWithNothingToActOn:
    """No operator hands back a child the validator refuses as declaring nothing (Q110).

    ``Operator.apply`` runs `transform.settle_inert_rules` after every edit, so a rule
    whose subject the edit took away goes with it, and the sentence says so. Each pair
    below produced a refused child before that, on the parent built for it; the test
    asserts that no draw is refused now **and** that the settle actually fired, so a
    parent that stops reaching its pair fails here rather than passing on nothing.
    """

    SEEDS = range(40)

    @pytest.mark.parametrize(
        ("clause", "build", "operator"),
        _TAKES_IT_AWAY,
        ids=[f"case{clause}-{operator}-{build.__name__}" for clause, build, operator in _TAKES_IT_AWAY],
    )
    def test_an_edit_takes_the_rule_it_emptied_with_it(
        self, clause: str, build: Callable[[], dict], operator: str
    ) -> None:
        parent = build()
        assert validate_spec(parent) == [], validate_spec(parent)
        found = next(one for one in OPERATORS if one.name == operator)
        settled = 0
        for seed in self.SEEDS:
            child = copy.deepcopy(parent)
            summary = found.apply(child, random.Random(seed))
            if summary is None:
                continue
            errors = validate_spec(child)
            assert errors == [], f"case {clause}, {operator}, seed {seed}: {errors}"
            settled += "had nothing left to act on" in summary
        assert settled, f"case {clause}: {operator} never reached the pair in {len(self.SEEDS)} seeds"

    @pytest.mark.parametrize("operator", ["set-the-blast", "hand-the-pieces-back"])
    def test_a_rule_about_capturing_is_not_offered_where_nothing_is_taken(
        self, operator: str
    ) -> None:
        """Cases 4 and 5 from the other side: an operator that *adds* the rule.

        With nothing takeable the added rule is refused on every draw, so the operator
        declines instead, on the module's own rule. The settle would have handed back the
        parent's game, which `mutate` also throws away.
        """
        parent = _stamped(_takes_only(_gardner(), {}))
        assert validate_spec(parent) == []
        found = next(one for one in OPERATORS if one.name == operator)
        for seed in self.SEEDS:
            assert found.apply(copy.deepcopy(parent), random.Random(seed)) is None, seed

    def test_the_win_by_arrival_is_about_a_royal_that_stands(self) -> None:
        """Case 13 from the adding side: a royal declared first and placed nowhere.

        `royal_id` names the first royal declared, and a `reach` about a piece that never
        stands is refused, so on this parent every draw was refused. The operator walks
        the royal that can stand instead.
        """
        parent = _gardner()
        parent["pieces"].insert(
            0,
            {"id": "J", "name": "jester", "royal": True, "value": 0,
             "moves": copy.deepcopy(_piece(parent, "K")["moves"])},
        )
        parent["rules"]["win"] = "royal_capture"
        _stamped(parent)
        assert validate_spec(parent) == []
        found = next(one for one in OPERATORS if one.name == "win-by-arrival")
        for seed in range(6):
            child = copy.deepcopy(parent)
            assert found.apply(child, random.Random(seed)) is not None
            assert child["rules"]["reach"]["piece"] == "K", seed
            assert validate_spec(child) == [], seed


def _refused_as_inert(spec: dict, fragment: str) -> bool:
    return any(fragment in error for error in validate_spec(spec))


def _no_royal(rule: str, value: object) -> Callable[[], dict]:
    def build() -> dict:
        spec = _without_royalty(_gardner())
        spec["rules"][rule] = value
        return spec

    return build


def _with_rules(**rules: object) -> Callable[[], dict]:
    def build() -> dict:
        spec = _gardner()
        spec["rules"].update(rules)
        return spec

    return build


def _nothing_taken(**rules: object) -> Callable[[], dict]:
    def build() -> dict:
        spec = _takes_only(_gardner(), {})
        if "must_capture" in rules:
            _without_royalty(spec)
        spec["rules"].update(rules)
        return spec

    return build


# (case, spec carrying exactly one inert rule, what the validator says, what the settle drops)
_ONE_INERT_RULE = [
    ("1", _with_rules(en_passant=True), "rules.en_passant can generate no capture", ["en_passant"]),
    ("3", _nothing_taken(must_capture=True), "rules.must_capture can never bind", ["must_capture"]),
    ("4", _nothing_taken(explode_on_capture={"radius": 1}), "rules.explode_on_capture can never fire", ["explode_on_capture"]),
    ("5", _nothing_taken(drops={"from_captures": True}), "rules.drops can never fill a hand", ["drops"]),
    ("6", _with_rules(drops={"from_captures": False}), "nothing else fills a hand", ["drops"]),
    ("7", _no_royal("bare_royal", "loss"), "rules.bare_royal is", ["bare_royal"]),
    ("8", _no_royal("check_limit", 3), "rules.check_limit is", ["check_limit"]),
    ("9", _no_royal("forbid_check", True), "rules.forbid_check is", ["forbid_check"]),
    ("10", _no_royal("royal_facing", "forbidden"), 'rules.royal_facing "forbidden"', ["royal_facing"]),
    ("11", _with_rules(check_limit=3, forbid_check=True), 'beside "forbid_check"', ["check_limit"]),
    ("12", _with_rules(castling={"king": "K", "rook": "P", "king_steps": 2}), "rules.castling can never happen", ["castling"]),
    ("13", _zed(lambda spec: None, promotes=False), "rules.reach is about 'Z'", ["reach"]),
    ("13", _zed(lambda spec: None), "rules.promotion[1] is about 'Z'", ["promotion:Z"]),
]


class TestTheSettleIsTheValidatorsOwnReading:
    """`transform.settle_inert_rules` against each of the twelve clauses, one at a time.

    It asks the validator's predicates under the validator's guards, so it must drop
    exactly the rule each clause refuses and leave a spec the validator accepts, and it
    must be a no-op on every spec the validator already accepts. The last test is the
    tripwire for a thirteenth clause: the settle has an arm per clause, and nothing else
    would notice one missing.
    """

    @pytest.mark.parametrize(
        ("clause", "build", "fragment", "gone"),
        _ONE_INERT_RULE,
        ids=[f"case{row[0]}-{row[3][0]}" for row in _ONE_INERT_RULE],
    )
    def test_the_settle_drops_exactly_the_refused_rule(
        self, clause: str, build: Callable[[], dict], fragment: str, gone: list[str]
    ) -> None:
        from chess_variant_sandbox.spec.transform import settle_inert_rules

        spec = build()
        assert _refused_as_inert(spec, fragment), validate_spec(spec)
        assert settle_inert_rules(spec) == gone
        # Stamped only now: a spec refused for its rule validates under no version at all.
        assert validate_spec(_stamped(spec)) == [], f"case {clause}: {validate_spec(spec)}"

    def test_the_settle_changes_no_spec_the_validator_accepts(self) -> None:
        from chess_variant_sandbox.spec.transform import settle_inert_rules

        for path in ALL_SPECS:
            spec = load_spec(path)
            before = copy.deepcopy(spec)
            assert settle_inert_rules(spec) == [], path.stem
            assert spec == before, path.stem

    def test_the_settle_has_an_arm_for_every_clause(self) -> None:
        """Twelve clauses, written as twelve ``errors.append`` sites.

        A new clause is a new way for an operator's child to be refused, and the settle
        will not know about it. Add its arm to `transform._drop_inert`, a row to
        `_ONE_INERT_RULE` and to the reference record's table, then move this number.
        """
        import inspect

        source = inspect.getsource(schema._check_inert_rules)
        assert source.count("errors.append(") == 12


# ------------------------------------ the refusals outside `_check_inert_rules`, operator by operator
#
# Q436. Q110 audited the twelve clauses; the validator's other *can never fire* refusals were
# not part of it. No committed spec declares `promote_on_entry`, so the corpus sweep cannot
# reach its three refusals at all: the operators walked into them from parents one and two
# steps from the corpus (`promote-what-arrives` on a child that gained a hand, a strike or a
# contact table). Each pair is rebuilt here on gardner, and
# `docs/reference/bugs-investigations.md` ("Q436") has the sweep the pairs come from.


def _arrives(route: Callable[[dict], None], second: bool = False) -> Callable[[], dict]:
    """Gardner with `promote_on_entry`, live only by ``route``; ``second`` also promotes B."""

    def build() -> dict:
        spec = _gardner()
        if second:
            spec["rules"]["promotion"] = [
                spec["rules"]["promotion"],
                {"piece": "B", "rank": "last", "to": ["Q"]},
            ]
        route(spec)
        spec["rules"]["promote_on_entry"] = True
        return _stamped(spec)

    build.__name__ = f"arrives_{route.__name__.strip('_')}"
    return build


def _by_hand(spec: dict) -> None:
    spec["rules"]["drops"] = {"from_captures": True}


def _by_hand_short_of_the_zone(spec: dict) -> None:
    """P may be dropped only on ranks 2 to 4, never on its zone; anything else anywhere."""
    squares = [f"{file}{rank}" for rank in (2, 3, 4) for file in "abcde"]
    spec["rules"]["drops"] = {"from_captures": True, "region": {"P": squares}}


def _by_hand_on_the_far_rank(spec: dict) -> None:
    """P may be dropped only on rank 5, which is its zone until the board grows."""
    spec["rules"]["drops"] = {"from_captures": True, "region": {"P": [f"{f}5" for f in "abcde"]}}


def _by_a_demoting_strike(spec: dict) -> None:
    spec["rules"]["strike"] = {
        "effect": "demote", "position": "stay", "chain": [{"from": "N", "to": "P"}], "immune": ["K"],
    }


def _by_a_contact(spec: dict) -> None:
    spec["rules"]["contacts"] = {
        "position": "advance", "links": [{"mover": "N", "onto": "B", "mover_to": "P"}], "immune": ["K"],
    }


def _by_a_merge(spec: dict) -> None:
    spec["rules"]["contacts"] = {
        "position": "advance",
        "links": [{"mover": "P", "onto": "N", "own": True, "mover_to": "B"}],
        "immune": ["K"],
    }


def _by_a_stone(spec: dict) -> None:
    """A hand only a stock fills, and the stock names only the stone. The validator
    accepted the flag beside it until Q483, reading P as droppable anywhere (no region)
    without asking whether P ever reaches a hand; it refuses it now."""
    from chess_variant_sandbox.spec.transform import set_stone

    set_stone(spec, "X")


def _by_a_stone_and_a_stocked_pawn(spec: dict) -> None:
    """The stone's hand, with P in the stock too: P is held from ply zero."""
    _by_a_stone(spec)
    spec["rules"]["drops"]["stock"]["P"] = 2


def _by_a_stone_and_captures(spec: dict) -> None:
    """The stone's hand, filled from captures as well: a taken P goes into it."""
    _by_a_stone(spec)
    spec["rules"]["drops"]["from_captures"] = True


def _wild(*npcs: dict, scoring: str | None = None, without: tuple[str, ...] = ()) -> Callable[[], dict]:
    """Gardner with wild pieces on rank 3, which no army stands on."""

    def build() -> dict:
        from chess_variant_sandbox.spec.transform import drop_piece

        spec = _gardner()
        for pid in without:
            drop_piece(spec, pid)
        spec["rules"]["npcs"] = copy.deepcopy(list(npcs))
        if scoring is not None:
            spec["rules"]["scoring"] = {"count": scoring}
        return _stamped(spec)

    return build


# One wild bishop that pays and one wild knight that does not; take the bishop away.
_PAYING_AND_NOT = _wild(
    {"piece": "B", "route": ["b3"]}, {"piece": "N", "route": ["d3"], "bounty": 0}, scoring="points"
)
_PAYING_AND_NOT.__name__ = "paying_and_not"
# Only the pawn is kept off the wild knight; take the pawn away and it is kept off nobody.
_KEPT_OFF_ONE = _wild({"piece": "N", "route": ["c3"], "immune": ["K", "R"]}, without=("Q", "B"))
_KEPT_OFF_ONE.__name__ = "kept_off_one"

# (refusal, parent, operator): every pair the Q436 sweep found, and the two it predicts
# only from a parent no operator writes (two wild types, one paying; an immunity one
# type short of everybody). E1-E3 are `_check_entry_promotion_coherence`'s three.
#
# Q436's `place-a-stone` row is gone: its parent carried the flag beside a stone-only hand,
# which is itself the refusal Q483 added, so no valid parent reaches that pair any more.
# `TestAHandNoPromotingPieceReaches` holds the refusal and its settle.
_WALKS_INTO_IT = [
    ("E1", _arrives(_by_hand), "remove-a-piece"),
    ("E2", _arrives(_by_hand), "hand-the-pieces-back"),
    ("E2", _arrives(_by_a_demoting_strike), "soften-the-capture"),
    ("E2", _arrives(_by_a_demoting_strike), "remove-a-piece"),
    ("E2", _arrives(_by_a_contact), "become-what-you-take"),
    ("E2", _arrives(_by_a_merge, second=True), "merge-your-own"),
    ("E3", _arrives(_by_hand_short_of_the_zone, second=True), "remove-a-piece"),
    ("E3", _arrives(_by_hand_on_the_far_rank), "resize-the-board"),
    ("points", _PAYING_AND_NOT, "remove-a-piece"),
    ("immune", _KEPT_OFF_ONE, "remove-a-piece"),
]

# What each refusal says, for telling it apart from `resize-the-board`'s design refusals.
_OUTSIDE_FRAGMENTS = {
    "E1": "rules.promote_on_entry needs rules.promotion",
    "E2": "rules.promote_on_entry: nothing enters play",
    "E3": "rules.promote_on_entry can never fire",
    "points": 'rules.scoring.count "points" needs',
    "immune": "protects it from nothing",
}


class TestNoEditLeavesARuleOutsideTheClausesWithNothingToActOn:
    """Q110's guarantee, for the *can never fire* refusals outside `_check_inert_rules`.

    `transform.settle_inert_rules` asks `_check_entry_promotion_coherence`,
    `wild_pieces_pay` and `schema.npc_immune_attackers` too, so the flag, the count or the
    immunity goes with what it needed. Each pair produced a refused child before (Q436);
    the test asserts that none is refused now **and** that the settle fired, so a parent
    that stops reaching its pair fails here rather than passing on nothing.
    `resize-the-board` may still be refused, for a square its shrunk board strands, and
    never for the flag.
    """

    SEEDS = range(40)

    @pytest.mark.parametrize(
        ("refusal_name", "build", "operator"),
        _WALKS_INTO_IT,
        ids=[f"{name}-{operator}-{build.__name__}" for name, build, operator in _WALKS_INTO_IT],
    )
    def test_an_edit_takes_the_rule_it_emptied_with_it(
        self, refusal_name: str, build: Callable[[], dict], operator: str
    ) -> None:
        parent = build()
        assert validate_spec(parent) == [], validate_spec(parent)
        found = next(one for one in OPERATORS if one.name == operator)
        settled = 0
        for seed in self.SEEDS:
            child = copy.deepcopy(parent)
            summary = found.apply(child, random.Random(seed))
            if summary is None:
                continue
            errors = validate_spec(child)
            if operator == "resize-the-board":
                errors = [e for e in errors if _OUTSIDE_FRAGMENTS[refusal_name] in e]
            assert errors == [], f"{refusal_name}, {operator}, seed {seed}: {errors}"
            settled += "had nothing left to act on" in summary
        assert settled, f"{refusal_name}: {operator} never reached the pair in {len(self.SEEDS)} seeds"

    def test_the_clock_is_not_scored_in_points_nobody_pays(self) -> None:
        """The points refusal from the adding side: `score-the-clock` on wild pieces worth
        nothing. `set_scoring` counted points wherever a wild piece walked, and a type worth
        nothing (antichess's M, which `set-a-patrol` can loose) pays nothing, so every draw
        was refused. It now counts what the spec can supply instead."""
        parent = _gardner()
        _piece(parent, "N")["value"] = 0
        parent["rules"]["npcs"] = [{"piece": "N", "route": ["c3"]}]
        _stamped(parent)
        assert validate_spec(parent) == []
        found = next(one for one in OPERATORS if one.name == "score-the-clock")
        child = copy.deepcopy(parent)
        assert found.apply(child, random.Random(0)) is not None
        assert child["rules"]["scoring"]["count"] == "material"
        assert validate_spec(child) == []

    def test_wall_off_a_square_misses_a_wild_route_under_a_rotated_setup(self) -> None:
        """`_check_obstacle_coherence`'s wild-route refusal (*a rule cannot be about a square
        nobody can stand on*), from `wall-off-a-square`. The operator kept a wall's rank
        flip clear where `set_obstacle` mirrors it by the spec's orientation, so under
        `"rotate"` the wall's image could fall on a route's mirror: 28 refused children
        two steps from the corpus, `marchfield` and `shogi` after `set-a-patrol` (Q436).
        Here the route's rotated image is d3 and e3, on the rank the rank flip keeps."""
        parent = _gardner()
        parent["setup"]["orientation"] = "rotate"
        parent["rules"]["npcs"] = [{"piece": "N", "route": ["a3", "b3"], "mirror": True}]
        _stamped(parent)
        assert validate_spec(parent) == []
        found = next(one for one in OPERATORS if one.name == "wall-off-a-square")
        applied = 0
        for seed in self.SEEDS:
            child = copy.deepcopy(parent)
            if found.apply(child, random.Random(seed)) is None:
                continue
            applied += 1
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"
        assert applied


def _refused_outside(**rules: object) -> Callable[[], dict]:
    def build() -> dict:
        spec = _gardner()
        for key, value in rules.items():
            if value is None:
                spec["rules"].pop(key, None)
            else:
                spec["rules"][key] = value
        return spec

    return build


_NOBODY_ON_RANK_FIVE = {"from_captures": True, "region": {"P": [f"{f}{r}" for r in (2, 3, 4) for f in "abcde"]}}

# (refusal, spec carrying exactly one such rule, what the validator says, what the settle drops)
_ONE_RULE_OUTSIDE = [
    ("E1", _refused_outside(promotion=None, drops={"from_captures": True}, promote_on_entry=True),
     "rules.promote_on_entry needs rules.promotion", ["promote_on_entry"]),
    ("E2", _refused_outside(promote_on_entry=True), "nothing enters play except by moving", ["promote_on_entry"]),
    ("E3", _refused_outside(drops=_NOBODY_ON_RANK_FIVE, promote_on_entry=True),
     "rules.promote_on_entry can never fire", ["promote_on_entry"]),
    ("E3-no-hand", _arrives(_by_a_stone), "P never reaches a hand", ["promote_on_entry"]),
    ("E2-empty-hand", _refused_outside(drops={}, promote_on_entry=True),
     "nothing enters play except by moving", ["promote_on_entry"]),
    ("points-no-npcs", _refused_outside(scoring={"count": "points"}), "needs rules.npcs", ["scoring"]),
    ("points-no-bounty", _wild({"piece": "N", "route": ["d3"], "bounty": 0}, scoring="points"),
     "needs a wild piece with a bounty", ["scoring"]),
    ("immune", _wild({"piece": "N", "route": ["c3"], "immune": ["K", "Q", "R", "B", "P"]}),
     "protects it from nothing", ["immune:N"]),
]


class TestTheSettleReadsTheRefusalsOutsideTheClauses:
    """The Q436 arms of `settle_inert_rules`, one refusal at a time, as
    `TestTheSettleIsTheValidatorsOwnReading` holds the twelve clauses; its no-op test
    over every committed spec covers these arms too."""

    @pytest.mark.parametrize(
        ("name", "build", "fragment", "gone"), _ONE_RULE_OUTSIDE, ids=[row[0] for row in _ONE_RULE_OUTSIDE]
    )
    def test_the_settle_drops_exactly_the_refused_rule(
        self, name: str, build: Callable[[], dict], fragment: str, gone: list[str]
    ) -> None:
        from chess_variant_sandbox.spec.transform import settle_inert_rules

        spec = build()
        assert _refused_as_inert(spec, fragment), validate_spec(spec)
        assert settle_inert_rules(spec) == gone
        assert validate_spec(_stamped(spec)) == [], f"{name}: {validate_spec(spec)}"

    def test_every_refusal_of_the_flag_is_about_the_flag(self) -> None:
        """The settle drops `promote_on_entry` on *any* refusal
        `_check_entry_promotion_coherence` makes, which is right only while each of them
        names the flag. A new refusal there that is about something else fails here."""
        import inspect

        source = inspect.getsource(schema._check_entry_promotion_coherence)
        assert source.count("errors.append(") == 3
        assert source.count('"rules.promote_on_entry') == 3


class TestAHandNoPromotingPieceReaches:
    """Q483: E3 asks whether a promoting piece can ever be in a hand.

    It read a promoting piece with no `drops.region` entry as droppable anywhere, so a
    hand only a stock fills (`from_captures: false`, `archive/stones`) was accepted beside
    a `promote_on_entry` that could never fire. A hand fills from the stock and from
    captures, a captured piece as its `promoted_from`, and nothing else; the refusal is
    that neither route reaches any promoting type."""

    STONE_ONLY = (
        "rules.promote_on_entry can never fire: no piece with a promotion rule can enter "
        "play inside its zone -- P never reaches a hand, since rules.drops.from_captures is "
        "false and rules.drops.stock does not name it"
    )

    def test_a_stone_only_hand_is_refused_by_name(self) -> None:
        assert validate_spec(_arrives(_by_a_stone)()) == [self.STONE_ONLY]

    @pytest.mark.parametrize(
        "route", [_by_a_stone_and_a_stocked_pawn, _by_a_stone_and_captures, _by_hand],
        ids=lambda route: route.__name__.strip("_"),
    )
    def test_a_hand_the_promoting_piece_reaches_keeps_the_flag(self, route: Callable[[dict], None]) -> None:
        assert validate_spec(_arrives(route)()) == []

    def test_a_piece_captured_back_as_its_base_never_reaches_a_hand(self) -> None:
        """The other half of the capture route: Q promotes, and a taken Q is held as P."""
        spec = _gardner()
        spec["rules"]["promotion"] = {"piece": "Q", "rank": "last", "to": ["R"]}
        spec["rules"]["drops"] = {"from_captures": True}
        spec["rules"]["promote_on_entry"] = True
        _stamped(spec)
        assert validate_spec(spec) == []
        _piece(spec, "Q")["promoted_from"] = "P"
        assert validate_spec(spec) == [
            "rules.promote_on_entry can never fire: no piece with a promotion rule can enter "
            "play inside its zone -- Q never reaches a hand, since rules.drops.stock does not "
            "name it and a captured Q is held as P"
        ]

    def test_promote_what_arrives_declines_on_stones(self) -> None:
        """The operator asks the validator on a copy, so on `archive/stones` it now has
        nothing to offer instead of a child whose flag can never fire."""
        stones = load_spec(ROOT / "archive" / "stones.json")
        found = next(one for one in OPERATORS if one.name == "promote-what-arrives")
        child = copy.deepcopy(stones)
        assert found.apply(child, random.Random(0)) is None
        assert child == stones

    def test_an_edit_to_a_child_main_accepted_takes_the_flag(self) -> None:
        """`archive/stones` with the flag, the child the operator used to write: any edit
        now settles it, because `_drop_inert` asks the same function."""
        from chess_variant_sandbox.spec.transform import settle_inert_rules

        parent = load_spec(ROOT / "archive" / "stones.json")
        parent["rules"]["promote_on_entry"] = True
        assert any("P never reaches a hand" in error for error in validate_spec(parent))
        assert settle_inert_rules(copy.deepcopy(parent)) == ["promote_on_entry"]
        found = next(one for one in OPERATORS if one.name == "reprice-a-piece")
        child = copy.deepcopy(parent)
        summary = found.apply(child, random.Random(0))
        assert summary is not None and "promotion on entry had nothing left to act on" in summary
        assert "promote_on_entry" not in child["rules"]
        assert validate_spec(child) == []


class TestTheLastStructuralRefusals:
    """Q484: the two refusals outside `resize-the-board` that Q436's sweep left, neither a
    rule with nothing to act on. Each parent is one or two steps from the corpus, the shape
    the sweep found; each test failed on main's code."""

    SEEDS = range(40)

    @staticmethod
    def _children(parent: dict, operator: str) -> list[tuple[int, dict]]:
        found = next(one for one in OPERATORS if one.name == operator)
        out = []
        for seed in TestTheLastStructuralRefusals.SEEDS:
            child = copy.deepcopy(parent)
            if found.apply(child, random.Random(seed)) is not None:
                out.append((seed, child))
        return out

    def test_soften_the_capture_places_the_survivor_on_a_hex_board(self) -> None:
        """*rules.strike.landing.symmetry 'rotate4' ... carry some of them off the hex
        lattice*: `soften-the-capture` crossing `choose` wrote the square board's king step,
        `rotate4`, on `hex-glinski-promoting` with a strike (15 refused children). It now
        writes the six neighbours under `none`."""
        from chess_variant_sandbox.spec.transform import HEX_STEP_LANDING, set_strike

        parent = hex_glinski_promoting()
        set_strike(parent, effect="demote", position="stay")
        _stamped(parent)
        assert validate_spec(parent) == [], validate_spec(parent)
        placed = 0
        for seed, child in self._children(parent, "soften-the-capture"):
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"
            strike = child["rules"].get("strike") or {}
            if strike.get("position") == "choose":
                placed += 1
                assert strike["landing"] == HEX_STEP_LANDING
        assert placed, "soften-the-capture never placed the survivor"

    def test_the_hex_landing_is_every_neighbour_once(self) -> None:
        """Six distinct steps, each a side of the cell in the axial convention `schema`
        documents, so nothing the lattice calls a neighbour is left out."""
        from chess_variant_sandbox.spec.transform import HEX_STEP_LANDING

        steps = {tuple(vector) for vector in HEX_STEP_LANDING["vectors"]}
        assert steps == {(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)}
        assert len(HEX_STEP_LANDING["vectors"]) == 6

    def test_remove_a_piece_keeps_the_only_placed_type(self) -> None:
        """*setup.placements must be a non-empty array*: `dull` with `place-a-stone`'s stone
        in hand places one type, and `remove-a-piece` took it (8 refused children). It now
        takes the stone, the one other candidate, on every seed."""
        parent = load_spec(RULESETS / "dull.json")
        parent.pop("provenance", None)
        parent["pieces"].append({"id": "X", "name": "stone", "moves": []})
        parent["rules"]["drops"] = {"from_captures": False, "stock": {"X": "unlimited"}}
        _stamped(parent)
        assert validate_spec(parent) == [], validate_spec(parent)
        children = self._children(parent, "remove-a-piece")
        assert children
        for seed, child in children:
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"
            assert [piece["id"] for piece in child["pieces"]] == ["G"]

    def test_remove_a_piece_never_takes_horde_white_army(self) -> None:
        """Q937: horde's White has only pawns under extinction `any`; removing the pawn type
        made a child over at ply zero, which the validator now refuses. The pawn is never
        drawn, and every child that is drawn validates."""
        parent = load_spec(RULESETS / "horde.json")
        parent.pop("provenance", None)
        children = self._children(parent, "remove-a-piece")
        assert children
        for seed, child in children:
            assert "P" in {piece["id"] for piece in child["pieces"]}, f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"

    def test_remove_a_piece_keeps_a_summed_extinction_reachable(self) -> None:
        """A small spec where one removal is short of `min_count` and one is not: five grunts
        and two wardens a side under extinction `any`, min_count 5. Taking the grunts leaves
        two; taking the wardens leaves five. Only the wardens are ever taken."""
        parent = load_spec(RULESETS / "dull.json")
        parent.pop("provenance", None)
        parent["pieces"].append(
            {"id": "W", "name": "warden", "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "lateral"}]}
        )
        parent["setup"]["placements"].append({"piece": "W", "squares": ["a2", "e2"]})
        parent["rules"]["extinction"] = {"pieces": "any", "min_count": 5, "result": "loss"}
        parent = _stamped(parent)
        assert validate_spec(parent) == [], validate_spec(parent)
        children = self._children(parent, "remove-a-piece")
        assert children
        for seed, child in children:
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"
            assert [piece["id"] for piece in child["pieces"]] == ["G"], f"seed {seed}"


# --------------------------------------------------------- surface 8, as a standing test
#
# `new-move-primitive` surface 8 -- *if the rule reaches a new kind of game, an operator
# must be able to cross that axis* -- was answered by inspection four bumps running and was
# wrong every time. What a bump owes this file is one row per name its collector can
# return, a parent that already has the feature, and, only when no operator can reach it, a
# `_CANNOT_ENTER` / `_CANNOT_LEAVE` row carrying the reason. The sweep then answers the
# question that used to be judged.
#
# **Nothing here can derive the inventory.** A collector answers *what does this spec use*,
# not *what could I ever say*, so the union of the ``names`` below is the tree's only
# enumeration of the bump -- the same list `describe.py` and
# `reference/schema-showcase-coverage.md` are stated against. What is mechanical is that it
# cannot be left behind: `_FEATURES_ARE_FOR_SCHEMA` fires on the commit that raises
# `SCHEMA_VERSION`, and the author is holding the list at exactly that moment.
_FEATURES_ARE_FOR_SCHEMA = 27


@dataclass(frozen=True)
class _Feature:
    """One feature of the current bump, and the parent that demonstrates it.

    ``host`` is the `rules` key the feature *rides on*. A modifier -- `immune` on a wild
    piece, `by`/`of` on an explosion, `stock` on a hand -- has an axis of its own only
    while its host is still there: an operator that removes every wild piece has not
    crossed `npcs[].immune`, it has crossed `npcs` and taken the modifier down with it. A
    ``host`` of ``None`` says the feature *is* the rule, so removing it is the crossing.

    ``keeps_army`` is the second half of the same question, and it is ``False`` exactly
    once. A crossing normally has to leave the piece list alone, or `remove-a-piece`
    deleting the type a list happens to name counts as reaching that list -- which is
    `_forget_piece` repairing a dangling reference on its way past, not an edit anything
    can ask for. Q51 is the exception because one of its two inventory names, *a piece with
    no moves of its own*, is a fact about the piece list: no edit can put a stone in a game
    without adding one, so requiring the same army there would state a condition that
    cannot be met rather than a gap that can be closed.

    ``parent`` builds the smallest edit to a committed spec that turns the feature on, so
    what the leave arm measures is the operator set and not the parent's other novelties.
    """

    host: str | None
    names: frozenset[str]
    keeps_army: bool
    parent_name: str
    parent: Callable[[], dict]
    # Where `test_an_operator_can_enter_the_feature` looks first (Q891): the specs where the
    # feature is likeliest to be entered most cheaply, as a host rule's presence is for a
    # modifier. Only the order of a search that stops at its first answer; never what it finds.
    likely: Callable[[dict], bool] = lambda spec: False


def _chess() -> dict:
    return load_spec(RULESETS / "chess.json")


def _parent_never() -> dict:
    never = _chess()
    never["schema_version"] = 10
    never["rules"]["repetition_draw"] = REPETITION_NEVER
    return never


def _parent_immune() -> dict:
    wolves = load_spec(ROOT / "archive" / "wolves.json")
    wolves["schema_version"] = 10
    for npc in wolves["rules"]["npcs"]:
        # Every wild piece of a type, because capture permission is a fact about the
        # type: the validator refuses two wild 'W' with different immunities by name.
        npc["immune"] = ["N"]
    return wolves


def _parent_blast() -> dict:
    atomic = load_spec(RULESETS / "atomic.json")
    atomic["schema_version"] = 10
    atomic["rules"]["explode_on_capture"]["by"] = ["Q"]
    return atomic


def _parent_stock() -> dict:
    stock = load_spec(RULESETS / "crazyhouse.json")
    stock["schema_version"] = 10
    stock["pieces"].append({"id": "W", "name": "wall", "value": 1, "moves": []})
    stock["rules"]["drops"]["stock"] = {"W": 2}
    return stock


def _parent_contacts() -> dict:
    contacts = _chess()
    contacts["schema_version"] = 10
    contacts["rules"].pop("en_passant", None)
    contacts["rules"]["contacts"] = {
        "position": "advance",
        "immune": ["K"],
        "links": [{"mover": "P", "onto": "N", "mover_to": "N"}],
    }
    return contacts


def _parent_own() -> dict:
    own = _parent_contacts()
    own["schema_version"] = 11
    own["rules"]["contacts"]["links"] = [
        {"mover": "P", "onto": "R", "own": True, "mover_to": "Q"}
    ]
    return own


def _parent_sweep() -> dict:
    sweep = _chess()
    sweep["schema_version"] = 13
    for piece in sweep["pieces"]:
        if piece["id"] == "R":
            piece["moves"][0]["kind"] = "sweep"
    return sweep


def _parent_hex() -> dict:
    """The showcase with its promotion rule off, so the lattice is the only feature on."""
    hexed = hex_glinski_promoting()
    hexed["rules"].pop("promotion")
    return hexed


def _parent_shaped_zone() -> dict:
    shaped = _chess()
    shaped["schema_version"] = 14
    promotion = shaped["rules"]["promotion"]
    promotion.pop("rank", None)
    promotion["squares"] = ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8", "d7", "e7"]
    return shaped


def _parent_on_leaving() -> dict:
    """Shogi with the flag on the silver alone: the smallest edit that turns it on.

    The committed shogi carries it on all six entries since Q454, so the others are taken
    off here: one flag is what a single `promote-on-leaving` draw can take away.
    """
    shogi = load_spec(RULESETS / "shogi.json")
    shogi["schema_version"] = 15
    for entry in shogi["rules"]["promotion"]:
        entry.pop("on_leaving", None)
        if entry["piece"] == "S":
            entry["on_leaving"] = True
    return shogi


def _parent_native_hexagon() -> dict:
    """`archive/hex-glinski-hexagon`, the walled parent with `board.shape` for its walls."""
    return load_spec(ROOT / "archive" / "hex-glinski-hexagon.json")


def _parent_hexagon_seam() -> dict:
    """The native twin with its clear pair joined, the smallest edit that turns a seam on."""
    seamed = _parent_native_hexagon()
    seamed["rules"]["wrap"] = {"nw_se": "both"}
    return seamed


def _parent_strike(**rule: Any) -> Callable[[], dict]:
    """Chess with one schema-17 strike: the smallest edit that turns the feature on."""

    def build() -> dict:
        spec = _chess()
        spec["schema_version"] = 17
        spec["rules"]["strike"] = {"effect": "capture", "immune": ["K"], **rule}
        return spec

    return build


def _parent_kamikaze() -> dict:
    """Atomic with its blast narrowed to radius 0, which refuses `immune`."""
    atomic = load_spec(RULESETS / "atomic.json")
    atomic["schema_version"] = 17
    atomic["rules"]["explode_on_capture"] = {"radius": 0}
    return atomic


def _parent_keeping_link() -> dict:
    """`archive/merger`'s cousin in one edit: chess's contact parent plus a keeping link.

    The pawn promotes on the last rank, so a link keeping a pawn that takes a rook as a
    pawn overrides that promotion: the smallest edit that turns schema 18's link on.
    """
    spec = _parent_contacts()
    spec["schema_version"] = 18
    spec["rules"]["contacts"]["links"].append({"mover": "P", "onto": "R", "mover_to": "P"})
    return spec


def _parent_walled_rank() -> dict:
    """`archive/omega`, the one game whose castling rank holds walls beyond its rooks."""
    return load_spec(ROOT / "archive" / "omega.json")


def _parent_per_type() -> dict:
    """Chess plus an extinction over five kinds counted apart: 18c's smallest edit."""
    chess = load_spec(RULESETS / "chess.json")
    chess["schema_version"] = 18
    chess["rules"]["extinction"] = {"pieces": ["P", "N", "B", "R", "Q"], "per_type": True}
    return chess


def _parent_only_lost() -> dict:
    """Chess whose pawn promotes only into what its side has lost: 18c's smallest edit."""
    chess = load_spec(RULESETS / "chess.json")
    chess["schema_version"] = 18
    chess["rules"]["promotion"]["only_lost"] = True
    return chess


def _parent_placement() -> dict:
    """Chess whose officers other than the king start in hand: 18d's smallest edit.

    The king stays placed, so the leave arm can take the flag off alone; a stocked royal
    is the showcases' (`placement-chess`, `sittuyin`), and their way off puts the army
    back on the board. Castling goes, since a dropped rook has moved.
    """
    chess = load_spec(RULESETS / "chess.json")
    chess["schema_version"] = 18
    chess["setup"]["placements"] = [
        placement for placement in chess["setup"]["placements"]
        if placement["piece"] in ("P", "K")
    ]
    chess["rules"].pop("castling")
    rank = [f"{file}1" for file in "abcdefgh"]
    chess["rules"]["drops"] = {
        "from_captures": False,
        "placement": True,
        "stock": {"Q": 1, "R": 2, "B": 2, "N": 2},
        "region": {piece: rank for piece in ("Q", "R", "B", "N")},
    }
    return chess


def _parent_either_face() -> dict:
    """Shogi whose hand may drop either face: 18e's smallest edit.

    Shogi's promoted pieces each name their base as `promoted_from`, one face per base,
    so the flag goes on with nothing else changed.
    """
    shogi = load_spec(RULESETS / "shogi.json")
    shogi["schema_version"] = 18
    shogi["rules"]["drops"]["either_face"] = True
    return shogi


def _parent_leaving() -> dict:
    """King of the hill decided by leaving the centre: 18f's smallest edit.

    The committed game already has a `reach` rule with no `safe`, so the flag goes on
    with nothing else changed.
    """
    hill = load_spec(RULESETS / "kingofthehill.json")
    hill["schema_version"] = 18
    hill["rules"]["reach"]["on"] = "leave"
    return hill


def _parent_turned(symmetry: str) -> dict:
    """Glinski's hexagonal chess with the rook's six steps written once and turned: 18b's
    smallest edit, for `rotate6`, and the knight's twelve as one jump for `dihedral12`."""
    hexy = hex_glinski_promoting()
    hexy["schema_version"] = 18
    piece, vectors = ("R", [[1, 0]]) if symmetry == "rotate6" else ("N", [[1, 2]])
    for move in next(p for p in hexy["pieces"] if p["id"] == piece)["moves"]:
        move["vectors"], move["symmetry"] = vectors, symmetry
    return hexy


def _portal_parent(version: int = 19, **pair: object) -> dict:
    """Gardner with one fixed two-way pair on b3 and d3 (`plan-portals.md` §6's Wormhole),
    plus whatever ``pair`` sets: schema 19's smallest edit (20's with ``version``)."""
    gardner = load_spec(RULESETS / "gardner.json")
    gardner["schema_version"] = version
    gardner["pieces"].append({
        "id": "O", "name": "portal", "value": 0,
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                   "mode": "move"}],
    })
    gardner["rules"]["portals"] = [{"piece": "O", "orange": "b3", "blue": "d3", **pair}]
    return gardner


def _ring_parent(**first: object) -> dict:
    """Gardner with a ring of three on the empty third rank, a3 to c3 to e3 and round
    (`plan-portals-2.md` §2.1), plus whatever ``first`` sets on the first end: schema 21's
    smallest edit."""
    gardner = _portal_parent(21)
    gardner["rules"]["portals"] = [{"piece": "O", "ends": [
        {"at": "a3", "to": 1, **first}, {"at": "c3", "to": 2}, {"at": "e3", "to": 0},
    ]}]
    return gardner


def _parent_one_patrol() -> dict:
    """`archive/wolves-patrol` as archived: two wolves on one circuit, the showcase."""
    return load_spec(ROOT / "archive" / "wolves-patrol.json")


def _parent_rear_guard() -> dict:
    """`rulesets/torus-less-extended` as committed: a rear guard of pawns placed facing
    backwards that no move turns, the showcase."""
    return load_spec(RULESETS / "torus-less-extended.json")


def _parent_rounds(**extra: object) -> dict:
    """Gardner played in rounds (`tests/_simultaneous_gardner.json`): schema 26's smallest
    edit, plus whatever ``extra`` sets on the rule."""
    gardner = load_spec(RULESETS / "gardner.json")
    gardner["schema_version"] = 26
    gardner["rules"]["win"] = "royal_capture"
    gardner["rules"]["simultaneous"] = dict(extra)
    return gardner


def _parent_chase(key: str, value: object) -> dict:
    """Gardner in rounds with one of schema 27's chase switches (Q942a), and material scoring
    under `both`, which needs a score to settle by."""
    rounds = _parent_rounds(**{key: value})
    rounds["schema_version"] = 27
    if key == "both":
        rounds["rules"]["scoring"] = {"count": "material"}
    return rounds


def _parent_spy() -> dict:
    """Gardner in rounds with each side's c-pawn a spy (`plan-spy.md` §5's SP1, Q935a): a
    wazir that captures nothing."""
    rounds = _parent_rounds()
    rounds["schema_version"] = 27
    rounds["pieces"].append({
        "id": "S", "name": "spy", "value": 2, "spy": True, "captures": [],
        "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}],
    })
    for placement in rounds["setup"]["placements"]:
        if placement["piece"] == "P":
            placement["squares"] = [square for square in placement["squares"] if square != "c2"]
    rounds["setup"]["placements"].append({"piece": "S", "squares": ["c2"]})
    return rounds


def _parent_beats() -> dict:
    """Rounds where a pawn beats nothing in a collision (§6.8's second showcase idea)."""
    rounds = _parent_rounds()
    next(p for p in rounds["pieces"] if p["id"] == "P")["beats"] = []
    return rounds


def _parent_circle() -> dict:
    """Two armies of different pieces in a circle, so no collision can tie (§3.1, F)."""
    steps = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    pieces = [
        {"id": "R", "moves": steps, "captures": ["s", "r"], "royal": True},
        {"id": "S", "moves": steps, "captures": ["p", "s"]},
        {"id": "P", "moves": steps, "captures": ["r", "p"]},
        {"id": "r", "moves": steps, "captures": ["S"], "royal": True},
        {"id": "s", "moves": steps, "captures": ["P"]},
        {"id": "p", "moves": steps, "captures": ["R"]},
    ]
    return {
        "schema_version": 26, "name": "circle", "board": {"width": 4, "height": 4},
        "pieces": pieces,
        "setup": {"mirror": False, "placements": [
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "P", "player": 0, "squares": ["c1"]},
            {"piece": "r", "player": 1, "squares": ["d4"]},
            {"piece": "s", "player": 1, "squares": ["c4"]},
            {"piece": "p", "player": 1, "squares": ["b4"]},
        ]},
        "rules": {"win": "royal_capture", "simultaneous": {"tie": "impossible"}},
    }


def _parent_own_kinds() -> dict:
    """Two armies of different kinds, each losing by running out of any one of its own
    (`docs/reference/extinction-any-kind.md` §1's spec under schema 27's ``"own"``)."""
    def leap(vector: list[int], symmetry: str) -> list[dict]:
        return [{"kind": "leap", "vectors": [vector], "symmetry": symmetry, "mode": "both"}]

    return {
        "schema_version": 27, "name": "gap", "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "A", "value": 3, "moves": leap([1, 0], "rotate4")},
            {"id": "B", "value": 3, "moves": leap([1, 1], "rotate4")},
            {"id": "C", "value": 3, "moves": leap([1, 2], "dihedral8")},
            {"id": "E", "value": 3, "moves": leap([2, 0], "rotate4")},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "A", "player": 0, "squares": ["b1", "e1"]},
            {"piece": "B", "player": 0, "squares": ["c1", "d1"]},
            {"piece": "C", "player": 1, "squares": ["b6", "e6"]},
            {"piece": "E", "player": 1, "squares": ["c6", "d6"]},
        ]},
        "rules": {"win": "none", "stalemate": "draw", "max_plies": 200, "repetition_draw": 3,
                  "extinction": {"pieces": ["A", "B", "C", "E"], "per_type": "own"}},
    }


def _armies_differ(spec: dict) -> bool:
    """Where `count-each-kind` enters ``"own"``: the two armies share fewer than two kinds."""
    first, second = opening_counts(spec)
    return len(set(first) & set(second)) < 2 and len(set(first) | set(second)) >= 2


# Schema 27's inventory so far (`plan-spy.md` §2.2): Q936's ``"own"``, riding on
# `rules.extinction`; `count-each-kind` enters it on a game whose armies differ and leaves
# it to the summed count. Q942a's three chase switches ride on `rules.simultaneous`, and
# `curb-the-chase` enters and leaves each; Q935a adds its row here. Schema 26's four rows
# (Q918a) retired with the bump; `play-at-once`, `break-the-tie` and `set-a-collision`
# still cross them.
_FEATURES: dict[str, _Feature] = {
    "Q936 per_type own": _Feature(
        "extinction", frozenset({"rules.extinction.per_type.own"}), True, "gap+own",
        _parent_own_kinds, likely=_armies_differ,
    ),
    "Q942a rest": _Feature(
        "simultaneous", frozenset({"rules.simultaneous.rest"}), True, "rounds+rest",
        lambda: _parent_chase("rest", True), likely=is_simultaneous,
    ),
    "Q942a swap": _Feature(
        "simultaneous", frozenset({"rules.simultaneous.swap"}), True, "rounds+swap",
        lambda: _parent_chase("swap", "collision"), likely=is_simultaneous,
    ),
    "Q942a both": _Feature(
        "simultaneous", frozenset({"rules.simultaneous.both"}), True, "rounds+both",
        lambda: _parent_chase("both", "score"), likely=is_simultaneous,
    ),
    # Q935a: the spy rides on the rounds it decides; `plant-a-spy` enters it (making the
    # game one in rounds first where it alternates) and leaves it, and `play-at-once`
    # takes it down with the rounds.
    "Q935a spy": _Feature(
        "simultaneous", frozenset({"pieces.spy"}), True, "rounds+spy", _parent_spy,
        likely=is_simultaneous,
    ),
}

# A feature whose showcase is owed by a named later task, and which that task deletes.
# Exact in both directions, as `_CANNOT_ENTER` is: a showcase that lands fails
# `test_every_feature_of_this_schema_has_a_showcase` until its row goes. Empty since
# Q454 put schema 15's flag on every promotion entry of `rulesets/shogi.json`, which is
# that schema's showcase; empty again since Q772a archived schema 19's seven
# (`archive/wormhole.json` and its arms, Divide, Ferry, Carousel), and since schema 23,
# whose showcase `archive/wolves-patrol.json` landed with the bump (Q850).
_SHOWCASE_OWED: dict[str, str] = {
    # Q936's `"own"` went on 2026-10-17, when Q936s archived its showcase, two armies of
    # different kinds (`archive/riders-and-leapers`; `extinction-any-kind.md` §5, coverage
    # row 72). Empty again at schema 27.
    # Q942a's three chase switches went on 2026-10-17, when Q942c archived S1 to S4
    # (`archive/gardner-in-rounds-resting`, `-where-swaps-collide`, `-scored`,
    # `-against-the-chase`; `plan-round-chase.md` §5, coverage rows 68 to 70).
    # Q935a's spy went on 2026-10-17, when Q935f archived SP1 and SP2
    # (`archive/gardner-with-spies`, `-resting`; `plan-spy.md` §5, coverage row 71).
    # Empty again at schema 26: Q918e archived its four showcases (`archive/gardner-in-rounds`,
    # `-with-meek-pawns`, `-with-the-initiative`, `archive/watering-hole`).
    # Empty again at schema 25: its showcase, `rulesets/torus-less-extended.json`, is the
    # game the bump was for (Q869). Schema 24's fold is shown by `archive/crease.json` (Q844b).
}


# Every feature every schema from 9 on added, by the name its collector returns (Q791).
# The collectors answer *what does this spec use*, never *what could a spec ever say*, so
# the universe is hand-written, as `_FEATURES` is; `test_every_feature_of_every_schema_has_a_showcase`
# reads each collector over the committed and archived games. Inert spellings a collector
# names but that compile to the game that was already there (`board.lattice=square`,
# `board.shape=grid`, `promotion.on_leaving=false`) owe no showcase and are not listed.
_EVERY_SCHEMA_FEATURES: dict[int, frozenset[str]] = {
    6: frozenset({
        'rules.strike.effect "capture"', 'rules.strike.position "advance"',
        "rules.strike.immune naming a non-royal piece",
    }),
    9: frozenset({"rules.npcs", "rules.promote_on_entry", 'rules.scoring.count "points"'}),
    10: frozenset({
        'rules.repetition_draw "never"', "rules.npcs[].immune", "rules.explode_on_capture.by",
        "rules.explode_on_capture.of", "rules.drops.stock", "rules.contacts",
        "a piece with no moves of its own",
    }),
    11: frozenset({"rules.contacts.links[].own"}),
    12: frozenset({
        "rules.strike.position=choose", "rules.strike.landing",
        "rules.contacts.position=choose", "rules.contacts.landing",
    }),
    13: frozenset({"the 'sweep' move kind"}),
    14: frozenset({"board.lattice=hex", "rules.promotion.squares"}),
    15: frozenset({"rules.promotion.on_leaving"}),
    16: frozenset({"board.shape=hexagon", "rules.wrap.hexagon"}),
    17: frozenset({
        "rules.strike.position=stay(capture)", "rules.strike.position=home",
        "rules.strike.switches_sides", "rules.explode_on_capture.radius=0",
    }),
    18: frozenset({
        "rules.contacts.mover_to=mover(promotion)", "rules.castling(walled rank)",
        "rules.extinction.per_type", "rules.promotion.only_lost", "rules.drops.placement",
        "rules.drops.either_face", "rules.reach.on=leave", "symmetry=rotate6",
        "symmetry=dihedral12",
    }),
    19: frozenset({"rules.portals", "rules.portals.moved_by", "rules.portals.passage"}),
    20: frozenset({"rules.portals.no_straight_back"}),
    21: frozenset({
        "rules.portals.ends", "rules.portals.ends.to", "rules.portals.ends.enter",
        "rules.portals.carries",
    }),
    # Schema 22's one name moved to 24 with the folds' revised rules (Q867b).
    22: frozenset(),
    23: frozenset({"rules.npcs.also_at"}),
    24: frozenset({"rules.portals.kind"}),
    25: frozenset({"pieces[].facing.turns"}),
    26: frozenset({
        "rules.simultaneous", "rules.simultaneous.tie 'initiative'",
        "rules.simultaneous.tie 'impossible'", "pieces[].beats",
    }),
    27: frozenset({
        "rules.extinction.per_type.own", "rules.simultaneous.rest", "rules.simultaneous.swap",
        "rules.simultaneous.both", "pieces.spy",
    }),
}

# A feature deliberately without a showcase, with the reason and the audit row that says
# so. Exact in both directions: an entry whose feature gains a game fails the test until
# its row goes. Empty since Q787 closed rows 16b, 21 and 32b, the last three gaps of the
# 2026-10-09 audit; `reference/schema-showcase-coverage.md` rows 7, 10 and 11 are schema 6,
# whose schema 6 rows come from `_schema_6_uses` (Q795).
_NO_SHOWCASE_ON_PURPOSE: dict[str, str] = {
    # Schema 27's `rules.extinction.per_type.own` went on 2026-10-17, when Q936s archived
    # its showcase (`extinction-any-kind.md` §5; coverage row 72).
    # Schema 27's `pieces.spy` went on 2026-10-17, when Q935f archived its showcases
    # (`plan-spy.md` §5; coverage row 71).
    # Schema 27's three `rules.simultaneous` switches went on 2026-10-17, when Q942c
    # archived their showcases (`plan-round-chase.md` §5; coverage rows 68 to 70).
    # Schema 26's four names went on 2026-10-15, when Q918e archived their showcases
    # (`plan-simultaneous.md`, *Q918e*; coverage rows 64 to 67).
    # Schema 21's four names and schema 24's one went on 2026-10-13, when Q844b archived
    # their showcases (`archive/sluice.json`, `pawnproof-wormhole`, `crease`; coverage rows
    # 57 to 61, `reference/portal-showcase-ideas.md`).
    # Schema 19's three rows went on 2026-10-10, when Q772a archived its showcases
    # (`plan-portals.md` §7 P4, coverage rows 53 to 55).
    "schema 6: rules.strike.immune naming a non-royal piece": (
        "audit row 11: all nine strike games list only `K`, which is royal and already "
        "untakeable, so the key is inert in every one; a game that names a non-royal "
        "piece is owed, and this entry goes the day it lands"
    ),
}


def _schema_6_uses(spec: dict) -> set[str]:
    """Everything in this spec that schema 6 added and a game can show, by name (Q795).

    Schema 6 (`strike`) has no collector in `spec/schema.py`, so this one lives here, in
    the style of `_schema_nine_uses`. It reads *effective* values: a strike with no
    `effect` is a capture and one with no `position` advances, as the schema defaults say.
    `immune` is named only when it lists a non-royal piece, since listing a royal is inert
    (audit row 11). The rest of schema 6 (obstacles, chain, demote, promote, stay, swap) is
    shown by audit rows 1 to 6, 8 and 9 and is not in the table.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    strike = rules.get("strike") if isinstance(rules, dict) else None
    if not isinstance(strike, dict):
        return used
    if strike.get("effect", "capture") == "capture":
        used.add('rules.strike.effect "capture"')
    if strike.get("position", "advance") == "advance":
        used.add('rules.strike.position "advance"')
    immune = strike.get("immune")
    royal = {p.get("id") for p in spec.get("pieces", []) if isinstance(p, dict) and p.get("royal")}
    if isinstance(immune, list) and any(i not in royal for i in immune):
        used.add("rules.strike.immune naming a non-royal piece")
    return used


def _feature_parents() -> dict[str, dict]:
    """The archived games that use the current bump, and one constructed parent each.

    Until 2026-09-07 no committed or archived spec used any of schema 10, so this was the
    constructed half alone and `test_no_committed_spec_declares_this_schema` asserted that
    premise, saying the leave arm should read the real spec the day a showcase landed.
    Q101 landed seven (rows 23--30 of `reference/schema-showcase-coverage.md`) and the
    test duly fired.

    **Both halves, not the real one instead of the constructed one**, and the reason is
    that they answer different questions. A constructed parent is the smallest edit that
    turns one feature on, so what the sweep measures there is the operator set and not the
    parent's other novelties -- and it is the half a *future* bump inherits, before any
    showcase for it exists. An archived showcase is the game somebody committed, which is
    where an operator has to work for a design run to steer along the axis at all.

    The difference is not hypothetical. `archive/stones` and `archive/three-stones` are
    chess plus a stone, so their hand is filled by the stock and by nothing else -- taking
    the stone away therefore has to take `rules.drops` with it, because a hand nothing
    fills is refused by name. On those two the `drops.stock` axis is uncrossable *with its
    host rule standing*, which is a fact about a game whose only hand is a stock rather
    than a gap in the operator set, and `crazyhouse+stock` is what still asks the question
    the leave arm means to ask.

    Read from the directory rather than listed, for `ALL_SPECS`' reason: a showcase that
    lands after this is written should be swept without anyone remembering to add it.
    """
    archived = {
        path.stem: spec
        for path in sorted((ROOT / "archive").glob("*.json"))
        if _inventory(spec := load_spec(path))
    }
    return archived | {
        feature.parent_name: feature.parent() for feature in _FEATURES.values()
    }


# The audit's verdict, as data (Q108). An entry says *no operator can cross this axis in
# this direction*, and carries the reason. Stated the way
# `test_only_the_one_documented_operator_is_ever_refused` states its allowance: exact in
# both directions, so an operator that closes one of these gaps fails the test below and
# the change is deleting the row -- which is what makes surface 8 auditable at all rather
# than something each bump remembers or forgets.
#
# **Both are empty, and that is the point of them.** Q108 found three of five features
# invisible and one half-visible, wrote them here, and then closed all four; a schema 11
# that lands without an operator adds its rows back rather than saying nothing.
#
# Q224 closed schema 12's row. `soften-the-capture` and `become-what-you-take` cross
# `choose` on a standing rule, and schema 11's `own` link, retired from `_FEATURES` with
# the bump, is held by `TestTheOwnLinkIsReachable` below.
#
# Q229 added four rows, each the legitimate answer written down, and they retired with
# schema 14's `_FEATURES` rows. Schema 15's one feature had `promote-on-leaving` both ways.
#
# Q281 added two, the answer `board.lattice` got in schema 14: a hexagon's seams were
# `join-the-edges` both ways, and the shape itself only the model crosses. They retired
# with schema 16's `_FEATURES` rows.
#
# Q540's four are `leave-something-behind` both ways, so the tables were empty again.
#
# Q781 adds one. `wall-off-a-square` can enter the walled rank -- it keeps clear of every
# castling span and stamps the child 18 -- but no committed or archived game without it
# has a free square on a castling rank outside the spans (every castling army fills its
# back rank, and `omega` already has it), and no operator adds castling to a walled game.
# It retired with schema 18's `_FEATURES` rows (Q771); schema 19's three are
# `open-a-portal` both ways, so the tables are empty again.
_CANNOT_ENTER: dict[str, str] = {}
_CANNOT_LEAVE: dict[str, str] = {}


@pytest.fixture(scope="module")
def feature_sweep() -> list[_Child]:
    """The leave arm: every operator against a parent that already has each feature."""
    return _sweep_over(_feature_parents())


class TestEveryFeatureOfThisSchemaIsReachable:
    """Surface 8 of `new-move-primitive`, as a test rather than as a judgement.

    *"If the rule reaches a new kind of game, an operator must be able to cross that
    axis."* That sentence was walked at least four times and passed features nothing could
    reach, and the reason is that **it is conditional on a judgement about your own
    feature**: a nested modifier does not feel like a new kind of game, so an honest worker
    reads the clause as not applying and moves on. The test the surface named was green
    throughout -- `test_every_grid_axis_has_an_operator_that_can_cross_it` asks about
    `grid.AXES`, and none of schema 10's five features moves one, so it answered a
    different question truthfully. This class asks the question the surface meant: the
    bump's own inventory (the collector resolved at the top of this file) against the
    operator set, in both directions, for whatever `SCHEMA_VERSION` currently is.

    Two directions, because half an axis is a design run that can only ever drift one way
    -- the defect `_limit_the_game` had for `repetition_draw` before Q86, and the reason
    every add-or-remove operator in the module has two halves.

    Q100 built `rules.contacts` and found that no operator could add, remove or alter a
    contact table -- and could not fix it, because `loop/` was outside its areas. The other
    four features of schema 10 landed under the same area constraint in Q86 and nobody had
    checked them, which is where the audit below starts.

    **What the audit found**, at six seeds over the 53 committed and archived parents for
    the enter arm and over five constructed parents for the leave arm:

    | feature | enter | leave | closed by |
    |---|---|---|---|
    | `repetition_draw "never"` | `limit-the-game`, 50 specs | `limit-the-game` | -- |
    | `npcs[].immune` | `set-a-patrol`, 33 specs | **nothing** | `_hunt_the_wild` |
    | `explode_on_capture.by`/`.of` | **nothing** | **nothing** | `set-the-blast` |
    | `drops.stock` + a moveless piece | **nothing** | **nothing** | `place-a-stone` |
    | `rules.contacts` | **nothing** | **nothing** | `become-what-you-take` |

    So **one of the five was reachable, one was half an axis, and three were axes only
    the model operator could ever cross** -- the state Q100 reported for its own feature
    turning out to be the state of most of the bump. All four gaps are closed and both
    tables above are empty; the corpus sweep went 7381 children / 49 refused to **8197 /
    49**, the 49 still `resize-the-board`'s documented ones and the three new operators
    refused **nothing** in 816 draws between them.

    `explode_on_capture.by`/`.of` is entered on `atomic` alone, because it is the only
    committed spec with an explosion to narrow, and that is the intended shape rather
    than a weak result: `set-the-blast` adds the explosion in one draw and narrows it in
    the next, which is how `promote-what-arrives` reaches a game that has to have a hand
    or a strike first. One edit per operator is the module's rule, not an accident.

    **What counts as crossing this axis** is stated twice over, because both readings
    flatter the operator set if left out. A modifier -- `immune` on a wild piece, `by` on
    an explosion, `stock` on a hand -- is credited only while its *host rule survives*:
    `set-a-patrol` can drop `immune` only by clearing every wild piece, which is the
    `npcs` axis and not this one. And a crossing must leave *the same army*: the only
    route off `explode_on_capture.by` is `remove-a-piece` deleting the type the list
    names, which is `_forget_piece` repairing a dangling reference on its way past, not an
    operator that can be asked for this edit. Both routes exist; neither is an axis a
    design run can steer along, which is the property surface 8 is about.

    The live gaps are `_CANNOT_ENTER` and `_CANNOT_LEAVE` above, each with its reason.

    **The audit above now also has a prose home** —
    `docs/reference/engine-and-spec-record.md`, *Schema 10*, section *Surface 8: could a
    design run reach any of this?* — because a finding that lived only in a test docstring
    is a finding nobody reading the schema's write-up will meet. This is the executable
    half and stays here; that is the record, and the two say the same thing.

    **This is a ship-time gate, not a standing one, and Q117 decided that deliberately.**
    The sweep reads `SCHEMA_VERSION`'s inventory and nothing else, so it fires on the commit
    that raises the version -- when the author is holding the whole list -- and goes quiet
    one bump later. What it therefore cannot catch is a feature of an *older* schema whose
    last operator is afterwards removed or narrowed: strip `set-the-blast`'s narrowing half
    once schema 11 exists and `explode_on_capture.by` becomes uncrossable with every test in
    this class green, because by then `_FEATURES` is schema 11's table.

    The general fix is a **standing** table over every bump. It was costed and declined; the
    numbers are here so the next reader can re-decide rather than re-derive.

    * **The rows.** `reference/schema-showcase-coverage.md` enumerates 30 features across
      schemas 6 to 10 (11, 1, 5, 5, 8) and does not enumerate schemas 1 to 5 at all, so step
      one is writing an enumeration for five bumps that has never existed.
    * **The collectors.** `chess_variant_sandbox/spec/schema.py` has exactly two, `_schema_nine_uses` and
      `_schema_ten_uses`. Eight more would be needed and none is mechanical: a collector
      returns *names* rather than keys precisely because a key set misses features like *a
      piece with no moves of its own*, so each one is a fresh judgement about a bump whose
      author is long gone.
    * **The parents.** Every row needs a constructed parent that is valid, uses the feature,
      and -- by `test_the_constructed_parents_are_valid_and_use_the_feature`'s
      `used <= claimed` -- uses nothing outside the table. On a historical bump that forces
      the whole bump to be tabled at once or the exactness to be relaxed, and the exactness
      is what makes either arm mean anything.

    Against that, what the gap has cost so far is **nothing measurable**. The failure mode
    is regression, and there has not been one: as of 2026-09-07 `chess_variant_sandbox/loop/operators.py`
    holds 31 operators and its whole history adds 31 `Operator(` constructions and removes
    none. `explode_on_capture` shipping with no operator at all -- the story that motivates
    this class -- is the *other* failure, a feature that never had one, and that is exactly
    what this class now catches at the bump that adds it.

    The gap is also narrower than "everything before the current schema". The **coarse**
    families stay guarded for every schema, because
    `test_every_grid_axis_has_an_operator_that_can_cross_it` is written against `grid.AXES`
    -- board, army, hand, ending, rules, capture, edges, taking -- and fails if a narrowing
    makes any of them uncrossable. What is unguarded is a *specific historical key or
    modifier below that granularity*: `explode_on_capture.by`, `strike.immune`,
    `npcs[].bounty`, once their own bump has passed.

    A table covering schemas 10 and 9 only -- the two with collectors -- was considered and
    is worse than none: it would read as total while covering two bumps of ten, which is the
    assumption this note exists to stop. **Build the standing table when one of two things
    changes**: an operator is removed or narrowed for the first time, so the history above
    stops being evidence; or a bump lands whose features move no `grid.AXES` axis *and*
    something later depends on their staying reachable.
    """

    def test_this_schema_has_an_inventory_collector(self) -> None:
        """The first of the two gates that make the rest of this class inherit a bump.

        Everything below reads the bump's inventory, and it finds the collector by
        version rather than by import so that schema 11 gets this class for free. That
        only works while the name follows the convention.
        """
        assert _uses is not None, (
            f"SCHEMA_VERSION is {SCHEMA_VERSION} and `chess_variant_sandbox/spec/schema.py` has no "
            f"`{_COLLECTOR}`. `new-move-primitive` gates a bump with a collector "
            "returning the *names* of everything the bump added -- see `_schema_ten_uses` "
            "-- and this file looks it up by version. Name it to the convention, or teach "
            "`_NUMBER_WORDS` the spelling you chose."
        )

    def test_the_feature_table_is_for_the_current_schema(self) -> None:
        """The second gate, and the one that fires on the bump itself.

        Nothing can derive an inventory from a collector -- it answers *what does this
        spec use*, not *what could I ever say* -- so `_FEATURES` is hand-written and this
        is what stops it being hand-*forgotten*. It fails on the commit that raises
        `SCHEMA_VERSION`, which is the one moment the author has the whole list in front
        of them, and the failure asks for the two things a row needs that judgement cannot
        supply: which rule the feature rides on, and a parent that already has it.

        Four bumps answered surface 8 by inspection and were wrong four times. This is the
        same question with the answer computed instead.
        """
        assert _FEATURES_ARE_FOR_SCHEMA == SCHEMA_VERSION, (
            f"`_FEATURES` was written for schema {_FEATURES_ARE_FOR_SCHEMA} and "
            f"SCHEMA_VERSION is now {SCHEMA_VERSION}. Surface 8 of `new-move-primitive`: "
            f"every name `{_COLLECTOR}` can return needs a row here -- a host rule, "
            "whether a crossing must keep the army, and a parent that already uses it -- "
            "and the sweep then says whether an operator can enter and leave it. A "
            "feature no operator can reach is an axis only a model call can cross; say so "
            "in `_CANNOT_ENTER` / `_CANNOT_LEAVE` with the reason, or write the operator. "
            "Then set `_FEATURES_ARE_FOR_SCHEMA` to the new version."
        )

    def test_every_feature_of_this_schema_has_a_showcase(self) -> None:
        """The third gate, and the one a bump owes the corpus rather than the operators.

        This asserted the opposite until 2026-09-07 -- that *no* committed or archived
        spec declared any of the bump -- because that was true, and it said the leave arm
        should read the real spec the day a showcase landed. Q101 landed seven and closed
        `reference/schema-showcase-coverage.md` rows 23--30, so the assertion is inverted:
        `_feature_parents` now sweeps the archived games beside the constructed parents,
        and a feature nothing demonstrates would drop out of that half in silence.

        A schema-11 bump therefore fails here until its showcases exist, which is
        `new-move-primitive`'s "a bump is not finished until each feature it adds has a
        ruleset that demonstrates it" measured instead of remembered. The failure names
        the features with no game, which is the list of specs to write.
        """
        # Committed or archived, as `new-move-primitive` asks: schema 16's are
        # `archive/hex-glinski-hexagon` and `archive/hex-glinski-flanks` (Q281a4).
        shown = {
            label: sorted(
                path.stem for path in ALL_SPECS if _inventory(load_spec(path)) & feature.names
            )
            for label, feature in _FEATURES.items()
        }
        unshown = {label for label, games in shown.items() if not games}
        assert unshown == set(_SHOWCASE_OWED), (
            f"a feature of schema {SCHEMA_VERSION} has no committed or archived ruleset "
            "that uses it, so nothing has ever played it and the leave arm tests it only on "
            f"a parent this file built: {shown}. `_SHOWCASE_OWED` names "
            f"{sorted(_SHOWCASE_OWED)}; a showcase that has landed deletes its row there"
        )

    def test_every_feature_of_every_schema_has_a_showcase(self) -> None:
        """The third gate, for the schemas that have already passed (Q791).

        The test above reads the current schema's `_FEATURES`, so once a bump went by, a
        feature of an earlier schema with no game stopped being checked: row 21 of
        `reference/schema-showcase-coverage.md` was bare from schema 9 to Q787 and
        nothing failed. This runs every version's collector over the committed and
        archived games and names each feature no game uses.
        """
        specs = [load_spec(path) for path in ALL_SPECS]
        unshown: set[str] = set()
        unlisted: dict[int, set[str]] = {}
        for version, listed in _EVERY_SCHEMA_FEATURES.items():
            collector = (
                _schema_6_uses if version == 6
                else getattr(schema, f"_schema_{_NUMBER_WORDS[version]}_uses")
            )
            seen = set().union(*(collector(spec) for spec in specs))
            unshown |= {f"schema {version}: {name}" for name in listed - seen}
            # A name the collector returns that the table lacks is a feature the table
            # forgot, except the inert spellings named in the table's comment.
            extra = {n for n in seen - listed if not n.endswith(("=square", "=grid", "=false"))}
            if extra:
                unlisted[version] = extra
        assert not unlisted, (
            f"a collector returns names `_EVERY_SCHEMA_FEATURES` does not list: {unlisted}"
        )
        exempt = set(_NO_SHOWCASE_ON_PURPOSE)
        assert unshown == exempt, (
            "a rule feature is used by no committed or archived ruleset, so nothing has "
            f"ever played it: {sorted(unshown - exempt)}. Write a showcase "
            "(`new-move-primitive`) or, if the audit says it owes none, name it in "
            f"`_NO_SHOWCASE_ON_PURPOSE`. Exempt but now shown: {sorted(exempt - unshown)}"
        )

    def test_the_every_schema_table_agrees_with_the_current_features(self) -> None:
        """The table above is hand-written, so the bump that adds a schema must add its row."""
        assert max(_EVERY_SCHEMA_FEATURES) == SCHEMA_VERSION, (
            f"`_EVERY_SCHEMA_FEATURES` stops at schema {max(_EVERY_SCHEMA_FEATURES)} and "
            f"SCHEMA_VERSION is {SCHEMA_VERSION}: add the new bump's names"
        )
        current = frozenset().union(*(f.names for f in _FEATURES.values()))
        assert _EVERY_SCHEMA_FEATURES[SCHEMA_VERSION] == current

    def test_the_constructed_parents_are_valid_and_use_the_feature(self) -> None:
        """A leave arm built on an invalid parent measures nothing, and an operator's
        every child would be refused for the parent's reason rather than its own.

        Exact in the second direction too: a parent that quietly uses a name no row claims
        is a feature that has entered the sweep without entering the table, and the enter
        and leave arms would then be silent about it.
        """
        claimed = frozenset().union(*(feature.names for feature in _FEATURES.values()))
        for label, feature in _FEATURES.items():
            parent = feature.parent()
            assert validate_spec(parent) == [], f"{label}: {validate_spec(parent)}"
            used = _inventory(parent)
            assert used & feature.names, (
                f"{label}: its parent {feature.parent_name} uses {sorted(used)}, none of "
                f"which is {sorted(feature.names)}, so the leave arm measures nothing"
            )
            assert used <= claimed, (
                f"{label}: its parent {feature.parent_name} also uses "
                f"{sorted(used - claimed)}, which no row of `_FEATURES` names"
            )

    def test_no_operator_is_refused_or_dead_on_a_parent_that_uses_the_bump(
        self, feature_sweep
    ) -> None:
        """`TestEveryOperatorOverTheWholeCorpus`'s two questions, asked where the corpus
        cannot ask them.

        No committed spec uses schema 10, so the whole-corpus sweep has never once handed
        an operator a contact table, a stock or an immune wild piece -- and every refusal
        clause the bump added is therefore invisible to it. The audit found five
        operator/spec pairs that were **total losses** on these parents and 79 refused
        children: `soften-the-capture`, `set-a-patrol` and `hand-the-pieces-back` all
        produced a strike, a patrol or a hand beside a contact table, each refused as two
        rules answering one question, because every one of those guards was written in
        one direction only -- the shape Q85 found three times and this bump added four
        more of. `hand-the-pieces-back` also popped a hand out from under the moveless
        piece it was holding, and `remove-a-piece` left a contact link naming a piece it
        had just deleted.
        """
        applied = [child for child in feature_sweep if child.summary is not None]
        refused = {child.operator for child in applied if child.invalid}
        assert refused <= {"resize-the-board"}, (
            f"operators refused on a spec that uses schema {SCHEMA_VERSION}: "
            f"{sorted(refused - {'resize-the-board'})}. Every schema-10 feature refuses "
            "some other rule beside it, and a guard written in one direction only is an "
            "operator that is dead on the games the bump was for."
        )
        drawn: dict[tuple[str, str], int] = defaultdict(int)
        valid: dict[tuple[str, str], int] = defaultdict(int)
        for child in applied:
            drawn[(child.operator, child.spec)] += 1
            if not child.errors:
                valid[(child.operator, child.spec)] += 1
        dead = sorted(pair for pair in drawn if not valid[pair])
        assert not dead, f"applied and refused every time on a bump parent: {dead}"

    @pytest.mark.parametrize(
        "operator", ["become-what-you-take", "set-the-blast", "place-a-stone", "merge-your-own"]
    )
    def test_the_new_operators_undo_cleanly(self, specs, operator) -> None:
        """Added and taken away again is the game that was there, hash for hash.

        `test_the_new_endings_undo_cleanly`'s property for the three operators Q108 adds,
        and the one each of them could get wrong is a different one. `become-what-you-take`
        and `set-the-blast` have to put the *stamp* back -- both raise it to 10 and
        `schema_version` is in the mechanics hash, so a child that came home under a
        different number would be a new game with an identical rulebook, which is the
        waste `settle_schema_version` exists to prevent. `place-a-stone` has to undo a
        *piece*, a stock and, where it made one, the hand itself: a `drops` left holding
        `from_captures: false` and no stock is refused by name.
        """
        apply = next(op for op in OPERATORS if op.name == operator).apply
        parent = copy.deepcopy(specs["gardner"])
        before = spec_hash(parent)
        child = copy.deepcopy(parent)
        assert apply(child, random.Random(2)), f"{operator} did not apply to gardner"
        assert validate_spec(child) == [], validate_spec(child)
        assert spec_hash(child) != before

        for seed in range(40):
            back = copy.deepcopy(child)
            if apply(back, random.Random(seed)) is None:
                continue
            assert validate_spec(back) == [], f"seed {seed}: {validate_spec(back)}"
            if spec_hash(back) == before:
                return
        pytest.fail(f"{operator} never returned gardner to itself in 40 seeds")

    # A feature `_CANNOT_ENTER` names costs a whole pass of the corpus to show no operator
    # enters it (schema 26's three, half a minute each), so its arm is the slow tier's.
    @pytest.mark.parametrize(
        "feature",
        [
            pytest.param(name, marks=pytest.mark.slow) if name in _CANNOT_ENTER else name
            for name in sorted(_FEATURES)
        ],
    )
    def test_an_operator_can_enter_the_feature(self, feature) -> None:
        """Some operator turns a spec that does not use the feature into one that does.

        Read off the whole corpus rather than off five parents, for
        `TestEveryOperatorOverTheWholeCorpus`'s reason: an operator that can reach a rule
        on one spec and no other is indistinguishable, from outside, from one that reaches
        it everywhere.

        Searched to the first entry rather than read off a whole sweep (Q746): every
        operator on every committed spec at the sweep's seeds, a child counting when it
        gains a name, keeps the army where the row asks, and `mutate` would keep it. The
        refusal is asked only of a child that gained, so a missing entry, the case a
        `_CANNOT_ENTER` row states, costs a pass of the corpus without the refusals.
        """
        names = _FEATURES[feature].names
        keeps_army = _FEATURES[feature].keeps_army

        def entered(parent: dict, child: dict) -> bool:
            try:
                gained = _inventory(child) - _inventory(parent)
            except Exception:  # noqa: BLE001 -- a child the collector cannot read; ask the validator
                if refusal(parent, child):
                    return False
                gained = _inventory(child) - _inventory(parent)
            if not gained & names:
                return False
            army = [{piece.get("id") for piece in spec.get("pieces", [])} for spec in (parent, child)]
            if keeps_army and army[0] != army[1]:
                return False
            return not refusal(parent, child)

        # A spec that already carries the host rule is searched first, since a modifier is
        # entered most cheaply where its host stands; any order finds the same answer.
        host = _FEATURES[feature].host
        first = _first_child(
            entered,
            likely=lambda spec: bool(host and (spec.get("rules") or {}).get(host))
            or _FEATURES[feature].likely(spec),
        )
        who = [first] if first else []
        gap = _CANNOT_ENTER.get(feature)
        assert bool(who) is (gap is None), (
            f"{feature}: entered by {who or 'nothing'}, and _CANNOT_ENTER "
            f"{'names it: ' + gap if gap else 'does not name it'}. An operator that "
            "closes a gap deletes the row; a feature that stops being reachable adds one."
        )

    @pytest.mark.parametrize("feature", sorted(_FEATURES))
    def test_an_operator_can_leave_the_feature(self, feature_sweep, feature) -> None:
        """And back out again, without deleting the rule the feature rides on."""
        host = _FEATURES[feature].host
        names = _FEATURES[feature].names
        keeps_army = _FEATURES[feature].keeps_army
        who = sorted(
            {
                f"{c.operator} on {c.spec}"
                for c in feature_sweep
                if c.lost & names
                and (c.same_army or not keeps_army)
                and (host is None or host in c.survived)
            }
        )
        gap = _CANNOT_LEAVE.get(feature)
        assert bool(who) is (gap is None), (
            f"{feature}: left by {who[:6] or 'nothing'}"
            + (f" (with rules.{host} still standing)" if host else "")
            + f", and _CANNOT_LEAVE {'names it: ' + gap if gap else 'does not name it'}."
        )


class TestTheOwnLinkIsReachable:
    """Schema 11's `contacts.links[].own`, held after its `_FEATURES` row retired (Q224).

    The surface-8 gate reads only the current bump's collector, so the own link lost its
    row at schema 12 with no operator able to write it. These are that row's two arms and
    its refusal check, asked of `_schema_eleven_uses` directly.
    """

    _NAME = "rules.contacts.links[].own"

    @staticmethod
    def _uses_own(spec: dict) -> bool:
        return TestTheOwnLinkIsReachable._NAME in schema._schema_eleven_uses(spec)

    @staticmethod
    def _own_parents() -> dict[str, dict]:
        return {"merger": load_spec(ROOT / "archive" / "merger.json"), "chess+own": _parent_own()}

    def test_an_operator_can_enter_the_own_link(self) -> None:
        """`merge-your-own` writes an own link on a spec that has none, army unchanged."""
        apply = next(op for op in OPERATORS if op.name == "merge-your-own").apply
        entered = []
        for path in ALL_SPECS:
            parent = load_spec(path)
            if self._uses_own(parent):
                continue
            child = copy.deepcopy(parent)
            if apply(child, random.Random(0)) is None:
                continue
            assert validate_spec(child) == [], f"{path.stem}: {validate_spec(child)}"
            assert self._uses_own(child)
            assert [p["id"] for p in child["pieces"]] == [p["id"] for p in parent["pieces"]]
            entered.append(path.stem)
        assert len(entered) > 20, f"entered on only {entered}"

    def test_an_operator_can_leave_the_own_link_with_the_table_standing(self) -> None:
        for name, parent in self._own_parents().items():
            assert validate_spec(parent) == [] and self._uses_own(parent)
            left = False
            for operator in OPERATORS:
                for seed in _SWEEP_SEEDS:
                    child = copy.deepcopy(parent)
                    if operator.apply(child, random.Random(seed)) is None:
                        continue
                    if validate_spec(child):
                        assert operator.name == "resize-the-board", (
                            f"{operator.name} refused on {name}: {validate_spec(child)}"
                        )
                        continue
                    left |= (
                        not self._uses_own(child)
                        and bool(child["rules"].get("contacts"))
                        and child["pieces"] == parent["pieces"]
                    )
            assert left, f"nothing left the own link on {name} with rules.contacts standing"

    def test_every_merge_link_the_operator_may_write_is_valid(self, specs) -> None:
        """Every candidate, not a sampled one, so a refusal is excluded by construction."""
        from chess_variant_sandbox.spec.transform import merge_links, set_own_link

        for name, parent in specs.items():
            if parent["rules"].get("must_capture") or any(
                parent["rules"].get(key)
                for key in ("strike", "explode_on_capture", "drops", "npcs")
            ):
                continue
            for link in merge_links(parent):
                child = set_own_link(copy.deepcopy(parent), link)
                assert validate_spec(child) == [], f"{name} + {link}: {validate_spec(child)}"


class TestTheLandingSettles:
    """`choose` crossed on a standing rule, and settled when its survivor goes (Q224)."""

    def test_place_and_unplace_returns_the_parent(self) -> None:
        apply = next(op for op in OPERATORS if op.name == "soften-the-capture").apply
        parent = load_spec(ROOT / "archive" / "hierarchy-rise-stay.json")
        before = spec_hash(parent)
        placed = None
        for seed in range(40):
            child = copy.deepcopy(parent)
            if apply(child, random.Random(seed)) and child["rules"].get("strike", {}).get("position") == "choose":
                placed = child
                break
        assert placed is not None, "soften-the-capture never placed the survivor"
        assert validate_spec(placed) == [], validate_spec(placed)
        for seed in range(40):
            back = copy.deepcopy(placed)
            if apply(back, random.Random(seed)) and spec_hash(back) == before:
                return
        pytest.fail("soften-the-capture never returned hierarchy-rise-stay to itself")

    def test_dropping_the_last_survivor_settles_choose(self) -> None:
        """A contact table whose only surviving link names the dropped piece."""
        from chess_variant_sandbox.spec.transform import drop_piece

        spec = _chess()
        spec["rules"].pop("en_passant", None)
        spec["rules"]["contacts"] = {
            "position": "choose",
            "landing": {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"},
            "immune": ["K"],
            "links": [
                {"mover": "N", "onto": "B", "onto_to": "N"},
                {"mover": "P", "onto": "R", "mover_to": "R"},
            ],
        }
        spec["schema_version"] = 12
        assert validate_spec(spec) == [], validate_spec(spec)
        drop_piece(spec, "B")
        assert validate_spec(spec) == [], validate_spec(spec)
        assert spec["rules"]["contacts"]["position"] == "advance"
        assert "landing" not in spec["rules"]["contacts"]


def _rise_at(position: str, key: str, links: list[dict[str, Any]]) -> dict:
    """`hierarchy-rise-stay` with its rule re-spelled under ``key`` as ``links``, at ``position``.

    ``immune`` is ``["K"]`` either way, the showcase's own, so a link from or onto the king
    is written and kept nothing alive.
    """
    spec = load_spec(ROOT / "archive" / "hierarchy-rise-stay.json")
    rules = spec["rules"]
    links = copy.deepcopy(links)
    if key == "strike":
        rules["strike"].update(position=position, chain=links)
        rule = rules["strike"]
    else:
        rules.pop("strike")
        rule = rules["contacts"] = {"position": position, "immune": ["K"], "links": links}
        spec["schema_version"] = max(spec["schema_version"], 10)
    if position == "choose":
        rule["landing"] = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
        spec["schema_version"] = max(spec["schema_version"], 12)
    return spec


class TestAnImmuneLinkKeepsNothingAlive:
    """Q413: the operators ask `choose`'s question the way the validator asks it.

    `transform.keeps_a_survivor` counted links, so a link from or onto an immune piece read
    as a survivor although a strike never reaches one: the miscount Q271 took out of the
    validator. `set_landing`, reached through `_place_the_survivor`, then wrote `choose` on
    a rule the validator refuses, and `settle_landing` kept `choose` where it had to become
    `advance`. It now asks `schema.strike_survives_anything`. No operator writes an immune
    link, but a proposal can, so each case starts from a spec that already carries one.
    """

    # A strike never reaches the king, so this chain keeps nothing alive.
    DEAD_CHAIN = [{"from": "K", "to": "Q"}]
    # A rule, because its second link changes the mover, but it keeps no target alive.
    DEAD_TABLE = [
        {"mover": "P", "onto": "K", "onto_to": "Q"},
        {"mover": "P", "onto": "N", "mover_to": "N"},
    ]
    LIVE = {
        "strike": [{"from": "P", "to": "N"}],
        "contacts": [{"mover": "N", "onto": "B", "onto_to": "N"}],
    }
    OPERATOR = {"strike": "soften-the-capture", "contacts": "become-what-you-take"}
    SEEDS = range(12)

    def _dead(self, key: str) -> list[dict[str, Any]]:
        return self.DEAD_CHAIN if key == "strike" else self.DEAD_TABLE

    def test_a_strike_with_nothing_to_place_is_no_parent(self) -> None:
        """Since Q538 the dead chain is refused, so no operator is handed one to place.

        The table stays below, because its second link changes the mover and keeps it a
        rule.
        """
        from chess_variant_sandbox.spec.schema import PLAYED_AS_CAPTURE

        (refusal,) = validate_spec(_rise_at("stay", "strike", self.DEAD_CHAIN))
        assert PLAYED_AS_CAPTURE in refusal, refusal

    @pytest.mark.parametrize("key", ["contacts"])
    def test_the_operator_never_places_a_survivor_nothing_leaves(self, key: str) -> None:
        parent = _rise_at("stay", key, self._dead(key))
        assert validate_spec(parent) == [], validate_spec(parent)
        apply = next(op for op in OPERATORS if op.name == self.OPERATOR[key]).apply
        for seed in self.SEEDS:
            child = copy.deepcopy(parent)
            assert apply(child, random.Random(seed)) is not None, f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"
            assert (child["rules"].get(key) or {}).get("position") != "choose", f"seed {seed}"
        # `_place_the_survivor` spends the seed's first draw on whether to cross at all,
        # so the case asks nothing unless some seed takes the crossing half.
        assert any(random.Random(seed).random() < 0.5 for seed in self.SEEDS)

    @pytest.mark.parametrize("key", ["strike", "contacts"])
    def test_one_live_link_beside_it_is_still_placed(self, key: str) -> None:
        """The other side: an immune link does not stop a live one from being placed."""
        parent = _rise_at("stay", key, self.LIVE[key] + self._dead(key))
        assert validate_spec(parent) == [], validate_spec(parent)
        apply = next(op for op in OPERATORS if op.name == self.OPERATOR[key]).apply
        placed = 0
        for seed in self.SEEDS:
            child = copy.deepcopy(parent)
            assert apply(child, random.Random(seed)) is not None, f"seed {seed}"
            assert validate_spec(child) == [], f"seed {seed}: {validate_spec(child)}"
            placed += (child["rules"].get(key) or {}).get("position") == "choose"
        assert placed, f"{self.OPERATOR[key]} never placed the survivor"

    @pytest.mark.parametrize("key", ["strike", "contacts"])
    def test_dropping_the_last_live_link_settles_choose(self, key: str) -> None:
        """`drop_piece` takes the live link and leaves the immune one, and `settle_landing`
        has to read that as nothing left to place."""
        from chess_variant_sandbox.spec.transform import drop_piece

        spec = _rise_at("choose", key, self.LIVE[key] + self._dead(key))
        assert validate_spec(spec) == [], validate_spec(spec)
        drop_piece(spec, "N" if key == "strike" else "B")
        assert validate_spec(spec) == [], validate_spec(spec)
        if key == "strike":
            # The immune link left behind reaches nobody, so the rule goes too (Q538).
            assert "strike" not in spec["rules"]
            return
        assert spec["rules"][key]["position"] == "advance"
        assert "landing" not in spec["rules"][key]


class TestTheClocks:
    """`limit-the-game` and `transform.never_forced` both write a clock, and both wrote
    values the schema refuses -- the two halves of Q85's silent-waste class that no
    committed spec is close enough to the bound to expose.
    """

    def test_the_bounds_the_producers_read_are_the_bounds_the_validator_enforces(
        self, specs
    ) -> None:
        """`CLOCK_BOUNDS` exists so that nothing has to guess, which only works while it
        is the same table the refusal comes from. Asked of `validate_spec` rather than
        read off the constant twice."""
        for key, (low, high) in CLOCK_BOUNDS.items():
            for value, wanted in ((low, True), (high, True), (low - 1, False), (high + 1, False)):
                spec = copy.deepcopy(specs["chess"])
                spec["rules"][key] = value
                accepted = not validate_spec(spec)
                assert accepted is wanted, (
                    f"rules.{key} = {value}: validator "
                    f"{'accepted' if accepted else 'refused'} it, CLOCK_BOUNDS says "
                    f"{'in' if wanted else 'out of'} range {low}..{high}"
                )

    def test_a_clock_edit_never_leaves_the_schema_bounds(self, specs) -> None:
        """The defect Q84 found: `max(2, int(current * scale))` had a floor and no
        ceiling, so `repetition_draw` at 10 doubled to 20 against a cap of 10 and the
        child was thrown away without a word.

        Walked over the boundary neighbourhood of every clock rather than sampled,
        because a boundary bug is a fact about the ends and nowhere else. No committed
        ruleset sits near a cap -- the clocks *drift* there over a run, which is why this
        was invisible to every test that starts from a committed spec.
        """
        from chess_variant_sandbox.loop.operators import _limit_the_game

        for key, (low, high) in CLOCK_BOUNDS.items():
            for value in (low, low + 1, low + 2, high // 2, high - 2, high - 1, high):
                for seed in range(12):
                    spec = copy.deepcopy(specs["chess"])
                    spec["rules"][key] = value
                    summary = _limit_the_game(spec, random.Random(seed))
                    if summary is None:
                        continue
                    assert not validate_spec(spec), (
                        f"{key} at {value}, seed {seed}: {summary!r} -> "
                        f"{validate_spec(spec)}"
                    )

    def test_the_clock_that_may_say_never_still_crosses_both_ways(self, specs) -> None:
        """Schema 10 gave `repetition_draw` a value that is not a size, and the clamp must
        not mistake a word for a number to be squeezed into 2..10."""
        from chess_variant_sandbox.loop.operators import _limit_the_game
        from chess_variant_sandbox.spec.transform import settle_schema_version

        spec = copy.deepcopy(specs["chess"])
        spec["rules"]["repetition_draw"] = REPETITION_NEVER
        # `"never"` is schema 10, and a parent whose stamp does not say so is invalid
        # before any operator touches it -- which is a badly built test, not a defect.
        settle_schema_version(spec)
        assert not validate_spec(spec)
        reached = set()
        for seed in range(40):
            child = copy.deepcopy(spec)
            if _limit_the_game(child, random.Random(seed)) is None:
                continue
            assert not validate_spec(child)
            reached.add(child["rules"]["repetition_draw"])
        assert any(isinstance(value, int) for value in reached), (
            "no draw brought the repetition rule back, so the axis is one-way"
        )

    def test_never_forced_writes_a_no_progress_clock_the_schema_accepts(
        self, specs
    ) -> None:
        """Q56's defect. The two clocks have ceilings an order of magnitude apart, and
        pushing the no-progress clock out to the ply limit ignored the lower one.

        Both halves are asserted, because clamping the no-progress clock alone would let
        it fire *first* on exactly the specs the clamp is for -- a mutant that quietly
        stops being the mutant it is named after.
        """
        from chess_variant_sandbox.spec.transform import never_forced

        for max_plies in (2, 200, 400, 1000, 1001, 2000, 9999):
            spec = copy.deepcopy(specs["chess"])
            spec["rules"]["max_plies"] = max_plies
            never_forced(spec)
            assert not validate_spec(spec), (
                f"max_plies {max_plies}: {validate_spec(spec)}"
            )
            assert spec["rules"]["no_progress_draw"] >= spec["rules"]["max_plies"], (
                f"max_plies {max_plies}: the no-progress clock would fire first, so this "
                "is no longer the mutant that removes every reason to make progress"
            )

    def test_never_forced_leaves_every_committed_spec_valid(self) -> None:
        """Over the whole corpus, since it is a mutant the metrics build from every one."""
        from chess_variant_sandbox.spec.transform import never_forced

        for path in ALL_SPECS:
            spec = never_forced(load_spec(path))
            assert not validate_spec(spec), f"{path.stem}: {validate_spec(spec)}"


class TestDroppingAPiece:
    def test_a_dropped_piece_leaves_no_reference_behind(self) -> None:
        """`drop_piece`'s cascade, held complete by measurement rather than by care.

        It cleaned the definition, the placements, castling and promotion, and left the
        capture matrix, the icon table, the strike chain, the drop region and stock, the
        wild routes, the extinction list and the explosion's immunity naming a piece that
        no longer existed. `remove-a-piece` was therefore a total loss on 25 of the 53
        committed and archived specs, silently, for the second time in this function's
        life -- the promotion-list bug its own docstring describes is the first.

        Every piece of every spec, so a schema bump that puts a piece id somewhere new
        fails here rather than in a design run. A dropped piece may leave the spec
        *invalid for a reason that is not a dangling reference* -- an army of one, a
        placement of a piece that is now the only mover -- so this asserts on the shape of
        the complaint, which is the thing the cascade is responsible for.
        """
        from chess_variant_sandbox.spec.transform import drop_piece, pieces

        dangling = ("is not a declared piece", "names unknown piece")
        checked = 0
        for path in ALL_SPECS:
            original = load_spec(path)
            for piece in pieces(original):
                spec = drop_piece(load_spec(path), str(piece["id"]))
                checked += 1
                left = [
                    error for error in validate_spec(spec)
                    if any(phrase in error for phrase in dangling)
                ]
                assert not left, f"{path.stem} without {piece['id']!r}: {left}"
        assert checked > 200, f"only {checked} (spec, piece) pairs; the corpus moved"

    def test_dropping_the_piece_a_narrowed_blast_names_takes_the_rule_with_it(self) -> None:
        """Q435: an emptied `by`/`of` must not read as "everyone".

        `_forget_piece` pruned `explode_on_capture.by`/`.of` the same way it prunes every
        other list -- pop the id, then pop the key if the list is now empty -- but an
        *absent* `by`/`of` means the blast fires on every capture, not on none. So popping
        the key when the last named piece leaves silently turned a narrowed blast into
        unrestricted atomic chess. `archive/queen-bomb` is atomic chess with the blast
        narrowed to captures made by the queen (`rules.explode_on_capture.by == ["Q"]`);
        `remove-a-piece` taking the queen is exactly the case, and before the fix the
        result validated and still had `explode_on_capture` standing, unrestricted.
        """
        from chess_variant_sandbox.spec.transform import drop_piece

        spec = load_spec(ROOT / "archive" / "queen-bomb.json")
        assert spec["rules"]["explode_on_capture"]["by"] == ["Q"]
        spec = drop_piece(spec, "Q")
        assert validate_spec(spec) == [], validate_spec(spec)
        assert "explode_on_capture" not in spec["rules"], (
            "the blast survived losing every piece its `by` list named, unrestricted"
        )

    def test_a_dropped_piece_leaves_no_reference_behind_in_a_bump_parent(self) -> None:
        """The same walk over the features the corpus cannot supply.

        The cascade is *"held complete by measurement rather than by care"* over every
        committed spec -- and no committed spec uses schema 10, so the bump put two more
        places a piece id can hide where that walk could not see them. Both bit:
        `rules.contacts.links` names up to four pieces per link and had no branch at all,
        so a dropped piece left a link pointing at nothing; and `rules.drops.stock` had
        one, but emptying it leaves a hand that nothing fills, which the validator refuses
        for a reason that is not a dangling reference and so needs its own clause.
        """
        from chess_variant_sandbox.spec.transform import drop_piece, pieces

        for name, parent in _feature_parents().items():
            for piece in pieces(parent):
                spec = drop_piece(copy.deepcopy(parent), str(piece["id"]))
                left = [
                    error
                    for error in validate_spec(spec)
                    if "is not a declared piece" in error
                    or "names unknown piece" in error
                    or "nothing else fills a hand" in error
                ]
                assert not left, f"{name} without {piece['id']!r}: {left}"


class TestSplittingAPiece:
    """`transform.split_piece` changes what some pieces are, and nothing else about them.

    It used to write the new piece's placement as ``{"piece": id, "squares": [...]}``
    whatever the placements it took from said, so a spec that writes each army out
    (``mirror: false``) had the second player's share handed to the first, and a facing
    piece's share turned to its default. G1's reading found it: 27 of 206 `add-a-piece`
    children read as two rules, the second a setup the operator never meant to move
    (Q330, `docs/archive/plan-game-graph.md` §6).
    """

    @staticmethod
    def _cells(spec: dict, ids: set[str], as_id: str) -> list[tuple]:
        """Every starting piece of ``ids``, mirror applied, each read as ``as_id``.

        The normal form's own reading of the setup (`relate._setup_cells`, which is what
        `normal_form(spec).cells` holds), without the rest of the normal form: on a
        rotating spec that rest compiles the child twice, and shogi's fourteen splits were
        most of this class's 5.6 s in the gate (Q749).
        """
        from chess_variant_sandbox.spec.relate import _int, _setup_cells

        board = spec.get("board") or {}
        width, height = _int(board.get("width"), 8), _int(board.get("height"), 8)
        orientation = str((spec.get("setup") or {}).get("orientation", "mirror"))
        cells = _setup_cells(spec, width, height, orientation)
        return sorted(
            (player, file_, rank, as_id, facing)
            for player, file_, rank, pid, facing in cells
            if pid in ids
        )

    def test_a_split_keeps_every_piece_where_it_stood_and_whose_it_was(self) -> None:
        """Over every piece a split can take, of every committed spec.

        Read the child's setup with the new id folded back into its source and it must be
        the parent's setup exactly: the same player, square and facing for every piece.
        """
        from chess_variant_sandbox.spec.transform import pieces, split_piece

        checked = 0
        for path in ALL_SPECS:
            parent = load_spec(path)
            for piece in pieces(parent):
                source = str(piece["id"])
                child = split_piece(copy.deepcopy(parent), source, "X")
                if len(pieces(child)) == len(pieces(parent)):
                    continue  # nothing to share, or a royal
                checked += 1
                assert self._cells(child, {source, "X"}, source) == self._cells(
                    parent, {source}, source
                ), f"{path.stem}: splitting {source!r} moved a piece's owner or facing"
                assert validate_spec(child) == [], f"{path.stem} {source}: {validate_spec(child)}"
        assert checked > 200, f"only {checked} splits; the corpus moved"

    def test_a_split_of_a_written_out_army_keeps_the_second_players_share(self) -> None:
        """The shape that broke, on its own: two armies written out, one placement each."""
        from chess_variant_sandbox.spec.transform import split_piece

        parent = load_spec(ROOT / "archive" / "weathervane.json")
        assert parent["setup"]["mirror"] is False
        child = split_piece(copy.deepcopy(parent), "N", "X")
        placed = {
            (placement.get("player", 0), square)
            for placement in child["setup"]["placements"]
            if placement["piece"] == "X"
            for square in placement["squares"]
        }
        assert {player for player, _ in placed} == {0, 1}, placed


class TestTheProseTables:
    def test_score_the_clock_has_a_subject_for_every_count_the_schema_allows(self) -> None:
        """The `KeyError('points')` defect, stated as the invariant that would have
        prevented it. Read off `SCORING_COUNTS` rather than listed here, so the next
        count to be added fails this instead of a design run."""
        from chess_variant_sandbox.loop.operators import _SCORE_SUBJECTS

        missing = set(SCORING_COUNTS) - set(_SCORE_SUBJECTS)
        assert not missing, f"_SCORE_SUBJECTS has no prose for {sorted(missing)}"
        assert not set(_SCORE_SUBJECTS) - set(SCORING_COUNTS)

    def test_score_the_clock_survives_a_spec_with_wild_pieces(self) -> None:
        """`archive/wolves` is the spec this crashed on at every seed. It was the only
        committed spec with `rules.npcs` then; seven `wolves` specs carry it now."""
        from chess_variant_sandbox.loop.operators import _score_the_clock

        wolves = load_spec(ROOT / "archive" / "wolves.json")
        assert wolves["rules"].get("npcs"), "wolves has stopped being the npcs showcase"
        for seed in range(6):
            spec = copy.deepcopy(wolves)
            summary = _score_the_clock(spec, random.Random(seed))
            assert summary and "points" in summary
            assert spec["rules"]["scoring"]["count"] == "points"
            assert not validate_spec(spec)


class TestScreen:
    """The middle rung of the cost ladder.

    Its whole risk is the shatranj mistake at one rung higher: a threshold that looks
    reasonable and quietly throws out real games for being slow to reveal themselves. So
    the test is the same one the smoke gate has -- run it over the whole committed corpus
    and require that it keeps every real ruleset.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("name", [n for n in CORPUS if n not in _NOT_REAL_GAMES])
    def test_every_real_ruleset_survives(self, specs, name) -> None:
        result = screen(specs[name], seed=0, workers=1)
        assert result.passed, f"{name} rejected: {result.reasons} {result.stats}"

    def test_the_shallow_control_is_no_longer_rejected_here(self, specs) -> None:
        """It used to be, and the threshold was deliberately not moved to keep it so.

        Phase C gave the search bounds, quiescence and ordering beyond captures, and the
        stronger player now wins one game of eight in `dull` where it used to win none.
        The screen asks one question -- *can a searching player win this at all* -- and
        the honest answer changed. Raising `MIN_DECISIVE` to two would reject it again
        and would put `makruk` and `shatranj`, which measure exactly two, on the line
        with no margin: a threshold fitted between adjacent points, which is the shatranj
        mistake this rung exists not to repeat.

        What still has to hold is the division of labour, and that is what this asserts:
        the index gates `dull` out on `decisive_rate`, where a judgement needing competent
        play belongs.
        """
        result = screen(specs["dull"])
        assert result.passed, "if this rung rejects dull again, re-read the comment above"
        assert result.stats["decisive"] < 2, (
            "dull is expected to be barely winnable, not comfortably so; if the searching "
            "player now wins it easily, the control has stopped being a control"
        )

    def test_an_unwinnable_ruleset_is_rejected_here_too(self, specs) -> None:
        """Hierarchy chess as its designer first stated it: nothing can capture a king.

        The smoke gate already catches it, and so does this rung, which is the answer to
        "is the screen doing anything the gate below it does not?" for this particular
        failure -- no, and that is fine. Both are cheap and they fail differently on
        other rulesets.
        """
        result = screen(specs["rps-hierarchy-chess"], seed=0, workers=1)
        assert not result.passed

    def test_it_is_cheap_enough_to_be_worth_it(self, specs) -> None:
        """A screen that costs what it saves is not a screen.

        Cost is CPU time, which three other workers of the fast suite cannot inflate the
        way they can a wall clock (Q377c). One worker plays in this thread, so the
        process's own time is all of it.
        """
        import time

        started = time.process_time()
        screen(specs["chess"], seed=0, workers=1)
        assert time.process_time() - started < 8.0


class TestDistance:
    def test_a_spec_is_zero_from_itself(self, specs) -> None:
        for name in ("chess", "shogi", "dull"):
            assert distance(specs[name], specs[name]) == 0.0

    def test_no_two_committed_rulesets_measure_identical(self, specs) -> None:
        """Zero has to mean "the same description", or nothing downstream can use it.

        This caught two real pairs. Berolina swaps the pawn's move and capture, so it has
        chess's counts exactly and differs only in which vector captures; knightmate makes
        the knight royal and the knights king-movers, which leaves every count identical
        too. Both measured 0.000 until the features were conditioned on mode and on
        royalty.
        """
        for first, second in itertools.combinations(CORPUS, 2):
            gap = distance(specs[first], specs[second])
            assert gap > 0.0, f"{first} and {second} measure identical"

    def test_a_variant_of_chess_is_nearer_to_it_than_a_different_game(self, specs) -> None:
        chess = specs["chess"]
        assert distance(chess, specs["almost"]) < distance(chess, specs["shogi"])
        assert distance(chess, specs["knightmate"]) < distance(chess, specs["dobutsu"])
        assert distance(chess, specs["threecheck"]) < distance(chess, specs["xiangqi"])

    def test_it_is_symmetric_and_bounded(self, specs) -> None:
        for first, second in itertools.combinations(list(CORPUS)[:8], 2):
            forward = distance(specs[first], specs[second])
            assert forward == distance(specs[second], specs[first])
            assert 0.0 <= forward <= 1.0

    def test_one_local_edit_moves_a_spec_a_little(self, specs) -> None:
        """The property the operator layer needs: a mechanical child is a *neighbour*.

        Measured against the corpus, whose median pairwise distance is 0.166 -- a single
        edit has to land well inside that or "local" is not what it means.
        """
        rng = random.Random(2)
        chess = specs["chess"]
        gaps = []
        for _ in range(40):
            child = mutate(chess, rng)
            if child is not None:
                gaps.append(distance(chess, child.spec))
        assert gaps
        assert sorted(gaps)[len(gaps) // 2] < 0.05

    def test_the_feature_vector_is_flat_and_finite(self, specs) -> None:
        for name in CORPUS:
            values = features(specs[name])
            assert all(isinstance(value, float) for value in values.values())
            assert all(value == value for value in values.values())  # no NaN



# ---------------------------------------------------------------------------------------
# Q735: what `distance` and `features` compute, pinned to the number (Q734m, `spec/`).
# ---------------------------------------------------------------------------------------

# Each feature's scale, written out here on purpose: the test pins what the code says, so
# a changed scale fails it and the change is made here on purpose. A feature missing from
# the table is not asserted (a new feature adds its row; `test_the_pinned_table_names_only_
# features_distance_reads` holds the other direction).
_PINNED_SCALES: dict[str, float] = {
    "area": 100.0, "aspect": 1.0, "piece_types": 8.0, "army": 40.0, "rule_mass": 90.0,
    "special_rules": 6.0, "vectors_per_piece": 8.0, "slide_share": 1.0, "leap_share": 1.0,
    "hop_share": 1.0, "sweep_share": 1.0, "royal_types": 2.0, "royal_step": 2.0,
    "royal_slides": 1.0, "move_orthogonal_share": 1.0, "capture_orthogonal_share": 1.0,
    "mean_step": 3.0, "has_promotion": 1.0, "has_drops": 1.0, "has_stock": 1.0,
    "stock_is_endless": 1.0, "drops_placement": 1.0, "drops_either_face": 1.0,
    "has_a_moveless_piece": 1.0, "has_castling": 1.0, "win_is_capture": 1.0,
    "wins_by_arrival": 1.0, "wins_by_check_count": 1.0, "wins_by_extinction": 1.0,
    "wins_by_bare_royal": 1.0, "restricts_captures": 1.0, "has_territory": 1.0,
    "repaints_territory": 1.0, "must_capture": 1.0, "wraps_files": 1.0, "wraps_ranks": 1.0,
    "wraps_asymmetrically": 1.0, "wraps_n_s": 1.0, "wraps_ne_sw": 1.0, "wraps_nw_se": 1.0,
    "explodes": 1.0, "explodes_selectively": 1.0, "has_obstacles": 1.0, "strikes": 1.0,
    "strike_promotes": 1.0, "swaps_on_strike": 1.0, "contacts_by_pair": 1.0,
    "contacts_change_the_mover": 1.0, "contacts_meet_own": 1.0, "contacts_keep_the_mover": 1.0,
    "strike_places_the_survivor": 1.0, "strike_stays_put": 1.0, "strike_rebirths": 1.0,
    "capturer_switches_sides": 1.0, "explodes_alone": 1.0, "has_facing": 1.0,
    "turns_freely": 1.0, "facing_fixed": 1.0, "has_npcs": 1.0, "npcs_bite": 1.0,
    "npcs_pay": 1.0, "npcs_are_hard_to_kill": 1.0, "promotes_on_entry": 1.0,
    "no_repetition_draw": 1.0, "hex_lattice": 1.0, "promotion_zone_share": 1.0,
    "promotes_on_leaving": 1.0, "extinction_per_type": 1.0, "promotes_only_lost": 1.0,
    "wins_by_leaving": 1.0, "portal_pairs": 1.0, "portals_move": 1.0, "portals_one_way": 1.0,
    "portals_no_back": 1.0, "portals_ring": 1.0, "portals_funnel": 1.0,
    "portals_other_map": 1.0, "portals_filtered": 1.0, "portals_fold": 1.0,
    "npcs_share_a_walk": 1.0,
}


def _gap_in_a_distance(monkeypatch: pytest.MonkeyPatch, name: str, gap: float) -> float:
    """`distance` between two vectors equal but for ``name``, unrounded and not capped by
    ``features``: the vectors are stood in for the specs, so one feature is isolated."""
    import chess_variant_sandbox.spec.distance as module

    vectors = {"left": {}, "right": {name: gap}}
    monkeypatch.setattr(module, "features", lambda spec: vectors[spec["side"]])
    monkeypatch.setattr(module, "round", lambda value, places: value, raising=False)
    return module.distance({"side": "left"}, {"side": "right"})


class TestDistanceIsPinnedToTheNumber:
    def test_the_pinned_table_names_only_features_distance_reads(self) -> None:
        from chess_variant_sandbox.spec.distance import _SCALES

        assert set(_PINNED_SCALES) <= set(_SCALES)

    @pytest.mark.parametrize("name", sorted(_PINNED_SCALES))
    def test_a_feature_half_its_scale_apart_contributes_half_a_share(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """Each feature is divided by its own scale: half a scale apart is half a share of
        the mean, which a scale one off (a board of 100 squares read as 101) moves."""
        from chess_variant_sandbox.spec.distance import _SCALES

        got = _gap_in_a_distance(monkeypatch, name, _PINNED_SCALES[name] / 2)
        assert got == pytest.approx(0.5 / len(_SCALES), rel=1e-9)

    def test_a_gap_past_its_scale_is_capped_at_one_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 3x4 board against a 12x10 one is 1.08 scales apart in area and counts as one:
        no feature may outweigh the 1/len(_SCALES) it is worth."""
        from chess_variant_sandbox.spec.distance import _SCALES

        got = _gap_in_a_distance(monkeypatch, "area", _PINNED_SCALES["area"] * 1.08)
        assert got == pytest.approx(1 / len(_SCALES), rel=1e-9)

    def test_the_distance_is_rounded_to_four_places(self) -> None:
        """Chess against its bare-royal twin differs in the flag (1/1), the rule mass (52
        to 58, over 90) and the special rules (3 to 4, over 6): the mean over the 80
        features is 0.015417, which rounds to 0.0154 and not to 0.015 or 0.01542."""
        chess = _chess()
        twin = copy.deepcopy(chess)
        twin["rules"]["bare_royal"] = "win"
        twin = _stamped(twin)
        assert validate_spec(twin) == []
        assert distance(chess, twin) == 0.0154


def _one_leap_beside_a_stone() -> dict:
    """The smallest valid spec with exactly one move descriptor, and a stone with none."""
    return {
        "schema_version": 10,
        "name": "one-leap",
        "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "S", "name": "stone", "moves": []},
            {"id": "L", "name": "leaper", "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}
            ]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "S", "squares": ["a1"]}, {"piece": "L", "squares": ["b1"]},
        ]},
        "rules": {
            "win": "none", "stalemate": "loss",
            "drops": {"from_captures": False, "stock": {"S": "unlimited"}},
        },
    }


class TestFeaturesReadTheSpecExactly:
    def test_one_move_descriptor_is_the_whole_of_the_kinds(self) -> None:
        spec = _one_leap_beside_a_stone()
        assert validate_spec(spec) == []
        read = features(spec)
        assert read["leap_share"] == 1.0
        assert read["slide_share"] == read["hop_share"] == read["sweep_share"] == 0.0

    def test_a_spec_with_no_move_at_all_reads_zero_shares_and_no_step(self) -> None:
        spec = _one_leap_beside_a_stone()
        spec["pieces"].pop()
        spec["setup"]["placements"].pop()
        assert validate_spec(spec) == []
        read = features(spec)
        for name in ("slide_share", "leap_share", "hop_share", "sweep_share", "mean_step"):
            assert read[name] == 0.0, name
        # With nothing counted the orthogonal shares sit midway, not at either end.
        assert read["move_orthogonal_share"] == 0.5
        assert read["capture_orthogonal_share"] == 0.5

    def test_a_royal_slides_only_beyond_one_square(self) -> None:
        """A royal written as a slide of range 1 steps; of range 2 it slides."""
        reads = {}
        for reach in (1, 2):
            spec = _chess()
            spec["rules"].pop("castling")
            king = _piece(spec, "K")
            king["moves"] = [
                {"kind": "slide", "vectors": [[1, 0], [1, 1]], "range": reach,
                 "symmetry": "dihedral8"}
            ]
            assert validate_spec(spec) == []
            reads[reach] = features(spec)["royal_slides"]
        assert reads == {1: 0.0, 2: 1.0}

    def test_bare_royal_is_its_own_feature(self) -> None:
        chess = _chess()
        twin = copy.deepcopy(chess)
        twin["rules"]["bare_royal"] = "win"
        assert features(chess)["wins_by_bare_royal"] == 0.0
        assert features(twin)["wins_by_bare_royal"] == 1.0


# ---------------------------------------------------------------------------------------
# Q735: the edits in `spec/transform.py` that no test called directly (Q734m, `spec/`).
# Every operator re-stamps its child with `settle_schema_version`, so an edit's own version
# stamp, the defaults it writes and its guards against a malformed spec only show when the
# function is called by itself on a spec that declares schema 1.
# ---------------------------------------------------------------------------------------


def _schema_one_chess() -> dict:
    spec = _chess()
    spec["schema_version"] = 1
    return spec


class TestAnEditStampsTheVersionItNeeds:
    """The version an edit writes is a claim about the features the spec now uses: one too
    low is refused by the validator, one too high overstates (and rehashes) the spec."""

    @pytest.mark.parametrize(
        ("edit", "stamp"),
        [
            (lambda s: transform.restrict_captures(s, "N", ["P"]), 3),
            (lambda s: transform.set_scoring(s, "material"), 3),
            (lambda s: transform.set_scoring(s, "points"), 9),
            (lambda s: transform.set_territory(s), 4),
            (lambda s: transform.set_turn(s), 4),
            (lambda s: transform.set_obstacle(s, "e4"), 6),
            (lambda s: transform.set_facing(s, "N"), 8),
            (lambda s: transform.set_npc(s, "N", ["a3", "a4"]), 9),
            (lambda s: transform.set_npc(s, "N", ["a3", "a4"], immune=["Q"]), 10),
            (lambda s: transform.set_strike(s), 6),
            (lambda s: transform.set_stone(s, "X"), 10),
            (lambda s: transform.set_explosion(s), 2),
            (lambda s: transform.set_explosion(s, by=["Q"]), 10),
            (lambda s: transform.set_explosion(s, of=["Q"]), 10),
            (lambda s: transform.set_explosion(s, by=[]), 10),
            (lambda s: transform.set_contacts(s), 10),
            (lambda s: transform.set_own_link(s, {"mover": "P", "onto": "N", "own": True,
                                                  "mover_to": "B"}), 11),
            (lambda s: transform.set_landing(transform.set_strike(s), "strike"), 12),
        ],
        ids=[
            "restrict_captures", "scoring", "scoring-points", "territory", "turn", "obstacle",
            "facing", "npc", "npc-immune", "strike", "stone", "explosion", "explosion-by",
            "explosion-of", "explosion-by-empty", "contacts", "own-link", "landing",
        ],
    )
    def test_the_stamp_on_schema_one_chess(self, edit: Callable[[dict], dict], stamp: int) -> None:
        assert edit(_schema_one_chess())["schema_version"] == stamp

    def test_an_edit_never_lowers_a_higher_stamp(self) -> None:
        spec = _schema_one_chess()
        spec["schema_version"] = 12
        assert transform.set_territory(spec)["schema_version"] == 12


class TestTheDefaultsAnEditWrites:
    def test_the_explosion_defaults_to_atomics_radius(self) -> None:
        spec = transform.set_explosion(_chess())
        assert spec["rules"]["explode_on_capture"] == {"radius": 1}

    def test_a_new_wall_is_mirrored(self) -> None:
        spec = transform.set_obstacle(_chess(), "e4")
        assert spec["rules"]["obstacles"] == {"squares": ["e4"], "mirror": True}

    def test_a_wild_piece_carries_a_bounty_only_when_asked(self) -> None:
        bare = transform.set_npc(_chess(), "N", ["a3", "a4"])
        assert "bounty" not in bare["rules"]["npcs"][0]
        paid = transform.set_npc(_chess(), "N", ["a3", "a4"], bounty=2)
        assert paid["rules"]["npcs"][0]["bounty"] == 2

    def test_a_territory_floor_is_a_bare_majority_of_the_smallest_board(self) -> None:
        """A 2x2 board is four squares: a majority is three, and the floor is three."""
        spec = _chess()
        spec["board"] = {"width": 2, "height": 2}
        assert transform.set_territory(spec)["rules"]["territory"]["win_at"] == 3

    def test_a_split_moves_half_the_squares_by_default(self) -> None:
        spec = transform.split_piece(_chess(), "P", "X")
        held = {p["piece"]: p["squares"] for p in spec["setup"]["placements"]
                if p["piece"] in ("P", "X")}
        assert len(held["P"]) == 4
        assert len(held["X"]) == 4

    def test_a_split_of_an_undeclared_piece_changes_nothing(self) -> None:
        before = _chess()
        assert transform.split_piece(copy.deepcopy(before), "Z", "X") == before

    def test_set_facing_gives_a_facing_to_that_piece_alone(self) -> None:
        spec = transform.set_facing(_chess(), "N")
        assert {p["id"]: p.get("facing") for p in spec["pieces"] if "facing" in p} == {
            "N": {"axes": 4}
        }


class TestScoringChoosesTheCountTheValuesCanSupply:
    def test_equal_values_count_pieces(self) -> None:
        spec = _chess()
        for piece in spec["pieces"]:
            piece["value"] = 1
        assert transform.set_scoring(spec)["rules"]["scoring"] == {"count": "pieces"}

    def test_two_distinct_values_count_material(self) -> None:
        spec = _chess()
        for piece in spec["pieces"]:
            piece["value"] = 1 if piece["id"] in ("P", "N", "B") else 3
        assert transform.set_scoring(spec)["rules"]["scoring"] == {"count": "material"}


class TestTheLadderRanksAPieceWithNoValueAsOne:
    """`set_strike`, `set_contacts` and `merge_links` rank by declared value and read an
    undeclared one as 1: against a 1.5 it ranks below, against a 0.5 above. A default of 2
    flips the first order and a default of 0 the second."""

    @staticmethod
    def _ladder(*others: dict) -> dict:
        spec = {"pieces": [{"id": "X"}, *others]}
        return spec

    def test_set_strike(self) -> None:
        above = transform.set_strike(self._ladder({"id": "Y", "value": 1.5}))
        assert above["rules"]["strike"]["chain"] == [{"from": "Y", "to": "X"}]
        below = transform.set_strike(self._ladder({"id": "Z", "value": 0.5}))
        assert below["rules"]["strike"]["chain"] == [{"from": "X", "to": "Z"}]

    def test_set_contacts(self) -> None:
        above = transform.set_contacts(self._ladder({"id": "Y", "value": 1.5}))
        assert [(link["mover"], link["onto"]) for link in above["rules"]["contacts"]["links"]] == [("X", "Y")]
        below = transform.set_contacts(self._ladder({"id": "Z", "value": 0.5}))
        assert [(link["mover"], link["onto"]) for link in below["rules"]["contacts"]["links"]] == [("Z", "X")]

    def test_merge_links(self) -> None:
        above = transform.merge_links(self._ladder({"id": "Y", "value": 1.5}))
        assert [(link["mover"], link["onto"], link["mover_to"]) for link in above] == [("X", "X", "Y")]
        below = transform.merge_links(self._ladder({"id": "Z", "value": 0.5}))
        assert [(link["mover"], link["onto"], link["mover_to"]) for link in below] == [("Z", "Z", "X")]


class TestDroppingAPieceKeepsWhatStillStands:
    def test_a_promotion_that_only_named_the_dropped_piece_goes_alone(self) -> None:
        """P promotes only into X and N into Q: dropping X removes P's rule and keeps N's,
        and the promoting piece N still holds en passant's premise."""
        spec = _chess()
        spec["pieces"].append({"id": "X", "name": "x", "moves": []})
        spec["rules"]["promotion"] = [
            {"piece": "P", "rank": "last", "to": ["X"]},
            {"piece": "N", "rank": "last", "to": ["Q"]},
        ]
        spec = transform.drop_piece(spec, "X")
        assert spec["rules"]["promotion"] == [{"piece": "N", "rank": "last", "to": ["Q"]}]
        assert spec["rules"]["en_passant"] is True

    def test_en_passant_survives_while_a_promoting_piece_does(self) -> None:
        spec = transform.drop_piece(_chess(), "N")
        assert spec["rules"]["promotion"]["to"] == ["Q", "R", "B"]
        assert spec["rules"]["en_passant"] is True

    def test_a_hand_that_fills_from_captures_by_default_survives(self) -> None:
        """`from_captures` unwritten means true: the hand has something to fill it."""
        spec = _chess()
        spec["rules"]["drops"] = {"no_doubled": ["P"]}
        spec = transform.drop_piece(spec, "N")
        assert spec["rules"]["drops"] == {"no_doubled": ["P"]}

    @pytest.mark.parametrize(
        ("count", "survives"), [("points", False), ("material", True)]
    )
    def test_only_a_points_count_goes_with_the_last_wild_piece(
        self, count: str, survives: bool
    ) -> None:
        spec = _chess()
        spec["rules"]["npcs"] = [{"piece": "N", "route": ["a3", "a4"]}]
        spec["rules"]["scoring"] = {"count": count}
        spec = transform.drop_piece(spec, "N")
        assert "npcs" not in spec["rules"]
        assert ("scoring" in spec["rules"]) is survives

    def test_a_strike_keeps_the_links_between_the_pieces_that_remain(self) -> None:
        spec = transform.set_strike(_chess())
        spec = transform.drop_piece(spec, "P")
        assert spec["rules"]["strike"]["chain"] == [
            {"from": "Q", "to": "R"}, {"from": "R", "to": "N"}, {"from": "N", "to": "B"},
        ]
        assert spec["rules"]["strike"]["immune"] == ["K"]

    def test_the_bare_capture_spelling_is_not_judged_by_the_sides_it_switches(self) -> None:
        """`{"effect": "capture"}` is the default said aloud and stays legal; it names no
        piece, and dropping one is not a reason to read it as a strike that says nothing."""
        spec = _chess()
        spec["rules"]["strike"] = {"effect": "capture"}
        spec = transform.drop_piece(spec, "P")
        assert spec["rules"]["strike"] == {"effect": "capture"}

    @pytest.mark.parametrize(
        ("switching", "stays"), [(["Q"], True), (["P"], False)], ids=["still-live", "emptied"]
    )
    def test_a_capture_strike_lives_by_the_sides_it_still_switches(
        self, switching: list[str], stays: bool
    ) -> None:
        """Dropping P empties a `switches_sides` naming only P, and a capture strike that
        said nothing else goes with it; one that still switches Q stays."""
        spec = _chess()
        spec["rules"]["strike"] = {
            "effect": "capture", "position": "advance", "immune": ["K"],
            "chain": [{"from": "Q", "to": "R"}], "switches_sides": switching,
        }
        spec = transform.drop_piece(spec, "P")
        assert ("strike" in spec["rules"]) is stays

    def test_a_strike_with_no_link_left_is_no_rule(self) -> None:
        spec = transform.set_strike(_chess())
        for gone in ("P", "B", "N", "R"):
            spec = transform.drop_piece(spec, gone)
        assert "strike" not in spec["rules"]


class TestTheGuardsAgainstAMalformedSpec:
    def test_enpassant_without_a_promotion_is_left_alone(self) -> None:
        spec = _chess()
        spec["rules"].pop("promotion")
        assert transform.settle_en_passant(copy.deepcopy(spec)) == spec

    def test_pieces_that_can_stand_of_a_spec_missing_either_half(self) -> None:
        assert transform.pieces_that_can_stand({"rules": {}}) == set()
        assert transform.pieces_that_can_stand({"pieces": _chess()["pieces"]}) == set()

    def test_wild_pieces_pay_of_a_spec_with_no_rules(self) -> None:
        assert transform.wild_pieces_pay({"pieces": _chess()["pieces"]}) is False

    def test_settle_inert_rules_of_a_spec_with_no_rules(self) -> None:
        assert transform.settle_inert_rules({"pieces": _chess()["pieces"]}) == []

    @pytest.mark.parametrize("junk", ["junk", {"name": "nameless"}], ids=["not-a-dict", "no-id"])
    def test_a_piece_that_is_not_a_piece_is_skipped(self, junk: object) -> None:
        spec = _chess()
        spec["pieces"].append(junk)
        assert transform.capture_is_possible(spec) is True
        assert transform.settle_inert_rules(spec) == []

    def test_keeps_a_survivor_skips_a_piece_with_no_id(self) -> None:
        spec = transform.set_strike(_chess())
        spec["pieces"].append({"name": "nameless"})
        assert transform.keeps_a_survivor(spec, "strike") is True


class TestSettlingInertRules:
    def test_a_promotion_list_stays_a_list_when_one_entry_goes(self) -> None:
        """An unplaced Z promoting into Q: only its entry goes, and the survivor keeps the
        list spelling (a lone dict would be another spelling, and another hash)."""
        spec = _chess()
        spec["pieces"].append({"id": "Z", "name": "z", "moves": [
            {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]})
        spec["rules"].pop("en_passant")
        spec["rules"]["promotion"] = [
            {"piece": "P", "rank": "last", "to": ["Q"]},
            {"piece": "Z", "rank": "last", "to": ["Q"]},
        ]
        assert transform.settle_inert_rules(spec) == ["promotion:Z"]
        assert spec["rules"]["promotion"] == [{"piece": "P", "rank": "last", "to": ["Q"]}]

    def test_the_last_promotion_takes_en_passant_with_it(self) -> None:
        """A promotion only an unplaced pawn-like Z has goes, and en passant with it, and
        the two are reported in that order."""
        spec = _chess()
        spec["pieces"].append({"id": "Z", "name": "z", "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move"}]})
        spec["rules"]["promotion"] = {"piece": "Z", "rank": "last", "to": ["Q"]}
        assert transform.settle_inert_rules(spec) == ["promotion:Z", "en_passant"]
        assert "promotion" not in spec["rules"]
        assert "en_passant" not in spec["rules"]


class TestNeverForcedWithNoPlyLimit:
    def test_the_no_progress_clock_is_the_default_ply_limit_and_no_limit_is_added(self) -> None:
        spec = _chess()
        spec["rules"].pop("max_plies")
        rules = transform.never_forced(spec)["rules"]
        assert rules["no_progress_draw"] == 400
        assert "max_plies" not in rules


class TestClearFacingTakesOnePieceBack:
    def test_only_that_pieces_facing_and_starting_facing_go(self) -> None:
        spec = load_spec(RULESETS / "chess-oriented-pawns.json")
        spec["pieces"].reverse()
        _piece(spec, "N")["facing"] = {"axes": 4}
        spec["setup"]["placements"].append(
            {"piece": "N", "squares": ["d4"], "player": 0, "facing": "forward"}
        )
        spec = transform.clear_facing(spec, "P")
        assert {p["id"]: p.get("facing") for p in spec["pieces"] if "facing" in p} == {
            "N": {"axes": 4}
        }
        faced = {p["piece"] for p in spec["setup"]["placements"] if "facing" in p}
        assert faced == {"N"}


class TestTheChainsAnEditWrites:
    def test_a_demoting_strike_walks_down_the_values(self) -> None:
        strike = transform.set_strike(_chess(), "demote")["rules"]["strike"]
        assert strike["chain"] == [
            {"from": "Q", "to": "R"}, {"from": "R", "to": "N"},
            {"from": "N", "to": "B"}, {"from": "B", "to": "P"},
        ]
        assert strike["immune"] == ["K"]
        assert strike["effect"] == "demote"
        assert strike["position"] == "stay"

    def test_a_promoting_strike_walks_up_the_values(self) -> None:
        chain = transform.set_strike(_chess(), "promote")["rules"]["strike"]["chain"]
        assert chain == [
            {"from": "P", "to": "N"}, {"from": "N", "to": "B"},
            {"from": "B", "to": "R"}, {"from": "R", "to": "Q"},
        ]

    def test_a_contact_table_links_each_rung_to_the_next(self) -> None:
        contacts = transform.set_contacts(_chess(), "promote")["rules"]["contacts"]
        assert contacts["links"] == [
            {"mover": "P", "onto": "N", "mover_to": "N"},
            {"mover": "N", "onto": "B", "mover_to": "B"},
            {"mover": "B", "onto": "R", "mover_to": "R"},
            {"mover": "R", "onto": "Q", "mover_to": "Q"},
        ]
        assert contacts["immune"] == ["K"]
        assert contacts["position"] == "advance"

    def test_a_demoting_contact_table_walks_down(self) -> None:
        links = transform.set_contacts(_chess(), "demote")["rules"]["contacts"]["links"]
        assert [(link["mover"], link["onto"]) for link in links] == [
            ("Q", "R"), ("R", "N"), ("N", "B"), ("B", "P"),
        ]


def _struck_chess_with_a_link_only_from_the_immune_king() -> dict:
    spec = _chess()
    spec["rules"]["strike"] = {
        "effect": "demote", "position": "stay",
        "chain": [{"from": "K", "to": "Q"}], "immune": ["K"],
    }
    return spec


class TestChoosingWhereTheSurvivorLands:
    def test_a_strike_that_leaves_nobody_standing_is_not_given_a_landing(self) -> None:
        spec = _struck_chess_with_a_link_only_from_the_immune_king()
        assert transform.keeps_a_survivor(spec, "strike") is False
        before = copy.deepcopy(spec)
        assert transform.set_landing(spec, "strike") == before

    def test_a_strike_that_still_keeps_a_survivor_is_left_under_choose(self) -> None:
        spec = transform.set_landing(transform.set_strike(_chess()), "strike")
        assert spec["rules"]["strike"]["position"] == "choose"
        before = copy.deepcopy(spec)
        assert transform.settle_landing(spec) == before

    def test_a_choose_with_nothing_to_place_becomes_advance_without_its_landing(self) -> None:
        spec = _struck_chess_with_a_link_only_from_the_immune_king()
        spec["rules"]["strike"]["position"] = "choose"
        spec["rules"]["strike"]["landing"] = transform.step_landing(spec)
        rule = transform.settle_landing(spec)["rules"]["strike"]
        assert rule["position"] == "advance"
        assert "landing" not in rule

    def test_a_landing_beside_another_position_is_dropped(self) -> None:
        spec = transform.set_strike(_chess())
        spec["rules"]["strike"]["landing"] = transform.step_landing(spec)
        assert "landing" not in transform.settle_landing(spec)["rules"]["strike"]


class TestMergingYourOwn:
    def test_chess_offers_the_pawn_onto_each_rung_but_the_top(self) -> None:
        assert transform.merge_links(_chess()) == [
            {"mover": "P", "onto": "P", "own": True, "mover_to": "N"},
            {"mover": "P", "onto": "N", "own": True, "mover_to": "B"},
            {"mover": "P", "onto": "B", "own": True, "mover_to": "R"},
            {"mover": "P", "onto": "R", "own": True, "mover_to": "Q"},
        ]

    def test_a_pair_that_has_an_own_link_is_not_offered_again(self) -> None:
        spec = _chess()
        spec["rules"]["contacts"] = {
            "position": "advance", "immune": ["K"],
            "links": [{"mover": "P", "onto": "N", "own": True, "mover_to": "B"}],
        }
        assert ("P", "N") not in {(link["mover"], link["onto"]) for link in transform.merge_links(spec)}

    def test_a_foe_link_does_not_take_the_pair(self) -> None:
        spec = _chess()
        spec["rules"]["contacts"] = {
            "position": "advance", "immune": ["K"],
            "links": [{"mover": "P", "onto": "N", "mover_to": "B"}],
        }
        assert ("P", "N") in {(link["mover"], link["onto"]) for link in transform.merge_links(spec)}

    def test_an_immune_piece_is_not_merged_onto(self) -> None:
        spec = _chess()
        spec["rules"]["contacts"] = {"position": "advance", "immune": ["K", "N"], "links": []}
        assert "N" not in {link["onto"] for link in transform.merge_links(spec)}


class TestTheRoundOperators:
    """Schema 26 (Q918a): the operators that enter the tie and `beats` on a game played in
    rounds, which the corpus-wide enter arm cannot reach until a committed game is one."""

    @pytest.mark.parametrize(
        ("operator", "parent", "name"),
        [
            ("break-the-tie", _parent_rounds, "rules.simultaneous.tie 'initiative'"),
            ("break-the-tie", lambda: _parent_circle() | {"rules": {"win": "royal_capture", "simultaneous": {}}},
             "rules.simultaneous.tie 'impossible'"),
            ("set-a-collision", _parent_rounds, "pieces[].beats"),
        ],
    )
    def test_it_enters_the_feature(self, operator: str, parent: Callable[[], dict], name: str) -> None:
        apply = next(op for op in OPERATORS if op.name == operator)
        for seed in _SWEEP_SEEDS:
            child = parent()
            if apply.apply(child, random.Random(seed)) is None:
                continue
            if name in schema._schema_twenty_six_uses(child) and not refusal(parent(), child):
                return
        raise AssertionError(f"{operator} never entered {name}")
