"""Schema 16: the native hexagon and its three seams (Q281a4, the gate).

`board.shape: "hexagon"` makes a hex rhombus's corners absent rather than walled, and
`rules.wrap` joins a hexagon's three pairs of opposite sides, `n_s`, `ne_sw` and `nw_se`,
each independently and per player (`docs/game-ideas.md` *Q281 specified*;
`docs/reference/hex-build-plan.md` *Q281a4*). Q281a1 to a3 built the resolver, the oracle,
the compiler and the readers behind a validator that refused the keys; this file is what
opening the gate owes:

* **the gate**: the version, the collector, the nine refusals by name;
* **the twin**: `archive/hex-glinski-hexagon` compiles to its walled parent's tables;
* **the oracle end to end**: `tests/hex_geometry.py` against every target the engine
  generates through `compile_spec`, over a2's grid of settings;
* **the pre-registered lone-piece table**, out of the engine, and the symmetries;
* **every reader on the real showcase files**, and the operators.

R >= 4 is `slow` wherever a whole grid of settings is compiled: a torus at R = 7 takes
about nine seconds to compile.
"""

from __future__ import annotations

import copy
import functools
import json
import random
from pathlib import Path
from typing import Any

import hex_geometry as oracle
import pytest
from retired_specs import REBUILT

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import move_to
from chess_variant_sandbox.loop.operators import (
    OPERATORS,
    _clear_pairs,
    _join_the_edges,
    _move_a_placement,
    _resize,
    _wall_off_a_square,
)
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.complexity import _RULE_COSTS, measure_complexity
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.relate import identity_key
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, SHAPES, WRAP_PAIRS, absent_squares
from tests.feature_inventory import _schema_sixteen_uses

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
PIECE_ORDER = ("rook", "bishop", "queen", "knight", "king")
PIECE_IDS = {"rook": "R", "bishop": "B", "queen": "Q", "knight": "N", "king": "K"}
PAIRS = tuple(WRAP_PAIRS)
ZOBRIST = frozenset(
    {"zobrist_pieces", "zobrist_side", "zobrist_ep", "zobrist_colour", "zobrist_turn"}
)

# The walled parent's ladder (`tests/test_schema_fourteen.py`), which the twin inherits.
LADDER = {1: 51, 2: 2_541, 3: 132_581}
LADDER_DEEP = {4: 6_873_785}

# a2's grid: every pair independently "none", "both" or one named side.
SHARED_SETTINGS: list[dict[str, str]] = [
    {"n_s": n_s, "ne_sw": ne_sw, "nw_se": nw_se}
    for n_s in ("none", "n", "s", "both")
    for ne_sw in ("none", "ne", "sw", "both")
    for nw_se in ("none", "nw", "se", "both")
]


def _load(name: str) -> dict[str, Any]:
    if name in REBUILT:  # retired by Q868 or Q870; rebuilt from a game in the library
        return REBUILT[name]()
    return json.loads((ARCHIVE / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def parent() -> dict[str, Any]:
    return _load("hex-glinski-promoting")


@pytest.fixture
def twin() -> dict[str, Any]:
    return _load("hex-glinski-hexagon")


@pytest.fixture
def flanks() -> dict[str, Any]:
    return _load("hex-glinski-flanks")


def _errors(spec: dict[str, Any], fragment: str) -> list[str]:
    return [error for error in validate_spec(spec) if fragment in error]


def _label(setting: dict[str, Any]) -> str:
    return ",".join(f"{key}={value}" for key, value in setting.items() if value != "none") or "none"


# -------------------------------------------------------------------- a lone piece


def _lone_spec(radius: int, wrap: dict[str, Any], pieces: tuple[str, ...]) -> dict[str, Any]:
    """The five oracle pieces on a native hexagon of ``radius``, one type each, no royal."""
    width = 2 * radius + 1
    declared = []
    for name in pieces:
        moves = []
        slides = [list(v) for v, slide in oracle.PIECES[name] if slide]
        leaps = [list(v) for v, slide in oracle.PIECES[name] if not slide]
        if slides:
            moves.append({"kind": "slide", "vectors": slides})
        if leaps:
            moves.append({"kind": "leap", "vectors": leaps})
        declared.append({"id": PIECE_IDS[name], "name": name, "value": 5, "moves": moves})
    cells = [oracle.name(cell) for cell in oracle.cells(radius)]
    first = PIECE_IDS[pieces[0]]
    rules: dict[str, Any] = {"win": "annihilation"}
    if wrap:
        rules["wrap"] = wrap
    return {
        "schema_version": 16,
        "name": "lone pieces",
        "board": {"width": width, "height": width, "lattice": "hex", "shape": "hexagon"},
        "pieces": declared,
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": first, "player": 0, "squares": [cells[0]]},
            {"piece": first, "player": 1, "squares": [cells[-1]]},
        ]},
        "rules": rules,
    }


def _pieces_at(radius: int, wrap: dict[str, Any]) -> tuple[str, ...]:
    """Every oracle piece the geometry answers for at this radius.

    At R = 1 a knight's leap of 3 is more than R + 1: refusal 7 refuses it with a side
    joined, and the oracle raises rather than answer where it lands two copies out
    (`tests/test_hex_geometry.py`), so a2 left it out there too.
    """
    del wrap
    return tuple(p for p in PIECE_ORDER if p != "knight") if radius == 1 else PIECE_ORDER


# For the table and the symmetries, at R = 3 and 5, which reuse a handful of settings.
@functools.lru_cache(maxsize=16)
def _engine(radius: int, wrap_json: str) -> Engine:
    wrap = json.loads(wrap_json)
    spec = _lone_spec(radius, wrap, _pieces_at(radius, wrap))
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _engine_for(radius: int, wrap: dict[str, Any]) -> Engine:
    return _engine(radius, json.dumps(wrap, sort_keys=True))


def _engine_targets(engine: Engine, cell: oracle.Cell, piece: str, player: int = 0) -> set[int]:
    state = engine.make_state({oracle.name(cell): (PIECE_IDS[piece], player)}, side_to_move=player)
    moves = engine.legal_moves(state)
    assert engine.perft(state, 1) == len(moves)
    targets = [move_to(m) for m in moves]
    assert len(targets) == len(set(targets)), "a target generated twice"
    return set(targets)


def _oracle_targets(radius: int, leave: frozenset[str], cell: oracle.Cell, piece: str) -> set[int]:
    width = 2 * radius + 1
    step = oracle.hex_step(radius, leave)
    return {oracle.square(t, width) for t in oracle.targets(step, cell, oracle.PIECES[piece])}


def _wrap_json(setting: dict[str, Any]) -> dict[str, Any]:
    """A setting, ``{pair: value}`` or ``{pair: (p0, p1)}``, as `rules.wrap` spells it."""
    out: dict[str, Any] = {}
    for key, value in setting.items():
        if isinstance(value, tuple) and value[0] != value[1]:
            out[f"{key}_by_player"] = {"0": value[0], "1": value[1]}
        else:
            shared = value[0] if isinstance(value, tuple) else value
            if shared != "none":
                out[key] = shared
    return out


def _leave(setting: dict[str, Any], player: int) -> frozenset[str]:
    return oracle.may_leave({
        key: (value[player] if isinstance(value, tuple) else value)
        for key, value in setting.items()
    })


def _per_player_sample(seed: int, count: int) -> list[dict[str, tuple[str, str]]]:
    """a2's seeded sample of per-player settings, a dozen (the plan's question 5)."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        out.append({
            key: (
                rng.choice(("none", pair.low, pair.high, "both")),
                rng.choice(("none", pair.low, pair.high, "both")),
            )
            for key, pair in WRAP_PAIRS.items()
        })
    return out


def _check_against_the_oracle(radius: int, setting: dict[str, Any]) -> None:
    # Built here rather than through the cache: each setting is compiled once, and a
    # near-torus at R = 7 is several hundred megabytes of tables.
    wrap = _wrap_json(setting)
    spec = _lone_spec(radius, wrap, _pieces_at(radius, wrap))
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    for player in (0, 1):
        leave = _leave(setting, player)
        for piece in _pieces_at(radius, wrap):
            for cell in oracle.cells(radius):
                expected = _oracle_targets(radius, leave, cell, piece)
                got = _engine_targets(engine, cell, piece, player)
                assert got == expected, (radius, setting, player, piece, oracle.name(cell))


# ======================================================================= the gate


class TestTheVersionIsBumped:
    def test_the_schema_version_is_sixteen_or_later(self) -> None:
        assert SCHEMA_VERSION >= 16
        assert SHAPES == ("grid", "hexagon")

    def test_the_collector_names_both_features(self, twin, flanks) -> None:
        assert _schema_sixteen_uses(twin) == {"board.shape=hexagon"}
        assert _schema_sixteen_uses(flanks) == {"board.shape=hexagon", "rules.wrap.hexagon"}
        inert = copy.deepcopy(twin)
        inert["rules"]["wrap"] = {"n_s_by_player": {"0": "none", "1": "none"}}
        assert "rules.wrap.hexagon" in _schema_sixteen_uses(inert)

    def test_the_preamble_teaches_it(self) -> None:
        """Q281a5: the model is told the shape and all three pairs, with their sides."""
        from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE

        assert '`board.shape: "hexagon"`' in SCHEMA_PREAMBLE
        for key, pair in WRAP_PAIRS.items():
            assert f"`{key}`" in SCHEMA_PREAMBLE, key
            assert f'`"{pair.low}"`' in SCHEMA_PREAMBLE, pair.low
            assert f'`"{pair.high}"`' in SCHEMA_PREAMBLE, pair.high
        assert "`n_s_by_player`" in SCHEMA_PREAMBLE


class TestTheNineRefusals:
    """`docs/game-ideas.md` *Q281 specified*, *Validator refusals*, each by its message."""

    def test_1_a_shape_outside_the_two(self, twin) -> None:
        twin["board"]["shape"] = "circle"
        assert _errors(twin, "board.shape must be one of ('grid', 'hexagon'), got 'circle'")

    @pytest.mark.parametrize("lattice", ["square", None])
    def test_2_a_hexagon_off_the_hex_lattice(self, twin, lattice) -> None:
        if lattice is None:
            twin["board"].pop("lattice")
        else:
            twin["board"]["lattice"] = lattice
        assert _errors(twin, 'board.shape "hexagon" needs board.lattice "hex"')

    @pytest.mark.parametrize(("width", "height"), [(11, 9), (10, 10), (12, 12)])
    def test_3_a_hexagon_that_is_not_odd_and_square(self, twin, width, height) -> None:
        twin["board"].update(width=width, height=height)
        (error,) = _errors(twin, "a hexagon of cells is as tall as it is wide, and odd")
        assert f"got {width} by {height}" in error

    @pytest.mark.parametrize("radius", [1, 7])
    def test_3_every_radius_from_one_to_seven_is_a_board(self, radius) -> None:
        assert validate_spec(_lone_spec(radius, {}, ("rook",))) == []

    # Refusal 4: the eight sets, each naming `a1`, a corner that is not a cell.
    def test_4_a_placement(self, twin) -> None:
        twin["setup"]["placements"].append({"piece": "P", "squares": ["a1"]})
        assert _errors(twin, "setup.placements stands a piece off the hexagon at a1")

    def test_4_a_placement_through_its_mirror(self, twin) -> None:
        """k11 is the corner a1 turns into, so a1 is named by the rotated army."""
        twin["setup"]["placements"].append({"piece": "P", "squares": ["k11"]})
        assert _errors(twin, "setup.placements stands a piece off the hexagon at k11")

    def test_4_a_reach_goal(self, twin) -> None:
        twin["rules"]["reach"] = {"piece": "K", "squares": ["a1", "f6"], "mirror": "rotate"}
        assert _errors(twin, "rules.reach.squares names a square off the hexagon at a1")

    def test_4_a_scoring_region(self, twin) -> None:
        twin["rules"]["scoring"] = {"count": "region", "region": ["a1", "f6"]}
        assert _errors(twin, "rules.scoring.region names a square off the hexagon at a1")

    def test_4_a_drop_region(self, twin) -> None:
        twin["rules"]["drops"] = {"from_captures": True, "region": {"P": ["a1", "f6"]}}
        assert _errors(twin, "rules.drops.region['P'] names a square off the hexagon at a1")

    def test_4_a_promotion_zone(self, twin) -> None:
        twin["rules"]["promotion"]["squares"].append("a1")
        assert _errors(twin, "rules.promotion.squares names a square off the hexagon at a1")

    def test_4_a_wild_route(self, twin) -> None:
        twin["rules"]["npcs"] = [{"piece": "N", "route": ["a1", "f6"]}]
        assert _errors(twin, "rules.npcs[0] walks off the hexagon at a1")

    def test_4_castling_off_the_middle_rank(self, twin) -> None:
        """Schema 18 (Q781) reads castling's span, not its rank. A hexagon's row is one
        run of cells, so a castling between a king and a rook on it never leaves them: it
        is the walled rank. A destination off the hexagon is refused."""
        twin["rules"]["castling"] = {"king": "K", "rook": "R", "king_steps": 2}
        assert validate_spec(twin) == []
        for placement in twin["setup"]["placements"]:
            if placement["piece"] == "R":
                placement["squares"] = ["c4", "f1"]
            if placement["piece"] == "B":
                placement["squares"] = ["i1", "f2", "f3"]
        twin["rules"]["castling"] = {
            "king": "K", "rook": "R", "king_file": {"queenside": "d", "kingside": "i"},
        }
        (error,) = _errors(twin, "rules.castling happens along rank 1, which runs off the hexagon")
        assert "at d1, e1 between the king" in error

    def test_4_a_wall_on_a_corner(self, twin) -> None:
        twin["rules"]["obstacles"] = {"squares": ["a1", "f6"]}
        assert _errors(twin, "rules.obstacles.squares names a square off the hexagon at a1")

    def test_4_a_wall_inside_the_hexagon_is_still_a_wall(self, twin) -> None:
        twin["rules"]["obstacles"] = {"squares": ["f6"]}
        assert validate_spec(twin) == []

    def test_4_a_rank_spelled_zone_stays_lawful(self, twin) -> None:
        """The coordinator's call (2026-09-27): lawful as it is with walls."""
        twin["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["Q"]}
        assert _errors(twin, "off the hexagon") == []

    @pytest.mark.parametrize("key", ["files", "ranks_by_player"])
    def test_5_a_rectangles_keys_on_a_hexagon(self, twin, key) -> None:
        twin["rules"]["wrap"] = {key: "both" if key == "files" else {"0": "top", "1": "top"}}
        assert _errors(twin, f"rules.wrap.{key} joins a rectangle's edges, and this board is a hexagon")

    @pytest.mark.parametrize("key", PAIRS)
    def test_5_a_hexagons_keys_on_a_grid(self, key) -> None:
        chess = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        chess["schema_version"] = 16
        chess["rules"]["wrap"] = {key: "both"}
        assert _errors(chess, f"rules.wrap.{key} joins a hexagon's sides, and this board is not one")

    def test_5_a_hexagons_keys_on_a_hex_rhombus(self, parent) -> None:
        parent["schema_version"] = 16
        parent["rules"]["wrap"] = {"nw_se": "both"}
        assert _errors(parent, "rules.wrap.nw_se joins a hexagon's sides")

    @pytest.mark.parametrize(("key", "value"), [("n_s", "left"), ("ne_sw", "n"), ("nw_se", "forward")])
    def test_6_a_value_outside_the_pairs_four(self, twin, key, value) -> None:
        twin["rules"]["wrap"] = {key: value}
        assert _errors(twin, f"rules.wrap.{key} must be one of")

    def test_6_no_forward_shorthand(self, twin) -> None:
        """The user, 2026-09-28: sides per player only."""
        twin["rules"]["wrap"] = {"n_s_by_player": {"0": "forward", "1": "forward"}}
        assert _errors(twin, "rules.wrap.n_s_by_player['0'] must be one of")

    def test_6_a_pair_beside_its_per_player_twin(self, twin) -> None:
        twin["rules"]["wrap"] = {"n_s": "both", "n_s_by_player": {"0": "n", "1": "s"}}
        assert _errors(twin, "rules.wrap has both 'n_s' and 'n_s_by_player'")

    def test_6_a_per_player_object_keyed_otherwise(self, twin) -> None:
        twin["rules"]["wrap"] = {"ne_sw_by_player": {"0": "ne"}}
        assert _errors(twin, "rules.wrap.ne_sw_by_player must be an object keyed by '0' and '1'")

    def test_7_a_vector_longer_than_r_plus_one_with_a_side_joined(self, flanks) -> None:
        rook = next(p for p in flanks["pieces"] if p["id"] == "R")
        rook["moves"].append({"kind": "leap", "vectors": [[7, 0]]})
        (error,) = _errors(flanks, "steps further than 6 cells")
        assert error.startswith("pieces[") and "radius 5" in error

    def test_7_is_lawful_with_no_side_joined(self, twin) -> None:
        rook = next(p for p in twin["pieces"] if p["id"] == "R")
        rook["moves"].append({"kind": "leap", "vectors": [[7, 0]]})
        assert validate_spec(twin) == []

    def test_7_r_plus_one_itself_is_lawful(self, flanks) -> None:
        rook = next(p for p in flanks["pieces"] if p["id"] == "R")
        rook["moves"].append({"kind": "leap", "vectors": [[6, 0], [3, 3], [-6, 6]]})
        assert _errors(flanks, "steps further") == []

    def test_7_a_landing(self, flanks) -> None:
        flanks["rules"]["contacts"] = {
            "position": "choose", "landing": {"vectors": [[1, 0], [4, 3]]}, "links": [],
        }
        assert _errors(flanks, "rules.contacts.landing steps further than 6 cells")

    @pytest.mark.parametrize(("radius", "blast", "refused"), [(3, 2, False), (3, 3, True), (1, 1, False)])
    def test_7_a_blast(self, radius, blast, refused) -> None:
        spec = _lone_spec(radius, {"n_s": "both"}, ("rook",))
        spec["rules"]["explode_on_capture"] = {"radius": blast}
        assert bool(_errors(spec, "rules.explode_on_capture.radius")) is refused

    def test_7_a_knight_at_radius_one(self) -> None:
        spec = _lone_spec(1, {"nw_se": "se"}, ("rook", "knight"))
        assert _errors(spec, "pieces[1].moves[0] steps further than 2 cells")

    @pytest.mark.parametrize(
        ("wrap", "refused"),
        [({"n_s": "both"}, True), ({"ne_sw_by_player": {"0": "ne", "1": "none"}}, True),
         ({"nw_se": "both"}, False), ({}, False)],
    )
    def test_8_no_doubled_across_a_folding_pair(self, twin, wrap, refused) -> None:
        twin["rules"]["drops"] = {"from_captures": True, "no_doubled": ["P"]}
        if wrap:
            twin["rules"]["wrap"] = wrap
        assert bool(_errors(twin, "rules.drops.no_doubled on a hexagon")) is refused

    def test_9_blocks_leaps_on_a_hexagon(self, twin) -> None:
        twin["rules"]["obstacles"] = {"squares": ["f6"], "blocks_leaps": True}
        assert _errors(twin, "rules.obstacles.blocks_leaps on a hexagon")

    def test_9_is_not_refused_on_a_rhombus(self, parent) -> None:
        """Schema 14's gap stays open on the rhombus: refusing it would invalidate specs."""
        parent["rules"]["obstacles"]["blocks_leaps"] = True
        assert _errors(parent, "blocks_leaps") == []

    def test_a_sweep_beside_a_hexagon_seam_is_still_refused(self, flanks) -> None:
        rook = next(p for p in flanks["pieces"] if p["id"] == "R")
        rook["moves"][0]["kind"] = "sweep"
        assert _errors(flanks, "wrap")


# ======================================================================= the twin


class TestTheTwin:
    """`hex-glinski-hexagon` is `hex-glinski-promoting` spelled as a shape."""

    def test_it_is_the_parent_with_the_walls_made_a_shape(self, parent, twin) -> None:
        rebuilt = copy.deepcopy(parent)
        rebuilt["board"]["shape"] = "hexagon"
        del rebuilt["rules"]["obstacles"]
        for key in ("schema_version", "name", "description", "provenance"):
            rebuilt.pop(key, None)
            twin.pop(key, None)
        assert rebuilt == twin

    def test_the_absent_cells_are_the_parents_walls(self, parent, twin) -> None:
        compiled = compile_spec(parent)
        assert absent_squares(twin["board"]) == compiled.obstacles
        assert compile_spec(twin).absent == compiled.obstacles

    def test_its_tables_equal_the_parents_array_by_array(self, parent, twin) -> None:
        """All but the Zobrist draws, which are seeded by the mechanics hash.

        So the tables fingerprint differs (`tests/test_schema_nine.py::FROZEN`), as
        `wolves-immune`'s does from `wolves`'. The core's transposition table matches keys
        exactly, so the draws change no move and no score short of a 64-bit collision.
        """
        mine = canonical_tables(compile_spec(twin))
        theirs = canonical_tables(compile_spec(parent))
        assert mine.keys() == theirs.keys()
        differ = {key for key in mine if mine[key] != theirs[key]}
        assert differ == ZOBRIST

    @pytest.mark.parametrize("depth", sorted(LADDER))
    def test_the_ladder(self, twin, depth) -> None:
        engine = Engine(twin)
        assert engine.perft(engine.initial_state(), depth) == LADDER[depth]

    @pytest.mark.slow
    @pytest.mark.parametrize("depth", sorted(LADDER_DEEP))
    def test_the_deep_ladder(self, twin, depth) -> None:
        engine = Engine(twin)
        assert engine.perft(engine.initial_state(), depth) == LADDER_DEEP[depth]

    def test_the_flanks_opening_is_the_twins(self, twin, flanks) -> None:
        """No crossing is possible at ply 1 (the spec, *The showcases*)."""
        a, b = Engine(twin), Engine(flanks)
        assert sorted(a.legal_moves(a.initial_state())) == sorted(b.legal_moves(b.initial_state()))


class TestTheSeamPlays:
    """The short-game walk (`tests/test_schema_two.py`) walks every archived spec for
    twenty plies, and at ply 1 nothing can cross the flanks' seam. This walks until
    something does, checking the incremental key against a full recompute every ply."""

    @pytest.mark.parametrize("name", ["hex-glinski-flanks", "hex-glinski-flanks-one-way"])
    def test_a_random_game_crosses_and_the_key_holds(self, twin, name) -> None:
        seamed = Engine(_load(name))
        plain = Engine(twin)
        crossings = 0
        for seed in range(4):
            rng = random.Random(seed)
            state = seamed.initial_state()
            for _ in range(120):
                moves = seamed.legal_moves(state)
                if seamed.terminal(state, moves) is not None:
                    break
                move = rng.choice(moves)
                # A crossing is a move the unjoined twin's tables do not have.
                crossings += move not in plain.legal_moves(state)
                state = seamed.apply(state, move)
                assert state.key == seamed._compute_hash(state), (name, seed)
        assert crossings > 0, f"{name}: four walks played no move across the seam"


# ============================================================ the oracle, end to end


class TestTheOracleThroughTheEngine:
    """a1's oracle against every target `Engine` generates for a lone piece, both players.

    a2 compared the landing function itself; this goes through `validate_spec`,
    `compile_spec`, the bridge and the core. R >= 4 is `slow`.
    """

    @pytest.mark.parametrize("radius", [1, 2, 3])
    @pytest.mark.parametrize("setting", SHARED_SETTINGS, ids=_label)
    def test_shared_settings(self, radius, setting) -> None:
        _check_against_the_oracle(radius, setting)

    @pytest.mark.slow
    @pytest.mark.parametrize("radius", [4, 5, 6, 7])
    @pytest.mark.parametrize("setting", SHARED_SETTINGS, ids=_label)
    def test_shared_settings_larger_radii(self, radius, setting) -> None:
        _check_against_the_oracle(radius, setting)

    @pytest.mark.parametrize("setting", _per_player_sample(3, 12), ids=_label)
    def test_per_player_sample_at_r3(self, setting) -> None:
        _check_against_the_oracle(3, setting)

    @pytest.mark.slow
    @pytest.mark.parametrize("setting", _per_player_sample(5, 12), ids=_label)
    def test_per_player_sample_at_r5(self, setting) -> None:
        _check_against_the_oracle(5, setting)


# ====================================================== the pre-registered table


# `docs/game-ideas.md` *What a seam does to the tables*, as `tests/test_hex_geometry.py`
# holds it for the oracle; here out of the engine.
TABLES: dict[int, dict[str, tuple[int, ...]]] = {
    5: {
        "nothing joined": (2_070, 1_080, 3_150, 720, 900),
        "one pair both ways": (3_276, 5_364, 6_614, 844, 964),
        "one pair one way": (2_708, 3_246, 5_101, 782, 932),
        "n_s and ne_sw both ways": (8_190, 7_232, 8_190, 968, 1_028),
        "the torus": (8_190, 8_190, 8_190, 1_092, 1_092),
    },
    3: {
        "nothing joined": (498, 246, 744, 216, 324),
        "one pair both ways": (774, 1_002, 1_214, 292, 364),
        "one pair one way": (651, 632, 1_044, 254, 344),
        "n_s and ne_sw both ways": (1_332, 1_296, 1_332, 368, 404),
        "the torus": (1_332, 1_332, 1_332, 444, 444),
    },
}
ROWS: dict[str, list[dict[str, str]]] = {
    "nothing joined": [{}],
    "one pair both ways": [{pair: "both"} for pair in PAIRS],
    "one pair one way": [
        {pair: side} for pair in PAIRS for side in oracle.PAIR_SIDES[pair]
    ],
    "n_s and ne_sw both ways": [{"n_s": "both", "ne_sw": "both"}],
    "the torus": [{pair: "both" for pair in PAIRS}],
}


def _perft_total(radius: int, wrap: dict[str, str], piece: str) -> int:
    engine = _engine_for(radius, wrap)
    return sum(
        engine.perft(engine.make_state({oracle.name(cell): (PIECE_IDS[piece], 0)}), 1)
        for cell in oracle.cells(radius)
    )


class TestTheLonePieceTable:
    @pytest.mark.parametrize("row", sorted(ROWS))
    def test_at_r3(self, row) -> None:
        for wrap in ROWS[row]:
            got = tuple(_perft_total(3, wrap, piece) for piece in PIECE_ORDER)
            assert got == TABLES[3][row], wrap

    @pytest.mark.slow
    @pytest.mark.parametrize("row", sorted(ROWS))
    def test_at_r5(self, row) -> None:
        for wrap in ROWS[row]:
            got = tuple(_perft_total(5, wrap, piece) for piece in PIECE_ORDER)
            assert got == TABLES[5][row], wrap


# ================================================================== the symmetries


def _turn_cell(radius: int, cell: oracle.Cell) -> oracle.Cell:
    q, r = oracle.turn((cell[0] - radius, cell[1] - radius))
    return q + radius, r + radius


# A sixth of a turn carries (2R+1, -R) to (R, R+1): `se` to `ne`, and on round.
_TURN_SIDE = {"se": "ne", "ne": "n", "n": "nw", "nw": "sw", "sw": "s", "s": "se"}
_TURN_PAIR = {"nw_se": "ne_sw", "ne_sw": "n_s", "n_s": "nw_se"}


def _turn_setting(setting: dict[str, str]) -> dict[str, str]:
    return {
        _TURN_PAIR[key]: _TURN_SIDE.get(value, value) for key, value in setting.items()
    }


class TestTheSymmetries:
    @pytest.mark.parametrize(
        "radius", [3, pytest.param(5, marks=pytest.mark.slow)]
    )
    def test_the_torus_is_the_same_from_every_cell(self, radius) -> None:
        engine = _engine_for(radius, {pair: "both" for pair in PAIRS})
        n = 3 * radius * radius + 3 * radius + 1
        for piece, count in (("rook", n - 1), ("bishop", n - 1), ("queen", n - 1),
                             ("knight", 12), ("king", 12)):
            counts = {len(_engine_targets(engine, cell, piece)) for cell in oracle.cells(radius)}
            assert counts == {count}, piece

    @pytest.mark.parametrize(
        "setting", [{"n_s": "both"}, {"ne_sw": "sw"}, {"nw_se": "se", "n_s": "n"}], ids=_label
    )
    def test_a_sixth_of_a_turn_gives_the_next_pairs_moves(self, setting) -> None:
        radius = 3
        width = 2 * radius + 1
        engine = _engine_for(radius, setting)
        turned = _engine_for(radius, _turn_setting(setting))
        for piece in PIECE_ORDER:
            for cell in oracle.cells(radius):
                image = {
                    oracle.square(_turn_cell(radius, (t % width, t // width)), width)
                    for t in _engine_targets(engine, cell, piece)
                }
                assert image == _engine_targets(turned, _turn_cell(radius, cell), piece)

    @pytest.mark.parametrize("pair", PAIRS)
    def test_both_is_the_union_of_the_two_one_way_sides(self, pair) -> None:
        radius = 3
        low, high = WRAP_PAIRS[pair].low, WRAP_PAIRS[pair].high
        both = _engine_for(radius, {pair: "both"})
        one = _engine_for(radius, {pair: low})
        other = _engine_for(radius, {pair: high})
        for piece in PIECE_ORDER:
            for cell in oracle.cells(radius):
                assert _engine_targets(both, cell, piece) == (
                    _engine_targets(one, cell, piece) | _engine_targets(other, cell, piece)
                )

    def test_per_player_is_each_players_own_setting(self) -> None:
        radius = 3
        split = _engine_for(radius, {"ne_sw_by_player": {"0": "ne", "1": "both"}})
        first = _engine_for(radius, {"ne_sw": "ne"})
        second = _engine_for(radius, {"ne_sw": "both"})
        for piece in PIECE_ORDER:
            for cell in oracle.cells(radius):
                assert _engine_targets(split, cell, piece, 0) == _engine_targets(first, cell, piece, 0)
                assert _engine_targets(split, cell, piece, 1) == _engine_targets(second, cell, piece, 1)


# ============================================================ every reader, on the files


class TestEveryReaderOnTheShowcases:
    def test_the_board_sentence(self, twin, flanks) -> None:
        for spec in (twin, flanks):
            assert "a hexagonal board of 91 cells, six on a side" in describe_spec(spec)
            assert "cut off" not in describe_spec(spec)

    def test_the_seam_sentences(self, flanks) -> None:
        one_way = _load("hex-glinski-flanks-one-way")
        assert "The board has no upper-left or lower-right side" in describe_spec(flanks)
        text = describe_spec(one_way)
        assert "only by leaving by the lower-right side" in text
        assert "only by leaving by the upper-left side" in text

    def test_the_twin_and_its_parent_are_one_node(self, parent, twin) -> None:
        assert identity_key(twin) == identity_key(parent)

    def test_the_twin_and_its_parent_differ_only_where_a3_said(self, parent, twin) -> None:
        mine, theirs = features(twin), features(parent)
        assert {k for k in mine if mine[k] != theirs[k]} == {
            "has_obstacles", "rule_mass", "special_rules",
        }

    def test_the_seam_is_a_distance_and_a_feature(self, twin, flanks) -> None:
        assert features(flanks)["wraps_nw_se"] > 0
        assert features(twin)["wraps_nw_se"] == 0
        assert distance(twin, flanks) > 0

    def test_the_seam_costs_wrap_and_the_shape_costs_nothing(self, parent, twin, flanks) -> None:
        wrap_cost = _RULE_COSTS["wrap"]
        assert measure_complexity(flanks).special_rules == (
            measure_complexity(twin).special_rules + wrap_cost
        )
        # The twin loses the walled parent's `obstacles` charge and gains nothing for its
        # shape, a board key.
        assert measure_complexity(parent).special_rules == (
            measure_complexity(twin).special_rules + _RULE_COSTS["obstacles"]
        )

    @pytest.mark.parametrize(
        "wrap", [{"n_s": "none"}, {"ne_sw_by_player": {"0": "none", "1": "none"}}]
    )
    def test_an_inert_seam_costs_nothing(self, twin, wrap) -> None:
        plain = measure_complexity(twin).special_rules
        twin["rules"]["wrap"] = wrap
        assert measure_complexity(twin).special_rules == plain

    def test_the_json_schema_offers_the_keys(self) -> None:
        schema = rule_spec_schema()["properties"]
        assert schema["board"]["properties"]["shape"]["enum"] == list(SHAPES)
        wrap = schema["rules"]["properties"]["wrap"]["properties"]
        for key, pair in WRAP_PAIRS.items():
            assert wrap[key]["enum"] == ["none", pair.low, pair.high, "both"]
            assert set(wrap[f"{key}_by_player"]["properties"]) == {"0", "1"}


# ================================================================== the operators


class TestTheOperators:
    def test_join_the_edges_joins_only_the_clear_pair_on_glinskis_opening(self, twin) -> None:
        assert _clear_pairs(twin) == ["nw_se"]
        for seed in range(20):
            child = copy.deepcopy(twin)
            assert _join_the_edges(child, random.Random(seed)) is not None
            assert child["rules"]["wrap"] == {"nw_se": "both"}
            assert validate_spec(child) == []

    def test_and_takes_it_away_again(self, twin, flanks) -> None:
        assert _join_the_edges(flanks, random.Random(0)) == "the board has edges again"
        assert spec_hash(flanks) == spec_hash(twin)

    def test_it_declines_when_no_pair_is_clear(self, twin) -> None:
        twin["setup"]["placements"].append({"piece": "P", "squares": ["a6"]})
        assert validate_spec(twin) == []
        assert _clear_pairs(twin) == []
        assert _join_the_edges(twin, random.Random(0)) is None

    def test_resize_declines_a_hexagon(self, twin) -> None:
        assert all(_resize(copy.deepcopy(twin), random.Random(s)) is None for s in range(10))

    def test_walls_and_placements_go_only_on_cells(self, twin) -> None:
        absent = absent_squares(twin["board"])
        for seed in range(40):
            walled = copy.deepcopy(twin)
            if _wall_off_a_square(walled, random.Random(seed)) is not None:
                assert validate_spec(walled) == [], validate_spec(walled)
            moved = copy.deepcopy(twin)
            if _move_a_placement(moved, random.Random(seed)) is not None:
                assert validate_spec(moved) == [], validate_spec(moved)
                for placement in moved["setup"]["placements"]:
                    for name in placement["squares"]:
                        assert (int(name[1:]) - 1) * 11 + ord(name[0]) - ord("a") not in absent

    @pytest.mark.parametrize("name", ["hex-glinski-hexagon", "hex-glinski-flanks"])
    def test_no_operator_raises_on_a_showcase(self, name) -> None:
        spec = _load(name)
        for operator in OPERATORS:
            for seed in range(3):
                operator.apply(copy.deepcopy(spec), random.Random(seed))


# ------------------------------------------------ Q922: the hexagon refusals, edge by edge


def _hexagon_errors(width: Any, rules: dict[str, Any], pieces: list[Any] | None = None) -> list[str]:
    from chess_variant_sandbox.spec.schema import _check_hexagon_coherence

    spec = {"board": {"shape": "hexagon", "width": width, "height": width}, "rules": rules}
    errors: list[str] = []
    _check_hexagon_coherence(spec, pieces or [], rules, errors)
    return errors


JOINED = {"wrap": {"n_s": "wrap"}}
LEAP_AT = [{"id": "L", "moves": [{"kind": "leap", "vectors": [[2, 0]]}]}]


class TestTheHexagonsFrameMustHaveARadius:
    """No refusal below reads a radius from a frame that has none: it is refused elsewhere."""

    @pytest.mark.parametrize("width", [None, "7", 7.0, True, 6, 8, 2])
    def test_a_frame_with_no_centre_is_left_to_its_own_refusal(self, width: Any) -> None:
        rules = {"obstacles": {"squares": [], "blocks_leaps": True}, **JOINED, "drops": {"no_doubled": True}}
        assert _hexagon_errors(width, rules, LEAP_AT) == []

    @pytest.mark.parametrize("width", [3, 5, 7, 9])
    def test_an_odd_frame_is_read(self, width: Any) -> None:
        errors = _hexagon_errors(width, {"obstacles": {"squares": [], "blocks_leaps": True}})
        assert len(errors) == 1 and errors[0].startswith("rules.obstacles.blocks_leaps on a hexagon")


class TestAStepAtMostOneBeyondTheRadiusIsFine:
    """On a hexagon of radius R with a side joined, a step of hex length R + 1 is exact."""

    @pytest.mark.parametrize(
        ("vector", "refused"),
        [([2, 0], False), ([3, 0], False), ([4, 0], True), ([2, 2], True), ([3, -3], False),
         ([-2, -2], True), ([-3, 0], False), ([0, -4], True)],
        ids=str,
    )
    def test_a_landing_vector_on_radius_two(self, vector: list[int], refused: bool) -> None:
        # Width 5 is radius 2, so the bound is 3.
        rules = {**JOINED, "strike": {"effect": "demote", "landing": {"vectors": [vector]}}}
        assert bool(_hexagon_errors(5, rules)) is refused

    @pytest.mark.parametrize(("vector", "refused"), [([3, 0], False), ([4, 0], True)], ids=str)
    def test_a_move_vector_on_radius_two(self, vector: list[int], refused: bool) -> None:
        pieces = [{"id": "L", "moves": [{"kind": "leap", "vectors": [vector]}]}]
        assert bool(_hexagon_errors(5, JOINED, pieces)) is refused

    @pytest.mark.parametrize(
        ("explosion", "refused"),
        [
            ({}, False),                 # the default radius is one: 2 > 2 is not
            ({"radius": 1}, False),
            ({"radius": 2}, True),
            ({"radius": True}, False),
            ({"radius": "2"}, False),
        ],
        ids=str,
    )
    def test_a_blast_on_radius_one(self, explosion: dict, refused: bool) -> None:
        # Width 3 is radius 1, so the bound is 2 and a blast of radius r needs 2r <= 2.
        assert bool(_hexagon_errors(3, {**JOINED, "explode_on_capture": explosion})) is refused

    def test_the_default_blast_on_a_single_cell_is_too_wide(self) -> None:
        assert _hexagon_errors(1, {**JOINED, "explode_on_capture": {}}) != []
        assert _hexagon_errors(1, {**JOINED, "explode_on_capture": {"radius": 0}}) == []


class TestTheLongestVectorIgnoresWhatIsNotAVector:
    @pytest.mark.parametrize(
        ("vectors", "longest"),
        [
            ([[1, 0]], 1),
            ([[2, -1]], 2),
            ([[1, 1]], 2),
            ([[1, 0], [5], [1, 2, 3], [1.5, 0], [True, 9], "ab", None, [0, "a"], [], [3, 0]], 3),
            ([[5], [1, 2, 3], [True, 9], [1.5, 0]], 0),
            ([], 0),
            ("[1,0]", 0),
            (None, 0),
            ({"a": [1, 0]}, 0),
        ],
        ids=str,
    )
    def test_the_longest(self, vectors: Any, longest: int) -> None:
        from chess_variant_sandbox.spec.schema import _longest_hex_vector

        assert _longest_hex_vector(vectors) == longest
