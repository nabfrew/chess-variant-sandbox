"""Schema 19 (Q771): portals, `docs/plan-portals.md` §§1-5 and §7 P1.

A pair of portals is two ends on the board; a piece whose move ends on an end open to its
player comes out one king's step beyond the other end, in the direction it travelled (the
sign of its vector), and its move ends there. The core's transits are held against
`tests/portal_oracle.py`, a declared test oracle that reads the spec's descriptors and walks
them square by square, over random walks of four portal games and a set of hand-built
positions, one for each case of the plan's §2.5 and N2.

The plan asked for the §2.5 cases as cargo tests. A `Game` cannot be built in `cargo test`:
its `Tables` are read from the dict `engine/bridge.py` sends, and the extension-module build
links no Python to make one. So they are here, driven through the core as every other
movegen test is, one test per case and named for it.

`TestTheBumpMovedNothing` is the inertness proof: every committed and archived spec hashes
and compiles as `FROZEN` pinned it, and chess is still chess.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from portal_oracle import PortalOracle, Position
from test_schema_nine import FROZEN

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_PORTAL,
    move_from,
    move_portal,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import NO_SQUARE
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.measurement import tables_fingerprint
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import (
    SCHEMA_VERSION,
    _schema_nineteen_uses,
)
from chess_variant_sandbox.spec.transform import settle_schema_version

ROOT = Path(__file__).resolve().parents[1]
STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "move"}


def _ruleset(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8"))


def _with_portals(
    parent: str, pairs: list[dict[str, Any]], moves: list[dict[str, Any]] | None = None,
    **rules: Any,
) -> dict[str, Any]:
    spec = _ruleset(parent)
    spec["schema_version"] = 19
    spec["name"] = f"{parent}+portals"
    spec["pieces"].append(
        {"id": "O", "name": "portal", "value": 0, "moves": [] if moves is None else moves}
    )
    spec["rules"]["portals"] = pairs
    spec["rules"].update(rules)
    return spec


def wormhole() -> dict[str, Any]:
    """§6's Wormhole: Gardner with one fixed two-way pair on b3 and d3."""
    return _with_portals("gardner", [{"piece": "O", "orange": "b3", "blue": "d3"}])


def ferry() -> dict[str, Any]:
    """Ferry (*Showcase games*): Los Alamos, two numbered pairs either player moves."""
    both = {"orange": [0, 1], "blue": [0, 1]}
    return _with_portals(
        "losalamos",
        [
            {"piece": "O", "orange": "b3", "blue": "e4", "moved_by": both},
            {"piece": "O", "orange": "e3", "blue": "b4", "moved_by": both},
        ],
        [STEP],
    )


def carousel() -> dict[str, Any]:
    """Carousel: Los Alamos on a 6 x 8 cylinder, each pair one way for its owner only."""
    spec = _with_portals(
        "losalamos",
        [
            {"piece": "O", "orange": "b3", "blue": "c5",
             "passage": {"0": "orange_to_blue", "1": "closed"}},
            {"piece": "O", "orange": "b6", "blue": "c4",
             "passage": {"0": "closed", "1": "orange_to_blue"}},
        ],
        wrap={"files": "both"},
    )
    spec["board"]["height"] = 8
    return spec


def keepers() -> dict[str, Any]:
    """§6's Keepers on chess: each end moved by one player, one way per player."""
    return _with_portals(
        "chess",
        [{"piece": "O", "orange": "a4", "blue": "h5",
          "moved_by": {"orange": [0], "blue": [1]},
          "passage": {"0": "orange_to_blue", "1": "blue_to_orange"}}],
        [STEP],
    )


GAMES = {"wormhole": wormhole, "ferry": ferry, "carousel": carousel, "keepers": keepers}


def _position(engine: Engine, state: Any) -> Position:
    spec = engine.spec
    found: Position = {}
    for square, value in enumerate(state.board):
        if value:
            owner = spec.owner_of(value)
            found[square] = (
                spec.piece_ids[spec.type_of(value)], "portal" if owner == WILD else owner
            )
    return found


def _unmoved(state: Any) -> set[int]:
    return {square for square in range(len(state.board)) if state.unmoved >> square & 1}


def _core_transits(engine: Engine, state: Any) -> set[tuple[int, int, int, str | None]]:
    spec = engine.spec
    found = set()
    for move in engine.pseudo_legal_moves(state):
        if move_type(move) != MOVE_PORTAL:
            continue
        promoted = move_promotion(move)
        entry = engine.portal_entry(state, move)
        assert entry is not None
        found.add(
            (move_from(move), entry, move_to(move),
             spec.piece_ids[promoted - 1] if promoted else None)
        )
    return found


def _core_portal_moves(engine: Engine, state: Any) -> set[tuple[int, int]]:
    spec = engine.spec
    return {
        (move_from(move), move_to(move))
        for move in engine.pseudo_legal_moves(state)
        if state.board[move_from(move)] and spec.owner_of(state.board[move_from(move)]) == WILD
    }


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _texts(engine: Engine, state: Any) -> list[str]:
    return engine.move_texts(state)


class TestTheBumpMovedNothing:
    """Every committed and archived spec hashes and compiles as `FROZEN` pinned it."""

    @pytest.mark.parametrize("path", sorted(FROZEN), ids=lambda p: Path(p).stem)
    def test_every_ruleset_is_the_game_it_was(self, path: str) -> None:
        spec = load_spec(ROOT / path)
        digest, tables = FROZEN[path]
        assert spec_hash(spec) == digest, f"{path} is not the file that was frozen"
        assert tables_fingerprint(spec) == tables, f"{path} compiles to different tables"

    def test_chess_is_still_chess(self) -> None:
        assert spec_hash(_ruleset("chess")) == "4c1922ffae26"

    def test_only_the_showcases_use_schema_nineteen(self) -> None:
        """Nothing written before the bump uses it: the users are exactly Q772a's seven
        showcases (`docs/reference/portal-showcases.md`), all in `archive/`, and schema
        20's Ferry beside them (Q807). Q772e's two-way and Q772f's fixed Divide, and
        `wormhole-adrift`, left the library on 2026-10-12 (Q868). Q844b's three portal
        showcases of schemas 21 and 24 joined them on 2026-10-13."""
        assert sorted(
            path for path in FROZEN if _schema_nineteen_uses(load_spec(ROOT / path))
        ) == sorted(
            f"archive/{name}.json"
            for name in ("carousel", "crease", "cylinder-wormhole", "divide", "ferry",
                         "ferry-no-straight-back", "pawnproof-wormhole", "shut-wormhole",
                         "sluice", "wormhole")
        )

    def test_a_game_without_portals_sends_every_portal_key_empty(self) -> None:
        tables = canonical_tables(compile_spec(_ruleset("chess")))
        for key in ("has_portals", "portal_exits", "leap_dirs", "attack_dirs"):
            assert key not in tables


class TestTheVersionIsBumped:
    def test_the_schema_version_is_nineteen_or_later(self) -> None:
        assert SCHEMA_VERSION >= 19

    def test_the_collector_names_each_key(self) -> None:
        assert _schema_nineteen_uses(wormhole()) == {"rules.portals"}
        assert _schema_nineteen_uses(ferry()) == {"rules.portals", "rules.portals.moved_by"}
        assert _schema_nineteen_uses(carousel()) == {"rules.portals", "rules.portals.passage"}

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_an_older_stamp_is_refused_by_name(self, name: str) -> None:
        spec = GAMES[name]()
        assert validate_spec(spec) == [], validate_spec(spec)
        spec["schema_version"] = 18
        assert any("added in schema 19" in error for error in validate_spec(spec))

    def test_a_portal_game_settles_to_nineteen(self) -> None:
        spec = wormhole()
        spec["schema_version"] = 1
        assert settle_schema_version(spec)["schema_version"] == 19


def _refusal(spec: dict[str, Any], fragment: str) -> None:
    errors = validate_spec(spec)
    assert any(fragment in error for error in errors), errors


class TestWhatTheValidatorRefuses:
    def test_a_portal_may_not_capture(self) -> None:
        spec = ferry()
        spec["pieces"][-1]["moves"] = [{**STEP, "mode": "both"}]
        _refusal(spec, 'must say mode "move"')

    def test_a_portal_moves_alike_for_both_players(self) -> None:
        spec = ferry()
        spec["pieces"][-1]["moves"] = [
            {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}
        ]
        _refusal(spec, "reads differently for the two players")

    def test_a_portal_nobody_moves_may_have_no_moves_but_a_moved_one_may_not(self) -> None:
        assert validate_spec(wormhole()) == []
        spec = ferry()
        spec["pieces"][-1]["moves"] = []
        _refusal(spec, "has no moves, but rules.portals lets a player move")

    def test_a_portal_is_worth_nothing_and_never_royal(self) -> None:
        spec = wormhole()
        spec["pieces"][-1]["value"] = 1
        _refusal(spec, "must have value 0")

    def test_a_portal_is_not_placed_by_the_setup(self) -> None:
        spec = wormhole()
        spec["setup"]["placements"].append({"piece": "O", "squares": ["c3"]})
        _refusal(spec, "may not be placed by setup.placements")

    def test_an_end_on_an_occupied_square_is_refused(self) -> None:
        spec = wormhole()
        spec["rules"]["portals"][0]["orange"] = "b2"
        _refusal(spec, "is a square setup.placements puts a piece on")

    def test_two_ends_on_one_square_are_refused(self) -> None:
        spec = wormhole()
        spec["rules"]["portals"].append({"piece": "O", "orange": "b3", "blue": "c3"})
        _refusal(spec, "two portals on one square")

    @pytest.mark.parametrize("rule, value", [
        ("territory", {"mode": "permanent"}),
        ("explode_on_capture", {"radius": 1}),
    ])
    def test_rules_refused_beside_portals(self, rule: str, value: Any) -> None:
        spec = wormhole()
        spec["rules"][rule] = value
        _refusal(spec, f"refused beside rules.{rule}")

    def test_a_hex_lattice_is_refused(self) -> None:
        spec = wormhole()
        spec["board"]["lattice"] = "hex"
        _refusal(spec, "refused on a hex lattice")

    def test_a_piece_may_not_take_the_name_of_an_end(self) -> None:
        spec = wormhole()
        spec["pieces"][1]["id"] = "1o"
        for placement in spec["setup"]["placements"]:
            if placement["piece"] == "N":
                placement["piece"] = "1o"
        _refusal(spec, "is the name of a portal end")

    def test_passage_words_are_checked(self) -> None:
        spec = wormhole()
        spec["rules"]["portals"][0]["passage"] = {"0": "sideways"}
        _refusal(spec, "passage.0 must be one of")


class TestTheCompiledRule:
    def test_each_end_is_an_internal_type_of_the_third_owner(self) -> None:
        compiled = compile_spec(ferry())
        assert compiled.piece_ids[-4:] == ("1o", "1b", "2o", "2b")
        assert compiled.n_owners == 3
        portals = compiled.portals
        assert portals is not None
        start = compiled.initial_board[compiled.square_index(1, 2)]
        assert compiled.owner_of(start) == WILD and compiled.piece_ids[compiled.type_of(start)] == "1o"

    def test_no_type_may_take_an_end(self) -> None:
        compiled = compile_spec(wormhole())
        ends = compiled.portals.end_types
        assert all(not row[end] for row in compiled.capture_allowed for end in ends)

    def test_nothing_is_one_way_where_a_portal_is_open(self) -> None:
        assert compile_spec(_ruleset("gardner")).one_way_types
        assert not compile_spec(wormhole()).one_way_types

    def test_the_momentum_is_the_sign_of_the_vector(self) -> None:
        compiled = compile_spec(wormhole())
        knight = compiled.piece_ids.index("N")
        b1 = compiled.square_index(1, 0)
        (targets, _, _), = compiled.leaps[0][knight][b1]
        masks = compiled.portals.leap_dirs[0][knight][b1][0]
        by_target = dict(zip(targets, masks, strict=True))
        # (1, 2) to c3 is up and to the right: direction 1 (north-east) for both modes.
        assert by_target[compiled.square_index(2, 2)] == (1 << 1) | (1 << 9)

    def test_the_tables_cross_with_the_flag(self) -> None:
        tables = build_tables(compile_spec(carousel()))
        assert tables["has_portals"] is True
        assert len(tables["portal_exits"]) == 2 * 48 * 8
        assert len(tables["leap_dirs"]) == len(tables["leap_squares"])
        assert len(tables["attack_dirs"]) * 8 == len(tables["attack_srcs"])


class TestTheOracleAgreesOnRandomWalks:
    """The core's transits and portal moves equal the oracle's, ply by ply."""

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_random_walks(self, name: str) -> None:
        spec = GAMES[name]()
        engine = Engine(spec)
        oracle = PortalOracle(spec)
        rng = random.Random(19)
        seen = 0
        for _ in range(4):
            state = engine.initial_state()
            for _ in range(60):
                if engine.terminal(state):
                    break
                position = _position(engine, state)
                player = state.side_to_move
                transits = oracle.transits(position, player, _unmoved(state))
                assert _core_transits(engine, state) == transits
                assert _core_portal_moves(engine, state) == oracle.portal_moves(position, player)
                seen += len(transits)
                legal = engine.legal_moves(state)
                assert set(legal) <= set(engine.pseudo_legal_moves(state))
                for move in legal:
                    after = engine.apply(state, move)
                    king = next(
                        (sq for sq, (pid, owner) in _position(engine, after).items()
                         if pid == "K" and owner == player),
                        None,
                    )
                    if king is not None:
                        assert not oracle.attacked_through(
                            _position(engine, after), 1 - player, king, _unmoved(after)
                        ), engine.describe_move(state, move)
                state = engine.apply(state, rng.choice(legal))
                assert state.key == engine._compute_hash(state)
        assert seen > 0


def _placed(spec: dict[str, Any], placement: dict[str, tuple[str, int]], side: int = 0,
            moved: tuple[str, ...] = ()) -> tuple[Engine, Any]:
    engine = Engine(spec)
    return engine, engine.make_state(placement, side_to_move=side, moved=moved)


def _sparse(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with its setup cut to a king and a pawn a side, freeing every other square
    for a portal; a hand-built position ignores the setup but the validator reads it."""
    spec = copy.deepcopy(spec)
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["e1"]}, {"piece": "P", "squares": ["a2"]},
    ]
    return spec


def _bare(spec: dict[str, Any], squares: dict[str, str]) -> dict[str, Any]:
    """``spec`` with its portals on ``squares`` (end name -> square) and an empty setup but
    the two kings, so a hand-built position validates as the same game."""
    spec = copy.deepcopy(spec)
    for index, pair in enumerate(spec["rules"]["portals"]):
        pair["orange"] = squares.get(f"{index + 1}o", pair["orange"])
        pair["blue"] = squares.get(f"{index + 1}b", pair["blue"])
    return spec


class TestTheLandingRules:
    """`docs/plan-portals.md` §2.5, one case each, and N2's two."""

    def _wormhole(self, placement: dict[str, tuple[str, int]], side: int = 0,
                  spec: dict[str, Any] | None = None) -> tuple[Engine, Any]:
        base = {"b3": ("1o", WILD), "d3": ("1b", WILD), "e1": ("K", 0), "a5": ("K", 1)}
        return _placed(spec or wormhole(), {**base, **placement}, side)

    def test_a_transit_comes_out_one_step_beyond_in_the_direction_travelled(self) -> None:
        engine, state = self._wormhole({"b1": ("R", 0)})
        assert "Rb1-b3>d4" in _texts(engine, state)

    def test_a_knight_and_a_bishop_both_come_out_on_the_diagonal(self) -> None:
        engine, state = self._wormhole({"a1": ("N", 0), "c2": ("B", 0)})
        texts = _texts(engine, state)
        assert "Na1-b3>e4" in texts  # (1, 2): up and to the right
        assert "Bc2-b3>c4" in texts  # (-1, 1): up and to the left

    def test_a_blocked_exit_is_no_move(self) -> None:
        engine, state = self._wormhole({"b1": ("R", 0), "d4": ("P", 0)})
        assert not [t for t in _texts(engine, state) if t.startswith("Rb1-b3>")]

    def test_an_exit_off_the_board_is_no_move(self) -> None:
        spec = _bare(wormhole(), {"1b": "e3"})
        engine, state = _placed(
            spec, {"b3": ("1o", WILD), "e3": ("1b", WILD), "e1": ("K", 0), "a5": ("K", 1),
                   "a3": ("R", 0)},
        )
        assert not [t for t in _texts(engine, state) if t.startswith("Ra3-b3>")]

    def test_an_exit_onto_a_wall_is_no_move(self) -> None:
        spec = wormhole()
        engine, state = self._wormhole({"a3": ("R", 0)}, spec=spec)
        assert "Ra3-b3>e3" in _texts(engine, state)
        spec["rules"]["obstacles"] = {"squares": ["e3"], "mirror": False}
        engine, state = self._wormhole({"a3": ("R", 0)}, spec=spec)
        assert not [t for t in _texts(engine, state) if t.startswith("Ra3-b3>")]

    def test_an_exit_onto_another_portal_is_blocked(self) -> None:
        spec = _with_portals("gardner", [
            {"piece": "O", "orange": "b3", "blue": "d3"},
            {"piece": "O", "orange": "e3", "blue": "c3"},
        ])
        engine, state = _placed(spec, {
            "b3": ("1o", WILD), "d3": ("1b", WILD), "e3": ("2o", WILD), "c3": ("2b", WILD),
            "e1": ("K", 0), "e5": ("K", 1), "a3": ("R", 0),
        })
        # Right into b3 would come out right of d3, on e3: the orange end of pair 2.
        assert not [t for t in _texts(engine, state) if t.startswith("Ra3-b3>")]

    def test_a_piece_it_may_not_take_blocks_the_exit(self) -> None:
        spec = wormhole()
        rook = next(p for p in spec["pieces"] if p["id"] == "R")
        rook["captures"] = ["P", "N", "B", "R", "K"]
        engine, state = self._wormhole({"b1": ("R", 0), "d4": ("Q", 1)}, spec=spec)
        assert not [t for t in _texts(engine, state) if t.startswith("Rb1-b3>")]
        engine, state = self._wormhole({"b1": ("R", 0), "d4": ("N", 1)}, spec=spec)
        assert "Rb1-b3>xd4" in _texts(engine, state)

    def test_coming_back_to_the_origin_is_not_a_move(self) -> None:
        engine, state = self._wormhole({"c3": ("R", 0)})
        texts = _texts(engine, state)
        # Left into b3 comes out left of d3, on c3: the rook's own square, a pass.
        assert not [t for t in texts if t.startswith("Rc3-b3>")]
        # Right into d3 comes out right of b3, on c3 again.
        assert not [t for t in texts if t.startswith("Rc3-d3>")]

    def test_a_capture_happens_at_the_exit(self) -> None:
        engine, state = self._wormhole({"b1": ("R", 0), "d4": ("N", 1)})
        move = engine.find_move(state, "Rb1-b3>xd4")
        after = engine.apply(state, move)
        assert _position(engine, after)[_sq(engine, "d4")] == ("R", 0)
        assert after.board[_sq(engine, "b1")] == 0
        assert _position(engine, after)[_sq(engine, "b3")] == ("1o", "portal")

    def test_a_pawn_promotes_where_it_comes_out(self) -> None:
        spec = _sparse(_with_portals("gardner", [{"piece": "O", "orange": "c4", "blue": "b4"}]))
        engine, state = _placed(spec, {
            "c4": ("1o", WILD), "b4": ("1b", WILD), "e1": ("K", 0), "e3": ("K", 1),
            "c3": ("P", 0),
        })
        texts = _texts(engine, state)
        assert {"Pc3-c4>b5=Q", "Pc3-c4>b5=N"} <= set(texts)
        assert "Pc3-c4>b5" not in texts  # promotion is mandatory here
        move = engine.find_move(state, "Pc3-c4>b5=Q")
        assert _position(engine, engine.apply(state, move))[_sq(engine, "b5")] == ("Q", 0)

    def test_a_pawns_push_comes_out_only_onto_an_empty_square(self) -> None:
        engine, state = self._wormhole({"b2": ("P", 0), "d4": ("N", 1)})
        assert not [t for t in _texts(engine, state) if t.startswith("Pb2-b3>")]

    def test_a_transit_opens_no_en_passant_window(self) -> None:
        spec = _with_portals("chess", [{"piece": "O", "orange": "e3", "blue": "c5"}])
        engine, state = _placed(spec, {
            "e3": ("1o", WILD), "c5": ("1b", WILD), "e1": ("K", 0), "e8": ("K", 1),
            "e2": ("P", 0), "d7": ("P", 1),
        })
        after = engine.apply(state, engine.find_move(state, "Pe2-e3>c6"))
        assert after.ep_target == NO_SQUARE

    def test_a_closed_end_is_an_obstacle(self) -> None:
        spec = wormhole()
        spec["rules"]["portals"][0]["passage"] = {"0": "closed"}
        engine, state = self._wormhole({"b1": ("R", 0)}, spec=spec)
        texts = _texts(engine, state)
        assert not [t for t in texts if t.startswith("Rb1-b3")]
        assert "Rb1-b2" in texts and "Rb1-b4" not in texts

    def test_a_one_way_pair_is_entered_only_at_its_open_end(self) -> None:
        spec = wormhole()
        spec["rules"]["portals"][0]["passage"] = {"0": "orange_to_blue"}
        engine, state = self._wormhole({"b1": ("R", 0), "d1": ("R", 0)}, spec=spec)
        texts = _texts(engine, state)
        assert "Rb1-b3>d4" in texts
        assert not [t for t in texts if t.startswith("Rd1-d3>")]

    def test_two_ways_round_a_cylinder_are_two_moves(self) -> None:
        spec = carousel()
        engine, state = _placed(spec, {
            "b3": ("1o", WILD), "c5": ("1b", WILD), "b6": ("2o", WILD), "c4": ("2b", WILD),
            "d1": ("K", 0), "d8": ("K", 1), "e3": ("R", 0),
        })
        texts = _texts(engine, state)
        # Left along the third rank into b3 comes out left of c5; right the long way
        # round the seam comes out on its right.
        assert {"Re3-b3>b5", "Re3-b3>d5"} <= set(texts)

    def test_an_attack_through_a_portal_is_check(self) -> None:
        engine, state = self._wormhole({"b1": ("R", 1), "a5": ("K", 1), "d4": ("K", 0),
                                        "e1": ("N", 0)})
        assert engine.in_check(state, 0)

    def test_a_move_that_lets_an_enemy_through_to_the_king_is_illegal(self) -> None:
        engine, state = self._wormhole({"b1": ("R", 1), "b2": ("N", 0), "d4": ("K", 0),
                                        "e1": ("N", 0)})
        assert not engine.in_check(state, 0)
        assert not [t for t in _texts(engine, state) if t.startswith("Nb2")]

    def test_a_portal_move_can_give_check(self) -> None:
        spec = ferry()
        engine, state = _placed(spec, {
            "b3": ("1o", WILD), "e4": ("1b", WILD), "e3": ("2o", WILD), "b4": ("2b", WILD),
            "a1": ("K", 0), "f5": ("K", 1), "b1": ("R", 0),
        })
        # Up the b-file into b3 the rook comes out above the blue end: e5 now, which is
        # empty. Step the blue end to f4 and it comes out on f5.
        assert not engine.in_check(state, 1)
        move = engine.find_move(state, "1b:e4-f4")
        after = engine.apply(state, move)
        # Up into b3 comes out above f4, on f5: the black king.
        assert engine.in_check(after, 1)

    def test_a_portal_move_that_exposes_the_movers_king_is_illegal(self) -> None:
        spec = ferry()
        engine, state = _placed(spec, {
            "b3": ("1o", WILD), "e4": ("1b", WILD), "e3": ("2o", WILD), "b4": ("2b", WILD),
            "f5": ("K", 0), "a6": ("K", 1), "b1": ("R", 1),
        })
        # Moving the blue end to f4 would let the rook come out of it onto f5.
        assert "1b:e4-f4" not in _texts(engine, state)
        assert "1b:e4-d4" in _texts(engine, state)


class TestTheMoveWord:
    def test_a_transit_names_the_end_it_went_in_by(self) -> None:
        engine, state = TestTheLandingRules()._wormhole({"b1": ("R", 0)})
        move = engine.find_move(state, "Rb1-b3>d4")
        assert move_type(move) == MOVE_PORTAL
        assert engine.spec.piece_ids[move_portal(move)] == "1o"

    def test_a_transit_is_reversible_and_keeps_the_repetition_window(self) -> None:
        placement = {"b3": ("1o", WILD), "d3": ("1b", WILD), "e1": ("K", 0), "a5": ("K", 1),
                     "b1": ("R", 0)}
        # Nothing unmoved, so no right is lost and only the move itself could clear it.
        engine, state = _placed(wormhole(), placement, moved=tuple(
            ("b3", "d3", "e1", "a5", "b1")))
        after = engine.apply(state, engine.find_move(state, "Rb1-b3>d4"))
        assert len(after.repetition) == len(state.repetition) + 1
        assert after.halfmove == state.halfmove + 1


class TestFairyStockfishRefuses:
    def test_portals_are_refused_by_name(self) -> None:
        with pytest.raises(Untranslatable, match="rules.portals"):
            translate(compile_spec(wormhole()))
