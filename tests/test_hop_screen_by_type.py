"""SR3c (`docs/plan-schema-next.md` §1 rank 7): a hop screen named by type.

`pieces[].moves[].screen` takes a list of piece ids beside `any`/`own`/`enemy`: the types that
may serve as the hop's screen, of either side. Janggi's cannon lists every piece but the cannon,
so it may not jump another cannon (its *capture* of a cannon is `captures`). The core's
`screen_matches` reads the list as a mask over board values (`Tables::screen_types`) at its five
call sites in `game.rs` (movegen, the foreign and held moves, the attack sources), and a pouncer's
reach reads it too (N13: a screen by type names no owner). The games here are xiangqi with its
cannon made Janggi's, only because xiangqi is a committed game with a cannon: nothing below reads
xiangqi's rules.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from test_pounce import _spec as _pounce_spec

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import SCREEN_TYPES, compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.transform import drop_piece
from tests.feature_inventory import _schema_sixty_uses

ROOT = Path(__file__).resolve().parent.parent
HOPPERS = (
    "rulesets/xiangqi.json", "rulesets/bastion.json", "archive/cannonade.json",
    "archive/kraken-ford.json",
)


def _xiangqi() -> dict[str, Any]:
    return load_spec(ROOT / "rulesets" / "xiangqi.json")


def _janggi_cannon(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Xiangqi whose cannon is Janggi's: it moves and takes only by hopping, never over
    another cannon (the screen lists every piece but `C`) and never taking one."""
    spec = copy.deepcopy(spec or _xiangqi())
    others = [piece["id"] for piece in spec["pieces"] if piece["id"] != "C"]
    for piece in spec["pieces"]:
        if piece["id"] == "C":
            piece["moves"] = [{
                "kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "screen": others,
            }]
            piece["captures"] = list(others)
    return spec


def _texts(engine: Engine, state: Any) -> list[str]:
    return [engine.describe_move(state, move) for move in engine.legal_moves(state)]


class TestTheSpelling:
    def test_janggis_cannon_is_valid(self) -> None:
        assert validate_spec(_janggi_cannon()) == []

    def test_an_empty_list_is_refused(self) -> None:
        spec = _janggi_cannon()
        spec["pieces"][5]["moves"][0]["screen"] = []
        assert any("screen is an empty list" in e for e in validate_spec(spec))

    def test_an_unknown_id_is_refused(self) -> None:
        spec = _janggi_cannon()
        spec["pieces"][5]["moves"][0]["screen"] = ["X"]
        assert any("screen entry 'X' is not a declared piece" in e for e in validate_spec(spec))

    def test_a_repeat_is_refused(self) -> None:
        spec = _janggi_cannon()
        spec["pieces"][5]["moves"][0]["screen"] = ["R", "R"]
        assert any("screen repeats a piece id" in e for e in validate_spec(spec))

    def test_a_list_off_a_hop_is_refused(self) -> None:
        spec = _xiangqi()
        spec["pieces"][4]["moves"][0]["screen"] = ["R"]
        assert any("screen only means something on a hop" in e for e in validate_spec(spec))

    def test_the_normal_form_sorts_it(self) -> None:
        spec = _janggi_cannon()
        spec["pieces"][5]["moves"][0]["screen"] = ["R", "A", "S"]
        assert normal_form(spec)["pieces"][5]["moves"][0]["screen"] == ["A", "R", "S"]
        other = copy.deepcopy(spec)
        other["pieces"][5]["moves"][0]["screen"] = ["S", "R", "A"]
        assert spec_hash(spec) == spec_hash(other)

    def test_every_piece_listed_is_any(self) -> None:
        """A list naming every declared piece is `any`: the same name and the same tables."""
        listed = _xiangqi()
        listed["pieces"][5]["moves"][1]["screen"] = [p["id"] for p in listed["pieces"]]
        assert validate_spec(listed) == []
        assert spec_hash(listed) == spec_hash(_xiangqi())
        assert build_tables(compile_spec(listed)) == build_tables(compile_spec(_xiangqi()))
        assert _schema_sixty_uses(listed) == set()


class TestTheOwnerWordsAreTodaysTables:
    @pytest.mark.parametrize("path", HOPPERS)
    def test_a_committed_hopper_crosses_no_type_list(self, path: str) -> None:
        compiled = compile_spec(load_spec(ROOT / path))
        assert compiled.screen_types == ()
        assert build_tables(compiled)["hop_screen_types"] == []
        assert "hop_screen_types" not in canonical_tables(compiled)

    @pytest.mark.parametrize("owner", ["own", "enemy"])
    def test_an_owner_word_crosses_its_old_code(self, owner: str) -> None:
        spec = _xiangqi()
        spec["pieces"][5]["moves"][1]["screen"] = owner
        compiled = compile_spec(spec)
        codes = {e[4] for f in compiled.hops for t in f for sq in t for e in sq}
        assert codes == {1 if owner == "own" else 2}
        assert compiled.screen_types == ()

    def test_a_list_crosses_its_types(self) -> None:
        compiled = compile_spec(_janggi_cannon())
        cannon = compiled.piece_ids.index("C")
        assert compiled.screen_types == (
            tuple(i for i in range(compiled.n_types) if i != cannon),
        )
        codes = {e[4] for f in compiled.hops for t in f for sq in t for e in sq}
        assert codes == {SCREEN_TYPES}
        flags = build_tables(compiled)["hop_screen_types"]
        assert len(flags) == compiled.n_types and flags[cannon] == 0 and sum(flags) == 6


class TestJanggisCannonPlays:
    def test_it_may_not_jump_a_cannon(self) -> None:
        engine = Engine(_janggi_cannon())
        state = engine.make_state({
            "e1": ("K", 0), "d10": ("K", 1), "a1": ("C", 0), "a3": ("C", 1), "b1": ("R", 1),
        })
        texts = [t for t in _texts(engine, state) if t.startswith("Ca1")]
        # Up the file the first piece is a cannon, so nothing; along the rank the chariot
        # screens, and every square past it is a landing.
        assert texts and all(not t.startswith("Ca1-a") and "xa" not in t for t in texts)
        assert {t[-2:] for t in texts} == {"c1", "d1"}

    def test_it_jumps_any_other_piece_of_either_side(self) -> None:
        engine = Engine(_janggi_cannon())
        state = engine.make_state({
            "e1": ("K", 0), "d10": ("K", 1), "a1": ("C", 0), "a3": ("S", 0), "a6": ("R", 1),
        })
        texts = [t for t in _texts(engine, state) if t.startswith("Ca1")]
        assert {t[-2:] for t in texts if t[-2] == "a"} == {"a4", "a5", "a6"}

    def test_it_may_not_take_a_cannon(self) -> None:
        engine = Engine(_janggi_cannon())
        state = engine.make_state({
            "e1": ("K", 0), "d10": ("K", 1), "a1": ("C", 0), "a3": ("S", 0), "a6": ("C", 1),
        })
        texts = [t for t in _texts(engine, state) if t.startswith("Ca1")]
        assert not any(t.endswith("a6") for t in texts)
        assert {t[-2:] for t in texts if t[-2] == "a"} == {"a4", "a5"}

    def test_check_through_a_cannon_is_no_check(self) -> None:
        """The attack sources read the screen too: a cannon behind another cannon gives no
        check, behind a chariot it does, and the filter reads the same answer."""
        engine = Engine(_janggi_cannon())
        screened_by_cannon = engine.make_state(
            {"e1": ("K", 0), "d10": ("K", 1), "d6": ("C", 1), "d2": ("C", 0)}, side_to_move=1
        )
        screened_by_chariot = engine.make_state(
            {"e1": ("K", 0), "d10": ("K", 1), "d6": ("R", 1), "d2": ("C", 0)}, side_to_move=1
        )
        assert not engine.in_check(screened_by_cannon, 1)
        assert engine.in_check(screened_by_chariot, 1)
        # The legality filter asks the same sources: stepping down the file keeps the
        # check behind a chariot, and is free behind a cannon.
        assert "Kd10-d9" not in _texts(engine, screened_by_chariot)
        assert "Kd10-d9" in _texts(engine, screened_by_cannon)

    def test_xiangqis_cannon_still_checks_through_a_cannon(self) -> None:
        engine = Engine(_xiangqi())
        state = engine.make_state(
            {"e1": ("K", 0), "d10": ("K", 1), "d6": ("C", 1), "d2": ("C", 0)}, side_to_move=1
        )
        assert engine.in_check(state, 1)

    def test_a_game_walks_with_the_key_checked_each_ply(self) -> None:
        engine = Engine(_janggi_cannon())
        rng = random.Random(7)
        state = engine.initial_state()
        hopped = 0
        for _ in range(200):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            move = rng.choice(moves)
            hopped += engine.describe_move(state, move).startswith("C")
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
        assert hopped > 0

    def test_perft_matches_apply(self) -> None:
        engine = Engine(_janggi_cannon())
        state = engine.initial_state()
        counted = sum(
            len(engine.legal_moves(engine.apply(state, move))) for move in engine.legal_moves(state)
        )
        assert engine.perft(state, 2) == counted


class TestTheSurfaces:
    def test_the_described_rules_name_the_screens(self) -> None:
        text = describe_spec(_janggi_cannon())
        assert "exactly one piece of these kinds, of either side: A, E, H, K, R, S" in text

    def test_it_is_a_distance_from_any(self) -> None:
        any_screen = _janggi_cannon()
        any_screen["pieces"][5]["moves"][0]["screen"] = "any"
        assert distance(_janggi_cannon(), any_screen) > 0

    def test_dropping_a_listed_piece_leaves_a_valid_game(self) -> None:
        assert validate_spec(drop_piece(_janggi_cannon(), "S")) == []

    def test_dropping_every_listed_piece_drops_the_hop(self) -> None:
        spec = _janggi_cannon()
        spec["pieces"][5]["moves"][0]["screen"] = ["S"]
        dropped = drop_piece(spec, "S")
        cannon = next(p for p in dropped["pieces"] if p["id"] == "C")
        assert cannon["moves"] == []

    def test_the_inventory_names_it(self) -> None:
        assert _schema_sixty_uses(_janggi_cannon()) == {"pieces.moves.hop.screen.types"}
        assert _schema_sixty_uses(_xiangqi()) == set()


def _pouncing_hopper(screen: Any) -> dict[str, Any]:
    """`tests/test_pounce.py`'s board, its lurker a rook-line hopper whose screen is `screen`."""
    spec = _pounce_spec()
    lurker = next(p for p in spec["pieces"] if p["id"] == "Z")
    lurker["moves"] = [{"kind": "hop", "vectors": [[1, 0]], "symmetry": "dihedral8",
                        "screen": screen}]
    return spec


class TestAPouncerScreenedByType:
    """N13 (`docs/plan-schema-next.md` §2): a pouncing hop whose screen names an owner stays
    refused, and one whose screen names types is allowed: it names no owner."""

    def test_an_owner_word_is_refused_and_a_list_is_not(self) -> None:
        assert any("screen names an owner" in e for e in validate_spec(_pouncing_hopper("own")))
        assert validate_spec(_pouncing_hopper(["T"])) == []

    def test_the_reach_crosses_its_screen(self) -> None:
        tables = build_tables(compile_spec(_pouncing_hopper(["T"])))
        assert set(tables["pounce_reach_screen"]) == {0, SCREEN_TYPES}  # the hounds read any
        assert build_tables(compile_spec(_pouncing_hopper("any")))["pounce_reach_screen"] == []

    @pytest.mark.parametrize(("screen", "pounces"), [("T", True), ("S", False)])
    def test_it_pounces_only_over_a_listed_screen(self, screen: str, pounces: bool) -> None:
        engine = Engine(_pouncing_hopper(["T"]))
        state = engine.make_state({
            "a1": ("K", 0), "a8": ("K", 1), "h1": ("Z", 2), "h3": (screen, 0), "h5": ("Q", 0),
        })
        _, phase = engine.apply_with_phase(state, engine.find_move(state, "Ka1-b1"))
        took = [(step.origin, step.target) for step in phase if step.kind == "pounce"]
        assert bool(took) is pounces
