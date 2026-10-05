"""Schema 15: `promotion.on_leaving` (Q452, traditional shogi's B1).

A promotion entry that says `on_leaving: true` promotes its piece on a move that *starts*
in the zone as well as on one that ends there, wherever it lands. That is real shogi's rule
and Fairy-Stockfish's for every piece in `promotedPieceType`; before it the DSL promoted on
arriving only (`docs/reference/traditional-shogi.md`, section 1). The compiler keeps the flag only where the
piece has a move out of its zone, the bridge sends the types as `promo_on_leaving`, and the
core asks the origin once per piece at both of the generator's promotion sites.

The inertness proof is `test_schema_nine.py::TestTheBumpMovedNothing`, which pins every
committed and archived ruleset's mechanics hash and tables fingerprint, and the perft
literals, which are unchanged. This file holds the feature. The pre-registered number is
shogi's perft(5) with the flag on every entry: 19,861,490, Fairy-Stockfish's, against
19,859,785 without it (Q450).

`rulesets/shogi.json` was not changed here: B3 (Q454) put the flag on all six of its
entries and re-froze its hash. Every test here reads an inline copy with the flag taken
off, and puts back what it needs.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec, pieces_that_can_leave_their_zone
from chess_variant_sandbox.engine.moves import (
    MOVE_PROMOTION,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, _schema_fifteen_uses
from chess_variant_sandbox.spec.transform import settle_schema_version

ROOT = Path(__file__).resolve().parents[1]
EVERY = frozenset({"P", "L", "N", "S", "B", "R"})
# The pieces of shogi with a move out of their zone. The pawn, lance and knight move
# forward only, and forward from the far three ranks stays in them or leaves the board.
LEAVING = frozenset({"S", "B", "R"})

# From the opening, with the stop at a terminal honoured. Depths 1 to 4 are shared: no
# piece can have left its zone before the fifth ply. Depth 5 is where they part, by 1,705.
SHARED = {1: 30, 2: 900, 3: 25_470}
SHARED_DEEP = {4: 719_731}
FLAGGED_FIVE = 19_861_490  # Fairy-Stockfish's shogi, pre-registered in Q450
UNFLAGGED_FIVE = 19_859_785  # promotion on arriving only: this engine's own, a frozen pin


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8"))


def _flagged(
    spec: dict[str, Any], pieces: frozenset[str] = EVERY, value: bool = True
) -> dict[str, Any]:
    """``spec`` stamped 15 with ``on_leaving: value`` on the promotion entries of ``pieces``."""
    out = copy.deepcopy(spec)
    out["schema_version"] = 15
    promotion = out["rules"]["promotion"]
    for entry in promotion if isinstance(promotion, list) else [promotion]:
        if entry["piece"] in pieces:
            entry["on_leaving"] = value
    return out


@pytest.fixture(scope="module")
def shogi() -> dict[str, Any]:
    """Shogi promoting on arriving only: the committed file with any flag taken off.

    Taken off rather than assumed absent: B3 (Q454) put the flag on `rulesets/shogi.json`,
    and this file compares the two games.
    """
    spec = _load("shogi")
    for entry in spec["rules"]["promotion"]:
        entry.pop("on_leaving", None)
    return spec


@pytest.fixture(scope="module")
def chess() -> dict[str, Any]:
    return _load("chess")


@pytest.fixture(scope="module")
def today(shogi: dict[str, Any]) -> Engine:
    return Engine(shogi)


@pytest.fixture(scope="module")
def leaving(shogi: dict[str, Any]) -> Engine:
    return Engine(_flagged(shogi))


def _words(engine: Engine, state: Any, origin: str, target: str) -> set[tuple[int, str]]:
    """The legal moves from ``origin`` to ``target``: (kind, what it becomes, or '')."""
    ids = engine.spec.piece_ids
    frm, to = engine._square_index(origin), engine._square_index(target)
    return {
        (
            move_type(move),
            ids[move_promotion(move)] if move_type(move) == MOVE_PROMOTION else "",
        )
        for move in engine.legal_moves(state)
        if move_from(move) == frm and move_to(move) == to
    }


PLAIN = (0, "")


# ------------------------------------------------------------------------ the version


class TestTheVersionIsBumped:
    def test_the_schema_version_is_fifteen(self) -> None:
        assert SCHEMA_VERSION >= 15

    def test_the_collector_names_the_feature(
        self, shogi: dict[str, Any], chess: dict[str, Any]
    ) -> None:
        assert _schema_fifteen_uses(shogi) == set()
        assert _schema_fifteen_uses(chess) == set()
        assert _schema_fifteen_uses(_flagged(shogi)) == {"rules.promotion.on_leaving"}
        assert _schema_fifteen_uses(_flagged(shogi, value=False)) == {
            "rules.promotion.on_leaving=false"
        }
        # The single-entry spelling of `rules.promotion` is read too.
        assert _schema_fifteen_uses(_flagged(chess, frozenset({"P"}))) == {
            "rules.promotion.on_leaving"
        }

    def test_an_older_stamp_is_refused_by_name(self, shogi: dict[str, Any]) -> None:
        spec = _flagged(shogi)
        assert validate_spec(spec) == []
        spec["schema_version"] = 14
        errors = validate_spec(spec)
        assert any(
            "rules.promotion.on_leaving" in e and "schema 15" in e for e in errors
        ), errors

    def test_false_written_out_is_a_schema_15_key_and_a_different_hash(
        self, shogi: dict[str, Any]
    ) -> None:
        spelled = _flagged(shogi, value=False)
        assert validate_spec(spelled) == []
        spelled["schema_version"] = 14
        assert any("schema 15" in e for e in validate_spec(spelled))
        assert spec_hash(_flagged(shogi, value=False)) != spec_hash(shogi)

    def test_the_flag_must_be_a_boolean(self, shogi: dict[str, Any]) -> None:
        spec = _flagged(shogi)
        spec["rules"]["promotion"][3]["on_leaving"] = "yes"
        assert any(
            "rules.promotion[3].on_leaving must be a boolean" in e for e in validate_spec(spec)
        ), validate_spec(spec)

    def test_the_settled_stamp_follows_the_flag(self, shogi: dict[str, Any]) -> None:
        spec = _flagged(shogi, LEAVING)
        spec["schema_version"] = 2
        settle_schema_version(spec)
        assert spec["schema_version"] == 15
        for entry in spec["rules"]["promotion"]:
            entry.pop("on_leaving", None)
        settle_schema_version(spec)
        assert spec["schema_version"] < 15


# ------------------------------------------------------------------------ the compiler


class TestTheCompilerKeepsTheFlagOnlyWhereItCanFire:
    def test_shogis_three_pieces_that_can_leave_keep_it(self, leaving: Engine) -> None:
        ids = leaving.spec.piece_ids
        kept = {ids[rule.piece_type] for rule in leaving.spec.promotions if rule.on_leaving}
        assert kept == LEAVING

    def test_the_cheap_question_is_the_compilers(
        self, shogi: dict[str, Any], chess: dict[str, Any]
    ) -> None:
        """`pieces_that_can_leave_their_zone` agrees with the compiled flag, flag or none."""
        assert pieces_that_can_leave_their_zone(shogi) == LEAVING
        assert pieces_that_can_leave_their_zone(chess) == frozenset()
        makruk = _load("makruk")
        assert pieces_that_can_leave_their_zone(makruk) == {"P"}
        compiled = compile_spec(_flagged(makruk))
        assert [rule.on_leaving for rule in compiled.promotions] == [True]

    def test_a_flag_that_cannot_fire_compiles_to_the_game_without_it(
        self, chess: dict[str, Any]
    ) -> None:
        """A chess pawn's zone is the rank it cannot move from.

        Only the Zobrist draw differs, because it is seeded by the mechanics hash and the
        key moves that hash.
        """
        flagged = compile_spec(_flagged(chess, frozenset({"P"})))
        assert [rule.on_leaving for rule in flagged.promotions] == [False]

        def game(tables: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in tables.items() if not k.startswith("zobrist")}

        assert game(canonical_tables(flagged)) == game(canonical_tables(compile_spec(chess)))
        engine = Engine(flagged)
        assert engine.perft(engine.initial_state(), 3) == 8_902

    def test_the_boundary_sends_the_types_and_nothing_for_a_game_without_them(
        self, shogi: dict[str, Any], leaving: Engine
    ) -> None:
        ids = leaving.spec.piece_ids
        sent = build_tables(leaving.spec)["promo_on_leaving"]
        assert sorted(ids[t] for t in sent) == sorted(LEAVING)
        assert "promo_on_leaving" in canonical_tables(leaving.spec)
        assert "promo_on_leaving" not in canonical_tables(compile_spec(shogi))

    def test_false_compiles_to_the_game_without_the_key(self, shogi: dict[str, Any]) -> None:
        declined = compile_spec(_flagged(shogi, value=False))
        assert not any(rule.on_leaving for rule in declined.promotions)
        assert "promo_on_leaving" not in canonical_tables(declined)


# ------------------------------------------------------------------------ the core


class TestAMoveOutOfTheZonePromotes:
    """A silver on e7 is inside its zone (the far three ranks) and d6 is outside it."""

    PLACED = {"e1": ("K", 0), "a9": ("K", 1), "e7": ("S", 0)}

    def test_without_the_flag_a_move_out_is_plain(self, today: Engine) -> None:
        state = today.make_state(self.PLACED, moved=["e7"])
        assert _words(today, state, "e7", "d6") == {PLAIN}
        assert _words(today, state, "e7", "d8") == {PLAIN, (MOVE_PROMOTION, "Z")}

    def test_with_the_flag_it_may_promote(self, leaving: Engine) -> None:
        state = leaving.make_state(self.PLACED, moved=["e7"])
        assert _words(leaving, state, "e7", "d6") == {PLAIN, (MOVE_PROMOTION, "Z")}
        assert _words(leaving, state, "e7", "f6") == {PLAIN, (MOVE_PROMOTION, "Z")}
        # A move that ends in the zone promotes as it always did.
        assert _words(leaving, state, "e7", "d8") == {PLAIN, (MOVE_PROMOTION, "Z")}

    def test_a_move_that_neither_starts_nor_ends_in_the_zone_is_unchanged(
        self, leaving: Engine
    ) -> None:
        state = leaving.make_state(
            {"e1": ("K", 0), "a9": ("K", 1), "e5": ("S", 0)}, moved=["e5"]
        )
        assert _words(leaving, state, "e5", "d4") == {PLAIN}
        assert _words(leaving, state, "e5", "d6") == {PLAIN}

    def test_the_second_players_zone_is_the_near_three_ranks(self, leaving: Engine) -> None:
        state = leaving.make_state(
            {"e9": ("K", 1), "a1": ("K", 0), "e3": ("S", 1)}, side_to_move=1, moved=["e3"]
        )
        assert _words(leaving, state, "e3", "d4") == {PLAIN, (MOVE_PROMOTION, "Z")}

    def test_a_slide_out_of_the_zone_promotes_on_every_square(self, leaving: Engine) -> None:
        state = leaving.make_state(
            {"e1": ("K", 0), "a9": ("K", 1), "h8": ("R", 0)}, moved=["h8"]
        )
        for target in ("h6", "h5", "h2"):
            assert _words(leaving, state, "h8", target) == {PLAIN, (MOVE_PROMOTION, "D")}

    def test_under_mandatory_every_move_out_must_promote(self, shogi: dict[str, Any]) -> None:
        spec = _flagged(shogi)
        for entry in spec["rules"]["promotion"]:
            entry["mandatory"] = True
        engine = Engine(spec)
        state = engine.make_state(self.PLACED, moved=["e7"])
        assert _words(engine, state, "e7", "d6") == {(MOVE_PROMOTION, "Z")}

    def test_a_hop_out_of_the_zone_promotes_on_an_empty_square_and_a_capture(
        self, shogi: dict[str, Any]
    ) -> None:
        """The generator's second promotion site: the hop's inline empty-square branch.

        The rook is made a grasshopper-like hopper that may land anywhere past its screen,
        so e5 is a quiet landing past the pawn on e6, and e3 a capture.
        """
        spec = _flagged(shogi)
        rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
        rook["moves"] = [
            {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
        ]
        engine = Engine(spec)
        placed = {
            "a1": ("K", 0), "i9": ("K", 1), "e7": ("R", 0), "e6": ("P", 0), "e3": ("G", 1),
        }
        state = engine.make_state(placed, moved=["e7"])
        assert _words(engine, state, "e7", "e5") == {PLAIN, (MOVE_PROMOTION, "D")}
        assert _words(engine, state, "e7", "e3") == {PLAIN, (MOVE_PROMOTION, "D")}
        unflagged = copy.deepcopy(spec)
        for entry in unflagged["rules"]["promotion"]:
            entry.pop("on_leaving")
        plain = Engine(unflagged)
        state = plain.make_state(placed, moved=["e7"])
        assert _words(plain, state, "e7", "e5") == {PLAIN}
        assert _words(plain, state, "e7", "e3") == {PLAIN}


# ------------------------------------------------------------------------ perft


@pytest.mark.parametrize(("depth", "expected"), sorted(SHARED.items()))
def test_the_flag_moves_nothing_before_a_piece_can_leave(
    today: Engine, leaving: Engine, depth: int, expected: int
) -> None:
    for engine in (today, leaving):
        assert engine.perft(engine.initial_state(), depth, stop_at_terminal=True) == expected


@pytest.mark.slow
@pytest.mark.parametrize(("depth", "expected"), sorted(SHARED_DEEP.items()))
def test_the_flag_moves_nothing_at_depth_four(
    today: Engine, leaving: Engine, depth: int, expected: int
) -> None:
    for engine in (today, leaving):
        assert engine.perft(engine.initial_state(), depth, stop_at_terminal=True) == expected


@pytest.mark.slow
def test_shogi_with_the_flag_gives_fairy_stockfishs_perft_five(leaving: Engine) -> None:
    """The pre-registered number (Q450): published shogi at depth 5."""
    assert leaving.perft(leaving.initial_state(), 5, stop_at_terminal=True) == FLAGGED_FIVE


@pytest.mark.slow
def test_shogi_without_the_flag_is_short_by_the_moves_out_of_the_zone(today: Engine) -> None:
    """The shogi B3 (Q454) replaced (`630828e31147`): 1,705 short of the published number."""
    assert today.perft(today.initial_state(), 5, stop_at_terminal=True) == UNFLAGGED_FIVE
    assert FLAGGED_FIVE - UNFLAGGED_FIVE == 1_705


# ------------------------------------------------------------------------ played


def test_a_game_promotes_on_the_way_out_and_keeps_its_key(leaving: Engine) -> None:
    """Random play reaches a promotion out of the zone, and the incremental key holds.

    Perft checks the generator; this checks `make` and the Zobrist key on the moves the
    flag adds, which `test_schema_two`'s twenty-ply walk would never reach.
    """
    spec = leaving.spec
    zones = {
        (player, rule.piece_type): rule.squares[player]
        for rule in spec.promotions
        for player in (0, 1)
    }
    rng = random.Random(3)
    out_of_zone = 0
    for _game in range(6):
        state = leaving.initial_state()
        for _ in range(300):
            moves = leaving.legal_moves(state)
            if leaving.terminal(state, moves) is not None:
                break
            move = rng.choice(moves)
            if move_type(move) == MOVE_PROMOTION:
                player = state.side_to_move
                piece = spec.type_of(state.board[move_from(move)])
                zone = zones[(player, piece)]
                if move_from(move) in zone and move_to(move) not in zone:
                    out_of_zone += 1
            state = leaving.apply(state, move)
            assert state.key == leaving._compute_hash(state)
    assert out_of_zone > 0


# ------------------------------------------------------------------------ surfaces


class TestEverySurfaceSaysIt:
    def test_the_json_schema_offers_the_key(self) -> None:
        schema = rule_spec_schema()
        promotion = schema["properties"]["rules"]["properties"]["promotion"]
        assert promotion["anyOf"][0]["properties"]["on_leaving"]["type"] == "boolean"

    def test_the_description_says_the_move_out_promotes(self, shogi: dict[str, Any]) -> None:
        said = describe_spec(_flagged(shogi, LEAVING))
        assert "S on a move that starts or ends in ranks 7, 8, 9 may become one of Z" in said
        assert "P reaching ranks 7, 8, 9 may become one of Q" in said
        # The rule's prose, not the committed description, which says it in its own words.
        assert "on a move that starts or ends in ranks" not in describe_spec(shogi)

    def test_the_rule_costs_nothing_on_top_of_promotion(self, shogi: dict[str, Any]) -> None:
        assert (
            measure_complexity(_flagged(shogi)).special_rules
            == measure_complexity(shogi).special_rules
        )

    def test_the_two_games_are_a_distance_apart(self, shogi: dict[str, Any]) -> None:
        flagged = _flagged(shogi, LEAVING)
        assert features(flagged)["promotes_on_leaving"] == 1.0
        assert features(shogi)["promotes_on_leaving"] == 0.0
        assert distance(flagged, shogi) > 0

    def test_the_preamble_teaches_it(self) -> None:
        assert "`on_leaving: true`" in SCHEMA_PREAMBLE

    def test_fairy_stockfish_reads_the_flag_off_the_compiled_entry(
        self, shogi: dict[str, Any]
    ) -> None:
        from chess_variant_sandbox.fairy import supports

        assert supports(shogi) is False
        assert supports(_flagged(shogi, LEAVING)) is True
