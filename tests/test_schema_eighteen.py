"""Schema 18, as 18a opens it (Q781): Einstein's keeping link and Omega's walled rank.

Schema 18 is one version built in several tasks (`docs/archive/plan-schema-17-18.md`
§9.3). 18a adds
two lifted restrictions, neither of which moves a table of any ruleset that does not use
it, and neither of which needs the core to change:

* the keeping link -- ``contacts.links[]`` with ``mover_to`` equal to ``mover``, accepted
  where the mover has a promotion of its own. The core writes the promotion, then the
  contact over it (Q539), so a queen that captures in Einstein chess stays a queen;
* the walled rank -- castling on a rank that holds a wall or an absent cell, as long as no
  castling crosses or lands on one (`schema.castling_spans`). Omega's corner cells.

18c (Q783) adds two nested keys, and the core changes for both:

* ``extinction.per_type`` -- each listed type counted on its own, so a side loses when any
  one of them runs out (Extinction chess). The side that made the last move is read
  first, Fairy-Stockfish's order;
* ``promotion.only_lost`` -- a target offered only while the mover has fewer of it on the
  board than it started with (Grand chess). Refused beside a hand that fills from
  captures, a circe strike and ``promote_on_entry``.

18d (Q784) adds one nested key, and the core changes for it:

* ``drops.placement`` -- while a side holds anything in hand, a drop is its only move
  (Fairy-Stockfish's ``mustDrop``), and a royal may start in the stock: then it is neither
  in check nor lost while it waits, and its own drop is the one move that can expose it.
  Placement chess and Sittuyin.

18e (Q785) adds one nested key, and the core changes for it:

* ``drops.either_face`` -- a piece in hand may be dropped as itself or as its other face,
  the piece whose ``promoted_from`` it is, spent from the same slot (Kyoto shogi). Refused
  beside a strike that places anything but ``advance``, ``switches_sides`` and
  ``explode_on_capture``, and wherever a face could not name its one slot.

18f (Q786) adds one new value, and the core changes for it:

* ``reach.on: "leave"`` -- the game is decided by the move that takes the named piece off
  a goal square (Jeson Mor). The core keeps one position field for it, the side that has
  just left (``goal_left_by``): set in ``make``, restored in ``unmake``, carried on the
  snapshot and across the boundary, out of the key, and read in reach's slot of the
  ending order. Refused beside ``safe``.

18b (Q782) adds two new values on ``symmetry``, and neither the compiler nor the core
changes for them:

* ``rotate6`` and ``dihedral12`` -- the hexagon's turns, accepted only on a hex lattice, so
  a hex step is written once and turned six ways (and mirrored). A spelling:
  `complexity.expand_symmetry` turns them in axial coordinates, and a spec written with
  them compiles to its long form's tables (`TestHexSymmetries`).

`TestTheBumpMovedNothing` is the inertness proof each task repeats in its own commit.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from retired_specs import REBUILT
from test_schema_nine import FROZEN

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_PROMOTION,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.measurement import tables_fingerprint
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.complexity import expand_symmetry, measure_complexity
from chess_variant_sandbox.spec.describe import describe_pieces, describe_spec
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.relate import identity_key
from chess_variant_sandbox.spec.schema import (
    SCHEMA_VERSION,
    _schema_eighteen_uses,
    castling_spans,
)
from chess_variant_sandbox.spec.transform import set_obstacle, settle_schema_version

ROOT = Path(__file__).resolve().parents[1]
SHOWCASES = (
    "einstein", "omega", "extinction-chess", "grand", "placement-chess", "sittuyin", "kyoto",
    "jeson-mor", "hex-glinski-turned", "hex-glinski-turned-mirrored", "pinwheel",
)
# Q870: the two turned showcases left the library for `pinwheel`, which writes a one-handed
# knight `rotate6` beside `dihedral12` officers; the tests read them rebuilt
# (`tests/retired_specs.py`).
RETIRED_SHOWCASES = frozenset({"hex-glinski-turned", "hex-glinski-turned-mirrored"})
# Games that use a schema 18 key through a value a later bump added, so are no showcase of
# schema 18: `"per_type": "own"` (schema 27) is a value on 18c's `per_type`.
LATER_USERS = frozenset({"riders-and-leapers"})  # Q936s, `extinction-any-kind.md` §5
KEEPING = "rules.contacts.mover_to=mover(promotion)"
WALLED = "rules.castling(walled rank)"
PER_TYPE = "rules.extinction.per_type"
ONLY_LOST = "rules.promotion.only_lost"
PLACEMENT = "rules.drops.placement"
EITHER_FACE = "rules.drops.either_face"
LEAVE = "rules.reach.on=leave"
ROTATE6 = "symmetry=rotate6"
DIHEDRAL12 = "symmetry=dihedral12"
# What each showcase uses of schema 18. Sittuyin's pawn promotes only into a lost general,
# so it uses 18c's flag beside its own; `placement-chess` is 18d's one-feature showcase.
FEATURE = {
    "einstein": {KEEPING},
    "omega": {WALLED},
    "extinction-chess": {PER_TYPE},
    "grand": {ONLY_LOST},
    "placement-chess": {PLACEMENT},
    "sittuyin": {PLACEMENT, ONLY_LOST},
    "kyoto": {EITHER_FACE},
    "jeson-mor": {LEAVE},
    "hex-glinski-turned": {ROTATE6},
    "hex-glinski-turned-mirrored": {DIHEDRAL12},
    "pinwheel": {ROTATE6, DIHEDRAL12},
}


def _load(name: str) -> dict[str, Any]:
    if name in REBUILT:  # retired by Q868 or Q870; rebuilt from a game in the library
        return REBUILT[name]()
    folder = "rulesets" if name == "chess" else "archive"
    return json.loads((ROOT / folder / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def games() -> dict[str, dict[str, Any]]:
    return {name: _load(name) for name in (*SHOWCASES, "chess")}


def _square(name: str, width: int) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _occupant(engine: Engine, state: Any, square: str) -> tuple[int, str] | None:
    value = state.board[_square(square, engine.spec.width)]
    if value == 0:
        return None
    return engine.spec.owner_of(value), engine.spec.piece_ids[engine.spec.type_of(value)]


def _moves_between(engine: Engine, state: Any, origin: str, target: str) -> list[int]:
    width = engine.spec.width
    wanted = (_square(origin, width), _square(target, width))
    return [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == wanted]


def _without_keeping(einstein: dict[str, Any]) -> dict[str, Any]:
    """Q533's audit spelling: the same game with the queen's keeping links taken out."""
    spec = copy.deepcopy(einstein)
    links = spec["rules"]["contacts"]["links"]
    spec["rules"]["contacts"]["links"] = [link for link in links if link["mover"] != "Q"]
    settle_schema_version(spec)
    return spec


def _einstein_position(einstein: dict[str, Any]) -> dict[str, Any]:
    """White queen d1 against a black pawn on d5, both kings: one capture to read."""
    spec = copy.deepcopy(einstein)
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "Q", "squares": ["d1"]},
            {"piece": "K", "player": 1, "squares": ["h8"]},
            {"piece": "P", "player": 1, "squares": ["d5"]},
        ],
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _omega_cleared(omega: dict[str, Any]) -> dict[str, Any]:
    """Omega with the knights, bishops and queen off the board, so both rooks are free."""
    spec = copy.deepcopy(omega)
    spec["setup"]["placements"] = [
        placement
        for placement in spec["setup"]["placements"]
        if placement["piece"] not in ("N", "B", "Q")
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestTheVersionIsBumped:
    def test_the_schema_version_is_eighteen(self) -> None:
        assert SCHEMA_VERSION >= 18

    def test_the_collector_names_each_showcase_once(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        assert _schema_eighteen_uses(games["chess"]) == set()
        for name, feature in FEATURE.items():
            assert _schema_eighteen_uses(games[name]) == feature, name

    @pytest.mark.parametrize("name", sorted(FEATURE))
    def test_an_older_stamp_is_refused_by_name(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        spec = copy.deepcopy(games[name])
        spec["schema_version"] = 17
        errors = validate_spec(spec)
        assert (
            f'the spec uses {", ".join(sorted(FEATURE[name]))}, added in schema 18; declare '
            '"schema_version": 18 to use it'
        ) in errors, errors

    @pytest.mark.parametrize("name", SHOWCASES)
    def test_the_showcase_settles_to_eighteen(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        spec = copy.deepcopy(games[name])
        spec["schema_version"] = 1
        assert settle_schema_version(spec)["schema_version"] == 18


class TestTheBumpMovedNothing:
    """Every committed and archived spec hashes and compiles as `FROZEN` pinned it.

    `FROZEN` is `tests/test_schema_nine.py`'s table, which every bump since schema 9 has
    extended and none has re-frozen except on purpose; asking it again here puts the proof
    in the commit that raises the version, as `docs/archive/plan-schema-17-18.md` §9.2 asks
    of each 18 task. Only the two showcases use anything schema 18 added.
    """

    @pytest.mark.parametrize("path", sorted(FROZEN), ids=lambda p: Path(p).stem)
    def test_every_ruleset_is_the_game_it_was(self, path: str) -> None:
        spec = load_spec(ROOT / path)
        digest, tables = FROZEN[path]
        assert spec_hash(spec) == digest, f"{path} is not the file that was frozen"
        assert tables_fingerprint(spec) == tables, f"{path} compiles to different tables"

    def test_chess_is_still_chess(self, games: dict[str, dict[str, Any]]) -> None:
        assert spec_hash(games["chess"]) == "4c1922ffae26"

    def test_only_the_showcases_use_schema_eighteen(self) -> None:
        users = {
            Path(path).stem for path in FROZEN if _schema_eighteen_uses(load_spec(ROOT / path))
        }
        assert users == (set(SHOWCASES) - RETIRED_SHOWCASES) | LATER_USERS


class TestTheKeepingLink:
    def test_a_self_link_with_no_promotion_still_changes_nothing(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = copy.deepcopy(games["einstein"])
        spec["rules"]["promotion"] = [
            entry for entry in spec["rules"]["promotion"] if entry["piece"] != "Q"
        ]
        errors = validate_spec(spec)
        assert any("points 'Q' at itself, which changes nothing" in e for e in errors), errors
        assert _schema_eighteen_uses(spec) == set()

    def test_an_own_link_is_not_lifted(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["einstein"])
        spec["rules"]["contacts"]["links"].append(
            {"mover": "Q", "onto": "R", "own": True, "mover_to": "Q", "onto_to": "B"}
        )
        errors = validate_spec(spec)
        assert any("points 'Q' at itself" in e for e in errors), errors

    def test_the_compiler_writes_the_mover_over_its_promotion(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        compiled = compile_spec(games["einstein"])
        strike = compiled.strike
        assert strike is not None
        n = compiled.n_types
        queen = compiled.piece_ids.index("Q")
        for onto in ("P", "N", "B", "R", "Q"):
            assert strike.mover_to[queen * n + compiled.piece_ids.index(onto)] == queen

    def test_a_queen_that_captures_stays_a_queen(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(_einstein_position(games["einstein"]))
        state = engine.initial_state()
        (take,) = _moves_between(engine, state, "d1", "d5")
        assert _occupant(engine, engine.apply(state, take), "d5") == (0, "Q")
        assert engine.describe_move(state, take) == "Qd1xd5/Q->Q"
        # A quiet move still demotes: the link is about the capture alone.
        (quiet,) = _moves_between(engine, state, "d1", "d2")
        assert _occupant(engine, engine.apply(state, quiet), "d2") == (0, "R")

    def test_the_audit_spelling_leaves_a_rook(self, games: dict[str, dict[str, Any]]) -> None:
        engine = Engine(_einstein_position(_without_keeping(games["einstein"])))
        state = engine.initial_state()
        (take,) = _moves_between(engine, state, "d1", "d5")
        assert _occupant(engine, engine.apply(state, take), "d5") == (0, "R")

    def test_perft_matches_the_audit_spelling_until_a_queen_captures(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        kept, plain = Engine(games["einstein"]), Engine(_without_keeping(games["einstein"]))
        for depth in (1, 2, 3):
            assert kept.perft(kept.initial_state(), depth) == plain.perft(
                plain.initial_state(), depth
            )

    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["einstein"])
        rng = random.Random(18)
        state = engine.initial_state()
        for _ in range(160):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            taking = [m for m in moves if state.board[move_to(m)]]
            move = rng.choice(taking) if taking and rng.random() < 0.5 else rng.choice(moves)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state), "hash drifted"


class TestTheWalledRank:
    def test_the_spans_stop_short_of_the_walls(self, games: dict[str, dict[str, Any]]) -> None:
        omega = games["omega"]
        spans = castling_spans(omega["rules"], omega["setup"], 12, 12, "mirror")
        files = {"".join(chr(ord("a") + f) for f in sorted(fs)) for _, fs in spans}
        assert files == {"cdefg", "ghij"}
        assert {rank for rank, _ in spans} == {1, 10}

    def test_a_wall_between_king_and_rook_is_still_refused(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        for version in (17, 18):
            spec = _omega_cleared(games["omega"])
            spec["schema_version"] = version
            spec["rules"]["obstacles"]["squares"].append("d2")
            errors = validate_spec(spec)
            assert any(
                "rules.castling happens along rank 2, which holds an obstacle at d2" in e
                for e in errors
            ), errors

    def test_a_wall_on_a_destination_is_refused(self) -> None:
        """File castling lands outside the king and rook: king d1 to a1, rook c1 to b1."""
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        spec["schema_version"] = 18
        spec["rules"]["castling"] = {
            "king": "K",
            "rook": "R",
            "king_file": {"queenside": "a", "kingside": "g"},
            "rook_file": {"queenside": "b", "kingside": "f"},
        }
        spec["setup"]["placements"] = [
            {"piece": "P", "squares": [f"{f}2" for f in "abcdefgh"]},
            {"piece": "R", "squares": ["c1", "h1"]},
            {"piece": "K", "squares": ["d1"]},
        ]
        assert castling_spans(spec["rules"], spec["setup"], 8, 8, "mirror")[0] == (
            0, frozenset({0, 1, 2, 3}),
        )
        set_obstacle(spec, "a1")
        errors = validate_spec(spec)
        assert any("obstacle at a1 between the king" in e for e in errors), errors

    def test_both_sides_castle_both_ways_past_the_walls(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(_omega_cleared(games["omega"]))
        state = engine.initial_state()
        castles = sorted(
            engine.describe_move(state, m)
            for m in engine.legal_moves(state)
            if move_type(m) == MOVE_CASTLE
        )
        assert len(castles) == 2, castles
        (kingside,) = [
            m for m in engine.legal_moves(state)
            if move_type(m) == MOVE_CASTLE and move_to(m) == _square("i2", 12)
        ]
        after = engine.apply(state, kingside)
        assert _occupant(engine, after, "i2") == (0, "K")
        assert _occupant(engine, after, "h2") == (0, "R")
        assert _occupant(engine, after, "j2") is None
        (queenside,) = [
            m for m in engine.legal_moves(state)
            if move_type(m) == MOVE_CASTLE and move_to(m) == _square("e2", 12)
        ]
        after = engine.apply(state, queenside)
        assert _occupant(engine, after, "e2") == (0, "K")
        assert _occupant(engine, after, "f2") == (0, "R")
        assert _occupant(engine, after, "c2") is None
        # Black, after a quiet white pawn move.
        (step,) = _moves_between(engine, state, "b3", "b4")
        black = engine.apply(state, step)
        targets = sorted(
            move_to(m) for m in engine.legal_moves(black) if move_type(m) == MOVE_CASTLE
        )
        assert targets == [_square("e11", 12), _square("i11", 12)]

    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(_omega_cleared(games["omega"]))
        rng = random.Random(18)
        state = engine.initial_state()
        castled = False
        for _ in range(120):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            castles = [m for m in moves if move_type(m) == MOVE_CASTLE]
            move = castles[0] if castles else rng.choice(moves)
            castled |= bool(castles)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state), "hash drifted"
        assert castled


class TestTheOtherSurfaces:
    def test_the_description_says_the_queen_stays(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        text = describe_spec(games["einstein"])
        assert "when queen takes rook, the queen stays as it is" in text
        assert "decided after any promotion that move would make" in text

    def test_the_keeping_link_has_a_distance(self, games: dict[str, dict[str, Any]]) -> None:
        assert distance(games["einstein"], _without_keeping(games["einstein"])) > 0.0

    @pytest.mark.parametrize(
        ("name", "key"),
        # Omega's walls cross since Q703 (`*` in the FEN); its 12x12 board does not.
        [("einstein", "rules.contacts"), ("omega", "the board is 12x12")],
    )
    def test_fairy_refuses_it_by_name(
        self, games: dict[str, dict[str, Any]], name: str, key: str
    ) -> None:
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(games[name]))
        assert any(key in reason for reason in refusal.value.reasons)

    def test_a_wall_beyond_the_rooks_stamps_eighteen(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        """`transform.set_obstacle`, which `wall-off-a-square` calls, raises the stamp."""
        spec = copy.deepcopy(games["omega"])
        spec["rules"].pop("obstacles")
        spec["schema_version"] = 6
        assert validate_spec(spec) == [], validate_spec(spec)
        set_obstacle(spec, "a2")
        assert spec["schema_version"] == 18
        assert _schema_eighteen_uses(spec) == {WALLED}


# --------------------------------------------------------------------------- 18c (Q783)


def _go(engine: Engine, state: Any, text: str) -> Any:
    """Play the one legal move whose text starts with ``text``."""
    (move,) = [
        m for m in engine.legal_moves(state) if engine.describe_move(state, m).startswith(text)
    ]
    return engine.apply(state, move)


def _summed(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game with the listed types counted together again."""
    spec = copy.deepcopy(spec)
    spec["rules"]["extinction"].pop("per_type")
    settle_schema_version(spec)
    return spec


def _double_extinction(extinction: dict[str, Any]) -> dict[str, Any]:
    """Black's last pawn takes White's last knight and promotes: both run out at once."""
    spec = copy.deepcopy(extinction)
    spec["rules"].pop("castling")
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "R", "squares": ["a1"]}, {"piece": "B", "squares": ["c1", "f1"]},
        {"piece": "Q", "squares": ["d1"]}, {"piece": "K", "squares": ["e1"]},
        {"piece": "N", "squares": ["g1"]}, {"piece": "P", "squares": ["a2", "b3"]},
        {"piece": "R", "player": 1, "squares": ["h8"]},
        {"piece": "N", "player": 1, "squares": ["b8", "g8"]},
        {"piece": "B", "player": 1, "squares": ["c8", "f8"]},
        {"piece": "Q", "player": 1, "squares": ["d8"]},
        {"piece": "K", "player": 1, "squares": ["e8"]},
        {"piece": "P", "player": 1, "squares": ["h3"]},
    ]}
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestExtinctionPerType:
    def test_a_spelling_that_states_nothing_is_refused(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        for pieces, said in (("any", "'any' lists no types"), (["Q"], "with one type listed")):
            spec = copy.deepcopy(games["extinction-chess"])
            spec["rules"]["extinction"]["pieces"] = pieces
            errors = validate_spec(spec)
            assert any(said in e for e in errors), errors

    def test_a_side_that_starts_run_out_is_refused(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = copy.deepcopy(games["extinction-chess"])
        spec["rules"]["extinction"]["min_count"] = 2
        errors = validate_spec(spec)
        assert any("starts with 1 of 'Q', fewer than min_count 2" in e for e in errors), errors

    def test_false_written_out_is_named_and_is_the_summed_game(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = _summed(games["extinction-chess"])
        spec["rules"]["extinction"]["per_type"] = False
        settle_schema_version(spec)
        assert _schema_eighteen_uses(spec) == {"rules.extinction.per_type=false"}
        # The hashes differ (the key is written), so the Zobrist draws do; nothing else.
        def without_draws(game: dict[str, Any]) -> dict[str, Any]:
            tables = canonical_tables(compile_spec(game))
            return {k: v for k, v in tables.items() if not k.startswith("zobrist")}

        assert without_draws(spec) == without_draws(_summed(games["extinction-chess"]))

    def test_running_out_of_one_kind_ends_it(self, games: dict[str, dict[str, Any]]) -> None:
        """Seed 0 of the pre-registered probe: Black's queen goes at ply 25."""
        apart = games["extinction-chess"]
        for spec, ends in ((apart, True), (_summed(apart), False)):
            engine = Engine(spec)
            rng = random.Random(0)
            state = engine.initial_state()
            for _ in range(25):
                state = engine.apply(state, rng.choice(engine.legal_moves(state)))
            outcome = engine.terminal(state, engine.legal_moves(state))
            if ends:
                assert outcome is not None and outcome.reason == "extinction", outcome
                assert outcome.winner == 0
            else:
                assert outcome is None, outcome

    def test_the_side_that_moved_is_read_first(self, games: dict[str, dict[str, Any]]) -> None:
        """Fairy-Stockfish's order, checked against pyffish 0.0.90 when it was written."""
        engine = Engine(_double_extinction(games["extinction-chess"]))
        state = engine.initial_state()
        for text in ("Pb3-b4", "Ph3-h2", "Pb4-b5", "Ph2xg1=Q"):
            state = _go(engine, state, text)
        outcome = engine.terminal(state, engine.legal_moves(state))
        assert outcome is not None and outcome.reason == "extinction"
        assert outcome.winner == 0, "Black moved, ran out of pawns as White ran out of knights"

    def test_perft_matches_the_summed_spelling_until_a_kind_runs_out(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        apart = Engine(games["extinction-chess"])
        together = Engine(_summed(games["extinction-chess"]))
        for depth in (1, 2, 3):
            assert apart.perft(apart.initial_state(), depth) == together.perft(
                together.initial_state(), depth
            )

    def test_a_dropped_type_takes_the_flag_with_it(self) -> None:
        from chess_variant_sandbox.spec.transform import drop_piece

        spec = load_spec(ROOT / "rulesets" / "chess.json")
        spec["schema_version"] = 18
        spec["rules"]["extinction"] = {"pieces": ["Q", "N"], "per_type": True}
        assert validate_spec(spec) == [], validate_spec(spec)
        drop_piece(spec, "N")
        assert spec["rules"]["extinction"] == {"pieces": ["Q"]}


def _grand_position(grand: dict[str, Any], **promotion: Any) -> dict[str, Any]:
    """A white pawn on e9 one step from the last rank, its side's queen and rook standing.

    Black's queen can take White's on j1, which is the loss that frees a target.
    """
    spec = copy.deepcopy(grand)
    spec["rules"].pop("en_passant")
    spec["rules"]["promotion"]["to"] = ["Q", "R"]
    spec["rules"]["promotion"].pop("only_lost")
    spec["rules"]["promotion"].update(promotion)
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "K", "squares": ["c1"]}, {"piece": "Q", "squares": ["j1"]},
        {"piece": "R", "squares": ["b1"]}, {"piece": "P", "squares": ["e9"]},
        {"piece": "K", "player": 1, "squares": ["h10"]},
        {"piece": "Q", "player": 1, "squares": ["j10"]},
        {"piece": "R", "player": 1, "squares": ["i10"]},
    ]}
    settle_schema_version(spec)
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _from(engine: Engine, state: Any, square: str) -> list[str]:
    origin = _square(square, engine.spec.width)
    return sorted(
        engine.describe_move(state, m) for m in engine.legal_moves(state) if move_from(m) == origin
    )


class TestPromotionOnlyLost:
    @pytest.mark.parametrize(
        ("edit", "said"),
        [
            ({"drops": {"from_captures": True}}, "beside rules.drops"),
            (
                {"strike": {"effect": "capture", "position": "home", "immune": ["K"]}},
                'position is "home"',
            ),
            ({"promote_on_entry": True}, "beside rules.promote_on_entry"),
        ],
    )
    def test_a_rule_that_gives_a_piece_back_is_refused(
        self, games: dict[str, dict[str, Any]], edit: dict[str, Any], said: str
    ) -> None:
        spec = copy.deepcopy(games["grand"])
        spec["rules"].update(edit)
        errors = validate_spec(spec)
        assert any(said in e for e in errors), errors

    def test_a_stone_only_hand_is_accepted(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["grand"])
        spec["pieces"].append({"id": "S", "name": "stone", "value": 1, "moves": []})
        spec["rules"]["drops"] = {"from_captures": False, "stock": {"S": "unlimited"}}
        assert not any("only_lost" in e for e in validate_spec(spec)), validate_spec(spec)

    def test_a_target_a_side_never_had_is_refused(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = copy.deepcopy(games["grand"])
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "A"
        ]
        errors = validate_spec(spec)
        assert any("offers 'A' to player 0 only once one is lost" in e for e in errors), errors

    def test_the_compiler_counts_the_setup(self, games: dict[str, dict[str, Any]]) -> None:
        compiled = compile_spec(games["grand"])
        (rule,) = compiled.promotions
        assert rule.only_lost
        index = compiled.piece_ids.index
        for player in (0, 1):
            counted = [rule.opening[player][index(p)] for p in "QCARBNP"]
            assert counted == [1, 1, 1, 2, 2, 2, 10]

    def test_a_target_waits_for_a_loss(self, games: dict[str, dict[str, Any]]) -> None:
        engine = Engine(_grand_position(games["grand"], only_lost=True))
        state = engine.initial_state()
        assert _from(engine, state, "e9") == []  # the last rank needs a promotion
        state = _go(engine, _go(engine, state, "Kc1-c2"), "Qj10xj1")
        assert _from(engine, state, "e9") == ["Pe9-e10=Q"]

    def test_the_spelling_without_it_offers_both(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(_grand_position(games["grand"]))
        assert _from(engine, engine.initial_state(), "e9") == ["Pe9-e10=Q", "Pe9-e10=R"]

    def test_an_optional_zone_keeps_the_plain_move(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        spec = _grand_position(games["grand"], only_lost=True)
        spec["setup"]["placements"][3]["squares"] = ["e8"]
        engine = Engine(spec)
        assert _from(engine, engine.initial_state(), "e8") == ["Pe8-e9"]

    def test_a_hop_into_the_zone_is_filtered_too(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        """The generator's second promotion site: a hop landing on an empty square."""
        spec = _grand_position(games["grand"], only_lost=True)
        pawn = next(p for p in spec["pieces"] if p["id"] == "P")
        pawn["moves"] = [{"kind": "hop", "vectors": [[0, 1]], "mode": "move"}]
        spec["rules"].pop("forbid_immobility")
        spec["setup"]["placements"][3]["squares"] = ["e7"]
        spec["setup"]["placements"].append({"piece": "N", "squares": ["e8"]})
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = engine.initial_state()
        pawn_square = _square("e7", 10)
        hops = [m for m in engine.legal_moves(state) if move_from(m) == pawn_square]
        assert hops, "the pawn should hop its screen"
        assert all(move_type(m) != MOVE_PROMOTION for m in hops)
        state = _go(engine, _go(engine, state, "Kc1-c2"), "Qj10xj1")
        offered = {
            move_promotion(m)
            for m in engine.legal_moves(state)
            if move_from(m) == pawn_square and move_type(m) == MOVE_PROMOTION
        }
        assert offered == {engine.spec.piece_ids.index("Q")}

    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["grand"])
        rng = random.Random(18)
        state = engine.initial_state()
        for _ in range(200):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), "hash drifted"


class TestTheCountsElsewhere:
    def test_the_descriptions_say_so(self, games: dict[str, dict[str, Any]]) -> None:
        text = describe_spec(games["extinction-chess"])
        assert "each counted on its own" in text
        assert "it is the side that moved that has it" in text
        assert "only into a piece its side has lost" in describe_spec(games["grand"])

    def test_each_has_a_distance(self, games: dict[str, dict[str, Any]]) -> None:
        assert distance(games["extinction-chess"], _summed(games["extinction-chess"])) > 0.0
        plain = copy.deepcopy(games["grand"])
        plain["rules"]["promotion"].pop("only_lost")
        assert distance(games["grand"], plain) > 0.0

    def test_fairy_counts_each_kind_as_we_do(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["extinction-chess"])
        spec["rules"].pop("castling")  # a non-royal king's castling is refused there
        config = translate(compile_spec(spec)).config
        assert "extinctionPieceTypes = BKNPQR" in config
        assert "extinctionPieceCount = 0" in config
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(_summed(spec)))
        assert any("counts its listed types together" in r for r in refusal.value.reasons)

    def test_fairy_limits_the_promotion_by_the_setup(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        config = translate(compile_spec(games["grand"])).config
        assert "promotionLimit = n:2 b:2 r:2 q:1 a:1 c:1" in config
        assert "doubleStepRankMin = 3" in config
        assert "doubleStepRank = 3" in config
        uneven = copy.deepcopy(games["grand"])
        uneven["setup"]["mirror"] = False
        mirrored = [
            {**p, "player": 1, "squares": [f"{sq[0]}{11 - int(sq[1:])}" for sq in p["squares"]]}
            for p in uneven["setup"]["placements"]
        ]
        mirrored[0]["squares"] = mirrored[0]["squares"][:1]  # Black starts with one rook
        uneven["setup"]["placements"] += mirrored
        assert validate_spec(uneven) == [], validate_spec(uneven)
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(uneven))
        assert any("promotionLimit is one count" in r for r in refusal.value.reasons)


# --------------------------------------------------------------------------- 18d (Q784)


def _hand_only(spec: dict[str, Any]) -> dict[str, Any]:
    """Chess's officers but the king in hand, the king placed: the phase with no royal held."""
    spec = copy.deepcopy(spec)
    spec["setup"] = {"mirror": True, "placements": [
        {"piece": "P", "squares": [f"{file}2" for file in "abcdefgh"]},
        {"piece": "K", "squares": ["e1"]},
    ]}
    spec["rules"].pop("castling", None)
    rank = [f"{file}1" for file in "abcdefgh"]
    spec["rules"]["drops"] = {
        "from_captures": False,
        "placement": True,
        "stock": {"Q": 1, "R": 2, "B": 2, "N": 2},
        "region": {piece: rank for piece in ("Q", "R", "B", "N")},
    }
    spec["schema_version"] = 18
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _king_into_a_rook(placement: dict[str, Any]) -> dict[str, Any]:
    """Both kings in hand; a black rook on h1 sweeps the first rank up to a white knight on d1.

    White's only moves are its king's drops, and only a1, b1 and c1 are out of the rook's
    line: the three drops the filter keeps.
    """
    spec = copy.deepcopy(placement)
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "N", "player": 0, "squares": ["d1"]},
        {"piece": "P", "player": 0, "squares": ["a2"]},
        {"piece": "R", "player": 1, "squares": ["h1"]},
        {"piece": "P", "player": 1, "squares": ["a7"]},
    ]}
    spec["rules"]["drops"]["stock"] = {"K": 1}
    spec["rules"]["drops"]["region"] = {"K": [f"{file}1" for file in "abcdefgh"]}
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestThePlacementPhase:
    def test_while_a_piece_is_held_a_drop_is_the_only_move(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["placement-chess"])
        state = engine.initial_state()
        rng = random.Random(784)
        for ply in range(16):
            moves = engine.legal_moves(state)
            assert moves and all(move_type(m) == MOVE_DROP for m in moves), ply
            state = engine.apply(state, rng.choice(moves))
        assert state.hands == ((0,) * 6, (0,) * 6)
        assert not any(move_type(m) == MOVE_DROP for m in engine.legal_moves(state))

    def test_perft_counts_the_drops(self, games: dict[str, dict[str, Any]]) -> None:
        """Five kinds on eight squares, then each side's second drop onto seven."""
        engine = Engine(games["placement-chess"])
        state = engine.initial_state()
        assert [engine.perft(state, depth) for depth in (1, 2, 3)] == [40, 1_600, 51_520]
        sittuyin = Engine(games["sittuyin"])
        # Twenty free squares on the three back ranks for four kinds, eight for the chariot.
        assert [sittuyin.perft(sittuyin.initial_state(), d) for d in (1, 2)] == [88, 7_744]

    def test_a_royal_in_hand_is_neither_lost_nor_in_check(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["placement-chess"])
        state = engine.initial_state()
        assert engine.terminal(state, engine.legal_moves(state)) is None
        assert not engine.in_check(state, 0) and not engine.in_check(state, 1)

    def test_the_royal_may_not_drop_into_check(self, games: dict[str, dict[str, Any]]) -> None:
        engine = Engine(_king_into_a_rook(games["placement-chess"]))
        state = engine.initial_state()
        landed = sorted(engine.describe_move(state, m) for m in engine.legal_moves(state))
        assert landed == ["K@a1", "K@b1", "K@c1"], landed

    def test_the_phase_off_lets_a_held_piece_wait(self, games: dict[str, dict[str, Any]]) -> None:
        held = _hand_only(games["chess"])
        waiting = copy.deepcopy(held)
        waiting["rules"]["drops"].pop("placement")
        settle_schema_version(waiting)
        assert _schema_eighteen_uses(waiting) == set()
        on, off = Engine(held), Engine(waiting)
        moves_on = on.legal_moves(on.initial_state())
        moves_off = off.legal_moves(off.initial_state())
        assert all(move_type(m) == MOVE_DROP for m in moves_on)
        assert set(moves_on) < set(moves_off), "the phase only takes moves away"
        assert any(move_type(m) != MOVE_DROP for m in moves_off)

    def test_false_written_out_is_named_and_is_the_game_without_it(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        waiting = _hand_only(games["chess"])
        waiting["rules"]["drops"].pop("placement")
        settle_schema_version(waiting)
        written = copy.deepcopy(waiting)
        written["rules"]["drops"]["placement"] = False
        settle_schema_version(written)
        assert _schema_eighteen_uses(written) == {"rules.drops.placement=false"}

        def without_draws(game: dict[str, Any]) -> dict[str, Any]:
            tables = canonical_tables(compile_spec(game))
            return {k: v for k, v in tables.items() if not k.startswith("zobrist")}

        assert without_draws(written) == without_draws(waiting)

    def test_a_royal_in_hand_needs_the_phase(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["placement-chess"])
        spec["rules"]["drops"].pop("placement")
        settle_schema_version(spec)
        errors = validate_spec(spec)
        assert any("holds the royal 'K' outside the placement phase" in e for e in errors)
        assert "no royal piece appears in setup.placements" in errors

    def test_a_royal_is_stocked_once_and_not_also_placed(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        twice = copy.deepcopy(games["placement-chess"])
        twice["rules"]["drops"]["stock"]["K"] = 2
        assert any("holds 2 of the royal 'K'" in e for e in validate_spec(twice))
        # Schema 10 accepted this, and the core then guarded one king of two.
        both = load_spec(ROOT / "rulesets" / "chess.json")
        both["schema_version"] = 10
        both["rules"]["drops"] = {"from_captures": False, "stock": {"K": 1}}
        errors = validate_spec(both)
        assert any("both placed by setup.placements and stocked" in e for e in errors), errors

    def test_only_lost_counts_the_stock_under_the_phase(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        compiled = compile_spec(games["sittuyin"])
        (rule,) = compiled.promotions
        general = compiled.piece_ids.index("F")
        assert rule.opening[0][general] == rule.opening[1][general] == 1
        engine = Engine(games["sittuyin"])
        state = engine.initial_state()
        rng = random.Random(4)
        for _ in range(400):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            standing = {compiled.piece_at(state.board, sq) for sq in range(compiled.n_squares)}
            if (state.side_to_move, general) in standing:
                assert not any(move_type(m) == MOVE_PROMOTION for m in moves)
            state = engine.apply(state, rng.choice(moves))

    @pytest.mark.parametrize("name", ["placement-chess", "sittuyin"])
    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        engine = Engine(games[name])
        rng = random.Random(18)
        state = engine.initial_state()
        for _ in range(200):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), "hash drifted"


class TestThePlacementPhaseElsewhere:
    def test_the_description_says_so(self, games: dict[str, dict[str, Any]]) -> None:
        text = describe_spec(games["placement-chess"])
        assert "playing one onto the board is their only move" in text
        assert "a royal still in hand is neither in check nor lost" in text

    def test_it_has_a_distance(self, games: dict[str, dict[str, Any]]) -> None:
        held = _hand_only(games["chess"])
        waiting = copy.deepcopy(held)
        waiting["rules"]["drops"].pop("placement")
        assert distance(held, waiting) > 0.0

    def test_fairy_must_drop_from_the_stocked_pocket(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        config = translate(compile_spec(games["placement-chess"])).config
        assert "mustDrop = true" in config
        assert "[NNBBRRQKnnbbrrqk]" in config
        assert "whiteDropRegion = a1 b1 c1 d1 e1 f1 g1 h1" in config
        assert "sittuyinRookDrop" not in config

    def test_fairy_writes_sittuyins_rook_and_refuses_its_diagonal(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(games["sittuyin"]))
        assert any("not whole ranks" in r for r in refusal.value.reasons)
        twin = copy.deepcopy(games["sittuyin"])
        twin["rules"].pop("promotion")
        config = translate(compile_spec(twin)).config
        assert "sittuyinRookDrop = true" in config
        assert "mustDrop = true" in config

    def test_an_unlimited_stock_is_still_refused_there(self) -> None:
        stones = load_spec(ROOT / "archive" / "stones.json")
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(stones))
        assert any('"unlimited"' in r for r in refusal.value.reasons)


# --------------------------------------------------------------------------- 18e (Q785)


def _with_a_pawn_in_hand(kyoto: dict[str, Any]) -> dict[str, Any]:
    """Kyoto with one pawn in each hand from the start: a hand to read on the first ply."""
    spec = copy.deepcopy(kyoto)
    spec["rules"]["drops"]["stock"] = {"P": 1}
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _one_face(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game with the flag taken off: a held piece comes back as itself only."""
    spec = copy.deepcopy(spec)
    spec["rules"]["drops"].pop("either_face")
    settle_schema_version(spec)
    return spec


def _drop_texts(engine: Engine, state: Any) -> list[str]:
    return sorted(
        engine.describe_move(state, m)
        for m in engine.legal_moves(state)
        if move_type(m) == MOVE_DROP
    )


def _refusals(spec: dict[str, Any]) -> list[str]:
    return [e for e in validate_spec(spec) if "either_face" in e]


class TestDropEitherFace:
    def test_a_held_piece_is_offered_both_faces(self, games: dict[str, dict[str, Any]]) -> None:
        engine = Engine(_with_a_pawn_in_hand(games["kyoto"]))
        state = engine.initial_state()
        drops = _drop_texts(engine, state)
        pawns = [text[2:] for text in drops if text.startswith("P@")]
        rooks = [text[2:] for text in drops if text.startswith("R@")]
        assert len(pawns) == 15 and pawns == rooks, drops
        assert len(drops) == 30

    def test_without_the_flag_it_comes_back_as_itself(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(_one_face(_with_a_pawn_in_hand(games["kyoto"])))
        state = engine.initial_state()
        assert all(text.startswith("P@") for text in _drop_texts(engine, state))
        assert engine.perft(state, 1) == 27
        assert Engine(_with_a_pawn_in_hand(games["kyoto"])).perft(state, 1) == 42

    def test_the_other_face_is_spent_from_the_base(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(_with_a_pawn_in_hand(games["kyoto"]))
        state = engine.initial_state()
        pawn, rook = engine.spec.piece_ids.index("P"), engine.spec.piece_ids.index("R")
        assert state.hands[0][pawn] == 1
        (move,) = [
            m
            for m in engine.legal_moves(state)
            if move_type(m) == MOVE_DROP and engine.describe_move(state, m) == "R@c3"
        ]
        assert move_promotion(move) == rook  # the move word carries the face
        after = engine.apply(state, move)
        assert after.hands[0][pawn] == 0 and after.hands[0][rook] == 0
        assert _occupant(engine, after, "c3") == (0, "R")
        # unmake gives back the slot that was spent
        assert engine.perft(state, 2) == sum(
            engine.perft(engine.apply(state, m), 1) for m in engine.legal_moves(state)
        )

    def test_perft_is_the_one_face_spelling_until_a_drop(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        both, one = Engine(games["kyoto"]), Engine(_one_face(games["kyoto"]))
        # No capture can come before the third ply, so nothing is held until the fifth.
        assert [both.perft(both.initial_state(), d) for d in (1, 2, 3, 4)] == [
            12, 137, 1_636, 18_268,
        ]
        assert [one.perft(one.initial_state(), d) for d in (1, 2, 3, 4)] == [
            12, 137, 1_636, 18_268,
        ]

    def test_a_hand_never_holds_a_turned_face(self, games: dict[str, dict[str, Any]]) -> None:
        engine = Engine(games["kyoto"])
        faces = {engine.spec.piece_ids.index(f) for f in "RBGT"}
        rng = random.Random(785)
        state = engine.initial_state()
        held_any = False
        for _ in range(300):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            assert not any(state.hands[p][f] for p in (0, 1) for f in faces)
            held_any = held_any or any(any(hand) for hand in state.hands)
            state = engine.apply(state, rng.choice(moves))
        assert held_any

    def test_false_written_out_is_named_and_is_the_game_without_it(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        without = _one_face(games["kyoto"])
        written = copy.deepcopy(without)
        written["rules"]["drops"]["either_face"] = False
        settle_schema_version(written)
        assert _schema_eighteen_uses(written) == {"rules.drops.either_face=false"}

        def without_draws(game: dict[str, Any]) -> dict[str, Any]:
            tables = canonical_tables(compile_spec(game))
            return {k: v for k, v in tables.items() if not k.startswith("zobrist")}

        assert without_draws(written) == without_draws(without)
        assert compile_spec(without).drops.faces == ()  # type: ignore[union-attr]

    def test_the_compiler_pairs_each_base_with_its_face(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        compiled = compile_spec(games["kyoto"])
        drops = compiled.drops
        assert drops is not None
        ids = compiled.piece_ids
        pairs = {ids[slot]: ids[face] for slot, face in enumerate(drops.faces) if face >= 0}
        assert pairs == {"P": "R", "S": "B", "N": "G", "L": "T"}
        assert drops.hand_slot(ids.index("R")) == ids.index("P")

    @pytest.mark.parametrize(
        ("rule", "says"),
        [
            ({"strike": {"effect": "capture", "position": "stay"}}, "position is 'stay'"),
            ({"strike": {"effect": "capture", "position": "home"}}, "position is 'home'"),
            ({"strike": {"switches_sides": ["S"]}}, "rules.strike.switches_sides"),
            ({"explode_on_capture": {"radius": 1}}, "rules.explode_on_capture"),
        ],
    )
    def test_a_rule_that_already_says_what_a_taken_piece_becomes_is_refused(
        self, games: dict[str, dict[str, Any]], rule: dict[str, Any], says: str
    ) -> None:
        spec = copy.deepcopy(games["kyoto"])
        spec["rules"].update(copy.deepcopy(rule))
        assert any(says in e for e in _refusals(spec)), validate_spec(spec)

    def test_a_flag_with_no_pair_is_refused(self) -> None:
        crazyhouse = load_spec(ROOT / "rulesets" / "crazyhouse.json")
        crazyhouse["schema_version"] = 18
        crazyhouse["rules"]["drops"]["either_face"] = True
        assert any("pairs nothing" in e for e in _refusals(crazyhouse))
        stones = load_spec(ROOT / "archive" / "kyoto.json")
        stones["rules"]["drops"] = {
            "from_captures": False, "stock": {"K": 1}, "either_face": True,
        }
        assert _refusals(stones)

    def test_a_face_must_name_one_slot(self, games: dict[str, dict[str, Any]]) -> None:
        two = copy.deepcopy(games["kyoto"])
        for piece in two["pieces"]:
            if piece["id"] == "T":
                piece["promoted_from"] = "N"  # the knight now has the gold and the tokin
        assert any("is the promoted_from of 'G', 'T'" in e for e in _refusals(two))
        chain = copy.deepcopy(games["kyoto"])
        for piece in chain["pieces"]:
            if piece["id"] == "P":
                piece["promoted_from"] = "S"  # the pawn is a face and a base
        assert any("is both a face" in e for e in _refusals(chain))
        stocked = copy.deepcopy(games["kyoto"])
        stocked["rules"]["drops"]["stock"] = {"R": 1}
        assert any("holds 'R', the other face" in e for e in _refusals(stocked))
        royal = copy.deepcopy(games["kyoto"])
        for piece in royal["pieces"]:
            if piece["id"] == "T":
                piece["promoted_from"] = "K"
        assert any("pairs the royal piece 'K'" in e for e in _refusals(royal))

    def test_a_dropped_pair_takes_the_flag_with_it(self) -> None:
        from chess_variant_sandbox.spec.transform import drop_piece

        spec = load_spec(ROOT / "rulesets" / "shogi.json")
        spec["schema_version"] = 18
        spec["rules"]["drops"]["either_face"] = True
        for piece_id in ("Q", "X", "Y", "Z", "H", "D"):
            drop_piece(spec, piece_id)
        assert "either_face" not in spec["rules"]["drops"]

    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["kyoto"])
        rng = random.Random(18)
        state = engine.initial_state()
        for _ in range(300):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), "hash drifted"


class TestDropEitherFaceElsewhere:
    def test_the_description_says_so(self, games: dict[str, dict[str, Any]]) -> None:
        text = describe_spec(games["kyoto"])
        assert "may be played as itself or as its other face" in text

    def test_it_has_a_distance(self, games: dict[str, dict[str, Any]]) -> None:
        assert distance(games["kyoto"], _one_face(games["kyoto"])) > 0.0

    def test_fairy_drops_the_promoted_face_and_turns_back_by_demotion(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        from chess_variant_sandbox.fairy.position import uci_of

        variant = translate(compile_spec(games["kyoto"]))
        config = variant.config
        assert "dropPromoted = true" in config
        assert "pieceDemotion = true" in config
        assert "promotedPieceType = P:R S:B N:G L:T" in config
        engine = Engine(_with_a_pawn_in_hand(games["kyoto"]))
        state = engine.initial_state()
        spelled = {
            uci_of(engine.spec, state, m, variant.chars, variant.promotion_suffix)
            for m in engine.legal_moves(state)
        }
        assert {"+P@c3", "P@c3", "a1a2-", "b1b2+"} <= spelled

    def test_fairy_writes_the_audit_spelling_as_it_did(self) -> None:
        """No `promoted_from`, no flip: Q533's both-ways map is written as before."""
        audit = copy.deepcopy(_load("kyoto"))
        for piece in audit["pieces"]:
            piece.pop("promoted_from", None)
        audit["rules"]["drops"].pop("either_face")
        settle_schema_version(audit)
        config = translate(compile_spec(audit)).config
        assert "pieceDemotion" not in config and "dropPromoted" not in config
        assert "promotedPieceType = P:R R:P S:B B:S G:N N:G T:L L:T" in config

    def test_fairy_refuses_shogi_with_it_for_its_pawn_rules(self) -> None:
        shogi = load_spec(ROOT / "rulesets" / "shogi.json")
        shogi["schema_version"] = 18
        shogi["rules"]["drops"]["either_face"] = True
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(shogi))
        assert any("no_doubled or drops.no_mate" in r for r in refusal.value.reasons)

    def test_the_page_is_sent_the_pairs(self, games: dict[str, dict[str, Any]]) -> None:
        from chess_variant_sandbox.web.play import drop_faces_payload

        compiled = compile_spec(games["kyoto"])
        ids = compiled.piece_ids
        faces = drop_faces_payload(compiled.drops)
        assert faces is not None and faces[ids.index("P")] == ids.index("R")
        shogi = compile_spec(load_spec(ROOT / "rulesets" / "shogi.json"))
        assert drop_faces_payload(shogi.drops) is None
        board = (ROOT / "chess_variant_sandbox" / "web" / "static" / "page" / "board.js")
        assert "SESSION.ruleset.dropFaces" in board.read_text(encoding="utf-8")


def _arriving(spec: dict[str, Any]) -> dict[str, Any]:
    """Q533's audit spelling: the same game won on arrival (``on`` left out)."""
    twin = copy.deepcopy(spec)
    twin["rules"]["reach"].pop("on")
    settle_schema_version(twin)
    return twin


def _play(engine: Engine, state: Any, origin: str, target: str) -> Any:
    (move,) = _moves_between(engine, state, origin, target)
    return engine.apply(state, move)


def _rifle_on_a_goal() -> dict[str, Any]:
    """Rifle chess (a capture keeps the capturer on its origin), won by leaving d4.

    White's knight starts on d4, the goal, with a black pawn on e6 it can shoot.
    """
    spec = _load("rifle")
    spec["schema_version"] = 18
    spec["setup"] = {
        "mirror": False,
        "placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "N", "squares": ["d4"]},
            {"piece": "K", "player": 1, "squares": ["h8"]},
            {"piece": "P", "player": 1, "squares": ["e6", "a7"]},
        ],
    }
    del spec["rules"]["castling"]
    spec["rules"]["reach"] = {"piece": "N", "squares": ["d4"], "mirror": False, "on": "leave"}
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestWinByLeaving:
    def test_standing_on_the_goal_decides_nothing(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        leaving, arriving = Engine(games["jeson-mor"]), Engine(_arriving(games["jeson-mor"]))
        here, there = leaving.initial_state(), arriving.initial_state()
        for origin, target in [("c1", "d3"), ("c9", "d7"), ("d3", "e5")]:
            here = _play(leaving, here, origin, target)
            there = _play(arriving, there, origin, target)
        # The audit spelling ends on arrival; the leaving game goes on.
        ended = arriving.terminal(there)
        assert ended is not None and (ended.winner, ended.reason) == (0, "reached")
        assert leaving.terminal(here) is None
        assert here.goal_left_by == -1
        # Black moves elsewhere, and the knight steps off: White wins on that move.
        here = _play(leaving, here, "a9", "b7")
        assert leaving.terminal(here) is None
        won = _play(leaving, here, "e5", "f7")
        outcome = leaving.terminal(won)
        assert outcome is not None and (outcome.winner, outcome.reason) == (0, "reached")
        assert won.goal_left_by == 0

    def test_the_side_that_left_wins_whoever_moves_next(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["jeson-mor"])
        initial = engine.initial_state()
        for side in (0, 1):
            outcome = engine.terminal(initial.with_(goal_left_by=side))
            assert outcome is not None and (outcome.winner, outcome.reason) == (side, "reached")
        assert engine.terminal(initial) is None

    def test_a_capture_that_keeps_its_origin_does_not_leave(self) -> None:
        engine = Engine(_rifle_on_a_goal())
        state = engine.initial_state()
        shot = _play(engine, state, "d4", "e6")
        assert _occupant(engine, shot, "d4") == (0, "N")
        assert _occupant(engine, shot, "e6") is None
        assert shot.goal_left_by == -1 and engine.terminal(shot) is None
        stepped = _play(engine, state, "d4", "c2")
        outcome = engine.terminal(stepped)
        assert outcome is not None and (outcome.winner, outcome.reason) == (0, "reached")

    def test_perft_is_the_arrival_spelling_and_the_endings_differ(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        leaving, arriving = Engine(games["jeson-mor"]), Engine(_arriving(games["jeson-mor"]))
        here, there = leaving.initial_state(), arriving.initial_state()
        # Plain perft walks through terminal positions, so the two trees are one tree, and
        # walking it makes and unmakes the field at every move off e5.
        plain = [leaving.perft(here, d) for d in (1, 2, 3, 4)]
        assert plain == [arriving.perft(there, d) for d in (1, 2, 3, 4)]
        # Stopping at the endings: a knight on e5 at ply 3 ends the twin and not this
        # game, which no move can end before ply 5 (two moves there, one off).
        assert leaving.perft(here, 4, stop_at_terminal=True) == plain[3]
        assert arriving.perft(there, 4, stop_at_terminal=True) < plain[3]

    @pytest.mark.slow
    def test_the_first_move_off_the_goal_ends_the_tree(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["jeson-mor"])
        state = engine.initial_state()
        # Ply 5 is the first that can leave e5; the tree stops below it, not at it.
        assert engine.perft(state, 5, stop_at_terminal=True) == engine.perft(state, 5)
        assert engine.perft(state, 6, stop_at_terminal=True) < engine.perft(state, 6)

    def test_on_arrive_written_out_is_named_and_is_the_arrival_game(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        without = _arriving(games["jeson-mor"])
        written = copy.deepcopy(without)
        written["rules"]["reach"]["on"] = "arrive"
        settle_schema_version(written)
        assert _schema_eighteen_uses(written) == {"rules.reach.on=arrive"}

        def without_draws(game: dict[str, Any]) -> dict[str, Any]:
            tables = canonical_tables(compile_spec(game))
            return {k: v for k, v in tables.items() if not k.startswith("zobrist")}

        assert without_draws(written) == without_draws(without)
        assert "reach_leaves" not in canonical_tables(compile_spec(without))

    def test_safe_and_other_values_are_refused(self, games: dict[str, dict[str, Any]]) -> None:
        spec = copy.deepcopy(games["jeson-mor"])
        spec["rules"]["reach"]["safe"] = True
        assert any("leave safe out" in error for error in validate_spec(spec))
        spec = copy.deepcopy(games["jeson-mor"])
        spec["rules"]["reach"]["on"] = "pass"
        assert any("rules.reach.on must be one of" in e for e in validate_spec(spec))

    def test_the_goal_distance_reads_reach_then_step_off(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        leaving = compile_spec(games["jeson-mor"])
        arriving = compile_spec(_arriving(games["jeson-mor"]))
        e5 = _square("e5", 9)
        for player in (0, 1):
            here = leaving.goal_distance[player][0]
            there = arriving.goal_distance[player][0]
            assert there[e5] == 0 and here[e5] == 1
            # A knight can step off e5, so every square is one move further than arriving.
            assert list(here) == [d + 1 for d in there]
        assert leaving.goal_diameter == arriving.goal_diameter + 1

    def test_a_goal_the_piece_cannot_leave_is_no_goal(self) -> None:
        # A piece that only steps forward, aimed at the last rank's h8: the first player's
        # gets there and can never step off, the second player's walks away from it.
        spec = _rifle_on_a_goal()
        spec["pieces"].append({
            "id": "S", "name": "slug", "value": 1,
            "moves": [{"kind": "leap", "vectors": [[0, 1]], "symmetry": "none", "mode": "both"}],
        })
        spec["rules"]["reach"] = {"piece": "S", "squares": ["h8"], "mirror": False, "on": "leave"}
        spec["setup"]["placements"].append({"piece": "S", "squares": ["h2"]})
        assert validate_spec(spec) == [], validate_spec(spec)
        compiled = compile_spec(spec)
        slug = compiled.piece_ids.index("S")
        assert set(compiled.goal_distance[0][slug]) == {255}
        assert compiled.goal_distance[1][slug][_square("h8", 8)] == 1

    def test_a_short_game_keeps_its_incremental_hash(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        engine = Engine(games["jeson-mor"])
        rng = random.Random(18)
        state = engine.initial_state()
        for _ in range(300):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), "hash drifted"
        assert state.goal_left_by in (0, 1)


class TestWinByLeavingElsewhere:
    def test_the_description_says_so(self, games: dict[str, dict[str, Any]]) -> None:
        text = describe_spec(games["jeson-mor"])
        assert "N moving off e5 is a win" in text
        assert "then a move off a winning square, then" in text

    def test_it_has_a_distance(self, games: dict[str, dict[str, Any]]) -> None:
        assert distance(games["jeson-mor"], _arriving(games["jeson-mor"])) > 0.0

    def test_fairy_refuses_it_by_name(self, games: dict[str, dict[str, Any]]) -> None:
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(games["jeson-mor"]))
        assert any('rules.reach.on "leave"' in r for r in refusal.value.reasons)

    def test_the_page_is_told(self) -> None:
        board = ROOT / "chess_variant_sandbox" / "web" / "static" / "page" / "board.js"
        assert "SESSION.ruleset.goalLeaves" in board.read_text(encoding="utf-8")
        play = ROOT / "chess_variant_sandbox" / "web" / "play.py"
        assert '"goalLeaves"' in play.read_text(encoding="utf-8")


# 18b (Q782): the two showcases and the committed game each is written from.
TURNED = ("hex-glinski-turned", "hex-glinski-turned-mirrored")
LONG_FORM = "hex-glinski-promoting"
ZOBRIST = frozenset(
    {"zobrist_pieces", "zobrist_side", "zobrist_ep", "zobrist_colour", "zobrist_turn"}
)
HEX_SIDES = {(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)}
HEX_DIAGONALS = {(1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)}


def _written_out(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with every descriptor's vectors listed under ``none``: its long form."""
    long = copy.deepcopy(spec)
    for piece in long["pieces"]:
        for move in piece.get("moves", []):
            symmetry = move.pop("symmetry", "none")
            listed: set[tuple[int, int]] = set()
            for vector in move["vectors"]:
                listed |= expand_symmetry((vector[0], vector[1]), symmetry)
            move["vectors"] = [list(v) for v in sorted(listed)]
            move["symmetry"] = "none"
    return long


def _with_rook_move(symmetry: str, vectors: list[list[int]], lattice: str = "hex") -> dict[str, Any]:
    """The long form (or chess, on a square board) with the rook's move one descriptor."""
    spec = _load(LONG_FORM if lattice == "hex" else "chess")
    spec["schema_version"] = 18
    rook = next(piece for piece in spec["pieces"] if piece["id"] == "R")
    rook["moves"] = [{"kind": "slide", "vectors": vectors, "symmetry": symmetry}]
    return spec


def _choosing_strike(symmetry: str) -> dict[str, Any]:
    return {
        "effect": "demote", "chain": [{"from": "Q", "to": "R"}], "immune": ["K"],
        "position": "choose", "landing": {"vectors": [[1, 0]], "symmetry": symmetry},
    }


class TestHexSymmetries:
    """`rotate6` and `dihedral12` are a hex spelling, and nothing else.

    `docs/archive/plan-schema-17-18.md` §9.1 (hex symmetries): no core change, the turn is in `expand_symmetry`, and
    a showcase written with them plays its long form's perft. The verdict is *shown by
    identity* (`docs/reference/schema-showcase-coverage.md`, *Schema 18* note 2).
    """

    def test_rotate6_turns_a_neighbour_round_the_six(self) -> None:
        assert expand_symmetry((1, 0), "rotate6") == HEX_SIDES
        assert expand_symmetry((1, 1), "rotate6") == HEX_DIAGONALS
        assert expand_symmetry((1, 0), "dihedral12") == HEX_SIDES

    def test_dihedral12_is_the_same_twelve_whichever_mirror_generates_it(self) -> None:
        """The plan: no axis has to be chosen. The designer's `_hex_mirror` is another one."""
        from chess_variant_sandbox.web.designing import _hex_mirror

        for vector in [(1, 2), (2, 1), (3, -1), (1, 0), (2, 2), (4, -1)]:
            turned = expand_symmetry(vector, "rotate6")
            other = turned | expand_symmetry(_hex_mirror(vector), "rotate6")
            assert expand_symmetry(vector, "dihedral12") == other, vector
        knight = expand_symmetry((1, 2), "dihedral12")
        assert len(knight) == 12
        assert knight == expand_symmetry((1, 2), "rotate6") | expand_symmetry((2, 1), "rotate6")

    @pytest.mark.parametrize("symmetry", ["rotate6", "dihedral12"])
    def test_both_are_closed_under_the_second_players_half_turn(self, symmetry: str) -> None:
        """Why the hex `rotate` orientation needs nothing new (§9.1)."""
        for vector in [(1, 2), (3, -1), (2, 2), (0, 1)]:
            turned = expand_symmetry(vector, symmetry)
            assert {(-x, -y) for x, y in turned} == turned

    @pytest.mark.parametrize("name", TURNED)
    def test_the_showcase_is_its_long_form_written_shorter(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        """Written out, the showcase is `hex-glinski-promoting` move for move."""
        mine, theirs = _written_out(games[name]), _written_out(_load(LONG_FORM))
        for key in ("schema_version", "name", "description", "provenance"):
            mine.pop(key, None)
            theirs.pop(key, None)
        assert mine == theirs

    @pytest.mark.parametrize("name", TURNED)
    def test_its_tables_are_the_long_forms_byte_for_byte(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        """Every boundary array is equal but the five Zobrist draws, which the mechanics
        hash seeds, as `hex-glinski-hexagon`'s are (`tests/test_schema_sixteen.py`). The
        same holds against the showcase's own long form, which has a hash of its own."""
        mine = canonical_tables(compile_spec(games[name]))
        for long_form in (_load(LONG_FORM), _written_out(games[name])):
            theirs = canonical_tables(compile_spec(long_form))
            assert mine.keys() == theirs.keys()
            assert {key for key in mine if mine[key] != theirs[key]} == ZOBRIST
            same = {key: value for key, value in mine.items() if key not in ZOBRIST}
            other = {key: value for key, value in theirs.items() if key not in ZOBRIST}
            # A dump of every array, bytes and all, compared as text.
            assert repr(sorted(same.items())) == repr(sorted(other.items()))

    @pytest.mark.parametrize("name", TURNED)
    def test_its_perft_is_the_long_forms_to_depth_three(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        turned, long_form = Engine(games[name]), Engine(_load(LONG_FORM))
        for depth in (1, 2, 3):
            assert turned.perft(turned.initial_state(), depth) == long_form.perft(
                long_form.initial_state(), depth
            ), depth

    @pytest.mark.parametrize("name", TURNED)
    def test_its_rule_mass_is_the_long_forms_and_its_text_shorter(
        self, games: dict[str, dict[str, Any]], name: str
    ) -> None:
        """The plan's block predicted a lower rule mass; it is equal, by design.
        `rule_mass` counts descriptors and *expanded* directions, concepts rather than
        spelling, so no hex game's rule mass moves. What is shorter is the spec."""
        mine, theirs = measure_complexity(games[name]), measure_complexity(_load(LONG_FORM))
        assert mine.rule_mass == theirs.rule_mass
        assert mine.distinct_directions == theirs.distinct_directions
        assert mine.spec_bytes < theirs.spec_bytes

    @pytest.mark.parametrize("name", TURNED)
    def test_it_reads_as_its_long_form(self, games: dict[str, dict[str, Any]], name: str) -> None:
        """The page's sentences and `relate`'s normal form see one game."""
        mine = [piece.moves for piece in describe_pieces(games[name])]
        theirs = [piece.moves for piece in describe_pieces(_load(LONG_FORM))]
        assert mine == theirs
        assert identity_key(games[name]) == identity_key(_load(LONG_FORM))

    @pytest.mark.parametrize("symmetry", ["rotate6", "dihedral12"])
    def test_a_hex_turn_is_refused_on_a_square_board(self, symmetry: str) -> None:
        errors = validate_spec(_with_rook_move(symmetry, [[1, 0]], lattice="square"))
        assert any(
            f"symmetry {symmetry!r} turns vectors by the hexagon's 60-degree turns" in e
            for e in errors
        ), errors

    @pytest.mark.parametrize("symmetry", ["lateral", "rotate4", "dihedral8"])
    def test_a_square_turn_is_still_refused_on_a_hex_board(self, symmetry: str) -> None:
        errors = validate_spec(_with_rook_move(symmetry, [[1, 0]]))
        assert any(f"symmetry {symmetry!r} expands vectors by the square's" in e for e in errors)

    def test_an_older_stamp_names_the_turn(self) -> None:
        spec = _with_rook_move("rotate6", [[1, 0]])
        assert validate_spec(spec) == []
        spec["schema_version"] = 17
        assert any(
            "the spec uses symmetry=rotate6, added in schema 18" in e for e in validate_spec(spec)
        )

    def test_a_turn_past_the_component_bound_is_refused(self) -> None:
        """`[4, 4]` turned once is `[-4, 8]`, inside the bound; `[5, 5]` is `[-5, 10]`, which
        its long form could not write, so the shorthand may not say it either."""
        assert validate_spec(_with_rook_move("rotate6", [[4, 4]])) == []
        errors = validate_spec(_with_rook_move("dihedral12", [[5, 5]]))
        assert any("turns [5, 5] past the vector components' +/-8" in e for e in errors), errors

    def test_a_landing_takes_its_lattices_turns(self, games: dict[str, dict[str, Any]]) -> None:
        """A landing is expanded as a move is, so it is refused off its lattice as a move is."""
        square = _load("chess")
        square["schema_version"] = 18
        square["rules"]["strike"] = _choosing_strike("rotate6")
        errors = validate_spec(square)
        assert any("rules.strike.landing.symmetry 'rotate6' turns" in e for e in errors), errors
        hexy = copy.deepcopy(games["hex-glinski-turned"])
        hexy["rules"]["strike"] = _choosing_strike("rotate6")
        assert validate_spec(hexy) == []
        hexy["rules"]["strike"] = _choosing_strike("rotate4")
        assert any(
            "rules.strike.landing.symmetry 'rotate4' expands" in e for e in validate_spec(hexy)
        )

    def test_the_collector_names_a_landing_alone(self) -> None:
        spec = _load(LONG_FORM)
        spec["rules"]["strike"] = _choosing_strike("dihedral12")
        assert _schema_eighteen_uses(spec) == {DIHEDRAL12}


class TestHexSymmetriesElsewhere:
    def test_the_description_says_so(self, games: dict[str, dict[str, Any]]) -> None:
        assert "in all six directions" in describe_spec(games["hex-glinski-turned"])

    def test_a_turn_that_adds_directions_has_a_distance(self) -> None:
        """Through `rule_mass`'s expanded directions, as the square's turns do. A turn whose
        directions another piece already has reads as distance zero, on either lattice:
        `distance` counts the union of directions, not each piece's (`chess` with the rook
        cut to one direction is zero from it turned `rotate4` too)."""
        turned = _with_rook_move("rotate6", [[2, 0]])
        assert distance(turned, _with_rook_move("none", [[2, 0]])) > 0.0

    def test_fairy_refuses_the_hex_board_before_the_turn(
        self, games: dict[str, dict[str, Any]]
    ) -> None:
        with pytest.raises(Untranslatable):
            translate(compile_spec(games["hex-glinski-turned"]))

    def test_the_designer_offers_each_lattice_its_own_turns(self) -> None:
        from chess_variant_sandbox.web.designing import _candidate_moves

        offered = {move.get("symmetry", "none") for move in _candidate_moves([(1, 0)], "hex")}
        assert offered == {"none", "rotate6", "dihedral12"}
        square = {move.get("symmetry", "none") for move in _candidate_moves([(1, 0)], "square")}
        assert square == {"none", "lateral", "rotate4", "dihedral8"}
