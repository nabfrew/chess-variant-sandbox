"""Schema 27 opens (Q936): ``"per_type": "own"``, each listed kind counted against its own side.

`docs/reference/extinction-any-kind.md` §2 and §3: the value and its refusals, the compiled
per-side list (`compile._side_types`, which the spy reuses, `docs/plan-spy.md` §2.4), the
core's count (`Game::short_of`) in alternating play, mover first, and in rounds, where both
sides running out at once is a draw. The bump is inert: the per-side list crosses the
boundary empty for every spelling but ``"own"``, so no stored ruleset's tables move.

The game is the record's §1 reproduction: two armies of different ids on a 6x6 board, player
0 two wazirs ``A`` and two ferzes ``B``, player 1 two knights ``C`` and two dabbabas ``E``.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import _side_types, compile_spec
from chess_variant_sandbox.engine.state import GameState, Outcome
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, opening_counts
from tests.feature_inventory import _schema_twenty_seven_uses

ROOT = Path(__file__).resolve().parents[1]
STORED = sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))
OWN = "rules.extinction.per_type.own"
SHOWCASE = "riders-and-leapers"
"""`"own"`'s showcase (Q936s, `docs/reference/extinction-any-kind.md` §5)."""


def _leap(vector: list[int], symmetry: str) -> list[dict[str, Any]]:
    return [{"kind": "leap", "vectors": [vector], "symmetry": symmetry, "mode": "both"}]


def gap(extinction: dict[str, Any] | None = None, **rules: Any) -> dict[str, Any]:
    """The record's §1 spec at schema 27, under ``"own"`` unless told otherwise."""
    return {
        "schema_version": 27, "name": "gap", "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "A", "value": 3, "moves": _leap([1, 0], "rotate4")},
            {"id": "B", "value": 3, "moves": _leap([1, 1], "rotate4")},
            {"id": "C", "value": 3, "moves": _leap([1, 2], "dihedral8")},
            {"id": "E", "value": 3, "moves": _leap([2, 0], "rotate4")},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "A", "player": 0, "squares": ["b1", "e1"]},
            {"piece": "B", "player": 0, "squares": ["c1", "d1"]},
            {"piece": "C", "player": 1, "squares": ["b6", "e6"]},
            {"piece": "E", "player": 1, "squares": ["c6", "d6"]},
        ]},
        "rules": {
            "win": "none", "stalemate": "draw", "max_plies": 200, "repetition_draw": 3,
            "extinction": extinction or {"pieces": ["A", "B", "C", "E"], "per_type": "own"},
            **rules,
        },
    }


def _errors_about_own(spec: dict[str, Any]) -> list[str]:
    return [e for e in validate_spec(spec) if "per_type" in e]


# ------------------------------------------------------------------- the version gate


class TestTheGate:
    def test_the_version_is_twenty_seven(self) -> None:
        assert SCHEMA_VERSION >= 27

    def test_the_collector_names_own_and_nothing_else(self) -> None:
        assert _schema_twenty_seven_uses(gap()) == {OWN}
        for spelling in (True, False):
            spec = gap({"pieces": ["A", "B", "C", "E"], "per_type": spelling})
            assert OWN not in _schema_twenty_seven_uses(spec)
        assert _schema_twenty_seven_uses(gap({"pieces": ["A", "B", "C", "E"]})) == set()

    def test_only_the_showcase_says_own(self) -> None:
        """Inert when it landed: nothing committed or archived had used a value that did not
        exist. Since Q936s one archived game says it, the showcase, and nothing else does."""
        users = {path.stem for path in STORED if OWN in _schema_twenty_seven_uses(load_spec(path))}
        assert users == {SHOWCASE}


# ---------------------------------------------------------------------- validation


class TestValidation:
    def test_own_over_two_armies_of_different_kinds_validates(self) -> None:
        assert validate_spec(gap()) == []

    def test_true_over_the_same_list_is_still_refused(self) -> None:
        """The record's first row: `true` counts every kind against both sides."""
        errors = validate_spec(gap({"pieces": ["A", "B", "C", "E"], "per_type": True}))
        assert len(errors) == 4

    def test_a_value_that_is_neither_is_refused(self) -> None:
        errors = validate_spec(gap({"pieces": ["A", "B", "C", "E"], "per_type": "mine"}))
        assert errors == ['rules.extinction.per_type must be true, false or "own"']

    def test_own_over_any_is_refused(self) -> None:
        errors = _errors_about_own(gap({"pieces": "any", "per_type": "own"}))
        assert errors and "'any' lists no types" in errors[0]

    def test_own_over_one_type_is_refused(self) -> None:
        errors = _errors_about_own(gap({"pieces": ["A"], "per_type": "own"}))
        assert errors and "with one type listed" in errors[0]

    def test_a_kind_neither_side_starts_with_is_refused(self) -> None:
        spec = gap({"pieces": ["A", "B", "C", "E", "F"], "per_type": "own"})
        spec["pieces"].append({"id": "F", "value": 3, "moves": _leap([1, 0], "rotate4")})
        assert _errors_about_own(spec) == [
            'rules.extinction.per_type "own": neither player starts with \'F\', so it is '
            "counted against nobody; drop it from pieces"
        ]

    def test_a_kind_a_side_starts_short_of_is_refused(self) -> None:
        errors = _errors_about_own(gap({"pieces": ["A", "B", "C", "E"], "per_type": "own", "min_count": 3}))
        assert len(errors) == 4
        assert all("fewer than min_count 3" in e for e in errors)

    def test_min_count_two_is_what_each_side_starts_with(self) -> None:
        assert validate_spec(gap({"pieces": ["A", "B", "C", "E"], "per_type": "own", "min_count": 2})) == []

    def test_every_kind_started_by_both_sides_says_true(self) -> None:
        spec = gap({"pieces": ["A", "B"], "per_type": "own"})
        spec["setup"] = {"placements": [
            {"piece": "A", "squares": ["b1", "e1"]}, {"piece": "B", "squares": ["c1", "d1"]},
        ]}
        assert _errors_about_own(spec) == [
            'rules.extinction.per_type "own": both players start with every listed kind, so '
            "each is counted against both; say per_type true"
        ]
        spec["rules"]["extinction"]["per_type"] = True
        assert validate_spec(spec) == []

    def test_one_counted_kind_a_side_says_drop_per_type(self) -> None:
        spec = gap({"pieces": ["A", "C"], "per_type": "own"})
        assert _errors_about_own(spec) == [
            'rules.extinction.per_type "own": no player starts with two of the listed '
            "kinds, so counting each on its own is the count together; drop per_type"
        ]
        spec["rules"]["extinction"].pop("per_type")
        assert validate_spec(spec) == []

    def test_one_side_counted_and_the_other_not_is_a_game(self) -> None:
        """Only player 0's kinds listed: player 1 can never run out, and nothing else can
        say this (`true` and the summed rule both refuse it, §1)."""
        assert validate_spec(gap({"pieces": ["A", "B"], "per_type": "own"})) == []
        assert validate_spec(gap({"pieces": ["A", "B"], "per_type": True}))
        assert validate_spec(gap({"pieces": ["A", "B"]}))

    def test_beside_drops_is_refused(self) -> None:
        spec = gap(drops={"from_captures": True})
        errors = _errors_about_own(spec)
        assert "H11" in refused_ids(errors), validate_spec(spec)

    def test_beside_rounds_validates(self) -> None:
        assert validate_spec(gap(simultaneous={})) == []


# ------------------------------------------------------------------------ compile


class TestTheCompiledList:
    def test_own_resolves_each_side_s_kinds_from_the_opening_board(self) -> None:
        compiled = compile_spec(gap())
        rule = compiled.extinction
        assert rule is not None and rule.per_type and rule.own
        index = {pid: i for i, pid in enumerate(compiled.piece_ids)}
        assert rule.side_types == ((index["A"], index["B"]), (index["C"], index["E"]))

    def test_true_is_the_listed_types_for_both_sides(self) -> None:
        spec = gap({"pieces": ["A", "B"], "per_type": True})
        spec["setup"] = {"placements": [
            {"piece": "A", "squares": ["b1", "e1"]}, {"piece": "B", "squares": ["c1", "d1"]},
        ]}
        rule = compile_spec(spec).extinction
        assert rule is not None and rule.per_type and not rule.own
        assert rule.side_types == (rule.piece_types, rule.piece_types)

    def test_the_summed_rule_has_no_lists(self) -> None:
        rule = compile_spec(gap({"pieces": ["A", "B", "C", "E"]})).extinction
        assert rule is not None and not rule.per_type and rule.side_types == ((), ())

    def test_the_helper_keeps_the_listed_order_and_drops_repeats(self) -> None:
        compiled = compile_spec(gap())
        board = compiled.initial_board
        n_types, n_facings = compiled.n_types, compiled.n_facings
        assert _side_types(board, n_types, n_facings, (1, 0, 1, 3)) == ((1, 0), (3,))
        assert _side_types(board, n_types, n_facings, ()) == ((), ())

    def test_the_boundary_sends_the_lists_only_under_own(self) -> None:
        assert build_tables(compile_spec(gap()))["extinction_side_types"] == [[0, 1], [2, 3]]


def _stored_agree(paths: list[Path]) -> None:
    """Over stored games: the helper reads the army `schema.opening_counts` reads (one army,
    two readers), and the boundary sends no per-side list for any of them (inert) but the one
    that says `"own"`, which gets each side's own listed kinds."""
    for path in paths:
        spec = load_spec(path)
        compiled = compile_spec(spec)
        every = tuple(range(len(compiled.piece_ids)))
        lists = _side_types(compiled.initial_board, compiled.n_types, compiled.n_facings, every)
        counts = opening_counts(spec)
        for player in (0, 1):
            named = {compiled.piece_ids[t] for t in lists[player]}
            assert named == {p for p, n in counts[player].items() if n}, (path.stem, player)
        sent = build_tables(compiled)["extinction_side_types"]
        if OWN not in _schema_twenty_seven_uses(spec):
            assert sent == [], path.stem
            continue
        listed = set(spec["rules"]["extinction"]["pieces"])
        assert [{compiled.piece_ids[t] for t in side} for side in sent] == [
            {p for p, n in counts[player].items() if n and p in listed} for player in (0, 1)
        ], path.stem


def test_stored_extinction_games_agree() -> None:
    """The games an extinction rule is in, fast: every per-type and summed one stored."""
    _stored_agree([p for p in STORED if '"extinction"' in p.read_text(encoding="utf-8")])


@pytest.mark.slow
def test_every_stored_game_agrees() -> None:
    _stored_agree(STORED)


# ------------------------------------------------------------------------- play


def _engine(**rules: Any) -> Engine:
    return Engine(gap(**rules))


def _position(engine: Engine, pieces: dict[str, tuple[str, int]], to_move: int) -> GameState:
    return engine.make_state(pieces, side_to_move=to_move, ply=10)


FULL_ZERO = {"b1": ("A", 0), "e1": ("A", 0), "c1": ("B", 0), "d1": ("B", 0)}
FULL_ONE = {"b6": ("C", 1), "e6": ("C", 1), "c6": ("E", 1), "d6": ("E", 1)}


class TestAlternatingPlay:
    def test_the_game_is_not_over_before_the_first_move(self) -> None:
        """The record's whole point: §1's last row was over at ply zero."""
        engine = _engine()
        state = engine.initial_state()
        assert engine.terminal(state) is None
        assert engine.legal_moves(state)

    def test_running_out_of_one_own_kind_loses(self) -> None:
        engine = _engine()
        no_e = {"b6": ("C", 1), "e6": ("C", 1)}
        for to_move in (0, 1):
            state = _position(engine, FULL_ZERO | no_e, to_move)
            assert engine.terminal(state) == Outcome(winner=0, reason="extinction")
        no_a = {"c1": ("B", 0), "d1": ("B", 0)}
        state = _position(engine, no_a | FULL_ONE, 1)
        assert engine.terminal(state) == Outcome(winner=1, reason="extinction")

    def test_a_kind_the_side_did_not_start_with_is_never_counted(self) -> None:
        """Player 0 holds a knight it gained and no dabbaba: neither is its own kind."""
        engine = _engine()
        state = _position(engine, FULL_ZERO | FULL_ONE | {"a3": ("C", 0)}, 0)
        assert engine.terminal(state) is None

    def test_both_short_at_once_reads_the_last_mover_first(self) -> None:
        engine = _engine()
        both_short = {"b1": ("A", 0), "b6": ("C", 1)}
        # Player 1 to move: player 0 made the last move, so it ran out first.
        assert engine.terminal(_position(engine, both_short, 1)) == Outcome(1, "extinction")
        assert engine.terminal(_position(engine, both_short, 0)) == Outcome(0, "extinction")

    def test_random_games_end_where_the_oracle_says(self) -> None:
        """A declared test oracle: a side has run out when a kind it started with is gone."""
        engine = _engine()
        spec = engine.spec
        own = [{spec.piece_ids[t] for t in side} for side in spec.extinction.side_types]
        endings = 0
        for seed in range(12):
            rng = random.Random(seed)
            state = engine.initial_state()
            while True:
                outcome = engine.terminal(state)
                kinds = [set(), set()]
                for value in state.board:
                    if value:
                        player, piece_type = divmod(value - 1, spec.n_types)
                        kinds[player].add(spec.piece_ids[piece_type])
                short = [bool(own[side] - kinds[side]) for side in (0, 1)]
                if any(short):
                    mover = 1 - state.side_to_move
                    loser = mover if short[mover] else 1 - mover
                    assert outcome == Outcome(1 - loser, "extinction"), (seed, state.ply)
                    endings += 1
                    break
                if outcome is not None:
                    assert outcome.reason != "extinction"
                    break
                assert engine.hash(state) == engine._compute_hash(state)
                state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        assert endings >= 6


class TestRounds:
    def test_the_game_is_not_over_before_the_first_round(self) -> None:
        engine = _engine(simultaneous={})
        assert engine.simultaneous
        assert engine.terminal(engine.initial_state()) is None

    def test_one_side_short_after_a_round_is_that_side_s_result(self) -> None:
        engine = _engine(simultaneous={})
        state = _position(engine, FULL_ZERO | {"b6": ("C", 1), "e6": ("C", 1)}, 0)
        assert engine.terminal(state) == Outcome(winner=0, reason="extinction")

    def test_both_sides_short_after_a_round_is_a_draw(self) -> None:
        engine = _engine(simultaneous={})
        state = _position(engine, {"b1": ("A", 0), "b6": ("C", 1)}, 0)
        assert engine.terminal(state) == Outcome(winner=None, reason="extinction")

    def test_a_round_that_takes_one_kind_from_each_side_draws(self) -> None:
        """Played, not built: each side takes the other's last of a kind in one round."""
        engine = _engine(simultaneous={})
        state = _position(engine, {
            "c4": ("A", 0), "d2": ("B", 0), "e4": ("C", 1), "c5": ("E", 1),
        }, 0)
        assert engine.terminal(state) is None
        wazir_takes = _find(engine, state, 0, "c4", "c5")
        knight_takes = _find(engine, state, 1, "e4", "d2")
        after = engine.apply_round(state, wazir_takes, knight_takes)
        assert engine.terminal(after) == Outcome(winner=None, reason="extinction")
        # One side's taking alone: the other side's wazir steps aside instead.
        wazir_steps = _find(engine, state, 0, "c4", "b4")
        after = engine.apply_round(state, wazir_steps, knight_takes)
        assert engine.terminal(after) == Outcome(winner=1, reason="extinction")


def _find(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    for move in engine.legal_moves_for(state, player):
        text = engine.describe_move(state, move)
        if origin in text and text.rstrip("+#").endswith(target):
            return move
    raise LookupError((origin, target, [engine.describe_move(state, m) for m in engine.legal_moves_for(state, player)]))


# ------------------------------------------------------------------ fairy-stockfish


def test_fairy_stockfish_refuses_own() -> None:
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    with pytest.raises(Untranslatable) as refused:
        translate(gap())
    assert any('per_type "own"' in reason for reason in refused.value.reasons)


@pytest.mark.parametrize("spelling", [True, False])
def test_the_other_spellings_cross_as_before(spelling: bool) -> None:
    spec = gap({"pieces": ["A", "B"], "per_type": spelling})
    spec["setup"] = {"placements": [
        {"piece": "A", "squares": ["b1", "e1"]}, {"piece": "B", "squares": ["c1", "d1"]},
    ]}
    assert validate_spec(spec) == []
    assert build_tables(compile_spec(copy.deepcopy(spec)))["extinction_side_types"] == []
