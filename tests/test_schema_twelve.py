"""Schema 12: `position: "choose"`, the attacker places the struck piece (Q20).

The attacker takes the square, as under `advance`, and a struck piece that survives is
placed on an empty square of the declared `landing` set, read from where it stood in the
mover's frame. Specified in `docs/game-ideas.md`, *Q20 specified*.

The inertness proof is `test_schema_nine.py::TestTheBumpMovedNothing`, which pins the
mechanics hash and tables fingerprint of every committed and archived ruleset. This file
holds the feature itself: gated, refused where it means nothing, played by the core, and
read by every surface `new-move-primitive` names.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_EN_PASSANT,
    MOVE_PROMOTION,
    decode_move,
    encode_move,
    move_from,
    move_landing,
    move_promotion,
    move_to,
)
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import _SCALES, distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import (
    MAX_LANDING_CANDIDATES,
    PLAYED_AS_CAPTURE,
    SCHEMA_VERSION,
    _schema_twelve_uses,
    strike_changes_anything,
    strike_survives_anything,
)

ROOT = Path(__file__).resolve().parents[1]

KING_STEP = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


@pytest.fixture(scope="module")
def rise() -> dict[str, Any]:
    return json.loads((ROOT / "archive" / "hierarchy-rise.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def throw() -> dict[str, Any]:
    return json.loads((ROOT / "archive" / "hierarchy-rise-throw.json").read_text(encoding="utf-8"))


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _name(square: int, width: int = 8) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


def _moves_between(engine: Engine, state: Any, origin: str, target: str) -> list[int]:
    wanted = (_square(origin), _square(target))
    return [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == wanted]


def _play(engine: Engine, state: Any, line: list[tuple[str, str]]) -> Any:
    for origin, target in line:
        (move,) = _moves_between(engine, state, origin, target)
        state = engine.apply(state, move)
    return state


def _occupant(engine: Engine, state: Any, square: str) -> tuple[int, str] | None:
    value = state.board[_square(square)]
    if value == 0:
        return None
    return engine.spec.owner_of(value), engine.spec.piece_ids[engine.spec.type_of(value)]


def _position(throw: dict[str, Any], placements: list[dict[str, Any]]) -> dict[str, Any]:
    """The showcase's rules on a hand-built position."""
    spec = copy.deepcopy(throw)
    spec["rules"].pop("castling")
    spec["setup"] = {"mirror": False, "placements": placements}
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


class TestTheVersionIsBumped:
    def test_the_schema_version_is_twelve(self) -> None:
        assert SCHEMA_VERSION >= 12

    def test_the_collector_names_both_spellings(
        self, rise: dict[str, Any], throw: dict[str, Any]
    ) -> None:
        assert _schema_twelve_uses(rise) == set()
        assert _schema_twelve_uses(throw) == {
            "rules.strike.position=choose",
            "rules.strike.landing",
        }
        contacts = {
            "rules": {
                "contacts": {"position": "choose", "landing": KING_STEP, "links": []}
            }
        }
        assert _schema_twelve_uses(contacts) == {
            "rules.contacts.position=choose",
            "rules.contacts.landing",
        }

    def test_an_older_stamp_is_refused_by_name(self, throw: dict[str, Any]) -> None:
        spec = copy.deepcopy(throw)
        spec["schema_version"] = 11
        errors = validate_spec(spec)
        assert any("position=choose" in e and "schema 12" in e for e in errors), errors


class TestTheValidator:
    def test_the_showcase_validates(self, throw: dict[str, Any]) -> None:
        assert validate_spec(throw) == []

    def test_choose_needs_a_landing(self, throw: dict[str, Any]) -> None:
        spec = copy.deepcopy(throw)
        spec["rules"]["strike"].pop("landing")
        assert any("needs rules.strike.landing" in e for e in validate_spec(spec))

    def test_a_landing_needs_choose(self, rise: dict[str, Any]) -> None:
        spec = copy.deepcopy(rise)
        spec["schema_version"] = 12
        spec["rules"]["strike"]["landing"] = KING_STEP
        errors = validate_spec(spec)
        assert any("only means something under position 'choose'" in e for e in errors)

    @pytest.mark.parametrize(
        "landing, said",
        [
            ({"vectors": [[1, 0]], "range": 2}, "range"),
            ({"vectors": []}, "non-empty"),
            ({"vectors": [[0, 0]]}, "null vector"),
            ({"vectors": [[9, 0]]}, "within"),
            ({"vectors": [[1, 0]], "symmetry": "spiral"}, "symmetry"),
            ("king", "must be an object"),
        ],
    )
    def test_a_landing_is_checked_as_a_descriptor_is(
        self, throw: dict[str, Any], landing: Any, said: str
    ) -> None:
        spec = copy.deepcopy(throw)
        spec["rules"]["strike"]["landing"] = landing
        errors = validate_spec(spec)
        assert any("rules.strike.landing" in e and said in e for e in errors), errors

    def test_choose_beside_a_table_that_keeps_nothing_is_refused(
        self, throw: dict[str, Any]
    ) -> None:
        spec = copy.deepcopy(throw)
        spec["rules"].pop("strike")
        spec["rules"]["contacts"] = {
            "position": "choose",
            "landing": KING_STEP,
            "immune": ["K"],
            "links": [{"mover": "P", "onto": "N", "mover_to": "N"}],
        }
        errors = validate_spec(spec)
        assert any("'choose' needs a link that keeps a target alive" in e for e in errors)

    def test_the_contact_spelling_validates(self, throw: dict[str, Any]) -> None:
        spec = copy.deepcopy(throw)
        spec["rules"].pop("strike")
        spec["rules"]["contacts"] = {
            "position": "choose",
            "landing": KING_STEP,
            "immune": ["K"],
            "links": [{"mover": "P", "onto": "N", "onto_to": "B"}],
        }
        assert validate_spec(spec) == []


def _strike_at(throw: dict[str, Any], position: str, chain: list[dict[str, str]]) -> dict[str, Any]:
    """The showcase with its chain replaced, under ``position``; ``immune`` stays ``["K"]``."""
    spec = copy.deepcopy(throw)
    strike = spec["rules"]["strike"]
    strike["position"] = position
    strike["chain"] = chain
    if position != "choose":
        strike.pop("landing")
    return spec


def _contacts_at(
    throw: dict[str, Any], position: str, links: list[dict[str, Any]]
) -> dict[str, Any]:
    """The showcase re-spelled as a contact table of ``links``, with ``immune: ["K"]``."""
    spec = copy.deepcopy(throw)
    spec["rules"].pop("strike")
    spec["rules"]["contacts"] = {"position": position, "immune": ["K"], "links": links}
    if position == "choose":
        spec["rules"]["contacts"]["landing"] = KING_STEP
    return spec


def _survives(spec: dict[str, Any]) -> bool:
    compiled = compile_spec(spec).strike
    return compiled is not None and compiled.survives_anything


class TestNothingToPlaceIsTheCompilersAnswer:
    """Q271: `advance` and `choose` ask what `StrikeRule.survives_anything` asks.

    A strike never reaches an immune piece, so a link from one keeps nothing alive, and a
    chain of nothing else compiles to no rule at all. The two refusals used to count links
    rather than survivors: such a chain validated under `choose` with nothing it could
    ever place, and was refused under `advance` for a survivor that could never exist.
    """

    DEAD = [{"from": "K", "to": "Q"}]
    LIVE = [{"from": "P", "to": "N"}]

    def test_a_chain_whose_every_link_is_immune_is_refused_under_choose(
        self, throw: dict[str, Any]
    ) -> None:
        errors = validate_spec(_strike_at(throw, "choose", self.DEAD))
        assert any(
            "'choose' needs a chain that keeps a struck piece alive" in e
            and "rules.strike.immune" in e
            for e in errors
        ), errors

    def test_the_compiler_drops_that_chain(self, throw: dict[str, Any]) -> None:
        """The premise: the chain compiles to plain capture, and is refused for it (Q538)."""
        spec = _strike_at(throw, "stay", self.DEAD)
        ids = [piece["id"] for piece in throw["pieces"]]
        assert not strike_changes_anything(spec["rules"]["strike"], "strike", ids)
        (refusal,) = validate_spec(spec)
        assert PLAYED_AS_CAPTURE in refusal, refusal

    def test_a_table_whose_every_target_is_immune_is_refused_under_choose(
        self, throw: dict[str, Any]
    ) -> None:
        # Still a rule, because the second link changes the mover, but nothing survives.
        links = [
            {"mover": "P", "onto": "K", "onto_to": "Q"},
            {"mover": "P", "onto": "N", "mover_to": "N"},
        ]
        errors = validate_spec(_contacts_at(throw, "choose", links))
        assert any(
            "'choose' needs a link that keeps a target alive" in e
            and "rules.contacts.immune" in e
            for e in errors
        ), errors
        compiled = compile_spec(_contacts_at(throw, "stay", links)).strike
        assert compiled is not None and not compiled.survives_anything

    def test_one_live_link_beside_an_immune_one_still_validates(
        self, throw: dict[str, Any]
    ) -> None:
        assert validate_spec(_strike_at(throw, "choose", self.LIVE + self.DEAD)) == []
        own = [{"mover": "P", "onto": "N", "own": True, "onto_to": "B"}]
        assert validate_spec(_contacts_at(throw, "choose", own)) == []

    def test_the_dead_chain_is_refused_under_advance_as_stating_nothing(
        self, throw: dict[str, Any]
    ) -> None:
        """The other half: under `advance` the dead chain is refused, but not by that one.

        Until Q538 it stood, because `advance` is what `choose` is refused as. It is still
        not refused for a survivor it cannot have: its one refusal is Q538's, that it
        states a chain the core never plays, so the two refusals do not send a designer
        round in a circle.
        """
        for spec in (
            _strike_at(throw, "advance", self.DEAD),
            _contacts_at(throw, "advance", [{"mover": "P", "onto": "K", "onto_to": "Q"}]),
        ):
            (refusal,) = validate_spec(spec)
            assert PLAYED_AS_CAPTURE in refusal, refusal

    def test_advance_names_only_the_links_that_keep_a_piece_alive(
        self, throw: dict[str, Any]
    ) -> None:
        errors = validate_spec(_strike_at(throw, "advance", self.LIVE + self.DEAD))
        (refusal,) = [e for e in errors if "'advance' has nowhere to put" in e]
        assert "(P)" in refusal, refusal

    def test_every_one_link_rule_is_refused_exactly_where_the_compiler_says(
        self, throw: dict[str, Any]
    ) -> None:
        """`choose` stands exactly where something survives, `advance` exactly where not.

        Every one-link chain over the showcase's six pieces, and a pawn's one-link table
        onto each of them, that the validator accepts under `stay` or refuses there only
        as stating nothing (Q538). Both sides of each case, so neither refusal can drift
        alone. Since Q538 a rule with no survivor is refused under `advance` too, but as
        stating nothing and never as having a survivor to put somewhere.
        """
        ids = [piece["id"] for piece in throw["pieces"]]
        rules = [
            (_strike_at, "strike", [{"from": a, "to": b}]) for a in ids for b in ids if a != b
        ] + [
            (_contacts_at, "contacts",
             [{"mover": "P", "onto": b, "onto_to": "Q" if b != "Q" else "R"}])
            for b in ids
        ]
        seen = {True: 0, False: 0}
        for build, key, links in rules:
            errors = validate_spec(build(throw, "stay", links))
            if errors and not (len(errors) == 1 and PLAYED_AS_CAPTURE in errors[0]):
                continue
            survives = strike_survives_anything(build(throw, "stay", links)["rules"][key], key, ids)
            if survives:
                assert _survives(build(throw, "stay", links)), links
            seen[survives] += 1
            assert (validate_spec(build(throw, "choose", links)) == []) is survives, links
            advance = validate_spec(build(throw, "advance", links))
            nowhere = [e for e in advance if "'advance' has nowhere to put" in e]
            assert bool(nowhere) is survives, links
            if not survives:
                assert len(advance) == 1 and PLAYED_AS_CAPTURE in advance[0], links
        assert seen[True] and seen[False], seen


class TestWhetherTheRuleIsThereIsTheCompilersAnswer:
    """Q417: `strike_changes_anything` is the compiler keeping the rule, for both spellings.

    `strike_survives_anything` answers the question `choose` and `advance` ask, and for a
    strike it is the whole answer, since a strike changes only its victim. A contacts
    table is also kept when only the attacker changes, so a reader asking whether capture
    is still capture needs the whole answer, and this is it.
    """

    def test_every_one_link_rule_is_kept_exactly_where_the_compiler_keeps_it(
        self, throw: dict[str, Any]
    ) -> None:
        ids = [piece["id"] for piece in throw["pieces"]]
        rules = [
            (_strike_at, "strike", [{"from": a, "to": b}]) for a in ids for b in ids if a != b
        ] + [
            (_contacts_at, "contacts", [{"mover": "P", "onto": b, **change}])
            for b in ids
            for change in (
                {"onto_to": "Q" if b != "Q" else "R"},
                {"mover_to": "Q"},
                {"mover_to": "Q", "onto_to": "Q" if b != "Q" else "R"},
            )
        ]
        seen: dict[tuple[str, bool, bool], int] = {}
        for build, key, links in rules:
            spec = build(throw, "stay", links)
            errors = validate_spec(spec)
            # Q538: the validator refuses a dropped rule for exactly that, so it is
            # dropped where that is its one refusal and kept where there is none.
            dropped = len(errors) == 1 and PLAYED_AS_CAPTURE in errors[0]
            if errors and not dropped:
                continue
            kept = not dropped
            if kept:
                assert compile_spec(spec).strike is not None, links
            rule = spec["rules"][key]
            assert strike_changes_anything(rule, key, ids) is kept, links
            survives = strike_survives_anything(rule, key, ids)
            if key == "strike":
                assert survives is kept, links
            case = (key, kept, survives)
            seen[case] = seen.get(case, 0) + 1
        # Both spellings kept and dropped, and a table kept with nothing left standing:
        # the case the survivor question answers wrongly.
        assert {
            ("strike", True, True),
            ("strike", False, False),
            ("contacts", True, True),
            ("contacts", True, False),
            ("contacts", False, False),
        } <= set(seen), seen

    def test_a_table_whose_every_target_is_immune_is_no_rule(
        self, throw: dict[str, Any]
    ) -> None:
        ids = [piece["id"] for piece in throw["pieces"]]
        dead = [
            {"mover": "P", "onto": "K", "mover_to": "Q"},
            {"mover": "N", "onto": "K", "onto_to": "Q"},
        ]
        spec = _contacts_at(throw, "stay", dead)
        (refusal,) = validate_spec(spec)
        assert PLAYED_AS_CAPTURE in refusal, refusal
        assert not strike_changes_anything(spec["rules"]["contacts"], "contacts", ids)
        # One link onto a pawn that changes only the mover is a rule with no survivor.
        live = _contacts_at(throw, "stay", [*dead, {"mover": "P", "onto": "P", "mover_to": "N"}])
        assert validate_spec(live) == []
        assert strike_changes_anything(live["rules"]["contacts"], "contacts", ids)
        assert not strike_survives_anything(live["rules"]["contacts"], "contacts", ids)


class TestTheCompiledTables:
    def test_no_other_position_sends_a_landing(self, rise: dict[str, Any]) -> None:
        compiled = compile_spec(rise)
        assert compiled.strike is not None and compiled.strike.landing == ()
        tables = build_tables(compiled)
        assert tables["strike_landing"] == [] and tables["strike_landing_index"] == []
        assert not {"strike_landing", "strike_landing_index"} & set(canonical_tables(compiled))

    def test_a_king_step_resolves_in_ascending_order(self, throw: dict[str, Any]) -> None:
        strike = compile_spec(throw).strike
        assert strike is not None and strike.position == "choose"
        for player in (0, 1):
            assert [_name(s) for s in strike.landing[player][_square("a1")]] == [
                "b1", "a2", "b2"
            ]
            centre = strike.landing[player][_square("e4")]
            assert list(centre) == sorted(centre) and len(centre) == 8
            assert _square("e4") not in centre

    def test_the_second_army_reads_the_set_mirrored(self, throw: dict[str, Any]) -> None:
        spec = copy.deepcopy(throw)
        spec["rules"]["strike"]["landing"] = {"vectors": [[0, 1]]}
        strike = compile_spec(spec).strike
        assert strike is not None
        assert [_name(s) for s in strike.landing[0][_square("e4")]] == ["e5"]
        assert [_name(s) for s in strike.landing[1][_square("e4")]] == ["e3"]

    def test_a_wall_is_never_a_landing_square(self, throw: dict[str, Any]) -> None:
        spec = copy.deepcopy(throw)
        spec["rules"]["obstacles"] = {"squares": ["d4"]}
        strike = compile_spec(spec).strike
        assert strike is not None
        assert _square("d4") not in strike.landing[0][_square("e4")]

    def test_the_word_bound_is_refused_by_name(self, throw: dict[str, Any]) -> None:
        compiled = compile_spec(throw)
        assert rust_supports(compiled) is None
        strike = compiled.strike
        assert strike is not None
        wide = tuple(range(MAX_LANDING_CANDIDATES + 1))
        landing = tuple(
            tuple(wide if square == 0 else row for square, row in enumerate(per))
            for per in strike.landing
        )
        widened = dataclasses.replace(compiled, strike=dataclasses.replace(strike, landing=landing))
        refusal = rust_supports(widened)
        assert refusal is not None and "rules.strike.landing" in refusal and "129" in refusal


class TestTheMoveWord:
    def test_the_landing_bits_leave_every_other_field_alone(self) -> None:
        for landing in (0, 1, 64, 127):
            move = encode_move(200, 255, MOVE_PROMOTION, 31, landing)
            assert decode_move(move) == (200, 255, MOVE_PROMOTION, 31)
            assert move_promotion(move) == 31 and move_landing(move) == landing
        assert encode_move(3, 9) == encode_move(3, 9, landing=0)


class TestTheCorePlaysIt:
    def test_perft_matches_the_parent_until_a_capture(
        self, rise: dict[str, Any], throw: dict[str, Any]
    ) -> None:
        swap, choose = Engine(rise), Engine(throw)
        for depth in (1, 2):
            assert choose.perft(choose.initial_state(), depth) == swap.perft(
                swap.initial_state(), depth
            )

    def test_a_boxed_in_piece_cannot_be_taken(self, throw: dict[str, Any]) -> None:
        def position(with_b8: bool) -> Engine:
            black = ["a8", "a7", "b7"] + (["b8"] if with_b8 else [])
            return Engine(
                _position(
                    throw,
                    [
                        {"piece": "K", "squares": ["e1"]},
                        {"piece": "P", "squares": ["h2"]},
                        {"piece": "N", "squares": ["b6"]},
                        {"piece": "K", "player": 1, "squares": ["h8"]},
                        {"piece": "N", "player": 1, "squares": black[:1]},
                        {"piece": "B", "player": 1, "squares": black[1:]},
                    ],
                )
            )

        boxed = position(with_b8=True)
        assert _moves_between(boxed, boxed.initial_state(), "b6", "a8") == []
        free = position(with_b8=False)
        (move,) = _moves_between(free, free.initial_state(), "b6", "a8")
        assert free.landing_square(free.initial_state(), move) == _square("b8")
        after = free.apply(free.initial_state(), move)
        assert _occupant(free, after, "a8") == (0, "N")
        assert _occupant(free, after, "b8") == (1, "B")

    def test_en_passant_places_its_victim_and_may_use_the_origin(
        self, throw: dict[str, Any]
    ) -> None:
        engine = Engine(
            _position(
                throw,
                [
                    {"piece": "K", "squares": ["e1"]},
                    {"piece": "P", "squares": ["e5"]},
                    {"piece": "K", "player": 1, "squares": ["e8"]},
                    {"piece": "P", "player": 1, "squares": ["d7"]},
                ],
            )
        )
        state = _play(engine, engine.initial_state(), [("e1", "f1"), ("d7", "d5")])
        takes = _moves_between(engine, state, "e5", "d6")
        assert all(decode_move(m)[2] == MOVE_EN_PASSANT for m in takes)
        placed = {_name(engine.landing_square(state, m) or -1): m for m in takes}
        assert sorted(placed) == ["c4", "c5", "c6", "d4", "e4", "e5", "e6"]
        texts = [engine.describe_move(state, m) for m in takes]
        assert len(set(texts)) == len(texts)

        after = engine.apply(state, placed["e5"])
        assert _occupant(engine, after, "e5") == (1, "N")
        assert _occupant(engine, after, "d6") == (0, "P")
        assert _occupant(engine, after, "d5") is None
        assert after.key == engine._compute_hash(after)

    def test_make_and_unmake_agree_with_a_walk_that_never_unmakes(
        self, throw: dict[str, Any]
    ) -> None:
        """Perft plays every move with `make` and takes it back with `unmake`. `apply`
        builds each position afresh and never unmakes. A survivor `unmake` failed to lift
        would show up as a different count one ply later."""
        engine = Engine(
            _position(
                throw,
                [
                    {"piece": "K", "squares": ["e1"]},
                    {"piece": "P", "squares": ["e5", "c2"]},
                    {"piece": "N", "squares": ["f3"]},
                    {"piece": "K", "player": 1, "squares": ["e8"]},
                    {"piece": "P", "player": 1, "squares": ["d7", "d3"]},
                    {"piece": "B", "player": 1, "squares": ["g4"]},
                ],
            )
        )
        state = _play(engine, engine.initial_state(), [("e1", "d1"), ("d7", "d5")])
        walked = 0
        placing = 0
        for first in engine.legal_moves(state):
            placing += engine.landing_square(state, first) is not None
            one = engine.apply(state, first)
            for second in engine.legal_moves(one):
                walked += len(engine.legal_moves(engine.apply(one, second)))
        assert placing > 1
        assert engine.perft(state, 3) == walked

    def test_a_short_game_keeps_its_incremental_hash(self, throw: dict[str, Any]) -> None:
        engine = Engine(throw)
        rng = random.Random(20)
        state = engine.initial_state()
        placed = 0
        for _ in range(120):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            placing = [m for m in moves if engine.landing_square(state, m) is not None]
            placed += bool(placing)
            move = rng.choice(placing) if placing else rng.choice(moves)
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state), "hash drifted"
        assert placed > 5


class TestTheOtherSurfaces:
    def test_the_description_says_where_the_piece_goes(self, throw: dict[str, Any]) -> None:
        said = describe_spec(throw)
        assert "places the struck piece on any empty square a king's step" in said
        assert "cannot be taken" in said

    def test_a_move_names_its_landing_square(self, throw: dict[str, Any]) -> None:
        engine = Engine(throw)
        state = _play(
            engine,
            engine.initial_state(),
            [("e2", "e4"), ("d7", "d5")],
        )
        texts = [engine.describe_move(state, m) for m in _moves_between(engine, state, "e4", "d5")]
        assert texts and all("(placed on " in text for text in texts)
        assert len(set(texts)) == len(texts)

    def test_it_has_a_distance_feature_of_its_own(
        self, rise: dict[str, Any], throw: dict[str, Any]
    ) -> None:
        assert "strike_places_the_survivor" in _SCALES
        assert features(throw)["strike_places_the_survivor"] == 1.0
        assert features(rise)["strike_places_the_survivor"] == 0.0
        assert distance(throw, rise) > 0.0

    def test_a_landing_is_charged_as_a_descriptor(
        self, rise: dict[str, Any], throw: dict[str, Any]
    ) -> None:
        before, after = measure_complexity(rise), measure_complexity(throw)
        assert after.move_descriptors == before.move_descriptors + 1
        assert after.distinct_directions == before.distinct_directions + 8

    def test_fairy_refuses_it_by_name(self, throw: dict[str, Any]) -> None:
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(throw))
        assert any("rules.strike" in reason for reason in refusal.value.reasons)

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        rules = rule_spec_schema()["properties"]["rules"]["properties"]
        for key in ("strike", "contacts"):
            assert "choose" in rules[key]["properties"]["position"]["enum"]
            assert "vectors" in rules[key]["properties"]["landing"]["properties"]
