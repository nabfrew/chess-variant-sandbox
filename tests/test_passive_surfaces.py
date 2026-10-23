"""Schema 30's lines of sight on the surfaces that read a spec without playing it (Q999a,
`docs/archive/plan-passive.md` §4.2, §4.3): the words a piece and the rules are said in, the designer's
words, the complexity charge, the distance features, the relation between two games, the
settle that keeps an edit from leaving a sight that can never bind, and the three operators.

The rule itself (the validator, the compiler, the core and its oracle) is `tests/test_sight.py`'s.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.loop.operators import OPERATORS, refusal
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_pieces, describe_rules, describe_spec
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.plain import plain_problem
from chess_variant_sandbox.spec.relate import identity_key, normal_form, rules_involved
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, piece_sights
from chess_variant_sandbox.spec.transform import drop_piece, settle_inert_rules
from tests.feature_inventory import _schema_thirty_uses

ROOT = Path(__file__).resolve().parents[1]
SIGHT_SPECS = sorted((ROOT / "tests").glob("_sight_*.json"))

FREEZE_QUEEN = {
    "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
    "effect": "freeze", "whose": "enemy", "affects": ["Q"],
}
SHIELDING_KING = {
    "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 2, "through": True,
    "effect": "shield", "whose": "own",
}
DISARMING_KNIGHT = {
    "vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1,
    "effect": "disarm", "whose": "enemy",
}


def _chess(**sights: Any) -> dict[str, Any]:
    """Chess at schema 30, with each named type given the listed lines of sight."""
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    spec["schema_version"] = SCHEMA_VERSION
    for piece in spec["pieces"]:
        if piece["id"] in sights:
            piece["sight"] = copy.deepcopy(sights[piece["id"]])
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _notes(spec: dict[str, Any], pid: str) -> tuple[str, ...]:
    return next(piece.notes for piece in describe_pieces(spec) if piece.id == pid)


# ------------------------------------------------------------------------- describe.py


def test_each_sight_is_one_sentence_on_its_piece() -> None:
    spec = _chess(Q=[FREEZE_QUEEN], K=[SHIELDING_KING], N=[DISARMING_KNIGHT])
    assert _notes(spec, "Q") == (
        "freezes enemy queens it sees any distance in all eight directions, as far as the "
        "first piece in the way (see the rules)",
    )
    assert _notes(spec, "K") == (
        "shields its own pieces it sees up to two squares in all eight directions, through "
        "any piece between (see the rules)",
    )
    # A one-jump line has nothing between, so neither blocking clause is said.
    assert _notes(spec, "N") == (
        "disarms enemy pieces it sees at a jump of two and one, in all eight directions "
        "(see the rules)",
    )


def test_the_rules_say_what_each_effect_does_once_and_only_for_the_effects_used() -> None:
    frozen = " ".join(describe_rules(_chess(Q=[FREEZE_QUEEN])))
    assert "A frozen piece cannot move at all" in frozen
    assert "so it gives no check" in frozen
    assert "A freeze is never lifted" in frozen and "(or checkmate, in check)" in frozen
    assert "disarmed" not in frozen and "shielded" not in frozen
    shielded = " ".join(describe_rules(_chess(K=[SHIELDING_KING])))
    assert "a shielded royal is never in check" in shielded
    assert "answered by shielding the royal" in shielded
    assert "frozen" not in shielded and "never lifted" not in shielded
    assert "line of sight" not in " ".join(describe_rules(_chess()))


def test_in_rounds_an_effect_decides_what_is_offered_never_the_collision() -> None:
    spec = load_spec(ROOT / "tests" / "_sight_rounds.json")
    said = describe_rules(spec)
    assert (
        "An effect decides which moves are offered, never how the round resolves: a "
        "disarmed piece can still win a collision." in said
    )
    # No check in rounds, so nothing says one.
    paragraph = next(line for line in said if line.startswith("A piece in another's line"))
    assert "check" not in paragraph


@pytest.mark.parametrize("path", SIGHT_SPECS, ids=lambda path: path.stem)
def test_a_sight_taken_away_changes_the_description(path: Path) -> None:
    """Complexity charges a sight (`_sight_extras`), so the prose must state it."""
    spec = load_spec(path)
    for piece in spec["pieces"]:
        if "sight" not in piece:
            continue
        without = copy.deepcopy(spec)
        next(p for p in without["pieces"] if p["id"] == piece["id"]).pop("sight")
        assert describe_spec(without) != describe_spec(spec), piece["id"]


# ---------------------------------------------------------------------------- plain.py


def test_a_problem_inside_a_line_of_sight_names_the_line() -> None:
    spec = _chess(Q=[FREEZE_QUEEN])
    spec["pieces"][4]["sight"][0]["effect"] = "melt"
    said = [plain_problem(problem, spec) for problem in validate_spec(spec)]
    assert any(line.startswith("Piece 5 (queen), line of sight 1: 'what it does'") for line in said)


# ----------------------------------------------------------------------- complexity.py


@pytest.mark.parametrize(
    ("sights", "charge"),
    [
        ({"Q": [FREEZE_QUEEN]}, 1.0),
        ({"Q": [FREEZE_QUEEN], "R": [dict(FREEZE_QUEEN, affects=["R"])]}, 1.25),
        ({"Q": [FREEZE_QUEEN], "N": [DISARMING_KNIGHT]}, 2.0),
        ({"K": [SHIELDING_KING]}, 1.25),
        ({"Q": [FREEZE_QUEEN], "N": [DISARMING_KNIGHT], "K": [SHIELDING_KING]}, 3.25),
    ],
    ids=["one freeze", "two freezes", "freeze and disarm", "a shield aura", "all three"],
)
def test_each_effect_is_one_rule_each_further_line_a_quarter_and_an_aura_a_quarter(
    sights: dict[str, Any], charge: float
) -> None:
    plain = measure_complexity(_chess()).special_rules
    assert measure_complexity(_chess(**sights)).special_rules - plain == pytest.approx(charge)


# ------------------------------------------------------------------------- distance.py


def test_a_spec_differing_only_by_a_sight_is_not_at_distance_zero() -> None:
    chess = _chess()
    for sights in ({"Q": [FREEZE_QUEEN]}, {"K": [SHIELDING_KING]}, {"N": [DISARMING_KNIGHT]}):
        assert distance(chess, _chess(**sights)) > 0, sights
    shielded = features(_chess(K=[SHIELDING_KING]))
    assert shielded["sight_shield_share"] == pytest.approx(1 / 6)
    assert shielded["sight_aura_share"] == 1.0
    assert shielded["sight_freeze_share"] == shielded["sight_disarm_share"] == 0.0
    assert {key: value for key, value in features(chess).items() if key.startswith("sight_")} == {
        "sight_freeze_share": 0.0, "sight_disarm_share": 0.0, "sight_shield_share": 0.0,
        "sight_command_share": 0.0, "sight_aura_share": 0.0, "sight_turncoat_share": 0.0,
    }


def test_a_freezing_queen_and_a_disarming_queen_measure_apart() -> None:
    """Q1011: `_SCALES` lacked the sight features, so `distance` never read them and two
    games differing only in the effect of one queen's sight measured 0.0 from each other."""
    freezing = _chess(Q=[FREEZE_QUEEN])
    disarming = _chess(Q=[dict(FREEZE_QUEEN, effect="disarm")])
    assert distance(freezing, disarming) > 0
    assert distance(freezing, disarming) == distance(disarming, freezing)
    # one share each way: the freeze share and the disarm share, of the 100 features
    assert distance(freezing, disarming) == pytest.approx(2 * (1 / 6) / 100, abs=1e-4)


# --------------------------------------------------------------------------- relate.py


def test_a_sight_is_a_rule_the_relation_sees() -> None:
    assert "piece.sight" in rules_involved(normal_form(_chess(Q=[FREEZE_QUEEN])))
    assert "piece.sight" not in rules_involved(normal_form(_chess()))
    assert identity_key(_chess(Q=[FREEZE_QUEEN])) != identity_key(_chess())
    assert identity_key(_chess(Q=[FREEZE_QUEEN])) != identity_key(
        _chess(Q=[dict(FREEZE_QUEEN, effect="disarm")])
    )


def test_a_sight_spelled_two_ways_is_one_game() -> None:
    """The defaults written out, the vectors in another order, and a one-step line made an
    aura (nothing between to see through) are spellings, not rules."""
    written = dict(FREEZE_QUEEN, vectors=[[1, 1], [1, 0]], range=0, min_range=1)
    assert identity_key(_chess(Q=[written])) == identity_key(_chess(Q=[FREEZE_QUEEN]))
    step = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
            "effect": "shield", "whose": "own"}
    assert identity_key(_chess(K=[step])) == identity_key(_chess(K=[dict(step, through=True)]))


@pytest.mark.parametrize("path", SIGHT_SPECS, ids=lambda path: path.stem)
def test_the_relation_reads_a_sight_live_exactly_where_the_compiler_keeps_one(
    path: Path,
) -> None:
    """`relate._live_sights` asks `piece_sights` rather than compiling; this holds the two
    readings together."""
    spec = load_spec(path)
    assert (compile_spec(spec).sight is not None) == bool(piece_sights(spec))
    assert ("piece.sight" in rules_involved(normal_form(spec))) == bool(piece_sights(spec))


# ------------------------------------------------------------------------ transform.py


def test_an_edit_that_leaves_a_sight_unable_to_bind_drops_it() -> None:
    """A disarm on pawns whose only capture an edit took away can never bind: the settle
    takes the line, and the key with its last line, so `mutate` does not draw again."""
    spec = _chess(N=[dict(DISARMING_KNIGHT, affects=["P"]), DISARMING_KNIGHT])
    pawn = spec["pieces"][0]
    pawn["moves"] = [move for move in pawn["moves"] if move.get("mode") != "capture"]
    spec["rules"].pop("en_passant", None)
    assert any("affects nothing that could be there" in e for e in validate_spec(spec))
    assert settle_inert_rules(spec) == ["sight:N"]
    assert spec["pieces"][1]["sight"] == [DISARMING_KNIGHT]
    assert validate_spec(spec) == []
    spec["pieces"][1]["sight"] = [dict(DISARMING_KNIGHT, affects=["P"])]
    assert settle_inert_rules(spec) == ["sight:N"]
    assert "sight" not in spec["pieces"][1]
    assert validate_spec(spec) == []


def test_a_removed_piece_takes_the_lines_that_touched_only_it() -> None:
    spec = _chess(Q=[FREEZE_QUEEN, dict(FREEZE_QUEEN, affects=["Q", "R"])])
    drop_piece(spec, "R")
    assert spec["pieces"][-2]["sight"] == [FREEZE_QUEEN, FREEZE_QUEEN]
    spec = _chess(B=[dict(FREEZE_QUEEN, affects=["R"])])
    drop_piece(spec, "R")
    assert all("sight" not in piece for piece in spec["pieces"])
    assert validate_spec(spec) == []


# ------------------------------------------------------------------------ the operators


def _operator(name: str) -> Any:
    return next(op for op in OPERATORS if op.name == name)


def test_cast_a_gaze_gives_a_type_its_own_capture_geometry() -> None:
    chess = _chess()
    cast = _operator("cast-a-gaze")
    effects: set[str] = set()
    for seed in range(12):
        child = copy.deepcopy(chess)
        said = cast.apply(child, random.Random(seed))
        assert said is not None and said.startswith("the ")
        assert refusal(chess, child) == [], refusal(chess, child)
        sights = piece_sights(child)
        (pid, (sight,)), = sights.items()
        effects.add(sight.effect)
        assert sight.whose == ("own" if sight.effect == "shield" else "enemy")
        piece = next(p for p in child["pieces"] if p["id"] == pid)
        captures = [m for m in piece["moves"] if m.get("mode", "both") != "move"]
        assert any(
            [list(v) for v in sight.vectors] == m["vectors"]
            and sight.range == (1 if m["kind"] == "leap" else m.get("range", 0))
            for m in captures
        ), (pid, sight)
    # Q1007c added the fourth effect, `command`, cast on the enemy as freeze and disarm are.
    assert effects == {"freeze", "disarm", "shield", "command"}


def test_cast_a_gaze_casts_nothing_beside_portals_or_an_explosion() -> None:
    atomic = load_spec(ROOT / "rulesets" / "atomic.json")
    for seed in range(6):
        assert _operator("cast-a-gaze").apply(copy.deepcopy(atomic), random.Random(seed)) is None


def test_blind_a_piece_and_widen_a_gaze_cross_back() -> None:
    spec = _chess(K=[SHIELDING_KING], N=[DISARMING_KNIGHT])
    widened = copy.deepcopy(spec)
    said = _operator("widen-a-gaze").apply(widened, random.Random(0))
    # Only the king's line is longer than a step; the knight's has nothing to see through.
    assert said is not None and "stops at the first piece" in said
    assert "through" not in widened["pieces"][-1]["sight"][0]
    assert "pieces.sight.through" not in _schema_thirty_uses(widened)
    assert refusal(spec, widened) == []
    blinded = copy.deepcopy(spec)
    said = _operator("blind-a-piece").apply(blinded, random.Random(0))
    assert said is not None and " no longer " in said
    assert sum("sight" in piece for piece in blinded["pieces"]) == 1
    assert refusal(spec, blinded) == []


def test_an_operator_that_would_be_refused_beside_a_sight_declines() -> None:
    """A blast is refused beside any sight, and a sweep or a wild piece beside a shield."""
    spec = _chess(K=[SHIELDING_KING])
    for name in ("set-the-blast", "sweep-a-slider", "set-a-patrol"):
        for seed in range(6):
            child = copy.deepcopy(spec)
            if _operator(name).apply(child, random.Random(seed)) is not None:
                assert validate_spec(child) == [], (name, seed, validate_spec(child))


# ------------------------------------------------------------------------ the proposer


def test_the_proposer_is_told_the_sight_its_effects_and_its_edges() -> None:
    from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE

    for words in ("`pieces[].sight`", '"freeze"', '"disarm"', '"shield"', '"through": true',
                  "gives no check", "There is no lift", "`rules.portals`",
                  "`rules.explode_on_capture`", "can never bind"):
        assert words in SCHEMA_PREAMBLE, words
