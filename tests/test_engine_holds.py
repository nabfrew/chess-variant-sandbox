"""U1 (and U1b's eleventh, H7): the combinations the engine refuses are named by the validator.

`docs/reference/unheld-pairs-plan.md` §1.2 and §1.3 found ten combinations that validated clean
and were then refused by `bridge.rust_supports` or the compiler, so an author heard them from a
traceback. The validator now refuses each by name (`spec/engine_holds.py`, messages in
`refusals.HOLD_REFUSALS`), the engine keeps its refusal as the second line, and `validate`
prints whatever the engine refuses as a refusal. Fixtures: the record's smallest specs
(`tests/engine_hold_specs.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from engine_hold_specs import HOLDS, push_beside_held_piece

from chess_variant_sandbox import cli
from chess_variant_sandbox.engine import bridge, moves
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import engine_holds, load_spec, validate_spec
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.refusals import refusal, refused_ids
from chess_variant_sandbox.spec.schema import MAX_LANDING_CANDIDATES, SpecError

ROOT = Path(__file__).resolve().parents[1]
IDS = list(HOLDS)


def _engine_refusal(spec: dict[str, Any]) -> str:
    """What the compiler or the boundary says of ``spec``, the validator's hold check aside."""
    with engine_holds.checking():
        assert validate_spec(spec) == []
        try:
            compiled = compile_spec(spec)
        except SpecError as error:
            return " ".join(error.errors)
    return bridge.rust_supports(compiled) or ""


@pytest.mark.parametrize("hold", IDS)
def test_the_validator_names_each_hold(hold: str) -> None:
    build, _ = HOLDS[hold]
    errors = validate_spec(build())
    assert errors == [refusal(f"hold.{hold}")]
    assert refused_ids(errors) == {f"hold.{hold}"}


def test_a_shared_piece_is_held_beside_a_push_as_a_held_one_is() -> None:
    errors = validate_spec(push_beside_held_piece("shared"))
    assert refused_ids(errors) == {"hold.H2"}


def test_a_majority_written_as_an_object_is_held_as_the_string_is() -> None:
    """R-L6e (F8, R-L6b's soak: 63 held cases): `control` written in its neighbourhood spelling
    (`{"majority": {...}}`) reached the engine unseen, since the cheap reading compared the raw
    value with the string. It reads the normal form through `schema.piece_control` now."""
    control = {"majority": {"vectors": [[1, 0]], "symmetry": "rotate4"}}
    spec = push_beside_held_piece(control)
    assert engine_holds._may_hold(normal_form(spec))
    assert refused_ids(validate_spec(spec)) == {"hold.H2"}
    assert "majority control" in _engine_refusal(spec)


@pytest.mark.parametrize("hold", IDS)
def test_the_engine_still_refuses_each_as_the_second_line(hold: str) -> None:
    """The validator moved nothing out of the engine: without its check, the compiler or the
    boundary refuses the same spec in the words the record quotes."""
    build, words = HOLDS[hold]
    assert words in _engine_refusal(build())


@pytest.mark.parametrize("hold", IDS)
def test_each_message_says_what_the_author_can_do(hold: str) -> None:
    message = refusal(f"hold.{hold}")
    assert "pieces[]" in message or "rules." in message
    assert any(word in message for word in ("leave one out", "give the", "use fewer"))


def test_the_bounds_are_the_move_words() -> None:
    assert engine_holds.TAG_VALUES == moves.MAX_SWEEP_RAYS == bridge.PUSH_DIRECTIONS_MAX
    assert engine_holds.TAG_VALUES == MAX_LANDING_CANDIDATES
    assert engine_holds.LONGEST_FOLD_RAY == moves.MAX_SWEEP_THROUGH


def test_no_library_game_reaches_the_compile() -> None:
    """The hold check costs a stored game nothing: its cheap reading lets every one through
    without compiling (and so refuses none)."""
    paths = sorted((ROOT / "rulesets").glob("*.json"))
    assert len(paths) > 20
    for path in paths:
        assert not engine_holds._may_hold(load_spec(path)), path.name


def test_compiling_asks_no_hold() -> None:
    """`compile_spec` opens with `validate_or_raise`, which skips the hold check, so the
    engine's own refusal comes from the boundary as before and nothing compiles twice."""
    build, words = HOLDS["H1"]
    compiled = compile_spec(build())
    assert words in (bridge.rust_supports(compiled) or "")


# ------------------------------------------------------------------------ the CLI


def _validate(tmp_path: Path, spec: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    code = cli.main(["validate", str(path)])
    return code, capsys.readouterr().out


@pytest.mark.parametrize("hold", ["H1", "H6", "H7", "C4"])
def test_validate_refuses_a_hold_by_name(
    hold: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _validate(tmp_path, HOLDS[hold][0](), capsys)
    assert code == 1
    assert refusal(f"hold.{hold}") in out


@pytest.mark.parametrize(("hold", "said"), [
    ("H1", "the engine cannot play these rules yet"),
    ("H6", "found when compiling"),
])
def test_validate_prints_an_engine_refusal_without_a_traceback(
    hold: str, said: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 4: a spec the validator passes and the engine refuses (here, with the hold check
    switched off) ends in a plain refusal and a non-zero exit, from the boundary or the
    compiler alike."""
    monkeypatch.setattr(engine_holds, "_may_hold", lambda spec: False)
    build, words = HOLDS[hold]
    code, out = _validate(tmp_path, build(), capsys)
    assert code == 1
    assert said in out and words in out
    assert "Traceback" not in out


# ------------------------------------------------------------------------ §2.3


def _crossings_in_rounds(both: str, tie: str | None) -> dict[str, Any]:
    spec = load_spec(ROOT / "rulesets" / "crossings.json")
    spec["rules"]["reach"]["both"] = both
    spec["rules"]["simultaneous"] = {} if tie is None else {"tie": tie}
    return spec


def test_mover_in_a_round_without_the_initiative_is_contradictory() -> None:
    """`goals.rounds.mover`: no one mover made a round's position, so it is its own row."""
    errors = validate_spec(_crossings_in_rounds("mover", None))
    assert any('both "mover" beside rules.simultaneous without the initiative' in e for e in errors)
    assert not any("rules.reach.both beside rules.simultaneous" in e for e in errors), errors


def test_the_rest_of_a_counting_goal_in_rounds_stays_not_built() -> None:
    errors = validate_spec(_crossings_in_rounds("draw", None))
    assert any("not yet read by the counting goal" in e for e in errors), errors
    assert not any("without the initiative" in e for e in errors)
