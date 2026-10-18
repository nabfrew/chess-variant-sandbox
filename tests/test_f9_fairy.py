"""The Fairy-Stockfish reporting paths nothing offline reached (archive/plan-tests F9, Q751b).

`fairy/` re-expresses the DSL for an outside engine and reports what it found. Most of
what was unrun is the *report*: the sentence a divergence is printed as, what a check
that could not compare says, the temporary definition files, and the `fairy` command's
three outcomes. These tests build the results by hand where the engine would only make
them slower, and run the real binary and `pyffish` where this machine has them (each such
test skips, with its reason, where it does not).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from test_fairy_results import FOOLS_MATE

from chess_variant_sandbox import cli, fairy
from chess_variant_sandbox.fairy import results as fairy_results
from chess_variant_sandbox.fairy.results import Difference, ResultCheck, check_results
from chess_variant_sandbox.fairy.verify import Verification
from chess_variant_sandbox.spec.canonical import spec_hash

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"



@pytest.fixture
def engine_here() -> None:
    """Skip, with the reason, where this machine has no usable Fairy-Stockfish."""
    if not fairy.available():
        pytest.skip("no usable Fairy-Stockfish binary on this machine")


needs_binary = pytest.mark.usefixtures("engine_here")


def load(name: str) -> dict[str, Any]:
    return json.loads((RULESETS / f"{name}.json").read_text(encoding="utf-8"))


class TestDifferenceText:
    def test_names_the_rule_the_moves_and_what_each_engine_said(self) -> None:
        difference = Difference(
            key="stalemate", fen="x", moves=["a2a3", "b7b6"], ours="a draw", theirs="a win"
        )
        assert difference.text() == "stalemate: ours says a draw after a2a3 b7b6, theirs says a win"

    def test_a_position_reached_without_moves_and_a_difference_about_no_rule(self) -> None:
        difference = Difference(key="", fen="x", moves=[], ours="a win", theirs="not over")
        assert difference.text() == "ours says a win, theirs says not over"


class TestResultCheckReport:
    def test_a_refusal_is_its_reasons_and_nothing_else(self) -> None:
        check = ResultCheck("h", "g", ok=False, reasons=["no drops over there", "no edges"],
                            questions=9, differences=[Difference("", "f", [], "a", "b")])
        assert check.summary() == "no drops over there; no edges"

    def test_differences_are_listed_when_there_are_no_reasons(self) -> None:
        check = ResultCheck(
            "h", "g", ok=False,
            differences=[Difference("max_plies", "f", ["e2e4"], "a draw", "a win"),
                         Difference("", "f", [], "a win", "a draw")],
        )
        assert check.summary() == (
            "max_plies: ours says a draw after e2e4, theirs says a win; "
            "ours says a win, theirs says a draw"
        )

    def test_agreement_says_how_many_positions_and_what_ended(self) -> None:
        check = ResultCheck("h", "g", ok=True, questions=42,
                            agreed={"checkmate": 3, "stalemate": 1})
        assert check.summary() == "agree at 42 positions; ended by checkmate 3, stalemate 1"

    def test_coverage_lists_the_agreed_the_admitted_and_the_unreached_in_that_order(self) -> None:
        check = ResultCheck(
            "h", "g", ok=True,
            agreed={"stalemate": 2, "checkmate": 5},
            admitted={"extinction": 1},
            unreached={"reach": "never played", "check_limit": "ditto"},
        )
        assert check.coverage() == (
            "ended by checkmate 5, stalemate 2; differ as noted: extinction 1; "
            "not reached: check_limit, reach"
        )

    def test_a_check_that_compared_no_ending_says_so(self) -> None:
        assert ResultCheck("h", "g", ok=True).coverage() == "no ending compared"


class TestVerificationSummary:
    def test_verified_says_the_depth_and_the_positions_with_its_notes(self) -> None:
        verification = Verification("h", "g", ok=True, depth=3, positions=120,
                                    notes=["a mating drop", "a clock"])
        assert verification.summary() == (
            "verified to depth 3 over 120 positions; a mating drop; a clock"
        )

    def test_verified_without_notes_has_no_trailing_clause(self) -> None:
        assert Verification("h", "g", ok=True, depth=2, positions=5).summary() == (
            "verified to depth 2 over 5 positions"
        )

    def test_a_failure_is_its_reasons_or_just_diverged(self) -> None:
        assert Verification("h", "g", ok=False, reasons=["a", "b"]).summary() == "a; b"
        assert Verification("h", "g", ok=False).summary() == "diverged"


class TestHash:
    def test_is_the_specs_identity_hash(self) -> None:
        spec = load("dull")
        assert fairy_results._hash(spec) == spec_hash(spec)

    def test_is_empty_for_a_spec_that_cannot_be_hashed_rather_than_an_error(self) -> None:
        assert fairy_results._hash(None) == ""  # type: ignore[arg-type]

    def test_a_refused_translation_is_reported_under_the_specs_hash_and_name(self) -> None:
        """No referee is needed to be told no: the reasons come from the translation."""
        spec = load("torus")
        check = check_results(spec, referee=None)  # type: ignore[arg-type]
        assert check.ok is False
        assert check.ruleset_hash == spec_hash(spec)
        assert check.name == spec["name"]
        assert check.reasons and check.questions == 0 and check.games == 0


class TestConfigFiles:
    def test_the_definition_files_are_removed_and_forgotten(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fairy, "_CONFIG_PATHS", {})
        variant = fairy.translation(load("dull"))
        assert variant is not None
        path = fairy.config_path(variant)
        assert os.path.isfile(path)
        assert Path(path).read_text(encoding="utf-8") == variant.config
        assert fairy.config_path(variant) == path, "one file per variant"
        fairy._clean_configs()
        assert not os.path.exists(path)
        assert fairy._CONFIG_PATHS == {}

    def test_a_file_already_gone_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fairy, "_CONFIG_PATHS", {"a": "does-not-exist.ini"})
        fairy._clean_configs()
        assert fairy._CONFIG_PATHS == {}


@needs_binary
class TestMatedAfter:
    """`UciEngine.mated_after` on the engine's own FENs: fool's mate is the position."""

    AFTER_TWO = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"

    def test_the_mating_move_is_mate_and_a_quiet_one_is_not(self) -> None:
        assert FOOLS_MATE[-1] == "d8h4"
        with fairy.UciEngine() as engine:
            assert engine.mated_after(self.AFTER_TWO, "d8h4") is True
            assert engine.mated_after(self.AFTER_TWO, "a7a6") is False

    def test_check_that_can_be_answered_is_not_mate(self) -> None:
        with fairy.UciEngine() as engine:
            # 1.e4 f6 2.Qh5+ g6: the check is answerable
            assert engine.mated_after(
                "rnbqkbnr/ppppp1pp/5p2/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2", "g7g6"
            ) is False
            assert engine.mated_after(
                "rnbqkbnr/ppppp1pp/5p2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "d1h5"
            ) is False

    def test_stalemate_is_not_mate(self) -> None:
        with fairy.UciEngine() as engine:
            # black king h8 boxed by Qg6 and Kf7 after Qg6: no check, no reply
            assert engine.mated_after("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1", "g1g6") is False


class FakeEngine:
    """Stands where `fairy.UciEngine` is entered, so the command's report can be driven."""

    def __enter__(self) -> FakeEngine:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class TestFairyCommand:
    def run(self, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
        code = cli.main(["fairy", *argv])
        return code, capsys.readouterr().out

    def test_config_prints_the_generated_definition_and_nothing_is_checked(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dull = RULESETS / "dull.json"
        code, out = self.run(capsys, str(dull), "--config")
        assert code == 0
        variant = fairy.translation(load("dull"))
        assert variant is not None
        assert out.startswith(f"# {dull}\n")
        assert variant.config in out
        assert "engine:" not in out

    def test_config_for_a_ruleset_with_no_translation_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out = self.run(capsys, str(RULESETS / "torus.json"), "--config")
        assert code == 0 and out.rstrip().endswith("# no translation")

    def test_without_a_binary_a_translation_is_written_and_a_refusal_says_why(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fairy, "binary_path", lambda: None)
        code, out = self.run(capsys, str(RULESETS / "dull.json"), str(RULESETS / "torus.json"),
                             "--no-results")
        assert code == 0
        assert "no Fairy-Stockfish binary found" in out and "nothing is downloaded" in out
        assert "writes   " in out and "translated; no engine to check it against" in out
        assert "refused  " in out
        assert "0 verified, 0 diverged, 1 refused, of 2 rulesets" in out
        assert "results against" not in out

    def test_without_the_referee_the_results_check_is_reported_as_not_made(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fairy, "binary_path", lambda: None)
        monkeypatch.setattr(cli, "_fairy_referee", lambda: (None, "pyffish is not installed"))
        code, out = self.run(capsys, str(RULESETS / "dull.json"))
        assert code == 0
        assert "results: not checked; pyffish is not installed" in out
        assert "results against" not in out

    def _scripted(self, monkeypatch: pytest.MonkeyPatch, *, moves_ok: bool,
                  results_ok: bool) -> None:
        monkeypatch.setattr(fairy, "binary_path", lambda: "fake-engine")
        monkeypatch.setattr(fairy, "UciEngine", FakeEngine)
        monkeypatch.setattr(fairy, "config_path", lambda variant: "unused.ini")
        monkeypatch.setattr(
            fairy, "verify",
            lambda spec, engine, path, **kw: Verification(
                "h", "dull", ok=moves_ok, depth=kw["depth"], positions=7,
                reasons=[] if moves_ok else ["move lists differ"], fen="the FEN",
            ),
        )
        difference = Difference("stalemate", "a FEN", ["a1a2"], "a draw", "a win")
        monkeypatch.setattr(
            fairy, "check_results",
            lambda spec, referee, games: ResultCheck(
                "h", "dull", ok=results_ok, questions=4, agreed={"checkmate": 1},
                differences=[] if results_ok else [difference],
                reasons=[] if results_ok else ["a reason"],
            ),
        )
        monkeypatch.setattr(cli, "_fairy_referee", lambda: (object(), "pyffish 9"))

    def test_everything_agreeing_is_verified_twice_and_exits_zero(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._scripted(monkeypatch, moves_ok=True, results_ok=True)
        code, out = self.run(capsys, str(RULESETS / "dull.json"), "--depth", "2")
        assert code == 0
        assert "engine: fake-engine" in out and "results: pyffish 9" in out
        assert "verified dull" in out and "verified to depth 2 over 7 positions" in out
        assert "agree at 4 positions; ended by checkmate 1" in out
        assert "1 verified, 0 diverged, 0 refused, of 1 rulesets" in out
        assert "results against pyffish 9: 1 agree, 0 differ, of 1 translations" in out
        assert "1 both move-verified and result-verified" in out

    def test_a_move_divergence_prints_the_position_and_exits_one(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._scripted(monkeypatch, moves_ok=False, results_ok=True)
        code, out = self.run(capsys, str(RULESETS / "dull.json"))
        assert code == 1
        assert "DIVERGED dull" in out and "move lists differ" in out and "the FEN" in out
        assert "0 verified, 1 diverged" in out
        assert "0 both move-verified" in out

    def test_a_results_difference_prints_each_difference_and_exits_one(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._scripted(monkeypatch, moves_ok=True, results_ok=False)
        code, out = self.run(capsys, str(RULESETS / "dull.json"))
        assert code == 1
        assert "RESULTS DIFFER" in out
        assert "stalemate: ours says a draw after a1a2, theirs says a win" in out
        assert "a FEN" in out and "a reason" in out
        assert "0 agree, 1 differ" in out

    @needs_binary
    def test_the_real_engine_and_referee_verify_a_small_ruleset(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("pyffish")
        code, out = self.run(
            capsys, str(RULESETS / "dull.json"), "--depth", "1", "--games", "1",
            "--plies", "6", "--result-games", "1",
        )
        assert code == 0, out
        assert "verified dull" in out
        assert "1 both move-verified and result-verified" in out


class TestFairyReferee:
    def test_is_the_pyffish_backend_and_its_version(self) -> None:
        pyffish = pytest.importorskip("pyffish")
        from chess_variant_sandbox.fairy.verify import PyffishBackend

        referee, library = cli._fairy_referee()
        assert isinstance(referee, PyffishBackend)
        assert library == "pyffish " + ".".join(str(part) for part in pyffish.version())

    def test_without_the_library_there_is_no_referee_and_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(__import__("sys").modules, "pyffish", None)
        assert cli._fairy_referee() == (None, "pyffish is not installed (it is a dev dependency)")
