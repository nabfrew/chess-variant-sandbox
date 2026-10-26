"""The duck in Fairy-Stockfish (generation 42, N4c, `docs/plan-npc.md` §4.6).

The classic duck (`goes: "anywhere"`, no `start`, no sight, no wild pieces) crosses as
`wallingRule = duck`; every other duck is refused by name. A move there is both halves of our
turn (`position.turn_word`), so `verify` compares at turn starts, and the opening perft holds
our perft(2k) to their perft(k). The moves are verified against `pyffish` 0.0.90, which has
the duck; the installed 2021 binary does not read `wallingRule`, has no `UNREAD` entry for it,
and so offers no opponent for a duck game, named.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.fairy import position
from chess_variant_sandbox.fairy.unread import UNREAD
from chess_variant_sandbox.fairy.unread import reasons as unread_reasons
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

ROOT = Path(__file__).resolve().parents[1]
DUCK = json.loads((ROOT / "tests" / "_duck_chess.json").read_text(encoding="utf-8"))
SHEPHERD = json.loads((ROOT / "tests" / "_duck_shepherd.json").read_text(encoding="utf-8"))


def _pyffish() -> Any:
    return pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")


def _with_duck(**duck: Any) -> dict[str, Any]:
    spec = copy.deepcopy(DUCK)
    spec["rules"]["duck"].update(duck)
    return spec


def _refusals(spec: dict[str, Any]) -> list[str]:
    with pytest.raises(Untranslatable) as refused:
        translate(spec)
    return refused.value.reasons


class TestTheTranslation:
    def test_the_classic_duck_is_a_walling_rule_and_takes_no_piece_slot(self) -> None:
        variant = translate(DUCK)
        lines = variant.config.splitlines()
        assert "wallingRule = duck" in lines
        assert not any(line.startswith("customPiece") for line in lines)
        assert variant.royalty == "extinct"

    def test_every_other_duck_is_refused_by_name(self) -> None:
        walker = _with_duck(goes="by_moves")
        duck = next(p for p in walker["pieces"] if p["id"] == "D")
        duck["moves"] = [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
                          "mode": "move"}]
        assert any('"by_moves"' in r for r in _refusals(walker))
        assert any("rules.duck.start" in r for r in _refusals(_with_duck(start="d4")))
        said = _refusals(SHEPHERD)
        assert any("rules.duck beside rules.npcs" in r for r in said)
        # Not the long turn's refusal: the duck's two-move turn is the duck's half.
        assert not any(r.startswith("rules.turn") for r in _refusals(_with_duck(start="d4")))

    def test_the_duck_is_a_wall_in_the_fen_and_a_turn_one_move(self) -> None:
        engine = Engine(DUCK)
        variant = translate(DUCK)
        state = engine.initial_state()
        e4 = next(m for m in engine.legal_moves(state)
                  if engine.describe_move(state, m) == "Pe2-e4")
        half = engine.apply(state, e4)
        d5 = next(m for m in engine.legal_moves(half)
                  if engine.describe_move(half, m) == "the duck lands on d5")
        word = position.turn_word(engine.spec, state, e4, d5, variant.chars)
        assert word == "e2e4,e4d5"
        assert position.turn_word(engine.spec, state, e4, None, variant.chars) == "e2e4"
        turned = engine.apply(half, d5)
        fen = position.variant_fen(variant, engine.spec, turned)
        assert fen.split()[0] == "rnbqkbnr/pppppppp/8/3*4/4P3/8/PPPP1PPP/RNBQKBNR"
        assert fen.split()[1] == "b" and fen.split()[-1] == "1"


class TestTheInstalledBinary:
    def test_it_has_no_entry_for_the_walling_rule_and_says_why_by_name(self) -> None:
        assert "wallingRule" not in UNREAD
        said = unread_reasons(translate(DUCK).config, {"wallingRule"})
        assert len(said) == 1 and "has no duck" in said[0]
        assert "no Fairy-Stockfish opponent" in said[0]


class TestAgainstPyffish:
    def test_the_moves_agree_at_turn_starts(self) -> None:
        _pyffish()
        result = verify(DUCK, PyffishBackend(), "", depth=2, games=1, plies=24)
        assert result.ok, result.reasons
        # perft(2) = 640 here and perft(1) there, then the walk's turn starts.
        assert result.positions > 640

    def test_a_definition_without_the_walling_rule_diverges(self) -> None:
        _pyffish()
        variant = translate(DUCK)
        broken = dataclasses.replace(
            variant,
            name=variant.name + "nowall",
            config=variant.config.replace("wallingRule = duck\n", "").replace(
                f"[{variant.name}]", f"[{variant.name}nowall]"),
        )
        backend = PyffishBackend()
        backend.load(broken, "")
        state = Engine(DUCK).initial_state()
        assert backend.perft(position.variant_fen(broken, Engine(DUCK).spec, state), 1) == 20

    @pytest.mark.slow
    def test_the_moves_agree_to_depth_three_over_three_long_games(self) -> None:
        _pyffish()
        result = verify(DUCK, PyffishBackend(), "", depth=3, games=3, plies=400)
        assert result.ok, result.reasons

    @pytest.mark.slow
    def test_the_results_agree(self) -> None:
        _pyffish()
        from chess_variant_sandbox.fairy.results import check_results

        check = check_results(DUCK, PyffishBackend(), "")
        assert check.ok, [d for d in check.differences]
        assert check.agreed.get("royal_captured", 0) > 0
