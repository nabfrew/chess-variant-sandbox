"""The goal square as the installed Fairy-Stockfish reads it (Q915).

Q912 found the 2021 build blind to a translated `rules.reach`: it reads the goal only as
``whiteFlag``/``blackFlag`` and dropped ``flagRegionWhite``/``flagRegionBlack`` in
silence, so one move from the goal it played on. `verify` compares moves and the result
check asks `pyffish`, so neither saw it. These tests ask the binary itself, and hold the
check that offers it (`fairy.checked`) to the build's own account of what it read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_variant_sandbox import fairy
from chess_variant_sandbox.fairy import unread
from chess_variant_sandbox.fairy.variant import translate
from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# One move from the goal, with the goal move Fairy-Stockfish's own built-in variant plays
# here as mate in one (Q912's record, §4). Under the old translation the binary played
# `e1c2` and `b2b3`.
ONE_FROM_THE_GOAL = (
    ("racingkings.json", "8/6K1/8/8/8/8/krbn1BR1/qrbnNBRQ w - - 0 1", "g7f8"),
    ("kingofthehill.json", "rnbqkbnr/pppppppp/8/8/8/2K5/PP1PPPPP/RNBQ1BNR w kq - 0 1", "c3d4"),
)

needs_binary = pytest.mark.skipif(
    not fairy.available(),
    reason="no Fairy-Stockfish binary is installed; these tests ask the binary, not pyffish",
)


def _written(spec_name: str) -> dict[str, str]:
    return unread.options(translate(load_spec(RULESETS / spec_name)).config)


@pytest.mark.parametrize("name", ["racingkings.json", "kingofthehill.json", "dobutsu.json"])
def test_the_goal_is_written_in_both_spellings(name: str) -> None:
    written = _written(name)
    assert written["whiteFlag"] == written["flagRegionWhite"] != "-"
    assert written["blackFlag"] == written["flagRegionBlack"] != "-"


class TestTheBuildsOwnAccount:
    """`unread.reasons`: a key the build skips needs a reason, under its condition."""

    def test_check_output_is_read(self) -> None:
        output = (
            "Fairy-Stockfish 14 LB by Fabian Fichter\n"
            "Parsing variant: x\n"
            "enPassantRegion - Invalid value - for type Bitboard\n"
            "Invalid option: flagRegionWhite\n"
        )
        assert unread.parse_check(output) == {"enPassantRegion", "flagRegionWhite"}
        assert unread.parse_check("Unknown command: check\n") is None
        assert unread.parse_check("Parsing variant: x\nsomething new\n") == {"?something new"}

    def test_a_newer_spelling_needs_its_older_one_beside_it(self) -> None:
        both = "flagRegionWhite = d4\nwhiteFlag = d4\n"
        assert unread.reasons(both, {"flagRegionWhite"}) == []
        alone = "flagRegionWhite = d4\n"
        assert unread.reasons(alone, {"flagRegionWhite"}) == [
            "the installed Fairy-Stockfish does not read flagRegionWhite = d4, "
            "so it would play another game"
        ]

    def test_a_key_with_no_reason_refuses(self) -> None:
        assert unread.reasons("flagPieceSafe = true\n", {"flagPieceSafe"})
        assert unread.reasons("x = 1\n", {"?a warning nobody has read"})

    def test_a_condition_is_not_widened(self) -> None:
        stepping = "doubleStep = true\nenPassantRegion = -\n"
        assert unread.reasons(stepping, {"enPassantRegion"})
        assert unread.reasons("doubleStep = false\n", {"enPassantRegion"}) == []
        assert unread.reasons("pawn = p\nblastImmuneTypes = N\n", {"blastImmuneTypes"})
        assert unread.reasons("pawn = p\nblastImmuneTypes = P\n", {"blastImmuneTypes"}) == []


@needs_binary
@pytest.mark.parametrize(("name", "fen", "goal"), ONE_FROM_THE_GOAL)
def test_the_binary_plays_the_goal_move(name: str, fen: str, goal: str) -> None:
    """The binary itself, under our translation, finds the arrival as mate in one."""
    variant = translate(load_spec(RULESETS / name))
    path = fairy.config_path(variant)
    with fairy.UciEngine() as engine:
        engine.load(variant, path)
        engine.new_game()
        assert engine.best_move(fen, 6400) == goal
        found = engine.unread_options(path)
    assert found is not None
    assert {"whiteFlag", "blackFlag", "flagPiece"}.isdisjoint(found)


@needs_binary
@pytest.mark.slow  # two binary verifications and a refusal, about 5 s
def test_the_check_and_the_binary_agree_on_the_goal() -> None:
    """`fairy.checked` offers both games where the binary reads every goal key, and
    turns dobutsu away by naming the goal rule the 2021 build has no word for."""
    for name, _, _ in ONE_FROM_THE_GOAL:
        spec = load_spec(RULESETS / name)
        variant = translate(spec)
        with fairy.UciEngine() as engine:
            found = engine.unread_options(fairy.config_path(variant))
        assert found is not None
        assert unread.reasons(variant.config, found) == [], name
        assert fairy.checked(spec) is not None, name
    dobutsu = load_spec(RULESETS / "dobutsu.json")
    with fairy.UciEngine() as engine:
        found = engine.unread_options(fairy.config_path(translate(dobutsu)))
    if found is not None and "flagPieceSafe" in found:
        said = fairy.verdict(dobutsu)
        assert said is not None and not said.ok
        assert any("flagPieceSafe" in reason for reason in said.reasons)
