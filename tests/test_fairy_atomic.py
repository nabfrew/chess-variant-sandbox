"""Atomic's touching kings (Q1219): why `fairy/` refuses a blast beside a royal.

R-P3b0's atomic cell at 1,600 nodes stopped at seed 22 with `MoveMismatch`: Fairy-Stockfish
played `h5g6`, stepping its king beside ours, and our engine did not list the move. Their
blast is their own atomic chess, where a capture whose blast would take the mover's own royal
is illegal, so a royal beside the other royal is never attacked. Our atomic spec says no such
thing: a capture whose blast takes both royals is a move (the legality filter's "the
opponent's royal is gone" branch), and check is geometry. The spec is the game, so the
translation is what is wrong, and `variant.BLAST_BESIDE_A_ROYAL` refuses it by name.

The tests: the refusal (no engine); the position itself, put to `pyffish` with the refusal
switched off, so a later build or spec that closes the gap is noticed; and, as a strict xfail,
a fault in our core the investigation found beside it: a blast that takes both royals is
scored for the second player whoever made it (`docs/bugs.md`, Q1219).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import fairy
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.fairy import position, variant
from chess_variant_sandbox.fairy.variant import BLAST_BESIDE_A_ROYAL, Untranslatable, translate
from chess_variant_sandbox.spec import load_spec

ATOMIC = Path(__file__).resolve().parents[1] / "rulesets" / "atomic.json"


def _atomic() -> dict[str, Any]:
    return load_spec(ATOMIC)


# Seed 22's position, white (the first player) to move: `8/2p4N/1p5k/3B1prK/p2bnP2/3R4/P7/8`.
SEED_22 = {
    "c7": ("P", 1), "h7": ("N", 0), "b6": ("P", 1), "h6": ("K", 1), "d5": ("B", 0),
    "f5": ("P", 1), "g5": ("R", 1), "h5": ("K", 0), "a4": ("P", 1), "d4": ("B", 1),
    "e4": ("N", 1), "f4": ("P", 0), "d3": ("R", 0), "a2": ("P", 0),
}


class TestABlastBesideARoyalIsRefused:
    def test_atomic_is_refused_by_name(self) -> None:
        spec = _atomic()
        with pytest.raises(Untranslatable) as caught:
            translate(spec)
        assert BLAST_BESIDE_A_ROYAL in caught.value.reasons
        assert fairy.supports(spec) is False

    @pytest.mark.parametrize(
        "rules",
        [
            {"win": "royal_capture"},
            {"explode_on_capture": {"radius": 1, "immune": ["P", "K"]}},
        ],
    )
    def test_the_other_shapes_of_a_blast_beside_a_royal_are_refused_too(
        self, rules: dict[str, Any]
    ) -> None:
        """Both diverged on the late check's probe (record, *Atomic's touching kings*)."""
        spec = _atomic()
        spec["rules"].update(rules)
        if rules.get("win") == "royal_capture":
            spec["rules"].pop("castling")
        assert BLAST_BESIDE_A_ROYAL in fairy.why_not(spec)

    def test_a_blast_with_no_royal_is_not_refused_for_it(self) -> None:
        """The refusal names the pair; a blast where nothing is royal is not it."""
        spec = _atomic()
        for piece in spec["pieces"]:
            piece.pop("royal", None)
        spec["rules"]["win"] = "extinction"
        spec["rules"]["extinction"] = {"pieces": ["K"]}
        spec["rules"].pop("castling")
        assert BLAST_BESIDE_A_ROYAL not in fairy.why_not(spec)


class TestThePositionOfSeed22:
    """The refused definition at the position, so the premise is held, not remembered."""

    def test_the_two_engines_part_exactly_where_the_royals_touch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sf = pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
        monkeypatch.setattr(variant, "_check_blast_beside_a_royal", lambda *_: None)
        spec = _atomic()
        written = translate(spec)
        sf.load_variant_config(written.config)
        engine = Engine(spec)
        state = engine.make_state(SEED_22, side_to_move=0)
        fen = position.variant_fen(written, engine.spec, state)
        assert fen.startswith("8/2p4N/1p5k/3B1prK/p2bnP2/3R4/P7/8 w ")
        ours = {
            position.uci_of(engine.spec, state, move, written.chars, written.promotion_suffix)
            for move in engine.legal_moves(state)
        }
        theirs = set(sf.legal_moves(written.name, fen, []))
        # Ours: the king on h5 is attacked by the rook on g5 (geometry), so only a reply to
        # check, and two of the replies are captures whose blast takes both kings.
        assert ours == {"f4g5", "h5h4", "h7g5"}
        # Theirs: the kings touch, so neither is attacked; every move but a capture that
        # would blow up their own king, `h5g6` (the move they played) among them.
        assert "h5g6" in theirs and "h5h4" in theirs
        assert not {"f4g5", "h7g5"} & theirs
        assert len(theirs - ours) == 23


class TestTheLateCheck:
    """`fairy/late.py`, the check that compares the engines past the opening (Q1219)."""

    def test_it_finds_seed_22_in_the_refused_definition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
        from chess_variant_sandbox.fairy.late import check_states
        from chess_variant_sandbox.fairy.verify import PyffishBackend

        monkeypatch.setattr(variant, "_check_blast_beside_a_royal", lambda *_: None)
        spec = _atomic()
        state = Engine(spec).make_state(SEED_22, side_to_move=0)
        compared, _, found, _ = check_states(
            spec, [state], PyffishBackend(), perfts=0, depth=1
        )
        assert compared == 1 and len(found) == 1
        assert found[0].only_ours == ["f4g5", "h7g5"]
        assert "h5g6" in found[0].only_theirs

    def test_a_game_that_translates_agrees_from_late_positions(self) -> None:
        pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
        from chess_variant_sandbox.fairy.late import late_check

        spec = load_spec(ATOMIC.parent / "gardner.json")
        result = late_check(spec, records=None, positions=20, perfts=2)
        assert result.source == "random" and result.positions >= 20
        assert result.verdict == "agrees", result.divergences[:1]

    def test_a_refused_game_is_reported_refused(self) -> None:
        from chess_variant_sandbox.fairy.late import late_check

        result = late_check(_atomic(), records=None)
        assert result.verdict == "refused"
        assert BLAST_BESIDE_A_ROYAL in result.reasons


class TestABlastThatTakesBothRoyals:
    """A capture whose blast takes both royals, made by either player.

    The record's reading (`engine-and-spec-record.md`, *A royal that cannot capture*, Q1032)
    is that it is admitted "as a win for the mover", and the legality filter admits it on that
    premise. The core scored it for the second player whoever made it: the blast ending was
    asked for the first player first, and the first player's royal is gone. So the first
    player's double blast was a legal move that lost, and the second player's one that won.
    Fixed by R-L6c (F6): with both royals gone the side whose move made the position wins
    (`Game::royal_loss_winner`).
    """

    # The first player's pawn on f4 takes the rook on g5; the blast takes h5 and h6.
    WHITE_TAKES = {"h5": ("K", 0), "f4": ("P", 0), "h6": ("K", 1), "g5": ("R", 1)}
    # The same, mirrored for the second player: its pawn on f5 takes the rook on g4, and the
    # blast takes both kings, on h3 and h4.
    BLACK_TAKES = {"h3": ("K", 0), "g4": ("R", 0), "h4": ("K", 1), "f5": ("P", 1)}

    def _after(self, pieces: dict[str, tuple[str, int]], side: int, uci: str) -> Any:
        engine = Engine(_atomic())
        state = engine.make_state(pieces, side_to_move=side)
        chars = tuple(engine.spec.piece_ids)
        moves = {
            position.uci_of(engine.spec, state, move, chars, ""): move
            for move in engine.legal_moves(state)
        }
        assert uci in moves, sorted(moves)
        return engine.terminal(engine.apply(state, moves[uci]))

    def test_the_second_players_double_blast_wins_for_the_mover(self) -> None:
        outcome = self._after(self.BLACK_TAKES, 1, "f5g4")
        assert outcome is not None and outcome.reason == "royal_destroyed"
        assert outcome.winner == 1

    def test_the_first_players_double_blast_wins_for_the_mover(self) -> None:
        outcome = self._after(self.WHITE_TAKES, 0, "f4g5")
        assert outcome is not None and outcome.reason == "royal_destroyed"
        assert outcome.winner == 0
