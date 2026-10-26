"""N7: *the hydra*, the showcase of pounce on prey, pounce on attackers and split
(`docs/plan-npc.md` §6.2).

Chess with a stationary hydra pair on d4 and d5 that takes what attacks it (every type but the
king), growing a head on each attacker's square up to six, and four hounds on three-square wing
loops that take the mover's pieces next to them, the black side's one ply behind the white
side's. This is V1 on the file itself:

- after every ply of seeded random games (3 fast, 20 slow) the core's legal list, check answer,
  board and phase log equal the pounce oracle's (`tests/pounce_oracle.py`), never more than six
  hydra pieces stand, and no phase eats a king;
- hand positions: mirrored positions choosing mirrored victims, a king beside a hydra unharmed
  (and taking it), a head taken at once by a piece covering the square it grew on, and a hound
  off its loop standing, then resuming on a loop it pounced back onto.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from pounce_oracle import PounceOracle

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "archive" / "the-hydra.json"
CAP = 6
KINGS = {"a1": ("K", 0), "h8": ("K", 1)}


def _file() -> dict[str, Any]:
    return copy.deepcopy(load_spec(FILE))


def _engine() -> Engine:
    return Engine(_file())


def _name(square: int) -> str:
    return "abcdefgh"[square % 8] + str(square // 8 + 1)


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, list[tuple[str, str, str]]]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, [(_name(s.origin), _name(s.target), s.kind) for s in phase]


def _wild(engine: Engine, state: GameState, piece: str) -> list[str]:
    kind = engine.spec.piece_ids.index(piece)
    return sorted(
        _name(sq) for sq, v in enumerate(state.board)
        if v and engine.spec.owner_of(v) == 2 and engine.spec.type_of(v) == kind
    )


def test_the_file_is_the_registered_spelling() -> None:
    spec = _file()
    assert validate_spec(spec) == []
    assert spec_hash(spec) == "0f126fe608ef"
    rules = spec["rules"]
    assert rules["win"] == "checkmate" and rules["max_plies"] == 400
    assert rules["repetition_draw"] == 3 and rules["no_progress_draw"] == 100
    assert rules["scoring"] == {"count": "material"}
    hydra, *hounds = rules["npcs"]
    assert hydra == {"piece": "Y", "route": ["d4"], "mirror": True, "pounce": "attackers",
                     "split": CAP, "bounty": 0}
    # The registered loops: the drafted squares, the black side's one ply behind the white side's.
    assert [h["route"] for h in hounds] == [
        ["b4", "a5", "a3"], ["a6", "b5", "a4"], ["g4", "h5", "h3"], ["h6", "g5", "h4"]]
    assert all(h == {"piece": "H", "route": h["route"], "pounce": "prey"} for h in hounds)
    by_id = {p["id"]: p for p in spec["pieces"]}
    assert by_id["Y"]["captures"] == ["P", "N", "B", "R", "Q"]
    assert "captures" not in by_id["H"]


WHITE_SIDE = {"b4", "a5", "a3", "g4", "h5", "h3"}


def test_the_black_side_runs_one_ply_behind() -> None:
    """On a mirrored line, the hounds of the loops by Black's camp stand, after every ply, on the
    mirror of the squares those by White's camp stood on a ply earlier, and each army's camp is
    first reached after its own second move: White's on ply 3, Black's, mirrored, on ply 4."""
    engine = _engine()
    mirror = {f"{f}{r}": f"{f}{9 - r}" for f in "abcdefgh" for r in range(1, 9)}
    state = engine.initial_state()
    seen, logs = [_wild(engine, state, "H")], []
    for text in ("Pd2-d3", "Pd7-d6", "Bc1-d2", "Bc8-d7"):
        state, log = _play(engine, state, text)
        seen.append(_wild(engine, state, "H"))
        logs.append(sorted((a, b) for a, b, kind in log if kind == "pounce"))
    for before, after in zip(seen[:2], seen[1:3], strict=True):
        white = sorted(mirror[s] for s in before if s in WHITE_SIDE)
        assert white == [s for s in after if s not in WHITE_SIDE]
    assert logs[:2] == [[], []]
    assert logs[2] == [("a3", "a2"), ("h3", "g2")]
    assert logs[3] == sorted((mirror[a], mirror[b]) for a, b in logs[2])


def _walk(seed: int, plies: int) -> tuple[int, int]:
    checker = PounceOracle(_file())
    engine = checker.engine
    spec = engine.spec
    royal = {i for i, pid in enumerate(spec.piece_ids) if pid == "K"}
    rng, state = random.Random(seed), engine.initial_state()
    copies = most = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        assert moves == checker.legal_moves(state)
        assert engine.in_check(state) == checker.in_check(state)
        move = rng.choice(moves)
        after, phase = engine.apply_with_phase(state, move)
        assert checker.check(state, move) == after
        assert after.key == engine._compute_hash(after)
        fuzz.every_npc_stands_on_its_own_route(engine, after)
        assert not any(s.eaten and spec.type_of(s.eaten) in royal for s in phase)
        heads = len(_wild(engine, after, "Y"))
        assert heads <= CAP
        copies += sum(1 for s in phase if s.kind == "copy")
        most = max(most, heads)
        state = after
    return copies, most


class TestTheWalk:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_seeded_walk(self, seed: int) -> None:
        _walk(seed, 60)

    @pytest.mark.slow
    def test_the_oracle_agrees_over_twenty_walks(self) -> None:
        results = [_walk(seed, 400) for seed in range(20)]
        assert sum(c for c, _ in results) > 20
        assert max(m for _, m in results) == CAP


class TestHandPositions:
    def test_mirrored_positions_choose_mirrored_victims(self) -> None:
        """A hound off its loop beside two knights of the mover's, equal in value and distance:
        the one nearer the mover's own first rank, read from either seat."""
        engine = _engine()
        white = engine.make_state({**KINGS, "c4": ("H", 2), "b3": ("N", 0), "b5": ("N", 0)})
        _, log = _play(engine, white, "Ka1-a2")
        assert log == [("c4", "b3", "pounce")]
        black = engine.make_state(
            {"a8": ("K", 1), "h1": ("K", 0), "c5": ("H", 2), "b6": ("N", 1), "b4": ("N", 1)},
            side_to_move=1)
        _, log = _play(engine, black, "Ka8-a7")
        assert log == [("c5", "b6", "pounce")]

    def test_a_king_beside_a_hydra_is_unharmed_and_takes_it(self) -> None:
        engine = _engine()
        state = engine.make_state({"e2": ("K", 0), "h8": ("K", 1), "d4": ("Y", 2), "a7": ("P", 1)})
        state, log = _play(engine, state, "Ke2-e3")
        assert log == [] and _wild(engine, state, "Y") == ["d4"]
        state, _ = _play(engine, state, "Pa7-a6")
        state, log = _play(engine, state, "Ke3xd4")
        assert log == [] and _wild(engine, state, "Y") == []

    def test_a_knight_attacking_a_hydra_grows_a_head(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, "g1": ("N", 0), "d4": ("Y", 2)})
        state, log = _play(engine, state, "Ng1-f3")
        assert log == [("d4", "f3", "copy")]
        assert _wild(engine, state, "Y") == ["d4", "f3"]

    def test_a_head_is_taken_at_once_by_a_covering_piece(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, "g1": ("N", 0), "d4": ("Y", 2), "h5": ("B", 1)})
        state, log = _play(engine, state, "Ng1-f3")
        assert log == [("d4", "f3", "copy")]
        state, log = _play(engine, state, "Bh5xf3")
        assert log == [] and _wild(engine, state, "Y") == ["d4"]

    def test_a_hound_off_its_loop_stands(self) -> None:
        engine = _engine()
        state = engine.make_state({**KINGS, "c4": ("H", 2)})
        state, log = _play(engine, state, "Ka1-a2")
        assert log == [] and _wild(engine, state, "H") == ["c4"]

    def test_a_hound_pouncing_onto_a_loop_resumes_it(self) -> None:
        """From c4 (off every loop) onto a knight on b4 (the white side's loop), then on along
        that loop to a5."""
        engine = _engine()
        state = engine.make_state({**KINGS, "c4": ("H", 2), "b4": ("N", 0)})
        state, log = _play(engine, state, "Ka1-a2")
        assert log == [("c4", "b4", "pounce")]
        state, log = _play(engine, state, "Kh8-g8")
        assert log == [("b4", "a5", "step")]
