"""Wild pieces that see (generation 37, N2, `docs/archive/plan-npc.md` §4.2).

A wild type may carry `pieces[].sight` with ``whose: "any"``: freeze, disarm or shield, on both
players' pieces alike, read from the board as a player's sight is. A shield beside `rules.npcs`
is allowed, with the rule the old refusal asked for: **a wild piece never takes a shielded
piece** (a step onto one waits, a return holds back from one, and a shielded royal on a wild
piece's next square is not in check from it). The shields a phase reads are the board's at the
phase's start, for every wild piece in it (§3.1).

Hand positions on a 6x6 board, the refusals, the inert spellings, the analysis column and the
words. The oracle walks are `tests/test_sight_walks.py`'s (`_sight_wild_wolves`,
`_sight_wild_small`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import SIGHT_FREEZE, SIGHT_SHIELD, compile_spec
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec

ROOT = Path(__file__).resolve().parents[1]
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
RING = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1, "through": True,
        "whose": "any"}


def _spec(
    npcs: list[dict[str, Any]],
    wolf_sight: str | None = None,
    win: str = "checkmate",
    warden: bool = True,
    hound_sight: str | None = None,
) -> dict:
    """Kings, a rook and a knight each on 6x6; a warden (``G``) that shields its own side's
    pieces beside it; the wolf ``W`` (taking a king, a rook or a knight, never a warden) with
    a ring of ``wolf_sight`` if given, and a hound ``H`` (a second wild type) with a ring of
    ``hound_sight``."""
    pieces: list[dict[str, Any]] = [
        {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING},
        {"id": "R", "name": "rook", "value": 5,
         "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        {"id": "N", "name": "knight", "value": 3,
         "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
        {"id": "W", "name": "wolf", "value": 3, "moves": KING, "captures": ["K", "R", "N"]},
        {"id": "H", "name": "hound", "value": 2, "moves": KING, "captures": ["K", "R", "N"]},
    ]
    if warden:
        pieces.insert(3, {"id": "G", "name": "warden", "value": 2, "moves": KING,
                          "sight": [dict(RING, effect="shield", whose="own")]})
    if wolf_sight:
        pieces[-2]["sight"] = [dict(RING, effect=wolf_sight)]
    if hound_sight:
        pieces[-1]["sight"] = [dict(RING, effect=hound_sight)]
    return {
        "schema_version": 31,
        "name": "wild sight hand positions",
        "board": {"width": 6, "height": 6},
        "pieces": pieces,
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "R", "player": 0, "squares": ["b1"]},
            {"piece": "R", "player": 1, "squares": ["e6"]},
            {"piece": "N", "player": 0, "squares": ["c1"]},
            {"piece": "N", "player": 1, "squares": ["d6"]},
        ] + ([
            {"piece": "G", "player": 0, "squares": ["d1"]},
            {"piece": "G", "player": 1, "squares": ["c6"]},
        ] if warden else [])},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 200, "npcs": npcs},
    }


def _engine(npcs: list[dict[str, Any]], **kwargs: Any) -> Engine:
    spec = _spec(npcs, **kwargs)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _at(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[engine._square_index(square)]
    if not value:
        return None
    spec = engine.spec
    owner, rest = divmod(value - 1, spec.n_facings * spec.n_types)
    return spec.piece_ids[rest % spec.n_types], owner


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, tuple]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, phase


class TestAWolfsFreeze:
    def test_a_wolfs_freeze_lifts_a_check(self) -> None:
        """Black's rook on a6 gives check down the a-file; a wolf standing on b5 freezes it,
        so white is not in check, and the king may walk up the file. Without the freeze it
        is check, and a2 is barred."""
        place = {"a1": ("K", 0), "f6": ("K", 1), "a6": ("R", 1), "b5": ("W", 2)}
        frozen = _engine([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze")
        state = frozen.make_state(place)
        assert not frozen.in_check(state)
        assert frozen.effects(state).bits[frozen._square_index("a6")] == SIGHT_FREEZE
        assert "Ka1-a2" in frozen.move_texts(state)
        plain = _engine([{"piece": "W", "route": ["b5"]}])
        state = plain.make_state(place)
        assert plain.in_check(state)
        assert "Ka1-a2" not in plain.move_texts(state)

    def test_the_freeze_walks_with_the_wolf(self) -> None:
        """The wolf steps b5 to c5 after white's move, and the rook it froze is free again on
        the board the move leaves, which is the board legality asks about: so white is not in
        check now, but may not stay on the a-file."""
        engine = _engine([{"piece": "W", "route": ["b5", "c5"]}], wolf_sight="freeze")
        state = engine.make_state(
            {"a1": ("K", 0), "f6": ("K", 1), "a6": ("R", 1), "b5": ("W", 2)}
        )
        assert not engine.in_check(state)
        texts = engine.move_texts(state)
        assert "Ka1-a2" not in texts and "Ka1-b1" in texts and "Ka1-b2" in texts
        state, _ = _play(engine, state, "Ka1-b1")
        assert _at(engine, state, "c5") == ("W", 2)
        assert engine.effects(state).bits[engine._square_index("a6")] == 0

    def test_a_disarm_on_a_checker_lifts_the_check_too(self) -> None:
        engine = _engine([{"piece": "W", "route": ["b5"]}], wolf_sight="disarm")
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "a6": ("R", 1), "b5": ("W", 2)})
        assert not engine.in_check(state)
        # Disarmed, the rook still moves; only its captures are gone.
        assert any(text.startswith("Ra6-") for text in engine.move_texts(state.with_(side_to_move=1)))


class TestAWolfHeldByAShield:
    def test_a_wolf_waits_at_a_shielded_piece(self) -> None:
        """The wolf walks a3, b3, c3. White's knight on b3 stands beside its warden on c2, so the
        wolf waits on a3 rather than eat it; with the warden elsewhere, the knight is eaten."""
        npcs = [{"piece": "W", "route": ["a3", "b3", "c3"]}]
        engine = _engine(npcs)
        state = engine.make_state({
            "f1": ("K", 0), "f6": ("K", 1), "b3": ("N", 0), "c2": ("G", 0), "a3": ("W", 2),
        })
        state, phase = _play(engine, state, "Kf1-e1")
        assert phase == () and _at(engine, state, "a3") == ("W", 2)
        assert _at(engine, state, "b3") == ("N", 0)
        bare = engine.make_state({
            "f1": ("K", 0), "f6": ("K", 1), "b3": ("N", 0), "e2": ("G", 0), "a3": ("W", 2),
        })
        bare, phase = _play(engine, bare, "Kf1-e1")
        assert len(phase) == 1 and _at(engine, bare, "b3") == ("W", 2)

    def test_a_shielded_royal_on_the_wolfs_next_square_is_not_in_check(self) -> None:
        npcs = [{"piece": "W", "route": ["a3", "b3", "c3"]}]
        engine = _engine(npcs)
        shielded = engine.make_state({
            "b3": ("K", 0), "f6": ("K", 1), "c2": ("G", 0), "a3": ("W", 2),
        })
        assert not engine.in_check(shielded)
        bare = engine.make_state({"b3": ("K", 0), "f6": ("K", 1), "e1": ("G", 0), "a3": ("W", 2)})
        assert engine.in_check(bare)

    def test_a_king_may_step_onto_the_wolfs_next_square_under_a_shield(self) -> None:
        npcs = [{"piece": "W", "route": ["a3", "b3", "c3"]}]
        engine = _engine(npcs)
        state = engine.make_state({
            "b2": ("K", 0), "f6": ("K", 1), "c2": ("G", 0), "a3": ("W", 2),
        })
        texts = engine.move_texts(state)
        assert "Kb2-b3" in texts
        # Without the warden beside b3 the step is into the wolf's next square: refused.
        bare = engine.make_state({"b2": ("K", 0), "f6": ("K", 1), "e1": ("G", 0), "a3": ("W", 2)})
        assert "Kb2-b3" not in engine.move_texts(bare)

    def test_the_shields_are_read_at_the_phases_start(self) -> None:
        """A hound carrying a shield steps c2 to d2 first (the lower square), leaving the knight
        on b3 bare; the wolf, stepping after it in the same phase, still waits, since the
        knight was shielded when the phase began (§3.1). It is eaten in the next phase."""
        npcs = [
            {"piece": "W", "route": ["a3", "b3", "c3"]},
            {"piece": "H", "route": ["c2", "d2"]},
        ]
        engine = _engine(npcs, warden=False, hound_sight="shield")
        state = engine.make_state({
            "f1": ("K", 0), "f6": ("K", 1), "b3": ("N", 0), "a3": ("W", 2), "c2": ("H", 2),
        })
        state, phase = _play(engine, state, "Kf1-e1")
        assert _at(engine, state, "d2") == ("H", 2)
        assert _at(engine, state, "a3") == ("W", 2) and _at(engine, state, "b3") == ("N", 0)
        assert [(step.origin, step.target) for step in phase] == [
            (engine._square_index("c2"), engine._square_index("d2")),
        ]
        state, phase = _play(engine, state, "Kf6-f5")
        assert _at(engine, state, "b3") == ("W", 2)

    def test_a_return_holds_back_from_a_shielded_piece(self) -> None:
        """Taken on c3, the wolf is due back on a3 in the same phase (``respawn: 1``); white's
        knight there is beside its warden on b2, so the return holds. With the warden away the
        knight is eaten."""
        npcs = [{"piece": "W", "route": ["a3", "b3", "c3"], "respawn": 1}]
        engine = _engine(npcs)
        state = engine.make_state({
            "f1": ("K", 0), "f6": ("K", 1), "c1": ("R", 0), "c3": ("W", 2), "a3": ("N", 0),
            "b2": ("G", 0),
        })
        state, phase = _play(engine, state, "Rc1xc3")
        assert [step for step in phase if step.kind == "return"] == []
        assert _at(engine, state, "a3") == ("N", 0) and state.respawn == (1,)
        bare = engine.make_state({
            "f1": ("K", 0), "f6": ("K", 1), "c1": ("R", 0), "c3": ("W", 2), "a3": ("N", 0),
            "e2": ("G", 0),
        })
        bare, phase = _play(engine, bare, "Rc1xc3")
        assert _at(engine, bare, "a3") == ("W", 2)

    def test_a_wolfs_own_shield_bars_a_players_capture(self) -> None:
        engine = _engine([{"piece": "W", "route": ["c3"]}], wolf_sight="shield")
        state = engine.make_state({
            "a1": ("K", 0), "f6": ("K", 1), "d1": ("R", 0), "d3": ("N", 1),
        } | {"c3": ("W", 2)})
        assert "Rd1xd3" not in engine.move_texts(state)
        assert engine.effects(state).bits[engine._square_index("d3")] == SIGHT_SHIELD


def _mirror(name: str) -> str:
    return f"{name[0]}{7 - int(name[1:])}"


class TestAMirroredAura:
    def test_an_aura_reads_alike_for_both_seats(self) -> None:
        """A shielding wolf on c3 and its mirror on c4, a knight of each side beside each and a
        rook of the other side lined up on it: each seat's capture of the shielded knight is
        barred, and the mirrored position (ranks flipped, colours swapped, the other side to
        move) gives the mirrored list and the mirrored effects."""
        engine = _engine([{"piece": "W", "route": ["c3"], "mirror": True}], wolf_sight="shield")
        place = {
            "a1": ("K", 0), "f6": ("K", 1), "c3": ("W", 2), "c4": ("W", 2),
            "d3": ("N", 1), "d1": ("R", 0), "b4": ("N", 0), "b6": ("R", 1), "e2": ("N", 0),
        }
        mirrored = {
            _mirror(square): (piece, owner if owner == 2 else 1 - owner)
            for square, (piece, owner) in place.items()
        }
        state = engine.make_state(place)
        image = engine.make_state(mirrored, side_to_move=1)
        texts = engine.move_texts(state)
        assert "Rd1xd3" not in texts and "Rd1-d2" in texts
        flip = {
            text: text
            for text in texts
        }
        assert len(flip) == len(texts)

        def reflect(text: str) -> str:
            return "".join(
                str(7 - int(char)) if char.isdigit() else char for char in text
            )

        assert sorted(reflect(text) for text in texts) == sorted(engine.move_texts(image))
        bits = engine.effects(state).bits
        image_bits = engine.effects(image).bits
        for square in place:
            index = engine._square_index(square)
            assert bits[index] == image_bits[engine._square_index(_mirror(square))], square
        assert bits[engine._square_index("d3")] == SIGHT_SHIELD
        assert bits[engine._square_index("b4")] == SIGHT_SHIELD
        assert "Rb6xb4" not in engine.move_texts(state.with_(side_to_move=1))


class TestRefusals:
    @pytest.mark.parametrize("whose", ["enemy", "own"])
    def test_a_wild_sight_must_touch_both_sides(self, whose: str) -> None:
        spec = _spec([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze")
        spec["pieces"][4]["sight"][0]["whose"] = whose
        assert any("which has no side; use 'any'" in e for e in validate_spec(spec))

    def test_a_command_on_a_wild_piece(self) -> None:
        spec = _spec([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze")
        spec["pieces"][4]["sight"][0]["effect"] = "command"
        assert any("nobody commands for it" in e for e in validate_spec(spec))

    def test_a_wild_sight_that_reads_differently_for_the_two_players(self) -> None:
        spec = _spec([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze")
        spec["pieces"][4]["sight"] = [
            {"vectors": [[0, 1]], "range": 2, "effect": "freeze", "whose": "any"}
        ]
        assert any("reads differently for the two players" in e for e in validate_spec(spec))
        # Forward and back is its own mirror image.
        spec["pieces"][4]["sight"][0]["vectors"] = [[0, 1], [0, -1]]
        assert validate_spec(spec) == []

    def test_an_affects_naming_a_wild_type_stays_refused(self) -> None:
        spec = _spec([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze")
        spec["pieces"][4]["sight"][0]["affects"] = ["W"]
        assert any("a wild piece (rules.npcs), which nothing affects" in e
                   for e in validate_spec(spec))

    def test_a_shield_beside_wild_pieces_is_allowed(self) -> None:
        assert validate_spec(_spec([{"piece": "W", "route": ["a3", "b3"]}])) == []


class TestInert:
    def test_a_game_without_a_seeing_wild_piece_compiles_as_before(self) -> None:
        """No wild type sees: the forward lines keep the players' frames only."""
        compiled = compile_spec(_spec([{"piece": "W", "route": ["a3", "b3"]}]))
        assert compiled.sight is not None and len(compiled.sight.lines) == compiled.n_frames
        seeing = compile_spec(_spec([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze"))
        assert seeing.sight is not None and len(seeing.sight.lines) == seeing.n_frames + 1

    @pytest.mark.parametrize("name", ["wolves", "wolves-patrol", "wolves-bounty", "madrasi"])
    def test_the_stored_games_keep_their_names(self, name: str) -> None:
        spec = json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))
        assert validate_spec(spec) == []
        assert spec_hash(spec) == spec_hash(copy.deepcopy(spec))


class TestWordsAndTallies:
    def test_the_words_say_the_wild_sight_and_the_shield_rule(self) -> None:
        text = describe_spec(_spec([{"piece": "W", "route": ["a3", "b3"]}], wolf_sight="freeze"))
        assert "never takes a shielded piece" in text
        assert "wolf" in text and "freezes" in text

    def test_the_analysis_counts_the_wild_owner(self) -> None:
        from chess_variant_sandbox.tournament.analysis import (
            WILD_SIGHT_SERIES,
            count_sight,
            sight_tally,
        )

        engine = _engine([{"piece": "W", "route": ["b5"]}], wolf_sight="freeze")
        tally = sight_tally(engine)
        assert tally is not None and list(tally)[-3:] == list(WILD_SIGHT_SERIES)
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "a6": ("R", 1), "b5": ("W", 2)})
        count_sight(engine, tally, state, 1)
        assert tally["wild_frozen"] == [1] and tally["frozen"] == [1]
        count_sight(engine, tally, state, 0)
        assert tally["wild_frozen"] == [1, 0]
        plain = _engine([{"piece": "W", "route": ["b5"]}])
        assert "wild_frozen" not in (sight_tally(plain) or {})
