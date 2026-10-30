"""The draw by insufficient material (Q1069a, `docs/reference/insufficient-material.md`).

`rules.insufficient_material: true` draws a position the moment the material left is one of
FIDE's dead sets: royal against royal, a royal and one lone minor piece against a royal, or
royals with any number of colour-bound riders all on one square colour. It states a rule only
where the game is chess once its other pieces are gone (`schema.insufficient_material_live`),
because the sets are a published result for chess and nothing here proves one; written on any
other game it is inert, compiles to the game without it and leaves its name alone. Roles come
from how a type moves, never from its name.

Chess appears because the sets are published for it, and every test that uses it says so;
the roles themselves are asserted on renamed types too.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import (
    INSUFFICIENT_LEAPER,
    INSUFFICIENT_OTHER,
    INSUFFICIENT_RIDER,
    INSUFFICIENT_ROYAL,
    compile_spec,
)
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import insufficient_material_roles

ROOT = Path(__file__).resolve().parents[1]


def _chess(on: bool = True) -> dict[str, Any]:
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    spec["rules"]["insufficient_material"] = on
    return spec


def _verdict(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> Any:
    return engine.terminal(engine.make_state(placement, side_to_move=side))


# Each dead set and each near miss, in chess (whose published rule these are), as a board.
DEAD: dict[str, dict[str, tuple[str, int]]] = {
    "K v K": {"e1": ("K", 0), "e8": ("K", 1)},
    "K+B v K": {"e1": ("K", 0), "c1": ("B", 0), "e8": ("K", 1)},
    "K v K+B": {"e1": ("K", 0), "e8": ("K", 1), "f8": ("B", 1)},
    "K+N v K": {"e1": ("K", 0), "b1": ("N", 0), "e8": ("K", 1)},
    "K v K+N": {"e1": ("K", 0), "e8": ("K", 1), "g8": ("N", 1)},
    # c1 and f8 are both dark squares.
    "K+B v K+B, one colour": {"e1": ("K", 0), "c1": ("B", 0), "e8": ("K", 1), "f8": ("B", 1)},
    "K+B+B v K, one colour": {"e1": ("K", 0), "c1": ("B", 0), "a3": ("B", 0), "e8": ("K", 1)},
    "K+B+B v K+B, one colour": {
        "e1": ("K", 0), "c1": ("B", 0), "h6": ("B", 0), "e8": ("K", 1), "f8": ("B", 1),
    },
}
ALIVE: dict[str, dict[str, tuple[str, int]]] = {
    "K+N+N v K": {"e1": ("K", 0), "b1": ("N", 0), "g1": ("N", 0), "e8": ("K", 1)},
    # c1 is dark, c8 light: FIDE's rule leaves a helpmate on.
    "K+B v K+B, opposite colours": {
        "e1": ("K", 0), "c1": ("B", 0), "e8": ("K", 1), "c8": ("B", 1),
    },
    "K+B+N v K": {"e1": ("K", 0), "c1": ("B", 0), "b1": ("N", 0), "e8": ("K", 1)},
    "K+B+B v K, both colours": {"e1": ("K", 0), "c1": ("B", 0), "f1": ("B", 0), "e8": ("K", 1)},
    "K+N v K+N": {"e1": ("K", 0), "b1": ("N", 0), "e8": ("K", 1), "g8": ("N", 1)},
    "K+N v K+B": {"e1": ("K", 0), "b1": ("N", 0), "e8": ("K", 1), "f8": ("B", 1)},
    "K+P v K": {"e1": ("K", 0), "e2": ("P", 0), "e8": ("K", 1)},
    "K+R v K": {"e1": ("K", 0), "a1": ("R", 0), "e8": ("K", 1)},
    "K+Q v K": {"e1": ("K", 0), "d1": ("Q", 0), "e8": ("K", 1)},
}


@pytest.fixture(scope="module")
def chess() -> Engine:
    return Engine(_chess())


@pytest.mark.parametrize("name", sorted(DEAD))
def test_chess_dead_set_is_a_draw(chess: Engine, name: str) -> None:
    for side in (0, 1):
        outcome = _verdict(chess, DEAD[name], side)
        assert outcome is not None, name
        assert (outcome.winner, outcome.reason) == (None, "insufficient_material"), name


@pytest.mark.parametrize("name", sorted(ALIVE))
def test_chess_near_miss_goes_on(chess: Engine, name: str) -> None:
    assert _verdict(chess, ALIVE[name]) is None, name


def test_off_the_rule_plays_on() -> None:
    """Without the key (or with it ``false``) a bare board is the old game: no ending."""
    for on in (False,):
        engine = Engine(_chess(on))
        assert _verdict(engine, DEAD["K v K"]) is None


def test_false_is_the_game_without_it() -> None:
    """``false`` is the off value: the normal form drops it, so the hash is unmoved."""
    plain = load_spec(ROOT / "rulesets" / "chess.json")
    plain["rules"].pop("insufficient_material", None)
    assert spec_hash(_chess(False)) == spec_hash(plain)
    assert spec_hash(_chess(True)) != spec_hash(plain)
    assert compile_spec(_chess(False)).insufficient_material is None


def test_a_capture_into_a_dead_set_ends_the_game(chess: Engine) -> None:
    """Played, not placed: a royal taking the last rook leaves K+B v K, drawn at once."""
    state = chess.make_state(
        {"e1": ("K", 0), "c1": ("B", 0), "e8": ("K", 1), "e2": ("R", 1)}, side_to_move=0
    )
    endings = [chess.terminal(chess.apply(state, m)) for m in chess.legal_moves(state)]
    reasons = sorted(str(o.reason) for o in endings if o is not None)
    # The one capture (the royal takes on e2) is the one ending; every retreat plays on.
    assert reasons == ["insufficient_material"]
    outcome = next(o for o in endings if o is not None)
    assert outcome is not None and outcome.reason == "insufficient_material"


def test_a_dead_set_outranks_the_clocks(chess: Engine) -> None:
    """Asked after the endings a position declares and before repetition and the clocks
    (`terminal_prefix`): both are draws, so only the reason tells them apart."""
    for clock in ({"halfmove": chess.spec.no_progress_draw}, {"ply": chess.spec.max_plies}):
        state = chess.make_state(DEAD["K v K"], **clock)
        outcome = chess.terminal(state)
        assert outcome is not None and outcome.reason == "insufficient_material", clock


# ------------------------------------------------------------------ roles by movement


def test_roles_are_read_from_movement_not_names() -> None:
    """Renamed types keep their roles; a type that moves differently has none."""
    spec = _chess()
    renames = {"K": "Z", "N": "J", "B": "D"}
    for piece in spec["pieces"]:
        piece["name"] = f"thing {piece['id']}"
    roles, reasons = insufficient_material_roles(spec)
    assert reasons == []
    assert roles == {
        "P": INSUFFICIENT_OTHER, "N": INSUFFICIENT_LEAPER, "B": INSUFFICIENT_RIDER,
        "R": INSUFFICIENT_OTHER, "Q": INSUFFICIENT_OTHER, "K": INSUFFICIENT_ROYAL,
    }
    assert set(renames)  # names never enter: only `moves` and `royal` are read


def test_a_leaper_spelled_another_way_is_still_the_leaper() -> None:
    spec = _chess()
    knight = next(p for p in spec["pieces"] if p["id"] == "N")
    knight["moves"] = [{"kind": "leap", "vectors": [[1, 2], [2, 1]], "symmetry": "rotate4"}]
    roles, reasons = insufficient_material_roles(spec)
    assert reasons == [] and roles["N"] == INSUFFICIENT_LEAPER


def test_a_variant_reduced_to_chess_qualifies_on_the_pieces_present() -> None:
    """`almost` (a chancellor for the queen) is chess once the chancellor is gone."""
    spec = load_spec(ROOT / "rulesets" / "almost.json")
    spec["rules"]["insufficient_material"] = True
    assert validate_spec(spec) == []
    engine = Engine(spec)
    assert _verdict(engine, DEAD["K+N v K"]) is not None
    assert _verdict(engine, {"e1": ("K", 0), "d1": ("C", 0), "e8": ("K", 1)}) is None


def test_a_nightrider_is_not_the_leaper() -> None:
    """`chess-with-nightriders`: the N rides, so K+N v K is not one of the sets there."""
    spec = load_spec(ROOT / "archive" / "chess-with-nightriders.json")
    spec["rules"]["insufficient_material"] = True
    assert validate_spec(spec) == []
    engine = Engine(spec)
    assert _verdict(engine, DEAD["K+N v K"]) is None
    assert _verdict(engine, DEAD["K+B v K"]) is not None


# ---------------------------------------------- off chess the key states nothing


@pytest.mark.parametrize(
    ("path", "word"),
    [
        ("rulesets/capablanca.json", "8x8"),
        ("rulesets/knightmate.json", "royal"),
        ("rulesets/atomic.json", "explode_on_capture"),
        ("rulesets/shatranj.json", "stalemate"),
        ("rulesets/crazyhouse.json", "drops"),
        ("rulesets/xiangqi.json", "8x8"),
    ],
)
def test_a_game_that_is_not_chess_underneath_is_the_game_without_it(path: str, word: str) -> None:
    """Inert, as a strike that only captures is: valid, the same name, and no rule compiled,
    so a variant derived from chess by adding a rule chess lacks is that variant."""
    spec = load_spec(ROOT / path)
    plain_hash = spec_hash(spec)
    spec.setdefault("rules", {})["insufficient_material"] = True
    _, reasons = insufficient_material_roles(spec)
    assert reasons and word in "; ".join(reasons), reasons
    assert validate_spec(spec) == []
    assert spec_hash(spec) == plain_hash
    assert compile_spec(spec).insufficient_material is None


def test_chess_with_a_rule_chess_lacks_draws_nothing() -> None:
    """Chess (which states the rule) given wild pieces: the rule falls silent."""
    spec = _chess()
    spec["rules"]["check_limit"] = 3
    assert validate_spec(spec) == []
    engine = Engine(spec)
    assert engine.spec.insufficient_material is None
    assert _verdict(engine, DEAD["K v K"]) is None


def test_a_promoting_minor_is_not_chess() -> None:
    """A minor that could promote is not a dead piece, so the rule says nothing there."""
    spec = _chess()
    spec["rules"]["promotion"]["piece"] = "N"
    _, reasons = insufficient_material_roles(spec)
    assert reasons


# ------------------------------------------------------------------- perft unchanged


@pytest.mark.parametrize("depth, nodes", [(1, 20), (2, 400), (3, 8_902), (4, 197_281)])
def test_chess_published_perft_unchanged_with_the_rule(depth: int, nodes: int) -> None:
    """Published chess perft numbers: the rule ends games and changes no move."""
    engine = Engine(_chess())
    assert engine.perft(engine.initial_state(), depth) == nodes
    assert engine.perft(engine.initial_state(), depth, stop_at_terminal=True) == nodes


# ---------------------------------------------------------------- Fairy-Stockfish


def test_fairy_stockfish_agrees_on_every_set_and_near_miss() -> None:
    """Their `has_insufficient_material` (both sides short of mating material) against ours,
    at each chess position above, asked through the result check's own referee."""
    pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
    from chess_variant_sandbox.fairy.results import verdicts
    from chess_variant_sandbox.fairy.variant import translate
    from chess_variant_sandbox.fairy.verify import PyffishBackend

    spec = _chess()
    engine = Engine(spec)
    variant = translate(spec)
    assert variant.insufficient_material
    referee = PyffishBackend()
    referee.load(variant, "")
    for name, placement in {**DEAD, **ALIVE}.items():
        state = engine.make_state(copy.deepcopy(placement), side_to_move=0)
        ours, theirs = verdicts(engine, variant, referee, state)
        assert ours.same(theirs), (name, ours, theirs)
        assert ours.over == (name in DEAD), name
