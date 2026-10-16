"""Schema 6: squares nothing may occupy, and captures that transform rather than remove.

The two rules share a theme -- *a capture is not the only thing that can happen when two
pieces meet, and not every square is a place two pieces can meet* -- and they fail in
opposite ways, which is why they are tested so differently here.

**An obstacle can only fail by moving a count.** It is resolved entirely at compile time
into the per-square tables, exactly as schema 5's seam was, so every question about it is
a question about how many moves exist and a perft answers it. The tests below are
therefore mostly move counts and move sets.

**A strike cannot fail by moving a count at all**, and that is the whole difficulty. It
is generated as a capture, blocked like a capture and permitted by the same matrix; only
``make`` differs. So no perft in this repository can see a strike bug, in the same way
none of them could see the atomic explosion desynchronising the Zobrist key. What can see
one is the incremental hash checked against the slow definition at every ply, the
placement of the two pieces asserted square by square, and the no-progress clock.

**The plan this file implements predicted a stronger perft invariant than is true**, and
the correction is recorded here rather than quietly dropped. It said a strike ruleset and
its ``effect: capture`` spelling would have "identical perfts at every depth". They do
not, and they should not: a strike changes the *position* after a capture, so from the
first ply at which a struck piece survives, the two are counting different trees.
``hierarchy-rise`` diverges at depth 4 and ``hierarchy-fall`` at depth 5, and the depth
each diverges at is itself a fact worth pinning. The invariant that actually says
"movegen was left alone" is one level down and is stronger than the perft claim was:
**from any given position the two spellings generate the same set of moves**, which
``test_a_strike_generates_exactly_the_moves_a_capture_does`` walks a whole game to check.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import (
    PLAYED_AS_CAPTURE,
    SpecError,
    reflect_square,
    validate_spec,
)

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def chess() -> dict[str, Any]:
    return load_spec(RULESETS / "chess.json")


def with_rules(**rules: Any) -> dict[str, Any]:
    spec = chess()
    spec["schema_version"] = 6
    spec["rules"].update(rules)
    return spec


# The demotion chain and its reverse, read off chess's own declared piece values. The
# pair is the sharpest natural experiment available: same board, same army, same starting
# material, one list the reverse of the other.
FALL = [
    {"from": "Q", "to": "R"},
    {"from": "R", "to": "B"},
    {"from": "B", "to": "N"},
    {"from": "N", "to": "P"},
]
RISE = [{"from": link["to"], "to": link["from"]} for link in reversed(FALL)]


def hierarchy_fall() -> dict[str, Any]:
    return with_rules(
        strike={"effect": "demote", "position": "stay", "chain": FALL, "immune": ["K"]}
    )


def hierarchy_rise() -> dict[str, Any]:
    return with_rules(
        strike={"effect": "promote", "position": "swap", "chain": RISE, "immune": ["K"]}
    )


def quarry() -> dict[str, Any]:
    return with_rules(
        obstacles={"squares": ["c4", "f4"], "mirror": True, "blocks_leaps": True}
    )


def probe(pieces: list[dict[str, Any]], placements: list[dict[str, Any]],
          **rules: Any) -> dict[str, Any]:
    """A bare 8x8 board with only the pieces a test needs.

    Chess's opening array fills two whole ranks, so a wall placed anywhere interesting
    stands on a piece and the spec is refused before the test can run. A probe with a
    king and one other piece puts the wall where the question is.
    """
    return {
        "schema_version": 6,
        "name": "probe",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": True,
             "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                        "symmetry": "rotate4", "mode": "both"}]},
            *pieces,
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["h8"]},
            *placements,
        ]},
        "rules": {"win": "royal_capture", **rules},
    }


def square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))


def occupants(engine: Engine, state: Any) -> dict[str, str]:
    """The board as ``{square name: "wQ"}``, which is what an assertion wants to read."""
    spec = engine.spec
    out: dict[str, str] = {}
    for index, value in enumerate(state.board):
        if not value:
            continue
        found = spec.piece_at(state.board, index)
        assert found is not None
        player, piece_type = found
        file, rank = index % spec.width, index // spec.width
        out[f"{chr(ord('a') + file)}{rank + 1}"] = "wb"[player] + spec.piece_ids[piece_type]
    return out


# --------------------------------------------------------------------- inertness


def test_the_bump_moved_no_committed_hash() -> None:
    """Every committed and archived ruleset still hashes to what it hashed to.

    The first thing a schema bump owes. ``schema_version`` is part of the mechanics
    hash, so a rule that leaked into the hash of a spec that does not use it would
    orphan every calibration run in ``runs/`` at once, and the failure would look like
    a missing measurement rather than like a bug.
    """
    root = Path(__file__).resolve().parents[1]
    frozen = {
        "chess": "4c1922ffae26",
        "xiangqi": "151582aed409",
        # Q454, 2026-09-25: traditional shogi, re-frozen on purpose (was 630828e31147).
        "shogi": "54c731ca6e95",
        "atomic": "6bd4d9c5769e",
        "crazyhouse": "bb5d97520084",
        "cylinder": "a9306688fe57",
        "rps-v2": "82a4788c02f0",
        "dull": "fe7c42772c71",
    }
    for name, digest in frozen.items():
        spec = load_spec(root / "rulesets" / f"{name}.json")
        assert spec_hash(spec) == digest, f"{name} rehashed under schema 6"


def test_the_default_spelling_of_each_rule_is_the_game_that_was_already_there() -> None:
    """``{"effect": "capture"}`` and a chain that keeps nothing alive both compile away.

    The counterpart to ``{"files": "none"}`` under schema 5 and ``{"moves": 1}`` under
    schema 4. There is no such spelling for obstacles: an empty square list is
    *rejected* rather than ignored, so the honest form of this claim for that half is
    "an absent key", which every ruleset above already tests.
    """
    assert compile_spec(with_rules(strike={"effect": "capture"})).strike is None
    # A chain naming only pieces that are immune leaves nothing alive, so it is plain
    # capture written the long way and the compiler drops it for the same reason. Asked
    # of the compiler's leaf, because since Q538 the validator refuses the whole spec.
    from chess_variant_sandbox.spec.schema import strike_changes_anything

    dead = {
        "effect": "demote",
        "position": "stay",
        "chain": [{"from": "K", "to": "Q"}],
        "immune": ["K"],
    }
    assert not strike_changes_anything(dead, "strike", ["K", "Q", "R", "B", "N", "P"])

    plain = Engine(chess())
    spelled = Engine(with_rules(strike={"effect": "capture"}))
    for depth in range(1, 5):
        assert spelled.perft(spelled.initial_state(), depth) == plain.perft(
            plain.initial_state(), depth
        )


# --------------------------------------------------------------------- obstacles


def test_a_ray_ends_at_a_wall_rather_than_passing_through_it() -> None:
    """The edit most likely to be made backwards, and the one perft would not catch.

    A square outside a piece's ``region`` gets mode 0 and the ray *continues* past it,
    because a palace is somewhere a piece may not stop. An obstacle is somewhere nothing
    may be, so the line ends. Written the other way a rook shoots through a wall, and
    every perft in the suite would happily count it.
    """
    spec = with_rules(obstacles={"squares": ["d4"]})
    engine = Engine(spec)
    state = engine.make_state(
        {"a1": ("K", 0), "a8": ("K", 1), "d1": ("R", 0)}, side_to_move=0
    )
    reached = {move_to(m) for m in engine.legal_moves(state) if move_from(m) == square("d1")}
    assert square("d2") in reached and square("d3") in reached
    assert square("d4") not in reached, "the rook may not stand on the wall"
    for beyond in ("d5", "d6", "d7", "d8"):
        assert square(beyond) not in reached, f"the rook saw {beyond} through the wall"

    # The contrast that gives the assertion its meaning: a region confines where a piece
    # may *stop* and does not end the line.
    confined = chess()
    confined["schema_version"] = 6
    for piece in confined["pieces"]:
        if piece["id"] == "R":
            piece["region"] = [
                f"{file}{rank}"
                for file in "abcdefgh"
                for rank in range(1, 9)
                if not (file == "d" and rank == 4)
            ]
    walled = Engine(confined)
    state = walled.make_state(
        {"a1": ("K", 0), "a8": ("K", 1), "d1": ("R", 0)}, side_to_move=0
    )
    reached = {move_to(m) for m in walled.legal_moves(state) if move_from(m) == square("d1")}
    assert square("d4") not in reached, "the region still forbids stopping there"
    assert square("d5") in reached, "but a region does not end the line"


def test_a_wall_stops_a_hop_too() -> None:
    """A cannon cannot shoot through a wall, because a wall is not a screen.

    Hops walk the same ray loop as slides, so this falls out of the break rather than
    needing its own rule -- and it is the reading a cannon deserves: there is nothing
    behind a wall to land on.
    """
    spec = probe(
        [{"id": "C", "name": "cannon", "value": 4, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"},
            {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
        ]},
         {"id": "P", "name": "pawn", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]}],
        [{"piece": "C", "player": 0, "squares": ["d1"]},
         {"piece": "P", "player": 0, "squares": ["d3"]},
         {"piece": "P", "player": 1, "squares": ["d7"]}],
        obstacles={"squares": ["d5"]},
    )
    engine = Engine(spec)
    state = engine.initial_state()
    reached = {move_to(m) for m in engine.legal_moves(state) if move_from(m) == square("d1")}
    assert square("d7") not in reached, "the cannon fired through a wall"


@pytest.mark.parametrize("blocks_leaps,expected", [(False, True), (True, False)])
def test_blocks_leaps_decides_whether_a_knight_may_vault_a_wall(
    blocks_leaps: bool, expected: bool
) -> None:
    """Two different games, and neither is obviously the default -- so the spec says.

    The square a jump goes over is ``_lame_path``: one step along the longer axis, which
    is the rule this project already has for a blocked leap and the one the xiangqi
    horse is actually described by. Inventing a second notion of "the way past" would
    leave two answers to one question.
    """
    knight = [{"id": "N", "name": "knight", "value": 3, "moves": [
        {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]}]
    placed = [{"piece": "N", "player": 0, "squares": ["d1"]}]

    # A knight on d1 jumps to e3 over d2, its leg by the `_lame_path` rule.
    engine = Engine(probe(knight, placed,
                          obstacles={"squares": ["d2"], "blocks_leaps": blocks_leaps}))
    state = engine.initial_state()
    reached = {move_to(m) for m in engine.legal_moves(state) if move_from(m) == square("d1")}
    assert (square("e3") in reached) is expected

    # Whichever way that goes, a knight may never *land* on a wall.
    other = Engine(probe(knight, placed,
                         obstacles={"squares": ["e3"], "blocks_leaps": blocks_leaps}))
    state = other.initial_state()
    reached = {move_to(m) for m in other.legal_moves(state) if move_from(m) == square("d1")}
    assert square("e3") not in reached


# A one-square stepper in all eight directions plus a knight, so one piece asks both
# questions a wall can raise of a leap: what a step passes over, and what a jump does.
STEPPER = [{"id": "S", "name": "stepper", "value": 3, "moves": [
    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"},
    {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]}]


def _named(index: int) -> str:
    return f"{chr(ord('a') + index % 8)}{index // 8 + 1}"


def _stepper_reaches(at: str, walls: list[str]) -> set[str]:
    """The squares a stepper on ``at`` reaches, under walls that ``blocks_leaps``."""
    spec = probe(STEPPER, [{"piece": "S", "player": 0, "squares": [at]}],
                 obstacles={"squares": walls, "blocks_leaps": True})
    engine = Engine(spec)
    state = engine.initial_state()
    return {_named(move_to(m)) for m in engine.legal_moves(state)
            if move_from(m) == square(at)}


def _shift(name: str, dx: int, dy: int) -> str:
    return f"{chr(ord(name[0]) + dx)}{int(name[1:]) + dy}"


@pytest.mark.parametrize("diagonal", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
@pytest.mark.parametrize("beside", ["file", "rank"])
def test_a_one_square_diagonal_step_passes_over_nothing(
    diagonal: tuple[int, int], beside: str
) -> None:
    """Q880: a wall beside a piece never stops its diagonal step, on any diagonal.

    ``_lame_path`` used to floor-divide the equal-axis midpoint, so ``(-1, 1)`` halved to
    ``(-1, 0)`` and ``(1, -1)`` to ``(0, -1)``: a wall on the square beside stopped two of
    the four diagonals and not their mirror images, which in Quarry kept a king on d5 off
    c6 for c5's sake. A one-square step goes over no square at all.
    """
    dx, dy = diagonal
    wall = _shift("d4", dx, 0) if beside == "file" else _shift("d4", 0, dy)
    assert _shift("d4", dx, dy) in _stepper_reaches("d4", [wall])


@pytest.mark.parametrize("wall", ["d4", "e5"])
def test_a_walls_effect_on_a_leap_is_the_same_in_every_mirror(wall: str) -> None:
    """Mirror the board, wall and piece together, and the moves mirror with them.

    The stepper stands on each of the wall's eight neighbours (and a knight's leap from
    there crosses it too), under the file, the rank and the point reflection. A wall
    belongs to the board, and a rule that read one side of a wall and not the other
    would be a rule nobody wrote.
    """
    def file_mirror(name: str) -> str:
        return f"{chr(ord('h') - (ord(name[0]) - ord('a')))}{name[1:]}"

    def rank_mirror(name: str) -> str:
        return f"{name[0]}{9 - int(name[1:])}"

    mirrors = (file_mirror, rank_mirror, lambda name: file_mirror(rank_mirror(name)))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) == (0, 0):
                continue
            at = _shift(wall, dx, dy)
            reached = _stepper_reaches(at, [wall])
            for mirror in mirrors:
                seen = _stepper_reaches(mirror(at), [mirror(wall)])
                assert seen == {mirror(name) for name in reached}, (at, wall, mirror)


def test_an_explosion_does_not_try_to_clear_a_wall() -> None:
    """A blast clears squares, and a wall is not a square that can be cleared."""
    spec = with_rules(
        obstacles={"squares": ["d5"]}, explode_on_capture={"radius": 1}
    )
    compiled = compile_spec(spec)
    assert compiled.explosion is not None
    for blast in compiled.explosion.blast:
        assert square("d5") not in blast


def test_quarry_loses_exactly_the_moves_the_walls_take() -> None:
    """Chess plus four walls, hand-checked at ply zero.

    c4, f4 and their mirrors c5 and f5. Nothing else about the opening position touches
    a wall, so the only moves lost are the two pawn double-steps that would land on one,
    and the count is 18 rather than 20. A number small enough to derive by hand is worth
    more than a deep one that only agrees with itself.
    """
    engine = Engine(quarry())
    state = engine.initial_state()
    assert len(engine.legal_moves(state)) == 18
    reached = {move_to(m) for m in engine.legal_moves(state)}
    for walled in ("c4", "f4"):
        assert square(walled) not in reached


# ------------------------------------------------------------- obstacles refused


@pytest.mark.parametrize(
    "rules,fragment",
    [
        ({"obstacles": {"squares": ["a2"]}}, "stands a piece on an obstacle"),
        ({"obstacles": {"squares": []}}, "non-empty array"),
        ({"obstacles": {"squares": ["z9"]}}, "not a square on this board"),
        ({"obstacles": {"squares": ["d4"], "mirror": "yes"}}, "must be a boolean"),
    ],
)
def test_an_obstacle_set_that_cannot_mean_anything_is_refused(
    rules: dict[str, Any], fragment: str
) -> None:
    errors = validate_spec(with_rules(**rules))
    assert any(fragment in error for error in errors), errors


def test_a_rule_that_names_a_wall_is_refused_by_name() -> None:
    """Four rules can name a square, and a wall is a square nobody can stand on."""
    spec = with_rules(
        obstacles={"squares": ["d5"]},
        reach={"piece": "K", "squares": ["d5"], "mirror": False},
    )
    errors = validate_spec(spec)
    assert any("rules.reach.squares names an obstacle" in error for error in errors), errors

    spec = with_rules(
        obstacles={"squares": ["d5"]},
        scoring={"count": "region", "region": ["d5", "e5"]},
    )
    errors = validate_spec(spec)
    assert any("rules.scoring.region names an obstacle" in error for error in errors), errors


def castling_probe(walls: list[str] | None) -> dict[str, Any]:
    """The review's reproduction: a sparse back rank that castles, and a wall on it.

    Chess's own back rank is full, so ``_check_obstacle_coherence``'s placement check
    already rejects a wall there and the corpus could never reach this. It takes a board
    somebody designed -- or ``wall-off-a-square`` mutated -- with room on the rank the
    king castles along.
    """
    spec: dict[str, Any] = {
        "schema_version": 6,
        "name": "castle-onto-a-wall",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": True,
             "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]],
                        "symmetry": "rotate4", "mode": "both"}]},
            {"id": "R", "name": "rook", "value": 5,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "squares": ["e1"]},
            {"piece": "R", "squares": ["a1", "h1"]},
        ]},
        "rules": {"win": "checkmate",
                  "castling": {"king": "K", "rook": "R", "king_steps": 2}},
    }
    if walls is not None:
        spec["rules"]["obstacles"] = {"squares": walls, "mirror": True}
    return spec


def test_a_wall_on_the_castling_rank_is_refused_by_name() -> None:
    """The defect this schema shipped, and the third instance of one root cause.

    ``generate_castling`` decides a path is clear by asking whether a piece stands on
    each square, and a wall is not a piece standing anywhere -- so the king does not
    merely castle *through* a wall, it ends up standing on one, which is the single
    thing ``rules.obstacles`` exists to forbid. The control is the same spec with no
    walls: it must still castle both ways, or the test would pass on a board that simply
    cannot castle at all.
    """
    assert validate_spec(castling_probe(None)) == []

    errors = validate_spec(castling_probe(["c1"]))
    assert any("rules.castling happens along rank 1" in error for error in errors), errors
    # ``mirror`` reflects the wall onto rank 8, which is the *other* player's castling
    # rank -- so both are named, and each is its own sentence.
    assert any("rank 8" in error for error in errors), errors


def test_a_wall_off_the_castling_rank_stays_legal() -> None:
    """A rank is the granularity, so ``quarry`` keeps playing.

    Walls on c4 and f4, castling on rank 1: this is the case a blanket refusal of
    ``castling`` beside ``obstacles`` would have cost, and it is the reason the check is
    a validation by rank rather than a hold in ``rust_supports``.
    """
    assert validate_spec(castling_probe(["c4", "f4"])) == []

    stored = load_spec(Path(__file__).resolve().parents[1] / "archive" / "quarry.json")
    assert validate_spec(stored) == []
    assert spec_hash(stored) == "51d9b36b4e93"
    engine = Engine(stored)
    assert engine.legal_moves(engine.initial_state())


def reach_probe(orientation: str, mirror: bool | str, walls: list[str] | None) -> dict[str, Any]:
    """Chess with a goal on d5, whose rank image is d4 and whose half-turn image is e4."""
    spec = with_rules(reach={"piece": "K", "squares": ["d5"], "mirror": mirror})
    spec["schema_version"] = 7
    spec["setup"]["orientation"] = orientation
    if walls:
        spec["rules"]["obstacles"] = {"squares": walls}
    return spec


@pytest.mark.parametrize("wall", ["d4", "e4"])
@pytest.mark.parametrize("mirror", [True, "rotate", False])
@pytest.mark.parametrize("orientation", ["mirror", "rotate"])
def test_the_validator_refuses_a_reach_goal_exactly_where_the_compiler_puts_one_on_a_wall(
    orientation: str, mirror: bool | str, wall: str
) -> None:
    """``reach.mirror`` names its own reflection, and both readers must agree on it.

    The validator once reflected the goal by ``setup.orientation``. It then accepted a
    mirrored army with a rotated goal on a wall, and refused a rotated army whose
    rank-reflected goal was open. The expected verdict comes from the compiled table.
    """
    open_board = compile_spec(reach_probe(orientation, mirror, None))
    assert open_board.reach is not None
    second_goal = open_board.reach.squares[1]

    errors = validate_spec(reach_probe(orientation, mirror, [wall]))
    refused = any("rules.reach.squares names an obstacle" in error for error in errors)
    assert refused == (square(wall) in second_goal), errors


def test_one_corpus_spec_reads_reach_mirror_apart_from_its_orientation() -> None:
    """Which of the committed and archived specs the two reflections tell apart.

    ``intransitive`` is the only one whose goal sets differ: its army is mirrored and its
    goal rotated, so the second player's goal is a1 and not i1. ``dobutsu`` pairs its
    squares differently under the two readings, but a whole rank is its own image. None
    of the six has walls, so the old validator never read ``reach`` on any of them.
    """
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "rulesets").glob("*.json")) + sorted(
        (root / "archive").glob("*.json")
    )
    # 77 since Q540's four schema-17 showcases, none of which declares `reach`; 80 since
    # Q787's three gap showcases, none of which does either; 82 since Q781's two schema-18
    # showcases, which do not either; 84 since Q783's two schema-18c showcases, likewise;
    # 86 since Q784's two placement showcases, likewise; 87 since Q785's `kyoto`, likewise;
    # 88 since Q786's `jeson-mor`, which does, aiming both sides at e5 (so the two
    # readings agree); 90 since Q782's two hex-symmetry showcases, which declare none;
    # 97 since Q772a's seven portal showcases, which declare none either; 98 since Q807's
    # Ferry with no straight back, which does not either; 99 since Q850's `wolves-patrol`,
    # which does not either; 100 since Q772e's `divide-two-way`, which does not either; 101
    # since Q772f's `divide-fixed`, which does not either; 87 since the library audit
    # (Q868, 2026-10-12) retired fourteen near copies, none of which declared `reach`; 84
    # since its second pass (Q870) retired four hexagon variants for `pinwheel`, which
    # declares none; 87 since Q844b's three portal showcases, which declare none either; 91
    # since Q918e's four round showcases, of which `watering-hole` does, sending both sides
    # to d4 (`mirror: false`, so the two readings agree).
    assert len(paths) == 91

    with_reach: dict[str, tuple[frozenset[int], frozenset[int]]] = {}
    for path in paths:
        spec = load_spec(path)
        if "reach" not in spec["rules"]:
            continue
        assert "obstacles" not in spec["rules"], path.name
        compiled = compile_spec(spec)
        assert compiled.reach is not None
        width, height = spec["board"]["width"], spec["board"]["height"]
        by_orientation = frozenset(
            other_rank * width + other_file
            for other_file, other_rank in (
                reflect_square(s % width, s // width, width, height, compiled.orientation)
                for s in compiled.reach.squares[0]
            )
        )
        if spec["rules"]["reach"].get("mirror", True) is False:
            by_orientation = compiled.reach.squares[0]
        with_reach[path.stem] = (compiled.reach.squares[1], by_orientation)

    assert sorted(with_reach) == [
        "dobutsu", "intransitive", "jeson-mor", "kingofthehill", "racingkings", "rps-v3",
        "watering-hole",
    ]
    differ = sorted(name for name, (built, other) in with_reach.items() if built != other)
    assert differ == ["intransitive"]
    assert with_reach["intransitive"][0] == {square("a1", 9)}
    assert with_reach["intransitive"][1] == {square("i1", 9)}


def test_a_wall_set_that_bricks_a_player_in_is_refused() -> None:
    """The degenerate case a proposal will find, and the compiler already knows it.

    ``immobile_squares`` is the set of squares a piece has no table entry from, which is
    exactly what a piece walled in has. Note what this deliberately is *not*: it is not
    "has a legal move", which is a question about a position and therefore the engine's
    to answer rather than the compiler's. A piece its own army happens to block is a
    normal chess opening.
    """
    # A lone rook boxed in on all four sides.
    spec = {
        "schema_version": 6,
        "name": "bricked",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "R", "name": "rook", "value": 5, "royal": True,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "R", "player": 0, "squares": ["c3"]},
            {"piece": "R", "player": 1, "squares": ["a1"]},
        ]},
        "rules": {"win": "royal_capture",
                  "obstacles": {"squares": ["c2", "c4", "b3", "d3"]}},
    }
    with pytest.raises(SpecError) as caught:
        compile_spec(spec)
    assert "walls player 0 in" in str(caught.value)


def test_a_wall_between_two_royals_breaks_royal_facing() -> None:
    """What A1 predicted, got wrong, and item 1 made true.

    A1 said an obstacle between two royals "must break `royal_facing`, and that falls
    out of the ray breaking, or it does not". It did not: `royals_face` was the one test
    in the core that worked a line out by *arithmetic*, walking the file asking what
    stood on each square, and a wall is not a piece standing anywhere. The combination
    was refused rather than fixed.

    It is a compiled table now, so a wall closes a line the way it closes every other
    line in this project -- the ray simply breaks at it. The control is the same
    position without the wall, or this would pass on a board where nothing ever faces.
    """
    plain = Engine(load_spec(RULESETS / "xiangqi.json"))
    walled_spec = load_spec(RULESETS / "xiangqi.json")
    walled_spec["schema_version"] = 6
    walled_spec["rules"]["obstacles"] = {"squares": ["e5"]}
    walled = Engine(walled_spec)

    # Two generals down an open file, and a soldier away on the a-file. Facing is a
    # legality filter, so the reading is "the bystander is frozen": every move it has
    # leaves the two generals looking at each other. Asked of the soldier rather than of
    # the king, which can always step off the file and so always has a move.
    position = {"e1": ("K", 0), "e9": ("K", 1), "a5": ("S", 1)}
    bystander = square("a5", width=9)

    def bystander_moves(engine: Engine) -> list[int]:
        state = engine.make_state(position, side_to_move=1)
        return [m for m in engine.legal_moves(state) if move_from(m) == bystander]

    assert bystander_moves(plain) == []
    assert bystander_moves(walled) != []


# ----------------------------------------------------------------------- strikes


def test_a_strike_generates_exactly_the_moves_a_capture_does() -> None:
    """The invariant that says movegen was left alone, and the one perft cannot state.

    A strike is generated as a capture -- same mode bit, same blocking, same
    ``may_capture`` test -- so from *the same position* the two spellings must offer the
    same *pseudo-legal* moves. The legality filter may differ because a strike
    transformation can leave the moving side's king in check where a plain capture
    would not, so the comparison is on pseudo-legal moves rather than legal ones.
    Walked over a whole game rather than checked at ply zero, because the
    positions a strike reaches are the ones a capture ruleset never sees.
    """
    for build in (hierarchy_fall, hierarchy_rise):
        spec = build()
        plain = copy.deepcopy(spec)
        plain["rules"]["strike"] = {"effect": "capture"}
        struck, ordinary = Engine(spec), Engine(plain)
        rng = random.Random(11)
        state = struck.initial_state()
        for _ in range(140):
            moves = struck.legal_moves(state)
            if struck.terminal(state, moves) is not None:
                break
            assert sorted(struck.pseudo_legal_moves(state)) == sorted(ordinary.pseudo_legal_moves(state))
            state = struck.apply(state, rng.choice(moves))


def test_perft_agrees_until_a_struck_piece_first_survives_and_then_must_not() -> None:
    """Both halves, because only the second one proves the rule does anything.

    The plan predicted identical perfts at every depth. That is false and ought to be:
    a strike changes the position a capture leaves behind, so the two are counting
    different trees from the first ply at which something survives being taken.
    ``hierarchy-rise`` diverges at depth 3 -- a struck pawn becomes a knight, and the
    surviving piece can affect the legality filter (self-check) one ply earlier than
    the old code detected -- while ``hierarchy-fall`` holds to depth 4 and diverges
    at 5, because a pawn is the *bottom* of its chain and is taken outright.
    The depth each diverges at is a consequence of the chain's direction, which is why
    it is pinned rather than merely noted.
    """
    expected = {"fall": 5, "rise": 3}
    for label, build in (("fall", hierarchy_fall), ("rise", hierarchy_rise)):
        spec = build()
        plain = copy.deepcopy(spec)
        plain["rules"]["strike"] = {"effect": "capture"}
        struck, ordinary = Engine(spec), Engine(plain)
        first_divergence = None
        for depth in range(1, 6):
            a = struck.perft(struck.initial_state(), depth)
            b = ordinary.perft(ordinary.initial_state(), depth)
            if a != b:
                # Deeper rungs cannot move the first divergence, and the deepest are the
                # dear ones (Q377).
                first_divergence = depth
                break
        assert first_divergence == expected[label], (
            f"{label} first diverged at {first_divergence}, expected {expected[label]}"
        )


def test_where_the_two_pieces_end_up_under_each_position_rule() -> None:
    """The three placements, asserted square by square.

    A survivor can be in exactly two places -- where it was, or where its attacker was --
    and ``advance`` is the third case, where nothing survives at all. Nothing about any
    of this changes a move count, so an assertion on the board is the only thing that
    can see it.
    """
    board = {"e1": ("K", 0), "e8": ("K", 1), "d4": ("R", 0), "d7": ("Q", 1)}

    stay = Engine(with_rules(
        strike={"effect": "demote", "position": "stay", "chain": FALL, "immune": ["K"]}
    ))
    state = stay.make_state(board, side_to_move=0)
    move = next(m for m in stay.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    after = occupants(stay, stay.apply(state, move))
    assert after["d4"] == "wR", "the attacker stood fast"
    assert after["d7"] == "bR", "the queen was demoted where it stood"

    swap = Engine(with_rules(
        strike={"effect": "demote", "position": "swap", "chain": FALL, "immune": ["K"]}
    ))
    state = swap.make_state(board, side_to_move=0)
    move = next(m for m in swap.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    after = occupants(swap, swap.apply(state, move))
    assert after["d7"] == "wR", "the attacker took the square"
    assert after["d4"] == "bR", "the demoted queen landed where the attacker came from"

    # `advance` is only legal where nothing survives, so the chain is spelled to remove.
    advance = Engine(with_rules(
        strike={"effect": "capture", "position": "advance"}
    ))
    state = advance.make_state(board, side_to_move=0)
    move = next(m for m in advance.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    after = occupants(advance, advance.apply(state, move))
    assert after["d7"] == "wR" and "d4" not in after, "an ordinary capture"


def test_a_promoting_swap_hands_the_defender_the_better_piece() -> None:
    """The brainstorm's own upgrade variant, which is the whole point of `rise`."""
    engine = Engine(hierarchy_rise())
    state = engine.make_state(
        {"e1": ("K", 0), "e8": ("K", 1), "d4": ("R", 0), "d7": ("P", 1)}, side_to_move=0
    )
    move = next(m for m in engine.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    after = occupants(engine, engine.apply(state, move))
    assert after["d7"] == "wR", "the rook advanced"
    assert after["d4"] == "bN", "the struck pawn came back a knight, behind the rook"


def test_the_incremental_key_survives_a_strike() -> None:
    """The atomic-explosion bug's shape, in a new place.

    A struck piece that is still on the board and not in the Zobrist key is invisible
    until two positions collide, and no perft can see it because a strike changes no
    move count anywhere. So the incremental key is checked against the slow definition
    at every ply of a great many games.
    """
    for build in (hierarchy_fall, hierarchy_rise, quarry):
        engine = Engine(build())
        for seed in range(12):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(160):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                state = engine.apply(state, rng.choice(moves))
                assert state.key == engine._compute_hash(state), (
                    f"{engine.spec.name} hash drifted at ply {state.ply}"
                )


def test_the_clock_resets_on_material_and_not_on_a_taken_piece() -> None:
    """Progress, not capture, and until schema 6 those were the same sentence.

    A demotion moves material down and is progress. A swap between two pieces of the
    same declared value moves nothing at all, and a game of those could shuffle for ever
    -- so it must *not* reset the clock, or the repetition window becomes a memory leak
    and ``no_progress_draw`` can never fire.
    """
    demoting = Engine(with_rules(
        strike={"effect": "demote", "position": "stay", "chain": FALL, "immune": ["K"]}
    ))
    state = demoting.make_state(
        {"e1": ("K", 0), "e8": ("K", 1), "d4": ("R", 0), "d7": ("Q", 1)}, side_to_move=0
    ).with_(halfmove=40)
    move = next(m for m in demoting.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    assert demoting.apply(state, move).halfmove == 0, "Q(9) became R(5): progress"

    # Two pieces of equal declared value, swapped: the board holds exactly what it held.
    spec = chess()
    spec["schema_version"] = 6
    for piece in spec["pieces"]:
        if piece["id"] in ("B", "N"):
            piece["value"] = 3
    spec["rules"]["strike"] = {
        "effect": "demote",
        "position": "swap",
        "chain": [{"from": "B", "to": "N"}],
        "immune": ["K"],
    }
    level = Engine(spec)
    state = level.make_state(
        {"e1": ("K", 0), "e8": ("K", 1), "d4": ("R", 0), "d7": ("B", 1)}, side_to_move=0
    ).with_(halfmove=40)
    move = next(m for m in level.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    assert level.apply(state, move).halfmove == 41, (
        "a bishop for a knight of the same declared value is not progress"
    )


def test_an_immune_piece_and_a_forbidden_capture_both_fall_back_to_the_old_behaviour() -> None:
    """Two rules that run before a strike, and neither is disturbed by one."""
    engine = Engine(hierarchy_fall())
    # The king is immune, so taking one removes it as it always did.
    state = engine.make_state(
        {"e1": ("K", 0), "d7": ("K", 1), "d4": ("R", 0)}, side_to_move=0
    )
    move = next(m for m in engine.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    after = occupants(engine, engine.apply(state, move))
    assert "d7" not in after or after["d7"] == "wR"
    assert "bK" not in after.values(), "an immune royal is captured, not demoted"

    # `captures: []` runs first: a piece that may take nothing never strikes anything.
    spec = hierarchy_fall()
    for piece in spec["pieces"]:
        if piece["id"] == "R":
            piece["captures"] = []
    blocked = Engine(spec)
    state = blocked.make_state(
        {"e1": ("K", 0), "e8": ("K", 1), "d4": ("R", 0), "d7": ("Q", 1)}, side_to_move=0
    )
    assert not [m for m in blocked.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7")]


def test_a_strike_that_creates_check_is_filtered_from_legal_moves() -> None:
    """Regression: the check filter must account for the surviving transformed piece.

    A swap-strike where the survivor lands on the attacker's origin can leave a
    transformed piece that gives check -- the old code did not apply the strike
    transformation inside the legality filter, so the move was allowed and the
    king could be captured on the next ply without the game ending.
    """
    # White rook on e4, white king on a4, black queen on e8, black king on h8.
    # Under "swap" with demotion: white rook goes to e8, black queen is demoted
    # to a rook and placed on e4 (the attacker's origin).  That black rook on e4
    # now attacks the white king on a4 along the 4th rank -- the move must be
    # illegal.
    engine = Engine(with_rules(
        strike={"effect": "demote", "position": "swap", "chain": FALL, "immune": ["K"]}
    ))
    state = engine.make_state(
        {"a4": ("K", 0), "h8": ("K", 1), "e4": ("R", 0), "e8": ("Q", 1)},
        side_to_move=0,
    )
    capture_moves = [
        m for m in engine.legal_moves(state)
        if move_from(m) == square("e4") and move_to(m) == square("e8")
    ]
    assert capture_moves == [], (
        "Re4xe8 should be illegal: the demoted queen on e4 checks the white king on a4"
    )


def test_a_game_initialised_with_opponent_in_check_ends_immediately() -> None:
    """Regression: a position where the non-moving side's king is in check is terminal.

    Standard chess initialised with black in check should be detected as already won
    for white, rather than allowing white to capture the king without the game ending.
    """
    engine = Engine(with_rules())  # standard chess rules, schema 6
    # White queen on d7 checks the black king on e8; white king on a1.
    state = engine.make_state(
        {"a1": ("K", 0), "e8": ("K", 1), "d7": ("Q", 0)},
        side_to_move=0,
    )
    assert engine.in_check(state, 1), "black should be in check"
    outcome = engine.terminal(state)
    assert outcome is not None, "the game should be over immediately"
    assert outcome.winner == 0, "white should win"


# --------------------------------------------------------------- strikes refused


@pytest.mark.parametrize(
    "strike,fragment",
    [
        (
            {"effect": "demote", "position": "stay", "chain": FALL},
            "not in rules.strike.immune",
        ),
        (
            {"effect": "demote", "position": "stay",
             "chain": FALL + [{"from": "K", "to": "Q"}], "immune": []},
            "names the royal piece",
        ),
        (
            {"effect": "demote", "position": "advance", "chain": FALL, "immune": ["K"]},
            "has nowhere to put a struck piece that survives",
        ),
        (
            {"effect": "demote", "position": "stay", "immune": ["K"]},
            "no chain says where a struck piece goes",
        ),
        (
            {"effect": "demote", "position": "stay", "immune": ["K"],
             "chain": [{"from": "Q", "to": "R"}, {"from": "R", "to": "Q"}]},
            "cycles at",
        ),
        (
            {"effect": "demote", "position": "stay", "immune": ["K"],
             "chain": [{"from": "Q", "to": "Q"}]},
            "points 'Q' at itself",
        ),
        (
            {"effect": "demote", "position": "stay", "immune": ["K"],
             "chain": [{"from": "Q", "to": "R"}, {"from": "Q", "to": "B"}]},
            "two links from 'Q'",
        ),
        (
            {"effect": "demote", "position": "stay", "immune": ["K"],
             "chain": [{"from": "Q", "to": "Z"}]},
            "not a declared piece",
        ),
    ],
)
def test_a_strike_rule_that_cannot_mean_anything_is_refused(
    strike: dict[str, Any], fragment: str
) -> None:
    errors = validate_spec(with_rules(strike=strike))
    assert any(fragment in error for error in errors), errors


@pytest.mark.parametrize(
    "strike,fragment",
    [
        # `rifle-b`, the find, read as rifle chess and played an ordinary capture. Since
        # schema 17 (Q540) `stay` under `capture` *is* rifle; see the test below.
        ({"effect": "capture", "position": "swap"}, "position 'swap'"),
        ({"effect": "capture", "chain": FALL}, "effect 'capture' and a chain"),
        ({"effect": "capture", "immune": ["K"]}, "effect 'capture' and an immune list"),
        # A chain whose every `from` is immune, under each position that is not `choose`.
        *(
            (
                {"effect": effect, "position": position, "chain": [{"from": "Q", "to": "R"}],
                 "immune": ["K", "Q"]},
                "starts every link at a piece in rules.strike.immune",
            )
            for effect in ("demote", "promote")
            for position in ("stay", "swap", "advance")
        ),
    ],
)
def test_a_strike_the_core_plays_as_plain_capture_is_refused(
    strike: dict[str, Any], fragment: str
) -> None:
    """Q538: a strike that states a rule and compiles to plain capture.

    The compiler drops every one of these (``strike_changes_anything`` is false), so the
    core removes the victim and moves the attacker in, whatever ``position`` said. One
    refusal each, and nothing else refused beside it.
    """
    spec = with_rules(strike=strike)
    from chess_variant_sandbox.spec.schema import strike_changes_anything

    assert not strike_changes_anything(strike, "strike", [p["id"] for p in spec["pieces"]])
    errors = validate_spec(spec)
    assert len(errors) == 1 and fragment in errors[0] and PLAYED_AS_CAPTURE in errors[0], errors


@pytest.mark.parametrize(
    "strike",
    [
        {"effect": "capture", "position": "stay"},
        {"effect": "capture", "position": "stay", "immune": ["K"]},
        {"position": "stay", "immune": ["K"]},
    ],
)
def test_rifle_is_schema_seventeens(strike: dict[str, Any]) -> None:
    """Q538's find, `rifle-b`, became a rule in Q540: `stay` under `"capture"` removes the
    victim and leaves the attacker where it stood. Under schema 6 it is refused by name as
    schema 17's; under 17 it validates once every royal is immune."""
    from chess_variant_sandbox.spec.schema import strike_changes_anything

    spec = with_rules(strike=strike)
    assert strike_changes_anything(strike, "strike", [p["id"] for p in spec["pieces"]])
    assert any("added in schema 17" in error for error in validate_spec(spec))
    spec["schema_version"] = 17
    errors = validate_spec(spec)
    assert errors == [] if "immune" in strike else any("not in rules.strike.immune" in e for e in errors)


@pytest.mark.parametrize(
    "strike",
    [
        {"effect": "capture"},
        {"effect": "capture", "position": "advance"},
        {"effect": "capture", "immune": []},
        {},
    ],
)
def test_the_bare_capture_spelling_is_still_legal(strike: dict[str, Any]) -> None:
    """The default said aloud is not refused: it is how schema 6 proved itself inert."""
    assert validate_spec(with_rules(strike=strike)) == []


def test_settling_drops_a_strike_a_pruned_chain_left_with_nothing_to_strike() -> None:
    """``settle_inert_rules`` removes what the new refusal refuses, and nothing sound."""
    from chess_variant_sandbox.spec.transform import settle_inert_rules

    dead = with_rules(
        strike={"effect": "demote", "position": "stay", "chain": [{"from": "Q", "to": "R"}],
                "immune": ["K", "Q"]}
    )
    assert settle_inert_rules(dead) == ["strike"]
    assert "strike" not in dead["rules"]
    assert validate_spec(dead) == []

    for sound in (
        {"effect": "capture"},
        {"effect": "demote", "position": "stay", "chain": FALL, "immune": ["K"]},
    ):
        spec = with_rules(strike=copy.deepcopy(sound))
        assert settle_inert_rules(spec) == []
        assert spec["rules"]["strike"] == sound


def test_a_promotion_chain_may_cycle_where_a_demotion_chain_may_not() -> None:
    """A cycle under ``promote`` is a legal and interesting game, and it is asymmetric.

    Taking the top piece hands the defender the bottom one, which is a real design. A
    cycle under ``demote`` is an infinite ladder and material would never leave the
    board, which is the difference the validator is being asked to know.
    """
    ring = [
        {"from": "P", "to": "N"},
        {"from": "N", "to": "B"},
        {"from": "B", "to": "P"},
    ]
    assert not validate_spec(with_rules(
        strike={"effect": "promote", "position": "swap", "chain": ring, "immune": ["K"]}
    ))
    errors = validate_spec(with_rules(
        strike={"effect": "demote", "position": "stay", "chain": ring, "immune": ["K"]}
    ))
    assert any("cycles at" in error for error in errors), errors


@pytest.mark.parametrize("other", ["explode_on_capture", "drops"])
def test_two_rules_may_not_both_say_what_happens_to_a_taken_piece(other: str) -> None:
    """An exploding strike would destroy what it just transformed; a strike under drops
    would put the same piece in a hand *and* leave it standing.

    Refusals rather than orderings, because neither ordering is the obvious one and
    picking one silently is how a plausible wrong game gets played.
    """
    extra = {"radius": 1} if other == "explode_on_capture" else {"from_captures": True}
    spec = with_rules(
        strike={"effect": "demote", "position": "stay", "chain": FALL, "immune": ["K"]},
        **{other: extra},
    )
    errors = validate_spec(spec)
    assert any("both say what happens to a taken piece" in error for error in errors), errors


# ------------------------------------------------------------- the other surfaces


def test_a_schema_five_spec_may_not_reach_for_schema_six() -> None:
    spec = with_rules(obstacles={"squares": ["d4"]})
    spec["schema_version"] = 5
    errors = validate_spec(spec)
    assert any("added in schema 6" in error for error in errors), errors


def test_the_prose_says_what_each_rule_does() -> None:
    """A rule the prose has not learned is a rule nobody is told about.

    ``tests/test_spec.py`` already enforces that removing a costed rule changes the
    description. This checks the harder half: that what the description *says* is
    something a player could act on.
    """
    from chess_variant_sandbox.spec.describe import describe_spec

    text = describe_spec(hierarchy_fall())
    assert "does not remove it" in text or "not remove it" in text
    assert "demoted" in text
    assert "queen becomes rook" in text
    assert "king" in text.lower()

    walls = describe_spec(quarry())
    assert "walls" in walls
    assert "jump over one" in walls  # blocks_leaps is on, so it may not


def test_the_two_hierarchy_games_do_not_measure_identical() -> None:
    """The `repaints_territory` lesson, a fourth time.

    ``hierarchy-fall`` and ``hierarchy-rise`` differ in the direction of one list. A
    rule with a cost but no feature of its own would make them distance zero, and the
    loop would treat one as a rediscovery of the other and skip the tournament that is
    the entire reason for running both.
    """
    from chess_variant_sandbox.spec.distance import distance

    assert distance(hierarchy_fall(), hierarchy_rise()) > 0.0
    assert distance(hierarchy_fall(), chess()) > 0.0
    assert distance(quarry(), chess()) > 0.0


@pytest.mark.parametrize("position", ["stay", "swap", "advance"])
def test_a_chain_of_only_immune_links_measures_as_no_strike(position: str) -> None:
    """Q414: a strike the compiler drops is chess, and measures zero from it.

    Every ``from`` below is immune, so a strike never reaches one and the chain is plain
    capture written the long way. The compiler drops it and `complexity` charges nothing
    for it (Q95), but `distance` used to read any non-``capture`` chain as a strike, so
    this spec, which compiles to chess's own tables, measured 0.018 to 0.036 from chess.
    The live counterpart, one link reachable, still measures as a strike.

    Since Q538 the validator refuses the dead spelling (a chain stated and never played),
    so this now holds `distance` for a spec read before it is validated.
    """
    from chess_variant_sandbox.spec.distance import distance, features

    dead = with_rules(
        strike={
            "effect": "promote",
            "position": position,
            "chain": RISE,
            "immune": ["K", "P", "N", "B", "R"],
        }
    )
    assert any(PLAYED_AS_CAPTURE in error for error in validate_spec(dead))
    assert features(dead) == features(chess())
    assert distance(dead, chess()) == 0.0

    if position == "advance":
        return  # the validator refuses `advance` beside a survivor
    live = copy.deepcopy(dead)
    live["rules"]["strike"]["immune"] = ["K", "P", "N", "B"]
    assert validate_spec(live) == []
    assert compile_spec(live).strike is not None
    assert features(live)["strikes"] == 1.0
    assert features(live)["strike_promotes"] == 1.0
    assert distance(live, chess()) > 0.0


def test_the_grid_separates_a_transforming_capture_and_pads_older_labels() -> None:
    """The eighth axis, and the reason it was added rather than argued for.

    Schema 6's three showcases all landed in the chess cell and took it from five to
    eight, past the same threshold the fifth and seventh axes exist to hold. Obstacles
    deliberately get no axis: a wall is a fact about the board's shape, which the
    ``board`` axis is already a coarse reading of.
    """
    from chess_variant_sandbox.loop import grid

    assert grid.descriptor(chess()) == grid.descriptor(quarry())
    assert grid.descriptor(chess()) != grid.descriptor(hierarchy_fall())
    assert grid.descriptor(hierarchy_fall()) == grid.descriptor(hierarchy_rise())
    # A label written before the axis existed is a prefix, and padding it is the
    # migration rather than a guess: a spec written before schema 6 removes what it takes.
    assert grid.parse("1-1-0-0-1-0-0") == grid.descriptor(chess())


def test_the_operators_can_reach_both_rules_without_a_model_call() -> None:
    """A rule reachable only by a model call is a rule the search cannot cross to."""
    from chess_variant_sandbox.loop.operators import OPERATORS

    names = {op.name for op in OPERATORS}
    assert {"wall-off-a-square", "soften-the-capture"} <= names

    wall = next(op for op in OPERATORS if op.name == "wall-off-a-square")
    spec = chess()
    note = wall.apply(spec, random.Random(3))
    assert note and not validate_spec(spec), validate_spec(spec)
    assert spec["rules"]["obstacles"]["squares"]

    soften = next(op for op in OPERATORS if op.name == "soften-the-capture")
    spec = chess()
    note = soften.apply(spec, random.Random(3))
    assert note and not validate_spec(spec), validate_spec(spec)
    assert spec["rules"]["strike"]["chain"]
    # And back again, which is what makes it an axis rather than a one-way door. Seed 0
    # draws the removal half; the other half places the survivor (schema 12).
    note = soften.apply(spec, random.Random(0))
    assert note and "strike" not in spec["rules"]
    assert spec["schema_version"] < 6, "the version settles when the rule goes"


def test_the_move_text_says_what_a_strike_actually_did() -> None:
    """"Rd4xd7" would say the rook moved to d7 when under ``stay`` it never left d4.

    A record a reader cannot follow is a record, and this is user-facing in three
    places: the CLI, the visualiser and interactive play.
    """
    engine = Engine(hierarchy_fall())
    state = engine.make_state(
        {"e1": ("K", 0), "e8": ("K", 1), "d4": ("R", 0), "d7": ("Q", 1)}, side_to_move=0
    )
    move = next(m for m in engine.legal_moves(state)
                if move_from(m) == square("d4") and move_to(m) == square("d7"))
    text = engine.describe_move(state, move)
    assert "->R" in text and "stands fast" in text, text


def test_the_board_draws_its_walls(tmp_path: Any) -> None:
    """A board that does not draw its obstacles is a board on which pieces refuse a
    square for no visible reason.

    This is the territory colouring's failure again -- three committed rulesets were
    *won* on a quantity the board never drew -- and it is the kind of gap no engine test
    can see, because every game plays correctly and only the picture is wrong.
    """
    from chess_variant_sandbox.tournament.play import play_game
    from chess_variant_sandbox.viz.html import render_game

    spec = quarry()
    record = play_game(Engine(spec), ("random", "random"), seed=0)
    out = render_game(spec, record, tmp_path / "board.html")
    html = out.read_text(encoding="utf-8")
    # c4 and f4 with their mirrors c5 and f5, as square indices on an 8-wide board.
    assert '"obstacles": [26, 29, 34, 37]' in html
    assert "const WALLS" in html
    assert "--wall" in html, "the palette has to name the colour the walls are drawn in"


def test_the_interactive_board_is_told_where_the_walls_are() -> None:
    """The same gap on the surface a person actually plays on.

    The snapshot is what the page draws from, so asserting on it is asserting on what a
    player can see -- and the page is checked for the matching reader, because a payload
    nothing reads would pass a test and still draw an ordinary board.
    """
    from chess_variant_sandbox.web.library import Game
    from chess_variant_sandbox.web.play import Session

    spec = quarry()
    engine = Engine(spec)
    game = Game(
        hash=spec_hash(spec), name=str(spec["name"]),
        description=str(spec.get("description", "")),
        path=Path("archive/quarry.json"), source="committed", spec=spec,
    )
    session = Session(
        id="t", game=game, seats=("human", "human"), seed=0,
        state=engine.initial_state(),
    )
    # c4 and f4 with their mirrors c5 and f5, as square indices on an 8-wide board.
    assert session.snapshot()["obstacles"] == [26, 29, 34, 37]

    # Imported rather than read off disk: the page is a package of parts assembled into
    # one string, and a test that opened `web/page.py` would be asserting about a file
    # layout rather than about the page.
    from page_text import PAGE

    assert "SESSION.obstacles" in PAGE, (
        "the payload carries the walls and the page never reads them"
    )


def test_the_showcase_rulesets_play_a_whole_game_through() -> None:
    """A ruleset is verified when it has been *played*, not when it has been counted.

    Three separate bugs in this project's history survived every perft in the suite,
    because none of them was about counting moves. This runs the whole stack -- agents,
    evaluator, terminal detection -- and the search exercises ``make``/``unmake``, which
    a random playout never touches.
    """
    from chess_variant_sandbox.tournament.play import play_game

    for build in (hierarchy_fall, hierarchy_rise, quarry):
        spec = build()
        record = play_game(Engine(spec), ("minimax:nodes=400", "greedy"), seed=0)
        assert record.plies > 0
        assert record.reason, "the game reached a stated verdict"
        assert len(record.moves) == record.plies


# ------------------------------------------------ Q922: the strike checks' edges


class TestEveryRoyalIsCheckedAgainstTheImmuneList:
    """`_check_strike_coherence` visits every royal: a spared one does not end the walk."""

    @staticmethod
    def _errors(immune: list[str], chain: list[dict[str, str]]) -> list[str]:
        from chess_variant_sandbox.spec.schema import _check_strike_coherence

        pieces = [{"id": "A", "royal": True}, "junk", {"id": "N"}, {"id": "B", "royal": True}]
        rules = {"strike": {"effect": "demote", "chain": chain, "immune": immune}}
        errors: list[str] = []
        _check_strike_coherence(pieces, rules, errors)
        return errors

    def test_a_spared_royal_ahead_of_an_unspared_one(self) -> None:
        (error,) = self._errors(["A"], [{"from": "N", "to": "N"}])
        assert error.startswith("the royal piece 'B' is not in rules.strike.immune")

    def test_a_spared_royal_ahead_of_one_the_chain_names(self) -> None:
        (error,) = self._errors(["A"], [{"from": "B", "to": "N"}])
        assert error.startswith("rules.strike.chain names the royal piece 'B'")

    def test_every_royal_spared_is_no_error(self) -> None:
        assert self._errors(["A", "B"], [{"from": "N", "to": "N"}]) == []


@pytest.mark.parametrize("junk", ["capture", ["position"], 3, True])
@pytest.mark.parametrize("key", ["strike", "contacts"])
def test_a_strike_that_is_not_a_table_states_nothing(junk: Any, key: str) -> None:
    """The shape check refuses it; this one must not crash asking what it states."""
    from chess_variant_sandbox.spec.schema import strike_states_nothing

    assert strike_states_nothing(junk, key, ["P", "N"]) == []
    assert strike_states_nothing({}, key, ["P", "N"]) == []
