"""The lifted-pair fuzz (R-L6a): its generator, its coverage, and every check failing on a
planted fault.

A check that cannot be shown to fail is not a check. So each check below is first run on a
real record of a small spec holding its rule, where it must pass and must have judged at
least one event of its rule; then on the same record doctored -- a board edited, a move word
swapped, an engine answer stubbed to say something wrong -- where it must fire. The specs are
written here rather than drawn from the generator, so an operator that changes what it draws
moves no test.

The engine faults the fuzz found are pinned at the bottom (F1 to F4 fixed by R-L6c, F5 by
U1b), each with the smallest spec that shows it (`docs/reference/fuzz-lifted-pairs.md` §5).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import SIGHT_FREEZE, WILD
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_NORMAL,
    MOVE_PROMOTION,
    MOVE_SWEEP,
    encode_move,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import Effects, Outcome, PhaseStep
from chess_variant_sandbox.fuzz_lifted import checks, generate, keys
from chess_variant_sandbox.fuzz_lifted.checks import CHECKS, CheckFailure
from chess_variant_sandbox.fuzz_lifted.record import Record, Step, play
from chess_variant_sandbox.fuzz_lifted.run import Coverage, check_case, run
from chess_variant_sandbox.spec.schema import validate_spec

A_TO_F = ("A", "B", "C", "D", "E", "F")

# The row list a seed recorded before R-L4b (the capture list's 18 rows) drew from: a seed names
# a row by its place in the list, so the seeds pinned below replay against that list.
_BEFORE_THE_LIST = tuple(row for row in keys.lifted() if row.by != "R-L4b")
# R-L4b5 appended two rows; seeds pinned before it name their row among the rows before them.
_BEFORE_THE_ROUND_ROWS = tuple(row for row in keys.lifted() if row.id not in ("T1.rounds", "T4.rounds"))


# --------------------------------------------------------------------- the lifted list

def test_every_lifted_key_path_has_a_detector() -> None:
    """A new `[[lifted]]` row naming a path nobody reads fails here, not silently."""
    assert keys.lifted(), "spec/refusals.toml has no [[lifted]] rows"
    assert keys.unknown_keys() == []


def test_every_row_of_batches_a_to_f_has_a_check_and_a_maker() -> None:
    """Done-when (2): a check for every pair batches A to F lifted, and a generator for it."""
    rows = [row for row in keys.lifted() if row.batch in A_TO_F]
    assert len(rows) >= 60, len(rows)
    unchecked = [row.id for row in rows if row.id not in checks.checked_pairs()]
    assert unchecked == []
    buildable = {row.id for row in generate.buildable()}
    assert [row.id for row in rows if row.id not in buildable] == []


def test_a_check_names_only_rows_that_exist() -> None:
    ids = {row.id for row in keys.lifted()}
    assert checks.checked_pairs() <= ids
    assert len({check.name for check in CHECKS}) == len(CHECKS)


def test_the_batch_is_read_from_the_task_that_lifted_it() -> None:
    assert keys.batch_of("SR8c") == "F"
    assert keys.batch_of("SR2b, SR4h") == "B"
    assert keys.batch_of("SR1d") == "A"
    assert keys.batch_of("SR1a") == "none"
    assert keys.batch_of("Q1074b8") == "none"


# ------------------------------------------------------------------------ the generator

def test_a_seed_rebuilds_exactly_the_same_case() -> None:
    first, second = generate.case(41), generate.case(41)
    assert first.spec == second.spec and first.row == second.row and first.makers == second.makers


def test_every_case_played_is_one_the_validator_accepts() -> None:
    """Nothing refused is played: a built case validates, and a seed that reached no valid
    spec says so with no spec."""
    built = [generate.case(seed) for seed in range(40)]
    made = [case for case in built if case.spec is not None]
    assert len(made) >= 25
    for case in made:
        assert validate_spec(case.spec) == [], case.describe()
        assert case.row in case.present, case.describe()
    for case in built:
        if case.spec is None:
            assert case.why


def test_the_hex_rows_are_built_on_a_hex_board() -> None:
    hexes = [generate.case(seed) for seed in range(120)]
    hexes = [c for c in hexes if c.spec is not None and c.parent == "hex"]
    assert hexes
    assert all(c.spec["board"]["lattice"] == "hex" for c in hexes)


def test_a_run_reports_what_it_covered_and_resumes(tmp_path) -> None:
    """Done-when (4) and the checkpoint: the report names the rows with neither a case nor
    a check, and a run killed and resumed from its file ends where an unbroken one does."""
    whole = run(range(0, 24))
    report = whole.report()
    assert "neither a case nor a check" in report
    assert "never occurred in a case" in report
    assert "cases an hour" in report
    checkpoint = tmp_path / "soak.json"
    run(range(0, 10), checkpoint=checkpoint)
    resumed = run(range(0, 24), checkpoint=checkpoint)
    assert resumed.next_seed == 24
    assert resumed.occurred == whole.occurred
    assert resumed.played == whole.played
    assert resumed.failures == whole.failures


def test_a_failure_carries_the_seed_and_the_spec(monkeypatch) -> None:
    def always(record: Record) -> int:
        raise CheckFailure("step 0: planted")

    planted = (dataclasses.replace(CHECKS[0], keys=(), run=lambda v: always(v.record)),)
    monkeypatch.setattr("chess_variant_sandbox.fuzz_lifted.run.CHECKS", planted)
    coverage = Coverage()
    for seed in range(6):
        check_case(seed, coverage)
    assert coverage.failures
    failure = coverage.failures[0]
    assert failure["message"] == "step 0: planted"
    rebuilt = generate.case(failure["seed"])
    assert rebuilt.spec == failure["spec"]


# ------------------------------------------------------------------------- the checks

def _six(**rules: Any) -> dict[str, Any]:
    spec = copy.deepcopy(generate.SIX)
    spec["rules"].update(rules)
    return spec


def _piece(spec: dict[str, Any], piece_id: str) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["id"] == piece_id)


def _check(name: str) -> checks.Check:
    return next(check for check in CHECKS if check.name == name)


def _judged_record(name: str, spec: dict[str, Any], seeds: int = 120,
                   want: Callable[[Record], bool] | None = None) -> Record:
    """A record of ``spec`` the check passes and judged something in (and ``want`` holds)."""
    assert validate_spec(spec) == [], validate_spec(spec)
    check = _check(name)
    assert check.applies(spec)
    engine = Engine(spec)
    for seed in range(seeds):
        record = play(spec, seed, engine=engine)
        try:
            judged = check(record)
        except CheckFailure as failure:
            if checks.KNOWN_PREFIX in str(failure):
                continue  # a recurrence of a fault already found, pinned below
            raise
        if judged and (want is None or want(record)):
            return record
    raise AssertionError(f"no seed in {seeds} gave {name} an event to judge")


class _Stubbed:
    """An engine with one answer replaced: the planted engine fault (the engine's own
    attributes are read-only)."""

    def __init__(self, engine: Engine, **answers: Any) -> None:
        self._engine = engine
        self._answers = answers

    def __getattr__(self, name: str) -> Any:
        return self._answers[name] if name in self._answers else getattr(self._engine, name)


def _stubbed(record: Record, **answers: Any) -> Record:
    return Record(spec=record.spec, engine=_Stubbed(record.engine, **answers),  # type: ignore[arg-type]
                  steps=record.steps, outcome=record.outcome)


def _with_steps(record: Record, steps: list[Step]) -> Record:
    return Record(spec=record.spec, engine=record.engine, steps=steps, outcome=record.outcome)


def _edit_after(record: Record, index: int, board: dict[int, int] | None = None, **fields: Any) -> Record:
    steps = list(record.steps)
    step = steps[index]
    after = step.after
    if board:
        cells = bytearray(after.board)
        for square, value in board.items():
            cells[square] = value
        after = after.with_(board=bytes(cells))
    if fields:
        after = after.with_(**fields)
    steps[index] = dataclasses.replace(step, after=after)
    return _with_steps(record, steps)


def _fires(name: str, record: Record) -> str:
    with pytest.raises(CheckFailure) as caught:
        _check(name)(record)
    return str(caught.value)


def _first(record: Record, test: Callable[[Step], bool]) -> int:
    return next(i for i, step in enumerate(record.steps) if test(step))


def _empty_square(board: bytes, avoid: set[int] = frozenset()) -> int:  # type: ignore[assignment]
    return next(sq for sq, value in enumerate(board) if not value and sq not in avoid)


def test_the_duck_check_fires_on_a_duck_a_piece_move_carried() -> None:
    spec = _six(duck={"piece": "D"})
    spec["pieces"].append({"id": "D", "value": 0, "moves": []})
    record = _judged_record("duck_stands_alone_and_moves_last", spec)
    engine = record.engine
    index = _first(record, lambda s: s.index > 1 and checks.move_type(s.moves[0]) != checks.MOVE_DUCK)
    step = record.steps[index]
    duck = engine.spec.duck.piece_type
    at = next(sq for sq, v in enumerate(step.after.board)
              if v and engine.spec.owner_of(v) == WILD and engine.spec.type_of(v) == duck)
    to = _empty_square(step.after.board)
    planted = _edit_after(record, index, {at: 0, to: step.after.board[at]})
    assert "carried the duck" in _fires("duck_stands_alone_and_moves_last", planted)


def test_the_turn_check_fires_on_a_turn_one_move_short() -> None:
    record = _judged_record("a_turn_is_as_long_as_the_schedule", _six(turn={"moves": 2}))
    planted = _with_steps(record, record.steps[1:])
    assert "the schedule says" in _fires("a_turn_is_as_long_as_the_schedule", planted)


def test_the_en_passant_check_fires_on_a_window_offered_mid_turn() -> None:
    record = _judged_record("en_passant_only_on_the_turn_s_first_move",
                            _six(en_passant=True, turn={"moves": 2}))
    index = _first(record, lambda s: s.index > 0 and record.steps[s.index - 1].mover == s.mover)
    step = record.steps[index]
    word = encode_move(8, 15, checks.MOVE_EN_PASSANT)
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, offered=step.offered + (word,))
    assert "after the turn's first" in _fires("en_passant_only_on_the_turn_s_first_move", _with_steps(record, steps))


def test_the_rest_check_fires_on_a_resting_piece_moved_again() -> None:
    spec = _six()
    _piece(spec, "N")["rest"] = 2
    engine = Engine(spec)
    n_type = engine.spec.piece_ids.index("N")

    def knight_moved_then_waited(record: Record) -> bool:
        return any(
            s.index + 2 < len(record.steps) and engine.spec.type_of(s.before.board[move_from(s.moves[0])] or 1) == n_type
            and record.steps[s.index + 2].before.board[move_to(s.moves[0])] == s.after.board[move_to(s.moves[0])]
            for s in record.steps if move_type(s.moves[0]) == MOVE_NORMAL and s.before.board[move_from(s.moves[0])]
        )

    record = _judged_record("a_resting_piece_is_not_moved", spec, want=knight_moved_then_waited)
    index = next(
        s.index for s in record.steps
        if move_type(s.moves[0]) == MOVE_NORMAL and s.before.board[move_from(s.moves[0])]
        and engine.spec.type_of(s.before.board[move_from(s.moves[0])]) == n_type
        and s.index + 2 < len(record.steps)
        and record.steps[s.index + 2].before.board[move_to(s.moves[0])] == s.after.board[move_to(s.moves[0])]
    )
    landed = move_to(record.steps[index].moves[0])
    steps = list(record.steps)
    later = steps[index + 2]
    steps[index + 2] = dataclasses.replace(later, moves=(encode_move(landed, landed, MOVE_NORMAL),))
    assert "while it rests" in _fires("a_resting_piece_is_not_moved", _with_steps(record, steps))


def _wild_spec(**extra: Any) -> dict[str, Any]:
    entry = {"piece": "B", "route": ["b3", "c3", "c4", "b4"]}
    entry.update(extra)
    return _six(npcs=[entry])


def test_the_route_check_fires_on_a_wild_piece_off_its_route() -> None:
    record = _judged_record("wild_pieces_keep_their_routes", _wild_spec())
    engine = record.engine
    step = record.steps[0]
    at = next(sq for sq, v in enumerate(step.after.board) if v and engine.spec.owner_of(v) == WILD)
    to = next(sq for sq in range(engine.spec.n_squares)
              if not step.after.board[sq] and engine.spec.npcs.next[sq] < 0)
    planted = _edit_after(record, 0, {at: 0, to: step.after.board[at]})
    assert "no route runs" in _fires("wild_pieces_keep_their_routes", planted)


def _respawning_pouncer_spec() -> dict[str, Any]:
    """A bishop that pounces on its prey on a3-b3, mirrored onto a4-b4, coming back after a
    phase (N7, SR8g)."""
    return _six(npcs=[{"piece": "B", "route": ["a3", "b3"], "mirror": True, "pounce": "prey",
                       "respawn": 1}])


def _later_starts(record: Record, step: Step) -> list[int]:
    """The starts after the step's first return in the mover's reading order."""
    spec = record.engine.spec
    back = next(p for p in step.phase if p.kind == "return")
    starts = next(starts for _, starts in spec.npcs.respawn if back.target in starts)
    reading = spec.pounce.reading_rank[step.mover]
    return [sq for sq in starts if reading[sq] > reading[back.target]]


def _returns_short_of_the_last(record: Record) -> bool:
    return any(
        not step.round and any(p.kind == "return" for p in step.phase) and _later_starts(record, step)
        for step in record.steps
    )


def test_the_return_check_fires_on_a_pouncer_back_past_a_free_start() -> None:
    """The record's first return moved to a later start of its entry: the one it really came
    back on, left free, now stands before it."""
    name = "a_wild_piece_comes_back_on_its_first_free_start"
    record = _judged_record(name, _respawning_pouncer_spec(), seeds=400, want=_returns_short_of_the_last)
    index = _first(record, lambda s: not s.round and any(p.kind == "return" for p in s.phase)
                   and bool(_later_starts(record, s)))
    step = record.steps[index]
    back = next(p for p in step.phase if p.kind == "return")
    other = _later_starts(record, step)[0]
    planted = _edit_after(record, index, {back.target: back.eaten, other: step.after.board[back.target]})
    moved = tuple(dataclasses.replace(p, origin=other, target=other, eaten=step.after.board[other])
                  if p is back else p for p in step.phase)
    steps = list(planted.steps)
    steps[index] = dataclasses.replace(steps[index], phase=moved)
    assert "earlier in its order, was free" in _fires(name, _with_steps(planted, steps))


def _line_spec(**rules: Any) -> dict[str, Any]:
    """A knight that eats anything pacing rank 3 back and forth, beside check (N5, SR8h)."""
    line = [f"{f}3" for f in "abcdef"]
    return _six(npcs=[{"piece": "N", "route": line, "turns_back": True}], **rules)


def _royal_square(record: Record, board: bytes, player: int) -> int:
    royal = set(record.engine.spec.royal_types)
    spec = record.engine.spec
    return next(sq for sq, v in enumerate(board)
                if v and spec.owner_of(v) == player and spec.type_of(v) in royal)


def test_the_check_check_fires_on_a_royal_left_where_the_wild_piece_steps() -> None:
    """The mover's king moved, after the record's move, onto an empty square the next phase
    steps a wild piece onto: a move the engine should not have offered under checkmate."""
    from chess_variant_sandbox.engine.coming import coming_phase

    name = "a_royal_is_never_left_where_a_wild_piece_steps"
    record = _judged_record(name, _line_spec(win="checkmate"))

    def landing(step: Step) -> int | None:
        if step.round or record.engine.terminal(step.after) is not None:
            return None
        empty = [s.target for s in coming_phase(record.engine, step.after)
                 if s.kind != "return" and not step.after.board[s.target]]
        return empty[0] if empty else None

    index = _first(record, lambda s: landing(s) is not None)
    step = record.steps[index]
    target = landing(step)
    assert target is not None
    king = _royal_square(record, step.after.board, step.mover)
    planted = _edit_after(record, index, {king: 0, target: step.after.board[king]})
    assert "is left where a wild piece steps next" in _fires(name, planted)


def test_the_safe_goal_check_fires_on_a_goal_a_wild_piece_steps_onto(monkeypatch) -> None:
    """A reached ending held against a phase that, planted, steps onto every square: the engine
    counted a goal its own phase would take."""
    name = "a_safe_goal_is_not_where_a_wild_piece_steps"
    spec = _six(reach={"piece": "N", "squares": [f"{f}3" for f in "abcdef"], "safe": True},
                npcs=[{"piece": "B", "route": [f"{f}4" for f in "abcdef"], "turns_back": True,
                       "mirror": False}])
    record = _judged_record(name, spec, seeds=300)
    every = tuple(PhaseStep(sq, sq, 0, "step") for sq in range(36))
    monkeypatch.setattr("chess_variant_sandbox.engine.coming.coming_phase", lambda engine, state: every)
    assert "a wild piece steps onto next" in _fires(name, record)


def test_the_immunity_check_fires_when_the_taker_is_not_listed() -> None:
    """A deliberately wrong expectation: the record is the engine's, the spec is edited to
    list only a type the capture was not made by."""
    spec = _wild_spec(immune=["R", "N"])
    record = _judged_record("an_immune_wild_piece_falls_only_to_its_takers", spec, seeds=400)
    engine = record.engine
    index = _first(record, lambda s: not s.round and s.before.board[move_to(s.moves[0])]
                   and engine.spec.owner_of(s.before.board[move_to(s.moves[0])]) == WILD)
    taker = engine.spec.piece_ids[engine.spec.type_of(record.steps[index].before.board[move_from(record.steps[index].moves[0])])]
    doctored = copy.deepcopy(record.spec)
    doctored["rules"]["npcs"][0]["immune"] = [t for t in ("R", "N") if t != taker][:1]
    planted = Record(spec=doctored, engine=engine, steps=record.steps, outcome=record.outcome)
    assert "immune to all but" in _fires("an_immune_wild_piece_falls_only_to_its_takers", planted)


def _nobody_spec() -> dict[str, Any]:
    spec = _six()
    spec["setup"]["placements"].append({"piece": "N", "player": "nobody", "squares": ["c3"]})
    return spec


def test_the_standing_check_fires_on_a_standing_piece_that_moved() -> None:
    record = _judged_record("a_standing_piece_stands", _nobody_spec())
    engine = record.engine
    step = record.steps[0]
    at = next(sq for sq, v in enumerate(step.after.board) if v and engine.spec.owner_of(v) == WILD)
    to = _empty_square(step.after.board, {sq for sq, v in enumerate(step.before.board) if v})
    planted = _edit_after(record, 0, {at: 0, to: step.after.board[at]})
    assert "appeared on square" in _fires("a_standing_piece_stands", planted)


def _held_spec(facing: bool = False) -> dict[str, Any]:
    """SR8i: a held pawn-like type nobody owns on c3 (its image on c4), stepping forward and
    taking forward diagonally; with ``facing``, it faces, placed pointing right."""
    spec = _six()
    held: dict[str, Any] = {
        "id": "H", "value": 2, "control": "majority",
        "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
                  {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}],
    }
    placement: dict[str, Any] = {"piece": "H", "player": "nobody", "squares": ["c3"]}
    if facing:
        held["facing"] = {"axes": 4, "initial": "forward"}
        placement["facing"] = "right"
    spec["pieces"].append(held)
    spec["setup"]["placements"].append(placement)
    return spec


def _held_step(record: Record, kind: int) -> int:
    return _first(record, lambda s: not s.round and any(
        m & checks.MOVE_FOREIGN and move_type(m) == kind for m in s.moves
    ))


def test_the_holders_frame_check_fires_on_a_held_move_its_holder_could_not_make() -> None:
    """A held step re-aimed three ranks away: no holder's own pawn reaches it."""
    name = "a_held_piece_moves_in_its_holders_frame"
    record = _judged_record(name, _held_spec(), seeds=300,
                            want=lambda r: any(not s.round and any(
                                m & checks.MOVE_FOREIGN for m in s.moves) for s in r.steps))
    index = _held_step(record, MOVE_NORMAL)
    step = record.steps[index]
    move = step.moves[0]
    origin = move_from(move)
    far = next(sq for sq in range(36) if abs(sq // 6 - origin // 6) >= 3)
    word = encode_move(origin, far, MOVE_NORMAL, 0) | checks.MOVE_FOREIGN
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, moves=(word,))
    assert "could not reach" in _fires(name, _with_steps(record, steps))


def test_the_holders_frame_check_fires_on_a_held_turn_that_left_another_facing() -> None:
    """A held turn whose piece was left pointing elsewhere than its word says."""
    name = "a_held_piece_moves_in_its_holders_frame"
    record = _judged_record(name, _held_spec(facing=True), seeds=300,
                            want=lambda r: any(not s.round and any(
                                m & checks.MOVE_FOREIGN and move_type(m) == checks.MOVE_ROTATE
                                for m in s.moves) for s in r.steps))
    index = _held_step(record, checks.MOVE_ROTATE)
    step = record.steps[index]
    origin = move_from(step.moves[0])
    value = step.after.board[origin]
    spec = record.engine.spec
    other = (spec.facing_of(value) + 1) % spec.n_facings
    wrong = (2 * spec.n_facings + other) * spec.n_types + spec.type_of(value) + 1
    planted = _edit_after(record, index, {origin: wrong})
    assert "held turn" in _fires(name, planted)


def _held_promoting_spec() -> dict[str, Any]:
    """SR8j: a held king-stepper `H` (nobody's on c4, its image on c3) promotes in the middle
    two ranks, its holder's zone for either holder, to a rook, a bishop or the king (a royal,
    never offered to a held piece)."""
    spec = _six()
    spec["pieces"].append({
        "id": "H", "value": 2, "control": "majority",
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
    })
    spec["setup"]["placements"].append({"piece": "H", "player": "nobody", "squares": ["c4"]})
    spec["rules"]["promotion"] = [
        spec["rules"]["promotion"], {"piece": "H", "ranks": [2, 3], "to": ["R", "B", "K"]},
    ]
    return spec


def _held_promotion_record() -> tuple[Record, int]:
    name = "a_held_piece_promotes_as_its_holder_would"
    record = _judged_record(name, _held_promoting_spec(), seeds=300,
                            want=lambda r: any(not s.round and any(
                                m & checks.MOVE_FOREIGN and move_type(m) == MOVE_PROMOTION
                                for m in s.moves) for s in r.steps))
    return record, _held_step(record, MOVE_PROMOTION)


def test_the_held_promotion_check_fires_on_a_promoted_piece_that_joined_its_holder() -> None:
    """A held promotion that left the holder's own rook on the square, not a nobody's one."""
    name = "a_held_piece_promotes_as_its_holder_would"
    record, index = _held_promotion_record()
    step = record.steps[index]
    move = step.moves[0]
    spec = record.engine.spec
    holders = step.mover * spec.n_facings * spec.n_types + checks.move_promotion(move) + 1
    planted = _edit_after(record, index, {move_to(move): holders})
    assert "not a piece nobody owns" in _fires(name, planted)


def test_the_held_promotion_check_fires_on_a_promotion_into_a_royal() -> None:
    """A held promotion re-aimed at the king, which no held piece ever becomes."""
    name = "a_held_piece_promotes_as_its_holder_would"
    record, index = _held_promotion_record()
    step = record.steps[index]
    move = step.moves[0]
    king = record.engine.spec.piece_ids.index("K")
    word = encode_move(move_from(move), move_to(move), MOVE_PROMOTION, king) | checks.MOVE_FOREIGN
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, moves=(word,))
    assert "does not offer a held piece" in _fires(name, _with_steps(record, steps))


def test_the_held_promotion_check_fires_on_an_unpromoted_arrival() -> None:
    """A held pawn offered a plain step onto its holder's last rank under a mandatory rule."""
    name = "a_held_piece_promotes_as_its_holder_would"
    record, index = _held_promotion_record()
    step = record.steps[index]
    move = step.moves[0]
    plain = encode_move(move_from(move), move_to(move), MOVE_NORMAL, 0) | checks.MOVE_FOREIGN
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, offered=(*step.offered, plain))
    assert "without promoting" in _fires(name, _with_steps(record, steps))


def _held_sweeping_spec() -> dict[str, Any]:
    """SR8k: a held rook-sweeper `H` (nobody's on c4, its image on c3), so either player's
    held sweeps run along the middle ranks and files."""
    spec = _six()
    spec["pieces"].append({
        "id": "H", "value": 4, "control": "majority",
        "moves": [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}],
    })
    spec["setup"]["placements"].append({"piece": "H", "player": "nobody", "squares": ["c4"]})
    return spec


def _held_sweep_record() -> tuple[Record, int]:
    name = "a_held_sweep_removes_only_what_its_holder_could_capture"

    def swept(step: Step, spec: Any) -> bool:
        move = step.moves[0]
        return (not step.round and bool(move & checks.MOVE_FOREIGN) and move_type(move) == MOVE_SWEEP
                and any(step.before.board[q] for q in spec.sweep_path(step.before.board, move, step.mover)))

    record = _judged_record(name, _held_sweeping_spec(), seeds=400,
                            want=lambda r: any(swept(s, r.engine.spec) for s in r.steps))
    return record, _first(record, lambda s: swept(s, record.engine.spec))


def test_the_held_sweep_check_fires_on_a_swept_piece_of_the_holders() -> None:
    """A held sweep whose swept piece is planted as the holder's own: its ray should have
    ended there."""
    name = "a_held_sweep_removes_only_what_its_holder_could_capture"
    record, index = _held_sweep_record()
    step = record.steps[index]
    spec = record.engine.spec
    board = step.before.board
    square = next(q for q in spec.sweep_path(board, step.moves[0], step.mover) if board[q])
    holders = step.mover * spec.n_facings * spec.n_types + spec.type_of(board[square]) + 1
    cells = bytearray(board)
    cells[square] = holders
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, before=step.before.with_(board=bytes(cells)))
    assert "its holder could not capture" in _fires(name, _with_steps(record, steps))


def test_the_held_sweep_check_fires_on_a_swept_piece_left_standing() -> None:
    name = "a_held_sweep_removes_only_what_its_holder_could_capture"
    record, index = _held_sweep_record()
    step = record.steps[index]
    board = step.before.board
    square = next(q for q in record.engine.spec.sweep_path(board, step.moves[0], step.mover) if board[q])
    planted = _edit_after(record, index, {square: board[square]})
    assert "left a swept piece standing" in _fires(name, planted)


def test_the_held_sweep_check_fires_on_a_sweeper_that_joined_its_holder() -> None:
    name = "a_held_sweep_removes_only_what_its_holder_could_capture"
    record, index = _held_sweep_record()
    step = record.steps[index]
    move = step.moves[0]
    spec = record.engine.spec
    holders = step.mover * spec.n_facings * spec.n_types + spec.type_of(step.before.board[move_from(move)]) + 1
    planted = _edit_after(record, index, {move_to(move): holders})
    assert "not a piece nobody owns" in _fires(name, planted)


def test_the_allegiance_check_fires_on_a_piece_from_nowhere() -> None:
    record = _judged_record("no_piece_joins_a_side_from_nowhere", _nobody_spec())
    step = record.steps[0]
    to = _empty_square(step.after.board)
    planted = _edit_after(record, 0, {to: step.before.board[0]})
    assert "with 0 entering" in _fires("no_piece_joins_a_side_from_nowhere", planted)


def _market_spec(**purse: Any) -> dict[str, Any]:
    spec = _six(purse={"start": 6, "market": ["N"], **purse})
    _piece(spec, "N")["price"] = 2
    return spec


def test_the_purse_bound_check_fires_on_a_purse_in_debt() -> None:
    record = _judged_record("a_purse_stays_between_nothing_and_its_cap", _market_spec())
    planted = _edit_after(record, 0, purse=(-1, record.steps[0].after.purse[1]))
    assert "purse holds -1" in _fires("a_purse_stays_between_nothing_and_its_cap", planted)


def test_the_buy_check_fires_on_a_price_not_paid() -> None:
    record = _judged_record("a_buy_pays_its_price_onto_its_region", _market_spec())
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_BUY)
    after = record.steps[index].after
    purse = list(after.purse)
    purse[record.steps[index].mover] += 1
    planted = _edit_after(record, index, purse=tuple(purse))
    assert "the price is 2" in _fires("a_buy_pays_its_price_onto_its_region", planted)


def test_the_bounty_check_fires_on_a_payment_nothing_earned() -> None:
    spec = _market_spec(start=0)
    _piece(spec, "R")["bounty"] = 3
    record = _judged_record("a_capture_pays_the_bounty_of_what_it_removed", spec, seeds=300)
    points = list(record.steps[0].after.points)
    points[record.steps[0].mover] += 1.0
    planted = _edit_after(record, 0, points=tuple(points))
    assert "removed 0 of bounty" in _fires("a_capture_pays_the_bounty_of_what_it_removed", planted)


def test_the_castling_check_fires_on_a_partner_left_behind() -> None:
    record = _judged_record("a_castle_moves_both_its_pieces", _six(castling={"king": "K", "rook": "R"}),
                            seeds=400)
    index = _first(record, lambda s: move_type(s.moves[0]) == MOVE_CASTLE)
    step = record.steps[index]
    view = checks.View(record)
    partner, castled, _ = checks._castling_squares(view, step.before.board, step.moves[0], step.mover)
    planted = _edit_after(record, index, {castled: 0, partner: step.before.board[partner]})
    assert "castling partner" in _fires("a_castle_moves_both_its_pieces", planted)


def _shot(record: Record) -> int:
    return _first(record, lambda s: s.before.board[move_to(s.moves[0])] and move_type(s.moves[0]) == MOVE_NORMAL
                  and s.after.board[move_from(s.moves[0])] == s.before.board[move_from(s.moves[0])])


def test_the_shooter_check_fires_on_a_shooter_that_moved() -> None:
    spec = _six(strike={"effect": "capture", "position": "stay", "immune": ["K"]})
    record = _judged_record("a_shot_leaves_its_shooter_where_it_stood", spec,
                            want=lambda r: any(True for _ in [0]) and bool(_try(_shot, r)))
    index = _shot(record)
    planted = _edit_after(record, index, {move_from(record.steps[index].moves[0]): 0})
    assert "the shooter left" in _fires("a_shot_leaves_its_shooter_where_it_stood", planted)


def _try(find: Callable[[Record], int], record: Record) -> int | None:
    try:
        return find(record)
    except StopIteration:
        return None


def test_the_chain_check_fires_on_a_struck_piece_that_did_not_change() -> None:
    chain = [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]
    spec = _six(strike={"effect": "demote", "position": "stay", "chain": chain, "immune": ["K"]})
    record = _judged_record("a_struck_piece_becomes_its_link", spec,
                            want=lambda r: _try(_shot, r) is not None)
    index = _shot(record)
    target = move_to(record.steps[index].moves[0])
    planted = _edit_after(record, index, {target: record.steps[index].before.board[target]})
    assert "did not become" in _fires("a_struck_piece_becomes_its_link", planted)


def test_the_switch_check_fires_on_a_switcher_that_stayed_loyal() -> None:
    spec = _six(strike={"switches_sides": ["N", "B", "R"], "immune": ["K"]})
    record = _judged_record("a_switcher_changes_sides", spec, seeds=300)
    engine = record.engine
    switchers = {engine.spec.piece_ids.index(t) for t in ("N", "B", "R")}
    index = _first(record, lambda s: s.before.board[move_to(s.moves[0])] and s.before.board[move_from(s.moves[0])]
                   and engine.spec.type_of(s.before.board[move_from(s.moves[0])]) in switchers
                   and engine.spec.owner_of(s.before.board[move_to(s.moves[0])]) == 1 - s.mover)
    step = record.steps[index]
    planted = _edit_after(record, index, {move_to(step.moves[0]): step.before.board[move_from(step.moves[0])]})
    assert "not the other player's" in _fires("a_switcher_changes_sides", planted)


def test_the_spy_check_fires_on_a_phase_the_board_does_not_give() -> None:
    spec = _six(simultaneous={})
    _piece(spec, "N")["spy"] = True
    record = _judged_record("the_phase_follows_who_holds_a_spy", spec)
    real = record.engine.phase
    planted = _stubbed(record, phase=lambda state: dataclasses.replace(real(state), name="blind"))
    assert "with spies held" in _fires("the_phase_follows_who_holds_a_spy", planted)


def _sweep_spec() -> dict[str, Any]:
    spec = _six()
    _piece(spec, "R")["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    return spec


def test_the_sweep_check_fires_on_a_swept_piece_left_standing() -> None:
    record = _judged_record("a_sweep_clears_only_what_it_may", _sweep_spec(), seeds=300,
                            want=lambda r: any(move_type(s.moves[0]) == MOVE_SWEEP
                                               and r.engine.spec.sweep_path(s.before.board, s.moves[0])
                                               and any(s.before.board[q] for q in r.engine.spec.sweep_path(s.before.board, s.moves[0]))
                                               for s in r.steps))
    index = _first(record, lambda s: move_type(s.moves[0]) == MOVE_SWEEP and any(
        s.before.board[q] for q in record.engine.spec.sweep_path(s.before.board, s.moves[0])))
    step = record.steps[index]
    swept = next(q for q in record.engine.spec.sweep_path(step.before.board, step.moves[0]) if step.before.board[q])
    planted = _edit_after(record, index, {swept: step.before.board[swept]})
    assert "still stands" in _fires("a_sweep_clears_only_what_it_may", planted)


def test_the_blast_check_fires_on_a_capturer_that_survived() -> None:
    record = _judged_record("a_capture_that_explodes_takes_the_capturer", _six(explode_on_capture={"radius": 1}))
    index = _first(record, lambda s: s.before.board[move_to(s.moves[0])] and move_type(s.moves[0]) == MOVE_NORMAL
                   and record.engine.spec.owner_of(s.before.board[move_to(s.moves[0])]) == 1 - s.mover)
    step = record.steps[index]
    planted = _edit_after(record, index, {move_to(step.moves[0]): step.before.board[move_from(step.moves[0])]})
    assert "survived its own blast" in _fires("a_capture_that_explodes_takes_the_capturer", planted)


def test_the_contact_check_fires_on_a_meeting_that_ignored_its_link() -> None:
    spec = _six(contacts={"links": [{"mover": "P", "onto": "P", "mover_to": "B"}], "immune": ["K"]})
    record = _judged_record("a_meeting_follows_its_link", spec)
    index = _first(record, lambda s: s.before.board[move_to(s.moves[0])] and move_type(s.moves[0]) == MOVE_NORMAL)
    step = record.steps[index]
    target = move_to(step.moves[0])
    planted = _edit_after(record, index, {target: step.before.board[target]})
    assert "of its owner on square" in _fires("a_meeting_follows_its_link", planted)


def test_the_capture_duty_check_fires_on_a_quiet_move_beside_a_capture() -> None:
    record = _judged_record("must_capture_offers_only_captures", _six(must_capture=True))
    engine = record.engine
    index = _first(record, lambda s: any(checks.View(record).is_capture(s.before.board, m, s.mover) for m in s.offered))
    step = record.steps[index]
    origin = next(sq for sq, v in enumerate(step.before.board) if v and engine.spec.owner_of(v) == step.mover)
    quiet = encode_move(origin, _empty_square(step.before.board), MOVE_NORMAL)
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, offered=step.offered + (quiet,))
    assert "beside 1 other moves" in _fires("must_capture_offers_only_captures", _with_steps(record, steps))


def test_the_only_lost_check_fires_on_a_promotion_into_a_piece_never_lost() -> None:
    spec = _six(promotion={"piece": "P", "to": ["R", "B", "N"], "only_lost": True},
                drops={"from_captures": True}, promote_on_entry=True)
    record = _judged_record("a_promotion_takes_only_a_lost_target", spec, seeds=400)
    engine = record.engine
    step = record.steps[0]
    pawn = next(sq for sq, v in enumerate(step.before.board)
                if v and engine.spec.owner_of(v) == step.mover and engine.spec.piece_ids[engine.spec.type_of(v)] == "P")
    word = encode_move(pawn, _empty_square(step.before.board), MOVE_PROMOTION, engine.spec.piece_ids.index("R"))
    steps = list(record.steps)
    steps[0] = dataclasses.replace(step, moves=(word,))
    assert "has not lost" in _fires("a_promotion_takes_only_a_lost_target", _with_steps(record, steps))


def test_the_promoting_sweep_check_fires_on_a_sweeper_that_kept_its_type() -> None:
    spec = _sweep_spec()
    spec["rules"]["promotion"] = {"piece": "R", "to": ["B"]}
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    state = engine.initial_state()
    rook = engine.spec.initial_board[0]
    zone = sorted(engine.spec.promotions[0].squares[0])
    word = encode_move(0, zone[0], MOVE_SWEEP)
    cells = bytearray(state.board)
    cells[0], cells[zone[0]] = 0, rook
    step = Step(0, state, (word,), state.with_(board=bytes(cells), side_to_move=1), (), False, (word,))
    record = Record(spec=spec, engine=engine, steps=[step])
    assert "did not promote" in _fires("a_promoting_sweeper_promotes", record)


def test_the_portal_check_fires_on_a_portal_end_that_vanished() -> None:
    record = _judged_record("portal_ends_are_never_taken", _six(portals=[{"orange": "b3", "blue": "e4"}]))
    engine = record.engine
    end = next(sq for sq, v in enumerate(record.steps[0].after.board) if v and engine.spec.owner_of(v) == WILD)
    planted = _edit_after(record, 0, {end: 0})
    assert "stands 0 times" in _fires("portal_ends_are_never_taken", planted)


def test_the_region_check_fires_on_a_piece_outside_its_region() -> None:
    spec = _six(promotion={"piece": "P", "to": ["R", "N"]})
    _piece(spec, "B")["region"] = [f"{f}{r}" for f in "abcdef" for r in (1, 2, 3)]
    record = _judged_record("a_piece_stays_in_its_region", spec, seeds=300)
    engine = record.engine
    bishop = next(v for v in record.steps[0].before.board
                  if v and engine.spec.owner_of(v) == 0 and engine.spec.piece_ids[engine.spec.type_of(v)] == "B")
    far = next(sq for sq in range(24, 36) if not record.steps[0].after.board[sq] and not record.steps[0].before.board[sq])
    planted = _edit_after(record, 0, {far: bishop})
    message = _fires("a_piece_stays_in_its_region", planted)
    assert "outside its region" in message and checks.KNOWN_PREFIX not in message


def test_the_territory_check_fires_on_a_claim_for_the_wrong_player() -> None:
    record = _judged_record("a_move_claims_where_it_lands", _six(territory={"mode": "repaint"}))
    view = checks.View(record)
    index = _first(record, lambda s: checks._arrival(view, s) is not None)
    square, claimant = checks._arrival(view, record.steps[index])
    colour = bytearray(record.steps[index].after.colour)
    colour[square] = 2 - claimant
    planted = _edit_after(record, index, colour=bytes(colour))
    assert "not by player" in _fires("a_move_claims_where_it_lands", planted)


def test_the_goal_check_fires_when_an_arrival_decides_nothing() -> None:
    spec = _six(reach={"piece": "N", "squares": [f"{f}3" for f in "abcdef"]})
    record = _judged_record("an_arrival_on_a_goal_decides", spec, seeds=300)
    planted = _stubbed(record, terminal=lambda state, moves=None: None)
    assert "the game went on" in _fires("an_arrival_on_a_goal_decides", planted)


def test_the_sight_check_fires_on_a_frozen_piece_offered_a_move() -> None:
    spec = _six()
    _piece(spec, "B")["sight"] = [{"vectors": [[1, 1]], "symmetry": "rotate4", "range": 2,
                                   "effect": "freeze", "whose": "enemy"}]
    record = _judged_record("a_frozen_piece_makes_no_move", spec, seeds=300)
    real = record.engine.effects
    origin = move_from(record.steps[0].offered[0])

    def frozen(state: Any) -> Effects:
        effects = real(state)
        bits = list(effects.bits)
        bits[origin] |= SIGHT_FREEZE
        return dataclasses.replace(effects, bits=tuple(bits))

    assert "frozen piece" in _fires("a_frozen_piece_makes_no_move", _stubbed(record, effects=frozen))


def test_the_extinction_check_fires_on_an_ending_nobody_earned() -> None:
    import random

    spec = _six()
    assert generate._extinction_own(spec, random.Random(3))
    record = _judged_record("running_out_of_one_s_own_kind_ends_it", spec)
    planted = _stubbed(record, terminal=lambda state, moves=None: Outcome(winner=0, reason="extinction"))
    assert "nobody short" in _fires("running_out_of_one_s_own_kind_ends_it", planted)


# ------------------------------------------------- U2: nine pairs the core already played

def test_the_income_check_fires_on_a_coin_income_never_paid() -> None:
    spec = _market_spec(start=2)
    spec["rules"]["income"] = True
    spec["rules"]["scoring"] = {"count": "points"}
    _piece(spec, "R")["bounty"] = 1
    record = _judged_record("income_fills_the_purse_alone", spec)
    step = record.steps[0]
    nxt = step.after.side_to_move
    purse = list(step.after.purse)
    purse[nxt] += 1
    planted = _edit_after(record, 0, purse=tuple(purse))
    assert "their income makes it" in _fires("income_fills_the_purse_alone", planted)
    points = list(step.after.points)
    points[nxt] += 1.0
    planted = _edit_after(record, 0, points=tuple(points))
    assert "points moved" in _fires("income_fills_the_purse_alone", planted)


def _banking_spec() -> dict[str, Any]:
    spec = generate.load_spec(generate.CORPUS / "crossings.json")
    spec["rules"]["reach"]["count"] = 4
    spec["rules"]["extinction"] = {"pieces": ["R"], "min_count": 12}
    return spec


def test_the_bank_check_fires_on_a_side_short_on_the_board_playing_on() -> None:
    record = _judged_record(
        "a_banked_piece_is_counted_off_the_board", _banking_spec(), seeds=200,
        want=lambda r: r.outcome is not None and r.outcome.reason == "extinction",
    )
    planted = _stubbed(record, terminal=lambda state, moves=None: None)
    assert "the game went on" in _fires("a_banked_piece_is_counted_off_the_board", planted)


def test_the_bank_check_fires_on_an_extinction_after_the_count() -> None:
    record = _judged_record("a_banked_piece_is_counted_off_the_board", _banking_spec())
    planted = _edit_after(record, 0, banked=(4, 0))
    planted = _stubbed(planted, terminal=lambda state, moves=None: Outcome(winner=1, reason="extinction"))
    assert "which wins first" in _fires("a_banked_piece_is_counted_off_the_board", planted)


def _gating_spec() -> dict[str, Any]:
    spec = _six(drops={"from_captures": False, "stock": {"N": 1}, "gating": {"ranks": [0]}})
    spec["pieces"].append({"id": "H", "value": 2, "control": "shared", "moves": [
        {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]})
    spec["setup"]["placements"].append({"piece": "H", "player": "nobody", "squares": ["c1"]})
    return spec


def test_the_gate_check_fires_on_a_held_move_that_gates() -> None:
    record = _judged_record("only_a_player_s_unmoved_piece_gates", _gating_spec())
    index = _first(record, lambda s: any(checks.move_gate(m) for m in s.offered))
    step = record.steps[index]
    gated = next(m for m in step.offered if checks.move_gate(m))
    steps = list(record.steps)
    steps[index] = dataclasses.replace(step, offered=step.offered + (gated | checks.MOVE_FOREIGN,))
    assert "foreign move" in _fires("only_a_player_s_unmoved_piece_gates", _with_steps(record, steps))


def test_the_gate_check_fires_on_a_gate_that_left_nothing() -> None:
    record = _judged_record(
        "only_a_player_s_unmoved_piece_gates", _gating_spec(), seeds=300,
        want=lambda r: any(checks.move_gate(s.moves[0]) and move_type(s.moves[0]) != MOVE_CASTLE
                           for s in r.steps),
    )
    index = _first(record, lambda s: checks.move_gate(s.moves[0]) and move_type(s.moves[0]) != MOVE_CASTLE)
    planted = _edit_after(record, index, {move_from(record.steps[index].moves[0]): 0})
    assert "not on the square left" in _fires("only_a_player_s_unmoved_piece_gates", planted)


def test_the_displacement_check_fires_on_a_displaced_piece_that_claimed() -> None:
    spec = generate.load_spec(generate.CORPUS / "leapfrog.json")
    spec["rules"]["territory"] = {"mode": "repaint"}
    record = _judged_record("a_displaced_piece_claims_nothing", spec, seeds=200)
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_DISPLACE)
    step = record.steps[index]
    colour = bytearray(step.after.colour)
    other = move_from(step.moves[0])
    colour[other] = 2 - step.mover if colour[other] == step.mover + 1 else step.mover + 1
    planted = _edit_after(record, index, colour=bytes(colour))
    assert "changed square" in _fires("a_displaced_piece_claims_nothing", planted)


def test_the_window_check_fires_on_a_window_over_a_jumped_piece() -> None:
    spec = _six(en_passant=True)
    _piece(spec, "N")["moves"].append(
        {"kind": "hop", "vectors": [[0, 1], [1, 0]], "symmetry": "rotate4", "mode": "move", "run": True})
    record = _judged_record("a_window_opens_only_over_an_empty_square", spec, seeds=300)
    index = _first(record, lambda s: s.index > 0)
    step = record.steps[index]
    occupied = next(sq for sq, value in enumerate(step.after.board) if value)
    planted = _edit_after(record, index, ep_target=occupied)
    assert "occupied square" in _fires("a_window_opens_only_over_an_empty_square", planted)


def test_the_hire_check_fires_on_a_hired_piece_left_nobody_s() -> None:
    spec = _six(purse={"start": 6})
    spec["pieces"].append({"id": "H", "value": 2, "control": "shared", "price": 2, "moves": [
        {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]})
    spec["setup"]["placements"].append({"piece": "H", "player": "nobody", "squares": ["c3"]})
    record = _judged_record("a_hire_makes_the_piece_the_hirer_s", spec, seeds=300)
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_HIRE)
    step = record.steps[index]
    square = move_to(step.moves[0])
    planted = _edit_after(record, index, {square: step.before.board[square]})
    assert "is not the hirer's" in _fires("a_hire_makes_the_piece_the_hirer_s", planted)


# ------------------------------- R-L6d: the lifted pairs outside batches A to F

_LAME_CIRCLE = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "circle",
                "range": 0}


def _lame_rider_spec() -> dict[str, Any]:
    """PA4: the knight a turning lame rider."""
    spec = _six()
    _piece(spec, "N")["moves"] = [copy.deepcopy(_LAME_CIRCLE)]
    return spec


def _hex_lame_spec() -> dict[str, Any]:
    """X1 (and PA2's lattice): the hex knight a lame jump stating its legs, a hex rose beside it."""
    spec = copy.deepcopy(generate.HEX)
    _piece(spec, "N")["moves"] = [
        {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral12", "legs": [[[0, 1]]]},
        {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral12", "path": "circle"},
    ]
    return spec


def _hex_walls_spec() -> dict[str, Any]:
    """X8: the hex knight's leap stating its legs under `blocks_leaps`, walls in its way."""
    spec = copy.deepcopy(generate.HEX)
    _piece(spec, "N")["moves"][0]["legs"] = [[[1, 0], [0, 1]]]
    spec["rules"]["obstacles"] = {"squares": ["c3"], "blocks_leaps": True}
    return spec


def _unwalled_reach(spec: dict[str, Any]) -> Any:
    """The tables of ``spec`` with its leaps vaulting walls again (blocks_leaps off): where they
    reach and the real ones do not is a leap over a wall leg."""
    from chess_variant_sandbox.engine.compile import compile_spec

    twin = copy.deepcopy(spec)
    twin["rules"]["obstacles"]["blocks_leaps"] = False
    for piece in twin["pieces"]:
        for move in piece["moves"]:
            if move["kind"] == "leap":
                move.pop("legs", None)
    return compile_spec(twin)


@pytest.mark.parametrize("build", [_lame_rider_spec, _hex_lame_spec, _hex_walls_spec])
def test_the_ride_check_fires_on_a_landing_its_table_does_not_reach(build: Callable[[], dict[str, Any]]) -> None:
    spec = build()
    record = _judged_record("a_ride_lands_only_where_its_tables_reach", spec, seeds=200)
    engine = record.engine
    c = engine.spec
    knight = c.piece_ids.index("N")
    twin = _unwalled_reach(spec) if build is _hex_walls_spec else None
    for index, step in enumerate(record.steps):
        if step.round:
            continue
        board = step.before.board
        for origin, value in enumerate(board):
            if not value or c.owner_of(value) != step.mover or c.type_of(value) != knight:
                continue
            view = checks.View(record)
            reach = checks._reach_on(view, c.frame_at(value), knight, origin, board, False, step.mover)
            if twin is not None:  # a leap over a wall leg, which the real table drops
                frame = c.frame_at(value)
                vaults = {sq for targets, _, _ in twin.leaps[frame][knight][origin] for sq in targets}
                far = [sq for sq in sorted(vaults - set(reach)) if not board[sq]]
            else:
                far = [sq for sq in range(c.n_squares) if not board[sq] and sq not in reach
                       and sq not in c.obstacles]
            if not far:
                continue
            word = encode_move(origin, far[0], MOVE_NORMAL)
            steps = list(record.steps)
            steps[index] = dataclasses.replace(step, offered=step.offered + (word,))
            message = _fires("a_ride_lands_only_where_its_tables_reach", _with_steps(record, steps))
            assert "no entry of its table reaches" in message
            return
    raise AssertionError("no knight with a square it cannot reach")


def test_a_sweep_over_an_empty_line_is_read_as_its_slide() -> None:
    """R-L6d's trial: a sweep over an empty line is the slide's word (`MOVE_NORMAL`), so the
    holder's-frame and ride checks read the sweep table too (135 held sweepers' plain moves,
    and three hex lames beside a sweep, failed before)."""
    _judged_record("a_held_piece_moves_in_its_holders_frame", _held_sweeping_spec())


def _pouncing_path_spec() -> dict[str, Any]:
    """N20: a prey pouncer whose own slide turns."""
    spec = _six(npcs=[{"piece": "B", "route": ["b3", "c3", "c4", "b4"], "pounce": "prey"}])
    _piece(spec, "B")["moves"] = [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                                   "path": "zigzag"}]
    return spec


def test_the_pounce_check_fires_on_a_pounce_from_out_of_reach() -> None:
    record = _judged_record("a_pounce_follows_its_pouncer_s_moves", _pouncing_path_spec(), seeds=400)
    index = _first(record, lambda s: any(p.kind == "pounce" for p in s.phase))
    step = record.steps[index]
    which = next(i for i, p in enumerate(step.phase) if p.kind == "pounce")
    pounce = step.phase[which]
    for far in (sq for sq, value in enumerate(step.after.board) if not value and sq != pounce.origin):
        phase = list(step.phase)
        phase[which] = dataclasses.replace(pounce, origin=far)
        steps = list(record.steps)
        steps[index] = dataclasses.replace(step, phase=tuple(phase))
        try:
            _check("a_pounce_follows_its_pouncer_s_moves")(_with_steps(record, steps))
        except CheckFailure as failure:
            if "do not reach" in str(failure):
                return
    raise AssertionError("no square the pouncer could not have pounced from")


def _push_six(**rules: Any) -> dict[str, Any]:
    """The six by six whose rook and knight push where they would take."""
    spec = _six(**rules)
    for piece_id in ("R", "N"):
        for move in _piece(spec, piece_id)["moves"]:
            move["push"] = True
    return spec


def _pushes(step: Step) -> list[int]:
    return [m for m in step.offered if move_type(m) == checks.MOVE_PUSH]


def test_the_push_sight_check_fires_on_a_shielded_piece_pushed() -> None:
    spec = _push_six()
    _piece(spec, "B")["sight"] = [{"vectors": [[1, 1]], "symmetry": "rotate4", "range": 2,
                                   "effect": "shield", "whose": "own"}]
    record = _judged_record("a_push_moves_no_shielded_or_bound_piece", spec, seeds=300)
    index = _first(record, lambda s: bool(_pushes(s)))
    target = move_to(_pushes(record.steps[index])[0])
    real = record.engine.effects

    def shielded(state: Any) -> Effects:
        effects = real(state)
        if state is not record.steps[index].before:
            return effects
        bits = list(effects.bits)
        bits[target] |= checks.SIGHT_SHIELD
        return dataclasses.replace(effects, bits=tuple(bits))

    message = _fires("a_push_moves_no_shielded_or_bound_piece", _stubbed(record, effects=shielded))
    assert "moves the shielded piece" in message


def test_the_pushed_goal_check_fires_when_a_piece_pushed_onto_its_goal_decides_nothing() -> None:
    spec = _push_six(reach={"piece": ["R", "N"], "squares": [f"{f}6" for f in "abcdef"]})
    record = _judged_record("a_pushed_piece_standing_on_its_goal_decides", spec, seeds=300)
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_PUSH)
    step = record.steps[index]
    c = record.engine.spec
    goal = next(sq for sq in c.reach.squares[0] if not step.after.board[sq])
    runner = next(v for v in step.before.board if v and c.owner_of(v) == 0 and c.piece_ids[c.type_of(v)] == "R")
    planted = _stubbed(_edit_after(record, index, {goal: runner}), terminal=lambda state, moves=None: None)
    assert "the game went on" in _fires("a_pushed_piece_standing_on_its_goal_decides", planted)


def test_the_rest_check_fires_on_a_resting_pusher_that_pushes_again() -> None:
    """push.N5: a push is the pusher's move, so the pusher rests."""
    spec = _push_six()
    _piece(spec, "N")["rest"] = 2

    def pushed_then_stood(record: Record) -> bool:
        return any(move_type(s.moves[0]) == checks.MOVE_PUSH and s.index + 2 < len(record.steps)
                   and record.steps[s.index + 2].before.board[move_to(s.moves[0])]
                   == s.after.board[move_to(s.moves[0])]
                   and record.engine.spec.piece_ids[record.engine.spec.type_of(
                       s.before.board[move_from(s.moves[0])])] == "N"
                   for s in record.steps)

    record = _judged_record("a_resting_piece_is_not_moved", spec, seeds=600, want=pushed_then_stood)
    index = next(s.index for s in record.steps
                 if move_type(s.moves[0]) == checks.MOVE_PUSH and s.index + 2 < len(record.steps)
                 and record.steps[s.index + 2].before.board[move_to(s.moves[0])] == s.after.board[move_to(s.moves[0])]
                 and record.engine.spec.piece_ids[record.engine.spec.type_of(
                     s.before.board[move_from(s.moves[0])])] == "N")
    landed = move_to(record.steps[index].moves[0])
    steps = list(record.steps)
    steps[index + 2] = dataclasses.replace(steps[index + 2], moves=(encode_move(landed, landed, MOVE_NORMAL),))
    assert "while it rests" in _fires("a_resting_piece_is_not_moved", _with_steps(record, steps))


def test_the_displacement_check_fires_on_a_pushed_piece_that_claimed() -> None:
    """push.G2: the pusher claims its landing; a pushed piece claims nothing."""
    spec = _push_six(territory={"mode": "repaint"})
    record = _judged_record("a_displaced_piece_claims_nothing", spec, seeds=300)
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_PUSH)
    step = record.steps[index]
    line = record.engine.spec.push_line(step.before.board, step.moves[0])
    pushed_to = next((to for _, to in line if to >= 0), move_from(step.moves[0]))
    colour = bytearray(step.after.colour)
    colour[pushed_to] = 2 - step.mover if colour[pushed_to] == step.mover + 1 else step.mover + 1
    planted = _edit_after(record, index, colour=bytes(colour))
    assert "changed square" in _fires("a_displaced_piece_claims_nothing", planted)


def test_the_bounty_check_fires_on_a_push_off_that_paid_nothing() -> None:
    """push.purse: a pushed-off piece's bounty fills the pusher's purse and points."""
    spec = _push_six(purse={"start": 0, "market": ["N"]})
    _piece(spec, "N")["price"] = 2
    for piece_id in ("R", "B", "N", "P"):
        _piece(spec, piece_id)["bounty"] = 2
    c_engine = Engine(spec)

    def threw_off(record: Record) -> bool:
        return any(move_type(s.moves[0]) == checks.MOVE_PUSH
                   and c_engine.spec.push_taken(s.before.board, s.moves[0]) for s in record.steps)

    record = _judged_record("a_capture_pays_the_bounty_of_what_it_removed", spec, seeds=600, want=threw_off)
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_PUSH
                   and bool(c_engine.spec.push_taken(s.before.board, s.moves[0])))
    step = record.steps[index]
    planted = _edit_after(record, index, points=step.before.points)
    assert "paid the mover 0" in _fires("a_capture_pays_the_bounty_of_what_it_removed", planted)


def test_the_lean_copy_check_fires_on_a_move_the_lean_copy_skipped() -> None:
    """R-L5c2 (`common-path-plan.md` §2.5 item 5): a lean case's offered lists equal the full
    copy's; planted, a lean copy that skipped one move (as a gate whose family `rare_in_use`
    forgets would skip its arm) is caught at the step it was offered."""
    name = "the_lean_copy_offers_what_the_full_copy_offers"
    record = _judged_record(name, _six(), seeds=4)
    assert record.engine.core.lean_path()
    index = _first(record, lambda step: len(step.offered) > 1)
    steps = list(record.steps)
    steps[index] = dataclasses.replace(steps[index], offered=steps[index].offered[:-1])
    assert "the lean copy offered" in _fires(name, _with_steps(record, steps))


def test_the_lean_copy_check_judges_nothing_on_a_full_case() -> None:
    spec = _six(must_capture=True)
    record = play(spec, 0)
    assert not record.engine.core.lean_path()
    assert _check("the_lean_copy_offers_what_the_full_copy_offers")(record) == 0


def test_every_built_case_is_one_the_core_plays() -> None:
    """The run's `except NotImplementedError` is reached by no built case: over a hundred seeds
    (thousands in the record, §5 and U1b's scan) every one passes `bridge.rust_supports`. F5's
    seeds 1254 and 1744 build no case any more (the validator names the hold, `hold.H7`)."""
    from chess_variant_sandbox.engine.bridge import rust_supports
    from chess_variant_sandbox.engine.compile import compile_spec

    held = []
    for seed in [*range(100), 1254, 1744]:
        case = generate.case(seed)
        if case.spec is None:
            continue
        why = rust_supports(compile_spec(case.spec))
        if why is not None:
            held.append((seed, why))
    assert held == []


# ------------------------------------------------- R-L4b stage 3: the capture list's checks

_DEMOTE = {"effect": "demote", "immune": ["K"],
           "chain": [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]}


def _a_capture(record: Record) -> int:
    """The first step whose move takes the other side's piece on its target."""
    c = record.engine.spec
    return _first(record, lambda s: move_type(s.moves[0]) in (MOVE_NORMAL, MOVE_PROMOTION)
                  and s.before.board[move_to(s.moves[0])]
                  and c.owner_of(s.before.board[move_to(s.moves[0])]) == 1 - s.mover and not s.phase)


def test_the_one_place_check_fires_on_a_taken_piece_both_filed_and_standing() -> None:
    # T2: the hand files the taken piece ahead of the strike, so it is struck nowhere.
    spec = _six(capture=[{"do": "hand"}, {"do": "strike", "position": "stay", **_DEMOTE}], drops={})
    record = _judged_record("a_taken_piece_is_in_one_place", spec, want=lambda r: _try(_a_capture, r) is not None)
    index = _a_capture(record)
    step = record.steps[index]
    target = move_to(step.moves[0])
    planted = _edit_after(record, index, {target: step.before.board[target]})
    assert "more pieces" in _fires("a_taken_piece_is_in_one_place", planted)
    hands = [list(h) for h in step.before.hands]
    planted = _edit_after(record, index, hands=tuple(tuple(h) for h in hands))
    assert "filed 0 pieces" in _fires("a_taken_piece_is_in_one_place", planted)


def test_the_spared_piece_check_fires_on_a_capturer_that_outlived_its_blast() -> None:
    # T1.position: a blast that spares the taken piece ahead of a strike placed "swap", which
    # puts the struck piece on the square the capturer left.
    spec = _six(capture=[{"do": "blast", "radius": 1, "immune": ["K"], "spares_taken": True},
                         {"do": "strike", "position": "swap", **_DEMOTE}])

    def linked_victim(r: Record) -> bool:
        index = _try(_a_capture, r)
        if index is None:
            return False
        step, c = r.steps[index], r.engine.spec
        return c.piece_ids[c.type_of(step.before.board[move_to(step.moves[0])])] in ("R", "B", "N")
    record = _judged_record("a_spared_piece_stands_after_its_blast", spec, seeds=300, want=linked_victim)
    index = _a_capture(record)
    step = record.steps[index]
    origin, target = move_from(step.moves[0]), move_to(step.moves[0])
    planted = _edit_after(record, index, {target: step.before.board[origin]})
    assert "survived the blast" in _fires("a_spared_piece_stands_after_its_blast", planted)
    planted = _edit_after(record, index, {origin: 0})
    assert "does not stand" in _fires("a_spared_piece_stands_after_its_blast", planted)


def test_the_two_tables_check_fires_on_a_link_the_capturer_ignored() -> None:
    # T3: a strike placed "swap" and a contact table whose links change every capturer.
    links = [{"mover": a, "onto": b, "mover_to": "P"} for a in ("R", "B", "N") for b in ("R", "B", "N", "P")]
    spec = _six(capture=[{"do": "strike", "position": "swap", **_DEMOTE},
                         {"do": "contacts", "links": links, "immune": ["K"]}])
    record = _judged_record("two_tables_each_read_their_own_meetings", spec, seeds=300)
    c = record.engine.spec
    index = _first(record, lambda s: move_type(s.moves[0]) == MOVE_NORMAL
                   and s.before.board[move_to(s.moves[0])]
                   and c.owner_of(s.before.board[move_to(s.moves[0])]) == 1 - s.mover
                   and c.piece_ids[c.type_of(s.before.board[move_from(s.moves[0])])] in ("R", "B", "N"))
    step = record.steps[index]
    planted = _edit_after(record, index, {move_to(step.moves[0]): step.before.board[move_from(step.moves[0])]})
    assert "its contact link names" in _fires("two_tables_each_read_their_own_meetings", planted)


def test_the_list_sweep_check_fires_on_a_swept_piece_no_entry_reads() -> None:
    # W1.disagree: a strike reads the bishop's path only; the rook's swept pieces simply go.
    spec = _sweep_spec()
    _piece(spec, "B")["moves"] = [{"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4"}]
    spec["rules"]["capture"] = [{"do": "strike", "position": "stay", **_DEMOTE,
                                 "on": ["landing", "path", "collision"], "path_by": ["B"]},
                                {"do": "hand"}]
    spec["rules"]["drops"] = {}
    c = Engine(spec).spec

    def rook_swept(r: Record) -> bool:
        return any(move_type(s.moves[0]) == MOVE_SWEEP and c.type_of(s.before.board[move_from(s.moves[0])])
                   == c.piece_ids.index("R") and any(s.before.board[q] for q in c.sweep_path(
                       s.before.board, s.moves[0], s.mover) if q != move_to(s.moves[0])) for s in r.steps)
    record = _judged_record("a_sweep_follows_the_list", spec, seeds=400, want=rook_swept)
    index = _first(record, lambda s: move_type(s.moves[0]) == MOVE_SWEEP and c.type_of(
        s.before.board[move_from(s.moves[0])]) == c.piece_ids.index("R") and any(
        s.before.board[q] for q in c.sweep_path(s.before.board, s.moves[0], s.mover) if q != move_to(s.moves[0])))
    step = record.steps[index]
    swept = next(q for q in c.sweep_path(step.before.board, step.moves[0], step.mover)
                 if q != move_to(step.moves[0]) and step.before.board[q])
    planted = _edit_after(record, index, {swept: step.before.board[swept]})
    assert "still stands" in _fires("a_sweep_follows_the_list", planted)


def test_the_list_push_check_fires_on_a_pusher_gone_with_no_entry_reading_it() -> None:
    # push.X2.strike: a strike beside a push, reading only the landing.
    spec = _six(capture=[{"do": "strike", "position": "stay", **_DEMOTE}, {"do": "hand"}], drops={})
    for piece in spec["pieces"]:
        if not piece.get("royal"):
            piece["moves"] = [dict(move, push=True) if move.get("mode", "both") != "move" else move
                              for move in piece["moves"]]
    assert validate_spec(spec) == [], validate_spec(spec)
    record = _judged_record("a_pushed_off_piece_is_taken_as_the_list_says", spec, seeds=300)
    index = _first(record, lambda s: move_type(s.moves[0]) == checks.MOVE_PUSH and not s.phase)
    step = record.steps[index]
    planted = _edit_after(record, index, {move_to(step.moves[0]): 0})
    assert "did not stand" in _fires("a_pushed_off_piece_is_taken_as_the_list_says", planted)


def test_the_gate_check_fires_on_a_gated_piece_missing_after_the_blast() -> None:
    # gating.blast: the gated piece enters with the move, before the blast.
    spec = copy.deepcopy(generate.SIX)
    spec["rules"]["drops"] = {"from_captures": False, "stock": {"N": 1}, "gating": {"ranks": [0]}}
    spec["rules"]["explode_on_capture"] = {"radius": 1, "immune": ["K"]}
    _piece(spec, "R")["moves"] = [{"kind": "slide", "vectors": [[0, 1]], "symmetry": "rotate4"}]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    state = engine.make_state({"a1": ("R", 0), "d1": ("K", 0), "a5": ("P", 1), "f6": ("K", 1)},
                              hands=((0, 0, 0, 1, 0), (0, 0, 0, 0, 0)))
    gates = [m for m in engine.legal_moves(state) if checks._move_gate(m) and state.board[move_to(m)]]
    assert gates, [engine.describe_move(state, m) for m in engine.legal_moves(state)]
    after = engine.apply(state, gates[0])
    step = Step(0, state, (gates[0],), after, (), False, tuple(engine.legal_moves(state)))
    record = Record(spec=spec, engine=engine, steps=[step])
    assert _check("a_gated_piece_enters_before_the_blast")(record) == 1
    planted = _edit_after(record, 0, {move_from(gates[0]): 0})
    assert "is not there after the blast" in _fires("a_gated_piece_enters_before_the_blast", planted)


def test_the_cost_check_fires_on_a_capture_charged_for_its_link() -> None:
    # upkeep.U9: a link that changes the capturer after the move is not charged.
    links = [{"mover": a, "onto": b, "mover_to": "R"} for a in ("B", "N", "P") for b in ("R", "B", "N", "P")]
    spec = _six(contacts={"links": links, "immune": ["K"]}, purse={"start": 40, "upkeep": True})
    for piece in spec["pieces"]:
        piece["upkeep"] = 1
    assert validate_spec(spec) == [], validate_spec(spec)
    record = _judged_record("a_move_pays_for_what_it_made_itself", spec, seeds=300)
    index = _a_capture(record)
    step = record.steps[index]
    purses = list(step.after.purse)
    purses[step.mover] -= 1
    planted = _edit_after(record, index, purse=tuple(purses))
    assert "its own type's arrival" in _fires("a_move_pays_for_what_it_made_itself", planted)


@pytest.mark.parametrize("seed", [29216, 11288, 34367, 33806, 43264, 105284])
def test_the_capture_lists_first_soak_seeds_replay_clean(seed: int) -> None:
    """R-L4b3's first soak: a wild piece's bounty went unpaid where the core walked a list built
    without a pay entry (29216, 11288, 34367: `capture.bounty_declared` read only a written
    bounty, where a wild piece pays its type's value unwritten; fixed), a gate from an
    unlimited slot read as a piece from nowhere (33806) and a gated piece a blast took read as
    never gated (43264), two check gaps; and the final soak's one, a piece reborn on its home
    that a sweep under `"path"` had just emptied, placed `"home"` (105284), a third."""
    coverage = Coverage()
    check_case(seed, coverage, rows=_BEFORE_THE_ROUND_ROWS)
    assert coverage.played == 1 and coverage.failures == [], (seed, coverage.failures)


def _round_list_spec(**rules: Any) -> dict[str, Any]:
    """R-L4b5: the six-square army in rounds, a blast beside a strike that sends the struck
    piece home (T1.rounds' pair)."""
    spec = _six(simultaneous={}, explode_on_capture={"radius": 1, "immune": ["K"], "of": ["R", "N", "P"]},
                strike={"effect": "demote", "position": "home", "immune": ["K"],
                        "chain": [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]})
    spec["rules"].update(rules)
    return spec


def _blast_capture(record: Record) -> tuple[int, int, int] | None:
    """The first round in which a player's arrival took a standing enemy piece with neither a
    collision nor the victim moving: (step index, player, target)."""
    for index, step in enumerate(record.steps):
        if not step.round or step.phase:
            continue
        for i in (0, 1):
            move, other = step.moves[i], step.moves[1 - i]
            origin, target = checks.move_from(move), checks.move_to(move)
            taker, victim = step.before.board[origin], step.before.board[target]
            c = record.engine.spec
            if (checks.move_type(move) == checks.MOVE_NORMAL and taker and victim and origin != target
                    and c.owner_of(taker) == i and c.owner_of(victim) == 1 - i
                    and c.explosion is not None and c.type_of(victim) in (c.explosion.of_types or ())
                    and checks.move_from(other) != target and checks.move_to(other) != target):
                return index, i, target
    return None


def test_the_round_list_check_fires_on_a_capturer_that_survived_its_blast() -> None:
    """R-L4b5 (T1.rounds): a planted capturer left on the square its blast cleared."""
    record = _judged_record("a_round_follows_the_list", _round_list_spec(), seeds=300,
                            want=lambda r: _blast_capture(r) is not None)
    found = _blast_capture(record)
    assert found is not None
    index, player, target = found
    step = record.steps[index]
    planted = _edit_after(record, index, {target: step.before.board[checks.move_from(step.moves[player])]})
    assert "still stands there" in _fires("a_round_follows_the_list", planted)


def test_the_round_list_check_fires_on_a_piece_from_nowhere() -> None:
    """R-L4b5 (T4.rounds' check too): a planted piece in a hand after a round (rule 1)."""
    spec = _round_list_spec()
    spec["rules"].pop("strike")
    spec["rules"]["contacts"] = {"links": [{"mover": "N", "onto": "B", "mover_to": "B"}], "immune": ["K"]}
    record = _judged_record("a_round_follows_the_list", spec, seeds=60)
    index = _first(record, lambda s: s.round)
    hands = record.steps[index].after.hands
    planted = _edit_after(record, index, hands=(tuple(h + 1 for h in hands[0]) or (1,), hands[1]))
    assert "more pieces" in _fires("a_round_follows_the_list", planted)


@pytest.mark.parametrize("seed", [16758, 88595])
def test_the_round_list_soak_s_seeds_replay_clean(seed: int) -> None:
    """R-L4b5's soak: a drop from an unlimited slot enters play with its count unmoved, which
    `a_round_follows_the_list` first read as a piece from nowhere (16758, row T4.rounds; 88595,
    R19): a check gap, not an engine fault."""
    coverage = Coverage()
    check_case(seed, coverage)
    assert coverage.played == 1 and coverage.failures == [], (seed, coverage.failures)


def test_every_check_has_a_planted_fault_test() -> None:
    """The rule of this file, held: a new check without a test above fails here."""
    import inspect
    import sys

    source = inspect.getsource(sys.modules[__name__])
    missing = [check.name for check in CHECKS if f'"{check.name}"' not in source]
    assert missing == []


# --------------------------------------------------------------- engine faults found

def test_a_placement_for_nobody_on_a_portal_end_is_refused_or_keeps_the_end() -> None:
    """F1 (seed 630; fixed by R-L6c). The validator refused a portal end on a player's
    placement and accepted one on a placement for nobody; the opening board then had the
    nobody's piece and no portal end. A momentum end is a board byte, so the validator now
    refuses the nobody's placement there as it refuses a player's (`P10.end`)."""
    spec = copy.deepcopy(generate.FIVE)
    spec["setup"]["mirror"] = False
    spec["setup"]["placements"] = [
        {"piece": "K", "player": 0, "squares": ["c1"]}, {"piece": "K", "player": 1, "squares": ["c5"]},
        {"piece": "N", "player": "nobody", "squares": ["a3"]},
    ]
    spec["rules"] = {"win": "royal_capture", "portals": [{"orange": "a3", "blue": "e3"}]}
    errors = validate_spec(spec)
    assert any("a placement for nobody puts a piece on" in error for error in errors), errors


def test_a_promotion_target_with_a_region_never_lands_outside_it() -> None:
    """F2 (seeds 175, 251, 10910; fixed by R-L6c). A pawn promoted into a type whose region
    misses its promotion zone, and the new piece stood outside its region: a target is now
    offered only where it may stand (`Game::promotion_offered`)."""
    spec = _six()
    _piece(spec, "B")["region"] = [f"{f}{r}" for f in "abcdef" for r in (1, 2, 3)]
    if validate_spec(spec):
        return
    engine = Engine(spec)
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c5": ("P", 0)})
    landings = [m for m in engine.legal_moves(state)
                if move_type(m) == MOVE_PROMOTION and engine.spec.piece_ids[checks.move_promotion(m)] == "B"]
    assert landings == []


def test_a_piece_with_a_region_is_never_dropped_outside_it() -> None:
    """F3 (seed 2977; fixed by R-L6c). `compile._compile_drops` read `drops.region` and the
    walls, never `pieces[].region`, so a piece kept to its half was dropped anywhere."""
    spec = _six(drops={"from_captures": False, "stock": {"B": 1}})
    _piece(spec, "B")["region"] = [f"{f}{r}" for f in "abcdef" for r in (1, 2, 3)]
    if validate_spec(spec):
        return
    engine = Engine(spec)
    state = engine.initial_state()
    drops = [m for m in engine.legal_moves(state) if move_type(m) == checks.MOVE_DROP]
    assert drops and all(move_to(m) < 18 for m in drops)


def test_a_player_s_piece_of_a_wild_type_is_bound_by_the_immunity() -> None:
    """F4 (seeds 3960, 5974; fixed by R-L6c). `schema.npc_immune_attackers` exempts the wild
    types by type, so a player's own piece of the wild type took an immune wild piece too; the
    compiler now binds it (`schema.npc_owned_wild_attackers`)."""
    spec = _six(npcs=[{"piece": "N", "route": ["b3", "c3", "c4", "b4"], "immune": ["R"]}])
    assert validate_spec(spec) == []
    engine = Engine(spec)
    knight = engine.spec.piece_ids.index("N")
    state = engine.make_state({"f1": ("K", 0), "f6": ("K", 1), "a1": ("N", 0), "b3": ("N", 2)})
    assert engine.spec.owner_of(state.board[13]) == WILD
    takes = [m for m in engine.legal_moves(state)
             if move_to(m) == 13 and engine.spec.type_of(state.board[move_from(m)]) == knight]
    assert takes == []


def test_a_spec_the_validator_accepts_with_upkeep_beside_the_duck_is_one_the_core_plays() -> None:
    """F5 (R-L6d, seeds 1254 and 1744; fixed by U1b). The validator refused `purse.upkeep`
    beside a long turn by the declared turn, but the duck's compiled turn is long too
    (`compile._compile_duck_turn`), and `bridge.rust_supports` refused it: a valid spec `Engine`
    would not play. The validator now names the hold (`hold.H7`), so either it refuses the spec
    by that name or the core plays it."""
    from chess_variant_sandbox.engine.bridge import rust_supports
    from chess_variant_sandbox.engine.compile import compile_spec
    from chess_variant_sandbox.spec.refusals import refused_ids

    spec = _six(duck={"piece": "D"}, purse={"start": 5, "upkeep": True})
    spec["pieces"].append({"id": "D", "value": 0, "moves": []})
    _piece(spec, "R")["upkeep"] = 1
    errors = validate_spec(spec)
    if errors:
        assert refused_ids(errors) == {"hold.H7"}
        return
    assert rust_supports(compile_spec(spec)) is None


def _gate_beside_a_sweep_ray() -> dict[str, Any]:
    """F7's smallest case: a sweeping rook on a board whose ranks are joined (so two rays of
    one slot reach one square, and every sweep word carries its ray in the tag) beside gating."""
    spec = copy.deepcopy(generate.FIVE)
    spec["pieces"] = [
        {"id": "K", "royal": True, "value": 0,
         "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
        {"id": "R", "value": 5, "moves": [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        {"id": "P", "value": 1, "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]},
    ]
    spec["setup"] = {"placements": [{"piece": "R", "squares": ["a1"]}, {"piece": "K", "squares": ["c1"]},
                                    {"piece": "P", "squares": ["b2"]}]}
    spec["rules"] = {"win": "royal_capture", "wrap": {"ranks": "both"},
                     "drops": {"from_captures": False, "stock": {"R": 1}, "gating": {"ranks": [0]}}}
    return spec


def test_a_sweep_s_ray_is_never_read_as_a_gate() -> None:
    """F7 (seed 3169272; fixed by R-L6e). A `MOVE_SWEEP` in a game with `CompiledSpec.sweep_rays`
    carries its ray's index in bits 25-29, which a game with `drops.gating` read as the gate: a
    moved rook on a3 that sweeps over a5 (ray 1) was offered as gating a 'K' (type 0 + 1), and
    `make` left a second king on a3 that no stock held. The gate is now the tag of a gating kind
    only (`tables.rs::move_gate`): nothing stands on a sweeper's origin after a sweep."""
    spec = _gate_beside_a_sweep_ray()
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.initial_state()
    for text in ("Ra1-a3", "Kc5-d5"):
        state = engine.apply(state, next(m for m in engine.legal_moves(state)
                                         if engine.describe_move(state, m) == text))
    sweeps = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_SWEEP]
    assert sweeps
    left = {engine.describe_move(state, m): engine.apply(state, m).board[move_from(m)] for m in sweeps}
    assert {text: value for text, value in left.items() if value} == {}


def test_a_push_beside_a_majority_type_written_as_an_object_is_refused_or_played() -> None:
    """F8 (seeds 134959, 1106249, 2104486 and 20 more; fixed by R-L6e). `engine_holds._may_hold`
    asked `piece.get("control") in ("majority", "shared")`, the string spelling only, so a type
    written `{"majority": {...}}` never reached the compiled test, the validator accepted the
    spec, and `bridge.rust_supports` refused it ("majority control: a held piece's push"). It now
    reads the normal form through `schema.piece_control`: hold.H2 by name, or the core plays it."""
    from chess_variant_sandbox.engine.bridge import rust_supports
    from chess_variant_sandbox.engine.compile import compile_spec
    from chess_variant_sandbox.spec.refusals import refused_ids

    spec = _six()
    _piece(spec, "R")["moves"][0]["push"] = {"along": "move"}
    spec["pieces"].append({"id": "C", "value": 1, "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}],
                           "control": {"majority": {"vectors": [[1, 0]], "symmetry": "rotate4"}}})
    spec["setup"]["placements"].append({"piece": "C", "player": "nobody", "squares": ["c4"]})
    errors = validate_spec(spec)
    if errors:
        assert "hold.H2" in refused_ids(errors), errors
        return
    assert rust_supports(compile_spec(spec)) is None


def test_a_push_along_a_file_beside_another_type_s_facing_is_refused_or_built() -> None:
    """F9 (seeds 2117229, 3197740; fixed by R-L6e). The validator's D3 (`_check_push_move`) and D8
    read the pushing piece's own facing; the compiler builds every type in every facing frame
    once any type faces a quarter turn, and `_push_step` raised D3 on the pawn's turned step
    (1, 0), a frame no pawn stands in. The spec is sound (no pawn ever faces sideways), so the
    compiler was wrong: a frame a type never stands in pushes nothing now (`_compile_frames`),
    and the spec builds and plays its file push."""
    spec = _six()
    _piece(spec, "P")["moves"][0]["push"] = {"along": "file"}
    bishop = _piece(spec, "B")
    bishop["moves"] = [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
                       {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]
    bishop["facing"] = {"axes": 4, "initial": "forward"}
    assert validate_spec(spec) == []
    engine = Engine(spec)
    pawn = engine.spec.piece_ids.index("P")
    state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("P", 0), "c4": ("P", 1)})
    pushes = [m for m in engine.legal_moves(state)
              if move_type(m) == checks.MOVE_PUSH and engine.spec.type_of(state.board[move_from(m)]) == pawn]
    assert pushes, [engine.describe_move(state, m) for m in engine.legal_moves(state)]
    after = engine.apply(state, pushes[0])
    assert engine.spec.type_of(after.board[move_to(pushes[0])]) == pawn


def test_a_facing_whose_half_turn_changes_nothing_is_refused_or_built() -> None:
    """F10 (seed 2158850; fixed by R-L6e). A rook stepping sideways only (`[1, 0]` by `lateral`)
    given a quarter-turn facing: facing backward it moves exactly as facing forward, which
    `compile` refuses ("turning between them would be a move that does nothing"), and the
    validator accepted. The validator now says it (`schema._check_facing_changes_something`)."""
    spec = copy.deepcopy(generate.FIVE)
    rook = _piece(spec, "R")
    rook["moves"] = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "lateral"}]
    rook["facing"] = {"axes": 4}
    errors = validate_spec(spec)
    assert len(errors) == 1 and "facing backward it moves exactly as facing forward" in errors[0], errors
    rook["moves"][0]["symmetry"] = "none"  # rightward only: four different readings
    assert validate_spec(spec) == []
    Engine(spec)


def test_the_soak_s_check_gaps_replay_clean() -> None:
    """R-L6b's soak (2026-11-15) failed on eight readings the checks lacked, none an engine
    fault: a piece banked off its goal after a switch (160964) or a link (200348), a link
    promoting on entry (173245), a struck nobody's piece placed by `choose` (1102803), a held
    sweeper promoting (1109416), a pouncer taking a gated piece (1198608), a pawn struck in
    passing (2130750), a transit read through a portal end moved first in a round (3178451), and
    an `enemy` screen over a nobody's piece (3186030). Each replays clean."""
    for seed in (160964, 200348, 173245, 1102803, 1109416, 1198608, 2130750, 3178451, 3186030):
        coverage = Coverage()
        check_case(seed, coverage, rows=_BEFORE_THE_LIST)
        assert coverage.played == 1 and coverage.failures == [], (seed, coverage.failures)


@pytest.mark.parametrize("seed", [5116012, 5132971])
def test_a_victim_reborn_on_a_swept_home_replays_clean(seed: int) -> None:
    """R-L6e's fuzz hour (row W1): a sweep beside circe (a strike placed `home`, on `landing`)
    clears its path, then the landing's victim is reborn on its home, which the sweep had just
    emptied (`Rd1xd4 (sweeps d2) (reborn d2)`): a check gap, not an engine fault."""
    coverage = Coverage()
    check_case(seed, coverage, rows=_BEFORE_THE_LIST)
    assert coverage.played == 1 and coverage.failures == [], (seed, coverage.failures)


@pytest.mark.xfail(strict=True, reason="R-L6e's final-build trial (seed 8113368, row run.territory): "
                   "`fuzz.a_piece_that_entered_its_promotion_zone_has_promoted` reads a displaced piece "
                   "in its zone as unpromoted, which RULES.md allows (a displaced piece does not "
                   "promote); an oracle gap in chess_variant_sandbox/fuzz.py, not an engine fault")
def test_a_piece_displaced_into_its_zone_replays_clean() -> None:
    coverage = Coverage()
    check_case(8113368, coverage, rows=_BEFORE_THE_LIST)
    assert coverage.played == 1 and coverage.failures == [], coverage.failures


@pytest.mark.parametrize("seed", [
    3169272, 2100807, 2111219, 2138576,  # F7
    134959, 1106249, 2104486,  # F8 (three of 63)
    2117229, 3197740,  # F9
    2158850,  # F10
])
def test_the_soak_s_fault_seeds_play_clean(seed: int) -> None:
    """R-L6e: the seeds of R-L6b's F7 to F10, fixed: each builds a case the core plays, no check
    fails, none files a fault as known (`checks.KNOWN_FAULTS` is empty again) and none is held."""
    coverage = Coverage()
    check_case(seed, coverage, rows=_BEFORE_THE_LIST)
    assert coverage.played == 1 and coverage.failures == [], (seed, coverage.failures)
    assert not coverage.known and not coverage.held, (seed, coverage.known, coverage.held)
