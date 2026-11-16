"""The eleven engine holds the validator names (U1, `docs/reference/unheld-pairs-plan.md` §1.2,
§1.3; U1b, `docs/reference/refusal-debt.md` §12).

Eleven combinations used to validate clean and were then refused by the engine: seven pairs not
built (H1 to H6: a push beside gating, a held or shared piece, an attackers pouncer or upkeep; two
displacing hops of one piece to one square; two pushes of one piece pushing one landing two ways;
and U1b's H7, upkeep beside the duck, whose compiled turn is long, found by the fuzz as F5) and
four limits of the move word's five-bit tag (C1 to C4: push directions, sweep rays, a sweep
through a fold, a chosen landing). An author heard each late, from a traceback. Now
`schema.validate_spec` asks `engine_hold_refusals` last, and each comes back by name with the
message `refusals.HOLD_REFUSALS` holds (so `refusals.toml` classifies it and `RULES.md` counts it).
`bridge.rust_supports` and the compiler keep their refusals as the second line.

**Why this compiles.** Four of the ten are geometry (which landings a pushing or displacing entry
reaches, how many rays meet, how long a ray is on a joined board), and what geometry means is the
compiler's alone (CLAUDE.md, rule 4). So rather than read the DSL a second time, this asks the
compiled tables the boundary's own questions. It does so only for a spec that has passed every
other check (`validate_spec` calls it only on an empty error list) and only where a cheap reading
of the spec says a hold is possible (`_may_hold`): a push beside one of the four partners, a push
with two pushing moves on one type or on a joined board, or a bound on directions, rays, ray length
or landings past the word. Nothing else compiles, so validating any stored game costs what it did.
The cheap reading reads the spec's normal form (`normal.normal_form`) and every key through the
schema's one reader of it (`piece_control`, `move_push`, `hop_displaces`, ...), never a raw
spelling: R-L6e (F8) found H2 missed where `control` was written as an object.

The compile calls `validate_or_raise`, which calls the validator: `checking()` marks that nested
validation (and every validation `compile_spec` opens with, where the engine's own refusal follows
anyway) so this never recurses and never compiles twice for one build.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from math import gcd
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.refusals import refusal
from chess_variant_sandbox.spec.schema import (
    MAX_LANDING_CANDIDATES,
    SpecError,
    duck_piece,
    hop_displaces,
    move_push,
    npc_pounce,
    piece_control,
    portal_has_folds,
    purse_upkeep,
    push_directions_of,
    push_rule,
)

if TYPE_CHECKING:
    from chess_variant_sandbox.engine.compile import CompiledSpec

# The bounds the move word sets; `engine/moves.py` and `engine/bridge.py` hold the same numbers
# (`MAX_SWEEP_RAYS`, `MAX_SWEEP_THROUGH`, `PUSH_DIRECTIONS_MAX`), and `tests/test_engine_holds.py`
# holds these to them.
TAG_VALUES = 32
LONGEST_FOLD_RAY = 31

# The compiler's words for H6 (`compile._push_landing`), the one hold it raises itself.
_TWO_WAYS = "push the piece on one landing two ways"

_SKIP: ContextVar[bool] = ContextVar("engine_holds_skip", default=False)


@contextmanager
def checking() -> Iterator[None]:
    """Validation inside here asks no engine hold: the caller compiles next, and the engine's
    own refusal (or this module's outer call) says it."""
    token = _SKIP.set(True)
    try:
        yield
    finally:
        _SKIP.reset(token)


def _listed(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _moves(spec: dict[str, Any]) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    for piece in _listed(spec.get("pieces")):
        if isinstance(piece, dict):
            yield piece, [move for move in _listed(piece.get("moves")) if isinstance(move, dict)]


def _facings(spec: dict[str, Any]) -> int:
    most = 1
    for piece, _ in _moves(spec):
        facing = piece.get("facing")
        axes = facing.get("axes", 1) if isinstance(facing, dict) else 1
        if isinstance(axes, int) and not isinstance(axes, bool):
            most = max(most, axes)
    return most


def _push_steps_bound(spec: dict[str, Any]) -> int:
    """At least as many push directions as the compiler can make (C1): every step a pushing
    entry's directions could give, read both ways a move may be reduced, for both players and
    every facing."""
    steps: set[tuple[int, int]] = set()
    for _, moves in _moves(spec):
        for move in moves:
            pushed = move_push(move)
            if pushed is None:
                continue
            for dx, dy in push_directions_of(move):
                along = pushed.get("along")
                # A facing turns the move before its rank or file is read, so either sign.
                if along == "rank":
                    steps |= {(1, 0), (-1, 0)}
                elif along == "file":
                    steps |= {(0, 1), (0, -1)}
                else:
                    divisor = gcd(abs(dx), abs(dy)) or 1
                    steps |= {(dx, dy), (dx // divisor, dy // divisor)}
    return 2 * _facings(spec) * len(steps)


def _sweep_rays_bound(spec: dict[str, Any]) -> int:
    """At least the most sweep entries one type can hold in one slot (C2, C3)."""
    return max(
        (
            sum(len(push_directions_of(move)) for move in moves if move.get("kind") == "sweep")
            for _, moves in _moves(spec)
        ),
        default=0,
    )


def _joined(rules: dict[str, Any]) -> bool:
    wrap = rules.get("wrap")
    return isinstance(wrap, dict) and any(value not in (None, "none", False) for value in wrap.values())


def _landing_bound(rules: dict[str, Any]) -> int:
    """At least as many squares as one chosen landing can offer (C4)."""
    most = 0
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        landing = rule.get("landing") if isinstance(rule, dict) else None
        if isinstance(landing, dict):
            most = max(most, len(push_directions_of(landing)))
    return most


def _collinear_pair(directions: list[tuple[int, int]]) -> bool:
    """Whether two of ``directions`` point the same way (one a positive multiple of the other's
    step), the only way two rays from one square meet on a board with no seam or portal."""
    reduced = [
        (dx // (gcd(abs(dx), abs(dy)) or 1), dy // (gcd(abs(dx), abs(dy)) or 1))
        for dx, dy in directions
    ]
    return len(set(reduced)) < len(reduced)


def _may_hold(spec: dict[str, Any]) -> bool:
    """Whether a cheap reading of ``spec`` leaves any of the eleven possible. Never false where the
    compiled tables would hold one: each test is a superset of the boundary's."""
    rules = spec.get("rules")
    rules = rules if isinstance(rules, dict) else {}
    moves = list(_moves(spec))
    if purse_upkeep(spec) is not None and duck_piece(rules) is not None:
        return True  # H7
    if push_rule(spec) is not None:
        drops = rules.get("drops")
        if isinstance(drops, dict) and drops.get("gating"):
            return True  # H1
        if any(piece_control(piece) in ("majority", "shared") for piece, _ in moves):
            return True  # H2
        if any(npc_pounce(npc) == "attackers" for npc in _listed(rules.get("npcs"))):
            return True  # H3
        if purse_upkeep(spec) is not None:
            return True  # H4
        pushing = [sum(move_push(move) is not None for move in piece_moves) for _, piece_moves in moves]
        if any(count > 1 for count in pushing) or (any(pushing) and _joined(rules)):
            return True  # H6
        if _push_steps_bound(spec) > TAG_VALUES:
            return True  # C1
    for _, piece_moves in moves:
        displacing = [
            direction for move in piece_moves
            if hop_displaces(move)
            for direction in push_directions_of(move)
        ]
        if _collinear_pair(displacing):
            return True  # H5
    rays = _sweep_rays_bound(spec)
    if rays > TAG_VALUES:
        return True  # C2, C3
    if rays and portal_has_folds(rules):
        board = spec.get("board")
        board = board if isinstance(board, dict) else {}
        width, height = board.get("width", 0), board.get("height", 0)
        if isinstance(width, int) and isinstance(height, int):
            longest = width * height if _joined(rules) else max(width, height)
            if longest > LONGEST_FOLD_RAY:
                return True  # C3
    return _landing_bound(rules) > TAG_VALUES  # C4


def held(compiled: CompiledSpec) -> list[str]:
    """The ids of the holds the compiled tables meet, as the boundary reads them."""
    from chess_variant_sandbox.engine.compile import HOP_DISPLACE, MODE_MOVE

    found: list[str] = []
    push = compiled.push
    if push is not None:
        if compiled.drops is not None and compiled.drops.gates is not None:
            found.append("H1")
        if compiled.neutrals is not None and compiled.neutrals.has_majority:
            found.append("H2")
        if compiled.pounce is not None and 2 in compiled.pounce.kind:
            found.append("H3")
        if compiled.purse is not None and compiled.purse.upkeep:
            found.append("H4")
    # H7 (U1b, fuzz find F5): the duck's compiled turn is long whatever `rules.turn` says, and the
    # boundary holds upkeep beside a long turn (`bridge._upkeep_holds`). A declared long turn is
    # `upkeep.U2`'s, refused before this is asked.
    if compiled.duck is not None and compiled.purse is not None and compiled.purse.upkeep:
        found.append("H7")
    # H5: two displacing hop entries of one slot reach one landing (`bridge._displace_holds`).
    if any(
        len(landings) != len(set(landings))
        for frame in compiled.hops for per_type in frame for entries in per_type
        for landings in [[
            target
            for ray, modes, *_, code in entries if code & HOP_DISPLACE
            for target, mode in zip(ray, modes, strict=True) if mode & MODE_MOVE
        ]]
    ):
        found.append("H5")
    if push is not None and len(push.directions) > TAG_VALUES:
        found.append("C1")
    sweeps = [entries for frame in compiled.sweeps for per_type in frame for entries in per_type]
    widest = max((len(entries) for entries in sweeps), default=0)
    if compiled.sweep_rays and widest > TAG_VALUES:
        found.append("C2")
    if compiled.has_sweep and compiled.portals is not None and compiled.portals.folds is not None:
        longest = max((len(ray) for entries in sweeps for ray, *_ in entries), default=0)
        if widest > TAG_VALUES or longest > LONGEST_FOLD_RAY:
            found.append("C3")
    strike = compiled.strike
    if strike is not None and strike.landing and max(
        len(candidates) for per in strike.landing for candidates in per
    ) > MAX_LANDING_CANDIDATES:
        found.append("C4")
    return found


def engine_hold_refusals(spec: dict[str, Any]) -> list[str]:
    """The refusals of the eleven holds ``spec`` meets, by name. Asked by `validate_spec` of a spec
    with no other problem; empty where none is met (or inside `checking`)."""
    # R-L6e (F8): the cheap reading reads the normal form, through the schema's one reader of
    # each key, so no spelling of a key reaches the compiler unseen (an object `control` missed
    # H2: R-L6b's soak, 63 held cases).
    if _SKIP.get() or not _may_hold(normal_form(spec)):
        return []
    from chess_variant_sandbox.engine.compile import compile_spec

    with checking():
        try:
            compiled = compile_spec(spec)
        except SpecError as error:
            if any(_TWO_WAYS in str(message) for message in error.errors):
                return [refusal("hold.H6")]
            return []  # a compiler refusal of another kind: the compiler says it, as before
        except Exception:  # noqa: BLE001 -- a validator answers in a list; compiling says it
            return []
    return [refusal(f"hold.{rid}") for rid in held(compiled)]
