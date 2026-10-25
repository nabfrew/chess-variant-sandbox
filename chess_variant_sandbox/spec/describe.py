"""Render a rule spec as English.

Read by three things that all need the same sentences and want them in two shapes.
The judge and the proposer want one block of prose, because a model should be reacting
to a game rather than to a JSON document. A player wants the parts: this piece, that
rule, next to the board they are looking at. So the parts are the primitive --
``describe_pieces``, ``describe_setup``, ``describe_rules`` -- and ``describe_spec``
composes the block out of them. One statement of what a rule says, two renderings.

It also does quiet double duty as an elegance check: rules that are awkward to state in
a paragraph usually are awkward.

**A rule this module has not learned is a rule nobody is told about**, and that failure
is silent in a way the engine's is not -- the game plays correctly and the page simply
lies about it. It went unnoticed through two schema bumps: a rock-paper-scissors game
was rendered as three identically-moving pieces with no capture relation, won by
annihilation, with its territory rule missing entirely, which is not a description of
that game or of any game. Anything added to ``spec/schema.py`` belongs here in the same
change, and ``tests/test_spec.py`` holds every committed ruleset to it.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import gcd
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.schema import (
    BARE_ROYAL_HOLDER,
    DEFAULT_RANGE,
    REPETITION_NEVER,
    STOCK_UNLIMITED,
    WRAP_PAIRS,
    Sight,
    SpecError,
    absent_squares,
    has_speeds,
    is_simultaneous,
    npc_respawn,
    obstacle_squares,
    path_of,
    piece_rests,
    piece_sights,
    piece_speeds,
    portal_groups,
    portal_stand_in_shape,
    simultaneous_switch,
    spy_ids,
    wrap_stated,
)

if TYPE_CHECKING:
    from chess_variant_sandbox.engine import CompiledSpec

_SYMMETRY_TEXT = {
    "none": "",
    "lateral": " mirrored left and right",
    "rotate4": " in all four rotations",
    "dihedral8": " in all eight directions",
    "rotate6": " in all six turns",
    "dihedral12": " in all six turns and their mirrors",
}

# How many squares a region is spelled out in full before it is summarised. A palace is
# nine and worth reading; a shogi drop region is most of the board and is not.
_SQUARE_LIMIT = 10


@dataclass(frozen=True)
class PieceProse:
    """One piece, said the way a player is told it."""

    id: str
    name: str
    royal: bool
    moves: tuple[str, ...]
    # Everything true of the piece that is not one of its moves: where it may stand,
    # what it is allowed to take, what it reverts to when captured.
    notes: tuple[str, ...] = field(default_factory=tuple)
    # The same moves as the engine states them, one per entry of ``moves``: the vectors,
    # the symmetry, the reach. A reader who wants the rule rather than the sentence gets
    # it from here, and a descriptor nobody has taught the writer to say appears in both.
    descriptors: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "royal": self.royal,
            "moves": list(self.moves),
            "notes": list(self.notes),
            "descriptors": list(self.descriptors),
        }


def describe_spec(spec: dict[str, Any]) -> str:
    """A readable statement of the rules."""
    lines = [f"{spec.get('name', 'untitled')} - a two-player game on {_board_words(spec)}."]
    if spec.get("description"):
        lines.append("")
        lines.append(str(spec["description"]))

    lines.append("")
    lines.append("Pieces:")
    for piece in describe_pieces(spec):
        royal = " (royal - losing it loses the game)" if piece.royal else ""
        lines.append(f"  {piece.id} - {piece.name}{royal}")
        for move in piece.moves:
            lines.append(f"      {move}")
        for note in piece.notes:
            lines.append(f"      {note}")

    lines.append("")
    lines.append("Setup:")
    lines.extend(f"  {line}" for line in describe_setup(spec))

    lines.append("")
    lines.append("Rules:")
    lines.extend(f"  {line}" for line in describe_rules(spec))
    return "\n".join(lines)


def _board_words(spec: dict[str, Any]) -> str:
    """The board, as the opening sentence names it.

    A hex board (schema 14) is named by its cells, the walls cutting it read off the spec,
    because a reader told only "an 11 by 11 board" pictures a square grid. A hexagon
    (Q281) is named by its side, and its cells are the frame's less the absent corners
    and less any wall inside it, so only the walls are counted as cut off.
    """
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    if board.get("lattice") != "hex":
        return f"a {width} by {height} board"
    rules = spec.get("rules") or {}
    orientation = (spec.get("setup") or {}).get("orientation", "mirror")
    absent = absent_squares(board)
    walls = len(obstacle_squares(rules, width, height, orientation) - absent)
    cut = f" with {walls} cut off" if walls else ""
    if absent:
        side = _count((width + 1) // 2)
        return (
            f"a hexagonal board of {width * height - len(absent) - walls} cells, {side} on "
            f"a side{cut}; every cell touches six others"
        )
    article = "an" if str(width).startswith("8") or width in (11, 18) else "a"
    return (
        f"a hexagonal board of {width * height - walls} cells, {article} {width} by "
        f"{height} hex grid{cut}; every cell touches six others"
    )


# ------------------------------------------------------------------------- the parts


def describe_pieces(spec: dict[str, Any]) -> list[PieceProse]:
    """Every piece: what it is, how it moves, and what else is true of it.

    A portal group's moves are said as the piece every portal game listed before spelling
    31 (`schema.portal_stand_in_shape`): ``O - portal``, after the declared pieces, so the
    rules text of a portal game reads as it did.
    """
    hexy = (spec.get("board") or {}).get("lattice") == "hex"
    listed = portal_stand_in_shape(spec)["pieces"]
    fates = _moveless_fates(spec, listed)
    sights = _sight_notes(spec, hexy)
    return [
        _describe_piece(
            piece, hexy, fates.get(piece["id"], _MOVELESS["taken"]), sights.get(piece["id"], ())
        )
        for piece in listed
    ]


def describe_setup(spec: dict[str, Any]) -> list[str]:
    """Where the pieces start, and how the second army is derived from the first."""
    setup = spec["setup"]
    lines = [
        f"{placement['piece']} on {_squares(placement['squares'])}"
        + ("" if "player" not in placement else f" (player {placement['player']} only)")
        # Which way a group starts pointing is as much a fact about the opening position
        # as which squares it stands on, and a reader who is not told it cannot work out
        # what any of those pieces may do on the first move.
        + ("" if "facing" not in placement else f", facing {placement['facing']}")
        for placement in setup["placements"]
    ]
    if setup.get("mirror", True):
        # Two different derivations, and which one a game uses is visible on the board:
        # a mirrored army faces its counterpart down the file it started on, a rotated
        # one does not. Saying "mirrors" for both was wrong for shogi and dobutsu.
        lines.append(
            "The second player's army is the first rotated 180 degrees."
            if setup.get("orientation", "mirror") == "rotate"
            else "The second player's army mirrors the first across the board."
        )
    else:
        lines.append("The two armies are stated separately; neither is derived.")
    return lines


def describe_rules(spec: dict[str, Any]) -> list[str]:
    """Every rule beyond how the pieces move, each stated as a player would be told it."""
    rules = spec.get("rules", {})
    lines = [_describe_win(rules.get("win", "checkmate"))]
    # Schema 26 (Q918a): first of all, because it changes what a turn is more than a long
    # turn does, and every sentence below is read against it.
    lines.extend(_describe_simultaneous(spec, rules))
    # Schema 29 (Q983a): which piece may make a turn's move, read against what a turn is.
    lines.extend(_describe_rest(spec, rules))
    # Schema 30 (Q999a): which piece may move or be taken, read with the rest.
    lines.extend(_describe_sight(spec, rules))

    turn = rules.get("turn")
    if turn and int(turn.get("moves", 1)) > 1:
        # First in the list on purpose: it changes what "a turn" means, so every other
        # sentence below is read differently once you know it.
        lines.append(_describe_turn(turn))

    # Second for the same reason, one level down: a seam changes what a *line* is, so
    # every sentence about how a piece moves is read differently once you know it.
    lines.extend(_describe_wrap(spec, rules))
    # Third, one level down again: a wall is a fact about the board, so it is read
    # before any sentence about a piece; a strike is what every capture below means; a
    # wild piece is a third actor every other sentence has to be read against.
    lines.extend(_describe_obstacles(rules))
    lines.extend(_describe_strike(spec, rules))
    lines.extend(_describe_contacts(spec, rules))
    lines.extend(_describe_npcs(spec, rules))
    # Schema 19: a portal changes where a line goes, so it is read with the walls and
    # the wild pieces, before any sentence about what a capture or a promotion does.
    lines.extend(_describe_portals(spec, rules))

    lines.append(
        ("A player with no legal move at the start of a round " if is_simultaneous(rules)
         else "A player with no legal move ")
        + {"draw": "draws", "loss": "loses", "win": "wins"}[rules.get("stalemate", "draw")]
        + ("; both without one is a draw." if is_simultaneous(rules) else ".")
    )
    if rules.get("checkmate", "loss") == "win":
        lines.append("Being checkmated wins the game.")

    restricted = _capture_relation(spec)
    if restricted:
        lines.append(restricted)

    for promotion in _promotion_rules(rules):
        lines.append(_describe_promotion(promotion))
    if rules.get("promote_on_entry"):
        # Straight after the promotion rules it modifies, because it is read against
        # them: what "promotes" means here is the sentence above.
        lines.append(_describe_entry_promotion(rules))
    if rules.get("en_passant"):
        lines.append("A piece that steps two squares past an enemy that could have taken "
                     "it on the way may be captured as if it had stepped one.")
    if rules.get("castling") and "king_file" in rules["castling"]:
        castling = rules["castling"]
        files = castling["king_file"]
        lines.append(
            f"{castling['king']} may castle with an unmoved {castling['rook']}, "
            f"finishing on file {files['kingside']} or file {files['queenside']}."
        )
    elif rules.get("castling"):
        castling = rules["castling"]
        steps = castling.get("king_steps", 2)
        lines.append(
            f"{castling['king']} may move {steps} squares toward an unmoved "
            f"{castling['rook']}, which jumps to the square it crossed."
        )
    lines.extend(_describe_extra_rules(rules))
    lines.extend(_describe_scoring_rules(spec, rules))
    lines.append(_describe_clocks(rules))
    lines.extend(_describe_ending_order(rules))
    return lines


def describe_changes(parent: dict[str, Any], child: dict[str, Any]) -> list[str]:
    """What a fork changed from its parent, one sentence per change, in the parts' words.

    A diff of the two games' *sentences* rather than of their JSON, for the reason the
    module exists: a changed vector is not something a designer can read, and the
    sentence it produces is. So what a rule means is still said in exactly one place --
    the parts above -- and this only compares what they say. A spelling that changes no
    sentence changes nothing a player is told, and is not listed.

    Pieces are matched by id, so a piece that kept its id and changed its moves is one
    piece changed rather than one removed and another added. Within a piece, and within
    the setup and the rules, a sentence replaced in place is paired with the one it
    replaced, which is how "draws" becoming "loses" reads as one change and not two.
    """
    lines: list[str] = []
    before, after = _board_words(parent), _board_words(child)
    if before != after:
        lines.append(f"The board was {before}; it is now {after}.")

    theirs = {piece.id: piece for piece in describe_pieces(parent)}
    ours = {piece.id: piece for piece in describe_pieces(child)}
    for piece in ours.values():
        old = theirs.get(piece.id)
        if old is None:
            moves = "; ".join(f"“{move}”" for move in piece.moves)
            lines.append(f"A new piece, {piece.name}" + (f", moving {moves}." if moves else "."))
            continue
        name = piece.name
        if old.name != name:
            lines.append(f"{old.name} is now called {name}.")
        if old.royal != piece.royal:
            lines.append(f"{name} is {'now' if piece.royal else 'no longer'} royal.")
        lines.extend(_sentence_changes(name, old.moves, piece.moves, (
            "{subject} moved \u201c{was}\u201d; it now moves \u201c{now}\u201d.",
            "{subject} gains a move: \u201c{now}\u201d.",
            "{subject} loses a move: \u201c{was}\u201d.",
        )))
        lines.extend(_sentence_changes(name, old.notes, piece.notes, (
            "{subject}: \u201c{was}\u201d is now \u201c{now}\u201d.",
            "{subject}, newly: \u201c{now}\u201d.",
            "{subject}, no longer: \u201c{was}\u201d.",
        )))
    lines.extend(
        f"{piece.name} is gone." for piece_id, piece in theirs.items() if piece_id not in ours
    )

    for part, reading in (("Setup", describe_setup), ("Rule", describe_rules)):
        lines.extend(_sentence_changes(part, tuple(reading(parent)), tuple(reading(child)), (
            "{subject} changed: \u201c{was}\u201d is now \u201c{now}\u201d",
            "{subject} added: \u201c{now}\u201d",
            "{subject} dropped: \u201c{was}\u201d",
        )))
    return lines


def _sentence_changes(
    subject: str, before: tuple[str, ...], after: tuple[str, ...], said: tuple[str, str, str]
) -> list[str]:
    """Two lists of sentences, compared in order, pairing one that replaced another.

    ``said`` is how a replacement, an addition and a removal are written, with
    ``{subject}``, ``{was}`` and ``{now}`` filled in -- never spliced in beforehand, so a
    name with a brace in it is a name and not a placeholder.
    """
    replaced, added, removed = said
    lines: list[str] = []
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        was, now = before[i1:i2], after[j1:j2]
        paired = min(len(was), len(now))
        lines.extend(
            replaced.format(subject=subject, was=a, now=b) for a, b in zip(was[:paired], now[:paired], strict=True)
        )
        lines.extend(removed.format(subject=subject, was=a) for a in was[paired:])
        lines.extend(added.format(subject=subject, now=b) for b in now[paired:])
    return lines


def _describe_clocks(rules: dict[str, Any]) -> str:
    """The three clocks in one sentence, with schema 10's ``"never"`` said out loud.

    A repetition rule that cannot fire is not a large number and must not read as one:
    a player told "drawn after 4294967295 repetitions" has been told a number where they
    needed a fact about the game, and *repeating a position is not a draw here* changes
    how an endgame is played. So the clause is dropped from the list and stated on its
    own, which is also what makes the sentence honest when the other two clocks stay.
    """
    repetition = rules.get("repetition_draw", 3)
    clauses = [
        f"{rules.get('no_progress_draw', 100)} turns without a capture or other lasting "
        "change",
        f"or {rules.get('max_plies', 400)} turns in all",
    ]
    # Schema 26: the clocks count each player's move, so a round is two of them.
    rounds = " (a round counts as two)" if is_simultaneous(rules) else ""
    if repetition == REPETITION_NEVER:
        return (
            "Repeating a position is never a draw here. The game is drawn after "
            + ", ".join(clauses)
            + rounds
            + "."
        )
    return (
        f"The game is drawn after {repetition} repetitions, " + ", ".join(clauses) + rounds + "."
    )


# -------------------------------------------------------------------------- pieces


def capturing_types(compiled: CompiledSpec) -> tuple[bool, ...]:
    """Per type: can a piece of it capture anything at all.

    **The one reading of "can this piece capture"** that the rules text and the design
    page's hint (`web/jev_hint.py::facts`) share, so that neither tells a reader a piece
    takes something it has no move to take (Q132, Q426). ``capture_allowed`` is only a
    *permission*: a stone with no ``captures`` list is permitted everything and can
    exercise none of it.

    Two ways to capture, and both are read off the compiled tables:

    - **A capture move.** ``agents/evaluation.py::has_capture_reach``, imported rather
      than written again: the frame-0 ``modes & 2`` walk that the evaluator's prey pool
      (``prey_types``) reads too.
    - **A walking wild piece.** A wild piece never reads its movement table on the board;
      it eats what stands where it steps next if its type may take it (``npc_step_of`` in
      `game.rs`). So a type that walks a route of more than one square captures there,
      whatever its moves say. A one-square route never steps and never eats.
    """
    from chess_variant_sandbox.agents.evaluation import has_capture_reach

    walkers: set[int] = set()
    if compiled.npcs is not None:
        walkers = {
            piece
            for square, (piece, step) in enumerate(
                zip(compiled.npcs.piece, compiled.npcs.next, strict=True)
            )
            if piece >= 0 and step not in (-1, square)
        }
    return tuple(
        piece_type in walkers or has_capture_reach(compiled, piece_type)
        for piece_type in range(compiled.n_types)
    )


# What a piece with no moves is told about leaving its square, by what can remove it.
# The first is the case the sentence was written for (Q51): a stone anything may take.
# The other two are a stone nothing may take (the repaired `stones` and `three-stones`,
# Q131), where "until it is taken" told readers and the design page's hint a false rule
# (Q132): alone, and beside an explosion that can still reach it.
_MOVELESS = {
    "taken": "never moves: once it is on a square it stays there until it is taken",
    "kept": "never moves, and nothing can capture it: once it is on a square it stays "
            "there for the rest of the game",
    "blast": "never moves, and nothing can capture it: once it is on a square it stays "
             "there until an explosion beside it destroys it",
    # I1 (`docs/plan-pieces-moves.md` §4.3): a royal that never moves, by the win it is under.
    "royal_capture": "never moves, so it cannot run from an attack: it stays on its square, "
                     "and taking it is how the game is won (see the rules)",
    "checkmate": "never moves, so a check on it can be answered only by taking the checking "
                 "piece or blocking its line, never by stepping away",
}


def _moveless_fates(spec: dict[str, Any], listed: list[dict[str, Any]]) -> dict[str, str]:
    """For each piece with no moves, the ``_MOVELESS`` sentence that is true of it.

    Keyed by piece id, and empty when every piece moves, which is every spec but a
    handful, so nothing is compiled for them. A piece is taken when some type that can
    capture (``capturing_types``) may take it, and otherwise it stays, unless an explosion
    can reach it. A spec that does not compile keeps the first sentence, which is what
    every moveless piece was told before the other two existed.
    """
    moveless = [piece["id"] for piece in listed if not piece.get("moves")]
    if not moveless:
        return {}
    from chess_variant_sandbox.engine import compile_spec

    try:
        compiled = compile_spec(spec)
    except SpecError:
        return {}
    capturing = capturing_types(compiled)
    blast = compiled.explosion
    fates: dict[str, str] = {}
    royal = {piece["id"] for piece in listed if piece.get("royal")}
    win = (spec.get("rules") or {}).get("win", "checkmate")
    for piece_id in moveless:
        victim = compiled.piece_ids.index(piece_id)
        if piece_id in royal and win in ("royal_capture", "checkmate"):
            fates[piece_id] = _MOVELESS[win]
        elif any(
            capturing[attacker] and compiled.capture_allowed[attacker][victim]
            for attacker in range(compiled.n_types)
        ):
            fates[piece_id] = _MOVELESS["taken"]
        elif blast is not None and victim not in blast.immune_types:
            fates[piece_id] = _MOVELESS["blast"]
        else:
            fates[piece_id] = _MOVELESS["kept"]
    return fates


def _describe_piece(
    piece: dict[str, Any],
    hexy: bool = False,
    moveless: str = _MOVELESS["taken"],
    sights: Sequence[str] = (),
) -> PieceProse:
    notes: list[str] = []
    region = piece.get("region")
    if region:
        notes.append(f"may never leave {_squares(region)}")
    captures = piece.get("captures")
    if isinstance(captures, list):
        # Empty and absent are different, and the difference is the whole feature: a
        # piece with no entry takes anything, a piece with an empty list takes nothing
        # and still blocks lines.
        notes.append(
            f"may capture only {', '.join(captures)}" if captures
            else "may never capture anything, and still blocks a line like any piece"
        )
    reverts = piece.get("promoted_from")
    if reverts:
        notes.append(f"reverts to {reverts} when captured")
    if piece.get("spy") is True:
        # Schema 27 (Q935a): the rules say what the spy does; the piece says it is one.
        notes.append("is a spy: who still has one decides how play goes (see the rules)")
    rest = piece.get("rest")
    if isinstance(rest, int) and not isinstance(rest, bool) and rest > 0:
        # Schema 29 (Q983a): the rules say what a rest does; the piece says its own.
        notes.append(
            f"rests {rest} turn{'s' if rest > 1 else ''} after it moves (see the rules)"
        )
    speed = piece.get("speed")
    if isinstance(speed, int) and not isinstance(speed, bool) and speed > 0:
        # Schema 28 (Q943a): the rules say what a speed does; the piece says its own.
        notes.append(
            f"has speed {speed}: in a round its move is made before a slower piece's "
            "(see the rules)"
        )
    # Schema 30 (Q999a): the rules say what an effect does; the piece says what it sees.
    # Before the facing's note, whose "forward above" covers a sight's directions too.
    notes.extend(sights)
    facing = piece.get("facing")
    if isinstance(facing, dict):
        # Said as a rule a player could act on rather than as a field: what the piece
        # has, what it costs to change, and where it starts. The last clause is dropped
        # when it starts facing forward, which is what a reader assumes anyway.
        turns = (
            "any of the four quarter turns"
            if facing.get("axes") == 4
            else "forward or backward"
        )
        initial = facing.get("initial", "forward")
        if facing.get("turns", True) is False:
            # Schema 25 (Q869): the facing is the setup's, and the sentence says where a
            # piece that was never placed points, since that is the only other way in.
            notes.append(
                f"points {turns}, and forward above means whichever way it points; no "
                "move turns it, so each one points the way it was placed for the whole "
                f"game, and one that enters play later points {initial}"
            )
        else:
            where = "" if initial == "forward" else f", and starts facing {initial}"
            notes.append(
                f"points {turns}, and forward above means whichever way it points; instead "
                f"of moving it may spend a whole move turning to face another way{where}"
            )
    moves = piece.get("moves") or []
    if not moves:
        # Schema 10 (Q51). Without this the piece renders as a name and nothing else,
        # which reads as an omission rather than as the rule -- and *it never moves* is
        # the whole of what this piece is. Whether it is ever taken is a fact about the
        # whole spec, so `describe_pieces` decides which sentence it gets.
        notes.insert(0, moveless)
    descriptors = tuple(_move_descriptor(move) for move in moves)
    return PieceProse(
        id=piece["id"],
        name=str(piece.get("name", piece["id"])),
        royal=bool(piece.get("royal")),
        moves=tuple(
            _move_sentence(move, hexy) or _one_offset_line(move, hexy) or descriptor
            for move, descriptor in zip(moves, descriptors, strict=True)
        ),
        notes=tuple(notes),
        descriptors=descriptors,
    )


def _move_descriptor(move: dict[str, Any]) -> str:
    kind = move["kind"]
    symmetry = move.get("symmetry", "none")
    directions = set()
    for vector in move["vectors"]:
        directions |= expand_symmetry((vector[0], vector[1]), symmetry)

    vectors = ", ".join(f"({x:+d},{y:+d})" for x, y in move["vectors"])
    mode = {"move": "may only move", "capture": "may only capture", "both": "moves and captures"}[
        move.get("mode", "both")
    ]

    max_range = move.get("range", 0 if kind in ("slide", "hop", "sweep") else 1)
    min_range = move.get("min_range", 1)
    # Q1007a: a slide over a vector longer than one square is a rider (the nightrider), which
    # nothing stops but a piece where it would land; "until blocked" read as the lame rider.
    long_vector = any(max(abs(int(v[0])), abs(int(v[1]))) > 1 for v in move["vectors"])
    if kind == "leap":
        reach = "jumping directly, ignoring anything in between"
    elif kind == "lame" and max_range != 1:
        # Q1007a: the lame rider, jump after jump, each stoppable at its leg.
        jumps = (
            "any number of jumps" if max_range == 0
            else f"up to {max_range} jumps"
        ) + (f", stopping no nearer than jump {min_range}" if min_range > 1 else "")
        reach = (
            f"leaping again and again ({jumps}), each jump stopped by a piece standing on "
            "the square it passes, and the ride by a piece where it lands"
        )
    elif kind == "lame":
        # Not "path must be clear": the leap has exactly one square that can stop it,
        # and which square that is (the leg, the eye) is the rule people actually state.
        reach = "leaping, but stopped by a piece standing on the square it passes"
    elif kind == "slide" and long_vector and max_range != 1 and path_of(move) == "straight":
        jumps = "any number of jumps" if max_range == 0 else (
            f"exactly {max_range} jumps" if min_range == max_range
            else f"up to {max_range} jumps"
        )
        reach = (
            f"leaping again and again ({jumps}), stopped only by a piece on a square it "
            "would land on"
        )
    elif kind == "sweep":
        if max_range == 0:
            span = "any distance"
        elif min_range == max_range:
            span = f"exactly {max_range} square{'s' if max_range > 1 else ''}"
        else:
            span = f"up to {max_range} squares"
        reach = (
            f"jumping {span} along the line, and every piece it passes over, its own "
            "included, is removed; a piece it may not capture stops it"
        )
    elif kind == "slide" and path_of(move) != "straight":
        # T1: a turning path, counted in steps.
        steps = "any number of steps" if max_range == 0 else f"up to {max_range} steps"
        turn = (
            "turning to the next direction round at every step, either way"
            if path_of(move) == "circle"
            else "turning to a neighbouring direction and back at every step, a zigzag"
        )
        reach = f"{steps}, {turn}, stopped by a piece on a square it lands on"
    elif max_range == 0:
        reach = "any distance until blocked"
    elif min_range == max_range:
        reach = f"exactly {max_range} square{'s' if max_range > 1 else ''}, path must be clear"
    else:
        reach = f"up to {max_range} squares"

    if kind == "hop":
        screen = {"own": "one of your own pieces", "enemy": "one enemy piece"}.get(
            str(move.get("screen", "any")), "exactly one piece of either side"
        )
        screen_range = int(move.get("screen_range", 0))
        if screen_range == 1:
            where = ", landing immediately past it"
        elif screen_range > 1:
            where = f", landing at most {screen_range} squares past it"
        else:
            where = f", then {reach}"
        reach = f"leaping {screen}{where}"

    text = f"{mode} along {vectors}{_SYMMETRY_TEXT[symmetry]}, {reach}"
    if len(directions) > len(move["vectors"]):
        text += f" [{len(directions)} directions]"

    clauses = _condition_clauses(move.get("condition"))
    if clauses:
        text += " - " + " and ".join(clauses)
    return text


def _condition_clauses(condition: dict[str, Any] | None) -> list[str]:
    """When this move is available at all.

    Returns a list rather than a string because an empty one has to be distinguishable
    from an empty *clause*: a condition carrying only a region used to fall through
    every branch here and leave a dangling " - " on the end of the sentence, which is
    what every xiangqi soldier read like.
    """
    if not condition:
        return []
    clauses = []
    if "from_rank" in condition:
        clauses.append(f"only from its own rank {condition['from_rank'] + 1}")
    if condition.get("first_move"):
        clauses.append("only if it has never moved")
    if condition.get("region"):
        clauses.append(f"only from {_squares(condition['region'])}")
    return clauses


# ------------------------------------------------------------------ moves, in words
#
# The same descriptor a player would say out loud: *one square forward, not to take*,
# *any distance along all four diagonals*, *leaps two and one, in all eight directions*.
# ``_move_descriptor`` above stays exactly as it was and is what the page keeps behind
# the (i), because a sentence is a reading and the vectors are the rule.
#
# Two things this must not do. It must not re-derive geometry: a direction is
# ``expand_symmetry`` on the spec's own vectors and nothing here works out a square.
# And it must not guess -- a descriptor no phrase fits returns ``None`` and the reader
# gets the vectors, because a wrong sentence about how a piece moves is worse than a
# vector a reader has to puzzle over.
#
# The words are the designer's, from ``web/page/drawing.py``'s ``FIT_STEP_NAMES`` and
# ``fitVectorWords``, so a piece drawn in the designer and the same piece read on its
# game's page are said the same way.

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

# One step, named on a square board. Forward is +y and right is +x, both in the
# player's own orientation, which is how a spec's vectors are written.
_STEP_NAMES = {
    (0, 1): "forward", (0, -1): "back", (-1, 0): "left", (1, 0): "right",
    (-1, 1): "diagonally forward and left", (1, 1): "diagonally forward and right",
    (-1, -1): "diagonally back and left", (1, -1): "diagonally back and right",
}

# A whole set of steps named at once, where the set is a shape a player already has a
# word for. Keyed by the directions themselves, so the two vectors of a chess queen and
# the eight of anything else reach the same phrase.
_ORTHOGONAL = frozenset({(0, 1), (0, -1), (1, 0), (-1, 0)})
_DIAGONAL = frozenset({(1, 1), (-1, 1), (1, -1), (-1, -1)})
_UNIT_FAMILIES = {
    _ORTHOGONAL: "forward, back, left and right",
    _DIAGONAL: "along all four diagonals",
    _ORTHOGONAL | _DIAGONAL: "in all eight directions",
    frozenset({(1, 0), (-1, 0)}): "left and right",
    frozenset({(0, 1), (0, -1)}): "forward and back",
    frozenset({(1, 1), (-1, 1)}): "diagonally forward, either way",
    frozenset({(1, -1), (-1, -1)}): "diagonally back, either way",
}

# The hex lattice (schema 14), in the axial convention the validator polices: file `q`
# is +x and rank `r` is +y. The board draws a rank half a cell to the right of the one
# below it, so +y points forward and right rather than straight up, and a cell has no
# neighbour straight ahead at all. The names follow the drawing.
_HEX_STEP_NAMES = {
    (1, 0): "right", (-1, 0): "left",
    (0, 1): "forward-right", (-1, 1): "forward-left",
    (1, -1): "back-right", (0, -1): "back-left",
}
_HEX_SIDES = frozenset(_HEX_STEP_NAMES)
_HEX_DIAGONALS = frozenset({(1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)})
# The twelve cells three steps out that lie along none of the twelve lines above.
_HEX_OFF_LINE = frozenset(
    {(1, 2), (2, 1), (3, -1), (3, -2), (2, -3), (1, -3),
     (-1, -2), (-2, -1), (-3, 1), (-3, 2), (-2, 3), (-1, 3)}
)


@dataclass(frozen=True)
class _Aim:
    """Where a move goes, in words.

    ``step`` separates the two grammars. A step is a *direction* -- "forward", "along
    all four diagonals" -- so a distance goes in front of it and ``scale`` says how many
    squares one step of it covers. Anything else is a whole jump and stands alone.

    ``unit`` is the one kind of jump a line can still be built out of: an oblique offset
    such as two-and-one, which is a shape rather than a direction and yet repeats. A
    slide along it is a line of those offsets, so ``_oblique_span_words`` says it that
    way. Every other jump-shaped aim is a set of unrelated offsets and no line runs
    along it.
    """

    words: str
    step: bool = True
    scale: int = 1
    unit: bool = False


def _count(number: int) -> str:
    return _NUMBER_WORDS.get(number, str(number))


def move_sentence(move: dict[str, Any], hexy: bool = False) -> str:
    """One way of moving, said the way a player would say it.

    The public form of ``_move_sentence``, with the fallback ``describe_pieces`` already
    uses: a descriptor no phrase fits gets its vectors back rather than a guess, because
    a wrong sentence about how a piece moves is worse than a vector a reader has to
    puzzle over. The designer offers these as the readings of a click, so a piece taught
    in the editor and the same piece read on its game's page are said in one voice.
    """
    return (_move_sentence(move, hexy) or _one_offset_line(move, hexy)
            or _move_descriptor(move))


def _one_offset_line(move: dict[str, Any], hexy: bool) -> str | None:
    """A slide one oblique offset long, said as the leap it is, for a reader of a piece.

    Nothing stands between the two ends of an oblique step, so a nightrider cut to one
    step is the leap. ``_move_sentence`` says nothing for it on purpose, so that
    teach-a-move does not offer one rule in two sentences; a reader of the piece has no
    second sentence to confuse it with, and the vector fallback it got instead said
    *exactly 1 square, path must be clear*, two claims the move makes neither of (Q541).
    """
    if move.get("kind") != "slide" or int(move.get("range", DEFAULT_RANGE["slide"])) != 1:
        return None
    aim = _aim(move, hexy)
    if aim is None or not aim.unit:
        return None
    leap = {key: value for key, value in move.items() if key not in ("range", "min_range")}
    return _move_sentence({**leap, "kind": "leap"}, hexy)


def plain_move_sentence(move: dict[str, Any], hexy: bool = False) -> str | None:
    """The same sentence, or ``None`` where this vocabulary has no phrase for it.

    For a caller that would rather say nothing than fall back to the vectors -- the
    designer's menu of what a click could mean, where a reading nobody can read is worse
    than one reading fewer.
    """
    return _move_sentence(move, hexy)


def _move_sentence(move: dict[str, Any], hexy: bool = False) -> str | None:
    """One descriptor as a player would say it, or ``None`` if no phrase fits."""
    aim = _aim(move, hexy)
    if aim is None:
        return None
    kind = move["kind"]
    body = {
        "slide": _slide_words,
        "leap": _leap_words,
        "lame": _leap_words,
        "hop": _hop_words,
        "sweep": _sweep_words,
    }[kind](move, aim)
    if body is None:
        return None

    mode = {"move": ", not to take", "capture": ", only to take"}.get(
        str(move.get("mode", "both")), ""
    )
    clauses = _condition_clauses(move.get("condition"))
    only = (", and " + " and ".join(clauses)) if clauses else ""
    return f"{body}{mode}{only}"


def _aim(move: dict[str, Any], hexy: bool) -> _Aim | None:
    """The directions this descriptor reaches in, as one phrase.

    The whole set first, because a chess queen is two vectors and *in all eight
    directions* is one thing a player knows; per vector only when the union is not a
    shape with a name.
    """
    symmetry = str(move.get("symmetry", "none"))
    vectors = [(int(vector[0]), int(vector[1])) for vector in move["vectors"]]
    whole: set[tuple[int, int]] = set()
    for vector in vectors:
        whole |= expand_symmetry(vector, symmetry)
    together = _family(frozenset(whole), hexy)
    if together is not None:
        return together

    parts = [_family(frozenset(expand_symmetry(v, symmetry)), hexy) for v in vectors]
    if not parts or any(part is None for part in parts):
        return None
    said = [part for part in parts if part is not None]
    if len({(part.step, part.scale, part.unit) for part in said}) > 1:
        # Two grammars in one sentence, or two step lengths: sayable, but not by a rule
        # this simple, and the vectors are a better answer than a confusing sentence.
        return None
    return _Aim(", and ".join(part.words for part in said), said[0].step, said[0].scale,
                said[0].unit)


def _family(directions: frozenset[tuple[int, int]], hexy: bool) -> _Aim | None:
    """A set of directions as one phrase, or ``None`` where no phrase fits."""
    if not directions:
        return None
    if hexy:
        return _hex_family(directions)
    scale = _step_scale(directions)
    if scale:
        words = _unit_words(frozenset((x // scale, y // scale) for x, y in directions))
        return None if words is None else _Aim(words, True, scale)
    run = _step_run(directions)
    if run is not None:
        reach, units = run
        words = _unit_words(units)
        # Every distance from one out to `reach`, written as a separate vector each. It
        # is one jump with a reach rather than a line, so it carries its own distance.
        if words is not None:
            return _Aim(f"up to {_count(reach)} squares {words}", False)
    return _oblique_family(directions)


def _unit_words(units: frozenset[tuple[int, int]]) -> str | None:
    """A set of single steps, named as a shape where there is a word for one."""
    if units in _UNIT_FAMILIES:
        return _UNIT_FAMILIES[units]
    order = list(_STEP_NAMES)
    if len(units) > 3 or any(unit not in _STEP_NAMES for unit in units):
        return None
    named = [_STEP_NAMES[unit] for unit in sorted(units, key=order.index)]
    return " or ".join([", ".join(named[:-1]), named[-1]] if len(named) > 2 else named)


def _step_scale(directions: frozenset[tuple[int, int]]) -> int | None:
    """How many squares one step covers, when every direction is a multiple of a step.

    ``None`` whenever they are not -- a knight's vector reduces to itself and is a shape
    rather than a direction, and directions of two different lengths have no one answer.
    """
    scales = set()
    for x, y in directions:
        scale = gcd(abs(x), abs(y))
        if not scale or max(abs(x), abs(y)) // scale != 1:
            return None
        scales.add(scale)
    return scales.pop() if len(scales) == 1 else None


def _step_run(directions: frozenset[tuple[int, int]]) -> tuple[int, frozenset[Any]] | None:
    """A jump spelled out as every distance along the same steps, one vector each.

    Refused unless the distances run one, two, three ... with no gap and every direction
    reaches every one of them, because *up to seven squares* would otherwise be a claim
    about squares the piece cannot reach.
    """
    units: set[tuple[int, int]] = set()
    scales: set[int] = set()
    for x, y in directions:
        scale = gcd(abs(x), abs(y))
        if not scale or max(abs(x), abs(y)) // scale != 1:
            return None
        units.add((x // scale, y // scale))
        scales.add(scale)
    if len(scales) < 2 or scales != set(range(1, max(scales) + 1)):
        return None
    if len(directions) != len(units) * len(scales):
        return None
    return max(scales), frozenset(units)


def _oblique_family(directions: frozenset[tuple[int, int]]) -> _Aim | None:
    """A jump that lies along no line, such as two squares one way and one the other.

    Said by its shape and how many ways it is turned, which is read back off the
    symmetry group that produces exactly this set.
    """
    shapes = {tuple(sorted((abs(x), abs(y)))) for x, y in directions}
    if len(shapes) != 1:
        return None
    low, high = shapes.pop()
    if not low or low == high:
        return None
    for symmetry in ("dihedral8", "rotate4", "lateral", "none"):
        for base in sorted(directions):
            if expand_symmetry(base, symmetry) != directions:
                continue
            if symmetry == "dihedral8":
                return _Aim(f"{_count(high)} and {_count(low)}, in all eight directions",
                            False, unit=True)
            if symmetry == "rotate4":
                return _Aim(f"{_count(high)} and {_count(low)}, turned all four ways",
                            False, unit=True)
            across, along = base
            forward = f"{_count(abs(along))} {'forward' if along > 0 else 'back'}"
            if symmetry == "lateral":
                return _Aim(f"{forward} and {_count(abs(across))} to either side",
                            False, unit=True)
            return _Aim(
                f"{forward} and {_count(abs(across))} {'right' if across > 0 else 'left'}",
                False,
                unit=True,
            )
    return None


def _hex_family(directions: frozenset[tuple[int, int]]) -> _Aim | None:
    """The same, on a hexagonal board, where the square board's words do not hold.

    Six sides rather than four, six diagonals that do not bisect them, and a jump set of
    twelve. Anything else falls back, because a hex direction named by a square board's
    compass would be a wrong sentence rather than an awkward one.
    """
    if directions == _HEX_SIDES:
        return _Aim("in all six directions")
    if directions == _HEX_DIAGONALS:
        return _Aim("along all six diagonals")
    if directions == _HEX_SIDES | _HEX_DIAGONALS:
        return _Aim("in all twelve directions")
    if directions == _HEX_OFF_LINE:
        return _Aim("to the twelve cells three steps out that lie along no line", False)
    if len(directions) == 6 and directions <= _HEX_OFF_LINE and directions == expand_symmetry(
        min(directions), "rotate6"
    ):
        # One hand of the twelve (Q870's `pinwheel`): only `rotate6` writes it in one vector.
        return _Aim(
            "to every other one of the twelve cells three steps out that lie along no line,"
            " the six that all turn the same way round",
            False,
        )
    if directions <= _HEX_SIDES and len(directions) <= 3:
        return _Aim(" or ".join(_HEX_STEP_NAMES[step] for step in sorted(directions)))
    return None


def _span_words(move: dict[str, Any], aim: _Aim) -> str | None:
    """How far a line runs, with the direction on the end of it."""
    if not aim.step:
        return _oblique_span_words(move, aim)
    if aim.scale != 1:
        return None
    kind = move["kind"]
    far = int(move.get("range", DEFAULT_RANGE[kind]))
    near = int(move.get("min_range", 1))
    if far == 0:
        return f"any distance {aim.words}" if near <= 1 else (
            f"{_count(near)} squares or more {aim.words}"
        )
    if far == 1:
        return f"one square {aim.words}"
    if near == far:
        return f"exactly {_count(far)} squares {aim.words}"
    if near > 1:
        return f"{_count(near)} to {_count(far)} squares {aim.words}"
    return f"up to {_count(far)} squares {aim.words}"


def _oblique_span_words(move: dict[str, Any], aim: _Aim) -> str | None:
    """A line whose repeat is an oblique offset, such as two-and-one over and over.

    The distance is counted in offsets rather than in squares, which is what the engine
    counts a slide's range in, so the words say *steps of* and never *squares*.
    """
    if not aim.unit:
        return None
    far = int(move.get("range", DEFAULT_RANGE[move["kind"]]))
    near = int(move.get("min_range", 1))
    if far == 0:
        return (
            f"any number of steps of {aim.words}" if near <= 1
            else f"{_count(near)} steps or more of {aim.words}"
        )
    if far == 1:
        # A line one offset long is the leap itself -- there is no square in between an
        # oblique step for anything to stand on -- and the leap's phrase already says it.
        # Two sentences for one rule is a reader wondering what the difference is.
        return None
    if near == far:
        return f"exactly {_count(far)} steps of {aim.words}"
    if near > 1:
        return f"{_count(near)} to {_count(far)} steps of {aim.words}"
    return f"up to {_count(far)} steps of {aim.words}"


def _rider_count(move: dict[str, Any], aim: _Aim) -> str | None:
    """How many jumps a rider makes, with the jump on the end: *any number of jumps of two
    squares forward, back, left and right* (Q1007a, `docs/plan-schema-31.md` §2.5).

    For a jump longer than one square and repeated: a slide over it (the nightrider, option
    1) or a lame move with a ``range`` other than 1 (option 2). The distance is counted in
    jumps, which is what the engine counts both in, so the words never say *squares* of it.
    ``None`` for anything else, which keeps its own phrase.
    """
    if aim.step and aim.scale > 1:
        jump = f"two squares {aim.words}" if aim.scale == 2 else (
            f"{_count(aim.scale)} squares {aim.words}"
        )
    elif not aim.step and aim.unit:
        jump = aim.words
    else:
        return None
    far = int(move.get("range", DEFAULT_RANGE[move["kind"]]))
    near = int(move.get("min_range", 1))
    if far == 1:
        return None
    if far == 0:
        count = "any number of jumps" if near <= 1 else f"{_count(near)} jumps or more"
    elif near == far:
        count = f"exactly {_count(far)} jumps"
    elif near > 1:
        count = f"{_count(near)} to {_count(far)} jumps"
    else:
        count = f"up to {_count(far)} jumps"
    return f"{count} of {jump}"


def _path_words(move: dict[str, Any], aim: _Aim) -> str | None:
    """A slide whose path turns (T1, `docs/plan-pieces-moves.md` §5.2): *any number of steps of
    two and one, in all eight directions, each step turning to the next direction round, ...*.

    Counted in steps, which is what the engine counts its ``range`` in. ``None`` for a
    straight slide, and where the step has no phrase (the vectors say it then)."""
    path = path_of(move)
    if path == "straight":
        return None
    if aim.step and aim.scale == 1:
        noun, over = f"one-square steps {aim.words}", ""
    elif aim.step:
        noun = f"steps of {_count(aim.scale)} squares {aim.words}"
        over = ", jumping over the squares between"
    elif aim.unit:
        noun, over = f"steps of {aim.words}", ", jumping over the squares between"
    else:
        return None
    far = int(move.get("range", DEFAULT_RANGE["slide"]))
    near = int(move.get("min_range", 1))
    count = "any number of" if far == 0 else f"up to {_count(far)}"
    nearest = f", stopping no nearer than step {_count(near)}" if near > 1 else ""
    if path == "circle":
        turn = ("each step turning to the next direction round, always the same way (either "
                "way), so the path circles and never comes back onto a square it has been on")
    else:
        turn = ("turning to a neighbouring direction and back again at every step (either "
                "neighbour), so the path zigzags between the two")
    return (f"{count} {noun}, {turn}{nearest}; stopped by a piece on a square it lands on"
            f"{over}")


def _slide_words(move: dict[str, Any], aim: _Aim) -> str | None:
    if path_of(move) != "straight":
        return _path_words(move, aim)
    rider = _rider_count(move, aim)
    if rider is not None:
        # Option 1: nothing between two landings is looked at.
        return f"{rider}, stopped only by a piece on a square it would land on"
    span = _span_words(move, aim)
    if span is None:
        return None
    # A one-square line has nothing in between, so the clause that makes a slide a slide
    # would be a rule that can never fire.
    blocked = int(move.get("range", DEFAULT_RANGE["slide"])) != 1
    return span + (", stopped by anything in the way" if blocked else "")


def _leap_words(move: dict[str, Any], aim: _Aim) -> str | None:
    over = aim.scale > 1 or not aim.step
    if aim.step:
        words = f"{_count(aim.scale)} square{'' if aim.scale == 1 else 's'} {aim.words}"
    else:
        words = f"leaps {aim.words}"
    if move["kind"] == "lame":
        rider = _rider_count(move, aim)
        if rider is not None:
            # Option 2, the lame rider: each jump stoppable at its leg as well.
            return (f"{rider}, stopped by a piece on a square it would land on or on the "
                    "square any jump passes")
        if int(move.get("range", DEFAULT_RANGE["lame"])) != 1:
            return None  # a ride this vocabulary cannot say: the vectors, not a wrong leap
        # Not "the path must be clear": exactly one square can stop it, and which square
        # that is -- the leg, the eye -- is the rule people state.
        return words + (
            ", stopped by a piece standing on the square it passes" if over else ""
        )
    return words + (", jumping over whatever is in between" if over else "")


def _hop_words(move: dict[str, Any], aim: _Aim) -> str | None:
    if not aim.step or aim.scale != 1:
        return None
    screen = {"own": "one of your own pieces", "enemy": "one enemy piece"}.get(
        str(move.get("screen", "any")), "exactly one piece of either side"
    )
    # ``screen_range`` caps the landing from the screen; without one the descriptor's own
    # reach does, counted the same way from the screen outwards.
    capped = int(move.get("screen_range", 0))
    far = capped or int(move.get("range", DEFAULT_RANGE["hop"]))
    near = 1 if capped else int(move.get("min_range", 1))
    if far == 0:
        landing = "landing anywhere beyond it"
    elif far == 1:
        landing = "landing on the square just past it"
    elif near == far:
        landing = f"landing exactly {_count(far)} squares past it"
    elif near > 1:
        landing = f"landing {_count(near)} to {_count(far)} squares past it"
    else:
        landing = f"landing at most {_count(far)} squares past it"
    return f"jumps {aim.words}, over {screen}, {landing}"


def _sweep_words(move: dict[str, Any], aim: _Aim) -> str | None:
    span = _span_words(move, aim)
    if span is None:
        return None
    return (
        f"{span}, sweeping away every piece it passes over, its own included; a piece "
        "it may not capture stops it"
    )


# ------------------------------------------------------------- a rule, before it is written
#
# Everything above says a rule back once a spec has written it. This says what a rule
# *is* before anybody has filled one in, which is what a menu offering twenty-eight of
# them needs. The schema's own `description` is field help and thirteen of the rule keys
# have none at all, so a menu built from the schema could only ever show the key.
#
# Every sentence here was written against `engine/compile.py`'s handling of that key, and
# a sentence that is wrong is worse than a key. Where a rule's meaning turns on its
# arguments the sentence says the shape and leaves the arguments to the card's own fields,
# which are the schema's and are argument-built already.

RULE_WORDS: dict[str, tuple[str, str]] = {
    "bare_royal": (
        "a player left with royalty alone",
        "What happens to a player left with nothing on the board but their royal piece.",
    ),
    "castling": (
        "castling",
        "The royal piece and a piece that has not moved make one move together, the "
        "royal stepping toward it and that piece jumping to the square the royal crossed.",
    ),
    "check_limit": (
        "winning by giving checks",
        "Giving check this many times wins the game outright.",
    ),
    "checkmate": (
        "what being checkmated does",
        "The result for the player whose royal piece is trapped. A win here inverts the "
        "object of the game, so the aim is to be checkmated.",
    ),
    "contacts": (
        "what two pieces become when they meet",
        "One link per pair says what each of two pieces becomes when one moves onto the "
        "other, and where the two of them end up.",
    ),
    "drops": (
        "pieces played back onto the board",
        "A player may put a piece from their hand onto an empty square instead of moving "
        "one. The hand is filled by what they capture, by a stock they start with, or by "
        "both.",
    ),
    "en_passant": (
        "taking a piece in passing",
        "A piece that steps two squares past an enemy that could have taken it on the way "
        "may be captured as if it had stepped one.",
    ),
    "explode_on_capture": (
        "captures that explode",
        "A capture destroys the capturing piece and everything standing around the square "
        "it happened on.",
    ),
    "extinction": (
        "losing by running out of a piece",
        "A player whose whole army, or whose count of some piece, falls below a number "
        "gets this result.",
    ),
    "forbid_check": (
        "check is not allowed",
        "Giving check is not allowed at all.",
    ),
    "forbid_immobility": (
        "no move that strands a piece",
        "A move may not leave one of your own pieces somewhere it could never move from "
        "again.",
    ),
    "max_plies": (
        "the longest a game may run",
        "How many turns the game may run before it stops, counting both players' moves: "
        "each player's move is one turn. "
        "It stops as a draw unless a scoring rule says what to compare.",
    ),
    "must_capture": (
        "a capture must be taken",
        "A capture on offer is the only kind of move there is.",
    ),
    "no_progress_draw": (
        "how long a game may stall",
        "How many turns may pass with nothing irreversible -- no capture, no promotion, "
        "no step by a piece that cannot come back, no ground claimed for good -- before "
        "the game stops.",
    ),
    "npcs": (
        "wild pieces",
        "Wild pieces belong to neither player. After every move each one steps to the "
        "next square of its route and eats whatever stands there that its piece may "
        "take, or waits if it may not. A royal piece where one steps next is in check "
        "from it when that piece may take a royal, and taking one is worth its bounty.",
    ),
    "simultaneous": (
        "playing in rounds: both players move at once",
        "Both players choose a move at the same time, without seeing the other's, and both "
        "are made together. Only two pieces moving onto one square collide, and each piece "
        "says which pieces it beats when they do.",
    ),
    "portals": (
        "portals",
        "Pairs of squares that send a piece on: a move that ends on one comes out of the "
        "other, one step further on in the direction it was going, and ends there. Each "
        "pair says who may move its two ends and whose pieces may go in by which.",
    ),
    "obstacles": (
        "walls on the board",
        "Squares nothing may stand on. A sliding line stops at one, and a jump vaults it "
        "unless the rule says otherwise. This is how a board is given a shape other than "
        "a rectangle.",
    ),
    "promote_on_entry": (
        "promoting on arrival",
        "A piece that enters play inside its own promotion zone -- dropped there, or left "
        "standing there by a strike -- promotes at once, with no move to choose in.",
    ),
    "promotion": (
        "promotion",
        "A piece that reaches a zone turns into one of the pieces the rule lists, and a "
        "rule may let a move out of the zone promote it too.",
    ),
    "reach": (
        "winning by arriving",
        "Arriving on one of these squares decides the game there and then; or, where the "
        "rule says so, moving off one does, and standing there decides nothing.",
    ),
    "repetition_draw": (
        "drawing by repetition",
        "How many repetitions of a position draw the game, or never for a game with no "
        "draw by repetition at all.",
    ),
    "royal_facing": (
        "royal pieces facing each other",
        "Whether the two royal pieces may see each other down an open file.",
    ),
    "scoring": (
        "who wins when a clock runs out",
        "A clock that runs out settles the game by comparing something -- material, "
        "pieces, ground held -- instead of drawing it.",
    ),
    "stalemate": (
        "a player with no legal move",
        "The result for a player whose turn it is and who has no legal move.",
    ),
    "strike": (
        "what a taken piece becomes",
        "A piece that is taken turns into another piece instead of leaving the board, "
        "following a chain of what becomes what.",
    ),
    "territory": (
        "claiming ground",
        "Moving onto a square claims it, and claimed ground can win the game outright or "
        "settle it when a clock runs out.",
    ),
    "turn": (
        "more than one move a turn",
        "A turn is this many moves rather than one, and the opening turns can be given "
        "their own lengths.",
    ),
    "win": (
        "how the game is won",
        "The main object of the game -- trapping the royal piece, taking it, clearing the "
        "board of an army, or leaving it to the endings below.",
    ),
    "wrap": (
        "edges that join",
        "A piece that runs off one edge continues from the opposite one, which makes the "
        "board a cylinder or a torus.",
    ),
}

# Other names a rule goes by, for *find a rule* (Q995c, fix 6, F6): the words a player
# brings from the games that made a rule famous, which the rule's title and sentence don't
# use. A rule's own title and sentence are searched too, so these are only what those miss.
# Built in `web/cards.py` by Q995c, moved here by Q995e so the command line and the model
# read them too.
RULE_ALSO_CALLED: dict[str, tuple[str, ...]] = {
    "simultaneous": ("rounds", "simultaneous", "same time", "secret moves"),
    "explode_on_capture": ("atomic", "explosion", "blast"),
    "check_limit": ("three-check", "three check", "3-check", "checks"),
    "strike": ("circe", "rebirth", "reborn", "demote", "transform"),
    "obstacles": ("walls", "blocked squares", "holes", "rocks"),
    "drops": ("crazyhouse", "hand", "drop", "shogi"),
    "wrap": ("cylinder", "torus", "wraparound", "wrap around"),
    "reach": ("king of the hill", "race", "goal", "target square"),
    "extinction": ("extinction", "run out"),
    "npcs": ("wild", "neutral", "monster", "non-player"),
    "turn": ("double move", "marseillais", "progressive", "several moves", "two moves"),
    "castling": ("castle",),
    "en_passant": ("en passant",),
    "must_capture": ("antichess", "losing chess", "giveaway", "forced capture", "suicide"),
    "territory": ("territory", "claim", "paint"),
    "portals": ("teleport", "tunnel", "gate"),
    "forbid_check": ("no check",),
    "checkmate": ("losers", "misere", "selfmate"),
    "contacts": ("merge", "combine", "meet", "join"),
    "promote_on_entry": ("promotion zone", "drop zone"),
    "repetition_draw": ("threefold", "repetition"),
    "no_progress_draw": ("fifty-move", "fifty move", "50 move", "stall"),
    "max_plies": ("move limit", "length", "turn limit"),
    "royal_facing": ("flying general", "kings facing", "facing"),
    "bare_royal": ("bare king",),
    "scoring": ("points", "timeout", "score", "material"),
}


# Sibling fields of one rule that say the same thing in different ways. Only one of each
# group is read, so showing all of them at once is the promotion card offering three
# spellings of a zone with an *empty list* tick beside each (finding 2.8).
#
# **Ordered so the last option written is the one the compiler reads.** A promotion zone
# resolves `squares`, then `ranks`, then `rank` (`schema.promotion_squares`); targets
# resolve `to_by_player` before `to` (`schema.promotion_targets`); a wrap axis resolves
# `files_by_player` before `files` (`compile._compile_wrap`); castling reads `king_file`
# and ignores `king_steps` when both are there, which the validator refuses anyway.
# So a reader of this table matches from the end, and the plainest spelling comes first.
RULE_CHOICES: dict[str, tuple[dict[str, Any], ...]] = {
    "rules.promotion": (
        {
            "label": "where it promotes",
            "options": (
                {"label": "on one rank", "keys": ("rank",)},
                {"label": "in a band of ranks", "keys": ("ranks",)},
                {"label": "on named squares", "keys": ("squares",)},
            ),
        },
        {
            "label": "what it may become",
            "options": (
                {"label": "the same list for both players", "keys": ("to",)},
                {"label": "a list for each player", "keys": ("to_by_player",)},
            ),
        },
    ),
    "rules.castling": (
        {
            "label": "how far the royal piece goes",
            "options": (
                {"label": "a fixed number of steps", "keys": ("king_steps",)},
                {"label": "onto a named file", "keys": ("king_file", "rook_file")},
            ),
        },
    ),
    "rules.wrap": (
        {
            "label": "the side edges",
            "options": (
                {"label": "the same for both players", "keys": ("files",)},
                {"label": "one answer per player", "keys": ("files_by_player",)},
            ),
        },
        {
            "label": "the top and bottom edges",
            "options": (
                {"label": "the same for both players", "keys": ("ranks",)},
                {"label": "one answer per player", "keys": ("ranks_by_player",)},
            ),
        },
    ),
}

# One promotion rule and a list of them are the same card, so the list's items take the
# single rule's groups.
RULE_CHOICES["rules.promotion[]"] = RULE_CHOICES["rules.promotion"]

# A hexagon's wrap card: one group per pair of sides, in `schema.WRAP_PAIRS` order, spelled
# as the rectangle's axes are. Folded into `RULE_CHOICES` below since schema 16 (Q281a4),
# so the card offers them on every board and the validator refuses them on anything but a
# hexagon, as it refuses `files` on one; showing each shape only its own groups is the
# designer's (Q281c).
HEXAGON_WRAP_CHOICES: tuple[dict[str, Any], ...] = tuple(
    {
        "label": f"the {_label} sides",
        "options": (
            {"label": "the same for both players", "keys": (_key,)},
            {"label": "one answer per player", "keys": (f"{_key}_by_player",)},
        ),
    }
    for _key, _label in (
        ("n_s", "top and bottom"),
        ("ne_sw", "upper-right and lower-left"),
        ("nw_se", "upper-left and lower-right"),
    )
)
RULE_CHOICES["rules.wrap"] = RULE_CHOICES["rules.wrap"] + HEXAGON_WRAP_CHOICES


# What a game played in rounds counts in rounds (Q995e, F7 of `docs/reference/designer-coverage.md`).
# Two clocks are written in turns, one for each player's move, so in a game played in rounds
# a limit of 400 is 200 rounds. Said in rounds for the rules that are clocks: the rule's
# sentence, and a line giving the number written as rounds (`{turns}` and `{rounds}` are
# filled by the caller from the number in the draft).
ROUND_WORDS: dict[str, tuple[str, str]] = {
    "max_plies": (
        "How many rounds the game may run before it stops. The limit is written in turns, one "
        "for each player's move, so a round is two of them. It stops as a draw unless a "
        "scoring rule says what to compare.",
        "{turns} turns is {rounds} rounds.",
    ),
    "no_progress_draw": (
        "How many rounds may pass with nothing irreversible -- no capture, no promotion, no "
        "step by a piece that cannot come back, no ground claimed for good -- before the game "
        "stops. The limit is written in turns, one for each player's move, so a round is two "
        "of them.",
        "{turns} turns is {rounds} rounds.",
    ),
}

# How the opening choices are counted: one move for the player to move, or, in a game
# played in rounds, a choice each, made together.
OPENING_WORDS: dict[str, str] = {
    "turns": "{first} opening moves",
    "rounds": "each player choosing from {first} opening moves in the first round",
    "rounds_apart": "{first} opening moves for the first player and {second} for the second, "
                    "chosen together in the first round",
}


def opening_words(counts: Sequence[int], rounds: bool) -> str:
    """The opening count as the game is played: ``counts`` holds each seat's number of
    legal opening moves in a game played in rounds, or the one number of the player to move."""
    if not rounds:
        return OPENING_WORDS["turns"].format(first=counts[0])
    if len(counts) < 2 or counts[0] == counts[1]:
        return OPENING_WORDS["rounds"].format(first=counts[0])
    return OPENING_WORDS["rounds_apart"].format(first=counts[0], second=counts[1])


def rule_words() -> dict[str, dict[str, Any]]:
    """Every rule key, with its title and its opening sentence, for a caller drawing a menu.

    The designer's, through `GET /api/design`. Handed over as data so the page places
    what it is given and has no vocabulary of its own. A clock carries its words for a game
    played in rounds too (`ROUND_WORDS`).
    """
    out: dict[str, dict[str, Any]] = {
        key: {"title": title, "says": says}
        for key, (title, says) in RULE_WORDS.items()
    }
    for key, (says, count) in ROUND_WORDS.items():
        out[key].update({"roundsSays": says, "roundsCount": count})
    return out


def rule_choices() -> dict[str, Any]:
    """The alternative spellings inside a rule, for a form that shows one at a time."""
    return {
        path: [
            {
                "label": group["label"],
                "options": [
                    {"label": option["label"], "keys": list(option["keys"])}
                    for option in group["options"]
                ],
            }
            for group in groups
        ]
        for path, groups in RULE_CHOICES.items()
    }


# --------------------------------------------------------------------------- rules


def _capture_relation(spec: dict[str, Any]) -> str:
    """Who may take whom, when that is not "anybody may take anybody".

    Stated as a rule and not only as a note on each piece, because it is a *relation*:
    which piece beats which is the shape of the game, and a reader who has to assemble
    it from six separate piece entries has not been told it.
    """
    clauses = []
    for piece in spec["pieces"]:
        captures = piece.get("captures")
        if not isinstance(captures, list):
            continue
        clauses.append(
            f"{piece['id']} takes {', '.join(captures)}" if captures
            else f"{piece['id']} takes nothing"
        )
    if not clauses:
        return ""
    return (
        "What a piece may capture depends on what it is looking at. "
        + "; ".join(clauses)
        + ". Anything not listed blocks a line without being takeable."
    )


def _describe_rest(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Schema 29 (Q983a, `docs/plan-rest.md` §3, §5.7): rest as a piece-type property, in any
    game. One paragraph for the rule, and one sentence more where check is a rule, since a
    resting piece's check must still be answered (§3.6)."""
    rests = {pid: turns for pid, turns in piece_rests(spec).items() if turns > 0}
    if not rests:
        return []
    by_turns = "; ".join(
        f"{', '.join(pid for pid in rests if rests[pid] == turns)} {turns}"
        for turns in sorted(set(rests.values()), reverse=True)
    )
    turn = "a round is a turn of both players" if is_simultaneous(rules) else (
        "only its own player's turns count"
    )
    lines = [
        f"A piece that makes its player's move then rests for that player's next n turns, n "
        f"its type's ({by_turns}; every other piece never rests); {turn}. A resting piece "
        "may not make its player's move, and still stands: it attacks, blocks, and can be "
        "captured. A piece that turned in place, was dropped or struck without moving rests "
        "too, as the piece it is after the move; one captured, moved or changed by the "
        "other player rests no more. When every move a player has is a resting piece's, "
        "the rest is lifted for that turn, so the rest never leaves a player without a move."
    ]
    if rules.get("win", "checkmate") == "checkmate" or rules.get("check_limit"):
        lines.append("A resting piece gives check, and its check must still be answered.")
    return lines


# Schema 30 (Q999a, `docs/archive/plan-passive.md` §4.3): what a sight does to what it sees, and
# whose pieces those are, in the piece's sentence.
_SIGHT_VERBS = {
    "freeze": "freezes", "disarm": "disarms", "shield": "shields", "command": "commands",
}
_SIGHT_WHOSE_WORDS = {"enemy": "enemy", "own": "its own", "any": "either side's"}


def _plural(name: str) -> str:
    """A piece's name for several of them: *queens*, *bishops*, *wardens*."""
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    if len(name) > 1 and name.endswith("y") and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def _sight_reach(sight: Sight, hexy: bool) -> str:
    """Where a sight looks, in the words a slide of its geometry is said in, and whether a
    piece in the way stops it. Falls back to the vectors where no phrase fits, as a move's
    sentence does."""
    move = {
        "kind": "slide",
        "vectors": [list(vector) for vector in sight.vectors],
        "symmetry": sight.symmetry,
        "range": sight.range,
        "min_range": sight.min_range,
    }
    aim = _aim(move, hexy)
    span = _span_words(move, aim) if aim is not None else None
    if span is None and aim is not None and not aim.step and sight.range == 1:
        span = f"at a jump of {aim.words}"
    if span is None:
        vectors = ", ".join(f"({x:+d},{y:+d})" for x, y in sight.vectors)
        far = "to the edge of the board" if sight.range == 0 else (
            f"up to {_count(sight.range)} step{'s' if sight.range > 1 else ''}"
        )
        span = f"along {vectors}{_SYMMETRY_TEXT[sight.symmetry]}, {far}"
    if sight.range == 1 and sight.min_range == 1:
        # Nothing stands between a piece and a square one step away, so a ray and an aura
        # see the same squares and the clause would be a rule that never fires.
        return span
    if sight.through:
        return f"{span}, through any piece between"
    return f"{span}, as far as the first piece in the way"


def sight_sentence(sight: Sight, names: dict[str, str], hexy: bool = False) -> str:
    """One line of sight, said on its piece: *freezes enemy queens it sees any distance in
    all eight directions, as far as the first piece in the way*. ``names`` is each piece id's
    name. The operators that give and take a sight say it in these words too."""
    if sight.affects is None:
        # Q1007c: a command never touches a royal (`docs/plan-schema-31.md` §3.4).
        kinds = "non-royal pieces" if sight.effect == "command" else "pieces"
    else:
        plural = [_plural(names.get(pid, pid)) for pid in sight.affects]
        kinds = plural[0] if len(plural) == 1 else (
            ", ".join(plural[:-1]) + " and " + plural[-1]
        )
    said = (
        f"{_SIGHT_VERBS[sight.effect]} {_SIGHT_WHOSE_WORDS[sight.whose]} {kinds} it sees "
        f"{_sight_reach(sight, hexy)}"
    )
    if sight.effect == "command":
        said += (
            ": their opponent may move them too, but never to take the mover's own pieces"
            + (", and may make them take a piece of their own side" if sight.captures_own else "")
        )
    return said


def _sight_notes(spec: dict[str, Any], hexy: bool) -> dict[str, tuple[str, ...]]:
    """Each piece id's sights, one sentence each, in the order written."""
    names = {
        piece["id"]: str(piece.get("name", piece["id"]))
        for piece in spec.get("pieces") or []
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    return {
        pid: tuple(f"{sight_sentence(sight, names, hexy)} (see the rules)" for sight in sights)
        for pid, sights in piece_sights(spec).items()
    }


def _describe_sight(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Schema 30 (Q999a, `docs/archive/plan-passive.md` §2, §4.3): what a sight's effect does, in one
    paragraph, said only for the effects the game uses; and in rounds, that an effect decides
    what is offered and never how a round resolves (§2.8)."""
    effects = {sight.effect for sights in piece_sights(spec).values() for sight in sights}
    if not effects:
        return []
    rounds = is_simultaneous(rules)
    check = not rounds and (
        rules.get("win", "checkmate") == "checkmate" or bool(rules.get("check_limit"))
    )
    said = [
        "A piece in another's line of sight is under that sight's effect for as long as it "
        "stands there, read from the board as it is."
    ]
    if "freeze" in effects and "disarm" in effects:
        said.append(
            "A frozen piece cannot move at all; a disarmed piece may move but cannot capture; "
            "neither attacks anything" + (", so neither gives check." if check else ".")
        )
    elif "freeze" in effects:
        said.append(
            "A frozen piece cannot move at all (no step, capture, turn or castling), and "
            "attacks nothing" + (", so it gives no check." if check else ".")
        )
    elif "disarm" in effects:
        said.append(
            "A disarmed piece may move but cannot capture, and attacks nothing"
            + (", so it gives no check." if check else ".")
        )
    if "shield" in effects:
        said.append(
            "A shielded piece cannot be captured"
            + (", and a shielded royal is never in check." if check else ".")
        )
    if "command" in effects:
        # Q1007c (`docs/plan-schema-31.md` §3.2-§3.5).
        said.append(
            "A commanded piece still belongs to its owner, and its owner's opponent may also "
            "move it, as that player's move and as its owner would move it: onto an empty "
            "square, or (where the sight says so) onto a piece of its own side it may capture, "
            "never a royal and never a shielded piece. A commanded piece never castles, takes "
            "or opens en passant, or promotes, so a move that would have to promote is not "
            "offered; it counts as having moved. A royal is never commanded, a frozen piece is "
            "moved by nobody, and a disarmed one may not be made to capture."
        )
    if check:
        answers = []
        if effects & {"freeze", "disarm"}:
            answers.append(
                " or ".join(
                    word for effect, word in (("freeze", "freezing"), ("disarm", "disarming"))
                    if effect in effects
                ) + " the checker"
            )
        if "shield" in effects:
            answers.append("shielding the royal")
        if "command" in effects:
            answers.append("commanding the checker away or another piece into the way")
        said.append(
            f"So a check may be answered by {' or by '.join(answers)}, and a move that leaves "
            "the mover's own royal attacked once the effects are read again is not allowed."
        )
    said.append(
        "A sight works whatever is done to its own piece, and two sights of one effect are one."
    )
    if "freeze" in effects:
        said.append(
            "A freeze is never lifted: a player it leaves with no legal move has none, and "
            "the rule for having no legal move decides"
            + (" (or checkmate, in check)." if check else ".")
        )
    lines = [" ".join(said)]
    if rounds:
        lines.append(
            "An effect decides which moves are offered, never how the round resolves"
            + (": a disarmed piece can still win a collision." if "disarm" in effects else ".")
        )
    return lines


def _describe_simultaneous(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Schema 26 (Q918a): the round, the collision table, the tie, the void bar and what an
    ending reached by both means (`docs/reference/rounds.md` §2, §3); schema 27's chase
    switches (Q942a, `docs/archive/plan-round-chase.md` §3) where they change those sentences."""
    if not is_simultaneous(rules):
        return []
    tie = rules["simultaneous"].get("tie")
    swap = simultaneous_switch(rules, "swap")
    both = simultaneous_switch(rules, "both")
    spies = spy_ids(spec)
    if spies:
        # Schema 27 (Q935a, `docs/plan-spy.md` §1): the phases first, since every round
        # sentence below applies only once a spy is lost.
        named = " or ".join(spies)
        lines = [
            f"The spy ({named}) decides how the game is played. While both players have a "
            "spy on the board, they take turns as usual. When one player loses their last "
            "spy, they first play one ordinary reply move if it is their turn, so a spy's "
            "taker can be taken back; from then on the game is played in rounds in which "
            "the player without a spy announces their move first, and the other chooses "
            "knowing it. Once neither has a spy, both choose in secret. A spy lost is never "
            "regained.",
            "In a round both moves are made together, on the board both chose from. There "
            "is no check at any time: a royal piece may be left where it can be taken.",
        ]
    else:
        lines = [
            "Both players move at once. Each round, each player chooses one of their own "
            "legal moves on the same board without seeing the other's, and both are made "
            "together. There is no check: a royal piece may be left where it can be taken.",
        ]
    lines += [
        (
            "Two pieces moving onto the same square collide, and so do two pieces that swap "
            "squares, each moving onto the square the other left. A piece that moves onto a "
            "square its target has just left lands there if that move could land on an empty "
            "square, and otherwise stays where it was; paths that cross do not collide."
        ) if swap else (
            "Only two pieces moving onto the same square collide. A piece that moves onto a "
            "square its target has just left lands there if that move could land on an empty "
            "square, and otherwise stays where it was; paths that cross, and two pieces that "
            "swap squares, do not collide."
        ),
    ]
    speeds = {pid: speed for pid, speed in piece_speeds(spec).items() if speed > 0}
    if speeds and has_speeds(spec):
        # Schema 28 (Q943a, `docs/archive/plan-speed.md` §2.4, §4.3): the order before the
        # collision sentence, which then reads only for equal speeds.
        ranked = "; ".join(
            f"{', '.join(pid for pid in speeds if speeds[pid] == rank)} {rank}"
            for rank in sorted(set(speeds.values()), reverse=True)
        )
        lines.append(
            f"Each piece has a speed ({ranked}; every other piece 0). When the two moving "
            "pieces differ in speed, the faster move is made first, as a move of an "
            "alternating game, and if it ends the game the round ends there. The slower "
            "move is then made if its piece still stands where it was and the move can "
            "still be made on that board: onto a square the faster piece now holds it is a "
            "capture, if the piece may capture there. Otherwise the slower move is void and "
            "its piece stays, or is lost with its piece. So pieces of different speeds never "
            "collide or swap; pieces of equal speed move at once, as above."
        )
    clauses = []
    for piece in spec.get("pieces") or []:
        beats = piece.get("beats") if isinstance(piece, dict) else None
        if isinstance(beats, list):
            clauses.append(
                f"{piece['id']} beats {', '.join(beats)}" if beats else f"{piece['id']} beats nothing"
            )
    who = (
        "In a collision a piece beats what it may capture"
        + (" unless it says otherwise: " + "; ".join(clauses) if clauses else "")
        + ". If one beats the other it stands on the square and the other is eliminated; "
    )
    if tie == "initiative":
        who += (
            "if both or neither do, the player holding the initiative wins it, and the "
            "initiative passes to the other player. The second player holds it at the start."
        )
        if spies:
            who += " When the first announced round begins, it goes to the announcer."
    elif tie == "impossible":
        who += "no two pieces that can meet ever both win or both lose."
    else:
        who += "if both or neither do, both are eliminated."
    lines.append(who)
    ending = (
        "An ending both players reach in the same round (both royals taken, both armies "
        "gone, both on a goal) goes to the player ahead by the score, as at the clocks; "
        "equal scores are a draw."
        if both else "An ending both players reach in the same round is a draw."
    )
    lines.append(
        ending if swap else (
            "Two pieces that could only take each other and miss two rounds in a row may not "
            "try again until either of their squares changes. " + ending
        )
    )
    return lines


def _describe_turn(turn: dict[str, Any]) -> str:
    moves = int(turn["moves"])
    text = f"A turn is {moves} moves rather than one."
    ramp = turn.get("ramp")
    if ramp:
        lengths = ", ".join(str(entry) for entry in ramp)
        text += (
            f" The opening turns are shorter, taking {lengths}"
            f"{' move' if len(ramp) == 1 and ramp[0] == 1 else ' moves'} "
            f"before {moves} for the rest of the game."
        )
    return text


def _promotion_rules(rules: dict[str, Any]) -> list[dict[str, Any]]:
    promotion = rules.get("promotion")
    if isinstance(promotion, list):
        return promotion
    return [promotion] if promotion else []


def _describe_promotion(promotion: dict[str, Any]) -> str:
    if "squares" in promotion:
        # Schema 14: a zone that is a shape, listed in full up to a hexagon's far edge.
        where = _squares(promotion["squares"], limit=16)
    elif "ranks" in promotion:
        ranks = ", ".join(str(r + 1) for r in sorted(promotion["ranks"]))
        where = f"rank {ranks}" if len(promotion["ranks"]) == 1 else f"ranks {ranks}"
    elif promotion.get("rank", "last") == "last":
        where = "the far rank"
    else:
        where = f"rank {promotion['rank'] + 1}"

    by_player = promotion.get("to_by_player")
    if by_player:
        first = ", ".join(by_player["0"])
        second = ", ".join(by_player["1"])
        becomes = f"becomes one of {first} for the first player, or {second} for the second"
    else:
        becomes = f"becomes one of {', '.join(promotion['to'])}"

    may = "may become" if promotion.get("mandatory", True) is False else "becomes"
    if may == "may become":
        becomes = becomes.replace("becomes one of", "may become one of", 1)
    if promotion.get("only_lost") is True:
        # Schema 18 (Q783): Grand chess's rule, which a reader of the list alone would not
        # guess -- and with every target withheld, a compulsory promotion is no move.
        becomes += (
            ", but only into a piece its side has lost: one it has fewer of on the board "
            "than it started with"
        )
    if promotion.get("on_leaving") is True:
        # Schema 15: the move out of the zone promotes too, which is shogi's rule and the
        # one a player who has only been told "reaching" would not guess.
        return f"{promotion['piece']} on a move that starts or ends in {where} {becomes}"
    return f"{promotion['piece']} reaching {where} {becomes}"


def _describe_entry_promotion(rules: dict[str, Any]) -> str:
    """Schema 9's second key, said as the event a player will actually meet.

    Names what each promoting piece turns into, because the rule takes the *first* piece
    listed and a player reading "becomes one of Q, R, B, N" above would otherwise not
    know which. Where the two players promote into different armies, both are named.
    """
    becomes: list[str] = []
    for promotion in _promotion_rules(rules):
        if not isinstance(promotion, dict) or "piece" not in promotion:
            continue
        by_player = promotion.get("to_by_player")
        if isinstance(by_player, dict) and by_player.get("0") and by_player.get("1"):
            first, second = by_player["0"][0], by_player["1"][0]
            becomes.append(
                f"{promotion['piece']} to {first}"
                if first == second
                else f"{promotion['piece']} to {first} for the first player, or {second} "
                "for the second"
            )
        elif promotion.get("to"):
            becomes.append(f"{promotion['piece']} to {promotion['to'][0]}")
    into = f" ({'; '.join(becomes)})" if becomes else ""
    return (
        "A piece that enters play inside its promotion zone -- dropped there, or left "
        "standing there by a strike -- promotes at once, to the first piece listed"
        f"{into}, whether or not it could otherwise have declined."
    )


def _describe_obstacles(rules: dict[str, Any]) -> list[str]:
    """Squares nothing may occupy.

    Stated high in the list, beside the seam, and for the same reason: a wall changes
    what a *line* is, so every sentence about how a piece moves is read differently
    once you know where the walls are.
    """
    obstacles = rules.get("obstacles")
    if not isinstance(obstacles, dict) or not obstacles.get("squares"):
        return []
    squares = list(obstacles["squares"])
    mirrored = " (and their mirror images)" if obstacles.get("mirror") else ""
    lines = [
        f"{_squares(squares)}{mirrored} are walls. No piece may stand on them, and "
        "nothing can move or capture through one."
    ]
    lines.append(
        "A jumping piece may not land on a wall but may jump over one."
        if not obstacles.get("blocks_leaps")
        else "A wall stops a jumping piece too, which may neither land on one nor jump "
        "over one."
    )
    return lines


_KING_STEP = frozenset((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)) - {(0, 0)}

# The same landing on a hex lattice (schema 14, `transform.HEX_STEP_LANDING`, Q484): the
# six cells sharing a side with the struck one, spelled out under symmetry ``"none"``
# because a hex board refuses every other symmetry. This is the king step's own reading
# there -- every cell touching the struck one -- so it gets the king step's own kind of
# sentence rather than being read out as six raw vectors.
_HEX_NEIGHBOURS = frozenset({(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)})


def _landing_phrase(rule: dict[str, Any]) -> str:
    """Where ``position: "choose"`` may place a survivor, as a player would say it."""
    landing = rule.get("landing")
    if not isinstance(landing, dict):
        return "an empty square of the attacker's choosing"
    directions: set[tuple[int, int]] = set()
    for vector in landing.get("vectors") or []:
        if isinstance(vector, list) and len(vector) == 2:
            directions |= expand_symmetry(
                (vector[0], vector[1]), landing.get("symmetry", "none")
            )
    if directions == _KING_STEP:
        return "any empty square a king's step from where it stood"
    if directions == _HEX_NEIGHBOURS:
        return "any of the six empty cells touching where it stood"
    steps = ", ".join(f"[{x}, {y}]" for x, y in sorted(directions))
    return (
        "any empty square one of these steps from where it stood, forward being the "
        f"attacker's: {steps}"
    )


def _or_list(names: list[str]) -> str:
    """``bishop``, ``bishop or knight``, ``pawn, bishop or knight``."""
    return " or ".join([", ".join(names[:-1]), names[-1]] if len(names) > 2 else names)


def _describe_strike(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """What happens instead of removal when a piece is taken.

    The hardest rule in the schema to state briefly and the one it matters most to
    state, because a strike game *looks* exactly like the game it came from: the moves
    are the same moves, and only what they do is different.
    """
    strike = rules.get("strike")
    if not isinstance(strike, dict):
        return []
    effect = strike.get("effect", "capture")
    chain = strike.get("chain")
    position = strike.get("position", "advance")
    names = {
        piece["id"]: piece.get("name", piece["id"])
        for piece in spec.get("pieces", [])
        if isinstance(piece, dict) and "id" in piece
    }
    immune = [str(names.get(pid, pid)) for pid in strike.get("immune", [])]
    # Schema 17 (Q540): the capturer changes sides, under any effect.
    switching = strike.get("switches_sides")
    switch_lines = (
        [
            f"A {', '.join(str(names.get(pid, pid)) for pid in switching)} that captures "
            "changes sides: it becomes the other player's piece on the square it took."
        ]
        if isinstance(switching, list) and switching
        else []
    )
    # A1 (`docs/plan-pieces-moves.md` §6.2): only the listed pieces' captures strike.
    by = strike.get("by")
    by_names = (
        [str(names.get(pid, pid)) for pid in by] if isinstance(by, list) and by else []
    )
    by_lines = (
        [
            f"Only a capture made by a {_or_list(by_names)} strikes; every other piece "
            "captures as usual, moving onto the square it took."
        ]
        if by_names
        else []
    )
    if effect == "capture" and position == "stay" and by_names:
        # The arrow: rifle given to some pieces only.
        shooters = _or_list(by_names)
        lines = [
            f"A {shooters} shoots: a capture by a {shooters} removes its target and the "
            f"{shooters} stays where it stood, never moving onto the square it took. "
            "Every other piece captures as usual, moving onto the square it took."
        ]
        if immune:
            lines.append(
                f"{', '.join(immune)} cannot be shot, and a {shooters} that takes one "
                "moves onto its square as usual."
            )
        return lines
    if effect == "capture":
        # Schema 17 (Q540): what a plain capture leaves behind, when it is not plain.
        said = {
            "stay": "Taking a piece removes it, but the capturing piece stays where it "
            "stood and does not move to the square it took.",
            "home": "A piece that is taken is reborn on its home square. A piece whose "
            "type starts on at most one square of each colour goes back to the one of "
            "the capture square's colour, so it keeps its square colour; any other "
            "piece goes to its type's starting square on the capture's file, or failing "
            "that of the capture square's colour. If that square is occupied, or the "
            "type has no home, the piece is removed. A reborn piece stands as if it had "
            "never moved.",
        }.get(position)
        lines = [said] if said else []
        lines += switch_lines
        if lines:
            lines += by_lines
        if lines and immune:
            lines.append(f"{', '.join(immune)} are taken as usual.")
        return lines
    if not chain:
        return switch_lines

    verb = "demoted" if effect == "demote" else "promoted"
    links = [
        f"{names.get(link['from'], link['from'])} becomes "
        f"{names.get(link['to'], link['to'])}"
        for link in chain
        if isinstance(link, dict) and "from" in link and "to" in link
    ]
    where = {
        "stay": "the attacker does not move, and the struck piece changes where it "
        "stands",
        "swap": "the two exchange squares, so the attacker takes the square and the "
        "struck piece lands where the attacker came from",
        "advance": "the attacker takes the square",
        "choose": "the attacker takes the square and places the struck piece on "
        f"{_landing_phrase(strike)}; a piece with no such square cannot be taken",
        "home": "the attacker takes the square and the struck piece is reborn on its "
        "home square (for a type with at most one home of each colour, the one of the "
        "capture square's colour; otherwise the one on the capture's file, or failing "
        "that of the capture square's colour), or removed if that square is occupied",
    }[position]

    lines = [
        f"Taking a piece does not remove it. It is {verb} instead, and {where}.",
        f"In the chain, {'; '.join(links)}.",
        "Anything the chain does not name is captured outright, as usual.",
    ]
    if immune:
        lines.append(
            f"{', '.join(immune)} cannot be struck, so taking one removes it for good."
        )
    return lines + switch_lines + by_lines


def _describe_contacts(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """What two pieces that meet both become (schema 10, Q19a).

    Harder to state than a strike and for the same reason it is worth stating: the game
    *looks* like the one it came from, and the table is read one meeting at a time. Said
    per link rather than as two chains, because "a pawn taking a rook becomes a knight"
    is the sentence a player has to hold, and splitting it into what-the-mover-does and
    what-the-target-does would make them reassemble it themselves.
    """
    contacts = rules.get("contacts")
    if not isinstance(contacts, dict):
        return []
    links = [link for link in (contacts.get("links") or []) if isinstance(link, dict)]
    if not links:
        return []

    names = {
        piece["id"]: piece.get("name", piece["id"])
        for piece in spec.get("pieces", [])
        if isinstance(piece, dict) and "id" in piece
    }
    immune_ids = contacts.get("immune")
    immune_ids = immune_ids if isinstance(immune_ids, list) else []

    def said(pid: Any) -> str:
        return str(names.get(pid, pid))

    sentences = []
    for link in links:
        mover, onto = said(link.get("mover")), said(link.get("onto"))
        own = link.get("own") is True
        halves = []
        if "mover_to" in link and link["mover_to"] == link.get("mover"):
            # Schema 18 (Q781): Einstein's keeping link. Its whole content is that it
            # beats the promotion, so the sentence says what is kept.
            halves.append(f"the {mover} stays as it is")
        elif "mover_to" in link:
            halves.append(f"the {mover} becomes {said(link['mover_to'])}")
        if "onto_to" in link:
            halves.append(f"the {onto} becomes {said(link['onto_to'])}")
        else:
            halves.append(f"the {onto} is removed")
        # Schema 11: a link about the mover's own piece is a move of its own, and the
        # sentence has to say the mover may make it.
        meets = f"moves onto its own {onto}" if own else f"takes {onto}"
        sentences.append(f"when {mover} {meets}, {' and '.join(halves)}")

    where = {
        "stay": "the mover does not move, and a target that survives changes where it "
        "stands",
        "swap": "the two exchange squares, so the mover takes the square and a target "
        "that survives lands where the mover came from",
        "advance": "the mover takes the square",
        "choose": "the mover takes the square and places a target that survives on "
        f"{_landing_phrase(contacts)}; a target with no such square cannot be taken",
    }[contacts.get("position", "advance")]

    table = "; ".join(sentences)
    own_lines = (
        [
            "A piece may also move onto one of its own pieces where a link below says so, "
            "along the lines it captures on, and the link says what both become."
        ]
        if any(link.get("own") is True for link in links)
        else []
    )
    lines = own_lines + [
        "Some captures change both pieces, not just the one that was taken. Which "
        f"change depends on who took what, and {where}.",
        f"{table[:1].upper()}{table[1:]}.",
        "Any other capture is ordinary, so the piece taken is removed and the piece "
        "taking it is unchanged.",
    ]
    if any(isinstance(link, dict) and link.get("mover_to") == link.get("mover") for link in links):
        lines.append(
            "What a link says the taker becomes is decided after any promotion that move "
            "would make, so a taker a link keeps as it is stays what it was even where "
            "its promotion would have changed it."
        )
    immune = [said(pid) for pid in immune_ids]
    if immune:
        lines.append(
            f"{', '.join(immune)} cannot be reached by a contact, so taking one removes "
            "it for good and leaves the taker as it was."
        )
    return lines


def _describe_npcs(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Pieces neither player owns, said the way a player has to be told them.

    Four things per wild piece -- what it is, where it walks, what it eats, what taking
    it is worth -- and one sentence that changes how every position is read: a royal
    standing where a wild piece steps next is in check, when that piece's type may
    capture a royal. A wild piece that is not stated
    is a piece on the board the page cannot account for, which is the territory failure
    with a face on it.
    """
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return []
    by_id = {
        piece["id"]: piece
        for piece in spec.get("pieces", [])
        if isinstance(piece, dict) and "id" in piece
    }
    lines: list[str] = []
    for npc in npcs:
        if not isinstance(npc, dict):
            continue
        piece = by_id.get(npc.get("piece"), {})
        name = str(piece.get("name", npc.get("piece")))
        route = list(npc.get("route") or [])
        if len(route) <= 1:
            walk = f"stands on {_squares(route)} and never moves"
        else:
            walk = (
                f"starts on {route[0]} and, after every move, steps along "
                f"{_squares(route)} and back to {route[0]}"
            )
        captures = piece.get("captures")
        if captures is None:
            eats = "It eats any piece standing in its way"
        elif captures:
            eats = f"It eats only {', '.join(captures)} standing in its way"
        else:
            eats = "It eats nothing and simply blocks the square it stands on"
        # The core gives the check only when the wild piece's type may capture the
        # royal (`engine/compile.py`, the wild attack source): a list that leaves every
        # royal out gives none, and saying otherwise is a rule the game does not have.
        royals = {pid for pid, p in by_id.items() if p.get("royal")}
        threatens_royal = captures is None or any(pid in royals for pid in captures)
        if captures is None or captures:
            eats += "; a piece it eats is gone for good"
            if threatens_royal:
                eats += (
                    ", and a royal piece standing where it steps next is in check from it"
                )
            elif royals:
                eats += (
                    ", but it cannot take a royal piece, so one standing where it steps "
                    "next is not in check from it"
                )
            eats += ". Where it may not step, it waits"
        bounty = npc.get("bounty", piece.get("value", 1))
        worth = (
            f"Taking it is worth {bounty:g} points."
            if isinstance(bounty, (int, float)) and not isinstance(bounty, bool) and bounty
            else "Taking it is worth nothing."
        )
        # Schema 23: more wild pieces of this kind on the same route, in step.
        also = [name for name in npc.get("also_at") or [] if isinstance(name, str)]
        more = ""
        if also:
            count = "Another" if len(also) == 1 else f"{_count(len(also)).capitalize()} more"
            more = (
                f" {count} {'starts' if len(also) == 1 else 'start'} on {_squares(also)} and "
                f"{'walks' if len(also) == 1 else 'walk'} the same route the same way, in "
                "step; a wild piece that finds another in its way waits behind it, and "
                "follows it when it moves on."
            )
        mirrored = (
            " A second one walks the mirror image of that route."
            if npc.get("mirror") and not also
            else " The same again walks the mirror image of that route."
            if npc.get("mirror")
            else ""
        )
        # Schema 10. The fifth thing a player has to be told about a wild piece, and the
        # one that decides how the whole board plays around it: which of their pieces can
        # do anything about it. Said in the same sentence as what it eats, because the
        # two together are the exchange.
        immune = npc.get("immune")
        taken = ""
        if isinstance(immune, list):
            survivors = [str(pid) for pid in immune]
            taken = (
                f" Only {', '.join(survivors)} may take it."
                if survivors
                else " Nothing may take it."
            )
        lines.append(
            f"A wild {name} belongs to neither player. It {walk}.{more} {eats}.{taken} "
            f"{worth}{mirrored}{_respawn_words(npc, route, also, rules)}"
        )
    return lines


def _respawn_words(
    npc: dict[str, Any], route: list[Any], also: list[str], rules: dict[str, Any]
) -> str:
    """Generation 36 (N1, `docs/plan-npc.md` §4.1): when a taken wild piece comes back,
    where, and what holds it back. Empty for a piece that, once taken, is gone."""
    every = npc_respawn(npc)
    if not every:
        return ""
    rounds = is_simultaneous(rules)
    unit = "round" if rounds else "move"
    clock = (
        f"{every} {unit}{'s' if every != 1 else ''} after it is taken"
        + ("" if rounds else ", counting both players' moves")
    )
    starts = [str(route[0])] if route else []
    starts += also
    where = (
        f"on {starts[0]}" if len(starts) == 1
        else f"on the first of {_squares(starts)} that it may come back onto"
    )
    halves = " Each walk keeps its own count." if npc.get("mirror") else ""
    return (
        f" Once taken, it comes back {where}, {clock}, eating a piece standing there unless "
        "that piece is royal, another wild piece or one it may not take; while its square "
        f"is held it comes back at the first {unit} it is free. One comes back at a time, "
        f"and the count starts again while any of its kind is still missing.{halves}"
    )


def _passage_words(way: Any, first: Any, second: Any) -> str:
    """One player's way through a pair, naming its ends by the squares they start on.

    The file's keys call the two ends ``orange`` and ``blue``; a reader is never told that
    (Q838c, `docs/archive/plan-portal-look.md` §5): both ends are drawn in the pair's one colour,
    and the square says which end is meant.
    """
    if way == "orange_to_blue":
        return f"may go in only at {first}"
    if way == "blue_to_orange":
        return f"may go in only at {second}"
    if way == "closed":
        return "may not go in at all: to them both ends are solid blocks"
    return "may go in at either end"


def _describe_portals(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Schemas 19 and 21: groups of portals, said the way a player has to be told them.

    One sentence for what a portal does, then one per group: where its ends start, where
    each leads, who may move them, and whose pieces (and which pieces) may go in by which
    end. Read through `portal_groups`, the one reader of the spelling; a pair keeps the
    sentence it has had since schema 19, so a pair's prose reads as it always has. "First
    player" and "second player" because the sides' own names are the page's; `plain` and
    the board say White and Black where the ruleset does.
    """
    groups = portal_groups(rules)
    if not groups:
        return []
    # Schema 22 (Q843c), revised at schema 24 (Q867b): a fold is the second rule set, said
    # in its own sentence; the momentum sentence is said only where a momentum group is.
    momentum = [group for group in groups if group["kind"] != "fold"]
    paired = all(group["spelling"] == "pair" for group in momentum)
    lines = [] if not momentum else [
        "A piece whose move ends on a portal does not stop there: it comes out of the other "
        "portal of the same pair, one step beyond it in the direction it was travelling "
        "(a move up and to the right by any amounts comes out diagonally up and to the "
        "right), and its move ends there, capturing, promoting or winning as on any square. "
        "If that square is off the board, a wall, its own piece, a piece it may not take, "
        "another portal or the square it started from, the move cannot be made. Where it "
        "could arrive from two directions, the player chooses. A portal is never captured, "
        "and moving one is a move onto an empty square."
        if paired else
        "A piece whose move ends on a portal that lets it in does not stop there: it comes "
        "out of the portal that one leads to, one step beyond it in the direction it was "
        "travelling (a move up and to the right by any amounts comes out diagonally up and "
        "to the right), and its move ends there, capturing, promoting or winning as on any "
        "square. If that square is off the board, a wall, its own piece, a piece it may not "
        "take, another portal or the square it started from, the move cannot be made. "
        "Where it could arrive from two directions, the player chooses. A portal that does "
        "not let a piece in is a solid block to it. A portal is never captured, and moving "
        "one is a move onto an empty square."
    ]
    if len(momentum) < len(groups):
        lines.append(
        "A fold is a portal of another kind: it is not a piece, so a piece of either side "
        "may stand in it, and is captured there as anywhere. A move that ends on an empty "
        "fold that lets the piece in -- a leap, a slide or a hop -- may stay "
        "there, or go through and end instead on the fold that one leads to, as a move "
        "landing there: onto an empty square if the step could end on an empty square, onto "
        "an enemy piece it may take, which is captured, if the step could capture, never "
        "onto its own piece or back onto the square it started from, and promoting by the "
        "square it ends on. A move goes through one fold at most. A move that passes over "
        "a fold, or ends on one a piece stands in, is not changed by it, and a fold that "
        "does not let a piece in, or leads nowhere, is plain floor to it. A fold a player "
        "moves may land on a piece of either side, which then stands in it, but never on "
        "another portal; moving a fold never moves the piece in it. Castling, a drop and an "
        "en passant capture never go through a fold, and a move that went through one "
        "opens no en passant."
        )
    players = ("the first player", "the second player")
    for group in groups:
        ends = group["ends"]
        movers = []
        for end in ends:
            who = [players[p] for p in sorted(end["moved_by"])]
            if who:
                movers.append(f"the end on {end['at']} may be moved by {' or '.join(who)}")
        # Schema 20 (Q807): the bar on going straight back.
        barred = (
            f" A portal of this {'pair' if group['spelling'] == 'pair' else 'group'} that "
            "has just been moved may not be moved straight back onto the square it came "
            "from on the very next move, by either player."
            if group["no_straight_back"]
            else ""
        )
        filters = "".join(
            f" Only {_listed_ids(end['carries'])} may go in at {end['at']}."
            for end in ends if end["carries"] is not None and end["enter"]
        )
        if group["spelling"] == "pair":
            # The ends by the squares they start on; `orange` and `blue` are only the
            # file's keys.
            first, second = ends[0]["at"], ends[1]["at"]
            moved = "; ".join(movers) if movers else "nobody moves either end"
            ways = [
                f"{players[p]}'s pieces "
                f"{_passage_words(group['passage'][p], first, second)}"
                for p in (0, 1)
            ]
            lines.append(
                f"{_group_word(group, 'pair')} {group['number']} joins {first} and {second}: "
                f"{moved}; {ways[0]}, and {ways[1]}.{barred}{filters}"
            )
            continue
        leads = []
        for end in ends:
            if end["to"] is None:
                leads.append(f"the end on {end['at']} is only a way out")
                continue
            whose = (
                "" if end["enter"] == {0, 1}
                else f" ({players[min(end['enter'])]}'s pieces only)"
            )
            leads.append(f"the end on {end['at']} leads to {ends[end['to']]['at']}{whose}")
        moved = "; ".join(movers) if movers else "nobody moves any end"
        squares = [end["at"] for end in ends]
        lines.append(
            f"{_group_word(group, 'group')} {group['number']} has ends on "
            f"{', '.join(squares[:-1])} and {squares[-1]}: {'; '.join(leads)}; "
            f"{moved}.{barred}{filters}"
        )
    return lines


def _group_word(group: dict[str, Any], shape: str) -> str:
    """*Portal pair*, *Portal group*, or for a fold (schema 22) *Fold pair*, *Fold group*."""
    return f"{'Fold' if group['kind'] == 'fold' else 'Portal'} {shape}"


def _listed_ids(ids: Any) -> str:
    """Piece ids as a list in a sentence: *N*, *N and B*, *N, B and R*."""
    named = sorted(str(pid) for pid in ids)
    return named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"


def _describe_wrap(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """A board whose opposite edges are the same edge.

    Two sentences per axis at most: what the board is, and who may cross the join. They
    are separate because they are separate facts -- the join exists for everyone, and
    the permission to traverse it need not be the same for the two players, which is the
    only reason this rule needed a per-player spelling at all.
    """
    wrap = rules.get("wrap")
    if not wrap:
        return []
    if spec["board"].get("shape") == "hexagon":
        return _describe_hexagon_wrap(wrap, rules)
    width = spec["board"]["width"]
    height = spec["board"]["height"]
    last_file = chr(ord("a") + width - 1)

    resolved = wrap_stated(wrap)
    lines: list[str] = []
    files_join = False
    for axis, low, high, low_edge, high_edge in (
        ("files", "a", last_file, "left", "right"),
        ("ranks", "1", str(height), "bottom", "top"),
    ):
        stated = dict(zip(("0", "1"), resolved[axis], strict=True))
        if all(edge == "none" for edge in stated.values()):
            continue
        files_join = files_join or axis == "files"
        thing = "file" if axis == "files" else "rank"
        lines.append(
            f"The board has no {low_edge} or {high_edge} edge, so {thing} {low} and "
            f"{thing} {high} are neighbours."
        )
        crossing = {
            "both": "in either direction",
            low_edge: f"only by leaving {thing} {low} and returning on {thing} {high}",
            high_edge: f"only by leaving {thing} {high} and returning on {thing} {low}",
            "none": "not at all",
        }
        if stated["0"] == stated["1"]:
            lines.append(f"Either player's pieces may cross that join {crossing[stated['0']]}.")
        else:
            for player, label in (("0", "first"), ("1", "second")):
                lines.append(
                    f"The {label} player's pieces may cross that join "
                    f"{crossing[stated[player]]}."
                )

    if files_join and rules.get("castling"):
        # Castling is stated as a span of one rank rather than as a piece following a
        # line, and it is the one move that does not follow the board's shape. Said out
        # loud because a player who has just been told the board is a cylinder will
        # otherwise assume it does.
        lines.append("Castling does not use the join; king and rook stay on their rank.")
    return lines


# A hexagon's sides as a player sees them on the standard diagram: flat-topped, the first
# player at the bottom (`schema.WRAP_PAIRS`).
_SIDE_WORDS = {
    "n": "top", "s": "bottom",
    "ne": "upper-right", "sw": "lower-left",
    "nw": "upper-left", "se": "lower-right",
}


def _describe_hexagon_wrap(wrap: Any, rules: dict[str, Any]) -> list[str]:
    """A hexagon whose opposite sides are the same side (Q281).

    The rectangle's two sentences per pair, what the board is and who may cross, with the
    sides named rather than the files and ranks along them: a side of a hexagon is not one
    file or one rank. A third fact follows once every pair is joined, and it is the one a
    player would not guess: every line through a cell is the whole board.
    """
    lines: list[str] = []
    joined: list[str] = []
    for key, stated in wrap_stated(wrap, "hexagon").items():
        if all(side == "none" for side in stated):
            continue
        joined.append(key)
        pair = WRAP_PAIRS[key]
        low, high = _SIDE_WORDS[pair.low], _SIDE_WORDS[pair.high]
        lines.append(
            f"The board has no {low} or {high} side: a piece that leaves by one comes "
            "back in by the other, straight across."
        )
        crossing = {
            "both": "in either direction",
            pair.low: f"only by leaving by the {low} side and coming back in by the {high} one",
            pair.high: f"only by leaving by the {high} side and coming back in by the {low} one",
            "none": "not at all",
        }
        if stated[0] == stated[1]:
            lines.append(f"Either player's pieces may cross that join {crossing[stated[0]]}.")
        else:
            for side, label in zip(stated, ("first", "second"), strict=True):
                lines.append(
                    f"The {label} player's pieces may cross that join {crossing[side]}."
                )
    if len(joined) == len(WRAP_PAIRS):
        # On the torus each of the six sides and six diagonals generates the whole group
        # of cells (`docs/game-ideas.md` *Q281 specified*), so an unbounded slider sees
        # every cell. Said because nothing about a square torus prepares a player for it.
        lines.append(
            "Every line through a cell runs through every other cell before it comes back."
        )
    if rules.get("castling") and {"ne_sw", "nw_se"} & set(joined):
        # A step along a rank leaves by those four sides and never by the top or bottom.
        lines.append("Castling does not use the join; king and rook stay on their rank.")
    return lines


def _describe_extra_rules(rules: dict[str, Any]) -> list[str]:
    """The schema-2 endings, each stated the way a player would be told it."""
    lines = []
    if rules.get("must_capture"):
        lines.append("A capture, where one is available, is compulsory.")
    if rules.get("forbid_check"):
        lines.append("Giving check is not allowed.")
    if rules.get("forbid_immobility"):
        lines.append("No move may strand a piece where it could never move again.")
    if rules.get("royal_facing") == "forbidden":
        lines.append("The two royal pieces may never see each other down an open file.")
    if rules.get("bare_royal"):
        lines.append(f"The result for {BARE_ROYAL_HOLDER} is a {rules['bare_royal']}.")
    if rules.get("check_limit"):
        lines.append(f"Delivering {rules['check_limit']} checks wins.")
    reach = rules.get("reach")
    if reach and reach.get("on") == "leave":
        # Schema 18 (Q786): the move's origin decides, not where a piece stands.
        piece = reach.get("piece", "any piece")
        where = _squares(reach["squares"], limit=6)
        lines.append(
            f"{piece} moving off {where} is a {reach.get('result', 'win')}: it has to get "
            f"there first and then step away, and standing there decides nothing."
        )
    elif reach:
        piece = reach.get("piece", "any piece")
        safe = " and surviving there" if reach.get("safe") else ""
        lines.append(
            f"{piece} reaching {_squares(reach['squares'], limit=6)}{safe} is a "
            f"{reach.get('result', 'win')}."
        )
    extinction = rules.get("extinction")
    if extinction:
        what = "any piece" if extinction["pieces"] == "any" else ", ".join(extinction["pieces"])
        count = extinction.get("min_count", 1)
        result = extinction.get("result", "loss")
        if extinction.get("per_type") == "own":
            # Schema 27 (Q936): each type on its own, counted only against a side that
            # starts with it, and the same mover-first ordering as `per_type: true`.
            lines.append(
                f"Holding fewer than {count} of any one of {what}, each counted on its "
                f"own and only against a player who starts the game with it, is a "
                f"{result}; a kind a player gains later is not counted against them, and "
                "where one move does that to both sides, it is the side that moved that "
                "has it."
            )
        elif extinction.get("per_type") is True:
            # Schema 18 (Q783): each type on its own, and the one ordering a reader has to
            # be told -- a move can run out a type on both sides at once.
            lines.append(
                f"Holding fewer than {count} of any one of {what}, each counted on its "
                f"own, is a {result}; where one move does that to both sides, it is the "
                "side that moved that has it."
            )
        else:
            lines.append(f"Holding fewer than {count} of {what} is a {result}.")
    drops = rules.get("drops")
    if drops:
        lines.append(
            "Captured pieces change sides and may be played back onto the board."
            if drops.get("from_captures", True)
            else "Held pieces may be played onto the board."
        )
        lines.extend(_describe_stock(drops.get("stock")))
        if drops.get("placement") is True:
            # Schema 18 (Q784): the placement phase, which changes what a turn *is* for
            # the opening stretch of the game, so it is said right after the stock.
            lines.append(
                "While a player holds any piece in hand, playing one onto the board is "
                "their only move: the army in hand is placed before anything on the board "
                "moves, and a royal still in hand is neither in check nor lost."
            )
        if drops.get("either_face") is True:
            # Schema 18 (Q785): a held piece has two ways onto the board, and the pairs
            # are the `promoted_from` notes each piece already carries.
            lines.append(
                "A piece in hand may be played as itself or as its other face, the piece "
                "that reverts to it when captured, on any square it may be played on."
            )
        lines.extend(_drop_restrictions(drops))
    lines.extend(_describe_explosion(rules.get("explode_on_capture")))
    return lines


def _describe_explosion(explosion: Any) -> list[str]:
    """A capture that takes the neighbourhood with it, and (schema 10) *which* capture.

    Three separate facts and they are said in that order because that is the order a
    player needs them: whose capture explodes, what it has to take, and who survives
    standing next to it. ``by`` and ``of`` narrow *when* the rule fires and ``immune``
    narrows what it reaches, so a sentence that folded them together would be wrong about
    the one thing a player is most likely to get killed by.

    An empty ``by`` or ``of`` is a rule no capture can fire, and it is described as no
    rule at all -- the same answer ``_compile_explosion`` gives.
    """
    if not isinstance(explosion, dict) or not explosion:
        return []
    by = explosion.get("by")
    of = explosion.get("of")
    if isinstance(by, list) and not by or isinstance(of, list) and not of:
        return []
    immune = explosion.get("immune") or []
    spared = f", except {', '.join(immune)}" if immune else ""
    who = f"A capture by {', '.join(by)}" if isinstance(by, list) else "A capture"
    what = f" of {', '.join(of)}" if isinstance(of, list) else ""
    if explosion.get("radius", 1) == 0:
        # Schema 17 (Q540): kamikaze.
        return [
            f"{who}{what} destroys only the capturing piece and the piece it took; "
            "nothing beside them is touched."
        ]
    return [
        f"{who}{what} destroys the capturing piece and everything beside it{spared}."
    ]


def _describe_stock(stock: Any) -> list[str]:
    """What a player holds before anything has been captured (schema 10, Q51).

    Said before the restrictions on *where* a held piece may go, because it is the fact
    that makes them matter: a player who has never been told they start with a stone in
    hand will not look for the move, and it is available on ply one.

    A supply that never runs out is a different sentence from a count, not a large one.
    """
    if not isinstance(stock, dict) or not stock:
        return []
    endless = sorted(str(k) for k, v in stock.items() if v == STOCK_UNLIMITED)
    counted = sorted(
        (str(k), v)
        for k, v in stock.items()
        if v != STOCK_UNLIMITED and isinstance(v, int) and not isinstance(v, bool)
    )
    lines = []
    if counted:
        lines.append(
            "Each player starts holding "
            + ", ".join(f"{count} {piece}" for piece, count in counted)
            + ", playable from hand without ever having been captured."
        )
    if endless:
        lines.append(
            f"Each player has an endless supply of {', '.join(endless)}, and playing one "
            "from hand never uses it up."
        )
    return lines


def _drop_restrictions(drops: dict[str, Any]) -> list[str]:
    """Where a held piece may not be played. Three separate rules, all shogi's.

    Left out of the sentence above rather than folded into it: "captured pieces come
    back" is the rule, and these are the exceptions to it, which is how a rulebook
    orders them and how a player learns them.
    """
    lines = []
    region = drops.get("region") or {}
    for piece, squares in sorted(region.items()):
        lines.append(f"A dropped {piece} may only be placed on {_squares(squares)}.")
    doubled = drops.get("no_doubled") or []
    if doubled:
        lines.append(
            f"No {', '.join(doubled)} may be dropped onto a file that already holds "
            f"one of yours."
        )
    no_mate = drops.get("no_mate") or []
    if no_mate:
        lines.append(f"No {', '.join(no_mate)} may be dropped to deliver checkmate.")
    return lines


def _describe_scoring_rules(spec: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Schema 4's territory and schema 3's scoring: the two Go-shaped rules.

    They are stated together because they are usually read together -- a game that
    claims ground and a game that counts it at the end are the same game twice over --
    and separately from the endings above because neither one is an event. Nothing
    happens; the position is simply worth something.
    """
    lines = []
    territory = rules.get("territory")
    if territory:
        board = spec["board"]
        total = int(board["width"]) * int(board["height"])
        permanent = territory.get("mode", "permanent") == "permanent"
        lines.append(
            "Moving onto a square claims it for you"
            + (", and a claim can never be taken back." if permanent
               else ", and an opponent moving there claims it back.")
        )
        win_at = int(territory.get("win_at", 0) or 0)
        if win_at:
            lines.append(f"Holding {win_at} of the {total} squares wins.")

    scoring = rules.get("scoring")
    if scoring:
        counts = {
            "material": "the declared value of the pieces each player still has",
            "pieces": "how many pieces each player still has",
            "region": "how many of your own pieces stand on "
                      + _squares(scoring.get("region") or []),
            "territory": "how much ground each player has claimed",
            "points": "the points each player has collected by taking wild pieces",
        }
        what = counts.get(str(scoring.get("count", "material")), str(scoring.get("count")))
        lines.append(
            f"When the clock runs out the game is decided on {what}, rather than drawn. "
            f"More wins; equal is still a draw."
        )
    return lines


# The order the engine asks the stated endings in, from `game.rs`. Named by the rule key
# a spec writes, so a reader can line the sentence up with what they wrote.
#
# Everything above `repetition` is asked before a move is generated; `checkmate` and
# `stalemate` need the move list to be empty. `win` covers `royal_capture` and
# `annihilation`, which are the two spellings asked at that step -- `checkmate` is the
# third and it is not asked there.
#
# **Checkmate comes before the clock** (Q412, `9e4f098`, the user: *"mate wins"*): on the
# ply a clock runs out, a side to move in check is asked for its moves and a mate stands,
# so a scored clock never overrides one. A stalemate on that ply is still the clock's,
# which is why `stalemate` stays after `scoring`. `tests/test_ending_precedence.py` pins
# the engine's side of this.
_ENDING_ORDER: tuple[tuple[str, str], ...] = (
    ("win", "a royal piece taken or a whole army gone"),
    ("reach", "arrival on a winning square"),
    ("extinction", "running out of a named piece"),
    ("bare_royal", "being reduced to royalty alone"),
    ("territory", "holding enough ground"),
    ("check_limit", "a count of checks"),
    ("repetition", "repetition"),
    ("checkmate", "checkmate"),
    ("scoring", "the clock, decided on the score"),
    ("stalemate", "having no legal move"),
)


def _ending_is_declared(key: str, rules: dict[str, Any]) -> bool:
    """Whether this spec can actually end that way, from the rule keys it wrote."""
    if key == "win":
        return rules.get("win", "checkmate") in ("royal_capture", "annihilation")
    if key == "checkmate":
        return rules.get("win", "checkmate") == "checkmate"
    if key == "territory":
        territory = rules.get("territory")
        return isinstance(territory, dict) and int(territory.get("win_at", 0) or 0) > 0
    if key == "stalemate":
        # A draw is the default and is already stated one line up as "a player with no
        # legal move draws". Only a stalemate that *decides* the game is a way to win.
        return rules.get("stalemate", "draw") != "draw"
    if key == "repetition":
        # Always available, and always a draw, so it is never one of the ways to win --
        # it is listed in the order because it outranks the clock and a reader comparing
        # this sentence with the engine would otherwise find a step missing.
        return False
    return bool(rules.get(key))


def _describe_ending_order(rules: dict[str, Any]) -> list[str]:
    """Which ending is checked first, for a spec that states more than one.

    **A player told two ways to win and not which is checked first has not been told the
    rules.** It is decidable -- a game with `reach` and `extinction` can have both true
    on one position, and the answer is fixed -- so leaving it out is a gap and not a
    simplification. Silent for the great majority of specs, which state one ending and
    have nothing to disambiguate; chess is one of them.

    Read off `_ENDING_ORDER`, which is read off `game.rs`. Making the order a *declared
    list in the spec* is the right end state and is a schema bump; this only makes the
    current answer knowable.
    """
    stated = [
        (key, prose) for key, prose in _ENDING_ORDER if _ending_is_declared(key, rules)
    ]
    # Schema 18 (Q786): leaving is read in arrival's slot of the order (`game.rs`'s
    # `schema_two_ending`), so only its words change.
    reach = rules.get("reach")
    if isinstance(reach, dict) and reach.get("on") == "leave":
        stated = [
            (key, "a move off a winning square" if key == "reach" else prose)
            for key, prose in stated
        ]
    if len(stated) < 2:
        return []
    return [
        "When more than one of these is true at once, they are checked in the order "
        + ", then ".join(prose for _, prose in stated)
        + ", and the first that is true decides."
    ]


def _describe_win(win: str) -> str:
    return {
        "checkmate": "Win by trapping the enemy royal piece so it cannot escape capture.",
        "royal_capture": "Win by capturing the enemy royal piece.",
        "annihilation": "Win by leaving the opponent with no pieces.",
        "none": "The game is won only by the endings stated below.",
    }.get(win, f"Win by {win}.")


def _squares(squares: Any, limit: int = _SQUARE_LIMIT) -> str:
    """A square list, summarised once it stops being readable.

    A palace is nine squares and worth spelling out. A shogi drop region is most of the
    board, and "a2, b2, c2, ... (48 squares)" is the honest rendering of it -- listing
    all forty-eight tells a reader nothing they could not see by looking.
    """
    listed = list(squares or [])
    if not listed:
        return "nowhere"
    if len(listed) <= limit:
        return ", ".join(str(square) for square in listed)
    shown = ", ".join(str(square) for square in listed[:limit])
    return f"{shown}, ... ({len(listed)} squares)"
